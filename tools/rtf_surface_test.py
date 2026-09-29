"""RTF at the manipulation surfaces, per head-camera rate (M2b finish, owner body-regress; PLAN §0.10 b, d).

The robot owner's scripted picks at the H40 bedroom dresser were refused `DEGRADED` (world/sim_health.py: RTF_1s
< 0.95 held 5 s rejects `manipulate`): P1 ran at RTF 0.83-0.94 there with the head camera at 30 Hz + `--viz min`
(physics step p50 2.8 ms, render p50 11.4-12 ms at 27-28 Hz). This tool finds the head rate that keeps the timing
gate's bar (RTF p10 >= 0.98 over 1 s windows, docs/walk_diagnosis.md rule 3) where `manipulate` runs, without
starving System 1 (its frame gate forwards at most 1 frame/s, `brains/frame_gate.py` G1_HEAD min_interval_s; the
page and the arrival scans want >= ~5 Hz).

For every stand (default: the runtime's R.7 surface keypoints of the house that hold in-band pickables,
`world.mapgen` through `world.lite_world`, the same stands `navigate` walks to), it walks there with `go_to` and
measures at the keypoint and at a reach stance `--reach-in` m closer (`approach`, where `manipulate` runs after a
reposition). At each position every head rate in `--head-hz` is applied live with P1's `set_render_rates` (no
restart; the order rotates per position so drifts do not favour one rate), then after `--settle-s` it samples
gt.pose `rtf` (P1's 1 s window at 50 Hz) for `--stand-s`:

    rtf1s p10 / mean / min / share < 0.98 / < 0.95, the longest run below 0.95 (>= 5 s = `manipulate` DEGRADED),
    P1 heartbeat_pubs / overruns / hitches_gt25ms / lost_s deltas, render_hz, render ms, physics_hz_1s,
    head frames/s received on the camera PUB (5565: what System 1 and the page get).

The walks between stands are recorded as informational (the rate there is the one in force before the stand).
At the end the head rate is put back to `--restore-hz` (default: the head camera's rate at the start, get_cameras).

    WL_PORT_OFFSET=0 .venv-rt/bin/python -m tools.rtf_surface_test --out outputs/m2b_finish/bregress/rtf-<ts> \\
        [--house procthor-train-40] [--head-hz 30,15,10] [--stand-s 20] [--reach-in 0.12] [--only a,b] [--no-reach]

Writes rtf_surfaces.json (per stand, position and rate + a per-rate verdict over all positions) and prints a table.
Runs against a standing stack with the stack lock held and nothing else on the GPU (walk_diagnosis rule 1).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402
from body.config import ep, port_offset_from_env  # noqa: E402
from body.p1_client import P1Rpc, PoseSub  # noqa: E402

DEGRADED_BELOW, DEGRADED_HOLD_S = 0.95, 5.0            # world/sim_health.py SimHealthConfig defaults
UNSAFE_BELOW, UNSAFE_HOLD_S = 0.85, 3.0
GATE_P10 = 0.98                                        # PLAN §0.10 b


def gpu_apps() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e!r}"]


def house_stands(house: str, profile: str = "sonic", z_min: float | None = None,
                 z_max: float | None = None) -> list[dict]:
    """The Isaac map's surface keypoints (R.7 stands) that hold at least one pickable in the arm's height band."""
    from robot.profile import load_profile
    from world.lite_world import LiteWorld
    from world.mapgen import MapParams

    prof = load_profile(profile)
    mp = MapParams.from_dict({k: v for k, v in (prof.g1.get("mapgen") or {}).items() if k != "lite_world"})
    ws = prof.g1.get("workspace") or {}
    lo = float(ws.get("obj_z_min_m", 0.70)) if z_min is None else z_min
    hi = float(ws.get("obj_z_max_m", 1.25)) if z_max is None else z_max
    w = LiteWorld(house, map_params=mp, user_surface=prof.scene_config(house).get("user_surface"))
    m = w.static_map()
    out = []
    for name in m.surfaces:
        k = m.keypoints.get(name)
        if k is None:
            continue
        objs = [oid for oid, o in w.objects().items() if o.where == name and lo <= o.pos[2] - m.floor_z <= hi]
        if objs:
            out.append({"name": name, "x": round(float(k.x), 3), "y": round(float(k.y), 3),
                        "yaw": round(float(k.yaw), 4), "objects": sorted(objs)})
    return out


def parse_stands(s: str) -> list[dict]:
    """'name:x,y,yaw;name:x,y,yaw' (yaw in radians)."""
    out = []
    for part in s.split(";"):
        if not part.strip():
            continue
        name, xyz = part.split(":")
        x, y, yaw = (float(v) for v in xyz.split(","))
        out.append({"name": name.strip(), "x": x, "y": y, "yaw": yaw, "objects": []})
    return out


def order_stands(stands: list[dict], x0: float, y0: float) -> list[dict]:
    """Greedy nearest-next from the robot (straight-line; A* decides the real path)."""
    left, out, x, y = list(stands), [], x0, y0
    while left:
        i = min(range(len(left)), key=lambda j: math.hypot(left[j]["x"] - x, left[j]["y"] - y))
        s = left.pop(i)
        out.append(s)
        x, y = s["x"], s["y"]
    return out


class FrameCounter(threading.Thread):
    """Receive times of the head camera PUB (5565, sensor_server format): what System 1 and the page get."""

    def __init__(self, endpoint: str):
        super().__init__(daemon=True, name="head-frames")
        self.endpoint, self.t, self.lock, self._run = endpoint, [], threading.Lock(), True

    def run(self) -> None:
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 10)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(self.endpoint)
        while self._run:
            if s.poll(100):
                while True:
                    try:
                        s.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    with self.lock:
                        self.t.append(time.monotonic())
        s.close(0)

    def between(self, t0: float, t1: float) -> int:
        with self.lock:
            return sum(1 for t in self.t if t0 <= t <= t1)

    def stop(self) -> None:
        self._run = False


def longest_below(samples: list, thr: float) -> float:
    """Longest wall-time run of consecutive samples with rtf < thr (s)."""
    best, t_start = 0.0, None
    for t, _ts, r, _f in samples:
        if r is not None and math.isfinite(r) and r < thr:
            t_start = t if t_start is None else t_start
            best = max(best, t - t_start)
        else:
            t_start = None
    return round(best, 2)


def summarize(samples: list, s0: dict, s1: dict) -> dict:
    if len(samples) < 10:
        return {"n": len(samples)}
    rtf = np.array([s[2] for s in samples if s[2] is not None and math.isfinite(s[2])])
    dwall = samples[-1][0] - samples[0][0]
    dsim = samples[-1][1] - samples[0][1]

    def d(k):
        a, b = s0.get(k), s1.get(k)
        return None if a is None or b is None else round(b - a, 3)

    return {"n": len(samples), "wall_s": round(dwall, 2), "rtf_phase": round(dsim / dwall, 4) if dwall > 0 else None,
            "rtf1s_p10": round(float(np.percentile(rtf, 10)), 4), "rtf1s_mean": round(float(rtf.mean()), 4),
            "rtf1s_min": round(float(rtf.min()), 4), "below_0p98": round(float((rtf < 0.98).mean()), 4),
            "below_0p95": round(float((rtf < DEGRADED_BELOW).mean()), 4),
            "longest_below_0p95_s": longest_below(samples, DEGRADED_BELOW),
            "longest_below_0p85_s": longest_below(samples, UNSAFE_BELOW),
            "fallen": any(s[3] for s in samples),
            "heartbeat_pubs": d("heartbeat_pubs"), "overruns": d("overruns"), "hitches_gt25ms": d("hitches_gt25ms"),
            "lost_s": d("lost_s"), "render_hz": s1.get("render_hz"),
            "render_ms_mean": (s1.get("render_ms") or {}).get("mean"), "physics_hz_1s": s1.get("physics_hz_1s"),
            "camera_pub_hz": s1.get("camera_pub_hz"), "gc": (s1.get("gc") or {}).get("ms_max")}


class Runner:
    def __init__(self, a):
        self.a = a
        self.off = port_offset_from_env() if a.port_offset is None else a.port_offset
        self.rpc = P1Rpc(ep(5600 + self.off), timeout_s=10.0)
        self.samples: list = []
        self.lock = threading.Lock()
        self.sub = PoseSub(ep(5601 + self.off), on_pose=self._on_pose)
        self.sub.start()
        self.frames = FrameCounter(ep(5565 + self.off))
        self.frames.start()
        self.bc = BodyClient(port_offset=self.off).connect(10)

    def _on_pose(self, p) -> None:
        with self.lock:
            self.samples.append((p.recv_mono, p.t_sim, p.rtf, p.fallen))
            if len(self.samples) > 200000:
                del self.samples[:100000]

    def window(self, t0: float, t1: float) -> list:
        with self.lock:
            return [s for s in self.samples if t0 <= s[0] <= t1]

    def pose(self):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5:
            p = self.sub.latest() if hasattr(self.sub, "latest") else None
            if p is not None:
                return p
            time.sleep(0.05)
        raise RuntimeError("no gt.pose")

    def set_head(self, hz: float) -> dict:
        return self.rpc.call("set_render_rates", head_hz=float(hz), ego_hz=None)

    def measure(self, hz: float) -> dict:
        rep = self.set_head(hz)
        time.sleep(self.a.settle_s)
        s0 = self.rpc.call("get_stats")
        t0 = time.monotonic()
        time.sleep(self.a.stand_s)
        t1 = time.monotonic()
        s1 = self.rpc.call("get_stats")
        r = summarize(self.window(t0, t1), s0, s1)
        r["head_hz"] = hz
        r["head_frames_per_s"] = round(self.frames.between(t0, t1) / max(1e-6, t1 - t0), 2)
        r["set_reply"] = {k: v for k, v in (rep or {}).items() if k in ("ok", "head", "error")}
        r["gpu_apps"] = gpu_apps()
        return r

    def close(self) -> None:
        self.frames.stop()
        self.sub.stop()
        self.bc.close()


def verdict(res: dict, rates: list[float]) -> dict:
    out = {}
    for hz in rates:
        rows = [m for st in res["stands"] for pos in st.get("positions", []) for m in pos.get("rates", [])
                if m.get("head_hz") == hz and "rtf1s_p10" in m]
        if not rows:
            continue
        worst = min(rows, key=lambda m: m["rtf1s_p10"])
        out[str(hz)] = {"positions": len(rows), "p10_min": worst["rtf1s_p10"],
                        "p10_min_at": worst.get("_where"),
                        "p10_mean": round(float(np.mean([m["rtf1s_p10"] for m in rows])), 4),
                        "mean_min": min(m["rtf1s_mean"] for m in rows),
                        "positions_p10_ge_0p98": sum(1 for m in rows if m["rtf1s_p10"] >= GATE_P10),
                        "positions_degraded_5s": sum(1 for m in rows if m["longest_below_0p95_s"] >= DEGRADED_HOLD_S),
                        "positions_unsafe_3s": sum(1 for m in rows if m.get("longest_below_0p85_s", 0) >= UNSAFE_HOLD_S),
                        "heartbeat_pubs": sum(m.get("heartbeat_pubs") or 0 for m in rows),
                        "head_frames_per_s_min": min(m["head_frames_per_s"] for m in rows),
                        "pass_all_positions": all(m["rtf1s_p10"] >= GATE_P10 for m in rows)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--house", default="procthor-train-40")
    ap.add_argument("--stands", default=None, help="name:x,y,yaw;... (default: the house's R.7 surface keypoints)")
    ap.add_argument("--only", default=None, help="comma list of stand names to keep")
    ap.add_argument("--head-hz", default="30,15,10")
    ap.add_argument("--stand-s", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--reach-in", type=float, default=0.12, help="reach stance: approach this far towards the surface")
    ap.add_argument("--no-reach", action="store_true")
    ap.add_argument("--restore-hz", type=float, default=None, help="head rate at the end (default: the rate at the start)")
    ap.add_argument("--goto-timeout", type=float, default=150.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rates = [float(x) for x in a.head_hz.split(",") if x.strip()]
    stands = parse_stands(a.stands) if a.stands else house_stands(a.house)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        stands = [s for s in stands if s["name"] in keep]
    R = Runner(a)
    s_start = R.rpc.call("get_stats")
    head0 = next((c for c in (R.rpc.call("get_cameras") or {}).get("cameras") or [] if c.get("name") == "head"), {})
    # the rate in force now (another owner may have set it live), not the --camera-hz P1 started with
    restore_hz = a.restore_hz or float(head0.get("hz") or s_start.get("camera_hz_target") or 30.0)
    res = {"tool": "rtf_surface_test", "house": a.house, "head_hz": rates, "stand_s": a.stand_s,
           "settle_s": a.settle_s, "reach_in": None if a.no_reach else a.reach_in, "restore_hz": restore_hz,
           "gpu_apps_before": gpu_apps(), "load_before": os.getloadavg(),
           "p1_start": {**{k: s_start.get(k) for k in ("camera_hz_target", "render_hz", "rtf_total", "house")},
                        "head_hz": head0.get("hz"), "head_consumers": head0.get("consumers")},
           "bars": {"gate_p10": GATE_P10, "degraded": f"rtf_1s < {DEGRADED_BELOW} held {DEGRADED_HOLD_S} s",
                    "unsafe": f"rtf_1s < {UNSAFE_BELOW} held {UNSAFE_HOLD_S} s"},
           "stands": [], "walks": []}
    try:
        st = R.bc.status()
        if st.get("fault") or not st.get("in_control"):
            raise RuntimeError(f"robot not standing under SONIC: fault {st.get('fault')} in_control "
                               f"{st.get('in_control')}")
        p = R.pose()
        stands = order_stands(stands, p.x, p.y)
        res["order"] = [s["name"] for s in stands]
        print(f"[rtf_surf] {len(stands)} stands: {res['order']} rates {rates}", flush=True)
        for i, s in enumerate(stands):
            row = dict(s)
            t0 = time.monotonic()
            s0 = R.rpc.call("get_stats")
            h = R.bc.go_to(s["x"], s["y"], yaw=s["yaw"], timeout_s=a.goto_timeout)
            t1 = time.monotonic()
            s1 = R.rpc.call("get_stats")
            walk = summarize(R.window(t0, t1), s0, s1)
            walk.update({"to": s["name"], "state": h.state, "reason": h.reason})
            res["walks"].append(walk)
            row["go_to"] = {"state": h.state, "reason": h.reason, "pos_err": (h.result or {}).get("pos_err")}
            if walk.get("fallen"):
                raise RuntimeError("robot fell on the way")
            if not h.ok:
                print(f"[rtf_surf] {s['name']}: go_to {h.state} {h.reason}; measured where it stopped", flush=True)
            positions = [("keypoint", None)]
            if not a.no_reach:
                positions.append(("reach", a.reach_in))
            row["positions"] = []
            for j, (pname, dist) in enumerate(positions):
                pos = {"position": pname}
                if dist:
                    g = R.pose()
                    tx, ty = g.x + dist * math.cos(s["yaw"]), g.y + dist * math.sin(s["yaw"])
                    ha = R.bc.approach(tx, ty, yaw=s["yaw"], tol=(0.04, 5.0), timeout=45)
                    pos["approach"] = {"state": ha.state, "reason": ha.reason,
                                       "pos_err": (ha.result or {}).get("pos_err")}
                    if not ha.ok:
                        print(f"[rtf_surf] {s['name']}: approach {ha.state} {ha.reason}; measured where it stopped",
                              flush=True)
                g = R.pose()
                pos["pose"] = [round(g.x, 3), round(g.y, 3), round(math.degrees(g.yaw), 1)]
                k = (i * len(positions) + j) % len(rates)
                pos["rates"] = []
                for hz in rates[k:] + rates[:k]:                # rotate the order per position
                    m = R.measure(hz)
                    m["_where"] = f"{s['name']}/{pname}"
                    pos["rates"].append(m)
                    print(f"[rtf_surf] {s['name']:26s} {pname:8s} head {hz:4.0f} Hz: p10 {m.get('rtf1s_p10')} "
                          f"mean {m.get('rtf1s_mean')} min {m.get('rtf1s_min')} <0.95 run "
                          f"{m.get('longest_below_0p95_s')} s hb {m.get('heartbeat_pubs')} render "
                          f"{m.get('render_ms_mean')} ms phys {m.get('physics_hz_1s')} Hz frames "
                          f"{m.get('head_frames_per_s')}/s", flush=True)
                    if m.get("fallen"):
                        raise RuntimeError("robot fell")
                row["positions"].append(pos)
            res["stands"].append(row)
            (out / "rtf_surfaces.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
        print(f"[rtf_surf] ERROR {e!r}", flush=True)
    finally:
        try:
            res["restore"] = R.set_head(restore_hz)
        except Exception as e:  # noqa: BLE001
            res["restore"] = {"error": repr(e)}
        res["gpu_apps_after"] = gpu_apps()
        res["load_after"] = os.getloadavg()
        res["verdict"] = verdict(res, rates)
        (out / "rtf_surfaces.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        R.close()
    print("RTF_SURFACES " + json.dumps(res["verdict"]), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
