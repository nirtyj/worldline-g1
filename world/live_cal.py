"""R.7 live calibration on SONIC (box, M1 stack up, under the stack lock): what the prompt's walking and
manipulation numbers really are, and a live check of the arm envelope that world/workspace_cal.py computed.

    python -m world.live_cal walk --house procthor-train-40 --runs 3 --out DIR     walk / stop / turn / go_to / approach
    python -m world.live_cal arm  --house procthor-train-40 --out DIR              palm check + arm_script phase times
    python -m world.live_cal summarize DIR [DIR ...]                               the averages that go into config

Driven only through the body (BodyClient, docs/contracts/m1.md §3) with SONIC in the loop; measured on ground truth
(P1 gt.pose, 5601). Every op's result, the GT trace and a timing gate go to DIR:

  walk    per run: `walk` at vx 0.30 / 0.45 / 0.60 m/s for 5 s along a free line (steady GT speed, start latency, stop
          time and glide after the command ends), a `stop` 3 s into a 0.45 m/s walk, `turn_to` +-90 / 180 deg
          (duration, deg/s, final error), `go_to` legs between the house's keypoints at navigate's speed (the effective
          speed = A* path length / op duration, arrival error), and `approach` repositions at a counter stand towards
          a reach stance `stance_clearance_m` from the counter (config/g1.yaml) and back.
  arm     `arm_script grasp` to grasp points on the IK envelope (at the fitted sphere's edge and inside it) in free
          air in front of the robot: the body's own palm error (FK of the measured arm on the GT pelvis vs the goal,
          `palm_err_w_m`), IK error, move and settle time; then the full pick/place phase sequence (pregrasp, grasp,
          lift, carry, lower, release, retract) above a counter from a reach stance, timed per phase.

Timing gate (docs/walk_diagnosis.md rules 1 and 3, PLAN §0.10 d): RTF over each motion from gt.pose (t_sim over
receive time, 1 s windows, p10 >= 0.98), P1 heartbeat_pubs during the run (must not grow), falls (GT), and the GPU
use of processes other than P1 and the deploy (sampled by the caller's shell into gpu.csv when given). The bridge's
`irregular` leg-target share is not exposed by P1 (tools/groot_timing_gate.py says the same); it is not claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from body.client import BodyClient
from body.config import ep, ports as _ports
from body.p1_client import P1Rpc, PoseSub
from body.wire import wrap

SPEEDS = (0.30, 0.45, 0.60)
GOTO_SPEED = 0.45                        # services/navigation.py: go_to(speed=walking.cruise_mps)


# ====================================================================== recording
class Rec:
    """gt.pose trace: (recv_mono, t_sim, x, y, yaw, vx, vy, wz, pelvis_z, fallen)."""

    def __init__(self, endpoint: str):
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.sub = PoseSub(endpoint, on_pose=self._on)
        self.sub.start()

    def _on(self, p) -> None:
        with self.lock:
            self.rows.append((p.recv_mono, p.t_sim, p.x, p.y, p.yaw, p.vx, p.vy, p.wz, p.pelvis_z, float(p.fallen)))

    def arr(self, t0: float | None = None, t1: float | None = None) -> np.ndarray:
        with self.lock:
            a = np.asarray(self.rows, float) if self.rows else np.zeros((0, 10))
        if len(a) and t0 is not None:
            a = a[a[:, 0] >= t0]
        if len(a) and t1 is not None:
            a = a[a[:, 0] <= t1]
        return a

    def latest(self) -> tuple | None:
        with self.lock:
            return self.rows[-1] if self.rows else None

    def wait(self, timeout: float = 10.0) -> tuple:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = self.latest()
            if r is not None:
                return r
            time.sleep(0.05)
        raise RuntimeError("no gt.pose")


def rtf_windows(a: np.ndarray, win: float = 1.0) -> dict:
    """Sim seconds per receive second over consecutive `win` windows (the timing gate's RTF)."""
    if len(a) < 10:
        return {"n": 0}
    t, s = a[:, 0], a[:, 1]
    out = []
    k0 = 0
    for k in range(1, len(a)):
        if t[k] - t[k0] >= win:
            out.append((s[k] - s[k0]) / (t[k] - t[k0]))
            k0 = k
    tot = (s[-1] - s[0]) / max(1e-6, t[-1] - t[0])
    if not out:
        return {"n": 0, "total": round(float(tot), 4)}
    v = np.asarray(out)
    return {"n": int(v.size), "p10": round(float(np.percentile(v, 10)), 4), "min": round(float(v.min()), 4),
            "total": round(float(tot), 4)}


def speed_series(a: np.ndarray) -> np.ndarray:
    return np.hypot(a[:, 5], a[:, 6]) if len(a) else np.zeros(0)


def _stats(v) -> dict:
    v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4)}


# ====================================================================== the live session
class Live:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.P = _ports(a.port_offset)
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0)
        self.rec = Rec(ep(self.P["p1_pose"]))
        self.bc = BodyClient(port_offset=a.port_offset).connect(20)
        self.rec.wait()
        self.ops: list[dict] = []
        from world.lite_world import LiteWorld
        from robot.profile import load_profile
        from world.mapgen import MapParams
        self.prof = load_profile("sonic")
        self.world = LiteWorld(a.house, map_params=MapParams.from_dict(self.prof.g1.get("mapgen")))
        self.map = self.world.static_map()
        st = self.bc.status()
        if st.get("fault"):
            raise RuntimeError(f"body fault {st['fault']}: recover or reset first")
        if not st.get("in_control"):
            h = self.bc.stand(timeout=120)
            if not h.ok:
                raise RuntimeError(f"stand failed: {h.result}")
        # fence fields like the runtime's (m1.md §3.11): a control epoch above every halt/resume the body has seen
        f = (st.get("fences") or {})
        epoch = max([0] + [int(v) for v in (f.get("halt_epoch"), f.get("epoch_seen"), f.get("resume_epoch"))
                           if isinstance(v, int)]) + 1
        gen = max(1, int(f.get("generation_floor") or 1))
        self.bc.set_fence(control_epoch=epoch, generation=gen)
        self.start = {"status": {k: st.get(k) for k in ("mode", "in_control", "fault", "fences", "halt")},
                      "p1_stats": self.p1.try_call("get_stats"), "load": os.getloadavg(),
                      "fence": {"control_epoch": epoch, "generation": gen}}

    # ------------------------------------------------------------------ helpers
    def pose(self) -> tuple:
        r = self.rec.latest()
        return r[2], r[3], r[4]

    def heartbeats(self) -> int | None:
        st = self.p1.try_call("get_stats") or {}
        v = st.get("heartbeat_pubs")
        return int(v) if isinstance(v, (int, float)) else None

    def run_op(self, kind: str, fn, **meta) -> dict:
        hb0 = self.heartbeats()
        t0 = time.monotonic()
        p0 = self.rec.latest()
        h = fn()
        t1 = time.monotonic()
        hb1 = self.heartbeats()
        a = self.rec.arr(t0, t1 + 0.05)
        rtf = rtf_windows(a)
        hb = None if hb0 is None or hb1 is None else hb1 - hb0
        rec = {"kind": kind, "op": h.op, "args": h.args, "state": h.state, "result": h.result or {},
               "t0": t0, "t1": t1, "duration_s": round(t1 - t0, 3), "rtf": rtf, "heartbeats": hb,
               # walk_diagnosis rule 3 per op: heartbeats during the motion or RTF p10 < 0.98 -> INVALID
               "valid": bool(not hb and (rtf.get("p10") is None or rtf["p10"] >= 0.98)),
               "fell": bool(len(a) and a[:, 9].max() > 0), "pose0": list(p0[2:5]) if p0 else None,
               "pose1": list(self.rec.latest()[2:5]), **meta}
        self.ops.append(rec)
        print(f"[live_cal] {kind:10s} {h.op:10s} {h.state:9s} {rec['duration_s']:6.2f} s  {json.dumps(meta)[:120]}",
              flush=True)
        return rec

    def free_ray(self, x: float, y: float, need: float, clear: float = 0.40) -> tuple[float, float]:
        """The heading with the longest straight line of `clear` clearance from (x, y), and its length."""
        g = self.map.grid
        best = (0.0, -1.0)
        for k in range(72):
            yaw = -math.pi + k * math.pi / 36
            d = 0.0
            while d < need + 1.0:
                if g.clearance(x + (d + 0.05) * math.cos(yaw), y + (d + 0.05) * math.sin(yaw)) < clear:
                    break
                d += 0.05
            if d > best[1]:
                best = (yaw, d)
        return best

    def save(self, name: str, obj: Any) -> None:
        (self.out / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")

    def finish(self, extra: dict) -> dict:
        end = {"p1_stats": self.p1.try_call("get_stats"), "load": os.getloadavg()}
        hb0 = (self.start.get("p1_stats") or {}).get("heartbeat_pubs")
        hb1 = (end.get("p1_stats") or {}).get("heartbeat_pubs")
        a = self.rec.arr()
        gate = {"rtf_all_motion": rtf_windows(a), "heartbeat_pubs_delta": None if hb0 is None or hb1 is None
                else int(hb1) - int(hb0), "falls": int(sum(o["fell"] for o in self.ops)),
                "rtf_p10_per_op_min": min((o["rtf"].get("p10", 9) for o in self.ops if o["rtf"].get("n")),
                                          default=None)}
        gate["ops"] = len(self.ops)
        gate["ops_invalid"] = [f"{i}:{o['kind']}(hb {o['heartbeats']}, rtf p10 {o['rtf'].get('p10')})"
                               for i, o in enumerate(self.ops) if not o["valid"]]
        gate["heartbeats_in_ops"] = sum(o["heartbeats"] or 0 for o in self.ops)
        gate["valid"] = bool((gate["rtf_all_motion"].get("p10") or 0) >= 0.98 and gate["falls"] == 0
                             and not gate["heartbeats_in_ops"])
        np.savez_compressed(self.out / "gt_pose.npz", rows=a)
        rep = {"start": self.start, "end": end, "gate": gate, "ops": self.ops, **extra}
        self.save("live_cal.json", rep)
        print("[live_cal] gate " + json.dumps(gate), flush=True)
        return rep


# ====================================================================== walk
def walk_measure(L: Live, rec: dict, vx: float, dur: float) -> dict:
    a = L.rec.arr(rec["t0"], rec["t1"] + 0.05)
    sp = speed_series(a)
    t = a[:, 0] - rec["t0"]
    moving = np.nonzero(sp > 0.05)[0]
    lat = float(t[moving[0]]) if len(moving) else None
    steady = (t >= 1.5) & (t <= dur)
    ss = float(np.median(sp[steady])) if steady.any() else None
    # the GT displacement over the steady window, divided by its time
    idx = np.nonzero(steady)[0]
    disp = None
    if len(idx) > 5:
        i0, i1 = idx[0], idx[-1]
        disp = math.hypot(a[i1, 2] - a[i0, 2], a[i1, 3] - a[i0, 3]) / max(1e-3, t[i1] - t[i0])
    after = t > dur
    still = np.nonzero(after & (sp < 0.05))[0]
    stop_t = float(t[still[0]] - dur) if len(still) else None
    k_end = int(np.searchsorted(t, dur))
    glide = math.hypot(a[-1, 2] - a[min(k_end, len(a) - 1), 2], a[-1, 3] - a[min(k_end, len(a) - 1), 3]) \
        if len(a) else None
    return {"cmd_mps": vx, "steady_speed_mps": None if ss is None else round(ss, 4),
            "steady_disp_speed_mps": None if disp is None else round(disp, 4),
            "start_latency_s": None if lat is None else round(lat, 3),
            "stop_after_cmd_s": None if stop_t is None else round(stop_t, 3),
            "glide_after_cmd_m": None if glide is None else round(glide, 3),
            "body_stop_time_s": (rec["result"] or {}).get("stop_time_s"),
            "walked_m": (rec["result"] or {}).get("walked_m")}


def ensure_room(L: Live, need: float) -> float:
    x, y, yaw = L.pose()
    hd, d = L.free_ray(x, y, need)
    if abs(wrap(hd - yaw)) > math.radians(8):
        L.run_op("face", lambda: L.bc.turn_to(hd, timeout=40), target_deg=round(math.degrees(hd), 1),
                 free_m=round(d, 2))
    return d


def turn_measure(L: Live, rec: dict, target: float) -> dict:
    a = L.rec.arr(rec["t0"], rec["t1"])
    y0 = a[0, 4] if len(a) else None
    dyaw = abs(math.degrees(wrap(target - y0))) if y0 is not None else None
    err = math.degrees(wrap(L.pose()[2] - target))
    return {"turn_deg": None if dyaw is None else round(dyaw, 1), "final_err_deg": round(err, 2),
            "deg_per_s": None if not dyaw else round(dyaw / max(1e-3, rec["duration_s"]), 2)}


def walk_run(L: Live, k: int) -> dict:
    out: dict[str, list] = {"walk": [], "stop": [], "turn": [], "go_to": [], "approach": []}
    dur = L.a.walk_s
    # 1) walks at three speeds along free lines, turning around between them (the turns are measured too)
    for vx in SPEEDS:
        need = vx * dur + 0.8
        room = ensure_room(L, need)
        if room < need:
            print(f"[live_cal] only {room:.2f} m of room for {need:.2f} m; walking anyway", flush=True)
        rec = L.run_op("walk", lambda vx=vx: L.bc.walk(vx=vx, duration_s=dur, timeout=dur + 30), run=k, vx=vx)
        out["walk"].append({**walk_measure(L, rec, vx, dur), "state": rec["state"], "fell": rec["fell"],
                            "rtf": rec["rtf"], "valid": rec["valid"]})
    # 2) a stop 3 s into a 0.45 m/s walk
    ensure_room(L, 0.45 * 3.5 + 0.8)
    h = L.bc.walk(vx=0.45, duration_s=8.0, wait=False)
    t_walk = time.monotonic()
    time.sleep(3.0)
    rec = L.run_op("stop", lambda: L.bc.stop(timeout=10), run=k)
    h.wait(10)
    a = L.rec.arr(rec["t0"], rec["t1"] + 3.0)
    sp = speed_series(a)
    t = a[:, 0] - rec["t0"]
    still = np.nonzero(sp < 0.05)[0]
    out["stop"].append({"body_stop_time_s": (rec["result"] or {}).get("stop_time_s"),
                        "gt_to_rest_s": round(float(t[still[0]]), 3) if len(still) else None,
                        "glide_m": round(math.hypot(a[-1, 2] - a[0, 2], a[-1, 3] - a[0, 3]), 3) if len(a) else None,
                        "speed_at_stop_mps": round(float(sp[0]), 3) if len(sp) else None, "state": rec["state"],
                        "walk_state": h.state, "t_walk_to_stop_s": round(rec["t0"] - t_walk, 2), "valid": rec["valid"]})
    # 3) turns in place: +90, -90, 180 (relative to the commanded facing)
    for dd in (90.0, -90.0, 180.0):
        x, y, yaw = L.pose()
        tgt = wrap(yaw + math.radians(dd))
        rec = L.run_op("turn", lambda tgt=tgt: L.bc.turn_to(tgt, timeout=60), run=k, delta_deg=dd)
        out["turn"].append({**turn_measure(L, rec, tgt), "cmd_deg": dd, "state": rec["state"], "valid": rec["valid"]})
    # 4) go_to legs between keypoints at navigate's speed (A*, NAV_BACKEND=astar)
    legs = [s for s in L.a.legs.split(",") if s]
    for name in legs:
        kp = L.map.keypoints[name]
        x, y, _ = L.pose()
        rec = L.run_op("go_to", lambda kp=kp: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, timeout_s=180,
                                                         backend="astar"), run=k, to=name)
        r = rec["result"] or {}
        px, py, pyaw = L.pose()
        path = r.get("path_len_m")
        out["go_to"].append({"to": name, "state": rec["state"], "duration_s": rec["duration_s"],
                             "path_len_m": path, "straight_m": round(math.hypot(kp.x - x, kp.y - y), 3),
                             "walked_m": r.get("walked_m"),
                             "eff_speed_mps": round(path / rec["duration_s"], 4) if path else None,
                             "s_per_m": round(rec["duration_s"] / path, 3) if path else None,
                             "gt_pos_err_m": round(math.hypot(px - kp.x, py - kp.y), 3),
                             "gt_yaw_err_deg": round(math.degrees(wrap(pyaw - kp.yaw)), 2),
                             "replans": r.get("replans"), "fell": rec["fell"], "rtf": rec["rtf"], "valid": rec["valid"]})
    # 5) approach repositions at the counter stand: towards a reach stance stance_clearance_m from the counter,
    #    sideways along it, and back to the stand
    kp = L.map.keypoints[L.a.counter]
    L.run_op("go_to", lambda: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, timeout_s=180, backend="astar"),
             run=k, to=L.a.counter, setup=True)
    ws = L.prof.g1.get("workspace") or {}
    clear = float(ws.get("stance_clearance_m", 0.20))
    g = L.map.grid
    c, s = math.cos(kp.yaw), math.sin(kp.yaw)
    fwd = 0.0
    while fwd < 0.5 and g.clearance(kp.x + (fwd + 0.01) * c, kp.y + (fwd + 0.01) * s) >= clear:
        fwd += 0.01
    goals = [("to_counter", fwd, 0.0), ("sideways", fwd, -0.25), ("back_to_stand", 0.0, 0.0)]
    for label, f, l in goals:
        gx, gy = kp.x + c * f - s * l, kp.y + s * f + c * l
        rec = L.run_op("approach", lambda gx=gx, gy=gy: L.bc.approach(gx, gy, kp.yaw, timeout=60), run=k,
                       leg=label, fwd=round(f, 3), lat=l, goal_clearance_m=round(g.clearance(gx, gy), 3))
        r = rec["result"] or {}
        px, py, pyaw = L.pose()
        out["approach"].append({"leg": label, "state": rec["state"], "reason": r.get("reason") or r.get("error"),
                                "duration_s": rec["duration_s"],
                                "dist_m": round(math.hypot(gx - (rec["pose0"] or [gx])[0],
                                                           gy - (rec["pose0"] or [0, gy])[1]), 3),
                                "gt_pos_err_m": round(math.hypot(px - gx, py - gy), 3),
                                "gt_clearance_m": round(g.clearance(px, py), 3), "attempts": r.get("attempts"),
                                "fell": rec["fell"], "valid": rec["valid"]})
    return out


def cmd_walk(a) -> int:
    L = Live(a)
    runs = []
    try:
        for k in range(a.runs):
            if a.start_at:
                kp = L.map.keypoints[a.start_at]
                L.run_op("go_to", lambda kp=kp: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, backend="astar"),
                         to=a.start_at, setup=True)
            runs.append(walk_run(L, k))
            L.save("runs.json", runs)
    finally:
        L.finish({"runs": runs, "house": a.house})
    return 0


# ====================================================================== arm
def arm_targets(ws: dict, heights, lats, margins) -> list[tuple[float, float, float, float]]:
    """(fwd, lat, h, margin) grasp points for the right arm: on the fitted sphere's edge minus `margin` (m), capped by
    the reach window, at grasp height h above the floor."""
    out = []
    r0 = float(ws["arm_reach_m"])
    sz, sl, sf = float(ws["shoulder_z_m"]), float(ws["shoulder_lat_m"]), float(ws.get("shoulder_fwd_m", 0.0))
    hi = float(ws["reach_fwd_m"][1])
    for h in heights:
        for lat in lats:
            for m in margins:
                r = r0 + 0.02 - m                   # arm_reach_m is the IK radius minus a 2 cm margin already
                h2 = r ** 2 - (abs(lat) - sl) ** 2 - (h - sz) ** 2
                if h2 <= 0:
                    continue
                f = min(hi + 0.02, sf + math.sqrt(h2))
                if f < 0.15:
                    continue
                out.append((round(f, 3), -abs(lat), h, m))
    return out


def cmd_arm(a) -> int:
    from body.arm_script import pelvis_to_world
    L = Live(a)
    ws = L.prof.g1.get("workspace") or {}
    results: dict[str, Any] = {"free_air": [], "sequence": []}
    try:
        # A) free air: an open spot, grasp points on and inside the sphere
        if a.open_at:
            kp = L.map.keypoints[a.open_at]
            L.run_op("go_to", lambda: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, backend="astar"),
                     to=a.open_at, setup=True)
        ensure_room(L, 1.2)
        heights = [float(v) for v in a.heights.split(",")]
        lats = [float(v) for v in a.lats.split(",")]
        margins = [float(v) for v in a.margins.split(",")]
        for f, lat, h, m in arm_targets(ws, heights, lats, margins):
            p = L.bc.pose()
            floor = L.map.floor_z
            z_b = (floor + h) - p.z
            tgt_w = [float(v) for v in pelvis_to_world([f, lat, z_b], p)]
            rec = L.run_op("grasp", lambda tgt_w=tgt_w: L.bc.run("arm_script", {
                "phase": "grasp", "arm": "right", "target_w": tgt_w, "closure": 0.3, "hold_on_end": "target"},
                wait=True, timeout=40), fwd=f, lat=lat, h=h, margin=m)
            r = rec["result"] or {}
            results["free_air"].append({"fwd": f, "lat": lat, "h": h, "margin": m, "pelvis_z": round(p.z - floor, 4),
                                        "state": rec["state"], "reason": r.get("reason") or r.get("error"),
                                        "ik_err_m": r.get("ik_err_m"), "move_s": r.get("move_s"),
                                        "duration_s": rec["duration_s"], "palm_err_w_m": r.get("palm_err_w_m"),
                                        "palm_err_b_m": r.get("palm_err_b_m"), "pelvis_shift_m": r.get("pelvis_shift_m"),
                                        "palm_final_b": r.get("palm_final_b"), "goal_b": r.get("goal_b"),
                                        "fell": rec["fell"], "valid": rec["valid"]})
            L.save("arm_partial.json", results)
        L.run_op("retract", lambda: L.bc.run("arm_script", {"phase": "retract", "arm": "right"}, wait=True,
                                             timeout=30))
        # B) the pick/place phase sequence above a counter from a reach stance (timing; no object is attached)
        if a.counter:
            kp = L.map.keypoints[a.counter]
            L.run_op("go_to", lambda: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, backend="astar"),
                     to=a.counter, setup=True)
            clear = float(ws.get("stance_clearance_m", 0.20))
            g = L.map.grid
            c, s = math.cos(kp.yaw), math.sin(kp.yaw)
            fwd = 0.0
            while fwd < 0.5 and g.clearance(kp.x + (fwd + 0.01) * c, kp.y + (fwd + 0.01) * s) >= clear:
                fwd += 0.01
            L.run_op("approach", lambda: L.bc.approach(kp.x + c * fwd, kp.y + s * fwd, kp.yaw, timeout=60),
                     setup=True, fwd=round(fwd, 3))
            surf = L.map.surfaces[a.counter]
            top = surf.z_top
            for i, lat in enumerate((-0.18, -0.12, -0.24)[: a.sequences]):
                p = L.bc.pose()
                # a point 0.30 m ahead (the stance sweet spot), `lat` to the right, 3 cm above a 5 cm-tall object
                tgt_w = [float(v) for v in pelvis_to_world([0.30, lat, 0.0], p)]
                tgt_w[2] = top + 0.05 + float(ws.get("grasp_above_top_m", 0.03))
                seq = []
                for phase, extra in (("pregrasp", {}), ("grasp", {"closure": 0.6}), ("lift", {}), ("carry", {}),
                                     ("lower", {}), ("release", {}), ("retract", {})):
                    args = {"phase": phase, "arm": "right", **extra}
                    if phase in ("pregrasp", "grasp", "lower"):
                        args["target_w"] = tgt_w
                    rec = L.run_op(phase, lambda args=args: L.bc.run("arm_script", args, wait=True, timeout=40),
                                   seq=i, lat=lat)
                    r = rec["result"] or {}
                    seq.append({"phase": phase, "state": rec["state"], "reason": r.get("reason") or r.get("error"),
                                "duration_s": rec["duration_s"], "body_duration_s": r.get("duration_s"),
                                "move_s": r.get("move_s"), "palm_err_w_m": r.get("palm_err_w_m"),
                                "fell": rec["fell"], "valid": rec["valid"]})
                results["sequence"].append({"lat": lat, "target_w": tgt_w, "phases": seq,
                                            "t_pick_s": round(sum(x["duration_s"] for x in seq[:4]), 2),
                                            "t_place_s": round(sum(x["duration_s"] for x in seq[4:]), 2)})
                L.save("arm_partial.json", results)
            L.run_op("go_to", lambda: L.bc.go_to(kp.x, kp.y, kp.yaw, speed=GOTO_SPEED, backend="astar"),
                     to=a.counter, setup=True)
            # C) longer repositions (0.45-0.6 m, approach's max_dist 0.6): back and sideways from the stand, and
            #    back to it (the reach_stance budget approach_max_m)
            results["long_approach"] = []
            for label, f, l in (("back_side", -0.40, 0.30), ("to_stand", 0.0, 0.0), ("side_far", 0.0, -0.55),
                                ("to_stand", 0.0, 0.0)):
                gx, gy = kp.x + c * f - s * l, kp.y + s * f + c * l
                if g.clearance(gx, gy) < clear:
                    continue
                rec = L.run_op("approach", lambda gx=gx, gy=gy: L.bc.approach(gx, gy, kp.yaw, timeout=90),
                               leg=label, goal_clearance_m=round(g.clearance(gx, gy), 3))
                r = rec["result"] or {}
                px, py, _ = L.pose()
                results["long_approach"].append({"leg": label, "state": rec["state"],
                                                 "reason": r.get("reason") or r.get("error"),
                                                 "duration_s": rec["duration_s"],
                                                 "dist_m": round(math.hypot(gx - rec["pose0"][0], gy - rec["pose0"][1]), 3),
                                                 "gt_pos_err_m": round(math.hypot(px - gx, py - gy), 3),
                                                 "attempts": r.get("attempts"), "fell": rec["fell"],
                                                 "valid": rec["valid"]})
    finally:
        L.finish({"arm": results, "house": a.house, "workspace": ws})
    return 0


# ====================================================================== summarize
def summarize(dirs: list[str], valid_only: bool = True) -> dict:
    """The averages over every run in `dirs`; with valid_only, rows whose op failed the per-op timing gate
    (heartbeats during it, or RTF p10 < 0.98) are left out (rows from runs before the per-op gate have no flag and
    count as valid; their run's gate says what it knew)."""
    keep = (lambda r: r.get("valid", True)) if valid_only else (lambda r: True)
    walks: dict[float, list] = {}
    stops, turns, gotos, apps, gates = [], [], [], [], []
    arm_rows, seqs, longs = [], [], []
    for d in dirs:
        p = Path(d) / "live_cal.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        gates.append({"dir": d, **rep.get("gate", {})})
        for run in rep.get("runs") or []:
            for w in run["walk"]:
                if keep(w):
                    walks.setdefault(w["cmd_mps"], []).append(w)
            stops += [x for x in run["stop"] if keep(x)]
            turns += [x for x in run["turn"] if keep(x)]
            gotos += [x for x in run["go_to"] if keep(x)]
            apps += [x for x in run["approach"] if keep(x)]
        arm = rep.get("arm") or {}
        arm_rows += [x for x in arm.get("free_air") or [] if keep(x)]
        seqs += [q for q in arm.get("sequence") or [] if all(keep(x) for x in q["phases"])]
        longs += [x for x in arm.get("long_approach") or [] if keep(x)]
    ok_goto = [g for g in gotos if g["state"] == "succeeded" and g.get("path_len_m")]
    path = sum(g["path_len_m"] for g in ok_goto)
    tsum = sum(g["duration_s"] for g in ok_goto)
    out = {
        "gates": gates,
        "walk": {str(v): {"steady_speed_mps": _stats([w["steady_speed_mps"] for w in ws]),
                          "steady_disp_speed_mps": _stats([w["steady_disp_speed_mps"] for w in ws]),
                          "start_latency_s": _stats([w["start_latency_s"] for w in ws]),
                          "stop_after_cmd_s": _stats([w["stop_after_cmd_s"] for w in ws]),
                          "glide_after_cmd_m": _stats([w["glide_after_cmd_m"] for w in ws])}
                 for v, ws in sorted(walks.items())},
        "stop": {"gt_to_rest_s": _stats([s["gt_to_rest_s"] for s in stops]),
                 "body_stop_time_s": _stats([s["body_stop_time_s"] for s in stops]),
                 "glide_m": _stats([s["glide_m"] for s in stops])},
        "turn": {str(dd): {"duration_deg_per_s": _stats([t["deg_per_s"] for t in turns if t["cmd_deg"] == dd]),
                           "final_err_deg": _stats([abs(t["final_err_deg"]) for t in turns if t["cmd_deg"] == dd])}
                 for dd in sorted({t["cmd_deg"] for t in turns})},
        "go_to": {"legs": len(gotos), "succeeded": len(ok_goto), "path_m_total": round(path, 2),
                  "time_s_total": round(tsum, 2),
                  "effective_speed_mps": round(path / tsum, 4) if tsum else None,
                  "s_per_m": round(tsum / path, 3) if path else None,
                  "per_leg_eff_speed_mps": _stats([g["eff_speed_mps"] for g in ok_goto]),
                  "gt_pos_err_m": _stats([g["gt_pos_err_m"] for g in gotos]),
                  "gt_yaw_err_deg": _stats([abs(g["gt_yaw_err_deg"]) for g in gotos])},
        "approach": {"n": len(apps), "succeeded": sum(x["state"] == "succeeded" for x in apps),
                     "duration_s": _stats([x["duration_s"] for x in apps]),
                     "gt_pos_err_m": _stats([x["gt_pos_err_m"] for x in apps]),
                     "gt_clearance_m_at_counter": _stats([x["gt_clearance_m"] for x in apps
                                                          if x["leg"] != "back_to_stand"])},
        "long_approach": {"n": len(longs), "succeeded": sum(x["state"] == "succeeded" for x in longs),
                          "rows": [{k: x[k] for k in ("leg", "state", "dist_m", "duration_s", "gt_pos_err_m")}
                                   for x in longs]},
        "arm": {"n": len(arm_rows),
                "succeeded": sum(r["state"] == "succeeded" for r in arm_rows),
                "by_margin": {str(m): {"palm_err_w_p90_m": _stats([(r.get("palm_err_w_m") or {}).get("p90")
                                                                   for r in arm_rows if r["margin"] == m
                                                                   and r["state"] == "succeeded"]),
                                       "failed": [f"{r['fwd']},{r['lat']},{r['h']}: {r['reason']}" for r in arm_rows
                                                  if r["margin"] == m and r["state"] != "succeeded"]}
                              for m in sorted({r["margin"] for r in arm_rows})},
                "pelvis_z_m": _stats([r.get("pelvis_z") for r in arm_rows])},
        "arm_sequence": {"t_pick_s": _stats([q["t_pick_s"] for q in seqs]),
                         "t_place_s": _stats([q["t_place_s"] for q in seqs]),
                         "phase_s": {ph: _stats([x["duration_s"] for q in seqs for x in q["phases"]
                                                 if x["phase"] == ph])
                                     for ph in ("pregrasp", "grasp", "lift", "carry", "lower", "release", "retract")}},
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("walk", "arm"):
        s = sub.add_parser(name)
        s.add_argument("--port-offset", type=int, default=int(os.environ.get("WL_PORT_OFFSET", 0) or 0))
        s.add_argument("--house", default="procthor-train-40")
        s.add_argument("--out", required=True)
    w = sub.choices["walk"]
    w.add_argument("--runs", type=int, default=3)
    w.add_argument("--walk-s", type=float, default=5.0)
    w.add_argument("--legs", default="kitchen_counter_1a,living_room,bedroom_dresser_1b,start")
    w.add_argument("--counter", default="kitchen_counter_1a")
    w.add_argument("--start-at", default="start")
    m = sub.choices["arm"]
    m.add_argument("--open-at", default="living_room")
    m.add_argument("--heights", default="0.80,0.95,1.10,1.25")
    m.add_argument("--lats", default="0.0,0.15,0.30")
    m.add_argument("--margins", default="0.0,0.04")
    m.add_argument("--counter", default="kitchen_counter_1a")
    m.add_argument("--sequences", type=int, default=3)
    s = sub.add_parser("summarize")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--out", default=None)
    s.add_argument("--all", action="store_true", help="include ops that failed the per-op timing gate")
    a = ap.parse_args(argv)
    if a.cmd == "walk":
        return cmd_walk(a)
    if a.cmd == "arm":
        return cmd_arm(a)
    rep = summarize(a.dirs, valid_only=not a.all)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
