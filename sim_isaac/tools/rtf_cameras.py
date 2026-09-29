"""RTF with the M2b cameras while SONIC stands and walks (docs/contracts/p1_m2b.md §13; OD1 rate policy).

Needs the whole stack up (P1 + deploy + body, e.g. scripts/m1_up.sh) with the robot standing under SONIC. For each
camera configuration it switches P1's cameras through the `camera` REP op (same process, no restart), waits, then
samples gt.pose `rtf` (1 s window, 50 Hz) while the robot stands still and while it walks a back-and-forth pattern
through BodyClient (walk 0.4 m/s, two 90 deg turns, walk back, turn back). Configurations are interleaved over
--repeats rounds so slow drifts on the box do not bias one of them.

    /work/worldline-g1/.venv/bin/python -m sim_isaac.tools.rtf_cameras --port-offset 0 --out DIR \
        [--configs none,head,head+ego,head15+ego15] [--repeats 2] [--stand-s 20] [--walk-s 40]

Config names: none | head[HZ] | head[HZ]+ego[HZ] (HZ defaults to 30). Writes rtf_cameras.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402
from body.config import ep  # noqa: E402
from body.p1_client import P1Rpc, PoseSub  # noqa: E402


def parse_config(name: str) -> dict:
    if name == "none":
        return {"head": 0.0, "ego": 0.0}
    out = {"head": 0.0, "ego": 0.0}
    for part in name.split("+"):
        m = re.fullmatch(r"(head|ego)(\d+(?:\.\d+)?)?", part)
        if not m:
            raise SystemExit(f"bad config {name!r}")
        out[m.group(1)] = float(m.group(2) or 30.0)
    return out


def apply(rpc: P1Rpc, cfg: dict) -> dict:
    reps = {}
    if cfg["head"] > 0:
        reps["head"] = rpc.call("camera", name="head", on=True, hz=cfg["head"], consumer="default")
    else:
        reps["head"] = rpc.call("camera", name="head", on=False)
    if cfg["ego"] > 0:
        reps["ego"] = rpc.call("camera", name="ego_view", on=True, hz=cfg["ego"], consumer="rtf_cameras")
    else:
        reps["ego"] = rpc.call("camera", name="ego_view", on=False)
    return reps


def restore(rpc: P1Rpc) -> dict:
    """P1 defaults: head on at 30 Hz; ego_view off with its rate back at 30 Hz (a camera's hz persists)."""
    out = apply(rpc, {"head": 30.0, "ego": 0.0})
    out["ego_hz"] = rpc.call("camera", name="ego_view", hz=30.0)
    return out


def summarize(samples: list, stats0: dict, stats1: dict) -> dict:
    """samples: [(recv_mono, t_sim, rtf_1s, fallen)]."""
    if len(samples) < 2:
        return {"n": len(samples)}
    rtf = np.array([s[2] for s in samples if s[2] is not None and math.isfinite(s[2])])
    dwall = samples[-1][0] - samples[0][0]
    dsim = samples[-1][1] - samples[0][1]

    def d(k):
        a, b = stats0.get(k), stats1.get(k)
        return None if a is None or b is None else round(b - a, 3)

    return {"n": len(samples), "wall_s": round(dwall, 2), "rtf_phase": round(dsim / dwall, 4) if dwall > 0 else None,
            "rtf1s_mean": round(float(rtf.mean()), 4), "rtf1s_p10": round(float(np.percentile(rtf, 10)), 4),
            "rtf1s_p50": round(float(np.percentile(rtf, 50)), 4), "rtf1s_min": round(float(rtf.min()), 4),
            "rtf1s_below_0p98": round(float((rtf < 0.98).mean()), 4),
            "rtf1s_below_0p95": round(float((rtf < 0.95).mean()), 4),
            "fallen": any(s[3] for s in samples),
            "render_hz": stats1.get("render_hz"), "render_ms": (stats1.get("render_ms") or {}).get("mean"),
            "render_ms_p99": (stats1.get("render_ms") or {}).get("p99"),
            "overruns": d("overruns"), "lost_s": d("lost_s"), "hitches_gt25ms": d("hitches_gt25ms"),
            "heartbeat_pubs": d("heartbeat_pubs"), "lowcmd_leg_change_hz": stats1.get("lowcmd_leg_change_hz"),
            "cameras": {k: {kk: v.get(kk) for kk in ("on", "hz", "pub_hz")}
                        for k, v in (stats1.get("cameras") or {}).items()}}


def walk_pattern(bc: BodyClient, until: float, log: list) -> None:
    """Back and forth until the wall deadline: walk 4 s, turn 2 x 90 deg, walk 4 s, turn back."""
    while time.monotonic() < until:
        for step in ("walk", "turn", "turn", "walk", "turn", "turn"):
            if time.monotonic() >= until:
                return
            if step == "walk":
                h = bc.walk(vx=0.4, duration_s=4.0)
            else:
                h = bc.turn_to(math.pi / 2, relative=True, timeout=30.0)
            log.append({"op": step, "ok": h.ok, "reason": h.reason})
            if not h.ok and h.reason and ("fall" in h.reason or "fault" in h.reason):
                return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="none,head,head+ego")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--stand-s", type=float, default=20.0)
    ap.add_argument("--walk-s", type=float, default=40.0)
    ap.add_argument("--no-home", action="store_true", help="do not go_to the scene spawn first")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    off = a.port_offset
    rpc = P1Rpc(ep(5600 + off), timeout_s=10.0)
    sub = PoseSub(ep(5601 + off))
    sub.start()
    bc = BodyClient(port_offset=off).connect(10)
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    res: dict = {"configs": configs, "repeats": a.repeats, "stand_s": a.stand_s, "walk_s": a.walk_s, "runs": []}
    samples: list = []
    lock = threading.Lock()

    def on_pose(p):
        with lock:
            samples.append((p.recv_mono, p.t_sim, p.rtf, p.fallen))
    sub.on_pose = on_pose

    if not a.no_home:
        info = rpc.call("get_scene_info")
        sp = info.get("spawn") or {}
        if sp:
            h = bc.go_to(float(sp["x"]), float(sp["y"]), yaw=float(sp.get("yaw", 0.0)), timeout_s=120)
            res["home"] = {"ok": h.ok, "reason": h.reason}
    try:
        for rnd in range(a.repeats):
            for name in configs:
                cfg = parse_config(name)
                run = {"config": name, "round": rnd, "apply": apply(rpc, cfg)}
                time.sleep(a.settle_s)
                for phase in ("stand", "walk"):
                    s0 = rpc.call("get_stats")
                    with lock:
                        samples.clear()
                    t_end = time.monotonic() + (a.stand_s if phase == "stand" else a.walk_s)
                    if phase == "stand":
                        time.sleep(a.stand_s)
                    else:
                        ops: list = []
                        walk_pattern(bc, t_end, ops)
                        run["walk_ops"] = ops
                    s1 = rpc.call("get_stats")
                    with lock:
                        run[phase] = summarize(list(samples), s0, s1)
                    print(f"[rtf_cameras] round {rnd} {name:14s} {phase:5s} "
                          f"p10 {run[phase].get('rtf1s_p10')} mean {run[phase].get('rtf1s_mean')} "
                          f"phase {run[phase].get('rtf_phase')} render_hz {run[phase].get('render_hz')} "
                          f"fallen {run[phase].get('fallen')}", flush=True)
                    if run[phase].get("fallen"):
                        raise RuntimeError("robot fell")
                res["runs"].append(run)
                (out / "rtf_cameras.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        res["restore"] = restore(rpc)
        bc.stop()
        bc.close()
        sub.stop()
    # per-config aggregate over rounds
    agg: dict = {}
    for run in res["runs"]:
        for phase in ("stand", "walk"):
            r = run.get(phase) or {}
            if "rtf1s_p10" not in r:
                continue
            a_ = agg.setdefault(run["config"], {}).setdefault(phase, {"p10": [], "mean": [], "phase": [], "min": []})
            a_["p10"].append(r["rtf1s_p10"])
            a_["mean"].append(r["rtf1s_mean"])
            a_["phase"].append(r["rtf_phase"])
            a_["min"].append(r["rtf1s_min"])
    res["summary"] = {c: {ph: {"p10_min": min(v["p10"]), "p10_mean": round(float(np.mean(v["p10"])), 4),
                               "mean": round(float(np.mean(v["mean"])), 4),
                               "rtf_phase_mean": round(float(np.mean(v["phase"])), 4), "min": min(v["min"])}
                          for ph, v in d.items()} for c, d in agg.items()}
    (out / "rtf_cameras.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("RTF_CAMERAS " + json.dumps(res["summary"]), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())
