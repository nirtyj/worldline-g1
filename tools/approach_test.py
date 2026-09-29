"""B.6 live exit test (docs/M2.md §7.2, contract §3.13): short strafing repositions with op `approach`.

    .venv/bin/python -m tools.approach_test [--n 30] [--out outputs/body_wave/approach-<ts>] [--port-offset N]
    .venv/bin/python -m tools.approach_test --char       # SONIC pulse characterization only (velocity op pulses)

Each trial picks a goal 0.1-0.4 m from the robot's current pose in a random direction (forward, back, sideways,
diagonal; body frame), inside a disk of --radius around the start pose, with a yaw change of 0 or up to
+-yaw-max deg, and runs `approach` through BodyClient. The error is judged on ground truth (gt.pose, 0.5 s after the
op ended), never on the body's own result. Exit bar: p90 error <= 5 cm / 5 deg over 30 repositions, 0 falls.
--char: SLOW_WALK pulses at 0.2 m/s (the `velocity` op, then end) of several durations in 4 body directions; reports
the settled travel and the glide after IDLE, which is what `approach` learns per op.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

from body.client import BodyClient
from body.config import port_offset_from_env, ports as _ports
from body.wire import wrap

from .halt_test import Tap, pct, stats


def gt(tap: Tap) -> dict | None:
    r = tap.last("pose")
    return None if r is None else {"t": r[0], "x": r[1], "y": r[2], "yaw": r[3], "v": math.hypot(r[4], r[5]),
                                   "pelvis_z": r[6], "fallen": r[7]}


def settle(tap: Tap, max_s: float = 3.0, v_eps: float = 0.03) -> dict:
    t0 = time.monotonic()
    t_ok = None
    while time.monotonic() - t0 < max_s:
        p = gt(tap)
        if p is not None and p["v"] < v_eps:
            t_ok = t_ok or time.monotonic()
            if time.monotonic() - t_ok >= 0.5:
                break
        else:
            t_ok = None
        time.sleep(0.02)
    return gt(tap)


def char(bc: BodyClient, tap: Tap, out: str, speeds=(0.2,), durs=(0.12, 0.25, 0.5, 1.0),
         dirs=("fwd", "back", "left", "right")) -> dict:
    """SLOW_WALK pulses (op `velocity` at speed v for `dur` s, then end -> IDLE) from a settled stand; per pulse the
    ground-truth travel along/across the pulse direction at IDLE and settled, the dead time (first 1 cm along) and the
    glide after IDLE."""
    unit = {"fwd": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0),
            "fl": (0.7071, 0.7071), "fr": (0.7071, -0.7071), "bl": (-0.7071, 0.7071), "br": (-0.7071, -0.7071)}
    rows = []
    for v in speeds:
        for dur in durs:
            for dname in dirs:              # directions alternate per duration, so the net displacement stays small
                ux, uy = unit[dname]
                p0 = settle(tap)
                c, s = math.cos(p0["yaw"]), math.sin(p0["yaw"])
                stream = f"char-{dname}-{v}-{dur}-{int(time.time() * 1000)}"
                t0 = time.monotonic()
                bc.submit("velocity", {"vx": v * ux, "vy": v * uy, "wz": 0.0, "stream": stream, "t_wall": time.time()})
                while time.monotonic() - t0 < dur:
                    time.sleep(0.02)
                    bc.request("velocity", {"vx": v * ux, "vy": v * uy, "wz": 0.0, "stream": stream,
                                            "t_wall": time.time()})
                t_idle = time.monotonic()
                bc.request("velocity", {"stream": stream, "end": True})
                p1 = settle(tap, 4.0)

                def along(r):
                    dx, dy = r[1] - p0["x"], r[2] - p0["y"]
                    bx, by = c * dx + s * dy, -s * dx + c * dy
                    return bx * ux + by * uy, -bx * uy + by * ux

                w = tap.window("pose", t0, time.monotonic())
                t_move = next((round(r[0] - t0, 3) for r in w if along(r)[0] > 0.01), None)
                a_idle = along(min(w, key=lambda r: abs(r[0] - t_idle)))[0] if w else None
                a_end, x_end = along((0, p1["x"], p1["y"]))
                rows.append({"dir": dname, "v": v, "pulse_s": dur, "travel_along": round(a_end, 4),
                             "travel_cross": round(x_end, 4), "at_idle_along": None if a_idle is None else round(a_idle, 4),
                             "glide": None if a_idle is None else round(a_end - a_idle, 4), "t_first_1cm_s": t_move,
                             "yaw_change_deg": round(math.degrees(wrap(p1["yaw"] - p0["yaw"])), 2),
                             "settle_s": round(time.monotonic() - t_idle, 2), "fell": bool(p1["fallen"])})
                print(f"[approach_test char] {rows[-1]}", flush=True)
                if p1["fallen"]:
                    break
    with open(os.path.join(out, "char.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dmin", type=float, default=0.10)
    ap.add_argument("--dmax", type=float, default=0.40)
    ap.add_argument("--radius", type=float, default=0.45, help="goals stay within this of the start pose")
    ap.add_argument("--yaw-max", type=float, default=20.0)
    ap.add_argument("--yaw-frac", type=float, default=0.5, help="share of trials with a yaw change")
    ap.add_argument("--tol", type=float, nargs=2, default=(0.05, 5.0))
    ap.add_argument("--char", action="store_true")
    ap.add_argument("--char-speeds", type=float, nargs="+", default=[0.2])
    ap.add_argument("--char-durs", type=float, nargs="+", default=[0.12, 0.25, 0.5, 1.0])
    ap.add_argument("--char-dirs", nargs="+", default=["fwd", "back", "left", "right"])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--out", default=f"outputs/body_wave/approach-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args(argv)
    random.seed(a.seed)
    off = a.port_offset if a.port_offset is not None else port_offset_from_env()
    os.makedirs(a.out, exist_ok=True)
    bc = BodyClient(port_offset=off).connect(15)
    tap = Tap(_ports(off))
    t0 = time.monotonic()
    while gt(tap) is None and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    st = bc.status()
    if not st.get("in_control") or st.get("fault") or st.get("latched"):
        print(f"[approach_test] not ready: {st.get('mode')} fault {st.get('fault')} latched {st.get('latched')}")
        return 2
    try:
        if a.char:
            char(bc, tap, a.out, a.char_speeds, a.char_durs, a.char_dirs)
            return 0
        home = settle(tap)
        trials = []
        for i in range(a.n):
            p = settle(tap)
            for _ in range(200):
                d = random.uniform(a.dmin, a.dmax)
                ang = random.uniform(-math.pi, math.pi)
                gx = p["x"] + d * math.cos(p["yaw"] + ang)
                gy = p["y"] + d * math.sin(p["yaw"] + ang)
                if math.hypot(gx - home["x"], gy - home["y"]) <= a.radius:
                    break
            dyaw = random.uniform(-a.yaw_max, a.yaw_max) if random.random() < a.yaw_frac else 0.0
            gyaw = wrap(home["yaw"] + math.radians(dyaw)) if dyaw else wrap(home["yaw"])
            t1 = time.monotonic()
            h = bc.approach(gx, gy, yaw=gyaw, tol=tuple(a.tol), timeout=70)
            dur = time.monotonic() - t1
            time.sleep(0.5)
            q = gt(tap)
            e = math.hypot(q["x"] - gx, q["y"] - gy)
            ey = math.degrees(wrap(q["yaw"] - gyaw))
            zmin = min((r[6] for r in tap.window("pose", t1, time.monotonic())), default=None)
            fell = bool(q["fallen"]) or (zmin is not None and zmin < 0.55)
            r = h.result or {}
            rec = {"i": i, "start": {k: round(p[k], 4) for k in ("x", "y", "yaw")}, "goal": [round(gx, 4), round(gy, 4)],
                   "goal_yaw_deg": round(math.degrees(gyaw), 2), "dist_m": round(d, 3),
                   "dir_body_deg": round(math.degrees(ang), 1), "dyaw_deg": round(dyaw, 1),
                   "state": h.state, "reason": h.reason, "gt_pos_err": round(e, 4), "gt_yaw_err_deg": round(ey, 2),
                   "body_pos_err": r.get("pos_err"), "body_yaw_err_deg": r.get("yaw_err_deg"),
                   "attempts": r.get("attempts"), "turns": r.get("turns"), "moves": r.get("moves"),
                   "t_stop": r.get("t_stop"), "floor": r.get("floor"), "duration_s": round(dur, 2),
                   "pelvis_z_min": None if zmin is None else round(zmin, 4), "fell": fell}
            trials.append(rec)
            print(f"[approach_test {time.strftime('%H:%M:%S')}] {i}: d {d:.3f} m dir {math.degrees(ang):+.0f} deg "
                  f"dyaw {dyaw:+.1f} -> {h.state}/{h.reason} GT err {e * 100:.1f} cm {ey:+.1f} deg, attempts "
                  f"{r.get('attempts')}, turns {r.get('turns')}, t_stop {r.get('t_stop')}, floor {r.get('floor')}, {dur:.1f} s",
                  flush=True)
            if fell:
                print("[approach_test] FALL: stop", flush=True)
                break
        pe = [t["gt_pos_err"] for t in trials]
        ye = [abs(t["gt_yaw_err_deg"]) for t in trials]
        s = {"n": len(trials), "succeeded": sum(1 for t in trials if t["state"] == "succeeded"),
             "pos_err_m": {**stats(pe), "p90": pct(pe, 90)}, "yaw_err_deg": {**stats(ye), "p90": pct(ye, 90)},
             "duration_s": {**stats([t["duration_s"] for t in trials]), "p90": pct([t["duration_s"] for t in trials], 90)},
             "attempts": stats([t["attempts"] for t in trials]), "falls": sum(1 for t in trials if t["fell"]),
             "within_tol": sum(1 for t in trials if t["gt_pos_err"] <= a.tol[0] and abs(t["gt_yaw_err_deg"]) <= a.tol[1]),
             "tol": list(a.tol)}
        s["pass"] = {"p90_pos_le_5cm": s["pos_err_m"]["p90"] is not None and s["pos_err_m"]["p90"] <= 0.05,
                     "p90_yaw_le_5deg": s["yaw_err_deg"]["p90"] is not None and s["yaw_err_deg"]["p90"] <= 5.0,
                     "zero_falls": s["falls"] == 0, "n30": len(trials) >= 30}
        s["all_pass"] = all(s["pass"].values())
        with open(os.path.join(a.out, "trials.json"), "w") as f:
            json.dump(trials, f, indent=1)
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        pose = np.array(list(tap.pose), dtype=float) if tap.pose else np.zeros((0, 8))
        np.savez_compressed(os.path.join(a.out, "raw.npz"), pose=pose)
        print(f"[approach_test] summary {json.dumps(s)}", flush=True)
        # back to the start pose (evidence stays comparable between runs)
        bc.go_to(home["x"], home["y"], yaw=home["yaw"], timeout_s=60) if math.hypot(
            gt(tap)["x"] - home["x"], gt(tap)["y"] - home["y"]) > 0.6 else None
        return 0 if s["all_pass"] else 1
    finally:
        tap.close()
        bc.close()


if __name__ == "__main__":
    raise SystemExit(main())
