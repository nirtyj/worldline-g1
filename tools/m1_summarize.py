"""Summarise M1 drive-test runs (metrics.json + e2e.json) as a markdown table, and write runs_summary.json.

    python -m tools.m1_summarize /work/worldline-g1/outputs/m1/run-A /work/worldline-g1/outputs/m1/run-B ... [--out F]

One row per run: E1 (m1_up / m1_down exit + seconds, leftovers), E2 (60 s stand pelvis band), E3 (walk, turn, strafe,
stop, go_to legs), E4 (checks), RTF / rates. Every number comes from the run's own files; nothing is recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def row(run: str) -> dict:
    m = _load(os.path.join(run, "metrics.json")) or {}
    e = _load(os.path.join(run, "e2e.json")) or {}
    t = m.get("tests") or {}
    g = t.get("E3_goto") or {}
    legs = g.get("legs") or []
    e4 = m.get("e4") or {}
    r = {
        "run": os.path.basename(run.rstrip("/")),
        "all_pass": m.get("all_pass"),
        "summary": m.get("summary"),
        "m1_up_s": (e.get("m1_up") or {}).get("seconds"), "m1_up_exit": (e.get("m1_up") or {}).get("exit"),
        "m1_down_s": (e.get("m1_down") or {}).get("seconds"), "m1_down_exit": (e.get("m1_down") or {}).get("exit"),
        "leftover": (e.get("m1_down") or {}).get("leftover_pids"), "ports_bound": (e.get("m1_down") or {}).get("ports_still_bound"),
        "stand_z": [(t.get("E2_stand") or {}).get("pelvis_z_min"), (t.get("E2_stand") or {}).get("pelvis_z_max")],
        "stand_s": (t.get("E2_stand") or {}).get("hold_s"),
        "turn_err_deg": (t.get("E3_turn90") or {}).get("yaw_err_deg"),
        "walk_fwd_m": (t.get("E3_walk_forward") or {}).get("forward_m"),
        "walk_lat_m": (t.get("E3_walk_forward") or {}).get("lateral_m"),
        "strafe_lat_m": (t.get("E3_strafe") or {}).get("lateral_m"),
        "stop_s": (t.get("E3_stop") or {}).get("stop_time_gt_s"),
        "goto_pos_err_m": [l.get("pos_err_gt") for l in legs],
        "goto_yaw_err_deg": [l.get("yaw_err_gt_deg") for l in legs],
        "goto_rooms": g.get("rooms_visited"),
        "goto_path_m": [l.get("path_len_m") for l in legs],
        "goto_s": [l.get("duration_s") for l in legs],
        "falls": len(m.get("falls") or []), "fell_any": m.get("fell_any"),
        "e4_pass": e4.get("pass"), "e4_failed": [k for k, v in (e4.get("checks") or {}).items() if not v],
        "rtf_total": (m.get("rtf") or {}).get("p1_rtf_total"),
        "rtf_1s_min": (m.get("rtf") or {}).get("p1_rtf_1s_min"),
        "rtf_below_095": (m.get("rtf") or {}).get("p1_rtf_1s_below_0p95_frac"),
        "physics_hz": (m.get("rates") or {}).get("physics_hz_p1"),
        "gt_pose_hz": (m.get("rates") or {}).get("gt_pose_hz"),
        "camera_hz": (m.get("rates") or {}).get("camera_hz"),
        "g1_debug_hz": (m.get("rates") or {}).get("g1_debug_hz"),
        "planner_hz": (m.get("rates") or {}).get("planner_keepalive_hz"),
        "lowcmd_fresh_hz": (m.get("rates") or {}).get("lowcmd_fresh_hz_p1"),
        "leg_change_hz": e4.get("lowcmd_leg_change_hz_p1"),
        "tp_frames": (m.get("rates") or {}).get("third_person_frames"),
    }
    return r


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list):
        return "/".join(fmt(x) for x in v)
    return "-" if v is None else str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = [row(r) for r in a.runs]
    cols = ["run", "all_pass", "m1_up_s", "m1_down_s", "stand_z", "turn_err_deg", "walk_fwd_m", "walk_lat_m",
            "strafe_lat_m", "stop_s", "goto_pos_err_m", "goto_yaw_err_deg", "falls", "e4_pass", "rtf_total",
            "rtf_1s_min", "rtf_below_095"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
