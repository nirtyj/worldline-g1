"""Plots + metrics.json for one run_ref_loop.sh run directory.

usage: python report_ref.py RUN_DIR
Reads sim_trace.jsonl (50 Hz ground truth from sim_ref.py), sim_events.jsonl, sim_stats.json,
drive_trace.jsonl / drive_result.json (drive_ref.py), deploy_g1_debug.jsonl and deploy.log.
Writes RUN_DIR/report/{trajectory.png, timeseries.png} and RUN_DIR/metrics.json.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np


def jl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main(run: Path):
    rep = run / "report"
    rep.mkdir(exist_ok=True)
    tr = [r for r in jl(run / "sim_trace.jsonl") if "base_pos" in r]
    dtr = jl(run / "drive_trace.jsonl")
    events = [r for r in dtr if "event" in r]
    res = json.loads((run / "drive_result.json").read_text()) if (run / "drive_result.json").exists() else {}
    st = json.loads((run / "sim_stats.json").read_text()) if (run / "sim_stats.json").exists() else {}
    dlog = (run / "deploy.log").read_text(errors="replace") if (run / "deploy.log").exists() else ""

    t = np.array([r["t_wall"] for r in tr])
    ts = np.array([r.get("t_sim_total", r["t_sim"]) for r in tr])
    pos = np.array([r["base_pos"] for r in tr])
    yaw = np.unwrap(np.array([r["yaw"] for r in tr]))
    z = pos[:, 2]
    lc = np.array([r["foot_contact"]["left"] for r in tr], dtype=int)
    rc = np.array([r["foot_contact"]["right"] for r in tr], dtype=int)
    band = np.array([r["band"] for r in tr], dtype=int)
    cmdq = np.array([r.get("cmd_q_legs", [0] * 12) for r in tr])
    cnt = np.array([r.get("lowcmd_count", 0) for r in tr])
    chg = np.array([r.get("lowcmd_changes", 0) for r in tr])
    t0 = t[0] if len(t) else 0.0

    # --- walking windows from the drive events (mode != 0) ------------------------------------------
    cmd_ev = [(e["t_wall"], e) for e in events if e["event"] == "planner_cmd"]
    walk_windows = []
    cur = None
    for tw, e in cmd_ev:
        m = e.get("mode")
        if m is None:
            continue
        if m != 0 and cur is None:
            cur = tw
        elif m == 0 and cur is not None:
            walk_windows.append((cur, tw))
            cur = None
    # --- touchdown alternation during walking -------------------------------------------------------
    touch = []
    for i in range(1, len(tr)):
        for side, arr in (("L", lc), ("R", rc)):
            if arr[i] == 1 and arr[i - 1] == 0 and any(a <= t[i] <= b for a, b in walk_windows):
                touch.append((t[i], side))
    alt = sum(1 for i in range(1, len(touch)) if touch[i][1] != touch[i - 1][1])
    alt_ratio = alt / max(1, len(touch) - 1)

    # --- controlled window: band release .. command{stop} (after the stop the deploy exits, lowcmd ends and
    #     the unpowered robot collapses; that tail must not count as a fall or dilute the rates) ----------
    t_stop = next((e["t_wall"] for e in events if e["event"] == "command_stop_sent"), None)
    if t_stop is None:  # runs before the command_stop_sent event existed: last drive event
        t_stop = events[-1]["t_wall"] if events else (t[-1] if len(t) else 0.0)
    ctl = (band == 0) & (t <= t_stop)
    after_band = np.where(ctl)[0]
    sim_ev = jl(run / "sim_events.jsonl")
    t_rel = next((e["t_wall"] for e in sim_ev if e.get("event") == "band" and e.get("on") is False), None)
    falls_ctl = [e["t_wall"] for e in sim_ev if e.get("event") == "fall_reset" and t_rel is not None and t_rel <= e["t_wall"] <= t_stop]
    falls_all = [e["t_wall"] for e in sim_ev if e.get("event") == "fall_reset"]
    if len(after_band) > 10:
        i0, i1 = after_band[0], after_band[-1]
        dt = t[i1] - t[i0]
        lowcmd_hz = (cnt[i1] - cnt[i0]) / dt if dt > 0 else None
        change_hz = (chg[i1] - chg[i0]) / dt if dt > 0 else None
    else:
        lowcmd_hz = change_hz = None
    rtf = (ts[-1] - ts[0]) / (t[-1] - t[0]) if len(t) > 2 else None
    markers = {k: len(re.findall(p, dlog)) for k, p in {
        "init_done": r"Init Done", "planner_enabled": r"\[ZMQManager\] Planner enabled",
        "planner_initialized": r"Planner initialized successfully", "planner_motion_active": r"motion name is planner_motion",
        "planner_timeout": r"Planner timeout", "control_transition": r"transitioning to CONTROL",
        "emergency_stop": r"EMERGENCY STOP", "lowstate_lost": r"Lost LowState", "trt_engine_build": r"[Bb]uilding.*engine|Build.*engine",
    }.items()}
    # --- deploy control-loop latency (printed once per 50 ticks, g1_deploy_onnx_ref.cpp ~4080-4100) ------
    lat = {}
    for key, pat in {"obs_us": r"Obs: (\d+)us", "policy_us": r"Policy: (\d+)us", "obs2cmd_us": r"Obs 2 Motor Command: (\d+)us",
                     "lowstate_age_ms": r"LowState age: ([\d.]+)ms", "planner_model_us": r"Model: (\d+)us"}.items():
        vals = np.array([float(x) for x in re.findall(pat, dlog)])
        if len(vals):
            lat[key] = {"n": int(len(vals)), "p50": float(np.percentile(vals, 50)), "p90": float(np.percentile(vals, 90)),
                        "max": float(vals.max())}
    gpu = []
    if (run / "gpu.csv").exists():
        for line in (run / "gpu.csv").read_text().splitlines():
            try:
                gpu.append(float(line.split(",")[1].strip().rstrip(" %")))
            except (IndexError, ValueError):
                pass
    metrics = {
        "run": str(run), "pass": res.get("pass"), "tests": {r["test"]: r for r in res.get("tests", [])},
        "rtf_sim_over_wall": rtf, "sim_stats": st,
        "lowcmd_msg_hz_after_band": lowcmd_hz, "lowcmd_leg_target_change_hz_after_band": change_hz,  # wall-clock Hz in the controlled window
        "walk_windows_s": [(round(a - t0, 2), round(b - t0, 2)) for a, b in walk_windows],
        "touchdowns_while_walking": len(touch), "touchdown_alternation_ratio": round(alt_ratio, 3),
        "falls": len(falls_ctl),  # upstream fall-resets between band release and command{stop}
        "falls_total_incl_before_release_and_after_stop": len(falls_all),
        "controlled_window_s": (round(t[after_band[0]] - t0, 2), round(t_stop - t0, 2)) if len(after_band) else None,
        "pelvis_z_after_band": {"min": float(z[after_band].min()), "max": float(z[after_band].max())} if len(after_band) else None,
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))) if len(pos) > 1 else 0.0,
        "deploy_log_markers": markers, "deploy_loop_latency": lat,
        "gpu_util_pct": {"mean": float(np.mean(gpu)), "max": float(np.max(gpu))} if gpu else None, "planner_msgs_sent": res.get("planner_msgs_sent"), "g1_debug_msgs": res.get("g1_debug_msgs"),
    }
    json.dump(metrics, open(run / "metrics.json", "w"), indent=1, default=float)

    # --- plots ---------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=t - t0, s=3, cmap="viridis")
    for i in range(0, len(pos), 50):
        ax.arrow(pos[i, 0], pos[i, 1], 0.15 * math.cos(yaw[i]), 0.15 * math.sin(yaw[i]), head_width=0.03, color="0.4", lw=0.5)
    for tw, e in cmd_ev:
        i = int(np.searchsorted(t, tw))
        if i < len(pos):
            ax.annotate(f"m{e.get('mode')}", pos[i, :2], fontsize=7, color="crimson")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"MuJoCo reference: SONIC deploy (zmq_manager planner) - pelvis XY\npass={res.get('pass')}")
    fig.colorbar(sc, label="t [s]")
    fig.savefig(rep / "trajectory.png", dpi=120, bbox_inches="tight")

    fig, axs = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    tt = t - t0
    axs[0].plot(tt, z); axs[0].set_ylabel("pelvis z [m]"); axs[0].axhline(0.55, ls="--", c="r", lw=0.5)
    axs[0].fill_between(tt, 0, 1.1, where=band > 0, color="orange", alpha=0.15, label="elastic band on"); axs[0].legend(loc="lower right")
    axs[1].plot(tt, np.degrees(yaw)); axs[1].set_ylabel("yaw [deg]")
    v = np.array([math.hypot(*r["base_lin_vel_w"][:2]) for r in tr])
    axs[2].plot(tt, v); axs[2].set_ylabel("|v_xy| [m/s]")
    axs[3].plot(tt, lc + 0.05, label="left foot contact"); axs[3].plot(tt, rc * 0.9, label="right foot contact")
    axs[3].set_ylabel("contact"); axs[3].legend(loc="upper right")
    axs[4].plot(tt, cmdq[:, 0], label="L hip pitch target"); axs[4].plot(tt, cmdq[:, 3], label="L knee target")
    axs[4].plot(tt, cmdq[:, 6], label="R hip pitch target"); axs[4].plot(tt, cmdq[:, 9], label="R knee target")
    axs[4].set_ylabel("rt/lowcmd q [rad]"); axs[4].legend(loc="upper right", fontsize=7); axs[4].set_xlabel("wall time [s]")
    for a, b in walk_windows:
        for ax_ in axs:
            ax_.axvspan(a - t0, b - t0, color="green", alpha=0.07)
    for tw, e in cmd_ev:
        axs[1].axvline(tw - t0, c="crimson", lw=0.4)
    fig.suptitle("SONIC deploy in MuJoCo reference loop (green = planner walking command active)")
    fig.savefig(rep / "timeseries.png", dpi=110, bbox_inches="tight")
    print(json.dumps({k: metrics[k] for k in ("pass", "rtf_sim_over_wall", "lowcmd_msg_hz_after_band", "deploy_loop_latency",
                                             "lowcmd_leg_target_change_hz_after_band", "touchdown_alternation_ratio", "falls")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
