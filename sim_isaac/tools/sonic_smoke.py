"""SONIC-in-the-loop smoke test of wl-isaac (P1) with the unmodified gear_sonic_deploy.

Meant to run inside an isolated network namespace (sonic_netns_test.sh) so that DDS domain 0 and the contract ZMQ
ports do not collide with anyone else on the box. It plays a minimal body role:

  1. bind PUB tcp://*:5556 and stream planner IDLE at 50 Hz (keepalive: the deploy idles after 1 s of silence,
     zmq_manager.hpp:581-642); messages are built with gear_sonic's own builders
     (gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:23-86)
  2. wait for the deploy's "Init Done", send command{start, planner} (x3)
  3. release the band (P1 REP), stand --stand-s seconds
  4. SLOW_WALK (mode 1, localmotion_kplanner.hpp:78-106) forward in the planner frame for --walk-s seconds,
     then IDLE and settle
  5. report: pelvis height band, falls, displacement, alternating foot contacts, lowcmd leg-target change rate,
     RTF; P1 records an E4 trace (record op)
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np


def main():
    import msgpack
    import zmq

    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message

    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-log", required=True)
    ap.add_argument("--stand-s", type=float, default=60.0)
    ap.add_argument("--walk-s", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--turn-s", type=float, default=6.0)
    ap.add_argument("--out", default="/work/worldline-g1/outputs/m1/isaac/sonic_smoke/report.json")
    ap.add_argument("--init-timeout", type=float, default=240.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind("tcp://*:5556")
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 30000)
    req.connect("tcp://127.0.0.1:5600")

    def rep(op, **kw):
        req.send(json.dumps({"op": op, **kw}).encode())
        return json.loads(req.recv())

    poses: list[dict] = []
    stop = threading.Event()

    def pose_sub():
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        s.connect("tcp://127.0.0.1:5601")
        while not stop.is_set():
            if s.poll(100):
                _, body = s.recv_multipart()
                d = msgpack.unpackb(body, raw=False)
                d["_rx"] = time.time()
                poses.append(d)

    threading.Thread(target=pose_sub, daemon=True).start()

    state = {"mode": 0, "move": [0.0, 0.0, 0.0], "face": [1.0, 0.0, 0.0], "speed": -1.0}
    lock = threading.Lock()

    def keepalive():
        while not stop.is_set():
            with lock:
                m = build_planner_message(state["mode"], state["move"], state["face"], state["speed"], -1.0)
            pub.send(m)
            time.sleep(0.02)

    threading.Thread(target=keepalive, daemon=True).start()
    report = {"phases": {}}
    print("ping", rep("ping"), flush=True)

    # wait for the deploy's "Init Done"
    t0 = time.time()
    while time.time() - t0 < a.init_timeout:
        try:
            if "Init Done" in Path(a.deploy_log).read_text(errors="ignore"):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    else:
        report["error"] = "deploy did not print Init Done"
        out.write_text(json.dumps(report, indent=2))
        print("FAIL init", flush=True)
        stop.set()
        return 1
    report["init_done_s"] = round(time.time() - t0, 1)
    print("deploy Init Done after", report["init_done_s"], "s", flush=True)
    time.sleep(1.0)
    for _ in range(3):
        pub.send(build_command_message(start=True, stop=False, planner=True))
        time.sleep(0.1)
    rep("record", on=True, path=str(out.parent / "trace.npz"))
    time.sleep(3.0)   # policy takes over while the band still holds
    print("band release", rep("band", on=False, ramp_s=1.0), flush=True)
    t_rel = time.time()
    time.sleep(1.5)

    def window(t_a, t_b):
        return [p for p in poses if t_a <= p["_rx"] <= t_b]

    # stand
    t_s0 = time.time()
    time.sleep(a.stand_s)
    st = window(t_s0, time.time())
    pz = np.array([p["pelvis_z"] for p in st])
    xy = np.array([p["base_pos"][:2] for p in st])
    report["phases"]["stand"] = {
        "seconds": a.stand_s, "samples": len(st), "pelvis_z_min": float(pz.min()), "pelvis_z_max": float(pz.max()),
        "pelvis_z_mean": float(pz.mean()), "fallen_any": any(p["fallen"] for p in st),
        "xy_drift_m": float(np.linalg.norm(xy[-1] - xy[0])), "band_off": all(not p["band"] for p in st),
        "rtf_min": min(p["rtf"] for p in st if p["rtf"] is not None),
    }
    stats_stand = rep("get_stats")
    report["phases"]["stand"]["lowcmd_leg_change_hz"] = stats_stand.get("lowcmd_leg_change_hz")
    report["phases"]["stand"]["lowcmd_fresh_hz"] = stats_stand.get("lowcmd_fresh_hz")
    print("stand", report["phases"]["stand"], flush=True)

    # walk forward (planner frame +x = heading at planner init)
    p0 = poses[-1]
    with lock:
        state.update(mode=1, move=[1.0, 0.0, 0.0], face=[1.0, 0.0, 0.0], speed=a.speed)
    t_w0 = time.time()
    time.sleep(a.walk_s)
    stats_walk = rep("get_stats")
    with lock:
        state.update(mode=0, move=[0.0, 0.0, 0.0], speed=-1.0)
    t_w1 = time.time()
    time.sleep(3.0)
    p1 = poses[-1]
    wk = window(t_w0, t_w1)
    fl = np.array([p["foot_contact"]["left"] for p in wk], bool)
    fr = np.array([p["foot_contact"]["right"] for p in wk], bool)
    single_l = int(np.sum(fl & ~fr))
    single_r = int(np.sum(fr & ~fl))
    # number of left-only <-> right-only alternations
    seq = [("L" if (l and not r) else "R") for l, r in zip(fl, fr) if l != r]
    alternations = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    d = np.array(p1["base_pos"][:2]) - np.array(p0["base_pos"][:2])
    yaw0 = p0["yaw"]
    fwd = float(d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0))
    lat = float(-d[0] * math.sin(yaw0) + d[1] * math.cos(yaw0))
    stop_win = window(t_w1 + 1.5, time.time())
    report["phases"]["walk"] = {
        "seconds": a.walk_s, "speed_cmd": a.speed, "displacement_m": float(np.linalg.norm(d)),
        "forward_m": fwd, "lateral_m": lat, "yaw_change_deg": math.degrees(p1["yaw"] - yaw0),
        "fallen_any": any(p["fallen"] for p in window(t_w0, time.time())),
        "pelvis_z_min": float(min(p["pelvis_z"] for p in window(t_w0, time.time()))),
        "single_support_samples": {"left": single_l, "right": single_r}, "foot_alternations": alternations,
        "lowcmd_leg_change_hz": stats_walk.get("lowcmd_leg_change_hz"),
        "lowcmd_fresh_hz": stats_walk.get("lowcmd_fresh_hz"),
        "speed_after_stop_mps": float(np.mean([np.linalg.norm(p["base_lin_vel_w"][:2]) for p in stop_win]))
        if stop_win else None,
        "rtf_min": min(p["rtf"] for p in wk if p["rtf"] is not None),
    }
    print("walk", report["phases"]["walk"], flush=True)

    # turn in place: IDLE with a new facing (as keyboard Q/E do); planner-frame +y = 90 deg left of the heading
    # captured at planner init
    pt0 = poses[-1]
    with lock:
        state.update(mode=0, move=[0.0, 0.0, 0.0], face=[0.0, 1.0, 0.0], speed=-1.0)
    t_t0 = time.time()
    time.sleep(a.turn_s)
    pt1 = poses[-1]
    dyaw = math.degrees(math.atan2(math.sin(pt1["yaw"] - pt0["yaw"]), math.cos(pt1["yaw"] - pt0["yaw"])))
    report["phases"]["turn"] = {
        "seconds": a.turn_s, "facing_cmd_planner": [0.0, 1.0, 0.0], "yaw_change_deg": dyaw,
        "yaw_start_deg": math.degrees(pt0["yaw"]), "yaw_end_deg": math.degrees(pt1["yaw"]),
        "displacement_m": float(np.linalg.norm(np.array(pt1["base_pos"][:2]) - np.array(pt0["base_pos"][:2]))),
        "fallen_any": any(p["fallen"] for p in window(t_t0, time.time())),
    }
    print("turn", report["phases"]["turn"], flush=True)
    report["record"] = rep("record", on=False)
    report["stats"] = rep("get_stats")
    report["release_wall"] = t_rel
    s, w = report["phases"]["stand"], report["phases"]["walk"]
    report["checks"] = {
        "stand_no_fall": not s["fallen_any"] and s["band_off"] and s["pelvis_z_min"] > 0.55,
        "walked_forward_gt_1m": w["forward_m"] > 1.0 and not w["fallen_any"],
        "alternating_feet": w["foot_alternations"] >= 4,
        "leg_targets_change_at_policy_rate": (w["lowcmd_leg_change_hz"] or 0) > 40,
        "turned_about_90deg": 60.0 <= abs(report["phases"]["turn"]["yaw_change_deg"]) <= 120.0
        and not report["phases"]["turn"]["fallen_any"],
    }
    report["pass"] = all(report["checks"].values())
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("SONIC_SMOKE " + json.dumps({"pass": report["pass"], "checks": report["checks"]}), flush=True)
    stop.set()
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    import os
    import sys
    c = main()
    sys.stdout.flush()
    os._exit(c)
