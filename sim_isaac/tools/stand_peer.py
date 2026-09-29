"""Stand-alone DDS peer for testing wl-isaac without the SONIC deploy.

It plays the deploy's role on DDS: publishes rt/lowcmd at 500 Hz (the deploy's command-writer rate,
g1_deploy_onnx_ref.cpp:21) with a static stand command (default angles + the deploy/training kp/kd,
policy_parameters.hpp:143-240) and Dex3 hold commands, and checks what wl-isaac publishes.

    /work/envs/isaaclab/bin/python -m sim_isaac.tools.stand_peer --domain 7 --port-offset 100 \
        --hold-s 60 --joint-map --out /work/worldline-g1/outputs/m1/isaac/stand_test.json

Checks: lowstate/secondary_imu/dex3 state rates, lowstate joint q vs gt (Isaac by name), lowstate IMU quaternion
vs gt.pose base_quat, pelvis height band while standing without the band, falls, camera format, REP ops.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from sim_isaac import joint_map as jm


# motor position limits, $WBC/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml:220-233
MOTOR_LOWER = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
               -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
               -2.618, -0.52, -0.52, -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443,
               -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443]
MOTOR_UPPER = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
               2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
               2.618, 0.52, 0.52, 2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443,
               2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443]


def quat_angle(a, b) -> float:
    d = abs(float(np.dot(np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b))))
    return 2.0 * math.degrees(math.acos(min(1.0, d)))


class Peer:
    def __init__(self, domain: int, iface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, IMUState_, LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(domain, iface)
        self.crc = CRC()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.lh = unitree_hg_msg_dds__HandCmd_()
        self.rh = unitree_hg_msg_dds__HandCmd_()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.lpub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.lpub.Init()
        self.rpub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.rpub.Init()
        self.lock = threading.Lock()
        self.ls = None
        self.ls_times: list[float] = []
        self.ls_ticks: list[int] = []
        self.imu_times: list[float] = []
        self.imu = None
        self.hand_times: list[float] = []
        self.hand = None
        self.mode_machine = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_ls, 1)
        self.isub = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.isub.Init(self._on_imu, 1)
        self.hsub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.hsub.Init(self._on_hand, 1)
        self.q_target = np.array(jm.DEFAULT_ANGLES)
        self.kp = np.array(jm.KPS)
        self.kd = np.array(jm.KDS)
        self._stop = threading.Event()
        self.sent = 0
        self.th = threading.Thread(target=self._writer, daemon=True)

    def _on_ls(self, msg):
        now = time.perf_counter()
        with self.lock:
            self.ls = msg
            self.ls_times.append(now)
            self.ls_ticks.append(msg.tick)
            self.mode_machine = msg.mode_machine

    def _on_imu(self, msg):
        with self.lock:
            self.imu = msg
            self.imu_times.append(time.perf_counter())

    def _on_hand(self, msg):
        with self.lock:
            self.hand = msg
            self.hand_times.append(time.perf_counter())

    def start(self):
        self.th.start()

    def _writer(self):
        period = 1.0 / 500.0
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self.lock:
                q, kp, kd = self.q_target.copy(), self.kp.copy(), self.kd.copy()
                mm = self.mode_machine or 0
            self.cmd.mode_pr = 0
            self.cmd.mode_machine = mm
            for i in range(jm.NUM_MOTORS):
                m = self.cmd.motor_cmd[i]
                m.mode = 1
                m.q = float(q[i])
                m.dq = 0.0
                m.kp = float(kp[i])
                m.kd = float(kd[i])
                m.tau = 0.0
            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            for h, p in ((self.lh, self.lpub), (self.rh, self.rpub)):
                for i in range(7):
                    h.motor_cmd[i].q = 0.0
                    h.motor_cmd[i].kp = jm.DEX3_HOLD_KP
                    h.motor_cmd[i].kd = jm.DEX3_HOLD_KD
                p.Write(h)
            self.sent += 1
            nxt += period
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.perf_counter()

    def rate(self, times: list[float], window: float = 2.0) -> float:
        now = time.perf_counter()
        with self.lock:
            return sum(1 for t in times[-2000:] if t >= now - window) / window

    def stop(self):
        self._stop.set()


class Rep:
    def __init__(self, port: int):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.port = port
        self._mk()

    def _mk(self):
        self.s = self.ctx.socket(self.zmq.REQ)
        self.s.setsockopt(self.zmq.LINGER, 0)
        self.s.setsockopt(self.zmq.RCVTIMEO, 60000)
        self.s.connect(f"tcp://127.0.0.1:{self.port}")

    def __call__(self, op: str, **kw) -> dict:
        self.s.send(json.dumps({"op": op, **kw}).encode())
        try:
            return json.loads(self.s.recv())
        except self.zmq.Again:
            self.s.close(0)
            self._mk()
            return {"ok": False, "error": "timeout"}


class PoseSub:
    def __init__(self, port: int):
        import msgpack
        import zmq
        self.msgpack = msgpack
        self.s = zmq.Context.instance().socket(zmq.SUB)
        self.s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        self.s.connect(f"tcp://127.0.0.1:{port}")
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            if self.s.poll(100):
                topic, body = self.s.recv_multipart()
                d = self.msgpack.unpackb(body, raw=False)
                d["_rx"] = time.perf_counter()
                self.samples.append(d)

    def latest(self):
        return self.samples[-1] if self.samples else None

    def stop(self):
        self._stop.set()


def check_camera(port: int, out_png: str | None, timeout_s: float = 10.0) -> dict:
    import cv2
    import msgpack
    import zmq
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVHWM, 3)
    s.connect(f"tcp://127.0.0.1:{port}")
    t0 = time.perf_counter()
    times, msg = [], None
    while time.perf_counter() - t0 < timeout_s and len(times) < 60:
        if s.poll(200):
            msg = msgpack.unpackb(s.recv(), raw=False)
            times.append(time.perf_counter())
    s.close(0)
    if msg is None:
        return {"ok": False, "error": "no camera message"}
    keys = sorted(msg.keys())
    key = msg.get("camera") or next(iter(msg["images"]))     # M2b: "head" (docs/contracts/p1_m2b.md §5.2)
    b64 = msg["images"][key]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)  # sensor_server decode
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_png, img[..., ::-1])  # decoded array is RGB (MuJoCo convention); imwrite wants BGR
    hz = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None
    return {"ok": isinstance(b64, str) and img is not None and img.shape == (480, 640, 3)
            and "timestamps" in msg and msg.get(key) == b64,
            "keys": keys, "shape": list(img.shape), "rx_hz": hz, "mean_rgb": img.reshape(-1, 3).mean(0).tolist(),
            "png": out_png}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=7)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--port-offset", type=int, default=100)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--hold-s", type=float, default=60.0)
    ap.add_argument("--band-ramp-s", type=float, default=1.0)
    ap.add_argument("--joint-map", action="store_true")
    ap.add_argument("--gains", choices=["train", "stiff"], default="stiff",
                    help="train: the deploy/training kp/kd (a static PD stand with these soft ankles cannot balance: "
                         "ankle kp 28.5 Nm/rad << m*g*h ~ 240 Nm/rad); stiff: legs/waist kp 350-400 for a static "
                         "stand check")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--reset", action="store_true", help="reset_robot to the spawn pose (band on) first")
    ap.add_argument("--load-only", type=float, default=0.0,
                    help="only generate deploy-like DDS traffic (lowcmd 500 Hz + Dex3 cmds) for N seconds")
    ap.add_argument("--ops", action="store_true", help="exercise the REP ops (occupancy, topdown, record)")
    ap.add_argument("--out", default="/work/worldline-g1/outputs/m1/isaac/stand_test.json")
    a = ap.parse_args()
    out_dir = Path(a.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.load_only > 0:
        peer = Peer(a.domain, a.iface)
        peer.start()
        time.sleep(a.load_only)
        print(f"LOAD_DONE sent={peer.sent} lowstate_rx={len(peer.ls_times)}", flush=True)
        peer.stop()
        return 0
    rep = Rep(5600 + a.port_offset)
    report: dict = {"args": vars(a), "checks": {}}
    print("ping", rep("ping"), flush=True)
    if a.reset:
        sp = rep("get_scene_info")["spawn"]
        print("reset", rep("reset_robot", x=sp["x"], y=sp["y"], yaw=sp["yaw"]), flush=True)

    peer = Peer(a.domain, a.iface)
    t0 = time.perf_counter()
    while peer.ls is None and time.perf_counter() - t0 < 30:
        time.sleep(0.05)
    if peer.ls is None:
        print("FAIL: no rt/lowstate", flush=True)
        report["checks"]["lowstate_received"] = False
        Path(a.out).write_text(json.dumps(report, indent=2))
        return 1
    poses = PoseSub(5601 + a.port_offset)
    stand_kp, stand_kd = np.array(jm.KPS), np.array(jm.KDS)
    if a.gains == "stiff":
        stand_kp[:12] = 350.0
        stand_kd[:12] = 10.0
        stand_kp[12:15] = 400.0
        stand_kd[12:15] = 10.0
    peer.start()
    time.sleep(2.5)
    report["rates_hz"] = {"lowstate": peer.rate(peer.ls_times), "secondary_imu": peer.rate(peer.imu_times),
                          "dex3_left_state": peer.rate(peer.hand_times), "lowcmd_sent_total": peer.sent}
    report["mode_machine"] = peer.mode_machine
    print("rates", report["rates_hz"], flush=True)

    if a.camera:
        report["camera"] = check_camera(5565 + a.port_offset, str(out_dir / "ego_frame.png"))
        print("camera", {k: v for k, v in report["camera"].items() if k != "mean_rgb"}, flush=True)

    if a.joint_map:
        # stiff gains on every motor so that joints do not drag each other (soft waist + arm swing couple strongly)
        with peer.lock:
            peer.kp, peer.kd = np.full(jm.NUM_MOTORS, 300.0), np.full(jm.NUM_MOTORS, 5.0)
        rep("band", on=True, z=1.05)  # lift the feet off the ground
        time.sleep(2.0)
        names = rep("get_joint_state")["names"]
        jm_res = []
        for i, name in enumerate(jm.G1_MOTOR_JOINTS):
            base = np.array(rep("get_joint_state")["q"])  # per-joint baseline (the hanging robot sags slowly)
            # move 0.2 rad towards the side with more range (limits: g1_29dof_sonic_model12.yaml:220-233)
            d0 = jm.DEFAULT_ANGLES[i]
            delta = 0.2 if (MOTOR_UPPER[i] - d0) >= (d0 - MOTOR_LOWER[i]) else -0.2
            with peer.lock:
                peer.q_target = np.array(jm.DEFAULT_ANGLES)
                peer.q_target[i] += delta
            time.sleep(0.8)
            js = rep("get_joint_state")
            q = np.array(js["q"])
            moved = np.abs(q - base)
            j_isaac = names.index(name)
            signed = float(q[j_isaac] - base[j_isaac])
            # hands are excluded: at the default pose the hands hang next to the thighs and self-collision with a
            # moving thigh pushes the (kp=1.5) Dex3 fingers around
            others = [(names[k], round(float(moved[k]), 3)) for k in np.argsort(-moved)
                      if k != j_isaac and "_hand_" not in names[k]][:2]
            ok = moved[j_isaac] > 0.1 and np.sign(signed) == np.sign(delta) and all(m < 0.1 for _, m in others)
            jm_res.append({"motor": i, "joint": name, "cmd_delta": delta, "moved": round(signed, 3),
                           "max_other": others[0], "ok": bool(ok)})
            with peer.lock:
                peer.q_target = np.array(jm.DEFAULT_ANGLES)
            time.sleep(0.6)
        report["joint_map"] = jm_res
        report["checks"]["joint_map_all_ok"] = all(r["ok"] for r in jm_res)
        print("joint_map ok:", report["checks"]["joint_map_all_ok"],
              [r["joint"] for r in jm_res if not r["ok"]], flush=True)
        rep("band", on=True)  # back to the standing band height
        time.sleep(2.0)

    # --- stand: band on while settling, then release and hold
    with peer.lock:
        peer.kp, peer.kd = stand_kp.copy(), stand_kd.copy()
    report["stand_gains"] = {"kp": stand_kp.tolist(), "kd": stand_kd.tolist(), "profile": a.gains}
    time.sleep(a.settle_s)
    rep("record", on=True, path=str(out_dir / "stand_trace.npz"))
    r = rep("band", on=False, ramp_s=a.band_ramp_s)
    print("band release", r, flush=True)
    t_rel = time.perf_counter()
    n0 = len(poses.samples)
    t_rel_sim = poses.latest()["t_sim"] if poses.latest() else None
    # during the hold, compare lowstate vs gt at ~10 Hz
    cmp = {"imu_vs_gt_deg": [], "tick_vs_tsim_ms": []}
    while time.perf_counter() - t_rel < a.band_ramp_s + a.hold_s:
        time.sleep(0.1)
        p = poses.latest()
        with peer.lock:
            ls = peer.ls
        if p is None or ls is None:
            continue
        cmp["imu_vs_gt_deg"].append(quat_angle(list(ls.imu_state.quaternion), p["base_quat_wxyz"]))
        cmp["tick_vs_tsim_ms"].append(ls.tick - p["t_sim"] * 1000.0)
    report["rates_hz_end"] = {"lowstate": peer.rate(peer.ls_times), "secondary_imu": peer.rate(peer.imu_times),
                              "dex3_left_state": peer.rate(peer.hand_times)}
    js = rep("get_joint_state")
    with peer.lock:
        ls = peer.ls
        lsq = np.array([ls.motor_state[i].q for i in range(jm.NUM_MOTORS)])
    rec = rep("record", on=False)
    after = poses.samples[n0:]
    i_off = next((i for i, s in enumerate(after) if not s["band"]), len(after))
    hold = [s for s in after[i_off:] if s["_rx"] >= after[i_off]["_rx"] + 0.5] if i_off < len(after) else []
    pz = np.array([s["pelvis_z"] for s in hold]) if hold else np.array([np.nan])
    xy = np.array([s["base_pos"][:2] for s in hold]) if hold else np.zeros((1, 2))
    report["stand"] = {
        "hold_s": a.hold_s, "samples": len(hold), "pelvis_z_min": float(np.nanmin(pz)),
        "pelvis_z_max": float(np.nanmax(pz)), "pelvis_z_mean": float(np.nanmean(pz)),
        "xy_drift_m": float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else None,
        "fallen_any": any(s["fallen"] for s in hold), "band_off": all(not s["band"] for s in hold),
        "t_first_fallen_after_release_s": next((round(s["t_sim"] - t_rel_sim, 2) for s in poses.samples[n0:]
                                                 if s["fallen"]), None) if t_rel_sim is not None else None,
        "foot_contact_both_frac": float(np.mean([s["foot_contact"].get("left", False) and
                                                 s["foot_contact"].get("right", False) for s in hold]))
        if hold else None,
        "rtf_min": min((s["rtf"] for s in hold if s["rtf"] is not None), default=None),
        "lowstate_q_vs_isaac_max_abs": float(np.max(np.abs(lsq - np.array(js["motor_q"])))),
        "q_vs_target_max_abs": float(np.max(np.abs(np.array(js["motor_q"]) - np.array(jm.DEFAULT_ANGLES)))),
        "imu_vs_gt_deg_max": float(np.max(cmp["imu_vs_gt_deg"])) if cmp["imu_vs_gt_deg"] else None,
        "tick_minus_tsim_ms_median": float(np.median(cmp["tick_vs_tsim_ms"])) if cmp["tick_vs_tsim_ms"] else None,
        "trace": rec,
    }
    st = report["stand"]
    report["checks"].update({
        "lowstate_200hz": report["rates_hz_end"]["lowstate"] > 180,
        "secondary_imu_200hz": report["rates_hz_end"]["secondary_imu"] > 180,
        "dex3_state": report["rates_hz_end"]["dex3_left_state"] > 180,
        "stood_without_band": (not st["fallen_any"]) and st["band_off"] and st["pelvis_z_min"] > 0.65,
        "lowstate_matches_isaac": st["lowstate_q_vs_isaac_max_abs"] < 0.05,
        "imu_matches_gt": (st["imu_vs_gt_deg_max"] or 99) < 2.0,
    })
    report["stats"] = rep("get_stats")
    if a.ops:
        ops = {}
        ops["get_scene_info"] = rep("get_scene_info")
        ops["get_occupancy"] = rep("get_occupancy", robot_radius=0.25, force=True)
        ops["render_topdown"] = rep("render_topdown", path=str(out_dir / "topdown.png"), force=True)
        ops["get_pose"] = rep("get_pose")
        report["ops"] = ops
        report["checks"]["ops_ok"] = all(v.get("ok") for v in ops.values())
    report["pass"] = all(report["checks"].values())
    peer.stop()
    poses.stop()
    Path(a.out).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("STAND_TEST " + json.dumps({"pass": report["pass"], "checks": report["checks"], "stand": {
        k: v for k, v in st.items() if k != "trace"}}, default=str), flush=True)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    import os
    import sys
    code = main()
    sys.stdout.flush()
    os._exit(code)
