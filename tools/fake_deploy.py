"""Fake gear_sonic_deploy (zmq_manager input) for testing wl-body without the C++ deploy.

What it mirrors from WBC @b042411 (so wl-body bugs show up here):
- SUB connects to the SONIC input port and filters topics "command" and "planner"; messages are decoded with the
  same 1280-byte header rules and required fields (body/wire.py decode_* <-> zmq_packed_message_subscriber.hpp,
  zmq_manager.hpp:668-800). Malformed messages are counted and dropped, never guessed.
- States INIT (no "lowstate" = no gt.pose yet, then a 3 s ramp; "Init Done") -> WAIT_FOR_CONTROL -> CONTROL on
  command{start} (g1_deploy_onnx_ref.cpp:181, 2777-2790, 3834-3870). A start received during INIT is remembered
  (operator_state.start) and acted on once INIT is done, like the real state machine.
- Planner frame: on start the heading is captured from the "IMU" (P1 yaw) -> init_base_quat; movement/facing are
  planner-frame directions rotated by theta0 = yaw(init_base_quat) (+delta_heading=0) (localmotion_kplanner.hpp:591-624,
  g1_deploy_onnx_ref.cpp:560-606).
- Planner timeout 1 s -> IDLE with current facing (zmq_manager.hpp:581-642); static modes ignore speed/movement.
- command{stop} -> damping command and the process exits (g1_deploy_onnx_ref.cpp:2718-2745, 4513-4530): here the twist
  link carries damping=True and the fake stops publishing.
- g1_debug PUB: [b"g1_debug"][msgpack] ONLY in CONTROL (g1_deploy_onnx_ref.cpp:4002-4010) with index, base_quat,
  init_base_quat + delta_heading, last_action[29] (legs oscillate while walking), token_state[64], base_trans_target
  (zmq_output_handler.hpp:1-60); [b"robot_config"][msgpack] every 2 s in every state (:211-230).

Locomotion model (NOT SONIC, only plausible): first-order velocity response (tau 0.35 s, 92% of commanded speed),
yaw P-control limited to 1.2 rad/s with a +3 deg steady tracking bias. The twist goes to tools/fake_p1.py over the
fake_link PUB (stand-in for rt/lowcmd).

Run:  python -m tools.fake_deploy --port-offset 200
"""

from __future__ import annotations

import argparse
import collections
import math
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from body import joint_map as jm
from body.config import ep, ports as _ports
from body.wire import (LocomotionMode, decode_command, decode_planner, quat_wxyz_from_yaw, split_topic, wrap,
                       yaw_from_quat_wxyz)

MODE_DEFAULT_SPEED = {1: 0.4, 2: 1.2, 3: 2.5}
# SONIC's IDLE reference (g1_debug body_q_target, MuJoCo order) as measured on the M1 stack 2026-09-29
REF_Q29 = (0.07, 0.078, 0.159, 0.146, -0.176, -0.111, 0.101, -0.08, -0.356, 0.074, -0.124, 0.062,
           0.008, 0.02, 0.146, -0.03, 0.273, -0.612, 1.017, 0.053, 0.054, 0.013,
           -0.013, -0.266, 0.667, 1.023, -0.055, 0.194, -0.136)


class FakeDeploy:
    def __init__(self, port_offset: int = 200, ctx: zmq.Context | None = None, yaw_bias_deg: float = 3.0,
                 v_scale: float = 0.92, tau_v: float = 0.35, max_wz: float = 1.2, log=print):
        self.P = _ports(port_offset)
        self.ctx = ctx or zmq.Context.instance()
        self.yaw_bias = math.radians(yaw_bias_deg)
        self.v_scale, self.tau_v, self.max_wz = v_scale, tau_v, max_wz
        self.log = log
        self.state = "INIT"
        self.init_done_mono: float | None = None
        self.start_pending = False
        self.theta0: float | None = None
        self.planner: dict | None = None
        self.planner_mono = 0.0
        self.mode = LocomotionMode.IDLE
        self.facing_w: float | None = None
        self.v = np.zeros(2)
        self.wz = 0.0
        self.pose: dict | None = None
        self.phase = 0.0
        self.index = 0
        self.stats = collections.Counter()
        self.modes_seen = collections.Counter()
        self.planner_times: collections.deque[float] = collections.deque(maxlen=500)
        self.command_log: list[dict] = []
        # upper-body override model (zmq_manager.hpp:581-628 semantics; body/arm.py): the arms follow the override
        # with a first-order lag, else SONIC's IDLE reference arms (REF_Q29, as measured on the M1 stack)
        self.ref_q29 = list(REF_Q29)
        self.body_q = list(REF_Q29)
        self.has_upper = False
        self.has_hands = False
        self.upper: list[float] | None = None
        self.hands = {"left": list(jm.DEX3_DEPLOY_DEFAULT_LEFT), "right": list(jm.DEX3_DEPLOY_DEFAULT_RIGHT)}
        self.hand_q = {"left": list(jm.DEX3_DEPLOY_DEFAULT_LEFT), "right": list(jm.DEX3_DEPLOY_DEFAULT_RIGHT)}
        self.upper_log: collections.deque = collections.deque(maxlen=3000)   # (t_mono, mode, upper17|None, hands?)
        self.tau_arm = 0.08
        self.arm_bias = [0.0] * 29          # tests: a steady-state tracking error, like SONIC's (docs/arm_tracking.md)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "FakeDeploy":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="fake-deploy", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def planner_rate(self, window_s: float = 2.0) -> float:
        now = time.monotonic()
        ts = [t for t in self.planner_times if now - t <= window_s]
        return 0.0 if len(ts) < 2 else (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])

    # ---------------------------------------------------------------------------------------
    def _loop(self) -> None:
        sub = self.ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt(zmq.RCVHWM, 3)                     # zmq_manager.hpp:111-118 rcv_hwm=3
        sub.setsockopt(zmq.SUBSCRIBE, b"command")
        sub.setsockopt(zmq.SUBSCRIBE, b"planner")
        sub.connect(ep(self.P["sonic_in"]))
        imu = self.ctx.socket(zmq.SUB)
        imu.setsockopt(zmq.LINGER, 0)
        imu.setsockopt(zmq.RCVHWM, 50)                   # no CONFLATE: gt.pose is multipart
        imu.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        imu.connect(ep(self.P["p1_pose"]))
        dbg = self.ctx.socket(zmq.PUB)
        dbg.setsockopt(zmq.LINGER, 0)
        dbg.bind(ep(self.P["sonic_debug"]))
        link = self.ctx.socket(zmq.PUB)
        link.setsockopt(zmq.LINGER, 0)
        link.connect(ep(self.P["fake_link"]))
        dt = 0.02  # 50 Hz control
        nxt = time.monotonic()
        last_cfg = 0.0
        try:
            while self._running:
                self._drain_imu(imu)
                self._drain_input(sub)
                if self.state == "STOPPED":
                    for _ in range(5):
                        link.send_multipart([b"twist", msgpack.packb({"vx": 0.0, "vy": 0.0, "wz": 0.0,
                                                                      "damping": True})])
                        time.sleep(0.02)
                    self.log("[fake_deploy] command stop -> damping, exiting (like Stop())")
                    break
                now = time.monotonic()
                if self.state == "INIT" and self.init_done_mono is not None and now >= self.init_done_mono:
                    self.state = "WAIT_FOR_CONTROL"
                    self.log("[fake_deploy] Init Done -> WAIT_FOR_CONTROL")
                if self.state == "WAIT_FOR_CONTROL" and self.start_pending:
                    self._enter_control()
                self._control(dt)
                if self.state == "CONTROL":
                    link.send_multipart([b"twist", msgpack.packb({"vx": float(self.v[0]), "vy": float(self.v[1]),
                                                                  "wz": float(self.wz), "damping": False})])
                    dbg.send(b"g1_debug" + msgpack.packb(self._debug_msg(), use_bin_type=True))
                if now - last_cfg >= 2.0:
                    last_cfg = now
                    dbg.send(b"robot_config" + msgpack.packb({"fake": True, "policy": "fake_deploy"}))
                nxt += dt
                d = nxt - time.monotonic()
                if d > 0:
                    time.sleep(d)
                else:
                    nxt = time.monotonic()
        finally:
            for s in (sub, imu, dbg, link):
                s.close(0)

    def _drain_imu(self, imu) -> None:
        while True:
            try:
                frames = imu.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            payload = split_topic(frames, b"gt.pose")
            if payload is None:
                continue
            self.pose = msgpack.unpackb(payload, raw=False)
            if self.state == "INIT" and self.init_done_mono is None:
                self.init_done_mono = time.monotonic() + 3.0
                self.log("[fake_deploy] lowstate available -> INIT ramp (3 s)")

    def _drain_input(self, sub) -> None:
        while True:
            try:
                msg = sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            if msg.startswith(b"command"):
                try:
                    c = decode_command(msg)
                except Exception as e:
                    self.stats["bad_command"] += 1
                    self.log(f"[fake_deploy] bad command: {e}")
                    continue
                self.stats["command"] += 1
                self.command_log.append({**c, "t": time.time()})
                if c["stop"]:
                    self.stats["stop"] += 1
                    self.state = "STOPPED"
                    return
                if c["start"]:
                    self.stats["start"] += 1
                    if self.state in ("INIT", "WAIT_FOR_CONTROL"):
                        self.start_pending = True
            elif msg.startswith(b"planner"):
                try:
                    p = decode_planner(msg)
                except Exception as e:
                    self.stats["bad_planner"] += 1
                    self.log(f"[fake_deploy] bad planner: {e}")
                    continue
                self.stats["planner"] += 1
                self.planner_times.append(time.monotonic())
                self.planner = p
                self.planner_mono = time.monotonic()
                self.modes_seen[p["mode"]] += 1
                # every consumed planner message re-evaluates the override (zmq_manager.hpp:581-596)
                self.has_upper = "upper_body_position" in p
                self.has_hands = "left_hand_joints" in p or "right_hand_joints" in p
                if self.has_upper:
                    self.upper = p["upper_body_position"]
                for side in ("left", "right"):
                    if f"{side}_hand_joints" in p:
                        self.hands[side] = p[f"{side}_hand_joints"]
                self.upper_log.append((self.planner_mono, p["mode"], p.get("upper_body_position"),
                                       self.has_hands))

    def _enter_control(self) -> None:
        self.start_pending = False
        self.theta0 = float(self.pose["yaw"])
        self.facing_w = self.theta0
        self.state = "CONTROL"
        self.log(f"[fake_deploy] start -> CONTROL, theta0={math.degrees(self.theta0):.1f} deg")

    def _control(self, dt: float) -> None:
        if self.state != "CONTROL" or self.pose is None:
            return
        yaw = float(self.pose["yaw"])
        p = self.planner
        if p is None or time.monotonic() - self.planner_mono > 1.0:
            if p is not None:
                self.stats["planner_timeouts"] += 1
                self.planner = None
                self.has_upper = self.has_hands = False    # zmq_manager.hpp:618-628
            mode, move_p, facing_p, speed = LocomotionMode.IDLE, (0.0, 0.0), None, -1.0
        else:
            mode, move_p, facing_p, speed = p["mode"], p["movement"][:2], p["facing"][:2], p["speed"]
        self.mode = mode
        th = self.theta0
        if facing_p is not None and math.hypot(*facing_p) > 1e-9:
            self.facing_w = wrap(math.atan2(facing_p[1], facing_p[0]) + th)
        target_v = np.zeros(2)
        if mode not in LocomotionMode.STATIC and math.hypot(*move_p) > 1e-5:
            c, s = math.cos(th), math.sin(th)
            mw = np.array([c * move_p[0] - s * move_p[1], s * move_p[0] + c * move_p[1]])
            mw /= np.linalg.norm(mw)
            spd = speed if speed > 0 else MODE_DEFAULT_SPEED.get(mode, 0.4)
            target_v = mw * spd * self.v_scale
        self.v += (target_v - self.v) * min(1.0, dt / self.tau_v)
        err = wrap(self.facing_w + self.yaw_bias - yaw)
        self.wz = max(-self.max_wz, min(self.max_wz, 2.5 * err))
        if np.linalg.norm(self.v) > 0.05 or abs(self.wz) > 0.2:
            self.phase += 2 * math.pi * 1.8 * dt
        tgt = jm.mujoco_from_upper(self.upper, self.ref_q29) if (self.has_upper and self.upper) else self.ref_q29
        a = min(1.0, dt / self.tau_arm)
        for i in range(12, 29):
            self.body_q[i] += (tgt[i] + self.arm_bias[i] - self.body_q[i]) * a
        for side, dflt in (("left", jm.DEX3_DEPLOY_DEFAULT_LEFT), ("right", jm.DEX3_DEPLOY_DEFAULT_RIGHT)):
            ht = self.hands[side] if self.has_hands else dflt
            self.hand_q[side] = [q + (t - q) * min(1.0, dt / 0.15) for q, t in zip(self.hand_q[side], ht)]

    def _debug_msg(self) -> dict:
        self.index += 1
        yaw = float(self.pose["yaw"]) if self.pose else 0.0
        legs = [0.0] * 12
        if self.state == "CONTROL":
            s = math.sin(self.phase)
            legs = [(-0.3 + 0.25 * s) if i in (0, 6) else (0.6 + 0.3 * abs(s) if i in (3, 9) else 0.05 * s)
                    for i in range(12)]
        d = {"control_loop_type": "fake", "index": self.index, "ros_timestamp": 0.0,
             "base_quat": quat_wxyz_from_yaw(yaw), "base_ang_vel": [0.0, 0.0, self.wz],
             "body_q": list(self.body_q), "body_q_target": list(self.ref_q29),
             "left_hand_q": list(self.hand_q["left"]), "right_hand_q": list(self.hand_q["right"]),
             "last_action": legs + list(self.body_q[12:29]), "token_state": [0.0] * 64,
             "base_trans_target": [0.0, 0.0, 0.78]}
        if self.state == "CONTROL" and self.theta0 is not None:
            d["init_base_quat"] = quat_wxyz_from_yaw(self.theta0)
            d["delta_heading"] = 0.0
        return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=200)
    ap.add_argument("--yaw-bias-deg", type=float, default=3.0)
    args = ap.parse_args(argv)
    fd = FakeDeploy(args.port_offset, yaw_bias_deg=args.yaw_bias_deg).start()
    print(f"[fake_deploy] up on offset {args.port_offset}: {fd.P}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.wait(5.0) and fd._thread.is_alive():
        print(f"[fake_deploy] state={fd.state} mode={fd.mode} planner_hz={fd.planner_rate():.1f} "
              f"stats={dict(fd.stats)}", flush=True)
    fd.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
