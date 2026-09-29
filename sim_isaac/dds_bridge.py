"""G1DdsBridge: Unitree DDS <-> Isaac articulation, mirroring gear_sonic's MuJoCo bridge.

Reference (`$WBC` = GR00T-WholeBodyControl @ b042411):
- $WBC/gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:24-220  topics, message content, IMU conventions
- $WBC/gear_sonic/utils/mujoco_sim/base_sim.py:258-331,348-432     PD law, observation content, step order
- $WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:2291-2296,2642-2685 (deploy side)

Topics (unitree_hg IDL, identical to the real G1):
  PUB rt/lowstate (LowState_), rt/secondary_imu (IMUState_), rt/odostate (OdoState_), rt/dex3/{left,right}/state
  SUB rt/lowcmd (LowCmd_), rt/dex3/{left,right}/cmd (HandCmd_)

Differences in *mechanics* (not content) from the MuJoCo bridge, for speed (see fastdds.py):
- samples are (de)serialised with a verified fixed-layout CDR codec instead of cyclonedds-python's pure-Python
  serializer;
- subscribers are polled once per physics step (newest sample, KEEP_LAST(1) like the SDK's readers) instead of a
  listener thread per sample. The MuJoCo bridge also only ever uses the latest command (queueLen=1).

All arrays exchanged with the app are in *Isaac joint order*; the bridge maps to Unitree motor order by NAME.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import joint_map as jm
from .fastdds import FastReader, FastWriter, _set

N_M = jm.NUM_MOTORS
N_H = jm.NUM_HAND_MOTORS


def _paths(root: str, n: int, field: str) -> list[tuple]:
    return [(root, i, field) for i in range(n)]


def _vec(root: tuple, n: int) -> list[tuple]:
    return [root + (k,) for k in range(n)]


class LowStateCrc:
    """Unitree CRC of a LowState_ held in a FastWriter.

    The SDK computes the CRC over the *C struct* layout (unitree_sdk2py/utils/crc.py __packFmtHGLowState
    '<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I'*35 + '40B5I', 2092 bytes; deploy: Crc32Core(&low_state, sizeof/4 - 1),
    g1_deploy_onnx_ref.cpp:2650-2651), which differs from the CDR layout (CDR does not pad IMUState_ to 4). We mirror
    the fields we set into a C-layout buffer and run the SDK's crc32_core (crc_amd64.so) on its first 522 words.
    """

    MOTOR0 = 72        # byte offset of motor_state[0] in the C layout
    MOTOR_SIZE = 56

    def __init__(self, mode_machine: int):
        import ctypes

        from unitree_sdk2py.utils.crc import CRC
        self.buf = bytearray(2092)
        self.buf[9] = mode_machine & 0xFF                 # mode_machine (uint8 at byte 9)
        self.f32 = np.frombuffer(self.buf, dtype="<f4")
        self.u32 = np.frombuffer(self.buf, dtype="<u4")
        m = np.arange(N_M)
        base = (self.MOTOR0 + self.MOTOR_SIZE * m) // 4
        self.widx = {"q": base + 1, "dq": base + 2, "ddq": base + 3, "tau": base + 4,
                     "quat": np.arange(4, 8), "gyro": np.arange(8, 11), "acc": np.arange(11, 14)}
        self._fn = CRC().crc_lib.crc32_core
        self._arr = (ctypes.c_uint32 * 522).from_buffer(self.buf)

    def compute(self, w) -> int:
        for g, wi in self.widx.items():
            self.f32[wi] = w.f32[w.fidx[g]]
        self.u32[3] = w.u32[w.uidx["tick"][0]]
        return int(self._fn(self._arr, 522)) & 0xFFFFFFFF


class G1DdsBridge:
    def __init__(self, isaac_joint_names: list[str], domain: int, iface: str | None, crc: bool = True,
                 mode_machine: int = 0, heartbeat_ms: float = 50.0, log=print):
        from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
            unitree_hg_msg_dds__HandState_ as HandState_default,
            unitree_hg_msg_dds__IMUState_ as IMUState_default,
            unitree_hg_msg_dds__LowCmd_ as LowCmd_default,
            unitree_hg_msg_dds__LowState_ as LowState_default,
            unitree_hg_msg_dds__OdoState_ as OdoState_default,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, IMUState_, LowCmd_, LowState_, OdoState_

        self.log = log
        self.names = list(isaac_joint_names)
        idx = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in jm.G1_MOTOR_JOINTS + jm.DEX3_LEFT_JOINTS + jm.DEX3_RIGHT_JOINTS if n not in idx]
        if missing:
            raise RuntimeError(f"articulation is missing joints: {missing}")
        self.motor_idx = np.array([idx[n] for n in jm.G1_MOTOR_JOINTS], dtype=np.int64)
        self.left_idx = np.array([idx[n] for n in jm.DEX3_LEFT_JOINTS], dtype=np.int64)
        self.right_idx = np.array([idx[n] for n in jm.DEX3_RIGHT_JOINTS], dtype=np.int64)
        n = len(self.names)

        # ---- command state (Isaac order). Before the first lowcmd: hold the default pose with the deploy/training
        # gains (MuJoCo applies zero torque instead; see m1.md 1.2).
        self.q_t = np.zeros(n)
        self.dq_t = np.zeros(n)
        self.kp = np.zeros(n)
        self.kd = np.zeros(n)
        self.tau = np.zeros(n)
        self.q_t[self.motor_idx] = jm.DEFAULT_ANGLES
        self.kp[self.motor_idx] = jm.KPS
        self.kd[self.motor_idx] = jm.KDS
        for hi in (self.left_idx, self.right_idx):
            self.kp[hi] = jm.DEX3_HOLD_KP
            self.kd[hi] = jm.DEX3_HOLD_KD + jm.DEX3_JOINT_DAMPING

        self.crc_on = crc
        self.mode_machine = int(mode_machine)
        ChannelFactoryInitialize(domain, iface) if iface else ChannelFactoryInitialize(domain)
        self.domain, self.iface = domain, iface
        fac = ChannelFactory()
        part = getattr(fac, "_ChannelFactory__participant", None)
        if part is None:
            from cyclonedds.domain import DomainParticipant
            part = DomainParticipant(domain)
        self.participant = part

        def low_state_factory():
            s = LowState_default()
            s.mode_machine = self.mode_machine
            return s

        # ---- writers (unitree_sdk2py_bridge.py:65-97)
        self.w_low = FastWriter(part, "rt/lowstate", LowState_, low_state_factory, {
            "q": _paths("motor_state", N_M, "q"), "dq": _paths("motor_state", N_M, "dq"),
            "ddq": _paths("motor_state", N_M, "ddq"), "tau": _paths("motor_state", N_M, "tau_est"),
            "quat": _vec(("imu_state", "quaternion"), 4), "gyro": _vec(("imu_state", "gyroscope"), 3),
            "acc": _vec(("imu_state", "accelerometer"), 3),
        }, {"tick": [("tick",)], "crc": [("crc",)]}, log=log)
        self.w_low.slow_crc = crc
        self.low_crc = LowStateCrc(self.mode_machine)
        self.w_odo = FastWriter(part, "rt/odostate", OdoState_, OdoState_default, {
            "pos": _vec(("position",), 3), "ori": _vec(("orientation",), 4),
            "lin": _vec(("linear_velocity",), 3), "ang": _vec(("angular_velocity",), 3),
        }, {"tick": [("tick",)]}, log=log)
        self.w_imu = FastWriter(part, "rt/secondary_imu", IMUState_, IMUState_default, {
            "quat": _vec(("quaternion",), 4), "gyro": _vec(("gyroscope",), 3)}, log=log)
        hand_fields = {"q": _paths("motor_state", N_H, "q"), "dq": _paths("motor_state", N_H, "dq")}
        self.w_lh = FastWriter(part, "rt/dex3/left/state", HandState_, HandState_default, hand_fields, log=log)
        self.w_rh = FastWriter(part, "rt/dex3/right/state", HandState_, HandState_default, hand_fields, log=log)

        # ---- readers (unitree_sdk2py_bridge.py:89-97)
        cmd_fields = {f: _paths("motor_cmd", N_M, f) for f in ("q", "dq", "tau", "kp", "kd")}
        self.r_low = FastReader(part, "rt/lowcmd", LowCmd_, LowCmd_default, cmd_fields, log=log)
        hcmd_fields = {f: _paths("motor_cmd", N_H, f) for f in ("q", "dq", "tau", "kp", "kd")}
        self.r_lh = FastReader(part, "rt/dex3/left/cmd", HandCmd_, HandCmd_default, hcmd_fields, log=log)
        self.r_rh = FastReader(part, "rt/dex3/right/cmd", HandCmd_, HandCmd_default, hcmd_fields, log=log)

        self.codec_ok = self._verify_codecs()
        self.log(f"[dds] domain={domain} iface={iface} crc={crc} mode_machine={self.mode_machine} "
                 f"codec_verified={self.codec_ok}")

        # ---- stats
        self._pub_lock = threading.Lock()
        self.lowcmd_count = 0          # fresh lowcmd samples seen at physics steps (<= physics rate)
        self.leg_change_count = 0
        self.hand_cmd_count = 0
        self.first_lowcmd_wall = None
        self._lowcmd_t = None
        self._last_leg_q = None
        self._fresh_times: list[float] = []
        self._change_times: list[float] = []
        self.lowcmd_latency_ms: list[float] = []
        self._step = 0

        # ---- heartbeat: re-publish the last lowstate if physics stalls (m1.md 1.3)
        self.heartbeat_s = heartbeat_ms / 1000.0 if heartbeat_ms and heartbeat_ms > 0 else None
        self.heartbeat_pubs = 0
        self.lowstate_pubs = 0
        self._last_pub_wall = None
        self._have_state = False
        self._stop = threading.Event()
        if self.heartbeat_s:
            threading.Thread(target=self._heartbeat_loop, name="lowstate-heartbeat", daemon=True).start()

    # ------------------------------------------------------------------ codec self-test
    def _verify_codecs(self) -> dict:
        from unitree_sdk2py.utils.crc import CRC

        rng = np.random.default_rng(1)

        def fill_factory(w: FastWriter):
            def fill(s, fw):
                for g, paths in fw.float_fields.items():
                    vals = rng.uniform(-3, 3, len(paths)).astype(np.float32)
                    for p, v in zip(paths, vals):
                        _set(s, p, float(v))
                    fw.set(g, vals)
                for g, paths in fw.uint_fields.items():
                    if g == "crc":
                        continue
                    v = int(rng.integers(0, 2**31))
                    for p in paths:
                        _set(s, p, v)
                    fw.set_u(g, v)
            return fill

        res = {}
        crc = CRC()
        for w in (self.w_low, self.w_odo, self.w_imu, self.w_lh, self.w_rh):
            res[w.name] = w.verify(fill_factory(w), crc=((crc.Crc, self.low_crc.compute) if w is self.w_low else None))
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowCmd_
        res["rt/lowcmd"] = self.r_low.verify(unitree_hg_msg_dds__LowCmd_)
        res["rt/dex3/left/cmd"] = self.r_lh.verify(unitree_hg_msg_dds__HandCmd_)
        res["rt/dex3/right/cmd"] = self.r_rh.verify(unitree_hg_msg_dds__HandCmd_)
        return res

    # ------------------------------------------------------------------ main-thread API
    @property
    def have_lowcmd(self) -> bool:
        return self._lowcmd_t is not None

    def lowcmd_age_s(self) -> float | None:
        t = self._lowcmd_t
        return None if t is None else time.perf_counter() - t

    def pull_commands(self) -> bool:
        """Take the newest rt/lowcmd (every call) and Dex3 cmds (every 4th call) into the Isaac-order targets."""
        self._step += 1
        changed = False
        r = self.r_low.take_latest()
        if r is not None:
            v, ts = r
            now = time.perf_counter()
            mi = self.motor_idx
            self.q_t[mi] = v["q"]
            self.dq_t[mi] = v["dq"]
            self.kp[mi] = v["kp"]
            self.kd[mi] = v["kd"]
            self.tau[mi] = v["tau"]
            self._lowcmd_t = now
            self.lowcmd_count += 1
            if self.first_lowcmd_wall is None:
                self.first_lowcmd_wall = time.time()
            self._fresh_times.append(now)
            leg = v["q"][:12]
            if self._last_leg_q is None or not np.array_equal(leg, self._last_leg_q):
                self.leg_change_count += 1
                self._change_times.append(now)
                self._last_leg_q = leg.copy()
            if ts:
                self.lowcmd_latency_ms.append(time.time() * 1e3 - ts / 1e6)
                if len(self.lowcmd_latency_ms) > 2000:
                    del self.lowcmd_latency_ms[:1000]
            if len(self._fresh_times) > 4000:
                del self._fresh_times[:2000]
            if len(self._change_times) > 4000:
                del self._change_times[:2000]
            changed = True
        if self._step % 4 == 0:
            for rd, hi in ((self.r_lh, self.left_idx), (self.r_rh, self.right_idx)):
                h = rd.take_latest()
                if h is not None:
                    v, _ = h
                    self.q_t[hi] = v["q"]
                    self.dq_t[hi] = v["dq"]
                    self.kp[hi] = v["kp"]
                    self.kd[hi] = v["kd"] + jm.DEX3_JOINT_DAMPING   # + the MuJoCo finger joints' passive damping
                    self.tau[hi] = v["tau"]
                    self.hand_cmd_count += 1
                    changed = True
        return changed

    def publish(self, t_sim: float, q, dq, ddq, tau_est, base_pos, base_quat, base_lin_vel_w, base_ang_vel_b,
                base_lin_acc_w, torso_quat, torso_ang_vel_b) -> None:
        """Fill and publish lowstate/odostate/secondary_imu/dex3 state (unitree_sdk2py_bridge.py:170-219)."""
        mi = self.motor_idx
        tick = int(t_sim * 1e3) & 0xFFFFFFFF  # bridge :202
        with self._pub_lock:  # the heartbeat thread may re-publish these buffers
            w = self.w_low
            w.set("q", q[mi])
            w.set("dq", dq[mi])
            w.set("ddq", ddq[mi])
            w.set("tau", tau_est[mi])
            # IMU: quaternion wxyz (world), gyroscope in the pelvis frame, accelerometer = world linear acceleration
            # without gravity (MuJoCo qacc[:3]); bridge :189-194.
            w.set("quat", base_quat)
            w.set("gyro", base_ang_vel_b)
            w.set("acc", base_lin_acc_w)
            w.set_u("tick", tick)
            if self.crc_on and w.fast:
                w.set_u("crc", self.low_crc.compute(w))
            o = self.w_odo
            o.set("pos", base_pos)
            o.set("ori", base_quat)       # wxyz, as the MuJoCo bridge writes it (:187)
            o.set("lin", base_lin_vel_w)
            o.set("ang", base_ang_vel_b)
            o.set_u("tick", tick)
            self.w_imu.set("quat", torso_quat)
            self.w_imu.set("gyro", torso_ang_vel_b)
            self.w_lh.set("q", q[self.left_idx])
            self.w_lh.set("dq", dq[self.left_idx])
            self.w_rh.set("q", q[self.right_idx])
            self.w_rh.set("dq", dq[self.right_idx])
            for wr in (self.w_low, self.w_odo, self.w_imu, self.w_lh, self.w_rh):
                wr.write()
            self._last_pub_wall = time.perf_counter()
            self._have_state = True
            self.lowstate_pubs += 1

    def _heartbeat_loop(self):
        while not self._stop.wait(0.005):
            if not self._have_state:
                continue
            with self._pub_lock:
                if time.perf_counter() - self._last_pub_wall >= self.heartbeat_s:
                    self.w_low.write()
                    self.w_imu.write()
                    self._last_pub_wall = time.perf_counter()
                    self.heartbeat_pubs += 1

    def rates(self, window_s: float = 2.0) -> dict:
        now = time.perf_counter()
        lo = now - window_s
        fresh = sum(1 for t in self._fresh_times[-3000:] if t >= lo) / window_s
        ch = sum(1 for t in self._change_times[-3000:] if t >= lo) / window_s
        lat = np.asarray(self.lowcmd_latency_ms[-500:]) if self.lowcmd_latency_ms else None
        return {"lowcmd_fresh_hz": round(fresh, 1), "lowcmd_leg_change_hz": round(ch, 1),
                "lowcmd_latency_ms_p50": None if lat is None else round(float(np.median(lat)), 2),
                "lowcmd_slow_parses": self.r_low.slow_count}

    def close(self):
        self._stop.set()
