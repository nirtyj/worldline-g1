"""Headless gear_sonic MuJoCo reference simulator with a control socket, trace and video recording.

It runs the UPSTREAM simulator classes exactly as gear_sonic/scripts/run_sim_loop.py does
(SimLoopConfig -> load_wbc_yaml -> init_channel -> BaseSimulator; WBC @ b042411), with:
  * no viewer window (enable_onscreen False). The elastic band is toggled through a ZMQ REP socket
    instead of the GLFW '9' key (ElasticBand.enable, gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:352-386),
  * the same per-step loop as BaseSimulator.start (base_sim.py:598-637), plus hooks: trace at 50 Hz,
    offscreen video frames at IMAGE_DT, control-socket polling,
  * an extra rt/lowcmd DDS reader that counts messages and records leg targets (evidence for E4).
Physics, PD, DDS publishing and fall/reset behaviour are the upstream code, unchanged.

Run with the WBC .venv_sim python (Python 3.10), cwd anywhere:
  python sim_ref.py --ctl-port 5712 --out /work/worldline-g1/outputs/m1/deploy/run-X [--video] [--init-yaw-deg 0]

Control socket (REP, JSON in / JSON out):
  {"op":"ping"} | {"op":"band","on":bool} | {"op":"band_length","length":float} | {"op":"pose"}
  {"op":"stats"} | {"op":"quit"}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand


class YawBand(ElasticBand):
    """Upstream ElasticBand whose orientation hold targets a given yaw instead of identity.

    Only used with --init-yaw-deg != 0 (frame-convention experiment); the translational part and all
    gains are the upstream ones (unitree_sdk2py_bridge.py:352-378).
    """

    def __init__(self, yaw: float):
        super().__init__()
        self.q_target_inv = Rotation.from_euler("z", yaw).inv()

    def Advance(self, pose):
        pos, quat, lin_vel, ang_vel = pose[0:3], pose[3:7], pose[7:10], pose[10:13]
        f = self.kp_pos * (self.point - pos + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)
        rot = self.q_target_inv * Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        torque = -self.kp_ang * rot.as_rotvec() - self.kd_ang * ang_vel
        return np.concatenate([f, torque])


class RefSimulator(BaseSimulator):
    """BaseSimulator with instrumentation hooks; the step loop mirrors BaseSimulator.start."""

    def __init__(self, args, config, **kwargs):
        super().__init__(config=config, env_name="default", **kwargs)
        self.args = args
        env = self.sim_env
        self.m, self.d = env.mj_model, env.mj_data
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        # --- lowcmd monitor: a second reader on rt/lowcmd (same participant, same topic) ---------
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

        self.cmd_lock = threading.Lock()
        self.cmd_count = 0
        self.cmd_times: list[float] = []
        self.last_cmd_q = np.zeros(29)
        self.last_cmd_kp = np.zeros(29)
        self.last_cmd_mode_machine = -1
        self.last_cmd_msg = None
        self._decoded_msg = None
        self._last_leg_q = None
        self.tgt_changes = 0
        self.tgt_change_times: list[float] = []
        self.cmd_monitor = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.cmd_monitor.Init(self._on_lowcmd, 10)
        # --- foot geoms for contact flags ----------------------------------------------------------
        self.floor_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_bodies = {
            s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")
        }
        # --- optional initial yaw (frame experiment) -----------------------------------------------
        self.init_yaw = np.deg2rad(args.init_yaw_deg)
        if abs(self.init_yaw) > 1e-9:
            qw, qx, qy, qz = np.roll(Rotation.from_euler("z", self.init_yaw).as_quat(), 1)
            self.d.qpos[3:7] = [qw, qx, qy, qz]
            mujoco.mj_forward(self.m, self.d)
            old = env.elastic_band
            env.elastic_band = YawBand(self.init_yaw)
            env.elastic_band.enable = old.enable
            # upstream reset() (fall) goes back to the model default (yaw 0); we keep that behaviour
        # --- control socket ------------------------------------------------------------------------
        self.zctx = zmq.Context.instance()
        self.ctl = self.zctx.socket(zmq.REP)
        self.ctl.setsockopt(zmq.LINGER, 0)
        self.ctl.bind(f"tcp://127.0.0.1:{args.ctl_port}")
        # --- trace + video -------------------------------------------------------------------------
        self.trace_f = open(self.out / "sim_trace.jsonl", "w")
        self.events_f = open(self.out / "sim_events.jsonl", "w")
        self.falls = 0
        self.sim_time_total = 0.0     # sim seconds stepped (d.time restarts at 0 on an upstream fall reset)
        self.t_loop0 = None
        self.fall_times: list[float] = []
        self.t_wall0 = time.time()
        self.step_times: list[float] = []
        self.video = None
        if args.video:
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.m.body("pelvis").id
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 135.0, -20.0
            self.renderer = mujoco.Renderer(self.m, height=args.video_h, width=args.video_w)
            fps = round(1.0 / self.image_dt)
            self.video = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{args.video_w}x{args.video_h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-pix_fmt", "yuv420p", str(self.out / "sim_tracking.mp4")],
                stdin=subprocess.PIPE,
            )
        self.event("sim_ready", init_yaw_deg=args.init_yaw_deg, band=self.band_enabled())

    # ------------------------------------------------------------------------------------------------
    def _on_lowcmd(self, msg):
        # keep the DDS callback minimal (it holds the GIL at 500 Hz): count + keep a reference;
        # targets are decoded at 50 Hz in the trace hook (_decode_cmd)
        now = time.time()
        self.cmd_count += 1
        self.cmd_times.append(now)
        if len(self.cmd_times) > 5000:
            del self.cmd_times[:2500]
        self.last_cmd_msg = msg

    def _track_targets(self):
        """Called every physics step (200 Hz): detects each new leg-target vector (they change at 50 Hz)."""
        msg = self.last_cmd_msg
        if msg is None or msg is self._decoded_msg:
            return
        self._decoded_msg = msg
        mc = msg.motor_cmd
        q = [mc[i].q for i in range(12)]
        if q != self._last_leg_q:
            self._last_leg_q = q
            self.tgt_changes += 1
            self.tgt_change_times.append(time.time())
            if len(self.tgt_change_times) > 2000:
                del self.tgt_change_times[:1000]
            self.last_cmd_q[:12] = q
            self.last_cmd_kp[:12] = [mc[i].kp for i in range(12)]
            self.last_cmd_mode_machine = int(msg.mode_machine)

    def target_change_hz(self, now):
        return len([t for t in self.tgt_change_times[-400:] if now - t < 2.0]) / 2.0

    def band_enabled(self) -> bool:
        eb = self.sim_env.elastic_band
        return bool(eb is not None and eb.enable)

    def event(self, name, **kw):
        rec = {"t_sim": float(self.d.time), "t_wall": time.time(), "event": name, **kw}
        self.events_f.write(json.dumps(rec) + "\n")
        self.events_f.flush()
        print(f"[sim_ref] {name} {kw}", flush=True)

    def foot_contacts(self):
        res = {"left": False, "right": False}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.floor_gid not in (g1, g2):
                continue
            other = g2 if g1 == self.floor_gid else g1
            b = self.m.geom_bodyid[other]
            for s, fb in self.foot_bodies.items():
                if b == fb:
                    res[s] = True
        return res

    def pose(self):
        q = self.d.qpos
        quat = q[3:7].copy()
        yaw = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("zyx")[0]
        return {
            "t_sim": float(self.d.time), "t_wall": time.time(),
            "base_pos": q[0:3].tolist(), "base_quat_wxyz": quat.tolist(), "yaw": float(yaw),
            "base_lin_vel_w": self.d.qvel[0:3].tolist(), "base_ang_vel_b": self.d.qvel[3:6].tolist(),
            "pelvis_z": float(q[2]), "band": self.band_enabled(), "falls": self.falls,
            "foot_contact": self.foot_contacts(),
        }

    def stats(self):
        with self.cmd_lock:
            ts = list(self.cmd_times)
            n = self.cmd_count
        now = time.time()
        recent = [t for t in ts if now - t < 2.0]
        wall = (now - self.t_loop0) if self.t_loop0 else 0.0
        st = np.array(self.step_times[-2000:]) if self.step_times else np.zeros(1)
        return {
            "t_sim": float(self.d.time), "t_sim_total": self.sim_time_total, "t_wall_elapsed": wall,
            "rtf": float(self.sim_time_total / wall) if wall > 0 else 0.0,
            "lowcmd_count": n, "lowcmd_hz_2s": len(recent) / 2.0,
            "lowcmd_mode_machine": self.last_cmd_mode_machine,
            "lowcmd_leg_target_changes": self.tgt_changes, "lowcmd_leg_target_change_hz_2s": self.target_change_hz(now),
            "step_ms_mean": float(st.mean() * 1e3), "step_ms_p99": float(np.percentile(st, 99) * 1e3),
            "falls": self.falls, "band": self.band_enabled(), "pace": self.args.pace,
            "resyncs": getattr(self, "resyncs", 0),
        }

    def handle_ctl(self):
        try:
            raw = self.ctl.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            req = json.loads(raw)
            op = req.get("op")
            if op == "ping":
                rep = {"ok": True}
            elif op == "band":
                on = bool(req["on"])
                self.sim_env.elastic_band.enable = on
                self.event("band", on=on)
                rep = {"ok": True, "band": on}
            elif op == "band_length":
                self.sim_env.elastic_band.length = float(req["length"])
                self.event("band_length", length=float(req["length"]))
                rep = {"ok": True}
            elif op == "pose":
                rep = {"ok": True, **self.pose()}
            elif op == "stats":
                rep = {"ok": True, **self.stats()}
            elif op == "quit":
                self._running = False
                rep = {"ok": True}
            else:
                rep = {"ok": False, "error": f"unknown op {op}"}
        except Exception as e:  # never kill the physics loop on a bad request
            rep = {"ok": False, "error": repr(e)}
        self.ctl.send(json.dumps(rep).encode())

    # ------------------------------------------------------------------------------------------------
    def start(self):
        """Same loop as BaseSimulator.start (base_sim.py:598-637) + hooks."""
        sim_cnt = 0
        trace_every = max(1, int(round(0.02 / self.sim_dt)))  # 50 Hz
        img_every = int(self.image_dt / self.sim_dt)
        env = self.sim_env
        self.t_loop0 = time.time()
        next_t = time.monotonic()
        self.resyncs = 0
        try:
            while self._running:
                step_start = time.monotonic()
                z_before = float(self.d.qpos[2])
                env.sim_step()  # publish state, band, PD from latest lowcmd, mj_step, check_fall/reset
                self.sim_time_total += self.sim_dt
                self._track_targets()
                if getattr(env, "fall", False):
                    self.falls += 1
                    self.fall_times.append(time.time())
                    self.event("fall_reset", pelvis_z_before=z_before)
                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    env.update_viewer()
                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    env.update_reward()
                if sim_cnt % img_every == 0:
                    env.update_render_caches()
                    if self.video is not None:
                        self.renderer.update_scene(self.d, camera=self.cam)
                        self.video.stdin.write(self.renderer.render().tobytes())
                if sim_cnt % trace_every == 0:
                    rec = self.pose()
                    rec["t_sim_total"] = self.sim_time_total
                    with self.cmd_lock:
                        rec["lowcmd_count"] = self.cmd_count
                        rec["lowcmd_changes"] = self.tgt_changes
                        rec["cmd_q_legs"] = self.last_cmd_q[:12].round(5).tolist()
                        rec["cmd_kp_legs"] = self.last_cmd_kp[:12].round(2).tolist()
                    rec["qpos"] = self.d.qpos.round(5).tolist()  # full state -> offline video (render_ref.py)
                    self.trace_f.write(json.dumps(rec) + "\n")
                self.handle_ctl()
                if self.args.duration and self.d.time >= self.args.duration:
                    self.event("duration_reached")
                    break
                elapsed = time.monotonic() - step_start
                self.step_times.append(elapsed)
                if len(self.step_times) > 20000:
                    self.step_times = self.step_times[-10000:]
                if self.args.pace == "upstream":
                    # base_sim.py:627-631: sleep the remainder of this step; overruns and sleep overshoot are lost
                    sleep_time = self.sim_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    # absolute deadlines: overshoot of one sleep is recovered on the next step, so RTF stays 1.0
                    # whenever the mean step cost is below sim_dt; >100 ms behind -> resync instead of bursting
                    next_t += self.sim_dt
                    sleep_time = next_t - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -0.1:
                        next_t = time.monotonic()
                        self.resyncs += 1
                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.finish()

    def finish(self):
        st = self.stats()
        self.event("sim_exit", **st)
        json.dump(st, open(self.out / "sim_stats.json", "w"), indent=1)
        self.trace_f.close()
        if self.video is not None:
            self.video.stdin.close()
            self.video.wait(timeout=60)
        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (the deploy is hard-wired to 0)")
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--ctl-port", type=int, default=5712)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.0, help="sim seconds, 0 = until quit")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-w", type=int, default=640)
    ap.add_argument("--video-h", type=int, default=480)
    ap.add_argument("--init-yaw-deg", type=float, default=0.0)
    ap.add_argument("--camera-port", type=int, default=0, help="also publish ego_view like upstream (0 = off)")
    ap.add_argument("--pace", choices=["deadline", "upstream"], default="deadline",
                    help="wall-clock pacing: absolute deadlines (default) or the upstream per-step sleep")
    args = ap.parse_args()

    # == run_sim_loop.py:36-55, headless
    cfg = SimLoopConfig(interface=args.iface, enable_onscreen=False, enable_offscreen=bool(args.camera_port),
                        enable_image_publish=bool(args.camera_port), camera_port=args.camera_port or 5555)
    wbc_config = cfg.load_wbc_yaml()
    wbc_config["ENV_NAME"] = cfg.env_name
    wbc_config["DOMAIN_ID"] = args.domain
    init_channel(config=wbc_config)
    sim = RefSimulator(
        args, wbc_config,
        onscreen=False, offscreen=bool(args.camera_port), enable_image_publish=bool(args.camera_port),
    )
    signal.signal(signal.SIGTERM, lambda *_: setattr(sim, "_running", False))
    if args.camera_port:
        sim.start_image_publish_subprocess(start_method=cfg.mp_start_method, camera_port=args.camera_port)
        time.sleep(1)
    print(f"[sim_ref] running: domain={args.domain} iface={args.iface} ctl=tcp://127.0.0.1:{args.ctl_port} "
          f"dt={sim.sim_dt} band={sim.band_enabled()} out={args.out}", flush=True)
    sim.start()


if __name__ == "__main__":
    main()
