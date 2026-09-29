"""wl-isaac (P1): Isaac Sim 5.1 / Isaac Lab 2.3.2 host for a SONIC-driven Unitree G1.

    /work/envs/isaaclab/bin/python -m sim_isaac.app --house empty --physics-hz 200 \
        --dds-domain 0 --dds-iface lo --camera 640x480 --camera-hz 30 --rt-pace

Interfaces: docs/contracts/m1.md (DDS rt/lowstate|secondary_imu|odostate|dex3/*/state out, rt/lowcmd|dex3/*/cmd
in; ZMQ PUB 5565 camera (gear_sonic format), PUB 5601 gt.pose, REP 5600 control). The physics loop mirrors the
gear_sonic MuJoCo sim loop ($WBC/gear_sonic/utils/mujoco_sim/base_sim.py:389-432, 598-633): 200 Hz physics,
lowstate every step, camera every IMAGE_DT of sim time, wall-clock pacing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_isaac import joint_map as jm  # noqa: E402
from sim_isaac.mathutil import quat_rotate_inverse, up_z, yaw_from_quat  # noqa: E402

BASE_PORTS = {"rep": 5600, "gt_pub": 5601, "frames_pub": 5602, "camera": 5565}


def parse_args():
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--house", default="empty", help="house id for scenes/loader.py, or 'empty'")
    ap.add_argument("--house-dynamic", choices=["keep", "kinematic"], default=None,
                    help="scenes.loader dynamic_objects option (kinematic = loose props frozen)")
    ap.add_argument("--physics-hz", type=float, default=200.0)
    ap.add_argument("--physx-device", choices=["cpu", "cuda"], default="cpu",
                    help="PhysX pipeline: cpu (default, lowest latency for 1 robot) or cuda (GPU pipeline)")
    ap.add_argument("--pd", choices=["implicit", "explicit"], default="implicit")
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--dds-iface", default="lo")
    ap.add_argument("--no-dds", action="store_true", help="run without DDS (RTF measurements of the sim only)")
    ap.add_argument("--crc", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mode-machine", type=int, default=0, help="LowState.mode_machine (MuJoCo bridge leaves 0)")
    ap.add_argument("--heartbeat-ms", type=float, default=50.0)
    ap.add_argument("--camera", default="640x480", help="WxH of the head camera, or 'none'")
    ap.add_argument("--camera-hz", type=float, default=30.0, help="camera rate in sim time")
    ap.add_argument("--camera-vfov", type=float, default=45.0)
    ap.add_argument("--tp-camera", action="store_true", help="third-person chase camera on PUB 5602 'frame.tp'")
    ap.add_argument("--tp-hz", type=float, default=10.0)
    ap.add_argument("--rt-pace", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-lag-ms", type=float, default=100.0)
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--spawn", default=None, help="x,y,yaw (default: the scene's spawn)")
    ap.add_argument("--band-z", type=float, default=0.80, help="band anchor height above the floor")
    ap.add_argument("--no-band", action="store_true", help="start without the elastic band")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after this many wall seconds (0 = run)")
    ap.add_argument("--stats-out", default=None, help="write final stats JSON here")
    ap.add_argument("--stats-every", type=float, default=10.0, help="log stats every N wall seconds")
    ap.add_argument("--out-dir", default="/work/worldline-g1/outputs/m1/isaac", help="default dir for artefacts")
    ap.add_argument("--warmup-renders", type=int, default=30)
    ap.add_argument("--dl-denoiser", action=argparse.BooleanOptionalAction, default=None,
                    help="override the RTX preset's DL denoiser setting (default: keep the preset's)")
    ap.add_argument("--gc-freeze", action=argparse.BooleanOptionalAction, default=True,
                    help="gc.freeze() after start-up so collections do not rescan Kit's objects")
    ap.add_argument("--no-default-lights", action="store_true")
    AppLauncher.add_app_launcher_args(ap)
    # RTX preset: "balanced" (Isaac Lab default). "performance" renders ~1 ms faster in a house but the frames are
    # extremely noisy (Laplacian std 310 vs 2.2, even with the DL denoiser): tools/render_ab.sh, outputs/.../render_ab
    ap.set_defaults(rendering_mode="balanced")
    a = ap.parse_args()
    a.headless = True
    a.device = "cpu" if a.physx_device == "cpu" else "cuda:0"
    if a.camera != "none" or a.tp_camera:
        a.enable_cameras = True
    return a


class App:
    def __init__(self, args):
        self.a = args
        self.t_start_wall = time.time()
        self.running = True
        self.events: list[dict] = []

    # ------------------------------------------------------------------ logging
    def log(self, msg: str) -> None:
        print(f"[wl-isaac {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        a = self.a
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        from sim_isaac import g1_asset
        from sim_isaac.band import ElasticBand
        from sim_isaac.rt_pacer import DurationStats, RollingRate, RtPacer
        from sim_isaac.scene import build_scene

        self.torch = torch
        self.dt = 1.0 / a.physics_hz
        # bind the ZMQ server sockets first: fail fast if another instance holds the ports
        import zmq

        from sim_isaac.gt_server import GtServer
        off = a.port_offset
        self.ports = {k: v + off for k, v in BASE_PORTS.items()}
        self.zctx = zmq.Context.instance()
        self.gt = GtServer(self.zctx, self.ports["rep"], self.ports["gt_pub"], log=self.log)
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.dt, render_interval=1, device=a.device, enable_scene_query_support=True,
            # ground contact material used in training: modular_tracking_env_cfg.py:319-330, :971
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0),
            physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),  # modular_tracking_env_cfg.py:972
            render=sim_utils.RenderCfg(antialiasing_mode="Off", enable_dl_denoiser=a.dl_denoiser),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()

        t0 = time.perf_counter()
        lk = {"dynamic_objects": a.house_dynamic} if a.house_dynamic else {}
        self.scene = build_scene(self.stage, a.house, log=self.log, default_lights=not a.no_default_lights,
                                 loader_kwargs=lk)
        self.scene_load_s = time.perf_counter() - t0
        self.floor_z = self.scene.floor_z
        if a.spawn:
            sx, sy, syaw = (float(v) for v in a.spawn.split(","))
        else:
            sx, sy, syaw = self.scene.spawn
        self.spawn = (sx, sy, syaw)

        usd = g1_asset.usd_path()
        if not usd.exists():
            raise RuntimeError(f"G1 USD missing: {usd}. Build it: python -m sim_isaac.g1_asset --build")
        spawn_z = self.floor_z + a.band_z
        self.robot = Articulation(g1_asset.make_articulation_cfg(str(usd), "/World/G1", (sx, sy, spawn_z), syaw))
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/G1/.*_ankle_roll_link", update_period=0.0, history_length=1))

        # head camera at d435_link on torso_link (g1_29dof_with_hand.urdf:614-619)
        self.cam = None
        self.cam_wh = None
        if a.camera != "none":
            from sim_isaac.camera import spawn_camera_prim
            w, h = (int(v) for v in a.camera.lower().split("x"))
            self.cam_wh = (w, h)
            spawn_camera_prim("/World/G1/torso_link/ego_cam", w, h, a.camera_vfov, jm.D435_XYZ,
                              jm.quat_wxyz_from_rpy(*jm.D435_RPY))

        self.sim.reset()
        self.robot.update(0.0)
        self.names = list(self.robot.joint_names)
        self.n = len(self.names)
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.body_ids_t = torch.tensor([self.pelvis_id], dtype=torch.int32, device=self.sim.device)
        self.foot_names = self.contact.body_names
        self.log(f"articulation: {self.n} joints, {self.robot.num_bodies} bodies, pelvis={self.pelvis_id}, "
                 f"torso={self.torso_id}, feet={self.foot_names}, device={self.sim.device}")

        # joint maps and limits
        idx = {nme: i for i, nme in enumerate(self.names)}
        self.motor_idx = np.array([idx[x] for x in jm.G1_MOTOR_JOINTS])
        eff = self.robot.data.joint_effort_limits[0].detach().cpu().numpy().astype(np.float64) \
            if hasattr(self.robot.data, "joint_effort_limits") else np.full(self.n, 1e3)
        for nme, lim in jm.TRAIN_EFFORT_LIMIT.items():
            eff[idx[nme]] = lim
        self.effort_limit = eff
        self.default_q = self.robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)

        self._setup_fast_io()

        # bridge (or a local stand-in with the same command arrays when --no-dds)
        if a.no_dds:
            self.bridge = _NullBridge(self.names)
        else:
            from sim_isaac.dds_bridge import G1DdsBridge
            self.bridge = G1DdsBridge(self.names, a.dds_domain, a.dds_iface, crc=a.crc,
                                      mode_machine=a.mode_machine, heartbeat_ms=a.heartbeat_ms, log=self.log)
        self._written_kp = None
        self._written_kd = None
        if a.pd == "explicit":
            z = torch.zeros((1, self.n), device=self.sim.device)
            self.robot.write_joint_stiffness_to_sim(z)
            self.robot.write_joint_damping_to_sim(z)
        self._sync_gains(force=True)

        # band
        self.band = ElasticBand()
        st = self._read_state()
        if not a.no_band:
            self.band.engage(st["base_pos"], st["base_quat"], self.floor_z + a.band_z)

        # cameras / renderers
        self.capture = None
        if self.cam_wh:
            from sim_isaac.camera import RgbCapture
            self.capture = RgbCapture("/World/G1/torso_link/ego_cam", *self.cam_wh)
        self.chase = None
        if a.tp_camera:
            from sim_isaac.camera import ChaseCamera
            self.chase = ChaseCamera(self.stage, "/World/wl_chase_cam")
        if self.capture or self.chase:
            t0 = time.perf_counter()
            for _ in range(max(1, a.warmup_renders)):
                if self.chase:
                    self.chase.update_pose(st["base_pos"], st["yaw"])
                self.sim.render()
            self.warmup_s = time.perf_counter() - t0
            img = self.capture.read() if self.capture else None
            self.log(f"camera warm-up {a.warmup_renders} renders in {self.warmup_s:.1f}s; "
                     f"ego frame {None if img is None else img.shape}")
        else:
            self.warmup_s = 0.0
        # pre-render the top-down image once while the band holds the robot (render_topdown then only copies it)
        self.topdown_cache = None
        if self.a.enable_cameras:
            try:
                from sim_isaac.camera import render_topdown
                Path(a.out_dir).mkdir(parents=True, exist_ok=True)
                self.topdown_cache = render_topdown(self.sim, self.stage, self.scene.bounds,
                                                    str(Path(a.out_dir) / f"_topdown_{self.scene.house_id}.png"))
                self.log(f"top-down cached: {self.topdown_cache}")
            except Exception as e:  # noqa: BLE001
                self.log(f"top-down pre-render failed: {e}")

        # zmq
        import zmq

        from sim_isaac.camera import FramePublisher
        from sim_isaac.gt_server import GtServer

        self.cam_pub = FramePublisher(self.zctx, self.ports["camera"], "ego_view") if self.capture else None
        self.frames_pub = FramePublisher(self.zctx, self.ports["frames_pub"], "tp", mode="multipart",
                                         topic=b"frame.tp") if self.chase else None
        for op in ("ping", "get_pose", "get_scene_info", "get_occupancy", "band", "reset_robot", "get_stats",
                   "render_topdown", "get_joint_state", "record", "shutdown"):
            self.gt.register(op, getattr(self, f"op_{op}"))

        # stats
        self.pacer = RtPacer(self.dt, enabled=a.rt_pace, max_lag_s=a.max_lag_ms / 1000.0)
        self.step_stats = DurationStats()
        self.render_stats = DurationStats(600)
        self.render_rate = RollingRate(2.0)
        self.lowstate_rate = RollingRate(2.0)
        self.pose_rate = RollingRate(2.0)
        self.root_writes = 0
        self.gt_seq = 0
        self.cam_seq = 0
        self.last_pose: dict = {}
        self.fallen = False
        self.recording = None
        self.occ_cache: dict = {}
        self._cpu0 = os.times()
        self._wall0 = time.perf_counter()
        self.rtf_samples, self.rtf_below, self.rtf_min = 0, 0, 9.9
        self.hitches: list[dict] = []
        self.hitch_count = 0
        self._setup_gc()
        self.prof = {"cmd": 0.0, "band_write": 0.0, "physx_step": 0.0, "update_read": 0.0, "dds_publish": 0.0,
                     "n": 0}

    # ------------------------------------------------------------------ GC
    def _setup_gc(self) -> None:
        """Measure (and, with --gc-freeze, shorten) Python GC pauses: a full collection over Kit's millions of
        Python objects can stall the physics loop for tens of ms."""
        import gc

        self.gc_ms_total = 0.0
        self.gc_ms_max = 0.0
        self.gc_ms_window = 0.0
        self.gc_counts = [0, 0, 0]
        self._gc_t0 = None

        def cb(phase, info):
            if phase == "start":
                self._gc_t0 = time.perf_counter()
            elif self._gc_t0 is not None:
                ms = (time.perf_counter() - self._gc_t0) * 1e3
                self.gc_ms_total += ms
                self.gc_ms_window += ms
                self.gc_ms_max = max(self.gc_ms_max, ms)
                self.gc_counts[info.get("generation", 0)] += 1

        if self.a.gc_freeze:
            t0 = time.perf_counter()
            gc.collect()
            gc.freeze()  # everything allocated during start-up moves to the permanent generation
            self.log(f"gc.freeze(): {gc.get_freeze_count()} objects frozen "
                     f"(start-up collect {(time.perf_counter() - t0) * 1e3:.0f} ms, not counted in stats)")
        # registered after the start-up collect: gc stats describe collections inside the physics loop only
        gc.callbacks.append(cb)

    # ------------------------------------------------------------------ state
    def _setup_fast_io(self) -> None:
        """Direct PhysX tensor-view I/O (Isaac Lab's Articulation wrappers cost ~3 ms/step for one robot)."""
        t = self.torch
        dev = self.sim.device
        self.view = self.robot.root_physx_view
        self.view_idx = self.robot._ALL_INDICES
        nb = self.robot.num_bodies
        self.force_buf = t.zeros((nb, 3), dtype=t.float32, device=dev)
        self.torque_buf = t.zeros((nb, 3), dtype=t.float32, device=dev)
        # pelvis CoM offset in the pelvis frame: PhysX root velocity is the CoM velocity; the link-origin velocity
        # (what MuJoCo's free-joint qvel reports) is v_com + w x (R * -com_b)  (articulation_data.py:487-504)
        self.pelvis_com_b = self.robot.data.body_com_pos_b[0, self.pelvis_id].detach().cpu().numpy().astype(
            np.float64)
        self._dq_prev = None

    def _read_state(self) -> dict:
        """One device->host copy of everything the bridge and gt.pose need (PhysX tensor views)."""
        torch = self.torch
        v = self.view
        pid, tid = self.pelvis_id, self.torso_id
        lt = v.get_link_transforms()
        lv = v.get_link_velocities()
        la = v.get_link_accelerations()
        flat = torch.cat([
            v.get_dof_positions()[0], v.get_dof_velocities()[0],
            v.get_root_transforms()[0], v.get_root_velocities()[0],
            lt[0, tid], lv[0, tid, 3:6], la[0, pid, 0:3],
        ]).detach().to("cpu", dtype=torch.float64).numpy()
        n = self.n
        q, dq = flat[:n], flat[n:2 * n]
        o = 2 * n
        base_pos = flat[o:o + 3]
        bq = flat[o + 3:o + 7]                       # PhysX xyzw
        base_quat = np.array([bq[3], bq[0], bq[1], bq[2]])
        v_com, ang_w = flat[o + 7:o + 10], flat[o + 10:o + 13]
        tq = flat[o + 16:o + 20]
        torso_quat = np.array([tq[3], tq[0], tq[1], tq[2]])
        torso_ang_w = flat[o + 20:o + 23]
        acc_w = flat[o + 23:o + 26]
        from sim_isaac.mathutil import quat_rotate
        lin_w = v_com + np.cross(ang_w, quat_rotate(base_quat, -self.pelvis_com_b))
        ddq = np.zeros(n) if self._dq_prev is None else (dq - self._dq_prev) / self.dt
        self._dq_prev = dq
        return {"q": q, "dq": dq, "ddq": ddq, "base_pos": base_pos, "base_quat": base_quat, "lin_w": lin_w,
                "ang_w": ang_w, "ang_b": quat_rotate_inverse(base_quat, ang_w), "acc_w": acc_w,
                "torso_quat": torso_quat, "torso_ang_b": quat_rotate_inverse(torso_quat, torso_ang_w),
                "yaw": yaw_from_quat(base_quat)}

    def _sync_gains(self, force: bool = False) -> None:
        if self.a.pd != "implicit":
            return
        b = self.bridge
        if not force and self._written_kp is not None:
            dk = np.abs(b.kp - self._written_kp) > (0.01 * np.abs(self._written_kp) + 1e-6)
            dd = np.abs(b.kd - self._written_kd) > (0.01 * np.abs(self._written_kd) + 1e-6)
            if not dk.any() and not dd.any():
                return
        t = self.torch
        self.robot.write_joint_stiffness_to_sim(t.tensor(b.kp[None], dtype=t.float32, device=self.sim.device))
        self.robot.write_joint_damping_to_sim(t.tensor(b.kd[None], dtype=t.float32, device=self.sim.device))
        self._written_kp = b.kp.copy()
        self._written_kd = b.kd.copy()

    def _apply_commands(self, st: dict) -> np.ndarray:
        """Write targets for this physics step; returns tau_est (Isaac order) for lowstate."""
        t = self.torch
        b = self.bridge
        b.pull_commands()
        dev = self.sim.device
        pd_tau = b.kp * (b.q_t - st["q"]) + b.kd * (b.dq_t - st["dq"]) + b.tau
        tau_est = np.clip(pd_tau, -self.effort_limit, self.effort_limit)
        v, idx = self.view, self.view_idx
        if self.a.pd == "implicit":
            self._sync_gains()
            v.set_dof_position_targets(t.from_numpy(b.q_t.astype(np.float32)[None]).to(dev), idx)
            v.set_dof_velocity_targets(t.from_numpy(b.dq_t.astype(np.float32)[None]).to(dev), idx)
            v.set_dof_actuation_forces(t.from_numpy(b.tau.astype(np.float32)[None]).to(dev), idx)
        else:
            # explicit PD at the physics rate, clipped to the effort limit (base_sim.py:258-288, 423)
            v.set_dof_actuation_forces(t.from_numpy(tau_est.astype(np.float32)[None]).to(dev), idx)
        return tau_est

    def _apply_band(self, st: dict) -> None:
        w = self.band.advance(self.dt, st["base_pos"], st["base_quat"], st["lin_w"], st["ang_w"])
        if w is None:
            return
        t = self.torch
        pid = self.pelvis_id
        self.force_buf[pid] = t.from_numpy(w[0].astype(np.float32))
        self.torque_buf[pid] = t.from_numpy(w[1].astype(np.float32))
        # world-frame wrench like MuJoCo xfrc_applied (base_sim.py:412)
        self.view.apply_forces_and_torques_at_position(
            force_data=self.force_buf, torque_data=self.torque_buf, position_data=None, indices=self.view_idx,
            is_global=True)

    def _foot_contact(self) -> dict:
        f = self.contact.data.net_forces_w[0].detach().cpu().numpy()
        mag = np.linalg.norm(f, axis=-1)
        res = {}
        for nme, m in zip(self.foot_names, mag):
            res["left" if nme.startswith("left") else "right"] = bool(m > 20.0)
        return res

    def _pose_msg(self, st: dict, t_sim: float) -> dict:
        pz = float(st["base_pos"][2] - self.floor_z)
        fallen = pz < 0.45 or up_z(st["base_quat"]) < 0.5
        age = self.bridge.lowcmd_age_s()
        return {
            "seq": self.gt_seq, "t_sim": t_sim, "t_wall": time.time(), "rtf": self.pacer.rtf(1.0),
            "base_pos": st["base_pos"].tolist(), "base_quat_wxyz": st["base_quat"].tolist(),
            "base_lin_vel_w": st["lin_w"].tolist(), "base_ang_vel_w": st["ang_w"].tolist(),
            "yaw": float(st["yaw"]), "pelvis_z": pz, "fallen": bool(fallen),
            "foot_contact": self._foot_contact(), "band": bool(self.band.enabled),
            "lowcmd_age_s": None if age is None else round(age, 4),
        }

    def _event(self, name: str, **kw) -> None:
        ev = {"t_sim": self.pacer.t_sim, "t_wall": time.time(), "event": name, **kw}
        self.events.append(ev)
        self.gt.publish("gt.event", ev)
        self.log(f"event {json.dumps(ev)}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        a = self.a
        steps_per_pose = max(1, int(round(a.physics_hz / 50.0)))
        cam_dt = 1.0 / a.camera_hz if a.camera_hz > 0 else None
        tp_dt = 1.0 / a.tp_hz if a.tp_hz > 0 else None
        next_cam = 0.0
        next_tp = 0.0
        next_stats = time.perf_counter() + a.stats_every
        deadline = time.perf_counter() + a.duration if a.duration > 0 else None
        ready = {"ports": self.ports, "house": self.scene.house_id, "spawn": self.spawn, "floor_z": self.floor_z,
                 "physx_device": a.physx_device, "pd": a.pd, "dds_domain": a.dds_domain, "dds_iface": a.dds_iface,
                 "camera": a.camera, "camera_hz": a.camera_hz, "band": self.band.enabled,
                 "scene_load_s": round(self.scene_load_s, 2), "warmup_s": round(self.warmup_s, 2)}
        print("WL_ISAAC_READY " + json.dumps(ready), flush=True)
        self.pacer.mark_start_sim()
        st = self._read_state()
        prof = self.prof
        pc = time.perf_counter
        while self.running and not self._sim_closed():
            t0 = pc()
            tau_est = self._apply_commands(st)
            t1 = pc()
            self._apply_band(st)
            t2 = pc()
            self.sim.step(render=False)
            t3 = pc()
            self.pacer.step_done()
            t_sim = self.pacer.t_sim
            st = self._read_state()
            t4 = pc()
            if not a.no_dds:
                self.bridge.publish(t_sim, st["q"], st["dq"], st["ddq"], tau_est, st["base_pos"], st["base_quat"],
                                    st["lin_w"], st["ang_b"], st["acc_w"], st["torso_quat"], st["torso_ang_b"])
                self.lowstate_rate.tick(t4)
            t5 = pc()
            prof["cmd"] += t1 - t0
            prof["band_write"] += t2 - t1
            prof["physx_step"] += t3 - t2
            prof["update_read"] += t4 - t3
            prof["dds_publish"] += t5 - t4
            prof["n"] += 1
            self.step_stats.add((t5 - t0) * 1e3)

            # camera(s): rendered on the sim-time schedule of MuJoCo's IMAGE_DT (base_sim.py:624-625)
            do_cam = cam_dt is not None and self.capture is not None and t_sim + 1e-9 >= next_cam
            do_tp = tp_dt is not None and self.chase is not None and t_sim + 1e-9 >= next_tp
            if do_cam or do_tp:
                r0 = time.perf_counter()
                # note: every render updates all render products; toggling hydra_texture updates per frame was
                # tried and broke the chase stream, so the chase camera simply costs a second product per render
                if do_tp:
                    self.chase.update_pose(st["base_pos"], st["yaw"])
                self.sim.render()
                self.render_stats.add((time.perf_counter() - r0) * 1e3)
                self.render_rate.tick()
                if do_cam:
                    img = self.capture.read()
                    if img is not None:
                        self.cam_seq += 1
                        self.cam_pub.submit(img.copy(), t_sim, self.cam_seq)
                    next_cam += cam_dt
                    if next_cam < t_sim:
                        next_cam = t_sim + cam_dt
                if do_tp:
                    img = self.chase.cap.read()
                    if img is not None:
                        self.frames_pub.submit(img.copy(), t_sim, self.cam_seq,
                                               {"base_pos": st["base_pos"].tolist(), "yaw": float(st["yaw"])})
                    next_tp += tp_dt
                    if next_tp < t_sim:
                        next_tp = t_sim + tp_dt

            if self.pacer.n_steps % steps_per_pose == 0:
                self.contact.update(self.dt * steps_per_pose)
                self.gt_seq += 1
                self.last_pose = self._pose_msg(st, t_sim)
                r1 = self.last_pose["rtf"]
                if r1 is not None and t_sim - self.pacer._start_sim > 5.0:
                    self.rtf_samples += 1
                    self.rtf_below += r1 < 0.95
                    self.rtf_min = min(self.rtf_min, r1)
                self.gt.publish("gt.pose", self.last_pose)
                self.pose_rate.tick()
                if self.last_pose["fallen"] != self.fallen:
                    self.fallen = self.last_pose["fallen"]
                    self._event("fallen" if self.fallen else "recovered", pelvis_z=self.last_pose["pelvis_z"])
                if self.recording is not None:
                    self._record_sample(st, t_sim)
            if self.gt.poll(2):
                st = self._read_state()  # an op may have changed the sim (reset_robot, band)

            now = time.perf_counter()
            it_ms = (now - t0) * 1e3
            if it_ms > 25.0:
                self.hitches.append({"t_sim": round(t_sim, 3), "iter_ms": round(it_ms, 1),
                                     "step_ms": round((t5 - t0) * 1e3, 1),
                                     "render_ms": round(self.render_stats.buf[-1], 1)
                                     if (do_cam or do_tp) and self.render_stats.buf else 0.0,
                                     "gc_ms_since": round(self.gc_ms_window, 1)})
                self.hitch_count += 1
                if len(self.hitches) > 50:
                    del self.hitches[:25]
            self.gc_ms_window = 0.0
            if now >= next_stats:
                next_stats = now + a.stats_every
                self.log("stats " + json.dumps(self.stats(), default=str))
            if deadline is not None and now >= deadline:
                self.log("duration reached")
                break
            self.pacer.wait()

    def _sim_closed(self) -> bool:
        return False

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        now = time.perf_counter()
        c1 = os.times()
        cpu_s = (c1.user - self._cpu0.user) + (c1.system - self._cpu0.system)
        wall = now - self._wall0
        s = {
            "t_sim": round(self.pacer.t_sim, 3), "uptime_s": round(time.time() - self.t_start_wall, 1),
            "rtf_1s": _r(self.pacer.rtf(1.0)), "rtf_10s": _r(self.pacer.rtf(10.0)),
            "rtf_total": _r(self.pacer.rtf_total()),
            "rtf_1s_min": _r(self.rtf_min) if self.rtf_samples else None,
            "rtf_1s_below_0p95_frac": round(self.rtf_below / self.rtf_samples, 4) if self.rtf_samples else None,
            "physics_hz_1s": _r(self.pacer.physics_hz(1.0), 1), "physics_hz_10s": _r(self.pacer.physics_hz(10.0), 1),
            "physics_hz_target": self.a.physics_hz,
            "step_ms": self.step_stats.summary(), "render_ms": self.render_stats.summary(),
            "render_hz": round(self.render_rate.rate(), 1),
            "lowstate_pub_hz": round(self.lowstate_rate.rate(), 1), "gt_pose_hz": round(self.pose_rate.rate(), 1),
            "camera_pub_hz": round(self.cam_pub.rate(), 1) if self.cam_pub else 0.0,
            "camera_dropped": self.cam_pub.dropped if self.cam_pub else 0,
            "overruns": self.pacer.overruns, "lost_s": round(self.pacer.lost_s, 3),
            "root_writes": self.root_writes, "physx_device": self.a.physx_device, "pd_mode": self.a.pd,
            "rt_pace": self.a.rt_pace, "camera": self.a.camera, "camera_hz_target": self.a.camera_hz,
            "house": self.scene.house_id, "process_cpu_pct": round(100.0 * cpu_s / wall, 1) if wall > 0 else None,
            "gt_requests": self.gt.requests, "gt_slowest_ms": round(self.gt.slow_ms, 2),
            "hitches_gt25ms": self.hitch_count, "hitches_last": self.hitches[-5:],
            "gc": {"ms_total": round(self.gc_ms_total, 1), "ms_max": round(self.gc_ms_max, 1),
                   "counts": self.gc_counts, "frozen": self.a.gc_freeze},
            "step_breakdown_ms": {k: round(1e3 * v / max(1, self.prof["n"]), 3) for k, v in self.prof.items()
                                  if k != "n"},
        }
        if not self.a.no_dds:
            b = self.bridge
            s.update(b.rates())
            s.update({"lowcmd_count": b.lowcmd_count, "lowstate_pubs": b.lowstate_pubs,
                      "heartbeat_pubs": b.heartbeat_pubs, "hand_cmd_count": b.hand_cmd_count})
        return s

    # ------------------------------------------------------------------ REP ops
    def op_ping(self, req):
        return {"t_sim": self.pacer.t_sim, "t_wall": time.time(), "pid": os.getpid(), "house_id": self.scene.house_id,
                "band": self.band.enabled, "uptime_s": round(time.time() - self.t_start_wall, 1)}

    def op_get_pose(self, req):
        if not self.last_pose:
            self.last_pose = self._pose_msg(self._read_state(), self.pacer.t_sim)
        return dict(self.last_pose)

    def op_get_scene_info(self, req):
        d = self.scene.to_dict()
        d.setdefault("spawn", {"x": self.spawn[0], "y": self.spawn[1], "yaw": self.spawn[2]})
        return d

    def op_get_occupancy(self, req):
        from sim_isaac import occupancy
        rr = float(req.get("robot_radius", 0.25))
        res = float(req.get("resolution", 0.05))
        zmin, zmax = float(req.get("z_min", 0.10)), float(req.get("z_max", 1.60))
        key = (rr, res, zmin, zmax)
        if key in self.occ_cache and not req.get("refresh"):
            return dict(self.occ_cache[key])
        if self.scene.source == "scenes.loader" and not req.get("physx"):
            # the house agent's cached, verified grid (scenes/occupancy.py occupancy_reply)
            try:
                from scenes.occupancy import occupancy_reply
                info = occupancy_reply(self.scene.house_id, rr)
                self.occ_cache[key] = info
                return dict(info)
            except Exception as e:  # noqa: BLE001
                self.log(f"[occupancy] scenes.occupancy_reply failed ({e}); falling back")
        out_dir = Path(self.a.out_dir) / "occupancy"
        path = str(out_dir / f"{self.scene.house_id}_r{int(rr * 100)}_res{int(res * 100)}.npz")
        if self.scene.occupancy_npz and Path(self.scene.occupancy_npz).exists() and not req.get("physx"):
            z = np.load(self.scene.occupancy_npz)
            occ = z["occ"] if "occ" in z else z[list(z.keys())[0]]
            res_s = float(z["resolution"]) if "resolution" in z else res
            origin = [float(v) for v in z["origin"]] if "origin" in z else list(self.scene.bounds[:2])
            info = occupancy.save(path, occ, origin, res_s, rr, "scene", scene_npz=self.scene.occupancy_npz)
        else:
            if self.band.enabled is False and not req.get("force"):
                return {"ok": False, "error": "busy:controller_active (generation blocks the sim for seconds; "
                        "engage the band first or pass force=true)"}
            occ, origin = occupancy.generate_physx(self.scene.bounds, self.floor_z, res, zmin, zmax, log=self.log)
            info = occupancy.save(path, occ, origin, res, rr, "physx_overlap", z_band=np.array([zmin, zmax]))
        self.occ_cache[key] = info
        return dict(info)

    def op_band(self, req):
        on = bool(req.get("on", True))
        if on:
            st = self._read_state()
            self.band.engage(st["base_pos"], st["base_quat"], self.floor_z + float(req.get("z", self.a.band_z)))
        else:
            self.band.release(float(req.get("ramp_s", 0.0)))
        self._event("band", on=on, ramp_s=float(req.get("ramp_s", 0.0)))
        return {"band": on}

    def op_reset_robot(self, req):
        t = self.torch
        x, y = float(req["x"]), float(req["y"])
        yaw = float(req.get("yaw", 0.0))
        z = self.floor_z + float(req.get("z", self.a.band_z))
        dev = self.sim.device
        pose = t.tensor([[x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=t.float32, device=dev)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(t.zeros((1, 6), device=dev))
        dq = t.tensor(self.default_q[None], dtype=t.float32, device=dev)
        self.robot.write_joint_state_to_sim(dq, t.zeros_like(dq))
        self.robot.update(0.0)
        self.root_writes += 1
        if req.get("band", True):
            st = self._read_state()
            self.band.engage(st["base_pos"], st["base_quat"], z)
        self._event("reset_robot", x=x, y=y, yaw=yaw, band=self.band.enabled, root_writes=self.root_writes)
        return {"pose": {"x": x, "y": y, "z": z, "yaw": yaw}, "band": self.band.enabled,
                "root_writes": self.root_writes}

    def op_get_stats(self, req):
        return self.stats()

    def op_render_topdown(self, req):
        import shutil

        from sim_isaac.camera import render_topdown
        path = str(req.get("path") or (Path(self.a.out_dir) / f"topdown_{self.scene.house_id}.png"))
        if self.topdown_cache is None or req.get("fresh"):
            if not self.a.enable_cameras:
                return {"ok": False, "error": "rendering disabled (--camera none)"}
            if not self.band.enabled and not req.get("force") and self.topdown_cache is not None:
                return {"ok": False, "error": "busy:controller_active (engage the band or pass force=true)"}
            tmp = str(Path(self.a.out_dir) / f"_topdown_{self.scene.house_id}.png")
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            self.topdown_cache = render_topdown(self.sim, self.stage, self.scene.bounds, tmp)
        info = dict(self.topdown_cache)
        if path != info["path"]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(info["path"], path)
            info["path"] = path
        return info

    def op_get_joint_state(self, req):
        st = self._read_state()
        return {"names": self.names, "q": st["q"].tolist(), "dq": st["dq"].tolist(),
                "motor_names": jm.G1_MOTOR_JOINTS, "motor_q": st["q"][self.motor_idx].tolist(),
                "q_target": self.bridge.q_t.tolist(), "kp": self.bridge.kp.tolist(), "kd": self.bridge.kd.tolist()}

    def op_record(self, req):
        if req.get("on", True):
            self.recording = {"t_sim": [], "t_wall": [], "base_pos": [], "base_quat": [], "motor_q": [],
                              "q_target": [], "kp": [], "foot_l": [], "foot_r": [], "band": [], "lowcmd_count": []}
            self.record_path = str(req.get("path") or (Path(self.a.out_dir) / f"trace_{int(time.time())}.npz"))
            return {"recording": True, "path": self.record_path}
        rec, self.recording = self.recording, None
        if rec is None:
            return {"ok": False, "error": "not recording"}
        path = self.record_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        def _write():
            # compressing a few thousand samples takes ~0.2 s: do it off the physics thread (it stalled the loop)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, motor_names=np.array(jm.G1_MOTOR_JOINTS),
                                **{k: np.asarray(v) for k, v in rec.items()})
            os.replace(tmp, path)

        import threading
        th = threading.Thread(target=_write, name="record-writer", daemon=False)
        th.start()
        self._record_writers = [t for t in getattr(self, "_record_writers", []) if t.is_alive()] + [th]
        # "path" appears (atomic rename) once the write is done; "written": false until then
        return {"recording": False, "path": path, "samples": len(rec["t_sim"]), "written": False}

    def _record_sample(self, st, t_sim):
        r = self.recording
        b = self.bridge
        r["t_sim"].append(t_sim)
        r["t_wall"].append(time.time())
        r["base_pos"].append(st["base_pos"].copy())
        r["base_quat"].append(st["base_quat"].copy())
        r["motor_q"].append(st["q"][self.motor_idx].copy())
        r["q_target"].append(b.q_t[self.motor_idx].copy())
        r["kp"].append(b.kp[self.motor_idx].copy())
        fc = self.last_pose.get("foot_contact", {})
        r["foot_l"].append(bool(fc.get("left")))
        r["foot_r"].append(bool(fc.get("right")))
        r["band"].append(bool(self.band.enabled))
        r["lowcmd_count"].append(getattr(b, "lowcmd_count", 0))

    def op_shutdown(self, req):
        self.running = False
        return {"stopping": True}

    # ------------------------------------------------------------------ teardown
    def finish(self) -> dict:
        if self.recording is not None:  # still recording at shutdown: write what we have
            self.op_record({"on": False})
        for th in getattr(self, "_record_writers", []):
            th.join(timeout=30)
        s = self.stats()
        s["events"] = self.events[-50:]
        s["gpu_mem_mib"] = _gpu_mem_of(os.getpid())
        if self.a.stats_out:
            Path(self.a.stats_out).parent.mkdir(parents=True, exist_ok=True)
            Path(self.a.stats_out).write_text(json.dumps(s, indent=2, default=str) + "\n")
        print("WL_ISAAC_STATS " + json.dumps(s, default=str), flush=True)
        for c in (self.cam_pub, self.frames_pub):
            if c:
                c.close()
        self.gt.close()
        if hasattr(self.bridge, "close"):
            self.bridge.close()
        return s


class _NullBridge:
    """Same command arrays as G1DdsBridge (default stand pose) without DDS; for sim-only RTF runs."""

    def __init__(self, names):
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        self.q_t, self.dq_t, self.kp, self.kd, self.tau = (np.zeros(n) for _ in range(5))
        mi = np.array([idx[x] for x in jm.G1_MOTOR_JOINTS])
        self.q_t[mi] = jm.DEFAULT_ANGLES
        self.kp[mi] = jm.KPS
        self.kd[mi] = jm.KDS
        for x in jm.DEX3_LEFT_JOINTS + jm.DEX3_RIGHT_JOINTS:
            self.kp[idx[x]] = jm.DEX3_HOLD_KP
            self.kd[idx[x]] = jm.DEX3_HOLD_KD
        self.lowcmd_count = 0

    def pull_commands(self):
        return False

    def lowcmd_age_s(self):
        return None


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _gpu_mem_of(pid: int):
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            p, m = (x.strip() for x in line.split(","))
            if int(p) == pid:
                return int(m)
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    _ = launcher.app
    app = App(args)

    def _sig(signum, frame):  # noqa: ARG001
        app.running = False

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    code = 0
    try:
        app.setup()
        app.run()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    try:
        if hasattr(app, "pacer"):
            app.finish()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Kit shutdown can hang headless (eval_agent_trl.py does the same)


if __name__ == "__main__":
    main()
