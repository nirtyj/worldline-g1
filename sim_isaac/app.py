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

BASE_PORTS = {"rep": 5600, "gt_pub": 5601, "frames_pub": 5602, "camera": 5565, "ego": 5566}
M1_OPS = ("ping", "get_pose", "get_scene_info", "get_occupancy", "band", "reset_robot", "get_stats", "render_topdown",
          "get_joint_state", "record", "shutdown")
# docs/contracts/p1_m2b.md
M2B_OPS = ("get_objects", "attach", "detach", "release_all", "get_cameras", "camera", "set_render_rates",
           "get_link_poses", "detections", "reset_scene", "move_object", "set_object_pose", "push_object", "get_health")


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
    ap.add_argument("--camera", default="640x480", help="WxH of the robot cameras, or 'none' (no cameras at all)")
    ap.add_argument("--camera-hz", type=float, default=30.0, help="head (5565) camera rate in sim time")
    ap.add_argument("--camera-vfov", type=float, default=45.0, help="vertical FOV of the legacy d435 camera only")
    # M2b cameras (docs/contracts/p1_m2b.md §5): head on 5565, Arena-exact ego_view on 5566 (off until enabled)
    ap.add_argument("--cameras", default="head,ego_view",
                    help="robot cameras to create: head, ego_view, d435 (comma list; the 5565 camera is always added)")
    ap.add_argument("--stream-camera", choices=["head", "d435"], default="head",
                    help="camera published on 5565: head (M2b) or d435 (M1's exact view)")
    ap.add_argument("--ego-hz", type=float, default=30.0, help="ego_view rate in sim time while enabled")
    ap.add_argument("--ego-on", action="store_true", help="enable ego_view at start (consumer 'cli'; tests only)")
    ap.add_argument("--jpeg-q", type=int, default=80)
    ap.add_argument("--cam-warmup-frames", type=int, default=4,
                    help="frames rendered but not published after a camera is (re-)enabled")
    ap.add_argument("--seg-keep-s", type=float, default=3.0,
                    help="keep the detections segmentation annotator attached this long after the last call")
    ap.add_argument("--objects-hz", type=float, default=10.0, help="gt.objects rate (sim time); 0 = off")
    ap.add_argument("--health-hz", type=float, default=1.0, help="sim.health rate (wall time); 0 = off")
    ap.add_argument("--tp-camera", action="store_true", help="third-person chase camera on PUB 5602 'frame.tp'")
    ap.add_argument("--tp-hz", type=float, default=10.0)
    from viz.isaac_cams import add_p1_args, viz_enabled  # viz hook (docs/viz.md 9.1): --viz off|min|low|high
    add_p1_args(ap)
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
    ap.add_argument("--test-ops", action="store_true",
                    help="register the TEST-ONLY fault-injection ops (sim_isaac/test_ops.py: push_robot, rtf_throttle, "
                         "spawn_box, clear_box, test_ops_status) for eval/stack_suite.py; never in the demo stack")
    ap.add_argument("--test-box-size", default="0.3,1.6,1.2", help="with --test-ops: the spawn_box box, x,y,z metres")
    AppLauncher.add_app_launcher_args(ap)
    # RTX preset: "balanced" (Isaac Lab default). "performance" renders ~1 ms faster in a house but the frames are
    # extremely noisy (Laplacian std 310 vs 2.2, even with the DL denoiser): tools/render_ab.sh, outputs/.../render_ab
    ap.set_defaults(rendering_mode="balanced")
    a = ap.parse_args()
    a.headless = True
    a.device = "cpu" if a.physx_device == "cpu" else "cuda:0"
    if a.camera != "none" or a.tp_camera or viz_enabled(a):
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

        # robot cameras (docs/contracts/p1_m2b.md §5): prims under torso_link, spawned before physics starts; the
        # render products are created after sim.reset() (CameraRig)
        self.cam_specs = self._camera_specs()
        if self.cam_specs:
            from sim_isaac.camera import spawn_spec_camera
            from sim_isaac.cameras import prim_path_of
            for spec in self.cam_specs:
                spawn_spec_camera(prim_path_of(spec), spec)

        self.test_ops = None
        if a.test_ops:        # TEST-ONLY fault injection (eval/stack_suite.py); its box must exist before reset()
            from sim_isaac.test_ops import TestOps
            self.test_ops = TestOps(self, a.test_box_size, log=self.log)
            self.test_ops.spawn_prims(sim_utils, (sx, sy, self.floor_z))

        self.sim.reset()
        self.robot.update(0.0)
        self.names = list(self.robot.joint_names)
        self.n = len(self.names)
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.wrist_id = {arm: self.robot.find_bodies(f"{arm}_wrist_yaw_link")[0][0] for arm in ("left", "right")}
        self.body_names = list(self.robot.body_names)
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

        # live objects, attach/detach, scene reset (docs/contracts/p1_m2b.md §3, §4, §8)
        from sim_isaac.objects import ObjectTracker
        scene_objs = [dict(o) for o in (self.scene.to_dict().get("objects") or [])]
        raw_objs = {str(getattr(o, "id", "")): o for o in (getattr(self.scene.raw, "objects", None) or [])}
        for o in scene_objs:        # iTHOR multi-body objects: their other prims (not in get_scene_info)
            extra = getattr(raw_objs.get(str(o.get("id"))), "extra_prims", None)
            if extra:
                o["extra_prims"] = list(extra)
        self.objects = ObjectTracker(self.stage, scene_objs, self.floor_z, device=self.sim.device, log=self.log,
                                     event=self._event)
        from sim_isaac.segment import PrimIndex
        self.prim_index = PrimIndex(scene_objs)

        # cameras / renderers
        import zmq

        from sim_isaac.camera import FramePublisher
        self.rig = None
        if self.cam_specs:
            from sim_isaac.cameras import CameraRig
            self.rig = CameraRig(self.sim, self.stage, self.cam_specs, self.ports, self.zctx,
                                 hz={"head": a.camera_hz, "d435": a.camera_hz, "ego_view": a.ego_hz},
                                 jpeg_q=a.jpeg_q, warmup_frames=a.cam_warmup_frames, seg_keep_s=a.seg_keep_s,
                                 log=self.log, event=self._event)
        self.chase = None
        if a.tp_camera:
            from sim_isaac.camera import ChaseCamera
            self.chase = ChaseCamera(self.stage, "/World/wl_chase_cam")
        if self.rig or self.chase:
            t0 = time.perf_counter()
            for _ in range(max(1, a.warmup_renders)):
                if self.chase:
                    self.chase.update_pose(st["base_pos"], st["yaw"])
                self.sim.render()
            self.warmup_s = time.perf_counter() - t0
            shapes = {n: (None if (im := c.cap.read()) is None else im.shape) for n, c in self.rig.cams.items()} \
                if self.rig else {}
            self.log(f"camera warm-up {a.warmup_renders} renders in {self.warmup_s:.1f}s; frames {shapes}")
        else:
            self.warmup_s = 0.0
        # pre-render the top-down images once while the band holds the robot (render_topdown then only copies them)
        self.topdown_cache: dict[str, dict] = {}
        if self.a.enable_cameras:
            Path(a.out_dir).mkdir(parents=True, exist_ok=True)
            for mode in ("full", "furniture"):
                try:
                    self.topdown_cache[mode] = self._render_topdown_mode(mode)
                    self.log(f"top-down ({mode}) cached: {self.topdown_cache[mode]}")
                except Exception as e:  # noqa: BLE001
                    self.log(f"top-down ({mode}) pre-render failed: {e}")
        if self.rig:
            self.rig.finish_warmup(0.0)     # ego_view off (zero cost) until a consumer enables it
            if a.ego_on and "ego_view" in self.rig.cams:
                self.rig.camera("ego_view", 0.0, on=True, consumer="cli")

        # zmq
        from sim_isaac.gt_server import GtServer

        stream = self.rig.cams.get(a.stream_camera) if self.rig else None
        self.cam_pub = stream.pub if stream else None
        self.frames_pub = FramePublisher(self.zctx, self.ports["frames_pub"], "tp", mode="multipart",
                                         topic=b"frame.tp") if self.chase else None
        for op in M1_OPS + M2B_OPS:
            self.gt.register(op, getattr(self, f"op_{op}"))
        if self.test_ops is not None:
            self.test_ops.register(self.gt)
        from viz.isaac_cams import attach_p1  # viz hook: None unless --viz is on
        self.viz = attach_p1(self)

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
        self.last_health: dict = {}
        self.health_seq = 0
        self.objects_seq = 0
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
            lt[0, self.wrist_id["left"]], lt[0, self.wrist_id["right"]],
        ]).detach().to("cpu", dtype=torch.float64).numpy()
        n = self.n
        q, dq = flat[:n], flat[n:2 * n]
        o = 2 * n
        base_pos = flat[o:o + 3]
        bq = flat[o + 3:o + 7]                       # PhysX xyzw
        base_quat = np.array([bq[3], bq[0], bq[1], bq[2]])
        v_com, ang_w = flat[o + 7:o + 10], flat[o + 10:o + 13]
        torso_pos = flat[o + 13:o + 16]
        tq = flat[o + 16:o + 20]
        torso_quat = np.array([tq[3], tq[0], tq[1], tq[2]])
        wl, wr = flat[o + 26:o + 33], flat[o + 33:o + 40]      # wrist_yaw_link poses (palms: sim_isaac.objects)
        torso_ang_w = flat[o + 20:o + 23]
        acc_w = flat[o + 23:o + 26]
        from sim_isaac.mathutil import quat_rotate
        lin_w = v_com + np.cross(ang_w, quat_rotate(base_quat, -self.pelvis_com_b))
        ddq = np.zeros(n) if self._dq_prev is None else (dq - self._dq_prev) / self.dt
        self._dq_prev = dq
        return {"q": q, "dq": dq, "ddq": ddq, "base_pos": base_pos, "base_quat": base_quat, "lin_w": lin_w,
                "ang_w": ang_w, "ang_b": quat_rotate_inverse(base_quat, ang_w), "acc_w": acc_w,
                "torso_pos": torso_pos, "torso_quat": torso_quat,
                "torso_ang_b": quat_rotate_inverse(torso_quat, torso_ang_w), "yaw": yaw_from_quat(base_quat),
                "left_wrist_pos": wl[0:3], "left_wrist_quat": np.array([wl[6], wl[3], wl[4], wl[5]]),
                "right_wrist_pos": wr[0:3], "right_wrist_quat": np.array([wr[6], wr[3], wr[4], wr[5]])}

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
            "links": self._links(st),                                       # P1.5 (docs/contracts/p1_m2b.md §6)
            "waist_q": [round(float(v), 5) for v in st["q"][self.motor_idx[12:15]]],
        }

    def _links(self, st: dict) -> dict:
        from sim_isaac.objects import palm_pose
        out = {"torso_link": {"pos": st["torso_pos"].tolist(), "quat_wxyz": st["torso_quat"].tolist()}}
        for arm in ("left", "right"):
            p, q = palm_pose(st, arm)
            out[f"{arm}_palm"] = {"pos": p.tolist(), "quat_wxyz": q.tolist()}
        return out

    def _camera_specs(self) -> list:
        """The robot cameras of --cameras / --stream-camera / --camera WxH (docs/contracts/p1_m2b.md §5.1)."""
        a = self.a
        if a.camera == "none":
            return []
        from sim_isaac import wire
        w, h = (int(v) for v in a.camera.lower().split("x"))
        names = [x.strip() for x in a.cameras.split(",") if x.strip()]
        if a.stream_camera not in names:
            names.insert(0, a.stream_camera)
        if a.stream_camera == "d435" and "head" in names:
            names.remove("head")                    # one camera per port: 5565 carries d435 instead of head
        specs = []
        for nme in names:
            if nme not in wire.CAMERA_SPECS:
                raise SystemExit(f"--cameras: unknown camera {nme!r} (have {sorted(wire.CAMERA_SPECS)})")
            spec = wire.CAMERA_SPECS[nme]
            if nme == "d435" and abs(a.camera_vfov - 45.0) > 1e-9:
                from dataclasses import replace
                ha, va = wire._vfov_apertures(a.camera_vfov, spec.width, spec.height)
                spec = replace(spec, horizontal_aperture_mm=ha, vertical_aperture_mm=va)
            if nme != "ego_view":                   # ego_view stays exactly Arena's 640x480 camera
                spec = wire.with_resolution(spec, w, h)
            specs.append(spec)
        return specs

    def _event(self, event: str, /, **kw) -> None:
        pacer = getattr(self, "pacer", None)
        ev = {"t_sim": pacer.t_sim if pacer else 0.0, "t_wall": time.time(), "event": event, **kw}
        self.events.append(ev)
        self.gt.publish("gt.event", ev)
        self.log(f"event {json.dumps(ev)}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        a = self.a
        steps_per_pose = max(1, int(round(a.physics_hz / 50.0)))
        tp_dt = 1.0 / a.tp_hz if a.tp_hz > 0 else None
        next_tp = 0.0
        obj_dt = 1.0 / a.objects_hz if a.objects_hz > 0 else None
        next_obj = 0.0
        health_dt = 1.0 / a.health_hz if a.health_hz > 0 else None
        next_health = time.perf_counter() + (health_dt or 0.0)
        next_house = time.perf_counter()
        next_stats = time.perf_counter() + a.stats_every
        deadline = time.perf_counter() + a.duration if a.duration > 0 else None
        ready = {"ports": self.ports, "house": self.scene.house_id, "spawn": self.spawn, "floor_z": self.floor_z,
                 "physx_device": a.physx_device, "pd": a.pd, "dds_domain": a.dds_domain, "dds_iface": a.dds_iface,
                 "camera": a.camera, "camera_hz": a.camera_hz, "band": self.band.enabled,
                 "cameras": self.rig.on_map() if self.rig else {}, "p1_contract": _contract(),
                 "dynamic_objects": len(self.objects.dyn_ids), "test_ops": self.test_ops is not None,
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
            if self.test_ops is not None:
                self.test_ops.pre_step(st)     # push_robot's wrench (test-only)
            self.objects.pre_step(st)          # attach 'follow': held bodies to palm x grip offset
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

            # camera(s): rendered on the sim-time grid of each camera's rate (MuJoCo IMAGE_DT, base_sim.py:624-625);
            # one render serves every due camera (and renders every enabled product: docs/viz.md §4 finding 2)
            due = self.rig.due(t_sim) if self.rig else []
            do_tp = tp_dt is not None and self.chase is not None and t_sim + 1e-9 >= next_tp
            do_cam = bool(due)
            if do_cam or do_tp:
                r0 = time.perf_counter()
                # note: every render updates all render products; toggling hydra_texture updates per frame was
                # tried and broke the chase stream, so the chase camera simply costs a second product per render
                if do_tp:
                    self.chase.update_pose(st["base_pos"], st["yaw"])
                rs = self.rig.render() if self.rig else self._plain_render()
                self.render_stats.add((time.perf_counter() - r0) * 1e3)
                self.render_rate.tick()
                if do_cam:
                    self.rig.harvest(due, st, t_sim, rs)
                    self.cam_seq += 1
                if do_tp:
                    img = self.chase.cap.read()
                    if img is not None:
                        self.frames_pub.submit(img.copy(), t_sim, self.cam_seq,
                                               {"base_pos": st["base_pos"].tolist(), "yaw": float(st["yaw"])})
                    next_tp += tp_dt
                    if next_tp < t_sim:
                        next_tp = t_sim + tp_dt

            if self.viz:  # viz hook: chase/top cameras at --viz-hz (base_quat is wxyz)
                self.viz.step(t_sim, st["base_pos"], st["base_quat"])

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
                    if self.fallen:     # M2b name (docs/contracts/p1_m2b.md §10.2)
                        self._event("robot_fell", pelvis_z=self.last_pose["pelvis_z"],
                                    tilt_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, up_z(st["base_quat"]))))), 1),
                                    base_pos=[round(float(v), 3) for v in st["base_pos"]])
                    else:
                        self._event("robot_recovered", pelvis_z=self.last_pose["pelvis_z"])
                if self.recording is not None:
                    self._record_sample(st, t_sim)
            if obj_dt is not None and t_sim + 1e-9 >= next_obj:      # gt.objects + object_fell (P1.2, P1.9)
                next_obj = max(next_obj + obj_dt, t_sim)
                self._publish_objects(t_sim)
            if self.gt.poll(2):
                st = self._read_state()  # an op may have changed the sim (reset_robot, band)
            now = time.perf_counter()
            if health_dt is not None and now >= next_health:            # sim.health (P1.9)
                next_health = now + health_dt
                self.last_health = self._health()
                self.gt.publish("sim.health", self.last_health)
            if now >= next_house:
                next_house = now + 0.2
                if self.rig:
                    self.rig.housekeeping(t_sim)

            now = time.perf_counter()
            it_ms = (now - t0) * 1e3
            if it_ms > 25.0:
                self.hitches.append({"t_sim": round(t_sim, 3), "iter_ms": round(it_ms, 1),
                                     "step_ms": round((t5 - t0) * 1e3, 1),
                                     "render_ms": round(self.render_stats.buf[-1], 1)
                                     if (do_cam or do_tp) and self.render_stats.buf else 0.0,
                                     "cams": [c.spec.name for c in due],
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
            if self.test_ops is not None:
                self.test_ops.before_wait()    # rtf_throttle, expiries (test-only)
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
            "held": self.objects.held_map(),
            "house": self.scene.house_id, "process_cpu_pct": round(100.0 * cpu_s / wall, 1) if wall > 0 else None,
            "gt_requests": self.gt.requests, "gt_slowest_ms": round(self.gt.slow_ms, 2),
            "hitches_gt25ms": self.hitch_count, "hitches_last": self.hitches[-5:],
            "gc": {"ms_total": round(self.gc_ms_total, 1), "ms_max": round(self.gc_ms_max, 1),
                   "counts": self.gc_counts, "frozen": self.a.gc_freeze},
            "step_breakdown_ms": {k: round(1e3 * v / max(1, self.prof["n"]), 3) for k, v in self.prof.items()
                                  if k != "n"},
            # M2b (docs/contracts/p1_m2b.md §10.1)
            "cameras": self.rig.stats() if self.rig else {}, "stream_camera": self.a.stream_camera,
            "render_calls": self.rig.render_calls if self.rig else None,
            "rtf_3s": _r(self.pacer.rtf(3.0)), "rtf_5s": _r(self.pacer.rtf(5.0)),
            **{k: v for k, v in self.objects.stats().items() if k != "held"},
        }
        if not self.a.no_dds:
            b = self.bridge
            s.update(b.rates())
            s.update({"lowcmd_count": b.lowcmd_count, "lowstate_pubs": b.lowstate_pubs,
                      "heartbeat_pubs": b.heartbeat_pubs, "hand_cmd_count": b.hand_cmd_count})
        return s

    # ------------------------------------------------------------------ REP ops
    def op_ping(self, req):
        from sim_isaac.wire import TOPICS
        return {"t_sim": self.pacer.t_sim, "t_wall": time.time(), "pid": os.getpid(), "house_id": self.scene.house_id,
                "band": self.band.enabled, "uptime_s": round(time.time() - self.t_start_wall, 1),
                "ops": sorted(self.gt.handlers), "p1_contract": _contract(),
                "cameras": self.rig.on_map() if self.rig else {}, "topics": TOPICS}

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

    def _render_topdown_mode(self, mode: str) -> dict:
        from sim_isaac.camera import render_topdown
        suffix = "" if mode == "full" else f"_{mode}"
        tmp = str(Path(self.a.out_dir) / f"_topdown_{self.scene.house_id}{suffix}.png")
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        hide = (self.objects.hide_paths() + ["/World/G1"]) if mode == "furniture" else None
        info = render_topdown(self.sim, self.stage, self.scene.bounds, tmp, hide_paths=hide)
        info["mode"] = mode
        info.setdefault("hidden", 0)
        return info

    def op_render_topdown(self, req):
        import shutil

        from sim_isaac.wire import OpError
        mode = str(req.get("mode") or "full")
        if mode not in ("full", "furniture"):
            raise OpError("bad_arg", f"mode must be full|furniture, got {mode!r}")
        suffix = "" if mode == "full" else f"_{mode}"
        path = str(req.get("path") or (Path(self.a.out_dir) / f"topdown_{self.scene.house_id}{suffix}.png"))
        cached = self.topdown_cache.get(mode)
        if cached is None or req.get("fresh"):
            if not self.a.enable_cameras:
                return {"ok": False, "error": "rendering disabled (--camera none)"}
            if not self.band.enabled and not req.get("force") and cached is not None:
                return {"ok": False, "error": "busy:controller_active (engage the band or pass force=true)",
                        "code": "busy:controller_active"}
            self.topdown_cache[mode] = cached = self._render_topdown_mode(mode)
        info = dict(cached)
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

    # ------------------------------------------------------------------ M2b ops (docs/contracts/p1_m2b.md)
    def _plain_render(self) -> int:
        self.sim.render()
        return 0

    def _t(self) -> float:
        return self.pacer.t_sim

    def _publish_objects(self, t_sim: float) -> None:
        step = self.pacer.n_steps
        if not self.objects.dyn_ids:
            return
        self.objects_seq += 1
        self.gt.publish("gt.objects", {"seq": self.objects_seq, "t_sim": round(t_sim, 4), "t_wall": time.time(),
                                       "objects": self.objects.dynamic_records(step)})
        for ev in self.objects.fell_events(step, t_sim):
            self._event("object_fell", **ev)

    def _health(self) -> dict:
        from sim_isaac.wire import health_level
        p = self.pacer
        r3, r5 = p.rtf(3.0), p.rtf(5.0)
        level, why = health_level(r3, r5)
        self.health_seq += 1
        return {"seq": self.health_seq, "t_sim": round(p.t_sim, 3), "t_wall": time.time(),
                "rtf_1s": _r(p.rtf(1.0)), "rtf_3s": _r(r3), "rtf_5s": _r(r5), "rtf_10s": _r(p.rtf(10.0)),
                "level": level, "level_reason": why, "physics_hz_1s": _r(p.physics_hz(1.0), 1),
                "render_hz": round(self.render_rate.rate(), 1), "step_ms_p99": self.step_stats.summary()["p99"],
                "overruns": p.overruns, "lost_s": round(p.lost_s, 3), "hitches_gt25ms": self.hitch_count,
                "heartbeat_pubs": getattr(self.bridge, "heartbeat_pubs", 0), "band": bool(self.band.enabled),
                "fallen": bool(self.fallen), "held": self.objects.held_map(),
                "cameras": {n: {"on": c["on"], "hz": c["hz"], "pub_hz": c["pub_hz"]}
                            for n, c in (self.rig.stats().items() if self.rig else [])}}

    def op_get_health(self, req):
        return dict(self.last_health) if self.last_health else self._health()

    def op_get_objects(self, req):
        objs = self.objects.get_objects(self.pacer.n_steps, req.get("ids"), bool(req.get("dynamic_only", False)))
        return {"t_sim": round(self.pacer.t_sim, 4), "t_wall": time.time(), "seq": self.pacer.n_steps,
                "pose_source": "sim", "objects": objs}

    def _need(self, req, *keys):
        from sim_isaac.wire import OpError
        miss = [k for k in keys if req.get(k) is None]
        if miss:
            raise OpError("bad_arg", f"missing {miss}")

    def op_attach(self, req):
        from sim_isaac.wire import SNAP_M
        self._need(req, "id", "arm")
        st = self._read_state()
        return self.objects.attach(st, req["id"], str(req["arm"]), str(req.get("mode") or "follow"),
                                   offset=req.get("offset"), snap=bool(req.get("snap", True)),
                                   snap_m=float(req.get("snap_m", SNAP_M)), robot_view=self.view)

    def op_detach(self, req):
        self._need(req, "id")
        return self.objects.detach(req["id"], req.get("pose"), robot_view=self.view)

    def op_release_all(self, req):
        return {"released": self.objects.release_all()}

    def _rig(self):
        from sim_isaac.wire import OpError
        if self.rig is None:
            raise OpError("unknown_camera", "no cameras (--camera none)")
        return self.rig

    def op_get_cameras(self, req):
        rig = self.rig
        return {"cameras": rig.info() if rig else [], "stream_camera": self.a.stream_camera if rig else None,
                "render_hz": round(self.render_rate.rate(), 1)}

    def op_camera(self, req):
        self._need(req, "name")
        on = req.get("on")
        return self._rig().camera(str(req["name"]), self.pacer.t_sim, on=None if on is None else bool(on),
                                  hz=req.get("hz"), consumer=req.get("consumer"), ttl_s=req.get("ttl_s"))

    def op_set_render_rates(self, req):
        return self._rig().set_render_rates(self.pacer.t_sim, req.get("head_hz"), req.get("ego_hz"))

    def op_get_link_poses(self, req):
        st = self._read_state()
        want = req.get("links") or ["torso_link", "left_palm", "right_palm"]
        links = self._links(st)
        out, unknown = {}, []
        lt = None
        for nme in want:
            if nme in links:
                out[nme] = links[nme]
            elif nme.startswith("cam:") and self.rig and self.rig.pose_of(nme[4:], st) is not None:
                p, q = self.rig.pose_of(nme[4:], st)
                out[nme] = {"pos": p.tolist(), "quat_wxyz": q.tolist()}
            elif nme in self.body_names:
                if lt is None:
                    lt = self.view.get_link_transforms()[0].detach().to("cpu", dtype=self.torch.float64).numpy()
                t = lt[self.body_names.index(nme)]
                out[nme] = {"pos": t[0:3].tolist(), "quat_wxyz": [t[6], t[3], t[4], t[5]]}
            else:
                unknown.append(nme)
        avail = self.body_names + ["left_palm", "right_palm"] + \
            ([f"cam:{n}" for n in self.rig.cams] if self.rig else [])
        return {"t_sim": round(self.pacer.t_sim, 4), "links": out, "unknown": unknown, "available": avail}

    def op_detections(self, req):
        st = self._read_state()
        ids = set(str(i) for i in req["ids"]) if req.get("ids") else None
        step = self.pacer.n_steps
        rep = self._rig().detect(str(req.get("camera") or "head"), st, self.pacer.t_sim, self.prim_index,
                                 min_px=int(req.get("min_px", 40)), max_range=req.get("max_range"), ids=ids,
                                 bbox=bool(req.get("bbox", True)), objects_pos=self.objects.centres(step),
                                 held={oid: h.arm for oid, h in self.objects.held.items()})
        self.render_rate.tick()
        return rep

    def op_reset_scene(self, req):
        from sim_isaac.wire import OpError
        t0 = time.perf_counter()
        variant = str(req.get("variant") or "default")
        if variant != "default":
            raise OpError("unknown_variant", f"{variant!r}: wave 1 has only 'default'; pass placements in `poses`")
        out = self.objects.reset(req.get("poses"))
        robot = req.get("robot")
        robot_reset = False
        if robot:
            rr = {"x": self.spawn[0], "y": self.spawn[1], "yaw": self.spawn[2]} if robot is True else dict(robot)
            rr["band"] = bool(req.get("band", True))
            self.op_reset_robot(rr)
            robot_reset = True
        try:
            from scenes.loader import sleep_house
            out["sleep"] = sleep_house(self.stage)
        except Exception as e:  # noqa: BLE001
            out["sleep"] = {"error": repr(e)}
        ms = round((time.perf_counter() - t0) * 1e3, 1)
        self._event("reset_scene", variant=variant, objects_reset=out["objects_reset"], robot_reset=robot_reset,
                    ms=ms)
        return {"variant": variant, "robot_reset": robot_reset, "ms": ms, "object_writes": self.objects.object_writes,
                "root_writes": self.root_writes, **out}

    def op_move_object(self, req):
        self._need(req, "id", "pose")
        return self.objects.move(req["id"], req["pose"], req.get("vel"), by=str(req.get("op") or "move_object"))

    def op_set_object_pose(self, req):
        return self.op_move_object(req)

    def op_push_object(self, req):
        self._need(req, "id", "vel")
        return self.objects.push(req["id"], req["vel"])

    def op_shutdown(self, req):
        self.running = False
        return {"stopping": True}

    # ------------------------------------------------------------------ teardown
    def finish(self) -> dict:
        if getattr(self, "viz", None):  # viz hook
            self.viz.close()
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
        if getattr(self, "rig", None):
            self.rig.close()
        if self.frames_pub:
            self.frames_pub.close()
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


def _contract() -> str:
    from sim_isaac.wire import CONTRACT
    return CONTRACT


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
