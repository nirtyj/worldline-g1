"""Arena spike policies (mounted into the Arena container at /spike, imported via PYTHONPATH=/spike).

1. VelocityCommandPolicy: drives the G1 WBC action term (g1_wbc_joint = HOMIE v2, g1_wbc_agile_joint = AGILE)
   with a scripted navigate_command / base_height schedule and NO GR00T. Upper body + hands are held at the
   articulation default joint positions. Logs root pose to /eval/<tag>_vel_log.jsonl.

2. TimedGr00tRemotePolicy: the stock Gr00tRemoteClosedloopPolicy plus instrumentation:
   - per-chunk GR00T server round-trip latency (client.get_action wall time)
   - per env-step wall time (sim speed / RTF)
   - navigate_command / base_height_command actually sent to the WBC, root position
   - per-episode outcome read from the env's own TerminationManager (success / time_out / object_dropped ...)
   Written to /eval/<tag>_timing.json, /eval/<tag>_episodes.jsonl, /eval/<tag>_cmd_log.jsonl (flushed as it goes:
   Kit exits with os._exit, so atexit handlers never run).

Both record their own videos with PyAV (the stock --video path renders /OmniverseKit_Persp, which comes out black
headless on Isaac Sim 6.0.0-dev2): an ego video (robot_head_cam_rgb, the image GR00T sees) and a third-person video
from a camera prim we add at runtime. One mp4 pair per episode in /eval/videos/<tag>/.

Env knobs: SPIKE_TAG, SPIKE_EVAL_DIR (/eval), SPIKE_CAM="ex,ey,ez,tx,ty,tz" (third-person eye/target, world frame),
SPIKE_VIDEO_EVERY (2 -> 25 fps), SPIKE_VIDEO=0 to disable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab_arena.policy.policy_base import PolicyBase

TAG = os.environ.get("SPIKE_TAG", "run")
EVAL_DIR = os.environ.get("SPIKE_EVAL_DIR", "/eval")
CONTROL_DT = 0.02  # 200 Hz physics, decimation 4 -> 50 Hz env step
VIDEO_EVERY = int(os.environ.get("SPIKE_VIDEO_EVERY", "2"))
VIDEO_ON = os.environ.get("SPIKE_VIDEO", "1") != "0"


def _argv_int(flag: str):
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except Exception:
            return None
    return None


def _to_torch(x):
    try:
        import warp as wp

        return wp.to_torch(x)
    except Exception:
        return x


def _root_pose(env) -> np.ndarray:
    return _to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0].detach().cpu().numpy()


def _yaw_from_quat(q) -> float:
    # Isaac Lab 3 root_link_pose_w = (x, y, z, qx, qy, qz, qw)?  we log raw and compute yaw for both orders
    qx, qy, qz, qw = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


class _Mp4:
    def __init__(self, path: str, w: int, h: int, fps: int):
        import av

        self.c = av.open(path, "w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"crf": "23", "preset": "veryfast"}
        self.path = path
        self.n = 0

    def write(self, rgb: np.ndarray):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[..., :3]), format="rgb24")
        for p in self.s.encode(frame):
            self.c.mux(p)
        self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


class _Recorder:
    """Ego (obs camera) + third-person (runtime camera prim) mp4 writer, one file pair per episode."""

    def __init__(self, tag: str):
        self.dir = os.path.join(EVAL_DIR, "videos", tag)
        os.makedirs(self.dir, exist_ok=True)
        self.ep = 0
        self.ego = None
        self.tp = {}
        self._ann = None
        self._tp_failed = False
        self.k = 0

    def _setup_third_person(self, env):
        """SPIKE_CAM='ex,ey,ez,tx,ty,tz[;...]' (world frame). Camera 0 feeds the video; every camera gets PNG
        snapshots at a few steps (used to pick a viewpoint that is lit / not inside a wall)."""
        try:
            import omni.replicator.core as rep
            import omni.usd
            from pxr import Gf, UsdGeom, UsdLux

            cam_env = os.environ.get("SPIKE_CAM")
            poses = []
            if cam_env:
                for c in cam_env.split(";"):
                    v = [float(x) for x in c.split(",")]
                    poses.append((v[:3], v[3:6]))
            else:
                vc = env.unwrapped.cfg.viewer
                poses.append((list(vc.eye), list(vc.lookat)))
            stage = omni.usd.get_context().get_stage()
            if not getattr(_Recorder, "_scene_dumped", False):
                _Recorder._scene_dumped = True
                lights = [str(p.GetPath()) + ":" + p.GetTypeName() for p in stage.Traverse()
                          if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName()]
                print(f"[spike] stage lights ({len(lights)}): {lights[:20]}")
            self._anns = []
            for i, (eye, tgt) in enumerate(poses):
                path = f"/World/SpikeCam{i}"
                cam = UsdGeom.Camera.Define(stage, path)
                cam.CreateFocalLengthAttr(14.0)
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
                xf = UsdGeom.Xformable(cam)
                xf.ClearXformOpOrder()
                view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
                xf.AddTransformOp().Set(view.GetInverse())
                rp = rep.create.render_product(path, (960, 540))
                ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                ann.attach([rp])
                self._anns.append(ann)
                print(f"[spike] third-person camera {i} eye={eye} target={tgt}")
            self._ann = self._anns[0]
        except Exception as e:  # never break the eval for a video
            print(f"[spike] third-person camera setup failed: {e!r}")
            self._tp_failed = True

    def _snapshots(self):
        if self.k not in (20, 150, 400) or not getattr(self, "_anns", None):
            return
        try:
            from PIL import Image

            for i, ann in enumerate(self._anns):
                d = np.asarray(ann.get_data())
                if d.size:
                    Image.fromarray(d[..., :3].astype(np.uint8)).save(
                        os.path.join(self.dir, f"cam{i}_k{self.k:04d}.png"))
        except Exception as e:
            print(f"[spike] snapshot failed {e!r}")

    def frame(self, env, observation):
        if not VIDEO_ON:
            return
        if self._ann is None and not self._tp_failed:
            self._setup_third_person(env)
        self.k += 1
        self._snapshots()
        if self.k % VIDEO_EVERY:
            return
        fps = int(round(1.0 / (CONTROL_DT * VIDEO_EVERY)))
        try:
            img = observation["camera_obs"]["robot_head_cam_rgb"][0].detach().cpu().numpy().astype(np.uint8)
            if self.ego is None:
                self.ego = _Mp4(os.path.join(self.dir, f"ep{self.ep:02d}_ego.mp4"), img.shape[1], img.shape[0], fps)
            self.ego.write(img)
        except Exception as e:
            if self.k < 4:
                print(f"[spike] ego frame failed: {e!r}")
        for i, ann in enumerate(getattr(self, "_anns", None) or []):
            try:
                d = np.asarray(ann.get_data())
                if d.size:
                    if self.tp.get(i) is None:
                        self.tp[i] = _Mp4(os.path.join(self.dir, f"ep{self.ep:02d}_3p{i}.mp4"), d.shape[1], d.shape[0], fps)
                    self.tp[i].write(d[..., :3])
            except Exception as e:
                if self.k < 4:
                    print(f"[spike] 3p frame failed: {e!r}")

    def end_episode(self):
        for v in [self.ego] + list(self.tp.values()):
            if v is not None:
                try:
                    v.close()
                    print(f"[spike] wrote {v.path} ({v.n} frames)")
                except Exception as e:
                    print(f"[spike] close failed {e!r}")
        self.ego, self.tp = None, {}
        self.ep += 1


# ----------------------------------------------------------------------------------------------------------
# 1. Pure velocity-command driving of the Arena G1 WBC (no GR00T)
# ----------------------------------------------------------------------------------------------------------
@dataclass
class VelocityCommandPolicyArgs:
    vel_schedule: str = "50:0,0,0,0.75"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "VelocityCommandPolicyArgs":
        return cls(vel_schedule=args.vel_schedule)


class VelocityCommandPolicy(PolicyBase):
    """Scripted [vx, vy, wz, base_height] schedule -> the 50-D G1 WBC action (43 joints + nav3 + h1 + rpy3)."""

    name = "spike_velocity_command"
    config_class = VelocityCommandPolicyArgs

    def __init__(self, config: VelocityCommandPolicyArgs):
        super().__init__(config)
        self.segments = []  # list of (num_steps, [vx, vy, wz, h])
        for seg in config.vel_schedule.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            n, vals = seg.split(":")
            v = [float(x) for x in vals.split(",")]
            assert len(v) == 4, f"segment needs vx,vy,wz,h: {seg}"
            self.segments.append((int(n), v))
        self.total_steps = sum(n for n, _ in self.segments)
        self.step = 0
        os.makedirs(EVAL_DIR, exist_ok=True)
        self.log_path = os.path.join(EVAL_DIR, f"{TAG}_vel_log.jsonl")
        self._log = open(self.log_path, "w")
        self._t0 = None
        self.rec = _Recorder(TAG)
        print(f"[spike] VelocityCommandPolicy: {len(self.segments)} segments, {self.total_steps} steps -> {self.log_path}")

    def _cmd_at(self, step: int):
        acc = 0
        for n, v in self.segments:
            if step < acc + n:
                return v
            acc += n
        return self.segments[-1][1]

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        self.rec.frame(env, observation)
        device = torch.device(env.unwrapped.device)
        default_q = _to_torch(env.unwrapped.scene["robot"].data.default_joint_pos)
        num_envs = env.unwrapped.num_envs
        action = torch.zeros(env.action_space.shape, device=device).reshape(num_envs, -1)
        n_joints = action.shape[1] - 7
        action[:, :n_joints] = default_q[:, :n_joints].to(device)
        vx, vy, wz, h = self._cmd_at(self.step)
        action[:, n_joints : n_joints + 3] = torch.tensor([vx, vy, wz], device=device)
        action[:, n_joints + 3] = h
        action[:, n_joints + 4 :] = 0.0  # torso rpy
        if self.step % 10 == 0:
            p = _root_pose(env)
            rec = {"step": self.step, "t_sim": round(self.step * CONTROL_DT, 3),
                   "t_wall": round(time.perf_counter() - self._t0, 3), "cmd": [vx, vy, wz, h],
                   "root_pose": [round(float(x), 4) for x in p]}
            self._log.write(json.dumps(rec) + "\n")
            self._log.flush()
        self.step += 1
        if self.step >= self.total_steps:
            self.rec.end_episode()
        return action

    def reset(self, env_ids=None):
        if self.step > 0 and env_ids is not None:
            self._log.write(json.dumps({"step": self.step, "event": "env_reset"}) + "\n")
            self._log.flush()

    def has_length(self) -> bool:
        return True

    def length(self) -> int:
        return self.total_steps

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--vel_schedule", type=str, default="50:0,0,0,0.75",
                            help="';'-separated 'steps:vx,vy,wz,height' segments (50 Hz steps)")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "VelocityCommandPolicy":
        return VelocityCommandPolicy(VelocityCommandPolicyArgs.from_cli_args(args))


# ----------------------------------------------------------------------------------------------------------
# 2. Stock remote GR00T client + instrumentation
# ----------------------------------------------------------------------------------------------------------
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (  # noqa: E402
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyArgs,
)


class _TimedClient:
    def __init__(self, client, sink: list):
        self._c = client
        self._sink = sink

    def get_action(self, *a, **k):
        t = time.perf_counter()
        out = self._c.get_action(*a, **k)
        self._sink.append(time.perf_counter() - t)
        return out

    def __getattr__(self, name):
        return getattr(self._c, name)


class TimedGr00tRemotePolicy(Gr00tRemoteClosedloopPolicy):
    name = "spike_timed_gr00t_remote"

    def __init__(self, config: Gr00tRemoteClosedloopPolicyArgs):
        super().__init__(config)
        self.chunk_lat = []
        self.step_times = []
        self._last = None
        self._client = _TimedClient(self._client, self.chunk_lat)
        self._n = 0
        self._ep_start = 0
        self._env = None
        self._max_steps = _argv_int("--num_steps")
        os.makedirs(EVAL_DIR, exist_ok=True)
        self._cmd_f = open(os.path.join(EVAL_DIR, f"{TAG}_cmd_log.jsonl"), "w")
        self._ep_f = open(os.path.join(EVAL_DIR, f"{TAG}_episodes.jsonl"), "w")
        self.rec = _Recorder(TAG)
        self._path_len = 0.0
        self._prev_xy = None
        self._start_pose = None
        self._last_pose = None

    def get_action(self, env, observation):
        if self._env is None and os.environ.get("SPIKE_RERENDER", "0") == "1":
            # Arena's galileo_g1_locomanip env (0.2.1 and main) leaves num_rerenders_on_reset=0, so the first
            # observation after an auto-reset carries the PREVIOUS episode's last camera frame and GR00T's first
            # chunk of every later episode is computed from it. The static env sets 1 for exactly this reason.
            # ManagerBasedRLEnv.step reads cfg.num_rerenders_on_reset at every reset, so setting it here works.
            env.unwrapped.cfg.num_rerenders_on_reset = 1
            print("[spike] set env.cfg.num_rerenders_on_reset = 1")
        self._env = env
        if self._n == self._ep_start:
            try:
                q = observation["policy"]["robot_joint_pos"][0].detach().cpu().numpy()
                self._ep_q0 = [round(float(x), 3) for x in q]
            except Exception as e:
                self._ep_q0 = repr(e)
        now = time.perf_counter()
        if self._last is not None:
            self.step_times.append(now - self._last)
        self._last = now
        self.rec.frame(env, observation)
        act = super().get_action(env, observation)
        p = _root_pose(env)
        self._last_pose = p.copy()
        if self._start_pose is None:
            self._start_pose = p.copy()
        if self._prev_xy is not None:
            self._path_len += float(np.linalg.norm(p[:2] - self._prev_xy))
        self._prev_xy = p[:2].copy()
        if self._n % 5 == 0:
            a = act.reshape(act.shape[0], -1)[0].detach().cpu().numpy()
            n_j = a.shape[0] - 7
            self._cmd_f.write(json.dumps({"step": self._n, "nav": [round(float(x), 3) for x in a[n_j:n_j + 3]],
                                          "h": round(float(a[n_j + 3]), 3),
                                          "root": [round(float(x), 3) for x in p]}) + "\n")
            self._cmd_f.flush()
        self._n += 1
        if self._n % 250 == 0:
            self._dump()
        if self._max_steps is not None and self._n >= self._max_steps:
            self._dump()
            self.rec.end_episode()
        return act

    def reset(self, env_ids=None):
        if env_ids is not None and self._env is not None:
            outcome = {}
            try:
                tm = self._env.unwrapped.termination_manager
                for name in tm.active_terms:
                    outcome[name] = bool(_to_torch(tm.get_term(name))[0].item())
            except Exception as e:
                outcome["error"] = repr(e)
            p = self._last_pose  # pose at the last step before the env auto-reset
            rec = {"episode": self.rec.ep, "steps": self._n - self._ep_start,
                   "sim_s": round((self._n - self._ep_start) * CONTROL_DT, 2), "terms": outcome,
                   "success": outcome.get("success"),
                   "root_path_len_m": round(self._path_len, 2),
                   "start_root": None if self._start_pose is None else [round(float(x), 3) for x in self._start_pose[:3]],
                   "end_root_before_reset": None if p is None else [round(float(x), 3) for x in p[:3]],
                   "q0_at_episode_start": getattr(self, "_ep_q0", None)}
            self._ep_f.write(json.dumps(rec) + "\n")
            self._ep_f.flush()
            print(f"[spike] episode {rec}")
            self._ep_start = self._n
            self._path_len, self._prev_xy, self._start_pose = 0.0, None, None
            self.rec.end_episode()
            self._dump()
        super().reset(env_ids)

    def _dump(self):
        lat = np.array(self.chunk_lat[1:] if len(self.chunk_lat) > 1 else self.chunk_lat)
        st = np.array(self.step_times)
        summary = {
            "tag": TAG,
            "num_chunks": len(self.chunk_lat),
            "chunk_latency_ms": {
                "first": round(1000 * self.chunk_lat[0], 1) if self.chunk_lat else None,
                "mean": round(1000 * float(lat.mean()), 1) if lat.size else None,
                "p50": round(1000 * float(np.percentile(lat, 50)), 1) if lat.size else None,
                "p95": round(1000 * float(np.percentile(lat, 95)), 1) if lat.size else None,
                "max": round(1000 * float(lat.max()), 1) if lat.size else None,
            },
            "env_steps": int(self._n),
            "step_wall_ms_mean": round(1000 * float(st.mean()), 2) if st.size else None,
            "step_wall_ms_p50": round(1000 * float(np.percentile(st, 50)), 2) if st.size else None,
            "rtf": round(CONTROL_DT / float(st.mean()), 3) if st.size else None,
            "rtf_note": "sim s per wall s over all env steps incl. GR00T calls + our video encode (control dt 0.02 s)",
        }
        with open(os.path.join(EVAL_DIR, f"{TAG}_timing.json"), "w") as f:
            json.dump(summary, f, indent=1)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "TimedGr00tRemotePolicy":
        return TimedGr00tRemotePolicy(Gr00tRemoteClosedloopPolicyArgs.from_cli_args(args))
