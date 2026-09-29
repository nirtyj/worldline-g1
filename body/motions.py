"""Motion primitives run by the body service's 50 Hz control loop.

Each motion owns the SonicMux command while it is active. tick() returns None while running, or
("succeeded"|"failed", data). The service handles pre-emption/cancel (-> "canceled"), falls, stale pose/deploy
watchdogs, and the hold (planner IDLE) after every motion. Nothing here ever sends command{stop}.

Planner usage (WBC @b042411):
- walking: mode SLOW_WALK (1) with speed 0.2..0.8 m/s (keyboard_handler.hpp:267-272), movement = world direction
  converted to the planner frame, facing decoupled from movement (planner_onnx.md:152-176) -> strafing works.
  Strafe speed capped at 0.4 m/s (docs/source/tutorials/keyboard.md tip: feet collide above ~0.4 m/s sideways).
- turning in place: mode IDLE, movement 0, new facing (keyboard Q/E do exactly this, keyboard_handler.hpp:556-566;
  planner_onnx.md:163 "falls back to facing_direction ... for in-place turning"). cfg.turn_style="slowwalk" is a
  fallback (SLOW_WALK at v_min along the facing) in case IDLE turning proves too weak in Isaac.
- stop: mode IDLE with movement 0 and facing = current GT yaw (stops translation and rotation).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from .sonic_mux import PlannerCmd
from .wire import LocomotionMode, Pose, wrap

if TYPE_CHECKING:  # pragma: no cover
    from .service import MotionCtx


class MotionError(Exception):
    def __init__(self, reason: str, data: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.data = data or {}


def _f(args: dict, key: str, default=None, lo=None, hi=None):
    v = args.get(key, default)
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        raise MotionError("bad_args", {"arg": key})
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


class Motion:
    op = "motion"
    needs_control = True          # requires the deploy to be in control (stand first)
    progress_period = 0.5

    def __init__(self, op_id: str, args: dict, ctx: "MotionCtx"):
        self.id = op_id
        self.args = args or {}
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.t0 = time.monotonic()
        self.phase = "init"
        self.pose0: Pose | None = None
        self.last_progress = 0.0
        self.path_len = 0.0
        self._last_xy: np.ndarray | None = None
        self.timeout_s = _f(self.args, "timeout_s", self.default_timeout(), 0.5, 3600.0)

    def default_timeout(self) -> float:
        return 60.0

    # hooks --------------------------------------------------------------------------------------
    def start(self, pose: Pose) -> dict:
        self.pose0 = pose
        self._last_xy = pose.xy()
        return {}

    def tick(self, pose: Pose, now: float):
        raise NotImplementedError

    def progress(self, pose: Pose) -> dict:
        return {"phase": self.phase, "pose": pose.brief(), "elapsed_s": round(time.monotonic() - self.t0, 2)}

    def on_cancel(self, pose: Pose | None, reason: str) -> dict:
        return {"phase": self.phase, "reason": reason, **self._summary(pose)}

    # helpers ------------------------------------------------------------------------------------
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def track(self, pose: Pose) -> None:
        xy = pose.xy()
        if self._last_xy is not None:
            self.path_len += float(np.linalg.norm(xy - self._last_xy))
        self._last_xy = xy

    def _summary(self, pose: Pose | None) -> dict:
        if pose is None or self.pose0 is None:
            return {}
        d = pose.xy() - self.pose0.xy()
        c, s = math.cos(self.pose0.yaw), math.sin(self.pose0.yaw)
        return {"final_pose": pose.brief(), "start_pose": self.pose0.brief(),
                "displacement_m": round(float(np.linalg.norm(d)), 4),
                "displacement_body0": [round(float(c * d[0] + s * d[1]), 4), round(float(-s * d[0] + c * d[1]), 4)],
                "yaw_change_deg": round(math.degrees(wrap(pose.yaw - self.pose0.yaw)), 2),
                "walked_m": round(self.path_len, 4), "duration_s": round(self.elapsed(), 3)}

    def hold(self, facing_w: float | None) -> None:
        self.ctx.mux.hold(facing_w, owner=self.id)

    def walk_cmd(self, move_w, facing_w: float, speed: float) -> None:
        self.ctx.mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (float(move_w[0]), float(move_w[1])), float(facing_w),
                                    float(speed), -1.0, self.id))

    # Facing changes are stepped like the upstream keyboard Q/E handler: <= 30 deg per step from the last COMMANDED
    # facing (keyboard_handler.hpp:557-566), advancing only once the body has caught up. A single large IDLE facing
    # step is planned once and never corrected (static modes replan only on change, g1_deploy_onnx_ref.cpp:3727-3731);
    # in the reference runs 90 deg steps under-rotated by ~12 deg and fell under timing jitter (docs/walk_diagnosis.md).
    TURN_STEP = math.radians(30.0)
    TURN_CATCHUP = math.radians(15.0)

    def turn_cmd(self, facing_w: float, yaw_now: float) -> None:
        prev = self.ctx.mux.last_facing_w
        if prev is None:
            prev = yaw_now
        d = wrap(facing_w - prev)
        if abs(d) > self.TURN_STEP:
            caught_up = abs(wrap(prev - yaw_now)) < self.TURN_CATCHUP
            facing_w = wrap(prev + math.copysign(self.TURN_STEP, d)) if caught_up else prev
        if self.cfg.turn_style == "slowwalk":
            self.walk_cmd((math.cos(facing_w), math.sin(facing_w)), facing_w, self.cfg.v_min)
        else:
            self.hold(facing_w)


class Settle:
    """Wait until |v| < eps and |wz| < wz_eps for hold_s (or max_s)."""

    def __init__(self, eps: float, hold_s: float = 0.3, max_s: float = 1.5, wz_eps: float = 0.15):
        self.eps, self.hold_s, self.max_s, self.wz_eps = eps, hold_s, max_s, wz_eps
        self.t0 = time.monotonic()
        self.t_ok: float | None = None
        self.t_stopped: float | None = None

    def update(self, speed: float, wz: float) -> bool:
        now = time.monotonic()
        if speed < self.eps and abs(wz) < self.wz_eps:
            if self.t_ok is None:
                self.t_ok = now
            if self.t_stopped is None and speed < self.eps:
                self.t_stopped = now
        else:
            self.t_ok = None
        return (self.t_ok is not None and now - self.t_ok >= self.hold_s) or now - self.t0 >= self.max_s

    @property
    def stop_time(self) -> float | None:
        return None if self.t_stopped is None else self.t_stopped - self.t0


# ------------------------------------------------------------------------------------------------
class StandMotion(Motion):
    """Start SONIC control (if needed), hold IDLE, release P1's elastic band, verify the robot stays up."""

    op = "stand"
    needs_control = False

    def default_timeout(self) -> float:
        return 60.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        a = self.args
        self.release_band = bool(a.get("release_band", True))
        self.band_ramp_s = _f(a, "band_ramp_s", 1.0, 0.0, 10.0)  # m1.md §1.7: ramp_s scales band gains to 0
        self.settle_s = _f(a, "settle_s", 2.0, 0.0, 30.0)
        self.verify_s = _f(a, "verify_s", 3.0, 0.0, 600.0)
        self.start_timeout = _f(a, "start_timeout_s", 12.0, 1.0, 120.0)
        # the deploy's INIT ramp takes 3 s after lowstate appears (g1_deploy_onnx_ref.cpp:181, 2777-2790) and the
        # start handler waits only 5 s for the planner motion (zmq_manager.hpp:540-560): don't start too early.
        self.min_deploy_age = _f(a, "min_deploy_age_s", 4.0, 0.0, 60.0)
        self.yaw_hold = pose.yaw
        self.zmin, self.zmax = 1e9, -1e9
        self.started_here = False
        self.start_attempts = 0
        self.band_released = False
        self.deploy_wait = _f(a, "deploy_wait_s", 20.0, 0.0, 600.0)
        if self.ctx.mux.control_started and self.ctx.deploy.in_control():
            self.phase = "settle"
        else:
            self.hold(None)
            self.phase = "wait_deploy"
        self.t_phase = time.monotonic()
        return {"already_in_control": self.phase == "settle"}

    def _send_start(self, pose: Pose) -> None:
        # a (re)start re-initialises the planner frame at the current heading: forget any old frame first so a
        # stale theta0 (e.g. from a previous deploy run) cannot turn the robot; g1_debug then supplies the new one
        self.ctx.frame.reset()
        self.ctx.frame.set_fallback(pose.yaw)
        self.hold(pose.yaw)
        self.ctx.mux.start_control()
        self.started_here = True
        self.start_attempts += 1
        self.t_start = time.monotonic()

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        if self.phase == "wait_deploy":
            # robot_config arrives every ~2 s before CONTROL; wait for it, then for the INIT ramp
            dep = self.ctx.deploy
            if dep.alive(5.0) and dep.alive_for_s() >= self.min_deploy_age:
                self.yaw_hold = pose.yaw
                self._send_start(pose)
                self.phase = "starting"
            elif now - self.t_phase > self.deploy_wait + self.min_deploy_age:
                reason = "deploy_not_running" if not dep.alive(5.0) else "deploy_not_ready"
                return "failed", {"reason": reason, "deploy": dep.snapshot()}
            return None
        if self.phase == "starting":
            self.hold(self.yaw_hold)
            if self.ctx.deploy.in_control():
                self.phase, self.t_phase = "settle", now
                self.ctx.emit(self.id, "progress", {"phase": "in_control", "frame": self.ctx.frame.to_dict(),
                                                    "t_to_control_s": round(now - self.t_start, 3)})
            elif now - self.t_start > self.start_timeout:
                if self.start_attempts < 2:
                    self._send_start(pose)
                else:
                    return "failed", {"reason": "deploy_not_in_control", "deploy": self.ctx.deploy.snapshot()}
            return None
        if self.phase == "settle":
            self.hold(self.yaw_hold)
            if now - self.t_phase >= self.settle_s:
                if self.release_band:
                    rep = self.ctx.p1_call("band", on=False, ramp_s=self.band_ramp_s)
                    if rep is None or rep.get("ok") is False:
                        return "failed", {"reason": "band_release_failed", "reply": rep}
                    self.band_released = True
                    self.ctx.emit(self.id, "progress", {"phase": "band_released", "pose": pose.brief()})
                self.phase, self.t_phase = "verify", now
            return None
        if self.phase == "verify":
            self.hold(self.yaw_hold)
            self.zmin, self.zmax = min(self.zmin, pose.pelvis_z), max(self.zmax, pose.pelvis_z)
            if not (self.cfg.pelvis_z_min <= pose.pelvis_z <= self.cfg.pelvis_z_max):
                return "failed", {"reason": "pelvis_out_of_band", "pelvis_z": pose.pelvis_z}
            if now - self.t_phase >= self.verify_s + self.band_ramp_s:
                return "succeeded", {"band_released": self.band_released, "started_here": self.started_here,
                                     "pelvis_z_min": round(self.zmin, 4), "pelvis_z_max": round(self.zmax, 4),
                                     "frame": self.ctx.frame.to_dict(), **self._summary(pose)}
        return None


# ------------------------------------------------------------------------------------------------
class WalkMotion(Motion):
    """Velocity primitive: body-frame (vx, vy) and yaw_rate for duration_s (wall clock), then stop and settle.

    The body frame is the heading at the start of the op rotated by yaw_rate * t (so yaw_rate=0 walks a straight
    line whatever small heading errors occur). |v| < 0.05 -> pure turn (IDLE + moving facing)."""

    op = "walk"

    def default_timeout(self) -> float:
        return float(self.args.get("duration_s", 5.0)) + 15.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        a = self.args
        self.vx = _f(a, "vx", 0.0, -self.cfg.v_max, self.cfg.v_max)
        self.vy = _f(a, "vy", 0.0, -self.cfg.v_max, self.cfg.v_max)
        self.yaw_rate = _f(a, "yaw_rate", 0.0, -1.5, 1.5)
        self.duration = _f(a, "duration_s", 3.0, 0.0, 120.0)
        spd = math.hypot(self.vx, self.vy)
        notes = []
        if spd >= 0.05:
            vmax = self.cfg.v_strafe_max if abs(self.vy) > abs(self.vx) else self.cfg.v_max
            spd_c = min(max(spd, self.cfg.v_min), vmax)
            if abs(spd_c - spd) > 1e-6:
                notes.append(f"speed clamped {spd:.2f}->{spd_c:.2f} (SLOW_WALK {self.cfg.v_min}..{vmax})")
            self.speed = spd_c
        else:
            self.speed = 0.0
        prev = self.ctx.mux.last_facing_w
        self.facing0 = prev if (prev is not None and abs(wrap(prev - pose.yaw)) < math.radians(10)) else pose.yaw
        self.phase = "walk"
        self.settle: Settle | None = None
        return {"speed_cmd": round(self.speed, 3), "facing0_deg": round(math.degrees(self.facing0), 2),
                "notes": notes}

    def _facing(self, t: float) -> float:
        return wrap(self.facing0 + self.yaw_rate * t)

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        t = now - self.t0
        if self.phase == "walk":
            if t >= self.duration:
                self.phase = "stopping"
                self.final_facing = self._facing(self.duration)
                self.settle = Settle(self.cfg.stop_v_eps, 0.3, 2.0)
            else:
                f = self._facing(t)
                if self.speed > 0:
                    c, s = math.cos(f), math.sin(f)
                    move = (c * self.vx - s * self.vy, s * self.vx + c * self.vy)
                    self.walk_cmd(move, f, self.speed)
                else:
                    self.turn_cmd(f, pose.yaw)
                return None
        if self.phase == "stopping":
            self.hold(self.final_facing)
            vx, vy, wz = self.ctx.velocity()
            if self.settle.update(math.hypot(vx, vy), wz):
                return "succeeded", {"speed_cmd": self.speed, "stop_time_s": self.settle.stop_time,
                                     **self._summary(pose)}
        return None


# ------------------------------------------------------------------------------------------------
class TurnToMotion(Motion):
    """Turn in place to a world yaw (rad). Facing command is at most 90 deg ahead of the current yaw so the turn
    direction is unambiguous; succeeds when |err| < tol and the robot is settled."""

    op = "turn_to"

    def default_timeout(self) -> float:
        return 20.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        if "yaw" not in self.args and "yaw_deg" not in self.args:
            raise MotionError("bad_args", {"need": "yaw (rad) or yaw_deg"})
        yaw = _f(self.args, "yaw") if "yaw" in self.args else math.radians(_f(self.args, "yaw_deg"))
        if self.args.get("relative"):
            yaw = pose.yaw + yaw
        self.target = wrap(yaw)
        self.tol = math.radians(_f(self.args, "tol_deg", self.cfg.yaw_tol_deg, 1.0, 45.0))
        self.settle: Settle | None = None
        self.phase = "turn"
        return {"target_deg": round(math.degrees(self.target), 2),
                "initial_err_deg": round(math.degrees(wrap(self.target - pose.yaw)), 2)}

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        err = wrap(self.target - pose.yaw)
        if abs(err) > math.radians(90):
            cmd = wrap(pose.yaw + math.copysign(math.radians(90), err))
        else:
            cmd = self.target
        _, _, wz = self.ctx.velocity()
        if self.phase == "turn":
            self.turn_cmd(cmd, pose.yaw)
            if abs(err) < self.tol:
                self.phase = "settle"
                self.settle = Settle(self.cfg.stop_v_eps, 0.4, 2.5, wz_eps=0.1)
            return None
        # settle: hold the target (IDLE); re-enter turn if we drift out
        self.hold(self.target)
        vx, vy, wz = self.ctx.velocity()
        if self.settle.update(math.hypot(vx, vy), wz):
            err = wrap(self.target - pose.yaw)
            if abs(err) <= max(self.tol * 1.5, math.radians(3)):
                return "succeeded", {"target_deg": round(math.degrees(self.target), 2),
                                     "yaw_err_deg": round(math.degrees(err), 2), **self._summary(pose)}
            self.phase = "turn"
        return None

    def on_timeout(self, pose: Pose) -> dict:
        return {"yaw_err_deg": round(math.degrees(wrap(self.target - pose.yaw)), 2)}


# ------------------------------------------------------------------------------------------------
class StopMotion(Motion):
    """Planner IDLE with movement 0 and facing = current yaw; succeeds once |v| < eps (reports stop_time_s)."""

    op = "stop"
    needs_control = False

    def default_timeout(self) -> float:
        return 4.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        self.yaw_hold = pose.yaw
        self.hold(self.yaw_hold)
        self.settle = Settle(self.cfg.stop_v_eps, 0.25, float(self.args.get("max_s", 3.0)), wz_eps=0.3)
        self.phase = "stopping"
        vx, vy, wz = self.ctx.velocity()
        return {"v_at_stop": round(math.hypot(vx, vy), 3)}

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        self.hold(self.yaw_hold)
        vx, vy, wz = self.ctx.velocity()
        v = math.hypot(vx, vy)
        if self.settle.update(v, wz):
            st = self.settle.stop_time
            data = {"stopped": st is not None, "stop_time_s": None if st is None else round(st, 3),
                    "v_final": round(v, 3), **self._summary(pose)}
            return ("succeeded" if st is not None else "failed"), ({**data, "reason": "not_stopped"}
                                                                    if st is None else data)
        return None


MOTIONS = {m.op: m for m in (StandMotion, WalkMotion, TurnToMotion, StopMotion)}
