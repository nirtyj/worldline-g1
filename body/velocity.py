"""Streaming body-frame velocity -> SONIC planner commands.

Used by two things:
- the Drive API op ``velocity`` (VelocityMotion): {vx, vy, wz, stream?, t_wall?, watchdog_s?, end?}; the first
  message starts the op, later messages with the same ``stream`` update it (no events, no op records);
- the Nav2 go_to backend (body/nav2_backend.py): nav2/ros_bridge.py forwards Nav2's /cmd_vel as ``velocity``
  messages carrying ``goal_id`` = the body go_to op id; those only ever feed that active go_to.

Input frame: the robot body frame at the time of the message (REP-103: x forward, y left, z up), i.e. exactly what a
ROS controller publishes on /cmd_vel for base_link.

Conversion (SonicMux does the world -> planner-frame rotation, body/frames.py):
- translation: |v| < v_deadband -> no translation. Otherwise SLOW_WALK, movement = R(yaw_gt) (vx, vy) as a world
  direction, speed = clamp(|v|, v_min 0.2, v_max 0.8), <= v_strafe_max 0.4 when |vy| > |vx| (keyboard_handler.hpp:267-272,
  keyboard.md strafe tip). A 0.05..0.2 m/s request therefore walks at 0.2 m/s (SLOW_WALK's floor); the controller
  closes the loop on the pose.
- rotation: SONIC takes a facing direction, not a yaw rate. A facing setpoint integrates wz (facing += wz dt) and is
  kept within +-lead_max of the GT yaw (anti-windup; the lead also absorbs SONIC's short IDLE turn, m1.md §1.9).
  No translation and |wz| > wz_deadband -> IDLE + facing = turn in place (keyboard Q/E, keyboard_handler.hpp:556-566).
- watchdog: no fresh message for watchdog_s (0.3 s) -> IDLE with facing = GT yaw at the trip (stop translating and
  turning). Messages whose ``t_wall`` is older than watchdog_s on arrival are dropped as stale.
"""

from __future__ import annotations

import math
import time

from .motions import Motion, MotionError, Settle, _f
from .sonic_mux import PlannerCmd
from .wire import LocomotionMode, Pose, wrap


class VelocityCommander:
    """Holds the latest body-frame twist and turns it into a SonicMux command every control tick."""

    def __init__(self, cfg, mux, owner: str, watchdog_s: float | None = None):
        self.cfg = cfg
        self.mux = mux
        self.owner = owner
        self.watchdog_s = float(watchdog_s if watchdog_s is not None else getattr(cfg, "vel_watchdog_s", 0.3))
        self.v_deadband = float(getattr(cfg, "vel_deadband", 0.05))
        self.wz_deadband = float(getattr(cfg, "vel_wz_deadband", 0.02))
        self.lead_max = math.radians(float(getattr(cfg, "facing_lead_max_deg", 25.0)))
        self.cmd = (0.0, 0.0, 0.0)
        self.t_cmd: float | None = None       # monotonic receive time of the latest accepted message
        self.t_wall_cmd: float | None = None
        self.facing: float | None = None
        self.t_apply: float | None = None
        self.state = "wait"
        self.hold_facing: float | None = None
        self.stats = {"updates": 0, "stale_dropped": 0, "watchdog_trips": 0, "walk_ticks": 0, "turn_ticks": 0,
                      "hold_ticks": 0, "speed_clamped_up": 0}
        self.t_last_trip: float | None = None

    # -- input -----------------------------------------------------------------------------------
    def update(self, args: dict) -> bool:
        """Accept one message; returns False if it was stale (dropped)."""
        vx = _f(args, "vx", 0.0, -self.cfg.v_max, self.cfg.v_max)
        vy = _f(args, "vy", 0.0, -self.cfg.v_max, self.cfg.v_max)
        wz = _f(args, "wz", 0.0, -1.5, 1.5)
        tw = args.get("t_wall")
        if tw is not None:
            try:
                age = time.time() - float(tw)
            except (TypeError, ValueError):
                age = 0.0
            if age > self.watchdog_s:
                self.stats["stale_dropped"] += 1
                return False
            self.t_wall_cmd = float(tw)
        self.cmd = (vx, vy, wz)
        self.t_cmd = time.monotonic()
        self.stats["updates"] += 1
        return True

    def fresh(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.t_cmd is not None and now - self.t_cmd <= self.watchdog_s

    # -- output (called every control tick) ------------------------------------------------------
    def apply(self, pose: Pose, now: float) -> str:
        """Set the mux command for this tick. Returns walk | turn | hold | watchdog | wait."""
        dt = 0.0 if self.t_apply is None else min(0.1, max(0.0, now - self.t_apply))
        self.t_apply = now
        if self.facing is None:
            prev = self.mux.last_facing_w
            self.facing = prev if (prev is not None and abs(wrap(prev - pose.yaw)) < math.radians(10)) else pose.yaw
        if not self.fresh(now):
            if self.t_cmd is None:
                self.state = "wait"
                self.mux.hold(self.facing, owner=self.owner)
                return "wait"
            if self.state != "watchdog":
                self.stats["watchdog_trips"] += 1
                self.t_last_trip = now
                self.hold_facing = pose.yaw
                self.facing = pose.yaw
            self.state = "watchdog"
            self.mux.hold(self.hold_facing, owner=self.owner)
            return "watchdog"
        vx, vy, wz = self.cmd
        if abs(wz) > self.wz_deadband:
            self.facing = wrap(self.facing + wz * dt)
        e = wrap(self.facing - pose.yaw)
        if abs(e) > self.lead_max:
            self.facing = wrap(pose.yaw + math.copysign(self.lead_max, e))
        v = math.hypot(vx, vy)
        if v >= self.v_deadband:
            vmax = self.cfg.v_strafe_max if abs(vy) > abs(vx) else self.cfg.v_max
            speed = min(max(v, self.cfg.v_min), vmax)
            if v < self.cfg.v_min:
                self.stats["speed_clamped_up"] += 1
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            move_w = (c * vx - s * vy, s * vx + c * vy)
            self.mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, move_w, self.facing, speed, -1.0, self.owner))
            self.state = "walk"
            self.stats["walk_ticks"] += 1
            return "walk"
        self.mux.hold(self.facing, owner=self.owner)
        if abs(wz) > self.wz_deadband:
            self.state = "turn"
            self.stats["turn_ticks"] += 1
            return "turn"
        self.state = "hold"
        self.stats["hold_ticks"] += 1
        return "hold"

    def snapshot(self) -> dict:
        age = None if self.t_cmd is None else round(time.monotonic() - self.t_cmd, 3)
        return {"state": self.state, "cmd": [round(v, 3) for v in self.cmd], "cmd_age_s": age,
                "facing_deg": None if self.facing is None else round(math.degrees(self.facing), 2),
                "watchdog_s": self.watchdog_s, **self.stats}


class VelocityMotion(Motion):
    """Drive API op ``velocity``: a stream of body-frame twists (see module docstring).

    args: vx, vy (m/s), wz (rad/s), stream (str, default = op id), t_wall (sender time.time(), optional: stale
    messages are dropped), watchdog_s (0.3), max_duration_s (3600), end (true: stop now).
    Terminal: succeeded {ended_by: watchdog|client, updates, ...} once the robot is settled after the stream stops;
    canceled if pre-empted / stopped."""

    op = "velocity"
    progress_period = 1.0

    def default_timeout(self) -> float:
        return float(self.args.get("max_duration_s", 3600.0))

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        self.stream = str(self.args.get("stream") or self.id)
        wd = _f(self.args, "watchdog_s", getattr(self.cfg, "vel_watchdog_s", 0.3), 0.05, 2.0)
        self.vel = VelocityCommander(self.cfg, self.ctx.mux, self.id, watchdog_s=wd)
        if not self.vel.update(self.args):
            raise MotionError("stale_command", {"t_wall": self.args.get("t_wall")})
        self.phase = "stream"
        self.ended_by: str | None = None
        self.settle: Settle | None = None
        if self.args.get("end"):
            self._end("client", time.monotonic())
        return {"stream": self.stream, "watchdog_s": wd}

    def update(self, args: dict) -> dict:
        if args.get("end"):
            self._end("client", time.monotonic())
            return {"ended": True}
        ok = self.vel.update(args)
        if ok and self.phase == "stopping" and self.ended_by == "watchdog":
            self.phase, self.ended_by, self.settle = "stream", None, None   # stream resumed before the op ended
        return {"accepted": ok}

    def _end(self, why: str, now: float) -> None:
        self.ended_by = why
        self.phase = "stopping"
        self.settle = Settle(self.cfg.stop_v_eps, 0.25, 2.0, wz_eps=0.3)
        if why == "client":
            self.vel.t_cmd = None   # force the IDLE hold now
            self.vel.state = "watchdog"
            self.vel.hold_facing = None

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        if self.phase == "stream":
            st = self.vel.apply(pose, now)
            if st == "watchdog":
                self._end("watchdog", now)
            return None
        # stopping: IDLE hold, succeed once settled
        if self.vel.hold_facing is None:
            self.vel.hold_facing = pose.yaw
        self.ctx.mux.hold(self.vel.hold_facing, owner=self.id)
        vx, vy, wz = self.ctx.velocity()
        if self.settle.update(math.hypot(vx, vy), wz):
            return "succeeded", {"ended_by": self.ended_by, "stream": self.stream, "velocity": self.vel.snapshot(),
                                 "stop_time_s": self.settle.stop_time, **self._summary(pose)}
        return None

    def progress(self, pose: Pose) -> dict:
        return {**super().progress(pose), "velocity": self.vel.snapshot()}

    def on_cancel(self, pose, reason: str) -> dict:
        return {**super().on_cancel(pose, reason), "velocity": self.vel.snapshot()}


def route_velocity(svc, args: dict) -> dict | None:
    """Service hook for op ``velocity``. Returns a reply for stream updates, or None to start a new VelocityMotion.

    - args.goal_id set (Nav2 /cmd_vel via nav2/ros_bridge.py): feeds the active go_to with that id, never starts a
      motion; a stale goal id is rejected WITHOUT an event (stream updates are not ops).
    - args.stream matching the active VelocityMotion: update it (no event)."""
    m = svc.active
    gid = args.get("goal_id")
    if gid is not None:
        if m is not None and m.id == str(gid) and hasattr(m, "on_velocity"):
            ok = m.on_velocity(args)
            return {"ok": bool(ok), "state": "done" if ok else "rejected",
                    **({} if ok else {"error": "stale_command"})}
        return {"ok": False, "state": "rejected", "error": "stale_goal",
                "data": {"active": None if m is None else m.id}}
    stream = args.get("stream")
    if stream is not None and isinstance(m, VelocityMotion) and m.stream == str(stream):
        return {"ok": True, "state": "done", "data": {"id": m.id, **m.update(args)}}
    return None
