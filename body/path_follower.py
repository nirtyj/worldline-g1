"""go_to: A* on the P1 occupancy grid + pure pursuit that turns pose error into SONIC planner commands.

Frames: planner `movement`/`facing` are world-frame directions of the planner's own frame (planner_onnx.md:74-77,
152-176); body/frames.py rotates world-frame directions into it. The follower therefore works entirely in the
Isaac world frame using gt.pose (5601) and never assumes the deploy knows its XY position (it does not: the planner
context is kinematic, localmotion_kplanner.hpp:591-641; position is closed here with GT).

Phases: align (turn in place toward the path if the first heading error > 50 deg) -> follow (pure pursuit,
SLOW_WALK, facing = direction to the lookahead point, speed ramps down with sqrt(2 a d)) -> settle -> approach (up to
2 short holonomic corrections at v_min with facing held) -> final_turn (optional goal yaw) -> check.
Stuck: commanded to move but < stuck_min_progress_m over stuck_window_s -> IDLE, add a virtual obstacle ahead,
re-plan (max_replans), else failed/stuck; when that obstacle leaves no route, it is dropped and the mapped route is
tried again. Cross-track > 0.6 m -> re-plan.
"""

from __future__ import annotations

import collections
import math
import time

import numpy as np

from .motions import FacingServo, Motion, MotionError, Settle, _f
from .nav_grid import NavGrid, PlanResult, path_length
from .wire import Pose, wrap


class PurePursuit:
    def __init__(self, path: np.ndarray):
        self.path = np.asarray(path, dtype=float)
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.s[-1])
        self.idx = 0

    def project(self, p: np.ndarray, window: int = 40) -> tuple[float, float]:
        """Arc length of the closest path point (searching forward from the last index) and the cross-track error."""
        best_s, best_d, best_i = 0.0, float("inf"), self.idx
        n = len(self.path)
        lo, hi = max(0, self.idx - 3), min(n - 1, self.idx + window)
        for i in range(lo, max(lo + 1, hi)):
            a = self.path[i]
            b = self.path[min(i + 1, n - 1)]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d, best_s, best_i = d, float(self.s[i] + t * math.sqrt(L2)), i
        self.idx = best_i
        return best_s, best_d

    def point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.total)
        return np.array([np.interp(s, self.s, self.path[:, 0]), np.interp(s, self.s, self.path[:, 1])])

    def tangent_at(self, s: float) -> float:
        a = self.point_at(s - 0.05)
        b = self.point_at(s + 0.05)
        return math.atan2(b[1] - a[1], b[0] - a[0])


class StuckDetector:
    def __init__(self, window_s: float, min_progress: float):
        self.window = window_s
        self.min_progress = min_progress
        self.hist: collections.deque[tuple[float, np.ndarray, bool]] = collections.deque()

    def reset(self) -> None:
        self.hist.clear()

    def update(self, now: float, xy: np.ndarray, moving_cmd: bool) -> bool:
        self.hist.append((now, xy.copy(), moving_cmd))
        while self.hist and now - self.hist[0][0] > self.window:
            self.hist.popleft()
        if not self.hist or now - self.hist[0][0] < self.window * 0.95:
            return False
        if not all(m for _, _, m in self.hist):
            return False
        return float(np.linalg.norm(self.hist[-1][1] - self.hist[0][1])) < self.min_progress


class GoToMotion(Motion):
    op = "go_to"
    progress_period = 0.5

    def default_timeout(self) -> float:
        return 180.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        a = self.args
        if "x" not in a or "y" not in a:
            raise MotionError("bad_args", {"need": "x, y"})
        self.goal = np.array([_f(a, "x"), _f(a, "y")])
        yaw = a.get("yaw")
        if yaw is None and a.get("yaw_deg") is not None:
            yaw = math.radians(float(a["yaw_deg"]))
        self.goal_yaw = None if yaw is None else wrap(float(yaw))
        self.v_req = _f(a, "speed", self.cfg.v_default, self.cfg.v_min, self.cfg.v_max)
        self.pos_tol = _f(a, "pos_tol", self.cfg.pos_tol, 0.05, 1.0)
        self.final_pos_tol = _f(a, "final_pos_tol", self.cfg.final_pos_tol, self.pos_tol, 1.0)
        self.approach_tol = _f(a, "approach_tol", self.cfg.approach_tol, 0.05, self.final_pos_tol)
        self.yaw_tol = math.radians(_f(a, "yaw_tol_deg", self.cfg.yaw_tol_deg, 1.0, 45.0))
        self.final_yaw_tol = math.radians(_f(a, "final_yaw_tol_deg", self.cfg.final_yaw_tol_deg, 1.0, 45.0))
        self.nav: NavGrid = self.ctx.get_nav()
        if self.nav is None:
            raise MotionError("no_occupancy", {"detail": "P1 get_occupancy unavailable"})
        self.nav.clear_virtual()
        self.replans = 0
        self.approach_attempts = 0
        self.stuck_events: list[dict] = []
        self.plans: list[dict] = []
        self.stuck = StuckDetector(self.cfg.stuck_window_s, self.cfg.stuck_min_progress_m)
        self.servo: FacingServo | None = None
        self._plan(pose)
        self.facing_cmd = pose.yaw
        return {"plan": self.plans[-1], "goal": [float(self.goal[0]), float(self.goal[1])],
                "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
                "speed": self.v_req}

    # -- planning ---------------------------------------------------------------------------------
    def _plan(self, pose: Pose) -> None:
        res: PlanResult = self.nav.plan((pose.x, pose.y), (float(self.goal[0]), float(self.goal[1])))
        brief = res.brief()
        brief["t"] = round(self.elapsed(), 3)
        brief["virtual_obstacles"] = [list(map(float, v)) for v in self.nav.virtual]
        self.plans.append(brief)
        if not res.ok:
            raise MotionError(res.reason or "no_path", {"plan": brief})
        if res.goal_snapped is not None:
            self.goal = np.array(res.goal_snapped, dtype=float)
        self.pp = PurePursuit(res.path)
        self.stuck.reset()
        self.phase = "align"
        self.t_phase = time.monotonic()

    def _replan(self, pose: Pose, why: str, obstacle_xy=None) -> str | None:
        self.hold(pose.yaw)
        if self.replans >= self.cfg.max_replans:
            return why
        self.replans += 1
        if obstacle_xy is not None:
            self.nav.add_virtual_obstacle(float(obstacle_xy[0]), float(obstacle_xy[1]), 0.25)
        try:
            self._plan(pose)
        except MotionError as e:
            if obstacle_xy is None or e.reason not in ("no_path", "start_in_obstacle"):
                return f"{why}_then_{e.reason}"
            # the stuck guess (an unmapped obstacle ahead) closed the last route the map has: a narrow doorway where
            # SONIC's sway rubs a door frame (H40 kitchen -> living room, 2026-09-29 m1_drive_test). Drop that guess
            # and try the mapped route again; a real blockage gets stuck again and ends `stuck` after max_replans.
            self.nav.pop_virtual()
            try:
                self._plan(pose)
            except MotionError as e2:
                return f"{why}_then_{e2.reason}"
            self.plans[-1]["virtual_dropped"] = {"at": [round(float(obstacle_xy[0]), 3),
                                                        round(float(obstacle_xy[1]), 3)], "why": e.reason}
        self.ctx.emit(self.id, "progress", {"phase": "replanned", "why": why, "plan": self.plans[-1],
                                            "replans": self.replans})
        return None

    # -- control ----------------------------------------------------------------------------------
    def progress(self, pose: Pose) -> dict:
        d = float(np.linalg.norm(self.goal - pose.xy()))
        return {"phase": self.phase, "pose": pose.brief(), "d_goal": round(d, 3), "replans": self.replans,
                "elapsed_s": round(self.elapsed(), 2)}

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        p = pose.xy()
        d_goal = float(np.linalg.norm(self.goal - p))
        vx, vy, wz = self.ctx.velocity()
        speed = math.hypot(vx, vy)

        if self.phase == "align":
            s, _ = self.pp.project(p)
            look = self.pp.point_at(s + self.cfg.lookahead_m)
            want = math.atan2(look[1] - p[1], look[0] - p[0]) if d_goal > 0.1 else pose.yaw
            err = wrap(want - pose.yaw)
            if abs(err) < math.radians(50) or d_goal < self.pos_tol or now - self.t_phase > 10.0:
                self.phase, self.t_phase = "follow", now
                self.facing_cmd = want if abs(err) < math.radians(50) else pose.yaw
            else:
                self.facing_cmd = wrap(pose.yaw + max(-math.radians(90), min(math.radians(90), err)))
                self.turn_cmd(self.facing_cmd, pose.yaw)
                if abs(err) < math.radians(20):
                    self.phase, self.t_phase = "follow", now
                return None

        if self.phase == "follow":
            s, cross = self.pp.project(p)
            remaining = max(0.0, self.pp.total - s) + float(np.linalg.norm(self.pp.path[-1] - self.pp.point_at(s)))
            if d_goal < self.pos_tol or (self.pp.total - s < 0.05 and d_goal < 2 * self.pos_tol):
                self.hold(self.facing_cmd)
                self.phase, self.settle = "settle", Settle(self.cfg.stop_v_eps, 0.3, self.cfg.settle_s + 1.0)
                return None
            if cross > 0.6:
                why = self._replan(pose, "off_path")
                if why:
                    return "failed", {"reason": why, **self._result(pose)}
                return None
            L = max(0.35, min(self.cfg.lookahead_m, remaining))
            look = self.pp.point_at(s + L)
            if np.linalg.norm(look - p) < 0.05:
                look = self.goal
            move = look - p
            heading_move = math.atan2(move[1], move[0])
            if d_goal > 0.5:
                self.facing_cmd = heading_move
            herr_s = wrap(self.facing_cmd - pose.yaw)
            herr = abs(herr_s)
            if herr > math.radians(70) and d_goal > 0.5:
                # sharp corner: stop translating and turn first (facing at most 90 deg ahead -> unambiguous direction)
                cmd = self.facing_cmd if herr <= math.radians(90) else wrap(pose.yaw + math.copysign(math.pi / 2, herr_s))
                self.turn_cmd(cmd, pose.yaw)
                self.stuck.reset()
                return None
            v = min(self.v_req, math.sqrt(2.0 * self.cfg.a_dec * max(0.0, d_goal - 0.05)) + 0.05)
            if herr > math.radians(35):
                v = self.cfg.v_min
            v = max(self.cfg.v_min, v)
            self.walk_cmd(move, self.facing_cmd, v)
            if self.stuck.update(now, p, True):
                ahead = p + 0.35 * np.array([math.cos(heading_move), math.sin(heading_move)])
                self.stuck_events.append({"t": round(self.elapsed(), 2), "at": [float(p[0]), float(p[1])]})
                self.ctx.emit(self.id, "progress", {"phase": "stuck", "at": [float(p[0]), float(p[1])]})
                why = self._replan(pose, "stuck", ahead)
                if why:
                    return "failed", {"reason": why, **self._result(pose)}
            return None

        if self.phase == "settle":
            self.hold(self.facing_cmd)
            if self.settle.update(speed, wz):
                if d_goal > self.approach_tol and self.approach_attempts < 2:
                    self.approach_attempts += 1
                    self.phase, self.t_phase = "approach", now
                    self.approach_dir = (self.goal - p) / max(d_goal, 1e-6)
                else:
                    self._to_turn_or_check(now)
            return None

        if self.phase == "approach":
            to_goal = self.goal - p
            if d_goal < 0.08 or float(to_goal @ self.approach_dir) < 0.0 or now - self.t_phase > 6.0:
                self.hold(self.facing_cmd)
                self.phase, self.settle = "settle", Settle(self.cfg.stop_v_eps, 0.3, self.cfg.settle_s + 1.0)
                return None
            self.walk_cmd(to_goal, self.facing_cmd, self.cfg.v_min)
            return None

        if self.phase == "final_turn":
            # FacingServo: SONIC's IDLE turn stops ~20 % short, so push the facing past the goal yaw by the residual
            if self.servo is None:
                self.servo = FacingServo(self.goal_yaw, push=self.cfg.turn_push)
            err = self.servo.command(self, pose, now)
            self.facing_cmd = self.servo.want()
            if abs(err) < self.yaw_tol:
                self.phase, self.settle = "final_settle", Settle(self.cfg.stop_v_eps, 0.4, 2.5, wz_eps=0.1)
            elif now - self.t_phase > 20.0:
                self.phase = "check"
            return None

        if self.phase == "final_settle":
            self.hold(self.facing_cmd)
            if self.settle.update(speed, wz):
                err = wrap(self.goal_yaw - pose.yaw)
                if abs(err) > self.final_yaw_tol and now - self.t_phase < 20.0:
                    self.phase = "final_turn"
                else:
                    self.phase = "check"
            return None

        if self.phase == "check":
            self.hold(self.facing_cmd)
            r = self._result(pose)
            ok = r["pos_err"] <= self.final_pos_tol and (self.goal_yaw is None or
                                                          abs(r["yaw_err_deg"]) <= math.degrees(self.final_yaw_tol))
            if ok:
                return "succeeded", r
            return "failed", {"reason": "final_error", **r}
        return None

    def _to_turn_or_check(self, now: float) -> None:
        if self.goal_yaw is not None:
            self.phase, self.t_phase = "final_turn", now
        else:
            self.phase = "check"

    def _result(self, pose: Pose) -> dict:
        d = float(np.linalg.norm(self.goal - pose.xy()))
        yerr = None if self.goal_yaw is None else math.degrees(wrap(self.goal_yaw - pose.yaw))
        return {"goal": [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)],
                "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
                "pos_err": round(d, 4), "yaw_err_deg": None if yerr is None else round(yerr, 2),
                "path_len_m": round(self.plans[0]["length_m"], 3) if self.plans else None,
                "replans": self.replans, "approach_attempts": self.approach_attempts,
                "servo": None if self.servo is None else self.servo.to_dict(),
                "stuck_events": self.stuck_events, "plans": self.plans, **self._summary(pose)}

    def on_cancel(self, pose: Pose | None, reason: str) -> dict:
        self.nav.clear_virtual()
        out = {"phase": self.phase, "reason": reason}
        if pose is not None:
            out.update(self._result(pose))
        return out

    def on_timeout(self, pose: Pose) -> dict:
        return self._result(pose)
