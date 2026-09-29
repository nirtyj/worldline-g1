"""op `approach` (M2b B.6; docs/contracts/m1.md §3.13): a short strafing reposition, 0.1-0.4 m, to a stance pose.

    {x, y, yaw | yaw_deg (optional: keep the current yaw), v (0.2), v_lat (0.3), tol: [0.05 m, 5 deg] (or pos_tol,
     yaw_tol_deg), aim_frac (0.8), max_attempts (8), max_dist (0.6), timeout_s (45)}

No A*: one straight segment on ground truth, walked with SONIC's planner in SLOW_WALK with the movement pointing at
the goal and the facing held at the goal yaw (planner_onnx.md:152-176: facing and movement are independent, so SONIC
strafes or walks backwards without turning).

What SONIC does with such commands (live, procthor-train-38, velocity-op pulses from a stand, 2026-09-29,
outputs/body_wave/approach-char-*): nothing is repeatable open loop. The first 1 cm of travel comes 0.24-2.0 s after
the command (or never); SLOW_WALK 0.2 m/s travels 0-27 cm forward/back in 1-2 s and hardly steps sideways (< 3 cm in
2 s); 0.3 m/s steps sideways 0-16 cm, 0.4 m/s 7-50 cm; after IDLE the robot keeps going -5..+36 cm (more at higher
speed). So
the motion closes the loop on ground truth every tick instead of timing pulses:

  turn (FacingServo; only if |yaw error| > turn_first_deg)
  move: SLOW_WALK re-aimed at the goal every tick, at v (forward/back) .. v_lat (sideways), until the predicted
        travel reaches the distance: progress + v_along * T_stop >= distance (v_along = the progress rate over the
        last 0.2 s, T_stop per direction and learnt), or the robot passes the goal. Two learnt terms:
        - T_stop: after IDLE SONIC keeps going for a step or two, further the faster it was going (live run
          approach-20260929-073041: IDLE at 4-5 cm of progress, then 10-20 cm more after a quick start). A move that
          really stepped (>= 4 cm) teaches T_stop = travel after IDLE / v_along at IDLE (EMA 0.5, 0..1.5 s);
        - floor: before SONIC commits a step the pelvis shifts towards the movement with both feet down (1-3 cm in
          0.3-0.6 s forward/back; once 9.5 cm sideways at 0.3 m/s); an IDLE then cancels the step and the robot
          sways back (live run approach-20260929-072150: all 4 misses were 5-7 cm residuals stopped at 2-3 cm, net
          travel 0). No IDLE before min(distance, floor) of progress; a move that nets < 1.5 cm raises its
          direction's floor 2 cm above the progress it stopped at (<= 10 cm), one that overshoots the goal by more
          than pos_tol lowers it by 2 cm. It starts at 0 (a system without the shift never raises it)
  settle (IDLE at the goal yaw, 0.5 s still) -> measure, learn -> another move while the error is above
        aim = aim_frac * pos_tol (0.8: the standing pelvis still sways about 1 cm after a settle, so an op that stops
        at 4.9 cm is a 5.x cm stance half a second later; live runs 20260929-072150/-073041/-073827 had GT p90
        5.2-5.4 cm that way), a yaw correction (FacingServo), or done.

T_stop starts from cfg.approach_op_t_stop per direction; both terms are learnt per op and kept by the service as its
running estimate for the next op (in memory: a body restart starts again from the config). succeeded: pos_err <= aim and |yaw_err| <= yaw_tol after a settle, or pos_err <= pos_tol once the attempts are
used up; failed(final_error) otherwise (the data carries the errors either way). Pose source: gt.pose.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .motions import FacingServo, Motion, MotionError, Settle, _f
from .wire import Pose, wrap

DIRS = ("fwd", "back", "left", "right")


def dir_class(body_angle: float) -> str:
    """Movement direction relative to the facing: fwd within 45 deg, back within 45 deg, else left / right."""
    a = wrap(body_angle)
    if abs(a) <= math.radians(45):
        return "fwd"
    if abs(a) >= math.radians(135):
        return "back"
    return "left" if a > 0 else "right"


class ApproachMotion(Motion):
    op = "approach"
    progress_period = 0.5

    MOVE_MAX_S = 8.0            # one move never walks longer than this
    MIN_CLEAR_GOAL = 0.12       # raw-occupancy clearance of the goal (m), when the body has the map loaded
    MIN_CLEAR_PATH = 0.05
    LEARN_MIN_PROGRESS = 0.04   # only moves that really stepped teach T_stop
    NO_STEP = 0.015             # a move that nets less than this cancelled its step: raise the floor
    FLOOR_MAX = 0.10
    V_WIN_S = 0.2               # progress-rate window for v_along

    def default_timeout(self) -> float:
        return 45.0

    def start(self, pose: Pose) -> dict:
        super().start(pose)
        a = self.args
        if "x" not in a or "y" not in a:
            raise MotionError("bad_args", {"need": "x, y (world, m); yaw or yaw_deg optional"})
        self.goal = np.array([_f(a, "x"), _f(a, "y")])
        yaw = a.get("yaw")
        if yaw is None and a.get("yaw_deg") is not None:
            yaw = math.radians(_f(a, "yaw_deg"))
        self.goal_yaw = wrap(float(yaw)) if yaw is not None else pose.yaw
        tol = a.get("tol")
        pos_tol, yaw_tol_deg = 0.05, 5.0
        if isinstance(tol, (list, tuple)) and len(tol) == 2:
            pos_tol, yaw_tol_deg = float(tol[0]), float(tol[1])
        elif tol is not None:
            raise MotionError("bad_args", {"arg": "tol", "error": "[pos_m, yaw_deg]"})
        self.pos_tol = _f(a, "pos_tol", pos_tol, 0.02, 0.5)
        self.yaw_tol = math.radians(_f(a, "yaw_tol_deg", yaw_tol_deg, 1.0, 45.0))
        self.v = _f(a, "v", getattr(self.cfg, "approach_op_v", 0.2), self.cfg.v_min, self.cfg.v_strafe_max)
        self.v_lat = _f(a, "v_lat", getattr(self.cfg, "approach_op_v_lat", 0.3), self.cfg.v_min,
                        self.cfg.v_strafe_max)
        self.max_attempts = int(_f(a, "max_attempts", 8, 1, 12))
        self.aim = self.pos_tol * _f(a, "aim_frac", 0.8, 0.3, 1.0)
        self.turn_first = math.radians(_f(a, "turn_first_deg", 15.0, 3.0, 90.0))
        max_dist = _f(a, "max_dist", getattr(self.cfg, "approach_op_max_dist", 0.6), 0.05, 1.5)
        d0 = float(np.linalg.norm(self.goal - pose.xy()))
        if d0 > max_dist:
            raise MotionError("too_far", {"distance_m": round(d0, 3), "max_dist": max_dist,
                                          "hint": "approach is for short repositions; use go_to"})
        self.map_check = self._map_check(pose)
        svc = getattr(self.ctx, "svc", None)
        est = getattr(svc, "approach_est", None) if svc is not None else None
        t0 = float(getattr(self.cfg, "approach_op_t_stop", 0.4))
        self.t_stop = {d: float((est or {}).get("t_stop", {}).get(d, t0)) for d in DIRS}
        self.floor = {d: float((est or {}).get("floor", {}).get(d, 0.0)) for d in DIRS}
        self.attempts = 0
        self.moves: list[dict] = []
        self.turns = 0
        self.servo: FacingServo | None = None
        self.settle: Settle | None = None
        self.facing_cmd = self.goal_yaw
        yerr = wrap(self.goal_yaw - pose.yaw)
        now = time.monotonic()
        if abs(yerr) > self.turn_first:
            self._to_turn(now)
        elif d0 > self.aim:
            self._begin_move(pose, now)
        else:
            self.phase, self.t_phase = "settle", now
            self.settle = Settle(self.cfg.stop_v_eps, 0.5, 3.0, wz_eps=0.1)
            self._settle_kind = "check"
        return {"goal": [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)],
                "goal_yaw_deg": round(math.degrees(self.goal_yaw), 2), "distance_m": round(d0, 3),
                "yaw_err_deg": round(math.degrees(yerr), 2), "v": self.v, "v_lat": self.v_lat,
                "tol": [self.pos_tol, round(math.degrees(self.yaw_tol), 2)],
                "t_stop": {k: round(v, 3) for k, v in self.t_stop.items()},
                "floor": {k: round(v, 3) for k, v in self.floor.items()},
                "map_check": self.map_check, "phase": self.phase}

    def _map_check(self, pose: Pose) -> dict | None:
        """Straight-segment check on the raw occupancy (no inflation: a reach stance sits close to furniture), only
        when the service already has the map (never a P1 call from here)."""
        svc = getattr(self.ctx, "svc", None)
        nav = getattr(svc, "nav", None) if svc is not None else None
        if nav is None:
            return None
        g = self.goal
        cg = float(nav.clearance(float(g[0]), float(g[1])))
        if cg < self.MIN_CLEAR_GOAL:
            raise MotionError("goal_in_obstacle", {"goal_clearance_m": round(cg, 3), "min": self.MIN_CLEAR_GOAL})
        p0 = pose.xy()
        n = max(2, int(np.linalg.norm(g - p0) / 0.02) + 1)
        cmin = min(float(nav.clearance(*(p0 + (g - p0) * t))) for t in np.linspace(0.0, 1.0, n))
        if cmin < self.MIN_CLEAR_PATH:
            raise MotionError("path_blocked", {"min_clearance_m": round(cmin, 3), "min": self.MIN_CLEAR_PATH})
        return {"goal_clearance_m": round(cg, 3), "path_min_clearance_m": round(cmin, 3)}

    # -- phases -------------------------------------------------------------------------------------
    def _to_turn(self, now: float) -> None:
        self.servo = FacingServo(self.goal_yaw, push=self.cfg.turn_push)
        self.phase, self.t_phase = "turn", now
        self.turns += 1

    def _begin_move(self, pose: Pose, now: float) -> None:
        e = self.goal - pose.xy()
        d = float(np.linalg.norm(e))
        self.u0 = e / max(d, 1e-9)
        self.dcls = dir_class(math.atan2(self.u0[1], self.u0[0]) - self.goal_yaw)
        self.p_move0 = pose.xy()
        self.d_move0 = d
        self.min_progress = min(d, self.floor[self.dcls])
        self.prog_hist: list[tuple[float, float]] = [(now, 0.0)]
        self.v_idle = 0.0
        self.t_move0 = now
        self.t_first: float | None = None
        self.t_idle: float | None = None
        self.p_idle: np.ndarray | None = None
        self.attempts += 1
        self.phase, self.t_phase = "move", now

    def _speed(self, u: np.ndarray) -> float:
        """v for forward/back, v_lat for sideways, blended by the sideways share of the movement."""
        a = math.atan2(u[1], u[0]) - self.goal_yaw
        lat = abs(math.sin(a))
        return min(self.v + (self.v_lat - self.v) * lat, self.cfg.v_strafe_max if lat > 0.7 else self.cfg.v_max)

    def _stop_move(self, pose: Pose, now: float, why: str) -> None:
        self.hold(self.facing_cmd)
        self.t_idle, self.p_idle, self.idle_why = now, pose.xy(), why
        self.phase, self.t_phase = "settle", now
        self.settle = Settle(self.cfg.stop_v_eps, 0.5, 3.0, wz_eps=0.1)
        self._settle_kind = "move"

    def _learn(self, pose: Pose) -> dict:
        """After a move has settled: the travel after IDLE along the move direction becomes that direction's stop
        distance estimate (EMA 0.5, clamped 0..0.2 m)."""
        walk_s = (self.t_idle or self.t_move0) - self.t_move0
        progress = float((pose.xy() - self.p_move0) @ self.u0)
        prog_idle = float((self.p_idle - self.p_move0) @ self.u0) if self.p_idle is not None else 0.0
        glide = progress - prog_idle
        t_sample = None
        if progress >= self.LEARN_MIN_PROGRESS and self.v_idle >= 0.05:   # it stepped and was moving at IDLE
            t_sample = min(max(glide / self.v_idle, 0.0), 1.5)
            self.t_stop[self.dcls] = 0.5 * self.t_stop[self.dcls] + 0.5 * t_sample
        fl = self.floor[self.dcls]
        if progress < self.NO_STEP and self.idle_why == "predicted":
            self.floor[self.dcls] = min(max(fl, prog_idle) + 0.02, self.FLOOR_MAX)
        elif progress - self.d_move0 > self.pos_tol and prog_idle <= fl + 0.01:
            self.floor[self.dcls] = max(fl - 0.02, 0.0)
        rec = {"attempt": self.attempts, "dir": self.dcls, "d0": round(self.d_move0, 4),
               "min_progress": round(self.min_progress, 4), "walk_s": round(walk_s, 3),
               "t_first_1cm_s": None if self.t_first is None else round(self.t_first - self.t_move0, 3),
               "progress_at_idle": round(prog_idle, 4), "v_idle": round(self.v_idle, 3),
               "progress": round(progress, 4), "glide": round(glide, 4), "idle_why": self.idle_why,
               "t_stop_sample": None if t_sample is None else round(t_sample, 3),
               "t_stop": round(self.t_stop[self.dcls], 4), "floor": round(self.floor[self.dcls], 4),
               "err_after": round(float(np.linalg.norm(self.goal - pose.xy())), 4),
               "yaw_err_after_deg": round(math.degrees(wrap(self.goal_yaw - pose.yaw)), 2)}
        self.moves.append(rec)
        return rec

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        p = pose.xy()

        if self.phase == "turn":
            err = self.servo.command(self, pose, now)
            self.facing_cmd = self.servo.want()
            if abs(err) < max(self.yaw_tol * 0.8, math.radians(2.0)) or now - self.t_phase > 12.0:
                self.hold(self.facing_cmd)
                self.phase, self.t_phase = "settle", now
                self.settle = Settle(self.cfg.stop_v_eps, 0.5, 3.0, wz_eps=0.1)
                self._settle_kind = "turn"
            return None

        if self.phase == "move":
            e = self.goal - p
            d = float(np.linalg.norm(e))
            progress = float((p - self.p_move0) @ self.u0)
            if self.t_first is None and progress > 0.01:
                self.t_first = now
            self.prog_hist.append((now, progress))
            while len(self.prog_hist) > 2 and now - self.prog_hist[1][0] >= self.V_WIN_S:
                self.prog_hist.pop(0)
            (ta, pa), (tb, pb) = self.prog_hist[0], self.prog_hist[-1]
            v_al = (pb - pa) / (tb - ta) if tb - ta > 0.05 else 0.0
            self.v_idle = v_al
            why = None
            if progress >= self.min_progress and progress + max(v_al, 0.0) * self.t_stop[self.dcls] >= self.d_move0:
                why = "predicted"
            elif float(e @ self.u0) <= 0.0:
                why = "passed"
            elif now - self.t_move0 > self.MOVE_MAX_S:
                why = "move_timeout"
            if why:
                self._stop_move(pose, now, why)
                return None
            u = e / max(d, 1e-9)
            self.walk_cmd(u, self.facing_cmd, self._speed(u))
            return None

        if self.phase == "settle":
            self.hold(self.facing_cmd)
            vx, vy, wz = self.ctx.velocity()
            if self.settle.update(math.hypot(vx, vy), wz):
                if self._settle_kind == "move":
                    rec = self._learn(pose)
                    self.ctx.emit(self.id, "progress", {"phase": "move", **rec})
                return self._decide(pose, now)
            return None
        return None

    def _decide(self, pose: Pose, now: float):
        d_goal = float(np.linalg.norm(self.goal - pose.xy()))
        yerr = wrap(self.goal_yaw - pose.yaw)
        pos_good, pos_ok, yaw_ok = d_goal <= self.aim, d_goal <= self.pos_tol, abs(yerr) <= self.yaw_tol
        if pos_good and yaw_ok:
            return "succeeded", self._result(pose)
        if abs(yerr) > self.turn_first and self.turns < 3:
            self._to_turn(now)                                  # a big yaw error first (turning moves the robot)
            return None
        if not pos_good and self.attempts < self.max_attempts:
            self._begin_move(pose, now)
            return None
        if not yaw_ok and self.turns < 3:
            self._to_turn(now)
            return None
        if pos_ok and yaw_ok:
            return "succeeded", self._result(pose)              # within tol, attempts used up
        return "failed", {"reason": "final_error", **self._result(pose)}

    def _result(self, pose: Pose) -> dict:
        d = float(np.linalg.norm(self.goal - pose.xy()))
        svc = getattr(self.ctx, "svc", None)
        if svc is not None and hasattr(svc, "approach_est"):
            svc.approach_est = {"t_stop": {k: round(v, 4) for k, v in self.t_stop.items()},
                                "floor": {k: round(v, 4) for k, v in self.floor.items()}, "t_wall": time.time()}
        return {"goal": [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)],
                "goal_yaw_deg": round(math.degrees(self.goal_yaw), 2), "pos_err": round(d, 4),
                "yaw_err_deg": round(math.degrees(wrap(self.goal_yaw - pose.yaw)), 2),
                "tol": [self.pos_tol, round(math.degrees(self.yaw_tol), 2)], "aim": round(self.aim, 4),
                "attempts": self.attempts,
                "turns": self.turns, "moves": self.moves, "t_stop": {k: round(v, 4) for k, v in self.t_stop.items()},
                "floor": {k: round(v, 4) for k, v in self.floor.items()},
                "servo": None if self.servo is None else self.servo.to_dict(), "pose_source": "gt.pose",
                **self._summary(pose)}

    def progress(self, pose: Pose) -> dict:
        return {**super().progress(pose), "d_goal": round(float(np.linalg.norm(self.goal - pose.xy())), 4),
                "yaw_err_deg": round(math.degrees(wrap(self.goal_yaw - pose.yaw)), 2), "attempts": self.attempts}

    def on_cancel(self, pose: Pose | None, reason: str) -> dict:
        out = {"phase": self.phase, "reason": reason}
        if pose is not None:
            out.update(self._result(pose))
        return out

    def on_timeout(self, pose: Pose) -> dict:
        return self._result(pose)
