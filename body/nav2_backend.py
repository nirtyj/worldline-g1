"""go_to backend selection and the Nav2 backend (env NAV_BACKEND=nav2|astar, default nav2; per call args.backend).

nav2: ROS 2 Nav2 plans (Smac 2D) and controls (MPPI Omni / RPP); the body only turns Nav2's /cmd_vel into SONIC planner
commands (body/velocity.py) and checks the result with ground truth. ROS lives in ONE separate process,
nav2/ros_bridge.py (system Python 3.12 + /opt/ros/jazzy), reached over ZMQ REQ -> REP nav_bridge (5620 + offset):

  BodyClient.go_to ─► body Nav2GoToMotion.start ──REQ goto──► ros_bridge: ComputePathToPose pre-check (no_path,
                                                             goal_in_obstacle ... come back synchronously)
                                                             then NavigateToPose (bt_navigator)
  Nav2 /cmd_vel ─► ros_bridge ──DEALER op velocity {goal_id}──► body ROUTER ─► on_velocity ─► VelocityCommander ─► SonicMux
  tick(): status poll at 4 Hz ; Nav2 succeeded ─► IDLE settle ─► GT check (final_pos_tol / final_yaw_tol, one
          re-approach) ; Nav2 failed ─► failed with the bridge's mapped reason ; cancel / pre-empt / timeout ─► bridge cancel

astar: the original body/path_follower.py GoToMotion (A* + pure pursuit), kept as the fallback. When nav2 is
selected but the bridge is unreachable or Nav2 is not active, go_to falls back to astar (cfg.nav_fallback, env
NAV_FALLBACK=1 default) and says so in the result (backend, backend_fallback).
"""

from __future__ import annotations

import math
import time

import numpy as np

from .motions import Motion, MotionError, Settle, _f
from .p1_client import P1Error, P1Rpc
from .path_follower import GoToMotion
from .velocity import VelocityCommander
from .wire import Pose, wrap


class Nav2Unavailable(RuntimeError):
    pass


class Nav2Link:
    """REQ client for nav2/ros_bridge.py (lazy pirate: a timed-out REQ socket is recreated)."""

    def __init__(self, endpoint: str, ctx=None, timeout_s: float = 1.0):
        self.endpoint = endpoint
        self.rpc = P1Rpc(endpoint, timeout_s=timeout_s, ctx=ctx)

    def call(self, op: str, timeout_s: float = 1.0, **args) -> dict:
        try:
            return self.rpc.call(op, timeout_s=timeout_s, **args)
        except P1Error as e:
            raise Nav2Unavailable(str(e)) from e

    def ready(self, timeout_s: float = 0.5) -> tuple[bool, str]:
        try:
            rep = self.call("ping", timeout_s=timeout_s)
        except Nav2Unavailable as e:
            return False, f"bridge_unreachable ({self.endpoint}): {e}"
        if not rep.get("nav2_ready"):
            return False, f"nav2_not_ready: {rep.get('detail')}"
        return True, ""

    def close(self) -> None:
        self.rpc.close()


class AStarGoToMotion(GoToMotion):
    """The original A* go_to, tagged with backend info in every result."""

    backend_info: dict = {}

    def _result(self, pose: Pose) -> dict:
        return {**super()._result(pose), "backend": "astar", **self.backend_info}


def select_motion(svc, op: str, args: dict, default_cls):
    """Pick the class for a motion op. Only go_to has backends. Returns (cls, info)."""
    if op != "go_to":
        return default_cls, {}
    want = str(args.get("backend") or getattr(svc.cfg, "nav_backend", "nav2")).lower()
    if want == "nav2":
        ok, why = svc.nav2_link().ready(0.5)
        if ok:
            return Nav2GoToMotion, {"backend": "nav2"}
        if getattr(svc.cfg, "nav_fallback", True):
            svc.log(f"[nav] NAV_BACKEND=nav2 unavailable ({why}); go_to falls back to astar")
            return AStarGoToMotion, {"backend": "astar", "backend_requested": "nav2", "backend_fallback": why}
        return Nav2GoToMotion, {"backend": "nav2"}   # will fail nav2_unavailable at start
    return AStarGoToMotion, {"backend": "astar", "backend_requested": want}


class Nav2GoToMotion(Motion):
    op = "go_to"
    backend = "nav2"
    progress_period = 0.5

    def default_timeout(self) -> float:
        return 180.0

    # -- start --------------------------------------------------------------------------------------
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
        self.speed = None if a.get("speed") is None else _f(a, "speed", None, self.cfg.v_min, self.cfg.v_max)
        self.final_pos_tol = _f(a, "final_pos_tol", self.cfg.final_pos_tol, 0.05, 1.0)
        self.final_yaw_tol = math.radians(_f(a, "final_yaw_tol_deg", self.cfg.final_yaw_tol_deg, 1.0, 45.0))
        self.max_reapproach = int(_f(a, "max_reapproach", 1, 0, 3))
        self.link: Nav2Link = self.ctx.nav2_link()
        self.vel = VelocityCommander(self.cfg, self.ctx.mux, self.id)
        self.reapproach = 0
        self.plans: list[dict] = []
        self.nav_status: dict = {}
        self.nav_state = "init"
        self.t_poll = 0.0
        self.poll_fail_since: float | None = None
        self.hold_facing: float | None = None
        self.settle: Settle | None = None
        self.t_nav_done: float | None = None
        self.escapes: list[dict] = []
        self.max_escapes = int(_f(a, "max_escapes", 2, 0, 5))
        self.nav_retries = 0
        self.max_nav_retries = int(_f(a, "max_nav_retries", 0, 0, 3))
        self.esc: dict | None = None
        rep = self._try_goal(pose)
        out = {"backend": "nav2", "goal": [float(self.goal[0]), float(self.goal[1])],
               "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
               "plan": rep.get("plan"), "nav2_goal": rep.get("goal"), "speed_limit": self.speed}
        if self.phase == "escape":
            out["escape"] = self.esc
        return out

    def _goto_call(self) -> dict:
        args = {"id": self.id, "x": float(self.goal[0]), "y": float(self.goal[1]), "yaw": self.goal_yaw,
                "speed": self.speed}
        try:
            return self.link.call("goto", timeout_s=getattr(self.cfg, "nav2_goto_timeout_s", 8.0), **args)
        except Nav2Unavailable as e:
            raise MotionError("nav2_unavailable", {"detail": str(e), "backend": "nav2"})

    def _try_goal(self, pose: Pose) -> dict:
        """Send the goal. A start inside the inscribed zone (Nav2 START_OCCUPIED) gets an escape step first
        (phase escape); anything else that fails raises MotionError (synchronous no_path, goal_in_obstacle ...)."""
        rep = self._goto_call()
        if rep.get("ok"):
            self._accepted(rep)
            self.phase = "navigate"
            return rep
        esc = rep.get("escape") or {}
        if rep.get("reason") == "start_in_obstacle" and esc.get("dir") and len(self.escapes) < self.max_escapes:
            self._begin_escape(esc, pose)
            return rep
        data = {k: rep.get(k) for k in ("error_code", "error_name", "error_msg", "detail", "plan", "escape")
                if rep.get(k) is not None}
        raise MotionError(rep.get("reason") or "nav2_rejected", {"backend": "nav2", **data})

    def _accepted(self, rep: dict) -> None:
        if rep.get("plan"):
            self.plans.append({**rep["plan"], "t": round(self.elapsed(), 3)})
        g = rep.get("goal") or {}
        if g.get("snapped"):
            self.goal = np.array([float(g["x"]), float(g["y"])])
        self.nav_state = "active"
        self.nav_status = {}
        self.poll_fail_since = None
        self.t_poll = time.monotonic()
        self.vel = VelocityCommander(self.cfg, self.ctx.mux, self.id)

    def _begin_escape(self, esc: dict, pose: Pose) -> None:
        """Humanoid replacement for Nav2's BackUp: one slow straight step (SLOW_WALK v_min, facing held) out of the
        inscribed zone, in the direction the bridge found on the map, then the goal is re-sent."""
        self.esc = {**esc, "from": [round(pose.x, 3), round(pose.y, 3)], "t": round(self.elapsed(), 2)}
        self.escapes.append(self.esc)
        self.esc_p0 = pose.xy()
        self.esc_facing = pose.yaw
        self.t_esc = time.monotonic()
        self.phase = "escape"
        self.ctx.emit(self.id, "progress", {"phase": "escape", "escape": self.esc})

    def _retry_after_failure(self, pose: Pose, reason: str):
        """Nav2 gave up mid-route. If the robot ended inside the inscribed zone, step out and re-send; a plain
        stuck is re-sent once. Returns None (continuing) or a terminal tuple."""
        if reason in ("start_in_obstacle", "stuck", "no_path") and len(self.escapes) < self.max_escapes:
            try:
                h = self.link.call("escape", timeout_s=1.0)
            except Nav2Unavailable:
                h = {}
            if h.get("needed") and h.get("dir"):
                self._begin_escape(h, pose)
                return None
        if reason == "stuck" and self.nav_retries < self.max_nav_retries:
            self.nav_retries += 1
            self.ctx.emit(self.id, "progress", {"phase": "nav2_retry", "why": reason})
            try:
                self._try_goal(pose)
            except MotionError as e:
                return "failed", {"reason": e.reason, "first_failure": reason, **e.data, **self._result(pose)}
            return None
        return "failed", {"reason": reason, **self._result(pose)}

    # -- Nav2 /cmd_vel stream (service.route_velocity) -----------------------------------------------
    def on_velocity(self, args: dict) -> bool:
        if self.phase != "navigate":
            return False
        return self.vel.update(args)

    # -- control -----------------------------------------------------------------------------------
    def _poll(self, now: float):
        if now - self.t_poll < 0.25:
            return None
        self.t_poll = now
        try:
            rep = self.link.call("status", timeout_s=0.3, id=self.id)
        except Nav2Unavailable as e:
            if self.poll_fail_since is None:
                self.poll_fail_since = now
            if now - self.poll_fail_since > 3.0:
                return "failed", {"reason": "nav2_unavailable", "detail": str(e)}
            return None
        self.poll_fail_since = None
        self.nav_status = rep
        self.nav_state = str(rep.get("state") or "unknown")
        return None

    def tick(self, pose: Pose, now: float):
        self.track(pose)
        vx, vy, wz = self.ctx.velocity()
        if self.phase == "navigate":
            self.vel.apply(pose, now)            # watchdog: Nav2 silent (replanning, recovery wait) -> IDLE
            res = self._poll(now)
            if res is not None:
                return res[0], {**res[1], **self._result(pose)}
            if self.nav_state == "succeeded":
                self.phase, self.t_nav_done = "settle", now
                self.hold_facing = self.goal_yaw if self.goal_yaw is not None else pose.yaw
                self.ctx.mux.hold(self.hold_facing, owner=self.id)
                self.settle = Settle(self.cfg.stop_v_eps, 0.3, self.cfg.settle_s + 1.0)
            elif self.nav_state in ("failed", "canceled", "unknown"):
                reason = self.nav_status.get("reason") or ("nav2_canceled" if self.nav_state == "canceled"
                                                            else "nav2_failed")
                if self.nav_state == "failed":
                    self.ctx.mux.hold(pose.yaw, owner=self.id)
                    return self._retry_after_failure(pose, reason)
                return "failed", {"reason": reason, **self._result(pose)}
            return None
        if self.phase == "escape":
            e = self.esc
            moved = float(np.linalg.norm(pose.xy() - self.esc_p0))
            if moved >= float(e.get("dist") or 0.0) + 0.03 or now - self.t_esc > float(e.get("dist") or 0.3) / 0.2 + 2.0:
                self.ctx.mux.hold(self.esc_facing, owner=self.id)
                self.phase = "escape_settle"
                self.settle = Settle(self.cfg.stop_v_eps, 0.3, 2.0)
            else:
                self.walk_cmd(e["dir"], self.esc_facing, self.cfg.v_min)
            return None
        if self.phase == "escape_settle":
            self.ctx.mux.hold(self.esc_facing, owner=self.id)
            if self.settle.update(math.hypot(vx, vy), wz):
                self.esc["moved_m"] = round(float(np.linalg.norm(pose.xy() - self.esc_p0)), 3)
                try:
                    self._try_goal(pose)
                except MotionError as err:
                    return "failed", {"reason": err.reason, "after_escape": True, **err.data, **self._result(pose)}
            return None
        if self.phase == "settle":
            self.ctx.mux.hold(self.hold_facing, owner=self.id)
            if self.settle.update(math.hypot(vx, vy), wz):
                r = self._result(pose)
                ok = r["pos_err"] <= self.final_pos_tol and (
                    self.goal_yaw is None or abs(r["yaw_err_deg"]) <= math.degrees(self.final_yaw_tol))
                if ok:
                    return "succeeded", r
                if self.reapproach < self.max_reapproach:
                    self.reapproach += 1
                    self.ctx.emit(self.id, "progress", {"phase": "reapproach", "pos_err": r["pos_err"],
                                                        "yaw_err_deg": r["yaw_err_deg"]})
                    try:
                        self._try_goal(pose)
                    except MotionError as e:
                        return "failed", {"reason": "final_error", "reapproach_error": e.reason, **r}
                    return None
                return "failed", {"reason": "final_error", **self._result(pose)}
        return None

    # -- reporting -----------------------------------------------------------------------------------
    def progress(self, pose: Pose) -> dict:
        fb = self.nav_status.get("feedback") or {}
        return {"phase": self.phase, "pose": pose.brief(), "backend": "nav2",
                "d_goal": round(float(np.linalg.norm(self.goal - pose.xy())), 3),
                "nav2_state": self.nav_state, "distance_remaining": fb.get("distance_remaining"),
                "recoveries": fb.get("number_of_recoveries"), "velocity": self.vel.snapshot(),
                "elapsed_s": round(self.elapsed(), 2)}

    def _result(self, pose: Pose) -> dict:
        d = float(np.linalg.norm(self.goal - pose.xy()))
        yerr = None if self.goal_yaw is None else math.degrees(wrap(self.goal_yaw - pose.yaw))
        fb = self.nav_status.get("feedback") or {}
        nav = {k: self.nav_status.get(k) for k in ("state", "reason", "error_code", "error_name", "error_msg",
                                                   "cmd_vel_forwarded", "cmd_vel_rejected", "duration_s")
               if self.nav_status.get(k) is not None}
        nav["feedback"] = fb
        return {"backend": "nav2", "goal": [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)],
                "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
                "pos_err": round(d, 4), "yaw_err_deg": None if yerr is None else round(yerr, 2),
                "path_len_m": self.plans[0].get("length_m") if self.plans else None,
                "replans": fb.get("number_of_recoveries", 0), "approach_attempts": self.reapproach,
                "stuck_events": [], "escapes": self.escapes, "nav2_retries": self.nav_retries,
                "plans": self.plans, "nav2": nav, "velocity": self.vel.snapshot(),
                **self._summary(pose)}

    def _cancel_nav2(self) -> dict | None:
        try:
            return self.link.call("cancel", timeout_s=1.0, id=self.id)
        except Nav2Unavailable as e:
            return {"ok": False, "error": str(e)}

    def on_cancel(self, pose: Pose | None, reason: str) -> dict:
        rep = self._cancel_nav2()
        out = {"phase": self.phase, "reason": reason, "nav2_cancel": rep}
        if pose is not None:
            out.update(self._result(pose))
        return out

    def on_timeout(self, pose: Pose) -> dict:
        rep = self._cancel_nav2()
        return {**self._result(pose), "nav2_cancel": rep}
