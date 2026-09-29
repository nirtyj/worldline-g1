"""go_to backend selection and the Nav2 backend (env NAV_BACKEND=nav2|astar, default nav2; per call args.backend).

nav2: ROS 2 Nav2 plans (Smac 2D) and controls (RPP / MPPI Omni); the body only turns Nav2's /cmd_vel into SONIC planner
commands (body/velocity.py) and checks the result with ground truth. ROS lives in ONE separate process,
nav2/ros_bridge.py (system Python 3.12 + /opt/ros/jazzy), reached over ZMQ DEALER -> REP nav_bridge (5620 + offset):

  BodyClient.go_to ─► body Nav2GoToMotion.start ──goto──► ros_bridge: goal check (snap <= 0.5 m or goal_in_obstacle,
                                                         like A*), ComputePathToPose pre-check (no_path ...), then
                                                         NavigateToPose (bt_navigator)
  Nav2 /cmd_vel ─► ros_bridge ──DEALER op velocity {goal_id}──► body ROUTER ─► on_velocity ─► VelocityCommander ─► SonicMux
  tick(): status poll at 4 Hz ; Nav2 succeeded ─► IDLE settle ─► GT check (final_pos_tol / final_yaw_tol, one
          re-approach) ; Nav2 failed ─► escape step / one retry / failed with the bridge's mapped reason ;
          cancel / pre-empt / timeout ─► bridge cancel

The body runs ONE single-threaded 50 Hz loop (fall checks, SonicMux keepalive owner, every motion), so nothing here may
block it for long: every bridge call is asynchronous (send now, take the reply on a later tick). One go_to request
waits at most ~50 ms in total: the backend ping in select_motion (<= 50 ms; "bridge unavailable" is then cached for 3 s
and go_to falls back to A*), the pre-empted Nav2 goal's cancel reply (<= 20 ms, informational) and start() waiting for
the goto reply with what is left of the 50 ms (the reply takes 9-30 ms); anything slower is picked up by tick().

astar: the original body/path_follower.py GoToMotion (A* + pure pursuit), kept as the fallback. When nav2 is
selected but the bridge is unreachable or Nav2 is not active, go_to falls back to astar (cfg.nav_fallback, env
NAV_FALLBACK=1 default) and says so in the result (backend, backend_fallback).
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import os
import time

import numpy as np
import zmq

from .motions import Motion, MotionError, Settle, _f
from .path_follower import GoToMotion
from .velocity import VelocityCommander
from .wire import Pose, wrap

# Tunables that belong to this backend (kept here, not in body/config.py; override per call where noted).
PING_TIMEOUT_S = 0.05        # select_motion's readiness ping
DOWN_CACHE_S = 3.0           # "bridge unreachable" is remembered this long (go_to falls back to A* at once)
REQUEST_BUDGET_S = 0.05      # one go_to request (ping + pre-empted goal's cancel + start) waits at most this long
START_WAIT_S = 0.045         # start() waits at most this long (within the budget) for the goto reply; later: tick()
CANCEL_WAIT_S = 0.02         # cancel / timeout wait this long for the bridge's cancel reply (it is processed anyway)
POLL_PERIOD_S = 0.25         # status poll period while navigating
POLL_TIMEOUT_S = 0.5         # a status poll without a reply after this counts as a failed poll
UNAVAILABLE_S = 3.0          # polls failing for this long mid-goal -> failed nav2_unavailable
INFLATION_RADIUS_M = 0.45    # = nav2/params/nav2_g1.yaml inflation_radius (the bridge's escape hint uses it too)


class Nav2Unavailable(RuntimeError):
    pass


class Nav2Link:
    """Asynchronous client for nav2/ros_bridge.py's REP socket (DEALER, request ids echoed by the bridge as `rid`).

    - send() never blocks: with ZMQ_IMMEDIATE a bridge that is not connected fails at once (Nav2Unavailable);
    - take(rid) returns the reply if it has arrived (non-blocking), wait(rid, t) waits at most t;
    - a lost request just times out (no REQ state machine to wedge); ZMQ reconnects a restarted bridge by itself;
    - ready() pings with a 50 ms budget and caches "unreachable" for DOWN_CACHE_S.
    """

    def __init__(self, endpoint: str, ctx=None, timeout_s: float | None = None):
        self.endpoint = endpoint
        self.ctx = ctx or zmq.Context.instance()
        self.sock: zmq.Socket | None = None
        self.t_sock = 0.0
        self._ids = itertools.count(1)
        self._prefix = f"b{os.getpid()}-"
        self._pending: collections.OrderedDict[str, float] = collections.OrderedDict()   # rid -> t_sent (mono)
        self._replies: dict[str, tuple[float, dict]] = {}
        self._forgotten: collections.OrderedDict[str, float] = collections.OrderedDict()
        self._down: tuple[float, str] | None = None
        self.t_ready: float | None = None      # when the last ready() started (start() budgets its wait from it)
        self.stats = collections.Counter()

    def _socket(self) -> zmq.Socket:
        if self.sock is None:
            s = self.ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.IMMEDIATE, 1)      # no connection -> send fails at once instead of queueing
            s.setsockopt(zmq.SNDHWM, 32)
            s.setsockopt(zmq.RCVHWM, 256)
            s.connect(self.endpoint)
            self.sock = s
            self.t_sock = time.monotonic()
        return self.sock

    def send(self, op: str, **args) -> str:
        rid = f"{self._prefix}{next(self._ids)}"
        req = {"op": op, "rid": rid, **args, "args": args}
        raw = json.dumps(req, default=str).encode()
        s = self._socket()
        while True:
            try:
                s.send_multipart([b"", raw], zmq.NOBLOCK)
                break
            except zmq.Again as e:
                # a socket created a moment ago is still connecting (localhost: a few ms): wait <= ~30 ms for it
                if time.monotonic() - self.t_sock > 0.03:
                    self.stats["send_fail"] += 1
                    raise Nav2Unavailable(f"bridge not connected ({self.endpoint})") from e
                time.sleep(0.003)
            except zmq.ZMQError as e:
                self.stats["send_fail"] += 1
                raise Nav2Unavailable(f"send failed: {e}") from e
        self._pending[rid] = time.monotonic()
        self.stats["sent"] += 1
        return rid

    def _drain(self, timeout_ms: int = 0) -> None:
        s = self._socket()
        while s.poll(timeout_ms):
            timeout_ms = 0
            try:
                frames = s.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                rep = json.loads(frames[-1])
            except Exception:  # noqa: BLE001
                self.stats["bad_reply"] += 1
                continue
            rid = rep.get("rid") if isinstance(rep, dict) else None
            if rid is None and self._pending:            # a bridge without rid echo answers in order
                rid = next(iter(self._pending))
            self._pending.pop(rid, None)
            if rid in self._forgotten:
                self._forgotten.pop(rid, None)
                self.stats["late_dropped"] += 1
                continue
            self._replies[rid] = (time.monotonic(), rep)
            self.stats["replies"] += 1
        now = time.monotonic()
        for d in (self._replies, self._forgotten, self._pending):     # unclaimed / never answered: drop after 60 s
            for k in [k for k, v in d.items() if now - (v[0] if isinstance(v, tuple) else v) > 60.0]:
                d.pop(k, None)

    def take(self, rid: str | None) -> dict | None:
        if rid is None:
            return None
        self._drain(0)
        got = self._replies.pop(rid, None)
        return None if got is None else got[1]

    def wait(self, rid: str, timeout_s: float) -> dict | None:
        t_end = time.monotonic() + timeout_s
        while True:
            rep = self.take(rid)
            if rep is not None:
                return rep
            left = t_end - time.monotonic()
            if left <= 0:
                return None
            self._drain(max(1, int(left * 1000)))
            got = self._replies.pop(rid, None)
            if got is not None:
                return got[1]

    def forget(self, rid: str | None) -> None:
        """The caller no longer wants this reply (timed out / superseded): drop it when it arrives."""
        if rid is None:
            return
        if self._pending.pop(rid, None) is not None:
            self._forgotten[rid] = time.monotonic()
        self._replies.pop(rid, None)

    def call(self, op: str, timeout_s: float = 1.0, **args) -> dict:
        """Blocking call (tools / tests only; the body loop uses send + take)."""
        rid = self.send(op, **args)
        rep = self.wait(rid, timeout_s)
        if rep is None:
            self.forget(rid)
            self.stats["timeouts"] += 1
            raise Nav2Unavailable(f"no reply to {op} within {timeout_s:.2f} s")
        return rep

    def ready(self, timeout_s: float = PING_TIMEOUT_S) -> tuple[bool, str]:
        now = time.monotonic()
        self.t_ready = now
        if self._down is not None and now < self._down[0]:
            self.stats["ping_cached_down"] += 1
            return False, f"{self._down[1]} (cached for {DOWN_CACHE_S:.0f} s)"
        try:
            rep = self.call("ping", timeout_s=timeout_s)
        except Nav2Unavailable as e:
            why = f"bridge_unreachable ({self.endpoint}): {e}"
            self._down = (now + DOWN_CACHE_S, why)
            return False, why
        self._down = None
        if not rep.get("nav2_ready"):
            return False, f"nav2_not_ready: {rep.get('detail')}"
        return True, ""

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(0)
            self.sock = None


class AStarGoToMotion(GoToMotion):
    """The original A* go_to, tagged with backend info in every result."""

    backend_info: dict = {}

    def _result(self, pose: Pose) -> dict:
        return {**super()._result(pose), "backend": "astar", **self.backend_info}


def select_motion(svc, op: str, args: dict, default_cls):
    """Pick the class for a motion op. Only go_to has backends. Returns (cls, info). Blocks <= PING_TIMEOUT_S."""
    if op != "go_to":
        return default_cls, {}
    want = str(args.get("backend") or getattr(svc.cfg, "nav_backend", "nav2")).lower()
    if want == "nav2":
        ok, why = svc.nav2_link().ready()
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
        self.goal_req = [float(self.goal[0]), float(self.goal[1])]
        yaw = a.get("yaw")
        if yaw is None and a.get("yaw_deg") is not None:
            yaw = math.radians(float(a["yaw_deg"]))
        self.goal_yaw = None if yaw is None else wrap(float(yaw))
        self.speed = None if a.get("speed") is None else _f(a, "speed", None, self.cfg.v_min, self.cfg.v_max)
        self.final_pos_tol = _f(a, "final_pos_tol", self.cfg.final_pos_tol, 0.05, 1.0)
        self.final_yaw_tol = math.radians(_f(a, "final_yaw_tol_deg", self.cfg.final_yaw_tol_deg, 1.0, 45.0))
        self.max_reapproach = int(_f(a, "max_reapproach", 1, 0, 3))
        self.max_escapes = int(_f(a, "max_escapes", 2, 0, 5))
        self.max_nav_retries = int(_f(a, "max_nav_retries", 1, 0, 3))
        self.goto_timeout = float(getattr(self.cfg, "nav2_goto_timeout_s", 8.0))
        self.link: Nav2Link = self.ctx.nav2_link()
        self.vel = VelocityCommander(self.cfg, self.ctx.mux, self.id)
        self.reapproach = 0
        self.plans: list[dict] = []
        self.nav_status: dict = {}
        self.nav_state = "init"
        self.goto_rid: str | None = None
        self.goto_t = 0.0
        self.goto_ctx: dict = {}
        self.goto_ms: list[float] = []
        self.sends = 0
        self.poll_rid: str | None = None
        self.poll_t = 0.0
        self.t_poll = 0.0
        self.poll_fail_since: float | None = None
        self.poll_fail_detail = ""
        self.recoveries_done = 0             # Nav2 recoveries of earlier Nav2 goals of this go_to
        self.recoveries_cur = 0
        self.hold_facing: float | None = None
        self.settle: Settle | None = None
        self.t_nav_done: float | None = None
        self.escapes: list[dict] = []
        self.stuck_events: list[dict] = []
        self.nav_retries = 0
        self.esc: dict | None = None
        self._send_goto(pose, {})
        wait = START_WAIT_S
        if self.link.t_ready is not None and time.monotonic() - self.link.t_ready < 1.0:   # same request as the ping
            wait = min(wait, max(0.0, REQUEST_BUDGET_S - (time.monotonic() - self.link.t_ready)))
        rep = self.link.wait(self.goto_rid, wait) if wait > 0 else self.link.take(self.goto_rid)
        if rep is not None:
            self.goto_rid = None
            self._on_goto_reply(rep, pose)         # MotionError -> synchronous failure (no_path, goal_in_obstacle ...)
        out = {"backend": "nav2", "goal": [float(self.goal[0]), float(self.goal[1])],
               "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
               "plan": self.plans[-1] if self.plans else None,
               "nav2_goal": (rep or {}).get("goal"), "speed_limit": self.speed}
        if rep is None:
            out["goto_pending"] = True             # the reply comes on a later tick (failures then end the op)
        if self.phase == "escape":
            out["escape"] = self.esc
        return out

    def _cur_facing(self, pose: Pose) -> float:
        """The facing to hold: the last COMMANDED one (walk diagnosis: never rebuild commands from the measured yaw),
        unless nothing sensible was commanded yet."""
        f = self.vel.cmd_facing if self.vel.cmd_facing is not None else self.ctx.mux.last_facing_w
        return f if (f is not None and abs(wrap(f - pose.yaw)) < math.radians(45)) else pose.yaw

    def _send_goto(self, pose: Pose, ctx: dict) -> None:
        """Send the goal (asynchronously) and enter phase 'starting' until the reply arrives."""
        args = {"id": self.id, "x": float(self.goal[0]), "y": float(self.goal[1]), "yaw": self.goal_yaw,
                "speed": self.speed}
        self.link.forget(self.poll_rid)            # a status poll sent before this goal would describe the old one
        self.poll_rid = None
        try:
            self.goto_rid = self.link.send("goto", **args)
        except Nav2Unavailable as e:
            raise MotionError("nav2_unavailable", {"detail": str(e), "backend": "nav2", **ctx})
        self.goto_t = time.monotonic()
        self.goto_ctx = ctx
        self.sends += 1
        self.hold_facing = self._cur_facing(pose)
        self.vel.reset()                           # the old goal's /cmd_vel stream is over (stats are kept)
        self.phase = "starting"

    def _on_goto_reply(self, rep: dict, pose: Pose) -> None:
        """Handle the bridge's goto reply: accepted -> navigate; start inside the inscribed zone (Nav2
        START_OCCUPIED) -> escape step; anything else raises MotionError (no_path, goal_in_obstacle ...)."""
        self.goto_ms.append(round((time.monotonic() - self.goto_t) * 1e3, 1))
        if rep.get("ok"):
            self._accepted(rep)
            self.phase = "navigate"
            return
        esc = rep.get("escape") or {}
        if rep.get("reason") == "start_in_obstacle" and esc.get("dir") and len(self.escapes) < self.max_escapes:
            self._begin_escape(esc, pose)
            return
        data = {k: rep.get(k) for k in ("error_code", "error_name", "error_msg", "detail", "error", "plan", "escape",
                                         "goal_check") if rep.get(k) is not None}
        # no reason = the bridge itself failed on the request (its "error" says why)
        raise MotionError(rep.get("reason") or ("nav2_failed" if rep.get("error") else "nav2_rejected"),
                          {"backend": "nav2", **data})

    def _accepted(self, rep: dict) -> None:
        if rep.get("plan"):
            self.plans.append({**rep["plan"], "t": round(self.elapsed(), 3)})
        g = rep.get("goal") or {}
        if g.get("snapped"):
            self.goal = np.array([float(g["x"]), float(g["y"])])
        self.recoveries_done += self.recoveries_cur
        self.recoveries_cur = 0
        self.nav_state = "active"
        self.nav_status = {}
        self.poll_fail_since = None
        self.t_poll = time.monotonic()

    def _begin_escape(self, esc: dict, pose: Pose) -> None:
        """Humanoid replacement for Nav2's BackUp: one slow straight step (SLOW_WALK v_min, facing held) away from
        the walls, in the direction the bridge found on the map, then the goal is re-sent."""
        self.esc = {**esc, "from": [round(pose.x, 3), round(pose.y, 3)], "t": round(self.elapsed(), 2)}
        self.escapes.append(self.esc)
        self.esc_p0 = pose.xy()
        self.esc_facing = self._cur_facing(pose)
        self.t_esc = time.monotonic()
        self.phase = "escape"
        self.ctx.emit(self.id, "progress", {"phase": "escape", "escape": self.esc})

    def _retry_after_failure(self, pose: Pose, reason: str):
        """Nav2 gave up mid-route. Returns None (continuing) or a terminal tuple.

        - inside the inscribed zone (Nav2 cannot plan from here): escape step, re-send (<= max_escapes);
        - stuck (FollowPath gave up): ONE retry (max_nav_retries, default 1), preceded by an escape step when the
          clearance is below the inflation radius (0.45 m): the pose Nav2 gave up in is often hugging a wall;
        - anything else: failed with the mapped reason."""
        h = self.nav_status.get("escape") or {}
        ev = {"t": round(self.elapsed(), 2), "reason": reason, "error_name": self.nav_status.get("error_name"),
              "pos": [round(pose.x, 3), round(pose.y, 3)], "clearance": h.get("clearance"), "action": "none"}
        if reason == "stuck":
            self.stuck_events.append(ev)
        esc_ok = bool(h.get("needed") and h.get("dir")) and len(self.escapes) < self.max_escapes
        try:
            if reason in ("start_in_obstacle", "no_path", "stuck") and esc_ok and h.get("inscribed"):
                ev["action"] = "escape"
                self._begin_escape(h, pose)
                return None
            if reason == "stuck" and self.nav_retries < self.max_nav_retries:
                self.nav_retries += 1
                self.ctx.emit(self.id, "progress", {"phase": "nav2_retry", "why": reason,
                                                    "clearance": h.get("clearance"), "escape": esc_ok})
                if esc_ok:
                    ev["action"] = "escape+retry"
                    self._begin_escape(h, pose)
                else:
                    ev["action"] = "retry"
                    self._send_goto(pose, {"first_failure": reason})
                return None
        except MotionError as e:
            return "failed", {"reason": e.reason, "first_failure": reason, **e.data, **self._result(pose)}
        return "failed", {"reason": reason, **self._result(pose)}

    # -- Nav2 /cmd_vel stream (service.route_velocity) -----------------------------------------------
    def on_velocity(self, args: dict) -> bool:
        # 'starting': the bridge may forward the new goal's first /cmd_vel before its goto reply reaches us
        if self.phase not in ("navigate", "starting"):
            return False
        return self.vel.update(args)

    # -- bridge polling (never blocks) ---------------------------------------------------------------
    def _poll_failed(self, now: float, detail: str) -> None:
        if self.poll_fail_since is None:
            self.poll_fail_since = now
        self.poll_fail_detail = detail

    def _poll(self, now: float):
        if self.poll_rid is not None:
            rep = self.link.take(self.poll_rid)
            if rep is not None:
                self.poll_rid = None
                self.poll_fail_since = None
                self.nav_status = rep
                self.nav_state = str(rep.get("state") or "unknown")
                rc = (rep.get("feedback") or {}).get("number_of_recoveries")
                if rc is not None:
                    self.recoveries_cur = max(self.recoveries_cur, int(rc))
            elif now - self.poll_t > POLL_TIMEOUT_S:
                self.link.forget(self.poll_rid)
                self.poll_rid = None
                self._poll_failed(now, f"no status reply within {POLL_TIMEOUT_S} s")
        if self.poll_rid is None and now - self.t_poll >= POLL_PERIOD_S:
            self.t_poll = now
            try:
                self.poll_rid = self.link.send("status", id=self.id)
                self.poll_t = now
            except Nav2Unavailable as e:
                self._poll_failed(now, str(e))
        if self.poll_fail_since is not None and now - self.poll_fail_since > UNAVAILABLE_S:
            return "failed", {"reason": "nav2_unavailable", "detail": self.poll_fail_detail}
        return None

    # -- control -----------------------------------------------------------------------------------
    def tick(self, pose: Pose, now: float):
        self.track(pose)
        vx, vy, wz = self.ctx.velocity()
        if self.phase == "starting":
            rep = self.link.take(self.goto_rid)
            if rep is None:
                if now - self.goto_t > self.goto_timeout:
                    self.link.forget(self.goto_rid)
                    self.goto_rid = None
                    return "failed", {"reason": "nav2_unavailable", **self.goto_ctx,
                                      "detail": f"no goto reply within {self.goto_timeout:.0f} s", **self._result(pose)}
                self.ctx.mux.hold(self.hold_facing, owner=self.id)
                return None
            self.goto_rid = None
            try:
                self._on_goto_reply(rep, pose)
            except MotionError as e:
                if self.goto_ctx.get("kind") == "reapproach":
                    return "failed", {"reason": "final_error", "reapproach_error": e.reason, **e.data,
                                      **self._result(pose)}
                ctx = {k: v for k, v in self.goto_ctx.items() if k != "kind"}
                return "failed", {"reason": e.reason, **ctx, **e.data, **self._result(pose)}
            if self.phase != "navigate":
                return None
        if self.phase == "navigate":
            self.vel.apply(pose, now)            # watchdog: Nav2 silent (replanning, recovery wait) -> IDLE
            res = self._poll(now)
            if res is not None:
                return res[0], {**res[1], **self._result(pose)}
            if self.nav_state == "succeeded":
                self.phase, self.t_nav_done = "settle", now
                self.hold_facing = self.goal_yaw if self.goal_yaw is not None else self._cur_facing(pose)
                self.ctx.mux.hold(self.hold_facing, owner=self.id)
                self.settle = Settle(self.cfg.stop_v_eps, 0.3, self.cfg.settle_s + 1.0)
            elif self.nav_state in ("failed", "canceled", "unknown"):
                # canceled here = by someone other than this go_to (our own cancel ends the op first)
                reason = "nav2_canceled" if self.nav_state == "canceled" else (self.nav_status.get("reason")
                                                                              or "nav2_failed")
                if self.nav_state == "failed":
                    self.ctx.mux.hold(self._cur_facing(pose), owner=self.id)
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
                    self._send_goto(pose, {"after_escape": True})
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
                        self._send_goto(pose, {"kind": "reapproach"})
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
        """Result fields. Contract keys keep their meaning where Nav2 has one:
        replans = goal re-sends by the body (escape steps, retries, re-approaches; Nav2's own 1 Hz replanning is
        not counted), stuck_events = Nav2 'stuck' failures with the body's reaction, nav2_recoveries = Nav2 BT
        recoveries summed over the Nav2 goals of this go_to."""
        d = float(np.linalg.norm(self.goal - pose.xy()))
        yerr = None if self.goal_yaw is None else math.degrees(wrap(self.goal_yaw - pose.yaw))
        fb = self.nav_status.get("feedback") or {}
        nav = {k: self.nav_status.get(k) for k in ("state", "reason", "error_code", "error_name", "error_msg",
                                                   "cmd_vel_forwarded", "cmd_vel_rejected", "duration_s")
               if self.nav_status.get(k) is not None}
        nav["feedback"] = fb
        return {"backend": "nav2", "goal": [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)],
                "goal_requested": [round(v, 3) for v in self.goal_req],
                "goal_yaw_deg": None if self.goal_yaw is None else round(math.degrees(self.goal_yaw), 2),
                "pos_err": round(d, 4), "yaw_err_deg": None if yerr is None else round(yerr, 2),
                "path_len_m": self.plans[0].get("length_m") if self.plans else None,
                "replans": max(0, self.sends - 1), "approach_attempts": self.reapproach,
                "stuck_events": self.stuck_events, "escapes": self.escapes, "nav2_retries": self.nav_retries,
                "nav2_recoveries": self.recoveries_done + self.recoveries_cur, "goto_reply_ms": self.goto_ms,
                "plans": self.plans, "nav2": nav, "velocity": self.vel.snapshot(),
                **self._summary(pose)}

    def _cancel_nav2(self) -> dict:
        """Cancel the Nav2 goal. The bridge stops forwarding /cmd_vel for it as soon as it reads the request; we wait
        <= CANCEL_WAIT_S for the reply only to report it. A goto still in flight is processed before the cancel
        (one DEALER -> REP connection keeps the order)."""
        self.link.forget(self.poll_rid)
        self.poll_rid = None
        try:
            rid = self.link.send("cancel", id=self.id)
        except Nav2Unavailable as e:
            return {"ok": False, "error": str(e)}
        rep = self.link.wait(rid, CANCEL_WAIT_S)
        if self.goto_rid is not None:
            self.link.forget(self.goto_rid)
            self.goto_rid = None
        if rep is None:
            self.link.forget(rid)
            return {"ok": None, "sent": True, "detail": f"no reply within {CANCEL_WAIT_S * 1e3:.0f} ms "
                    "(the bridge still processes it)"}
        return rep

    def on_cancel(self, pose: Pose | None, reason: str) -> dict:
        rep = self._cancel_nav2()
        out = {"phase": self.phase, "reason": reason, "nav2_cancel": rep}
        if pose is not None:
            out.update(self._result(pose))
        return out

    def on_timeout(self, pose: Pose) -> dict:
        rep = self._cancel_nav2()
        return {**self._result(pose), "nav2_cancel": rep}
