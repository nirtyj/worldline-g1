"""NavigationService (doc §38; PLAN §6.3): keypoint -> pose -> body go_to, with result envelopes, cancel / halt
mapping and timeouts scaled for humanoid walking.

    navigate(location)      resolve aliases (user, rooms) -> the keypoint's stand pose (x, y, yaw) -> BodyPort.go_to
                            (SONIC through wl-body, or the lite body) -> NavigateResult in a ToolResult
    reposition(stance)      navigate(location="reach_stance"): a short go_to to the stance the last
                            check_reachability returned (<= approach_max_m, straight-line free), tight tolerance
    list_locations          services/locations.py (walking distance from the current GT pose)
    timeout_s               clamp(1.8 * path_m / v + 12, 20, 240) with v = the profile's effective walking speed
    at()                    keypoint within 0.30 m of the GT pose, else between [last keypoint, target]

Body terminal reasons (docs/contracts/m1.md §3.4) map onto the envelope's reason codes (api/reasons.py NAVIGATION):
see BODY_REASON. A cancel stops the body (planner IDLE) and waits for its settle; a halt (the bridge latched it)
ends running navigation `failed(halted)`; a walk that got stuck after the body's own replans is `failed(blocked)`
with `blocked_edge = [from, to]` (THOR never produced it; agent/state.apply_navigate consumes it).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, fields
from typing import Any

from api.execution import Execution, ResultHandle
from api.results import NamedLocation, NavigateResult, finish
from api.types import Pose2D, ServiceHealth
from world import coords

from .common import EventSink, HaltGate, start_execution
from .locations import LocationsService

BODY_REASON = {
    "no_path": "no_path", "goal_in_obstacle": "no_path", "start_in_obstacle": "no_path",
    "stuck": "blocked", "off_path": "blocked",
    "timeout": "timeout",
    "final_error": "stuck",
    "fallen": "fell", "fault:fallen": "fell", "pelvis_out_of_band": "fell",
    "halted": "halted", "estop": "halted",
    "not_standing": "nav_unhealthy", "no_pose": "nav_unhealthy", "pose_stale": "nav_unhealthy",
    "deploy_stale": "nav_unhealthy", "deploy_not_running": "nav_unhealthy", "deploy_not_ready": "nav_unhealthy",
    "deploy_not_in_control": "nav_unhealthy", "band_release_failed": "nav_unhealthy",
    "no_occupancy": "nav_unhealthy", "bad_args": "nav_unhealthy", "tf_error": "nav_unhealthy",
}


def map_body_reason(reason: str | None) -> str:
    r = str(reason or "")
    if r in BODY_REASON:
        return BODY_REASON[r]
    if r.startswith("nav2_"):
        return "nav_unhealthy"
    if r.startswith("fault:"):
        return "fell" if "fall" in r else "nav_unhealthy"
    if r.startswith("internal_error"):
        return "nav_unhealthy"
    return r or "nav_unhealthy"


@dataclass(frozen=True)
class NavConfig:
    cruise_mps: float = 0.45
    nav_speed_mps: float = 0.40
    arrive_tol_m: float = 0.30
    final_pos_tol_m: float = 0.25
    final_yaw_tol_deg: float = 12.0
    reposition_mps: float = 0.20
    reposition_tol_m: float = 0.10
    reposition_tol_deg: float = 8.0
    reposition_timeout_s: float = 12.0
    approach_max_m: float = 0.40
    timeout_k: float = 1.8
    timeout_base_s: float = 12.0
    timeout_min_s: float = 20.0
    timeout_max_s: float = 240.0
    cancel_settle_s: float = 1.5
    cancel_grace_s: float = 3.0            # how long a cancel/halt waits for the body's terminal event
    poll_s: float = 0.02

    @classmethod
    def from_dicts(cls, walking: dict | None = None, workspace: dict | None = None) -> "NavConfig":
        d = {**(walking or {})}
        if workspace and "approach_max_m" in workspace:
            d["approach_max_m"] = workspace["approach_max_m"]
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})


class NavigationService:
    def __init__(self, world: Any, body: Any, clock: Any, cfg: NavConfig | None = None, *,
                 gate: HaltGate | None = None, events: EventSink | None = None, observation_id=None,
                 locations: LocationsService | None = None):
        self.world = world
        self.body = body
        self.clock = clock
        self.cfg = cfg or NavConfig()
        self.gate = gate or HaltGate()
        self.events = events or EventSink()
        self.observation_id = observation_id
        self.locations = locations or LocationsService(world)
        self.map = world.static_map()
        self._last_at: str | None = None
        self._target: str | None = None
        self._moving = False
        self._handles: dict[str, ResultHandle] = {}
        self._results: dict[str, NavigateResult] = {}
        self.last_motion_t: float | None = None           # for the "no base motion since" reachability rule
        self._anchor: tuple[str, float, float] | None = None   # after a reposition: (keypoint, stance x, y)
        kp, _ = self._keypoint_near()
        self._last_at = kp or "start"

    @property
    def executor(self) -> str:
        return getattr(self.body, "name", "sonic_walk")

    # ------------------------------------------------------------------ names and poses
    def resolve(self, location: str) -> str:
        """user -> the delivery keypoint; a room name is its own keypoint; anything else unchanged."""
        if location == "user":
            u = self.map.lookup_keypoints().get("people", {}).get("user")
            return u["keypoint"] if u else location
        return location

    def keypoint_pose(self, name: str) -> Pose2D | None:
        k = self.map.keypoints.get(self.resolve(name))
        return None if k is None else Pose2D(k.x, k.y, k.yaw)

    def _keypoint_near(self, x: float | None = None, y: float | None = None) -> tuple[str | None, float]:
        if x is None:
            p = self.world.robot_pose()
            x, y = p.x, p.y
        return self.map.nearest_keypoint(x, y, self.cfg.arrive_tol_m)

    def at(self) -> tuple[str | None, list[str] | None]:
        if self._moving and self._target:
            return None, [self._last_at or "start", self._target]
        if self._anchor is not None:
            name, sx, sy = self._anchor
            p = self.world.robot_pose()
            if math.hypot(p.x - sx, p.y - sy) <= 0.15:
                return name, None                          # a reposition keeps the anchoring keypoint (PLAN §6.3)
            self._anchor = None
        kp, _ = self._keypoint_near()
        if kp is not None:
            return kp, None
        if self._target and self._last_at and self._target != self._last_at:
            return None, [self._last_at, self._target]
        return None, None

    @property
    def moving(self) -> bool:
        return self._moving

    # ------------------------------------------------------------------ timeouts / locations
    def path_length(self, from_pose: Any, location: str) -> float | None:
        k = self.map.keypoints.get(self.resolve(location))
        if k is None:
            return None
        x, y = (from_pose.x, from_pose.y) if hasattr(from_pose, "x") else (from_pose[0], from_pose[1])
        return self.map.grid.distance((x, y), (k.x, k.y))

    def timeout_s(self, from_pose: Any = None, location: str = "") -> float:
        """PLAN §1.3 #22: clamp(1.8 * path_m / v + 12, 20, 240)."""
        c = self.cfg
        if from_pose is None:
            from_pose = self.world.robot_pose()
        d = self.path_length(from_pose, location)
        if d is None:
            d = 40.0
        return max(c.timeout_min_s, min(c.timeout_max_s, c.timeout_k * d / max(c.nav_speed_mps, 0.05)
                                        + c.timeout_base_s))

    async def list_locations(self, current_pose: Any = None, query: str | None = None) -> list[NamedLocation]:
        return self.locations.list(current_pose, query)

    def health(self) -> ServiceHealth:
        h = self.body.health()
        if not h.ok:
            return h
        age_fn = getattr(self.world, "pose_age_s", None)
        if callable(age_fn):
            age = age_fn()
            if age > 0.5:
                return ServiceHealth(False, "down", f"localizer stale ({age:.2f} s)")
        p = self.world.robot_pose()
        if p.fallen:
            return ServiceHealth(False, "fault", "robot fell")
        return h

    # ------------------------------------------------------------------ executions
    async def navigate(self, location: str, *, execution: Execution, timeout_s: float | None = None) -> ResultHandle:
        if timeout_s is not None:
            execution.args = {**execution.args, "timeout_s": timeout_s}
        execution.args = {**execution.args, "location": location}
        return self.start(execution)

    async def reposition(self, stance_w: Pose2D, *, anchor: str, execution: Execution) -> ResultHandle:
        execution.args = {**execution.args, "location": "reach_stance", "anchor": anchor,
                          "stance": {"x": stance_w.x, "y": stance_w.y, "yaw": stance_w.yaw}}
        return self.start(execution)

    def start(self, execution: Execution) -> ResultHandle:
        """Dispatch a navigate execution (keypoint or reposition). Never blocks."""
        reposition = execution.action == "reposition" or execution.args.get("location") == "reach_stance" \
            or bool(execution.args.get("stance"))
        work = self._reposition_work if reposition else self._navigate_work
        h = start_execution(execution, lambda handle: work(execution, handle), clock=self.clock,
                            observation_id=self._obs, executor=self.executor)
        self._handles[execution.execution_id] = h
        return h

    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None:
        h = self._handles.get(execution_id)
        if h is not None:
            h.cancel(reason)

    async def status(self, execution_id: str) -> NavigateResult | None:
        return self._results.get(execution_id)

    def _obs(self) -> str | None:
        return self.observation_id() if callable(self.observation_id) else None

    # ------------------------------------------------------------------ the waiting loop
    async def _await_op(self, op: Any, h: ResultHandle, epoch: int, timeout_s: float) -> tuple[dict, str | None]:
        """Wait for the body op; returns (terminal, why) with why in {None, "cancel", "halt", "timeout"}."""
        t_end = self.clock.now() + timeout_s
        why: str | None = None
        while not op.done:
            if h.cancel_requested:
                why = "cancel"
            elif self.gate.halted_since(epoch):
                why = "halt"
            elif self.clock.now() > t_end:
                why = "timeout"
            if why:
                break
            await self.clock.sleep(self.cfg.poll_s)
        if why and not op.done:
            if why != "halt":
                op.cancel(why)                                 # body stop: planner IDLE, never command{stop}
            try:
                res = await self.clock.wait_for(op.result(), self.cfg.cancel_grace_s)
            except asyncio.TimeoutError:
                res = {"state": "failed", "data": {"reason": "no terminal event after stop"}}
            return res, why
        return await op.result(), why

    # ------------------------------------------------------------------ navigate(keypoint)
    async def _navigate_work(self, ex: Execution, h: ResultHandle):
        t0 = self.clock.now()
        c = self.cfg
        loc = self.resolve(str(ex.args.get("location", "")))
        k = self.map.keypoints.get(loc)
        start_kp, _ = self._keypoint_near()
        start_kp = start_kp or self._last_at
        if k is None:
            return self._finish(ex, "failed", loc, reason="unknown_location", at=start_kp, t0=t0)
        p = self.world.robot_pose()
        plan = self.map.grid.plan((p.x, p.y), (k.x, k.y))
        if not plan.ok:
            return self._finish(ex, "failed", loc, reason="no_path", at=start_kp, t0=t0,
                                extra={"body_reason": plan.reason})
        timeout = self.timeout_s(p, loc)
        if ex.args.get("timeout_s"):
            timeout = min(timeout, float(ex.args["timeout_s"]))
        epoch = self.gate.epoch
        if self.gate.latched:
            return self._finish(ex, "failed", loc, reason="halted", at=start_kp, t0=t0)
        self._moving, self._target = True, loc
        self._last_at = start_kp
        self._anchor = None
        self.events.emit("base_moving", execution_id=ex.execution_id, to=loc)
        try:
            op = await self.body.go_to(k.x, k.y, k.yaw, speed=c.cruise_mps, timeout_s=timeout + 5.0,
                                       final_pos_tol=c.final_pos_tol_m, final_yaw_tol_deg=c.final_yaw_tol_deg)
            res, why = await self._await_op(op, h, epoch, timeout)
        finally:
            self._moving = False
            self.last_motion_t = self.clock.now()
        data = res.get("data") or {}
        state = res.get("state")
        fp = self.world.robot_pose()
        final_err = math.hypot(fp.x - k.x, fp.y - k.y)
        here, _ = self._keypoint_near(fp.x, fp.y)
        body_extra = {"body_reason": data.get("reason"), "body_op": getattr(op, "id", None),
                      "final_pose": [round(fp.x, 3), round(fp.y, 3), round(fp.yaw, 4)]}
        common = dict(path_len_m=round(float(data.get("path_len_m") or plan.length), 2),
                      walked_m=round(float(data.get("walked_m") or data.get("displacement_m") or 0.0), 2),
                      replans=int(data.get("replans") or 0), final_err_m=round(final_err, 3), t0=t0,
                      extra=body_extra)
        if state == "succeeded" and why is None:
            self._last_at = loc
            self.events.emit("base_stopped", execution_id=ex.execution_id, at=loc)
            return self._finish(ex, "succeeded", loc, at=loc, **common)
        between = None if here else [start_kp or "start", loc]
        if here:
            self._last_at = here
        self.events.emit("base_stopped", execution_id=ex.execution_id, at=here, between=between)
        if why == "cancel":
            return self._finish(ex, "cancelled", loc, reason=h.cancel_reason or "cancelled", at=here,
                                between=between, **common)
        if why == "halt" or (state == "canceled" and self.gate.halted_since(epoch)):
            return self._finish(ex, "failed", loc, reason="halted", at=here, between=between, **common)
        if why == "timeout":
            return self._finish(ex, "timed_out", loc, reason="timeout", at=here, between=between, **common)
        if state == "canceled":                              # someone else pre-empted the body
            return self._finish(ex, "cancelled", loc, reason=str(data.get("reason") or "preempted"), at=here,
                                between=between, **common)
        reason = map_body_reason(data.get("reason"))
        blocked = [start_kp or "start", loc] if reason == "blocked" else None
        if reason == "fell":
            self.events.emit("safety_event", kind="fell", execution_id=ex.execution_id)
        return self._finish(ex, "failed", loc, reason=reason, at=here, between=between, blocked_edge=blocked,
                            **common)

    # ------------------------------------------------------------------ navigate(reach_stance)
    async def _reposition_work(self, ex: Execution, h: ResultHandle):
        t0 = self.clock.now()
        c = self.cfg
        st = ex.args.get("stance") or {}
        anchor = ex.args.get("anchor") or self._keypoint_near()[0] or self._last_at
        if not st or "x" not in st:
            return self._finish(ex, "failed", "reach_stance", reason="no_reach_stance", at=anchor, t0=t0,
                                kind="reposition")
        sx, sy = float(st["x"]), float(st["y"])
        syaw = float(st.get("yaw", self.world.robot_pose().yaw))
        p = self.world.robot_pose()
        d = math.hypot(sx - p.x, sy - p.y)
        extra = {"stance": {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(syaw, 4)},
                 "interim": "body go_to with a tight tolerance; PLAN's strafing `approach` op is M2b"}
        if d > c.approach_max_m + 0.05:
            return self._finish(ex, "failed", "reach_stance", reason="stance_not_reached", at=anchor, t0=t0,
                                kind="reposition", final_err_m=round(d, 3),
                                extra={**extra, "detail": f"stance {d:.2f} m away > approach_max_m"})
        if not self.map.grid.is_free(sx, sy) or not self.map.grid.segment_free((p.x, p.y), (sx, sy)):
            return self._finish(ex, "failed", "reach_stance", reason="stance_not_reached", at=anchor, t0=t0,
                                kind="reposition", final_err_m=round(d, 3),
                                extra={**extra, "detail": "no straight free line to the stance"})
        epoch = self.gate.epoch
        self._moving, self._target = True, anchor
        try:
            op = await self.body.go_to(sx, sy, syaw, speed=c.reposition_mps, timeout_s=c.reposition_timeout_s,
                                       final_pos_tol=c.reposition_tol_m, final_yaw_tol_deg=c.reposition_tol_deg)
            res, why = await self._await_op(op, h, epoch, c.reposition_timeout_s + 2.0)
        finally:
            self._moving = False
            self.last_motion_t = self.clock.now()
        fp = self.world.robot_pose()
        err = math.hypot(fp.x - sx, fp.y - sy)
        yaw_err = abs(math.degrees(coords.ang_diff(syaw, fp.yaw)))
        if anchor:
            self._anchor = (anchor, fp.x, fp.y)
            self._last_at = anchor
        data = res.get("data") or {}
        common = dict(kind="reposition", walked_m=round(math.hypot(fp.x - p.x, fp.y - p.y), 3),
                      final_err_m=round(err, 3), t0=t0, path_len_m=round(d, 3),
                      extra={**extra, "yaw_err_deg": round(yaw_err, 1), "body_reason": data.get("reason")})
        if why == "cancel":
            return self._finish(ex, "cancelled", "reach_stance", reason=h.cancel_reason or "cancelled", at=anchor,
                                **common)
        if why == "halt":
            return self._finish(ex, "failed", "reach_stance", reason="halted", at=anchor, **common)
        if why == "timeout":
            return self._finish(ex, "timed_out", "reach_stance", reason="timeout", at=anchor, **common)
        if res.get("state") == "succeeded" and err <= c.reposition_tol_m + 1e-6 and yaw_err <= c.reposition_tol_deg:
            return self._finish(ex, "succeeded", "reach_stance", at=anchor, **common)
        if res.get("state") == "canceled":
            return self._finish(ex, "cancelled", "reach_stance", reason=str(data.get("reason") or "preempted"),
                                at=anchor, **common)
        reason = "stance_not_reached"
        if res.get("state") == "failed":
            r = map_body_reason(data.get("reason"))
            if r in ("fell", "nav_unhealthy", "halted"):
                reason = r
        return self._finish(ex, "failed", "reach_stance", reason=reason, at=anchor, **common)

    # ------------------------------------------------------------------ envelope
    def _finish(self, ex: Execution, status: str, location: str, *, reason: str | None = None, at: str | None = None,
                between: list[str] | None = None, blocked_edge: list[str] | None = None, t0: float = 0.0,
                path_len_m: float = 0.0, walked_m: float = 0.0, replans: int = 0, final_err_m: float = 0.0,
                kind: str = "keypoint", extra: dict | None = None):
        now = self.clock.now()
        r = NavigateResult(execution_id=ex.execution_id, status=status, location=location, reason=reason, at=at,   # type: ignore[arg-type]
                           between=between, blocked_edge=blocked_edge, executor=self.executor,             # type: ignore[arg-type]
                           path_len_m=path_len_m, walked_m=walked_m, duration_s=round(now - t0, 2),
                           replans=replans, observation_id=self._obs(), kind=kind, final_err_m=final_err_m)   # type: ignore[arg-type]
        self._results[ex.execution_id] = r
        data = dict(r.__dict__)
        data.update(extra or {})
        if self.executor in ("kinematic_nav",):
            data["stepping_stone"] = True
        return finish(ex, status, data, t_end=round(now, 3), observation_id=self._obs(), executor=self.executor)
