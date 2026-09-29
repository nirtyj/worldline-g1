"""NavigationService (doc §38; PLAN §6.3): keypoint -> pose -> body go_to, with result envelopes, cancel / halt
mapping and timeouts scaled for humanoid walking.

    navigate(location)      resolve aliases (user, rooms) -> the keypoint's stand pose (x, y, yaw) -> BodyPort.go_to
                            (SONIC through wl-body, or the lite body) -> NavigateResult in a ToolResult
    reposition(stance)      navigate(location="reach_stance"): the body's `approach` op (B.6: a strafing
                            reposition on ground truth, tol 5 cm / 5 deg) to the stance the last check_reachability
                            returned (<= approach_max_m, straight-line free); a body without it (M1, lite) gets a
                            go_to with a tight tolerance, labelled INTERIM. A stance farther than approach_max_m or
                            without a straight free line (the outline-wide search's far stance: the other side of a
                            table) is reached in two legs under one lease: A* go_to to a free spot next to it (the
                            stance's `via`, else one found here), then that approach from where the walk ended
    list_locations          services/locations.py (walking distance from the current GT pose)
    timeout_s               clamp(1.8 * path_m / v + 12, 20, 240) with v = the profile's effective walking speed
    at()                    keypoint within 0.30 m of the GT pose, else between [last keypoint, target]

Body terminal reasons (docs/contracts/m1.md §3.4) map onto the envelope's reason codes (api/reasons.py NAVIGATION):
see BODY_REASON. A cancel stops the body (planner IDLE) and waits for its settle; a halt (the bridge latched it)
ends running navigation `failed(halted)`; a walk that got stuck after the body's own replans is `failed(blocked)`
with `blocked_edge = [from, to]` (THOR never produced it; agent/state.apply_navigate consumes it).

Body fences and leases (M2b B.2, docs/contracts/m1.md §3.11): every navigate leases the body (LOCOMOTION) for its
execution and releases it at the end; every body op carries the execution's fence (execution_id, generation,
control_epoch). The body's refusals keep their meaning (services.common.refusal): halted -> failed(halted), a stale
fence (stale_command, data.why control_epoch | generation | resume_epoch) -> failed(stale_result) + a stale_result
event, body_busy -> failed(body_busy).
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

from .common import EventSink, HaltGate, body_acquire, body_fence, body_release, refusal, start_execution
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
    reposition_timeout_s: float = 12.0     # the INTERIM go_to reposition (a body without `approach`)
    approach_timeout_s: float = 35.0       # the body's approach op (B.6 live: p50 11 s, p90 20-22 s)
    approach_tol_m: float = 0.05           # B.6 contract tolerance (p90 4.0 / 3.9 cm live)
    approach_tol_deg: float = 5.0
    approach_goal_clearance_m: float = 0.12   # the body's approach refuses a goal closer to an obstacle (raw map)
    approach_path_clearance_m: float = 0.05   # ... and a segment passing closer than this
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

    def uses_approach(self) -> bool:
        """navigate(reach_stance) runs the body's approach op (B.6) rather than the INTERIM tight go_to."""
        fn = getattr(self.body, "supports", None)
        try:
            return bool(fn("approach")) if callable(fn) else False
        except Exception:  # noqa: BLE001
            return False

    def _stance_line_ok(self, p: Any, sx: float, sy: float, approach: bool) -> bool:
        """The approach op's own map rule (contract §3.13: the goal 0.12 m and the segment 0.05 m of raw clearance),
        so the runtime refuses what the body would refuse and sends what it accepts: a reach stance stands closer
        to the furniture than A*'s inflated free space. The INTERIM go_to needs that free space."""
        g = self.map.grid
        if not approach:
            return bool(g.is_free(sx, sy) and g.segment_free((p.x, p.y), (sx, sy)))
        c = self.cfg
        if g.clearance(sx, sy) < c.approach_goal_clearance_m or g.is_outside(sx, sy):
            return False
        n = max(2, int(math.hypot(sx - p.x, sy - p.y) / 0.02) + 1)
        return all(g.clearance(p.x + (sx - p.x) * k / (n - 1), p.y + (sy - p.y) * k / (n - 1))
                   >= c.approach_path_clearance_m for k in range(n))

    def reposition_timeout_s(self, stance: dict | None = None) -> float:
        """The reposition's budget; a far stance adds its A* walk's navigate timeout."""
        t = self.cfg.approach_timeout_s if self.uses_approach() else self.cfg.reposition_timeout_s
        leg = self._far_leg(stance) if stance else None
        return t + (self._walk_timeout_s(leg[1]) if leg is not None else 0.0)

    def _walk_timeout_s(self, path_m: float) -> float:
        c = self.cfg
        return max(c.timeout_min_s, min(c.timeout_max_s, c.timeout_k * path_m / max(c.nav_speed_mps, 0.05)
                                        + c.timeout_base_s))

    def _far_leg(self, st: dict | None, p: Any = None) -> tuple[tuple[float, float], float] | None:
        """For a stance the approach cannot take from here (farther than approach_max_m, or no straight free line):
        (the A* go_to target next to it, the planned path length); None for a near stance or no such target."""
        if not st or "x" not in st:
            return None
        p = p or self.world.robot_pose()
        sx, sy = float(st["x"]), float(st["y"])
        approach = self.uses_approach()
        if math.hypot(sx - p.x, sy - p.y) <= self.cfg.approach_max_m + 0.05 and self._stance_line_ok(p, sx, sy,
                                                                                                    approach):
            return None
        via = self._staging(st, p, approach)
        if via is None:
            return None
        plan = self.map.grid.plan((p.x, p.y), via)
        return (via, float(plan.length)) if plan.ok else None

    def _staging(self, st: dict, p: Any, approach: bool) -> tuple[float, float] | None:
        """The A* go_to target of a far reposition: the check's `via` when it is still a free spot of the planner's
        map on the robot's component from which the final leg (approach, or the tight go_to) passes its own rule;
        else the nearest such spot within 0.45 m of the stance (the stance itself first)."""
        g = self.map.grid
        sx, sy = float(st["x"]), float(st["y"])
        comp = g.component(p.x, p.y)

        class _XY:
            def __init__(self, x: float, y: float):
                self.x, self.y = x, y

        def ok(qx: float, qy: float) -> bool:
            if not g.is_free(qx, qy) or g.nav.c_blocked[g.nav._world_to_c(qx, qy)]:
                return False
            if comp is not None and g.component(qx, qy) != comp:
                return False
            return self._stance_line_ok(_XY(qx, qy), sx, sy, approach)

        v = st.get("via")
        if isinstance(v, dict) and "x" in v and ok(float(v["x"]), float(v["y"])):
            return float(v["x"]), float(v["y"])
        if ok(sx, sy):
            return sx, sy
        r = 0.10
        while r <= 0.45 + 1e-9:
            best = None
            for k in range(16):
                a = k * math.pi / 8
                qx, qy = round(sx + r * math.cos(a), 3), round(sy + r * math.sin(a), 3)
                if ok(qx, qy):
                    c = g.clearance(qx, qy)
                    if best is None or c > best[0]:
                        best = (c, qx, qy)
            if best is not None:
                return best[1], best[2]
            r += 0.05
        return None

    async def _lease(self, ex: Execution) -> dict:
        return await body_acquire(self.body, ex, "LOCOMOTION")

    def _refused(self, ex: Execution, meaning: str | None, reason: Any, data: dict) -> str:
        """The envelope reason for a body refusal; a stale fence is also a stale_result event (PLAN §5.5)."""
        if meaning == "stale_result":
            self.events.emit("stale_result", execution_id=ex.execution_id, tool="navigate", why=data.get("why"),
                             body_reason=reason, source="body")
        return meaning or map_body_reason(reason)

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
        epoch = ex.control_epoch                          # one epoch space: the halt fences this number (R.2)
        if self.gate.latched:
            return self._finish(ex, "failed", loc, reason="halted", at=start_kp, t0=t0)
        lease = await self._lease(ex)
        if not lease.get("ok"):
            r = self._refused(ex, lease.get("meaning"), lease.get("reason"), dict(lease.get("data") or {}))
            return self._finish(ex, "failed", loc, reason=r, at=start_kp, t0=t0,
                                extra={"body_reason": lease.get("reason"), "lease": "refused"})
        self._moving, self._target = True, loc
        self._last_at = start_kp
        self._anchor = None
        self.events.emit("base_moving", execution_id=ex.execution_id, to=loc)
        op = None
        try:
            op = await self.body.go_to(k.x, k.y, k.yaw, speed=c.cruise_mps, timeout_s=timeout + 5.0,
                                       final_pos_tol=c.final_pos_tol_m, final_yaw_tol_deg=c.final_yaw_tol_deg,
                                       fence=body_fence(self.body, ex))
            res, why = await self._await_op(op, h, epoch, timeout)
        finally:
            self._moving = False
            self.last_motion_t = self.clock.now()
            await body_release(self.body, ex)
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
        if why == "halt" or (state == "canceled" and (self.gate.halted_since(epoch) or data.get("reason") == "halt")):
            return self._finish(ex, "failed", loc, reason="halted", at=here, between=between, **common)
        if why == "timeout":
            return self._finish(ex, "timed_out", loc, reason="timeout", at=here, between=between, **common)
        if state == "canceled":                              # someone else pre-empted the body
            return self._finish(ex, "cancelled", loc, reason=str(data.get("reason") or "preempted"), at=here,
                                between=between, **common)
        reason = self._refused(ex, refusal(data.get("reason"), data), data.get("reason"), data)
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
        approach = self.uses_approach()
        extra: dict[str, Any] = {"stance": {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(syaw, 4)},
                                 "reposition_op": "approach" if approach else "go_to"}
        if approach:
            tol_m, tol_deg, budget = c.approach_tol_m, c.approach_tol_deg, c.approach_timeout_s
        else:
            tol_m, tol_deg, budget = c.reposition_tol_m, c.reposition_tol_deg, c.reposition_timeout_s
            extra["interim"] = "body go_to with a tight tolerance; the body has no `approach` op (B.6)"
        via, walk_plan = None, None
        if d > c.approach_max_m + 0.05 or not self._stance_line_ok(p, sx, sy, approach):
            # a far stance (the outline-wide search): A* go_to next to it first
            via = self._staging(st, p, approach)
            if via is None:
                return self._finish(ex, "failed", "reach_stance", reason="stance_not_reached", at=anchor, t0=t0,
                                    kind="reposition", final_err_m=round(d, 3), extra={
                                        **extra, "detail": f"stance {d:.2f} m away and no free spot next to it that "
                                                           f"A* reaches and the final approach passes"})
            walk_plan = self.map.grid.plan((p.x, p.y), via)
            if not walk_plan.ok:
                return self._finish(ex, "failed", "reach_stance", reason="no_path", at=anchor, t0=t0,
                                    kind="reposition", final_err_m=round(d, 3),
                                    extra={**extra, "body_reason": walk_plan.reason,
                                           "detail": f"no A* path to the spot next to the stance ({walk_plan.reason})"})
            extra["via"] = {"x": round(via[0], 3), "y": round(via[1], 3)}
            extra["legs"] = ["go_to", extra["reposition_op"]]
        epoch = ex.control_epoch
        if self.gate.latched:
            return self._finish(ex, "failed", "reach_stance", reason="halted", at=anchor, t0=t0, kind="reposition",
                                final_err_m=round(d, 3), extra=extra)
        lease = await self._lease(ex)
        if not lease.get("ok"):
            r = self._refused(ex, lease.get("meaning"), lease.get("reason"), dict(lease.get("data") or {}))
            return self._finish(ex, "failed", "reach_stance", reason=r, at=anchor, t0=t0, kind="reposition",
                                final_err_m=round(d, 3), extra={**extra, "body_reason": lease.get("reason")})
        self._moving, self._target = True, anchor
        walk: dict[str, Any] | None = None                 # the far stance's A* leg: its outcome
        walked0, p1 = 0.0, p
        try:
            if via is not None and walk_plan is not None:
                t_walk = self._walk_timeout_s(float(walk_plan.length))
                op0 = await self.body.go_to(via[0], via[1], syaw, speed=c.cruise_mps, timeout_s=t_walk + 5.0,
                                            final_pos_tol=c.final_pos_tol_m, final_yaw_tol_deg=c.final_yaw_tol_deg,
                                            fence=body_fence(self.body, ex))
                res0, why0 = await self._await_op(op0, h, epoch, t_walk)
                d0 = res0.get("data") or {}
                p1 = self.world.robot_pose()
                walked0 = float(d0.get("walked_m") or d0.get("displacement_m") or 0.0) or \
                    math.hypot(p1.x - p.x, p1.y - p.y)
                walk = {"state": res0.get("state"), "why": why0, "body_reason": d0.get("reason"),
                        "path_len_m": round(float(d0.get("path_len_m") or walk_plan.length), 2),
                        "final_pose": [round(p1.x, 3), round(p1.y, 3), round(p1.yaw, 4)]}
                d1 = math.hypot(sx - p1.x, sy - p1.y)
                if why0 is None and res0.get("state") == "succeeded" and not (
                        d1 <= c.approach_max_m + 0.05 and self._stance_line_ok(p1, sx, sy, approach)):
                    walk["why"] = "approach_refused"
                    walk["detail"] = f"after the walk the stance is {d1:.2f} m away without a free straight line"
            if walk is None or (walk["why"] is None and walk["state"] == "succeeded"):
                if approach:
                    op = await self.body.approach(sx, sy, syaw, v=c.reposition_mps, tol=(tol_m, tol_deg),
                                                  timeout_s=budget, fence=body_fence(self.body, ex))
                else:
                    op = await self.body.go_to(sx, sy, syaw, speed=c.reposition_mps, timeout_s=budget,
                                               final_pos_tol=tol_m, final_yaw_tol_deg=tol_deg,
                                               fence=body_fence(self.body, ex))
                res, why = await self._await_op(op, h, epoch, budget + 2.0)
            else:
                res, why = res0, walk["why"]                  # the walk leg ended the reposition
        finally:
            self._moving = False
            self.last_motion_t = self.clock.now()
            await body_release(self.body, ex)
        fp = self.world.robot_pose()
        err = math.hypot(fp.x - sx, fp.y - sy)
        yaw_err = abs(math.degrees(coords.ang_diff(syaw, fp.yaw)))
        if anchor:
            self._anchor = (anchor, fp.x, fp.y)
            self._last_at = anchor
        data = res.get("data") or {}
        body_extra = {k: data.get(k) for k in ("pos_err", "yaw_err_deg", "attempts", "turns") if k in data}
        if walk is not None:
            extra["walk"] = walk
        common = dict(kind="reposition", walked_m=round(walked0 + math.hypot(fp.x - p1.x, fp.y - p1.y), 3),
                      final_err_m=round(err, 3), t0=t0,
                      path_len_m=round(d if walk_plan is None else float(walk_plan.length) + math.hypot(
                          sx - via[0], sy - via[1]), 3),                       # type: ignore[index]
                      extra={**extra, "yaw_err_deg": round(yaw_err, 1), "body_reason": data.get("reason"),
                             "tol": [tol_m, tol_deg], **({"body": body_extra} if body_extra else {})})
        if walk is not None and walk["why"] == "approach_refused":
            return self._finish(ex, "failed", "reach_stance", reason="stance_not_reached", at=anchor,
                                **{**common, "extra": {**common["extra"], "detail": walk["detail"]}})
        if walk is not None and walk["why"] is None and walk["state"] == "failed":
            # the A* walk failed: the body's reason, mapped as a keypoint navigate maps it (no_path, blocked, fell)
            meaning = refusal(data.get("reason"), data)
            reason = self._refused(ex, meaning, data.get("reason"), data) if meaning is not None \
                else map_body_reason(data.get("reason"))
            if reason == "fell":
                self.events.emit("safety_event", kind="fell", execution_id=ex.execution_id)
            return self._finish(ex, "failed", "reach_stance", reason=reason, at=anchor, **{
                **common, "extra": {**common["extra"], "detail": "the A* walk toward the reach stance failed"}})
        if why == "cancel":
            return self._finish(ex, "cancelled", "reach_stance", reason=h.cancel_reason or "cancelled", at=anchor,
                                **common)
        if why == "halt" or (res.get("state") == "canceled" and data.get("reason") == "halt"):
            return self._finish(ex, "failed", "reach_stance", reason="halted", at=anchor, **common)
        if why == "timeout":
            return self._finish(ex, "timed_out", "reach_stance", reason="timeout", at=anchor, **common)
        # success is judged on ground truth after the body's own settle: the stance the reachability asked for,
        # within the reposition's tolerance (the approach aims at 0.8 x its own; a go_to's tolerance is wider)
        ok_m = c.reposition_tol_m if not approach else max(c.reposition_tol_m, tol_m)
        ok_deg = c.reposition_tol_deg if not approach else max(c.reposition_tol_deg, tol_deg)
        if res.get("state") == "succeeded" and err <= ok_m + 1e-6 and yaw_err <= ok_deg:
            return self._finish(ex, "succeeded", "reach_stance", at=anchor, **common)
        if res.get("state") == "canceled":
            return self._finish(ex, "cancelled", "reach_stance", reason=str(data.get("reason") or "preempted"),
                                at=anchor, **common)
        reason = "stance_not_reached"
        if res.get("state") == "failed":
            meaning = refusal(data.get("reason"), data)
            if meaning is not None:
                reason = self._refused(ex, meaning, data.get("reason"), data)
            else:
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
