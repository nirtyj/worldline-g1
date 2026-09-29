"""ManipulationService (doc §39; PLAN §6.4): skill registry, check_reachability, and pick/place through an executor.

    check_reachability(object_type, object_id?, candidates, at)   services/reachability.py, judged from here;
                     always `succeeded` when it ran (reachable=false included). The newest result per type is kept
                     for the pick's stance check and object binding.
    execute(pick)    select a healthy skill (profile backend order) -> stance check (moved > 5 cm / 5 deg since
                     the reachability -> failed(base_moving), body untouched) -> executor phases -> verify the hand
                     holds it (GT hands; the only hand verifier) -> ManipulationResult.
    execute(place)   target served from where the robot stands (else failed(target_not_here)) -> a free spot on
                     it within reach from here (else no_room_in_reach / no_room_on_surface) -> executor -> verify
                     where == target.
Never walks (PLAN §1.3 #26): the base is where check_reachability judged it. `base_shift_m` reports any drift.

Executors (services/executors/): lite / kinematic_attach (STEPPING STONE, labelled everywhere), sonic_arm_script and
groot_sonic (stubs until M3/M4, unhealthy -> CAPABILITY rejections via `capability()`).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Iterable

from api.execution import Execution, ResultHandle
from api.results import ManipulationResult, ReachabilityResult, finish
from api.skills import SkillSpec
from api.types import ServiceHealth
from world import coords

from .common import EventSink, HaltGate, start_execution
from .executors.kinematic_attach import ManipJob
from .reachability import ReachabilityModel

STEPPING = ("kinematic_attach", "lite", "sonic_arm_script")


@dataclass(frozen=True)
class ManipConfig:
    stance_drift_m: float = 0.05
    stance_drift_deg: float = 5.0
    reach_sense_s: float = 0.2            # check_reachability "takes" this long (THOR 0.3 s)
    verify_hold_s: float = 0.3

    @classmethod
    def from_dict(cls, d: dict | None) -> "ManipConfig":
        d = dict(d or {})
        return cls(**{k: float(d[k]) for k in ("stance_drift_m", "stance_drift_deg", "reach_sense_s", "verify_hold_s")
                      if k in d})


class ManipulationService:
    def __init__(self, world: Any, registry: Any, reach: ReachabilityModel, executors: dict[str, Any], clock: Any,
                 cfg: ManipConfig | None = None, *, nav: Any = None, observation: Any = None,
                 gate: HaltGate | None = None, events: EventSink | None = None):
        self.world = world
        self.registry = registry
        self.reach = reach
        self.executors = executors                    # backend -> executor
        self.clock = clock
        self.cfg = cfg or ManipConfig()
        self.nav = nav
        self.observation = observation
        self.gate = gate or HaltGate()
        self.events = events or EventSink()
        self._last_reach: dict[str, tuple[ReachabilityResult, float, tuple[float, float, float]]] = {}
        self._handles: dict[str, ResultHandle] = {}
        self._results: dict[str, ManipulationResult] = {}

    def _obs(self) -> str | None:
        return self.observation.glance_record() if self.observation is not None else None

    # ------------------------------------------------------------------ registry / health
    async def list_skills(self):
        return self.registry

    def executor_for(self, skill: SkillSpec):
        return self.executors.get(skill.backend)

    def capability(self, action: str, object_type: str, arm: str | None = None) -> tuple[SkillSpec | None, str]:
        """(skill, "") or (None, why) -> the bridge raises Rejected("capability", "policy_unavailable", ...)."""
        if self.registry.select(action, object_type, arm if arm in ("left", "right") else None) is None:
            return None, f"no loaded skill can {action} {object_type}"
        skill, why = self.registry.select_healthy(action, object_type, arm if arm in ("left", "right") else None)
        if skill is not None and self.executor_for(skill) is None:
            return None, f"{skill.skill_id}: no executor for backend {skill.backend}"
        return skill, why

    def health(self, skill_id: str | None = None) -> ServiceHealth:
        if skill_id is not None:
            return self.registry.healthy(skill_id)
        states = [self.registry.healthy(s.skill_id) for s in self.registry.skills()]
        if any(h.ok for h in states):
            return ServiceHealth(True, "ok")
        detail = "; ".join(sorted({h.detail for h in states if h.detail}))
        return ServiceHealth(False, "down", detail or "no healthy skill")

    # ------------------------------------------------------------------ reachability
    async def check_reachability(self, object_type: str, object_id: str | None = None, *,
                                 candidates: Iterable[str] = (), at: str | None = None) -> ReachabilityResult:
        r = self.reach.check(object_type, object_id, candidates=list(candidates or ()), at=at)
        p = self.world.robot_pose()
        self._last_reach[object_type] = (r, self.clock.now(), (p.x, p.y, p.yaw))
        return r

    def last_reachability(self, object_type: str) -> ReachabilityResult | None:
        rec = self._last_reach.get(object_type)
        return rec[0] if rec else None

    def start_reachability(self, execution: Execution) -> ResultHandle:
        a = execution.args

        async def work(h: ResultHandle):
            await self.clock.sleep(self.cfg.reach_sense_s)
            if h.cancel_requested:
                return finish(execution, "cancelled", {"reason": h.cancel_reason or "cancelled",
                                                       "object_type": a.get("object_type")},
                              t_end=round(self.clock.now(), 3), observation_id=self._obs())
            r = await self.check_reachability(str(a.get("object_type")), a.get("object_id"),
                                              candidates=a.get("candidates") or (), at=a.get("at"))
            data = dict(r.__dict__)
            data["source"] = self.world.source
            return finish(execution, "succeeded", data, t_end=round(self.clock.now(), 3), observation_id=self._obs())

        return start_execution(execution, work, clock=self.clock, observation_id=self._obs)

    # ------------------------------------------------------------------ pick / place
    async def execute(self, action: str, object_type: str, *, arm: str | None = None, target: str | None = None,
                      object_id: str | None = None, execution: Execution) -> ResultHandle:
        execution.args = {**execution.args, "action": action, "object_type": object_type, "arm": arm,
                          "target": target, "object_id": object_id}
        return self.start(execution)

    def start(self, execution: Execution) -> ResultHandle:
        a = execution.args
        action = str(a.get("action") or execution.action or "pick")
        skill, _ = self.capability(action, str(a.get("object_type")), a.get("arm"))
        work = self._pick if action == "pick" else self._place
        h = start_execution(execution, lambda handle: work(execution, handle, skill), clock=self.clock,
                            observation_id=self._obs, executor=skill.executor if skill else None)
        self._handles[execution.execution_id] = h
        return h

    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None:
        h = self._handles.get(execution_id)
        if h is not None:
            h.cancel(reason)

    async def status(self, execution_id: str) -> ManipulationResult | None:
        return self._results.get(execution_id)

    def _result(self, ex: Execution, status: str, skill: SkillSpec | None, *, action: str, object_type: str,
                reason: str | None = None, arm: str | None = None, object_id: str | None = None,
                target: str | None = None, holding: bool | None = None, phase: str | None = None, t0: float = 0.0,
                pose0=None, surface: str | None = None, extra: dict | None = None):
        now = self.clock.now()
        p = self.world.robot_pose()
        shift = math.hypot(p.x - pose0[0], p.y - pose0[1]) if pose0 else 0.0
        executor = skill.executor if skill else "lite"
        r = ManipulationResult(execution_id=ex.execution_id, status=status, skill=skill.skill_id if skill else "",   # type: ignore[arg-type]
                               object_type=object_type, reason=reason, action=action, arm=arm, object_id=object_id,   # type: ignore[arg-type]
                               target=target, holding=holding, executor=executor, phase=phase,   # type: ignore[arg-type]
                               duration_s=round(now - t0, 2), base_shift_m=round(shift, 3), surface=surface)
        self._results[ex.execution_id] = r
        data = dict(r.__dict__)
        if skill is not None:
            data.update(stepping_stone=skill.stepping_stone, skill_label=skill.label, skill_status=skill.status,
                        hand_mismatch=skill.hand_mismatch)
        data["source"] = self.world.source
        data.update(extra or {})
        return finish(ex, status, data, t_end=round(now, 3), observation_id=self._obs(), executor=executor)

    def _resolve_arm(self, arm: str | None, reach: ReachabilityResult | None) -> str:
        if arm in ("left", "right"):
            return arm
        pref = reach.preferred_arm if reach is not None else "either"
        if pref in ("left", "right"):
            return pref
        hands = self.world.hands()
        free = [a for a in ("left", "right") if not hands.get(a)]
        return free[0] if len(free) == 1 else "right"

    async def _pick(self, ex: Execution, h: ResultHandle, skill: SkillSpec | None):
        t0 = self.clock.now()
        a = ex.args
        otype = str(a.get("object_type"))
        p0 = self.world.robot_pose()
        pose0 = (p0.x, p0.y, p0.yaw)
        kw = dict(action="pick", object_type=otype, t0=t0, pose0=pose0)
        if skill is None:
            _, why = self.capability("pick", otype, a.get("arm"))
            return self._result(ex, "failed", None, reason="policy_unavailable", phase="select_skill",
                                extra={"detail": why}, **kw)
        rec = self._last_reach.get(otype)
        reach = rec[0] if rec else None
        oid = a.get("object_id") or (reach.object_id if reach else None)
        arm = self._resolve_arm(a.get("arm"), reach)
        kw.update(arm=arm, object_id=oid)
        if not oid or self.world.object(oid) is None:
            return self._result(ex, "failed", skill, reason="grasp_missed", phase="select_skill",
                                extra={"detail": "no object bound (check_reachability first)"}, **kw)
        # stance check: the base must be where check_reachability judged it
        if rec is not None:
            _, _, (rx, ry, ryaw) = rec
            drift = math.hypot(p0.x - rx, p0.y - ry)
            dyaw = abs(math.degrees(coords.ang_diff(p0.yaw, ryaw)))
            if drift > self.cfg.stance_drift_m or dyaw > self.cfg.stance_drift_deg:
                return self._result(ex, "failed", skill, reason="base_moving", phase="stance_check",
                                    holding=False, extra={"drift_m": round(drift, 3), "drift_deg": round(dyaw, 1)},
                                    **kw)
        hands = self.world.hands()
        if hands.get(arm):
            return self._result(ex, "failed", skill, reason="hand_full", phase="select_skill", holding=False, **kw)
        exe = self.executor_for(skill)
        job = ManipJob("pick", oid, arm, skill.skill_id, epoch=self.gate.epoch)
        out = await self._run_executor(exe, job, h, skill)
        holding = self.world.hands().get(arm) == oid
        extra = {"phases": out.phases, **({"detail": out.detail} if out.detail else {})}
        if out.status == "succeeded":
            await self.clock.sleep(self.cfg.verify_hold_s)
            holding = self.world.hands().get(arm) == oid
            if not holding:
                return self._result(ex, "failed", skill, reason="grasp_failed", phase="verify", holding=False,
                                    extra=extra, **kw)
            return self._result(ex, "succeeded", skill, holding=True, phase="verify", extra=extra, **kw)
        if out.status == "cancelled":
            return self._result(ex, "cancelled", skill, reason=out.reason, phase=out.phase, holding=holding,
                                extra=extra, **kw)
        return self._result(ex, "timed_out" if out.reason == "timeout" else "failed", skill, reason=out.reason,
                            phase=out.phase, holding=holding, extra=extra, **kw)

    def _within_reach(self, x: float, y: float, z: float) -> bool:
        fwd, lat = self.reach.in_body_frame(x, y)
        ws = self.reach.ws
        return (ws.reach_fwd_m[0] <= fwd <= ws.reach_fwd_m[1] and abs(lat) <= ws.reach_lat_max_m
                and self.reach.shoulder_dist(fwd, lat, z) <= ws.arm_reach_m)

    async def _place(self, ex: Execution, h: ResultHandle, skill: SkillSpec | None):
        t0 = self.clock.now()
        a = ex.args
        otype = str(a.get("object_type"))
        p0 = self.world.robot_pose()
        kw = dict(action="place", object_type=otype, t0=t0, pose0=(p0.x, p0.y, p0.yaw))
        hands = self.world.hands()
        arm = a.get("arm") if a.get("arm") in ("left", "right") else None
        oid = a.get("object_id")
        if arm is None:
            arm = next((ar for ar, o in hands.items() if o and (o == oid or (oid is None and
                        (self.world.object(o).type == otype)))), None)
        held = hands.get(arm) if arm else None
        if oid is None:
            oid = held
        kw.update(arm=arm, object_id=oid)
        if skill is None:
            _, why = self.capability("place", otype, arm)
            return self._result(ex, "failed", None, reason="policy_unavailable", phase="select_skill",
                                holding=bool(held), extra={"detail": why}, **kw)
        if not held or held != oid:
            return self._result(ex, "failed", skill, reason="nothing_in_hand", phase="select_skill",
                                holding=bool(held), **kw)
        m = self.world.static_map()
        here = self.nav.at()[0] if self.nav is not None else self.world.keypoint_at(p0.x, p0.y)
        target = a.get("target")
        if target == "user":
            target = (m.lookup_keypoints().get("people", {}).get("user") or {}).get("deliver_to_surface")
        if not target:
            target = here if here in m.surfaces else None
            if target is None:
                return self._result(ex, "failed", skill, reason="no_surface_here", phase="select_skill",
                                    holding=True, **kw)
        kw["target"] = target
        if target not in m.surfaces:
            return self._result(ex, "failed", skill, reason="no_surface_here", phase="select_skill", holding=True,
                                extra={"detail": f"{target} is not a surface"}, **kw)
        if here != target:
            return self._result(ex, "failed", skill, reason="target_not_here", phase="select_skill", holding=True,
                                extra={"detail": f"at {here}; navigate to {target} first"}, **kw)
        spot = self.world.free_spot(target, oid, near=p0, within=self._within_reach)
        if spot is None:
            reason = "no_room_in_reach" if self.world.free_spot(target, oid) is not None else "no_room_on_surface"
            return self._result(ex, "failed", skill, reason=reason, phase="free_spot", holding=True, **kw)
        exe = self.executor_for(skill)
        job = ManipJob("place", oid, arm, skill.skill_id, spot=spot, target=target, epoch=self.gate.epoch)
        out = await self._run_executor(exe, job, h, skill)
        holding = self.world.hands().get(arm) == oid
        extra = {"phases": out.phases, "spot": [round(spot.x, 3), round(spot.y, 3), round(spot.z, 3)],
                 **({"detail": out.detail} if out.detail else {})}
        if out.status == "succeeded":
            await self.clock.sleep(self.cfg.verify_hold_s)
            o = self.world.object(oid)
            if o is None or o.where != target:
                return self._result(ex, "failed", skill, reason="object_dropped", phase="verify",
                                    holding=holding, surface=o.where if o else None, extra=extra, **kw)
            return self._result(ex, "succeeded", skill, holding=False, phase="verify", surface=target, extra=extra,
                                **kw)
        if out.status == "cancelled":
            return self._result(ex, "cancelled", skill, reason=out.reason, phase=out.phase, holding=holding,
                                extra=extra, **kw)
        return self._result(ex, "failed", skill, reason=out.reason, phase=out.phase, holding=holding, extra=extra,
                            **kw)

    async def _run_executor(self, exe: Any, job: ManipJob, h: ResultHandle, skill: SkillSpec):
        from .executors.kinematic_attach import ManipOutcome
        try:
            return await self.clock.wait_for(exe.run(job, h), skill.timeout_s())
        except asyncio.TimeoutError:
            try:
                await exe.cancel()
            except Exception:  # noqa: BLE001
                pass
            return ManipOutcome("failed", "timeout", self.world.hands().get(job.arm) == job.object_id, "execute")
