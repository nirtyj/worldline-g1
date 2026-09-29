"""KinematicAttachExecutor: the bring-up grasp (STEPPING STONE, PLAN §1.3 #11, §12.2).

No arm moves. The skill is a timed sequence of phases, and at the grasp phase the object is attached to the hand
through `SimControl.attach` (it follows the palm; LiteWorld does it in process, P1 needs the M2b `attach` op); at
the release phase `SimControl.detach` puts it at the chosen free spot. Every result says `executor:
"kinematic_attach"` (or `"lite"`) and `stepping_stone: true`, so the prompt shows [fallback], the UI an amber
badge and eval counts it as a fallback pass.

Cancel is honoured between phases (worst case one phase, <= ~1.2 s; a cancel during `grasp` still ends holding,
as in THOR); halt is honoured at the next 50 ms poll inside any phase (THOR drift fixed: place honours cancel in
every phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world.model import NotSupported

POLL_S = 0.05


@dataclass
class ManipJob:
    action: str                          # pick | place
    object_id: str
    arm: str
    skill_id: str
    spot: Any = None                     # place: api Pose3D of the object's centre on the target
    target: str | None = None
    epoch: int = 0                       # the runtime halt epoch at start (HaltGate.epoch)
    # the fence an executor that leases the body needs (groot_arms: session = execution_id, PLAN §6.6)
    execution_id: str = ""
    generation: int = 0
    control_epoch: int = 0
    object_type: str = ""
    skill: Any = None                    # api.skills.SkillSpec (prompt_template, max_duration_s, success, ...)


@dataclass
class ManipOutcome:
    status: str                          # succeeded | failed | cancelled
    reason: str | None = None
    holding: bool | None = None
    phase: str | None = None
    detail: str | None = None
    phases: list[dict] = field(default_factory=list)


class KinematicAttachExecutor:
    backend = "kinematic_attach"

    def __init__(self, world: Any, clock: Any, *, pick_phases: dict | None = None, place_phases: dict | None = None,
                 gate: Any = None, name: str = "kinematic_attach", attach_mode: str = "follow"):
        self.world = world
        self.clock = clock
        self.gate = gate
        self.name = name
        self.backend = "lite" if name == "lite" else "kinematic_attach"
        self.attach_mode = attach_mode
        self.pick_phases = dict(pick_phases or {"pregrasp": 1.2, "grasp": 1.2, "lift": 1.0})
        self.place_phases = dict(place_phases or {"lower": 1.2, "release": 0.8, "retract": 1.0})
        self._cancel = False

    async def cancel(self) -> None:
        self._cancel = True

    async def _phase(self, seconds: float, handle: Any, epoch: int) -> str | None:
        t_end = self.clock.now() + seconds
        while self.clock.now() < t_end:
            if self.gate is not None and self.gate.halted_since(epoch):
                return "halted"
            p = self.world.robot_pose()
            if p.fallen:
                return "fell"
            await self.clock.sleep(min(POLL_S, max(0.0, t_end - self.clock.now())))
        return None

    def _holding(self, job: ManipJob) -> bool:
        return self.world.hands().get(job.arm) == job.object_id

    async def run(self, job: ManipJob, handle: Any) -> ManipOutcome:
        phases = self.pick_phases if job.action == "pick" else self.place_phases
        done: list[dict] = []
        for i, (phase, secs) in enumerate(phases.items()):
            if i > 0 and (handle.cancel_requested or self._cancel):
                return ManipOutcome("cancelled", handle.cancel_reason or "cancelled", self._holding(job), phase,
                                    phases=done)
            t0 = self.clock.now()
            why = await self._phase(float(secs), handle, job.epoch)
            if why:
                return ManipOutcome("failed", why, self._holding(job), phase, phases=done)
            try:
                if job.action == "pick" and phase == "grasp":
                    self.world.attach(job.object_id, job.arm, self.attach_mode)
                if job.action == "place" and phase == "release":
                    self.world.detach(job.object_id, job.spot)
            except NotSupported as e:
                return ManipOutcome("failed", "controller_unavailable", self._holding(job), phase, detail=str(e),
                                    phases=done)
            except KeyError as e:
                return ManipOutcome("failed", "grasp_missed", False, phase, detail=f"unknown object {e}", phases=done)
            done.append({"phase": phase, "s": round(self.clock.now() - t0, 2)})
        return ManipOutcome("succeeded", None, self._holding(job), list(phases)[-1], phases=done)

    def health(self):
        from api.types import ServiceHealth
        caps = self.world.capabilities() if hasattr(self.world, "capabilities") else {}
        if caps.get("attach") and caps.get("detach"):
            return ServiceHealth(True, "ok", f"{self.name} (STEPPING STONE)")
        return ServiceHealth(False, "down", "P1 has no attach/detach op (M2b)")
