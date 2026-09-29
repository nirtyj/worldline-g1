"""The skill registry contract (PLAN 6.4; doc 13, 49).

The ``object_type`` tool enum is the union of ``object_types`` over the skills
LOADED at session start, frozen for the session (Invariant 9). Health never
changes the enum; an unhealthy skill is rejected at the CAPABILITY stage with
``policy unavailable: ...``.

``@vocab:pickupable`` in ``object_types`` expands to the FIXED THOR pickupable
vocabulary (``config/vocab/pickupable_types.yaml``), never to the scene catalog,
so the enum never tells the planner which types exist (``missing_object``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Protocol, runtime_checkable

from .types import ServiceHealth

Backend = Literal["groot", "sonic_arm_script", "kinematic_attach", "lite"]
SkillLabel = Literal["target", "experimental", "stepping_stone"]
VOCAB_PICKUPABLE = "@vocab:pickupable"

# backend -> the executor name results carry
EXECUTOR_OF_BACKEND: dict[str, str] = {"groot": "groot_sonic", "sonic_arm_script": "sonic_arm_script",
                                       "kinematic_attach": "kinematic_attach", "lite": "lite"}

# Selection order by profile (PLAN 6.4)
BACKEND_ORDER: dict[str, tuple[str, ...]] = {
    "full": ("groot", "sonic_arm_script"),
    "sonic": ("sonic_arm_script",),
    "bringup": ("kinematic_attach",),
    "lite": ("lite",),
    "real_g1": ("groot",),
}


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str                                  # "groot.pick.bottle.cloudwalk.v0" | "sonic.script.pick.v0" | ...
    action: Literal["pick", "place"]
    object_types: tuple[str, ...]                  # may contain "@vocab:pickupable"
    arms: tuple[str, ...] = ("left", "right")
    backend: Backend = "lite"
    label: SkillLabel = "target"
    status: Literal["available", "unofficial", "planned"] = "available"
    embodiment_tag: str | None = None              # "UNITREE_G1_SONIC"
    checkpoint: str | None = None
    policy_port: int | None = None
    prompt_template: str = ""                      # "grab the bottle" | "pick up the {label} with the {arm} hand"
    hand_type: Literal["dex3", "inspire", "umi"] = "dex3"   # != dex3 -> UI flag "hand_mismatch"
    initial_token: str | None = None
    initial_blend_s: float = 1.0
    stance: dict[str, Any] = field(default_factory=dict)  # {stand_off_m, lateral_m, yaw_to_object, tol: [m, deg]}
    licence: str = ""
    action_horizon: int = 40
    control_hz: float = 50.0
    replan_hz: float = 2.5
    token_abs_limit: float = 1.25
    max_duration_s: float = 25.0
    success: dict[str, Any] = field(default_factory=dict)  # {kind: gt_lifted, min_lift_m, max_dist_m, hold_s}

    @property
    def executor(self) -> str:
        return EXECUTOR_OF_BACKEND.get(self.backend, self.backend)

    @property
    def stepping_stone(self) -> bool:
        return self.label == "stepping_stone"

    @property
    def hand_mismatch(self) -> bool:
        return self.hand_type != "dex3"

    def handles(self, object_type: str) -> bool:
        return object_type in self.object_types

    def timeout_s(self) -> float:
        """Tool timeout (PLAN 1.3 #22): skill.max_duration_s + 10."""
        return self.max_duration_s + 10.0


@runtime_checkable
class SkillRegistry(Protocol):
    def skills(self) -> list[SkillSpec]: ...
    def loaded_object_types(self) -> list[str]: ...          # tool enum, frozen per session
    def healthy(self, skill_id: str) -> ServiceHealth: ...   # CAPABILITY stage only; never changes the enum
    def select(self, action: str, object_type: str, arm: str | None) -> SkillSpec | None: ...


def expand_types(types: Iterable[str], vocab: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for t in types:
        items = list(vocab) if t == VOCAB_PICKUPABLE else [t]
        for x in items:
            if x not in out:
                out.append(x)
    return tuple(out)


class StaticSkillRegistry:
    """A registry over a fixed list of skills; ``backend_order`` is the profile's selection order.

    ``health_fn(skill) -> ServiceHealth`` is consulted by ``healthy`` only (CAPABILITY)."""

    def __init__(self, skills: Iterable[SkillSpec], *, vocab: Iterable[str] = (),
                 backend_order: Iterable[str] = ("lite",),
                 health_fn: Callable[[SkillSpec], ServiceHealth] | None = None,
                 only: Iterable[str] | None = None) -> None:
        vocab = tuple(vocab)
        order = tuple(backend_order)
        loaded = [replace(s, object_types=expand_types(s.object_types, vocab)) for s in skills
                  if s.backend in order and s.status != "planned"]
        if only is not None:                              # e.g. --skills groot_only
            allowed = set(only)
            loaded = [s for s in loaded if s.backend in allowed]
        self._skills = loaded
        self._order = order
        self._health_fn = health_fn
        self._types = sorted({t for s in self._skills for t in s.object_types})   # frozen here

    def skills(self) -> list[SkillSpec]:
        return list(self._skills)

    def loaded_object_types(self) -> list[str]:
        return list(self._types)

    def get(self, skill_id: str) -> SkillSpec | None:
        return next((s for s in self._skills if s.skill_id == skill_id), None)

    def healthy(self, skill_id: str) -> ServiceHealth:
        s = self.get(skill_id)
        if s is None:
            return ServiceHealth(False, "unknown", f"no skill {skill_id}")
        if self._health_fn is None:
            return ServiceHealth(True, "ok")
        return self._health_fn(s)

    def candidates(self, action: str, object_type: str, arm: str | None = None) -> list[SkillSpec]:
        """Every skill that handles this call, in the profile's backend order."""
        rank = {b: i for i, b in enumerate(self._order)}
        found = [s for s in self._skills if s.action == action and s.handles(object_type)
                 and (arm is None or arm in s.arms)]
        return sorted(found, key=lambda s: rank.get(s.backend, len(rank)))

    def select(self, action: str, object_type: str, arm: str | None) -> SkillSpec | None:
        c = self.candidates(action, object_type, arm)
        return c[0] if c else None

    def select_healthy(self, action: str, object_type: str, arm: str | None) -> tuple[SkillSpec | None, str]:
        """The first healthy candidate, else (None, why) for a CAPABILITY rejection."""
        why = ""
        for s in self.candidates(action, object_type, arm):
            h = self.healthy(s.skill_id)
            if h.ok:
                return s, ""
            why = why or f"{s.skill_id}: {h.detail or h.state}"
        return None, why


__all__ = ["Backend", "SkillLabel", "VOCAB_PICKUPABLE", "EXECUTOR_OF_BACKEND", "BACKEND_ORDER", "SkillSpec",
           "SkillRegistry", "expand_types", "StaticSkillRegistry"]
