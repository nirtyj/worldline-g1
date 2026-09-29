"""The skill registry (PLAN §6.4; doc §13/§49): config/skills.yaml -> api.skills.StaticSkillRegistry.

The object_type tool enum is the union of object_types over the skills whose BACKEND the profile loads, expanded
with the FIXED pickupable vocabulary, and frozen at session start (Invariant 9). Health never changes the enum; it
only decides CAPABILITY rejections (`policy unavailable: ...`) and which backend runs.

Backend health in M2a (honest, per backend):
    lite               ok (pure Python)
    kinematic_attach   ok only when the world's SimControl can attach AND detach (LiteWorld: yes; P1: only when it
                       lists the M2b ops), else down "P1 has no attach/detach op (M2b)"
    sonic_arm_script   planned: needs the BodyServer arm_script op + attach (M3)
    groot              down: needs BodyServer vla_start + a PolicyServer (M4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from api.skills import EXECUTOR_OF_BACKEND, SkillSpec, StaticSkillRegistry
from api.types import ServiceHealth
from world.vocab import pickupable_types

SKILLS_FILE = Path(__file__).resolve().parents[1] / "config" / "skills.yaml"
BACKEND_OF_EXECUTOR = {v: k for k, v in EXECUTOR_OF_BACKEND.items()}


def load_skill_specs(path: str | Path | None = None) -> list[SkillSpec]:
    import yaml
    data = yaml.safe_load(Path(path or SKILLS_FILE).read_text()) or {}
    out = []
    fields = set(SkillSpec.__dataclass_fields__)
    for s in data.get("skills", []):
        kw = {k: v for k, v in s.items() if k in fields}
        kw["object_types"] = tuple(kw.get("object_types") or ())
        kw["arms"] = tuple(kw.get("arms") or ("left", "right"))
        out.append(SkillSpec(**kw))
    return out


def backends_for(executors: Iterable[str]) -> tuple[str, ...]:
    """Profile executor names (lite, kinematic_attach, sonic_arm_script, groot_sonic) -> registry backends."""
    return tuple(BACKEND_OF_EXECUTOR.get(e, e) for e in executors)


def backend_health(world: Any, extra: dict[str, Callable[[], ServiceHealth]] | None = None
                   ) -> Callable[[SkillSpec], ServiceHealth]:
    extra = dict(extra or {})

    def fn(s: SkillSpec) -> ServiceHealth:
        if s.backend in extra:
            return extra[s.backend]()
        if s.backend == "lite":
            return ServiceHealth(True, "ok", "lite (simulated)")
        if s.backend == "kinematic_attach":
            caps = world.capabilities() if hasattr(world, "capabilities") else {}
            if caps.get("attach") and caps.get("detach"):
                return ServiceHealth(True, "ok", "kinematic attach (STEPPING STONE)")
            return ServiceHealth(False, "down", "P1 has no attach/detach op (M2b)")
        if s.backend == "sonic_arm_script":
            return ServiceHealth(False, "planned", "needs the BodyServer arm_script op and a GT attach (M3)")
        if s.backend == "groot":
            return ServiceHealth(False, "down", "GR00T executor not available (needs BodyServer vla_start and a "
                                                "PolicyServer on %s; M4)" % (s.policy_port or 5550))
        return ServiceHealth(False, "unknown", f"backend {s.backend}")
    return fn


def build_registry(executors: Iterable[str], world: Any = None, *, path: str | Path | None = None,
                   only: Iterable[str] | None = None,
                   health: dict[str, Callable[[], ServiceHealth]] | None = None) -> StaticSkillRegistry:
    return StaticSkillRegistry(load_skill_specs(path), vocab=sorted(pickupable_types()),
                               backend_order=backends_for(executors), health_fn=backend_health(world, health),
                               only=backends_for(only) if only is not None else None)
