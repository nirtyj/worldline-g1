"""Where ManipulationService's executors come from: one registration point, keyed by the executor name a profile
lists (`config/profiles/<p>.yaml manipulation.executors`).

    name               backend (config/skills.yaml)   built by
    lite               lite                            KinematicAttachExecutor (in-process attach, lite profile)
    kinematic_attach   kinematic_attach                KinematicAttachExecutor (P1 attach/detach; STEPPING STONE)
    sonic_arm_script   sonic_arm_script                SonicArmScriptExecutor (stub until the body's arm script)
    groot_arms         groot                           services.executors.groot_arms:create (owner groot_rt;
                                                       experimental: N1.7 arm/hand chunks through the body `arm` op)
    groot_sonic        groot                           GrootSonicExecutor (the retired token route; a stub)

A factory takes an ExecutorContext and returns a ManipExecutor (api/services.py): `backend`, `name`,
`async run(job: ManipJob, handle) -> ManipOutcome`, `async cancel()`, `health() -> ServiceHealth`, and optionally
`close()` (G1Robot.shutdown calls it). A factory may be a callable or a "module:attr" string, imported only when a
profile asks for that executor; when the import fails the profile still builds and the executor reports itself
down (`UnavailableExecutor`), so its skills are rejected at CAPABILITY ("policy unavailable: ...") and the
object_type enum stays frozen.

Registering from elsewhere: `register_executor("my_exec", "pkg.mod:create", backend="groot")` before
robot.factory.build(); the profile then lists "my_exec".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from api.types import ServiceHealth

from .kinematic_attach import ManipJob, ManipOutcome


@dataclass
class ExecutorContext:
    """Everything an executor may need. `body` is the BodyPort (SonicBody: `.client` is wl-body's BodyClient);
    `world` is the WorldModel + SimControl (the ONLY ground-truth reader: success/failure checks go through it)."""
    name: str                                    # the profile's executor name
    world: Any
    body: Any
    clock: Any
    gate: Any                                    # services.common.HaltGate (halt epoch / latch)
    events: Any                                  # services.common.EventSink (emit(type, **fields))
    profile: Any = None                          # robot.profile.StackProfile
    manip_cfg: dict = field(default_factory=dict)   # config/g1.yaml `manipulation:`
    port_offset: int = 0
    extras: dict = field(default_factory=dict)


Factory = Callable[[ExecutorContext], Any]


@dataclass(frozen=True)
class Entry:
    backend: str
    factory: Factory | str


class UnavailableExecutor:
    """Stands in for an executor whose code is not importable here: always down, every run fails."""

    def __init__(self, name: str, backend: str, why: str):
        self.name = name
        self.backend = backend
        self.why = why

    def health(self) -> ServiceHealth:
        return ServiceHealth(False, "down", f"{self.name} not available: {self.why}")

    async def run(self, job: ManipJob, handle: Any) -> ManipOutcome:
        reason = "policy_unavailable" if self.backend == "groot" else "controller_unavailable"
        return ManipOutcome("failed", reason, False, "select_skill", detail=self.health().detail)

    async def cancel(self) -> None:
        return None


def _kinematic(ctx: ExecutorContext):
    from .kinematic_attach import KinematicAttachExecutor
    return KinematicAttachExecutor(ctx.world, ctx.clock, pick_phases=ctx.manip_cfg.get("pick_phases_s"),
                                   place_phases=ctx.manip_cfg.get("place_phases_s"), gate=ctx.gate, name=ctx.name,
                                   attach_mode="kinematic" if ctx.name == "lite" else "follow")


def _sonic_arm_script(ctx: ExecutorContext):
    from .sonic_arm_script import SonicArmScriptExecutor
    return SonicArmScriptExecutor()


def _groot_sonic(ctx: ExecutorContext):
    from .groot_sonic import GrootSonicExecutor
    return GrootSonicExecutor()


_REGISTRY: dict[str, Entry] = {
    "lite": Entry("lite", _kinematic),
    "kinematic_attach": Entry("kinematic_attach", _kinematic),
    "sonic_arm_script": Entry("sonic_arm_script", _sonic_arm_script),
    "groot_arms": Entry("groot", "services.executors.groot_arms:create"),
    "groot_sonic": Entry("groot", _groot_sonic),
}


def register_executor(name: str, factory: Factory | str, *, backend: str) -> None:
    _REGISTRY[name] = Entry(backend, factory)


def registered() -> dict[str, str]:
    """executor name -> registry backend."""
    return {n: e.backend for n, e in _REGISTRY.items()}


def backend_of(name: str) -> str:
    e = _REGISTRY.get(name)
    return e.backend if e is not None else name


def _resolve(ref: Factory | str) -> Factory:
    if callable(ref):
        return ref
    mod, _, attr = str(ref).partition(":")
    obj: Any = importlib.import_module(mod)
    for part in (attr or "create").split("."):
        obj = getattr(obj, part)
    return obj


def build_executor(ctx: ExecutorContext) -> Any:
    """The executor for ctx.name, or an UnavailableExecutor that says why not. Raises KeyError for a name that
    nobody registered (a profile typo should fail loudly at build time)."""
    e = _REGISTRY.get(ctx.name)
    if e is None:
        raise KeyError(f"unknown manipulation executor {ctx.name!r}; registered: {sorted(_REGISTRY)}")
    try:
        ex = _resolve(e.factory)(ctx)
    except (ImportError, AttributeError) as err:
        return UnavailableExecutor(ctx.name, e.backend, f"{e.factory} not importable ({err.__class__.__name__}: "
                                                         f"{err})")
    if getattr(ex, "backend", None) != e.backend:
        raise ValueError(f"executor {ctx.name!r} reports backend {getattr(ex, 'backend', None)!r}, "
                         f"registered as {e.backend!r}")
    return ex


__all__ = ["ExecutorContext", "Factory", "Entry", "UnavailableExecutor", "register_executor", "registered",
           "backend_of", "build_executor"]
