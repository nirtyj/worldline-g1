"""Every implementation matches its api/services.py Protocol: the members exist, async stays async, and each
Protocol parameter is accepted under the same name with the same kind (positional / keyword), with a default
wherever the Protocol has one. A drift between api/ and services/, world/ or robot/ fails here, not on the box.

Covered: RobotBridge (G1Robot on lite, the runtime's FakeRobot), the four services behind G1Robot, BodyPort
(LiteBody, SonicBody) and BodyOpHandle (BodyOp), ManipExecutor (lite/kinematic_attach, sonic_arm_script,
groot_sonic), WorldModel + SimControl (LiteWorld, IsaacGTWorldModel), RobotPoseLike, Localizer and FrameSource.
Isaac-backed classes are checked at class level (no P1 needed)."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

import api.services as S
from sim.clock import SimClock
from tests.contract.conftest import LITE_SCENE, lite_available

P = inspect.Parameter


def _members(proto: type) -> tuple[dict[str, Any], set[str]]:
    """(methods and properties, data attributes) the Protocol declares (its own body, not typing internals)."""
    methods: dict[str, Any] = {}
    attrs: set[str] = set()
    for klass in reversed(proto.__mro__):
        if klass in (object,) or klass.__module__ == "typing":
            continue
        attrs |= set(getattr(klass, "__annotations__", {}))
        for name, v in vars(klass).items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(v) or isinstance(v, property):
                methods[name] = v
    return methods, attrs - set(methods)


def _params(fn: Any) -> list[inspect.Parameter]:
    return [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]


def signature_problems(proto_fn: Any, impl_fn: Any, where: str) -> list[str]:
    out: list[str] = []
    if inspect.iscoroutinefunction(proto_fn) != inspect.iscoroutinefunction(impl_fn):
        out.append(f"{where}: async mismatch (protocol async={inspect.iscoroutinefunction(proto_fn)})")
    pp, ip = _params(proto_fn), _params(impl_fn)
    by_name = {p.name: p for p in ip}
    impl_var_kw = any(p.kind is P.VAR_KEYWORD for p in ip)
    impl_var_pos = any(p.kind is P.VAR_POSITIONAL for p in ip)
    proto_var_kw = any(p.kind is P.VAR_KEYWORD for p in pp)
    positional = [p for p in ip if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    pos_i = 0
    for p in pp:
        if p.kind in (P.VAR_KEYWORD, P.VAR_POSITIONAL):
            continue
        q = by_name.get(p.name)
        if p.kind is P.POSITIONAL_OR_KEYWORD:
            # callers pass it positionally at this index, or by name
            if pos_i < len(positional):
                if positional[pos_i].name != p.name:
                    out.append(f"{where}: positional #{pos_i} is {positional[pos_i].name!r}, protocol says {p.name!r}")
            elif not impl_var_pos:
                out.append(f"{where}: {p.name!r} cannot be passed positionally")
            pos_i += 1
        if q is None:
            if not impl_var_kw:
                out.append(f"{where}: missing parameter {p.name!r}")
            continue
        if p.kind is P.KEYWORD_ONLY and q.kind is P.POSITIONAL_ONLY:
            out.append(f"{where}: {p.name!r} must be accepted by keyword")
        if p.default is not P.empty and q.default is P.empty:
            out.append(f"{where}: {p.name!r} needs a default (the protocol lets callers omit it)")
    # the implementation may not require more than the protocol passes
    proto_names = {p.name for p in pp}
    for q in ip:
        if q.kind in (P.VAR_KEYWORD, P.VAR_POSITIONAL) or q.default is not P.empty or q.name in proto_names:
            continue
        if not proto_var_kw:
            out.append(f"{where}: implementation requires {q.name!r}, which the protocol never passes")
    return out


def problems(impl: Any, proto: type, *, instance: bool = True) -> list[str]:
    """What stops ``impl`` (an instance, or a class when instance=False) from meeting ``proto``."""
    cls = impl if inspect.isclass(impl) else type(impl)
    methods, attrs = _members(proto)
    out: list[str] = []
    for name in sorted(attrs):
        if instance and not hasattr(impl, name):
            out.append(f"{cls.__name__}.{name}: missing attribute")
    for name, pm in sorted(methods.items()):
        where = f"{cls.__name__}.{name}"
        im = inspect.getattr_static(cls, name, None)
        if im is None:
            if instance and hasattr(impl, name):          # a dataclass field or an instance attribute
                continue
            out.append(f"{where}: missing")
            continue
        if isinstance(pm, property):
            if not (isinstance(im, property) or not callable(im)):
                out.append(f"{where}: should be a property or attribute")
            continue
        if isinstance(im, (staticmethod, classmethod)):
            im = im.__func__
        if not callable(im):
            out.append(f"{where}: not callable")
            continue
        out += signature_problems(pm, im, where)
    return out


# ----------------------------------------------------------------------------------------------------------
# the checker itself
# ----------------------------------------------------------------------------------------------------------
def test_the_checker_catches_drift():
    class Proto:
        def a(self, x: int, *, y: int = 0) -> None: ...
        async def b(self) -> None: ...

    class Good:
        def a(self, x, *, y=1, z=None): ...
        async def b(self): ...

    class Bad:
        def a(self, z, *, y): ...          # wrong name, no default
        def b(self): ...                   # not async

    assert signature_problems(Proto.a, Good.a, "Good.a") == []
    msgs = signature_problems(Proto.a, Bad.a, "Bad.a") + signature_problems(Proto.b, Bad.b, "Bad.b")
    text = " | ".join(msgs)
    assert "positional #0 is 'z'" in text and "'y' needs a default" in text and "async mismatch" in text


# ----------------------------------------------------------------------------------------------------------
# the lite stack (instances)
# ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lite():
    why = lite_available()
    if why:
        pytest.skip(why)
    from robot.factory import build
    clock = SimClock(40.0)
    try:
        world, robot, frames = build("lite", LITE_SCENE, clock)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"no recorded house for {LITE_SCENE}: {e}")
    return world, robot, frames


def test_g1robot_is_a_robot_bridge(lite):
    _, robot, _ = lite
    assert isinstance(robot, S.RobotBridge)
    assert problems(robot, S.RobotBridge) == []


def test_the_runtime_fake_is_a_robot_bridge():
    from tests.unit.fakes_rt import FakeRobot
    robot = FakeRobot(SimClock(40.0))
    assert problems(robot, S.RobotBridge) == []


def test_services_match_their_protocols(lite):
    _, robot, _ = lite
    assert problems(robot.nav, S.NavigationService) == []
    assert problems(robot.manip, S.ManipulationService) == []
    assert problems(robot.obs, S.ObservationService) == []
    assert problems(robot.speech, S.SpeechService) == []
    for svc, proto in ((robot.nav, S.NavigationService), (robot.manip, S.ManipulationService),
                       (robot.obs, S.ObservationService), (robot.speech, S.SpeechService)):
        assert isinstance(svc, proto), (svc, proto)


def test_bodies_and_body_ops(lite):
    from robot.body_client import BodyOp, SonicBody
    _, robot, _ = lite
    assert problems(robot.body, S.BodyPort) == [] and isinstance(robot.body, S.BodyPort)
    assert problems(SonicBody, S.BodyPort, instance=False) == []
    assert SonicBody.name if hasattr(SonicBody, "name") else True

    async def mk():
        return BodyOp("go_to", "go_to-1", loop=asyncio.get_running_loop())
    op = asyncio.run(mk())
    assert problems(op, S.BodyOpHandle) == []


def test_manipulation_executors(lite):
    from services.executors import GrootSonicExecutor, SonicArmScriptExecutor
    _, robot, _ = lite
    ex = list(robot.executors.values())
    assert ex, "the lite profile loads the lite executor"
    for e in ex + [SonicArmScriptExecutor(robot.world, robot.body), GrootSonicExecutor()]:
        assert problems(e, S.ManipExecutor) == [], type(e).__name__


def test_world_models_and_sim_control(lite):
    from world.isaac_client import IsaacGTWorldModel
    from world.model import WorldLocalizer
    world, _, frames = lite
    assert problems(world, S.WorldModel) == [] and isinstance(world, S.WorldModel)
    assert problems(world, S.SimControl) == [] and isinstance(world, S.SimControl)
    assert problems(IsaacGTWorldModel, S.WorldModel, instance=False) == []
    assert problems(IsaacGTWorldModel, S.SimControl, instance=False) == []
    pose = world.robot_pose()
    assert problems(pose, S.RobotPoseLike) == [] and isinstance(pose, S.RobotPoseLike)
    assert problems(WorldLocalizer(world), S.Localizer) == []
    assert problems(frames, S.FrameSource) == []


def test_frame_sources_at_class_level():
    from world.frames import IsaacFrames, NoFrames
    assert problems(NoFrames, S.FrameSource, instance=False) == []
    assert problems(IsaacFrames, S.FrameSource, instance=False) == []


def test_world_model_reexports_are_the_api_protocols():
    import robot.body_client as bc
    import world.model as wm
    assert wm.WorldModel is S.WorldModel and wm.SimControl is S.SimControl and wm.Localizer is S.Localizer
    assert wm.NotSupported is S.NotSupported and bc.BodyPort is S.BodyPort
