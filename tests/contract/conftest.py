"""Contract tests: a RobotBridge backend must honour api/ (PLAN 10). Backends:

    fake    the runtime's own fake (always)
    lite    the world agent's lite robot (robot.factory, when its deps and house data exist)
    sonic   the live Isaac stack: P1 wl-isaac + P3 wl-body (+ P2 the SONIC deploy behind it), built with
            robot.factory.build("sonic", ...) on the ports of WL_PORT_OFFSET. Marked `box`: deselected by default
            (pyproject addopts), and skipped cleanly unless WL_BOX=1 even when selected. Run on the box with
                WL_BOX=1 WL_PORT_OFFSET=<n> .venv-rt/bin/python -m pytest -m box tests/contract
            It moves the real robot: the stack must be up (scripts/m1_up.sh or m2_up.sh), standing, in a house.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

import pytest

from tests.unit.conftest import isolated_runs  # noqa: F401  (autouse: runs/ goes to a temp dir)

LITE_SCENE = os.environ.get("WL_LITE_SCENE", "procthor-train-38")
BOX_SCENE = os.environ.get("WL_BOX_SCENE")          # default: the scene P1 reports (house_id)
_LIVE: list[tuple[Any, Any]] = []                   # (world, robot) built on the box, closed after each test
_P1: dict[str, Any] = {}                            # what P1's ping said (house_id)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {k: pyfuncitem.funcargs[k] for k in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


@pytest.fixture(autouse=True)
def _close_live_backends():
    yield
    while _LIVE:
        world, robot = _LIVE.pop()
        for fn in (getattr(getattr(robot, "monitor", None), "stop", None), getattr(robot.body, "close", None),
                   getattr(world, "close", None)):
            try:
                if callable(fn):
                    fn()
            except Exception:  # noqa: BLE001
                pass


def lite_available() -> str | None:
    """None when the lite backend can be built here, else why not."""
    try:
        import scipy  # noqa: F401
        from robot.factory import build  # noqa: F401
    except Exception as e:                      # the world agent's code or its deps are missing
        return f"robot.factory not importable: {e!r}"
    return None


def box_available() -> str | None:
    """None when the live stack may be used (WL_BOX=1 and P1 answers ping), else why not. Asked once per session."""
    if "why" not in _P1:
        _P1["why"] = _box_available()
    return _P1["why"]


def _box_available() -> str | None:
    if os.environ.get("WL_BOX") != "1":
        return "live stack only with WL_BOX=1 (and WL_PORT_OFFSET) on the box"
    why = lite_available()
    if why:
        return why
    try:
        from body.config import ep, ports
        from body.p1_client import P1Error, P1Rpc  # noqa: F401  (tests may read P1 directly; the runtime may not)
        off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
        rpc = P1Rpc(ep(ports(off)["p1_rep"]), timeout_s=2.0)
        try:
            rep = rpc.call("ping", timeout_s=2.0)
        finally:
            rpc.close()
        if not rep.get("ok", True):
            return f"P1 ping failed: {rep.get('error')}"
        _P1["house_id"] = rep.get("house_id")
    except Exception as e:  # noqa: BLE001
        return f"P1 not reachable on offset {os.environ.get('WL_PORT_OFFSET', '0')}: {e!r}"
    return None


def clock_for(name: str):
    """lite/fake run on a fast sim clock; the live stack runs on wall time (SONIC is wall-clock bound)."""
    from sim.clock import SimClock
    return SimClock(1.0 if name == "sonic" else 40.0)


def make_backend(name: str, clock: Any) -> Any:
    if name == "fake":
        from tests.unit.fakes_rt import FakeRobot
        return FakeRobot(clock)
    if name == "sonic":
        why = box_available()
        if why:
            pytest.skip(why)
        from robot.factory import build
        world, robot, _frames = build("sonic", BOX_SCENE or _P1.get("house_id"), clock, frames=None)
        _LIVE.append((world, robot))
        return robot
    why = lite_available()
    if why:
        pytest.skip(why)
    from robot.factory import build
    try:
        _world, robot, _frames = build("lite", LITE_SCENE, clock)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"no recorded house for {LITE_SCENE}: {e}")
    return robot


BACKENDS = ["fake", "lite", pytest.param("sonic", marks=pytest.mark.box)]
