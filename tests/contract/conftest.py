"""Contract tests: a RobotBridge backend must honour api/ (PLAN 10). Backends: the runtime's own
fake (always) and the world agent's lite robot (robot.factory, when its deps and house data exist).
Isaac profiles are behind ``-m box`` and are not collected here."""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

import pytest

from tests.unit.conftest import isolated_runs  # noqa: F401  (autouse: runs/ goes to a temp dir)

LITE_SCENE = os.environ.get("WL_LITE_SCENE", "procthor-train-38")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {k: pyfuncitem.funcargs[k] for k in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


def lite_available() -> str | None:
    """None when the lite backend can be built here, else why not."""
    try:
        import scipy  # noqa: F401
        from robot.factory import build  # noqa: F401
    except Exception as e:                      # the world agent's code or its deps are missing
        return f"robot.factory not importable: {e!r}"
    return None


def make_backend(name: str, clock: Any) -> Any:
    if name == "fake":
        from tests.unit.fakes_rt import FakeRobot
        return FakeRobot(clock)
    why = lite_available()
    if why:
        pytest.skip(why)
    from robot.factory import build
    try:
        _world, robot, _frames = build("lite", LITE_SCENE, clock)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"no recorded house for {LITE_SCENE}: {e}")
    return robot


BACKENDS = ["fake", "lite"]
