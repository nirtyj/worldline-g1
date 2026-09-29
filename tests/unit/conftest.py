"""Shared fixtures for runtime unit tests: runs/ goes to a temp dir, and a helper runs the
harness against the fake robot until a condition holds."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run ``async def`` tests with asyncio.run (no pytest-asyncio dependency)."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {k: pyfuncitem.funcargs[k] for k in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Episodes, spatial memory and the procedural graph never touch the repo's runs/."""
    import agent.episodes
    import agent.memory
    import agent.procedures
    monkeypatch.setattr(agent.episodes, "ROOT", tmp_path / "episodes")
    monkeypatch.setattr(agent.memory, "ROOT", tmp_path / "memory")
    monkeypatch.setattr(agent.procedures, "STORE", tmp_path / "procedures.json")
    yield tmp_path


async def run_until(rt: Any, cond: Callable[[], bool], wall_s: float = 10.0, what: str = "") -> None:
    """Run the runtime's tasks until ``cond()`` holds (checked every 10 ms of wall time)."""
    task = asyncio.ensure_future(rt.run())
    rt._test_task = task
    t0 = time.monotonic()
    try:
        while not cond():
            if task.done():
                task.result()                 # re-raise a crash
                raise AssertionError("runtime stopped")
            if time.monotonic() - t0 > wall_s:
                rows = [r["type"] + ":" + str(r.get("tool") or r.get("text") or r.get("why") or "")
                        for r in rt.tracer.rows[-40:]]
                raise AssertionError(f"timed out waiting for {what or cond}; last rows: {rows}")
            await asyncio.sleep(0.01)
    finally:
        pass


async def keep_running(rt: Any, cond: Callable[[], bool], wall_s: float = 10.0, what: str = "") -> None:
    """After run_until started the runtime: wait for another condition."""
    t0 = time.monotonic()
    task = rt._test_task
    while not cond():
        if task.done():
            task.result()
            raise AssertionError("runtime stopped")
        if time.monotonic() - t0 > wall_s:
            rows = [r["type"] + ":" + str(r.get("tool") or r.get("text") or r.get("why") or "")
                    for r in rt.tracer.rows[-40:]]
            raise AssertionError(f"timed out waiting for {what or cond}; last rows: {rows}")
        await asyncio.sleep(0.01)


async def stop(rt: Any) -> None:
    task = getattr(rt, "_test_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
