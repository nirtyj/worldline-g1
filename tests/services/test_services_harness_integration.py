"""The runtime's real harness (agent/harness.py Runtime) driving the lite G1Robot built by robot.factory: the F1 fetch
on recorded MolmoSpaces houses with a deterministic planner (the runtime agent's PolicyBrain, imported read-only).
This is the seam test between the runtime (agent/) and the world agent's robot/services/world stack."""

import asyncio
import time

import pytest

pytest.importorskip("agent.harness")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import agent.episodes
    import agent.memory
    import agent.procedures
    monkeypatch.setattr(agent.episodes, "ROOT", tmp_path / "episodes")
    monkeypatch.setattr(agent.memory, "ROOT", tmp_path / "memory")
    monkeypatch.setattr(agent.procedures, "STORE", tmp_path / "procedures.json")
    return tmp_path


def _rows(rt, type_):
    return [r for r in rt.tracer.rows if r["type"] == type_]


async def _fetch(house: str, want: str, oid: str, wall_s: float = 90.0):
    from agent.harness import Runtime
    from robot.factory import build
    from sim.clock import SimClock
    from sim.log import EventLog
    from tests.unit.fakes_rt import FakeUser, PolicyBrain
    clock = SimClock(40.0)
    log = EventLog(clock)
    world, robot, _ = build("lite", house, clock, log)
    user = FakeUser(clock)
    rt = Runtime(robot, user, PolicyBrain(want), clock)
    user.say(f"bring me the {want.replace('_', ' ')}")
    task = asyncio.ensure_future(rt.run())
    t0 = time.monotonic()
    try:
        while not _rows(rt, "delivered"):
            if task.done():
                task.result()
                raise AssertionError("runtime stopped")
            if time.monotonic() - t0 > wall_s:
                tail = [f"{r['type']}:{r.get('tool') or r.get('why') or ''}" for r in rt.tracer.rows[-30:]]
                raise AssertionError(f"no delivery; last rows {tail}")
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        await robot.shutdown()
    return rt, world, robot


def test_f1_on_house_40_alarm_clock_on_the_dresser(isolated):
    from api.results import validate_envelope

    async def main():
        rt, world, robot = await _fetch("procthor-train-40", "alarm_clock", "alarm_clock_1")
        assert world.object("alarm_clock_1").where == world.map.user_surface
        for e in rt.history:
            if e.result is not None:
                assert e.result.observation_id, e.tool_name
                assert validate_envelope(e.result) == [], (e.tool_name, validate_envelope(e.result))
        tools = [e.tool_name for e in rt.history]
        assert "check_reachability" in tools and "observe" in tools
        execs = {e.result.data.get("executor") for e in rt.history if e.result is not None and e.result.data.get("executor")}
        assert execs <= {"lite"}
    asyncio.run(main())


def test_f1_on_house_38_needs_a_reposition(isolated):
    async def main():
        rt, world, robot = await _fetch("procthor-train-38", "alarm_clock", "alarm_clock_1")
        assert world.object("alarm_clock_1").where == world.map.user_surface
        navs = [e for e in rt.history if e.tool_name == "navigate"]
        assert any(e.action == "reposition" and e.status == "succeeded" for e in navs)
    asyncio.run(main())
