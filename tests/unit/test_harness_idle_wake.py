"""Idle wake rule: once a task is finished (every utterance answered, no own goal, no running
execution, no open question from the robot), passive perception/belief changes do not wake the
planner; user input, safety events and execution results still do.

Capability changes (robot/health.py capability_changed) wake it only when they matter: a change to
UNSAFE (safety), a running or not yet started execution whose tool the change affects, or something
still open (a request, an own goal, a question we asked). Idle with nothing open they are logged and
recorded only: the live sim's RTF flapping at the DEGRADED floor (2026-09-29 stance-fix run) woke the
planner 59 times in 4 minutes after the answer."""

from __future__ import annotations

import asyncio

import pytest

from agent.state import ActionHandle
from api.tools import ToolCall
from tests.unit.conftest import keep_running, run_until, stop
from tests.unit.fakes_rt import ScriptBrain
from tests.unit.test_harness_flows import make


def _spy_wakes(rt):
    wakes: list[str] = []
    orig = rt._wake

    def spy(why: str) -> None:
        wakes.append(why)
        orig(why)

    rt._wake = spy
    return wakes


async def _bump_belief(rt, n: int = 4) -> None:
    """Make the next n observation-loop refreshes change belief (as a System 1 observation
    that moved an object would): the loop only compares revisions around its own refresh."""
    orig = rt._refresh_observation
    left = [n]

    def refresh(*a, **kw):
        out = orig(*a, **kw)
        if left[0] > 0:
            left[0] -= 1
            rt.belief._bump()
        return out

    rt._refresh_observation = refresh
    for _ in range(40):
        if left[0] == 0:
            break
        await asyncio.sleep(0.05)
    assert left[0] == 0, "the observation loop did not refresh"


async def _finished_task(line: str = "I am at the bowl."):
    brain = ScriptBrain([ToolCall("speak", {"text": line})], kinds={"just go to the bowl": "request"})
    rt, robot, user, _ = make(brain)
    wakes = _spy_wakes(rt)
    user.say("just go to the bowl")
    await run_until(rt, lambda: line in robot.said, wall_s=10, what="done line")
    await asyncio.sleep(0.5)                 # let the think loop settle (quiet rule)
    return rt, robot, user, brain, wakes


@pytest.mark.asyncio
async def test_no_planner_call_after_a_finished_task_when_only_observations_arrive():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        assert not rt._belief_wake_wanted()
        calls0, wakes0 = len(brain.contexts), len(wakes)
        await _bump_belief(rt)
        await asyncio.sleep(0.3)
    finally:
        await stop(rt)
    assert "perception changed belief" not in wakes[wakes0:]
    assert len(brain.contexts) == calls0


@pytest.mark.asyncio
async def test_a_new_utterance_still_wakes_the_planner_after_a_finished_task():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        calls0 = len(brain.contexts)
        user.say("what is on the table?")
        await keep_running(rt, lambda: len(brain.contexts) > calls0, wall_s=10, what="planner asked")
    finally:
        await stop(rt)
    assert any(w.startswith("utterance") for w in wakes)


@pytest.mark.asyncio
async def test_a_safety_event_still_wakes_the_planner_after_a_finished_task():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        calls0 = len(brain.contexts)
        rt._on_robot_event({"type": "safety_event", "kind": "near_miss"})
        await keep_running(rt, lambda: len(brain.contexts) > calls0, wall_s=10, what="planner asked")
        rt._on_robot_event(_cap("navigation", ok=False, state="unsafe", detail="sim below real time: UNSAFE"))
    finally:
        await stop(rt)
    assert "safety near_miss" in wakes and "capability changed" in wakes
    row = [r for r in rt.tracer.rows if r["type"] == "capability_changed"][-1]
    assert row["wake"] == "safety" and "UNSAFE" in (rt._note or "")


@pytest.mark.asyncio
async def test_an_open_question_from_the_robot_still_allows_a_belief_change_wake():
    rt, robot, user, brain, wakes = await _finished_task("Which bowl do you mean?")
    try:
        assert rt._question_open() and rt._belief_wake_wanted()
        wakes0 = len(wakes)
        await _bump_belief(rt)
        await keep_running(rt, lambda: "perception changed belief" in wakes[wakes0:], wall_s=10,
                           what="belief wake")
    finally:
        await stop(rt)


def _cap(capability: str = "manipulation", *, ok: bool = False, state: str = "degraded", detail: str = "") -> dict:
    """A HealthMonitor capability_changed event (robot/health.py)."""
    was = {"ok": True, "state": "ok"} if not ok else {"ok": False, "state": "degraded"}
    what = f"{capability} unavailable: {detail or state}" if not ok else f"{capability} available again"
    return {"type": "capability_changed", "capability": capability, "ok": ok, "state": state, "detail": what,
            "was": was}


def _flapping(n: int) -> list[dict]:
    """The live pattern: manipulation + sim going degraded and back, n times."""
    out = []
    for i in range(n):
        down = i % 2 == 0
        for cap in ("manipulation", "sim"):
            out.append(_cap(cap, ok=not down, state="degraded" if down else "ok",
                            detail="sim below real time: DEGRADED, rtf_5s 0.93"))
    return out


@pytest.mark.asyncio
async def test_flapping_capabilities_after_a_finished_task_call_no_planner():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        assert not rt._belief_wake_wanted()
        calls0, wakes0 = len(brain.contexts), len(wakes)
        for ev in _flapping(20):
            robot.emit(ev)                    # through the robot-event queue, as HealthMonitor sends them
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
    finally:
        await stop(rt)
    assert len(brain.contexts) == calls0 and "capability changed" not in wakes[wakes0:]
    rows = [r for r in rt.tracer.rows if r["type"] == "capability_changed"]
    assert len(rows) == 40 and all(r["wake"] is None for r in rows)          # still logged, never a wake
    assert sum(1 for e in rt.state.events if e.type == "capability_changed") == 40


@pytest.mark.asyncio
async def test_flapping_capabilities_with_no_task_at_all_call_no_planner():
    brain = ScriptBrain([])
    rt, robot, user, _ = make(brain)
    rt.persona.propose = lambda *a, **k: None         # no own goal either: nothing at all is open
    wakes = _spy_wakes(rt)
    await run_until(rt, lambda: True)
    try:
        await asyncio.sleep(0.2)
        calls0 = len(brain.contexts)
        for ev in _flapping(10):
            robot.emit(ev)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
    finally:
        await stop(rt)
    assert len(brain.contexts) == calls0 and "capability changed" not in wakes


@pytest.mark.asyncio
async def test_a_capability_drop_under_a_running_manipulate_still_wakes_the_planner():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        calls0 = len(brain.contexts)
        h = ActionHandle("man-000099", rt.task.intent_version, "manipulate", {"action": "pick"}, frozenset({"body"}))
        rt.actions[h.entry_id] = h           # started (no result yet)
        rt._on_robot_event(_cap("manipulation", detail="sim below real time: DEGRADED, rtf_5s 0.87"))
        await keep_running(rt, lambda: len(brain.contexts) > calls0, wall_s=10, what="planner asked")
    finally:
        rt.actions.pop("man-000099", None)
        await stop(rt)
    row = [r for r in rt.tracer.rows if r["type"] == "capability_changed"][-1]
    assert row["wake"] == "running manipulate" and "capability changed" in wakes
    assert "DEGRADED" in (brain.contexts[calls0].note or "")


@pytest.mark.asyncio
async def test_a_capability_change_that_misses_the_running_tool_does_not_wake():
    rt, robot, user, brain, wakes = await _finished_task()
    try:
        calls0 = len(brain.contexts)
        h = ActionHandle("nav-000099", rt.task.intent_version, "navigate", {"location": "kitchen"}, frozenset({"body"}))
        rt.actions[h.entry_id] = h
        rt._on_robot_event(_cap("manipulation"))             # DEGRADED rejects manipulate, not a running walk
        rt._on_robot_event(_cap("sim"))
        await asyncio.sleep(0.3)
        assert len(brain.contexts) == calls0
        rt._on_robot_event(_cap("sim", state="unsafe"))      # UNSAFE stops body tools: it matters (and is safety)
        await keep_running(rt, lambda: len(brain.contexts) > calls0, wall_s=10, what="planner asked")
    finally:
        rt.actions.pop("nav-000099", None)
        await stop(rt)
    assert [r["wake"] for r in rt.tracer.rows if r["type"] == "capability_changed"] == [None, None, "safety"]


@pytest.mark.asyncio
async def test_an_open_question_still_hears_capability_changes():
    rt, robot, user, brain, wakes = await _finished_task("Which bowl do you mean?")
    try:
        calls0 = len(brain.contexts)
        rt._on_robot_event(_cap("manipulation", ok=True, state="ok"))
        await keep_running(rt, lambda: len(brain.contexts) > calls0, wall_s=10, what="planner asked")
    finally:
        await stop(rt)
    assert [r["wake"] for r in rt.tracer.rows if r["type"] == "capability_changed"] == ["open"]
