"""Idle wake rule: once a task is finished (every utterance answered, no own goal, no running
execution, no open question from the robot), passive perception/belief changes do not wake the
planner; user input, safety/capability events and execution results still do."""

from __future__ import annotations

import asyncio

import pytest

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
        rt._on_robot_event({"type": "capability_changed", "detail": "arm policy down"})
    finally:
        await stop(rt)
    assert "safety near_miss" in wakes and "capability changed" in wakes


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
