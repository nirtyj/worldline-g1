"""The runtime on each backend, end to end (PLAN 10 "test_lite_f1_delivers"): a scripted F1 fetch
produces "delivered" and "place_learned" rows (the renamed literals still fire), every result is a
valid envelope with an observation id, and a stop mid-walk halts first and acknowledges once."""

from __future__ import annotations

import asyncio

import pytest

from agent.harness import Runtime
from api.results import validate_envelope
from sim.clock import SimClock
from tests.contract.conftest import LITE_SCENE, lite_available
from tests.unit.conftest import keep_running, run_until, stop
from api.tools import ToolCall
from tests.unit.fakes_rt import FakeRobot, FakeUser, PolicyBrain, ScriptBrain

SPEED = 40.0
CANDIDATES = ("alarm_clock", "apple", "book", "mug", "bottle", "remote_control")


def build(backend: str):
    clock = SimClock(SPEED)
    if backend == "fake":
        return clock, FakeRobot(clock, nav_s_per_m=2.0), "alarm_clock"
    why = lite_available()
    if why:
        pytest.skip(why)
    from robot.factory import build as factory_build
    try:
        world, robot, _ = factory_build("lite", LITE_SCENE, clock)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"no recorded house for {LITE_SCENE}: {e}")
    truth = world.truth()
    objs = truth.get("objects") if isinstance(truth, dict) else getattr(truth, "objects", {})
    surfaces = robot.lookup_keypoints()["surfaces"]
    want = next((t for t in CANDIDATES for o in objs.values()
                 if o.get("type") == t and o.get("where") in surfaces), None)
    if want is None:
        pytest.skip("no pickable candidate on a surface in this house")
    return clock, robot, want


async def _one_fetch(backend: str):
    clock, robot, want = build(backend)
    user = FakeUser(clock)
    brain = PolicyBrain(want)
    rt = Runtime(robot, user, brain, clock)
    user.say(f"bring me the {want.replace('_', ' ')}")
    try:
        await run_until(rt, lambda: any(r["type"] == "delivered" for r in rt.tracer.rows) or brain.want is None,
                        wall_s=60, what="delivered or gave up")
        await keep_running(rt, lambda: not rt.speech.busy() and not rt.actions, wall_s=20, what="settled")
    finally:
        await stop(rt)
    return rt, brain


@pytest.mark.parametrize("backend", ["fake", "lite"])
async def test_f1_delivers(backend):
    # One run, no retries: the lite body integrates over sim time (M2a), so a fetch either delivers or the
    # contract is broken.
    rt, brain = await _one_fetch(backend)
    for e in rt.history:
        if e.result is None:
            continue
        assert validate_envelope(e.result) == [], (e.execution_id, validate_envelope(e.result))
        assert e.result.observation_id, e.execution_id
    types = [r["type"] for r in rt.tracer.rows]
    assert "delivered" in types and "place_learned" in types, [
        (e.execution_id, e.status, e.result.summary if e.result else None) for e in rt.history
        if e.tool_name in ("manipulate", "check_reachability")]
    picks = [e for e in rt.history if e.tool_name == "manipulate" and e.action == "pick" and e.status == "succeeded"]
    places = [e for e in rt.history if e.tool_name == "manipulate" and e.action == "place" and e.status == "succeeded"]
    assert picks and places
    # every STEPPING STONE result is labelled (G11): lite executors are fallbacks
    for e in picks + places:
        assert e.result.data.get("executor") and e.result.data.get("skill")
        assert "fallback" in e.result.summary


@pytest.mark.parametrize("backend", ["fake", "lite"])
async def test_stop_mid_walk_halts_first_and_acks_once(backend):
    clock, robot, want = build(backend)
    m = robot.lookup_keypoints()
    dist = {b if a == "start" else a: d for a, b, d in m["edges"] if "start" in (a, b)}
    far = max(dist, key=dist.get)
    user = FakeUser(clock)
    brain = ScriptBrain([ToolCall("navigate", {"location": far})])
    rt = Runtime(robot, user, brain, clock)
    user.say("go over there")
    try:
        await run_until(rt, lambda: any(e.tool_name == "navigate" and not e.finished for e in rt.history),
                        wall_s=20, what="walking")
        walking = [e for e in rt.history if e.tool_name == "navigate" and not e.finished]
        user.say("stop")
        await keep_running(rt, lambda: rt.task.paused and any(r["type"] == "stop" for r in rt.tracer.rows),
                           what="stopped")
        await keep_running(rt, lambda: all(e.finished for e in walking), wall_s=10, what="walk ended")
        await keep_running(rt, lambda: not rt.speech.busy(), wall_s=10, what="ack played")
    finally:
        await stop(rt)
    stop_rows = [r for r in rt.tracer.rows if r["type"] == "stop"]
    assert stop_rows[0]["physical_first"] and stop_rows[0]["receipt"]["accepted"]
    acks = [e for e in rt.history if e.tool_name == "speak" and e.tag == "runtime:safety-ack"]
    assert len(acks) == 1
    nav = walking[0]
    assert nav.status in ("failed", "cancelled") and nav.data.get("reason") in ("halted", "cancelled"), nav.data
    assert nav.data.get("at") or nav.data.get("between") or rt.belief.robot_at.value is None
