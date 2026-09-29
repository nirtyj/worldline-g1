"""RobotBridge contract (PLAN 10, "test_robot_contract", parametrized over backends): the map and
read shapes the runtime consumes, every tool's envelope, halt within 50 ms with stopped, cancel
then result, observation ids, frozen enums. Rewrites ludo-runtime's tests/test_robot_presence.py."""

from __future__ import annotations

import asyncio
import time

import pytest

from api.execution import ExecutionManager, Rejected
from api.results import ToolResult, validate_envelope
from api.services import RobotBridge
from sim.clock import SimClock
from tests.contract.conftest import BACKENDS, make_backend

SPEED = 40.0


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def new(backend_name):
    clock = SimClock(SPEED)
    return clock, make_backend(backend_name, clock)


def test_it_is_a_robot_bridge(backend):
    _, robot = new(backend)
    assert isinstance(robot, RobotBridge)
    assert robot.profile.name in ("lite", "bringup", "sonic", "full", "real_g1")


def test_lookup_keypoints_shape(backend):
    _, robot = new(backend)
    m = robot.lookup_keypoints()
    for key in ("scene", "keypoints", "edges", "surfaces", "people", "rooms", "nav_speed_mps", "max_reach_height_m"):
        assert key in m, key
    assert "start" in m["keypoints"] and all("xy" in v for v in m["keypoints"].values())
    for s, info in m["surfaces"].items():
        assert info["keypoints"] and all(k in m["keypoints"] for k in info["keypoints"]), s
        assert isinstance(info["height_m"], (int, float))
    user = m["people"]["user"]
    assert user["deliver_to_surface"] in m["surfaces"] and user["keypoint"] in m["keypoints"]
    for a, b, d in m["edges"]:
        assert a in m["keypoints"] and b in m["keypoints"] and d >= 0
    for r, info in m["rooms"].items():
        assert "label" in info and all(s in m["keypoints"] for s in info.get("spots") or [])


def test_read_surface_shapes(backend):
    _, robot = new(backend)
    b = robot.base_state()
    assert {"moving", "at", "between", "xy"} <= set(b)
    for arm in ("left", "right"):
        g = robot.gripper(arm)
        assert {"closed", "width", "force"} <= set(g)
    t = robot.telemetry()
    assert {"pose", "moving", "grippers", "health"} <= set(t)
    assert "mode" in t.get("body", {})
    p = robot.perception()
    assert isinstance(p.get("objects", {}), dict)
    assert robot.memory() == []
    caps = robot.capabilities()
    assert {"navigation", "manipulation"} <= set(caps) and all(hasattr(h, "ok") for h in caps.values())
    assert robot.observation_id().startswith("obs-")


def test_registry_enum_is_frozen_and_uses_the_fixed_vocabulary(backend):
    _, robot = new(backend)
    reg = robot.registry()
    a = reg.loaded_object_types()
    assert a == reg.loaded_object_types() and a == sorted(a)
    assert {"alarm_clock", "apple", "banana"} <= set(a)          # banana: vocabulary, not scene GT
    assert reg.select("pick", "alarm_clock", None) is not None


async def _run(robot, clock, tool, args, *, action=None, timeout=60.0) -> ToolResult:
    ex = ExecutionManager(clock).create(tool, args, generation=1, control_epoch=0, action=action)
    h = robot.start(ex)
    return await clock.wait_for(h.result(), timeout)


def _farthest(m) -> str:
    dist: dict[str, float] = {}
    for a, b, d in m["edges"]:
        if a == "start":
            dist[b] = d
        elif b == "start":
            dist[a] = d
    return max(dist, key=dist.get)


def _check(res: ToolResult) -> None:
    assert validate_envelope(res) == [], (res.tool, res.status, validate_envelope(res), res.data)
    assert res.observation_id, res


async def test_every_tool_returns_a_valid_envelope(backend):
    clock, robot = new(backend)
    m = robot.lookup_keypoints()
    _check(await _run(robot, clock, "speak", {"text": "Hello there."}))
    locs = await _run(robot, clock, "list_locations", {})
    _check(locs)
    assert locs.data["locations"] and all("name" in x and "distance_m" in x for x in locs.data["locations"])
    d = [x["distance_m"] for x in locs.data["locations"] if x["distance_m"] is not None]
    assert d == sorted(d)                                        # nearest first
    glance = await _run(robot, clock, "observe", {"mode": "glance"})
    _check(glance)
    assert glance.data["views"] == [] and set(glance.data["hands"]) == {"left", "right"}
    target = next(iter(m["surfaces"]))
    kp = m["surfaces"][target]["keypoints"][0]
    nav = await _run(robot, clock, "navigate", {"location": kp}, action="keypoint", timeout=robot.timeout_s(
        "navigate", {"location": kp}) + 5)
    _check(nav)
    assert nav.status == "succeeded" and nav.data["at"] == kp and nav.data["executor"]
    scan = await _run(robot, clock, "observe", {"mode": "scan"})
    _check(scan)
    assert scan.data["views"], "a scan reports its views"
    reach = await _run(robot, clock, "check_reachability",
                       {"object_type": "banana", "candidates": [], "at": kp})
    _check(reach)
    # PLAN 6.4 order says not_found for "no instance in candidates"; not_seen_here leaks nothing either
    assert reach.status == "succeeded" and reach.data["reachable"] is False
    assert reach.data["reason"] in ("not_found", "not_seen_here") and reach.data["visible"] is False


async def test_halt_is_fast_and_stops_a_walk(backend):
    clock, robot = new(backend)
    m = robot.lookup_keypoints()
    far = _farthest(m)
    ex = ExecutionManager(clock).create("navigate", {"location": far}, generation=1, control_epoch=0)
    h = robot.start(ex)
    await clock.sleep(1.0)
    t0 = time.perf_counter()
    receipt = robot.halt()
    dt = time.perf_counter() - t0
    assert dt < 0.05 and receipt["accepted"] is True and receipt["stopped"] is True
    res = await clock.wait_for(h.result(), 10.0)
    _check(res)
    assert res.status in ("failed", "cancelled") and res.data.get("reason") == "halted"
    robot.resume(1)


async def test_cancel_then_result_resolves_with_where_it_stopped(backend):
    clock, robot = new(backend)
    m = robot.lookup_keypoints()
    far = _farthest(m)
    ex = ExecutionManager(clock).create("navigate", {"location": far}, generation=1, control_epoch=0)
    h = robot.start(ex)
    await clock.sleep(0.5)
    h.cancel("correction")
    h.cancel("again")                                            # idempotent
    res = await clock.wait_for(h.result(), 10.0)
    _check(res)
    assert res.status in ("cancelled", "succeeded")               # a very short walk may finish first
    if res.status == "cancelled":
        assert res.data.get("at") or res.data.get("between")


async def test_policy_down_is_a_capability_rejection_or_failed_result(backend):
    clock, robot = new(backend)
    if backend != "fake":
        pytest.skip("fault injection API differs per backend; covered on lite by the world agent's tests")
    robot.policy_down = True
    ex = ExecutionManager(clock).create("manipulate", {"action": "pick", "object_type": "apple"}, generation=1,
                                        control_epoch=0)
    with pytest.raises(Rejected) as e:
        robot.start(ex)
    assert e.value.stage == "capability" and e.value.message.startswith("policy unavailable")
