"""The M2b body surface through the runtime, live (`-m box`, WL_BOX=1; docs/M2.md R.2 / docs/M2b_wave1.md W2.3).
It MOVES the robot: the stack must be up and standing (scripts/m2_up.sh), with nothing else driving the body (stop
P5 first: two runtimes on one body share its epoch space).

    test_twenty_runtime_halts           20 x (navigate to a far keypoint through G1Robot, halt 0.3-1.0 s after it
                                        walks at >= 0.3 m/s): G1Robot.halt returns within 30 ms with stopped=True (the B.1 lane:
                                        body.halted received), the walk ends failed(halted), the robot is at rest
                                        within 1.5 s (5 consecutive ground-truth samples < 0.05 m/s, M1 E3), no fall;
                                        then resume with the next control_epoch. Writes $WL_BOX_OUT/halts.json.
    test_scan_approach_and_fences       the arrival scan is the waist scan (B.5), navigate(reach_stance) runs the
                                        approach op (B.6), a halted epoch's command is a stale_result, body events
                                        (body.mode, body.halted) reach robot.events() without polling.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from pathlib import Path

import pytest

from api.execution import ExecutionManager
from tests.contract.conftest import clock_for, make_backend

pytestmark = pytest.mark.box
N_HALTS = int(os.environ.get("WL_BOX_HALTS", "20"))
OUT = Path(os.environ.get("WL_BOX_OUT", "outputs/m2b_wave2/robot/box"))


def _pick_far_pair(robot) -> tuple[str, str]:
    """Two keypoints 3.5-7 m apart by walking distance (the walks are long enough to halt 2.5 s in)."""
    m = robot.world.static_map()
    names = [k for k in m.keypoints if k != "start"]
    best = None
    for a in names:
        for b in names:
            if a >= b:
                continue
            ka, kb = m.keypoints[a], m.keypoints[b]
            d = m.grid.distance((ka.x, ka.y), (kb.x, kb.y))
            if d is not None and 3.5 <= d <= 7.0 and (best is None or abs(d - 5.0) < abs(best[2] - 5.0)):
                best = (a, b, d)
    assert best is not None, "no keypoint pair 3.5-7 m apart"
    return best[0], best[1]


async def _speed_trace(world, seconds: float, dt: float = 0.02) -> list[tuple[float, float, float, bool]]:
    out = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        p = world.robot_pose()
        out.append((round(time.monotonic() - t0, 3), round(float(p.speed), 4), round(float(p.pelvis_z), 3),
                    bool(p.fallen)))
        await asyncio.sleep(dt)
    return out


def _at_rest_s(trace, v=0.05, n=5) -> float | None:
    run = 0
    for t, s, _, _ in trace:
        run = run + 1 if s < v else 0
        if run >= n:
            return t
    return None


async def test_twenty_runtime_halts():
    clock = clock_for("sonic")
    robot = make_backend("sonic", clock)
    world = robot.world
    q = robot.events()
    em = ExecutionManager(clock)
    a, b = _pick_far_pair(robot)
    rng = random.Random(7)
    epoch = 0
    trials = []
    targets = [a, b]
    # start at one end
    ex = em.create("navigate", {"location": a}, generation=1, control_epoch=epoch)
    r = await clock.wait_for(robot.start(ex).result(), robot.timeout_s("navigate", {"location": a}) + 10)
    assert r.status == "succeeded", r.summary
    for i in range(N_HALTS):
        to = targets[(i + 1) % 2]
        ex = em.create("navigate", {"location": to}, generation=1, control_epoch=epoch)
        h = robot.start(ex)
        # halt mid-WALK: after the in-place turn a navigate starts with, once the robot walks (>= 0.3 m/s for 0.1 s)
        t_start = time.monotonic()
        fast = 0
        while fast < 5 and time.monotonic() - t_start < 12.0 and not h.done:
            fast = fast + 1 if world.robot_pose().speed >= 0.3 else 0
            await asyncio.sleep(0.02)
        await asyncio.sleep(rng.uniform(0.3, 1.0))
        t_halt = time.monotonic() - t_start
        v_before = float(world.robot_pose().speed)
        t0 = time.perf_counter()
        rec = robot.halt(control_epoch=epoch)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        trace = await _speed_trace(world, 2.0)
        res = await clock.wait_for(h.result(), 10.0)
        rest = _at_rest_s(trace)
        fell = any(f or z < 0.55 for _, _, z, f in trace)
        trials.append({"i": i, "to": to, "t_halt_s": round(t_halt, 2), "speed_at_halt": round(v_before, 3),
                       "halt_call_ms": round(dt_ms, 2), "stopped": rec.get("stopped"), "rtt_ms": rec.get("rtt_ms"),
                       "body_handle_ms": rec.get("handle_ms"), "body_kind": rec.get("body_kind"),
                       "via": rec.get("via"), "epoch": rec.get("epoch"), "body_epoch": rec.get("body_epoch"),
                       "result": res.status, "reason": res.data.get("reason"), "at_rest_s": rest, "fell": fell,
                       "pelvis_z_min": min(z for _, _, z, _ in trace)})
        epoch += 1
        robot.resume(epoch)
        await asyncio.sleep(0.5)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    OUT.mkdir(parents=True, exist_ok=True)
    walked = [t for t in trials if t["speed_at_halt"] > 0.2]
    summary = {"n": len(trials), "walking_at_halt": len(walked),
               "stopped": sum(bool(t["stopped"]) for t in trials),
               "under_30ms": sum(t["halt_call_ms"] < 30.0 and bool(t["stopped"]) for t in trials),
               "halt_call_ms": {"p50": sorted(t["halt_call_ms"] for t in trials)[len(trials) // 2],
                                "max": max(t["halt_call_ms"] for t in trials)},
               "rtt_ms_max": max((t["rtt_ms"] or 0) for t in trials),
               "failed_halted": sum(t["result"] == "failed" and t["reason"] == "halted" for t in trials),
               "at_rest_1_5s": sum(t["at_rest_s"] is not None and t["at_rest_s"] <= 1.5 for t in trials),
               "falls": sum(t["fell"] for t in trials),
               "body_mode_events": sum(e.get("type") == "body_mode" for e in events),
               "sim_rtf": getattr(world.sim_health(), "rtf", None)}
    (OUT / "halts.json").write_text(json.dumps({"summary": summary, "trials": trials, "pair": [a, b]}, indent=1))
    print(json.dumps(summary))
    assert summary["falls"] == 0
    assert summary["under_30ms"] == len(trials), summary
    assert summary["failed_halted"] == len(trials), summary
    assert summary["body_mode_events"] > 0, "body.mode never reached robot.events()"


async def test_scan_approach_and_fences():
    clock = clock_for("sonic")
    robot = make_backend("sonic", clock)
    world = robot.world
    q = robot.events()
    em = ExecutionManager(clock)
    body = robot.body
    assert body.surface == "m2b" and body.session, "the body has no M2b surface"
    assert robot.stack_profile.scan_executor == "waist"
    m = robot.lookup_keypoints()
    target = next(iter(m["surfaces"]))
    kp = m["surfaces"][target]["keypoints"][0]
    ex = em.create("navigate", {"location": kp}, generation=1, control_epoch=0)
    r = await clock.wait_for(robot.start(ex).result(), robot.timeout_s("navigate", {"location": kp}) + 10)
    assert r.status == "succeeded", r.summary
    scan = await clock.wait_for(robot.start(em.create("observe", {"mode": "scan"}, generation=1, control_epoch=0,
                                                      action="scan")).result(), 30.0)
    d = scan.data
    assert d["mode"] == "scan" and d["scan_executor"] == "waist" and len(d["views"]) == 3, d
    assert d["scan_body"]["state"] == "succeeded" and len(d["scan_holds"]) == 3
    # a reposition of ~0.2 m sideways runs the approach op, with no INTERIM label
    p = world.robot_pose()
    sx, sy = p.x - 0.2 * math.sin(p.yaw), p.y + 0.2 * math.cos(p.yaw)
    if world.static_map().grid.is_free(sx, sy):
        rep = await clock.wait_for(robot.start(em.create(
            "navigate", {"location": "reach_stance", "anchor": kp, "stance": {"x": sx, "y": sy, "yaw": p.yaw}},
            generation=1, control_epoch=0)).result(), 60.0)
        assert rep.data["reposition_op"] == "approach" and "interim" not in rep.data, rep.data
        assert rep.status == "succeeded", rep.summary
    # halt at epoch 0, resume at 1: an epoch-0 command is a stale fence (stale_result), an epoch-1 one runs
    robot.halt(control_epoch=0)
    robot.resume(1)
    old = em.create("navigate", {"location": kp}, generation=1, control_epoch=0)
    r0 = await clock.wait_for(robot.start(old).result(), 20.0)
    assert r0.status == "failed" and r0.data["reason"] == "stale_result", r0.summary
    new = em.create("observe", {"mode": "scan"}, generation=1, control_epoch=1, action="scan")
    r1 = await clock.wait_for(robot.start(new).result(), 30.0)
    assert r1.data["mode"] == "scan" and r1.data["scan_body"]["state"] == "succeeded", r1.data
    await asyncio.sleep(0.3)
    types = set()
    while not q.empty():
        e = q.get_nowait()
        types.add(e.get("type") if e.get("type") != "body_event" else f"body_event:{e.get('topic')}")
    assert "body_mode" in types and "stale_result" in types, types
