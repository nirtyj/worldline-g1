"""B.1 halt lane (PUSH/PULL 5612 -> body.halted on 5611), over the real ZMQ contract with the fakes.

Ground truth comes from the fake P1 (speed) and the fake deploy (what SONIC was sent: planner mode, command{stop}
count); the body's own claims are only checked against them."""

import math
import statistics
import threading
import time

import pytest
import zmq

from body.frames import PlannerFrame
from body.sonic_mux import PlannerCmd, SonicMux
from body.wire import LocomotionMode, dumps_json

from .test_integration_fakes import Stack


def _wait(pred, timeout=3.0, dt=0.01):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture
def stack(port_offset, tmp_path):
    s = Stack(port_offset, tmp_path)
    yield s
    s.close()


def _topics(bc):
    seen = []
    bc.add_topic_listener(lambda t, m: seen.append((time.monotonic(), t, m)))
    return seen


def test_mux_latch_overrides_motions_and_resume_holds():
    f = PlannerFrame()
    f.set_fallback(0.0)
    mux = SonicMux("tcp://127.0.0.1:1", f, log=lambda *_: None)          # never bound: _effective() only
    mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (1.0, 0.0), 0.3, 0.5))
    assert mux._effective().mode == LocomotionMode.SLOW_WALK
    mux.latch(0.7, owner="halt:1")
    mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (0.0, 1.0), 0.2, 0.5))   # a motion still ticking: no effect
    e = mux._effective()
    assert mux.latched and e.mode == LocomotionMode.IDLE and e.facing_w == pytest.approx(0.7) and not e.moving()
    mux.unlatch()
    e = mux._effective()
    assert not mux.latched and e.mode == LocomotionMode.IDLE and e.facing_w == pytest.approx(0.7)   # no stale walk


def test_halt_mid_walk_latch_resume_and_fences(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    seen = _topics(bc)
    stack.stand()
    assert bc.turn_to(0.0).ok
    hw = bc.walk(vx=0.5, duration_s=10.0, wait=False)
    assert _wait(lambda: math.hypot(p1.vx, p1.vy) > 0.3, 3.0)
    r = bc.halt(3, timeout_s=0.5)
    assert r["acked"] and r["body"]["epoch"] == 3 and r["body"]["kind"] == "new"
    assert r["body"]["handle_ms"] < 10.0 and r["rtt_ms"] < 30.0, r
    assert r["body"]["latched"] and r["body"]["active"]["id"] == hw.id
    hw.wait(2.0)
    assert hw.state == "canceled" and hw.result["reason"] == "halt" and hw.result["halt_epoch"] == 3
    # at rest within 1.5 s (fake P1 ground truth), never command{stop}, planner IDLE on the wire
    t0 = time.monotonic()
    assert _wait(lambda: math.hypot(p1.vx, p1.vy) < 0.05, 1.5)
    assert time.monotonic() - t0 < 1.5
    assert dep.stats["stop"] == 0 and dep.mode == LocomotionMode.IDLE and not p1.collapsed
    st = bc.status()
    assert st["latched"] and st["halt_epoch"] == 3 and st["mode"] == "HOLD" and st["mux"]["latched"]
    # the latch: no epoch (an M1 tool) or an epoch <= 3 is rejected `halted`
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.state == "failed" and h.reason == "halted"
    h = bc.walk(vx=0.4, duration_s=1.0, control_epoch=3, execution_id="e3", generation=1)
    assert h.state == "failed" and h.reason == "halted"
    assert math.hypot(p1.vx, p1.vy) < 0.05
    # a re-send of the same halt is re-acked (kind repeat); stop is never gated
    r2 = bc.halt(3, timeout_s=0.5)
    assert r2["acked"] and r2["body"]["kind"] == "repeat"
    assert bc.stop().ok
    # resume: older than the halt -> stale_command (published); then resume 4 clears the latch
    rep = bc.resume(2)
    assert not rep["ok"] and rep["error"] == "stale_command"
    rep = bc.resume(4)
    assert rep["ok"] and rep["data"]["was_latched"]
    assert not bc.status()["latched"]
    # after the resume, epoch-3 commands are stale (their executions were ended by the halt); M1 tools work again
    h = bc.walk(vx=0.4, duration_s=1.0, control_epoch=3, execution_id="e3", generation=1)
    assert h.state == "failed" and h.reason == "stale_command"
    assert _wait(lambda: any(t == "body.stale_command" and m.get("op") == "walk" for _, t, m in seen), 1.0)
    h = bc.walk(vx=0.4, duration_s=0.6)
    assert h.ok, h.result
    # a late re-send of halt 3 after the resume is ignored (kind stale), the robot is not latched again
    r3 = bc.halt(3, timeout_s=0.5)
    assert r3["acked"] and r3["body"]["kind"] == "stale" and not r3["body"]["latched"]
    assert not bc.status()["latched"]
    # implicit resume: a command with a newer epoch than the latch clears it
    assert bc.halt(10, timeout_s=0.5)["body"]["kind"] == "new"
    h = bc.walk(vx=0.4, duration_s=0.6, control_epoch=11, execution_id="e11", generation=2)
    assert h.ok, h.result
    assert not bc.status()["latched"]
    assert any(t == "body.resumed" and m.get("source") == "implicit:walk" for _, t, m in seen)
    assert dep.stats["stop"] == 0 and dep.stats["planner_timeouts"] == 0


def test_halt_does_not_wait_for_a_busy_control_loop(stack):
    """The control thread is stuck in one 400 ms tick (as in a slow A* plan or a P1 call): the halt is still handled
    and acked on its own thread, and the leg motion ends `canceled` once the control thread is back."""
    bc, svc, p1 = stack.bc, stack.svc, stack.p1
    stack.stand()
    assert bc.turn_to(0.0).ok
    hw = bc.walk(vx=0.5, duration_s=10.0, wait=False)
    assert _wait(lambda: math.hypot(p1.vx, p1.vy) > 0.3, 3.0)
    orig = svc._tick
    stuck = threading.Event()

    def slow_tick(now):
        if not stuck.is_set():
            stuck.set()
            time.sleep(0.4)
        orig(now)

    svc._tick = slow_tick
    assert stuck.wait(1.0)
    rtts, handles = [], []
    for ep in range(20, 25):
        r = bc.halt(ep, timeout_s=0.2)
        assert r["acked"], r
        rtts.append(r["rtt_ms"])
        handles.append(r["body"]["handle_ms"])
    assert max(handles) < 10.0 and statistics.median(rtts) < 30.0, (rtts, handles)
    hw.wait(2.0)
    assert hw.state == "canceled" and hw.result["reason"] == "halt"
    assert _wait(lambda: math.hypot(p1.vx, p1.vy) < 0.05, 1.5)
    st = bc.status()
    assert st["halt"]["n"] == 5 and st["halt"]["lane"]["halts"] == 5 and st["halt_epoch"] == 24


def test_halt_via_router_op_and_bad_lane_messages(stack):
    bc, svc = stack.bc, stack.svc
    stack.stand()
    rep = bc.request("halt", {"epoch": 5})
    assert rep["ok"] and rep["data"]["epoch"] == 5 and rep["data"]["source"] == "op"
    assert not bc.request("halt", {})["ok"]
    # malformed lane messages are counted, never guessed
    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 1000)                      # deliver before close (the connect is asynchronous)
    push.connect(f"tcp://127.0.0.1:{bc.ports['body_halt']}")
    push.send(b"not json")
    push.send(dumps_json({"op": "halt"}))
    push.send(dumps_json({"op": "resume", "epoch": 6}))     # the lane also takes resume
    push.close()                                           # close(0) would drop them (linger 0)
    assert _wait(lambda: svc.halt_lane.stats["bad"] == 2 and svc.halt_lane.stats["resumes"] == 1, 2.0)
    assert _wait(lambda: not bc.status()["latched"], 1.0)
