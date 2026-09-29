"""B.3 modes and faults over the wire (fakes): body.state.mode + speed, body.mode events, body.fault{fell} with the
band and the soft recovery (path A), body.fault{deploy_lost} and its clearing, the opt-in runtime-ping watchdog."""

import math
import time

import pytest

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


def _collect(bc):
    seen = []
    bc.add_topic_listener(lambda t, m: seen.append((t, m)))
    return seen


def _modes(seen):
    return [m["mode"] for t, m in seen if t == "body.mode"]


def test_modes_speed_and_mode_events(stack):
    bc = stack.bc
    seen = _collect(bc)
    assert bc.status()["mode"] == "OFF"
    stack.stand()
    assert _wait(lambda: bc.status()["mode"] == "HOLD", 1.0)
    assert bc.turn_to(0.0).ok
    hw = bc.walk(vx=0.5, duration_s=2.5, wait=False)
    assert _wait(lambda: (bc.last_state or {}).get("mode") == "LOCOMOTION" and bc.last_state["speed"] > 0.3, 3.0)
    assert bc.last_state["pose_source"] == "gt.pose"
    hw.wait(10)
    assert _wait(lambda: bc.status()["mode"] == "HOLD", 1.0)
    arm = bc.arm_stream()
    assert arm.send(upper_body={"right_elbow_joint": 1.0})["ok"]
    assert _wait(lambda: bc.status()["mode"] == "ARM_STREAM", 1.0)
    arm.end()
    assert _wait(lambda: bc.status()["mode"] == "HOLD", 5.0)
    ms = _modes(seen)
    for a, b in (("TRANSITION", "HOLD"), ("HOLD", "LOCOMOTION"), ("LOCOMOTION", "HOLD"), ("HOLD", "ARM_STREAM"),
                 ("ARM_STREAM", "HOLD")):
        assert any(ms[i] == a and ms[i + 1] == b for i in range(len(ms) - 1)), (a, b, ms)
    ev = [m for t, m in seen if t == "body.mode"]
    assert all(e["prev"] != e["mode"] for e in ev) and ev[-1]["mode"] == "HOLD"


def test_fall_fault_band_and_soft_recovery(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    seen = _collect(bc)
    stack.stand()
    assert bc.turn_to(0.0).ok
    assert bc.arm_stream().send(upper_body={"right_elbow_joint": 1.0})["ok"]
    hw = bc.walk(vx=0.4, duration_s=8.0, wait=False)
    time.sleep(0.8)
    p1.collapsed = True
    hw.wait(5)
    assert hw.state == "failed" and hw.reason == "fallen"
    assert _wait(lambda: any(t == "body.fault" and m["kind"] == "fell" for t, m in seen), 1.0)
    assert _wait(lambda: p1.band_on, 2.0)                  # band engaged at once (sim)
    st = bc.status()
    assert st["mode"] == "FAULT" and st["fault"] == "fallen" and st["mux"]["upper"] is None   # arm override dropped
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.state == "failed" and h.reason == "fault:fallen"
    h = bc.recover(band_hold_s=0.3, stable_s=0.5, release_ramp_s=0.2, watch_s=0.6)
    assert h.ok, h.result
    assert h.result["label"] == "sim_recovery" and [s["op"] for s in h.result["steps"]] == \
        ["band", "reset_robot", "band"]
    assert not p1.band_on and not p1.collapsed and p1.root_writes == 1
    st = bc.status()
    assert st["fault"] is None and st["mode"] == "HOLD" and st["recoveries"] == 1
    assert any(t == "body.fault" and m.get("cleared") and m["by"] == "recover" for t, m in seen)
    ms = _modes(seen)
    assert "FAULT" in ms and ms[-1] == "HOLD"
    assert bc.walk(vx=0.4, duration_s=0.6).ok
    h = bc.recover()
    assert h.state == "failed" and h.reason == "no_fault"
    assert dep.stats["stop"] == 0                          # never command{stop}


def test_deploy_lost_fault_and_back(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    seen = _collect(bc)
    stack.stand()
    dep.stop()                                             # g1_debug stops (the deploy is gone)
    assert _wait(lambda: any(t == "body.fault" and m["kind"] == "deploy_lost" for t, m in seen), 3.0)
    assert bc.status()["mode"] == "FAULT" and bc.status()["fault"] == "deploy_lost"
    assert _wait(lambda: p1.band_on, 2.0)
    dep.start()                                            # back in CONTROL (same heading init)
    assert _wait(lambda: bc.status()["fault"] is None, 4.0)
    assert any(t == "body.fault" and m["kind"] == "deploy_lost" and m.get("cleared") for t, m in seen)
    assert bc.status()["mode"] == "HOLD"


def test_runtime_ping_watchdog_is_opt_in(stack):
    bc, p1 = stack.bc, stack.p1
    seen = _collect(bc)
    stack.stand()
    assert bc.turn_to(0.0).ok
    # a session that keeps pinging (0.1 s heartbeat, 0.4 s watchdog) walks normally
    rep = bc.hello("rt-a", watchdog_s=0.4, heartbeat_s=0.1)
    assert rep["ok"] and rep["data"]["watchdog_s"] == 0.4
    assert bc.walk(vx=0.4, duration_s=1.2).ok
    bc.bye()
    # a session that stops pinging while the robot walks: HOLD (an internal halt, reason runtime_lost)
    rep = bc.hello("rt-b", watchdog_s=0.4, heartbeat_s=None)
    hw = bc.walk(vx=0.5, duration_s=10.0, wait=False)
    hw.wait(3.0)
    assert hw.state == "canceled" and hw.result["reason"] == "halt" and hw.result["halt_reason"] == "runtime_lost"
    assert any(t == "body.session" and m["event"] == "lost" and m["held"] for t, m in seen)
    assert any(t == "body.halted" and m["reason"] == "runtime_lost" for t, m in seen)
    assert _wait(lambda: math.hypot(p1.vx, p1.vy) < 0.05, 1.5)
    st = bc.status()
    assert st["latched"] and st["mode"] == "HOLD"
    assert bc.ping_session()["ok"]
    assert _wait(lambda: any(t == "body.session" and m["event"] == "back" for t, m in seen), 1.0)
    assert bc.resume(st["halt_epoch"])["ok"]
    bc.bye()
    # without a session (M1 tools) nothing watches pings
    time.sleep(0.6)
    assert bc.walk(vx=0.4, duration_s=0.8).ok
