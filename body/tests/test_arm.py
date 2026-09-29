"""Arm channel (op `arm`, body/arm.py) + SonicMux upper-body overlay, over the real ZMQ contract with the fakes.

The fake deploy (tools/fake_deploy.py) re-evaluates the override on every planner message exactly like
zmq_manager.hpp:581-628 and moves its arms towards it, so these tests check the joint mapping end to end (by joint
NAME on the far side), the watchdog -> hold -> blend -> release timeline, ownership/pre-emption and the composition
with a walking motion.
"""

import math
import time

import pytest
import zmq

from body import joint_map as jm
from body.frames import PlannerFrame
from body.sonic_mux import PlannerCmd, SonicMux
from body.wire import LocomotionMode, decode_planner

from .test_integration_fakes import Stack

MJ17 = jm.UPPER_BODY_MUJOCO_JOINTS


def _wait(pred, timeout=3.0, dt=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


def _named_from_wire(u17):
    return dict(zip(jm.UPPER_BODY_JOINTS, u17))


def test_mux_overlay_on_every_planner_message(port_offset):
    ctx = zmq.Context.instance()
    endpoint = f"tcp://127.0.0.1:{5556 + port_offset}"
    f = PlannerFrame()
    f.set_fallback(0.0)
    mux = SonicMux(endpoint, f, keepalive_hz=50, log=lambda *_: None, ctx=ctx)
    mux.start()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"planner")
    sub.connect(endpoint)
    try:
        time.sleep(0.3)
        pos = [0.01 * i for i in range(17)]
        vel = [-0.1] * 17
        mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (1.0, 0.0), 0.0, 0.4))
        mux.set_upper(pos, vel, [0.1] * 7, None)
        time.sleep(0.15)
        while sub.poll(10):
            sub.recv()
        p = decode_planner(sub.recv())
        assert p["mode"] == LocomotionMode.SLOW_WALK and p["movement"][0] == pytest.approx(1.0)
        assert p["upper_body_position"] == pytest.approx(pos, abs=1e-6)
        assert p["upper_body_velocity"] == pytest.approx(vel, abs=1e-6)
        assert p["left_hand_joints"] == pytest.approx([0.1] * 7) and "right_hand_joints" not in p
        mux.clear_upper()
        time.sleep(0.1)
        while sub.poll(10):
            sub.recv()
        p = decode_planner(sub.recv())
        assert "upper_body_position" not in p and "left_hand_joints" not in p
        assert mux.stats["upper_sent"] > 0
        with pytest.raises(ValueError):
            mux.set_upper([0.0] * 14)
    finally:
        sub.close(0)
        mux.close()


@pytest.fixture
def stack(port_offset, tmp_path):
    s = Stack(port_offset, tmp_path)
    yield s
    s.close()


def _stream(arm, dur, **kw):
    t0 = time.monotonic()
    rep = None
    while time.monotonic() - t0 < dur:
        rep = arm.send(**kw)
        time.sleep(0.02)
    return rep


def test_arm_stream_mapping_watchdog_blend(stack):
    bc, dep = stack.bc, stack.dep
    # not standing -> rejected, reported as failed
    arm0 = bc.arm_stream()
    rep = arm0.send(upper_body={"right_elbow_joint": 1.4})
    assert not rep["ok"] and rep["error"] == "not_standing" and arm0.handle is None
    stack.stand()

    arm = bc.arm_stream(hold_s=0.4, blend_s=0.5, servo_ki=0.0)    # exact values on the far side: servo off
    rep = _stream(arm, 1.0, upper_body={"right_elbow_joint": 1.4, "left_shoulder_roll_joint": 0.6},
                  right_hand=0.0)
    assert rep["ok"], rep
    assert arm.handle.state == "accepted"
    # far side, by name: only the named joints moved away from SONIC's reference; waist follows the reference
    named = _named_from_wire(dep.upper)
    ref = dict(zip(jm.MUJOCO_JOINTS, dep.ref_q29))
    assert named["right_elbow_joint"] == pytest.approx(1.4, abs=1e-4)
    assert named["left_shoulder_roll_joint"] == pytest.approx(0.6, abs=1e-4)
    for n in MJ17:
        if n not in ("right_elbow_joint", "left_shoulder_roll_joint"):
            assert named[n] == pytest.approx(ref[n], abs=1e-4), n
    assert dep.has_upper and dep.has_hands
    assert dep.hands["right"] == pytest.approx([0.0] * 7)
    assert dep.hands["left"] == pytest.approx(list(jm.DEX3_DEPLOY_DEFAULT_LEFT))   # the other hand: deploy default
    assert dep.body_q[jm.MJ["right_elbow_joint"]] == pytest.approx(1.4, abs=0.05)
    st = bc.status()["arm"]
    assert st["state"] == "stream" and st["waist"] == "ref"

    # stop streaming: hold (same pose) after 0.3 s, blend after hold_s, release after blend_s
    t_stop = time.monotonic()
    assert _wait(lambda: bc.status()["arm"]["state"] == "hold", 1.0)
    assert named["right_elbow_joint"] == pytest.approx(_named_from_wire(dep.upper)["right_elbow_joint"], abs=1e-4)
    assert _wait(lambda: bc.status()["arm"]["state"] == "blend", 1.5)
    arm.handle.wait(5)
    t_done = time.monotonic() - t_stop
    assert arm.handle.state == "succeeded" and arm.handle.result["ended_by"] == "watchdog"
    assert 0.3 + 0.4 + 0.5 - 0.1 <= t_done <= 2.5
    assert _wait(lambda: not dep.has_upper and not dep.has_hands, 1.0)
    # the last override before release equals SONIC's reference (seamless hand-back)
    last = [u for (_, _, u, _) in dep.upper_log if u is not None][-1]
    assert jm.mj17_from_wire(last) == pytest.approx([ref[n] for n in MJ17], abs=2e-3)
    assert bc.status()["arm"]["state"] == "off"


def test_arm_ownership_preempt_end_and_walk(stack):
    bc, dep = stack.bc, stack.dep
    stack.stand()
    a = bc.arm_stream(stream="A", blend_s=0.3, servo_ki=0.0)
    b = bc.arm_stream(stream="B", blend_s=0.3, servo_ki=0.0)
    assert _stream(a, 0.3, upper_body=[*dep.ref_q29[12:15], *[0.3] * 14])["ok"]
    rep = b.send(upper_body={"left_elbow_joint": 0.2})
    assert not rep["ok"] and rep["error"] == "arm_busy"
    # bad args never change state
    rep = a.send(upper_body={"left_knee_joint": 0.1})
    assert not rep["ok"] and rep["error"] == "bad_args"
    rep = a.send(upper_body=[0.0] * 14)
    assert not rep["ok"] and rep["error"] == "bad_args"
    # pre-emption: B takes over, A's op ends canceled, A is locked out
    rep = b.send(upper_body={"left_elbow_joint": 0.2}, preempt=True)
    assert rep["ok"], rep
    a.handle.wait(2)
    assert a.handle.state == "canceled" and a.handle.result["reason"] == "preempted"
    assert a.send(upper_body={"left_elbow_joint": 0.5})["error"] == "arm_preempted"
    # B continues from A's pose (dict overlays A's last sent vector)
    time.sleep(0.1)
    named = _named_from_wire(dep.upper)
    assert named["left_elbow_joint"] == pytest.approx(0.2, abs=1e-4)
    assert named["right_elbow_joint"] == pytest.approx(0.3, abs=1e-4)
    # 17-list: waist is the client's when waist="cmd"
    rep = b.send(upper_body=[0.05, 0.0, 0.0, *[0.3] * 14], waist="cmd")
    assert rep["ok"]
    time.sleep(0.1)
    assert _named_from_wire(dep.upper)["waist_yaw_joint"] == pytest.approx(0.05, abs=1e-4)

    # composes with walking: the walk owns the legs (SLOW_WALK), the overlay rides along
    h = bc.walk(vx=0.4, duration_s=1.5, wait=False)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        b.send(upper_body={"right_shoulder_pitch_joint": -0.5})
        time.sleep(0.02)
    recent = [(m, u) for (t, m, u, _) in list(dep.upper_log) if t > t0 + 0.3]
    assert any(m == LocomotionMode.SLOW_WALK for m, _ in recent)
    assert all(u is not None for _, u in recent)
    assert _named_from_wire(recent[-1][1])["right_shoulder_pitch_joint"] == pytest.approx(-0.5, abs=1e-4)
    # client end -> blend -> release; the walk is unaffected
    assert b.end()["ok"]
    b.handle.wait(3)
    assert b.handle.state == "succeeded" and b.handle.result["ended_by"] == "client"
    # the pre-empted stream may start again once the arms are free
    assert _wait(lambda: bc.status()["arm"]["state"] == "off", 2.0)
    rep = a.send(upper_body={"left_elbow_joint": 0.5})
    assert rep["ok"] and rep["state"] == "accepted", rep
    assert a.end()["ok"]
    a.handle.wait(3)
    h.wait(20)
    assert h.ok, h.result
    assert _wait(lambda: not dep.has_upper, 1.0)

    # stop {arms: true} ends a stream too (canceled, after the blend)
    c = bc.arm_stream(stream="C", blend_s=0.3, servo_ki=0.0)
    assert _stream(c, 0.3, upper_body={"left_elbow_joint": 1.0})["ok"]
    assert bc.stop(arms=True).ok
    c.handle.wait(3)
    assert c.handle.state == "canceled" and c.handle.result["reason"] == "stop"
    assert _wait(lambda: not dep.has_upper, 1.0)


def test_arm_slew_limit_and_stale(stack):
    bc, dep = stack.bc, stack.dep
    stack.stand()
    a = bc.arm_stream(max_vel=2.0, servo_ki=0.0)
    ref_elbow = dep.ref_q29[jm.MJ["right_elbow_joint"]]
    rep = a.send(upper_body={"right_elbow_joint": ref_elbow + 1.0})
    assert rep["ok"]
    time.sleep(0.1)
    # 1 rad step at 2 rad/s: after ~0.1 s only ~0.2-0.3 rad of it has been sent
    moved = _named_from_wire(dep.upper)["right_elbow_joint"] - ref_elbow
    assert 0.02 < moved < 0.45
    _stream(a, 0.6, upper_body={"right_elbow_joint": ref_elbow + 1.0})
    assert _named_from_wire(dep.upper)["right_elbow_joint"] == pytest.approx(ref_elbow + 1.0, abs=1e-4)
    assert bc.status()["arm"]["stats"]["slew_limited_ticks"] > 5
    # a message older than the watchdog is dropped
    rep = bc.request("arm", {"stream": a.stream, "t_wall": time.time() - 1.0,
                             "upper_body": {"right_elbow_joint": 0.0}})
    assert not rep["ok"] and rep["error"] == "stale_command"
    a.end()
    a.handle.wait(5)
    assert a.handle.state == "succeeded"


def test_arm_servo_removes_steady_state_bias(stack):
    bc, dep = stack.bc, stack.dep
    stack.stand()
    j = jm.MJ["right_elbow_joint"]
    dep.arm_bias[j] = 0.15                      # the fake arm settles 0.15 rad past the target
    tgt = dep.ref_q29[j] - 0.3
    plain = bc.arm_stream(stream="plain", servo_ki=0.0)
    _stream(plain, 1.2, upper_body={"right_elbow_joint": tgt})
    assert dep.body_q[j] - tgt == pytest.approx(0.15, abs=0.02)
    plain.end()
    plain.handle.wait(5)
    servo = bc.arm_stream(stream="servo", servo_ki=3.0, servo_delay_s=0.1)
    _stream(servo, 3.0, upper_body={"right_elbow_joint": tgt})
    assert abs(dep.body_q[j] - tgt) < 0.02
    st = bc.status()["arm"]["servo"]
    assert st["ki"] == 3.0 and st["corr_max_abs"] == pytest.approx(0.15, abs=0.03)
    servo.end()
    servo.handle.wait(5)
    dep.arm_bias[j] = 0.0
