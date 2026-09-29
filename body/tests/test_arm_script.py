"""B.7 arm_script (+ CarryLock) and B.5 scan on the arm channel, tick by tick (body/tests/arm_sim.py)."""

import math

import numpy as np
import pytest

from body import g1_kin as K
from body import joint_map as jm
from body.arm import ARM_IDX, ArmError
from body.arm_script import pelvis_to_world, world_to_pelvis
from body.wire import Pose, quat_wxyz_from_yaw

from .arm_sim import DT, Rig

OK = (True, None, {})


def _pose(x=1.0, y=2.0, yaw=math.radians(30.0), z=0.78):
    return Pose({"base_pos": [x, y, z], "base_quat_wxyz": quat_wxyz_from_yaw(yaw), "yaw": yaw}, 0.0)


def _run_op(rig, op_id, max_s=12.0):
    t = 0.0
    while rig.terminal(op_id) is None and t < max_s:
        rig.run(DT)
        t += DT
    return rig.terminal(op_id)


def test_frames_round_trip():
    p = _pose()
    b = np.array([0.3, -0.2, 0.1])
    assert world_to_pelvis(pelvis_to_world(b, p), p) == pytest.approx(b, abs=1e-12)
    w = pelvis_to_world([0.3, 0.0, 0.0], p)
    assert w[:2] == pytest.approx([1.0 + 0.3 * math.cos(p.yaw), 2.0 + 0.3 * math.sin(p.yaw)], abs=1e-12)


def test_pick_sequence_carrylock_release_retract():
    rig = Rig(pose=_pose())
    ch = rig.ch
    obj_b = np.array([0.30, -0.20, 0.10])                      # a counter-top point in front-right of the pelvis
    obj_w = pelvis_to_world(obj_b, rig.pose)
    rep = ch.handle_arm_script("pg", {"phase": "pregrasp", "arm": "right", "target_w": obj_w.tolist()}, OK)
    assert rep["ok"] and rep["state"] == "accepted" and rep["data"]["ik_err_m"] < 0.01, rep
    assert ch.state()["mode"] == "script" and rig.records["pg"] == "arm_script"
    ev = _run_op(rig, "pg")
    assert ev["state"] == "succeeded" and ev["data"]["ended_by"] == "script" and ev["data"]["hold"] == "target"
    d = ev["data"]
    assert d["palm_err_w_m"]["p90"] < 0.01 and d["palm_err_w_m"]["n"] >= 40
    # pregrasp: standoff 0.10 m back along the approach and 0.05 m up; the hand opened
    assert np.linalg.norm(np.array(d["goal_w"])[:2] - obj_w[:2]) == pytest.approx(0.10, abs=1e-3)
    assert d["goal_w"][2] == pytest.approx(obj_w[2] + 0.05, abs=1e-9)
    assert rig.mux.upper[3] == pytest.approx([0.0] * 7)
    assert ch.state()["mode"] == "hold" and ch.hold.kind == "target"
    # grasp: palm to the grasp point, then the hand closes (0.8 of the fist) -> CarryLock
    ev = None
    rep = ch.handle_arm_script("g", {"phase": "grasp", "arm": "right", "target_w": obj_w.tolist()}, OK)
    assert rep["ok"], rep
    ev = _run_op(rig, "g")
    d = ev["data"]
    assert ev["state"] == "succeeded" and d["palm_err_w_m"]["p90"] < 0.01 and d["carry"] is True
    assert d["hand_closure_cmd"] == pytest.approx(0.8, abs=1e-3)
    assert rig.mux.upper[3] == pytest.approx(jm.hand_closure("right", 0.8))
    snap = ch.snapshot()
    assert snap["carry"]["engaged"] and snap["carry"]["palm_err_m"]["right"] < 0.01
    # the carry pose rides on every later tick (the legs may walk meanwhile: the override composes)
    held = rig.sent()
    rig.run(2.0)
    assert rig.sent() == pytest.approx(held, abs=1e-3)
    # lift 8 cm (world up) with the hand held closed
    pw0 = np.array(d["palm_final_w"])
    ev = (ch.handle_arm_script("l", {"phase": "lift", "arm": "right"}, OK), _run_op(rig, "l"))[1]
    assert ev["state"] == "succeeded" and ev["data"]["goal_w"][2] == pytest.approx(pw0[2] + 0.08, abs=0.01)
    assert ev["data"]["carry"] is True and rig.mux.upper[3] == pytest.approx(jm.hand_closure("right", 0.8))
    # release opens the hand without moving the arm; retract ends with a blend back to SONIC (override off)
    before = rig.sent()
    ev = (ch.handle_arm_script("r", {"phase": "release", "arm": "right"}, OK), _run_op(rig, "r"))[1]
    assert ev["state"] == "succeeded" and rig.mux.upper[3] == pytest.approx([0.0] * 7)
    assert [rig.sent()[k] for k in ARM_IDX] == pytest.approx([before[k] for k in ARM_IDX], abs=2e-3)
    ev = (ch.handle_arm_script("rt", {"phase": "retract", "arm": "right"}, OK), _run_op(rig, "rt"))[1]
    assert ev["state"] == "succeeded" and ev["data"]["hold"] == "stand"
    rig.run(2.0)
    assert rig.mux.upper is None and ch.state()["mode"] == "off"


def test_arm_script_rejections_change_nothing():
    rig = Rig(pose=_pose())
    ch = rig.ch
    far_w = pelvis_to_world([1.0, -0.2, 0.1], rig.pose).tolist()
    with pytest.raises(ArmError) as e:
        ch.handle_arm_script("x", {"phase": "grasp", "arm": "right", "target_w": far_w}, OK)
    assert e.value.reason == "ik_unreachable" and ch.state()["mode"] == "off" and rig.mux.upper is None
    for bad in ({"phase": "wave", "arm": "right"}, {"phase": "grasp", "arm": "right"},
                {"phase": "grasp", "arm": "middle", "target_w": far_w}, {"phase": "grasp", "arm": "left",
                                                                         "target_w": [1, 2]}):
        with pytest.raises(ArmError):
            ch.handle_arm_script("x", bad, OK)
    rep = ch.handle_arm_script("x", {"phase": "release", "arm": "left"}, (False, "not_standing", {}))
    assert rep["error"] == "not_standing" and rig.terminal("x")["state"] == "failed"
    # busy while a script runs; halted while latched
    ok_w = pelvis_to_world([0.3, 0.2, 0.1], rig.pose).tolist()
    assert ch.handle_arm_script("a", {"phase": "pregrasp", "arm": "left", "target_w": ok_w}, OK)["ok"]
    assert ch.handle_arm_script("b", {"phase": "release", "arm": "left"}, OK)["error"] == "arm_busy"
    assert rig.arm({"stream": "c", "upper_body": ch.reference_mj17()})["error"] == "arm_busy"
    ch.latch(1)
    ch.flush()
    assert rig.terminal("a")["data"]["ended_by"] == "halt"
    assert ch.handle_arm_script("b", {"phase": "release", "arm": "left"}, OK)["error"] == "halted"


def test_ik_locks_the_wrist_pitch_and_yaw():
    rig = Rig(pose=_pose())
    q0 = rig.ch.reference_mj17()
    seed = K.named_from_mj17(q0)
    q, err = K.ik_palm("right", (0.30, -0.20, 0.10), seed, lock=("right_wrist_pitch_joint", "right_wrist_yaw_joint"))
    assert err < 0.005
    assert q["right_wrist_pitch_joint"] == seed["right_wrist_pitch_joint"]
    assert q["right_wrist_yaw_joint"] == seed["right_wrist_yaw_joint"]


def test_scan_moves_only_the_waist_yaw_and_reports_each_hold():
    rig = Rig(pose=_pose(), plant_kw={"yaw_gain": 0.78})          # G0: SONIC follows waist yaw with ~0.78
    ch = rig.ch
    rep = ch.handle_scan("sc", {}, OK)
    assert rep["ok"] and rep["data"]["yaw_deg"] == [-35.0, 0.0, 35.0] and rep["data"]["hold_on_end"] == "stand"
    arms0 = [rig.sent()[k] for k in ARM_IDX] if rig.mux.upper else None
    arm_sent = []
    t = 0.0
    while rig.terminal("sc") is None and t < 10.0:
        rig.run(DT)
        t += DT
        if rig.mux.upper is not None and ch.state()["mode"] == "script":
            arm_sent.append([rig.sent()[k] for k in ARM_IDX])
    ev = rig.terminal("sc")
    assert ev["state"] == "succeeded"
    holds = [e["data"] for e in rig.events if e["id"] == "sc" and e["data"].get("kind") == "scan.hold"]
    assert [h["i"] for h in holds] == [0, 1, 2]
    assert [h["yaw_cmd_deg"] for h in holds] == pytest.approx([-35.0, 0.0, 35.0])
    for h in holds:                                               # the waist-yaw servo closes most of the 22 % gap
        assert abs(h["yaw_err_deg"]) < 4.0, h
    assert ev["data"]["holds"] == holds and ev["data"]["arm_dev_rad_max"] < 0.01
    spread = np.ptp(np.array(arm_sent), axis=0).max()
    assert spread < 1e-6                                          # the arms are held (no bias in this plant)
    rig.run(2.0)
    assert rig.mux.upper is None                                  # nothing was held before: blend back and off
    with pytest.raises(ArmError):
        ch.handle_scan("x", {"pitch_deg": [20]}, OK)
    with pytest.raises(ArmError):
        ch.handle_scan("x", {"yaw_deg": [90]}, OK)


def test_scan_takes_over_carrylock_and_restores_it():
    rig = Rig(pose=_pose())
    ch = rig.ch
    obj_w = pelvis_to_world([0.30, -0.20, 0.10], rig.pose).tolist()
    ch.handle_arm_script("g", {"phase": "grasp", "arm": "right", "target_w": obj_w}, OK)
    _run_op(rig, "g")
    carry = list(ch.hold.pose)
    assert ch.hold.is_carry()
    assert ch.handle_scan("sc", {"yaw_deg": [-20, 20]}, OK)["data"]["hold_on_end"] == "target"
    assert _run_op(rig, "sc")["state"] == "succeeded"
    assert ch.hold.kind == "target" and ch.hold.is_carry()
    assert [ch.hold.pose[k] for k in ARM_IDX] == pytest.approx([carry[k] for k in ARM_IDX], abs=1e-9)
    assert rig.mux.upper[3] == pytest.approx(jm.hand_closure("right", 0.8))


def test_takeovers_keep_the_servo_correction_without_counting_it_twice():
    """SONIC settles 0.15 rad off on the right elbow; the servo removes it. A scan that takes over the CarryLock
    hold, and a chunk session that takes the hold over after it, must not add the correction a second time (a jump
    of 0.15 rad) and the arm must stay on target throughout."""
    rig = Rig(pose=_pose())
    ch = rig.ch
    k = jm.UPPER_BODY_MUJOCO_JOINTS.index("right_elbow_joint")
    rig.plant.bias[k] = 0.15
    obj_w = pelvis_to_world([0.30, -0.20, 0.10], rig.pose).tolist()
    ch.handle_arm_script("g", {"phase": "grasp", "arm": "right", "target_w": obj_w, "settle_s": 4.0}, OK)
    _run_op(rig, "g")
    tgt = ch.hold.pose[k]
    assert abs(rig.q()[k] - tgt) < 0.01 and abs(ch.corr[k] + 0.15) < 0.01
    worst = 0.0
    ch.handle_scan("sc", {"yaw_deg": [-20, 20]}, OK)
    t = 0.0
    while rig.terminal("sc") is None and t < 10:
        rig.run(DT)
        t += DT
        worst = max(worst, abs(rig.q()[k] - tgt))
    assert worst < 0.02, worst
    from .arm_sim import base
    assert rig.arm({**base("gr"), "lead_s": 0.0})["state"] == "accepted"
    for _ in range(100):
        rig.run(DT)
        rig.arm({**base("gr"), "keepalive": True})
        worst = max(worst, abs(rig.q()[k] - tgt))
    assert worst < 0.02, worst
