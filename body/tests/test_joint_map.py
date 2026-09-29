"""body/joint_map.py pinned against upstream (PLAN §6.6 invariant 8).

When the WBC clone is present (conftest.wbc_dir) the index lists are parsed from the C++/Python sources; the golden
copies below are checked either way, so a silent edit of joint_map.py fails even without the clone.
"""

import os
import re
import xml.etree.ElementTree as ET

import pytest

from body import joint_map as jm

from .conftest import wbc_dir

# Golden copies (WBC @ b042411)
GOLD_UPPER_ISAACLAB = [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]       # policy_parameters.hpp:80
GOLD_UPPER_MUJOCO = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]      # policy_parameters.hpp:81
GOLD_ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                           11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]           # policy_parameters.hpp:93
GOLD_DEX3 = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
DEPLOY = "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"


def _cpp_array(src: str, name: str) -> list[int]:
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", src)
    assert m, name
    return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def test_golden_lists():
    assert list(jm.UPPER_BODY_ISAACLAB_IDX) == GOLD_UPPER_ISAACLAB
    assert list(jm.UPPER_BODY_FROM_MUJOCO) == GOLD_UPPER_MUJOCO
    assert list(jm.ISAACLAB_IDX_OF_MUJOCO) == GOLD_ISAACLAB_TO_MUJOCO
    assert list(jm.DEX3_SUFFIX) == GOLD_DEX3


def test_internal_consistency():
    # the two orderings are inverse permutations
    for i in range(29):
        assert jm.MUJOCO_IDX_OF_ISAACLAB[jm.ISAACLAB_IDX_OF_MUJOCO[i]] == i
    # the wire's MuJoCo list is its IsaacLab list mapped through mujoco_to_isaaclab
    assert [jm.MUJOCO_IDX_OF_ISAACLAB[j] for j in jm.UPPER_BODY_ISAACLAB_IDX] == list(jm.UPPER_BODY_FROM_MUJOCO)
    assert sorted(jm.UPPER_BODY_FROM_MUJOCO) == list(range(12, 29))
    assert jm.UPPER_BODY_JOINTS[:5] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                                        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint")
    assert jm.UPPER_BODY_JOINTS[-2:] == ("left_wrist_yaw_joint", "right_wrist_yaw_joint")
    assert set(jm.JOINT_LIMITS) == set(jm.MUJOCO_JOINTS) | set(jm.LEFT_HAND_JOINTS) | set(jm.RIGHT_HAND_JOINTS)


def test_upper_body_roundtrip_and_no_contiguous_slice():
    q = [float(i) for i in range(29)]              # value = MuJoCo index
    u = jm.upper_from_mujoco(q)
    assert u == [float(i) for i in GOLD_UPPER_MUJOCO]
    assert u != q[12:29]                           # invariant 8: a contiguous slice scrambles the arms
    assert jm.mujoco_from_upper(u, q) == q
    mj17 = q[12:29]
    assert jm.wire_from_mj17(mj17) == u
    assert jm.mj17_from_wire(u) == mj17
    assert jm.mj17_from_mujoco(q) == mj17
    # a left-elbow-only change lands on the left elbow's wire slot and nowhere else
    base = list(jm.DEFAULT_ANGLES[12:29])
    moved = jm.mj17_from_named({"left_elbow_joint": 1.3}, base)
    w0, w1 = jm.wire_from_mj17(base), jm.wire_from_mj17(moved)
    diff = [i for i in range(17) if w0[i] != w1[i]]
    assert diff == [jm.UPPER_BODY_JOINTS.index("left_elbow_joint")] == [9]
    with pytest.raises(KeyError):
        jm.mj17_from_named({"left_knee_joint": 0.1}, base)


def test_clamp_and_hands():
    v = [0.0] * 17
    v[jm.UPPER_BODY_MUJOCO_JOINTS.index("left_elbow_joint")] = 5.0
    c, n = jm.clamp_mj17(v)
    assert n == 1 and c[jm.UPPER_BODY_MUJOCO_JOINTS.index("left_elbow_joint")] == pytest.approx(2.0944)
    assert jm.hand_closure("left", 0.0) == [0.0] * 7
    assert jm.hand_closure("right", 1.0) == list(jm.DEX3_DEPLOY_DEFAULT_RIGHT)
    assert jm.hand_closure_of("left", jm.hand_closure("left", 0.4)) == pytest.approx(0.4)
    for side in ("left", "right"):
        # the deploy's fist uses 1.75 where the URDF limit is 1.7453: clamping moves it by < 5 mrad
        q, n = jm.clamp_hand(side, jm.DEX3_CLOSED[side])
        assert max(abs(a - b) for a, b in zip(q, jm.DEX3_CLOSED[side])) < 0.005
        for name, v in zip(jm.HAND_JOINTS[side], jm.DEX3_CLOSED[side]):
            lo, hi = jm.JOINT_LIMITS[name]
            assert lo - 0.005 <= v <= hi + 0.005


def test_matches_sim_isaac_joint_map():
    from sim_isaac import joint_map as sj
    assert list(jm.MUJOCO_JOINTS) == list(sj.G1_MOTOR_JOINTS)
    assert list(jm.DEFAULT_ANGLES) == list(sj.DEFAULT_ANGLES)
    assert list(jm.LEFT_HAND_JOINTS) == list(sj.DEX3_LEFT_JOINTS)
    assert list(jm.RIGHT_HAND_JOINTS) == list(sj.DEX3_RIGHT_JOINTS)


@pytest.mark.skipif(wbc_dir() is None, reason="WBC clone not available")
def test_against_upstream_sources():
    W = wbc_dir()
    src = open(os.path.join(W, DEPLOY, "include/policy_parameters.hpp")).read()
    assert _cpp_array(src, "upper_body_joint_isaaclab_order_in_isaaclab_index") == list(jm.UPPER_BODY_ISAACLAB_IDX)
    assert _cpp_array(src, "upper_body_joint_isaaclab_order_in_mujoco_index") == list(jm.UPPER_BODY_FROM_MUJOCO)
    assert _cpp_array(src, "isaaclab_to_mujoco") == list(jm.ISAACLAB_IDX_OF_MUJOCO)
    assert _cpp_array(src, "mujoco_to_isaaclab") == list(jm.MUJOCO_IDX_OF_ISAACLAB)
    m = re.search(r"default_angles\s*=\s*\{([^}]*)\}", src)
    vals = [float(re.sub(r"//.*", "", x).strip()) for x in re.sub(r"//[^\n]*", "", m.group(1)).split(",") if x.strip()]
    assert vals == list(jm.DEFAULT_ANGLES)
    # the deploy splices the planner's 17 values with exactly this list (g1_deploy_onnx_ref.cpp:784-791)
    cpp = open(os.path.join(W, DEPLOY, "src/g1_deploy_onnx_ref.cpp")).read()
    assert "current_motion_joint_pos[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = " \
           "upper_body_joint_positions_buffer_[i]" in cpp
    # second upstream source: the Pico teleop client's frozen-upper-body indices
    py = open(os.path.join(W, "gear_sonic/scripts/pico_manager_thread_server.py")).read()
    m = re.search(r"def _get_upper_body_joint_indices.*?return \[([^\]]*)\]", py, re.S)
    assert [int(x) for x in m.group(1).split(",")] == list(jm.UPPER_BODY_FROM_MUJOCO)
    # Dex3 order: MJCF joint order of the <side>_hand joints (base_sim.py collects them in model order)
    xml = ET.parse(os.path.join(W, "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")).getroot()
    names = [j.get("name") for j in xml.iter("joint") if j.get("name")]
    assert [n for n in names if "left_hand" in n] == list(jm.LEFT_HAND_JOINTS)
    assert [n for n in names if "right_hand" in n] == list(jm.RIGHT_HAND_JOINTS)
    body = [n for n in names if any(p in n for p in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"))]
    assert body == list(jm.MUJOCO_JOINTS)
    # the deploy's no-hand-data default (input_interface.hpp:333-362)
    ii = open(os.path.join(W, DEPLOY, "include/input_interface/input_interface.hpp")).read()
    assert "{0, 0, 1.75, -1.57, -1.75, -1.57, -1.75 }" in ii and "{0, 0, -1.75,  1.57,  1.75,  1.57,  1.75 }" in ii
    # URDF limits (main.urdf for the body, the hand URDF for Dex3)
    lim = {}
    for f in ("gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
              "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"):
        for j in ET.parse(os.path.join(W, f)).getroot().findall("joint"):
            if j.get("type") == "revolute" and j.get("name") not in lim:
                l = j.find("limit")
                lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
    for name, (lo, hi) in jm.JOINT_LIMITS.items():
        assert lim[name] == pytest.approx((lo, hi), abs=1e-9), name
