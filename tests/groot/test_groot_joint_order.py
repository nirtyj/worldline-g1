"""groot.joint_order: every order pinned to its upstream source (golden copies below), conversions by name, and
agreement with P3's body/joint_map.py whenever it is importable."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from groot import joint_order as jo

# $ARENA/isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml (release/0.2.1 @ 8b4a3a47), verbatim order
ARENA_GR00T_43DOF = {
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
                 "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
                  "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "left_hand": ["left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint",
                  "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
                  "left_hand_thumb_2_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    "right_hand": ["right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
                   "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
                   "right_hand_thumb_2_joint"],
}
# nvidia/Arena-G1-Static-PickNPlace-Task @ 37ba80a, lerobot/meta/modality.json slices of observation.state / action
MODALITY_SLICES = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22),
                   "left_hand": (22, 29), "right_arm": (29, 36), "right_hand": (36, 43)}
# $DEPLOY/include/policy_parameters.hpp:81 upper_body_joint_isaaclab_order_in_mujoco_index
POLICY_PARAMETERS_81 = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
# $WBC/gear_sonic/utils/mujoco_sim/base_sim.py:225-240 (Dex3 MJCF order)
DEX3_ORDER = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")


def test_groot_keys_match_arena_yaml():
    for key in jo.GROOT_STATE_KEYS:
        assert list(jo.GROOT_KEY_JOINTS[key]) == ARENA_GR00T_43DOF[key], key
    assert jo.GROOT_ACTION_KEYS == ("left_arm", "right_arm", "left_hand", "right_hand", "waist",
                                    "base_height_command", "navigate_command")
    assert {k: jo.GROOT_KEY_DIMS[k] for k in jo.GROOT_ACTION_KEYS} == {
        "left_arm": 7, "right_arm": 7, "left_hand": 7, "right_hand": 7, "waist": 3, "base_height_command": 1,
        "navigate_command": 3}


def test_lerobot43_matches_modality_json_slices():
    for group, (a, b) in MODALITY_SLICES.items():
        assert list(jo.LEROBOT_43_JOINTS[a:b]) == ARENA_GR00T_43DOF[group], group
    assert len(set(jo.LEROBOT_43_JOINTS)) == 43


def test_sonic_wire_order_is_interleaved():
    assert jo.SONIC_WIRE_FROM_MUJOCO == POLICY_PARAMETERS_81
    expect = list(jo.WAIST_JOINTS)
    for l, r in zip(jo.LEFT_ARM_JOINTS, jo.RIGHT_ARM_JOINTS):
        expect += [l, r]
    assert list(jo.SONIC_UPPER_JOINTS) == expect
    assert jo.SONIC_UPPER_JOINTS[3] == "left_shoulder_pitch_joint"
    assert jo.SONIC_UPPER_JOINTS[4] == "right_shoulder_pitch_joint"
    assert jo.MJ17_JOINTS == jo.WAIST_JOINTS + jo.LEFT_ARM_JOINTS + jo.RIGHT_ARM_JOINTS


def test_mj17_wire_roundtrip_by_name():
    v = np.arange(17, dtype=float)                   # value = mj17 index
    w = jo.mj17_to_wire(v)
    for i, n in enumerate(jo.SONIC_UPPER_JOINTS):
        assert w[i] == jo.MJ17_JOINTS.index(n)
    assert np.array_equal(jo.wire_to_mj17(w), v)
    batch = np.stack([v, v + 100])
    assert np.array_equal(jo.wire_to_mj17(jo.mj17_to_wire(batch)), batch)


def test_dex3_and_groot_hand_orders():
    for side in jo.SIDES:
        assert jo.DEX3_HAND_JOINTS[side] == tuple(f"{side}_hand_{s}_joint" for s in DEX3_ORDER)
        dex3 = np.arange(7, dtype=float)
        g = jo.dex3_to_groot_hand(side, dex3)
        # GR00T index_0 = Dex3 slot 5, thumb_0 = Dex3 slot 0, thumb_2 = Dex3 slot 2
        assert list(g) == [5, 6, 3, 4, 0, 1, 2]
        assert np.array_equal(jo.groot_to_dex3_hand(side, g), dex3)


def test_groot_state_from_body_by_name():
    q = np.arange(29, dtype=float)                   # value = MuJoCo index
    lh = 100 + np.arange(7, dtype=float)             # value = 100 + Dex3 slot
    rh = 200 + np.arange(7, dtype=float)
    s = jo.groot_state_from_body(q, lh, rh)
    assert list(s) == list(jo.GROOT_STATE_KEYS)
    assert list(s["left_arm"]) == list(range(15, 22))
    assert list(s["right_arm"]) == list(range(22, 29))
    assert list(s["waist"]) == [12, 13, 14]
    assert list(s["left_hand"]) == [105, 106, 103, 104, 100, 101, 102]
    assert list(s["right_hand"]) == [205, 206, 203, 204, 200, 201, 202]
    with pytest.raises(ValueError):
        jo.groot_state_from_body(np.zeros(28), lh, rh)


def test_lerobot43_roundtrip_through_body_order():
    v = np.arange(43, dtype=float)
    q29, lh, rh = jo.body_from_lerobot43(v)
    keys_direct = jo.groot_keys_from_lerobot43(v)
    keys_via_body = jo.groot_state_from_body(q29, lh, rh)
    for k in jo.GROOT_STATE_KEYS:
        a, b = MODALITY_SLICES[k]
        assert np.array_equal(keys_direct[k], v[a:b]), k
        assert np.array_equal(keys_via_body[k], v[a:b]), k
    rows = np.stack([v, v * 2])
    assert jo.groot_keys_from_lerobot43(rows)["left_hand"].shape == (2, 7)


def test_index_map_errors():
    with pytest.raises(KeyError):
        jo.index_map(["a", "b"], ["c"])
    with pytest.raises(ValueError):
        jo.index_map(["a", "a"], ["a"])
    with pytest.raises(ValueError):
        jo.reorder(np.zeros(3), ["a", "b"], ["a"])


def test_limits_and_stand_pose():
    names = set(jo.MUJOCO_JOINTS) | set(jo.DEX3_HAND_JOINTS["left"]) | set(jo.DEX3_HAND_JOINTS["right"])
    assert set(jo.URDF_LIMITS) == names and len(names) == 43
    assert all(lo < hi for lo, hi in jo.URDF_LIMITS.values())
    assert len(jo.STAND_Q29) == 29 and jo.STAND_WAIST == (0.0, 0.0, 0.0)
    for n, v in zip(jo.MUJOCO_JOINTS, jo.STAND_Q29):
        lo, hi = jo.URDF_LIMITS[n]
        assert lo <= v <= hi, n
    lo, hi = jo.limits_array(jo.LEFT_ARM_JOINTS, margin=0.02)
    assert lo[3] == pytest.approx(-1.0472 + 0.02) and hi[3] == pytest.approx(2.0944 - 0.02)


def test_urdf_limits_match_box_urdf():
    """Pins URDF_LIMITS to the URDF P1 simulates (on a box; GROOT_URDF overrides the path)."""
    from groot.urdf import BOX_URDF, load_limits
    path = Path(os.environ.get("GROOT_URDF", BOX_URDF))
    if not path.exists():
        pytest.skip(f"{path} not present (not on a box)")
    lim = load_limits(path)
    for n, (lo, hi) in jo.URDF_LIMITS.items():
        assert lim[n] == pytest.approx((lo, hi), abs=1e-9), n


def _load_joint_map():
    """body.joint_map (P3, uncommitted in M2b wave 1) if importable, else GROOT_JOINT_MAP_FILE, else None."""
    try:
        return importlib.import_module("body.joint_map")
    except ImportError:
        pass
    path = os.environ.get("GROOT_JOINT_MAP_FILE")
    if path and Path(path).exists():
        spec = importlib.util.spec_from_file_location("_groot_check_joint_map", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def test_orders_agree_with_body_joint_map():
    jm = _load_joint_map()
    if jm is None:
        pytest.skip("body.joint_map not importable (set GROOT_JOINT_MAP_FILE to check a file)")
    assert tuple(jm.MUJOCO_JOINTS) == jo.MUJOCO_JOINTS
    assert tuple(jm.UPPER_BODY_FROM_MUJOCO) == jo.SONIC_WIRE_FROM_MUJOCO
    assert tuple(jm.UPPER_BODY_JOINTS) == jo.SONIC_UPPER_JOINTS
    assert tuple(jm.UPPER_BODY_MUJOCO_JOINTS) == jo.MJ17_JOINTS
    assert tuple(jm.LEFT_HAND_JOINTS) == jo.DEX3_HAND_JOINTS["left"]
    assert tuple(jm.RIGHT_HAND_JOINTS) == jo.DEX3_HAND_JOINTS["right"]
    assert tuple(jm.DEFAULT_ANGLES) == jo.STAND_Q29
    for n, lim in jo.URDF_LIMITS.items():
        assert tuple(jm.JOINT_LIMITS[n]) == pytest.approx(lim, abs=1e-9), n
    v = np.random.default_rng(1).normal(size=17)
    assert np.allclose(jm.wire_from_mj17(list(v)), jo.mj17_to_wire(v))
    assert np.allclose(jm.mj17_from_wire(list(v)), jo.wire_to_mj17(v))
