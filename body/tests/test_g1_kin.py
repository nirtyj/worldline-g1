import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from body import g1_kin as K
from body import joint_map as jm

from .conftest import wbc_dir


def test_mirror_symmetry_and_zero_pose():
    q = {n: 0.0 for n in jm.UPPER_BODY_MUJOCO_JOINTS}
    p = K.points(q)
    # zero pose: upper arm down, elbow at 90 deg (forearm forward), palms ~24 cm in front of the pelvis
    assert p["left_palm"][0] == pytest.approx(0.241, abs=0.005) and p["left_palm"][1] > 0.1
    assert p["left_palm"] * np.array([1, -1, 1]) == pytest.approx(p["right_palm"], abs=2e-3)
    qd = K.named_from_q29(jm.DEFAULT_ANGLES)          # roll +-0.2 is a mirror pair
    pd = K.points(qd)
    assert pd["left_wrist"] * np.array([1, -1, 1]) == pytest.approx(pd["right_wrist"], abs=2e-3)


def test_ik_reaches_table_targets():
    seed = K.named_from_q29(jm.DEFAULT_ANGLES)
    # palm at table height (0.75 m world = -0.035 m in the pelvis frame at pelvis z 0.785), 22 cm forward, and a
    # counter reach (0.865 m). The G1's palm reaches at most ~0.26 m forward of the pelvis at table height
    # (shoulder-to-palm 0.372 m straight), so a table must be within ~0.35 m of the pelvis.
    for side, tgt in (("left", (0.22, 0.20, -0.035)), ("right", (0.22, -0.20, -0.035)), ("right", (0.34, -0.20, 0.08))):
        q, err = K.ik_palm(side, tgt, seed, seed)
        assert err < 0.005, (side, tgt, err)
        for j in K.ARM_CHAIN[side]:
            lo, hi = jm.JOINT_LIMITS[j]
            assert lo <= q[j] <= hi


@pytest.mark.skipif(wbc_dir() is None, reason="WBC clone not available")
def test_chain_matches_main_urdf():
    root = ET.parse(os.path.join(wbc_dir(), "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf")).getroot()
    J = {j.get("name"): j for j in root.findall("joint")}
    for name, (xyz, rpy, axis) in K.CHAIN_PARAMS.items():
        j = J[name]
        o = j.find("origin")
        assert [float(v) for v in o.get("xyz").split()] == pytest.approx(list(xyz), abs=1e-9), name
        assert [float(v) for v in o.get("rpy", "0 0 0").split()] == pytest.approx(list(rpy), abs=1e-9), name
        a = j.find("axis")
        if axis is None:
            assert j.get("type") == "fixed", name
        else:
            assert [float(v) for v in a.get("xyz").split()] == pytest.approx(list(axis)), name


def test_hand_points_follow_the_dex3_chain():
    """g1_kin.hand_points: the palm slab is +-4.4 cm thick along the palm z axis, the open fingers reach 17.6 cm along
    x, the right thumb sits on +y (the left one on -y), and closing the fingers curls them towards the thumb."""
    from body import joint_map as jm

    q = K.named_from_mj17(jm.DEFAULT_ANGLES[12:29])
    for side, sy in (("right", 1.0), ("left", -1.0)):
        P = K.arm_frames(side, q)["palm"]
        loc = (K.hand_points(side, q) - P[:3, 3]) @ P[:3, :3]          # back into the palm frame
        assert loc[:, 2].min() == pytest.approx(-0.044, abs=1e-6) and loc[:, 2].max() == pytest.approx(0.044, abs=1e-6)
        assert loc[:, 0].max() == pytest.approx(0.0777 + 0.0458 + 0.052, abs=1e-6)
        assert (loc[:, 1].max() if sy > 0 else -loc[:, 1].min()) > 0.1                  # the thumb
        closed = (K.hand_points(side, q, jm.DEX3_CLOSED[side]) - P[:3, 3]) @ P[:3, :3]
        tips_open, tips_closed = loc[48:, :], closed[48:, :]                              # the index_1 box
        assert sy * (tips_closed[:, 1].mean() - tips_open[:, 1].mean()) > 0.03
