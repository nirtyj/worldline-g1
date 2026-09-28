"""Pure-Python tests for sim_isaac (no Isaac). The DDS codec test needs unitree_sdk2py + cyclonedds (box env):
    /work/envs/isaaclab/bin/python -m pytest -q sim_isaac/tests
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from sim_isaac import joint_map as jm
from sim_isaac.band import ElasticBand
from sim_isaac.mathutil import quat_from_yaw, quat_mul, quat_rotate, quat_rotate_inverse, up_z, yaw_from_quat
from sim_isaac.rt_pacer import RtPacer


def test_joint_map_matches_deploy_constants():
    # policy_parameters.hpp:143-208: hip pitch = STIFFNESS_7520_22 ~ 99.1, ankle = 2*STIFFNESS_5020 ~ 28.5
    assert jm.KPS[0] == pytest.approx(0.025101925 * (20 * 3.1415926535) ** 2)
    assert jm.KPS[4] == pytest.approx(2 * 0.003609725 * (20 * 3.1415926535) ** 2)
    assert jm.KDS[3] == pytest.approx(2 * 2.0 * 0.025101925 * 20 * 3.1415926535)
    assert len(set(jm.G1_MOTOR_JOINTS)) == 29
    assert jm.G1_MOTOR_JOINTS[12] == "waist_yaw_joint" and jm.G1_MOTOR_JOINTS[22] == "right_shoulder_pitch_joint"
    assert jm.DEFAULT_ANGLES[3] == 0.669 and jm.DEFAULT_ANGLES[23] == -0.2
    assert set(jm.TRAIN_EFFORT_LIMIT) == set(jm.G1_MOTOR_JOINTS)


def test_quaternions():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 2.9):
        assert yaw_from_quat(quat_from_yaw(yaw)) == pytest.approx(yaw)
    q = quat_mul(quat_from_yaw(0.7), np.array([math.cos(0.2), math.sin(0.2), 0, 0]))
    v = np.array([0.3, -1.0, 2.0])
    assert quat_rotate_inverse(q, quat_rotate(q, v)) == pytest.approx(v)
    assert up_z(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)


def test_band_pulls_to_anchor_and_keeps_yaw():
    b = ElasticBand()
    q = quat_from_yaw(1.0)
    b.engage(np.array([1.0, 2.0, 0.7]), q, 0.8)
    f, t = b.advance(0.005, np.array([1.0, 2.0, 0.7]), q, np.zeros(3), np.zeros(3))
    assert f[2] == pytest.approx(1000.0) and abs(f[0]) < 1e-9       # 10000 N/m * 0.1 m up
    assert np.linalg.norm(t) < 1e-6                                  # upright at engage yaw: no torque
    tilt = quat_mul(q, np.array([math.cos(0.05), math.sin(0.05), 0, 0]))
    _, t2 = b.advance(0.005, np.array([1.0, 2.0, 0.8]), tilt, np.zeros(3), np.zeros(3))
    assert np.linalg.norm(t2) == pytest.approx(1000.0 * 0.1, rel=1e-3)  # restores the 0.1 rad roll
    b.release(ramp_s=0.01)
    assert b.advance(0.005, np.zeros(3), q, np.zeros(3), np.zeros(3)) is not None
    assert b.advance(0.005, np.zeros(3), q, np.zeros(3), np.zeros(3)) is None and not b.enabled


def test_pacer_rtf_window():
    p = RtPacer(0.005, enabled=False)
    p.mark_start_sim()
    for _ in range(400):
        p.step_done()
    assert p.t_sim == pytest.approx(2.0)
    assert p.rtf_total() > 1.0  # free-running and tiny: far faster than real time


@pytest.mark.skipif(importlib.util.find_spec("unitree_sdk2py") is None, reason="needs unitree_sdk2py (box env)")
def test_fast_codec_roundtrip():
    from sim_isaac.dds_bridge import G1DdsBridge
    names = jm.G1_MOTOR_JOINTS[::-1] + jm.DEX3_LEFT_JOINTS + jm.DEX3_RIGHT_JOINTS
    b = G1DdsBridge(names, 27, "lo", crc=True)
    assert all(b.codec_ok.values()), b.codec_ok
    b.close()
