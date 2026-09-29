"""Camera specs and frame metadata of the P1 M2b wire (docs/contracts/p1_m2b.md §5), no Isaac.

    .venv-rt/bin/python -m pytest -q sim_isaac/tests
"""
from __future__ import annotations

import base64
import math

import msgpack
import numpy as np
import pytest

from sim_isaac import wire
from sim_isaac.mathutil import quat_from_yaw


def test_ego_view_is_arena_head_camera():
    c = wire.EGO_VIEW
    # IsaacLab-Arena g1.py:103-106 offset on head_link; head_link = torso_link + (0.0039635, 0, -0.044)
    assert c.mount_xyz == pytest.approx((0.0039635 + 0.04485, 0.0, -0.044 + 0.35325))
    assert c.parent == "torso_link"
    assert (c.width, c.height) == (640, 480)
    # PinholeCameraCfg(focal_length=15), horizontal_aperture 20.955, vertical = 20.955 * 480 / 640
    assert c.focal_length_mm == 15.0 and c.horizontal_aperture_mm == 20.955
    assert c.vertical_aperture_mm == pytest.approx(15.71625)
    assert c.hfov_deg == pytest.approx(69.8686, abs=1e-3)
    assert c.vfov_deg == pytest.approx(55.2978, abs=1e-3)
    assert c.fx_px == pytest.approx(c.fy_px) and c.fx_px == pytest.approx(458.1246, abs=1e-3)
    assert c.clipping == (0.1, 5.0)
    # the ROS-convention quaternion (-0.62721, 0.62721, -0.32651, 0.32651) looks 35 deg down, straight ahead
    yaw, pitch = wire.forward_yaw_pitch(c.mount_quat_wxyz)
    assert math.degrees(pitch) == pytest.approx(35.0, abs=0.01)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    left = wire.quat_rotate(np.asarray(c.mount_quat_wxyz), np.array([0.0, 1.0, 0.0]))
    assert left == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)          # no roll: image x stays horizontal
    assert c.port == "ego" and c.default_on is False and c.key == "ego_view"


def test_head_matches_world_head_sim_model():
    """P1's head camera is world/perception.py HEAD_SIM: same mount, pitch and FOV, so gt-geometric visibility and
    the rendered image describe the same view."""
    from world.perception import HEAD_SIM, TORSO_FROM_PELVIS, camera_pose

    h = wire.HEAD
    assert h.hfov_deg == pytest.approx(HEAD_SIM.hfov_deg, abs=1e-9)
    assert h.vfov_deg == pytest.approx(HEAD_SIM.vfov_deg, abs=1e-9)
    assert h.mount_xyz == pytest.approx(HEAD_SIM.mount_xyz)
    assert math.radians(h.pitch_down_deg) == pytest.approx(HEAD_SIM.mount_pitch, abs=1e-9)
    for yaw in (-2.0, 0.0, 0.7):
        pelvis = np.array([1.5, -0.4, 0.787])
        q = quat_from_yaw(yaw)
        torso = pelvis + wire.quat_rotate(q, np.array(TORSO_FROM_PELVIS))
        p, cq = h.world_pose(torso, q)
        ref = camera_pose(HEAD_SIM, pelvis, yaw)
        assert p == pytest.approx(ref.pos, abs=1e-9)
        cy, cp = wire.forward_yaw_pitch(cq)
        assert cy == pytest.approx(ref.yaw, abs=1e-9) and cp == pytest.approx(ref.pitch_down, abs=1e-9)
    assert pelvis[2] + TORSO_FROM_PELVIS[2] + h.mount_xyz[2] == pytest.approx(1.357, abs=1e-3)   # ~1.35 m


def test_d435_is_m1_camera():
    from sim_isaac import joint_map as jm

    d = wire.D435
    assert d.mount_xyz == jm.D435_XYZ
    assert d.vfov_deg == pytest.approx(45.0) and d.hfov_deg == pytest.approx(57.82, abs=0.01)
    assert d.mount_quat_wxyz == pytest.approx(jm.quat_wxyz_from_rpy(*jm.D435_RPY))


def test_with_resolution_keeps_hfov_and_square_pixels():
    s = wire.with_resolution(wire.HEAD, 320, 240)
    assert s.hfov_deg == pytest.approx(90.0) and s.fx_px == pytest.approx(s.fy_px)
    assert wire.with_resolution(wire.HEAD, 640, 480) is wire.HEAD


def test_cam_pose_wl_matches_isaac_frames_formula():
    from world import coords
    from world.perception import HEAD_SIM, camera_pose

    cp = camera_pose(HEAD_SIM, (2.0, 3.0, 0.78), 0.4)
    q = wire.quat_from_matrix(cp.R)
    got = wire.cam_pose_wl(cp.pos, q)
    mx, mz = coords.to_map_xz(cp.pos[0], cp.pos[1])
    want = [round(mx, 3), round(mz, 3), round(coords.yaw_map_deg(cp.yaw), 1), round(math.degrees(cp.pitch_down), 1)]
    assert got == pytest.approx(want)


def _cv2_style_jpeg_b64(rgb: np.ndarray) -> str:
    """What P1 sends: cv2.imencode of an RGB array (so cv2.imdecode returns RGB). Without cv2, PIL on the channel-
    reversed array writes the same bytes' meaning."""
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., ::-1])).save(b, format="JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def test_frame_message_is_gear_sonic_compatible():
    """A 5565 head frame decodes with the M1 consumers (viz.common, body.wire when cv2 exists) and carries the §5.2
    metadata."""
    from viz.common import decode_head

    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:, :320] = (255, 0, 0)
    b64 = _cv2_style_jpeg_b64(rgb)
    meta = wire.frame_meta(wire.HEAD, seq=3, render_seq=7, t_sim=1.25, t_capture=1000.5, t_capture_mono=55.0,
                           cam_pos=[1, 2, 1.36], cam_quat=wire.HEAD.mount_quat_wxyz, base_lin_vel_w=[0.01, 0, 0],
                           base_ang_vel_w=[0, 0, 0.01])
    raw = msgpack.packb(wire.gear_sonic_message("head", b64, meta, 1000.52), use_bin_type=True)
    jpeg, m = decode_head(raw)                                   # viz: key "ego_view" absent -> first image
    assert jpeg is not None and m["t_wall"] is None and m["seq"] == 3
    d = msgpack.unpackb(raw, raw=False)
    assert d["timestamps"] == {"head": 1000.5} and list(d["images"]) == ["head"] and d["head"] == b64
    assert d["camera"] == "head" and d["render_seq"] == 7 and d["t_pub"] == 1000.52
    assert d["stationary"] is True and d["cam_pose_wl"][3] == pytest.approx(15.0, abs=0.05)
    assert d["hfov"] == pytest.approx(90.0) and d["vfov"] == pytest.approx(73.74, abs=0.01)
    assert wire.is_stationary([0.2, 0, 0], [0, 0, 0]) is False
    try:
        import cv2  # noqa: F401
    except ImportError:
        return
    from body.wire import decode_camera_message
    ts, imgs = decode_camera_message(raw)
    assert ts == {"head": 1000.5} and list(imgs) == ["head"]
    assert imgs["head"][240, 100, 0] > 200 and imgs["head"][240, 100, 2] < 50       # cv2 decode = RGB


def test_consumers_and_grid():
    c = wire.Consumers()
    c.add("groot:e1", now=10.0, ttl_s=5.0)
    c.add("rec", now=10.0)
    assert bool(c) and c.names() == ["groot:e1", "rec"]
    assert c.expire(14.9) == [] and c.expire(15.0) == ["groot:e1"]
    c.remove("rec")
    assert not c
    assert wire.next_due(0.0, 30.0) == pytest.approx(1 / 30)
    assert wire.next_due(1 / 30, 30.0) == pytest.approx(2 / 30)
    # 15 Hz ticks are a subset of the 30 Hz grid: commensurate cameras share renders
    g30 = {round(wire.next_due(k * 0.005, 30.0), 6) for k in range(400)}
    g15 = {round(wire.next_due(k * 0.005, 15.0), 6) for k in range(400)}
    assert g15 <= g30
    assert wire.next_due(3.0, 0.0) == math.inf
