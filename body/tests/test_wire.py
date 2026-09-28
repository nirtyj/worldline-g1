import importlib.util
import json
import math
import struct

import numpy as np
import pytest

from body import wire
from body.tests.conftest import wbc_dir


def _upstream():
    d = wbc_dir()
    if d is None:
        pytest.skip("WBC clone not found (set WBC_DIR)")
    spec = importlib.util.spec_from_file_location(
        "zmq_planner_sender", f"{d}/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_header_size_matches_upstream():
    up = _upstream()
    assert up.HEADER_SIZE == wire.HEADER_SIZE == 1280


@pytest.mark.parametrize("start,stop,planner", [(1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 0, 1)])
def test_command_bytes_equal_upstream(start, stop, planner):
    up = _upstream()
    assert wire.build_command_message(start, stop, planner) == up.build_command_message(start, stop, planner)


@pytest.mark.parametrize("args", [
    (0, [0, 0, 0], [1, 0, 0], -1.0, -1.0),
    (1, [0.6, -0.8, 0.0], [0.70710678, 0.70710678, 0.0], 0.45, -1.0),
    (2, [1, 0, 0], [-1, 0, 0], -1.0, 0.7),
])
def test_planner_bytes_equal_upstream(args):
    up = _upstream()
    assert wire.build_planner_message(*args) == up.build_planner_message(*args)


def test_planner_with_upper_body_equal_upstream():
    up = _upstream()
    ub = list(np.linspace(-0.3, 0.3, 17))
    h = [0.1] * 7
    a = wire.build_planner_message(1, [1, 0, 0], [1, 0, 0], 0.3, -1, ub, None, h, h)
    b = up.build_planner_message(1, [1, 0, 0], [1, 0, 0], 0.3, -1, ub, None, h, h)
    assert a == b


def test_decode_roundtrip_like_cpp():
    msg = wire.build_planner_message(1, [0.6, -0.8, 0.0], [0.0, 1.0, 0.0], 0.45, -1.0)
    assert msg.startswith(b"planner")
    hdr = json.loads(msg[7:7 + 1280].rstrip(b"\x00"))
    assert hdr["v"] == 1 and hdr["endian"] == "le" and [f["name"] for f in hdr["fields"]] == \
        ["mode", "movement", "facing", "speed", "height"]
    # payload offsets as ZMQPackedMessageSubscriber slices them
    base = 7 + 1280
    assert struct.unpack_from("<i", msg, base)[0] == 1
    assert struct.unpack_from("<fff", msg, base + 4) == pytest.approx((0.6, -0.8, 0.0), abs=1e-6)
    d = wire.decode_planner(msg)
    assert d["mode"] == 1 and d["speed"] == pytest.approx(0.45) and d["facing"] == pytest.approx([0, 1, 0])
    c = wire.decode_command(wire.build_command_message(True, False, True))
    assert c == {"start": True, "stop": False, "planner": True}


def test_decode_rejects_missing_fields():
    hdr = wire._build_header([{"name": "mode", "dtype": "i32", "shape": [1]}])
    with pytest.raises(ValueError):
        wire.decode_planner(b"planner" + hdr + struct.pack("<i", 1))


def test_split_topic_variants():
    assert wire.split_topic([b"gt.pose", b"abc"], b"gt.pose") == b"abc"
    assert wire.split_topic([b"gt.pose\x81\xa1a\x01"], b"gt.pose") == b"\x81\xa1a\x01"
    assert wire.split_topic([b"gt.pose abc"], b"gt.pose") == b"abc"
    assert wire.split_topic([b"other", b"x"], b"gt.pose") is None


def test_yaw_quat_roundtrip():
    for y in (-3.0, -1.0, 0.0, 0.5, 3.1):
        assert wire.yaw_from_quat_wxyz(wire.quat_wxyz_from_yaw(y)) == pytest.approx(y, abs=1e-9)


def test_camera_roundtrip():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :32] = (200, 10, 10)
    ts, imgs = wire.decode_camera_message(wire.encode_camera_message({"ego_view": img}, 12.5))
    assert ts == {"ego_view": 12.5}
    out = imgs["ego_view"]
    assert out.shape == img.shape
    assert abs(int(out[5, 5, 0]) - 200) < 20 and int(out[5, 5, 2]) < 40  # channel order preserved (cv2 round-trip)


def test_pose_parse_defaults():
    p = wire.Pose({"base_pos": [1, 2, 0.8], "base_quat_wxyz": wire.quat_wxyz_from_yaw(0.7)}, 0.0)
    assert p.yaw == pytest.approx(0.7) and p.pelvis_z == pytest.approx(0.8) and not p.fallen
