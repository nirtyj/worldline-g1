"""Object poses, attach placement, fall detection, health and instance counting of the P1 M2b wire
(docs/contracts/p1_m2b.md §3, §4, §7, §10), no Isaac."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim_isaac import wire
from sim_isaac.mathutil import quat_from_yaw
from sim_isaac.segment import PrimIndex, count_instances

AABB0 = [[1.0, 2.0, 0.9], [1.1, 2.2, 1.1]]
P0 = np.array([1.05, 2.1, 0.9])            # body origin at the bottom centre
Q0 = np.array([1.0, 0, 0, 0])


def test_moved_box_translation_and_yaw():
    b = wire.moved_box(AABB0, P0, Q0, P0 + [0.5, 0, -0.2], Q0)
    assert b[0] == pytest.approx([1.5, 2.0, 0.7]) and b[1] == pytest.approx([1.6, 2.2, 0.9])
    b = wire.moved_box(AABB0, P0, Q0, P0, quat_from_yaw(math.pi / 2))     # 0.1 x 0.2 footprint turns 90 deg
    ext = np.array(b[1]) - np.array(b[0])
    assert ext == pytest.approx([0.2, 0.1, 0.2], abs=1e-9)
    assert wire.box_center(b)[2] == pytest.approx(1.0)


def test_placed_pose_puts_centre_and_keeps_upright():
    p, q = wire.placed_pose(AABB0, P0, Q0, [3.0, 1.0, 0.8], None)
    assert wire.box_center(wire.moved_box(AABB0, P0, Q0, p, q)) == pytest.approx([3.0, 1.0, 0.8])
    assert q == pytest.approx(Q0)
    q_tilt = np.array([math.cos(0.3), math.sin(0.3), 0, 0])                # a tipped load pose stays as loaded
    p2, q2 = wire.placed_pose(AABB0, P0, q_tilt, [0, 0, 1], 0.5)
    assert wire.box_center(wire.moved_box(AABB0, P0, q_tilt, p2, q2)) == pytest.approx([0, 0, 1])
    assert wire.forward_yaw_pitch(q2)[0] == pytest.approx(wire.forward_yaw_pitch(q_tilt)[0] + 0.5, abs=1e-9)


def test_parse_pose_arg_forms():
    c, y = wire.parse_pose_arg([1, 2, 3])
    assert list(c) == [1, 2, 3] and y is None
    assert wire.parse_pose_arg([1, 2, 3, 0.2])[1] == 0.2
    assert list(wire.parse_pose_arg({"pos": [1, 2, 3], "yaw": 1.0})[0]) == [1, 2, 3]
    assert wire.parse_pose_arg({"x": 1, "y": 2, "z": 3})[1] is None
    with pytest.raises(ValueError):
        wire.parse_pose_arg([1, 2])


def test_compose_relative_roundtrip():
    pa, qa = np.array([0.3, -0.2, 1.0]), quat_from_yaw(0.8)
    pb, qb = np.array([1.0, 1.0, 0.5]), np.array([math.cos(0.2), 0, math.sin(0.2), 0])
    rp, rq = wire.relative(pa, qa, pb, qb)
    p, q = wire.compose(pa, qa, rp, rq)
    assert p == pytest.approx(pb) and q == pytest.approx(qb)


def test_object_record_fields():
    rec = wire.object_record("Mug|1", "mug_1", [1, 2, 3], [1, 0, 0, 0], AABB0, held_by=None, dynamic=True,
                             lin_vel=[0.1, 0, 0])
    assert set(rec) == {"id", "name", "pos", "quat_wxyz", "aabb", "held_by", "dynamic", "source", "lin_vel", "moving"}
    assert rec["source"] == "sim" and rec["moving"] is True
    held = wire.object_record("Mug|1", "mug_1", [1, 2, 3], [1, 0, 0, 0], AABB0, held_by="left", dynamic=True,
                              lin_vel=[0.1, 0, 0])
    assert held["lin_vel"] == [0.0, 0.0, 0.0] and held["moving"] is False
    assert wire.object_record("T|1", "t", [0, 0, 0], [1, 0, 0, 0], AABB0, held_by=None, dynamic=False)["source"] \
        == "static"


def test_fall_detector():
    f = wire.FallDetector(floor_z=0.0)
    assert f.update("a", 0.0, 0.9, 0.0, False) is None           # first rest: records 0.9
    assert f.update("a", 0.35, 0.9, 0.0, False) is None
    assert f.update("a", 1.0, 0.5, 1.2, False) is None           # falling
    assert f.update("a", 1.5, 0.02, 0.01, False) is None         # still, but not for rest_s yet
    ev = f.update("a", 1.9, 0.02, 0.01, False, pos=[1, 1, 0.07])
    assert ev["id"] == "a" and ev["drop_m"] == pytest.approx(0.88) and ev["on_floor"] is True
    assert f.update("a", 2.5, 0.02, 0.0, False) is None          # fires once
    f.update("b", 0.0, 0.9, 0.0, False)
    f.update("b", 0.4, 0.9, 0.0, False)
    assert f.update("b", 0.5, 0.2, 0.0, True) is None            # held: never a fall
    assert f.update("b", 1.0, 0.85, 0.0, False) is None and f.update("b", 1.4, 0.85, 0.0, False) is None


def test_health_levels():
    assert wire.health_level(1.0, 1.0)[0] == "ok"
    assert wire.health_level(0.9, 0.94)[0] == "degraded"
    assert wire.health_level(0.84, 0.99)[0] == "unsafe"
    assert wire.health_level(None, None)[0] == "ok"


def test_op_error_reply():
    e = wire.OpError("unknown_object", "'X'")
    assert e.reply() == {"ok": False, "code": "unknown_object", "error": "unknown_object: 'X'"}


def test_instance_counts_map_prims_to_objects():
    objs = [{"id": "AlarmClock|1", "name": "alarm_clock_1", "prim_path": "/World/House/Geometry/AlarmClock_1"},
            {"id": "Shelf|2", "name": "shelf_2", "prim_path": "/World/House/Geometry/Shelf_2",
             "extra_prims": ["/World/House/Geometry/Shelf_2b"]}]
    idx = PrimIndex(objs)
    seg = np.zeros((48, 64), np.uint32)
    seg[10:20, 5:15] = 3            # alarm clock mesh (deep child prim)
    seg[10:12, 20:22] = 4           # tiny: 4 px
    seg[30:40, 30:50] = 5           # shelf, second prim
    seg[0:5, :] = 6                 # robot hand
    seg[40:48, :] = 7               # wall
    id2 = {3: "/World/House/Geometry/AlarmClock_1/mesh_0/geo", 4: "/World/House/Geometry/Shelf_2/x",
           5: "/World/House/Geometry/Shelf_2b/visual", 6: "/World/G1/left_hand_index_1_link/visuals",
           7: "/World/House/Geometry/wall_3_visual", 0: "BACKGROUND"}
    dets, other = count_instances(seg, id2, idx, min_px=10)
    by = {d["id"]: d for d in dets}
    assert set(by) == {"AlarmClock|1", "Shelf|2"}
    assert by["AlarmClock|1"]["px"] == 100 and by["AlarmClock|1"]["bbox"] == [5, 10, 14, 19]
    assert by["Shelf|2"]["px"] == 200 + 4 and by["Shelf|2"]["bbox"] == [20, 10, 49, 39]
    assert other["robot"] == 5 * 64 and other["structure"] == 8 * 64
    assert sum(d["px"] for d in dets) + sum(other.values()) == seg.size
    only, _ = count_instances(seg, {str(k): v for k, v in id2.items()}, idx, min_px=10, ids={"Shelf|2"})
    assert [d["id"] for d in only] == ["Shelf|2"]


def test_batched_boxes_equal_moved_box():
    rng = np.random.default_rng(3)
    n = 5
    aabbs = []
    for _ in range(n):
        lo = rng.uniform(-2, 2, 3)
        aabbs.append([lo.tolist(), (lo + rng.uniform(0.05, 0.4, 3)).tolist()])
    P0 = rng.uniform(-2, 2, (n, 3))
    Q0 = rng.normal(size=(n, 4))
    Q0 /= np.linalg.norm(Q0, axis=1, keepdims=True)
    P = rng.uniform(-2, 2, (n, 3))
    Q = rng.normal(size=(n, 4))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    lc = wire.body_local_points(np.stack([wire.box_corners(a) for a in aabbs]), P0, Q0)
    C = wire.body_world_points(lc, P, Q)
    for k in range(n):
        ref = wire.moved_box(aabbs[k], P0[k], Q0[k], P[k], Q[k])
        assert C[k].min(axis=0) == pytest.approx(ref[0]) and C[k].max(axis=0) == pytest.approx(ref[1])
    assert wire.matrices_from_quats(Q)[2] == pytest.approx(wire.matrix_from_quat(Q[2]))
