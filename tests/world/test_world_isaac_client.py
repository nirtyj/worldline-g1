"""world/isaac_client.py against a fake P1 (REP + gt.pose PUB) serving a recorded house.

M1 P1 (no object-pose / attach ops): objects keep their scene_info poses, attach raises NotSupported.
M2b P1 (get_objects / attach / detach listed in ping.ops): live poses and attach work.
"""

import time

import pytest

pytest.importorskip("zmq")

from tests.fakes.fake_p1_world import FakeP1World, free_port_offset
from world.model import NotSupported


@pytest.fixture
def p1_m1():
    p = FakeP1World("procthor-train-38", port_offset=free_port_offset()).start()
    yield p
    p.stop()


@pytest.fixture
def p1_m2b():
    p = FakeP1World("procthor-train-38", port_offset=free_port_offset(), m2b=True).start()
    yield p
    p.stop()


def _client(off):
    from world.isaac_client import IsaacGTWorldModel
    return IsaacGTWorldModel(port_offset=off, camera="head_sim", rpc_timeout_s=3.0)


def test_builds_the_same_map_as_lite(p1_m1):
    from tests.fakes.fixtures import fresh_lite
    w = _client(p1_m1.off)
    try:
        lite = fresh_lite("procthor-train-38")
        assert list(w.map.surfaces) == list(lite.map.surfaces)
        assert w.lookup_keypoints()["edges"] == lite.lookup_keypoints()["edges"]
        assert w.source == "isaac-gt"
        assert w.map.topdown["w"] == 400 and w.map.topdown["size"] == pytest.approx(480 * 0.025 / 2)
    finally:
        w.close()


def test_pose_follows_gt_pose(p1_m1):
    w = _client(p1_m1.off)
    try:
        k = w.map.keypoints["bedroom_bed_1b"]
        p1_m1.set_pose(k.x, k.y, k.yaw)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            p = w.robot_pose()
            if abs(p.x - k.x) < 1e-6:
                break
            time.sleep(0.02)
        p = w.robot_pose()
        assert (p.x, p.y) == pytest.approx((k.x, k.y)) and p.source == "isaac-gt"
        assert w.pose_age_s() < 0.5
        assert "alarm_clock_1" in {d.id for d in w.detections()}
        assert w.perception()["source"] == "isaac-ground-truth"
    finally:
        w.close()


def test_m1_p1_has_no_attach(p1_m1):
    w = _client(p1_m1.off)
    try:
        caps = w.capabilities()
        assert caps["attach"] is False and caps["object_poses"] is False and caps["band"] is True
        with pytest.raises(NotSupported):
            w.attach("alarm_clock_1", "right")
        assert w.object("alarm_clock_1").pose_source == "scene"
        w.band(False, 1.0)
        assert p1_m1.band is False
    finally:
        w.close()


def test_m2b_ops(p1_m2b):
    w = _client(p1_m2b.off)
    try:
        caps = w.capabilities()
        assert caps["attach"] and caps["detach"] and caps["object_poses"]
        # live poses from get_objects
        sid = w.map.objects["mug_1"].scene_id
        p1_m2b.move_object(sid, [9.1, 1.17, 0.4])
        time.sleep(0.15)
        o = w.object("mug_1")
        assert o.pose_source == "sim" and o.pos[2] == pytest.approx(0.4, abs=1e-6)
        # attach / detach go through P1
        w.attach("alarm_clock_1", "right", "follow")
        assert "attach" in p1_m2b.calls and w.hands()["right"] == "alarm_clock_1"
        spot = w.free_spot("bedroom_dining_table_1a", "alarm_clock_1")
        w.detach("alarm_clock_1", spot)
        assert "detach" in p1_m2b.calls
        assert w.object("alarm_clock_1").where == "bedroom_dining_table_1a"
    finally:
        w.close()
