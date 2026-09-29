"""world/isaac_client.py on the P1 M2b wire (docs/contracts/p1_m2b.md v1) over tests/fakes/fake_p1_world.py (m2b=True):

    P1.1  discovery: ops, p1_contract, cameras -> capabilities
    P1.2  live object poses: every object `pose_source: "sim"`; gt.objects moves one within 0.5 s, no REP polling
    P1.3  attach/detach through P1; a held object follows the real palm
    P1.4  the head camera: head_sim on an M2b P1, the d435 model (labelled) on an M1 P1; enable_camera
    P1.5  link poses: palm positions, grasp_state.palm_dist_m, the camera pose from the real torso (waist lean)
    P1.6  detections(method="best"): P1's instance-id pixels; geometry otherwise and as the fallback
    P1.7  reset_scene / move_object
    P1.8  the furniture-only top render
    P1.9  sim.health -> ok/degraded/unsafe; gt.event robot_fell/object_fell -> drain_events
    P1.10 head frames carry their camera pose from the render (IsaacFrames)
"""

import math
import time

import pytest

pytest.importorskip("zmq")

from tests.fakes.fake_p1_world import FakeP1World, free_port_offset
from world.model import NotSupported

HOUSE = "procthor-train-38"


def _wait(cond, timeout=2.0, dt=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(dt)
    return cond()


@pytest.fixture
def m2b():
    p = FakeP1World(HOUSE, port_offset=free_port_offset(), m2b=True, health_hz=20.0).start()
    yield p
    p.stop()


def _client(off, camera="head_sim"):
    from world.isaac_client import IsaacGTWorldModel
    return IsaacGTWorldModel(port_offset=off, camera=camera, rpc_timeout_s=3.0)


def test_discovery_and_capabilities(m2b):
    w = _client(m2b.off)
    try:
        caps = w.capabilities()
        assert caps["p1_contract"] == "m2b-1" and caps["cameras"] == {"head": True, "ego_view": False}
        for k in ("attach", "detach", "object_poses", "segmentation", "enable_camera", "reset_scene", "move_object",
                  "detections:head", "detections:ego_view"):
            assert caps[k] is True, k
        assert _wait(lambda: w.capabilities()["link_poses"])
        assert w.cam.name == "head_sim" and w.camera_note == ""
        assert w.map.topdown["mode"] == "furniture" and "furniture-only" in w.map.topdown["caption"]
        assert m2b.requests[[r["op"] for r in m2b.requests].index("render_topdown")]["mode"] == "furniture"
    finally:
        w.close()


def test_m1_p1_models_the_d435_camera_and_says_so():
    p = FakeP1World(HOUSE, port_offset=free_port_offset()).start()
    try:
        w = _client(p.off)
        try:
            assert w.cam.name == "ego_d435" and "no head camera" in w.camera_note
            assert w.capabilities()["p1_contract"] is None and not w.capabilities()["segmentation"]
            with pytest.raises(NotSupported):
                w.enable_camera("ego_view", True)
            assert w.palm_position("right") is None
            assert w.map.topdown["mode"] == "full"
        finally:
            w.close()
    finally:
        p.stop()


def test_live_object_poses_from_gt_objects(m2b):
    w = _client(m2b.off)
    try:
        assert all(o.pose_source == "sim" for o in w.objects().values())          # P1.2 exit
        n_polls = m2b.calls.count("get_objects")
        sid = w.map.objects["mug_1"].scene_id
        m2b.move_object(sid, [9.1, 1.17, 0.4])
        assert _wait(lambda: abs(w.object("mug_1").pos[2] - 0.4) < 1e-6, 0.5), "moved within 0.5 s"
        o = w.object("mug_1")
        assert o.pose_source == "sim" and o.where != "bedroom_dresser_1a"
        assert m2b.calls.count("get_objects") == n_polls, "the stream, not REP polling"
    finally:
        w.close()


def test_attach_follows_the_real_palm_and_detach_places(m2b):
    w = _client(m2b.off)
    try:
        k = w.map.keypoints["bedroom_dresser_1a"]
        m2b.set_pose(k.x, k.y, k.yaw)
        assert _wait(lambda: abs(w.robot_pose().x - k.x) < 1e-6)
        w.attach("alarm_clock_1", "right", "follow")
        assert w.hands()["right"] == "alarm_clock_1" and w.object("alarm_clock_1").where == "hand:right"
        palm = w.palm_position("right")
        assert palm is not None
        assert _wait(lambda: math.dist(w.object("alarm_clock_1").pos, m2b.grip_point("right")) < 0.01)
        g = w.grasp_state("alarm_clock_1", "right")
        assert g.held and g.palm_dist_m == pytest.approx(0.07, abs=0.01)                 # the grip offset
        m2b.set_pose(k.x + 0.3, k.y, k.yaw)                                              # it walks: the object follows
        assert _wait(lambda: math.dist(w.object("alarm_clock_1").pos, m2b.grip_point("right")) < 0.01)
        assert w.hands()["right"] == "alarm_clock_1"
        spot = w.free_spot("bedroom_dining_table_1a", "alarm_clock_1")
        w.detach("alarm_clock_1", spot)
        assert w.hands()["right"] is None
        time.sleep(0.4)                                                                 # stream catches up
        assert w.object("alarm_clock_1").where == "bedroom_dining_table_1a"
        assert w.hands()["right"] is None, "P1's held_by does not undo a fresh detach"
    finally:
        w.close()


def test_camera_pose_from_the_real_torso(m2b):
    w = _client(m2b.off)
    try:
        assert _wait(lambda: w.link_pose("torso_link") is not None)
        nominal = w.camera_pose(pose=w.robot_pose())                                   # the model, no links
        live = w.camera_pose()
        assert math.dist(live.pos, nominal.pos) < 0.01
        assert live.pitch_down == pytest.approx(math.radians(15.0), abs=1e-3)
        m2b.torso_pitch = math.radians(20.0)                                           # SONIC leans the waist
        assert _wait(lambda: w.camera_pose().pitch_down > math.radians(30.0))
        assert w.camera_pose().pitch_down == pytest.approx(math.radians(35.0), abs=1e-3)
        ego = w.camera_pose(camera="ego_view")
        assert ego.pitch_down == pytest.approx(math.radians(55.0), abs=1e-3)           # 35 deg mount + 20 deg lean
    finally:
        w.close()


def test_detections_best_uses_p1_segmentation(m2b):
    w = _client(m2b.off)
    try:
        k = w.map.keypoints["bedroom_bed_1b"]
        m2b.set_pose(k.x, k.y, k.yaw)
        assert _wait(lambda: abs(w.robot_pose().x - k.x) < 1e-6)
        clock = w.map.objects["alarm_clock_1"].scene_id
        fridge = w.map.landmarks["fridge_1"].scene_ids[0]
        m2b.seg_px = {clock: 900, fridge: 20}
        geo = w.detections()
        assert all(d.method == "gt-geometric" for d in geo)
        best = w.detections(method="best")
        assert [d.id for d in best] == ["alarm_clock_1"] and best[0].px == 900.0      # the fridge: below min_px
        assert best[0].method == "fake-segmentation" and best[0].where == w.object("alarm_clock_1").where
        calls = m2b.calls.count("detections")
        w.detections(method="best")
        assert m2b.calls.count("detections") == calls, "cached for the same view"
        # a hypothetical view (a virtual scan) is never segmented: P1 renders only the live one
        assert all(d.method == "gt-geometric" for d in w.detections(method="best", yaw_offset=0.5))
        # the ego camera is off: P1 answers camera_off and the geometry stands in (counted)
        ego = w.detections(camera="ego_view", method="best")
        assert all(d.method == "gt-geometric" for d in ego) and "camera_off" in w.seg_stats["last_error"]
        rep = w.enable_camera("ego_view", True, consumer="man-000001", ttl_s=10)
        assert rep["on"] is True and m2b.cameras["ego_view"]["on"]
        assert m2b.requests[-1]["ttl_s"] == 10
        ego = w.detections(camera="ego_view", method="best")
        assert ego and ego[0].method == "fake-segmentation"
        w.enable_camera("ego_view", False, consumer="man-000001")
        assert not m2b.cameras["ego_view"]["on"]
    finally:
        w.close()


def test_sim_health_levels_and_events(m2b):
    w = _client(m2b.off)
    try:
        assert _wait(lambda: w.sim_health().source == "p1:sim.health")
        assert w.sim_health().ok and w.sim_health().rtf == pytest.approx(1.0)
        m2b.rtf = 0.91
        assert _wait(lambda: w.sim_health().state == "degraded")
        h = w.sim_health()
        assert "DEGRADED" in h.detail and h.rtf == pytest.approx(0.91)
        m2b.rtf = 0.8
        assert _wait(lambda: w.sim_health().state == "unsafe")
        m2b.rtf = 1.0
        assert _wait(lambda: w.sim_health().ok)
        m2b.emit_event("robot_fell", pelvis_z=0.3, tilt_deg=80.0, base_pos=[1, 2, 0.3])
        m2b.emit_event("object_fell", id=w.map.objects["mug_1"].scene_id, drop_m=0.7, on_floor=True)
        assert _wait(lambda: len(w._events) >= 2)
        evs = [e["event"] for e in w.drain_events()]
        assert "robot_fell" in evs and "object_fell" in evs and w.drain_events() == []
    finally:
        w.close()


def test_reset_scene_and_move_object(m2b):
    w = _client(m2b.off)
    try:
        w.attach("mug_1", "left")
        c = w.object("fork_1").pos
        w.move_object("fork_1", (c[0] + 0.1, c[1], c[2]))
        assert w.object("fork_1").pos[0] == pytest.approx(c[0] + 0.1)
        rep = w.reset_scene("default")
        assert rep["objects_reset"] > 0 and w.hands() == {"left": None, "right": None}
        assert w.object("fork_1").pos[0] == pytest.approx(c[0], abs=1e-6)
        assert w.object("mug_1").where == "bedroom_dresser_1b"
    finally:
        w.close()


def test_head_frames_carry_the_render_pose():
    from world.frames import IsaacFrames
    p = FakeP1World(HOUSE, port_offset=free_port_offset(), m2b=True, frames=True).start()
    try:
        w = _client(p.off)
        fr = IsaacFrames(w, port_offset=p.off)
        try:
            assert _wait(lambda: fr.latest("head") is not None, 3.0)
            f = fr.latest("head")
            assert f.camera == "head" and f.jpeg[:2] == b"\xff\xd8"
            x, y, _, yaw = p.pose
            assert f.cam_pose_wl[:2] == (round(x, 3), round(y, 3)) and f.cam_pose_wl[3] == 15.0
            assert f.stationary is True and f.hfov == 90.0 and f.w == 640
            assert fr.pose_stamp() == "render"
        finally:
            fr.close()
            w.close()
    finally:
        p.stop()
