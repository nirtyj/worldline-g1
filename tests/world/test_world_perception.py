"""world/perception.py: GT visibility (frustum + AABB-ray occlusion), THOR's visible/40px semantics."""

import math

import pytest

from api.observation import in_frustum
from tests.fakes.fixtures import fresh_lite
from world import coords
from world.perception import EGO_D435, HEAD_SIM, camera_pose


def test_camera_models_match_the_mounts():
    # d435 on torso_link (contract m1.md §1.4): ~1.24 m standing, ~47.6 deg down, HFOV 57.8
    cp = camera_pose(EGO_D435, (0.0, 0.0, 0.78), 0.0)
    assert cp.pos[2] == pytest.approx(1.24, abs=0.02)
    assert math.degrees(cp.pitch_down) == pytest.approx(47.6, abs=0.1)
    assert EGO_D435.hfov_deg == pytest.approx(57.8, abs=0.2)
    # PLAN's sim-added head camera: ~1.35 m, 15 deg down, 90 deg HFOV
    hp = camera_pose(HEAD_SIM, (0.0, 0.0, 0.78), 0.0)
    assert hp.pos[2] == pytest.approx(1.35, abs=0.02)
    assert math.degrees(hp.pitch_down) == pytest.approx(15.0, abs=0.01)
    assert HEAD_SIM.hfov_deg == pytest.approx(90.0, abs=0.1)
    # the camera looks where the pelvis faces
    cp2 = camera_pose(HEAD_SIM, (1.0, 2.0, 0.78), math.pi / 2)
    assert cp2.yaw == pytest.approx(math.pi / 2)
    assert cp2.pos[1] > 2.0


def _at(w, kp, yaw_offset=0.0):
    k = w.map.keypoints[kp]
    w.set_robot_pose(k.x, k.y, k.yaw + yaw_offset)
    return {d.id: d for d in w.detections()}


def test_sees_the_alarm_clock_from_its_stand():
    for cam in ("head_sim", "ego_d435"):
        w = fresh_lite("procthor-train-38", camera=cam)
        seen = _at(w, "bedroom_bed_1b")
        assert "alarm_clock_1" in seen, cam
        d = seen["alarm_clock_1"]
        assert d.where == "bedroom_bed_1b" and d.method == "gt-geometric" and d.px >= 40


def test_not_seen_when_facing_away():
    w = fresh_lite("procthor-train-38")
    assert "alarm_clock_1" in _at(w, "bedroom_bed_1b")
    assert "alarm_clock_1" not in _at(w, "bedroom_bed_1b", math.pi)


def test_closed_container_hides_its_contents():
    w = fresh_lite("procthor-train-38")
    fridge = w.map.landmarks["fridge_1"]
    # stand 1.2 m in front of the fridge, facing it
    cx, cy = fridge.center[:2]
    for kp in w.map.keypoints.values():
        pass
    x, y = cx, cy - 1.3
    w.set_robot_pose(x, y, math.pi / 2)
    seen = {d.id for d in w.detections()}
    assert "fridge_1" in seen
    assert not {"egg_1", "egg_2", "potato_2", "potato_3"} & seen


def test_walls_occlude_other_rooms():
    """From the living-room start the kitchen counter's objects are beyond a wall: never visible."""
    w = fresh_lite("procthor-train-38")
    for dyaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        seen = {d.id for d in _at(w, "start", dyaw).values()}
        assert not {"apple_1", "fork_1", "potato_1", "alarm_clock_1", "egg_1"} & seen


def test_range_limit():
    w = fresh_lite("procthor-train-38")
    k = w.map.keypoints["bedroom_bed_1b"]
    # back off 2.5 m from the stand, facing the bed: the clock is > 2.5 m from the camera
    w.set_robot_pose(k.x - 2.6 * math.cos(k.yaw), k.y - 2.6 * math.sin(k.yaw), k.yaw)
    s = w.sightings()["alarm_clock_1"]
    assert not s.visible and s.reason in ("too_far", "occluded", "out_of_frustum")


def test_perception_is_thor_shaped_and_labelled():
    w = fresh_lite("procthor-train-38")
    k = w.map.keypoints["kitchen_counter_1c"]
    w.set_robot_pose(k.x, k.y, k.yaw)
    p = w.perception()
    assert p["source"] == "lite-ground-truth"
    assert p["objects"], "a counter full of things"
    for oid, o in p["objects"].items():
        assert set(o) >= {"type", "label", "where", "pose", "distance_m", "visible", "confidence", "source"}
        assert set(o["pose"]) == {"x", "y", "z"}
        assert o["confidence"] == 1.0 and o["visible"] is True
        # THOR frame: pose.y is the height, pose.z the map z (= isaac y)
        spec = w.map.objects[oid]
        assert o["pose"]["z"] == pytest.approx(spec.pos[1], abs=0.01)
        assert o["pose"]["y"] == pytest.approx(spec.pos[2] - w.map.floor_z, abs=0.01)
    assert p["people"]["user"]["keypoint"] == w.map.user_surface


def test_held_objects_are_reported_in_hand():
    w = fresh_lite("procthor-train-38")
    w.attach("mug_1", "left")
    p = w.perception()
    assert p["objects"]["mug_1"]["where"] == "hand:left"


def test_scan_views_and_frustum_consistency():
    """A scan's LookData: views in Worldline's frame; what a view saw lies inside api.observation.in_frustum."""
    from api.observation import LookData
    w = fresh_lite("procthor-train-38")
    k = w.map.keypoints["kitchen_counter_1b"]
    w.set_robot_pose(k.x, k.y, k.yaw)
    views = [w.view_spec(w.camera_pose(yaw_offset=math.radians(d))) for d in (-35, 0, 35)]
    look = w.scan(views, at="kitchen_counter_1b")
    assert isinstance(look, LookData) and len(look.views) == 3
    assert look.at == "kitchen_counter_1b"
    items = [it for lst in look.surfaces.values() for it in lst]
    assert items
    inside = 0
    for it in items:
        pos = it["pos"]
        if any(in_frustum(v, pos, pos[2]) for v in look.views):
            inside += 1
    assert inside >= 0.8 * len(items)
    # views carry the 3-D fields agent/state.in_view uses
    v0 = look.views[0]
    assert v0.vfov and v0.cam_h and 1.2 < v0.cam_h < 1.45


def test_glance_look_has_no_views():
    w = fresh_lite("procthor-train-38")
    k = w.map.keypoints["bedroom_bed_1b"]
    w.set_robot_pose(k.x, k.y, k.yaw)
    d = w.look([], at="bedroom_bed_1b")
    assert d["views"] == [] and "bedroom_bed_1b" in d["surfaces"]
    assert d["hands"] == {"left": None, "right": None}
    assert [it["id"] for it in d["surfaces"]["bedroom_bed_1b"]] == ["alarm_clock_1"]


def test_truth_snapshot():
    w = fresh_lite("procthor-train-38")
    t = w.truth()
    assert t.source == "lite-gt"
    assert t.objects["alarm_clock_1"]["where"] == "bedroom_bed_1b"
    assert "sofa" in t.things.values() and "painting" in t.things.values()
    sx, sy, syaw = w.scene_data.spawn
    assert t.robot["x"] == pytest.approx(sx, abs=1e-3) and t.robot["yaw"] == pytest.approx(
        coords.yaw_map_deg(syaw), abs=0.1)
