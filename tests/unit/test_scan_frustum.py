"""Absence is asserted only inside the 3-D frustum (PLAN 5.10): api/observation.in_frustum and
agent/state.BeliefState.apply_observation."""

from __future__ import annotations

from agent.state import BeliefState, Fact, ObjectBelief, in_view
from api.observation import HEAD_VFOV_DEG, ViewSpec, in_frustum, scan_views

CAM_H = 1.35


def view(tilt=15.0, **kw):
    return ViewSpec(x=0.0, z=0.0, yaw=0.0, tilt=tilt, fov=90.0, range=2.5, vfov=HEAD_VFOV_DEG, cam_h=CAM_H, **kw)


def test_front_edge_of_a_table_is_out_of_the_level_row_and_in_the_pitched_row():
    pos = (0.0, 0.35, 0.77)                      # 0.35 m ahead, on a 0.72 m top (+5 cm object)
    assert not in_frustum(view(15.0), pos)       # ~59 deg below horizontal: under the frame
    assert in_frustum(view(35.0), pos)           # the waist-pitched row sees it


def test_coffee_table_from_half_a_metre():
    pos = (0.0, 0.5, 0.45)
    assert not in_frustum(view(15.0), pos)
    assert in_frustum(view(35.0), pos)


def test_closer_than_near_is_never_absent_and_far_is_out_of_range():
    assert not in_frustum(view(35.0), (0.0, 0.2, 1.0))
    assert not in_frustum(view(15.0), (0.0, 2.6, 0.8))
    assert in_frustum(view(15.0), (0.0, 2.0, 0.8))


def test_a_thor_shaped_view_keeps_the_old_2d_rule():
    thor = {"x": 0.0, "z": 0.0, "yaw": 0.0, "tilt": 25.0, "fov": 90, "range": 1.5}
    assert in_view(thor, (0.0, 0.35))            # no vfov/cam_h: 2-D path
    assert in_view(thor, (0.0, 0.35), h=0.77)    # a height alone doesn't switch it on
    assert not in_view(thor, (0.0, 1.45))        # range - 0.1
    assert not in_view(thor, (1.0, 0.2))         # outside fov/2 - 5


def test_scan_views_cover_two_rows_of_three_yaws():
    vs = scan_views(1.0, 2.0, 90.0, CAM_H)
    assert len(vs) == 6 and {v.tilt for v in vs} == {15.0, 35.0}
    assert sorted({round(v.yaw) for v in vs}) == [55, 90, 125]


def belief_with(oid, where, pose):
    b = BeliefState()
    b.objects[oid] = ObjectBelief(oid, oid.rsplit("_", 1)[0], None, None, Fact(where, "look", 1.0, True))
    b.objects[oid].pose = pose
    return b


def test_apply_observation_marks_absent_only_inside_the_frustum():
    level = [view(15.0).to_dict()]
    b = belief_with("mug_1", "coffee_table_1a", {"x": 0.0, "z": 0.5, "h": 0.45})
    b.apply_observation({"at": "sofa_1a", "surfaces": {}, "views": level, "hands": {"left": None, "right": None}}, 5.0)
    assert b.objects["mug_1"].where.value == "coffee_table_1a"          # out of frame: unchanged
    both = [view(15.0).to_dict(), view(35.0).to_dict()]
    b.apply_observation({"at": "sofa_1a", "surfaces": {}, "views": both, "hands": {"left": None, "right": None}}, 6.0)
    assert b.objects["mug_1"].where.value == "UNKNOWN" and b.objects["mug_1"].where.source == "look_absent"


def test_surface_height_from_the_map_is_used_when_the_object_has_none():
    b = belief_with("mug_1", "coffee_table_1a", None)
    xy = {"coffee_table_1a": (0.0, 0.5)}
    h = {"coffee_table_1a": 0.45}
    b.apply_observation({"at": "sofa_1a", "surfaces": {}, "views": [view(15.0).to_dict()],
                         "hands": {"left": None, "right": None}}, 5.0, xy, h)
    assert b.objects["mug_1"].where.value == "coffee_table_1a"


def test_a_glance_never_marks_anything_absent():
    b = belief_with("mug_1", "counter_1a", {"x": 0.0, "z": 1.0, "h": 0.9})
    b.apply_observation({"at": "counter_1a", "surfaces": {}, "views": [], "hands": {"left": None, "right": None}}, 5.0)
    assert b.objects["mug_1"].where.value == "counter_1a"


def test_observation_positions_carry_height():
    b = BeliefState()
    b.apply_observation({"at": "counter_1a", "surfaces": {"counter_1a": [
        {"id": "mug_1", "type": "mug", "pos": [1.0, 2.0, 0.93]}]}, "views": [], "hands": {"left": None, "right": None}},
        3.0)
    assert b.objects["mug_1"].pose == {"x": 1.0, "z": 2.0, "h": 0.93}
