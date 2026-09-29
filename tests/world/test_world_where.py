"""world/where.py: geometric receptacles with THOR's vocabulary (PLAN §6.2.2)."""

import pytest

from tests.fakes.fixtures import fresh_lite


@pytest.fixture(scope="module")
def w38():
    return fresh_lite("procthor-train-38")


@pytest.mark.parametrize("oid,where,rule", [
    ("alarm_clock_1", "bedroom_bed_1b", "surface_footprint"),     # on the bed, below its AABB top
    ("apple_1", "kitchen_counter_1c", "surface_top"),
    ("egg_1", "fridge", "container"),
    ("potato_2", "fridge", "container"),
    ("book_1", "bedroom_dresser_1c", "surface_top"),
    ("book_2", "living_room_tv_stand_1a", "surface_footprint"),    # a lower shelf of the TV stand
    ("bottle_1", "kitchen_dining_table_1b", "surface_top"),
    ("baseball_bat_1", "floor", "floor"),
])
def test_where_house38(w38, oid, where, rule):
    o = w38.object(oid)
    assert (o.where, o.where_rule) == (where, rule)


def test_where_other_receptacles_house15():
    w = fresh_lite("procthor-train-15")
    assert w.object("toilet_paper_1").where == "toilet"       # THOR: the parent's snake type
    assert w.object("bottle_1").where == "chair"
    assert w.object("apple_1").where == "kitchen_counter_1b"


def test_every_object_has_a_known_where(lite_house):
    wheres = {oid: o.where for oid, o in lite_house.objects().items()}
    assert sum(1 for w in wheres.values() if w == "unknown") == 0, wheres
    surfaces = set(lite_house.map.surfaces)
    on_surface = [w for w in wheres.values() if w in surfaces]
    assert len(on_surface) >= len(wheres) // 2


def test_attach_detach_moves_where():
    w = fresh_lite("procthor-train-38")
    k = w.map.keypoints["kitchen_dining_table_1b"]
    w.set_robot_pose(k.x, k.y, k.yaw)
    w.attach("bottle_1", "right")
    assert w.object("bottle_1").where == "hand:right"
    assert w.hands() == {"left": None, "right": "bottle_1"}
    # it follows the robot
    x0 = w.object("bottle_1").pos[0]
    w.set_robot_pose(k.x + 1.0, k.y, k.yaw)
    assert w.object("bottle_1").pos[0] == pytest.approx(x0 + 1.0)
    spot = w.free_spot("kitchen_dining_table_1a", "bottle_1")
    assert spot is not None
    w.detach("bottle_1", spot)
    o = w.object("bottle_1")
    assert o.where == "kitchen_dining_table_1a" and o.held_by is None
    assert w.hands() == {"left": None, "right": None}


def test_drop_goes_to_floor():
    w = fresh_lite("procthor-train-38")
    w.attach("mug_1", "left")
    w.detach("mug_1", None)
    assert w.object("mug_1").where == "floor"


def test_free_spot_avoids_other_objects():
    w = fresh_lite("procthor-train-38")
    s = "kitchen_counter_1c"
    spot = w.free_spot(s, "cup_1")
    assert spot is not None
    others = [o for o in w.objects().values() if o.where == s and o.oid != "cup_1"]
    for o in others:
        b = o.box
        inside = b[0][0] - 0.02 < spot.x < b[1][0] + 0.02 and b[0][1] - 0.02 < spot.y < b[1][1] + 0.02
        assert not inside, o.oid
    # a reach predicate that refuses everything -> None
    assert w.free_spot(s, "cup_1", within=lambda x, y, z: False) is None
