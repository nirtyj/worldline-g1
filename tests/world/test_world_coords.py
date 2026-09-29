"""world/coords.py: the only Isaac <-> Worldline conversion (PLAN §6.2.3)."""

import math

import pytest

from world import coords


@pytest.mark.parametrize("isaac_yaw,map_yaw", [(math.pi / 2, 0.0), (0.0, 90.0), (math.pi, 270.0),
                                                 (-math.pi / 2, 180.0)])
def test_yaw_conversion(isaac_yaw, map_yaw):
    assert coords.yaw_map_deg(isaac_yaw) == pytest.approx(map_yaw)
    assert coords.wrap_pi(coords.yaw_isaac_rad(map_yaw) - isaac_yaw) == pytest.approx(0.0, abs=1e-9)


def test_heading_matches_worldline_atan2_dx_dz():
    # Worldline heading = atan2(dx, dz); map z = isaac y
    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-2, 0.5)]:
        assert coords.heading_map_deg(dx, dy) == pytest.approx(
            coords.yaw_map_deg(coords.heading_isaac_rad(dx, dy)), abs=1e-9)


def test_right_hand_rule_matches_layout():
    """Facing +y (map yaw 0), an object at +x must be on the robot's right (agent/layout.py Line.along:
    right = (facing[1], -facing[0]) with facing = (sin(yaw), cos(yaw)) in map (x, z))."""
    yaw = math.radians(coords.yaw_map_deg(math.pi / 2))
    facing = (math.sin(yaw), math.cos(yaw))
    right = (facing[1], -facing[0])
    obj = coords.to_map_xz(1.0, 0.0)
    assert obj[0] * right[0] + obj[1] * right[1] > 0.99
    # and REP-103's body frame agrees: +x world is -y (right) in the body frame when facing +y
    fwd, left = coords.world_to_body((0.0, 0.0, math.pi / 2), 1.0, 0.0)
    assert left == pytest.approx(-1.0) and fwd == pytest.approx(0.0, abs=1e-12)


def test_thor_pose_swaps_height():
    assert coords.thor_pose(1.0, 2.0, 0.75) == {"x": 1.0, "y": 0.75, "z": 2.0}
    assert coords.map_pos(1.234, 5.678, 0.9) == [1.23, 5.68, 0.9]


def test_body_world_roundtrip():
    pose = (2.0, -1.0, 0.7)
    wx, wy = coords.body_to_world(pose, 0.4, -0.2)
    assert coords.world_to_body(pose, wx, wy) == pytest.approx((0.4, -0.2))
