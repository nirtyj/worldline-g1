"""The GR00T stance (M2b wave 2, W2.5; docs/groot_serving.md §8, config/skills.yaml): the ego_view pinhole model it
comes from, and its encoding against services/reachability.py's window check.

Pinned: the camera model reproduces the P1 mount (1.14 m at SONIC's stand); the stance (0.30 m ahead, 0.10 m left)
puts a low object below the upper third on the 0.78 m table, the 0.94 m counter and the 0.97 m dresser, and 0.40 m
does not on the counter or the dresser; only the apple skill carries the stance (the `any` skill would pin every
pickable to it, and a stance is strict today); ReachabilityModel.in_window accepts the pose the stance describes
(heading 18.4 deg off the bearing) and rejects other offsets.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from groot.stance import CAM_IN_PELVIS, PELVIS_Z, ego_view_uv, in_lower_two_thirds
from services.reachability import ReachabilityModel
from services.skills import load_skill_specs

SKILLS = {s.skill_id: s for s in load_skill_specs()}
APPLE = SKILLS["groot.pick.apple.arena_static_experimental.v0"]
ANY = SKILLS["groot.pick.any.arena_static_experimental.v0"]

# target centre heights of H40's three W2.5 surfaces with a low object on them (table 0.78, counter 0.94, dresser 0.97)
TABLE, COUNTER, DRESSER = 0.812, 0.981, 1.012


def test_the_camera_sits_where_p1_puts_it():
    assert PELVIS_Z + CAM_IN_PELVIS[2] == pytest.approx(1.14, abs=0.005)       # p1_m2b.md §5.1: ~1.14 m standing
    u, v = ego_view_uv(CAM_IN_PELVIS[0] + math.cos(math.radians(35)), 0.0, PELVIS_Z + CAM_IN_PELVIS[2]
                       - math.sin(math.radians(35)))
    assert (u, v) == (pytest.approx(320.0), pytest.approx(240.0))              # on the optical axis: the centre


def test_the_stance_frames_a_low_object_like_arena_on_all_three_surfaces():
    st = APPLE.stance
    f, l = st["stand_off_m"], st["lateral_m"]
    for z in (TABLE, COUNTER, DRESSER):
        u, v = ego_view_uv(f, l, z)
        assert in_lower_two_thirds(v) and 0.0 < u < 320.0, (z, u, v)          # lower two-thirds, left half
    for z in (COUNTER, DRESSER):                                                # 0.10 m farther: the upper third
        assert not in_lower_two_thirds(ego_view_uv(f + 0.10, l, z)[1])


def test_only_the_apple_skill_carries_the_stance():
    assert APPLE.stance["stand_off_m"] == 0.30 and APPLE.stance["lateral_m"] == 0.10
    assert ANY.stance == {}


def test_reachability_accepts_the_pose_the_stance_describes():
    rm = ReachabilityModel(world=None)
    pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    f, l = APPLE.stance["stand_off_m"], APPLE.stance["lateral_m"]
    assert rm.in_window(f, l, APPLE, pose, (f, l))                             # bearing 18.4 deg off the heading
    assert rm.in_window(f + 0.04, l - 0.04, APPLE, pose, (f + 0.04, l - 0.04))   # an arrival error inside tol
    assert not rm.in_window(f + 0.10, l, APPLE, pose, (f + 0.10, l))
    assert not rm.in_window(f, l + 0.10, APPLE, pose, (f, l + 0.10))
    r = math.hypot(f, l)                                                        # facing the object: it sits straight
    assert not rm.in_window(r, 0.0, APPLE, pose, (r, 0.0))                     # ahead, not 0.10 m left
