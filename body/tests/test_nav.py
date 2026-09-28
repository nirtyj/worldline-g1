import math

import numpy as np
import pytest

from body.nav_grid import NavGrid, path_length
from tools.fake_p1 import ORIGIN, RES, build_occupancy


def grid(**kw):
    return NavGrid(build_occupancy(), RES, ORIGIN, robot_radius=0.25, **kw)


def test_coordinates_roundtrip():
    g = grid()
    iy, ix = g.world_to_cell(1.23, 4.56)
    x, y = g.cell_to_world(iy, ix)
    assert abs(x - 1.23) <= RES and abs(y - 4.56) <= RES
    assert not g.is_free(4.0, 0.5)   # wall x=4
    assert g.is_free(2.0, 1.5)


def test_plan_through_doors_keeps_clearance():
    g = grid()
    r = g.plan((1.8, 1.4), (8.0, 4.5))   # kitchen -> living room, must use a door in the x=4 wall
    assert r.ok, r.reason
    p = r.path
    assert np.linalg.norm(p[0] - [1.8, 1.4]) < 1e-6 and np.linalg.norm(p[-1] - [8.0, 4.5]) < 1e-6
    for a, b in zip(p[:-1], p[1:]):
        assert g.segment_free(a, b) or np.linalg.norm(a - [1.8, 1.4]) < 0.3
    crossing = [q for q in p if 3.9 <= q[0] <= 4.1]
    assert crossing and all((1.0 < q[1] < 2.0) or (4.0 < q[1] < 5.0) for q in crossing)
    assert r.length >= np.linalg.norm(np.array([8.0, 4.5]) - [1.8, 1.4])
    assert r.plan_ms < 2000


def test_plan_around_furniture_and_snap():
    g = grid()
    r = g.plan((5.5, 3.0), (8.4, 3.0))   # coffee table at x 6.6..7.6, y 2.3..3.5 lies across the straight line
    assert r.ok
    assert not g.segment_free((5.5, 3.0), (8.4, 3.0))
    assert r.length > 2.9 + 0.3
    r2 = g.plan((5.5, 3.0), (7.1, 2.9), snap_radius=1.0)  # goal inside the table -> snapped out
    assert r2.ok and r2.goal_snapped is not None and g.is_free(*r2.goal_snapped)
    r3 = g.plan((5.5, 3.0), (7.1, 2.9), snap_radius=0.1)
    assert not r3.ok and r3.reason == "goal_in_obstacle"


def test_unreachable_and_virtual_obstacle():
    g = grid()
    r = g.plan((1.8, 1.4), (20.0, 20.0))
    assert not r.ok
    base = g.plan((5.0, 1.5), (5.0, 4.5))
    assert base.ok
    g.add_virtual_obstacle(5.0, 3.0, 0.8)
    r2 = g.plan((5.0, 1.5), (5.0, 4.5))
    assert r2.ok and r2.length > base.length
    g.clear_virtual()
    assert g.plan((5.0, 1.5), (5.0, 4.5)).length == pytest.approx(base.length, rel=1e-6)


def test_from_p1_reply_layouts(tmp_path):
    occ = build_occupancy()
    p = tmp_path / "o.npz"
    np.savez(p, occupancy=occ.T.astype(np.float32))  # x-major float omap style
    g = NavGrid.from_p1_reply({"ok": True, "path": str(p), "resolution": RES, "origin": list(ORIGIN),
                               "layout": "xy"})
    assert g.raw.shape == occ.shape and (g.raw == (occ > 0)).all()
    assert g.plan((1.8, 1.4), (2.0, 4.0)).ok


def test_ray_free_distance():
    g = grid()
    d = g.ray_free_distance(5.0, 1.5, 0.0, max_d=8.0)   # toward +x in the living room, stops at sofa/wall
    assert 3.0 < d < 5.0
