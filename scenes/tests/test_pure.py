"""Pure-Python tests for scenes/ (no Isaac): ids, grid maths, inflation, spawn, npz round trip,
and (when the cached assets exist, i.e. on the box) the generated house files.

    /work/envs/isaaclab/bin/python -m pytest -q scenes/tests/test_pure.py
    /work/envs/isaaclab/bin/python -m scenes.tests.test_pure          # without pytest
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from scenes.catalog import DEFAULT_HOUSES, parse_house_id, procthor_rooms
from scenes.occupancy import (
    FREE,
    OBSTACLE,
    OUTSIDE,
    choose_spawn,
    connectivity,
    finish,
    get_occupancy,
    load_npz,
    make_grid_frame,
    room_points,
)


def test_parse_house_id():
    for s in ("procthor-train-40", "procthor-10k-train-40", "train_40", "procthor-10k-train/40"):
        r = parse_house_id(s)
        assert r.house_id == "procthor-train-40" and r.scene_dir == "train_40"
        assert r.archive == "procthor-10k-train_train_40.tar.zst"
    for s in ("ithor-FloorPlan10", "FloorPlan10", "FloorPlan10_physics"):
        r = parse_house_id(s)
        assert r.house_id == "ithor-FloorPlan10" and r.scene_dir == "FloorPlan10_physics"
    try:
        parse_house_id("kitchen")
        raise AssertionError("should fail")
    except ValueError:
        pass


def test_procthor_rooms_named_like_worldline():
    rooms = procthor_rooms(parse_house_id("procthor-train-40"))
    assert {r["name"] for r in rooms} == {"bedroom", "kitchen", "living_room"}
    k = next(r for r in rooms if r["name"] == "kitchen")
    assert k["room_id"] == 6 and len(k["polygon_thor"]) >= 4


def _synthetic():
    # 6 m x 4 m box with a wall at x=3 and a 1.0 m door at y in [1.5, 2.5]
    res = 0.05
    origin, shape = make_grid_frame([[0, 0], [6, 4]], res, pad=0.0)
    raw = np.zeros(shape, np.uint8)
    raw[0, :] = raw[-1, :] = OBSTACLE
    raw[:, 0] = raw[:, -1] = OBSTACLE
    ix = int(3.0 / res)
    raw[:, ix] = OBSTACLE
    iy0, iy1 = int(1.5 / res), int(2.5 / res)
    raw[iy0:iy1, ix] = FREE
    rooms = [
        {"name": "left", "room_id": 1, "polygon": [[0, 0], [3, 0], [3, 4], [0, 4]]},
        {"name": "right", "room_id": 2, "polygon": [[3, 0], [6, 0], [6, 4], [3, 4]]},
    ]
    return finish("synthetic", raw, origin, res, 0.1, 1.6, 0.30, "test"), rooms


def test_grid_convention_and_inflation():
    occ, _ = _synthetic()
    iy, ix = occ.world_to_cell(1.0, 2.0)
    x, y = occ.cell_to_world(iy, ix)
    assert abs(x - 1.0) <= occ.resolution and abs(y - 2.0) <= occ.resolution
    assert occ.is_free(1.5, 2.0)
    assert not occ.is_free(3.0, 0.5)  # wall
    assert not occ.is_free(0.1, 2.0)  # within robot radius of the outer wall
    assert occ.is_free(3.0, 2.0)  # door centre: 0.5 m half-width > 0.30 m radius
    assert abs(occ.clearance(1.5, 2.0) - 1.45) < 0.1


def test_connectivity_spawn_room_points():
    occ, rooms = _synthetic()
    c = connectivity(occ, rooms, (1.5, 2.0))
    assert c["all_rooms_reachable"], c
    # close the door -> right room unreachable
    raw = occ.raw.copy()
    raw[:, int(3.0 / occ.resolution)] = OBSTACLE
    closed = finish("closed", raw, occ.origin, occ.resolution, 0.1, 1.6, 0.30, "test")
    c2 = connectivity(closed, rooms, (1.5, 2.0))
    assert not c2["rooms"]["right"]["reachable"] and c2["rooms"]["left"]["reachable"]
    sp = choose_spawn(occ, rooms, preferred={"x": 1.5, "y": 2.0, "yaw": 0.0})
    assert sp["source"] == "procthor_agent_start" and sp["forward_free_m"] >= 2.5
    rp = room_points(occ, rooms)
    assert all(p["ok"] for p in rp) and all(p["clearance_m"] > 0.5 for p in rp)
    # a straight run of >= 2 m exists from each room point
    assert all(occ.free_ray(p["x"], p["y"], p["yaw"]) >= 1.0 for p in rp)


def test_npz_roundtrip():
    occ, _ = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        occ.save(Path(d))
        o2 = load_npz(Path(d) / "occupancy.npz", "synthetic")
        assert np.array_equal(o2.raw, occ.raw) and np.array_equal(o2.inflated, occ.inflated)
        assert o2.origin == occ.origin and math.isclose(o2.resolution, occ.resolution)


def test_cached_houses_if_present():
    """On the box: every generated house has a grid whose spawn is free and all rooms reachable."""
    import json

    n = 0
    for h in DEFAULT_HOUSES:
        ref = parse_house_id(h)
        if not (ref.assets_dir / "occupancy.npz").exists():
            continue
        n += 1
        occ = get_occupancy(h)
        info = json.loads((ref.assets_dir / "house_info.json").read_text())
        sp = info["spawn"]
        assert occ.is_free(sp["x"], sp["y"]), (h, sp)
        assert info["connectivity"]["all_rooms_reachable"], (h, info["connectivity"])
        assert (occ.raw == OBSTACLE).any() and (occ.raw == OUTSIDE).any()
    print(f"checked {n} cached houses")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
