"""world/mapgen.py on recorded MolmoSpaces houses (procthor-train-38, -40, -15; iTHOR FloorPlan10).

Checks: keypoint/edge sanity (stands on free cells, edges symmetric and finite, names unique), THOR naming,
stand geometry for a G1 (0.45-0.60 m from the edge, facing its normal, same room), the lookup_keypoints() shape
the runtime reads, and that agent/layout.py can build LAYOUT lines from it.
"""

import math
import re

import numpy as np
import pytest

from tests.fakes.fixtures import PROCTHOR, RECORDED
from world import coords


@pytest.mark.parametrize("house", RECORDED)
def test_stands_on_free_cells_in_their_room(cached_map, house):
    m = cached_map(house)
    g = m.grid
    assert m.surfaces, "no surfaces"
    for s in m.surfaces.values():
        x, y = s.stand
        assert g.is_free(x, y), s.name
        assert g.clearance(x, y) >= m.params.relaxed_clearance_m - 1e-6, s.name
        if s.room:
            assert m.room_at(x, y) == s.room, (s.name, m.room_at(x, y))
        lo, hi = m.params.stand_off_m
        assert lo - 1e-6 <= s.stand_off_m <= (hi if not s.relaxed else m.params.stand_off_relaxed_m) + 1e-6
        # facing the stretch: stand -> centre within 20 deg of the stand yaw (layout.py snaps within 30)
        ang = math.atan2(s.center[1] - y, s.center[0] - x)
        assert abs(coords.ang_diff(ang, s.yaw)) <= math.radians(m.params.max_facing_err_deg) + 1e-6, s.name
        # stands never inside the furniture footprint
        assert not s.contains_xy(x, y)


@pytest.mark.parametrize("house", RECORDED)
def test_names_unique_and_keypoint_is_surface(cached_map, house):
    m = cached_map(house)
    names = list(m.keypoints)
    assert len(names) == len(set(names))
    for s in m.surfaces:
        assert s in m.keypoints and m.keypoints[s].kind == "surface"
    assert "start" in m.keypoints
    kmap = m.lookup_keypoints()
    for s, v in kmap["surfaces"].items():
        assert v["keypoints"] == [s]
    oids = list(m.objects)
    assert len(oids) == len(set(oids))
    for oid in oids:
        assert re.fullmatch(r"[a-z][a-z0-9_]*_\d+", oid), oid


@pytest.mark.parametrize("house", RECORDED)
def test_edges_symmetric_finite_and_complete(cached_map, house):
    m = cached_map(house)
    names = list(m.keypoints)
    pts = [(m.keypoints[n].x, m.keypoints[n].y) for n in names]
    D = m.grid.distances(pts)
    assert np.all(np.isfinite(D)), "some keypoint is unreachable"
    assert np.allclose(D, D.T, atol=0.02)
    n = len(names)
    assert len(m.edges) == n * (n - 1) // 2
    for a, b, d in m.edges:
        assert a != b and math.isfinite(d) and d > 0
        # geodesic >= straight line
        ka, kb = m.keypoints[a], m.keypoints[b]
        assert d >= math.hypot(ka.x - kb.x, ka.y - kb.y) - 0.15
    # edges roughly agree with the body's own planner (<= 20 % apart on a sample)
    for a, b, d in m.edges[:: max(1, len(m.edges) // 10)]:
        L = m.grid.path_length((m.keypoints[a].x, m.keypoints[a].y), (m.keypoints[b].x, m.keypoints[b].y))
        assert L is not None and abs(L - d) <= 0.2 * max(d, 1.0) + 0.3, (a, b, d, L)


@pytest.mark.parametrize("house", PROCTHOR)
def test_lookup_keypoints_shape(cached_map, house):
    m = cached_map(house)
    k = m.lookup_keypoints()
    for key in ("scene", "keypoints", "edges", "surfaces", "people", "rooms", "nav_speed_mps",
                "max_reach_height_m", "min_reach_height_m", "robot"):
        assert key in k
    assert k["robot"] == "unitree_g1"
    user = k["people"]["user"]
    assert user["deliver_to_surface"] in k["surfaces"] and user["keypoint"] == user["deliver_to_surface"]
    for name, kp in k["keypoints"].items():
        assert len(kp["xy"]) == 2 and "desc" in kp and "room" in kp and "yaw" in kp
    for room, r in k["rooms"].items():
        assert room in k["keypoints"], "every room has its own keypoint (PLAN 5.2)"
        assert k["keypoints"][room]["kind"] == "room"
        assert room in r["spots"]
    # surface names are room-prefixed THOR names
    for s in k["surfaces"]:
        assert any(s.startswith(r + "_") for r in k["rooms"]), s


def test_thor_names_house38(cached_map):
    m = cached_map("procthor-train-38")
    assert {"kitchen_counter_1a", "kitchen_dining_table_1a", "bedroom_bed_1a", "bedroom_dresser_1a",
            "living_room_tv_stand_1a", "living_room_sofa_1a"} <= set(m.surfaces)
    assert {"alarm_clock_1", "apple_1", "book_1", "book_2", "mug_1"} <= set(m.objects)
    assert "banana_1" not in m.objects
    assert {"fridge_1", "tv_1", "plant_1"} <= set(m.landmarks)
    assert m.surfaces["kitchen_counter_1a"].desc == "counter 1, part a in the kitchen"


def test_thor_names_house40_and_15(cached_map):
    m40 = cached_map("procthor-train-40")
    assert "alarm_clock_1" in m40.objects and {"book_1", "book_2"} <= set(m40.objects)
    assert "banana_1" not in m40.objects                         # eval missing_object
    assert any(s.startswith("kitchen_dining_table_1") for s in m40.surfaces)
    m15 = cached_map("procthor-train-15")
    assert "apple_1" in m15.objects and "microwave_1" in m15.landmarks   # eval fetch_search / unsupported


def test_ithor_names_have_no_room_prefix(cached_map):
    m = cached_map("ithor-FloorPlan10")
    assert "counter_1a" in m.surfaces and "stove_1" in m.landmarks and "spatula_1" in m.objects


@pytest.mark.parametrize("house", PROCTHOR)
def test_landmarks_near_same_room(cached_map, house):
    m = cached_map(house)
    for lm in m.landmarks.values():
        assert lm.near in m.keypoints
        if lm.room and any(k.room == lm.room for k in m.keypoints.values()):
            assert m.keypoints[lm.near].room == lm.room, (lm.name, lm.near)


def test_user_surface_override(lite38):
    from world.mapgen import build_static_map
    m = build_static_map(lite38.scene_data, lite38.grid, user_surface="kitchen_dining_table_1b")
    assert m.user_surface == "kitchen_dining_table_1b"
    m2 = build_static_map(lite38.scene_data, lite38.grid, user_surface="nope")
    assert m2.user_surface in m2.surfaces


@pytest.mark.parametrize("house", PROCTHOR)
def test_layout_builds_from_the_map(cached_map, house):
    """agent/layout.py (kept from Worldline) must accept the map: surfaces face an axis so lines form."""
    layout = pytest.importorskip("agent.layout")
    m = cached_map(house)
    kmap = m.lookup_keypoints()
    landmarks = {n: {"label": l.label, "near": l.near, "pos": list(coords.to_map_xz(*l.center[:2]))}
                 for n, l in m.landmarks.items()}
    lines = layout.build(kmap, landmarks)
    in_lines = {t.name for ln in lines for t in ln.things}
    # every stretch of a multi-stretch furniture sits on a line (its neighbours are on the same axis), except one
    # cut to an L/U footprint (R.5): its stand faces the counter part that is really there, off its siblings' line
    multi = [s for s in m.surfaces.values() if s.part]
    assert multi
    missing = [s.name for s in multi if s.name not in in_lines and not _cut(m, s)]
    assert not missing, missing
    assert layout.render(lines, kmap)


def test_layout_for_ui_and_occupancy(cached_map):
    m = cached_map("procthor-train-38")
    ui = m.layout_for_ui()
    assert ui["grid_step"] == 0.25 and len(ui["grid"]) > 100
    assert set(ui["keypoints"]) == set(m.keypoints)
    occ = m.occupancy
    assert occ["raw"].shape == occ["inflated"].shape and occ["resolution"] == pytest.approx(0.05)


# ------------------------------------------------------------------ R.5: stretches from footprints
def _cut(m, s) -> bool:
    """The stretch's cross-section is narrower than its furniture's AABB (the footprint cut moved an edge)."""
    f = next((t for t in m.things if t.scene_id == s.furniture_id), None)
    if f is None:
        return False
    (x0, y0, _), (x1, y1, _) = f.aabb
    along_x = (x1 - x0) >= (y1 - y0)
    full = (y1 - y0) / 2 if along_x else (x1 - x0) / 2
    got = s.half[1] if along_x else s.half[0]
    return got < full - 0.05


@pytest.mark.parametrize("house", RECORDED)
def test_every_stand_faces_its_furniture(cached_map, house):
    """R.5: the 10 cm of a stretch nearest its stand is mostly the furniture's footprint (walls and outside cells
    excluded from the count). Before R.5, the stands of L/U-shaped counters faced the empty part of the AABB (H40's
    user surface kitchen_counter_1a: 1.2 m of floor between the stand and the counter)."""
    from tests.fakes.fixtures import _lite
    from world.mapgen import Footprint, embedded_items
    w = _lite(house)
    m = w.map
    for s in m.surfaces.values():
        f = w.scene_data.item(s.furniture_id)
        fp = Footprint(m.grid, f, embedded_items(w.scene_data, f))
        (cx, cy), (hx, hy) = s.center, s.half
        side = {"+x": ((cx + hx - 0.05, cy), (0.05, hy)), "-x": ((cx - hx + 0.05, cy), (0.05, hy)),
                "+y": ((cx, cy + hy - 0.05), (hx, 0.05)), "-y": ((cx, cy - hy + 0.05), (hx, 0.05))}[s.side]
        assert fp.fill(*side) >= 0.5, (s.name, s.side, round(fp.fill(*side), 2))


def test_h15_u_counter_stands_inside_the_u(cached_map):
    """M2 gap 1: the apple's stretch (kitchen_counter_1b, the back run of a U-shaped counter) has its stand inside the
    U facing the back run, not on the open side of the 1.84 m-deep AABB 2.1 m from the apple."""
    m = cached_map("procthor-train-15")
    s = m.surfaces["kitchen_counter_1b"]
    apple = m.objects["apple_1"]
    assert s.contains_xy(*apple.pos[:2])
    assert 2 * s.half[1] < 0.7, "cut to the back run's depth"
    k = m.keypoints["kitchen_counter_1b"]
    assert math.dist((k.x, k.y), apple.pos[:2]) < 1.0
    assert s.stand_off_m <= 0.5 and not s.relaxed


def test_k10_no_stretch_covers_the_stove(cached_map):
    """M2 gap 1 (K10): counter_2 is cut at the built-in stove; the spatula's stretch and the other side flank it."""
    m = cached_map("ithor-FloorPlan10")
    stove = m.landmarks["stove_1"]
    (sx0, sy0, _), (sx1, sy1, _) = stove.aabb
    for s in m.surfaces.values():
        if s.furniture_id != m.surfaces["counter_2a"].furniture_id:
            continue
        ov = max(0.0, min(sx1, s.center[1] + s.half[1]) - max(sy0, s.center[1] - s.half[1])) \
            if False else max(0.0, min(sy1, s.center[1] + s.half[1]) - max(sy0, s.center[1] - s.half[1]))
        assert ov <= 0.06, (s.name, ov)
    assert {"counter_2a", "counter_2b", "counter_2c"} <= set(m.surfaces)
    ys = sorted((m.surfaces[n].center[1], n) for n in ("counter_2a", "counter_2b"))
    assert ys[0][1] == "counter_2a" and ys[0][0] < (sy0 + sy1) / 2 < ys[1][0], "2a and 2b flank the stove"
    assert stove.near in ("counter_2a", "counter_2b"), "a stand that faces the stove"


def test_footprints_off_is_thors_aabb_partition(lite38):
    """MapParams(footprints=False) reproduces the THOR partition exactly (the AABB stretches)."""
    import dataclasses
    from world.mapgen import MapParams, build_static_map
    old = build_static_map(lite38.scene_data, lite38.grid, dataclasses.replace(MapParams(), footprints=False))
    for s in old.surfaces.values():
        f = lite38.scene_data.item(s.furniture_id)
        (x0, y0, _), (x1, y1, _) = f.aabb
        along_x = (x1 - x0) >= (y1 - y0)
        assert (s.half[1] if along_x else s.half[0]) == pytest.approx(((y1 - y0) if along_x else (x1 - x0)) / 2)


# ------------------------------------------------------------------ free sides (free-standing tables and islands)
def _isaac_params(**over):
    """config/g1.yaml mapgen as an Isaac world builds it (the R.7 stands, not mapgen.lite_world)."""
    import dataclasses
    from robot.profile import load_profile
    from world.mapgen import MapParams
    mp = MapParams.from_dict({k: v for k, v in load_profile("lite").g1["mapgen"].items() if k != "lite_world"})
    return dataclasses.replace(mp, **over)


def _synthetic(tmp_path, **over):
    from tests.fakes import table_house
    from world.lite_world import LiteWorld
    return LiteWorld(table_house.write(tmp_path / "synthetic-table"), map_params=_isaac_params(**over))


def test_a_free_standing_table_gets_a_stand_on_every_free_side_and_a_wall_counter_keeps_one(tmp_path):
    w = _synthetic(tmp_path)
    m = w.map
    lo, hi = m.params.stand_off_m
    table = {n: s for n, s in m.surfaces.items() if n.startswith("kitchen_dining_table_1")}
    # its own stands (one per stretch, the sibling-side rule) first, then every other free side, lettered after them
    assert {n: (s.side, s.extra_side) for n, s in table.items()} == {
        "kitchen_dining_table_1a": ("+y", False), "kitchen_dining_table_1b": ("+y", False),
        "kitchen_dining_table_1c": ("-y", True), "kitchen_dining_table_1d": ("-x", True),
        "kitchen_dining_table_1e": ("-y", True), "kitchen_dining_table_1f": ("+x", True)}
    for s in table.values():
        assert not s.relaxed and lo - 1e-6 <= s.stand_off_m <= hi + 1e-6, s
        assert m.grid.clearance(*s.stand) >= m.params.stand_clearance_m - 1e-6
        ang = math.atan2(s.center[1] - s.stand[1], s.center[0] - s.stand[0])
        assert abs(coords.ang_diff(ang, s.yaw)) <= math.radians(m.params.max_facing_err_deg) + 1e-6, s.name
    pts = [m.keypoints[n] for n in table] + [m.keypoints[n] for n in ("kitchen_counter_1a", "kitchen_counter_1b")]
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            assert math.dist((a.x, a.y), (b.x, b.y)) >= m.params.stand_min_sep_m - 1e-6, (a.name, b.name)
    # the counter against the south wall: its back has no room, so one stand per stretch, on its front
    counter = {n: s.side for n, s in m.surfaces.items() if n.startswith("kitchen_counter_1")}
    assert counter == {"kitchen_counter_1a": "+y", "kitchen_counter_1b": "+y"}
    # objects keep the stretch's own name as their where; the user surface is never an added stand
    assert w.objects()["bottle_1"].where == "kitchen_dining_table_1b" and not m.surfaces[m.user_surface].extra_side
    # the flag off: the one-stand-per-stretch map
    off = _synthetic(tmp_path / "off", free_sides=False).map
    assert [n for n in off.surfaces if n.startswith("kitchen_dining_table")] == ["kitchen_dining_table_1a",
                                                                                "kitchen_dining_table_1b"]


@pytest.mark.parametrize("house", RECORDED)
@pytest.mark.parametrize("params", ["default", "isaac"])
def test_free_sides_keep_every_existing_stand_and_name(house, params):
    """Every stand of the one-stand-per-stretch map (free_sides off) is in the new map under the same name, at the
    same pose, facing the same stretch; the added ones come after them (the furniture's next letters), and the user
    surface and the skipped list do not change. So eval bindings, memory and eval/scenes.yaml keep their names."""
    import dataclasses
    from tests.fakes.fixtures import _lite
    from world.mapgen import EXTRA_LETTERS, MapParams, build_static_map
    w = _lite(house)
    mp = MapParams() if params == "default" else _isaac_params()
    on = build_static_map(w.scene_data, w.grid, dataclasses.replace(mp, free_sides=True), scene_key=house)
    off = build_static_map(w.scene_data, w.grid, dataclasses.replace(mp, free_sides=False), scene_key=house)
    for n, s in off.surfaces.items():
        t = on.surfaces[n]
        assert (t.stand, t.yaw, t.side, t.center, t.half, t.stand_off_m, t.relaxed) == \
            (s.stand, s.yaw, s.side, s.center, s.half, s.stand_off_m, s.relaxed), n
        assert not t.extra_side
    assert list(on.surfaces)[:len(off.surfaces)] == list(off.surfaces)
    for n in set(on.surfaces) - set(off.surfaces):
        t = on.surfaces[n]
        own = [x.part for x in off.surfaces.values() if x.furniture_id == t.furniture_id]
        assert t.extra_side and own and EXTRA_LETTERS.index(t.part) >= len(own), (n, own)   # after its own letters
    assert on.user_surface == off.user_surface and on.skipped == off.skipped
    assert {k: v.near for k, v in on.landmarks.items() if k.startswith("stove")} == \
        {k: v.near for k, v in off.landmarks.items() if k.startswith("stove")}          # eval: stove_1 near counter_2a


def test_h40_dining_table_has_one_free_side_the_chairs_block_the_rest():
    """H40 (the owner's live bug): the dining table stands in the room, but its five chairs are pushed in: only the
    -x side has room for a regular stand (0.27-0.45 m out, 0.25 m clear); the +x side's only spot is 0.47 m out at a
    corner and the ends have none, so the table is not free-standing by the stand rule and keeps 1a (-x, 0.27 m) and
    1b (-x, relaxed 0.62 m) as they were. What reaches its far objects is the reachability's outline-wide search
    (tests/services/test_far_stance.py)."""
    from tests.fakes.fixtures import _lite
    from world.mapgen import _find_stand, build_static_map
    w = _lite("procthor-train-40")
    mp = _isaac_params()
    m = build_static_map(w.scene_data, w.grid, mp, scene_key="procthor-train-40")
    table = {n: (s.side, round(s.stand_off_m, 2), s.relaxed) for n, s in m.surfaces.items()
             if n.startswith("kitchen_dining_table")}
    assert table == {"kitchen_dining_table_1a": ("-x", 0.27, False), "kitchen_dining_table_1b": ("-x", 0.62, True)}
    s = m.surfaces["kitchen_dining_table_1a"]
    comp = w.grid.component(*w.scene_data.spawn[:2])
    free = {f: _find_stand(w.scene_data, w.grid, mp, s.center, s.half, False, 0, 2, s.room, [], comp,
                           only_sides=(f,), relaxed_ok=False, strict=True) is not None
            for f in ("-x", "+x", "-y")}
    assert free == {"-x": True, "+x": False, "-y": False}
    loose = _find_stand(w.scene_data, w.grid, mp, s.center, s.half, False, 0, 2, s.room, [], comp,
                        only_sides=("+x",), relaxed_ok=False)
    assert loose is not None and loose[3] > mp.stand_off_m[1]                          # 0.47 m: past the range


def test_h38_dining_table_gets_stands_across_from_its_own():
    from tests.fakes.fixtures import _lite
    from world.mapgen import build_static_map
    w = _lite("procthor-train-38")
    m = build_static_map(w.scene_data, w.grid, _isaac_params(), scene_key="procthor-train-38")
    got = {n: (s.side, s.extra_side) for n, s in m.surfaces.items() if n.startswith("kitchen_dining_table")}
    assert got == {"kitchen_dining_table_1a": ("-y", False), "kitchen_dining_table_1b": ("-y", False),
                   "kitchen_dining_table_1c": ("+y", True), "kitchen_dining_table_1d": ("+y", True)}
