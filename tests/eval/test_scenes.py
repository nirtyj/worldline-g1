"""eval/scenes.yaml against the recorded MolmoSpaces houses, through world/ (the way the runtime builds
the map) and the page's own layout code: every scenario's objects exist where the binding says, absent
types are absent, the user surfaces and walk distances match, flagged bindings say why, and the
K10 other_side binding resolves through agent/layout.py's LAYOUT lines."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from eval import scenes as sc

ROOT = Path(__file__).resolve().parents[2]
HOUSES = ROOT / "tests" / "fakes" / "houses"          # recorded house_info.json + occupancy.npz (world agent's fixtures)
BIND = sc.load()


def test_scenes_yaml_covers_the_17_scenarios():
    from eval.suite import SCENARIOS
    assert [n for n, _ in SCENARIOS] == list(BIND.scenarios)
    assert len(BIND.scenarios) == 17
    assert {BIND.scene(h) for h in ("H40", "H15", "K10")} == {"procthor-train-40", "procthor-train-15", "ithor-FloorPlan10"}
    for name, s in BIND.scenarios.items():
        assert s["house"] in BIND.houses, name
        assert s.get("status", "ok") in ("ok", "flagged", "substituted"), name
        if s.get("status") == "flagged":
            assert s.get("flags") and (s.get("note") or name in ("replace_task", "question_midtask")), name
        if s.get("status") == "substituted":
            assert s.get("substitute") and s.get("original") and s.get("note") and s.get("use") in ("substitute", "original")


def test_time_limits_scale_for_humanoid_speed():
    for name, s in BIND.scenarios.items():
        lite, how_lite = BIND.time_limit(name, "lite")
        sonic, how = BIND.time_limit(name, "sonic")
        full, _ = BIND.time_limit(name, "full")
        if s.get("legs"):
            assert how["estimate_s"] > 60, name
            assert sonic >= 2.0 * s["thor_limit_s"] and sonic >= 2 * how["estimate_s"] - 0.1, name
            assert full >= sonic >= lite >= s["thor_limit_s"], name
        else:
            assert lite == sonic == full == s["thor_limit_s"], f"{name}: talk-only scenarios keep THOR's limit"
    # the H40 fetch: 10.9 m at 0.40 m/s, 2 scans, a pick and a place
    est = BIND.humanoid_estimate("fetch_other_room", "sonic")
    assert 120 < est < 200
    assert BIND.time_limit("fetch_other_room", "sonic", scale=3.0)[0] == 600.0
    assert BIND.wait_scale("bringup") == 1.3 and BIND.wait_scale("lite", 0.5) == 0.5


def test_second_request_substitute_and_original():
    then, second = BIND.second_request("addition")
    assert then == "Also bring me a mug." and second == ["mug_1", "mug_2"]
    then, second = BIND.second_request("addition", original=True)
    assert then == "Also bring me a book." and second == ["book_1", "book_2"]


# ---------------------------------------------------------------------------------------------- vs the houses
def _house_info(scene: str) -> dict:
    p = HOUSES / scene / "house_info.json"
    if not p.exists():
        pytest.skip(f"no recorded house {scene} under {HOUSES}")
    return json.loads(p.read_text())


@lru_cache(maxsize=None)
def _world(scene: str):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    try:
        from world.lite_world import LiteWorld
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"world/ not importable: {e}")
    if not (HOUSES / scene / "occupancy.npz").exists():
        pytest.skip(f"no recorded occupancy for {scene}")
    return LiteWorld(HOUSES / scene)


@pytest.mark.parametrize("house", ["H40", "H15", "K10"])
def test_bound_objects_exist_where_the_binding_says(house):
    h = BIND.house(house)
    w = _world(h["scene"])
    truth = w.truth()
    m = w.lookup_keypoints()
    assert m["people"]["user"]["deliver_to_surface"] == h["user_surface"]
    for oid, spec in (h.get("objects") or {}).items():
        o = truth.objects.get(oid)
        assert o is not None, f"{oid} not in {h['scene']}"
        assert o["type"] == spec["type"], oid
        assert o["where"] == spec["surface"], f"{oid} is on {o['where']}, binding says {spec['surface']}"
    for name, spec in (h.get("landmarks") or {}).items():
        lm = truth.landmarks.get(name)
        assert lm is not None, name
        near = spec.get("near") or spec.get("surface")
        if spec.get("near"):
            assert lm["near"] == near, f"{name} near {lm['near']}"
    types = set(truth.things.values()) | {o["type"] for o in truth.objects.values()}
    for t in h.get("absent_types") or []:
        assert t not in types, f"{t} must be absent from {h['scene']}"
    for leg, d in (h.get("walk_m") or {}).items():
        a, _, b = leg.partition(">")
        got = w.static_map().edge_length(a, b)
        assert got is not None and abs(got - d) < 0.3, f"{leg}: {got} vs {d}"


@pytest.mark.parametrize("house", ["H40", "H15", "K10"])
def test_heights_and_dynamic_flags_match_the_house_data(house):
    h = BIND.house(house)
    info = _house_info(h["scene"])
    by_name = {o["name"]: o for o in info["objects"]}
    floor = info.get("floor_z", 0.0)
    for oid, spec in (h.get("objects") or {}).items():
        o = by_name[oid]
        lo, hi = o["aabb"][0][2] - floor, o["aabb"][1][2] - floor
        assert abs(lo - spec["height_m"][0]) < 0.02 and abs(hi - spec["height_m"][1]) < 0.02, (oid, lo, hi)
        assert spec["dynamic"] == (not o["is_static"]), oid


def test_the_book_flags_are_true_for_h40():
    """Why `addition` is substituted: both books are static prims below the G1 reach band."""
    info = _house_info("procthor-train-40")
    books = [o for o in info["objects"] if o["category"] == "Book"]
    assert len(books) == 2 and all(o["is_static"] for o in books)
    try:
        from api.types import PROFILES
        reach_min = PROFILES["sonic"].reach_h_min_m
    except Exception:  # noqa: BLE001
        reach_min = 0.55
    assert all(o["aabb"][0][2] < reach_min for o in books)
    mugs = [o for o in info["objects"] if o["name"] in ("mug_1", "mug_2")]
    assert len(mugs) == 2 and not any(o["is_static"] for o in mugs)
    assert all(o["aabb"][0][2] >= reach_min for o in mugs)


def test_other_side_resolves_on_the_live_k10_map_and_in_the_layout_lines():
    b = BIND.scenario("other_side")
    w = _world(BIND.scene("K10"))
    from ui.truth import layout_message, world_grid
    lay = layout_message("ithor-FloorPlan10", w.lookup_keypoints(), w, world_grid(w), truth=w.truth())
    lm = lay["landmarks"][b["landmark"]]
    start = w.truth().objects[b["object"]]["where"]
    assert start == b["start_surface"]
    assert sc.other_side(lay, (lm["x"], lm["z"]), start) == b["target_surface"]
    # the runtime's own LAYOUT lines (agent/layout.py) say the stove is between the two
    try:
        from agent import layout
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"agent/ not importable: {e}")
    lines = layout.build(w.lookup_keypoints(), {b["landmark"]: {"label": lm["label"], "near": lm["near"],
                                                               "pos": [lm["x"], lm["z"]]}})
    text = layout.render(lines, w.lookup_keypoints())
    assert any(g in text for g in sc.golden_lines(b["landmark"], b["start_surface"], b["target_surface"],
                                                  label=lm["label"])), text
    assert b["golden"] in sc.golden_lines(b["landmark"], b["start_surface"], b["target_surface"], label=lm["label"])


def test_fixture_check_reads_truth_only():
    lay = {"user_surface": "kitchen_counter_1a", "things": ["alarm_clock", "book", "dresser"]}
    truth = {"objects": {"alarm_clock_1": {"type": "alarm_clock", "where": "bedroom_dresser_1b"}}}
    ok, notes = sc.check_fixtures(BIND, "H40", lay, truth, ["alarm_clock_1"])
    assert ok and notes == []
    ok, notes = sc.check_fixtures(BIND, "H40", lay, truth, ["alarm_clock_1", "mug_1"])
    assert not ok and "mug_1 is not in the house" in notes
    ok, notes = sc.check_fixtures(BIND, "H40", {**lay, "things": ["banana"]}, truth, [])
    assert not ok and any("banana" in n for n in notes)
    moved = {"objects": {"alarm_clock_1": {"type": "alarm_clock", "where": "kitchen_counter_1a"}}}
    ok, notes = sc.check_fixtures(BIND, "H40", {**lay, "user_surface": "x"}, moved, ["alarm_clock_1"])
    assert ok and len(notes) == 2, "position and user-surface differences are noted, not failed"


def test_types_present_reads_every_vocabulary():
    t = sc.types_present({"objects": {"a": {"type": "AlarmClock"}, "b": {"type": "cell_phone"}}},
                         {"things": ["basket_ball", "remote_control"], "landmarks": {"tv_1": {"type": "television"}}})
    for w in ("alarm clock", "clock", "cell phone", "phone", "basketball", "remote", "tv"):
        assert w in t, w
    assert "banana" not in t


def test_no_yaml_key_is_a_boolean():
    """YAML 1.1 reads unquoted on/off/yes/no as booleans; a key like that silently disappears."""
    def walk(d, path="scenes.yaml"):
        if isinstance(d, dict):
            for k, v in d.items():
                assert isinstance(k, str), f"{path}: key {k!r} is not a string"
                walk(v, f"{path}.{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                walk(v, f"{path}[{i}]")
    walk(BIND.data)
