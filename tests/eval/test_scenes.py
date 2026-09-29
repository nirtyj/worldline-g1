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


def test_fetch_targets_after_the_scenario_decisions():
    """R.7: H15's apple is beyond a G1's reach, so the two apple scenarios fetch the dish sponge; --original
    keeps THOR's apple."""
    for name in ("fetch_search", "question_midtask"):
        assert BIND.fetch_target(name) == ("dish_sponge_1", "dish sponge", "Bring me the dish sponge.")
        assert BIND.fetch_target(name, original=True) == ("apple_1", "apple", "Bring me the apple.")
        assert "apple_beyond_reach" in BIND.scenario(name)["flags"]
    assert BIND.fetch_target("fetch_other_room") == ("alarm_clock_1", "alarm clock", "Bring me the alarm clock.")


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
    # the map exactly as the G1 runtime builds it (robot/factory.py: config/g1.yaml mapgen, e.g. the `placeable`
    # user-surface rule, and config/stack.yaml's per-scene user_surface), since the suite scores what the runtime serves
    from robot.profile import load_profile
    from world.mapgen import MapParams
    prof = load_profile("lite")
    return LiteWorld(HOUSES / scene, map_params=MapParams.from_dict(prof.g1.get("mapgen")),
                     user_surface=prof.scene_config(scene).get("user_surface"))


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
    # the line the planner reads names the stove's two neighbours; the target is one of them, and the start surface
    # sits on the same layout line on the stove's other side (bowl_1 on counter_2c: 2c, 2b, the stove, 2a)
    assert b["golden"] in text, text
    assert b["target_surface"] in b["golden"]
    row = next(ln for ln in text.splitlines() if b["landmark"] in ln and "·" in ln)
    names = [x.strip().split(" ")[0] for x in row.split(":", 1)[1].split("·")]
    i_lm, i_start, i_tgt = names.index(b["landmark"]), names.index(b["start_surface"]), names.index(b["target_surface"])
    assert (i_start - i_lm) * (i_tgt - i_lm) < 0, row


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


# ---------------------------------------------------------------------------------------------- the G1 can do it
DELIVERIES = [("fetch_other_room", None, ["alarm_clock_1"]), ("fetch_search", None, ["dish_sponge_1"]),
              ("addition", "g1_alternative", ["alarm_clock_1", "wine_bottle_1"])]


TARGET_STANCE = {"stance_via": "approach", "stance_clearance_m": 0.20}    # R.7's validated reach stance


def _g1_stack(scene: str, target_stance: bool = True):
    """A lite service stack built as an Isaac world is: the R.7 stands (config mapgen, not mapgen.lite_world) and a
    reachability (and the place reach test that reads it) that judges with the calibrated G1 arm (config workspace,
    not workspace.lite_world); with target_stance, the approach stances 0.20 m from the furniture that R.7 validated
    live (config keeps A*'s 0.25 m until navigate(reach_stance) checks stances with ReachabilityModel.stance_ok)."""
    from services.reachability import G1Workspace
    from tests.services.conftest import Stack
    from world.mapgen import MapParams
    real = MapParams.for_source
    MapParams.for_source = lambda self, source: real(self, "isaac-gt")
    try:
        s = Stack(house=scene, profile="lite")
    finally:
        MapParams.for_source = real
    d = {k: v for k, v in s.robot.stack_profile.g1["workspace"].items() if k != "lite_world"}
    s.robot.reach.ws = G1Workspace.from_dict({**d, **(TARGET_STANCE if target_stance else {})})
    assert s.world.static_map().params.stand_off_m[0] < 0.30
    return s


def _at(s, kp: str, pose=None) -> None:
    k = s.world.static_map().keypoints[kp]
    x, y, yaw = pose if pose is not None else (k.x, k.y, k.yaw)
    s.world.set_robot_pose(x, y, yaw)
    s.robot.nav._last_at = kp
    s.robot.nav._anchor = (kp, x, y)


async def _pick_from_stand_or_one_stance(s, oid: str) -> str:
    o = s.world.object(oid)
    surf = o.where
    _at(s, surf)
    await s.run("observe", {"mode": "scan"})
    r = s.robot.reach.check(o.type, oid)
    how = "keypoint"
    if r.reason == "needs_reposition":
        st = r.stance
        assert st["distance_m"] <= s.robot.reach.ws.approach_max_m + 1e-6
        _at(s, surf, (st["x"], st["y"], st["yaw"]))
        r = s.robot.reach.check(o.type, oid)
        how = "one_approach"
    assert r.reachable, (oid, r.reason, getattr(r, "detail", None))
    pick = await s.run("manipulate", {"action": "pick", "object_type": o.type, "object_id": oid})
    assert pick.status == "succeeded", (oid, pick.summary)
    return how


@pytest.mark.parametrize("name, part, oids", DELIVERIES)
def test_every_bound_delivery_is_reachable_and_placeable_with_the_calibrated_arm(name, part, oids):
    """R.7: with the calibrated G1 arm (config/g1.yaml workspace) each object is reachable from its surface's stand
    or after one reach_stance, the pick succeeds, and it can be put down on the user surface from that surface's stand
    (place never repositions). Through the runtime's own services on the lite stack; the stance is reached by setting
    the pose (navigate(reach_stance) itself is services/navigation.py's)."""
    pytest.importorskip("scipy")
    b = BIND.scenario(name)
    if part:
        assert b[part]["second"] == oids[1:]
    scene = BIND.scene(b["house"])
    if not (HOUSES / scene / "occupancy.npz").exists():
        pytest.skip(f"no recorded house {scene}")
    from tests.services.conftest import run
    s = _g1_stack(scene)
    user = s.robot.lookup_keypoints()["people"]["user"]["deliver_to_surface"]
    assert user == BIND.house(b["house"])["user_surface"]

    async def main():
        out = []
        for oid in oids:
            how = await _pick_from_stand_or_one_stance(s, oid)
            _at(s, user)
            o = s.world.object(oid)
            place = await s.run("manipulate", {"action": "place", "object_type": o.type, "target": "user"})
            assert place.status == "succeeded", (oid, place.summary)
            assert s.world.object(oid).where == user
            out.append((oid, how))
        return out

    got = run(main())
    assert [o for o, _ in got] == oids


def test_the_g1_alternative_for_other_side_goes_round_the_stove():
    """K10 other_side with the calibrated arm: the spatula is beyond reach; the proposed bowl_1 goes from counter_2c
    (one reach_stance) down onto counter_2a from its stand."""
    pytest.importorskip("scipy")
    b = BIND.scenario("other_side")
    alt = b["g1_alternative"]
    from tests.services.conftest import run
    s = _g1_stack(BIND.scene("K10"))

    async def main():
        spatula = s.world.object(b["object"])
        _at(s, spatula.where)
        await s.run("observe", {"mode": "scan"})
        assert s.robot.reach.check(spatula.type, spatula.id).reason == "beyond_reach"
        o = s.world.object(alt["object"])
        assert o.where == alt["start_surface"]
        await _pick_from_stand_or_one_stance(s, o.id)
        _at(s, alt["target_surface"])
        pl = await s.run("manipulate", {"action": "place", "object_type": o.type, "target": alt["target_surface"]})
        assert pl.status == "succeeded", pl.summary
        return s.world.object(o.id).where

    assert run(main()) == alt["target_surface"]


def test_the_apple_is_beyond_a_g1s_reach_and_the_mugs_beyond_one_reposition():
    """The evidence behind the scenario decisions, with the calibrated arm."""
    pytest.importorskip("scipy")
    from tests.services.conftest import run
    s = _g1_stack(BIND.scene("H15"))

    async def h15():
        apple = s.world.object("apple_1")
        _at(s, apple.where)
        await s.run("observe", {"mode": "scan"})
        r = s.robot.reach.check("apple", "apple_1")
        assert r.reason == "beyond_reach" and "0.64 m from the nearest spot" in r.detail, r.detail
    run(h15())
    s = _g1_stack(BIND.scene("H40"))

    async def h40():
        for oid in ("mug_1", "mug_2"):
            o = s.world.object(oid)
            _at(s, o.where)
            await s.run("observe", {"mode": "scan"})
            # mug_1: no stance within one reposition; mug_2 is not even in view from its stand's scan (1.45 m away)
            assert s.robot.reach.check("mug", oid).reason in ("too_far", "beyond_reach", "not_seen_here"), oid
    run(h40())


@pytest.mark.parametrize("scene, oid", [("procthor-train-40", "alarm_clock_1"), ("procthor-train-40", "wine_bottle_1"),
                                        ("ithor-FloorPlan10", "bowl_1")])
def test_with_todays_config_the_stances_pass_navigations_own_check(scene, oid):
    """The INTERIM config (stance_via go_to, 0.25 m) gives stances that services/navigation.py's reach_stance check
    accepts today (A*'s inflated free space and a free straight segment, within approach_max_m), so the live F1 is
    not blocked while the 0.20 m approach stances wait for navigation."""
    pytest.importorskip("scipy")
    import math
    from tests.services.conftest import run
    s = _g1_stack(scene, target_stance=False)
    ws = s.robot.reach.ws
    assert ws.stance_via == "go_to" and ws.stance_clearance_m == 0.25

    async def main():
        o = s.world.object(oid)
        _at(s, o.where)
        await s.run("observe", {"mode": "scan"})
        r = s.robot.reach.check(o.type, oid)
        assert r.reason == "needs_reposition", r.reason
        st, p, g = r.stance, s.world.robot_pose(), s.world.static_map().grid
        assert g.is_free(st["x"], st["y"]) and g.segment_free((p.x, p.y), (st["x"], st["y"]))
        assert math.hypot(st["x"] - p.x, st["y"] - p.y) <= ws.approach_max_m + 0.05
    run(main())
