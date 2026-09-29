"""The bindings E-1 and the G1 runs play (eval/scenes.py Bindings.binding): THOR's apple becomes H15's dish sponge
(scenes.yaml's scenario decision), and scenes.yaml's g1_alternative replaces addition's mug and other_side's spatula
on the G1 profiles by default, on lite when the run asks for it (eval/live_suite.py does). Checked on a scripted
fake page, and on the lite world itself: there too no stance reaches either mug or the spatula, while the
alternatives are reachable, which is why E-1 plays them."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from eval import scenes as sc
from eval import suite
from test_suite_offline import H40_LAYOUT, H40_OBJECTS, K10_LAYOUT, FakePage  # tests/eval is on sys.path (conftest)

BIND = sc.load()
ROOT = Path(__file__).resolve().parents[2]
HOUSES = ROOT / "tests" / "fakes" / "houses"
H15_LAYOUT = {"user_surface": "living_room_dining_table_1a", "things": ["apple", "dish_sponge", "microwave"],
              "surfaces": {}, "landmarks": {}}
H15_OBJECTS = {"apple_1": {"type": "apple", "where": "kitchen_counter_1b"},
               "dish_sponge_1": {"type": "dish_sponge", "where": "bathroom_sink_basin_1"}}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(suite, "LOAD_TIMEOUT_S", 5.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(suite.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.05)))


async def _play(page: FakePage, fn, profile: str, *, original: bool = False, g1_alternative: bool | None = None,
                scale: float = 0.01):
    run = suite.Run(page, profile, scale, original, g1_alternative)
    reader = asyncio.create_task(run.reader())
    try:
        passed, note = await asyncio.wait_for(fn(run), 30)
    except suite.FixtureError as e:
        passed, note = False, f"fixture: {e}"
    reader.cancel()
    return suite.score(fn.__name__, passed, note, 1.0, run)


def _mover(moves: dict[str, tuple[str, str]]):
    """On a sentence containing a key, move object -> surface (what a delivery does to world truth)."""
    async def on_say(page: FakePage, text: str) -> None:
        await page.run_tool("navigate", "lite", "keypoint")
        for words, (oid, where) in moves.items():
            if words in text.lower() and oid in page.objects:
                page.objects[oid]["where"] = where
        await page.frame(events=[{"t": page.t, "type": "speech_started", "text": "Here you go."}])
    return on_say


# ---------------------------------------------------------------------------------------------- Bindings.binding
def test_which_binding_each_profile_plays():
    assert BIND.binding("addition", "lite")[0] == "substitute"                  # the mug, as written
    assert BIND.binding("other_side", "lite")[0] == "as_thor"                   # the spatula
    for p in ("bringup", "sonic", "full"):
        which, b = BIND.binding("addition", p)
        assert which == "g1_alternative" and b["second"] == ["wine_bottle_1"] and "wine bottle" in b["then"]
        which, b = BIND.binding("other_side", p)
        assert (which, b["object"], b["start_surface"], b["target_surface"]) == \
            ("g1_alternative", "bowl_1", "counter_2c", "counter_2a")
    assert BIND.binding("addition", "lite", g1_alternative=True)[0] == "g1_alternative"
    assert BIND.binding("other_side", "sonic", g1_alternative=False)[0] == "as_thor"
    which, b = BIND.binding("addition", "sonic", original=True)                # THOR's book, never the alternative
    assert which == "original" and b["second"] == ["book_1", "book_2"]
    assert BIND.binding("other_side", "sonic", original=True)[0] == "as_thor"
    which, b = BIND.binding("fetch_search", "lite")
    assert which == "substitute" and b["target"] == "dish_sponge_1"
    assert BIND.binding("fetch_search", "lite", original=True)[1]["target"] == "apple_1"
    assert BIND.binding("fetch_other_room", "sonic") == ("as_thor", BIND.scenario("fetch_other_room"))


def test_time_limits_follow_the_played_binding():
    """The alternative's own legs set its limit (addition: two walks to the dining table instead of the dresser)."""
    for name in ("addition", "other_side"):
        written = BIND.time_limit(name, "sonic", binding=BIND.binding(name, "sonic", g1_alternative=False)[1])
        alt = BIND.time_limit(name, "sonic", binding=BIND.binding(name, "sonic")[1])
        assert written == BIND.time_limit(name, "sonic")
        assert alt[1]["estimate_s"] != written[1]["estimate_s"], name
    assert BIND.time_limit("fetch_search", "lite", binding=BIND.binding("fetch_search", "lite")[1]) == \
        BIND.time_limit("fetch_search", "lite")


# ---------------------------------------------------------------------------------------------- the suite plays them
@pytest.mark.parametrize("fn", [suite.fetch_search, suite.question_midtask])
def test_the_apple_scenarios_fetch_the_dish_sponge(fn):
    page = FakePage(H15_LAYOUT, H15_OBJECTS, _mover({"sponge": ("dish_sponge_1", "living_room_dining_table_1a"),
                                                     "apple": ("apple_1", "living_room_dining_table_1a")}))
    res = asyncio.run(_play(page, fn, "lite"))
    said = [m["text"] for m in page.sent if m["type"] == "say"]
    assert said[0] == "Bring me the dish sponge." and not any("apple" in t for t in said), said
    assert res["passed"] and res["binding"] == "substitute" and res["fixtures_ok"], res
    page = FakePage(H15_LAYOUT, H15_OBJECTS, _mover({"apple": ("apple_1", "living_room_dining_table_1a")}))
    res = asyncio.run(_play(page, fn, "lite", original=True))
    assert [m["text"] for m in page.sent if m["type"] == "say"][0] == "Bring me the apple."
    assert res["binding"] == "original"


def test_addition_asks_for_the_wine_bottle_when_the_alternative_plays():
    objs = {**H40_OBJECTS, "wine_bottle_1": {"type": "wine_bottle", "where": "kitchen_dining_table_1a"}}
    moves = {"alarm clock": ("alarm_clock_1", "kitchen_counter_1a"), "wine bottle": ("wine_bottle_1", "kitchen_counter_1a")}
    page = FakePage(H40_LAYOUT, objs, _mover(moves))
    res = asyncio.run(_play(page, suite.addition, "lite", g1_alternative=True))
    assert res["passed"] and res["binding"] == "g1_alternative", res
    assert "Also bring me the wine bottle." in [m.get("text") for m in page.sent]
    assert res["time_limit_s"] == BIND.time_limit("addition", "lite", 0.01, BIND.binding("addition", "lite", False, True)[1])[0]
    page = FakePage(H40_LAYOUT, H40_OBJECTS, _mover(moves))                  # the fixture needs the bottle
    res = asyncio.run(_play(page, suite.addition, "sonic"))
    assert not res["passed"] and res["note"].startswith("fixture:") and "wine_bottle_1" in res["note"]


def test_other_side_moves_the_bowl_and_still_checks_the_stoves_neighbours():
    async def on_say(page: FakePage, text: str) -> None:
        call = {"n": 1, "via": "model", "input": "LAYOUT\ncounter_2c · counter_2b · stove_1 · counter_2a\n"
                                                 "stove_1 (stove) is between counter_2b and counter_2a"}
        await page.frame(calls=[call])
        assert "bowl" in text
        await page.run_tool("manipulate", "sonic_arm_script", "pick", "sonic.script.pick.v0")
        page.objects["bowl_1"]["where"] = "counter_2a"
        await page.frame()
    objs = {"spatula_1": {"type": "spatula", "where": "counter_2b"}, "bowl_1": {"type": "bowl", "where": "counter_2c"}}
    page = FakePage(K10_LAYOUT, objs, on_say)
    res = asyncio.run(_play(page, suite.other_side, "sonic"))
    assert res["passed"] and res["binding"] == "g1_alternative", res
    assert "bowl_1 from counter_2c to counter_2a (target counter_2a)" in res["note"]
    assert "LAYOUT had 'stove_1 (stove) is between counter_2b and counter_2a': True" in res["note"]


def test_score_rows_and_the_live_report_name_the_binding():
    from eval import live_suite
    row = {"name": "addition", "passed": True, "fallback_pass": True, "note": "ok", "binding": "g1_alternative",
           "binding_flags": ["x"], "executors_used": {"nav": {"lite": 2}}, "shortcuts": ["lite"], "honesty": [],
           "seconds": 1.0, "time_limit_s": 2.0}
    rep = live_suite.combine({"results": [row]}, None, {"git": {"head": "h", "dirty": []}, "planner": "p",
                                                        "model": "m", "system1": "s", "bindings": "b"})
    assert rep["scenarios"][0]["binding"] == "g1_alternative"
    text = live_suite.report_text(rep)
    assert "[binding: g1_alternative]" in text and "bindings: b" in text


# ---------------------------------------------------------------------------------------------- the lite world
def _lite_stack(scene: str):
    pytest.importorskip("scipy")
    if not (HOUSES / scene / "occupancy.npz").exists():
        pytest.skip(f"no recorded house {scene}")
    from tests.services.conftest import Stack
    return Stack(house=scene, profile="lite")


async def _check_from(s, oid: str, kp: str):
    k = s.world.static_map().keypoints[kp]
    s.world.set_robot_pose(k.x, k.y, k.yaw)
    s.robot.nav._last_at = kp
    s.robot.nav._anchor = (kp, k.x, k.y)
    await s.run("observe", {"mode": "scan"})
    o = s.world.object(oid)
    return s.robot.reach.check(o.type, oid)


def test_on_the_lite_world_no_stance_reaches_the_mugs_or_the_spatula_but_the_alternatives_are_reachable():
    """The lite world (M2a's arm and stands, INTERIM): from every dresser and counter stand of H40 neither mug is
    reachable or one reach_stance away, and K10's spatula is beyond reach, so the as-written addition and other_side
    cannot pass on lite either; the wine bottle and the bowl are reachable from their stand or after one stance."""
    from tests.services.conftest import run
    s = _lite_stack("procthor-train-40")
    stands = [k for k in s.world.static_map().keypoints if k.startswith(("bedroom_dresser", "kitchen_counter"))]
    assert len(stands) >= 4

    async def h40():
        for oid in ("mug_1", "mug_2"):
            for kp in stands:
                r = await _check_from(s, oid, kp)
                assert not r.reachable and r.reason in ("too_far", "not_seen_here"), (oid, kp, r.reason)
        r = await _check_from(s, "wine_bottle_1", s.world.object("wine_bottle_1").where)
        assert r.reachable or r.reason == "needs_reposition", r.reason
    run(h40())
    s = _lite_stack("ithor-FloorPlan10")

    async def k10():
        r = await _check_from(s, "spatula_1", "counter_2b")
        assert r.reason == "beyond_reach", r.reason
        r = await _check_from(s, "bowl_1", "counter_2c")
        assert r.reachable or r.reason == "needs_reposition", r.reason
    run(k10())
