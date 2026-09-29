"""The outline-wide reach-stance search (services/reachability.py find_far_stance) and the two-leg
navigate(reach_stance) it hands on (services/navigation.py), with the calibrated G1 arm (config/g1.yaml workspace)
on the R.7 stands (config mapgen), the way an Isaac world judges.

Synthetic free-standing table (tests/fakes/table_house.py): from its own stand an object at the near edge is
reachable, one near the far side needs_reposition to the other side, one at an end needs_reposition to that end,
one in the middle is beyond any reach (with why), and one whose only spots are walled off is an honest too_far with
no suggestion. Recorded H40 (the owner's live bug): from kitchen_dining_table_1a bottle_1 was too_far 1.09 m with
"no stand the robot knows reaches it"; the search finds the -y end of the table, the walk there is two legs (A*
go_to, then the approach) and the check from there says reachable. bowl_1 sits in front of a chair pushed under the
table: beyond reach as recorded, and an honest too_far a few cm nearer the edge (as it may settle live)."""

from __future__ import annotations

import math

import pytest

from api.summaries import summarize
from robot.lite_body import LiteBody
from services.reachability import G1Workspace
from tests.fakes import table_house
from tests.services.conftest import Stack, run

HOUSES = "tests/fakes/houses"


class ApproachBody(LiteBody):
    """The lite body plus a straight-line `approach` op (B.6's shape: a strafe on ground truth to the goal), so a
    reach stance 0.20 m from furniture (inside A*'s inflation) is reached exactly, as the SONIC body's op does."""

    def __init__(self, world, clock, walking=None):
        super().__init__(world, clock, walking)
        self.approaches: list[dict] = []
        self.go_tos: list[tuple[float, float]] = []

    def supports(self, what: str) -> bool:
        return True

    async def go_to(self, x, y, yaw=None, **kw):
        self.go_tos.append((round(x, 3), round(y, 3)))
        return await super().go_to(x, y, yaw, **{k: v for k, v in kw.items() if k != "fence"})

    async def approach(self, x, y, yaw=None, *, v=None, tol=None, timeout_s=None, fence=None):
        p0 = self.world.robot_pose()
        self.approaches.append({"x": x, "y": y, "yaw": yaw, "from": (round(p0.x, 3), round(p0.y, 3))})

        async def run_(state):
            p = self.world.robot_pose()
            n = 20
            for k in range(1, n + 1):
                self.world.set_robot_pose(p.x + (x - p.x) * k / n, p.y + (y - p.y) * k / n,
                                          p.yaw + (0.0 if yaw is None else ((yaw - p.yaw + math.pi) % (2 * math.pi)
                                                                            - math.pi) * k / n))
                await self.clock.sleep(self.tick)
            return "succeeded", {"pos_err": 0.0}
        return self._start("approach", run_, {"x": x, "y": y, "yaw": yaw})


def _g1_stack(house: str, house_dir=None) -> tuple[Stack, ApproachBody]:
    """A lite service stack built as an Isaac world is (the R.7 stands, the calibrated arm), with an approach body."""
    from world.mapgen import MapParams
    real = MapParams.for_source
    MapParams.for_source = lambda self, source: real(self, "isaac-gt")
    try:
        s = Stack(house=house, profile="lite", **({"house_dir": house_dir} if house_dir else {}))
    finally:
        MapParams.for_source = real
    d = {k: v for k, v in s.robot.stack_profile.g1["workspace"].items() if k != "lite_world"}
    s.robot.reach.ws = G1Workspace.from_dict(d)
    assert s.robot.reach.ws.far_search and s.robot.reach.ws.stance_via == "approach"
    body = ApproachBody(s.world, s.clock, s.robot.stack_profile.g1.get("walking"))
    for holder in (s.robot, s.robot.nav, s.robot.obs, s.robot.manip):
        holder.body = body
    s.body = body
    return s, body


async def _scan_at(s: Stack, kp: str) -> None:
    s.put_at(kp)
    s.robot.nav._last_at = kp
    s.robot.nav._anchor = None
    await s.run("observe", {"mode": "scan"})


def _check(s: Stack, oid: str):
    return s.robot.reach.check(s.world.object(oid).type, oid)


@pytest.fixture
def table(tmp_path):
    return _g1_stack("synthetic-table", table_house.write(tmp_path / "synthetic-table"))


# ------------------------------------------------------------------ the synthetic free-standing table
def test_the_synthetic_table_near_edge_far_side_end_and_middle(table):
    s, _ = table
    m = s.world.static_map()
    assert m.surfaces["kitchen_dining_table_1a"].side == "+y" and not m.surfaces["kitchen_dining_table_1a"].extra_side

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        # the near edge, in front of the stand: reachable from exactly here
        r = _check(s, "wine_bottle_1")
        assert r.reachable and r.reason is None, r
        # near the far side: no stance within approach_max_m; the other side of the table, by the shortest walk
        r = _check(s, "bottle_1")
        st = r.stance
        assert r.reason == "needs_reposition" and r.suggest_location == "reach_stance", r
        assert st["side"] == "-y" and st["reason"] == "other side of kitchen_dining_table_1", st
        assert st["distance_m"] > s.robot.reach.ws.approach_max_m and st["walk_m"] > st["distance_m"]
        assert {"x", "y", "yaw", "walk_m", "side", "reason", "via"} <= set(st)
        assert r.detail == ("reach stance on the other side of kitchen_dining_table_1 (-y side), "
                            f"{st['walk_m']:.1f} m walk")
        g = m.grid
        assert g.clearance(st["x"], st["y"]) >= s.robot.reach.ws.stance_clearance_m
        assert g.is_free(st["via"]["x"], st["via"]["y"])                       # the A* leg's goal: free space
        assert abs(math.degrees(st["yaw"]) - 90.0) <= s.robot.reach.ws.far_turn_max_deg   # facing the -y edge
        # at an end: the +x end
        r = _check(s, "mug_1")
        assert r.reason == "needs_reposition" and r.stance["side"] == "+x", r
        # the middle: 0.45 m past every edge, beyond any stance: why, and no suggestion
        r = _check(s, "bowl_1")
        assert r.reason == "beyond_reach" and r.suggest_location is None and r.stance is None, r
        assert "0.45 m past the nearest edge of kitchen_dining_table_1" in r.detail, r.detail
        line = summarize("check_reachability", "succeeded", dict(r.__dict__))
        assert "tell the user" in line and r.detail in line
    run(main())


def test_the_far_stance_is_walked_in_two_legs_and_the_check_from_there_says_reachable(table):
    s, body = table

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        res = await s.run("check_reachability", {"object_type": "bottle", "object_id": "bottle_1"})
        d = res.data
        assert d["reason"] == "needs_reposition", res.summary
        assert "reach stance on the other side of kitchen_dining_table_1" in res.summary
        assert "navigate(location='reach_stance')" in res.summary
        st = d["stance"]
        assert s.robot.timeout_s("navigate", {"location": "reach_stance", "stance": st}) > \
            s.robot.timeout_s("navigate", {"location": "reach_stance"})           # the walk's budget is added
        n = await s.run("navigate", {"location": "reach_stance", "anchor": "kitchen_dining_table_1a", "stance": st})
        assert n.status == "succeeded", n.summary
        assert n.data["legs"] == ["go_to", "approach"] and n.data["walk"]["state"] == "succeeded", n.data
        assert body.go_tos[-1] == (st["via"]["x"], st["via"]["y"])            # A* to the staging point ...
        assert body.approaches[-1]["x"] == st["x"] and body.approaches[-1]["y"] == st["y"]   # ... then the approach
        assert n.data["walked_m"] >= 0.8 * st["walk_m"] and n.data["final_err_m"] <= 0.05
        assert s.robot.nav.at() == ("kitchen_dining_table_1a", None)            # the reposition keeps its anchor
        r = await s.run("check_reachability", {"object_type": "bottle", "object_id": "bottle_1"})
        assert r.data["reachable"] and r.data["preferred_arm"] in ("left", "right", "either"), r.summary
        pick = await s.run("manipulate", {"action": "pick", "object_type": "bottle", "object_id": "bottle_1"})
        assert pick.status == "succeeded", pick.summary
    run(main())


def test_walled_off_spots_are_an_honest_too_far_with_no_suggestion(tmp_path):
    """The +x end walled off into a pocket the robot cannot walk into: mug_1 has spots within the arm's reach, none
    the robot can get to, so too_far says so and suggests nothing (not a stand that does not reach it either)."""
    walls = [(4.20, 1.90, 4.95, 1.97), (4.20, 2.93, 4.95, 3.00), (4.88, 1.90, 4.95, 3.00)]
    s, _ = _g1_stack("synthetic-table", table_house.write(tmp_path / "walled", extra_obstacles=walls))

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        r = _check(s, "mug_1")
        assert r.reason == "too_far" and r.suggest_location is None and r.stance is None, r
        assert r.detail.startswith("mug_1 is 0.08 m past the nearest edge of kitchen_dining_table_1, but no spot the "
                                   "robot can walk to around kitchen_dining_table_1 puts it inside the arm's reach"), \
            r.detail
        line = summarize("check_reachability", "succeeded", dict(r.__dict__))
        assert "no spot around that surface reaches it; tell the user" in line and "another stand" not in line
    run(main())


def test_far_search_off_keeps_m2a_too_far_with_a_stand(table):
    """workspace.far_search false (the lite world's INTERIM arm keeps it off): too_far + a stand, as before."""
    s, _ = table
    import dataclasses
    s.robot.reach.ws = dataclasses.replace(s.robot.reach.ws, far_search=False)

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        r = _check(s, "bottle_1")
        assert r.reason == "too_far" and r.detail is None and r.stance is None, r
    run(main())
    assert G1Workspace.from_dict({"far_search": False}).far_search is False
    lite = G1Workspace.from_dict(s.robot.stack_profile.g1["workspace"]).for_world(s.world)
    assert lite.far_search is False                                    # lite world: M2a's behaviour


# ------------------------------------------------------------------ the recorded H40 house (the live bug)
@pytest.fixture
def h40():
    import os
    if not os.path.exists(f"{HOUSES}/procthor-train-40/occupancy.npz"):
        pytest.skip("no recorded H40")
    return _g1_stack("procthor-train-40")


def test_h40_bottle_from_the_dining_table_stand_goes_round_to_the_minus_y_end(h40):
    s, body = h40

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        r = _check(s, "bottle_1")
        st = r.stance
        assert r.reason == "needs_reposition" and r.suggest_location == "reach_stance", r
        assert st["side"] == "-y" and st["reason"] == "other side of kitchen_dining_table_1", st
        assert 2.0 < st["walk_m"] < 3.5, st
        # the stance: between the -y end's chair and the table corner, the approach's 0.20 m from both
        assert s.world.static_map().grid.clearance(st["x"], st["y"]) >= 0.20
        n = await s.run("navigate", {"location": "reach_stance", "anchor": "kitchen_dining_table_1a", "stance": st})
        assert n.status == "succeeded" and n.data["legs"] == ["go_to", "approach"], n.summary
        r = _check(s, "bottle_1")
        assert r.reachable, r
        # from 1b (the relaxed stand further along the same side) the same far stance, a longer walk
        await _scan_at(s, "kitchen_dining_table_1b")
        r1b = _check(s, "bottle_1")
        assert r1b.reason == "needs_reposition" and r1b.stance["side"] == "-y"
        assert r1b.stance["walk_m"] > st["walk_m"]
    run(main())


def test_h40_bowl_behind_a_chair_is_beyond_reach_or_an_honest_too_far(h40):
    s, _ = h40

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        r = _check(s, "bowl_1")
        assert r.reason == "beyond_reach" and r.suggest_location is None, r
        assert "0.13 m past the nearest edge of kitchen_dining_table_1" in r.detail, r.detail
        # 6 cm nearer the edge (as it may settle live): a spot within the arm's horizontal reach exists, no stance
        # does (the chair pushed in under the table): too_far, why, and nothing suggested
        b = s.world.object("bowl_1")
        s.world.move_object("bowl_1", (b.pos[0] - 0.06, b.pos[1], b.pos[2]))
        r = _check(s, "bowl_1")
        assert r.reason == "too_far" and r.suggest_location is None and r.stance is None, r
        assert "past the nearest edge of kitchen_dining_table_1, but no spot the robot can walk to" in r.detail
    run(main())


def test_a_failed_walk_leg_maps_like_a_keypoint_navigate(table):
    """The far reposition's A* walk gets stuck: failed(blocked), the walk's outcome in data, no approach sent."""
    s, body = table

    async def main():
        await _scan_at(s, "kitchen_dining_table_1a")
        st = _check(s, "bottle_1").stance
        body.inject(stuck_after_s=1.0)
        n = await s.run("navigate", {"location": "reach_stance", "anchor": "kitchen_dining_table_1a", "stance": st})
        assert n.status == "failed" and n.data["reason"] == "blocked", n.data
        assert n.data["walk"]["state"] == "failed" and n.data["walk"]["body_reason"] == "stuck"
        assert not body.approaches
    run(main())
