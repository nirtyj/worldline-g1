"""NavigationService on the lite stack: envelopes, aliases, timeouts, cancel / halt / blocked / fell / timeout
mapping, reach_stance repositions, and the body-reason mapping table."""

import math

import pytest

from api.results import validate_envelope
from services.navigation import map_body_reason
from tests.services.conftest import Stack, run


def test_navigate_succeeds_with_a_valid_envelope():
    s = Stack()

    async def main():
        r = await s.run("navigate", {"location": "kitchen_counter_1b"})
        assert validate_envelope(r) == []
        assert r.status == "succeeded" and r.data["at"] == "kitchen_counter_1b"
        assert r.data["executor"] == "lite" and r.data["kind"] == "keypoint"
        assert r.data["path_len_m"] > 3.0 and r.data["duration_s"] > 5.0
        assert r.observation_id and r.observation_id.startswith("obs-g")
        k = s.world.map.keypoints["kitchen_counter_1b"]
        p = s.world.robot_pose()
        assert math.hypot(p.x - k.x, p.y - k.y) < 0.25
        assert s.robot.nav.at() == ("kitchen_counter_1b", None)
        assert s.robot.base_state()["at"] == "kitchen_counter_1b"
        assert "arrived at kitchen_counter_1b" in r.summary
    run(main())


def test_user_and_room_aliases():
    s = Stack()
    nav = s.robot.nav
    assert nav.resolve("user") == s.world.map.user_surface
    assert nav.resolve("kitchen") == "kitchen"               # a room is its own keypoint
    assert "kitchen" in s.robot.lookup_keypoints()["keypoints"]

    async def main():
        r = await s.run("navigate", {"location": "user"})
        assert r.status == "succeeded" and r.data["at"] == s.world.map.user_surface
    run(main())


def test_timeout_formula_scales_with_walking_speed():
    s = Stack()
    nav = s.robot.nav
    p = s.world.robot_pose()
    k = s.world.map.keypoints["bedroom_bed_1b"]
    d = s.world.map.grid.distance((p.x, p.y), (k.x, k.y))
    v = s.robot.profile.walk_speed_mps
    assert nav.timeout_s(p, "bedroom_bed_1b") == pytest.approx(max(20, min(240, 1.8 * d / v + 12)))
    assert nav.timeout_s(p, "start") == 20.0                 # clamp at the minimum
    assert s.robot.timeout_s("navigate", {"location": "bedroom_bed_1b"}) == pytest.approx(
        nav.timeout_s(p, "bedroom_bed_1b"))
    assert s.robot.timeout_s("navigate", {"location": "bedroom_bed_1b", "timeout_s": 25}) == 25


def test_unknown_location_fails_without_moving():
    s = Stack()

    async def main():
        p0 = s.world.robot_pose()
        r = await s.run("navigate", {"location": "moon"})
        assert r.status == "failed" and r.data["reason"] == "unknown_location"
        assert s.world.robot_pose().x == p0.x
    run(main())


# House 38's walk from the start to bedroom_bed_1b passes straight through the living_room_dining_table_1a/1b stands
# (0.9-2.8 m along the path); from 3.2 m to 8 m it is > 0.7 m from every keypoint. A robot stopped there is
# `between`, not `at` a keypoint it happened to cross (at() is "a keypoint within 0.30 m").
BETWEEN_S = 9.0                          # ~4 m along at 0.45 m/s: well inside the keypoint-free stretch


def test_cancel_mid_walk_ends_between_and_stops():
    s = Stack()

    async def main():
        h = s.robot.start(s.ex("navigate", {"location": "bedroom_bed_1b"}))
        await s.clock.sleep(BETWEEN_S)
        assert s.robot.base_state()["moving"]
        h.cancel("correction")
        r = await h.result()
        assert r.status == "cancelled" and r.data["reason"] == "correction"
        assert r.data["at"] is None and r.data["between"] == ["start", "bedroom_bed_1b"]
        assert 0.5 < r.data["walked_m"] < r.data["path_len_m"]
        p = s.world.robot_pose()
        await s.clock.sleep(1.0)
        assert s.world.robot_pose().x == pytest.approx(p.x) and s.world.robot_pose().speed < 0.05
        assert s.robot.nav.at() == (None, ["start", "bedroom_bed_1b"])
        assert validate_envelope(r) == []
    run(main())


def test_halt_latches_and_resume_clears():
    s = Stack()

    async def main():
        h = s.robot.start(s.ex("navigate", {"location": "bedroom_bed_1b"}))
        await s.clock.sleep(3.0)
        receipt = s.robot.halt()
        assert receipt["stopped"] is True and receipt["mode"] == "HOLD" and receipt["latency_ms"] < 30
        r = await h.result()
        assert r.status == "failed" and r.data["reason"] == "halted"
        # latched: the next walk ends halted without moving
        p0 = s.world.robot_pose()
        r2 = await s.run("navigate", {"location": "kitchen"})
        assert r2.status == "failed" and r2.data["reason"] == "halted"
        assert s.world.robot_pose().x == pytest.approx(p0.x)
        s.robot.resume(5)
        r3 = await s.run("navigate", {"location": "living_room"})
        assert r3.status == "succeeded"
    run(main())


def test_stuck_walk_is_blocked_with_an_edge():
    s = Stack()
    s.body.inject(stuck_after_s=2.0)

    async def main():
        r = await s.run("navigate", {"location": "kitchen_counter_1a"})
        assert r.status == "failed" and r.data["reason"] == "blocked"
        assert r.data["blocked_edge"] == ["start", "kitchen_counter_1a"]
        assert r.data["body_reason"] == "stuck"
    run(main())


def test_fall_fails_the_walk_and_blocks_navigation():
    from api.execution import Rejected
    s = Stack()
    s.body.inject(fall_after_s=2.0)
    q = None

    async def main():
        nonlocal q
        q = s.robot.events()
        r = await s.run("navigate", {"location": "kitchen_counter_1a"})
        assert r.status == "failed" and r.data["reason"] == "fell"
        kinds = []
        while not q.empty():
            ev = q.get_nowait()
            if ev.get("type") == "safety_event":
                kinds.append(ev.get("kind"))
        assert "fell" in kinds
        assert not s.robot.capabilities()["navigation"].ok
        with pytest.raises(Rejected) as e:
            s.robot.start(s.ex("navigate", {"location": "kitchen"}))
        assert e.value.stage == "capability" and e.value.code == "nav_unhealthy"
        assert "navigation stack unavailable" in e.value.message
    run(main())


def test_service_timeout_is_timed_out():
    s = Stack()

    async def main():
        r = await s.run("navigate", {"location": "bedroom_bed_1b", "timeout_s": BETWEEN_S})
        assert r.status == "timed_out" and r.data["reason"] == "timeout"
        assert r.data["between"] == ["start", "bedroom_bed_1b"]
    run(main())


def test_reach_stance_reposition_keeps_the_anchor():
    s = Stack()

    async def main():
        k = s.put_at("bedroom_bed_1b")
        s.robot.nav._last_at = "bedroom_bed_1b"
        stance = {"x": k.x + 0.25, "y": k.y, "yaw": k.yaw}
        r = await s.run("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b", "stance": stance})
        assert r.status == "succeeded" and r.data["kind"] == "reposition"
        assert r.data["at"] == "bedroom_bed_1b" and r.data["final_err_m"] <= 0.10
        assert "interim" in r.data
        assert s.robot.nav.at() == ("bedroom_bed_1b", None)
        assert "repositioned" in r.summary
        # farther than approach_max_m (the outline-wide search's far stance): A* go_to next to it, then the final
        # leg, under one lease; the anchor stays
        far = {"x": k.x + 1.5, "y": k.y, "yaw": k.yaw}
        r2 = await s.run("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b", "stance": far})
        assert r2.status == "succeeded", r2.summary
        assert r2.data["legs"] == ["go_to", "go_to"] and r2.data["walk"]["state"] == "succeeded"
        assert r2.data["via"] and r2.data["final_err_m"] <= 0.10 and r2.data["walked_m"] >= 1.2
        assert s.robot.nav.at() == ("bedroom_bed_1b", None)
        # no free spot next to the stance (the middle of the bed): refused before anything moves
        bed = s.world.map.surfaces["bedroom_bed_1b"]
        p0 = s.world.robot_pose()
        inside = {"x": bed.center[0], "y": bed.center[1], "yaw": k.yaw}
        r2 = await s.run("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b", "stance": inside})
        assert r2.status == "failed" and r2.data["reason"] == "stance_not_reached", r2.data
        assert "no free spot next to it" in r2.data["detail"]
        assert (s.world.robot_pose().x, s.world.robot_pose().y) == (p0.x, p0.y)
        # no stance at all
        r3 = await s.run("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b"})
        assert r3.status == "failed" and r3.data["reason"] == "no_reach_stance"
    run(main())


def test_list_locations_nearest_first_and_filtered():
    s = Stack()

    async def main():
        r = await s.run("list_locations", {})
        locs = r.data["locations"]
        d = [l["distance_m"] for l in locs]
        assert d == sorted(d)
        names = {l["name"] for l in locs}
        assert {"start", "kitchen", "user"} <= names
        assert any(l["type"] == "person" for l in locs)
        r2 = await s.run("list_locations", {"query": "bed"})
        assert {l["name"] for l in r2.data["locations"]} == {"bedroom_bed_1a", "bedroom_bed_1b", "bedroom_bed_1c"}
        assert validate_envelope(r2) == []
    run(main())


@pytest.mark.parametrize("body,mapped", [("no_path", "no_path"), ("goal_in_obstacle", "no_path"),
                                         ("stuck", "blocked"), ("off_path", "blocked"), ("timeout", "timeout"),
                                         ("fallen", "fell"), ("fault:fallen", "fell"), ("final_error", "stuck"),
                                         ("deploy_stale", "nav_unhealthy"), ("nav2_unavailable", "nav_unhealthy"),
                                         ("pose_stale", "nav_unhealthy"), ("not_standing", "nav_unhealthy")])
def test_body_reason_mapping(body, mapped):
    from api.reasons import NAVIGATION
    assert map_body_reason(body) == mapped
    assert mapped in NAVIGATION
