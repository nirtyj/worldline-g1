"""check_reachability: every reason code, judged from the current pose, in THOR's order, with G1 geometry."""

import math

from api.results import validate_envelope
from services.reachability import G1Workspace, ReachabilityModel
from tests.services.conftest import Stack, run


def _check(s, otype, oid=None, **kw):
    return s.robot.reach.check(otype, oid, **kw)


async def _scan_at(s, kp):
    s.put_at(kp)
    s.robot.nav._last_at = kp
    await s.run("observe", {"mode": "scan"})


def test_reachable_after_reposition_and_arm_choice():
    s = Stack()

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        # apple_1 sits 0.77 m deep on the counter: beyond a G1's reach from any stance (THOR's 1.5 m reached it)
        far = _check(s, "apple")
        assert far.object_id == "apple_1" and far.visible and far.reason == "too_far"
        r = _check(s, "fork", "fork_1")               # 0.15 m from the front edge
        assert r.object_id == "fork_1" and r.visible
        if r.reason == "needs_reposition":
            assert r.suggest_location == "reach_stance" and r.stance and r.stance["distance_m"] <= 0.40
            rr = await s.run("navigate", {"location": "reach_stance", "anchor": "kitchen_counter_1c",
                                          "stance": r.stance})
            assert rr.status == "succeeded"
            r = _check(s, "fork", "fork_1")
        assert r.reachable and r.reason is None and r.preferred_arm in ("left", "right", "either")
        assert r.skill_id == "lite.pick.v0" and r.at == "kitchen_counter_1c"
    run(main())


def test_preferred_arm_by_lateral_offset():
    s = Stack()
    m = s.robot.reach
    assert m.preferred_arm(0.2) == "left"
    assert m.preferred_arm(-0.2) == "right"
    assert m.preferred_arm(0.05) == "either"

    class OnlyRight:
        arms = ("right",)
    assert m.preferred_arm(0.05, OnlyRight()) == "right"


def test_base_moving():
    s = Stack()

    async def main():
        h = s.robot.start(s.ex("navigate", {"location": "kitchen"}))
        await s.clock.sleep(2.0)
        r = await s.run("check_reachability", {"object_type": "apple"})
        assert r.status == "succeeded" and r.data["reachable"] is False and r.data["reason"] == "base_moving"
        assert validate_envelope(r) == []
        h.cancel("done")
        await h.result()
    run(main())


def test_not_found_for_an_unknown_id():
    s = Stack()
    r = _check(s, "apple", "apple_99")
    assert r.reason == "not_found" and not r.reachable


def test_not_seen_here_hides_positions():
    s = Stack()
    s.put_at("living_room_tv_stand_1a")
    r = _check(s, "alarm_clock", candidates=["alarm_clock_1"])   # belief names it; it is in the bedroom
    assert r.reason == "not_seen_here" and r.visible is False and r.object_id == "alarm_clock_1"
    assert r.distance_m is None and r.stance is None


def test_nothing_perceivable_is_not_found_and_leaks_nothing():
    """PLAN 6.4 step 2. Without candidates, an unseen alarm clock (it exists, in the bedroom) and a banana (there
    is none) get the same answer, with no object id: neither an unseen instance nor an absence leaks."""
    s = Stack()
    s.put_at("living_room_tv_stand_1a")
    r = _check(s, "alarm_clock")
    r2 = _check(s, "banana")
    for x in (r, r2):
        assert x.reason == "not_found" and x.visible is False and x.object_id is None
        assert x.distance_m is None and x.stance is None and x.suggest_location is None


def test_in_hand_and_hand_full():
    s = Stack()

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        s.world.attach("potato_1", "left")
        assert _check(s, "potato", "potato_1").reason == "in_hand"
        r = _check(s, "fork", "fork_1")
        assert r.reason in ("hand_full", "needs_reposition")
        if r.reason == "needs_reposition":
            await s.run("navigate", {"location": "reach_stance", "anchor": "kitchen_counter_1c", "stance": r.stance})
            assert _check(s, "fork", "fork_1").reason == "hand_full"
    run(main())


def test_inside_or_on_a_non_surface():
    s = Stack(house="procthor-train-15")
    w = s.world
    # stand 1 m in front of the chair with the bottle on it, facing it
    b = w.object("bottle_1")
    assert b.where == "chair"
    x, y, _ = b.pos
    w.set_robot_pose(x + 1.0, y, math.pi)
    r = _check(s, "bottle", "bottle_1")
    assert r.visible and r.reason == "inside_or_on_chair"


def test_too_high_and_too_low():
    s = Stack()

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        low = ReachabilityModel(s.world, G1Workspace(obj_z_min_m=1.2), registry=s.robot.skill_registry,
                                observation=s.robot.obs, nav=s.robot.nav)
        assert low.check("fork", "fork_1").reason == "too_low"
        high = ReachabilityModel(s.world, G1Workspace(obj_z_max_m=0.5), registry=s.robot.skill_registry,
                                 observation=s.robot.obs, nav=s.robot.nav)
        assert high.check("fork", "fork_1").reason == "too_high"
    run(main())


def test_too_far_suggests_another_stretch():
    s = Stack()

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        # the salt shaker is on stretch 1b, > approach_max_m away from 1c's stand
        seen = s.robot.obs.seen_here("kitchen_counter_1c")
        target = next(o for o in ("salt_shaker_1", "pan_1") if o in seen)
        r = _check(s, s.world.object(target).type, target)
        assert r.reason == "too_far" and r.suggest_location == "kitchen_counter_1b"
        res = await s.run("check_reachability", {"object_type": s.world.object(target).type, "object_id": target})
        assert "try kitchen_counter_1b" in res.summary
    run(main())


def test_no_skill_when_the_registry_lacks_the_type():
    from services.skills import build_registry
    s = Stack()

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        reg = build_registry(["groot_sonic"], s.world)       # only the (unofficial) bottle skill loads
        assert reg.loaded_object_types() == ["bottle"]
        m = ReachabilityModel(s.world, G1Workspace(reach_fwd_m=(0.0, 3.0), reach_lat_max_m=3.0, arm_reach_m=9.0),
                              registry=reg, observation=s.robot.obs, nav=s.robot.nav)
        assert m.check("fork", "fork_1").reason == "no_skill"
    run(main())


def test_needs_reposition_stance_is_free_and_close():
    s = Stack()

    async def main():
        await _scan_at(s, "bedroom_bed_1b")
        r = _check(s, "alarm_clock")
        assert r.reason == "needs_reposition"
        st = r.stance
        assert st["distance_m"] <= 0.40 + 1e-6
        g = s.world.map.grid
        assert g.is_free(st["x"], st["y"]) and g.clearance(st["x"], st["y"]) >= 0.25
        assert {"x", "y", "yaw", "dx", "dy", "dyaw"} <= set(st)
    run(main())
