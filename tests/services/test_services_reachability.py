"""check_reachability: every reason code, judged from the current pose, in THOR's order, with G1 geometry."""

import math

import pytest

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


def test_no_skill_when_the_registry_lacks_the_type(tmp_path):
    from services.skills import build_registry
    s = Stack()
    (tmp_path / "skills.yaml").write_text(
        "skills:\n  - {skill_id: only.bottle.v0, action: pick, object_types: [bottle], backend: lite}\n")

    async def main():
        await _scan_at(s, "kitchen_counter_1c")
        reg = build_registry(["lite"], s.world, path=tmp_path / "skills.yaml")   # a registry with bottles only
        assert reg.loaded_object_types() == ["bottle"]
        m = ReachabilityModel(s.world, G1Workspace(reach_fwd_m=(0.0, 3.0), reach_lat_max_m=3.0, arm_reach_m=9.0),
                              registry=reg, observation=s.robot.obs, nav=s.robot.nav)
        assert m.check("fork", "fork_1").reason == "no_skill"
    run(main())


def test_needs_reposition_stance_is_free_and_close():
    """The lite world's (INTERIM M2a) arm: stance_via go_to, so a stance is in A*'s free space."""
    s = Stack()

    async def main():
        await _scan_at(s, "bedroom_bed_1b")
        r = _check(s, "alarm_clock")
        assert r.reason == "needs_reposition"
        st = r.stance
        assert st["distance_m"] <= 0.40 + 1e-6
        g = s.world.map.grid
        assert g.is_free(st["x"], st["y"]) and g.clearance(st["x"], st["y"]) >= 0.25
        assert {"x", "y", "yaw", "dx", "dy", "dyaw", "object_fwd", "object_left"} <= set(st)
        assert s.robot.reach.lite_arm_note and "INTERIM" in s.robot.reach.lite_arm_note
    run(main())


def _g1_arm(s, **over):
    """The calibrated G1 arm of config/g1.yaml (what an Isaac world uses), on this lite stack."""
    ws = {k: v for k, v in s.robot.stack_profile.g1["workspace"].items() if k != "lite_world"}
    return ReachabilityModel(s.world, G1Workspace.from_dict({**ws, **over}), registry=s.robot.skill_registry,
                             observation=s.robot.obs, nav=s.robot.nav)


APPROACH = {"stance_via": "approach", "stance_clearance_m": 0.20}      # R.7's validated reach stance


def test_the_lite_world_gets_the_interim_arm_and_an_isaac_world_the_calibrated_one():
    s = Stack()
    ws = G1Workspace.from_dict(s.robot.stack_profile.g1["workspace"])
    assert ws.arm_reach_m < 0.45 and ws.stance_via == "go_to" and ws.lite_world
    lite = ws.for_world(s.world)
    assert lite.arm_reach_m == 0.65 and lite.stance_via == "go_to" and lite.obj_z_min_m == 0.55 and not lite.lite_world

    class IsaacLike:
        source = "isaac-gt"
    assert ws.for_world(IsaacLike()) is ws


def test_calibrated_arm_sphere_matches_the_ik_envelope():
    """shoulder_dist is the fitted sphere: body/arm_script.py's IK reaches points just inside it and misses points
    well outside (world/workspace_cal.py samples the whole envelope)."""
    pytest.importorskip("numpy")
    from world.workspace_cal import EnvelopeSpec, grasp_ok
    s = Stack()
    m = _g1_arm(s)
    spec = EnvelopeSpec()
    floor = s.world.map.floor_z
    pz = m.ws.shoulder_z_m - 0.299                     # the standing pelvis height the sphere was measured at
    for h, lat in ((0.95, -0.15), (1.10, -0.30), (1.05, -0.05)):
        z = floor + h
        # the sphere's forward reach at this height and lateral offset
        f = m.ws.shoulder_fwd_m + math.sqrt(m.ws.arm_reach_m ** 2 - (abs(lat) - m.ws.shoulder_lat_m) ** 2
                                            - (z - (floor + m.ws.shoulder_z_m)) ** 2)
        assert m.shoulder_dist(f, lat, z) == pytest.approx(m.ws.arm_reach_m, abs=1e-6)
        assert grasp_ok(spec, f - 0.01, lat, h - pz)[0], (h, lat, f)
        assert not grasp_ok(spec, f + 0.06, lat, h - pz)[0], (h, lat, f)


def test_calibrated_stance_is_an_approach_stance_close_to_the_furniture():
    """The calibrated arm's reach stance: the pelvis stance_clearance_m (0.20) from the furniture on a straight
    segment from here (the body's approach), within approach_max_m, the object inside the sphere with margin."""
    s = Stack(house="procthor-train-40")
    m = _g1_arm(s, **APPROACH)

    async def main():
        await _scan_at(s, "bedroom_dresser_1b")
        r = m.check("alarm_clock", "alarm_clock_1")
        assert r.reason == "needs_reposition", r
        st = r.stance
        g = s.world.map.grid
        assert st["distance_m"] <= m.ws.approach_max_m + 1e-6
        assert g.clearance(st["x"], st["y"]) >= m.ws.stance_clearance_m and m.stance_ok(st["x"], st["y"])
        o = s.world.object("alarm_clock_1")
        assert m.shoulder_dist(st["object_fwd"], st["object_left"], m.grasp_z(o)) <= m.ws.arm_reach_m - 0.02 + 1e-9
        s.world.set_robot_pose(st["x"], st["y"], st["yaw"])
        s.robot.nav._anchor = ("bedroom_dresser_1b", st["x"], st["y"])
        r2 = m.check("alarm_clock", "alarm_clock_1")
        assert r2.reachable and r2.preferred_arm in ("left", "right", "either")
    run(main())


def test_beyond_reach_says_how_far_the_arm_reaches():
    s = Stack(house="procthor-train-15")
    m = _g1_arm(s, **APPROACH)

    async def main():
        await _scan_at(s, "kitchen_counter_1b")
        r = m.check("apple", "apple_1")
        assert r.reason == "beyond_reach"
        assert "0.64 m from the nearest spot the robot can stand" in r.detail and "arm reaches 0.50 m" in r.detail
    run(main())
