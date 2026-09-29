"""ManipulationService with the lite / kinematic_attach executors: pick, place, stance check, cancel, halt, place
failures, capability rejections, and STEPPING STONE labels."""

import pytest

from api.execution import Rejected
from api.results import validate_envelope
from tests.services.conftest import Stack, run


async def _ready_to_pick(s, kp="bedroom_bed_1b", otype="alarm_clock", oid="alarm_clock_1"):
    s.put_at(kp)
    s.robot.nav._last_at = kp
    await s.run("observe", {"mode": "scan"})
    r = await s.run("check_reachability", {"object_type": otype, "object_id": oid})
    if r.data["reason"] == "needs_reposition":
        await s.run("navigate", {"location": "reach_stance", "anchor": kp, "stance": r.data["stance"]})
        r = await s.run("check_reachability", {"object_type": otype, "object_id": oid})
    assert r.data["reachable"], r.summary
    return r


def test_pick_and_place_move_truth_and_are_labelled():
    s = Stack()

    async def main():
        reach = await _ready_to_pick(s)
        p = await s.run("manipulate", {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1"})
        assert validate_envelope(p) == []
        assert p.status == "succeeded" and p.data["holding"] is True
        arm = p.data["arm"]
        assert arm == (reach.data["preferred_arm"] if reach.data["preferred_arm"] in ("left", "right") else arm)
        assert p.data["executor"] == "lite" and p.data["skill"] == "lite.pick.v0"
        assert p.data["stepping_stone"] is True and "fallback" in p.summary
        assert s.world.object("alarm_clock_1").where == f"hand:{arm}"
        assert s.robot.gripper(arm)["closed"] is True
        n = await s.run("navigate", {"location": "user"})
        assert n.status == "succeeded"
        pl = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock", "target": "user"})
        assert pl.status == "succeeded" and pl.data["surface"] == s.world.map.user_surface
        assert s.world.object("alarm_clock_1").where == s.world.map.user_surface
        assert s.world.hands() == {"left": None, "right": None}
        assert pl.data["base_shift_m"] < 0.01                    # manipulate never walks
    run(main())


def test_pick_after_base_motion_fails_base_moving():
    s = Stack()

    async def main():
        await _ready_to_pick(s)
        p = s.world.robot_pose()
        s.world.set_robot_pose(p.x + 0.2, p.y, p.yaw)
        r = await s.run("manipulate", {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1"})
        assert r.status == "failed" and r.data["reason"] == "base_moving" and r.data["phase"] == "stance_check"
        assert s.world.hands() == {"left": None, "right": None}
    run(main())


def test_cancel_between_phases_and_halt_inside_a_phase():
    s = Stack()

    async def main():
        await _ready_to_pick(s)
        h = s.robot.start(s.ex("manipulate", {"action": "pick", "object_type": "alarm_clock",
                                              "object_id": "alarm_clock_1"}))
        await s.clock.sleep(0.5)                               # inside pregrasp
        h.cancel("correction")
        r = await h.result()
        assert r.status == "cancelled" and r.data["holding"] is False and r.data["phase"] == "grasp"
        # halt inside a phase ends failed(halted)
        h2 = s.robot.start(s.ex("manipulate", {"action": "pick", "object_type": "alarm_clock",
                                               "object_id": "alarm_clock_1"}))
        await s.clock.sleep(0.5)
        s.robot.halt()
        r2 = await h2.result()
        assert r2.status == "failed" and r2.data["reason"] == "halted"
    run(main())


def test_place_target_not_here_and_no_room():
    s = Stack()

    async def main():
        await _ready_to_pick(s)
        await s.run("manipulate", {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1"})
        r = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock",
                                       "target": "kitchen_counter_1a"})
        assert r.status == "failed" and r.data["reason"] == "target_not_here" and r.data["holding"] is True
        # no free spot within reach from here: make the reach window impossible
        from dataclasses import replace
        s.robot.reach.ws = replace(s.robot.reach.ws, reach_fwd_m=(5.0, 6.0))
        r2 = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock"})
        assert r2.status == "failed" and r2.data["reason"] == "no_room_in_reach"
        s.world.free_spot = lambda *a, **k: None
        r3 = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock"})
        assert r3.data["reason"] == "no_room_on_surface"
    run(main())


def test_place_with_nothing_in_hand():
    s = Stack()
    s.put_at("bedroom_bed_1b")

    async def main():
        r = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock"})
        assert r.status == "failed" and r.data["reason"] == "nothing_in_hand"
    run(main())


def test_capability_rejections_for_unhealthy_backends():
    s = Stack(overrides={"manipulation": {"executors": ["groot_sonic"]}})
    assert s.robot.registry().loaded_object_types() == ["bottle"]
    with pytest.raises(Rejected) as e:
        s.robot.start(s.ex("manipulate", {"action": "pick", "object_type": "bottle"}))
    assert e.value.stage == "capability" and e.value.message.startswith("policy unavailable:")
    assert "M4" in e.value.message


def test_kinematic_attach_needs_p1_attach_ops():
    """With a world that cannot attach (M1's P1), kinematic_attach is unhealthy: CAPABILITY rejection."""
    s = Stack(overrides={"manipulation": {"executors": ["kinematic_attach"]}})
    s.world.capabilities = lambda: {"attach": False, "detach": False}
    with pytest.raises(Rejected) as e:
        s.robot.start(s.ex("manipulate", {"action": "pick", "object_type": "alarm_clock"}))
    assert "P1 has no attach/detach op (M2b)" in e.value.message
    # the enum is frozen: the type stays loaded even though unhealthy
    assert "alarm_clock" in s.robot.registry().loaded_object_types()
    assert "banana" in s.robot.registry().loaded_object_types()


def test_registry_enum_is_the_fixed_vocabulary():
    s = Stack()
    types = s.robot.registry().loaded_object_types()
    assert "banana" in types and "alarm_clock" in types      # never derived from the scene (missing_object)
    assert "sofa" not in types
