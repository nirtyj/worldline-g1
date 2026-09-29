"""A one-armed skill in front of a two-armed fallback (the `full` profile: GR00T's Arena checkpoint is left-handed,
sonic_arm_script has both arms; M2b wave 2, eval-live).

Found in the offline full-profile E1 rehearsal (fake P1 + fake deploy + the real body + a fake PolicyServer + the real
page): check_reachability said "reachable with the left arm" for an object 0.27 m to the RIGHT of the pelvis (the
one-armed skill's reach stance was searched on both sides, and the sphere check used the nearer, right, shoulder), so
GR00T ran with the wrong hand and the groot_then_script fallback inherited that arm: ik_unreachable (0.16 m IK error)
on every pick. Pinned here, on H38's real map with the calibrated arm (config/g1.yaml workspace):

  - a one-armed skill's reach stance keeps the object on that arm's side, judged with that arm's sphere;
  - from a pose where the object sits on the other side and in the other arm's reach, the first skill with that arm
    answers (never a promise the one-armed skill cannot keep);
  - a pick without `arm` selects its skill for the arm the check named (PLAN 2.2 F1: pick with arm=preferred_arm).
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

from services.reachability import G1Workspace, ReachabilityModel
from tests.services.conftest import Stack

ROOT = Path(__file__).resolve().parents[2]
LEFT = SimpleNamespace(skill_id="groot.left", arms=("left",), stance={}, backend="groot")
BOTH = SimpleNamespace(skill_id="script.both", arms=("left", "right"), stance={}, backend="sonic_arm_script")


class Registry:
    """`full`'s order: the one-armed skill first, then the two-armed one."""

    def select(self, action, object_type, arm=None):
        return LEFT if arm in (None, "left") else BOTH


def _model(s: Stack, oid: str = "fork_1") -> ReachabilityModel:
    ws = dict(yaml.safe_load((ROOT / "config" / "g1.yaml").read_text())["workspace"])
    ws.pop("lite_world", None)                    # the calibrated G1 arm, as the Isaac profiles judge
    m = ReachabilityModel(s.world, G1Workspace.from_dict(ws), registry=Registry())
    m._visible_now = lambda: {oid}                # the object is in view (no scan needed for the geometry)
    return m


def _place(s: Stack, oid: str, fwd: float, left: float, yaw: float) -> None:
    """Stand the robot so the object's centre is `fwd` ahead and `left` to the left of the pelvis."""
    x, y, _ = s.world.objects()[oid].pos
    c, sn = math.cos(yaw), math.sin(yaw)
    s.world.set_robot_pose(x - (c * fwd - sn * left), y - (sn * fwd + c * left), yaw)


def test_the_one_armed_skills_stance_keeps_the_object_on_its_arm():
    """The rehearsal's geometry: H40's alarm clock from the Isaac dresser stand the page walked to (2.459, 0.925,
    -90 deg). Before: the stance put the clock 0.20 m to the right and promised the left arm at 0.49 m from its
    shoulder (reach 0.405 m)."""
    s = Stack(house="procthor-train-40")
    m = _model(s, "alarm_clock_1")
    s.world.set_robot_pose(2.459, 0.925, -math.pi / 2)
    r = m.check("alarm_clock", "alarm_clock_1")
    assert r.reason == "needs_reposition" and r.skill_id == "groot.left", (r.reason, r.skill_id)
    st = r.stance
    assert st["object_left"] >= -m.ws.either_deadband_m, st           # the left hand's side (the dead band counts)
    s.world.set_robot_pose(st["x"], st["y"], st["yaw"])
    r2 = m.check("alarm_clock", "alarm_clock_1")
    assert r2.reachable and r2.preferred_arm == "left" and r2.skill_id == "groot.left", (r2.reason, r2.preferred_arm)
    o = s.world.objects()["alarm_clock_1"]
    fwd, lat = m.in_body_frame(*o.pos[:2])
    assert m.shoulder_dist(fwd, lat, m.grasp_z(o), "left") <= m.ws.arm_reach_m     # the promise holds for that arm


def test_an_object_on_the_other_side_goes_to_the_skill_with_that_arm():
    s = Stack()
    m = _model(s)
    _place(s, "fork_1", 0.30, -0.22, math.pi / 2)        # 0.22 m to the right: the right arm reaches, the left not
    o = s.world.objects()["fork_1"]
    fwd, lat = m.in_body_frame(*o.pos[:2])
    gz = m.grasp_z(o)
    assert m.shoulder_dist(fwd, lat, gz, "right") <= m.ws.arm_reach_m < m.shoulder_dist(fwd, lat, gz, "left")
    r = m.check("fork", "fork_1")
    assert r.reachable and r.preferred_arm == "right" and r.skill_id == "script.both", \
        (r.reachable, r.reason, r.preferred_arm, r.skill_id)


def test_a_pick_without_an_arm_takes_the_skill_for_the_checked_arm():
    s = Stack()
    svc = s.robot.manip
    seen = []
    svc.capability = lambda action, otype, arm=None: (seen.append((action, otype, arm)), (None, "spy"))[1]
    reach = SimpleNamespace(reachable=True, preferred_arm="right", object_id="fork_1")
    svc._last_reach["fork"] = (reach, s.clock.now(), (0.0, 0.0, 0.0))

    first = []                                   # the skill selection each start makes (the spy's first call per start)

    async def main():
        for args in ({"action": "pick", "object_type": "fork"},                       # no arm: the checked arm
                     {"action": "pick", "object_type": "fork", "arm": "left"},        # the planner's arm wins
                     {"action": "place", "object_type": "fork"}):                     # place: the holding hand
            n = len(seen)
            h = svc.start(s.ex("manipulate", args))
            first.append(seen[n])
            await h.result()

    asyncio.run(main())
    assert first == [("pick", "fork", "right"), ("pick", "fork", "left"), ("place", "fork", None)], seen
