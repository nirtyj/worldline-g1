"""The live `sonic` stack against the api/ Protocols and the M2b world surface (`-m box`, WL_BOX=1; docs/M2.md E0).

Read-only: nothing here moves the robot (test_robot_contract[sonic] does). Skipped cleanly off the box."""

from __future__ import annotations

import pytest

import api.services as S
from tests.contract.conftest import clock_for, make_backend
from tests.contract.test_protocols import problems

pytestmark = pytest.mark.box


def test_live_stack_matches_the_protocols():
    robot = make_backend("sonic", clock_for("sonic"))
    world = robot.world
    assert problems(robot, S.RobotBridge) == []
    assert problems(robot.body, S.BodyPort) == [] and robot.body.name == "sonic_walk"
    assert problems(world, S.WorldModel) == [] and problems(world, S.SimControl) == []
    for svc, proto in ((robot.nav, S.NavigationService), (robot.manip, S.ManipulationService),
                       (robot.obs, S.ObservationService), (robot.speech, S.SpeechService)):
        assert problems(svc, proto) == [], proto.__name__
    for ex in robot.executors.values():
        assert problems(ex, S.ManipExecutor) == [], ex.name


def test_live_world_reports_its_p1_and_its_health():
    robot = make_backend("sonic", clock_for("sonic"))
    world = robot.world
    caps = world.capabilities()
    assert caps["band"] is True and caps["reset_robot"] is True
    h = world.sim_health()
    assert h.state in ("ok", "degraded", "unsafe") and h.source.startswith("p1:")
    assert world.pose_age_s() < 0.5 and robot.capabilities()["body"].state in ("ok", "unsafe", "degraded")
    assert set(robot.capabilities()) == set(S.CAPABILITIES)
    # the camera the world models is the camera P1 renders (head on an M2b P1, else the d435, labelled)
    assert world.cam.name == ("head_sim" if world.has_camera("head") else "ego_d435")
    m = robot.lookup_keypoints()
    assert m["camera"]["caption"].startswith("camera: ")
    if caps["object_poses"]:
        assert all(o.pose_source == "sim" for o in world.objects().values())
