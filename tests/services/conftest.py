import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fakes.fixtures import fresh_lite  # noqa: E402,F401

SPEED = 40.0          # sim seconds per wall second for lite runs


class Stack:
    """A lite stack for service tests: world, robot (G1Robot), body, clock, log, executions."""

    def __init__(self, house="procthor-train-38", profile="lite", speed=SPEED, overrides=None, **kw):
        from api.execution import ExecutionManager
        from robot.factory import build
        from robot.profile import load_profile
        from sim.clock import SimClock
        from sim.log import EventLog
        self.clock = SimClock(speed)
        self.log = EventLog(self.clock)
        prof = load_profile(profile, overrides=overrides)
        self.world, self.robot, self.frames = build(prof, house, self.clock, self.log, **kw)
        self.body = self.robot.body
        self.em = ExecutionManager(self.clock)
        self.gen = 1
        self.epoch = 0

    def ex(self, tool, args, **kw):
        return self.em.create(tool, args, generation=self.gen, control_epoch=self.epoch, **kw)

    async def run(self, tool, args, **kw):
        return await self.robot.start(self.ex(tool, args, **kw)).result()

    def put_at(self, keypoint, dyaw=0.0):
        k = self.world.map.keypoints[keypoint]
        self.world.set_robot_pose(k.x, k.y, k.yaw + dyaw)
        return k


@pytest.fixture
def stack():
    return Stack()


def run(coro):
    return asyncio.run(coro)
