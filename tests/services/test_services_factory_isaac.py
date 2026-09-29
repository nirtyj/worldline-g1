"""robot.factory.build("sonic", ...) over the wire, with no Isaac and no SONIC: a fake P1 (REP + gt.pose, serving a
recorded house) and a fake wl-body (ROUTER/PUB) on one port offset. Checks the Isaac-profile wiring end to end:
IsaacGTWorldModel + SonicBody + services, capability rejections for attach-less P1, and a navigate through the body
wire with executor sonic_walk."""

import asyncio
import time

import pytest

pytest.importorskip("zmq")

from api.execution import ExecutionManager, Rejected
from api.results import validate_envelope
from tests.fakes.fake_body_server import FakeBodyServer
from tests.fakes.fake_p1_world import FakeP1World

OFF = 460


@pytest.fixture
def fake_stack(monkeypatch):
    monkeypatch.setenv("WL_PORT_OFFSET", str(OFF))
    p1 = FakeP1World("procthor-train-38", port_offset=OFF).start()
    body = FakeBodyServer(port_offset=OFF, motion_s=0.5).start()
    yield p1, body
    body.stop()
    p1.stop()


def test_sonic_profile_builds_over_the_wire(fake_stack):
    from robot.factory import build
    from sim.clock import SimClock
    p1, fbody = fake_stack

    async def main():
        clock = SimClock(1.0)
        world, robot, frames = build("sonic", "procthor-train-38", clock, None, frames=None)
        try:
            assert world.source == "isaac-gt" and robot.body.name == "sonic_walk"
            m = robot.lookup_keypoints()
            assert m["profile"] == "sonic" and m["executors"]["navigate"] == "sonic_walk"
            assert m["camera"]["name"] == "ego_d435"
            assert "kinematic_attach" in robot.profile.stepping_stones
            caps = robot.capabilities()
            assert caps["body"].ok and caps["navigation"].ok
            assert not caps["manipulation"].ok                 # M1 P1: no attach; arm_script is M3
            em = ExecutionManager(clock)
            # manipulate: rejected at CAPABILITY with the doc's wording, the enum still frozen
            with pytest.raises(Rejected) as e:
                robot.start(em.create("manipulate", {"action": "pick", "object_type": "alarm_clock"},
                                      generation=1, control_epoch=0))
            assert e.value.message.startswith("policy unavailable:")
            assert "alarm_clock" in robot.registry().loaded_object_types()
            # navigate goes through the body wire
            r = await asyncio.wait_for(robot.start(em.create("navigate", {"location": "living_room"}, generation=1,
                                                             control_epoch=0)).result(), 10.0)
            assert r.status == "succeeded" and r.data["executor"] == "sonic_walk"
            assert validate_envelope(r) == []
            req = [q for q in fbody.requests if q["op"] == "go_to"][-1]
            k = world.map.keypoints["living_room"]
            assert req["args"]["x"] == pytest.approx(k.x) and req["args"]["y"] == pytest.approx(k.y)
            # halt goes to the body as a stop within the budget
            rec = robot.halt()
            assert rec["source"] == "sonic-mux" and rec["stopped"] is True
            assert any(q["op"] == "stop" for q in fbody.requests)
            # perception from gt.pose + the scene: THOR-shaped
            p = robot.perception()
            assert p["source"] == "isaac-ground-truth"
            t = robot.telemetry()
            assert t["health"]["source"] == "isaac-gt"
        finally:
            await robot.shutdown()
            world.close()
    asyncio.run(main())
