"""SonicBody over the real body.client.BodyClient against a scripted wl-body fake (M1 wire, contract §3.4), and
NavigationService's envelope mapping on top of it (sonic_walk executor)."""

import asyncio

import pytest

pytest.importorskip("zmq")

from services.navigation import NavConfig, NavigationService
from tests.fakes.fake_body_server import FakeBodyServer
from tests.fakes.fixtures import fresh_lite


@pytest.fixture
def fake_body():
    from tests.fakes.fake_p1_world import free_port_offset
    srv = FakeBodyServer(port_offset=free_port_offset(), motion_s=0.6).start()
    yield srv
    srv.stop()


def _body(srv):
    from robot.body_client import SonicBody
    return SonicBody(port_offset=srv.off, connect_wait_s=5.0)


def test_go_to_succeeds_and_cancel_stops(fake_body):
    async def main():
        b = _body(fake_body)
        try:
            op = await b.go_to(1.0, 2.0, 0.5, speed=0.45, timeout_s=30.0, final_pos_tol=0.25)
            res = await asyncio.wait_for(op.result(), 5.0)
            assert res["state"] == "succeeded" and res["data"]["path_len_m"] == 2.0
            req = [r for r in fake_body.requests if r["op"] == "go_to"][-1]
            assert req["args"] == {"x": 1.0, "y": 2.0, "yaw": 0.5, "speed": 0.45, "timeout_s": 30.0,
                                   "final_pos_tol": 0.25}
            fake_body.motion_s = 5.0
            op2 = await b.go_to(3.0, 3.0)
            await asyncio.sleep(0.2)
            op2.cancel("correction")
            res2 = await asyncio.wait_for(op2.result(), 5.0)
            assert res2["state"] == "canceled" and res2["data"]["reason"] == "stop"
            assert any(r["op"] == "stop" for r in fake_body.requests)
            assert b.health().ok and b.state()["mode"] in ("HOLD", "LOCOMOTION")
        finally:
            b.close()
    asyncio.run(main())


def test_halt_is_acked_within_budget_and_latches(fake_body):
    async def main():
        b = _body(fake_body)
        try:
            rec = b.halt(4)
            assert rec["stopped"] is True and rec["body_epoch"] == 4 and rec["source"] == "sonic-mux"
            assert "command{stop}" not in str(fake_body.requests)
            op = await b.go_to(1.0, 1.0)                     # latched: refused locally, nothing sent
            res = await asyncio.wait_for(op.result(), 2.0)
            assert res["state"] == "failed" and res["data"]["reason"] == "halted"
            b.resume(5)
            op2 = await b.go_to(1.0, 1.0)
            assert (await asyncio.wait_for(op2.result(), 5.0))["state"] == "succeeded"
            # a slow body misses the 30 ms budget: "Stopping now." (stopped=False)
            fake_body.reply_delay_s = 0.2
            assert b.halt()["stopped"] is False
        finally:
            b.close()
    asyncio.run(main())


def test_rejections_surface_as_failed(fake_body):
    async def main():
        b = _body(fake_body)
        try:
            fake_body.reject_reason = "not_standing"
            op = await b.go_to(1.0, 1.0)
            res = await asyncio.wait_for(op.result(), 5.0)
            assert res["state"] == "failed" and res["data"]["reason"] == "not_standing"
        finally:
            b.close()
    asyncio.run(main())


def test_navigation_over_sonic_body_maps_reasons(fake_body):
    """Envelopes with executor sonic_walk; body `stuck` -> blocked + blocked_edge; cancel -> cancelled."""
    from sim.clock import SimClock
    from api.execution import ExecutionManager

    async def main():
        world = fresh_lite("procthor-train-38")
        b = _body(fake_body)
        clock = SimClock(1.0)
        nav = NavigationService(world, b, clock, NavConfig())
        em = ExecutionManager(clock)
        try:
            r = await nav.start(em.create("navigate", {"location": "living_room"}, generation=1,
                                          control_epoch=0)).result()
            assert r.status == "succeeded" and r.data["executor"] == "sonic_walk" and r.data["at"] == "living_room"
            fake_body.fail_reason = "stuck"
            r2 = await nav.start(em.create("navigate", {"location": "kitchen"}, generation=1,
                                           control_epoch=0)).result()
            assert r2.status == "failed" and r2.data["reason"] == "blocked"
            assert r2.data["blocked_edge"] == ["living_room", "kitchen"] or r2.data["blocked_edge"][1] == "kitchen"
            fake_body.fail_reason = None
            fake_body.motion_s = 10.0
            h = nav.start(em.create("navigate", {"location": "kitchen"}, generation=1, control_epoch=0))
            await asyncio.sleep(0.3)
            h.cancel("correction")
            r3 = await asyncio.wait_for(h.result(), 5.0)
            assert r3.status == "cancelled" and r3.data["reason"] == "correction"
        finally:
            b.close()
    asyncio.run(main())
