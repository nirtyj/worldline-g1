"""SonicBody on the M2b body surface (R.2), against the REAL wl-body service over the body's own fakes
(tools/fake_p1.py + tools/fake_deploy.py, body/tests/test_integration_fakes.Stack): the runtime session and the epoch
map (one epoch space), fences on every op, the B.1 halt lane receipt and latch, a resume above the halt, stale fences
keyed on data.why, leases, B.3 events pushed to the runtime, and the approach / scan / arm_script ops.

Ground truth for motion comes from the fake P1 (stack.p1), never from the body's word."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

pytest.importorskip("zmq")

from api.execution import ExecutionManager                                     # noqa: E402
from body.tests.conftest import _free_block                                     # noqa: E402
from body.tests.test_integration_fakes import Stack                             # noqa: E402
from services.common import refusal                                             # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = Stack(_free_block(), tmp_path_factory.mktemp("sonic_body_m2b"))
    s.stand()
    assert s.bc.turn_to(0.0).ok
    yield s
    s.close()


def _off(stack) -> int:
    return stack.bc.ports["body_ctl"] - 5610


def _body(stack, **kw):
    from robot.body_client import SonicBody
    return SonicBody(port_offset=_off(stack), connect_wait_s=5.0, **kw)


def _ex(em, tool="navigate", generation=1, control_epoch=0):
    return em.create(tool, {"location": "x"}, generation=generation, control_epoch=control_epoch)


async def _until(pred, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def test_session_fences_halt_lane_resume_and_stale(stack):
    p1 = stack.p1

    async def main():
        b = _body(stack)
        em = ExecutionManager()
        seen: list[tuple[str, dict]] = []
        b.attach_events(lambda t, m: seen.append((t, m)))
        try:
            assert b.surface == "m2b" and b.session and b.hello_info["ok"]
            assert b.supports("approach") and b.supports("scan") and b.supports("arm_script")
            ex = _ex(em, generation=1, control_epoch=0)
            f = b.fence(ex)
            assert f == {"execution_id": ex.execution_id, "generation": b.gen_base + 1,
                         "control_epoch": b.epoch_base, "session": b.session}
            lease = await b.acquire(ex, "LOCOMOTION")
            assert lease["ok"] and lease["lease"]["owner"] == ex.execution_id and lease["lease"]["mode"] == "LOCOMOTION"
            op = await b.go_to(p1.x + 3.0, p1.y, 0.0, timeout_s=60.0, fence=f)
            assert await _until(lambda: math.hypot(p1.vx, p1.vy) > 0.2, 5.0), "the fake robot never walked"
            # B.1: the halt lane, inside the 30 ms budget, fencing the execution's own control_epoch
            t0 = time.perf_counter()
            rec = b.halt(0)
            dt = time.perf_counter() - t0
            assert rec["stopped"] is True and rec["source"] == "halt-lane" and rec["body_kind"] == "new", rec
            assert dt < 0.030 and rec["rtt_ms"] < 30 and rec["body_epoch"] == b.epoch_base
            res = await asyncio.wait_for(op.result(), 3.0)
            assert res["state"] == "canceled" and res["data"]["reason"] == "halt"
            await asyncio.sleep(1.5)
            assert math.hypot(p1.vx, p1.vy) < 0.05                    # at rest on the fake P1's ground truth
            # the runtime latch refuses locally; the lease died with the halt (its epoch)
            op_l = await b.turn_to(1.0, fence=f)
            assert (await op_l.result())["data"]["reason"] == "halted"
            assert await _until(lambda: b.lease is None, 1.0)
            # resume: the next executions carry epoch 1, above the halt's 0
            rep = b.resume(1)
            assert rep["ok"] and rep["body_epoch"] == b.epoch_base
            ex2 = _ex(em, generation=1, control_epoch=1)
            assert (await b.acquire(ex2, "LOCOMOTION"))["ok"]
            op2 = await b.turn_to(0.5, fence=b.fence(ex2))
            assert (await asyncio.wait_for(op2.result(), 20.0))["state"] == "succeeded"
            # the halted epoch's command is a stale fence at the body (keyed on data.why), not a failure
            await b.release(ex2.execution_id)
            op3 = await b.turn_to(0.0, fence=b.fence(ex))
            r3 = await asyncio.wait_for(op3.result(), 3.0)
            assert r3["state"] == "failed" and refusal(r3["data"]["reason"], r3["data"]) == "stale_result", r3
            assert r3["data"].get("why") == "control_epoch"
            # a second halt at the newer epoch is a new latch, not a late re-send of the first
            rec2 = b.halt(1)
            assert rec2["stopped"] and rec2["body_kind"] == "new"
            b.resume(2)
            await asyncio.sleep(0.3)
            topics = {t for t, _ in seen}
            assert {"body.halted", "body.resumed", "body.stale_command", "body.lease", "body.mode"} <= topics, topics
            assert any(t == "body.stale_command" and m.get("why") == "control_epoch" for t, m in seen)
        finally:
            b.close()
    asyncio.run(main())


def test_leases_foreign_owner_leftover_and_a_restarted_runtime(stack):
    async def main():
        em = ExecutionManager()
        b = _body(stack)
        try:
            a1 = _ex(em, control_epoch=0)
            assert (await b.acquire(a1, "ARM_SCRIPT"))["ok"]
            # a foreign owner (another tool) holds the lease: body_busy, keyed as such
            other = stack.bc.acquire("someone-else", b.gen_base + 1, b.epoch_base, mode="ANY")
            assert not other["ok"] and other["error"] == "body_busy"
            op = await b.turn_to(0.2, fence={"execution_id": "someone-else", "generation": b.gen_base + 1,
                                             "control_epoch": b.epoch_base})
            r = await op.result()
            assert r["state"] == "failed" and refusal(r["data"]["reason"], r["data"]) == "body_busy"
            # our own leftover lease (a lost release) is released by the next acquire
            a2 = _ex(em, control_epoch=0)
            rep = await b.acquire(a2, "LOCOMOTION")
            assert rep["ok"] and rep["lease"]["owner"] == a2.execution_id
            # a runtime that dies while latched: the next one resumes at startup and starts above every epoch
            b.halt(0)
        finally:
            b.close()
        b2 = _body(stack)
        try:
            info = b2.hello_info
            assert info["ok"] and info["latched"] is True and info["startup_resume"]["ok"] is True
            assert b2.epoch_base > b.epoch_base
            ex = _ex(ExecutionManager(), control_epoch=0)
            op = await b2.turn_to(0.0, fence=b2.fence(ex))
            assert (await asyncio.wait_for(op.result(), 20.0))["state"] == "succeeded"
        finally:
            b2.close()
    asyncio.run(main())


def test_approach_scan_and_arm_script_ops(stack):
    p1 = stack.p1

    async def main():
        b = _body(stack)
        em = ExecutionManager()
        try:
            ex = _ex(em, control_epoch=0)
            f = b.fence(ex)
            assert (await b.acquire(ex, "LOCOMOTION"))["ok"]
            x0, y0, yaw0 = p1.x, p1.y, p1.yaw
            gx, gy = x0 + 0.2 * math.cos(yaw0), y0 + 0.2 * math.sin(yaw0)
            op = await b.approach(gx, gy, yaw0, tol=(0.05, 5.0), timeout_s=40.0, fence=f)
            r = await asyncio.wait_for(op.result(), 45.0)
            assert r["state"] == "succeeded", r
            assert math.hypot(p1.x - gx, p1.y - gy) <= 0.06
            # B.5 waist scan: one scan.hold progress event per yaw, then succeeded (fenced, same lease owner)
            holds = []
            op = await b.scan([-35.0, 0.0, 35.0], move_s=0.5, hold_s=0.5, fence=f)
            op.on_progress(lambda ev: holds.append(ev["data"]) if ev["data"].get("kind") == "scan.hold" else None)
            r = await asyncio.wait_for(op.result(), 15.0)
            assert r["state"] == "succeeded" and [h["i"] for h in holds] == [0, 1, 2], (r, holds)
            # B.7 arm_script: a phase is accepted with its plan and ends succeeded into a hold; a cancel ends the
            # next one where the arm is (arm end, hold measured)
            ahead = (p1.x + 0.30 * math.cos(p1.yaw) - 0.2 * math.sin(p1.yaw),
                     p1.y + 0.30 * math.sin(p1.yaw) + 0.2 * math.cos(p1.yaw), 0.95)
            op = await b.arm_script("pregrasp", "left", target_w=ahead, settle_s=0.5, fence=f)
            r = await asyncio.wait_for(op.result(), 15.0)
            if r["state"] == "failed" and r["data"].get("reason") == "ik_unreachable":
                pytest.skip("the fake deploy's arm pose makes this goal unreachable")
            assert r["state"] == "succeeded" and (op.reply or {}).get("data", {}).get("ik_err_m") is not None, r
            op = await b.arm_script("grasp", "left", target_w=ahead, settle_s=3.0, fence=f)
            await asyncio.sleep(0.4)
            op.cancel("test")
            r = await asyncio.wait_for(op.result(), 5.0)
            assert r["state"] in ("succeeded", "canceled") and r["data"].get("ended_by") == "client", r
            assert await _until(lambda: (b.arm_state().get("hold") or {}).get("kind") == "measured", 1.0), \
                b.arm_state()                                        # body.state is published at 5 Hz
            rel = b.arm_release(f)
            assert rel.get("ok"), rel
        finally:
            await b.release(ex.execution_id)
            b.close()
    asyncio.run(main())
