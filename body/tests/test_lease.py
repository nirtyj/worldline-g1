"""B.2 leases and fencing over the wire (fakes): body_busy for a second owner, stale generations dropped and published
as body.stale_command, supersede by a newer generation, release, the arm op under a lease, a halt revoking the lease."""

import math
import time

import pytest

from .test_integration_fakes import Stack


def _wait(pred, timeout=3.0, dt=0.01):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture
def stack(port_offset, tmp_path):
    s = Stack(port_offset, tmp_path)
    yield s
    s.close()


def test_leases_busy_stale_supersede_release(stack):
    bc, p1 = stack.bc, stack.p1
    seen = []
    bc.add_topic_listener(lambda t, m: seen.append((t, m)))
    stack.stand()
    assert bc.turn_to(0.0).ok
    A = {"execution_id": "exA", "generation": 2, "control_epoch": 1}
    rep = bc.acquire("exA", 2, 1, mode="LOCOMOTION")
    assert rep["ok"] and rep["data"]["lease"]["owner"] == "exA"
    assert bc.status()["lease"]["owner"] == "exA"
    # a second owner and an unfenced M1 tool are both body_busy while A holds the lease
    h = bc.walk(vx=0.4, duration_s=1.0, execution_id="exB", generation=2, control_epoch=1)
    assert h.state == "failed" and h.reason == "body_busy" and h.result["lease"]["owner"] == "exA"
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.state == "failed" and h.reason == "body_busy"
    rep = bc.acquire("exB", 2, 1)
    assert not rep["ok"] and rep["error"] == "body_busy"
    # the owner drives; its op records carry the fence
    h = bc.walk(vx=0.4, duration_s=0.8, **A)
    assert h.ok, h.result
    rec = bc.status(op_id=h.id)["op"]
    assert rec["fence"] == A
    # a stale generation is dropped and published (it never moves the robot)
    x0 = p1.x
    h = bc.walk(vx=0.4, duration_s=1.0, execution_id="exOld", generation=1, control_epoch=1)
    assert h.state == "failed" and h.reason == "stale_command"
    assert _wait(lambda: any(t == "body.stale_command" and m.get("execution_id") == "exOld" and m.get("why") ==
                             "generation" for t, m in seen), 1.0)
    time.sleep(0.3)
    assert abs(p1.x - x0) < 0.02 and math.hypot(p1.vx, p1.vy) < 0.05
    # a newer generation supersedes the lease: A's running walk ends canceled (superseded)
    hw = bc.walk(vx=0.4, duration_s=10.0, wait=False, **A)
    assert _wait(lambda: hw.state == "accepted", 1.0)
    rep = bc.acquire("exC", 3, 1)
    assert rep["ok"] and rep["data"]["superseded"]["owner"] == "exA"
    hw.wait(3.0)
    assert hw.state == "canceled" and hw.result["reason"] == "superseded"
    assert any(t == "body.lease" and m["event"] == "revoked" and m["reason"] == "superseded" for t, m in seen)
    # A's later commands are stale now (generation 2 < 3)
    h = bc.walk(vx=0.4, duration_s=1.0, **A)
    assert h.state == "failed" and h.reason == "stale_command"
    # release: only the owner may
    rep = bc.release("exA")
    assert not rep["ok"] and rep["error"] == "not_owner"
    assert bc.release("exC")["ok"]
    assert bc.status()["lease"] is None
    assert bc.walk(vx=0.4, duration_s=0.6).ok                  # M1 tools work again without a lease
    # stop is never gated by a lease
    bc.acquire("exD", 3, 1)
    assert bc.stop().ok
    # a halt revokes the lease of its epoch
    r = bc.halt(1, timeout_s=0.5)
    assert r["acked"]
    assert _wait(lambda: bc.status()["lease"] is None, 1.0)
    assert any(t == "body.lease" and m["event"] == "revoked" and m["reason"] == "halt" for t, m in seen)
    assert bc.resume(2)["ok"]
    assert stack.dep.stats["stop"] == 0


def test_arm_op_under_lease_and_latch(stack):
    """The fences also gate the arm op (B.2): body_busy for a stream from another execution, halted while latched."""
    bc = stack.bc
    stack.stand()
    assert bc.acquire("exA", 1, 1, mode="ARM_STREAM")["ok"]
    other = bc.arm_stream(stream="s-other")
    rep = other.send(upper_body={"right_elbow_joint": 1.0}, execution_id="exB", generation=1, control_epoch=1)
    assert not rep["ok"] and rep["error"] == "body_busy"
    mine = bc.arm_stream(stream="s-mine")
    rep = mine.send(upper_body={"right_elbow_joint": 1.0}, execution_id="exA", generation=1, control_epoch=1)
    assert rep["ok"], rep
    assert bc.halt(1, timeout_s=0.5)["acked"]
    rep = mine.send(upper_body={"right_elbow_joint": 0.8}, execution_id="exA", generation=1, control_epoch=1)
    assert not rep["ok"] and rep["error"] == "halted"
    assert bc.resume(2)["ok"]
