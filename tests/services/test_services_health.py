"""robot/health.py on the lite stack and over the body wire:

  R.6       RTF -> capability: DEGRADED rejects manipulate, UNSAFE rejects every body tool (navigate, manipulate, the
            scan), with a fake sim.health (WorldModel.set_sim_health pins it); speech, glances and
            check_reachability still run. The object_type enum never changes.
  PLAN §5.2 the capability_changed producer: HealthMonitor emits one event per change (capabilities and per-skill
            health), the harness logs it and puts it in a NOTE.
  PLAN §5.6 an unacknowledged halt is re-sent every 100 ms until the body acks it (HaltResender, SonicBody over a
            fake wl-body that answers late).
"""

import asyncio
import threading
import time

import pytest

from api.execution import Rejected
from tests.services.conftest import Stack, run
from world.sim_health import fixed

DEGRADED = fixed("degraded", 0.91, source="fixture",
                 detail="sim below real time: DEGRADED, rtf 0.91 (< 0.95 for 5 s)")
UNSAFE = fixed("unsafe", 0.80, source="fixture", detail="sim below real time: UNSAFE, rtf 0.80 (< 0.85 for 3 s)")


def _start(s, tool, args, **kw):
    return s.robot.start(s.ex(tool, args, **kw))


# ---------------------------------------------------------------------------------------------- R.6
def test_degraded_rejects_manipulate_only():
    s = Stack()

    async def main():
        types = s.robot.registry().loaded_object_types()
        s.world.set_sim_health(DEGRADED)
        caps = s.robot.capabilities()
        assert not caps["manipulation"].ok and "DEGRADED" in caps["manipulation"].detail
        assert caps["navigation"].ok and caps["body"].ok and caps["sim"].state == "degraded"
        with pytest.raises(Rejected) as e:
            _start(s, "manipulate", {"action": "pick", "object_type": "alarm_clock"}, action="pick")
        assert e.value.stage == "capability" and e.value.code == "policy_unavailable"
        assert e.value.message.startswith("policy unavailable: sim below real time: DEGRADED")
        r = await _start(s, "navigate", {"location": "living_room"}, action="keypoint").result()
        assert r.status == "succeeded"
        assert s.robot.registry().loaded_object_types() == types            # the enum is frozen (Invariant 9)
        assert s.robot.telemetry()["body"]["sim"]["state"] == "degraded"
        assert s.robot.telemetry()["body"]["rtf"] == pytest.approx(0.91)
        s.world.set_sim_health(None)
        assert s.robot.capabilities()["manipulation"].ok
    run(main())


def test_unsafe_rejects_every_body_tool():
    s = Stack()

    async def main():
        s.world.set_sim_health(UNSAFE)
        caps = s.robot.capabilities()
        assert not caps["navigation"].ok and not caps["body"].ok and not caps["manipulation"].ok
        with pytest.raises(Rejected) as e:
            _start(s, "navigate", {"location": "living_room"}, action="keypoint")
        assert e.value.code == "nav_unhealthy" and "UNSAFE" in e.value.message
        with pytest.raises(Rejected) as e:
            _start(s, "manipulate", {"action": "place", "object_type": "alarm_clock"}, action="place")
        assert e.value.code == "policy_unavailable"
        with pytest.raises(Rejected) as e:
            _start(s, "observe", {"mode": "scan"}, action="scan")            # a scan turns the body
        assert e.value.code == "controller_unavailable"
        # what does not move the body still runs
        g = await _start(s, "observe", {"mode": "glance"}, action="glance").result()
        assert g.status == "succeeded"
        r = await _start(s, "check_reachability", {"object_type": "apple", "candidates": []}).result()
        assert r.status == "succeeded"
        sp = await _start(s, "speak", {"text": "I have to wait."}).result()
        assert sp.status == "succeeded"
    run(main())


def test_validator_rejects_at_capability_from_the_robot_capabilities():
    """agent/validate.py's CAPABILITY stage reads robot.capabilities(): the planner gets the same answer."""
    from agent.validate import _capability_stage, ValidationContext
    s = Stack()
    s.world.set_sim_health(DEGRADED)
    v = ValidationContext(belief=None, history=[], map={}, task=None, now=0.0,
                          capabilities=s.robot.capabilities(), registry=s.robot.registry())
    verdict = _capability_stage("manipulate", {"action": "pick", "object_type": "apple"}, v, "pick")
    assert verdict is not None and verdict.code == "policy_unavailable" and "DEGRADED" in verdict.message
    s.world.set_sim_health(UNSAFE)
    v.capabilities = s.robot.capabilities()
    verdict = _capability_stage("navigate", {"location": "kitchen"}, v, "keypoint")
    assert verdict is not None and verdict.code == "nav_unhealthy" and "UNSAFE" in verdict.message


# ---------------------------------------------------------------------------------------------- capability_changed
def test_monitor_emits_one_event_per_change():
    s = Stack()

    async def main():
        q = s.robot.events()                          # starts the monitor (baseline, no events)
        mon = s.robot.monitor
        assert mon.poll() == []
        s.world.set_sim_health(DEGRADED)
        evs = mon.poll()
        caps = {e["capability"]: e for e in evs}
        assert set(caps) == {"manipulation", "sim"}
        m = caps["manipulation"]
        assert m["ok"] is False and m["was"] == {"ok": True, "state": "ok"}
        assert m["detail"].startswith("manipulation unavailable: sim below real time: DEGRADED")
        assert mon.poll() == []                       # no change, no event
        s.world.set_sim_health(None)
        back = {e["capability"]: e for e in mon.poll()}
        assert back["manipulation"]["ok"] is True and "available again" in back["manipulation"]["detail"]
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        assert [e["type"] for e in got].count("capability_changed") == 4
        s.robot.monitor.stop()
    run(main())


def test_monitor_reports_a_skill_going_down():
    s = Stack()

    class Flaky:
        backend = "lite"
        name = "lite"
        up = True

        def health(self):
            from api.types import ServiceHealth
            return ServiceHealth(self.up, "ok" if self.up else "down", "" if self.up else "policy server not answering")

    async def main():
        from services.skills import build_registry
        ex = Flaky()
        reg = build_registry(["lite"], s.world, health={"lite": ex.health})
        from robot.health import HealthMonitor
        events = []
        mon = HealthMonitor(lambda: {}, reg, lambda type_, **f: events.append({"type": type_, **f}))
        mon.poll()
        ex.up = False
        mon.poll()
        skills = {e["skill_id"] for e in events}
        assert skills == {"lite.pick.v0", "lite.place.v0"}
        assert all(e["capability"] == "manipulation" and "policy server not answering" in e["detail"] for e in events)
        assert reg.loaded_object_types() == build_registry(["lite"], s.world).loaded_object_types()
    run(main())


def test_the_harness_logs_capability_changed_and_notes_it():
    from agent.harness import Runtime
    from tests.unit.conftest import keep_running, run_until, stop
    from tests.unit.fakes_rt import FakeUser, ScriptBrain

    s = Stack()

    async def main():
        s.robot.monitor.period_s = 0.05
        user = FakeUser(s.clock)
        rt = Runtime(s.robot, user, ScriptBrain([]), s.clock)
        try:
            await run_until(rt, lambda: s.robot.monitor._task is not None, wall_s=5, what="monitor running")
            s.world.set_sim_health(UNSAFE)
            await keep_running(rt, lambda: any(r["type"] == "capability_changed" and r.get("capability") == "navigation"
                                               for r in rt.tracer.rows), wall_s=5, what="capability_changed")
        finally:
            await stop(rt)
            s.robot.monitor.stop()
        row = next(r for r in rt.tracer.rows if r["type"] == "capability_changed" and r["capability"] == "navigation")
        assert row["ok"] is False and "UNSAFE" in row["detail"]
        assert "capability changed" in (rt._note or "") and "UNSAFE" in rt._note
    run(main())


# ---------------------------------------------------------------------------------------------- halt re-send
class LateBody:
    """send_halt acks on the n-th attempt."""
    def __init__(self, ack_on: int):
        self.ack_on = ack_on
        self.attempts = 0

    def send_halt(self, epoch, wait_s):
        self.attempts += 1
        return self.attempts >= self.ack_on


def test_resender_retries_every_100_ms_until_acked():
    from robot.health import HaltResender
    events = []
    body = LateBody(3)
    r = HaltResender(body, lambda type_, **f: events.append((type_, f)))
    t0 = time.monotonic()
    assert r.start(7)
    while r.active and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    dt = time.monotonic() - t0
    assert body.attempts == 3 and 0.25 <= dt < 1.0, (body.attempts, dt)
    assert events == [("halt_acked", {"epoch": 7, "attempts": 3, "latency_ms": events[0][1]["latency_ms"]})]


def test_resender_stops_on_resume_and_gives_up_after_max_s():
    from robot.health import HaltResender
    events = []
    body = LateBody(10 ** 6)
    r = HaltResender(body, lambda type_, **f: events.append((type_, f)))
    r.start(1)
    time.sleep(0.35)
    r.cancel()
    time.sleep(0.25)
    n = body.attempts
    assert 2 <= n <= 4 and not r.active
    time.sleep(0.25)
    assert body.attempts == n and events == []                      # resumed: nothing more is sent
    r2 = HaltResender(body, lambda type_, **f: events.append((type_, f)), max_s=0.3)
    r2.start(2)
    time.sleep(0.8)
    assert events and events[-1][0] == "safety_event" and events[-1][1]["kind"] == "halt_unacked"
    assert HaltResender(object(), lambda *a, **k: None).start(3) is False    # a body without send_halt


def test_sonic_body_late_ack_is_resent_and_acked(monkeypatch):
    """SonicBody over the fake wl-body answering the stop after 60 ms: the receipt says stopped=false (not within
    30 ms), the bridge re-sends every 100 ms, and the late ack ends the re-sends."""
    pytest.importorskip("zmq")
    from api.types import ServiceHealth
    from robot.body_client import SonicBody
    from robot.health import HaltResender
    from tests.fakes.fake_body_server import FakeBodyServer

    from tests.fakes.fake_p1_world import free_port_offset
    off = free_port_offset()
    fb = FakeBodyServer(port_offset=off).start()
    fb.reply_delay_s = 0.06
    try:
        from body.client import BodyClient
        client = BodyClient(port_offset=off).connect(wait_s=5.0)
        body = SonicBody(client, speed_fn=lambda: 0.01)
        rec = body.halt(1)
        assert rec["stopped"] is False and rec["at_rest"] is True
        events = []
        done = threading.Event()

        def emit(type_, **f):
            events.append((type_, f))
            done.set()
        r = HaltResender(body, emit)
        r.start(1)
        assert done.wait(2.0)
        assert events[0][0] == "halt_acked" and events[0][1]["attempts"] >= 1
        assert body.halt_acked(1)
        assert body.health() == ServiceHealth(True, "ok")
        st = body.state()
        assert "gt_pose" not in st and "pose" not in st                  # no simulator truth passes through
        body.close()
    finally:
        fb.stop()
