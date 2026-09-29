"""The RTF capability with hysteresis and dwell end to end on the robot side (PLAN §3.5; the 2026-09-29 stance-fix
live finding): world/sim_health.py RtfMonitor behind WorldModel.sim_health -> robot/health.py CapabilityPolicy ->
HealthMonitor's capability_changed and G1Robot.start's CAPABILITY gate.

    a sim.health trace in the live band (rtf_5s 0.92-0.96) for 5 min: no capability_changed at all (the old instant
        P1 level gave ~21 events a minute on the same trace)
    a real drop (rtf_5s < 0.90 for >= 3 s): one change to degraded (manipulation + sim), and one back
    a pick right after a walk, during a 1-2 s dip: not rejected; after a sustained drop: rejected (policy_unavailable)
"""

from __future__ import annotations

import pytest

from api.execution import Rejected
from api.types import ServiceHealth
from robot.health import CapabilityPolicy, HealthMonitor
from tests.services.conftest import Stack, run
from tests.fakes.rtf_trace import Clock, live_trace
from world.sim_health import RtfMonitor, SimHealthConfig

OK = ServiceHealth(True, "ok")


class _Svc:
    def health(self):
        return OK


class _World:
    def __init__(self, mon: RtfMonitor):
        self.mon = mon

    def sim_health(self):
        return self.mon.state()


class _Registry:
    def skills(self):
        return []


def _monitor(cfg: SimHealthConfig | None = None):
    clock = Clock()
    mon = RtfMonitor(cfg, clock=clock, source="p1:sim.health")
    policy = CapabilityPolicy(_World(mon), nav=_Svc(), manip=_Svc(), body=_Svc(), observation_detail=lambda: "")
    events: list[dict] = []
    hm = HealthMonitor(policy.capabilities, _Registry(), lambda type_, **f: events.append({"type": type_, **f}))
    return clock, mon, policy, hm, events


def _play(clock, mon, hm, trace):
    """P1's sim.health at 1 Hz, HealthMonitor polling at 4 Hz."""
    hm.poll()                                               # the baseline
    for r1, r3, r5 in trace:
        mon.update(r1, rtf_3s=r3, rtf_5s=r5, level="degraded" if r5 < 0.95 else "ok", source="p1:sim.health")
        for _ in range(4):
            clock.t += 0.25
            hm.poll()


def test_the_live_rtf_band_emits_no_capability_changed():
    clock, mon, policy, hm, events = _monitor()
    _play(clock, mon, hm, live_trace(300, 0.92, 0.96))
    assert events == []
    assert policy.gate("manipulate", {"action": "pick"}) is None
    # the old rule on the same trace: P1's instant level (rtf_5s < 0.95, no dwell) flapped every few seconds
    clock, mon, _, hm, old = _monitor(SimHealthConfig(degraded_below=0.95, degraded_hold_s=0.0, recover_above=0.95,
                                                      recover_s=0.0))
    _play(clock, mon, hm, live_trace(300, 0.92, 0.96))
    assert len(old) / 5.0 > 10                              # capability_changed per minute


def test_a_real_drop_is_one_change_each_way():
    clock, mon, policy, hm, events = _monitor()
    trace = [(1.0, 1.0, 1.0)] * 10 + [(0.86, 0.87, 0.87)] * 20 + [(0.99, 0.98, 0.97)] * 10
    _play(clock, mon, hm, trace)
    caps = [(e["capability"], e["state"]) for e in events]
    assert caps == [("manipulation", "degraded"), ("sim", "degraded"), ("manipulation", "ok"), ("sim", "ok")]
    assert "DEGRADED" in events[0]["detail"] and "rtf_5s 0.87" in events[0]["detail"]


def test_a_short_dip_does_not_reject_manipulate_a_sustained_drop_does():
    clock, mon, policy, hm, events = _monitor()
    _play(clock, mon, hm, [(0.95, 0.94, 0.93)] * 20)        # the live band
    _play(clock, mon, hm, [(0.80, 0.86, 0.88)] * 2)         # a 1-2 s dip right after a walk
    assert policy.gate("manipulate", {"action": "pick"}) is None and events == []
    _play(clock, mon, hm, [(0.80, 0.86, 0.88)] * 3)         # held 3 s: DEGRADED
    g = policy.gate("manipulate", {"action": "pick"})
    assert g is not None and g.code == "policy_unavailable" and "DEGRADED" in g.message
    assert policy.gate("navigate", {"location": "kitchen"}) is None


def test_a_pick_right_after_a_walk_is_not_rejected_during_a_dip():
    """The bridge's CAPABILITY gate (G1Robot.start) on a world whose sim health is the live monitor."""
    from tests.services.test_services_manipulation import _ready_to_pick
    s = Stack()
    clock = Clock()
    mon = RtfMonitor(clock=clock, source="p1:sim.health")
    s.world._live_sim_health = mon.state

    def feed(n, r3, r5):
        for _ in range(n):
            clock.t += 1.0
            mon.update(r5, rtf_3s=r3, rtf_5s=r5, level="degraded" if r5 < 0.95 else "ok")

    async def main():
        await _ready_to_pick(s)
        feed(20, 0.94, 0.93)                                 # the live band while it walked
        feed(2, 0.86, 0.88)                                  # a 1-2 s dip on arrival
        assert s.world.sim_health().ok
        p = await s.run("manipulate", {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1"})
        assert p.status == "succeeded", p.summary
        # a sustained drop is still a DEGRADED rejection
        feed(4, 0.86, 0.88)
        assert s.world.sim_health().state == "degraded"
        with pytest.raises(Rejected) as e:
            s.robot.start(s.ex("manipulate", {"action": "place", "object_type": "alarm_clock"}, action="place"))
        assert e.value.code == "policy_unavailable" and "DEGRADED" in e.value.message
    run(main())
