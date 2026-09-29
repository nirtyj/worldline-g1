"""world/sim_health.py: RTF -> ok / degraded / unsafe with PLAN §3.5's hold times, recovery, P1's own verdict,
staleness; and the WorldModel surface (sim_health, planar_speed) on lite."""

import pytest

from world.sim_health import RtfMonitor, SimHealthConfig, fixed


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def feed(m, clock, rtf, seconds, dt=0.5):
    h = None
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        h = m.update(rtf)
    return h


def test_degraded_needs_5_s_below_0_95_and_unsafe_3_s_below_0_85():
    c = Clock()
    m = RtfMonitor(clock=c)
    assert m.state().state == "ok" and m.state().rtf is None          # nothing known yet
    assert feed(m, c, 1.0, 2.0).state == "ok"
    assert feed(m, c, 0.90, 4.5).state == "ok"                       # below 0.95 for 4.5 s: not yet
    h = feed(m, c, 0.90, 1.0)
    assert h.state == "degraded" and h.degraded and not h.unsafe
    assert "DEGRADED" in h.detail and "0.90" in h.detail
    assert feed(m, c, 0.80, 2.5).state == "degraded"                 # below 0.85 for 2.5 s
    h = feed(m, c, 0.80, 1.0)
    assert h.state == "unsafe" and h.unsafe and "UNSAFE" in h.detail


def test_a_single_hitch_does_not_flap():
    """M1 measured a worst 1 s window of 0.82 in a quiet integrated run: one sample must not change anything."""
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 1.0, 10.0)
    assert feed(m, c, 0.82, 1.0).state == "ok"
    assert feed(m, c, 1.0, 5.0).state == "ok"


def test_recovery_needs_recover_s_above_the_threshold():
    c = Clock()
    m = RtfMonitor(SimHealthConfig(recover_s=2.0), clock=c)
    feed(m, c, 0.8, 6.0)
    assert m.state().state == "unsafe"
    assert feed(m, c, 0.9, 1.5).state == "unsafe"                    # above 0.85 for 1.5 s only
    assert feed(m, c, 0.9, 1.0).state == "degraded"                  # still below 0.95
    assert feed(m, c, 1.0, 1.5).state == "degraded"
    assert feed(m, c, 1.0, 1.0).state == "ok"


def test_instant_thresholds_and_stale_samples():
    c = Clock()
    m = RtfMonitor(SimHealthConfig(degraded_hold_s=0, unsafe_hold_s=0, stale_s=3.0), clock=c)
    c.t += 0.1
    assert m.update(0.93).state == "degraded"
    c.t += 0.1
    assert m.update(0.5).state == "unsafe"
    c.t += 5.0                                                       # P1 went quiet: unknown is not a verdict
    h = m.state()
    assert h.state == "ok" and h.rtf is None and "no RTF" in h.detail


def test_p1_verdict_wins_while_fresh():
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 1.0, 1.0)
    m.force("unsafe", "P1: RTF 0.70 over 3 s")
    h = m.state()
    assert h.state == "unsafe" and h.source == "p1:sim.health" and "0.70" in h.detail
    c.t += 10.0
    assert m.state().state == "ok"                                   # the forced verdict went stale
    m.force("bogus")                                                 # unknown states are ignored
    assert m.state().state == "ok"


def test_config_from_yaml_dict():
    cfg = SimHealthConfig.from_dict({"degraded_below": 0.9, "unsafe_hold_s": 1, "other": 3})
    assert cfg.degraded_below == 0.9 and cfg.unsafe_hold_s == 1.0 and cfg.degraded_hold_s == 5.0


def test_lite_world_has_no_real_time_constraint_and_a_pin(lite38):
    h = lite38.sim_health()
    assert h.ok and h.rtf is None and h.source == "lite-gt"
    lite38.set_sim_health(fixed("degraded", 0.9, source="fixture"))
    try:
        assert lite38.sim_health().state == "degraded" and lite38.rtf() == pytest.approx(0.9)
    finally:
        lite38.set_sim_health(None)
    assert lite38.sim_health().ok


def test_planar_speed_is_the_gt_pose_speed(lite38):
    k = lite38.map.keypoints["start"]
    lite38.set_robot_pose(k.x, k.y, k.yaw, vx=0.3, vy=0.4)
    try:
        assert lite38.planar_speed() == pytest.approx(0.5)
    finally:
        lite38.set_robot_pose(k.x, k.y, k.yaw)
    assert lite38.planar_speed() == pytest.approx(0.0)
