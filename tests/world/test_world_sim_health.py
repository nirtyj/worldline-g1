"""world/sim_health.py: RTF -> ok / degraded / unsafe with hysteresis and dwell (PLAN §3.5, 2026-09-29): degraded
after rtf_5s < 0.90 for 3 s, ok again at rtf_5s >= 0.94 for 2 s; unsafe after rtf_3s < 0.85 for 2 s (< 0.70 at
once), out at rtf_3s >= 0.90 for 2 s. P1's windows are used, its instant level is not; windows from 1 s samples when
P1 sends none (gt.pose); staleness; the live stance-fix trace (rtf_5s 0.92-0.96) gives no level change at all; and the
WorldModel surface (sim_health, planar_speed) on lite."""

import pytest

from tests.fakes.rtf_trace import Clock, live_trace
from world.sim_health import RtfMonitor, SimHealthConfig, fixed


def feed(m, clock, seconds, dt=0.5, rtf=None, r3=None, r5=None, level=None):
    """P1's sim.health every `dt`: the same windows for `seconds`."""
    h = None
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        h = m.update(rtf if rtf is not None else r3, rtf_3s=r3, rtf_5s=r5, level=level, source="p1:sim.health")
    return h


def test_defaults_are_the_hysteretic_floor():
    c = SimHealthConfig()
    assert (c.degraded_below, c.degraded_hold_s, c.recover_above, c.recover_s) == (0.90, 3.0, 0.94, 2.0)
    assert (c.unsafe_below, c.unsafe_hold_s, c.unsafe_now_below, c.unsafe_recover_above) == (0.85, 2.0, 0.70, 0.90)


def test_degraded_needs_3_s_below_0_90_and_unsafe_2_s_below_0_85():
    c = Clock()
    m = RtfMonitor(clock=c)
    assert m.state().state == "ok" and m.state().rtf is None          # nothing known yet
    assert feed(m, c, 2.0, r3=1.0, r5=1.0).state == "ok"
    assert feed(m, c, 3.0, r3=0.89, r5=0.88).state == "ok"           # 6 samples below 0.90: 2.5 s since the first
    h = feed(m, c, 0.5, r3=0.89, r5=0.88)                            # 3 s
    assert h.state == "degraded" and h.degraded and not h.unsafe
    assert "DEGRADED" in h.detail and "rtf_5s 0.88" in h.detail and "< 0.9 for 3 s" in h.detail
    assert h.rtf_5s == pytest.approx(0.88) and h.to_dict()["rtf_5s"] == pytest.approx(0.88)
    assert feed(m, c, 2.0, r3=0.80, r5=0.86).state == "degraded"     # rtf_3s below 0.85 for 1.5 s
    h = feed(m, c, 0.5, r3=0.80, r5=0.86)                            # 2 s
    assert h.state == "unsafe" and h.unsafe and "UNSAFE" in h.detail and "rtf_3s 0.80" in h.detail
    assert [(a, b) for _, a, b in m.transitions] == [("ok", "degraded"), ("degraded", "unsafe")]


def test_recovery_needs_the_higher_threshold_for_2_s():
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 4.0, r3=0.88, r5=0.88)
    assert m.state().state == "degraded"
    assert feed(m, c, 10.0, r3=0.93, r5=0.93).state == "degraded"   # above the floor, below 0.94: stays
    assert feed(m, c, 2.0, r3=0.95, r5=0.95).state == "degraded"    # >= 0.94 for 1.5 s only
    h = feed(m, c, 0.5, r3=0.95, r5=0.95)
    assert h.state == "ok" and h.detail.startswith("rtf ")
    assert len(m.transitions) == 2


def test_unsafe_leaves_through_degraded_until_the_5_s_window_recovers():
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 3.0, r3=0.80, r5=0.84)
    assert m.state().state == "unsafe"
    assert feed(m, c, 2.0, r3=0.91, r5=0.88).state == "unsafe"      # rtf_3s >= 0.90 for 1.5 s only
    h = feed(m, c, 0.5, r3=0.91, r5=0.88)
    assert h.state == "degraded" and "after UNSAFE" in h.detail
    assert feed(m, c, 2.5, r3=0.97, r5=0.95).state == "ok"
    # both windows back at once: straight to ok
    feed(m, c, 3.0, r3=0.80, r5=0.84)
    assert m.state().state == "unsafe"
    assert feed(m, c, 2.5, r3=0.97, r5=0.96).state == "ok"


def test_a_stalled_sim_is_unsafe_at_once():
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 2.0, r3=1.0, r5=1.0)
    h = feed(m, c, 0.5, r3=0.62, r5=0.75)
    assert h.state == "unsafe" and "< 0.7" in h.detail


def test_p1_instant_level_is_not_adopted_when_its_windows_are_there():
    """The live finding: P1 said degraded at rtf_5s 0.93 (its own 0.95 floor, no dwell); world keeps ok."""
    c = Clock()
    m = RtfMonitor(clock=c)
    h = feed(m, c, 30.0, rtf=0.90, r3=0.92, r5=0.93, level="degraded")
    assert h.state == "ok" and m.transitions == []


def test_a_p1_level_without_numbers_goes_through_the_same_dwell():
    c = Clock()
    m = RtfMonitor(clock=c)
    for _ in range(4):                       # 2 s of "unsafe": not yet
        c.t += 0.5
        m.update(None, level="unsafe")
    assert m.state().state == "ok"
    c.t += 0.5
    assert m.update(None, level="unsafe").state == "unsafe"
    for _ in range(5):
        c.t += 0.5
        m.update(None, level="ok")
    assert m.state().state == "ok"
    m.update(None, level="bogus")            # unknown levels carry nothing
    assert m.state().state == "ok"


def test_windows_from_1_s_samples_when_p1_sends_none():
    """gt.pose (M1 P1): the 1 s RTF at 50 Hz; one hitch (M1's worst 1 s window 0.82) changes nothing, a sustained
    drop does after the dwell."""
    c = Clock()
    m = RtfMonitor(clock=c)
    for i in range(500):                     # 10 s at 1.0 with a 1 s hitch at 0.82
        c.t += 0.02
        m.update(0.82 if 200 <= i < 250 else 1.0, source="p1:gt.pose")
    assert m.state().state == "ok" and m.transitions == []
    for _ in range(400):                     # 8 s at 0.87: the 5 s mean crosses 0.90 (~4 s), then 3 s of dwell
        c.t += 0.02
        m.update(0.87)
    h = m.state()
    assert h.state == "degraded" and h.source == "p1:gt.pose" and h.rtf_5s == pytest.approx(0.87)


def test_stale_samples_are_unknown_and_restart_the_dwell():
    c = Clock()
    m = RtfMonitor(clock=c)
    feed(m, c, 4.0, r3=0.88, r5=0.88)
    assert m.state().state == "degraded"
    c.t += 5.0                                                       # P1 went quiet: unknown is not a verdict
    h = m.state()
    assert h.state == "ok" and h.rtf is None and "no RTF" in h.detail
    assert feed(m, c, 2.0, r3=0.88, r5=0.88).state == "ok"           # a fresh dwell
    assert feed(m, c, 1.5, r3=0.88, r5=0.88).state == "degraded"


def _changes_per_minute(cfg, trace):
    c = Clock()
    m = RtfMonitor(cfg, clock=c)
    for r1, r3, r5 in trace:
        for _ in range(4):                   # read at 4 Hz like HealthMonitor
            c.t += 0.25
            m.state()
        m.update(r1, rtf_3s=r3, rtf_5s=r5, level="degraded" if r5 < 0.95 else "ok")
    return len(m.transitions) / (len(trace) / 60.0), m


def test_the_live_rtf_band_gives_no_level_changes():
    trace = live_trace(600, 0.92, 0.96)
    assert sum(1 for _, _, r5 in trace if r5 < 0.95) > 150          # the old floor would have fired constantly
    per_min, m = _changes_per_minute(SimHealthConfig(), trace)
    assert per_min == 0 and m.state().state == "ok"
    # the old rule (P1's instant level: degraded = rtf_5s < 0.95, no dwell) on the same trace
    old = SimHealthConfig(degraded_below=0.95, degraded_hold_s=0.0, recover_above=0.95, recover_s=0.0)
    old_per_min, _ = _changes_per_minute(old, trace)
    assert old_per_min > 5


def test_a_band_across_the_floor_changes_at_most_once_a_minute():
    """rtf_5s wandering 0.87-0.93 (across the 0.90 floor): the dwell and the 0.94 way back keep it to <= 1/min."""
    for seed in range(5):
        trace = live_trace(600, 0.87, 0.93, seed=seed)
        per_min, _ = _changes_per_minute(SimHealthConfig(), trace)
        assert per_min <= 1.0, (seed, per_min)


def test_config_from_yaml_dict():
    cfg = SimHealthConfig.from_dict({"degraded_below": 0.9, "unsafe_hold_s": 1, "other": 3})
    assert cfg.degraded_below == 0.9 and cfg.unsafe_hold_s == 1.0 and cfg.degraded_hold_s == 3.0


def test_the_box_config_matches_the_defaults():
    import yaml
    from pathlib import Path
    d = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "g1.yaml").read_text())["sim_health"]
    assert SimHealthConfig.from_dict(d) == SimHealthConfig()


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
