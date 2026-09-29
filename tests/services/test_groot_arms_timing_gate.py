"""tools/groot_timing_gate.py's offline pieces: the walk_diagnosis `irregular` share from a P1 record trace, the
deploy's loop-timing lines between two log offsets, and the hitches that fall in a phase (PLAN §0.11 timing gate)."""

from __future__ import annotations

import numpy as np

from tools.groot_timing_gate import DeployLog, StallClock, hitches_in, leg_target_windows

LINE = ("Loop timing - LowState age: {age}ms, Streaming data mean delay: 0ms, Streaming data std delay: 0ms, IMU age: "
        "1.7ms, Obs: 300us, Policy: {pol}us, Obs 2 Motor Command: {o2c}us, Post processing: 23us, Planner - Gather "
        "Input: 0us, Model: 0us, Convert50Hz: 0us, Total: 0us | HandCloseRatio: 1\n")


def _trace(tmp_path, new_target: list[bool]) -> str:
    """One sample per 20 ms of sim time; the 12 leg targets change where new_target[i] is True."""
    n = len(new_target) + 1
    q = np.zeros((n, 29))
    for i, chg in enumerate(new_target, start=1):
        q[i] = q[i - 1]
        if chg:
            q[i, :12] += 0.01
        q[i, 20] += 0.001            # an arm target moving every window must not count as a leg target
    p = tmp_path / "trace.npz"
    np.savez(p, t_sim=np.arange(n) * 0.02, q_target=q)
    return str(p)


def test_a_regular_trace_has_no_irregular_windows(tmp_path):
    r = leg_target_windows(_trace(tmp_path, [True] * 200))
    assert r["n_windows"] == 200 and r["no_new_leg_target_frac"] == 0.0 and r["irregular_est"] == 0.0


def test_empty_windows_count_twice_in_the_estimate(tmp_path):
    pattern = ([True] * 9 + [False]) * 20                  # 10 % of the windows see no new leg target
    r = leg_target_windows(_trace(tmp_path, pattern))
    assert r["no_new_leg_target_frac"] == 0.1 and r["irregular_est"] == 0.2


def test_a_sim_jump_is_not_a_window(tmp_path):
    p = _trace(tmp_path, [True] * 50)
    z = dict(np.load(p))
    z["t_sim"][25:] += 1.0                                  # a reset or a stall: that pair is not 20 ms apart
    np.savez(p, **z)
    assert leg_target_windows(p)["n_windows"] == 49


def test_a_missing_trace_is_reported(tmp_path):
    assert "error" in leg_target_windows(str(tmp_path / "none.npz"), wait_s=0.0)


def test_deploy_loop_lines_between_offsets(tmp_path):
    log = tmp_path / "deploy.log"
    log.write_text(LINE.format(age=9.9, pol=9999, o2c=99999) + "other line\n")
    d = DeployLog(str(log))
    o0 = d.offset()
    with log.open("a") as f:
        for i in range(10):
            f.write(LINE.format(age=1.0 + i * 0.1, pol=100 + i, o2c=400 + 10 * i))
    w = d.window(o0, d.offset())
    assert w["n"] == 10
    assert w["obs2cmd_us"]["max"] == 490.0 and w["policy_us"]["max"] == 109.0 and w["lowstate_age_ms"]["max"] == 1.9
    assert 400 < w["obs2cmd_us"]["p50"] < 490
    assert DeployLog(None).window(0, 10) == {"n": 0, "log": None}


def test_hitches_in_the_phase_window():
    st = {"hitches_last": [{"t_sim": 10.0, "render_ms": 60}, {"t_sim": 20.0, "render_ms": 90}]}
    assert [h["t_sim"] for h in hitches_in(st, 15.0, 25.0)] == [20.0]
    assert hitches_in(st, None, 25.0) == []


def test_the_stall_clock_places_a_phase_between_two_stalls():
    c = StallClock(29.9)
    c.observe({"hitches_last": [{"t_sim": 100.0, "render_ms": 120.0, "cams": ["head"]},
                                {"t_sim": 101.0, "render_ms": 30.0, "cams": ["head"]},     # too short: not the stall
                                {"t_sim": 110.0, "render_ms": 90.0, "cams": []}]})       # no head render: not it
    assert c.anchor == 100.0 and c.next_after(105.0) == 129.9
    assert c.wait_s(101.5, 20.0) == 0.0                     # 101.5 .. 121.5 fits before 129.9
    assert abs(c.wait_s(115.0, 20.0) - 15.9) < 1e-9          # would cross 129.9: start at 130.9
    assert c.wait_s(115.0, 40.0) == 0.0                      # longer than the period: cannot dodge, no wait
    c.observe({"hitches_last": [{"t_sim": 129.95, "render_ms": 70.0, "cams": ["head", "ego_view"]}]})
    assert c.anchor == 129.95                               # re-anchored on the beat
    assert StallClock().wait_s(10.0, 20.0) == 0.0            # no anchor yet: no wait


def test_the_trace_gives_the_p1_loop_wall_time_per_window_and_the_empty_count(tmp_path):
    p = _trace(tmp_path, ([True] * 4 + [False]) * 40)       # one empty window in five: a render every 100 ms
    z = dict(np.load(p))
    tw = np.cumsum(np.tile([0.035, 0.010, 0.020, 0.015, 0.020], 41)[: z["t_sim"].size])
    np.savez(p, **z, t_wall=tw)
    r = leg_target_windows(p)
    assert r["no_new_leg_target_n"] == 40 and r["no_new_leg_target_frac"] == 0.2 and r["irregular_est"] == 0.4
    w = r["wall_ms_per_window"]
    assert w["max"] == 35.0 and w["p50"] == 20.0 and w["over_30ms_frac"] == 0.2


def test_cpu_lists_parse_like_taskset():
    from tools.groot_timing_gate import parse_cpus
    assert parse_cpus("4-15") == set(range(4, 16))
    assert parse_cpus("4,6,8-9") == {4, 6, 8, 9}


def test_a_stopped_local_server_is_not_reported_as_the_placement(tmp_path, monkeypatch):
    import json

    import tools.groot_timing_gate as tg
    real = tg.Path
    state = tmp_path / "server-5550.json"
    state.write_text(json.dumps({"pid": 2 ** 22 + 12345, "started_utc": "x", "host": "h", "taskset": "4-15"}))

    def fake_path(p, *a):
        p = str(p)
        if p.endswith("server-5550.json"):
            return real(state)
        if p.endswith("link-5550.env"):
            return real(tmp_path / "none.env")
        return real(p, *a)
    monkeypatch.setattr(tg, "Path", fake_path)
    r = tg.server_placement()
    assert r["where"] == "link" and r["stale_local_state"]["pid"] == 2 ** 22 + 12345
    assert r["state"] == "no link state file"


def test_an_empty_window_is_a_catch_up_window_after_a_stall_and_still_got_lowcmd(tmp_path):
    """As measured on the main box (B gate): the deploy publishes lowcmd at 4x its 50 Hz policy, so an empty window
    still received messages; it is the short window P1 runs right after a stalled one."""
    p = _trace(tmp_path, ([True] * 7 + [False]) * 25)            # one empty in eight: VizCams' ~6 Hz self-render
    z = dict(np.load(p))
    n = z["t_sim"].size
    dw = np.tile([0.020] * 6 + [0.035, 0.008], 26)[: n - 1]      # the long window, then the empty catch-up window
    np.savez(p, **z, t_wall=np.concatenate([[0.0], np.cumsum(dw)]), lowcmd_count=np.arange(n) * 4)
    r = leg_target_windows(p)
    w = r["wall_ms_per_window"]
    assert r["no_new_leg_target_n"] == 25 and w["empties_under_12ms_frac"] == 1.0
    assert w["long_over_30ms_per_s"] == round(25 / (0.02 * (n - 1)), 2)
    assert r["lowcmd_per_window_p50"] == 4.0 and r["empty_windows_with_lowcmd"] == 25


def test_viz_stats_is_none_without_vizcams_and_keeps_the_forced_renders():
    from tools.groot_timing_gate import viz_stats

    class Rpc:
        def __init__(self, reply):
            self.reply = reply

        def call(self, op, **kw):
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply
    assert viz_stats(Rpc(RuntimeError("unknown op viz_stats"))) is None
    assert viz_stats(Rpc({"ok": True, "level": "min", "forced_renders": 12, "piggyback_captures": 3, "sent": 15,
                          "capture_ms_mean": 1.0})) == {"level": "min", "forced_renders": 12,
                                                        "piggyback_captures": 3, "sent": 15}
