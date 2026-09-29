"""ArmChannel tick by tick (body/tests/arm_sim.py: fake clock, recording mux, SONIC-like arm plant): the G0 defects
(D1 stop sticks, watchdog = failed, slew limit exact, servo overshoot), the B.1 halt latch interface, and B.8 chunk
mode against docs/contracts/arm_chunk.md v0.1 (the sequences tests/fakes/fake_arm_body.py implements on the runtime
side: session start, chunks with lead 0, a late chunk, an out-of-order seq, a stale session, a NaN chunk, cancel,
halt)."""

import math
import time

import pytest

from body import joint_map as jm
from body.arm import ARM_IDX, ArmError

from .arm_sim import DT, Rig, base, chunk

R_ELBOW = jm.UPPER_BODY_MUJOCO_JOINTS.index("right_elbow_joint")
L_SHP = jm.UPPER_BODY_MUJOCO_JOINTS.index("left_shoulder_pitch_joint")


def _ramp_rows(q0, k, slope, T=40, dt=0.02, t_off=0.0):
    """T rows of q0 with joint k moving at `slope` rad/s, row i at time t_off + i*dt."""
    out = []
    for i in range(T):
        r = list(q0)
        r[k] += slope * (t_off + i * dt)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ G0 defects
def test_stop_sticks_while_the_owner_keeps_streaming():
    rig = Rig({"arm_blend_s": 0.4})
    ch = rig.ch
    q0 = ch.reference_mj17()
    tgt = dict(right_elbow_joint=q0[R_ELBOW] + 0.3)
    assert rig.arm({"stream": "A", "upper_body": tgt}, op_id="a1")["state"] == "accepted"
    rig.run(0.5, each=lambda t: rig.arm({"stream": "A", "upper_body": tgt}))
    assert ch.end("stop")
    rejected = []
    rig.run(1.0, each=lambda t: rejected.append(rig.arm({"stream": "A", "upper_body": tgt})["error"]))
    assert set(rejected) == {"arm_stopped"}                    # during and after the blend: D1
    ev = rig.terminal("a1")
    assert ev["state"] == "canceled" and ev["data"]["ended_by"] == "stop" and ev["data"]["reason"] == "stop"
    assert rig.mux.upper is None and ch.state()["mode"] == "off"
    # a new stream id may start; the stopped one needs restart: true
    assert rig.arm({"stream": "A", "upper_body": tgt, "restart": True}, op_id="a2")["state"] == "accepted"
    assert rig.arm({"stream": "B", "upper_body": tgt})["error"] == "arm_busy"


def test_stop_ends_a_chunk_session_and_its_messages_are_stale():
    rig = Rig({"arm_blend_s": 0.3})
    q0 = rig.ch.reference_mj17()
    assert rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="c1")["ok"]
    rig.run(0.2)
    assert rig.ch.end("stop")
    rig.ch.flush()
    ev = rig.terminal("c1")
    assert ev["state"] == "canceled" and ev["data"]["ended_by"] == "stop" and ev["data"]["hold"] == "stand"
    rep = rig.arm({**base(), "chunk": chunk(2, rig.clock(), [q0] * 40)})
    assert rep["error"] == "stale_session" and rep["data"]["ended_by"] == "stop"
    rig.run(0.5)
    assert rig.mux.upper is None


def test_watchdog_ends_sessions_failed_client_silent():
    rig = Rig({"arm_hold_s": 0.2, "arm_blend_s": 0.3})
    q0 = rig.ch.reference_mj17()
    rig.arm({"stream": "A", "upper_body": q0}, op_id="t1")
    rig.run(1.2)                                        # 0.3 watchdog + 0.2 hold + 0.3 blend
    ev = rig.terminal("t1")
    assert ev["state"] == "failed" and ev["data"]["ended_by"] == "watchdog" and ev["data"]["reason"] == "client_silent"
    # chunk session: watchdog_s (default 2.0) -> failed into hold_on_end (default measured); the arm keeps the pose
    rig.arm({**base("s2"), "watchdog_s": 0.5, "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="c2")
    rig.run(0.8)
    ev = rig.terminal("c2")
    assert ev["state"] == "failed" and ev["data"]["reason"] == "client_silent" and ev["data"]["hold"] == "measured"
    assert rig.ch.state()["mode"] == "hold" and rig.ch.hold.kind == "measured"


def test_slew_limit_is_exact_whatever_the_message_rate():
    rig = Rig({"arm_servo_ki": 0.0})
    q0 = rig.ch.reference_mj17()
    goal = q0[R_ELBOW] + 1.0
    rig.arm({"stream": "A", "max_vel": 2.0, "upper_body": {"right_elbow_joint": goal}})
    vals = []

    def each(t):                                          # 3 messages per tick (150 Hz)
        for _ in range(3):
            rig.arm({"stream": "A", "upper_body": {"right_elbow_joint": goal}})

    for _ in range(40):
        rig.run(DT, each=each)
        vals.append(rig.sent()[R_ELBOW])
    steps = [b - a for a, b in zip(vals, vals[1:])]
    assert max(steps) <= 2.0 * DT + 1e-9
    assert vals[-1] == pytest.approx(goal, abs=1e-9)
    assert sum(1 for s in steps if s > 0.99 * 2.0 * DT) >= 20     # it really ran at the limit


def _step_overshoot(model: str) -> tuple[float, float]:
    rig = Rig({"arm_servo_model": model}, plant_kw={"dead_s": 0.09, "tau_s": 0.085})
    rig.plant.bias[R_ELBOW] = 0.12                          # SONIC settles 0.12 rad past the target
    q0 = rig.ch.reference_mj17()
    a = q0[R_ELBOW] - 0.2
    rig.arm({"stream": "S", "upper_body": {"right_elbow_joint": a}})
    rig.run(4.0, each=lambda t: rig.arm({"stream": "S", "upper_body": {"right_elbow_joint": a}}))
    ys = []
    b = a + 0.3

    def each(t):
        rig.arm({"stream": "S", "upper_body": {"right_elbow_joint": b}})
        ys.append(rig.q()[R_ELBOW])

    rig.run(4.0, each=each)
    over = max(0.0, (max(ys) - b) / 0.3)
    return 100.0 * over, abs(ys[-1] - b)


def test_servo_gated_model_cuts_the_step_overshoot_and_keeps_static_accuracy():
    ov_delay, err_delay = _step_overshoot("delay")
    ov_gated, err_gated = _step_overshoot("gated")
    assert err_delay < 0.01 and err_gated < 0.01            # both remove the 0.12 rad bias
    assert ov_delay > 3.0                                   # G0 servo: the integrator winds up in the transient
    assert ov_gated < 0.5 * ov_delay, (ov_gated, ov_delay)


# ------------------------------------------------------------------------------------------------ halt latch
def test_latch_freezes_measured_pose_keeps_hand_targets_and_rejects_stale_epochs():
    rig = Rig({"arm_servo_ki": 0.0, "arm_blend_s": 0.3})
    ch = rig.ch
    q0 = ch.reference_mj17()
    rows = _ramp_rows(q0, R_ELBOW, 0.8)
    half = jm.hand_closure("right", 0.5)
    assert rig.arm({**base(epoch=2), "right_hand": half, "left_hand": 0.0,
                    "chunk": chunk(1, rig.clock(), rows, hands=0.5)}, op_id="c1")["ok"]
    rig.run(0.4)
    q_meas = rig.q()
    sent_at = rig.sent()
    hands_at = (list(rig.mux.upper[2]), list(rig.mux.upper[3]))          # the hand TARGETS on the wire
    t0 = time.perf_counter()
    info = ch.latch(2, "halt")
    ms = (time.perf_counter() - t0) * 1e3
    assert info["latched"] and info["arms"] == "held" and info["op"] == "c1" and info["pending"]
    assert ms < 5.0 and info["latch_ms"] < 5.0
    assert rig.sent() == pytest.approx(sent_at, abs=1e-12)      # the wire froze at once: no new row after the ack
    assert rig.mux.upper[1] == pytest.approx([0.0] * 17)          # ... with zero velocity
    assert ch.state()["mode"] == "latched"
    # the rest is applied by the next tick (the lane never takes the arm lock); its terminal event comes with it
    assert rig.terminal("c1") is None
    d_prev = abs(rig.sent()[R_ELBOW] - q_meas[R_ELBOW])
    for _ in range(25):
        rig.run(DT)
        d = abs(rig.sent()[R_ELBOW] - q_meas[R_ELBOW])
        assert d <= d_prev + 1e-12                               # only ever towards the measured pose
        d_prev = d
    assert rig.sent()[R_ELBOW] == pytest.approx(q_meas[R_ELBOW], abs=1e-9)
    ev = rig.terminal("c1")
    assert ev["state"] == "canceled" and ev["data"]["ended_by"] == "halt" and ev["data"]["hold"] == "measured"
    li = ch.latch_info
    assert li["case"] == "session" and li["pose_source"] == "g1_debug.body_q" and li["hands_source"] == "last_target"
    assert li["waist"] == "ref"
    # hands: the last TARGET at the latch (B-D2: never the measured q, which lags and ratchets a grip open)
    assert rig.mux.upper[2] == pytest.approx(hands_at[0]) and rig.mux.upper[3] == pytest.approx(hands_at[1])
    # rejected while latched, whatever the epoch; stale epochs rejected for good
    for ep in (1, 2, 3):
        assert rig.arm({**base("s9", epoch=ep), "chunk": chunk(1, rig.clock(), rows)})["error"] == "halted"
    assert rig.arm({"stream": "legacy", "upper_body": q0})["error"] == "halted"
    assert ch.end("stop") is False and ch.state()["mode"] == "latched"
    assert ch.handle_scan("sc", {}, (True, None, {}))["error"] == "halted"
    assert ch.handle_arm_script("as", {"phase": "release", "arm": "left"}, (True, None, {}))["error"] == "halted"
    ch.unlatch(1)                                               # an older resume does not open a newer latch
    assert ch.state()["mode"] == "latched"
    ch.unlatch(2)
    assert ch.state()["mode"] == "hold" and ch.hold.kind == "latched"
    assert rig.arm({**base("s9", epoch=2), "chunk": chunk(1, rig.clock(), rows)})["error"] == "halted"
    assert rig.arm({"stream": "legacy", "upper_body": q0})["error"] == "halted"
    # a new owner with control_epoch > halt_epoch takes the latched pose over, continuing from it
    held = rig.sent()
    assert rig.arm({**base("s10", epoch=3)}, op_id="c3")["state"] == "accepted"
    rig.run(DT)
    assert rig.sent() == pytest.approx(held, abs=1e-6)          # before its first chunk: the pose it took over
    assert rig.arm({**base("s10", epoch=3), "end": True, "hold_on_end": "stand"})["ok"]
    rig.run(0.5)
    assert rig.mux.upper is None


def test_latch_then_end_blends_back_after_resume():
    rig = Rig({"arm_blend_s": 0.3})
    q0 = rig.ch.reference_mj17()
    rig.arm({"stream": "x0", "upper_body": q0}, op_id="t0")         # a v0.5 stream owns the arms
    rig.run(0.1)
    rig.ch.latch(5)
    rig.run(DT)
    assert rig.mux.upper is not None and rig.ch.state()["mode"] == "latched"
    assert rig.arm({"stream": "x", "end": True})["error"] == "halted"
    rig.ch.unlatch(5)
    assert rig.arm({"stream": "x", "end": True, "control_epoch": 6})["data"]["released"] == "latched"
    rig.run(0.5)
    assert rig.mux.upper is None and rig.ch.state()["mode"] == "off"
    assert rig.arm({"stream": "y", "upper_body": q0}, op_id="n")["state"] == "accepted"   # the fence is epochs only


def test_latch_is_fast_with_a_real_sized_state():
    rig = Rig()
    q0 = rig.ch.reference_mj17()
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 64)})
    rig.run(0.1)
    worst = 0.0
    for ep in range(10):
        t0 = time.perf_counter()
        rig.ch.latch(10 + ep)
        worst = max(worst, (time.perf_counter() - t0) * 1e3)
        rig.ch.unlatch(10 + ep)
        rig.arm({**base(f"s{ep}", epoch=11 + ep), "chunk": chunk(1, rig.clock(), [q0] * 64)})
        rig.run(0.04)
    assert worst < 5.0, worst


# ------------------------------------------------------------------------------------------------ B.8 chunk mode
def test_chunk_contract_sequence():
    """arm_chunk.md §7: session start, 3 chunks (lead 0), a late chunk, an out-of-order seq, a stale session_id, a
    NaN chunk, then cancel; the reply and event shapes of §1.4 and §3."""
    rig = Rig({"arm_servo_ki": 0.0, "arm_max_vel": 50.0})
    ch = rig.ch
    q0 = ch.reference_mj17()
    rep = rig.arm({**base(), "lead_s": 0.0, "hold_on_end": "measured", "left_hand": [0.0] * 7,
                   "right_hand": [0.0] * 7, "hands_blend_s": 0.3}, op_id="arm-s1")
    assert rep["ok"] and rep["state"] == "accepted" and rep["data"]["session"]["chunk_seq"] == 0
    assert ch.state() == {**ch.state(), "mode": "chunk", "owner": "s1", "session_id": "s1", "op_id": "arm-s1"}
    assert ch.snapshot()["modes"] == ["target", "chunk"]
    rig.run(0.4)                                                  # start hands blended open, arms on the start pose
    assert rig.mux.upper[2] == pytest.approx([0.0] * 7) and rig.sent() == pytest.approx(q0, abs=1e-9)
    # 3 chunks, each 0.4 s after the previous, rows = a ramp in the right elbow (consistent across chunks)
    ts = []
    for seq in (1, 2, 3):
        t0 = rig.clock() - 0.15                                    # 150 ms of "inference"
        rows = _ramp_rows(q0, R_ELBOW, 0.5, t_off=t0 - 1000.4)
        rep = rig.arm({**base(), "chunk": chunk(seq, t0, rows, inference_ms=150.0)})
        assert rep["ok"] and rep["state"] == "done" and "dropped" not in rep["data"], rep
        assert rep["data"]["session"]["chunk_seq"] == seq
        for _ in range(20):
            rig.run(DT)
            # time-indexed playback (lead 0): the row that belongs to now, interpolated
            want = q0[R_ELBOW] + 0.5 * (rig.clock() - 1000.4)
            ts.append((rig.sent()[R_ELBOW], want))
    after_xfade = [a - b for a, b in ts[6:20]] + [a - b for a, b in ts[26:40]] + [a - b for a, b in ts[46:60]]
    assert max(abs(x) for x in after_xfade) < 1e-6
    # a late chunk (newer than the one playing, but all rows in the past) is dropped `expired`; the old one plays on
    t_applied = rig.ch.sess.chunk.t0
    rep = rig.arm({**base(), "chunk": chunk(4, t_applied + 0.01, [q0] * 3)})
    assert rep["ok"] and rep["data"]["dropped"] == "expired"
    rep = rig.arm({**base(), "chunk": chunk(3, rig.clock(), [q0] * 40)})           # seq not above the applied one
    assert rep["ok"] and rep["data"]["dropped"] == "out_of_order"
    rep = rig.arm({**base(), "chunk": chunk(5, t_applied - 0.1, [q0] * 40)})       # t0 older than the applied one
    assert rep["ok"] and rep["data"]["dropped"] == "out_of_order"
    # stale session: another session_id, or a lower generation / epoch, on the owning stream
    for extra in ({"session_id": "old"}, {"generation": 0}, {"control_epoch": 0}):
        rep = rig.arm({**base(), **extra, "chunk": chunk(9, rig.clock(), [q0] * 40)})
        assert rep["error"] == "stale_session", extra
    # NaN: rejected bad_chunk, nothing changes, counted invalid
    bad = chunk(9, rig.clock(), [q0] * 40)
    bad["upper_body"][3][5] = float("nan")
    with pytest.raises(ArmError) as e:
        rig.arm({**base(), "chunk": bad})
    assert e.value.reason == "bad_chunk"
    # wrong order field / shapes
    for mut in ({"order": "isaac"}, {"dt": 0.5}, {"upper_body": [q0[:16]] * 40}, {"left_hand": [[0.0] * 7] * 39}):
        with pytest.raises(ArmError):
            rig.arm({**base(), "chunk": {**chunk(9, rig.clock(), [q0] * 40), **mut}})
    assert rig.arm({"stream": "s1", "upper_body": q0})["error"] == "mode_mismatch"
    assert rig.arm({**base(), "keepalive": True})["ok"]
    rig.run(0.3)
    prog = [e for e in rig.events if e["id"] == "arm-s1" and e["state"] == "progress"]
    assert prog, "no arm.progress"
    p = prog[-1]["data"]
    for f in ("kind", "session_id", "stream", "chunk_seq", "k", "T", "stall_s", "clamped_frac", "clamped_frac_total",
              "slew_frac", "latency_ms", "inference_ms", "lead_s", "cross_fades", "chunks"):
        assert f in p, f
    assert p["kind"] == "chunk" and p["chunk_seq"] == 3 and p["cross_fades"] == 2 and p["inference_ms"] == 150.0
    assert p["chunks"]["dropped"] == {"expired": 1, "out_of_order": 2, "invalid": 5, "stale_session": 3, "halted": 0}
    t_prog = [e["t"] for e in prog]
    assert all(0.19 < b - a < 0.25 for a, b in zip(t_prog, t_prog[1:]))       # 5 Hz
    # cancel (the runtime's end: stand with nothing in hand); later chunks of the session are stale
    rep = rig.arm({**base(), "end": True, "hold_on_end": "stand", "reason": "cancelled"})
    assert rep["ok"] and rep["data"]["hold"] == "stand"
    ev = rig.terminal("arm-s1")
    assert ev["state"] == "succeeded" and ev["data"]["ended_by"] == "client" and ev["data"]["reason"] == "cancelled"
    for f in ("session_id", "chunks", "clamped_frac_total", "stall_s_max", "duration_s", "hold"):
        assert f in ev["data"], f
    assert rig.arm({**base(), "chunk": chunk(10, rig.clock(), [q0] * 40)})["error"] == "stale_session"


def test_chunk_lead_cross_fade_stall_and_clamp():
    rig = Rig({"arm_servo_ki": 0.0, "arm_max_vel": 50.0})
    q0 = rig.ch.reference_mj17()
    t0 = rig.clock()
    rows = _ramp_rows(q0, R_ELBOW, 0.5)
    rig.arm({**base(), "chunk": chunk(1, t0, rows)})                  # lead_s default 0.15 (G0)
    rig.run(0.2)
    # lead: the row 0.15 s ahead of now is played
    assert rig.sent()[R_ELBOW] == pytest.approx(q0[R_ELBOW] + 0.5 * (rig.clock() - t0 + 0.15), abs=1e-6)
    # a chunk that disagrees by 0.1 rad: the 5-tick cross-fade spreads the jump
    before = rig.sent()[R_ELBOW]
    rows2 = [[v + (0.1 if k == R_ELBOW else 0.0) for k, v in enumerate(r)]
             for r in _ramp_rows(q0, R_ELBOW, 0.5, t_off=0.2)]
    rig.arm({**base(), "chunk": chunk(2, t0 + 0.2, [r for r in rows2[:30]])})
    vals = [before]
    for _ in range(8):
        rig.run(DT)
        vals.append(rig.sent()[R_ELBOW])
    steps = [b - a for a, b in zip(vals, vals[1:])]
    assert max(steps) < 0.1 / 5 + 0.5 * DT + 1e-6 and sum(steps) > 0.1
    # the chunk runs out: the last row is held and stall_s grows
    rig.run(0.8)
    snap = rig.ch.snapshot()["session"]
    assert snap["k"] == 29 and snap["stall_s"] > 0.2
    last = rig.sent()[R_ELBOW]
    rig.run(0.2)
    assert rig.sent()[R_ELBOW] == pytest.approx(last, abs=1e-9)
    # out-of-limit rows are clamped (URDF - 0.02 rad) and counted: 1 of 28 values
    far = [list(q0) for _ in range(40)]
    for r in far:
        r[R_ELBOW] = 5.0
    rig.arm({**base(), "keepalive": True})
    rig.arm({**base(), "chunk": chunk(3, rig.clock(), far)})
    rig.run(0.5)
    assert rig.sent()[R_ELBOW] == pytest.approx(jm.JOINT_LIMITS["right_elbow_joint"][1] - 0.02, abs=1e-9)
    prog = [e["data"] for e in rig.events if e["state"] == "progress"]
    assert prog[-1]["clamped_frac"] == pytest.approx(1 / 28, abs=1e-3)


def test_chunk_preempt_takeover_and_hold_on_end_target():
    rig = Rig({"arm_servo_ki": 0.0})
    q0 = rig.ch.reference_mj17()
    rows = _ramp_rows(q0, L_SHP, -0.4)
    rig.arm({**base("g1"), "lead_s": 0.0, "chunk": chunk(1, rig.clock(), rows, hands=0.7)}, op_id="g1")
    rig.run(0.5)
    rep = rig.arm({**base("g2"), "chunk": chunk(1, rig.clock(), rows)})
    assert rep["error"] == "arm_busy"
    rep = rig.arm({**base("g1"), "end": True, "hold_on_end": "target"})
    assert rep["data"]["hold"] == "target"
    last = rig.ch.hold.pose
    assert rig.ch.hold.is_carry() and rig.ch.snapshot()["carry"]["engaged"]
    rig.run(1.0)
    assert rig.sent()[L_SHP] == pytest.approx(last[L_SHP], abs=1e-9)          # CarryLock: the last row held
    assert rig.mux.upper[2] == pytest.approx(jm.hand_closure("left", 0.7))
    # a new stream takes the hold over without preempt, continuing from the pose being sent (rule 3)
    held = rig.sent()
    assert rig.arm({**base("g2")}, op_id="g2")["state"] == "accepted"
    rig.run(DT)
    assert rig.sent() == pytest.approx(held, abs=1e-6)
    assert rig.arm({**base("g3"), "preempt": True}, op_id="g3")["state"] == "accepted"
    rig.ch.flush()
    ev = rig.terminal("g2")
    assert ev["state"] == "canceled" and ev["data"]["ended_by"] == "preempted"
    assert rig.arm({**base("g2"), "chunk": chunk(2, rig.clock(), rows)})["error"] == "stale_session"


def test_fault_drops_the_override_and_fails_the_session():
    rig = Rig()
    q0 = rig.ch.reference_mj17()
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="f")
    rig.run(0.1)
    rig.ch.abort("fallen")
    ev = rig.terminal("f")
    assert ev["state"] == "failed" and ev["data"]["ended_by"] == "fault" and rig.mux.upper is None


def test_latch_with_servo_does_not_step_the_wire_and_settles_on_the_halt_point():
    """With the servo on and SONIC leaving an unconverged error (bias 0.15 rad on the elbow, a wrist that barely
    follows), holding the measured pose must not step the wire at the latch; the arm ends where it was measured."""
    rig = Rig({"arm_max_vel": 50.0})
    rig.plant.bias[R_ELBOW] = -0.15
    k_wp = jm.UPPER_BODY_MUJOCO_JOINTS.index("right_wrist_pitch_joint")
    rig.plant.bias[k_wp] = -0.5                                  # far beyond servo_max: never converges
    q0 = rig.ch.reference_mj17()
    rows = _ramp_rows(q0, R_ELBOW, 0.3)
    for r in rows:
        r[k_wp] = 0.4
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), rows)})
    rig.run(0.5)
    rig.arm({**base(), "chunk": chunk(2, rig.clock(), _ramp_rows(q0, R_ELBOW, 0.3, t_off=0.5))})
    rig.run(0.3)
    before = rig.sent()
    q_halt = rig.q()
    rig.ch.latch(4)
    steps = []
    prev = before
    for _ in range(150):
        rig.run(DT)
        cur = rig.sent()
        steps.append(max(abs(a - b) for a, b in zip(cur[3:], prev[3:])))
        prev = cur
    assert max(steps[:5]) < 0.01, steps[:5]                      # no step at the latch
    assert abs(rig.q()[R_ELBOW] - q_halt[R_ELBOW]) < 0.01          # settled on the halt point (it was moving)
    assert abs(rig.q()[k_wp] - q_halt[k_wp]) < 0.02                # the untracked wrist stays where it was


# ------------------------------------------------------------------------------------------------ wave 2 (body-fix)
def test_latch_never_waits_for_the_arm_lock():
    """B-D1: a handler or tick holding the arm lock (live: an arm_script IK, 198-233 ms) must not delay a halt."""
    import threading

    rig = Rig()
    q0 = rig.ch.reference_mj17()
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="c1")
    rig.run(0.1)
    held, release = threading.Event(), threading.Event()

    def hog():
        with rig.ch._lock:
            held.set()
            release.wait(2.0)

    th = threading.Thread(target=hog)
    th.start()
    assert held.wait(1.0)
    t0 = time.perf_counter()
    info = rig.ch.latch(3)
    ms = (time.perf_counter() - t0) * 1e3
    release.set()
    th.join()
    assert ms < 5.0 and info["latched"] and info["froze_wire"], (ms, info)
    assert rig.arm({**base(), "chunk": chunk(2, rig.clock(), [q0] * 40)})["error"] == "halted"
    rig.run(DT)
    assert rig.terminal("c1")["data"]["ended_by"] == "halt"


def test_a_halt_during_a_tick_never_commits_that_ticks_pose():
    """The lane latches while the control thread is inside a tick: the tick must not put its (newer) pose on the
    wire after the freeze; the next tick applies the latch."""
    rig = Rig({"arm_servo_ki": 0.0, "arm_max_vel": 50.0})
    ch = rig.ch
    q0 = ch.reference_mj17()
    rig.arm({**base(), "lead_s": 0.0, "chunk": chunk(1, rig.clock(), _ramp_rows(q0, R_ELBOW, 1.0))}, op_id="c1")
    rig.run(0.2)
    orig = ch._servo
    fired = []

    def servo_and_halt(*a):
        orig(*a)
        if not fired:
            fired.append(ch.latch(4))                       # "the lane", mid-tick

    ch._servo = servo_and_halt
    frozen = rig.sent()
    rig.clock.t += DT
    ch.tick(rig.clock.t)                                    # the tick that was running when the halt came
    assert fired and rig.sent() == pytest.approx(frozen, abs=1e-12)
    assert ch.stats["ticks_skipped_latch"] == 1 and rig.terminal("c1") is None
    rig.run(DT)                                             # the next tick applies it
    assert rig.terminal("c1")["data"]["ended_by"] == "halt" and ch.hold.kind == "latched"
    assert abs(rig.sent()[R_ELBOW] - frozen[R_ELBOW]) <= 50.0 * DT + 1e-9


def test_latch_leaves_free_arms_free_and_resume_gives_them_back():
    """B-D3: with no arm op, a halt adds no override (SONIC keeps swinging its own arms while walking), arm messages
    are still `halted` while latched, and after the resume the arms are exactly as free as before."""
    rig = Rig()
    ch = rig.ch
    assert rig.mux.upper is None
    info = ch.latch(7)
    assert info["latched"] is False and info["arms"] == "free" and not info["froze_wire"]
    rig.run(0.5)
    assert rig.mux.upper is None and rig.mux.log == [] and ch.state()["mode"] == "off"
    assert ch.state()["latched"] and ch.state()["arms"] == "free" and ch.latch_info["case"] == "off"
    assert rig.arm({"stream": "a", "upper_body": ch.reference_mj17()})["error"] == "halted"
    ch.unlatch(7)
    rig.run(0.5)
    assert rig.mux.upper is None and ch.state() == {**ch.state(), "mode": "off", "latched": False, "arms": "free"}
    # the next owner starts normally after the resume
    assert rig.arm({"stream": "b", "upper_body": ch.reference_mj17(), "control_epoch": 8}, op_id="b")["ok"]


def test_repeated_halts_keep_carrylock_and_the_hand_targets():
    """B-D2: a CarryLock hold (a chunk session ended `target` with the left hand closed) through 10 halt/resume
    cycles: the hold, the hand command on the wire and CarryLock are unchanged, even though the measured hand stops
    short of its target (hand_gain 0.93: an object in the hand), which the old latch copied (0.9936 -> 0.9461 live)."""
    rig = Rig(plant_kw={"hand_gain": 0.93})
    ch = rig.ch
    q0 = ch.reference_mj17()
    rows = _ramp_rows(q0, L_SHP, -0.4)
    rig.arm({**base("g1"), "lead_s": 0.0, "chunk": chunk(1, rig.clock(), rows, hands=0.9)}, op_id="g1")
    rig.run(0.6)
    rig.arm({**base("g1"), "end": True, "hold_on_end": "target"})
    rig.run(0.5)
    assert ch.snapshot()["carry"]["engaged"]
    pose0, hands0 = list(ch.hold.pose), (list(rig.mux.upper[2]), list(rig.mux.upper[3]))
    assert jm.hand_closure_of("left", rig.dep.latest["left_hand_q"]) < 0.9 * 0.95   # the hand lags its target
    for n in range(10):
        ch.latch(10 + n)
        rig.run(0.2)
        snap = ch.snapshot()
        assert snap["mode"] == "latched" and snap["carry"]["engaged"] and ch.hold.kind == "target"
        assert (list(rig.mux.upper[2]), list(rig.mux.upper[3])) == hands0
        ch.unlatch(10 + n)
        rig.run(0.1)
    assert ch.hold.kind == "target" and ch.hold.pose == pose0 and ch.snapshot()["carry"]["engaged"]
    assert (list(rig.mux.upper[2]), list(rig.mux.upper[3])) == hands0
    assert ch.latch_info["case"] == "hold" and ch.latch_info["carry"] is True


def test_repeated_halts_of_sessions_never_ratchet_the_grip():
    """B-D2 with sessions: each cycle a new session (newer epoch) takes the latched pose over and is halted again;
    the hand command stays the session's target, never the (lagging) measured closure."""
    rig = Rig(plant_kw={"hand_gain": 0.93})
    ch = rig.ch
    q0 = ch.reference_mj17()
    target = jm.hand_closure("right", 0.95)
    for n in range(10):
        sid = f"s{n}"
        rep = rig.arm({**base(sid, epoch=n + 1), "chunk": chunk(1, rig.clock(), [q0] * 40, hands=target)})
        assert rep["ok"], rep
        rig.run(0.4)
        ch.latch(n + 1)
        rig.run(0.1)
        assert rig.mux.upper[3] == pytest.approx(target, abs=1e-9), n
        ch.unlatch(n + 1)
    assert jm.hand_closure_of("right", rig.mux.upper[3]) == pytest.approx(0.95, abs=1e-6)


def test_latch_keeps_the_sessions_waist():
    """B-low: a session sending SONIC's reference waist (the default) must not get the MEASURED waist at a halt
    (live: a 0.12-0.14 rad waist step at every arm/chunk halt); a session commanding the waist keeps its value."""
    rig = Rig({"arm_max_vel": 50.0})
    rig.plant.bias[0] = 0.13                                  # SONIC's waist yaw sits 0.13 rad off its reference
    ch = rig.ch
    q0 = ch.reference_mj17()
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="c1")
    rig.run(0.5)
    assert abs(rig.q()[0] - q0[0]) > 0.1                       # measured waist != reference
    w_before = rig.sent()[0:3]
    ch.latch(2)
    rig.run(0.5)
    assert rig.sent()[0:3] == pytest.approx(w_before, abs=1e-9)
    assert ch.hold.waist_mode == "ref" and ch.latch_info["waist"] == "ref"
    # waist "cmd" (a v0.5 client naming the waist): the value it sent is held
    ch.unlatch(2)
    rig.arm({"stream": "v", "upper_body": {"waist_yaw_joint": 0.2}, "control_epoch": 3}, op_id="v")
    rig.run(0.5)
    w_cmd = rig.sent()[0:3]
    assert w_cmd[0] == pytest.approx(0.2, abs=1e-6)
    ch.latch(4)
    rig.run(0.5)
    assert rig.sent()[0:3] == pytest.approx(w_cmd, abs=1e-9) and ch.hold.waist_mode == "cmd"


def test_measured_hold_on_end_keeps_the_waist_too():
    rig = Rig()
    rig.plant.bias[0] = 0.13
    q0 = rig.ch.reference_mj17()
    rig.arm({**base(), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="c1")
    rig.run(0.5)
    w = rig.sent()[0:3]
    rig.arm({**base(), "end": True, "hold_on_end": "measured"})
    rig.run(0.5)
    assert rig.ch.hold.kind == "measured" and rig.ch.hold.waist_mode == "ref"
    assert rig.sent()[0:3] == pytest.approx(w, abs=1e-9)


def test_latch_pauses_a_blend_and_resume_finishes_it():
    rig = Rig({"arm_blend_s": 1.0, "arm_servo_ki": 0.0})
    ch = rig.ch
    q0 = ch.reference_mj17()
    far = list(q0)
    far[R_ELBOW] += 0.5
    rig.arm({"stream": "a", "upper_body": far}, op_id="a")
    rig.run(0.6)
    assert ch.end("stop")
    rig.run(0.3)                                                # mid-blend
    ch.latch(5)
    rig.run(DT)
    frozen = rig.sent()
    rig.run(1.5)                                                # longer than the blend: it must not finish
    assert rig.sent() == pytest.approx(frozen, abs=1e-12) and ch.state()["mode"] == "latched"
    assert rig.terminal("a")["data"]["ended_by"] == "halt" and ch.latch_info["case"] == "blend"
    ch.unlatch(5)
    rig.run(1.2)
    assert rig.mux.upper is None and ch.state()["mode"] == "off"   # free again, as before the halt


def test_v05_end_during_the_watchdog_hold_succeeds():
    """B-low: the client's `end` while its stream sits in the watchdog hold is a client end (succeeded), not
    client_silent."""
    rig = Rig({"arm_hold_s": 1.0, "arm_blend_s": 0.3})
    q0 = rig.ch.reference_mj17()
    rig.arm({"stream": "A", "upper_body": q0}, op_id="t1")
    rig.run(0.5)                                                # 0.3 s watchdog -> hold
    assert rig.ch.hold is not None and rig.ch.hold.kind == "watchdog"
    assert rig.arm({"stream": "A", "end": True})["ok"]
    rig.run(0.5)
    ev = rig.terminal("t1")
    assert ev["state"] == "succeeded" and ev["data"]["ended_by"] == "client" and ev["data"]["reason"] is None


def test_chunk_session_fence_covers_end_release_and_keepalive():
    """B-low: every message acting on a chunk session's stream is fenced, not only chunk-mode ones."""
    rig = Rig()
    ch = rig.ch
    q0 = ch.reference_mj17()
    rig.arm({**base("g", gen=2, epoch=5), "chunk": chunk(1, rig.clock(), [q0] * 40)}, op_id="g")
    rig.run(0.1)
    # without the session fields: bad_args; with another / older session: stale_session; nothing changes
    for msg in ({"stream": "g", "end": True}, {"stream": "g", "release": True}, {"stream": "g", "keepalive": True}):
        with pytest.raises(ArmError) as e:
            rig.arm(msg)
        assert e.value.reason == "bad_args"
    for extra in ({"session_id": "old"}, {"generation": 1}, {"control_epoch": 4}):
        for act in ({"end": True}, {"release": True}, {"keepalive": True}):
            m = {"stream": "g", "session_id": "g", "generation": 2, "control_epoch": 5, **extra, **act}
            assert rig.arm(m)["error"] == "stale_session", m
    assert ch.state()["mode"] == "chunk"
    # its own end (no mode field) ends it into a target hold; a stale release of that hold is refused, its own works
    assert rig.arm({"stream": "g", "session_id": "g", "generation": 2, "control_epoch": 5, "end": True,
                    "hold_on_end": "target"})["ok"]
    assert ch.hold.kind == "target" and ch.hold.fence["session_id"] == "g"
    assert rig.arm({"stream": "g", "session_id": "g0", "release": True})["error"] == "stale_session"
    assert rig.arm({"stream": "g", "session_id": "g", "keepalive": True, "mode": "chunk", "generation": 2,
                    "control_epoch": 5})["error"] == "stale_session"      # the session is over
    assert rig.arm({"stream": "g", "session_id": "g", "release": True})["data"]["released"] == "target"
    rig.run(2.0)
    assert rig.mux.upper is None


def test_stale_t_wall_says_why():
    rig = Rig()
    rep = rig.arm({"stream": "A", "upper_body": rig.ch.reference_mj17(), "t_wall": time.time() - 5.0})
    assert rep["error"] == "stale_command" and rep["data"]["why"] == "t_wall" and rep["data"]["age_s"] > 4.0
