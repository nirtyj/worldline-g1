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
def test_latch_freezes_measured_pose_keeps_hands_and_rejects_stale_epochs():
    rig = Rig({"arm_servo_ki": 0.0, "arm_blend_s": 0.3})
    ch = rig.ch
    q0 = ch.reference_mj17()
    rows = _ramp_rows(q0, R_ELBOW, 0.8)
    half = jm.hand_closure("right", 0.5)
    assert rig.arm({**base(epoch=2), "right_hand": half, "left_hand": 0.0,
                    "chunk": chunk(1, rig.clock(), rows, hands=0.5)}, op_id="c1")["ok"]
    rig.run(0.4)
    meas_hands = {s: list(rig.dep.latest[f"{s}_hand_q"]) for s in ("left", "right")}
    q_meas = rig.q()
    sent_at = rig.sent()
    t0 = time.perf_counter()
    info = ch.latch(2, "halt")
    ms = (time.perf_counter() - t0) * 1e3
    assert info["latched"] and info["pose_source"] == "g1_debug.body_q" and info["ended_op"] == "c1"
    assert ms < 5.0 and info["latch_ms"] < 5.0
    assert rig.sent() == pytest.approx(sent_at, abs=1e-12)      # the wire froze at once: no new row after the ack
    assert ch.state()["mode"] == "latched"
    # the terminal event is queued (no I/O in latch) and published by the next tick
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
    # hands: the measured q at the latch, never opened, never a fist
    assert rig.mux.upper[2] == pytest.approx(meas_hands["left"]) and rig.mux.upper[3] == pytest.approx(meas_hands["right"])
    # rejected while latched, whatever the epoch; stale epochs rejected for good
    for ep in (1, 2, 3):
        assert rig.arm({**base("s9", epoch=ep), "chunk": chunk(1, rig.clock(), rows)})["error"] == "halted"
    assert rig.arm({"stream": "legacy", "upper_body": q0})["error"] == "halted"
    assert ch.end("stop") is False and ch.state()["mode"] == "latched"
    assert ch.handle_scan("sc", {}, (True, None, {}))["error"] == "halted"
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
    rig.ch.latch(5)
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
