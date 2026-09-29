"""The page's GR00T strip (ui/groot_strip.py) from the events services/executors/groot_arms.py emits (arm_chunk.md
v0.1 names) and the manipulate result row. No policy, body or simulator."""

from __future__ import annotations

from ui.groot_strip import KEEP_S, GrootStrip

SKILL = "groot.pick.alarm_clock.arena_static.v0"


def started(eid="man-000009", t=10.0, skill=SKILL):
    return {"type": "started", "tool": "manipulate", "action": "pick", "execution_id": eid, "t": t,
            "args": {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1", "skill_id": skill}}


def test_a_groot_session_from_start_to_honest_outcome():
    g = GrootStrip()
    g.trace([started()])
    g.events([
        {"type": "policy.health", "ok": True, "state": "ok", "detail": "ping 12 ms", "endpoint": "tcp://127.0.0.1:5550",
         "checkpoint": "nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace", "executor": "groot_arms"},
        {"type": "manip.phase", "execution_id": "man-000009", "phase": "open_hands", "skill": SKILL,
         "executor": "groot_arms", "label": "experimental"},
        {"type": "groot.inference", "session": "man-000009", "execution_id": "man-000009", "ok": True,
         "latency_ms": 150.0, "chunk_idx": 1},
        {"type": "groot.inference", "session": "man-000009", "ok": True, "latency_ms": 170.0, "chunk_idx": 2},
        {"type": "groot.inference", "session": "man-000009", "ok": True, "latency_ms": 400.0, "dropped": "expired"},
        {"type": "groot.inference", "session": "man-000009", "ok": False, "error": "PolicyTimeout: 1.5 s"},
        {"type": "arm.progress", "session": "man-000009", "chunk_idx": 2, "k": 17, "stall_s": 0.0,
         "clamped_frac": 0.04, "latency_ms": 31.0, "cross_fades": 1, "inferences": 3,
         "dropped": {"expired": 1}, "source": "body"},
        {"type": "arm.progress", "session": "other-session", "chunk_idx": 99},            # not this session
    ])
    snap = g.snapshot(12.0)
    assert snap["session"] == "man-000009" and snap["skill"] == SKILL and snap["executor"] == "groot_arms"
    assert snap["label"] == "experimental" and snap["running"] and snap["phase"] == "open_hands"
    assert snap["latency_ms"] == {"p50": 170.0, "last": 400.0, "n": 3}
    assert snap["chunk_idx"] == 2 and snap["k"] == 17 and snap["chunk_age_ms"] == 31.0      # body: the chunk's age
    assert snap["clamped_frac"] == 0.04 and snap["stall"] is False and snap["inferences"] == 3
    assert snap["dropped"] == {"expired": 1} and snap["errors"] == 1
    assert snap["policy"]["ok"] is True and snap["policy"]["checkpoint"].endswith("Static-PickNPlace")
    g.events([{"type": "groot.stall", "session": "man-000009", "stall_s": 2.1}])
    assert g.snapshot(13.0)["stall"] is True
    g.trace([{"type": "result", "tool": "manipulate", "action": "pick", "execution_id": "man-000009", "t": 30.0,
              "status": "failed", "late": False, "summary": "pick failed: grasp_missed",
              "data": {"executor": "groot_arms", "skill": SKILL, "reason": "grasp_missed", "holding": False,
                       "label": "experimental", "inferences": 41, "chunks_dropped": {"expired": 2},
                       "clamped_frac": 0.06, "stall_s_max": 0.4, "latency_ms": {"p50": 161.0, "p95": 190.0, "n": 41},
                       "gt": {"predicate": "gt_lifted", "lift_max_m": 0.012}}}])
    snap = g.snapshot(31.0)
    assert not snap["running"] and snap["status"] == "failed" and snap["inferences"] == 41
    out = snap["outcome"]
    assert out["reason"] == "grasp_missed" and out["holding"] is False
    assert out["text"] == "failed · grasp_missed · in hand (GT): no · lifted at most 1 cm"
    assert g.snapshot(30.0 + KEEP_S + 1) is None                       # finished sessions leave the strip


def test_client_progress_latency_is_not_the_chunk_age_and_the_result_fills_p50():
    g = GrootStrip()
    g.trace([started()])
    g.events([{"type": "arm.progress", "session": "man-000009", "chunk_idx": 1, "latency_ms": 150.0,
               "source": "client", "inferences": 1}])
    snap = g.snapshot(11.0)
    assert snap["chunk_age_ms"] is None and snap["latency_ms"]["p50"] is None and snap["progress_source"] == "client"
    g.trace([{"type": "result", "tool": "manipulate", "execution_id": "man-000009", "t": 20.0, "status": "cancelled",
              "data": {"executor": "groot_arms", "skill": SKILL, "latency_ms": {"p50": 155.0}, "holding": False}}])
    assert g.snapshot(21.0)["latency_ms"]["p50"] == 155.0


def test_other_executors_never_open_a_session_and_health_falls_back_to_capabilities():
    g = GrootStrip()
    g.trace([started(skill="sonic.script.pick.v0"),
             {"type": "result", "tool": "manipulate", "execution_id": "man-000001", "t": 5.0, "status": "succeeded",
              "data": {"executor": "sonic_arm_script", "skill": "sonic.script.pick.v0"}}])
    assert g.snapshot(6.0) is None
    g.trace([{"type": "result", "tool": "manipulate", "execution_id": "man-000002", "t": 7.0, "status": "rejected",
              "data": {"executor": "groot_arms", "skill": SKILL, "reason": "policy unavailable: down"}}])
    snap = g.snapshot(8.0, {"manipulation": {"ok": False, "state": "down", "detail": "PolicyServer 5550 down"}})
    assert snap["session"] == "man-000002" and snap["status"] == "rejected"
    assert snap["policy"] == {"ok": False, "state": "down", "detail": "PolicyServer 5550 down", "source": "manipulation"}
