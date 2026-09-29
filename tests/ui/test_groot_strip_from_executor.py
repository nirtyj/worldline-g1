"""The page's GR00T strip (ui/groot_strip.py, owner ui) fed by what groot_arms (owner groot_rt) really emits, and by
the manipulate result rows the service (owner world) really builds (integration, M2b wave 1).

Two sources: a GrootArmExecutor session over the offline fakes (its event sink as the strip receives it) and the
result rows of test_groot_then_script's two cases (the GR00T result, and a fallback result after a failed GR00T
attempt). Nothing is hand-written in the shape the strip expects: the names come from the producers.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("groot.actions")

from tests.services.test_groot_arms import APPLE, Rig  # noqa: E402
from ui.groot_strip import GrootStrip  # noqa: E402


def _started(eid: str, skill: str = APPLE) -> dict:
    return {"type": "started", "tool": "manipulate", "action": "pick", "execution_id": eid, "t": 1.0,
            "args": {"action": "pick", "object_type": "apple", "object_id": "apple_1", "skill_id": skill}}


def test_the_strip_shows_a_real_executor_session():
    with Rig("lift", server={"close_after": 2}) as r:
        assert r.healthy().ok
        job, h = r.job("man-s1")
        out = asyncio.run(r.exe.run(job, h))
        assert out.status == "succeeded", out.detail
        g = GrootStrip()
        g.trace([_started("man-s1")])
        g.events([{"type": t, **f} for t, f in r.sink.events])
        snap = g.snapshot(2.0)
        assert snap["session"] == "man-s1" and snap["executor"] == "groot_arms" and snap["label"] == "experimental"
        assert snap["inferences"] == out.data["inferences"] and snap["latency_ms"]["n"] == out.data["inferences"]
        assert snap["phase"] in ("execute", "carry_lock", "verify", "enter", "view_check", "stance_check")
        assert snap["policy"]["ok"] is True and snap["policy"]["checkpoint"].startswith("nvidia/GN1x")
        assert snap["progress_source"] == "body" and snap["clamped_frac"] is not None
        # the result row as the service builds it (executor data merged, api/results envelope)
        row = {"type": "result", "tool": "manipulate", "execution_id": "man-s1", "status": "succeeded", "t": 3.0,
               "data": {**out.data, "executor": "groot_arms", "skill": APPLE, "holding": True, "reason": None}}
        g.trace([row])
        snap = g.snapshot(3.5)
        assert snap["running"] is False and snap["outcome"]["status"] == "succeeded"
        assert "in hand (GT): yes" in snap["outcome"]["text"] and snap["outcome"]["fallback_from"] is None


def test_the_strip_says_when_a_fallback_produced_the_result():
    g = GrootStrip()
    g.trace([_started("man-s2")])
    data = {"executor": "kinematic_attach", "skill": "bringup.attach.pick.v0", "holding": True, "reason": None,
            "inferences": 12, "attempts": [{"executor": "groot_arms", "status": "failed", "reason": "grasp_missed"},
                                           {"executor": "kinematic_attach", "status": "succeeded"}],
            "fallback_from": {"skill": APPLE, "executor": "groot_arms", "status": "failed", "reason": "grasp_missed"},
            "groot": {"label": "experimental", "latency_ms": {"p50": 150.5, "p95": 165.5, "n": 12},
                      "clamped_frac": 0.0, "gt": {"lift_max_m": 0.0}}}
    g.trace([{"type": "result", "tool": "manipulate", "execution_id": "man-s2", "status": "succeeded", "t": 9.0,
              "data": data}])
    snap = g.snapshot(9.5)
    o = snap["outcome"]
    assert o["fallback_from"]["reason"] == "grasp_missed" and snap["executor"] == "kinematic_attach"
    assert o["text"].startswith("groot_arms failed · grasp_missed -> fallback kinematic_attach (STEPPING STONE)")
    assert snap["label"] == "experimental" and snap["latency_ms"]["p50"] == 150.5 and snap["inferences"] == 12
