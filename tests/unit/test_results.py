"""api/results.py, reasons.py, summaries.py: the envelope, the mapping table, reason hints, summaries."""

from __future__ import annotations

import dataclasses

import pytest

from api.execution import ExecutionManager
from api.reasons import hint, is_known, lookup
from api.results import (ManipulationResult, NavigateResult, ReachabilityResult, ToolResult, envelope_status,
                         finish, is_fallback, rejected, validate_envelope)
from api.summaries import summarize


def ex(tool, args=None, **kw):
    return ExecutionManager().create(tool, args or {}, generation=kw.pop("generation", 3), control_epoch=1, **kw)


def test_tool_result_is_frozen_and_late_is_set_by_replace():
    r = finish(ex("navigate", {"location": "k"}), "succeeded", {"location": "k", "executor": "sonic_walk"},
               observation_id="obs-g1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.late = True                                        # type: ignore[misc]
    late = dataclasses.replace(r, late=True)
    assert late.late and not r.late and late.execution_id == r.execution_id


@pytest.mark.parametrize("raw,env", [
    ("SUCCEEDED", "succeeded"), ("ABORTED", "failed"), ("CANCELED", "cancelled"), ("TIMEOUT", "timed_out"),
    ("REJECTED", "rejected"), ("DROPPED", "cancelled"), ("dropped", "cancelled"),
    ("changed", "succeeded"), ("unchanged", "succeeded"), ("timed_out", "timed_out"), ("cancelled", "cancelled"),
])
def test_envelope_mapping(raw, env):
    assert envelope_status(raw) == env


def test_wait_unchanged_is_succeeded_with_data_status():
    r = finish(ex("wait_and_observe", {"timeout_s": 0}), "unchanged", {"observation_id": "obs-g4"})
    assert r.status == "succeeded" and r.data["status"] == "unchanged" and r.observation_id == "obs-g4"
    assert r.summary == "looked; nothing new"
    assert validate_envelope(r) == []


def test_dropped_speech_is_cancelled_superseded():
    e = ex("speak", {"text": "hi"}, generation=2)
    r = finish(e, "dropped", {})
    assert r.status == "cancelled" and r.data["reason"] == "superseded (generation 2)"


def test_reachability_false_is_still_succeeded():
    e = ex("check_reachability", {"object_type": "book"})
    r = finish(e, "succeeded", ReachabilityResult(False, False, "none", "too_far", object_type="book",
                                                  object_id="book_1", suggest_location="bedroom_bed_1b",
                                                  distance_m=0.9))
    assert r.status == "succeeded" and r.data["reachable"] is False
    assert r.summary == "book_1 not reachable from here: too_far (0.9 m); try bedroom_bed_1b"


def test_summary_examples_from_the_plan():
    e = ex("check_reachability", {"object_type": "alarm_clock"})
    ok = finish(e, "succeeded", ReachabilityResult(True, True, "right", None, object_type="alarm_clock",
                                                   object_id="alarm_clock_1", distance_m=0.41))
    assert ok.summary == "alarm_clock_1 visible, reachable with the right arm (0.4 m)"
    rp = finish(e, "succeeded", ReachabilityResult(False, True, "none", "needs_reposition", object_type="alarm_clock",
                                                   object_id="alarm_clock_1", distance_m=0.22))
    assert rp.summary.startswith("alarm_clock_1 visible but not reachable from this exact pose: needs_reposition")
    assert "navigate(location='reach_stance')" in rp.summary
    m = ex("manipulate", {"action": "pick", "object_type": "alarm_clock"}, action="pick")
    picked = finish(m, "succeeded", ManipulationResult(m.execution_id, "succeeded", "sonic.script.pick.v0",
                                                       "alarm_clock", None, arm="right", object_id="alarm_clock_1",
                                                       executor="sonic_arm_script", duration_s=9.1))
    assert picked.summary == ("picked alarm_clock_1 with the right hand [sonic.script.pick.v0, fallback], 9.1 s; "
                              "verifying")
    n = ex("navigate", {"location": "bedroom_dresser_1a"})
    arrived = finish(n, "succeeded", NavigateResult(n.execution_id, "succeeded", "bedroom_dresser_1a", None,
                                                    at="bedroom_dresser_1a", path_len_m=6.2, duration_s=17.0))
    assert arrived.summary == "arrived at bedroom_dresser_1a (6.2 m, 17 s, sonic_walk)"


def test_rejection_is_a_tool_result():
    r = rejected("manipulate", {"action": "pick"}, stage="state", code="no_reachability",
                 message="pick needs a successful check_reachability for alarm_clock right before it",
                 generation=4, control_epoch=2, execution_id="man-000009", observation_id="obs-g2")
    assert r.status == "rejected" and r.data["stage"] == "state" and r.data["code"] == "no_reachability"
    assert r.summary == ("rejected (state): pick needs a successful check_reachability for alarm_clock "
                         "right before it")
    assert validate_envelope(r) == []
    schema_stage = rejected("fly", {}, stage="schema", code="unknown_tool", message="unknown tool 'fly'",
                            generation=1, control_epoch=0)
    assert schema_stage.execution_id is None and validate_envelope(schema_stage) == []


def test_validate_envelope_catches_problems():
    good = finish(ex("recall", {"query": "mug"}), "succeeded", {"answer": "on the table"}, observation_id="obs-g1")
    assert validate_envelope(good) == []
    bad = good.to_dict()
    bad["status"] = "SUCCEEDED"
    del bad["late"]
    bad["extra"] = 1
    problems = validate_envelope(bad)
    assert any("status" in p for p in problems) and any("late" in p for p in problems)
    assert any("unexpected extra" in p for p in problems)
    no_data = finish(ex("navigate", {"location": "k"}), "succeeded", {})
    assert "navigate result without data.location" in validate_envelope(no_data)


def test_reason_codes_and_hints():
    for code in ("unknown_location", "no_path", "blocked", "halted", "fell", "stuck", "no_reach_stance",
                 "stance_not_reached", "base_moving", "not_seen_here", "too_low", "needs_reposition",
                 "out_of_workspace", "no_skill", "policy_unavailable", "grasp_missed", "object_dropped",
                 "not_in_ego_view", "no_room_in_reach", "policy_stall", "policy_out_of_bounds",
                 "controller_unavailable"):
        assert is_known(code) and hint(code), code
    assert lookup("inside_or_on_fridge_1").area == "reachability"
    assert "fridge_1" in hint("inside_or_on_fridge_1")
    assert hint("needs_reposition", "reachability") == "navigate(location='reach_stance'), then check again"
    assert hint("halted", "manipulation") and not hint("no_such_code")


def test_fallback_executors():
    for x in ("kinematic_nav", "kinematic_attach", "sonic_arm_script", "lite"):
        assert is_fallback(x)
    for x in ("sonic_walk", "groot_sonic", None):
        assert not is_fallback(x)


def test_summarize_rejected_and_unknown_tools():
    assert summarize("navigate", "rejected", {"stage": "enum", "reason": "unknown location 'x'"}) == \
        "rejected (enum): unknown location 'x'"
    assert summarize("teleport", "failed", {"reason": "internal_error"}).startswith("teleport failed: internal_error")


def test_every_result_can_carry_an_observation_id():
    r = ToolResult("speak", "spk-000001", "succeeded", "said it", {}, 1, 0)
    assert r.observation_id is None                        # the harness stamps the latest glance before storing
    assert dataclasses.replace(r, observation_id="obs-g9").observation_id == "obs-g9"
