"""api/state_machine.py: the derived tool state, the admission table exhaustively (PLAN 5.8), and
the wait_and_observe scan/glance choice."""

from __future__ import annotations

import itertools

import pytest

from api.execution import ExecutionManager
from api.state_machine import ADMISSION, ToolState, admit, derive_state, wait_observe_mode
from api.tools import TOOL_SPECS

S = ToolState


def ex(tool, status="running", **args):
    return ExecutionManager().create(tool, args, generation=1, control_epoch=0, status=status)


def test_every_tool_has_a_verdict_in_every_state():
    for tool in TOOL_SPECS:
        for state in ToolState:
            assert admit(tool, state).verdict in ("accept", "reject", "defer", "wake"), (tool, state)
    assert set(ADMISSION) == set(TOOL_SPECS)


@pytest.mark.parametrize("tool,state,verdict,note", [
    ("speak", S.NAVIGATING, "accept", ""), ("speak", S.WAITING, "wake", "a wake ends the wait first"),
    ("recall", S.STOPPED, "accept", ""), ("list_locations", S.FAULT, "accept", ""),
    ("check_reachability", S.NAVIGATING, "accept", "returns reachable=false, base_moving"),
    ("check_reachability", S.MANIPULATING, "reject", "body busy"),
    ("check_reachability", S.OBSERVING, "defer", ""), ("check_reachability", S.RECONCILING, "defer", ""),
    ("check_reachability", S.STOPPED, "accept", ""), ("check_reachability", S.FAULT, "reject", "fault"),
    ("navigate", S.IDLE, "accept", ""), ("navigate", S.NAVIGATING, "reject", "body busy"),
    ("navigate", S.MANIPULATING, "reject", "body busy"), ("navigate", S.OBSERVING, "defer", ""),
    ("navigate", S.STOPPED, "reject", "paused"), ("navigate", S.FAULT, "reject", "fault"),
    ("manipulate", S.CANCELLING, "defer", ""), ("manipulate", S.STOPPED, "reject", "paused"),
    ("wait_and_observe", S.IDLE, "accept", ""), ("wait_and_observe", S.MANIPULATING, "accept", "glance only"),
    ("wait_and_observe", S.NAVIGATING, "accept", "glance only"), ("wait_and_observe", S.OBSERVING, "defer", ""),
    ("wait_and_observe", S.FAULT, "accept", "glance only"), ("wait_and_observe", S.STOPPED, "accept", "glance only"),
])
def test_plan_table(tool, state, verdict, note):
    a = admit(tool, state)
    assert (a.verdict, a.note) == (verdict, note)


def test_derived_states():
    assert derive_state([]) == S.IDLE
    assert derive_state([ex("navigate", location="k")]) == S.NAVIGATING
    assert derive_state([ex("manipulate", action="pick")]) == S.MANIPULATING
    assert derive_state([ex("observe", mode="scan")]) == S.OBSERVING
    assert derive_state([ex("wait_and_observe")]) == S.WAITING
    assert derive_state([ex("navigate", status="cancelling", location="k")]) == S.CANCELLING
    assert derive_state([], reconciling=True) == S.RECONCILING
    assert derive_state([ex("navigate", location="k")], paused=True) == S.STOPPED
    assert derive_state([ex("navigate", location="k")], body_mode="FAULT") == S.FAULT
    assert derive_state([], body_mode="ESTOP", paused=True) == S.FAULT
    # a wait_and_observe during a manipulate: still MANIPULATING (the wait glances)
    assert derive_state([ex("manipulate", action="pick"), ex("wait_and_observe")]) == S.MANIPULATING
    assert derive_state([ex("navigate", location="k"), ex("check_reachability")]) == S.NAVIGATING


def test_terminal_executions_return_to_idle_or_stopped():
    """Property: once every execution is terminal, the state is IDLE, or STOPPED when paused."""
    tools = ["navigate", "manipulate", "observe", "wait_and_observe", "check_reachability", "speak"]
    terminal = ["succeeded", "failed", "cancelled", "timed_out", "rejected", "dropped"]
    for combo in itertools.product(tools, terminal):
        e = ex(combo[0], status=combo[1])
        assert derive_state([e]) == S.IDLE
        assert derive_state([e], paused=True) == S.STOPPED


def test_wait_observe_scans_only_without_a_lease():
    assert wait_observe_mode(body_lease_held=False, stationary=True, at_keypoint=True, last_scan_age_s=None) == "scan"
    assert wait_observe_mode(body_lease_held=True, stationary=True, at_keypoint=True, last_scan_age_s=None) == "glance"
    assert wait_observe_mode(body_lease_held=False, stationary=False, at_keypoint=True, last_scan_age_s=60) == "glance"
    assert wait_observe_mode(body_lease_held=False, stationary=True, at_keypoint=False, last_scan_age_s=60) == "glance"
    assert wait_observe_mode(body_lease_held=False, stationary=True, at_keypoint=True, last_scan_age_s=3) == "glance"
    assert wait_observe_mode(body_lease_held=False, stationary=True, at_keypoint=True, last_scan_age_s=6) == "scan"
