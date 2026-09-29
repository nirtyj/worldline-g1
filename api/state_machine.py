"""The tool-state machine (PLAN 5.8; doc 46 completed).

The state is DERIVED from active executions plus ``paused``, ``reconciling``,
``cancelling`` and the body mode. It is never set by hand; every change is a
``tool_state`` context event.

    IDLE --speak|list_locations|recall|check_reachability--> IDLE
    IDLE --navigate--> NAVIGATING --succeeded--> OBSERVING (arrival scan) --> IDLE
    IDLE --manipulate--> MANIPULATING --done--> OBSERVING (verify) --> IDLE
    IDLE --wait_and_observe--> WAITING --changed|timed_out|cancelled--> IDLE
    NAVIGATING|MANIPULATING --correction--> CANCELLING --> RECONCILING --> IDLE
    ANY --stop--> STOPPED --resume--> IDLE                                      [WL]
    NAVIGATING|MANIPULATING --fell--> FAULT --recovery--> RECONCILING --> IDLE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal


class ToolState(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    MANIPULATING = "MANIPULATING"
    OBSERVING = "OBSERVING"
    WAITING = "WAITING"
    CANCELLING = "CANCELLING"
    RECONCILING = "RECONCILING"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


FAULT_BODY_MODES = ("FAULT", "ESTOP")
Admit = Literal["accept", "reject", "defer", "wake"]
# "wake": the call is not admitted while WAITING; the wake source ends the wait first.


@dataclass(frozen=True)
class Admission:
    verdict: Admit
    note: str = ""             # e.g. "glance only", "returns base_moving", "body busy", "paused"

    @property
    def ok(self) -> bool:
        return self.verdict == "accept"


def derive_state(active: Iterable[Any], *, paused: bool = False, reconciling: bool = False,
                 body_mode: str | None = None) -> ToolState:
    """Derive the state from active executions (objects with tool_name/status/action/source).

    Order matters: FAULT > STOPPED > CANCELLING > RECONCILING > NAVIGATING/MANIPULATING >
    OBSERVING > WAITING > IDLE."""
    if body_mode in FAULT_BODY_MODES:
        return ToolState.FAULT
    if paused:
        return ToolState.STOPPED
    act = [e for e in active if getattr(e, "status", "running") in ("queued", "running", "cancelling")]
    body = [e for e in act if getattr(e, "tool_name", getattr(e, "tool", None)) in ("navigate", "manipulate")]
    if any(getattr(e, "status", "") == "cancelling" for e in body):
        return ToolState.CANCELLING
    if reconciling:
        return ToolState.RECONCILING
    for e in body:
        if getattr(e, "tool_name", getattr(e, "tool", None)) == "manipulate":
            return ToolState.MANIPULATING
    if body:
        return ToolState.NAVIGATING
    if any(getattr(e, "tool_name", getattr(e, "tool", None)) in ("observe", "look") for e in act):
        return ToolState.OBSERVING
    if any(getattr(e, "tool_name", getattr(e, "tool", None)) == "wait_and_observe" for e in act):
        return ToolState.WAITING
    return ToolState.IDLE


_A, _R, _D, _W = "accept", "reject", "defer", "wake"
_INSTANT = ("speak", "list_locations", "recall")

# tool -> state -> (verdict, note). PLAN 5.8 table, exhaustively.
ADMISSION: dict[str, dict[ToolState, Admission]] = {}
for _t in _INSTANT:
    ADMISSION[_t] = {s: Admission(_A) for s in ToolState}
    ADMISSION[_t][ToolState.WAITING] = Admission(_W, "a wake ends the wait first")
ADMISSION["check_reachability"] = {
    ToolState.IDLE: Admission(_A),
    ToolState.NAVIGATING: Admission(_A, "returns reachable=false, base_moving"),
    ToolState.MANIPULATING: Admission(_R, "body busy"),
    ToolState.OBSERVING: Admission(_D),
    ToolState.WAITING: Admission(_W, "a wake ends the wait first"),
    ToolState.CANCELLING: Admission(_D),
    ToolState.RECONCILING: Admission(_D),
    ToolState.STOPPED: Admission(_A),
    ToolState.FAULT: Admission(_R, "fault"),
}
for _t in ("navigate", "manipulate"):
    ADMISSION[_t] = {
        ToolState.IDLE: Admission(_A),
        ToolState.NAVIGATING: Admission(_R, "body busy"),
        ToolState.MANIPULATING: Admission(_R, "body busy"),
        ToolState.OBSERVING: Admission(_D),
        ToolState.WAITING: Admission(_W, "a wake ends the wait first"),
        ToolState.CANCELLING: Admission(_D),
        ToolState.RECONCILING: Admission(_D),
        ToolState.STOPPED: Admission(_R, "paused"),
        ToolState.FAULT: Admission(_R, "fault"),
    }
ADMISSION["wait_and_observe"] = {
    ToolState.IDLE: Admission(_A),
    ToolState.NAVIGATING: Admission(_A, "glance only"),
    ToolState.MANIPULATING: Admission(_A, "glance only"),
    ToolState.OBSERVING: Admission(_D),
    ToolState.WAITING: Admission(_W, "a wake ends the wait first"),
    ToolState.CANCELLING: Admission(_A, "glance only"),
    ToolState.RECONCILING: Admission(_A, "glance only"),
    ToolState.STOPPED: Admission(_A, "glance only"),
    ToolState.FAULT: Admission(_A, "glance only"),
}


def admit(tool: str, state: ToolState | str) -> Admission:
    s = ToolState(state)
    table = ADMISSION.get(tool)
    if table is None:
        return Admission(_R, f"unknown tool {tool!r}")
    return table[s]


def wait_observe_mode(*, body_lease_held: bool, stationary: bool, at_keypoint: bool,
                      last_scan_age_s: float | None, min_scan_gap_s: float = 5.0) -> Literal["glance", "scan"]:
    """wait_and_observe's first step (PLAN 5.1): a scan only if no body lease is held, the robot is
    stationary at a keypoint, and the last scan there is older than 5 s; otherwise a glance.
    A glance never requests the lease, so wait_and_observe can never fail with body_busy."""
    if body_lease_held or not stationary or not at_keypoint:
        return "glance"
    if last_scan_age_s is not None and last_scan_age_s <= min_scan_gap_s:
        return "glance"
    return "scan"


# Terminal transitions: after these, the derived state returns to IDLE or STOPPED
# (checked by test_state_machine as a property).
TRANSITIONS: tuple[tuple[str, str, str], ...] = (
    ("IDLE", "navigate", "NAVIGATING"),
    ("NAVIGATING", "succeeded", "OBSERVING"),
    ("NAVIGATING", "failed|timed_out", "IDLE"),
    ("OBSERVING", "done", "IDLE"),
    ("IDLE", "manipulate", "MANIPULATING"),
    ("MANIPULATING", "done", "OBSERVING"),
    ("IDLE", "wait_and_observe", "WAITING"),
    ("WAITING", "changed|timed_out|cancelled", "IDLE"),
    ("NAVIGATING|MANIPULATING", "correction", "CANCELLING"),
    ("CANCELLING", "cancelled", "RECONCILING"),
    ("RECONCILING", "done", "IDLE"),
    ("ANY", "stop", "STOPPED"),
    ("STOPPED", "resume", "IDLE"),
    ("NAVIGATING|MANIPULATING", "fell", "FAULT"),
    ("FAULT", "recovered", "RECONCILING"),
)


__all__ = ["ToolState", "FAULT_BODY_MODES", "Admission", "ADMISSION", "admit", "derive_state",
           "wait_observe_mode", "TRANSITIONS"]
