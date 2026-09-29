"""The result envelope and the typed results (PLAN 5.3; doc 41 plus Worldline fields).

``ToolResult`` is the ONLY result shape the harness consumes. It is frozen:
``late`` and ``observation_id`` are set with ``dataclasses.replace`` before the
result is stored or emitted. ``data`` is ``asdict(typed result)`` plus extras.

Envelope mapping:
  speak queued                         -> succeeded, data.speech="queued"
  check_reachability that ran          -> succeeded (also for reachable=false)
  wait changed/unchanged/timed_out/... -> succeeded / succeeded(data.status="unchanged") / timed_out / cancelled
  speech dropped                       -> cancelled, reason "superseded (generation N)"
  THOR ABORTED, GR00T grasp failure    -> failed, reason kept
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:                                    # pragma: no cover
    from .execution import Execution

EnvelopeStatus = Literal["succeeded", "failed", "cancelled", "timed_out", "rejected"]
ENVELOPE_STATUSES: tuple[str, ...] = ("succeeded", "failed", "cancelled", "timed_out", "rejected")
ResultSource = Literal["brain", "harness", "persona"]

# Executors that are STEPPING STONES (PLAN 12.2): labelled [fallback] everywhere. "lite" is the
# pure-Python profile: everything physical there is a stand-in (PLAN 2.1, L0).
STEPPING_STONE_EXECUTORS: tuple[str, ...] = ("kinematic_nav", "kinematic_attach", "sonic_arm_script", "lite")
NAV_EXECUTORS: tuple[str, ...] = ("sonic_walk", "kinematic_nav", "lite")
# groot_arms: GR00T N1.7 arm/hand joint chunks streamed through the body `arm` op (owner decision 7, 2026-09-29;
# docs/groot_arms_design.md). It replaces the token route groot_sonic, which stays listed until M2b retires it.
MANIP_EXECUTORS: tuple[str, ...] = ("groot_arms", "groot_sonic", "sonic_arm_script", "kinematic_attach", "lite")


# The TARGET executors (PLAN 2.1): a result, and an eval pass, counts as a target result only when it came from
# one of these. Everything else (the stepping stones, "lite", an unknown or missing executor) is a fallback.
TARGET_EXECUTORS: tuple[str, ...] = ("sonic_walk", "groot_arms", "groot_sonic")


# The trace's `result` row (agent/harness.py Runtime._log_result; docs/api.md §3). Every row carries these fields,
# whatever produced it: a tool that ran (kind "tool"), a line of speech ("speech"), recall ("recall") or a rejection
# ("rejection", status "rejected"; execution_id is None only for a SCHEMA rejection). `skill` is kept as the old name
# of `tool`; rows also carry action, source, executor, summary and data.
RESULT_ROW_KINDS: tuple[str, ...] = ("tool", "speech", "recall", "rejection")
RESULT_ROW_FIELDS: tuple[str, ...] = ("kind", "execution_id", "tool", "status", "observation_id", "generation",
                                      "control_epoch", "t_start", "t_end", "late")


def is_fallback(executor: str | None) -> bool:
    return executor in STEPPING_STONE_EXECUTORS


def is_target(executor: str | None) -> bool:
    """True only for the target executors; the UI and the eval share this one definition of a target pass."""
    return executor in TARGET_EXECUTORS


@dataclass(frozen=True)
class ToolResult:
    tool: str
    execution_id: str | None        # None only for schema-stage rejections
    status: EnvelopeStatus
    summary: str                    # one line for the planner (api/summaries.py)
    data: dict[str, Any]
    generation: int
    control_epoch: int
    source: ResultSource = "brain"
    t_start: float = 0.0
    t_end: float = 0.0
    observation_id: str | None = None   # the latest glance, or the observation this result ran
    late: bool = False              # generation < current at completion; set only by the harness

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    @property
    def reason(self) -> str | None:
        r = self.data.get("reason")
        return str(r) if r is not None else None

    @property
    def executor(self) -> str | None:
        return self.data.get("executor")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ----------------------------------------------------------------------
# Typed results (their asdict() goes into ToolResult.data)
# ----------------------------------------------------------------------
@dataclass
class SpeakResult:                   # doc 4.1
    status: Literal["queued", "failed"]
    utterance_id: str
    played: float | None = None      # fraction played, when cut


@dataclass
class NamedLocation:                 # doc 5.1 + [WL]
    name: str
    distance_m: float | None
    type: Literal["surface", "room", "person", "start"]
    room: str | None = None
    desc: str | None = None
    height_m: float | None = None
    too_high: bool = False
    too_low: bool = False


@dataclass
class ListLocationsResult:
    locations: list[NamedLocation] = field(default_factory=list)


@dataclass
class NavigateResult:                # doc 6.1 + [WL]
    execution_id: str
    status: EnvelopeStatus
    location: str
    reason: str | None = None
    at: str | None = None
    between: list[str] | None = None
    blocked_edge: list[str] | None = None
    executor: Literal["sonic_walk", "kinematic_nav", "lite"] = "sonic_walk"
    path_len_m: float = 0.0
    walked_m: float = 0.0
    duration_s: float = 0.0
    replans: int = 0
    settle_s: float = 0.0
    observation_id: str | None = None
    kind: Literal["keypoint", "reposition"] = "keypoint"
    final_err_m: float = 0.0
    detail: str | None = None       # a human-readable why for a failure (e.g. "stance 0.52 m away")


@dataclass
class ReachabilityResult:            # doc 8.1 + [WL]
    reachable: bool
    visible: bool
    preferred_arm: Literal["left", "right", "either", "none"]
    reason: str | None = None
    object_type: str = ""
    object_id: str | None = None
    suggest_location: str | None = None     # a keypoint (too_far) or "reach_stance" (needs_reposition)
    stance: dict[str, Any] | None = None    # {x, y, yaw} world (REP-103) + {dx, dy, dyaw} from here; a far stance
                                            # (the outline-wide search) + {walk_m, side, reason, via: {x, y}}
    distance_m: float | None = None
    height_m: float | None = None
    skill_id: str | None = None
    at: str | None = None                   # the keypoint it was judged from
    detail: str | None = None               # a human-readable why (never a hidden position)


@dataclass
class ManipulationResult:            # doc 10-12 + [WL]
    execution_id: str
    status: EnvelopeStatus
    skill: str
    object_type: str
    reason: str | None = None
    action: Literal["pick", "place"] = "pick"
    arm: str | None = None
    object_id: str | None = None
    target: str | None = None
    holding: bool | None = None
    executor: Literal["groot_arms", "groot_sonic", "sonic_arm_script", "kinematic_attach", "lite"] = "groot_sonic"
    phase: str | None = None
    inferences: int = 0
    chunks_dropped: dict[str, int] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    base_shift_m: float = 0.0       # base motion caused by whole-body GR00T tokens (PLAN 1.3 #25)
    surface: str | None = None      # place: where it was put


@dataclass
class WaitResult:                    # doc 18.1 + "unchanged"
    status: Literal["changed", "unchanged", "timed_out", "cancelled"]
    observation_id: str
    summary: str | None = None
    changes: list[str] = field(default_factory=list)


@dataclass
class RecallResult:                  # [WL]
    answer: str


WAIT_TO_ENVELOPE: dict[str, EnvelopeStatus] = {"changed": "succeeded", "unchanged": "succeeded",
                                               "timed_out": "timed_out", "cancelled": "cancelled"}

# The old Worldline/THOR statuses -> envelope statuses (one mechanical rename, PLAN 1.3 #15).
LEGACY_STATUS: dict[str, str] = {"SUCCEEDED": "succeeded", "ABORTED": "failed", "FAILED": "failed",
                                 "CANCELED": "cancelled", "CANCELLED": "cancelled", "TIMEOUT": "timed_out",
                                 "REJECTED": "rejected", "DROPPED": "dropped"}


def envelope_status(status: str) -> str:
    """Map any status spelling (legacy uppercase, exec 'dropped', wait statuses) to lowercase."""
    s = LEGACY_STATUS.get(status, status)
    if s in WAIT_TO_ENVELOPE:
        return WAIT_TO_ENVELOPE[s]
    if s == "dropped":
        return "cancelled"
    return s


def typed_data(obj: Any, **extra: Any) -> dict[str, Any]:
    d = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) and not isinstance(obj, type) else dict(obj or {})
    d.update(extra)
    return d


def finish(execution: "Execution", status: str, data: dict[str, Any] | Any = None, *,
           summary: str | None = None, observation_id: str | None = None, t_end: float | None = None,
           executor: str | None = None) -> ToolResult:
    """Build the frozen envelope for an execution. The summary is generated if not given.

    ``data`` may be a typed result dataclass or a dict. ``status`` may be a typed-result
    status (wait's ``changed``), a legacy status or an envelope status."""
    from .summaries import summarize
    raw = status
    env = envelope_status(status)
    d = typed_data(data)
    if executor is not None and "executor" not in d:
        d["executor"] = executor
    if execution.tool_name == "wait_and_observe" and raw in WAIT_TO_ENVELOPE and "status" not in d:
        d["status"] = raw
    if raw == "dropped" and "reason" not in d:
        d["reason"] = f"superseded (generation {execution.generation})"
    if summary is None:
        summary = summarize(execution.tool_name, env, d, action=execution.action, args=execution.args)
    t0 = execution.t_start
    return ToolResult(tool=execution.tool_name, execution_id=execution.execution_id, status=env,  # type: ignore[arg-type]
                      summary=summary, data=d, generation=execution.generation,
                      control_epoch=execution.control_epoch, source=execution.source,
                      t_start=t0, t_end=t0 if t_end is None else t_end,
                      observation_id=observation_id if observation_id is not None else d.get("observation_id"))


def rejected(tool: str, args: dict[str, Any] | None, *, stage: str, code: str, message: str,
             generation: int, control_epoch: int, source: ResultSource = "brain", t: float = 0.0,
             execution_id: str | None = None, observation_id: str | None = None) -> ToolResult:
    """A rejection is itself a tool result (doc 21). ``execution_id`` is None only for SCHEMA."""
    from .summaries import rejection_summary
    return ToolResult(tool=tool, execution_id=execution_id, status="rejected",
                      summary=rejection_summary(stage, message),
                      data={"stage": stage, "code": code, "reason": message, "args": dict(args or {})},
                      generation=generation, control_epoch=control_epoch, source=source,
                      t_start=t, t_end=t, observation_id=observation_id)


# ----------------------------------------------------------------------
# JSON Schema of the envelope (contract tests validate every result against it)
# ----------------------------------------------------------------------
def envelope_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ToolResult",
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "execution_id": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": list(ENVELOPE_STATUSES)},
            "summary": {"type": "string"},
            "data": {"type": "object"},
            "generation": {"type": "integer"},
            "control_epoch": {"type": "integer"},
            "source": {"type": "string", "enum": ["brain", "harness", "persona"]},
            "t_start": {"type": "number"},
            "t_end": {"type": "number"},
            "observation_id": {"type": ["string", "null"]},
            "late": {"type": "boolean"},
        },
        "required": ["tool", "execution_id", "status", "summary", "data", "generation", "control_epoch",
                     "source", "t_start", "t_end", "observation_id", "late"],
        "additionalProperties": False,
    }


# Required data keys per (tool, status) for succeeded results; checked by validate_envelope.
REQUIRED_DATA: dict[str, tuple[str, ...]] = {
    "navigate": ("location", "executor"),
    "check_reachability": ("reachable", "visible", "preferred_arm", "object_type"),
    "manipulate": ("skill", "object_type", "action", "executor"),
    "wait_and_observe": ("status", "observation_id"),
    "list_locations": ("locations",),
    "recall": ("answer",),
    "observe": ("hands", "views", "mode", "observation_id"),
}


def validate_envelope(d: Any) -> list[str]:
    """Problems with a ToolResult (or its dict), as messages. Stdlib stand-in for jsonschema."""
    if isinstance(d, ToolResult):
        d = d.to_dict()
    if not isinstance(d, dict):
        return ["result must be an object"]
    schema = envelope_schema()
    problems: list[str] = []
    for k in schema["required"]:
        if k not in d:
            problems.append(f"missing {k}")
    for k in d:
        if k not in schema["properties"]:
            problems.append(f"unexpected {k}")
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict,
             "null": type(None)}
    for k, prop in schema["properties"].items():
        if k not in d:
            continue
        allowed = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
        v = d[k]
        if not any(isinstance(v, types[t]) and not (t in ("integer", "number") and isinstance(v, bool))
                   for t in allowed):
            problems.append(f"{k} has the wrong type")
        if "enum" in prop and v not in prop["enum"]:
            problems.append(f"{k}={v!r} not in {prop['enum']}")
    if d.get("status") == "rejected":
        for k in ("stage", "code"):
            if k not in (d.get("data") or {}):
                problems.append(f"rejected result without data.{k}")
    elif d.get("status") == "succeeded":
        for k in REQUIRED_DATA.get(str(d.get("tool")), ()):
            if k not in (d.get("data") or {}):
                problems.append(f"{d.get('tool')} result without data.{k}")
    if d.get("status") != "rejected" and d.get("execution_id") is None:
        problems.append("only schema-stage rejections may lack an execution_id")
    return problems


__all__ = ["EnvelopeStatus", "ENVELOPE_STATUSES", "ResultSource", "STEPPING_STONE_EXECUTORS",
           "RESULT_ROW_KINDS", "RESULT_ROW_FIELDS",
           "NAV_EXECUTORS", "MANIP_EXECUTORS", "TARGET_EXECUTORS", "is_fallback", "is_target", "ToolResult", "SpeakResult", "NamedLocation",
           "ListLocationsResult", "NavigateResult", "ReachabilityResult", "ManipulationResult", "WaitResult",
           "RecallResult", "WAIT_TO_ENVELOPE", "LEGACY_STATUS", "envelope_status", "typed_data", "finish",
           "rejected", "envelope_schema", "REQUIRED_DATA", "validate_envelope"]
