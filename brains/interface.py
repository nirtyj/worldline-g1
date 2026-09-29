"""The contract between a runtime (the harness) and a brain (the model).

The runtime owns the robot, the clock, belief, rules, cancellation and
speech. The brain owns interpretation and choosing the next step. In the
playground the brain is brains/composite.py around the planner (agent/model.py),
which does both.

    kind = await brain.classify(utterance, ctx)    # one of KINDS
    call = await brain.next_action(ctx)            # a ToolCall

The runtime decides what a kind means (only a correction cancels anything,
"stop" never waits for the model), executes tool calls under its rules, and
records every action -- the brain's and its own, such as verification looks --
as an Execution (api/execution.py).

``ToolCall.tag`` is opaque bookkeeping: copy it into the Execution unchanged.
Model brains leave it None. ToolCall, the tools and the Execution object come
from api/ (the contract); this module re-exports them under their old names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from api.execution import Execution
from api.results import ToolResult
from api.tools import BODY_TOOLS as _BODY_TOOLS
from api.tools import SENSE_TOOLS as _SENSE_TOOLS
from api.tools import TOOL_SPECS, SchemaContext, ToolCall
from api.types import UNKNOWN

KINDS = (
    "request",      # a new task: "bring me the mug", "what's on the table?"
    "correction",   # changes the current task: "no, the alarm clock instead"
    "addition",     # adds a task without changing the current one: "also grab a napkin"
    "question",     # needs an answer, changes nothing: "how long will it take?"
    "stop",         # halt now
    "resume",       # carry on after a stop
    "answer",       # answers the robot's question: "the blue one"
    "chitchat",     # nothing to do
    "constraint",   # changes how a task is done without necessarily replacing it
    "observation",  # user reports world state; update context and reconsider
)

# The tools live in api/tools.py (the ONLY definition). TOOLS maps name -> ToolSpec.
TOOLS = TOOL_SPECS
BODY_TOOLS = _BODY_TOOLS           # hold the one body resource; one at a time
SENSE_TOOLS = _SENSE_TOOLS

# HistoryEntry is now api.execution.Execution: data, status and t_start are real mutable
# fields; id / created_for / tool are read-only aliases (PLAN 5.4).
HistoryEntry = Execution


@dataclass
class BrainInput:
    now: float                            # sim time
    map: dict[str, Any]                   # robot.lookup_keypoints()
    utterances: list[Any]                 # every Utterance delivered so far, oldest first
    kinds: dict[str, str]                 # utterance id -> kind, as classified
    intent_version: int
    belief: dict[str, Any]                # see BELIEF_EXAMPLE below
    history: list[Execution]              # every execution so far, oldest first
    active: list[Execution]               # executions queued or running now
    paused: bool = False                  # stopped by the user and not yet resumed
    note: str | None = None               # e.g. why the last call was rejected
    robot_state: dict[str, Any] = field(default_factory=dict)  # latest fused telemetry (never rendered)
    perception: dict[str, Any] = field(default_factory=dict)   # latest semantic perception (never rendered)
    task_state: dict[str, Any] = field(default_factory=dict)   # directive/goal/control epoch
    directive: dict[str, Any] | None = None                    # latest System-1 directive
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    state_revision: int = 0
    control_epoch: int = 0
    own_goal: dict[str, Any] | None = None                    # the persona's goal, when idle
    notes: list[dict[str, Any]] = field(default_factory=list) # what the user said about the home
    observations: list[dict[str, Any]] = field(default_factory=list)  # what System 1 noticed; unverified
    guidance: str = ""                                        # from the procedural graph, for this step
    tool_results: list[ToolResult] = field(default_factory=list)   # the latest envelopes, oldest first
    tools_ctx: SchemaContext | None = None                    # enum sources + numeric slots, frozen per session
    tool_state: str = "IDLE"                                  # api.state_machine.ToolState, derived
    profile: dict[str, Any] = field(default_factory=dict)     # RobotProfile numeric slots (prompt template)


@runtime_checkable
class Brain(Protocol):
    async def classify(self, utterance: Any, ctx: BrainInput) -> str: ...
    async def next_action(self, ctx: BrainInput) -> ToolCall: ...


@runtime_checkable
class System1(Protocol):
    """The robot's fast attention: it sees every message and, continuously, the fused
    state and the head camera. The server loads one from SYSTEM1="module:factory",
    factory(on_status) -> System1, and runs it next to the runtime (ui/server.py).

    route(text) returns its label for a message, or None:
        {"kind": one of KINDS, "target": "mug" or "", "replaces_task": bool,
         "says_yes": True / False / None, "confidence": 0..1}
    observe() returns what it noticed in the frames since it was last asked:
        [{"what": "the fridge door is open", "where": a keypoint or None, "confidence": 0..1}]

    The runtime applies a label only when it is valid and confident (else the
    planner classifies), and observations enter belief and memory as unverified
    hints, never as object facts."""
    status: str                                    # "connecting", "ready", "error", ...

    async def run(self) -> None: ...
    async def route(self, text: str) -> dict[str, Any] | None: ...
    async def update(self, context: dict[str, Any]) -> None: ...
    async def frame(self, jpeg: bytes, where: str | None) -> None: ...
    async def robot_said(self, text: str) -> None: ...
    async def observe(self) -> list[dict[str, Any]]: ...


@dataclass
class BrainInfo:
    """What the runner hands a brain factory: ``create_brain(info) -> Brain``."""
    scenario_id: str
    map: dict[str, Any]
    meanings: dict[str, dict[str, Any]] | None   # scripted brains only; None for LLM brains
    options: dict[str, str] = field(default_factory=dict)   # --opt key=value from the command line
    clock: Any = None                            # the sim clock, for brains that simulate latency


# ----------------------------------------------------------------------
# Belief: what the runtime believes, rendered as plain data for the brain.
# ----------------------------------------------------------------------
BELIEF_EXAMPLE: dict[str, Any] = {
    "robot": {"at": "bedroom_dresser_1a",     # keypoint, None when between keypoints, or "UNKNOWN"
              "between": None},                # ["hallway", "kitchen_counter_1a"] when stopped on an edge
    "hands": {
        "left": {"holding": None, "verified": True, "source": "look", "t": 12.4},
        # holding: an object id, None (empty) or "UNKNOWN" (e.g. after a cancelled grasp)
        "right": {"holding": "UNKNOWN", "verified": False, "source": "cancel", "t": 13.0},
    },
    "objects": {
        "alarm_clock_1": {
            "type": "alarm_clock", "brand": None, "color": None, "label": "alarm clock",
            "where": "bedroom_dresser_1",      # a surface, "hand:left"/"hand:right", "floor" or "UNKNOWN"
            "x": None, "depth": None,          # THOR-era image position; None on the G1
            "pose": {"x": 1.2, "z": 0.4, "h": 0.86},   # map position (Worldline frame) and height, when known
            "seen_from": "bedroom_dresser_1a", # the keypoint it was seen from
            "verified": True,                  # confirmed by an observation (False: memory, a skill's claim)
            "source": "look",                  # look (an observation), memory, skill, late_result, user ...
            "t": 12.4,                         # sim time of that information
        },
    },
    "blocked": [["hallway", "kitchen_counter_1a"]],  # edges the robot has found blocked
}


def check_belief(belief: dict[str, Any]) -> list[str]:
    """Problems with a belief dict, as messages. Empty when it matches the schema."""
    problems: list[str] = []
    if not isinstance(belief, dict):
        return ["belief must be a dict"]
    robot = belief.get("robot")
    if not isinstance(robot, dict) or "at" not in robot:
        problems.append("belief['robot'] must be a dict with 'at'")
    hands = belief.get("hands")
    if not isinstance(hands, dict) or set(hands) != {"left", "right"}:
        problems.append("belief['hands'] must have exactly 'left' and 'right'")
    else:
        for arm, hand in hands.items():
            if not isinstance(hand, dict) or "holding" not in hand or "verified" not in hand:
                problems.append(f"belief['hands'][{arm!r}] needs 'holding' and 'verified'")
    objects = belief.get("objects")
    if not isinstance(objects, dict):
        problems.append("belief['objects'] must be a dict of object id -> info")
    else:
        for oid, info in objects.items():
            missing = [k for k in ("type", "brand", "color", "where", "verified") if k not in info]
            if missing:
                problems.append(f"belief['objects'][{oid!r}] is missing {missing}")
    return problems


__all__ = ["UNKNOWN", "KINDS", "TOOLS", "BODY_TOOLS", "SENSE_TOOLS", "ToolCall", "HistoryEntry", "Execution",
           "ToolResult", "SchemaContext", "BrainInput", "Brain", "System1", "BrainInfo", "BELIEF_EXAMPLE",
           "check_belief"]
