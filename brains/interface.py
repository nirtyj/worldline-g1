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
as a HistoryEntry.

``ToolCall.tag`` is opaque bookkeeping: copy it into the HistoryEntry
unchanged. Model brains leave it None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

UNKNOWN = "UNKNOWN"

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

TOOLS: dict[str, dict[str, Any]] = {
    "say": {
        "description": "Say something to the user. Speech is queued and played in order; it does not block other actions.",
        "args": {"text": {"type": "string", "description": "What to say, one or two short sentences."}},
    },
    "navigate": {
        "description": "Drive the base to a named keypoint. Takes a few seconds per metre; the arms must be idle.",
        "args": {"to": {"type": "string", "description": "A keypoint name from the map."}},
    },
    "look": {
        "description": "Look at the surfaces at the current keypoint and at both hands (0.5 s). Updates belief.",
        "args": {},
    },
    "reachability": {
        "description": "Check whether an object can be picked from the current keypoint, and with which arm. Required before every pick.",
        "args": {"object": {"type": "string", "description": "An object id from belief."}},
    },
    "pick": {
        "description": "Pick up an object with one arm, using the manipulation policy (about 4-5 s).",
        "args": {"object": {"type": "string", "description": "An object id from belief."},
                 "arm": {"type": "string", "enum": ["left", "right"], "description": "The arm reachability returned."}},
    },
    "place": {
        "description": "Put the object held in `arm` on the surface in front of the robot (about 3-4 s). When the "
                       "request says where it should end up, pass goal: the runtime checks it after the place.",
        "args": {"object": {"type": "string", "description": "The object id being held."},
                 "arm": {"type": "string", "enum": ["left", "right"]},
                 "goal": {"type": "string", "optional": True,
                          "description": "Where the request wants it, as one of: 'on X', 'next to X', 'left of X', "
                                         "'right of X', 'between X and Y', 'other side of X from Y', 'across from X', "
                                         "'near X', 'in X'. X and Y are ids from MAP, LAYOUT or BELIEF. Leave it out "
                                         "when the request doesn't say where."}},
    },
    "recall": {
        "description": "Ask the robot's memory, instantly and without moving: where something is, was, or usually is; "
                       "what is in a room or on a surface; what is next to what (the layout); what the user told you; "
                       "what was asked or delivered in earlier sessions.",
        "args": {"query": {"type": "string", "description": "What to look up, e.g. 'mug', 'kitchen', "
                                                            "'what did the user ask for last time'."}},
    },
    "wait": {
        "description": "Do nothing until something changes: the user speaks or an action finishes.",
        "args": {},
    },
}

BODY_TOOLS = ("navigate", "pick", "place")      # move the robot; one at a time
SENSE_TOOLS = ("look", "reachability")


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None        # copy into HistoryEntry.tag
    reason: str | None = None     # free text, for traces


@dataclass
class HistoryEntry:
    id: int
    tool: str
    args: dict[str, Any]
    created_for: int              # the intent_version this action served
    t_start: float
    status: str                   # queued, running, SUCCEEDED, ABORTED, CANCELED, REJECTED,
                                  # TIMEOUT, FAILED or DROPPED (speech removed from the queue)
    control_epoch: int = 0        # emergency/control generation in which it was created
    data: dict[str, Any] = field(default_factory=dict)   # the skill's result data, unmodified
    t_end: float | None = None
    tag: str | None = None
    source: str = "brain"         # "brain" or "harness" (e.g. a verification look)

    @property
    def finished(self) -> bool:
        return self.status not in ("queued", "running")


@dataclass
class BrainInput:
    now: float                            # sim time
    map: dict[str, Any]                   # robot.lookup_keypoints()
    utterances: list[Any]                 # every Utterance delivered so far, oldest first
    kinds: dict[str, str]                 # utterance id -> kind, as classified
    intent_version: int
    belief: dict[str, Any]                # see BELIEF_EXAMPLE below
    history: list[HistoryEntry]           # every action so far, oldest first
    active: list[HistoryEntry]            # actions queued or running now
    paused: bool = False                  # stopped by the user and not yet resumed
    note: str | None = None               # e.g. why the last call was rejected
    robot_state: dict[str, Any] = field(default_factory=dict)  # latest fused telemetry
    perception: dict[str, Any] = field(default_factory=dict)   # latest semantic perception
    task_state: dict[str, Any] = field(default_factory=dict)   # directive/goal/control epoch
    directive: dict[str, Any] | None = None                    # latest System-1 directive
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    state_revision: int = 0
    control_epoch: int = 0
    own_goal: dict[str, Any] | None = None                    # the persona's goal, when idle
    notes: list[dict[str, Any]] = field(default_factory=list) # what the user said about the home
    observations: list[dict[str, Any]] = field(default_factory=list)  # what System 1 noticed; unverified
    guidance: str = ""                                        # from the procedural graph, for this step


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
    "robot": {"at": "kitchen_table",          # keypoint, None when between keypoints, or "UNKNOWN"
              "between": None},                # ["hallway", "kitchen_table"] when stopped on an edge
    "hands": {
        "left": {"holding": None, "verified": True, "source": "look", "t": 12.4},
        # holding: an object id, None (empty) or "UNKNOWN" (e.g. after a cancelled grasp)
        "right": {"holding": "UNKNOWN", "verified": False, "source": "cancel", "t": 13.0},
    },
    "objects": {
        "mug_1": {
            "type": "mug", "brand": None, "color": None, "label": "mug",
            "where": "kitchen_table",          # a surface, "hand:left"/"hand:right", "floor" or "UNKNOWN"
            "x": 0.31, "depth": 0.2,           # position as last seen (0 = left), None if unknown
            "seen_from": "kitchen_table",      # the keypoint it was seen from
            "verified": True,                  # confirmed by a look (False: memory, a skill's claim)
            "source": "look",                  # look, memory, skill, late_result, user ...
            "t": 12.4,                         # sim time of that information
        },
    },
    "blocked": [["hallway", "kitchen_table"]],  # edges the robot has found blocked
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
