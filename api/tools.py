"""Planner-facing tools: the ONLY definition (PLAN 5.1, 5.2).

Six Ludi tools (doc 31) plus Worldline's ``recall`` [WL]:

    speak, list_locations, navigate, check_reachability, manipulate, wait_and_observe, recall

``json_schemas(ctx)`` renders provider-neutral JSON Schema, one dict per tool,
with enums filled by argument name from a ``SchemaContext`` built once per
session (Invariant 9: schemas never change within a session). Descriptions
are templates; only numeric slots (``RobotProfile.slots()``) differ by profile.

Stdlib only; importable on Python 3.11 and 3.12.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

ToolKind = Literal["instant", "sense", "physical", "wait", "internal"]
Provider = Literal["neutral", "anthropic", "gemini", "openai"]


@dataclass(frozen=True)
class Arg:
    type: Literal["string", "number"]
    description: str = ""
    optional: bool = False
    enum: tuple[str, ...] | None = None                  # fixed values
    enum_from: Literal["locations", "surfaces", "skill_types"] | None = None   # filled once per session
    enum_extra: tuple[str, ...] = ()                     # appended to an enum_from enum
    minimum: float | None = None
    maximum: float | None = None
    default: Any = None
    extension: bool = False                              # [WL] argument beyond the Ludi doc


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind                                       # dispatch + state machine
    resources: frozenset[str]                            # {"body"} | {"sense"} | {"speech"} | {"wait"} | {}
    description: str                                     # template; numeric slots only
    args: Mapping[str, Arg]
    extension: bool = False                              # [WL]

    @property
    def required(self) -> list[str]:
        return [k for k, a in self.args.items() if not a.optional]


@dataclass
class ToolCall:
    """One decision from the planner (brains call it the next action)."""
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None                               # opaque bookkeeping; copied into the Execution
    reason: str | None = None                            # free text, for traces
    extra: list["ToolCall"] = field(default_factory=list)   # calls beyond the first in one reply (rejected)


# ----------------------------------------------------------------------
# The tools
# ----------------------------------------------------------------------
GOAL_FORMS_TEXT = ("'on X', 'next to X', 'left of X', 'right of X', 'between X and Y', "
                   "'other side of X from Y', 'across from X', 'near X', 'in X'")

TOOL_SPECS: dict[str, ToolSpec] = {
    "speak": ToolSpec(
        "speak", "instant", frozenset({"speech"}),
        "Say something to the user. Speech is queued and played in order; it never blocks walking "
        "or manipulation.",
        {"text": Arg("string", "One or two short sentences.")}),
    "list_locations": ToolSpec(
        "list_locations", "instant", frozenset(),
        "List the named places the robot can walk to, nearest first, with walking distance from "
        "where it is now.",
        {"query": Arg("string", "Optional filter, e.g. 'kitchen' or 'table'.", optional=True)}),
    "navigate": ToolSpec(
        "navigate", "physical", frozenset({"body"}),
        "Walk to a named location (about {s_per_m:.1f} s per metre). Whatever the hands hold stays "
        "held. The robot looks around when it arrives. location='reach_stance' takes the short step "
        "that the last check_reachability asked for (up to {approach_max_m:.2f} m); it does not look "
        "around.",
        {"location": Arg("string", "A named location: a keypoint, a room, 'user', or 'reach_stance'.",
                         enum_from="locations", enum_extra=("reach_stance",)),
         "timeout_s": Arg("number", "Optional; can only lower the default timeout.", optional=True,
                          minimum=1.0, maximum=240.0)}),
    "check_reachability": ToolSpec(
        "check_reachability", "sense", frozenset({"sense"}),
        "Check whether an object of this type is visible from where the robot stands now, within "
        "arm reach from this exact pose, and which arm to use. Required right before every pick. If "
        "it says needs_reposition, call navigate(location='reach_stance') and check again.",
        {"object_type": Arg("string", "The kind of object.", enum_from="skill_types"),
         "object_id": Arg("string", "Which one, when BELIEF has several: an id from BELIEF, e.g. "
                          "alarm_clock_1.", optional=True, extension=True)}),
    "manipulate": ToolSpec(
        "manipulate", "physical", frozenset({"body"}),
        "Run a trained manipulation skill from where the robot stands; it never walks (a pick takes "
        "about {t_pick:.0f} s, a place about {t_place:.0f} s). pick needs a successful "
        "check_reachability for the same object right before it. place puts the held object on "
        "target, which must be a surface served from where the robot stands (default: the surface "
        "here). When the request says where it should end up, pass goal: the runtime checks it "
        "after the place.",
        {"action": Arg("string", "pick or place.", enum=("pick", "place")),
         "object_type": Arg("string", "The kind of object.", enum_from="skill_types"),
         "arm": Arg("string", "The arm check_reachability preferred (for place: the holding hand).",
                    optional=True, enum=("left", "right")),
         "target": Arg("string", "place only: the surface to put it on, or 'user'.", optional=True,
                       enum_from="surfaces", enum_extra=("user",)),
         "object_id": Arg("string", "An id from BELIEF, e.g. alarm_clock_1.", optional=True,
                          extension=True),
         "goal": Arg("string", "Where the request wants it, as one of: " + GOAL_FORMS_TEXT + ". X and "
                     "Y are ids from MAP, LAYOUT or BELIEF. Leave it out when the request doesn't say "
                     "where.", optional=True, extension=True)}),
    "wait_and_observe": ToolSpec(
        "wait_and_observe", "wait", frozenset({"wait"}),
        "Take a fresh look from where the robot is, then hold still and keep observing until "
        "something changes (the user speaks, an action finishes, something new is seen) or "
        "timeout_s passes. timeout_s=0 just looks once.",
        {"timeout_s": Arg("number", "Seconds to keep observing; 0 looks once.", optional=True,
                          minimum=0.0, maximum=60.0, default=10.0),
         "reason": Arg("string", "Why, in a few words.", optional=True)}),
    "recall": ToolSpec(
        "recall", "instant", frozenset(),
        "Ask the robot's memory, instantly and without moving: where something is, was, or usually "
        "is; what is in a room or on a surface; what is next to what (the layout); what the user "
        "told you; what was asked or delivered in earlier sessions.",
        {"query": Arg("string", "What to look up, e.g. 'mug', 'kitchen', 'what did the user ask "
                      "for last time'.")},
        extension=True),
}

# WL_LOOK_TOOL=1 escape hatch (PLAN 1.3 #8): re-adds a planner-visible look.
LOOK_SPEC = ToolSpec(
    "look", "sense", frozenset({"sense"}),
    "Look around from where the robot stands (a waist scan, about 5 s) and at both hands. "
    "Updates belief.",
    {}, extension=True)

# Internal, never planner-visible: harness-initiated observations (arrival scan, verify, reconcile).
OBSERVE_SPEC = ToolSpec(
    "observe", "internal", frozenset({"sense"}),
    "Internal observation (glance or scan).",
    {"mode": Arg("string", "", enum=("glance", "scan")), "why": Arg("string", "", optional=True)})

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SPECS)
PLANNER_TOOLS = TOOL_NAMES
BODY_TOOLS: tuple[str, ...] = ("navigate", "manipulate")          # hold the one body resource
SENSE_TOOLS: tuple[str, ...] = ("check_reachability",)
INSTANT_TOOLS: tuple[str, ...] = ("speak", "list_locations", "recall")
WAIT_TOOLS: tuple[str, ...] = ("wait_and_observe",)
INTERNAL_TOOLS: tuple[str, ...] = ("observe",)
# "Directly preceded" (PLAN 1.3 #17): only these may sit between a check_reachability and its pick.
NEUTRAL_TOOLS: tuple[str, ...] = ("speak", "list_locations", "recall")

# The Worldline/THOR names that must not appear as tool literals any more (test_tool_vocab).
OLD_TOOL_NAMES: tuple[str, ...] = ("say", "look", "reachability", "pick", "place", "wait")
RENAMES: dict[str, str] = {"say": "speak", "reachability": "check_reachability", "pick": "manipulate",
                           "place": "manipulate", "wait": "wait_and_observe", "look": "wait_and_observe"}

MANIP_ACTIONS: tuple[str, ...] = ("pick", "place")
ARM_VALUES: tuple[str, ...] = ("left", "right")
REACH_STANCE = "reach_stance"
USER_ALIAS = "user"
UNKNOWN_TOOL_FALLBACK = "wait_and_observe"   # an unknown tool name becomes wait_and_observe(10, reason)


def spec(name: str, include_look: bool = False) -> ToolSpec | None:
    if name in TOOL_SPECS:
        return TOOL_SPECS[name]
    if include_look and name == "look":
        return LOOK_SPEC
    if name == "observe":
        return OBSERVE_SPEC
    return None


def tool_specs(include_look: bool = False) -> dict[str, ToolSpec]:
    out = dict(TOOL_SPECS)
    if include_look:
        out["look"] = LOOK_SPEC
    return out


def kind_of(tool: str) -> ToolKind | None:
    s = spec(tool, include_look=True)
    return s.kind if s else None


def resources_of(tool: str, args: Mapping[str, Any] | None = None) -> frozenset[str]:
    """Resources an execution of this tool holds. A scan also holds the body (it moves the waist)."""
    if tool == "observe" and (args or {}).get("mode") == "scan":
        return frozenset({"sense", "body"})
    s = spec(tool, include_look=True)
    return s.resources if s else frozenset()


# ----------------------------------------------------------------------
# Schema generation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaContext:
    """Enum sources and numeric slots, fixed once per session (PLAN 5.2, Invariant 9)."""
    locations: tuple[str, ...] = ()        # keypoints ∪ room names ∪ {"user"}; reach_stance is added
    surfaces: tuple[str, ...] = ()         # map surfaces; "user" is added for target
    skill_types: tuple[str, ...] = ()      # SkillRegistry.loaded_object_types()
    slots: Mapping[str, float] = field(default_factory=dict)
    include_look: bool = False
    schema_rev: int = 0

    @classmethod
    def from_map(cls, map_: Mapping[str, Any], skill_types: Iterable[str],
                 slots: Mapping[str, float], include_look: bool = False, schema_rev: int = 0) -> "SchemaContext":
        kps = list((map_.get("keypoints") or {}).keys())
        rooms = [r for r in (map_.get("rooms") or {}).keys() if r not in kps]
        locations = sorted(set(kps)) + sorted(set(rooms))
        if USER_ALIAS not in locations:
            locations.append(USER_ALIAS)
        surfaces = sorted((map_.get("surfaces") or {}).keys())
        return cls(tuple(locations), tuple(surfaces), tuple(sorted(set(skill_types))), dict(slots),
                   include_look, schema_rev)

    def enum_for(self, a: Arg) -> list[str] | None:
        if a.enum is not None:
            return list(a.enum)
        if a.enum_from is None:
            return None
        base = {"locations": self.locations, "surfaces": self.surfaces,
                "skill_types": self.skill_types}[a.enum_from]
        out = list(base)
        for extra in a.enum_extra:
            if extra not in out:
                out.append(extra)
        return out


class _Slots(string.Formatter):
    """Formats only named numeric slots; a missing slot renders as '?' so text stays parseable."""

    def get_value(self, key, args, kwargs):      # noqa: D401
        if isinstance(key, str):
            return kwargs.get(key, float("nan"))
        return super().get_value(key, args, kwargs)

    def format_field(self, value, format_spec):
        if isinstance(value, float) and value != value:      # NaN: slot not provided
            return "?"
        return super().format_field(value, format_spec)


def render_description(ts: ToolSpec, slots: Mapping[str, float]) -> str:
    return _Slots().format(ts.description, **dict(slots))


def _arg_schema(a: Arg, ctx: SchemaContext) -> dict[str, Any]:
    out: dict[str, Any] = {"type": a.type}
    if a.description:
        out["description"] = a.description
    enum = ctx.enum_for(a)
    if enum is not None:
        out["enum"] = enum
    if a.minimum is not None:
        out["minimum"] = a.minimum
    if a.maximum is not None:
        out["maximum"] = a.maximum
    return out


def tool_schema(ts: ToolSpec, ctx: SchemaContext, provider: Provider = "neutral") -> dict[str, Any]:
    props = {name: _arg_schema(a, ctx) for name, a in ts.args.items()}
    params: dict[str, Any] = {"type": "object", "properties": props}
    required = ts.required
    if required:
        params["required"] = required
    if provider == "anthropic":
        params["additionalProperties"] = False
    return {"name": ts.name, "description": render_description(ts, ctx.slots), "parameters": params}


def json_schemas(ctx: SchemaContext, provider: Provider = "neutral") -> list[dict[str, Any]]:
    """Provider-neutral JSON Schema, one per planner tool, in TOOL_SPECS order.

    Optional args are left out of ``required``; ``["string","null"]`` unions are never
    used; ``additionalProperties: false`` is emitted only for Anthropic (PLAN 5.1)."""
    return [tool_schema(ts, ctx, provider) for ts in tool_specs(ctx.include_look).values()]


# ----------------------------------------------------------------------
# SCHEMA-stage checks (the first validation stage; PLAN 5.7). Pure: no belief, no map.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaIssue:
    code: str          # unknown_tool | missing_arg | bad_arg_type | out_of_range | empty_text | empty_query
    message: str


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def coerce_args(tool: str, args: Mapping[str, Any] | None, include_look: bool = False) -> dict[str, Any]:
    """Drop unknown keys and None values; turn numeric strings into numbers for number args."""
    ts = spec(tool, include_look)
    if ts is None:
        return dict(args or {})
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        a = ts.args.get(k)
        if a is None or v is None:
            continue
        if a.type == "number" and isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                pass
        out[k] = v
    return out


def schema_issue(tool: str, args: Mapping[str, Any] | None, include_look: bool = False) -> SchemaIssue | None:
    """The first SCHEMA-stage problem with this call, or None."""
    ts = spec(tool, include_look)
    if ts is None or ts.kind == "internal":
        names = ", ".join(tool_specs(include_look))
        return SchemaIssue("unknown_tool", f"unknown tool {tool!r}; use one of: {names}")
    a = dict(args or {})
    if tool == "speak" and not str(a.get("text") or "").strip():
        return SchemaIssue("empty_text", "speak needs some text")
    if tool == "recall" and not str(a.get("query") or "").strip():
        return SchemaIssue("empty_query", "recall needs a query")
    for name in ts.required:
        if name not in a or a[name] is None or (isinstance(a[name], str) and not a[name].strip()):
            return SchemaIssue("missing_arg", f"{tool} needs {name}")
    for name, value in a.items():
        arg = ts.args.get(name)
        if arg is None:
            continue
        if arg.type == "string" and not isinstance(value, str):
            return SchemaIssue("bad_arg_type", f"{tool}: {name} must be a string")
        if arg.type == "number":
            if not _is_number(value):
                return SchemaIssue("bad_arg_type", f"{tool}: {name} must be a number")
            if (arg.minimum is not None and value < arg.minimum) or (arg.maximum is not None and value > arg.maximum):
                return SchemaIssue("out_of_range", f"{tool}: {name} must be between {arg.minimum:g} and "
                                                   f"{arg.maximum:g}")
    return None


__all__ = ["Arg", "ToolSpec", "ToolCall", "TOOL_SPECS", "LOOK_SPEC", "OBSERVE_SPEC", "TOOL_NAMES",
           "PLANNER_TOOLS", "BODY_TOOLS", "SENSE_TOOLS", "INSTANT_TOOLS", "WAIT_TOOLS", "INTERNAL_TOOLS",
           "NEUTRAL_TOOLS", "OLD_TOOL_NAMES", "RENAMES", "MANIP_ACTIONS", "ARM_VALUES", "REACH_STANCE",
           "USER_ALIAS", "UNKNOWN_TOOL_FALLBACK", "GOAL_FORMS_TEXT", "SchemaContext", "SchemaIssue",
           "spec", "tool_specs", "kind_of", "resources_of", "render_description", "tool_schema",
           "json_schemas", "coerce_args", "schema_issue"]
