"""The validation pipeline (PLAN 5.7; doc 21 order). Every Worldline rule from the old
``harness._check`` is re-homed here, at its stage:

    SCHEMA -> (resolve aliases) -> ENUM -> STATE -> CAPABILITY      first failure wins

    v = ValidationContext(...)          # built by the harness from belief, history, map, task ...
    verdict = validate(call, v)         # Verdict(ok, stage, code, message, args=<resolved>)

A rejection becomes a ``rejected`` Execution and ToolResult (data {stage, code}), a
``rejection`` context event, the NOTE ``rejected tool(args): message`` and an ACTIONS line.
The reconcile hold (D9) and the sense-lock wait (D10) are deferrals in the harness, not here.

Alias resolution rewrites args BEFORE the ENUM stage: ``user`` -> people.user.keypoint
(location) or deliver_to_surface (target); a room name -> the room's keypoint;
``reach_stance`` -> the stance of the newest check_reachability (only when its STATE
rule passes). C8a/C8b, timeouts and persona ``ready`` therefore only see real keypoints.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from api.execution import Execution
from api.skills import SkillRegistry
from api.state_machine import ToolState
from api.tools import (ARM_VALUES, BODY_TOOLS, MANIP_ACTIONS, NEUTRAL_TOOLS, REACH_STANCE, USER_ALIAS,
                       ToolCall, coerce_args, schema_issue)
from api.types import ARMS, UNKNOWN, ServiceHealth

REACH_FRESH_S = 30.0


class Stage(str, Enum):
    SCHEMA = "schema"
    ENUM = "enum"
    STATE = "state"
    CAPABILITY = "capability"


@dataclass(frozen=True)
class Verdict:
    ok: bool
    stage: Stage | None = None
    code: str = ""
    message: str = ""                     # what the planner reads
    args: dict[str, Any] = field(default_factory=dict)   # resolved args (also set on rejection, best effort)
    action: str | None = None             # manipulate: pick|place; navigate: keypoint|reposition


def _reject(stage: Stage, code: str, message: str, args: dict[str, Any] | None = None) -> Verdict:
    return Verdict(False, stage, code, message, dict(args or {}))


@dataclass
class ValidationContext:
    belief: Any                           # agent.state.BeliefState
    history: list[Execution]              # every execution, oldest first
    map: dict[str, Any]
    task: Any                             # agent.state.TaskState (paused, utterances, intent_version, ...)
    now: float
    own_goal: Any = None                  # agent.persona.OwnGoal | None
    registry: SkillRegistry | None = None
    skill_types: tuple[str, ...] = ()     # the frozen object_type enum
    capabilities: dict[str, ServiceHealth] = field(default_factory=dict)
    last_motion_t: float = -1e9           # the last base motion (navigate, reposition, scan, base_shift)
    tool_state: ToolState = ToolState.IDLE
    goal_missed: set[str] = field(default_factory=set)
    reach_fresh_s: float = REACH_FRESH_S
    include_look: bool = False
    layout_parse: Callable[[str], Any] | None = None   # goal text -> parsed goal or None


# ----------------------------------------------------------------------
# Helpers over the map and history
# ----------------------------------------------------------------------
def deliver_to(map_: dict[str, Any]) -> str | None:
    return ((map_.get("people") or {}).get("user") or {}).get("deliver_to_surface")


def user_keypoint(map_: dict[str, Any]) -> str | None:
    return ((map_.get("people") or {}).get("user") or {}).get("keypoint")


def resolve_location(loc: str, map_: dict[str, Any]) -> str:
    """user / room names -> a real keypoint. Unknown names come back unchanged (ENUM rejects them)."""
    kps = map_.get("keypoints") or {}
    if loc == USER_ALIAS:
        return user_keypoint(map_) or loc
    if loc in kps:
        return loc
    room = (map_.get("rooms") or {}).get(loc)
    if room:
        kp = room.get("keypoint")
        if kp in kps:
            return kp
        spots = [s for s in room.get("spots") or [] if s in kps]
        if spots:
            return spots[0]
    return loc


def resolve_target(target: str | None, map_: dict[str, Any]) -> str | None:
    if target == USER_ALIAS:
        return deliver_to(map_) or target
    return target


def surface_here(map_: dict[str, Any], at: str | None) -> str | None:
    if at is None:
        return None
    hi = map_.get("max_reach_height_m", 99.0)
    lo = map_.get("min_reach_height_m", -1.0)
    for s, info in (map_.get("surfaces") or {}).items():
        if at in (info.get("keypoints") or []) and lo <= info.get("height_m", 0.0) <= hi:
            return s
    return None


def served_from(map_: dict[str, Any], surface: str, at: str | None) -> bool:
    return at is not None and at in ((map_.get("surfaces") or {}).get(surface) or {}).get("keypoints", [])


def nearest_names(name: str, names: Iterable[str], n: int = 5) -> list[str]:
    names = list(names)
    close = difflib.get_close_matches(str(name), names, n=n, cutoff=0.0)
    return close or names[:n]


def active_body(history: list[Execution]) -> list[Execution]:
    return [e for e in history if not e.finished and "body" in e.resources]


def newest_reach(v: ValidationContext) -> Execution | None:
    for e in reversed(v.history):
        if e.tool_name == "check_reachability" and e.status == "succeeded":
            return e
    return None


def fresh_reach(v: ValidationContext, object_type: str, object_id: str | None = None) -> dict[str, Any] | None:
    """PLAN 1.3 #17 "directly preceded": the newest successful check_reachability for this
    object_type (and bound object_id) must be the last execution with a sense, body or wait
    resource; only speak, list_locations and recall may sit between. No base motion since,
    and at most REACH_FRESH_S old. Rejected or dropped executions never ran and don't count."""
    for e in reversed(v.history):
        if e.status in ("rejected", "dropped"):
            continue
        if e.tool_name in NEUTRAL_TOOLS:
            continue
        if e.tool_name != "check_reachability" or e.status != "succeeded":
            return None
        d = e.data or {}
        if d.get("object_type", e.args.get("object_type")) != object_type:
            return None
        if object_id is not None and d.get("object_id") not in (None, object_id):
            return None
        if e.t_start < v.last_motion_t or v.now - e.t_start > v.reach_fresh_s:
            return None
        return {**d, "execution_id": e.execution_id}
    return None


def reach_candidates(v: ValidationContext, object_type: str, object_id: str | None) -> list[str]:
    """Without object_id: believed instances of that type, visible first, then the last scan here."""
    if object_id is not None:
        return [object_id]
    at = v.belief.robot_at.value
    obs = [(oid, ob) for oid, ob in v.belief.objects.items() if ob.type == object_type]

    def rank(item: tuple[str, Any]) -> tuple[int, int, int, str]:
        oid, ob = item
        return (0 if ob.visible else 1, 0 if (ob.seen_from == at and at is not None) else 1,
                0 if ob.where.verified else 1, oid)
    return [oid for oid, _ in sorted(obs, key=rank)]


def holding_arm(belief: Any, object_type: str, object_id: str | None) -> tuple[str | None, str | None]:
    """The hand verified to hold an object of this type (and id): (arm, object_id)."""
    for arm in ARMS:
        f = belief.holding[arm]
        if not f.verified or f.value in (None, UNKNOWN):
            continue
        ob = belief.objects.get(f.value)
        kind = ob.type if ob is not None else str(f.value).rsplit("_", 1)[0]
        if kind == object_type and (object_id is None or f.value == object_id):
            return arm, f.value
    return None, None


# ----------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------
def validate(call: ToolCall, v: ValidationContext) -> Verdict:
    tool = call.tool
    a = coerce_args(tool, call.args, v.include_look)

    # ---- SCHEMA (C1, C2, C3, argument types and ranges) ----
    issue = schema_issue(tool, a, v.include_look)
    if issue is not None:
        return _reject(Stage.SCHEMA, issue.code, issue.message, a)

    # ---- resolve aliases (not a check) ----
    if tool == "navigate" and a.get("location") != REACH_STANCE:
        a["location"] = resolve_location(str(a["location"]), v.map)
    if tool == "manipulate" and a.get("target") is not None:
        a["target"] = resolve_target(str(a["target"]), v.map)

    # ---- ENUM ----
    r = _enum_stage(tool, a, v)
    if r is not None:
        return r

    # ---- STATE ----
    r, action = _state_stage(tool, a, v)
    if r is not None:
        return r

    # ---- CAPABILITY ----
    r = _capability_stage(tool, a, v, action)
    if r is not None:
        return r
    return Verdict(True, args=a, action=action)


def _enum_stage(tool: str, a: dict[str, Any], v: ValidationContext) -> Verdict | None:
    kps = v.map.get("keypoints") or {}
    if tool == "navigate":
        loc = a["location"]
        if loc != REACH_STANCE and loc not in kps:
            names = list(kps) + [r for r in (v.map.get("rooms") or {}) if r not in kps] + [USER_ALIAS]
            return _reject(Stage.ENUM, "unknown_location",
                           f"unknown location {loc!r}; nearest names: {', '.join(nearest_names(loc, names))}", a)
    if tool in ("check_reachability", "manipulate"):
        if tool == "manipulate" and a.get("action") not in MANIP_ACTIONS:
            return _reject(Stage.ENUM, "invalid_action", "action must be 'pick' or 'place'", a)
        ot = a["object_type"]
        if ot not in v.skill_types:
            return _reject(Stage.ENUM, "unknown_object_skill",
                           f"unknown object skill {ot!r}; skills exist for: {', '.join(v.skill_types) or 'nothing'}", a)
        oid = a.get("object_id")
        if oid is not None:
            ob = v.belief.objects.get(oid)
            if ob is None:
                return _reject(Stage.ENUM, "unknown_object",
                               f"unknown object {oid!r}; use an id from BELIEF (or leave object_id out)", a)
            if ob.type != ot:
                return _reject(Stage.ENUM, "object_type_mismatch", f"{oid} is a {ob.type}, not a {ot}", a)
    if tool == "manipulate":
        if a.get("arm") is not None and a["arm"] not in ARM_VALUES:
            return _reject(Stage.ENUM, "invalid_arm", "invalid arm: arm must be 'left' or 'right'", a)
        tgt = a.get("target")
        if tgt is not None and tgt not in (v.map.get("surfaces") or {}):
            return _reject(Stage.ENUM, "invalid_target",
                           f"unknown target {tgt!r}; use a surface from MAP or 'user'", a)
    return None


def _state_stage(tool: str, a: dict[str, Any], v: ValidationContext) -> tuple[Verdict | None, str | None]:
    b = v.belief
    goal = v.own_goal
    # C4 own-goal whitelist
    if goal is not None and tool not in ("speak", "recall", "list_locations") and tool not in goal.tools:
        return _reject(Stage.STATE, "own_goal_whitelist",
                       f"your own goal ({goal.drive}) allows only {', '.join(goal.tools)}; "
                       f"never move objects unless the user asks", a), None
    body = tool in BODY_TOOLS
    # C5 nobody asked and no own goal
    if body and goal is None and not v.task.utterances:
        return _reject(Stage.STATE, "nobody_asked",
                       "nobody has asked for anything and you have no own goal: call wait_and_observe", a), None
    # C6 paused
    if body and v.task.paused:
        return _reject(Stage.STATE, "paused",
                       "the user said stop: say something if needed and wait until they say to continue", a), None
    # C7 body busy (one body resource)
    busy = active_body(v.history)
    if body and busy:
        e = busy[0]
        return _reject(Stage.STATE, "body_busy",
                       f"already running {e.tool_name} ({e.execution_id}); wait for it to finish", a), None
    if tool == "check_reachability" and any(e.tool_name == "manipulate" for e in busy):
        e = next(e for e in busy if e.tool_name == "manipulate")
        return _reject(Stage.STATE, "body_busy",
                       f"already running {e.tool_name} ({e.execution_id}); wait for it to finish", a), None

    if tool == "navigate":
        if a["location"] == REACH_STANCE:
            r = _reach_stance(a, v)
            return r, "reposition"
        to = a["location"]
        blocked = [e for e in v.history if e.tool_name == "navigate" and e.args.get("location") == to
                   and e.generation == v.task.intent_version
                   and (e.data or {}).get("reason") in ("blocked", "no_path")]
        if len(blocked) >= 2:                                       # C8b
            return _reject(Stage.STATE, "location_failed_twice",
                           f"{to} couldn't be reached twice; tell the user instead", a), "keypoint"
        return None, "keypoint"

    if tool == "check_reachability":
        if b.robot_at.value is None:                                # C10
            return _reject(Stage.STATE, "between_keypoints",
                           "the robot is between keypoints; navigate to one first", a), None
        a["candidates"] = reach_candidates(v, a["object_type"], a.get("object_id"))
        a["at"] = b.robot_at.value
        return None, None

    if tool == "manipulate":
        if a["action"] == "pick":
            return _pick_rules(a, v), "pick"
        return _place_rules(a, v), "place"
    return None, None


def _reach_stance(a: dict[str, Any], v: ValidationContext) -> Verdict | None:
    e = newest_reach(v)
    msg = "reach_stance needs a check_reachability that asked for it; call check_reachability first"
    if e is None:
        return _reject(Stage.STATE, "no_reach_stance", msg, a)
    d = e.data or {}
    if (d.get("reason") != "needs_reposition" or not d.get("stance") or e.generation != v.task.intent_version
            or v.now - e.t_start > v.reach_fresh_s or e.t_start < v.last_motion_t):
        return _reject(Stage.STATE, "no_reach_stance", msg, a)
    a.update({"stance": dict(d["stance"]), "anchor": d.get("at") or v.belief.robot_at.value,
              "reach_execution_id": e.execution_id, "object_type": d.get("object_type"),
              "object_id": d.get("object_id")})
    return None


def _free_arm(b: Any) -> str:
    free = [arm for arm in ARMS if b.hand_known_empty(arm)]
    return free[0] if len(free) == 1 else "right"


def _pick_rules(a: dict[str, Any], v: ValidationContext) -> Verdict | None:
    b = v.belief
    ot = a["object_type"]
    reach = fresh_reach(v, ot, a.get("object_id"))
    oid = a.get("object_id") or (reach or {}).get("object_id")
    if oid is not None:
        a["object_id"] = oid
    # arm: omitted -> the reachability's preferred_arm; either -> the free hand, else right
    pref = (reach or {}).get("preferred_arm")
    if a.get("arm") is None and pref in ARMS:
        a["arm"] = pref
    elif a.get("arm") is None and pref == "either":
        a["arm"] = _free_arm(b)
    arm = a.get("arm")
    # C12a just placed it and nobody asked since
    if oid is not None:
        placed = [e for e in v.history if e.tool_name == "manipulate" and e.action == "place"
                  and (e.data or {}).get("object_id", e.args.get("object_id")) == oid and e.status == "succeeded"]
        heard = any(u.t_end >= placed[-1].t_start for u in v.task.utterances) if placed else True
        if placed and not heard and oid not in v.goal_missed:       # a missed goal may be fixed
            return _reject(Stage.STATE, "just_placed",
                           f"you just put {oid} down and nobody has asked for anything since; "
                           f"say it's done or wait", a)
    # C12b hand not known empty
    if arm is not None and not b.hand_known_empty(arm):
        return _reject(Stage.STATE, "hand_not_empty",
                       f"the {arm} hand is not known to be empty; call wait_and_observe(timeout_s=0) first, "
                       f"or put down what it holds", a)
    # C12f two failed grasps of this object in this generation
    if oid is not None:
        failed = [e for e in v.history if e.tool_name == "manipulate" and e.action == "pick"
                  and (e.data or {}).get("object_id", e.args.get("object_id")) == oid
                  and e.generation == v.task.intent_version and e.status in ("failed", "timed_out")]
        if len(failed) >= 2:
            return _reject(Stage.STATE, "grasp_failed_twice",
                           f"two grasps of {oid} have failed; tell the user instead of retrying", a)
    # C12c directly preceded; C12d reachable; C12e arm
    if reach is None:
        return _reject(Stage.STATE, "no_reachability",
                       f"pick without reachability check: pick needs a successful check_reachability for "
                       f"{ot} right before it", a)
    if not reach.get("reachable"):
        why = reach.get("reason")
        extra = "; navigate(location='reach_stance'), then check again" if why == "needs_reposition" else \
            "; reposition or tell the user"
        return _reject(Stage.STATE, "not_reachable",
                       f"reachability says {oid or ot} can't be reached from here ({why}){extra}", a)
    if pref in ARMS and arm != pref:
        return _reject(Stage.STATE, "arm_mismatch", f"reachability says to use the {pref} arm", a)
    if arm is None:
        a["arm"] = _free_arm(b)
    return None


def _place_rules(a: dict[str, Any], v: ValidationContext) -> Verdict | None:
    b = v.belief
    ot = a["object_type"]
    at = b.robot_at.value
    # C13a a verified hold of an object of object_type (binds that id and the arm)
    if a.get("arm") is not None:
        f = b.holding[a["arm"]]
        ob = b.objects.get(f.value) if f.value not in (None, UNKNOWN) else None
        kind = ob.type if ob is not None else (str(f.value).rsplit("_", 1)[0] if f.value not in (None, UNKNOWN) else None)
        ok = f.verified and kind == ot and (a.get("object_id") in (None, f.value))
        arm, oid = (a["arm"], f.value) if ok else (None, None)
    else:
        arm, oid = holding_arm(b, ot, a.get("object_id"))
    if arm is None:
        which = f"the {a['arm']} hand is" if a.get("arm") else "no hand is"
        return _reject(Stage.STATE, "not_holding",
                       f"{which} not verified to hold a {ot}; call wait_and_observe(timeout_s=0) first", a)
    a["arm"], a["object_id"] = arm, oid
    # C13b a surface here / target served from here
    tgt = a.get("target")
    if tgt is None:
        here = surface_here(v.map, at)
        if here is None:
            return _reject(Stage.STATE, "no_surface_here",
                           "there is no surface to put it on here; navigate to one first", a)
        a["target"] = here
    elif not served_from(v.map, tgt, at):
        kps = ((v.map.get("surfaces") or {}).get(tgt) or {}).get("keypoints") or []
        go = kps[0] if kps else tgt
        return _reject(Stage.STATE, "target_not_here", f"you are at {at}; navigate to {go} first", a)
    # C13d (rule 7, [WL]): two failed places on this target in this generation -> stop retrying
    failed = [e for e in v.history if e.tool_name == "manipulate" and e.action == "place"
              and e.generation == v.task.intent_version and e.status in ("failed", "timed_out")
              and e.args.get("target") == a["target"]]
    if len(failed) >= 2:
        return _reject(Stage.STATE, "place_failed_twice",
                       f"two places on {a['target']} have failed; try another surface or tell the user", a)
    # C13c goal parses
    g = a.get("goal")
    if g and v.layout_parse is not None and v.layout_parse(str(g)) is None:
        from . import layout
        return _reject(Stage.STATE, "goal_not_understood",
                       f"goal {g!r} not understood; write it as one of: {', '.join(layout.GOAL_FORMS)}, "
                       f"with ids from MAP, LAYOUT or BELIEF (or leave goal out)", a)
    return None


def _capability_stage(tool: str, a: dict[str, Any], v: ValidationContext, action: str | None) -> Verdict | None:
    caps = v.capabilities or {}
    if tool == "navigate":
        h = caps.get("navigation")
        if h is not None and not h.ok:
            return _reject(Stage.CAPABILITY, "nav_unhealthy",
                           f"navigation stack unavailable ({h.detail or h.state})", a)
    if tool == "manipulate":
        h = caps.get("manipulation")
        if h is not None and not h.ok:
            return _reject(Stage.CAPABILITY, "policy_unavailable", f"policy unavailable: {h.detail or h.state}", a)
        reg = v.registry
        if reg is not None:
            select_healthy = getattr(reg, "select_healthy", None)
            if callable(select_healthy):
                skill, why = select_healthy(action or a.get("action"), a["object_type"], a.get("arm"))
            else:
                skill = reg.select(action or a.get("action"), a["object_type"], a.get("arm"))
                why = ""
                if skill is not None:
                    sh = reg.healthy(skill.skill_id)
                    if not sh.ok:
                        skill, why = None, f"{skill.skill_id}: {sh.detail or sh.state}"
            if skill is None:
                return _reject(Stage.CAPABILITY, "policy_unavailable",
                               f"policy unavailable: {why or 'no skill for ' + a['object_type'] + ' with that arm'}", a)
            a["skill_id"] = skill.skill_id
    return None


__all__ = ["Stage", "Verdict", "ValidationContext", "validate", "fresh_reach", "newest_reach", "resolve_location",
           "resolve_target", "surface_here", "served_from", "reach_candidates", "holding_arm", "deliver_to",
           "user_keypoint", "REACH_FRESH_S"]
