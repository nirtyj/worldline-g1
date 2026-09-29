"""Reason codes (PLAN 5.3) and validation codes (PLAN 5.7), each with a one-line hint.

A result's ``data["reason"]`` is one of these codes (or ``inside_or_on_<x>``).
The hint is what the planner can do about it; summaries append it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Area = Literal["navigation", "reachability", "manipulation", "speech", "observation", "validation", "general"]


@dataclass(frozen=True)
class Reason:
    code: str
    area: Area
    hint: str


def _r(area: Area, rows: dict[str, str]) -> dict[str, Reason]:
    return {code: Reason(code, area, hint) for code, hint in rows.items()}


NAVIGATION = _r("navigation", {
    "unknown_location": "use a name from list_locations",
    "no_path": "that place can't be reached from here; try another stand of the same surface or tell the user",
    "blocked": "the way is blocked; try another route or tell the user",
    "halted": "the robot was stopped",
    "cancelled": "the walk was cancelled",
    "fell": "the robot fell; wait for recovery",
    "stuck": "the robot got stuck; it will replan once",
    "timeout": "the walk took too long",
    "body_busy": "another body action is running; wait for it",
    "stale_result": "a newer stop or correction fenced this command; it did not run",
    "nav_unhealthy": "navigation is unavailable right now",
    "sim_slow": "the simulator is below real time; walking is capped",
    "no_reach_stance": "call check_reachability first; reach_stance needs one that asked for it",
    "stance_not_reached": "the reposition missed; check reachability again or try the keypoint",
})

REACHABILITY = _r("reachability", {
    "base_moving": "wait until the robot has stopped, then check again",
    "not_found": "no object of that type is known; look somewhere else",
    "in_hand": "it is already in a hand",
    "not_seen_here": "not visible from here; navigate to where it is, or wait_and_observe(timeout_s=0)",
    "too_high": "too high for the robot to reach; tell the user",
    "too_low": "too low for the robot to reach; tell the user",
    "needs_reposition": "navigate(location='reach_stance'), then check again",
    "too_far": "navigate to the suggested location, then check again",
    "out_of_workspace": "the arm can't reach it from this pose; try another stand",
    "beyond_reach": "it sits deeper than the arm reaches from anywhere the robot can stand; tell the user",
    "hand_full": "put down what the robot holds first",
    "no_skill": "no skill can handle that object",
    "policy_unavailable": "the manipulation policy is down; tell the user",
})

MANIPULATION = _r("manipulation", {
    "grasp_failed": "the grasp failed; wait_and_observe(timeout_s=0), then retry once or tell the user",
    "grasp_missed": "the hand closed on nothing; check reachability again",
    "object_dropped": "the object fell; look for it",
    "not_in_ego_view": "the object is not in the hand camera's view; check reachability again",
    "no_surface_here": "there is no surface here; navigate to one first",
    "target_not_here": "navigate to the target first",
    "no_room_on_surface": "the surface is full; choose another",
    "no_room_in_reach": "no free spot within reach; navigate to another stretch of that surface",
    "halted": "the robot was stopped",
    "fell": "the robot fell; wait for recovery",
    "policy_stall": "the policy stopped making progress",
    "policy_out_of_bounds": "the policy produced unsafe actions and was stopped",
    "policy_unavailable": "the manipulation policy is down; tell the user",
    "controller_unavailable": "the whole-body controller is not running",
    "body_busy": "another body action is running; wait for it",
    "stale_result": "a newer stop or correction fenced this command; it did not run",
    "timeout": "the skill took too long",
    "base_moving": "the robot moved since the reachability check; check again",
    "nothing_in_hand": "the robot holds nothing to place",
    "ik_unreachable": "the arm can't reach the grasp point from exactly here; check reachability again",
})

OBSERVATION = _r("observation", {
    "timeout": "the look took too long; move on, or look again later",
    "halted": "the robot was stopped",
    "cancelled": "the look was cancelled",
    "body_busy": "another body action is running; wait for it",
    "stale_result": "a newer stop or correction fenced this command; it did not run",
})

GENERAL = _r("general", {
    "cancelled": "it was cancelled",
    "cancelled before start": "it was cancelled before it started",
    "superseded": "a newer request replaced it",
    "stale_result": "a newer stop or correction fenced this command; it did not run",
    "internal_error": "something went wrong inside the robot; try again or tell the user",
    "shutdown": "the robot is shutting down",
})

# Validation (rejection) codes by stage; the message itself is built by agent/validate.py.
VALIDATION = _r("validation", {
    # SCHEMA
    "unknown_tool": "call one of the listed tools",
    "missing_arg": "pass every required argument",
    "bad_arg_type": "fix the argument type",
    "out_of_range": "use a value inside the range",
    "empty_text": "say something",
    "empty_query": "ask something",
    "two_calls": "call exactly one tool per turn",
    # ENUM
    "unknown_location": "use a name from list_locations",
    "unknown_object_skill": "use an object type a skill exists for",
    "unknown_object": "use an id from BELIEF",
    "object_type_mismatch": "the id and the object_type disagree",
    "invalid_arm": "arm must be left or right",
    "invalid_target": "target must be a surface or user",
    "invalid_action": "action must be pick or place",
    # STATE
    "own_goal_whitelist": "stay within the own goal's tools",
    "nobody_asked": "nobody has asked for anything; wait",
    "paused": "the user said stop; wait until they say to continue",
    "body_busy": "wait for the running body action to finish",
    "location_failed_twice": "tell the user instead",
    "tell_user_blocked": "tell the user the way is blocked (speak), then try another route or wait",
    "between_keypoints": "navigate to a keypoint first",
    "no_reach_stance": "call check_reachability first",
    "just_placed": "say it's done or wait",
    "hand_not_empty": "wait_and_observe(timeout_s=0) first, or put down what it holds",
    "grasp_failed_twice": "tell the user instead of retrying",
    "no_reachability": "call check_reachability right before the pick",
    "not_reachable": "reposition or tell the user",
    "arm_mismatch": "use the arm check_reachability preferred",
    "not_holding": "wait_and_observe(timeout_s=0) first",
    "no_surface_here": "navigate to a surface first",
    "target_not_here": "navigate to the target first",
    "goal_not_understood": "write the goal in one of the listed forms",
    "place_failed_twice": "try another surface or tell the user",
    # CAPABILITY
    "nav_unhealthy": "navigation is unavailable; tell the user",
    "policy_unavailable": "the manipulation policy is unavailable; tell the user",
    "controller_unavailable": "the whole-body controller is unavailable",
})

ALL: dict[str, Reason] = {}
for _table in (GENERAL, VALIDATION, MANIPULATION, REACHABILITY, NAVIGATION):
    ALL.update(_table)            # later tables win for shared codes (area-specific hints)

INSIDE_OR_ON = "inside_or_on_"


def lookup(code: str | None, area: Area | None = None) -> Reason | None:
    if not code:
        return None
    if code.startswith(INSIDE_OR_ON):
        return Reason(code, "reachability", f"it is inside or on {code[len(INSIDE_OR_ON):]}; "
                                            f"it can't be picked from there")
    if area is not None:
        table = {"navigation": NAVIGATION, "reachability": REACHABILITY, "manipulation": MANIPULATION,
                 "observation": OBSERVATION, "validation": VALIDATION, "general": GENERAL}.get(area, {})
        if code in table:
            return table[code]
    return ALL.get(code)


def hint(code: str | None, area: Area | None = None) -> str:
    r = lookup(code, area)
    return r.hint if r else ""


def is_known(code: str | None) -> bool:
    return lookup(code) is not None


# the area whose hint a tool's result gets (a shared code such as `timeout` reads differently per tool)
AREA_OF_TOOL: dict[str, Area] = {"navigate": "navigation", "check_reachability": "reachability",
                                 "manipulate": "manipulation", "observe": "observation", "look": "observation",
                                 "wait_and_observe": "observation"}

__all__ = ["Reason", "NAVIGATION", "REACHABILITY", "MANIPULATION", "OBSERVATION", "GENERAL", "VALIDATION", "ALL",
           "AREA_OF_TOOL",
           "INSIDE_OR_ON", "lookup", "hint", "is_known"]
