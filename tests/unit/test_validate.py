"""agent/validate.py: every rule lands at its stage with the doc's wording, first failure wins,
aliases resolve before ENUM, the directly-preceded rule and the reach_stance rule."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.state import BeliefState, Fact, ObjectBelief, TaskState
from agent.validate import Stage, ValidationContext, fresh_reach, validate
from api.execution import ExecutionManager
from api.tools import ToolCall
from api.types import ServiceHealth
from tests.unit.fakes_rt import FakeRobot, house_map


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def now(self):
        return self.t


def utt(text="bring me the alarm clock", t=1.0):
    return SimpleNamespace(id="u1", text=text, t_end=t)


class World:
    """A validation context builder with a mutable history."""

    def __init__(self, at="bedroom_dresser_1a"):
        self.map = house_map()
        self.belief = BeliefState()
        self.belief.set_pose(at, None, "odometry", 0.0)
        for arm in ("left", "right"):
            self.belief.set_hand(arm, None, "gripper", 0.0, verified=True)
        self.task = TaskState(intent_version=1)
        self.task.utterances.append(utt())
        self.m = ExecutionManager(Clock())
        self.now = 100.0
        self.last_motion_t = 50.0
        self.caps = {"navigation": ServiceHealth(True, "ok"), "manipulation": ServiceHealth(True, "ok")}
        self.registry = FakeRobot(Clock()).registry()
        self.own_goal = None
        self.add_obj("alarm_clock_1", "alarm_clock", "bedroom_dresser_1a")
        self.add_obj("book_1", "book", "kitchen_counter_1b")

    def add_obj(self, oid, kind, where, verified=True):
        self.belief.objects[oid] = ObjectBelief(oid, kind, None, None, Fact(where, "look", 60.0, verified))

    def run(self, tool, status="succeeded", t=None, **args):
        e = self.m.create(tool, args, generation=self.task.intent_version, control_epoch=0, t=t or self.now - 1)
        e.status = status
        e.t_start = t or self.now - 1
        return e

    def reach(self, reachable=True, reason=None, t=None, **data):
        e = self.run("check_reachability", t=t, object_type=data.get("object_type", "alarm_clock"))
        e.data = {"reachable": reachable, "reason": reason, "object_type": data.pop("object_type", "alarm_clock"),
                  "object_id": data.pop("object_id", "alarm_clock_1"), "preferred_arm": data.pop("arm", "right"),
                  "at": self.belief.robot_at.value, **data}
        return e

    def ctx(self, **kw):
        base = dict(belief=self.belief, history=self.m.all, map=self.map, task=self.task, now=self.now,
                    own_goal=self.own_goal, registry=self.registry,
                    skill_types=tuple(self.registry.loaded_object_types()), capabilities=self.caps,
                    last_motion_t=self.last_motion_t)
        base.update(kw)
        return ValidationContext(**base)

    def check(self, tool, **args):
        return validate(ToolCall(tool, args), self.ctx())


def expect(v, stage, code, text=None):
    assert not v.ok and v.stage == stage and v.code == code, v
    if text:
        assert text in v.message, v.message


# ---------------------------------------------------------------- SCHEMA
def test_schema_stage_first():
    w = World()
    w.task.paused = True
    expect(w.check("fly"), Stage.SCHEMA, "unknown_tool", "unknown tool 'fly'; use one of:")
    expect(w.check("speak", text=""), Stage.SCHEMA, "empty_text", "speak needs some text")
    expect(w.check("recall", query=" "), Stage.SCHEMA, "empty_query", "recall needs a query")
    expect(w.check("navigate"), Stage.SCHEMA, "missing_arg")          # before paused (STATE)
    expect(w.check("wait_and_observe", timeout_s=99), Stage.SCHEMA, "out_of_range")


# ---------------------------------------------------------------- aliases + ENUM
def test_aliases_resolve_before_enum():
    w = World(at="start")
    v = w.check("navigate", location="user")
    assert v.ok and v.args["location"] == "living_room_table_1a" and v.action == "keypoint"
    v = w.check("navigate", location="kitchen")
    assert v.ok and v.args["location"] == "kitchen"                   # a room keypoint
    expect(w.check("navigate", location="garage"), Stage.ENUM, "unknown_location", "unknown location 'garage'")
    assert "nearest names:" in w.check("navigate", location="kitchen_countr").message


def test_enum_stage_rules():
    w = World()
    w.task.paused = True                                              # ENUM before STATE
    expect(w.check("check_reachability", object_type="unicorn"), Stage.ENUM, "unknown_object_skill",
           "unknown object skill 'unicorn'; skills exist for: ")
    expect(w.check("check_reachability", object_type="alarm_clock", object_id="clock_9"), Stage.ENUM,
           "unknown_object")
    expect(w.check("check_reachability", object_type="book", object_id="alarm_clock_1"), Stage.ENUM,
           "object_type_mismatch", "alarm_clock_1 is a alarm_clock, not a book")
    expect(w.check("manipulate", action="throw", object_type="book"), Stage.ENUM, "invalid_action")
    expect(w.check("manipulate", action="pick", object_type="book", arm="middle"), Stage.ENUM, "invalid_arm",
           "invalid arm")
    expect(w.check("manipulate", action="place", object_type="book", target="garage"), Stage.ENUM, "invalid_target")


def test_banana_is_in_the_enum_even_when_absent():
    w = World()
    v = w.check("check_reachability", object_type="banana")
    assert v.ok and v.args["candidates"] == []                        # the service answers not_found


# ---------------------------------------------------------------- STATE
def test_own_goal_nobody_asked_and_paused():
    w = World()
    w.own_goal = SimpleNamespace(drive="glance", tools=("wait_and_observe",))
    expect(w.check("navigate", location="kitchen"), Stage.STATE, "own_goal_whitelist", "allows only wait_and_observe")
    assert w.check("speak", text="hi").ok and w.check("recall", query="mug").ok
    w.own_goal = None
    w.task.utterances.clear()
    expect(w.check("navigate", location="kitchen"), Stage.STATE, "nobody_asked", "call wait_and_observe")
    w.task.utterances.append(utt())
    w.task.paused = True
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "paused", "the user said stop")
    assert w.check("speak", text="Okay.").ok
    assert w.check("wait_and_observe", timeout_s=0).ok


def test_body_busy():
    w = World()
    nav = w.run("navigate", status="running", location="kitchen")
    expect(w.check("navigate", location="bedroom"), Stage.STATE, "body_busy",
           f"already running navigate ({nav.execution_id}); wait for it to finish")
    assert w.check("check_reachability", object_type="alarm_clock").ok        # allowed while walking
    nav.status = "succeeded"
    w.run("manipulate", status="running", action="pick", object_type="book")
    expect(w.check("check_reachability", object_type="alarm_clock"), Stage.STATE, "body_busy")


def test_same_location_failed_twice():
    w = World(at="start")
    for _ in range(2):
        e = w.run("navigate", status="failed", location="kitchen")
        e.data = {"reason": "blocked"}
    expect(w.check("navigate", location="kitchen"), Stage.STATE, "location_failed_twice", "tell the user")
    w.task.intent_version += 1                                        # a new generation: allowed again
    assert w.check("navigate", location="kitchen").ok


def test_blocked_twice_on_any_route_tell_the_user_first():
    """G6: after two `blocked` walks in this request the planner must tell the user before another route; once
    told, one more try is allowed, and the next blocked walk asks again."""
    w = World(at="start")
    for loc in ("kitchen", "living_room"):
        e = w.run("navigate", status="failed", location=loc)
        e.data = {"reason": "blocked", "blocked_edge": ["start", loc]}
    expect(w.check("navigate", location="bedroom_dresser_1a"), Stage.STATE, "tell_user_blocked", "tell the user")
    v = w.check("navigate", location="bedroom_dresser_1a")
    assert "start-kitchen" in v.message and "start-living_room" in v.message
    assert w.check("speak", text="The way is blocked; I'll try the hallway.").ok       # speaking is allowed
    sp = w.run("speak", text="The way is blocked; I'll try the hallway.")
    sp.source = "brain"
    assert w.check("navigate", location="bedroom_dresser_1a").ok                      # told: one more try
    e = w.run("navigate", status="failed", location="bedroom_dresser_1a")
    e.data = {"reason": "blocked", "blocked_edge": ["start", "bedroom_dresser_1a"]}
    expect(w.check("navigate", location="kitchen_counter_1b"), Stage.STATE, "tell_user_blocked")
    w.task.intent_version += 1                                                       # a new request: allowed
    assert w.check("navigate", location="kitchen_counter_1b").ok


def test_reachability_between_keypoints():
    w = World()
    w.belief.set_pose(None, ["start", "kitchen"], "odometry", 90.0)
    expect(w.check("check_reachability", object_type="book"), Stage.STATE, "between_keypoints")


def test_reach_candidates_order_visible_then_scanned_here():
    w = World()
    w.add_obj("alarm_clock_2", "alarm_clock", "bedroom_dresser_1a")
    w.belief.objects["alarm_clock_2"].visible = True
    w.belief.objects["alarm_clock_1"].seen_from = "bedroom_dresser_1a"
    v = w.check("check_reachability", object_type="alarm_clock")
    assert v.args["candidates"] == ["alarm_clock_2", "alarm_clock_1"] and v.args["at"] == "bedroom_dresser_1a"
    assert w.check("check_reachability", object_type="alarm_clock", object_id="alarm_clock_1").args["candidates"] == \
        ["alarm_clock_1"]


# ---------------------------------------------------------------- directly preceded (PLAN 1.3 #17)
def test_pick_directly_preceded_allowed_interleavings():
    w = World()
    w.reach()
    w.run("speak", text="Got it.")
    w.run("list_locations")
    w.run("recall", query="mug")
    w.run("navigate", status="rejected", location="garage")           # rejected calls never ran
    v = w.check("manipulate", action="pick", object_type="alarm_clock")
    assert v.ok, v
    assert v.args["arm"] == "right" and v.args["object_id"] == "alarm_clock_1" and v.action == "pick"
    assert v.args["skill_id"] == "lite.pick.v0"


@pytest.mark.parametrize("between", [
    ("wait_and_observe", {"timeout_s": 0}),
    ("observe", {"mode": "glance"}),
    ("check_reachability", {"object_type": "book"}),
    ("navigate", {"location": "kitchen"}),
])
def test_pick_directly_preceded_forbidden_interleavings(between):
    w = World()
    w.reach()
    w.run(between[0], **between[1])
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "no_reachability",
           "pick without reachability check")


def test_pick_reach_goes_stale_after_motion_or_ttl():
    w = World()
    w.reach(t=95.0)
    w.last_motion_t = 96.0                                            # e.g. a scan or base shift since
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "no_reachability")
    w.last_motion_t = 50.0
    w.now = 95.0 + 31.0
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "no_reachability")


def test_pick_reachability_says_no_and_arm_mismatch():
    w = World()
    w.reach(reachable=False, reason="too_far")
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "not_reachable",
           "can't be reached from here (too_far)")
    w2 = World()
    w2.reach(arm="left")
    expect(w2.check("manipulate", action="pick", object_type="alarm_clock", arm="right"), Stage.STATE,
           "arm_mismatch", "reachability says to use the left arm")
    w3 = World()
    w3.reach(arm="either")
    w3.belief.set_hand("right", "book_1", "look", 1.0, verified=True)
    w3.belief.objects["book_1"].where = Fact("hand:right", "look", 1.0, True)
    v = w3.check("manipulate", action="pick", object_type="alarm_clock")
    assert v.ok and v.args["arm"] == "left"                           # either -> the free hand


def test_pick_hand_not_empty_just_placed_and_two_failures():
    w = World()
    w.reach()
    w.belief.mark_hand_unknown("right", "cancel", 99.0)
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "hand_not_empty",
           "call wait_and_observe(timeout_s=0) first")
    w = World()
    placed = w.run("manipulate", action="place", object_type="alarm_clock", t=90.0)
    placed.data = {"object_id": "alarm_clock_1"}
    w.reach()
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "just_placed")
    w = World()
    for _ in range(2):
        f = w.run("manipulate", status="failed", action="pick", object_type="alarm_clock", object_id="alarm_clock_1")
        f.data = {"object_id": "alarm_clock_1"}
    w.reach()
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "grasp_failed_twice")


# ---------------------------------------------------------------- reach_stance (PLAN 1.3 #26)
def test_reach_stance_needs_a_needs_reposition_check():
    w = World()
    expect(w.check("navigate", location="reach_stance"), Stage.STATE, "no_reach_stance")
    w.reach(reachable=False, reason="needs_reposition", stance={"x": 1.0, "y": 2.0, "yaw": 0.0, "dx": 0.2})
    v = w.check("navigate", location="reach_stance")
    assert v.ok and v.action == "reposition" and v.args["stance"]["dx"] == 0.2
    assert v.args["anchor"] == "bedroom_dresser_1a" and v.args["reach_execution_id"].startswith("rch-")
    w.task.intent_version += 1                                        # a new generation
    expect(w.check("navigate", location="reach_stance"), Stage.STATE, "no_reach_stance")


def test_reach_stance_after_base_motion_is_rejected_and_pick_needs_a_fresh_check():
    w = World()
    w.reach(reachable=False, reason="needs_reposition", stance={"x": 1.0, "y": 2.0, "yaw": 0.0}, t=98.0)
    w.last_motion_t = 98.5
    expect(w.check("navigate", location="reach_stance"), Stage.STATE, "no_reach_stance")
    w = World()
    w.reach(reachable=False, reason="needs_reposition", stance={"x": 1.0, "y": 2.0, "yaw": 0.0}, t=97.0)
    w.run("navigate", location="reach_stance", t=98.0)
    w.last_motion_t = 99.0
    expect(w.check("manipulate", action="pick", object_type="alarm_clock"), Stage.STATE, "no_reachability")


# ---------------------------------------------------------------- place (C13)
def hold(w, oid="alarm_clock_1", arm="right"):
    w.belief.set_hand(arm, oid, "look", 99.0, verified=True)
    w.belief.objects[oid].where = Fact(f"hand:{arm}", "look", 99.0, True)


def test_place_rules():
    w = World(at="living_room_table_1a")
    expect(w.check("manipulate", action="place", object_type="alarm_clock"), Stage.STATE, "not_holding",
           "not verified to hold a alarm_clock")
    hold(w)
    v = w.check("manipulate", action="place", object_type="alarm_clock", target="user")
    assert v.ok and v.args["target"] == "living_room_table_1a" and v.args["arm"] == "right"
    assert v.args["object_id"] == "alarm_clock_1"
    v = w.check("manipulate", action="place", object_type="alarm_clock")        # default: the surface here
    assert v.ok and v.args["target"] == "living_room_table_1a"
    expect(w.check("manipulate", action="place", object_type="alarm_clock", target="kitchen_counter_1a"),
           Stage.STATE, "target_not_here", "you are at living_room_table_1a; navigate to kitchen_counter_1a first")
    w2 = World(at="start")
    hold(w2)
    expect(w2.check("manipulate", action="place", object_type="alarm_clock"), Stage.STATE, "no_surface_here")


def test_place_goal_must_parse():
    w = World(at="living_room_table_1a")
    hold(w)
    ctx = w.ctx(layout_parse=lambda text: None if "nowhere" in text else object())
    v = validate(ToolCall("manipulate", {"action": "place", "object_type": "alarm_clock", "goal": "nowhere nice"}), ctx)
    expect(v, Stage.STATE, "goal_not_understood", "not understood")
    v = validate(ToolCall("manipulate", {"action": "place", "object_type": "alarm_clock", "goal": "on it"}), ctx)
    assert v.ok


# ---------------------------------------------------------------- CAPABILITY
def test_capability_stage_last():
    w = World(at="start")
    w.caps["navigation"] = ServiceHealth(False, "down", "deploy lost")
    expect(w.check("navigate", location="kitchen"), Stage.CAPABILITY, "nav_unhealthy",
           "navigation stack unavailable (deploy lost)")
    w.task.paused = True                                              # STATE before CAPABILITY
    expect(w.check("navigate", location="kitchen"), Stage.STATE, "paused")


def test_policy_unavailable_keeps_the_enum():
    w = World()
    w.reach()
    robot = FakeRobot(Clock())
    robot.policy_down = True
    w.registry = robot.registry()
    v = w.check("manipulate", action="pick", object_type="alarm_clock")
    expect(v, Stage.CAPABILITY, "policy_unavailable", "policy unavailable: lite.pick.v0")
    assert "alarm_clock" in w.registry.loaded_object_types()


def test_fresh_reach_helper_binds_the_instance():
    w = World()
    w.reach(object_id="alarm_clock_1")
    assert fresh_reach(w.ctx(), "alarm_clock")["object_id"] == "alarm_clock_1"
    assert fresh_reach(w.ctx(), "alarm_clock", "alarm_clock_2") is None
    assert fresh_reach(w.ctx(), "book") is None
