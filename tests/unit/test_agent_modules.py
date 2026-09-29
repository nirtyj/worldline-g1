"""The adapted agent modules: context bus + events, procedures.step_of, persona tools and timeouts,
narrator G1 lines, episodes KEEP, memory vocabulary, fused-state jitter, mutants, the planner's
ACTIONS rendering, Jev vocabulary, and the skill registry contract (api/skills.py)."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from agent.context import ContextBus
from agent.episodes import KEEP
from agent.fused_state import Directive, _fingerprint
from agent.memory import VOLATILE_TYPES, load_vocab
from agent.narrator import Narrator
from agent.persona import GOAL_TIMEOUT_S, STUCK_S, Persona
from agent.procedures import step_of, tasks_of
from agent.state import BeliefState
from api.events import ALIASES, EVENT_TYPES, event_type_for
from api.skills import BACKEND_ORDER, SkillSpec, StaticSkillRegistry, expand_types
from api.types import ServiceHealth
from sim.clock import SimClock
from tests.unit.fakes_rt import house_map


# ---------------------------------------------------------------- context events
def test_context_bus_keeps_rows_and_emits_doc_events():
    bus = ContextBus(SimClock(1.0), fence=lambda: (3, 7))
    seen = []
    bus.event_sinks.append(seen.append)
    bus.log("heard", id="u1", text="bring me the mug")
    bus.log("result", tool="navigate", status="succeeded")
    bus.log("goal_check", ok=True)
    assert [r["type"] for r in bus.rows] == ["heard", "result", "goal_check"]      # sinks still read rows
    assert [e.event_type for e in seen] == ["user_utterance", "tool_result", "goal_check"]
    assert seen[0].generation == 3 and seen[0].control_epoch == 7 and seen[0].seq == 1
    assert seen[0].payload["row_type"] == "heard" and seen[0].payload["text"] == "bring me the mug"
    assert bus.recent_events(2)[-1]["event_type"] == "goal_check"


def test_event_vocabulary():
    for t in ("user_utterance", "model_reasoning", "tool_call", "tool_started", "tool_result", "visual_observation",
              "speech_queued", "interruption", "rejection", "stale_result", "late_result", "safety_event",
              "capability_changed", "tool_state", "body_mode"):
        assert t in EVENT_TYPES
    assert event_type_for("decision") == "tool_call" and event_type_for("stop") == "interruption"
    assert event_type_for("narrate") == "narrate" and ALIASES["rejected"] == "rejection"


# ---------------------------------------------------------------- procedures
@pytest.mark.parametrize("row,step", [
    ({"type": "result", "tool": "check_reachability", "status": "succeeded", "data": {"reachable": True}},
     "check_reachability:ok"),
    ({"type": "result", "tool": "check_reachability", "status": "succeeded", "data": {"reason": "not_seen_here"}},
     "check_reachability:not_seen_here"),
    ({"type": "result", "tool": "manipulate", "action": "pick", "status": "succeeded"}, "manipulate:pick:ok"),
    ({"type": "result", "tool": "manipulate", "action": "place", "status": "failed"}, "manipulate:place:failed"),
    ({"type": "result", "tool": "navigate", "status": "succeeded"}, "navigate:ok"),
    ({"type": "result", "tool": "observe", "status": "succeeded", "source": "harness", "why": "arrival"}, "look"),
    ({"type": "result", "tool": "observe", "status": "succeeded", "source": "harness", "why": "verify pick"}, "verify"),
    ({"type": "result", "tool": "observe", "status": "succeeded", "source": "harness", "why": "wait_and_observe"}, None),
    ({"type": "result", "tool": "wait_and_observe", "status": "succeeded"}, "look"),
    ({"type": "result", "tool": "list_locations", "status": "succeeded"}, "list_locations"),
    ({"type": "rejected", "tool": "manipulate"}, "rejected:manipulate"),
    ({"type": "say_queued", "text": "Which one?"}, "ask"),
    ({"type": "say_queued", "text": "On it."}, "speak"),
])
def test_step_of(row, step):
    assert step_of(row) == step


def test_tasks_of_counts_navigate_and_manipulate():
    rows = [{"type": "session_start", "deliver_to": "table_1a"}, {"type": "heard", "id": "u1", "text": "bring it"},
            {"type": "classified", "id": "u1", "kind": "request"},
            {"type": "result", "tool": "navigate", "status": "succeeded"},
            {"type": "result", "tool": "manipulate", "action": "pick", "status": "succeeded"},
            {"type": "delivered", "object": "mug_1"}]
    tasks = tasks_of(rows)
    assert len(tasks) == 1 and tasks[0]["ok"] and tasks[0]["steps"] == ["navigate:ok", "manipulate:pick:ok"]


# ---------------------------------------------------------------- persona
def test_persona_uses_the_new_tools_and_humanoid_timeouts():
    assert (GOAL_TIMEOUT_S, STUCK_S) == (150.0, 20.0)
    p = Persona("optimize")
    b = BeliefState()
    b.set_pose("start", None, "odometry", 0.0)
    goals = p.drives(b, house_map(), 100.0)
    assert goals[0][1] == "ask" and goals[0][4] == ("speak",)
    p.permission = "granted"
    goals = {g[1]: g[4] for g in p.drives(b, house_map(), 100.0)}
    assert goals["map"] == ("navigate", "wait_and_observe", "speak")
    assert goals["ready"] == ("navigate", "speak")
    p2 = Persona("medium")
    glance = [g for g in p2.drives(b, house_map(), 100.0) if g[1] == "glance"]
    assert glance and glance[0][4] == ("wait_and_observe",) and "wait_and_observe(timeout_s=0)" in glance[0][3]
    all_tools = {t for g in p.drives(b, house_map(), 100.0) for t in g[4]}
    assert not all_tools & {"say", "look", "pick", "place", "manipulate"}


def test_persona_ready_compares_the_resolved_user_keypoint():
    p = Persona("optimize")
    p.permission = "granted"
    b = BeliefState()
    b.set_pose("kitchen_counter_1a", None, "odometry", 0.0)
    ready = [g for g in p.drives(b, house_map(), 1.0) if g[1] == "ready"][0]
    assert ready[2] == "living_room_table_1a"                      # never the alias "user"


# ---------------------------------------------------------------- narrator
def narrator(executors):
    m = dict(house_map(), executors=executors)
    return Narrator(m, BeliefState(), "living_room_table_1a")


def test_narrator_g1_lines():
    n = narrator({"navigate": "sonic_walk"})
    n.row({"type": "started", "tool": "navigate", "action": "keypoint", "args": {"location": "kitchen_counter_1a"}})
    n.row({"type": "started", "tool": "navigate", "action": "keypoint", "args": {"location": "kitchen_counter_1b"}})
    n.row({"type": "started", "tool": "manipulate", "action": "pick",
           "args": {"object_type": "apple", "skill_id": "sonic.script.pick.v0"}})
    n.row({"type": "started", "tool": "manipulate", "action": "pick",
           "args": {"object_type": "bottle", "skill_id": "groot.pick.bottle.cloudwalk.v0"}})
    n.row({"type": "started", "tool": "navigate", "action": "reposition", "args": {"location": "reach_stance"}})
    n.row({"type": "safety_event", "kind": "fell"})
    lines = [t for _, t in n._out]
    assert lines.count("Walking (SONIC)") == 1
    assert "[fallback] attaching" in lines and "Grasping with GR00T" in lines
    assert "Stepping into reach" in lines and "Fell" in lines
    k = narrator({"navigate": "kinematic_nav"})
    k.row({"type": "started", "tool": "navigate", "action": "keypoint", "args": {"location": "kitchen_counter_1a"}})
    assert any(t.startswith("[fallback]") for _, t in k._out)


def test_narrator_results_use_lowercase_statuses():
    n = narrator({})
    n.row({"type": "result", "tool": "navigate", "status": "succeeded", "data": {"at": "kitchen_counter_1a"}})
    n.row({"type": "result", "tool": "navigate", "status": "cancelled", "data": {"reason": "cancelled"}})
    n.row({"type": "result", "tool": "manipulate", "action": "pick", "status": "failed", "data": {"reason": "grasp_failed"}})
    n.row({"type": "result", "tool": "check_reachability", "status": "succeeded",
           "data": {"reachable": False, "reason": "needs_reposition"}})
    lines = [t for _, t in n._out]
    assert lines[0].startswith("At the") and lines[1].startswith("Stopped on the way (cancelled)")
    assert "The pick didn't work (grasp failed)" in lines and "Almost in reach; stepping closer" in lines


# ---------------------------------------------------------------- episodes, memory, fused state
def test_episode_keep_has_the_g1_rows():
    assert {"tool_result", "rejection", "visual_observation", "goal_check", "body_mode", "safety_event",
            "stale_result"} <= KEEP


def test_memory_vocabulary_merges_the_fixed_list(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("# c\nversion: 1\ntypes:\n  - alarm_clock\n  - banana      # note\n  - soap_bar\nother: 1\n")
    assert load_vocab(p) == {"alarm_clock", "banana", "soap_bar"}
    assert load_vocab(tmp_path / "missing.yaml") == set()
    assert "alarm_clock" in VOLATILE_TYPES


def test_fused_fingerprint_ignores_rtf_jitter_and_directive_has_replaces_task():
    a = {"pose": {"x": 1}, "body": {"mode": "HOLD", "rtf": 0.99, "gdebug_age_ms": 12}}
    b = {"pose": {"x": 1}, "body": {"mode": "HOLD", "rtf": 1.02, "gdebug_age_ms": 40}}
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint({"pose": {"x": 1}, "body": {"mode": "LOCOMOTION", "rtf": 1.0}})
    d = Directive("d-u1", 1.0, "correction", "no, the apple", replaces_task=True)
    assert d.to_dict()["replaces_task"] is True


# ---------------------------------------------------------------- mutants
def test_every_mutant_constructs():
    from agent import mutants
    from tests.unit.fakes_rt import FakeRobot, FakeUser, ScriptBrain
    clock = SimClock(20.0)
    for name in ("no_drop_speech", "no_keyword_stop", "forget_cancelled_grasp", "trust_success",
                 "no_wait_for_chunk", "cancel_on_everything", "no_stale_check"):
        rt = getattr(mutants, name)(FakeRobot(clock), FakeUser(clock), ScriptBrain([]), clock)
        assert rt.tools_ctx.skill_types


def test_no_wait_for_chunk_ignores_a_busy_body():
    from agent.mutants import NoWaitForChunk
    from agent.harness import Runtime
    from api.tools import ToolCall
    from tests.unit.fakes_rt import FakeRobot, FakeUser, ScriptBrain
    clock = SimClock(20.0)
    for cls, ok in ((Runtime, False), (NoWaitForChunk, True)):
        rt = cls(FakeRobot(clock), FakeUser(clock), ScriptBrain([]), clock)
        rt.task.utterances.append(SimpleNamespace(id="u1", text="go", t_end=0.0))
        rt._new_exec("navigate", {"location": "kitchen"}, status="running")
        v = rt._check(ToolCall("navigate", {"location": "bedroom"}))
        assert v.ok is ok, (cls.__name__, v)


# ---------------------------------------------------------------- the planner's ACTIONS lines
def test_actions_lines_show_generation_id_executor_fallback_and_late():
    from agent.model import ReferenceBrain
    from api.execution import ExecutionManager
    from api.results import finish
    m = ExecutionManager()
    nav = m.create("navigate", {"location": "kitchen_counter_1a"}, generation=2, control_epoch=1, t=12.0)
    nav.status = "succeeded"
    nav.result = finish(nav, "succeeded", {"location": "kitchen_counter_1a", "at": "kitchen_counter_1a",
                                           "executor": "kinematic_nav", "path_len_m": 3.0, "duration_s": 8.0})
    nav.data = dict(nav.result.data)
    late = m.create("manipulate", {"action": "pick", "object_type": "apple"}, generation=1, control_epoch=1, t=13.0)
    late.status = "cancelled"
    import dataclasses
    late.result = dataclasses.replace(finish(late, "cancelled", {"reason": "cancelled", "skill": "lite.pick.v0",
                                                                  "executor": "lite", "object_type": "apple"}),
                                      late=True)
    late.data = {**late.result.data, "late": True}
    spk = m.create("speak", {"text": "hi"}, generation=2, control_epoch=1)
    ctx = SimpleNamespace(history=[nav, late, spk])
    text = ReferenceBrain(None, house_map())._actions(ctx)
    lines = text.splitlines()
    assert len(lines) == 2                                         # speech is in CONVERSATION, not ACTIONS
    assert re.match(r"^\[\s*12\.0\] g2 nav-000001 navigate\(location=kitchen_counter_1a\) -> succeeded "
                    r"\[kinematic_nav, fallback\] arrived at kitchen_counter_1a", lines[0]), lines[0]
    assert "g1 man-000002 manipulate(action=pick, object_type=apple) -> cancelled" in lines[1]
    assert "(late: from an earlier request)" in lines[1] and "fallback" in lines[1]


# ---------------------------------------------------------------- skills registry contract
def test_registry_enum_from_loaded_skills_and_fixed_vocab():
    vocab = ("alarm_clock", "apple", "banana", "bottle")
    skills = [SkillSpec("groot.pick.bottle.cloudwalk.v0", "pick", ("bottle",), ("right",), "groot", "experimental",
                        "unofficial", hand_type="inspire"),
              SkillSpec("groot.pick.household.isaac.v1", "pick", ("alarm_clock", "apple"), backend="groot",
                        status="planned"),
              SkillSpec("sonic.script.pick.v0", "pick", ("@vocab:pickupable",), backend="sonic_arm_script",
                        label="stepping_stone", max_duration_s=12.0),
              SkillSpec("sonic.script.place.v0", "place", ("@vocab:pickupable",), backend="sonic_arm_script",
                        label="stepping_stone")]
    down = {"groot.pick.bottle.cloudwalk.v0"}
    full = StaticSkillRegistry(skills, vocab=vocab, backend_order=BACKEND_ORDER["full"],
                               health_fn=lambda s: ServiceHealth(s.skill_id not in down, "down" if s.skill_id in down
                                                                 else "ok", "PolicyServer ping failed"))
    assert full.loaded_object_types() == ["alarm_clock", "apple", "banana", "bottle"]     # planned skill not loaded
    assert full.select("pick", "bottle", "right").backend == "groot"                      # groot first in full
    s, why = full.select_healthy("pick", "bottle", "right")
    assert s.skill_id == "sonic.script.pick.v0" and why == ""                             # groot_then_script
    assert full.select("pick", "bottle", "left").backend == "sonic_arm_script"            # groot is right-only
    assert full.get("groot.pick.bottle.cloudwalk.v0").hand_mismatch
    assert full.get("sonic.script.pick.v0").timeout_s() == 22.0 and full.get("sonic.script.pick.v0").stepping_stone
    sonic = StaticSkillRegistry(skills, vocab=vocab, backend_order=BACKEND_ORDER["sonic"])
    assert sonic.loaded_object_types() == sorted(vocab)                                   # the fixed vocabulary
    no_bottle = StaticSkillRegistry(skills, vocab=vocab[:3], backend_order=BACKEND_ORDER["full"],
                                    health_fn=lambda s: ServiceHealth(s.backend != "groot", "down"))
    assert no_bottle.select_healthy("pick", "bottle", "right")[0] is None                 # -> policy unavailable
    groot_only = StaticSkillRegistry(skills, vocab=vocab, backend_order=BACKEND_ORDER["full"], only=("groot",))
    assert groot_only.loaded_object_types() == ["bottle"]
    assert expand_types(("@vocab:pickupable", "mug"), vocab) == ("alarm_clock", "apple", "banana", "bottle", "mug")


# ---------------------------------------------------------------- Jev vocabulary
def test_jev_vocabulary_grows_with_the_registry():
    from brains import system1_jev
    before = set(system1_jev.TARGET_WORDS)
    after = set(system1_jev.set_vocabulary(["aluminum_foil", "soap_bottle"]))
    assert {"aluminum_foil", "soap_bottle"} <= after and before <= after


def test_observer_prompt_has_the_own_body_line():
    from brains.system1 import OBSERVER_SYSTEM
    assert "The robot's own arms, hands and anything they hold may appear at the edges of the frame; never report them." \
        in OBSERVER_SYSTEM
