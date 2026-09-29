"""The harness on the new API against the fake robot: F1 fetch, correction (generation bump,
late result), stop/resume (halt first), reach_stance handoff (G14), wait_and_observe,
rejections as tool results, capability rejection, stale decisions and multi-call rejection."""

from __future__ import annotations

import asyncio

import pytest

from agent.harness import Runtime
from api.results import validate_envelope
from api.tools import ToolCall
from sim.clock import SimClock
from tests.unit.conftest import keep_running, run_until, stop
from tests.unit.fakes_rt import FakeRobot, FakeUser, PolicyBrain, ScriptBrain

SPEED = 20.0


def make(brain=None, **robot_kw):
    clock = SimClock(SPEED)
    robot = FakeRobot(clock, **robot_kw)
    user = FakeUser(clock)
    brain = brain or PolicyBrain("alarm_clock")
    rt = Runtime(robot, user, brain, clock)
    return rt, robot, user, brain


def rows(rt, type_=None, **match):
    out = []
    for r in rt.tracer.rows:
        if type_ is not None and r["type"] != type_:
            continue
        if all(r.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def results(rt):
    return [e.result for e in rt.history if e.result is not None]


@pytest.mark.asyncio
async def test_f1_fetch_delivers_and_every_result_is_a_valid_envelope():
    rt, robot, user, brain = make()
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: rows(rt, "delivered"), wall_s=15, what="delivered")
        await keep_running(rt, lambda: "Here it is." in robot.said, wall_s=10, what="done line")
    finally:
        await stop(rt)
    tools = [e.tool_name for e in rt.history]
    # navigate -> arrival scan -> check_reachability -> manipulate(pick) -> verify glance -> navigate(user) ...
    assert "check_reachability" in tools and tools.count("manipulate") == 2
    picks = [e for e in rt.history if e.tool_name == "manipulate" and e.action == "pick"]
    assert picks[0].status == "succeeded" and picks[0].args["arm"] == "right"   # arm defaulted from reachability
    assert picks[0].args["object_id"] == "alarm_clock_1"                         # bound by the reachability
    arrivals = [e for e in rt.history if e.tool_name == "observe" and e.args.get("why") == "arrival"]
    assert arrivals and all(e.action == "scan" for e in arrivals)
    verifies = [e for e in rt.history if e.tool_name == "observe" and str(e.args.get("why")).startswith("verify")]
    assert verifies and verifies[0].action == "glance"
    assert robot.objects["alarm_clock_1"].where == "living_room_table_1a"
    assert rt.belief.objects["alarm_clock_1"].where.value == "living_room_table_1a"
    assert rows(rt, "place_learned", object="alarm_clock_1")
    for res in results(rt):
        assert res.observation_id, res                      # doc 42/51: every result carries one
        assert validate_envelope(res) == [], (res.tool, validate_envelope(res))
    ids = [e.execution_id for e in rt.history]
    assert len(set(ids)) == len(ids)
    assert all(i.split("-")[0] in ("spk", "loc", "nav", "rch", "man", "wai", "rec", "obs") for i in ids)
    # the user alias was resolved before the execution was created
    navs = [e for e in rt.history if e.tool_name == "navigate"]
    assert navs[-1].args["location"] == "living_room_table_1a"


@pytest.mark.asyncio
async def test_correction_cancels_marks_late_and_bumps_generation():
    rt, robot, user, brain = make(nav_s_per_m=4.0)       # a slow walk to the bedroom
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "navigate" and not e.finished for e in rt.history),
                        what="navigate running")
        g0 = rt.task.intent_version
        e0 = rt.task.control_epoch
        brain.want = "apple"
        user.say("no, the apple instead", directive=None)
        await keep_running(rt, lambda: any(e.tool_name == "navigate" and e.status == "cancelled"
                                           for e in rt.history), what="walk cancelled")
    finally:
        await stop(rt)
    nav = next(e for e in rt.history if e.tool_name == "navigate" and e.status == "cancelled")
    assert rt.task.intent_version == g0 + 1 and rt.task.control_epoch >= e0 + 1
    assert nav.generation == g0
    assert nav.result.late is True and nav.data["late"] is True        # I5: late, world information only
    assert rows(rt, "late_result", execution_id=nav.execution_id)
    assert rt.belief.between is not None or rt.belief.robot_at.value is None
    assert rows(rt, "correction")


@pytest.mark.asyncio
async def test_stop_halts_first_acks_once_and_resume_bumps_epoch():
    rt, robot, user, brain = make(nav_s_per_m=4.0)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "navigate" and not e.finished for e in rt.history),
                        what="navigate running")
        user.say("stop")
        await keep_running(rt, lambda: rt.task.paused and robot.halts, what="halted")
        await keep_running(rt, lambda: "I've stopped." in robot.said, what="ack")
        epoch = rt.task.control_epoch
        user.say("stop")                                   # a second stop: halt again, no second ack
        await asyncio.sleep(0.2)
        user.say("okay, carry on")
        await keep_running(rt, lambda: not rt.task.paused, what="resumed")
    finally:
        await stop(rt)
    stop_row = rows(rt, "stop")[0]
    assert stop_row["physical_first"] is True and stop_row["receipt"]["stopped"] is True
    assert robot.said.count("I've stopped.") == 1
    assert len(robot.halts) >= 2
    assert rt.task.control_epoch > epoch and robot.resumes and robot.resumes[-1] == rt.task.control_epoch
    nav = next(e for e in rt.history if e.tool_name == "navigate")
    assert nav.status == "failed" and nav.data.get("reason") == "halted"


@pytest.mark.asyncio
async def test_reach_stance_handoff_then_fresh_check_then_pick():
    rt, robot, user, brain = make()
    robot.objects["alarm_clock_1"].needs_reposition = True
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "manipulate" and e.action == "pick" and e.finished
                                        for e in rt.history), wall_s=15, what="pick")
    finally:
        await stop(rt)
    seq = [(e.tool_name, e.action, (e.data or {}).get("reason")) for e in rt.history
           if e.tool_name in ("check_reachability", "navigate", "manipulate") and e.status != "rejected"]
    i = seq.index(("check_reachability", None, "needs_reposition"))
    assert seq[i + 1][:2] == ("navigate", "reposition")
    assert seq[i + 2][0] == "check_reachability" and seq[i + 3][:2] == ("manipulate", "pick")
    repo = next(e for e in rt.history if e.action == "reposition")
    assert repo.args["stance"]["dx"] == pytest.approx(0.22) and repo.args["anchor"] == "bedroom_dresser_1a"
    # the reposition was followed by a glance, not an arrival scan
    after = rt.history[rt.history.index(repo) + 1]
    assert after.tool_name == "observe" and after.action == "glance"


@pytest.mark.asyncio
async def test_pick_without_reachability_is_a_rejected_tool_result():
    brain = ScriptBrain([ToolCall("navigate", {"location": "bedroom_dresser_1a"}),
                         ToolCall("manipulate", {"action": "pick", "object_type": "alarm_clock"})])
    rt, robot, user, _ = make(brain)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: rows(rt, "rejected"), wall_s=10, what="rejection")
    finally:
        await stop(rt)
    rej = next(e for e in rt.history if e.status == "rejected")
    assert rej.tool_name == "manipulate" and rej.result.data["stage"] == "state"
    assert rej.result.data["code"] == "no_reachability"
    assert "pick without reachability check" in rej.result.summary
    assert rej.result.execution_id == rej.execution_id and rej.result.observation_id
    assert "rejected manipulate" in (rt._note or "")
    assert not [e for e in robot.started if e.tool_name == "manipulate"]


@pytest.mark.asyncio
async def test_unknown_location_and_two_calls_are_rejected():
    brain = ScriptBrain([ToolCall("navigate", {"location": "garage"}),
                         ToolCall("speak", {"text": "Hello."}, extra=[ToolCall("navigate", {"location": "kitchen"})])])
    rt, robot, user, _ = make(brain)
    user.say("hello")
    try:
        await run_until(rt, lambda: len(rows(rt, "rejected")) >= 2, wall_s=10, what="two rejections")
    finally:
        await stop(rt)
    rej = [e.result for e in rt.history if e.status == "rejected"]
    assert rej[0].data["code"] == "unknown_location" and "nearest names" in rej[0].data["reason"]
    assert rej[1].data["code"] == "two_calls" and rej[1].data["reason"] == "two action tools in one turn"
    assert rej[1].execution_id is None                     # schema-stage rejections carry no execution_id


@pytest.mark.asyncio
async def test_policy_down_is_a_capability_rejection_and_the_enum_is_unchanged():
    rt, robot, user, brain = make()
    before = list(rt.tools_ctx.skill_types)
    robot.policy_down = True
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.status == "rejected" and e.tool_name == "manipulate" for e in rt.history),
                        wall_s=15, what="capability rejection")
    finally:
        await stop(rt)
    rej = next(e for e in rt.history if e.status == "rejected" and e.tool_name == "manipulate")
    assert rej.result.data["stage"] == "capability" and rej.result.data["reason"].startswith("policy unavailable")
    assert list(rt.tools_ctx.skill_types) == before and "alarm_clock" in before   # Invariant 9


@pytest.mark.asyncio
async def test_wait_and_observe_zero_is_unchanged_and_wakes_on_utterance():
    brain = ScriptBrain([ToolCall("wait_and_observe", {"timeout_s": 0}),
                         ToolCall("wait_and_observe", {"timeout_s": 30, "reason": "waiting for the user"})],
                        kinds={"hi": "request", "what is on the table?": "question"})
    rt, robot, user, _ = make(brain)
    user.say("hi")
    try:
        await run_until(rt, lambda: len([e for e in rt.history if e.tool_name == "wait_and_observe"]) >= 2
                        and not [e for e in rt.history if e.tool_name == "wait_and_observe"][-1].finished or
                        len([e for e in rt.history if e.tool_name == "wait_and_observe" and e.finished]) >= 2,
                        wall_s=10, what="second wait running")
        user.say("what is on the table?")
        await keep_running(rt, lambda: len([e for e in rt.history if e.tool_name == "wait_and_observe"
                                            and e.finished]) >= 2, what="second wait done")
    finally:
        await stop(rt)
    waits = [e for e in rt.history if e.tool_name == "wait_and_observe"]
    first, second = waits[0].result, waits[1].result
    # timeout 0: either the first look changed belief (it saw the table) or nothing changed -> succeeded
    assert first.status == "succeeded" and first.data["status"] in ("changed", "unchanged")
    assert second.status == "succeeded" and second.data["status"] == "changed"
    # every wait started with an observation
    obs_for_wait = [e for e in rt.history if e.tool_name == "observe" and e.args.get("why") == "wait_and_observe"]
    assert len(obs_for_wait) >= 2


@pytest.mark.asyncio
async def test_wait_during_manipulate_glances_and_never_takes_the_body():
    calls = [ToolCall("navigate", {"location": "bedroom_dresser_1a"}),
             ToolCall("check_reachability", {"object_type": "alarm_clock"}),
             ToolCall("manipulate", {"action": "pick", "object_type": "alarm_clock"})]
    brain = ScriptBrain(calls)
    rt, robot, user, _ = make(brain)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "manipulate" and not e.finished for e in rt.history),
                        wall_s=10, what="pick running")
        assert rt.tool_state() == "MANIPULATING"
        await rt._observe("wait_and_observe", mode=rt._choose_observe_mode())
    finally:
        await stop(rt)
    obs = [e for e in rt.history if e.tool_name == "observe" and e.args.get("why") == "wait_and_observe"]
    assert obs and obs[-1].action == "glance" and "body" not in obs[-1].resources


@pytest.mark.asyncio
async def test_stale_decision_is_dropped_when_the_generation_moves():
    clock = SimClock(SPEED)
    robot = FakeRobot(clock)
    user = FakeUser(clock)
    brain = ScriptBrain([ToolCall("navigate", {"location": "kitchen_counter_1a"})], delay=2.0, clock=clock,
                        kinds={"bring me the apple": "request", "no, the book instead": "correction"})
    rt = Runtime(robot, user, brain, clock)
    user.say("bring me the apple")
    try:
        await run_until(rt, lambda: brain.contexts, what="brain asked")
        user.say("no, the book instead")
        await keep_running(rt, lambda: rows(rt, "stale_decision"), what="stale decision")
    finally:
        await stop(rt)
    assert rows(rt, "stale_decision")[0]["tool"] == "navigate"
    assert not [e for e in robot.started if e.tool_name == "navigate"]


@pytest.mark.asyncio
async def test_prompt_never_renders_robot_state_or_perception():
    from agent.model import ReferenceBrain
    rt, robot, user, brain = make()
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: len(brain.contexts) >= 3, wall_s=10, what="a few decisions")
    finally:
        await stop(rt)
    ctx = brain.contexts[-1]
    ctx.robot_state = {"SECRET_ROBOT_STATE": 1}
    ctx.perception = {"objects": {"SECRET_GT_OBJECT": {}}}
    text = ReferenceBrain(None, ctx.map).render(ctx)
    assert "SECRET_ROBOT_STATE" not in text and "SECRET_GT_OBJECT" not in text
    assert "ACTIONS" in text and "g1 " in text or "g0 " in text


@pytest.mark.asyncio
async def test_quiet_rule_two_no_change_waits_then_sleep():
    brain = ScriptBrain([ToolCall("wait_and_observe", {"timeout_s": 0})] * 10, kinds={"hmm": "chitchat"})
    rt, robot, user, _ = make(brain)
    user.say("hmm")                                         # an open, unanswered utterance
    try:
        await run_until(rt, lambda: any(r["type"] == "wait_quiet" for r in rt.tracer.rows), what="quiet")
        await asyncio.sleep(0.3)
    finally:
        await stop(rt)
    waits = [e for e in rt.history if e.tool_name == "wait_and_observe"]
    assert 1 <= len(waits) <= 2                             # rule 7: no new information -> stop asking
    assert all(w.result.status == "succeeded" for w in waits)


@pytest.mark.asyncio
async def test_correction_during_a_pick_is_late_world_information_and_the_hand_is_reconciled():
    calls = [ToolCall("navigate", {"location": "bedroom_dresser_1a"}),
             ToolCall("check_reachability", {"object_type": "alarm_clock"}),
             ToolCall("manipulate", {"action": "pick", "object_type": "alarm_clock"})]
    brain = ScriptBrain(calls, kinds={"bring me the alarm clock": "request", "no, the apple instead": "correction"})
    rt, robot, user, _ = make(brain)
    robot.ignore_cancel = True                              # the grasp finishes anyway (a chunk can't be cut)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "manipulate" and not e.finished for e in rt.history),
                        wall_s=10, what="pick running")
        user.say("no, the apple instead")
        await keep_running(rt, lambda: any(r["type"] == "reconcile_done" for r in rt.tracer.rows), wall_s=10,
                           what="reconciled")
    finally:
        await stop(rt)
    pick = next(e for e in rt.history if e.tool_name == "manipulate")
    assert pick.generation < rt.task.intent_version and pick.result.late is True
    assert pick.status == "succeeded" and robot.hands["right"] == "alarm_clock_1"      # it really holds it
    # the late success was not taken as progress: hand UNKNOWN, then the reconcile observation verified it
    unknown = [r for r in rt.tracer.rows if r["type"] == "reconcile_start"]
    assert unknown and any(e.tool_name == "observe" and e.args.get("why") == "after cancel" for e in rt.history)
    assert rt.belief.holding["right"].value == "alarm_clock_1" and rt.belief.holding["right"].verified
    assert not [r for r in rt.tracer.rows if r["type"] == "delivered"]


@pytest.mark.asyncio
async def test_a_fall_is_a_safety_event_that_pauses_and_says_so_once():
    rt, robot, user, brain = make(nav_s_per_m=4.0)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(e.tool_name == "navigate" and not e.finished for e in rt.history),
                        what="walking")
        robot.emit({"type": "safety_event", "kind": "fell", "t": 1.23})     # EventLog-shaped: carries its own t
        ack = "I've lost my balance; I'm stopping until I'm steady."
        await keep_running(rt, lambda: rt.task.paused and ack in robot.said, what="paused and said")
        robot.emit({"type": "capability_changed", "capability": "manipulation", "detail": "policy server down"})
        robot.emit({"type": "body_mode", "mode": "FAULT"})
        await keep_running(rt, lambda: rt.tool_state() == "FAULT", what="fault state")
    finally:
        await stop(rt)
    assert [r for r in rt.tracer.rows if r["type"] == "safety_event"][0]["kind"] == "fell"
    assert robot.halts and "I've lost my balance; I'm stopping until I'm steady." in robot.said
    assert "policy server down" in (rt._note or "") or any(r["type"] == "capability_changed" for r in rt.tracer.rows)
    assert not robot.estops                                  # a fall never sends command{stop} (invariant 1)
