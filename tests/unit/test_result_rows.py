"""The trace's `result` rows (M2a verifier findings 2 and 3): every result row, whatever produced it, carries the
envelope identity and fences (api.results.RESULT_ROW_FIELDS), and speech, recall and rejections produce result rows
too (with a `kind`). The eval and the page check exactly these field names."""

from __future__ import annotations

import pytest

from agent.harness import Runtime
from api.results import RESULT_ROW_FIELDS, RESULT_ROW_KINDS
from api.tools import ToolCall
from sim.clock import SimClock
from tests.unit.conftest import keep_running, run_until, stop
from tests.unit.fakes_rt import FakeRobot, FakeUser, PolicyBrain, ScriptBrain

SPEED = 20.0


def _check_rows(rt) -> list[dict]:
    res = [r for r in rt.tracer.rows if r["type"] == "result"]
    assert res
    for r in res:
        missing = [k for k in RESULT_ROW_FIELDS if k not in r]
        assert not missing, (r["tool"], r.get("kind"), missing)
        assert r["kind"] in RESULT_ROW_KINDS, r
        assert r["observation_id"], (r["tool"], r["kind"])          # doc 42/51: every result carries one
        assert isinstance(r["generation"], int) and isinstance(r["control_epoch"], int), r
        assert isinstance(r["t_start"], (int, float)) and isinstance(r["t_end"], (int, float)), r
        assert r["t_end"] >= r["t_start"] - 1e-6, r
        assert r["late"] in (True, False)
        assert r["status"] == str(r["status"]).lower()
        if r["kind"] != "rejection" or r.get("stage") != "schema":
            assert r["execution_id"], r
        assert r["skill"] == r["tool"]                              # the old name stays for older readers
    # one row per finished execution, and nothing written twice
    ids = [r["execution_id"] for r in res if r["execution_id"]]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    return res


@pytest.mark.asyncio
async def test_every_result_row_has_the_fields_on_an_f1_fetch():
    clock = SimClock(SPEED)
    robot = FakeRobot(clock)
    user = FakeUser(clock)
    rt = Runtime(robot, user, PolicyBrain("alarm_clock"), clock)
    user.say("bring me the alarm clock")
    try:
        await run_until(rt, lambda: any(r["type"] == "delivered" for r in rt.tracer.rows), wall_s=15,
                        what="delivered")
        await keep_running(rt, lambda: not rt.speech.busy() and not rt.actions, wall_s=10, what="settled")
    finally:
        await stop(rt)
    res = _check_rows(rt)
    kinds = {r["kind"] for r in res}
    assert {"tool", "speech"} <= kinds
    tools = {r["tool"] for r in res if r["kind"] == "tool"}
    assert {"navigate", "observe", "check_reachability", "manipulate"} <= tools
    speech = [r for r in res if r["kind"] == "speech"]
    assert all(r["tool"] == "speak" and r.get("text") for r in speech)
    # the rows agree with the stored envelopes
    by_id = {e.execution_id: e.result for e in rt.history if e.result is not None}
    for r in res:
        env = by_id.get(r["execution_id"])
        if env is None:
            continue
        assert (r["status"], r["generation"], r["control_epoch"], r["observation_id"]) == \
            (env.status, env.generation, env.control_epoch, env.observation_id), r["execution_id"]


@pytest.mark.asyncio
async def test_list_locations_recall_and_rejections_are_result_rows():
    clock = SimClock(SPEED)
    robot = FakeRobot(clock)
    user = FakeUser(clock)
    brain = ScriptBrain([ToolCall("list_locations", {}),
                         ToolCall("recall", {"query": "where is the alarm clock?"}),
                         ToolCall("navigate", {"location": "the moon"}),                  # ENUM rejection
                         ToolCall("manipulate", {"action": "pick", "object_type": "alarm_clock"}),  # STATE
                         ToolCall("speak", {"text": "Done."})])
    rt = Runtime(robot, user, brain, clock)
    user.say("where are you?")
    try:
        await run_until(rt, lambda: any(r["type"] == "result" and r.get("kind") == "speech"
                                        and r.get("text") == "Done." for r in rt.tracer.rows),
                        wall_s=15, what="the last line played")
    finally:
        await stop(rt)
    res = _check_rows(rt)
    locs = [r for r in res if r["tool"] == "list_locations"]
    assert locs and locs[0]["kind"] == "tool" and locs[0]["observation_id"].startswith("obs-")
    rec = [r for r in res if r["tool"] == "recall"]
    assert rec and rec[0]["kind"] == "recall" and rec[0]["data"]["answer"]
    rej = [r for r in res if r["kind"] == "rejection"]
    assert {r["tool"] for r in rej} >= {"navigate", "manipulate"}
    assert all(r["status"] == "rejected" and r.get("stage") and r.get("code") for r in rej)
    pick = next(r for r in rej if r["tool"] == "manipulate")
    assert pick["action"] == "pick"
    # the old rows are still written (the page and the procedures read them)
    assert any(r["type"] == "rejected" for r in rt.tracer.rows)
    assert any(r["type"] == "recall" for r in rt.tracer.rows)


def test_procedures_and_narrator_skip_non_tool_rows():
    from agent.narrator import Narrator
    from agent.procedures import step_of
    assert step_of({"type": "result", "kind": "speech", "tool": "speak", "status": "succeeded"}) is None
    assert step_of({"type": "result", "kind": "rejection", "tool": "manipulate", "status": "rejected"}) is None
    assert step_of({"type": "result", "kind": "recall", "tool": "recall", "status": "succeeded"}) is None
    assert step_of({"type": "result", "kind": "tool", "tool": "navigate", "status": "succeeded"}) == "navigate:ok"
    assert step_of({"type": "result", "tool": "navigate", "status": "succeeded"}) == "navigate:ok"   # older rows
    from tests.unit.fakes_rt import house_map
    n = Narrator(house_map(), None, None)
    n.row({"type": "result", "kind": "rejection", "tool": "manipulate", "action": "pick", "status": "rejected",
           "data": {"reason": "no reachability"}})
    n.row({"type": "result", "kind": "speech", "tool": "speak", "status": "succeeded", "data": {}})
    assert n._out == [], "a rejection is not narrated as a failed grasp, and speech is not a result line"
    n.row({"type": "result", "kind": "tool", "tool": "manipulate", "action": "pick", "status": "failed",
           "data": {"reason": "grasp_failed"}})
    assert [t for _, t in n._out] == ["The pick didn't work (grasp failed)"]
