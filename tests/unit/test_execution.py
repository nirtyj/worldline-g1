"""api/execution.py: invariants I1-I5, ids, generations/epochs (fence rules, stale drop) and the
compatibility of Execution with the THOR-era HistoryEntry uses (test_execution_compat)."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from api.execution import (GENERATION_RULES, PREFIX, Execution, ExecutionManager, Fence, Rejected, ResultHandle,
                           default_action, is_stale_event, should_cancel)
from api.results import finish


def mgr():
    return ExecutionManager()


def test_ids_prefixes_and_one_sequence():
    m = mgr()
    a = m.create("navigate", {"location": "k"}, generation=1, control_epoch=0)
    b = m.create("manipulate", {"action": "pick", "object_type": "mug"}, generation=1, control_epoch=0)
    c = m.create("observe", {"mode": "scan"}, generation=1, control_epoch=0)
    assert (a.execution_id, b.execution_id, c.execution_id) == ("nav-000001", "man-000002", "obs-000003")
    assert b.action == "pick" and c.action == "scan" and a.action == "keypoint"
    assert c.resources == {"sense", "body"}
    assert set(PREFIX.values()) >= {"spk", "loc", "nav", "rch", "man", "wai", "rec", "obs"}
    assert default_action("navigate", {"location": "reach_stance"}) == "reposition"


def test_i1_result_always_resolves_even_before_the_loop_exists():
    e = mgr().create("speak", {"text": "hi"}, generation=1, control_epoch=0)
    h = ResultHandle(e)
    h.resolve(finish(e, "succeeded", {"utterance_id": "x"}))        # resolved before anyone awaits

    async def main():
        return await h.result()
    assert asyncio.run(main()).status == "succeeded"


def test_i1_result_is_shielded_from_the_awaiting_task_being_cancelled():
    async def main():
        e = mgr().create("navigate", {"location": "k"}, generation=1, control_epoch=0)
        h = ResultHandle(e)
        waiter = asyncio.ensure_future(h.result())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        h.resolve(finish(e, "succeeded", {"location": "k", "executor": "lite"}))
        return await h.result()
    assert asyncio.run(main()).status == "succeeded"


def test_i2_cancel_before_start_and_cancel_is_an_idempotent_request():
    calls = []
    e = mgr().create("navigate", {"location": "k"}, generation=1, control_epoch=0)
    h = ResultHandle(e, on_cancel=calls.append)
    e.status = "running"
    h.cancel("correction")
    h.cancel("again")
    assert calls == ["correction"] and h.cancel_requested and e.status == "cancelling"
    assert e.cancel_reason == "correction" and not h.done


def test_i3_terminal_is_immutable_and_one_result_per_execution():
    e = mgr().create("navigate", {"location": "k"}, generation=1, control_epoch=0)
    h = ResultHandle(e)
    first = finish(e, "succeeded", {"location": "k", "executor": "lite"})
    assert h.resolve(first) is True
    assert h.resolve(finish(e, "failed", {"reason": "blocked"})) is False
    assert e.status == "succeeded" and e.result is first
    h.cancel("too late")                                  # after terminal: nothing changes
    assert e.status == "succeeded" and not h.cancel_requested
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.status = "failed"                           # type: ignore[misc]


def test_i4_generation_and_epoch_are_stamped_at_creation():
    fence = Fence(generation=4, control_epoch=2)
    e = mgr().create("navigate", {"location": "k"}, generation=fence.generation, control_epoch=fence.control_epoch)
    fence.apply("correction")
    assert (e.generation, e.control_epoch) == (4, 2) and (fence.generation, fence.control_epoch) == (5, 3)


def test_i5_late_is_older_generation():
    fence = Fence(generation=5, control_epoch=0)
    assert fence.is_late(4) and not fence.is_late(5)


@pytest.mark.parametrize("trigger,dg,de", [
    ("request_idle", 1, 0), ("request_busy", 0, 0), ("correction", 1, 1), ("stop", 0, 1), ("resume", 0, 1),
    ("constraint", 0, 1), ("own_goal_dropped", 0, 1),
])
def test_generation_table(trigger, dg, de):
    f = Fence(3, 7)
    f.apply(trigger)
    assert (f.generation - 3, f.control_epoch - 7) == (dg, de)
    assert trigger in GENERATION_RULES


def test_what_each_trigger_cancels():
    m = mgr()
    nav = m.create("navigate", {"location": "k"}, generation=1, control_epoch=0, status="running")
    rch = m.create("check_reachability", {"object_type": "mug"}, generation=1, control_epoch=0, status="running")
    per = m.create("navigate", {"location": "k"}, generation=2, control_epoch=0, status="running", source="persona")
    done = m.create("navigate", {"location": "k"}, generation=1, control_epoch=0, status="succeeded")
    f = Fence(2, 0)
    f.apply("correction")                                  # -> generation 3
    assert should_cancel(nav, "correction", f) and should_cancel(rch, "correction", f)
    assert not should_cancel(done, "correction", f)
    assert should_cancel(nav, "stop", f) and not should_cancel(rch, "stop", f)     # stop cancels the body only
    assert should_cancel(per, "own_goal_dropped", f) and not should_cancel(nav, "own_goal_dropped", f)
    assert not should_cancel(nav, "resume", f)


def test_decision_staleness_and_stale_events():
    f = Fence(2, 5)
    assert not f.decision_stale(2, 5) and f.decision_stale(1, 5) and f.decision_stale(2, 4)
    m = mgr()
    live = m.create("navigate", {"location": "k"}, generation=2, control_epoch=5, status="running")
    over = m.create("navigate", {"location": "k"}, generation=2, control_epoch=5, status="cancelled")
    assert not is_stale_event(live.execution_id, m.by_id)
    assert is_stale_event(over.execution_id, m.by_id) and is_stale_event("nav-999999", m.by_id)
    assert is_stale_event(None, m.by_id)


def test_rejected_keeps_goalrejected_compat():
    r = Rejected("capability", "policy_unavailable", "policy unavailable: down")
    assert (r.stage, r.code, r.reason) == ("capability", "policy_unavailable", "policy unavailable: down")


def test_execution_compat_with_history_entry_uses():
    """The THOR-era writes still work on an Execution (PLAN 5.4, test_execution_compat)."""
    e = mgr().create("speak", {"text": "hi"}, generation=3, control_epoch=1, t=2.0)
    assert (e.id, e.created_for, e.tool) == (e.execution_id, 3, "speak")
    assert e.t_start == 2.0                               # t_created until it starts
    e.t_start = 2.5                                       # SpeechQueue: item.entry.t_start = now
    e.data = {"reason": "x"}                              # SpeechQueue: item.entry.data = {...}
    e.data["late"] = True                                 # harness: e.data["late"] = True
    e.t_end = 3.0                                         # harness._finish assigns e.t_end
    e.status = "succeeded"
    assert (e.t_start, e.t_started, e.t_end, e.t_ended, e.finished) == (2.5, 2.5, 3.0, 3.0, True)
    assert e.late is True and e.to_dict()["late"] is True
    with pytest.raises(AttributeError):
        e.id = "x"                                        # type: ignore[misc]


def test_speech_queue_runs_on_executions_and_drops_old_generations():
    from agent.skills import SpeechItem, SpeechQueue
    from sim.clock import SimClock
    from tests.unit.fakes_rt import FakeRobot

    async def main():
        clock = SimClock(20.0)
        robot = FakeRobot(clock)
        m = ExecutionManager(clock)
        q = SpeechQueue(robot, clock)
        got = []
        q.on_result = lambda e, r: got.append((e.execution_id, r.status, r.data.get("reason")))
        a = m.create("speak", {"text": "one two"}, generation=1, control_epoch=0)
        b = m.create("speak", {"text": "old line"}, generation=1, control_epoch=0)
        c = m.create("speak", {"text": "new line"}, generation=2, control_epoch=1)
        runner = asyncio.ensure_future(q.run())
        q.enqueue(SpeechItem(a, "one two", 1, 0))
        q.enqueue(SpeechItem(b, "old line", 1, 0))
        q.enqueue(SpeechItem(c, "new line", 2, 1))
        await asyncio.sleep(0)
        dropped = q.drop_older_than(2)
        while q.busy():
            await asyncio.sleep(0.01)
        runner.cancel()
        return a, b, c, dropped, got, robot.said
    a, b, c, dropped, got, said = asyncio.run(main())
    assert b.status == "dropped" and b.result.status == "cancelled"
    assert b.result.data["reason"] == "superseded (generation 2)"
    assert c.status == "succeeded" and "new line" in said and "old line" not in said
    assert "old line" in dropped


def test_run_execution_escalates_cancel_then_halt_and_always_resolves():
    from agent.skills import run_execution
    from sim.clock import SimClock
    from tests.unit.fakes_rt import FakeRobot

    async def main():
        clock = SimClock(50.0)
        robot = FakeRobot(clock, nav_s_per_m=10.0)
        robot.ignore_cancel = True                        # only a halt stops it
        e = ExecutionManager(clock).create("navigate", {"location": "bedroom_dresser_1a"}, generation=1,
                                           control_epoch=0)
        res = await run_execution(robot, clock, e, timeout=1.0)
        return res, robot.halts
    res, halts = asyncio.run(main())
    assert res.status == "timed_out" and res.data["reason"] == "timeout"
    assert "ignored cancel; halted" in res.data["detail"] and halts
    assert res.data["halted_by_runtime"] is True             # the harness releases its own halt (_finish)


def test_run_execution_turns_a_capability_rejection_into_a_result():
    from agent.skills import run_execution
    from sim.clock import SimClock
    from tests.unit.fakes_rt import FakeRobot

    async def main():
        clock = SimClock(50.0)
        robot = FakeRobot(clock)
        robot.policy_down = True
        e = ExecutionManager(clock).create("manipulate", {"action": "pick", "object_type": "mug"}, generation=1,
                                           control_epoch=0)
        return await run_execution(robot, clock, e, timeout=5.0)
    res = asyncio.run(main())
    assert res.status == "rejected" and res.data["stage"] == "capability"
    assert res.execution_id is not None and res.observation_id
