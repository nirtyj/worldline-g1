"""eval/stack_suite.py: the lite subset of the G1 stack scenarios (PLAN 9.3) offline, in process (page server, the
real runtime, the scripted planner, the System 1 stub, the lite stack, faults injected into the lite body and the
skill registry), and the E5 GR00T plumbing scorer against a scripted fake page (no GR00T offline)."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("websockets")

from eval import stack_suite as ss  # noqa: E402
from test_suite_offline import H40_LAYOUT, H40_OBJECTS, FakePage  # noqa: E402


def _run(ids, **kw):
    try:
        return asyncio.run(asyncio.wait_for(ss.run_suite(ids, speed=10.0, **kw), 300))
    except FileNotFoundError as e:                      # no recorded house data on this machine
        pytest.skip(f"no recorded house: {e}")


def _crit(res, name):
    return next(c["ok"] for c in res["criteria"] if c["name"] == name)


def test_the_lite_subset_offline():
    out = _run(list(ss.LITE_SUBSET), scale=0.1)
    by = {r["id"]: r for r in out["results"]}
    for gid in ("G2", "G5", "G11", "G12", "G13", "G14"):
        assert by[gid]["verdict"] == "PASS*", (gid, by[gid]["criteria"], by[gid]["notes"])
        assert by[gid]["fallback_pass"] and not by[gid]["target_pass"]      # lite: every body result is a stand-in
    # G5 on lite: the injected outage is labelled, and the running-call half does not apply
    assert any("injected" in n for n in by["G5"]["notes"] + by["G5"]["injected"])
    assert _crit(by["G5"], "a running call fails policy_unavailable within 3 s, then HOLD") == ss.NA
    # G6 on lite: the blocked result is real; whether the robot tells the user is the planner's (the scripted one never)
    g6 = by["G6"]
    assert _crit(g6, "navigate failed: blocked") is True and _crit(g6, "with blocked_edge") is True
    assert _crit(g6, "after one replan") == ss.NA
    assert g6["verdict"] == ("PASS*" if _crit(g6, "tells the user") else "FAIL")
    assert out["lite_subset"]["of"] == 7 and out["applicable"] == 7
    json.dumps(out, default=str)


def test_scenarios_outside_the_profile_are_na_and_missing_hooks_skip():
    out = _run(["G1", "G3", "G7"])
    assert {r["id"]: r["verdict"] for r in out["results"]} == {"G1": "N/A", "G3": "N/A", "G7": "N/A"}

    async def remote():
        page = FakePage(H40_LAYOUT, H40_OBJECTS)
        run = ss.StackRun(page, "sonic", 0.01, ss.HookInjector({}))
        reader = asyncio.create_task(run.reader())
        try:
            return [await ss.run_one(s, run, local=False) for s in ss.SPECS if s.id in ("G7", "G9", "G4")]
        finally:
            reader.cancel()
    res = asyncio.run(remote())
    assert [r["verdict"] for r in res] == ["N/A", "SKIPPED", "SKIPPED"]      # G4 is full-only; G7/G9 need hooks
    assert "push_robot" in res[1]["notes"][0] or "throttle_rtf" in res[1]["notes"][0]


def test_outcome_verdicts():
    o = ss.Outcome()
    o.check("a", True)
    o.check("b", ss.NA)
    assert o.verdict() == "PASS"
    o.check("c", None)
    assert o.verdict() == "UNVERIFIED"
    o.check("d", False)
    assert o.verdict() == "FAIL"
    assert ss.Outcome().verdict() == "UNVERIFIED"


# ---------------------------------------------------------------------------------------------- E5 on a fake page
class GrootPage(FakePage):
    """A fake full-profile page: each "Bring me ..." starts a GR00T pick; "never mind" cancels it within 0.5 s,
    "stop" halts it (a stop row with a receipt, then failed(halted)); otherwise it fails grasp_missed after 4 s,
    except the second plain trial, which lifts the object (GT: hand:right)."""

    def __init__(self, dead_after: int | None = None) -> None:
        super().__init__(H40_LAYOUT, H40_OBJECTS, self._on_say)
        self.trials = 0
        self.current: dict | None = None
        self.dead_after = dead_after          # from this trial on the body refuses GR00T at `enter` (the fallback runs)

    def dead(self, n: int) -> bool:
        return self.dead_after is not None and n >= self.dead_after

    async def _on_say(self, page: FakePage, text: str) -> None:
        t = text.lower()
        if t.startswith("bring me"):
            self.trials += 1
            eid = f"man-{self.trials:06d}"
            self.current = {"eid": eid, "open": True, "n": self.trials}
            await self.frame(trace=[{"t": self.t, "type": "started", "tool": "manipulate", "action": "pick",
                                     "execution_id": eid, "args": {"action": "pick", "skill_id": "groot.pick.alarm_clock.arena_static.v0"}}],
                             active=[{"id": eid, "tool": "manipulate"}])
            if not self.dead(self.trials):     # the GR00T session runs: groot_arms' first inference event
                await self.frame(events=[{"t": self.t, "type": "groot.inference", "session": eid, "ok": True,
                                          "latency_ms": 40.0}])
            asyncio.get_running_loop().create_task(self._finish_later(eid, self.trials))
        elif self.current and self.current["open"] and ("never mind" in t or t == "stop"):
            cur = self.current
            cur["open"] = False
            if t == "stop":
                await self.frame(trace=[{"t": self.t, "type": "stop", "receipt": {"accepted": True, "stopped": True,
                                                                                  "latency_ms": 4.0}}])
                await self._result(cur["eid"], "failed", "halted", cur["n"])
            else:
                await self._result(cur["eid"], "cancelled", "cancelled", cur["n"])
        else:
            await self.frame()

    async def _finish_later(self, eid: str, n: int) -> None:
        for _ in range(8):
            await asyncio.sleep(0.05)
            await self.frame()
        if self.current and self.current["eid"] == eid and self.current["open"]:
            self.current["open"] = False
            if n == 2:
                self.objects["alarm_clock_1"]["where"] = "hand:right"
            await self._result(eid, "succeeded" if n == 2 else "failed", None if n == 2 else "grasp_missed", n)

    async def _result(self, eid: str, status: str, reason: str | None, n: int = 0) -> None:
        skill = "groot.pick.alarm_clock.arena_static.v0"
        if self.dead(n):                      # groot_then_script: GR00T refused at enter, the script's result
            data = {"executor": "sonic_arm_script", "skill": "sonic.script.pick.v0", "reason": reason, "inferences": 0,
                    "attempts": [{"executor": "groot_arms", "skill": skill, "status": "failed",
                                  "reason": "controller_unavailable"},
                                 {"executor": "sonic_arm_script", "skill": "sonic.script.pick.v0", "status": status,
                                  "reason": reason}],
                    "fallback_from": {"executor": "groot_arms", "status": "failed", "reason": "controller_unavailable"}}
            ex = "sonic_arm_script"
        else:
            data = {"executor": "groot_arms", "skill": skill, "reason": reason, "inferences": 7, "label": "experimental"}
            ex = "groot_arms"
        await self.frame(trace=[{"t": self.t, "type": "result", "tool": "manipulate", "action": "pick",
                                 "execution_id": eid, "status": status, "executor": ex, "data": data}])


def test_e5_scores_ten_trials_from_the_page_alone(monkeypatch):
    monkeypatch.setattr(ss.suite, "LOAD_TIMEOUT_S", 5.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.02)))
    monkeypatch.setattr(ss.suite.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.02)))

    async def go():
        page = GrootPage()
        run = ss.StackRun(page, "full", 0.02, ss.HookInjector({}))
        reader = asyncio.create_task(run.reader())
        try:
            return await ss.run_one(next(s for s in ss.SPECS if s.id == "E5"), run, local=False), page
        finally:
            reader.cancel()
    res, page = asyncio.run(go())
    trials = res["extra"]["trials"]
    assert len(trials) == 10 and page.trials == 10
    assert [t["kind"] for t in trials] == ["plain"] * 4 + ["cancel"] * 3 + ["halt"] * 3
    assert all(t["status"] == "cancelled" for t in trials if t["kind"] == "cancel")
    assert all(t["status"] == "failed" and t["reason"] == "halted" and t["receipt"]["stopped"]
               for t in trials if t["kind"] == "halt")
    assert res["verdict"] == "PASS", res["criteria"]                         # plumbing met; success not required
    assert "1/4 plain picks ended in hand (GT)" in res["extra"]["success"]
    assert any("experimental GR00T skill" in h for h in res["honesty"])
    assert res["target_pass"]                                                 # groot_arms is a target executor
    assert all(t["groot_running_at_act"] for t in trials if t["kind"] in ("cancel", "halt"))


def test_e5_fails_when_gr00t_never_ran_and_the_stops_hit_the_fallback(monkeypatch):
    """The offline E5 rehearsal (M2b wave 2): from trial 4 the body refused every GR00T session at `enter`
    (stale_session), so GR00T ran 0 inferences and every cancel and halt landed on sonic_arm_script - and the old
    scorer still said cancel 3/3, halt 3/3. GR00T plumbing is only met when GR00T ran and was the one stopped."""
    monkeypatch.setattr(ss.suite, "LOAD_TIMEOUT_S", 5.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.02)))
    monkeypatch.setattr(ss.suite.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.02)))

    async def go():
        page = GrootPage(dead_after=4)
        run = ss.StackRun(page, "full", 0.02, ss.HookInjector({}))
        reader = asyncio.create_task(run.reader())
        try:
            return await ss.run_one(next(s for s in ss.SPECS if s.id == "E5"), run, local=False)
        finally:
            reader.cancel()
    res = asyncio.run(go())
    crit = {c["name"] if isinstance(c, dict) and "name" in c else str(c): c for c in res["criteria"]}
    assert res["verdict"] != "PASS", res["criteria"]
    text = " ".join(str(c) for c in res["criteria"])
    assert "3/10 with inferences > 0" in text and "cancel 0/3" in text and "halt 0/3" in text, text
    assert crit                                                               # criteria are listed one by one
