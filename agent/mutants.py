"""Mutants: the reference runtime with one safeguard removed each.

Run one (the server's reset message takes it in `agent`) to see what the safeguard
buys: the same conversation goes wrong in a specific, visible way. Same seven
mutants as on THOR, on the new tool names (PLAN 4.4).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from api.tools import ToolCall

from .harness import Runtime
from .skills import run_execution
from .validate import Verdict, validate


class NoDropSpeech(Runtime):
    """Corrections don't drop or cut the old version's speech."""

    def _correct(self, utt: Any) -> None:
        self.speech.drop_older_than = lambda version: []      # type: ignore[method-assign]
        super()._correct(utt)


class NoKeywordStop(Runtime):
    """"Stop" goes through the model like everything else (fails once the model is slow)."""

    async def _listen(self) -> None:
        while True:
            utt = await self.user.next()
            self.task.utterances.append(utt)
            self._utt_q.put_nowait(utt)


class ForgetCancelledGrasp(Runtime):
    """After a cancel, assume the hand is as it was before the grasp: empty.
    Late results are thrown away and nothing is re-observed."""

    async def _reconcile(self) -> None:
        arms: set[str] = set()
        while True:
            for h in self._canceled:
                if h.skill == "manipulate" and h.args.get("arm"):
                    arms.add(h.args["arm"])
            pending = [h.task for h in self._canceled if h.task is not None and not h.task.done()]
            if not pending:
                break
            await asyncio.wait(pending)
        self._canceled.clear()
        for arm in arms:
            self.belief.set_hand(arm, None, "assumed", self.clock.now(), verified=True)
        self._wake("reconciled")

    def _correct(self, utt: Any) -> None:
        super()._correct(utt)
        for arm in ("left", "right"):
            if self.belief.holding[arm].source == "cancel":
                self.belief.set_hand(arm, None, "assumed", self.clock.now(), verified=True)

    def _finish(self, h, e, res) -> None:
        super()._finish(h, e, res)
        cancelled = h.cancel_requested or h.created_for < self.task.intent_version
        if h.skill == "manipulate" and cancelled and h.args.get("arm"):
            self.belief.set_hand(h.args["arm"], None, "assumed", self.clock.now(), verified=True)


class TrustSuccess(Runtime):
    """No verification glance after a manipulate: a reported success is believed."""

    async def _body(self, h, e) -> None:
        if h.skill == "manipulate":
            res = await run_execution(self.robot, self.clock, e, 60.0, on_handle=self._binder(h))
            self._finish(h, e, res)
            arm = h.args.get("arm")
            oid = res.data.get("object_id") or h.args.get("object_id")
            if res.ok and not h.cancel_requested and h.created_for == self.task.intent_version and arm:
                if e.action == "pick":
                    self.belief.set_hand(arm, oid, "skill", self.clock.now(), verified=True)
                else:
                    self.belief.set_hand(arm, None, "skill", self.clock.now(), verified=True)
                ob = self.belief.objects.get(oid) if oid else None
                if ob is not None:
                    ob.where = dataclasses.replace(ob.where, verified=True)
            self._wake(f"{h.skill} finished")
            return
        await super()._body(h, e)


class NoWaitForChunk(Runtime):
    """Cancel on correction, but don't wait for the cancelled actions to finish
    before starting new ones: no reconcile gate, and a busy body is ignored."""

    def _reconciling(self) -> bool:
        return False

    def _check(self, call: ToolCall) -> Verdict:
        v = self._vctx()
        v.history = [e for e in self.history if e.finished or "body" not in e.resources]   # "it is gone"
        return validate(call, v)


class CancelOnEverything(Runtime):
    """Any new utterance cancels what's running, as if it were a correction."""

    def _apply(self, utt: Any, kind: str) -> None:
        if kind in ("question", "addition", "answer", "chitchat"):
            self._correct(utt)
            return
        super()._apply(utt, kind)


class NoStaleCheck(Runtime):
    """Carry out a decision even if the user changed the request while the model
    was thinking (fails once the model is slow)."""

    def _stale(self, version: int) -> bool:
        return False


def _factory(cls):
    def create_runtime(robot, user, brain, clock):
        return cls(robot, user, brain, clock)
    create_runtime.__doc__ = cls.__doc__
    return create_runtime


no_drop_speech = _factory(NoDropSpeech)
no_keyword_stop = _factory(NoKeywordStop)
forget_cancelled_grasp = _factory(ForgetCancelledGrasp)
trust_success = _factory(TrustSuccess)
no_wait_for_chunk = _factory(NoWaitForChunk)
cancel_on_everything = _factory(CancelOnEverything)
no_stale_check = _factory(NoStaleCheck)
