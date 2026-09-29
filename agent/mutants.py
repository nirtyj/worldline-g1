"""Mutants: the reference runtime with one safeguard removed each.

Run one (the server's reset message takes it in `agent`) to see what the safeguard
buys: the same conversation goes wrong in a specific, visible way.
"""

from __future__ import annotations

import asyncio
from typing import Any

from brains.interface import ToolCall

from .harness import Runtime
from .skills import run_goal


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
                arms |= {r[4:] for r in h.resources if r.startswith("arm:")}
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

    def _finish(self, h, e, out) -> None:
        super()._finish(h, e, out)
        cancelled = h.cancel_requested or h.created_for < self.task.intent_version
        if h.skill in ("pick", "place") and cancelled:
            self.belief.set_hand(h.args["arm"], None, "assumed", self.clock.now(), verified=True)


class TrustSuccess(Runtime):
    """No verification look after pick and place: a reported success is believed."""

    async def _body(self, h, e) -> None:
        if h.skill in ("pick", "place"):
            out = await run_goal(self.robot, self.clock, h.skill, h.args, 40.0, on_goal=self._binder(h))
            self._finish(h, e, out)
            arm, oid = h.args["arm"], h.args["object"]
            if out.ok and not h.cancel_requested and h.created_for == self.task.intent_version:
                if h.skill == "pick":
                    self.belief.set_hand(arm, oid, "skill", self.clock.now(), verified=True)
                    ob = self.belief.objects.get(oid)
                    if ob is not None:
                        ob.where.verified = True
                else:
                    self.belief.set_hand(arm, None, "skill", self.clock.now(), verified=True)
                    ob = self.belief.objects.get(oid)
                    if ob is not None:
                        ob.where.verified = True
            self._wake(f"{h.skill} finished")
            return
        await super()._body(h, e)


class NoWaitForChunk(Runtime):
    """Cancel on correction, but don't wait for the cancelled actions to finish
    before starting new ones."""

    def _reconciling(self) -> bool:
        return False

    def _check(self, call: ToolCall) -> tuple[bool, str]:
        ok, why = super()._check(call)
        if not ok and why.startswith("already running"):
            return True, ""                  # the cancelled action "is gone"
        return ok, why


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
