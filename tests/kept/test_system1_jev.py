"""System 1 with Jev labelling (brains/system1_jev.py), against a fake Jev client that returns
the SDK's own response type, and the fake Gemini Live sessions from tests/test_system1.py.

    .venv-rt/bin/python -m pytest tests/kept/test_system1_jev.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import unittest
from typing import Any

from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeAuthenticationError

from brains.interface import KINDS, System1
from brains.system1_jev import JevSystemOne, create
from tests.kept.test_system1 import FakeClient, FakeSession, observes


def response(kind: str = "request", p: float = 0.9, answer: str = "none", replaces: float = 0.2,
             target: str = "none") -> SystemOneResponse:
    rest = {k: round((1 - p) / (len(KINDS) - 1), 4) for k in KINDS if k != kind}
    return SystemOneResponse.model_validate({
        "model": "jev-1.13.0", "usage": {"input_tokens": 420, "output_tokens": 75},
        "answers": {
            "kind": {"type": "choice", "choice": kind, "confidence": p, "probabilities": {kind: p, **rest}},
            "answer": {"type": "choice", "choice": answer, "confidence": 0.9,
                       "probabilities": {answer: 0.9, **{a: 0.05 for a in ("yes", "no", "none") if a != answer}}},
            "replaces_task": {"type": "noul", "noul": replaces},
            "target": {"type": "choice", "choice": target, "confidence": 0.8, "probabilities": {target: 0.85}},
        }})


class FakeJev:
    """Duck-typed AsyncTypeSafeClient: records calls, answers from a list (or raises)."""

    def __init__(self, replies: list[Any], delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.delay = delay

    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        self.calls.append((state, questions))
        if self.delay:
            await asyncio.sleep(self.delay)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, BaseException):
            raise r
        return r


async def started(jev: FakeJev, observer: Any = None, gemini_key: str = "g", **kw: Any):
    statuses: list[tuple[str, str]] = []
    s1 = JevSystemOne("t", gemini_key, lambda st, d: statuses.append((st, d)), jev_client=jev,
                      client=FakeClient([], [observer or FakeSession()]), reconnect_delay=0.01, **kw)
    task = asyncio.create_task(s1.run())
    for _ in range(200):
        if s1.router.status in ("ready", "error") and (not gemini_key or s1.observer.status == "ready"):
            break
        await asyncio.sleep(0.01)
    return s1, task, statuses


async def stopped(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class JevLabelTest(unittest.TestCase):
    def test_contract_and_factory(self):
        self.assertIsInstance(JevSystemOne("t", "g"), System1)
        old = {k: os.environ.get(k) for k in ("TYPESAFE_API_KEY", "GEMINI_API_KEY", "SYSTEM1_JEV_MODEL")}
        try:
            os.environ.update(TYPESAFE_API_KEY="tk", GEMINI_API_KEY="gk", SYSTEM1_JEV_MODEL="jev-1.13.0")
            s1 = create(lambda st, d: None)
            self.assertEqual((s1.router.api_key, s1.api_key, s1.router._model_arg), ("tk", "gk", "jev-1.13.0"))
            self.assertEqual(s1.observer.model, "gemini-3.8-live")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_one_call_with_four_typed_questions_and_the_robots_context(self):
        async def main():
            jev = FakeJev([response()])
            s1, task, statuses = await started(jev)
            self.assertEqual(s1.status, "ready")
            self.assertIn("labels jev-1.13.0: ready; observations gemini-3.8-live: ready", s1.detail)
            await s1.update({"at": "kitchen_1", "goal": "bring the mug", "in_view": ["alarm_clock_2"],
                             "holding": {"left": "mug_1"}})
            await s1.robot_said("Want me to check the kitchen?")
            await s1.route("sure, have a look")
            state, qs = jev.calls[-1]
            self.assertEqual(state, {"robot": {"at": "kitchen_1", "goal": "bring the mug", "in_view": ["alarm_clock_2"],
                                               "holding": {"left": "mug_1"}},
                                     "robot_last_said": "Want me to check the kitchen?", "message": "sure, have a look"})
            self.assertEqual(set(qs), {"kind", "answer", "replaces_task", "target"})
            self.assertIsInstance(qs["kind"], Choice)
            self.assertEqual(set(qs["kind"].criteria), set(KINDS))
            self.assertEqual(set(qs["answer"].criteria), {"yes", "no", "none"})
            self.assertIsInstance(qs["replaces_task"], Noul)
            words = set(qs["target"].criteria)
            self.assertTrue({"alarm clock", "mug", "apple", "none"} <= words)   # in view, held, known, none
            self.assertEqual(len(jev.calls), 2)                                    # the key check, then the message
            await stopped(task)
        asyncio.run(main())

    def test_response_becomes_the_contracts_label(self):
        async def main():
            jev = FakeJev([response(), response("answer", 0.93, answer="yes", replaces=0.1),
                           response("correction", 0.8, replaces=0.7, target="apple"),
                           response("answer", 0.6, answer="no")])
            s1, task, _ = await started(jev)
            yes = await s1.route("sure, have a look")
            self.assertEqual({k: yes[k] for k in ("kind", "says_yes", "replaces_task", "target", "confidence")},
                             {"kind": "answer", "says_yes": True, "replaces_task": False, "target": "", "confidence": 0.93})
            self.assertEqual(list(yes["probabilities"])[0], "answer")
            corr = await s1.route("never mind, get me an apple")
            self.assertEqual((corr["kind"], corr["replaces_task"], corr["target"]), ("correction", True, "apple"))
            self.assertEqual((await s1.route("nah"))["says_yes"], False)
            self.assertEqual(s1.router.stats.context_tokens, 420)
            self.assertIn("labels jev-1.13.0", s1.describe())
            await stopped(task)
        asyncio.run(main())

    def test_an_unsure_label_has_low_confidence(self):
        async def main():
            s1, task, _ = await started(FakeJev([response(), response("correction", 0.42)]))
            r = await s1.route("no")
            self.assertEqual((r["kind"], r["confidence"]), ("correction", 0.42))   # below 0.5: the planner decides
            await stopped(task)
        asyncio.run(main())

    def test_slow_or_failing_calls_fall_back_to_the_planner(self):
        async def main():
            jev = FakeJev([response()])
            s1, task, _ = await started(jev, route_timeout=0.2)
            jev.delay = 1.0
            t0 = time.monotonic()
            self.assertIsNone(await s1.route("stop"))
            self.assertAlmostEqual(time.monotonic() - t0, 0.2, delta=0.1)
            jev.delay = 0.0
            jev.replies = [RuntimeError("502 bad gateway")]
            self.assertIsNone(await s1.route("stop"))
            self.assertIn("502", s1.router.stats.last_error)
            self.assertEqual(s1.status, "ready")                                   # one bad call isn't an outage
            self.assertEqual(s1.stats.fallbacks, 2)
            await stopped(task)
        asyncio.run(main())

    def test_a_bad_key_is_an_error_and_labels_go_to_the_planner(self):
        async def main():
            import httpx2
            bad = TypeSafeAuthenticationError(401, {"error": "invalid API key"}, httpx2.Headers(), message="invalid API key")
            s1, task, statuses = await started(FakeJev([bad]))
            self.assertEqual(s1.status, "error")
            self.assertIn("Authentication", s1.detail)
            await stopped(task)
        asyncio.run(main())

    def test_no_keys(self):
        async def main():
            statuses = []
            s1 = JevSystemOne("", "", lambda st, d: statuses.append((st, d)))
            await asyncio.wait_for(s1.run(), 1.0)
            self.assertEqual(s1.status, "error")
            self.assertIn("TYPESAFE_API_KEY is not set", s1.detail)
            self.assertIsNone(await s1.route("stop"))
        asyncio.run(main())


class ObserverStillGeminiTest(unittest.TestCase):
    def test_frames_and_observations_go_through_gemini_live(self):
        async def main():
            observer = FakeSession(observes([[{"what": "a spill on the floor", "where": "hall"}]]))
            jev = FakeJev([response()])
            s1, task, _ = await started(jev, observer)
            await s1.update({"at": "hall", "goal": "bring the apple"})
            await s1.frame(b"jpeg", "hall")
            self.assertEqual(await s1.observe(), [{"what": "a spill on the floor", "where": "hall", "confidence": 0.5}])
            self.assertNotIn("apple", json.dumps(observer.texts()))              # the goal never reaches the observer
            self.assertEqual(len(jev.calls), 1)                                   # frames never go to Jev
            await stopped(task)
        asyncio.run(main())

    def test_labels_work_without_a_gemini_key(self):
        async def main():
            s1, task, _ = await started(FakeJev([response(), response("stop", 0.97)]), gemini_key="")
            self.assertEqual((await s1.route("hold on"))["kind"], "stop")
            self.assertEqual(await s1.observe(), [])
            await stopped(task)
        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
