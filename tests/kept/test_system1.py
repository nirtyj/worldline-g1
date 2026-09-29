"""System 1 (brains/system1.py) against fake Gemini Live sessions.

The fakes emit the SDK's own message types (google.genai.types), so a wrong
attribute name fails here rather than in the live server. System 1 opens two
sessions: the router (tool "route") and the observer (tool "observe"); the fake
client hands out prepared sessions by which tool the session declares.

    .venv-rt/bin/python -m pytest tests/kept/test_system1.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import unittest
from typing import Any, Callable

from google.genai import types

from brains.interface import System1
from brains.system1 import SystemOne, create, normalize_items, normalize_route

CLOSE = object()


def call(name: str, **args: Any) -> types.LiveServerMessage:
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id=f"{name}-{args.get('turn')}", name=name, args=args)]))


def done() -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))


def talks(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(
        output_transcription=types.Transcription(text=text)))


def usage(prompt: int) -> types.LiveServerMessage:
    return types.LiveServerMessage(usage_metadata=types.UsageMetadata(prompt_token_count=prompt,
                                                                      total_token_count=prompt + 50))


def resumption(handle: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(session_resumption_update=types.LiveServerSessionResumptionUpdate(
        new_handle=handle, resumable=True))


def go_away() -> types.LiveServerMessage:
    return types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left="5s"))


def turn_of(text: str) -> int:
    return int(text.split("#", 1)[1].split(":", 1)[0])


Script = Callable[[str], list[tuple[float, Any]]]      # question text -> [(delay_s, message)]


class FakeSession:
    def __init__(self, script: Script | None = None, opening: list[Any] | None = None) -> None:
        self.script = script or (lambda text: [])
        self.sent: list[tuple[list[Any], bool]] = []
        self.tool_responses: list[Any] = []
        self.q: asyncio.Queue[Any] = asyncio.Queue()
        for m in opening or []:
            self.q.put_nowait(m)

    async def send_client_content(self, *, turns: Any, turn_complete: bool) -> None:
        parts = turns[0]["parts"]
        self.sent.append((parts, turn_complete))
        if turn_complete:
            for delay, msg in self.script(parts[0]["text"]):
                asyncio.get_running_loop().call_later(delay, self.q.put_nowait, msg)

    async def send_tool_response(self, *, function_responses: Any) -> None:
        self.tool_responses.append(function_responses)

    async def receive(self) -> Any:
        while True:
            m = await self.q.get()
            if m is CLOSE:
                raise ConnectionError("socket closed")
            yield m
            if m.server_content is not None and m.server_content.turn_complete:
                return

    def texts(self) -> list[str]:
        return [p[0]["text"] for p, _ in self.sent]


class FakeClient:
    """client.aio.live.connect(model=, config=): prepared sessions (or errors) per channel."""

    def __init__(self, router: list[Any] | None = None, observer: list[Any] | None = None,
                 reconnect_takes: float = 0.0) -> None:
        self.queues = {"route": list(router or []), "observe": list(observer or [])}
        self.configs: dict[str, list[Any]] = {"route": [], "observe": []}
        self.reconnect_takes = reconnect_takes            # how long every connect after the first takes
        client = self

        class Live:
            @contextlib.asynccontextmanager
            async def connect(self, *, model: str, config: Any) -> Any:
                tool = config.tools[0].function_declarations[0].name
                if client.configs[tool]:
                    await asyncio.sleep(client.reconnect_takes)
                client.configs[tool].append(config)
                q = client.queues[tool]
                s = q.pop(0) if q else FakeSession()
                if isinstance(s, Exception):
                    raise s
                yield s

        class Aio:
            live = Live()
        self.aio = Aio()


def routes(route_args: dict[str, Any] | None = None, delay: float = 0.05) -> Script:
    """A router script: every USER #n gets a route call carrying n."""
    def script(text: str) -> list[tuple[float, Any]]:
        if not text.startswith("USER #"):
            return []
        args = route_args or {"kind": "request", "confidence": 0.9}
        return [(delay, call("route", turn=turn_of(text), **args)), (delay + 0.02, done())]
    return script


def observes(answers: list[list[dict[str, Any]]], delay: float = 0.05) -> Script:
    """An observer script: each OBSERVE #n gets the next prepared answer."""
    def script(text: str) -> list[tuple[float, Any]]:
        if not text.startswith("OBSERVE #") or not answers:
            return []
        return [(delay, call("observe", turn=turn_of(text), items=answers.pop(0))), (delay + 0.02, done())]
    return script


async def started(router: Any = None, observer: Any = None, **kw: Any):
    statuses: list[tuple[str, str]] = []
    client = FakeClient([router or FakeSession()], [observer or FakeSession()])
    s1 = SystemOne("test-key", lambda st, d: statuses.append((st, d)), client=client, reconnect_delay=0.01, **kw)
    task = asyncio.create_task(s1.run())
    for _ in range(200):
        if s1.router.status == "ready" and s1.observer.status in ("ready", "error"):
            break
        await asyncio.sleep(0.01)
    return s1, task, statuses, client


async def stopped(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class NormalizeTest(unittest.TestCase):
    def test_route(self):
        self.assertEqual(normalize_route({"turn": 3, "kind": "request", "target": "  the mug ", "replaces_task": 1,
                                          "says_yes": "none", "confidence": 1.7}),
                         {"kind": "request", "target": "the mug", "replaces_task": True, "says_yes": None,
                          "confidence": 1.0})
        self.assertIsNone(normalize_route({"kind": "dance", "confidence": 1}))
        self.assertEqual(normalize_route({"kind": "answer", "says_yes": "No"})["says_yes"], False)
        self.assertEqual(normalize_route({"kind": "answer", "says_yes": "yes"})["says_yes"], True)
        self.assertEqual(normalize_route({"kind": "stop"})["confidence"], 0.0)      # missing: the planner decides
        self.assertEqual(normalize_route({"kind": "stop", "confidence": "high"})["confidence"], 0.0)
        self.assertEqual(normalize_route({"kind": "stop", "confidence": float("nan")})["confidence"], 0.0)

    def test_items(self):
        items = normalize_items({"items": [{"what": "the fridge door is open", "where": "kitchen_1", "confidence": 0.8},
                                           {"what": "  ", "where": "x"}, "junk",
                                           {"what": "a towel on the floor", "where": ""},
                                           {"what": "a person in the hallway", "confidence": -2},
                                           {"what": "one too many"}]})
        self.assertEqual([i["what"] for i in items],
                         ["the fridge door is open", "a towel on the floor", "a person in the hallway"])
        self.assertEqual((items[1]["where"], items[1]["confidence"], items[2]["confidence"]), (None, 0.5, 0.0))
        self.assertEqual(normalize_items({}), [])


class Contract(unittest.TestCase):
    def test_implements_the_protocol_and_factory(self):
        self.assertIsInstance(SystemOne("k"), System1)
        old = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "SYSTEM1_MODEL")}
        try:
            os.environ["GEMINI_API_KEY"], os.environ["SYSTEM1_MODEL"] = "abc", "some-live-model"
            s1 = create(lambda st, d: None)
            self.assertEqual((s1.api_key, s1.model, s1.router.model), ("abc", "some-live-model", "some-live-model"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_no_key(self):
        async def main():
            statuses = []
            s1 = SystemOne("", lambda st, d: statuses.append((st, d)))
            await s1.run()
            self.assertEqual(statuses, [("error", "GEMINI_API_KEY is not set")])
            self.assertIsNone(await s1.route("stop"))
            self.assertEqual(await s1.observe(), [])
        asyncio.run(main())


class SessionsTest(unittest.TestCase):
    def test_two_sessions_each_with_one_silent_non_blocking_tool(self):
        async def main():
            s1, task, statuses, client = await started()
            r, o = client.configs["route"][0], client.configs["observe"][0]
            self.assertEqual([f.name for f in r.tools[0].function_declarations], ["route"])
            self.assertEqual([f.name for f in o.tools[0].function_declarations], ["observe"])
            for cfg in (r, o):
                self.assertEqual(cfg.tools[0].function_declarations[0].behavior, types.Behavior.NON_BLOCKING)
                self.assertIn("never speak", cfg.system_instruction.lower())
            self.assertNotIn("FRAME", r.system_instruction)                   # the router never sees images
            self.assertEqual(o.media_resolution, types.MediaResolution.MEDIA_RESOLUTION_LOW)
            self.assertEqual(s1.status, "ready")
            self.assertIn("router ready, observer ready", s1.detail)
            await stopped(task)
        asyncio.run(main())

    def test_ready_follows_the_router(self):
        async def main():
            s1, task, statuses, _ = await started(observer=RuntimeError("503"))
            self.assertEqual(s1.status, "ready")                            # routing still works
            self.assertIn("observer error (RuntimeError: 503)", s1.detail)
            await stopped(task)
        asyncio.run(main())


class RouteTest(unittest.TestCase):
    def test_label_and_silent_tool_response(self):
        async def main():
            router = FakeSession(routes({"kind": "correction", "target": "apple", "replaces_task": True,
                                         "says_yes": "none", "confidence": 0.85}))
            s1, task, _, _ = await started(router)
            r = await s1.route("no, the apple instead")
            self.assertEqual(r, {"kind": "correction", "target": "apple", "replaces_task": True, "says_yes": None,
                                 "confidence": 0.85})
            self.assertEqual(router.texts(), ["USER #1: no, the apple instead"])
            self.assertTrue(router.sent[0][1])                                # a question completes the turn
            await asyncio.sleep(0.05)
            fr = router.tool_responses[0][0]
            self.assertEqual((fr.id, fr.name, fr.response), ("route-1", "route", {"ok": True}))
            self.assertEqual(fr.scheduling, types.FunctionResponseScheduling.SILENT)
            self.assertEqual((s1.stats.routes, s1.stats.routed), (1, 1))
            await stopped(task)
        asyncio.run(main())

    def test_talking_instead_of_routing_falls_back_at_turn_end(self):
        async def main():
            def script(text):
                return [(0.05, talks("Sure, I'll get")), (0.06, talks(" the mug!")), (0.08, done())]
            s1, task, _, _ = await started(FakeSession(script), route_timeout=2.0)
            t0 = time.monotonic()
            self.assertIsNone(await s1.route("bring me the mug"))
            self.assertLess(time.monotonic() - t0, 0.5)                       # not the 2 s timeout
            self.assertEqual(s1.router.stats.spoke, ["Sure, I'll get the mug!"])
            self.assertEqual(s1.stats.fallbacks, 1)
            await stopped(task)
        asyncio.run(main())

    def test_a_bare_turn_end_resolves_nothing(self):
        async def main():
            def script(text):                  # an earlier turn finishing (no talk, no call), then the answer
                return [(0.02, done()), (0.1, call("route", turn=1, kind="stop", confidence=0.99)), (0.12, done())]
            s1, task, _, _ = await started(FakeSession(script))
            self.assertEqual((await s1.route("hold on"))["kind"], "stop")
            await stopped(task)
        asyncio.run(main())

    def test_a_router_that_stays_down_gives_up_within_a_second(self):
        async def main():
            client = FakeClient([RuntimeError("down")] * 5, [FakeSession()])
            s1 = SystemOne("k", client=client, reconnect_delay=5.0, route_timeout=2.0)
            task = asyncio.create_task(s1.run())
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            self.assertIsNone(await s1.route("stop"))
            self.assertAlmostEqual(time.monotonic() - t0, 1.0, delta=0.15)  # waited for a fresh session, then gave up
            await stopped(task)
        asyncio.run(main())

    def test_timeout(self):
        async def main():
            s1, task, _, _ = await started(route_timeout=0.2)
            t0 = time.monotonic()
            self.assertIsNone(await s1.route("hello?"))
            self.assertAlmostEqual(time.monotonic() - t0, 0.2, delta=0.1)
            await stopped(task)
        asyncio.run(main())

    def test_a_late_call_for_an_earlier_message_is_ignored(self):
        async def main():
            def script(text):
                if text.startswith("USER #1"):         # answers #1 at 0.35 s: after it timed out at 0.2 s,
                    return [(0.35, call("route", turn=1, kind="request", target="mug", confidence=0.9))]
                return [(0.18, call("route", turn=2, kind="stop", confidence=0.99)), (0.19, done())]   # while #2 waits
            s1, task, _, _ = await started(FakeSession(script), route_timeout=0.2)
            self.assertIsNone(await s1.route("bring me the mug"))           # timed out
            self.assertEqual((await s1.route("stop"))["kind"], "stop")       # not the late "request" for #1
            await stopped(task)
        asyncio.run(main())

    def test_a_newer_message_supersedes_a_pending_one(self):
        async def main():
            def script(text):
                n = turn_of(text)
                return [(0.3 if n == 1 else 0.05, call("route", turn=n, kind="request" if n == 1 else "stop",
                                                       confidence=0.9)), (0.35, done())]
            s1, task, _, _ = await started(FakeSession(script))
            first = asyncio.create_task(s1.route("bring me the mug"))
            await asyncio.sleep(0.02)
            second = await s1.route("stop")
            self.assertIsNone(await first)                                   # handed to the planner
            self.assertEqual(second["kind"], "stop")
            await stopped(task)
        asyncio.run(main())

    def test_a_dropped_session_resolves_the_pending_question(self):
        async def main():
            s1, task, statuses, _ = await started(FakeSession(lambda text: [(0.05, CLOSE)]), route_timeout=2.0)
            t0 = time.monotonic()
            self.assertIsNone(await s1.route("bring me the mug"))
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertTrue(any(st == "error" and "socket closed" in d for st, d in statuses))
            await stopped(task)
        asyncio.run(main())


class ContextAndObserveTest(unittest.TestCase):
    def test_state_goes_to_both_robot_lines_to_the_router_only(self):
        async def main():
            router, observer = FakeSession(), FakeSession()
            s1, task, _, _ = await started(router, observer)
            ctx = {"at": "kitchen_1", "holding": {}, "goal": "bring the mug"}
            await s1.update(ctx)
            await s1.update(dict(ctx))                                        # unchanged: not sent again
            await s1.update({**ctx, "at": "hallway"})
            await s1.robot_said("On my way.")
            await s1.robot_said("")
            self.assertEqual(len(router.texts()), 3)
            self.assertTrue(router.texts()[0].startswith("STATE {") and '"at":"kitchen_1"' in router.texts()[0])
            self.assertEqual(router.texts()[2], "ROBOT SAID: On my way.")
            self.assertEqual(len(observer.texts()), 2)                        # states only
            self.assertNotIn("bring the mug", observer.texts()[0])          # and never the goal
            self.assertFalse(any(tc for _, tc in router.sent + observer.sent))
            await stopped(task)
        asyncio.run(main())

    def test_frames_go_to_the_observer_only_then_observe(self):
        async def main():
            router = FakeSession()
            observer = FakeSession(observes([[{"what": "the fridge door is open", "where": "kitchen_1",
                                               "confidence": 0.8},
                                              {"what": "a towel on the floor", "where": ""}]]))
            s1, task, _, _ = await started(router, observer)
            self.assertEqual(await s1.observe(), [])                          # no frames: no call at all
            self.assertEqual(observer.sent, [])
            await s1.frame(b"\xff\xd8jpeg-1", "kitchen_1")
            await s1.frame(b"\xff\xd8jpeg-2", None)
            (label, img), tc = observer.sent[0]
            self.assertEqual(label, {"text": "FRAME at kitchen_1"})
            self.assertEqual((img.inline_data.mime_type, img.inline_data.data), ("image/jpeg", b"\xff\xd8jpeg-1"))
            self.assertFalse(tc)
            self.assertEqual(router.sent, [])                                 # the router never gets a frame
            items = await s1.observe()
            self.assertEqual(observer.texts()[-1], "OBSERVE #1: 2 new frames since the last OBSERVE")
            self.assertEqual(items, [{"what": "the fridge door is open", "where": "kitchen_1", "confidence": 0.8},
                                     {"what": "a towel on the floor", "where": None, "confidence": 0.5}])
            self.assertEqual(await s1.observe(), [])                          # nothing new since
            self.assertEqual((s1.stats.observes, s1.stats.observed), (1, 2))
            await stopped(task)
        asyncio.run(main())

    def test_an_observation_is_placed_where_a_recent_frame_was_taken(self):
        async def main():
            observer = FakeSession(observes([[{"what": "a bowl on the table", "where": "start"},   # an old place
                                              {"what": "a mug on the counter", "where": "kitchen_1"}],
                                             [{"what": "a towel on the floor", "where": "hall_2"}]]))
            s1, task, _, _ = await started(observer=observer)
            await s1.frame(b"jpeg", "kitchen_1")
            await s1.frame(b"jpeg", "hall_2")
            self.assertEqual([i["where"] for i in await s1.observe()], ["hall_2", "kitchen_1"])
            await s1.frame(b"jpeg", None)                                     # between keypoints
            self.assertEqual((await s1.observe())[0]["where"], None)         # hall_2 is no longer recent
            await stopped(task)
        asyncio.run(main())

    def test_routing_does_not_wait_for_an_observation(self):
        async def main():
            router = FakeSession(routes({"kind": "stop", "confidence": 0.95}, delay=0.05))
            observer = FakeSession(observes([[{"what": "a spill", "where": "hall"}]], delay=0.5))
            s1, task, _, _ = await started(router, observer)
            await s1.frame(b"jpeg", "hall")
            obs = asyncio.create_task(s1.observe())
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            self.assertEqual((await s1.route("stop!"))["kind"], "stop")
            self.assertLess(time.monotonic() - t0, 0.3)
            self.assertEqual([i["what"] for i in await obs], ["a spill"])     # and the observation still arrives
            await stopped(task)
        asyncio.run(main())

    def test_one_observation_at_a_time(self):
        async def main():
            observer = FakeSession(observes([[{"what": "a spill", "where": "hall"}]], delay=0.3))
            s1, task, _, _ = await started(observer=observer)
            await s1.frame(b"jpeg", "hall")
            first = asyncio.create_task(s1.observe())
            await asyncio.sleep(0.05)
            await s1.frame(b"jpeg", "hall")
            self.assertEqual(await s1.observe(), [])                          # busy: skipped, not stacked
            self.assertEqual(len(await first), 1)
            await stopped(task)
        asyncio.run(main())


class ReconnectTest(unittest.TestCase):
    def test_go_away_resumes_the_router(self):
        async def main():
            first = FakeSession(opening=[resumption("handle-1"), go_away()])
            second = FakeSession(routes())
            client = FakeClient([first, second], [FakeSession()])
            s1 = SystemOne("k", client=client, reconnect_delay=0.01)
            task = asyncio.create_task(s1.run())
            for _ in range(200):
                if len(client.configs["route"]) == 2 and s1.router.status == "ready":
                    break
                await asyncio.sleep(0.01)
            self.assertIsNone(client.configs["route"][0].session_resumption.handle)
            self.assertEqual(client.configs["route"][1].session_resumption.handle, "handle-1")
            self.assertEqual((await s1.route("stop"))["kind"], "request")    # the new session answers
            await stopped(task)
        asyncio.run(main())

    def test_a_big_router_context_starts_a_fresh_session_with_the_state_restored(self):
        async def main():
            def big(text):
                if not text.startswith("USER #"):
                    return []
                return [(0.02, call("route", turn=turn_of(text), kind="request", confidence=0.9)),
                        (0.03, usage(9000)), (0.04, done())]
            first = FakeSession(big, opening=[resumption("handle-1")])
            second = FakeSession(routes())
            client = FakeClient([first, second], [FakeSession()], reconnect_takes=0.3)
            s1 = SystemOne("k", client=client, reconnect_delay=5.0)            # a rotation must not wait for this
            task = asyncio.create_task(s1.run())
            while s1.router.status != "ready":
                await asyncio.sleep(0.01)
            await s1.update({"at": "hall"})
            await s1.robot_said("On my way.")
            self.assertEqual((await s1.route("bring me the mug"))["kind"], "request")   # answered, then rotated
            while s1.router.stats.rotations == 0:
                await asyncio.sleep(0.005)
            self.assertEqual(s1.router.status, "connecting")                  # the switch is under way
            await s1.update({"at": "kitchen"})                                # during the switch: kept for later
            t0 = time.monotonic()
            self.assertEqual((await s1.route("stop"))["kind"], "request")     # waits for the fresh session
            self.assertGreater(time.monotonic() - t0, 0.2)
            self.assertIsNone(client.configs["route"][1].session_resumption.handle)   # fresh, not resumed
            self.assertEqual(second.texts()[:2], ['STATE {"at":"kitchen"}', "ROBOT SAID: On my way."])
            self.assertEqual((s1.router.stats.rotations, s1.router.stats.reconnects), (1, 0))
            self.assertEqual(s1.router.stats.total_tokens, 9050)
            self.assertIn("1 fresh sessions", s1.describe())
            await stopped(task)
        asyncio.run(main())

    def test_frames_from_before_an_observer_rotation_are_forgotten(self):
        async def main():
            first = FakeSession(lambda text: [(0.02, call("observe", turn=turn_of(text), items=[])),
                                              (0.03, usage(9000)), (0.04, done())]
                                if text.startswith("OBSERVE #") else [])
            second = FakeSession(observes([[{"what": "x", "where": "b"}]]))
            client = FakeClient([FakeSession()], [first, second])
            s1 = SystemOne("k", client=client, reconnect_delay=5.0)
            task = asyncio.create_task(s1.run())
            while s1.observer.status != "ready":
                await asyncio.sleep(0.01)
            await s1.frame(b"jpeg", "a")
            self.assertEqual(await s1.observe(), [])                          # answered, then it rotates
            for _ in range(100):
                if len(client.configs["observe"]) == 2 and s1.observer.status == "ready":
                    break
                await asyncio.sleep(0.01)
            await s1.frame(b"jpeg", "b")
            self.assertEqual(await s1.observe(), [{"what": "x", "where": "b", "confidence": 0.5}])
            self.assertEqual(second.texts()[-1], "OBSERVE #2: 1 new frame since the last OBSERVE")
            await stopped(task)
        asyncio.run(main())

    def test_connect_error_then_recovery(self):
        async def main():
            client = FakeClient([RuntimeError("503 unavailable"), FakeSession(routes())], [FakeSession()])
            statuses = []
            s1 = SystemOne("k", lambda st, d: statuses.append((st, d)), client=client, reconnect_delay=0.01)
            task = asyncio.create_task(s1.run())
            for _ in range(200):
                if s1.router.status == "ready":
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(any(st == "error" and "503 unavailable" in d for st, d in statuses))
            self.assertEqual(s1.status, "ready")
            self.assertIsNotNone(await s1.route("go"))
            await stopped(task)
        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
