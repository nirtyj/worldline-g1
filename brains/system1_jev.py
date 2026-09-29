"""System 1 with two engines: Jev labels messages, Gemini 3.8 Live notices things in frames.

    SYSTEM1=brains.system1_jev:create      # TYPESAFE_API_KEY and GEMINI_API_KEY from .env

Labelling a message is a typed decision from text, which is what TypeSafe's Jev is
built for: it returns a probability for every option instead of generating text.
Each message is one stateless call with four questions:

    kind           choice among the ten kinds          -> kind, confidence = P(kind)
    answer         choice: yes / no / not an answer     -> says_yes
    replaces_task  yes/no probability                   -> replaces_task (P >= 0.5)
    target         choice among object words + "none"   -> target (Jev can't copy free text)

The state sent with it is the robot's latest fused state, its last line and the
message, so there is no session context to grow and nothing to rotate. Observing
needs images, which Jev doesn't take, so that stays on the Gemini Live observer
from brains/system1.py.

Measured on jev-1.13.0: 80-200 ms a call (the first one ~200 ms), and "hold on" while
driving came back stop at P=0.98. The probabilities are what the runtime's
confidence threshold (0.5) was meant to read; Gemini said 0.8-1.0 for everything.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from typing import Any, Callable

from brains.interface import KINDS
from brains.system1 import ROUTE_TIMEOUT_S, ChannelStats, SystemOne
from agent.memory import VOLATILE_TYPES

KIND_CRITERIA = {
    "request": "a new task for the robot: 'bring me the mug', 'what's on the table?', 'go to the kitchen'",
    "correction": "changes the task the robot is doing now: 'no, the red one', 'actually the apple instead', "
                  "'bring it to the bedroom instead', 'never mind, forget it'",
    "addition": "adds another task and keeps the current one: 'also grab the bread', 'and a spoon too'",
    "question": "asks something and changes nothing: 'how long will it take?', 'where are you going?', "
                "'what are you holding?'",
    "stop": "halt now: 'stop', 'hold on', 'wait wait', 'hang on a sec', 'freeze', 'whoa', also misspelled or "
            "in another language ('stpo', 'para', 'arrête'). Not 'don't stop', not 'stop by the kitchen'",
    "resume": "carry on after a stop: 'okay, go ahead', 'continue', 'you can keep going', 'carry on'",
    "answer": "replies to the question the robot just asked (robot_last_said ends with a question) and answers "
              "that question, even when it sounds like a command: 'the blue one', 'yes', 'nah', 'have a look', "
              "'the kitchen'",
    "constraint": "changes how to do the task without replacing it: 'don't go into the bedroom', "
                  "'be careful, it's hot', 'use your left hand', 'stop by the kitchen on the way'",
    "observation": "tells the robot a fact about the home or where things are: 'my keys are usually on the "
                   "counter', 'the keys are in the drawer', 'I moved the mug to the sink'. A fact is an "
                   "observation unless it answers the question the robot just asked",
    "chitchat": "nothing to do: 'thanks', 'good job', 'hello'",
}
assert set(KIND_CRITERIA) == set(KINDS), "every kind needs a description"

ANSWER_CRITERIA = {
    "yes": "the message says yes to a yes/no question the robot just asked ('sure', 'have a look', 'go ahead')",
    "no": "the message says no to a yes/no question the robot just asked ('nah', 'not now', 'maybe later')",
    "none": "the robot didn't just ask a yes/no question, or the message doesn't answer it",
}

# Words a message may name an object by (Jev picks one; it can't copy text out of the message).
TARGET_WORDS = sorted(VOLATILE_TYPES | {
    "cup", "plate", "bowl", "pot", "pan", "vase", "statue", "lettuce", "potato", "keys", "phone", "glass",
    "towel", "blanket", "shoe", "bag", "wallet", "charger", "remote", "toy", "medicine", "water"})


def set_vocabulary(types) -> list[str]:
    """Add the registry's object types and the fixed pickupable vocabulary to TARGET_WORDS
    (PLAN 8.5). Ids stay <snake_type>_<n>: Jev can't copy free text out of a message."""
    global TARGET_WORDS
    TARGET_WORDS = sorted(set(TARGET_WORDS) | {str(t) for t in types if t})
    return TARGET_WORDS


def _word(object_id: str) -> str:
    """'cell_phone_2' -> 'cell_phone'."""
    head, _, tail = object_id.rpartition("_")
    return head if tail.isdigit() else object_id


class JevRouter:
    """The labelling half of System 1, on Jev. Same surface the SystemOne composite expects of a router."""

    def __init__(self, api_key: str, on_change: Callable[[], None], model: str | None = None,
                 client: Any = None, timeout: float = ROUTE_TIMEOUT_S) -> None:
        self.api_key = api_key
        self.model = model or "jev"
        self._model_arg = model
        self._client = client
        self._on_change = on_change
        self.timeout = timeout
        self.status, self.detail = "connecting", ""
        self.stats = ChannelStats()
        self.latencies: list[float] = []
        self.context: dict[str, Any] = {}
        self.last_said = ""
        self.last_answer: dict[str, Any] = {}     # the raw probabilities of the latest call, for the page

    def _set(self, status: str, detail: str = "") -> None:
        self.status, self.detail = status, detail
        self._on_change()

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
            self._client = AsyncTypeSafeClient(api_key=self.api_key, model=self._model_arg, timeout=self.timeout,
                                               retry=RetryPolicy(max_retries=0))   # a retry would blow the budget
        return self._client

    async def run(self) -> None:
        """Check the key with one call, then stay ready (every call is independent)."""
        if not self.api_key:
            self._set("error", "TYPESAFE_API_KEY is not set")
            return
        while True:
            label = await self.ask("hello", 5.0, None, warmup=True)
            if label is not None:
                self._set("ready", self.model)
                break
            self._set("error", self.stats.last_error or "no answer")
            if "Authentication" in self.stats.last_error or "Permission" in self.stats.last_error:
                return                                   # a bad key won't fix itself
            await asyncio.sleep(5.0)
        await asyncio.Event().wait()                     # stay alive next to the observer

    async def state(self, rendered: str) -> None:
        try:
            self.context = json.loads(rendered)
        except ValueError:
            pass

    def questions(self) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul
        words = sorted(set(TARGET_WORDS) | {_word(o) for o in self.context.get("in_view") or []}
                       | {_word(o) for o in (self.context.get("holding") or {}).values() if o})
        return {
            "kind": Choice(instructions="What does the user's message (in 'message') do to the robot's task?",
                           criteria=KIND_CRITERIA),
            "answer": Choice(instructions="Is the message a yes or a no to a yes/no question the robot just asked "
                                          "(robot_last_said)?", criteria=ANSWER_CRITERIA),
            "replaces_task": Noul(instructions="The message cancels or replaces the robot's current task (robot.goal), "
                                              "rather than adjusting it, adding to it or asking about it.",
                                  criteria={"true": "the current task is over", "false": "the current task goes on"}),
            "target": Choice(instructions="Which object does the message name? 'none' if it names no object.",
                             criteria={w.replace("_", " "): None for w in words} | {"none": "no object named"}),
        }

    async def ask(self, text: str, timeout: float, fallback: Any, warmup: bool = False) -> Any:
        state = {"robot": self.context, "robot_last_said": self.last_said or None, "message": text}
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(self._get_client().system_one(state, self.questions()), timeout)
        except Exception as e:                           # timeouts, API errors: the planner classifies
            self.stats.last_error = f"{type(e).__name__}: {e}"[:300]
            return fallback
        dt = time.monotonic() - t0
        if not warmup:
            self.latencies = (self.latencies + [dt])[-200:]
        self.model = r.model or self.model
        self.stats.context_tokens = r.usage.input_tokens or 0
        self.stats.total_tokens += (r.usage.input_tokens or 0) + (r.usage.output_tokens or 0)
        return self.label(r)

    def label(self, r: Any) -> dict[str, Any] | None:
        """A Jev response as the System 1 contract's label (plus the top probabilities, for the page)."""
        kind = r.choices.get("kind")
        if kind is None or kind.choice not in KINDS:
            return None
        answer = r.choices.get("answer")
        says = None
        if answer is not None and answer.choice in ("yes", "no") and answer.probabilities.get(answer.choice, 0) >= 0.5:
            says = answer.choice == "yes"
        replaces = r.nouls.get("replaces_task")
        target = r.choices.get("target")
        top = sorted(kind.probabilities.items(), key=lambda kv: -kv[1])[:3]
        self.last_answer = {name: a.model_dump() for name, a in r.answers.items()}
        return {"kind": kind.choice,
                "target": "" if target is None or target.choice == "none" else target.choice,
                "replaces_task": bool(replaces is not None and replaces.noul >= 0.5),
                "says_yes": says,
                "confidence": round(float(kind.probabilities.get(kind.choice, 0.0)), 3),
                "probabilities": {k: round(v, 3) for k, v in top}}


class JevSystemOne(SystemOne):
    """SystemOne with the router swapped for Jev; the Gemini Live observer is unchanged."""

    def __init__(self, typesafe_key: str, gemini_key: str, on_status: Callable[[str, str], None] | None = None,
                 jev_model: str | None = None, jev_client: Any = None, **kw: Any) -> None:
        super().__init__(gemini_key, on_status, **kw)
        self.router = JevRouter(typesafe_key, self._channel_changed, model=jev_model, client=jev_client,
                                timeout=self.route_timeout)
        self.status, self.detail = "connecting", ""

    def _channel_changed(self) -> None:
        r, o = self.router, self.observer
        detail = f"labels {r.model}: {r.status}; observations {o.model}: {o.status}"
        err = r.detail if r.status == "error" else (o.detail if o.status == "error" else "")
        if err:
            detail += f" ({err})"
        if (r.status, detail) != (self.status, self.detail):
            self.status, self.detail = r.status, detail
            self._on_status(r.status, detail)

    async def run(self) -> None:
        if not self.api_key:                              # no Gemini key: label with Jev, observe nothing
            await self.router.run()
            return
        await asyncio.gather(self.router.run(), self.observer.run())

    async def robot_said(self, text: str) -> None:
        if text:
            self.router.last_said = text

    def describe(self) -> str:
        lat = self.router.latencies
        med = f"{statistics.median(lat) * 1000:.0f} ms median" if lat else "no calls yet"
        o = self.observer.stats
        return (f"labels {self.router.model} ({med}, {self.router.stats.context_tokens} tokens a call); "
                f"observer {o.context_tokens} tokens, {o.rotations} fresh sessions")


def create(on_status: Callable[[str, str], None]) -> JevSystemOne:
    """The server's factory: SYSTEM1=brains.system1_jev:create."""
    return JevSystemOne(os.environ.get("TYPESAFE_API_KEY", ""), os.environ.get("GEMINI_API_KEY", ""), on_status,
                        jev_model=os.environ.get("SYSTEM1_JEV_MODEL") or None,
                        model=os.environ.get("SYSTEM1_MODEL", "gemini-3.8-live"))
