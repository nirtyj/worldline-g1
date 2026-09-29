"""System 1: Gemini 3.8 Live, always on. The robot's fast attention.

Two Live sessions, so that labelling a message never waits on vision:

    router     text only: the fused state, what the robot says, and every message you
               type (stop included). Answers each message with one route call:
               route(kind, target, replaces_task, says_yes, confidence).
    observer   the fused state and the head-camera frames the frame gate lets through.
               Answers "what did you notice?" with one observe call: observe(items).

The runtime decides what a label means and what an observation is worth: a label
is applied only when it is valid and confident enough, otherwise the planner
classifies; an observation enters belief as an unverified hint. System 1 never
speaks and has no robot tools.

    s1 = SystemOne(api_key, on_status)
    asyncio.create_task(s1.run())                 # keeps both sessions open, reconnects
    await s1.update(context_dict)                 # the fused state, when it changes
    await s1.frame(jpeg, "kitchen_counter_1a")    # a frame worth seeing
    await s1.robot_said("On my way.")
    route = await s1.route("Bring me the mug.")   # -> dict, or None if unsure, slow or down
    items = await s1.observe()                    # -> [{"what", "where", "confidence"}]

Plug it into the server with SYSTEM1=brains.system1:create (reads GEMINI_API_KEY,
and SYSTEM1_MODEL to override the model).

Why it's built this way (measured on gemini-3.8-live, 12 messages per setup):
- Latency grows with a session's context: a route took 0.7 s with 2k tokens, 1.4-2.8 s
  at 6k and 2.6 s at 15k, i.e. past the 2 s budget after a few minutes of frames. So
  routing has its own text-only session (about 1-2k tokens), and each session starts
  over (between questions, with the latest state restored) once its context passes a
  limit. A small sliding window was tried first; the server rejected the trimmed
  session ("1007 invalid argument").
- Tool replies are SILENT (and the tools declared NON_BLOCKING, 3.8 Live's default):
  otherwise the model reacts to each reply, and the next message waits behind that.
  Median 1.00 s -> 0.60 s; p90 1.16 s -> 0.7-0.9 s.
- Messages go in with send_client_content, not send_realtime_input: realtime text
  was slower here (1.19 s median), and client content keeps the order of state,
  robot lines and messages.
- Each question is numbered ("USER #7: ...") and the call must echo the number, so a
  late call for an earlier message is never taken as the label of a newer one. If the
  model talks instead of calling, the caller gets its fallback when that turn ends.
- gemini-3.8-live doesn't take a thinking level and only answers in audio (TEXT is
  rejected); the audio is never played. gemini-3.1-flash-live-preview routed the same
  12 messages about 0.15 s faster (SYSTEM1_MODEL=gemini-3.1-flash-live-preview).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from brains.interface import KINDS

MODEL = "gemini-3.8-live"
ROUTE_TIMEOUT_S = 2.0          # longer than this and the planner classifies instead
OBSERVE_TIMEOUT_S = 8.0
MAX_OBSERVATIONS = 3
UNSEEN_BY_OBSERVER = ("goal", "own_goal", "robot_last_said")   # what someone wants primes what it "sees"
ROUTER_MAX_TOKENS = 3000       # the router starts over past this (it starts at about 1.2k)
OBSERVER_MAX_TOKENS = 4000     # the observer too (about 25 frames at low resolution)

ROUTER_SYSTEM = """You are System 1 of a home robot: its fast ears. You never speak and never \
answer the user; the robot's planner does that. You receive, as they happen:
- STATE: the robot's fused state (where it is, what it holds, what it sees, the task)
- ROBOT SAID: what the robot just said
- USER #n: a message the user typed

After every USER #n message, call route exactly once:
  turn           n, the number after USER #
  kind           request (a new task: "bring me the mug", "what's on the table?")
                 correction (changes the current task: "no, the alarm clock instead")
                 addition (adds a task, keeps the current one: "also grab the bread")
                 question (needs an answer, changes nothing: "how long will it take?")
                 stop (halt now: "stop", "hold on", "freeze", "wait wait")
                 resume (carry on after a stop: "okay, go ahead")
                 answer (answers the robot's own question: "the red one", "yes")
                 constraint (changes how to do the task: "don't go into the bedroom")
                 observation (tells the robot something about the home: "my keys are usually on the counter")
                 chitchat (nothing to do: "thanks", "nice job")
  target         the object the message is about, in the user's words ("mug", "alarm clock"), or ""
  replaces_task  true only when the message cancels or replaces the task in STATE
  says_yes       "yes" or "no" when the message answers a yes/no question the robot just asked \
(the last ROBOT SAID ends with one): "have a look" is yes, "not now" is no. Otherwise "none".
  confidence     0 to 1: how sure you are of the kind. Below 0.5 the planner decides instead, \
so give a low number when a message could be read two ways.

A reply to a question the robot just asked is kind answer, even when it sounds like a command \
("have a look", "go check", "the kitchen"). STATE and ROBOT SAID are context: never call route \
for them and never reply to them. Never answer the user and never speak: your audio is never \
played. Your only output is the route call."""

OBSERVER_SYSTEM = """You are the eyes of a home robot. You never speak. You receive, as they happen:
- STATE: the robot's fused state (where it is, what it holds, what it already sees, the task)
- FRAME: a frame from its head camera, with the keypoint it was taken at
- OBSERVE #n: a request to report what you noticed

When you get OBSERVE #n, call observe exactly once, with turn n and at most 3 things worth the robot \
knowing from the frames since the last OBSERVE: something that changed, something unexpected, a hazard \
(something on the floor, a spill, an open door or drawer), or a person. Don't list furniture, and \
don't repeat objects already in STATE's in_view. Call observe with an empty list when nothing is \
worth saying. Name only what you can clearly see; if a small object could be one of several things, \
describe it ("a small green round object") instead of guessing what it is. \
The robot's own arms, hands and anything they hold may appear at the edges of the frame; never report them.
  what        one short sentence ("the fridge door is open")
  where       the keypoint of the frame it is in (from its FRAME line), or ""
  confidence  0 to 1

STATE and FRAME are context: never reply to them. Never speak: your audio is never played. \
Your only output is the observe call."""

ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer", "description": "The number after USER #."},
        "kind": {"type": "string", "enum": list(KINDS)},
        "target": {"type": "string", "description": "The object the message is about, or empty."},
        "replaces_task": {"type": "boolean"},
        "says_yes": {"type": "string", "enum": ["yes", "no", "none"]},
        "confidence": {"type": "number", "description": "0 to 1"},
    },
    "required": ["turn", "kind", "confidence"],
}

OBSERVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn": {"type": "integer", "description": "The number after OBSERVE #."},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "where": {"type": "string", "description": "Keypoint from the FRAME line, or empty."},
                    "confidence": {"type": "number", "description": "0 to 1"},
                },
                "required": ["what"],
            },
        },
    },
    "required": ["turn", "items"],
}


def _clamp01(v: Any, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else max(0.0, min(1.0, f))


def normalize_route(args: dict[str, Any]) -> dict[str, Any] | None:
    """A route call's arguments as the contract's label, or None if the kind is invalid."""
    kind = args.get("kind")
    if kind not in KINDS:
        return None
    says = {"yes": True, "no": False}.get(str(args.get("says_yes", "")).lower())
    return {"kind": kind, "target": str(args.get("target") or "").strip()[:60],
            "replaces_task": bool(args.get("replaces_task", False)), "says_yes": says,
            "confidence": _clamp01(args.get("confidence"), 0.0)}      # missing -> 0: the planner decides


def normalize_items(args: dict[str, Any]) -> list[dict[str, Any]]:
    """An observe call's arguments as the contract's items: at most three, each with text."""
    out = []
    for it in args.get("items") or []:
        if not isinstance(it, dict):
            continue
        what = str(it.get("what") or "").strip()[:140]
        if not what:
            continue
        where = str(it.get("where") or "").strip() or None
        out.append({"what": what, "where": where, "confidence": _clamp01(it.get("confidence"), 0.5)})
        if len(out) == MAX_OBSERVATIONS:
            break
    return out


@dataclass
class _Ask:
    tool: str                        # "route" or "observe"
    turn: int                        # the number the model has to echo back
    future: asyncio.Future[Any]
    fallback: Any
    spoke: str = ""                  # anything it said instead of calling the tool (never played)

    def resolve(self, value: Any) -> None:
        if not self.future.done():
            self.future.set_result(value)

    def answers(self, name: str, args: dict[str, Any]) -> bool:
        """A call answers this question if the tool matches and the echoed turn (when given) does.
        A late call for an earlier question carries that question's number and is ignored."""
        if name != self.tool:
            return False
        turn = args.get("turn")
        try:
            return turn is None or int(turn) == self.turn
        except (TypeError, ValueError):
            return False


@dataclass
class ChannelStats:
    reconnects: int = 0
    rotations: int = 0               # fresh sessions started because the context grew too big
    context_tokens: int = 0          # the prompt size of the latest answer: how big the context is
    total_tokens: int = 0            # everything billed so far
    last_error: str = ""
    spoke: list[str] = field(default_factory=list)   # the last few times it talked instead of calling


class _Channel:
    """One Live session with one tool: connect, keep it open, ask numbered questions, start over
    when the context grows past ``max_tokens``, resume after the server's go-away notice."""

    def __init__(self, name: str, system: str, tool: str, schema: dict[str, Any], tag: str,
                 normalize: Callable[[dict[str, Any]], Any], model: str, max_tokens: int,
                 client: Callable[[], Any], on_change: Callable[[], None], reconnect_delay: float) -> None:
        self.name, self.system, self.tool, self.schema, self.tag = name, system, tool, schema, tag
        self.normalize = normalize
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._on_change = on_change
        self.reconnect_delay = reconnect_delay
        self.status, self.detail = "connecting", ""
        self.session: Any = None
        self.stats = ChannelStats()
        self._resume: str | None = None
        self._ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._inflight: _Ask | None = None
        self._turns = 0
        self.last_state = ""                          # restored into a fresh session
        self.last_said = ""

    def _set(self, status: str, detail: str = "") -> None:
        self.status, self.detail = status, detail
        self._on_change()

    def _config(self) -> Any:
        from google.genai import types
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],               # 3.8 Live answers in audio only; it's never played
            system_instruction=self.system,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name=self.tool, description=f"Answer {self.tag} #n.", parameters_json_schema=self.schema,
                behavior=types.Behavior.NON_BLOCKING)])],
            output_audio_transcription=types.AudioTranscriptionConfig(),   # to log it if it talks instead
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,    # frames: keep them cheap
            context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
            session_resumption=types.SessionResumptionConfig(handle=self._resume),
        )

    async def run(self) -> None:
        backoff = self.reconnect_delay
        while True:
            rotated = False
            try:
                self._set("connecting", self.model)
                async with self._client().aio.live.connect(model=self.model, config=self._config()) as session:
                    self.session = session
                    await self._restore()
                    self._ready.set()
                    self._set("ready", self.model)
                    backoff = self.reconnect_delay
                    rotated = await self._receive(session)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats.last_error = f"{type(e).__name__}: {e}"[:300]
                self._set("error", self.stats.last_error)
            finally:
                self.session = None
                self._ready.clear()
                if self._inflight is not None:
                    self._inflight.resolve(self._inflight.fallback)
            if rotated:
                continue                                     # a fresh session right away
            self.stats.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    async def _restore(self) -> None:
        """Give a new session the latest state and robot line (a resumed one already has them)."""
        if self.last_state:
            await self.send([{"text": f"STATE {self.last_state}"}], turn_complete=False)
        if self.last_said:
            await self.send([{"text": f"ROBOT SAID: {self.last_said}"}], turn_complete=False)

    async def _receive(self, session: Any) -> bool:
        """Returns True to start a fresh session (context too big), False to reconnect and resume."""
        from google.genai import types
        while True:
            async for m in session.receive():
                upd = m.session_resumption_update
                if upd is not None and upd.new_handle:
                    self._resume = upd.new_handle
                if m.go_away is not None:
                    return False
                if m.usage_metadata is not None:
                    self.stats.context_tokens = m.usage_metadata.prompt_token_count or self.stats.context_tokens
                    self.stats.total_tokens += m.usage_metadata.total_token_count or 0
                ask = self._inflight
                if m.tool_call is not None:
                    responses = []
                    for fc in m.tool_call.function_calls or []:
                        args = dict(fc.args or {})
                        if ask is not None and ask.answers(fc.name, args):
                            ask.resolve(self.normalize(args))
                        responses.append(types.FunctionResponse(
                            id=fc.id, name=fc.name, response={"ok": True},
                            scheduling=types.FunctionResponseScheduling.SILENT))   # nothing to react to
                    if responses:
                        async with self._send_lock:
                            await session.send_tool_response(function_responses=responses)
                sc = m.server_content
                if sc is None or ask is None:
                    continue
                if sc.output_transcription is not None and sc.output_transcription.text:
                    ask.spoke += sc.output_transcription.text
                # A turn that ends with talk and no call: it answered instead of calling, so hand back
                # the fallback now. (A bare turn_complete may belong to an earlier question, so it
                # resolves nothing on its own.)
                if sc.turn_complete and ask.spoke and not ask.future.done():
                    self.stats.spoke = (self.stats.spoke + [ask.spoke.strip()[:200]])[-5:]
                    ask.resolve(ask.fallback)
            idle = self._inflight is None or self._inflight.future.done()
            if self.stats.context_tokens > self.max_tokens and idle:
                self.stats.rotations += 1                    # between questions: start over small
                self.stats.context_tokens = 0
                self._resume = None
                self.session = None                          # nothing more goes to the closing session:
                self._ready.clear()                          # questions wait for the fresh one
                self._set("connecting", "fresh session")
                return True

    async def send(self, parts: list[Any], turn_complete: bool) -> bool:
        session = self.session
        if session is None:
            return False
        try:
            async with self._send_lock:
                await session.send_client_content(turns=[{"role": "user", "parts": parts}],
                                                  turn_complete=turn_complete)
            return True
        except Exception as e:                               # the run loop reconnects
            self.stats.last_error = f"send: {type(e).__name__}: {e}"[:300]
            return False

    async def ask(self, text: str, timeout: float, fallback: Any) -> Any:
        """Send one numbered question ("USER #7: ...") and wait for the matching call. During a switch
        to a fresh session (about half a second) it waits for the new one instead of giving up."""
        if not self._ready.is_set():
            try:
                await asyncio.wait_for(self._ready.wait(), min(1.0, timeout / 2))
            except TimeoutError:
                return fallback
        if self.session is None:
            return fallback
        cur = self._inflight
        if cur is not None:
            cur.resolve(cur.fallback)                        # a newer question wins
        self._turns += 1
        ask = _Ask(self.tool, self._turns, asyncio.get_running_loop().create_future(), fallback)
        self._inflight = ask
        try:
            if not await self.send([{"text": f"{self.tag} #{ask.turn}: {text}"}], turn_complete=True):
                return fallback
            try:
                return await asyncio.wait_for(asyncio.shield(ask.future), timeout)
            except TimeoutError:
                return fallback
        finally:
            ask.resolve(fallback)
            if self._inflight is ask:
                self._inflight = None

    async def state(self, rendered: str) -> None:
        if rendered != self.last_state:
            self.last_state = rendered                   # kept even if the send misses: a new session gets it
            await self.send([{"text": f"STATE {rendered}"}], turn_complete=False)

    @property
    def busy(self) -> bool:
        return self._inflight is not None and not self._inflight.future.done()


@dataclass
class Stats:
    routes: int = 0
    routed: int = 0
    observes: int = 0
    observed: int = 0
    fallbacks: int = 0


class SystemOne:
    def __init__(self, api_key: str, on_status: Callable[[str, str], None] | None = None,
                 model: str = MODEL, client: Any = None, route_timeout: float = ROUTE_TIMEOUT_S,
                 observe_timeout: float = OBSERVE_TIMEOUT_S, reconnect_delay: float = 1.0) -> None:
        self.api_key = api_key
        self.model = model
        self.route_timeout = route_timeout
        self.observe_timeout = observe_timeout
        self._client = client
        self._on_status = on_status or (lambda status, detail: None)
        self.status, self.detail = "connecting", ""
        self.stats = Stats()
        common = dict(model=model, client=self._get_client, on_change=self._channel_changed,
                      reconnect_delay=reconnect_delay)
        self.router = _Channel("router", ROUTER_SYSTEM, "route", ROUTE_SCHEMA, "USER", normalize_route,
                               max_tokens=ROUTER_MAX_TOKENS, **common)
        self.observer = _Channel("observer", OBSERVER_SYSTEM, "observe", OBSERVE_SCHEMA, "OBSERVE", normalize_items,
                                 max_tokens=OBSERVER_MAX_TOKENS, **common)
        self._frames_since_observe = 0
        self._frame_wheres: list[str | None] = []       # where each frame since the last observe was taken
        self._observer_rotations = 0

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _channel_changed(self) -> None:
        """System 1 is ready when the router is: routing is what the runtime waits on."""
        r, o = self.router, self.observer
        status = r.status
        detail = f"{self.model}: router {r.status}, observer {o.status}"
        err = r.detail if r.status == "error" else (o.detail if o.status == "error" else "")
        if err:
            detail += f" ({err})"
        if (status, detail) != (self.status, self.detail):
            self.status, self.detail = status, detail
            self._on_status(status, detail)

    async def run(self) -> None:
        """Keep both sessions open for as long as the server runs."""
        if not self.api_key:
            self.status, self.detail = "error", "GEMINI_API_KEY is not set"
            self._on_status(self.status, self.detail)
            return
        await asyncio.gather(self.router.run(), self.observer.run())

    def describe(self) -> str:
        """One line for the page's call list: context sizes and fresh sessions."""
        r, o = self.router.stats, self.observer.stats
        return (f"router {r.context_tokens} tokens, observer {o.context_tokens} tokens, "
                f"{r.rotations + o.rotations} fresh sessions")

    # ------------------------------------------------------------------
    # The contract (brains/interface.py System1)
    # ------------------------------------------------------------------
    async def route(self, text: str) -> dict[str, Any] | None:
        self.stats.routes += 1
        r = await self.router.ask(text, self.route_timeout, None)
        if r is None:
            self.stats.fallbacks += 1
        else:
            self.stats.routed += 1
        return r

    async def update(self, context: dict[str, Any]) -> None:
        rendered = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
        await self.router.state(rendered)
        # The observer never learns what anyone wants: told the goal was "bring the apple", it reported
        # "an apple on the counter" 5 times out of 5 for a frame with a head of lettuce (0/5 without).
        seen = {k: v for k, v in context.items() if k not in UNSEEN_BY_OBSERVER}
        await self.observer.state(json.dumps(seen, sort_keys=True, separators=(",", ":"), default=str))

    async def frame(self, jpeg: bytes, where: str | None) -> None:
        from google.genai import types
        if self.observer.stats.rotations != self._observer_rotations:   # frames sent before a fresh session
            self._observer_rotations = self.observer.stats.rotations      # are gone with the old one
            self._frames_since_observe, self._frame_wheres = 0, []
        label = f"FRAME at {where}" if where else "FRAME between keypoints"
        if await self.observer.send([{"text": label}, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
                                    turn_complete=False):
            self._frames_since_observe += 1
            self._frame_wheres.append(where)

    async def robot_said(self, text: str) -> None:
        if text:
            self.router.last_said = text
            await self.router.send([{"text": f"ROBOT SAID: {text}"}], turn_complete=False)

    async def observe(self) -> list[dict[str, Any]]:
        if self.observer.stats.rotations != self._observer_rotations:
            self._observer_rotations = self.observer.stats.rotations
            self._frames_since_observe, self._frame_wheres = 0, []
        n = self._frames_since_observe
        if n == 0 or self.observer.busy:
            return []
        self.stats.observes += 1
        self._frames_since_observe = 0
        wheres, self._frame_wheres = self._frame_wheres, []
        items = await self.observer.ask(f"{n} new frame{'s' if n != 1 else ''} since the last OBSERVE",
                                        self.observe_timeout, [])
        # The model sees every frame still in its context and sometimes names an older frame's place
        # (where the robot started, say). Keep a place only if a frame since the last OBSERVE was taken
        # there; otherwise use the newest frame's place (None between keypoints: the runtime then uses
        # where the robot is).
        valid = {w for w in wheres if w}
        latest = wheres[-1] if wheres else None
        for it in items:
            if it["where"] not in valid:
                it["where"] = latest
        self.stats.observed += len(items)
        return items


def create(on_status: Callable[[str, str], None]) -> SystemOne:
    """The server's factory: SYSTEM1=brains.system1:create."""
    return SystemOne(os.environ.get("GEMINI_API_KEY", ""), on_status, model=os.environ.get("SYSTEM1_MODEL", MODEL))
