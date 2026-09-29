"""The runtime: event loop, utterance handling, rules, executor, speech and trace.

Four asyncio tasks:

  listen     takes each utterance from the user. "Stop" halts the robot right
             here, before any model call.
  interpret  classifies utterances in order and applies them. Only a
             correction cancels anything; additions, questions and answers
             leave running actions alone.
  think      asks the brain for the next step, checks it against the rules,
             and dispatches it. Body actions run in the background; the brain
             is asked again when something changes.
  speech     plays queued lines in order (see skills.SpeechQueue).

Rules, enforced here rather than in a prompt:

  1. A late result updates belief, never the plan.
  2. After a cancel, every resource it touched is UNKNOWN until the robot looks.
  3. Tool calls are checked before they run; a rejection says what to do next.
  4. The brain sees belief, never simulator ground truth.
  5. "Stop" skips the brain.
  6. Every skill and brain call has a timeout.
  7. Retries need new information and fit a budget; then tell the user.
  8. One clock everywhere.
"""

from __future__ import annotations

import asyncio
import difflib
import itertools
import re
import time
from typing import Any

from brains.interface import KINDS, TOOLS, UNKNOWN, BrainInput, HistoryEntry, ToolCall

from .fused_state import (OBSERVATION_PERIOD_S, Directive, FusedRuntimeState,
                          RobotObservationAdapter)
from . import layout
from .episodes import EpisodeLog, load_episodes
from .narrator import Narrator
from .memory import SpatialMemory
from .procedures import ProceduralGraph, step_of
from .recall import Recaller
from .persona import GOAL_TIMEOUT_S, STUCK_S, Persona
from .skills import Outcome, SpeechItem, SpeechQueue, navigate_timeout, run_goal
from .state import ARMS, ActionHandle, BeliefState, TaskState, TraceLog

BODY = ("navigate", "pick", "place")
SENSE = ("look", "reachability")
CLASSIFY_TIMEOUT = 8.0
BRAIN_TIMEOUT = 20.0
ECHO_S = 15.0             # an observation of what belief confirmed this recently, at the same spot, is an echo
SENSE_TIMEOUT = 12.0       # a look scans six to nine views, each a render
PICK_TIMEOUT = 15.0
PLACE_TIMEOUT = 12.0
REACH_FRESH_S = 30.0
MAX_DECISIONS_PER_WAKE = 40

STOP_RE = re.compile(r"^\W*(stop|freeze|halt)\b|^\W*(please|hey|robot)\W+stop\b", re.I)
NOT_STOP_RE = re.compile(r"\b(don.?t|do not|never)\s+stop\b", re.I)


def is_stop(text: str) -> bool:
    return bool(STOP_RE.search(text)) and not NOT_STOP_RE.search(text)


def fallback_kind(text: str) -> str:
    """Keyword classification, used only when the brain fails or times out."""
    t = text.lower().strip()
    if is_stop(t):
        return "stop"
    if re.search(r"\b(go ahead|carry on|continue|keep going|resume)\b", t):
        return "resume"
    if re.search(r"^(actually|wait|sorry|no\b|no,|i meant)|\binstead\b|\bmake it\b", t):
        return "correction"
    if re.search(r"\b(also|too|as well)\b", t):
        return "addition"
    if t.endswith("?"):
        return "question"
    if re.search(r"^the \w+ one\b|^(the )?(left|right|red|blue)\b", t):
        return "answer"
    return "request"


class Runtime:
    def __init__(self, robot: Any, user: Any, brain: Any, clock: Any) -> None:
        self.robot, self.user, self.brain, self.clock = robot, user, brain, clock
        self.map = robot.lookup_keypoints()
        now = clock.now()
        self.belief = BeliefState()
        self.belief.load_memory(robot.memory())
        self.memory = SpatialMemory(self.map.get("scene"))      # what earlier sessions saw
        self.belief.load_memory(self.memory.entries(now))
        self._memory_rev = self.belief.rev
        self.persona = Persona()
        self._surface_xy = {k: tuple(v["xy"]) for k, v in self.map["surfaces"].items() if v.get("xy")}
        self._busy_t = now                    # last moment anything was going on
        base = robot.base_state()
        self.belief.set_pose(base["at"], None, "odometry", now)
        for arm in ARMS:
            grip = robot.gripper(arm)
            empty = not (grip["closed"] and grip["width"] > 0.01)
            self.belief.set_hand(arm, None if empty else UNKNOWN, "gripper", now, verified=empty)
        self.task = TaskState()
        self.observations = RobotObservationAdapter(robot)
        self.state = FusedRuntimeState(self.belief, self.task)
        self.state.ingest(self.observations.sample(now))
        self.history: list[HistoryEntry] = []
        self._ids = itertools.count(1)
        self.actions: dict[int, ActionHandle] = {}
        self.tracer = TraceLog(clock)
        scene = self.map.get("scene")
        self.episodes = EpisodeLog(scene)                     # what happens, written as it happens
        self.tracer.sinks.append(self.episodes.write)
        deliver_to = ((self.map.get("people") or {}).get("user") or {}).get("deliver_to_surface")
        self.tracer.log("session_start", scene=scene, wall=round(time.time(), 1), deliver_to=deliver_to)
        self.recaller = Recaller(scene)
        self.procedures = ProceduralGraph.load()              # what usually works, learned from all episodes
        self.procedures.learn(load_episodes())
        self._task_row: int | None = None                    # where the current request's trace starts
        self._places = {oid: ob.where.value for oid, ob in self.belief.objects.items()}
        self._deliver_to = deliver_to
        self.narrator = Narrator(self.map, self.belief, deliver_to)     # progress lines for the chat
        self.tracer.sinks.append(self.narrator.row)
        self._delivered: set[int] = set()                    # place entries already logged as deliveries
        self.speech = SpeechQueue(robot, clock)
        self._wake_evt = asyncio.Event()
        self._utt_q: asyncio.Queue = asyncio.Queue()
        self._sense_lock = asyncio.Lock()
        self._note: str | None = None
        self._picked_from: dict[str, str] = {}      # object -> the surface it was last picked up from
        self._goal_missed: set[str] = set()          # objects whose last place failed its goal check
        self._canceled: list[ActionHandle] = []
        self._reconcile_task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._errors: list[BaseException] = []
        self._thinking = False
        self._interpreting = False
        self._rejects = 0
        self._brain_errors = 0
        self._last_motion_t = now
        self._emergency_ids = itertools.count(1)
        self._directives_by_utterance: dict[str, Directive] = {}
        self.noticed: list[dict[str, Any]] = []             # what System 1 noticed this session
        self.step_mode = False                               # the page's step mode: wait before each planner call
        self._step_evt = asyncio.Event()
        self._stepping: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Entry points used by the server (ui/server.py)
    # ------------------------------------------------------------------
    async def run(self) -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._listen())
                tg.create_task(self._interpret_loop())
                tg.create_task(self._think_loop())
                tg.create_task(self.speech.run())
                tg.create_task(self._observation_loop())
                tg.create_task(self._persona_loop())
        finally:
            for task in list(self._bg):
                task.cancel()
            self.memory.update(self.belief, self.clock.now(), force=True)
            self.episodes.close()

    def idle(self) -> bool:
        return (not self.actions and not self.speech.busy() and self._utt_q.empty()
                and not self._interpreting and not self._thinking and not self._wake_evt.is_set()
                and not self._reconciling())

    def trace(self) -> list[dict[str, Any]]:
        return self.tracer.rows + [{"t": round(self.clock.now(), 3), "type": "final_belief",
                                    "belief": self.belief.to_dict()}]

    def runtime_snapshot(self) -> dict[str, Any]:
        active = [self._history_dict(e) for e in self.history if not e.finished]
        recent = [self._history_dict(e) for e in self.history[-12:]]
        return self.state.snapshot(active=active, recent_actions=recent)

    def emergency_stop(self, text: str = "stop", source: str = "safety") -> bool:
        """Priority-zero entry point for partial voice, UI, and safety monitors."""
        if self.task.paused:
            return False
        now = self.clock.now()
        directive = Directive(
            id=f"emergency-{next(self._emergency_ids)}", t=round(now, 3), kind="stop",
            text=text.strip() or "stop", source=source, confidence=1.0,
        )
        self._accept_directive(directive)
        self._stop_now_id(directive.id, reason=source)
        self._wake("emergency stop")
        return True

    def _refresh_observation(self) -> set[str]:
        frame = self.observations.sample(self.clock.now())
        return self.state.ingest(frame)

    async def _observation_loop(self) -> None:
        """Continuously replace latest state; wake only on useful idle changes."""
        while True:
            rev = self.belief.rev
            self._refresh_observation()
            robot = self.state.robot or {}
            pose = robot.get("pose") or {}
            xy = (pose["x"], pose["z"]) if pose.get("x") is not None else pose.get("xy")
            self.narrator.pose(xy, bool(robot.get("moving")), self.clock.now())
            for kind, text in self.narrator.take():
                self.tracer.log("narrate", kind=kind, text=text)
            # Only a change to what the robot believes (an object seen somewhere new)
            # is worth a model call; poses and distances change on every frame.
            # Action completion already wakes the planner, so only wake when idle.
            if (self.belief.rev != rev and self.task.utterances and not self.actions
                    and not self._thinking):
                self._wake("perception changed belief")
            if self.belief.rev != self._memory_rev:     # a look, a result or the camera taught us something
                self._memory_rev = self.belief.rev
                self._note_places()
                if self.memory.update(self.belief, self.clock.now()):
                    self.tracer.log("memory_saved", objects=len(self.memory.items))
            await self.clock.sleep(OBSERVATION_PERIOD_S)

    # ------------------------------------------------------------------
    # The persona: the robot's own goals, only when nobody needs anything
    # ------------------------------------------------------------------
    async def _persona_loop(self) -> None:
        while True:
            await self.clock.sleep(0.5)
            now = self.clock.now()
            if not self.idle() or self.task.paused:
                self._busy_t = now
            p = self.persona
            if p.goal is not None:
                if p.is_done(self.belief):
                    self._end_own_goal("done")
                elif now - p.goal.t_start > GOAL_TIMEOUT_S or now - self._busy_t > STUCK_S:
                    self._end_own_goal("gave_up")
                continue
            if self.task.paused:
                continue
            last_heard = self.task.utterances[-1].t_end if self.task.utterances else -1e9
            quiet = now - max(self._busy_t, last_heard)
            self._start_own_goal(p.propose(self.belief, self.map, now, quiet))

    def _start_own_goal(self, g: Any) -> None:
        if g is None:
            return
        now = self.clock.now()
        self.tracer.log("persona_goal", id=g.id, drive=g.drive, target=g.target, text=g.text)
        self.state.record("own_goal", now, priority=4, drive=g.drive, target=g.target)
        self._busy_t = now
        if not self._thinking:
            # A goal chained mid-think (right after a look) is seen on the next ask
            # anyway; waking here would re-ask while the next move is running.
            self._wake(f"own goal {g.drive} {g.target}")

    def _end_own_goal(self, outcome: str) -> None:
        g = self.persona.finish(outcome, self.clock.now())
        if g is None:
            return
        self.tracer.log("persona_goal_end", id=g.id, drive=g.drive, target=g.target, outcome=outcome)
        if outcome == "dropped":
            # the user comes first: stop whatever the own goal had running
            self.task.control_epoch += 1          # and drop decisions made for it
            for h in list(self.actions.values()):
                if h.source == "persona" and not h.cancel_requested:
                    h.cancel()
                    self._canceled.append(h)
            self._start_reconcile()

    # ------------------------------------------------------------------
    # Memory: episodes, places, recall, procedures
    # ------------------------------------------------------------------
    def _note_places(self) -> None:
        """Log each verified change of place once: episodes record where things were
        learned to be, and a verified place on the user's surface after a place is a delivery."""
        for oid, ob in self.belief.objects.items():
            where = ob.where.value
            if self._places.get(oid) == where or ob.where.source == "memory":
                continue
            was = self._places.get(oid)
            self._places[oid] = where
            if not ob.where.verified and ob.where.source != "look_absent":
                continue
            self.tracer.log("place_learned", object=oid, place=where, was=was, source=ob.where.source)
        self._check_deliveries()

    def _check_deliveries(self) -> None:
        """A successful place whose object is now verified on the user's surface is a delivery.
        The camera often sees it land before the place goal finishes, so this runs on both."""
        for e in self.history:
            if e.tool != "place" or e.status != "SUCCEEDED" or e.id in self._delivered:
                continue
            ob = self.belief.objects.get(e.args.get("object"))
            if ob is not None and ob.where.verified and ob.where.value == self._deliver_to:
                self._delivered.add(e.id)
                self.tracer.log("delivered", object=ob.id, surface=ob.where.value)

    def _recall(self, call: ToolCall, v: int) -> None:
        query = str(call.args.get("query", "")).strip()
        e = self._entry("recall", {"query": query}, v, call.tag)
        answer = self.recaller.answer(query, self.belief, self.memory, self.map, self.clock.now(), self.tracer.rows)
        e.status, e.data, e.t_end = "SUCCEEDED", {"answer": answer}, round(self.clock.now(), 3)
        self.tracer.log("recall", query=query, answer=answer)

    # ------------------------------------------------------------------
    # System 1: what it is told, and what it notices
    # ------------------------------------------------------------------
    def system1_context(self) -> dict[str, Any]:
        """What System 1 is told about the robot as it changes. Only what the robot
        senses and believes, never the simulator."""
        b = self.belief
        at = b.robot_at.value
        kps = self.map.get("keypoints") or {}
        room = (kps.get(at) or {}).get("room") if at else None
        says = [e for e in self.history if e.tool == "say" and e.status not in ("DROPPED", "REJECTED")]
        return {
            "at": at, "between": list(b.between) if b.between else None,
            "room": (self.map.get("rooms", {}).get(room) or {}).get("label", room) if room else None,
            "moving": bool((self.state.robot or {}).get("moving")),
            "holding": {arm: b.holding[arm].value for arm in ARMS if b.holding[arm].value},
            "in_view": sorted(oid for oid, ob in b.objects.items() if ob.visible),
            "goal": self.task.goal, "stopped": self.task.paused,
            "running": [e.tool for e in self.history if e.status in ("queued", "running") and e.tool != "say"],
            "robot_last_said": says[-1].args.get("text") if says else None,
            "own_goal": self.persona.goal.text if self.persona.goal is not None else None,
        }

    def add_observations(self, items: list[dict[str, Any]], source: str = "system1") -> int:
        """System 1's observations enter as unverified hints: this session's list (shown to
        the planner as NOTICED) and spatial memory's observations, never as object facts."""
        now = self.clock.now()
        # Belief keeps the last keypoint until a drive ends; while moving, "here" isn't known.
        moving = bool((self.state.robot or {}).get("moving"))
        at = None if moving else self.belief.robot_at.value
        kps = self.map.get("keypoints") or {}
        added = 0
        for it in items[:3]:
            text = str(it.get("what") or "").strip()
            if not text:
                continue
            where = it.get("where") if it.get("where") in kps else at
            try:
                conf = max(0.0, min(1.0, float(it.get("confidence", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            if any(o["text"].lower() == text.lower() and o["where"] == where and now - o["t"] < 120
                   for o in self.noticed):
                continue                                   # already noticed here, recently
            echo = self._echoes_belief(text, where, now)
            if echo:                                       # "the spatula is on the counter" right after placing it
                self.tracer.log("observation_dropped", text=text, where=where, why=f"repeats belief: {echo}")
                continue
            obs = {"t": round(now, 2), "text": text, "where": where, "confidence": round(conf, 2), "source": source}
            self.noticed = (self.noticed + [obs])[-30:]
            self.memory.add_observation(text, where, conf, source)
            self.tracer.log("observation", text=text, where=where, confidence=round(conf, 2), source=source)
            added += 1
        return added

    def _echoes_belief(self, text: str, where: str | None, now: float) -> str | None:
        """The object this observation names, if belief confirmed it at the same spot (or in a
        hand) in the last ECHO_S seconds: the camera saw the robot's own pick or place."""
        words = text.lower()
        for oid, ob in self.belief.objects.items():
            kind = str(ob.type or "").replace("_", " ").lower()
            if not kind or not re.search(rf"\b{re.escape(kind)}s?\b", words):
                continue
            w = str(ob.where.value or "")
            if ob.where.verified and now - ob.where.t < ECHO_S and (w == where or w.startswith("hand")):
                return oid
        return None

    def _noticed_for_prompt(self) -> list[dict[str, Any]]:
        """For the prompt: this session's observations, then a few remembered from earlier ones."""
        now = self.clock.now()
        recent = [{**o, "ago_s": now - o["t"]} for o in self.noticed[-6:]]
        seen = {(o["text"].lower(), o["where"]) for o in recent}
        earlier = [o for o in self.memory.observations() if (o["text"].lower(), o.get("where")) not in seen]
        return earlier[-3:] + recent

    def task_steps(self) -> list[str]:
        """The current request's trace as abstract steps (agent/procedures.py step_of)."""
        if self._task_row is None:
            return []
        return [s for s in (step_of(r) for r in self.tracer.rows[self._task_row:]) if s]

    def _guidance(self) -> str:
        """What the procedural graph suggests for the step the current request is at."""
        if self._task_row is None or self.persona.goal is not None:
            return ""
        return self.procedures.guidance(self.task_steps())

    # ------------------------------------------------------------------
    # Step mode: the page holds each planner call until the user presses Step
    # ------------------------------------------------------------------
    def set_step_mode(self, on: bool) -> None:
        self.step_mode = on
        if not on:
            self._step_evt.set()                         # release a call that is waiting

    def step(self) -> None:
        self._step_evt.set()

    def stepping(self) -> dict[str, Any] | None:
        return self._stepping

    async def _step_gate(self) -> None:
        if not self.step_mode:
            return
        self._step_evt.clear()
        ctx = self._ctx()
        preview = getattr(self.brain, "preview", None)
        self._stepping = {"since": round(self.clock.now(), 2), "version": ctx.intent_version,
                          "preview": preview(ctx) if callable(preview) else None}
        self.tracer.log("step_waiting", version=ctx.intent_version)
        try:
            await self._step_evt.wait()
        finally:
            self._stepping = None

    def set_persona(self, level: str | bool) -> None:
        if isinstance(level, bool):
            level = "optimize" if level else "off"
        self.persona.level = level
        if self.persona.goal is not None:
            self._end_own_goal("dropped")
        self.tracer.log("persona_level", level=level)

    @staticmethod
    def _history_dict(entry: HistoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id, "tool": entry.tool, "args": dict(entry.args),
            "status": entry.status, "created_for": entry.created_for,
            "control_epoch": entry.control_epoch, "source": entry.source,
            "t_start": entry.t_start, "t_end": entry.t_end,
            "data": dict(entry.data or {}),
        }

    # ------------------------------------------------------------------
    # Listening and interpreting
    # ------------------------------------------------------------------
    async def _listen(self) -> None:
        while True:
            utt = await self.user.next()
            self.task.utterances.append(utt)
            self.tracer.log("heard", id=utt.id, text=utt.text)
            self.state.record("utterance", self.clock.now(), priority=2, id=utt.id, text=utt.text)
            agreed = self.persona.heard(utt.text, self.clock.now(),
                                        reply=(getattr(utt, "directive", None) or {}).get("reply"))
            if self.persona.goal is not None:
                self._end_own_goal("dropped")
            if agreed:      # "yes, have a look": the persona leads, one spot at a time
                self._start_own_goal(self.persona.propose(self.belief, self.map, self.clock.now(), quiet_s=1e9))
            if is_stop(utt.text):
                self._accept_directive(self._directive_for(utt, "stop"))
                self._stop_now(utt, reason="keyword")        # rule 5: no model in the loop
            self._utt_q.put_nowait(utt)

    async def _interpret_loop(self) -> None:
        while True:
            utt = await self._utt_q.get()
            self._interpreting = True
            try:
                kind = await self._classify(utt)
                self.task.kinds[utt.id] = kind
                directive = self._accept_directive(self._directive_for(utt, kind))
                if kind in ("request", "correction"):
                    self._task_row = len(self.tracer.rows)         # the trace of this task starts here
                if kind == "observation":                          # memory, whatever the persona level
                    about = (directive.target or {}).get("object")
                    self.memory.add_note(utt.text, about)      # "it's next to the toaster"
                    self.tracer.log("note_saved", text=utt.text, about=about)
                self.tracer.log("classified", id=utt.id, kind=kind, directive=directive.to_dict())
                self._apply(utt, kind)
            finally:
                self._interpreting = False
            self._wake(f"utterance {utt.id} ({kind})")

    async def _classify(self, utt: Any) -> str:
        try:
            kind = await self.clock.wait_for(self.brain.classify(utt, self._ctx()), CLASSIFY_TIMEOUT)
            if kind in KINDS:
                return kind
            self.tracer.log("classify_bad_kind", id=utt.id, kind=kind)
        except asyncio.TimeoutError:
            self.tracer.log("classify_timeout", id=utt.id)
        except Exception as e:
            self.tracer.log("classify_error", id=utt.id, error=repr(e))
        return fallback_kind(utt.text)

    def _directive_for(self, utt: Any, kind: str) -> Directive:
        existing = self._directives_by_utterance.get(utt.id)
        if existing is not None:
            return existing
        raw = dict(getattr(utt, "directive", None) or {})
        confidence = raw.get("confidence", 1.0 if raw else 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5
        return Directive(
            id=str(raw.get("id") or f"d-{utt.id}"), t=round(float(utt.t_end), 3), kind=kind,
            text=str(raw.get("text") or utt.text),
            source=str(raw.get("source") or "runtime-classifier"),
            target=dict(raw.get("target") or {}), confidence=confidence,
            supersedes=raw.get("supersedes") or None, reply=raw.get("reply") or None,
        )

    def _accept_directive(self, directive: Directive) -> Directive:
        utterance_id = directive.id[2:] if directive.id.startswith("d-u") else None
        if utterance_id and utterance_id in self._directives_by_utterance:
            return self._directives_by_utterance[utterance_id]
        if utterance_id:
            self._directives_by_utterance[utterance_id] = directive
        if any(d.id == directive.id for d in self.task.directives):
            return next(d for d in self.task.directives if d.id == directive.id)
        self.task.directives.append(directive)
        del self.task.directives[:-100]
        self.task.current_directive = directive
        self.state.record("directive", directive.t, priority=0 if directive.kind == "stop" else 2,
                          directive=directive.to_dict())
        return directive

    def _apply(self, utt: Any, kind: str) -> None:
        if kind == "stop":
            if not self.task.paused:
                self._stop_now(utt, reason="classified")
        elif kind == "resume":
            self.task.control_epoch += 1
            self.task.paused = False
        elif kind == "correction":
            self._correct(utt)
        elif kind == "request":
            if not self.actions and not self.speech.busy():
                self.task.intent_version += 1
            self.task.goal = utt.text
        elif kind == "constraint":
            # Preserve useful running motion, but invalidate decisions that
            # were still being generated without the new constraint.
            self.task.control_epoch += 1
        # addition, question, answer, chitchat (and a request while busy): nothing is
        # cancelled; the brain queues or answers it.

    def _stop_now(self, utt: Any, reason: str) -> None:
        self._stop_now_id(utt.id, reason)

    def _stop_now_id(self, event_id: str, reason: str) -> None:
        # Priority zero: the actuator command is deliberately the first side
        # effect.  Logging, speech, reconciliation, and models come afterward.
        receipt = self.robot.halt()
        if self.task.paused:
            # Already stopped (e.g. the partial transcript stopped us and now the
            # final "stop" arrives): halting again is free, a second ack is not.
            self.tracer.log("stop", id=event_id, reason=reason, already_paused=True)
            return
        self.task.control_epoch += 1
        self.task.paused = True
        cut = self.speech.cut_all()
        canceled = []
        for h in list(self.actions.values()):
            if h.resources & {"base", "arm:left", "arm:right"} and not h.done:
                h.cancel()
                self._canceled.append(h)
                canceled.append(h.skill)
        self._refresh_observation()
        confirmed = bool(isinstance(receipt, dict) and receipt.get("stopped"))
        self.tracer.log("stop", id=event_id, reason=reason, canceled=canceled, cut=cut,
                        epoch=self.task.control_epoch, physical_first=True)
        self.state.record("emergency_stop", self.clock.now(), priority=0, id=event_id,
                          reason=reason, canceled=canceled, confirmed=confirmed,
                          control_epoch=self.task.control_epoch)
        self._say_immediate("I've stopped." if confirmed else "Stopping now.")
        self._start_reconcile()

    def _correct(self, utt: Any) -> None:
        self.task.intent_version += 1
        self.task.control_epoch += 1
        self.task.goal = utt.text
        v = self.task.intent_version
        dropped = self.speech.drop_older_than(v)
        canceled = []
        for h in list(self.actions.values()):
            if h.created_for < v and not h.done:
                h.cancel()
                self._canceled.append(h)
                canceled.append(h.skill)
                for r in h.resources:
                    if r.startswith("arm:"):
                        self.belief.mark_hand_unknown(r[4:], "cancel", self.clock.now())
        self.tracer.log("correction", id=utt.id, version=v, epoch=self.task.control_epoch,
                        dropped=dropped, canceled=canceled)
        self.state.record("task_corrected", self.clock.now(), priority=1, id=utt.id,
                          intent_version=v, control_epoch=self.task.control_epoch,
                          canceled=canceled)
        self._start_reconcile()

    # ------------------------------------------------------------------
    # Reconciling after a cancel or a stop
    # ------------------------------------------------------------------
    def _reconciling(self) -> bool:
        return self._reconcile_task is not None and not self._reconcile_task.done()

    def _start_reconcile(self) -> None:
        if not self._canceled or self._reconciling():
            return
        self._reconcile_task = self._spawn(self._reconcile())

    async def _reconcile(self) -> None:
        self.tracer.log("reconcile_start", actions=[h.skill for h in self._canceled])
        arms: set[str] = set()
        while True:
            for h in self._canceled:
                arms |= {r[4:] for r in h.resources if r.startswith("arm:")}
            pending = [h.task for h in self._canceled if h.task is not None and not h.task.done()]
            if not pending:
                break
            await asyncio.wait(pending)       # each skill has its own timeout
        self._canceled.clear()
        now = self.clock.now()
        for arm in arms:                      # rule 2
            hint = self.belief.hints.get(arm)
            self.belief.mark_hand_unknown(arm, "cancel", now, hint=hint)
        if arms:
            await self._observe("after cancel")
        self._refresh_observation()
        self.tracer.log("reconcile_done", belief=self.belief.to_dict()["hands"])
        self.state.record("reconciled", self.clock.now(), priority=1,
                          belief=self.belief.to_dict()["hands"],
                          control_epoch=self.task.control_epoch)
        self._wake("reconciled")

    async def _observe(self, why: str, glance: bool = False) -> None:
        """A harness-initiated look, recorded in history like any other action. A glance
        checks the view the camera already has instead of turning to scan."""
        args = {"glance": True} if glance else {}
        entry = self._entry("look", args, source="harness", tag=f"auto:{why}")
        h = ActionHandle(entry.id, self.task.intent_version, "look", args, frozenset({"sense"}), source="harness")
        self.actions[entry.id] = h
        h.task = asyncio.current_task()
        async with self._sense_lock:
            entry.status, entry.t_start = "running", self.clock.now()
            out = await run_goal(self.robot, self.clock, "look", args, SENSE_TIMEOUT)
        self._finish(h, entry, out)

    # ------------------------------------------------------------------
    # Thinking and dispatch
    # ------------------------------------------------------------------
    def _wake(self, why: str) -> None:
        self._wake_evt.set()

    async def _think_loop(self) -> None:
        while True:
            await self._wake_evt.wait()
            self._wake_evt.clear()
            if self._errors:
                raise self._errors[0]
            self._thinking = True
            try:
                await self._think()
            finally:
                self._thinking = False

    async def _think(self) -> None:
        for _ in range(MAX_DECISIONS_PER_WAKE):
            await self._step_gate()                      # step mode; before v and epoch are read
            v = self.task.intent_version
            epoch = self.task.control_epoch
            call = await self._ask()
            if call is None:
                return
            # Stale only if the user changed something (correction, stop, resume,
            # constraint). Sensor changes don't invalidate a decision: _check below
            # validates it against the belief as it is now.
            if self._stale(v) or epoch != self.task.control_epoch:
                self.tracer.log("stale_decision", tool=call.tool, args=call.args, made_for=v,
                                now=self.task.intent_version, made_in_epoch=epoch,
                                epoch=self.task.control_epoch)
                continue
            if call.tool == "wait":
                return
            if call.tool == "say" and not self.task.utterances and self.persona.goal is None:
                # nobody has said anything and no own goal asks it to speak: chatter
                self.tracer.log("unprompted_say_dropped", text=call.args.get("text"))
                return
            if call.tool == "say" and self._repeats_last_line(str(call.args.get("text", ""))):
                # Saying the same line again with nothing new heard: treat it as "wait".
                self.tracer.log("repeat_say_dropped", text=call.args.get("text"))
                return
            goal = self.persona.goal
            if call.tool == "say" and goal is not None and (goal.said or "say" not in goal.tools):
                # an own goal gets one short line at most; the rest is narration
                self.tracer.log("persona_quiet", text=call.args.get("text"))
                self._note = "you're on your own goal: act (navigate, look) or wait; don't narrate"
                continue
            if call.tool == "say":
                ok, why = self._check(call)
                if ok:
                    self._say(call, v, epoch)
                    if goal is not None:
                        goal.said = True
                else:
                    self._reject(call, why, v)
                continue
            if call.tool == "recall":
                ok, why = self._check(call)
                if ok:
                    self._recall(call, v)
                else:
                    self._reject(call, why, v)
                continue
            if self._reconciling():
                # Only speech goes out until the cancelled actions have finished and the
                # robot has looked. Then ask again with the new belief.
                self.tracer.log("held_for_reconcile", tool=call.tool, args=call.args)
                await asyncio.shield(self._reconcile_task)
                continue
            if self._sense_lock.locked():
                async with self._sense_lock:
                    pass
                continue
            ok, why = self._check(call)
            if not ok:
                self._reject(call, why, v)
                if self._rejects >= 3:
                    self.tracer.log("rejection_limit")
                    return
                continue
            self._rejects = 0
            self._note = None
            if call.tool in SENSE:
                await self._run_sense(call, v)
                continue
            self._start_body(call, v)
            return                                         # asked again when it finishes
        self.tracer.log("decision_limit")

    def _repeats_last_line(self, text: str) -> bool:
        said = [e for e in self.history if e.tool == "say" and e.status not in ("DROPPED", "REJECTED")]
        if not said or not text.strip():
            return False
        last = said[-1]
        heard_since = any(u.t_end >= last.t_start for u in self.task.utterances)
        said = re.sub(r"[^a-z0-9 ]", "", last.args.get("text", "").lower())
        new = re.sub(r"[^a-z0-9 ]", "", text.lower())
        return (not heard_since and self.clock.now() - last.t_start < 20.0
                and difflib.SequenceMatcher(None, said, new).ratio() >= 0.8)   # "Here's the kettle!" ~ "Here's your kettle!"

    def _stale(self, version: int) -> bool:
        return self.task.intent_version != version

    async def _ask(self) -> ToolCall | None:
        ctx = self._ctx()
        t0 = self.clock.now()
        try:
            call = await self.clock.wait_for(self.brain.next_action(ctx), BRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            self.tracer.log("brain_timeout")
            self._note = "your previous decision timed out"
            call = None
        except Exception as e:
            self.tracer.log("brain_error", error=repr(e))
            call = None
        if call is not None and not isinstance(call, ToolCall):
            self.tracer.log("brain_malformed", got=repr(call))
            self._note = "your previous reply was not a tool call"
            call = None
        if call is None:
            self._brain_errors += 1
            if self._brain_errors <= 3:
                self._spawn(self._wake_later(2.0, "retry after brain error"))
            return None
        self._brain_errors = 0
        self.tracer.log("decision", tool=call.tool, args=call.args, tag=call.tag, reason=call.reason,
                        version=ctx.intent_version, epoch=ctx.control_epoch,
                        revision=ctx.state_revision, latency=round(self.clock.now() - t0, 3))
        return call

    async def _wake_later(self, seconds: float, why: str) -> None:
        await self.clock.sleep(seconds)
        self._wake(why)

    def _ctx(self) -> BrainInput:
        current = self.task.current_directive
        return BrainInput(
            now=round(self.clock.now(), 2), map=self.map, utterances=list(self.task.utterances),
            kinds=dict(self.task.kinds), intent_version=self.task.intent_version,
            belief=self.belief.to_dict(), history=list(self.history),
            active=[e for e in self.history if e.status in ("queued", "running")],
            paused=self.task.paused, note=self._note,
            robot_state=dict(self.state.robot), perception=dict(self.state.perception),
            task_state=self.state.task_dict(),
            directive=current.to_dict() if hasattr(current, "to_dict") else current,
            recent_events=self.state.recent(), state_revision=self.state.revision,
            control_epoch=self.task.control_epoch,
            own_goal=self.persona.goal.to_dict() if self.persona.goal is not None else None,
            notes=self.memory.notes(), guidance=self._guidance(), observations=self._noticed_for_prompt())

    # ------------------------------------------------------------------
    # Rules (rule 3)
    # ------------------------------------------------------------------
    def _check(self, call: ToolCall) -> tuple[bool, str]:
        tool, a = call.tool, dict(call.args or {})
        if tool not in TOOLS:
            return False, f"unknown tool {tool!r}; use one of: {', '.join(TOOLS)}"
        if tool == "say":
            return (True, "") if str(a.get("text", "")).strip() else (False, "say needs some text")
        if tool == "recall":
            return (True, "") if str(a.get("query", "")).strip() else (False, "recall needs a query")
        goal = self.persona.goal
        if goal is not None and tool not in ("say", "recall") and tool not in goal.tools:
            return False, (f"your own goal ({goal.drive}) allows only {', '.join(goal.tools)}; "
                           f"never move objects unless the user asks")
        if tool in BODY and self.persona.goal is None and not self.task.utterances:
            return False, "nobody has asked for anything and you have no own goal: call wait"
        if tool in BODY and self.task.paused:
            return False, "the user said stop: say something if needed and wait until they say to continue"
        if tool in BODY:
            busy = [h.skill for h in self.actions.values() if h.resources & {"base", "arm:left", "arm:right"}]
            if busy:
                return False, f"already running {busy[0]}; wait for it to finish"
        if tool == "navigate":
            to = a.get("to")
            if to not in self.map["keypoints"]:
                return False, f"unknown keypoint {to!r}; use one of: {', '.join(self.map['keypoints'])}"
            blocked = [e for e in self.history if e.tool == "navigate" and e.args.get("to") == to
                       and e.created_for == self.task.intent_version and (e.data or {}).get("reason") in ("blocked", "no_path")]
            if len(blocked) >= 2:
                return False, f"{to} couldn't be reached twice; tell the user instead"
            return True, ""
        if tool in ("reachability", "pick", "place"):
            oid = a.get("object")
            if oid not in self.belief.objects:
                return False, f"unknown object {oid!r}; use an id from belief (look first)"
        if tool == "reachability":
            if self.belief.robot_at.value is None:
                return False, "the robot is between keypoints; navigate to one first"
            return True, ""
        if tool in ("pick", "place") and a.get("arm") not in ARMS:
            return False, "arm must be 'left' or 'right'"
        if tool == "pick":
            oid, arm = a["object"], a["arm"]
            placed = [e for e in self.history if e.tool == "place" and e.args.get("object") == oid
                      and e.status == "SUCCEEDED"]
            heard = any(u.t_end >= placed[-1].t_start for u in self.task.utterances) if placed else True
            if placed and not heard and oid not in self._goal_missed:        # a missed goal may be fixed
                return False, (f"you just put {oid} down and nobody has asked for anything since; "
                               f"say it's done or wait")
            if not self.belief.hand_known_empty(arm):
                return False, f"the {arm} hand is not known to be empty; look first or put down what it holds"
            reach = self._fresh_reach(oid)
            if reach is None:
                return False, f"pick needs a successful reachability check for {oid} from this spot first"
            if not reach.get("reachable"):
                return False, (f"reachability says {oid} can't be reached from here ({reach.get('reason')}); "
                               f"reposition or tell the user")
            if reach.get("arm") != arm:
                return False, f"reachability says to use the {reach.get('arm')} arm"
            failed = [e for e in self.history if e.tool == "pick" and e.args.get("object") == oid
                      and e.created_for == self.task.intent_version
                      and e.status in ("FAILED", "ABORTED", "TIMEOUT")]
            if len(failed) >= 2:
                return False, f"two grasps of {oid} have failed; tell the user instead of retrying"
            return True, ""
        if tool == "place":
            oid, arm = a["object"], a["arm"]
            if not self.belief.hand_holds(arm, oid):
                return False, f"the {arm} hand is not verified to hold {oid}; look first"
            if self._surface_here() is None:
                return False, "there is no surface to put it on here; navigate to one first"
            if a.get("goal") and layout.parse(str(a["goal"]), self._layout(), self.map, set(self.belief.objects)) is None:
                return False, (f"goal {a['goal']!r} not understood; write it as one of: {', '.join(layout.GOAL_FORMS)}, "
                               f"with ids from MAP, LAYOUT or BELIEF (or leave goal out)")
            return True, ""
        return True, ""

    def _fresh_reach(self, oid: str) -> dict[str, Any] | None:
        now = self.clock.now()
        last_pick = max([e.t_start for e in self.history if e.tool == "pick" and e.args.get("object") == oid
                         and e.status != "REJECTED"] or [-1e9])
        for e in reversed(self.history):
            if e.tool == "reachability" and e.args.get("object") == oid and e.status == "SUCCEEDED":
                if e.t_start >= self._last_motion_t and e.t_start > last_pick and now - e.t_start <= REACH_FRESH_S:
                    return e.data
                return None
        return None

    def _surface_here(self) -> str | None:
        at = self.belief.robot_at.value
        for s, info in self.map["surfaces"].items():
            if at in info["keypoints"] and info["height_m"] <= self.map["max_reach_height_m"]:
                return s
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _entry(self, tool: str, args: dict[str, Any], created_for: int | None = None,
               tag: str | None = None, source: str = "brain") -> HistoryEntry:
        e = HistoryEntry(id=next(self._ids), tool=tool, args=dict(args),
                         created_for=self.task.intent_version if created_for is None else created_for,
                         t_start=round(self.clock.now(), 3), status="queued",
                         control_epoch=self.task.control_epoch, tag=tag, source=source)
        self.history.append(e)
        return e

    def _say(self, call: ToolCall, v: int, epoch: int | None = None) -> None:
        text = str(call.args.get("text", "")).strip()
        e = self._entry("say", {"text": text}, v, call.tag)
        speech_epoch = self.task.control_epoch if epoch is None else epoch
        e.control_epoch = speech_epoch
        self.speech.enqueue(SpeechItem(e, text, v, speech_epoch))
        self.tracer.log("say_queued", text=text, version=v, epoch=speech_epoch)

    def _say_immediate(self, text: str) -> None:
        """Non-blocking safety acknowledgement, queued only after halt."""
        e = self._entry("say", {"text": text}, self.task.intent_version,
                        tag="runtime:safety-ack", source="harness")
        self.speech.enqueue_priority(
            SpeechItem(e, text, self.task.intent_version, self.task.control_epoch))
        self.tracer.log("safety_ack_queued", text=text, epoch=self.task.control_epoch)
        self.state.record("safety_ack", self.clock.now(), priority=2, text=text,
                          control_epoch=self.task.control_epoch)

    def _reject(self, call: ToolCall, why: str, v: int) -> None:
        e = self._entry(call.tool, dict(call.args or {}), v, call.tag)
        e.status, e.data, e.t_end = "REJECTED", {"reason": why}, round(self.clock.now(), 3)
        self._note = f"rejected {call.tool}({call.args}): {why}"
        self._rejects += 1
        self.tracer.log("rejected", tool=call.tool, args=call.args, why=why)

    async def _run_sense(self, call: ToolCall, v: int) -> None:
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._entry(call.tool, call.args, v, call.tag, source=src)
        h = ActionHandle(e.id, v, call.tool, dict(call.args), frozenset({"sense"}), source=src)
        self.actions[e.id] = h
        h.task = self._spawn(self._sense(h, e))
        await asyncio.wait([h.task])

    async def _sense(self, h: ActionHandle, e: HistoryEntry) -> None:
        async with self._sense_lock:
            e.status, e.t_start = "running", round(self.clock.now(), 3)
            if h.cancel_requested:
                out = Outcome("CANCELED", {"reason": "cancelled before start"})
            else:
                out = await run_goal(self.robot, self.clock, h.skill, h.args, SENSE_TIMEOUT,
                                     on_goal=self._binder(h))
        self._finish(h, e, out)

    def _start_body(self, call: ToolCall, v: int) -> None:
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._entry(call.tool, call.args, v, call.tag, source=src)
        e.status = "running"
        res = {"base"} if call.tool == "navigate" else {f"arm:{call.args['arm']}"}
        h = ActionHandle(e.id, v, call.tool, dict(call.args), frozenset(res), source=src)
        self.actions[e.id] = h
        h.task = self._spawn(self._body(h, e))
        self.tracer.log("started", tool=call.tool, args=call.args, version=v)
        self.state.record("behavior_started", self.clock.now(), priority=3,
                          tool=call.tool, args=dict(call.args), version=v,
                          control_epoch=self.task.control_epoch)

    def _binder(self, h: ActionHandle):
        def bind(goal: Any) -> None:
            h.goal = goal
            if h.cancel_requested:
                goal.cancel()
        return bind

    async def _body(self, h: ActionHandle, e: HistoryEntry) -> None:
        a = h.args
        if h.cancel_requested:
            out = Outcome("CANCELED", {"reason": "cancelled before start"})
        elif h.skill == "navigate":
            timeout = navigate_timeout(self.map, self.belief.robot_at.value, a["to"], self.belief.blocked)
            out = await run_goal(self.robot, self.clock, "navigate", a, timeout, on_goal=self._binder(h))
        else:
            # The THOR robot runs the grasp itself, chunk by chunk, and honours
            # cancel only between chunks, so pick and place are plain goals here.
            timeout = PICK_TIMEOUT if h.skill == "pick" else PLACE_TIMEOUT
            here = self._surface_here()
            out = await run_goal(self.robot, self.clock, h.skill, {k: v for k, v in a.items() if k != "goal"},
                                 timeout, on_goal=self._binder(h))
            if h.skill == "pick" and out.status == "SUCCEEDED" and here:
                self._picked_from[a["object"]] = here
        self._finish(h, e, out)
        if h.skill in ("pick", "place") and not h.cancel_requested:
            # never trust a success flag: a glance straight ahead, and the full look if it didn't show the object
            await self._observe(f"verify {h.skill}", glance=True)
            ob = self.belief.objects.get(a["object"])
            if h.skill == "place" and out.status == "SUCCEEDED" and not (
                    ob is not None and ob.where.verified and ob.where.value == here):
                await self._observe("verify place: not in the glance")
            if h.skill == "place" and a.get("goal") and out.status == "SUCCEEDED":
                self._check_goal(a["object"], str(a["goal"]))
        self._wake(f"{h.skill} finished")

    def _layout(self) -> list[layout.Line]:
        marks = {k: {"label": lm.label, "near": lm.near.value, "pos": lm.pos} for k, lm in self.belief.landmarks.items()}
        return layout.build(self.map, marks)

    def _check_goal(self, oid: str, text: str) -> None:
        """Where did it really end up, and is that what the request asked for? Checked on the layout
        (after the verifying glance), not taken from the planner; a miss goes back to it as a NOTE."""
        lines = self._layout()
        goal = layout.parse(text, lines, self.map, set(self.belief.objects))
        ob = self.belief.objects.get(oid)
        where = str(ob.where.value) if ob is not None else None
        if goal is None:
            return
        res = layout.check(goal, where, lines, self.map, origin=self._picked_from.get(oid))
        self.tracer.log("goal_check", object=oid, goal=text, ok=res.ok, where=where, expected=res.expected, why=res.why)
        (self._goal_missed.add if res.ok is False else self._goal_missed.discard)(oid)
        if res.ok is False:
            self._note = (f"goal check failed: {oid} is on {where}, but the goal was '{text}': {res.why}. "
                          f"Move it there, or tell the user what happened; don't say it's done.")
        elif res.ok is None:
            self._note = (f"goal check: '{text}' can't be checked ({res.why}). Tell the user where {oid} really is "
                          f"({where}) instead of claiming the goal.")

    def _finish(self, h: ActionHandle, e: HistoryEntry, out: Outcome) -> None:
        now = self.clock.now()
        late = h.created_for < self.task.intent_version
        e.status, e.data, e.t_end = out.status, dict(out.data), round(now, 3)
        if late:
            e.data["late"] = True            # rule 1: recorded, but not progress for the new request
        self.actions.pop(h.entry_id, None)
        skill = h.skill
        if skill == "navigate":
            self.belief.apply_navigate(out.status, out.data, now)
            self._last_motion_t = now
        elif skill == "look" and out.ok:
            self.belief.apply_look(out.data, now, self._surface_xy)
        elif skill in ("pick", "place"):
            arm, oid = h.args["arm"], h.args["object"]
            if late or h.cancel_requested or not out.ok:
                hint = oid if (skill == "pick" and out.data.get("holding")) else None
                self.belief.mark_hand_unknown(arm, "late_result" if late else f"{skill}_{out.status.lower()}",
                                              now, hint=hint)
            elif skill == "pick":
                self.belief.claim_pick(arm, oid, now)
            else:
                self.belief.claim_place(arm, oid, self._surface_here(), now)
        self._refresh_observation()
        self._note_places()
        if self.persona.goal is not None and self.persona.is_done(self.belief):
            self._end_own_goal("done")            # before System 2 is asked again,
            if not self.task.paused:              # and hand it the next one straight away
                self._start_own_goal(self.persona.propose(self.belief, self.map, now, quiet_s=1e9))
        brief = {k: v for k, v in out.data.items() if k not in ("surfaces", "views", "landmarks")}
        if skill == "look" and out.ok:
            brief["saw"] = sorted(v["id"] for items in (out.data.get("surfaces") or {}).values() for v in items)
            brief["landmarks"] = [lm["id"] for lm in out.data.get("landmarks") or []]
        self.tracer.log("result", skill=skill, status=out.status, late=late, data=brief, source=e.source)
        self.state.record("behavior_result", now, priority=3, tool=skill,
                          status=out.status, late=late, control_epoch=e.control_epoch, data=brief)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------
    def _spawn(self, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg.add(task)

        def done(t: asyncio.Task) -> None:
            self._bg.discard(t)
            if not t.cancelled() and t.exception() is not None:
                self._errors.append(t.exception())
                self._wake_evt.set()

        task.add_done_callback(done)
        return task
