"""The runtime: event loop, utterance handling, rules, executor, speech and trace.

Five asyncio tasks (plus the robot-event reader):

  listen     takes each utterance from the user. "Stop" halts the robot right
             here, before any model call.
  interpret  classifies utterances in order and applies them. Only a
             correction cancels anything; additions, questions and answers
             leave running actions alone.
  think      asks the brain for the next step, validates it (agent/validate.py:
             SCHEMA -> ENUM -> STATE -> CAPABILITY), and dispatches it through
             ``robot.start(execution)``. Body executions run in the background;
             the brain is asked again when something changes.
  speech     plays queued lines in order (see skills.SpeechQueue).
  observe    10 Hz fused state, narrator, tool-state events, memory.

Rules, enforced here rather than in a prompt:

  1. A late result updates belief, never the plan.
  2. After a cancel, every resource it touched is UNKNOWN until the robot looks.
  3. Tool calls are checked before they run; a rejection says what to do next.
  4. The brain sees belief, never simulator ground truth.
  5. "Stop" skips the brain.
  6. Every skill and brain call has a timeout.
  7. Retries need new information and fit a budget; then tell the user.
  8. One clock everywhere.

Tools (api/tools.py): speak, list_locations, navigate, check_reachability,
manipulate, wait_and_observe, and recall [WL]. ``look`` is internal: an arrival
scan after every keypoint navigate, a verify glance after every manipulate, a
reconcile look after a cancel, and the observation that starts wait_and_observe.
"""

from __future__ import annotations

import asyncio
import dataclasses
import difflib
import os
import re
import time
from typing import Any

from api.execution import Execution, ExecutionManager
from api.results import ToolResult, finish
from api.results import rejected as rejected_result
from api.state_machine import ToolState, derive_state, wait_observe_mode
from api.tools import BODY_TOOLS, NEUTRAL_TOOLS, SchemaContext, ToolCall, resources_of
from api.types import PROFILES
from brains.interface import KINDS, UNKNOWN, BrainInput

from . import layout
from .context import ContextBus
from .episodes import EpisodeLog, load_episodes
from .fused_state import (OBSERVATION_PERIOD_S, Directive, FusedRuntimeState,
                          RobotObservationAdapter)
from .memory import SpatialMemory
from .narrator import Narrator
from .persona import GOAL_TIMEOUT_S, STUCK_S, Persona
from .procedures import ProceduralGraph, step_of
from .recall import Recaller
from .skills import SpeechItem, SpeechQueue, navigate_timeout, run_execution
from .state import ARMS, ActionHandle, BeliefState, TaskState
from .validate import Stage, ValidationContext, Verdict, surface_here, validate

BODY = BODY_TOOLS                         # ("navigate", "manipulate"): the one body resource
SENSE = ("check_reachability",)
INSTANT = ("speak", "list_locations", "recall")
CLASSIFY_TIMEOUT = 8.0
BRAIN_TIMEOUT = 20.0
ECHO_S = 15.0             # an observation of what belief confirmed this recently, at the same spot, is an echo
SENSE_TIMEOUT = 12.0      # a check_reachability, or a two-row scan (about 5 s on SONIC)
GLANCE_TIMEOUT = 4.0
INSTANT_TIMEOUT = 5.0
PICK_TIMEOUT = 25.0       # fallbacks when the robot can't say (RobotBridge.timeout_s)
PLACE_TIMEOUT = 20.0
REACH_FRESH_S = 30.0
MAX_DECISIONS_PER_WAKE = 40
WAIT_POLL_S = 0.1
QUIET_NOCHANGE_LIMIT = 2  # two no-change waits in a row: sleep until an external wake (rule 7)
RESULTS_KEPT = 40

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
        self.profile = getattr(robot, "profile", None) or PROFILES["lite"]
        self.include_look = os.environ.get("WL_LOOK_TOOL") == "1"
        reg_fn = getattr(robot, "registry", None)
        self.registry = reg_fn() if callable(reg_fn) else None
        skill_types = (self.registry.loaded_object_types() if self.registry is not None
                       else list(self.map.get("skill_types") or []))
        # Enums and numeric slots are fixed here, once per session (Invariant 9).
        self.tools_ctx = SchemaContext.from_map(self.map, skill_types, self.profile.slots(), self.include_look)
        self.belief = BeliefState()
        self.belief.load_memory(robot.memory())
        self.memory = SpatialMemory(self.map.get("scene"))      # what earlier sessions saw
        self.belief.load_memory(self.memory.entries(now))
        self._memory_rev = self.belief.rev
        self.persona = Persona()
        self._surface_xy = {k: tuple(v["xy"]) for k, v in self.map["surfaces"].items() if v.get("xy")}
        self._surface_h = {k: float(v["height_m"]) for k, v in self.map["surfaces"].items()
                           if v.get("height_m") is not None}
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
        self.executions = ExecutionManager(clock)
        self.history: list[Execution] = self.executions.all      # every execution, oldest first
        self.actions: dict[str, ActionHandle] = {}
        self.tracer = ContextBus(clock, fence=lambda: (self.task.intent_version, self.task.control_epoch))
        scene = self.map.get("scene")
        self.episodes = EpisodeLog(scene)                     # what happens, written as it happens
        self.tracer.sinks.append(self.episodes.write)
        deliver_to = ((self.map.get("people") or {}).get("user") or {}).get("deliver_to_surface")
        self.tracer.log("session_start", scene=scene, wall=round(time.time(), 1), deliver_to=deliver_to,
                        profile=self.profile.name, skill_types=list(self.tools_ctx.skill_types))
        self.recaller = Recaller(scene)
        self.procedures = ProceduralGraph.load()              # what usually works, learned from all episodes
        self.procedures.learn(load_episodes())
        self._task_row: int | None = None                    # where the current request's trace starts
        self._places = {oid: ob.where.value for oid, ob in self.belief.objects.items()}
        self._deliver_to = deliver_to
        self.narrator = Narrator(self.map, self.belief, deliver_to)     # progress lines for the chat
        self.tracer.sinks.append(self.narrator.row)
        self._delivered: set[str] = set()                    # place executions already logged as deliveries
        self.speech = SpeechQueue(robot, clock)
        self.speech.on_result = self._speech_result
        self._wake_evt = asyncio.Event()
        self._wake_why = ""
        self._utt_q: asyncio.Queue = asyncio.Queue()
        self._sense_lock = asyncio.Lock()
        self._note: str | None = None
        self._picked_from: dict[str, str] = {}      # object -> the surface it was last picked up from
        self._goal_missed: set[str] = set()          # objects whose last place failed its goal check
        self._canceled: list[ActionHandle] = []
        self._reconcile_task: asyncio.Task | None = None
        self._reconcile_mode = "auto"
        self._bg: set[asyncio.Task] = set()
        self._errors: list[BaseException] = []
        self._thinking = False
        self._interpreting = False
        self._rejects = 0
        self._brain_errors = 0
        self._last_motion_t = now                   # base motion (navigate, reposition, scan, base shift)
        self._last_scan: dict[str, float] = {}      # keypoint -> when the last scan there ended
        self._nochange = 0                          # consecutive no-change waits (quiet rule)
        self._results: list[ToolResult] = []        # the latest envelopes, for BrainInput.tool_results
        self._tool_state = ToolState.IDLE
        self._body_mode: str | None = None
        self._emergency_ids = iter(range(1, 1 << 30))
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
                tg.create_task(self._robot_events_loop())
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

    def tool_state(self) -> str:
        """IDLE / NAVIGATING / MANIPULATING / OBSERVING / WAITING / CANCELLING / RECONCILING /
        STOPPED / FAULT, derived from active executions (api/state_machine.py)."""
        return derive_state(self.executions.active(), paused=self.task.paused,
                            reconciling=self._reconciling(), body_mode=self._body_mode_now()).value

    def executions_snapshot(self, n: int = 30) -> list[dict[str, Any]]:
        return [self._history_dict(e) for e in self.history[-n:]]

    def runtime_snapshot(self) -> dict[str, Any]:
        active = [self._history_dict(e) for e in self.history if not e.finished]
        recent = [self._history_dict(e) for e in self.history[-12:]]
        snap = self.state.snapshot(active=active, recent_actions=recent)
        snap["tool_state"] = self.tool_state()
        snap["executions"] = recent
        snap["context_events"] = self.tracer.recent_events(30)
        return snap

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

    def operator_estop(self, reason: str = "operator") -> dict[str, Any]:
        """The UI's "kill controller (hardware e-stop)" button: never the chat "stop"."""
        fn = getattr(self.robot, "estop", None)
        receipt = fn(reason) if callable(fn) else {"accepted": False}
        self.tracer.log("safety_event", kind="estop", reason=reason, receipt=receipt)
        self.state.record("safety_event", self.clock.now(), priority=0, kind="estop", reason=reason)
        self.emergency_stop("estop", source="estop")
        return receipt

    def _refresh_observation(self) -> set[str]:
        frame = self.observations.sample(self.clock.now())
        return self.state.ingest(frame)

    def _body_mode_now(self) -> str | None:
        body = (self.state.robot or {}).get("body") or {}
        return body.get("mode") or self._body_mode

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
            ts = ToolState(self.tool_state())
            if ts != self._tool_state:
                self.tracer.log("tool_state", state=ts.value, was=self._tool_state.value)
                self._tool_state = ts
            # Only a change to what the robot believes (an object seen somewhere new)
            # is worth a model call; poses and distances change on every frame.
            # Action completion already wakes the planner, so only wake when idle.
            if (self.belief.rev != rev and self.task.utterances and not self.actions
                    and not self._thinking):
                self._wake("perception changed belief")
            if self.belief.rev != self._memory_rev:     # an observation, a result or the camera taught us something
                self._memory_rev = self.belief.rev
                self._note_places()
                if self.memory.update(self.belief, self.clock.now()):
                    self.tracer.log("memory_saved", objects=len(self.memory.items))
            await self.clock.sleep(OBSERVATION_PERIOD_S)

    async def _robot_events_loop(self) -> None:
        """Robot-side events (RobotBridge.events()): safety, capability, body mode, stale results."""
        fn = getattr(self.robot, "events", None)
        q = fn() if callable(fn) else None
        if q is None or not hasattr(q, "get"):
            return
        while True:
            ev = await q.get()
            if not isinstance(ev, dict):
                continue
            self._on_robot_event(ev)

    def _on_robot_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        now = self.clock.now()
        fields = {k: v for k, v in ev.items() if k != "type"}
        if kind == "safety_event":
            self.tracer.log("safety_event", **fields)
            self.state.record("safety_event", now, priority=0, **fields)
            what = str(ev.get("kind") or "")
            if what in ("fell", "deploy_lost", "estop", "fault") and not self.task.paused:
                self._stop_now_id(f"safety-{what}", reason=f"safety:{what}",
                                  ack=("I've lost my balance; I'm stopping until I'm steady." if what == "fell"
                                       else "I've had to stop: my controller needs a restart."))
            self._wake(f"safety {what}")
        elif kind == "capability_changed":
            self.tracer.log("capability_changed", **fields)
            self.state.record("capability_changed", now, priority=2, **fields)
            self._note = f"capability changed: {ev.get('detail') or ev.get('capability') or ''}".strip()
            self._wake("capability changed")
        elif kind == "body_mode":
            mode = ev.get("mode")
            if mode != self._body_mode:
                self._body_mode = mode
                self.tracer.log("body_mode", **fields)
                self.state.record("body_mode", now, priority=3, **fields)
        elif kind == "stale_result":
            self.tracer.log("stale_result", **fields)

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
        self._nochange = 0
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
                    h.cancel("own goal dropped")
                    if h.skill != "wait_and_observe":
                        self._canceled.append(h)
            self._reconcile_mode = "auto"
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
        The camera often sees it land before the place finishes, so this runs on both."""
        for e in self.history:
            if (e.tool_name != "manipulate" or e.action != "place" or e.status != "succeeded"
                    or e.execution_id in self._delivered):
                continue
            ob = self.belief.objects.get((e.data or {}).get("object_id") or e.args.get("object_id"))
            if ob is not None and ob.where.verified and ob.where.value == self._deliver_to:
                self._delivered.add(e.execution_id)
                self.tracer.log("delivered", object=ob.id, surface=ob.where.value, execution_id=e.execution_id)

    def _recall(self, call: ToolCall, verdict: Verdict, v: int) -> None:
        query = str(verdict.args.get("query", "")).strip()
        e = self._new_exec("recall", {"query": query}, v, call.tag, status="running")
        answer = self.recaller.answer(query, self.belief, self.memory, self.map, self.clock.now(), self.tracer.rows)
        res = finish(e, "succeeded", {"answer": answer}, t_end=round(self.clock.now(), 3),
                     observation_id=self._obs_id())
        self._store(e, res)
        self.tracer.log("recall", query=query, answer=answer, execution_id=e.execution_id)

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
        says = [e for e in self.history if e.tool_name == "speak" and e.status not in ("dropped", "rejected")]
        return {
            "at": at, "between": list(b.between) if b.between else None,
            "room": (self.map.get("rooms", {}).get(room) or {}).get("label", room) if room else None,
            "moving": bool((self.state.robot or {}).get("moving")),
            "holding": {arm: b.holding[arm].value for arm in ARMS if b.holding[arm].value},
            "in_view": sorted(oid for oid, ob in b.objects.items() if ob.visible),
            "goal": self.task.goal, "stopped": self.task.paused,
            "running": [e.tool_name for e in self.history if e.status in ("queued", "running", "cancelling")
                        and e.tool_name != "speak"],
            "robot_last_said": says[-1].args.get("text") if says else None,
            "own_goal": self.persona.goal.text if self.persona.goal is not None else None,
            "tool_state": self.tool_state(),
        }

    def add_observations(self, items: list[dict[str, Any]], source: str = "system1") -> int:
        """System 1's observations enter as unverified hints: this session's list (shown to
        the planner as NOTICED) and spatial memory's observations, never as object facts."""
        now = self.clock.now()
        # Belief keeps the last keypoint until a walk ends; while moving, "here" isn't known.
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
    def _history_dict(entry: Execution) -> dict[str, Any]:
        d = entry.to_dict()
        d.update({"id": entry.execution_id, "created_for": entry.generation,
                  "summary": entry.result.summary if entry.result is not None else None})
        return d

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
                self._nochange = 0                          # new information
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
            replaces_task=raw.get("replaces_task"),
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
            self._resume()
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

    def _resume(self) -> None:
        """Resume: epoch+1 (also clears the body's halt latch); speech from before is fenced."""
        self.task.control_epoch += 1
        self.task.paused = False
        fn = getattr(self.robot, "resume", None)
        if callable(fn):
            try:
                fn(self.task.control_epoch)
            except Exception as e:                        # never let a resume crash the runtime
                self.tracer.log("resume_error", error=repr(e))
        dropped = self.speech.drop_before_epoch(self.task.control_epoch)
        self.tracer.log("resume", epoch=self.task.control_epoch, dropped=dropped)

    def _stop_now(self, utt: Any, reason: str) -> None:
        self._stop_now_id(utt.id, reason)

    def _stop_now_id(self, event_id: str, reason: str, ack: str | None = None) -> None:
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
            if h.resources & {"body"} and not h.done and not h.cancel_requested:
                h.cancel("halted")
                self._canceled.append(h)
                canceled.append(h.skill)
            elif h.skill == "wait_and_observe" and not h.cancel_requested:
                h.cancel("stop")
        self._refresh_observation()
        confirmed = bool(isinstance(receipt, dict) and receipt.get("stopped"))
        self.tracer.log("stop", id=event_id, reason=reason, canceled=canceled, cut=cut,
                        epoch=self.task.control_epoch, physical_first=True, receipt=receipt)
        self.state.record("emergency_stop", self.clock.now(), priority=0, id=event_id,
                          reason=reason, canceled=canceled, confirmed=confirmed,
                          control_epoch=self.task.control_epoch)
        self._say_immediate(ack or ("I've stopped." if confirmed else "Stopping now."))
        self._reconcile_mode = "glance"               # after a stop the robot only talks: no waist twist
        self._start_reconcile()

    def _correct(self, utt: Any) -> None:
        self.task.intent_version += 1
        self.task.control_epoch += 1
        self.task.goal = utt.text
        v = self.task.intent_version
        dropped = self.speech.drop_older_than(v)
        canceled = []
        now = self.clock.now()
        for h in list(self.actions.values()):
            if h.created_for < v and not h.done and not h.cancel_requested:
                h.cancel("correction")
                if h.skill == "wait_and_observe":
                    continue
                self._canceled.append(h)
                canceled.append(h.skill)
                if h.skill == "manipulate" and h.args.get("arm") in ARMS:
                    self.belief.mark_hand_unknown(h.args["arm"], "cancel", now)
        self.tracer.log("correction", id=utt.id, version=v, epoch=self.task.control_epoch,
                        dropped=dropped, canceled=canceled)
        self.state.record("task_corrected", self.clock.now(), priority=1, id=utt.id,
                          intent_version=v, control_epoch=self.task.control_epoch,
                          canceled=canceled)
        self._reconcile_mode = "auto"
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
                if h.skill == "manipulate" and h.args.get("arm") in ARMS:
                    arms.add(h.args["arm"])
            pending = [h.task for h in self._canceled if h.task is not None and not h.task.done()]
            if not pending:
                break
            await asyncio.wait(pending)       # each execution has its own timeout
        self._canceled.clear()
        now = self.clock.now()
        for arm in arms:                      # rule 2
            hint = self.belief.hints.get(arm)
            self.belief.mark_hand_unknown(arm, "cancel", now, hint=hint)
        if arms:
            await self._observe("after cancel", mode="glance" if self.task.paused else self._reconcile_mode)
        self._refresh_observation()
        self.tracer.log("reconcile_done", belief=self.belief.to_dict()["hands"])
        self.state.record("reconciled", self.clock.now(), priority=1,
                          belief=self.belief.to_dict()["hands"],
                          control_epoch=self.task.control_epoch)
        self._wake("reconciled")

    # ------------------------------------------------------------------
    # Internal observations (arrival scan, verify, reconcile, wait_and_observe)
    # ------------------------------------------------------------------
    def _choose_observe_mode(self) -> str:
        at = self.belief.robot_at.value
        last = self._last_scan.get(at) if at else None
        return wait_observe_mode(
            body_lease_held=any("body" in e.resources for e in self.executions.active()),
            stationary=not bool((self.state.robot or {}).get("moving")) and not self.task.paused,
            at_keypoint=at is not None,
            last_scan_age_s=None if last is None else self.clock.now() - last)

    async def _observe(self, why: str, mode: str = "glance", source: str = "harness") -> ToolResult | None:
        """A harness-initiated observation, recorded like any other execution. A glance checks the
        view the camera already has; a scan turns the waist (two rows) and needs no body lease."""
        if mode == "auto":
            mode = self._choose_observe_mode()
        if mode == "scan" and any("body" in e.resources for e in self.executions.active()):
            mode = "glance"                                 # never twist the waist under someone's lease
        tool = "observe"
        args = {"mode": mode, "why": why}
        e = self._new_exec(tool, args, source=source, tag=f"auto:{why}", action=mode)
        h = ActionHandle(e.execution_id, self.task.intent_version, tool, args, e.resources, source=source)
        self.actions[e.execution_id] = h
        h.task = self._spawn(self._observe_run(h, e))
        await asyncio.wait([h.task])
        return e.result

    async def _observe_run(self, h: ActionHandle, e: Execution) -> None:
        async with self._sense_lock:
            if h.cancel_requested:
                res = finish(e, "cancelled", {"reason": "cancelled before start", "mode": e.action},
                             t_end=round(self.clock.now(), 3), observation_id=self._obs_id())
            else:
                e.status, e.t_start = "running", round(self.clock.now(), 3)
                timeout = SENSE_TIMEOUT if e.action == "scan" else GLANCE_TIMEOUT
                res = await run_execution(self.robot, self.clock, e, timeout, on_handle=self._binder(h),
                                          profile=self.profile)
        self._finish(h, e, res)
        if e.action == "scan" and res.ok:
            at = res.data.get("at") or self.belief.robot_at.value
            now = self.clock.now()
            if at:
                self._last_scan[at] = now
            self._last_motion_t = now                   # a scan moves the waist: reachability goes stale

    # ------------------------------------------------------------------
    # Thinking and dispatch
    # ------------------------------------------------------------------
    def _wake(self, why: str) -> None:
        self._wake_why = why
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
            # constraint). Sensor changes don't invalidate a decision: validation
            # checks it against the belief as it is now.
            if self._stale(v) or epoch != self.task.control_epoch:
                self.tracer.log("stale_decision", tool=call.tool, args=call.args, made_for=v,
                                now=self.task.intent_version, made_in_epoch=epoch,
                                epoch=self.task.control_epoch)
                continue
            for extra in call.extra:                     # doc 21: one action per turn
                self._reject(extra, Verdict(False, Stage.SCHEMA, "two_calls", "two action tools in one turn"), v)
            tool = call.tool
            if tool == "wait_and_observe":
                verdict = self._check(call)
                if not verdict.ok:
                    self._reject(call, verdict, v)
                    continue
                if await self._wait_and_observe(call, verdict, v, epoch):
                    continue
                return
            if tool == "speak" and not self.task.utterances and self.persona.goal is None:
                # nobody has said anything and no own goal asks it to speak: chatter
                self.tracer.log("unprompted_say_dropped", text=call.args.get("text"))
                return
            if tool == "speak" and self._repeats_last_line(str(call.args.get("text", ""))):
                # Saying the same line again with nothing new heard: treat it as a wait.
                self.tracer.log("repeat_say_dropped", text=call.args.get("text"))
                return
            goal = self.persona.goal
            if tool == "speak" and goal is not None and (goal.said or "speak" not in goal.tools):
                # an own goal gets one short line at most; the rest is narration
                self.tracer.log("persona_quiet", text=call.args.get("text"))
                self._note = "you're on your own goal: act (navigate, wait_and_observe) or wait; don't narrate"
                continue
            if tool in INSTANT:
                verdict = self._check(call)
                if not verdict.ok:
                    self._reject(call, verdict, v)
                    continue
                if tool == "speak":
                    self._speak(call, verdict, v, epoch)
                    self._rejects = 0                     # an accepted new line resets the budget
                    if goal is not None:
                        goal.said = True
                elif tool == "recall":
                    self._recall(call, verdict, v)
                else:
                    await self._run_instant(call, verdict, v)
                continue
            if self._reconciling():
                # Only speech goes out until the cancelled actions have finished and the
                # robot has looked. Then ask again with the new belief. (D9: a deferral.)
                self.tracer.log("held_for_reconcile", tool=call.tool, args=call.args)
                await asyncio.shield(self._reconcile_task)
                continue
            if self._sense_lock.locked():                 # D10: an observation is running
                async with self._sense_lock:
                    pass
                continue
            verdict = self._check(call)
            if not verdict.ok:
                self._reject(call, verdict, v)
                if self._rejects >= 3:
                    self.tracer.log("rejection_limit")
                    return
                continue
            self._rejects = 0
            self._note = None
            self._nochange = 0
            if tool in SENSE or tool == "look":
                await self._run_sense(call, verdict, v)
                continue
            self._start_body(call, verdict, v)
            return                                         # asked again when it finishes
        self.tracer.log("decision_limit")

    def _repeats_last_line(self, text: str) -> bool:
        said = [e for e in self.history if e.tool_name == "speak" and e.status not in ("dropped", "rejected")]
        if not said or not text.strip():
            return False
        last = said[-1]
        heard_since = any(u.t_end >= last.t_start for u in self.task.utterances)
        said_text = re.sub(r"[^a-z0-9 ]", "", last.args.get("text", "").lower())
        new = re.sub(r"[^a-z0-9 ]", "", text.lower())
        return (not heard_since and self.clock.now() - last.t_start < 20.0
                and difflib.SequenceMatcher(None, said_text, new).ratio() >= 0.8)   # "Here's the kettle!" ~ "Here's your kettle!"

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
                        revision=ctx.state_revision, latency=round(self.clock.now() - t0, 3),
                        extra=[c.tool for c in call.extra] or None)
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
            active=[e for e in self.history if not e.finished],
            paused=self.task.paused, note=self._note,
            robot_state=dict(self.state.robot), perception=dict(self.state.perception),
            task_state=self.state.task_dict(),
            directive=current.to_dict() if hasattr(current, "to_dict") else current,
            recent_events=self.state.recent(), state_revision=self.state.revision,
            control_epoch=self.task.control_epoch,
            own_goal=self.persona.goal.to_dict() if self.persona.goal is not None else None,
            notes=self.memory.notes(), guidance=self._guidance(), observations=self._noticed_for_prompt(),
            tool_results=list(self._results[-12:]), tools_ctx=self.tools_ctx, tool_state=self.tool_state(),
            profile=self.profile.slots())

    # ------------------------------------------------------------------
    # Rules (rule 3): agent/validate.py
    # ------------------------------------------------------------------
    def _vctx(self) -> ValidationContext:
        caps_fn = getattr(self.robot, "capabilities", None)
        try:
            caps = caps_fn() if callable(caps_fn) else {}
        except Exception:
            caps = {}
        lines = None

        def parse(text: str) -> Any:
            nonlocal lines
            if lines is None:
                lines = self._layout()
            return layout.parse(text, lines, self.map, set(self.belief.objects))
        return ValidationContext(
            belief=self.belief, history=self.history, map=self.map, task=self.task, now=self.clock.now(),
            own_goal=self.persona.goal, registry=self.registry, skill_types=self.tools_ctx.skill_types,
            capabilities=caps, last_motion_t=self._last_motion_t,
            tool_state=ToolState(self.tool_state()), goal_missed=self._goal_missed,
            reach_fresh_s=getattr(self.profile, "reach_fresh_s", REACH_FRESH_S),
            include_look=self.include_look, layout_parse=parse)

    def _check(self, call: ToolCall) -> Verdict:
        return validate(call, self._vctx())

    def _surface_here(self) -> str | None:
        return surface_here(self.map, self.belief.robot_at.value)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _new_exec(self, tool: str, args: dict[str, Any], generation: int | None = None, tag: str | None = None,
                  source: str = "brain", action: str | None = None, status: str = "queued",
                  epoch: int | None = None) -> Execution:
        return self.executions.create(
            tool, args, generation=self.task.intent_version if generation is None else generation,
            control_epoch=self.task.control_epoch if epoch is None else epoch, source=source, tag=tag,
            action=action, status=status, t=round(self.clock.now(), 3))  # type: ignore[arg-type]

    def _obs_id(self) -> str | None:
        fn = getattr(self.robot, "observation_id", None)
        try:
            return fn() if callable(fn) else None
        except Exception:
            return None

    def _store(self, e: Execution, res: ToolResult) -> ToolResult:
        """Record a terminal result on its execution (frozen envelope + mutable working copy)."""
        if res.observation_id is None:
            res = dataclasses.replace(res, observation_id=self._obs_id())
        e.status = res.status
        e.data = dict(res.data)
        e.t_end = res.t_end
        e.result = res
        if res.data.get("executor"):
            e.executor = res.data.get("executor")
        self._results = (self._results + [res])[-RESULTS_KEPT:]
        return res

    def _speech_result(self, e: Execution, res: ToolResult) -> None:
        if res.observation_id is None:
            res = dataclasses.replace(res, observation_id=self._obs_id())
        e.result = res
        self._results = (self._results + [res])[-RESULTS_KEPT:]

    def _speak(self, call: ToolCall, verdict: Verdict, v: int, epoch: int | None = None) -> None:
        text = str(verdict.args.get("text", "")).strip()
        speech_epoch = self.task.control_epoch if epoch is None else epoch
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._new_exec("speak", {"text": text}, v, call.tag, source=src, epoch=speech_epoch)
        self.speech.enqueue(SpeechItem(e, text, v, speech_epoch))
        self.tracer.log("say_queued", text=text, version=v, epoch=speech_epoch, execution_id=e.execution_id)

    # the THOR-era name, for callers outside the harness
    def _say(self, call: ToolCall, v: int, epoch: int | None = None) -> None:
        self._speak(call, Verdict(True, args=dict(call.args)), v, epoch)

    def _say_immediate(self, text: str) -> None:
        """Non-blocking safety acknowledgement, queued only after halt."""
        e = self._new_exec("speak", {"text": text}, self.task.intent_version,
                           tag="runtime:safety-ack", source="harness")
        self.speech.enqueue_priority(
            SpeechItem(e, text, self.task.intent_version, self.task.control_epoch))
        self.tracer.log("safety_ack_queued", text=text, epoch=self.task.control_epoch)
        self.state.record("safety_ack", self.clock.now(), priority=2, text=text,
                          control_epoch=self.task.control_epoch)

    def _reject(self, call: ToolCall, verdict: Verdict, v: int) -> None:
        stage = verdict.stage.value if verdict.stage is not None else "state"
        src = "persona" if self.persona.goal is not None else "brain"
        args = dict(call.args or {})
        e = self._new_exec(call.tool, args, v, call.tag, source=src, status="rejected")
        res = rejected_result(call.tool, args, stage=stage, code=verdict.code, message=verdict.message,
                              generation=v, control_epoch=self.task.control_epoch, source=src,
                              t=round(self.clock.now(), 3),
                              execution_id=None if stage == "schema" else e.execution_id,
                              observation_id=self._obs_id())
        self._store(e, res)
        self._note = f"rejected {call.tool}({call.args}): {verdict.message}"
        self._rejects += 1
        self.tracer.log("rejected", tool=call.tool, args=call.args, why=verdict.message, stage=stage,
                        code=verdict.code, execution_id=e.execution_id)

    def _timeout(self, e: Execution) -> float:
        fn = getattr(self.robot, "timeout_s", None)
        t: float | None = None
        if callable(fn):
            try:
                t = float(fn(e.tool_name, dict(e.args)))
            except Exception:
                t = None
        if t is None:
            if e.tool_name == "navigate":
                t = 20.0 if e.action == "reposition" else navigate_timeout(
                    self.map, self.belief.robot_at.value, e.args.get("location", ""), self.belief.blocked)
            elif e.tool_name == "manipulate":
                spec = self.registry.get(e.args.get("skill_id")) if self.registry is not None and hasattr(
                    self.registry, "get") else None
                t = spec.timeout_s() if spec is not None else (PICK_TIMEOUT if e.action == "pick" else PLACE_TIMEOUT)
            elif e.tool_name == "list_locations":
                t = INSTANT_TIMEOUT
            else:
                t = SENSE_TIMEOUT
        if e.tool_name == "navigate" and e.args.get("timeout_s"):
            t = min(t, float(e.args["timeout_s"]))            # timeout_s can only lower the default
        return t

    async def _run_instant(self, call: ToolCall, verdict: Verdict, v: int) -> None:
        """list_locations: instant, no resource; the result comes back into ACTIONS."""
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._new_exec(call.tool, verdict.args, v, call.tag, source=src)
        res = await run_execution(self.robot, self.clock, e, self._timeout(e), profile=self.profile)
        self._store(e, res)
        self.tracer.log("result", skill=call.tool, tool=call.tool, action=None, execution_id=e.execution_id,
                        status=res.status, late=False, data={k: v for k, v in res.data.items()},
                        source=src, summary=res.summary)

    async def _run_sense(self, call: ToolCall, verdict: Verdict, v: int) -> None:
        src = "persona" if self.persona.goal is not None else "brain"
        tool = call.tool
        args = dict(verdict.args)
        if tool == "look":                                 # WL_LOOK_TOOL=1: a planner-visible scan
            args = {"mode": "scan", "why": "look"}
        e = self._new_exec(tool, args, v, call.tag, source=src, action=verdict.action)
        h = ActionHandle(e.execution_id, v, tool, dict(args), e.resources, source=src)
        self.actions[e.execution_id] = h
        h.task = self._spawn(self._sense(h, e))
        await asyncio.wait([h.task])

    async def _sense(self, h: ActionHandle, e: Execution) -> None:
        async with self._sense_lock:
            e.status, e.t_start = "running", round(self.clock.now(), 3)
            if h.cancel_requested:
                res = finish(e, "cancelled", {"reason": "cancelled before start"}, t_end=round(self.clock.now(), 3),
                             observation_id=self._obs_id())
            else:
                res = await run_execution(self.robot, self.clock, e, self._timeout(e), on_handle=self._binder(h),
                                          profile=self.profile)
        self._finish(h, e, res)

    def _start_body(self, call: ToolCall, verdict: Verdict, v: int) -> None:
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._new_exec(call.tool, verdict.args, v, call.tag, source=src, action=verdict.action,
                           status="running")
        e.t_start = round(self.clock.now(), 3)
        h = ActionHandle(e.execution_id, v, call.tool, dict(e.args), e.resources, source=src)
        self.actions[e.execution_id] = h
        h.task = self._spawn(self._body(h, e))
        self.tracer.log("started", tool=call.tool, args=self._brief_args(e.args), action=e.action, version=v,
                        execution_id=e.execution_id)
        self.state.record("behavior_started", self.clock.now(), priority=3,
                          tool=call.tool, action=e.action, args=self._brief_args(e.args), version=v,
                          execution_id=e.execution_id, control_epoch=self.task.control_epoch)

    @staticmethod
    def _brief_args(args: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in args.items() if k not in ("candidates", "stance")}

    def _binder(self, h: ActionHandle):
        def bind(handle: Any) -> None:
            h.handle = handle
            if h.cancel_requested:
                handle.cancel(h.cancel_reason or "cancelled")
        return bind

    async def _body(self, h: ActionHandle, e: Execution) -> None:
        a = h.args
        now = round(self.clock.now(), 3)
        if h.cancel_requested:
            res = finish(e, "cancelled", {"reason": "cancelled before start", "location": a.get("location"),
                                          "object_type": a.get("object_type"), "action": e.action},
                         t_end=now, observation_id=self._obs_id())
            here = None
        else:
            here = self._surface_here()
            res = await run_execution(self.robot, self.clock, e, self._timeout(e), on_handle=self._binder(h),
                                      profile=self.profile)
            if h.skill == "manipulate" and e.action == "pick" and res.ok and here:
                oid = res.data.get("object_id") or a.get("object_id")
                if oid:
                    self._picked_from[oid] = here
        self._finish(h, e, res)
        late = e.generation < self.task.intent_version
        if h.skill == "navigate" and res.ok and not h.cancel_requested and not late and not self.task.paused:
            if e.action == "reposition":
                await self._observe("after reposition", mode="glance")
            else:
                await self._observe("arrival", mode="scan")     # the robot looks around when it arrives
        if h.skill == "manipulate" and not h.cancel_requested and not self.task.paused:
            # never trust a success flag: a glance straight ahead, and a scan if it didn't show the object
            await self._observe(f"verify {e.action}", mode="glance")
            oid = (e.data or {}).get("object_id") or a.get("object_id")
            target = (e.data or {}).get("surface") or a.get("target") or here
            ob = self.belief.objects.get(oid) if oid else None
            if e.action == "place" and res.ok and not (
                    ob is not None and ob.where.verified and ob.where.value == target):
                await self._observe("verify place: not in the glance", mode="auto")
            if e.action == "place" and a.get("goal") and res.ok and oid:
                self._check_goal(oid, str(a["goal"]))
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

    def _finish(self, h: ActionHandle, e: Execution, res: ToolResult) -> None:
        now = self.clock.now()
        late = h.created_for < self.task.intent_version
        if late:
            res = dataclasses.replace(res, late=True)    # rule 1: recorded, but not progress for the new request
        res = self._store(e, res)
        e.t_end = round(now, 3)
        if late:
            e.data["late"] = True
        self.actions.pop(h.entry_id, None)
        tool = h.skill
        d = res.data
        if tool == "navigate":
            self.belief.apply_navigate(res.status, d, now)
            self._last_motion_t = now
        elif tool in ("observe", "look") and res.ok:
            self.belief.apply_observation(d, now, self._surface_xy, self._surface_h)
        elif tool == "manipulate":
            arm = h.args.get("arm") or d.get("arm")
            oid = d.get("object_id") or h.args.get("object_id")
            if arm in ARMS:
                if late or h.cancel_requested or not res.ok:
                    hint = oid if (e.action == "pick" and d.get("holding")) else None
                    self.belief.mark_hand_unknown(arm, "late_result" if late else f"{e.action}_{res.status}",
                                                  now, hint=hint)
                elif e.action == "pick" and oid:
                    self.belief.claim_pick(arm, oid, now)
                elif oid:
                    self.belief.claim_place(arm, oid, d.get("surface") or h.args.get("target") or self._surface_here(),
                                            now)
            try:
                if float(d.get("base_shift_m") or 0.0) > 0.02:  # whole-body tokens moved the base (PLAN 1.3 #25)
                    self._last_motion_t = now
            except (TypeError, ValueError):
                pass
        self._refresh_observation()
        self._note_places()
        if self.persona.goal is not None and self.persona.is_done(self.belief):
            self._end_own_goal("done")            # before System 2 is asked again,
            if not self.task.paused:              # and hand it the next one straight away
                self._start_own_goal(self.persona.propose(self.belief, self.map, now, quiet_s=1e9))
        brief = {k: v for k, v in d.items() if k not in ("surfaces", "views", "landmarks", "candidates")}
        if tool in ("observe", "look") and res.ok:
            brief["saw"] = sorted(v["id"] for items in (d.get("surfaces") or {}).values() for v in items)
            brief["landmarks"] = [lm["id"] for lm in d.get("landmarks") or []]
        self.tracer.log("result", skill=tool, tool=tool, action=e.action, execution_id=e.execution_id,
                        status=res.status, late=late, data=brief, source=e.source, executor=d.get("executor"),
                        summary=res.summary, observation_id=res.observation_id,
                        why=e.args.get("why") if tool in ("observe", "look") else None)
        if late:
            self.tracer.log("late_result", tool=tool, execution_id=e.execution_id, generation=e.generation,
                            now=self.task.intent_version, status=res.status)
        self.state.record("behavior_result", now, priority=3, tool=tool, action=e.action,
                          status=res.status, late=late, control_epoch=e.control_epoch, data=brief)

    # ------------------------------------------------------------------
    # wait_and_observe (PLAN 5.1): look first, then hold still until something changes
    # ------------------------------------------------------------------
    def _belief_fingerprint(self) -> dict[str, Any]:
        b = self.belief
        return {"objects": {oid: (ob.where.value, ob.where.verified) for oid, ob in b.objects.items()},
                "landmarks": {k: lm.near.value for k, lm in b.landmarks.items()},
                "hands": {arm: b.holding[arm].value for arm in ARMS}}

    @staticmethod
    def _belief_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for oid, (where, _v) in after["objects"].items():
            old = before["objects"].get(oid)
            if old is None:
                out.append(f"saw {oid} on {where}" if where not in (None, UNKNOWN) else f"new {oid}")
            elif old[0] != where:
                if where == UNKNOWN:
                    out.append(f"{oid} no longer on {old[0]}")
                else:
                    out.append(f"{oid} now {where}" + (f" (was {old[0]})" if old[0] not in (None, UNKNOWN) else ""))
        for k in after["landmarks"]:
            if k not in before["landmarks"]:
                out.append(f"saw {k}")
        for arm, held in after["hands"].items():
            if before["hands"].get(arm) != held:
                out.append(f"{arm} hand: {held or 'empty'}")
        return out

    def _request_open(self) -> bool:
        """An utterance nobody has answered yet, or physical work in this generation after the last line."""
        says = [e for e in self.history if e.tool_name == "speak" and e.status not in ("dropped", "rejected")
                and e.source != "harness"]
        last_say = says[-1].t_start if says else -1e9
        for u in self.task.utterances:
            if self.task.kinds.get(u.id) in ("stop", "chitchat"):
                continue
            if not any(s.t_start >= u.t_end - 1e-6 for s in says):
                return True
        work = [e for e in self.history if e.generation == self.task.intent_version and e.source == "brain"
                and e.tool_name in ("navigate", "manipulate", "check_reachability") and e.status != "rejected"]
        return bool(work) and work[-1].t_start > last_say and self.task.goal is not None

    def _quiet_continue(self) -> bool:
        """Quiet rule (PLAN 5.1 step 5): after a no-change result, ask again only if a request is
        open or an own goal is active, and never after two no-change results in a row."""
        self._nochange += 1
        keep = (self._request_open() or self.persona.goal is not None) and self._nochange < QUIET_NOCHANGE_LIMIT
        if not keep:
            self.tracer.log("wait_quiet", nochange=self._nochange)
        return keep

    async def _wait_and_observe(self, call: ToolCall, verdict: Verdict, v: int, epoch: int) -> bool:
        """Returns True when the think loop should ask the brain again right away."""
        args = dict(verdict.args)
        timeout = float(args.get("timeout_s", 10.0))
        src = "persona" if self.persona.goal is not None else "brain"
        e = self._new_exec("wait_and_observe", args, v, call.tag, source=src, status="running")
        e.t_start = round(self.clock.now(), 3)
        h = ActionHandle(e.execution_id, v, "wait_and_observe", args, e.resources, source=src)
        self.actions[e.execution_id] = h          # a correction, a stop or an own-goal drop cancels it
        self.tracer.log("started", tool="wait_and_observe", args=args, action=None, version=v,
                        execution_id=e.execution_id)
        before = self._belief_fingerprint()
        noticed0 = len(self.noticed)
        finished0 = {x.execution_id for x in self.history if x.finished}
        obs = await self._observe("wait_and_observe", mode=self._choose_observe_mode(), source="harness")
        obs_id = obs.observation_id if obs is not None else self._obs_id()
        changes = self._belief_changes(before, self._belief_fingerprint())

        def done(status: str, summary: str | None = None, extra: list[str] | None = None) -> None:
            data = {"status": status, "observation_id": obs_id, "summary": summary, "changes": list(extra or []),
                    "reason": "cancelled" if status == "cancelled" else None}
            res = finish(e, status, data, t_end=round(self.clock.now(), 3), observation_id=obs_id)
            h_late = e.generation < self.task.intent_version
            if h_late:
                res = dataclasses.replace(res, late=True)
            self._store(e, res)
            self.actions.pop(e.execution_id, None)
            self.tracer.log("result", skill="wait_and_observe", tool="wait_and_observe", action=None,
                            execution_id=e.execution_id, status=res.status, late=h_late,
                            data={"status": status, "summary": summary, "changes": list(extra or [])},
                            source=src, summary=res.summary, observation_id=obs_id)

        def cancelled() -> bool:
            return (h.cancel_requested or self.task.intent_version != v or self.task.control_epoch != epoch)

        if cancelled():
            done("cancelled")
            return False
        if changes:
            done("changed", "; ".join(changes[:6]), changes)
            self._nochange = 0
            return True
        if timeout <= 0:
            done("unchanged")
            return self._quiet_continue()
        # A wake that arrived while the brain was thinking or the robot was looking counts:
        # the event is not cleared here.
        deadline = self.clock.now() + timeout
        why: str | None = None
        while self.clock.now() < deadline:
            if cancelled():
                done("cancelled")
                return False
            if self._wake_evt.is_set():
                why = self._wake_why or "woken"
                self._wake_evt.clear()
                break
            now_fp = self._belief_fingerprint()
            ch = self._belief_changes(before, now_fp)
            if ch:
                why = "; ".join(ch[:6])
                break
            if len(self.noticed) > noticed0:
                why = f"noticed: {self.noticed[-1]['text']}"
                break
            ended = [x for x in self.history if x.finished and x.execution_id not in finished0
                     and x.tool_name not in ("speak", "observe", "wait_and_observe") and x.status != "rejected"]
            if ended:
                why = f"{ended[-1].tool_name} {ended[-1].status}"
                break
            await self.clock.sleep(WAIT_POLL_S)
        if why is not None:
            done("changed", why, [why])
            self._nochange = 0
            return True
        done("timed_out")
        return self._quiet_continue()

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


__all__ = ["Runtime", "is_stop", "fallback_kind", "BODY", "SENSE", "INSTANT", "REACH_FRESH_S", "NEUTRAL_TOOLS",
           "resources_of"]
