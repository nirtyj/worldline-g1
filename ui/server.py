#!/usr/bin/env python3
"""Worldline on G1: chat with a Unitree G1 in an Isaac Sim house and watch the runtime work.

    .venv/bin/python -m ui.server --profile sonic --scene procthor-train-40   # then open http://localhost:8765
    .venv/bin/python -m ui.server --profile lite                              # pure Python, no simulator

Keys (never read from ludo-runtime): $WORLDLINE_ENV, then <repo>/.env, then
~/.config/ludo-g1/secrets.env. GEMINI_API_KEY for the Gemini planner, ANTHROPIC_API_KEY
only if you pick Claude, TYPESAFE_API_KEY for System 1's Jev labels.

    chat ─(text)─► InteractiveUser.next() ─► Runtime (agent/) ─► CompositeBrain ─► planner
                                               │   ▲               (classify, next tool call)
                                               ▼   └── perception + body sense (fused state)
                     robot facade (robot/): services ─► BodyServer ─► SONIC ─► G1 in Isaac Sim
                                               │
         world model (world/, the only ground-truth reader) ──► the page's "reality" panels
         viz FrameTap (head 5565, chase/top 5602) ────────────► the page's camera panes

The page is an observer: it reads the simulator's ground truth through the world model,
which the planner never can. That is what lets you see belief and truth disagree. Every
ground-truth panel says so ("sim ground truth: not visible to the planner").
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import enum
import http
import importlib
import inspect
import json
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from websockets.asyncio.server import broadcast, serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from ui.cameras import CameraFeed, CameraSource, NoCameras, TapCameras, WorldCameras  # noqa: E402
from ui.interactive_user import InteractiveUser  # noqa: E402
from ui.recorder import CallRecorder  # noqa: E402
from ui.robot_map import RobotMap  # noqa: E402
from ui.truth import (DISPLAY_STEP, FALLBACK_LABELS, GT_LABEL, display_cells, layout_message,  # noqa: E402
                      stepping_stones, to_plain, truth_payload, world_grid)

INDEX = Path(__file__).resolve().parent / "index.html"
LOG_DIR = ROOT / "runs" / "playground"
TICK_S = 0.2
PLANNER = "agent.model:create_brain"
# System 1 plugs in here: "module:factory", where factory(on_status) returns an object with
# `status`, and async run(), route(text), update(context), frame(jpeg, where), robot_said(text)
# and observe() (see brains/interface.py System1). Default: Jev labels + the Gemini Live observer
# (PLAN 8.1). SYSTEM1=off: the planner classifies every message.
SYSTEM1 = os.environ.get("SYSTEM1", "brains.system1_jev:create")
S1_ROUTE_TIMEOUT_S = 2.0      # longer than this and the planner classifies instead
S1_OBSERVE_EVERY_S = 5.0      # ask for observations this often while new frames arrive
S1_OWN_GRACE_S = 3.0          # frames this soon after a manipulate still show the robot's own doing (PLAN 8.4)
QUIET_EVENTS = {"speech_ended"}
MEMORY_ROWS = {"memory_saved", "place_learned", "note_saved", "delivered", "recall", "persona_goal_end", "observation"}
VIEWS = ROOT / "ui" / "views"
SPEECH_TOOLS = ("speak", "say")                 # "say" only so a THOR-era trace still renders
OBSERVE_TOOLS = ("navigate", "observe", "wait_and_observe")
SCAN_TOOLS = ("observe", "wait_and_observe")

AGENTS = [
    ("agent", "Runtime (agent/)"),
    ("agent.mutants:forget_cancelled_grasp", "Mutant: forgets cancelled grasp"),
    ("agent.mutants:cancel_on_everything", "Mutant: cancels on everything"),
    ("agent.mutants:no_wait_for_chunk", "Mutant: no reconcile gate"),
    ("agent.mutants:no_drop_speech", "Mutant: keeps stale speech"),
    ("agent.mutants:trust_success", "Mutant: trusts success flag"),
    ("agent.mutants:no_keyword_stop", "Mutant: stop goes through model"),
    ("agent.mutants:no_stale_check", "Mutant: no stale-decision check"),
]
MODELS = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash (fast)"),
    ("gemini-3.8-flash:think", "Gemini 3.8 Flash (thinking)"),
    ("claude-sonnet-5", "Sonnet 5"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5 (faster, cheaper)"),
]
PROFILES = [
    ("lite", "lite · pure Python, no simulator"),
    ("bringup", "bringup · Isaac, kinematic base + attach (fallbacks)"),
    ("sonic", "sonic · SONIC walks, arm script + attach grasp (fallback)"),
    ("full", "full · SONIC walks, GR00T manipulates"),
]
# MolmoSpaces houses in Isaac (PLAN 0.3); the eval binds H40/H15/K10 to the first three (eval/scenes.yaml).
SCENES = [
    ("procthor-train-40", "House 40 (H40)"),
    ("procthor-train-15", "House 15 (H15)"),
    ("ithor-FloorPlan10", "Kitchen 10 (K10)"),
    ("procthor-train-38", "House 38 (M1 walk house)"),
    ("procthor-train-59", "House 59"),
]


def load_env_files() -> list[Path]:
    """KEY=VALUE lines into the environment, existing values win (PLAN 4.1 key order)."""
    paths = [Path(p) for p in [os.environ.get("WORLDLINE_ENV")] if p]
    paths += [ROOT / ".env", Path.home() / ".config" / "ludo-g1" / "secrets.env"]
    used = []
    for path in paths:
        if not path.is_file():
            continue
        used.append(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))
    return used


def _import(spec: str) -> Any:
    """"package.module:attr" -> the attribute (or the module when there's no ":attr")."""
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module


def _jsonable(o: Any) -> Any:
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(bytes(o)).decode()
    if isinstance(o, enum.Enum):
        return o.value
    if is_dataclass(o) or callable(getattr(o, "to_dict", None)) or callable(getattr(o, "tolist", None)):
        return to_plain(o)
    return str(o)


def dumps(msg: dict[str, Any]) -> str:
    return json.dumps(msg, default=_jsonable)


def _call(obj: Any, name: str, *a: Any, default: Any = None) -> Any:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001  (a broken read must not kill the page's ticker)
        return default


# ----------------------------------------------------------------------------------------------
# What the server needs from the rest of the stack. Resolved lazily, so the page's own tests run
# with fakes and nothing here imports agent/, brains/, robot/ or world/ at module load.
# ----------------------------------------------------------------------------------------------
@dataclass
class Deps:
    build: Callable[..., Any] | None = None                 # robot.factory.build(profile, scene, clock, log)
    create_planner: Callable[[Any], Any] | None = None      # agent.model.create_brain(info)
    create_runtime: Callable[..., Any] | None = None        # (spec, robot, user, brain, clock) -> runtime
    brain_info: Callable[..., Any] | None = None            # brains.interface.BrainInfo
    composite: Callable[[Any], Any] | None = None           # brains.composite.CompositeBrain
    clock: Callable[[float], Any] | None = None             # sim.clock.SimClock
    event_log: Callable[[Any], Any] | None = None           # sim.log.EventLog
    is_stop: Callable[[str], bool] | None = None            # agent.harness.is_stop
    frame_gate: Callable[[], Any] | None = None             # brains.frame_gate.FrameGate (G1 head preset if any)
    system1: Callable[..., Any] | None = None               # SYSTEM1 factory(on_status)
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> "Deps":
        """Fill every unset hook from the real modules (imported here, not at module load)."""
        if self.build is None:
            self.build = _import("robot.factory:build")
        if self.create_planner is None:
            self.create_planner = lambda info: _import(PLANNER)(info)
        if self.create_runtime is None:
            def create_runtime(spec: str, robot: Any, user: Any, brain: Any, clock: Any) -> Any:
                return _import(spec if ":" in spec else spec + ":create_runtime")(robot, user, brain, clock)
            self.create_runtime = create_runtime
        if self.brain_info is None:
            self.brain_info = _import("brains.interface:BrainInfo")
        if self.composite is None:
            self.composite = _import("brains.composite:CompositeBrain")
        if self.clock is None:
            self.clock = _import("sim.clock:SimClock")
        if self.event_log is None:
            self.event_log = _import("sim.log:EventLog")
        if self.is_stop is None:
            self.is_stop = _import("agent.harness:is_stop")
        if self.frame_gate is None:
            self.frame_gate = _default_frame_gate
        return self


def _default_frame_gate() -> Any:
    fg = _import("brains.frame_gate")
    cfg = getattr(fg, "G1_HEAD", None)
    if cfg is not None:
        try:
            return fg.FrameGate(cfg)
        except TypeError:
            try:
                return fg.FrameGate(config=cfg)
            except TypeError:
                pass
    return fg.FrameGate()


def _gate_decide(gate: Any, frame: Any, pose: Any, now: float, expected: bool, stationary: bool | None) -> Any:
    try:
        if "stationary" in inspect.signature(gate.decide).parameters:
            return gate.decide(frame, pose, now, expected=expected, stationary=stationary)
    except (TypeError, ValueError):
        pass
    return gate.decide(frame, pose, now, expected=expected)


def _decode_rgb(jpeg: bytes) -> Any:
    """JPEG -> RGB array for the frame gate (Pillow + numpy), or None without them."""
    try:
        import io

        import numpy as np
        from PIL import Image
        im = Image.open(io.BytesIO(jpeg))
        im.draft("RGB", (160, 120))
        return np.asarray(im.convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


def _row(r: Any, i: int) -> dict[str, Any]:
    """A trace row as a dict: TraceLog rows are dicts; api.events.InteractionEvent is flattened
    to {t, type, generation, control_epoch, **payload} (old row names are event_type aliases)."""
    if isinstance(r, dict):
        return {**r, "i": i}
    et = getattr(r, "event_type", None)
    if et is not None:
        payload = to_plain(getattr(r, "payload", {}) or {})
        return {"t": getattr(r, "timestamp", 0.0), "type": et, "seq": getattr(r, "seq", i),
                "generation": getattr(r, "generation", None), "control_epoch": getattr(r, "control_epoch", None),
                **(payload if isinstance(payload, dict) else {"payload": payload}), "i": i}
    d = to_plain(r)
    return {**(d if isinstance(d, dict) else {"value": d}), "i": i}


def _tool_of(e: Any) -> str:
    return str(getattr(e, "tool_name", None) or getattr(e, "tool", "") or "")


def _eid(e: Any) -> str:
    return str(getattr(e, "execution_id", None) or getattr(e, "id", ""))


def _finished(e: Any) -> bool:
    f = getattr(e, "finished", None)
    if isinstance(f, bool):
        return f
    return str(getattr(e, "status", "")).lower() not in ("queued", "running", "cancelling", "executing", "canceling")


def derive_tool_state(active: list[dict[str, Any]], paused: bool, reconciling: bool, body_mode: str | None) -> str:
    """PLAN 5.8's tool state when the runtime does not publish its own (api/state_machine.py)."""
    if body_mode in ("FAULT", "ESTOP"):
        return "FAULT"
    if paused:
        return "STOPPED"
    if any(e.get("cancel_requested") or str(e.get("status")).lower() in ("cancelling", "canceling") for e in active):
        return "CANCELLING"
    if reconciling:
        return "RECONCILING"
    tools = [e.get("tool") for e in active if e.get("tool") not in SPEECH_TOOLS]
    if "manipulate" in tools:
        return "MANIPULATING"
    if "navigate" in tools:
        return "NAVIGATING"
    if "observe" in tools:
        return "OBSERVING"
    if "wait_and_observe" in tools:
        return "WAITING"
    return "IDLE"


class Session:
    """One house, one robot stack, one runtime, one brain, fed by the chat box."""

    def __init__(self, scene: str, agent: str, model: str, profile: str, deps: Deps) -> None:
        self.scene, self.agent, self.model, self.profile, self.deps = scene, agent, model, profile, deps
        self.error: str | None = None
        self.runtime: Any = None
        self.planner: Any = None
        self.task: asyncio.Task | None = None
        self.recorder: CallRecorder | None = None
        self.calls_log: Path | None = None
        self.world: Any = None
        self.robot: Any = None
        self.frames: Any = None
        self.map: dict[str, Any] = {}
        self.layout: dict[str, Any] = {}
        self.stones = stepping_stones(profile)
        self._ev_i = 0
        self._tr_i = 0
        self._call_rev = 0
        self._bg: list[asyncio.Task] = []

    async def start(self) -> None:
        d = self.deps
        self.clock = d.clock(1.0)               # real models run in wall time
        self.log = d.event_log(self.clock)
        built = d.build(self.profile, self.scene, self.clock, self.log)
        if inspect.isawaitable(built):
            built = await built
        built = list(built) + [None, None, None]
        self.world, self.robot, self.frames = built[0], built[1], built[2]
        self.map = self.robot.lookup_keypoints()
        self.grid = world_grid(self.world)
        self.layout = layout_message(self.scene, self.map, self.world, self.grid, profile=self.profile)
        free = display_cells(self.grid, DISPLAY_STEP, "walk") if self.grid else {tuple(c) for c in self.layout["grid"]}
        floor = display_cells(self.grid, DISPLAY_STEP, "floor") if self.grid else None
        self.robot_map = RobotMap(free, DISPLAY_STEP, floor)     # what the robot itself knows of the space
        self.user = InteractiveUser(self.clock, self.log)
        model, _, mode = self.model.partition(":")
        options = {"model": model, "temperature": "none", "profile": self.profile}
        if model.startswith("gemini"):
            # Gemini thinks by default; "fast" turns that off (about 1-2 s a step)
            options["thinking_budget"] = "none" if mode == "think" else "0"
        info = d.brain_info(scenario_id=self.scene, map=self.map, meanings=None, options=options, clock=self.clock)
        try:
            self.planner = d.create_planner(info)
        except Exception as e:  # noqa: BLE001
            self.error = f"The planner could not start: {e}"
            return
        self.calls_log = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.scene}_{self.profile}_model_calls.txt"
        self.recorder = CallRecorder(self.planner, self.clock, self._version, self.calls_log)
        brain = d.composite(self.planner)
        try:
            self.runtime = d.create_runtime(self.agent, self.robot, self.user, brain, self.clock)
        except Exception:  # noqa: BLE001
            self.error = "The runtime could not start:\n" + traceback.format_exc()
            return
        self.task = asyncio.create_task(self.runtime.run())

    async def stop(self) -> None:
        for t in self._bg:
            t.cancel()
        if self.task:
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        if self.robot is not None:
            res = _call(self.robot, "shutdown")
            if inspect.isawaitable(res):
                try:
                    await asyncio.wait_for(res, 5.0)
                except Exception:  # noqa: BLE001
                    pass
        close = getattr(self.world, "close", None)
        if callable(close):
            try:
                res = close()
                if inspect.isawaitable(res):
                    await asyncio.wait_for(res, 5.0)
            except Exception:  # noqa: BLE001
                pass

    def _version(self) -> int:
        rt = self.runtime
        if rt is None:
            return 0
        task = getattr(rt, "task", None)
        return task.intent_version if task is not None else getattr(rt, "version", 0)

    def hear(self, text: str, directive: dict[str, Any] | None = None) -> None:
        if self.runtime is None:
            raise RuntimeError(self.error or "nothing is running")
        self.user.say(text, directive)

    def record_system1(self, purpose: str, input_text: str, response: Any, latency_s: float | None) -> None:
        """System 1's calls go in the page's model-call list next to the planner's."""
        if self.recorder is None:
            return
        now = round(self.clock.now(), 2)
        self.recorder.record_external({
            "via": "system1", "purpose": purpose, "t_start": round(now - (latency_s or 0), 2), "t_end": now,
            "version_start": self._version(), "version_end": self._version(), "input": input_text,
            "response": {"tool": purpose, "args": response, "text": ""}, "latency_s": latency_s,
            "status": "ok" if response else "fallback"})

    # ------------------------------------------------------------------
    # What the page sees
    # ------------------------------------------------------------------
    def _trace_rows(self) -> list[Any]:
        rt = self.runtime
        if rt is None:
            return []
        rows = getattr(getattr(rt, "tracer", None), "rows", None)
        if rows is None:
            rows = getattr(getattr(rt, "context", None), "rows", None)
        if rows is None:
            rows = getattr(rt, "_log", None) or []
        return rows

    def _events_from(self, start: int) -> list[dict[str, Any]]:
        return [{**e, "i": i} for i, e in enumerate(self.log.events[start:], start)
                if e["type"] not in QUIET_EVENTS]

    def _trace_from(self, start: int) -> list[dict[str, Any]]:
        return [_row(r, i) for i, r in enumerate(self._trace_rows()[start:], start)]

    def init_message(self) -> dict[str, Any]:
        return {"type": "init",
                "config": {"scene": self.scene, "agent": self.agent, "model": self.model, "profile": self.profile,
                           "stepping_stones": sorted(self.stones), "fallback_labels": FALLBACK_LABELS,
                           "truth_label": GT_LABEL},
                "map": self.map, "layout": self.layout,
                "events": self._events_from(0), "trace": self._trace_from(0),
                "calls": self.recorder.calls if self.recorder else [],
                "calls_log": str(self.calls_log.relative_to(ROOT)) if self.calls_log else None,
                "error": self.error, **self.state(), "robot_map": self.robot_map.full()}

    def memory_message(self) -> dict[str, Any]:
        """What the robot has stored, for the page's Memory tab: spatial memory,
        notes, earlier sessions and the procedural graph. Sent on change, not every frame."""
        rt = self.runtime
        mem = getattr(rt, "memory", None)
        if mem is None:
            return {"type": "memory", "scene": self.scene, "available": False}
        try:
            from agent.episodes import summarize
            from agent.memory import usual_place, volatility
        except Exception:  # noqa: BLE001
            return {"type": "memory", "scene": self.scene, "available": False}
        wall = time.time()
        room_of = {k: v.get("room") for k, v in (self.map.get("keypoints") or {}).items()}
        spatial = []
        for oid, m in sorted(mem.data["objects"].items()):
            hist = m.get("history") or []
            spatial.append({
                "id": oid, "type": m.get("type"), "surface": m.get("surface"), "room": room_of.get(m.get("surface")),
                "status": m.get("status", "seen"), "usual": usual_place(hist),
                "volatility": volatility(m.get("type", ""), hist), "seen_ago_s": round(wall - m.get("seen_wall", wall)),
                "history": [{"surface": h["surface"], "room": room_of.get(h["surface"]), "ago_s": round(wall - h["wall"])}
                            for h in hist][::-1]})
        landmarks = [{"id": k, "label": m.get("label"), "near": m.get("near"), "room": room_of.get(m.get("near"))}
                     for k, m in sorted(mem.data["landmarks"].items())]
        past = getattr(getattr(rt, "recaller", None), "past", None) or []
        sessions = [summarize(rows) for rows in past[-9:]]
        sessions.append({**summarize(self._trace_rows()), "wall": wall - self.clock.now(), "current": True})
        g = getattr(rt, "procedures", None)
        procedures = ({"nodes": g.nodes, "edges": g.edges, "rules": g.rules, "tasks": g.tasks,
                       "learned_wall": g.learned_wall} if g is not None else None)
        return {"type": "memory", "scene": self.scene, "available": True, "wall": wall,
                "spatial": spatial, "landmarks": landmarks, "notes": mem.notes(),
                "observations": mem.observations()[::-1],
                "looked": sorted(mem.data["looked"]), "sessions": sessions[::-1], "procedures": procedures}

    def frame(self) -> dict[str, Any]:
        events = self._events_from(self._ev_i)
        trace = self._trace_from(self._tr_i)
        self._ev_i = len(self.log.events)
        self._tr_i = len(self._trace_rows())
        calls: list[dict[str, Any]] = []
        if self.recorder:
            calls = self.recorder.changed_since(self._call_rev)
            self._call_rev = self.recorder.rev
        st = self.state()
        self._track(trace, st)
        return {"type": "frame", "events": events, "trace": trace, "calls": calls, **st,
                "robot_map": self.robot_map.take_new()}

    def _robot_pose(self, st: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
        """The robot's own pose estimate (its localizer; GT-backed in sim), else the truth pose."""
        tel = _call(self.robot, "telemetry", default=None) or {}
        p = tel.get("pose") or {}
        if p.get("x") is not None and p.get("z") is not None:
            return float(p["x"]), float(p["z"]), float(p.get("yaw") or 0.0), float(p.get("horizon") or 0.0)
        r = ((st or self.state()).get("truth") or {}).get("robot") or {}
        return float(r.get("x") or 0.0), float(r.get("z") or 0.0), float(r.get("yaw") or 0.0), float(r.get("horizon") or 0.0)

    def _track(self, trace: list[dict[str, Any]], st: dict[str, Any]) -> None:
        """Feed the robot's own map: its pose (odometry), whether it is scanning,
        and where it stood when it verified where something is."""
        x, z, yaw, _ = self._robot_pose(st)
        rt = st.get("runtime") or {}
        looking = (rt.get("tool_state") == "OBSERVING"
                   or any(e.get("tool") in SCAN_TOOLS for e in rt.get("active") or [])
                   or any(e.get("tool") in SCAN_TOOLS for e in (st.get("truth") or {}).get("executions") or []))
        self.robot_map.pose(self.clock.now(), x, z, yaw, looking)
        st["view"] = self.robot_map.frustum(x, z, yaw, looking)
        for r in trace:
            place = str(r.get("place") or "")
            if r.get("type") == "place_learned" and place and place != "UNKNOWN" and not place.startswith("hand"):
                self.robot_map.found(r["t"], r["object"], place, x, z)

    def _execution(self, e: Any, cancel_requested: bool) -> dict[str, Any]:
        data = {k: v for k, v in (getattr(e, "data", None) or {}).items() if k not in ("surfaces",)}
        args = dict(getattr(e, "args", None) or {})
        executor = getattr(e, "executor", None) or data.get("executor")
        result = getattr(e, "result", None)
        late = bool(data.get("late") or getattr(result, "late", False))
        gen = getattr(e, "generation", None)
        if gen is None:
            gen = getattr(e, "created_for", None)
        status = str(getattr(e, "status", "") or "")
        t0 = getattr(e, "t_start", None)
        if t0 is None:
            t0 = getattr(e, "t_created", 0.0)
        return {"id": _eid(e), "tool": _tool_of(e), "action": getattr(e, "action", None) or args.get("action"),
                "args": args, "v": gen, "e": getattr(e, "control_epoch", None), "status": status,
                "source": getattr(e, "source", None), "t": round(float(t0 or 0.0), 2), "late": late,
                "cancel_requested": cancel_requested, "executor": executor, "skill": data.get("skill"),
                "fallback": bool(executor in self.stones or data.get("fallback")),
                "fallback_label": FALLBACK_LABELS.get(executor or ""), "data": data}

    def _body_stack(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        tel = _call(self.robot, "telemetry", default=None) or {}
        body = dict(tel.get("body") or {})
        nav = dict(body.pop("nav", None) or tel.get("nav") or {})
        if not nav.get("route"):
            route = getattr(self.robot, "route", None)
            if route:
                nav["route"] = route
        caps = _call(self.robot, "capabilities", default=None) or {}
        caps = {k: to_plain(v) for k, v in caps.items()} if isinstance(caps, dict) else {}
        health = to_plain(tel.get("health")) or {}
        stack = {"profile": self.profile, "rtf": body.get("rtf", health.get("rtf")), "health": health,
                 "services": caps, "stepping_stones": sorted(self.stones)}
        return body, stack, nav

    def state(self) -> dict[str, Any]:
        base = _call(self.robot, "base_state", default=None) or {}
        raw_truth = _call(self.world, "truth", default=None)
        truth = truth_payload(raw_truth, base, "lite-gt" if self.profile == "lite" else "isaac-gt")
        body, stack, nav = self._body_stack()
        robot_execs = _call(self.robot, "active_executions", default=None) or []
        truth["executions"] = [self._execution(e, bool(getattr(e, "cancel_requested", False)))
                               for e in robot_execs if _tool_of(e) not in SPEECH_TOOLS]
        truth["goals"] = [{"id": x["id"], "skill": x["tool"], "args": x["args"], "status": x["status"],
                           "executor": x["executor"]} for x in truth["executions"]]
        if body.get("mode") and not truth["robot"].get("mode"):
            truth["robot"]["mode"] = body.get("mode")
        out: dict[str, Any] = {"t": round(self.clock.now(), 2), "truth": truth, "runtime": None,
                               "nav": nav, "body": body, "stack": stack}
        rt = self.runtime
        if rt is None:
            return out
        belief = getattr(rt, "belief", None)
        if hasattr(belief, "to_dict"):
            belief = belief.to_dict()
        handles = getattr(rt, "actions", {})
        cancel_req: set[str] = set()
        if isinstance(handles, dict):
            for h in handles.values():
                if getattr(h, "cancel_requested", False):
                    cancel_req.add(str(getattr(h, "execution_id", None) or getattr(h, "entry_id", "")))
        history = list(getattr(rt, "history", []) or [])
        thinking = bool(getattr(rt, "_thinking", False) or getattr(rt, "_interpreting", False)
                        or getattr(rt, "_working", False))
        speech = getattr(rt, "speech", None)
        speech_state = None
        if speech is not None and hasattr(speech, "items"):
            cur = speech.current[0] if speech.current else None
            ver = lambda i: getattr(i, "generation", None) if getattr(i, "generation", None) is not None else getattr(i, "created_for", 0)  # noqa: E731
            speech_state = {"current": {"text": cur.text, "version": ver(cur)} if cur else None,
                            "queued": [{"text": i.text, "version": ver(i)} for i in speech.items]}
        stats = None
        if callable(getattr(self.planner, "stats", None)):
            try:
                stats = self.planner.stats()
            except Exception:  # noqa: BLE001
                stats = None
        crashed = None
        if self.task is not None and self.task.done() and not self.task.cancelled() and self.task.exception():
            crashed = "".join(traceback.format_exception(self.task.exception()))
        task = getattr(rt, "task", None)
        active = [self._execution(e, _eid(e) in cancel_req) for e in history if not _finished(e)]
        paused = bool(getattr(task, "paused", False))
        reconciling = bool(_call(rt, "_reconciling", default=False))
        ts = getattr(rt, "tool_state", None)
        if callable(ts):
            try:
                ts = ts()
            except Exception:  # noqa: BLE001
                ts = None
        tool_state = str(getattr(ts, "value", ts)) if ts else derive_tool_state(active, paused, reconciling, body.get("mode"))
        out["runtime"] = {
            "belief": belief, "version": self._version(),
            "control_epoch": getattr(task, "control_epoch", 0), "paused": paused, "thinking": thinking,
            "reconciling": reconciling, "tool_state": tool_state,
            "active": active,
            "recent": [self._execution(e, False) for e in history[-12:] if _tool_of(e) not in SPEECH_TOOLS],
            "executions": [self._execution(e, _eid(e) in cancel_req) for e in history[-30:]],
            "speech": speech_state, "stats": stats, "crashed": crashed,
        }
        persona = getattr(rt, "persona", None)
        if persona is not None:
            out["runtime"]["persona"] = persona.snapshot(rt.clock.now())
        if callable(getattr(rt, "stepping", None)):
            out["runtime"]["step"] = {"on": rt.step_mode, "waiting": rt.stepping()}
        if callable(getattr(rt, "task_steps", None)):
            out["runtime"]["procedure"] = {"steps": rt.task_steps()[-6:], "guidance": rt._guidance()}
        fused = getattr(rt, "state", None)
        if fused is not None and callable(getattr(fused, "snapshot", None)):
            out["runtime"]["fused"] = fused.snapshot(
                active=active, recent_actions=[self._execution(e, False) for e in history[-12:]])
        return out


class Hub:
    def __init__(self, scene: str, profile: str = "lite", deps: Deps | None = None,
                 cameras: CameraSource | str = "auto", port_offset: int | None = None,
                 system1: str | None = SYSTEM1) -> None:
        self.clients: set[Any] = set()
        self.session: Session | None = None
        self.default = scene
        self.profile = profile
        self.deps = deps or Deps()
        self._lock = asyncio.Lock()
        self._camera_mode = cameras if isinstance(cameras, str) else "given"
        self._fixed_cameras: CameraSource | None = None if isinstance(cameras, str) else cameras
        self._tap: CameraSource | None = None
        self._port_offset = port_offset
        self.feed = CameraFeed(NoCameras())
        self.persona_level = "off"                   # own goals only when you turn them on
        self.step_mode = False                       # the page's step mode, kept across sessions like persona
        self._memory_due, self._memory_sent = False, 0.0
        self.system1_spec = None if (system1 or "").lower() in ("", "off", "none") else system1
        self.system1: Any = None
        self.s1_status, self.s1_detail = ("off", "no System 1 configured (SYSTEM1=off)")
        self._s1_event = False                       # an arrival, a look or a scene change: observe now
        self.s1_gate: Any = None                     # which head-camera frames System 1 gets

    # ------------------------------------------------------------------ cameras
    def _cameras_for(self, s: Session) -> CameraSource:
        """The session's panes: the fixed source if one was given, the robot factory's frame source
        if it has the CameraSource calls, world frames on lite, else viz's FrameTap."""
        if self._fixed_cameras is not None:
            return self._fixed_cameras
        mode = self._camera_mode
        f = s.frames
        if mode in ("auto", "frames") and f is not None and all(callable(getattr(f, n, None)) for n in ("names", "rev", "jpeg", "meta")):
            return f
        if mode == "none":
            return NoCameras()
        if mode == "world" or (mode == "auto" and s.profile == "lite"):
            return WorldCameras(f if f is not None and callable(getattr(f, "latest_frame", None)) else s.world)
        if self._tap is None:
            try:
                self._tap = TapCameras(port_offset=self._port_offset)
            except Exception as e:  # noqa: BLE001
                self._notice(f"No camera feed: viz FrameTap could not start ({e}).")
                return WorldCameras(s.world)
        return self._tap

    # ------------------------------------------------------------------ System 1
    def start_system1(self) -> None:
        """Load the System 1 plug-in, if one is configured, and keep it running."""
        if not self.system1_spec and self.deps.system1 is None:
            return
        try:
            factory = self.deps.system1 or _import(self.system1_spec)
            self.system1 = factory(self._on_s1_status)
        except Exception as e:  # noqa: BLE001
            self._on_s1_status("error", f"could not load {self.system1_spec}: {e}")
            return
        asyncio.create_task(self.system1.run())
        asyncio.create_task(self._system1_feed())

    def _on_s1_status(self, status: str, detail: str = "") -> None:
        self.s1_status, self.s1_detail = status, detail
        broadcast(self.clients, dumps({"type": "system1", "status": status, "detail": detail}))

    def _s1_ready(self) -> bool:
        return self.system1 is not None and getattr(self.system1, "status", "") == "ready"

    async def _system1_feed(self) -> None:
        """Keep System 1 current: the fused state when it changes, a head-camera frame only
        when the frame gate says it shows something new (a changed scene or a new view, at
        most one a second), and a request for observations every few seconds while there are
        new frames, or at once after a scene change, an arrival or a look."""
        last_ctx, last_ctx_t, last_rev, last_obs_t, last_own_t = None, 0.0, -1, 0.0, -1e9
        new_frames, observing = 0, None
        while True:
            await asyncio.sleep(0.25)
            s, s1 = self.session, self.system1
            if not self._s1_ready() or s is None or s.runtime is None:
                continue
            now = time.monotonic()
            try:
                ctx = s.runtime.system1_context()
                if ctx != last_ctx and now - last_ctx_t >= 1.0:
                    await s1.update(ctx)
                    last_ctx, last_ctx_t = ctx, now
                src = self.feed.source
                if "head" not in src.names():
                    continue
                rev = src.rev("head")
                if rev != last_rev:
                    running = [str(t) for t in (ctx.get("running") or [])]
                    if any(t in ("manipulate", "pick", "place") or t.startswith("manipulate") for t in running):
                        last_own_t = now
                    own = now - last_own_t < S1_OWN_GRACE_S     # and the check right after it
                    jpeg = src.jpeg("head")
                    rgb = _decode_rgb(jpeg) if jpeg else None
                    meta = src.meta("head")
                    moving = bool(ctx.get("moving") or ctx.get("between"))
                    stationary = meta.get("stationary")
                    if stationary is None:
                        stationary = not moving
                    if self.s1_gate is None:
                        self.s1_gate = self.deps.frame_gate()
                    if rgb is None:                            # no decoder: send at most one frame a second
                        send, reason = now - getattr(self, "_s1_last_send", 0.0) >= 1.0, "no decoder"
                    else:
                        d = _gate_decide(self.s1_gate, rgb, s._robot_pose(), now, own, stationary)
                        send, reason = d.send, d.reason
                    if reason != "too soon":                   # too soon: look at this frame again next time
                        last_rev = rev
                    if send and jpeg:
                        self._s1_last_send = now
                        # belief keeps the last keypoint while walking; a frame taken on the way
                        # isn't "at" it, or observations get filed where the robot started
                        await s1.frame(jpeg, None if moving else ctx.get("at"))
                        new_frames += 1
                        if reason == "scene changed":          # something changed in front of it: ask now
                            self._s1_event = True
                due = now - last_obs_t >= S1_OBSERVE_EVERY_S or self._s1_event
                if new_frames and due and (observing is None or observing.done()):
                    self._s1_event, last_obs_t, n = False, now, new_frames
                    new_frames = 0
                    observing = asyncio.create_task(self._observe(s, n))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                await asyncio.sleep(2.0)

    async def _observe(self, s: "Session", frames: int) -> None:
        t0 = time.monotonic()
        try:
            items = await self.system1.observe()
        except Exception:  # noqa: BLE001
            return
        if s is self.session and s.runtime is not None:
            s.runtime.add_observations(list(items or []), source="system1")
            describe = getattr(self.system1, "describe", None)
            session = f"; {describe()}" if callable(describe) else ""
            gate = self.s1_gate.summary() if self.s1_gate is not None and hasattr(self.s1_gate, "summary") else "-"
            s.record_system1("observe", f"OBSERVE ({frames} new frame{'s' if frames != 1 else ''} since the last one; "
                                        f"frames so far: {gate}{session})",
                             {"items": list(items or [])}, round(time.monotonic() - t0, 2))

    async def _route(self, s: "Session", text: str) -> dict[str, Any] | None:
        """System 1's label for a message, as the directive the runtime reads, or None."""
        if not self._s1_ready():
            return None
        t0 = time.monotonic()
        try:
            r = await asyncio.wait_for(self.system1.route(text), S1_ROUTE_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            r = None
        latency = round(time.monotonic() - t0, 2)
        s.record_system1("route", f"USER: {text}", r, latency)
        if not r or not r.get("kind"):
            return None
        reply = {True: "yes", False: "no"}.get(r.get("says_yes")) if isinstance(r.get("says_yes"), bool) else None
        return {"text": text, "kind": r["kind"], "source": "system1", "confidence": r.get("confidence", 0.0),
                "target": {"object": r["target"]} if r.get("target") else {}, "reply": reply,
                "replaces_task": bool(r.get("replaces_task")), "latency_s": latency}

    def meta(self) -> dict[str, Any]:
        return {"scenes": SCENES, "agents": AGENTS, "models": MODELS, "profiles": PROFILES,
                "profile": self.session.profile if self.session else self.profile,
                "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
                "system1": {"status": self.s1_status, "detail": self.s1_detail}}

    # ------------------------------------------------------------------
    def _notice(self, text: str) -> None:
        broadcast(self.clients, dumps({"type": "notice", "text": text}))

    # ------------------------------------------------------------------
    async def reset(self, scene: str, agent: str, model: str, forget: bool = False, profile: str | None = None) -> None:
        async with self._lock:
            profile = profile or self.profile
            old, self.session = self.session, None
            if old:
                await old.stop()              # the old runtime saves its memory as it stops...
            if forget:                        # ...so forgetting has to come after that
                (ROOT / "runs" / "memory" / f"{scene}.json").unlink(missing_ok=True)
                self._notice(f"Forgot everything about {scene}: no seen objects, landmarks, looks or notes.")
            broadcast(self.clients, dumps({"type": "loading", "scene": scene, "profile": profile}))
            new = Session(scene, agent, model, profile, self.deps)
            try:
                await new.start()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                self._notice("Could not load this house:\n" + traceback.format_exc()[-800:])
                return
            self.session, self.profile = new, profile
            if self.s1_gate is not None:
                self.s1_gate.reset()                  # a new house: the first frame is new again
            if callable(getattr(new.runtime, "set_persona", None)):
                new.runtime.set_persona(self.persona_level)
            if callable(getattr(new.runtime, "set_step_mode", None)):
                new.runtime.set_step_mode(self.step_mode)
            self.feed = CameraFeed(self._cameras_for(new))
            broadcast(self.clients, dumps({**new.init_message(), "meta": self.meta()}))
            broadcast(self.clients, dumps(new.memory_message()))

    async def tick(self) -> None:
        s = self.session
        if s is None or not self.clients:
            return
        frame = s.frame()
        frame["system1"] = {"status": self.s1_status, "detail": self.s1_detail}
        broadcast(self.clients, dumps(frame))
        if self._s1_ready():
            for e in frame["events"]:
                if e.get("type") == "speech_started":
                    asyncio.create_task(self.system1.robot_said(e.get("text", "")))
            if any(r.get("type") == "result" and (r.get("tool") or r.get("skill")) in OBSERVE_TOOLS
                   and str(r.get("status", "")).lower() == "succeeded" for r in frame["trace"]):
                self._s1_event = True
        if any(r.get("type") in MEMORY_ROWS for r in frame["trace"]):
            self._memory_due = True
        if self._memory_due and time.monotonic() - self._memory_sent > 1.0:
            self._memory_due, self._memory_sent = False, time.monotonic()
            broadcast(self.clients, dumps(s.memory_message()))
        self._send_cameras()

    async def ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def _send_cameras(self, force: bool = False, ws: Any = None) -> None:
        for m in self.feed.due(force=force):
            msg = dumps({**m, "jpeg": base64.b64encode(m["jpeg"]).decode()})
            if ws is not None:
                asyncio.create_task(ws.send(msg))
            else:
                broadcast(self.clients, msg)

    # ------------------------------------------------------------------
    async def handler(self, ws: Any) -> None:
        self.clients.add(ws)
        try:
            if self.session is not None:
                await ws.send(dumps({**self.session.init_message(), "meta": self.meta()}))
                await ws.send(dumps(self.session.memory_message()))
                self._send_cameras(force=True, ws=ws)
            else:
                await ws.send(dumps({"type": "loading", "scene": self.default, "profile": self.profile,
                                     "meta": self.meta()}))
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                await self._on_message(ws, msg)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def _on_message(self, ws: Any, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        s = self.session
        try:
            if kind == "say" and s is not None:
                text = str(msg.get("text", "")).strip()
                if not text:
                    return
                if self.deps.is_stop(text) and s.runtime is not None:
                    s.runtime.emergency_stop(text, source="keyword")   # halt() first; System 1 labels it next
                s.hear(text, await self._route(s, text))            # every message goes through System 1
            elif kind == "hello" and s is not None:          # the page missed the first init
                await ws.send(dumps({**s.init_message(), "meta": self.meta()}))
                self._send_cameras(force=True, ws=ws)
            elif kind == "persona":                            # off, quiet, medium, optimize
                level = str(msg.get("level") or "medium")
                if level not in ("off", "quiet", "medium", "optimize"):
                    raise ValueError(f"unknown persona level {level!r}")
                self.persona_level = level
                if s is not None and callable(getattr(s.runtime, "set_persona", None)):
                    s.runtime.set_persona(level)
            elif kind == "step":                              # on, off, next
                mode = str(msg.get("mode") or "next")
                if mode in ("on", "off"):
                    self.step_mode = mode == "on"
                if s is not None and s.runtime is not None and callable(getattr(s.runtime, "step", None)):
                    if mode == "next":
                        s.runtime.step()
                    else:
                        s.runtime.set_step_mode(self.step_mode)
            elif kind == "estop" and s is not None:          # the operator kill button (PLAN 9.4): never the chat stop
                if not msg.get("confirm"):
                    raise ValueError("the kill button needs confirm: true")
                fn = getattr(s.robot, "estop", None)
                if not callable(fn):
                    raise ValueError("this robot has no estop()")
                res = fn("operator kill button")
                if inspect.isawaitable(res):
                    res = await res
                s.log.emit("safety_stop", reason="operator kill button (estop)", receipt=to_plain(res))
                self._notice("Controller killed (estop). The deploy exits; a supervised restart is needed.")
            elif kind == "memory" and s is not None:
                await ws.send(dumps(s.memory_message()))
            elif kind == "relearn" and s is not None and getattr(s.runtime, "procedures", None) is not None:
                from agent.episodes import load_episodes
                g = s.runtime.procedures                     # recount from every episode on disk; rules are kept
                g.learn(await asyncio.to_thread(load_episodes))
                g.save()
                broadcast(self.clients, dumps(s.memory_message()))
            elif kind == "reset":
                profile = msg.get("profile") or None
                if profile is not None and profile not in {p for p, _ in PROFILES}:
                    raise ValueError(f"unknown profile {profile!r}")
                await self.reset(msg.get("scene") or self.default, msg.get("agent") or "agent",
                                 msg.get("model") or MODELS[0][0], forget=bool(msg.get("forget")), profile=profile)
        except Exception as e:  # noqa: BLE001
            await ws.send(dumps({"type": "notice", "text": str(e)}))

    def close(self) -> None:
        for src in (self._tap, self._fixed_cameras):
            if src is not None:
                try:
                    src.close()
                except Exception:  # noqa: BLE001
                    pass


def process_request(connection: Any, request: Any) -> Any:
    path = request.path.split("?")[0]
    if path == "/ws":
        return None
    if path in ("/", "/index.html"):
        resp = connection.respond(http.HTTPStatus.OK, INDEX.read_text())
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    m = re.fullmatch(r"/views/([a-z_]+\.js)", path)            # the page's view scripts, nothing else
    if m and (VIEWS / m.group(1)).is_file():
        resp = connection.respond(http.HTTPStatus.OK, (VIEWS / m.group(1)).read_text())
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/javascript; charset=utf-8"
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")


async def serve_hub(hub: Hub, host: str, port: int) -> Any:
    """Bind the page's HTTP + websocket server (max message 4 MB, as before)."""
    return await serve(hub.handler, host, port, process_request=process_request, max_size=2 ** 22)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scene", default="procthor-train-40", help="a MolmoSpaces house id (procthor-train-40, ithor-FloorPlan10)")
    ap.add_argument("--profile", default=os.environ.get("WL_PROFILE", "lite"), choices=[p for p, _ in PROFILES])
    ap.add_argument("--cameras", default="auto", choices=["auto", "tap", "world", "frames", "none"],
                    help="auto: the factory's frame source, world frames on lite, else viz FrameTap (5565/5602)")
    ap.add_argument("--port-offset", type=int, default=None, help="shift the FrameTap ports (WL_PORT_OFFSET)")
    args = ap.parse_args()
    load_env_files()
    hub = Hub(args.scene, args.profile, cameras=args.cameras, port_offset=args.port_offset)
    hub.deps.resolve()
    try:
        try:
            server = await serve_hub(hub, args.host, args.port)
        except OSError as e:
            print(f"Port {args.port} is in use ({e.strerror}): a playground server is probably already running "
                  f"at http://{args.host}:{args.port}. Stop it first.", file=sys.stderr)
            return
        hub.start_system1()
        async with server:
            print(f"Worldline on http://{args.host}:{args.port}  (profile {args.profile}, loading {args.scene}…)", flush=True)
            await hub.reset(args.scene, "agent", MODELS[0][0], profile=args.profile)
            if hub.session and hub.session.error:
                print(hub.session.error, file=sys.stderr)
            print("ready", flush=True)
            await hub.ticker()
    finally:
        if hub.session is not None:
            try:
                await asyncio.wait_for(hub.session.stop(), 8)   # the runtime saves memory as it stops
            except Exception:  # noqa: BLE001
                pass
        hub.close()
        print("stopped", flush=True)


def _on_signal(signum: int, frame: Any) -> None:
    # `kill` or a closed terminal should shut down like Ctrl-C, so main()'s finally
    # stops the session (memory saved, body left in HOLD by the robot facade).
    raise KeyboardInterrupt


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
