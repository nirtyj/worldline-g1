"""G1 stack scenarios G1-G14 (PLAN 9.3) and the E5 GR00T plumbing run (docs/M2.md 7.5), scored on what the page shows.

    .venv-rt/bin/python -m eval.stack_suite                                   # offline: the lite subset, in process
    .venv-rt/bin/python -m eval.stack_suite --only G2,G14 --speed 5
    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8765/ws --profile sonic \\
        --hook kill_policy='ssh ... pkill -f run_gr00t_server' --hook restore_policy='...'    # wave 2, a live P5
    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8765/ws --profile full --only E5

Two ways to run, one referee:

  offline (no --url)  starts the page server in this process on `lite` (ui/server.py Hub, the real runtime and the
                      lite stack), with the scripted planner and the System 1 stub unless --planner/--system1 say
                      otherwise. Faults are injected into the in-process objects (the lite body's `inject`, the skill
                      registry's health): see LocalInjector. Every such injection is labelled `injected` in the result.
  --url               referees a running page server (the box's P5 on sonic/full, through 00_infra/tunnel.sh 8765).
                      Faults the page cannot cause are injected by shell hooks (--hook name=cmd): kill_policy,
                      restore_policy, spawn_box, clear_box, push_robot, kill_deploy, throttle_rtf, restore_rtf,
                      delay_proxy_on, delay_proxy_off, place_object. A scenario whose hook is missing is SKIPPED.
                      A hook command may use the injection's arguments as {placeholders} (push_robot {newtons},
                      throttle_rtf {rtf}, delay_proxy_on {ms}, place_object {object} {offset_m}); write literal
                      braces as {{ }}. Hooks run on the laptop, e.g. an ssh to the box.

Scoring reads only the page's messages (frame.truth from world.truth(), trace rows, events, model calls), exactly like
eval/suite.py; the injectors are fixtures, never a source of verdicts. Each scenario reports its criteria one by one:
true, false, `None` (not observable on this run: the verdict is UNVERIFIED, never a pass) or "n/a" (does not apply
on this profile, e.g. "after one replan" on the lite body, which never replans). A pass that rests on a STEPPING STONE
executor is PASS* (fallback), as in eval/suite.py. Every GR00T result is labelled experimental.

PLAN 2.3 bars: lite subset (G2 G5 G6 G11 G12 G13 G14) >= 7/7, sonic >= 12/14, full >= 13/14.
E5 (full): 10 x manipulate(pick) through groot_arms with the off-the-shelf N1.7 checkpoint: 0 falls, cancel 3/3,
halt 3/3, every result names groot_arms, the skill and the GT outcome. Success is reported, not required.

Writes <out>.json (default runs/eval/<stamp>_stack_<profile>.json) and, with --trace-dir, each scenario's trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scenes as sc  # noqa: E402
from eval import suite  # noqa: E402

NA = "n/a"
LITE_SUBSET = ("G2", "G5", "G6", "G11", "G12", "G13", "G14")
BARS = {"lite": (7, 7), "bringup": (9, None), "sonic": (12, 14), "full": (13, 14)}
H40 = suite.H40
STOP_ACK = re.compile(r"\bstop", re.I)
TOLD_BLOCKED = re.compile(r"block|can't get|cannot get|can not get|couldn't get|no way|in the way|obstacle|stuck|"
                          r"path|couldn't reach|can't reach", re.I)
GROOT_EXECUTORS = ("groot_arms", "groot_sonic")


# ----------------------------------------------------------------------------------------------------------
# What a run sees
# ----------------------------------------------------------------------------------------------------------
class StackRun(suite.Run):
    """eval/suite.py's Run plus every event and a thin copy of every frame (sim time, the robot's true pose and
    fall state, the body mode, the believed hands, where tracked objects are, the GR00T strip)."""

    def __init__(self, ws: Any, profile: str = "lite", scale: float | None = None,
                 injector: "Injector | None" = None) -> None:
        super().__init__(ws, profile, scale)
        self.events: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.injector = injector or Injector()
        self.track: set[str] = set()
        self.injected: list[str] = []

    def ingest(self, m: dict[str, Any]) -> None:
        if m.get("type") == "init":
            self.events, self.frames = [], []
        super().ingest(m)
        if m.get("type") in ("init", "frame"):
            self.events += list(m.get("events") or [])
            tr = m.get("truth") or {}
            rt = m.get("runtime") or {}
            robot = tr.get("robot") or {}
            self.frames.append({
                "t": float(m.get("t") or 0.0),
                "x": robot.get("x"), "z": robot.get("z"), "yaw": robot.get("yaw"),
                "fallen": robot.get("fallen"), "upright": robot.get("upright"),
                "mode": (m.get("body") or {}).get("mode") or robot.get("mode"),
                "hands": ((rt.get("belief") or {}).get("hands") or {}),
                "where": {oid: ((tr.get("objects") or {}).get(oid) or {}).get("where") for oid in self.track},
                "tool_state": rt.get("tool_state"), "paused": rt.get("paused"),
                "groot": m.get("groot"), "stack": m.get("stack") or {},
            })

    # -- queries --------------------------------------------------------
    def results(self, tool: str | None = None, **match: Any) -> list[dict[str, Any]]:
        out = [r for r in self.trace if r.get("type") == "result" and (tool is None or r.get("tool") == tool)]
        for k, v in match.items():
            out = [r for r in out if (r.get(k) if k in r else (r.get("data") or {}).get(k)) == v]
        return out

    def row_for(self, eid: str | None, kind: str = "result") -> dict[str, Any] | None:
        return next((r for r in self.trace if r.get("type") == kind and r.get("execution_id") == eid), None)

    def started_rows(self, tool: str, action: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.rows("started", tool=tool) if action is None or r.get("action") == action
                or (r.get("args") or {}).get("action") == action]

    def speeds(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """(t, planar speed m/s) from the true pose in consecutive frames between t0 and t1 (sim time)."""
        fr = [f for f in self.frames if f["x"] is not None and t0 - 0.5 <= f["t"] <= t1]
        out = []
        for a, b in zip(fr, fr[1:]):
            dt = b["t"] - a["t"]
            if dt > 1e-3:
                out.append((b["t"], math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / dt))
        return out

    def fell(self, t0: float = -1e9) -> bool:
        return (any(f["t"] >= t0 and (f["fallen"] or f["upright"] is False) for f in self.frames)
                or any(r.get("kind") == "fell" for r in self.rows("safety_event") if (r.get("t") or 0) >= t0))

    def said_after(self, t: float, pattern: re.Pattern[str] | str) -> list[str]:
        pat = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
        return [text for st, text in self.said if st >= t and pat.search(text)]

    def model_inputs(self) -> list[str]:
        return [str(c.get("input") or "") for c in self.calls.values() if c.get("via") == "model"]

    def tool_schemas(self) -> list[Any]:
        return [c.get("tool_schemas") for c in self.calls.values()
                if c.get("via") == "model" and c.get("purpose") == "next_action" and c.get("tool_schemas")]

    async def wait_row(self, pred: Callable[[dict[str, Any]], bool], timeout: float, start: int = 0) -> dict[str, Any] | None:
        box: list[dict[str, Any]] = []

        def found() -> bool:
            for r in self.trace[start:]:
                if pred(r):
                    box.append(r)
                    return True
            return False
        await self.until(found, timeout)
        return box[0] if box else None

    async def wait_sim(self, seconds: float, timeout: float = 60.0) -> None:
        t0 = self.now()
        await self.until(lambda: self.now() - t0 >= seconds, timeout)

    async def wait_event(self, pred: Callable[[dict[str, Any]], bool], timeout: float) -> dict[str, Any] | None:
        box: list[dict[str, Any]] = []

        def found() -> bool:
            for e in self.events:
                if pred(e):
                    box.append(e)
                    return True
            return False
        await self.until(found, timeout)
        return box[0] if box else None

    async def inject(self, name: str, **kw: Any) -> tuple[bool, str]:
        ok, note = await self.injector.do(name, **kw)
        self.injected.append(f"{name}: {note}" if ok else f"{name} FAILED: {note}")
        return ok, note


# ----------------------------------------------------------------------------------------------------------
# Fault injection (fixtures only; scoring never reads from here)
# ----------------------------------------------------------------------------------------------------------
class Injector:
    """No injection available: every scenario that needs one is SKIPPED."""

    def has(self, name: str) -> bool:
        return False

    async def do(self, name: str, **kw: Any) -> tuple[bool, str]:
        return False, f"no injector for {name}"


class HookInjector(Injector):
    """Shell hooks for a live stack (--hook name=cmd): the command runs on the laptop (e.g. an ssh to the box)."""

    def __init__(self, hooks: dict[str, str]) -> None:
        self.hooks = dict(hooks)

    def has(self, name: str) -> bool:
        return name in self.hooks

    async def do(self, name: str, **kw: Any) -> tuple[bool, str]:
        cmd = self.hooks.get(name)
        if not cmd:
            return False, f"no --hook {name}=..."
        cmd = cmd.format(**{k: shlex.quote(str(v)) for k, v in kw.items()})
        p = await asyncio.to_thread(subprocess.run, cmd, shell=True, capture_output=True, text=True, timeout=120)
        tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
        return p.returncode == 0, f"hook exit {p.returncode}: {tail[0][:160]}"


class LocalInjector(Injector):
    """The offline (lite, in-process) stand-ins for the box's faults. Each is labelled `injected` in the result:
      stuck        LiteBody.inject(stuck_after_s=...): the walk stops making progress -> failed(blocked)
                   (the Isaac variant spawns a box in a corridor through WorldRPC)
      policy_down  the skill registry reports every skill unhealthy ("injected: policy server down"); new manipulate
                   calls must be rejected at CAPABILITY. policy_up undoes it.
      clear        LiteBody.clear_fault() and no stuck injection"""

    NAMES = ("stuck", "clear", "policy_down", "policy_up")

    def __init__(self, hub: Any) -> None:
        self.hub = hub
        self._saved: Any = None

    def has(self, name: str) -> bool:
        return name in self.NAMES

    def _robot(self) -> Any:
        s = self.hub.session
        return getattr(s, "robot", None) if s is not None else None

    async def do(self, name: str, **kw: Any) -> tuple[bool, str]:
        robot = self._robot()
        if robot is None:
            return False, "no session"
        body = getattr(robot, "body", None)
        if name == "stuck":
            if not callable(getattr(body, "inject", None)):
                return False, "the body has no inject()"
            body.inject(stuck_after_s=float(kw.get("after_s", 3.0)))
            return True, f"injected: lite body stuck {kw.get('after_s', 3.0)} s into the next walk"
        if name == "clear":
            if callable(getattr(body, "inject", None)):
                body.inject(stuck_after_s=None)
            if callable(getattr(body, "clear_fault", None)):
                body.clear_fault()
            return True, "cleared"
        reg = getattr(robot, "skill_registry", None)
        if name == "policy_down":
            if reg is None or not hasattr(reg, "_health_fn"):
                return False, "no skill registry health hook"
            from api.types import ServiceHealth
            self._saved = reg._health_fn
            reg._health_fn = lambda s: ServiceHealth(False, "down", "injected: policy server down")
            return True, "injected: every skill unhealthy (policy server down)"
        if name == "policy_up":
            if reg is not None and hasattr(reg, "_health_fn"):
                reg._health_fn = self._saved
            return True, "restored"
        return False, f"unknown injection {name}"

    def schemas(self) -> Any:
        """The session's tool schemas as the planner gets them (the runtime's frozen SchemaContext)."""
        rt = getattr(self.hub.session, "runtime", None)
        ctx = getattr(rt, "tools_ctx", None)
        if ctx is None:
            return None
        from api.tools import json_schemas
        return json_schemas(ctx)

    def planner_contexts(self) -> list[Any]:
        return list(getattr(getattr(self.hub.session, "planner", None), "contexts", None) or [])


# ----------------------------------------------------------------------------------------------------------
# Scenario plumbing
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Outcome:
    criteria: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: Any, detail: Any = None) -> Any:
        self.criteria.append({"name": name, "ok": ok, "detail": detail})
        return ok

    def verdict(self) -> str:
        oks = [c["ok"] for c in self.criteria if c["ok"] != NA]
        if not oks:
            return "UNVERIFIED"
        if any(o is False for o in oks):
            return "FAIL"
        if any(o is None for o in oks):
            return "UNVERIFIED"
        return "PASS"


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    profiles: tuple[str, ...]
    fn: Callable[[StackRun, Outcome], Awaitable[None]]
    needs: tuple[str, ...] = ()          # injections the scenario cannot run without, per mode ("local"/"hook")
    lite: str = "native"                 # native | injected | none


def _needs(spec: Spec, run: StackRun, local: bool) -> list[str]:
    names = [n for n in spec.needs if (n.startswith("local:") and local) or (n.startswith("hook:") and not local)]
    return [n.split(":", 1)[1] for n in names if not run.injector.has(n.split(":", 1)[1])]


async def _fetch_start(r: StackRun, scenario: str = "fetch_other_room", oid: str = "alarm_clock_1",
                       label: str = "alarm clock", forget: bool = True,
                       before: Callable[[], Awaitable[Any]] | None = None) -> float:
    """Load the fetch scenario's house (H40), run `before` (an injection into the new session), and ask for the
    object; returns the sim time the request was said."""
    r.current = scenario
    b = suite.BIND.scenario(scenario)
    r.track = {oid, "mug_1"}
    await r.load(suite.BIND.scene(b["house"]), forget=forget)
    if before is not None:
        await before()
    t = r.now()
    await r.say(f"Bring me the {label}.")
    return t


def _res_of(r: StackRun, started: dict[str, Any] | None) -> dict[str, Any] | None:
    return r.row_for(started.get("execution_id")) if started else None


# ----------------------------------------------------------------------------------------------------------
# G1-G14
# ----------------------------------------------------------------------------------------------------------
async def g1_halt_midstride(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    nav = await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "navigate", r.wait(60))
    if nav is None:
        o.check("a walk started", False)
        return
    await r.wait_sim(3.0, r.wait(30))
    t_stop = r.now()
    await r.say("stop")
    stop = await r.wait_row(lambda x: x.get("type") == "stop", 10)
    await r.wait_sim(2.0, 20)
    rec = (stop or {}).get("receipt") or {}
    lat = rec.get("latency_ms")
    o.check("receipt stopped within 30 ms", bool(rec.get("stopped")) and lat is not None and float(lat) <= 30.0,
            {"stopped": rec.get("stopped"), "latency_ms": lat, "source": rec.get("source")})
    sp = r.speeds(t_stop, t_stop + 2.0)
    below = next((t for t, v in sp if v < 0.1), None)
    rest = next((t for t, v in sp if v < 0.05), None)
    o.check("speed < 0.1 m/s within 1.2 s", None if not sp else (below is not None and below - t_stop <= 1.2),
            {"t": None if below is None else round(below - t_stop, 2)})
    o.check("at rest (< 0.05 m/s) within 1.5 s", None if not sp else (rest is not None and rest - t_stop <= 1.5),
            {"t": None if rest is None else round(rest - t_stop, 2), "receipt_at_rest": rec.get("at_rest")})
    o.check("no fall", not r.fell(t_stop - 5))
    acks = r.said_after(t_stop, STOP_ACK)
    o.check("one acknowledgement", len(acks) == 1, acks)


async def g2_walk_cancel(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    nav = await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "navigate", r.wait(60))
    if nav is None:
        o.check("a walk started", False)
        return
    await r.wait_sim(2.0, r.wait(20))
    t_c = r.now()
    await r.say("No, bring me the mug instead.")
    res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("execution_id") == nav.get("execution_id"),
                           r.wait(30))
    d = (res or {}).get("data") or {}
    o.check("a correction bumped the generation", bool(r.rows("correction")),
            [(x.get("version"), x.get("epoch")) for x in r.rows("correction")])
    o.check("the walk ended cancelled", (res or {}).get("status") == "cancelled",
            {"status": (res or {}).get("status"), "reason": d.get("reason")})
    o.check("with between (or at a keypoint)", bool(d.get("between") or d.get("at")),
            {"between": d.get("between"), "at": d.get("at")})
    settle = d.get("settle_s")
    if settle is None and res is not None:
        sp = r.speeds(t_c, float(res.get("t") or t_c) + 2.0)
        rest = next((t for t, v in sp if t >= float(res.get("t") or t_c) and v < 0.05), None)
        settle = None if rest is None else round(rest - float(res.get("t") or t_c), 2)
    o.check("settle <= 1.5 s", None if settle is None else float(settle) <= 1.5, {"settle_s": settle})
    o.check("no fall", not r.fell(t_c - 5))


async def _groot_pick_started(r: StackRun, timeout: float) -> dict[str, Any] | None:
    return await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "manipulate"
                            and (x.get("args") or {}).get("action", x.get("action")) == "pick", timeout)


def _chunks_after(r: StackRun, session: str | None, t: float) -> tuple[int | None, int]:
    """(chunk-bearing arm.progress events for this session after t, or None when the session never reported any
    chunk telemetry; all such events of the session)."""
    evs = [e for e in r.events if e.get("type") in ("arm.progress", "arm_progress", "arm.chunk")
           and (e.get("session") or e.get("execution_id")) == session]
    if not evs:
        return None, 0
    before = [e for e in evs if float(e.get("t") or 0) <= t]
    last_idx = max((e.get("chunk_idx") or -1) for e in before) if before else -1
    after = [e for e in evs if float(e.get("t") or 0) > t and (e.get("chunk_idx") or -1) > last_idx]
    return len(after), len(evs)


def _hold_within(r: StackRun, t: float, limit: float) -> Any:
    modes = [f for f in r.frames if f["t"] >= t and f["mode"]]
    if not modes:
        return None
    hold = next((f["t"] for f in modes if f["mode"] == "HOLD"), None)
    return hold is not None and hold - t <= limit


async def g3_grasp_cancel(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    pick = await _groot_pick_started(r, r.wait(240))
    if pick is None:
        o.check("a pick started", False)
        return
    await r.wait_sim(2.0, 30)
    t_c = r.now()
    await r.say("No, bring me the mug instead.")
    res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("execution_id") == pick.get("execution_id"),
                           r.wait(20))
    d = (res or {}).get("data") or {}
    o.check("the pick ended cancelled", (res or {}).get("status") == "cancelled", {"status": (res or {}).get("status")})
    o.check("the pick ran on GR00T", d.get("executor") in GROOT_EXECUTORS, {"executor": d.get("executor"),
                                                                           "skill": d.get("skill")})
    t_ack = float((res or {}).get("t") or t_c)
    after, n = _chunks_after(r, pick.get("execution_id"), t_ack)
    o.check("zero chunks after the cancel ack", None if after is None else after == 0, {"after": after, "events": n})
    o.check("HOLD within 1.3 s", _hold_within(r, t_ack, 1.3))
    o.check("upright", not r.fell(t_c - 2))
    arm = d.get("arm") or (pick.get("args") or {}).get("arm")
    unknown = any(str(((f["hands"] or {}).get(arm) or {}).get("holding")) == "UNKNOWN" for f in r.frames
                  if f["t"] >= t_c) if arm else None
    o.check("hand UNKNOWN in belief", unknown, {"arm": arm})
    o.check("reconcile scan", bool([x for x in r.rows("reconcile_start") if (x.get("t") or 0) >= t_c]))


async def g4_stale_chunk(r: StackRun, o: Outcome) -> None:
    ok, note = await r.inject("delay_proxy_on", ms=600)
    o.notes.append(f"injected: delay proxy 600 ms ({note})")
    try:
        await g3_grasp_cancel(r, o)
        stale = [e for e in r.events if "stale" in str(e.get("reason") or e.get("type") or "")]
        o.extra["stale_events"] = len(stale)
    finally:
        await r.inject("delay_proxy_off")


async def g5_policy_down(r: StackRun, o: Outcome) -> None:
    local = isinstance(r.injector, LocalInjector)
    box: dict[str, Any] = {}

    async def down() -> None:
        box["schemas"] = r.injector.schemas() if local else None
        box["ok"], box["note"] = await r.inject("policy_down" if local else "kill_policy")
        o.notes.append(box["note"])
    try:
        t0 = await _fetch_start(r, before=down)
        schemas0 = box.get("schemas")
        if not box.get("ok"):
            o.check("policy taken down", False, box.get("note"))
            return
        rej = await r.wait_row(lambda x: (x.get("type") == "rejected" and "policy unavailable" in str(x.get("why")))
                               or (x.get("type") == "result" and x.get("tool") == "manipulate"
                                   and "fallback" in str(x.get("summary") or "")), r.limit())
        o.check("new call rejected with 'policy unavailable' (or a labelled fallback)", rej is not None,
                None if rej is None else {"type": rej.get("type"), "stage": rej.get("stage"), "why": rej.get("why"),
                                          "summary": rej.get("summary")})
        if rej is not None and rej.get("type") == "rejected":
            o.check("rejected at the CAPABILITY stage", rej.get("stage") == "capability", rej.get("stage"))
        await r.until(r.idle, r.wait(60))
        if local:
            o.check("tool schemas unchanged by the outage (Invariant 9)", r.injector.schemas() == schemas0)
        else:
            sch = r.tool_schemas()
            o.check("tool schemas unchanged by the outage (Invariant 9)",
                    None if len(sch) < 2 else all(s == sch[0] for s in sch), {"model calls": len(sch)})
        o.check("no fall", not r.fell(t0))
    finally:
        await r.inject("policy_up" if local else "restore_policy")
    if local:
        o.check("a running call fails policy_unavailable within 3 s, then HOLD", NA,
                "lite: the lite executor has no policy to lose mid-call (full only)")
        return
    # the running-call half (full): a GR00T pick is running when P4 dies
    t0 = await _fetch_start(r, forget=False)
    pick = await _groot_pick_started(r, r.wait(240))
    if pick is None:
        o.check("a running GR00T pick", False)
        return
    await r.wait_sim(1.5, 20)
    t_kill = r.now()
    await r.inject("kill_policy")
    try:
        res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("execution_id") == pick.get("execution_id"),
                               r.wait(30))
        d = (res or {}).get("data") or {}
        took = None if res is None else float(res.get("t") or 0) - t_kill
        o.check("running call failed(policy_unavailable) within 3 s",
                (res or {}).get("status") == "failed" and d.get("reason") == "policy_unavailable" and took is not None
                and took <= 3.0, {"status": (res or {}).get("status"), "reason": d.get("reason"), "took_s": took})
        o.check("then HOLD", _hold_within(r, float((res or {}).get("t") or t_kill), 1.5))
    finally:
        await r.inject("restore_policy")


async def g6_blocked_path(r: StackRun, o: Outcome) -> None:
    local = isinstance(r.injector, LocalInjector)
    box: dict[str, Any] = {}

    async def block() -> None:
        box["ok"], box["note"] = await (r.inject("stuck", after_s=3.0) if local else r.inject("spawn_box"))
        o.notes.append(box["note"])
    try:
        t0 = await _fetch_start(r, before=block)
        if not box.get("ok"):
            o.check("obstacle injected", False, box.get("note"))
            return
        res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("tool") == "navigate"
                               and (x.get("data") or {}).get("reason") == "blocked", r.limit())
        d = (res or {}).get("data") or {}
        o.check("navigate failed: blocked", res is not None, None if res is None else res.get("summary"))
        o.check("with blocked_edge", bool(d.get("blocked_edge")) and len(d.get("blocked_edge") or []) == 2,
                d.get("blocked_edge"))
        if local:
            o.check("after one replan", NA, "lite body: no replanning; stuck is injected directly")
        else:
            o.check("after one replan", None if d.get("replans") is None else int(d["replans"]) >= 1,
                    {"replans": d.get("replans")})
        if res is not None:
            await r.until(lambda: bool(r.said_after(float(res.get("t") or t0), TOLD_BLOCKED)), r.wait(60))
        told = r.said_after(float((res or {}).get("t") or t0), TOLD_BLOCKED) if res else []
        o.check("tells the user", bool(told), told[:2] or [t for _, t in r.said][-3:])
        o.check("no fall", not r.fell(t0))
    finally:
        await r.inject("clear" if local else "clear_box")


async def g7_fall_recovery(r: StackRun, o: Outcome) -> None:
    t0 = await _fetch_start(r)
    await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "navigate", r.wait(60))
    await r.wait_sim(2.0, 30)
    t_push = r.now()
    ok, note = await r.inject("push_robot", newtons=250)
    o.notes.append(note)
    if not ok:
        o.check("push injected", False, note)
        return
    fell = await r.wait_row(lambda x: x.get("type") == "safety_event" and x.get("kind") == "fell", 30)
    o.check("safety_event(fell)", fell is not None)
    stop = await r.wait_row(lambda x: x.get("type") == "stop" and "fell" in str(x.get("reason")), 10)
    o.check("paused", stop is not None or any(f["paused"] for f in r.frames if f["t"] >= t_push))
    o.check("spoken notice", bool(r.said_after(t_push, r"balance|fell|fall|steady")), r.said_after(t_push, r".")[:2])
    await r.until(lambda: any(e.get("type") in ("body.ready", "body_ready") and (e.get("t") or 0) >= t_push
                              for e in r.events), 70)
    ready = next((e for e in r.events if e.get("type") in ("body.ready", "body_ready") and (e.get("t") or 0) >= t_push),
                 None)
    rec = next((e for e in r.events if e.get("type") in ("sim_recovery", "sim.recovery") and (e.get("t") or 0) >= t_push),
               None)
    path = (rec or {}).get("path")
    took = None if ready is None else float(ready.get("t") or 0) - t_push
    o.check("sim_recovery{path} labelled", rec is not None, rec)
    o.check("body.ready <= 20 s (path A) or <= 60 s (path B)", None if ready is None else
            (took <= 20 if path in ("A", "a", None) else took <= 60), {"took_s": took, "path": path})
    o.check("no command{stop} sent", None if not any("command" in str(e.get("type")) for e in r.events) else
            not any(e.get("type") in ("body.command_stop", "command_stop") for e in r.events))
    o.check("reconcile after recovery", bool([x for x in r.rows("reconcile_start") if (x.get("t") or 0) >= t_push]))
    o.extra["t0"] = t0


async def g8_deploy_restart(r: StackRun, o: Outcome) -> None:
    await r.begin_plain(H40)
    for variant in ("a: kill P2", "b: kill button (estop)"):
        t = r.now()
        if variant.startswith("a"):
            ok, note = await r.inject("kill_deploy")
        else:
            await r.send(type="estop", confirm=True)
            ok, note = True, "page estop"
        o.notes.append(f"{variant}: {note}")
        if not ok:
            o.check(f"{variant}: injected", False, note)
            continue
        await r.until(lambda: any(e.get("type") in ("body.ready", "body_ready") and (e.get("t") or 0) > t
                                  for e in r.events), 90)
        ready = next((e for e in r.events if e.get("type") in ("body.ready", "body_ready") and (e.get("t") or 0) > t),
                     None)
        took = None if ready is None else float(ready.get("t") or 0) - t
        o.check(f"{variant}: body.ready <= 60 s (target 30-45 s)", None if ready is None else took <= 60,
                {"took_s": took})
        rec = next((e for e in r.events if e.get("type") in ("sim_recovery", "sim.recovery") and (e.get("t") or 0) > t),
                   None)
        o.check(f"{variant}: path B labelled", None if rec is None else str(rec.get("path")).upper() == "B", rec)
    o.check("a third recovery disables the body tools", None, "not exercised (costly); verify by hand")


async def g9_rtf_degraded(r: StackRun, o: Outcome) -> None:
    ok, note = await r.inject("throttle_rtf", rtf=0.9)
    o.notes.append(note)
    if not ok:
        o.check("RTF throttled", False, note)
        return
    try:
        t0 = await _fetch_start(r)
        rej = await r.wait_row(lambda x: x.get("type") == "rejected" and x.get("tool") == "manipulate", r.limit())
        o.check("manipulate rejected (DEGRADED)", rej is not None and re.search(r"rtf|degraded|sim_slow|slow",
                                                                               str(rej.get("why")), re.I) is not None,
                (rej or {}).get("why"))
        sp = [v for t, v in r.speeds(t0, r.now())]
        cap = next((f["stack"].get("walk_cap_mps") for f in reversed(r.frames) if f["stack"].get("walk_cap_mps")), None)
        o.check("walking capped", None if cap is None or not sp else max(sp) <= float(cap) * 1.1,
                {"cap": cap, "max_speed": round(max(sp), 2) if sp else None})
    finally:
        await r.inject("restore_rtf")


async def g10_carry_walk(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    oid = "alarm_clock_1"
    pick = await r.wait_row(lambda x: x.get("type") == "result" and x.get("tool") == "manipulate"
                            and x.get("action") == "pick" and x.get("status") == "succeeded", r.limit())
    if pick is None:
        o.check("picked", False)
        return
    place = await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "manipulate"
                             and x.get("action") == "place", r.limit())
    t1 = float((place or {}).get("t") or r.now())
    fr = [f for f in r.frames if float(pick.get("t") or 0) < f["t"] < t1]
    wh = [f["where"].get(oid) for f in fr]
    o.check("where stays hand:* while carrying", bool(wh) and all(str(w).startswith("hand") for w in wh),
            sorted({str(w) for w in wh}))
    dist = sum(math.hypot(b["x"] - a["x"], b["z"] - a["z"]) for a, b in zip(fr, fr[1:])
               if a["x"] is not None and b["x"] is not None)
    o.extra["carried_m"] = round(dist, 2)
    if dist < 10.0:
        o.notes.append(f"carried {dist:.1f} m (PLAN's 10 m; H40's route is shorter)")


async def g11_fallback_is_labelled(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), r.limit())
    stones = sc.stepping_stones() | {"lite"}
    res = [x for x in r.results() if (x.get("executor") or (x.get("data") or {}).get("executor")) in stones]
    if not res:
        o.check("stepping-stone results to check", NA, "no stepping-stone executor ran")
        return
    manip = [x for x in res if x.get("tool") == "manipulate"]
    o.check("every stepping-stone result names its executor (and skill for manipulate)",
            all((x.get("executor") or (x.get("data") or {}).get("executor")) for x in res)
            and all((x.get("data") or {}).get("skill") for x in manip), len(res))
    o.check("the manipulate summaries the planner reads say [fallback]",
            bool(manip) and all("fallback" in str(x.get("summary")) for x in manip),
            [x.get("summary") for x in manip][:3])
    cfg = set(((r.init or {}).get("config") or {}).get("stepping_stones") or [])
    used = {x.get("executor") or (x.get("data") or {}).get("executor") for x in res}
    o.check("the page lists them as stepping stones (amber badge)", used <= cfg, {"used": sorted(used),
                                                                                  "page": sorted(cfg)})
    fb = [e for f in r.frames_seen for e in (f["runtime"].get("executions") or []) if e.get("executor") in used]
    o.check("the page's executions mark them fallback", bool(fb) and all(e.get("fallback") for e in fb), len(fb))
    inputs = r.model_inputs()
    if inputs:
        acts = [i.split("ACTIONS (latest last)")[-1] for i in inputs if "ACTIONS (latest last)" in i]
        seen = [a for a in acts if any(x.get("execution_id") and x["execution_id"] in a for x in res)]
        o.check("the prompt's ACTIONS lines show [fallback]", bool(seen) and all("fallback" in a for a in seen),
                len(seen))
    elif isinstance(r.injector, LocalInjector):
        o.check("the prompt's ACTIONS lines show [fallback]", _render_actions_fallback(r, res),
                "scripted planner: ACTIONS rendered by agent.model from the contexts it was given")
    else:
        o.check("the prompt's ACTIONS lines show [fallback]", None, "no model calls to read the prompt from")


def _render_actions_fallback(r: StackRun, res: list[dict[str, Any]]) -> Any:
    ctxs = r.injector.planner_contexts() if isinstance(r.injector, LocalInjector) else []
    if not ctxs:
        return None
    try:
        from agent.model import ReferenceBrain
        brain = ReferenceBrain.__new__(ReferenceBrain)
        text = ReferenceBrain.render(brain, ctxs[-1])
    except Exception as e:  # noqa: BLE001
        r.injected.append(f"prompt render failed: {e!r}")
        return None
    acts = text.split("ACTIONS (latest last)")[-1].split("RUNNING NOW")[0]
    lines = [ln for ln in acts.splitlines() if any(x.get("execution_id") and x["execution_id"] in ln for x in res)]
    return bool(lines) and all("fallback" in ln for ln in lines)


async def g12_late_result(r: StackRun, o: Outcome) -> None:
    await _fetch_start(r)
    pick = await r.wait_row(lambda x: x.get("type") == "started" and x.get("tool") == "manipulate"
                            and x.get("action") == "pick", r.limit())
    if pick is None:
        o.check("a pick started", False)
        return
    t_c = r.now()
    await r.say("No, bring me the mug instead.")
    res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("execution_id") == pick.get("execution_id"),
                           r.wait(30))
    if res is None:
        o.check("the pick finished", False)
        return
    corr = next((x for x in r.rows("correction") if (x.get("t") or 0) >= t_c - 0.01), None)
    after = corr is not None and r.trace.index(res) > r.trace.index(corr)
    o.check("the pick's result arrived after the correction", after,
            {"correction_t": (corr or {}).get("t"), "result_t": res.get("t"), "status": res.get("status")})
    o.check("late=true on the result row", res.get("late") is True, res.get("late"))
    o.check("a late_result row", bool(r.row_for(pick.get("execution_id"), "late_result")))
    arm = (res.get("data") or {}).get("arm") or (pick.get("args") or {}).get("arm")
    t_res = float(res.get("t") or t_c)
    hands = [((f["hands"] or {}).get(arm) or {}) for f in r.frames if f["t"] >= t_res] if arm else []
    o.check("hand UNKNOWN (or unverified) in belief after it", None if not hands else
            any(str(h.get("holding")) == "UNKNOWN" or h.get("verified") is False for h in hands[:10]), {"arm": arm})
    done = r.said_after(t_res, r"here it is|here you go|done|got (it|the alarm)")
    o.check("no 'done' spoken for the old request", not [d for d in done if "alarm" in d.lower() or "here" in d.lower()]
            or r.where("mug_1") == r.user_surface, done[:3])


async def g13_schema_parity(r: StackRun, o: Outcome) -> None:
    from agent.model import system_prompt
    from api.results import envelope_schema
    from api.tools import SchemaContext, json_schemas
    from api.types import PROFILES
    t0 = await _fetch_start(r, forget=False)
    m = (r.init or {}).get("map") or {}
    start = next((x for x in r.trace if x.get("type") == "session_start"), {})
    types = start.get("skill_types") or ["alarm_clock", "apple", "mug"]
    mask = re.compile(r"\d+(\.\d+)?")

    def strip(schemas: list[dict[str, Any]]) -> Any:
        s = json.loads(json.dumps(schemas))
        for t in s:
            t.pop("description", None)
            for p in t["parameters"]["properties"].values():
                p.pop("description", None)
        return s
    per = {name: json_schemas(SchemaContext.from_map(m, types, p.slots())) for name, p in PROFILES.items()}
    ref = strip(next(iter(per.values())))
    o.check("tool schemas (names, args, enums) equal across profiles", all(strip(v) == ref for v in per.values()),
            sorted(per))
    descs = {n: [mask.sub("#", t["description"]) for t in v] for n, v in per.items()}
    o.check("tool descriptions equal with numbers masked", len({json.dumps(v) for v in descs.values()}) == 1)
    systems = {n: mask.sub("#", system_prompt(p.slots())) for n, p in PROFILES.items()}
    o.check("SYSTEM equal with numbers masked", len(set(systems.values())) == 1)
    o.check("one envelope schema", bool(envelope_schema().get("required")))
    local = isinstance(r.injector, LocalInjector)
    before = r.injector.schemas() if local else None
    ok, note = await r.inject("policy_down" if local else "kill_policy")
    o.notes.append(f"policy outage mid-session: {note}")
    await r.wait_sim(5.0, 30)
    await r.until(r.idle, r.wait(120))
    await r.inject("policy_up" if local else "restore_policy")
    if local:
        o.check("schemas unchanged across the session, through a policy outage", ok and r.injector.schemas() == before)
    else:
        sch = r.tool_schemas()
        o.check("schemas unchanged across the session, through a policy outage",
                None if (len(sch) < 2 or not ok) else all(s == sch[0] for s in sch), {"model calls": len(sch)})
    o.extra["t0"] = t0


async def g14_reposition_handoff(r: StackRun, o: Outcome) -> None:
    if not isinstance(r.injector, LocalInjector) and r.injector.has("place_object"):
        await r.inject("place_object", object="alarm_clock_1", offset_m=0.25)
    await _fetch_start(r)
    await r.until(lambda: bool(r.results("manipulate", action="pick")), r.limit())
    rch = r.results("check_reachability")
    need = next((i for i, x in enumerate(rch) if (x.get("data") or {}).get("reason") == "needs_reposition"), None)
    o.check("check_reachability -> needs_reposition", need is not None,
            [(x.get("data") or {}).get("reason") for x in rch])
    if need is None:
        o.notes.append("the stance was already good: no reposition to hand off (place the object 0.25 m out)")
        return
    rows = r.trace
    i_need = rows.index(rch[need])
    rep = next((x for x in rows[i_need:] if x.get("type") == "result" and x.get("tool") == "navigate"
                and x.get("action") == "reposition"), None)
    o.check("navigate(reach_stance) succeeded", (rep or {}).get("status") == "succeeded",
            None if rep is None else rep.get("summary"))
    i_rep = rows.index(rep) if rep else len(rows)
    again = next((x for x in rows[i_rep:] if x.get("type") == "result" and x.get("tool") == "check_reachability"), None)
    o.check("a fresh check_reachability says reachable", bool(((again or {}).get("data") or {}).get("reachable")))
    i_again = rows.index(again) if again else len(rows)
    man = next((x for x in rows[i_again:] if x.get("type") == "result" and x.get("tool") == "manipulate"), None)
    o.check("then manipulate", man is not None and man.get("status") == "succeeded",
            None if man is None else man.get("summary"))
    if man is not None:
        st = r.row_for(man.get("execution_id"), "started") or {}
        t0, t1 = float(st.get("t") or man.get("t_start") or 0), float(man.get("t") or 0)
        fr = [f for f in r.frames if t0 <= f["t"] <= t1 and f["x"] is not None]
        drift = max((math.hypot(f["x"] - fr[0]["x"], f["z"] - fr[0]["z"]) for f in fr), default=None) if fr else None
        loco = [f["mode"] for f in r.frames if t0 <= f["t"] <= t1 and f["mode"] == "LOCOMOTION"]
        o.check("manipulate sends no LOCOMOTION command (no base motion)",
                None if drift is None else (drift < 0.02 and not loco),
                {"drift_m": None if drift is None else round(drift, 3), "locomotion_frames": len(loco),
                 "base_shift_m": (man.get("data") or {}).get("base_shift_m")})


# ----------------------------------------------------------------------------------------------------------
# E5: GR00T plumbing (full)
# ----------------------------------------------------------------------------------------------------------
E5_PLAN = ("plain",) * 4 + ("cancel",) * 3 + ("halt",) * 3
INFERENCE_EVENTS = ("groot.inference", "vla.inference", "groot_inference")


async def e5_groot_plumbing(r: StackRun, o: Outcome, trials: tuple[str, ...] = E5_PLAN,
                            scenario: str = "fetch_other_room", oid: str = "alarm_clock_1",
                            label: str = "alarm clock") -> None:
    rows = []
    for i, kind in enumerate(trials):
        t0 = await _fetch_start(r, scenario, oid, label, forget=False)
        pick = await _groot_pick_started(r, r.wait(240))
        row: dict[str, Any] = {"trial": i + 1, "kind": kind, "execution_id": (pick or {}).get("execution_id")}
        if pick is None:
            row["error"] = "no pick started"
            rows.append(row)
            continue
        t_act = None
        if kind in ("cancel", "halt"):
            # act while the GR00T session runs: 0.5 s after its first inference (a fixed delay can land on the
            # labelled fallback once a zero-shot attempt ends early, and then E5 would score the script, not GR00T)
            eid = pick.get("execution_id")
            first = await r.wait_event(lambda e: e.get("type") in INFERENCE_EVENTS
                                       and (e.get("session") or e.get("execution_id")) == eid, r.wait(30))
            row["groot_running_at_act"] = first is not None
            await r.wait_sim(0.5 if first is not None else 2.0, 30)
            t_act = r.now()
            await r.say("No, never mind." if kind == "cancel" else "stop")
        res = await r.wait_row(lambda x: x.get("type") == "result" and x.get("execution_id") == pick.get("execution_id"),
                               r.wait(60))
        d = (res or {}).get("data") or {}
        await asyncio.sleep(1.0)
        stop = next((x for x in r.rows("stop") if t_act is not None and (x.get("t") or 0) >= t_act), None)
        # groot_then_script (profile full): a failed GR00T attempt hands over to a labelled fallback, and the result
        # is the fallback's; the GR00T attempt is data.attempts[0]. Only a GR00T attempt that succeeded counts as a
        # GR00T success (a kinematic-attach fallback also ends with the object in hand).
        att = [a for a in (d.get("attempts") or []) if isinstance(a, dict)]
        g = att[0] if att else {"executor": d.get("executor"), "skill": d.get("skill"),
                                "status": (res or {}).get("status"), "reason": d.get("reason")}
        row.update(status=(res or {}).get("status"), reason=d.get("reason"), executor=d.get("executor"),
                   skill=d.get("skill"), holding=d.get("holding"), gt_where=r.where(oid), fell=r.fell(t0),
                   inferences=d.get("inferences"), took_s=None if (res is None or t_act is None)
                   else round(float(res.get("t") or 0) - t_act, 2),
                   receipt=(stop or {}).get("receipt"), late=(res or {}).get("late"),
                   groot_executor=g.get("executor"), groot_skill=g.get("skill"), groot_status=g.get("status"),
                   groot_reason=g.get("reason"), fallback=(d.get("fallback_from") or {}).get("executor") is not None,
                   gt_success=g.get("executor") in GROOT_EXECUTORS and g.get("status") == "succeeded"
                   and str(r.where(oid) or "").startswith("hand"))
        rows.append(row)
        if kind == "halt":
            await r.say("Okay, carry on.")
        await r.say("stop")
        await r.until(r.idle, r.wait(20))
    o.extra["trials"] = rows
    ran = [x for x in rows if x.get("status")]
    o.check("every trial ran a pick", len(ran) == len(trials), [x.get("error") for x in rows if x.get("error")])
    o.check("0 falls", not any(x.get("fell") for x in rows))
    # GR00T plumbing means GR00T ran: a session the body refused at `enter` (0 inferences) hands straight to the
    # fallback, and a cancel or halt that lands on the fallback says nothing about the GR00T session
    o.check(f"every GR00T attempt ran the policy ({sum(bool(x.get('inferences')) for x in ran)}/{len(ran)} with "
            f"inferences > 0)", bool(ran) and all(x.get("inferences") for x in ran),
            [(x["trial"], x.get("groot_status"), x.get("groot_reason")) for x in ran if not x.get("inferences")][:5])
    canc = [x for x in rows if x["kind"] == "cancel"]
    ok_c = [x.get("status") == "cancelled" and (x.get("took_s") or 99) <= 3.0 and x.get("groot_status") == "cancelled"
            for x in canc]
    o.check(f"cancel {sum(ok_c)}/{len(canc)} (the GR00T session cancelled within the 3 s grace)",
            bool(canc) and all(ok_c), [(x["trial"], x.get("groot_status"), x.get("executor")) for x in canc])
    halt = [x for x in rows if x["kind"] == "halt"]
    ok_h = [bool((x.get("receipt") or {}).get("stopped")) and x.get("status") == "failed"
            and x.get("reason") == "halted" and x.get("groot_reason") == "halted" for x in halt]
    o.check(f"halt {sum(ok_h)}/{len(halt)} (receipt stopped, the GR00T session failed(halted))",
            bool(halt) and all(ok_h), [(x["trial"], x.get("groot_reason"), x.get("executor")) for x in halt])
    o.check("every result names groot_arms (or its GR00T attempt, before a labelled fallback), the skill and the GT "
            "outcome",
            bool(ran) and all(x.get("groot_executor") == "groot_arms"
                              and str(x.get("groot_skill") or "").startswith("groot.")
                              and x.get("gt_where") is not None for x in ran),
            [(x.get("groot_executor"), x.get("groot_skill"), x.get("executor")) for x in ran][:3])
    succ = [x for x in rows if x["kind"] == "plain"]
    o.extra["success"] = f"{sum(bool(x.get('gt_success')) for x in succ)}/{len(succ)} plain picks ended in hand (GT); " \
                         "reported, not required (experimental checkpoint)"
    o.notes.append(o.extra["success"])


# ----------------------------------------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------------------------------------
ALL = ("lite", "bringup", "sonic", "full")
SPECS: list[Spec] = [
    Spec("G1", "halt_midstride", ("sonic", "full"), g1_halt_midstride),
    Spec("G2", "walk_cancel", ALL, g2_walk_cancel),
    Spec("G3", "grasp_cancel", ("full",), g3_grasp_cancel),
    Spec("G4", "stale_chunk_after_correction", ("full",), g4_stale_chunk, needs=("hook:delay_proxy_on",)),
    Spec("G5", "policy_down", ("full", "lite"), g5_policy_down, needs=("local:policy_down", "hook:kill_policy"),
         lite="injected"),
    Spec("G6", "blocked_path", ALL, g6_blocked_path, needs=("local:stuck", "hook:spawn_box"), lite="injected"),
    Spec("G7", "fall_recovery", ("sonic", "full"), g7_fall_recovery, needs=("hook:push_robot",)),
    Spec("G8", "deploy_restart", ("sonic", "full"), g8_deploy_restart, needs=("hook:kill_deploy",)),
    Spec("G9", "rtf_degraded", ("sonic", "full"), g9_rtf_degraded, needs=("hook:throttle_rtf",)),
    Spec("G10", "carry_walk", ("sonic", "full"), g10_carry_walk),
    Spec("G11", "fallback_is_labelled", ALL, g11_fallback_is_labelled),
    Spec("G12", "late_result_is_world_info", ALL, g12_late_result),
    Spec("G13", "schema_parity", ALL, g13_schema_parity),
    Spec("G14", "reposition_handoff", ALL, g14_reposition_handoff),
    Spec("E5", "groot_plumbing", ("full",), e5_groot_plumbing),
]
SPEED = {"G12": 5.0}             # offline: at most this sim speed (the correction must land during the lite pick)


async def _begin_plain(self: StackRun, house: str) -> None:
    self.current = ""
    await self.load(house, forget=False)
StackRun.begin_plain = _begin_plain      # type: ignore[attr-defined]


async def run_one(spec: Spec, run: StackRun, local: bool) -> dict[str, Any]:
    o = Outcome()
    t0 = time.monotonic()
    run.injected = []
    run.fixtures, run.frames_seen = {}, []
    if run.profile not in spec.profiles:
        verdict, o.notes = "N/A", [f"not a {run.profile} scenario (PLAN 9.3: {', '.join(spec.profiles)})"]
    elif (missing := _needs(spec, run, local)):
        verdict, o.notes = "SKIPPED", [f"no injector for {', '.join(missing)}"
                                       + ("" if local else " (pass --hook name=cmd)")]
    else:
        try:
            await spec.fn(run, o)
            verdict = o.verdict()
        except Exception as e:  # noqa: BLE001  (a broken scenario fails, the suite goes on)
            import traceback
            o.notes.append(f"error: {e!r} {traceback.format_exc()[-400:]}")
            verdict = "FAIL"
    used = run.executors() if verdict not in ("N/A", "SKIPPED") else {"nav": {}, "manip": {}, "skills": {}}
    hon = sc.honesty(used, profile=run.profile)
    groot = sorted({s for s in (used.get("skills") or {}) if str(s).startswith("groot.")})
    labels = list(hon["labels"]) + (["experimental GR00T skill: " + ", ".join(groot)] if groot else [])
    shown = "PASS*" if verdict == "PASS" and hon["fallback"] else verdict
    return {"id": spec.id, "name": spec.name, "verdict": shown, "passed": verdict == "PASS",
            "fallback_pass": verdict == "PASS" and hon["fallback"], "target_pass": verdict == "PASS" and not hon["fallback"],
            "profile": run.profile, "applies": run.profile in spec.profiles, "lite_mode": spec.lite,
            "criteria": o.criteria, "notes": o.notes, "injected": list(run.injected), "extra": o.extra,
            "executors_used": used, "honesty": labels, "seconds": round(time.monotonic() - t0, 1)}


def line(res: dict[str, Any]) -> str:
    crit = "; ".join(f"{'ok' if c['ok'] is True else 'NO' if c['ok'] is False else '??' if c['ok'] is None else 'n/a'} "
                     f"{c['name']}" for c in res["criteria"])
    notes = f" | {'; '.join(res['notes'])}" if res["notes"] else ""
    return f"{res['verdict']:<10} {res['id']:<4} {res['name']:<28} {res['seconds']:>6.1f}s  {crit}{notes}"


def summarize(results: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    applicable = [x for x in results if x["applies"] and x["verdict"] != "SKIPPED" and x["id"] != "E5"]
    passed = sum(x["passed"] for x in applicable)
    need, of = BARS.get(profile, (None, None))
    lite = [x for x in results if x["id"] in LITE_SUBSET]
    return {"profile": profile, "wall": round(time.time()), "passed": passed, "applicable": len(applicable),
            "bar": {"need": need, "of": of, "met": None if need is None else passed >= need},
            "lite_subset": {"passed": sum(x["passed"] for x in lite), "of": len(LITE_SUBSET)} if profile == "lite" else None,
            "verdicts": {x["id"]: x["verdict"] for x in results}, "results": results}


# ----------------------------------------------------------------------------------------------------------
# Running it
# ----------------------------------------------------------------------------------------------------------
async def run_suite(ids: list[str], *, url: str | None = None, profile: str = "lite", speed: float = 5.0,
                    planner: str = "brains.scripted:create", system1: str = "tests.kept.system1_stub:create",
                    hooks: dict[str, str] | None = None, trace_dir: Path | None = None,
                    scale: float | None = None) -> dict[str, Any]:
    from websockets.asyncio.client import connect
    hub = server = ticker = None
    speed_box = {"v": speed}
    if url is None:
        from sim.clock import SimClock
        from ui.server import Deps, Hub, _import, serve_hub
        deps = Deps(create_planner=lambda info: _import(planner)(info),
                    clock=lambda _s: SimClock(speed_box["v"])).resolve()
        hub = Hub(H40, profile, deps=deps, cameras="auto", system1=system1)
        server = await serve_hub(hub, "127.0.0.1", 0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws"
        hub.start_system1()
        ticker = asyncio.create_task(hub.ticker())
    injector: Injector = LocalInjector(hub) if hub is not None else HookInjector(hooks or {})
    results = []
    try:
        async with connect(url, max_size=2 ** 24) as ws:
            run = StackRun(ws, profile, scale, injector)
            reader = asyncio.create_task(run.reader())
            for spec in [s for s in SPECS if s.id in ids]:
                speed_box["v"] = min(speed, SPEED.get(spec.id, speed))
                res = await run_one(spec, run, local=hub is not None)
                if trace_dir is not None and res["verdict"] not in ("N/A", "SKIPPED"):
                    res["trace_path"] = str(run.dump(trace_dir / f"stack_{profile}_{spec.id}.jsonl"))
                results.append(res)
                print(line(res), flush=True)
            reader.cancel()
    finally:
        if hub is not None:
            ticker.cancel()
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
    out = summarize(results, profile)
    out.update(mode="offline (in-process lite)" if hub is not None else f"live page {url}",
               planner=planner if hub is not None else "(the page's)", system1=system1 if hub is not None else "(the page's)",
               speed=speed if hub is not None else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="referee a running page server (default: offline, in process, lite)")
    ap.add_argument("--profile", default="lite", choices=list(ALL))
    ap.add_argument("--only", default="", help="comma-separated ids (G1..G14, E5); default: every G scenario")
    ap.add_argument("--speed", type=float, default=5.0, help="offline sim clock speed (scripted planner)")
    ap.add_argument("--planner", default="brains.scripted:create", help="offline planner factory")
    ap.add_argument("--system1", default="tests.kept.system1_stub:create", help="offline System 1 factory")
    ap.add_argument("--hook", action="append", default=[], help="name=shell command (live stack fault injection)")
    ap.add_argument("--time-scale", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    args = ap.parse_args()
    ids = [x.strip().upper() for x in args.only.split(",") if x.strip()] or [s.id for s in SPECS if s.id != "E5"]
    hooks = dict(h.split("=", 1) for h in args.hook if "=" in h)
    if args.url is None and (args.planner != "brains.scripted:create" or "stub" not in args.system1):
        from ui.server import load_env_files
        load_env_files()                  # live models offline: keys from ~/.config/ludo-g1/secrets.env, never printed
        if args.speed != 1.0:
            print("live models answer in wall time: --speed forced to 1.0", flush=True)
            args.speed = 1.0
    if args.url is None and not os.environ.get("WORLDLINE_RUNS"):
        os.environ["WORLDLINE_RUNS"] = tempfile.mkdtemp(prefix="wl-stack-")
        from eval.offline_episode import _isolate_runs
        _isolate_runs(Path(os.environ["WORLDLINE_RUNS"]))
    out = asyncio.run(run_suite(ids, url=args.url, profile=args.profile, speed=args.speed, planner=args.planner,
                                system1=args.system1, hooks=hooks, scale=args.time_scale,
                                trace_dir=Path(args.trace_dir) if args.trace_dir else None))
    path = Path(args.out) if args.out else suite.OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_stack_{args.profile}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    b = out["bar"]
    print(f"\n{out['passed']}/{out['applicable']} applicable G scenarios passed on {args.profile} ({out['mode']}); "
          f"bar {b['need']}/{b['of']}: {'met' if b['met'] else 'NOT met' if b['met'] is False else '-'} · {path}")
    if out.get("lite_subset"):
        print(f"lite subset (G2 G5 G6 G11 G12 G13 G14): {out['lite_subset']['passed']}/7")
    return 0 if b["met"] else 1


if __name__ == "__main__":
    sys.exit(main())
