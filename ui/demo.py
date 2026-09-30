"""The Worldline UI's Demo panel, server side (docs/demo.md). The steps are config/demo_steps.yaml, the same list
scripts/demo.sh runs. Each line goes through the chat's own say path (Hub.say: a stop word halts first, then System 1
labels the line and the runtime hears it, exactly as if typed), with tools/say.py's timing (tools.demo_steps.say_lines:
"@N" lines N s after the previous one, otherwise wait for an idle runtime), and each step is judged by the same PASS
rule as scripts/demo.sh (tools.demo_steps.verdict). Progress streams to every open page.

    page -> server  {type: demo, action: run, steps: [ids] | "all"}   one demo at a time
                    {type: demo, action: stop}                        no further lines; "stop" said to the robot once
                    {type: demo, action: record, on: true|false}      the Sim Viewer's POST /api/record (8765 + offset)
                    {type: demo, action: fresh}                       Isaac profiles: m2_down.sh && m2_up.sh, detached
                    {type: demo, action: steps}                       send demo_steps again
    server -> page  {type: demo_steps, steps: [...], error, source, state, record, fresh}   on connect
                    {type: demo_run, step: id, state: idle|running|pass|fail, detail, line?, s?, running}
                    {type: demo_run, step: null, state: running|pass|fail, detail, running}   the run as a whole
                    {type: demo_record, on, busy, dir, detail}
                    {type: demo_fresh, state: restarting, detail, log}

On the box (/work exists, not the lite profile) a run holds the stack lock /work/locks/stack.d as `ui-demo`
(docs/bringup.md) and releases it after; a lock held by anyone else stops the run before its first line. With the
steps file's `groot: off` (the default, as scripts/demo.sh --groot off) a run on the box's full profile takes the GR00T
link down (scripts/groot_link.sh down) and brings it back after (ensure), so the pick uses the labelled SONIC arm script. Evidence per
run: runs/demo/<ts>/stepN.json (tools/say.py --json's shape) and results.md (scripts/demo.sh's table).
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from tools.demo_steps import Demo, Step, load, say_lines, table, verdict

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = Path("/work/locks/stack.d")
OWNER = "ui-demo"                 # the stack lock's owner while a demo runs
FRESH_OWNER = "ui-demo-fresh"     # ... and while the detached restart runs
VIZ_PORT = 8765                   # the Sim Viewer (viz/server.py), + the port offset
FRESH_NOTE = "restarting the stack (~60 s)…"


class SessionGone(Exception):
    """The page started a new session (or none runs) while a step was going."""


# ---------------------------------------------------------------------------------------------- the stack lock
class StackLock:
    """/work/locks/stack.d, taken with mkdir, its owner file "<name> <epoch>" (docs/bringup.md). path None: no lock
    (the laptop, the lite profile). A lock left under our own name (a P5 that died mid-demo) is taken over."""

    def __init__(self, path: Path | None, owner: str = OWNER) -> None:
        self.path, self.owner, self.owned = path, owner, False

    @classmethod
    def for_profile(cls, profile: str) -> "StackLock":
        env = os.environ.get("WL_STACK_LOCK", "")
        if env.lower() == "off":
            return cls(None)
        if env:
            return cls(Path(env))
        if profile == "lite" or not LOCK_DIR.parent.parent.is_dir():
            return cls(None)
        return cls(LOCK_DIR)

    def holder(self) -> str | None:
        if self.path is None or not self.path.is_dir():
            return None
        try:
            return (self.path / "owner").read_text().strip() or "?"
        except OSError:
            return "?"

    def acquire(self) -> tuple[bool, str]:
        if self.path is None:
            return True, "no stack lock here"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.mkdir()
        except FileExistsError:
            who = self.holder() or "?"
            if who.split()[0] == FRESH_OWNER:
                return False, "the stack restart is still finishing (stack lock ui-demo-fresh): try again in a moment"
            if who.split()[0] != self.owner:
                return False, (f"the stack lock is held by '{who}': someone is using the stack "
                               f"(wait, or if it is stale: rm -rf {self.path})")
            took = f"took over the stack lock left as '{who}'"
        except OSError as e:                            # best effort: a lock we cannot make does not stop a demo
            return True, f"stack lock not taken ({e})"
        else:
            took = f"stack lock taken as {self.owner}"
        try:
            (self.path / "owner").write_text(f"{self.owner} {int(time.time())}\n")
        except OSError:
            pass
        self.owned = True
        return True, took

    def release(self) -> None:
        if not self.owned or self.path is None:
            return
        self.owned = False
        who = self.holder()
        if who is None or who.split()[0] == self.owner:
            shutil.rmtree(self.path, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- the page server
class SessionChannel:
    """tools.demo_steps.Channel on one page session: say through the Hub, read its trace and runtime directly."""

    def __init__(self, hub: Any, session: Any) -> None:
        self.hub, self.s = hub, session
        self.base = len(session._trace_rows())

    def _check(self) -> None:
        if self.hub.session is not self.s:
            raise SessionGone("a new session started during this step")

    async def say(self, text: str) -> None:
        self._check()
        await self.hub.say(text)

    def count(self) -> int:
        return len(self.s._trace_rows()) - self.base

    def rows(self, start: int) -> list[dict]:
        return self.s._trace_from(self.base + start)

    def busy(self, ignore_wait: bool) -> bool:
        self._check()
        return self.s.busy(ignore_wait)


class HubHost:
    """What the runner needs from ui.server's Hub."""

    def __init__(self, hub: Any, emit: Callable[[dict], None]) -> None:
        self.hub, self.emit = hub, emit

    def session(self) -> Any:
        s = self.hub.session
        return s if s is not None and s.runtime is not None else None

    def channel(self) -> SessionChannel:
        s = self.session()
        if s is None:
            raise SessionGone("no session is running")
        return SessionChannel(self.hub, s)

    def ready(self) -> tuple[bool, str]:
        """tools/say.py --wait-ready: System 1 ready (or not configured) and nothing running but a wait."""
        s = self.session()
        if s is None:
            return False, "no session is running"
        s1 = self.hub.s1_status
        busy = s.busy(True)
        return (s1 in ("ready", "off") and not busy), f"System 1 {s1}, runtime {'busy' if busy else 'idle'}"

    def end_state(self) -> dict:
        s = self.session()
        st = s.state() if s is not None else {}
        rt = st.get("runtime") or {}
        belief = rt.get("belief") or {}
        return {"robot": belief.get("robot"), "hands": belief.get("hands"), "paused": rt.get("paused"),
                "system1": {"status": self.hub.s1_status, "detail": self.hub.s1_detail}}

    async def say(self, text: str) -> None:
        await self.hub.say(text)

    def profile(self) -> str:
        s = self.hub.session
        return s.profile if s is not None else self.hub.profile

    def scene(self) -> str:
        s = self.hub.session
        return s.scene if s is not None else self.hub.default

    def port_offset(self) -> int:
        off = getattr(self.hub, "_port_offset", None)
        if off is None:
            off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
        return int(off)

    def port(self) -> int | None:
        return getattr(self.hub, "port", None)


# ---------------------------------------------------------------------------------------------- helpers
def _post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def tmux_session() -> str:
    """The stack's tmux session: WL_SESSION, else the session of P5's own tmux pane, else wl-m2."""
    if os.environ.get("WL_SESSION"):
        return os.environ["WL_SESSION"]
    pane = os.environ.get("TMUX_PANE")
    if pane and shutil.which("tmux"):
        try:
            out = subprocess.run(["tmux", "display-message", "-p", "-t", pane, "#S"], capture_output=True,
                                 text=True, timeout=3).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return "wl-m2"


def run_script(args: list[str], timeout: float) -> int:
    """bash scripts/... in the repo, without tmux's variables (P5 runs in a tmux pane); the exit code (124: timeout)."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE")}
    try:
        return subprocess.run(["bash", *args], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError:
        return 127


P5_ONLY_ENV = ("TMUX", "TMUX_PANE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def spawn_detached(cmd: str, log: Path) -> subprocess.Popen:
    """bash -c cmd under nohup in a new session (setsid), so it outlives P5 (m2_down.sh stops P5 first). Without P5's
    own settings: tmux's variables (the helper is not in P5's pane) and m2_p5.sh's thread caps."""
    env = {k: v for k, v in os.environ.items() if k not in P5_ONLY_ENV}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        return subprocess.Popen(["nohup", "bash", "-c", cmd], cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True, env=env)


# ---------------------------------------------------------------------------------------------- the runner
class DemoRunner:
    def __init__(self, host: Any, *, steps_file: Path | str | None = None, out_dir: Path | None = None,
                 lock: StackLock | None = None, viz_url: str | None = None,
                 spawn: Callable[[str, Path], Any] = spawn_detached, sh: Callable[[list[str], float], int] = run_script,
                 poll_s: float = 0.3) -> None:
        self.host = host
        self.steps_file = steps_file
        self._out_dir = out_dir
        self._lock = lock                            # None: StackLock.for_profile at each run
        self._viz_url = viz_url
        self.spawn = spawn
        self.sh = sh
        self.poll_s = poll_s
        self.demo: Demo | None = None
        self.error: str | None = None
        self.states: dict[int, dict[str, Any]] = {}
        self.task: asyncio.Task | None = None
        self.current: int | None = None
        self.line: dict[str, Any] | None = None
        self.summary = ""
        self.record: dict[str, Any] = {"on": False, "busy": False, "dir": None, "detail": ""}
        self.fresh_sent: dict[str, Any] | None = None
        self.lock: StackLock | None = None
        self._rec_task: asyncio.Task | None = None
        self._link_back = False                      # the GR00T link was up before the run: ensure it after
        self.reload()

    # ------------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def out_dir(self) -> Path:
        if self._out_dir is not None:
            return self._out_dir
        return Path(os.environ.get("WORLDLINE_RUNS") or ROOT / "runs") / "demo"

    def reload(self) -> None:
        try:
            self.demo, self.error = load(self.steps_file), None
        except Exception as e:  # noqa: BLE001  (a broken file shows on the panel, the page keeps working)
            self.demo, self.error = None, f"cannot read the demo steps: {e}"

    def viz_url(self) -> str | None:
        """The Sim Viewer's base URL (WL_VIZ_URL, else 127.0.0.1:8765 + offset), or None: on lite there is no sim
        to record (and a laptop's 8765 may be a tunnel to the box's viewer), or 8765 + offset is this page itself."""
        url = os.environ.get("WL_VIZ_URL") or self._viz_url
        if url:
            return url.rstrip("/")
        port = VIZ_PORT + self.host.port_offset()
        if self.host.profile() == "lite" or self.host.port() == port:
            return None
        return f"http://127.0.0.1:{port}"

    def _no_viz(self) -> str:
        if self.host.profile() == "lite":
            return "no Sim Viewer on the lite profile"
        return f"no Sim Viewer: this page itself is on :{self.host.port()}"

    def fresh_info(self) -> dict[str, Any]:
        if self.host.profile() == "lite":
            return {"ok": False, "why": "only on the Isaac profiles: lite has no stack to restart (Start new session)"}
        return {"ok": True, "why": "m2_down.sh, then m2_up.sh with this profile and house; about a minute"}

    def snapshot(self) -> dict[str, Any]:
        return {"running": self.running, "current": self.current, "line": self.line, "summary": self.summary,
                "steps": {str(k): v for k, v in self.states.items()}}

    def steps_message(self) -> dict[str, Any]:
        self.reload()
        d = self.demo
        viz = self.viz_url()
        return {"type": "demo_steps", "steps": [s.public() for s in d.steps] if d else [], "error": self.error,
                "source": d.source if d else None, "state": self.snapshot(),
                "record": {**self.record, "available": viz is not None,
                           "viz": viz or self._no_viz()},
                "fresh": self.fresh_info()}

    def _emit_step(self, sid: int, state: str, detail: str = "", **extra: Any) -> None:
        self.states[sid] = {"state": state, "detail": detail, **{k: v for k, v in extra.items() if k == "s"}}
        self.host.emit({"type": "demo_run", "step": sid, "state": state, "detail": detail, "running": self.running,
                        **extra})

    def _emit_run(self, state: str, detail: str, running: bool | None = None) -> None:
        if state != "running":
            self.summary = detail
        self.host.emit({"type": "demo_run", "step": None, "state": state, "detail": detail,
                        "running": self.running if running is None else running})

    def _emit_record(self, **kw: Any) -> None:
        self.record.update(kw)
        self.host.emit({"type": "demo_record", **self.record})

    # ------------------------------------------------------------------ page messages
    async def handle(self, msg: dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "run":
            self.start(msg.get("steps", "all"))
        elif action == "stop":
            await self.stop()
        elif action == "record":                  # in the background: a stop encodes for up to a minute, and the
            if self.record["busy"] or (self._rec_task is not None and not self._rec_task.done()):
                raise ValueError("the recorder is busy")  # page's chat (its Stop) must not wait behind it
            self._rec_task = asyncio.create_task(self.set_record(bool(msg.get("on"))))
        elif action == "fresh":
            self.fresh()
        elif action == "steps":
            self.host.emit(self.steps_message())
        else:
            raise ValueError(f"unknown demo action {action!r}")

    def start(self, steps: Any) -> None:
        if self.running:
            raise ValueError("a demo is already running (Stop it first)")
        if self.fresh_sent:
            raise ValueError("the stack is restarting")
        self.reload()
        if self.demo is None:
            raise ValueError(self.error or "no demo steps")
        if steps == "all" or steps is None:
            ids = self.demo.ids
        else:
            if not isinstance(steps, list) or not steps:
                raise ValueError("steps: a list of step ids, or \"all\"")
            ids = []
            for x in steps:
                try:
                    sid = int(x)
                    self.demo.step(sid)
                except (TypeError, ValueError, KeyError):
                    raise ValueError(f"no demo step {x!r}") from None
                ids.append(sid)
        if self.host.session() is None:
            raise ValueError("no session is running: start one first")
        self.task = asyncio.create_task(self._run(self.demo, ids))

    async def stop(self) -> None:
        """No further lines, then "stop" said to the robot once (the chat's keyword path: halt, no model)."""
        task = self.task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:            # cancelled before its first line: nothing was said or taken
            self.current, self.line = None, None
            self._emit_run("fail", "stopped from the page", running=False)
        try:
            await self.host.say("stop")
        except Exception:  # noqa: BLE001  (no session: nothing to stop)
            pass

    def close(self) -> None:
        """P5 is going down: stop the run and give the stack lock back."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        if self.lock is not None:
            self.lock.release()

    # ------------------------------------------------------------------ one run
    async def _run(self, demo: Demo, ids: list[int]) -> None:
        t0 = time.monotonic()
        run_dir = self.out_dir / time.strftime("%Y%m%d-%H%M%S")
        rows: list[tuple[int, str, bool, int, str]] = []
        for sid in ids:
            self._emit_step(sid, "idle", "queued")
        self.summary = ""
        self._emit_run("running", f"steps {', '.join(map(str, ids))}", running=True)
        self.lock = self._lock or StackLock.for_profile(self.host.profile())
        got, why = self.lock.acquire()
        if not got:
            for sid in ids:
                self._emit_step(sid, "idle", "")
            self._emit_run("fail", why, running=False)
            return
        self._link_back = False
        try:
            await self._groot_off(demo)
            ready, why = await self._wait_ready(demo)
            if not ready:
                for sid in ids:
                    self._emit_step(sid, "idle", "")
                self._emit_run("fail", why, running=False)
                return
            self._warn_hands(demo, ids)
            for k, sid in enumerate(ids):
                rows.append(await self._step(demo, demo.step(sid), run_dir))
                if k + 1 < len(ids) and demo.between_s > 0:
                    await asyncio.sleep(demo.between_s)
            npass = sum(r[2] for r in rows)
            self._finish(run_dir, rows, t0, "pass" if npass == len(ids) else "fail",
                         f"{npass}/{len(ids)} passed in {self._took(t0)}")
        except SessionGone as e:
            self._abort(ids, rows, f"{e}: demo ended")
            self._finish(run_dir, rows, t0, "fail", f"{e}: demo ended after {self._took(t0)}")
        except asyncio.CancelledError:
            self._abort(ids, rows, "stopped from the page")
            self._finish(run_dir, rows, t0, "fail", f"stopped from the page after {self._took(t0)}"
                         + (f" ({sum(r[2] for r in rows)}/{len(rows)} passed)" if rows else ""))
        except Exception as e:  # noqa: BLE001  (the page must hear that the run ended)
            traceback.print_exc()
            self._abort(ids, rows, f"error: {e}")
            self._finish(run_dir, rows, t0, "fail", f"the demo stopped on an error: {e}")
        finally:
            if self._link_back:
                await self._groot_back()
            self.lock.release()
            self.current, self.line = None, None

    def _groot_applies(self, demo: Demo) -> bool:
        """GR00T off for the run: the steps file says so, the profile is full, and this is the box (the lock is real)."""
        return (demo.groot == "off" and self.host.profile() == "full" and self.lock is not None
                and self.lock.path is not None and (ROOT / "scripts" / "groot_link.sh").is_file())

    async def _groot_off(self, demo: Demo) -> None:
        """scripts/demo.sh --groot off: the link down while the steps run; one that was up comes back after."""
        if not self._groot_applies(demo):
            return
        self._link_back = await asyncio.to_thread(self.sh, ["scripts/groot_link.sh", "check"], 20.0) == 0
        await asyncio.to_thread(self.sh, ["scripts/groot_link.sh", "down"], 20.0)
        self._emit_run("running", "GR00T off for the demo: link down, the pick uses the labelled SONIC arm script",
                       running=True)

    async def _groot_back(self) -> None:
        rc = await asyncio.to_thread(self.sh, ["scripts/groot_link.sh", "ensure"], 120.0)
        if rc != 0:
            self.host.emit({"type": "notice", "text": f"The GR00T link did not come back (groot_link.sh ensure rc {rc}): "
                                                      "run bash scripts/groot_link.sh ensure on the box."})

    @staticmethod
    def _took(t0: float) -> str:
        s = int(time.monotonic() - t0)
        return f"{s // 60} min {s % 60} s"

    def _abort(self, ids: list[int], rows: list, why: str) -> None:
        done = {r[0] for r in rows}
        if self.current is not None and self.current not in done:
            self._emit_step(self.current, "fail", why)
        for sid in ids:
            if sid not in done and sid != self.current:
                self._emit_step(sid, "idle", "")

    def _finish(self, run_dir: Path, rows: list, t0: float, state: str, detail: str) -> None:
        self.current, self.line = None, None
        if rows:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "results.md").write_text(table(rows, int(time.monotonic() - t0)) + "\n")
                detail += f" · {run_dir.name}"
            except OSError:
                pass
        # the task is still finishing: say "not running" explicitly
        self._emit_run(state, detail, running=False)

    async def _wait_ready(self, demo: Demo) -> tuple[bool, str]:
        """tools/say.py --wait-ready before the first line: up to ready_s for System 1 and an idle runtime."""
        t0, last = time.monotonic(), None
        while True:
            ok, why = self.host.ready()
            if ok:
                return True, why
            if time.monotonic() - t0 > demo.ready_s:
                return False, f"not ready after {demo.ready_s:g} s: {why}"
            if why != last:
                self._emit_run("running", f"waiting until System 1 is ready and the robot is idle (now: {why})",
                               running=True)
                last = why
            await asyncio.sleep(self.poll_s)

    def _warn_hands(self, demo: Demo, ids: list[int]) -> None:
        needy = [s.id for s in (demo.step(i) for i in ids) if s.needs and "empty hands" in str(s.needs)]
        if not needy:
            return
        hands = (self.host.end_state() or {}).get("hands") or {}
        held = {a: v.get("holding") for a, v in hands.items() if isinstance(v, dict)
                and v.get("holding") not in (None, "nothing", "", "UNKNOWN")}
        if held:
            self.host.emit({"type": "notice", "text": f"The robot is holding {', '.join(map(str, held.values()))}: "
                            f"step {', '.join(map(str, needy))} will fail (place does not work yet). Fresh restart first."})

    async def _step(self, demo: Demo, step: Step, run_dir: Path) -> tuple[int, str, bool, int, str]:
        self.current = step.id
        t0, wall0 = time.monotonic(), time.time()
        self._emit_step(step.id, "running", step.shows)
        ch = self.host.channel()

        def on(event: str, text: str = "", delay: float = 0.0, answering: str | None = None) -> None:
            phase = {"wait": f"sending in {delay:g} s", "sent": "answering the robot's question" if answering else "said",
                     "idle": "waiting until the robot is idle",
                     "max_s": f"still busy after {step.max_s:g} s: going on"}[event]
            self.line = {"step": step.id, "text": text, "phase": phase}
            self._emit_step(step.id, "running", phase, line=text)

        sent, status = await say_lines(ch, step.lines, idle_s=demo.idle_s, max_s=step.max_s,
                                       ignore_wait=demo.idle_ignores_wait, if_asked=step.if_asked,
                                       poll_s=self.poll_s, on=on)
        d = {"runner": "ui", "status": status, "wall_s": round(time.time() - wall0, 1), "sent": sent,
             "trace": ch.rows(0), "end": self.host.end_state(),
             "config": {"scene": self.host.scene(), "profile": self.host.profile()}}
        good, why = verdict(step, d)
        s = int(time.monotonic() - t0)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / f"step{step.id}.json").write_text(json.dumps(d, default=str))
        except OSError:
            pass
        self.line = None
        self._emit_step(step.id, "pass" if good else "fail", why, s=s)
        return step.id, step.title, good, s, why

    # ------------------------------------------------------------------ recording
    async def set_record(self, on: bool) -> None:
        if self.record["busy"]:
            raise ValueError("the recorder is busy")
        url = self.viz_url()
        if url is None:
            self._emit_record(on=False, detail=self._no_viz())
            return
        if on == self.record["on"]:
            self._emit_record()
            return
        self._emit_record(busy=True, detail="starting the recording…" if on else
                          "stopping: encoding the composite can take a minute…")
        body = {"action": "start", "label": "ui-demo"} if on else {"action": "stop"}
        try:
            rep = await asyncio.to_thread(_post_json, url + "/api/record", body, 30.0 if on else 150.0)
        except (OSError, ValueError, urllib.error.URLError) as e:
            self._emit_record(busy=False, on=False if on else self.record["on"],
                              detail=f"the Sim Viewer did not answer at {url} ({getattr(e, 'reason', e)})")
            return
        if on:
            ok = bool(rep.get("ok"))
            self._emit_record(busy=False, on=ok, dir=rep.get("dir") if ok else self.record["dir"],
                              detail=f"recording to {rep.get('dir')}" if ok else
                              f"the recording did not start: {rep.get('error') or rep}")
        else:
            summary = rep.get("summary") if isinstance(rep.get("summary"), dict) else {}
            where = summary.get("dir") or self.record["dir"]
            self._emit_record(busy=False, on=False, dir=where,
                              detail=(f"recorded {summary.get('duration_s', '?')} s to {where}" if rep.get("ok")
                                      else f"stopped: {rep.get('error') or 'no summary'} ({where})"))

    # ------------------------------------------------------------------ fresh restart
    def fresh_command(self) -> str:
        q = shlex.quote
        profile, scene, offset = self.host.profile(), self.host.scene(), self.host.port_offset()
        port = self.host.port() or (VIZ_PORT + offset)
        session = tmux_session()
        up = (f"bash scripts/m2_up.sh --profile {q(profile)} --scene {q(scene)} --viz low --p5-port {port} "
              f"--session {q(session)}" + (f" --port-offset {offset}" if offset else ""))
        if profile == "full" and (self.demo is None or self.demo.groot == "off"):
            up += " --groot off"       # as scripts/demo.sh --fresh: a GR00T link that is down cannot stop the restart
        body = f"bash scripts/m2_down.sh --session {q(session)} && {up}"
        # P5 runs under m2_p5.sh's taskset (CPUs 4-15): the restart starts from all CPUs, as from an ssh shell
        body = ("command -v taskset >/dev/null && command -v nproc >/dev/null && "
                "taskset -cp \"0-$(( $(nproc --all) - 1 ))\" $$ >/dev/null 2>&1; " + body)
        lock = StackLock.for_profile(profile) if self._lock is None else self._lock
        if lock.path is None:
            return f"echo \"[ui-demo] fresh restart $(date)\"; {body}"
        p = q(str(lock.path))
        return (f"echo \"[ui-demo] fresh restart $(date)\"; mkdir -p {q(str(lock.path.parent))}; "
                f"case \"$(awk '{{print $1}}' {p}/owner 2>/dev/null)\" in {OWNER}) rm -rf {p};; esac; "
                f"if mkdir {p} 2>/dev/null; then echo \"{FRESH_OWNER} $(date +%s)\" > {p}/owner; trap 'rm -rf {p}' EXIT; "
                f"else echo \"[ui-demo] the stack lock is held by '$(cat {p}/owner 2>/dev/null)': not restarting\"; "
                f"exit 4; fi; {body}")

    def fresh(self) -> None:
        info = self.fresh_info()
        if not info["ok"]:
            raise ValueError(f"Fresh restart is {info['why']}")
        if self.running:
            raise ValueError("a demo is running: Stop it first")
        lock = StackLock.for_profile(self.host.profile()) if self._lock is None else self._lock
        who = lock.holder()
        if who and who.split()[0] == FRESH_OWNER:
            raise ValueError("a stack restart is already running (stack lock ui-demo-fresh)")
        if who and who.split()[0] != OWNER:
            raise ValueError(f"the stack lock is held by '{who}': someone is using the stack, not restarting it")
        cmd = self.fresh_command()
        log = self.out_dir / f"fresh-{time.strftime('%Y%m%d-%H%M%S')}.log"
        proc = self.spawn(cmd, log)
        self.fresh_sent = {"pid": getattr(proc, "pid", None), "log": str(log), "t": time.time()}
        self.host.emit({"type": "demo_fresh", "state": "restarting", "detail": FRESH_NOTE, "log": str(log)})
        asyncio.get_running_loop().create_task(self._watch_fresh(proc, log))

    async def _watch_fresh(self, proc: Any, log: Path, every_s: float = 1.0) -> None:
        """Normally m2_down.sh stops this server before the helper ends. If the helper ends first (the lock was
        taken, m2_down failed), the page hears it and can run and restart again."""
        poll = getattr(proc, "poll", None)
        if not callable(poll):
            return
        while (rc := poll()) is None:
            await asyncio.sleep(every_s)
        self.fresh_sent = None
        try:
            tail = [x for x in log.read_text(errors="replace").splitlines() if x.strip()][-1:]
        except OSError:
            tail = []
        self.host.emit({"type": "demo_fresh", "state": "failed", "log": str(log),
                        "detail": f"the restart ended (rc {rc}) with this page still up" + (f": {tail[0][:200]}" if tail else "")})
