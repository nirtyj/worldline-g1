"""The Demo panel (ui/demo.py) and the line timing it shares with tools/say.py (tools.demo_steps.say_lines), against a
pretend runtime; then the websocket protocol through the real page server with the UI fakes. No simulator, no LLM.

  say_lines   "@N" lines go N s after the previous line without waiting for idle; plain lines wait for an idle
              runtime (twice as long while nothing happened); max_s moves on and says so; an if_asked pair is
              answered once
  runner      a run says each step's lines, judges them with the step's PASS rule, streams demo_run, writes the
              evidence and results.md, and holds the stack lock only while it runs; stop sends no further line and
              says "stop" once; one demo at a time; a lock held by someone else stops the run before its first line
  record      the Sim Viewer's POST /api/record, start and stop; no Sim Viewer when this page has its port
  fresh       lite refuses; an Isaac profile spawns m2_down.sh && m2_up.sh with the page's profile, house and port,
              under the stack lock; a helper that ends while the page is up is reported
  protocol    demo_steps on connect; a run through the Hub reaches the runtime by the chat's say path
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tools.demo_steps import say_lines
from ui.demo import DemoRunner, StackLock

STEPS = """
defaults: {idle_s: 0.15, idle_ignores_wait: true, ready_s: 2, between_s: 0}
steps:
  - id: 1
    title: memory note
    shows: a note is kept
    lines: ["my keys are on the counter"]
    max_s: 3
    pass: {rule: note_saved}
    pass_text: a note_saved row
  - id: 2
    title: recall
    shows: recall of the note
    lines: ["where are my keys?"]
    max_s: 3
    pass: {rule: said, pattern: counter}
  - id: 3
    title: walk and stop
    shows: stop mid-walk
    lines: ["go to the dining table", "@0.3 stop", "@0.3 okay, carry on"]
    max_s: 3
    pass: {rule: stop_resume, location: dining_table}
"""


def run(coro, timeout: float = 20):
    return asyncio.run(asyncio.wait_for(coro, timeout))


# ---------------------------------------------------------------------------------------------- a pretend runtime
class Sim:
    """say() records the line (monotonic time) and lets `react` add trace rows and keep the runtime busy."""

    def __init__(self, react=None) -> None:
        self.rows: list[dict] = []
        self.said: list[tuple[float, str]] = []
        self.busy_until = 0.0
        self.react = react or (lambda sim, text: None)
        self.hands: dict = {"left": {"holding": None}, "right": {"holding": None}}

    async def say(self, text: str) -> None:
        self.said.append((time.monotonic(), text))
        self.react(self, text)

    def row(self, **r: Any) -> None:
        self.rows.append({"t": time.monotonic(), **r})

    def busy(self, ignore_wait: bool) -> bool:
        return time.monotonic() < self.busy_until


class Chan:
    def __init__(self, sim: Sim) -> None:
        self.sim, self.base = sim, len(sim.rows)

    async def say(self, text: str) -> None:
        await self.sim.say(text)

    def count(self) -> int:
        return len(self.sim.rows) - self.base

    def rows(self, start: int) -> list[dict]:
        return self.sim.rows[self.base + start:]

    def busy(self, ignore_wait: bool) -> bool:
        return self.sim.busy(ignore_wait)


class Host:
    def __init__(self, sim: Sim, profile: str = "sonic", port: int | None = 8766) -> None:
        self.sim, self._profile, self._port = sim, profile, port
        self.msgs: list[dict] = []
        self.live = True

    def session(self):
        return self if self.live else None

    def channel(self) -> Chan:
        return Chan(self.sim)

    def ready(self):
        return (not self.sim.busy(True)), "System 1 ready, runtime " + ("busy" if self.sim.busy(True) else "idle")

    def end_state(self) -> dict:
        return {"hands": self.sim.hands}

    async def say(self, text: str) -> None:
        await self.sim.say(text)

    def emit(self, msg: dict) -> None:
        self.msgs.append(msg)

    def profile(self) -> str:
        return self._profile

    def scene(self) -> str:
        return "procthor-train-40"

    def port_offset(self) -> int:
        return 0

    def port(self):
        return self._port

    def runs(self) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == "demo_run"]


def react(sim: Sim, text: str) -> None:
    """A runtime that answers: a note, a recall, a walk that a stop and a resume interrupt."""
    sim.row(type="heard", text=text)
    if text.startswith("my keys"):
        sim.row(type="decision", tool="remember")
        sim.row(type="note_saved", text=text)
        sim.busy_until = time.monotonic() + 0.2
    elif text.startswith("where are my keys"):
        sim.row(type="result", kind="speech", t_start=time.monotonic(), text="On the kitchen counter.")
    elif text.startswith("go to"):
        sim.row(type="decision", tool="navigate")
        sim.busy_until = time.monotonic() + 30          # a long walk: only the timed lines interrupt it
    elif text == "stop":
        sim.row(type="stop", canceled=["navigate"])
        sim.busy_until = 0.0
    elif text.startswith("okay"):
        sim.row(type="resume")
        sim.row(type="result", tool="navigate", status="succeeded", data={"location": "kitchen_dining_table_1a"},
                summary="arrived at kitchen_dining_table_1a")


def _runner(tmp_path: Path, sim: Sim, **kw) -> tuple[DemoRunner, Host]:
    (tmp_path / "steps.yaml").write_text(STEPS)
    host = kw.pop("host", None) or Host(sim)
    r = DemoRunner(host, steps_file=tmp_path / "steps.yaml", out_dir=tmp_path / "out",
                   lock=kw.pop("lock", StackLock(tmp_path / "locks" / "stack.d")), poll_s=0.03, **kw)
    return r, host


# ---------------------------------------------------------------------------------------------- say_lines
def test_timed_lines_do_not_wait_for_idle():
    sim = Sim(react)

    async def go():
        t0 = time.monotonic()
        sent, status = await say_lines(Chan(sim), ["go to the dining table", "@0.3 stop", "@0.2 okay, carry on"],
                                       idle_s=0.1, max_s=5, ignore_wait=True, poll_s=0.02)
        return t0, sent, status

    t0, sent, status = run(go())
    assert status == "ok" and [s["text"] for s in sent] == ["go to the dining table", "stop", "okay, carry on"]
    (a, _), (b, _), (c, _) = sim.said
    assert 0.28 <= b - a < 0.6, "the stop goes 0.3 s after the walk started, not after the 30 s walk"
    assert 0.18 <= c - b < 0.5
    assert [s["trace_at"] for s in sent] == [0, 2, 4]


def test_plain_lines_wait_for_idle_twice_as_long_when_nothing_happened():
    def busy_then_result(sim: Sim, text: str) -> None:
        if text == "a":
            sim.busy_until = time.monotonic() + 0.3
            sim.row(type="result", tool="observe", status="succeeded")
    sim = Sim(busy_then_result)
    sent, status = run(say_lines(Chan(sim), ["a", "b", "c"], idle_s=0.2, max_s=5, ignore_wait=False, poll_s=0.02))
    (ta, _), (tb, _), (tc, _) = sim.said
    assert 0.48 <= tb - ta < 0.9, "b waits for the 0.3 s of work and then 0.2 s idle"
    assert 0.38 <= tc - tb < 0.8, "after b nothing happened: 2 x idle_s"
    assert status == "ok"


def test_max_s_moves_on_and_if_asked_answers_once():
    def asks(sim: Sim, text: str) -> None:
        if text == "pick up the bottle":
            sim.row(type="result", kind="speech", text="Which one do you mean?")
            sim.row(type="result", kind="speech", text="Which one, the white or the green?")
            sim.busy_until = time.monotonic() + 60
    sim = Sim(asks)
    events: list[tuple] = []
    sent, status = run(say_lines(Chan(sim), ["pick up the bottle"], idle_s=0.1, max_s=0.4, ignore_wait=True,
                                 if_asked=[("which (one|bottle)", "the white one")], poll_s=0.02,
                                 on=lambda e, **k: events.append((e, k.get("text")))))
    assert status == "max_s"
    assert [s["text"] for s in sent] == ["pick up the bottle", "the white one"]
    assert sent[1]["answering"] == "which (one|bottle)"
    assert ("max_s", "pick up the bottle") in events and ("sent", "the white one") in events


# ---------------------------------------------------------------------------------------------- the runner
def test_run_all_judges_each_step_and_holds_the_lock_while_it_runs(tmp_path):
    sim = Sim(react)
    r, host = _runner(tmp_path, sim)
    lock = tmp_path / "locks" / "stack.d"

    async def go():
        r.start("all")
        held = []
        while r.running:
            held.append(lock.is_dir() and (lock / "owner").read_text().split()[0])
            await asyncio.sleep(0.02)
        return held

    held = run(go())
    assert "ui-demo" in held and not lock.exists(), "taken while running, given back after"
    assert [t for _, t in sim.said] == ["my keys are on the counter", "where are my keys?", "go to the dining table",
                                        "stop", "okay, carry on"]
    ends = {m["step"]: m for m in host.runs() if m["step"] is not None and m["state"] in ("pass", "fail")}
    assert {k: v["state"] for k, v in ends.items()} == {1: "pass", 2: "pass", 3: "pass"}
    assert ends[3]["detail"].startswith("stop=1 (canceled ['navigate']), resume=1, arrived after resume")
    lines = [m.get("line") for m in host.runs() if m.get("line")]
    assert lines[:2] == ["my keys are on the counter", "my keys are on the counter"]     # said, then waiting for idle
    assert "@0.3 stop" not in lines and "stop" in lines
    last = host.runs()[-1]
    assert last["step"] is None and last["state"] == "pass" and last["running"] is False
    assert last["detail"].startswith("3/3 passed in 0 min")
    out = next((tmp_path / "out").iterdir())
    ev = json.loads((out / "step3.json").read_text())
    assert [s["text"] for s in ev["sent"]] == ["go to the dining table", "stop", "okay, carry on"]
    assert ev["trace"][0] == {**ev["trace"][0], "type": "heard", "text": "go to the dining table"}
    assert "| 3 | walk and stop | PASS |" in (out / "results.md").read_text()
    snap = r.steps_message()
    assert [s["id"] for s in snap["steps"]] == [1, 2, 3] and snap["state"]["steps"]["2"]["state"] == "pass"


def test_a_failing_step_fails_with_its_evidence(tmp_path):
    sim = Sim(lambda s, t: s.row(type="result", kind="speech", text="I have no idea."))
    r, host = _runner(tmp_path, sim)

    async def go():
        r.start([2])
        await r.task

    run(go())
    end = [m for m in host.runs() if m["step"] == 2][-1]
    assert end["state"] == "fail" and end["detail"] == "said: 'I have no idea.'"
    assert host.runs()[-1]["state"] == "fail" and host.runs()[-1]["detail"].startswith("0/1 passed")


def test_stop_sends_no_further_line_and_says_stop_once(tmp_path):
    sim = Sim(react)
    r, host = _runner(tmp_path, sim)
    (tmp_path / "steps.yaml").write_text(STEPS.replace('"@0.3 stop", "@0.3 okay, carry on"', '"@5 okay, carry on"'))

    async def go():
        r.start([3, 1])
        while not sim.said:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        await r.handle({"action": "stop"})
        await asyncio.sleep(0.2)

    run(go())
    assert [t for _, t in sim.said] == ["go to the dining table", "stop"], "the timed line never went"
    ends = [m for m in host.runs() if m["state"] != "running"]
    assert {"step": 3, "state": "fail", "detail": "stopped from the page"}.items() <= ends[-3].items() or \
        any(m["step"] == 3 and m["state"] == "fail" and m["detail"] == "stopped from the page" for m in ends)
    assert any(m["step"] == 1 and m["state"] == "idle" for m in ends), "the step that never ran is back to idle"
    assert ends[-1]["step"] is None and ends[-1]["running"] is False and "stopped from the page" in ends[-1]["detail"]
    assert not r.running and not (tmp_path / "locks" / "stack.d").exists()


def test_one_demo_at_a_time_and_bad_steps(tmp_path):
    sim = Sim(react)
    r, host = _runner(tmp_path, sim)

    async def go():
        with pytest.raises(ValueError, match="no demo step 7"):
            r.start([7])
        with pytest.raises(ValueError, match="list of step ids"):
            r.start("some")
        r.start([3])
        with pytest.raises(ValueError, match="already running"):
            r.start("all")
        await r.handle({"action": "stop"})
        host.live = False
        with pytest.raises(ValueError, match="no session"):
            r.start("all")

    run(go())


def test_a_lock_held_by_someone_else_stops_the_run(tmp_path):
    sim = Sim(react)
    lock = tmp_path / "locks" / "stack.d"
    lock.mkdir(parents=True)
    (lock / "owner").write_text("demoscript 1790723009\n")
    r, host = _runner(tmp_path, sim)

    async def go():
        r.start("all")
        await r.task

    run(go())
    assert sim.said == [] and (lock / "owner").read_text().startswith("demoscript"), "never taken, never removed"
    last = host.runs()[-1]
    assert last["state"] == "fail" and "held by 'demoscript 1790723009'" in last["detail"]


def test_stack_lock(tmp_path, monkeypatch):
    p = tmp_path / "l" / "stack.d"
    a = StackLock(p)
    assert a.acquire()[0] and (p / "owner").read_text().startswith("ui-demo ")
    b = StackLock(p)                                   # a lock left under our own name is taken over
    ok, why = b.acquire()
    assert ok and why.startswith("took over")
    b.release()
    assert not p.exists()
    p.mkdir(parents=True)
    (p / "owner").write_text("ops 1\n")
    c = StackLock(p)
    assert c.acquire()[0] is False and c.holder() == "ops 1"
    c.release()
    assert p.exists(), "never removes someone else's lock"
    assert StackLock(None).acquire() == (True, "no stack lock here")
    monkeypatch.setenv("WL_STACK_LOCK", "off")
    assert StackLock.for_profile("full").path is None
    monkeypatch.delenv("WL_STACK_LOCK")
    assert StackLock.for_profile("lite").path is None


def test_wait_ready_gives_up(tmp_path):
    sim = Sim(react)
    sim.busy_until = time.monotonic() + 60
    r, host = _runner(tmp_path, sim)

    async def go():
        r.start([1])
        await r.task

    run(go())
    assert sim.said == []
    assert host.runs()[-1]["detail"].startswith("not ready after 2 s: System 1 ready, runtime busy")
    assert all(m["state"] == "idle" for m in host.runs() if m["step"] == 1 and m is not host.runs()[0])


def test_groot_off_on_the_full_profile_takes_the_link_down_and_back(tmp_path):
    """scripts/demo.sh --groot off (the steps file's default): link down for the run, ensure after, on full only."""
    calls: list[str] = []

    def sh(args, timeout):
        calls.append(args[-1])
        return 0

    sim = Sim(react)
    r, host = _runner(tmp_path, sim, host=Host(sim, profile="full"), sh=sh)

    async def go():
        r.start([1])
        await r.task
        assert calls == ["check", "down", "ensure"]
        assert any("GR00T off for the demo" in m["detail"] for m in host.runs())
        calls.clear()
        r.start([3])                                     # stopped mid-step: the link still comes back
        while not sim.said or sim.said[-1][1] != "go to the dining table":
            await asyncio.sleep(0.01)
        await r.handle({"action": "stop"})
        assert calls == ["check", "down", "ensure"]

    run(go())
    calls.clear()
    r2, _ = _runner(tmp_path, Sim(react), host=Host(Sim(react), profile="sonic"), sh=sh)
    run(_run_one(r2, [1]))
    assert calls == [], "only the full profile uses GR00T"
    (tmp_path / "steps.yaml").write_text(STEPS.replace("between_s: 0}", "between_s: 0, groot: on}"))
    r3 = DemoRunner(Host(Sim(react), profile="full"), steps_file=tmp_path / "steps.yaml", out_dir=tmp_path / "out",
                    lock=StackLock(tmp_path / "locks" / "stack.d"), poll_s=0.03, sh=sh)
    run(_run_one(r3, [1]))
    assert calls == [], "groot: on keeps the link"
    down_calls: list[str] = []
    r4, _ = _runner(tmp_path, Sim(react), host=Host(Sim(react), profile="full"), lock=StackLock(None),
                    sh=lambda a, t: down_calls.append(a[-1]) or 0)
    run(_run_one(r4, [1]))
    assert down_calls == [], "not on the box (no lock): the laptop has no link to take down"


async def _run_one(r: DemoRunner, ids: list[int]) -> None:
    r.start(ids)
    await r.task


# ---------------------------------------------------------------------------------------------- recording
class _Viz(http.server.BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Viz.calls.append(body)
        rep = ({"ok": True, "recording": True, "dir": "/rec/run-1"} if body["action"] == "start"
               else {"ok": True, "summary": {"dir": "/rec/run-1", "duration_s": 12.5}})
        data = json.dumps(rep).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_record_start_and_stop(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Viz)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Viz.calls = []
    try:
        r, host = _runner(tmp_path, Sim(), viz_url=f"http://127.0.0.1:{srv.server_address[1]}")

        async def go():
            await r.handle({"action": "record", "on": True})
            with pytest.raises(ValueError, match="busy"):
                await r.handle({"action": "record", "on": False})
            await r._rec_task
            await r.handle({"action": "record", "on": False})
            await r._rec_task

        run(go())
    finally:
        srv.shutdown()
    assert _Viz.calls == [{"action": "start", "label": "ui-demo"}, {"action": "stop"}]
    recs = [m for m in host.msgs if m["type"] == "demo_record"]
    assert [(m["on"], m["busy"]) for m in recs] == [(False, True), (True, False), (True, True), (False, False)]
    assert recs[1]["detail"] == "recording to /rec/run-1" and recs[-1]["detail"] == "recorded 12.5 s to /rec/run-1"


def test_record_without_a_sim_viewer(tmp_path, monkeypatch):
    monkeypatch.delenv("WL_VIZ_URL", raising=False)
    r, host = _runner(tmp_path, Sim(), host=Host(Sim(), profile="sonic", port=8765))
    assert r.viz_url() is None and r.steps_message()["record"]["available"] is False
    run(r.set_record(True))
    assert host.msgs[-1]["on"] is False and "this page itself is on :8765" in host.msgs[-1]["detail"]
    lite, host_l = _runner(tmp_path, Sim(), host=Host(Sim(), profile="lite", port=8791))
    assert lite.viz_url() is None, "lite has no sim to record; the laptop's 8765 may be a tunnel to the box"
    assert lite.steps_message()["record"]["viz"] == "no Sim Viewer on the lite profile"
    r2, host2 = _runner(tmp_path, Sim(), viz_url="http://127.0.0.1:9")      # nothing listens there
    run(r2.set_record(True))
    assert host2.msgs[-1]["on"] is False and host2.msgs[-1]["detail"].startswith("the Sim Viewer did not answer")


# ---------------------------------------------------------------------------------------------- fresh restart
class _Proc:
    def __init__(self) -> None:
        self.pid, self.rc = 4242, None

    def poll(self):
        return self.rc


def test_fresh_on_lite_is_refused(tmp_path):
    r, host = _runner(tmp_path, Sim(), host=Host(Sim(), profile="lite"))
    assert r.steps_message()["fresh"]["ok"] is False
    with pytest.raises(ValueError, match="only on the Isaac profiles"):
        run(r.handle({"action": "fresh"}))


def test_fresh_spawns_the_restart_under_the_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("WL_SESSION", "wl-m2")
    spawned: list[tuple[str, Path]] = []
    proc = _Proc()

    def spawn(cmd: str, log: Path):
        spawned.append((cmd, log))
        return proc

    r, host = _runner(tmp_path, Sim(), spawn=spawn)

    async def go():
        await r.handle({"action": "fresh"})
        with pytest.raises(ValueError, match="restarting"):
            r.start("all")
        proc.rc = 4                                      # the helper ended with the page still up
        await asyncio.sleep(1.3)

    run(go())
    (cmd, log), = spawned
    assert ("bash scripts/m2_down.sh --session wl-m2 && bash scripts/m2_up.sh --profile sonic "
            "--scene procthor-train-40 --viz low --p5-port 8766 --session wl-m2") in cmd
    assert "ui-demo-fresh" in cmd and "trap" in cmd and str(tmp_path / "locks" / "stack.d") in cmd
    assert log.parent == tmp_path / "out" and log.name.startswith("fresh-")
    fr = [m for m in host.msgs if m["type"] == "demo_fresh"]
    assert fr[0]["state"] == "restarting" and fr[0]["detail"].startswith("restarting the stack")
    assert fr[-1]["state"] == "failed" and "rc 4" in fr[-1]["detail"]
    assert r.fresh_sent is None


def test_fresh_refused_while_running_or_locked(tmp_path):
    sim = Sim(react)
    r, host = _runner(tmp_path, sim, spawn=lambda c, l: pytest.fail("must not spawn"))

    async def go():
        r.start([3])
        with pytest.raises(ValueError, match="Stop it first"):
            r.fresh()
        await r.handle({"action": "stop"})

    run(go())
    lock = tmp_path / "locks" / "stack.d"
    lock.mkdir(parents=True)
    (lock / "owner").write_text("ops 1\n")
    with pytest.raises(ValueError, match="held by 'ops 1'"):
        r.fresh()


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="no bash")
def test_fresh_command_runs_the_lock_dance(tmp_path):
    """The helper's shell, with echo stand-ins for the stack scripts: takes the lock as ui-demo-fresh, runs down and
    up, gives the lock back; a lock held by someone else stops it (exit 4)."""
    import subprocess
    r, _ = _runner(tmp_path, Sim())
    cmd = r.fresh_command().replace("bash scripts/m2_down.sh", "echo DOWN; cat " + str(tmp_path / "locks/stack.d/owner")
                                    + "; echo").replace("bash scripts/m2_up.sh", "echo UP")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert out.returncode == 0 and "DOWN" in out.stdout and "ui-demo-fresh" in out.stdout and "UP --profile" in out.stdout
    assert not (tmp_path / "locks" / "stack.d").exists()
    (tmp_path / "locks" / "stack.d").mkdir(parents=True)
    (tmp_path / "locks" / "stack.d" / "owner").write_text("ops 1\n")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert out.returncode == 4 and "DOWN" not in out.stdout and "held by 'ops 1'" in out.stdout


# ---------------------------------------------------------------------------------------------- the websocket protocol
websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

from ui_fakes import FakeExecution, FakeRuntime, FakeTap, make_deps  # noqa: E402


class AnsweringRuntime(FakeRuntime):
    """The UI fakes' runtime, but it answers: a speech result naming the counter, and a walk that finishes."""

    async def run(self) -> None:
        while True:
            utt = await self.user.next()
            self.heard.append(utt)
            now = self.clock.now()
            self.tracer.rows.append({"t": now, "type": "heard", "id": utt.id, "text": utt.text})
            self.tracer.rows.append({"t": now, "type": "classified", "id": utt.id, "kind": "question",
                                     "directive": utt.directive})
            self.tracer.rows.append({"t": now, "type": "result", "kind": "speech", "t_start": now,
                                     "text": "They are on the kitchen counter."})
            self.history.append(FakeExecution(f"s-{len(self.history)}", "speak", {}, status="succeeded"))


def test_protocol_through_the_page_server(tmp_path):
    from ui.cameras import TapCameras
    from ui.server import Hub, serve_hub

    async def recv(ws, kind: str, pred=lambda m: True, timeout: float = 5.0) -> dict:
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if m.get("type") == kind and pred(m):
                return m

    async def go():
        built: dict = {}
        deps = make_deps(built)
        deps.create_runtime = lambda spec, robot, user, brain, clock: built.setdefault(
            "runtime", AnsweringRuntime(robot, user, brain, clock))
        hub = Hub("procthor-train-40", "sonic", deps=deps, cameras=TapCameras(FakeTap()), system1="off")
        server = await serve_hub(hub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        (tmp_path / "steps.yaml").write_text(STEPS)
        hub.demo = DemoRunner(hub.demo.host, steps_file=tmp_path / "steps.yaml", out_dir=tmp_path / "out",
                              lock=StackLock(None), poll_s=0.03)
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                steps = await recv(ws, "demo_steps")
                assert [s["id"] for s in steps["steps"]] == [1, 2, 3] and steps["error"] is None
                assert steps["fresh"]["ok"] and steps["state"]["running"] is False
                assert steps["record"]["available"] is True and steps["record"]["viz"] == "http://127.0.0.1:8765"
                await ws.send(json.dumps({"type": "demo", "action": "run", "steps": [2]}))
                first = await recv(ws, "demo_run", lambda m: m.get("line"))
                assert first == {**first, "step": 2, "state": "running", "line": "where are my keys?", "running": True}
                await ws.send(json.dumps({"type": "demo", "action": "run", "steps": "all"}))
                note = await recv(ws, "notice")
                assert "already running" in note["text"]
                done = await recv(ws, "demo_run", lambda m: m["step"] == 2 and m["state"] != "running")
                assert done["state"] == "pass" and done["detail"] == "said: 'They are on the kitchen counter.'"
                end = await recv(ws, "demo_run", lambda m: m["step"] is None and not m["running"])
                assert end["state"] == "pass"
                rt = built["runtime"]
                assert [u.text for u in rt.heard] == ["where are my keys?"]
                assert any(e["type"] == "utterance" and e["text"] == "where are my keys?" for e in hub.session.log.events), \
                    "the chat shows it like a typed line"
                await ws.send(json.dumps({"type": "demo", "action": "bogus"}))
                assert "unknown demo action" in (await recv(ws, "notice"))["text"]
                await ws.send(json.dumps({"type": "hello"}))
                again = await recv(ws, "demo_steps")
                assert again["state"]["steps"]["2"]["state"] == "pass"
        finally:
            server.close()
            await hub.session.stop()
            hub.close()

    run(go(), 30)


def test_page_has_the_demo_panel():
    html = (Path(__file__).resolve().parents[2] / "ui" / "index.html").read_text()
    for needle in ('id="btn-demo"', 'id="demo-all"', 'id="demo-stop"', 'id="demo-rec"', 'id="demo-fresh"',
                   'id="demo-confirm"', 'id="demo-steps"', '"demo_steps"', '"demo_run"', '"demo_record"',
                   '"demo_fresh"', 'action: "run"', 'action: "stop"', 'action: "record"', 'action: "fresh"'):
        assert needle in html, needle
    assert html.count("window.confirm(") == 1, "the fresh restart confirms in the page, only Kill uses confirm()"
