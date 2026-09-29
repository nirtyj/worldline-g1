"""The live-stack fault injections of eval/stack_suite.py (docs/eval_hooks.md), offline:

- eval/hooks.py's hook maps and HookInjector's log (the last JSON line of a hook, exit codes, timeouts);
- tools/hooks/delay_proxy.py on real ZMQ sockets (pass-through, then +300 ms, control socket);
- tools/hooks/obstacle.py on synthetic houses (one door: the box cuts every route; two doors: it says so);
- tools/hooks/cli.py against HookableFakeP1 (the test ops on the fake P1) + the fake deploy + the REAL body service:
  a 250 N push collapses the fake robot, the body latches fault `fallen`, `recover` runs the body's path A;
- the G7 / G8 / G9 scorers on a scripted page: the stack's own recovery is scored inside the window, the operator
  fixture after it is recorded and never scored.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import threading
import time
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("zmq")
pytest.importorskip("websockets")

import zmq  # noqa: E402

from eval import hooks as H  # noqa: E402
from eval import stack_suite as ss  # noqa: E402
from test_suite_offline import H40_LAYOUT, H40_OBJECTS, FakePage  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------------------------------- hook maps
def test_the_box_map_covers_every_injection_the_scenarios_need():
    hooks = H.preset("box")
    needed = {n.split(":", 1)[1] for s in ss.SPECS for n in s.needs if n.startswith("hook:")}
    assert needed <= set(hooks), needed - set(hooks)
    assert set(H.FAULTS) | set(H.FIXTURES) <= set(hooks)
    # the placeholders the scenarios pass, exactly as HookInjector formats them
    assert "--force-n 250 " in hooks["push_robot"].format(newtons="250")
    assert "--rtf 0.9 " in hooks["throttle_rtf"].format(rtf="0.9")
    assert "--ms 600" in hooks["delay_proxy_on"].format(ms="600")
    assert "groot_server.sh stop --port 5550" in hooks["kill_policy"]           # P4 on the MAIN box (PLAN 0.11)
    laptop = H.preset("laptop")
    assert laptop["push_robot"].format(newtons="250").count("--force-n 250") == 1
    assert laptop["kill_deploy"].startswith("BREV_NAME=ludo-g1-brev2 ") and "ssh.sh" in laptop["kill_deploy"]
    assert H.preset("none") == {}
    with pytest.raises(ValueError):
        H.preset("dev")


def test_hook_injector_keeps_each_runs_json_exit_code_and_time(monkeypatch):
    inj = ss.HookInjector({
        "good": "echo progress; echo '{{\"ok\": true, \"n\": {n}}}'",
        "bad": "echo '{{\"ok\": false, \"error\": \"nope\"}}'; exit 1",
        "slow": "sleep 5"})
    ok, note = asyncio.run(inj.do("good", n=3))
    assert ok and '"n": 3' in note
    ok, note = asyncio.run(inj.do("bad"))
    assert not ok and "nope" in note and "exit 1" in note
    monkeypatch.setattr(ss.HookInjector, "TIMEOUT_S", 0.3)
    ok, note = asyncio.run(inj.do("slow"))
    assert not ok and "timed out" in note
    assert [x["name"] for x in inj.log] == ["good", "bad", "slow"]
    assert inj.log[0]["json"] == {"ok": True, "n": 3} and inj.log[0]["rc"] == 0 and inj.log[2]["rc"] is None
    assert inj.last("bad")["json"]["error"] == "nope"
    ok, note = asyncio.run(inj.do("missing"))
    assert not ok and "no --hook missing" in note
    assert ss.last_json("a\n{not json}\n{\"x\": 1}\ntrailing") == {"x": 1} and ss.last_json("") is None


# ---------------------------------------------------------------------------------------------------- delay proxy
def test_delay_proxy_passes_through_then_holds_replies():
    from tools.hooks.delay_proxy import DelayProxy, ctl_call
    ctx = zmq.Context()
    up, lis, ctl = _free_port(), _free_port(), _free_port()
    stop = threading.Event()

    def echo_server():                       # a REP like the PolicyServer: one request at a time
        s = ctx.socket(zmq.REP)
        s.bind(f"tcp://127.0.0.1:{up}")
        while not stop.is_set():
            if s.poll(50):
                s.send(b"re:" + s.recv())
        s.close(0)
    th = threading.Thread(target=echo_server, daemon=True)
    th.start()
    px = DelayProxy(f"tcp://127.0.0.1:{lis}", f"tcp://127.0.0.1:{up}", f"tcp://127.0.0.1:{ctl}", 0.0, ctx=ctx,
                    log=lambda *_: None)
    tp = threading.Thread(target=px.run, daemon=True)
    tp.start()
    c = ctx.socket(zmq.REQ)
    c.setsockopt(zmq.RCVTIMEO, 3000)
    c.connect(f"tcp://127.0.0.1:{lis}")
    try:
        def rtt(msg: bytes) -> float:
            t0 = time.monotonic()
            c.send(msg)
            assert c.recv() == b"re:" + msg
            return time.monotonic() - t0
        rtt(b"warm")
        assert rtt(b"a") < 0.15
        assert ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "set", "ms": 300}, ctx=ctx)["ms"] == 300
        assert 0.29 <= rtt(b"b") < 0.9
        st = ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "get"}, ctx=ctx)
        assert st["requests"] == 3 and st["replies"] == 3 and st["held"] >= 1 and st["max_hold_ms"] >= 290
        assert ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "set", "ms": -1}, ctx=ctx)["ok"] is False
        ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "set", "ms": 0}, ctx=ctx)
        assert rtt(b"c") < 0.15
        ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "stop"}, ctx=ctx)
        tp.join(3)
        assert not tp.is_alive()
        assert ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "get"}, timeout_s=0.3, ctx=ctx)["ok"] is False
    finally:
        stop.set()
        c.close(0)
        th.join(2)
        ctx.term()


def test_delay_proxy_drops_requests_while_the_policy_server_is_down():
    """G5/G13 kill P4 behind the proxy: a request then times out at the client (as on a direct link) and is not
    queued for the server that comes back later."""
    from tools.hooks.delay_proxy import DelayProxy, ctl_call
    ctx = zmq.Context()
    up, lis, ctl = _free_port(), _free_port(), _free_port()
    px = DelayProxy(f"tcp://127.0.0.1:{lis}", f"tcp://127.0.0.1:{up}", f"tcp://127.0.0.1:{ctl}", 0.0, ctx=ctx,
                    log=lambda *_: None)
    tp = threading.Thread(target=px.run, daemon=True)
    tp.start()
    c = ctx.socket(zmq.REQ)
    c.setsockopt(zmq.RCVTIMEO, 400)
    c.setsockopt(zmq.LINGER, 0)
    c.connect(f"tcp://127.0.0.1:{lis}")
    try:
        c.send(b"x")
        with pytest.raises(zmq.Again):
            c.recv()
        st = ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "get"}, ctx=ctx)
        assert st["requests"] == 1 and st["dropped"] == 1 and st["replies"] == 0
    finally:
        ctl_call(f"tcp://127.0.0.1:{ctl}", {"op": "stop"}, ctx=ctx)
        tp.join(3)
        c.close(0)
        ctx.term()


# ---------------------------------------------------------------------------------------------------- obstacle
def _house(doors):
    """8 x 4 m, a wall at x = 4 m with the given door intervals in y; 0.05 m cells."""
    from body.nav_grid import NavGrid
    res = 0.05
    occ = np.zeros((80, 160), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    occ[:, 79:81] = 1
    for y0, y1 in doors:
        occ[int(y0 / res):int(y1 / res), 79:81] = 0
    return NavGrid(occ, res, (0.0, 0.0), robot_radius=0.25)


def test_the_box_goes_across_the_only_door_and_cuts_every_route():
    from tools.hooks.obstacle import place_across_path
    g = _house([(1.5, 2.5)])
    p = place_across_path(g, (1.0, 2.0), (7.0, 2.0), (0.3, 1.6, 1.2))
    assert p["blocks_all_routes"] is True and abs(p["x"] - 4.0) < 0.25 and abs(p["y"] - 2.0) < 0.3
    assert abs(np.cos(p["yaw"])) > 0.9                          # thin side along the path (+x), long side across


def test_a_second_door_is_reported_not_hidden():
    from tools.hooks.obstacle import place_across_path
    g = _house([(0.4, 1.3), (2.8, 3.6)])
    p = place_across_path(g, (1.0, 1.0), (7.0, 1.0), (0.3, 1.6, 1.2))
    assert p["blocks_all_routes"] is False and p["replan_with_box"] == "a detour exists"
    with pytest.raises(ValueError):
        place_across_path(g, (1.0, 1.0), (1.5, 1.0), (0.3, 1.6, 1.2))     # too short for a box


# ---------------------------------------------------------------------------------------------------- the CLI
def _cli(off, *args):
    from tools.hooks import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["--port-offset", str(off), "--session", "hooks-test", *args])
    out = json.loads(buf.getvalue().strip().splitlines()[-1])
    return rc, out


def _free_offset() -> int:
    import random

    from body.config import BASE_PORTS
    for _ in range(200):
        off = random.randrange(20000, 40000, 100)
        ok = True
        for p in BASE_PORTS.values():
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p + off))
            except OSError:
                ok = False
            finally:
                s.close()
            if not ok:
                break
        if ok:
            return off
    raise RuntimeError("no free port block")


@pytest.fixture
def fake_stack(tmp_path, monkeypatch):
    from body.client import BodyClient
    from body.config import BodyConfig
    from body.service import BodyService
    from tools.fake_deploy import FakeDeploy
    from tools.hooks import cli
    from tools.hooks.fake_ops import HookableFakeP1
    off = _free_offset()
    p1 = HookableFakeP1(off, str(tmp_path / "p1"), log=lambda *_: None, rtf=1.0).start()
    dep = FakeDeploy(off, log=lambda *_: None).start()
    svc = BodyService(BodyConfig(port_offset=off), log_dir=str(tmp_path / "body"), log=None)
    th = threading.Thread(target=svc.run, daemon=True)
    th.start()
    bc = BodyClient(port_offset=off).connect(15)
    t0 = time.monotonic()
    while svc.pose_sub.latest() is None and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    h = bc.stand(release_band=True, settle_s=0.5, verify_s=1.0)
    assert h.ok, h.result
    # the fake deploy is a thread, not a g1_deploy_onnx_ref process: stand in for its pid (never a real one)
    monkeypatch.setattr(cli, "deploy_pid", lambda session, port: 424242)
    yield off, p1, svc, bc
    bc.close()
    svc.stop()
    th.join(5)
    dep.stop()
    p1.stop()


def test_hooks_cli_push_fall_recover_throttle_box_on_the_fake_stack(fake_stack):
    off, p1, svc, bc = fake_stack
    rc, out = _cli(off, "status")
    assert rc == 0 and out["p1"]["active"] is False and out["body"]["fault"] is None and not out["hooks_active"]

    # G7's push: the fake robot collapses, the real body latches fault `fallen` and puts the band on
    rc, out = _cli(off, "push", "--force-n", "250", "--dir", "left", "--watch-s", "1.0")
    assert rc == 0 and out["fell"] is True and out["p1"]["fake_fell"] is True
    assert out["p1"]["push"]["force_w"][1] != 0.0
    t0 = time.monotonic()
    while bc.status().get("fault") != "fallen" and time.monotonic() - t0 < 5:
        time.sleep(0.1)
    assert bc.status()["fault"] == "fallen"

    # the operator's recovery: the body's own path A (band, reset, stabilise, release, watch)
    rc, out = _cli(off, "recover", "--timeout", "60")
    assert rc == 0, out
    assert out["path"].startswith("A") and out["steps"][0]["path"] == "A" and out["recoveries"] == 1
    st = bc.status()
    assert st["fault"] is None and not p1.collapsed and p1.band_on is False
    rc, out = _cli(off, "recover")
    assert rc == 0 and out["path"] == "none needed"

    # G9's throttle, G6's box, and clear-all leaves nothing active
    rc, out = _cli(off, "throttle", "--rtf", "0.9", "--duration-s", "60")
    assert rc == 0 and p1.rtf_override == 0.9 and out["p1"]["throttle"]["target"] == 0.9
    rc, out = _cli(off, "spawn-box", "--x", "5.0", "--y", "1.5", "--yaw", "0.0")
    assert rc == 0 and out["p1"]["box"]["cells"] > 0
    rc, out = _cli(off, "status")
    assert out["hooks_active"] is True and out["p1"]["throttle"]["target"] == 0.9 and out["p1"]["box"]["x"] == 5.0
    rc, out = _cli(off, "clear-all")
    assert rc == 0 and out["active_after"] is False and out["unthrottle"] and out["clear_box"]
    assert p1.rtf_override == 1.0 and p1.box is None
    rc, out = _cli(off, "spawn-box", "--toward-xy", "1.0,1.0", "--x", "1", "--y", "1", "--min-from-robot", "50")
    assert rc == 1 and "too short" in out["error"]


def test_kill_deploy_never_touches_a_deploy_of_another_stack(monkeypatch):
    from tools.hooks import cli
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "", "returncode": 0})())
    rc, out = _cli(_free_offset(), "kill-deploy")
    assert rc == 1 and "no g1_deploy_onnx_ref running" in out["error"]


# ---------------------------------------------------------------------------------------------------- scorers
class HookPage(FakePage):
    """A live page for G7 / G8 / G9: frames tick on their own and carry the body mode and the robot's fall state;
    the test's hooks change them (a push fells the robot, a deploy kill faults the body). With `recovers`, the
    stack recovers by itself (sim_recovery, HOLD again) a moment later."""

    def __init__(self, recovers: bool = False) -> None:
        super().__init__(H40_LAYOUT, H40_OBJECTS, self._on_say)
        self.mode, self.fallen, self.recovers = "HOLD", False, recovers
        self._tick: asyncio.Task | None = None

    async def frame(self, trace=(), events=(), calls=(), active=()) -> None:
        self.t += 0.5
        await self.inbox.put(json.dumps({
            "type": "frame", "t": self.t, "trace": list(trace), "events": list(events), "calls": list(calls),
            "truth": {"objects": self.objects, "robot": {"x": 1.0, "z": 1.0, "yaw": 0.0, "fallen": self.fallen,
                                                         "upright": not self.fallen}},
            "body": {"mode": self.mode}, "runtime": {"active": list(active), "executions": list(self.execs)}}))

    async def send(self, raw: str) -> None:
        m = json.loads(raw)
        if m["type"] == "estop":
            self.mode = "ESTOP"
            await self.frame(trace=[{"t": self.t, "type": "safety_event", "kind": "estop"}])
            if self.recovers:
                asyncio.get_running_loop().create_task(self._recover("B"))
        await super().send(raw)
        if self._tick is None:
            self._tick = asyncio.get_running_loop().create_task(self._ticker())

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            await self.frame()

    async def _on_say(self, page: FakePage, text: str) -> None:
        if text.lower().startswith("bring me"):
            await self.frame(trace=[{"t": self.t, "type": "started", "tool": "navigate", "execution_id": "nav-1",
                                     "args": {"location": "bedroom"}}], active=[{"id": "nav-1"}])
            if getattr(self, "reject_manip", False):
                await asyncio.sleep(0.05)
                await self.frame(trace=[{"t": self.t, "type": "rejected", "tool": "manipulate", "stage": "capability",
                                         "why": "policy unavailable: sim below real time: DEGRADED, rtf_5s 0.90 "
                                                "(< 0.95)"}])

    async def push(self) -> dict:
        self.fallen, self.mode = True, "FAULT"
        await self.frame(trace=[{"t": self.t, "type": "safety_event", "kind": "fell", "source": "p1"},
                                {"t": self.t, "type": "stop", "reason": "safety:fell"}],
                         events=[{"t": self.t, "type": "speech_started",
                                  "text": "I've lost my balance; I'm stopping until I'm steady."}])
        if self.recovers:
            asyncio.get_running_loop().create_task(self._recover("A"))
        return {"ok": True, "fell": True}

    async def kill_deploy(self) -> dict:
        self.mode = "FAULT"
        await self.frame(trace=[{"t": self.t, "type": "safety_event", "kind": "deploy_lost", "source": "body"}])
        if self.recovers:
            asyncio.get_running_loop().create_task(self._recover("B"))
        return {"ok": True, "killed_pid": 1}

    async def _recover(self, path: str) -> None:
        await asyncio.sleep(0.05)
        self.fallen, self.mode = False, "HOLD"
        await self.frame(trace=[{"t": self.t, "type": "reconcile_start"}],
                         events=[{"t": self.t, "type": "sim_recovery", "path": path, "count": 1}])

    async def operator(self) -> dict:
        self.fallen, self.mode = False, "HOLD"
        await self.frame()
        return {"ok": True, "path": "A (body recover, operator-triggered)"}


class PyHooks(ss.HookInjector):
    """Hooks that call Python (the page's fault handlers) instead of a shell; logged like shell hooks."""

    def __init__(self, fns: dict) -> None:
        super().__init__({k: "true" for k in fns})
        self.fns = fns

    async def do(self, name: str, **kw):
        res = await self.fns[name](**kw)
        self.log.append({"name": name, "args": kw, "rc": 0, "s": 0.0, "json": res, "tail": json.dumps(res)})
        return bool(res.get("ok")), json.dumps(res)


def _fast(monkeypatch, window: float = 1.0):
    monkeypatch.setattr(ss.suite, "LOAD_TIMEOUT_S", 5.0)
    monkeypatch.setattr(ss, "RECOVERY_WINDOW_S", window)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ss.suite.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.02)))


def _play(page: HookPage, gid: str, fns: dict, profile: str = "sonic"):
    async def go():
        run = ss.StackRun(page, profile, 0.02, PyHooks(fns))
        reader = asyncio.create_task(run.reader())
        try:
            return await ss.run_one(next(s for s in ss.SPECS if s.id == gid), run, local=False), run.injector
        finally:
            reader.cancel()
            if page._tick:
                page._tick.cancel()
    return asyncio.run(go())


def _crit(res, prefix):
    return next(c["ok"] for c in res["criteria"] if c["name"].startswith(prefix))


async def _none_needed(**_):
    return {"ok": True, "path": "none needed"}


@pytest.mark.parametrize("recovers", [False, True])
def test_g7_scores_the_stacks_own_recovery_and_records_the_operator_fixture(monkeypatch, recovers):
    _fast(monkeypatch)
    page = HookPage(recovers=recovers)
    calls = []

    async def recover_robot(**_):
        calls.append(page.fallen)
        return await page.operator()

    async def push_robot(newtons):
        assert newtons == 250
        return await page.push()
    res, inj = _play(page, "G7", {"push_robot": push_robot, "recover_robot": recover_robot})
    assert _crit(res, "safety_event(fell)") is True and _crit(res, "paused") is True
    assert _crit(res, "spoken notice") is True
    assert _crit(res, "no command{stop} sent") is True and _crit(res, "the deploy process stays alive") is True
    assert _crit(res, "sim_recovery{path} labelled") is recovers
    assert _crit(res, "upright and ready") is recovers
    assert _crit(res, "reconcile after recovery") is recovers
    assert res["verdict"] == ("PASS" if recovers else "FAIL"), res["criteria"]
    # the preflight and the post-window fixture ran; the fixture is recorded, labelled and not scored
    assert [x["name"] for x in inj.log] == ["recover_robot", "push_robot", "recover_robot"]
    fx = res["extra"]["fixtures"][0]
    assert fx["hook"] == "recover_robot" and "operator fixture" in fx["label"]
    assert calls == [False, not recovers]                  # without the stack's recovery the fixture found it down


@pytest.mark.parametrize("recovers", [False, True])
def test_g8_kill_and_estop_then_the_operator_path_b(monkeypatch, recovers):
    _fast(monkeypatch)
    page = HookPage(recovers=recovers)

    async def restore_deploy(**_):
        return await page.operator()
    res, inj = _play(page, "G8", {"kill_deploy": lambda **_: page.kill_deploy(), "restore_deploy": restore_deploy,
                                   "recover_robot": _none_needed})
    for v in ("a: kill P2", "b: kill button (estop)"):
        assert _crit(res, f"{v}: the stack noticed") is True
        assert _crit(res, f"{v}: the band catches") is True
        assert _crit(res, f"{v}: body.ready <= 60 s") is recovers
        assert _crit(res, f"{v}: path B labelled") is recovers
    assert res["verdict"] == ("UNVERIFIED" if recovers else "FAIL")        # the third-recovery check is by hand
    assert [x["hook"] for x in res["extra"]["fixtures"]] == ["restore_deploy", "restore_deploy"]
    assert any(m.get("type") == "estop" for m in page.sent)


def test_g9_throttles_after_the_load_and_always_restores(monkeypatch):
    _fast(monkeypatch)
    page = HookPage()
    page.reject_manip = True
    order = []

    async def throttle_rtf(rtf):
        order.append(("throttle", rtf, page.scene))
        return {"ok": True}

    async def restore_rtf(**_):
        order.append(("restore",))
        return {"ok": True}
    res, _ = _play(page, "G9", {"throttle_rtf": throttle_rtf, "restore_rtf": restore_rtf,
                                "recover_robot": _none_needed})
    assert order[0][:2] == ("throttle", 0.9) and order[0][2] and order[-1] == ("restore",)
    assert _crit(res, "manipulate rejected (DEGRADED)") is True
    assert _crit(res, "walking capped") is None                           # the page shows no walking cap
    assert res["verdict"] == "UNVERIFIED"
