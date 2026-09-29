"""The live-stack fault injections of eval/stack_suite.py (docs/eval_hooks.md), offline:

- eval/hooks.py's hook maps and HookInjector's log (the last JSON line of a hook, exit codes, timeouts);
- tools/hooks/delay_proxy.py on real ZMQ sockets (pass-through, then +300 ms, control socket);
- tools/hooks/obstacle.py on synthetic houses (one door: the box cuts every route; two doors: it says so);
- tools/hooks/cli.py against HookableFakeP1 (the test ops on the fake P1) + the fake deploy + the REAL body service:
  a 250 N push collapses the fake robot, the body latches fault `fallen`, `recover` runs the body's path A;
- the G7 / G8 / G9 scorers on a scripted page: the stack's own recovery is scored inside the window, the operator
  fixture after it is recorded and never scored; G7's push escalation (250 N, then stronger only while the robot
  stays up); G4 / G5 / G6 / G13 with the hooks' order (fault after the load, the fixture always after);
- `home` (the reset_fixup) on the fake stack, and the page reset that left the robot down: fixed up, then the page
  reloaded keeping the standing robot, recorded as an operator fixture.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
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
    # PLAN 0.12: P4 on the dev box behind the OD3 link; the main box's link key is port-forward only, so on the box
    # the outage is the link cut (and said so), and from the laptop the dev-box server itself is stopped
    assert "policy cut --via link --port 5550" in hooks["kill_policy"]
    assert "policy restore --via link" in hooks["restore_policy"]
    local = H.preset("box", policy="local")
    assert "policy cut --via local" in local["kill_policy"]                     # PLAN 0.11's layout, kept
    assert "kill_policy" not in H.preset("box", policy="none")                  # G5/G13 then SKIPPED
    with pytest.raises(ValueError):
        H.preset("box", policy="dev")
    laptop = H.preset("laptop", policy="link")
    assert set(laptop) == set(hooks)
    assert laptop["push_robot"].format(newtons="250").count("--force-n 250") == 1
    assert laptop["kill_deploy"].startswith("BREV_NAME=ludo-g1-brev2 ") and "ssh.sh" in laptop["kill_deploy"]
    assert laptop["kill_policy"].startswith("BREV_NAME=ludo-g1-arena ") and "groot_server.sh stop" in laptop["kill_policy"]
    dev, main = laptop["restore_policy"].split(" && BREV_NAME=")
    assert dev.startswith("BREV_NAME=ludo-g1-arena ") and "start --port 5550 --warm" in dev
    assert main.startswith("ludo-g1-brev2 ") and "policy restore --via link" in main
    assert H.INFRA.endswith("ludo_robotics_prep_g1/00_infra") or "WL_INFRA" in os.environ   # derived, not typed in
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


def test_hooks_cli_home_stands_the_robot_at_the_spawn_through_a_fall_and_a_halt_latch(fake_stack):
    """`home` (the suite's reset_fixup): P1 reset_scene {robot}, a latched fall cleared, a halt latch released at the
    body's own epoch (as the page's reset does), then the stand; each stand is reported."""
    off, p1, svc, bc = fake_stack
    rc, out = _cli(off, "push", "--force-n", "250", "--dir", "left", "--watch-s", "0.5")
    t0 = time.monotonic()
    while bc.status().get("fault") != "fallen" and time.monotonic() - t0 < 5:
        time.sleep(0.1)
    assert bc.status()["fault"] == "fallen" and p1.collapsed
    ack = bc.halt(7, timeout_s=1.0, reason="test: a session's stop")
    assert ack["acked"] and bc.status()["latched"] is True
    p1.x, p1.y = p1.spawn[0] + 1.5, p1.spawn[1]                     # somewhere else than the spawn
    rc, out = _cli(off, "home", "--timeout", "60")
    assert rc == 0, out
    assert out["reset_scene"]["ok"] and out["reset_scene"]["robot_reset"] is True
    first = out["tries"][0]
    assert first["fault_before"] == "fallen" and first["cleared"] == "fallen" and first["resumed_halt_epoch"] == 7
    assert out["stand"]["state"] == "succeeded" and len(out["tries"]) == 1
    assert abs(out["pose"][0] - p1.spawn[0]) < 0.3 and abs(out["pose"][1] - p1.spawn[1]) < 0.3
    st = bc.status()
    assert st["fault"] is None and st["latched"] is False and st["in_control"] and not p1.collapsed


def test_hooks_cli_home_refuses_without_a_deploy(monkeypatch):
    from tools.hooks import cli
    monkeypatch.setattr(cli, "deploy_pid", lambda session, port: None)
    rc, out = _cli(_free_offset(), "home")
    assert rc == 1 and "recover" in out["error"]


def test_policy_cut_and_restore_are_judged_by_a_ping_not_by_the_script(monkeypatch):
    """G5/G13's outage (tools.hooks policy): ok iff P4 stops (cut) / starts (restore) answering on the port, whatever
    the script said. The scripts are stand-ins here: groot_link.sh down kills the fake server, ensure revives it."""
    from tests.fakes.fake_policy_server import FakePolicyServer
    from tools.hooks import cli
    srv = FakePolicyServer(latency_s=0.0).start()
    port = int(srv.endpoint.rsplit(":", 1)[1])
    ran = []

    def script(*args, timeout):
        ran.append(args[:2])
        if args[1] == "down":
            srv.die()
        elif args[1] == "ensure":
            srv.revive()
        return {"rc": 0, "tail": []}
    monkeypatch.setattr(cli, "_script", script)
    try:
        rc, out = _cli(_free_offset(), "policy", "cut", "--via", "link", "--port", str(port))
        assert rc == 0 and out["ping_before"]["ok"] and not out["ping_after"]["ok"] and "link cut" in out["what"]
        rc, out = _cli(_free_offset(), "policy", "restore", "--via", "link", "--port", str(port))
        assert rc == 0 and out["ping_after"]["ok"] and out["what"] == "P4 answers again"
        # a cut that leaves P4 answering (the wrong box, a second server) is not ok
        monkeypatch.setattr(cli, "_script", lambda *a, timeout: {"rc": 0, "tail": []})
        rc, out = _cli(_free_offset(), "policy", "cut", "--via", "local", "--port", str(port))
        assert rc == 1 and out["ping_after"]["ok"]
        assert ran == [("scripts/groot_link.sh", "down"), ("scripts/groot_link.sh", "ensure")]
    finally:
        srv.stop()


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
            if getattr(self, "after_walk", None):
                await self.after_walk()
            if getattr(self, "reject_manip", False):
                await asyncio.sleep(0.05)
                await self.frame(trace=[{"t": self.t, "type": "rejected", "tool": "manipulate", "stage": "capability",
                                         "why": "policy unavailable: sim below real time: DEGRADED, rtf_5s 0.87 "
                                                "(< 0.9 for 3 s; ok again at >= 0.94 for 2 s)"}])

    async def block(self) -> dict:
        """G6: the box is in the corridor; the next walk ends blocked after one replan and the user is told."""
        async def later() -> None:
            await asyncio.sleep(0.05)
            await self.frame(trace=[{"t": self.t, "type": "result", "tool": "navigate", "execution_id": "nav-1",
                                     "status": "failed", "summary": "navigate failed: blocked",
                                     "data": {"reason": "blocked", "blocked_edge": ["hall", "bedroom"],
                                              "replans": 1}}],
                             events=[{"t": self.t + 0.1, "type": "speech_started",
                                      "text": "Something is in the way to the bedroom; I can't get through."}])
        self.after_walk = later
        return {"ok": True, "placed": {"blocks_all_routes": True}}

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
    assert order[0][:2] == ("throttle", 0.87) and order[0][2] and order[-1] == ("restore",)
    assert _crit(res, "manipulate rejected (DEGRADED)") is True
    assert _crit(res, "walking capped") is None                           # the page shows no walking cap
    assert res["verdict"] == "UNVERIFIED"


def test_g6_places_the_box_after_the_load_scores_blocked_and_always_clears(monkeypatch):
    _fast(monkeypatch)
    page = HookPage()
    order = []

    async def spawn_box(**_):
        order.append(("spawn", page.scene))
        return await page.block()

    async def clear_box(**_):
        order.append(("clear",))
        return {"ok": True}
    res, _ = _play(page, "G6", {"spawn_box": spawn_box, "clear_box": clear_box, "recover_robot": _none_needed})
    assert order[0][0] == "spawn" and order[0][1] and order[-1] == ("clear",)
    for name in ("navigate failed: blocked", "with blocked_edge", "after one replan", "tells the user", "no fall"):
        assert _crit(res, name) is True, (name, res["criteria"])
    assert res["verdict"] == "PASS"


def test_g4_fails_at_once_when_the_delay_cannot_be_set_and_restores_it_when_it_can(monkeypatch):
    _fast(monkeypatch)
    page = HookPage()

    async def proxy_absent(ms):
        return {"ok": False, "error": "no reply from the delay proxy at tcp://127.0.0.1:5549"}
    res, inj = _play(page, "G4", {"delay_proxy_on": proxy_absent, "recover_robot": _none_needed}, profile="full")
    assert res["verdict"] == "FAIL" and _crit(res, "delay proxy on (600 ms)") is False
    assert [x["name"] for x in inj.log] == ["recover_robot", "delay_proxy_on"]
    page2, calls = HookPage(), []

    async def on(ms):
        calls.append(("on", ms))
        return {"ok": True, "proxy": {"ms": ms}}

    async def off(**_):
        calls.append(("off",))
        return {"ok": True}
    monkeypatch.setattr(ss.suite.Run, "wait", lambda self, s: 0.2)     # no pick ever starts on this page
    res, _ = _play(page2, "G4", {"delay_proxy_on": on, "delay_proxy_off": off, "recover_robot": _none_needed},
                   profile="full")
    assert calls == [("on", 600), ("off",)] and _crit(res, "a pick started") is False


class PolicyPage(HookPage):
    """G5 / G13 on a live page (full): P4 down -> a new manipulate is rejected at CAPABILITY; P4 killed under a running
    GR00T pick -> the pick fails policy_unavailable at once and the body holds. Every planner call carries the same
    tool schemas unless `drift` changes them once P4 has been down (what Invariant 9 forbids)."""

    SCHEMAS = [{"name": "manipulate", "parameters": {"properties": {"action": {"enum": ["pick", "place"]}}}}]

    def __init__(self, drift: bool = False) -> None:
        super().__init__()
        self.down, self.drift, self.n_calls, self.running, self.kills = False, drift, 0, None, 0

    def _call(self) -> dict:
        self.n_calls += 1
        sch = self.SCHEMAS if not (self.drift and self.kills) else [{"name": "manipulate", "parameters": {}}]
        return {"n": self.n_calls, "via": "model", "purpose": "next_action", "input": "", "tool_schemas": sch}

    async def _on_say(self, page, text: str) -> None:
        if not text.lower().startswith("bring me"):
            return
        await self.frame(trace=[{"t": self.t, "type": "started", "tool": "navigate", "execution_id": "nav-1",
                                 "args": {"location": "bedroom"}}], calls=[self._call()])
        if self.down:
            await self.frame(trace=[{"t": self.t, "type": "rejected", "tool": "manipulate", "stage": "capability",
                                     "why": "policy unavailable: the GR00T PolicyServer does not answer"}],
                             calls=[self._call()])
        else:
            self.running = "man-2"
            await self.frame(trace=[{"t": self.t, "type": "started", "tool": "manipulate", "execution_id": "man-2",
                                     "args": {"action": "pick", "object": "alarm_clock_1"}}],
                             calls=[self._call()], active=[{"id": "man-2"}])

    async def kill(self) -> dict:
        self.down, self.kills = True, self.kills + 1
        if self.running:
            eid, self.running = self.running, None
            await self.frame(trace=[{"t": self.t, "type": "result", "tool": "manipulate", "execution_id": eid,
                                     "status": "failed", "summary": "manipulate failed: policy_unavailable",
                                     "data": {"reason": "policy_unavailable", "executor": "groot_arms",
                                              "skill": "groot.pick.any"}}], calls=[self._call()])
        return {"ok": True, "what": "P4 unreachable from the main box: the OD3 link cut"}

    async def restore(self) -> dict:
        self.down = False
        return {"ok": True, "what": "P4 answers again"}


def test_g5_live_takes_p4_down_before_the_request_and_under_a_running_pick(monkeypatch):
    _fast(monkeypatch)
    page, order = PolicyPage(), []

    async def kill_policy(**_):
        order.append(("kill", page.running))
        return await page.kill()

    async def restore_policy(**_):
        order.append(("restore",))
        return await page.restore()
    res, inj = _play(page, "G5", {"kill_policy": kill_policy, "restore_policy": restore_policy,
                                  "recover_robot": _none_needed}, profile="full")
    assert [x["name"] for x in inj.log] == ["recover_robot", "kill_policy", "restore_policy", "kill_policy",
                                            "restore_policy"]
    assert order == [("kill", None), ("restore",), ("kill", "man-2"), ("restore",)]     # 2nd kill: under the pick
    for name in ("new call rejected", "rejected at the CAPABILITY stage", "tool schemas unchanged",
                 "running call failed(policy_unavailable) within 3 s", "then HOLD", "no fall"):
        assert _crit(res, name) is True, (name, res["criteria"])
    assert res["verdict"] == "PASS"


def test_g5_live_restores_p4_even_when_the_outage_cannot_be_injected(monkeypatch):
    _fast(monkeypatch)
    page, order = PolicyPage(), []

    async def kill_policy(**_):
        order.append("kill")
        return {"ok": False, "error": "P4 still answering after the cut"}

    async def restore_policy(**_):
        order.append("restore")
        return {"ok": True}
    res, _ = _play(page, "G5", {"kill_policy": kill_policy, "restore_policy": restore_policy,
                                "recover_robot": _none_needed}, profile="full")
    assert res["verdict"] == "FAIL" and _crit(res, "policy taken down") is False
    assert order == ["kill", "restore"]                     # nothing left cut


@pytest.mark.parametrize("drift", [False, True])
def test_g13_live_schemas_across_a_policy_outage(monkeypatch, drift):
    _fast(monkeypatch)
    page = PolicyPage(drift=drift)
    res, inj = _play(page, "G13", {"kill_policy": lambda **_: page.kill(), "restore_policy": lambda **_: page.restore(),
                                   "recover_robot": _none_needed}, profile="full")
    assert [x["name"] for x in inj.log] == ["recover_robot", "kill_policy", "restore_policy"]
    assert page.n_calls >= 2 and page.kills == 1             # planner calls on both sides of the outage
    assert _crit(res, "schemas unchanged across the session, through a policy outage") is (not drift)
    assert res["verdict"] == ("FAIL" if drift else "PASS")
    assert not page.down                                     # P4 back after the outage


def test_g7_escalates_the_push_only_while_the_robot_stays_up_and_says_which_push_felled_it(monkeypatch):
    """Live, SONIC steps out of PLAN's 250 N standing: the next push is stronger (400 N, then 600 N), only after the
    one before did not fell the robot within G7_FALL_WAIT_S; extra.pushes says which one did, and a note says so."""
    _fast(monkeypatch)
    monkeypatch.setattr(ss, "G7_FALL_WAIT_S", 0.3)
    page = HookPage(recovers=True)
    forces = []

    async def push_robot(newtons):
        forces.append(newtons)
        if newtons < 400:
            return {"ok": True, "fell": False}                    # SONIC stepped out of it
        return await page.push()
    res, inj = _play(page, "G7", {"push_robot": push_robot, "recover_robot": _none_needed})
    assert forces == [250, 400]
    assert [(p["newtons"], p["fell"]) for p in res["extra"]["pushes"]] == [(250, False), (400, True)]
    assert any("250 N did not fell SONIC" in n and "400 N" in n for n in res["notes"])
    fell = next(c for c in res["criteria"] if c["name"] == "safety_event(fell)")
    assert fell["ok"] is True and fell["detail"]["push_n"] == 400
    assert res["verdict"] == "PASS", res["criteria"]


def test_g7_fails_honestly_when_no_push_fells_the_robot(monkeypatch):
    _fast(monkeypatch)
    monkeypatch.setattr(ss, "G7_FALL_WAIT_S", 0.2)
    monkeypatch.setattr(ss, "RECOVERY_WINDOW_S", 0.3)
    page = HookPage()

    async def push_robot(newtons):
        return {"ok": True, "fell": False}
    res, _ = _play(page, "G7", {"push_robot": push_robot, "recover_robot": _none_needed})
    assert [p["newtons"] for p in res["extra"]["pushes"]] == list(ss.G7_PUSHES_N)
    assert _crit(res, "safety_event(fell)") is False and res["verdict"] == "FAIL"


class ResetSagPage(HookPage):
    """A live page whose own reset leaves the robot down the first `sags` times (init.config.sim_reset.ok false, the
    stand after P1's reset_robot ended fallen); a reset with reset_robot=false keeps the robot and always succeeds."""

    def __init__(self, sags: int = 1) -> None:
        super().__init__()
        self.sags, self.resets = sags, []

    async def send(self, raw: str) -> None:
        m = json.loads(raw)
        if m["type"] != "reset":
            return await super().send(raw)
        keep = m.get("reset_robot") is False
        self.resets.append("keep_robot" if keep else "robot")
        self.sent.append(m)
        self.scene, self.profile = m["scene"], m.get("profile") or "lite"
        self.objects = json.loads(json.dumps(self.objects0))
        if keep:
            sr = {"ok": True, "robot": False, "s": 0.1}
        elif self.sags > 0:
            self.sags -= 1
            sr = {"ok": False, "robot": True, "why": "stand after the robot reset ended failed (fallen)",
                  "stand": {"state": "failed", "reason": "fallen"}, "s": 4.2}
        else:
            sr = {"ok": True, "robot": True, "s": 8.1}
        await self.inbox.put(json.dumps({"type": "init", "config": {"scene": self.scene, "profile": self.profile,
                                                                    "sim_reset": sr},
                                         "layout": self.layout, "map": {"keypoints": {}},
                                         "truth": {"objects": self.objects}, "runtime": {"active": []}}))
        if self._tick is None:
            self._tick = asyncio.get_running_loop().create_task(self._ticker())


@pytest.mark.parametrize("sags", [0, 1])
def test_a_page_reset_that_left_the_robot_down_is_fixed_up_and_reloaded_keeping_the_robot(monkeypatch, sags):
    _fast(monkeypatch)
    page = ResetSagPage(sags=sags)
    page.reject_manip = True
    order = []

    async def reset_fixup(**_):
        order.append(("home", list(page.resets)))
        return {"ok": True, "tries": [{"state": "failed", "reason": "fallen"}, {"state": "succeeded"}]}

    async def throttle_rtf(rtf):
        order.append(("throttle", list(page.resets)))
        return {"ok": True}
    res, inj = _play(page, "G9", {"reset_fixup": reset_fixup, "throttle_rtf": throttle_rtf,
                                  "restore_rtf": lambda **_: _none_needed(), "recover_robot": _none_needed})
    if not sags:
        assert page.resets == ["robot"] and "reset_fixups" not in res["extra"]
        assert [x["name"] for x in inj.log] == ["recover_robot", "throttle_rtf", "restore_rtf"]
        return
    # the fixture ran after the failed page reset, then the page reloaded keeping the (now standing) robot, and only
    # then did the scenario inject its own fault
    assert page.resets == ["robot", "keep_robot"]
    assert order == [("home", ["robot"]), ("throttle", ["robot", "keep_robot"])]
    fx = res["extra"]["reset_fixups"]
    assert len(fx) == 1 and fx[0]["fixture_ok"] is True and fx[0]["stands"] == 2
    assert fx[0]["page_sim_reset"]["stand"]["reason"] == "fallen"
    assert fx[0]["reload_keeping_robot"]["ok"] is True and fx[0]["reload_keeping_robot"]["robot"] is False
    assert "operator fixture" in fx[0]["label"]
    assert _crit(res, "manipulate rejected (DEGRADED)") is True       # the scenario itself still scored
    assert [x["name"] for x in inj.log] == ["recover_robot", "reset_fixup", "throttle_rtf", "restore_rtf"]

