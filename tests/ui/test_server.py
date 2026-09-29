"""The page's server against fakes: pages load, websocket messages flow both ways, and the
camera panes carry frames from a fake viz FrameTap. No simulator, no LLM calls."""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

from ui.cameras import TapCameras  # noqa: E402
from ui.server import Hub, serve_hub  # noqa: E402
from ui_fakes import JPEG_A, JPEG_B, FakeTap, make_deps  # noqa: E402


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 20))


async def _start(tap: FakeTap | None = None, built: dict | None = None, profile: str = "sonic"):
    built = {} if built is None else built
    tap = tap or FakeTap()
    hub = Hub("procthor-train-40", profile, deps=make_deps(built), cameras=TapCameras(tap), system1="off")
    server = await serve_hub(hub, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return hub, server, port, tap, built


async def _recv(ws, kind: str, timeout: float = 5.0) -> dict:
    """The next message of this type (others are skipped)."""
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if m.get("type") == kind:
            return m


def _get(port: int, path: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, "", ""


def test_pages_load():
    async def go():
        hub, server, port, _, _ = await _start()
        try:
            status, ctype, body = await asyncio.to_thread(_get, port, "/")
            assert status == 200 and ctype.startswith("text/html")
            assert "Worldline" in body and "/views/plan.js" in body and "/views/diagram.js" in body
            for view in ("plan", "diagram", "memory"):
                status, ctype, js = await asyncio.to_thread(_get, port, f"/views/{view}.js")
                assert status == 200 and "javascript" in ctype and "RobotViews" in js
            assert (await asyncio.to_thread(_get, port, "/views/../server.py"))[0] == 404
            assert (await asyncio.to_thread(_get, port, "/ui/server.py"))[0] == 404
        finally:
            server.close()
            hub.close()
    run(go())


def test_loading_then_init_frame_and_memory():
    async def go():
        hub, server, port, tap, built = await _start()
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                first = json.loads(await ws.recv())
                assert first["type"] == "loading" and first["meta"]["profile"] == "sonic"
                assert [p for p, _ in first["meta"]["profiles"]] == ["lite", "bringup", "sonic", "full"]
                await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
                init = await _recv(ws, "init")
                assert built["profile"] == "sonic" and built["scene"] == "procthor-train-40"
                assert init["config"]["profile"] == "sonic"
                assert "sonic_arm_script" in init["config"]["stepping_stones"]
                lay = init["layout"]
                assert lay["user_surface"] == "kitchen_dining_table_1b"
                assert lay["human"]["keypoint"] == "kitchen_dining_table_1b"
                assert lay["grid_step"] == 0.25 and len(lay["grid"]) > 20       # nav grid from occupancy
                assert set(lay["rooms"]) == {"bedroom", "kitchen"} and lay["rooms"]["bedroom"]["polygon"]
                assert lay["occupancy"]["png"] and lay["occupancy"]["extent"] == [0.0, 0.0, 4.0, 3.0]
                assert "not visible to the planner" in lay["truth_label"]
                truth = init["truth"]
                assert truth["label"].startswith("sim ground truth")
                assert truth["objects"]["alarm_clock_1"]["where"] == "bedroom_dresser_1a"
                assert truth["objects"]["alarm_clock_1"]["x"] == 0.8 and truth["objects"]["alarm_clock_1"]["z"] == 2.6
                assert truth["robot"]["x"] == 1.0 and truth["robot"]["yaw"] == 90.0 and truth["robot"]["upright"]
                assert init["body"]["mode"] == "HOLD" and init["stack"]["profile"] == "sonic"
                assert init["runtime"]["tool_state"] == "IDLE"
                mem = await _recv(ws, "memory")
                assert mem["scene"] == "procthor-train-40"
                await hub.tick()
                frame = await _recv(ws, "frame")
                assert frame["truth"]["objects"]["book_1"]["y"] == 0.55
                assert frame["nav"]["route"] == [[1.0, 1.5], [2.0, 1.5]]
                assert frame["view"]["rows"] and frame["view"]["range"] == 2.5
                assert frame["robot_map"]["trail"], "the first pose starts the trail"
        finally:
            server.close()
            await hub.session.stop()
            hub.close()
    run(go())


def test_chat_stop_and_executions_flow():
    async def go():
        hub, server, port, tap, built = await _start()
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                await _recv(ws, "init")
                await ws.send(json.dumps({"type": "say", "text": "Bring me the alarm clock."}))
                rt = built["runtime"]
                for _ in range(100):
                    if rt.history:
                        break
                    await asyncio.sleep(0.02)
                assert rt.heard and rt.heard[0].text == "Bring me the alarm clock."
                await hub.tick()
                frame = await _recv(ws, "frame")
                active = frame["runtime"]["active"]
                assert active and active[0]["tool"] == "navigate" and active[0]["executor"] == "sonic_walk"
                assert active[0]["id"].startswith("nav-") and active[0]["fallback"] is False
                assert frame["runtime"]["tool_state"] == "NAVIGATING"
                assert any(r["type"] == "started" and r["tool"] == "navigate" for r in frame["trace"])
                assert frame["truth"]["executions"][0]["tool"] == "navigate"
                # the chat Stop: the keyword lane runs halt() before anything else
                await ws.send(json.dumps({"type": "say", "text": "stop"}))
                for _ in range(100):
                    if rt.stops:
                        break
                    await asyncio.sleep(0.02)
                assert rt.stops == [("stop", "keyword")] and built["robot"].halts == 1
                # persona level and step mode reach the runtime
                await ws.send(json.dumps({"type": "persona", "level": "optimize"}))
                await ws.send(json.dumps({"type": "step", "mode": "on"}))
                for _ in range(100):
                    if rt.persona_level == "optimize" and rt.step_mode:
                        break
                    await asyncio.sleep(0.02)
                assert rt.persona_level == "optimize" and rt.step_mode is True
        finally:
            server.close()
            await hub.session.stop()
            hub.close()
    run(go())


def test_kill_button_needs_confirm_and_is_not_the_chat_stop():
    async def go():
        hub, server, port, tap, built = await _start()
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                await _recv(ws, "init")
                await ws.send(json.dumps({"type": "estop"}))
                note = await _recv(ws, "notice")
                assert "confirm" in note["text"] and built["robot"].estops == []
                await ws.send(json.dumps({"type": "estop", "confirm": True}))
                note = await _recv(ws, "notice")
                assert "estop" in note["text"] and built["robot"].estops == ["operator kill button"]
                assert built["robot"].halts == 0, "the kill button never goes through halt()"
                await hub.tick()
                frame = await _recv(ws, "frame")
                assert any(e["type"] == "safety_stop" for e in frame["events"])
        finally:
            server.close()
            await hub.session.stop()
            hub.close()
    run(go())


def test_camera_panes_show_frames_from_the_tap():
    async def go():
        tap = FakeTap()
        tap.put("head", JPEG_A, {"t_sim": 1.0, "seq": 1})
        tap.put("chase", JPEG_B, {"t_sim": 1.0, "seq": 7, "source": "tp"})
        tap.put("top", JPEG_A, {"seq": 3, "t_wall": 100.0, "source": "top", "extent": [0.0, 0.0, 8.66, 8.66]})
        hub, server, port, tap, built = await _start(tap)
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                await _recv(ws, "init")
                got = {}
                while len(got) < 3:
                    m = await _recv(ws, "camera")
                    got[m["which"]] = m
                assert base64.b64decode(got["head"]["jpeg"]) == JPEG_A
                assert base64.b64decode(got["chase"]["jpeg"]) == JPEG_B
                assert got["top"]["meta"]["extent"] == [0.0, 0.0, 8.66, 8.66]
                assert "ground truth" in got["top"]["meta"]["caption"]
                assert "System 1" in got["head"]["meta"]["caption"]
                # a new head frame goes out on the next tick; an unchanged chase frame does not
                tap.put("head", JPEG_B, {"t_sim": 1.2, "seq": 2})
                await asyncio.sleep(0.25)
                await hub.tick()
                m = await _recv(ws, "camera")
                assert m["which"] == "head" and base64.b64decode(m["jpeg"]) == JPEG_B
                # VizCams re-sends the same top snapshot (same seq and t_wall): the page is not sent it again
                tap.put("top", JPEG_A, {"seq": 3, "t_wall": 100.0, "source": "top", "extent": [0.0, 0.0, 8.66, 8.66]})
                await asyncio.sleep(1.1)
                sent = hub.feed.due()
                assert [x["which"] for x in sent] == [], sent
        finally:
            server.close()
            await hub.session.stop()
            hub.close()
    run(go())


def test_lite_profile_uses_world_frames():
    from dataclasses import dataclass

    from ui.cameras import WorldCameras

    @dataclass
    class Frame:
        rev: int
        t_wall: float
        jpeg: bytes
        cam_pose_wl: tuple
        stationary: bool = True

    class W:
        def latest_frame(self, cam):
            return Frame(4, 1.0, JPEG_A, (1.0, 2.0, 90.0, 15.0)) if cam == "head" else None

    cams = WorldCameras(W())
    assert cams.names() == ["head"] and cams.rev("head") == 4 and cams.jpeg("head") == JPEG_A
    assert "jpeg" not in cams.meta("head") and cams.meta("head")["cam_pose_wl"] == (1.0, 2.0, 90.0, 15.0)


def test_reset_rejects_unknown_profile():
    async def go():
        hub, server, port, tap, built = await _start()
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                await _recv(ws, "loading")
                await ws.send(json.dumps({"type": "reset", "scene": "procthor-train-15", "profile": "turbo"}))
                note = await _recv(ws, "notice")
                assert "unknown profile" in note["text"]
                await ws.send(json.dumps({"type": "reset", "scene": "procthor-train-15", "profile": "bringup"}))
                init = await _recv(ws, "init")
                assert init["config"]["scene"] == "procthor-train-15" and init["config"]["profile"] == "bringup"
                assert "kinematic_nav" in init["config"]["stepping_stones"]
        finally:
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
    run(go())


def test_system1_gets_head_frames_through_the_gate_and_observations_reach_the_runtime(monkeypatch):
    """The System 1 couplings (PLAN 8.2, 8.4): state updates, gated head frames with the keypoint the robot
    stands at, an observe call whose items go to runtime.add_observations. A fake System 1, no network."""
    import ui.server as srv

    monkeypatch.setattr(srv, "S1_OBSERVE_EVERY_S", 0.3)

    class FakeS1:
        status = "ready"

        def __init__(self, on_status):
            self.frames, self.updates, self.said = [], [], []

        async def run(self):
            await asyncio.sleep(3600)

        async def update(self, ctx):
            self.updates.append(ctx)

        async def frame(self, jpeg, where):
            self.frames.append((jpeg, where))

        async def observe(self):
            return [{"what": "a door left open", "where": "start", "confidence": 0.8}]

        async def robot_said(self, text):
            self.said.append(text)

        async def route(self, text):
            return {"kind": "request", "confidence": 0.9}

    class Gate:
        def __init__(self):
            self.calls = []

        def decide(self, frame, pose, now, expected=False, stationary=None):
            from types import SimpleNamespace
            self.calls.append((pose, expected, stationary))
            return SimpleNamespace(send=True, reason="new view")

        def reset(self):
            pass

        def summary(self):
            return "sent 1"

    async def go():
        tap = FakeTap()
        tap.put("head", JPEG_A, {"seq": 1})
        built = {}
        deps = make_deps(built)
        deps.system1 = FakeS1
        gate = Gate()
        deps.frame_gate = lambda: gate
        monkeypatch.setattr(srv, "_decode_rgb", lambda jpeg: [[0]])
        hub = Hub("procthor-train-40", "sonic", deps=deps, cameras=TapCameras(tap), system1="x")
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        observed = []
        built["runtime"].add_observations = lambda items, source: observed.append((items, source))
        hub.start_system1()
        await asyncio.sleep(1.2)
        tap.put("head", JPEG_B, {"seq": 2})
        await asyncio.sleep(1.2)
        s1 = hub.system1
        await hub.session.stop()
        return s1, gate, observed

    s1, gate, observed = run(go())
    assert [f[0] for f in s1.frames][:2] == [JPEG_A, JPEG_B] and s1.frames[0][1] == "start"
    assert s1.updates and s1.updates[0]["at"] == "start"
    assert gate.calls and gate.calls[0][0] == (1.0, 1.5, 90.0, 15.0), "the robot's own pose (telemetry) feeds the gate"
    assert gate.calls[0][2] is True, "not moving -> stationary"
    assert observed and observed[0][1] == "system1" and observed[0][0][0]["what"] == "a door left open"
