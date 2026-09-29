"""The page against the REAL runtime (agent/, on the api/ contract) with the runtime agent's scripted
brain and in-process robot (tests/unit/fakes_rt.py): no simulator, no LLM. A chat message goes in, the
harness fetches the alarm clock, and the page's frames carry executions, tool states, trace rows and
truth the way the eval reads them."""

from __future__ import annotations

import asyncio
import json

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

try:
    from agent import create_runtime
    from agent.harness import is_stop
    from brains.composite import CompositeBrain
    from sim.clock import SimClock
    from sim.log import EventLog
    from tests.unit.fakes_rt import FakeRobot, PolicyBrain
except Exception as e:  # noqa: BLE001
    pytest.skip(f"runtime or its test fakes not importable: {e}", allow_module_level=True)

from ui.cameras import NoCameras  # noqa: E402
from ui.server import Deps, Hub, serve_hub  # noqa: E402

SPEED = 20.0


class TruthFromFakeRobot:
    """A world model stand-in: truth() from the fake robot's own dict world (Worldline frame)."""

    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot

    def truth(self) -> dict:
        objs = {}
        for oid, o in self.robot.objects.items():
            pos = getattr(o, "pos", None) or (0.0, 0.0, 0.0)
            objs[oid] = {"type": o.type, "label": o.type.replace("_", " "), "where": o.where,
                         "x": pos[0], "z": pos[1], "y": pos[2] if len(pos) > 2 else None}
        return {"robot": {"x": self.robot.xy[0], "z": self.robot.xy[1], "yaw": 0.0}, "objects": objs,
                "hands": dict(self.robot.hands), "source": "lite-gt"}


def test_the_page_follows_a_real_runtime_fetch():
    built: dict = {}

    def build(profile, scene, clock, log):
        robot = FakeRobot(clock)
        built["robot"] = robot
        return TruthFromFakeRobot(robot), robot, None

    def create_rt(spec, robot, user, brain, clock):
        rt = create_runtime(robot, user, brain, clock)
        built["runtime"] = rt
        return rt

    from brains.interface import BrainInfo
    deps = Deps(build=build, create_planner=lambda info: PolicyBrain("alarm_clock"), create_runtime=create_rt,
                brain_info=BrainInfo, composite=CompositeBrain, clock=lambda speed: SimClock(SPEED),
                event_log=EventLog, is_stop=is_stop, frame_gate=lambda: None)

    async def go():
        hub = Hub("procthor-train-40", "lite", deps=deps, cameras=NoCameras(), system1="off")
        server = await serve_hub(hub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="lite")
        assert hub.session.error is None, hub.session.error
        states, trace, execs, delivered = set(), [], {}, False
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                init = json.loads(await ws.recv())
                assert init["type"] == "init" and init["runtime"]["tool_state"] == "IDLE"
                await ws.send(json.dumps({"type": "say", "text": "Bring me the alarm clock."}))
                for _ in range(400):
                    await hub.tick()
                    while True:
                        try:
                            m = json.loads(await asyncio.wait_for(ws.recv(), 0.02))
                        except asyncio.TimeoutError:
                            break
                        if m.get("type") != "frame":
                            continue
                        rt = m["runtime"]
                        states.add(rt["tool_state"])
                        trace += m["trace"]
                        for e in rt["executions"]:
                            execs[e["id"]] = e
                        if any(r.get("type") == "delivered" for r in m["trace"]):
                            delivered = True
                    if delivered:
                        break
                    await asyncio.sleep(0.02)
        finally:
            server.close()
            await hub.session.stop()
            hub.close()
        return states, trace, execs, delivered

    states, trace, execs, delivered = asyncio.run(asyncio.wait_for(go(), 60))
    assert delivered, [(r.get("type"), r.get("error")) for r in trace][-30:]
    assert "NAVIGATING" in states and "MANIPULATING" in states
    picks = [e for e in execs.values() if e["tool"] == "manipulate" and e["action"] == "pick"]
    assert picks and picks[0]["status"] == "succeeded" and picks[0]["id"].startswith("man-")
    assert picks[0]["v"] is not None and picks[0]["e"] is not None, "generation / epoch fences are shown"
    started = [r for r in trace if r.get("type") == "started"]
    assert any(r["tool"] == "manipulate" and r.get("action") == "pick" for r in started), \
        "eval's started('manipulate', action='pick') has what it needs"
    results = [r for r in trace if r.get("type") == "result"]
    assert results and all(str(r["status"]) == str(r["status"]).lower() for r in results), "lowercase envelopes"
    assert built["robot"].objects["alarm_clock_1"].where != "bedroom_dresser_1a"


def test_the_eval_suite_scores_a_real_runtime_over_the_websocket(monkeypatch):
    """eval/suite.py -> websocket -> ui/server.py -> agent/ runtime -> fake robot; scored on truth.
    The fake robot's executor is "lite", so the pass is a labelled (non-target) pass."""
    from eval import suite

    monkeypatch.setattr(suite, "LOAD_TIMEOUT_S", 20.0)

    def build(profile, scene, clock, log):
        robot = FakeRobot(clock)
        return TruthFromFakeRobot(robot), robot, None

    from brains.interface import BrainInfo
    deps = Deps(build=build, create_planner=lambda info: PolicyBrain("alarm_clock"),
                create_runtime=lambda spec, robot, user, brain, clock: create_runtime(robot, user, brain, clock),
                brain_info=BrainInfo, composite=CompositeBrain, clock=lambda speed: SimClock(SPEED),
                event_log=EventLog, is_stop=is_stop, frame_gate=lambda: None)

    async def go():
        hub = Hub("procthor-train-40", "lite", deps=deps, cameras=NoCameras(), system1="off")
        server = await serve_hub(hub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        ticker = asyncio.create_task(hub.ticker())
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                run = suite.Run(ws, "lite")
                reader = asyncio.create_task(run.reader())
                t0 = asyncio.get_running_loop().time()
                passed, note = await suite.fetch_other_room(run)
                res = suite.score("fetch_other_room", passed, note, asyncio.get_running_loop().time() - t0, run)
                reader.cancel()
        finally:
            ticker.cancel()
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
        return res

    res = asyncio.run(asyncio.wait_for(go(), 90))
    assert res["passed"], res["note"]
    assert res["executors_used"]["nav"].get("lite") and res["executors_used"]["manip"].get("lite") == 2
    assert res["fallback_pass"] and not res["target_pass"] and res["shortcuts"] == ["lite"]
    assert res["fixtures_ok"], res.get("fixture_notes")
    assert any("user surface is living_room_table_1a" in n for n in res["fixture_notes"]), \
        "the fake house's user surface differs from H40's binding: noted, not failed"
