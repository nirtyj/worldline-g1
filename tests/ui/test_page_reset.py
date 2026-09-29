"""The page's reset puts the simulated house back before the runtime starts (W2.8; P1 reset_scene,
docs/contracts/p1_m2b.md §8): objects to their load poses, the robot to the spawn, then the body stands it again.
Fakes only: a world with the P1.7 op and a body client with `stand`. No simulator."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402

from ui.cameras import TapCameras  # noqa: E402
from ui.server import Hub, serve_hub  # noqa: E402
from ui_fakes import FakeRobot, FakeTap, FakeWorld, make_deps  # noqa: E402


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 20))


class ResettableWorld(FakeWorld):
    """FakeWorld plus IsaacGTWorldModel's P1.7 surface: capabilities()["reset_scene"] and reset_scene()."""

    def __init__(self, log: list, fail: bool = False) -> None:
        super().__init__()
        self.log, self.fail = log, fail

    def capabilities(self) -> dict:
        return {"reset_scene": True}

    def reset_scene(self, variant: str = "default", *, poses=None, robot=None) -> dict:
        self.log.append(("reset_scene", variant, robot))
        if self.fail:
            raise RuntimeError("P1 reply timed out")
        return {"variant": variant, "objects_reset": 34, "robot_reset": bool(robot), "released": ["Apple|1"],
                "ms": 13.8, "ok": True}


class StandClient:
    def __init__(self, log: list, state: str = "succeeded", reason: str | None = None, latched: int | None = None
                 ) -> None:
        self.log, self.state, self.reason, self.latched = log, state, reason, latched

    def stand(self, wait: bool = True, timeout: float | None = None, **kw):
        self.log.append(("stand", wait, timeout))
        if self.latched is not None:
            return SimpleNamespace(state="failed", result={"reason": "halted"})
        return SimpleNamespace(state=self.state, result={"band_released": self.state == "succeeded",
                                                         **({"reason": self.reason} if self.reason else {})})

    def status(self):
        return {"fences": {"halt_epoch": self.latched if self.latched is not None else 0}}

    def resume(self, epoch: int):
        self.log.append(("resume", epoch))
        self.latched = None
        return {"ok": True}


def _deps(built: dict, log: list, *, stand_state: str = "succeeded", reason: str | None = None,
          fail: bool = False, with_stand: bool = True, latched: int | None = None):
    deps = make_deps(built)
    make_runtime = deps.create_runtime

    def build(profile, scene, clock, ev_log):
        w = ResettableWorld(log, fail=fail)
        r = FakeRobot(w)
        r.body = SimpleNamespace(client=StandClient(log, stand_state, reason, latched) if with_stand else None)
        built.update(world=w, robot=r, profile=profile, scene=scene)
        return w, r, None

    def create_runtime(spec, robot, user, brain, clock):
        log.append(("runtime",))
        return make_runtime(spec, robot, user, brain, clock)

    deps.build, deps.create_runtime = build, create_runtime
    return deps


async def _recv(ws, kind: str, timeout: float = 5.0) -> dict:
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if m.get("type") == kind:
            return m


def test_page_reset_resets_the_scene_then_stands_the_robot_before_the_runtime_starts():
    async def go():
        built, log = {}, []
        hub = Hub("procthor-train-40", "sonic", deps=_deps(built, log), cameras=TapCameras(FakeTap()), system1="off")
        server = await serve_hub(hub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                await _recv(ws, "loading")
                await ws.send(json.dumps({"type": "reset", "scene": "procthor-train-40", "profile": "sonic"}))
                note = await _recv(ws, "notice")
                init = await _recv(ws, "init")
                assert log == [("reset_scene", "default", True), ("stand", True, 90.0), ("runtime",)]
                sr = init["config"]["sim_reset"]
                assert sr["ok"] and sr["robot"] and sr["p1"]["objects_reset"] == 34
                assert sr["p1"]["released"] == ["Apple|1"] and sr["stand"]["state"] == "succeeded"
                assert "34 objects back at their load poses, the robot back at the start, standing" in note["text"]
                # objects only
                log.clear()
                await ws.send(json.dumps({"type": "reset", "scene": "procthor-train-40", "reset_robot": False}))
                init = await _recv(ws, "init")
                assert log == [("reset_scene", "default", None), ("runtime",)]
                assert init["config"]["sim_reset"]["ok"] and not init["config"]["sim_reset"]["robot"]
                # the sim left as it is
                log.clear()
                await ws.send(json.dumps({"type": "reset", "scene": "procthor-train-40", "reset_sim": False}))
                init = await _recv(ws, "init")
                assert log == [("runtime",)] and init["config"]["sim_reset"] is None
        finally:
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
    run(go())


def test_the_servers_first_load_does_not_touch_the_sim():
    async def go():
        built, log = {}, []
        hub = Hub("procthor-train-40", "sonic", deps=_deps(built, log), cameras=TapCameras(FakeTap()), system1="off")
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        try:
            assert log == [("runtime",)] and hub.session.sim_reset is None
        finally:
            await hub.session.stop()
            hub.close()
    run(go())


@pytest.mark.parametrize("case, why", [
    ({"stand_state": "failed", "reason": "halted"}, "stand after the robot reset ended failed (halted)"),
    ({"fail": True}, "P1 reset_scene failed: P1 reply timed out"),
    ({"with_stand": False}, "the robot was reset into the band, but this body has no stand op"),
])
def test_a_failed_reset_is_reported_and_the_session_still_starts(case, why):
    async def go():
        built, log = {}, []
        hub = Hub("procthor-train-40", "sonic", deps=_deps(built, log, **case), cameras=TapCameras(FakeTap()),
                  system1="off")
        notes = []
        hub._notice = notes.append
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic", reset_sim={"robot": True})
        try:
            sr = hub.session.sim_reset
            assert sr["ok"] is False and sr["why"] == why
            assert notes == [f"Scene reset failed: {why}"]
            assert log[-1] == ("runtime",), "the page still gets a session; the failure is on it"
        finally:
            await hub.session.stop()
            hub.close()
    run(go())


def test_a_world_without_reset_scene_is_left_alone():
    """lite (a new LiteWorld is clean on every reset) and an M1 P1 have no reset_scene: skipped, no notice."""
    async def go():
        built = {}
        hub = Hub("procthor-train-40", "lite", deps=make_deps(built), cameras=TapCameras(FakeTap()), system1="off")
        notes = []
        hub._notice = notes.append
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="lite", reset_sim={"robot": True})
        try:
            assert hub.session.sim_reset["skipped"] and notes == []
        finally:
            await hub.session.stop()
            hub.close()
    run(go())


def test_a_halt_latch_left_by_the_last_session_is_released_before_the_stand():
    """A reset is a fresh start: the body's halt latch (the user's last "stop") is released at the body's own halt
    epoch and the stand runs again."""
    async def go():
        built, log = {}, []
        hub = Hub("procthor-train-40", "sonic", deps=_deps(built, log, latched=7), cameras=TapCameras(FakeTap()),
                  system1="off")
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic", reset_sim={"robot": True})
        try:
            assert log == [("reset_scene", "default", True), ("stand", True, 90.0), ("resume", 7),
                           ("stand", True, 90.0), ("runtime",)]
            sr = hub.session.sim_reset
            assert sr["ok"] and sr["resumed_halt_epoch"] == 7 and sr["stand"]["state"] == "succeeded"
        finally:
            await hub.session.stop()
            hub.close()
    run(go())
