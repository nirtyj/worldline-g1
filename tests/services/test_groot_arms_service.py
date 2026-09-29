"""groot_arms inside the real runtime path, offline: G1Robot -> ManipulationService -> the executor registry
(services/executors/registry.py, the world owner's registration point) -> GrootArmExecutor, on the lite stack (H38),
with the fake PolicyServer and the fake chunk-mode arm body. The lite world plays the physics part: when the measured
left hand closes, the fake body's tick attaches the alarm clock to it (LiteWorld's kinematic attach), which is what
ground truth then sees as a lift.

What this pins: CAPABILITY uses the executor's health (a dead policy is `rejected("policy unavailable: ...")` and the
enum is unchanged), and a GR00T pick through the service ends `succeeded` with executor `groot_arms`, the GR00T skill,
label `experimental` and the service's own hand verification.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("groot.actions")
registry = pytest.importorskip("services.executors.registry")

from api.execution import Rejected  # noqa: E402
from services.executors.groot_arms import GrootArmExecutor, _groot_helpers, hand_closure  # noqa: E402
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from tests.services.conftest import Stack, run  # noqa: E402
from tests.services.test_groot_arms import fast_cfg  # noqa: E402


def _set_speed(clock, speed: float) -> None:
    """Continue the session clock at another speed (setup runs fast; GR00T runs in wall time)."""
    now = clock.now()
    clock.speed = float(speed)
    clock._t0 = time.monotonic() - now / clock.speed


class Offline:
    def __init__(self, **cfg):
        self.srv = FakePolicyServer(close_after=1).start()
        self.body = FakeArmBody()
        self.exe: GrootArmExecutor | None = None
        self.cfg = cfg

    def factory(self, ctx):
        # view_min_px 1: lite's ego_view detections are the geometric estimate; this test is about plumbing
        self.exe = GrootArmExecutor(ctx.world, arm=self.body, sensors=self.body,
                                    cfg=fast_cfg(self.srv, view_min_px=1.0, max_duration_s=8.0, **self.cfg),
                                    gate=ctx.gate, events=ctx.events, helpers=_groot_helpers())
        return self.exe

    def grasp_by_attach(self, world, oid: str) -> None:
        state = {"t": None, "done": False}

        def tick(b: FakeArmBody, now: float) -> None:
            if state["done"]:
                return
            if hand_closure("left", b.hand_q["left"]) >= 0.6:
                state["t"] = state["t"] or now
                if now - state["t"] >= 0.2:
                    world.attach(oid, "left", "kinematic")
                    state["done"] = True
        self.body.on_tick(tick)

    def close(self):
        if self.exe is not None:
            self.exe.close()
        self.body.close()
        self.srv.stop()


async def _ready(s, kp="bedroom_bed_1b", otype="alarm_clock", oid="alarm_clock_1"):
    s.put_at(kp)
    s.robot.nav._last_at = kp
    await s.run("observe", {"mode": "scan"})
    r = await s.run("check_reachability", {"object_type": otype, "object_id": oid})
    if r.data["reason"] == "needs_reposition":
        await s.run("navigate", {"location": "reach_stance", "anchor": kp, "stance": r.data["stance"]})
        r = await s.run("check_reachability", {"object_type": otype, "object_id": oid})
    assert r.data["reachable"] and r.data["preferred_arm"] == "left", r.summary     # the GR00T skill is left-handed
    return r


def test_groot_pick_through_the_manipulation_service():
    o = Offline()
    registry.register_executor("groot_arms_offline", o.factory, backend="groot")
    try:
        s = Stack(overrides={"manipulation": {"executors": ["groot_arms_offline", "lite"]}})
        o.grasp_by_attach(s.world, "alarm_clock_1")
        assert s.robot.executors["groot"] is o.exe

        async def main():
            await _ready(s)
            _set_speed(s.clock, 1.0)
            t_end = time.monotonic() + 5.0
            while not o.exe.health().ok and time.monotonic() < t_end:
                await s.clock.sleep(0.05)
            p = await s.run("manipulate", {"action": "pick", "object_type": "alarm_clock",
                                           "object_id": "alarm_clock_1", "arm": "left"})
            assert p.status == "succeeded", (p.summary, p.data.get("detail"))
            d = p.data
            assert d["executor"] == "groot_arms" and d["skill"] == "groot.pick.any.arena_static_experimental.v0"
            assert d["skill_label"] == "experimental" and d["stepping_stone"] is False and d["holding"] is True
            assert s.world.hands()["left"] == "alarm_clock_1"
            assert "groot_arms" in d.get("detail", "") and "experimental" in d.get("detail", "")
            assert [ph["phase"] for ph in d["phases"]][-2:] == ["execute", "carry_lock"]
            assert o.srv.prompts[-1] == "move the alarm clock to the plate"
            assert o.body.messages("end", o.exe.last_session.id)[-1]["args"]["hold_on_end"] == "target"
        run(main())
    finally:
        o.close()


def test_a_dead_policy_is_a_capability_rejection_and_the_enum_stays():
    o = Offline()
    registry.register_executor("groot_arms_offline", o.factory, backend="groot")
    try:
        s = Stack(overrides={"manipulation": {"executors": ["groot_arms_offline"]}})
        t_end = time.monotonic() + 5.0
        while not o.exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        types = s.robot.registry().loaded_object_types()
        assert "apple" in types and "alarm_clock" in types
        ok, why = s.robot.manip.capability("pick", "apple", "left")
        assert ok is not None and ok.skill_id == "groot.pick.apple.arena_static_experimental.v0" and why == ""
        o.srv.die()
        t_end = time.monotonic() + 3.0
        while o.exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        with pytest.raises(Rejected) as e:
            s.robot.start(s.ex("manipulate", {"action": "pick", "object_type": "apple", "arm": "left"}))
        assert e.value.stage == "capability" and e.value.code == "policy_unavailable"
        assert e.value.message.startswith("policy unavailable: groot.pick.apple.arena_static_experimental.v0: ")
        assert o.srv.endpoint in e.value.message
        assert s.robot.registry().loaded_object_types() == types
    finally:
        o.close()
