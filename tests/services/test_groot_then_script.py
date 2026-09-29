"""ManipulationService x groot_arms (integration, M2b wave 1): the executor's typed data reaches the result, and the
`full` profile's policy `groot_then_script` (PLAN §6.4) hands a failed GR00T attempt over to the labelled fallback.

Offline, on the lite stack (H38) with the fake PolicyServer and the fake chunk-mode arm body, like
test_groot_arms_service.py. The lite executor stands in for the fallback (on Isaac: sonic_arm_script, else
kinematic_attach); both are STEPPING STONEs, so the result says `fallback`.

Pinned:
- a GR00T attempt that fails zero-shot (grasp_missed) is followed at once by the fallback from the same stance; the
  result is the fallback's (executor, skill, stepping_stone, `fallback` in the summary), data.attempts lists both,
  data.fallback_from names the GR00T attempt and data.groot keeps its numbers;
- a halt during the GR00T attempt ends failed(halted) with no fallback;
- first_healthy (every other profile) never retries;
- the manipulate tool timeout leaves room for the fallback.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("groot.actions")
registry = pytest.importorskip("services.executors.registry")

from tests.services.conftest import Stack, run  # noqa: E402
from tests.services.test_groot_arms_service import Offline, _ready, _set_speed  # noqa: E402

ANY = "groot.pick.any.arena_static_experimental.v0"
PICK = {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1", "arm": "left"}


def _stack(o: Offline, policy: str) -> Stack:
    registry.register_executor("groot_arms_offline", o.factory, backend="groot")
    return Stack(overrides={"manipulation": {"executors": ["groot_arms_offline", "lite"], "policy": policy}})


async def _healthy(s: Stack, o: Offline) -> None:
    _set_speed(s.clock, 1.0)                       # GR00T runs in wall time
    t_end = time.monotonic() + 5.0
    while not o.exe.health().ok and time.monotonic() < t_end:
        await s.clock.sleep(0.05)
    assert o.exe.health().ok


def test_a_failed_groot_attempt_hands_over_to_the_labelled_fallback():
    o = Offline()                                  # hands close after 1 inference; nothing couples a lift: zero-shot
    try:
        s = _stack(o, "groot_then_script")
        assert s.robot.manip.policy == "groot_then_script"

        async def main():
            await _ready(s)
            await _healthy(s, o)
            p = await s.run("manipulate", dict(PICK))
            d = p.data
            assert p.status == "succeeded", (p.summary, d.get("detail"))
            assert d["executor"] == "lite" and d["skill"] == "lite.pick.v0" and d["stepping_stone"] is True
            assert "fallback" in p.summary
            assert [(a["executor"], a["status"]) for a in d["attempts"]] == [("groot_arms", "failed"),
                                                                           ("lite", "succeeded")]
            first = d["attempts"][0]
            assert first["skill"] == ANY and first["label"] == "experimental" and first["reason"] == "grasp_missed"
            assert d["fallback_from"] == {"skill": ANY, "executor": "groot_arms", "status": "failed",
                                          "reason": "grasp_missed"}
            assert d["inferences"] >= 1 and isinstance(d["chunks_dropped"], dict)    # the GR00T attempt's numbers
            g = d["groot"]
            assert g["label"] == "experimental" and g["latency_ms"]["n"] >= 1 and g["session_id"]
            assert "label" not in d                # the flat keys describe the attempt that produced the result
            assert d["detail"].startswith("after groot_arms failed(grasp_missed)")
            phases = d["phases"]
            assert phases[0]["phase"] == "stance_check" and phases[-1].get("attempt") == 2
            assert s.world.hands()["left"] == "alarm_clock_1"
            ev = [e for e in s.log.events if e.get("type") == "manip.fallback"]
            assert ev and ev[-1]["from_executor"] == "groot_arms" and ev[-1]["to_executor"] == "lite"
        run(main())
    finally:
        o.close()


def test_a_halt_during_the_groot_attempt_is_not_retried():
    o = Offline()
    o.srv.close_after = None                      # open hands: no grasp_missed, the session runs until the halt
    try:
        s = _stack(o, "groot_then_script")

        async def main():
            await _ready(s)
            await _healthy(s, o)
            h = s.robot.start(s.ex("manipulate", dict(PICK)))
            t_end = time.monotonic() + 5.0
            while (o.exe.last_session is None or o.exe.last_session.chunks_sent < 1) and time.monotonic() < t_end:
                await s.clock.sleep(0.05)
            receipt = s.robot.halt()
            assert receipt["stopped"] is True
            p = await h.result()
            d = p.data
            assert p.status == "failed" and d["reason"] == "halted", (p.summary, d.get("detail"))
            assert d["executor"] == "groot_arms" and d["skill"] == ANY
            assert [a["executor"] for a in d["attempts"]] == ["groot_arms"] and "fallback_from" not in d
            assert d["label"] == "experimental" and d["inferences"] >= 1      # executor data merged flat
            assert s.world.hands()["left"] is None
        run(main())
    finally:
        o.close()


def test_first_healthy_never_retries_and_the_timeout_leaves_room_for_the_fallback():
    o, o2 = Offline(), Offline()
    try:
        g = _stack(o2, "groot_then_script")
        s = _stack(o, "first_healthy")
        skill = s.robot.registry().select("pick", "alarm_clock", "left")
        assert skill.skill_id == ANY
        assert s.robot.manip.fallback_for(skill, "pick", "alarm_clock", "left") is None
        fb = g.robot.manip.fallback_for(skill, "pick", "alarm_clock", "left")
        assert fb is not None and fb.skill_id == "lite.pick.v0"
        assert g.robot.manip.fallback_for(skill, "pick", "alarm_clock", "right") is not None
        assert s.robot.timeout_s("manipulate", PICK) == skill.timeout_s()
        assert g.robot.timeout_s("manipulate", PICK) == pytest.approx(
            skill.timeout_s() + fb.max_duration_s + g.robot.manip.cfg.verify_hold_s)

        async def main():
            await _ready(s)
            await _healthy(s, o)
            p = await s.run("manipulate", dict(PICK))
            assert p.status == "failed" and p.data["reason"] == "grasp_missed", p.summary
            assert p.data["executor"] == "groot_arms" and len(p.data["attempts"]) == 1
            assert s.world.hands()["left"] is None
        run(main())
    finally:
        o.close()
        o2.close()
