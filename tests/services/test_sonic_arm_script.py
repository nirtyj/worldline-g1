"""R.1 SonicArmScriptExecutor (STEPPING STONE: the body's B.7 arm_script + a P1 attach) through ManipulationService,
offline: a lite world (in-process attach/detach) and a scripted body that answers arm_script phases like wl-body
(accepted with a plan, then a terminal event with the palm error; docs/contracts/m1.md §3.9).

Pinned: the phase order (pregrasp, grasp, the GT gate, the attach, lift, carry; lower, detach, release, retract), the
GT gate (no attach when the palm is not at the object: grasp_missed, the hand opens, the arm goes back), cancel ends
the running phase where the arm is (arm end, hold measured) -> cancelled, a halt -> failed(halted) with no attach,
the body lease (ARM_SCRIPT) and fences, and groot_then_script with this executor as the fallback: both attempts in
data.attempts, the fallback's own data flat, the GR00T attempt's under data.groot."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from api.types import ServiceHealth
from robot.body_client import BodyOp
from services.executors import registry
from services.executors.kinematic_attach import ManipOutcome
from services.executors.sonic_arm_script import ArmScriptConfig, SonicArmScriptExecutor
from tests.services.conftest import Stack, run

PICK = {"action": "pick", "object_type": "alarm_clock", "object_id": "alarm_clock_1", "arm": "left"}


class ScriptBody:
    """The M2b body surface the executor and the service use, scripted: arm_script phases take `phase_s` and end
    succeeded with the palm at the goal (or `palm_off` metres away), `fail` maps a phase to a refusal reason."""

    name = "sonic_walk"

    def __init__(self, phase_s: float = 0.05):
        self.phase_s = phase_s
        self.calls: list[dict] = []
        self.leases: list[tuple] = []
        self.fail: dict[str, str] = {}
        self.palm_off = 0.0
        self.session = "rt-test"
        self.carry_engaged = False
        self._palm = (0.0, 0.0, 1.0)

    # surface
    def supports(self, what: str) -> bool:
        return True

    def health(self) -> ServiceHealth:
        return ServiceHealth(True, "ok")

    def arm_state(self) -> dict:
        return {"carry": {"engaged": self.carry_engaged, "carry_arm": "left" if self.carry_engaged else None}}

    def fence(self, execution) -> dict:
        return {"execution_id": execution.execution_id, "generation": 100 + execution.generation,
                "control_epoch": 50 + execution.control_epoch, "session": self.session}

    async def acquire(self, execution, mode: str) -> dict:
        self.leases.append(("acquire", execution.execution_id, mode))
        return {"ok": True, "lease": {"owner": execution.execution_id, "mode": mode}}

    async def release(self, execution_id: str) -> dict:
        self.leases.append(("release", execution_id, None))
        return {"ok": True}

    async def arm_script(self, phase: str, arm: str, *, target_w: Any = None, fence: dict | None = None,
                         **extra: Any) -> BodyOp:
        loop = asyncio.get_running_loop()
        rec = {"phase": phase, "arm": arm, "target_w": target_w, "fence": dict(fence or {}), **extra}
        self.calls.append(rec)
        op = BodyOp("arm_script", f"as-{phase}-{len(self.calls)}", loop=loop)
        if phase in self.fail:
            op.finish("failed", {"reason": self.fail[phase], **({"why": "control_epoch"}
                                                              if self.fail[phase] == "stale_command" else {})})
            return op
        op.reply = {"ok": True, "state": "accepted", "data": {"ik_err_m": 0.001, "move_s": 0.5}}
        if target_w is not None:
            self._palm = (target_w[0] + self.palm_off, target_w[1], target_w[2])
        palm = list(self._palm)

        def end(state="succeeded", **d):
            if not op.done:
                op.finish(state, {"phase": phase, "palm_final_w": palm, "ended_by": d.pop("ended_by", "script"),
                                  "palm_err_w_m": {"median": 0.01, "p90": 0.02, "max": 0.02}, **d})

        def cancel(reason):
            rec["cancelled"] = reason
            loop.call_later(0.02, lambda: end("succeeded", ended_by="client", hold="measured"))
        op._cancel_fn = cancel
        loop.call_later(self.phase_s, end)
        if phase in ("lift", "carry") or (phase == "grasp" and extra.get("closure")):
            self.carry_engaged = phase in ("grasp", "lift", "carry")
        if phase in ("release", "retract"):
            self.carry_engaged = False
        return op


def _stack(body: ScriptBody, cfg: dict | None = None, executors=("sonic_arm_script_test", "lite"),
           policy: str = "first_healthy") -> Stack:
    c = {"min_lift_m": -1.0, **(cfg or {})}            # the lite world's attach does not lift the box: see below

    def make(ctx):
        return SonicArmScriptExecutor(ctx.world, body, gate=ctx.gate, events=ctx.events,
                                      cfg=ArmScriptConfig.from_dict(c))
    registry.register_executor("sonic_arm_script_test", make, backend="sonic_arm_script")
    s = Stack(overrides={"manipulation": {"executors": list(executors), "policy": policy}})
    s.robot.manip.body = body                         # the service leases and fences through this body
    return s


async def _at_dresser(s: Stack, kp: str = "bedroom_bed_1b"):
    """At a reachable stance for H38's alarm clock (like tests/services/test_services_manipulation.py)."""
    s.put_at(kp)
    s.robot.nav._last_at = kp
    await s.run("observe", {"mode": "scan"})
    r = await s.run("check_reachability", {"object_type": "alarm_clock", "object_id": "alarm_clock_1"})
    if r.data["reason"] == "needs_reposition":
        await s.run("navigate", {"location": "reach_stance", "anchor": kp, "stance": r.data["stance"]})
        r = await s.run("check_reachability", {"object_type": "alarm_clock", "object_id": "alarm_clock_1"})
    assert r.data["reachable"], r.summary
    _set_speed(s.clock, 1.0)                           # setup runs fast; the arm runs in wall time (the body's)
    return r


def _set_speed(clock, speed: float) -> None:
    now = clock.now()
    clock.speed = float(speed)
    clock._t0 = time.monotonic() - now / clock.speed


def test_pick_runs_the_b7_phases_attaches_and_holds_carry_then_place():
    body = ScriptBody()
    s = _stack(body)

    async def main():
        await _at_dresser(s)
        box0 = s.world.object("alarm_clock_1").box
        p = await s.run("manipulate", dict(PICK))
        d = p.data
        assert p.status == "succeeded", (p.summary, d.get("detail"))
        assert d["executor"] == "sonic_arm_script" and d["skill"] == "sonic.script.pick.v0"
        assert d["stepping_stone"] is True and "fallback" in p.summary
        assert "STEPPING STONE" in d["grasp"] and d["arm_op"].startswith("arm_script")
        assert [c["phase"] for c in body.calls] == ["pregrasp", "grasp", "lift", "carry"]
        names = [ph["phase"] for ph in d["phases"]]
        assert names == ["stance_check", "pregrasp", "grasp", "gt_gate", "attach", "lift", "carry", "verify"]
        assert d["gt_gate"]["palm_to_grasp_m"] <= 0.01 and d["gt_gate"]["palm_source"] == "body.palm_final_w"
        assert d["carry_lock"]["engaged"] is True and d["attempts"][0]["executor"] == "sonic_arm_script"
        # the grasp point is the object's top centre + grasp_above_m; every phase carries the body's fence
        assert body.calls[0]["target_w"][2] == pytest.approx(box0[1][2] + 0.03)
        f = s.body.fence if hasattr(s.body, "fence") else None
        assert all(c["fence"]["control_epoch"] == 50 and c["fence"]["generation"] == 101 and
                   c["fence"]["execution_id"] == p.execution_id and c["fence"]["session"] == "rt-test"
                   for c in body.calls), (body.calls, f)
        assert ("acquire", p.execution_id, "ARM_SCRIPT") in body.leases and body.leases[-1][0] == "release"
        assert s.world.hands()["left"] == "alarm_clock_1"
        # place: lower above the free spot, detach there (STEPPING STONE), release, retract
        body.calls.clear()
        target = s.world.map.user_surface                  # its keypoint has the surface's name
        s.put_at(target)
        s.robot.nav._last_at = target
        await s.run("observe", {"mode": "glance"})
        pl = await s.run("manipulate", {"action": "place", "object_type": "alarm_clock", "arm": "left",
                                        "target": target})
        assert pl.status == "succeeded", (pl.summary, pl.data.get("detail"))
        assert [c["phase"] for c in body.calls] == ["lower", "release", "retract"]
        assert [ph["phase"] for ph in pl.data["phases"]][:3] == ["stance_check", "lower", "detach"]
        assert s.world.object("alarm_clock_1").where == target and s.world.hands()["left"] is None
    run(main())


def test_gt_gate_miss_never_attaches_and_gives_the_arm_back():
    body = ScriptBody()
    body.palm_off = 0.25                              # the palm stops 25 cm from the grasp point
    s = _stack(body, executors=("sonic_arm_script_test",))

    async def main():
        await _at_dresser(s)
        p = await s.run("manipulate", dict(PICK))
        assert p.status == "failed" and p.data["reason"] == "grasp_missed", p.summary
        assert "no attach" in p.data["detail"] and p.data["gt_gate"]["palm_to_grasp_m"] > 0.2
        assert [c["phase"] for c in body.calls] == ["pregrasp", "grasp", "release", "retract"]
        assert s.world.hands()["left"] is None and p.data["holding"] is False
    run(main())


def test_cancel_ends_the_phase_where_the_arm_is_and_a_halt_is_failed_halted():
    body = ScriptBody(phase_s=3.0)
    s = _stack(body, executors=("sonic_arm_script_test",))

    async def main():
        await _at_dresser(s)
        h = s.robot.start(s.ex("manipulate", dict(PICK)))
        t_end = time.monotonic() + 3.0
        while not body.calls and time.monotonic() < t_end:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        h.cancel("correction")
        p = await h.result()
        assert p.status == "cancelled" and p.data["reason"] == "correction", p.summary
        assert body.calls[-1]["phase"] == "pregrasp" and body.calls[-1]["cancelled"] == "cancelled"
        assert len(body.calls) == 1 and s.world.hands()["left"] is None
        # a halt mid-phase: the body's latch holds the arm; failed(halted), nothing attached, no fallback
        body.calls.clear()
        _set_speed(s.clock, 40.0)
        await _at_dresser(s)
        h = s.robot.start(s.ex("manipulate", dict(PICK)))
        while not body.calls and time.monotonic() < t_end + 5:
            await asyncio.sleep(0.01)
        rec = s.robot.halt()
        assert rec["stopped"]
        p = await h.result()
        assert p.status == "failed" and p.data["reason"] == "halted", p.summary
        assert s.world.hands()["left"] is None and len(p.data["attempts"]) == 1
        s.robot.resume(s.epoch + 1)
    run(main())


def test_a_stale_fence_is_a_stale_result_not_a_failure_of_the_arm():
    body = ScriptBody()
    body.fail["pregrasp"] = "stale_command"
    s = _stack(body, executors=("sonic_arm_script_test",))

    async def main():
        await _at_dresser(s)
        p = await s.run("manipulate", dict(PICK))
        assert p.status == "failed" and p.data["reason"] == "stale_result", p.summary
    run(main())


class FailingGroot:
    """A GR00T stand-in (backend groot) that fails zero-shot with its own typed data."""
    backend = "groot"
    name = "groot_arms"

    def health(self):
        return ServiceHealth(True, "ok", "fake")

    async def cancel(self):
        return None

    async def run(self, job, handle):
        out = ManipOutcome("failed", "grasp_missed", False, "execute", detail="zero-shot",
                           phases=[{"phase": "execute"}])
        out.data = {"executor": "groot_arms", "label": "experimental", "inferences": 7,
                    "chunks_dropped": {"expired": 1}, "latency_ms": {"p50": 150.0},
                    "attempts": [{"executor": "groot_arms", "skill": job.skill_id, "label": "experimental",
                                  "status": "failed", "reason": "grasp_missed", "inferences": 7}]}
        return out


def test_groot_then_script_falls_back_to_the_arm_script_with_both_attempts_merged():
    body = ScriptBody()
    registry.register_executor("groot_failing_test", lambda ctx: FailingGroot(), backend="groot")
    s = _stack(body, executors=("groot_failing_test", "sonic_arm_script_test"), policy="groot_then_script")

    async def main():
        await _at_dresser(s)
        p = await s.run("manipulate", dict(PICK))
        d = p.data
        assert p.status == "succeeded", (p.summary, d.get("detail"))
        assert d["executor"] == "sonic_arm_script" and "fallback" in p.summary and "after groot_arms" in p.summary
        assert [(a["executor"], a["status"]) for a in d["attempts"]] == [("groot_arms", "failed"),
                                                                       ("sonic_arm_script", "succeeded")]
        assert d["inferences"] == 7 and d["chunks_dropped"] == {"expired": 1}     # typed: the GR00T attempt's
        assert d["groot"]["latency_ms"] == {"p50": 150.0} and d["groot"]["label"] == "experimental"
        assert "STEPPING STONE" in d["grasp"] and d["carry_lock"]["engaged"] is True   # flat: the fallback's
        assert d["fallback_from"]["reason"] == "grasp_missed"
        modes = [m for a, _, m in body.leases if a == "acquire"]
        assert modes == ["ARM_STREAM", "ARM_SCRIPT"] and body.leases[-1][0] == "release"
        r = s.robot.manip._results[p.execution_id]
        assert r.inferences == 7 and len(r.attempts) == 2
    run(main())
