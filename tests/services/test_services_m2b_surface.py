"""The services on the M2b body surface (R.2), offline: the lite world and a lite body that also speaks the M2b
surface the services use (fence, acquire / release, approach, scan with scan.hold progress, CarryLock state).

Pinned: navigate leases the body (LOCOMOTION) and releases it, every body op carries the execution's fence; body
refusals keep their meaning (a stale fence -> failed(stale_result) + a stale_result event; a refused lease ->
body_busy); navigate(reach_stance) runs the body's approach (no INTERIM label); the arrival scan is the waist scan
(scan_executor waist: the view sampled in each hold, the ARM_SCRIPT lease), a glance while carrying, and the labelled
in-place turns on a body without the op; the too_far summary without a suggestion never says "navigate to the
suggested location"."""

from __future__ import annotations

import asyncio
import time
import uuid

from api.summaries import summarize
from robot.body_client import BodyOp
from robot.lite_body import LiteBody
from tests.services.conftest import Stack, run


class M2bLiteBody(LiteBody):
    def __init__(self, world, clock, walking=None, *, scan: bool = True):
        super().__init__(world, clock, walking)
        self.has_scan = scan
        self.leases: list[tuple] = []
        self.fences: list[tuple[str, dict]] = []
        self.refuse_lease: str | None = None
        self.go_to_refusal: tuple[str, dict] | None = None
        self.approaches: list[dict] = []
        self.scans: list[dict] = []
        self.carry = False

    def supports(self, what: str) -> bool:
        return what != "scan" or self.has_scan

    def fence(self, execution) -> dict:
        return {"execution_id": execution.execution_id, "generation": 10 + execution.generation,
                "control_epoch": 20 + execution.control_epoch, "session": "rt-test"}

    def arm_state(self) -> dict:
        return {"carry": {"engaged": self.carry}}

    async def acquire(self, execution, mode: str) -> dict:
        if self.refuse_lease:
            return {"ok": False, "reason": self.refuse_lease, "data": {"lease": {"owner": "other"}}}
        self.leases.append(("acquire", execution.execution_id, mode))
        return {"ok": True, "lease": {"owner": execution.execution_id, "mode": mode}}

    async def release(self, execution_id: str) -> dict:
        self.leases.append(("release", execution_id, None))
        return {"ok": True}

    async def go_to(self, x, y, yaw=None, *, speed=None, timeout_s=None, final_pos_tol=None, final_yaw_tol_deg=None,
                    fence=None):
        self.fences.append(("go_to", dict(fence or {})))
        if self.go_to_refusal is not None:
            o = BodyOp("go_to", f"go_to-{uuid.uuid4().hex[:6]}", loop=asyncio.get_running_loop())
            o.finish("failed", {"reason": self.go_to_refusal[0], **self.go_to_refusal[1]})
            return o
        return await super().go_to(x, y, yaw, speed=speed, timeout_s=timeout_s, final_pos_tol=final_pos_tol,
                                   final_yaw_tol_deg=final_yaw_tol_deg)

    async def approach(self, x, y, yaw=None, *, v=None, tol=None, timeout_s=None, fence=None):
        self.approaches.append({"x": x, "y": y, "yaw": yaw, "tol": tol, "fence": dict(fence or {})})
        return await super().go_to(x, y, yaw, speed=v, final_pos_tol=(tol or (0.05, 5))[0])

    async def scan(self, yaw_deg, *, move_s=None, hold_s=None, fence=None, **kw):
        loop = asyncio.get_running_loop()
        op = BodyOp("scan", f"scan-{uuid.uuid4().hex[:6]}", loop=loop)
        rec = {"yaw_deg": list(yaw_deg), "move_s": move_s, "hold_s": hold_s, "fence": dict(fence or {})}
        self.scans.append(rec)
        period = (move_s or 0.8) + (hold_s or 0.8)
        for i, y in enumerate(yaw_deg):
            loop.call_later((i + 1) * period, op._progress_event,
                            {"state": "progress", "data": {"kind": "scan.hold", "i": i, "yaw_deg": y,
                                                           "yaw_cmd_deg": y, "pitch_deg": 0.3, "yaw_err_deg": 0.5}})
        loop.call_later(len(yaw_deg) * period + 0.1, lambda: op.finish("succeeded", {"yaw_err_deg_max": 0.9,
                                                                                   "arm_dev_rad_max": 0.5,
                                                                                   "base_shift_m": 0.004}))
        op._cancel_fn = lambda reason: rec.__setitem__("cancelled", reason)
        return op


def _stack(scan_executor: str = "turn_in_place", **body_kw) -> tuple[Stack, M2bLiteBody]:
    s = Stack(overrides={"scan_executor": scan_executor})
    body = M2bLiteBody(s.world, s.clock, s.robot.stack_profile.g1.get("walking"), **body_kw)
    for holder in (s.robot, s.robot.nav, s.robot.obs, s.robot.manip):
        holder.body = body
    s.body = body
    return s, body


def test_navigate_leases_fences_and_maps_refusals():
    s, body = _stack()
    ev: list[dict] = []

    async def main():
        q = s.robot.events()
        r = await s.run("navigate", {"location": "living_room"})
        assert r.status == "succeeded" and r.data["executor"] == "lite"
        ex_id = r.execution_id
        assert body.leases[:2] == [("acquire", ex_id, "LOCOMOTION"), ("release", ex_id, None)]
        assert body.fences[-1][1] == {"execution_id": ex_id, "generation": 11, "control_epoch": 20,
                                      "session": "rt-test"}
        # a stale fence at the body is a stale result (keyed on data.why), not a navigation failure
        body.go_to_refusal = ("stale_command", {"why": "control_epoch", "halt_epoch": 25})
        r2 = await s.run("navigate", {"location": "kitchen"})
        assert r2.status == "failed" and r2.data["reason"] == "stale_result", r2.summary
        assert "fenced" in r2.summary
        # stale_command without a fence why (a t_wall-stale stream message) is not a stale result
        body.go_to_refusal = ("stale_command", {"t_wall": 1.0})
        r3 = await s.run("navigate", {"location": "kitchen"})
        assert r3.data["reason"] == "stale_command"
        body.go_to_refusal = None
        # another owner's lease: body_busy, nothing sent
        body.refuse_lease = "body_busy"
        n = len(body.fences)
        r4 = await s.run("navigate", {"location": "kitchen"})
        assert r4.status == "failed" and r4.data["reason"] == "body_busy" and len(body.fences) == n
        await asyncio.sleep(0.05)
        while not q.empty():
            ev.append(q.get_nowait())
        assert any(e["type"] == "stale_result" and e.get("execution_id") == r2.execution_id for e in ev)
    run(main())


def test_reach_stance_runs_the_approach_op_without_the_interim_label():
    s, body = _stack()

    async def main():
        s.put_at("bedroom_bed_1b")
        s.robot.nav._last_at = "bedroom_bed_1b"
        await s.run("observe", {"mode": "scan"})
        r = await s.run("check_reachability", {"object_type": "alarm_clock", "object_id": "alarm_clock_1"})
        assert r.data["reason"] == "needs_reposition", r.summary
        n = await s.run("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b",
                                     "stance": r.data["stance"]})
        assert n.status == "succeeded", n.summary
        assert n.data["reposition_op"] == "approach" and "interim" not in n.data and n.data["tol"] == [0.05, 5.0]
        assert body.approaches and body.approaches[-1]["tol"] == (0.05, 5.0)
        assert body.approaches[-1]["fence"]["execution_id"] == n.execution_id
        assert s.robot.timeout_s("navigate", {"location": "reach_stance"}) == s.robot.nav.cfg.approach_timeout_s + 5
    run(main())


def test_the_arrival_scan_is_the_waist_scan_a_glance_while_carrying_and_labelled_turns_without_the_op():
    s, body = _stack("waist")
    s.robot.obs.cfg = type(s.robot.obs.cfg)(executor="waist", waist_move_s=0.05, waist_hold_s=0.1)

    async def main():
        s.put_at("bedroom_bed_1b")
        s.robot.nav._last_at = "bedroom_bed_1b"
        t0 = time.monotonic()
        r = await s.run("observe", {"mode": "scan"})
        d = r.data
        assert r.status == "succeeded" and d["mode"] == "scan" and d["scan_executor"] == "waist", d
        assert len(d["views"]) == 3 and [h["i"] for h in d["scan_holds"]] == [0, 1, 2]
        assert d["scan_body"]["state"] == "succeeded" and time.monotonic() - t0 < 5
        assert body.scans[-1]["yaw_deg"] == [-35.0, 0.0, 35.0]
        assert body.scans[-1]["fence"]["execution_id"] == r.execution_id
        assert ("acquire", r.execution_id, "ARM_SCRIPT") in body.leases
        # carrying (CarryLock): a glance instead, and the result says why
        body.carry = True
        r2 = await s.run("observe", {"mode": "scan"})
        assert r2.data["mode"] == "glance" and r2.data["scan_fallback"] == "glance"
        assert "carrying" in r2.data["scan_note"] and len(body.scans) == 1
    run(main())

    s2, body2 = _stack("waist", scan=False)

    async def main2():
        s2.put_at("bedroom_bed_1b")
        s2.robot.nav._last_at = "bedroom_bed_1b"
        r = await s2.run("observe", {"mode": "scan"})
        assert r.data["scan_executor"] == "turn_in_place" and "INTERIM" in r.data["scan_note"]
        assert "no waist `scan` op" in r.data["scan_note"]
    run(main2())


def test_too_far_without_a_suggestion_never_says_navigate_to_the_suggested_location():
    d = {"reachable": False, "visible": True, "reason": "too_far", "object_id": "apple_1", "distance_m": 1.1,
         "suggest_location": None}
    line = summarize("check_reachability", "succeeded", d)
    assert "suggested location" not in line and "tell the user" in line
    d["suggest_location"] = "kitchen_counter_1b"
    line = summarize("check_reachability", "succeeded", d)
    assert "try kitchen_counter_1b" in line
