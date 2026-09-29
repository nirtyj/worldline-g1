"""SonicArmScriptExecutor (M2b R.1; backend and executor `sonic_arm_script`): SONIC reaches with the body's B.7
`arm_script` op (IK on main.urdf in the body, a min-jerk joint trajectory into SONIC's upper-body override,
docs/contracts/m1.md §3.9), and the object is held by a P1 attach (P1.3, `fixed_joint`).

Label: STEPPING STONE everywhere (the skill's label `stepping_stone`; api.results.STEPPING_STONE_EXECUTORS): the arm
really moves under SONIC, but "the object stays in the hand" is a ground-truth attach, not the Dex3 grip. It is the
labelled fallback behind groot_arms (policy groot_then_script) and the pick executor of the `sonic` profile.

    pick    stance_check   the base at rest (SONIC glides after a reposition) and upright
            pregrasp       arm_script pregrasp to the grasp point (the object's top centre + grasp_above_m):
                           standoff + above, hand open
            grasp          arm_script grasp: the palm to the grasp point, the hand closes to `closure`
            gt_gate        GT: the palm (P1.5 link pose) within attach_max_m of the grasp point, else
                           failed(grasp_missed) (the hand opens and the arm goes back to SONIC)
            attach         P1 attach {id, arm, mode: fixed_joint} (STEPPING STONE)
            lift           arm_script lift (+lift_m, world up)
            carry          arm_script carry: the carry pose, kept above the support's top so the tuck clears its edge;
                           the script ends into a `target` hold with the hand closed = CarryLock (body.state.arm.carry)
            verify         GT, what is in the hand at the end: the object is held (P1 held_by / the attach) and its
                           centre rose >= min_lift_m within the verify window (sampled for verify_window_s right
                           after the lift phase, while the carry tuck starts, plus once at the end) -> succeeded;
                           held but never that high -> failed(held_not_lifted) (still holding it); not held ->
                           failed(grasp_failed). The carry pose may end lower than the start (it hangs the object
                           below the palm once it is off the support), so the end sample alone is not the lift.
    place   lower          arm_script lower: the palm above the free spot the service chose
            detach         P1 detach {id, pose: the spot} (STEPPING STONE: the object is set down there, upright)
            release        arm_script release (the hand opens)
            retract        arm_script retract, ending `stand`: the arm blends back to SONIC's own reference
            (the service verifies where it landed)

Cancel and halt, like groot_arms: a cancel ends the running phase where the arm is (`arm {end, hold_on_end:
measured}`: nothing opens, nothing drops) and the outcome is `cancelled`; a halt is the body's latch (the arms freeze
on the measured pose, hands never open) and the outcome is failed(halted); a fall is failed(fell). Body refusals keep
their meaning (services.common.refusal): halted, stale_result (a fence), body_busy (another lease). Nothing here
walks (PLAN §1.3 #26): the stance is navigate(reach_stance)'s.

GT confinement: every ground-truth read is a WorldModel call (object, robot_pose, palm_position, hands), and the
attach/detach are SimControl calls. Timing is wall time (the body and P1 are wall-clock processes).
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, fields
from typing import Any

from api.types import ServiceHealth
from world.model import NotSupported

from ..common import refusal
from .kinematic_attach import ManipJob, ManipOutcome

EXECUTOR = "sonic_arm_script"
BACKEND = "sonic_arm_script"
GRASP_LABEL = "P1 attach fixed_joint (STEPPING STONE: a ground-truth attach holds the object, not the Dex3 grip)"


@dataclass
class ArmScriptConfig:
    """Profile section `sonic_arm_script:` (StackProfile.raw); unset keys keep these."""
    grasp_above_m: float = 0.03          # grasp point = the object's top centre + this (the body's live pick, §8.4)
    closure: float = 0.6                 # grasp: hand closure (0..1 of the deploy's fist; the live pick used 0.6)
    attach_mode: str = "fixed_joint"     # P1.3: fixed_joint (the mass hangs on the arm) | follow
    attach_max_m: float = 0.10           # GT gate: the palm this close to the grasp point, else grasp_missed
    lift_m: float = 0.06
    min_lift_m: float = 0.03             # verify: the object rose at least this much ...
    verify_window_s: float = 0.6         # ... within this window after the lift phase (P1 streams object poses at
    verify_dt_s: float = 0.05            # 10 Hz: one sample can be stale) or at the end
    carry: bool = True                   # tuck into the carry pose after the lift
    carry_x_m: float = 0.22              # carry pose, pelvis frame (the body's CARRY_B, body/arm_script.py)
    carry_y_m: float = 0.20
    carry_z_min_m: float = -0.02
    carry_clear_m: float = 0.12          # ... but the palm at least this far above the support's top (the tuck
                                         # pulls back at the current height, then goes to it: it clears the edge)
    place_above_m: float = 0.05          # lower: the palm this far above the resting object's top
    max_ik_err_m: float = 0.03           # pregrasp / grasp / lower: accept an IK goal this close (the body's default
                                         # 0.02 refused a grasp 2.2 cm off after SONIC stepped the pelvis back 12 cm
                                         # during the pregrasp, live); the GT gate below judges where the palm got
    pregrasp_settle_s: float = 1.2       # the body's defaults are 2.5 s for pregrasp / grasp / lower (palm-error
    grasp_settle_s: float = 2.0          # windows); shorter here so a pick fits the skill's time budget
    lower_settle_s: float = 1.5
    settle_s: float = 0.8                # lift / carry / retract
    phase_timeout_s: float = 15.0        # one phase, reply to terminal
    cancel_grace_s: float = 3.0
    stance_settle_s: float = 1.5         # wait for SONIC's glide to end before the first phase
    rest_speed: float = 0.05
    rest_wz: float = 0.1

    @classmethod
    def from_dict(cls, d: dict | None) -> "ArmScriptConfig":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            v = d[f.name]
            kw[f.name] = str(v) if f.name == "attach_mode" else bool(v) if f.name == "carry" else float(v)
        return cls(**kw)


@dataclass
class ArmScriptOutcome(ManipOutcome):
    """ManipOutcome + the executor's data (phases with palm errors, the attach, CarryLock) for the result."""
    data: dict = field(default_factory=dict)


def _dist(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _r(v: Any, n: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, n)
    if isinstance(v, (list, tuple)):
        return [_r(x, n) for x in v]
    return v


class SonicArmScriptExecutor:
    backend = BACKEND
    name = EXECUTOR

    def __init__(self, world: Any, body: Any, *, gate: Any = None, events: Any = None,
                 cfg: ArmScriptConfig | None = None, name: str = EXECUTOR):
        self.world = world
        self.body = body
        self.gate = gate
        self.events = events
        self.cfg = cfg or ArmScriptConfig()
        self.name = name
        self._cancel = False
        self._op: Any = None
        self._handle: Any = None

    # ------------------------------------------------------------------ health
    def health(self) -> ServiceHealth:
        sup = getattr(self.body, "supports", None)
        if not callable(sup) or not sup("arm_script"):
            return ServiceHealth(False, "down", "the body has no arm_script op (B.7)")
        caps = self.world.capabilities() if hasattr(self.world, "capabilities") else {}
        if not (caps.get("attach") and caps.get("detach")):
            return ServiceHealth(False, "down", "P1 has no attach/detach op (P1.3)")
        bh = self.body.health() if hasattr(self.body, "health") else ServiceHealth(True, "ok")
        if not bh.ok:
            return ServiceHealth(False, bh.state, f"body: {bh.detail or bh.state}")
        return ServiceHealth(True, "ok", f"{self.name} (STEPPING STONE: B.7 arm script + {GRASP_LABEL})")

    async def cancel(self) -> None:
        """ManipulationService calls this on its own timeout: the running phase ends where the arm is."""
        self._cancel = True
        op = self._op
        if op is not None and not op.done:
            op.cancel("cancel")

    # ------------------------------------------------------------------ helpers
    def _emit(self, type: str, **f: Any) -> None:
        if self.events is not None:
            try:
                self.events.emit(type, **f)
            except Exception:  # noqa: BLE001
                pass

    def _fence(self, job: ManipJob) -> dict:
        f = {"execution_id": job.execution_id, "generation": int(job.generation),
             "control_epoch": int(job.control_epoch)}
        sess = getattr(self.body, "session", None)
        if sess:
            f["session"] = sess
        return {k: v for k, v in f.items() if v not in (None, "")}

    def _stopped(self, handle: Any, job: ManipJob) -> str | None:
        if self.gate is not None and self.gate.halted_since(job.epoch):
            return "halted"
        if getattr(handle, "cancel_requested", False) or self._cancel:
            return "cancelled"
        try:
            if self.world.robot_pose().fallen:
                return "fell"
        except Exception:  # noqa: BLE001
            pass
        return None

    def _holding(self, job: ManipJob) -> bool:
        try:
            return self.world.hands().get(job.arm) == job.object_id
        except Exception:  # noqa: BLE001
            return False

    def _palm(self, arm: str) -> tuple[float, float, float] | None:
        fn = getattr(self.world, "palm_position", None)
        try:
            p = fn(arm) if callable(fn) else None
        except Exception:  # noqa: BLE001
            p = None
        return None if p is None else tuple(float(v) for v in p)   # type: ignore[return-value]

    def _carry_now(self) -> dict:
        fn = getattr(self.body, "arm_state", None)
        try:
            c = dict(((fn() if callable(fn) else {}) or {}).get("carry") or {})
        except Exception:  # noqa: BLE001
            c = {}
        return {k: c.get(k) for k in ("engaged", "carry_arm", "closure", "palm_err_m", "kind") if k in c}

    async def _carry(self, wait_s: float = 0.8) -> dict:
        """body.state.arm.carry once the body reports it (body.state is published at 5 Hz, so right after the carry
        phase it may still show the script running)."""
        t_end = time.monotonic() + wait_s
        c = self._carry_now()
        while not c.get("engaged") and time.monotonic() < t_end:
            await asyncio.sleep(0.1)
            c = self._carry_now()
        return c

    def _closer_spot(self, job: ManipJob, fwd_max: float = 0.38):
        """A free spot on the target inside the arm script's own envelope (pelvis frame: 0.18-fwd_max m ahead, on the
        arm's side within 0.25 m), nearest the robot first; None if there is none. Used when the service's spot is
        out of the arm's IK reach (the service judges reach with the workspace model, the body with its IK)."""
        fn = getattr(self.world, "free_spot", None)
        if not callable(fn) or not job.target:
            return None
        p = self.world.robot_pose()
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        side = 1.0 if job.arm == "left" else -1.0

        def within(x: float, y: float, z: float) -> bool:
            dx, dy = x - p.x, y - p.y
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
            return 0.18 <= fwd <= fwd_max and -0.08 <= side * lat <= 0.25
        try:
            return fn(job.target, job.object_id, near=p, within=within)
        except Exception:  # noqa: BLE001
            return None

    async def _stance(self) -> tuple[str, str] | None:
        t_end = time.monotonic() + self.cfg.stance_settle_s
        while True:
            p = self.world.robot_pose()
            if getattr(p, "fallen", False):
                return "fell", "the robot is down"
            if p.speed <= self.cfg.rest_speed and abs(p.wz) <= self.cfg.rest_wz:
                return None
            if time.monotonic() >= t_end:
                return "base_moving", f"base still moving after {self.cfg.stance_settle_s:.1f} s " \
                                      f"({p.speed:.2f} m/s)"
            await asyncio.sleep(0.05)

    async def _phase(self, job: ManipJob, handle: Any, phases: list[dict], phase: str, **args: Any
                     ) -> tuple[dict, str | None]:
        """One arm_script phase. Returns (terminal {state, data}, why) with why in {None, halted, cancelled, fell,
        timeout}; a refusal comes back as a failed terminal with the body's reason."""
        t0 = time.monotonic()
        why = self._stopped(handle, job)
        if why is not None:
            return {"state": "failed", "data": {"reason": why}}, why
        op = await self.body.arm_script(phase, job.arm, fence=self._fence(job), **args)
        self._op = op
        self._emit("manip.phase", execution_id=job.execution_id, phase=phase, skill=job.skill_id,
                   executor=self.name, label="stepping_stone")
        t_end = t0 + self.cfg.phase_timeout_s
        why = None
        while not op.done:
            why = self._stopped(handle, job)
            if why is None and time.monotonic() > t_end:
                why = "timeout"
            if why is not None:
                if why in ("cancelled", "timeout"):
                    op.cancel(why)                  # the script ends where the arm is (arm end, hold measured)
                break
            await asyncio.sleep(0.02)
        try:
            res = await asyncio.wait_for(op.result(), self.cfg.cancel_grace_s if why else 1.0)
        except asyncio.TimeoutError:
            res = {"state": "failed", "data": {"reason": f"no terminal event ({why or 'phase'})"}}
        self._op = None
        d = res.get("data") or {}
        plan = (getattr(op, "reply", None) or {}).get("data") or {}
        rec: dict[str, Any] = {"phase": phase, "s": round(time.monotonic() - t0, 2), "state": res.get("state"),
                               "op": getattr(op, "id", None)}
        if d.get("reason"):
            rec["reason"] = d.get("reason")
        for k in ("ik_err_m", "move_s"):
            if plan.get(k) is not None:
                rec[k] = _r(plan[k])
        for k in ("palm_err_w_m", "palm_err_b_m"):
            if isinstance(d.get(k), dict):
                rec[k] = {q: d[k].get(q) for q in ("median", "p90", "max")}
        if d.get("pelvis_shift_m") is not None:
            rec["pelvis_shift_m"] = _r(d.get("pelvis_shift_m"))
        if why:
            rec["stopped_by"] = why
        phases.append(rec)
        return res, why

    def _fail_reason(self, res: dict, why: str | None, default: str) -> str:
        if why in ("halted", "fell", "timeout"):
            return why
        d = res.get("data") or {}
        r = d.get("reason")
        m = refusal(r, d)
        if m is not None:
            return m
        if r in ("fault:fallen", "fallen"):
            return "fell"
        if r in ("not_standing", "deploy_stale", "no_pose") or str(r).startswith("fault:"):
            return "controller_unavailable"
        if r == "ik_unreachable":
            return "ik_unreachable"
        if res.get("state") == "canceled" and (d.get("ended_by") == "halt" or r == "halt"):
            return "halted"
        return default

    def _outcome(self, status: str, reason: str | None, job: ManipJob, phase: str, phases: list[dict], t_run: float,
                 detail: str = "", **data: Any) -> ArmScriptOutcome:
        holding = self._holding(job)
        attempt = {"executor": self.name, "skill": job.skill_id, "label": "stepping_stone", "status": status,
                   "reason": reason, "duration_s": round(time.monotonic() - t_run, 2), "phases": len(phases)}
        out = {"executor": self.name, "grasp": GRASP_LABEL, "arm_op": "arm_script (B.7)", "attempts": [attempt],
               **data}
        text = f"{self.name} (STEPPING STONE: scripted arm + GT attach)" + (f"; {detail}" if detail else "")
        return ArmScriptOutcome(status, reason, holding, phase, detail=text[:400], phases=list(phases), data=out)

    async def _abort_arm(self, job: ManipJob, phases: list[dict], handle: Any) -> None:
        """After a failed grasp: open the hand and give the arm back to SONIC (best effort, never raises)."""
        for ph in ("release", "retract"):
            try:
                res, why = await self._phase(job, handle, phases, ph, settle_s=0.5)
            except Exception:  # noqa: BLE001
                return
            if why is not None or res.get("state") != "succeeded":
                return

    async def _sample_rise(self, job: ManipJob, zc0: float, window_s: float) -> dict:
        """The object's rise (GT centre z minus its start) sampled every verify_dt_s for `window_s`: the lift's
        evidence, taken right after the lift phase."""
        rises: list[float] = []
        t_end = time.monotonic() + max(0.0, window_s)
        while True:
            try:
                o = self.world.object(job.object_id)
            except Exception:  # noqa: BLE001
                o = None
            if o is not None:
                rises.append(float(o.pos[2]) - zc0)
            if time.monotonic() >= t_end:
                break
            await asyncio.sleep(self.cfg.verify_dt_s)
        return {"max_m": max(rises) if rises else None, "n": len(rises)}

    # ------------------------------------------------------------------ run
    async def run(self, job: ManipJob, handle: Any) -> ArmScriptOutcome:
        self._cancel = False
        self._handle = handle
        t_run = time.monotonic()
        phases: list[dict] = []
        if job.action == "place":
            return await self._place(job, handle, phases, t_run)
        return await self._pick(job, handle, phases, t_run)

    def _stop_outcome(self, why: str, job: ManipJob, phase: str, phases: list[dict], t_run: float,
                      **data: Any) -> ArmScriptOutcome:
        if why == "cancelled":
            return self._outcome("cancelled", getattr(self._handle, "cancel_reason", None) or "cancelled", job, phase,
                                 phases, t_run, "cancelled: the arm stopped where it was", **data)
        detail = {"halted": "halted: the body latched the arms", "fell": "the robot fell",
                  "timeout": f"phase {phase} ran past {self.cfg.phase_timeout_s:.0f} s"}.get(why, why)
        return self._outcome("failed", why, job, phase, phases, t_run, detail, **data)

    async def _pick(self, job: ManipJob, handle: Any, phases: list[dict], t_run: float) -> ArmScriptOutcome:
        c = self.cfg
        o = self.world.object(job.object_id)
        if o is None:
            return self._outcome("failed", "grasp_missed", job, "select_skill", phases, t_run,
                                 f"unknown object {job.object_id}")
        why = await self._stance()
        phases.append({"phase": "stance_check", "ok": why is None})
        if why is not None:
            return self._outcome("failed", why[0], job, "stance_check", phases, t_run, why[1])
        (x0, y0, z0), (x1, y1, z1) = o.box
        grasp_w = ((x0 + x1) / 2, (y0 + y1) / 2, z1 + c.grasp_above_m)
        support_top = z0
        zc0 = (z0 + z1) / 2
        extra: dict[str, Any] = {"grasp_w": _r(list(grasp_w)), "support_top_m": _r(support_top)}
        # pregrasp, grasp
        res, why = await self._phase(job, handle, phases, "pregrasp", target_w=grasp_w, settle_s=c.pregrasp_settle_s,
                                     max_ik_err_m=c.max_ik_err_m)
        if why is not None:
            return self._stop_outcome(why, job, "pregrasp", phases, t_run, **extra)
        if res.get("state") != "succeeded":
            r = self._fail_reason(res, why, "grasp_failed")
            return self._outcome("failed", r, job, "pregrasp", phases, t_run,
                                 f"pregrasp {res.get('state')}: {(res.get('data') or {}).get('reason')}", **extra)
        res, why = await self._phase(job, handle, phases, "grasp", target_w=grasp_w, closure=c.closure,
                                     settle_s=c.grasp_settle_s, max_ik_err_m=c.max_ik_err_m)
        if why is not None:
            return self._stop_outcome(why, job, "grasp", phases, t_run, **extra)
        if res.get("state") != "succeeded":
            r = self._fail_reason(res, why, "grasp_failed")
            await self._abort_arm(job, phases, handle)
            return self._outcome("failed", r, job, "grasp", phases, t_run,
                                 f"grasp {res.get('state')}: {(res.get('data') or {}).get('reason')}", **extra)
        # GT gate: is the palm at the object? (the attach is the stepping stone, not a free pass)
        palm = self._palm(job.arm)
        src = "p1.link_pose"
        if palm is None:
            pf = (res.get("data") or {}).get("palm_final_w")
            palm, src = (tuple(pf) if isinstance(pf, (list, tuple)) and len(pf) == 3 else None), "body.palm_final_w"
        d_palm = None if palm is None else _dist(palm, grasp_w)
        extra["gt_gate"] = {"palm_to_grasp_m": _r(d_palm), "max_m": c.attach_max_m, "palm_source": src}
        phases.append({"phase": "gt_gate", "ok": d_palm is not None and d_palm <= c.attach_max_m,
                       "palm_to_grasp_m": _r(d_palm)})
        if d_palm is None or d_palm > c.attach_max_m:
            await self._abort_arm(job, phases, handle)
            what = "no palm pose" if d_palm is None else f"palm {d_palm:.3f} m from the grasp point"
            return self._outcome("failed", "grasp_missed", job, "gt_gate", phases, t_run,
                                 f"{what} (> {c.attach_max_m:.2f} m): no attach", **extra)
        why = self._stopped(handle, job)
        if why is not None:
            return self._stop_outcome(why, job, "attach", phases, t_run, **extra)
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(self.world.attach, job.object_id, job.arm, c.attach_mode)
        except NotSupported as e:
            return self._outcome("failed", "controller_unavailable", job, "attach", phases, t_run, str(e), **extra)
        except KeyError as e:
            return self._outcome("failed", "grasp_missed", job, "attach", phases, t_run, f"unknown object {e}", **extra)
        except Exception as e:  # noqa: BLE001 - a P1 error (hand_busy, not_movable, ...)
            await self._abort_arm(job, phases, handle)
            return self._outcome("failed", "grasp_failed", job, "attach", phases, t_run,
                                 f"P1 attach failed: {e}"[:200], **extra)
        phases.append({"phase": "attach", "s": round(time.monotonic() - t0, 3), "mode": c.attach_mode,
                       "stepping_stone": True})
        # lift, then the carry tuck (kept above the support top so it clears the edge)
        res, why = await self._phase(job, handle, phases, "lift", lift_m=c.lift_m, settle_s=c.settle_s)
        if why is not None:
            return self._stop_outcome(why, job, "lift", phases, t_run, **extra)
        if res.get("state") != "succeeded":
            return self._outcome("failed", self._fail_reason(res, why, "grasp_failed"), job, "lift", phases, t_run,
                                 f"lift {res.get('state')}: {(res.get('data') or {}).get('reason')}", **extra)
        # the lift's evidence: the object's height over a short window from here (the carry tuck starts meanwhile;
        # it pulls back at the lift height first)
        sampler = asyncio.ensure_future(self._sample_rise(job, zc0, c.verify_window_s))
        try:
            if c.carry:
                p = self.world.robot_pose()
                side = 1.0 if job.arm == "left" else -1.0
                zb = max(c.carry_z_min_m, support_top + c.carry_clear_m - float(p.z))
                carry_b = [c.carry_x_m, side * c.carry_y_m, zb]
                extra["carry_b"] = _r(carry_b)
                res, why = await self._phase(job, handle, phases, "carry", carry_b=carry_b, settle_s=c.settle_s)
                if why is not None:
                    return self._stop_outcome(why, job, "carry", phases, t_run, **extra)
                if res.get("state") != "succeeded":
                    # the lift hold (hand closed) stays: CarryLock at the lift pose
                    extra["carry_note"] = f"carry tuck {res.get('state')} ({(res.get('data') or {}).get('reason')}); " \
                                          f"holding the lift pose"
            window = await sampler
        finally:
            if not sampler.done():
                sampler.cancel()
        # verify on GT: what is in the hand at the end, and whether it came up within the window
        extra["carry_lock"] = await self._carry()
        o2 = self.world.object(job.object_id)
        held = self._holding(job)
        rose_end = None if o2 is None else float(o2.pos[2]) - zc0
        rises = [v for v in (window.get("max_m"), rose_end) if v is not None]
        rose = max(rises) if rises else None
        lifted = rose is not None and rose >= c.min_lift_m
        extra["verify"] = {"held_by": job.arm if held else None, "rose_m": _r(rose),
                           "rose_after_lift_m": _r(window.get("max_m")), "rose_end_m": _r(rose_end),
                           "window_s": c.verify_window_s, "samples": window.get("n"), "min_lift_m": c.min_lift_m}
        phases.append({"phase": "verify", "ok": bool(held and lifted), "held": held, "lifted": lifted})
        if not held:
            return self._outcome("failed", "grasp_failed", job, "verify", phases, t_run,
                                 "the object is not in the hand after the lift", **extra)
        if not lifted:
            return self._outcome("failed", "held_not_lifted", job, "verify", phases, t_run,
                                 f"in the hand, but it rose only {0.0 if rose is None else rose:.3f} m "
                                 f"(< {c.min_lift_m:.2f}) within {c.verify_window_s:.1f} s of the lift", **extra)
        return self._outcome("succeeded", None, job, "verify", phases, t_run,
                             f"lifted {rose:.2f} m; CarryLock {'engaged' if extra['carry_lock'].get('engaged') else 'not reported'}",
                             **extra)

    async def _place(self, job: ManipJob, handle: Any, phases: list[dict], t_run: float) -> ArmScriptOutcome:
        c = self.cfg
        o = self.world.object(job.object_id)
        spot = job.spot
        if o is None or spot is None:
            return self._outcome("failed", "nothing_in_hand" if o is None else "no_room_on_surface", job,
                                 "select_skill", phases, t_run, "no object or no spot")
        (x0, y0, z0), (x1, y1, z1) = o.box
        half_h = (z1 - z0) / 2
        sx, sy, sz = float(spot.x), float(spot.y), float(spot.z)
        place_w = (sx, sy, sz + half_h + c.place_above_m)
        extra: dict[str, Any] = {"place_w": _r(list(place_w)), "spot": _r([sx, sy, sz])}
        why = await self._stance()
        phases.append({"phase": "stance_check", "ok": why is None})
        if why is not None:
            return self._outcome("failed", why[0], job, "stance_check", phases, t_run, why[1], **extra)
        res, why = await self._phase(job, handle, phases, "lower", target_w=place_w, settle_s=c.lower_settle_s,
                                     max_ik_err_m=c.max_ik_err_m)
        if why is None and res.get("state") != "succeeded" and \
                (res.get("data") or {}).get("reason") == "ik_unreachable":
            spot2 = self._closer_spot(job)                   # the service's spot is beyond the arm: a closer one
            if spot2 is not None:
                sx, sy, sz = float(spot2.x), float(spot2.y), float(spot2.z)
                place_w = (sx, sy, sz + half_h + c.place_above_m)
                extra.update(spot_retry={"from": extra["spot"], "to": _r([sx, sy, sz])}, place_w=_r(list(place_w)),
                             spot=_r([sx, sy, sz]))
                res, why = await self._phase(job, handle, phases, "lower", target_w=place_w,
                                             settle_s=c.lower_settle_s, max_ik_err_m=c.max_ik_err_m)
        if why is not None:
            return self._stop_outcome(why, job, "lower", phases, t_run, **extra)
        if res.get("state") != "succeeded":
            r = self._fail_reason(res, why, "no_room_in_reach")
            if r == "ik_unreachable":
                r = "no_room_in_reach"
            return self._outcome("failed", r, job, "lower", phases, t_run,
                                 f"lower {res.get('state')}: {(res.get('data') or {}).get('reason')}", **extra)
        palm = self._palm(job.arm)
        extra["palm_to_place_m"] = None if palm is None else _r(_dist(palm, place_w))
        why = self._stopped(handle, job)
        if why is not None:
            return self._stop_outcome(why, job, "detach", phases, t_run, **extra)
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(self.world.detach, job.object_id, (sx, sy, sz))
        except NotSupported as e:
            return self._outcome("failed", "controller_unavailable", job, "detach", phases, t_run, str(e), **extra)
        except Exception as e:  # noqa: BLE001
            return self._outcome("failed", "object_dropped", job, "detach", phases, t_run,
                                 f"P1 detach failed: {e}"[:200], **extra)
        phases.append({"phase": "detach", "s": round(time.monotonic() - t0, 3), "stepping_stone": True})
        for ph in ("release", "retract"):
            res, why = await self._phase(job, handle, phases, ph, settle_s=c.settle_s)
            if why is not None:
                # the object is down already: a stop now only leaves the arm where it is
                extra["after_detach"] = f"{ph} stopped ({why})"
                if why == "cancelled":
                    break
                return self._stop_outcome(why, job, ph, phases, t_run, **extra)
            if res.get("state") != "succeeded":
                extra["after_detach"] = f"{ph} {res.get('state')} ({(res.get('data') or {}).get('reason')})"
                break
        return self._outcome("succeeded", None, job, "retract", phases, t_run,
                             "set down at the spot; arm back to SONIC", **extra)


__all__ = ["EXECUTOR", "BACKEND", "GRASP_LABEL", "ArmScriptConfig", "ArmScriptOutcome", "SonicArmScriptExecutor"]
