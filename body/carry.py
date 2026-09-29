"""Holds on the arm channel, and CarryLock (PLAN §1.3 #12, docs/M2.md B.7).

A *hold* is what the arm channel (body/arm.py) keeps sending on every planner message when no session is streaming:
the upper body (17, `joint_map.UPPER_BODY_MUJOCO_JOINTS` order) and, if any, both Dex3 hands. The body's servo keeps
running on the held pose, so the arm stays where the hold says (SONIC alone leaves 0.1-0.3 rad of pose-dependent
error, docs/arm_tracking.md §3.3). Kinds:

    target    the last target of the session that ended (`hold_on_end: "target"`). After a grasp or a lift with the
              hand closed this IS CarryLock: the carry pose (arm + closed hand) rides on every later planner message,
              including while a `walk` / `go_to` owns the legs (the override composes with any leg motion).
    measured  the measured upper body at the end tick (g1_debug body_q, through joint_map) and the hands' last target
              (`hold_on_end: "measured"`): stop where the arm is, drop nothing.
    latched   the halt latch (ArmChannel.latch): the measured upper body and the measured hands. Nobody may take it
              over before `resume`; afterwards a new owner with control_epoch > halt_epoch may, or `end` / `release`
              blends it back to SONIC.
    watchdog  a v0.5 target stream that went silent (hold_s, then blend); the op is still alive and its owner may
              resume.

A hold ends by a take-over (a new session continues from the pose being sent), `release` / `end` (min-jerk blend to
SONIC's own reference, then the override is dropped), `stop {arms: true}` (the same blend), or a fault (dropped at
once).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import joint_map as jm

HOLD_KINDS = ("target", "measured", "latched", "watchdog")


@dataclass
class Hold:
    kind: str                                   # HOLD_KINDS
    pose: list[float]                           # mj17 servo reference (what the arm should be at)
    hands: dict = field(default_factory=lambda: {"left": None, "right": None})   # Dex3 order, or None per hand
    waist_mode: str = "ref"                     # "ref": SONIC's reference waist; "cmd": pose[0:3] is sent
    stream: str | None = None                   # the stream that left it (release / end by that stream)
    op_id: str | None = None                    # the op that left it
    source: str = ""                            # where `pose` came from (evidence)
    t_mono: float = field(default_factory=time.monotonic)
    epoch: int | None = None                    # latched: the halt epoch

    def closure(self) -> dict:
        """Closure fraction per hand (0 open .. 1 the deploy's fist), None for a hand that is not held."""
        out = {}
        for s in ("left", "right"):
            h = self.hands.get(s)
            out[s] = None if h is None else round(jm.hand_closure_of(s, h), 3)
        return out

    def is_carry(self, closed_min: float = 0.3) -> bool:
        """CarryLock = a target hold with at least one hand closed."""
        return self.kind == "target" and any(c is not None and c >= closed_min for c in self.closure().values())

    def brief(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        return {"kind": self.kind, "stream": self.stream, "op": self.op_id, "source": self.source,
                "waist": self.waist_mode, "age_s": round(now - self.t_mono, 2), "closure": self.closure(),
                "carry": self.is_carry(), "epoch": self.epoch,
                "pose_mj17": [round(v, 4) for v in self.pose]}


def carry_info(hold: Hold | None, measured_mj17: list[float] | None, now: float | None = None) -> dict:
    """body.state.arm.carry: whether CarryLock is engaged and, when it is, the palm error of the measured arm against
    the held pose (FK on main.urdf, pelvis frame, both on the measured waist)."""
    if hold is None or not hold.is_carry():
        return {"engaged": False}
    out = {"engaged": True, **hold.brief(now)}
    out.pop("pose_mj17", None)
    if measured_mj17 is not None:
        from . import g1_kin as K

        want = K.named_from_mj17(hold.pose)
        got = K.named_from_mj17(measured_mj17)
        for w in jm.WAIST_JOINTS:                   # compare arms on the same (measured) waist
            want[w] = got[w]
        pw, pg = K.points(want), K.points(got)
        out["palm_err_m"] = {s: round(float(((pw[f"{s}_palm"] - pg[f"{s}_palm"]) ** 2).sum() ** 0.5), 4)
                             for s in ("left", "right")}
    return out
