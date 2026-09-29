"""B.7 ArmScript: scripted reach / grasp / lift / carry / lower / release / retract through the arm channel.

Op `arm_script` (handler `ArmChannel.handle_arm_script`, same signature as the `arm` op handler):

    {phase: pregrasp|grasp|lift|carry|lower|release|retract, arm: left|right, target_w: [x, y, z] (world, m),
     duration_s, settle_s, hold_on_end: target|measured|stand, closure (grasp, 0..1 of the deploy's fist; 0.8),
     standoff_m (0.10), above_m (0.05), lift_m (0.08), carry_b ([x, y, z] pelvis frame), max_ik_err_m (0.02),
     v_joint (0.7 rad/s), preempt, stream, execution_id, generation, control_epoch, servo_* (as `arm`)}

One call = one phase: the palm target is solved once at the start (damped-least-squares IK on main.urdf,
body/g1_kin.py, with the wrist pitch and yaw locked at their current values: SONIC tracks wrist pitch with a gain of
0.2-0.45 and wrist yaw with 0.5-0.9 at 0.5 Hz, docs/arm_tracking.md §3.2), then a min-jerk joint trajectory from the
pose being sent (continuity) to the solution is streamed through the channel at 50 Hz, the hand moves as the phase
says, and the arm settles for `settle_s` while the palm error is measured. The script session then ends
(`succeeded`, ended_by `script`) into `hold_on_end` (default `target`: the final arm + hand pose rides on every later
planner message, i.e. CarryLock once the hand is closed; `retract` defaults to `stand`: blend back to SONIC).

    phase     palm goal                                              hand
    pregrasp  target_w - standoff_m along the approach + above_m up  opens during the move
    grasp     target_w (the grasp point)                             open during the move, then closes to `closure`
    lift      the current palm + lift_m (world up)                   held
    carry     carry_b (pelvis frame; default in front of the hip)    held
    lower     target_w (the place point)                             held
    release   (no arm motion)                                        opens
    retract   the current palm - standoff_m along the approach +     held
              above_m up

The approach direction is the horizontal direction from the pelvis to target_w at the start (pregrasp) or from the
pelvis to the palm (retract). World <-> pelvis frame uses the ground-truth pelvis pose (P1 gt.pose, full
quaternion) at the start of the phase: the legs hold IDLE during a manipulation, so the target is fixed in the
pelvis frame for the phase.

Result (terminal `succeeded` data): `phase, arm, target_w, goal_w, goal_b, ik_err_m, move_s, palm_err_b_m` (FK of the
measured arm vs the IK goal, pelvis frame: tracking only) and `palm_err_w_m` (the measured palm in the world vs
`goal_w`: IK + tracking + pelvis sway), each `{median, p90, max, n}` over the last `measure_s` of the settle, and
`palm_final_w`. A goal the IK cannot reach within `max_ik_err_m` is rejected `ik_unreachable` before anything moves.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import g1_kin as K
from . import joint_map as jm
from .arm import ARM_IDX, ArmError, _hand, _minjerk, _num

PHASES = ("pregrasp", "grasp", "lift", "carry", "lower", "release", "retract")
DEFAULT_HOLD = {"pregrasp": "target", "grasp": "target", "lift": "target", "carry": "target", "lower": "target",
                "release": "target", "retract": "stand"}
NEEDS_TARGET = ("pregrasp", "grasp", "lower")
# a carry pose close to the body, palm ~0.2 m in front of the pelvis at hip height (pelvis frame)
CARRY_B = {"left": (0.22, 0.20, -0.02), "right": (0.22, -0.20, -0.02)}
LOCK = ("wrist_pitch", "wrist_yaw")


def quat_R(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def world_to_pelvis(p_w: Sequence[float], pose) -> np.ndarray:
    R = quat_R(pose.quat)
    return R.T @ (np.asarray(p_w, float) - np.array([pose.x, pose.y, pose.z]))


def pelvis_to_world(p_b: Sequence[float], pose) -> np.ndarray:
    return quat_R(pose.quat) @ np.asarray(p_b, float) + np.array([pose.x, pose.y, pose.z])


def _stats(v: list[float]) -> dict:
    if not v:
        return {"median": None, "p90": None, "max": None, "n": 0}
    a = np.asarray(v, float)
    return {"median": round(float(np.median(a)), 4), "p90": round(float(np.percentile(a, 90)), 4),
            "max": round(float(a.max()), 4), "n": int(a.size)}


def _vec3(args: dict, key: str) -> np.ndarray | None:
    v = args.get(key)
    if v is None:
        return None
    try:
        a = np.asarray([float(x) for x in v], float)
    except (TypeError, ValueError):
        raise ArmError("bad_args", {"arg": key, "error": "[x, y, z] in m"})
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ArmError("bad_args", {"arg": key, "error": "[x, y, z] in m"})
    return a


class ArmScriptPlan:
    op = "arm_script"
    waist_mode = "ref"
    servo_idx = ARM_IDX
    progress_hz = 2.0

    def __init__(self, phase: str, arm: str, q0: list[float], q1: list[float], hands0: dict, hand_keys: list,
                 move_s: float, settle_s: float, measure_s: float, hold_on_end: str, goal_b: np.ndarray | None,
                 goal_w: np.ndarray | None, target_w: np.ndarray | None, ik_err: float | None, pose0,
                 via: list | None = None):
        self.phase, self.arm = phase, arm
        self.q0, self.q1 = list(q0), list(q1)
        # joint-space min-jerk segments [(t_start, t_end, q_from, q_to)]: through the Cartesian via points (IK) first
        pts = [list(q0)] + [list(q) for q, _ in (via or [])] + [list(q1)]
        dist = [max((abs(b - a) for a, b in zip(u[3:], v[3:])), default=0.0) for u, v in zip(pts, pts[1:])]
        tot = sum(dist) or 1.0
        self.segs, t = [], 0.0
        for (u, v), d in zip(zip(pts, pts[1:]), dist):
            dur = move_s * (d / tot) if len(pts) > 2 else move_s
            self.segs.append((t, t + dur, u, v))
            t += dur
        self.via_b = [None if b is None else [round(float(x), 4) for x in b] for _, b in (via or [])]
        self.hands0 = {s: (None if hands0.get(s) is None else list(hands0[s])) for s in ("left", "right")}
        self.hand_keys = hand_keys                  # [(t0, t1, side, from7, to7)]
        self.move_s, self.settle_s, self.measure_s = move_s, settle_s, measure_s
        t_hand = max([k[1] for k in hand_keys], default=0.0)
        self.t_settle = max(move_s, t_hand)
        self.duration_s = self.t_settle + settle_s
        self.hold_on_end = hold_on_end
        self.goal_b, self.goal_w, self.target_w, self.ik_err = goal_b, goal_w, target_w, ik_err
        self.pose0 = pose0
        self.err_b: list[float] = []
        self.err_w: list[float] = []
        self.palm_w_last: np.ndarray | None = None
        self.palm_b_last: np.ndarray | None = None

    def sample(self, t: float, ref: list[float]) -> tuple[list[float], dict]:
        q = list(self.q1)
        for t0, t1, u, v in self.segs:
            if t < t1:
                a = 1.0 if t1 <= t0 else _minjerk((t - t0) / (t1 - t0))
                q = [x + (y - x) * a for x, y in zip(u, v)]
                break
        hands = dict(self.hands0)
        for t0, t1, side, h0, h1 in self.hand_keys:
            if t < t0:
                continue
            b = 1.0 if t1 <= t0 else _minjerk((t - t0) / (t1 - t0))
            hands[side] = [x + (y - x) * b for x, y in zip(h0, h1)]
        return q, hands

    def on_tick(self, t: float, now: float, ch) -> list:
        if self.goal_b is None or t < self.duration_s - self.measure_s:
            return []
        qm = ch.measured_mj17()
        if qm is None or ch.deploy.age_s() > 0.1:
            return []
        palm_b = K.points(K.named_from_mj17(qm))[f"{self.arm}_palm"]
        self.palm_b_last = palm_b
        self.err_b.append(float(np.linalg.norm(palm_b - self.goal_b)))
        pose = ch.gt_pose()
        if pose is not None and self.goal_w is not None:
            pw = pelvis_to_world(palm_b, pose)
            self.palm_w_last = pw
            self.err_w.append(float(np.linalg.norm(pw - self.goal_w)))
        return []

    def progress(self, t: float) -> dict:
        stage = "move" if t < self.move_s else ("hand" if t < self.t_settle else "settle")
        return {"phase": self.phase, "arm": self.arm, "stage": stage, "t": round(t, 2),
                "duration_s": round(self.duration_s, 2)}

    def brief(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]   # noqa: E731
        return {"phase": self.phase, "arm": self.arm, "target_w": r(self.target_w), "goal_w": r(self.goal_w),
                "goal_b": r(self.goal_b), "via_b": self.via_b,
                "ik_err_m": None if self.ik_err is None else round(self.ik_err, 4),
                "move_s": round(self.move_s, 2), "duration_s": round(self.duration_s, 2),
                "hold_on_end": self.hold_on_end}

    def result(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]   # noqa: E731
        h_end = self.sample(self.duration_s, self.q1)[1].get(self.arm)
        closure = None if h_end is None else round(jm.hand_closure_of(self.arm, h_end), 3)
        return {**self.brief(), "palm_err_b_m": _stats(self.err_b), "palm_err_w_m": _stats(self.err_w),
                "palm_final_w": r(self.palm_w_last), "palm_final_b": r(self.palm_b_last),
                "hand_closure_cmd": closure,
                "carry": self.hold_on_end == "target" and closure is not None and closure >= 0.3}


def _palm_b(q17: list[float], arm: str) -> np.ndarray:
    return K.points(K.named_from_mj17(q17))[f"{arm}_palm"]


def build(ch, args: dict, now: float) -> ArmScriptPlan:
    phase = args.get("phase")
    if phase not in PHASES:
        raise ArmError("bad_args", {"arg": "phase", "error": " | ".join(PHASES)})
    arm = args.get("arm")
    if arm not in ("left", "right"):
        raise ArmError("bad_args", {"arg": "arm", "error": "left | right"})
    hold = args.get("hold_on_end") or DEFAULT_HOLD[phase]
    if hold not in ("target", "measured", "stand"):
        raise ArmError("bad_args", {"arg": "hold_on_end", "error": "target | measured | stand"})
    target_w = _vec3(args, "target_w")
    if phase in NEEDS_TARGET and target_w is None:
        raise ArmError("bad_args", {"arg": "target_w", "error": f"{phase} needs target_w [x, y, z] (world)"})
    standoff = _num(args, "standoff_m", 0.0, 0.4, 0.10)
    above = _num(args, "above_m", -0.2, 0.4, 0.05)
    lift = _num(args, "lift_m", -0.2, 0.4, 0.08)
    max_ik = _num(args, "max_ik_err_m", 0.001, 0.5, 0.02)
    v_joint = _num(args, "v_joint", 0.1, 3.0, 0.7)
    closure = _num(args, "closure", 0.0, 1.0, 0.8)
    q0 = ch.continuity_pose()
    hands0 = {s: (None if ch.hands_sent.get(s) is None else list(ch.hands_sent[s])) for s in ("left", "right")}
    # IK on SONIC's (measured) waist: the channel sends SONIC's reference waist (waist "ref")
    qm = ch.measured_mj17()
    seed17 = list(q0)
    if qm is not None:
        seed17[0:3] = qm[0:3]
    seed = K.named_from_mj17(seed17)
    pose = ch.gt_pose(wait_s=0.3)
    if (target_w is not None or phase in ("lift", "retract")) and pose is None:
        raise ArmError("no_pose", {"hint": "no ground-truth pelvis pose (P1 gt.pose)"})
    palm0_b = _palm_b(seed17, arm)
    goal_b = goal_w = None
    if phase in ("pregrasp", "grasp", "lower"):
        goal_w = np.array(target_w, float)
        if phase == "pregrasp":
            d = goal_w[:2] - np.array([pose.x, pose.y])
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-6 else np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
            goal_w = goal_w - np.array([d[0], d[1], 0.0]) * standoff + np.array([0.0, 0.0, above])
        goal_b = world_to_pelvis(goal_w, pose)
    elif phase == "lift":
        goal_w = pelvis_to_world(palm0_b, pose) + np.array([0.0, 0.0, lift])
        goal_b = world_to_pelvis(goal_w, pose)
    elif phase == "retract":
        pw = pelvis_to_world(palm0_b, pose)
        d = pw[:2] - np.array([pose.x, pose.y])
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        goal_w = pw - np.array([d[0], d[1], 0.0]) * standoff + np.array([0.0, 0.0, above])
        goal_b = world_to_pelvis(goal_w, pose)
    elif phase == "carry":
        cb = _vec3(args, "carry_b")
        goal_b = np.array(cb if cb is not None else CARRY_B[arm], float)
        goal_w = None if pose is None else pelvis_to_world(goal_b, pose)
    ik_err = None
    q1 = list(q0)
    via = []
    lock = tuple(f"{arm}_{j}_joint" for j in LOCK)

    def solve(pt_b, seed_named):
        qn, err = K.ik_palm(arm, pt_b, seed_named, q_rest=seed, lock=lock)
        q = list(q0)
        for n_ in K.ARM_CHAIN[arm]:
            q[jm.UPPER_BODY_MUJOCO_JOINTS.index(n_)] = float(qn[n_])
        return jm.clamp_mj17(q, margin=0.02)[0], qn, err

    if goal_b is not None and args.get("via", True):
        # Cartesian via points, so the hand does not sweep through the furniture it works on: pregrasp rises (palm
        # back) before it reaches forward, carry pulls back before it goes down, lower comes from above
        cands = []
        if phase == "pregrasp" and palm0_b[2] < goal_b[2] - 0.03:
            # at the goal height, as far back as the arm reaches there (the palm cannot come close to the shoulder)
            cands = [np.array([goal_b[0] - dx, goal_b[1], goal_b[2]]) for dx in (0.12, 0.10, 0.08, 0.06)]
        elif phase == "carry" and palm0_b[0] > goal_b[0] + 0.03:
            cands = [np.array([goal_b[0] + dx, goal_b[1], max(palm0_b[2], goal_b[2])]) for dx in (0.0, 0.03, 0.06)]
        elif phase == "lower" and palm0_b[2] < goal_b[2] + 0.04:
            cands = [np.array([goal_b[0] - 0.05, goal_b[1], goal_b[2] + 0.06])]
        for vb in cands:
            qv, qvn, ev = solve(vb, seed)
            if ev <= 0.015:
                via.append((qv, vb))
                seed = {**seed, **qvn}
                break
    if goal_b is not None:
        q_named, ik_err = K.ik_palm(arm, goal_b, seed, q_rest=seed, lock=lock)
        if ik_err > max_ik:
            raise ArmError("ik_unreachable", {"phase": phase, "arm": arm, "goal_b": [round(float(x), 4) for x in goal_b],
                                              "ik_err_m": round(ik_err, 4), "max_ik_err_m": max_ik,
                                              "hint": "reposition (the G1 palm reaches ~0.26-0.40 m in front of the "
                                                      "pelvis, docs/arm_tracking.md §1)"})
        for n_ in K.ARM_CHAIN[arm]:
            q1[jm.UPPER_BODY_MUJOCO_JOINTS.index(n_)] = float(q_named[n_])
        q1, _ = jm.clamp_mj17(q1, margin=0.02)
    dq = max(abs(b - a) for a, b in zip(q0[3:], q1[3:]))
    move_s = 0.0 if phase == "release" else _num(args, "duration_s", 0.3, 15.0, min(max(dq / v_joint, 0.8), 4.0))
    if via and args.get("duration_s") is None:
        dq_all = sum(max(abs(b - a) for a, b in zip(u[3:], v[3:]))
                     for u, v in zip([q0] + [q for q, _ in via], [q for q, _ in via] + [q1]))
        move_s = min(max(dq_all / v_joint, 0.8), 5.0)
    settle_s = _num(args, "settle_s", 0.0, 10.0, 2.0 if phase in ("pregrasp", "grasp", "lower") else 1.0)
    measure_s = min(1.0, max(0.0, settle_s - 0.3))
    # hands
    h_open = {"left": list(jm.DEX3_OPEN), "right": list(jm.DEX3_OPEN)}
    h_cur = hands0[arm] if hands0[arm] is not None else list(jm.DEX3_CLOSED[arm])
    keys = []
    if phase == "pregrasp":
        keys.append((0.0, max(move_s, 0.5), arm, h_cur, h_open[arm]))
    elif phase == "grasp":
        close_s = _num(args, "close_s", 0.1, 5.0, 0.6)
        h_closed = _hand(closure, arm, "closure")
        if max(abs(x) for x in h_cur) > 1e-3:
            keys.append((0.0, max(move_s, 0.3), arm, h_cur, h_open[arm]))
            h_cur = h_open[arm]
        keys.append((move_s, move_s + close_s, arm, h_cur, h_closed))
    elif phase == "release":
        open_s = _num(args, "open_s", 0.1, 5.0, 0.6)
        keys.append((0.0, open_s, arm, h_cur, h_open[arm]))
    return ArmScriptPlan(phase, arm, q0, q1, hands0, keys, move_s, settle_s, measure_s, hold, goal_b, goal_w,
                         target_w, ik_err, pose, via=via)
