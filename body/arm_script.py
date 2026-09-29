"""B.7 ArmScript: scripted reach / grasp / lift / carry / lower / release / retract through the arm channel.

Op `arm_script` (handler `ArmChannel.handle_arm_script`, same signature as the `arm` op handler):

    {phase: pregrasp|grasp|lift|carry|lower|release|retract, arm: left|right, target_w: [x, y, z] (world, m),
     duration_s, settle_s, hold_on_end: target|measured|stand, closure (grasp, 0..1 of the deploy's fist; 0.8),
     approach (front|above: the pregrasp in front of the target, or straight above it), preshape (0..0.6: the "open"
     hand of pregrasp/grasp; 0 = flat), clear_z (world z of the surface under the hand: the palm goal is raised until
     the hand's collision boxes, g1_kin.hand_points with the Dex3 pose the phase ends with, clear it by clearance_m
     (0.01)), avoid_boxes (world AABBs the hand and forearm must stay out of on the way: via points are searched so
     the joint-space path clears them; their top is the default clear_z), standoff_m (0.10), above_m
     (0.05; 0.08 with approach above), lift_m (0.08),
     carry_b ([x, y, z] pelvis frame), max_ik_err_m (0.02), v_joint (0.7 rad/s), preempt, stream, execution_id,
     generation, control_epoch, servo_* (as `arm`)}

One call = one phase: the palm target is solved once at the start (damped-least-squares IK on main.urdf,
body/g1_kin.py, with the wrist pitch and yaw locked at their current values: SONIC tracks wrist pitch with a gain of
0.2-0.45 and wrist yaw with 0.5-0.9 at 0.5 Hz, docs/arm_tracking.md §3.2), then a min-jerk joint trajectory from the
pose being sent (continuity) to the solution is streamed through the channel at 50 Hz, the hand moves as the phase
says, and the arm settles for `settle_s` while the palm error is measured.

Threads (M2b wave 2, B-D1): prepare() runs on the control thread under the arm lock (args, frames: cheap); solve()
(via points + the goal IK, 2-230 ms) runs in the channel's IK worker, a separate PROCESS on the service
(body/ik_worker.py), so neither the 50 Hz tick nor the halt lane ever waits for it; finish() builds the plan when the
session starts, from the pose being sent at that moment. The settle's 10 Hz re-solve goes to the same worker. The script session then ends
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
import time
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
APPROACHES = ("front", "above")      # pregrasp: standoff in front of the target + above_m, or straight above it


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
                 via: list | None = None, track_world: bool = False, seed_waist: list | None = None):
        self.phase, self.arm = phase, arm
        self.approach, self.preshape = "front", 0.0
        self.clear: dict | None = None                  # clear_z: the hand-clearance goal raise (finish())
        self.path: dict | None = None                   # avoid_boxes: the collision-checked path (solve())
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
        # world tracking (the goal is fixed in the world; SONIC shifts the pelvis 2-5 cm when the arm reaches)
        self.track = bool(track_world and goal_w is not None)
        self.q1_goal, self.q1_cur = list(q1), list(q1)
        self.goal_b_cur = None if goal_b is None else np.array(goal_b, float)
        self.t_track = -1.0
        self.track_updates = 0
        self.track_submits = 0
        self.track_errors = 0
        self._track_fut = None
        self._track_gb = None
        self.ik_ms: float | None = None                 # the start solve's time in the IK worker
        self.ik_seed: str | None = None                 # continuity | default (the retry from SONIC's default arm)
        self.seed_waist = list(seed_waist) if seed_waist is not None else list(q1[0:3])
        self.pelvis_shift_m: float | None = None
        self.err_b: list[float] = []
        self.err_w: list[float] = []
        self.palm_w_last: np.ndarray | None = None
        self.palm_b_last: np.ndarray | None = None

    def sample(self, t: float, ref: list[float]) -> tuple[list[float], dict]:
        q = list(self.q1_cur)
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
        if self.track and t >= self.move_s:
            self._track(t, ch)
        if self.goal_b is None or t < self.duration_s - self.measure_s:
            return []
        qm = ch.measured_mj17()
        if qm is None or ch.deploy.age_s() > 0.1:
            return []
        palm_b = K.points(K.named_from_mj17(qm))[f"{self.arm}_palm"]
        self.palm_b_last = palm_b
        self.err_b.append(float(np.linalg.norm(palm_b - self.goal_b_cur)))
        pose = ch.gt_pose()
        if pose is not None and self.goal_w is not None:
            pw = pelvis_to_world(palm_b, pose)
            self.palm_w_last = pw
            self.err_w.append(float(np.linalg.norm(pw - self.goal_w)))
        return []

    def _track(self, t: float, ch) -> None:
        """Settle phase: re-solve the world goal in the current pelvis frame at 10 Hz, and move the joint target
        there at <= 0.5 rad/s (the IK seed is the last solution, so a re-solve takes a few iterations). The re-solve
        runs on the channel's IK worker (body/ik_worker.py: a worker process on the service, so the 50 Hz tick and the
        halt lane never wait for it); its answer is picked up by a later tick (inline worker: this tick)."""
        fut = self._track_fut
        if fut is not None and fut.done():
            self._track_fut = None
            try:
                qa, err = fut.result()
            except Exception as e:  # noqa: BLE001 - a lost re-solve only skips one update
                self.track_errors += 1
                note = getattr(ch.ik, "note_failure", None)
                if callable(note):
                    note(e)
                qa, err = None, 1.0
            if qa is not None and err < 0.02:
                q = list(self.q1_goal)
                for n_, v in qa.items():
                    q[jm.UPPER_BODY_MUJOCO_JOINTS.index(n_)] = float(v)
                self.q1_goal = jm.clamp_mj17(q, margin=0.02)[0]
                self.goal_b_cur = self._track_gb
                self.track_updates += 1
        if self._track_fut is None and t - self.t_track >= 0.1:
            self.t_track = t
            pose = ch.gt_pose()
            if pose is not None:
                if self.pose0 is not None:
                    self.pelvis_shift_m = round(float(np.hypot(pose.x - self.pose0.x, pose.y - self.pose0.y)), 4)
                gb = world_to_pelvis(self.goal_w, pose)
                if float(np.linalg.norm(gb - self.goal_b_cur)) > 0.003:
                    seed17 = list(self.q1_goal)
                    qm = ch.measured_mj17()
                    seed17[0:3] = qm[0:3] if qm is not None else self.seed_waist
                    self._track_gb = gb
                    self._track_fut = ch.ik.submit(solve_track, self.arm, [float(x) for x in gb], seed17)
                    self.track_submits += 1
                    if self._track_fut.done():          # inline worker: apply in this tick, as before
                        self._track(t, ch)
                        return
        step = 0.5 * 0.02
        self.q1_cur = [c + min(max(g - c, -step), step) for c, g in zip(self.q1_cur, self.q1_goal)]

    def progress(self, t: float) -> dict:
        stage = "move" if t < self.move_s else ("hand" if t < self.t_settle else "settle")
        return {"phase": self.phase, "arm": self.arm, "stage": stage, "t": round(t, 2),
                "duration_s": round(self.duration_s, 2)}

    def brief(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]   # noqa: E731
        return {"phase": self.phase, "arm": self.arm, "target_w": r(self.target_w), "goal_w": r(self.goal_w),
                "goal_b": r(self.goal_b), "via_b": self.via_b,
                "ik_err_m": None if self.ik_err is None else round(self.ik_err, 4),
                "ik_ms": self.ik_ms, "ik_seed": self.ik_seed, "approach": self.approach, "preshape": self.preshape,
                "clear": self.clear, "path": self.path,
                "move_s": round(self.move_s, 2), "duration_s": round(self.duration_s, 2),
                "hold_on_end": self.hold_on_end}

    def result(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]   # noqa: E731
        h_end = self.sample(self.duration_s, self.q1)[1].get(self.arm)
        closure = None if h_end is None else round(jm.hand_closure_of(self.arm, h_end), 3)
        return {**self.brief(), "track_world": self.track, "track_updates": self.track_updates,
                "track_submits": self.track_submits, "track_errors": self.track_errors,
                "pelvis_shift_m": self.pelvis_shift_m, "palm_err_b_m": _stats(self.err_b),
                "palm_err_w_m": _stats(self.err_w),
                "palm_final_w": r(self.palm_w_last), "palm_final_b": r(self.palm_b_last),
                "hand_closure_cmd": closure,
                "carry": self.hold_on_end == "target" and closure is not None and closure >= 0.3}


def _palm_b(q17: list[float], arm: str) -> np.ndarray:
    return K.points(K.named_from_mj17(q17))[f"{arm}_palm"]


def prepare(ch, args: dict, now: float) -> dict:
    """Control thread, under the arm lock, cheap (no IK): parse and check the args, read the pose being sent, the
    measured waist and the GT pelvis pose, and put the phase's palm goal in the pelvis frame. Raises ArmError
    (bad_args, no_pose) before anything changes. The result feeds solve() (in the IK worker) and then finish()."""
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
    approach = args.get("approach") or "front"
    if approach not in APPROACHES:
        raise ArmError("bad_args", {"arg": "approach", "error": " | ".join(APPROACHES)})
    standoff = _num(args, "standoff_m", 0.0, 0.4, 0.10)
    above = _num(args, "above_m", -0.2, 0.4, 0.05 if approach == "front" else 0.08)
    lift = _num(args, "lift_m", -0.2, 0.4, 0.08)
    boxes = _boxes(args)
    clear_z = None if args.get("clear_z") is None else _num(args, "clear_z", -5.0, 5.0, 0.0)
    if clear_z is None and boxes and phase in ("pregrasp", "grasp", "lower"):
        clear_z = max(b[1][2] for b in boxes)       # the top of what the hand works over
    p = {"phase": phase, "arm": arm, "hold": hold, "target_w": target_w, "approach": approach, "clear_z": clear_z,
         "clearance": _num(args, "clearance_m", 0.0, 0.1, 0.01),
         "max_ik": _num(args, "max_ik_err_m", 0.001, 0.5, 0.02), "v_joint": _num(args, "v_joint", 0.1, 3.0, 0.7),
         "closure": _num(args, "closure", 0.0, 1.0, 0.8), "preshape": _num(args, "preshape", 0.0, 0.6, 0.0),
         "duration_s": None if args.get("duration_s") is None else _num(args, "duration_s", 0.3, 15.0, 1.0),
         "settle_s": _num(args, "settle_s", 0.0, 10.0, 2.5 if phase in ("pregrasp", "grasp", "lower") else 1.0),
         "close_s": _num(args, "close_s", 0.1, 5.0, 0.6), "open_s": _num(args, "open_s", 0.1, 5.0, 0.6),
         "via": bool(args.get("via", True)),
         "track": bool(args.get("track_world", phase in ("pregrasp", "grasp", "lower")))}
    if args.get("closure") is not None:
        _hand(p["closure"], arm, "closure")
    q0 = ch.continuity_pose()
    # IK on SONIC's (measured) waist: the channel sends SONIC's reference waist (waist "ref")
    qm = ch.measured_mj17()
    seed17 = list(q0)
    if qm is not None:
        seed17[0:3] = qm[0:3]
    pose = ch.gt_pose(wait_s=0.3)
    if (target_w is not None or phase in ("lift", "retract")) and pose is None:
        raise ArmError("no_pose", {"hint": "no ground-truth pelvis pose (P1 gt.pose)"})
    palm0_b = _palm_b(seed17, arm)
    goal_b = goal_w = None
    if phase in ("pregrasp", "grasp", "lower"):
        goal_w = np.array(target_w, float)
        if phase == "pregrasp":
            if approach == "above":             # straight above the target: the grasp then only descends
                goal_w = goal_w + np.array([0.0, 0.0, above])
            else:
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
    clear_b = None
    if clear_z is not None and goal_w is not None and phase in ("pregrasp", "grasp", "lower") and pose is not None:
        # the surface under the hand, as a pelvis-frame height at the goal (the pelvis is upright within a few deg)
        clear_b = float(world_to_pelvis([goal_w[0], goal_w[1], clear_z], pose)[2])
    p.update({"q0": q0, "seed17": seed17, "pose": pose, "palm0_b": palm0_b, "goal_b": goal_b, "goal_w": goal_w,
              "job": {"phase": phase, "arm": arm, "seed17": list(seed17), "approach": approach, "via": p["via"],
                      "goal_b": None if goal_b is None else [float(x) for x in goal_b],
                      "palm0_b": [float(x) for x in palm0_b], "clear_b": clear_b, "clearance": p["clearance"],
                      "hand7": _end_hand(p, arm, ch), "boxes": boxes,
                      "hand0": list(ch.hands_sent[arm]) if ch.hands_sent.get(arm) is not None else
                      list(jm.DEX3_CLOSED[arm]),
                      "pose": None if pose is None else {"xyz": [pose.x, pose.y, pose.z],
                                                         "quat": [float(v) for v in pose.quat]}}})
    return p


def _boxes(args: dict) -> list:
    """avoid_boxes: world AABBs [[x0, y0, z0], [x1, y1, z1]] the hand and forearm must stay out of (the furniture the
    hand works over, the object); avoid_box: one."""
    raw = args.get("avoid_boxes")
    if raw is None and args.get("avoid_box") is not None:
        raw = [args["avoid_box"]]
    if raw is None:
        return []
    try:
        out = []
        for b in raw:
            lo, hi = [float(x) for x in b[0]], [float(x) for x in b[1]]
            if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in lo + hi) or \
                    any(h < l for l, h in zip(lo, hi)):
                raise ValueError
            out.append([lo, hi])
        return out[:8]
    except (TypeError, ValueError, IndexError):
        raise ArmError("bad_args", {"arg": "avoid_boxes", "error": "a list of world AABBs [[x0, y0, z0], [x1, y1, z1]]"})


def _end_hand(p: dict, arm: str, ch) -> list[float]:
    """The Dex3 pose the phase ends with (for the clearance check): grasp closes to `closure`, pregrasp / release
    open to `preshape` / flat, the rest keep the hand being sent."""
    if p["phase"] == "grasp":
        return list(_hand(p["closure"], arm, "closure"))
    if p["phase"] == "pregrasp":
        return list(jm.hand_closure(arm, p["preshape"]))
    if p["phase"] == "release":
        return list(jm.DEX3_OPEN)
    h = ch.hands_sent.get(arm)
    return list(h) if h is not None else list(jm.DEX3_CLOSED[arm])


def solve(job: dict) -> dict:
    """The phase's IK, in the IK worker (a pure function of plain data, body/ik_worker.py): Cartesian via points and
    the goal solve. Returns the arm joints by name (the caller writes them into the pose being sent when the session
    starts), the via points, the goal's IK error and the solve time."""
    t0 = time.perf_counter()
    arm, phase = job["arm"], job["phase"]
    seed = K.named_from_mj17(job["seed17"])
    rest = dict(seed)
    lock = tuple(f"{arm}_{j}_joint" for j in LOCK)
    out = {"arm_q": None, "via": [], "ik_err": None}
    if job["goal_b"] is None:
        out["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        return out
    goal_b = np.asarray(job["goal_b"], float)
    palm0_b = np.asarray(job["palm0_b"], float)
    if job["via"] and not job.get("boxes"):
        # Cartesian via points, so the hand does not sweep through the furniture it works on: pregrasp rises (palm
        # back) before it reaches forward, carry pulls back before it goes down, lower comes from above; a grasp
        # after an `above` pregrasp needs none (it only descends). With avoid_boxes the path is searched instead.
        cands = []
        if phase == "pregrasp" and palm0_b[2] < goal_b[2] - 0.03:
            # at the goal height, as far back as the arm reaches there (the palm cannot come close to the shoulder)
            cands = [np.array([goal_b[0] - dx, goal_b[1], goal_b[2]]) for dx in (0.12, 0.10, 0.08, 0.06)]
        elif phase == "carry" and palm0_b[0] > goal_b[0] + 0.03:
            cands = [np.array([goal_b[0] + dx, goal_b[1], max(palm0_b[2], goal_b[2])]) for dx in (0.0, 0.03, 0.06)]
        elif phase == "lower" and palm0_b[2] < goal_b[2] + 0.04:
            cands = [np.array([goal_b[0] - 0.05, goal_b[1], goal_b[2] + 0.06])]
        for vb in cands:
            qvn, ev = K.ik_palm(arm, vb, seed, q_rest=rest, lock=lock)
            if ev <= 0.015:
                out["via"].append(({n_: float(qvn[n_]) for n_ in K.ARM_CHAIN[arm]}, [float(x) for x in vb]))
                seed = {**seed, **qvn}
                break
    q_named, err = K.ik_palm(arm, goal_b, seed, q_rest=rest, lock=lock)
    out["ik_seed"] = "continuity"
    if err > 0.01:
        # damped least squares can stall in a local minimum from a strongly bent arm (e.g. mid-blend, or after a
        # carry tuck): try once more from SONIC's default arm (the waist and the locked wrist joints as they are)
        seed2 = dict(seed)
        for n_ in K.ARM_CHAIN[arm]:
            if n_ not in lock:
                seed2[n_] = float(jm.DEFAULT_ANGLES[jm.MJ[n_]])
        q2, err2 = K.ik_palm(arm, goal_b, seed2, q_rest=seed2, lock=lock)
        if err2 < err:
            q_named, err, out["ik_seed"] = q2, err2, "default"
    # hand clearance (clear_z): raise the palm goal until the hand's collision boxes (g1_kin.hand_points, at the
    # solved orientation) is `clearance` above the surface under it; the hand is ~4.4 cm thick below the palm origin
    out["goal_b"] = [float(x) for x in goal_b]
    out["goal_raise_m"] = 0.0
    if job.get("clear_b") is not None:
        hand7 = job.get("hand7")
        for _ in range(4):
            low = float(K.hand_points(arm, {**seed, **q_named}, hand7)[:, 2].min())
            out["hand_low_b"] = round(low, 4)
            need = job["clear_b"] + job["clearance"] - low
            if need <= 0.002:
                break
            goal_b = goal_b + np.array([0.0, 0.0, need])
            out["goal_raise_m"] = round(out["goal_raise_m"] + need, 4)
            q_named, err = K.ik_palm(arm, goal_b, {**seed, **q_named}, q_rest=rest, lock=lock)
        out["goal_b"] = [float(x) for x in goal_b]
        out["hand_low_b"] = round(float(K.hand_points(arm, {**seed, **q_named}, hand7)[:, 2].min()), 4)
    if job.get("boxes") and job.get("pose") is not None:
        _search_path(job, out, arm, phase, seed, rest, lock, q_named, np.asarray(out["goal_b"], float),
                     job["palm0_b"])
    out["arm_q"] = {n_: float(q_named[n_]) for n_ in K.ARM_CHAIN[arm]}
    out["ik_err"] = float(err)
    out["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    return out


def _R_t(pose: dict):
    return quat_R(pose["quat"]), np.asarray(pose["xyz"], float)


def _hits(job: dict, arm: str, q_named: dict, R, t, margin: float) -> int:
    """How many hand / forearm points are inside the avoid boxes (expanded by margin; the forearm by 3.5 cm more),
    for the phase's start and end hand."""
    n = 0
    fa = K.forearm_points(arm, q_named) @ R.T + t
    for h in (job["hand0"], job["hand7"]):
        pts = K.hand_points(arm, q_named, h) @ R.T + t
        for lo, hi in job["boxes"]:
            lo_, hi_ = np.asarray(lo) - margin, np.asarray(hi) + margin
            n += int(np.all((pts > lo_) & (pts < hi_), axis=1).sum())
            n += int(np.all((fa > lo_ - 0.035) & (fa < hi_ + 0.035), axis=1).sum())
    return n


def _path_hits(job, arm, qs: list, R, t, margin: float, n: int = 12) -> int:
    """Collision count along joint-space segments qs[0] -> qs[1] -> ... (the min-jerk path is the straight line in
    joint space), n samples per segment."""
    tot = 0
    for qa, qb in zip(qs, qs[1:]):
        for k in range(1, n + 1):
            f = k / n
            tot += _hits(job, arm, {j: qa[j] + (qb[j] - qa[j]) * f for j in qa}, R, t, margin)
    return tot


def _search_path(job, out, arm, phase, seed, rest, lock, q_goal, goal_b, palm0_b) -> None:
    """avoid_boxes: find via points so that the hand and forearm stay out of the boxes on the way to the goal (a
    pregrasp from the arm's rest pose sweeps the open hand forward under a table edge otherwise: live, the fingers
    caught the dresser's front edge and SONIC stepped back 0.22 m). Candidates, first collision-free wins: direct;
    one via at the goal height pulled back 0.12-0.28 m; straight up at the current palm x, then forward. Reported in
    out["path"]; nothing is rejected (the boxes are the caller's model of the world)."""
    R, t = _R_t(job["pose"])
    margin = job["clearance"]
    chain = K.ARM_CHAIN[arm]
    q0 = {j: float(seed[j]) for j in chain}
    qg = {j: float(q_goal[j]) for j in chain}
    base = {**seed}
    cands = [[]]
    if phase in ("pregrasp", "lower", "carry"):
        cands += [[[goal_b[0] - dx, goal_b[1], goal_b[2]]] for dx in (0.12, 0.16, 0.20, 0.24, 0.28)]
        cands += [[[palm0_b[0] - 0.02, palm0_b[1], goal_b[2]]],
                  [[palm0_b[0] - 0.04, palm0_b[1], goal_b[2]], [goal_b[0] - 0.12, goal_b[1], goal_b[2]]]]
    tried = []
    best = None
    for vias in cands:
        qs, sd, ok = [q0], dict(seed), True
        for vb in vias:
            qv, ev = K.ik_palm(arm, vb, sd, q_rest=rest, lock=lock)
            if ev > 0.015:
                ok = False
                break
            qs.append({j: float(qv[j]) for j in chain})
            sd = {**sd, **qv}
        if not ok:
            tried.append({"vias": [[round(x, 3) for x in v] for v in vias], "ik": "unreachable"})
            continue
        qs.append(qg)
        hits = _path_hits(job, arm, [{**base, **q} for q in qs], R, t, margin)
        tried.append({"vias": [[round(x, 3) for x in v] for v in vias], "hits": hits})
        if best is None or hits < best[0]:
            best = (hits, vias, qs)
        if hits == 0:
            break
    if best is None:
        out["path"] = {"ok": False, "tried": tried}
        return
    hits, vias, qs = best
    out["via"] = [({j: qs[i + 1][j] for j in chain}, [float(x) for x in vb]) for i, vb in enumerate(vias)]
    out["path"] = {"ok": hits == 0, "hits": hits, "vias": [[round(x, 3) for x in v] for v in vias],
                   "tried": len(tried), "goal_hits": _hits(job, arm, {**base, **qg}, R, t, margin)}


def solve_track(arm: str, goal_b: list, seed17: list) -> tuple[dict, float]:
    """The settle's world-goal re-solve (ArmScriptPlan._track), in the IK worker: 40 iterations from the last solution."""
    seed = K.named_from_mj17(seed17)
    lock = tuple(f"{arm}_{j}_joint" for j in LOCK)
    qn, err = K.ik_palm(arm, goal_b, seed, q_rest=seed, lock=lock, iters=40)
    return {n_: float(qn[n_]) for n_ in K.ARM_CHAIN[arm]}, float(err)


def _with_arm(q17: list[float], arm_q: dict) -> list[float]:
    q = list(q17)
    for n_, v in arm_q.items():
        q[jm.UPPER_BODY_MUJOCO_JOINTS.index(n_)] = float(v)
    return jm.clamp_mj17(q, margin=0.02)[0]


def finish(ch, p: dict, sol: dict) -> ArmScriptPlan:
    """Control thread, under the arm lock, when the session starts: `ik_unreachable` if the goal is out of reach,
    else the plan from the pose being sent NOW (it may have moved while the worker solved: a blend, a servo)."""
    phase, arm = p["phase"], p["arm"]
    if p["goal_b"] is not None and (sol.get("ik_err") is None or sol["ik_err"] > p["max_ik"]):
        raise ArmError("ik_unreachable", {"phase": phase, "arm": arm,
                                          "goal_b": [round(float(x), 4) for x in p["goal_b"]],
                                          "ik_err_m": None if sol.get("ik_err") is None else round(sol["ik_err"], 4),
                                          "max_ik_err_m": p["max_ik"], "ik_ms": sol.get("ms"),
                                          "hint": "reposition (the G1 palm reaches ~0.26-0.40 m in front of the "
                                                  "pelvis, docs/arm_tracking.md §1)"})
    goal_b, goal_w = p["goal_b"], p["goal_w"]
    if sol.get("goal_raise_m"):
        goal_b = np.asarray(sol["goal_b"], float)
        goal_w = None if goal_w is None else np.asarray(goal_w, float) + np.array([0.0, 0.0, sol["goal_raise_m"]])
    q0 = ch.continuity_pose()
    hands0 = {s: (None if ch.hands_sent.get(s) is None else list(ch.hands_sent[s])) for s in ("left", "right")}
    q1 = _with_arm(q0, sol["arm_q"]) if sol.get("arm_q") else list(q0)
    via = [(_with_arm(q0, qa), np.asarray(vb, float)) for qa, vb in sol.get("via") or []]
    v_joint = p["v_joint"]
    dq = max(abs(b - a) for a, b in zip(q0[3:], q1[3:]))
    move_s = 0.0 if phase == "release" else (p["duration_s"] if p["duration_s"] is not None
                                             else min(max(dq / v_joint, 0.8), 4.0))
    if via and p["duration_s"] is None:
        dq_all = sum(max(abs(b - a) for a, b in zip(u[3:], v[3:]))
                     for u, v in zip([q0] + [q for q, _ in via], [q for q, _ in via] + [q1]))
        move_s = min(max(dq_all / v_joint, 0.8), 5.0)
    settle_s = p["settle_s"]
    measure_s = min(1.0, max(0.0, settle_s - 0.3))
    # hands: `preshape` is the "open" hand of pregrasp / grasp (0 = flat; a slight curl keeps the fingertips up)
    h_open = {"left": jm.hand_closure("left", p["preshape"]), "right": jm.hand_closure("right", p["preshape"])}
    h_cur = hands0[arm] if hands0[arm] is not None else list(jm.DEX3_CLOSED[arm])
    keys = []
    if phase == "pregrasp":
        keys.append((0.0, max(move_s, 0.5), arm, h_cur, h_open[arm]))
    elif phase == "grasp":
        h_closed = _hand(p["closure"], arm, "closure")
        if max(abs(x - y) for x, y in zip(h_cur, h_open[arm])) > 1e-3:
            keys.append((0.0, max(move_s, 0.3), arm, h_cur, h_open[arm]))
            h_cur = h_open[arm]
        keys.append((move_s, move_s + p["close_s"], arm, h_cur, h_closed))
    elif phase == "release":
        keys.append((0.0, p["open_s"], arm, h_cur, list(jm.DEX3_OPEN)))
    plan = ArmScriptPlan(phase, arm, q0, q1, hands0, keys, move_s, settle_s, measure_s, p["hold"], goal_b,
                         goal_w, p["target_w"], sol.get("ik_err"), p["pose"], via=via, track_world=p["track"],
                         seed_waist=p["seed17"][0:3])
    plan.ik_ms, plan.ik_seed = sol.get("ms"), sol.get("ik_seed")
    plan.clear = None if p["clear_z"] is None else {"clear_z": p["clear_z"], "clearance_m": p["clearance"],
                                                    "goal_raise_m": sol.get("goal_raise_m"),
                                                    "hand_low_b": sol.get("hand_low_b")}
    plan.path = sol.get("path")
    plan.approach, plan.preshape = p["approach"], p["preshape"]
    return plan


def build(ch, args: dict, now: float) -> ArmScriptPlan:
    """prepare + solve + finish in the caller's thread (tests, tools). The service runs solve() in its IK worker
    (ArmChannel.handle_arm_script)."""
    p = prepare(ch, args, now)
    return finish(ch, p, solve(p["job"]))
