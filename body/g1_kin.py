"""G1 arm kinematics in the pelvis frame (numpy only): forward kinematics and a small position IK.

The chain parameters are copied from the URDF P1 simulates, $WBC/gear_sonic/data/assets/robot_description/urdf/g1/
main.urdf (sim_isaac/g1_asset.py:29); body/tests/test_g1_kin.py re-parses the URDF when the WBC clone is present.
Frames: pelvis (x forward, y left, z up). Points: `wrist` = the wrist_yaw_link origin, `palm` =
left/right_hand_palm_link (fixed joint, 4.15 cm beyond the wrist yaw axis).

Used by tools/arm_track_test.py (wrist/palm tracking error, keyframes for the reach-grasp-lift script), by
body/arm_script.py (one IK solve per phase at the op start, then FK of the measured arm for the palm error) and by
body/carry.py (CarryLock palm error in body.state). The 50 Hz path itself never solves IK.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .joint_map import JOINT_LIMITS, MJ, UPPER_BODY_MUJOCO_JOINTS

# joint: (origin xyz, origin rpy, axis or None for fixed)
CHAIN_PARAMS: dict[str, tuple[tuple, tuple, tuple | None]] = {
    "waist_yaw_joint": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "waist_roll_joint": ((-0.0039635, 0.0, 0.044), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "waist_pitch_joint": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left_shoulder_pitch_joint": ((0.0039563, 0.10022, 0.24778), (0.27931, 5.4949e-05, -0.00019159), (0.0, 1.0, 0.0)),
    "left_shoulder_roll_joint": ((0.0, 0.038, -0.013831), (-0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "left_shoulder_yaw_joint": ((0.0, 0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left_elbow_joint": ((0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left_wrist_roll_joint": ((0.1, 0.00188791, -0.01), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "left_wrist_pitch_joint": ((0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left_wrist_yaw_joint": ((0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left_hand_palm_joint": ((0.0415, 0.003, 0.0), (0.0, 0.0, 0.0), None),
    "right_shoulder_pitch_joint": ((0.0039563, -0.10021, 0.24778), (-0.27931, 5.4949e-05, 0.00019159), (0.0, 1.0, 0.0)),
    "right_shoulder_roll_joint": ((0.0, -0.038, -0.013831), (0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "right_shoulder_yaw_joint": ((0.0, -0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right_elbow_joint": ((0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right_wrist_roll_joint": ((0.1, -0.00188791, -0.01), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "right_wrist_pitch_joint": ((0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right_wrist_yaw_joint": ((0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right_hand_palm_joint": ((0.0415, -0.003, 0.0), (0.0, 0.0, 0.0), None),
}
WAIST_CHAIN = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_CHAIN = {s: tuple(f"{s}_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                   "wrist_roll", "wrist_pitch", "wrist_yaw")) for s in ("left", "right")}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a: Sequence[float], q: float) -> np.ndarray:
    x, y, z = a
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


_ORIGIN = {n: (np.array(p[0]), _rpy(*p[1])) for n, p in CHAIN_PARAMS.items()}


def _step(T: np.ndarray, joint: str, q: float) -> np.ndarray:
    xyz, R0 = _ORIGIN[joint]
    ax = CHAIN_PARAMS[joint][2]
    M = np.eye(4)
    M[:3, :3] = R0 if ax is None else R0 @ _axis_angle(ax, q)
    M[:3, 3] = xyz
    return T @ M


def arm_frames(side: str, q_named: dict) -> dict[str, np.ndarray]:
    """4x4 transforms in the pelvis frame: torso, wrist (wrist_yaw_link), palm. q_named: joint -> rad (0 if absent)."""
    T = np.eye(4)
    for j in WAIST_CHAIN:
        T = _step(T, j, float(q_named.get(j, 0.0)))
    torso = T
    for j in ARM_CHAIN[side]:
        T = _step(T, j, float(q_named.get(j, 0.0)))
    wrist = T
    palm = _step(T, f"{side}_hand_palm_joint", 0.0)
    return {"torso": torso, "wrist": wrist, "palm": palm}


def named_from_mj17(v17: Sequence[float]) -> dict:
    return {n: float(v) for n, v in zip(UPPER_BODY_MUJOCO_JOINTS, v17)}


def named_from_q29(q29: Sequence[float]) -> dict:
    return {n: float(q29[MJ[n]]) for n in UPPER_BODY_MUJOCO_JOINTS}


def points(q_named: dict, frame: str = "pelvis") -> dict[str, np.ndarray]:
    """{'left_wrist','left_palm','right_wrist','right_palm'} positions (3,) in the pelvis or torso frame."""
    out = {}
    for s in ("left", "right"):
        f = arm_frames(s, q_named)
        if frame == "torso":
            Ti = np.linalg.inv(f["torso"])
            out[f"{s}_wrist"] = (Ti @ f["wrist"])[:3, 3]
            out[f"{s}_palm"] = (Ti @ f["palm"])[:3, 3]
        else:
            out[f"{s}_wrist"] = f["wrist"][:3, 3]
            out[f"{s}_palm"] = f["palm"][:3, 3]
    return out


def ik_palm(side: str, target_xyz: Sequence[float], q_seed: dict, q_rest: dict | None = None,
            palm_dir: Sequence[float] | None = None, w_dir: float = 0.05, w_rest: float = 0.02,
            iters: int = 200, tol: float = 2e-3, margin: float = 0.05, lock: Sequence[str] = ()) -> tuple[dict, float]:
    """Damped-least-squares IK for the palm position (pelvis frame) over the 7 arm joints; waist fixed at q_seed.

    palm_dir: optional desired direction of the palm's +z axis (the fingers close towards palm -z on Dex3; used
    softly with weight w_dir). q_rest: posture the null space is pulled to (weight w_rest). Joint limits are
    enforced with `margin`. lock: arm joints held at their q_seed value (e.g. the wrist pitch, which SONIC barely
    tracks: docs/arm_tracking.md §3.2). Returns (q_named, final position error in m)."""
    joints = tuple(j for j in ARM_CHAIN[side] if j not in lock)
    q = dict(q_seed)
    rest = q_rest or q_seed
    tgt = np.asarray(target_xyz, float)
    lo = np.array([JOINT_LIMITS[j][0] + margin for j in joints])
    hi = np.array([JOINT_LIMITS[j][1] - margin for j in joints])
    x = np.array([q.get(j, 0.0) for j in joints])
    x = np.clip(x, lo, hi)

    def resid(xv):
        qq = {**q, **dict(zip(joints, xv))}
        P = arm_frames(side, qq)["palm"]
        r = [P[:3, 3] - tgt]
        if palm_dir is not None:
            r.append(w_dir * (P[:3, 2] - np.asarray(palm_dir, float)))
        r.append(w_rest * (xv - np.array([rest.get(j, 0.0) for j in joints])))
        return np.concatenate(r)

    lam = 1e-2
    for stage in (0, 1):       # stage 1: position only (posture/orientation terms dropped) if stage 0 fell short
        for _ in range(iters):
            r = resid(x)
            if stage == 1:
                r = r[:3]
            pos_err = float(np.linalg.norm(r[:3]))
            if pos_err < tol:
                break
            J = np.zeros((len(r), len(joints)))
            for k in range(len(joints)):
                d = np.zeros(len(joints))
                d[k] = 1e-5
                J[:, k] = ((resid(x + d)[:len(r)]) - r) / 1e-5
            dx = -np.linalg.solve(J.T @ J + lam * np.eye(len(joints)), J.T @ r)
            n = np.linalg.norm(dx)
            if n > 0.2:
                dx *= 0.2 / n
            x = np.clip(x + dx, lo, hi)
        if float(np.linalg.norm(resid(x)[:3])) < tol:
            break
    q.update(dict(zip(joints, (float(v) for v in x))))
    err = float(np.linalg.norm(arm_frames(side, q)["palm"][:3, 3] - tgt))
    return q, err
