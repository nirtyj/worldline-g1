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


# Dex3 hand collision envelope. Link chain and joint axes from the URDF P1 simulates (assets/g1/g1_sonic_dex3.urdf:
# main.urdf with the Dex3 joints made revolute and the thumb_1 collision box from g1_29dof_with_hand.urdf), boxes =
# the bounding boxes of the collision meshes ($WBC/gear_sonic/data/assets/robot_description/meshes/g1/*_hand_*.STL),
# read 2026-09-29. Palm frame: x along the fingers, z across them (index at +z, middle at -z), y towards the thumb
# on the right hand (the left hand mirrors y). The palm alone is a slab +-4.4 cm along z, which is near vertical at
# table-height reaches: a palm target 3 cm above an object top puts the hand 1.4 cm into the object.
def _box(x0, x1, y0, y1, z0, z1):
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def _hand_links(side: str) -> list:
    """(link, parent, origin xyz, axis, Dex3 index or None, box corners) for the palm and the 7 finger links."""
    m = 1.0 if side == "right" else -1.0                       # the left hand mirrors y
    yb = lambda a, b: (min(m * a, m * b), max(m * a, m * b))   # noqa: E731
    fing = _box(-0.007, 0.055, *yb(-0.013, 0.015), -0.013, 0.013)
    tip = _box(-0.007, 0.052, *yb(-0.009, 0.009), -0.013, 0.013)
    return [("palm", None, (0.0, 0.0, 0.0), None, None, _box(0.0, 0.087, *yb(-0.021, 0.02), -0.044, 0.044)),
            ("thumb_0", "palm", (0.0255, 0.0, 0.0), (0, 1, 0), 0, _box(-0.012, 0.01, *yb(0.0, 0.028), -0.01, 0.01)),
            ("thumb_1", "thumb_0", (-0.0025, m * 0.0193, 0.0), (0, 0, 1), 1, _box(-0.011, 0.009, *yb(0.017, 0.047),
                                                                                   -0.01, 0.01)),
            ("thumb_2", "thumb_1", (0.0, m * 0.0458, 0.0), (0, 0, 1), 2, _box(-0.009, 0.009, *yb(-0.007, 0.052),
                                                                               -0.013, 0.013)),
            ("middle_0", "palm", (0.0777, -m * 0.0016, -0.0285), (0, 0, 1), 3, fing),
            ("middle_1", "middle_0", (0.0458, 0.0, 0.0), (0, 0, 1), 4, tip),
            ("index_0", "palm", (0.0777, -m * 0.0016, 0.0285), (0, 0, 1), 5, fing),
            ("index_1", "index_0", (0.0458, 0.0, 0.0), (0, 0, 1), 6, tip)]


HAND_LINKS = {s: _hand_links(s) for s in ("left", "right")}


def forearm_points(side: str, q_named: dict, n: int = 5) -> np.ndarray:
    """n points from the elbow joint to the wrist yaw joint (the forearm's axis) in the pelvis frame, (n, 3). The
    forearm is a ~3.5 cm radius link: callers check these with that much extra margin."""
    T = np.eye(4)
    for j in WAIST_CHAIN:
        T = _step(T, j, float(q_named.get(j, 0.0)))
    elbow = None
    for j in ARM_CHAIN[side]:
        T = _step(T, j, float(q_named.get(j, 0.0)))
        if j.endswith("elbow_joint"):
            elbow = T[:3, 3].copy()
    wrist = T[:3, 3]
    return np.array([elbow + (wrist - elbow) * k / (n - 1) for k in range(n)])


def hand_points(side: str, q_named: dict, hand7: Sequence[float] | None = None) -> np.ndarray:
    """The hand's collision boxes' corners in the pelvis frame, (64, 3), for the arm pose q_named and the Dex3 joint
    angles hand7 (Dex3 order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1; default open)."""
    h = [0.0] * 7 if hand7 is None else [float(v) for v in hand7]
    T = {"palm": arm_frames(side, q_named)["palm"]}
    out = []
    for name, parent, xyz, axis, k, box in HAND_LINKS[side]:
        if parent is not None:
            M = np.eye(4)
            M[:3, :3] = _axis_angle(axis, h[k])
            M[:3, 3] = xyz
            T[name] = T[parent] @ M
        c = np.asarray(box, float)
        out.append(c @ T[name][:3, :3].T + T[name][:3, 3])
    return np.vstack(out)


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


def palm_jacobian(side: str, q_named: dict, joints: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(palm 4x4, d palm_position / d q (3 x n), d palm_z_axis / d q (3 x n)) in the pelvis frame for `joints` (a
    subset of ARM_CHAIN[side]); revolute joints, so column i is axis_i x (p - origin_i) (and axis_i x z)."""
    T = np.eye(4)
    for j in WAIST_CHAIN:
        T = _step(T, j, float(q_named.get(j, 0.0)))
    axes, orgs = [], []
    for j in ARM_CHAIN[side]:
        xyz, R0 = _ORIGIN[j]
        M = np.eye(4)
        M[:3, :3] = R0
        M[:3, 3] = xyz
        Tj = T @ M
        ax = CHAIN_PARAMS[j][2]
        axes.append(Tj[:3, :3] @ np.asarray(ax, float))
        orgs.append(Tj[:3, 3])
        Rq = np.eye(4)
        Rq[:3, :3] = _axis_angle(ax, float(q_named.get(j, 0.0)))
        T = Tj @ Rq
    P = _step(T, f"{side}_hand_palm_joint", 0.0)
    p, z = P[:3, 3], P[:3, 2]
    idx = [ARM_CHAIN[side].index(j) for j in joints]
    Jp = np.stack([np.cross(axes[i], p - orgs[i]) for i in idx], axis=1)
    Jz = np.stack([np.cross(axes[i], z) for i in idx], axis=1)
    return P, Jp, Jz


def ik_palm(side: str, target_xyz: Sequence[float], q_seed: dict, q_rest: dict | None = None,
            palm_dir: Sequence[float] | None = None, w_dir: float = 0.05, w_rest: float = 0.02,
            iters: int = 200, tol: float = 2e-3, margin: float = 0.05, lock: Sequence[str] = ()) -> tuple[dict, float]:
    """Damped-least-squares IK for the palm position (pelvis frame) over the 7 arm joints; waist fixed at q_seed.

    palm_dir: optional desired direction of the palm's +z axis (the fingers close towards palm -z on Dex3; used
    softly with weight w_dir). q_rest: posture the null space is pulled to (weight w_rest). Joint limits are
    enforced with `margin`. lock: arm joints held at their q_seed value (e.g. the wrist pitch, which SONIC barely
    tracks: docs/arm_tracking.md §3.2). Analytic Jacobian (palm_jacobian): ~1-3 ms per solve, so it can run on the
    body's control thread at an op start. Returns (q_named, final position error in m)."""
    joints = tuple(j for j in ARM_CHAIN[side] if j not in lock)
    n = len(joints)
    q = dict(q_seed)
    rest = q_rest or q_seed
    rest_v = np.array([rest.get(j, 0.0) for j in joints])
    tgt = np.asarray(target_xyz, float)
    pdir = None if palm_dir is None else np.asarray(palm_dir, float)
    lo = np.array([JOINT_LIMITS[j][0] + margin for j in joints])
    hi = np.array([JOINT_LIMITS[j][1] - margin for j in joints])
    x = np.array([q.get(j, 0.0) for j in joints])
    x = np.clip(x, lo, hi)
    lam = 1e-3
    for stage in (0, 1):       # stage 1: position only (posture/orientation terms dropped) if stage 0 fell short
        for _ in range(iters):
            P, Jp, Jz = palm_jacobian(side, {**q, **dict(zip(joints, x))}, joints)
            rp = P[:3, 3] - tgt
            if float(np.linalg.norm(rp)) < tol:
                break
            rows, Js = [rp], [Jp]
            if stage == 0:
                if pdir is not None:
                    rows.append(w_dir * (P[:3, 2] - pdir))
                    Js.append(w_dir * Jz)
                rows.append(w_rest * (x - rest_v))
                Js.append(w_rest * np.eye(n))
            r, J = np.concatenate(rows), np.vstack(Js)
            dx = -np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ r)
            nn = np.linalg.norm(dx)
            if nn > 0.2:
                dx *= 0.2 / nn
            x_new = np.clip(x + dx, lo, hi)
            moved = float(np.abs(x_new - x).max())
            x = x_new
            if moved < 1e-5:           # converged (to the regularized optimum, or pinned at a limit)
                break
        P, _, _ = palm_jacobian(side, {**q, **dict(zip(joints, x))}, joints)
        if float(np.linalg.norm(P[:3, 3] - tgt)) < tol:
            break
    q.update(dict(zip(joints, (float(v) for v in x))))
    err = float(np.linalg.norm(arm_frames(side, q)["palm"][:3, 3] - tgt))
    return q, err
