"""G1 joint orders, the SONIC upper-body override layout, Dex3 hand order and joint limits (pure Python).

Every constant is copied from upstream and pinned by body/tests/test_joint_map.py (parsed from the WBC clone when
it is available, otherwise against golden copies). `$WBC` = GR00T-WholeBodyControl @ b042411,
`$DEPLOY` = $WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref.

Three orders exist (PLAN §6.6 "Joint order", invariant 8):

1. **MuJoCo / Unitree motor order** (29): rt/lowstate, rt/lowcmd, g1_debug body_q / last_action / body_q_target
   (zmq_output_handler.hpp:25-27, 315-321), P1 get_joint_state motor_q. `MUJOCO_JOINTS` below.
2. **IsaacLab order** (29): the policy's internal order ($DEPLOY/include/policy_parameters.hpp:93-99).
3. **SONIC upper-body wire order** (17): the planner message's `upper_body_position[i]` / `upper_body_velocity[i]`
   is written to IsaacLab index `upper_body_joint_isaaclab_order_in_isaaclab_index[i]` of the reference motion
   (policy_parameters.hpp:80, g1_deploy_onnx_ref.cpp:784-791, 861-867). In MuJoCo indices that is
   `upper_body_joint_isaaclab_order_in_mujoco_index` (policy_parameters.hpp:81) = `UPPER_BODY_FROM_MUJOCO`: waist
   (yaw, roll, pitch), then the two arms INTERLEAVED joint by joint (L sh_pitch, R sh_pitch, L sh_roll, R sh_roll, ...).
   A contiguous `q29[12:29]` slice would therefore put left-arm values on right-arm joints.

The 17-vector our own API accepts (op `arm`, `upper_body` as a list) is the **named MuJoCo order** of those joints,
`UPPER_BODY_MUJOCO_JOINTS` = MUJOCO_JOINTS[12:29] (waist 3, left arm 7, right arm 7). `wire_from_mj17` converts it
to the wire order; nothing else in the code base may build a wire vector by hand.

Dex3 hands (7 per hand, `left_hand_joints` / `right_hand_joints` in the planner message, zmq_manager.hpp:915-980):
thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1 = MJCF joint order used by the MuJoCo bridge
($WBC/gear_sonic/utils/mujoco_sim/base_sim.py:225-240, 300-331; g1_29dof_with_hand.xml). The deploy passes them
straight to rt/dex3/<side>/cmd (g1_deploy_onnx_ref.cpp:3991-3992) with kp 1.5 / kd 0.1, clips them to
`max_close_ratio` x the close limits and limits each write to +-0.25 rad from the measured position
(dex3_hands.hpp:114-190, 419-463). q = 0 is fully open for every Dex3 joint.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# ---------------------------------------------------------------------------------------------------------------
# 29-DoF body
# ---------------------------------------------------------------------------------------------------------------

# $DEPLOY/include/robot_parameters.hpp:90-130 (G1JointIndex) == sim_isaac/joint_map.py G1_MOTOR_JOINTS
MUJOCO_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
MJ = {n: i for i, n in enumerate(MUJOCO_JOINTS)}

# policy_parameters.hpp:93-95 `isaaclab_to_mujoco` ("mujoco order in isaaclab index"): for MuJoCo joint i, its
# IsaacLab index is ISAACLAB_IDX_OF_MUJOCO[i] (zmq_output_handler.hpp:319: body_q_mujoco[i] = q[isaaclab_to_mujoco[i]]).
ISAACLAB_IDX_OF_MUJOCO: tuple[int, ...] = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                                           11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
# policy_parameters.hpp:97-98 `mujoco_to_isaaclab` ("isaaclab order in mujoco index"): IsaacLab joint j is MuJoCo
# joint MUJOCO_IDX_OF_ISAACLAB[j].
MUJOCO_IDX_OF_ISAACLAB: tuple[int, ...] = (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                                           16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
ISAACLAB_JOINTS: tuple[str, ...] = tuple(MUJOCO_JOINTS[m] for m in MUJOCO_IDX_OF_ISAACLAB)

# policy_parameters.hpp:80-81
UPPER_BODY_ISAACLAB_IDX: tuple[int, ...] = (2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
UPPER_BODY_FROM_MUJOCO: tuple[int, ...] = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
N_UPPER = 17
UPPER_BODY_JOINTS: tuple[str, ...] = tuple(MUJOCO_JOINTS[i] for i in UPPER_BODY_FROM_MUJOCO)   # wire order
UPPER_BODY_MUJOCO_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[12:29]                             # API order
WAIST_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[12:15]
LEFT_ARM_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[15:22]
RIGHT_ARM_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[22:29]
ARM_JOINTS: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
# wire[i] = mj17[_WIRE_FROM_MJ17[i]]  and  mj17[k] = wire[_MJ17_FROM_WIRE[k]]
_WIRE_FROM_MJ17: tuple[int, ...] = tuple(i - 12 for i in UPPER_BODY_FROM_MUJOCO)
_MJ17_FROM_WIRE: tuple[int, ...] = tuple(_WIRE_FROM_MJ17.index(k) for k in range(N_UPPER))

# policy_parameters.hpp:210-240 (standing pose, MuJoCo order; = training init_state)
DEFAULT_ANGLES: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)

# URDF limits (rad): $WBC/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf (the URDF P1 converts,
# sim_isaac/g1_asset.py:29) for the body, g1_29dof_with_hand.urdf for the Dex3 joints. The deploy's
# dex3_hands.hpp:460-463 MAX/MIN_LIMITS_* equal the hand values to 2 decimals.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-2.5307, 2.8798), "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576), "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236), "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798), "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576), "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236), "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618), "waist_roll_joint": (-0.52, 0.52), "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704), "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618), "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054), "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704), "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618), "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.972222054, 1.972222054), "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
    "right_wrist_yaw_joint": (-1.614429558, 1.614429558),
    "left_hand_thumb_0_joint": (-1.04719755, 1.04719755), "left_hand_thumb_1_joint": (-0.72431163, 1.04719755),
    "left_hand_thumb_2_joint": (0.0, 1.74532925), "left_hand_middle_0_joint": (-1.57079632, 0.0),
    "left_hand_middle_1_joint": (-1.74532925, 0.0), "left_hand_index_0_joint": (-1.57079632, 0.0),
    "left_hand_index_1_joint": (-1.74532925, 0.0),
    "right_hand_thumb_0_joint": (-1.04719755, 1.04719755), "right_hand_thumb_1_joint": (-1.04719755, 0.72431163),
    "right_hand_thumb_2_joint": (-1.74532925, 0.0), "right_hand_middle_0_joint": (0.0, 1.57079632),
    "right_hand_middle_1_joint": (0.0, 1.74532925), "right_hand_index_0_joint": (0.0, 1.57079632),
    "right_hand_index_1_joint": (0.0, 1.74532925),
}

# ---------------------------------------------------------------------------------------------------------------
# Dex3
# ---------------------------------------------------------------------------------------------------------------
DEX3_SUFFIX: tuple[str, ...] = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
LEFT_HAND_JOINTS: tuple[str, ...] = tuple(f"left_hand_{s}_joint" for s in DEX3_SUFFIX)
RIGHT_HAND_JOINTS: tuple[str, ...] = tuple(f"right_hand_{s}_joint" for s in DEX3_SUFFIX)
N_HAND = 7
DEX3_OPEN: tuple[float, ...] = (0.0,) * N_HAND
# What the deploy commands when no hand override is active (input_interface.hpp:333-362): fingers at their close
# limits, thumb_2 closed, thumb_0/1 at 0. This is what the robot's hands do in M1 today (fists).
DEX3_DEPLOY_DEFAULT_LEFT: tuple[float, ...] = (0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75)
DEX3_DEPLOY_DEFAULT_RIGHT: tuple[float, ...] = (0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75)
# A power-grasp closure (fingers and thumb_2 fully closed, thumb_0/1 at 0) = the deploy default; closure f in [0, 1]
# scales it (hand_closure()).
DEX3_CLOSED = {"left": DEX3_DEPLOY_DEFAULT_LEFT, "right": DEX3_DEPLOY_DEFAULT_RIGHT}
HAND_JOINTS = {"left": LEFT_HAND_JOINTS, "right": RIGHT_HAND_JOINTS}


# ---------------------------------------------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------------------------------------------
def upper_from_mujoco(q29: Sequence[float]) -> list[float]:
    """29-D MuJoCo-order vector -> the 17-D planner `upper_body_position` (wire order)."""
    if len(q29) != 29:
        raise ValueError(f"expected 29 values, got {len(q29)}")
    return [float(q29[i]) for i in UPPER_BODY_FROM_MUJOCO]


def mujoco_from_upper(u17: Sequence[float], base29: Sequence[float] | None = None) -> list[float]:
    """17-D wire-order vector -> 29-D MuJoCo order (legs from base29, default DEFAULT_ANGLES)."""
    if len(u17) != N_UPPER:
        raise ValueError(f"expected 17 values, got {len(u17)}")
    q = [float(v) for v in (base29 if base29 is not None else DEFAULT_ANGLES)]
    for i, m in enumerate(UPPER_BODY_FROM_MUJOCO):
        q[m] = float(u17[i])
    return q


def wire_from_mj17(v17: Sequence[float]) -> list[float]:
    """17 values in UPPER_BODY_MUJOCO_JOINTS order (waist, left arm, right arm) -> wire order."""
    if len(v17) != N_UPPER:
        raise ValueError(f"expected 17 values, got {len(v17)}")
    return [float(v17[k]) for k in _WIRE_FROM_MJ17]


def mj17_from_wire(u17: Sequence[float]) -> list[float]:
    if len(u17) != N_UPPER:
        raise ValueError(f"expected 17 values, got {len(u17)}")
    return [float(u17[k]) for k in _MJ17_FROM_WIRE]


def mj17_from_mujoco(q29: Sequence[float]) -> list[float]:
    return [float(v) for v in q29[12:29]]


def mj17_from_named(named: Mapping[str, float], base_mj17: Sequence[float]) -> list[float]:
    """Overlay {joint_name: q} on base_mj17 (UPPER_BODY_MUJOCO_JOINTS order). Unknown names raise KeyError."""
    out = [float(v) for v in base_mj17]
    for name, v in named.items():
        k = _UPPER_MJ17_IDX.get(name)
        if k is None:
            raise KeyError(name)
        out[k] = float(v)
    return out


_UPPER_MJ17_IDX = {n: k for k, n in enumerate(UPPER_BODY_MUJOCO_JOINTS)}


def clamp_mj17(v17: Sequence[float], margin: float = 0.0) -> tuple[list[float], int]:
    """Clamp to the URDF limits (shrunk by margin). Returns (values, number of joints clamped)."""
    out, n = [], 0
    for name, v in zip(UPPER_BODY_MUJOCO_JOINTS, v17):
        lo, hi = JOINT_LIMITS[name]
        c = min(max(float(v), lo + margin), hi - margin)
        n += c != float(v)
        out.append(c)
    return out, n


def clamp_hand(side: str, q7: Sequence[float]) -> tuple[list[float], int]:
    if len(q7) != N_HAND:
        raise ValueError(f"expected 7 hand values, got {len(q7)}")
    out, n = [], 0
    for name, v in zip(HAND_JOINTS[side], q7):
        lo, hi = JOINT_LIMITS[name]
        c = min(max(float(v), lo), hi)
        n += c != float(v)
        out.append(c)
    return out, n


def hand_closure(side: str, frac: float) -> list[float]:
    """Dex3 7-vector for a closure fraction (0 = open, 1 = the deploy's fist)."""
    f = min(max(float(frac), 0.0), 1.0)
    return [f * c for c in DEX3_CLOSED[side]]


def hand_closure_of(side: str, q7: Sequence[float], joints: Sequence[int] = (2, 3, 4, 5, 6)) -> float:
    """Closure fraction estimated from a measured hand vector (mean over thumb_2 and the four finger joints)."""
    c = DEX3_CLOSED[side]
    return sum(float(q7[j]) / c[j] for j in joints) / len(joints)
