"""Joint orders, default pose and gains for the G1 (pure Python, no Isaac imports).

All values are copied from upstream sources; the source is cited next to each constant.
`$WBC` = GR00T-WholeBodyControl @ b042411.
"""
from __future__ import annotations

import math

# Unitree hardware motor order == MuJoCo order.
# $WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp:90-130 (G1JointIndex)
# $WBC/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml:119-148 (WeakMotorJointIndex)
G1_MOTOR_JOINTS: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_MOTORS = 29
LEG_MOTORS = list(range(12))

# Dex3 motor order per hand = MJCF joint order used by the MuJoCo bridge (hand motor i -> i-th "<side>_hand" joint):
# $WBC/gear_sonic/utils/mujoco_sim/base_sim.py:228-240,300-331; joint order in
# $WBC/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml (left_hand_thumb_0 .. left_hand_index_1).
_DEX3_SUFFIX = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
DEX3_LEFT_JOINTS: list[str] = [f"left_hand_{s}_joint" for s in _DEX3_SUFFIX]
DEX3_RIGHT_JOINTS: list[str] = [f"right_hand_{s}_joint" for s in _DEX3_SUFFIX]
NUM_HAND_MOTORS = 7

# Default standing pose in motor order: $WBC/gear_sonic_deploy/.../include/policy_parameters.hpp:210-240
# (identical to the training init_state, $WBC/gear_sonic/envs/manager_env/robots/g1.py:223-235).
DEFAULT_ANGLES: list[float] = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]

# Gains the deploy sends in rt/lowcmd: policy_parameters.hpp:38-60 (armature constants, 10 Hz, zeta=2) and
# :143-208 (kps/kds). They equal the training ImplicitActuator stiffness/damping (g1.py:10-26, 238-357).
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _k(a: float) -> float:
    return a * NATURAL_FREQ * NATURAL_FREQ


def _d(a: float) -> float:
    return 2.0 * DAMPING_RATIO * a * NATURAL_FREQ


_K5020, _K7520_14, _K7520_22, _K4010 = _k(ARMATURE_5020), _k(ARMATURE_7520_14), _k(ARMATURE_7520_22), _k(ARMATURE_4010)
_D5020, _D7520_14, _D7520_22, _D4010 = _d(ARMATURE_5020), _d(ARMATURE_7520_14), _d(ARMATURE_7520_22), _d(ARMATURE_4010)

KPS: list[float] = [
    _K7520_22, _K7520_22, _K7520_14, _K7520_22, 2 * _K5020, 2 * _K5020,
    _K7520_22, _K7520_22, _K7520_14, _K7520_22, 2 * _K5020, 2 * _K5020,
    _K7520_14, 2 * _K5020, 2 * _K5020,
    _K5020, _K5020, _K5020, _K5020, _K5020, _K4010, _K4010,
    _K5020, _K5020, _K5020, _K5020, _K5020, _K4010, _K4010,
]
KDS: list[float] = [
    _D7520_22, _D7520_22, _D7520_14, _D7520_22, 2 * _D5020, 2 * _D5020,
    _D7520_22, _D7520_22, _D7520_14, _D7520_22, 2 * _D5020, 2 * _D5020,
    _D7520_14, 2 * _D5020, 2 * _D5020,
    _D5020, _D5020, _D5020, _D5020, _D5020, _D4010, _D4010,
    _D5020, _D5020, _D5020, _D5020, _D5020, _D4010, _D4010,
]

# Dex3 hold gains used by the deploy: $WBC/gear_sonic_deploy/.../include/dex3_hands.hpp:308-332 (kp=1.5, kd=0.1)
DEX3_HOLD_KP = 1.5
DEX3_HOLD_KD = 0.1

# Training effort limits per joint (effort_limit_sim, g1.py:246-251,278,286,294,311-319); used for --pd explicit clipping
# and tau_est. Hands: URDF <limit effort>, read at runtime from the articulation.
TRAIN_EFFORT_LIMIT: dict[str, float] = {}
for _side in ("left", "right"):
    TRAIN_EFFORT_LIMIT.update({
        f"{_side}_hip_yaw_joint": 88.0, f"{_side}_hip_roll_joint": 139.0, f"{_side}_hip_pitch_joint": 139.0,
        f"{_side}_knee_joint": 139.0, f"{_side}_ankle_pitch_joint": 50.0, f"{_side}_ankle_roll_joint": 50.0,
        f"{_side}_shoulder_pitch_joint": 25.0, f"{_side}_shoulder_roll_joint": 25.0,
        f"{_side}_shoulder_yaw_joint": 25.0, f"{_side}_elbow_joint": 25.0, f"{_side}_wrist_roll_joint": 25.0,
        f"{_side}_wrist_pitch_joint": 5.0, f"{_side}_wrist_yaw_joint": 5.0,
    })
TRAIN_EFFORT_LIMIT.update({"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0})

# Head camera mount: d435_link in g1_29dof_with_hand.urdf:614-619 (parent torso_link).
D435_XYZ = (0.0576235, 0.01753, 0.41987)
D435_RPY = (0.0, 0.8307767239493009, 0.0)
# Pelvis IMU link (main.urdf:602-607); orientation is identical to the pelvis frame.
IMU_IN_PELVIS_XYZ = (0.04525, 0.0, -0.08339)


def quat_wxyz_from_rpy(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


assert len(G1_MOTOR_JOINTS) == NUM_MOTORS and len(DEFAULT_ANGLES) == NUM_MOTORS
assert len(KPS) == NUM_MOTORS and len(KDS) == NUM_MOTORS
