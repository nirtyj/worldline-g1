"""Joint orders for the GR00T N1.7 G1 checkpoint, SONIC and Dex3. Every conversion goes BY NAME.

Pure Python + numpy, no import from body/ (P3 owns body/joint_map.py; tests/groot/test_joint_order.py checks that the
two agree whenever body.joint_map is importable). Sources (`$WBC` = GR00T-WholeBodyControl @ b042411,
`$DEPLOY` = $WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref, `$ARENA` = IsaacLab-Arena release/0.2.1 @ 8b4a3a47):

1. **GR00T keys** (the checkpoint's NEW_EMBODIMENT modality config, `experiment_cfg/conf.yaml`, identical to
   `$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`):
   state  left_arm 7, right_arm 7, left_hand 7, right_hand 7, waist 3;
   action the same five + base_height_command 1 + navigate_command 3 (never executed here).
   Joint names per key: `$ARENA/isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml`. Arm order =
   the MuJoCo arm order; **hand order = index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2**.
2. **LeRobot 43** (`observation.state` / `action` of nvidia/Arena-G1-Static-PickNPlace-Task @ 37ba80a,
   `lerobot/meta/info.json` names; sliced by `meta/modality.json`): legs 12, waist 3, L arm 7, L hand 7, R arm 7,
   R hand 7, hands in GR00T order.
3. **MuJoCo / Unitree 29** (`$DEPLOY/include/robot_parameters.hpp:90-130`; g1_debug `body_q`, rt/lowstate,
   docs/contracts/m1.md §1.3).
4. **SONIC upper-body wire order 17** (the planner message's `upper_body_position[i]`): MuJoCo indices
   `upper_body_joint_isaaclab_order_in_mujoco_index` (`$DEPLOY/include/policy_parameters.hpp:81`) = waist 3, then
   the two arms INTERLEAVED joint by joint. A contiguous `q29[12:29]` slice is NOT this order.
5. **mj17** = MuJoCo[12:29] (waist 3, L arm 7, R arm 7): the order of a 17-list in the body `arm` op
   (body/arm.py `_vec17`, main checkout).
6. **Dex3 order** (g1_debug `left/right_hand_q`, planner `left/right_hand_joints`, body `arm` op hands):
   thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1 (`$WBC/gear_sonic/utils/mujoco_sim/base_sim.py:225-240`).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# 29-DoF body (MuJoCo / Unitree motor order)
# ---------------------------------------------------------------------------------------------------------------
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
LEG_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[0:12]
WAIST_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[12:15]
LEFT_ARM_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[15:22]
RIGHT_ARM_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[22:29]

# mj17: the body `arm` op's list order (waist, left arm, right arm; MuJoCo order)
MJ17_JOINTS: tuple[str, ...] = MUJOCO_JOINTS[12:29]

# SONIC wire order: policy_parameters.hpp:81 (MuJoCo indices of upper_body_position[0..16])
SONIC_WIRE_FROM_MUJOCO: tuple[int, ...] = (12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
SONIC_UPPER_JOINTS: tuple[str, ...] = tuple(MUJOCO_JOINTS[i] for i in SONIC_WIRE_FROM_MUJOCO)
N_UPPER = 17

# SONIC's standing pose, MuJoCo order ($DEPLOY/include/policy_parameters.hpp:210-240; = the training init_state).
STAND_Q29: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)
STAND_WAIST: tuple[float, float, float] = STAND_Q29[12:15]

# ---------------------------------------------------------------------------------------------------------------
# Dex3 hands
# ---------------------------------------------------------------------------------------------------------------
N_HAND = 7
DEX3_SUFFIX: tuple[str, ...] = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
GROOT_HAND_SUFFIX: tuple[str, ...] = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
SIDES = ("left", "right")


def hand_joints(side: str, suffixes: Sequence[str]) -> tuple[str, ...]:
    if side not in SIDES:
        raise ValueError(f"side must be left|right, got {side!r}")
    return tuple(f"{side}_hand_{s}_joint" for s in suffixes)


DEX3_HAND_JOINTS: dict[str, tuple[str, ...]] = {s: hand_joints(s, DEX3_SUFFIX) for s in SIDES}
GROOT_HAND_JOINTS: dict[str, tuple[str, ...]] = {s: hand_joints(s, GROOT_HAND_SUFFIX) for s in SIDES}

# ---------------------------------------------------------------------------------------------------------------
# GR00T / Arena keys (NEW_EMBODIMENT, projector slot 10)
# ---------------------------------------------------------------------------------------------------------------
EMBODIMENT_TAG = "new_embodiment"
VIDEO_KEY = "ego_view"
LANGUAGE_KEY = "annotation.human.task_description"
GROOT_STATE_KEYS: tuple[str, ...] = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
GROOT_ACTION_KEYS: tuple[str, ...] = GROOT_STATE_KEYS + ("base_height_command", "navigate_command")
GROOT_KEY_JOINTS: dict[str, tuple[str, ...]] = {
    "left_arm": LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
    "left_hand": GROOT_HAND_JOINTS["left"],
    "right_hand": GROOT_HAND_JOINTS["right"],
    "waist": WAIST_JOINTS,
}
# non-joint action keys (component names are ours; order from Arena's action layout, docs/arena_vs_sonic.md §3)
GROOT_COMMAND_COMPONENTS: dict[str, tuple[str, ...]] = {
    "base_height_command": ("base_height_m",),
    "navigate_command": ("vx_mps", "vy_mps", "wz_radps"),
}
GROOT_KEY_DIMS: dict[str, int] = {**{k: len(v) for k, v in GROOT_KEY_JOINTS.items()},
                                  **{k: len(v) for k, v in GROOT_COMMAND_COMPONENTS.items()}}
ACTION_HORIZON = 40          # N1.7 (conf.yaml action_horizon; delta_indices range(40))
CONTROL_HZ = 50.0            # LeRobot fps 50 (info.json); SONIC's 50 Hz policy tick
DT = 1.0 / CONTROL_HZ
IMAGE_HW: tuple[int, int] = (480, 640)

# LeRobot 43 (info.json `observation.state.names` == `action.names`)
LEROBOT_43_JOINTS: tuple[str, ...] = (
    LEG_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + GROOT_HAND_JOINTS["left"] + RIGHT_ARM_JOINTS
    + GROOT_HAND_JOINTS["right"])

# ---------------------------------------------------------------------------------------------------------------
# URDF limits (rad), from the URDF P1 simulates: /work/worldline-g1/assets/g1/g1_sonic_dex3.urdf (sha256 e6cb9f36…,
# built by sim_isaac/g1_asset.py from $WBC/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf + the Dex3
# joints of $WBC/gear_sonic/data/robots/g1/g1_29dof_with_hand.urdf). tests/groot pins this table to the file.
# ---------------------------------------------------------------------------------------------------------------
URDF_LIMITS: dict[str, tuple[float, float]] = {
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
# conversions (by name)
# ---------------------------------------------------------------------------------------------------------------
def index_map(src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """Indices idx such that values_dst = values_src[..., idx]. Every dst name must exist in src."""
    pos = {n: i for i, n in enumerate(src)}
    if len(pos) != len(src):
        raise ValueError("duplicate joint names in src")
    missing = [n for n in dst if n not in pos]
    if missing:
        raise KeyError(f"joints not in source order: {missing}")
    return np.array([pos[n] for n in dst], dtype=np.intp)


def reorder(values, src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """values[..., len(src)] -> values[..., len(dst)], matched by joint name."""
    v = np.asarray(values)
    if v.shape[-1] != len(src):
        raise ValueError(f"last dim {v.shape[-1]} != {len(src)} source joints")
    return v[..., index_map(src, dst)]


def dex3_to_groot_hand(side: str, q7) -> np.ndarray:
    return reorder(q7, DEX3_HAND_JOINTS[side], GROOT_HAND_JOINTS[side])


def groot_to_dex3_hand(side: str, q7) -> np.ndarray:
    return reorder(q7, GROOT_HAND_JOINTS[side], DEX3_HAND_JOINTS[side])


def mj17_to_wire(v17) -> np.ndarray:
    return reorder(v17, MJ17_JOINTS, SONIC_UPPER_JOINTS)


def wire_to_mj17(u17) -> np.ndarray:
    return reorder(u17, SONIC_UPPER_JOINTS, MJ17_JOINTS)


def groot_state_from_body(body_q_mj29, left_hand_dex3, right_hand_dex3) -> dict[str, np.ndarray]:
    """g1_debug-style state (29 MuJoCo + two Dex3 hands) -> the five GR00T state keys (float64, last dim D)."""
    q = np.asarray(body_q_mj29, dtype=np.float64)
    if q.shape[-1] != len(MUJOCO_JOINTS):
        raise ValueError(f"body q must have 29 values (MuJoCo order), got {q.shape[-1]}")
    out = {k: reorder(q, MUJOCO_JOINTS, GROOT_KEY_JOINTS[k]) for k in ("left_arm", "right_arm", "waist")}
    out["left_hand"] = dex3_to_groot_hand("left", np.asarray(left_hand_dex3, dtype=np.float64))
    out["right_hand"] = dex3_to_groot_hand("right", np.asarray(right_hand_dex3, dtype=np.float64))
    return {k: out[k] for k in GROOT_STATE_KEYS}


def body_from_lerobot43(v43) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A LeRobot 43-D row/array -> (body q 29 MuJoCo order, left hand Dex3 order, right hand Dex3 order)."""
    v = np.asarray(v43)
    return (reorder(v, LEROBOT_43_JOINTS, MUJOCO_JOINTS),
            reorder(v, LEROBOT_43_JOINTS, DEX3_HAND_JOINTS["left"]),
            reorder(v, LEROBOT_43_JOINTS, DEX3_HAND_JOINTS["right"]))


def groot_keys_from_lerobot43(v43) -> dict[str, np.ndarray]:
    """A LeRobot 43-D row/array -> the five GR00T joint keys (what GR00T's own loader slices via modality.json)."""
    v = np.asarray(v43)
    return {k: reorder(v, LEROBOT_43_JOINTS, GROOT_KEY_JOINTS[k]) for k in GROOT_STATE_KEYS}


def limits_array(names: Sequence[str], limits: Mapping[str, tuple[float, float]] = URDF_LIMITS,
                 margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([limits[n][0] + margin for n in names], dtype=np.float64)
    hi = np.array([limits[n][1] - margin for n in names], dtype=np.float64)
    return lo, hi
