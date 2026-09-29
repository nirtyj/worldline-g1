# GR00T drives the G1's arms and hands on SONIC: design for recommendation (b)

Status: **design, 2026-09-29. Nothing here has run yet.** Owner decision (2026-09-29): recommendation (b) of
`docs/arena_vs_sonic.md`. GR00T outputs arm and Dex3 hand joint targets. They are streamed into SONIC through the
planner message's upper-body override. SONIC keeps the legs and balance, and walking still comes from its planner.
SONIC stays the only body controller (PLAN §0.2).

This document covers the GR00T side: the embodiment, the data, the fine-tune, the inference loop and the
milestones. The body side, meaning the `arm` op in P3 and the SONIC upper-body tracking test, belongs to the body
agent. This design depends on both (§6, G0).

Tags: **[v]** read in code (file:line given). **[m]** measured, with the run named. **[u]** unverified or my
estimate. Each [u] has a check in §6.

Sources, all read for this document:

| Short name | What | Revision |
|---|---|---|
| `$WBC` | NVlabs/GR00T-WholeBodyControl (SONIC deploy, `gear_sonic`, `decoupled_wbc`) | `b042411` |
| `$DEPLOY` | `$WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref` | `b042411` |
| `$GR00T` | NVIDIA/Isaac-GR00T | `51d4c89` |
| `$ARENA` | IsaacLab-Arena `release/0.2.1` | `8b4a3a47` |
| `$ILAB` | IsaacLab | v2.3.2 `37ddf62` |
| HF Static | `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` (`experiment_cfg/*`, `processor_config.json`) | `7f78beb` |
| HF Loco | `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` (`experiment_cfg/*`) | `gn1_6` @ `dfe74af` |
| HF Base | `nvidia/GR00T-N1.7-3B` (`processor_config.json`, `statistics.json`) | `2fc962b` |
| HF Data | `nvidia/Arena-G1-Static-PickNPlace-Task` (`lerobot/meta/*`) | `37ba80a` |

---

## 0. Answers in brief

| Question | Answer |
|---|---|
| Embodiment | `NEW_EMBODIMENT`, with **exactly** Arena's G1 N1.7 modality config (`$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py`) and its `modality.json`, unchanged. Not `UNITREE_G1`: its 50-step horizon fails N1.7's 40-step check (§2.1) |
| Video | `ego_view`, 640x480 RGB at 50 Hz (latest frame held). No crop on our side; GR00T resizes and crops itself. **Camera: match Arena's head camera.** Our d435 mount does not see an object on a 0.95 m counter from the stance we use (§2.3). Owner decision OD1 |
| State | `left_arm, right_arm, left_hand, right_hand, waist` (31-D), measured, from `g1_debug`. No base velocity: the base is still during `manipulate` |
| Action | Same 7 keys as Arena, all **ABSOLUTE**, 40 steps (0.8 s at 50 Hz). Arms and hands are executed. `waist`, `base_height_command` and `navigate_command` are kept in the schema only so that the layout stays compatible with the Arena checkpoint. They are recorded as constants and never executed. **Navigation stays out of GR00T entirely** |
| Normalization | GR00T's default: q01/q99 min-max to [-1, 1] with clipping, recomputed from our dataset |
| Arena checkpoint as init | **Yes, the N1.7 Static checkpoint**. It has the same model, tag slot, keys, joint semantics, Dex3 order, 640x480 input, 50 Hz and horizon. Its camera and scene differ, and its licence is ambiguous (§2.7). **The N1.6 Loco checkpoint cannot be used**: it is a different architecture |
| Data | Scripted IK and keyframe picks driven through the body `arm` op, using ground-truth object poses. Randomized, labelled automatically from ground truth, and recorded in Arena's LeRobot v2.1 schema. 100-150 successful demos per skill. First objects: apple (or the mug, if THOR's large apples defeat the Dex3 grasp), then alarm clock, then mug. Books are not graspable here. Mimic-style augmentation is second; teleop is optional (§3) |
| Fine-tune | One L40S with Isaac off (the dev box): the default recipe (frozen VLM, tune projector and DiT), about 35 GB. Estimated 4-8 h per run [u]. Run both inits and a data-size sweep. Evaluation is closed-loop in Isaac: 20 fixed-seed trials per type, success judged from ground truth (§4) |
| Inference | PolicyServer on 5550 → a GR00T client thread in the `ManipulationService` executor → timestamped 40-step chunks to P3's `arm` stream. P3 plays them at 50 Hz, indexing by time since the observation (latency compensation), cross-fading chunks and clamping. `ManipulationService` holds the lease, fences sessions, cancels, and detects success from ground truth, then hands over to CarryLock (§5) |
| Effort | 14-18.5 engineer-days (including G0 and the P1 ops) plus 1.5-2 GPU-days, gated by G0 (the SONIC tracking test). PLAN M6 budgeted 9.5-14.5 ED (§6) |

---

## 1. What SONIC does with the override (re-verified in code, `$WBC` @ `b042411`)

This is the channel GR00T writes into. Every row matters for the design.

| Fact | Where | Consequence for GR00T |
|---|---|---|
| The planner message carries optional `upper_body_position` / `upper_body_velocity` (17 values each) and `left/right_hand_joints` (7 values each), f32 or f64 | `$DEPLOY/include/input_interface/zmq_manager.hpp:885-913, 916-944, 946-1006` [v] | Nothing new is needed on the wire; `body/wire.py` already packs these fields |
| The 17-vector is in SONIC's **IsaacLab-interleaved** order. Element i goes to IsaacLab index `upper_body_joint_isaaclab_order_in_isaaclab_index[i]` = {2,5,8,11,12,15,16,…}, i.e. MuJoCo indices {12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28} | `$DEPLOY/include/policy_parameters.hpp:80-81`; `$DEPLOY/src/g1_deploy_onnx_ref.cpp:784-791` [v] | GR00T's arm order must go through `body/joint_map.wire_from_mj17`. A contiguous `body_q[12:29]` slice would put left-arm values on right-arm joints (PLAN §6.6, invariant 8) |
| **No blending and no mode dependence.** When the override is active, the 17 joints are **replaced** in every one of the reference frames the encoder reads (`motion_joint_positions_10frame_step5`: 10 frames, 5 ticks apart, so 0.9 s ahead), all with the same latest target. The code path is identical in IDLE and SLOW_WALK | `g1_deploy_onnx_ref.cpp:738-800, 1749`; `policy/release/observation_config.yaml:58-62` (encoder mode `g1`) [v] | SONIC sees a *step* reference that GR00T keeps moving; the arm motion is SONIC's tracking response, not direct PD. While walking, the planner's arm swing is overwritten. That is upstream's `PLANNER_FROZEN_UPPER_BODY` stream mode (`$WBC/gear_sonic/scripts/pico_manager_thread_server.py:1832-1836`), i.e. CarryLock |
| Reference velocities are replaced by `upper_body_velocity` (only while playing). **If it is never sent they are zeros** | `g1_deploy_onnx_ref.cpp:819-870`; `input_interface.hpp:382-391` [v] | Default: send positions only (zero velocity), matching the future frames, which are constant. Sending finite-difference velocities is an A/B for the G0 tracking test |
| The override stays on only while each **consumed** planner message carries the field. `has_upper_body_control_` is re-evaluated on every message (input thread, 100 Hz). It is cleared by a message without the field, by the 1 s planner timeout, by stop/estop, and on a switch into PLANNER mode unless a planner message under 100 ms old exists | `zmq_manager.hpp:254-266, 315-361, 586-592, 620-628` [v] | P3 must attach the override to **every** planner message while an arm stream or CarryLock is active. Dropping it snaps the arms back to the planner pose, which SONIC tracks at its own speed. Release must therefore blend back to the stand pose first (§5.3) |
| **No limit or rate check** on the 17 values anywhere in the deploy | the whole path above [v] | P3 must clamp to the URDF limits and rate-limit (`joint_map.clamp_mj17`, §5.3) |
| Hands bypass the policy. The 7+7 targets go straight to `rt/dex3/*/cmd`, written by the 500 Hz command writer. Each write is clipped to `max_close_ratio` × the close limits (default 1.0) and to ±0.25 rad around the **measured** position | `g1_deploy_onnx_ref.cpp:2715, 3990-3992`; `include/dex3_hands.hpp:114-190, 419-463`; `input_interface.hpp:498` [v] | Hand targets are PD targets (P1 applies kp 1.5 / kd 0.1, `docs/contracts/m1.md` §1.2). Grasp strength in PhysX is a risk (R3) |
| With no hand data, the deploy commands a **fist** (left `{0,0,1.75,-1.57,-1.75,-1.57,-1.75}`). If only one hand is sent, the other reuses the last value the deploy ever received for it (the fist only if none was ever received) | `input_interface.hpp:341-362` [v] | Always send both hands. Before a skill starts, P3 opens the hands (q = 0) so the start state matches the demos |
| Dex3 order on the wire and in `g1_debug`: `thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1` | `$WBC/gear_sonic/utils/mujoco_sim/base_sim.py:225-240`; `gear_sonic_deploy/g1/g1_29dof_with_hand.xml` [v] | GR00T/Arena use `index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2`. Reorder by name (§2.6) |
| `g1_debug.body_q` has 29 values in MuJoCo/Unitree order, with default offsets added (absolute). `left/right_hand_q` are the measured Dex3 values in Dex3 order. `token_state[64]` is the encoder's token, published every tick | `$DEPLOY/include/output_interface/zmq_output_handler.hpp:25-75, 315-321` [v] | This is the state source for GR00T, the same as on the real robot. `token_state` is also recorded, which gives SONIC-token labels for free (§3.1) |

---

## 2. The GR00T N1.7 embodiment

### 2.1 Options

| Option | Tag and slot | For | Against | Verdict |
|---|---|---|---|---|
| **A. Arena's G1 N1.7 config under `NEW_EMBODIMENT`** | `new_embodiment`, projector slot 10 (`$GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:68-75`) | This is the config NVIDIA's own G1 Dex3 N1.7 checkpoint was trained with (HF Static `experiment_cfg/conf.yaml`: `embodiment_tag: new_embodiment`, keys below). The file ships ready to use, and our data can mix with the CC-BY Arena dataset | Slot 10 of the *base* model is only a fine-tune placeholder, so from the base the projector starts cold. Arena init fixes that (§2.7) | **Chosen** |
| B. `UNITREE_G1` (`unitree_g1_full_body_with_waist_height_nav_cmd`) | slot 25, shared with the `REAL_G1` pretrain tag (`processing_gr00t_n1d7.py:78-83`) | Slot 25 was pretrained on real G1 + Dex3 data. HF Base `statistics.json` has 7-D hands with Dex3 ranges, in GR00T order | Its registered config has **50** action steps (`$GR00T/gr00t/configs/data/embodiment_configs.py:115-191`). N1.7's `action_horizon` is 40 (`gr00t/configs/model/gr00t_n1d7.py:76`), and `validate_action_horizons` raises (`processing_gr00t_n1d7.py:132-155`). Relative arms, legs in state. It cannot be re-registered (`embodiment_configs.py:355-361` asserts) | Rejected |
| C. A `REAL_G1`-shaped schema in slot 25 | slot 25 | Uses the real-G1 pretraining prior, and the base model is commercially licensed | It needs wrist EEF 9-D (xyz + rot6d, frame [u]) through FK, relative EEF and relative arm actions, and 2 video frames (t-20, t) (HF Base `processor_config.json`). Reaching the slot would mean patching `MODALITY_CONFIGS` for a posttrain tag. Unverified plumbing | Plan B if A underperforms or the Arena licence blocks it (§6, OD2) |
| D. `UNITREE_G1_SONIC` (64-D motion tokens + hands) | slot 11 | PLAN's original M6 route | Decision (b) replaces it. Our demos still record `token_state`, so it stays open | Not now |

### 2.2 The modality config (option A)

Use `$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py` **verbatim**, copied into
the repo as `groot/g1_arms_config.py`, and `$ARENA/.../g1/modality.json` as the dataset's `meta/modality.json`.
Content [v]:

```python
video:    delta_indices=[0],          keys=["ego_view"]
state:    delta_indices=[0],          keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"]   # 7+7+7+7+3
action:   delta_indices=range(40),    keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist",
                                            "base_height_command", "navigate_command"]                  # 7+7+7+7+3+1+3
          action_configs = 7 x ActionConfig(rep=ABSOLUTE, type=NON_EEF, format=DEFAULT)
language: delta_indices=[0],          keys=["annotation.human.task_description"]
register_modality_config(..., embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

`modality.json` slices one 43-D `observation.state` / `action` vector (legs 0-12, waist 12-15, left_arm 15-22,
left_hand 22-29, right_arm 29-36, right_hand 36-43) and reads `teleop.base_height_command`,
`teleop.navigate_command` and `observation.images.ego_view`.

### 2.3 Video

- **Key:** `ego_view`, one frame (delta 0), 480x640x3 uint8 RGB. It is recorded at 50 Hz by holding the latest
  camera frame (P1 renders at 30 Hz), which is exactly what the policy sees at run time.
- **Preprocessing is GR00T's, not ours.** In training it resizes the short edge to 256, takes a random crop of
  230/256 = 0.90 of the frame and resizes back. At eval time the crop is centred (`use_albumentations_transforms`,
  `$GR00T/gr00t/model/gr00t_n1d7/image_augmentations.py:398-426`; image sizes `gr00t_n1d7.py:54-55`; the Static
  checkpoint uses the same, HF Static `conf.yaml`). The 4:3 aspect ratio is kept. Send the full 640x480 frame.
  Colour jitter as Arena: brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08.
- **Camera geometry: recommend matching Arena's head camera (owner decision OD1).**

  | | P1 today (`ego`, `docs/contracts/m1.md` §1.4) | Arena G1 (`$ARENA/isaaclab_arena/embodiments/g1/g1.py:104-106, 508-538`) |
  |---|---|---|
  | Mount | `torso_link` + (0.058, 0.018, 0.420) = the real D435 | `head_link` + (0.045, 0, 0.353); `head_link` = `torso_link` + (0.004, 0, −0.054) (`g1_29dof_with_hand.urdf:571-575`) → torso + (0.049, 0, 0.299) [u: assumes Arena's `rev_1_0` USD matches the URDF] |
  | Height, standing (pelvis 0.787 m [m], torso +0.054) | ≈ 1.26 m | ≈ 1.14 m |
  | Pitch down | 47.6° | 35.0° (computed from the quaternion) |
  | FOV at 640x480 | VFOV 45°, HFOV 57.8° | f 15 mm, aperture 20.955 mm: HFOV 69.9°, VFOV 55.3° |
  | Visible band below horizontal | 25°-70° | 7°-63° |

  Take an object centred about 1.0 m high on a 0.94-1.05 m counter (`eval/scenes.yaml`: `apple_1`, `mug_2`), seen
  0.5-0.7 m ahead from the 0.35-0.50 m stance the world agent now uses. With the d435 mount it sits 20-28° below
  the camera, at or above the image's top edge (25°); it only enters the image within about 0.55 m. With the Arena
  mount it sits 11-16° below, well inside the image. The hands at chest height are also more central in the Arena
  view. Matching Arena also keeps the Arena checkpoint's visual prior, and makes the Arena dataset usable for
  co-training.

  The cost: Arena's head camera is not a real G1 sensor, whereas the d435 mount is. This is a P1 config change
  (mount + intrinsics) for the isaac agent. **G1 first renders both views** at three stances in the train-38
  kitchen before any data is recorded (§6).

### 2.4 State

| Key | Dim | Source at run time and in recording | Notes |
|---|---|---|---|
| `left_arm`, `right_arm` | 7 + 7 | `g1_debug.body_q[15:22]`, `[22:29]` (MuJoCo order = GR00T arm order: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) | Measured, absolute rad. Same order as `$ARENA/.../gr00t_43dof_joint_space.yaml` [v] |
| `left_hand`, `right_hand` | 7 + 7 | `g1_debug.left_hand_q` / `right_hand_q`, reordered from Dex3 to GR00T order | Measured |
| `waist` | 3 | `g1_debug.body_q[12:15]` | About 0, since the waist is held |

- **No base velocity and no projected gravity.** The base is still during `manipulate` (PLAN #17, #26). An extra
  key would change the input layout the Arena init was trained on.
- `g1_debug` is the source rather than P1's ground truth, because it is what the real robot provides. P1's
  `get_joint_state` is used only for cross-checks.
- `state_dropout_prob` = 0.2, the fine-tune default (`$GR00T/gr00t/configs/finetune_config.py:61`), as Arena used.
  Note that the model-config default is 0.8 (`gr00t_n1d7.py:118`); `launch_finetune.py` overrides it.

### 2.5 Action

- **Arms: ABSOLUTE joint targets.** This is the representation of the Arena Static checkpoint: every key is
  `rep: ABSOLUTE` in HF Static `conf.yaml`. The finetune script sets `use_relative_action = True` globally
  (`$GR00T/gr00t/experiment/launch_finetune.py:102`), but only keys marked RELATIVE are converted. SONIC also takes
  absolute targets. Relative arms (the `UNITREE_G1` / `REAL_G1` choice) are an ablation for option C only.
- **Hands: ABSOLUTE, effectively binary.** Arena's hand actions are open (0) or one fixed closed pose. The left
  hand in GR00T order is `index -0.6/-1.2, middle -0.6/-1.2, thumb_0 0, thumb_1 0.7, thumb_2 0.7`, and the right
  hand is mirrored (HF Static `dataset_statistics.json`: min/max). Our scripted demos use the same two poses, so
  the init's hand head stays meaningful. Note this is **not** the deploy's fist (§1).
- **Not executed:**
  - `waist` is recorded as the held value (0, 0, 0). Arena's waist action is identically 0 too: its dataset is
    `..._locked_waist_...`.
  - `base_height_command` is recorded as a constant (0.75) and never sent. SONIC's `height` stays −1.
  - `navigate_command` is recorded as (0, 0, 0). At run time it is logged as `groot_nav_pred` and dropped (the
    same rule as `docs/arena_vs_sonic.md` §4.1).
  - Constant dimensions normalize safely (range floored at 1e-8, `$GR00T/gr00t/data/state_action/state_action_processor.py:167-169`).
- **Why navigation stays out:** the Ludi doc (§15) says GR00T drives the arms and navigation is separate.
  `manipulate` never moves the base (PLAN #17, #26). The Arena probe showed the one walking checkpoint replays a
  memorised route (`arena_vs_sonic.md` §2.3). Our planner already walks the house on SONIC.
- **Horizon:** 40 steps = 0.8 s at 50 Hz. This is fixed by N1.7 (`gr00t_n1d7.py:76`) and equals SONIC's 50 Hz tick.
- **Normalization:** `use_percentiles=True` maps q01/q99 to [-1, 1] and clips outliers
  (`state_action_processor.py:158-169, 263, 414`). Stats are recomputed from our dataset; Arena set
  `override_pretraining_statistics: true`.

### 2.6 Joint orders (all by name; one module)

| Space | Arm order | Hand order | Where |
|---|---|---|---|
| GR00T / Arena / LeRobot `observation.state` (43) | legs 12, waist 3, L arm 7, **L hand 7**, R arm 7, **R hand 7** | index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2 | `$ARENA/.../gr00t_43dof_joint_space.yaml`; HF Data `info.json` names [v] |
| Our `arm` op API (`body/joint_map.UPPER_BODY_MUJOCO_JOINTS`) | waist 3, L arm 7, R arm 7 (MuJoCo order) | Dex3: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1 | `body/joint_map.py` (body agent, in progress) |
| SONIC wire (17) | waist 3, then L/R **interleaved** | Dex3 | §1 |
| `g1_debug` | 29, MuJoCo order | Dex3 | §1 |

The GR00T client converts GR00T ↔ API with one name-keyed table, `groot/joint_order.py`. Only P3 converts API ↔
wire (`joint_map.wire_from_mj17`). The tests compare against the Arena yaml, `joint_map.py` and the HF Data names.

### 2.7 Can an Arena checkpoint initialize our fine-tune?

**N1.7 Static: yes.** Checked item by item:

| Item | Arena Static | Ours | Match |
|---|---|---|---|
| Model | Gr00tN1d7 from `GR00T-N1.7-3B` @ `2fc962b` (Cosmos-Reason2-2B, DiT 32 layers, horizon 40) | same base | yes |
| Tag / projector slot | `new_embodiment` / 10 | same | yes, so the trained slot-10 projector and action decoder carry over |
| Keys and layout | §2.2 | identical | yes |
| Joint semantics | Unitree `g1_29dof_with_hand_rev_1_0` (`g1.py:239-240`) | SONIC `main.urdf` + Dex3 from `g1_29dof_with_hand.urdf` | same Unitree joint names, zeros and signs. The Arena closed hand pose lies inside our Dex3 limits with the same signs (`joint_map.JOINT_LIMITS`) [v] |
| Hand | Dex3, 7 per hand | Dex3 | yes |
| Rate / horizon | 50 Hz / 40 | 50 / 40 | yes |
| Image | 480x640 ego (`exports/g1-static-apple-b1-480x640`) | 480x640 | yes; **mount and FOV differ unless OD1** |
| Lower body | AGILE, waist locked, PD arms | SONIC tracks the arms | arm dynamics differ. Our demos are recorded through SONIC, so fine-tuning learns SONIC's lag |
| Task | apple, shelf to plate, **left hand only** (right-hand action std 0.018 vs left 0.598) | pick from counters and dressers, both hands | partial |
| Training | 200 XR teleop demos (251 LeRobot episodes, 35,066 frames at 50 fps), 65k steps, batch 10, `tune_visual: true`, 1 GPU | — | — |
| Measured in its own scene | 8 of 37 = 22% (`arena_vs_sonic.md` §2.1) | — | a weak prior, not a solved skill |

- **What it buys:** a trained slot-10 projector and action head for exactly this G1+Dex3 layout, and a visual
  prior of G1 hands reaching for fruit. My estimate is that it cuts the demos needed per skill from about 150 to
  about 100, and reduces the steps needed [u]. G4 measures this directly: Arena-init against base-init on the same
  data.
- **How:** `--base-model-path <local Static ckpt> --embodiment-tag NEW_EMBODIMENT --modality-config-path groot/g1_arms_config.py`.
  [u] G4 checks in a dry run that the slot-10 weights are loaded, not reinitialized, and that the model config
  (32-layer DiT) comes from the checkpoint.
- **Licence (decision OD2).** The Static model card says "non-commercial use only". Its `LICENCE` file is the NVIDIA
  Open Model License Agreement. The base `GR00T-N1.7-3B` is "ready for commercial/non-commercial use" under the
  Open Model License. A fine-tune from Static inherits Static's terms. If the ambiguity matters, ship the
  base-init model.
- **N1.6 Loco-Manipulation: no.** It is Gr00tN1d6 (Eagle backbone, `max_state_dim` 29, horizon 50 in its
  `final_model_config.json`), so its weights cannot seed an N1.7. It is also non-commercial and replays one route.
- **Arena dataset (CC-BY-4.0, 200 human demos)** can be mixed in (mix ratio about 0.2-0.3) for motion variety,
  **only if OD1 matches the camera**. Its arms moved as PD targets on AGILE, not through SONIC. For slow reaches
  that difference is small [u].

---

## 3. Data: demos in our stack

### 3.1 Dataset schema: Arena's LeRobot v2.1, plus extras

Write datasets that Arena's `modality.json` reads unchanged (HF Data `lerobot/meta/info.json` [v]):

| Feature | Shape | Our value |
|---|---|---|
| `observation.images.ego_view` | video 480x640x3, h264, **fps 50** | latest P1 frame at each 50 Hz tick (held; P1 renders at 30 Hz) |
| `observation.state` | 43 f32, GR00T order | measured: legs and waist from `g1_debug.body_q`, hands from `left/right_hand_q` |
| `action` | 43 f32, GR00T order | **the targets P3 sent this tick** (arms and waist from the 17-vector, hands from the hand fields; legs = `g1_debug.last_action[0:12]`, unused) |
| `teleop.navigate_command` / `teleop.base_height_command` / `teleop.torso_orientation_rpy_command` | 3 / 1 / 3 | 0 / 0.75 / 0 |
| `observation.eef_pose`, `action.eef_pose` | 14 | palm poses in the pelvis frame from FK (not used by the model; kept for Mimic and for option C) |
| `annotation.human.task_description`, `task_index`, `episode_index`, `frame_index`, `index`, `timestamp`, `next.reward`, `next.done` | – | as Arena; `timestamp` = sim time since episode start |
| **extras** (ignored by `modality.json`) | | `observation.sonic_token` (64, `g1_debug.token_state`), `observation.base_pose_w` (7, ground truth), `observation.object_pose_w` (7, target object, ground truth), `observation.t_sim` |

**Recording rules:**

- One row per SONIC tick (50 Hz). The action is the **commanded** target, never the next measured state, so the
  policy learns commands that SONIC then tracks. The state is the measured value from the same tick.
- Frames: the newest P1 frame at the tick (JPEG q80 on 5565, decoded). The frame age is recorded; rows whose frame
  is older than 100 ms are flagged.
- Per-episode metadata goes to a sidecar, `meta/worldline_episodes.jsonl`, rather than into LeRobot's
  `episodes.jsonl` [u]: seed, house, surface, object id and type, arm, stance, `success`, `failure_reason`,
  `scripted_success_t`, `base_shift_m`, timing gate (irregular, RTF).
- Writer: `$WBC/gear_sonic/data/exporter.py` `Gr00tDataExporter` (LeRobot v2.1, the writer SONIC's own exporter
  uses), with the features above. The feature naming follows `$WBC/decoupled_wbc/data/utils.py:63-137`, which is
  the real G1 decoupled-WBC schema.
- `observation.sonic_token` records a SONIC-token label for every frame. The deploy's encoder runs in PLANNER mode
  on the reference that contains our override, which is how `run_data_exporter.py:663` fills `action.motion_token`.
  The same demos can therefore later train `UNITREE_G1_SONIC` [u: check that the tokens decode back to the motion].

### 3.2 Method A (primary): scripted IK/keyframe picks through the `arm` op, with ground-truth poses

`tools/collect_demos.py`, one process on the main box that talks only to the contract ports:

1. **Reset.**
   - P1 `reset_robot{x, y, yaw}` puts the robot at the sampled stance with the band on. Then body `stand`
     releases the band; this takes about 8 s [m] (`docs/M1.md` E1).
   - [u] The teleport happens while the deploy runs. SONIC's planner still holds the old facing, so P3 must re-align
     the IDLE facing to the new yaw before the band is released. If that cannot be done cleanly, reset by walking
     instead (`go_to` to the stance, about 15-25 s per episode).
   - P1 `set_object_pose{id, pose}` places the object (a **new P1 op**, §3.6).
   - Wait for the object to settle and sleep (1 s).
   - Arms start at SONIC's stand pose (`joint_map.DEFAULT_ANGLES`). The hands are opened over 0.3 s.
2. **Plan.**
   - Read the object pose from ground truth and transform it into the pelvis frame.
   - Pick a grasp from a per-type template: side power grasp; palm offset and approach axis per type; jitter of
     ±1 cm in position and ±10° in wrist yaw.
   - Waypoints: pre-grasp (8-10 cm back along the approach axis), grasp, close, lift (+8 cm), hold (1 s).
   - IK for the 7 arm joints, with the waist fixed at 0: Pink/pinocchio on `g1_29dof_with_hand.urdf`. The task and
     posture setup follows `$WBC/decoupled_wbc/control/teleop/solver/body/body_ik_solver.py:56-174` (FrameTask +
     PostureTask); a plain damped least-squares solver on pinocchio also works.
   - Reject infeasible or joint-limit-hugging solutions (margin 0.05 rad) and self-collision with the torso
     (capsule check).
3. **Execute.**
   - Minimum-jerk interpolation at 50 Hz between the waypoints. Total 4-7 s, speed scale 0.7-1.3.
   - Stream through the body `arm` op in chunk mode (§5.3), exactly as GR00T will.
   - Hand close: blend 0 → Arena's closed pose over 0.5 s at the grasp waypoint.
4. **Perturb (DART-style, p = 0.3).** On a random segment, the *executed* target gets a smooth 2-4 cm offset, while
   the *recorded* action stays the expert's corrective target. The data then contains recoveries without teaching
   the noise.
5. **Label from ground truth.**
   - `success` = PLAN §6.4 `gt_lifted`: the object is at least 5 cm above its support and within 12 cm of the arm's
     palm for 1.0 s, no fall, and |base shift| under 5 cm.
   - Otherwise `failure_reason` is one of `grasp_missed`, `object_dropped`, `ik_fail`, `fell`, `base_shift`,
     `timing_invalid`.
   - Only successes with a VALID timing gate (`docs/walk_diagnosis.md` rule 3) go into the training split. The
     failures are kept for analysis.

**Randomization per episode:**

- house/surface from the training houses {train-38 kitchen counters and dining table, train-59 kitchen,
  FloorPlan10 counters}, heights 0.75-1.13 m. train-40 (alarm clock, mugs) and train-15 (apple) are the eval
  houses (`eval/scenes.yaml` H40, H15) and stay **held out** (§4.3 set B). If set B fails while set A passes, add
  their *other* surfaces to training, never the eval surface itself;
- object pose on the surface: ±10 cm, and yaw uniform for symmetric types;
- stance: stand-off 0.35-0.50 m from the edge, lateral offset putting the object 5-25 cm to the arm's side, yaw
  ±10°;
- motion speed and the grasp jitter above;
- dome-light intensity ±30% [optional].

The same generator, run without recording, **is** the `sonic_arm_script` backend (PLAN #11). It is the stepping
stone, and it also gives an upper bound for evaluation.

**Throughput [u]:** reset about 10 s, motion 5-7 s, hold 1 s, bookkeeping 2 s, so about 20 s per episode. That is
about 180 attempts per hour. At an 80% scripted success rate, 150 demos take about 1 hour of wall time. The stack
runs at RTF 1, so there is no parallelism. Run it on a quiet box; the SONIC timing rule applies.

### 3.3 Method B (second): Mimic-style augmentation from few source demos

Once 10-20 source demos exist (scripted-with-noise or teleop), generate new ones the way Isaac Lab Mimic does:

1. Split each source into object-centric subtasks (reach+grasp, lift).
2. Transform the palm trajectory of the chosen segment into the new object frame
   (`$ILAB/source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py:57`
   `transform_source_data_segment_using_object_pose`; `:327` `generate_eef_subtask_trajectory`).
3. Interpolate from the current palm pose, solve IK, execute through `arm`, and keep the successes.

Port these functions, about 300 lines, **not** the framework. Mimic's `DataGenerator` drives a lockstep Isaac Lab
`ManagerBasedRLMimicEnv` whose action is an IK EEF target, and our body is a separate wall-clock process.

**`locomanipulation_sdg` is not adopted.** Its generator
(`$ILAB/scripts/imitation_learning/locomanipulation_sdg/generate_data.py:139-160`: grasp, lift, navigate, approach,
drop-off) adds path-following navigation with its own lower body velocity action
(`.../envs/g1_locomanipulation_sdg_env.py:177-226`). GR00T does not navigate in (b), and SONIC walks. The useful
part, replaying a recorded grasp relative to the object in a new scene, is exactly the Mimic transform above.

### 3.4 Method C (optional): teleoperation

| Route | Status here |
|---|---|
| SONIC's own VR teleop, `pico_manager_thread_server.py` (`PLANNER_VR_3PT` with hand IK) → our planner channel | Needs a PICO headset and a low-latency link to the cloud box. This is the only route that yields human-quality motion on SONIC. Only if the owner has the hardware |
| Isaac Lab XR teleop (CloudXR) | Needs Isaac Lab to drive the arms; our arms are driven by SONIC. Not applicable |
| Browser 3-D gizmo on the viz page → palm target → the same IK → `arm` | Feasible (viz already streams frames), but slow: about 1-2 min per demo. Good for 10-20 *source* demos for Method B, not for bulk |

### 3.5 How many demos, which objects first

- **Target per skill (type × arm):** 100 successes with the Arena init, 150 with the base init. The G4 sweep
  (50 / 100 / 150) decides the real number [u].
  - Reference points: the Arena Static checkpoint used 200 human demos and reaches 22% in its own scene. The
    SO100 example in `$GR00T` fits 5 demos only on training trajectories. Scripted demos are unimodal and easier
    to fit than human teleop, but brittle, hence the DART noise.
- **Objects,** from `eval/scenes.yaml` (dynamic props on reachable surfaces). Sizes are the AABB extents in the
  houses' `house_info.json` (`outputs/houses/*/`, `outputs/m1/house/*/`) [v]:

  | Order | Type | Instances (house: AABB extents, top height) | Why | Watch out |
  |---|---|---|---|---|
  | 1 | `apple` | train-38: 10.5×10.9×11.2 cm, 1.05 m; train-59: 13×13×15 cm; FloorPlan10: 15×16×12 cm; train-15 (eval #7): 13×13×12 cm | eval #7; the Arena init's object and hand | **THOR apples are large** (10-16 cm), probably near the limit of a Dex3 power grasp [u]. They roll when pushed, which is Arena's main failure. Approach from the side; the DART noise teaches recoveries. If G3's scripted success stays under 80%, the mug goes first |
  | 2 | `alarm_clock` | train-40 (eval #1-#3): 4.7×17×16 cm on the dresser, top 1.13 m; train-38: 16×10×18 cm, top 0.72 m | eval #1-#3; boxy; does not roll | Grasp across the 4.7 cm side. Near shoulder height (1.08 m) in train-40. Only two instances, of different shapes, so the hold-out rule is relaxed: train with train-38's clock moved onto counter-height surfaces **and** train-40's own clock on train-40's other surfaces; evaluate on the dresser only |
  | 3 | `mug` | train-38, train-59, FloorPlan10, train-15 (×2), train-40 (×2): 8-13 cm with the handle, tops 0.86-1.09 m | in every house; body about 8-9 cm | handle orientation |
  | – | `book` | – | – | **Excluded**: static prims lying at 0.53-0.57 m in the mattress, below the reach band (`eval/scenes.yaml` notes) |
  | later | `spatula` (eval #17) | FloorPlan10 | – | thin; needs a pinch grasp; not v1 |

  Every training house has an apple and a mug, so episodes move the existing instance (`set_object_pose`); no
  spawning op is needed for v1. A `spawn_object{asset}` op would only be needed for more instance variety.

- **Arm:** start with the **left** arm for the apple (matching the Arena init). Then add the right arm to the same
  model; the prompt names the arm. PLAN's `preferred_arm` picks by lateral offset, so a one-arm skill is usable
  through reposition.
- **Prompt:** `"pick up the {label} with the {arm} hand"` (PLAN §6.4), with 3-4 paraphrases in training. It stays
  fixed per session at run time.

### 3.6 Dependencies for data collection

| Needed | Owner | Status |
|---|---|---|
| Body `arm` op with chunk mode, hands, clamps, CarryLock (§5.3) | body agent | in progress (`body/joint_map.py` untracked) |
| SONIC upper-body tracking test passed (G0) | body agent | not run |
| P1 `get_object_poses{ids}` (live PhysX pose, 10 Hz ok) and `set_object_pose{id, pose}` (episode reset, logged like `root_writes`) | isaac agent | **missing**; P1 today has `reset_robot` only (`sim_isaac/app.py:588-737`) |
| P1 palm poses (`*_wrist_yaw_link` + a fixed palm offset; `merge_fixed_joints=True` may have merged `*_hand_palm_link` [u]) | isaac agent | add to `get_pose` or a `get_link_poses` op |
| P1 camera mount and intrinsics per OD1 | isaac agent | config change |
| Props awake and dynamic when touched (`sleep_house` re-validation, `docs/scenes.md` §11) | house agent | open |
| P1 Dex3 thumbs: `thumb_0`/`thumb_1` of both hands never follow their targets (P1 `q_target` moves, the joint does not; they get pushed to other positions and joint limits and stay there), and the right hand stops at 0.72 closure. Arena's closed pose needs `thumb_1` = 0.7 (`docs/arm_tracking.md` §3.7; re-checked 2026-09-29, `outputs/verify_arm/`) | isaac agent | **missing** |

---

## 4. Fine-tune

### 4.1 Command (from `$GR00T/getting_started/finetune_new_embodiment.md`)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path <Static ckpt dir | nvidia/GR00T-N1.7-3B> \
  --dataset-path /work/groot/data/g1_pick_<type>_v1 \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path groot/g1_arms_config.py \
  --num-gpus 1 --global-batch-size 32 --max-steps 15000 --save-steps 2500 --save-total-limit 6 \
  --learning-rate 1e-4 --state-dropout-prob 0.2 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --output-dir /work/groot/runs/<name>
```

The defaults keep the VLM frozen and tune the projector, DiT and VL layer norm (`finetune_config.py:49-58`). Arena
used `tune_visual: true`, which the docs put at 80 GB+ per GPU (`$GR00T/getting_started/hardware_recommendation.md:54`).
Leave it off on an L40S.

### 4.2 Compute

- **VRAM:** the default fine-tune stays under about 35 GB per GPU (`hardware_recommendation.md:53`), so one L40S
  (48 GB) is enough **with Isaac off**. It does not fit next to the live stack (PLAN §3.4).
- **Where:** the dev box `ludo-g1-arena` (L40S), once the other agent's installation is finished, or the main box
  while the stack is down. The dataset is small: 150 demos × about 300 frames = 45k rows plus 150 short mp4s, well
  under 1 GB.
- **Time [u]:** no official throughput figure exists in `$GR00T`.
  - My estimate is 1-2 s per step at batch 32 on an L40S, i.e. 4-8 h for 15k steps. Measure the first 200 steps
    and re-plan.
  - 15k × 32 = 480k samples ≈ 10 epochs of 45k frames, comparable to Arena Static's 650k samples on 35k frames.
- **Runs (G4):** (1) Arena-init at 150 demos; (2) base-init at 150; (3)-(4) the better init at 50 and 100. That is
  about 1.5-2 GPU-days in total.

### 4.3 Evaluation protocol

1. **Open loop (a cheap gate).** `gr00t/eval/open_loop_eval.py --embodiment-tag NEW_EMBODIMENT --execution-horizon 20`
   on 10 held-out demos, every checkpoint. MSE must fall across checkpoints and the predicted curves must follow
   ground truth (`finetune_new_embodiment.md`, "Is my fine-tune working?").
2. **Closed loop in Isaac** (the real metric). The product path is `ManipulationService` → GR00T executor → `arm`
   (§5), on the live stack:
   - 20 trials per (type, arm) on fixed seeds that were not used in training. Stance and object pose are drawn from
     the training distribution.
   - Set A: train-38 kitchen (in distribution). Set B: the held-out eval houses (train-40 dresser for the alarm
     clock; train-15 `kitchen_counter_1b` for the apple).
   - Per trial: reset as in §3.2; start the skill; stop on ground-truth success or 25 s. Metrics: success,
     time-to-success, failure reason, falls, base shift, inference latency p50/p95, chunks dropped, clamp count,
     SONIC timing gate.
   - Trials with an INVALID timing gate are re-run, not counted (`docs/walk_diagnosis.md`).
   - Report Wilson 95% intervals (20 trials give about ±20 points). Report next to them the scripted generator's
     success on the same seeds, which is the ceiling, and the Arena Static zero-shot result, which is the floor.
   - Record videos with `viz/recorder.py --until-event` (`docs/viz.md`).
   - Run a first-episode vs later-episode split, as a check against the Arena harness effect (`arena_vs_sonic.md`
     §2.4).
3. **Product scenarios:** eval #7 (apple) and #1 (alarm clock) on the `full` profile with `executor=groot`
   (PLAN M6 exit).

**Pass bar (PLAN M6):** at least 50% over 20 trials on at least one type, and eval #1 or #7 passes with GR00T and
no fallback.

---

## 5. Inference integration

### 5.1 Components

```
P4  PolicyServer :5550 (Isaac-GR00T venv, py3.12; REP, msgpack+numpy; run_gr00t_server.py --port 5550)
      ▲ get_action(obs)  ~150 ms                       (5555 is DCGM on the box: m1.md §0)
      │
P5  ManipulationService ── GrootArmExecutor (ManipExecutor "groot")
      │   lease + session fence; GT success/failure at 10 Hz via WorldModel
      └── GrootArmClient thread: SUB 5565 (ego frames) + SUB 5557 (g1_debug) → obs → REQ 5550
                                  → chunk (40 × [mj17 + 7 + 7], t0) → DEALER 5610 op "arm" (stream)
P3  wl-body: arm stream (chunk mode) → 50 Hz: index by time, cross-fade, clamp → SonicMux planner IDLE
                                          + upper_body_position (wire order) + left/right_hand_joints → P2 (5556)
```

- **The client lives in P5, in its own thread.** It has the execution context and the ground-truth world model,
  and the runtime venv only needs pyzmq, msgpack and numpy. The client is a vendored copy of `PolicyClient` with
  its no-pickle `MsgSerializer` (`$GR00T/gr00t/policy/server_client.py:22-57, 145-178`).
- **P3 keeps the 50 Hz timing** and stays policy-agnostic: it only knows joint targets, the same way P1 is
  controller-agnostic. P5's asyncio and LLM jitter therefore only delays chunk arrival, and the time-based index
  absorbs that.
- **PolicyServer placement (decision OD3).** The main box has room: about 15 GB per checkpoint [m, dev box]. But
  GR00T's GPU bursts compete with Isaac rendering and the deploy's TensorRT, and contention is the first-ranked
  cause of SONIC falls (`docs/walk_diagnosis.md`).
  - G2 measures the timing gate with the server running.
  - If the gate turns INVALID: TensorRT (L40: 26 Hz vs 7.8 Hz eager, `hardware_recommendation.md` table) or a
    remote server on the dev box.
  - A warm-up `get_action` runs at skill load; the first call takes 3.8 s [m, `arena_vs_sonic.md` §2.5].

### 5.2 The GR00T client loop (per session)

Adapted from `$WBC/gear_sonic/scripts/run_vla_inference.py` (async worker `:600-651`, chunk indexing `:705-741`) and
`gear_sonic/utils/inference/vla_utils.py:65-106`. The difference: indexing moves to P3, keyed on time.

1. **Observation.**
   - The newest ego frame (RGB 480x640) and the newest `g1_debug`.
   - Freshness: frame at most 100 ms old, state at most 40 ms old, and the two at most 60 ms apart. Otherwise count
     `obs_stale` and retry at the next tick.
   - `t_obs` = the monotonic receive time of that `g1_debug`. Index 0 of the chunk corresponds to `t_obs`, as in
     training, where row t pairs the state and the latest frame at tick t with the action sent at tick t.
   - `obs = {video: {ego_view: (1,1,480,640,3)}, state: {left_arm…waist: (1,1,D)}, language: {annotation.human.task_description: [[prompt]]}}`
     (`$GR00T/getting_started/policy.md`).
2. **Trigger.** Start a new inference when none is in flight and either 0.4 s have passed since the last chunk
   started (upstream's 2.5 Hz, `run_vla_inference.py:78`) or fewer than 12 steps of the current chunk remain.
3. **Call.** REQ 5550 with a 1.5 s timeout. Measured latency is 148-153 ms mean, 161-165 ms p95 (N1.7, dev box, no
   sim) [m]; on the main box [u].
4. **Validate.** Session still active; no NaN; shapes (40×7 per key). Drop the result if
   `now − t_obs > 0.8 s − 0.1 s` (`expired`).
5. **Post** `arm{stream, session, epoch, chunk}` to P3:
   - arms and waist as mj17 = [waist held] + L arm + R arm;
   - hands reordered to Dex3 order;
   - `navigate_command` / `base_height_command` / GR00T's waist go only to telemetry (`groot_nav_pred`).

### 5.3 What the `arm` op must do for GR00T (request to the body agent)

Build this on the op being written, with the same streaming pattern as `velocity` (`docs/contracts/m1.md` §3.4):

| Field / behaviour | Spec |
|---|---|
| Start | `arm{stream, lease/execution_id, generation, control_epoch, hold_on_end: "target"\|"measured"\|"stand"}`. Pre-empts only a non-motion state. The harness never runs `manipulate` concurrently with `navigate` |
| Single target (scripts) | `upper_body` (mj17, API order) + `left_hand`, `right_hand` (Dex3 order); optional `blend_s` |
| **Chunk (GR00T, the generator)** | `chunk{t0_mono, dt: 0.02, upper_body: T×17, left_hand: T×7, right_hand: T×7, session}`. Monotonic clocks are shared on one Linux box |
| 50 Hz tick | `k = round((now − t0_mono)/dt)`. If k ≥ T: hold the last row and add to `stall_s`. On a new chunk, cross-fade from the old chunk's value at the same time to the new one over 5 ticks (100 ms). GR00T's RTC is "only a low-level model primitive… not wired into `Gr00tPolicy` or the server-client path" (`$GR00T/getting_started/real_world_deployment.md:413`) |
| Safety | Clamp to URDF limits with a 0.02 rad margin (`joint_map.clamp_mj17`, `clamp_hand`). Arm rate limit 0.06 rad per tick (3 rad/s) [u, tuned in G0]. Reject a chunk containing NaN. Count clamps; report `clamped_frac` in progress events |
| Every planner message | IDLE (`movement` 0, facing held) + `upper_body_position` (via `wire_from_mj17`) + **both** hands. Optional `upper_body_velocity` from finite differences, if G0 shows it helps |
| End | `end: true` → `hold_on_end`. `target`: CarryLock keeps the last target on every later planner message, including SLOW_WALK. `measured`: hold the measured pose (cancel without an object). `stand`: blend to the stand pose over 1.5 s, then drop the override (§1: never drop it abruptly) |
| Watchdog | No chunk and no message for 1.0 s → hold the last target, end `ended_by: watchdog` |
| Halt (5612) | Latch at once: drop the stream, hold the **measured** upper body, **never open the hands**, reject stale epochs (PLAN §5.6) |
| Events | `arm.progress{session, chunk_idx, k, stall_s, clamped_frac, latency_ms, cross_fades}` at 5 Hz; one terminal event |

### 5.4 Ownership, fencing, cancellation

- **Lease.** `GrootArmExecutor.run(job, handle)`:
  1. `BodyClient.acquire(execution, mode=ARM_STREAM)` (a new body mode that replaces PLAN's `VLA_TOKENS`), fenced
     by execution id, generation and control epoch (PLAN §6.6).
  2. The stance check (PLAN §6.4 step 1, no approach phase) and the view check (object visible in the GR00T camera
     segmentation, at least 200 px [u: needs P1 instance ids on that camera]).
  3. Open the hands over 0.3 s.
  4. Start the client.
- **Session fence.** `session = execution_id`. The prompt is fixed per session (PLAN §7.4 "prompt immutability").
  Results from an old session are dropped in the client (`stale_session`), and chunks from an old session are
  dropped in P3.
- **Cancel** (user correction, `ManipulationService.cancel`):
  1. Stop triggering new inferences; mark the session cancelled.
  2. Send `arm{end, hold_on_end}`: `target` if ground truth says the object is in hand (CarryLock), otherwise
     `measured`, followed by a scripted retract to stand.
  3. Result `cancelled` within the 3.0 s body grace (PLAN #22).
- **Halt** (stop word, safety): P3 latches first (§5.3); the executor sees the stream end and returns `failed(halted)`.

### 5.5 Success and failure from ground truth (`WorldModel` in P5, 10 Hz)

| Outcome | Predicate | Action |
|---|---|---|
| `succeeded` | `gt_lifted` (§3.2) for 1.0 s | `arm{end, hold_on_end:"target"}` → CarryLock engaged → `ManipulationResult{holding: object_id, executor: "groot", inferences, chunks_dropped, clamped_frac, base_shift_m}` |
| `failed(grasp_missed)` | hand closure over 0.6 (`joint_map.hand_closure_of` on the measured hand) and the object more than 12 cm from the palm for 1.5 s | end with `measured`, then retract |
| `failed(object_dropped)` | lifted earlier, now below support + 2 cm or more than 20 cm from the palm | same |
| `failed(policy_stall)` | `stall_s` > 5 s (event at 2 s) | same |
| `failed(policy_unavailable)` | 2 consecutive REQ errors or timeouts | same; registry health marks the skill (PLAN #27: the enum stays frozen) |
| `failed(policy_out_of_range)` | `clamped_frac` > 0.2 over 1 s [u] | same |
| `failed(fell)` | P3 fault | P3 has already gone to FAULT |
| `timed_out` | `max_duration_s` = 25 s | end with `measured` |

`base_shift_m` is reported on every result. More than 5 cm invalidates reachability (PLAN #25 kept only as a
check; GR00T no longer moves the legs).

### 5.6 Hand-over to the rest of the body

- **Carry.** CarryLock holds the last GR00T target through `navigate`. SONIC walks with the upper body frozen, which
  is upstream's `PLANNER_FROZEN_UPPER_BODY` (§1).
- **Place.** It stays the scripted `sonic.script.place.v0` through the same `arm` op, until a GR00T place skill
  exists. After the release, blend to stand, then drop the override.
- **Registry entry** (`config/skills.yaml`, M4 owners):
  - `groot.pick.<type>.isaac.v1`: backend `groot`, label `target`, `policy_port 5550`;
  - `embodiment: new_embodiment/g1_arms_dex3`, `hand_type: dex3`, `action_horizon 40`, `control_hz 50`,
    `replan_hz 2.5`, `max_duration_s 25`;
  - `stance` = the §3.2 sampling ranges, `success: gt_lifted`.
  - `SkillSpec` fields that only apply to SONIC tokens (`initial_token`, `token_abs_limit`) become optional.
- **PLAN changes this implies** (not edited here):
  - §1.1 and #25: GR00T drives arms and hands only. Doc §15 is met, and PLAN's deviation D1 (GR00T tokens move
    the legs) is gone.
  - #11 and #21: the target is `groot.pick.*.isaac.v1` on this action space.
  - §6.4 pick phases: the `enter` token blend is replaced by opening the hands.
  - §7.4: the VlaStreamer token path is replaced by the `arm` chunk stream.
  - M6.1-M6.2: the embodiment is `NEW_EMBODIMENT` (Arena schema), not `UNITREE_G1_SONIC`.

---

## 6. Milestones, effort, risks

### 6.1 Milestones (engineer-days are my estimates)

| # | Milestone | Effort | Exit |
|---|---|---|---|
| G0 | *(body agent, prerequisite)* `arm` op + **SONIC upper-body tracking test**. Stream arm and hand trajectories from HF Data (CC-BY, 50 Hz) through the override while standing; A/B with velocities | 1-1.5 | Wrist error under about 2 cm, lag under about 60 ms, 0 falls (`arena_vs_sonic.md` §6 step 1). **If it fails, stop: the design is void and the owner re-decides** |
| G1 | Schema and plumbing: `groot/g1_arms_config.py`, `groot/joint_order.py` + tests, the recorder (`tools/record_demos.py`, Arena schema + extras); camera render check for OD1 | 2 | 3 recorded demos load in `$GR00T`'s loader (`open_loop_eval.py` runs against them); replaying their actions through `arm` reproduces the recorded state within the G0 error; OD1 decided |
| G2 | Inference plumbing with the Arena Static checkpoint zero-shot: PolicyServer 5550, `GrootArmClient`, chunk mode in `arm`, `GrootArmExecutor` v0 with fencing, cancel, halt and ground-truth outcomes | 2 | 10 sessions, 0 falls, latency p50/p95 known, cancel 3/3 and halt 3/3 without an arm jerk; timing gate VALID with the server on the main box (or OD3 moves it). Success is not expected |
| G3 | Data: P1 object pose and reset ops *(isaac agent, about 1 day)*, IK, grasp templates, randomization, DART, ground-truth labels, auto-reset loop | 3-4 (+ about 1 h wall per 150 demos) | Scripted success at least 80% on the apple (left); 150 successes recorded; dataset card with camera parameters |
| G4 | Fine-tune on the dev box: 4 runs (§4.2) | 1 + 1.5-2 GPU-days | Open-loop MSE falls; best init and data size picked on 20 closed-loop trials |
| G5 | Closed loop + product path: registry entry, eval sets A and B, eval #7 on `full` | 2 | **At least 50% over 20 trials** on the apple; eval #7 passes with `executor=groot` (PLAN M6 exit) |
| G6 | Second and third type (alarm clock, mug), right arm, Mimic augmentation, optional TensorRT | 3-5 | 2 or more types at 50% or better; eval #1 passes |
| **Total** | | **14-17.5 ED, + about 1 ED of P1 ops, + 1.5-2 GPU-days** | PLAN M6 budgeted 9.5-14.5 ED for the token route |

G1 can start now, in parallel with G0. G2 needs G0's `arm` op. G3 needs the P1 ops.

### 6.2 Risks

| # | Risk | Likelihood / impact | Mitigation / check |
|---|---|---|---|
| R1 | SONIC tracks a moving step reference with lag or overshoot (the future frames are constant, velocity zero, §1) | medium / high | G0 measures it. A/B with `upper_body_velocity`. Demos are recorded through SONIC, so the policy learns the lag |
| R2 | Reaching shifts the centre of mass; SONIC steps or falls | low-medium / high | Keep the workspace limits (`config/g1.yaml` reach band); speed scale; `base_shift_m` and fall gates on every demo and trial |
| R3 | Dex3 grasps slip in PhysX (kp 1.5, friction, prop mass) | medium / high | Measure in G3 first. Tune P1 Dex3 gains or friction inside real-hardware bounds [u]. **No weld assist in training data**; if one is ever needed it is labelled STEPPING STONE |
| R4 | GR00T GPU bursts on the main box break SONIC timing (the walk-diagnosis first cause) | medium / high | Timing gate in G2; TensorRT; remote server (OD3); fine-tuning never runs on the live box |
| R5 | Scripted demos are too clean; the policy cannot recover | high / medium | DART noise; randomization; Mimic augmentation; optional teleop source demos; closed-loop set B |
| R6 | The d435 view misses counter-height objects | high if OD1 = no / high | OD1 (Arena camera); render check in G1 |
| R7 | Licence: Arena Static card says non-commercial | – / medium | OD2: the base-init run exists anyway; ship it if needed |
| R8 | Data throughput: wall clock, one environment at RTF 1 | certain / low | About 1 h per 150 demos is enough; run overnight batches on a quiet box |
| R9 | Missing P1 ops (object pose and reset, palm link) | certain / medium | Requested in §3.6; about 1 day for the isaac agent |
| R10 | Frame staleness: 30 Hz render, 50 Hz rows, JPEG plus mp4 compression | low / low | The same hold-last-frame at train and run time; frame age logged; a 50 Hz render is an option if RTF allows |
| R11 | Apple rolls when touched (Arena's main failure) | medium / medium | Side approach; DART; alarm clock as the second type |
| R12 | Props asleep or kinematic under contact (`scenes.md` §11) | medium / medium | Check in G3 before collecting |

### 6.3 Owner decisions needed

| # | Decision | Recommendation |
|---|---|---|
| OD1 | GR00T camera: Arena head camera (35° down, HFOV 70°) or the current d435 mount | **Arena camera** for GR00T, after the G1 render check; the d435 feed stays for the UI |
| OD2 | Init: Arena Static (licence ambiguous) or base N1.7-3B | Run both in G4; ship the better one if the licence allows, otherwise base |
| OD3 | PolicyServer on the main box or the dev box | Main box if the timing gate stays VALID in G2 |
| OD4 | First skill | Apple, left hand; the mug instead if THOR's 10-16 cm apples defeat the scripted Dex3 grasp in G3. Then the alarm clock (eval #1) |

### 6.4 Unverified items and where they are checked

- Arena's `rev_1_0` USD `head_link` equals the URDF offset → G1 render check.
- Arena-init loads the slot-10 weights and the 32-layer DiT from the checkpoint → G4 dry run.
- Demos needed per skill (100 vs 150) and fine-tune time per step → G4.
- Tokens recorded in PLANNER mode decode back to the motion → a later check, only needed for the token route.
- Main-box GR00T latency and timing-gate impact → G2.
- Dex3 grasp holding in PhysX; prop dynamics under contact; alarm clock size vs Dex3 → G3.
- P1 palm link after `merge_fixed_joints=True` → G3 (isaac agent).
- A sidecar `meta/worldline_episodes.jsonl` does not disturb GR00T's LeRobot loader → G1.
- Teleport reset while the deploy runs keeps SONIC's facing consistent → G3 (with the body agent).
