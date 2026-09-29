# Arena spike: official G1 GR00T checkpoints in Isaac Lab Arena (box `ludo-g1-arena`)

The question: should Worldline-on-G1 adopt the Isaac Lab Arena G1 controller path (a learned lower body
driven by `navigate_command` + `base_height_command`, with GR00T driving arms and hands) alongside SONIC or
instead of it? What follows is what ran, the numbers, and what executes the navigation command.

## Stack that ran (documented path: Arena base container + GR00T server on the host)

| piece | version |
|---|---|
| IsaacLab-Arena | `release/0.2.1` @ `8b4a3a47` (the only branch that still ships `galileo_g1_static_pick_and_place`, which the N1.7 checkpoint needs; `main` removed it) |
| Isaac Sim | container `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` (reports `6.0.0-rc.22`), public, no NGC login needed |
| Isaac Lab | Arena submodule @ `e57379c6` (Isaac Lab 3.0 dev, `isaaclab` ext 4.5.24) |
| N1.6 server (loco-manip ckpt, `--revision gn1_6`) | Isaac-GR00T @ `e29d8fc5` (the Arena submodule pin), py3.10, torch 2.7.1+cu126, transformers 4.51.3, flash-attn 2.7.4.post1 |
| N1.7 server (static apple ckpt) | Isaac-GR00T @ `4b1dca9d` (the static_apple docs pin), py3.10, torch 2.7.1+cu128, transformers 4.57.3; backbone `nvidia/Cosmos-Reason2-2B` (downloaded with our token) |
| Arena client | stock `Gr00tRemoteClosedloopPolicy` (GR00T `PolicyClient` from the e29d8fc submodule) wrapped only for timing and video |

Main track for comparison: `/work/repos/IsaacLab` is v2.3.2 on Isaac Sim 5.1. Arena 0.2.x needs Isaac Sim 6 and
Isaac Lab 3. A `feature/arena_v0.2_on_lab_2.3` branch and a matching dataset branch exist, but I did not test them.

## How to run (on the box; every script is in `/work/arena/spike` = this dir)

```
bash setup_box.sh            # clone Arena 0.2.1 + submodules, gr00t_n17 clone, pull isaac-sim:6.0.0-dev2, build isaaclab_arena:latest
bash download_models.sh      # both GN1x ckpts (no optimizer states), Cosmos-Reason2-2B, N1.7 base metadata
bash setup_gr00t_envs.sh     # uv envs gr00t_n16 (e29d8fc) and gr00t_n17 (4b1dca9); CUDA_HOME=/usr/local/cuda for flash-attn
tmux new -d -s srv_n16 'bash start_server.sh n16'   # port 6555 (5555 belongs to nv-hostengine/DCGM on Nebius)
tmux new -d -s srv_n17 'bash start_server.sh n17'   # port 6556
bash start_container.sh      # detached Arena container 'arena' (same flags as docker/run_docker.sh)
bash run_eval.sh n16 n16_eval --num_episodes 10                 # loco-manip, g1_wbc_joint (HOMIE v2)
SPIKE_RERENDER=1 bash run_eval.sh n16 n16_eval_rerender --num_episodes 10
DEVICE=cpu bash run_eval.sh n16 n16_eval_cpu --num_episodes 6
bash run_eval.sh n17 n17_eval --num_episodes 10                 # static apple, g1_wbc_agile_joint (AGILE)
VIDEO=0 bash run_eval.sh n17 n17_novid --num_episodes 20
bash run_veldrive.sh homie ; bash run_veldrive.sh agile         # WBC with pure velocity commands, NO GR00T
```

Outputs are in `/work/arena/eval` and pulled to `worldline-g1/outputs/arena/`. Each run has `<tag>_episodes.jsonl`
(per episode, from the env's own TerminationManager), `<tag>_timing.json`, `<tag>_cmd_log.jsonl` (the nav and height
commands actually sent to the WBC), and `<tag>_vram.log`. Videos are in `outputs/arena/<tag>/epNN_{ego,3p0,3p1}.mp4`:
`ego` is `robot_head_cam_rgb`, the image GR00T sees; `3p0` is a side view; `3p1` is top-down.

Gotchas we hit:
- The stock `--video` records `/OmniverseKit_Persp`, which renders black when headless on this build. So
  `arena_spike_policies._Recorder` adds its own camera prims plus replicator render products.
- `docker exec -u uid:gid` drops the `isaac-sim` group, and then `SimulationApp` is `None`. Use `-u <name>`.
- Kit exits with `os._exit`. Stdout needs `PYTHONUNBUFFERED=1`, and `atexit` never runs.
- The static-apple YAML `model_path` must be edited, as the docs say. The copy used here is `g1_static_apple_gr00t_closedloop_config.yaml`.

## Results (single env, L40S, success counted by the env's own terminations)

| checkpoint / task | run | episodes | success | failure modes |
|---|---|---|---|---|
| GN1x Loco-Manip **N1.6**, `galileo_g1_locomanip_pick_and_place`, `g1_wbc_joint` | GPU physics (docs default) | 10 | **1** | 8 time-outs (30 s), 1 box dropped |
| same | GPU + rerender-on-reset fix | 10 | **2** | 8 time-outs |
| same | CPU physics (docs note) | 6 | **1** | 5 time-outs |
| same | smoke (first episode of a process) | 1 | 1 | |
| GN1x Static PnP **N1.7**, `galileo_g1_static_pick_and_place`, `g1_wbc_agile_joint` | with video | 10 | **3** | 7 time-outs (6 s), object_moved_rate 0.5 |
| same | no video | 20 | **4** | 16 time-outs, object_moved_rate 0.65 |

Totals: N1.6 got 5/27 (19%) and N1.7 got 7/30 (23%). The docs show `success_rate 1.0` on a 1-episode smoke
run, and our first episodes match that. Over many episodes, neither checkpoint reaches the docs' implied quality on this stack.

What the robot does (checked on extracted frames):
- **N1.6 walks.** In every episode that did not drop the box, GR00T turns the robot about 100 degrees right in
  place, walks 2.4 to 4.4 m of root path, and squats (`base_height_command` goes down to 0.60). Its `navigate_command` ranges over vx 0 to 0.40,
  vy -0.06 to 0.11 and wz -0.39 to 0.09, and is nonzero in 50% of steps. In successes it grasps the box with both
  hands, carries it to the table and lowers it into the blue bin in 14.7 to 20.0 s.
- **N1.6's dominant failure is a missed grasp.** The hands close in front of the box, the box stays on the shelf,
  and the robot walks to the bin anyway and mimes the place until the 30 s time-out.
- **N1.7** stays in place (AGILE balancing). In successes (2.4 to 5.2 s) the right hand grasps the apple and sets it
  on the plate. In failures the hand pushes the apple around (it rolls) until the 6 s time-out. From its second
  episode on, the checkpoint also emits `base_height_command` of 1.0 and a yaw rate of about -0.29 rad/s. Both are the
  q99/q01 extremes of its own training statistics; they cause a yaw drift of 4 to 8 degrees per episode.
- **Harness bug found:** the loco-manip env (0.2.1 and `main`) does not set `num_rerenders_on_reset`. After an
  auto-reset, the first GR00T chunk (1 s) of every episode is computed from the previous episode's final frame.
  `outputs/arena/*/ep01_ego.mp4` frame 0 shows the robot still at the bin. The static env sets it to 1 for exactly this
  reason. Fixing it at runtime (`SPIKE_RERENDER=1`) moved 1/10 to 2/10, which is not the main cause of failure.
- The WBC action term's `reset()` does not reset the HOMIE/AGILE policy state (obs history, gait phase, LSTM).
  I noted this but did not isolate its effect.

Performance:
- GR00T round trip, client side, 640x480 image over ZMQ on localhost: N1.6 has a mean of 133 ms and p95 of 147 ms
  per 50-step chunk. N1.7 has a mean of 152 ms and p95 of 164 ms per 40-step chunk; its first call takes 3.8 s.
  That is one call per 0.8 to 1.0 s of sim.
- VRAM: N1.6 server 7.2 GB, N1.7 server 6.5 to 6.7 GB, Isaac Sim client 7.5 to 8.1 GB (5.4 GB with CPU physics).
  One checkpoint needs about 15 GB in total; both servers plus the sim use 21.8 GB.
- Sim speed (single env, 640x480 head camera): RTF 0.23 to 0.26 without our video recorder and 0.19 to 0.20 with
  it, at 78 to 105 ms per 50 Hz step. The first launch compiled shaders in 407 s; later launches take 8 s.

## What executes navigate_command / base_height / torso for the G1 in Arena

- **Action term:** `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py::G1DecoupledWBCJointAction`,
  the same on 0.2.1 and `main`. The action is 50-D: 43 joint targets in sim order, then `navigate_cmd[3]`
  (vx, vy, wz in the body frame), then `base_height[1]`, then `torso_rpy[3]`. `process_actions()` runs once per env step:
  physics is 200 Hz with decimation 4, so the WBC runs at **50 Hz**. `apply_actions()` sends the same joint position
  targets to PhysX PD at 200 Hz.
- **Composition:** `wbc_policy_factory.get_wbc_policy` returns `G1DecoupledWholeBodyPolicy`. Its upper body is
  `IdentityPolicy`, which passes the arm, hand and waist targets from GR00T straight through. Its lower body is one of:
  - **HOMIE v2**, `policy/g1_homie_policy.py`, used by `g1_wbc_joint` and `g1_wbc_pink`, which is what the N1.6
    loco-manip checkpoint uses. It is two ONNX files, `stand.onnx` and `walk.onnx`, fetched from
    `{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/models/homie_v2/`, and switches between them on |navigate_cmd| < 0.05.
    The observation is 86 values x 6 frames of history: cmd scaled by [2, 2, 0.5], height, torso rpy, ang vel,
    gravity, 29 q, 29 dq, last action, plus a 0.75 Hz gait clock. It outputs 15 targets (12 legs + 3 waist) as
    0.25 * a + default. Config: `config/g1_homie_v2.yaml`. The norm is taken over all envs, so with more than one env,
    one walking env switches every env to `walk.onnx`.
  - **WBC-AGILE**, `policy/g1_agile_policy.py`, used by `g1_wbc_agile_joint` and `g1_wbc_agile_pink`, which is what
    the N1.7 static checkpoint uses. It is a recurrent LSTM student (hidden size 256), downloaded from
    `github.com/nvidia-isaac/WBC-AGILE@7259792c/.../unitree_g1_velocity_height_recurrent_student.onnx` (Apache-2.0).
    The observation is [vx, vy, wz, h], ang vel, projected gravity, 29 q_rel, 29 dq*0.1, and the last 12 actions.
    It outputs 12 leg targets, clipped to +-6, scaled and offset. It ignores torso rpy and does not drive the waist.
    It needs the `G1_AGILE_CFG` actuator gains, which are 2 to 4 times softer than Arena's default G1.
  - Pose-goal navigation, used only for Mimic data generation: `utils/p_controller.py` (`use_p_control`,
    `navigation_subgoals`, max 0.4 m/s) turns (x, y, heading) subgoals into `navigate_cmd`.
- **Driving it without GR00T works.** `arena_spike_policies.VelocityCommandPolicy` builds the 50-D action from a
  scripted schedule and holds the upper body at its defaults. It ran in the same Galileo scene with no GR00T server
  involved (see `outputs/arena/veldrive_{homie,agile}/`):

  | command | HOMIE v2 | AGILE |
  |---|---|---|
  | vx -0.3 m/s for 2 s | 0.50 m back | 0.61 m back |
  | wz -0.5 rad/s for 3.5 s (-100 deg) | -98 deg, overshoots to -116 | -97 deg, no overshoot |
  | vx +0.4 m/s for 4 s | about 0.9 m (about 0.25 m/s) | about 1.25 m (about 0.35 m/s) |
  | vy +0.2 m/s for 2 s | 0.19 m | 0.30 m |
  | height 0.75 to 0.60 | pelvis drops 0.15 m and recovers | drops 0.12 m and recovers |

  Neither run fell. AGILE tracks velocity better. Both are small CPU ONNX policies, so they can be lifted out of Arena
  into our own Isaac Lab env as a navigation controller: AGILE is an ONNX file, a ~290-line observation adapter
  (`g1_agile_policy.py`), and the gain config.

## Licences
- GN1x Loco-Manip (N1.6): NVIDIA One-Way **Non-commercial** licence.
- GN1x Static PnP (N1.7): the card says "non-commercial use only" and "NVIDIA Open Model Agreement", but the
  `LICENCE` file in the repo is the NVIDIA **Open Model License**, which says it is commercially usable. These
  contradict each other; ask NVIDIA before any commercial use.
- Datasets: Static PnP is CC-BY-4.0. Cosmos-Reason2-2B is NVIDIA Open Model License, gated (our token already has access).
- Arena code: Apache-2.0. WBC-AGILE: Apache-2.0 (BSD-3 for the rsl_rl patch). HOMIE v2 ONNX comes from NVIDIA's
  Isaac Lab Nucleus assets.
- Nothing needed an NGC login. No gated model had to be accepted beyond Cosmos-Reason2-2B.

## Verification pass (2026-09-29): see `docs/arena_vs_sonic.md`

- **Re-runs and probes:**
  - `verify_n16`, `verify_n17`: re-runs.
  - `probe_n16_nobin`: the bin is held under the floor, and N1.6 still walks its one trained route to the empty table.
  - `probe_n16_wbcreset`: a control run.
  - `probe_n16_seed1..7`, `probe_n17_seed1..4`: one episode per fresh process.
- **Code:** `arena_probe_policies.py` (`BinProbeGr00tPolicy`), selected with `POLICY_TYPE=...` in `run_eval.sh`.
  `arena_exec.sh` passes `SPIKE_BIN_POSE` and `SPIKE_WBC_RESET` through.
- **Corrections to the sections above:**
  1. N1.7 grasps with the **left** hand, not the right.
  2. The WBC state **is** reset between episodes, by the env's `reset_wbc_policy` event. The action term's `reset()`
     does not do it, but that is not a bug.
  3. N1.6 succeeds **10 of 12 times from a clean start** (episode 0 of a process) but only 1 of 24 times on later
     episodes in the same process. The per-run rates above are dominated by this. The cause is not isolated, so
     evaluate one episode per process.
