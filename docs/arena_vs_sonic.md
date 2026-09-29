# Arena controller path vs SONIC: evidence and recommendation

Status: **independent verification of the Arena spike, plus two new probes, 2026-09-29.** Box `ludo-g1-arena`
(Nebius L40S, Isaac Sim 6.0.0-dev2 container, IsaacLab-Arena `release/0.2.1` @ `8b4a3a47`). Spike code:
`arena_spike/` (recipe in `arena_spike/README.md`). Evidence: `outputs/arena/` (videos per run folder, JSON and logs
in `outputs/arena/logs/`); box copy in `/work/arena/eval/`.

The question: should Worldline-on-G1 adopt the Isaac Lab Arena G1 controller path (a learned lower body commanded by
`navigate_command` + `base_height`, with GR00T driving arms and hands) alongside or instead of SONIC? And, as the owner
asked: is there an official checkpoint that navigates?

## 0. Answer

**Recommendation: hybrid, with SONIC staying the only body controller.**

1. **Navigation stays with our planner on SONIC.** Do not adopt Arena's controller or any GR00T checkpoint for
   navigation. M1 already walks the G1 through a MolmoSpaces house under SONIC with A* / Nav2 (`docs/M1.md`: 9 of 9
   go_to legs across 3 rooms, 0 falls). No official checkpoint navigates a house: the one Arena checkpoint that walks
   replays the single route it was trained on. With the bin removed from the scene it still turned about -100 deg and
   walked to the same empty table (§2.3).
2. **Take the manipulation pieces from Arena, not its controller.** Two things are worth adopting:
   - GR00T's decoupled action space: arm and hand joint targets, which is what Arena's checkpoints output.
   - Arena's Isaac Lab data pipeline, for fine-tuning.

   The action space can run on SONIC. SONIC's planner message already takes a 17-joint upper-body target and 7+7 hand
   joints (§4.2). Adopt it for M6 only if a one-day SONIC upper-body tracking test passes (§6, step 1). This choice
   is new and needs your decision; PLAN §0.2 ("SONIC only") is unaffected.
3. **Keep `ludo-g1-arena` as a reference harness** for NVIDIA's own checkpoints. Do not port Arena 0.2.x into the
   main stack: it needs Isaac Sim 6 / Isaac Lab 3, and the main stack is on 5.1 / 2.3.2.

**Checkpoints that emit a navigation command.** The owner's question was whether any checkpoint has nav. Every one
of these outputs one, but none is a general navigator:

| Checkpoint | Nav output | What it actually does | Run here? |
|---|---|---|---|
| `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` (N1.6, rev `gn1_6`) | `navigate_command` (vx, vy, wz) + base height | Walks one fixed route in one room: pick the box from the shelf, turn about -100 deg, walk about 2.5 m to the table, place. All 100 Mimic demos use the same scripted route (`navigation_subgoals` in `galileo_g1_locomanip_pick_and_place_environment.py`). It walks there even when the bin has been removed (§2.3). Non-commercial | **Yes**: 10 of 12 from a clean start, but 1 of 24 on later episodes in the same process (§2.4) |
| `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` (N1.7) | `navigate_command` exists in its action space | vx = vy = 0 throughout its training data. It never walks; AGILE only balances | **Yes**: 8 of 37 (22%) |
| `nvidia/g1_locomanip_finetune` (N1.5) | base velocity | **The only goal-conditioned one.** Its inputs include `state.goal_pose`, `state.object_pose` and `state.end_fixture_pose` in the robot frame (Isaac Lab `develop` `scripts/imitation_learning/locomanipulation_sdg/gr00t/rollout_policy.py:279-293`). Trained in the small Isaac Lab steering-wheel SDG scene. Non-commercial | No (the rollout script needs Isaac Lab `develop`; a small backport to 2.3.2) |
| `nvidia/GR00T-N1.7-ApplePnP-V1` | `navigate_command` | Real-G1 ONNX export for static apple pick-and-place | No (a real-robot export) |

If a checkpoint that walks to a goal is wanted for a demo, `g1_locomanip_finetune` is the only candidate. It is
worth a 1-2 day spike (§6, step 5), but expect a small-scene skill, not house navigation.

---

## 1. What ran

| Piece | Version |
|---|---|
| Simulator | `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` (reports 6.0.0-rc.22), public, no NGC login; Isaac Lab 3.0 dev (Arena submodule @ `e57379c6`) |
| Arena | `release/0.2.1` @ `8b4a3a47`, the last branch that ships `galileo_g1_static_pick_and_place` |
| N1.6 server | Isaac-GR00T @ `e29d8fc5`, checkpoint rev `gn1_6` @ `dfe74af`, port 6555 |
| N1.7 server | Isaac-GR00T @ `4b1dca9d`, checkpoint `main` @ `7f78beb`, backbone Cosmos-Reason2-2B (gated; our token already has access), port 6556 |
| Client | Arena's stock `Gr00tRemoteClosedloopPolicy`, wrapped for timing, logs and video (`arena_spike_policies.py`) |
| Probes (this pass) | `arena_probe_policies.py`: `SPIKE_BIN_POSE` holds the destination bin elsewhere. `SPIKE_WBC_RESET` resets the lower-body controller from the policy; this turned out to be a control only, because the env already resets it |

Tasks: `galileo_g1_locomanip_pick_and_place` (brown box from the shelf to the blue bin on the table,
`g1_wbc_joint` = HOMIE v2 legs, 30 s episodes) and `galileo_g1_static_pick_and_place` (apple to plate,
`g1_wbc_agile_joint` = AGILE legs, 6 s episodes). One env, 640x480 head camera, success judged by the env's own
termination terms.

---

## 2. Results

### 2.1 Success rates (every run, including my re-runs)

| Run | Checkpoint | Episodes | Success | Notes |
|---|---|---|---|---|
| `n16_eval` | N1.6 loco-manip, GPU physics | 10 | 1 | 8 time-outs, 1 box dropped (the ego view shows it slipping between the hands) |
| `n16_eval_rerender` | same, rerender-on-reset fix | 10 | 2 | 8 time-outs |
| `n16_eval_cpu` | same, CPU physics | 6 | 1 | 5 time-outs |
| `n16_smoke2` | same | 1 | 1 | |
| `verify_n16` (this pass) | same, rerender fix | 2 | 1 | Episode 0 succeeded in 19.3 s, episode 1 timed out: the earlier pattern again |
| `probe_n16_wbcreset` (this pass) | same, plus an extra controller-state reset each episode | 6 | 1 | Episode 0 succeeded; 4 time-outs, 1 box dropped |
| `probe_n16_seed1`…`seed7` (this pass) | same, one episode per fresh process, seeds 1-7 (default is 42) | 7 | 5 | Seeds 1 and 6 timed out |
| **N1.6 total** | | **42** | **12 (29%)** | **Clean start (episode 0 of a process, GPU): 10 of 12. Later episodes: 1 of 24** (§2.4) |
| `n17_eval` | N1.7 static apple | 10 | 3 | object_moved_rate 0.5 |
| `n17_novid` | same, no video | 20 | 4 | object_moved_rate 0.65 |
| `verify_n17` (this pass) | same | 3 | 0 | 3 time-outs |
| `probe_n17_seed1`…`seed4` (this pass) | same, one episode per fresh process | 4 | 1 | |
| **N1.7 total** | | **37** | **8 (22%)** | Clean start 2 of 7, later episodes 6 of 30: no first-episode effect |
| `probe_n16_nobin` (this pass) | N1.6, bin held under the floor | 3 | 0 (by design) | See §2.3 |

**Cross-checks.**
- Each count was re-derived from `<tag>_episodes.jsonl`, and each matches Arena's own `Metrics: {'success_rate': …}`
  line in the run log.
- The local copies are byte-identical to the box copies (md5).
- The joint state at every episode start is identical to episode 0 (max |dq| = 0.000).

### 2.2 What the robot does (checked on extracted frames)

- **N1.6 success.** Two-hand grasp on the shelf, lift, turn about -100 deg, walk to the table, lower the box into
  the bin. Takes 14.7-20.1 s; one took 29.3 s.
  - Videos: `outputs/arena/n16_eval/ep00_3p0.mp4` (side view), `n16_eval/ep00_ego.mp4` (what GR00T sees),
    `n16_eval_rerender/ep02_3p0.mp4` (fastest, 14.7 s).
- **N1.6 dominant failure: missed grasp.** The hands close in front of the box and the box stays on the shelf. The
  robot then walks to the bin anyway and mimes the place until the 30 s time-out (`n16_eval/ep05_3p0.mp4`).
- **N1.6 commands, recomputed from `cmd_log`.**
  - `navigate_command`: vx 0 to 0.40, vy -0.11 to 0.11, wz -0.39 to 0.10.
  - |nav| ≥ 0.05 in 50-62% of steps; `base_height` 0.60 to 0.75.
  - Every episode that did not drop the box turned to about -100 deg between 6 and 11.5 s and ended at the table
    edge (y -0.79 to -1.17), whether or not it held the box. Two of those episodes kept turning at the end, to -171
    and -176 deg.
- **N1.7 success.** The **left** hand grasps the apple and sets it on the plate, 2.4-5.2 s (`n17_eval/ep00_ego.mp4`).
  The main failure is the hand pushing the apple, which rolls, until the 6 s time-out (`n17_eval/ep03_ego.mp4`).
- **N1.7 commands.** vx = vy = 0 always. wz goes to -0.30 and `base_height` to 1.00. Both are the q01/q99 extremes
  of the checkpoint's own training statistics.
- **Velocity drive, no GR00T** (`veldrive_{homie,agile}`). Both lower bodies walked, turned, strafed and squatted in
  the Galileo scene without falling:
  - HOMIE v2: +0.4 m/s tracked at about 0.24 m/s, and a -100 deg turn overshot to -111 deg, settling at -105 to
    -107 deg.
  - AGILE: about 0.34 m/s, and the same turn landed at -94 to -96 deg.

  Top-down: `veldrive_agile/ep00_3p1.mp4`.

**Corrections to the spike report and the brief:**
1. N1.7 uses the **left** hand, not the right. The frames show the arm entering from image-left. The checkpoint's
   own dataset statistics agree: right-hand action std is 0.018, left-hand 0.598.
2. N1.7's actions are **absolute**, not relative: every `new_embodiment` action key has `rep: ABSOLUTE` in
   `processor_config.json`.
3. HOMIE's turn numbers are as above: -111 at the end of the command, -105 to -107 settled. The spike reported -98
   settled with a -116 overshoot. The conclusion stands: AGILE tracks better.
4. The lower-body controller state **is** reset between episodes. `G1DecoupledWBCJointAction.reset()` does not do
   it, but the env's `reset_wbc_policy` event does (`g1_events.reset_decoupled_wbc_joint_policy`). The spike's "side
   issue" is not a bug, and adding my own reset changed nothing.
5. The aggregate rates understate N1.6. From a clean start it succeeds 10 of 12 times; the 21% came from Arena's
   multi-episode loop (§2.4). Updated totals: N1.6 is 12 of 42, N1.7 is 8 of 37.

### 2.3 Probe 1: does N1.6's navigation depend on where the goal is?

**Method.** `SPIKE_BIN_POSE=0,0,-5` holds the blue bin 5 m under the floor at every step. The top-down video
confirms the table is bare apart from the drill. Nothing else changed; 3 episodes, rerender fix on.

**Result.** The navigation is unchanged:

| | Turn (yaw < -80 deg reached at) | End position | Root path |
|---|---|---|---|
| Normal scene: 21 episodes (`n16_eval`, `n16_eval_rerender`, `verify_n16`; the early box drop excluded) | 6.0-11.5 s | table edge, y -0.79 to -1.17 | 2.4-4.4 m |
| Bin removed, 3 episodes | 7.4-8.8 s | table edge, y -1.03 to -1.20 | 4.0-4.3 m |

**What the videos show.**
- In episode 0 the robot grasps the box, turns, carries it to the empty table and holds it over the spot where the
  bin was.
- It then shuffles around the table edge for the remaining about 20 s; yaw swings to -124 and -178 deg.
- Videos: `probe_n16_nobin/ep00_3p0.mp4` (side) and `ep00_3p1.mp4` (top-down, no bin).

**Conclusion.** The checkpoint reacts to the image locally, but where it walks is the memorised Mimic route, not a
goal it perceives. Put in a ProcTHOR house, its `navigate_command` would replay that route into whatever is there.
It cannot be a navigation skill; at most it is a demo of the controller interface.

### 2.4 Probe 2: N1.6 is good from a clean start and bad after an auto-reset

**First episode vs later episodes** (N1.6, GPU physics, every normal-scene run):

| | Episodes | Success |
|---|---|---|
| Episode 0 of a fresh process: seed 42 (5 runs) and seeds 1-7 (7 runs) | 12 | **10 (83%)** |
| Every later episode in the same process (`n16_eval`, `n16_eval_rerender`, `verify_n16`, `probe_n16_wbcreset`) | 24 | **1 (4%)** |

Clean-start successes take 18.1-20.1 s; one took 29.3 s.

**Ruled out:**
- **Layout luck.** Five of the seven other seeds also succeeded, so it is not seed 42's box placement.
- **The stale-frame bug.** It happens with the rerender fix on.
- **Controller state.** The env already resets the lower body; my extra reset gave 1 of 6.
- **Joint state.** The joint state at the start of each episode is identical to episode 0's.

**Not isolated.** Something that the env's auto-reset does not restore degrades the later episodes. Candidates are
PD/actuator targets, rigid-body caches, or client/server session state. Later episodes mostly fail at the grasp.

**N1.7 does not show the effect** (2 of 7 clean starts, 6 of 30 later).

**Practical rule.** Until the cause is found, evaluate Arena checkpoints with one episode per process (which is what
`probe_n16_seed*` did). **On its own task and scene, the N1.6 checkpoint is competent: about 83%.**

### 2.5 Speed and memory

- **GR00T round trip** (client side, localhost ZMQ, 640x480):
  - N1.6: mean 133-135 ms, p95 141-148 ms per 50-step chunk.
  - N1.7: mean 148-153 ms, p95 161-165 ms per 40-step chunk; the first call takes 3.8 s.
- **Sim speed:** RTF 0.19-0.26, i.e. 78-107 ms per 50 Hz step with one env and the head camera. The simulation
  blocks during each GR00T call, so a low RTF costs nothing but wall time.
- **VRAM:** about 15 GB per checkpoint plus the sim; both servers plus the sim use 21.8 GB.

---

## 3. How the Arena controller works

```
GR00T server (N1.6 / N1.7, own venv, ZMQ)  <- ego_view 640x480 + 31-D state (arms 14, hands 14, waist 3)
      │ chunk of 50 (N1.6) / 40 (N1.7) steps: arms, hands, waist, base_height_command, navigate_command
      ▼
Arena policy -> 50-D env action = [43 joint targets (sim order) | nav vx,vy,wz (body frame) | height | torso rpy = 0]
      ▼  G1DecoupledWBCJointAction.process_actions(), once per env.step = 50 Hz
G1DecoupledWholeBodyPolicy
  ├─ upper body: IdentityPolicy -> arm + hand targets pass straight through (waist owned by the lower body)
  └─ lower body (CPU ONNX):
       HOMIE v2 (g1_wbc_joint): stand.onnx if |nav| < 0.05 else walk.onnx; obs 86 × 6 history + gait clock -> 12 legs + 3 waist
       AGILE    (g1_wbc_agile_joint): LSTM student, obs [vx,vy,wz,h] + proprio -> 12 legs; needs G1_AGILE_CFG gains
      ▼  apply_actions(): PhysX joint PD at 200 Hz (decimation 4)
```

- **The process model is lockstep.** The controller runs inside `env.step`, and the GR00T call blocks the loop. No
  wall-clock threads, no real-time requirement. This is the opposite of SONIC's deploy (PLAN #19).
- **Real-robot parity.** Arena's HOMIE `stand.onnx` / `walk.onnx` are byte-identical to
  `GR00T-WholeBodyControl-{Balance,Walk}.onnx` in NVIDIA's real-robot decoupled WBC. I checked the LFS sha256
  (`f645da59…`) against the downloaded file. That repo calls it the controller used with GR00T N1.5 and N1.6 on the
  real G1.
- **Known harness defects** (upstream, on `release/0.2.1` and `main`):
  1. The loco-manip env leaves `num_rerenders_on_reset = 0`, so the first GR00T chunk of each later episode sees
     the previous episode's last frame.
  2. Later episodes in the same process degrade (83% to 4% for N1.6, §2.4). The cause is not isolated.

  The lower-body controller state is **not** one of them: the `reset_wbc_policy` event resets it.
- **Pose-goal navigation exists, but only for data generation.** `utils/p_controller.py` turns (x, y, heading)
  subgoals into `navigate_cmd`, capped at 0.4 m/s. It is used for Mimic data generation. It is what produced the one
  route the N1.6 checkpoint learned.

---

## 4. Integration design for Worldline (hybrid: SONIC body, decoupled GR00T manipulation)

### 4.1 What stays as built in M1

- **Processes.** P1 `wl-isaac` (Isaac Sim 5.1 / Lab 2.3.2, house, DDS bridge), P2 `wl-sonic` (unmodified deploy)
  and P3 `wl-body` (`BodyClient`, SonicMux, A* + pure pursuit or Nav2).
- **Navigation.** `navigate` / `go_to` / the `velocity` op go to the SONIC planner, as today (`docs/M1.md`,
  `docs/nav2.md`).
- **The Ludi contract.** `manipulate` never moves the base (PLAN #17, #26). GR00T's `navigate_command` is
  **never executed** in the product path; it is logged as telemetry only.

### 4.2 New: a decoupled-GR00T executor on SONIC

The SONIC deploy's planner message already carries `upper_body_position[17]` (waist 3 + left arm 7 + right arm 7)
and `left/right_hand_joints[7]`.
- These fields are parsed in `zmq_manager.hpp:776-913` @ `b042411`.
- `g1_deploy_onnx_ref.cpp:783-795` splices the 17 upper-body targets into the reference motion that SONIC tracks.
- Our `body/wire.py:build_planner_message` can already send them; CarryLock (PLAN #12) was going to use the same
  channel.

That is the same content as Arena's action minus the legs, so a GR00T checkpoint in the decoupled embodiment can
drive SONIC:

| GR00T output (per 50 Hz tick of a chunk) | Sent to SONIC as |
|---|---|
| left_arm 7, right_arm 7 | 14 of the 17 `upper_body_position` entries (reordered by joint name into SONIC's order) |
| waist 3 | The other 3 entries: hold the stand value. GR00T's waist output is not used in Arena either: HOMIE owns the waist, and AGILE locks it at 0, as the N1.7 data shows (waist action std 0) |
| left_hand 7, right_hand 7 | `left_hand_joints`, `right_hand_joints` (reordered by name; the Dex3 order differs between Arena, GR00T and the real ONNX export) |
| `base_height_command` | Hold the stand height. SONIC's planner `height` semantics are **[u]**, so do not pass it through until measured |
| `navigate_command` | **Dropped**: mode IDLE, movement 0, facing = current yaw. Logged as `groot_nav_pred` |

**Loop.** `ManipulationService` takes the body lease and runs this loop:
1. Read the ego frame (P1 head camera, 640x480) and the 31-D state (arms, hands and waist from P1's `g1_debug` /
   GT).
2. Call the GR00T server synchronously.
3. Stream the chunk at 50 Hz through SonicMux.
4. Repeat until the skill's success check or its `max_duration_s`.

On exit, CarryLock keeps sending the last upper-body and hand targets.

**The open question is fidelity.** Arena applies arm targets as direct PD targets on a HOMIE/AGILE body. SONIC
instead tracks them as part of a whole-body reference, with its own latency and tracking error. Policies trained on
Arena-style data may therefore see different arm dynamics on SONIC. Test this before building M6 on it (§6, step 1).

### 4.3 Where the training data comes from

- **Collection.** Arena / Isaac Lab ship the whole decoupled-embodiment pipeline:
  - `g1_wbc_pink` teleop (Pink IK upper body);
  - Mimic data generation (10 teleop demos → 100+ generated demos);
  - the HDF5→LeRobot converter;
  - the N1.7 fine-tune config `g1_sim_wbc_data_gr00t_n_1_7_config.py`;
  - and the N1.7 base already knows the G1 `unitree_g1_full_body_with_waist_height_nav_cmd` embodiment.
- **Arms at train vs deploy time.** Demos recorded on the Arena body (HOMIE/AGILE legs, PD arms) can train the arm
  skill, which is then executed on SONIC. The legs are standing in both cases.
- **The SONIC-token route** (`UNITREE_G1_SONIC`, PLAN §1.1) needs demos labelled with SONIC motion tokens. NVIDIA's
  workflow gets those from real-robot VR teleop through SONIC. We have no sim equivalent yet **[u]**.

### 4.4 If the tracking test fails

The fallback is a lockstep `wbc` body profile: the AGILE lower body in-process in P1, used only for `manipulate`
episodes. That reopens PLAN §0.2, so it needs an owner decision. It is not recommended now.

---

## 5. Trade-offs

| | SONIC (M1, main box) | Arena decoupled controller (this spike) |
|---|---|---|
| **Walks in a house** | **Yes, measured.** procthor-train-38: walk 3.4-3.6 m, turn error 0.5-2.6 deg, 9 of 9 go_to legs over 3 rooms with 0.03-0.16 m error, 0 falls, 3 of 3 cycles (`docs/M1.md` §0) | Only in Arena's Galileo room; houses **[u]**. It needs Isaac Sim 6 or a backport to 5.1 |
| **Velocity tracking** | Planner messages: world-frame direction, facing and speed | Body-frame (vx, vy, wz). AGILE gave 0.34 m/s for a 0.4 command; HOMIE 0.24 m/s with a 7 deg turn overshoot |
| **Official GR00T checkpoints that run** | None with Dex3 (community checkpoint for plumbing only, PLAN #21) | Two ran, in their own training scene only. N1.6 loco-manip: 10 of 12 from a clean start. N1.7 static: 8 of 37 |
| **A checkpoint that navigates** | – | None generalises. N1.6 replays one route (§2.3); `g1_locomanip_finetune` is goal-conditioned but untested |
| **What GR00T controls (Ludi, doc §15)** | The whole body through tokens: a recorded deviation (PLAN #25) | Arm and hand joints only; matches the doc. The executor in §4.2 keeps this on SONIC |
| **Timing** | Wall-clock, RTF ≥ 1 needed (0.994 measured) | Lockstep, so RTF 0.2 only costs wall time |
| **Real robot** | Same binary and DDS as the real G1; NVIDIA's current real-G1 VLA route | HOMIE ONNX is byte-identical to the real decoupled WBC (a ROS 2 loop on the robot); AGILE's real deploy **[u]** |
| **Software risk** | Pinned, measured, in our stack | Arena is alpha. 0.2.x needs Isaac Sim 6 / Lab 3.0, and `main` already dropped the static task. Upstream it has a stale-frame bug and an unexplained degradation after auto-reset (§2.4) |
| **Licences** | SONIC: NVIDIA Open Model License | N1.6 loco-manip and `g1_locomanip_finetune` are non-commercial. For N1.7 Static, the card says non-commercial but the LICENCE file is the Open Model License (commercial allowed): ask NVIDIA. Arena, AGILE and GR00T code are Apache-2.0 |

**Why hybrid, and why not "instead of".**
- **Navigation is solved on SONIC and unsolved by every GR00T checkpoint.** Swapping the body would throw away a
  measured house walker for a room-specific one.
- **Arena's real value is on the manipulation side.** It has official checkpoints and data tooling for an action
  space that matches Ludi's "GR00T drives the arms".
- **That action space fits SONIC's existing upper-body channel.** No second controller and no second Isaac version
  are needed.

---

## 6. Next steps (effort is my estimate)

| # | Step | Effort | Exit |
|---|---|---|---|
| 1 | **SONIC upper-body tracking test** on the M1 stack: stream demo arm and hand trajectories from the CC-BY dataset `nvidia/Arena-G1-Static-PickNPlace-Task` (LeRobot parquet, 50 Hz) through `upper_body_position` + hand joints while standing. Measure joint and wrist tracking error, latency and balance | 0.5-1 day | Proposed thresholds: wrist error under about 2 cm, latency under about 60 ms, no falls. If they pass, adopt §4.2 as the M6 manipulation interface. Owner decides |
| 2 | Decoupled-GR00T executor (`ManipulationService` backend): joint-name maps (Arena, GR00T, SONIC, Dex3), chunk streaming, nav drop, CarryLock hand-off, telemetry | 1-2 days | The N1.7 static checkpoint drives the arms on SONIC in a house (not expected to succeed; plumbing only) |
| 3 | Keep `ludo-g1-arena` as a regression harness for NVIDIA checkpoints. Evaluate one episode per process, with the rerender fix on, until the auto-reset degradation (§2.4) is found. Optionally bisect it by comparing sim state after the auto-reset with a fresh start | 0.5 day (bisecting: 0.5-1 day more) | A clean-start rate over 20 or more processes per checkpoint |
| 4 | House data: Mimic data generation for one house skill (for example "pick the apple from the counter"). Option (a): in Arena on Isaac Sim 6 with our house loaded (collision-filter fix, prestartup load). Option (b): Isaac Lab 2.3.2 locomanipulation SDG. Then fine-tune N1.7 | 3-5 days, plus fine-tuning GPU time **[u]** | Over 50% success of the fine-tuned skill in the house |
| 5 | Optional nav-checkpoint spike: `nvidia/g1_locomanip_finetune` in Isaac Lab's steering-wheel SDG scene (backport `launch_simulation`), testing whether goal-pose conditioning generalises across start and goal poses | 1-2 days | Evidence for or against a goal-conditioned GR00T navigator; product navigation stays on the planner either way |

Nothing here blocks the Worldline runtime port (PLAN M0-M2), which continues on SONIC.

---

## 7. Evidence index

- **Spike code:** `arena_spike/` (README has the recipe). This pass added `arena_spike/arena_probe_policies.py` and
  two pass-through knobs, `POLICY_TYPE` in `run_eval.sh` and `SPIKE_BIN_POSE` / `SPIKE_WBC_RESET` in
  `arena_exec.sh`.
- **Logs:** `outputs/arena/logs/<tag>_{episodes.jsonl,cmd_log.jsonl,timing.json,vram.log}` and `<tag>.log`.
- **Best videos:**
  - N1.6 full success: `outputs/arena/n16_eval/ep00_3p0.mp4` (side), `outputs/arena/n16_eval/ep00_ego.mp4` (GR00T's view).
  - N1.6 fastest success: `outputs/arena/n16_eval_rerender/ep02_3p0.mp4`.
  - N1.6 missed grasp, then walks to the bin anyway: `outputs/arena/n16_eval/ep05_3p0.mp4`.
  - Probe 1, bin removed and the same route walked: `outputs/arena/probe_n16_nobin/ep00_3p0.mp4`, `ep00_3p1.mp4`.
  - N1.7 success (left hand): `outputs/arena/n17_eval/ep00_ego.mp4`, `ep00_3p0.mp4`; failure (apple pushed): `n17_eval/ep03_ego.mp4`.
  - WBC velocity drive without GR00T: `outputs/arena/veldrive_agile/ep00_3p1.mp4`, `outputs/arena/veldrive_homie/ep00_3p1.mp4`.
  - SONIC in the house, for comparison: `outputs/m1/run-20260929-031832/third_person.mp4`.
- **Box state after this pass:** GR00T servers in tmux `srv_n16` and `srv_n17` (about 14 GB VRAM) and container
  `arena` are still up on `ludo-g1-arena`. Stop them with `tmux kill-session -t srv_n16` (and `srv_n17`) and
  `sudo docker rm -f arena`.
