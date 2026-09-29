# SONIC arm and hand tracking through the arm channel (recommendation (b))

Status: **measured on the live M1 stack, 2026-09-29** (box `ludo-g1-brev2`, procthor-train-38, P1 Isaac Sim 5.1 at
RTF 0.98-1.00 median, unmodified deploy @ `b042411`). 4 runs, about 22 minutes of testing, 0 falls.
Code: `body/arm.py`, `body/joint_map.py`, `body/g1_kin.py`, `body/sonic_mux.py` (overlay), `tools/arm_track_test.py`.
Contract: `docs/contracts/m1.md` §3.9. Evidence: `outputs/arm_track/<run>/` (§7).

Question (owner decision 2026-09-29, recommendation (b)): GR00T drives the G1's arms and hands with joint targets that
are streamed into SONIC through the planner message's upper-body override (`upper_body_position[17]`,
`left/right_hand_joints[7]`); SONIC keeps the legs and balance. How well does SONIC track such targets, standing,
walking and turning? The gate proposed in `docs/arena_vs_sonic.md` §6 step 1 was: wrist error under about 2 cm,
latency under about 60 ms, no falls.

## 0. Answer: conditional GO

**GO for (b) as the architecture, with three conditions. NO-GO for running Arena's PD-arm checkpoints on it
unchanged.** Against the pre-registered gate: no falls passes; latency fails (150-200 ms, not 60); wrist/palm
error passes for held poses once the body's servo is on (2-13 mm in typical poses) but fails in motion (22-25 mm
RMS, 45-58 mm max over a reach-grasp-lift).

| Gate | Raw SONIC tracking (run 1) | With servo (ki 2) + 0.15 s client lead (runs 2-3) | Verdict |
|---|---|---|---|
| Falls | 0 in 13.8 min (incl. 3 walk legs and 2 turns with the arms driven) | 0 in 8 min (6 walk legs, 4 turns with the arms driven) | **pass** |
| Balance | torso tilt ≤ 7.1° standing (arms moving), ≤ 6.8° walking; base drift ≤ 2.5 cm per segment | tilt ≤ 9.4°; drift ≤ 2.9 cm per segment | **pass** |
| Latency (sine phase lag, shoulder/elbow/wrist roll) | 127-215 ms (0.2-1 Hz); step dead time 70-120 ms, t50 120-180 ms | residual −19..+75 ms after the lead | **fail** raw; pass only for clients that know their future targets |
| Palm error, held poses | 17-52 mm (arms-up 75 mm) | 2-13 mm; counter reach 19-24 mm; arms-up 11-36 mm | **pass** for typical poses with the servo |
| Palm error, reach-grasp-lift (1.0-1.5 s moves) | 40 mm RMS, 68 mm max | 22-25 mm RMS, 45-54 mm max | **fail** (> 20 mm) |
| Wrist pitch | gain 0.22-0.46 at 0.2-1 Hz (not tracked) | unchanged (0.22-0.42) | **fail**: effectively a ±0.1 rad joint |
| Hands (Dex3) | fingers and thumb_2 of the left hand within 0.02 rad; right hand closes to 0.72; both thumbs' rotation joints stuck | same | channel **pass**, sim hand **fail** (P1/asset, §3.7) |

Conditions for GO:
1. **The body's arm servo stays on** (`servo_ki` 2.0, now the default, §3.5). It removes the pose-dependent
   steady-state error that SONIC leaves on the arms (0.1-0.3 rad raw).
2. **The GR00T executor sends each action chunk about 0.15 s ahead** (it has the whole chunk, so it can: send
   `chunk[i + 7]` at step i at 50 Hz). This removes most of the lag for shoulder, elbow and wrist roll (§3.5).
3. **Manipulation skills are trained or fine-tuned on data executed through this same path** (SONIC arms), not
   taken from Arena's PD-arm embodiment as is. On SONIC the wrist pitch barely moves, shoulder yaw and roll are
   attenuated to 0.6-0.8, and a 1-1.5 s reach still ends 2-5 cm off; a visually closed-loop policy trained on
   these dynamics can absorb that, one trained on Arena's crisp PD arms is unlikely to.

What (b) is good for today: reaching, carrying (palm RMS 9-17 mm while walking with the servo), handing over,
pointing, holding poses while walking and turning. What it is not good for without condition 3: grasps that need
sub-2 cm wrist precision or wrist-pitch articulation. The fallback in `docs/arena_vs_sonic.md` §4.4 (a lockstep
AGILE/HOMIE lower body for manipulation episodes) would reopen PLAN §0.2 and is not recommended on this evidence:
the failures are precision, not safety, and conditions 1-3 are cheap.

---

## 1. What was built

**Arm channel (op `arm`, `body/arm.py`, contract §3.9).** A streaming op, separate from the one active leg motion:
while it is active SonicMux adds `upper_body_position[17]` + `upper_body_velocity[17]` (+ both hands once given) to
**every** 50 Hz planner message, whatever mode/movement/facing the leg motion sends. Hence it composes with IDLE,
`walk`, `velocity`, `turn_to` and `go_to` (measured, §3.8).
- Input `upper_body` by joint name (dict) or as 17 values in named MuJoCo order; `left_hand`/`right_hand` as 7 Dex3
  values or a closure 0..1. Clamped to URDF limits − 0.02 rad, slew-limited to 6 rad/s (the limiter engaged on 12
  ticks in 22 min, all in the step test).
- Watchdog 0.3 s → hold the last pose 1.0 s → min-jerk blend (1.5 s) to SONIC's **own** reference arms, read live
  from `g1_debug.body_q_target`, hands to the deploy's default fist → override released. The watchdog tripped 12
  times during the runs, mostly while the tool paused its stream for 1.5 s to start a video recorder (fixed in the
  tool since); each time the stream resumed from hold/blend without a fall.
- One owner at a time (`stream` id): `arm_busy` unless `preempt: true`; a stream that stopped (hold/blend) can be
  taken over; `stop {arms: true}` blends back; a fall drops the override at once.
- Waist: SONIC's own reference waist by default (`waist: "ref"`), so the client owns only the 14 arm joints.

**Joint order (`body/joint_map.py`).** The 17 values are in SONIC's IsaacLab-interleaved order
(`policy_parameters.hpp:80-81`, `g1_deploy_onnx_ref.cpp:784-791`): waist, then L/R pairs joint by joint.
`wire_from_mj17` is the only builder. `body/tests/test_joint_map.py` parses both upstream lists
(`policy_parameters.hpp`, `pico_manager_thread_server.py:1664-1668`), the deploy's splice line, the MJCF joint order
(Dex3 order thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1) and the URDF limits, and shows that a
contiguous `q[12:29]` slice would scramble the arms (PLAN §6 invariant 8). A live smoke test moved only the right
elbow when only `right_elbow_joint` was commanded.

**FK/IK (`body/g1_kin.py`).** Arm chains of `main.urdf` (the URDF P1 simulates), pinned by a test. Palm and wrist
positions in the pelvis frame; a damped-least-squares palm IK for the scripted poses. Reach at table height (0.75 m)
is short on the G1: the palm gets at most ~0.26 m in front of the pelvis (shoulder-to-palm 0.372 m straight), so a
table must be within ~0.35 m of the pelvis.

## 2. Method (`tools/arm_track_test.py`)

- **Commands** at 50 Hz through `BodyClient.arm_stream()`, logged at the client's send time, so every latency is end
  to end: client → wl-body → SonicMux → deploy input thread → encoder/policy → rt/lowcmd → PD → joint.
- **Measurement.** `g1_debug` at 50 Hz: `body_q` (lowstate as the deploy consumed it, MuJoCo order),
  `last_action` (the policy's joint targets = rt/lowcmd q), `body_q_target` (SONIC's reference without the override),
  `left/right_hand_q` (Dex3 state). gt.pose for the pelvis. Cross-checks: P1 `get_joint_state` at the end of each
  held pose agrees with `g1_debug` within 0.008 rad (arms) and 0.0012 rad (hands); P1 `record` (motor_q from the sim,
  ~40 Hz) gives step dead times 10 ms shorter than `g1_debug` (67-107 vs 78-118 ms), as expected.
- **Phases.** `baseline` (4 s, no override) · `static` (6 poses: SONIC reference, policy default, both palms forward
  at table height, right palm to a counter at 0.865 m, arms up, default; 1.5 s min-jerk, 4 s holds; steady state =
  last 2 s) · `steps` (±0.3 rad on R elbow, R shoulder pitch, L elbow) · `sine` (±0.3 rad, centre moved so the sine
  never clips, 0.2/0.5/1.0 Hz, each of the 7 arm joints, left arm, right arm, both; 2 Hz on 3 joints; `vel=est`
  A/B) · `grasp` (IK keyframes: pregrasp → approach → close → lift → hold → place → open → retreat, 10.5 s, one
  continuous trajectory) · `hands` (both Dex3 open → 50 % → closed → open) · `walk` (SLOW_WALK 0.3 m/s `walk` op
  ~9.7 s legs and 180° `turn_to`s: without override, then with a carry pose, a 0.5 Hz sine, the default pose, and
  the grasp script replayed while walking).
- **Metrics** (`metrics.json`): per-joint gain, phase lag → latency (least-squares sine fit of command and
  measurement on their own timestamps) for the measurement and for the policy action, RMS error (raw and
  lag-compensated), cross-talk; −3 dB bandwidth; step dead time / t50 / rise / overshoot; steady-state joint error
  and palm/wrist position error (FK of measured vs commanded arms, same measured waist, pelvis frame); hand closure;
  pelvis height, torso tilt, base drift per segment; falls; walk results.

| Run | Config | Phases | Duration |
|---|---|---|---|
| `20260929-051536` | raw: servo off, no lead | all | 13.8 min |
| `20260929-053804-servo` | servo ki 2, lead 0.15 s, chase video | baseline, static, sine (right arm, 0.5/1 Hz), grasp†, hands, walk | 4.6 min |
| `20260929-054623-servo-grasp` | same, grasp as one continuous trajectory | baseline, static, grasp, walk | 2.4 min |
| `20260929-055110-servo4` | servo ki 4, max 0.6, lead 0.15 s | baseline, static, grasp | 0.9 min |

† In run 2 the grasp keyframes were streamed one by one, so the lead did not carry across keyframes; run 3 fixed it.

## 3. Results

### 3.1 Balance, falls, drift

0 falls in all runs (fall = gt `fallen` or pelvis z < 0.55 m). Standing with the arms moving: torso tilt ≤ 7.1° raw
and ≤ 9.4° with the servo (both maxima during the fastest large move, arms-up → default: 2.7 rad of shoulder pitch in
1.5 s); ≤ 11.4° with ki 4. Without override: tilt ≤ 2.9°. Pelvis height std ≤ 1 mm while standing. Base drift ≤ 2.5-3.0
cm per segment (≤ 0.4 cm without override), about 0.13-0.5 m summed over all standing segments of a run; the robot
re-steps rather than slides. Plot: `pelvis.png`.

### 3.2 Latency

**Steps** (±0.3 rad, run 1): the policy action starts moving 58 ms after the client sends (10 % at 57-58 ms), the
joint 78-118 ms after (10 %), 50 % at 118-178 ms; shoulder pitch settles within 5 % in 0.22-0.28 s with ≤ 1.5 %
overshoot, the elbow ends 0.01-0.05 rad short. `steps.png`.

**Sines** (run 1, one arm at a time, mean of left and right; latency = phase lag / 2πf):

| Joint | 0.2 Hz gain / latency | 0.5 Hz | 1.0 Hz | 2.0 Hz | lag-compensated RMS at 0.5 / 1 Hz |
|---|---|---|---|---|---|
| shoulder pitch | 1.13 / 201 ms | 1.06 / 173 ms | 1.02 / 146 ms | 1.05 / 119 ms | 0.016 / 0.013 rad |
| shoulder roll | 0.83 / 186 | 0.78 / 151 | 0.79 / 131 | – | 0.049 / 0.049 |
| shoulder yaw | 0.71 / 171 | 0.66 / 175 | 0.57 / 152 | – | 0.074 / 0.093 |
| elbow | 0.93 / 127 | 0.88 / 145 | 0.80 / 131 | 0.69 / 102 | 0.028 / 0.043 |
| wrist roll | 1.02 / 215 | 0.91 / 190 | 0.73 / 163 | – | 0.023 / 0.058 |
| wrist pitch | **0.45** / 375 | **0.34** / 327 | **0.22** / 254 | 0.17 / 182 | 0.142 / 0.166 |
| wrist yaw | 0.95 / 389 | 0.72 / 318 | 0.46 / 243 | – | 0.061 / 0.115 |

- The lag is mostly in the policy, not in the plumbing: the policy's own action already lags the command by
  68-160 ms (shoulder, elbow, wrist roll) and 188-330 ms (wrist pitch/yaw). The override replaces all 10 future
  reference frames the encoder sees with the same constant pose (`g1_deploy_onnx_ref.cpp:738-800`), so every new
  target looks to SONIC like a small step to be absorbed smoothly.
- Latency falls with frequency (a dynamic lag, not a pure delay), so a fixed 0.15 s lead fits shoulder pitch, elbow
  and wrist roll well; after it the residual shape error is 0.013-0.058 rad (last column).
- Both arms at once (run 1): shoulder pitch 1.05-1.19, elbow 0.75-0.84, wrist roll 0.78-1.02 as before; shoulder
  roll 0.71 and shoulder yaw 0.35-0.43 are worse than one arm at a time. Latency 111-206 ms (wrist pitch/yaw
  239-385 ms).
- −3 dB bandwidth (one arm): shoulder pitch > 2 Hz, elbow ≈ 1.9 Hz (right) / > 1 Hz, wrist roll > 1 Hz, shoulder
  roll > 1 Hz (flat gain ≈ 0.8, an attenuation not a roll-off), wrist yaw 0.4-0.6 Hz, shoulder yaw 0.3 Hz /
  < 0.2 Hz, wrist pitch < 0.2 Hz. `bode.png`, `sine_examples.png`.
- Cross-talk: exciting one joint moves the others of the same arm by 0.03-0.10 rad RMS (shoulder pitch → elbow the
  largest); exciting one wrist pitch moves the other wrist pitch by up to 0.10 rad RMS.
- `upper_body_velocity` feed-forward (`vel: est`, right elbow): gain 0.93 vs 0.91 at 0.5 Hz, 0.86 vs 0.84 at 1 Hz,
  latency −11 ms and −3 ms. Within run-to-run noise (±0.05 gain, ±10 ms); the default stays `zero` (= upstream).

### 3.3 Static accuracy

Steady-state error in the last 2 s of each 4 s hold. Palm error = FK of the measured arm vs FK of the commanded arm,
both on the measured waist, pelvis frame (`static_poses.png`).

| Pose | Raw: palm L / R, worst joint | Servo ki 2 (run 2 / run 3): palm L / R, worst joint |
|---|---|---|
| SONIC's own reference | 43 / 32 mm, R wrist roll −0.28 rad | 8 / 13, 2 / 12 mm; R wrist pitch −0.15 |
| policy default angles | 24 / 17 mm, R elbow +0.16 | 7 / 6, 6 / 5 mm; ≤ 0.03 rad |
| both palms forward at table height | 18 / 23 mm, R elbow +0.14 | 9 / 13, 5 / 7 mm; R wrist pitch −0.09..−0.11 |
| right palm to a counter (0.865 m) | 23 / 52 mm, L elbow +0.28 | 24 / 10, 19 / 23 mm; shoulder yaw −0.10..−0.14 |
| arms up (shoulder pitch −2.5) | 72 / 75 mm, R shoulder yaw +0.22 | 36 / 11, 16 / 20 mm; L wrist roll −0.27..−0.30 |
| carry pose before / after a walk | 27 / 20, 19 / 22 mm | 10 / 13, 6 / 9; 5 / 8, 6 / 9 mm |

- Raw, the error is pose dependent and mostly **in the policy's action**, not PD sag: e.g. at table height the policy
  commands the shoulders 0.30-0.32 rad above the target to hold the arm against gravity and the joint ends within
  0.04 rad; but it also leaves the elbows 0.10-0.16 rad too straight and, in its own reference pose, the wrist roll
  0.23-0.28 rad off. SONIC does not track even its **own** reference arms better than this (no override: 0.22-0.29
  rad worst joint, palm 5-43 mm), so this is the policy's accuracy, not a problem of the override.
- With the servo, 13 of 14 joints sit within 0.01 rad in ordinary poses (all but wrist pitch). What remains: wrist pitch (−0.06..−0.15 rad,
  the servo's ±0.4 rad authority is not enough for a joint with gain ~0.4), shoulder yaw in the counter reach, and
  the extreme arms-up pose. Settling into a 0.05 rad band takes 0.8-3.6 s where it happens.
- ki 4 (max 0.6 rad) was not better (palms 4-41 mm, arms-up L wrist roll −0.47 rad, tilt 11.4°); ki 2 is the default.

### 3.4 Reach-grasp-lift (right arm, 10.5 s, IK keyframes, Dex3 close/open)

| | Palm RMS | Palm p95 | Palm max | Best lag vs desired |
|---|---|---|---|---|
| Raw (run 1), standing | 40.5 mm | 59.6 | 67.5 | 0.18 s |
| Raw, while walking at 0.3 m/s | 59.5 | 85.1 | 93.8 | 0.19 |
| Servo + lead (run 2, keyframes streamed separately) | 25.0 / 24.9 walking | 46.0 / 45.3 | 53.6 / 57.3 | 0.15 |
| Servo + lead (run 3, one continuous trajectory) | **22.3 / 23.7 walking** | 37.8 / 38.8 | 50.3 / 45.0 | 0.16 |
| Servo ki 4 + lead | 27.0 | 51.8 | 58.0 | 0.15 |

Even with the 0.15 s lead the measurement still trails the desired trajectory by ~0.16 s: during whole-arm reaching
the elbow and wrist pitch lag ~0.3 s behind what is sent, and the elbow stops ~0.1-0.15 rad short of a near-straight
target (1.4 rad) while the policy's own action sits 0.3 rad below it. Joint RMS in run 3: elbow 0.13, wrist pitch
0.13, wrist roll 0.09 (uncommanded, disturbed by the reach), shoulder yaw 0.07, shoulder pitch 0.07, shoulder roll
0.03, wrist yaw 0.04 rad. The palm ends 2-4 cm low throughout (`grasp.png`). A GR00T policy closing the loop through
the camera at 2.5 Hz can correct this kind of slow, repeatable error; an open-loop script or a policy trained on PD arms
cannot.

### 3.5 What the servo and the lead buy (right arm sines, run 2 vs run 1)

| Joint | 0.5 Hz raw: gain / latency / RMS | 0.5 Hz servo + lead | 1 Hz raw | 1 Hz servo + lead |
|---|---|---|---|---|
| shoulder pitch | 1.04 / 175 ms / 0.115 rad | 0.99 / 28 ms / 0.020 | 1.00 / 146 / 0.175 | 0.92 / −7 / 0.032 |
| shoulder roll | 0.75 / 152 / 0.097 | 0.73 / 75 / 0.069 | 0.75 / 131 / 0.145 | 0.67 / −19 / 0.077 |
| shoulder yaw | 0.63 / 178 / 0.118 | 0.78 / 73 / 0.061 | 0.53 / 154 / 0.169 | 0.63 / 11 / 0.079 |
| elbow | 0.91 / 156 / 0.160 | 0.88 / 5 / 0.040 | 0.84 / 135 / 0.207 | 0.77 / −14 / 0.060 |
| wrist roll | 0.90 / 187 / 0.120 | 0.94 / 25 / 0.029 | 0.72 / 164 / 0.179 | 0.69 / 22 / 0.073 |
| wrist pitch | 0.33 / 317 / 0.199 | 0.42 / 240 / 0.159 | 0.22 / 248 / 0.222 | 0.22 / 142 / 0.183 |
| wrist yaw | 0.76 / 317 / 0.181 | 0.87 / 251 / 0.150 | 0.49 / 240 / 0.232 | 0.44 / 128 / 0.158 |

The lead is what removes the delay; the servo removes the bias and lifts low-frequency gain a little. Neither fixes
the attenuated joints (shoulder roll/yaw) at 1 Hz or the wrist pitch.

### 3.6 Walking and turning with the arms

All 21 walk legs and turns (3 runs × 7, with and without an override) succeeded (`walk` op, 0.3 m/s, 9.6-9.9 s;
`turn_to` 180°). Walks covered 2.2-3.2 m with cross-track ≤ 0.167 m without arms and ≤ 0.125 m with arms; torso tilt
5.9-9.4° without arms and 6.0-8.2° with; pelvis z ≥ 0.734 m. Arm tracking during the legs:

| | Raw (run 1) | Servo + lead (runs 2-3) |
|---|---|---|
| carry pose (both palms forward), walking | arm RMS 0.078 rad (max joint 0.25), palm RMS 42 / 45 mm | 0.032-0.040 rad, palm 9-17 mm |
| 0.5 Hz sine (R shoulder pitch + L elbow), turning 180° | 0.068 rad, palm 32 / 36 mm | 0.037-0.041 rad, palm 13-16 mm |
| same, walking | 0.071 rad, palm 41 / 47 mm | 0.038-0.039 rad, palm 15-17 mm; R shoulder pitch gain 1.02-1.03, residual latency 18 ms, RMS 0.020-0.022 rad; L elbow gain 0.61-0.67 (0.74-0.77 while turning) |
| default pose, turning | 0.069 rad, palm 31 / 21 mm | 0.024-0.025 rad, palm 7-10 mm |

The planner's arm swing is replaced by the override while walking (upstream does the same in its
`PLANNER_FROZEN_UPPER_BODY` teleop mode, `pico_manager_thread_server.py:1830-1840`); no effect on the gait was seen.

### 3.7 Dex3 hands

The channel is exact: the deploy's hand action (`last_*_hand_action`) equals the command in every hold. The sim hands
are not:
- **Left hand:** middle/index fingers and thumb_2 reach the command within 0.02 rad (closure error ≤ 0.005); 90 % of
  a 0.5-closure change in 0.36-0.59 s (the deploy limits each write to ±0.25 rad from the measured position at kp 1.5).
- **Right hand:** closes to 0.445 for 0.5 and **0.72 for 1.0**: `middle_0` stops 1.27 rad short and `thumb_2` 0.84
  rad short, blocked (the pattern of a collision with the thumb).
- **Both thumbs' rotation joints never move:** thumb_0 / thumb_1 stay at −0.35 / −0.72 rad (left) and +0.35 / −1.05
  rad (right) whatever is commanded; thumb_1 sits on its joint limit on both hands. P1 receives the targets
  (`get_joint_state.q_target` moves), so this is on the P1 / hand-asset side (initial pose or collision), for the isaac
  agent. Until it is fixed, grasp success measured in P1 understates what the channel can do.

`hands.png`.

## 4. Limits and caveats

- One simulator (Isaac, P1), one house, one robot position; RTF 0.98-1.00 median (p5 0.87-0.99). A run with
  timing jitter would add lag (docs/walk_diagnosis.md); these runs were on a quiet box (load 2.4-8, the owner's viz
  server running; the chase camera was on for runs 2-3).
- No payload: the hands held nothing. Holding an object adds gravity load that the servo would have to absorb.
- Accuracy is joint space + FK on `main.urdf`; the pelvis frame excludes pelvis sway (the torso tilts up to 9° when the
  arms move fast, which moves the hand in the world by up to ~1-2 cm more).
- The servo's delay (0.15 s) and gain (2/s) are tuned for shoulder/elbow; wrist pitch/yaw lag 250-390 ms, so the
  servo sees part of their lag as error.
- Arena checkpoints were not run through the channel (their Galileo scene and PD arms are the Arena harness, §5).

## 5. What to do next (M6)

1. **Executor (`docs/arena_vs_sonic.md` §6 step 2)**: `BodyClient.arm_stream()` is the interface. Map GR00T's
   left/right arm and hand keys by name (`joint_map`), send chunk rows at 50 Hz with a 7-row (0.14 s) lead, leave the
   waist to SONIC, keep `servo_ki` 2. Drop `navigate_command`.
2. **Data**: collect demonstrations whose arms are executed by SONIC (Isaac Lab teleop or Mimic replayed through this
   channel), then fine-tune; evaluate an Arena checkpoint through the channel only as a plumbing test.
3. **Hands**: ask the isaac agent to fix the Dex3 thumbs and the right hand's closure in P1 (§3.7).
4. **Wrist pitch**: treat as ±0.1 rad in skill design (orient the palm with shoulder/elbow/wrist roll), or test whether
   SONIC tracks it better from the VR 3-point / wrist targets (`motion_joint_positions_wrists_10frame_step1`, the
   `smpl`/`teleop` encoder modes), which is a different input path.
5. Re-run `tools/arm_track_test.py` after any deploy or P1 change; `--phases static,grasp,walk` takes 2.5 minutes.

## 6. Reproduce

```bash
# box, M1 stack up; restart the body so it runs the arm channel
bash scripts/m1_restart_body.sh
.venv/bin/python -m tools.arm_track_test --video --servo-ki 0 --out outputs/arm_track/$(date +%Y%m%d-%H%M%S)   # raw, 14 min
.venv/bin/python -m tools.arm_track_test --video --chase --servo-ki 2 --lead 0.15 --phases baseline,static,grasp,walk \
    --out outputs/arm_track/$(date +%Y%m%d-%H%M%S)-servo                                                         # 2.5 min
.venv/bin/python -m tools.arm_track_test --analyze outputs/arm_track/<run>        # re-analyse (also on the laptop)
python -m tools.arm_track_test --fake --quick --out /tmp/arm-fake                # plumbing check against the fakes
```
Unit/integration tests: `body/tests/test_joint_map.py`, `test_g1_kin.py`, `test_arm.py` (50 body tests pass).

## 7. Evidence index

`outputs/arm_track/` (laptop and box `/work/worldline-g1/outputs/arm_track/`):
- `20260929-051536/` raw run: `metrics.json`, `bode.png`, `sine_examples.png`, `static_poses.png`, `steps.png`,
  `grasp.png`, `hands.png`, `pelvis.png`, `walk.png`, `raw.npz` (all samples), `run.json` (segments, events, P1
  joint-state snapshots), `p1_steps.npz` (P1 record); videos `video/*-arm-{static,sine_example,grasp,walk}/`
  (head + top + composite; the chase camera was off in P1 during this run).
- `20260929-053804-servo/`, `20260929-054623-servo-grasp/`: servo + lead, same files; videos with the **chase**
  camera (`composite.mp4`, `contact_sheet.png`; P1 VizCams set to `low` for the run and restored to `off`).
- `20260929-055110-servo4/`: ki 4 comparison (no video).
