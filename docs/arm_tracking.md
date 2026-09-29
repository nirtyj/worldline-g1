# SONIC arm and hand tracking through the arm channel (recommendation (b))

Status: **measured on the live M1 stack, 2026-09-29** (G0 §0-§7; body wave §8; wave-2 fixes of the verifier's defects §9) (box `ludo-g1-brev2`, procthor-train-38, P1 Isaac Sim 5.1 at
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
| Palm error, reach-grasp-lift (1.0-1.5 s moves) | 40 mm RMS, 68 mm max | 22-25 mm RMS, 45-57 mm max | **fail** (> 20 mm) |
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
and ≤ 9.0° with the servo (both maxima during the fastest large move, arms-up → default: 2.7 rad of shoulder pitch in
1.5 s); ≤ 11.4° with ki 4. The 9.4° whole-run maximum of run 3 is a walk leg *without* the override (§3.6). Without override: tilt ≤ 2.9°. Pelvis height std ≤ 1 mm while standing. Base drift ≤ 2.5-3.0
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

---

## 8. Body wave (2026-09-29, owner body-arm): G0 defects, halt latch, chunk mode, arm_script + CarryLock, scan

Status: **built and run live** on `ludo-g1-brev2` (procthor-train-38, P1 RTF 10 s 0.99-1.00, unmodified deploy),
final body code = commit `c044754` (md5 of `body/arm.py` on the box = laptop). 0 falls in every run of this section.
Code: `body/arm.py` (ArmChannel), `body/arm_script.py`, `body/scan.py`, `body/carry.py`, `body/g1_kin.py` (analytic
palm Jacobian), `tools/arm_wave_test.py` (live chunk / pick / scan), `tools/arm_track_test.py` (`--servo-model`,
`--servo-extra`). Tests: `body/tests/test_arm_channel.py`, `test_arm_script.py` (tick by tick on a SONIC-like plant,
`arm_sim.py`), `test_arm.py` (ZMQ + fakes). Contract: `docs/contracts/m1.md` §3.9 v0.6. Evidence:
`outputs/body_wave/<run>/` (laptop and box).

### 8.1 G0 defects

| Defect (G0 verifier) | Fix | Evidence |
|---|---|---|
| D1 `stop {arms: true}` does not stick while the owner streams | The stopped stream is rejected `arm_stopped` during and after the blend until `restart: true` or a new stream id; a chunk session ends at once (`canceled`, ended_by `stop`) and its messages are `stale_session` | `test_arm.py` (the owner streams 0.5 s after the stop: every message `arm_stopped`, nothing reaches the deploy), `test_arm_channel.py` |
| D2 a halt never touches the arms | `ArmChannel.latch(epoch)` (below, §8.3) | live 6/6, §8.3 |
| A watchdog-ended session reports `succeeded` | `failed`, reason `client_silent`, ended_by `watchdog` (chunk sessions into `hold_on_end`) | tests; body-core publishes `body.fault{policy_lost}` on it |
| Slew limit ≈ 2× `max_vel` (tick() also ran on every message with dt floored at 0.02 s) | Only the 50 Hz tick advances the limiter; a message only sets the target | `test_arm.py`: streaming at 100 Hz, the far side moves at 1.2-2.5 rad/s for `max_vel` 2 (the old code advanced the limiter on every message too: at 100 Hz messages that is 150 steps/s); tick-level test: every step ≤ `max_vel`·dt at 150 messages/s |
| Servo adds 4.5-13.5 % overshoot | Diagnosed, partly reduced, §8.2 | 6 live runs |

### 8.2 Servo "overshoot": what it is, and before/after

`arm_track_test --phases static,steps --lead 0.15` (6 steps of ±0.3 rad on R elbow, R shoulder pitch, L elbow, 2.5 s
each; 6 static poses), same box and house, one variant per run. "G0 overshoot" = the verifier's metric (max over the
2.5 s hold); "transient" = max in the first 0.8 s; "late std" = std of the joint over 1.0-2.5 s after the step, % of
the step; palm = worst palm error per static pose (mm).

| Run | Servo | G0 overshoot mean / max | Transient mean / max | Late std | Final error max | Static palm (6 poses) |
|---|---|---|---|---|---|---|
| `20260929-083704-servo-raw` | off | 1.3 / 4.0 % | 0.6 / 1.9 % | 0.7 % | 0.087 rad | 30, 19, 21, 52, 84, 19 |
| `20260929-075526-servo-delay` | G0 (`delay`, ki 2) | 7.6 / 11.2 % | 3.5 / 8.0 % | 5.1 % | 0.024 | 15, 7, 8, 28, 36, 4 |
| `20260929-075628-servo-gated` | **`gated`, ki 2 (default)** | 6.5 / 12.5 % | **2.1 / 7.8 %** | 5.4 % | 0.043 | 15, 5, 7, 27, 38, 4 |
| `20260929-083759-servo-ki1` | `gated`, ki 1 | 3.1 / 13.8 % | 0.0 / 0.0 % | 5.7 % | 0.076 | 11, 4, 7, 14, 38, 7 |
| `20260929-083905-servo-db01` | `gated`, ki 2, error deadband 0.01 rad | 7.3 / 12.2 % | 1.6 / 9.4 % | 5.6 % | 0.012 | 10, 7, 11, 15, 52, 7 |

- The G0 "overshoot" is mostly **not** a step overshoot: it is a slow wander of the held joint (late std 5-6 % of
  the step, ~0.015-0.017 rad) that every servo variant shows and the raw arm does not (0.7 %). A static-hold
  diagnostic (`outputs/body_wave/servo_diag.py`, R elbow 0.3 rad below SONIC's reference, 10 s): with the servo
  the arm joints wander with std up to 0.020-0.025 rad around a mean error of 0.0003-0.0008 rad; without it they
  are steady (std ≤ 0.0034) but the elbow sits **0.267 rad** off target. A hysteresis freeze of the correction
  (stop integrating once converged) made it worse (std 0.043-0.057 rad), so the wander is SONIC's arm being less
  steady at poses it does not reach on its own, which the active servo partly damps; it is not integrator windup.
  The freeze and deadband options were removed again.
- What the `gated` model (FOPDT reference, dead time 0.09 s + tau 0.085 s, integration weighted by
  exp(−|dy_ref/dt| / 0.3 rad/s)) buys: the transient overshoot drops from 3.5 to 2.1 % mean (tick-level sim: 8.1 →
  0.5 %), static accuracy unchanged (palms within ±2 mm of the G0 servo pose by pose). ki 1 removes the transient
  entirely but converges slower (final error up to 0.076 rad after 2.5 s). Default: `gated`, ki 2; `delay` stays
  selectable (`servo_model`).
- The servo correction now carries over a take-over (the new session starts from the pre-correction reference; the
  old code counted it twice) and measured/latched holds preload it (`corr = sent − measured`), so a halt does not
  step the wire (the first live chunk run stepped the wrist pitch and elbow by 0.13-0.19 rad at the slew limit).

### 8.3 B.8 chunk mode, cancel and halt (G2-style)

`arm_wave_test chunk` emulates `groot_arms`: 10 sessions (alternating arms), each a start message (open hands,
`lead_s` 0.15, `hold_on_end` measured) and then every 0.4 s an "inference": t0 = receive time of the newest g1_debug,
150 ms of simulated inference, a 40-row chunk (50 Hz, SONIC wire order) of a smooth synthetic reach (IK keyframe out
of the measured start pose, a Hann-windowed 0.5 Hz sway, hand closing to 0.6, back) with a per-chunk random offset
(σ 0.015 rad) for GR00T's chunk-to-chunk disagreement. 4 normal ends (`stand`, `target` taken over by the next
session, `measured`), 3 cancels (end `measured` mid-sway), 3 halts on the body's halt lane (PUSH 5612, then
`resume` and `end` on the latched pose). After each cancel/halt ack, 3 in-flight chunks carrying +0.4 rad on the
moving elbow.

| Final run `20260929-084352-final-chunk` | Result |
|---|---|
| Sessions / falls | 10/10 opened, 0 falls (GT), RTF 10 s 1.00 |
| Chunks | 98 sent, 98 applied, 0 dropped; age at the body (t0 → reply) p50 155 ms, p95 163, max 168 |
| Joint step per planner message (wire, 14 arm joints) | chunk-to-chunk boundaries (0-120 ms after 88 chunk replies) max **0.032 rad**, p99 0.030; elsewhere in the sessions max 0.030, p99 0.027; the synthetic trajectory's own max step 0.029 (the 5-tick cross-fade adds ≤ 0.003 rad) |
| First chunk of a session | max 0.056 rad (the cross-fade from the held reference to the first row, i.e. the servo residual of the pose it takes over) |
| Measured (g1_debug, 20 ms samples) | max 0.086 rad within 0.3 s after a boundary (p99 0.047), 0.061 elsewhere (p99 0.040) |
| Clamp / slew / stall | clamped_frac 0, slew_frac 0, stall 0 |
| Cancel | 3/3: ack 1.1-1.3 ms; the terminal counters show exactly the 7 chunks accepted before the ack applied; 9/9 late chunks rejected `stale_session`; the wire froze at the ack (all 14 arm joints within 0.013 rad over the next 50 ms, against ~0.03 rad per 20 ms before it) |
| Halt | 3/3: body.halted RTT 0.64-0.78 ms (lane), `ArmChannel.latch` 0.18-0.30 ms, arms latched on the measured pose, hands at the measured q (closure at the halt vs 1 s later within 0.02: never opened); 9/9 late chunks rejected `halted`; the arm op ended `canceled`, ended_by `halt`, hold `measured`; after a halt or cancel the moving elbow carries on 0.13-0.24 rad (SONIC's lag + the 0.15 s lead the wire was ahead) and the servo brings it back towards the halt point within ~1 s (arm drift after 1 s: 0.045-0.069 rad) |

Earlier runs, same day: `20260929-075823-chunk` (before the correction preload and with trajectories starting from
SONIC's reference: the wire hit the slew limit at halts/cancels and at session starts), `20260929-080301-chunk`
(same results as the final run).

### 8.4 B.7 arm_script + CarryLock: a pick toward a real object, then a 2 m carry walk

Object: `RemoteControl|surface|2|30` on the bedroom dresser (`Dresser|2|1`, top 0.98 m, 0.127 m from the front
edge). Grasp point = its top centre + 3 cm (a hovering top grasp; no physical grasp: P1's attach arrives with wave 1).
`go_to` to 0.26 m from the edge (A*-safe), `approach` (body B.6) until the grasp point is ~0.34 m ahead of the pelvis
and 0.20 m right; 3 × (pregrasp, grasp (hand to 0.6), pregrasp again); then pregrasp, grasp, lift 6 cm, `approach` 0.25
m back with the lifted arm held (CarryLock), `carry` tuck, `turn_to` 180°, `walk` 0.3 m/s × 7 s, release, retract.
Palm error = FK of the measured arm (g1_debug) on the GT pelvis pose vs the grasp point, 50 Hz, last 1 s of each
grasp's settle, measured by the tool itself and by the body.

| Run | Grasp palm error to the grasp point (tool, 150 samples) | Per grasp p90 | Pelvis shift during a grasp | Carry walk (GT) | Palm drift while walking (vs the held pose) |
|---|---|---|---|---|---|
| `20260929-083319-pick` | median 1.8, **p90 2.1**, max 2.2 cm | 1.1, 2.2, 2.1 cm | 2.1-3.4 cm | 2.72 m, 0 falls | RMS 2.2, p90 2.9, max 4.6 cm |
| `20260929-084511-final-pick` | median 2.0, **p90 3.9**, max 4.2 cm | 1.9, 2.2, 4.2 cm | 3.5-4.7 cm | 2.36 m, 0 falls | **RMS 1.1, p90 1.6, max 3.5 cm**; CarryLock palm error 0.4 cm at the end |

- 5 of 6 grasps meet p90 < 3 cm; the sixth (4.2 cm) followed the largest pelvis shift (4.7 cm): the re-solved goal
  sat 0.37 m ahead and the extended arm sagged ~3 cm (G0 §3.4: "the palm ends 2-4 cm low"). The 3 cm bar is met by
  one run and missed by the other; a closer stance (grasp point ≤ 0.32 m ahead) is the obvious next step.
- What it took (each found live): the IK had to lock wrist pitch and yaw (SONIC barely tracks them); the pregrasp
  rises before it reaches (run `080443`: the first pregrasp hit the dresser front, 18 cm error) and the tuck goes
  back before down (runs `080443`, `082958`: the fist stopped on the dresser edge 15 cm high); A* keeps 0.25 m from
  furniture so the last 5-10 cm need `approach` (run `080443`: all grasps `ik_unreachable` at 0.43-0.52 m); SONIC's
  pelvis steps back 2-5 cm and yaws 1-3° while the arm reaches, so a goal solved once missed by 4.5-8.7 cm (run
  `082958`, pelvis-frame tracking 0.7-2 cm) and the settle now re-solves the world goal at 10 Hz.
- CarryLock = a `target` hold whose session closed a hand on purpose (`carry_arm`); the deploy-default fist filled in
  for the other hand does not count (run `080443` reported `engaged` with an open right hand before this fix).

### 8.5 B.5 waist scan

`scan {yaw_deg [-35, 0, 35], move 0.8 s, hold 0.8 s}`: waist yaw only, roll/pitch SONIC's, waist-yaw servo (ki 5).

| Final run `20260929-084744-final-scan` | Achieved yaw per hold (deg) | Max yaw error | Waist pitch (not commanded) | Arm joint deviation | Base shift | Falls |
|---|---|---|---|---|---|---|
| Standing, free arms (SONIC's reference, no arm servo) | −34.1, −0.9, +34.3 | 0.9° | 0.3-0.6° | 0.56 rad | 5 mm | 0 |
| Standing, arms held by the servo | −31.8, −1.1, +30.9 | 4.1° | 0.2-0.8° | 0.59 rad | 7 mm | 0 |
| At the dresser, free arms | −34.3, −0.7, +34.1 | 0.9° | 0.4-0.5° | 0.56 rad | 3 mm | 0 |

**Only the waist is commanded, but not only the waist moves.** Under a waist-yaw override SONIC's policy also turns
both shoulder yaws (+0.7-0.8 rad per rad of waist yaw, r 0.98) and the wrist rolls (+0.4-0.6), and moves the shoulder
pitches ±0.23 (fit on `20260929-080648-scan`), whatever the arms are sent: SONIC's own reference (0.56-0.65 rad),
a servo hold (0.59-0.62 rad), or a feed-forward cancelling the fitted coupling (0.63 rad, and less waist yaw; the
preset was dropped). The palms move 6-10 cm in the torso frame. The first run (`080648`) held the free arms at SONIC's
reference *with* the servo and dragged them 0.58 rad; free arms now stay on SONIC's live reference with no arm servo.
Consequence: no waist scan while carrying (CarryLock), or a smaller yaw.

### 8.6 Deviations from `docs/contracts/arm_chunk.md` v0.1 (for the integrator)

1. `lead_s` defaults to 0.15 (G0 condition 2), not 0.0; an explicit value wins (`groot_arms` sends 0.0 today).
2. A watchdog-ended session is `failed` (reason `client_silent`), not `succeeded`.
3. Rows are interpolated linearly at the continuous time index (the reported `k` is rounded).
4. `slew_frac` counts slew-limited **arm** values over ticks × 28 (hands are never slew-limited by the body).
5. `stop {arms: true}` ends a chunk session at once (`canceled`, ended_by `stop`) and blends back; its later
   messages are `stale_session`.
6. The halt latch holds the **measured** hands (the body-core/body-arm interface), not their last target.
7. Additive: `arm.progress` and terminal data carry `max_step_rad`; the terminal data also `slew_frac_total`,
   `cross_fades`, `lead_s`, `chunk_seq`, `kind`, the fence fields.

### 8.7 Reproduce

```bash
# box, M1 stack up, body on this code (scripts/m1_restart_body.sh); take the stack lock first
.venv/bin/python -m tools.arm_wave_test chunk --sessions 10 --cancels 3 --halts 3 --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-chunk
.venv/bin/python -m tools.arm_wave_test pick --trials 3 --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-pick
.venv/bin/python -m tools.arm_wave_test scan --at-counter --scan-hold-variants --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-scan
.venv/bin/python -m tools.arm_track_test --lead 0.15 --servo-model gated --phases static,steps --no-plots --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-servo
python -m tools.arm_wave_test chunk --fake --sessions 3 --cancels 1 --halts 1 --out /tmp/aw     # plumbing on the fakes
```

## 9. Wave 2 (2026-09-29, owner body-fix): the body-wave verifier's defects B-D1..B-D4 and B-low

Status: **built and run live on the dev box `ludo-g1-arena`** (procthor-train-38, unmodified deploy, P1 with the
rebuilt Dex3 G1 USD: `.asset_hash` 24d3c85f... → d3069de7..., `outputs/m2b_wave2/bodyfix/g1_asset_build.log`),
body code = the laptop tree pushed at 11:14 / 11:28 box time (md5 of `body/arm.py`, `body/arm_script.py` checked on
the box). 0 falls and 0 `command{stop}` in every run of this section. Code: `body/arm.py` (lock-free latch, wire lock,
ownership-aware latch/resume, `ScriptPending`), `body/ik_worker.py` (new), `body/arm_script.py` (prepare / solve /
finish, `clear_z`, `avoid_boxes`, `approach`, `preshape`), `body/g1_kin.py` (`hand_points`, `forearm_points`),
`body/carry.py`, `body/service.py` (deferred replies, the worker, `sys.setswitchinterval`), `body/velocity.py`,
`tools/halt_test.py` (`--script --chunk --carry --free-walks`), `tools/arm_wave_test.py` (`grasp`, `--clear`,
`--rise-gap`). Contract: `docs/contracts/m1.md` §3 v0.7, `docs/contracts/arm_chunk.md` v0.2. Evidence:
`outputs/m2b_wave2/bodyfix/` (laptop; the box has the same under `/work/worldline-g1/outputs/m2b_wave2/bodyfix/`).

**Validity (walk_diagnosis rules).** No other GPU job ran: the GR00T PolicyServer (tmux `groot-server`, 6.6 GB) was
loaded but idle, and no GR00T request can reach it while the dev stack lock is held (it was, owner `bodyfix`). P1
RTF over the runs: `rtf_total` 0.996-0.997, `rtf_1s_min` 0.825-0.868, 1.6 % of 1 s windows below 0.95. **P1 published
lowstate heartbeats during the runs** (`heartbeat_pubs` +3 / +20 / +5 in the three grasp runs, from a 2 Hz
`get_stats` poll, `p1_stats.jsonl`): P1 has a render hitch of 45-133 ms about every 30 s (`get_stats.hitches_last`,
head camera only, also with the robot standing still). By rule 3 these runs are therefore **INVALID as SONIC
timing-gate evidence**; the body-side measurements below (halt handling and receipt, the wire, the palm errors) do
not depend on SONIC's timing, and no fall or instability was seen. The hitch is a P1 issue (request to isaac).

### 9.1 B-D1: a halt waited behind the arm_script IK

Two causes, both confirmed:
1. `ArmChannel.latch()` took the arm lock, which `handle_arm_script` held for the whole IK (198-233 ms for an
   unreachable goal: every iteration of both DLS stages).
2. **The GIL convoy.** Even without the lock, a thread running `g1_kin.ik_palm` starves every other thread in the
   process: numpy's `linalg.solve` releases and re-takes the GIL every iteration, each re-take counts as a switch, so
   a waiting thread never gets to force one. Laptop probe (`outputs/m2b_wave2/bodyfix/gil_probe/`, a PULL thread's
   receive latency, 60 messages from another process): idle p50 0.47 / max 0.74 ms; the IK looping in the same
   process **p50 64.5 s**, p99 131 s; the IK in a spawn worker process p50 0.09 / max 0.15 ms. (FK at 50 Hz in the
   same process: p99 1.15 ms; a 40-iteration IK at 50 Hz: p99 6.2 ms.) So a worker *thread* would not have fixed it.

Fix: the IK runs in a worker **process** (`body/ik_worker.py`, one spawn-context process started with the service,
ready in 0.39 s live); `arm_script` prepares the goal on the control thread (cheap), submits the solve and answers the
ROUTER request when it is back (a `ScriptPending` the service polls every loop), re-checking the gate first; the
settle's 10 Hz re-solve goes to the worker too. `latch()` never takes the arm lock: under a wire lock held for
microseconds it sets the flag, freezes the override and queues the rest for the next tick (the tick checks for a
queued latch before it commits a pose). The service sets `sys.setswitchinterval(0.001)`.

Live, `tools/halt_test.py --walk 10 --script 20 --chunk 10 --carry 10 --free-walks 3`
(`outputs/m2b_wave2/bodyfix/halt-20260929-111624/`, `all_pass: true`):

| Halts | n | body handling (lane receive → publish) | client receipt (send → `body.halted` at the client) |
|---|---|---|---|
| during an unreachable-goal arm_script IK (377-429 ms in the worker; the halt 4-29 ms after the request; all 20 overlapped: the script answered `halted` after the solve) | 20 | p50 0.069, p99 0.128, max 0.136 ms | p50 0.51, max 0.86 ms |
| mid chunk session | 10 | p50 0.094, max 0.171 ms | p50 0.52, max 2.78 ms |
| mid walk (0.27-0.90 m/s at the halt) | 10 | p50 0.088, max 0.158 ms | p50 0.54, max 0.75 ms |
| on a CarryLock hold | 10 | p50 0.083, max 0.169 ms | p50 0.45, max 0.58 ms |
| with free arms | 1 | 0.082 ms | 0.42 ms |
| **all** | **51** | **p50 0.083, p99 0.17, max 0.171 ms** (bar p99 < 10) | **p50 0.51, p95 0.81, max 2.78 ms** (bar max < 30) |

Before (verifier `outputs/body_wave/bverify/20260929-095605-D/lock_live.json`): 176-221 ms receipts during the IK.
First IDLE planner message on the SONIC input after the send: p50 0.61, max 2.02 ms. Walks: 10/10 `canceled`
(halt), at rest (M1 E3 criterion) p50 0.86, max 1.18 s, travel after the halt p50 0.21, max 0.40 m. The arm latch
was applied by the next tick 4.0-18.8 ms after the lane (chunk trials, `latch.apply_ms`).

### 9.2 B-D2: the latch held the measured hand q

The latch copied `g1_debug.left/right_hand_q`, which lags a closed hand (an object, or just the finger PD), so every
halt commanded the hand a little more open (verifier: 0.9936 → 0.9461 over 10 halts), and it replaced a CarryLock
`target` hold by a `latched` one (CarryLock read disengaged). Fix: a halt keeps a hold **exactly** (pose, hands,
waist, carry), and a session it stops leaves the hands' **last target**. Live (same run): 10 halt/resume cycles on a
CarryLock hold (right hand closed to 0.99): the hand command on the wire identical in all 10 (latched and after the
resume), CarryLock engaged 10/10 latched and 10/10 after the resume, measured closure 0.985 before and after; 10
chunk sessions halted with hands closing to 0.6: the hand command after the halt equal to the last target 10/10.
Unit tests pin it with a plant whose hand stops at 93 % of its target (`test_repeated_halts_*`).

### 9.3 B-D3: a halt froze SONIC's free arms

The latch always installed a measured hold, even with no arm op, and resume kept it (9 later walks ran with frozen
arms). Fix: the latch acts on who owns the arms (a moving session → a latched measured hold; a hold → kept; a blend
→ paused; **free arms → left free**), and resume gives back exactly what it found. Live: a halt while standing with
free arms: `arms_latched` false, arm mode `off`, no override on the wire while latched; after the resume the arm mode
`off` and 3 walks with 0 % override on the wire, the shoulder pitches swinging 0.10-0.89 rad (3 walks before the
halt: 0.11-0.99 rad; the swing depends on the walk, not on the halt).

### 9.4 B-D4: scripted grasp palm error p90 4.5-4.9 cm

Root cause (geometry, then live): the palm target of the scripted "hover" grasp was the object top + 3 cm, but the
Dex3 hand is a slab ±4.4 cm thick about the palm origin (P1's collision meshes; the palm z axis is near vertical at
table-height reaches), so the goal put the hand 1.4-1.8 cm **into** the object: the palm stopped 1.4-2.8 cm high,
as the verifier saw. And the pregrasp, rising from the arm's rest pose at the A* stance (the pelvis ~0.23 m from
the dresser edge), swept the open fingers forward under the dresser top: live `grasp-20260929-112144-base`, the
measured palm stalled at world z ~0.95 under the 0.981 m top edge while the command went on up, SONIC stepped back
0.22 m (pelvis_shift 0.216 m, palm error 15-18 cm), and the next goals were out of reach (`ik_unreachable`).

Fixes (body): `arm_script` `clear_z` raises the palm goal until the hand's collision boxes (`g1_kin.hand_points`:
palm, finger and thumb boxes at the solved orientation and the hand pose the phase ends with) clear the surface by
`clearance_m` (1 cm); `avoid_boxes` checks the joint-space path (hand + forearm) against world boxes and searches via
points; the IK retries from SONIC's default arm when the current seed stalls. Tool sequence (`arm_wave_test grasp
--clear --rise-gap 0.40`): go_to 0.40 m from the support edge, raise the hand over the support there (`carry` phase,
`avoid_boxes` = the support), `approach` in with the arm held, then pregrasp / grasp with `clear_z` = the object top
and `avoid_boxes` = support + object.

| Run (`outputs/m2b_wave2/bodyfix/`) | Object / support (top) | Goal raise over top + 3 cm | Palm error to the commanded goal, per grasp p90 (tool: FK of g1_debug on GT pelvis, last 1 s) | body `palm_err_b` p90 (tracking) | Palm error to the naive point (top + 3 cm), p90 |
|---|---|---|---|---|---|
| `grasp-20260929-112922-clear` | Fork / CounterTop 0.937 | 5.1-5.3 cm | 1.82, 1.71, 1.82 cm | 1.67-1.78 cm | 6.6-6.8 cm |
| | RemoteControl / Dresser 0.981 | 3.4-3.8 cm | 1.21, 1.44, 1.35 cm | 1.22-1.32 cm | 4.1-4.9 cm |
| | CellPhone / DiningTable 0.741 | – | not run: the A* stance snapped away from the chairs and `approach` answered `too_far` | | |
| `grasp-20260929-113427-clear2` | Potato / CounterTop 0.937 (object top 1.04) | 2.5-2.7 cm | 1.91, 1.65, 1.30 cm | 1.26-1.78 cm | 3.3-3.8 cm |
| | RemoteControl / DiningTable 0.680 | – | not run: `approach` `goal_in_obstacle` (chairs) | | |

**9 grasps, 3 objects, 2 surfaces: per-grasp p90 1.21-1.91 cm; pooled p90 1.73 cm (300 samples, run 112922) and
1.65 cm (150 samples, run 113427); max 2.04 cm. The < 3 cm bar is met against the goal the script commands.** Honest
reading: that goal is 2.5-5.3 cm above "object top + 3 cm", by construction (the lowest collision-free hover of this
hand at this orientation); against the naive point the error is 3.3-6.8 cm and is the raise. SONIC's pelvis moved
0.3-1.4 cm during the grasps (was 2-5 cm), the mean palm error vector (pelvis frame) at most 1.3 cm in any axis.
0 falls. The raise phase itself: collision-free path found (`path.ok`) at 0.40 m on the counter; at the dresser the
checked path reported hits (the forearm margin is conservative) and the raise still succeeded. Low dining tables
surrounded by chairs are not reachable with this stance logic (A* snap + `approach` too_far / goal_in_obstacle).

### 9.5 B-low

| Item | Fix | Test / live |
|---|---|---|
| The latch / `measured` hold commanded the **measured** waist while sessions sent SONIC's reference: a 0.12-0.14 rad waist step at every arm / chunk halt | holds keep the waist as the session sent it (`ref` stays the live reference, `cmd` / a scan's yaw the values on the wire) | `test_latch_keeps_the_sessions_waist`, `test_measured_hold_on_end_keeps_the_waist_too`; live: waist change on the wire in the 1 s after 10 chunk halts p50 0.000, max 0.015 rad (SONIC's own reference moving) |
| v0.5 `end` during the watchdog hold reported `failed` / `client_silent` | the owner's `end` is a client end: `succeeded` | `test_v05_end_during_the_watchdog_hold_succeeds` |
| Chunk-session fence gaps (`release: true`, a non-chunk `end` skipped it) | every message acting on a chunk session's stream is fenced (`bad_args` without `session_id`, `stale_session` for another / older session or an ended one); a hold keeps its session's fence for `release` | `test_chunk_session_fence_covers_end_release_and_keepalive`; live: 30 chunks sent after 10 halt acks, 30 rejected `halted`, none on the wire |
| `stale_command` meant two things | the late-message case carries `data.why: "t_wall"` (+ `age_s`, `watchdog_s`) in the arm op and the velocity op; the fences' keep `why: control_epoch \| resume_epoch \| generation` (ops-groot's `groot_arms` already keys on it: `t_wall` is counted, not fatal) | `test_stale_t_wall_says_why` |
| m1.md §3.4 / §3.10 / §3.11 drift | re-checked against the code: body.state keys, the arm ops in the op table, the reasons list (`planner_frame_unknown`, `not_stopped`, `stale_goal`), which refusals get events, the `body.stale_command` payload, the halt steps and payloads | `docs/contracts/m1.md` v0.7 |
| B.5 pitch row impossible | documented: SONIC does not move waist pitch under the override (G0), the scan is yaw-only (`pitch_deg` ≠ 0 is `bad_args`), and the arms turn with the waist (§8.5) | m1.md §3.9 `scan`; the `docs/M2.md` B.5 row needs the lead's edit |

### 9.6 Regressions

`body/tests`: 111 passed in `.venv-rt` (the cv2 camera test deselected; 92 before), 19 of them new. The live regression chain
(pick + CarryLock walk with the new stance, `m1_drive_test`, `halt_test --walk 20 --arm 5`, `arm_wave_test chunk
10/3/3`) was started at 11:38 box time (`outputs/m2b_wave2/bodyfix/reg-20260929-113846/` on the dev box) and **was
cut off**: both Brev boxes went to STOPPED at ~11:45 (not by this owner); its results were not pulled. The halt run
of §9.1 covers the body-wave halt suite's walk trials (10 walks, all pass) and chunk halts; the chunk-cancel and
drive-test regressions are still owed.

### 9.7 Reproduce

```bash
# dev box, stack lock held, M1 stack up (scripts/m1_up.sh --session <owner>-m1)
.venv/bin/python -m tools.halt_test --walk 10 --arm 0 --script 20 --chunk 10 --carry 10 --free-walks 3 --out outputs/m2b_wave2/bodyfix/halt-$(date +%Y%m%d-%H%M%S)
.venv/bin/python -m tools.arm_wave_test grasp --list --out /tmp/gl                          # candidates in the house
.venv/bin/python -m tools.arm_wave_test grasp --objects "Fork|surface|6|3,RemoteControl|surface|2|30,Potato|surface|6|15" --trials 3 --clear --rise-gap 0.40 --out outputs/m2b_wave2/bodyfix/grasp-$(date +%Y%m%d-%H%M%S)
.venv/bin/python -m tools.arm_wave_test pick --object "RemoteControl|surface|2|30" --clear --rise-gap 0.40 --out outputs/m2b_wave2/bodyfix/pick-$(date +%Y%m%d-%H%M%S)
python outputs/m2b_wave2/bodyfix/gil_probe/gilprobe.py idle inline process                   # laptop: the GIL convoy
```
