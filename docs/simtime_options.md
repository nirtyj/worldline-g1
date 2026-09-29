# Running SONIC and the stack on sim time (study, 2026-09-29)

Status: **read-only study, no code changed.** It answers the owner's question: "Why aren't we running everything on
sim time? Why is SONIC running on the real clock? Wouldn't that solve the dropped clock cycles?"

Sources (two read-only investigations, merged here):

- `DEPLOY.cpp` = `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` in the WBC repo
  (GR00T-WholeBodyControl @ b042411, dev box `ludo-g1-arena:/work/repos/GR00T-WholeBodyControl`).
  `INC/` = the `include/` dir next to it. `SDK` = `gear_sonic_deploy/thirdparty/unitree_sdk2`.
- All other paths are in this repo (`worldline-g1`).
- ED = engineer-days, as in PLAN.md.

---

## 0. Answer in brief

**Why SONIC runs on the wall clock today.**

1. `gear_sonic_deploy` is the real-robot binary and has no sim-time mode. It runs four periodic threads (input
   100 Hz, control 50 Hz, planner 10 Hz, command writer 500 Hz) on kernel timers (`timerfd` on `CLOCK_MONOTONIC`)
   inside the prebuilt `libunitree_sdk2.a` (`DEPLOY.cpp:11-21, 2185-2188, 2613-2621`). It stamps every input with its
   own receive time and never reads `LowState_.tick` (`INC/utils.hpp:59-97`). No flag or config changes this.
2. We chose to run it unmodified for real-robot parity (PLAN.md:154, decision #19: "Wall clock ... rejected:
   sim-time lockstep"). M1 measured that this works on a quiet box: RTF 0.994-0.995, 0 falls (docs/M1.md:160-176).
   Lockstep was kept as the documented contingency (PLAN.md:31 §0.2, PLAN.md:1679-1690 §7.3.6, docs/M1.md:174-176).
3. Upstream's own sim2sim loop (MuJoCo `base_sim.py:598-637`) also sleeps to the wall clock. NVIDIA supports
   lockstep only in the Python decoupled WBC, not in the SONIC C++ deploy.

**What actually "drops".** The deploy does not drop ticks: `g1_debug.index` advanced exactly 50/s through every fall
(docs/contracts/sonic_deploy.md:263-266). Isaac (P1) is what stalls. The deploy keeps ticking on the wall clock, so
while P1 is behind:

- control ticks run on the same (stale or heartbeat) lowstate, so the policy's 10-tick history fills with duplicates
  and the reference-motion cursor runs ahead of a frozen robot by (stall / 20 ms) frames;
- when P1 catches up, several physics steps share one leg target, and some 20 ms sim windows get none and others two.
  This is the "irregular windows" metric: 39-57 % with GR00T on the main box, against a bar of 15 %
  (PLAN.md:42 §0.12).

This mechanism is read from the code; docs/walk_diagnosis.md:18 records contention -> falls as "correlation verified,
mechanism hypothesis".

**What sim time (lockstep) would fix.** If P1 and the deploy run in lockstep (one control tick per 20 ms of sim
time, always on fresh state), stale-state ticks, irregular windows, lowstate heartbeats and the 500 ms "lowstate
absent" exit all go away by construction. Box contention then makes the run **slower** (RTF < 1), not **worse**. RTF
becomes a throughput number instead of a validity condition, and GR00T could share the main box again in that mode.

**What it costs.**

1. **A patched deploy.** About 100-150 C++ LOC in `DEPLOY.cpp` behind a new `--sim-clock` flag, plus a P1 barrier
   of about 100-150 Python LOC. It is small because the policy, encoder, history and planner math already count in
   ticks, not seconds (§1). Effort 3-6 ED (option B).
2. **"Everything" is much more than SONIC.** The body service (P3), the GR00T chunk indexing and the runtime clock are
   wall-clock too. With only P1<->P2 in lockstep, at RTF 0.85 they run about 15 % fast relative to physics. Moving
   them onto sim time is another 8-11 ED (option D, 11-17 ED in total).
3. **Model latency stays wall time.** GR00T and Gemini answer in wall time. Only pausing physics during model calls
   (Arena's model) takes that out of sim time, and that breaks the live voice demo and hides latency the real robot
   will see.
4. **Parity.** The claim changes from "the unmodified real-robot binary" to "upstream control code with a sim-clock
   scheduler". Wall-clock runs have to stay as the parity check.
5. **Contention is not removed.** The deploy's compute and a DDS round trip join P1's critical path every 20 ms, so
   the maximum RTF drops a little.

**Short answer to the question:** yes for SONIC's part. Lockstep removes the class of failure where a stall or burst in
Isaac leaves SONIC acting on stale state. It does not make the rest of the stack sim-timed, and it costs the
"unmodified binary" claim. Recommendation (§4): for the POC, stay on the wall clock unless the pending timing-gate
re-measure with the remote PolicyServer fails. Longer term, build a switchable `clock: wall|lockstep` mode, starting
with the flag-gated deploy patch validated in the MuJoCo reference loop.

---

## 1. The deploy's timing model

Every place where the deploy depends on time, and what a sim-clock build (option B) would have to change.

| # | Dependency | Citation | Today | Change for a sim-clock build |
|---|---|---|---|---|
| 1 | Thread scheduler | `DEPLOY.cpp:2185-2188, 2613-2624` | `CreateRecurrentThreadEx(UT_CPU_ID_NONE)`: input 10 ms, command writer 2 ms, control 20 ms, planner 100 ms (only when a planner is loaded). `publish_dt_` 0.002, `control_dt_` 0.02, `planner_dt_` 0.1, `input_dt_` 0.01 | With the flag on: do not create the control, planner and writer threads. One new sim-tick thread runs them instead (row 7, row 10). Input keeps its own 100 Hz wall thread (row 9) |
| 2 | How `RecurrentThread` keeps time | SDK `include/unitree/common/thread/recurrent_thread.hpp:12-79`; `lib/x86_64/libunitree_sdk2.a`, `RecurrentThread::ThreadFunc()` at +0x92..0x9c (`timerfd_create`, edi=1), +0x17f (`timerfd_settime`), +0x1d4 (call), +0x1f0 (`read`) | Checked by disassembly: a periodic `timerfd` on `CLOCK_MONOTONIC`; loop = `mFunc(); read(fd)`. Missed expirations merge: after an overrun the next tick runs at once and the rest are dropped. No burst, no catch-up | There is no clock to inject (prebuilt library). The change has to be at the call sites in `DEPLOY.cpp` |
| 3 | Lowstate intake | `DEPLOY.cpp:2293-2297, 2645-2686, 2842-2847, 2968-2987`; `INC/utils.hpp:59-97`; SDK `idl/hg/LowState_.hpp:31` | DDS listener, queue length 1. `LowStateHandler` copies the message and calls `low_state_buffer_.SetData()`, which stamps `steady_clock::now()`. Control reads the newest buffer each tick. `LowState_.tick` is never read. P1 already writes `tick` = sim ms (`sim_isaac/dds_bridge.py:276`) | In `LowStateHandler`, after `SetData`: if `floor(tick / 20 ms)` changed, store the tick and notify a condition variable. An edge on the tick tolerates heartbeats and dropped samples, unlike counting every 4th message |
| 4 | Hard watchdog: lowstate absent 500 ms | `DEPLOY.cpp:299, 2797-2811, 3837-3877, 4523-4534`; INIT waits at `2719-2752` | `CheckSafety()` fails if newest lowstate is > 500 ms old (wall), in WAIT_FOR_CONTROL and CONTROL. Failure -> `stop` -> threads joined, one damping command (kp 0, kd 8), **process exits**. This is the only wall timer that kills the process | In lockstep a control tick only runs on fresh lowstate, so sim-time staleness is 0 by construction. Replace the 500 ms test with a wall "simulator dead" timeout set by a flag (10-30 s) |
| 5 | Soft thresholds (warnings only) | `DEPLOY.cpp:290, 298, 359-360, 3008-3031, 3074-3110, 4076-4100`; `INC/output_interface/zmq_output_handler.hpp:211-228, 242` | `LOW_STATE_LATE` 50 ms and `STREAMING_DATA_ABSENT` 150 ms feed only the audio/TTS warnings. `TOKEN_TIMEOUT_MS` 200 ms warns only in external-token mode (not ours). `robot_config` re-sent every 2 s. "Loop timing" line is informational | None |
| 6 | Control math counts ticks (the key enabler) | `DEPLOY.cpp:2474-2476` + `src/state_logger.cpp:199-235` (history); `DEPLOY.cpp:3337-3360` (frame advance); `427-455, 616-690, 740-817` (encoder look-ahead); `2189, 2776-2777` (INIT ramp); `3411, 3695` (replan counter); `INC/localmotion_kplanner.hpp:215, 557, 395-520, 644-690` (planner frames) | The 10-frame histories are the last 10 control ticks (exact-stride path of `StateLogger::GetLatest`). `current_frame_ += 1` per tick. Encoder look-ahead reads frames +0..+45 (0.9 s in frame units). INIT ramp adds `control_dt_` per tick (3 s = 150 ticks). Replan counter adds `planner_dt_` per call. Planner `gen_frame_ = current_frame_ + 2`; 30->50 Hz resampling in frame units | **None.** Only *when* `Control()` and `Planner()` fire changes. No observation, encoder, policy or planner code is touched |
| 7 | Planner thread races control | `DEPLOY.cpp:229-234, 3595-3833, 3184-3285` (cross-fade `3218-3270`); `INC/localmotion_kplanner.hpp:533-589` | 10 Hz thread. Reads `current_frame_` without the motion lock, runs TensorRT (3.0 ms p50, 8.7 ms worst), resamples. Control picks the result up at its next `CurrentFrameAdvancement` and cross-fades over 8 frames from `max(0, gen_frame - current_frame_)`. How many control frames pass during inference depends on wall timing | Call `Planner()` synchronously every 5th sim tick, before `Control()`. Deterministic; the plan lands in the same tick instead of usually the next one. A small difference to check by A/B (§4) |
| 8 | ZMQ planner input | `INC/input_interface/zmq_manager.hpp:26-30, 104-145, 254-266, 578-645, 885-913, 1198-1200`; `zmq_packed_message_subscriber.hpp:192, 234, 373` | Two receive threads, 100 ms receive timeout, HWM 3. Messages stamped `steady_clock::now()`. Input thread uses each once. After 1000 ms (wall) with no message: force IDLE, clear upper-body and hand flags. On a switch into PLANNER mode it only trusts a message newer than 100 ms. `upper_body_position` goes straight into a buffer: the arm override is whatever is newest when the control tick reads it | None for the minimal patch (P3 sends at 50 Hz wall, so at RTF <= 1 every sim second still gets >= 50 messages). Optional: pass a clock into `ZMQManager` to put the 1 s timeout on sim time (~30 LOC) |
| 9 | `command{start}` blocking wait | `INC/input_interface/zmq_manager.hpp:343-362, 470-505, 527-575` | `start` blocks the Input thread in 100 ms sleeps until the control thread has switched to `planner_motion`. `PLANNER_INIT_TIMEOUT` 5 s wall -> `stop` -> exit. `stop` exits the process; a repeated `start` is ignored | Input **must keep its own thread**: folded into a single tick thread, this wait deadlocks. P1 must keep stepping during the wait. The 5 s would only trip below RTF ~0.05 |
| 10 | Command writer, lowcmd, Dex3 | `DEPLOY.cpp:2694-2716`; `INC/dex3_hands.hpp:115-197`; SDK `idl/hg/LowCmd_.hpp:26-30` | `LowCommandWriter` (500 Hz wall) packs the newest `motor_command_buffer_` into `LowCmd_` and calls `dex3_hands_.writeOnce()` (clip to max-close ratio, clamp +-0.25 rad from the newest hand state). `LowCmd_.reserve_[4]` (uint32) is never set | In sim mode: one `LowCommandWriter()` call per control tick (this also writes Dex3), with `reserve_[0]` = the lowstate tick it answers. That is the lockstep ack, with no IDL change |
| 11 | `g1_debug` and dumps | `INC/output_interface/zmq_output_handler.hpp:161-197, 247-251, 293-294`; `src/state_logger.cpp:104-107, 277-287`; `DEPLOY.cpp:2172, 3499-3575, 4059-4067, 4166-4190` | Sent every control tick; `index` = StateLogger entry = control ticks. Built-in dumps: `--policy-input-logfile` (436-D obs every tick), `--planner-motion-logfile`, `--target-motion-logfile`, `--enable-csv-logs` | None. In lockstep, `index x 0.02 s` is sim time, free for P3 and GR00T. The dumps are the tick-by-tick reference for validation |
| 12 | Main thread, CPU placement | `DEPLOY.cpp:2632-2640, 4523-4526` | `main()` loops on `sleep(0.02)` = `sleep(0)` (unsigned int): a busy spin. `SetThreadPriority()` pins only the main thread to CPU 0 | None (relevant only to the CPU budget) |
| 13 | Planner precision | `DEPLOY.cpp:4173, 4222-4223, 4355-4366`; `sonic/run_deploy.sh:27` | Usage text says FP16 default, code initialises `plannerFp16 = false`; we pass no flag, so the planner runs FP32 TensorRT | None (relevant to option C numerics) |

Per-tick CONTROL pipeline (`DEPLOY.cpp:3834-4110`), unchanged by a sim-clock build: `CheckSafety` ->
`GatherRobotStateToLogger` (one ring entry) -> `GatherInputInterfaceData` (17-D upper body, 7-D hands) -> under the
motion lock `UpdateHeadingState` + `GatherObservations` (encoder, TensorRT, inline) -> `CreatePolicyCommand` (decoder;
`q_target = default_angles + action * g1_action_scale`) -> Dex3 `setAllJointsCommand` -> publish `g1_debug` -> dumps
-> `CurrentFrameAdvancement`. Measured: decoder 0.1 ms p50 / 6.2 ms worst; obs->cmd p90 0.4-0.5 ms quiet, 3-7 ms
under load (docs/contracts/sonic_deploy.md:263-266; docs/walk_diagnosis.md:18).

**Not a way out:** an `LD_PRELOAD` time-dilation tool (libfaketime) on the unmodified binary was not tested. Even if it
slowed the `timerfd` intervals and `steady_clock`, it only matches rates; it has no per-tick barrier and cannot absorb
a hitch.

---

## 2. Options

| | What | Effort (ED) | Fixes SONIC timing? | P3 / GR00T timing at RTF < 1 | Parity cost |
|---|---|---|---|---|---|
| A | Keep the wall clock, isolate load | 0-1 | No (avoids it) | Correct (all wall) | None |
| B | Sim-clock deploy patch + P1<->P2 lockstep handshake | 3-6 | Yes, by construction | Stretched by 1/RTF | Low-moderate, flag-gated |
| C | SONIC re-implemented in P1 (PLAN §7.3.6) | 10-16 (+8-11 for the rest) | Yes | Stretched unless D's work is done too | High: not the robot binary |
| D | Switchable `clock: wall\|lockstep` across the stack (B + P3, GR00T, runtime, referee) | 11-17 | Yes | Correct in sim time | Wall mode keeps full parity |

### A. Keep the wall clock and remove the contention (status quo)

- **What changes:** nothing in SONIC. The GR00T PolicyServer stays on the dev box behind the OD3 tunnel
  (PLAN.md:42 §0.12). CPU pinning for the deploy, a render budget (`session_head_hz`, `frame_sync` in
  `services/executors/groot_arms.py:180-196`), and the per-run timing gate (walk_diagnosis rule 3: INVALID if
  irregular > 0.15, RTF < 0.98, or heartbeats during motion; `tools/groot_timing_gate.py`).
- **Effort:** 0-1 ED. The re-measure with the remote server is already planned (PLAN §0.12).
- **Risks:** the GR00T client, the `ego_view` render and the 0.92 MB request encoding still load the main box and may
  still fail the gate. Heavier houses or more cameras eat the ~29 % free-running headroom (docs/M1.md:167). INVALID
  runs cost wall time. Nothing is fixed structurally.
- **Parity cost:** none. The unmodified binary, DDS topology and threads match the real G1.

### B. Sim-clock patch of the deploy with a lockstep handshake (P1<->P2 only)

**Deploy** (`DEPLOY.cpp` only, new flag `--sim-clock`, off by default; about 100-150 C++ LOC):

1. `LowStateHandler`: after `SetData`, if `floor(LowState_.tick / 20 ms)` changed, store the tick and notify a
   condition variable (§1 row 3).
2. A new sim-tick thread waits on it and runs, in order: `Planner()` on every 5th tick, `Control()`, then one
   `LowCommandWriter()` with `LowCmd_.reserve_[0] = tick` (this includes the Dex3 `writeOnce`).
3. The control, planner and command-writer `RecurrentThread`s are not created in sim mode. Input stays a 100 Hz wall
   thread, so the `command{start}` wait cannot deadlock (§1 row 9).
4. `CheckSafety` in sim mode uses a wall "simulator dead" timeout from a flag (10-30 s) instead of 500 ms.
5. `Stop()` notifies and joins the new thread.
6. Optional: the `ZMQManager` 1 s planner timeout on sim time (~30 LOC).

Carry it as a patch file that `sonic/build_deploy.sh` applies, so the WBC checkout stays pinned at b042411 and clean;
the build tooling already exists (`sonic/build_deploy.sh:1-20`) and TensorRT engines are reused after a hash check
(docs/contracts/sonic_deploy.md:267-269). With the flag off, the binary behaves as today.

**P1 (`sim_isaac/`, about 100-150 Python LOC), flag `--lockstep`:**

- Extend `FastReader` to decode one uint32 field (`reserve[0]`); today it extracts only float fields
  (`sim_isaac/fastdds.py:151-205`).
- On each 20 ms boundary step: publish odostate, secondary IMU and Dex3 state first and lowstate **last**
  (today lowstate is written first, `sim_isaac/dds_bridge.py:303-304`), so the torso IMU and hand state for that step
  are already in the deploy's buffers when the tick fires.
- Then block until the lowcmd with `reserve[0] == tick` arrives, while still serving `gt.poll` RPCs. On a wall
  timeout (1-2 s): band on, fall back to free-running, emit an event, so a dead or restarting deploy (G8) is handled.
- Heartbeat off in lockstep (`sim_isaac/dds_bridge.py:174-182, 309-318`).
- `RtPacer` (`sim_isaac/rt_pacer.py:64-119`) stays as a "never faster than wall time" cap at 1.0 for demos, because
  P3, GR00T and voice are on the wall clock; it can be off (`--no-rt-pace`) for eval-only runs.
- Optional: the same barrier in `sonic/mujoco_ref/sim_ref.py` (a `--pace lockstep` next to `deadline` and
  `upstream`, `sim_ref.py:427-433`) for the A/B against the MuJoCo reference.

- **Effort:** 3-6 ED. The two investigations estimated 2-4 ED and 5-8 ED; the difference is how much patch debugging
  and validation is budgeted. Parts: C++ patch and rebuild 0.5-3; P1 barrier and fallback 0.5-2; MuJoCo barrier 0.5;
  validation 1-2 (§4).
- **Risks:**
  - A patch on a vendored binary, re-applied on every upstream bump.
  - Scheduling differs slightly: the planner is synchronous and tick-aligned (on the robot it lands asynchronously
    within 100 ms); lowcmd and Dex3 are written at 50 Hz instead of 500 Hz.
  - P3 commands still enter through the wall-clock Input thread, so their timing in sim time is not deterministic and
    runs are not replayable.
  - The start wait deadlocks if anyone moves Input into the tick thread.
  - The deploy's compute and the DDS round trip join P1's critical path: about 1-2 ms per 20 ms quiet; under load
    obs->cmd p90 is 3-7 ms plus up to 8.7 ms of planner every 5th tick. This lowers the maximum RTF but no longer
    harms control.
  - P3, GR00T chunk playback and voice stay on the wall clock, so at RTF < 1 they are distorted (§3). P3's 0.5 s and
    1 s staleness watchdogs still fire on any sim pause longer than that.
  - Results become more optimistic than the real robot, which has real latency.
- **Parity cost:** low to moderate. Control, observation, encoder, policy and planner code are identical; only thread
  scheduling and the 500 ms watchdog differ, and only with the flag on. DDS topics and the ZMQ API are unchanged, so P1
  and P3 still exercise the real-robot interface. `docs/parity.md` must name the mode of every result.

### C. In-process lockstep SONIC (PLAN §7.3.6)

- **What changes:** re-implement the deploy inside P1's physics loop (decimation 4), keep the 5556/5557 surface for P3.
  No upstream Python version of the deploy's planner integration or observation assembly exists: `gear_sonic/` has
  only ONNX export helpers and the Isaac Lab training observation terms
  (`gear_sonic/utils/inference_helpers.py:22-451`; `gear_sonic/envs/manager_env/mdp/observations.py:922-1060`).
  Components (Python LOC estimate, C++ source):
  1. Encoder, decoder, planner V2 runners (planner V2 takes 11 inputs incl. a 4-frame context, token count, seed):
     100-150 (`INC/encoder.hpp`, `control_policy.hpp`, `localmotion_kplanner_tensorrt.hpp:26-27, 88-104, 190-210, 353-377`).
  2. State ring and the five 10-tick histories (zero padding at start, gravity from the quaternion, isaaclab order,
     default-angle offset): 100-150 (`DEPLOY.cpp:1454-1647, 2842-2990`; `state_logger.cpp:180-235`).
  3. Encoder observations, mode g1 layout, all 14 entries, 17-D upper-body override, mode fallback: 150-250
     (`DEPLOY.cpp:427-1000, 2022-2100`).
  4. Heading state and quaternion helpers: 80-120 (`DEPLOY.cpp:559-615`).
  5. Planner wrapper (context, yaw normalisation, 30->50 Hz resampling, `gen_frame + 2`): 250-350
     (`INC/localmotion_kplanner.hpp:280-709`).
  6. Replan scheduler with exact float-compare triggers: 80-120 (`DEPLOY.cpp:3595-3833`).
  7. Frame advance, 8-frame cross-fade, IDLE readaptation: 120-180 (`DEPLOY.cpp:3184-3437`).
  8. Action -> PD, gains, INIT ramp, WAIT/CONTROL state machine: 100-150 (`DEPLOY.cpp:2762-2811, 3137-3165, 3834-3880`).
  9. 5556 ZMQ adapter with `zmq_manager` semantics: 200-300.
  10. 5557 `g1_debug` / `robot_config` publisher with the keys P3 reads: 100-150.
  11. Dex3 clip and clamp: 50-80 (`INC/dex3_hands.hpp:115-300`).
  12. Parity harness diffing against the binary's `--policy-input-logfile`, `--planner-motion-logfile`,
      `--target-motion-logfile`: 250-400.

  Total about 1,600-2,400 Python LOC.
- **Effort:** 10-16 ED for SONIC alone (PLAN estimated 10-15): port 5-8, parity chasing 3-5, Kit integration
  (GPU inference in the physics thread, GIL, onnxruntime or TensorRT inside Isaac's Python) 1-2, soak 1. P3, GR00T and
  runtime still need D's clock work (8-11 ED) for RTF != 1 or pauses.
- **Risks:** fidelity (exact float-compare replan triggers, a chosen planner/control schedule, zero-padding semantics,
  quaternion conventions, encoder layout and mode fallback, onnxruntime vs TensorRT numerics, the upper-body override
  that decision (b) depends on). PLAN §7.3.6 cites upstream issue #233 as evidence that observation assembly is
  fragile. Every upstream deploy change must be re-ported. Inference in Kit adds 1-10 ms per tick to the physics thread.
- **Parity cost:** high. The sim no longer runs the robot's code, and the DDS/Unitree topology is no longer tested.
  PLAN.md:1690 requires `docs/parity.md` to say so.

### D. Both modes, switchable (everything on sim time in lockstep mode)

- **What changes:** one profile key, `clock: wall|lockstep`. It selects:
  - P1 `--lockstep` (barrier, heartbeat off, RtPacer cap 1.0 or off) and the P2 binary (unmodified, or patched with
    `--sim-clock`): all of B;
  - a P3 `Timebase` (monotonic, or `g1_debug.index x 0.02`), used only by physics-time code: motion durations, servo
    dynamics, settle windows, path-follower phase limits, approach time constants, arm servo delay. About 40-60 of the
    ~100 time calls in `body/`; liveness timers stay wall;
  - GR00T chunk `t0` in sim time (from `g1_debug.index x 0.02`; frames already carry `t_sim`,
    `sim_isaac/cameras.py:125`), `lead_s` in sim time, freshness and expiry as sim age; the arm_chunk contract gets a
    timebase field;
  - the runtime clock split in two: a robot clock read from P1's `t_sim` (`gt.pose` already carries it,
    `sim_isaac/app.py:477`) for action timeouts and ages, and a model clock that stays wall;
  - `sim_health` gating off, G9 not applicable, referee deadlines in sim time with a wall backstop, the UI showing
    `t_sim` and RTF side by side;
  - optional, off by default: pause physics during GR00T inference (Arena-style) for deterministic evals.
- **Effort:** 11-17 ED = B 3-6 + P3 timebase 3-4 + GR00T chunks 1.5-2 + runtime clock split 2-3 + referee/UI/health
  1.5-2. (A narrower cut covering only B + P3 + GR00T was estimated at 5-9 ED.)
- **Risks:** a refactor across about 100 wall-clock call sites, and two clock semantics to test in every scenario.
  Tuned constants (FacingServo push 0.6, arm servo ki 2.0/s with a 0.15 s delay, pure-pursuit lookahead, approach
  time constants; `body/config.py:99-125`) must be re-checked. RTF > 1 buys little: free-running headroom is ~1.3x on
  a quiet box. The pause knob makes GR00T evals latency-free and optimistic, like Arena.
- **Parity cost:** wall mode keeps full parity and is the real-robot parity and soak mode. Lockstep runs carry B's
  parity cost. With the pause knob on, GR00T latency behaviour no longer matches the robot.

---

## 3. The rest of the stack under sim time

"B" = P1<->P2 lockstep only; "D" = the full switchable mode. "Stays wall" = liveness, IPC or model time that should
never move to sim time.

| Component | Clock today (citation) | Under B | Under D |
|---|---|---|---|
| P1 pacing, `RtPacer` | Wall: sleeps when ahead, bursts when behind, drops backlog > 100 ms (`sim_isaac/rt_pacer.py:64-119`; `app.py:77-78, 307, 667`) | Barrier paces; RtPacer only as a 1.0 cap. `overruns`/`lost_s` stop meaning anything | Same; off for eval |
| P1 lowstate heartbeat | Wall, 50 ms (`sim_isaac/dds_bridge.py:174-182, 309-318`; `app.py:55`) | Off (it would hide a dead P1 main loop; a re-publish carries the same tick, so it cannot trigger a control tick) | Off |
| P1 lowcmd intake | Newest sample each step, zero-order hold (`dds_bridge.py:221-270`; `fastdds.py:151-205`) | Wait for `reserve[0] == tick` on boundary steps; uint decode added | Same |
| P1 cameras, recorder, `gt.pose`, viz | Already on the sim grid: `rig.due(t_sim)`, `gt.pose` every 4 steps (`app.py:477, 582-643`; `cameras.py:87-100`) | Unchanged; renders only lower RTF | Unchanged |
| P1 TTLs, `sim.health` 1 Hz, housekeeping | Wall (`app.py:609-667`; `cameras.py:132-141, 160, 222`; `test_ops.py:198-209`) | Stays wall | Stays wall |
| P1 heavy GT ops (occupancy, top-down render) | Refused unless band on (`app.py:752-765, 818-828`) | Block the sim and SONIC together, safely; but P3's staleness watchdogs fire (below) | Allowed without the band once P3 staleness is in sim time |
| P3 `SonicMux` 50 Hz keepalive, `cmd_stale_s` 0.3 | Wall (`body/sonic_mux.py:240-255, 333-366`; `body/config.py:50-51`) | Fine at RTF <= 1 (>= 50 messages per sim second) | Send once per sim tick (on `g1_debug`); `cmd_stale_s` stays wall |
| P3 motion and servo timing | Wall, `time.monotonic` inside `_tick` (`body/service.py:880-915`; `motions.py:63-91, 139-156, 372-406`; `path_follower.py:139, 185, 250, 265`; `approach.py:110-127`; `velocity.py:94-119`; `arm.py:1379-1409, 1599-1617`) | Stretched by 1/RTF: at RTF 0.85, servo dead time ~15 % off and timeouts ~15 % shorter in sim terms | `Timebase` from `g1_debug.index x 0.02` |
| P3 staleness watchdogs: `pose_stale_s` 0.5, `debug_stale_s` 1.0, `deploy_lost_s` 1.0 | Wall age since receipt (`body/config.py:66-67, 117-121`; `service.py:664-701, 767-773`; `deploy_monitor.py:113-124`) | Any sim pause > 0.5-1 s fails the motion or engages the band | Split: staleness in sim time (always fresh in lockstep); liveness "no new sim tick for N s" stays wall |
| P3 velocity estimate | Already uses `t_sim` differences (`body/p1_client.py:157-170`) | Correct | Correct |
| P3 halt lane (30 ms receipt) | Wall IPC (`body/halt.py:1-69`; `robot/body_client.py:53, 132-139, 522-545`) | Stays wall; physical stop lands at the next sim tick | Same; judge "stopped within 1.2 s" in sim time |
| P3 session watchdog 1.0 s, `t_wall` message checks | Wall (`body/fence.py:236-273`; `body/arm.py:714-731`; `body/velocity.py:94`) | Stays wall | Stays wall |
| GR00T chunk indexing: `t0_mono`, `lead_s` 0.15 | Wall (`services/executors/groot_arms.py:64-66, 193, 641-660`; `body/arm.py:873-890, 1511-1516`; `docs/contracts/arm_chunk.md:31, 41-44`) | Rows play 1/RTF fast relative to physics (~15 % at RTF 0.85) | `t0` in sim time; `lead_s` in sim time |
| GR00T freshness: frame 0.15 s, state 0.06 s | Wall age (`groot_arms.py:184-189, 540-560`) | A sim pause makes observations look stale (spurious `obs_stale` drops) | Sim age |
| GR00T RPC timeouts 1.5 s / 0.5 s, stream watchdog 2.0 s | Wall (`groot_arms.py:165-166, 578-583`; `body/arm.py:817-861`) | Stays wall | Stays wall |
| GR00T model latency (~100+ ms) | Wall | Real latency, measured in sim time (the robot-like behaviour) | Same; optional pause knob removes it (Arena model) |
| Runtime `SimClock` | `time.monotonic x speed`, not linked to P1; live stack runs `SimClock(1.0)` (`sim/clock.py:1-51`); timeouts in `agent/harness.py:69-80`, `agent/persona.py:41-47`, `services/navigation.py:272-290` | Robot timeouts (SENSE 12, PICK 25, ...) shrink in sim terms at RTF < 1 | Two clocks: robot clock from P1 `t_sim`, model clock wall (the clock is already injected) |
| Gemini Live / System 1 | Wall (`brains/system1.py:59-61, 358-378`; `ui/server.py:866-925, 954`) | Stays wall; the 3 s own-action grace can expire before the sim shows the change | Stays wall; express the grace in sim time |
| UI | Wall 0.2 s tick (`ui/server.py:75, 399-442`) | Stays wall | Stays wall; show `t_sim` and RTF |
| `sim_health` DEGRADED (< 0.95 for 5 s) / UNSAFE (< 0.85 for 3 s) | Wall-controller protection (`world/sim_health.py:1-7, 55-60`) | SONIC no longer needs it; keep DEGRADED gating `manipulate` while GR00T chunks are on wall time, and UNSAFE as P3 protection | Off in lockstep |
| Eval referee | `time.monotonic` deadlines (`eval/suite.py:95-125`; `eval/stack_suite.py:151-176, 650-659, 759-771`) | Runs below RTF 1 can hit wall deadlines | Deadlines in sim time with a wall backstop; G9 (throttle to RTF 0.9) not applicable |
| Timing gate | walk_diagnosis rule 3; `tools/groot_timing_gate.py` | `irregular = 0` and `heartbeats = 0` become assertions that the barrier works; RTF is reported, not gated | Same |
| Nav2 (deferred) | `use_sim_time: false` (`nav2/params/nav2_g1.yaml:18, 37, 105, 132, 154`) | n/a | If revived: P1 publishes `/clock`, `use_sim_time: true` |
| MuJoCo reference loop | Wall: `--pace deadline` / `upstream` (`sonic/mujoco_ref/sim_ref.py:305-390, 427-433`) | Add `--pace lockstep` for the A/B | Same |

---

## 4. Recommendation

### (i) Interview POC, now

1. **Stay on the wall clock (A).** Finish the timing-gate re-measure with the PolicyServer on the dev box
   (PLAN.md:42 §0.12). If it passes (RTF p10 >= 0.98, irregular <= 0.15, 0 heartbeats during motion), do not start
   lockstep before the demo. "The real-robot binary, unmodified, on the wall clock" is also the strongest parity story
   to show.
2. **If the gate fails, or a demo needs GR00T on the main box, do B, not C.** B is 3-6 ED against 10-16 ED, keeps the
   robot's control code, and is flag-gated so wall mode is untouched. PLAN.md:31 §0.2 already allows "a sim-clock
   deploy patch" as a SONIC fallback, so no plan change is needed.
3. **Label every run** with its clock mode in `metrics.json` and in `docs/parity.md`.

### (ii) Longer term

Build **D** incrementally, in this order: B -> GR00T chunk `t0` in sim time -> P3 `Timebase` -> runtime robot clock ->
referee, UI and `sim_health`. Wall mode stays the default for parity and soak runs; lockstep becomes the default for
tests and evals, especially loaded or GR00T runs. Do C only if the patch route is blocked; nothing found here blocks it.

### First step

Write `sonic/patches/sim_clock.patch` (touches only `DEPLOY.cpp`, flag `--sim-clock`, default off, §2.B items 1-5),
applied by `sonic/build_deploy.sh` on the pinned b042411 checkout into a separate build dir, so the unpatched binary
stays available. At the same time add `--pace lockstep` to `sonic/mujoco_ref/sim_ref.py` (read `LowCmd_.reserve[0]`,
publish lowstate last, wait for the echo). Validate in MuJoCo first; Isaac only after that. Build and test on a box
with no live run.

### How to validate it

| Step | Setup | Pass criteria |
|---|---|---|
| 1. Flag off is unchanged | Patched binary without `--sim-clock`, MuJoCo ref loop, quiet box | Same checks as the walk_diagnosis quiet-box runs: walk_forward 2.70-2.72 m in 6 s, heading error <= 3.3 deg, 0 falls (docs/walk_diagnosis.md:10-11) |
| 2. Lockstep A/B, quiet box | MuJoCo `--pace lockstep` + `--sim-clock` vs wall mode, the sonic_deploy.md §6/§7 scenarios (gentle envelope and aggressive), `--policy-input-logfile` and `--planner-motion-logfile` on | Speeds, stop times and turn errors inside the wall-mode run-to-run spread; per-tick obs match wall mode wherever the state matches; the synchronous-planner effect quantified |
| 3. Stress | MuJoCo lockstep with `stall_matrix.sh` stalls of 0.3 / 1 / 3 s and a busy box | Deploy never exits; 0 irregular windows; fall rate not above the quiet-box rate (wall mode on a busy box: 1-4 falls per gentle run, sonic_deploy.md §7) |
| 4. Determinism | Two identical scripted MuJoCo lockstep runs | Trajectories match, up to the wall-clock timing of harness commands through the Input thread |
| 5. M1 drive test | Isaac P1 `--lockstep` + patched deploy, `tools/m1_drive_test` (E2-E5) | E2-E5 pass with numbers inside the M1 spread (docs/M1.md §5.2); E4: leg targets and `g1_debug` at 50 per sim second |
| 6. Timing gate | `tools/groot_timing_gate.py` in lockstep with GR00T on the main box | `irregular = 0` and `heartbeats = 0` (hard assertions); 0 falls in `stand_groot`; RTF p10 recorded as throughput, which also answers the lockstep-RTF unknown |

Steps 1-4 need no Isaac. If step 2 shows that the synchronous planner changes behaviour measurably, try running
`Planner()` on its own thread but gated on the sim tick before giving up on determinism.

---

## 5. Unknowns

1. **Lockstep RTF on the main box.** The serial P1->P2->P1 round trip adds the deploy's compute (0.4 ms quiet, 3-7 ms
   contended, planner up to 8.7 ms every 5th tick) plus DDS loopback to every 20 ms. M1's budget was ~0.84 s of wall
   time per sim second (docs/M1.md:170). Not measured. Handshake cost in P1's Python (`ddspy_take` polling vs a
   waitset) estimated 0.2-1 ms per 20 ms, also not measured.
2. **Synchronous planner vs threaded planner.** Whether tick-aligned plans change measured walking, turning and
   stopping. Needs the A/B in §4 step 2.
3. **Queue-length-1 lowstate listener** (`DEPLOY.cpp:2295`): whether a slow callback can drop the 20 ms boundary
   sample. Edge detection tolerates other drops and P1 waiting for the echo should guarantee delivery in steady state;
   not tested under load.
4. **PhysX determinism.** Whether Isaac's CPU PhysX is bit-identical run to run at a fixed dt, which replayable
   lockstep runs need. Not checked.
5. **Kit idling at the barrier.** Whether Isaac tolerates the physics loop waiting tens of ms with no `sim.step`
   while camera housekeeping, GT RPCs and the viz recorder are still serviced. Not checked.
6. **P3 constants at RTF 0.8-0.95** under B only (P3 on the wall clock): FacingServo push 0.6, arm servo ki 2.0/s with
   0.15 s delay, pure pursuit, approach time constants.
7. **Remote PolicyServer re-measure.** If it brings the gate back under the bar, B is insurance, not a blocker.
8. **Arena checkpoint and latency.** Whether `GN1x-Tuned-Arena-G1-Static-PickNPlace` (trained and evaluated lockstep,
   with no inference latency; docs/arena_vs_sonic.md:181-183) does better with an Arena-style pause than with latency
   compensation in sim time. Decides the default of D's pause knob.
9. **onnxruntime-gpu or TensorRT Python inside Isaac Sim 5.1's Kit Python** (option C only). Not checked.
10. **Upstream issue #233** (PLAN §7.3.6's evidence that observation assembly is fragile) was not read.
11. **Upstream plans.** `gear_sonic`'s `SimLoopConfig.sim_sync_mode` (`gear_sonic/utils/mujoco_sim/configs.py:133`)
    exists but nothing reads it; whether the WBC maintainers plan a sim-sync mode for the C++ deploy is unknown.
12. **libfaketime** on the unmodified binary: not tested. At best it matches rates; it cannot give a per-tick barrier.
