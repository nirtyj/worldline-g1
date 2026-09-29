# SONIC deploy contract (P2 wl-sonic)

Owner: deploy component (`worldline-g1/sonic/`). Status: **v2 (2026-09-29)**. Everything is read from code
(file:line given); items tagged **[MEASURED]** were confirmed on the box with the MuJoCo reference loop
(`sonic/mujoco_ref/`) and against wl-isaac (P1, `sonic/isaac_char/`); run dirs in §8.

## 0. What P1 (wl-isaac) and P3 (wl-body) must know: summary

1. The deploy is wall-clock only (threads at 100/50/10/500 Hz; there is no sim-time mode). DDS domain 0 is
   hard-coded, so run one deploy per network namespace. `rt/lowstate` and `rt/secondary_imu` must flow before it
   starts, and must never pause for 500 ms or longer, or the process exits (§2).
2. `rt/lowcmd` arrives at ~500 Hz; the leg `q` targets inside change at 50 Hz **[MEASURED: 498-500 msg/s, 49.8-50.0
   target changes/s]**. Apply `tau + kp(q*-q) + kd(dq*-dq)` from the latest message every physics step (§2).
3. Hand-over **[MEASURED]**: never start the policy (`command start`) with the robot hanging high in the band.
   It flails (hip-yaw targets up to 10 rad), and the later drop survives on a quiet box but fell on a busy one.
   Hold the pelvis at standing height with the feet
   touching the floor (band anchor ~0.80 m), send `command start`, stream IDLE, then release the band about 1 s
   later. Measured: pelvis z 0.787 m, 0 falls, drift < 0.09 m in 15 s (§3).
4. Walking **[MEASURED]**: SLOW_WALK reaches 85-95 % of the commanded speed at 0.5-0.7 m/s (about 0.25 m/s for a
   0.3 m/s command). Switching to IDLE stops the robot in 0.4-1.1 s. Walking drifts to the right by about 2-8 cm per
   metre (Isaac: 0.12-0.18 m over 1.8-2.5 m at 0.5-0.7 m/s), so close the position loop on ground truth (§6, §7).
5. Turning in place **[MEASURED]**: IDLE + a new `facing` turns only about 80 % of the command (45° gives 35-37°,
   90° gives 73-77°, 135° gives 115-120°), in MuJoCo and in Isaac alike. The planner's own target
   (`g1_debug.base_quat_target`) already stops short; the deploy has no facing deadband (exact compare,
   `DEPLOY.cpp:3671-3673`). **Action for P3:** `body/motions.py` `TurnToMotion` as committed in 9185dd7 (≤ 30° steps,
   final command = target, success needs |err| < `yaw_tol_deg` = 6°) **never reached 6° on P1**: errors after 20 s
   were +9.9° (90°), −15.2° (−90°), −10.4° (180°), +16.3° (45°), −12.4° (−45°), so every turn would time out as
   `failed`. Adding a residual push (once |yaw rate| < 0.08 rad/s for 0.5 s, move the command past the target by
   0.6 × the remaining error) reached < 6° in 2.6-6.4 s in 5 of 5 turns with no fall (§6, run `p1char-bodyturn-*`).
   Push-and-settle loops converge to within 3° (§6, §7).
6. Falls **[MEASURED]**: on **wl-isaac (P1)** SONIC had **0 falls** in 2 characterisation runs (about 12 min: 14
   in-place turns up to 180°, 14 walk start/stop cycles at 0.3-0.7 m/s, 5 curves up to 35°/s, a 60 s stand) at
   RTF 0.996 with P1's normal hitches (§7). The MuJoCo reference falls now and then (roughly once per 4 min of
   manoeuvres even with valid timing, more on a busy box): a sim2sim gap, not an Isaac issue. The deploy's own
   50 Hz loop never missed a tick. P3 should still treat a fall as `failed`.
7. `command {stop:1}` **terminates the deploy process**. To stop walking, send planner IDLE, or stop publishing
   (after 1 s the deploy forces IDLE).

Upstream: NVlabs/GR00T-WholeBodyControl @ `b042411` at `/work/repos/GR00T-WholeBodyControl` (box).
Paths below are relative to that repo. `DEPLOY.cpp` = `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`,
`INC/` = `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/`, `MJ/` = `gear_sonic/utils/mujoco_sim/`.

The deploy binary is **unmodified**. `sonic/run_deploy.sh` runs the same command line `deploy.sh`
would run (`deploy.sh:553-588`) directly, which removes the `Proceed? [Y/n]` prompt and the per-start
`just build`, and adds `--zmq-port/--zmq-out-port` (real CLI flags, `DEPLOY.cpp:4339-4340, 4411-4416`).

---

## 1. Process, ports, DDS domain

| Item | Value | Source |
|---|---|---|
| Binary | `gear_sonic_deploy/target/release/g1_deploy_onnx_ref <iface> policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all --zmq-host localhost --disable-crc-check` | `deploy.sh:240-246, 404-406, 581-588` |
| DDS domain | **0, hard-coded**. There is no flag. | `DEPLOY.cpp:2218` `ChannelFactory::Instance()->Init(0, networkInterface)`; Dex3 same (`INC/dex3_hands.hpp:52`) |
| DDS interface | first CLI arg; `lo` for sim | `deploy.sh:126-146` |
| ZMQ input | SUB **connects** to `tcp://<zmq-host>:<zmq-port>` (default 5556), topics `command`, `planner`, `pose`. The sender must **bind** the PUB. | `INC/input_interface/zmq_packed_message_subscriber.hpp:201-202`; `zmq_manager.hpp:88-145` |
| ZMQ output | PUB **binds** `tcp://*:<zmq-out-port>` (default 5557), topics `g1_debug` (every control tick) and `robot_config` | `INC/output_interface/zmq_output_handler.hpp:132-143` |
| Stdin | `o`/`O` = emergency stop and exit (stdin is set non-blocking) | `zmq_manager.hpp:166-175`; `zmq_endpoint_interface.hpp:141` |

Consequences:
- Only **one** deploy + simulator pair can run on `lo` at a time, because the domain is fixed at 0.
  The build-phase rule "the deploy agent owns domain 0 on lo" is therefore a hard requirement, not a
  convention. The Isaac bridge must publish on **domain 0** when it is paired with the deploy
  (domain 7 is fine for bridge-only tests). For parallel tests, give each pair its own network namespace
  (`sudo unshare -n` / `ip netns`, which gives a private `lo`). **[MEASURED]** This works: the Isaac agent's
  `sim_isaac/tools/sonic_netns_test.sh` and `scripts/body_netns_e2e.sh` run the real deploy that way, and a
  diagnostics agent ran MuJoCo + deploy pairs in `unshare -n` while `sonic/mujoco_ref` ran on the host `lo`.
  Neither sim saw the other's `rt/lowcmd` (500 msg/s, not 1000).
- `run_deploy.sh` and `run_ref_loop.sh` refuse to start while another deploy runs **in the same network
  namespace**. They match on `/proc/PID/exe` plus `/proc/PID/ns/net` (`sonic_env.sh: deploys_in_my_netns`), so
  deploys in other namespaces are ignored. `--force` overrides the check.
- Build-phase test ports (+100): `run_deploy.sh --zmq-port 5656 --zmq-out-port 5657`.

`sonic/run_deploy.sh` (all subcommands are non-interactive):

| Command | What it does |
|---|---|
| `start [--session S --window W --log F --zmq-port P --zmq-out-port P2 --iface lo --wait-init SECS --taskset CPUS --rt-prio N --force] [-- extra binary args]` | Creates or reuses tmux session S (default `deploy-sonic`), window W (default `sonic`), and runs the binary there with stdout+stderr appended to F (default `/work/logs/wl/<S>.log`, one `=== start` marker per run) and the pid in `/work/logs/wl/<S>.pid`. With `--wait-init`, it returns once the log shows `Init Done` (exit 1 if the deploy dies or times out). The first start builds the TensorRT engines (about 2 min); later starts reach `Init Done` in 32-36 s **[MEASURED]**, mostly the 5 s `MotionSwitcherClient` wait, engine loading and the 3 s INIT ramp. |
| `fg [same options]` | Same command line in the foreground (for a tmux window someone else owns). |
| `stop [--session S --window W]` | Sends `o` on stdin (emergency stop: damping command, then `main()` returns, `zmq_manager.hpp:166-175`), then SIGINT, then SIGKILL; removes the pid file and the tmux window. |
| `status [--session S]` | Prints the pid, uptime, CPU and the last log lines; exit 1 if not running. |
| `rt PRIO [--session S]` | Sets SCHED_RR PRIO (sudo chrt) on every deploy thread except the busy-spinning main thread. |

tmux targets are always exact (`=session:=window`). tmux 3.4 prefix-matches bare names **[MEASURED]**, so
`-t deploy-ref` also matched a session named `deploy-refq1` once `deploy-ref` was gone.

## 2. DDS contract (what the simulator must provide)

IDL types: `unitree_hg` (`LowState_`, `LowCmd_`, `IMUState_`, `HandState_`, `HandCmd_`). Python:
`unitree_sdk2py.idl.unitree_hg.msg.dds_` and `unitree_sdk2py.idl.default.unitree_hg_msg_dds__*`
(`MJ/unitree_sdk2py_bridge.py:14-21, 37-43`).

| Topic | Dir (sim view) | Type | Required? | Deploy reads | Source |
|---|---|---|---|---|---|
| `rt/lowstate` | PUB | `LowState_` | **yes** | `motor_state[i].q/.dq` (i = 0..28, Unitree/MuJoCo joint order), `imu_state.quaternion` (w,x,y,z), `imu_state.gyroscope`, `mode_machine`, `crc` (only if CRC check on), `wireless_remote` (gamepad modes only), `motor_state[i].motorstate/temperature/tau_est` (logging only) | `INC/robot_parameters.hpp:27-29`; `DEPLOY.cpp:2294, 2645-2679, 2842-2935, 3448-3475` |
| `rt/secondary_imu` | PUB | `IMUState_` | **yes**: control stops if it has never arrived | `quaternion`, `gyroscope`, `accelerometer`. **Logged only**, not a policy input | `DEPLOY.cpp:2296, 2681-2685, 2847-2850, 2932-2934` |
| `rt/lowcmd` | SUB | `LowCmd_` | yes | apply `motor_cmd[i].{q, dq, kp, kd, tau}` | `DEPLOY.cpp:2291, 2694-2715` |
| `rt/dex3/{left,right}/state` | PUB | `HandState_` | no: if absent the hand state is zeros | `motor_state[0..6].q/.dq` | `INC/dex3_hands.hpp:56-60`; `DEPLOY.cpp:2939-2957` |
| `rt/dex3/{left,right}/cmd` | SUB | `HandCmd_` | no | 7 motors, PD like the body | `INC/dex3_hands.hpp:56-60` |
| `rt/odostate` | (MuJoCo publishes it) | `OdoState_` | **no**: the deploy does not subscribe | `MJ/unitree_sdk2py_bridge.py:70-74` |

### Field semantics the policy depends on

- **Joint order** is the Unitree hardware / MuJoCo order `robot_parameters.hpp:90-126` (0-5 left leg
  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll; 6-11 right leg; 12-14 waist
  yaw, roll, pitch; 15-21 left arm shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll,
  wrist_pitch, wrist_yaw; 22-28 right arm). The deploy remaps to the IsaacLab order internally
  (`INC/policy_parameters.hpp` `mujoco_to_isaaclab`, `isaaclab_to_mujoco`). **The Isaac bridge must
  map Isaac articulation DOF order to this order by joint name.** Do not assume the orders match.
- `q` is in radians and absolute (URDF zero). The deploy subtracts `default_angles` itself
  (`DEPLOY.cpp:2863-2864`). Ankles are in PR (pitch/roll) mode: the deploy writes `mode_pr = 0`
  (`DEPLOY.cpp:2695`, `robot_parameters.hpp:78-81`).
- `imu_state.quaternion` = **pelvis** orientation in the world, **(w, x, y, z)**. It feeds the policy's
  gravity direction history (`DEPLOY.cpp:1613-1631`), the heading/anchor orientation, and the
  planner initialisation (`DEPLOY.cpp:3610`). MuJoCo sends free-joint `qpos[3:7]` (`MJ/base_sim.py:351`,
  `MJ/unitree_sdk2py_bridge.py:190`).
- `imu_state.gyroscope` = pelvis angular velocity **in the pelvis (body) frame**, rad/s. It feeds the
  policy (`his_base_angular_velocity_10frame_step1`, `DEPLOY.cpp:1592-1608`,
  `policy/release/observation_config.yaml`). MuJoCo sends free-joint `qvel[3:6]`, which MuJoCo expresses
  in the local body frame (`MJ/base_sim.py:352`, `MJ/unitree_sdk2py_bridge.py:192`).
  **Isaac: use `root_ang_vel_b` (body frame), not `root_ang_vel_w`.**
- `imu_state.accelerometer`: MuJoCo sends `qacc[0:3]`, the world-frame linear acceleration without
  gravity (`MJ/base_sim.py:353`). This is not what a real IMU measures, and the deploy only logs it.
  Any value works. Sending projected gravity plus acceleration is more realistic.
- `rt/secondary_imu.quaternion` = torso_link world orientation (w,x,y,z); `gyroscope` = torso angular
  velocity in the torso frame (`MJ/base_sim.py:359-371`, `mj_objectVelocity(..., flg_local=1)`).
  MuJoCo leaves `accelerometer` at zeros.
- `tick` = sim time in ms (`MJ/unitree_sdk2py_bridge.py:202`). The deploy ignores it: all its
  staleness checks use its own wall-clock receive time (`DataBuffer::GetDataWithTime`).
- `mode_machine`: the MuJoCo bridge **never sets it**, so it stays 0 (the yaml `MODE_MACHINE: 5`
  is not read by any code: `grep -rn MODE_MACHINE gear_sonic/` hits only the yaml). The deploy
  latches whatever arrives and echoes it back in `LowCmd_.mode_machine` (`DEPLOY.cpp:2675-2678, 2697`).
  **Isaac: leave it at 0**, as the reference does. It has no effect in sim.
- `crc`: MuJoCo does not fill it. The deploy must run with `--disable-crc-check` in sim (`deploy.sh:404-406`),
  and `run_deploy.sh` always passes it. With the check disabled, the `body_dq > 35 rad/s` safety stop
  is also skipped (`DEPLOY.cpp:2865`). The deploy still writes a CRC into `LowCmd_`
  (`DEPLOY.cpp:2710`). Ignore it.

### Timing

| Stream | Rate | Notes |
|---|---|---|
| `rt/lowstate`, `rt/secondary_imu`, hand states (sim → deploy) | **every physics step** (MuJoCo: 200 Hz, `SIMULATE_DT 0.005`) | `MJ/base_sim.py:389-391`, `wbc_configs/g1_29dof_sonic_model12.yaml:22` |
| `rt/lowcmd` (deploy → sim) | **500 Hz publish** (command-writer thread). The targets inside change at **50 Hz** (control thread). | `DEPLOY.cpp:11-21, 2186-2188, 2613-2621` |
| Deploy threads | input 100 Hz, control 50 Hz, planner 10 Hz, writer 500 Hz, all on the **wall clock** (`CreateRecurrentThreadEx`). There is no sim-time mode. | `DEPLOY.cpp:11-21, 2613-2621` |
| Lowstate staleness limit | **500 ms**. If the newest `rt/lowstate` is older than that in WAIT_FOR_CONTROL or CONTROL, the deploy sets `stop` and **the process exits**. | `DEPLOY.cpp:299, 2797-2812, 3846-3871` |

Evidence item E4 ("lowcmd ~50 Hz with changing leg targets") should therefore be read as "about 500 Hz
of `rt/lowcmd` messages whose leg `q` targets change at about 50 Hz". **[MEASURED]** In the MuJoCo reference,
between the band release and `command stop`: 498 msg/s and 49.8 distinct leg-target vectors/s (`quiet-a`
metrics.json). Isaac P1 with the deploy: leg targets change at 50 Hz (`m1.md` §1.9).

**What the simulator does with `rt/lowcmd` every physics step** (MuJoCo reference, `MJ/base_sim.py:389-432`):
1. `prepare_obs()` reads the current state, then `PublishLowState()` publishes lowstate, odostate,
   secondary_imu and hand states (before the step).
2. Elastic band (if enabled): an external wrench on the pelvis (`enable_waist=True`, `MJ/base_sim.py:185-195`)
   holds it at z = 1.0 m + `length` with kp 10000 / kd 1000 (linear) and kp 1000 / kd 10 (angular)
   (`MJ/unitree_sdk2py_bridge.py:352-378`).
3. For each of the 29 body motors, using the **latest** received `LowCmd_` (zero-order hold, no
   interpolation): `tau = motor_cmd.tau + motor_cmd.kp*(motor_cmd.q - q) + motor_cmd.kd*(motor_cmd.dq - dq)`
   (`MJ/base_sim.py:258-288`). Hands the same with `HandCmd_` (`:300-331`).
4. Clip to the per-joint effort limits (`MJ/base_sim.py:423`), write `ctrl`, then `mj_step` once (dt 0.005).
5. If pelvis z < 0.2 m the sim logs "Robot has fallen" and resets (`MJ/base_sim.py:508-515`).
6. Sleep to hold wall-clock pace (`MJ/base_sim.py:627-631`). The upstream loop does not catch up when it runs
   late, so RTF < 1 means the deploy sees a slower world. `sonic/mujoco_ref/sim_ref.py` uses absolute
   deadlines instead (`--pace deadline`, default). It catches up lag under 100 ms and resyncs beyond that, which
   keeps RTF at 1.000 on this box (`--pace upstream` reproduces the original).

Isaac guidance: run the same explicit PD at the physics rate, with the Isaac joint drives at
stiffness=damping=0 and effort-mode torque. Or use implicit PD with the per-step kp/kd written
into the drive gains, as SONIC training does (`gear_sonic/envs/manager_env/robots/g1.py:199-300`).
Either way the gains come **from the message**. The deploy sends its own kp/kd (`INC/policy_parameters.hpp`
`kps`/`kds`: hip/knee from `STIFFNESS_7520_22/14`, ankles and waist roll/pitch 2x `STIFFNESS_5020`, ...).

## 3. ZMQ input: `zmq_manager` protocol

Wire format of every message: `topic_bytes + 1280-byte NUL-padded JSON header + packed little-endian fields`
(`INC/input_interface/zmq_packed_message_subscriber.hpp:9-20, 99`;
`gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:14-27`). Build messages with
`gear_sonic.utils.teleop.zmq.zmq_planner_sender.build_command_message` / `build_planner_message`.
They are pure Python (json, struct, numpy), so the file can be copied or imported.

### `command` topic: `{start: u8, stop: u8, planner: u8}` (`zmq_manager.hpp:673-765`)
- `start=1, planner=1` → enables the planner, waits up to 5 s for it to initialise, then sets
  `operator_state.start` (WAIT_FOR_CONTROL → CONTROL), and plays the planner motion (IDLE = stand).
  A repeated `start` is ignored while already started (`zmq_manager.hpp:527`).
- `planner=0` switches to STREAMED_MOTION (`pose` topic, GR00T tokens) with a safety reset.
  `planner=1` switches back (`zmq_manager.hpp:236-275`).
- **`stop=1` terminates the deploy process.** It sets `operator_state.stop`, `main()` leaves its
  loop, sends a damping command (kp=0, kd=8) and exits (`zmq_manager.hpp:343-362`; `DEPLOY.cpp:4515-4534, 2718-2750`).
  **To stop walking, send `planner` mode 0 (IDLE) or stop publishing (1 s timeout). Never send `stop`,
  except to shut down.**
- The `delta_heading` field built by `build_command_message` is **ignored** by `zmq_manager` (not decoded).

### `planner` topic (`zmq_manager.hpp:769-975`)
Required: `mode: i32`, `movement: f32[3]`, `facing: f32[3]`. Optional: `speed: f32` (−1 = mode default),
`height: f32` (−1 = default), `upper_body_position/velocity: f32[17]`, `left/right_hand_joints: f32[7]`,
`vr_position/orientation/compliance`.
- Modes (`INC/localmotion_kplanner.hpp:78-106`): 0 IDLE, 1 SLOW_WALK (0.1-0.8 m/s), 2 WALK (0.8-2.5), 3 RUN,
  4-8 squat/kneel/crawl, 9-16 boxing, 17-26 styled walks. For M1 use **0 and 1** (optionally 2).
- The deploy normalises `movement` and `facing` (`zmq_manager.hpp:610-611`). A zero `movement` with
  mode 1 means "walk in place / turn toward `facing`".
- **Frame: `movement` and `facing` are in the PLANNER frame. Its +X is the robot's heading at the moment
  the planner was initialised** (at `command start`, or when PLANNER mode is re-entered). The planner
  context is reset to zero yaw at the origin (`INC/localmotion_kplanner.hpp:332-352, 591-624`), and
  `facing = [1,0,0]` means "keep the initial heading". A world-frame path follower must record
  `yaw0 = world pelvis yaw at start` and send `R(-yaw0) · v_world`. **[MEASURED]** With that conversion the
  walk displacement direction is within 2.2-2.7° of the commanded world heading (`drive_ref.py` `frame_world_heading`).
- Keepalive: every message is consumed once. If none arrives for **1 s**, the manager forces IDLE and
  keeps the last facing (`zmq_manager.hpp:582-630`). Send ≥10 Hz, and resend identical messages;
  re-sending the same values does not force a replan.
- Replanning: on a change of mode, facing or height, or of speed or movement while moving, and otherwise
  every 1 s while moving (`DEPLOY.cpp:3700-3732`; intervals `DEPLOY.cpp:231-234`). The planner runs at 10 Hz; its 30 Hz output is
  resampled to 50 Hz.

### Start sequence and hand-over (what `sonic/mujoco_ref/drive_ref.py:249-306` `startup()` does) [MEASURED]
1. Simulator up, publishing `rt/lowstate` + `rt/secondary_imu` at the physics rate, elastic band **on**.
2. Bind the planner/command PUB (the deploy's SUB connects to it) and wait ≥ 0.5 s (ZMQ slow joiner). Start
   streaming planner IDLE (`mode=0, movement=[0,0,0], facing=[1,0,0]`) at ≥ 10 Hz right away. Messages
   that arrive before control starts are simply consumed.
3. Start the deploy (`run_deploy.sh start --wait-init ...`). It loads the TensorRT engines (`<onnx>.trt` cache next to
   each ONNX; the first run builds them in about 2 min), waits about 5 s in `MotionSwitcherClient` (no robot
   service in sim, `DEPLOY.cpp:2279-2289`), and creates the DDS channels and threads. `Control()` in INIT then ramps
   from the current joint positions to `default_angles` over 3 s (`duration_ = 3.0`, `DEPLOY.cpp:2758-2793`),
   prints **`Init Done`** and holds that pose with PD (WAIT_FOR_CONTROL). No `g1_debug` is published yet; only
   `robot_config`. Measured: `Init Done` 32-36 s after start.
4. **Put the feet on the floor before starting the policy.** In upstream MuJoCo the band holds the pelvis at
   1.0 m (`MJ/unitree_sdk2py_bridge.py:352-378`), which leaves the feet about 0.17 m in the air. The upstream manual
   procedure is `]` (start) and then `9` (drop) (`docs/source/getting_started/quickstart.md:121-124`), so the
   policy starts while the robot hangs. **[MEASURED]** That is out of distribution: while it hangs, the leg
   targets swing by several rad (hip yaw up to ±10 rad). On a loaded box (load 9-14) the robot then fell
   0.7-0.9 s after the drop (runs `mujoco-ref-20260928-2308/2315/2323`). The early run that passed also fell at
   the drop, and was rescued only by MuJoCo's fall reset (`mj_resetData` teleports it standing,
   `MJ/base_sim.py:508-530`), which Isaac does not have. On a quieter box (load ~3) a diagnostics agent's
   hanging-start runs survived the drop (`/work/walkdiag/lensB/runs/v1up*`), so the drop is survivable but has
   no margin. Procedure used since then, with no hand-over fall in any run: ramp the band anchor down 0.20 m over
   2 s (to z ≈ 0.80 m, pelvis 0.765 m, feet at the floor), **then** send `command start`. P1's band already
   defaults to that height (`m1.md` §1.7, anchor `floor + 0.80`).
5. Send `command {start:1, stop:0, planner:1}` every 0.1-0.2 s until the first `g1_debug` arrives (0.2 s after the
   first send, measured). The deploy prints `[ZMQManager] Planner enabled`, `transitioning to CONTROL`,
   `Planner initialized successfully!`, `motion name is planner_motion`. A repeated `start` is a no-op.
6. Release the band about 1 s after control starts. Measured over 15 s: pelvis z 0.787 m (min = max), 0 falls,
   XY drift 0.085-0.087 m (`quiet-a`, `char-*`). On P1: the same sequence with `band {on:false, ramp_s:1.0}` (P1's band starts
   at floor + 0.80 m, so step 4 is already satisfied) held pelvis z ≥ 0.786 m with no hand-over fall in 3 runs
   (`p1char-*`); the Isaac agent's smoke test and `m1_up.sh` / `body stand` use the same order.
7. Walk: `mode=1, movement=dir, facing=heading, speed=0.3..0.7`. Keep resending at ≥ 10 Hz (our driver: 20 Hz).

## 4. ZMQ output: `g1_debug`
`b"g1_debug" + msgpack map` every control tick (50 Hz), non-blocking send. Keys include `base_quat`,
`base_ang_vel`, `body_torso_quat`, `body_q[29]`/`body_dq[29]` (MuJoCo order), `last_action[29]`,
`token_state[64]`, `init_base_quat`, `delta_heading`, `base_trans_target`, `base_quat_target`,
`body_q_target[29]` (`INC/output_interface/zmq_output_handler.hpp:18-75`). There is no explicit
"planner mode" key. Planner mode is shown by the deploy log (`[ZMQManager] Planner enabled`,
`Planner initialized successfully!`) and by `base_trans_target` advancing while walking.

## 5. Operational gotchas
- `main()` waits with `while(!stop) sleep(0.02);`. `sleep()` takes an `unsigned int`, so this is `sleep(0)`:
  **a busy spin at 100 % of one core** (`DEPLOY.cpp:4523-4526`). `SetThreadPriority()` runs *after* the input,
  command-writer, control and planner threads are created with `UT_CPU_ID_NONE` (`DEPLOY.cpp:2613-2624`). It
  pins **only the main thread** to CPU 0 and asks for SCHED_FIFO, which fails silently without CAP_SYS_NICE
  (`DEPLOY.cpp:2632-2640`). The worker threads float. **Every** deploy on the box spins on CPU 0, so keep
  latency-critical work (the physics loop) off CPU 0. Budget one full vCPU for P2's spinner.
  **[MEASURED]** The control loop itself is robust to box load: `g1_debug.index` advanced exactly 50 ticks/s,
  with no skipped tick, through every fall burst (`char-20260929-005226`). The policy (TRT decoder) takes
  0.1 ms at p50 and up to 6.2 ms at worst; the planner model takes 3.0 ms p50 and up to 8.7 ms, well inside the
  20 ms / 100 ms periods (deploy `Loop timing` lines, 9 runs).
- First start builds the TensorRT engines from the ONNX files: `policy/release/policy_model_decoder.trt` (41 MB),
  `encoder_model_encoder.trt` (50 MB), `planner/target_vel/V2/planner_planner_sonic.trt` (747 MB). Later starts
  reuse them; their hash is checked (`src/TRTInference/InferenceEngine.cpp:143, 330-372`).
- When the deploy exits (`o`, `stop`, lowstate timeout) it publishes one damping command (kp=0, kd=8). With no
  lowcmd after that, the simulator keeps the last command, and the robot goes limp and falls **[MEASURED]**.
  Engage the band first (m1_down.sh does).
- The deploy never reads the robot's XY position. Position feedback is the caller's job (ground truth in sim).
- `g1_debug` is only published in CONTROL, so a consumer that waits for it is waiting for `command start`.

## 6. Planner command response [MEASURED, MuJoCo reference, WBC @ b042411, planner `target_vel/V2`]

Source: `sonic/mujoco_ref/char_ref.py` (runs `char-20260929-005226`, `char-20260929-010126`) and
`drive_ref.py` (5 runs). "Achieved" is ground-truth pelvis yaw or position. "Planner" is the yaw change of
`g1_debug.base_quat_target`, the planner's own reference.

**Turn in place: IDLE, movement 0, new `facing` = current yaw + Δ, 6.5 s** (keyboard Q/E send ±30° steps,
`INC/input_interface/keyboard_handler.hpp:554-566`)

| Δ commanded | achieved (runs) | planner target | residual | t to 90 % | translation |
|---|---|---|---|---|---|
| +45° | 35.3, 34.4 | 41.7, 41.2 | +10° | 2.1-2.3 s | ≤ 0.01 m |
| −45° | −35.6, −35.8 | −41.8, −42.6 | −9° | 2.5 s | ≤ 0.02 m |
| +90° | 75.8; drive_ref 75.1-76.3 (5 runs); Isaac P1 74.7-75.0 | 77.4-80.0 | +14..+15° | 0.5-0.7 s | 0.08-0.12 m |
| −90° | −76.3, −77.3 | −84.5, −82.1 | −13..−14° | 0.8 s | 0.03 m |
| +135° | 119.0, 116.2 | 122.1, 121.9 | +16..+19° | 0.7 s | 0.15 m |
| −135° | −116.9, −119.8 | −120.2, −123.7 | −15..−18° | 0.7 s | 0.02-0.11 m |

The undershoot is systematic: the robot achieves about 80-88 % of Δ, and the planner's own target already
stops 3-13° short. **Closed-loop turn** (what P3's `turn_to` should do): command `facing = target`, wait until
|yaw rate| < 0.08 rad/s and |v| < 0.08 m/s for 0.5 s, then set `facing += residual error` and repeat.

| Δ | final error | time | iterations |
|---|---|---|---|
| +90° | −1.7°, −2.3° | 9.7 s, 7.6 s | 3 each |
| −90° | +2.7°, +1.7° | 6.6 s, 8.2 s | 3 |
| 180° (10 trials, incl. between walk cycles) | −3.1°..+3.4° (8 trials); one trial with tol 5° ended at 4.8°; one fell during a simulator-stall burst (§7) | 3.8-15 s | 2-4 |

**wl-body `TurnToMotion` emulation on P1** (`char_ref.py --sections bodyturn`, run `p1char-bodyturn-20260929-015011`,
the exact stepping of `body/motions.py:121-135` in commit 9185dd7: facing advances in ≤ 30° steps from the last
commanded facing once the robot is within 15° of it; final command = target; success = |err| < 6°; timeout 20 s):

| Δ | as committed: reached 6°? error at timeout | + residual push 0.6: time to < 6° (push applied) |
|---|---|---|
| +90° | no, +9.9° | 5.6 s (+6.2°) |
| −90° | no, −15.2° | 4.1 s (−7.4°) |
| 180° | no, −10.4° | 6.4 s (−6.1°) |
| +45° | no, +16.3° | 2.7 s (+9.7°) |
| −45° | no, −12.4° | 2.6 s (−9.1°) |

The small final steps (15-30°) undershoot proportionally more than one big step. Without the push, the robot
settles 10-16° short and stays there.

With gain 1 the first correction overshoots (for example 13° → −8° → −1.7°), so a gain of about 0.5-0.7 on the
residual should converge in fewer steps (untested). **SLOW_WALK with movement 0 and a new facing** reached
86-88° for ±90° in 3 of 4 trials, but in one trial the planner target ran on to −130° and the robot translated
0.89 m, and one trial fell during a simulator-stall burst (§7). It is not recommended as the default turn.

**Walking: SLOW_WALK, movement = facing = heading, 4 s, then IDLE** (16 cycles in 2 runs; the 3 that fell during simulator-stall bursts are excluded, §7)

| speed cmd | steady speed (last 2 s) | distance in 4 s | lateral drift | stop time (v < 0.1 m/s for 0.3 s) |
|---|---|---|---|---|
| 0.3 m/s | 0.17-0.26 m/s | 0.84-0.93 m | −0.03 m | 0.39-0.93 s |
| 0.5 m/s | 0.42-0.48 m/s | 1.73-1.93 m (6 s: 2.62-2.70 m) | −0.02..−0.11 m | 0.62-0.95 s |
| 0.7 m/s | 0.62-0.65 m/s | 2.55-2.64 m | −0.15..−0.21 m | 0.60-0.65 s |

Yaw drift per cycle is −3.2..+2.1°. All stops are well inside the E3 limit of 1.5 s. The drift is small but
systematic (to the right), so P3 must close the XY loop on ground truth; `go_to` cannot be dead reckoning.
Strafing (SLOW_WALK, movement ⊥ facing, 0.3 m/s, 4 s) gives 0.89-1.12 m lateral, |forward| < 0.06 m,
|Δyaw| < 1°.

**Curved walking: SLOW_WALK, movement = facing rotating at a constant rate, 6 s**

| speed, yaw rate | yaw achieved / commanded | heading lag mean / max | falls |
|---|---|---|---|
| 0.5 m/s, +20°/s | 107° / 120° | 9-10° / 16° | 0 |
| 0.5 m/s, −20°/s | 105-106° / 120° | 18-19° / 26-27° | 0 |
| 0.4 m/s, +35°/s | 188-191° / 210° | 22° / 29-30° | 0 |

Pure pursuit works without stopping, but expect heading lag of 10-20° and keep the lookahead large enough
that the lag does not cut corners into furniture.

## 7. Falls in the MuJoCo reference: what does and does not cause them [MEASURED]

SONIC in the MuJoCo reference loop is not fall-free. It passes the default scenario (`drive_ref.py`) in most
runs, but falls now and then in the longer characterisation runs. An independent diagnosis
(`docs/walk_diagnosis.md`) reached the same ranking (contention first). Its harness rules are now applied in
`sonic/mujoco_ref`: stop scoring at the first fall, build commands from the commanded facing, a VALID/INVALID
timing gate in `report_ref.py` (`metrics.json: timing_gate`), and refusal to run on a busy box unless `--allow-busy`.
Findings:

- **Not the deploy.** Its control loop ticked at exactly 50.0/s (`g1_debug.index`), including through every
  fall. Policy inference takes ≤ 6.2 ms and the planner ≤ 8.7 ms. Lowstate age at the policy is ≤ 12 ms.
- **Not DDS cross-talk.** A read-only listener on domain 0 / host `lo` during a run saw exactly 200 lowstate
  msg/s and 500 lowcmd msg/s, i.e. exactly one simulator and one deploy.
- **Not the MuJoCo Dex3 hand model**, even though `scene_43dof.xml:3` says "not stable for simulation". Hand joints
  moved ≤ 0.3 rad per 20 ms (median 0.0001) in falling and non-falling runs alike.
- **Simulator stalls correlate but are not sufficient.** The Python sim process stalls for 15-107 ms inside
  `env.sim_step()` or through sleep overshoot (`sim_stalls.jsonl`; 18-47 per 2.5 min run while other agents'
  Isaac jobs ran). The trace is written from a separate thread, so these stalls are not file I/O. 9 of 14 fall onsets had a ≥ 30 ms stall in the 1.5 s before, but
  stalls are frequent, so many such coincidences are expected by chance. The controlled injection matrix
  (`stall_matrix.sh`: after the band release, block the physics loop for N ms every 3 sim-s, then burst to
  catch up or drop the lag; each config ran the same 8 walk start/stop cycles at 0.3-0.7 m/s + 3 curves,
  one run each):

  | config | injected stalls | natural stalls > 15 ms (max) | cycles with a fall | curves with a fall | fall resets |
  |---|---|---|---|---|---|
  | base (no injection) | 0 | 18 (61 ms) | 1/8 | 1/3 | 4 |
  | burst60 (60 ms, then catch up) | 46 | 42 (64 ms) | 1/8 | 3/3 | 13 |
  | drop60 (60 ms, lag dropped) | 44 | 44 (148 ms) | 2/8 | 0/3 | 20 |
  | drop120 (120 ms, lag dropped) | 42 | 20 (89 ms) | 1/8 | 0/3 | 2 |

  Forty-two injected 120 ms freezes did not make things worse than no injection. With one run per config, the
  differences are within run-to-run noise. **Isolated physics hitches up to 120 ms every 3 s are mostly
  survivable.** Neither burst catch-up nor dropping the lag is clearly better.
- **Aggressive commands are the main risk.** Fall onsets were at 0.7 m/s walk cycles (2 of 2 in one run),
  a 0.4 m/s curve at 35°/s (yaw-rate spike to 9 rad/s with both feet off the floor, then a stumble), 135-180°
  single-step facing jumps, and the first seconds after walk start or stop. The conservative envelope results are
  below.

**Conservative envelope, MuJoCo** (`char_ref.py --sections gentle`, 3 runs of about 4.5 min each: turns with the
facing ramped at 30°/s (like repeated keyboard Q/E presses) plus closed-loop correction (gain 0.6); SLOW_WALK
0.4 m/s cycles of 5 s walking then IDLE; curves at 0.4 m/s and ±15°/s; then a 60 s IDLE stand):

| run | sim stalls > 15 ms (max) | 8 turns (±90, ±180, 45, 3 × 180) | 6 walk cycles 0.4 m/s | 2 curves 15°/s | 60 s stand |
|---|---|---|---|---|---|
| `gentle-20260929-012145` | 19 (62 ms) | 7 ok, 1 fell (−180°); \|err\| ≤ 2.75°, 5.5-13 s | 6/6 ok; v 0.33-0.38 m/s, stop 0.45-1.17 s | 2/2 ok, lag 8-15° | ok, z ≥ 0.786 |
| `gentle-20260929-012605` | 49 (74 ms) | 8/8 ok; \|err\| ≤ 2.73°, 4.3-13.4 s | 5/6 ok, 1 fell | 2/2 ok, lag 7-17° | ok, z ≥ 0.786 |
| `gentle-20260929-013025` | **259 (142 ms)**, box busy | 5 ok, 3 × 180° fell | 6/6 ok | 1/2 ok | ok, z ≥ 0.769 |

The envelope lowers the fall rate but does not remove it in MuJoCo: 1, 1 and 4 fall events per run. The worst
run is the one with 13× more simulator stalls (other agents started jobs during it), so **a busy box, not an
isolated hitch, is what raises the fall rate**. The IDLE stand never fell: 3 × 60 s here plus every stand check
in §8. Turn accuracy with the ramp and closed loop: every turn without a fall ended within 2.75° of target (20 of 20; 23 of 24 including the turns that fell).
The Isaac (P1) numbers for the same scenario are below, and they are what M1 depends on.

**The same scenarios on wl-isaac (P1, Isaac Sim 5.1 / PhysX, empty scene, head camera 640x480 at 30 Hz, CPU PhysX
200 Hz, `--rt-pace`) with the unmodified deploy** (`sonic/isaac_char/run_p1_char.sh`, host `lo`, DDS domain 0,
ports +400; runs `p1char-20260929-013818` = open-loop turns + gentle envelope, `p1char-aggr-20260929-014450` =
SLOW_WALK turns, closed-loop turns, 0.3-0.7 m/s cycles, 20-35°/s curves):

| | Isaac P1 (2 runs, about 12 min of manoeuvres) |
|---|---|
| falls in the controlled window | **0** (14 in-place turns incl. 180°, 14 walk cycles at 0.3-0.7 m/s, 5 curves up to 35°/s, 60 s stand) |
| open-loop IDLE turn (achieved / planner target) | 45°: 37.1 / 41.9; −45°: −35.1 / −40.0; 90°: 73.5 / 79.7; −90°: −75.6 / −83.3; 135°: 116.7 / 123.0; −135°: −115.2 / −123.3; same undershoot as MuJoCo |
| closed-loop turn, gain 1 | +90°: −0.35° (9.3 s); −90°: +1.46° (8.2 s); 180°: +2.86° (11.0 s) |
| ramped 30°/s + closed loop, gain 0.6 | 8 turns, final \|err\| 0.01-2.68°, 10.9-20 s (settling is slower than in MuJoCo) |
| SLOW_WALK + zero movement turn | ±90° → residual 5.4° / −9.0°, 0.08-0.13 m translation, no fall |
| steady speed (cmd → achieved) | 0.3 → 0.23-0.26; 0.4 → 0.22-0.37; 0.5 → 0.43-0.47; 0.7 → 0.59 m/s |
| stop time (IDLE, v < 0.1 m/s) | 0.39-1.11 s |
| lateral drift (to the right) | 0.05-0.13 m per 1.4-1.7 m at 0.4 m/s; 0.12-0.18 m per 1.8-2.5 m at 0.5-0.7 m/s |
| curves (yaw achieved / commanded, mean lag) | 15°/s: 114 / 120, 8°; −15°/s: −104 / −120, 15°; ±20°/s: 105-106 / 120, 10-18°; 35°/s: 191 / 210, 23° |
| 60 s IDLE stand | pelvis z ≥ 0.786 m, 0 falls |
| P1 timing during the runs | RTF 0.996-0.997, worst 1 s 0.83-0.84; 12-18 hitches > 25 ms, 6-7 overruns (0.76-0.97 s dropped), 14-20 lowstate heartbeats; leg targets 49.5-50 Hz |

**Conclusion: in the integration simulator the controller is markedly more robust than in the MuJoCo reference.**
SONIC was trained in Isaac/PhysX (`gear_sonic/envs/manager_env/robots/g1.py`). The MuJoCo falls are a sim2sim
gap made worse by box load, not a property that carries over to Isaac. Two runs on an empty floor do not prove
the houses; the M1 drive test (E3) in the house is the real check. The undershoot, speed and drift numbers are
the same in both sims, so P3's closed loops are needed either way.

What this means for P1 (Isaac): keep physics-loop hitches short (they add risk and waste the deploy's 20 ms
budget), but P1's current policy (catch up lag under 100 ms, drop beyond, lowstate heartbeat, `m1.md` §1.10) is
consistent with this data. Report the worst stall per second next to RTF: average RTF hides stalls
(per-second RTF stayed at 0.98-1.02 while 40-90 ms stalls happened). What it means for P3: use the
conservative envelope below by default, and keep checking `pelvis_z` / `fallen` from ground truth. A fall
must end the motion as `failed`.

## 8. Evidence index (box: `/work/worldline-g1/outputs/m1/deploy/`, pulled to the laptop under `outputs/m1/deploy/`)

| Run | What it shows |
|---|---|
| `final-20260929-015747/` | **MuJoCo reference with the final harness and the rebuilt binary (build_deploy.sh after the ROS2 fix): 13/13 checks, timing gate VALID** (irregular 0.107, RTF 1.000; one other Isaac job on the box, `--allow-busy`): stand 15 s (drift 0.03 m); walk 2.68 m at 0.5 m/s; turn +76.7°; strafe 0.86 m; stop mid-walk in 0.47 s; planner timeout → IDLE; **60 s long stand z 0.785-0.786**; 0 falls; lowcmd 497 msg/s, leg targets 49.8/s; 31 touchdowns while walking (alternation 0.83). `video_third_person.mp4`, `video_head_camera.mp4`, `report/`, `metrics.json`, `sim_stalls.jsonl`. Repeats of the default scenario after the fixes (`repeat-20260929-0201*`..`-0204*`, no render): every run with a **VALID** timing gate passed (3/3: `final`, `repeat-020243`, `quiet-a`). Runs with an **INVALID** gate passed 1/2: `repeat-020424` fell in the strafe with irregular 0.27 and 72 sim stalls > 15 ms, and the fall gate stopped scoring there. |
| `quiet-a-20260929-003755/` | **MuJoCo reference pass, 12/12 checks, timing gate VALID** (irregular 0.019, RTF 1.000): control start 0.2 s after `command start`; g1_debug 50.2 Hz; lowcmd 498 msg/s with leg targets changing 49.8/s; stand 15 s at z 0.787 m; walk 2.70 m at 0.5 m/s (displacement within 2.2° of the commanded heading); turn +76.3°; strafe 1.12 m; stop mid-walk in 0.78 s; planner-silence timeout → IDLE with no fall; `command stop` ends the process. `video_third_person.mp4`, `video_head_camera.mp4` (offline replay of the 50 Hz ground-truth qpos trace), `report/trajectory.png`, `report/timeseries.png` (pelvis z, yaw, speed, foot contacts, lowcmd leg targets), `metrics.json` (30 touchdowns while walking, left/right alternation 0.86). |
| `p1char-20260929-013818/`, `p1char-aggr-20260929-014450/` | **The same driver against wl-isaac (P1) + the unmodified deploy**: 0 falls; turn, speed, stop, drift and curve numbers (§7); `trace.npz` (P1 50 Hz record: motor q, q_target, kp, foot contacts), `p1_stats_end.json`, `deploy.log`, `drive_trace.jsonl`, `char.json`. |
| `p1char-bodyturn-*` | wl-body `TurnToMotion` emulation on P1 (§6, P3 turn guidance). |
| `char-20260929-005226/`, `char-20260929-010126/` | MuJoCo planner-response characterisation (§6); falls coincide with a busy box (timing gate INVALID for `-010126`). |
| `gentle-20260929-012145/`, `-012605/`, `-013025/` | MuJoCo conservative envelope (§7). |
| `stall-{base,burst60,drop60,drop120}-*`, `stall_matrix-20260929-010715.txt` | MuJoCo stall-injection matrix (§7). |
| `mujoco-ref-20260928-225118` … `-234941` | Earlier runs: the hanging-start hand-over and the effect of box load. |

Reproduce:
- MuJoCo: `bash /work/worldline-g1/sonic/mujoco_ref/run_ref_loop.sh [--driver char_ref.py] [--tag T] [--no-render] [--allow-busy] [-- driver args]`
  (ports 5656/5657/5712, DDS domain 0 on the host `lo`; refuses to start while another deploy runs in the same
  netns or while GPU/Isaac jobs run, unless `--allow-busy`).
- Isaac: `bash /work/worldline-g1/sonic/isaac_char/run_p1_char.sh [--sections turns,gentle|slowwalk,closed,cycles,curves|bodyturn] [--house ID --netns]`
  (host `lo`, DDS domain 0, ports +400; one Isaac instance).
