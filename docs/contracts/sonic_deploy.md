# SONIC deploy contract (P2 wl-sonic)

Owner: deploy component (`worldline-g1/sonic/`). Status: **draft v1, read from code; sections marked
MEASURED are confirmed on the box with the MuJoCo reference loop** (see "Evidence" at the end).

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
  (domain 7 is fine for bridge-only tests). An alternative for parallel tests is a separate network
  namespace per pair (`sudo unshare -n` gives a private `lo`). This has not been tested.
- Build-phase test ports (+100): `run_deploy.sh --zmq-port 5656 --zmq-out-port 5657`.

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
of `rt/lowcmd` messages whose leg `q` targets change at about 50 Hz".

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
6. Sleep to hold wall-clock pace (`MJ/base_sim.py:627-631`). The loop does not catch up when it runs
   late, so RTF < 1 means the deploy sees a slower world.

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
  `yaw0 = world pelvis yaw at start` and send `R(-yaw0) · v_world`. [MEASURED: see §6]
- Keepalive: every message is consumed once. If none arrives for **1 s**, the manager forces IDLE and
  keeps the last facing (`zmq_manager.hpp:582-630`). Send ≥10 Hz, and resend identical messages;
  re-sending the same values does not force a replan.
- Replanning: on a change of mode, facing or height, or of speed or movement while moving, and otherwise
  every 1 s while moving (`DEPLOY.cpp:3700-3732`; intervals `DEPLOY.cpp:231-234`). The planner runs at 10 Hz; its 30 Hz output is
  resampled to 50 Hz.

### Start sequence (what `sonic/mujoco_ref` does; the integrated stack should do the same)
1. Simulator up, publishing `rt/lowstate` + `rt/secondary_imu` at the physics rate, elastic band **on**.
2. Start the deploy. It loads TensorRT engines (`<onnx>.trt` cache next to each ONNX; the first run builds
   them, which takes minutes), waits about 5 s in `MotionSwitcherClient` (no robot service in sim,
   `DEPLOY.cpp:2279-2289`), creates the DDS channels and threads. Then `Control()` in INIT ramps from the
   current joint positions to `default_angles` over 3 s (`duration_ = 3.0`, `DEPLOY.cpp:2758-2793`) and
   prints **`Init Done`**.
3. Bind the PUB on the ZMQ port **before** step 2, or wait about 0.5 s after binding (ZMQ slow joiner).
   Then send `command {start:1, stop:0, planner:1}` a few times (every 100 ms until the log shows
   `[ZMQManager] motion name is planner_motion` / `Planner initialized successfully!`).
4. Start the planner keepalive at ≥10 Hz with `mode=0, movement=[0,0,0], facing=[1,0,0]` (stand).
5. Release the elastic band. In MuJoCo the band holds the pelvis at 1.0 m, so the robot drops about
   0.2 m onto its feet, as in the upstream manual procedure (`docs/source/getting_started/quickstart.md:121-124`).
   Lowering it first is gentler.
6. Walk: `mode=1, movement=dir, facing=heading, speed=0.3..0.6`.

## 4. ZMQ output: `g1_debug`
`b"g1_debug" + msgpack map` every control tick (50 Hz), non-blocking send. Keys include `base_quat`,
`base_ang_vel`, `body_torso_quat`, `body_q[29]`/`body_dq[29]` (MuJoCo order), `last_action[29]`,
`token_state[64]`, `init_base_quat`, `delta_heading`, `base_trans_target`, `base_quat_target`,
`body_q_target[29]` (`INC/output_interface/zmq_output_handler.hpp:18-75`). There is no explicit
"planner mode" key. Planner mode is shown by the deploy log (`[ZMQManager] Planner enabled`,
`Planner initialized successfully!`) and by `base_trans_target` advancing while walking.

## 5. Operational gotchas (from code; confirm on the box)
- `main()` waits with `while(!stop) sleep(0.02);`. `sleep()` takes an `unsigned int`, so this is
  `sleep(0)`: **a busy spin at 100 % of one core**. `SetThreadPriority()` also pins the main thread to
  **CPU 0** (`DEPLOY.cpp:2624-2640, 4523-4526`); SCHED_FIFO fails silently without CAP_SYS_NICE.
  Budget one full vCPU for P2, and keep Isaac's busy threads off CPU 0 if possible (`taskset`).
- First start builds the TensorRT engines from the ONNX files (encoder, decoder, planner). Later starts
  reuse `*.trt` next to the ONNX files. Their hash is checked (`src/TRTInference/InferenceEngine.cpp:143, 330-372`).
- When the deploy exits (O, `stop`, lowstate timeout) it publishes one damping command. With no lowcmd
  after that, the simulator keeps the last command: in MuJoCo the robot goes limp and falls.
- The deploy never reads the robot's XY position. Position feedback is the caller's job (ground truth in sim).

## 6. Evidence (MuJoCo reference loop)
_Filled in after the runs; see `/work/worldline-g1/outputs/m1/deploy/`._
