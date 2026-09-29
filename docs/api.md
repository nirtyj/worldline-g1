# The robot API as built (M2b wave 2)

Status: 2026-09-29, owner robot (wave 2; wave 1: world). The contract lives in `api/` (stdlib only, importable from py3.11 and py3.12); this
page says what each part is for and where it is implemented. PLAN.md §5-§6 is the design; docs/parity.md lists the
deviations from the Ludi doc and the honesty labels; docs/contracts/ has the process wires (P1, P3, the arm chunk).

```
agent/ (harness, validation, belief)  --  api.services.RobotBridge  --  robot/bridge.py G1Robot
                                                                          |-- services/ (navigation, observation,
                                                                          |     reachability, manipulation, speech)
                                                                          |     `-- services/executors/ (registry.py)
                                                                          |-- robot/health.py (capabilities, monitor,
                                                                          |     halt re-send)
                                                                          |-- BodyPort: robot/lite_body.py | robot/body_client.py
                                                                          `-- WorldModel + SimControl: world/ (the only GT reader)
```

## 1. Tools (`api/tools.py`, the only definition)

`speak(text)`, `list_locations(query?)`, `navigate(location, timeout_s?)`, `check_reachability(object_type,
object_id?)`, `manipulate(action: pick|place, object_type, arm?, object_id?, target?, goal?)`,
`wait_and_observe(timeout_s, reason?)`, and `recall(query)` [WL]. `observe(mode: glance|scan)` is internal (the
harness's arrival scan, verify glance, reconcile look). Resources: `navigate` and `manipulate` hold the one body
resource; `check_reachability` is a sense; `speak`, `list_locations`, `recall` are instant. Enums (`location`,
`target`, `object_type`) are filled once per session (Invariant 9). `navigate(location="reach_stance")` is the
reposition a `needs_reposition` answer asks for.

## 2. Results (`api/results.py`)

Every tool, including a rejection, resolves to a frozen `ToolResult`:

| Field | Meaning |
|---|---|
| `tool`, `execution_id` | `execution_id` is None only for a SCHEMA-stage rejection |
| `status` | `succeeded`, `failed`, `cancelled`, `timed_out`, `rejected` (lowercase) |
| `summary` | the one line the planner reads (`api/summaries.py`); stepping stones say `fallback` |
| `data` | `asdict` of the typed result (`NavigateResult`, `ReachabilityResult`, `ManipulationResult`, ...) plus `executor`, `source`, `reason`, `detail` |
| `generation`, `control_epoch` | the fences the execution was created under |
| `source` | `brain`, `harness` or `persona` |
| `t_start`, `t_end` | session clock seconds |
| `observation_id` | the observation the result ran, else the latest glance (never None) |
| `late` | the result came back for an older generation (it updates belief, not the plan) |

Reason codes are `api/reasons.py` (each with the hint the summary appends). M2b adds `beyond_reach` to
check_reachability: the object sits deeper than the arm reaches from any spot the robot can stand on (its body
radius clear of obstacles), so no stand, stance or reposition can help. The result's `detail` gives the distance.

`api.results.is_target(executor)` is the one definition of a target result (`sonic_walk`, `groot_arms`,
`groot_sonic`); every other executor is a stepping stone or unknown, and makes a pass a labelled fallback pass.

## 3. The trace's result rows (`agent/harness.py Runtime._log_result`)

Every result produces exactly one trace row `{"type": "result", ...}` with these fields
(`api.results.RESULT_ROW_FIELDS`), whatever produced it:

| Field | |
|---|---|
| `kind` | `tool` (a started execution: navigate, manipulate, observe, check_reachability, list_locations, wait_and_observe), `speech` (a line played, cut or dropped), `recall`, `rejection` (`api.results.RESULT_ROW_KINDS`) |
| `execution_id`, `tool`, `status`, `observation_id`, `generation`, `control_epoch`, `t_start`, `t_end`, `late` | as in the envelope |

Rows also carry `skill` (the old name of `tool`), `action`, `source`, `executor`, `summary` and `data` (a trimmed
copy: an observation's `saw` and `landmarks` instead of the full LookData); rejections add `stage` and `code`,
speech adds `text`. The older rows (`rejected`, `recall`, `say_queued`) are still written; `agent/procedures.py`
and `agent/narrator.py` read only `kind: tool` result rows.

## 4. Capabilities, sim health and events

`RobotBridge.capabilities()` returns `ServiceHealth(ok, state, detail)` for `navigation`, `manipulation`,
`observation`, `speech`, `body` and `sim` (`api.services.CAPABILITIES`). `robot/health.py CapabilityPolicy` builds
them from the services' own health and the simulator's real-time health, which is world's
(`WorldModel.sim_health()`, `world/sim_health.py`):

| Sim state | When | Rejected at CAPABILITY (validator and `G1Robot.start`) |
|---|---|---|
| `ok` | the default; back from `degraded` at rtf_5s >= 0.94 held 2 s; unknown RTF (lite; a silent P1 is navigation's problem) | nothing |
| `degraded` | rtf_5s < 0.90 held 3 s | `manipulate`: `policy unavailable: sim below real time: DEGRADED, ...` |
| `unsafe` | rtf_3s < 0.85 held 2 s, or < 0.70 at once; out at rtf_3s >= 0.90 held 2 s | `navigate` (`nav_unhealthy`), `manipulate` (`policy_unavailable`), `observe(scan)` (`controller_unavailable`); speech, glances, check_reachability and waits still run |

World applies these to P1's `sim.health` windows (rtf_3s, rtf_5s; the mean of gt.pose's 1 s RTF when P1 sends none)
and does not adopt P1's instant `level` (PLAN §3.5). Thresholds, hold times and the way back are `config/g1.yaml
sim_health`. The `object_type` enum never changes.

`RobotBridge.events()` returns a queue of robot events (subscribing starts `HealthMonitor`, every 0.25 s):

| Event | Producer | Harness |
|---|---|---|
| `capability_changed {capability, skill_id?, ok, state, detail, was}` | HealthMonitor: a capability's or a loaded skill's health changed (`ok` or `state`) | logged (trace row `wake`: why it woke the brain, or null); the brain is woken with a NOTE only on UNSAFE, when a running execution uses the capability, or while a request / own goal / question is open |
| `safety_event {kind: fell, source: sim}` | P1 `gt.event robot_fell` (world's `drain_events`) | stop without asking the model, safety ack |
| `safety_event {kind: fell}` | a walk that ended `fell` (navigation) | same |
| `safety_event {kind: estop}` | `estop()` | stop |
| `safety_event {kind: halt_unacked}` | HaltResender gave up (10 s) | logged |
| `halt_acked {epoch, attempts, latency_ms}` | HaltResender: a late ack | logged by the EventLog |
| `object_fell {id, drop_m, on_floor, ...}`, `sim_event {event, ...}` | other P1 `gt.event`s | EventLog only |
| `body_mode {mode, prev, prev_s, fault, latched, arm_mode, source: body}` | the body's `body.mode` topic (B.3), pushed: `SonicBody.attach_events` hands every `body.*` topic to `G1Robot._on_body_topic` on the runtime's loop (no polling) | logged; the tool state reads FAULT/ESTOP from it |
| `safety_event {kind: fell \| deploy_lost, source: body}` | `body.fault` (a fell within 3 s of P1's `robot_fell` is not repeated) | stop, safety ack |
| `safety_event {kind: runtime_lost, recoverable, body_epoch}` | `body.halted {source: watchdog}`: the body's runtime-session watchdog halted the robot (this runtime's heartbeat lapsed). The gate latches at that epoch, so running body executions end `failed(halted)` | the harness cancels the body actions, then resumes above the epoch (`_recover_body_halt`) unless the user said stop |
| `stale_result {execution_id, op, why}` | `body.stale_command` (a fence refused a command), and the services when a refusal is a stale fence | logged |
| `body_fault {kind: policy_lost \| ..., cleared?}`, `body_event {topic: halted \| resumed \| lease \| session, ...}` | the other `body.*` topics | trace only |

## 5. Halt, cancel, estop (PLAN §5.6); one epoch space; fences, leases, sessions (M2b R.2)

`RobotBridge.halt(control_epoch=None)` latches and returns within 30 ms:
`{accepted, stopped, at_rest, mode: "HOLD", epoch, body_epoch, source, via, rtt_ms, body_kind, latency_ms, resend}`.
On the M2b body (`body.state.fences` present) `SonicBody.halt` is the **B.1 halt lane**: PUSH `{op: halt, epoch}` on
5612, then wait for `body.halted{epoch}` on 5611 (`BodyClient.halt`); `stopped` = the body latched it in time
(`body_kind: new | repeat`; a `stale` answer is re-sent above the body's last resume). Never `command{stop}`. An M1
body gets the old `stop` request, labelled `via: "body stop op (INTERIM: this body has no halt lane)"`. `at_rest`
is judged on `WorldModel.planar_speed()` (< 0.05 m/s). When `stopped` is false, `HaltResender` re-sends the halt
every 100 ms (`BodyPort.send_halt(epoch, wait_s)`) until the body acks it, `resume()` cancels it, or 10 s pass.
Running body executions end `failed(halted)`. `estop()` is the operator kill button only (band on in sim, then
`shutdown_control`; the deploy exits).

**One epoch space.** The halt fences the executions' own `control_epoch`: the harness passes its current one
(`Runtime._robot_halt`), `run_execution`'s escalation passes none (the robot uses the newest control_epoch it has
seen). `HaltGate` (services/common.py) has no counter of its own: `halt(epoch)` latches at it, `halted_since(ce)` is
`latched and ce <= epoch`, and `ManipJob.epoch` is the execution's control_epoch. `resume(control_epoch)` names the
epoch the next executions carry; the harness bumps it before every resume (a user resume, the release of a
runtime-internal halt, the recovery after a body watchdog halt). On the wire `SonicBody` adds one constant per
runtime session (`epoch_base`, `gen_base`: above every epoch and generation the body reported at `hello`), so a
restarted runtime is never stale; `resume` sends the body its own halt epoch and moves the constant up if the next
epoch would not be above it.

**Fences and leases.** `SonicBody.fence(execution)` = `{execution_id, generation, control_epoch, session}` (body
numbers); every body op carries it. Each body execution leases the body (`acquire`/`release`, contract §3.11):
navigate `LOCOMOTION`, a waist scan `ARM_SCRIPT`, a pick/place `ARM_STREAM` (groot_arms), `ARM_SCRIPT`
(sonic_arm_script) or `MANIP` (kinematic_attach) across every attempt; the lease is released at the end, and a
lease this runtime still holds for a finished execution is released by the next acquire. Refusals keep their meaning
(`services.common.refusal`, keyed on `data.why` for `stale_command`): `halted` -> `failed(halted)`, a stale fence
(`why` control_epoch | generation | resume_epoch) -> `failed(stale_result)` + a `stale_result` event, `body_busy`
-> `failed(body_busy)`. A `stale_command` without a fence `why` (a t_wall-stale stream message) is not a stale
result.

**Runtime session.** `SonicBody` sends `hello{session, watchdog_s: 1.0}` at connect and pings every 0.25 s from
its own thread (`BodyClient.hello`), `bye` on close: a dead runtime makes the body hold the robot while it moves
(internal halt, reason `runtime_lost`). A latch left by an earlier runtime is resumed at `hello`.

**Resume and the arms.** `G1Robot.resume` also gives the halt latch's arm pose back to SONIC's own arms
(`arm {release}`) when no hand holds anything; with something in a hand the latched pose stays until the next
arm session (the place) takes it over.

## 6. WorldModel additions (M2b)

| Method | |
|---|---|
| `sim_health()` | `world.sim_health.SimHealth {state, rtf, detail, source, since_s, age_s}` |
| `planar_speed()` | m/s from the GT pose (robot/ never reads the body's copy of gt.pose) |
| `camera_pose(camera=)`, `camera_model(camera)` | `head` (the profile's perception camera) or `ego_view` (GR00T's, Arena's G1 head camera); on Isaac from the real torso link (P1.5) when there is no view override |
| `detections(camera="head", ..., method=None)` | `method=None`: GT geometry (`gt-geometric`), every caller; `method="best"`: the most faithful source: on Isaac, P1's instance-id segmentation of the live view (P1.6, cached 0.2 s, falls back to the geometry and counts it in `seg_stats`). It stalls P1's physics 15-35 ms per call (p1_m2b.md §13), so only the observation service's explicit glances and scan views ask for it; periodic glance records, perception and reachability stay geometric. On the dev box it agreed with the geometry on 66 % of 50 views, and the render supported P1 in each inspected disagreement |
| `palm_position(arm)` | (x, y, z) from P1.5 link poses, else None; `grasp_state()` then also gives `palm_dist_m` |
| `enable_camera(camera, on, consumer=, ttl_s=, hz=)` | P1's `camera` op (OD1: render `ego_view` only while a session needs it; P1 keeps `hz` across off/on, so pass it); NotSupported elsewhere |
| `drain_events()` | P1 `gt.event`s since the last call |
| `capabilities()` (SimControl) | `attach`, `detach`, `object_poses`, `link_poses`, `segmentation`, `enable_camera`, `reset_scene`, `move_object`, `detections:head`, `detections:ego_view`, `p1_contract`, `cameras` |
| `reset_scene(variant, poses=, robot=)`, `move_object(id, center)` | P1.7 fixture ops |

`world/isaac_client.py` speaks the P1 M2b wire (docs/contracts/p1_m2b.md v1) and still runs against an M1 P1,
labelling what it cannot do. `world/frames.py IsaacFrames` reads the head camera itself so the render's camera pose
(P1.10) reaches System 1.

## 7. Manipulation executors (`services/executors/registry.py`)

A profile lists executor names (`config/profiles/<p>.yaml manipulation.executors`); `build_executor(ctx)` turns
each into a `ManipExecutor` (`backend`, `name`, `async run(job, handle) -> ManipOutcome`, `async cancel()`,
`health()`, optional `close()`). `ExecutorContext` carries `name, world, body, clock, gate, events, profile,
manip_cfg, port_offset, extras`. `ManipJob` carries the fence an executor that leases the body needs:
`execution_id, generation, control_epoch` (the body's fence numbers, `SonicBody.fence`), `epoch` (the execution's
control_epoch, which `HaltGate.halted_since` compares), `skill`, `object_type`, and for a place the `spot` and
`target`. `ManipOutcome.data` is the executor's own result data: `ManipulationService` merges its typed fields
(`inferences`, `chunks_dropped`, `attempts`) into `ManipulationResult` and the rest flat into `result.data`; after a
`groot_then_script` fallback the flat keys are the fallback's own, the GR00T attempt's sit under `data.groot`, and
`data.attempts` lists both.

| Name | Backend | Factory |
|---|---|---|
| `lite` | `lite` | KinematicAttachExecutor (in-process attach) |
| `kinematic_attach` | `kinematic_attach` | KinematicAttachExecutor (P1 attach/detach; STEPPING STONE) |
| `sonic_arm_script` | `sonic_arm_script` | `SonicArmScriptExecutor` (R.1, STEPPING STONE): the body's B.7 `arm_script` phases (pick: pregrasp, grasp, a GT gate on the palm, a P1 `fixed_joint` attach, lift, carry into CarryLock; place: lower, P1 detach at the free spot, release, retract) |
| `groot_arms` | `groot` | `services.executors.groot_arms:create` (owner groot_rt; experimental) |
| `groot_sonic` | `groot` | the retired token route, always down |

A factory given as `"module:attr"` is imported only when a profile asks for it; if the import fails the profile still
builds and the executor reports itself down, so its skills are rejected at CAPABILITY. A skill's health is its
executor's `health()`. Results name the executor that ran (`executor.name`). New executors register with
`register_executor(name, factory, backend=...)`.

## 8. Contract tests on the live stack

`tests/contract/conftest.py` has a `sonic` backend marked `box` (deselected by default; `pyproject.toml`
`addopts = -m 'not live and not box'`). It builds `robot.factory.build("sonic", <P1's house>)` on the ports of
`WL_PORT_OFFSET` and is skipped unless `WL_BOX=1` and P1 answers `ping`:

```
WL_BOX=1 WL_PORT_OFFSET=0 .venv-rt/bin/python -m pytest -m box tests/contract -k sonic
```

It moves the real robot (navigate to a surface, a scan, a halt within 30 ms, a cancel): run it on a standing stack
under the box's stack lock, with P5 stopped (two runtimes on one body share its epoch space).
`tests/contract/test_box_runtime.py` adds the R.2 exit: 20 runtime halts mid-walk (`$WL_BOX_OUT/halts.json`), the
waist scan, the approach reposition, a stale fence and the pushed body events.

## 9. Live on the main box (M2b wave 2, owner robot, 2026-09-29)

`ludo-g1-brev2`, H40 (`procthor-train-40`), `sonic` profile, unmodified deploy, body = master at 69d0748..91eebb6
(`body/` unchanged by this wave's robot work), runtime = master + the robot commits, run from a snapshot tree
`/work/robot-wl` (so other owners' uncommitted work was not on the box). Under the main stack lock `robot`, no other
GPU job, P5 stopped during the `-m box` runs. Evidence: `outputs/m2b_wave2/robot/` (laptop).

| Check | Result | Evidence |
|---|---|---|
| E0 `-m box tests/contract` (WL_BOX=1) | 11 passed, 1 skipped (`policy_down` fault injection is fake-only by design); run 1 had one failure (`check_reachability` right after the waist scan answered `base_moving`: SONIC sways the pelvis while the waist returns), fixed by waiting for the base to be still after the scan (`scan_rest_s`, 0.1-0.46 s live) | `box/box2/pytest_box_2.log`, run 1 `box/box/` |
| 20 runtime halts mid-walk (`G1Robot.halt`, B.1 lane) | 20/20 stopped, 20/20 `failed(halted)`; `halt()` p50 0.82 / max 1.77 ms, body.halted round trip max 1.64 ms, body handling ~0.1 ms; speed at the halt 0.24-0.95 m/s; at rest (5 GT samples < 0.05 m/s) within 1.5 s 19/20 (the 20th 1.67 s); 0 falls, pelvis z min 0.741 m; sim RTF 1.007; 42 `body_mode` events reached `robot.events()` | `box/box2/halts.json` |
| Waist scan (F1 arrival scans) | achieved waist yaw -36.3..+36.3 deg at the three holds, `yaw_err_deg_max` 0.8-1.3 | `f1-sonic-r2/episode.jsonl` |
| Reach stance by `approach` | 0.254 m, final error 1.5 cm / 2.1 deg in 3 attempts | same |
| Scripted pick (R.1) from a set stance (0.30 m ahead, 0.18 m right) | succeeded in 15.4 s: grasp palm error median 2.35 cm, GT gate 2.56 cm, `fixed_joint` attach, lifted 0.148 m, carry pose; a 7.4 m carry walk kept it; the place then failed `ik_unreachable` at the service's spot (the IK-envelope retry was added after this run) | on the box only: `/work/robot-wl/outputs/m2b_wave2/robot/pick4/scripted_pick.json` (both boxes were stopped at 11:39 UTC before the pull) |
| F1 with the scripted planner | not passed: navigate, waist scan, check_reachability (`needs_reposition`), the approach reposition and the second check (`reachable`, right arm) ran; the pick failed `ik_unreachable` because HEAD's workspace model put the object 0.456 m ahead (the arm script's IK reaches about 0.42 m at that height, and SONIC stepped the pelvis back ~12 cm during the pregrasp). The calibrated workspace (R.7, owner world-cal, uncommitted at the time) targets 0.22-0.36 m | `f1-sonic-r1/`, `f1-sonic-r2/` (episode, recording: head/chase/top/composite mp4) |
