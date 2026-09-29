# The robot API as built (M2b wave 1)

Status: 2026-09-29, owner world. The contract lives in `api/` (stdlib only, importable from py3.11 and py3.12); this
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
| `ok` | RTF >= 0.95, or unknown (lite; a silent P1 is navigation's problem) | nothing |
| `degraded` | RTF < 0.95 for 5 s (or P1's `sim.health.level`, from its 5 s window) | `manipulate`: `policy unavailable: sim below real time: DEGRADED, ...` |
| `unsafe` | RTF < 0.85 for 3 s (or P1's level, 3 s window) | `navigate` (`nav_unhealthy`), `manipulate` (`policy_unavailable`), `observe(scan)` (`controller_unavailable`); speech, glances, check_reachability and waits still run |

Thresholds and hold times are `config/g1.yaml sim_health`. The `object_type` enum never changes.

`RobotBridge.events()` returns a queue of robot events (subscribing starts `HealthMonitor`, every 0.25 s):

| Event | Producer | Harness |
|---|---|---|
| `capability_changed {capability, skill_id?, ok, state, detail, was}` | HealthMonitor: a capability's or a loaded skill's health changed (`ok` or `state`) | logged, NOTE for the planner, the brain is woken |
| `safety_event {kind: fell, source: sim}` | P1 `gt.event robot_fell` (world's `drain_events`) | stop without asking the model, safety ack |
| `safety_event {kind: fell}` | a walk that ended `fell` (navigation) | same |
| `safety_event {kind: estop}` | `estop()` | stop |
| `safety_event {kind: halt_unacked}` | HaltResender gave up (10 s) | logged |
| `halt_acked {epoch, attempts, latency_ms}` | HaltResender: a late ack | logged by the EventLog |
| `object_fell {id, drop_m, on_floor, ...}`, `sim_event {event, ...}` | other P1 `gt.event`s | EventLog only |
| `body_mode`, `stale_result` | the body (B.3) | logged |

## 5. Halt, cancel, estop (PLAN §5.6)

`RobotBridge.halt()` bumps the halt epoch, latches, and returns within 30 ms:
`{accepted, stopped, at_rest, mode: "HOLD", body_epoch, source, latency_ms, resend}`. `stopped` is true only if the
body acknowledged within the budget (`SonicBody`: a `stop` op, planner IDLE, never `command{stop}`; the M1 body has
no halt lane). `at_rest` is judged on `WorldModel.planar_speed()` (< 0.05 m/s). When `stopped` is false,
`HaltResender` re-sends the halt every 100 ms (`BodyPort.send_halt(epoch, wait_s)`) until the body acks it,
`resume()` cancels it, or 10 s pass. Running body executions end `failed(halted)`. `estop()` is the operator kill
button only (band on in sim, then `shutdown_control`; the deploy exits).

## 6. WorldModel additions (M2b)

| Method | |
|---|---|
| `sim_health()` | `world.sim_health.SimHealth {state, rtf, detail, source, since_s, age_s}` |
| `planar_speed()` | m/s from the GT pose (robot/ never reads the body's copy of gt.pose) |
| `camera_pose(camera=)`, `camera_model(camera)` | `head` (the profile's perception camera) or `ego_view` (GR00T's, Arena's G1 head camera); on Isaac from the real torso link (P1.5) when there is no view override |
| `detections(camera="head", ..., method=None)` | `method=None`: GT geometry (`gt-geometric`), every caller; `method="best"`: the most faithful source: on Isaac, P1's instance-id segmentation of the live view (P1.6, cached 0.2 s, falls back to the geometry and counts it in `seg_stats`). The observation service's glances and scans ask for `best` |
| `palm_position(arm)` | (x, y, z) from P1.5 link poses, else None; `grasp_state()` then also gives `palm_dist_m` |
| `enable_camera(camera, on, consumer=, ttl_s=)` | P1's `camera` op (OD1: render `ego_view` only while a session needs it); NotSupported elsewhere |
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
`execution_id, generation, control_epoch, epoch` (the halt epoch), `skill`, `object_type`, and for a place the
`spot` and `target`.

| Name | Backend | Factory |
|---|---|---|
| `lite` | `lite` | KinematicAttachExecutor (in-process attach) |
| `kinematic_attach` | `kinematic_attach` | KinematicAttachExecutor (P1 attach/detach; STEPPING STONE) |
| `sonic_arm_script` | `sonic_arm_script` | stub until the body's arm script (B.7) |
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
under the dev-box stack lock.
