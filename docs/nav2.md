# Navigation with ROS 2 Nav2 (go_to backend `nav2`)

Status: **implemented and tested end to end on the fakes** (fake P1 with a kinematic G1 in the real
`procthor-train-38` occupancy + fake SONIC deploy; the real wl-body, bridge and Nav2). **Not yet run against Isaac or
the real SONIC deploy** (SONIC forward walking is being fixed separately). Code: `nav2/`, `body/velocity.py`,
`body/nav2_backend.py` (+ small hooks in `body/service.py`, `body/config.py`). Evidence:
`outputs/m1/nav2/fake-rpp-final/` and `outputs/m1/nav2/fake-mppi-final/` (box: `/work/worldline-g1/outputs/m1/nav2/`).

Owner decision (2026-09-28): Nav2 is the navigation backend now; ground truth stays the localisation source; the
built-in A* + pure pursuit stays as a fallback backend.

## 1. Architecture

```
            BodyClient.go_to(x, y, yaw)                    (unchanged API)
                     │ ROUTER :5610
   ┌─────────────────▼──────────────────────────────┐      ┌──────────────────────────────────────────────────┐
   │ P3 wl-body (.venv, Python 3.11)                │ REQ  │ nav2/ros_bridge.py (system Python 3.12 + Jazzy)  │
   │  go_to ─► select_motion (NAV_BACKEND)          │─────►│  REP :5620  goto | cancel | status | plan |      │
   │   nav2  ─► Nav2GoToMotion                      │ 5620 │             escape | reload_map | stats | ping   │
   │   astar ─► path_follower.GoToMotion (fallback) │      │  goto = ComputePathToPose pre-check              │
   │  op velocity {vx,vy,wz,goal_id} ◄──────────────┼──────│        + NavigateToPose                          │
   │   ─► VelocityCommander ─► SonicMux ─► :5556    │DEALER│  /cmd_vel ─► op velocity (only while a goal is   │
   └──────────────────────────────┬─────────────────┘ 5610 │             active)                              │
          SONIC planner (P2) ◄────┘                         │  gt.pose (:5601) ─► /odom + TF odom->base_link   │
                                                            │  get_occupancy (:5600) ─► /map (latched, raw)    │
                                                            │  static TF map->odom = identity                  │
                                                            └──────────┬───────────────────────────────────────┘
                                                                       │ ROS 2, domain 42, Fast DDS, localhost
                                                            ┌──────────▼───────────────────────────────────────┐
                                                            │ Nav2 (one component_container_isolated)          │
                                                            │ bt_navigator ─ planner_server (Smac 2D)          │
                                                            │ controller_server (RPP) ─► cmd_vel_nav ─►        │
                                                            │ velocity_smoother ─► /cmd_vel                    │
                                                            │ behavior_server (wait) ─ lifecycle_manager       │
                                                            └──────────────────────────────────────────────────┘
```

- **One ROS process of ours.** The body venv is Python 3.11 and Jazzy's rclpy is built for 3.12, so every ROS
  dependency lives in `nav2/ros_bridge.py` (system python3 + `/opt/ros/jazzy`), which talks ZMQ to the rest. Nav2
  itself is C++ in one container process. Nothing in P1, P2 or the body imports ROS.
- **DDS isolation.** `nav2/ros_env.sh`: `ROS_DOMAIN_ID=42`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` (Jazzy default),
  `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, `CYCLONEDDS_URI` unset. The G1 low-level topics (`rt/lowcmd`,
  `rt/lowstate`, ...) are unitree_sdk2 / CycloneDDS on domain 0 (build-phase tests: domain 7), so Nav2 traffic can
  never reach them: different domain and different DDS vendor. `ros2 doctor` on domain 42: all 5 checks pass.
- **Ports.** One new contract port: `nav_bridge` = 5620 (+ offset), REP, bound by the bridge
  (`body/config.py BASE_PORTS`). The bridge is a client of P1 (5600, 5601) and of the body (5610).
- **Frames.** `map -> odom` static identity (ground-truth localisation; the later no-GT stage replaces it with
  AMCL / KISS-ICP without touching Nav2's config), `odom -> base_link` from every `gt.pose` (50 Hz, stamped with
  `gt.pose.t_wall`). `base_link` = the pelvis projected on the floor (x, y, yaw; roll, pitch, height dropped).
  `/odom` twist is in `base_link`.
- **Map.** The bridge asks P1 `get_occupancy` (the same call the body makes), loads the npz and publishes the
  **raw** `occ` grid (100 blocked or outside, 0 free) as a latched `OccupancyGrid` on `/map`; Nav2's inflation layers
  add the robot radius. The grid convention is identical (row 0 = lowest y, origin = corner of cell (0,0)), so no
  flip. A map_server-compatible `map.pgm` + `map.yaml` is also written to the run dir (`nav2/wl_map.py`). No
  map_server, no AMCL.

## 2. wl-body changes (additive)

### 2.1 Streaming op `velocity`

Request `{"op": "velocity", "args": {vx, vy, wz, stream?, t_wall?, watchdog_s? (0.3), end?}}` in the robot body
frame (REP-103: x forward, y left; m/s, rad/s), exactly what a ROS controller publishes for `base_link`.

- The first message starts a `velocity` op (normal `accepted` event, pre-empts the active motion like any motion,
  needs `stand` first). Later messages with the same `stream` **update** it: reply `state: "done"`, **no events and
  no op record** (they are not ops). `end: true` stops it. Messages whose `t_wall` is older than the watchdog are
  dropped as stale.
- **Watchdog:** no fresh message for 0.3 s ⇒ planner IDLE with facing = GT yaw at the trip; the op ends `succeeded`
  with `ended_by: "watchdog"` once the robot is settled (a stream that resumes before that continues the op).
- **Conversion** (`body/velocity.py`, reuses SonicMux and its planner-frame conversion `body/frames.py`):
  - `|v| < 0.05` ⇒ no translation. Otherwise `SLOW_WALK`, movement = `R(yaw_gt)·(vx, vy)` as a world direction,
    speed = clamp(|v|, 0.2, 0.8), ≤ 0.4 when `|vy| > |vx|` (strafe). A 0.05-0.2 m/s request walks at 0.2 m/s
    (SLOW_WALK's floor); the controller closes the loop on the pose.
  - Rotation: SONIC takes a facing, not a yaw rate. A facing setpoint integrates `wz` and is kept within ±25° of the
    GT yaw (anti-windup; the lead also absorbs SONIC's short IDLE turn, m1.md §1.9). No translation and `|wz| > 0.02`
    ⇒ `IDLE` + facing = turn in place (keyboard Q/E).
- Nav2's `/cmd_vel` arrives through the same op with `args.goal_id` = the body go_to op id: those messages only feed
  that active go_to and never start a motion; a stale `goal_id` is rejected without an event.

### 2.2 go_to backend selection

`NAV_BACKEND=nav2|astar` (env, default **nav2**; `body.service --nav-backend`; per call `args.backend`). With nav2
selected, each go_to pings the bridge (≤ 0.5 s); if it does not answer or Nav2 is not active, go_to **falls back to
A*** and says so in the accepted and terminal data (`backend: "astar"`, `backend_requested: "nav2"`,
`backend_fallback: "<why>"`; `NAV_FALLBACK=0` disables the fallback). `BodyClient` is unchanged; results keep the
contract keys (`goal, pos_err, yaw_err_deg, path_len_m, replans, approach_attempts, stuck_events, plans[]`) plus
`backend, nav2 {state, error_name, feedback, cmd_vel_forwarded}, velocity {...}, escapes, nav2_retries`.

### 2.3 `Nav2GoToMotion` (body/nav2_backend.py)

1. **start** (synchronous, like A*): bridge `goto` = `ComputePathToPose` pre-check, so `no_path`,
   `goal_in_obstacle`, `start_in_obstacle` come back in the reply with the plan (length, ≤ 60 points, plan time).
   Without a goal yaw the bridge uses the path's arrival heading as the NavigateToPose orientation (no final spin);
   a goal moved by the planner tolerance is snapped to the path end (`goal_snapped`). A `speed` arg becomes a
   `/speed_limit`.
2. **navigate**: `/cmd_vel` → `VelocityCommander` every control tick; status poll 4 Hz; a silent Nav2 (replanning,
   recovery wait) trips the 0.3 s watchdog ⇒ IDLE.
3. Nav2 **succeeded** ⇒ IDLE settle holding the goal yaw ⇒ GT check against `final_pos_tol` (0.25) /
   `final_yaw_tol_deg` (12); one re-approach (goal re-sent) if outside, else `final_error`.
4. **escape** (the humanoid replacement for Nav2's BackUp): if the robot is inside the inscribed zone (clearance <
   0.30 m) Nav2 cannot plan (`START_OCCUPIED`). The bridge finds the shortest straight step (16 directions, ≤ 0.6 m)
   to clearance ≥ 0.37 m on the map; the body walks it at 0.2 m/s with the facing held, settles and re-sends the goal
   (≤ 2 per go_to). Also used when Nav2 fails mid-route and the robot ended inside the inscribed zone.
5. cancel / pre-emption / body timeout ⇒ bridge `cancel` (the bridge stops forwarding `/cmd_vel` for that goal
   immediately) and the service's IDLE hold.

Failure reasons (Nav2 Jazzy error codes → body reasons, `nav2/ros_bridge.py REASONS`):

| Nav2 | body reason |
|---|---|
| ComputePathToPose `NO_VALID_PATH` 208, `TIMEOUT` 207, `GOAL_OUTSIDE_MAP` 204 | `no_path` |
| `GOAL_OCCUPIED` 206 | `goal_in_obstacle` |
| `START_OCCUPIED` 205, `START_OUTSIDE_MAP` 203 | `start_in_obstacle` (after the escape attempts) |
| FollowPath `FAILED_TO_MAKE_PROGRESS` 105, `PATIENCE_EXCEEDED` 104, `NO_VALID_CONTROL` 106 | `stuck` |
| FollowPath `CONTROLLER_TIMED_OUT` 107; body `timeout_s` | `timeout` |
| TF errors 102 / 202 | `tf_error` |
| NavigateToPose canceled by someone else | `nav2_canceled` |
| bridge unreachable > 3 s mid-goal | `nav2_unavailable` |

## 3. Nav2 configuration (`nav2/params/nav2_g1.yaml`)

| Item | Setting | Why |
|---|---|---|
| Nodes | controller, planner, behavior (wait only), bt_navigator, velocity_smoother, lifecycle manager in **one** `component_container_isolated` | one DDS participant; 7.5% vs 12.1% of a core for separate processes (§5) |
| Not run | map_server, AMCL, smoother/waypoint/route/docking servers, collision_monitor | map + GT from the bridge; no sensors in M1 |
| Controller | **Regulated Pure Pursuit** (default), 20 Hz: 0.5 m/s desired, min approach / regulated min speed 0.2, rotate to heading > 45° at 0.6 rad/s, no reversing, collision look-ahead 1.5 s | §3.1 |
| Alternative | MPPI, Omni model (`params/controller_mppi.yaml`, `up.sh --controller mppi`): vx -0.1..0.6, vy ≤ 0.25, wz ≤ 0.8, batch 1000 × 40 steps | measured, §3.1 |
| Planner | **Smac 2D**, `cost_travel_multiplier` 2 (centred in doorways, like the body's clearance-weighted A*), tolerance 0.25 m, no unknown | NavFn hugs the inflation edge |
| Costmaps | static + inflation only; `robot_radius` 0.30, inflation 0.45 (scaling 4.0); local 3 m rolling window at 5 Hz, global 1 Hz | TODO in the yaml: obstacle/voxel layer from the head D435 / a MID-360 + collision_monitor |
| Limits | velocity smoother (open loop, 20 Hz): v ≤ [0.6, 0.4, 0.8], accel [0.6, 0.5, 1.2], decel [0.8, 0.6, 1.5] | gentle; SONIC re-plans on every command change |
| Goal | `SimpleGoalChecker` stateful, 0.25 m / 12°; progress checker 0.25 m in 10 s | turning in place for a few s is still progress |
| BT | `nav2/bt/navigate_humanoid.xml`: default replanning tree (1 Hz), recoveries = clear costmaps + wait 2 s, **no Spin, no BackUp**, 2 outer retries; BT tick 50 ms | a fast spin or a backing burst is the wrong surprise for a whole-body policy; the body's escape replaces BackUp |

Launch detail worth knowing: in composition mode the full params files are also passed to the **container**
process, because the costmap sub-nodes are created from the process-global arguments; launch_ros only hands each
component its own keys (without this the costmaps silently ran with defaults: radius 0.1, an obstacle layer).
Harmless start-up log: Smac's "Inflation layer either not found or inflation is not set sufficiently for optimized
non-circular collision checking" (the footprint is circular, that check does not apply).

### 3.1 Controller choice: RPP by measurement, not MPPI

SONIC cannot walk slower than 0.2 m/s and follows commands with ~0.3-0.5 s of lag. MPPI optimises over continuous
speeds from the *measured* velocity: at rest it first deadlocked (its first command is capped at measured +
a_max·dt = 0.03 m/s, below the body's 0.05 dead band ⇒ `FAILED_TO_MAKE_PROGRESS`, fixed by moving the gentle ramp to
the velocity smoother), then, once moving, it kept asking for 0.05-0.2 m/s near goals and in corners, which the body
must round up to 0.2 m/s; its predictions are then wrong and it orbits the goal and swings wide. Same fakes, same 4
goals, final code (`fake-rpp-final` vs `fake-mppi-final`):

| Goal | Plan m | RPP walked m / s / GT err m | MPPI Omni walked m / s / GT err m |
|---|---|---|---|
| kitchen (no yaw) | 10.2 | 10.4 / 32.1 / 0.142 | 18.2 / 64.7 / 0.060 |
| bedroom (yaw 180°) | 5.6 | 5.8 / 19.7 / 0.092 | 8.2 / 29.4 / 0.221 |
| living room (yaw 90°) | 15.6 | 15.6 / 44.3 / 0.138 | 20.1 / 72.3 / **failed final_error 0.273** |
| kitchen 2 (yaw 0°) | 8.9 | 9.1 / 23.8 / 0.145 | 10.8 / 34.9 / 0.197 |
| walking ticks where a < 0.2 m/s command was rounded up to 0.2 | | 2.1% | 35% (goals 0, 1, 3) |
| Nav2 CPU while navigating (mean / p95) | | 7.5% / 9% of a core | 11.6% / 15% |

RPP tracks the Smac path within a few cm, never asks for < 0.2 m/s while moving and turns in place when the path
turns sharply, which is how SONIC turns anyway (IDLE + facing). MPPI's advantage (strafing through tight spots) did
not matter in this house. Revisit MPPI on the real SONIC dynamics if strafing is needed.

## 4. Tests on the fakes

`bash nav2/tools/run_fake_e2e.sh [--controller rpp|mppi]` (tmux `nav2-fakes` + `nav2-e2e`, all contract ports
+700, preflight refuses busy ports): `nav2/tools/fake_stack.py` (the body agent's `tools/fake_p1.py` with
`--house-dir assets/houses/procthor-train-38` + `tools/fake_deploy.py`, which decodes SonicMux's planner messages
with the planner frame = GT yaw at start, 1 s planner timeout, and integrates them kinematically with a +3° heading
bias), the real `body.service` with `NAV_BACKEND=nav2`, `nav2/up.sh`, then `nav2/tools/nav2_fake_test.py` which uses
**only BodyClient** and checks every result against GT from its own `gt.pose` subscription. The same test runs
unchanged against the real stack (without the fake-only `escape` and `stuck`).

Result, RPP (`outputs/m1/nav2/fake-rpp-final/metrics.json`): **all 7 tests pass**.

| Test | Result |
|---|---|
| goals: 4 go_to across 3 rooms (living → kitchen → bedroom → living → kitchen) | 4/4 succeeded, backend nav2, GT error 0.092-0.145 m, yaw error ≤ 1.6°, walked 1.00-1.04× the plan length, 0 re-approaches, 0 watchdog trips |
| unreachable | goal inside the bed: `failed no_path` (NO_VALID_PATH) in 13 ms; goal outside the map: `failed no_path` (GOAL_OUTSIDE_MAP) in 4 ms; no motion. (House 38 has no free-but-disconnected pocket at r 0.30; the test also covers that case when a house has one.) |
| cancel mid-route | after 1.0 m at 0.46 m/s: go_to `canceled` (stop), Nav2 goal `canceled`, 0 `/cmd_vel` forwarded after the cancel, GT speed < 0.05 m/s 0.78 s later |
| timeout | `timeout_s` 4 ⇒ `failed timeout` at 4.01 s, Nav2 goal canceled |
| escape (fake `reset_robot` into clearance 0.225 m) | START_OCCUPIED ⇒ 0.20 m step at 202° (moved 0.29 m) ⇒ goal re-sent ⇒ succeeded, GT error 0.15 m |
| velocity watchdog | 70 messages at 20 Hz (turn in place wz 0.5 for 1.5 s, then vx 0.4 for 2 s), then silence: planner IDLE **0.320 s** after the last message; op `succeeded ended_by watchdog`; GT stopped 1.0 s after the last message (fake deploy lag) |
| stuck (fake `spawn_obstacle`, physics only, blocks the bedroom door) | FAILED_TO_MAKE_PROGRESS after 6 recoveries ⇒ `failed stuck` in 76 s |

Artifacts per run: `trajectory.png` (all runs over the map with the inscribed/inflation bands and Nav2's plans),
`goal_<k>.png` / `cancel.png` (map zoom + `/cmd_vel`, SONIC planner messages and GT speed/yaw-rate traces),
`watchdog.png`, `nav2/cmd_vel.jsonl` (every `/cmd_vel`, 4542 in the RPP run), `body/planner_cmds.jsonl` (every
planner command change as sent to SONIC), `nav2/goals.jsonl`, `pose.csv`, `events.jsonl`, `cpu.csv`,
`cpu_summary.json`. Unit tests: `body/tests/test_velocity.py` (conversion, clamps, turn in place, anti-windup,
watchdog, stale drop, backend selection); the existing 29 body tests still pass (go_to falls back to A* there).

## 5. CPU

Per process from `/proc` (1 s samples, 100% = one core), composed RPP config, box shared with other agents' jobs:

| Process | standing (idle) | navigating mean / p95 / max | RSS |
|---|---|---|---|
| Nav2 container | 6.3% | 7.5% / 9% / 10% | 117 MB |
| ros_bridge.py | 2.0% | 4.5% / 6% / 6% | 99 MB |
| `ros2 launch` | 0% | 0.1% | 80 MB |
| **ROS side total** | **~8%** | **~12% of one core** (≈ 0.75% of the 16-vCPU box) | ~300 MB |

Separate processes (RPP, `--no-composition`), navigating: controller 2.6%, bt_navigator 2.7%, planner 1.8%,
lifecycle manager 1.9%, behaviour server 1.6%, velocity smoother 1.5% = 12.1%: the cost is mostly per-node
overhead (DDS, 50 Hz TF into every TF listener), not the algorithms, hence composition. MPPI (batch 1000 × 40)
adds ~4% (11.6% mean, 17% max). Kept low by: BT tick 50 ms instead of 10 ms (also 5× fewer feedback messages into the
Python bridge, bridge 10.7% → 4.5%), local costmap 3 m at 5 Hz, costmap publishing 1 / 0.5 Hz, no sensors layers.
`up.sh` runs Nav2 and the bridge at `nice 5` (`NAV2_NICE`) and accepts `NAV2_TASKSET` to keep them off the wall-clock
deploy's cores.

## 6. Launch

```bash
bash nav2/install_ros.sh                       # once: Jazzy ros-base + navigation2 + nav2-bringup (niced)
source nav2/ros_env.sh                         # any manual ros2 command (domain 42)
bash nav2/up.sh [--port-offset N] [--session wl-nav2] [--controller rpp|mppi] [--no-composition] [--log-dir D]
python3 nav2/tools/bridge_cli.py [--port-offset N] ping | stats | status ID | plan X Y | escape | wait
bash nav2/down.sh [--session wl-nav2]
```

`up.sh` needs P1 serving `gt.pose` and `get_occupancy` (the bridge retries until it does; Nav2's costmaps only
activate once TF exists); it waits until the bridge reports `nav2_ready` (map published, `gt.pose` fresh, lifecycle
nodes active; ~3 s) and prints the run dir. For the integrated stack, `nav2/m1_hook.sh` gives the integrate agent
two lines to adopt (not applied here; `scripts/` is theirs):

```bash
# scripts/m1_up.sh, right after "body up" (before the deploy, so Nav2's ~1 s start-up burst cannot disturb SONIC):
bash "$WL/nav2/m1_hook.sh" up "$OFFSET" "$SESSION" || say "WARNING: Nav2 not up; go_to falls back to A*"
# scripts/m1_down.sh, before the body stop:
bash "$WL/nav2/m1_hook.sh" down "$OFFSET" "$SESSION" || true
```

`NAV_BACKEND=astar` skips Nav2 and makes the body use A* directly. In `scripts/body_netns_e2e.sh` the hook would run
inside the private network namespace with the rest, DDS discovery then happening on that namespace's `lo`
(untested).

## 7. Known issues and next steps

- **Fakes only.** First real-stack steps: adopt the hook, run `nav2_fake_test.py --tests
  goals,unreachable,cancel,timeout,watchdog --label "Isaac + SONIC"`, then tune on SONIC: `facing_lead_max_deg`
  (25°) against SONIC's IDLE-turn shortfall, RPP `rotate_to_heading_angular_vel` (0.6) against SONIC's turn rate,
  `desired_linear_vel` (0.5), the velocity smoother accelerations. Needs SONIC forward walking fixed first.
- **Stuck detection is slow** (~76 s): progress checker 10 s × FollowPath retry × 3 outer attempts. Fine with a static
  map; lower `movement_time_allowance` or the BT retries once sensors make recoveries meaningful.
- **Sensors:** TODO in `params/nav2_g1.yaml` (obstacle/voxel layer from the head D435 depth or a MID-360 LiDAR, plus
  collision_monitor). Until then Nav2 only knows the static house map; spawned or moved objects are invisible to it.
- Port offset hygiene: other agents run full stacks at +300 and +400 on this box. The e2e runner now refuses to start
  unless every port of its offset is free (an early run at +400 connected the bridge to another agent's real P1: it
  only read `gt.pose` and got a refused `get_occupancy`; no command was sent).
- `docs/architecture.html` (Navigation section, "PathFollower A*") predates this; its owner should switch it to Nav2.
- **Learned navigation checkpoint (owner's question).** NVIDIA **COMPASS** (`nvidia/COMPASS` on Hugging Face,
  NVlabs/COMPASS, NVIDIA Open Model License) is a cross-embodiment mobility policy on the X-Mobility backbone whose
  model card lists `g1` among the supported embodiments; inputs are a camera image, robot state and a route,
  outputs are desired linear/angular velocities, runtime TensorRT, with a ROS 2 deployment folder. That output is
  exactly the `velocity` op / `/cmd_vel` interface built here, so it could replace the RPP controller (with Smac still
  providing the route) without touching wl-body or SONIC. Not downloaded or evaluated yet: it targets Isaac Sim
  4.5 / Isaac Lab 3.0 beta for training, its G1 locomotion layer differs from SONIC, and its camera must match ours.
