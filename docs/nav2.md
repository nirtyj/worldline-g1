# Navigation with ROS 2 Nav2 (go_to backend `nav2`)

Status: **implemented and tested end to end on the fakes; verifier findings fixed (2026-09-29, §3.2, §4.1)** (fake P1 with a kinematic G1 in the real
`procthor-train-38` occupancy + fake SONIC deploy; the real wl-body, bridge and Nav2). **Not yet run against Isaac or
the real SONIC deploy** (SONIC forward walking is being fixed separately). Code: `nav2/`, `body/velocity.py`,
`body/nav2_backend.py` (+ small hooks in `body/service.py`, `body/config.py`). Evidence:
`outputs/m1/nav2/fake-rpp-final/`, `outputs/m1/nav2/fake-mppi-final/` and, after the fixes,
`outputs/m1/nav2/fix-e2e-20260929-033108/` (box: `/work/worldline-g1/outputs/m1/nav2/`, the fix run in
`/work/wl-nav2fix/outputs/m1/nav2/`).

Owner decision (2026-09-28): Nav2 is the navigation backend now; ground truth stays the localisation source; the
built-in A* + pure pursuit stays as a fallback backend.

## 1. Architecture

```
            BodyClient.go_to(x, y, yaw)                    (unchanged API)
                     │ ROUTER :5610
   ┌─────────────────▼──────────────────────────────┐      ┌──────────────────────────────────────────────────┐
   │ P3 wl-body (.venv, Python 3.11)                │DEALER│ nav2/ros_bridge.py (system Python 3.12 + Jazzy)  │
   │  go_to ─► select_motion (NAV_BACKEND)          │─────►│  REP :5620  goto | cancel | status | plan |      │
   │   nav2  ─► Nav2GoToMotion (async)              │ 5620 │   goal_check | escape | reload_map | stats | ping│
   │   astar ─► path_follower.GoToMotion (fallback) │      │  goto = goal check (like A*) + ComputePathToPose │
   │  op velocity {vx,vy,wz,goal_id} ◄──────────────┼──────│        pre-check + NavigateToPose                │
   │   ─► VelocityCommander ─► SonicMux ─► :5556    │DEALER│  /cmd_vel ─► op velocity (only while a goal is   │
   └──────────────────────────────┬─────────────────┘ 5610 │             active)                              │
          SONIC planner (P2) ◄────┘                         │  gt.pose (:5601) ─► /odom + TF odom->base_link   │
                                                            │  get_occupancy (:5600) ─► /map (latched, raw)    │
                                                            │  static TF map->odom = identity                  │
                                                            └──────────┬───────────────────────────────────────┘
                                                                       │ ROS 2, domain 42 + offset//100, Fast DDS, lo
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
- **DDS isolation.** `nav2/ros_env.sh`: `ROS_DOMAIN_ID = 42 + port_offset // 100` (integrated stack 42, fake E2E at
  +700 → 49, +900 → 51; `WL_ROS_DOMAIN_ID` overrides), `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` (Jazzy default),
  `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, `CYCLONEDDS_URI` unset. The G1 low-level topics (`rt/lowcmd`,
  `rt/lowstate`, ...) are unitree_sdk2 / CycloneDDS on domain 0 (build-phase tests: domain 7), so Nav2 traffic can
  never reach them: different domain and different DDS vendor. `ros2 doctor` on domain 42: all 5 checks pass.
  Two stacks on one domain would share `/cmd_vel`, `/tf`, `/odom`, `/map` and accept each other's
  `navigate_to_pose` goals, so `up.sh` refuses to start when `ros2 node list` already shows a `wl_ros_bridge` on the
  domain, and a running bridge that sees a second one reports not-ready (the body then falls back to A*).
- **Ports.** One new contract port: `nav_bridge` = 5620 (+ offset), REP, bound by the bridge
  (`body/config.py BASE_PORTS`, docs/contracts/m1.md §0). The body talks to it through a DEALER socket with
  `ZMQ_IMMEDIATE` (no queueing to an absent bridge) and request ids (`rid`, echoed by the bridge), so no call ever
  waits on a REQ state machine. The bridge is a client of P1 (5600, 5601) and of the body (5610).
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
  needs `stand` first). Later messages **must pass the same `stream`** (default: the op id returned in the accepted
  reply) to **update** it: reply `state: "done"`, **no events and no op record** (they are not ops). A message
  without a matching `stream` starts a new op and cancels the old one. `end: true` stops it. Messages whose `t_wall`
  is older than the watchdog are dropped as stale.
- **Watchdog:** no fresh message for 0.3 s ⇒ planner IDLE holding the last **commanded** facing (walk diagnosis: never
  rebuild commands from the measured yaw); the op ends `succeeded` with `ended_by: "watchdog"` once the robot is
  settled (a stream that resumes before that continues the op).
- **Conversion** (`body/velocity.py`, reuses SonicMux and its planner-frame conversion `body/frames.py`):
  - `|v| < 0.05` ⇒ no translation. Otherwise `SLOW_WALK`, movement = `R(yaw_gt)·(vx, vy)` as a world direction,
    speed = clamp(|v|, 0.2, 0.8), ≤ 0.4 when `|vy| > |vx|` (strafe). A 0.05-0.2 m/s request walks at 0.2 m/s
    (SLOW_WALK's floor); the controller closes the loop on the pose.
  - Rotation: SONIC takes a facing, not a yaw rate, and every IDLE facing change is a re-plan. `wz` is integrated
    into a demanded facing that goes to SONIC through the same stepped rule as `Motion.turn_cmd`
    (docs/walk_diagnosis.md): at most 30° from the last **commanded** facing, advancing only while the body is within
    15° of it (anti-windup against the command, never a clamp to the measured yaw). Walking: small changes go out
    every tick (the mux's 2° deadband batches them), so straight walking and pure-pursuit curvature stay
    responsive. No translation and `|wz| > 0.02` ⇒ `IDLE` turn in place (keyboard Q/E) in discrete steps: one step
    per 0.4 s at most (at once when a full 30° step is pending), plus the remainder when the rotation request stops.
- Nav2's `/cmd_vel` arrives through the same op with `args.goal_id` = the body go_to op id: those messages only feed
  that active go_to and never start a motion; a stale `goal_id` is rejected without an event.

### 2.2 go_to backend selection

`NAV_BACKEND=nav2|astar` (env, default **nav2**; `body.service --nav-backend`; per call `args.backend`). With nav2
selected, each go_to pings the bridge with a **50 ms** budget; "unreachable" is cached for 3 s (a go_to in that window
does not ping again), so a missing bridge costs the 50 Hz loop at most one 50 ms wait per 3 s (and a bridge that is
not running at all fails at once: `ZMQ_IMMEDIATE`). If it does not answer or Nav2 is not active, go_to **falls back
to A*** and says so in the accepted and terminal data (`backend: "astar"`, `backend_requested: "nav2"`,
`backend_fallback: "<why>"`; `NAV_FALLBACK=0` disables the fallback). `BodyClient` is unchanged; results keep the
contract keys (`goal, pos_err, yaw_err_deg, path_len_m, replans, approach_attempts, stuck_events, plans[]`) plus
`backend, goal_requested, nav2 {state, error_name, feedback, cmd_vel_forwarded}, velocity {...}, escapes,
nav2_retries, nav2_recoveries, goto_reply_ms`. Under nav2, `replans` = goal re-sends by the body (escape steps,
retries, re-approaches; Nav2's own 1 Hz replanning is not counted), `stuck_events` = Nav2 `stuck` failures with the
body's reaction (`escape`, `escape+retry`, `retry`, `none`), `nav2_recoveries` = Nav2 BT recoveries.

### 2.3 `Nav2GoToMotion` (body/nav2_backend.py)

1. **start**: bridge `goto` = **goal check**, the body A*'s rule replicated in numpy (`NavGrid.plan`, robot radius
   0.25, 0.10 m grid): a goal closer than 0.25 m to a blocked or unknown cell, or outside the map (clamped into it
   first), moves to the nearest free 0.10 m cell within 0.5 m, else `goal_in_obstacle`. So both backends give the same
   reason (a goal inside the bed: `goal_in_obstacle` on both, where Smac's own 0.25 m tolerance used to make it
   NO_VALID_PATH = `no_path`; 400 random goals over the house: same reason and same snapped goal). A goal between
   0.25 m and Nav2's inscribed radius 0.30 m is left to Smac's tolerance (goal snapped to the path end). Then a
   `ComputePathToPose` pre-check (`no_path`, `start_in_obstacle` with the plan: length, ≤ 60 points, plan time), then
   NavigateToPose. One go_to request waits at most ~50 ms in total (the ping, a pre-empted Nav2 goal's cancel reply
   ≤ 20 ms, then `start()` waits for the goto reply with what is left; the reply took 9-30 ms), so these failures
   normally come back synchronously in the go_to reply; a later reply is taken by `tick()` and a failure then ends the op after
   `accepted` (same reason).
   Without a goal yaw the bridge uses the path's arrival heading as the NavigateToPose orientation (no final spin);
   a snapped goal is reported in `goal` (`goal_requested` keeps the request). A `speed` arg becomes a `/speed_limit`.
2. **navigate**: `/cmd_vel` → `VelocityCommander` every control tick; status poll 4 Hz, asynchronous (sent on one tick,
   read on a later one; a poll without reply after 0.5 s counts as failed, 3 s of failed polls ⇒ `nav2_unavailable`),
   so a slow or dead bridge never blocks the 50 Hz loop; a silent Nav2 (replanning, recovery wait) trips the 0.3 s
   watchdog ⇒ IDLE.
3. Nav2 **succeeded** ⇒ IDLE settle holding the goal yaw ⇒ GT check against `final_pos_tol` (0.25) /
   `final_yaw_tol_deg` (12); one re-approach (goal re-sent) if outside, else `final_error`.
4. **escape** (the humanoid replacement for Nav2's BackUp): if the robot is inside the inscribed zone (clearance <
   0.30 m) Nav2 cannot plan (`START_OCCUPIED`). The bridge finds the shortest straight step (16 directions, ≤ 0.6 m)
   to clearance ≥ 0.37 m on the map; the body walks it at 0.2 m/s with the facing held, settles and re-sends the goal
   (≤ 2 per go_to). Also used when Nav2 fails mid-route and the robot ended inside the inscribed zone.
5. **stuck** (FollowPath `FAILED_TO_MAKE_PROGRESS` / `PATIENCE_EXCEEDED` / `NO_VALID_CONTROL`): the goal is re-sent
   once (`max_nav_retries`, default 1). When the clearance where Nav2 gave up is below the inflation radius (0.45 m),
   an escape step comes first (target clearance min(0.45, clearance + 0.08) or, if no step within 0.6 m reaches it,
   the step with the best gain ≥ 0.03 m). The failed goal's `status` carries this hint (`escape`), so no extra call.
6. cancel / pre-emption / body timeout ⇒ bridge `cancel` (the bridge stops forwarding `/cmd_vel` for that goal as soon
   as it reads the request; the body waits ≤ 20 ms for the reply, only to report it) and the service's IDLE hold.

Failure reasons (Nav2 Jazzy error codes → body reasons, `nav2/ros_bridge.py REASONS`):

| Nav2 | body reason |
|---|---|
| bridge goal check: no free cell within 0.5 m of the goal, or outside the map (`GOAL_BLOCKED`) | `goal_in_obstacle` (same as A*) |
| ComputePathToPose `NO_VALID_PATH` 208, `TIMEOUT` 207, `GOAL_OUTSIDE_MAP` 204 | `no_path` |
| `GOAL_OCCUPIED` 206 (cannot happen after the goal check) | `goal_in_obstacle` |
| `START_OCCUPIED` 205, `START_OUTSIDE_MAP` 203 | `start_in_obstacle` (after the escape attempts) |
| FollowPath `FAILED_TO_MAKE_PROGRESS` 105, `PATIENCE_EXCEEDED` 104, `NO_VALID_CONTROL` 106 | `stuck` |
| FollowPath `CONTROLLER_TIMED_OUT` 107; body `timeout_s` | `timeout` |
| TF errors 102 / 202 | `tf_error` |
| NavigateToPose canceled by someone else | `nav2_canceled` |
| NavigateToPose refused by bt_navigator | `nav2_rejected` |
| FollowPath `FOLLOW_UNKNOWN` 100 / no error code | `nav2_failed` / `nav2_aborted` |
| `INVALID_CONTROLLER` 101, `INVALID_PLANNER` 201 | `nav2_config` |
| bridge not connected at start, no goto reply within 8 s, or no status reply for 3 s mid-goal | `nav2_unavailable` |

## 3. Nav2 configuration (`nav2/params/nav2_g1.yaml`)

| Item | Setting | Why |
|---|---|---|
| Nodes | controller, planner, behavior (wait only), bt_navigator, velocity_smoother, lifecycle manager in **one** `component_container_isolated` | one DDS participant; 7.5% vs 12.1% of a core for separate processes (§5) |
| Not run | map_server, AMCL, smoother/waypoint/route/docking servers, collision_monitor | map + GT from the bridge; no sensors in M1 |
| Controller | **Regulated Pure Pursuit** (default), 20 Hz: 0.5 m/s desired, min approach / regulated min speed 0.2, rotate to heading > 45° at 0.6 rad/s, no reversing, **`use_collision_detection: false`** | §3.1, §3.2 |
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

### 3.2 RPP collision detection off while the costmaps are static (verifier finding, 2026-09-29)

With `use_collision_detection: true` RPP re-tests the arc to the carrot against the local costmap. The costmaps are
static-only (no sensors), so that only re-checks the house map the Smac plan already avoids, and in the ~0.8 m
passage of procthor-train-38 near (5.7, 9.7) (best clearance 0.40 m, robot radius 0.30 m) the robot's heading drift
made the arc touch the inscribed band: "detected collision ahead!" → `PATIENCE_EXCEEDED` → `stuck` at (6.00, 9.61)
with 0.36 m clearance, and every later go_to from there failed the same way (verifier: 8/16 goals; it depended on
`heading_bias_ki`, 0.4 → 8/8, 0.0 → lockups). Now: detection off (TODO in the yaml: back on together with the
obstacle/voxel layer and collision_monitor), and `max_nav_retries` 1 with an escape step first when the clearance is
below the inflation radius (§2.3). With detection off nothing re-checks the arc against the map: in the fix E2E the
GT clearance stayed ≥ 0.30 m on every regular goal (min 0.304-0.364 m, as before the change), and dipped to 0.25 m
for 0.8 s only in the lockup-pose trial, which starts 0.36 m from the wall pointing 15° towards it (fake collision
radius 0.15 m; the real G1's arms need ~0.30 m).

## 4. Tests on the fakes

`bash nav2/tools/run_fake_e2e.sh [--port-offset 700] [--controller rpp|mppi] [--goal-plan std|passage] [--ki LIST]`
(tmux `nav2-fakes-<offset>` + `nav2-e2e-<offset>`, all contract ports + offset, ROS domain 42 + offset // 100,
preflight refuses busy ports; `WL_RUN_PREFIX="nice -n 19 taskset -c 12-15" NAV2_NICE=19 NAV2_TASKSET=12-15` pins
everything on a shared box): `nav2/tools/fake_stack.py` (the body agent's `tools/fake_p1.py` with
`--house-dir assets/houses/procthor-train-38` + `tools/fake_deploy.py`, which decodes SonicMux's planner messages
with the planner frame = GT yaw at start, 1 s planner timeout, and integrates them kinematically with a +3° heading
bias), the real wl-body (`nav2/tools/fake_body.py` = `body.service` + a test-only `heading_bias_ki` override) with
`NAV_BACKEND=nav2`, `nav2/up.sh`, then `nav2/tools/nav2_fake_test.py` which uses **only BodyClient** and checks every
result against GT from its own `gt.pose` subscription. The same test runs unchanged against the real stack (without
the fake-only `escape`, `stuck` and the passage plan's `reset_robot`). `--ki "0.4,0.0"` restarts the body for each
value; every value runs the goals, the last one also the other tests (`stuck` last: its obstacle stays).

### 4.1 Fix round after the verifier (2026-09-29), `outputs/m1/nav2/fix-e2e-20260929-033108/`

One run, port offset +900 (ROS domain 51), everything `nice 19` on CPUs 12-15, `--goal-plan passage` (8 goals per
value, 7 of them through the ~0.8 m passage near (5.7, 9.7); goal 6 starts at the verifier's lockup pose
(6.00, 9.61, 165°) via `reset_robot`), `--ki "0.4,0.0"`. RPP ran with `max_angular_accel` 3.0 in this run; it is
back at 1.0 in the committed params (no measurable gain on the fakes: turns in place ≥ 60° averaged 26.4°/s vs
26.1°/s before, the velocity smoother's ramp dominates). Result: **16/16 goals, 14/14 passage traversals, no lockup,
no stuck event, no escape or retry needed**; all other tests pass except `unreachable`, which exposed a goal-check
crash for goals far outside the map (`nav2_rejected` instead of `goal_in_obstacle`); fixed and re-run alone:
`fix-unreachable-20260929-034256`, pass.

| Test | Result |
|---|---|
| goals, `heading_bias_ki` 0.4 | 8/8 succeeded, GT error 0.085-0.152 m, yaw error ≤ 1.6°, 17.6-44.6 s, walked 0.99-1.04× the plan, 0 re-sends |
| goals, `heading_bias_ki` 0.0 (the verifier's failing case) | 8/8 succeeded, GT error 0.082-0.109 m, yaw error ≤ 4.4°, 18.9-44.9 s, 0 re-sends |
| passage clearance (GT, centre to nearest obstacle cell) | ≥ 0.304 m on all 14 regular goals (before the fix: 0.316-0.381 m); 0.25 m for 0.8 s in the lockup-pose trial (starts 15° off towards the wall; fake collision radius 0.15 m) |
| turning in place (stepped facing) | IDLE facing steps ≤ 13.9° (≈ RPP's 0.6 rad/s × 0.4 s), 104-106 per 8 goals; the first 4 goals (the same trips as `fake-rpp-final`) took 41 IDLE facing changes where the old continuous ramp sent 192 (each one a SONIC re-plan) |
| goto reply (goal check + plan + accept) | 8.9-30.4 ms (median 16.7 ms): always inside `start()`'s wait (45 ms in this run, now what is left of the 50 ms request budget), so failures stay synchronous |
| unreachable (after the fix) | bed centre: Nav2 `goal_in_obstacle` (GOAL_BLOCKED) in 12 ms = A* `goal_in_obstacle`; goal outside the map: both `goal_in_obstacle` (3 ms); goal 0.2 m from a wall: both snap to (6.075, 1.375); offline, 400 random goals over the house give the same reason and the same snapped goal on both |
| cancel mid-route | after 1.02 m at 0.44 m/s: go_to `canceled`, Nav2 goal `canceled`, 0 `/cmd_vel` forwarded after the cancel, GT speed < 0.05 m/s 0.74 s later |
| timeout | `timeout_s` 4 ⇒ `failed timeout` at 4.02 s, Nav2 goal canceled |
| escape (fake `reset_robot` into clearance 0.20 m) | START_OCCUPIED ⇒ 0.20 m step at 90° (moved 0.29 m) ⇒ goal re-sent ⇒ succeeded, GT error 0.105 m |
| velocity watchdog | 70 messages (wz 0.5 for 1.5 s, then vx 0.4 for 2 s): planner IDLE 0.323 s after the last message, op `succeeded ended_by watchdog`, turned 42.8° (∫wz = 43°), moved 0.83 m |
| stuck (fake `spawn_obstacle` on the path) | FAILED_TO_MAKE_PROGRESS (clearance 0.53 m ⇒ no escape step) ⇒ one retry ⇒ FAILED_TO_MAKE_PROGRESS ⇒ `failed stuck` after 139.7 s, 12 Nav2 recoveries, 2 `stuck_events` |

Unit tests (`body/tests`, 36 pass): `test_velocity.py` adds stepped IDLE turning (≥ 0.4 s apart, ≤ 30° from the
commanded facing, frozen while the body lags > 15°), walking facing gated by catch-up, watchdog holding the commanded
facing, the 50 ms ping and the 3 s "unreachable" cache, and a slow fake bridge (goto after 0.3 s, status after 0.2 s)
during which no `tick()` takes more than 10 ms.

### 4.2 First build (`outputs/m1/nav2/fake-rpp-final/`)

4/4 goals, unreachable (then `no_path` for the bed), cancel, timeout, escape, watchdog (IDLE 0.320 s after the last
message), stuck (76 s, no retry) all passed. The verifier's rerun of the same tree (`verify-rpp-20260929-022627`)
locked up in the passage (§3.2).

Artifacts per run: `trajectory.png` (all runs over the map with the inscribed/inflation bands and Nav2's plans),
`goal_<k>.png` / `cancel.png` (map zoom + `/cmd_vel`, SONIC planner messages and GT speed/yaw-rate traces),
`watchdog.png`, `nav2/cmd_vel.jsonl` (every `/cmd_vel`), `body/planner_cmds.jsonl` (every planner command change as
sent to SONIC), `nav2/goals.jsonl`, `pose.csv`, `events.jsonl`, `cpu.csv`, `cpu_summary.json`, `summary.json`.

## 5. CPU

Per process from `/proc` (1 s samples, 100% = one core), composed RPP config, fix E2E run (everything `nice 19` on
CPUs 12-15). `nav2/tools/cpu_monitor.py` now counts only descendants of the run's own tmux sessions (`--session`):
before, it matched command lines box-wide and the verifier's "body" and "launch" groups included other agents'
processes. Here every group is one process (the body two, one per `heading_bias_ki` value, one after the other).

| Process | standing (idle) p50 | navigating mean / p95 / max | RSS |
|---|---|---|---|
| Nav2 container | 8% | 9.7% / 11% / 11% | 119 MB |
| ros_bridge.py | 3% | 6.3% / 7% / 8% | 104 MB |
| `ros2 launch` | 0% | 0.1% / 1% / 2% | 80 MB |
| **ROS side total** | **~11%** | **~16% of one core** (≈ 1% of the 16-vCPU box) | ~300 MB |
| wl-body (for reference) | 7% | 6.7% / 8% / 9% | 71 MB |

The first build's table said the bridge idles at 2.0%; its own `cpu_summary.json` has 4.07% (mean, incl. the start-up
burst), and the idle median here is 3%. Separate processes (RPP, `--no-composition`, first build), navigating:
controller 2.6%, bt_navigator 2.7%, planner 1.8%, lifecycle manager 1.9%, behaviour server 1.6%, velocity smoother
1.5% = 12.1%: the cost is mostly per-node overhead (DDS, 50 Hz TF into every TF listener), not the algorithms, hence
composition. MPPI (batch 1000 × 40) adds ~4% (11.6% mean, 17% max). Kept low by: BT tick 50 ms instead of 10 ms (also
5× fewer feedback messages into the Python bridge, bridge 10.7% → 4.5%), local costmap 3 m at 5 Hz, costmap
publishing 1 / 0.5 Hz, no sensors layers. `up.sh` runs Nav2 and the bridge at `nice 5` (`NAV2_NICE`) and accepts
`NAV2_TASKSET` to keep them off the wall-clock deploy's cores.

## 6. Launch

```bash
bash nav2/install_ros.sh                       # once: Jazzy ros-base + navigation2 + nav2-bringup (niced)
WL_PORT_OFFSET=N source nav2/ros_env.sh        # any manual ros2 command (domain 42 + N // 100)
bash nav2/up.sh [--port-offset N] [--session wl-nav2] [--controller rpp|mppi] [--no-composition] [--log-dir D]
python3 nav2/tools/bridge_cli.py [--port-offset N] ping | stats | status ID | plan X Y | goal_check X Y | escape [B] | wait
bash nav2/down.sh [--session wl-nav2]
```

`up.sh` computes the ROS domain (42 + offset // 100, `WL_ROS_DOMAIN_ID` overrides), refuses to start when a
`wl_ros_bridge` already runs on it (`ros2 node list --no-daemon`, ~3 s), and exports it to the bridge and Nav2.
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
  goals,unreachable,cancel,timeout,watchdog --label "Isaac + SONIC"` (and `--goal-plan passage` without the
  fake-only reset goal), then tune on SONIC: the IDLE step period (0.4 s, `vel_idle_step_period_s`) and the stepped
  facing against SONIC's IDLE-turn shortfall (the velocity op has no FacingServo-style push; RPP closes the loop on
  the yaw instead), RPP `rotate_to_heading_angular_vel` (0.6) against SONIC's turn rate, `desired_linear_vel` (0.5),
  the velocity smoother accelerations. `body/config.py facing_lead_max_deg` is no longer used (the clamp to the
  measured yaw is gone); its owner can drop it.
- **Stuck detection is slow**: ~70 s per Nav2 attempt (progress checker 10 s × FollowPath retry × 3 outer attempts),
  ~140 s with the body's one retry. Fine with a static map; lower `movement_time_allowance` or the BT retries once
  sensors make recoveries meaningful.
- **No collision re-check while following** (RPP `use_collision_detection: false`, §3.2): the robot follows the
  collision-free Smac plan, but pure pursuit can cut inside it; worst case in the fix E2E 0.25 m GT clearance
  (lockup-pose trial). Real SONIC drifts more than the fake (−3.8° / 0.27 m over 2.5 m open loop, m1.md §1.9), so
  watch the passage in the first real run.
- **Sensors:** TODO in `params/nav2_g1.yaml` (obstacle/voxel layer from the head D435 depth or a MID-360 LiDAR, plus
  collision_monitor, then RPP collision detection back on). Until then Nav2 only knows the static house map; spawned
  or moved objects are invisible to it.
- Port offset hygiene: other agents run full stacks at +300 and +400 on this box. The e2e runner now refuses to start
  unless every port of its offset is free (an early run at +400 connected the bridge to another agent's real P1: it
  only read `gt.pose` and got a refused `get_occupancy`; no command was sent).
- `docs/architecture.html` (Navigation section, "PathFollower A*") predates this; its owner should switch it to Nav2.
- **Learned navigation checkpoint (owner's question).** NVIDIA **COMPASS** (`nvidia/COMPASS` on Hugging Face,
  NVlabs/COMPASS, NVIDIA Open Model License) is a cross-embodiment mobility policy on the X-Mobility backbone whose
  model card lists `g1` among the supported embodiments; inputs are a camera image, robot state and a route,
  outputs are desired linear/angular velocities, runtime TensorRT, with a ROS 2 deployment folder. That output is
  exactly the `velocity` op / `/cmd_vel` interface built here, so it could replace the RPP controller (with Smac still
  providing the route) without touching wl-body or SONIC. Not downloaded or evaluated yet: its G1 locomotion layer
  differs from SONIC and its camera must match ours (the model card does not state an Isaac Sim version).
  `arena_spike/`'s `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` (GR00T N1.6) outputs a `navigate_command`
  (forward, sideways, turn speeds) for its own leg controllers, not SONIC; it would plug into the same `velocity` op.
