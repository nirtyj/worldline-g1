# nav2/: ROS 2 Nav2 as the go_to backend of wl-body

Full design, parameters, test results and CPU numbers: [`docs/nav2.md`](../docs/nav2.md).

```
BodyClient.go_to ──► wl-body (P3, .venv py3.11) ──DEALER :5620──► ros_bridge.py (py3.12 + ROS 2 Jazzy) ──► Nav2
                        ▲   Nav2GoToMotion (async, never blocks     goal check (snap ≤ 0.5 m | goal_in_obstacle),
                        │   the 50 Hz loop)                         ComputePathToPose pre-check, NavigateToPose
                        │                                           /map (P1 occupancy), /odom + TF (P1 gt.pose)
                        └── op velocity {vx,vy,wz,goal_id} ◄──────  /cmd_vel (RPP → velocity smoother)
   SonicMux ──► :5556 SONIC planner (SLOW_WALK / IDLE + facing in ≤ 30° steps)
```

ROS domain: `42 + port_offset // 100` (offset 0 → 42, +700 → 49, +900 → 51; `WL_ROS_DOMAIN_ID` overrides).
`up.sh` refuses to start when a `wl_ros_bridge` already runs on that domain, and a bridge that sees a second one
reports not-ready (go_to then falls back to A*).

| File | What |
|---|---|
| `install_ros.sh` | ROS 2 Jazzy ros-base + navigation2 + nav2-bringup (+ MPPI, RPP, Smac, NavFn, tf2 tools), niced, idempotent |
| `ros_env.sh` | `source` before any ROS command: `ROS_DOMAIN_ID=42 + WL_PORT_OFFSET/100` (or `WL_ROS_DOMAIN_ID`), `rmw_fastrtps_cpp`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` |
| `ros_bridge.py` | the one ROS process: gt.pose → `/odom` + TF, P1 occupancy → `/map`, `/cmd_vel` → body, REP 5620 (echoes `rid`), A*-compatible goal check, escape hints |
| `launch/wl_nav2.launch.py` | controller, planner, behaviour server, BT navigator, velocity smoother, lifecycle manager in one container |
| `params/nav2_g1.yaml` | parameters for the G1 + SONIC (RPP controller without collision detection while the costmaps are static, Smac 2D, static costmaps r 0.30 / inflation 0.45) |
| `params/controller_mppi.yaml` | overlay: MPPI Omni instead of RPP (`--controller mppi`) |
| `bt/navigate_humanoid.xml` | default NavigateToPose tree without Spin / BackUp |
| `up.sh` / `down.sh` | start / stop bridge + Nav2 in tmux (`wl-nav2`), per-offset ROS domain, refuse a domain that already has a bridge, wait until ready |
| `m1_hook.sh` | two-line hook for `scripts/m1_up.sh` / `m1_down.sh` (`NAV_BACKEND=nav2` default, `astar` skips Nav2) |
| `wl_map.py` | occupancy npz → OccupancyGrid values / map_server pgm+yaml |
| `tools/` | fake stack, fake body (`fake_body.py`: test-only `heading_bias_ki` override), end-to-end test, CPU monitor (counts only processes of the given tmux sessions), bridge CLI |

```bash
# once
bash nav2/install_ros.sh
# with P1 (and ideally the body) up, offset as for the rest of the stack
bash nav2/up.sh --port-offset 0            # tmux wl-nav2; prints the run dir once Nav2 is ready
python3 nav2/tools/bridge_cli.py ping      # nav2_ready, map, pose rate, ros_domain_id
python3 nav2/tools/bridge_cli.py goal_check X Y   # what the bridge would do with a goal (snap / goal_in_obstacle)
bash nav2/down.sh
# end-to-end on the fakes (no Isaac, no SONIC), every contract port + 700 (ROS domain 49); on a shared box pin it:
WL_RUN_PREFIX="nice -n 19 taskset -c 12-15" NAV2_NICE=19 NAV2_TASKSET=12-15 \
  bash nav2/tools/run_fake_e2e.sh [--port-offset 900] [--controller rpp|mppi] [--goal-plan std|passage] [--ki "0.4,0.0"]
```

The body selects the backend per go_to: env `NAV_BACKEND=nav2|astar` (default nav2), `args.backend` per call; if the
bridge does not answer within 50 ms (remembered for 3 s) or Nav2 is not active, go_to falls back to A* and says so
(`result.backend`, `result.backend_fallback`). `BodyClient` is unchanged.
