# Fault injection for the G1 stack scenarios (E4 hooks)

Owner: hooks (M2b finish). Status: built and tested offline; live results in §6. Everything here is
**TEST-ONLY**: the demo stack never exposes it (P1's ops need `--test-ops`, the delay proxy and the deploy kill are
eval-only commands).

`eval/stack_suite.py` (PLAN §9.3, G1-G14) referees the live page from what the page shows. Five scenarios need a fault
the page cannot cause. Before this work none of them could run live (wave 2, eval-live: "at most about 9/14").

| Scenario | Fault | Injection | Fixture after the scoring window |
|---|---|---|---|
| G4 `stale_chunk_after_correction` | +600 ms on the GR00T link | `tools/hooks/delay_proxy.py` on 5551 → 5550 (P5 on `WL_GROOT_ENDPOINT=tcp://127.0.0.1:5551`) | delay back to 0 ms |
| G5 / G13 `policy_down` | P4 dies | `scripts/groot_server.sh stop` (main box, PLAN §0.11) | `groot_server.sh start --warm` |
| G6 `blocked_path` | a box in the corridor | P1 `spawn_box`, placed across the robot's own A* route (`tools/hooks/obstacle.py`) | P1 `clear_box` |
| G7 `fall_recovery` | 250 N lateral push | P1 `push_robot {force_n: 250, dir: left, duration_s: 0.5}` | `tools.hooks recover` (body path A) |
| G8 `deploy_restart` | (a) P2 crash, (b) kill button | (a) SIGKILL of the deploy; (b) the page's `estop` (the deliberate estop test) | `tools.hooks recover` (operator path B) |
| G9 `rtf_degraded` | RTF 0.9 | P1 `rtf_throttle {target: 0.9}` | `rtf_throttle {off}` |

## 1. The pieces

**P1 test ops** (`sim_isaac/test_ops.py`; registered only with `python -m sim_isaac.app ... --test-ops`, i.e.
`scripts/m2_up.sh --isaac-args "--test-ops"`). Each one logs a `gt.event` `test_op` and ends by itself, so a suite that
dies mid-scenario cannot leave the sim degraded.

| Op | Arguments (defaults) | What it does |
|---|---|---|
| `push_robot` | `force_n` 250 (≤ 1000), `dir` left (left/right/forward/back in the robot's frame, a world `[dx, dy]`, or world degrees), `duration_s` 0.5 (≤ 2, sim time) | A world-frame force on the pelvis, applied every physics step like the band's wrench |
| `rtf_throttle` | `target` 0.9 (0.5-0.999; ≥ 1 or `off` ends it), `duration_s` 120 (≤ 900, wall) | Moves RtPacer's schedule anchor `dt·(1/target − 1)` later per step: the pacer sleeps longer, RTF settles at the target with no overruns, and P1's own `sim.health` reports degraded/unsafe as for a slow sim |
| `spawn_box` / `clear_box` | `pose {x, y, yaw}`, `ttl_s` 600 (≤ 1800); `size` must equal `--test-box-size` (0.3,1.6,1.2 m) | One kinematic box, created at start-up and parked 60 m from the spawn, is moved onto the floor at the pose. It collides with the robot but is in no occupancy map, so the walk gets stuck and navigation ends `blocked` |
| `test_ops_status` | | `{active, push, throttle, box, box_size, counts}` |

Diff in `sim_isaac/app.py`: two argparse flags, the box before `sim.reset()`, the registration, and two calls in the
physics loop (`pre_step` after the band wrench, `before_wait` before the pacer's sleep), each behind
`if self.test_ops is not None`.

**Box-side CLI** (`python -m tools.hooks [--port-offset N] [--session wl-m2] CMD`, `tools/hooks/cli.py`). One JSON line
per command, exit 0 iff ok.

| Command | Notes |
|---|---|
| `push [--force-n --dir --duration-s --watch-s]` | `--watch-s` reports `fell`, `t_fall_s`, `pelvis_z_min` from `get_pose` |
| `throttle --rtf 0.9 [--duration-s]`, `unthrottle` | |
| `spawn-box --toward OBJ \| --toward-xy X,Y \| --x --y --yaw`, `clear-box` | `--toward` plans like the body (A* on P1's occupancy), tries the narrowest points of the route first and picks the first where the box leaves **no** path; else the narrowest spot it spans, with `blocks_all_routes: false` |
| `kill-deploy` | SIGKILL (a crash; never `o` / `command{stop}`) of the g1_deploy_onnx_ref whose command line binds **this stack's** g1_debug port |
| `restart-deploy [--wait-init 180] [--no-stand]` | `sonic/run_deploy.sh start` in the stack's `deploy` window with m1_up.sh's arguments, then the body `stand`. Refuses while a deploy is alive |
| `recover` | deploy dead → `restart-deploy` (path B); fault `fallen` → the body's `recover` (path A), else P1 `reset_robot` + `clear_fault` + `stand`; healthy → "none needed" |
| `delay-proxy start\|set\|stop\|status [--ms]` | the proxy in tmux `hooks-delay`: ROUTER 5551 → DEALER 5550, control REP 5549; `--where reply` (default) holds each reply |
| `p5-endpoint proxy\|direct` | sets/unsets `WL_GROOT_ENDPOINT` in the stack's tmux session environment, then `scripts/m2_p5.sh restart` (the p5 window inherits it) |
| `status`, `clear-all` | `clear-all`: unthrottle, clear-box, delay 0; never touches the deploy |

**Hook maps** (`eval/hooks.py`): `--hooks box` (the suite runs on the box) or `--hooks laptop` (each command through
`00_infra/ssh.sh`). `--hook name=cmd` overrides an entry, `--no-hook name` drops one.

**Scoring changes** (`eval/stack_suite.py`):

- G7 and G8 score the stack's **own** recovery inside `RECOVERY_WINDOW_S` (65 s: path B's 60 s + margin). Ready =
  a `body.ready` event, or what the page shows today: the body back in HOLD and upright after being down (FAULT /
  ESTOP / TRANSITION / OFF / fallen). "No command{stop}" = the body never entered ESTOP and no estop event (the page can
  tell; the deploy exits on command{stop}). After the window, a fixture (`recover_robot` / `restore_deploy`) puts
  the robot back for the next scenario; it is recorded in `extra.fixtures` with its timing and labelled
  "operator fixture", never scored.
- Before every live scenario, `recover_robot` runs as a preflight (a no-op, "none needed", on a healthy stack).
- G9 throttles after the scene reset (a reset and a stand at RTF 0.9 are not the test) and always restores.
- G4 fails at once when the delay cannot be set (the proxy must be in P5's loop).
- `clear_all` runs at the end of every live run; the result JSON carries `hooks` and `hooks_log` (every hook's exit
  code, seconds and JSON reply).

## 2. Running E4 live (main box)

```bash
# the stack with the test ops (P1 must be (re)started with the flag; the page and viz ports as in bringup.md §6.1)
bash scripts/m2_up.sh --profile full --scene procthor-train-40 --viz min --p5-port 8766 --isaac-args "--test-ops"
# G4 only: the proxy in the loop (then back: delay-proxy stop; p5-endpoint direct)
.venv-rt/bin/python -m tools.hooks delay-proxy start && .venv-rt/bin/python -m tools.hooks p5-endpoint proxy
# the suite, on the box, against P5
.venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8766/ws --profile full --hooks box \
    --out outputs/m2b_finish/hooks/e4_full.json --trace-dir outputs/m2b_finish/hooks/e4_traces
.venv-rt/bin/python -m tools.hooks status          # hooks_active must be false afterwards
```

## 3. What the hooks cannot make pass (open, owners in brackets)

- **No recovery supervisor exists** [ops + runtime]. PLAN §7.3.3/§7.7 (`ops/supervisor.py`, path A after a fall,
  path B after a deploy exit, `sim_recovery{path}` and `body.ready` on the page, the recovery limit) is not built:
  the harness pauses on `fell`/`deploy_lost` and nothing recovers. The body's `recover` op (path A) and the
  `restart-deploy` sequence (path B) both exist and are what the fixtures run, so a supervisor only has to call them
  and emit the events. Until then G7's and G8's recovery criteria fail honestly.
- **No walking cap under DEGRADED** [world + runtime]: PLAN §3.5 caps walking below RTF 0.95; nothing publishes a
  cap, so G9's "walking capped" is UNVERIFIED.
- **F7 / G5 conflict** [lead]: PLAN §2.2 F7 (running call `failed(policy_unavailable)` ≤ 3 s, then HOLD) vs PLAN §6.4
  (`groot_then_script` falls back after any GR00T failure). Unchanged here.
- G14's `place_object` hook is not built: G14 relies on the object needing a reposition by itself.

## 4. Offline evidence

`tests/eval/test_stack_suite_hooks.py` (12 tests) and `sim_isaac/tests/test_test_ops.py` (6): the hook maps; the
HookInjector log; the delay proxy on real sockets (+300 ms held, pass-through at 0); the box placement (one door: it
cuts every route; two doors: it says a detour exists); the CLI against the fake P1 with the test ops + the fake
deploy + the **real** body service (a 250 N push collapses the fake robot, the body latches `fallen`, `recover` runs
the body's path A, `recoveries` = 1, throttle/box/clear-all); the G7/G8/G9 scorers on a scripted page with and
without the stack's own recovery; the throttle arithmetic on the real RtPacer.

## 5. Safety

- `kill-deploy` only kills the deploy bound to the stack's own g1_debug port; `restart-deploy` refuses while one is
  alive (one deploy per box). Nothing here sends `command{stop}`; G8(b)'s estop is the page's operator kill button,
  the deliberate estop test.
- Every P1 test op expires (push: its duration; throttle: `duration_s`; box: `ttl_s`), and `clear-all` +
  `status` (`hooks_active: false`) end every live run.

## 6. Live results

See the M2b finish report for this owner (hooks); the artefacts are under `outputs/m2b_finish/hooks/`.
