# Live-stack bring-up (M2b): Worldline on the SONIC-walking G1 in Isaac

Owner: ops (`scripts/m2_*.sh`, `scripts/p5_probe.py`, this file). Status: **rehearsed on the dev box `ludo-g1-arena`,
2026-09-29 (M2b wave 1)**; §5 has what ran and what it showed. §6 is the main-box sequence for wave 2, to run
verbatim on `ludo-g1-brev2`. This is docs/M2.md §7.4.

---

## 0. At a glance

| Step | Where | Command |
|---|---|---|
| Push the code | laptop | `WL_ROOT=<checkout> BREV_NAME=<box> 00_infra/sync_wl.sh push` |
| Runtime venv (once, and after a dependency change) | box | `bash scripts/m2_venv.sh` |
| Keys (once) | laptop | `BREV_NAME=<box> 00_infra/secrets.sh` |
| Up, cold, to a ready page | box | `bash scripts/m2_up.sh [--profile sonic\|full] [--scene procthor-train-40] [--viz min]` |
| Page in the browser | laptop | `BREV_NAME=<box> 00_infra/tunnel.sh 8765`, then http://127.0.0.1:8765 |
| Smoke test (F1 referee + recording + evidence) | laptop | `BREV_NAME=<box> scripts/m2_smoke.sh --label <name> [--session wl-m2]` |
| Swap planner / System 1, stack keeps standing | box | `bash scripts/m2_p5.sh restart --planner ... --system1 ...` |
| Down | box | `bash scripts/m2_down.sh` (P5 first, body holds, then the M1 stack) |

`00_infra/` is `ludo_robotics_prep_g1/00_infra/`. Box-side commands run in `/work/worldline-g1` after
`source /etc/profile.d/ludo.sh` (the scripts source it themselves).

---

## 1. Processes, ports, start order

| | Process | Env | Ports (+ offset) | Started by | Ready when |
|---|---|---|---|---|---|
| P1 | `sim_isaac.app` (Isaac Sim 5.1, the house, the G1, DDS bridge) | `/work/envs/isaaclab` | REP 5600, PUB 5601 `gt.pose`, PUB 5602 viz frames, PUB 5565 head camera | `m1_up.sh` | ping + `WL_ISAAC_READY`, gt.pose ≥ 40 Hz, lowstate, a camera frame; band on |
| P3 | `body.service` (BodyClient API, SonicMux, A* + pure pursuit) | `.venv` | ROUTER 5610, PUB 5611, PUB 5556 (SONIC input) | `m1_up.sh` | `body.state` on 5611 |
| P2 | `g1_deploy_onnx_ref` (unmodified SONIC deploy) | C++ | SUB 5556, PUB 5557; DDS domain 0 on `lo` (hard-coded) | `m1_up.sh` via `sonic/run_deploy.sh` | `Init Done` + `robot_config` |
| stand | body `stand` | | | `m1_up.sh` | SONIC holds the robot, band released, upright 3 s |
| P5 | `ui.server` (page, websocket, the Worldline runtime, System 1, planner) | `.venv-rt` | HTTP/WS 8765 | `m2_p5.sh` (from `m2_up.sh`) | HTTP 200 on `/` and an `init` for the scene/profile with no error on `/ws` (`scripts/p5_probe.py`) |
| P4 | GR00T PolicyServer (`full` only) | Isaac-GR00T venv, on the **dev box** (OD3) | 5550, reached from the main box through `scripts/groot_link.sh` | owner groot_srv | `groot_link.sh check` pings it |
| rec | `viz/recorder.py --control` | `viz/.venv` | control REP 5630 | `m2_smoke.sh` | `ctl status` |

- All ports bind 127.0.0.1. Only the page port (8765 + offset) is tunnelled to the laptop; nothing is exposed.
- `--port-offset N` shifts every ZMQ port and the page port together. The deploy's DDS domain does not move, so
  **one stack per box** (`run_deploy.sh` refuses a second deploy).
- `viz/box.sh server` also defaults to 8765. With P5 up, run the viz page on offset 1 (`bash viz/box.sh server 1`,
  port 8766, its own tunnel) if you want it as well.
- CPU pinning (from `m1_up.sh`; M1's rule, `docs/walk_diagnosis.md`): the deploy on 0-3, Isaac, body and P5 on 4-15
  (`P5_TASKSET`, default 4-15). A quiet box is still required: no other Isaac/GPU job while SONIC runs.
- go_to is A* + pure pursuit (`NAV_BACKEND=astar`, forced by `m2_up.sh`): Nav2 is deferred (PLAN §0.8).

`m2_up.sh` runs `m1_up.sh` unchanged for P1, P3, P2 and the stand, then starts P5 as the `p5` window of the same tmux
session, then (for `--profile full`) runs `scripts/groot_link.sh check`. A failed GR00T check is a warning, not a
stop: `full` then rejects GR00T calls as `policy unavailable` and keeps its labelled fallbacks. The page runs
`SimClock(1.0)` (ui.server's default `--speed`), so every runtime timeout is in wall seconds.

Every run writes `outputs/m2/stack-<ts>/` on the box (`stack-latest-<session>` links to it): `config.env`,
`m2_up.log` (m1_up's lines stamped with epoch times), `stages.tsv`, **`stages.json`** (seconds per stage and the
total), `p5_ready.json`, `p5.env`, and `m1` (a link to the M1 run dir with P1's stats and the body log dir). Logs go to
`/work/logs/wl/<session>-<ts>-{isaac,deploy,body,p5}.log`.

---

## 2. One-time box setup

1. **Code.** `WL_ROOT=<checkout> BREV_NAME=<box> 00_infra/sync_wl.sh push`. It excludes `.git`, `outputs`,
   `__pycache__`, `.venv*` and `.pytest_cache`, and has no `--delete`, so box-side venvs, `outputs/` and `runs/` survive
   every push. The box runs whatever was pushed last, including uncommitted work: push right before a live run.
2. **Runtime venv.** `bash scripts/m2_venv.sh` builds `/work/worldline-g1/.venv-rt` (Python 3.11, uv) from
   `pyproject.toml` plus its `test` extra, then imports the 18 modules P5 loads (numpy, scipy, pillow, pyzmq, msgpack,
   websockets, pyyaml, google-genai 2.25.0, typesafe_sdk 0.7.2, pytest, `ui.server`, `robot.factory`,
   `world.isaac_client`, `world.frames`, `viz.tap`, `brains.system1_jev`, `brains.scripted`, `agent.model`). It is
   idempotent. A venv that does not run on the box (the main box still has the laptop's macOS copy from an old push,
   `docs/devbox.md` §5.2) is detected and rebuilt. `--check` only runs the import check.
3. **Keys.** In a login shell on the box, `GEMINI_API_KEY`, `TYPESAFE_API_KEY` and `HF_TOKEN` must be set (check
   without printing: `bash -lc 'for k in GEMINI_API_KEY TYPESAFE_API_KEY HF_TOKEN; do test -n "${!k:-}" && echo "$k set" || echo "$k MISSING"; done'`).
   If one is missing: `BREV_NAME=<box> 00_infra/secrets.sh` (it copies `~/.config/ludo-g1/secrets.env` without the
   `BREV_`/`AWS_`/`LAMBDA_` keys and prints only names). `ui.server` reads that file itself; `m2_p5.sh start` refuses
   to start the live planner or Jev without the keys. `ludo-runtime/.env` is never read. Never run these scripts
   under `bash -x`: `/etc/profile.d/ludo.sh` sources the keys (`docs/devbox.md` §6.1).
4. Already there on both boxes (`docs/devbox.md` §2): the Isaac env, the SONIC deploy build and TensorRT engines,
   the body `.venv`, `viz/.venv`, the MolmoSpaces houses and `assets/houses/<id>/`.

---

## 3. Up, down, swap

```bash
# box, /work/worldline-g1
bash scripts/m2_up.sh                                           # sonic, procthor-train-40, session wl-m2, offset 0
bash scripts/m2_up.sh --profile full --viz min                  # + the GR00T link check; VizCams chase camera for recordings
bash scripts/m2_up.sh --planner brains.scripted:create --system1 tests.kept.system1_stub:create   # offline brains (6.1)
bash scripts/m2_up.sh --dry-run ...                             # print the resolved house, ports and commands only
bash scripts/m2_p5.sh status                                    # one JSON line: ok, scene, profile, System 1, stepping stones
bash scripts/m2_p5.sh restart --planner agent.model:create_brain --system1 brains.system1_jev:create   # live brains (6.2)
bash scripts/m2_down.sh --p5-only                               # stop P5 only; the robot keeps standing under SONIC
bash scripts/m2_down.sh                                         # everything
tmux attach -t wl-m2                                            # windows: isaac, body, deploy, monitor, p5
```

- `--scene` takes a scene key or `<house>@<variant>`; the house P1 loads is resolved by the runtime's own
  `robot.factory.parse_scene`, so P1 and P5 always load the same house.
- `--viz off|min|low|high` sets P1's VizCams level (`docs/viz.md` §7.2: `min` for recordings and demos, never
  `high` with SONIC in the loop; `off` for RTF measurements).
- **Idempotent.** `m2_up.sh` on a session whose body answers and whose deploy is in CONTROL skips stages 1-4; a P5
  that already answers is left alone. A half-up session (it exists but the body or deploy does not answer) is an
  error: run `m2_down.sh` first. `m2_down.sh` on a session that does not exist does nothing.
- **Down order.** `m2_down.sh` stops P5 with SIGINT (the page server stops its session: the runtime saves memory, the
  robot façade cancels its body ops and closes its client), then sends body `stop` (planner IDLE: the robot stands
  under SONIC, the M1 equivalent of HOLD until B.3 adds named modes; never `command{stop}`), then runs `m1_down.sh`
  (band on, deploy shutdown, body, P1, kill only that session).
- The eval's reset does not move the robot on Isaac (no `reset_scene` yet, P1.7). For repeatable episodes restart
  the stack between runs (`m2_down.sh` then `m2_up.sh`, about 70 s), as the rehearsal did.

---

## 4. Smoke tests (docs/M2.md §7.4 step 6)

`scripts/m2_smoke.sh` runs on the laptop against a stack that is already up:

1. reads the page's port from the box's P5 state and probes it;
2. opens the tunnel (`00_infra/tunnel.sh <port>`; a keepalive ssh holds the ControlMaster, and so the forward, open
   for the whole episode);
3. starts the viz recorder on the box (`viz/recorder.py --control`, tmux `<session>-rec`);
4. runs the referee on the laptop: `.venv-rt/bin/python -m eval.offline_episode --url ws://127.0.0.1:<port>/ws
   --profile <p> --timeout 600`. It sends the scenario's reset (H40, `forget`), says "Bring me the alarm clock.",
   scores on world truth, and checks the F1 shape step by step, the result envelope, System 1 and the honesty
   labels;
5. stops the recorder and pulls the recording, the m2 and m1 run dirs and the P1/P2/P3/P5 logs to
   `outputs/m2b_wave1/ops/<label>/` (`--out` moves it). Exit status = the episode's.

| Test | P5 started with | Command (laptop) |
|---|---|---|
| 6.1 offline brains | `--planner brains.scripted:create --system1 tests.kept.system1_stub:create` | `BREV_NAME=<box> scripts/m2_smoke.sh --label smoke61-rN --session <s>` |
| 6.2 live brains | the defaults: `agent.model:create_brain` (Gemini) and `brains.system1_jev:create` (Jev + Gemini Live) | `BREV_NAME=<box> scripts/m2_smoke.sh --label smoke62-rN --session <s>` |

---

## 5. Dev-box rehearsal (wave 1, 2026-09-29)

**Setup.** Dev box `ludo-g1-arena`, quiet: the idle gate was clear and the Arena container used 0 % CPU. The stack
lock (`/work/locks/stack.d`, owner `ops`) was held from 06:09:03 to 06:42:48 UTC. tmux session `ops-m2`, port
offset 0, page 8765, `--viz min`, scene `procthor-train-40`. Evidence (laptop):
`outputs/m2b_wave1/ops/<run>/` = `episode.{jsonl,summary.json,txt}`, `recording/` (head, chase, top and composite
mp4, `contact_sheet.png`, `summary.json`), `m2_run/` (`stages.json`), `m1_run/`, `logs/` (P1, P2, P3, P5).

**Code on the box.**

| Runs | Tree | What it was |
|---|---|---|
| cold start 1, smoke61-r1 | `/work/worldline-g1` | the shared tree as pushed at 06:08: commit 7597ec8 plus other owners' uncommitted ui/eval/viz work |
| cold start at 06:18 (refused) | `/work/worldline-g1` | the shared tree at 06:18, now including the isaac owner's uncommitted P1 work (OD1 head + `ego_view` cameras, 37 dynamic props). P1 ran at **RTF 0.61** (physics 122 Hz), so gt.pose came at 30.9 Hz (< 40) and `m1_up.sh` refused to go on, as designed. `m2_down.sh` cleaned up the half-up session |
| cold starts 2-3, smoke62-r1, smoke61-r2 | `/work/ops-wl` | an isolated copy, so the isaac owner's box tree was not touched: commit 2bbfdf0 with `sim_isaac/` at that commit (the M1 P1), overlaid at 06:25 with the laptop working tree for everything else (other owners' uncommitted runtime, robot, ui and eval work). `.venv`, `.venv-rt`, `viz/.venv` and `assets` link to `/work/worldline-g1`. The scripts take `WL=/work/ops-wl`, and `m2_smoke.sh` takes `BOX_WL=/work/ops-wl` |

After the rehearsal the isaac owner committed P1's M2b wire (61c9c8a: OD1 head on 5565 + `ego_view` on 5566, live objects, attach/detach, sim.health, ...), and groot_rt/ui+eval committed their work (HEAD 1c648ac). **ops has not run that P1 or HEAD on a live stack**; its RTF with the new cameras is the first thing to check in wave 2 (`m1_up.sh` refuses gt.pose < 40 Hz, and the 06:18 in-progress version ran at RTF 0.61).

**Setup steps.** Runtime venv from nothing: `m2_venv.sh` took 2.7 s wall (warm uv cache) and ended "18 modules
import". The full suite in that venv on the box: 483 passed, 2 skipped, 1 deselected, in 72 s. Keys: `GEMINI_API_KEY`,
`TYPESAFE_API_KEY`, `HF_TOKEN` and `ANTHROPIC_API_KEY` were all set in a login shell, so `secrets.sh` was not needed.

**Cold start to a ready page (exit a: under 10 min). Pass, 3 of 3.** Seconds, from `stages.json`:

| Start | P1 | body | deploy (Init Done) | stand | P5 | **total** |
|---|---|---|---|---|---|---|
| 1 (06:09, scripted brains) | 17.7 | 0.7 | 36.9 | 8.1 | 1.0 | **64.4** |
| 2 (06:23, live brains) | 17.0 | 0.7 | 32.3 | 8.2 | 1.9 | **60.0** |
| 3 (06:33, scripted brains) | 16.9 | 0.7 | 32.1 | 8.1 | 1.2 | **59.0** |

The TensorRT engines were already cached from the dev box's M1 verification, so a first-ever start on a box adds the
engine build (109 s on the dev box, `docs/devbox.md` §3). Other lifecycle checks:
- `m2_up.sh` on a stack that was already up: 1.2 s, "stages 1-4 skipped", "P5 already up".
- `m2_down.sh`: 6.6, 6.8 and 7.0 s. No stack process and no stack port was left afterwards.
- `m2_down.sh --p5-only` followed by `m2_p5.sh start`: 3 s, with the robot standing throughout.
- The recorder and the tunnel worked in all three smoke runs: head 29.8-29.9 fps, chase 9.95-9.97 fps, 411-412 s
  recorded per run.

**Smoke tests (exits b and c).** Every run was one `fetch_other_room` episode ("Bring me the alarm clock.", H40),
each from a fresh stack (cold starts 1, 2 and 3). **None passes F1.** The expected stop is `manipulate(pick)`: the
`sonic` profile has no working manipulation executor yet (R.1 `sonic_arm_script` needs B.7; `kinematic_attach` needs
P1.3). Before that, two runs stopped earlier on a runtime bug (§7, item 1). Because the first 6.1 run did not pass,
6.1 ran twice, not three times, and 6.2 ran once.

| F1 step (eval/offline_episode.py) | 6.1 r1 (scripted, stub) | 6.1 r2 (scripted, stub) | 6.2 r1 (Gemini planner, Jev + Gemini Live) |
|---|---|---|---|
| System 1 labelled the request | ok (1.3 s) | ok | ok (1.5 s, Jev) |
| navigate to a stand, `sonic_walk` (target) | ok: 8.4 m in 25 s | ok: 8.4 m in 23 s | ok: 5.8 m in 20 s, then 0.8 m in 8 s |
| arrival scan | ok at the first stand, then 2 timed out (12 s budget) | **no** scan succeeded (timed out after 15.0 s). The referee's "ok" matched a glance fallback row (§7, item 3) | **no**: 4 of 4 scans timed out (12.7-14.3 s) |
| check_reachability sees the alarm clock | - | - | ok, after `needs_reposition` (0.7 m) → `navigate(reach_stance)` 0.24 m, `sonic_walk` |
| check_reachability says reachable | - | - | ok: right arm, 0.5 m |
| pick | - | - | **rejected (CAPABILITY)**: "policy unavailable: P1 has no attach/detach op (M2b); needs the BodyServer arm_script op and a GT attach (M3)" |
| verify glance, navigate(user), place, delivered | - | - | - |
| where it stopped | the 2nd scan timed out → halt latched → 11 navigates failed `halted` in 0.2 s → "I couldn't find the alarm_clock." | the same after the 1st scan: 13 navigates failed `halted` in 0.4 s | the planner told the user it cannot pick, then waited |
| walked / falls / P1 RTF (final `rtf_total`) | 15.2 m / 0 / 0.996 | 9.2 m / 0 / 0.995 | 9.4 m / 0 / 0.995 |
| contract checks | the 2 new envelope checks failed on this older runtime; the rest ok | all ok | all ok except "every manipulate result names its executor and skill" (the rejected pick has none; §7, item 3) |
| verdict | FAIL, 17 decisions | FAIL, 17 decisions | FAIL, 12 decisions |

What this shows:
- **Proven live.** The whole bring-up path and its readiness probes. Worldline's page on the `sonic` profile. System 1
  (stub, and Jev + Gemini Live) labelled the request and got gated head frames (10-23 observe calls per run). The
  Gemini planner chose the calls. `navigate` and `reach_stance` ran under SONIC as `sonic_walk`, with 0 falls in about
  34 minutes of stack time and P1 RTF 0.995-0.996. Reachability, the labelled CAPABILITY rejection, the result
  envelope, and the honesty labels ("attach grasp ... STEPPING STONE" in the summary; `sonic_arm_script` listed as a
  shortcut) all worked.
- **Not proven yet.** Any manipulation on `sonic`. A scan that fits its budget. Delivery. GR00T was not run (`full`
  and the link were not part of this rehearsal; `groot_link.sh` is groot_srv's).

**Bring the dev-box stack up again.** The stack was stopped and the lock released at the end of the rehearsal.

```bash
cd /Users/nirty/workspace/ludo-interview/ludo_robotics_prep_g1/00_infra && export BREV_NAME=ludo-g1-arena
./ssh.sh 'for i in $(seq 1 80); do tmux ls 2>/dev/null | grep -qE "^(dev-|verify)" || exit 0; sleep 30; done; exit 1'
./ssh.sh 'mkdir -p /work/locks; for i in $(seq 1 80); do mkdir /work/locks/stack.d 2>/dev/null && { echo "<you> $(date +%s)" > /work/locks/stack.d/owner; exit 0; }; sleep 30; done; exit 1'
WL_ROOT=/Users/nirty/workspace/ludo-interview/worldline-g1-m2b ./sync_wl.sh push
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_up.sh --session <you>-m2 --viz min'   # ~60 s; add --planner/--system1 for 6.1
./tunnel.sh 8765                                                                         # http://127.0.0.1:8765
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_down.sh --session <you>-m2'; ./ssh.sh 'rm -rf /work/locks/stack.d'
```

If the shared tree's P1 is mid-change again (the 06:18 refusal), use a snapshot tree the same way: extract
`git archive <commit>` into `/work/<you>-wl`, link `.venv`, `.venv-rt`, `viz/.venv` and `assets` to
`/work/worldline-g1`, and run the scripts with `WL=/work/<you>-wl` (and `BOX_WL=` for `m2_smoke.sh`).
`/work/ops-wl` from this rehearsal is still on the box.

---

## 6. Wave 2: the main box `ludo-g1-brev2`, verbatim

Preconditions: the G0 arm-tracking work is finished and its owner agrees to the main box being taken over (only one
deploy per box; SONIC needs a quiet box). The GR00T PolicyServer runs on the dev box (OD3, owner groot_srv,
`scripts/groot_server.sh`, `docs/groot_serving.md` §5); the dev box must then run **no** Isaac/SONIC stack, only P4.
The GR00T lines below follow `docs/groot_serving.md` §5; if they disagree, that file wins.

```bash
# ---------- laptop
cd /Users/nirty/workspace/ludo-interview/ludo_robotics_prep_g1/00_infra
export BREV_NAME=ludo-g1-brev2
./ssh.sh 'tmux ls; pgrep -fa "g1_deploy|sim_isaac.app|body.service|ui.server" | grep -v pgrep; nvidia-smi --query-compute-apps=pid,name --format=csv'
#   must show no stack: if wl-m1 (G0) is still up, its owner stops it (bash scripts/m1_down.sh), not us
WL_ROOT=/Users/nirty/workspace/ludo-interview/worldline-g1 ./sync_wl.sh push     # the merged checkout (or a worktree)

# ---------- main box: runtime venv and keys
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_venv.sh'
#   rebuilds the broken macOS .venv-rt copy that is still on the main box (docs/devbox.md §5.2); ends "runtime venv ok"
./ssh.sh 'bash -lc '"'"'for k in GEMINI_API_KEY TYPESAFE_API_KEY HF_TOKEN; do test -n "${!k:-}" && echo "$k set" || echo "$k MISSING"; done'"'"
#   only if one is MISSING:  ./secrets.sh

# ---------- GR00T link (OD3; scripts/groot_link.sh, owner groot_srv; docs/groot_serving.md §5)
BREV_NAME=ludo-g1-arena ./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_server.sh start --warm && bash scripts/groot_server.sh status'
./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_link.sh keygen'                 # prints the public key (once)
./ssh.sh 'curl -s ifconfig.me'                                                      # <main box IP> as the dev box sees it
BREV_NAME=ludo-g1-arena ./ssh.sh "cd /work/worldline-g1 && bash scripts/groot_link.sh install-key '<that public key>' --from <main box IP>"
./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_link.sh up --host <dev box IP> && bash scripts/groot_link.sh check'
#   <dev box IP>: BREV_IP in 00_infra/.state-ludo-g1-arena/instance.env

# ---------- main box: the stack (session wl-m2, offset 0, page 8765; ~60-65 s on the dev box)
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_up.sh --profile full --scene procthor-train-40 --viz min'
#   ends "READY: page http://127.0.0.1:8765"; stage times in outputs/m2/stack-latest-wl-m2/stages.json;
#   the GR00T check result is in config.env (groot_link=ok|failed|skipped)

# ---------- laptop: the user's browser
./tunnel.sh 8765          # then open http://127.0.0.1:8765  (laptop port 8765 must be free: lsof -iTCP:8765 -sTCP:LISTEN)

# ---------- smoke tests (laptop; the stack stays up between them)
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_p5.sh restart --session wl-m2 --planner brains.scripted:create --system1 tests.kept.system1_stub:create'
cd ../../worldline-g1 && BREV_NAME=ludo-g1-brev2 scripts/m2_smoke.sh --label main-smoke61-r1 --session wl-m2 --profile full
#   for a clean start between runs: ./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_down.sh && bash scripts/m2_up.sh --profile full --viz min ...'
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_p5.sh restart --session wl-m2 --planner agent.model:create_brain --system1 brains.system1_jev:create'
BREV_NAME=ludo-g1-brev2 scripts/m2_smoke.sh --label main-smoke62-r1 --session wl-m2 --profile full

# ---------- down (when the lead says so)
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m2_down.sh'
./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_link.sh down'
BREV_NAME=ludo-g1-arena ./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_link.sh remove-key'
BREV_NAME=ludo-g1-arena ./ssh.sh 'cd /work/worldline-g1 && bash scripts/groot_server.sh stop'
```

Ports on the main box (all 127.0.0.1, offset 0): P1 5600/5601/5602/5565 and 5566 (`ego_view`, OD1; rendered only while
enabled), P2 5556/5557, P3 5610/5611 (+5612 halt lane, B.1), P4 5550 (the local end of the GR00T link),
P5 **8765** (the only port the laptop needs), recorder control 5630, DCGM 5555 (taken, never ours). DDS domain 0 on
`lo`.

---

## 7. Known issues and open items (found in the rehearsal; owner in brackets)

1. **A scan timeout latches the halt, and nothing clears it** (runtime + world). This was reproduced in 6.1 r1 and
   6.1 r2.
   - The harness gives a scan `SENSE_TIMEOUT` = 12 s (`agent/harness.py:72`, sized for the ~5 s waist scan). The
     INTERIM `turn_in_place` scan on SONIC takes 12.7-15.1 s: 3 `turn_to` plus the settles plus the turn back.
   - On the timeout, `run_execution` (`agent/skills.py`) cancels the scan. The scan checks for cancel only between
     turns (`services/observation.py` `_scan`), so it outlives the grace, and the harness calls `robot.halt()`.
   - The halt latches. Only `_resume()` (a `resume` directive) clears it. `NavigationService.start` then fails every
     later `navigate` with `halted` in about 0 s (`services/navigation.py`, `if self.gate.latched`).
   - The scripted planner walked through all 11-13 remaining keypoints in under 0.5 s and gave up.
   - 6.2 hit the same scan timeouts but its navigates went on. The cause of the difference was not established.
   - Suggested fixes: take the scan budget from `RobotBridge.timeout_s("observe")` (30 s) or from the scan executor;
     make `_scan` cancel its in-flight `turn_to`; do not leave the latch set after a harness-internal timeout halt
     (or resume at the next planner decision).
   - The scan's timeout summary reads "timeout (the walk took too long)", which is navigate's wording
     (`api/reasons.py:34`).
2. **The in-progress P1 runs at RTF 0.61** (isaac): the uncommitted OD1 cameras and dynamic props in the shared tree,
   06:18 (above). SONIC cannot run below RTF 0.98 (`docs/walk_diagnosis.md`). This needs a measurement with each part
   off before P1 is merged.
3. **Referee** (ui+eval):
   - (a) "arrival scan" matched a row with `action: scan` whose `data.mode` was `glance` (`scan_fallback: glance`,
     6.1 r2 row 89). The predicate should require `data.mode == "scan"`.
   - (b) "every manipulate result names its executor and skill" fails on a CAPABILITY-rejected pick. The check
     should exempt rejections, or the runtime should name the skill it rejected.
4. **The d435 head camera sees only the dresser front at the dresser stand**, while GT perception reports the alarm
   clock visible (6.2 contact sheet) (isaac P1.4, world R.3). This is docs/M2.md gap 3.
5. **System 1's frame gate broke on 2bbfdf0** (ui+eval): `_default_frame_gate` passed the G1_HEAD preset positionally
   as `scene_cells`, so every `decide()` raised (5 tracebacks in the first minute of cold start 2, before the overlay).
   It was fixed in the owner's tree then and is committed now (HEAD 1c648ac, `FrameGate(config=cfg)`). Item 1's
   `SENSE_TIMEOUT` = 12 s and the latch check in `services/navigation.py` are still in HEAD.
6. **The eval's reset does not move the robot on Isaac** (no `reset_scene`, P1.7). For repeatable episodes restart
   the stack (about 60-65 s up, about 7 s down).
7. **Main box, before wave 2** (from `docs/devbox.md` §6.1, not re-checked here): `/work/logs/wl/isaac-cdds.log`
   holds plaintext keys. Delete it and rotate the keys.
8. `00_infra/sync_wl.sh push` does not exclude `runs/` or `.env*` (docs/M2.md §7.4 step 3 says it should). A
   laptop `runs/` or `.env` would be copied over the box's. That is a request to the 00_infra owner.
9. Laptop port 8766 was held by an ssh forward from another session during the rehearsal. The page tunnel on 8765 was
   free, and `m2_smoke.sh` cancels its own forward when it exits.
