# M2b wave 1: what it delivered, what ran live, and the wave-2 plan

Status: **wave 1 integrated, 2026-09-29**, branch `m2b-wave1`. Six owners (isaac, world, groot_rt, groot_srv, ops,
ui) built in parallel. The integrator then reconciled their wires, ran the suite and did one live smoke on the dev
box `ludo-g1-arena`. The main box was not touched.

Build order in force (PLAN §0.8): Nav2 and GR00T fine-tuning are deferred. The goal is the whole loop end to end:
Worldline, then navigate, SONIC walking, manipulate, GR00T arms (off-the-shelf N1.7, **experimental**), cancel, halt
and policy-down semantics, and the labelled fallback. GR00T grasp success is not expected zero-shot.

Every number here comes from a run named next to it. Evidence is under `outputs/m2b_wave1/` (not committed).

---

## 1. What each owner delivered

| Owner | Delivered | Live evidence (dev box) | Evidence |
|---|---|---|---|
| **isaac** (done) | P1 M2b wire `docs/contracts/p1_m2b.md` v1 (`p1_contract: m2b-1`): <br>- op discovery;<br>- live object poses and `gt.objects` at 10 Hz;<br>- attach/detach `follow`/`fixed_joint` (STEPPING STONE);<br>- head camera on 5565 and the Arena-exact `ego_view` on 5566, rendered only while enabled (OD1);<br>- link poses;<br>- instance-id `detections`;<br>- `reset_scene`, `move_object` and `push_object`;<br>- the furniture top render;<br>- `sim.health`, `robot_fell` and `object_fell`.<br>`tools/fake_p1.py` serves the same wire. | Attach: 50/50 cycles in each mode, 0 tensor-view errors.<br>RTF with SONIC walking, p10: 0.9971 (head) and 0.9925 (head + ego_view).<br>M1 drive test `all_pass`.<br>`reset_scene` 12.7 ms.<br>Detections agree with gt-geometric on 66 % of views; the bar is 90 %, **not met**. | `outputs/m2b_wave1/isaac/`, p1_m2b.md §13 |
| **world** (partial) | GT confinement and its lint.<br>R.6: sim health becomes a capability.<br>`robot/health.py`: capability_changed, HaltResender.<br>One trace result-row shape (`api.results.RESULT_ROW_FIELDS`).<br>The executor registry.<br>R.3: `IsaacGTWorldModel` on the M2b wire.<br>R.5: mapgen from footprints.<br>The `beyond_reach` reason.<br>**R.2 is not done.** | Box contract tests: 2 passed, then 7 passed and 1 skipped.<br>4 halts mid-walk: 1.5-2.2 ms each, all stopped.<br>navigate 3.06 m as `sonic_walk`. | `outputs/m2b_wave1/world/devbox/` |
| **groot_rt** (done) | `services/executors/groot_arms.py` (R.4).<br>`docs/contracts/arm_chunk.md` v0.1.<br>Two experimental skills.<br>The `full` profile.<br>`groot_sonic` retired. | Against the real PolicyServer with a fake body: 15 inferences, p50 150.5 / p95 165.5 ms.<br>cancel 3/3, halt 3/3, 0 chunks after the ack.<br>F7 `policy_unavailable` in 1.92 s. | `outputs/m2b_wave1/groot_rt/live-20260929T071846Z/` |
| **groot_srv** (done) | The `groot/` client package: PolicyClient, obs, ArmChunk, joint orders by name.<br>`scripts/groot_server.sh` and `scripts/groot_link.sh` (OD3).<br>`docs/groot_serving.md`. | Server ready in 18.2 s, 6.65 GB VRAM.<br>get_action p50 144.2 / p95 148.2 ms (n = 100).<br>Open loop beats the hold-state baseline on every episode and key.<br>0 of 515,200 targets outside the URDF limits. | `outputs/m2b_wave1/groot/` |
| **ops** (done) | `scripts/m2_up.sh`, `m2_down.sh`, `m2_p5.sh`, `m2_venv.sh`, `m2_smoke.sh`, `p5_probe.py`.<br>`docs/bringup.md`. | Cold start to a ready page: 59.0-64.4 s, 3/3.<br>The F1 smoke failed: a scan timeout latched the halt, and the pick was rejected at CAPABILITY. | `outputs/m2b_wave1/ops/` |
| **ui** (partial) | GT confinement in `ui/`.<br>Page additions: the GR00T strip, the list_locations panel, scan thumbnails, the ego pane.<br>A hardened F1 referee.<br>`eval/stack_suite.py` (G1-G14 and E5) and `eval/live_suite.py` (E-1). | E-1 on the laptop: **13/17** (bar 15/17). The 4 failures are reachability.<br>The lite stack subset is 6/7; G6 does not tell the user. | `outputs/m2b_wave1/ui/` |

---

## 2. What the integration reconciled

| Wire | Finding | Change | Test |
|---|---|---|---|
| P1 contract vs the two fakes | `tools/fake_p1.py` (isaac) and `tests/fakes/fake_p1_world.py` (world) serve the same keys for every op and topic. world's `IsaacGTWorldModel` reads the same capabilities from both. | none needed | `tests/contract/test_p1_m2b_fakes.py` (3). Both fakes serve H38 on free ports. Checks the reply keys and error codes, `sim.health`, `gt.objects`, world on both fakes, and groot_arms' `ZmqSensors` on isaac's 5566 after world enables it |
| ego_view in `groot_arms` | P1's `detections` answers `camera_off` unless the camera is on, but the view check ran before the enable. P1 also keeps a camera's `hz` across off/on, which is why the isaac recording ran at 15 Hz. | The camera is enabled **before** the view check, with `hz: 30`, `consumer: <execution id>` and `ttl_s: 10`. The session then waits up to 1.5 s for the first frame and records `camera_first_frame_s`. The camera is disabled on every exit. | `test_groot_arms.py` (the camera test now checks hz, and that the camera comes before the arm) |
| groot/ API vs `groot_arms` | Already exercised: the real `groot/` package runs against the fake PolicyServer (`test_groot_arms.py`, 23 tests). groot_srv's open-loop evidence favours the dataset sentence (3 of 3 runs). | The prompt is now `groot.obs.DEFAULT_PROMPT`, in `config/skills.yaml` and in the executor's default. The `any` skill uses the same sentence with the label swapped. | `test_groot_arms.py`, `test_groot_arms_service.py` |
| `arm_chunk.md` vs the body's `arm` op (master `4b081c2`, read-only) | The body implemented B.8 on master (details in `arm_chunk.md` §8):<br>- `lead_s` defaults to 0.15 there;<br>- the chunk watchdog ends with `client_silent`;<br>- `stop {arms}` ends a chunk session (`ended_by: stop`); its later messages are `stale_session`. | `groot_arms` sends `lead_s: 0.15` and counts it in its own expiry check.<br>`arm_stopped` maps to `halted`.<br>`arm_chunk.md` §8 records the as-built differences. | `tests/services/test_groot_arms_body_arm.py`: the real executor against the real `ArmChannel` at 50 Hz with the body's `arm_sim` plant. It skips on this branch. **On a trial merge (this tree + master's `body/`, `tools/`): 3 passed** (full session, body halt latch, cancel) |
| Trace rows (world) vs the referee (ui) | The referee's `ENVELOPE_KEYS` plus `kind` equal `RESULT_ROW_FIELDS`. Two false results in ops' rehearsal:<br>- a glance fallback counted as the arrival scan;<br>- a capability-rejected pick failed the "names its executor" check. | The arrival scan must have `data.mode == "scan"`.<br>Rejections are exempt from the naming check. | `tests/eval/test_offline_checks.py` (+3) |
| ManipulationService (world) vs `groot_arms` (groot_rt) | `GrootArmOutcome.data` never reached the result, and `groot_then_script` was configured but not implemented. | `inferences`, `chunks_dropped` and `attempts` fill the typed fields; the rest goes into `data`.<br>**`groot_then_script`**: a GR00T attempt that fails for any reason except halted, fell or cancelled, with nothing in the hand, hands over at once to the next healthy non-GR00T candidate (sonic_arm_script, else kinematic_attach). The result then carries the fallback's executor and label, plus `data.attempts` (both), `data.fallback_from` and `data.groot`.<br>The tool timeout leaves room for the fallback.<br>The E5 scorer judges the GR00T attempt.<br>The GR00T strip and the summary tag say `[..., fallback, after groot_arms <reason>]`. | `tests/services/test_groot_then_script.py` (3), `tests/ui/test_groot_strip_from_executor.py` (2), `test_groot_arms_service.py` |
| Harness scan budget (ops' F1 blocker, `docs/bringup.md` §7 item 1) | The fixed 12 s scan budget is shorter than SONIC's in-place scan (12.7-15.1 s). The runtime's own halt after an ignored cancel was never cleared, so every later navigate failed `halted`. | The scan budget is the robot's (`timeout_s("observe")`, 30 s).<br>`run_execution` flags `halted_by_runtime`, and the harness releases that halt unless the user said stop.<br>A scan's in-flight turn is cancelled on cancel or halt.<br>Hints are per tool: an observe timeout no longer reads "the walk took too long". | `tests/unit/test_harness_flows.py` (+1: fails without the fix), `test_execution.py` |
| Shared files | Requests from isaac, groot_srv and ops. | `pyproject.toml` testpaths gain `tests/groot` and `sim_isaac/tests`.<br>`m1_up.sh` preflight checks 5566.<br>Notes added to `p1_m2b.md` §12 and `groot_serving.md` §4. | the suite |

---

## 3. What ran live (dev box, integration pass, 08:58-09:04 box time)

Setup:

- Under the stack lock `integ`, with the branch pushed (md5 of the changed files checked on the box).
- `scripts/m2_up.sh --profile sonic --session integ-m2 --scene procthor-train-40 --planner brains.scripted:create --system1 off`.
- The PolicyServer was groot_srv's `groot-server` (127.0.0.1:5550, started 06:44, ping 0.54 ms).
- The smoke tool is `tools/groot_live_smoke.py` (new).
- Evidence: `outputs/m2b_wave1/integ/box/`.

**Bring-up.** Ready in 59.6 s (`m2_up.log`):

| Stage | Time |
|---|---|
| P1 | 17.3 s |
| body | 0.7 s |
| deploy | 32.2 s |
| stand | 8.2 s |
| P5 | 1.2 s |

P1 answered `m2b-1` with 25 ops, head on and ego_view off.

**Counter stance** (H40 `kitchen_counter_1b`, `pepper_shaker_1`):

1. navigate: `sonic_walk`.
2. Arrival scan: `instance_id_segmentation_fast`, sees 6 objects.
3. check_reachability: `needs_reposition` (0.719 m).
4. reach_stance: 0.24 m.
5. check_reachability: reachable at 0.505 m. It prefers the right arm; the session ran the checkpoint's left arm.

**GR00T chain up to the body boundary** (`live-20260929T090108Z`). The chain is: live ego_view frame (5566) + live
g1_debug (5557), then `groot.obs.build_observation`, the PolicyServer, `to_arm_chunk`, and a chunk-mode `arm`
message. The `arm` op is not on this branch, so the message went to the contract's reference body
(`tests/fakes/fake_arm_body.py`). That body plays the chunks by time and reports `clamped_frac`; nothing moved.

| Measure | Value |
|---|---|
| ego_view, first frame after the enable | 0.182 s |
| ego_view, rate received | 29.44 Hz |
| view check | 13,213 px of the target (P1 segmentation) |
| inferences / chunks sent | 30 / 30 |
| dropped / stale observations | 0 / 0 |
| latency | p50 177.8 ms, p95 219.4 ms (n = 30) |
| clamped_frac (body-side, reference body) | 0.0 |
| stall_s_max | 0.18 |
| outcome | `failed(timeout)` after 12.64 s. Honest: no arm moved, lift 0, `world.palm_position` live, palm-to-object minimum 0.66 m |
| prompt | the dataset sentence with "pepper shaker" |

**Predicted trajectories** (`trajectories.{png,json}`). FK comes from master's `body/g1_kin.py`, in the pelvis
frame.

| Measure | Value |
|---|---|
| Left palm, x range | 0.265-0.355 m |
| Left palm, y range | 0.156-0.304 m |
| Left palm, z range | 0.133-0.220 m |
| Left palm travel within a chunk | p50 0.06 m, max 0.102 m |
| Right palm | static; right hand 0 in every chunk (a left-hand checkpoint) |
| Left hand | reaches Arena's closed pose in 7 of 30 chunks |
| Largest arm step | 0.127 rad per 20 ms, above the body's 6 rad/s slew limit in 1 of 30 chunks |

**Frames.** `frame_01.png`, `frame_06.png` and `frame_11.png` are the exact RGB arrays sent to the policy. Colours
are right (the lettuce is green). The counter top fills the frame, and the objects sit at its top edge. This
confirms isaac's note: at a reach stance about 0.5 m from the object, the Arena-exact ego_view (35° down) sees the
surface's top edge. GR00T's stance is a wave-2 item (W2.6).

**First run** (`live-20260929T085944Z`). 4 inferences (p50 186.8 / p95 193.1 ms), then `grasp_missed` after
2.14 s. This was an artifact of the smoke setup: the real Dex3 hands sit in the deploy's default fist when no hand
fields are sent, and the ground-truth judge read that closure. The tool now turns that judge off when the chunks
stop at the boundary, and says so in `smoke.json`.

**Halt of a GR00T session** (`G1Robot.halt()`: the runtime gate, then the body `stop` op, which is INTERIM until
B.1):

- receipt 1.26 ms, `stopped: true`;
- the session ended `failed(halted)` 36 ms later;
- 0 chunks after the ack (fence-to-ack 0.10 ms), hold `measured`.

**Halts mid-walk through the body** (kitchen_counter_1a and back, halted 2.5 s into each walk):

| Walk | Receipt | Stopped | Result | Speed 1.5 s later | Fall |
|---|---|---|---|---|---|
| 1 | 1.71 ms | yes | `failed(halted)` | 0.010 m/s | no |
| 2 | 1.82 ms | yes | `failed(halted)` | 0.006 m/s | no |

**`full` profile, policy down → labelled fallback** (`full_fallback/out.json`):

- `groot_arms` reported itself down: "the body's arm op has no chunk mode yet (B.8 ...)".
- `sonic_arm_script` is `planned`.
- CAPABILITY chose `bringup.attach.pick.v0`.
- pick `succeeded` through P1 attach, labelled `[bringup.attach.pick.v0, fallback]`, 3.7 s.
- place `succeeded` back on `kitchen_counter_1b`.

**P1 over the run:** `rtf_total` 0.991 and 29 heartbeat publishes; sim health ok (rtf 1.00) at the start and the
end.

**Teardown.** Everything was stopped afterwards: `m2_down.sh` and `groot_server.sh stop` (port 5550 free). The GPU
was at 0 MiB and the lock was released.

**Not run live:**

- chunks into SONIC (no `arm` op on this branch);
- `groot_then_script` falling back after a GR00T attempt (offline only);
- the OD3 tunnel;
- anything on the main box;
- E0-E5.

---

## 4. Tests

| Run | Result |
|---|---|
| Before this pass, `.venv-rt/bin/python -m pytest` | 580 passed, 1 skipped, 19 deselected (156.6 s) |
| After, run 1 | 662 passed, 5 skipped, 22 deselected (179.4 s) |
| After, run 2 | 662 passed, 5 skipped, 22 deselected (180.4 s) |
| `python -m eval.offline_episode --out outputs/m2b_wave1/integ/offline_f1` | every step and check ok, `PASS*` (lite, labelled fallback) |
| GT confinement lints (`tests/unit/test_lints.py`, `tests/ui/test_ui_m2b.py`) | green, inside the runs above |
| Trial merge (this tree + master's `body/`, `tools/`, `sim_isaac/`; master at `1673463`, then at `84ddbfe`), `test_groot_arms_body_arm.py` | 3 passed, 3 runs in a row at each master |
| Trial merge at `1673463`, the whole default suite | 666 passed, 3 skipped, 22 deselected (the body-arm test and groot's `body.joint_map` agreement test run there) |
| Trial merge at `84ddbfe`, the whole default suite | 670 passed, 4 skipped, 22 deselected (master's `sim_isaac/tests/test_hands.py` runs there too) |
| Trial merge `body/tests` in `.venv-rt` | 92 passed, 1 failed (`test_wire.py::test_camera_roundtrip` needs `cv2`, which is not in `.venv-rt`; it fails the same way on this branch alone) |

The suite now also runs `tests/groot` and `sim_isaac/tests`.

New tests:

- `tests/contract/test_p1_m2b_fakes.py` (3)
- `tests/services/test_groot_then_script.py` (3)
- `tests/services/test_groot_arms_body_arm.py` (3, skipped until the merge)
- `tests/ui/test_groot_strip_from_executor.py` (2)
- `tests/eval/test_offline_checks.py` (+3)
- `tests/unit/test_harness_flows.py` (+1)

---

## 5. Known gaps after wave 1

1. **No GR00T chunk has reached SONIC.** The body's `arm` op, chunk mode, arm_script with CarryLock, waist scan,
   halt lane, leases, modes and approach are committed on master (B.1-B.8, body wave; master `arm_tracking.md` §8
   and `docs/contracts/m1.md` §3.9-§3.13 have their live results, including chunk mode with synthetic chunks: 10
   sessions, cancel 3/3, halt 3/3). They are not on this branch, so the runtime still uses the INTERIM halt (body
   `stop`), the in-place scan and the tight-`go_to` reposition.
2. **Two epoch spaces.** `HaltGate` counts its own epochs; the body latch fences by the executions'
   `control_epoch` (R.2).
3. **GR00T stance.** At the reach stance the target sits at the ego_view's top edge (§3 frames). There is no
   GR00T-specific stance yet.
4. **E-1 is 13/17 and the lite stack subset is 6/7.**
   - The 4 E-1 failures are reachability.
   - `fetch_search` and `question_midtask` put the apple beyond G1 reach (`beyond_reach`) and need a scenario
     decision.
   - A `too_far` result with no suggestion still tells the planner to "navigate to the suggested location".
   - G6: the planner does not tell the user after repeated `blocked`.
5. **P1.6 detections agree 66 %, not 90 %.** They are used only for glances, scans and the GR00T view check.
6. **The page's reset does not call P1 `reset_scene`,** so smoke runs restart the stack.
7. **GR00T zero-shot:**
   - stochastic chunks;
   - 1 of 30 live chunks steps above the 6 rad/s slew limit (groot_srv's open loop: 8 of 94);
   - the right hand is always 0.
8. **Not live-verified yet:**
   - `robot_fell` / `object_fell` → `safety_event`;
   - HaltResender against a late live ack;
   - the OD3 tunnel's latency with the 922 KB request.
9. **Main-box hygiene** (`docs/devbox.md` §6.1): delete `/work/logs/wl/isaac-cdds.log` and rotate the keys.
10. **`sync_wl.sh push`** does not exclude `runs/` or `.env*` (00_infra).
11. **`groot.wire`** decodes only the 4b1dca9d npy format; msgpack_numpy replies (Isaac-GR00T 51d4c89) would fail.

---

## 6. Wave-2 plan: the whole loop tested end to end on the main box

In order. Each step has one owner and an exit criterion. Nav2, demo collection and fine-tuning stay deferred.

| # | Task | Owner | Exit |
|---|---|---|---|
| W2.0 | Main-box prep: stop the G0 stack (`wl-m1`); delete `isaac-cdds.log` and rotate the keys; `sync_wl.sh` excludes `runs/` and `.env*`. | ops + lead | The main box is idle, clean and keyed |
| W2.1 | **Merge** master (body wave: B.1-B.8, `docs/contracts/m1.md` v0.6, `arm_tracking.md` §8; the P1 Dex3 hands `5136e79`) with `m2b-wave1`. Two files changed on both sides: `docs/contracts/m1.md` (keep both sections) and `sim_isaac/README.md`. Then take the body and isaac owners' still-uncommitted work once they commit it. | integrator + body + isaac | Full suite green twice in `.venv-rt`, `body/tests` green in the body venv, `test_groot_arms_body_arm.py` runs (not skipped) 3/3 |
| W2.2 | **Body with real GR00T chunks:** the body already ran chunk mode live with synthetic chunks (10 sessions, cancel 3/3, halt 3/3) and B.5/B.7 (master `arm_tracking.md` §8). Left: replay `tests/fakes/fake_arm_body.py`'s sequences in `body/tests` (expired, out_of_order, stale session, NaN); then G2 with real GR00T chunks, 10 sessions through `tools/groot_live_smoke.py --arm body` on the dev box. Close whatever of B.5-B.7 `docs/M2.md` §7.2 still lists open after the merge. | body | G2 with GR00T's chunks: 0 falls, cancel 3/3, halt 3/3 without an arm jerk, `slew_frac` reported |
| W2.3 | **R.2 `SonicBody` on the new body surface:**<br>- halt through the B.1 lane (`BodyClient.halt`), with one epoch space (`G1Robot.halt(control_epoch)`; `HaltGate` epochs = the executions' `control_epoch`);<br>- B.2 leases per body execution;<br>- B.3 `body.mode` / `body.fault` → `G1Robot.events()`;<br>- the waist scan (`ScanConfig.executor: waist`);<br>- `reach_stance` via `approach`. | world | `-m box` contract tests on `sonic` green on the dev box; halt receipt < 30 ms 20/20; no INTERIM label on halt, scan or reposition |
| W2.4 | **R.1 `sonic_arm_script`:** the executor over B.7 `arm_script` + P1.3 attach (STEPPING STONE) + CarryLock. | world | `manipulate(pick)` on `sonic` succeeds from a reachable stance with `executor=sonic_arm_script`, `[fallback]` in prompt, page and eval JSON |
| W2.5 | **GR00T stance:** a per-skill stance so the target sits in the lower two-thirds of `ego_view` (`SkillSpec.stance`; measured on H40 `kitchen_counter_1b`, `bedroom_dresser_1b`, `kitchen_dining_table_1a`). | world + groot_rt | View check ≥ 200 px with the target's centroid below the image's upper third on 3 surfaces |
| W2.6 | **GR00T link on the main box** (OD3): `docs/groot_serving.md` §5 steps 1-8. Keep the server on the dev box, use the tunnel (autossh or the reconnect loop), bench with `obs_ep0_f60.npz`, and run SONIC's timing gate with the GR00T client streaming (walk_diagnosis rules). | groot_srv + ops | `groot_link.sh check` ok; get_action p95 measured main → dev and below the executor's 1.5 s timeout; RTF p10 ≥ 0.98 walking with the client active |
| W2.7 | **`m2_up.sh --profile full` on the main box**, then one live GR00T session into the real `arm` op (`tools/groot_live_smoke.py --arm body`), then `groot_then_script` live: the GR00T attempt fails zero-shot and the labelled fallback (sonic_arm_script, else kinematic_attach) finishes the pick. | ops + integrator | Chunks applied > 0; 0 falls; body `clamped_frac`/`slew_frac` in the result; cancel and halt fence (0 chunks after the ack); a fallback result lists both attempts |
| W2.8 | **Runtime and eval fixes:**<br>- a `too_far` result with no suggestion gets its own hint;<br>- after 2 `blocked` results, tell the user (G6);<br>- the page's reset calls P1 `reset_scene`;<br>- rebind or flag `fetch_search` and `question_midtask` (`beyond_reach`);<br>- rerun E-1. | runtime + ui/eval | E-1 ≥ 15/17 on lite with live models; lite subset 7/7 |
| W2.9 | **E0-E5 on the main box** (`docs/M2.md` §7.5), inside one stack-lock window per stage:<br>- E0: contract `-m box`;<br>- E1: `eval.offline_episode --url … --profile sonic` 3/3;<br>- E2: F1, F3, F4, F6, F8 3/3 each;<br>- E3: 17 scenarios ≥ 14/17 on `sonic`;<br>- E4: `stack_suite` G1-G14 ≥ 12/14;<br>- E5 (GR00T plumbing): `stack_suite --profile full --only E5`, 10 picks. | ui/eval + integrator | `docs/M2.md` §7.6. E5: 0 falls, cancel 3/3, halt 3/3, every result names the `groot_arms` attempt, its skill and the GT outcome; success reported, not required |

Critical path: W2.1 → W2.2 → W2.3 → W2.7 → W2.9 (E5). W2.4 gates E1-E3 (the target pick executor on `sonic`). W2.5,
W2.6 and W2.8 can run in parallel with W2.2-W2.4.
