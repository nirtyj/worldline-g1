# M2b status (2026-09-29, end of day)

Plain status for the owner. Numbers come from the linked docs or from the wrap runs of 2026-09-29 18:20-18:47 UTC.
Wrap outputs are under `outputs/m2b_finish/wrap/` (git-ignored, laptop only).

## 1. What works live today

| Item | Result | Evidence |
|---|---|---|
| M1: SONIC G1 stands and walks in a ProcTHOR house in Isaac, driven through our API | E1-E7 pass (8/8 up/down cycles, 9/9 60 s stands, 3/3 walk/turn/strafe/stop/go_to runs, RTF 0.994-0.995) | `docs/M1.md` §0 |
| Worldline navigation live on Isaac + SONIC, incl. the owner's live-Gemini "go to the bowl" run | ran live on the main box (owner run) | owner run; no run folder is linked from the repo docs |
| Stop / halt | 20/20 mid-walk halts stopped, `halt()` p50 0.82 ms, at rest within 1.5 s 19/20 (20th 1.67 s), 0 falls | `docs/api.md` §9 |
| Body wave: halt lane, chunk playback, scripted pick + carry | scripted pick succeeded in 15.4 s (grasp palm error median 2.35 cm, lifted 0.148 m), 7.4 m carry kept the object | `docs/api.md` §9, `docs/arm_tracking.md` §8-§10 |
| GR00T chain (PolicyServer on the dev box through the OD3 tunnel into the body `arm` op) | remote sessions 50 inferences / 50 chunks, clamped 0.0; get_action p50 about 143 ms | `docs/groot_serving.md`, wrap gates A-C |
| Planner idle-wake fix | planner no longer wakes on passive belief changes after a finished task; 4 new tests, default suite 815 passed | commit 9b6b551 |

### Timing: GR00T local vs off-box (SONIC standing with GR00T inference, bar RTF p10 >= 0.98, no heartbeats)

| Setup | stand_groot RTF p10 | get_action p50 | Gate |
|---|---|---|---|
| Local PolicyServer, unpinned / CPUs 12-15 (A0, B1) | 0.84-0.92 | 164-187 ms | fail |
| Local, 15 Hz ego (B2, B3) | 0.969-0.976 | | fail |
| Local, 10 Hz ego (B4, B4b, B5) | 0.9959-0.9973 | | fail (heartbeats) |
| Off-box, ego/head 10 Hz (A) | 0.9998 | 143 ms | fail (3 heartbeats) |
| Off-box, 2.5 Hz (B) | 0.9925-0.9939 | 143-145 ms | fail (5 heartbeats) |
| Off-box, area256 (C) | 0.9946-0.9947 | about 140 ms | fail (2 heartbeats) |

Read: off-box GR00T keeps RTF and balance healthy (0 falls in all 12 gates), but every valid run still has 1-3 heartbeats per 20 s.
`walk_base` with no GR00T also has heartbeats (B 11, D 14), so the heartbeat rule fails while walking whether or not GR00T runs.
The high `irregular` share in `stand_base` comes from head-camera render windows (30 Hz head: 0.0-0.4 empty; VizCams off: 0-0.003).
Gate D `stand_groot` is invalid (3 sessions "superseded", 0 inferences; cause not found).

## 2. Not done or not verified

- Full eval stages E0-E5 were not run as one set on the full stack.
- GR00T grasp success is not expected (zero-shot N1.7, no fine-tune; M6 deferred).
- Place failed `ik_unreachable` at the service's spot (IK-envelope retry added after, not re-verified live).
- E3 17 scenarios and the E4 stack suite were not run on the full stack.
- Live G2 check (3 GR00T sessions into the real arm op) skipped: the stack lock was held.
- `body/tests/test_integration_fakes.py::test_watchdogs_fall_and_shutdown` fails in the main checkout only (passes in a clean worktree); cause not found.
- Wake fix: one full default-suite run, not two; body suite not re-run after it.
- Box state at wrap: stack lock owned by `gmain` on both boxes; tmux `bregress-c2` waiter (runs `chain2.sh` if the lock frees) gives up about 18:48 UTC. Remove the lock after that.

## 3. Decisions of the day (PLAN §0)

| § | Decision |
|---|---|
| 0.11 | GR00T PolicyServer on the main box (superseded by 0.12) |
| 0.12 | GR00T back on the dev box via the OD3 tunnel: local GR00T broke SONIC wall-clock timing (RTF p10 0.84-0.97) |
| 0.13 | Stay on the wall clock for the POC; `--sim-clock` deploy patch (option B) only if the gate still fails |
| 0.14 | Robot API boundary: navigation (A*) and manipulation (GR00T) become P3's skills; Worldline sees only the API |

## 4. Next steps, in order

1. Move the robot API into P3 per §0.14.
2. Lean eval with the real Gemini planner on the full stack.
3. Sim-clock patch (§0.13 option B) only if timing fails in that eval.

## 5. Bring the stack up and open the UIs

```
BREV_NAME=ludo-g1-brev2 ../ludo_robotics_prep_g1/00_infra/ssh.sh      # main box
bash scripts/groot_link.sh ensure                                      # OD3 tunnel to the dev-box PolicyServer
scripts/m2_up.sh --profile full                                        # Isaac + SONIC + body + P5 (see docs/bringup.md)
../ludo_robotics_prep_g1/00_infra/tunnel.sh 8766 8765                  # on the laptop
```

Worldline UI: http://localhost:8766. Sim Viewer: http://localhost:8765. Stop with `scripts/m2_down.sh`.
