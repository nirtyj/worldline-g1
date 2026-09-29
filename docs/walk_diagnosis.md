# SONIC walking: diagnosis (2026-09-29)

Independent diagnosis (two investigators + judge, read-only on the deploy agent's files) of why the headless
MuJoCo reference test (`sonic/mujoco_ref/`) failed walking. Full evidence and run data:
`scratchpad/walkdiag/{runs,lensB_runs}` (session scratchpad).

## Verdict

**SONIC walks correctly on a quiet box.** With the unchanged harness and the same arguments as a failing run,
3/3 runs passed every check (quiet-a-20260929-003755 12/12, h1ref-005733 11/11, h2refrt-005958 11/11):
walk_forward 2.70–2.72 m in 6 s (~0.45 m/s), heading error 2.2–3.3°, |lateral| ≤ 0.16 m, 0 falls. Straight
SLOW_WALK at 0.3–0.7 m/s: fall-free 8/8 cycles. The loaded batch (4–8 Isaac Lab processes, load 4.6–14): 1/8.

## Root causes, ranked

| # | Cause | Status | Fix |
|---|---|---|---|
| 1 | **CPU/GPU contention** on the shared box breaks the timing between the wall-clock deploy and the sim. In fall windows 21–34% of 20 ms sim windows lacked exactly one new leg target (clean passes 0–12%); deploy obs→cmd p90 3–7 ms during the IDLE stand (quiet 0.4–0.5 ms); TensorRT policy p90 4 ms (quiet 0.1 ms) | correlation verified, mechanism hypothesis | quiet box for SONIC tests; CPU pinning; timing gate → INVALID |
| 2 | Upstream MuJoCo fall reset (`base_sim.py:508-527`, `mj_resetData`) teleports the robot; the harness measured across teleports (all >1 m/s and wrong-direction results had falls in the window) | verified | stop scoring at the first fall |
| 3 | Harness built commands from the **measured** yaw (after a reset, arbitrary) | verified | track the commanded planner-frame facing like the keyboard client |
| 4 | Single 90° IDLE facing step (planned once, under-rotates ~12°); SLOW_WALK with zero movement (replans every 1 s) | behaviour verified | ≤ 30° facing steps; zero movement → IDLE |

Rejected as causes: wire/protocol (the deploy echoes exact values), walk speed 0.5 m/s, pacing mode, MJCF.

## Applied to the Isaac path (wl-body), commit "body: SONIC command fixes from walk diagnosis"

- `body/sonic_mux.py`: a non-static mode with no movement is sent as IDLE (keyboard_handler.hpp:681-686).
- `body/motions.py` `Motion.turn_cmd`: facing is stepped ≤ 30° from the last **commanded** facing and advances only
  once the body is within 15° (keyboard Q/E semantics). Covers turn_to, walk and the path follower.
- Body tests: 29/29 pass.

## Rules for M1 integration and any SONIC test

1. **Run SONIC tests with no other Isaac/GPU jobs on the box**, or mark the run INVALID. Isaac itself must share
   the GPU with the TensorRT deploy in the integrated stack, so also:
2. **Pin CPUs** (`m1_up.sh` DEPLOY/ISAAC/BODY_TASKSET): give the deploy dedicated cores; consider `chrt -f` for the
   deploy's control thread; keep Isaac render rate modest (head cam ≤ 15–30 Hz, chase cam optional).
3. **Timing gate** (P1 bridge side, `sim_isaac/dds_bridge.py` rates()): per run, compute the share of 20 ms windows
   without exactly one new leg target (`irregular`), the sim RTF over the motion window, heartbeat_pubs during
   motion. INVALID if irregular > 0.15, RTF < 0.98, or heartbeats fired during a motion.
4. **Fall gate**: stop scoring at the first `fallen` (service.py already fails the active motion).
5. Band hand-over as in `sim_isaac/band.py` (anchor floor+0.80 m, yaw at engage, ramped release) matches the
   MuJoCo setup that passes; optionally release only with both feet in contact and |ω| < 0.2 rad/s.

## Still to apply in the MuJoCo harness (deploy agent's files)

P1 fall gate, P2 commanded facing, P3 INVALID verdict in report_ref.py, P4 preflight refusing to run when other
Isaac/GPU processes exist. Verification: 5 runs on a quiet box, all must pass (thresholds in the judge's report).
