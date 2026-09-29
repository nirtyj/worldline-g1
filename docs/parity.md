# Parity: where this stack differs from the Ludi robot API doc, and what every shortcut is labelled

Status: M2b wave 1 (2026-09-29, owner world). Sources: PLAN.md §5.11 (deviations D1-D8), §12.2 (honesty labels),
§0.7-§0.8 (owner decisions), docs/M2.md, docs/contracts/p1_m2b.md. The tests named in the last column enforce each
row; "doc §N" is the Ludi robot API reference.

## 1. What is the same in sim and on the real robot (doc §33)

- **One tool interface.** `api/tools.py` is the only definition of the planner's tools; `api/schemas/*.json` are
  generated from it (`python -m api.gen_schemas --check`). Enums are filled once per session and never change
  (Invariant 9); a skill that goes unhealthy is rejected at CAPABILITY, it never leaves the `object_type` enum.
- **One prompt template.** Tool descriptions and the SYSTEM prompt are one template per tool. Only the numeric slots
  of `api.types.RobotProfile.slots()` (walking speed, pick/place times, reach band, wait limit) differ by profile.
  `tests/contract/test_parity.py` masks the digits and compares the text across `lite`, `bringup`, `sonic`, `full`.
- **One result envelope.** Every tool result is an `api.results.ToolResult` (lowercase status, `observation_id`,
  `generation`, `control_epoch`, timing); every trace `result` row carries the same fields
  (`api.results.RESULT_ROW_FIELDS`, docs/api.md §3).
- **One façade.** The runtime (`agent/`, `brains/`, `llmkit/`) sees only `api.services.RobotBridge`; `robot/bridge.py
  G1Robot` implements it for every profile, and `tests/contract/test_robot_contract.py` runs the same contract on
  the runtime's fake, on `lite`, and (`-m box`, WL_BOX=1) on the live `sonic` stack.

## 2. Deviations from the doc (PLAN §5.11)

| # | Doc says | This stack does | Why | Status (M2b wave 1) |
|---|---|---|---|---|
| D1 | §15/§36: GR00T controls "arm + gripper", not whole-body navigation | **Now matches the doc.** Owner decision (b) (PLAN §0.7): GR00T N1.7 outputs arm + Dex3 hand joint targets, streamed through the body `arm` op into SONIC's planner upper-body override; SONIC keeps the legs. `manipulate` never walks; `base_shift_m` is still reported on every result | The token route (UNITREE_G1_SONIC tokens moving the whole body, PLAN §1.3 #25) is retired | Executor `groot_arms` (owner groot_rt, `services/executors/groot_arms.py`), label `experimental`; `groot_sonic` stays only as a down stub |
| D2 | §33: same tool interface **and system prompt** in sim and real | Same schemas and template; only the numeric slots differ per profile | The bodies really differ in speed; "how long" answers should be true | `test_parity` |
| D3 | §2/§31: six tools | Six tools plus `recall` [WL] | Worldline's memory layer | `api/tools.py` |
| D4 | §8/§10: type-only arguments | `object_id` [WL], optional, a plain string checked against belief (no enum) | Instance ids for belief, the goal check and the eval | `agent/validate.py` ENUM stage |
| D5 | §9: "navigate / reposition" (unspecified) | `navigate(location="reach_stance")` [WL] | A doc-consistent reposition without a seventh tool | INTERIM executor: a tight-tolerance `go_to` until the body's `approach` op (B.6) |
| D6 | §45: ego + panorama cameras | A sim-added wide `head` camera (System 1, scans) plus GR00T's `ego_view` (OD1: Arena's G1 head camera exactly, rendered only while enabled). No panorama yet (M7) | PLAN §1.3 #7; OD1 | The head camera is labelled `camera: head (sim-added)` (§3). An M1 P1 has no head camera: the world then models the d435 camera it does render and says so (`IsaacGTWorldModel.camera_note`) |
| D7 | §46: `WAITING` exits unspecified; no `OBSERVING`/`STOPPED`/`FAULT` states | States added (`api/state_machine.py`) | Doc gap | `tests/unit/test_state_machine.py` |
| D8 | §21: harness validation | Also the Worldline rules C4-C13 and the deferrals D9/D10 (`agent/validate.py`) | Kept Worldline behaviour | `tests/unit/test_validate.py` |

## 3. Honesty labels (PLAN §12.2): every shortcut, and where it shows

| Shortcut | Label | Visible in | Counted as |
|---|---|---|---|
| Kinematic base motion (the `lite` body) | `executor: lite`, STEPPING STONE | Result, ACTIONS `[fallback]`, page badge, eval `executors_used` | Fallback pass (`PASS*`) |
| Kinematic attach grasp | `skill: bringup.attach.*` / `lite.*`, `executor: kinematic_attach` / `lite`, `stepping_stone: true`; P1's attach reply carries `stepping_stone: true` too | same | Fallback pass |
| SONIC arm script + GT attach | `skill: sonic.script.*`, `executor: sonic_arm_script` | same | Fallback pass (walking is real SONIC) |
| Off-the-shelf GR00T checkpoint | `executor: groot_arms`, skill label `experimental`, checkpoint name in the registry | Result, skill registry, page GR00T strip, eval | Target executor; success reported as it happens (not expected zero-shot, PLAN §0.8) |
| **Ground truth as perception** | `source: isaac-gt` / `lite-gt` on every observation, perception row and result; `method: gt-geometric` (frustum + AABB occlusion) or `instance_id_segmentation_fast` (P1.6) on every detection and observation | Results, observations (`extra.method`), the page's truth panel, this document | Allowed at this stage (PLAN §13). Inside the runtime every GT read lives in `world/` (GT confinement, below) |
| Sim-added head camera | `camera: head (sim-added)`: `lookup_keypoints()["camera"]["caption"]` when the model is `sim_added` | Page camera caption, this document | — |
| Reposition without `approach` | `navigate(reach_stance)` result says INTERIM (a `go_to` with a 0.10 m tolerance) | Result | — |
| Scan without the waist | `scan_executor: turn_in_place`, `scan_note: INTERIM` on every scan observation | Observation extras, page | — |
| Halt without the latch lane | receipt `via: "body stop op (M1 body has no halt lane)"`; an unacked halt is re-sent every 100 ms (`robot/health.py`) | Halt receipt, trace (`halt_acked`, `safety_event halt_unacked`) | — |
| A fall | P1 `robot_fell` becomes `safety_event{kind: fell, source: sim}` (the harness stops); PLAN's `sim_recovery{path}` event is the body's (B.3), not emitted yet | Trace, page, eval row | Counts toward the limit |
| Scene fixtures moving objects | `fixtures_ok` row; P1 `move_object` / `push_object` are test-only and counted in `object_writes` | eval JSON | — |
| Elastic band | `band` state | Page body panel | Never engaged during scored motion |
| `follow` attach (object kinematic while held) | `attach_mode: follow` in `grasp_state` and results | Page hand badge, eval JSON | Part of the STEPPING STONE grasp |
| Sim below real time | `sim.state` `degraded` / `unsafe` with the RTF (`WorldModel.sim_health`); CAPABILITY rejections "policy unavailable: sim below real time ..." / "navigation stack unavailable (sim below real time ...)"; `capability_changed` event + NOTE | Result, trace, page (`telemetry.body.sim`) | — |

## 4. Ground-truth confinement

The runtime may read simulator truth only through `world/` (PLAN §6.2): the WorldModel's semantic methods
(`robot_pose`, `detections`, `object`, `hands`, `free_spot`, `sim_health`, `planar_speed`, `palm_position`, ...).
`tests/unit/test_lints.py::test_gt_reads_live_in_world` checks `agent/`, `brains/`, `llmkit/`, `robot/`, `services/`
and `ui/` for: imports of P1's GT client (`body.p1_client`) or the sim process packages; `gt_pose` field access; P1
GT topic literals (`gt.pose`, `gt.event(s)`, `gt.objects`, `sim.health`); and a `viz.tap.FrameTap` built without
`gt_pose=False`. The M2a verifier's findings (the body's copy of gt.pose used for health, rest and telemetry;
the page's FrameTap on gt.pose) are fixed and pinned by `test_the_gt_lint_catches_the_m2a_findings`.

The planner never sees GT: `agent/` builds its prompt from belief only, and GT reaches belief only as perception
hints, observation `LookData` and result `data` fields.

## 5. Not at parity yet (tracked)

- No perception stack: `real_g1` is a skeleton (PLAN §13).
- GR00T grasp success is not expected with the off-the-shelf checkpoint; fine-tuning is deferred (PLAN §0.8).
- The G1 workspace is uncalibrated (reach 0.20-0.55 m forward, `config/g1.yaml`); a sampling of the arm chain
  (`body/g1_kin.py`) puts the palm at about 0.41 m forward at counter height standing upright and 0.50 m with a
  20 deg waist lean (R.7 calibrates it).
