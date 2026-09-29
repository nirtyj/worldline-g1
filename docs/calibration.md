# R.7 calibration: walking, the arm's workspace, and what a G1 can reach in the houses

Status: **measured 2026-09-29** (owner world-cal, M2b wave 2). Every number below comes from a run named next to
it; the evidence is under `outputs/m2b_wave2/world-cal/` (not committed; box copies under
`/work/worldline-g1/outputs/m2b_wave2/world-cal/` on `ludo-g1-arena`).

It sets the numbers the planner prompt quotes (`api.types.PROFILES`), the G1 workspace that
`check_reachability` judges with (`config/g1.yaml` `workspace`), the keypoint stands (`mapgen`), and it
explains the eval scenario decisions (`eval/scenes.yaml`).

---

## 1. How it was measured

| Part | Tool | Where |
|---|---|---|
| Walk, stop, turn, go_to, approach | `python -m world.live_cal walk --runs 3` | dev box `ludo-g1-arena`, M1 stack (`scripts/m1_up.sh --house procthor-train-40`, NAV_BACKEND=astar), unmodified SONIC deploy, under the dev stack lock `worldcal` |
| Live palm check, arm-script phase times, long approaches | `python -m world.live_cal arm` | same stack |
| IK envelope + sphere fit | `python -m world.workspace_cal envelope` | laptop, `body/g1_kin.py` (read-only) |
| Coverage in the houses | `python -m world.workspace_cal coverage` | laptop, the recorded houses H40, H15, K10, H38 through `services/reachability.py` |
| 20-view visibility audit (P1.6) | `python -m world.vis_audit` + `verdicts.json` by hand | dev box, P1 alone (DDS domain 7, offset 100, band on, stiff DDS peer) |

Runs:

- `walk-20260929-104348`: 3 runs.
- `walk-20260929-110044`: 3 runs, with the timing gate per op.
- `arm-20260929-105524`: 24 free-air grasps, 3 pick/place sequences, 4 long approaches.
- `visaudit-20260929-111339`: 20 views.

**Timing gate** (`docs/walk_diagnosis.md` rules 1 and 3, PLAN §0.10 d). It is judged per op: P1 `heartbeat_pubs`
during the op, and RTF p10 ≥ 0.98 over 1 s windows of `gt.pose`. The box was quiet apart from ops-groot's
PolicyServer: it was loaded (6.7 GB) but idle for the whole run, at 0-10 % GPU, which is Isaac plus the deploy.

The gate failed often, and for a reason outside these runs. P1 on this stack stalls **every 29.9 s of sim time**:
one head-camera render takes 100-160 ms (`get_stats.hitches_last`, e.g. `t_sim` 905.9 / 935.7 / 965.6 / 995.5 /
1025.4, `render_ms` 103-159). Each stall fires a lowstate heartbeat, so any op longer than about 30 s is INVALID.

- 18 of 60 ops failed the gate in the gated walk session.
- 10 of 54 ops failed it in the arm session.
- Session RTF p10 was 0.9948 (total 0.998), with a worst 1 s window of 0.83-0.85.
- There were 0 falls.

A single 0.15 s stall changes a 20 s duration by under 1 %. The averages below therefore use the ops that passed
the gate, plus the first walk session, which had no per-op gate yet. The all-ops numbers are given next to them.
(Request to isaac/lead: find the 30 s render stall.)

## 2. Walking (SONIC, H40)

| Measure | Result | n |
|---|---|---|
| `walk` straight, commanded 0.30 / 0.45 / 0.60 m/s | steady GT speed **0.40 / 0.57 / 0.71 m/s** (SONIC walks ~25 % faster than commanded) | 5 / 6 / 5 |
| start latency (command to 5 cm/s) | 0.29 / 0.24 / 0.17 s | same |
| stop after the walk's command ends | 0.88 / 1.06 / 0.94 s, glide 0.18 / 0.37 / 0.25 m | same |
| `stop` 3 s into a 0.45 m/s walk | **0.89 s** to rest (0.77-1.10), glide 0.21 m | 6 |
| `turn_to` ±90° / 180° | **17.8 °/s** (about 5.0 s per 90°) / **25.7 °/s** (7.0 s); final error 0.7-2.7° | 11 / 6 |
| `go_to` between keypoints at 0.45 m/s (navigate's cruise) | **0.291 m/s effective = 3.44 s per metre** (97.6 m in 335.8 s); all 24 legs 0.297 m/s | 17 |
| go_to arrival error (GT) | 0.11 m mean, 0.18 m p90; yaw 2.4° mean | 24 |
| `approach` 0.1-0.3 m (to 0.15-0.20 m from a counter, sideways, back) | **15/15** within 5 cm, 3.5 cm mean; **15.1 s mean**, 18.2 median, 22.2 p90; the pelvis stood 0.15-0.20 m from the counter, 0 falls | 15 |
| `approach` 0.53-0.55 m | 3 of 4 (7.8-17.6 s); 1 `final_error` after 8 attempts | 4 |

The navigate speed the prompt quotes is the go_to number: it includes the turn at the start, the stops and the
final approach, which is what "about N s per metre" means to the planner.

## 3. The arm

### 3.1 IK envelope and sphere

`world/workspace_cal.py` samples `body/arm_script.py`'s own IK:

- the seed is SONIC's standing arms;
- the wrist pitch and yaw are locked;
- the null space is pulled to the seed;
- the waist is at 0.

A grasp point passes when the `grasp` goal solves to within 1 cm and the `pregrasp` goal (10 cm back, 5 cm up)
solves to within the body's 2 cm. The grid is 16 heights, 9 laterals and 1 cm forward steps, on the right arm.
`outputs/.../envelope/envelope_right.json`:

| Grasp point height (m above floor) | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.00-1.20 | 1.25 | 1.30 | 1.35 |
|---|---|---|---|---|---|---|---|---|---|---|
| Best forward reach from the pelvis (m) | 0.18 | 0.26 | 0.32 | 0.36 | 0.38 | 0.41 | 0.42-0.43 | 0.40 | 0.38 | 0.34 |

Nothing is reachable at 0.60-0.65 m. The boundary is a sphere around the shoulder:

- centre (0.004, −0.136, +0.299) m in the pelvis frame, on the arm's side;
- radius **0.4255 m**;
- residual rms **3.7 mm**, p90 4.7 mm (3 of 122 rows dropped: IK split runs).

`services/reachability.py`'s shoulder model now is that sphere. The config values:

| Key | Value | Where it comes from |
|---|---|---|
| `shoulder_z_m` | 1.083 | the live standing pelvis, 0.784 m (0.782-0.786, n=24), plus 0.299 |
| `shoulder_lat_m` | 0.136 | the fit |
| `arm_reach_m` | 0.405 | the fit minus 2 cm |
| `grasp_above_top_m` | 0.03 | sonic_arm_script's grasp point (B.7's top grasp) |

Sideways, on the reaching arm's side, the palm gets about 0.50 m from the pelvis; straight ahead it gets 0.43 m.

### 3.2 Live palm check (`arm-20260929-105524`)

24 `arm_script grasp` goals were sent in free air in front of the robot:

- on the sphere's IK edge and 4 cm inside it;
- at grasp heights 0.80 / 0.95 / 1.10 / 1.25 m;
- at lateral 0 / 0.15 / 0.30 m to the right.

**22 of 24 were accepted** (IK error 0.3-11 mm). The 2 rejected (`ik_unreachable`, 2.5-2.8 cm) sat exactly on the
sphere's edge at lateral 0, which is why `arm_reach_m` keeps 2 cm inside the fit.

The palm error (FK of the measured arm on the GT pelvis vs the goal, the body's `palm_err_w_m` p90) depends on
how high the arm is raised:

- at 0.80 m, all 6 points came within 1.0-2.4 cm;
- at 0.95-1.25 m, only 1 of 16 came within 3 cm; the rest were 4.6-17.9 cm off.

The error is almost all to the robot's left (+y, towards the midline) and up: e.g. goal (0.359, −0.150, +0.166),
palm (0.332, −0.027, +0.247). This is the same direction as B-D4 (palm 3.5-4.6 cm left since the Dex3 fix), and
larger. It is a tracking problem, not a reach limit: the IK solved every one of these. The body under test was
body-fix's work in progress (its results carry new fields: `ik_ms`, `ik_seed`, `track_submits`). Request to
body-fix, with this data.

### 3.3 Pick and place times

These are the three arm-script sequences above a counter from a reach stance, with no object attached:

- **pick phases** (pregrasp + grasp + lift + carry): **16.5 and 18.0 s**; a third run had its grasp rejected
  `ik_unreachable` and took 14.2 s;
- **place phases** (lower + release + retract): **8.8 / 10.0 / 10.1 s**, 9.6 s mean;
- phase times: pregrasp 6.0-7.0 s, grasp 4.0-4.5, lift 1.8-2.2, carry 3.7-5.7, lower 4.9-5.3, release 1.6,
  retract 2.0-3.5.

A pick that needs a reach stance adds the approach (15 s mean, §2).

## 4. What a G1 can reach in the houses

`world/workspace_cal.py coverage` runs every pickable that starts on a map surface through the runtime's own
`check_reachability`, with the calibrated arm. The robot starts at the object's surface keypoint; on
`needs_reposition` it moves to the stance and checks again. There are 130 pickables in H40/H15/K10/H38, 110 of them
in the height band. `outputs/.../coverage_final/coverage.json`:

| | In-band pickables |
|---|---|
| Physically reachable at all (some spot 0.20 m from obstacles within the arm's horizontal reach) | **59 / 110 = 54 %** |
| Reachable from the keypoint or after one straight reach stance (≤ 0.60 m), this config | **25 / 110 = 22.7 %** (24.5 % counting another stretch's stand) |
| Same with a two-step reach stance (A* next to the stance, then the approach; `stance_via: two_step`, up to 2 m) | 53 / 110 = 48 % |
| Before R.7 (M2a's config: reach 0.55 m from the pelvis, stance 0.25 m from obstacles) | 18 / 119 = 15 % (and it called reachable some grasps the arm cannot do) |

So "most household pickables" are not reachable. The body and the houses cap it at 54 %: ProcTHOR scatters props
over the whole depth of 0.6 m counters, and a G1 standing 0.20 m from the edge reaches 0.23-0.30 m past it. The
one-step reach stance is the next limit, because many objects sit far from their stretch's only stand. A two-step
reach stance in navigation would take this to 48 % (request to robot).

What changed to get from 15 % to 22.7 %, in the runtime and the config:

- **stance search** (`find_stance`): the object may sit anywhere in the reach window, seen from any direction around
  it (every 10°), rather than straight ahead in a 15 cm band;
- **approach stances**: 0.20 m from the furniture on a straight segment, not A*'s 0.25 m inflated space. Live, the
  body stood 0.15-0.20 m from a counter 15/15 with no fall;
- **approach_max_m 0.60**: the body approach's `max_dist`;
- **beyond_reach** compares against the arm's real horizontal reach at the object's height.

**Stands** (`mapgen`): `stand_off_m` [0.27, 0.45] (was [0.35, 0.50]) and `stand_clearance_m` 0.25. Place never
repositions, and with the calibrated arm no spot on any user surface was in reach from a 0.35 m stand. From 0.27 m,
the H40 and H15 user surfaces accept a placement. Live, go_to to the moved `kitchen_counter_1a` stand (0.30 m of raw
clearance) succeeded 8 of 8, with the body's final error 1.6-6.1 cm. A deep-stretch split for L/U counter legs (`split_deep_m`) is in `world/mapgen.py` but off (0): it gains
4 points and renames the counter stretches the bindings use.

## 5. Scenario decisions (eval/scenes.yaml)

| Scenario | Decision | Evidence |
|---|---|---|
| `fetch_search`, `question_midtask` (E-1 blockers) | **Rebound** to `dish_sponge_1` (bathroom sink, H15; the only sponge; reachable after one reach stance), `status: substituted`, THOR's apple under `original` | The apple is `beyond_reach`: 0.64 m from any spot the pelvis can stand vs 0.50 m reach at its grasp height (`tests/eval/test_scenes.py`) |
| `addition` | Kept (mug); flagged `mugs_out_of_the_calibrated_arm`; `g1_alternative`: wine_bottle_1 | With the calibrated arm neither mug is in one reposition's reach. The wine bottle is (tested). The CD next to the alarm clock is not rendered (§6) |
| `other_side` | Kept (spatula); `status: flagged` `spatula_beyond_reach`; `g1_alternative`: bowl_1 from counter_2c to counter_2a | The spatula is 0.70 m from any stance vs 0.49 m reach, and 0.30 m long across the counter (unplaceable near 2a's edge). The bowl goes round the stove (tested) |

`Bindings.fetch_target(name)` gives a fetch scenario's object, label and sentence after the decision. Requests to
eval-live:

- `eval/suite.py`'s `fetch_search` and `question_midtask` should use it (they still say "apple");
- `addition` and `other_side` should switch to `g1_alternative` for the G1 runs, updating
  `tests/eval/test_suite_offline.py`, which pins mug and spatula.

## 6. P1.6 20-view hand audit (PLAN §0.10 a)

The robot was teleported to 20 H40 surface stands, yaw ±25°. P1's instance segmentation (`detections`, min 40 px)
and world's gt-geometric visibility were drawn on the same head render. `visaudit-20260929-111339/view_NN.png`,
`audit.json`, `verdicts.json`, `audit_summary.json`:

- **12 of 20 views agree**, and every agreeing box sits on its object;
- **13 disagreements in 8 views: the image supports the segmentation 13 of 13**, gt-geometric 0, unclear 0:
  - `remote_control_1` (4 views): its centre is 0.81 m, inside the dresser under its 0.97 m top; geo calls it visible;
  - `cd_1` (3): the 3 mm CD is not rendered on the dresser top; geo counts 1.9-4.0 k px of wood;
  - `vase_3` (2): hidden behind the TV screen, which is not a geo occluder;
  - `remote_control_2` (2): in plain view on the TV stand; geo calls it occluded;
  - `wine_bottle_1`, `pot_1` (1 each): geo boxes are slivers at the frame edge, and no object pixels are in frame.

Verdict: segmentation stays the source of truth (PLAN §0.10 a). One consequence: an object the render does not show
(`cd_1`) cannot be found by a scan on Isaac, so it is not a scenario object.

## 7. Prompt numbers (`api.types.PROFILES`, owner robot) and within-10 % check

| Slot | Proposed | Measured | Match |
|---|---|---|---|
| sonic, full, bringup `walk_speed_mps` (`{v}`, `{s_per_m}`) | 0.29 (3.4 s/m) | 0.291 m/s, 3.44 s/m | 0.3 % |
| sonic `t_pick_s` | 17 | 16.5 / 18.0 s (17.2 mean) | 1 % |
| sonic, full `t_place_s` | 10 | 9.6 s mean | 4 % |
| full `t_pick_s` | 30 (estimate) | GR00T's zero-shot attempt (wave 1 about 12.6 s to its timeout) + 17 s fallback; ops-groot to measure | — |
| `approach_max_m` (config) | 0.60 | 0.53-0.55 m repositions 3/4; the approach op's `max_dist` 0.6 | — |
| `h_min` / `h_max` (config `obj_z_min_m` / `obj_z_max_m`) | 0.70 / 1.25 | IK envelope: grasp points 0.78-1.33 m reach ≥ 0.30 m ahead; live grasps accepted at 0.80-1.25 m | — |

**The lite world is INTERIM.** It keeps M2a's geometry until the tests that pick H38's bed alarm clock move to
an object the G1 can reach:

- the arm (`workspace.lite_world`: reach 0.55 m ahead, judged at the object's centre, stances in A*'s free space);
- the search (`stance_search: band`);
- the stands (`mapgen.lite_world`: 0.35-0.50 m);
- the reachability says so in `lite_arm_note`.

That alarm clock's centre is at 0.62 m, below the calibrated band; its grasp point is 0.69 m, where the palm reaches
0.18 m ahead. With the calibrated geometry on the lite world, the default suite had 24 failures, almost all of them other owners'
tests failing on that one object (`too_low`); with the INTERIM blocks, the default suite is green. The Isaac worlds (sonic, full) build the R.7 stands and judge
with the calibrated arm.

On the lite world as E-1 runs it (checked through the runtime's services on the lite stack; the calibrated-arm
versions are tests in `tests/eval/test_scenes.py`):

| Object | Result |
|---|---|
| H40 alarm clock | one reach stance, then picked and placed |
| H15 dish sponge | one reach stance, then delivered |
| H40 wine bottle | one reach stance, then delivered |
| K10 bowl | from the stand |
| H40 mug_1 | `too_far` (as in wave 1) |
| H40 mug_2 | not seen from its stand |
| K10 spatula | `beyond_reach` |

So E-1 can reach 15/17 once `eval/suite.py` asks for the dish sponge, and 17/17-capable bindings once `addition` and
`other_side` take their `g1_alternative`.
