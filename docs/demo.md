# Demo script: Worldline on the G1, one run

`scripts/demo.sh` says nine things to the live Worldline UI, in order, and judges each from the page trace.
It runs on the main box against the full-profile stack: Isaac + SONIC + body, the live planner (Gemini 3.8 Flash),
System 1 (Jev + Gemini Live), house `procthor-train-40`. By default the GR00T link is down during the demo, so the
pick uses the labelled SONIC arm-script fallback (`--groot on` keeps the GR00T attempt; see the limits below).

## Command

From the laptop, in `worldline-g1/` (fresh stack, recorded, about 10 min: 70 s bring-up, 8-9 min of steps):

```
BREV_NAME=ludo-g1-brev2 ../ludo_robotics_prep_g1/00_infra/ssh.sh 'bash /work/worldline-g1/scripts/demo.sh --fresh --record'
```

To watch it live: `../ludo_robotics_prep_g1/00_infra/tunnel.sh 8766 8765`, then http://localhost:8766
(Worldline UI) and http://localhost:8765 (Sim Viewer). The page reconnects once `--fresh` has brought the stack back up.

| Option | What it does |
|---|---|
| `--fresh` | `m2_down.sh`, then `m2_up.sh --profile full --scene procthor-train-40 --viz low --p5-port 8766`, viz-server ensured, then waits for the page and System 1. Also moves the house's spatial memory to `runs/memory_backup/`, so the robot starts knowing nothing |
| `--groot off` (default) | takes the OD3 GR00T link down for the demo (`m2_up.sh --groot off`): `full` finds GR00T unavailable and the pick goes straight to the labelled SONIC arm script. The link is brought back up at exit (`groot_link.sh ensure`) |
| `--groot on` | keeps GR00T: the pick first runs the experimental groot_arms attempt (26 s, zero-shot, times out), then the script |
| `--keep-memory` | with `--fresh`: keep the memory and notes of earlier sessions |
| `--record` | Sim Viewer recording around the steps; the run dir is printed (`composite.mp4`, `contact_sheet.png`) |
| `--only 4,5` | only these steps |
| `--list` | print the steps |
| `--no-lock` | skip the stack lock. By default the script takes `/work/locks/stack.d` as `demo` and releases it at exit; if someone else holds it, the script stops (`DEMO_LOCK_OWNER=<owner>` runs under that owner's lock) |

Without `--fresh` the script uses the stack as it is. It warns if the robot is already holding something, because
place does not work yet (see the limits below) and step 8 then fails.

Outputs go to `/work/worldline-g1/outputs/demo/run-<ts>/`: `stepN.log` (what `tools/say.py` printed), `stepN.json`
(lines sent + every trace row), `results.md` (the table), `stack.log`. The script exits 0 only when every step passes.

## The steps

The steps, their timing and their PASS rules live in one file, `config/demo_steps.yaml` (the rules' code:
`tools/demo_steps.py`). This script reads it (`python -m tools.demo_steps bash` for the lines, `results` for the
table), and so does the Worldline UI's Demo panel (below), so both run the same lines and judge them the same way.
Edit the file, not the script.

| # | Said | What it shows | PASS when the trace has |
|---|---|---|---|
| 1 | "my keys are usually on the kitchen counter" | System 1 labels a statement about the home; it is kept word for word as a note | a `note_saved` row |
| 2 | "go to the bedroom dresser", 6 s later "where are you going?" | navigation (A* + SONIC walk), and a question answered while the walk goes on | navigate to the dresser succeeded, and speech that started after the question and before arrival |
| 3 | "go to the dining table", 6 s later "stop", 5 s later "okay, carry on" | halt mid-walk (body halted first, then the ack), resume, and the planner picks the task up again | a `stop` row, a `resume` row, then navigate to a dining table succeeded |
| 4 | "where are my keys?" | recall of the note from step 1 | speech naming the counter |
| 5 | "where is the vase?" | spatial memory built by the looks during the walks (vase_1 on the dresser, vase_2 on the kitchen counter) | speech naming the dresser or the counter |
| 6 | "go to the living room", 5 s later "no, go to the kitchen counter instead" | a correction replaces the running task | navigate to a kitchen counter succeeded after the correction |
| 7 | "pick up the bowl" | honest refusal: the robot walks to the table and checks; bowl_1 is beyond the arm's reach from every place the robot can stand, and it says so (the check gives the distances) | a `check_reachability` result with `beyond_reach`, then speech |
| 8 | "pick up the white bottle" (if asked which: "the white one") | navigate, check_reachability, reach_stance on the far side of the table (about 3.4 m round it), check again, then manipulate with the SONIC arm script (labelled `fallback`; with `--groot on`, after the GR00T attempt); held and lifted. If the grasp misses (`ik_unreachable`), the planner repositions a few cm and picks again | a pick result `succeeded` with `holding: true` |
| 9 | "what are you holding?", then "thanks" | a state question and chitchat | speech naming the bottle |

## From the page: the Demo panel

The same steps run from the Worldline UI: press **Demo** in the chat header (http://localhost:8766 through
`tunnel.sh 8766 8765`). The steps live in one file, `config/demo_steps.yaml`, which this script reads too
(`python -m tools.demo_steps list|bash|results`), so the panel and the script always run the same lines with the
same PASS rules. Hover a step to see its lines and its PASS rule.

| Control | What it does |
|---|---|
| **Run** (a step) / **Run all** | P5 says each line through the chat's own path (a stop word halts first, then System 1 labels the line, then the runtime hears it), so the chat shows it as if typed. The timing is `tools/say.py`'s: a plain line waits until the robot has been idle for 5 s; "@N text" goes N s after the previous line. Each step then shows PASS or FAIL from the same rule as the script, with the evidence line. One demo at a time; the buttons are off while it runs. On the full profile GR00T is off while it runs, as with the script's default `--groot off` (the link comes back after; `groot: on` in the steps file keeps it) |
| **Stop** | no further demo lines, then "stop" to the robot once (the keyword path: halt, no model call) |
| **Record** | the Sim Viewer's recording (`POST :8765/api/record`), start and stop; the recording's folder shows in the panel. Isaac profiles only |
| **Fresh restart** | Isaac profiles, after a confirm in the page: a detached `m2_down.sh && m2_up.sh --profile <this> --scene <this> --viz low --p5-port <this>` (plus `--groot off` on full, as `--fresh` does, so a GR00T link that is down cannot stop the restart). The page shows the restart (about a minute) and reconnects by itself. Unlike `--fresh` it keeps the house's spatial memory |

On the box a run holds the stack lock as `ui-demo` and gives it back after; a lock held by anyone else stops the run
before its first line (the restart takes it as `ui-demo-fresh`). Each run leaves `runs/demo/<ts>/stepN.json` (the
shape of `tools/say.py --json`) and `results.md` (the script's table). The panel also works on the laptop's lite
profile (`python -m ui.server --profile lite`), without the lock, Record or Fresh restart.

## Expected output (per step, then the table)

```
==================== step 8: pick (reach stance + arm) ====================
connected: scene procthor-train-40, profile full, planner gemini-3.8-flash

you      pick up the white bottle
  label    correction (by Jev, target bottle, conf 0.61)
  plan     speak(text="I'll pick up the white bottle.")
  plan     navigate(location='kitchen_dining_table_1a')
  succeeded navigate: arrived at kitchen_dining_table_1a (1.1 m, 20 s, sonic_walk)  [sonic_walk]
  succeeded observe: scan at kitchen_dining_table_1a: sees bottle_1, bowl_1, butter_knife_1, knife_1, potato_1, statue_1, wine_bottle_1
  plan     check_reachability(object_id='bottle_1', object_type='bottle')
  succeeded check_reachability: bottle_1 visible but not reachable from this exact pose: needs_reposition (1.1 m): reach stance on the other side of kitchen_dining_table_1 (-y side), 2.5 m walk; ...
  plan     navigate(location='reach_stance')
  succeeded navigate: repositioned 3.44 m to the reach stance (sonic_walk); check reachability again  [sonic_walk]
  succeeded check_reachability: bottle_1 visible, reachable with the left arm (0.5 m)
  plan     manipulate(action='pick', object_type='bottle', object_id='bottle_1', arm='left')
  failed    manipulate: pick bottle_1 failed [sonic.script.pick.v0, fallback]: ik_unreachable (...); hand empty  [sonic_arm_script]
  succeeded navigate: repositioned 0.11 m to the reach stance (sonic_walk); check reachability again  [sonic_walk]
  succeeded check_reachability: bottle_1 visible, reachable with the left arm (0.4 m)
  succeeded manipulate: picked bottle_1 with the left hand [sonic.script.pick.v0, fallback], 12.1 s; verifying  [sonic_arm_script]
  robot    "I have picked up the white bottle."
...
==================== results ====================
| step | behaviour | result | s | evidence |
|---|---|---|---|---|
| 1 | memory note | PASS | 15 | note saved: 'my keys are usually on the kitchen counter' |
| 2 | navigate + question mid-walk | PASS | 44 | arrived at bedroom_dresser_1a (5.8 m, 16 s, sonic_walk); answered while walking: 'I am heading to the bedroom dresser.' |
| 3 | stop / resume while walking | PASS | 39 | stop=1 (canceled ['navigate']), resume=1, arrived after resume: arrived at kitchen_dining_table_1a (3.3 m, 10 s, sonic_walk) |
| 4 | recall a note | PASS | 28 | said: "You mentioned your keys are usually on the kitchen counter, but I haven't checked there yet." |
| 5 | spatial memory | PASS | 26 | said: 'The vase is on the bedroom dresser.' |
| 6 | correction mid-walk | PASS | 53 | labelled ['correction']; arrived at kitchen_counter_1a (5.3 m, 20 s, sonic_walk) |
| 7 | honest refusal (reach) | PASS | 39 | bowl_1 visible but not reachable from here: beyond_reach (0.9 m): bowl_1 is 0.52 m from the nearest spot ...; said "I can't reach the bowl because it's too far back on the table." |
| 8 | pick (reach stance + arm) | PASS | 133 | picked bottle_1 with the left hand [sonic.script.pick.v0, fallback], 12.1 s; ...; end hands {'left': 'bottle_1'}; path ... |
| 9 | question + chitchat | PASS | 37 | said: 'I am holding the white bottle.' |

9/9 steps passed; total 8 min 22 s
```

## Live runs (2026-09-29, main box)

| Run | Script version | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Steps time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | step 7 "can you reach the bowl?", GR00T on | PASS | PASS | PASS | PASS | PASS | PASS | FAIL (answered from belief, no check) | PASS (after groot_arms timeout) | PASS | 8 min 34 s |
| 2 | step 7 "pick up the bowl", GR00T on | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL (ik_unreachable, then reach_stance loop) | FAIL (hands empty, said so) | 9 min 37 s |
| 3 | final (GR00T off) | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS (retry after ik_unreachable) | PASS | 8 min 22 s |
| 4 | final (GR00T off) | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS (first try, 0.4 m, 14.7 s) | PASS | 8 min 1 s |

Plus about 70 s for `--fresh` (m2_down 8 s, m2_up 60 s) and a few seconds to stop the recording. Run 4's recording
(Sim Viewer composite, 7 min) is `outputs/demo/composite.mp4` + `contact_sheet.png` on the laptop (git-ignored); the
robot stayed upright in every run.

## Known limits (what the script avoids)

- **Place does not work** (`no_room_in_reach` on every surface tried), so there is no "bring me ..." or "put it
  down" step, and the robot ends the demo holding the bottle. Run with `--fresh` before the next demo.
- **GR00T does not grasp, and its attempt can spoil the pick** (why `--groot off` is the default): zero-shot N1.7
  times out after 26 s. In run 2 the scripted fallback then failed `ik_unreachable`, and every later check put the
  bottle 0.48-0.51 m away (it was 0.44 m before the attempt); earlier, "pick up the wine bottle" failed the same way
  after the base drifted 0.2 m.
- **Bug: reach_stance loops when the object sits at the edge of the reach.** After run 2's failed grasp,
  `check_reachability` answered `needs_reposition` 8 times in a row while `navigate(reach_stance)` moved the robot
  0.00-0.08 m each time (the stance it proposes is where the robot already stands, object_left 0.40 m); the planner
  gave up after about 2 min ("too far across the table"). Trace: `outputs/demo/run-20260929-232055/step8.json` on the box.
- **Bug: reachable is looser than the scripted IK.** `check_reachability` says "reachable with the left arm (0.5 m)"
  and `sonic_arm_script` then fails `ik_unreachable` from that pose (run 3; also the wine-bottle run). It works at 0.4 m. The
  planner recovers with a small reposition when the stance loop above does not kick in.
- **Planner answers "can you reach X?" from belief** ("The bowl is on the dining table, so I cannot reach it from
  here") without check_reachability, so step 7 asks "pick up the bowl" instead, which makes it walk there and check.
- **Say "the white bottle"**: "the bottle" is ambiguous (bottle_1 and wine_bottle_1 on the same table); the planner
  may ask which one, and the script answers "the white one".
- **No page reset**: resetting the scene from the page made the robot fall once; the script never resets, `--fresh`
  restarts the stack instead.
- Jev sometimes labels a new request as `correction` (steps 3 and 8, conf 0.55-0.65) after a finished task; the
  planner still does the right thing.
- Answers come from a live LLM, so the wording changes from run to run; the PASS checks look for the facts (a note
  saved, a place named, a result status), not for exact sentences.
