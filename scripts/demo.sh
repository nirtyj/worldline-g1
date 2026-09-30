#!/usr/bin/env bash
# The Worldline-on-G1 demo: the most important behaviours in one run, said to the live Worldline UI in order
# (tools/say.py), then a PASS/FAIL table judged from the page trace. Runs ON the main box. docs/demo.md.
#
#   scripts/demo.sh [--fresh] [--record] [--groot off|on] [--only N[,N..]] [--list] [--keep-memory] [--label L] [--no-lock]
#
#   --fresh        scripts/m2_down.sh, then scripts/m2_up.sh --profile full --scene procthor-train-40 --viz low
#                  --p5-port 8766 (live planner + Jev + Gemini Live), the GR00T link and the Sim Viewer (viz-server,
#                  8765) ensured; waits for the page and System 1. The robot starts standing, hands empty.
#                  Also starts the house's spatial memory empty (runs/memory/<scene>.json is moved to
#                  runs/memory_backup/), so what the robot knows in the demo it learned in the demo.
#   --keep-memory  with --fresh: keep the spatial memory and notes of earlier sessions
#   --groot off    (default) the OD3 GR00T link is taken down for the demo (m2_up.sh --groot off), so `full` rejects
#                  the GR00T attempt as policy_unavailable in ~3 s and picks with the labelled SONIC arm-script
#                  fallback. The link is brought back (groot_link.sh ensure) at exit.
#   --groot on     keep GR00T: the pick first runs the experimental 26 s groot_arms attempt (zero-shot, times out),
#                  then the script. 2 of 3 live picks passed this way: the attempt can move the base or the bottle,
#                  the script then fails ik_unreachable and reach_stance loops (docs/demo.md, known limits)
#   --record       Sim Viewer recording around the steps (POST :8765/api/record); prints the run dir
#                  (composite.mp4, contact_sheet.png, head/chase/top.mp4, summary.json)
#   --only N,M     only these steps (numbers from --list); step 8 needs the robot's hands empty
#   --list         print the steps and exit
#   --no-lock      do not check or take /work/locks/stack.d (default: take it as `demo`, released at exit; a lock
#                  held by someone else stops the demo unless DEMO_LOCK_OWNER names that owner)
#
# Laptop one-liner (the stack's output streams back; about 15 min with --fresh):
#   BREV_NAME=ludo-g1-brev2 ../ludo_robotics_prep_g1/00_infra/ssh.sh 'bash /work/worldline-g1/scripts/demo.sh --fresh --record'
# Watch it: ../ludo_robotics_prep_g1/00_infra/tunnel.sh 8766 8765 -> http://localhost:8766 (Worldline UI),
#           http://localhost:8765 (Sim Viewer). Outputs: /work/worldline-g1/outputs/demo/run-<ts>/ (stepN.log,
#           stepN.json, results.md). Exit 0 when every step run passed, 1 otherwise.
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true

WL=${WL:-/work/worldline-g1}
PY_RT=${PY_RT:-$WL/.venv-rt/bin/python}
SCENE=procthor-train-40; SESSION=wl-m2; P5_PORT=8766; VIZ_PORT=8765
FRESH=0; RECORD=0; ONLY=""; LIST=0; KEEP_MEM=0; LABEL=demo; USE_LOCK=1; GROOT=off
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift;;
    --record) RECORD=1; shift;;
    --only) ONLY="$2"; shift 2;;
    --list) LIST=1; shift;;
    --keep-memory) KEEP_MEM=1; shift;;
    --label) LABEL="$2"; shift 2;;
    --groot) GROOT="$2"; shift 2;;
    --no-lock) USE_LOCK=0; shift;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) echo "unknown option $1 (--help)" >&2; exit 2;;
  esac
done
case "$GROOT" in on|off) ;; *) echo "--groot must be on or off" >&2; exit 2;; esac

# ---------------------------------------------------------------- the steps
# config/demo_steps.yaml: the one list this script and the Worldline UI's Demo panel (ui/demo.py) run. tools/demo_steps.py
# prints step_def N (sets NAME, MAXS = s to wait on one line, LINES = what to say, "@N text" = N s after the previous
# line, EXTRA = more tools.say options; returns 1 for no such step), STEP_IDS, NSTEPS, SAY_OPTS, READY_S, BETWEEN_S.
SRC=$WL; [[ -f "$SRC/tools/demo_steps.py" ]] || SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY_STEPS=$PY_RT; [[ -x "$PY_STEPS" ]] || PY_STEPS=$SRC/.venv-rt/bin/python; [[ -x "$PY_STEPS" ]] || PY_STEPS=python3
STEPS_SH=$(cd "$SRC" && "$PY_STEPS" -m tools.demo_steps bash) || { echo "cannot read the steps ($SRC/config/demo_steps.yaml)" >&2; exit 2; }
eval "$STEPS_SH"

if [[ "$LIST" == 1 ]]; then
  for n in "${STEP_IDS[@]}"; do
    step_def "$n"; printf '%d  %-30s' "$n" "$NAME"; printf ' "%s"' "${LINES[@]}"; echo
  done
  exit 0
fi
STEPS=()
if [[ -n "$ONLY" ]]; then
  IFS=, read -ra STEPS <<< "$ONLY"
  for n in "${STEPS[@]}"; do step_def "$n" >/dev/null || { echo "no step $n (--list)" >&2; exit 2; }; done
else
  STEPS=("${STEP_IDS[@]}")
fi

T0=$(date +%s)
TS=$(date +%Y%m%d-%H%M%S)
RUN=$WL/outputs/demo/run-$TS
mkdir -p "$RUN"
say() { echo "[demo $(date +%H:%M:%S)] $*" | tee -a "$RUN/demo.log"; }
banner() { printf '\n==================== %s ====================\n' "$*" | tee -a "$RUN/demo.log"; }

# ---------------------------------------------------------------- the stack lock (docs/bringup.md)
LOCK=/work/locks/stack.d; OWN_LOCK=0; LINK_DOWN=0
cleanup() {
  if [[ "$LINK_DOWN" == 1 ]]; then        # the GR00T link as we found it: up (the stack's GR00T calls work again)
    bash "$WL/scripts/groot_link.sh" ensure >/dev/null 2>&1 && echo "[demo] GR00T link restored" || echo "[demo] WARNING: GR00T link not restored: bash scripts/groot_link.sh ensure"
  fi
  [[ "$OWN_LOCK" == 1 ]] && rm -rf "$LOCK"
  true
}
trap cleanup EXIT
if [[ "$USE_LOCK" == 1 ]]; then
  mkdir -p /work/locks
  if mkdir "$LOCK" 2>/dev/null; then
    echo "demo $(date +%s)" > "$LOCK/owner"; OWN_LOCK=1
    say "stack lock taken as demo (released at exit)"
  else
    holder=$(awk '{print $1}' "$LOCK/owner" 2>/dev/null)
    if [[ -n "${DEMO_LOCK_OWNER:-}" && "$holder" == "$DEMO_LOCK_OWNER" ]]; then
      say "stack lock held by $holder (DEMO_LOCK_OWNER): going on"
    else
      echo "[demo] the stack lock is held by '$(cat "$LOCK/owner" 2>/dev/null)': someone is using the stack." >&2
      echo "[demo] wait, or if it is stale: rm -rf $LOCK (or DEMO_LOCK_OWNER=$holder, or --no-lock)" >&2
      exit 4
    fi
  fi
fi

# ---------------------------------------------------------------- GR00T off: the OD3 link down for the demo
if [[ "$GROOT" == off ]]; then
  if bash "$WL/scripts/groot_link.sh" check >/dev/null 2>&1; then LINK_DOWN=1; fi
  bash "$WL/scripts/groot_link.sh" down >/dev/null 2>&1 || true
  say "GR00T off for the demo: link down, the pick uses the labelled SONIC arm-script fallback (--groot on keeps it)"
fi

# ---------------------------------------------------------------- --fresh: a new stack
viz_up() { curl -sf -m 3 "http://127.0.0.1:$VIZ_PORT/" >/dev/null; }
if [[ "$FRESH" == 1 ]]; then
  banner "fresh stack: $SCENE, profile full, viz low, page :$P5_PORT"
  t=$(date +%s)
  bash "$WL/scripts/m2_down.sh" --session "$SESSION" 2>&1 | tee -a "$RUN/stack.log" | tail -3
  if [[ "$KEEP_MEM" == 0 ]]; then
    mem=$WL/runs/memory/$SCENE.json
    if [[ -f "$mem" ]]; then
      mkdir -p "$WL/runs/memory_backup"
      mv "$mem" "$WL/runs/memory_backup/$SCENE-$TS.json"
      say "spatial memory moved to runs/memory_backup/$SCENE-$TS.json (the demo starts knowing nothing; --keep-memory keeps it)"
    fi
  fi
  if [[ "$GROOT" == on ]]; then
    bash "$WL/scripts/groot_link.sh" ensure 2>&1 | tee -a "$RUN/stack.log" | tail -2
  fi
  if ! bash "$WL/scripts/m2_up.sh" --profile full --scene "$SCENE" --viz low --p5-port "$P5_PORT" --session "$SESSION" \
       --groot "$([[ "$GROOT" == on ]] && echo link || echo off)" 2>&1 | tee -a "$RUN/stack.log" | grep -E 'READY|ERROR|stage times|m2_up.*session'; then :; fi
  grep -q 'READY' "$RUN/stack.log" || { say "m2_up.sh did not print READY (log $RUN/stack.log)"; exit 5; }
  say "stack up in $(( $(date +%s) - t )) s"
fi
if ! viz_up; then
  say "Sim Viewer (viz-server) not answering on :$VIZ_PORT: starting it"
  bash "$WL/viz/box.sh" server 0 | tee -a "$RUN/demo.log"
  for _ in $(seq 1 30); do viz_up && break; sleep 1; done
  viz_up || say "WARNING: the Sim Viewer did not come up (no recording)"
fi

# ---------------------------------------------------------------- page + System 1 ready, hands empty
banner "waiting for the page and System 1"
(cd "$WL" && "$PY_RT" -m tools.say --wait-ready "$READY_S" --json "$RUN/ready.json") 2>&1 | tee -a "$RUN/demo.log"
rc=${PIPESTATUS[0]}
(( rc == 0 )) || { say "the page or System 1 is not ready (tools.say rc $rc)"; exit 5; }
held=$("$PY_RT" -c 'import json,sys
h=(json.load(open(sys.argv[1])).get("end") or {}).get("hands") or {}
print(" ".join("%s:%s" % (a, v.get("holding")) for a, v in h.items() if v.get("holding") not in (None, "nothing", "", "UNKNOWN")))' "$RUN/ready.json")
[[ -n "$held" ]] && say "WARNING: the robot is holding something ($held): step 8 will fail (place does not work yet); use --fresh"

# ---------------------------------------------------------------- recording
REC_DIR=""
rec() { curl -s -m 120 -X POST -H 'Content-Type: application/json' -d "$1" "http://127.0.0.1:$VIZ_PORT/api/record"; }
if [[ "$RECORD" == 1 ]]; then
  out=$(rec "{\"action\":\"start\",\"label\":\"$LABEL\"}")
  REC_DIR=$(echo "$out" | "$PY_RT" -c 'import json,sys; print(json.load(sys.stdin).get("dir") or "")' 2>/dev/null)
  if [[ -n "$REC_DIR" ]]; then say "recording to $REC_DIR"; else say "WARNING: recording did not start: $out"; fi
fi

# ---------------------------------------------------------------- the steps
for n in "${STEPS[@]}"; do
  step_def "$n"
  banner "step $n: $NAME"
  ts=$(date +%s)
  (cd "$WL" && "$PY_RT" -m tools.say "${SAY_OPTS[@]}" --max-s "$MAXS" --json "$RUN/step$n.json" \
      "${EXTRA[@]}" "${LINES[@]}") 2>&1 | tee "$RUN/step$n.log"
  echo "$n ${PIPESTATUS[0]} $(( $(date +%s) - ts ))" >> "$RUN/steps.tsv"
  sleep "$BETWEEN_S"
done

# ---------------------------------------------------------------- stop the recording
if [[ -n "$REC_DIR" ]]; then
  say "stopping the recording (encoding the composite can take a minute)"
  rec '{"action":"stop"}' > "$RUN/record_stop.json"
  say "recording: $REC_DIR (composite.mp4, contact_sheet.png)"
  echo "$REC_DIR" > "$RUN/recording_dir"
fi

# ---------------------------------------------------------------- PASS/FAIL from the trace
banner "results"
# PASS/FAIL per step: the rules in config/demo_steps.yaml (tools/demo_steps.py RULES, shared with the UI's Demo panel)
(cd "$SRC" && "$PY_RT" -m tools.demo_steps results "$RUN" "$(( $(date +%s) - T0 ))") | tee "$RUN/results.md"
rc=${PIPESTATUS[0]}
say "run dir $RUN${REC_DIR:+ ; recording $REC_DIR}"
exit "$rc"
