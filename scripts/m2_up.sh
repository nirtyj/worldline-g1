#!/usr/bin/env bash
# Bring up the M2 live stack from cold (docs/M2.md §7.4, docs/bringup.md): the M1 stack through scripts/m1_up.sh,
# unchanged, then P5 (Worldline's page server) in its own tmux window of the same session.
#
#   scripts/m2_up.sh [--profile sonic|full] [--scene procthor-train-40] [--port-offset N] [--session wl-m2]
#                    [--viz off|min|low|high] [--p5-port P] [--planner MOD:FN] [--system1 MOD:FN|off]
#                    [--isaac-args "..."] [--no-groot-check] [--dry-run]
#
# Stages (each waits on a readiness probe; the time of each goes to <run>/stages.json):
#   1. P1 wl-isaac    m1_up.sh: REP ping + WL_ISAAC_READY, gt.pose >= 40 Hz, lowstate, camera; band on
#   2. P3 wl-body     m1_up.sh: body.state on 5611 (binds the SONIC input before the deploy connects)
#   3. P2 wl-sonic    m1_up.sh: deploy "Init Done" + robot_config
#   4. stand          m1_up.sh: SONIC takes the weight, band released, upright
#   5. P5 wl-runtime  scripts/m2_p5.sh start: ui.server --profile P --scene S, SimClock(1.0) (wall-second timeouts);
#                     ready = HTTP 200 + an `init` for that scene/profile with no error (scripts/p5_probe.py)
#   6. GR00T link     --profile full only: scripts/groot_link.sh ensure (the PolicyServer runs on the dev box,
#                     reached through an SSH tunnel, OD3, docs/groot_serving.md §5: ping it, and bring the tunnel up
#                     again with the last host if it is down). A failed check is a warning, because `full` then
#                     rejects GR00T calls as `policy unavailable` and keeps its labelled fallbacks.
# Defaults: session wl-m2, scene procthor-train-40 (H40, the F1 house), no viz cameras, the live planner and
# System 1 (scripts/m2_p5.sh). go_to stays A* + pure pursuit (NAV_BACKEND=astar; Nav2 is deferred, PLAN §0.8).
# --viz passes P1's VizCams level (docs/viz.md: min for recordings and demos, never high with SONIC in the loop).
# Idempotent: a session whose body and deploy already answer skips stages 1-4; a ready P5 is left alone. A half-up
# session is an error: run scripts/m2_down.sh first. Logs: /work/logs/wl/<session>-<ts>-*.log.
# Env passed through to m1_up.sh: DEPLOY_TASKSET, ISAAC_TASKSET, BODY_TASKSET, BODY_ARGS, P1_TIMEOUT, DEPLOY_WAIT_S.
set -euo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true

WL=${WL:-/work/worldline-g1}
PY_RT=${PY_RT:-$WL/.venv-rt/bin/python}
PY_BODY=${PY_BODY:-$WL/.venv/bin/python}
LOGD=${LOGD:-/work/logs/wl}
M2_STATE=${M2_STATE:-$WL/outputs/m2}
PROFILE=sonic; SCENE=procthor-train-40; OFFSET=0; SESSION=wl-m2; VIZ=off; P5_PORT=""; PLANNER=""; SYSTEM1=""
ISAAC_ARGS=${ISAAC_ARGS:-}; GROOT_CHECK=1; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --port-offset) OFFSET="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    --viz) VIZ="$2"; shift 2;;
    --p5-port) P5_PORT="$2"; shift 2;;
    --planner) PLANNER="$2"; shift 2;;
    --system1) SYSTEM1="$2"; shift 2;;
    --isaac-args) ISAAC_ARGS="$2"; shift 2;;
    --no-groot-check) GROOT_CHECK=0; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,29p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
case "$PROFILE" in sonic|full) ;; *) echo "--profile must be sonic or full (lite/bringup need no stack)" >&2; exit 2;; esac
case "$VIZ" in off|min|low|high) ;; *) echo "--viz must be off, min, low or high" >&2; exit 2;; esac
[[ "$OFFSET" =~ ^[0-9]+$ ]] || { echo "--port-offset must be a number" >&2; exit 2; }
[[ "$VIZ" != off ]] && ISAAC_ARGS="${ISAAC_ARGS:+$ISAAC_ARGS }--viz $VIZ"
P5_PORT=${P5_PORT:-$((8765 + OFFSET))}
# one tree and one state dir for every script this calls (m1_up.sh, m2_p5.sh, m1_down.sh read these from the env)
export WL PY_RT PY_BODY LOGD M2_STATE NAV_BACKEND=astar WL_PORT_OFFSET=$OFFSET ISAAC_ARGS
P5_ARGS=(--session "$SESSION" --port-offset "$OFFSET" --port "$P5_PORT" --profile "$PROFILE" --scene "$SCENE"
         ${PLANNER:+--planner "$PLANNER"} ${SYSTEM1:+--system1 "$SYSTEM1"})

say() { echo "[m2_up $(date +%H:%M:%S)] $*"; }
die() { [[ -n "${RUN:-}" && -f "$RUN/stages.tsv" ]] && stage_times >&2 || true
      echo "[m2_up] ERROR: $*" >&2; echo "[m2_up] logs: $LOGD/$SESSION-* ; tear down with scripts/m2_down.sh --session $SESSION" >&2; exit 1; }
now() { echo "${EPOCHREALTIME:-$(date +%s)}"; }

# The house P1 loads is the scene's (config/stack.yaml `scenes:`; "<house>@<variant>" -> <house>), resolved by the
# runtime's own code so P1 and P5 cannot disagree. This also proves the runtime venv imports before Isaac starts.
house_of() {
  (cd "$WL" && "$PY_RT" -c 'import sys
from robot.factory import parse_scene
from robot.profile import load_profile
print(parse_scene(sys.argv[1], load_profile(sys.argv[2]))[0])' "$SCENE" "$PROFILE")
}

if [[ "$DRY" == 1 ]]; then
  HOUSE=$(house_of 2>/dev/null || echo "${SCENE%%@*}")
  echo "session=$SESSION profile=$PROFILE scene=$SCENE house=$HOUSE port_offset=$OFFSET nav_backend=$NAV_BACKEND"
  echo "tree: WL=$WL (child scripts see WL=$(bash -c 'echo $WL') M2_STATE=$(bash -c 'echo $M2_STATE'))"
  echo "m1_up: bash $WL/scripts/m1_up.sh --house $HOUSE --port-offset $OFFSET --session $SESSION (ISAAC_ARGS='$ISAAC_ARGS')"
  echo "p5: bash $WL/scripts/m2_p5.sh start ${P5_ARGS[*]}"
  echo "page: http://127.0.0.1:$P5_PORT (laptop: BREV_NAME=<box> 00_infra/tunnel.sh $P5_PORT)"
  if [[ "$PROFILE" == full && "$GROOT_CHECK" == 1 ]]; then
    if [[ -f "$WL/scripts/groot_link.sh" ]]; then echo "groot: bash $WL/scripts/groot_link.sh ensure"
    else echo "groot: skipped (scripts/groot_link.sh not present; owner groot_srv)"; fi
  fi
  exit 0
fi

TS=$(date +%Y%m%d-%H%M%S)
RUN=$M2_STATE/stack-$TS
mkdir -p "$LOGD" "$RUN"
LOG_UP=$RUN/m2_up.log
ln -sfn "$RUN" "$M2_STATE/stack-latest-$SESSION"
printf 'session=%s\nprofile=%s\nscene=%s\nport_offset=%s\np5_port=%s\nviz=%s\n' \
  "$SESSION" "$PROFILE" "$SCENE" "$OFFSET" "$P5_PORT" "$VIZ" > "$RUN/config.env"
stage() { printf '%s %s\n' "$1" "$(now)" >> "$RUN/stages.tsv"; }   # stage name, epoch seconds
stage_times() {  # <run>/stages.json from stages.tsv (also on failure, so a slow or failed stage shows up)
  "$PY_RT" - "$RUN" <<'EOF2'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
rows = [l.split() for l in (run / "stages.tsv").read_text().splitlines() if l.strip()]
t = {k: float(v) for k, v in rows}
order = ["start", "p1", "body", "deploy", "stand", "p5", "groot", "ready"]
seen = [k for k in order if k in t]
out = {"t_epoch": t, "since_start_s": {k: round(t[k] - t["start"], 1) for k in seen},
       "stage_s": {b: round(t[b] - t[a], 1) for a, b in zip(seen, seen[1:])},
       "total_s": round(t["ready"] - t["start"], 1) if "ready" in t else None}
(run / "stages.json").write_text(json.dumps(out, indent=1) + "\n")
print("stage times (s):", " ".join(f"{k}={v}" for k, v in out["stage_s"].items()), "| total", out["total_s"])
EOF2
}
stage start

# ---- preflight (before anything starts)
[[ -x "$PY_RT" ]] || die "runtime venv missing: $PY_RT (bash scripts/m2_venv.sh)"
[[ -x "$PY_BODY" ]] || die "body venv missing: $PY_BODY"
HOUSE=$(house_of) || die "the runtime venv cannot load the profile/scene ($PY_RT; bash scripts/m2_venv.sh)"
echo "house=$HOUSE" >> "$RUN/config.env"
say "session $SESSION: profile $PROFILE, scene $SCENE (house $HOUSE), port offset $OFFSET, page port $P5_PORT, viz $VIZ"

# ---- stages 1-4: the M1 stack (skipped when this session already has one standing)
wr() { (cd "$WL" && "$PY_BODY" -m tools.wait_ready "$@" --port-offset "$OFFSET"); }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  if wr body --timeout 5 >/dev/null && wr control --timeout 5 >/dev/null; then
    say "M1 stack already up in session $SESSION (body answers, deploy in CONTROL): stages 1-4 skipped"
    echo "m1=reused" >> "$RUN/config.env"
  else
    die "tmux session $SESSION exists but its body/deploy do not answer: run scripts/m2_down.sh --session $SESSION first"
  fi
else
  # m1_up.sh's own lines, each stamped with the epoch time, so the stage times below come from its probes
  set +e
  bash "$WL/scripts/m1_up.sh" --house "$HOUSE" --port-offset "$OFFSET" --session "$SESSION" 2>&1 \
    | while IFS= read -r line; do printf '%s %s\n' "$(now)" "$line"; done | tee -a "$LOG_UP"
  rc=${PIPESTATUS[0]}
  set -e
  # each M1 stage's time is the stamp on the m1_up line that reports it
  awk '/\] P1 up, band on/{print "p1", $1} /\] body up/{print "body", $1} /\] deploy up \(Init Done\)/{print "deploy", $1}
       /\] STANDING under SONIC control/{print "stand", $1}' "$LOG_UP" >> "$RUN/stages.tsv"
  (( rc == 0 )) || die "m1_up.sh failed (rc $rc; its log: $LOG_UP)"
  ln -sfn "$(readlink -f "$WL/outputs/m1/stack-latest-$SESSION")" "$RUN/m1"
  echo "m1=started" >> "$RUN/config.env"
fi

# ---- stage 5: P5
set +e
bash "$WL/scripts/m2_p5.sh" start "${P5_ARGS[@]}" 2>&1 | tee -a "$LOG_UP"
rc=${PIPESTATUS[0]}
set -e
(( rc == 0 )) || die "P5 did not come up (scripts/m2_p5.sh rc $rc)"
stage p5
cp "$M2_STATE/p5-$SESSION.ready.json" "$RUN/p5_ready.json" 2>/dev/null || true
cp "$M2_STATE/p5-$SESSION.env" "$RUN/p5.env" 2>/dev/null || true

# ---- stage 6: GR00T link (full only)
GROOT=n/a
if [[ "$PROFILE" == full && "$GROOT_CHECK" == 1 ]]; then
  if [[ -f "$WL/scripts/groot_link.sh" ]]; then
    if bash "$WL/scripts/groot_link.sh" ensure 2>&1 | tee -a "$LOG_UP"; then GROOT=ok; stage groot
    else GROOT=failed; say "WARNING: GR00T link check failed: full rejects GR00T calls (policy unavailable) and uses its fallbacks"; fi
  else
    GROOT=skipped; say "GR00T link check skipped: scripts/groot_link.sh is not in this tree (owner groot_srv)"
  fi
fi
echo "groot_link=$GROOT" >> "$RUN/config.env"
stage ready

stage_times
say "READY: page http://127.0.0.1:$P5_PORT (laptop: BREV_NAME=<box> 00_infra/tunnel.sh $P5_PORT) ; tmux attach -t $SESSION ; run dir $RUN"
