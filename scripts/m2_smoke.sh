#!/usr/bin/env bash
# Laptop side of the M2 smoke tests (docs/M2.md §7.4 step 6; docs/bringup.md): referee one F1 episode on the box's
# page through an SSH tunnel, record it with the viz recorder, and pull the recording and the stack's logs.
#
#   BREV_NAME=<box> scripts/m2_smoke.sh --label NAME [--session wl-m2] [--profile sonic] [--timeout 600]
#                                       [--out DIR] [--no-record]
#
# Needs a stack already up on the box (scripts/m2_up.sh), started with the planner and System 1 the smoke test calls
# for; the page's port and offset are read from the box's P5 state. Steps:
#   1. tunnel     laptop:<port> -> box:<port> (00_infra/tunnel.sh; the forward lives in the ssh ControlMaster, which
#                 a keepalive ssh holds open for the whole episode)
#   2. recorder   box: viz/recorder.py --control in tmux <session>-rec (head, chase/top, telemetry; docs/viz.md §6)
#   3. episode    laptop: python -m eval.offline_episode --url ws://127.0.0.1:<port>/ws --profile P (the F1 referee;
#                 it sends the scenario's reset, "Bring me the alarm clock.", and scores on world truth)
#   4. collect    box: recorder stop; pull the recording, the m2 and m1 run dirs and the P1/P2/P3/P5 logs to
#                 <out>/<label>/ (default out: outputs/m2b_wave1/ops)
# Exit status: the episode's (0 = every F1 step and check passed). A missing step is a result, not a script error.
# BOX_WL: the stack's code tree on the box (default /work/worldline-g1; the one m2_up.sh ran from, its WL).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
INFRA=${INFRA:-$ROOT/../ludo_robotics_prep_g1/00_infra}
LABEL=""; SESSION=wl-m2; PROFILE=sonic; TIMEOUT=600; OUT=$ROOT/outputs/m2b_wave1/ops; RECORD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --no-record) RECORD=0; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[[ -n "$LABEL" ]] || { echo "--label is required" >&2; exit 2; }
[[ -n "${BREV_NAME:-}" ]] || { echo "set BREV_NAME (ludo-g1-arena for the dev box, ludo-g1-brev2 for the main box)" >&2; exit 2; }
# shellcheck disable=SC1091
source "$INFRA/lib.sh"                      # ssh_opts, bssh, log, die (and set -euo pipefail)
set +e
WL=${BOX_WL:-/work/worldline-g1}
DEST=$OUT/$LABEL
mkdir -p "$DEST"
say() { echo "[m2_smoke $(date +%H:%M:%S)] $*"; }

# ---- the page's port and offset, from the box's P5 state
P5ENV=$(bssh "cat $WL/outputs/m2/p5-$SESSION.env 2>/dev/null") || die "no P5 state for session $SESSION on $BREV_NAME"
PORT=$(sed -n 's/^P5_PORT=//p' <<< "$P5ENV"); OFFSET=$(sed -n 's/^P5_OFFSET=//p' <<< "$P5ENV")
P5_LOG=$(sed -n 's/^P5_LOG=//p' <<< "$P5ENV")
[[ -n "$PORT" ]] || die "no P5_PORT in the box's P5 state"
bssh "cd $WL && .venv-rt/bin/python scripts/p5_probe.py --port $PORT --profile $PROFILE --timeout 10" > "$DEST/p5_before.json" \
  || die "the page on $BREV_NAME:$PORT is not ready: $(cat "$DEST/p5_before.json")"
say "page ready on $BREV_NAME:$PORT (offset $OFFSET): $(head -c 300 "$DEST/p5_before.json")"
{ echo "brev_name=$BREV_NAME"; echo "session=$SESSION"; echo "port=$PORT"; echo "offset=$OFFSET"; echo "profile=$PROFILE"; echo "box_wl=$WL"
  echo "laptop_git=$(git -C "$ROOT" rev-parse --short HEAD) dirty_files=$(git -C "$ROOT" status --porcelain | grep -vcE '^\?\? (\.venv-rt|outputs)$')"
  echo "$P5ENV"; } > "$DEST/smoke.env"

# ---- 1. tunnel (and a keepalive, so the ControlMaster that holds the forward outlives the episode)
bssh "sleep $((TIMEOUT + 900))" >/dev/null 2>&1 &
KEEP=$!
"$INFRA/tunnel.sh" "$PORT" >/dev/null 2>&1 &
TUN=$!
for _ in $(seq 1 40); do lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 0.5; done
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || { kill $KEEP $TUN 2>/dev/null; die "tunnel to $PORT did not come up"; }
"$ROOT/.venv-rt/bin/python" "$ROOT/scripts/p5_probe.py" --port "$PORT" --timeout 20 > "$DEST/p5_tunnel.json" \
  || say "WARNING: the page is not reachable through the tunnel: $(cat "$DEST/p5_tunnel.json")"
cleanup() {
  ssh "${ssh_opts[@]}" -O cancel -L "$PORT:localhost:$PORT" "$BREV_NAME" >/dev/null 2>&1 || true
  kill "$TUN" "$KEEP" 2>/dev/null || true
}
trap cleanup EXIT

# ---- 2. recorder
CTL=tcp://127.0.0.1:$((5630 + OFFSET))
if [[ "$RECORD" == 1 ]]; then
  bssh "tmux kill-session -t '=$SESSION-rec' 2>/dev/null; mkdir -p /work/logs/wl; tmux new-session -d -s '$SESSION-rec' \
    'cd $WL && viz/.venv/bin/python viz/recorder.py --control $CTL --port-offset $OFFSET --label $LABEL --duration $((TIMEOUT + 120)) 2>&1 | tee -a /work/logs/wl/$SESSION-rec-$LABEL.log'; sleep 3; \
    cd $WL && viz/.venv/bin/python viz/recorder.py ctl $CTL status" > "$DEST/recorder_start.json" 2>&1 \
    && say "recording (box tmux $SESSION-rec, control $CTL)" || say "WARNING: recorder did not start: $(head -c 300 "$DEST/recorder_start.json")"
fi

# ---- 3. the episode (the referee runs here, on the laptop)
say "episode: eval.offline_episode --url ws://127.0.0.1:$PORT/ws --profile $PROFILE --timeout $TIMEOUT"
T0=$SECONDS
(cd "$ROOT" && .venv-rt/bin/python -m eval.offline_episode --url "ws://127.0.0.1:$PORT/ws" --profile "$PROFILE" \
   --timeout "$TIMEOUT" --out "$DEST/episode") 2>&1 | tee "$DEST/episode.txt"
RC=${PIPESTATUS[0]}
say "episode finished in $((SECONDS - T0)) s wall, exit $RC"

# ---- 4. collect
if [[ "$RECORD" == 1 ]]; then
  bssh "cd $WL && viz/.venv/bin/python viz/recorder.py ctl $CTL stop" > "$DEST/recorder_stop.json" 2>&1
  REC_DIR=$("$ROOT/.venv-rt/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(((d.get("summary") or {}).get("dir")) or "")' \
            "$DEST/recorder_stop.json" 2>/dev/null)
  if [[ -n "$REC_DIR" ]]; then
    WL_ROOT=$ROOT "$INFRA/sync_wl.sh" pull "$REC_DIR/" "$DEST/recording" && say "recording -> $DEST/recording"
  else
    say "WARNING: no recording dir from the recorder: $(head -c 300 "$DEST/recorder_stop.json")"
  fi
  bssh "tmux kill-session -t '=$SESSION-rec' 2>/dev/null; true"
fi
M2RUN=$(bssh "readlink -f $WL/outputs/m2/stack-latest-$SESSION")
M1RUN=$(bssh "readlink -f $WL/outputs/m1/stack-latest-$SESSION")
[[ -n "$M2RUN" ]] && WL_ROOT=$ROOT "$INFRA/sync_wl.sh" pull "$M2RUN/" "$DEST/m2_run"
[[ -n "$M1RUN" ]] && WL_ROOT=$ROOT "$INFRA/sync_wl.sh" pull "$M1RUN/" "$DEST/m1_run"
LOGS=$(bssh "cat $M1RUN/isaac_log $M1RUN/deploy_log $M1RUN/body_log 2>/dev/null; echo $P5_LOG")
mkdir -p "$DEST/logs"
for f in $LOGS; do WL_ROOT=$ROOT "$INFRA/sync_wl.sh" pull "$f" "$DEST/logs"; done
say "evidence in $DEST (episode exit $RC)"
exit "$RC"
