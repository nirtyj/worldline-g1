#!/usr/bin/env bash
# P5, Worldline's page server (`python -m ui.server`), as the `p5` window of a stack's tmux session.
# scripts/m2_up.sh and scripts/m2_down.sh call it; on its own it swaps the planner or System 1 on a running stack.
#
#   scripts/m2_p5.sh start   [--session wl-m2] [--port-offset N] [--port P] [--profile sonic] [--scene procthor-train-40]
#                            [--planner MOD:FN] [--system1 MOD:FN|off] [--timeout S]
#   scripts/m2_p5.sh stop    [--session wl-m2]    SIGINT: the server stops its session (runtime saves memory, the robot
#                                                  facade cancels its body ops), then the window closes
#   scripts/m2_p5.sh restart [start options]      stop, then start; options not given keep the last start's values
#   scripts/m2_p5.sh status  [--session wl-m2]    one JSON line from scripts/p5_probe.py; exit 0 when the page is ready
#
# Defaults: port 8765 + offset, the live planner (agent.model:create_brain, Gemini) and System 1
# brains.system1_jev:create (Jev + Gemini Live). The offline smoke test (docs/M2.md §7.4 step 6.1) uses
# --planner brains.scripted:create --system1 tests.kept.system1_stub:create. The page runs SimClock(1.0) (ui.server's
# default --speed), so every runtime timeout is in wall seconds. Keys come from ~/.config/ludo-g1/secrets.env
# (00_infra/secrets.sh), which ui.server reads itself; start refuses to launch a live planner or System 1 without them.
# Ready = HTTP 200 on / and an `init` for that scene and profile with no error on /ws (scripts/p5_probe.py).
# CPU pinning: P5_TASKSET (default 4-15 on Linux, like the body; "" disables): never the SONIC deploy's 0-3, and that
# includes the GR00T client of `full` (frame decode, the 0.92 MB request) that runs in P5. P5_THREADS (default 2) caps
# the CPU thread pools of numpy/BLAS in P5 (OMP/OpenBLAS/MKL; "" = no cap). Logs: $LOGD/<session>-<ts>-p5.log.
# The last start's options live in $M2_STATE/p5-<session>.env.
set -euo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true

WL=${WL:-/work/worldline-g1}
PY_RT=${PY_RT:-$WL/.venv-rt/bin/python}
LOGD=${LOGD:-/work/logs/wl}
M2_STATE=${M2_STATE:-$WL/outputs/m2}
SECRETS=${SECRETS:-$HOME/.config/ludo-g1/secrets.env}
LIVE_PLANNER=agent.model:create_brain
LIVE_SYSTEM1=brains.system1_jev:create

cmd=${1:-status}; shift || true
[[ "$cmd" == -h || "$cmd" == --help ]] && { sed -n '2,21p' "$0"; exit 0; }
SESSION=wl-m2; OFFSET=""; PORT=""; PROFILE=""; SCENE=""; PLANNER=""; SYSTEM1=""; TIMEOUT=${P5_TIMEOUT:-300}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    --port-offset) OFFSET="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --planner) PLANNER="$2"; shift 2;;
    --system1) SYSTEM1="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    -h|--help) sed -n '2,21p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
STATE=$M2_STATE/p5-$SESSION.env
# options not given on the command line: the last start's, then the defaults
if [[ -f "$STATE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE"
  OFFSET=${OFFSET:-${P5_OFFSET:-}}; PORT=${PORT:-${P5_PORT:-}}; PROFILE=${PROFILE:-${P5_PROFILE:-}}
  SCENE=${SCENE:-${P5_SCENE:-}}; PLANNER=${PLANNER:-${P5_PLANNER:-}}; SYSTEM1=${SYSTEM1:-${P5_SYSTEM1:-}}
fi
OFFSET=${OFFSET:-${WL_PORT_OFFSET:-0}}; PORT=${PORT:-$((8765 + OFFSET))}; PROFILE=${PROFILE:-sonic}
SCENE=${SCENE:-procthor-train-40}; PLANNER=${PLANNER:-$LIVE_PLANNER}; SYSTEM1=${SYSTEM1:-$LIVE_SYSTEM1}
if [[ "$(uname -s)" == Linux ]]; then P5_TASKSET=${P5_TASKSET-4-15}; else P5_TASKSET=""; fi
P5_THREADS=${P5_THREADS-2}

say() { echo "[m2_p5 $(date +%H:%M:%S)] $*"; }
die() { echo "[m2_p5] ERROR: $*" >&2; exit 1; }
has_window() { tmux list-windows -t "=$SESSION" -F '#W' 2>/dev/null | grep -qx p5; }
pane_pid() { tmux list-panes -t "=$SESSION:p5" -F '#{pane_pid}' 2>/dev/null | head -1; }
pane_busy() {  # the window's shell still has a child (the server or its tee)
  local p; p=$(pane_pid); [[ -n "$p" ]] && pgrep -P "$p" >/dev/null 2>&1
}
port_open() { "$([[ -x "$PY_RT" ]] && echo "$PY_RT" || echo python3)" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)"; }
probe() { (cd "$WL" && "$PY_RT" scripts/p5_probe.py --port "$PORT" "$@"); }
has_key() {  # a key is set in the environment or the secrets file (the value is never printed)
  [[ -n "${!1:-}" ]] || grep -qE "^(export )?$1=.+" "$SECRETS" 2>/dev/null
}

start() {
  [[ -x "$PY_RT" ]] || die "runtime venv missing: $PY_RT (bash scripts/m2_venv.sh)"
  if has_window; then
    if probe --scene "$SCENE" --profile "$PROFILE" --timeout "$TIMEOUT" > "$M2_STATE/p5-$SESSION.ready.json"; then
      say "P5 already up on 127.0.0.1:$PORT (session $SESSION); left as is"
      return 0
    fi
    die "window $SESSION:p5 exists but the page is not ready (scripts/m2_p5.sh restart, or look at the log in $STATE)"
  fi
  port_open && die "port $PORT is already in use by another process"
  if [[ "$PLANNER" == "$LIVE_PLANNER" ]]; then
    has_key GEMINI_API_KEY || die "the live planner needs GEMINI_API_KEY ($SECRETS; 00_infra/secrets.sh)"
  fi
  if [[ "$SYSTEM1" == "$LIVE_SYSTEM1" ]]; then
    has_key TYPESAFE_API_KEY && has_key GEMINI_API_KEY \
      || die "System 1 $SYSTEM1 needs TYPESAFE_API_KEY and GEMINI_API_KEY ($SECRETS; 00_infra/secrets.sh)"
  fi
  local ts log env_src p5
  mkdir -p "$M2_STATE" "$LOGD"
  ts=$(date +%Y%m%d-%H%M%S)
  log=$LOGD/$SESSION-$ts-p5.log
  cat > "$STATE" <<EOF
P5_OFFSET=$OFFSET
P5_PORT=$PORT
P5_PROFILE=$PROFILE
P5_SCENE=$SCENE
P5_PLANNER=$PLANNER
P5_SYSTEM1=$SYSTEM1
P5_LOG=$log
EOF
  # like m1_up.sh: the window sources the box env itself (a running tmux server passes its own env, not ours)
  env_src="source /etc/profile.d/ludo.sh 2>/dev/null; export WL_PORT_OFFSET=$OFFSET PYTHONUNBUFFERED=1${WORLDLINE_RUNS:+ WORLDLINE_RUNS=$WORLDLINE_RUNS}${P5_THREADS:+ OMP_NUM_THREADS=$P5_THREADS OPENBLAS_NUM_THREADS=$P5_THREADS MKL_NUM_THREADS=$P5_THREADS};"
  # tee -i: Ctrl-C reaches only the server, so its shutdown lines (session stop, "stopped") still reach the log
  p5="$env_src cd $WL && exec ${P5_TASKSET:+taskset -c $P5_TASKSET }$PY_RT -u -m ui.server --host 127.0.0.1 --port $PORT --port-offset $OFFSET --profile $PROFILE --scene $SCENE --planner $PLANNER --system1 $SYSTEM1 2>&1 | tee -i -a $log"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux new-window -d -t "=$SESSION:" -n p5 "bash --noprofile --norc"
  else
    tmux new-session -d -s "$SESSION" -n p5 -x 220 -y 50 "bash --noprofile --norc"
  fi
  tmux send-keys -t "=$SESSION:p5" "$p5" C-m
  say "P5 starting: ui.server --profile $PROFILE --scene $SCENE --planner $PLANNER --system1 $SYSTEM1 on 127.0.0.1:$PORT (log $log)"
  local t0=$SECONDS
  while (( SECONDS - t0 < TIMEOUT )); do
    if probe --scene "$SCENE" --profile "$PROFILE" --timeout 5 > "$M2_STATE/p5-$SESSION.ready.json" 2>/dev/null; then
      say "P5 ready in $((SECONDS - t0)) s: $(head -c 400 "$M2_STATE/p5-$SESSION.ready.json")"
      return 0
    fi
    if (( SECONDS - t0 > 5 )) && ! pane_busy; then
      tail -n 30 "$log" >&2 || true
      die "P5 exited during start-up (log $log)"
    fi
    sleep 2
  done
  tail -n 30 "$log" >&2 || true
  die "P5 not ready within $TIMEOUT s: $(cat "$M2_STATE/p5-$SESSION.ready.json" 2>/dev/null | head -c 400)"
}

stop() {
  if ! has_window; then
    say "no P5 window in session $SESSION: nothing to stop"
    return 0
  fi
  local i
  tmux send-keys -t "=$SESSION:p5" C-c
  for i in $(seq 1 80); do pane_busy || break; sleep 0.25; done     # the server gives the session 8 s to stop
  if pane_busy; then
    say "P5 still running 20 s after SIGINT; sending SIGINT again"
    tmux send-keys -t "=$SESSION:p5" C-c
    for i in $(seq 1 40); do pane_busy || break; sleep 0.25; done
  fi
  pane_busy && say "WARNING: P5 did not exit; killing its window"
  tmux kill-window -t "=$SESSION:p5" 2>/dev/null || true
  for i in $(seq 1 20); do port_open || break; sleep 0.25; done
  port_open && say "WARNING: port $PORT still open after P5 stopped" || say "P5 stopped (port $PORT closed)"
}

case "$cmd" in
  start) start;;
  stop) stop;;
  restart) stop; start;;
  status) probe --scene "$SCENE" --profile "$PROFILE" --timeout 5;;
  *) echo "usage: scripts/m2_p5.sh start|stop|restart|status [options]" >&2; exit 2;;
esac
