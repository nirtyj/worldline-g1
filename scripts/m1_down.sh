#!/usr/bin/env bash
# Tear down the M1 stack started by scripts/m1_up.sh, in reverse order, without dropping the robot:
#   1. body stop           planner IDLE (a user stop never sends command{stop})
#   2. P1 band on          the elastic band takes the weight
#   3. deploy shutdown     body shutdown_control (IDLE, then command{stop}: the deploy damps and exits) and
#                          sonic/run_deploy.sh stop as a backstop ('o' on stdin, then SIGINT)
#   4. body                SIGINT in its window (the service exits; it never sends command{stop} on its own)
#   5. P1                  REP shutdown, then SIGINT in its window
#   6. tmux kill-session   only the session m1_up.sh created (exact-name targets '=NAME', never a prefix match)
#
#   scripts/m1_down.sh [--session NAME] [--port-offset N] [--fake]
set -uo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true
WL=${WL:-/work/worldline-g1}
PY_BODY=${PY_BODY:-$WL/.venv/bin/python}
SESSION=""; OFFSET=""; FAKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    --port-offset) OFFSET="$2"; shift 2;;
    --fake) FAKE=1; shift;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ -z "$SESSION" ]]; then SESSION=$([[ "$FAKE" == 1 ]] && echo body-m1fake || echo wl-m1); fi
RUN=$(readlink -f "$WL/outputs/m1/stack-latest-$SESSION" 2>/dev/null || true)
if [[ -z "$OFFSET" ]]; then
  OFFSET=$(cat "$RUN/port_offset" 2>/dev/null || echo 0)
fi
export WL_PORT_OFFSET=$OFFSET
say() { echo "[m1_down $(date +%H:%M:%S)] $*"; }
cli() { (cd "$WL" && timeout "${2:-20}" "$PY_BODY" -m tools.body_cli --port-offset "$OFFSET" $1); }
alive() { tmux list-windows -t "=$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"; }
win_pid_gone() {  # wait until the window's shell has no child process
  local w=$1 t=${2:-20} i pane
  pane=$(tmux list-panes -t "=$SESSION:$w" -F '#{pane_pid}' 2>/dev/null | head -1)
  [[ -z "$pane" ]] && return 0
  for ((i = 0; i < t * 4; i++)); do
    kill -0 "$pane" 2>/dev/null || return 0
    pgrep -P "$pane" >/dev/null 2>&1 || return 0
    sleep 0.25
  done
  return 1
}

if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
  say "no tmux session $SESSION: nothing to stop"
  exit 0
fi
say "session $SESSION, port offset $OFFSET"

# 1-3: park the robot on the band, then stop SONIC control
if alive body; then
  cli stop 10 >/dev/null 2>&1 && say "body: stopped (planner IDLE)" || say "body: stop failed (continuing)"
fi
cli "p1 band on=true" 10 >/dev/null 2>&1 && say "P1: band on" || say "P1: band on failed (continuing)"
sleep 1.0
if alive body && alive deploy; then
  cli "shutdown_control --confirm" 15 >/dev/null 2>&1 && say "deploy: command{stop} sent (band holds the robot)" \
    || say "deploy: shutdown_control failed (continuing)"
  sleep 1.5
fi
if alive deploy; then
  if [[ "$FAKE" == 1 ]]; then
    tmux send-keys -t "=$SESSION:deploy" C-c; win_pid_gone deploy 10 || true
  else
    bash "$WL/sonic/run_deploy.sh" stop --session "$SESSION" --window deploy || say "run_deploy.sh stop reported an error"
  fi
  tmux kill-window -t "=$SESSION:deploy" 2>/dev/null || true
  say "deploy stopped"
fi

# 4: body
for w in monitor body; do
  if alive "$w"; then
    tmux send-keys -t "=$SESSION:$w" C-c
    win_pid_gone "$w" 10 || say "$w did not exit on SIGINT"
    tmux kill-window -t "=$SESSION:$w" 2>/dev/null || true
  fi
done
say "body stopped"

# 5: P1
if alive isaac; then
  cli "p1 shutdown" 20 >/dev/null 2>&1 || true
  win_pid_gone isaac 30 || { tmux send-keys -t "=$SESSION:isaac" C-c; win_pid_gone isaac 30 || say "P1 still running after SIGINT"; }
  tmux kill-window -t "=$SESSION:isaac" 2>/dev/null || true
  say "P1 stopped"
fi

tmux kill-session -t "=$SESSION" 2>/dev/null || true
for base in 5556 5557 5565 5600 5601 5610 5611; do
  p=$((base + OFFSET))
  ss -ltn "sport = :$p" | grep -q LISTEN && say "WARNING: port $p still bound"
done
say "down"
