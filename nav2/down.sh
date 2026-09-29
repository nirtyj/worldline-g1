#!/usr/bin/env bash
# Stop the ROS navigation side started by nav2/up.sh: SIGINT the bridge (cancels an active goal) and Nav2, then kill
# ONLY that tmux session.   nav2/down.sh [--session wl-nav2]
set -uo pipefail
SESSION=wl-nav2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if ! tmux has-session -t "=$SESSION" 2>/dev/null; then echo "[nav2/down] no session $SESSION"; exit 0; fi
for w in bridge nav2; do tmux send-keys -t "=$SESSION:$w" C-c 2>/dev/null || true; done
for i in $(seq 1 20); do          # wait (<= 10 s) until the panes' children (bridge, ros2 launch) have exited
  alive=0
  for p in $(tmux list-panes -s -t "=$SESSION" -F '#{pane_pid}' 2>/dev/null); do
    pgrep -P "$p" >/dev/null && alive=1
  done
  [[ $alive == 0 ]] && break
  sleep 0.5
done
tmux kill-session -t "=$SESSION" 2>/dev/null || true
echo "[nav2/down] session $SESSION stopped"
