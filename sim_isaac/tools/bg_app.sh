#!/usr/bin/env bash
# Start wl-isaac in a tmux session (isaac-<name>) after stopping any previous isaac-* app session of this
# component and waiting for its ports to free up. Blocks until WL_ISAAC_READY (or failure).
#   bash sim_isaac/tools/bg_app.sh <name> [app args...]      log: /work/logs/wl/isaac-<name>.log
set -uo pipefail
name=$1; shift
here="$(cd "$(dirname "$0")" && pwd)"
log=/work/logs/wl/isaac-$name.log
mkdir -p /work/logs/wl
for s in $(tmux ls -F '#{session_name}' 2>/dev/null | grep '^isaac-app-' || true); do
  tmux send-keys -t "$s" C-c 2>/dev/null; sleep 2; tmux kill-session -t "$s" 2>/dev/null || true
done
off=0; prev=""
for x in "$@"; do [[ "$prev" == "--port-offset" ]] && off=$x; prev=$x; done
rep=$((5600 + off))
for _ in $(seq 1 90); do
  ss -ltn 2>/dev/null | grep -q ":$rep " || break; sleep 1
done
ss -ltn 2>/dev/null | grep -q ":$rep " && { echo "port $rep still busy (someone else's app?)"; exit 3; }
rm -f "$log"
tmux new -d -s "isaac-app-$name" "bash $here/run_app.sh $* > $log 2>&1; echo EXIT=\$? >> $log"
timeout 600 bash -c "until grep -q -E 'WL_ISAAC_READY|EXIT=' $log; do sleep 2; done"
grep -E "WL_ISAAC_READY|Traceback|Error:|EXIT=" "$log" | cut -c1-400 | head -5
