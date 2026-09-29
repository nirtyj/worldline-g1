#!/usr/bin/env bash
# Restart ONLY the body service (P3) of a running M1 stack (development / tuning), then re-adopt SONIC control.
#
#   scripts/m1_restart_body.sh [--session wl-m1] [--no-stand] [-- extra body.service args]
#
# Safe while the robot stands: while P3 is down the deploy hears no planner message for > 1 s and forces planner IDLE
# (zmq_manager.hpp:581-642), so the robot keeps standing. The new body binds 5556 again (the deploy's SUB reconnects),
# and `stand` re-sends command{start}, which the deploy ignores while it is already started
# (zmq_manager.hpp:526 `if (start_control_ && !operator_state.start)`); the planner frame is re-read from g1_debug
# (init_base_quat is unchanged), and the band release is a no-op.
set -euo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true
WL=${WL:-/work/worldline-g1}
PY_BODY=${PY_BODY:-$WL/.venv/bin/python}
SESSION=wl-m1; STAND=1; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    --no-stand) STAND=0; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
RUN=$(readlink -f "$WL/outputs/m1/stack-latest-$SESSION")
OFFSET=$(cat "$RUN/port_offset" 2>/dev/null || echo 0)
NAV_BACKEND=${NAV_BACKEND:-astar}
tmux has-session -t "=$SESSION" 2>/dev/null || { echo "no session $SESSION" >&2; exit 1; }
pane=$(tmux list-panes -t "=$SESSION:body" -F '#{pane_pid}' | head -1)
tmux send-keys -t "=$SESSION:body" C-c
for i in $(seq 1 40); do pgrep -P "$pane" >/dev/null 2>&1 || break; sleep 0.25; done
pgrep -P "$pane" >/dev/null 2>&1 && { echo "body did not exit" >&2; exit 1; }
ts=$(date +%H%M%S)
ENV_SRC="source /etc/profile.d/ludo.sh 2>/dev/null; export WL_PORT_OFFSET=$OFFSET NAV_BACKEND=$NAV_BACKEND;"
tmux send-keys -t "=$SESSION:body" "$ENV_SRC cd $WL && exec ${BODY_TASKSET-taskset -c 4-15 }$PY_BODY -u -m body.service --port-offset $OFFSET --log-dir $RUN/body-$ts ${EXTRA[*]:-} 2>&1 | tee -a /work/logs/wl/$SESSION-body-restart-$ts.log" C-m
(cd "$WL" && "$PY_BODY" -m tools.wait_ready body --timeout 30 --port-offset "$OFFSET") >/dev/null
echo "[m1_restart_body] body restarted (log dir $RUN/body-$ts)"
if [[ "$STAND" == 1 ]]; then
  (cd "$WL" && "$PY_BODY" -m tools.body_cli --port-offset "$OFFSET" --timeout 60 stand) | head -c 300; echo
fi
