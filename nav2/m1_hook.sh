#!/usr/bin/env bash
# Hook for scripts/m1_up.sh / scripts/m1_down.sh (for the integrate agent to adopt; this file does not edit them).
#
#   bash nav2/m1_hook.sh up   <port_offset> <m1_session>   # start the ROS side if NAV_BACKEND=nav2 (default)
#   bash nav2/m1_hook.sh down <port_offset> <m1_session>   # stop it (no-op if it is not running)
#
# Lines to add:
#   m1_up.sh, right after "body up" (P1 must already serve gt.pose + get_occupancy; start Nav2 BEFORE the wall-clock
#   deploy so its start-up burst of ~40% of one core for ~1 s cannot disturb SONIC):
#       bash "$WL/nav2/m1_hook.sh" up "$OFFSET" "$SESSION" || say "WARNING: Nav2 not up; go_to falls back to A*"
#   m1_down.sh, before the body stop (cancels an active Nav2 goal first):
#       bash "$WL/nav2/m1_hook.sh" down "$OFFSET" "$SESSION" || true
# The body needs no flag: NAV_BACKEND defaults to nav2 in wl-body and falls back to A* (result.backend = "astar",
# result.backend_fallback = why) whenever the bridge does not answer or Nav2 is not active.
# NAV_BACKEND=astar skips Nav2 entirely (and makes wl-body use A* without asking the bridge).
# Env passed through: NAV2_CONTROLLER (rpp|mppi), NAV2_TASKSET (CPU pinning), NAV2_NICE (default 5).
set -uo pipefail
WL=${WL:-/work/worldline-g1}
CMD=${1:?up|down}; OFFSET=${2:-0}; M1=${3:-wl-m1}
SESSION="${M1}-nav2"; [[ "$M1" == wl-m1 ]] && SESSION=wl-nav2     # the integrated stack: tmux wl-m1 + wl-nav2
case "$CMD" in
  up)
    if [[ "${NAV_BACKEND:-nav2}" != nav2 ]]; then echo "[nav2/m1_hook] NAV_BACKEND=${NAV_BACKEND} -> Nav2 not started"; exit 0; fi
    RUN=$(readlink -f "$WL/outputs/m1/stack-latest-$M1" 2>/dev/null || true)   # m1_up.sh's run dir, if any
    LOGDIR=""; [[ -n "$RUN" && -d "$RUN" ]] && LOGDIR="$RUN/nav2"
    exec bash "$WL/nav2/up.sh" --port-offset "$OFFSET" --session "$SESSION" ${LOGDIR:+--log-dir "$LOGDIR"} --wait "${NAV2_WAIT:-120}"
    ;;
  down)
    exec bash "$WL/nav2/down.sh" --session "$SESSION"
    ;;
  *) echo "usage: $0 up|down <port_offset> <m1_session>" >&2; exit 2;;
esac
