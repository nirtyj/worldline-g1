#!/usr/bin/env bash
# Tear down the M2 stack started by scripts/m2_up.sh: P5 first, the body left holding, then the M1 stack.
#
#   scripts/m2_down.sh [--session wl-m2] [--port-offset N] [--p5-only]
#
#   1. P5      scripts/m2_p5.sh stop: SIGINT, so the page server stops its session (the runtime saves memory; the robot
#              facade cancels its running body ops and closes its client). Nothing new reaches the body after this.
#   2. HOLD    body `stop`: planner IDLE, the robot stands under SONIC (never command{stop}). M1's body has no named
#              HOLD mode yet (B.3); standing IDLE with no op running is its HOLD.
#      --p5-only stops here: the M1 stack keeps standing, ready for scripts/m2_p5.sh start or scripts/m2_up.sh.
#   3. M1      scripts/m1_down.sh --session S: band on -> deploy shutdown -> body -> P1 -> kill only that session.
# Idempotent: a session that is not running is reported and left alone (exit 0).
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true
WL=${WL:-/work/worldline-g1}
PY_BODY=${PY_BODY:-$WL/.venv/bin/python}
M2_STATE=${M2_STATE:-$WL/outputs/m2}
export WL PY_BODY M2_STATE                  # m2_p5.sh and m1_down.sh act on the same tree and state
SESSION=wl-m2; OFFSET=""; P5_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    --port-offset) OFFSET="$2"; shift 2;;
    --p5-only) P5_ONLY=1; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ -z "$OFFSET" ]]; then
  OFFSET=$(sed -n 's/^port_offset=//p' "$M2_STATE/stack-latest-$SESSION/config.env" 2>/dev/null | head -1)
  OFFSET=${OFFSET:-$(cat "$WL/outputs/m1/stack-latest-$SESSION/port_offset" 2>/dev/null || echo 0)}
fi
export WL_PORT_OFFSET=$OFFSET
say() { echo "[m2_down $(date +%H:%M:%S)] $*"; }
alive() { tmux list-windows -t "=$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"; }

if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
  say "no tmux session $SESSION: nothing to stop"
  exit 0
fi
say "session $SESSION, port offset $OFFSET"

# 1. P5
bash "$WL/scripts/m2_p5.sh" stop --session "$SESSION" || say "P5 stop reported an error (continuing)"

# 2. the body holds: planner IDLE, standing
if alive body; then
  if (cd "$WL" && timeout 10 "$PY_BODY" -m tools.body_cli --port-offset "$OFFSET" stop >/dev/null 2>&1); then
    say "body: stop (planner IDLE, standing under SONIC)"
  else
    say "body: stop failed (continuing)"
  fi
fi
if [[ "$P5_ONLY" == 1 ]]; then
  say "P5 stopped; the M1 stack in session $SESSION keeps standing"
  exit 0
fi

# 3. the M1 stack (Nav2 is never started by m2_up.sh; m1_down.sh skips it)
bash "$WL/scripts/m1_down.sh" --session "$SESSION" --port-offset "$OFFSET"
say "down"
