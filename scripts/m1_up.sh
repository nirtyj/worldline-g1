#!/usr/bin/env bash
# Bring up the M1 stack from cold in tmux session wl-m1 (windows: isaac, deploy, body, monitor).
#
#   scripts/m1_up.sh [--house ID] [--port-offset N] [--session NAME] [--fake] [--no-stand] [--isaac-args "..."]
#
# Order (docs/contracts/m1.md, docs/contracts/sonic_deploy.md §3 "Start sequence"):
#   1. P1 wl-isaac   python -m sim_isaac.app ... ; wait: REP ping + "WL_ISAAC_READY", gt.pose >= 40 Hz,
#                    lowstate publishing (get_stats.lowstate_pub_hz), camera frame; band ON (P1 starts with it on).
#   2. P3 wl-body    python -m body.service : binds the SONIC input PUB (5556) BEFORE the deploy connects
#                    (slow joiner) and streams planner IDLE keepalive; wait: body.state on 5611.
#   3. P2 wl-sonic   sonic/run_deploy.sh start --session <this> --window deploy --wait-init ; wait: "Init Done"
#                    in the deploy log (INIT ramp finished, g1_deploy_onnx_ref.cpp:2787-2790) and robot_config on 5557.
#   4. stand         tools/body_cli stand: command{start,planner} -> wait for g1_debug with init_base_quat (CONTROL)
#                    -> 2 s IDLE hold -> band release (ramp 1 s) -> 3 s upright check. On failure the band stays on.
#   5. monitor       tools/m1_monitor (1 Hz status line).
# --fake runs tools/fake_p1 + tools/fake_deploy instead of Isaac + the deploy (default offset 200, session
# body-m1fake) so the script itself can be tested without a GPU.
# NOTE: the deploy's DDS domain is hard-coded to 0 (g1_deploy_onnx_ref.cpp:2218), so the real stack always uses
# DDS domain 0 on lo whatever --port-offset is; only the ZMQ ports move.
# CPU pinning (optional env): DEPLOY_TASKSET (e.g. 0-3: the deploy pins its busy-spinning main thread to CPU 0 itself,
# sonic_deploy.md §5), ISAAC_TASKSET (e.g. 4-15), BODY_TASKSET. CPUs 2k/2k+1 are hyperthread siblings on the box.
# The wall-clock deploy is sensitive to CPU contention: the deploy agent's MuJoCo reference passed on a quiet box and
# fell repeatedly while other Isaac jobs loaded all cores (outputs/m1/deploy/mujoco-ref-*). Run M1 on a quiet box.
set -euo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true

WL=${WL:-/work/worldline-g1}
PY_ISAAC=${PY_ISAAC:-/work/envs/isaaclab/bin/python}
PY_BODY=${PY_BODY:-$WL/.venv/bin/python}
HOUSE=${HOUSE:-procthor-10k-train-40}
OFFSET=0; SESSION=""; FAKE=0; STAND=1; ISAAC_ARGS=${ISAAC_ARGS:-}; P1_TIMEOUT=${P1_TIMEOUT:-1500}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --house) HOUSE="$2"; shift 2;;
    --port-offset) OFFSET="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    --fake) FAKE=1; shift;;
    --no-stand) STAND=0; shift;;
    --isaac-args) ISAAC_ARGS="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ "$FAKE" == 1 ]]; then
  [[ "$OFFSET" == 0 ]] && OFFSET=200
  SESSION=${SESSION:-body-m1fake}
else
  SESSION=${SESSION:-wl-m1}
fi
export WL_PORT_OFFSET=$OFFSET
TS=$(date +%Y%m%d-%H%M%S)
LOGD=/work/logs/wl
RUN=$WL/outputs/m1/stack-$TS
mkdir -p "$LOGD" "$RUN"
LOG_ISAAC=$LOGD/$SESSION-$TS-isaac.log
LOG_DEPLOY=$LOGD/$SESSION-$TS-deploy.log
LOG_BODY=$LOGD/$SESSION-$TS-body.log
echo "$SESSION" > "$RUN/session"; echo "$OFFSET" > "$RUN/port_offset"
ln -sfn "$RUN" "$WL/outputs/m1/stack-latest-$SESSION"

say() { echo "[m1_up $(date +%H:%M:%S)] $*"; }
die() { echo "[m1_up] ERROR: $*" >&2; echo "[m1_up] logs: $LOGD/$SESSION-$TS-*.log ; tear down with scripts/m1_down.sh --session $SESSION" >&2; exit 1; }
wr() { (cd "$WL" && "$PY_BODY" -m tools.wait_ready "$@" --port-offset "$OFFSET"); }
cli() { (cd "$WL" && "$PY_BODY" -m tools.body_cli --port-offset "$OFFSET" "$@"); }

# ---- preflight: never touch anything we did not start
tmux has-session -t "=$SESSION" 2>/dev/null && die "tmux session $SESSION already exists (scripts/m1_down.sh --session $SESSION)"
for base in 5556 5557 5565 5600 5601 5610 5611; do
  p=$((base + OFFSET))
  if ss -ltn "sport = :$p" | grep -q LISTEN; then die "port $p already bound by another process"; fi
done
[[ -x "$PY_BODY" ]] || die "body venv missing: $PY_BODY"

# ---- 1. P1
tmux new-session -d -s "$SESSION" -n isaac -x 220 -y 50 "bash --noprofile --norc"
if [[ "$FAKE" == 1 ]]; then
  tmux send-keys -t "=$SESSION:isaac" "cd $WL && exec $PY_BODY -u -m tools.fake_p1 --port-offset $OFFSET --out $RUN/fake_p1 2>&1 | tee -a $LOG_ISAAC" C-m
else
  tmux send-keys -t "=$SESSION:isaac" "cd $WL && exec ${ISAAC_TASKSET:+taskset -c $ISAAC_TASKSET }$PY_ISAAC -u -m sim_isaac.app --house $HOUSE --physics-hz 200 --dds-domain 0 --dds-iface lo --camera 640x480 --camera-hz 30 --rt-pace --physx-device cpu --port-offset $OFFSET $ISAAC_ARGS 2>&1 | tee -a $LOG_ISAAC" C-m
fi
say "P1 starting (log $LOG_ISAAC); first Isaac launch compiles shaders (up to ~15 min)"
wr p1 --timeout "$P1_TIMEOUT" || die "P1 did not answer ping"
if [[ "$FAKE" != 1 ]]; then
  for i in $(seq 1 120); do grep -q WL_ISAAC_READY "$LOG_ISAAC" && break; sleep 1; done
  grep -q WL_ISAAC_READY "$LOG_ISAAC" || die "no WL_ISAAC_READY line from P1"
fi
wr gtpose --timeout 120 --min-hz 40 || die "gt.pose not flowing at >= 40 Hz"
wr lowstate --timeout 60 || die "P1 not publishing rt/lowstate"
wr camera --timeout 60 || say "WARNING: no camera frame yet (continuing)"
cli p1 band on=true >/dev/null || die "band on failed"
say "P1 up, band on"

# ---- 2. P3 body (binds the SONIC input PUB before the deploy connects)
tmux new-window -t "=$SESSION" -n body "bash --noprofile --norc"
tmux send-keys -t "=$SESSION:body" "cd $WL && exec ${BODY_TASKSET:+taskset -c $BODY_TASKSET }$PY_BODY -u -m body.service --port-offset $OFFSET --log-dir $RUN/body 2>&1 | tee -a $LOG_BODY" C-m
wr body --timeout 30 || die "body service not up"
say "body up (planner IDLE keepalive running)"

# ---- 3. P2 deploy
if [[ "$FAKE" == 1 ]]; then
  tmux new-window -t "=$SESSION" -n deploy "bash --noprofile --norc"
  tmux send-keys -t "=$SESSION:deploy" "cd $WL && exec $PY_BODY -u -m tools.fake_deploy --port-offset $OFFSET 2>&1 | tee -a $LOG_DEPLOY" C-m
  for i in $(seq 1 30); do grep -q "Init Done" "$LOG_DEPLOY" 2>/dev/null && break; sleep 1; done
else
  # DEPLOY_FORCE=1 only inside an isolated network namespace (scripts/body_netns_e2e.sh): run_deploy.sh refuses to
  # start while another g1_deploy_onnx_ref runs anywhere on the host, because DDS domain 0 on lo would be shared.
  bash "$WL/sonic/run_deploy.sh" start --session "$SESSION" --window deploy --log "$LOG_DEPLOY" \
      --zmq-port $((5556 + OFFSET)) --zmq-out-port $((5557 + OFFSET)) --wait-init "${DEPLOY_WAIT_S:-900}" \
      ${DEPLOY_FORCE:+--force} ${DEPLOY_TASKSET:+--taskset $DEPLOY_TASKSET} || die "deploy did not reach Init Done"
fi
wr deploy_log --file "$LOG_DEPLOY" --timeout 30 || die "no 'Init Done' in $LOG_DEPLOY"
wr deploy --timeout 30 || die "deploy not publishing on $((5557 + OFFSET))"
say "deploy up (Init Done)"

# ---- 4. stand + band release (SONIC takes the weight; failure leaves the band on)
if [[ "$STAND" == 1 ]]; then
  if cli --timeout 120 stand > "$RUN/stand.json"; then
    say "STANDING under SONIC control: $(cat "$RUN/stand.json" | head -c 400)"
  else
    cli p1 band on=true >/dev/null || true
    die "stand failed (band re-engaged): $(cat "$RUN/stand.json" | head -c 600)"
  fi
fi

# ---- 5. monitor
tmux new-window -t "=$SESSION" -n monitor "bash --noprofile --norc"
tmux send-keys -t "=$SESSION:monitor" "cd $WL && exec $PY_BODY -m tools.m1_monitor --port-offset $OFFSET" C-m
say "stack up: tmux attach -t $SESSION ; run dir $RUN ; drive test: (cd $WL && $PY_BODY -m tools.m1_drive_test --port-offset $OFFSET)"
