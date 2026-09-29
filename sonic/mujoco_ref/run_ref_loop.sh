#!/usr/bin/env bash
# Known-good reference loop: gear_sonic MuJoCo sim (headless, upstream classes) + the unmodified C++ deploy
# (zmq_manager, sim, lo, DDS domain 0) + drive_ref.py (command/planner messages built with gear_sonic's
# builders). Fully non-interactive; the elastic band is released through sim_ref.py's control socket.
#
#   bash /work/worldline-g1/sonic/mujoco_ref/run_ref_loop.sh [--init-yaw-deg D] [--long-stand S] [--live-video]
#        [--band-lower M] [--no-stop-test] [--nice N] [--tag NAME] [-- drive_ref.py args...]
#   --live-video  render the tracking video inside the real-time loop (costs RTF); default: offline replay
#                 of the ground-truth trace by render_ref.py after the run
#   --nice N      sudo renice the sim and deploy processes to N (e.g. -5) against CPU contention on the box
#   --pace P      sim wall-clock pacing: deadline (default) | upstream (base_sim.py per-step sleep)
#   --rt PRIO     SCHED_RR PRIO for the sim (all threads) and the deploy (all threads but main), via sudo chrt
#   --driver F    scenario script in this dir: drive_ref.py (default, pass/fail scenario) | char_ref.py (characterisation)
#   --taskset C   run the sim and the deploy under taskset -c C (e.g. 0-3)
#   --no-render   skip the offline mp4 render (render_ref.py) after the run
#   --allow-busy  run even if other GPU / Isaac jobs are on the box (default: refuse, see docs/walk_diagnosis.md)
#   --sim-args S  extra sim_ref.py args, e.g. "--inject-stall-ms 60 --inject-every-s 3 [--inject-drop]"
#
# Ports (build phase, contract +100): planner/command PUB 5656, g1_debug 5657, sim control REP 5712.
# Outputs: /work/worldline-g1/outputs/m1/deploy/<tag>-<ts>/ {sim.log, deploy.log, sim_trace.jsonl,
#   sim_events.jsonl, sim_stats.json, sim_tracking.mp4, drive_*.json(l), deploy_g1_debug.jsonl, cpu.txt, report/}
set -euo pipefail
source "$(dirname "$0")/../sonic_env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

DRIVER=drive_ref.py; TASKSET=""; SIM_EXTRA=""; RENDER=1; ALLOW_BUSY=0; INIT_YAW=0; LONG_STAND=0; VIDEO=""; NICE=""; RT=""; PACE=deadline; BAND_LOWER=0.20; STOP_TEST=--send-stop; TAG=mujoco-ref; DRIVE_EXTRA=()
ZMQ_PORT=5656; ZMQ_OUT_PORT=5657; CTL_PORT=5712
while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver) DRIVER="$2"; shift 2;;
    --taskset) TASKSET="$2"; shift 2;;
    --sim-args) SIM_EXTRA="$2"; shift 2;;
    --no-render) RENDER=0; shift;;
    --allow-busy) ALLOW_BUSY=1; shift;;
    --init-yaw-deg) INIT_YAW="$2"; shift 2;;
    --long-stand) LONG_STAND="$2"; shift 2;;
    --live-video) VIDEO=--video; shift;;
    --nice) NICE="$2"; shift 2;;
    --rt) RT="$2"; shift 2;;
    --pace) PACE="$2"; shift 2;;
    --band-lower) BAND_LOWER="$2"; shift 2;;
    --no-stop-test) STOP_TEST=""; shift;;
    --tag) TAG="$2"; shift 2;;
    --) shift; DRIVE_EXTRA=("$@"); break;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
[[ -x "$VENV_SIM/bin/python" ]] || die ".venv_sim missing: bash $SONIC_DIR/build_deploy.sh"
RUN="$OUT_ROOT/$TAG-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"; ln -sfn "$RUN" "$OUT_ROOT/$TAG-latest"
S=deploy-ref
log "run dir $RUN"
for p in $ZMQ_PORT $ZMQ_OUT_PORT $CTL_PORT; do
  ss -ltn "sport = :$p" | grep -q LISTEN && die "port $p busy"
done
[[ -z "$(deploys_in_my_netns)" ]] || die "a deploy is already running in this network namespace (DDS domain 0 on lo is exclusive)"
# busy-box preflight (docs/walk_diagnosis.md rule 1): other Isaac / GPU jobs disturb the wall-clock timing between
# the deploy and the sim, so the result would describe the box, not the controller. --allow-busy overrides (the run
# is then marked busy in host.txt and the report's timing gate decides VALID/INVALID).
ISAAC_PIDS=$(isaac_jobs | tr '\n' ' ')
BUSY="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr '\n' ';')${ISAAC_PIDS:+ isaac_pids=$ISAAC_PIDS}"
if [[ -n "${BUSY// /}" && "$ALLOW_BUSY" != 1 ]]; then
  rmdir "$RUN" 2>/dev/null; rm -f "$OUT_ROOT/$TAG-latest"
  die "box busy (GPU/Isaac jobs: $BUSY); rerun on a quiet box or pass --allow-busy"
fi

cleanup() {
  set +e
  bash "$SONIC_DIR/run_deploy.sh" stop --session "$S" --log "$RUN/deploy.log" >/dev/null 2>&1
  "$VENV_SIM/bin/python" - "$CTL_PORT" <<'EOF' 2>/dev/null
import json, sys, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 2000); s.setsockopt(zmq.LINGER, 0)
s.connect(f"tcp://127.0.0.1:{sys.argv[1]}"); s.send(json.dumps({"op": "quit"}).encode()); s.recv()
EOF
  for i in $(seq 1 40); do tmux has-session -t "=$S" 2>/dev/null || break; sleep 0.5; done
  tmux kill-session -t "=$S" 2>/dev/null  # exact match: a bare name prefix-matches e.g. deploy-refq1
}
trap cleanup EXIT

# 1) simulator (band on)
tmux new-session -d -s "$S" -n sim -x 200 -y 50 "bash --noprofile --norc"
tmux send-keys -t "=$S:=sim" "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl; cd $(printf %q "$WBC_DIR") && exec ${TASKSET:+taskset -c $TASKSET }$(printf %q "$VENV_SIM/bin/python") -u $(printf %q "$HERE/sim_ref.py") --ctl-port $CTL_PORT --out $(printf %q "$RUN") --init-yaw-deg $INIT_YAW --pace $PACE $VIDEO $SIM_EXTRA > $(printf %q "$RUN/sim.log") 2>&1" C-m
for i in $(seq 1 120); do grep -q "\[sim_ref\] running" "$RUN/sim.log" 2>/dev/null && break; sleep 0.5; done
grep -q "\[sim_ref\] running" "$RUN/sim.log" || { tail -30 "$RUN/sim.log"; die "sim did not start"; }
log "sim up: $(grep '\[sim_ref\] running' "$RUN/sim.log")"

# 2) deploy (waits for Init Done; the first start builds the TensorRT engines)
T_DEP=$(date +%s)
bash "$SONIC_DIR/run_deploy.sh" start --session "$S" --window sonic --zmq-port $ZMQ_PORT --zmq-out-port $ZMQ_OUT_PORT \
  --log "$RUN/deploy.log" --wait-init 1200 ${TASKSET:+--taskset $TASKSET}
log "deploy ready after $(( $(date +%s) - T_DEP ))s"
DPID=$(cat "$LOG_ROOT/$S.pid")
SPID=$(pgrep -f "sim_ref.py --ctl-port $CTL_PORT" | head -1)
if [[ -n "$NICE" ]]; then
  log "renicing all threads of sim $SPID and deploy $DPID to $NICE"
  for tid in $(ls /proc/$DPID/task); do sudo renice -n "$NICE" -p "$tid" >/dev/null 2>&1; done
  for tid in $(ls /proc/$SPID/task); do sudo renice -n "$NICE" -p "$tid" >/dev/null 2>&1; done
fi
if [[ -n "$RT" ]]; then
  bash "$SONIC_DIR/run_deploy.sh" rt "$RT" --session "$S"
  n=0; for tid in $(ls /proc/$SPID/task); do sudo -n chrt -r -p "$RT" "$tid" >/dev/null 2>&1 && n=$((n+1)); done
  log "SCHED_RR $RT applied to $n sim threads"
fi
ps -L -o tid,cls,rtprio,ni,comm -p "$DPID" > "$RUN/deploy_threads.txt" 2>/dev/null || true
ps -L -o tid,cls,rtprio,ni,comm -p "$SPID" > "$RUN/sim_threads.txt" 2>/dev/null || true
echo "busy_at_start=[${BUSY:-}] sim_pid=$SPID deploy_pid=$DPID taskset=${TASKSET:-none} rt=${RT:-none} nice=${NICE:-0} nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)" > "$RUN/host.txt"

# 3) drive
( while kill -0 "$DPID" 2>/dev/null; do echo "$(date +%s) $(ps -o %cpu=,rss= -p "$DPID")" ; top -b -n1 -H -p "$DPID" | sed -n '8,14p' | awk '{print "   thr", $1, $9, $10, $12}'; sleep 5; done ) > "$RUN/cpu.txt" 2>/dev/null &
CPU_MON=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$RUN/gpu.csv" 2>/dev/null &
GPU_MON=$!
# load.txt: epoch load1 n_isaac_python n_deploys_all_netns n_mujoco_sims_all_netns top3_cpu_procs
( while kill -0 "$DPID" 2>/dev/null; do
    echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) $(ps -eo args | grep -c '[i]saaclab/bin/python') $(pgrep -fc 'target/release/g1_deploy_onnx_ref') $(pgrep -fc 'sim_ref|run_sim_loop|sim_min') $(ps -eo pcpu,comm --sort=-pcpu | sed -n '2,4p' | awk '{printf "%s:%s,", $2, $1}')"
    sleep 2; done ) > "$RUN/load.txt" 2>/dev/null &
LOAD_MON=$!
set +e
"$VENV_SIM/bin/python" -u "$HERE/$DRIVER" --zmq-port $ZMQ_PORT --zmq-out-port $ZMQ_OUT_PORT --ctl-port $CTL_PORT \
  --out "$RUN" --band-lower "$BAND_LOWER" --long-stand-secs "$LONG_STAND" $STOP_TEST "${DRIVE_EXTRA[@]}" 2>&1 | tee "$RUN/drive.log"
DRIVE_RC=${PIPESTATUS[0]}
set -e
sleep 2
if [[ -n "$STOP_TEST" ]]; then
  if kill -0 "$DPID" 2>/dev/null; then log "deploy still alive after ZMQ stop"; echo alive > "$RUN/deploy_after_stop.txt"
  else log "deploy exited after ZMQ stop (as the code says)"; echo exited > "$RUN/deploy_after_stop.txt"; fi
fi
kill "$CPU_MON" "$GPU_MON" "$LOAD_MON" 2>/dev/null || true
cleanup; trap - EXIT
# 4) report (plots + metrics); matplotlib from the Isaac Lab env if .venv_sim has none
PY=$VENV_SIM/bin/python; "$PY" -c 'import matplotlib' 2>/dev/null || PY=/work/envs/isaaclab/bin/python
"$PY" "$HERE/report_ref.py" "$RUN" || log "report failed"
if [[ "$RENDER" == 1 ]]; then
  (cd "$WBC_DIR" && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV_SIM/bin/python" "$HERE/render_ref.py" "$RUN") || log "offline render failed"
fi
log "done rc=$DRIVE_RC: $RUN"
exit $DRIVE_RC
