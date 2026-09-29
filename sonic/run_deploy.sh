#!/usr/bin/env bash
# Start / stop the UNMODIFIED gear_sonic_deploy (g1_deploy_onnx_ref) in zmq_manager input mode, sim mode.
#
#   run_deploy.sh start  [opts]   start detached in tmux (default session "deploy-sonic"), optionally wait for "Init Done"
#   run_deploy.sh fg     [opts]   run in the foreground (for a tmux window someone else owns, e.g. m1_up.sh)
#   run_deploy.sh stop   [opts]   clean stop: 'o' on stdin (EMERGENCY STOP -> damping cmd -> exit), then SIGINT/SIGKILL
#   run_deploy.sh status [opts]   running? pid, uptime, last log lines
#   run_deploy.sh rt PRIO [opts]  give all threads except main() SCHED_RR PRIO (sudo chrt); see --rt-prio
#
# Options:
#   --iface IFACE          DDS network interface (default lo). The DDS domain is hard-coded to 0 in the binary
#                          (g1_deploy_onnx_ref.cpp:2218); there is no domain flag.
#   --zmq-host HOST        host of the command/planner/pose PUB the deploy connects to (default localhost)
#   --zmq-port PORT        default 5556 (contract); build-phase tests use 5656
#   --zmq-out-port PORT    g1_debug PUB bind port, default 5557 (contract); build-phase tests use 5657
#   --session NAME         tmux session (default deploy-sonic). If it exists, a window is added.
#   --window NAME          tmux window name (default sonic)
#   --log FILE             default /work/logs/wl/<session>.log (appended; a "=== start" marker per run)
#   --wait-init SECS       start: wait until the deploy prints "Init Done" (0 = do not wait; default 0)
#   --taskset CPUS         run under taskset -c CPUS (the binary still pins its main thread to CPU 0)
#   --rt-prio N            start: after "Init Done", put every deploy thread except the main thread under
#                          SCHED_RR priority N (sudo chrt). The main thread is left alone on purpose: it loops on
#                          sleep(0) (g1_deploy_onnx_ref.cpp:4523-4526) and would spin a core at RT priority. On the
#                          real robot the deploy runs privileged and requests SCHED_FIFO itself (cpp:2632-2640).
#   --force                start even if another g1_deploy_onnx_ref runs in this network namespace (DDS domain 0
#                          on the same lo is shared!). Deploys in other netns are ignored automatically.
#   -- ARGS...             extra args passed verbatim to the binary (e.g. --planner-precision 32)
#
# tmux targets always use exact matching ("=session:=window"): tmux 3.4 prefix-matches bare names, so
# "-t deploy-ref" would hit a session called "deploy-refq1" once "deploy-ref" is gone.
#
# Command line = what deploy.sh sim --input-type zmq_manager would run (deploy.sh:240-246, 404-406, 553-588),
# without the [Y/n] prompt and the per-start `just build` (build with build_deploy.sh instead).
set -euo pipefail
source "$(dirname "$0")/sonic_env.sh"

CMD="${1:-}"; shift || true
IFACE=lo; ZMQ_HOST=localhost; ZMQ_PORT=$SONIC_ZMQ_PORT_DEFAULT; ZMQ_OUT_PORT=$SONIC_ZMQ_OUT_PORT_DEFAULT
SESSION=deploy-sonic; WINDOW=sonic; LOGF=""; WAIT_INIT=0; TASKSET=""; FORCE=0; EXTRA=(); RT_PRIO=""
[[ "$CMD" == rt ]] && { RT_PRIO="${1:?usage: run_deploy.sh rt PRIO [--session S]}"; shift; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface) IFACE="$2"; shift 2;;
    --zmq-host) ZMQ_HOST="$2"; shift 2;;
    --zmq-port) ZMQ_PORT="$2"; shift 2;;
    --zmq-out-port) ZMQ_OUT_PORT="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --log) LOGF="$2"; shift 2;;
    --wait-init) WAIT_INIT="$2"; shift 2;;
    --taskset) TASKSET="$2"; shift 2;;
    --rt-prio) RT_PRIO="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
PIDF="$LOG_ROOT/$SESSION.pid"
LOGPATHF="$LOG_ROOT/$SESSION.logpath"   # start records the log path here so status/stop find a custom --log
if [[ -z "$LOGF" && "$CMD" != start && "$CMD" != fg && -s "$LOGPATHF" ]]; then LOGF=$(cat "$LOGPATHF"); fi
LOGF="${LOGF:-$LOG_ROOT/$SESSION.log}"

deploy_args() {
  printf '%q ' "$DEPLOY_BIN" "$IFACE" \
    policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all \
    --zmq-host "$ZMQ_HOST" --zmq-port "$ZMQ_PORT" --zmq-out-port "$ZMQ_OUT_PORT" \
    --disable-crc-check ${EXTRA[@]+"${EXTRA[@]}"}
}
running_pid() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null && tr '\0' ' ' < "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null | grep -q g1_deploy_onnx_ref && cat "$PIDF"; }
preflight() {
  [[ -x "$DEPLOY_BIN" ]] || die "deploy binary missing: bash $SONIC_DIR/build_deploy.sh"
  for f in policy/release/model_decoder.onnx policy/release/model_encoder.onnx policy/release/observation_config.yaml \
           planner/target_vel/V2/planner_sonic.onnx reference/example; do
    [[ -e "$DEPLOY_DIR/$f" ]] || die "missing $DEPLOY_DIR/$f (build_deploy.sh step 5)"
  done
  if p=$(running_pid); then die "deploy already running for session $SESSION (pid $p)"; fi
  others=$(deploys_in_my_netns | tr '\n' ' ')
  if [[ -n "${others// /}" && "$FORCE" != 1 ]]; then
    die "another g1_deploy_onnx_ref is running in this network namespace (pid $others) on DDS domain 0; stop it or pass --force"
  fi
  if ss -ltn "sport = :$ZMQ_OUT_PORT" | grep -q LISTEN; then die "g1_debug port $ZMQ_OUT_PORT already bound"; fi
  ip link show "$IFACE" >/dev/null 2>&1 || die "no interface $IFACE"
}
apply_rt() {  # apply_rt PID PRIO: SCHED_RR for all threads except the main thread
  local pid=$1 prio=$2 n=0 tid
  for tid in $(ls "/proc/$pid/task"); do
    [[ "$tid" == "$pid" ]] && continue
    sudo -n chrt -r -p "$prio" "$tid" >/dev/null 2>&1 && n=$((n + 1))
  done
  log "SCHED_RR $prio applied to $n threads of deploy pid $pid (main thread left SCHED_OTHER)"
}
# The shell line run inside tmux / the foreground: record pid, log everything, exec the binary.
inner_cmd() {
  local ts; ts=$(date -Is)
  local pre=""; [[ -n "$TASKSET" ]] && pre="taskset -c $TASKSET "
  cat <<EOF
cd $(printf %q "$DEPLOY_DIR") && echo "=== start $ts iface=$IFACE zmq=$ZMQ_HOST:$ZMQ_PORT out=$ZMQ_OUT_PORT ===" >> $(printf %q "$LOGF") && echo \$\$ > $(printf %q "$PIDF") && echo $(printf %q "$LOGF") > $(printf %q "$LOGPATHF") && exec > >(stdbuf -oL tee -a $(printf %q "$LOGF")) 2>&1 && export LD_LIBRARY_PATH=$(printf %q "$(deploy_ld_path)") TensorRT_ROOT=$(printf %q "$TensorRT_ROOT") && exec ${pre}$(deploy_args)
EOF
}

case "$CMD" in
  start)
    preflight
    touch "$LOGF"; start_line=$(wc -l < "$LOGF")
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      tmux new-window -d -t "=$SESSION" -n "$WINDOW" "bash --noprofile --norc"
    else
      tmux new-session -d -s "$SESSION" -n "$WINDOW" -x 200 -y 50 "bash --noprofile --norc"
    fi
    tmux send-keys -t "=$SESSION:=$WINDOW" "$(inner_cmd)" C-m
    log "started deploy in tmux $SESSION:$WINDOW (log $LOGF)"
    if [[ "$WAIT_INIT" != 0 ]]; then
      for ((i = 0; i < WAIT_INIT * 2; i++)); do
        if tail -n +"$((start_line + 1))" "$LOGF" | grep -q "Init Done"; then
          log "deploy ready: Init Done after $((i / 2))s (pid $(cat "$PIDF"))"
          [[ -n "$RT_PRIO" ]] && apply_rt "$(cat "$PIDF")" "$RT_PRIO"
          exit 0
        fi
        if ((i > 6)) && ! running_pid >/dev/null; then
          tail -n +"$((start_line + 1))" "$LOGF" | tail -30 >&2; die "deploy exited during start-up"
        fi
        sleep 0.5
      done
      tail -n +"$((start_line + 1))" "$LOGF" | tail -30 >&2
      die "no 'Init Done' within ${WAIT_INIT}s (first start builds TensorRT engines; is the simulator publishing rt/lowstate?)"
    fi
    ;;
  fg)
    preflight
    exec bash --noprofile --norc -c "$(inner_cmd)"
    ;;
  stop)
    pid=$(running_pid || true)
    if [[ -z "$pid" ]]; then log "not running (session $SESSION)"; tmux kill-window -t "=$SESSION:=$WINDOW" 2>/dev/null || true; exit 0; fi
    # 1) EMERGENCY STOP on stdin (zmq_manager.hpp:166-175): Stop() sends a damping command and main() exits.
    if tmux has-session -t "=$SESSION" 2>/dev/null; then tmux send-keys -t "=$SESSION:=$WINDOW" o C-m 2>/dev/null || true; fi
    for ((i = 0; i < 20; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    # 2) SIGINT, 3) SIGKILL
    if kill -0 "$pid" 2>/dev/null; then log "no exit after 'o'; SIGINT"; kill -INT "$pid" 2>/dev/null || true; sleep 2; fi
    if kill -0 "$pid" 2>/dev/null; then log "SIGKILL"; kill -KILL "$pid" 2>/dev/null || true; sleep 0.5; fi
    kill -0 "$pid" 2>/dev/null && die "pid $pid still alive"
    rm -f "$PIDF"
    tmux kill-window -t "=$SESSION:=$WINDOW" 2>/dev/null || true
    log "deploy stopped (pid $pid)"
    ;;
  rt)
    pid=$(running_pid) || die "not running (session $SESSION)"
    apply_rt "$pid" "$RT_PRIO"
    ;;
  status)
    if pid=$(running_pid); then
      echo "running pid=$pid etime=$(ps -o etime= -p "$pid" | tr -d ' ') cpu=$(ps -o %cpu= -p "$pid" | tr -d ' ')% session=$SESSION log=$LOGF"
      tail -5 "$LOGF" 2>/dev/null || true
    else
      echo "not running (session $SESSION)"; exit 1
    fi
    ;;
  *) sed -n '2,26p' "$0"; exit 2;;
esac
