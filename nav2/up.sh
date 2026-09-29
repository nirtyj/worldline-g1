#!/usr/bin/env bash
# Start the ROS side of Worldline-G1 navigation in tmux: ros_bridge (ZMQ <-> ROS) + Nav2 (one container).
#
#   nav2/up.sh [--port-offset N] [--session wl-nav2] [--controller rpp|mppi] [--no-composition] [--no-smoother]
#              [--log-dir DIR] [--wait S] [--map-npz FILE]
#
# Needs P1 (gt.pose + get_occupancy) up first: the bridge publishes /map + TF from it, and Nav2's costmaps only
# activate once odom->base_link exists. wl-body can start before or after (go_to falls back to A* until Nav2 is ready).
# Waits until the bridge reports nav2_ready (map published, gt.pose fresh, lifecycle nodes active), prints the run dir.
# Env: NAV2_TASKSET (e.g. 12-15) pins bridge + Nav2; NAV2_NICE (default 5) keeps them below the wall-clock SONIC deploy.
# ROS env: nav2/ros_env.sh (ROS_DOMAIN_ID=42, rmw_fastrtps_cpp, localhost discovery; never the Unitree domain 0).
set -euo pipefail
WL=${WL:-/work/worldline-g1}
OFFSET=${WL_PORT_OFFSET:-0}; SESSION=wl-nav2; CONTROLLER=${NAV2_CONTROLLER:-rpp}; COMPOSE=true; SMOOTHER=true
WAIT=120; LOGDIR=""; MAPNPZ=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-offset) OFFSET="$2"; shift 2;;
    --session) SESSION="$2"; shift 2;;
    --controller) CONTROLLER="$2"; shift 2;;
    --no-composition) COMPOSE=false; shift;;
    --no-smoother) SMOOTHER=false; shift;;
    --log-dir) LOGDIR="$2"; shift 2;;
    --wait) WAIT="$2"; shift 2;;
    --map-npz) MAPNPZ="$2"; shift 2;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
LOGDIR=${LOGDIR:-$WL/outputs/m1/nav2/run-$(date +%Y%m%d-%H%M%S)-$SESSION}
mkdir -p "$LOGDIR" "$WL/outputs/m1/nav2"
say() { echo "[nav2/up $(date +%H:%M:%S)] $*"; }
die() { echo "[nav2/up] ERROR: $*" >&2; echo "[nav2/up] logs: $LOGDIR ; tear down: nav2/down.sh --session $SESSION" >&2; exit 1; }
tmux has-session -t "=$SESSION" 2>/dev/null && die "tmux session $SESSION already exists (nav2/down.sh --session $SESSION)"
BR_PORT=$((5620 + OFFSET))
ss -ltn "sport = :$BR_PORT" | grep -q LISTEN && die "port $BR_PORT already bound (another bridge?)"
[[ -f /opt/ros/jazzy/setup.bash ]] || die "ROS 2 Jazzy missing: bash nav2/install_ros.sh"
PRE="source $WL/nav2/ros_env.sh; cd $WL; exec nice -n ${NAV2_NICE:-5} ${NAV2_TASKSET:+taskset -c $NAV2_TASKSET }"
echo "$OFFSET" > "$LOGDIR/port_offset"; echo "$SESSION" > "$LOGDIR/session"; echo "$CONTROLLER" > "$LOGDIR/controller"
ln -sfn "$LOGDIR" "$WL/outputs/m1/nav2/latest-$SESSION"

tmux new-session -d -s "$SESSION" -n bridge -x 200 -y 50 "bash --noprofile --norc"
tmux send-keys -t "=$SESSION:bridge" "$PRE python3 -u nav2/ros_bridge.py --port-offset $OFFSET --log-dir $LOGDIR ${MAPNPZ:+--map-npz $MAPNPZ} 2>&1 | tee -a $LOGDIR/bridge.log" C-m
tmux new-window -t "=$SESSION" -n nav2 "bash --noprofile --norc"
tmux send-keys -t "=$SESSION:nav2" "$PRE ros2 launch nav2/launch/wl_nav2.launch.py controller:=$CONTROLLER use_composition:=$COMPOSE velocity_smoother:=$SMOOTHER 2>&1 | tee -a $LOGDIR/nav2.log" C-m
say "bridge + Nav2 ($CONTROLLER, composition=$COMPOSE) starting in tmux $SESSION, offset $OFFSET, logs $LOGDIR"
if ! python3 "$WL/nav2/tools/bridge_cli.py" --port-offset "$OFFSET" --timeout "$WAIT" wait > "$LOGDIR/ready.json"; then
  die "Nav2 not ready after ${WAIT}s: $(head -c 400 "$LOGDIR/ready.json")"
fi
say "Nav2 READY: $(head -c 300 "$LOGDIR/ready.json")"
echo "$LOGDIR"
