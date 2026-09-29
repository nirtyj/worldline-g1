#!/usr/bin/env bash
# Box-side launcher for the viz stack (run ON the box, from anywhere). tmux sessions are named viz-*.
#
#   viz/box.sh setup                         # create viz/.venv (uv): pyzmq msgpack numpy pillow imageio[ffmpeg] aiohttp pytest
#   viz/box.sh sim   [test_stack args...]    # VIZ TEST stand-in P1 (kinematic G1) in tmux viz-sim, offset 100
#   viz/box.sh server [OFFSET]               # web UI on 127.0.0.1:(8765+OFFSET) in tmux viz-server (default 0)
#   viz/box.sh record SECONDS [OFFSET] [LABEL]   # one recording, foreground, prints the run dir
#   viz/box.sh status | logs NAME | stop     # stop kills only viz-sim / viz-server / viz-rec
set -euo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true
WL=/work/worldline-g1
VIZ=$WL/viz
PY=$VIZ/.venv/bin/python
ISAAC_PY=/work/envs/isaaclab/bin/python
LOGS=$WL/outputs/viz_test
mkdir -p "$LOGS"

cmd=${1:-status}; shift || true
case "$cmd" in
  setup)
    cd "$VIZ" && [[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
    uv pip install --python "$PY" -q pyzmq msgpack numpy pillow "imageio[ffmpeg]" aiohttp pytest
    "$PY" -c "import zmq, msgpack, numpy, PIL, imageio_ffmpeg, aiohttp, pytest; print('viz venv ok')";;
  sim)
    tmux kill-session -t viz-sim 2>/dev/null || true
    tmux new-session -d -s viz-sim -c "$WL" \
      "source /etc/profile.d/ludo.sh; $ISAAC_PY $VIZ/test_stack.py $* 2>&1 | tee $LOGS/sim.log; sleep 3600"
    echo "viz-sim started; log $LOGS/sim.log (wait for WL_VIZ_TEST_READY)";;
  server)
    off=${1:-0}
    tmux kill-session -t viz-server 2>/dev/null || true
    tmux new-session -d -s viz-server -c "$WL" \
      "source /etc/profile.d/ludo.sh; $PY $VIZ/server.py --port-offset $off 2>&1 | tee $LOGS/server_$off.log"
    echo "viz-server on 127.0.0.1:$((8765 + off)); laptop: 00_infra/tunnel.sh $((8765 + off))";;
  record)
    secs=${1:-30}; off=${2:-0}; label=${3:-cli}
    "$PY" "$VIZ/recorder.py" --duration "$secs" --port-offset "$off" --label "$label";;
  status)
    tmux ls 2>/dev/null | grep '^viz-' || echo "no viz-* sessions"
    ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | grep -E ':(5565|5665|56[0-9]{2}|57[0-9]{2}|8765|8865)$' | sort || true;;
  logs)
    tail -n 40 "$LOGS/${1:-sim}.log";;
  stop)
    for s in viz-sim viz-server viz-rec; do tmux kill-session -t "$s" 2>/dev/null && echo "killed $s" || true; done;;
  *) echo "usage: viz/box.sh setup|sim|server|record|status|logs|stop"; exit 2;;
esac
