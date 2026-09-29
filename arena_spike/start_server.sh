#!/usr/bin/env bash
# Start a GR00T policy server (GR00T's stock run_gr00t_server.py) on the host, outside Docker, as the Arena docs do.
#   bash start_server.sh n16   -> N1.6 loco-manip ckpt, Isaac-GR00T e29d8fc, port 6555
#   bash start_server.sh n17   -> N1.7 static-apple ckpt, Isaac-GR00T 4b1dca9, port 6556 (5555 is taken by nv-hostengine/DCGM on Nebius)
set -euo pipefail
source /etc/profile.d/ludo.sh
A=/work/arena
ARENA=$A/IsaacLab-Arena
case "${1:?n16|n17}" in
  n16) DIR=$A/gr00t_n16; PORT=6555
       MODCFG=$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py
       CKPT=$A/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000;;
  n17) DIR=$A/gr00t_n17; PORT=6556
       MODCFG=$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py
       CKPT=$A/models/isaaclab_arena/static_apple_tutorial/gn1x_tuned_static_apple;;
esac
cd "$DIR"
echo "[server $1] $(git log -1 --format='%h %s') ckpt=$CKPT port=$PORT"
exec .venv/bin/python gr00t/eval/run_gr00t_server.py \
  --modality-config-path "$MODCFG" \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda --host 127.0.0.1 --port "$PORT"
