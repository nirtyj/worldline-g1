#!/usr/bin/env bash
# Closed-loop eval of an Arena G1 GR00T checkpoint (Arena client in the container -> GR00T server on the host).
#   bash run_eval.sh n16 <tag> <policy_runner length args...>   e.g. bash run_eval.sh n16 n16_ep1 --num_steps 1500
#   bash run_eval.sh n17 <tag> --num_episodes 10
# Extra env: DEVICE (default: runner default = cuda:0), NUM_ENVS (default 1), VIDEO=0 to disable video,
#   POLICY_TYPE (default arena_spike_policies.TimedGr00tRemotePolicy; arena_probe_policies.BinProbeGr00tPolicy + SPIKE_BIN_POSE=x,y,z for the goal probe).
set -euo pipefail
which=${1:?n16|n17}; tag=${2:?tag}; shift 2
S=/work/arena/spike
# Videos come from arena_spike_policies._Recorder (ego + third-person mp4 per episode); the stock --video path
# renders /OmniverseKit_Persp which is black headless here. VIDEO=0 disables our recorder.
export SPIKE_VIDEO=${VIDEO:-1}
DEV_ARGS=""; [[ -n "${DEVICE:-}" ]] && DEV_ARGS="--device $DEVICE"
case $which in
  n16) export SPIKE_CAM=${SPIKE_CAM:-1.2,-2.6,0.7,0.0,-0.4,-0.3;0.3,-0.2,2.2,0.0,-0.6,-0.8}
       CFG=isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml; PORT=6555
       TASK="galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_joint";;
  n17) CFG=/spike/g1_static_apple_gr00t_closedloop_config.yaml; PORT=6556  # docs: set model_path to the served ckpt
       TASK="galileo_g1_static_pick_and_place --object apple_01_objaverse_robolab --destination clay_plates_hot3d_robolab --embodiment g1_wbc_agile_joint";;
esac
mkdir -p /work/arena/eval/videos/$tag
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; echo ---; sleep 10; done ) > /work/arena/eval/${tag}_vram.log 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
t0=$(date +%s)
SPIKE_TAG=$tag bash $S/arena_exec.sh "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --headless $DEV_ARGS \
  --policy_type ${POLICY_TYPE:-arena_spike_policies.TimedGr00tRemotePolicy} \
  --policy_config_yaml_path $CFG --remote_host 127.0.0.1 --remote_port $PORT \
  --num_envs ${NUM_ENVS:-1} --enable_cameras $* $TASK"
echo "[run_eval] $tag wall=$(( $(date +%s) - t0 ))s"
