#!/usr/bin/env bash
# Drive the Arena G1 lower-body WBC with pure velocity / height commands (no GR00T) in the Galileo loco-manip scene.
#   bash run_veldrive.sh homie   -> --embodiment g1_wbc_joint       (HOMIE v2 stand.onnx / walk.onnx)
#   bash run_veldrive.sh agile   -> --embodiment g1_wbc_agile_joint (WBC-AGILE recurrent student onnx)
# Schedule (50 Hz steps): stand 1 s, back up 2 s, turn right ~100 deg, walk forward 4 s, strafe left 2 s,
# squat to 0.6 m, stand back up, stop.
set -euo pipefail
which=${1:?homie|agile}
case $which in
  homie) EMB=g1_wbc_joint;;
  agile) EMB=g1_wbc_agile_joint;;
esac
SCHED="${SCHED:-50:0,0,0,0.75;100:-0.3,0,0,0.75;175:0,0,-0.5,0.75;200:0.4,0,0,0.75;100:0,0.2,0,0.75;100:0,0,0,0.75;100:0,0,0,0.6;100:0,0,0,0.75;75:0,0,0,0.75}"
tag=veldrive_$which
S=/work/arena/spike
export SPIKE_CAM=${SPIKE_CAM:-1.2,-2.6,0.7,0.0,-0.4,-0.3;0.3,-0.2,2.2,0.0,-0.6,-0.8}
t0=$(date +%s)
SPIKE_TAG=$tag bash $S/arena_exec.sh "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py --headless \
  --policy_type arena_spike_policies.VelocityCommandPolicy --vel_schedule '$SCHED' \
  --num_envs 1 --enable_cameras galileo_g1_locomanip_pick_and_place --object brown_box --embodiment $EMB"
echo "[run_veldrive] $tag wall=$(( $(date +%s) - t0 ))s"
