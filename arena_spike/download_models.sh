#!/usr/bin/env bash
# Download the two Arena G1 GR00T checkpoints (inference files only: no optimizer / deepspeed states / ONNX blobs).
set -euo pipefail
source /etc/profile.d/ludo.sh
M=/work/arena/models/isaaclab_arena
HF="uvx --from huggingface_hub[cli,hf_xet] hf"
mkdir -p $M/locomanipulation_tutorial $M/static_apple_tutorial
echo "== N1.6 loco-manip (revision gn1_6)"
$HF download nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation --revision gn1_6 \
  --exclude 'global_step20000/*' --exclude 'optimizer.pt' \
  --local-dir $M/locomanipulation_tutorial/checkpoint-20000
echo "== N1.7 static apple (main)"
$HF download nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace \
  --exclude 'optimizer.pt' --exclude 'exports/*/onnx/*/*.onnx.data' \
  --local-dir $M/static_apple_tutorial/gn1x_tuned_static_apple
echo "== backbones"
$HF download nvidia/Cosmos-Reason2-2B || echo "Cosmos-Reason2-2B download FAILED"
$HF download nvidia/GR00T-N1.7-3B --include '*.json' --include '*.yaml' --include '*.md' || echo "N1.7 base meta FAILED"
du -sh $M/*/*
echo "== downloads done"
