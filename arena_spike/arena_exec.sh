#!/usr/bin/env bash
# Run a command inside the running Arena container as the host user, from the Arena repo root.
#   bash arena_exec.sh 'python isaaclab_arena/evaluation/policy_runner.py ...'
# -u <name> (not uid:gid) so the isaac-sim supplementary group applies. Use /isaac-sim/python.sh explicitly; PYTHONPATH gets /spike (arena_spike_policies.py).
set -euo pipefail
CMD="$*"
exec sudo docker exec -u "$(id -un)" -e HOME=/home/$(id -un) -e PYTHONPATH=/spike \
  -e PYTHONUNBUFFERED=1 -e SPIKE_TAG="${SPIKE_TAG:-run}" -e SPIKE_EVAL_DIR="${SPIKE_EVAL_DIR:-/eval}" \
  -e SPIKE_CAM="${SPIKE_CAM:-}" -e SPIKE_VIDEO="${SPIKE_VIDEO:-1}" -e SPIKE_VIDEO_EVERY="${SPIKE_VIDEO_EVERY:-2}" -e SPIKE_RERENDER="${SPIKE_RERENDER:-0}" -e SPIKE_BIN_POSE="${SPIKE_BIN_POSE:-}" -e SPIKE_WBC_RESET="${SPIKE_WBC_RESET:-0}" \
  -w /workspaces/isaaclab_arena arena bash -c "$CMD"
