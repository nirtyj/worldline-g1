#!/usr/bin/env bash
# Run wl-isaac (P1) on the box with the Isaac Lab env. Extra args go to sim_isaac.app.
#   bash sim_isaac/tools/run_app.sh --house empty --dds-domain 7 --port-offset 100
set -euo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR:-/tmp}"
exec /work/envs/isaaclab/bin/python -m sim_isaac.app "$@"
