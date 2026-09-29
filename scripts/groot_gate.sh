#!/usr/bin/env bash
# SONIC's timing gate with the GR00T client active (tools/groot_timing_gate.py), run the way P5 runs the client: pinned
# off the SONIC deploy's cores (--cpus, default 4-15) and with its CPU thread pools capped (OMP/OpenBLAS/MKL = 1). Also
# records per-CPU load (mpstat, 5 s) next to the result. The main-box stack must be up (scripts/m2_up.sh) and the
# PolicyServer reachable (the OD3 link: scripts/groot_link.sh ensure). Hold the dev lock, then the main lock.
#
#   bash scripts/groot_gate.sh <tag> [tools.groot_timing_gate args...]
#   -> outputs/m2b_finish/gmain/timing-<ts>-<tag>/{timing_gate.json, run.log, mpstat.txt, trace_*.npz}
# Env: GATE_OUT (default outputs/m2b_finish/gmain), GATE_THREADS (default 1), WL (default /work/worldline-g1).
set -euo pipefail
WL=${WL:-/work/worldline-g1}
tag=${1:?usage: scripts/groot_gate.sh <tag> [gate args...]}; shift
cd "$WL"
O=${GATE_OUT:-outputs/m2b_finish/gmain}/timing-$(date +%Y%m%d-%H%M%S)-$tag
mkdir -p "$O"
T=${GATE_THREADS:-1}
MP=""
if command -v mpstat >/dev/null 2>&1; then
  mpstat -P ALL 5 > "$O/mpstat.txt" 2>/dev/null & MP=$!
fi
trap '[[ -n "$MP" ]] && kill "$MP" 2>/dev/null || true' EXIT
OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T WL_PORT_OFFSET=${WL_PORT_OFFSET:-0} \
  .venv-rt/bin/python -m tools.groot_timing_gate --out "$O" --tag "$tag" "$@" 2>&1 | tee "$O/run.log"
echo "DONE $O"
