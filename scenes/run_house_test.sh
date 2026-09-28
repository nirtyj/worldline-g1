#!/usr/bin/env bash
# Run scenes.test_house for one or more houses on the box (headless Isaac Sim 5.1).
#   bash scenes/run_house_test.sh [--tag T] [--dynamic-objects keep|kinematic] HOUSE...
# Logs: /work/logs/wl/house-test-<house>[-<tag>].log ; evidence: /work/worldline-g1/outputs/m1/house/
set -uo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true
export PYTHONUNBUFFERED=1
PY=/work/envs/isaaclab/bin/python
WL=/work/worldline-g1
cd "$WL"
EXTRA=(); TAG=""
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --tag) TAG=$2; EXTRA+=(--tag "$2"); shift 2;;
    *) EXTRA+=("$1" "$2"); shift 2;;
  esac
done
[[ $# -gt 0 ]] || set -- procthor-train-40
rc_all=0
for h in "$@"; do
  log=/work/logs/wl/house-test-$h${TAG:+-$TAG}.log
  echo "== $h -> $log"
  timeout 1800 "$PY" -m scenes.test_house --house "$h" "${EXTRA[@]}" > "$log" 2>&1
  rc=$?; echo "   rc=$rc"; [[ $rc -eq 0 ]] || rc_all=$rc
  grep -E "^\[house\]" "$log" | tail -4
done
exit $rc_all
