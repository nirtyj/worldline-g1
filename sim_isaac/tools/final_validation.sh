#!/usr/bin/env bash
# wl-isaac (P1) final stand-alone validation, one Isaac instance at a time:
#   0. pure tests (joint map, math, band, pacer, DDS codec vs the library)
#   1. DDS stand test in a house on domain 7 / ports +100 with the stand peer (static stand lowcmd from a DDS peer,
#      lowstate vs ground truth, joint map by name, camera format, REP ops)
#   2. RTF matrix (E6) on the final code: bare / +camera / +camera+house, CPU vs GPU PhysX, paced vs free-running
#   3. SONIC-in-the-loop smoke test (unmodified deploy) in a private network namespace: stand, walk, turn
#     bash sim_isaac/tools/final_validation.sh [OUT_ROOT] [HOUSE]
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh
cd "$(dirname "$0")/../.."
ROOT=${1:-/work/worldline-g1/outputs/m1/isaac/final-$(date +%Y%m%d-%H%M%S)}
HOUSE=${2:-procthor-train-40}
PY=/work/envs/isaaclab/bin/python
mkdir -p "$ROOT"
echo "== final validation -> $ROOT (house $HOUSE)"
{ date; uptime; nproc; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader; git -C /work/worldline-g1 log -1 --oneline 2>/dev/null
  md5sum sim_isaac/*.py; } > "$ROOT/env.txt" 2>&1

echo "== 0. pure tests"
$PY -m pytest -q sim_isaac/tests > "$ROOT/pytest.log" 2>&1; echo "pytest exit $?" | tee -a "$ROOT/pytest.log"

echo "== 1. DDS stand test ($HOUSE, domain 7, +100)"
bash sim_isaac/tools/bg_app.sh final --house "$HOUSE" --dds-domain 7 --port-offset 100 --duration 400 \
  --stats-out "$ROOT/stand_app_stats.json" --out-dir "$ROOT/stand"
$PY -m sim_isaac.tools.stand_peer --domain 7 --port-offset 100 --hold-s 60 --camera --joint-map --ops \
  --gains stiff --out "$ROOT/stand_house.json" > "$ROOT/stand_peer.log" 2>&1
echo "stand_peer exit $?"; tail -3 "$ROOT/stand_peer.log"
$PY - <<'EOF'
import json, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 20000); s.connect("tcp://127.0.0.1:5700")
s.send(b'{"op":"shutdown"}'); print(s.recv()[:200])
EOF
timeout 60 bash -c 'until grep -q EXIT= /work/logs/wl/isaac-final.log; do sleep 2; done'
cp /work/logs/wl/isaac-final.log "$ROOT/stand_app.log"

echo "== 2. RTF matrix"
bash sim_isaac/tools/measure_rtf.sh "$ROOT/rtf" 60 bare_cpu_rt bare_gpu_rt cam30_cpu_rt house_cam30_cpu_rt \
  house_cam30_cpu_free house_cam30_gpu_rt house_cam15_cpu_rt > "$ROOT/rtf.log" 2>&1
cat "$ROOT/rtf/rtf_summary.md"

echo "== 3. SONIC netns smoke"
OUT="$ROOT/sonic_smoke" bash sim_isaac/tools/sonic_netns_test.sh "$HOUSE" 60 8 10 > "$ROOT/sonic_smoke.log" 2>&1
echo "sonic smoke exit $?"; tail -3 "$ROOT/sonic_smoke.log"
echo "FINAL_VALIDATION_DONE $ROOT"
