#!/usr/bin/env bash
# RTF matrix for wl-isaac (E6). Each config: start the app (DDS domain 7, ports +100), generate deploy-like DDS
# traffic (lowcmd 500 Hz + Dex3 cmds) with the stand peer, collect the app's final stats JSON.
#   bash sim_isaac/tools/measure_rtf.sh [OUT_DIR] [DURATION_S] [CONFIG_NAME ...]
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh
cd "$(dirname "$0")/../.."
OUT=${1:-/work/worldline-g1/outputs/m1/isaac/rtf}
DUR=${2:-60}
shift 2 2>/dev/null || true
mkdir -p "$OUT"
PY=/work/envs/isaaclab/bin/python
HOUSE=${HOUSE:-procthor-train-40}
declare -A CFG=(
  [bare_cpu_rt]="--camera none --physx-device cpu"
  [bare_cpu_free]="--camera none --physx-device cpu --no-rt-pace"
  [bare_gpu_rt]="--camera none --physx-device cuda"
  [cam30_cpu_rt]="--camera 640x480 --camera-hz 30 --physx-device cpu"
  [cam30_cpu_free]="--camera 640x480 --camera-hz 30 --physx-device cpu --no-rt-pace"
  [cam30_gpu_rt]="--camera 640x480 --camera-hz 30 --physx-device cuda"
  [cam30_gpu_free]="--camera 640x480 --camera-hz 30 --physx-device cuda --no-rt-pace"
  [cam15_cpu_rt]="--camera 640x480 --camera-hz 15 --physx-device cpu"
  [cam30_perf_cpu_rt]="--camera 640x480 --camera-hz 30 --physx-device cpu --rendering_mode performance"
  [house_cam30_cpu_rt]="--house $HOUSE --camera 640x480 --camera-hz 30 --physx-device cpu"
  [house_cam30_cpu_free]="--house $HOUSE --camera 640x480 --camera-hz 30 --physx-device cpu --no-rt-pace"
  [house_cam30_gpu_rt]="--house $HOUSE --camera 640x480 --camera-hz 30 --physx-device cuda"
  [house_cam30_kin_cpu_rt]="--house $HOUSE --house-dynamic kinematic --camera 640x480 --camera-hz 30 --physx-device cpu"
  [house_cam15_cpu_rt]="--house $HOUSE --camera 640x480 --camera-hz 15 --physx-device cpu"
)
ORDER=(bare_cpu_rt bare_cpu_free bare_gpu_rt cam30_cpu_rt cam30_cpu_free cam30_gpu_rt cam30_gpu_free cam15_cpu_rt
       cam30_perf_cpu_rt house_cam30_cpu_rt house_cam30_cpu_free house_cam30_gpu_rt house_cam30_kin_cpu_rt
       house_cam15_cpu_rt)
[[ $# -gt 0 ]] && ORDER=("$@")
for name in "${ORDER[@]}"; do
  args=${CFG[$name]:-}
  [[ -z "$args" ]] && { echo "unknown config $name"; continue; }
  log="$OUT/$name.log"
  echo "=== $name: $args" | tee "$log"
  { echo "--- concurrent GPU apps before:"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
    uptime; } > "$OUT/$name.env.txt" 2>&1
  PYTHONUNBUFFERED=1 $PY -m sim_isaac.app $args --dds-domain 7 --port-offset 100 --duration "$DUR" \
      --stats-every 10 --stats-out "$OUT/$name.json" >> "$log" 2>&1 &
  app=$!
  for _ in $(seq 1 240); do grep -q -E "WL_ISAAC_READY|Traceback" "$log" && break; kill -0 $app 2>/dev/null || break; sleep 1; done
  if grep -q WL_ISAAC_READY "$log"; then
    $PY -m sim_isaac.tools.stand_peer --domain 7 --port-offset 100 --load-only "$((DUR - 2))" >> "$OUT/$name.peer.log" 2>&1 &
  fi
  wait $app
  echo "--- exit $?" >> "$log"
  { echo "--- concurrent GPU apps after:"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
    uptime; } >> "$OUT/$name.env.txt" 2>&1
  sleep 3
done
$PY -m sim_isaac.tools.summarize_rtf "$OUT"
