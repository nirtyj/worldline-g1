#!/usr/bin/env bash
# One full M1 cycle on the host, from cold, with all evidence in one run dir (E1-E6):
#   scripts/m1_up.sh  ->  tools/m1_drive_test.py (BodyClient only, GT assertions)  ->  scripts/m1_down.sh
# on the CONTRACT ports and DDS domain 0 on lo (tmux session wl-m1). Refuses to start if the stack is already up.
#
#   bash scripts/m1_e2e.sh [--house ID] [--stand-s 60] [--tests stand,turn,walk,strafe,stop,goto] [--tp-camera]
#
# Output: /work/worldline-g1/outputs/m1/run-<ts>/ : m1_up.log, drive_test.log, m1_down.log (+ timing), metrics.json,
# head_camera.mp4, topdown.mp4, [third_person.mp4], trajectory.png, gait.png, pose.csv, events.jsonl, p1_record.npz,
# debug_legs.npz, p1_stats_final.json, logs/ (P1, deploy, body), box_load_{before,after}.txt, e2e.json.
set -uo pipefail
source /etc/profile.d/ludo.sh 2>/dev/null || true
WL=${WL:-/work/worldline-g1}
PY_BODY=$WL/.venv/bin/python
HOUSE=${HOUSE:-procthor-train-38}; STAND_S=60; TESTS=stand,turn,walk,strafe,stop,goto; TP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --house) HOUSE="$2"; shift 2;;
    --stand-s) STAND_S="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --tp-camera) TP=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
TS=$(date +%Y%m%d-%H%M%S)
OUT=$WL/outputs/m1/run-$TS
mkdir -p "$OUT/logs"
say() { echo "[m1_e2e $(date +%H:%M:%S)] $*" | tee -a "$OUT/e2e.log"; }
load() { { date -Is; uptime; ps -eo pcpu,etime,args --sort=-pcpu | head -15 | cut -c1-200;
           nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader; } > "$1" 2>&1; }

say "house=$HOUSE stand_s=$STAND_S tests=$TESTS tp_camera=$TP out=$OUT"
load "$OUT/box_load_before.txt"
ISAAC_ARGS_E2E=${ISAAC_ARGS:-}
[[ "$TP" == 1 ]] && ISAAC_ARGS_E2E="$ISAAC_ARGS_E2E --tp-camera --tp-hz 10"
t0=$(date +%s.%N)
HOUSE=$HOUSE ISAAC_ARGS="$ISAAC_ARGS_E2E" bash "$WL/scripts/m1_up.sh" --house "$HOUSE" > "$OUT/m1_up.log" 2>&1
rc_up=$?
t1=$(date +%s.%N)
say "m1_up exit $rc_up after $(printf %.1f "$(echo "$t1 - $t0" | bc)") s"
STACK=$(readlink -f "$WL/outputs/m1/stack-latest-wl-m1")
rc_drive=99
if [[ $rc_up == 0 ]]; then
  (cd "$WL" && "$PY_BODY" -u -m tools.m1_drive_test --port-offset 0 --stand-s "$STAND_S" --tests "$TESTS" \
      --out "$OUT") > "$OUT/drive_stdout.log" 2>&1
  rc_drive=$?
  say "drive test exit $rc_drive: $(grep RESULT "$OUT/drive_stdout.log" | cut -c1-400)"
  (cd "$WL" && "$PY_BODY" -m tools.body_cli p1 get_stats) > "$OUT/p1_stats_end.json" 2>/dev/null
fi
load "$OUT/box_load_after.txt"
BODY_PID=$(pgrep -f "body.service --port-offset 0 --log-dir $STACK/body" | head -1)
t2=$(date +%s.%N)
bash "$WL/scripts/m1_down.sh" > "$OUT/m1_down.log" 2>&1
rc_down=$?
t3=$(date +%s.%N)
sleep 1
# the stack's own processes: P1 pid (ping reply in m1_up.log), deploy pid ("deploy ready ... (pid N)"), body pid
P1_PID=$(grep -o '"pid": [0-9]*' "$OUT/m1_up.log" | head -1 | grep -o '[0-9]*$')
DEP_PID=$(grep -o 'Init Done after [0-9]*s (pid [0-9]*' "$OUT/m1_up.log" | grep -o '[0-9]*$')
leftover=""
for p in $P1_PID $DEP_PID $BODY_PID; do kill -0 "$p" 2>/dev/null && leftover="$leftover $p"; done
bound=$(ss -ltn | grep -cE ":(5556|5557|5565|5600|5601|5602|5610|5611) ")
say "stack pids: p1=$P1_PID deploy=$DEP_PID body=${BODY_PID:-?}"
say "m1_down exit $rc_down after $(printf %.1f "$(echo "$t3 - $t2" | bc)") s; leftover pids: '${leftover}'; contract ports still bound: $bound"
for f in isaac_log deploy_log body_log; do
  p=$(cat "$STACK/$f" 2>/dev/null) && [[ -f "$p" ]] && cp "$p" "$OUT/logs/"
done
cp "$STACK/stand.json" "$OUT/" 2>/dev/null
cp "$STACK/p1_stats_final.json" "$OUT/" 2>/dev/null
cp -r "$STACK/body" "$OUT/logs/body_service" 2>/dev/null
cat > "$OUT/e2e.json" <<EOF
{"ts": "$TS", "house": "$HOUSE", "tp_camera": $TP, "stack_dir": "$STACK",
 "m1_up": {"exit": $rc_up, "seconds": $(echo "$t1 - $t0" | bc)},
 "drive_test": {"exit": $rc_drive},
 "m1_down": {"exit": $rc_down, "seconds": $(echo "$t3 - $t2" | bc), "leftover_pids": "${leftover}", "ports_still_bound": $bound}}
EOF
say "done: $OUT"
[[ $rc_up == 0 && $rc_drive == 0 && $rc_down == 0 && -z "${leftover// /}" && $bound == 0 ]]
