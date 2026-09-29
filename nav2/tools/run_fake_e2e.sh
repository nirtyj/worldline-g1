#!/usr/bin/env bash
# Nav2 end-to-end on the FAKES (no Isaac, no SONIC): fake P1 (kinematic G1, real house occupancy) + fake deploy,
# the real wl-body (NAV_BACKEND=nav2), the real ros_bridge + Nav2, BodyClient-only test, CPU sampling.
#
#   bash nav2/tools/run_fake_e2e.sh [--port-offset 700] [--controller rpp|mppi] [--house-dir DIR] [--tests ...] [--out DIR]
#
# tmux sessions nav2-fakes (fakes + body + cpu) and nav2-e2e (bridge + Nav2); both are removed at the end.
# Output: outputs/m1/nav2/fake-<controller>-<ts>/ (metrics.json, trajectory.png, goal_*.png, cancel.png, watchdog.png,
# cpu.csv, cpu_summary.json, body/ planner_cmds.jsonl, nav2/ cmd_vel.jsonl goals.jsonl nav2.log).
set -euo pipefail
WL=${WL:-/work/worldline-g1}
OFFSET=700; CONTROLLER=rpp; HOUSE=$WL/assets/houses/procthor-train-38; TESTS=goals,unreachable,cancel,timeout,escape,watchdog,stuck; OUT=""
EXTRA=""; CPUX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-offset) OFFSET="$2"; shift 2;;
    --controller) CONTROLLER="$2"; shift 2;;
    --house-dir) HOUSE="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --no-composition) EXTRA="$EXTRA --no-composition"; CPUX="--per-process"; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
OUT=${OUT:-$WL/outputs/m1/nav2/fake-$CONTROLLER-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
PY=$WL/.venv/bin/python
cd "$WL"
for s in nav2-fakes nav2-e2e; do tmux has-session -t "=$s" 2>/dev/null && { echo "session $s exists"; exit 1; }; done
# preflight: EVERY port of this offset must be free, or we would talk to someone else's P1/body (the bridge only
# connects to P1 and the body, so a busy P1 port would not fail by itself)
for base in 5556 5557 5565 5600 5601 5602 5610 5611 5620 5690; do
  p=$((base + OFFSET))
  if ss -ltn "sport = :$p" | grep -q LISTEN; then echo "[run_fake_e2e] port $p (offset $OFFSET) is in use: pick another --port-offset" >&2; exit 1; fi
done
cleanup() {
  bash "$WL/nav2/down.sh" --session nav2-e2e >/dev/null 2>&1 || true
  tmux send-keys -t "=nav2-fakes:cpu" C-c 2>/dev/null || true
  tmux send-keys -t "=nav2-fakes:body" C-c 2>/dev/null || true
  tmux send-keys -t "=nav2-fakes:fakes" C-c 2>/dev/null || true
  sleep 2
  tmux kill-session -t "=nav2-fakes" 2>/dev/null || true
}
trap cleanup EXIT
tmux new-session -d -s nav2-fakes -n fakes -x 200 -y 50 "bash --noprofile --norc"
tmux send-keys -t "=nav2-fakes:fakes" "cd $WL && exec $PY -u -m nav2.tools.fake_stack --port-offset $OFFSET --house-dir $HOUSE --out $OUT/fakes 2>&1 | tee $OUT/fakes.log" C-m
tmux new-window -t "=nav2-fakes" -n cpu "bash --noprofile --norc"
tmux send-keys -t "=nav2-fakes:cpu" "cd $WL && exec python3 nav2/tools/cpu_monitor.py --out $OUT/cpu.csv --period 1.0 $CPUX --group nav2=component_container_isolated,lib/nav2_,lib/nav2_lifecycle_manager --group bridge=nav2/ros_bridge.py --group launch=jazzy/bin/ros2 --group body=body.service --group fakes=nav2.tools.fake_stack" C-m
tmux new-window -t "=nav2-fakes" -n body "bash --noprofile --norc"
tmux send-keys -t "=nav2-fakes:body" "cd $WL && NAV_BACKEND=nav2 exec $PY -u -m body.service --port-offset $OFFSET --log-dir $OUT/body 2>&1 | tee $OUT/body.log" C-m
sleep 2
bash nav2/up.sh --port-offset "$OFFSET" --session nav2-e2e --controller "$CONTROLLER" --log-dir "$OUT/nav2" --wait 90 $EXTRA
set +e
$PY -u -m nav2.tools.nav2_fake_test --port-offset "$OFFSET" --out "$OUT" --body-log "$OUT/body" --bridge-log "$OUT/nav2" \
    --house-dir "$HOUSE" --tests "$TESTS" 2>&1 | tee "$OUT/test.log"
RC=${PIPESTATUS[0]}
sleep 5   # idle CPU tail
python3 nav2/tools/cpu_summary.py --cpu "$OUT/cpu.csv" --phases "$OUT/phases.json" --out "$OUT/cpu_summary.json" || true
echo "[run_fake_e2e] rc=$RC out=$OUT"
exit $RC
