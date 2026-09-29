#!/usr/bin/env bash
# Nav2 end-to-end on the FAKES (no Isaac, no SONIC): fake P1 (kinematic G1, real house occupancy) + fake deploy,
# the real wl-body (NAV_BACKEND=nav2), the real ros_bridge + Nav2, BodyClient-only test, CPU sampling.
#
#   bash nav2/tools/run_fake_e2e.sh [--port-offset 700] [--controller rpp|mppi] [--house-dir DIR] [--tests ...]
#        [--goal-plan std|passage] [--ki "0.4,0.0"] [--out DIR] [--no-composition]
#
# --ki: the body's heading_bias_ki values (nav2/tools/fake_body.py, env WL_FAKE_HEADING_BIAS_KI; empty = the
# body/config.py default). The body is restarted (and stood up again) for each value; every value runs the goals test, the LAST value
# also runs the rest of --tests (stuck last: its spawned obstacle stays in the fake P1). Output per value:
# OUT/ki<value>/ (metrics.json, trajectory.png, goal_*.png, body/ planner_cmds.jsonl ...); shared: OUT/nav2/
# (cmd_vel.jsonl goals.jsonl nav2.log), cpu.csv, cpu_summary.json, summary.json.
# Env: WL_RUN_PREFIX (e.g. "nice -n 19 taskset -c 12-15") is prepended to every process started here; NAV2_NICE /
# NAV2_TASKSET do the same for the bridge + Nav2 (nav2/up.sh). tmux sessions nav2-fakes-<offset> (fakes, body,
# cpu) and nav2-e2e-<offset> (bridge + Nav2) are removed at the end. ROS domain: 42 + offset // 100 (nav2/up.sh).
set -euo pipefail
WL=${WL:-/work/worldline-g1}
OFFSET=700; CONTROLLER=rpp; HOUSE=$WL/assets/houses/procthor-train-38; TESTS=goals,unreachable,cancel,timeout,escape,watchdog,stuck; OUT=""
EXTRA=""; CPUX=""; KIS=""; GOAL_PLAN=std
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port-offset) OFFSET="$2"; shift 2;;
    --controller) CONTROLLER="$2"; shift 2;;
    --house-dir) HOUSE="$2"; shift 2;;
    --tests) TESTS="$2"; shift 2;;
    --goal-plan) GOAL_PLAN="$2"; shift 2;;
    --ki) KIS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --no-composition) EXTRA="$EXTRA --no-composition"; CPUX="--per-process"; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
OUT=${OUT:-$WL/outputs/m1/nav2/fake-$CONTROLLER-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
PY=$WL/.venv/bin/python
RUN=${WL_RUN_PREFIX:-}
FS=nav2-fakes-$OFFSET; ES=nav2-e2e-$OFFSET
cd "$WL"
for s in $FS $ES; do tmux has-session -t "=$s" 2>/dev/null && { echo "session $s exists"; exit 1; }; done
# preflight: EVERY port of this offset must be free, or we would talk to someone else's P1/body (the bridge only
# connects to P1 and the body, so a busy P1 port would not fail by itself)
for base in 5556 5557 5565 5600 5601 5602 5610 5611 5620 5690; do
  p=$((base + OFFSET))
  if ss -ltn "sport = :$p" | grep -q LISTEN; then echo "[run_fake_e2e] port $p (offset $OFFSET) is in use: pick another --port-offset" >&2; exit 1; fi
done
say() { echo "[run_fake_e2e $(date +%H:%M:%S)] $*"; }
stop_body() {
  tmux send-keys -t "=$FS:body" C-c 2>/dev/null || true
  for _ in $(seq 1 40); do
    p=$(tmux list-panes -t "=$FS:body" -F '#{pane_pid}' 2>/dev/null || true)
    [[ -z "$p" ]] && return 0
    pgrep -P "$p" >/dev/null || return 0
    sleep 0.25
  done
}
start_body() {   # $1 = ki ('' = config default), $2 = log dir
  tmux send-keys -t "=$FS:body" "cd $WL && WL_FAKE_HEADING_BIAS_KI='$1' NAV_BACKEND=nav2 exec $RUN $PY -u -m nav2.tools.fake_body --port-offset $OFFSET --log-dir $2/body 2>&1 | tee $2/body.log" C-m
  for _ in $(seq 1 60); do        # until the body answers ping on its ROUTER
    $RUN $PY - "$OFFSET" <<'PYEOF' >/dev/null 2>&1 && return 0
import sys
from body.client import BodyClient
c = BodyClient(port_offset=int(sys.argv[1]))
c.connect(1.0)
sys.exit(0 if c.request("ping").get("ok") else 1)
PYEOF
    sleep 0.5
  done
  return 1
}
cleanup() {
  bash "$WL/nav2/down.sh" --session "$ES" >/dev/null 2>&1 || true
  tmux send-keys -t "=$FS:cpu" C-c 2>/dev/null || true
  stop_body
  tmux send-keys -t "=$FS:fakes" C-c 2>/dev/null || true
  sleep 2
  tmux kill-session -t "=$FS" 2>/dev/null || true
}
trap cleanup EXIT
tmux new-session -d -s "$FS" -n fakes -x 200 -y 50 "bash --noprofile --norc"
tmux send-keys -t "=$FS:fakes" "cd $WL && exec $RUN $PY -u -m nav2.tools.fake_stack --port-offset $OFFSET --house-dir $HOUSE --out $OUT/fakes 2>&1 | tee $OUT/fakes.log" C-m
tmux new-window -t "=$FS" -n cpu "bash --noprofile --norc"
tmux send-keys -t "=$FS:cpu" "cd $WL && exec $RUN python3 nav2/tools/cpu_monitor.py --out $OUT/cpu.csv --period 1.0 $CPUX --session $FS --session $ES --group nav2=component_container_isolated,lib/nav2_,lib/nav2_lifecycle_manager --group bridge=nav2/ros_bridge.py --group launch=jazzy/bin/ros2 --group body=nav2.tools.fake_body --group fakes=nav2.tools.fake_stack" C-m
tmux new-window -t "=$FS" -n body "bash --noprofile --norc"
IFS=',' read -r -a KI_LIST <<< "${KIS:-}"
[[ ${#KI_LIST[@]} -gt 0 ]] || KI_LIST=("")
tag() { [[ -n "$1" ]] && echo "ki$1" || echo "default"; }
FIRST=$OUT/$(tag "${KI_LIST[0]}"); mkdir -p "$FIRST"
start_body "${KI_LIST[0]}" "$FIRST" || { say "body did not come up"; exit 1; }
bash nav2/up.sh --port-offset "$OFFSET" --session "$ES" --controller "$CONTROLLER" --log-dir "$OUT/nav2" --wait 90 $EXTRA
set +e
RC=0; PHASES=()
for i in "${!KI_LIST[@]}"; do
  ki=${KI_LIST[$i]}; D=$OUT/$(tag "$ki"); mkdir -p "$D"
  if (( i > 0 )); then
    stop_body; start_body "$ki" "$D" || { say "body did not come up (ki=$ki)"; RC=1; break; }
  fi
  T=goals; (( i == ${#KI_LIST[@]} - 1 )) && T=$TESTS
  say "heading_bias_ki=${ki:-default}: tests $T -> $D"
  $RUN $PY -u -m nav2.tools.nav2_fake_test --port-offset "$OFFSET" --out "$D" --body-log "$D/body" --bridge-log "$OUT/nav2" \
      --house-dir "$HOUSE" --tests "$T" --goal-plan "$GOAL_PLAN" --force-stand \
      --label "FAKE P1 + FAKE deploy, heading_bias_ki=${ki:-default}" 2>&1 | tee "$D/test.log"
  r=${PIPESTATUS[0]}; (( r != 0 )) && RC=$r
  PHASES+=("$D/phases.json")
done
sleep 5   # idle CPU tail
$RUN python3 - "$OUT" "${PHASES[@]}" <<'PYEOF' || true
import json, sys
out, files = sys.argv[1], sys.argv[2:]
ph = []
for f in files:
    try:
        ph += json.load(open(f))
    except OSError:
        pass
json.dump(ph, open(f"{out}/phases.json", "w"), indent=1)
PYEOF
$RUN python3 nav2/tools/cpu_summary.py --cpu "$OUT/cpu.csv" --phases "$OUT/phases.json" --out "$OUT/cpu_summary.json" || true
$RUN python3 - "$OUT" <<'PYEOF' || true
import glob, json, os, sys
out = sys.argv[1]
summ = {"runs": {}}
for m in sorted(glob.glob(f"{out}/*/metrics.json")):
    d = json.load(open(m))
    g = d.get("goals") or {}
    summ["runs"][os.path.basename(os.path.dirname(m))] = {
        "label": d.get("label"), "pass": d.get("pass"),
        "tests": {k: (v.get("pass") if isinstance(v, dict) else None) for k, v in d.items()
                  if isinstance(v, dict) and "pass" in v},
        "goals": [(x["k"], x["room"], x["state"], x["reason"], x["gt_pos_err"], x.get("through_passage"),
                   x.get("duration_s")) for x in g.get("goals", [])]}
allg = [x for r in summ["runs"].values() for x in r["goals"]]
summ["goal_attempts"] = len(allg)
summ["goals_succeeded"] = sum(1 for x in allg if x[2] == "succeeded")
summ["through_passage"] = sum(1 for x in allg if x[5])
summ["through_passage_succeeded"] = sum(1 for x in allg if x[5] and x[2] == "succeeded")
json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "runs"}))
PYEOF
say "rc=$RC out=$OUT"
exit $RC
