#!/usr/bin/env bash
# SONIC-in-the-loop smoke test of wl-isaac inside a private network namespace.
#
# The deploy's DDS domain is hard-coded to 0 (sonic/run_deploy.sh notes g1_deploy_onnx_ref.cpp:2218) and the deploy
# agent owns domain 0 on the host's lo. A network namespace has its own lo, so DDS domain 0 and the contract ZMQ
# ports (5556/5557/5600/5601/5565) inside it are invisible to everyone else on the box.
#
#   bash sim_isaac/tools/sonic_netns_test.sh [HOUSE] [STAND_S] [WALK_S]
# Output: /work/worldline-g1/outputs/m1/isaac/sonic_smoke/ (report.json, trace.npz, ego.mp4, topdown.mp4, logs)
set -uo pipefail
HOUSE=${1:-procthor-train-40}
STAND_S=${2:-60}
WALK_S=${3:-8}
NS=wlisaac
WL=/work/worldline-g1
OUT=$WL/outputs/m1/isaac/sonic_smoke
LOGS=/work/logs/wl
PY=/work/envs/isaaclab/bin/python
DUR=$((STAND_S + WALK_S + 400))
mkdir -p "$OUT" "$LOGS"
rm -f "$OUT"/*.log "$LOGS/isaac-ns-app.log" "$LOGS/isaac-ns-deploy.log" "$LOGS/isaac-ns-smoke.log"

sudo -n ip netns add $NS 2>/dev/null || true
sudo -n ip -n $NS link set lo up
ns() { sudo -n ip netns exec $NS sudo -n -u "$USER" "$@"; }

cleanup() {
  echo "cleanup"
  for p in $(sudo -n ip netns pids $NS 2>/dev/null); do sudo -n kill -INT "$p" 2>/dev/null; done
  sleep 3
  for p in $(sudo -n ip netns pids $NS 2>/dev/null); do sudo -n kill -KILL "$p" 2>/dev/null; done
  sudo -n ip netns del $NS 2>/dev/null || true
}
trap cleanup EXIT

# 1. P1 (domain 0, contract ports, inside the namespace)
ns bash -c "source /etc/profile.d/ludo.sh; cd $WL; export PYTHONUNBUFFERED=1; exec $PY -m sim_isaac.app \
  --house $HOUSE --dds-domain 0 --dds-iface lo --camera 640x480 --camera-hz 30 --duration $DUR \
  --stats-every 10 --stats-out $OUT/p1_stats.json --out-dir $OUT" > "$LOGS/isaac-ns-app.log" 2>&1 &
timeout 400 bash -c "until grep -q -E 'WL_ISAAC_READY|Traceback' $LOGS/isaac-ns-app.log; do sleep 2; done"
grep -q WL_ISAAC_READY "$LOGS/isaac-ns-app.log" || { tail -30 "$LOGS/isaac-ns-app.log"; exit 1; }
echo "P1 ready"

# 2. driver (binds 5556 before the deploy connects) + video recorders
ns bash -c "source /etc/profile.d/ludo.sh; cd $WL; export PYTHONUNBUFFERED=1; exec $PY -m sim_isaac.tools.sonic_smoke \
  --deploy-log $LOGS/isaac-ns-deploy.log --stand-s $STAND_S --walk-s $WALK_S --out $OUT/report.json" \
  > "$LOGS/isaac-ns-smoke.log" 2>&1 &
smoke=$!
ns bash -c "source /etc/profile.d/ludo.sh; cd $WL; exec $PY -m sim_isaac.tools.record_video --port-offset 0 \
  --seconds $((STAND_S + WALK_S + 120)) --ego --topdown --out-dir $OUT" > "$OUT/record_video.log" 2>&1 &
sleep 2

# 3. the unmodified deploy (zmq_manager, sim, lo) via the deploy agent's launcher; stdin never delivers 'o'
ns bash -c "sleep 100000 | bash $WL/sonic/run_deploy.sh fg --force --session isaac-ns-sonic \
  --log $LOGS/isaac-ns-deploy.log" > "$OUT/deploy_stdout.log" 2>&1 &

wait $smoke
echo "smoke exit $?"
tail -5 "$LOGS/isaac-ns-smoke.log"
sleep 5
cp "$LOGS/isaac-ns-app.log" "$LOGS/isaac-ns-deploy.log" "$LOGS/isaac-ns-smoke.log" "$OUT/" 2>/dev/null
