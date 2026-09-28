#!/usr/bin/env bash
# Body-in-the-loop end-to-end run with the REAL wl-isaac (P1) + the REAL unmodified gear_sonic_deploy (P2) + wl-body
# (P3), isolated in a private network namespace: the namespace has its own lo, so DDS domain 0 (hard-coded in the
# deploy) and the contract ports 5556/5557/5565/5600/5601/5610/5611 are invisible to everyone else on the box
# (same idea as sim_isaac/tools/sonic_netns_test.sh). Inside it runs exactly what the integrated stack runs:
#   scripts/m1_up.sh  ->  tools/m1_drive_test.py (BodyClient only, GT assertions)  ->  scripts/m1_down.sh
# with a private tmux server (TMUX_TMPDIR) so every process lives in the namespace.
#
#   bash scripts/body_netns_e2e.sh [HOUSE] [STAND_S] [TESTS]
# Output: /work/worldline-g1/outputs/m1/body/netns-<house>-<ts>/ (m1_up.log, drive.log, metrics.json, mp4s, pngs)
set -uo pipefail
HOUSE=${1:-procthor-train-40}
STAND_S=${2:-60}
TESTS=${3:-stand,turn,walk,strafe,stop,goto}
NS=${NS:-wlbody}
SESSION=body-e2e
WL=/work/worldline-g1
TS=$(date +%Y%m%d-%H%M%S)
OUT=$WL/outputs/m1/body/netns-$HOUSE-$TS
ME=$(id -un)
TMUXD=/scratch/tmp/wlbody-tmux-$TS
mkdir -p "$OUT" "$TMUXD" && chmod 700 "$TMUXD"
say() { echo "[body_netns_e2e $(date +%H:%M:%S)] $*" | tee -a "$OUT/e2e.log"; }

if sudo -n ip netns list | grep -qw "$NS"; then say "netns $NS already exists (another run?): refusing"; exit 1; fi
sudo -n ip netns add "$NS" || { say "cannot create netns"; exit 1; }
sudo -n ip -n "$NS" link set lo up
ns() { sudo -n ip netns exec "$NS" sudo -n -u "$ME" env HOME="$HOME" PATH="$PATH" TMUX_TMPDIR="$TMUXD" \
       DEPLOY_FORCE=1 DEPLOY_TASKSET="${DEPLOY_TASKSET:-0-3}" ISAAC_TASKSET="${ISAAC_TASKSET:-4-15}" \
       BODY_TASKSET="${BODY_TASKSET:-4-15}" "$@"; }

cleanup() {
  say "cleanup"
  ns bash "$WL/scripts/m1_down.sh" --session "$SESSION" > "$OUT/m1_down.log" 2>&1
  ns tmux kill-server 2>/dev/null   # the private server in $TMUXD only
  for p in $(sudo -n ip netns pids "$NS" 2>/dev/null); do sudo -n kill -INT "$p" 2>/dev/null; done
  sleep 3
  for p in $(sudo -n ip netns pids "$NS" 2>/dev/null); do sudo -n kill -KILL "$p" 2>/dev/null; done
  sudo -n ip netns del "$NS" 2>/dev/null || true
  cp /work/logs/wl/$SESSION-*-{isaac,deploy,body}.log "$OUT/" 2>/dev/null || true
  say "done: $OUT"
}
trap cleanup EXIT

say "house=$HOUSE stand_s=$STAND_S tests=$TESTS ns=$NS out=$OUT"
{ uptime; ps -eo pcpu,etime,cmd --sort=-pcpu | head -12; nvidia-smi --query-compute-apps=pid,used_memory,name \
    --format=csv,noheader; } > "$OUT/box_load_before.txt" 2>&1
ns bash "$WL/scripts/m1_up.sh" --house "$HOUSE" --session "$SESSION" --isaac-args "${ISAAC_ARGS:-}" > "$OUT/m1_up.log" 2>&1
rc=$?
say "m1_up exit $rc: $(tail -2 "$OUT/m1_up.log" | tr '\n' ' ' | cut -c1-300)"
if [[ $rc != 0 ]]; then exit 1; fi
ns bash -c "cd $WL && .venv/bin/python -u -m tools.m1_drive_test --port-offset 0 --stand-s $STAND_S --tests $TESTS \
    --out $OUT" > "$OUT/drive.log" 2>&1
say "drive test exit $?: $(grep RESULT "$OUT/drive.log" | cut -c1-300)"
