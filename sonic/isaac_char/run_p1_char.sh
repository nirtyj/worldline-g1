#!/usr/bin/env bash
# Planner-response characterisation of the UNMODIFIED SONIC deploy against wl-isaac (P1, Isaac Sim 5.1 / PhysX),
# the integration simulator, instead of the MuJoCo reference. Same driver as the MuJoCo loop
# (sonic/mujoco_ref/char_ref.py, --p1 backend: P1 REP API, docs/contracts/m1.md §1.6).
#
# Default: the host network namespace, DDS domain 0 on lo (the deploy agent's domain; the script refuses to start if
# another deploy runs in this namespace) and all ZMQ ports at contract + OFFSET (default 400: +100/+200/+300 are used
# by the isaac/body/viz agents' tests). --netns runs everything in a private namespace ("wldeploy", own lo, contract
# ports) like sim_isaac/tools/sonic_netns_test.sh; a netns has no network, so only local assets load there (houses:
# yes; --house empty: no, its ground plane is fetched from the Omniverse S3 bucket).
#
#   bash sonic/isaac_char/run_p1_char.sh [--house empty|ID] [--sections LIST] [--tag T] [--camera 640x480|none]
#        [--offset N] [--netns] [-- extra char_ref.py args]
# Output: /work/worldline-g1/outputs/m1/deploy/<tag>-<ts>/ {p1.log, deploy.log, drive.log, drive_result.json,
#   drive_trace.jsonl, char.json, deploy_g1_debug.jsonl, p1_stats.json, trace.npz (P1 record)}
set -uo pipefail
source "$(dirname "$0")/../sonic_env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOUSE=empty; SECTIONS="turns,gentle"; TAG=p1char; CAMERA=640x480; EXTRA=(); OFFSET=400; USE_NS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --house) HOUSE="$2"; shift 2;;
    --sections) SECTIONS="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --camera) CAMERA="$2"; shift 2;;
    --offset) OFFSET="$2"; shift 2;;
    --netns) USE_NS=1; OFFSET=0; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
NS=wldeploy
PY_ISAAC=/work/envs/isaaclab/bin/python
RUN="$OUT_ROOT/$TAG-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$RUN"; ln -sfn "$RUN" "$OUT_ROOT/$TAG-latest"
P_REP=$((5600 + OFFSET)); P_SONIC=$((5556 + OFFSET)); P_DBG=$((5557 + OFFSET))
log "run dir $RUN (house $HOUSE, sections $SECTIONS, netns $USE_NS, port offset $OFFSET)"
if [[ "$USE_NS" == 1 ]]; then
  if sudo -n ip netns list | grep -qw "$NS"; then die "netns $NS exists (another run?); refusing"; fi
  sudo -n ip netns add "$NS" || die "cannot create netns $NS"
  sudo -n ip -n "$NS" link set lo up
  ns() { sudo -n ip netns exec "$NS" sudo -n -u "$USER" env HOME="$HOME" PATH="$PATH" "$@"; }
else
  [[ -z "$(deploys_in_my_netns)" ]] || die "a deploy is already running in this network namespace (DDS domain 0 on lo)"
  for b in 5556 5557 5565 5600 5601 5602; do
    ss -ltn "sport = :$((b + OFFSET))" | grep -q LISTEN && die "port $((b + OFFSET)) busy: $(ss -ltnp "sport = :$((b + OFFSET))" | tail -1)"
  done
  ns() { "$@"; }
fi
P1PID=""; DEPPID=""
cleanup() {
  set +e
  log "cleanup"
  ns "$VENV_SIM/bin/python" -c "import zmq,json;s=zmq.Context().socket(zmq.REQ);s.setsockopt(zmq.RCVTIMEO,3000);s.setsockopt(zmq.LINGER,0);s.connect('tcp://127.0.0.1:$P_REP');s.send(b'{\"op\":\"shutdown\"}');s.recv()" >/dev/null 2>&1
  sleep 3
  if [[ "$USE_NS" == 1 ]]; then
    for p in $(sudo -n ip netns pids "$NS" 2>/dev/null); do sudo -n kill -INT "$p" 2>/dev/null; done
    sleep 3
    for p in $(sudo -n ip netns pids "$NS" 2>/dev/null); do sudo -n kill -KILL "$p" 2>/dev/null; done
    sudo -n ip netns del "$NS" 2>/dev/null
  else
    bash "$SONIC_DIR/run_deploy.sh" stop --session deploy-p1char-sonic >/dev/null 2>&1
    for p in $P1PID $DEPPID; do kill -INT "$p" 2>/dev/null; done
    sleep 3
    for p in $P1PID $DEPPID; do kill -KILL "$p" 2>/dev/null; done
    pkill -u "$USER" -x -f "sleep 99991" 2>/dev/null   # our stdin holder only (exact command line)
  fi
}
trap cleanup EXIT

# 1) P1 (contract ports + DDS domain 0 inside the namespace), same flags as scripts/m1_up.sh
ns bash -c "source /etc/profile.d/ludo.sh; cd $WL_ROOT; export PYTHONUNBUFFERED=1; exec $PY_ISAAC -m sim_isaac.app \
  --house $HOUSE --physics-hz 200 --dds-domain 0 --dds-iface lo --camera $CAMERA --camera-hz 30 --rt-pace \
  --physx-device cpu --port-offset $OFFSET --duration 3000 --stats-out $RUN/p1_stats.json --out-dir $RUN" > "$RUN/p1.log" 2>&1 &
P1PID=$!
timeout 900 bash -c "until grep -q -E 'WL_ISAAC_READY|Traceback' '$RUN/p1.log'; do sleep 2; done"
grep -q WL_ISAAC_READY "$RUN/p1.log" || { tail -30 "$RUN/p1.log"; die "P1 did not become ready"; }
log "P1 ready"

# 2) the unmodified deploy (zmq_manager, sim, lo); stdin is a pipe that never delivers 'o' (sleep 99991: a
#    distinctive holder so cleanup kills only ours)
ns bash -c "sleep 99991 | bash $SONIC_DIR/run_deploy.sh fg --session deploy-p1char-sonic --zmq-port $P_SONIC --zmq-out-port $P_DBG --log $RUN/deploy.log" > "$RUN/deploy_stdout.log" 2>&1 &
DEPPID=$!
timeout 600 bash -c "until grep -q 'Init Done' '$RUN/deploy.log' 2>/dev/null; do sleep 1; done" || { tail -30 "$RUN/deploy.log"; die "no Init Done"; }
log "deploy Init Done"

# 3) record a P1 trace (E4-style evidence) and run the characterisation driver
ns "$VENV_SIM/bin/python" -c "import zmq,json;s=zmq.Context().socket(zmq.REQ);s.setsockopt(zmq.RCVTIMEO,3000);s.connect('tcp://127.0.0.1:$P_REP');s.send(json.dumps({'op':'record','on':True,'path':'$RUN/trace.npz'}).encode());print(s.recv())" || true
( cd "$WBC_DIR" && ns "$VENV_SIM/bin/python" -u "$SONIC_DIR/mujoco_ref/char_ref.py" --p1 --ctl-port $P_REP \
    --zmq-port $P_SONIC --zmq-out-port $P_DBG --out "$RUN" --band-lower 0 --send-stop --sections "$SECTIONS" \
    "${EXTRA[@]}" 2>&1 | tee "$RUN/drive.log" )
RC=${PIPESTATUS[0]}
ns "$VENV_SIM/bin/python" -c "import zmq,json;s=zmq.Context().socket(zmq.REQ);s.setsockopt(zmq.RCVTIMEO,3000);s.connect('tcp://127.0.0.1:$P_REP');s.send(json.dumps({'op':'record','on':False}).encode());print(s.recv());s.send(b'{\"op\":\"get_stats\"}');open('$RUN/p1_stats_end.json','w').write(s.recv().decode())" || true
log "done rc=$RC: $RUN"
exit $RC
