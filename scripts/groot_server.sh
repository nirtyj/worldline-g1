#!/usr/bin/env bash
# GR00T N1.7 PolicyServer (P4) for the G1 arms: nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace @ 7f78beb, label
# "experimental". Runs on the DEV box (OD3) in tmux 'groot-server', bound to 127.0.0.1 only; the main box reaches it
# through scripts/groot_link.sh. Contract and numbers: docs/groot_serving.md.
#
#   bash scripts/groot_server.sh start  [--port 5550] [--ckpt DIR] [--gpu 0] [--wait 600] [--warm]
#   bash scripts/groot_server.sh stop   [--port 5550]
#   bash scripts/groot_server.sh status [--port 5550]     # tmux, pid, port, VRAM, ping, state file
#   bash scripts/groot_server.sh ping   [--port 5550]
#
# --warm runs one get_action after start (the first call pays CUDA warm-up, ~seconds) and records its time.
# Env: GROOT_DIR   Isaac-GR00T checkout with its .venv (default /work/arena/gr00t_n17 = 4b1dca9d, the pin of Arena's
#                  static_apple workflow; the Arena spike served this checkpoint from it)
#      GROOT_CLIENT_PY  python with numpy + pyzmq + msgpack for ping/warm (default: /work/groot/venv, else the repo .venv)
#      HF_HUB_OFFLINE   default 1 (the checkpoint dir and nvidia/Cosmos-Reason2-2B are cached in $HF_HOME)
# Logs: /work/logs/groot/server-<port>.log; state: /work/logs/groot/server-<port>.json.
# Never run this under `bash -x`: /etc/profile.d/ludo.sh sources the API keys (docs/devbox.md §6.1).
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
if [[ -f /etc/profile.d/ludo.sh ]]; then source /etc/profile.d/ludo.sh >/dev/null 2>&1 || true; fi
set -u
export HF_HOME="${HF_HOME:-/work/hf-cache}"

GROOT_DIR="${GROOT_DIR:-/work/arena/gr00t_n17}"
GROOT_SHA_EXPECTED=4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
CKPT_DEFAULT=/work/arena/models/isaaclab_arena/static_apple_tutorial/gn1x_tuned_static_apple
LOG_DIR=/work/logs/groot

cmd="${1:-}"; shift || true
PORT=5550; CKPT="$CKPT_DEFAULT"; GPU=0; WAIT=600; WARM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --wait) WAIT="$2"; shift 2;;
    --warm) WARM=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "bad --port $PORT" >&2; exit 2; }
SESSION=groot-server; [[ "$PORT" == 5550 ]] || SESSION="groot-server-$PORT"
LOG="$LOG_DIR/server-$PORT.log"
STATE="$LOG_DIR/server-$PORT.json"
ENDPOINT="tcp://127.0.0.1:$PORT"

log() { printf '[groot_server %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

client_py() {
  local c
  for c in "${GROOT_CLIENT_PY:-}" /work/groot/venv/bin/python "$REPO/.venv/bin/python" python3; do
    [[ -n "$c" ]] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import numpy, zmq, msgpack' >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

server_pid() {  # the python process itself (not the tmux/bash wrapper whose command line also names the script)
  local p
  for p in $(pgrep -f "run_gr00t_server.py" || true); do
    [[ "$(cat "/proc/$p/comm" 2>/dev/null)" == python* ]] || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "--port $PORT( |$)" && { echo "$p"; return 0; }
  done
  return 0
}
listening() { ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; }
vram_mib() {  # used GPU memory of one pid (MiB), empty if none
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v p="$1" '$1==p {print $2}' | head -1
}
do_ping() {
  local py; py="$(client_py)" || { log "no python with numpy+pyzmq+msgpack (set GROOT_CLIENT_PY)"; return 2; }
  (cd "$REPO" && "$py" -m groot.policy_client ping --endpoint "$ENDPOINT" --timeout "${1:-2}")
}

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux $SESSION already exists"; status; return 0
  fi
  listening && die "port $PORT is already in use (ss -ltnp 'sport = :$PORT')"
  [[ -x "$GROOT_DIR/.venv/bin/python" ]] || die "$GROOT_DIR/.venv missing (arena_spike/setup_gr00t_envs.sh n17)"
  local sha; sha="$(git -C "$GROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  [[ "$sha" == "$GROOT_SHA_EXPECTED" ]] || log "WARNING: $GROOT_DIR is at $sha, expected $GROOT_SHA_EXPECTED"
  local f; for f in config.json processor_config.json statistics.json model.safetensors.index.json; do
    [[ -f "$CKPT/$f" ]] || die "checkpoint $CKPT has no $f"
  done
  mkdir -p "$LOG_DIR"
  {
    echo "===== $(date -u +%FT%TZ) start port=$PORT gpu=$GPU ckpt=$CKPT groot=$sha offline=${HF_HUB_OFFLINE:-1}"
  } >> "$LOG"
  local t0; t0=$(date +%s.%N)
  tmux new-session -d -s "$SESSION" "cd '$GROOT_DIR' && export CUDA_VISIBLE_DEVICES='$GPU' HF_HOME='$HF_HOME' \
HF_HUB_OFFLINE='${HF_HUB_OFFLINE:-1}' NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1 && \
.venv/bin/python gr00t/eval/run_gr00t_server.py --model-path '$CKPT' --embodiment-tag NEW_EMBODIMENT \
--device cuda --host 127.0.0.1 --port $PORT 2>&1 | tee -a '$LOG'"
  log "tmux $SESSION started; waiting up to ${WAIT}s for ping on $ENDPOINT (log $LOG)"
  local deadline=$(( $(date +%s) + WAIT ))
  until do_ping 2 >/dev/null 2>&1; do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
      tail -30 "$LOG" >&2; die "server exited during start"
    fi
    (( $(date +%s) < deadline )) || { tail -30 "$LOG" >&2; die "no ping within ${WAIT}s (tmux $SESSION left running)"; }
    sleep 2
  done
  local ready; ready=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
  local pid; pid="$(server_pid)"
  local bind; bind="$(ss -ltn "sport = :$PORT" | awk 'NR>1{print $4}' | paste -sd, -)"
  local vram; vram="$(vram_mib "$pid")"
  log "ready in ${ready}s: pid $pid, listening $bind, VRAM ${vram:-?} MiB"
  local warm_json="null"
  if [[ "$WARM" == 1 ]]; then
    local py; py="$(client_py)"
    warm_json="$(cd "$REPO" && "$py" -m groot.bench --endpoint "$ENDPOINT" --first-call --n 0 --pings 0 --warmup 0 \
                 --timeout 5 2>/dev/null | "$py" -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("first_call_ms")))')"
    log "first get_action (warm-up): ${warm_json} ms; VRAM now $(vram_mib "$pid") MiB"
  fi
  local vram_warm=""
  [[ "$WARM" == 1 && -n "$pid" ]] && vram_warm="$(vram_mib "$pid")"
  cat > "$STATE" <<JSON
{"port": $PORT, "pid": ${pid:-null}, "session": "$SESSION", "bind": "$bind", "ckpt": "$CKPT",
 "groot_dir": "$GROOT_DIR", "groot_sha": "$sha", "gpu": "$GPU", "started_utc": "$(date -u +%FT%TZ)",
 "ready_s": $ready, "vram_mib_after_load": ${vram:-null}, "first_call_ms": ${warm_json:-null},
 "vram_mib_after_warm": ${vram_warm:-null}}
JSON
  [[ "$bind" == "127.0.0.1:$PORT" ]] || die "server bound to '$bind', expected 127.0.0.1:$PORT only"
}

stop() {
  local pid; pid="$(server_pid)"
  tmux kill-session -t "$SESSION" 2>/dev/null && log "tmux $SESSION killed" || log "no tmux $SESSION"
  if [[ -n "$pid" ]]; then
    local i; for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then log "pid $pid still alive: SIGTERM"; kill "$pid" 2>/dev/null || true; sleep 3; fi
    if kill -0 "$pid" 2>/dev/null; then log "pid $pid still alive: SIGKILL"; kill -9 "$pid" 2>/dev/null || true; fi
  fi
  listening && die "port $PORT still listening" || log "stopped (port $PORT free)"
  echo "===== $(date -u +%FT%TZ) stop" >> "$LOG" 2>/dev/null || true
}

status() {
  local pid; pid="$(server_pid)"
  echo "session:   $(tmux has-session -t "$SESSION" 2>/dev/null && echo "$SESSION up" || echo "$SESSION down")"
  echo "pid:       ${pid:-none}"
  echo "listen:    $(ss -ltn "sport = :$PORT" 2>/dev/null | awk 'NR>1{print $4}' | paste -sd, - || true)"
  [[ -n "$pid" ]] && echo "vram_mib:  $(vram_mib "$pid")"
  echo "ping:      $(do_ping 2 2>/dev/null || true)"
  [[ -f "$STATE" ]] && echo "state:     $(tr -d '\n' < "$STATE")"
  return 0
}

case "$cmd" in
  start) start;;
  stop) stop;;
  status) status;;
  ping) do_ping 3;;
  *) sed -n '2,20p' "$0"; exit 2;;
esac
