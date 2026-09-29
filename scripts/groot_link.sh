#!/usr/bin/env bash
# OD3 link: the MAIN box reaches the GR00T PolicyServer on the DEV box through an SSH tunnel
#   main 127.0.0.1:5550  ==ssh -L==>  dev 127.0.0.1:5550 (scripts/groot_server.sh)
# with a dedicated ed25519 key that the dev box authorizes for exactly that forward (restrict, port-forwarding,
# permitopen="127.0.0.1:5550", a forced command, optional from=). Kept up by autossh when installed, else by a
# reconnect loop, in tmux 'groot-link'. Wave-2 steps: docs/groot_serving.md §5.
#
# MAIN box:
#   bash scripts/groot_link.sh keygen                        # ~/.ssh/groot_link_ed25519 (once); prints the public key
#   bash scripts/groot_link.sh pin-host --host DEV_IP 'ssh-ed25519 AAAA…'   # the dev box's host key (from the laptop's
#                                                            # 00_infra/.state-ludo-g1-arena/known_hosts): no TOFU
#   bash scripts/groot_link.sh up --host DEV_IP [--port 5550] [--local-port 5550] [--user ubuntu]
#                                                            # a running link to another host is replaced (new IPs
#                                                            # after a Brev restart, groot_serving.md §5.2)
#   bash scripts/groot_link.sh down | status | check [--local-port 5550]
#   bash scripts/groot_link.sh ensure [--local-port 5550]     # check; if the link is down, bring it up again with the
#                                                            # host of the last `up` (m2_up.sh --profile full runs this)
# Env: GROOT_LINK_TASKSET  CPUs of the tunnel's ssh client (default 4-15 on the main box: never the SONIC deploy's
#                          0-3, m1_up.sh; "" = no pinning); GROOT_LINK_NICE (default 5)
# DEV box:
#   bash scripts/groot_link.sh install-key 'ssh-ed25519 AAAA… groot-link@main' [--from MAIN_IP] [--port 5550]
#                                                            (or --pub-file FILE)
#   bash scripts/groot_link.sh remove-key [--tag groot-link]
#   bash scripts/groot_link.sh selftest [--port 5550]        # wave-1 check of the restrictions, on the dev box alone
#
# The dev box keeps these keys in ~/.ssh/authorized_keys2 (sshd's default AuthorizedKeysFile lists it; checked with
# `sshd -T`), so the operator keys in ~/.ssh/authorized_keys are never edited.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${GROOT_LINK_KEY:-$HOME/.ssh/groot_link_ed25519}"
KNOWN="$HOME/.ssh/groot_link_known_hosts"
AUTH2="$HOME/.ssh/authorized_keys2"
SESSION=groot-link
LOG_DIR="${GROOT_LINK_LOG_DIR:-/work/logs/groot}"
FORCED_CMD='echo groot-link: port-forward only; exit 1'

log() { printf '[groot_link %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

cmd="${1:-}"; shift || true
HOST=""; USER_="ubuntu"; PORT=5550; LPORT=""; FROM=""; TAG="groot-link"; PUB=""; HOSTKEY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --user) USER_="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --local-port) LPORT="$2"; shift 2;;
    --from) FROM="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --pub-file) PUB="$(cat "$2")"; shift 2;;
    ssh-ed25519\ *) PUB="$1"; HOSTKEY="$1"; shift;;
    *) die "unknown arg $1 (quote the public key as ONE argument)";;
  esac
done
LPORT="${LPORT:-$PORT}"
[[ "$PORT" =~ ^[0-9]+$ && "$LPORT" =~ ^[0-9]+$ ]] || die "bad port"
LINK_TASKSET="${GROOT_LINK_TASKSET-$([[ "$(uname -s)" == Linux ]] && echo 4-15)}"
LINK_NICE="${GROOT_LINK_NICE:-5}"

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
ping_port() {  # ping the PolicyServer through local port $1
  local py; py="$(client_py)" || die "no python with numpy+pyzmq+msgpack (set GROOT_CLIENT_PY)"
  (cd "$REPO" && "$py" -m groot.policy_client ping --endpoint "tcp://127.0.0.1:$1" --timeout "${2:-3}")
}

# authorized_keys2 line: everything off (restrict = no pty, no agent/X11 forwarding, no user rc, no port
# forwarding), then local forwarding back on, only to 127.0.0.1:PORT, and any exec/shell request runs FORCED_CMD.
auth_line() {  # $1 = public key "ssh-ed25519 AAAA… comment", $2 = from (optional), $3 = port
  local opts="restrict,port-forwarding,permitopen=\"127.0.0.1:$3\",command=\"$FORCED_CMD\""
  [[ -n "$2" ]] && opts="from=\"$2\",$opts"
  local k; k="$(awk '{print $1" "$2}' <<<"$1")"
  echo "$opts $k $TAG"
}

tunnel_cmd() {  # the ssh client command of the link (also used by selftest against 127.0.0.1)
  echo "ssh -N -i '$KEY' -o IdentitiesOnly=yes -o BatchMode=yes -o ExitOnForwardFailure=yes \
-o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
-o UserKnownHostsFile='$KNOWN' -L 127.0.0.1:$LPORT:127.0.0.1:$PORT $USER_@$HOST"
}

keygen() {
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  if [[ ! -f "$KEY" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C "groot-link@$(hostname -s)" -f "$KEY"
    log "created $KEY"
  else
    log "$KEY exists"
  fi
  cat "$KEY.pub"
}

STATE_ENV() { echo "$LOG_DIR/link-$LPORT.env"; }   # the last `up`'s host/user/port, for `ensure`

up() {
  local want="$HOST"
  if [[ -f "$(STATE_ENV)" ]]; then
    source "$(STATE_ENV)"
    [[ -z "$HOST" ]] && { HOST="$LINK_HOST"; USER_="$LINK_USER"; PORT="$LINK_PORT"; }
  fi
  [[ -n "$HOST" ]] || die "--host DEV_IP required (no earlier up recorded in $(STATE_ENV))"
  [[ -f "$KEY" ]] || die "no $KEY (run: groot_link.sh keygen, then install the key on the dev box)"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ -n "$want" && "$want" != "${LINK_HOST:-}" ]]; then   # the dev box's IP changed (a Brev restart)
      log "tmux $SESSION goes to ${LINK_HOST:-?}: replacing it with $want"; down
    else
      log "tmux $SESSION already up"; status; return 0
    fi
  fi
  mkdir -p "$LOG_DIR"
  printf 'LINK_HOST=%q\nLINK_USER=%q\nLINK_PORT=%q\n' "$HOST" "$USER_" "$PORT" > "$(STATE_ENV)"
  local ssh_cmd; ssh_cmd="$(tunnel_cmd)"
  # CPU placement: the tunnel's ssh encrypts ~2.3 MB/s while GR00T runs (0.92 MB requests at 2.5 Hz); keep it off the
  # SONIC deploy's cores (0-3) and below the stack's priority
  local pre=""
  [[ -n "$LINK_TASKSET" ]] && command -v taskset >/dev/null 2>&1 && pre="taskset -c $LINK_TASKSET "
  [[ "$LINK_NICE" != 0 ]] && pre="${pre}nice -n $LINK_NICE "
  local runner
  if command -v autossh >/dev/null 2>&1; then
    runner="AUTOSSH_GATETIME=0 AUTOSSH_POLL=30 exec ${pre}autossh -M 0 ${ssh_cmd#ssh }"
  else  # reconnect loop: ssh exits when the link dies (ServerAlive 3 x 5 s) or the forward cannot bind
    runner="while true; do ${pre}$ssh_cmd; echo \"\$(date -u +%FT%TZ) ssh exited rc=\$?; reconnect in 2 s\"; sleep 2; done"
  fi
  tmux new-session -d -s "$SESSION" "exec >>'$LOG_DIR/link-$LPORT.log' 2>&1; echo \"\$(date -u +%FT%TZ) up $HOST (taskset ${LINK_TASKSET:-none}, nice $LINK_NICE)\"; $runner"
  log "tmux $SESSION: 127.0.0.1:$LPORT -> $HOST 127.0.0.1:$PORT ($(command -v autossh >/dev/null && echo autossh || echo 'reconnect loop'), taskset ${LINK_TASKSET:-none}, nice $LINK_NICE)"
  local i; for i in $(seq 1 15); do ping_port "$LPORT" 2 >/dev/null 2>&1 && { log "PolicyServer answers through the link"; return 0; }; sleep 1; done
  log "no ping through the link yet (server down? see $LOG_DIR/link-$LPORT.log); the link keeps retrying"
}

pin_host() {  # the dev box's host key, so the first connect is not trust-on-first-use; replaces older pins
  [[ -n "$HOST" && "$HOSTKEY" == ssh-ed25519\ * ]] || die "pin-host --host DEV_IP 'ssh-ed25519 AAAA…'"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  echo "$HOST $(awk '{print $1" "$2}' <<<"$HOSTKEY")" > "$KNOWN" && chmod 600 "$KNOWN"
  log "pinned $HOST in $KNOWN"
}

link_ssh_pids() {  # the ssh/autossh clients of this link only (never a shell whose command line mentions it)
  local p
  for p in $(pgrep -x ssh; pgrep -x autossh); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qF -- "-L 127.0.0.1:$LPORT:127.0.0.1:$PORT " && echo "$p"
  done
  return 0
}

down() {
  tmux kill-session -t "$SESSION" 2>/dev/null && log "tmux $SESSION killed" || log "no tmux $SESSION"
  local p; for p in $(link_ssh_pids); do kill "$p" 2>/dev/null && log "killed ssh $p"; done
  return 0
}

status() {
  echo "session: $(tmux has-session -t "$SESSION" 2>/dev/null && echo up || echo down)"
  echo "listen:  $(ss -ltn "sport = :$LPORT" 2>/dev/null | awk 'NR>1{print $4}' | paste -sd, - || true)"
  echo "ping:    $(ping_port "$LPORT" 2 2>/dev/null || true)"
  local p; for p in $(link_ssh_pids); do echo "ssh:     pid $p, $(taskset -cp "$p" 2>/dev/null | sed 's/.*: /cpus /'), nice $(ps -o ni= -p "$p" | tr -d ' ')"; done
  [[ -f "$(STATE_ENV)" ]] && echo "host:    $(sed -n 's/^LINK_HOST=//p' "$(STATE_ENV)")"
  [[ -f "$LOG_DIR/link-$LPORT.log" ]] && { echo "log tail:"; tail -3 "$LOG_DIR/link-$LPORT.log"; }
  return 0
}

ensure() {  # the link answers, or is brought up again (the tmux loop itself can be killed by hand or a reboot)
  if ping_port "$LPORT" 3 2>/dev/null; then return 0; fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux $SESSION is up but the PolicyServer does not answer (dev-box server down? see $LOG_DIR/link-$LPORT.log)"
    return 1
  fi
  [[ -f "$(STATE_ENV)" ]] || { log "link down and no earlier up recorded ($(STATE_ENV)): run up --host DEV_IP"; return 1; }
  log "link down: bringing it up again with the last host"
  up
  ping_port "$LPORT" 3
}

install_key() {
  [[ "$PUB" == ssh-ed25519\ * ]] || die "pass the public key line: install-key 'ssh-ed25519 AAAA… comment'"
  local line; line="$(auth_line "$PUB" "$FROM" "$PORT")"
  local b64; b64="$(awk '{print $2}' <<<"$PUB")"
  touch "$AUTH2" && chmod 600 "$AUTH2"
  if grep -qF "$b64" "$AUTH2"; then
    log "key already in $AUTH2; replacing its line"
    grep -vF "$b64" "$AUTH2" > "$AUTH2.tmp" || true
    mv "$AUTH2.tmp" "$AUTH2" && chmod 600 "$AUTH2"
  fi
  echo "$line" >> "$AUTH2"
  log "installed in $AUTH2: $(sed -E 's/(ssh-ed25519 [A-Za-z0-9+\/]{12})[A-Za-z0-9+\/=]+/\1…/' <<<"$line")"
}

remove_key() {
  [[ -f "$AUTH2" ]] || { log "no $AUTH2"; return 0; }
  local n; n="$(grep -c " $TAG\$" "$AUTH2" || true)"
  grep -v " $TAG\$" "$AUTH2" > "$AUTH2.tmp" || true
  mv "$AUTH2.tmp" "$AUTH2" && chmod 600 "$AUTH2"
  log "removed $n line(s) tagged '$TAG' from $AUTH2"
}

selftest() {
  # On the DEV box only: a throwaway key, authorized with the production restrictions plus from="127.0.0.1",
  # a loopback tunnel to this box's own sshd, then the checks. Everything is removed at the end.
  local tmp; tmp="$(mktemp -d)"
  local tport=$((LPORT + 10000)) nport=$((LPORT + 10001))
  KEY="$tmp/key"; KNOWN="$tmp/known_hosts"; TAG="groot-link-selftest"; HOST=127.0.0.1
  local pids=() fail=0 out
  cleanup() { local p; for p in "${pids[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done
              remove_key >/dev/null 2>&1 || true; rm -rf "$tmp"; }
  trap cleanup EXIT
  ssh-keygen -q -t ed25519 -N '' -C "groot-link-selftest" -f "$KEY"
  PUB="$(cat "$KEY.pub")"; FROM=127.0.0.1
  install_key
  check() { if [[ "$1" == ok ]]; then echo "PASS  $2"; else echo "FAIL  $2"; fail=1; fi; }
  # 1. the allowed forward works: PolicyServer ping through tport
  LPORT=$tport; eval "$(tunnel_cmd) 2>'$tmp/pos.err' &"; pids+=($!); sleep 3
  if ping_port "$tport" 3 >/dev/null 2>&1; then check ok "forward 127.0.0.1:$tport -> 127.0.0.1:$PORT answers ping"
  else check no "forward 127.0.0.1:$tport -> 127.0.0.1:$PORT answers ping (is groot_server.sh up?)"; fi
  # 2. a forward to anything else is refused by permitopen (sshd port 22 would answer with a banner)
  ssh -N -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o UserKnownHostsFile="$KNOWN" -L "127.0.0.1:$nport:127.0.0.1:22" "$USER_@127.0.0.1" 2>"$tmp/neg.err" &
  pids+=($!); sleep 3
  out="$(timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$nport && head -c 20 <&3" 2>/dev/null || true)"
  [[ "$out" != SSH-* ]] && grep -q "prohibited\|open failed" "$tmp/neg.err" \
    && check ok "forward to 127.0.0.1:22 refused (sshd: $(grep -o 'administratively prohibited[^"]*' "$tmp/neg.err" | head -1))" \
    || check no "forward to 127.0.0.1:22 refused (got '${out}')"
  # 3. a command runs the forced command, never the requested one
  out="$(ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
         -o UserKnownHostsFile="$KNOWN" "$USER_@127.0.0.1" 'id -un' 2>/dev/null || true)"
  [[ "$out" == "groot-link: port-forward only" ]] && check ok "exec request gets the forced command ('$out')" \
    || check no "exec request gets the forced command (got '$out')"
  # 4. no pty
  out="$(ssh -tt -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
         -o UserKnownHostsFile="$KNOWN" "$USER_@127.0.0.1" 2>&1 </dev/null || true)"
  grep -q "PTY allocation request failed" <<<"$out" && check ok "pty refused" || check no "pty refused (got: ${out:0:80})"
  # 5. the operator key file was not touched and the test line goes away
  cleanup; trap - EXIT
  grep -q "groot-link-selftest" "$AUTH2" 2>/dev/null && check no "selftest key removed" || check ok "selftest key removed"
  return $fail
}

case "$cmd" in
  keygen) keygen;;
  up) up;;
  down) down;;
  status) status;;
  pin-host) pin_host;;
  check) ping_port "$LPORT" 3;;
  ensure) ensure;;
  install-key) install_key;;
  remove-key) remove_key;;
  auth-line) [[ -n "$PUB" ]] || die "auth-line 'ssh-ed25519 AAAA…' [--from IP]"; auth_line "$PUB" "$FROM" "$PORT";;
  selftest) selftest;;
  *) sed -n '2,28p' "$0"; exit 2;;
esac
