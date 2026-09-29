#!/usr/bin/env bash
# Stall-tolerance matrix: how does the unmodified SONIC deploy (wall clock) react when the simulator's physics loop
# blocks for X ms (an Isaac render hitch, a descheduled sim process) and then either bursts to catch up or drops the
# lag (time slip)? Runs run_ref_loop.sh --driver char_ref.py (walk start/stop cycles + curved walks) once per config.
#
#   bash sonic/mujoco_ref/stall_matrix.sh [CONFIGS...]     default: base burst60 drop60 drop120
#   config names: base | burstN | dropN   (N = stall ms, injected every 3 sim-s after the band release)
# Output: one run dir per config under /work/worldline-g1/outputs/m1/deploy/stall-<config>-<ts>/ + a summary line
# per config in /work/worldline-g1/outputs/m1/deploy/stall_matrix-<ts>.txt
set -uo pipefail
source "$(dirname "$0")/../sonic_env.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=("$@"); ((${#CONFIGS[@]})) || CONFIGS=(base burst60 drop60 drop120)
SUM="$OUT_ROOT/stall_matrix-$(date +%Y%m%d-%H%M%S).txt"
for c in "${CONFIGS[@]}"; do
  case "$c" in
    base) SIM="";;
    burst*) SIM="--inject-stall-ms ${c#burst} --inject-every-s 3";;
    drop*) SIM="--inject-stall-ms ${c#drop} --inject-every-s 3 --inject-drop";;
    *) die "unknown config $c";;
  esac
  log "=== $c: sim args '$SIM'"
  bash "$HERE/run_ref_loop.sh" --driver char_ref.py --tag "stall-$c" --no-render ${ALLOW_BUSY:+--allow-busy} --sim-args "$SIM" -- --sections cycles,curves \
    > "$LOG_ROOT/deploy-stall-$c.log" 2>&1
  RUN=$(readlink -f "$OUT_ROOT/stall-$c-latest")
  "$VENV_SIM/bin/python" - "$RUN" "$c" <<'EOF' | tee -a "$SUM"
import json, sys
run, c = sys.argv[1], sys.argv[2]
r = json.load(open(f"{run}/drive_result.json"))
st = json.load(open(f"{run}/sim_stats.json"))
ch = json.load(open(f"{run}/char.json")) if __import__("os").path.exists(f"{run}/char.json") else {}
wc, cv = ch.get("walk_cycles", []), ch.get("curve", [])
tests = {t["test"]: t for t in r["tests"]}
print(json.dumps({"config": c, "run": run, "pass": r["pass"],
                  "falls_since_release": tests.get("no_falls_since_release", {}).get("falls"),
                  "cycles_with_fall": sum(1 for x in wc if x["falls"]), "n_cycles": len(wc),
                  "curves_with_fall": sum(1 for x in cv if x["falls"]), "n_curves": len(cv),
                  "injected_stalls": st.get("injected_stalls"), "natural_stalls_gt15ms": st.get("stalls_gt15ms"),
                  "stall_max_ms": st.get("stall_max_ms"), "rtf": round(st.get("rtf", 0), 3)}))
EOF
done
log "summary: $SUM"
