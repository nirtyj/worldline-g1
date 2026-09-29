#!/usr/bin/env bash
# Build or refresh P5's runtime venv on the box (docs/M2.md §7.4 step 1): $WL/.venv-rt, Python 3.11 via uv, with
# pyproject.toml's dependencies and its `test` extra (pytest), then an import check of every module P5 loads.
#
#   bash scripts/m2_venv.sh [--recreate] [--check]      # --check: only the import check, no install
#
# Idempotent: an existing venv is reused and brought up to date with `uv pip install -r pyproject.toml`. A venv whose
# interpreter does not run here (e.g. a macOS copy from an old sync) is rebuilt. 00_infra/sync_wl.sh push excludes
# '.venv*', so a push never touches this venv. The body (.venv), viz (viz/.venv) and Isaac (/work/envs/isaaclab)
# environments are separate and are not changed.
set -euo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true
WL=${WL:-/work/worldline-g1}
VENV=${VENV:-$WL/.venv-rt}
PY=$VENV/bin/python
RECREATE=0; CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1; shift;;
    --check) CHECK=1; shift;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
say() { echo "[m2_venv $(date +%H:%M:%S)] $*"; }

if [[ "$CHECK" == 0 ]]; then
  command -v uv >/dev/null || { echo "uv not on PATH (source /etc/profile.d/ludo.sh)" >&2; exit 1; }
  [[ "$RECREATE" == 1 ]] && rm -rf "$VENV"
  if [[ -e "$VENV" ]] && ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
    say "$VENV does not run here or is not Python 3.11: rebuilding it"
    rm -rf "$VENV"
  fi
  [[ -x "$PY" ]] || uv venv --python 3.11 "$VENV"
  t0=$SECONDS
  uv pip install --python "$PY" -r "$WL/pyproject.toml" --extra test
  say "dependencies installed in $((SECONDS - t0)) s"
fi

(cd "$WL" && "$PY" - <<'EOF'
import importlib, sys
from importlib import metadata
mods = ["numpy", "scipy", "PIL", "zmq", "msgpack", "websockets", "yaml", "google.genai", "typesafe_sdk", "pytest",
        # what P5 itself imports on the Isaac profiles (the robot factory, the GT world model, the frame tap)
        "ui.server", "robot.factory", "world.isaac_client", "world.frames", "viz.tap", "brains.system1_jev",
        "brains.scripted", "agent.model"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{m}: {type(e).__name__}: {e}")
pins = {d: metadata.version(d) for d in ("google-genai", "typesafe_sdk", "websockets", "numpy", "pyzmq")}
print(f"python {sys.version.split()[0]} at {sys.executable}; " + ", ".join(f"{k} {v}" for k, v in pins.items()))
if bad:
    print("IMPORT FAILURES:\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"runtime venv ok: {len(mods)} modules import")
EOF
)
