#!/usr/bin/env bash
# Run on the LAPTOP. Brings recordings from the box to worldline-g1/outputs/recordings/ (rsync, incremental).
#
#   viz/pull_recordings.sh            # every run
#   viz/pull_recordings.sh latest     # only the newest run (prints its local path)
#   viz/pull_recordings.sh <run-name> # one run, e.g. 20260928-231500-ui
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WL="$(cd "$HERE/.." && pwd)"
INFRA="${INFRA:-$WL/../ludo_robotics_prep_g1/00_infra}"
REMOTE=/work/worldline-g1/outputs/recordings
LOCAL="$WL/outputs/recordings"
mkdir -p "$LOCAL"
what="${1:-all}"
if [[ "$what" == "all" ]]; then
  "$INFRA/sync_wl.sh" pull "$REMOTE/" "$LOCAL"
  ls -1t "$LOCAL" | head -5 | sed "s|^|$LOCAL/|"
  exit 0
fi
if [[ "$what" == "latest" ]]; then
  what="$("$INFRA/ssh.sh" "ls -1t $REMOTE 2>/dev/null | head -1" | tr -d '\r')"
  [[ -n "$what" ]] || { echo "no recordings on the box under $REMOTE" >&2; exit 1; }
fi
"$INFRA/sync_wl.sh" pull "$REMOTE/$what" "$LOCAL"
echo "$LOCAL/$what"
ls -1 "$LOCAL/$what"
