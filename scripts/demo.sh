#!/usr/bin/env bash
# The Worldline-on-G1 demo: the most important behaviours in one run, said to the live Worldline UI in order
# (tools/say.py), then a PASS/FAIL table judged from the page trace. Runs ON the main box. docs/demo.md.
#
#   scripts/demo.sh [--fresh] [--record] [--groot off|on] [--only N[,N..]] [--list] [--keep-memory] [--label L] [--no-lock]
#
#   --fresh        scripts/m2_down.sh, then scripts/m2_up.sh --profile full --scene procthor-train-40 --viz low
#                  --p5-port 8766 (live planner + Jev + Gemini Live), the GR00T link and the Sim Viewer (viz-server,
#                  8765) ensured; waits for the page and System 1. The robot starts standing, hands empty.
#                  Also starts the house's spatial memory empty (runs/memory/<scene>.json is moved to
#                  runs/memory_backup/), so what the robot knows in the demo it learned in the demo.
#   --keep-memory  with --fresh: keep the spatial memory and notes of earlier sessions
#   --groot off    (default) the OD3 GR00T link is taken down for the demo (m2_up.sh --groot off), so `full` rejects
#                  the GR00T attempt as policy_unavailable in ~3 s and picks with the labelled SONIC arm-script
#                  fallback. The link is brought back (groot_link.sh ensure) at exit.
#   --groot on     keep GR00T: the pick first runs the experimental 26 s groot_arms attempt (zero-shot, times out),
#                  then the script. 2 of 3 live picks passed this way: the attempt can move the base or the bottle,
#                  the script then fails ik_unreachable and reach_stance loops (docs/demo.md, known limits)
#   --record       Sim Viewer recording around the steps (POST :8765/api/record); prints the run dir
#                  (composite.mp4, contact_sheet.png, head/chase/top.mp4, summary.json)
#   --only N,M     only these steps (numbers from --list); step 8 needs the robot's hands empty
#   --list         print the steps and exit
#   --no-lock      do not check or take /work/locks/stack.d (default: take it as `demo`, released at exit; a lock
#                  held by someone else stops the demo unless DEMO_LOCK_OWNER names that owner)
#
# Laptop one-liner (the stack's output streams back; about 15 min with --fresh):
#   BREV_NAME=ludo-g1-brev2 ../ludo_robotics_prep_g1/00_infra/ssh.sh 'bash /work/worldline-g1/scripts/demo.sh --fresh --record'
# Watch it: ../ludo_robotics_prep_g1/00_infra/tunnel.sh 8766 8765 -> http://localhost:8766 (Worldline UI),
#           http://localhost:8765 (Sim Viewer). Outputs: /work/worldline-g1/outputs/demo/run-<ts>/ (stepN.log,
#           stepN.json, results.md). Exit 0 when every step run passed, 1 otherwise.
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh; true

WL=${WL:-/work/worldline-g1}
PY_RT=${PY_RT:-$WL/.venv-rt/bin/python}
SCENE=procthor-train-40; SESSION=wl-m2; P5_PORT=8766; VIZ_PORT=8765
FRESH=0; RECORD=0; ONLY=""; LIST=0; KEEP_MEM=0; LABEL=demo; USE_LOCK=1; GROOT=off
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift;;
    --record) RECORD=1; shift;;
    --only) ONLY="$2"; shift 2;;
    --list) LIST=1; shift;;
    --keep-memory) KEEP_MEM=1; shift;;
    --label) LABEL="$2"; shift 2;;
    --groot) GROOT="$2"; shift 2;;
    --no-lock) USE_LOCK=0; shift;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) echo "unknown option $1 (--help)" >&2; exit 2;;
  esac
done
case "$GROOT" in on|off) ;; *) echo "--groot must be on or off" >&2; exit 2;; esac

# ---------------------------------------------------------------- the steps
# step_def N sets NAME, MAXS (s to wait on one line), LINES (what to say; "@N text" = N s after the previous line)
# and EXTRA (more tools.say options).
NSTEPS=9
step_def() {
  EXTRA=()
  case "$1" in
    1) NAME="memory note";                 MAXS=60;  LINES=("my keys are usually on the kitchen counter");;
    2) NAME="navigate + question mid-walk"; MAXS=150; LINES=("go to the bedroom dresser" "@6 where are you going?");;
    3) NAME="stop / resume while walking";  MAXS=180; LINES=("go to the dining table" "@6 stop" "@5 okay, carry on");;
    4) NAME="recall a note";               MAXS=60;  LINES=("where are my keys?");;
    5) NAME="spatial memory";              MAXS=60;  LINES=("where is the vase?");;
    6) NAME="correction mid-walk";         MAXS=150; LINES=("go to the living room" "@5 no, go to the kitchen counter instead");;
    # "can you reach the bowl?" was answered from belief with no check ("cannot reach it from here"); a request
    # makes the planner walk there and check_reachability, which answers beyond_reach with the distances
    7) NAME="honest refusal (reach)";      MAXS=150; LINES=("pick up the bowl");;
    8) NAME="pick (reach stance + arm)";   MAXS=330; LINES=("pick up the white bottle")
       EXTRA=(--if-asked "which (one|bottle|of)" "the white one");;
    9) NAME="question + chitchat";         MAXS=60;  LINES=("what are you holding?" "thanks");;
    *) return 1;;
  esac
}

if [[ "$LIST" == 1 ]]; then
  for n in $(seq 1 $NSTEPS); do
    step_def "$n"; printf '%d  %-30s' "$n" "$NAME"; printf ' "%s"' "${LINES[@]}"; echo
  done
  exit 0
fi
STEPS=()
if [[ -n "$ONLY" ]]; then
  IFS=, read -ra STEPS <<< "$ONLY"
  for n in "${STEPS[@]}"; do step_def "$n" >/dev/null || { echo "no step $n (--list)" >&2; exit 2; }; done
else
  STEPS=($(seq 1 $NSTEPS))
fi

T0=$(date +%s)
TS=$(date +%Y%m%d-%H%M%S)
RUN=$WL/outputs/demo/run-$TS
mkdir -p "$RUN"
say() { echo "[demo $(date +%H:%M:%S)] $*" | tee -a "$RUN/demo.log"; }
banner() { printf '\n==================== %s ====================\n' "$*" | tee -a "$RUN/demo.log"; }

# ---------------------------------------------------------------- the stack lock (docs/bringup.md)
LOCK=/work/locks/stack.d; OWN_LOCK=0; LINK_DOWN=0
cleanup() {
  if [[ "$LINK_DOWN" == 1 ]]; then        # the GR00T link as we found it: up (the stack's GR00T calls work again)
    bash "$WL/scripts/groot_link.sh" ensure >/dev/null 2>&1 && echo "[demo] GR00T link restored" || echo "[demo] WARNING: GR00T link not restored: bash scripts/groot_link.sh ensure"
  fi
  [[ "$OWN_LOCK" == 1 ]] && rm -rf "$LOCK"
  true
}
trap cleanup EXIT
if [[ "$USE_LOCK" == 1 ]]; then
  mkdir -p /work/locks
  if mkdir "$LOCK" 2>/dev/null; then
    echo "demo $(date +%s)" > "$LOCK/owner"; OWN_LOCK=1
    say "stack lock taken as demo (released at exit)"
  else
    holder=$(awk '{print $1}' "$LOCK/owner" 2>/dev/null)
    if [[ -n "${DEMO_LOCK_OWNER:-}" && "$holder" == "$DEMO_LOCK_OWNER" ]]; then
      say "stack lock held by $holder (DEMO_LOCK_OWNER): going on"
    else
      echo "[demo] the stack lock is held by '$(cat "$LOCK/owner" 2>/dev/null)': someone is using the stack." >&2
      echo "[demo] wait, or if it is stale: rm -rf $LOCK (or DEMO_LOCK_OWNER=$holder, or --no-lock)" >&2
      exit 4
    fi
  fi
fi

# ---------------------------------------------------------------- GR00T off: the OD3 link down for the demo
if [[ "$GROOT" == off ]]; then
  if bash "$WL/scripts/groot_link.sh" check >/dev/null 2>&1; then LINK_DOWN=1; fi
  bash "$WL/scripts/groot_link.sh" down >/dev/null 2>&1 || true
  say "GR00T off for the demo: link down, the pick uses the labelled SONIC arm-script fallback (--groot on keeps it)"
fi

# ---------------------------------------------------------------- --fresh: a new stack
viz_up() { curl -sf -m 3 "http://127.0.0.1:$VIZ_PORT/" >/dev/null; }
if [[ "$FRESH" == 1 ]]; then
  banner "fresh stack: $SCENE, profile full, viz low, page :$P5_PORT"
  t=$(date +%s)
  bash "$WL/scripts/m2_down.sh" --session "$SESSION" 2>&1 | tee -a "$RUN/stack.log" | tail -3
  if [[ "$KEEP_MEM" == 0 ]]; then
    mem=$WL/runs/memory/$SCENE.json
    if [[ -f "$mem" ]]; then
      mkdir -p "$WL/runs/memory_backup"
      mv "$mem" "$WL/runs/memory_backup/$SCENE-$TS.json"
      say "spatial memory moved to runs/memory_backup/$SCENE-$TS.json (the demo starts knowing nothing; --keep-memory keeps it)"
    fi
  fi
  if [[ "$GROOT" == on ]]; then
    bash "$WL/scripts/groot_link.sh" ensure 2>&1 | tee -a "$RUN/stack.log" | tail -2
  fi
  if ! bash "$WL/scripts/m2_up.sh" --profile full --scene "$SCENE" --viz low --p5-port "$P5_PORT" --session "$SESSION" \
       --groot "$([[ "$GROOT" == on ]] && echo link || echo off)" 2>&1 | tee -a "$RUN/stack.log" | grep -E 'READY|ERROR|stage times|m2_up.*session'; then :; fi
  grep -q 'READY' "$RUN/stack.log" || { say "m2_up.sh did not print READY (log $RUN/stack.log)"; exit 5; }
  say "stack up in $(( $(date +%s) - t )) s"
fi
if ! viz_up; then
  say "Sim Viewer (viz-server) not answering on :$VIZ_PORT: starting it"
  bash "$WL/viz/box.sh" server 0 | tee -a "$RUN/demo.log"
  for _ in $(seq 1 30); do viz_up && break; sleep 1; done
  viz_up || say "WARNING: the Sim Viewer did not come up (no recording)"
fi

# ---------------------------------------------------------------- page + System 1 ready, hands empty
banner "waiting for the page and System 1"
(cd "$WL" && "$PY_RT" -m tools.say --wait-ready 240 --json "$RUN/ready.json") 2>&1 | tee -a "$RUN/demo.log"
rc=${PIPESTATUS[0]}
(( rc == 0 )) || { say "the page or System 1 is not ready (tools.say rc $rc)"; exit 5; }
held=$("$PY_RT" -c 'import json,sys
h=(json.load(open(sys.argv[1])).get("end") or {}).get("hands") or {}
print(" ".join("%s:%s" % (a, v.get("holding")) for a, v in h.items() if v.get("holding") not in (None, "nothing", "", "UNKNOWN")))' "$RUN/ready.json")
[[ -n "$held" ]] && say "WARNING: the robot is holding something ($held): step 8 will fail (place does not work yet); use --fresh"

# ---------------------------------------------------------------- recording
REC_DIR=""
rec() { curl -s -m 120 -X POST -H 'Content-Type: application/json' -d "$1" "http://127.0.0.1:$VIZ_PORT/api/record"; }
if [[ "$RECORD" == 1 ]]; then
  out=$(rec "{\"action\":\"start\",\"label\":\"$LABEL\"}")
  REC_DIR=$(echo "$out" | "$PY_RT" -c 'import json,sys; print(json.load(sys.stdin).get("dir") or "")' 2>/dev/null)
  if [[ -n "$REC_DIR" ]]; then say "recording to $REC_DIR"; else say "WARNING: recording did not start: $out"; fi
fi

# ---------------------------------------------------------------- the steps
for n in "${STEPS[@]}"; do
  step_def "$n"
  banner "step $n: $NAME"
  ts=$(date +%s)
  (cd "$WL" && "$PY_RT" -m tools.say --idle-ignores-wait --idle-s 5 --max-s "$MAXS" --json "$RUN/step$n.json" \
      "${EXTRA[@]}" "${LINES[@]}") 2>&1 | tee "$RUN/step$n.log"
  echo "$n ${PIPESTATUS[0]} $(( $(date +%s) - ts ))" >> "$RUN/steps.tsv"
  sleep 2
done

# ---------------------------------------------------------------- stop the recording
if [[ -n "$REC_DIR" ]]; then
  say "stopping the recording (encoding the composite can take a minute)"
  rec '{"action":"stop"}' > "$RUN/record_stop.json"
  say "recording: $REC_DIR (composite.mp4, contact_sheet.png)"
  echo "$REC_DIR" > "$RUN/recording_dir"
fi

# ---------------------------------------------------------------- PASS/FAIL from the trace
banner "results"
"$PY_RT" - "$RUN" "$(( $(date +%s) - T0 ))" <<'EOF' | tee "$RUN/results.md"
import json, re, sys
from pathlib import Path

run = Path(sys.argv[1]); total_s = int(sys.argv[2])
steps = {}
for l in (run / "steps.tsv").read_text().split("\n"):
    if l.strip():
        n, rc, s = l.split()
        steps[int(n)] = (int(rc), int(s))


def load(n):
    p = run / f"step{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def after(d, k=0):
    """Trace rows from the k-th line sent on."""
    sent = d.get("sent") or []
    return d["trace"][sent[k]["trace_at"]:] if len(sent) > k else []


def heard_t(d, text):
    for r in d["trace"]:
        if r.get("type") == "heard" and r.get("text", "").strip().lower() == text.strip().lower():
            return r.get("t", 0.0)
    return None


speech = lambda rows: [r for r in rows if r.get("type") == "result" and r.get("kind") == "speech"]
said = lambda rows: " | ".join(str(r.get("text")) for r in speech(rows))
res = lambda rows, tool: [r for r in rows if r.get("type") == "result" and r.get("tool") == tool]
ok = lambda r: str(r.get("status")).lower() == "succeeded"
since = lambda rows, t: [r for r in rows if t is not None and (r.get("t_start") or r.get("t") or 0) >= t]


def j1(d):
    rows = after(d)
    notes = [r for r in rows if r.get("type") == "note_saved"]
    return bool(notes), f"note saved: {notes[0].get('text')!r}" if notes else "no note_saved row"


def j2(d):
    rows = after(d)
    nav = [r for r in res(rows, "navigate") if ok(r) and "dresser" in str((r.get("data") or {}).get("location"))]
    tq = heard_t(d, "where are you going?")
    t_arr = nav[0]["t"] if nav else None
    ans = [r for r in speech(rows) if tq is not None and (r.get("t_start") or 0) >= tq]
    while_walking = [r for r in ans if t_arr is None or (r.get("t_start") or 0) <= t_arr]
    txt = said(ans)[:160]
    if nav and while_walking:
        return True, f"{nav[0]['summary'][:70]}; answered while walking: {said(while_walking)[:120]!r}"
    return False, f"navigate ok={bool(nav)}; answer={txt!r} ({'while walking' if while_walking else 'not while walking'})"


def j3(d):
    rows = after(d)
    stops = [r for r in rows if r.get("type") == "stop" and not r.get("already_paused")]
    resumes = [r for r in rows if r.get("type") == "resume" and stops and r["t"] >= stops[0]["t"]]
    nav = [r for r in res(rows, "navigate") if ok(r) and resumes and r["t"] >= resumes[0]["t"]
           and "dining_table" in str((r.get("data") or {}).get("location"))]
    msg = f"stop={len(stops)} (canceled {stops[0].get('canceled') if stops else '-'}), resume={len(resumes)}, " \
          f"arrived after resume: {nav[0]['summary'][:60] if nav else 'no'}"
    return bool(stops and resumes and nav), msg


def j_said(pattern):
    def j(d):
        rows = after(d)
        sp = speech(rows)
        hit = [r for r in sp if re.search(pattern, str(r.get("text")), re.I)]
        return bool(hit), f"said: {said(hit or sp)[:200]!r}"
    return j


def j6(d):
    rows = after(d)
    tc = heard_t(d, "no, go to the kitchen counter instead")
    lab = [r.get("kind") for r in rows if r.get("type") == "classified" and tc is not None and r.get("t", 0) >= tc]
    nav = [r for r in res(since(rows, tc), "navigate") if ok(r) and "kitchen_counter" in str((r.get("data") or {}).get("location"))]
    return bool(nav), f"labelled {lab[:1]}; {nav[0]['summary'][:70] if nav else 'kitchen counter not reached'}"


def j7(d):
    rows = after(d)
    cr = [r for r in res(rows, "check_reachability") if "beyond_reach" in str(r.get("summary"))]
    sp = [r for r in speech(rows) if cr and (r.get("t_start") or 0) >= cr[0]["t"] - 0.5]
    return bool(cr and sp), (f"{cr[0]['summary'][:110]}; said {said(sp)[:100]!r}" if cr else
                             f"no beyond_reach; said {said(rows)[:120]!r}")


def j8(d):
    rows = after(d)
    picks = [r for r in res(rows, "manipulate") if r.get("action") == "pick"]
    good = [r for r in picks if ok(r) and (r.get("data") or {}).get("holding")]
    fb = [r for r in rows if r.get("type") == "manip.fallback"]
    hands = (d.get("end") or {}).get("hands") or {}
    held = {a: v.get("holding") for a, v in hands.items() if v.get("holding") not in (None, "nothing", "")}
    path = " -> ".join(r.get("tool") + (f"({(r.get('args') or {}).get('location')})" if r.get("tool") == "navigate" else "")
                       for r in rows if r.get("type") == "decision" and r.get("tool") not in ("speak", "wait_and_observe", "recall"))
    ex = good[0].get("summary", "")[:90] if good else (picks[-1].get("summary", "")[:120] if picks else "no pick")
    return bool(good), f"{ex}; fallbacks {[f.get('from_executor') + ':' + str(f.get('reason')) for f in fb]}; end hands {held}; path {path}"


JUDGE = {1: j1, 2: j2, 3: j3, 4: j_said(r"counter"), 5: j_said(r"dresser|counter"), 6: j6, 7: j7, 8: j8,
         9: j_said(r"bottle")}
NAMES = {1: "memory note", 2: "navigate + question mid-walk", 3: "stop / resume while walking", 4: "recall a note",
         5: "spatial memory", 6: "correction mid-walk", 7: "honest refusal (reach)", 8: "pick (reach stance + arm)",
         9: "question + chitchat"}
print(f"| step | behaviour | result | s | evidence |\n|---|---|---|---|---|")
npass = 0
for n, (rc, s) in steps.items():
    d = load(n)
    if d is None or rc != 0:
        verdict, why = False, f"tools.say rc {rc}"
    else:
        try:
            verdict, why = JUDGE[n](d)
        except Exception as e:  # noqa: BLE001
            verdict, why = False, f"judge error {e!r}"
        if d.get("status") == "max_s":
            why += " (a line hit --max-s)"
    npass += verdict
    print(f"| {n} | {NAMES[n]} | {'PASS' if verdict else 'FAIL'} | {s} | {why.replace('|', '/')} |")
print(f"\n{npass}/{len(steps)} steps passed; total {total_s // 60} min {total_s % 60} s")
sys.exit(0 if npass == len(steps) else 1)
EOF
rc=${PIPESTATUS[0]}
say "run dir $RUN${REC_DIR:+ ; recording $REC_DIR}"
exit "$rc"
