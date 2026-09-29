"""E-1 (docs/M2.md 7.5): the 17 Worldline scenarios on the `lite` profile with LIVE models, on the laptop.

    .venv-rt/bin/python -m eval.live_suite --out outputs/m2b_wave1/ui/e-1

It starts the page server (ui/server.py) on `lite` with the real planner (agent.model:create_brain, Gemini) and
the real System 1 (brains.system1_jev:create: Jev labels + the Gemini Live observer), runs eval/suite.py against
it once, reruns only the failures once, and writes:

    <out>/pass1.json, <out>/pass2.json   the suite's summaries (per scenario: pass/fail, note, executors, costs)
    <out>/traces/<pass>_<scenario>.jsonl every trace row, model call (input + response) and spoken line
    <out>/report.json, <out>/report.txt  the combined verdict per scenario and the model-call counts
    <out>/server.log                     the page server's output
    <out>/runs/                          memory, episodes, procedures and the planner's call log of this run

Keys are loaded by the page server from ~/.config/ludo-g1/secrets.env into its own environment (never printed);
this script only checks that the key names it needs are set somewhere, and never reads their values.
Bar (PLAN M1 exit, docs/M2.md E-1): >= 15/17. Everything on `lite` is a STEPPING STONE body, so every pass here
is a fallback pass (PASS*): E-1 measures the prompt and the harness, not the robot.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SECRETS = Path.home() / ".config" / "ludo-g1" / "secrets.env"
NEEDED = ("GEMINI_API_KEY", "TYPESAFE_API_KEY")
BAR = 15


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def key_names_present() -> dict[str, bool]:
    """Which needed key NAMES are set (environment or the secrets file). Values are never read into this process."""
    names: set[str] = {k for k in NEEDED if os.environ.get(k)}
    if SECRETS.is_file():
        for line in SECRETS.read_text().splitlines():
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and v.strip():
                names.add(k.strip())
    return {k: k in names for k in NEEDED}


def git_state() -> dict[str, Any]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"head": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": [ln for ln in run("status", "--porcelain").splitlines() if not ln.endswith((".venv-rt", "outputs"))]}


def start_server(out: Path, port: int, planner: str, system1: str, scene: str) -> subprocess.Popen:
    env = {**os.environ, "WORLDLINE_RUNS": str(out / "runs"), "PYTHONUNBUFFERED": "1"}
    log = (out / "server.log").open("w")
    cmd = [PY, "-m", "ui.server", "--profile", "lite", "--scene", scene, "--port", str(port),
           "--planner", planner, "--system1", system1]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(out: Path, proc: subprocess.Popen, timeout_s: float = 180.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        text = (out / "server.log").read_text(errors="replace")
        if "\nready" in text or text.startswith("ready"):
            return True
        time.sleep(1.0)
    return False


def run_suite(out: Path, port: int, tag: str, only: list[str] | None) -> dict[str, Any]:
    env = {**os.environ, "WORLDLINE_RUNS": str(out / "runs"), "PYTHONUNBUFFERED": "1"}
    cmd = [PY, "-m", "eval.suite", "--url", f"ws://127.0.0.1:{port}/ws", "--profile", "lite", "--tag", tag,
           "--out", str(out / f"{tag}.json"), "--trace-dir", str(out / "traces")]
    if only:
        cmd += ["--only", ",".join(only)]
    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    path = out / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {"results": [], "error": f"no {path.name}"}


def combine(p1: dict[str, Any], p2: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    rows = []
    rerun = {r["name"]: r for r in (p2 or {}).get("results") or []}
    cost_keys = ("model_calls", "model_errors", "classify_calls", "next_action_calls", "rule_classifies",
                 "tokens_in", "tokens_out", "s1_route_calls", "s1_observe_calls", "decisions")
    for r in p1.get("results") or []:
        again = rerun.get(r["name"])
        final = again if again is not None else r
        rows.append({"name": r["name"], "final": "pass" if final["passed"] else "fail",
                     "verdict": ("PASS*" if final["fallback_pass"] else "PASS") if final["passed"] else "FAIL",
                     "passed_first": r["passed"], "rerun": again is not None,
                     "passed_rerun": None if again is None else again["passed"],
                     "reason": final["note"], "first_reason": r["note"] if again is not None else None,
                     "executors_used": final["executors_used"], "shortcuts": final["shortcuts"],
                     "honesty": final["honesty"], "seconds": final["seconds"], "time_limit_s": final["time_limit_s"],
                     "trace": final.get("trace_path"), "first_trace": r.get("trace_path") if again is not None else None,
                     "costs": {k: int(r.get(k) or 0) + int((again or {}).get(k) or 0) for k in cost_keys}})
    passed = sum(1 for r in rows if r["final"] == "pass")
    totals = {k: sum(r["costs"][k] for r in rows) for k in cost_keys}
    return {**meta, "passed": passed, "total": len(rows), "bar": BAR, "meets_bar": passed >= BAR,
            "passed_first_pass": sum(1 for r in rows if r["passed_first"]),
            "note": "lite: every body result is a STEPPING STONE (executor 'lite'); passes are fallback passes (PASS*)",
            "costs": totals, "scenarios": rows}


def report_text(rep: dict[str, Any]) -> str:
    out = [f"E-1 · 17 scenarios on lite with live models · {rep['passed']}/{rep['total']} "
           f"({'meets' if rep['meets_bar'] else 'BELOW'} the bar {rep['bar']}/17; first pass {rep['passed_first_pass']}"
           f"/{rep['total']}) · git {rep['git']['head']} ({len(rep['git']['dirty'])} dirty paths)",
           f"planner {rep['planner']} (model {rep['model']}) · System 1 {rep['system1']} · {rep['note']}", ""]
    for r in rep["scenarios"]:
        ex = "; ".join(f"{g}: {', '.join(f'{k}x{v}' for k, v in (r['executors_used'].get(g) or {}).items())}"
                       for g in ("nav", "manip") if r["executors_used"].get(g)) or "no body result"
        again = f" (first pass FAIL: {r['first_reason']})" if r["rerun"] else ""
        c = r["costs"]
        out.append(f"{r['verdict']:<5} {r['name']:<18} {r['reason']}{again}")
        out.append(f"      executors [{ex}] · calls: planner {c['model_calls']} ({c['next_action_calls']} next, "
                   f"{c['classify_calls']} classify), System 1 label {c['s1_route_calls']}, observe {c['s1_observe_calls']}"
                   f" · tokens {c['tokens_in']} in / {c['tokens_out']} out · {r['seconds']:.0f}/{r['time_limit_s']:.0f} s")
    c = rep["costs"]
    out += ["", f"totals: planner calls {c['model_calls']} ({c['next_action_calls']} next action, {c['classify_calls']} "
                f"classify, {c['model_errors']} errors), {c['rule_classifies']} rule classifications, System 1 label calls "
                f"{c['s1_route_calls']}, System 1 observe calls {c['s1_observe_calls']}, tokens {c['tokens_in']} in / "
                f"{c['tokens_out']} out, {c['decisions']} decisions"]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m2b_wave1" / "ui" / "e-1"))
    ap.add_argument("--planner", default="agent.model:create_brain")
    ap.add_argument("--system1", default="brains.system1_jev:create")
    ap.add_argument("--scene", default="procthor-train-40")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated scenario names (default: all 17)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = key_names_present()
    if args.planner == "agent.model:create_brain" and not all(keys.values()):
        print(f"missing keys: {[k for k, ok in keys.items() if not ok]}", file=sys.stderr)
        return 2
    port = args.port or free_port()
    from ui.server import MODELS
    meta = {"stage": "E-1", "profile": "lite", "planner": args.planner, "model": MODELS[0][0], "system1": args.system1,
            "git": git_state(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "port": port}
    proc = start_server(out, port, args.planner, args.system1, args.scene)
    try:
        if not wait_ready(out, proc):
            print(f"the page server did not come up; see {out / 'server.log'}", file=sys.stderr)
            return 3
        p1 = run_suite(out, port, "pass1", [n for n in args.only.split(",") if n] or None)
        fails = [r["name"] for r in p1.get("results") or [] if not r["passed"]]
        p2 = run_suite(out, port, "pass2", fails) if fails and not args.no_rerun else None
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(20)
        except Exception:  # noqa: BLE001
            proc.kill()
    rep = combine(p1, p2, {**meta, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    text = report_text(rep)
    (out / "report.txt").write_text(text)
    print(text)
    return 0 if rep["meets_bar"] else 1


if __name__ == "__main__":
    sys.exit(main())
