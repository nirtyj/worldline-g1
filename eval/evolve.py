"""Evolve the procedural graph: propose rules from past failures, keep them only if they help.

    .venv/bin/python eval/evolve.py                   # needs the playground server running
    .venv/bin/python eval/evolve.py --baseline runs/eval/<stamp>_baseline.json

1. Learn the graph from every episode on disk, and ask the refiner model for up
   to three candidate rules (agent/procedures.py propose).
2. Score the suite without them (or reuse a baseline run).
3. Put the candidates on trial (the server reads the graph at every room load)
   and score the suite again.
4. Keep the rules if the trial passes at least as many scenarios, with no more
   than 10% more planner calls; otherwise reject them.

As in the Procedural Graphs paper, an edit is committed only when it doesn't hurt
held-out results, and a rejected edit stays in the store so it isn't proposed blindly again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def procedures(cmd: str) -> str:
    out = subprocess.run([PY, "-m", "agent.procedures", cmd], cwd=ROOT, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip()[-2000:])
    return out.stdout


def suite(tag: str, only: str) -> dict:
    cmd = [PY, "eval/suite.py", "--tag", tag] + (["--only", only] if only else [])
    subprocess.run(cmd, cwd=ROOT)
    latest = sorted((ROOT / "runs" / "eval").glob(f"*_{tag}.json"))[-1]
    return json.loads(latest.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="a suite result to compare against instead of running one")
    ap.add_argument("--only", default="", help="limit both suite runs to these scenarios")
    ap.add_argument("--no-propose", action="store_true", help="try the candidates already in the store")
    args = ap.parse_args()

    print("== 1. learn and propose")
    out = procedures("show" if args.no_propose else "propose")
    if "candidate " not in out and "[candidate]" not in out:
        print("no candidate rules proposed; nothing to try")
        return
    print("\n== 2. baseline")
    base = json.loads(Path(args.baseline).read_text()) if args.baseline else suite("baseline", args.only)
    print("\n== 3. trial")
    procedures("trial")
    trial = suite("trial", args.only)
    better = trial["passed"] >= base["passed"] and trial["decisions"] <= base["decisions"] * 1.1
    print(f"\nbaseline {base['passed']}/{base['total']} passed, {base['decisions']} decisions · "
          f"trial {trial['passed']}/{trial['total']} passed, {trial['decisions']} decisions")
    print("== 4.", "keep the rules" if better else "reject the rules")
    procedures("keep" if better else "reject")


if __name__ == "__main__":
    main()
