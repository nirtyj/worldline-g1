"""Say lines to a running Worldline UI and print what the robot does, live. It never resets the scene.

    .venv-rt/bin/python -m tools.say "go to the dining table" "pick up the bottle"
    .venv-rt/bin/python -m tools.say "go to the bedroom dresser" "@6 stop" "@4 okay, carry on"
    .venv-rt/bin/python -m tools.say --url ws://127.0.0.1:8777/ws "where is the vase?"      # the laptop stand-in

A plain line waits until the runtime has been idle for --idle-s after the previous line (or --max-s); a line
starting with "@N " is sent N seconds after the previous line instead, e.g. to interrupt a walk with "@6 stop".
Printed: the planner's tool calls, each result with its executor label, what the robot says, what System 1
labels the line as (Jev kind), and what System 1 notices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time

URL = "ws://127.0.0.1:8766/ws"


def _row(r: dict) -> str | None:
    t = r.get("type")
    if t == "decision":
        call = r.get("call") or {}
        tool = call.get("tool") or r.get("tool")
        args = call.get("args") or r.get("args") or {}
        return f"  plan     {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
    if t == "result":
        if r.get("kind") == "speech":
            return f'  robot    "{r.get("text") or r.get("summary")}"'
        if r.get("tool") == "observe" and r.get("action") == "glance":
            return None
        ex = r.get("executor")
        return f"  {str(r.get('status')):10}{r.get('tool')}: {r.get('summary')}" + (f"  [{ex}]" if ex else "")
    if t == "classified":
        d = r.get("directive") or {}
        src = d.get("source") or "planner"
        tgt = (d.get("target") or {}).get("object") if isinstance(d.get("target"), dict) else d.get("target")
        conf = d.get("confidence")
        return f"  label    {r.get('kind')} (by {'Jev' if src == 'system1' else src}" + \
               (f", target {tgt}" if tgt else "") + (f", conf {conf:.2f}" if isinstance(conf, (int, float)) else "") + ")"
    if t == "note_saved":
        return f"  memory   note saved: {r.get('text') or r.get('note')}"
    if t in ("stop", "resume"):
        return f"  {t:9}{r.get('source') or r.get('why') or ''}"
    if t == "rejected":
        return f"  rejected {r.get('tool')}: {r.get('why') or r.get('reason')}"
    if t == "observation":
        return f"  noticed  {r.get('text')} (at {r.get('where')}, unverified)"
    if t == "recall":
        return f"  recall   {str(r.get('answer')).replace(chr(10), ' | ')}"
    return None


async def main(a: argparse.Namespace) -> int:
    from websockets.asyncio.client import connect

    from eval import suite

    async with connect(a.url, max_size=2 ** 24) as ws:
        run = suite.Run(ws, a.profile)
        reader = asyncio.create_task(run.reader())
        try:
            if not await run.until(lambda: run.init is not None, 30):
                print(f"no init from {a.url} (is the Worldline UI up and the tunnel open?)", file=sys.stderr)
                return 2
            cfg = run.init.get("config", {})
            print(f"connected: scene {cfg.get('scene')}, profile {cfg.get('profile')}, planner {cfg.get('model')}")
            seen = len(run.trace)
            for i, raw in enumerate(a.lines):
                m = re.match(r"@(\d+(?:\.\d+)?)\s+(.*)", raw)
                line = m.group(2) if m else raw
                if m:
                    await asyncio.sleep(float(m.group(1)))
                print(f"\nyou      {line}")
                await run.say(line)
                if i + 1 < len(a.lines) and a.lines[i + 1].startswith("@"):
                    continue                      # timed follow-up: don't wait for idle
                t0 = time.monotonic()
                idle_since, acted = None, False
                while time.monotonic() - t0 < a.max_s:
                    await asyncio.sleep(0.3)
                    for r in run.trace[seen:]:
                        s = _row(r)
                        if s:
                            print(s, flush=True)
                        acted = acted or r.get("type") in ("decision", "result")
                    seen = len(run.trace)
                    busy = not run.idle() or (run.frame or {}).get("runtime", {}).get("thinking")
                    if busy:
                        idle_since = None
                    else:
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since >= (a.idle_s if acted else a.idle_s * 2):
                            break
                else:
                    print(f"  (still busy after {a.max_s:.0f} s; moving on)")
            for r in run.trace[seen:]:
                s = _row(r)
                if s:
                    print(s)
        finally:
            reader.cancel()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="+", help="what to say, in order; '@N text' sends N s after the previous line")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--idle-s", type=float, default=6.0, help="idle this long after a result before the next line")
    ap.add_argument("--max-s", type=float, default=300.0, help="give up waiting on one line after this long")
    sys.exit(asyncio.run(main(ap.parse_args())))
