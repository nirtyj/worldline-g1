"""Say lines to a running Worldline UI and print what the robot does, live. It never resets the scene.

    .venv-rt/bin/python -m tools.say "go to the dining table" "pick up the bottle"
    .venv-rt/bin/python -m tools.say "go to the bedroom dresser" "@6 stop" "@4 okay, carry on"
    .venv-rt/bin/python -m tools.say --url ws://127.0.0.1:8777/ws "where is the vase?"      # the laptop stand-in
    .venv-rt/bin/python -m tools.say --wait-ready 180                                     # only wait for System 1
    .venv-rt/bin/python -m tools.say --json out.json --idle-ignores-wait \
        --if-asked "which (one|bottle)" "the white one" "pick up the white bottle"          # scripts/demo.sh

A plain line waits until the runtime has been idle for --idle-s after the previous line (or --max-s); a line
starting with "@N " is sent N seconds after the previous line instead, e.g. to interrupt a walk with "@6 stop".
Printed: the planner's tool calls, each result with its executor label, what the robot says, what System 1
labels the line as (Jev kind), and what System 1 notices.

Options for scripts (all off by default, so the plain use above is unchanged):
  --wait-ready S        before the first line, wait up to S s for System 1 `ready` and an idle runtime (exit 3 if not)
  --idle-ignores-wait   a running wait_and_observe does not count as busy (the planner often waits for the user)
  --if-asked RE REPLY   when the robot says something matching RE (case-insensitive), say REPLY once, then keep waiting
  --json PATH           write what was sent and every trace row seen (plus the end belief: where, hands) as JSON
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
        return f"  {t:9}{r.get('source') or r.get('why') or r.get('reason') or ''}" + \
            (f" (cancelled {', '.join(r['canceled'])})" if r.get("canceled") else "")
    if t == "rejected":
        return f"  rejected {r.get('tool')}: {r.get('why') or r.get('reason')}"
    if t == "observation":
        return f"  noticed  {r.get('text')} (at {r.get('where')}, unverified)"
    if t == "recall":
        return f"  recall   {str(r.get('answer')).replace(chr(10), ' | ')}"
    return None


def _busy(run, ignore_wait: bool) -> bool:
    rt = (run.frame or {}).get("runtime") or {}
    active = rt.get("active") or []
    if ignore_wait:
        active = [e for e in active if (e.get("tool") or e.get("skill")) != "wait_and_observe"]
    return bool(active) or bool(rt.get("thinking"))


def _dump(a: argparse.Namespace, run, sent: list[dict], t_start: float, status: str) -> None:
    if not a.json:
        return
    rt = (run.frame or {}).get("runtime") or {}
    belief = rt.get("belief") or {}
    out = {"url": a.url, "status": status, "wall_s": round(time.time() - t_start, 1),
           "config": (run.init or {}).get("config", {}), "sent": sent, "trace": run.trace,
           "end": {"robot": belief.get("robot"), "hands": belief.get("hands"), "paused": rt.get("paused"),
                   "system1": (run.frame or {}).get("system1")}}
    with open(a.json, "w") as f:
        json.dump(out, f, default=str)


async def main(a: argparse.Namespace) -> int:
    from websockets.asyncio.client import connect

    from eval import suite

    t_start = time.time()
    sent: list[dict] = []
    status = "ok"
    async with connect(a.url, max_size=2 ** 24) as ws:
        run = suite.Run(ws, a.profile)
        reader = asyncio.create_task(run.reader())
        try:
            if not await run.until(lambda: run.init is not None, 30):
                print(f"no init from {a.url} (is the Worldline UI up and the tunnel open?)", file=sys.stderr)
                return 2
            cfg = run.init.get("config", {})
            print(f"connected: scene {cfg.get('scene')}, profile {cfg.get('profile')}, planner {cfg.get('model')}")
            if a.json or a.wait_ready:
                # the first frame after init carries rows the server held while no page was connected: not ours
                await run.until(lambda: (run.frame or {}).get("type") == "frame", 5)
            if a.wait_ready:
                s1 = lambda: ((run.frame or {}).get("system1") or ((run.init or {}).get("meta") or {}).get("system1")
                              or {}).get("status")
                ok = await run.until(lambda: s1() == "ready" and not _busy(run, True), a.wait_ready)
                print(f"System 1 {s1()}, runtime {'busy' if _busy(run, True) else 'idle'}"
                      + ("" if ok else f" after {a.wait_ready:.0f} s: NOT READY"), flush=True)
                if not ok:
                    _dump(a, run, sent, t_start, "not_ready")
                    return 3
            seen = len(run.trace)
            asked = [False] * len(a.if_asked)
            for i, raw in enumerate(a.lines):
                m = re.match(r"@(\d+(?:\.\d+)?)\s+(.*)", raw)
                line = m.group(2) if m else raw
                if m:                             # print what happens while waiting to send it
                    t_send = time.monotonic() + float(m.group(1))
                    while (left := t_send - time.monotonic()) > 0:
                        await asyncio.sleep(min(0.3, left))
                        for r in run.trace[seen:]:
                            if (s := _row(r)):
                                print(s, flush=True)
                        seen = len(run.trace)
                print(f"\nyou      {line}")
                sent.append({"text": line, "wall": round(time.time(), 3), "trace_at": len(run.trace)})
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
                        if r.get("type") == "result" and r.get("kind") == "speech":
                            for k, (pat, reply) in enumerate(a.if_asked):
                                if not asked[k] and re.search(pat, str(r.get("text") or ""), re.I):
                                    asked[k] = True
                                    print(f"\nyou      {reply}   (asked: /{pat}/)", flush=True)
                                    sent.append({"text": reply, "wall": round(time.time(), 3),
                                                 "trace_at": len(run.trace), "answering": pat})
                                    await run.say(reply)
                                    idle_since = None
                    seen = len(run.trace)
                    busy = _busy(run, a.idle_ignores_wait)
                    if busy:
                        idle_since = None
                    else:
                        idle_since = idle_since or time.monotonic()
                        if time.monotonic() - idle_since >= (a.idle_s if acted else a.idle_s * 2):
                            break
                else:
                    print(f"  (still busy after {a.max_s:.0f} s; moving on)")
                    status = "max_s"
            for r in run.trace[seen:]:
                s = _row(r)
                if s:
                    print(s)
            _dump(a, run, sent, t_start, status)
        finally:
            reader.cancel()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", nargs="*", help="what to say, in order; '@N text' sends N s after the previous line")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--idle-s", type=float, default=6.0, help="idle this long after a result before the next line")
    ap.add_argument("--max-s", type=float, default=300.0, help="give up waiting on one line after this long")
    ap.add_argument("--wait-ready", type=float, default=0.0, metavar="S",
                    help="first wait up to S s for System 1 ready and an idle runtime (exit 3 if not)")
    ap.add_argument("--idle-ignores-wait", action="store_true", help="a running wait_and_observe is not busy")
    ap.add_argument("--if-asked", nargs=2, action="append", default=[], metavar=("RE", "REPLY"),
                    help="when the robot says something matching RE, say REPLY (once per pair)")
    ap.add_argument("--json", default=None, metavar="PATH", help="write lines sent and trace rows seen as JSON")
    args = ap.parse_args()
    if not args.lines and not args.wait_ready:
        ap.error("say at least one line (or --wait-ready S)")
    sys.exit(asyncio.run(main(args)))
