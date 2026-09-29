"""How well System 1 labels messages: a fixed set of labelled messages, each with the
robot's situation, run through a real Gemini Live session (GEMINI_API_KEY from .env;
a few cents a run).

    .venv-thor/bin/python -m eval.system1_routes            # all cases
    .venv-thor/bin/python -m eval.system1_routes stop       # only cases whose group contains "stop"
    .venv-thor/bin/python -m eval.system1_routes --jev      # label with Jev (TYPESAFE_API_KEY)

Prints each miss, accuracy per group, confidence on right vs wrong labels (is 0.5 a
useful threshold?), and latency against the 2 s budget.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass

from brains.system1_jev import JevSystemOne
from brains.system1 import MODEL, ROUTE_TIMEOUT_S, SystemOne
from tests.system1_live_check import load_env

IDLE = {"at": "living_room_sofa_1", "holding": {}, "goal": None, "running": [], "stopped": False}
FETCHING = {"at": None, "moving": True, "holding": {}, "goal": "bring the user the white mug",
            "running": ["navigate"], "stopped": False}
CARRYING = {"at": None, "moving": True, "holding": {"right": "mug_1"}, "goal": "bring the user the white mug",
            "running": ["navigate"], "stopped": False}
STOPPED = {**FETCHING, "moving": False, "running": [], "stopped": True}
WHICH = "Which mug: the blue one or the white one?"
CHECK = "I didn't find it here. Want me to check the kitchen?"


@dataclass
class Case:
    group: str
    text: str
    ok: tuple[str, ...]                 # acceptable kinds
    state: dict
    said: str | None = None             # the robot's last line
    yes: bool | None | str = "any"      # expected says_yes, or "any"


CASES = [
    Case("stop", "stop", ("stop",), FETCHING),
    Case("stop", "STOP!!", ("stop",), CARRYING),
    Case("stop", "hold on", ("stop",), FETCHING),
    Case("stop", "wait wait wait", ("stop",), CARRYING),
    Case("stop", "hang on a sec", ("stop",), FETCHING),
    Case("stop", "freeze", ("stop",), CARRYING),
    Case("stop", "whoa whoa", ("stop",), CARRYING),
    Case("stop", "stpo", ("stop",), FETCHING),
    Case("stop", "para!", ("stop",), FETCHING),
    Case("stop", "arrête", ("stop",), CARRYING),
    Case("not-stop", "don't stop", ("resume", "chitchat", "constraint"), FETCHING),
    Case("not-stop", "stop by the kitchen on the way", ("addition", "constraint"), FETCHING),
    Case("not-stop", "wait, are you going to the kitchen?", ("question",), FETCHING),
    Case("not-stop", "no need to stop at the table", ("constraint", "correction"), FETCHING),
    Case("correction", "no, the red one", ("correction",), FETCHING),
    Case("correction", "actually, bring the apple instead", ("correction",), FETCHING),
    Case("correction", "not that mug, the other one", ("correction",), CARRYING),
    Case("correction", "never mind, forget it", ("correction", "stop"), FETCHING),
    Case("correction", "bring it to the bedroom instead", ("correction",), CARRYING),
    Case("addition", "and a spoon too", ("addition",), FETCHING),
    Case("addition", "also check if the stove is on", ("addition",), CARRYING),
    Case("question", "where are you going?", ("question",), FETCHING),
    Case("question", "is the mug clean?", ("question",), CARRYING),
    Case("question", "what are you holding", ("question",), CARRYING),
    Case("answer", "the blue one", ("answer",), STOPPED, said=WHICH, yes=None),
    Case("answer", "the one on the left", ("answer",), STOPPED, said=WHICH, yes=None),
    Case("answer", "yeah sure", ("answer",), STOPPED, said=CHECK, yes=True),
    Case("answer", "nah", ("answer",), STOPPED, said=CHECK, yes=False),
    Case("answer", "maybe later", ("answer",), STOPPED, said=CHECK, yes=False),
    Case("answer", "have a look", ("answer",), STOPPED, said=CHECK, yes=True),
    Case("resume", "ok continue", ("resume",), STOPPED),
    Case("resume", "you can keep going", ("resume",), STOPPED),
    Case("constraint", "don't go into the bedroom", ("constraint",), FETCHING),
    Case("constraint", "be careful with it, it's hot", ("constraint",), CARRYING),
    Case("constraint", "use your left hand", ("constraint",), FETCHING),
    Case("observation", "the keys are in the drawer", ("observation",), IDLE),
    Case("observation", "I moved the mug to the sink", ("observation", "correction"), FETCHING),
    Case("chitchat", "good job!", ("chitchat",), IDLE),
    Case("chitchat", "hello there", ("chitchat",), IDLE),
    Case("request", "bring me the cup", ("request",), IDLE),
    Case("request", "brign me teh cup plz", ("request",), IDLE),
    Case("request", "what's on the dining table?", ("request", "question"), IDLE),
    Case("ambiguous", "no", ("correction", "stop", "answer", "chitchat"), FETCHING),
    Case("ambiguous", "yes", ("answer", "chitchat", "resume"), FETCHING),
    Case("ambiguous", "bring me an apple", ("request", "addition", "correction"), CARRYING),
]


async def main(only: str | None) -> int:
    load_env()
    cases = [c for c in CASES if only is None or only in c.group]
    if JEV:
        s1 = JevSystemOne(os.environ.get("TYPESAFE_API_KEY", ""), os.environ.get("GEMINI_API_KEY", ""),
                          model=os.environ.get("SYSTEM1_MODEL", MODEL), route_timeout=6.0)
    else:
        s1 = SystemOne(os.environ.get("GEMINI_API_KEY", ""), model=os.environ.get("SYSTEM1_MODEL", MODEL),
                       route_timeout=6.0)                   # measure the real latency; the budget is applied below
    task = asyncio.create_task(s1.run())
    t0 = time.monotonic()
    while s1.status != "ready":
        if s1.status == "error" or time.monotonic() - t0 > 20:
            print("System 1 not ready:", s1.status, s1.detail)
            return 1
        await asyncio.sleep(0.1)
    rows = []
    neutral = {id(IDLE): "Ready when you are.", id(FETCHING): "I'll bring you the mug.",
               id(CARRYING): "I've got the mug; bringing it to you.", id(STOPPED): "I've stopped."}
    for c in cases:
        t_wait = time.monotonic()
        while s1.router.status != "ready" and time.monotonic() - t_wait < 15:   # a fresh session is starting
            await asyncio.sleep(0.05)
        await s1.update(c.state)
        await s1.robot_said(c.said or neutral[id(c.state)])   # every case says what the robot said last
        t = time.monotonic()
        r = await s1.route(c.text)
        dt = time.monotonic() - t
        kind = r["kind"] if r else None
        right = kind in c.ok and (c.yes == "any" or (r and r["says_yes"] == c.yes))
        rows.append((c, r, dt, right))
        if not right:
            print(f"  MISS [{c.group}] {c.text!r}: got {kind} (says_yes {r and r['says_yes']}, "
                  f"confidence {r and r['confidence']}), wanted {'/'.join(c.ok)}"
                  f"{'' if c.yes == 'any' else f' with says_yes {c.yes}'}  ({dt:.2f}s)")
    task.cancel()
    rs = s1.router.stats
    print(f"router: {rs.rotations} fresh sessions, {rs.reconnects} reconnects, last error: {rs.last_error or 'none'}")
    groups = sorted({c.group for c in cases})
    print("\nper group: " + ", ".join(f"{g} {sum(r[3] for r in rows if r[0].group == g)}/"
                                      f"{sum(1 for r in rows if r[0].group == g)}" for g in groups))
    right = [r for r in rows if r[3]]
    print(f"right {len(right)}/{len(rows)}")
    conf_ok = [r[1]["confidence"] for r in rows if r[3] and r[1]]
    conf_bad = [r[1]["confidence"] for r in rows if not r[3] and r[1]]
    if conf_ok:
        print(f"confidence when right: min {min(conf_ok):.2f} median {statistics.median(conf_ok):.2f}; "
              f"when wrong: {[round(x, 2) for x in conf_bad]}  (below 0.5 goes to the planner)")
    lat = sorted(r[2] for r in rows)
    over = sum(1 for x in lat if x > ROUTE_TIMEOUT_S)
    print(f"latency p50 {lat[len(lat) // 2]:.2f}s p90 {lat[int(len(lat) * 0.9)]:.2f}s max {lat[-1]:.2f}s; "
          f"{over} of {len(lat)} over the {ROUTE_TIMEOUT_S}s budget (those would go to the planner)")
    return 0


JEV = "--jev" in sys.argv

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--jev"]
    raise SystemExit(asyncio.run(main(args[0] if args else None)))
