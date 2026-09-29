"""Scenario suite: scripted conversations with the robot, scored against the simulator's truth.

Run it against a live playground server (it drives the same websocket the page uses):

    .venv-thor/bin/python ui/server.py            # in one terminal
    .venv-thor/bin/python eval/suite.py           # in another
    .venv-thor/bin/python eval/suite.py --only fetch_other_room,recall_history --tag trial

Each scenario loads a house (some wipe memory first, some rely on what earlier
scenarios left in it), says things at scripted moments, and passes or fails on
what really happened: where the object is, what the robot said, what it did.
Every session also lands in the episode log, so a suite run doubles as training
data for the procedural graph.

Writes runs/eval/<stamp>_<tag>.json and prints a table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "eval"
H40, H15 = "procthor-train-40", "procthor-train-15"
K10 = "FloorPlan10"                     # Kitchen 10: counter 1 runs counter_1b · counter_1a · stove · counter_2


class Run:
    """One websocket session and everything it has seen."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.init: dict[str, Any] | None = None
        self.frame: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self.said: list[tuple[float, str]] = []         # (runtime t, text) the robot started saying
        self.calls: dict[int, dict[str, Any]] = {}

    async def reader(self) -> None:
        async for raw in self.ws:
            m = json.loads(raw)
            if m.get("type") == "init":
                self.init, self.trace, self.said, self.calls = m, [], [], {}
            elif m.get("type") == "frame":
                self.frame = m
                self.trace += m.get("trace", [])
                for e in m.get("events", []):
                    if e.get("type") == "speech_started":
                        self.said.append((e.get("t", 0.0), e.get("text", "")))
                for c in m.get("calls", []):
                    self.calls[c["n"]] = c

    async def send(self, **msg: Any) -> None:
        await self.ws.send(json.dumps(msg))

    async def load(self, scene: str, forget: bool) -> None:
        self.init = None
        await self.send(type="persona", level="off")        # own goals would make runs less repeatable
        await self.send(type="step", mode="off")             # a step mode left on by the page would hang the run
        await self.send(type="reset", scene=scene, forget=forget)
        await self.until(lambda: self.init is not None and self.init["config"]["scene"] == scene, 150)
        await asyncio.sleep(1.0)

    async def say(self, text: str) -> None:
        await self.send(type="say", text=text)

    async def until(self, pred: Callable[[], bool], timeout: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                if pred():
                    return True
            except (KeyError, TypeError, IndexError):
                pass
            await asyncio.sleep(0.3)
        return False

    # -- what happened -------------------------------------------------
    @property
    def user_surface(self) -> str:
        return self.init["layout"]["user_surface"]

    def where(self, oid: str) -> str | None:
        return ((self.frame or {}).get("truth", {}).get("objects", {}).get(oid) or {}).get("where")

    def idle(self) -> bool:
        return not (self.frame or {}).get("runtime", {}).get("active")

    def rows(self, kind: str, **match: Any) -> list[dict[str, Any]]:
        return [r for r in self.trace if r.get("type") == kind and all(r.get(k) == v for k, v in match.items())]

    def started(self, tool: str) -> bool:
        return bool(self.rows("started", tool=tool))

    def said_since(self, t: float, pattern: str) -> bool:
        return any(st >= t and re.search(pattern, text, re.I) for st, text in self.said)

    def now(self) -> float:
        return float((self.frame or {}).get("t", 0.0))

    def metrics(self) -> dict[str, Any]:
        first_request = next((r["t"] for r in self.trace if r.get("type") == "classified"
                              and r.get("kind") in ("request", "correction")), 1e9)
        calls = [c for c in self.calls.values() if c.get("via") == "model"]
        return {
            "decisions": len(self.rows("decision")),
            "rejected": len(self.rows("rejected")),
            "stale": len(self.rows("stale_decision")),
            "recalls": len(self.rows("recall")),
            "tokens_in": sum(int(c.get("tokens_in") or 0) for c in calls),
            "unasked": sum(1 for r in self.rows("started") if r["t"] < first_request),
            "questions": sum(1 for _, text in self.said if text.rstrip().endswith("?")),
            "labels": len(self.rows("classified")),
            "s1_labels": sum(1 for r in self.rows("classified") if (r.get("directive") or {}).get("source") == "system1"),
        }

    def s1_calls(self, purpose: str) -> list[dict[str, Any]]:
        return [c for c in self.calls.values() if c.get("via") == "system1" and c.get("purpose") == purpose]

    def label(self, text: str) -> dict[str, Any]:
        """The directive the runtime used for a message (kind, source, confidence, reply ...)."""
        for r in reversed(self.rows("classified")):
            d = r.get("directive") or {}
            if d.get("text") == text:
                return {**d, "kind": r.get("kind")}
        return {}


# ----------------------------------------------------------------------
# Scenarios. Each returns (passed, note).
# ----------------------------------------------------------------------
async def fetch(r: Run, oid: str, label: str, timeout: float = 200) -> tuple[bool, str]:
    await r.say(f"Bring me the {label}.")
    ok = await r.until(lambda: r.where(oid) == r.user_surface and r.idle(), timeout)
    return ok, f"{oid} ended on {r.where(oid)}"


async def fetch_other_room(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=True)
    return await fetch(r, "alarm_clock_1", "alarm clock")


async def fetch_search(r: Run) -> tuple[bool, str]:
    await r.load(H15, forget=True)
    return await fetch(r, "apple_1", "apple")


async def correction(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    await r.say("Bring me a book.")
    await r.until(lambda: r.started("navigate"), 40)
    await r.say("No, bring me the alarm clock instead.")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), 200)
    books = [b for b in ("book_1", "book_2") if r.where(b) == r.user_surface]
    return ok and not books, f"alarm clock on {r.where('alarm_clock_1')}; books delivered: {books or 'none'}"


async def stop_resume(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), 40)
    await asyncio.sleep(2.0)
    await r.say("stop")
    stopped = await r.until(lambda: bool(r.rows("stop")), 10)
    await asyncio.sleep(5.0)
    await r.say("Okay, carry on.")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), 200)
    acks = sum(1 for _, t in r.said if "stopped" in t.lower())
    return ok and stopped and acks == 1, f"halted={stopped}, delivered={ok}, 'stopped' said {acks}x"


async def remember_where(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)                 # memory from the scenarios above
    t0 = r.now()
    await r.say("Where did you put the alarm clock last time?")
    ok = await r.until(lambda: r.said_since(t0, r"table|kitchen"), 30)
    moved = r.started("navigate")
    return ok and not moved, f"answered from memory={ok}, drove first={moved}"


async def recall_history(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    t0 = r.now()
    await r.say("What did I ask you to bring me before?")
    ok = await r.until(lambda: r.said_since(t0, r"alarm clock|book|banana"), 30)   # everything the suite asks for in house 40
    return ok, f"named an earlier request={ok}, recalls={len(r.rows('recall'))}"


async def question_midtask(r: Run) -> tuple[bool, str]:
    await r.load(H15, forget=False)
    await r.say("Bring me the apple.")
    await r.until(lambda: r.started("navigate"), 40)
    t0 = r.now()
    await r.say("What are you holding right now?")
    answered = await r.until(lambda: any(st >= t0 for st, _ in r.said), 20)
    ok = await r.until(lambda: r.where("apple_1") == r.user_surface and r.idle(), 200)
    return ok and answered, f"answered={answered}, delivered={ok}"


async def note_only(r: Run) -> tuple[bool, str]:
    await r.load(H15, forget=False)
    await r.say("By the way, my keys are usually on the kitchen counter.")
    await asyncio.sleep(15)
    noted = bool(r.rows("note_saved"))
    moved = r.started("navigate") or r.started("pick")
    return noted and not moved, f"noted={noted}, moved={moved}"


async def unsupported(r: Run) -> tuple[bool, str]:
    await r.load(H15, forget=False)
    t0 = r.now()
    await r.say("Put the apple in the microwave.")
    ok = await r.until(lambda: r.said_since(t0, r"can't|cannot|can not|unable|not able|don't have a way"), 40)
    return ok, f"said it can't={ok}"


async def missing_object(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    t0 = r.now()
    await r.say("Bring me the banana.")                  # there is no banana in this house
    told = await r.until(lambda: r.said_since(t0, r"can't find|couldn't find|could not find|no banana|"
                                                   r"not find|didn't find|don't see|haven't found|isn't here|not here"), 150)
    await r.until(r.idle, 20)
    looks = len(r.rows("started", tool="navigate"))
    return told, f"told you it isn't here={told}, drove to {looks} spots, {r.now() - t0:.0f} s"


# ----------------------------------------------------------------------
# System 1, memory layers and the harder conversation turns
# ----------------------------------------------------------------------
# Words for things you could pick up. An observation naming one the house doesn't have is a hallucination.
THING_WORDS = {"apple", "banana", "orange", "bread", "egg", "tomato", "potato", "lettuce", "mug", "cup", "bowl",
               "plate", "book", "laptop", "phone", "cell phone", "remote", "remote control", "keys", "key chain",
               "pen", "pencil", "bottle", "wine bottle", "spoon", "fork", "knife", "butter knife", "spatula",
               "sponge", "dish sponge", "towel", "cloth", "pillow", "newspaper", "watch", "vase", "statue", "box",
               "alarm clock", "basketball", "teddy bear", "spray bottle", "soap", "tissue box", "candle", "pot", "pan"}


def _types_present(r: Run) -> set[str]:
    objs = ((r.frame or {}).get("truth") or {}).get("objects") or {}
    out = set()
    for v in objs.values():
        t = str(v.get("type") or "").replace("_", " ")
        out |= {t, t.split()[-1], t.replace(" ", "")}          # "basket ball" is also "basketball"
    return out | {"phone" if "cell phone" in out else "", "keys" if "key chain" in out else "",
                  "remote" if "remote control" in out else "", "sponge" if "dish sponge" in out else ""}


async def hold_on(r: Run) -> tuple[bool, str]:
    """A stop the keyword check misses: only System 1's label can stop the robot."""
    await r.load(H40, forget=False)
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), 40)
    await asyncio.sleep(2.0)
    t_say = time.monotonic()
    await r.say("hang on a sec")
    stopped = await r.until(lambda: bool(r.rows("stop", reason="classified")), 10)
    took = time.monotonic() - t_say
    lab = r.label("hang on a sec")
    await asyncio.sleep(3.0)
    await r.say("okay, go ahead")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), 200)
    return (ok and stopped and lab.get("source") == "system1",
            f"stopped on the label={stopped} after {took:.1f}s (label {lab.get('kind')} from {lab.get('source')}, "
            f"P={lab.get('confidence')}), delivered={ok}")


async def replace_task(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    await r.say("Bring me a book.")
    await r.until(lambda: r.started("navigate"), 40)
    text = "Never mind the book, get me the alarm clock."
    await r.say(text)
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), 200)
    books = [b for b in ("book_1", "book_2") if r.where(b) == r.user_surface]
    lab = r.label(text)
    return (ok and not books,
            f"alarm clock on {r.where('alarm_clock_1')}; books delivered: {books or 'none'}; "
            f"label {lab.get('kind')} from {lab.get('source')}")


async def addition(r: Run) -> tuple[bool, str]:
    await r.load(H40, forget=False)
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), 40)
    text = "Also bring me a book."
    await r.say(text)
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle()
                       and any(r.where(b) == r.user_surface for b in ("book_1", "book_2")), 320)
    books = [b for b in ("book_1", "book_2") if r.where(b) == r.user_surface]
    lab = r.label(text)
    return ok, (f"alarm clock on {r.where('alarm_clock_1')}, books delivered: {books or 'none'}; "
                f"label {lab.get('kind')} from {lab.get('source')}")


async def observations(r: Run) -> tuple[bool, str]:
    """System 1 watches the camera during a fetch: it must look, and never report a thing the house lacks."""
    await r.load(H40, forget=False)
    n0 = len(r.s1_calls("observe"))
    delivered, where = await fetch(r, "alarm_clock_1", "alarm clock")
    obs = [row for row in r.rows("observation") if row.get("source") == "system1"]
    present = _types_present(r)
    fake = []
    for o in obs:
        text = str(o.get("text", "")).lower()
        for w in THING_WORDS:
            # "a round orange object": a colour, not the fruit
            if re.search(rf"\b{w}s?\b(?!\s+(object|thing|item|shape|ball|blob|box|cloth))", text) and w not in present:
                fake.append(f"{w} ({o.get('text')})")
    kps = set(((r.init or {}).get("map") or {}).get("keypoints") or {})
    bad_where = [o.get("where") for o in obs if o.get("where") not in kps and o.get("where") is not None]
    looked = len(r.s1_calls("observe")) - n0
    return (delivered and looked > 0 and not fake and not bad_where,
            f"{where}; observe calls {looked}, observations {len(obs)} "
            f"({'; '.join(str(o.get('text')) + ' @ ' + str(o.get('where')) for o in obs[:4]) or '-'}), "
            f"not in the house: {fake or 'none'}, unknown places: {bad_where or 'none'}")


async def procedural(r: Run) -> tuple[bool, str]:
    """The planner gets what the procedural graph learned from earlier episodes."""
    await r.load(H40, forget=False)
    n0 = max(r.calls or {0: None})
    ok, note = await fetch(r, "alarm_clock_1", "alarm clock")
    inputs = [str(c.get("input") or "") for n, c in r.calls.items() if n > n0 and c.get("via") == "model"]
    guided = sum("LEARNED FROM PAST TASKS" in i for i in inputs)
    return ok and guided > 0, f"{note}; planner calls with learned guidance: {guided}/{len(inputs)}"


async def permission_yes(r: Run) -> tuple[bool, str]:
    """In a house it hasn't mapped, the robot asks to look around; a yes (labelled by System 1) starts it."""
    await r.load(H15, forget=True)
    t0 = r.now()
    await r.send(type="persona", level="optimize")
    asked = await r.until(lambda: any(st >= t0 and text.rstrip().endswith("?") for st, text in r.said), 90)
    if not asked:
        await r.send(type="persona", level="off")
        return False, "the robot never asked"
    text = "sure, have a look"
    await r.say(text)
    roams = await r.until(lambda: any(g.get("drive") == "map" for g in r.rows("persona_goal")), 60)
    lab = r.label(text)
    await r.send(type="persona", level="off")
    return (roams and lab.get("reply") == "yes",
            f"asked={asked}, answer labelled {lab.get('kind')}/{lab.get('reply')} by {lab.get('source')}, "
            f"started exploring={roams}")


async def other_side(r: Run) -> tuple[bool, str]:
    """"The other side of the stove" needs to know what the stove sits between. The spatula starts
    on counter_1a, right of it is the stove, then counter_2. Memory is kept (it's your kitchen)."""
    await r.load(K10, forget=False)
    n0 = max(r.calls or {0: None})
    start = r.where("spatula_1")
    await r.say("move the spatula to the other side of the stove")
    ok = await r.until(lambda: r.where("spatula_1") == "counter_2" and r.idle(), 200)
    inputs = [str(c.get("input") or "") for n, c in r.calls.items() if n > n0 and c.get("via") == "model"]
    knew = any("stove_1 (stove) is between counter_1a and counter_2" in i for i in inputs)
    checks = [f"{c.get('goal')!r}: {c.get('ok')}" for c in r.rows("goal_check")]
    return ok and start == "counter_1a", (f"spatula_1 from {start} to {r.where('spatula_1')}; "
                                         f"LAYOUT had the stove between counter_1a and counter_2: {knew}; "
                                         f"goal checks: {', '.join(checks) or 'none'}")


SCENARIOS: list[tuple[str, Callable[[Run], Awaitable[tuple[bool, str]]]]] = [
    ("fetch_other_room", fetch_other_room),
    ("correction", correction),
    ("stop_resume", stop_resume),
    ("remember_where", remember_where),
    ("recall_history", recall_history),
    ("missing_object", missing_object),
    ("fetch_search", fetch_search),
    ("question_midtask", question_midtask),
    ("note_only", note_only),
    ("unsupported", unsupported),
    ("hold_on", hold_on),
    ("replace_task", replace_task),
    ("addition", addition),
    ("observations", observations),
    ("procedural", procedural),
    ("permission_yes", permission_yes),
    ("other_side", other_side),
]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    results = []
    async with connect(args.url, max_size=2 ** 24) as ws:
        run = Run(ws)
        reader = asyncio.create_task(run.reader())
        for name, fn in SCENARIOS:
            if only and name not in only:
                continue
            t0 = time.monotonic()
            try:
                passed, note = await fn(run)
            except Exception as e:                         # a broken scenario fails, the suite goes on
                passed, note = False, f"error: {e!r}"
            res = {"name": name, "passed": passed, "seconds": round(time.monotonic() - t0, 1),
                   "note": note, **run.metrics()}
            results.append(res)
            print(f"{'PASS' if passed else 'FAIL'}  {name:<18} {res['seconds']:>6.1f}s  decisions {res['decisions']:>3}  "
                  f"rejected {res['rejected']}  recalls {res['recalls']}  tokens {res['tokens_in']:>6}  "
                  f"labels {res['s1_labels']}/{res['labels']} by System 1  {note}", flush=True)
        reader.cancel()
    summary = {"tag": args.tag, "wall": round(time.time()), "passed": sum(r["passed"] for r in results),
               "total": len(results), "decisions": sum(r["decisions"] for r in results),
               "tokens_in": sum(r["tokens_in"] for r in results), "results": results}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.tag}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"\n{summary['passed']}/{summary['total']} passed · {summary['decisions']} decisions · "
          f"{summary['tokens_in']} tokens in · {path.relative_to(ROOT)}")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
