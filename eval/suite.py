"""Scenario suite: scripted conversations with the robot, scored against the simulator's truth.

Run it against a live page server (it drives the same websocket the page uses):

    .venv/bin/python -m ui.server --profile sonic                     # in one terminal (on the box)
    .venv/bin/python eval/suite.py --profile sonic                    # in another
    .venv/bin/python eval/suite.py --profile lite --only fetch_other_room,recall_history --tag trial

Each scenario loads a house (some wipe memory first, some rely on what earlier scenarios left in it), says
things at scripted moments, and passes or fails on what really happened: where the object is, what the robot
said, what it did. The houses and object ids come from eval/scenes.yaml (MolmoSpaces H40, H15 and Kitchen 10),
time limits are scaled for a humanoid (eval/scenes.py time_limit), and every pass records which executors
produced it: a pass that rests on a sim shortcut (kinematic base, attach grasp) is a fallback pass, never a
target pass (PLAN 2.3, 12.2).

The referee reads ground truth only as the page shows it: frame.truth, which the page builds from
world.truth(), the one ground-truth reader on the runtime side. It never talks to the simulator.

Every session also lands in the episode log, so a suite run doubles as training data for the procedural graph.
Writes runs/eval/<stamp>_<profile>_<tag>.json and prints a table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import scenes as sc  # noqa: E402

OUT = Path(os.environ.get("WORLDLINE_RUNS") or ROOT / "runs") / "eval"
LOAD_TIMEOUT_S = 300.0                   # an Isaac reset: band on, reset_scene, robot reset, band release (PLAN 9.1)
BIND = sc.load()
H40, H15, K10 = BIND.scene("H40"), BIND.scene("H15"), BIND.scene("K10")


class Run:
    """One websocket session and everything it has seen."""

    def __init__(self, ws: Any, profile: str = "lite", scale: float | None = None, original: bool = False,
                 g1_alternative: bool | None = None) -> None:
        self.ws = ws
        self.profile = profile
        self.scale = scale
        self.original = original
        self.g1_alternative = g1_alternative           # None: scenes.yaml's g1_alternative on the G1 profiles only
        self.init: dict[str, Any] | None = None
        self.frame: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self.said: list[tuple[float, str]] = []         # (runtime t, text) the robot started saying
        self.calls: dict[int, dict[str, Any]] = {}
        self.frames_seen: list[dict[str, Any]] = []     # for executors: a thin copy of each frame's executions
        self.current: str = ""                          # the scenario running now
        self.fixtures: dict[str, Any] = {}

    async def reader(self) -> None:
        async for raw in self.ws:
            self.ingest(json.loads(raw))

    def ingest(self, m: dict[str, Any]) -> None:
        if m.get("type") == "init":
            self.init, self.trace, self.said, self.calls = m, [], [], {}
            self.frame = m
        elif m.get("type") == "frame":
            self.frame = m
            self.trace += m.get("trace", [])
            for e in m.get("events", []):
                if e.get("type") == "speech_started":
                    self.said.append((e.get("t", 0.0), e.get("text", "")))
            for c in m.get("calls", []):
                self.calls[c["n"]] = c
            rt = m.get("runtime") or {}
            if rt.get("executions") or rt.get("recent"):
                self.frames_seen.append({"runtime": {"executions": rt.get("executions") or [],
                                                     "recent": rt.get("recent") or []}})
                del self.frames_seen[:-50]

    async def send(self, **msg: Any) -> None:
        await self.ws.send(json.dumps(msg))

    async def load(self, scene: str, forget: bool) -> None:
        self.init = None
        await self.send(type="persona", level="off")        # own goals would make runs less repeatable
        await self.send(type="step", mode="off")             # a step mode left on by the page would hang the run
        await self.send(type="reset", scene=scene, forget=forget, profile=self.profile)
        ok = await self.until(lambda: self.init is not None and self.init["config"]["scene"] == scene
                              and self.init["config"].get("profile", self.profile) == self.profile, LOAD_TIMEOUT_S)
        if not ok:
            raise TimeoutError(f"{scene} did not load within {LOAD_TIMEOUT_S:.0f} s")
        await asyncio.sleep(1.0)
        self.trace_mark = len(self.trace)

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

    # -- binding and time ----------------------------------------------
    def binding(self, name: str | None = None) -> tuple[str, dict[str, Any]]:
        """(which, scenario) as this run plays it (eval/scenes.py Bindings.binding)."""
        return BIND.binding(name or self.current, self.profile, self.original, self.g1_alternative)

    def limit(self, name: str | None = None) -> float:
        """This scenario's time limit on this profile (THOR limit x time scale, >= 2 x humanoid estimate)."""
        name = name or self.current
        return BIND.time_limit(name, self.profile, self.scale, self.binding(name)[1])[0]

    def wait(self, seconds: float) -> float:
        """A body-dependent wait inside a script, scaled like the limits."""
        return seconds * BIND.wait_scale(self.profile, self.scale)

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

    def started(self, tool: str, action: str | None = None) -> bool:
        """A tool started, e.g. started("navigate"), started("manipulate", action="pick")."""
        return any(action is None or r.get("action") == action or (r.get("args") or {}).get("action") == action
                   for r in self.rows("started", tool=tool))

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
            "late": len(self.rows("late_result")),
            "recalls": len(self.rows("recall")),
            "tokens_in": sum(int(c.get("tokens_in") or 0) for c in calls),
            "unasked": sum(1 for r in self.rows("started") if r["t"] < first_request),
            "questions": sum(1 for _, text in self.said if text.rstrip().endswith("?")),
            "labels": len(self.rows("classified")),
            "s1_labels": sum(1 for r in self.rows("classified") if (r.get("directive") or {}).get("source") == "system1"),
            **self.costs(),
        }

    def costs(self) -> dict[str, int]:
        """What this scenario cost in model calls (the page's call list: the planner's calls through
        llmkit, rule-decided classifications, and System 1's label and observe calls)."""
        calls = list(self.calls.values())
        model = [c for c in calls if c.get("via") == "model"]
        return {"model_calls": len(model),
                "model_errors": sum(1 for c in model if c.get("status") == "error"),
                "classify_calls": sum(1 for c in model if c.get("purpose") == "classify"),
                "next_action_calls": sum(1 for c in model if c.get("purpose") == "next_action"),
                "rule_classifies": sum(1 for c in calls if c.get("via") == "rule"),
                "tokens_out": sum(int(c.get("tokens_out") or 0) for c in model),
                "s1_route_calls": len(self.s1_calls("route")), "s1_observe_calls": len(self.s1_calls("observe"))}

    def dump(self, path: Path) -> Path:
        """This scenario's trace rows and model calls (no system prompts or schemas) as JSON lines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = ("n", "via", "purpose", "status", "t_start", "t_end", "latency_s", "tokens_in", "tokens_out",
                "version_start", "version_end", "input", "response", "error")
        with path.open("w") as f:
            for r in self.trace:
                f.write(json.dumps({"src": "trace", **r}, default=str) + "\n")
            for _, c in sorted(self.calls.items()):
                f.write(json.dumps({"src": "call", **{k: c.get(k) for k in keep}}, default=str) + "\n")
            for t, text in self.said:
                f.write(json.dumps({"src": "said", "t": t, "text": text}) + "\n")
        return path

    def executors(self) -> dict[str, dict[str, int]]:
        return sc.executors_used(self.trace, self.frames_seen)

    def s1_calls(self, purpose: str) -> list[dict[str, Any]]:
        return [c for c in self.calls.values() if c.get("via") == "system1" and c.get("purpose") == purpose]

    def label(self, text: str) -> dict[str, Any]:
        """The directive the runtime used for a message (kind, source, confidence, reply ...)."""
        for r in reversed(self.rows("classified")):
            d = r.get("directive") or {}
            if d.get("text") == text:
                return {**d, "kind": r.get("kind")}
        return {}

    # -- scenario start --------------------------------------------------
    async def begin(self, name: str) -> None:
        """Load the scenario's house and assert its fixtures on world truth."""
        self.current = name
        b = self.binding(name)[1]
        house = b["house"]
        await self.load(BIND.scene(house), forget=bool(b.get("forget")))
        need = [x for x in [b.get("target"), b.get("object")] if x] + list(b.get("distractors") or [])
        need += list(b.get("second") or [])
        ok, notes = sc.check_fixtures(BIND, house, (self.init or {}).get("layout") or {},
                                      (self.frame or {}).get("truth") or {}, need)
        self.fixtures = {"fixtures_ok": ok, "fixture_notes": notes}
        if not ok:
            raise FixtureError("; ".join(notes))


class FixtureError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Scenarios. Each returns (passed, note).
# ----------------------------------------------------------------------
async def fetch(r: Run, oid: str, label: str, timeout: float | None = None, say: str | None = None) -> tuple[bool, str]:
    await r.say(say or f"Bring me the {label}.")
    ok = await r.until(lambda: r.where(oid) == r.user_surface and r.idle(), timeout or r.limit())
    return ok, f"{oid} ended on {r.where(oid)}"


async def fetch_other_room(r: Run) -> tuple[bool, str]:
    await r.begin("fetch_other_room")
    return await fetch(r, "alarm_clock_1", "alarm clock")


async def fetch_search(r: Run) -> tuple[bool, str]:
    """H15's apple is beyond a G1's reach (R.7), so scenes.yaml fetches the dish sponge; --original: the apple."""
    await r.begin("fetch_search")
    oid, label, say = BIND.fetch_target("fetch_search", r.original)
    return await fetch(r, oid, label, say=say)


async def correction(r: Run) -> tuple[bool, str]:
    await r.begin("correction")
    await r.say("Bring me a book.")
    await r.until(lambda: r.started("navigate"), r.wait(40))
    await r.say("No, bring me the alarm clock instead.")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), r.limit())
    books = [b for b in ("book_1", "book_2") if r.where(b) == r.user_surface]
    return ok and not books, f"alarm clock on {r.where('alarm_clock_1')}; books delivered: {books or 'none'}"


async def stop_resume(r: Run) -> tuple[bool, str]:
    await r.begin("stop_resume")
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), r.wait(40))
    await asyncio.sleep(2.0)
    await r.say("stop")
    stopped = await r.until(lambda: bool(r.rows("stop")), 10)
    await asyncio.sleep(5.0)
    await r.say("Okay, carry on.")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), r.limit())
    acks = sum(1 for _, t in r.said if "stopped" in t.lower())
    return ok and stopped and acks == 1, f"halted={stopped}, delivered={ok}, 'stopped' said {acks}x"


async def remember_where(r: Run) -> tuple[bool, str]:
    await r.begin("remember_where")                 # memory from the scenarios above
    t0 = r.now()
    await r.say("Where did you put the alarm clock last time?")
    ok = await r.until(lambda: r.said_since(t0, BIND.scenario("remember_where")["expect_said"]), r.limit())
    moved = r.started("navigate")
    return ok and not moved, f"answered from memory={ok}, walked first={moved}"


async def recall_history(r: Run) -> tuple[bool, str]:
    await r.begin("recall_history")
    t0 = r.now()
    await r.say("What did I ask you to bring me before?")
    ok = await r.until(lambda: r.said_since(t0, BIND.scenario("recall_history")["expect_said"]), r.limit())
    return ok, f"named an earlier request={ok}, recalls={len(r.rows('recall'))}"


async def question_midtask(r: Run) -> tuple[bool, str]:
    await r.begin("question_midtask")
    oid, _, say = BIND.fetch_target("question_midtask", r.original)
    await r.say(say)
    await r.until(lambda: r.started("navigate"), r.wait(40))
    t0 = r.now()
    await r.say("What are you holding right now?")
    answered = await r.until(lambda: any(st >= t0 for st, _ in r.said), 20)
    ok = await r.until(lambda: r.where(oid) == r.user_surface and r.idle(), r.limit())
    return ok and answered, f"answered={answered}, delivered={ok} ({oid} on {r.where(oid)})"


async def note_only(r: Run) -> tuple[bool, str]:
    await r.begin("note_only")
    await r.say("By the way, my keys are usually on the kitchen counter.")
    await asyncio.sleep(r.limit())
    noted = bool(r.rows("note_saved"))
    moved = r.started("navigate") or r.started("manipulate")
    return noted and not moved, f"noted={noted}, moved={moved}"


async def unsupported(r: Run) -> tuple[bool, str]:
    await r.begin("unsupported")
    t0 = r.now()
    await r.say("Put the apple in the microwave.")
    ok = await r.until(lambda: r.said_since(t0, r"can't|cannot|can not|unable|not able|don't have a way"), r.limit())
    return ok, f"said it can't={ok}"


async def missing_object(r: Run) -> tuple[bool, str]:
    await r.begin("missing_object")
    t0 = r.now()
    await r.say("Bring me the banana.")                  # there is no banana in this house
    told = await r.until(lambda: r.said_since(t0, r"can't find|couldn't find|could not find|no banana|"
                                                   r"not find|didn't find|don't see|haven't found|isn't here|not here"),
                         r.limit())
    await r.until(r.idle, r.wait(20))
    looks = len(r.rows("started", tool="navigate"))
    return told, f"told you it isn't here={told}, walked to {looks} spots, {r.now() - t0:.0f} s"


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
    """Every type in the house, from world truth (objects, and the house's full vocabulary sent in init)."""
    return sc.types_present((r.frame or {}).get("truth") or {}, (r.init or {}).get("layout") or {})


async def hold_on(r: Run) -> tuple[bool, str]:
    """A stop the keyword check misses: only System 1's label can stop the robot."""
    await r.begin("hold_on")
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), r.wait(40))
    await asyncio.sleep(2.0)
    t_say = time.monotonic()
    await r.say("hang on a sec")
    stopped = await r.until(lambda: bool(r.rows("stop", reason="classified")), 10)
    took = time.monotonic() - t_say
    lab = r.label("hang on a sec")
    await asyncio.sleep(3.0)
    await r.say("okay, go ahead")
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), r.limit())
    return (ok and stopped and lab.get("source") == "system1",
            f"stopped on the label={stopped} after {took:.1f}s (label {lab.get('kind')} from {lab.get('source')}, "
            f"P={lab.get('confidence')}), delivered={ok}")


async def replace_task(r: Run) -> tuple[bool, str]:
    await r.begin("replace_task")
    await r.say("Bring me a book.")
    await r.until(lambda: r.started("navigate"), r.wait(40))
    text = "Never mind the book, get me the alarm clock."
    await r.say(text)
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle(), r.limit())
    books = [b for b in ("book_1", "book_2") if r.where(b) == r.user_surface]
    lab = r.label(text)
    return (ok and not books,
            f"alarm clock on {r.where('alarm_clock_1')}; books delivered: {books or 'none'}; "
            f"label {lab.get('kind')} from {lab.get('source')}")


async def addition(r: Run) -> tuple[bool, str]:
    """Two deliveries: the second request arrives while the first runs. On H40 the THOR text asked for a book;
    both books are unreachable static prims there, so scenes.yaml substitutes a mug (see its note); no G1 stance
    reaches either mug (R.7), so its g1_alternative asks for the wine bottle."""
    await r.begin("addition")
    b = r.binding("addition")[1]
    then, second = b.get("then"), list(b.get("second") or [])
    await r.say("Bring me the alarm clock.")
    await r.until(lambda: r.started("navigate"), r.wait(40))
    await r.say(then)
    ok = await r.until(lambda: r.where("alarm_clock_1") == r.user_surface and r.idle()
                       and any(r.where(b) == r.user_surface for b in second), r.limit())
    got = [b for b in second if r.where(b) == r.user_surface]
    lab = r.label(then)
    return ok, (f"alarm clock on {r.where('alarm_clock_1')}, second ({'/'.join(second)}) delivered: {got or 'none'}; "
                f"said {then!r}; label {lab.get('kind')} from {lab.get('source')}")


async def observations(r: Run) -> tuple[bool, str]:
    """System 1 watches the camera during a fetch: it must look, and never report a thing the house lacks."""
    await r.begin("observations")
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
    await r.begin("procedural")
    n0 = max(r.calls or {0: None})
    ok, note = await fetch(r, "alarm_clock_1", "alarm clock")
    inputs = [str(c.get("input") or "") for n, c in r.calls.items() if n > n0 and c.get("via") == "model"]
    guided = sum("LEARNED FROM PAST TASKS" in i for i in inputs)
    return ok and guided > 0, f"{note}; planner calls with learned guidance: {guided}/{len(inputs)}"


async def permission_yes(r: Run) -> tuple[bool, str]:
    """In a house it hasn't mapped, the robot asks to look around; a yes (labelled by System 1) starts it."""
    await r.begin("permission_yes")
    t0 = r.now()
    await r.send(type="persona", level="optimize")
    asked = await r.until(lambda: any(st >= t0 and text.rstrip().endswith("?") for st, text in r.said), r.limit())
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
    """"The other side of the stove" needs to know what the stove sits between. In MolmoSpaces Kitchen 10 the
    spatula starts on counter_2b, the stove is next along the wall, then counter_2a (scenes.yaml has why this
    differs from THOR's counter_1a / counter_2). Both surfaces are also resolved from the live map and truth.
    The spatula is beyond a G1's reach (R.7); scenes.yaml's g1_alternative moves bowl_1 from counter_2c, one stretch
    further from the stove, to counter_2a. Either way the golden line names the stove's two neighbours."""
    base = BIND.scenario("other_side")
    b = r.binding("other_side")[1]
    await r.begin("other_side")
    n0 = max(r.calls or {0: None})
    lay = (r.init or {}).get("layout") or {}
    start = r.where(b["object"])
    lm = (lay.get("landmarks") or {}).get(b["landmark"]) or {}
    target = b["target_surface"]
    if start and lm.get("x") is not None:
        live = sc.other_side(lay, (float(lm["x"]), float(lm["z"])), start)
        if live and live != target:
            r.fixtures.setdefault("fixture_notes", []).append(f"live map puts the other side at {live}, scenes.yaml at {target}")
            target = live
    await r.say(b["say"])
    ok = await r.until(lambda: r.where(b["object"]) == target and r.idle(), r.limit())
    inputs = [str(c.get("input") or "") for n, c in r.calls.items() if n > n0 and c.get("via") == "model"]
    near = start if b["start_surface"] == base["start_surface"] else base["start_surface"]   # the stove's near neighbour
    golden = sc.golden_lines(b["landmark"], near or base["start_surface"], target)
    knew = any(g in i for i in inputs for g in golden)
    checks = [f"{c.get('goal')!r}: {c.get('ok')}" for c in r.rows("goal_check")]
    return ok and start == b["start_surface"], (f"{b['object']} from {start} to {r.where(b['object'])} (target {target}); "
                                                f"LAYOUT had {golden[0]!r}: {knew}; goal checks: {', '.join(checks) or 'none'}")


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


def score(name: str, passed: bool, note: str, seconds: float, run: Run) -> dict[str, Any]:
    """One result row: pass/fail, the binding, the time limit and how it was derived, the executors that
    produced the result, and the honesty label (a pass through a STEPPING STONE is a fallback pass)."""
    which, b = run.binding(name)
    limit, how = BIND.time_limit(name, run.profile, run.scale, b)
    used = run.executors()
    hon = sc.honesty(used, grasp=bool(b.get("grasp")), profile=run.profile)
    return {"name": name, "passed": passed, "seconds": round(seconds, 1), "note": note,
            "profile": run.profile, "house": b["house"], "scene": BIND.scene(b["house"]),
            "binding": which, "binding_status": b.get("status", "ok"), "binding_flags": list(b.get("flags") or []),
            "time_limit_s": limit, "time_limit": how,
            "executors_used": used, "shortcuts": hon["shortcuts"], "honesty": hon["labels"],
            "fallback_pass": bool(passed and hon["fallback"]), "target_pass": bool(passed and not hon["fallback"]),
            **run.fixtures, **run.metrics()}


def summarize(results: list[dict[str, Any]], profile: str, tag: str) -> dict[str, Any]:
    by_exec: dict[str, dict[str, int]] = {}
    for r in results:
        for group in ("nav", "manip"):
            for ex in (r["executors_used"].get(group) or {}):
                d = by_exec.setdefault(ex, {"scenarios": 0, "passed": 0})
                d["scenarios"] += 1
                d["passed"] += int(r["passed"])
    cost_keys = ("model_calls", "model_errors", "classify_calls", "next_action_calls", "rule_classifies",
                 "tokens_out", "s1_route_calls", "s1_observe_calls")
    return {"tag": tag, "profile": profile, "wall": round(time.time()),
            "passed": sum(r["passed"] for r in results), "total": len(results),
            "target_passes": sum(r["target_pass"] for r in results),
            "fallback_passes": sum(r["fallback_pass"] for r in results),
            "by_executor": by_exec,
            "decisions": sum(r["decisions"] for r in results),
            "tokens_in": sum(r["tokens_in"] for r in results),
            "costs": {k: sum(int(r.get(k) or 0) for r in results) for k in cost_keys}, "results": results}


def line(res: dict[str, Any]) -> str:
    verdict = "PASS" if res["passed"] else "FAIL"
    if res["fallback_pass"]:
        verdict = "PASS*"
    lab = f"  [{'; '.join(res['honesty'])}]" if res["honesty"] else ""
    return (f"{verdict:<5} {res['name']:<18} {res['seconds']:>6.1f}s/{res['time_limit_s']:<6.0f} decisions "
            f"{res['decisions']:>3}  rejected {res['rejected']}  recalls {res['recalls']}  tokens {res['tokens_in']:>6}  "
            f"calls {res.get('model_calls', 0)}+{res.get('s1_route_calls', 0)}+{res.get('s1_observe_calls', 0)}  "
            f"labels {res['s1_labels']}/{res['labels']} by System 1  {res['note']}{lab}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--profile", default="lite", choices=sorted(BIND.data["profiles"]))
    ap.add_argument("--time-scale", type=float, default=None, help="override the profile's time scale")
    ap.add_argument("--original", action="store_true", help="run THOR's text where scenes.yaml substituted one")
    ap.add_argument("--g1-alternative", choices=("auto", "on", "off"), default="auto",
                    help="scenes.yaml's g1_alternative bindings (addition, other_side): auto = on the G1 profiles only")
    ap.add_argument("--out", default=None, help="summary JSON path (default: runs/eval/<stamp>_<profile>_<tag>.json)")
    ap.add_argument("--trace-dir", default=None, help="write each scenario's trace rows, model calls and speech here")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    from websockets.asyncio.client import connect
    results = []
    async with connect(args.url, max_size=2 ** 24) as ws:
        run = Run(ws, args.profile, args.time_scale, args.original,
                  {"auto": None, "on": True, "off": False}[args.g1_alternative])
        reader = asyncio.create_task(run.reader())
        for name, fn in SCENARIOS:
            if only and name not in only:
                continue
            t0 = time.monotonic()
            run.fixtures, run.frames_seen = {}, []
            try:
                passed, note = await fn(run)
            except FixtureError as e:
                passed, note = False, f"fixture: {e}"
            except Exception as e:  # noqa: BLE001  (a broken scenario fails, the suite goes on)
                passed, note = False, f"error: {e!r}"
            res = score(name, passed, note, time.monotonic() - t0, run)
            if args.trace_dir:
                res["trace_path"] = str(run.dump(Path(args.trace_dir) / f"{args.tag}_{name}.jsonl"))
            results.append(res)
            print(line(res), flush=True)
        reader.cancel()
    summary = summarize(results, args.profile, args.tag)
    path = Path(args.out) if args.out else OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.profile}_{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    c = summary["costs"]
    print(f"\n{summary['passed']}/{summary['total']} passed on {args.profile} "
          f"({summary['target_passes']} target, {summary['fallback_passes']} fallback: PASS* rests on a sim shortcut) · "
          f"{summary['decisions']} decisions · {summary['tokens_in']} tokens in · {path}")
    print(f"  model calls: planner {c['model_calls']} ({c['classify_calls']} classify, {c['next_action_calls']} next "
          f"action, {c['model_errors']} errors; {c['tokens_out']} tokens out), {c['rule_classifies']} classified by "
          f"rule; System 1: {c['s1_route_calls']} label calls, {c['s1_observe_calls']} observe calls")
    for ex, d in sorted(summary["by_executor"].items()):
        tag = " (STEPPING STONE)" if ex in sc.stepping_stones() else ""
        print(f"  {ex}{tag}: {d['passed']}/{d['scenarios']} scenarios passed")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
