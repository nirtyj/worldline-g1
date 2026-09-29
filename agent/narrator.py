"""The robot's progress, in plain words, for the chat: "Heading to the dresser in
the bedroom, where memory says the alarm clock usually is", "Passing the kitchen
counter", "Found the alarm clock on the dresser".

Built only from what the runtime already knows: its trace rows (what was asked,
what it started, what came back, what memory said) and its own pose on its map.
No model call, and nothing here reaches the planner's prompt or the speech queue:
these lines are a status display, not the robot talking.

    narrator = Narrator(map_, belief, deliver_to)
    tracer.sinks.append(narrator.row)          # it reads every trace row
    narrator.pose(xy, moving, now)             # 10 Hz, from the observation loop
    for kind, text in narrator.take():         # then log them as "narrate" rows
        tracer.log("narrate", kind=kind, text=text)
"""

from __future__ import annotations

import math
import re
from typing import Any

PASS_M = 0.9          # within this of a spot while driving: "passing the ..."
ARRIVING_M = 1.6      # this close to where it's going, it's arriving, not passing
QUIET_S = 1.5         # at most one "passing" line this often


def _nice(name: str) -> str:
    return name.replace("_", " ")


class Narrator:
    def __init__(self, map_: dict[str, Any], belief: Any, deliver_to: str | None) -> None:
        self.map, self.belief, self.deliver_to = map_, belief, deliver_to
        self.kps = map_.get("keypoints") or {}
        self.rooms = {k: v.get("label", _nice(k)) for k, v in (map_.get("rooms") or {}).items()}
        self.target: str | None = None           # the thing the current request is about
        self.wanted: str | None = None           # its name, even if it has never been seen ("banana")
        self.heading: str | None = None
        self.at: str | None = "start"            # the last spot it arrived at
        self.passed: set[str] = set()
        self.room: str | None = None
        self._last_pass_t = -1e9
        self._out: list[tuple[str, str]] = []
        self._heard: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def _spot(self, name: str | None) -> str:
        """kitchen_dining_table_1b -> "the dining table in the kitchen"."""
        if not name:
            return "somewhere"
        if name == "start":
            return "where I started"
        kp = self.kps.get(name) or {}
        room = kp.get("room")
        label = _nice(name[len(room) + 1:] if room and name.startswith(room + "_") else name)
        label = re.sub(r"\s\d+[a-z]?$", "", label)          # "dining table 1b" -> "dining table"
        if name == self.deliver_to:
            return "your spot"
        return f"the {label}" + (f" in the {self.rooms.get(room, _nice(room))}" if room else "")

    def _kind(self, oid: str | None) -> str:
        """alarm_clock_1 -> "alarm clock"."""
        ob = self.belief.objects.get(oid) if oid else None
        return _nice(ob.type) if ob is not None else _nice(re.sub(r"_[0-9]+$", "", oid or "it"))

    def _thing(self, oid: str | None) -> str:
        return "the " + self._kind(oid)

    def _say(self, kind: str, text: str) -> None:
        if self._out and self._out[-1][1] == text:
            return
        self._out.append((kind, text))

    def _goal_words(self, goal: Any) -> str:
        """ "other side of stove_1 from counter_1a" -> "on the other side of the stove"."""
        g = re.sub(r"\s+from\s+\S+$", "", str(goal or ""))
        def word(m: re.Match) -> str:                   # spots keep their number: "counter 1a"
            return _nice(m.group(0)) if m.group(0) in self.kps else "the " + _nice(m.group(1))
        g = re.sub(r"\b([a-z]+(?:_[a-z]+)*)_\d+[a-z]?\b", word, g)
        return ("on the " + g) if g.startswith("other side") else g

    def _known(self, oid: str | None) -> bool:
        """Does belief say where this is (a spot or a hand), so there's nothing to look for?"""
        ob = self.belief.objects.get(oid) if oid else None
        return ob is not None and ob.where.value not in (None, "UNKNOWN")

    def _guess_target(self, text: str) -> str | None:
        """Which known kind of thing a request mentions ("the alarm clock" -> alarm_clock_1)."""
        words = text.lower()
        best = None
        for oid, ob in self.belief.objects.items():
            kind = _nice(ob.type).lower()
            if kind and re.search(rf"\b{re.escape(kind)}s?\b", words) and (best is None or len(kind) > best[0]):
                best = (len(kind), oid)
        return best[1] if best else None

    # ------------------------------------------------------------------ trace rows in
    def row(self, r: dict[str, Any]) -> None:
        """A trace sink: it runs inside tracer.log, so it must never raise into the runtime."""
        try:
            self._row(r)
        except Exception as e:                        # a status line is never worth a crash
            self._say("problem", f"(narrator error: {type(e).__name__})")

    def _row(self, r: dict[str, Any]) -> None:
        t = r.get("type")
        a = r.get("args") or {}
        if t == "heard":
            self._heard[r.get("id")] = r.get("text", "")
        elif t == "classified":
            kind, text = r.get("kind"), self._heard.get(r.get("id"), "")
            if kind in ("request", "correction"):
                named = ((r.get("directive") or {}).get("target") or {}).get("object")   # System 1's target
                self.target = self._guess_target(named or text)
                m = re.search(r"\b(?:bring|get|fetch|find|grab)\b(?: me| us)?(?: the| a| an| my| some)? ([a-z][a-z ]*?)(?: instead| please| to| from| on|[.?!,]|$)", text.lower())
                self.wanted = self._kind(self.target) if self.target else (named or (m.group(1).strip() if m else None))
                what = f"looking for the {self.wanted}" if self.wanted and not self._known(self.target) else f"“{text.strip()}”"
                self._say("task", ("Changing plans: " if kind == "correction" else "Working on it: ") + what)
            elif kind == "question":
                self._say("think", "Working out an answer")
            elif kind == "observation":
                self._say("memory", "Noting that down")
        elif t == "stop":
            self._say("stop", "Stopped. Waiting for you")
        elif t == "recall":
            answer = str(r.get("answer", ""))
            usual = re.search(r"usually (\w+)", answer)
            if usual and usual.group(1) in self.kps:
                found = f"memory says it's usually on {self._spot(usual.group(1))}"
            elif answer.startswith("Nothing in memory"):
                found = "nothing in memory about it"
            else:
                found = _nice(answer.split("\n")[0])
                found = found if len(found) < 100 else found[:97] + "…"
            self._say("memory", f"Checking memory for “{r.get('query')}”: {found}")
        elif t == "started":
            tool = r.get("tool")
            if tool == "navigate":
                to = a.get("to")
                self.heading, self.passed = to, {self._spot(to), self._spot(self.at)}   # names already said
                why = ""
                ob = self.belief.objects.get(self.target) if self.target else None
                if to == self.deliver_to and self.target and self._held(self.target):
                    why = f", to bring you {self._thing(self.target)}"
                elif self.target and self._held(self.target):          # carrying it somewhere else
                    why = f", with {self._thing(self.target)}"
                elif ob is not None and ob.where.value == to:
                    why = f", where {'memory says' if ob.where.source == 'memory' else 'I saw'} {self._thing(self.target)} is"
                elif ob is not None and ob.usual == to:
                    why = f", where {self._thing(self.target)} usually is"
                elif self.wanted:
                    why = f", to look for the {self.wanted}"
                self._say("move", f"Heading to {self._spot(to)}{why}")
            elif tool == "look":
                self._say("look", "Looking around" + (f" for the {self.wanted}" if self.wanted else ""))
            elif tool == "pick":
                self._say("arm", f"Picking up {self._thing(a.get('object'))}")
            elif tool == "place":
                self._say("arm", f"Putting {self._thing(a.get('object'))} down")
            elif tool == "reachability":
                self._say("arm", f"Checking I can reach {self._thing(a.get('object'))}")
        elif t == "result":
            skill, status, d = r.get("skill"), r.get("status"), r.get("data") or {}
            if skill == "navigate" and status == "SUCCEEDED":
                self.heading, self.at = None, d.get("at")
                self._say("move", f"At {self._spot(d.get('at'))}")
            elif skill == "navigate" and status in ("ABORTED", "CANCELED"):
                self.heading = None
                if d.get("reason") != "halted":                  # a stop was already reported
                    self._say("move", f"Stopped on the way ({_nice(str(d.get('reason') or status.lower()))})")
            elif skill == "look" and r.get("source") != "harness":
                saw = [s for s in d.get("saw") or [] if s != self.target]
                if saw:
                    names = sorted({self._kind(s) for s in saw})
                    self._say("look", "I see here: " + (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]))
            elif skill in ("pick", "place") and status != "SUCCEEDED":
                self._say("problem", f"The {skill} didn't work ({_nice(str(d.get('reason') or status.lower()))})")
            elif skill == "reachability" and not d.get("reachable"):
                self._say("problem", f"Can't reach it from here ({_nice(str(d.get('reason') or 'no'))})")
        elif t == "place_learned":
            oid, place, was = r.get("object"), str(r.get("place") or ""), str(r.get("was") or "")
            if oid == self.target:
                if place == "UNKNOWN":
                    self._say("problem", f"{self._thing(oid).capitalize()} isn't where I expected")
                elif place.startswith("hand"):
                    self._say("arm", f"Got {self._thing(oid)}")
                elif place == self.deliver_to:
                    pass                                   # "delivered" says it
                elif was.startswith("hand"):               # it put it there itself: not a find
                    self._say("done", f"{self._thing(oid).capitalize()} is on {self._spot(place)} now")
                else:
                    self._say("found", f"Found {self._thing(oid)} on {self._spot(place)}")
        elif t == "goal_check":
            thing, where = self._thing(r.get("object")), self._spot(r.get("where"))
            if r.get("ok"):
                self._say("done", f"Checked: {thing} is {self._goal_words(r.get('goal'))}")
            elif r.get("ok") is False:
                self._say("problem", f"Not right yet: {thing} is on {where}, not {self._goal_words(r.get('goal'))}")
            else:
                self._say("problem", f"Can't check “{r.get('goal')}”: {r.get('why')}")
        elif t == "delivered":
            self._say("done", f"Delivered {self._thing(r.get('object'))} to you")
            self.target = self.wanted = None
        elif t == "rejected":
            self._say("problem", f"Rethinking: {r.get('tool')} wasn't allowed ({r.get('why')})")
        elif t == "observation":
            self._say("look", f"Noticed: {r.get('text')}")
        elif t == "persona_goal":
            drive, spot = r.get("drive"), r.get("target")
            own = {"glance": "having a look around from here",
                   "ask": "asking if I may look around",
                   "map": f"going to see what's on {self._spot(spot)}",
                   "refresh": f"checking {self._spot(spot)} again, it's been a while",
                   "ready": "going back to wait by you"}.get(drive, _nice(str(drive)))
            self._say("own", f"Nothing to do, so on my own: {own}")
        elif t == "step_waiting":
            self._say("think", "Waiting for Step before the next decision")

    def _held(self, oid: str) -> bool:
        ob = self.belief.objects.get(oid)
        return ob is not None and str(ob.where.value).startswith("hand")

    # ------------------------------------------------------------------ pose in
    def pose(self, xy: Any, moving: bool, now: float) -> None:
        """While driving: the spots it passes, and each room it enters."""
        try:
            self._pose(xy, moving, now)
        except Exception as e:
            self._say("problem", f"(narrator error: {type(e).__name__})")

    def _pose(self, xy: Any, moving: bool, now: float) -> None:
        if not moving or not xy or not self.kps:
            return
        x, z = float(xy[0]), float(xy[1])
        near = min(self.kps.items(), key=lambda kv: math.dist((x, z), kv[1]["xy"]))
        name, kp = near
        room = kp.get("room")
        if room and room != self.room:
            if self.room is not None:
                self._say("move", f"Entering the {self.rooms.get(room, _nice(room))}")
            self.room = room
        d = math.dist((x, z), kp["xy"])
        goal = self.kps.get(self.heading or "")
        arriving = goal is not None and math.dist((x, z), goal["xy"]) < ARRIVING_M
        label = self._spot(name)
        if (d < PASS_M and name != "start" and label not in self.passed and not arriving
                and now - self._last_pass_t > QUIET_S):
            self.passed.add(label)                       # one line per name: bed 1a and 1b are one bed
            self._last_pass_t = now
            self._say("move", f"Passing {label}")

    def take(self) -> list[tuple[str, str]]:
        out, self._out = self._out, []
        return out
