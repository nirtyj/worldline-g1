"""The persona: the robot's own standing goals, for when nobody needs anything.

Gemini (System 1) hears, Claude (System 2) plans each step. The persona is a
third, slower part with goals of its own: know the home, keep that knowledge
fresh, be close to the user. When the robot has been idle for a while it picks
the most useful of those and hands it to System 2 as an OWN GOAL ("look at
counter_1b: you've never checked it"). System 2 still chooses the steps and the
runtime still checks every one; the persona only decides what is worth doing,
and each goal says which tools it may use.

It never competes with the user. It starts only when nothing is running and
nobody has spoken for a while, the runtime drops its goal the moment anyone
speaks, and an own goal can never pick up or put down objects.

Levels (how pushy it is):
  off       nothing
  quiet     never moves or speaks (notes of what the user says are kept at every level)
  medium    + turns its head: looks around from where it stands when that spot
            hasn't been checked recently
  optimize  + asks once ("this looks like a kitchen, can I look around?"); if the
            user agrees it drives to every spot it hasn't seen, re-checks old
            looks, then goes back to the user

Drives (the highest score wins; nearer spots score higher):
  ask       optimize, before its first move this session
  map       a spot never looked at          -> go and look there
  refresh   a spot not looked at for a while -> look again, things may have moved
  ready     away from the user when idle     -> go back to them
  glance    medium: look around from here
"""

from __future__ import annotations

import collections
import itertools
import re
from dataclasses import asdict, dataclass
from typing import Any

LEVELS = ("off", "quiet", "medium", "optimize")
IDLE_S = 8.0            # quiet this long before the persona starts a goal
CHAIN_S = 1.5           # ...or this long right after finishing one
REFRESH_S = 300.0       # a look older than this is worth repeating
GOAL_TIMEOUT_S = 75.0   # give up on a goal after this long
STUCK_S = 10.0          # ...or when nothing has happened for this long
RETRY_AFTER_S = 300.0   # don't retry a target that failed for this long
ANSWER_WAIT_S = 30.0    # no reply to "can I look around?" this long means no

YES = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|go ahead|go for it|please do|of course|fine|do it)\b", re.I)
NO = re.compile(r"\b(no|nope|not now|don'?t|stay|later)\b", re.I)


@dataclass
class OwnGoal:
    id: str
    drive: str                 # ask, map, refresh, ready, glance
    target: str                # a keypoint
    text: str                  # what System 2 reads
    tools: tuple[str, ...]     # what it may use for this goal
    t_start: float
    said: bool = False         # one short line per goal, at most

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Persona:
    name = "robot"

    def __init__(self, level: str = "medium") -> None:
        self.level = level if level in LEVELS else "medium"
        self.goal: OwnGoal | None = None
        self.last: list[tuple[float, str, str]] = []        # (score, drive, target), for the page
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=12)
        self._ids = itertools.count(1)
        self._failed: dict[str, float] = {}
        self.finished_t = -1e9
        self.permission = "unasked"        # optimize: unasked, waiting, granted, declined
        self._asked_t = -1e9

    @property
    def acts(self) -> bool:
        return self.level in ("medium", "optimize")

    # ------------------------------------------------------------------
    def heard(self, text: str, now: float, reply: str | None = None) -> bool:
        """The user said something. If it answers "can I look around?", note the answer:
        Gemini's yes/no when it gave one, else the words. True when they just said yes."""
        if self.permission != "waiting":
            return False
        if reply in ("yes", "no"):
            self.permission = "granted" if reply == "yes" else "declined"
        elif NO.search(text) and not YES.search(text):
            self.permission = "declined"
        elif YES.search(text):
            self.permission = "granted"
        else:
            return False
        self.history.append({"id": "-", "drive": "answer", "target": self.permission,
                             "outcome": "yes" if self.permission == "granted" else "no",
                             "t": round(now, 1), "took_s": 0.0, "by": "classifier" if reply else "words"})
        return self.permission == "granted"

    def tick(self, now: float) -> None:
        if self.permission == "waiting" and now - self._asked_t > ANSWER_WAIT_S:
            self.permission = "declined"          # silence is a no

    def drives(self, belief: Any, map_: dict[str, Any], now: float) -> list[tuple[float, str, str, str, tuple]]:
        """Every goal worth doing now at this level: (score, drive, target, text, tools), best first."""
        out: list[tuple[float, str, str, str, tuple]] = []
        here = belief.robot_at.value
        spots = list(map_["surfaces"])
        looked = belief.looked
        roam = self.level == "optimize" and self.permission == "granted"
        if self.level == "optimize" and self.permission == "unasked":
            unseen = [s for s in spots if s not in looked]
            if len(unseen) >= max(1, len(spots) // 2):
                out.append((4.0, "ask", here or "start",
                            "You've just arrived somewhere you haven't mapped. In one short sentence, say "
                            "what kind of room this looks like (from the landmarks and objects in BELIEF) "
                            "and ask the user if you may take a quick look around. Then wait for their answer.",
                            ("say",)))
            else:
                self.permission = "granted"       # mostly mapped already: nothing to ask
                roam = True
        if roam:
            dist = _distances(map_, here)
            for spot in spots:
                if now - self._failed.get(spot, -1e9) < RETRY_AFTER_S:
                    continue
                d = dist.get(spot, 5.0)
                seen = looked.get(spot)
                if seen is None:
                    out.append((3.0 - 0.1 * d, "map", spot,
                                f"Look at {spot}: you have never looked there. Go there and look, "
                                f"so you know what's on it and around it.", ("navigate", "look", "say")))
                elif now - seen.t > REFRESH_S:
                    age = now - seen.t
                    out.append((1.5 + min(age / 1200, 1.0) - 0.05 * d, "refresh", spot,
                                f"Look at {spot} again: your last look there was {_ago(age)} ago "
                                f"and things may have moved.", ("navigate", "look", "say")))
            user = (map_.get("people") or {}).get("user", {}).get("keypoint")
            if user and here != user and now - self._failed.get(user, -1e9) >= RETRY_AFTER_S:
                out.append((1.0, "ready", user,
                            f"Go back to {user}, next to the user, so you're close by for their next request.",
                            ("navigate", "say")))
        if self.acts and here is not None and not roam and self.permission != "waiting":
            seen = looked.get(here)
            if (seen is None or now - seen.t > REFRESH_S) and now - self._failed.get(here, -1e9) >= RETRY_AFTER_S:
                out.append((2.0, "glance", here,
                            f"Look around from where you are ({here}) without moving, so you know "
                            f"what's here. Don't navigate.", ("look",)))
        out.sort(key=lambda g: -g[0])
        self.last = [(round(s, 2), dr, tg) for s, dr, tg, _, _ in out[:6]]
        return out

    def propose(self, belief: Any, map_: dict[str, Any], now: float, quiet_s: float) -> OwnGoal | None:
        self.tick(now)
        if not self.acts or self.goal is not None:
            self.last = [] if not self.acts else self.last
            return None
        needed = CHAIN_S if now - self.finished_t < 30 else IDLE_S
        options = self.drives(belief, map_, now)
        if quiet_s < needed or not options:
            return None
        _, drive, target, text, tools = options[0]
        self.goal = OwnGoal(f"g{next(self._ids)}", drive, target, text, tools, round(now, 2))
        return self.goal

    def is_done(self, belief: Any) -> bool:
        g = self.goal
        if g is None:
            return False
        if g.drive == "ask":
            return g.said
        if g.drive == "ready":
            return belief.robot_at.value == g.target
        looked = belief.looked.get(g.target)
        return looked is not None and looked.source != "memory" and looked.t >= g.t_start

    def finish(self, outcome: str, now: float) -> OwnGoal | None:
        """outcome: done, gave_up, dropped (the user spoke) or off."""
        g, self.goal = self.goal, None
        if g is None:
            return None
        if g.drive == "ask":
            if outcome == "done":
                self.permission, self._asked_t = "waiting", now
            else:
                self.permission = "declined" if outcome == "gave_up" else self.permission
        if outcome == "gave_up":
            self._failed[g.target] = now
        self.finished_t = now if outcome == "done" and g.drive != "ask" else -1e9
        self.history.append({"id": g.id, "drive": g.drive, "target": g.target, "outcome": outcome,
                             "t": round(now, 1), "took_s": round(now - g.t_start, 1)})
        return g

    def snapshot(self, now: float) -> dict[str, Any]:
        g = self.goal
        return {
            "name": self.name, "level": self.level, "permission": self.permission,
            "enabled": self.level != "off",
            "goal": None if g is None else {**g.to_dict(), "age_s": round(now - g.t_start, 1)},
            "drives": [{"score": s, "drive": d, "target": t} for s, d, t in self.last],
            "history": list(self.history)[::-1],
        }


def _distances(map_: dict[str, Any], here: str | None) -> dict[str, float]:
    if here is None:
        return {}
    out = {here: 0.0}
    for a, b, d in map_.get("edges", []):
        if a == here:
            out[b] = d
        elif b == here:
            out[a] = d
    return out


def _ago(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.0f} h"
