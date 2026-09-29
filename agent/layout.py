"""Where things are relative to each other, worked out from the robot's own map.

The planner is good with names and bad with geometry: given coordinates it guesses,
given nothing it guesses too ("the other side of the stove" went to the wrong
counter). So the relations are computed here, in code, and handed over as a few
lines of text, the way 3D scene graphs are given to LLM planners (SayPlan,
ConceptGraphs): what is next to what, what a landmark sits between, which lines of
counters face each other.

Built only from what the robot has: its map service (where it stands for each
surface, and each surface's centre) and the landmarks it has seen (the camera or a
look gives their position). Nothing from the simulator's house file.

    lines = build(map_, landmarks)       # landmarks: {id: {"label", "near", "pos": [x, z]}}
    render(lines, map_)                  # the LAYOUT section of the prompt
    about(query, lines, map_)            # what recall says about one spot or landmark

A line is surfaces faced the same way from where the robot stands, sitting along
one axis (a run of counters on one wall), with the landmarks that sit on that axis
between or beside them, ordered left to right as you face them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

AXIS_TOL_DEG = 30.0      # a surface faced within this of an axis is faced along that axis
LINE_M = 0.4             # surfaces this close in depth (along the facing) are on one line
LANDMARK_LINE_M = 0.6    # landmarks are bulkier: a fridge's centre sits further back
GAP_M = 1.8              # neighbours further apart than this start a new line
NEAR_M = 1.2             # "near" something: this close, or its neighbour on the line
LAYOUT_WORDS = {"layout", "side", "sides", "between", "next", "beside", "left", "right", "opposite", "across",
                "near", "nearby", "neighbour", "neighbor", "relative", "where", "position", "positions", "order"}


@dataclass
class Thing:
    name: str
    kind: str                     # "surface" or "landmark"
    label: str
    x: float
    z: float
    room: str | None


@dataclass
class Line:
    facing: tuple[int, int]       # unit axis from where you stand toward the surfaces
    room: str | None
    things: list[Thing] = field(default_factory=list)    # left to right as you face them

    def along(self, t: Thing) -> float:
        rx, rz = self.facing[1], -self.facing[0]          # your right hand, facing the line
        return t.x * rx + t.z * rz

    def depth(self, t: Thing) -> float:
        return t.x * self.facing[0] + t.z * self.facing[1]

    def surfaces(self) -> list[Thing]:
        return [t for t in self.things if t.kind == "surface"]


def _axis(dx: float, dz: float) -> tuple[int, int] | None:
    """The axis you face when looking along (dx, dz), if it is close to one."""
    if math.hypot(dx, dz) < 0.05:
        return None
    yaw = math.degrees(math.atan2(dx, dz)) % 360
    axis = round(yaw / 90) % 4 * 90
    if abs((yaw - axis + 180) % 360 - 180) > AXIS_TOL_DEG:
        return None
    return round(math.sin(math.radians(axis))), round(math.cos(math.radians(axis)))


def build(map_: dict[str, Any], landmarks: dict[str, dict[str, Any]] | None) -> list[Line]:
    kps = map_.get("keypoints") or {}
    groups: dict[tuple[tuple[int, int], str | None], list[Thing]] = {}
    for name, s in (map_.get("surfaces") or {}).items():
        center, stand = s.get("xy"), (kps.get((s.get("keypoints") or [name])[0]) or {}).get("xy")
        if not center or not stand:
            continue
        facing = _axis(center[0] - stand[0], center[1] - stand[1])
        if facing is None:
            continue
        room = (kps.get(name) or {}).get("room")
        label = str(s.get("desc") or name).removeprefix("the ")
        groups.setdefault((facing, room), []).append(Thing(name, "surface", label, center[0], center[1], room))

    lines: list[Line] = []
    for (facing, room), things in groups.items():
        probe = Line(facing, room)
        things.sort(key=probe.depth)
        rows: list[list[Thing]] = []
        for t in things:                                  # split by depth: two walls facing the same way
            if rows and abs(probe.depth(t) - probe.depth(rows[-1][0])) <= LINE_M:
                rows[-1].append(t)
            else:
                rows.append([t])
        for row in rows:
            row.sort(key=probe.along)
            cur = Line(facing, room, [row[0]])
            for t in row[1:]:                             # split by gaps: far apart isn't "next to"
                if probe.along(t) - probe.along(cur.things[-1]) > GAP_M:
                    lines.append(cur)
                    cur = Line(facing, room)
                cur.things.append(t)
            lines.append(cur)

    for lid, lm in sorted((landmarks or {}).items()):
        pos = lm.get("pos")
        if not pos or len(pos) < 2 or None in tuple(pos)[:2]:
            continue
        room = (kps.get(str(lm.get("near"))) or {}).get("room")
        thing = Thing(lid, "landmark", str(lm.get("label") or lid), float(pos[0]), float(pos[1]), room)
        best, best_d = None, LANDMARK_LINE_M
        for ln in lines:
            if ln.room != room:
                continue
            d = abs(ln.depth(thing) - sum(ln.depth(s) for s in ln.surfaces()) / len(ln.surfaces()))
            a = [ln.along(s) for s in ln.surfaces()]
            if d <= best_d and min(a) - GAP_M <= ln.along(thing) <= max(a) + GAP_M:
                best, best_d = ln, d
        if best is not None:
            best.things.append(thing)
            best.things.sort(key=best.along)
    return [ln for ln in lines if len(ln.things) > 1]


def _name(t: Thing) -> str:
    return f"{t.name} ({t.label})" if t.kind == "landmark" else t.name


def _neighbours(ln: Line, t: Thing) -> tuple[Thing | None, Thing | None]:
    """The nearest surface on its left and on its right in its line."""
    i = ln.things.index(t)
    left = next((s for s in reversed(ln.things[:i]) if s.kind == "surface"), None)
    right = next((s for s in ln.things[i + 1:] if s.kind == "surface"), None)
    return left, right


def _between(ln: Line, t: Thing) -> str:
    left, right = _neighbours(ln, t)
    if left and right:
        return f"{_name(t)} is between {left.name} and {right.name}"
    if left or right:
        s = left or right
        return f"{_name(t)} is at the {'right' if left else 'left'} end, next to {s.name}"
    return ""


def _facing_each_other(lines: list[Line]) -> list[tuple[int, int]]:
    out = []
    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if a.room != b.room or a.facing != (-b.facing[0], -b.facing[1]):
                continue
            aa = [a.along(t) for t in a.things]
            bb = [a.along(t) for t in b.things]            # both measured along a's axis
            if min(aa) <= max(bb) and min(bb) <= max(aa):
                out.append((i, j))
    return out


def render(lines: list[Line], map_: dict[str, Any]) -> str:
    """The LAYOUT section: each line left to right, what each landmark sits between, and which face each other."""
    if not lines:
        return ""
    rooms = {k: v.get("label", k) for k, v in (map_.get("rooms") or {}).items()}
    out = []
    for i, ln in enumerate(lines):
        where = f"{rooms.get(ln.room, ln.room)}, " if ln.room else ""
        out.append(f"  {where}line {i + 1}: " + " · ".join(_name(t) for t in ln.things))
    for i, j in _facing_each_other(lines):
        out.append(f"  line {i + 1} and line {j + 1} face each other")
    for ln in lines:
        out += [f"  {b}" for b in (_between(ln, t) for t in ln.things if t.kind == "landmark") if b]
    return "\n".join(out)


def about(query: str, lines: list[Line], map_: dict[str, Any]) -> list[str]:
    """What recall says about layout, expanding only what was asked (as SayPlan does): which
    spots satisfy a relation ("other side of the stove", "left of the toaster"); the line,
    neighbours and nearest drives of each spot or landmark named; the whole layout when the
    query is about the layout itself."""
    words = set(re.findall(r"[a-z0-9_]+", query.lower()))
    out: list[str] = []
    goal = parse(query, lines, map_)
    if goal is not None and goal.rel != "on":
        out.append(_answer(goal, lines, map_))
    for ln in lines:
        order = " · ".join(_name(t) for t in ln.things)
        for t in ln.things:
            label = t.label.lower()
            if t.name in words or (t.kind == "landmark" and re.search(rf"\b{re.escape(label)}s?\b", query.lower())):
                if t.kind == "landmark":
                    out.append(_between(ln, t) + f" (left to right as you face it: {order})")
                else:
                    near = _drives(t.name, map_)
                    out.append(f"{t.name}: {_sides(ln, t) or 'alone on its line'} (left to right as you face it: {order})"
                               + (f"; nearest drives: {near}" if near else ""))
    if words & LAYOUT_WORDS and lines and goal is None:
        out.insert(0, "Layout, each line left to right as you face it:\n" + render(lines, map_))
    return out


def _drives(spot: str, map_: dict[str, Any], n: int = 3) -> str:
    """The closest spots by driving distance, from the map service's path lengths."""
    d = [(m, b if a == spot else a) for a, b, m in map_.get("edges") or [] if spot in (a, b)]
    return ", ".join(f"{k} {m:.1f} m" for m, k in sorted(d)[:n])


def _sides(ln: Line, t: Thing) -> str:
    """What is on each side of a surface: the next thing, and past a landmark, the next surface."""
    i, parts = ln.things.index(t), []
    for side, step in (("left", -1), ("right", 1)):
        j, seq = i + step, []
        while 0 <= j < len(ln.things):
            seq.append(_name(ln.things[j]))
            if ln.things[j].kind == "surface":
                break
            j += step
        if seq:
            parts.append(f"{', then '.join(seq)} on its {side}")
    return "; ".join(parts)


# ----------------------------------------------------------------------
# Relations: "the other side of the stove", worked out on the layout.
# The planner passes one with a place (goal=...); the runtime checks where the
# object really ended up (the VeriGraph idea: check the result against the
# relation asked for, not against the planner's say-so). Recall answers them too.
# ----------------------------------------------------------------------
GOAL_FORMS = ("on X", "next to X", "left of X", "right of X", "between X and Y",
              "other side of X from Y", "across from X", "near X", "in X")
_PATTERNS = (
    ("between", r"between (?:the )?(?P<a>.+?) and (?:the )?(?P<b>.+)"),
    ("other_side", r"(?:on |to )?(?:the )?(?:other|opposite|far) side of (?:the )?(?P<a>.+?)(?: from (?:the )?(?P<b>.+))?"),
    ("across", r"(?:on the )?(?:\w+ )?across (?:from )?(?:the )?(?P<a>.+)"),
    ("left_of", r"(?:on |to )?(?:the )?left (?:of|side of) (?:the )?(?P<a>.+)"),
    ("right_of", r"(?:on |to )?(?:the )?right (?:of|side of) (?:the )?(?P<a>.+)"),
    ("next_to", r"(?:next to|beside|by|besides) (?:the )?(?P<a>.+)"),
    ("near", r"(?:near|close to|nearby) (?:the )?(?P<a>.+)"),
    ("in", r"(?:in|inside|into) (?:the |a )?(?P<a>.+)"),
    ("on", r"(?:on|onto|on top of) (?:the )?(?P<a>.+)"),
)
_WORDING = {"on": "on {a}", "in": "in {a}", "next_to": "next to {a}", "left_of": "left of {a}", "right_of": "right of {a}",
            "between": "between {a} and {b}", "other_side": "on the other side of {a}", "across": "across from {a}",
            "near": "near {a}"}


@dataclass
class Goal:
    rel: str
    a: str                         # the thing it is relative to (a surface, landmark or object id)
    b: str | None = None           # between's second thing, or the side it starts from ("other side of X from Y")
    text: str = ""

    def words(self) -> str:
        return _WORDING[self.rel].format(a=self.a, b=self.b)


@dataclass
class Check:
    ok: bool | None                # None: it can't be checked
    expected: list[str]            # the surfaces that would satisfy it
    why: str


def _resolve(name: str, lines: list[Line], map_: dict[str, Any], extra: set[str] | None = None) -> str | None:
    """An id, or a label ("the stove") that names exactly one landmark or surface."""
    name = name.strip().strip(".?!,").lower()
    ids = set(map_.get("surfaces") or {}) | {t.name for ln in lines for t in ln.things} | (extra or set())
    if name.replace(" ", "_") in ids:
        return name.replace(" ", "_")
    hits = {t.name for ln in lines for t in ln.things if re.fullmatch(rf"{re.escape(t.label.lower())}s?", name)}
    hits |= {i for i in (extra or set()) if re.fullmatch(rf"{re.escape(re.sub(r'_[0-9]+$', '', i).replace('_', ' '))}s?", name)}
    return hits.pop() if len(hits) == 1 else None


def parse(text: str, lines: list[Line], map_: dict[str, Any], objects: set[str] | None = None) -> Goal | None:
    """ "other side of stove_1 from counter_1a", "left of the toaster", "in cup_1" -> a Goal, or None."""
    t = re.sub(r"\s+", " ", text.strip().lower().replace("?", ""))
    t = re.sub(r"^(?:put it |move it |place it |where is |what is |whats |which spot is |spots? )", "", t)
    for rel, pat in _PATTERNS:
        m = re.fullmatch(pat, t)
        if not m:
            continue
        a = _resolve(m.group("a"), lines, map_, objects if rel == "in" else None)
        b = _resolve(m.group("b"), lines, map_) if m.groupdict().get("b") else None
        if a is None or (m.groupdict().get("b") and b is None):
            return None
        return Goal(rel, a, b, text)
    return None


def _where(name: str, lines: list[Line]) -> tuple[Line, int] | None:
    for ln in lines:
        for i, t in enumerate(ln.things):
            if t.name == name:
                return ln, i
    return None


def targets(goal: Goal, lines: list[Line], map_: dict[str, Any], origin: str | None = None) -> tuple[list[str] | None, str]:
    """The surfaces that satisfy a goal, nearest first, or None and why it can't be worked out."""
    if goal.rel == "on":
        return ([goal.a], "") if goal.a in (map_.get("surfaces") or {}) else (None, f"{goal.a} isn't a surface")
    if goal.rel == "in":
        return None, f"the robot can't put things into containers like {goal.a} yet"
    at = _where(goal.a, lines)
    if at is None:
        return None, f"{goal.a} isn't on the layout (not seen yet, or alone on its wall)"
    ln, i = at
    me = ln.things[i]
    surf = [(j, t) for j, t in enumerate(ln.things) if t.kind == "surface" and t.name != goal.a]
    left = [t.name for j, t in reversed(surf) if j < i]       # nearest first
    right = [t.name for j, t in surf if j > i]
    if goal.rel == "left_of":
        return left, ""
    if goal.rel == "right_of":
        return right, ""
    if goal.rel == "next_to":
        return left[:1] + right[:1], ""
    if goal.rel == "near":
        close = [t.name for _, t in surf if math.dist((me.x, me.z), (t.x, t.z)) <= NEAR_M]
        return list(dict.fromkeys(left[:1] + right[:1] + close)), ""
    if goal.rel == "between":
        other = _where(goal.b or "", lines)
        if other is None or other[0] is not ln:
            return None, f"{goal.a} and {goal.b} aren't on one line"
        lo, hi = sorted((i, other[1]))
        return [t.name for j, t in surf if lo < j < hi], ""
    if goal.rel == "other_side":
        start = goal.b or origin
        s = _where(start or "", lines)
        if s is None or s[0] is not ln:
            return None, f"can't tell which side is 'the other side' of {goal.a}" + (f" from {start}" if start else "")
        return (right if s[1] < i else left), ""
    if goal.rel == "across":
        cands = sorted((abs(ln.along(t) - ln.along(me)), t.name) for other in lines
                       if other.room == ln.room and other.facing == (-ln.facing[0], -ln.facing[1]) for t in other.surfaces())
        return [n for d, n in cands if d <= 1.5] or [n for _, n in cands[:1]], ""
    return None, f"unknown relation {goal.rel}"


def check(goal: Goal, where: str | None, lines: list[Line], map_: dict[str, Any], origin: str | None = None) -> Check:
    """Does an object that ended up on `where` satisfy the goal?"""
    want, why = targets(goal, lines, map_, origin)
    if want is None:
        return Check(None, [], why)
    if where in want:
        return Check(True, want, f"{where} is {goal.words()}")
    return Check(False, want, f"{where} is not {goal.words()}" + (f"; {', '.join(want)} would be" if want else "; no spot is"))


def _answer(goal: Goal, lines: list[Line], map_: dict[str, Any]) -> str:
    if goal.rel == "other_side" and goal.b is None:            # no starting side given: both readings
        at = _where(goal.a, lines)
        if at is not None:
            ln, i = at
            sides = [t.name for t in ln.things[:i][::-1] if t.kind == "surface"][:1] + \
                    [t.name for t in ln.things[i + 1:] if t.kind == "surface"][:1]
            parts = [f"from {s}: {', '.join(targets(Goal('other_side', goal.a, s), lines, map_)[0] or []) or 'nothing'}"
                     for s in sides]
            if parts:
                return f"the other side of {goal.a}, " + "; ".join(parts)
    want, why = targets(goal, lines, map_)
    return f"spots {goal.words()}: {', '.join(want) or 'none'}" if want is not None else f"can't tell: {why}"
