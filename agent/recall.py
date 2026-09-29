"""The recall tool: the planner asks memory instead of carrying all of it in every prompt.

    recall("mug")                      where the mug is, was, and usually is
    recall("kitchen")                  what is known to be in the kitchen, room by surface
    recall("what did I ask for last time")   requests and deliveries from earlier sessions
    recall("keys")                     also finds the user's notes that mention keys

It answers locally and instantly from belief, spatial memory (agent/memory.py),
the user's notes and the episode log (agent/episodes.py). No model call.
"""

from __future__ import annotations

import re
import time
from typing import Any

from brains.interface import UNKNOWN

from . import layout
from .episodes import load_episodes, session_wall
from .memory import usual_place

STOP_WORDS = {"the", "a", "an", "my", "is", "are", "where", "what", "whats", "did", "do", "you", "i", "me",
              "of", "in", "on", "to", "for", "it", "was", "were", "have", "has", "seen", "see", "find",
              "about", "and", "or", "any", "last", "time", "user", "please", "can", "could", "there"}
HISTORY_WORDS = {"last", "before", "earlier", "yesterday", "previous", "ago", "asked", "ask", "did", "history", "sessions",
                 "request", "requests", "requested", "deliver", "delivered", "deliveries", "delivery",
                 "bring", "brought", "fetch", "fetched"}
MAX_CHARS = 1100


def _ago(seconds: float) -> str:
    s = max(0.0, seconds)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.0f} h" if s < 172800 else f"{s / 86400:.0f} days"


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP_WORDS:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


class Recaller:
    def __init__(self, scene: str | None) -> None:
        self.scene = scene
        self.past = load_episodes(scene) if scene else []      # earlier sessions, read once

    def answer(self, query: str, belief: Any, memory: Any, map_: dict[str, Any], now: float,
               current: list[dict[str, Any]]) -> str:
        words = _words(query)
        raw = set(re.findall(r"[a-z]+", query.lower()))
        lines: list[str] = []
        rooms = map_.get("rooms") or {}
        room_of = {k: v.get("room") for k, v in (map_.get("keypoints") or {}).items()}

        # objects: now, memory, usual place, history
        mem = memory.data["objects"]
        ids = sorted(set(belief.objects) | set(mem))
        hits = [oid for oid in ids if any(w in oid.replace("_", " ").split() or w in oid for w in words)]
        for oid in hits[:6]:
            ob = belief.objects.get(oid)
            m = mem.get(oid, {})
            parts = []
            if ob is not None:
                w = ob.where
                if w.source == "memory":
                    parts.append(f"not seen this session (memory says {w.value})")
                elif w.value == UNKNOWN:
                    parts.append(f"not where expected on the last look ({w.source})")
                else:
                    parts.append(f"now {w.value} ({'seen' if w.verified else w.source}, {_ago(now - w.t)} ago)")
            hist = m.get("history") or []
            if hist:
                wall = time.time()
                usual = usual_place(hist)
                parts.append("usually " + usual + (f" ({room_of.get(usual)})" if room_of.get(usual) else ""))
                parts.append("seen at " + ", ".join(f"{h['surface']} {_ago(wall - h['wall'])} ago" for h in hist[-3:][::-1]))
            if m.get("status") in ("missed", "moved"):
                parts.append(f"memory status: {m['status']}")
            lines.append(f"{oid}: " + "; ".join(parts or ["nothing known"]))

        # a room or a surface: what is known to be there
        tree = memory.tree(map_)
        for rname, info in rooms.items():
            label_words = set(info["label"].split())
            if label_words & set(words) or rname in raw:
                content = tree.get(rname, {})
                inside = "; ".join(f"{s}: {', '.join(o)}" for s, o in content.items()) or "nothing remembered yet"
                marks = [k for k, lm in belief.landmarks.items() if room_of.get(lm.near.value) == rname]
                lines.append(f"{info['label']}: spots {', '.join(info['spots'])}. Objects: {inside}."
                             + (f" Landmarks: {', '.join(marks)}." if marks else ""))
        for surface in (map_.get("surfaces") or {}):
            if surface in raw or surface.replace("_", " ") in query.lower():
                here = [oid for oid, m in mem.items() if m.get("surface") == surface]
                lines.append(f"{surface}: {', '.join(here) or 'nothing remembered there'}")

        # landmarks, then what is next to what
        for k, lm in belief.landmarks.items():
            if any(w in k or w in lm.label for w in words):
                lines.append(f"{k} ({lm.label}): near {lm.near.value}")
        marks = {k: {"label": lm.label, "near": lm.near.value, "pos": lm.pos} for k, lm in belief.landmarks.items()}
        lines += layout.about(query, layout.build(map_, marks), map_)

        # the user's notes
        for n in memory.notes():
            if any(w in n["text"].lower() for w in words) or (n.get("about") and any(w in n["about"] for w in words)):
                lines.append(f'note ({_ago(n["ago_s"])} ago): "{n["text"]}"')

        # what System 1 noticed (unverified)
        for o in memory.observations():
            if any(w in o["text"].lower() for w in words) or (o.get("where") and any(w in o["where"] for w in words)):
                lines.append(f'noticed {_ago(o["ago_s"])} ago' + (f' at {o["where"]}' if o.get("where") else "")
                             + f' (unverified): {o["text"]}')

        # what happened before: requests and deliveries in earlier sessions and this one
        if raw & HISTORY_WORDS:
            lines += self._history(words, current, now)

        if not lines:
            known = ", ".join(r["label"] for r in rooms.values()) or "this room"
            kinds = sorted({str(m.get("type", "")).replace("_", " ") for m in mem.values()}
                           | {ob.type.replace("_", " ") for ob in belief.objects.values()})
            lines.append(f"Nothing in memory matches {query!r}: it has never been seen here. Rooms: {known}. "
                         f"Kinds of things seen: {', '.join(kinds[:40]) or 'none yet'}. "
                         f"Spots not looked at yet may still hold it.")
        text = "\n".join(lines)
        return text if len(text) <= MAX_CHARS else text[:MAX_CHARS - 1] + "…"

    def _history(self, words: list[str], current: list[dict[str, Any]], now: float) -> list[str]:
        wall = time.time()
        events: list[tuple[float, str]] = []
        for rows in self.past + [current]:
            start = session_wall(rows) or wall - now
            asked = {r.get("id") for r in rows if r.get("type") == "classified"
                     and r.get("kind") in ("request", "correction")}      # not questions or chat
            for r in rows:
                t = start + r.get("t", 0.0) if rows is not current else wall - (now - r.get("t", 0.0))
                if r.get("type") == "heard" and r.get("id") in asked:
                    events.append((t, f'user asked "{r.get("text")}"'))
                elif r.get("type") == "delivered":
                    events.append((t, f"delivered {r.get('object')} to {r.get('surface')}"))
        topic = [w for w in words if w not in HISTORY_WORDS]           # "bring", "asked" say nothing about what
        if topic:
            focused = [e for e in events if any(w in e[1].lower() for w in topic)]
            events = focused or events
        return [f"{_ago(wall - t)} ago: {text}" for t, text in events[-6:]] or ["no earlier sessions recorded"]
