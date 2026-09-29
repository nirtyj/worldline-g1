"""Spatial memory: where things are, where they usually are, and what the user said.

Only the runtime writes it, and only from verified belief (a look, or the camera
while driving). It never reads the simulator. A new session loads it into belief
as source="memory", verified=False: a hint about where to go, to be confirmed by
looking, because things may have moved since.

    runs/memory/<scene>.json
      {"objects":   {"bread_1": {"type": "bread", "surface": "counter_2b", "pos": [x, z],
                                 "seen_wall": ..., "status": "seen",
                                 "history": [{"surface": "counter_2a", "wall": ...}, ...]}},
       "landmarks": {"microwave_1": {"label": "microwave", "near": "counter_1d", ...}},
       "looked":    {"counter_1d": {"seen_wall": ...}},
       "notes":     {"its next to the toaster": {"about": "microwave", "seen_wall": ...}}}

Each object keeps its last few sightings, so memory knows where a thing usually
is, not only where it was last. How a miss is handled depends on how much that
kind of thing moves (after LT-Mem):
  seen on a surface (verified)           -> status seen, a sighting added to its history
  seen in a hand                         -> status held (last place kept in the history)
  looked for and missing, a stable thing -> status missed: its place is held once
  looked for and missing again, or a
  thing that moves around                -> status moved: no current place, but the
                                            history and the usual place remain
  unverified claims and remembered data  -> never written back
"""

from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from brains.interface import UNKNOWN

ROOT = Path(os.environ.get("WORLDLINE_RUNS") or Path(__file__).resolve().parents[1] / "runs") / "memory"
SECTIONS = ("objects", "landmarks", "looked", "notes", "observations")
MAX_NOTES = 30
MAX_OBSERVATIONS = 40
HISTORY = 8

# Things people carry around: a miss means it has probably moved.
VOLATILE_TYPES = {
    "apple", "bread", "egg", "tomato", "potato", "lettuce", "cell_phone", "credit_card", "key_chain",
    "remote_control", "pen", "pencil", "book", "mug", "cup", "newspaper", "cloth", "dish_sponge",
    "spray_bottle", "watch", "bottle", "wine_bottle", "soap_bar", "tissue_box", "toilet_paper", "pillow",
    "laptop", "basket_ball", "baseball_bat", "tennis_racket", "teddy_bear", "alarm_clock", "cd", "box",
    "butter_knife", "knife", "fork", "spoon", "spatula", "ladle", "salt_shaker", "pepper_shaker",
}


def load_vocab(path: Path | None = None) -> set[str]:
    """The fixed pickupable vocabulary (config/vocab/pickupable_types.yaml, PLAN 1.3 #10), read with a
    tiny parser (a ``types:`` list of ``- name`` lines) so the runtime needs no YAML library."""
    path = path or Path(__file__).resolve().parents[1] / "config" / "vocab" / "pickupable_types.yaml"
    out: set[str] = set()
    try:
        in_types = False
        for line in path.read_text().splitlines():
            if re.match(r"^types\s*:", line):
                in_types = True
                continue
            if in_types:
                m = re.match(r"^\s*-\s*([a-z0-9_]+)", line)
                if m:
                    out.add(m.group(1))
                elif line.strip() and not line.startswith((" ", "\t", "#")):
                    in_types = False
    except OSError:
        pass
    return out


VOLATILE_TYPES = VOLATILE_TYPES | load_vocab()     # THOR snake_case, the same names


def volatility(kind: str, history: list[dict[str, Any]]) -> str:
    """'stable' or 'volatile': learned from its history once there is some, else by kind."""
    if len(history) >= 3:
        moves = sum(1 for a, b in zip(history, history[1:]) if a["surface"] != b["surface"])
        return "volatile" if moves / (len(history) - 1) > 0.34 else "stable"
    return "volatile" if kind in VOLATILE_TYPES else "stable"


def usual_place(history: list[dict[str, Any]]) -> str | None:
    """Where it has been seen most often, recent sightings counting a little more."""
    if not history:
        return None
    score: collections.Counter[str] = collections.Counter()
    for i, h in enumerate(history):
        score[h["surface"]] += 1.0 + 0.1 * i
    return score.most_common(1)[0][0]


class SpatialMemory:
    def __init__(self, scene: str | None, root: Path | None = None) -> None:
        root = ROOT if root is None else root                  # looked up at call time (tests relocate it)
        self.path = root / f"{scene}.json" if scene else None
        self.data: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in SECTIONS}
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
            except (OSError, ValueError):
                raw = {}
            if raw and not any(k in raw for k in SECTIONS):
                raw = {"objects": raw}                       # the first, objects-only format
            for k in SECTIONS:
                self.data[k] = dict(raw.get(k) or {})
        for m in self.data["objects"].values():             # older files had no history
            m.setdefault("status", "seen")
            m.setdefault("history", [{"surface": m["surface"], "wall": m["seen_wall"]}] if m.get("surface") else [])

    @property
    def items(self) -> dict[str, dict[str, Any]]:
        return self.data["objects"]

    # ------------------------------------------------------------------
    def entries(self, now: float) -> list[dict[str, Any]]:
        """Memory in the shape BeliefState.load_memory takes, timed on the session clock."""
        wall = time.time()
        ago = lambda w: round(now - (wall - w), 1)
        out = []
        for oid, m in sorted(self.data["objects"].items()):
            hist = m.get("history") or []
            out.append({"id": oid, "kind": "object", "type": m.get("type", "object"),
                        "surface": m.get("surface") if m.get("status") in ("seen", "missed") else None,
                        "pos": m.get("pos"), "last_seen_t": ago(m["seen_wall"]), "status": m.get("status"),
                        "usual": usual_place(hist), "volatility": volatility(m.get("type", ""), hist),
                        "history": [{"surface": h["surface"], "t": ago(h["wall"])} for h in hist[-3:]]})
        out += [{"id": k, "kind": "landmark", "label": m.get("label"), "near": m.get("near"),
                 "pos": m.get("pos"), "last_seen_t": ago(m["seen_wall"])} for k, m in sorted(self.data["landmarks"].items())]
        out += [{"id": k, "kind": "looked", "last_seen_t": ago(m["seen_wall"])} for k, m in sorted(self.data["looked"].items())]
        return out

    def update(self, belief: Any, now: float, force: bool = False) -> bool:
        """Fold what belief verified into memory; write the file if something changed
        (or ``force``, at the end of a session, to keep last-seen times fresh)."""
        wall = time.time()
        to_wall = lambda t: round(wall - (now - t), 1)
        changed = False
        objects = self.data["objects"]
        for oid, ob in belief.objects.items():
            fact = ob.where
            if fact.source == "memory":
                continue                              # nothing new was learned about it
            at_wall = to_wall(fact.t)
            m = objects.get(oid)
            if m is not None and max(m.get("seen_wall", 0), m.get("absent_wall", 0)) >= at_wall:
                continue                              # memory already has this, or something newer
            here = fact.value
            if fact.verified and isinstance(here, str) and here != UNKNOWN and not here.startswith("hand"):
                if m is None:
                    m = objects[oid] = {"type": ob.type, "history": []}
                    changed = True
                hist = m["history"]
                if hist and hist[-1]["surface"] == here:
                    hist[-1]["wall"] = at_wall        # the same sighting, refreshed
                else:
                    hist.append({"surface": here, "wall": at_wall})
                    del hist[:-HISTORY]
                    changed = True
                changed |= m.get("surface") != here or m.get("status") != "seen"
                m.update(surface=here, status="seen", missed=0, seen_wall=at_wall,
                         pos=[round(ob.pose["x"], 2), round(ob.pose["z"], 2)] if ob.pose else m.get("pos"))
            elif m is None:
                continue
            elif fact.verified and isinstance(here, str) and here.startswith("hand"):
                if m.get("status") != "held":
                    m.update(status="held", surface=None, seen_wall=at_wall)
                    changed = True
            elif fact.source == "look_absent" and m.get("status") in ("seen", "missed"):
                m["absent_wall"] = at_wall
                if volatility(m.get("type", ""), m["history"]) == "stable" and not m.get("missed"):
                    m.update(status="missed", missed=1)          # hold its place once
                else:
                    m.update(status="moved", surface=None, missed=m.get("missed", 0) + 1)
                changed = True
        for k, lm in belief.landmarks.items():
            if lm.near.source == "memory":
                continue
            if k not in self.data["landmarks"]:
                changed = True
            self.data["landmarks"][k] = {"label": lm.label, "near": lm.near.value,
                                         "pos": [round(v, 2) for v in lm.pos] if lm.pos else None,
                                         "seen_wall": to_wall(lm.near.t)}
        for k, f in belief.looked.items():
            if f.source == "memory":
                continue
            if k not in self.data["looked"]:
                changed = True
            self.data["looked"][k] = {"seen_wall": to_wall(f.t)}
        if changed or force:
            self.save()
        return changed

    # ------------------------------------------------------------------
    def add_note(self, text: str, about: str | None = None) -> None:
        """Something the user said about the home, kept word for word."""
        key = " ".join(text.lower().split())
        self.data["notes"][key] = {"text": text.strip(), "about": about, "seen_wall": round(time.time(), 1)}
        for old in sorted(self.data["notes"], key=lambda k: self.data["notes"][k]["seen_wall"])[:-MAX_NOTES]:
            del self.data["notes"][old]
        self.save()

    def add_observation(self, text: str, where: str | None, confidence: float, source: str) -> None:
        """What System 1 noticed that the object list can't hold ("the fridge door is open").
        Kept unverified, with where it came from, like a note; the same text at the same
        spot is refreshed rather than repeated."""
        key = f"{(where or '-')}|{text.strip().lower()}"
        self.data["observations"][key] = {"text": text.strip(), "where": where, "confidence": round(confidence, 2),
                                          "source": source, "seen_wall": round(time.time(), 1)}
        for old in sorted(self.data["observations"],
                          key=lambda k: self.data["observations"][k]["seen_wall"])[:-MAX_OBSERVATIONS]:
            del self.data["observations"][old]
        self.save()

    def observations(self) -> list[dict[str, Any]]:
        wall = time.time()
        return [{**o, "ago_s": round(wall - o["seen_wall"])}
                for o in sorted(self.data["observations"].values(), key=lambda o: o["seen_wall"])]

    def notes(self) -> list[dict[str, Any]]:
        wall = time.time()
        return [{"text": n["text"], "about": n.get("about"), "ago_s": round(wall - n["seen_wall"])}
                for n in sorted(self.data["notes"].values(), key=lambda n: n["seen_wall"])]

    def tree(self, map_: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
        """room -> surface -> objects last seen there (the hierarchy the recall tool reports)."""
        room_of = {k: v.get("room") or "here" for k, v in (map_.get("keypoints") or {}).items()}
        out: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for oid, m in sorted(self.data["objects"].items()):
            if m.get("surface"):
                out[room_of.get(m["surface"], "here")][m["surface"]].append(oid)
        return {r: dict(s) for r, s in out.items()}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)
