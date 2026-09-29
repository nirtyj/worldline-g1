"""Ground-truth "where" of every object, computed geometrically (PLAN §6.2.2; THOR semantics re-derived).

AI2-THOR reported `parentReceptacles`; MolmoSpaces has no receptacle relations (docs/scenes.md §8), so they are
derived here from AABBs. The answer uses THOR's vocabulary (thor/world.py:547-560), which belief, memory, the
planner prompt and eval scoring all read:

    "hand:<arm>"          held (attached)                                    [THOR: "hand"; PLAN §4.4 state.py]
    <surface stretch>     on surface furniture: the nearest stretch of it     kitchen_counter_1b
    <alias>/<snake type>  surface furniture that has no stand (skipped/folded) dining_table
    <container snake>     inside a closed container                          fridge, microwave, cabinet, drawer
    (through a held container or a stacked prop, depth <= 2)                 the bowl's / book's where
    "floor"               centre below 0.2 m
    "unknown"

Rules, in order (the first match wins):
  1. hand: attached to a hand (the sim attach; M2b adds "within 0.12 m of the palm and lifted >= 3 cm").
  2. container: the object's centre is strictly inside a closed container's box (fridge, microwave, cabinet, ...).
     This runs before the surface rule so an apple in a microwave standing on a counter is "microwave".
  3. surface top: xy inside a surface furniture footprint (+3 cm) and its bottom within [-0.02, +0.05] m of the
     furniture top (AABB z max).
  4. held container / stacked: resting in or on another object (bowl, plate, ..., or any prop): its where.
  5. surface footprint: xy inside a surface footprint and bottom between the furniture's bottom and top (lower
     shelves, a sofa seat, a bed under its pillows). The most specific (smallest) footprint wins.
  5b. other receptacle: resting on or in any other solid thing (a chair seat, a toilet tank): snake(type).
  6. floor: centre height < 0.2 m.
  7. unknown.
Each answer also names the rule that produced it (`WhereAnswer.rule`), for tests and debugging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from . import vocab
from .mapgen import StaticMap

Box = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class WhereParams:
    top_below: float = 0.02
    top_above: float = 0.05
    footprint_margin: float = 0.03
    container_shrink: float = 0.01
    floor_h: float = 0.20
    max_depth: int = 2


@dataclass(frozen=True)
class WhereAnswer:
    where: str
    rule: str
    support: str | None = None          # scene id / object id of what it rests on or is inside


def center(b: Box) -> tuple[float, float, float]:
    return tuple((b[0][i] + b[1][i]) / 2 for i in range(3))   # type: ignore[return-value]


def _inside_xy(b: Box, x: float, y: float, m: float) -> bool:
    return b[0][0] - m <= x <= b[1][0] + m and b[0][1] - m <= y <= b[1][1] + m


def _inside_strict(b: Box, p, shrink: float) -> bool:
    return all(b[0][i] + shrink < p[i] < b[1][i] - shrink for i in range(3))


class WhereModel:
    """Computes where for objects given their current boxes. Static furniture comes from the StaticMap."""

    def __init__(self, smap: StaticMap, params: WhereParams | None = None):
        self.map = smap
        self.p = params or WhereParams()
        self.floor_z = smap.floor_z
        self.surface_items = [t for t in smap.things if vocab.is_surface(t.thor_type)]
        self.container_items = [t for t in smap.things
                                if vocab.is_container(t.thor_type) and not vocab.is_pickupable(t.thor_type)]
        # any other solid, non-pickupable thing an object can rest on or in (chair seat, toilet tank, stool)
        self.other_items = [t for t in smap.things
                            if not vocab.is_surface(t.thor_type) and not vocab.is_pickupable(t.thor_type)
                            and t.thor_type not in vocab.NON_SOLID_TYPES and t.thor_type not in ("Painting",)]
        self._stretches: dict[str, list] = {}
        for s in smap.surfaces.values():
            if not getattr(s, "extra_side", False):     # a stand on another free side shares its stretch's box: an
                self._stretches.setdefault(s.furniture_id, []).append(s)   # object's where stays the stretch's name

    # ------------------------------------------------------------------
    def furniture_name(self, scene_id: str, x: float, y: float) -> str:
        """The nearest stretch of this furniture, else its alias, else snake(type) (thor/world.py:555-557)."""
        segs = self._stretches.get(scene_id)
        if segs:
            return min(segs, key=lambda s: math.dist(s.center, (x, y))).name
        if scene_id in self.map.alias:
            return self.map.alias[scene_id]
        it = next((t for t in self.map.things if t.scene_id == scene_id), None)
        return vocab.snake(it.thor_type) if it else "unknown"

    def where(self, oid: str, boxes: Mapping[str, Box], held: Mapping[str, str] | None = None,
              depth: int = 0) -> WhereAnswer:
        """oid's where. boxes: current AABB of every pickupable object (oid -> box); held: oid -> arm."""
        p = self.p
        held = held or {}
        if oid in held:
            return WhereAnswer(f"hand:{held[oid]}", "hand")
        box = boxes.get(oid)
        if box is None:
            return WhereAnswer("unknown", "missing")
        c = center(box)
        bottom = box[0][2]
        # 2. inside a closed container
        for t in self.container_items:
            if _inside_strict(t.aabb, c, p.container_shrink):
                return WhereAnswer(vocab.snake(t.thor_type), "container", t.scene_id)
        # 3. on a surface top
        tops = [t for t in self.surface_items if _inside_xy(t.aabb, c[0], c[1], p.footprint_margin)
                and t.aabb[1][2] - p.top_below <= bottom <= t.aabb[1][2] + p.top_above]
        if tops:
            t = min(tops, key=lambda t: abs(bottom - t.aabb[1][2]))
            return WhereAnswer(self.furniture_name(t.scene_id, c[0], c[1]), "surface_top", t.scene_id)
        # 4. in / on another object (held container or stacked prop)
        if depth < p.max_depth:
            best = None
            for other, ob in boxes.items():
                if other == oid or other in held:
                    continue
                if not _inside_xy(ob, c[0], c[1], 0.0):
                    continue
                if not (ob[0][2] - p.top_below <= bottom <= ob[1][2] + p.top_above):
                    continue
                if c[2] <= center(ob)[2] - 1e-3 and not self._is_held_container(other):
                    continue                       # below the other's centre: not resting on it
                area = (ob[1][0] - ob[0][0]) * (ob[1][1] - ob[0][1])
                if best is None or area < best[0]:
                    best = (area, other)
            if best is not None:
                inner = self.where(best[1], boxes, held, depth + 1)
                rule = "held_container" if self._is_held_container(best[1]) else "stacked"
                return WhereAnswer(inner.where, rule, best[1])
        # 5. within a surface footprint, below its top (shelves, seats, beds)
        foot = [t for t in self.surface_items if _inside_xy(t.aabb, c[0], c[1], p.footprint_margin)
                and t.aabb[0][2] - p.top_below <= bottom <= t.aabb[1][2] + p.top_above]
        if foot:
            t = min(foot, key=lambda t: (t.aabb[1][0] - t.aabb[0][0]) * (t.aabb[1][1] - t.aabb[0][1]))
            return WhereAnswer(self.furniture_name(t.scene_id, c[0], c[1]), "surface_footprint", t.scene_id)
        # 5b. resting on / in another solid thing (THOR: the parent receptacle's snake type)
        other = [t for t in self.other_items if _inside_xy(t.aabb, c[0], c[1], 0.0)
                 and t.aabb[0][2] + p.floor_h <= bottom <= t.aabb[1][2] + p.top_above]
        if other:
            t = min(other, key=lambda t: (t.aabb[1][0] - t.aabb[0][0]) * (t.aabb[1][1] - t.aabb[0][1]))
            return WhereAnswer(vocab.snake(t.thor_type), "other_receptacle", t.scene_id)
        if c[2] - self.floor_z < p.floor_h:
            return WhereAnswer("floor", "floor")
        return WhereAnswer("unknown", "none")

    def _is_held_container(self, oid: str) -> bool:
        spec = self.map.objects.get(oid)
        return bool(spec and spec.thor_type in vocab.HELD_CONTAINERS)

    def all(self, boxes: Mapping[str, Box], held: Mapping[str, str] | None = None) -> dict[str, WhereAnswer]:
        return {oid: self.where(oid, boxes, held) for oid in boxes}

    def surface_for_where(self, where: str) -> str | None:
        """The map surface a where names (a stretch), or None (container, floor, hand, unknown)."""
        return where if where in self.map.surfaces else None
