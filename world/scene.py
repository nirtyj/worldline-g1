"""Scene data: rooms and items of a MolmoSpaces house, from P1's `get_scene_info` reply or a recorded
`house_info.json` (both come from scenes/loader.py HouseInfo; docs/contracts/m1.md §1.6, docs/scenes.md §4).

Pure Python (no numpy). Coordinates are Isaac world (x, y floor, z up); nothing here converts to Worldline's frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import vocab

Vec3 = tuple[float, float, float]
AABB = tuple[Vec3, Vec3]


@dataclass(frozen=True)
class Room:
    name: str                       # "kitchen", "living_room", "bedroom_2" (ProcTHOR naming, thor/procthor.py:53-68)
    label: str                      # "living room"
    type: str                       # "LivingRoom"
    polygon: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    room_id: int | None = None

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(self.polygon, x, y)


@dataclass(frozen=True)
class SceneItem:
    scene_id: str                   # ProcTHOR / MolmoSpaces id, e.g. "AlarmClock|surface|2|31"
    name: str                       # house agent's name, e.g. "alarm_clock_1" (informational)
    category: str                   # MolmoSpaces category (= THOR type, before aliases)
    thor_type: str                  # AI2-THOR type after vocab.CATEGORY_ALIASES
    room: str | None
    pos: Vec3
    aabb: AABB
    is_static: bool = True
    articulated: bool = False
    body_path: str | None = None
    prim_path: str | None = None

    @property
    def center(self) -> Vec3:
        (x0, y0, z0), (x1, y1, z1) = self.aabb
        return ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)

    @property
    def size(self) -> Vec3:
        (x0, y0, z0), (x1, y1, z1) = self.aabb
        return (x1 - x0, y1 - y0, z1 - z0)


@dataclass
class SceneData:
    house_id: str
    floor_z: float
    bounds: tuple[float, float, float, float]            # xmin, ymin, xmax, ymax
    rooms: list[Room]
    items: list[SceneItem]
    spawn: tuple[float, float, float]                    # x, y, yaw (rad)
    room_points: dict[str, tuple[float, float, float]] = field(default_factory=dict)   # room -> (x, y, yaw)
    source: dict[str, Any] = field(default_factory=dict)
    occupancy_npz: str | None = None
    kind: str = ""                                       # "procthor" | "ithor" | "" (empty scene)

    # ------------------------------------------------------------------ rooms
    def room_at(self, x: float, y: float, nearest: bool = True) -> str | None:
        """Which room's floor polygon contains (x, y)? The nearest room centre if none does
        (thor/procthor.py:71-78, same rule)."""
        for r in self.rooms:
            if r.contains(x, y):
                return r.name
        if not self.rooms or not nearest:
            return None
        return min(self.rooms, key=lambda r: (r.center[0] - x) ** 2 + (r.center[1] - y) ** 2).name

    def room(self, name: str) -> Room | None:
        return next((r for r in self.rooms if r.name == name), None)

    def item(self, scene_id: str) -> SceneItem | None:
        return next((o for o in self.items if o.scene_id == scene_id), None)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_scene_info(cls, d: dict[str, Any]) -> "SceneData":
        """From P1's get_scene_info reply (contract §1.6) or a house_info.json dict (HouseInfo.to_dict)."""
        rooms = []
        for r in d.get("rooms") or []:
            poly = tuple((float(p[0]), float(p[1])) for p in r["polygon"])
            c = r.get("center") or (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
            name = r.get("name") or vocab.snake(r.get("type", "room"))
            rooms.append(Room(name=name, label=name.replace("_", " "), type=r.get("type", ""), polygon=poly,
                              center=(float(c[0]), float(c[1])), room_id=r.get("id", r.get("room_id"))))
        items = []
        for o in d.get("objects") or []:
            bb = o.get("aabb")
            if not bb:
                continue
            aabb = (tuple(float(v) for v in bb[0]), tuple(float(v) for v in bb[1]))
            pos = tuple(float(v) for v in (o.get("pos") or [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]))
            cat = o.get("category") or ""
            items.append(SceneItem(scene_id=str(o["id"]), name=str(o.get("name") or ""), category=cat,
                                   thor_type=vocab.thor_type(cat), room=o.get("room") or None, pos=pos,   # type: ignore[arg-type]
                                   aabb=aabb, is_static=bool(o.get("is_static", True)),   # type: ignore[arg-type]
                                   articulated=bool(o.get("articulated", False)), body_path=o.get("body_path"),
                                   prim_path=o.get("prim_path")))
        sp = d.get("spawn") or {}
        spawn = (float(sp.get("x", 0.0)), float(sp.get("y", 0.0)), float(sp.get("yaw", 0.0)))
        b = d.get("bounds")
        if b is None and d.get("bounds_xy"):
            (x0, y0), (x1, y1) = d["bounds_xy"]
            b = [x0, y0, x1, y1]
        if b is None:
            xs = [p[0] for r in rooms for p in r.polygon] or [0.0]
            ys = [p[1] for r in rooms for p in r.polygon] or [0.0]
            b = [min(xs), min(ys), max(xs), max(ys)]
        rp = {}
        for p in d.get("room_points") or []:
            if p.get("ok", True) and p.get("room"):
                rp[p["room"]] = (float(p["x"]), float(p["y"]), float(p.get("yaw", 0.0)))
        src = d.get("source") if isinstance(d.get("source"), dict) else {"source": d.get("source")}
        kind = str(d.get("kind") or src.get("kind") or "")
        return cls(house_id=str(d.get("house_id", "")), floor_z=float(d.get("floor_z", 0.0) or 0.0),
                   bounds=tuple(float(v) for v in b), rooms=rooms, items=items, spawn=spawn,   # type: ignore[arg-type]
                   room_points=rp, source=src, occupancy_npz=d.get("occupancy_npz"), kind=kind)

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneData":
        return cls.from_scene_info(json.loads(Path(path).read_text()))


def point_in_polygon(poly: Iterable[tuple[float, float]], x: float, y: float) -> bool:
    """Even-odd rule (thor/procthor.py:81-90)."""
    pts = list(poly)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def aabb_contains_xy(aabb: AABB, x: float, y: float, margin: float = 0.0) -> bool:
    (x0, y0, _), (x1, y1, _) = aabb
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


def xy_dist_to_rect(px: float, py: float, cx: float, cy: float, hx: float, hy: float) -> float:
    """Distance from a floor point to an axis-aligned rectangle (thor/world.py:_edge_dist)."""
    dx = max(abs(px - cx) - hx, 0.0)
    dy = max(abs(py - cy) - hy, 0.0)
    return math.hypot(dx, dy)
