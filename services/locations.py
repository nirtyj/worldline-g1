"""list_locations (doc §5.1; PLAN §5.11): the named places the robot can walk to, nearest first, with the walking
distance from where it is now (one Dijkstra over the occupancy grid from the current GT pose).

Types: "surface" (a stand), "room" (a room keypoint), "start", "person" (`user`, the delivery keypoint). Surfaces
carry their height and too_high / too_low against the G1 workspace, so the planner can say "the robot can't reach
that shelf" before walking there.
"""

from __future__ import annotations

import math
from typing import Any

from api.results import NamedLocation
from api.types import Pose2D


class LocationsService:
    def __init__(self, world: Any, workspace: dict | None = None):
        self.world = world
        ws = dict(workspace or {})
        self.h_min = float(ws.get("obj_z_min_m", 0.55))
        self.h_max = float(ws.get("obj_z_max_m", 1.20))

    def _entries(self) -> list[tuple[str, str, Any]]:
        m = self.world.static_map()
        out = []
        for k in m.keypoints.values():
            typ = {"surface": "surface", "room": "room", "start": "start"}.get(k.kind, "surface")
            out.append((k.name, typ, k))
        if m.user_surface and m.user_surface in m.keypoints:
            out.append(("user", "person", m.keypoints[m.user_surface]))
        return out

    def list(self, current: Pose2D | tuple | None = None, query: str | None = None) -> list[NamedLocation]:
        m = self.world.static_map()
        if current is None:
            p = self.world.robot_pose()
            current = (p.x, p.y)
        cx, cy = (current.x, current.y) if hasattr(current, "x") else (float(current[0]), float(current[1]))
        entries = self._entries()
        pts = [(k.x, k.y) for _, _, k in entries]
        D = m.grid.distances([(cx, cy)], pts)[0] if pts else []
        words = [w for w in (query or "").lower().replace("_", " ").split() if w]
        locs = []
        for (name, typ, k), d in zip(entries, D):
            desc = k.desc if typ != "person" else f"where the user wants things delivered ({k.name})"
            hay = set(f"{name} {desc} {k.room or ''} {typ}".lower().replace("_", " ").replace(",", " ").split())
            if words and not all(w in hay or w.rstrip("s") in hay or f"{w}s" in hay for w in words):
                continue
            height = too_high = too_low = None
            s = m.surfaces.get(k.name) if typ in ("surface", "person") else None
            if s is not None:
                height = s.height
                too_high = s.height > self.h_max
                too_low = s.height < self.h_min
            locs.append(NamedLocation(name=name, distance_m=round(float(d), 2) if math.isfinite(d) else None,
                                      type=typ, room=k.room, desc=desc, height_m=height,
                                      too_high=bool(too_high), too_low=bool(too_low)))
        locs.sort(key=lambda l: (l.distance_m is None, l.distance_m if l.distance_m is not None else 0.0, l.name))
        return locs
