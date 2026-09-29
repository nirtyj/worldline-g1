"""A synthetic one-room house for the free-sides (world/mapgen.py) and far-stance (services/reachability.py
find_far_stance, services/navigation.py's two-leg reach_stance) tests, written as a recorded house
(house_info.json + occupancy.npz) so LiteWorld loads it exactly as it loads the MolmoSpaces houses.

    kitchen 6.0 x 3.8 m, walls 5 cm
    dining table  x 2.40-4.20, y 2.00-2.90, top 0.76 m: free-standing (free floor on all four sides); its own
                  stands (1a, 1b: one per stretch) are on its +y side (both long sides tie; +y is tried first)
    counter       x 0.80-2.80, y 0.00-0.60, top 0.90 m: against the south wall (one free long side)
    on the table  wine_bottle_1 0.06 m past the +y edge, in front of 1a (reachable from that stand; tall enough
                           for the head camera to see it that close)
                  bottle_1      0.08 m past the -y edge, behind 1b (the other side)
                  mug_1         0.08 m past the +x end, half-way across (the end)
                  bowl_1        the middle, 0.45 m past every edge (beyond any reach)
    spawn         (1.2, 1.6), in the room, facing +x
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOM = (0.0, 0.0, 6.0, 3.8)
TABLE = ((2.40, 2.00, 0.0), (4.20, 2.90, 0.76))
COUNTER = ((0.80, 0.00, 0.0), (2.80, 0.60, 0.90))
TOP = TABLE[1][2]
OBJECTS = {            # scene id: (category, centre xy, half size xyz)
    "WineBottle|surface|1|3": ("WineBottle", (2.70, 2.84), (0.04, 0.04, 0.15)),
    "Bottle|surface|1|4": ("Bottle", (3.75, 2.08), (0.04, 0.04, 0.12)),
    "Mug|surface|1|5": ("Mug", (4.12, 2.45), (0.05, 0.05, 0.05)),
    "Bowl|surface|1|6": ("Bowl", (3.30, 2.45), (0.08, 0.08, 0.04)),
}
RES = 0.05
ORIGIN = (-0.3, -0.3)
SPAWN = (1.2, 1.6, 0.0)


def _item(sid: str, cat: str, aabb, static: bool = True) -> dict:
    (x0, y0, z0), (x1, y1, z1) = aabb
    return {"id": sid, "name": "", "category": cat, "room": "kitchen", "pos": [(x0 + x1) / 2, (y0 + y1) / 2,
            (z0 + z1) / 2], "aabb": [[x0, y0, z0], [x1, y1, z1]], "is_static": static}


def write(dir_: str | Path, *, extra_obstacles: list[tuple[float, float, float, float]] = ()) -> Path:
    """Write the house into dir_ (created); extra_obstacles: (x0, y0, x1, y1) boxes of obstacle cells."""
    d = Path(dir_)
    d.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = ROOM
    objects = [_item("DiningTable|1|1|0", "DiningTable", TABLE), _item("CounterTop|1|2", "CounterTop", COUNTER)]
    for sid, (cat, (cx, cy), (hx, hy, hz)) in OBJECTS.items():
        objects.append(_item(sid, cat, ((cx - hx, cy - hy, TOP), (cx + hx, cy + hy, TOP + 2 * hz)), static=False))
    info = {
        "house_id": "synthetic-table", "kind": "procthor", "source": "tests/fakes/table_house.py", "floor_z": 0.0,
        "bounds_xy": [[x0, y0], [x1, y1]],
        "rooms": [{"room_id": 1, "name": "kitchen", "type": "Kitchen",
                   "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "center": [(x0 + x1) / 2, (y0 + y1) / 2]}],
        "objects": objects,
        "spawn": {"x": SPAWN[0], "y": SPAWN[1], "yaw": SPAWN[2]},
        "room_points": [{"room": "kitchen", "room_id": 1, "ok": True, "x": SPAWN[0], "y": SPAWN[1], "yaw": 0.0}],
    }
    (d / "house_info.json").write_text(json.dumps(info, indent=1))
    W = int(round((x1 - ORIGIN[0] + 0.3) / RES))
    H = int(round((y1 - ORIGIN[1] + 0.3) / RES))
    xs = ORIGIN[0] + (np.arange(W) + 0.5) * RES
    ys = ORIGIN[1] + (np.arange(H) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    raw = np.zeros((H, W), dtype=np.uint8)
    inside = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    raw[~inside] = 2
    wall = inside & ((X < x0 + 0.05) | (X > x1 - 0.05) | (Y < y0 + 0.05) | (Y > y1 - 0.05))
    raw[wall] = 1
    for (a0, b0, _), (a1, b1, _) in (TABLE, COUNTER):
        raw[inside & (X > a0) & (X < a1) & (Y > b0) & (Y < b1)] = 1
    for a0, b0, a1, b1 in extra_obstacles:
        raw[inside & (X > a0) & (X < a1) & (Y > b0) & (Y < b1)] = 1
    np.savez(d / "occupancy.npz", occ=(raw != 0), raw=raw, resolution=np.array([RES]), origin=np.array(ORIGIN))
    return d
