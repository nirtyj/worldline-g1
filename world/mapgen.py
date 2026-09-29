"""Worldline's static MAP from a MolmoSpaces house: the port of THOR's `_layout_surfaces`, `_name_objects`,
`_name_landmarks` (ludo-runtime thor/world.py:351-445) and `ThorRobot.lookup_keypoints` (thor/robot.py:70-103),
with G1 parameters (PLAN §7.1 table) instead of THOR's reachable-grid teleport stands.

Output shape (`StaticMap.lookup_keypoints()`), THOR keys unchanged so the planner prompt, layout, persona, recall and
eval keep working, plus the G1 extras PLAN §6.2.1 lists:

    {scene, keypoints:{name:{desc, xy:[x,z], room, yaw, kind}},
     edges:[[a, b, metres], ...]                     # all pairs, walking distance on the occupancy grid
     surfaces:{name:{keypoints:[name], desc, height_m, xy:[x,z], room, half, furniture, type}},
     people:{user:{deliver_to_surface, keypoint}},
     rooms:{name:{label, spots:[kp...], center, polygon}},
     nav_speed_mps, max_reach_height_m, min_reach_height_m, robot:"unitree_g1", grid_step, source}

Naming (so the planner prompt and eval keep their words):
  surfaces   one per stretch of at most SEGMENT_M of surface furniture: `[room_]base_N[a-h]`, numbered per
             (room, kind) in scene-id order, base = "counter" for counter tops else snake(THOR type):
             kitchen_counter_1a, living_room_tv_stand_1, bedroom_bed_1.
  keypoints  == surface names (one stand per stretch), plus `start` and one keypoint per room (its name).
             Free-standing tables and islands (`free_sides`) also get a stand on every other free side: more
             surfaces of the same stretches, lettered after the furniture's own (H38 kitchen_dining_table_1c/1d
             across the table from 1a/1b); an object's `where` stays the stretch's own name (world/where.py).
  objects    pickupable things (fixed vocabulary), `snake(type)_n` numbered in scene-id order: alarm_clock_1.
  landmarks  non-pickupable LANDMARK_TYPES: fridge_1, stove_1 (all burners one), tv_1; `near` = the nearest
             keypoint IN THE SAME ROOM (THOR drift fixed, PLAN §4.4).

Stretches from footprints (M2 gap 1, R.5): a stretch starts as a segment of the furniture's AABB, then is cut to
the furniture's real footprint in it: the occupancy grid's obstacle cells inside the segment, minus built-in appliances
that break through the top (an item mostly inside the footprint, rising from below the top to at least the top: the
stove in a counter). A cut moves an edge only when it moves by more than `footprint_min_cut_m`, so plain rectangular
furniture keeps its AABB stretches exactly. So an L- or U-shaped counter's stands face the counter part that is
really there (H15: the back run inside the U, not the open side of a 1.84 m-deep box), and a stretch never covers a
stove top (K10 counter_2b). A segment with (almost) nothing left is skipped (`skipped`, "built-in appliance").

Stand points (G1): `stand_off_m` from the stretch's front edge (config/g1.yaml: 0.27-0.45 m), facing the edge normal
(so layout's 4-axis facing holds), `stand_clearance_m` from every obstacle (0.25: a keypoint is an A* go_to goal
outside the body's 0.25 m inflation), in the furniture's room, reachable from the spawn, >= 0.5 m from other stands.
Free sides (`free_sides`, on by default; False rebuilds the one-stand-per-stretch maps): a table or an island
(`free_standing_types`) whose two long sides both have room for a regular stand is free-standing; after every
furniture's own stands (which stay exactly as they were: names, letters, poses), each of its stretches gets a stand
on every other side where one fits by the same rules (regular stand-off only, strictly inside the range), without
the sibling-side preference. A run against a wall (its back has no room) keeps one stand per stretch, and the user
surface is never one of the added stands.
(PLAN §7.1 said 0.45-0.60. R.7 measured the arm: it reaches 0.37-0.41 m ahead of the pelvis at counter height and
place never repositions, so a stand 0.35 m back reached no spot on any user surface; from 0.27 m a spot 6 cm past the
edge is in reach; docs/calibration.md §4.) If nothing qualifies, a relaxed pass allows up to 1.0 m and 0.25 m
clearance (`stand_relaxed`); otherwise the stretch is skipped (listed in `skipped`, like THOR).

L/U legs (`split_deep_m`, off by default): on furniture whose footprint fills less than `split_shaped_fill` of its
AABB, a stretch deeper than `split_deep_m` and than it is wide is split along its depth, each part with stands on its
own long sides (docs/calibration.md §4: +4 points of household coverage, but it renames the counter stretches the
eval bindings use).

Coordinates: everything inside StaticMap is Isaac world (x, y, yaw rad); `lookup_keypoints()` converts to
Worldline's frame through world/coords.py only.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import coords, vocab
from .nav_grid import WorldGrid
from .scene import Room, SceneData, SceneItem

LETTERS = "abcdefgh"
EXTRA_LETTERS = "abcdefghijklmnop"      # stands on the other free sides follow the stretches' letters


@dataclass(frozen=True)
class MapParams:
    segment_m: float = 1.0                  # PLAN §7.1 (THOR 1.3)
    stand_off_m: tuple[float, float] = (0.35, 0.50)  # PLAN §7.1 says 0.45-0.60; see the module docstring
    stand_off_relaxed_m: float = 1.0
    stand_step_m: float = 0.05
    lateral_max_m: float = 0.30
    lateral_step_m: float = 0.10
    max_facing_err_deg: float = 20.0        # layout.py snaps stand->centre to an axis within 30 deg
    stand_min_sep_m: float = 0.5
    stand_min_sep_opposed_m: float = 0.3    # back-to-back stands (facing > 90 deg apart), e.g. across a galley
    stand_clearance_m: float = 0.30         # body robot radius 0.25 + 5 cm
    relaxed_clearance_m: float = 0.25
    end_side_penalty: float = 0.3
    open_side_weight: float = 0.3           # score bonus per metre of clearance (<= 1 m) at the stand
    sibling_side_penalty: float = 0.4       # stretches of one furniture prefer one side (so layout lines form)
    room_prefix: bool | None = None         # None: prefix surface names with the room unless the scene is iTHOR
    shelf_merge_m: float = 0.35            # stacked shelf levels fold into the lowest (thor/world.py:366-371)
    robot_radius: float = 0.25              # body's go_to inflation (body/config.py)
    nav_speed_mps: float = 0.45
    max_reach_height_m: float = 1.20
    min_reach_height_m: float = 0.55
    grid_step: float = 0.25                 # UI / RobotMap lattice
    user_surface_rule: str = "thor"         # "thor": nearest stand to the start (Worldline/eval parity);
                                            # "placeable": the same among surfaces a G1 can place on from the stand
    user_max_stand_off_m: float = 0.45      # "placeable": regular stand-off at most this
    room_keypoints: bool = True
    footprints: bool = True                 # cut stretches to the occupancy footprint (R.5); False: THOR's AABBs
    footprint_min_cut_m: float = 0.10       # an edge moves only when the footprint moves it more than this
    footprint_min_fill: float = 0.10        # a segment with less of its area left than this is skipped
    split_deep_m: float = 0.0               # split a stretch deeper than this (an L/U leg) along its depth; 0: off
    split_shaped_fill: float = 0.8          # ... only on furniture whose footprint fills less of its AABB than this
    free_sides: bool = True                 # free-standing furniture: a stand on EVERY free side (new letters after
                                            # the existing ones); False: one stand per stretch (the old maps)
    free_standing_types: tuple = ("DiningTable", "CounterTop", "CoffeeTable", "SideTable", "Desk")   # tables, islands
    lite_world: tuple = ()                  # INTERIM overrides for a LiteWorld (config mapgen.lite_world), for_source

    @classmethod
    def from_dict(cls, d: dict | None) -> "MapParams":
        d = dict(d or {})
        kw = {}
        for f in cls.__dataclass_fields__:
            if f in d:
                v = d[f]
                if f == "lite_world":
                    kw[f] = tuple(sorted((str(k), tuple(x) if isinstance(x, list) else x)
                                         for k, x in dict(v or {}).items()))
                else:
                    kw[f] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)

    def for_source(self, source: str | None) -> "MapParams":
        """The parameters a world with this `source` builds its map with: a LiteWorld (lite-gt) takes the INTERIM
        `lite_world` overrides (M2a's stands, matching the lite world's M2a arm, services/reachability.py
        G1Workspace.for_world); the Isaac worlds build the R.7 stands."""
        if not self.lite_world or source != "lite-gt":
            return self
        import dataclasses
        return dataclasses.replace(self, lite_world=(), **dict(self.lite_world))


@dataclass(frozen=True)
class Surface:
    name: str
    furniture_id: str               # scene id of the furniture
    thor_type: str
    kind: str                       # "counter", "dining table", ...
    num: int
    part: str                       # "" or "a".."h"
    room: str | None
    desc: str                       # "counter 1, part a in the kitchen"
    center: tuple[float, float]     # stretch centre (x, y)
    half: tuple[float, float]       # half-size of the stretch in x and y
    z_top: float                    # world z of the furniture top (AABB)
    z_bottom: float
    height: float                   # z_top above the floor, rounded (THOR height_m)
    stand: tuple[float, float]
    yaw: float                      # Isaac yaw (rad) of the stand, facing the stretch
    side: str                       # which face of the stretch the stand is at: "+x", "-x", "+y", "-y"
    stand_off_m: float              # stand distance from the front edge
    relaxed: bool = False
    extra_side: bool = False        # a stand added on another free side of free-standing furniture (free_sides)

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(x - self.center[0]) <= self.half[0] + margin and abs(y - self.center[1]) <= self.half[1] + margin


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    yaw: float                      # Isaac rad
    kind: str                       # "surface" | "start" | "room"
    room: str | None
    desc: str


@dataclass(frozen=True)
class Landmark:
    name: str                       # "fridge_1", "stove_1"
    type: str                       # snake of the label: "fridge", "stove", "tv"
    label: str
    thor_type: str
    scene_ids: tuple[str, ...]
    center: tuple[float, float, float]
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]]
    near: str | None
    room: str | None
    container: bool = False


@dataclass(frozen=True)
class ObjectSpec:
    oid: str                        # "alarm_clock_1"
    scene_id: str
    thor_type: str                  # "AlarmClock"
    type: str                       # "alarm_clock"
    label: str                      # "alarm clock"
    room: str | None
    pos: tuple[float, float, float]
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass
class StaticMap:
    scene: str
    house_id: str
    params: MapParams
    floor_z: float
    rooms: dict[str, Room]
    surfaces: dict[str, Surface]
    keypoints: dict[str, Keypoint]
    landmarks: dict[str, Landmark]
    objects: dict[str, ObjectSpec]              # pickupables at scene load (initial poses)
    things: list[SceneItem]                     # every scene item (occluders, truth vocabulary)
    edges: list[list[Any]]                      # [a, b, metres]
    user_surface: str | None
    grid: WorldGrid
    alias: dict[str, str] = field(default_factory=dict)          # folded shelf furniture id -> surface
    skipped: list[tuple[str, str]] = field(default_factory=list)
    topdown: dict[str, Any] = field(default_factory=dict)
    source: str = "isaac-gt"
    extras: dict[str, Any] = field(default_factory=dict)
    _lookup: dict[str, Any] | None = None

    @property
    def occupancy(self) -> dict:
        """UI grid (ui/truth.py grid_view): resolution, origin, raw codes, inflated."""
        import numpy as np
        d = self.grid.data
        raw = d.raw_codes if d.raw_codes is not None else d.occ.astype(np.uint8)
        return {"resolution": d.res, "origin": list(d.origin), "raw": raw,
                "inflated": self.grid.nav.inflated.astype(np.uint8), "source": d.source}

    # ------------------------------------------------------------------ queries
    def keypoint(self, name: str) -> Keypoint | None:
        return self.keypoints.get(name)

    def surfaces_of(self, furniture_id: str) -> list[Surface]:
        return [s for s in self.surfaces.values() if s.furniture_id == furniture_id]

    def surface_served_from(self, keypoint: str | None) -> Surface | None:
        """The surface whose stand is this keypoint (keypoint id == surface id)."""
        return self.surfaces.get(keypoint) if keypoint else None

    def room_at(self, x: float, y: float) -> str | None:
        for r in self.rooms.values():
            if r.contains(x, y):
                return r.name
        if not self.rooms:
            return None
        return min(self.rooms.values(), key=lambda r: (r.center[0] - x) ** 2 + (r.center[1] - y) ** 2).name

    def nearest_keypoint(self, x: float, y: float, max_d: float = math.inf,
                         kinds: Iterable[str] = ("surface", "start", "room")) -> tuple[str | None, float]:
        kinds = set(kinds)
        best, bd = None, math.inf
        for k in self.keypoints.values():
            if k.kind not in kinds:
                continue
            d = math.hypot(k.x - x, k.y - y)
            if d < bd:
                best, bd = k.name, d
        return (best, bd) if bd <= max_d else (None, bd)

    def edge_length(self, a: str, b: str) -> float | None:
        for e in self.edges:
            if (e[0] == a and e[1] == b) or (e[0] == b and e[1] == a):
                return float(e[2])
        return None

    # ------------------------------------------------------------------ the THOR-shaped map
    def lookup_keypoints(self) -> dict[str, Any]:
        if self._lookup is not None:
            return self._lookup
        kps: dict[str, Any] = {}
        for k in self.keypoints.values():
            mx, mz = coords.to_map_xz(k.x, k.y)
            kps[k.name] = {"desc": k.desc, "xy": [round(mx, 3), round(mz, 3)], "room": k.room,
                           "yaw": round(coords.yaw_map_deg(k.yaw), 1), "kind": k.kind}
        surfaces = {}
        for s in self.surfaces.values():
            mx, mz = coords.to_map_xz(*s.center)
            surfaces[s.name] = {"keypoints": [s.name], "desc": f"the {s.desc}", "height_m": s.height,
                                "xy": [round(mx, 2), round(mz, 2)], "room": s.room,
                                "half": [round(s.half[0], 3), round(s.half[1], 3)],
                                "furniture": s.furniture_id, "type": vocab.surface_base(s.thor_type)}
        people = {}
        if self.user_surface:
            people["user"] = {"deliver_to_surface": self.user_surface, "keypoint": self.user_surface}
        rooms = {}
        for name, r in self.rooms.items():
            rooms[name] = {"label": r.label, "spots": [k for k, v in kps.items() if v.get("room") == name],
                           "center": [round(v, 2) for v in coords.to_map_xz(*r.center)],
                           "polygon": [[round(v, 3) for v in coords.to_map_xz(*p)] for p in r.polygon]}
        p = self.params
        self._lookup = {
            "scene": self.scene, "keypoints": kps, "edges": [list(e) for e in self.edges], "surfaces": surfaces,
            "people": people, "rooms": rooms, "nav_speed_mps": p.nav_speed_mps,
            "max_reach_height_m": p.max_reach_height_m, "min_reach_height_m": p.min_reach_height_m,
            "robot": "unitree_g1", "grid_step": p.grid_step, "source": self.source,
        }
        return self._lookup

    def layout_for_ui(self) -> dict[str, Any]:
        """The THOR `init.layout` shape (ui/server.py:224-240) in Worldline's frame."""
        kmap = self.lookup_keypoints()
        human = None
        if self.user_surface:
            k = self.keypoints[self.user_surface]
            human = {"x": round(k.x, 3), "z": round(k.y, 3), "keypoint": self.user_surface,
                     "deliver_to_surface": self.user_surface}
        return {
            "scene": self.scene, "topdown": dict(self.topdown),
            "keypoints": {n: {"x": v["xy"][0], "z": v["xy"][1], "yaw": v["yaw"]} for n, v in kmap["keypoints"].items()},
            "surfaces": {n: {"desc": v["desc"], "x": v["xy"][0], "z": v["xy"][1], "height": v["height_m"]}
                         for n, v in kmap["surfaces"].items()},
            "user_surface": self.user_surface, "human": human,
            "grid": self.grid.free_cells(self.params.grid_step), "grid_step": self.params.grid_step,
            "rooms": {n: {"label": r["label"], "x": r["center"][0], "z": r["center"][1], "polygon": r["polygon"]}
                      for n, r in kmap["rooms"].items()},
        }


# ====================================================================== build
def build_static_map(scene: SceneData, grid: WorldGrid, params: MapParams | None = None, *,
                     scene_key: str | None = None, source: str = "isaac-gt",
                     topdown: dict[str, Any] | None = None, user_surface: str | None = None) -> StaticMap:
    """user_surface overrides THOR's rule (the surface whose stand is nearest the start), e.g. from
    eval/scenes.yaml; an unknown name is ignored (the rule applies)."""
    p = params or MapParams()
    rooms = {r.name: r for r in scene.rooms}
    sx, sy, syaw = scene.spawn
    comp = grid.component(sx, sy)

    surfaces, alias, skipped = _layout_surfaces(scene, grid, p, comp)
    keypoints: dict[str, Keypoint] = {
        "start": Keypoint("start", sx, sy, syaw, "start", scene.room_at(sx, sy), "where the robot started, next to you")}
    for s in surfaces.values():
        keypoints[s.name] = Keypoint(s.name, s.stand[0], s.stand[1], s.yaw, "surface", s.room, f"in front of the {s.desc}")
    if p.room_keypoints:
        for r in scene.rooms:
            kp = _room_keypoint(scene, grid, r, comp)
            if kp is not None and r.name not in keypoints:
                keypoints[r.name] = Keypoint(r.name, kp[0], kp[1], kp[2], "room", r.name, f"the middle of the {r.label}")

    objects = _name_objects(scene)
    landmarks = _name_landmarks(scene, keypoints)
    edges = _edges(grid, keypoints)
    if user_surface not in surfaces:
        # THOR's rule (the surface whose stand is nearest the start); with user_surface_rule "placeable" (the G1
        # profiles, config/g1.yaml) restricted to surfaces a G1 can place on from their stand
        user_surface = None
        # (the stands added on other free sides never: the user stays where the one-stand-per-stretch map put them)
        own = [s for s in surfaces.values() if not s.extra_side]
        good = [s for s in own if not s.relaxed and s.stand_off_m <= p.user_max_stand_off_m + 1e-6
                and p.min_reach_height_m <= s.height <= p.max_reach_height_m]
        pool = (good if p.user_surface_rule == "placeable" else []) or own
        if pool:
            user_surface = min(pool, key=lambda s: math.dist(s.stand, (sx, sy))).name
    return StaticMap(scene=scene_key or scene.house_id, house_id=scene.house_id, params=p, floor_z=scene.floor_z,
                     rooms=rooms, surfaces=surfaces, keypoints=keypoints, landmarks=landmarks, objects=objects,
                     things=list(scene.items), edges=edges, user_surface=user_surface, grid=grid, alias=alias,
                     skipped=skipped, topdown=topdown or _topdown(scene), source=source)


STRUCTURE_TYPES = {"Wall", "Floor", "Ceiling", "Painting", "Window", "Doorway", "Doorframe", "Door"}


def embedded_items(scene: SceneData, furniture: SceneItem) -> list[SceneItem]:
    """Built-in appliances that break through this furniture's top: not pickupable, not structure, not a surface,
    at least half of their floor area inside its AABB, rising from below the top (>5 cm) to at least the top (-3 cm)
    (K10: the stove in counter_2; a cabinet under a counter stays below the top and is not one)."""
    (x0, y0, _), (x1, y1, z1) = furniture.aabb
    out = []
    for it in scene.items:
        t = it.thor_type
        if it is furniture or vocab.is_pickupable(t) or vocab.is_surface(t) or t in STRUCTURE_TYPES \
                or t in vocab.NON_SOLID_TYPES:
            continue
        (a0, b0, c0), (a1, b1, c1) = it.aabb
        ov = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
        area = max(1e-6, (a1 - a0) * (b1 - b0))
        if ov / area >= 0.5 and c0 < z1 - 0.05 and c1 >= z1 - 0.03:
            out.append(it)
    return out


LINE_FILL = 0.2                 # a grid row/column of a stretch counts as furniture if this share of it is


class Footprint:
    """The furniture's real floor footprint on the occupancy grid (obstacle cells in its AABB), minus embedded
    appliances. An edge is only ever cut over free floor: the strip it gives up must be mostly free floor inside the
    house (the inside of an L or U), never a wall, the outside, or another obstacle, so furniture against walls keeps
    its AABB stretches."""

    def __init__(self, grid: WorldGrid, furniture: SceneItem, carve: list[SceneItem]):
        d = grid.data
        self.res = d.res
        self.origin = d.origin
        (x0, y0, _), (x1, y1, _) = furniture.aabb
        h, w = d.occ.shape
        self.i0 = max(0, int(math.floor((x0 - d.origin[0]) / d.res)))
        self.i1 = min(w, int(math.ceil((x1 - d.origin[0]) / d.res)))
        self.j0 = max(0, int(math.floor((y0 - d.origin[1]) / d.res)))
        self.j1 = min(h, int(math.ceil((y1 - d.origin[1]) / d.res)))
        win = (slice(self.j0, self.j1), slice(self.i0, self.i1))
        if d.raw_codes is not None:
            m = d.raw_codes[win] == 1
            self.outside = d.raw_codes[win] == 2
        else:
            m = d.occ[win].copy()
            self.outside = np.zeros_like(m)
        self.floor = ~m & ~self.outside                  # free floor inside the house (before carving)
        self.carved = bool(carve)
        for it in carve:
            (a0, b0, _), (a1, b1, _) = it.aabb
            ci0 = max(0, int(math.floor((a0 - d.origin[0]) / d.res)) - self.i0)
            ci1 = min(m.shape[1], int(math.ceil((a1 - d.origin[0]) / d.res)) - self.i0)
            cj0 = max(0, int(math.floor((b0 - d.origin[1]) / d.res)) - self.j0)
            cj1 = min(m.shape[0], int(math.ceil((b1 - d.origin[1]) / d.res)) - self.j0)
            if ci1 > ci0 and cj1 > cj0:
                m[cj0:cj1, ci0:ci1] = False
        self.mask = m

    def runs(self, along_x: bool, *, bridge_m: float, min_len_m: float) -> list[tuple[float, float]]:
        """Intervals of the long axis (world coordinates) where the footprint has cells."""
        prof = self.mask.any(axis=0 if along_x else 1)
        base = (self.origin[0] + self.i0 * self.res) if along_x else (self.origin[1] + self.j0 * self.res)
        idx = prof.nonzero()[0]
        if not len(idx):
            return []
        runs: list[list[int]] = [[int(idx[0]), int(idx[0])]]
        gap = int(round(bridge_m / self.res))
        for k in idx[1:]:
            if k - runs[-1][1] - 1 <= gap:
                runs[-1][1] = int(k)
            else:
                runs.append([int(k), int(k)])
        out = [(base + a * self.res, base + (b + 1) * self.res) for a, b in runs]
        return [r for r in out if r[1] - r[0] >= min_len_m]

    def _window(self, center: tuple[float, float], half: tuple[float, float]):
        """Index ranges (in the mask) of the cells whose centre lies in the box, or None."""
        ox, oy, r = self.origin[0], self.origin[1], self.res
        ci0 = max(0, int(math.ceil((center[0] - half[0] - ox) / r - 0.5)) - self.i0)
        ci1 = min(self.mask.shape[1], int(math.floor((center[0] + half[0] - ox) / r - 0.5)) + 1 - self.i0)
        cj0 = max(0, int(math.ceil((center[1] - half[1] - oy) / r - 0.5)) - self.j0)
        cj1 = min(self.mask.shape[0], int(math.floor((center[1] + half[1] - oy) / r - 0.5)) + 1 - self.j0)
        if ci1 <= ci0 or cj1 <= cj0:
            return None
        return ci0, ci1, cj0, cj1

    def fill(self, center: tuple[float, float], half: tuple[float, float]) -> float:
        """Footprint share of the box's cells inside the house."""
        w = self._window(center, half)
        if w is None:
            return 1.0
        ci0, ci1, cj0, cj1 = w
        inside = ~self.outside[cj0:cj1, ci0:ci1]
        n = int(inside.sum())
        return float((self.mask[cj0:cj1, ci0:ci1] & inside).sum()) / n if n else 1.0

    def cut(self, center: tuple[float, float], half: tuple[float, float], min_cut: float
            ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        """The segment box cut to the footprint cells inside it: (center, half, fill), or None if nothing is left.
        An edge moves only if it moves by more than `min_cut` over floor inside the house."""
        w = self._window(center, half)
        if w is None:
            return center, half, 1.0
        ci0, ci1, cj0, cj1 = w
        sub = self.mask[cj0:cj1, ci0:ci1]
        flo = self.floor[cj0:cj1, ci0:ci1]
        fill = self.fill(center, half)
        if not sub.any():
            return None
        # rows / columns that are really furniture: at least LINE_FILL of them (a wall line along the box's edge,
        # one or two cells thick, is not)
        rows = (sub.mean(axis=1) >= LINE_FILL).nonzero()[0]
        cols = (sub.mean(axis=0) >= LINE_FILL).nonzero()[0]
        if not len(rows) or not len(cols):
            return center, half, fill
        ox, oy, r = self.origin[0], self.origin[1], self.res
        bx0, bx1 = center[0] - half[0], center[0] + half[0]
        by0, by1 = center[1] - half[1], center[1] + half[1]

        def trim(old: float, new: float, strip) -> float:
            if abs(new - old) <= min_cut or strip.size == 0 or float(strip.mean()) < 0.5:
                return old                               # a small move, or not over free floor
            return new
        nx0 = trim(bx0, ox + (self.i0 + ci0 + cols[0]) * r, flo[:, :cols[0]])
        nx1 = trim(bx1, ox + (self.i0 + ci0 + cols[-1] + 1) * r, flo[:, cols[-1] + 1:])
        ny0 = trim(by0, oy + (self.j0 + cj0 + rows[0]) * r, flo[:rows[0], :])
        ny1 = trim(by1, oy + (self.j0 + cj0 + rows[-1] + 1) * r, flo[rows[-1] + 1:, :])
        return ((nx0 + nx1) / 2, (ny0 + ny1) / 2), ((nx1 - nx0) / 2, (ny1 - ny0) / 2), fill


def _segments(o: SceneItem, fp: Footprint | None, p: MapParams, along_x: bool
              ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The furniture's stretches along its long axis, as (centre, half) boxes before the footprint cut.

    THOR's rule: n = ceil(length / segment_m) equal parts of the AABB. When a built-in appliance cuts the footprint,
    the long axis is first split into runs where the furniture really is (the stove leaves a gap; gaps up to 0.10 m
    are bridged, runs shorter than 0.25 m dropped) and each run is split by the same rule; one run covering the AABB
    gives exactly THOR's parts."""
    (x0, y0, _), (x1, y1, _) = o.aabb
    lo, hi = (x0, x1) if along_x else (y0, y1)
    runs = [(lo, hi)]
    if fp is not None and fp.carved:
        got = fp.runs(along_x, bridge_m=0.10, min_len_m=0.25)
        if got and not (len(got) == 1 and got[0][0] - lo <= p.footprint_min_cut_m
                        and hi - got[0][1] <= p.footprint_min_cut_m):
            runs = got
    out = []
    for r0, r1 in runs:
        n = max(1, math.ceil((r1 - r0) / p.segment_m - 1e-9))
        for i in range(n):
            a, b = r0 + (r1 - r0) * i / n, r0 + (r1 - r0) * (i + 1) / n
            if along_x:
                out.append((((a + b) / 2, (y0 + y1) / 2), ((b - a) / 2, (y1 - y0) / 2)))
            else:
                out.append((((x0 + x1) / 2, (a + b) / 2), ((x1 - x0) / 2, (b - a) / 2)))
    return out


def _layout_surfaces(scene: SceneData, grid: WorldGrid, p: MapParams, comp: int | None):
    counts: dict[tuple[str | None, str], int] = collections.Counter()
    used: list[tuple[float, float, float]] = []
    skipped: list[tuple[str, str]] = []
    alias: dict[str, str] = {}
    surf = [o for o in scene.items if vocab.is_surface(o.thor_type)]
    # stacked shelf levels: keep the lowest as the surface, fold the rest into it
    primary: dict[str, str] = {}
    for o in sorted(surf, key=lambda o: o.center[2]):
        below = next((q for q in primary.values()
                      if scene.item(q).thor_type == o.thor_type                      # type: ignore[union-attr]
                      and math.dist(scene.item(q).center[:2], o.center[:2]) < p.shelf_merge_m), None)   # type: ignore[union-attr]
        primary[o.scene_id] = below or o.scene_id
    out: dict[str, Surface] = {}
    second: list[tuple[dict, dict[int, str]]] = []  # free-standing furniture for the free-sides pass
    for o in sorted(surf, key=lambda o: o.scene_id):
        if primary[o.scene_id] != o.scene_id:
            continue
        kind = vocab.SURFACE_TYPES[o.thor_type]
        (x0, y0, z0), (x1, y1, z1) = o.aabb
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ex, ey = x1 - x0, y1 - y0
        along_x = ex >= ey
        fp = Footprint(grid, o, embedded_items(scene, o)) if p.footprints else None
        segs = _segments(o, fp, p, along_x)
        n = len(segs)
        room = scene.room_at(cx, cy) if scene.rooms else None
        counts[(room, kind)] += 1
        num = counts[(room, kind)]
        base = f"{vocab.surface_base(o.thor_type)}_{num}"
        prefix = p.room_prefix if p.room_prefix is not None else scene.kind != "ithor"
        if room and prefix:
            base = f"{room}_{base}"
        # the stretches after the footprint cut: (box, half, stand axis, end index, end count, split). Only a
        # furniture whose footprint fills little of its AABB (an L or U: H40's counter 0.60, H15's 0.66) has legs
        shaped = fp is not None and fp.fill((cx, cy), (ex / 2, ey / 2)) < p.split_shaped_fill
        boxes: list[tuple] = []
        for i, (seg, half) in enumerate(segs):
            if fp is not None:
                got = fp.cut(seg, half, p.footprint_min_cut_m)
                if got is None or got[2] < p.footprint_min_fill:
                    skipped.append((o.scene_id + (f"#{LETTERS[i]}" if n > 1 else ""),
                                    "no footprint left in this stretch (built-in appliance)"))
                    continue
                seg, half, _ = got
            boxes += _split_deep(seg, half, along_x, i, n, p) if shaped else [(seg, half, along_x, i, n, False)]
        boxes = boxes[:len(LETTERS)]
        n_parts = len(boxes)
        sibling_side = None
        taken: dict[int, str] = {}                  # box index -> the side its stand is on
        where = f" in the {scene.room(room).label}" if room and scene.room(room) else ""   # type: ignore[union-attr]
        ctx = dict(o=o, boxes=boxes, base=base, kind=kind, num=num, room=room, where=where, z0=z0, z1=z1)
        for j, (seg, half, ax, i, n_i, deep) in enumerate(boxes):
            st = _find_stand(scene, grid, p, seg, half, ax, i, n_i, room, used, comp,
                             None if deep else sibling_side)
            if st is None:
                skipped.append((o.scene_id + (f"#{LETTERS[j]}" if n_parts > 1 else ""), "no free stand in the room"))
                continue
            if not deep:
                sibling_side = sibling_side or st[2]
            taken[j] = st[2]
            _add_surface(out, used, scene, ctx, j, "" if n_parts == 1 else LETTERS[j], st, False)
        if p.free_sides and o.thor_type in p.free_standing_types and not shaped:
            second.append((ctx, taken))
    # free-standing furniture (a table or an island with free floor on both long sides): a stand on every free side.
    # A second pass, after every furniture's stands above, so those are exactly what they were (names, letters and
    # poses: eval bindings, memory and scenes.yaml keep working) and the new ones yield to them (spacing); they get
    # the letters after the furniture's own. No sibling-side preference: each side is searched on its own, with every
    # other stand rule (regular stand-off and clearance, facing, room, spawn component, spacing).
    for ctx, taken in second:
        boxes = ctx["boxes"]
        for k, (j, st) in enumerate(_free_side_stands(scene, grid, p, boxes, taken, ctx["room"], used, comp)):
            if len(boxes) + k >= len(EXTRA_LETTERS):
                break
            _add_surface(out, used, scene, ctx, j, EXTRA_LETTERS[len(boxes) + k], st, True)
    for sid, first in primary.items():
        if sid != first:
            home = next((s.name for s in out.values() if s.furniture_id == first), None)
            if home:
                alias[sid] = home
    return out, alias, skipped


def _add_surface(out: dict[str, Surface], used: list, scene: SceneData, ctx: dict, j: int, part: str, st,
                 extra: bool) -> None:
    """One stand of stretch `j` of a furniture -> a Surface named base + part (and its spot in `used`)."""
    (px, py), yaw, side, d, relaxed = st
    o, seg, half = ctx["o"], ctx["boxes"][j][0], ctx["boxes"][j][1]
    used.append((px, py, yaw))
    name = ctx["base"] + part
    desc = f"{ctx['kind']} {ctx['num']}{', part ' + part if part else ''}{ctx['where']}"
    out[name] = Surface(name=name, furniture_id=o.scene_id, thor_type=o.thor_type, kind=ctx["kind"], num=ctx["num"],
                        part=part, room=ctx["room"], desc=desc, center=seg, half=half, z_top=ctx["z1"],
                        z_bottom=ctx["z0"], height=round(ctx["z1"] - scene.floor_z, 2), stand=(px, py), yaw=yaw,
                        side=side, stand_off_m=round(d, 3), relaxed=relaxed, extra_side=extra)


def _split_deep(seg, half, along_x: bool, i: int, n: int, p: MapParams) -> list[tuple]:
    """A stretch whose footprint runs deeper (across the furniture's long axis) than `split_deep_m` and than it is
    wide is a leg of an L- or U-shaped top: a stand at its narrow end would face up to ~2 m of top that no arm
    reaches (H40 kitchen_counter_1b, H15 kitchen_counter_1a/1c: 0.82-0.92 m wide, 1.83 m deep). It is split along
    its depth into parts of at most segment_m, each with stands on its own long sides (the inside of the L or U), the
    ends only at the ends of the leg. Other stretches pass unchanged: (box, half, stand axis, end index, end count,
    split)."""
    hx, hy = half
    depth, width = (2 * hy, 2 * hx) if along_x else (2 * hx, 2 * hy)
    if p.split_deep_m <= 0 or depth <= p.split_deep_m or depth <= width:
        return [(seg, half, along_x, i, n, False)]
    k = max(2, math.ceil(depth / p.segment_m - 1e-9))
    out = []
    for m in range(k):
        if along_x:                               # the leg runs along y: parts stacked in y, stands on the +-x sides
            y0 = seg[1] - hy + 2 * hy * m / k
            y1 = seg[1] - hy + 2 * hy * (m + 1) / k
            out.append(((seg[0], (y0 + y1) / 2), (hx, (y1 - y0) / 2), False, m, k, True))
        else:
            x0 = seg[0] - hx + 2 * hx * m / k
            x1 = seg[0] - hx + 2 * hx * (m + 1) / k
            out.append((((x0 + x1) / 2, seg[1]), ((x1 - x0) / 2, hy), True, m, k, True))
    return out


def _side_name(nx: float, ny: float) -> str:
    return ("+x" if nx > 0 else "-x") if nx else ("+y" if ny > 0 else "-y")


def _faces(along_x: bool, i: int, n: int) -> tuple[list[str], list[str]]:
    """A stretch's candidate faces: (the long sides, the ends it has: only at the ends of the furniture)."""
    if along_x:
        return ["+y", "-y"], (["-x"] if i == 0 else []) + (["+x"] if i == n - 1 else [])
    return ["+x", "-x"], (["-y"] if i == 0 else []) + (["+y"] if i == n - 1 else [])


def _free_side_stands(scene: SceneData, grid: WorldGrid, p: MapParams, boxes: list[tuple], taken: dict[int, str],
                      room: str | None, used, comp) -> list[tuple[int, tuple]]:
    """Stands on the other free sides of a free-standing furniture: [(box index, stand)] in box order, long sides
    before ends.

    A side is FREE when a regular stand fits in front of it (stand_off_m from the edge, stand_clearance_m from every
    obstacle, within the stand-off range, facing it, in the room, on the spawn's component: the strip in front of it
    has free floor), judged
    without the other stands. The furniture is free-standing when BOTH its long sides are free for some stretch; a
    run against a wall (its back is not free) keeps its one stand per stretch. Then every stretch gets a stand on
    each free side its stand is not on already (the ends only at the ends of the furniture), >= stand_min_sep_m from
    every other stand (the stands are placed in this order, so a later one yields). No relaxed stands here: a side
    whose strip is blocked (chairs pushed in) is not free."""
    parts = [(j, b) for j, b in enumerate(boxes) if not b[5]]          # deep (split) legs are not table sides
    if not parts:
        return []
    free: dict[tuple[int, str], bool] = {}
    for j, (seg, half, ax, i, n_i, _) in parts:
        longs, ends = _faces(ax, i, n_i)
        for f in longs + ends:
            free[(j, f)] = _find_stand(scene, grid, p, seg, half, ax, i, n_i, room, [], comp,
                                       only_sides=(f,), relaxed_ok=False, strict=True) is not None
    longs0 = _faces(parts[0][1][2], 0, 1)[0]
    if not all(any(free.get((j, f)) for j, _ in parts) for f in longs0):
        return []
    out: list[tuple[int, tuple]] = []
    mine = list(used)
    for j, (seg, half, ax, i, n_i, _) in parts:
        longs, ends = _faces(ax, i, n_i)
        for f in longs + ends:
            if f == taken.get(j) or not free.get((j, f)):
                continue
            st = _find_stand(scene, grid, p, seg, half, ax, i, n_i, room, mine, comp, only_sides=(f,),
                             relaxed_ok=False, strict=True)
            if st is not None:
                out.append((j, st))
                mine.append((st[0][0], st[0][1], st[1]))
    return out


def _find_stand(scene: SceneData, grid: WorldGrid, p: MapParams, seg, half, along_x: bool, i: int, n: int,
                room: str | None, used, comp, sibling_side: str | None = None, *,
                only_sides: tuple[str, ...] | None = None, relaxed_ok: bool = True, strict: bool = False):
    """Best stand for one stretch: (xy, yaw, side, stand_off, relaxed) or None. only_sides: search just these faces;
    relaxed_ok=False: no relaxed pass; strict: no stand-off past the range's end (the 5 cm steps from its start can
    overshoot it by up to a step: 0.27 + 4 x 0.05 = 0.47 > 0.45; kept for the one-stand-per-stretch pass, whose
    stands must not move)."""
    hx, hy = half
    # candidate faces: the long sides always; the short ends only at the ends of the furniture
    if along_x:
        sides = [((0, 1), False), ((0, -1), False)]
        if i == 0:
            sides.append(((-1, 0), True))
        if i == n - 1:
            sides.append(((1, 0), True))
    else:
        sides = [((1, 0), False), ((-1, 0), False)]
        if i == 0:
            sides.append(((0, -1), True))
        if i == n - 1:
            sides.append(((0, 1), True))
    square = abs(2 * hx - 2 * hy) < 0.1 * max(2 * hx, 2 * hy, 1e-6) and n == 1
    lo, hi = p.stand_off_m
    if only_sides is not None:
        sides = [sd for sd in sides if _side_name(*sd[0]) in only_sides]
    passes = [(lo, hi, p.stand_clearance_m, False), (lo, p.stand_off_relaxed_m, p.relaxed_clearance_m, True)]
    if not relaxed_ok:
        passes = passes[:1]
    max_err = math.radians(p.max_facing_err_deg)
    for d_lo, d_hi, clr, relaxed in passes:
        best, best_score = None, math.inf
        n_d = int(round((d_hi - d_lo) / p.stand_step_m)) + 1
        for (nx, ny), is_end in sides:
            ex_pt = (seg[0] + nx * hx, seg[1] + ny * hy)
            tx, ty = -ny, nx                                     # lateral axis along the edge
            hlat = hy if nx else hx
            depth = hx if nx else hy
            lat_max = min(p.lateral_max_m, hlat)
            lats = [0.0]
            k = 1
            while k * p.lateral_step_m <= lat_max + 1e-9:
                lats += [k * p.lateral_step_m, -k * p.lateral_step_m]
                k += 1
            penalty = 0.0 if (square or not is_end) else p.end_side_penalty
            side_name = _side_name(nx, ny)
            if sibling_side is not None and side_name != sibling_side:
                penalty += p.sibling_side_penalty
            for j in range(n_d):
                d = d_lo + j * p.stand_step_m
                if strict and d > d_hi + 1e-9:
                    continue
                for lat in lats:
                    if math.atan2(abs(lat), d + depth) > max_err:
                        continue
                    score = (d - lo) + 0.5 * abs(lat) + penalty
                    if score - p.open_side_weight >= best_score:
                        continue
                    px, py = ex_pt[0] + nx * d + tx * lat, ex_pt[1] + ny * d + ty * lat
                    c_here = grid.clearance(px, py)
                    if c_here < clr or not grid.is_free(px, py):
                        continue
                    # prefer the open side (a stand in a narrow corridor blocks the furniture across it)
                    score -= p.open_side_weight * min(c_here, 1.0)
                    if score >= best_score:
                        continue
                    if room and scene.room_at(px, py) != room:
                        continue
                    yaw_c = math.atan2(-ny, -nx)
                    if any(math.dist((px, py), u[:2]) < (p.stand_min_sep_opposed_m
                                                         if abs(coords.ang_diff(yaw_c, u[2])) > math.pi / 2
                                                         else p.stand_min_sep_m) for u in used):
                        continue
                    if comp is not None and grid.component(px, py) != comp:
                        continue
                    best, best_score = ((px, py), math.atan2(-ny, -nx), side_name, d, relaxed), score
        if best is not None:
            return best
    return None


def _room_keypoint(scene: SceneData, grid: WorldGrid, r: Room, comp) -> tuple[float, float, float] | None:
    rp = scene.room_points.get(r.name)
    if rp is not None and grid.is_free(rp[0], rp[1]) and (comp is None or grid.component(rp[0], rp[1]) == comp):
        return rp
    best = grid.best_free_in(lambda x, y: r.contains(x, y), component=comp)
    if best is None:
        return None
    return best[0], best[1], 0.0


def _name_objects(scene: SceneData) -> dict[str, ObjectSpec]:
    counts: dict[str, int] = collections.Counter()
    out = {}
    for o in sorted(scene.items, key=lambda o: o.scene_id):
        if not vocab.is_pickupable(o.thor_type):
            continue
        t = vocab.snake(o.thor_type)
        counts[t] += 1
        oid = f"{t}_{counts[t]}"
        out[oid] = ObjectSpec(oid=oid, scene_id=o.scene_id, thor_type=o.thor_type, type=t,
                              label=vocab.words(o.thor_type), room=o.room or scene.room_at(*o.center[:2]),
                              pos=o.center, aabb=o.aabb)
    return out


def _name_landmarks(scene: SceneData, keypoints: dict[str, Keypoint]) -> dict[str, Landmark]:
    groups: dict[str, list[list[SceneItem]]] = collections.defaultdict(list)
    for o in sorted(scene.items, key=lambda o: o.scene_id):
        t = o.thor_type
        if not vocab.is_landmark(t):
            continue
        if t in vocab.GROUPED_LANDMARKS and groups[t]:
            groups[t][0].append(o)
        else:
            groups[t].append([o])
    out = {}
    for t, members in groups.items():
        for i, group in enumerate(members, 1):
            cx = sum(o.center[0] for o in group) / len(group)
            cy = sum(o.center[1] for o in group) / len(group)
            cz = sum(o.center[2] for o in group) / len(group)
            lo = tuple(min(o.aabb[0][k] for o in group) for k in range(3))
            hi = tuple(max(o.aabb[1][k] for o in group) for k in range(3))
            room = scene.room_at(cx, cy)
            same = [k for k in keypoints.values() if k.room == room and k.kind in ("surface", "start")] \
                or [k for k in keypoints.values() if k.room == room] or list(keypoints.values())
            # the nearest stand that faces it (within 90 deg), else the nearest: a stand across a galley that faces
            # the other counter is no place to look at a stove from
            near = min(same, key=lambda k: (abs(coords.ang_diff(math.atan2(cy - k.y, cx - k.x), k.yaw)) > math.pi / 2
                                            and k.kind == "surface", math.hypot(k.x - cx, k.y - cy))).name \
                if same else None
            name = f"{vocab.landmark_base(t)}_{i}"
            out[name] = Landmark(name=name, type=vocab.landmark_base(t), label=vocab.LANDMARK_TYPES[t], thor_type=t,
                                 scene_ids=tuple(o.scene_id for o in group), center=(cx, cy, cz),
                                 aabb=(lo, hi), near=near, room=room, container=vocab.is_container(t))   # type: ignore[arg-type]
    return out


def _edges(grid: WorldGrid, keypoints: dict[str, Keypoint]) -> list[list[Any]]:
    names = list(keypoints)
    pts = [(keypoints[n].x, keypoints[n].y) for n in names]
    D = grid.distances(pts)
    edges = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            d = float(D[i, j])
            if math.isfinite(d):
                edges.append([a, names[j], round(d, 2)])
    return edges


def _topdown(scene: SceneData, w: int = 640, h: int = 480, margin: float = 0.5) -> dict[str, Any]:
    """THOR's topdown {cx, cz, size, w, h}: an orthographic view centred on the house whose vertical half-span is
    `size` metres (ui/index.html:548-551). No image here; the UI draws the static furniture render (§9.4)."""
    x0, y0, x1, y1 = scene.bounds
    cx, cz = coords.to_map_xz((x0 + x1) / 2, (y0 + y1) / 2)
    size = max((y1 - y0) / 2, (x1 - x0) / 2 * h / w) + margin
    return {"cx": round(cx, 3), "cz": round(cz, 3), "size": round(size, 3), "w": w, "h": h}
