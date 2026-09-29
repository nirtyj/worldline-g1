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
  objects    pickupable things (fixed vocabulary), `snake(type)_n` numbered in scene-id order: alarm_clock_1.
  landmarks  non-pickupable LANDMARK_TYPES: fridge_1, stove_1 (all burners one), tv_1; `near` = the nearest
             keypoint IN THE SAME ROOM (THOR drift fixed, PLAN §4.4).

Stand points (G1): 0.45-0.60 m from the stretch's front edge, facing the edge normal (so layout's 4-axis facing
holds), free with >= 0.30 m clearance, in the furniture's room, reachable from the spawn, >= 0.5 m from other
stands. If nothing qualifies, a relaxed pass allows up to 1.0 m and 0.25 m clearance (`stand_relaxed`); otherwise
the stretch is skipped (listed in `skipped`, like THOR).

Coordinates: everything inside StaticMap is Isaac world (x, y, yaw rad); `lookup_keypoints()` converts to
Worldline's frame through world/coords.py only.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import coords, vocab
from .nav_grid import WorldGrid
from .scene import Room, SceneData, SceneItem

LETTERS = "abcdefgh"


@dataclass(frozen=True)
class MapParams:
    segment_m: float = 1.0                  # PLAN §7.1 (THOR 1.3)
    stand_off_m: tuple[float, float] = (0.45, 0.60)
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
    room_keypoints: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "MapParams":
        d = dict(d or {})
        kw = {}
        for f in cls.__dataclass_fields__:
            if f in d:
                v = d[f]
                kw[f] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)


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
        user_surface = None
        if surfaces:
            user_surface = min(surfaces.values(), key=lambda s: math.dist(s.stand, (sx, sy))).name
    return StaticMap(scene=scene_key or scene.house_id, house_id=scene.house_id, params=p, floor_z=scene.floor_z,
                     rooms=rooms, surfaces=surfaces, keypoints=keypoints, landmarks=landmarks, objects=objects,
                     things=list(scene.items), edges=edges, user_surface=user_surface, grid=grid, alias=alias,
                     skipped=skipped, topdown=topdown or _topdown(scene), source=source)


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
    for o in sorted(surf, key=lambda o: o.scene_id):
        if primary[o.scene_id] != o.scene_id:
            continue
        kind = vocab.SURFACE_TYPES[o.thor_type]
        (x0, y0, z0), (x1, y1, z1) = o.aabb
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ex, ey = x1 - x0, y1 - y0
        n = max(1, math.ceil(max(ex, ey) / p.segment_m - 1e-9))
        along_x = ex >= ey
        room = scene.room_at(cx, cy) if scene.rooms else None
        counts[(room, kind)] += 1
        num = counts[(room, kind)]
        base = f"{vocab.surface_base(o.thor_type)}_{num}"
        prefix = p.room_prefix if p.room_prefix is not None else scene.kind != "ithor"
        if room and prefix:
            base = f"{room}_{base}"
        sibling_side = None
        for i in range(n):
            f = (i + 0.5) / n - 0.5
            seg = (cx + f * ex, cy) if along_x else (cx, cy + f * ey)
            half = (ex / n / 2, ey / 2) if along_x else (ex / 2, ey / n / 2)
            st = _find_stand(scene, grid, p, seg, half, along_x, i, n, room, used, comp, sibling_side)
            if st is None:
                skipped.append((o.scene_id + (f"#{LETTERS[i]}" if n > 1 else ""), "no free stand in the room"))
                continue
            (px, py), yaw, side, d, relaxed = st
            sibling_side = sibling_side or side
            used.append((px, py, yaw))
            part = "" if n == 1 else LETTERS[i]
            name = base + part
            where = f" in the {scene.room(room).label}" if room and scene.room(room) else ""   # type: ignore[union-attr]
            desc = f"{kind} {num}{', part ' + part if part else ''}{where}"
            out[name] = Surface(name=name, furniture_id=o.scene_id, thor_type=o.thor_type, kind=kind, num=num,
                                part=part, room=room, desc=desc, center=seg, half=half, z_top=z1, z_bottom=z0,
                                height=round(z1 - scene.floor_z, 2), stand=(px, py), yaw=yaw, side=side,
                                stand_off_m=round(d, 3), relaxed=relaxed)
    for sid, first in primary.items():
        if sid != first:
            home = next((s.name for s in out.values() if s.furniture_id == first), None)
            if home:
                alias[sid] = home
    return out, alias, skipped


def _find_stand(scene: SceneData, grid: WorldGrid, p: MapParams, seg, half, along_x: bool, i: int, n: int,
                room: str | None, used, comp, sibling_side: str | None = None):
    """Best stand for one stretch: (xy, yaw, side, stand_off, relaxed) or None."""
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
    passes = [(lo, hi, p.stand_clearance_m, False), (lo, p.stand_off_relaxed_m, p.relaxed_clearance_m, True)]
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
            side_name = ("+x" if nx > 0 else "-x") if nx else ("+y" if ny > 0 else "-y")
            if sibling_side is not None and side_name != sibling_side:
                penalty += p.sibling_side_penalty
            for j in range(n_d):
                d = d_lo + j * p.stand_step_m
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
            near = min(same, key=lambda k: math.hypot(k.x - cx, k.y - cy)).name if same else None
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
