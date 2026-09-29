"""What the page shows as reality: the sim's ground truth, read only through the world model.

The UI is an observer. It draws the house the way the simulator knows it (occupancy,
rooms, the robot's true pose, where every object really is) next to what the runtime
believes, which is how belief and truth can be seen to disagree. Everything here comes
from `world/` (the only ground-truth reader on the runtime side, PLAN 6.2) or from the
robot facade's own map (`lookup_keypoints()`); nothing here talks to the simulator.

Frames (PLAN 6.2.3). The page keeps Worldline's frame: map x = Isaac x, map z = Isaac y,
height = Isaac z, yaw in degrees clockwise from +z. Occupancy rows are +y and columns +x,
so a cell maps straight to (x, z) with no flip; only the PNG is written top row first.

Shapes this module accepts (duck-typed, so the world agent's classes and plain dicts both
work; see `grid_view`, `truth_payload`):

  occupancy   an object or dict with `resolution`, `origin` (x0, y0: the corner of cell
              (0, 0)) and uint8 arrays `raw` (0 free, 1 obstacle, 2 outside), `inflated`
              and optionally `inflated_low` (scenes.occupancy.Occupancy has exactly these)
  truth       world.truth(): a dict (or dataclass) with `robot`, `objects`, `hands`
"""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable

from ui.png import encode_palette_png

DISPLAY_STEP = 0.25          # the page's nav-grid and explored-floor cell (THOR's GRID)
GT_LABEL = "sim ground truth: not visible to the planner"
# STEPPING STONE executors (PLAN 12.2): results they produce count as fallback passes.
STEPPING_STONES = frozenset({"kinematic_nav", "kinematic_attach", "sonic_arm_script"})
TARGET_EXECUTORS = frozenset({"sonic_walk", "groot_sonic"})
FALLBACK_LABELS = {
    "kinematic_nav": "KINEMATIC (fallback)",
    "kinematic_attach": "KINEMATIC ATTACH (fallback)",
    "sonic_arm_script": "ARM SCRIPT + ATTACH (fallback)",
}


def stepping_stones(profile: str | None = None) -> frozenset[str]:
    """STEPPING STONE executors from api/ (api.results + the profile's own list), else PLAN 12.2's list."""
    out = set(STEPPING_STONES)
    try:
        from api.results import STEPPING_STONE_EXECUTORS  # type: ignore
        out |= set(STEPPING_STONE_EXECUTORS)
        from api.types import PROFILES  # type: ignore
        p = PROFILES.get(profile or "")
        if p is not None and p.stepping_stones:
            out |= set(p.stepping_stones)
    except Exception:  # noqa: BLE001  (api/ not importable: the literal list)
        pass
    return frozenset(out)


# What each profile uses a STEPPING STONE for (PLAN 2.1, 6.7), if api/ has no list of its own.
PROFILE_FALLBACKS = {"lite": (), "bringup": ("kinematic_nav", "kinematic_attach"),
                     "sonic": ("sonic_arm_script",), "full": ("sonic_arm_script",), "real_g1": ()}


def profile_fallbacks(profile: str | None) -> tuple[str, ...]:
    """The STEPPING STONE executors this profile actually uses (for "fallbacks in this profile")."""
    try:
        from api.types import PROFILES  # type: ignore
        p = PROFILES.get(profile or "")
        if p is not None:
            return tuple(p.stepping_stones)
    except Exception:  # noqa: BLE001
        pass
    return PROFILE_FALLBACKS.get(profile or "", ())


def to_plain(obj: Any) -> Any:
    """Dataclasses, objects with to_dict(), tuples: into JSON-able dicts and lists."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_plain(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_plain(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        elif obj is not None and getattr(obj, n, None) is not None:
            return getattr(obj, n)
    return default


# ----------------------------------------------------------------------------------------------
# coordinates: world/coords.py is the one conversion point; this is only its fallback
# ----------------------------------------------------------------------------------------------
def yaw_map_deg(yaw_isaac_rad: float) -> float:
    """Isaac yaw (radians, CCW from +x) -> Worldline yaw (degrees, CW from +z)."""
    try:
        from world import coords  # type: ignore
        for name in ("yaw_isaac_to_map_deg", "yaw_to_map_deg", "isaac_yaw_to_map"):
            fn = getattr(coords, name, None)
            if callable(fn):
                return float(fn(yaw_isaac_rad))
    except Exception:  # noqa: BLE001
        pass
    return (90.0 - math.degrees(yaw_isaac_rad)) % 360.0


def yaw_from_quat_wxyz(q: Iterable[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ----------------------------------------------------------------------------------------------
# occupancy
# ----------------------------------------------------------------------------------------------
@dataclass
class GridView:
    """An occupancy grid in the page's terms. `raw[iy][ix]`: 0 free, 1 obstacle, 2 outside;
    `walk[iy][ix]`: 1 where the robot centre may not go (the nav stack's inflated layer)."""
    resolution: float
    x0: float
    y0: float
    raw: Any
    walk: Any
    source: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.raw), len(self.raw[0]) if len(self.raw) else 0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, zmin, xmax, zmax) in the Worldline frame (= Isaac xy)."""
        ny, nx = self.shape
        return self.x0, self.y0, self.x0 + nx * self.resolution, self.y0 + ny * self.resolution


def grid_view(occ: Any) -> GridView | None:
    """Adapt whatever the world model gives for its occupancy/nav grid into a GridView."""
    if occ is None:
        return None
    if isinstance(occ, GridView):
        return occ
    raw = _get(occ, "raw", "occ", "blocked")
    res = _get(occ, "resolution", "res", "cell_m")
    origin = _get(occ, "origin")
    if raw is None or res is None or origin is None:
        return None
    walk = _get(occ, "inflated_low", "occ_inflated", "inflated")
    if walk is None:
        walk = raw
    rows = raw.tolist() if hasattr(raw, "tolist") else [list(r) for r in raw]
    wrows = walk.tolist() if hasattr(walk, "tolist") else [list(r) for r in walk]
    return GridView(float(res), float(origin[0]), float(origin[1]), rows, wrows,
                    str(_get(occ, "method", "source", default="")))


def world_grid(world: Any) -> GridView | None:
    """The occupancy grid from the world model: static_map().occupancy / .nav_grid, or world.occupancy()."""
    sm = None
    fn = getattr(world, "static_map", None)
    if callable(fn):
        try:
            sm = fn()
        except Exception:  # noqa: BLE001
            sm = None
    for cand in (_get(sm, "occupancy"), _get(sm, "nav_grid", "navgrid", "grid")):
        g = grid_view(cand)
        if g is None and cand is not None:
            g = grid_view(_get(cand, "occupancy", "occ"))     # a NavGrid that wraps an Occupancy
        if g is not None:
            return g
    fn = getattr(world, "occupancy", None)
    if callable(fn):
        try:
            return grid_view(fn())
        except Exception:  # noqa: BLE001
            return None
    return None


def display_cells(grid: GridView, step: float = DISPLAY_STEP, layer: str = "walk",
                  min_free: float | None = None) -> set[tuple[int, int]]:
    """Free cells at the page's resolution, THOR-indexed: (round(x/step), round(z/step)).

    layer "walk": the inflated nav layer (where the robot centre may go: the nav grid); a
    display cell is free when at least half of the fine cells whose centres fall inside it are.
    layer "floor": raw free floor (what a camera can see across); every fine cell must be free,
    so a 10 cm wall still blocks the line of sight at 0.25 m."""
    if min_free is None:
        min_free = 0.5 if layer == "walk" else 1.0
    arr = grid.walk if layer == "walk" else grid.raw
    ny, nx = grid.shape
    counts: dict[tuple[int, int], list[int]] = {}
    r = grid.resolution
    for iy in range(ny):
        row = arr[iy]
        z = grid.y0 + (iy + 0.5) * r
        cz = round(z / step)
        for ix in range(nx):
            x = grid.x0 + (ix + 0.5) * r
            c = (round(x / step), cz)
            n = counts.get(c)
            if n is None:
                n = counts[c] = [0, 0]
            n[1] += 1
            if row[ix] == 0:
                n[0] += 1
    return {c for c, (free, tot) in counts.items() if tot and free / tot >= min_free}


# palette: 0 outside/unknown (transparent), 1 free floor, 2 obstacle, 3 free but not walkable (inflation)
OCC_PALETTE = [(0, 0, 0, 0), (38, 50, 72, 255), (200, 196, 180, 255), (58, 50, 40, 255)]


def occupancy_png(grid: GridView) -> bytes:
    """The grid as an image, top row = max z (image up = +z, as the page draws the house)."""
    ny, nx = grid.shape
    rows = []
    for iy in range(ny - 1, -1, -1):
        raw, walk = grid.raw[iy], grid.walk[iy]
        rows.append(bytes(0 if raw[ix] == 2 else 2 if raw[ix] == 1 else 3 if walk[ix] else 1 for ix in range(nx)))
    return encode_palette_png(rows, nx, OCC_PALETTE)


def topdown_from_extent(extent: Iterable[float], w: int = 640, h: int = 480) -> dict[str, float]:
    """The page's orthographic mapping (`uv()` in index.html) for a view of this extent.

    extent (xmin, zmin, xmax, zmax). The drawing is w x h units; the extent is fitted inside
    it with square pixels, so k = h / (2 * size) holds in both directions."""
    xmin, zmin, xmax, zmax = (float(v) for v in extent)
    cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2
    half_x, half_z = max((xmax - xmin) / 2, 1e-6), max((zmax - zmin) / 2, 1e-6)
    size = max(half_z, half_x * h / w)          # half the vertical span in metres
    return {"cx": round(cx, 4), "cz": round(cz, 4), "size": round(size, 4), "w": w, "h": h}


# ----------------------------------------------------------------------------------------------
# rooms, layout
# ----------------------------------------------------------------------------------------------
def world_rooms(world: Any, map_rooms: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Room name -> {label, x, z, polygon[[x, z]]} from the world model's static map,
    labels from the robot map when it has them."""
    out: dict[str, dict[str, Any]] = {}
    sm = None
    fn = getattr(world, "static_map", None)
    if callable(fn):
        try:
            sm = fn()
        except Exception:  # noqa: BLE001
            sm = None
    rooms = _get(sm, "rooms", default=None)
    items: list[tuple[str, Any]] = []
    if isinstance(rooms, dict):
        items = list(rooms.items())
    elif isinstance(rooms, (list, tuple)):
        items = [(str(_get(r, "name", "id", default=i)), r) for i, r in enumerate(rooms)]
    for name, r in items:
        poly = [[float(p[0]), float(p[1])] for p in (_get(r, "polygon", default=[]) or [])]
        c = _get(r, "center", "centre")
        if c is None and poly:
            c = [sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)]
        c = c or [0.0, 0.0]
        label = _get(r, "label", default=None) or ((map_rooms or {}).get(name) or {}).get("label") or name.replace("_", " ")
        out[name] = {"label": str(label), "x": round(float(c[0]), 3), "z": round(float(c[1]), 3), "polygon": poly}
    for name, r in (map_rooms or {}).items():         # rooms the robot map names but the world didn't draw
        out.setdefault(name, {"label": str(r.get("label") or name), "x": 0.0, "z": 0.0, "polygon": []})
    return out


def layout_message(scene: str, map_: dict[str, Any], world: Any, grid: GridView | None,
                   *, profile: str = "", top_extent: Iterable[float] | None = None) -> dict[str, Any]:
    """The init message's `layout`: the same keys the THOR page used, now built from the
    robot map (keypoints, surfaces, people) and the world model (rooms, occupancy)."""
    kps: dict[str, dict[str, float]] = {}
    for name, k in (map_.get("keypoints") or {}).items():
        xy = k.get("xy") or [k.get("x", 0.0), k.get("z", 0.0)]
        kps[name] = {"x": float(xy[0]), "z": float(xy[1]), "yaw": float(k.get("yaw") or 0.0)}
    surfaces: dict[str, dict[str, Any]] = {}
    for name, s in (map_.get("surfaces") or {}).items():
        xy = s.get("xy")
        if xy is None:
            stand = kps.get((s.get("keypoints") or [name])[0]) or {"x": 0.0, "z": 0.0}
            xy = [stand["x"], stand["z"]]
        surfaces[name] = {"desc": s.get("desc") or name.replace("_", " "), "x": float(xy[0]), "z": float(xy[1]),
                          "height": s.get("height_m", s.get("height"))}
    user = ((map_.get("people") or {}).get("user") or {})
    user_surface = user.get("deliver_to_surface")
    human = None
    ukp = user.get("keypoint") or user_surface
    if ukp and ukp in kps:
        human = {"x": kps[ukp]["x"], "z": kps[ukp]["z"], "keypoint": ukp, "deliver_to_surface": user_surface}
    rooms = world_rooms(world, map_.get("rooms"))
    grid_cells: list[list[int]] = []
    occupancy = None
    if grid is not None:
        grid_cells = sorted([list(c) for c in display_cells(grid, DISPLAY_STEP, "walk")])
        xmin, zmin, xmax, zmax = grid.extent
        occupancy = {"png": base64.b64encode(occupancy_png(grid)).decode(),
                     "extent": [round(xmin, 4), round(zmin, 4), round(xmax, 4), round(zmax, 4)],
                     "resolution": grid.resolution, "source": grid.source,
                     "legend": {"free": "free floor", "obstacle": "collider 0.10-1.60 m",
                                "inflated": "free, but too close for the robot centre", "outside": "outside every room"}}
    ext = list(top_extent) if top_extent else None
    if ext is None and grid is not None:
        ext = list(grid.extent)
    if ext is None:
        pts = [(k["x"], k["z"]) for k in kps.values()] + [tuple(p) for r in rooms.values() for p in r["polygon"]]
        if pts:
            xs, zs = [p[0] for p in pts], [p[1] for p in pts]
            ext = [min(xs) - 1, min(zs) - 1, max(xs) + 1, max(zs) + 1]
        else:
            ext = [-5.0, -5.0, 5.0, 5.0]
    return {
        "scene": scene, "profile": profile, "topdown": topdown_from_extent(ext),
        "keypoints": kps, "surfaces": surfaces, "user_surface": user_surface, "human": human,
        "grid": grid_cells, "grid_step": DISPLAY_STEP, "rooms": rooms, "occupancy": occupancy,
        "truth_label": GT_LABEL,
    }


# ----------------------------------------------------------------------------------------------
# truth for the frame message
# ----------------------------------------------------------------------------------------------
def _obj_entry(oid: str, o: Any) -> dict[str, Any]:
    o = to_plain(o) if not isinstance(o, dict) else o
    typ = _get(o, "type", "category", default="")
    x = _get(o, "x")
    z = _get(o, "z")
    h = _get(o, "y", "height", "h")
    pos = _get(o, "pos")
    if x is None and isinstance(pos, (list, tuple)) and len(pos) >= 2:     # LookData convention [x, z, h]
        x, z = pos[0], pos[1]
        h = pos[2] if len(pos) > 2 and h is None else h
    pose = _get(o, "pose")
    if x is None and isinstance(pose, dict):                              # Pose3D: Isaac frame (x, y, z-up)
        x, z, h = pose.get("x"), pose.get("y"), pose.get("z")
    where = _get(o, "where", default=None)
    held_by = _get(o, "held_by")
    if held_by and not (where or "").startswith("hand"):
        where = f"hand:{held_by}"
    return {"type": str(typ), "label": str(_get(o, "label", default=str(typ).replace("_", " "))),
            "where": where, "x": None if x is None else round(float(x), 3),
            "y": None if h is None else round(float(h), 3), "z": None if z is None else round(float(z), 3),
            "visible": bool(_get(o, "visible", default=False))}


def truth_payload(truth: Any, base: dict[str, Any] | None = None,
                  source: str = "isaac-gt") -> dict[str, Any]:
    """world.truth() (+ the robot facade's base_state) as the frame's `truth` block."""
    t = to_plain(truth) if truth is not None else {}
    if not isinstance(t, dict):
        t = {}
    base = base or {}
    r = t.get("robot") or {}
    x, z = r.get("x"), r.get("z")
    yaw = r.get("yaw")
    if x is None and isinstance(r.get("pose"), dict):                     # Isaac-frame pose in the snapshot
        p = r["pose"]
        x, z = p.get("x"), p.get("y")
        if all(k in p for k in ("qw", "qx", "qy", "qz")):
            yaw = yaw_map_deg(yaw_from_quat_wxyz((p["qw"], p["qx"], p["qy"], p["qz"])))
    if x is None and base.get("xy"):
        x, z = base["xy"][0], base["xy"][1]
    robot = {
        "x": float(x or 0.0), "z": float(z or 0.0), "yaw": float(yaw or 0.0),
        "horizon": float(r.get("horizon") or r.get("tilt") or 0.0),
        "at": None if base.get("moving") else base.get("at", r.get("at")),
        "moving": bool(base.get("moving", r.get("moving", False))),
        "between": base.get("between", r.get("between")),
        "pelvis_z": r.get("pelvis_z"), "upright": r.get("upright"), "fallen": r.get("fallen"),
        "mode": r.get("mode"),
    }
    objs_in = t.get("objects") or {}
    if isinstance(objs_in, list):
        objs_in = {str(_get(o, "id", default=i)): o for i, o in enumerate(objs_in)}
    objects = {oid: _obj_entry(oid, o) for oid, o in objs_in.items()}
    hands = t.get("hands") or {}
    arms_in = t.get("arms") or {}
    arms = {}
    for arm in ("left", "right"):
        a = arms_in.get(arm) or {}
        holding = a.get("holding", hands.get(arm))
        if holding is None:
            holding = next((oid for oid, o in objects.items() if o["where"] == f"hand:{arm}"), None)
        arms[arm] = {"holding": holding, "phase": a.get("phase") or ("holding" if holding else "free")}
    return {"source": t.get("source") or source, "label": GT_LABEL, "robot": robot, "arms": arms,
            "objects": objects, "t": t.get("t") or t.get("t_sim")}
