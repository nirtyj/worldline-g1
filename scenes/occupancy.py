"""2D occupancy grids for the G1 in a MolmoSpaces house.

Two halves:
  * pure numpy/scipy (importable anywhere, e.g. by body/ PathFollower): `Occupancy`,
    `get_occupancy(house_id)`, `occupancy_summary(house_id)`, inflation, spawn/waypoint picking;
  * Isaac-side generation (needs a running Isaac Sim with the house loaded and physics
    initialised): `generate_occupancy(house, ...)`.

Grid convention (all arrays are [H, W] = [ny, nx], row index = y, column index = x):
    world_x = origin[0] + (ix + 0.5) * resolution
    world_y = origin[1] + (iy + 0.5) * resolution
`origin` is the world XY of the lower-left CORNER of cell (0, 0). World frame = Isaac stage
frame (Z up, metres) = MolmoSpaces MJCF frame; THOR (x, y_up, z) maps to world (x, z, y)
(molmospaces molmo_spaces/housegen/utils.py:107-116 unity_to_mj_pos).

Layers saved in <HOUSE_ASSETS_ROOT>/<house_id>/occupancy.npz:
    raw       uint8  0 free, 1 obstacle (any house collider between z_min and z_max), 2 outside the house
    inflated  uint8  1 where the robot CENTRE may not go: raw != 0 or closer than robot_radius to it
    inflated_low uint8 same as `inflated` but also treating `low` cells as obstacles (stricter;
                     recommended for walking: the G1 should not step on props lying on the floor)
    dist      f32    metres from the cell centre to the nearest non-free cell (raw != 0)
    low       uint8  1 where a collider lies between z=low_z_min and z_min (trip hazards: thresholds,
                     rugs, flat mats). Informational; not in `inflated`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from scenes.catalog import HOUSE_ASSETS_ROOT, parse_house_id, point_in_polygon

FREE, OBSTACLE, OUTSIDE = 0, 1, 2

DEFAULTS = dict(resolution=0.05, z_min=0.10, z_max=1.60, robot_radius=0.30, low_z_min=0.02)


# ----------------------------------------------------------------------------------------------
# Data model (pure numpy)
# ----------------------------------------------------------------------------------------------


@dataclass
class Occupancy:
    house_id: str
    resolution: float
    origin: tuple[float, float]
    raw: np.ndarray
    inflated: np.ndarray
    dist: np.ndarray
    low: np.ndarray | None = None
    z_min: float = DEFAULTS["z_min"]
    z_max: float = DEFAULTS["z_max"]
    robot_radius: float = DEFAULTS["robot_radius"]
    method: str = ""
    meta: dict = field(default_factory=dict)
    inflated_low: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape  # (ny, nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) in world metres, for matplotlib imshow(origin='lower')."""
        ny, nx = self.raw.shape
        return (
            self.origin[0],
            self.origin[0] + nx * self.resolution,
            self.origin[1],
            self.origin[1] + ny * self.resolution,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(math.floor((x - self.origin[0]) / self.resolution))
        iy = int(math.floor((y - self.origin[1]) / self.resolution))
        return iy, ix

    def cell_to_world(self, iy: int, ix: int) -> tuple[float, float]:
        return (
            self.origin[0] + (ix + 0.5) * self.resolution,
            self.origin[1] + (iy + 0.5) * self.resolution,
        )

    def in_bounds(self, iy: int, ix: int) -> bool:
        return 0 <= iy < self.raw.shape[0] and 0 <= ix < self.raw.shape[1]

    def is_free(self, x: float, y: float, inflated: bool = True) -> bool:
        iy, ix = self.world_to_cell(x, y)
        if not self.in_bounds(iy, ix):
            return False
        return bool((self.inflated if inflated else self.raw)[iy, ix] == 0)

    def clearance(self, x: float, y: float) -> float:
        iy, ix = self.world_to_cell(x, y)
        return float(self.dist[iy, ix]) if self.in_bounds(iy, ix) else 0.0

    def free_ray(self, x: float, y: float, yaw: float, max_m: float = 6.0, margin: float | None = None) -> float:
        """Distance the robot centre can travel straight along `yaw` staying `margin` from obstacles."""
        margin = self.robot_radius if margin is None else margin
        step = self.resolution / 2
        d = 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        while d < max_m:
            nd = d + step
            if self.clearance(x + nd * c, y + nd * s) < margin:
                break
            d = nd
        return d

    # -- persistence --
    def save(self, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        npz = out_dir / "occupancy.npz"
        np.savez_compressed(
            npz,
            raw=self.raw,
            inflated=self.inflated,
            dist=self.dist.astype(np.float32),
            low=self.low if self.low is not None else np.zeros_like(self.raw),
            inflated_low=self.inflated_low if self.inflated_low is not None else self.inflated,
            resolution=np.float64(self.resolution),
            origin=np.asarray(self.origin, np.float64),
            z_band=np.asarray([self.z_min, self.z_max], np.float64),
            robot_radius=np.float64(self.robot_radius),
        )
        meta = self.summary(npz)
        (out_dir / "occupancy_meta.json").write_text(json.dumps(meta, indent=2))
        save_png(self, out_dir / "occupancy.png")
        return meta

    def summary(self, npz_path: Path | None = None) -> dict:
        ny, nx = self.raw.shape
        return {
            "house_id": self.house_id,
            "npz": str(npz_path) if npz_path else None,
            "resolution": self.resolution,
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "shape": [ny, nx],
            "extent_xy": list(self.extent),
            "z_band": [self.z_min, self.z_max],
            "robot_radius": self.robot_radius,
            "inflation": f"cells closer than robot_radius={self.robot_radius} m to an obstacle/outside cell are blocked",
            "method": self.method,
            "cell_convention": "arr[iy, ix]; world_x = origin_x + (ix+0.5)*res; world_y = origin_y + (iy+0.5)*res",
            "values": {"raw": "0 free, 1 obstacle, 2 outside", "inflated": "1 blocked for robot centre", "inflated_low": "inflated, also inflating low colliders (recommended for walking)", "low": "1 low collider (trip hazard) in [low_z_min, z_min)"},
            "counts": {
                "free": int((self.raw == FREE).sum()),
                "obstacle": int((self.raw == OBSTACLE).sum()),
                "outside": int((self.raw == OUTSIDE).sum()),
                "free_after_inflation": int((self.inflated == 0).sum()),
                "free_after_inflation_incl_low": int((self.inflated_low == 0).sum()) if self.inflated_low is not None else None,
                "low": int(self.low.sum()) if self.low is not None else 0,
            },
            "free_area_m2": float((self.inflated == 0).sum() * self.resolution**2),
            **self.meta,
        }


def load_npz(path: Path, house_id: str = "") -> Occupancy:
    z = np.load(path)
    zb = z["z_band"]
    meta_p = path.parent / "occupancy_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return Occupancy(
        house_id=house_id or meta.get("house_id", ""),
        resolution=round(float(z["resolution"]), 6),
        origin=(float(z["origin"][0]), float(z["origin"][1])),
        raw=z["raw"],
        inflated=z["inflated"],
        dist=z["dist"],
        low=z["low"] if "low" in z else None,
        inflated_low=z["inflated_low"] if "inflated_low" in z else None,
        z_min=round(float(zb[0]), 6),
        z_max=round(float(zb[1]), 6),
        robot_radius=round(float(z["robot_radius"]), 6),
        method=meta.get("method", ""),
    )


def get_occupancy(house_id: str, root: Path | None = None) -> Occupancy:
    """Load the cached grid for a house (generated by scenes.test_house / generate_occupancy)."""
    ref = parse_house_id(house_id)
    d = (root or HOUSE_ASSETS_ROOT) / ref.house_id
    p = d / "occupancy.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m scenes.test_house --house {ref.house_id}` on the box first")
    return load_npz(p, ref.house_id)


def occupancy_summary(house_id: str, root: Path | None = None) -> dict:
    """What P1's REP `get_occupancy` returns: npz path + resolution, origin, inflation."""
    ref = parse_house_id(house_id)
    d = (root or HOUSE_ASSETS_ROOT) / ref.house_id
    meta = json.loads((d / "occupancy_meta.json").read_text())
    meta["npz"] = str(d / "occupancy.npz")
    meta["png"] = str(d / "occupancy.png")
    return meta


# ----------------------------------------------------------------------------------------------
# Building blocks (pure numpy)
# ----------------------------------------------------------------------------------------------


def make_grid_frame(bounds_xy, resolution: float, pad: float = 0.3):
    """Snap padded XY bounds to the resolution. Returns origin (x0, y0), (ny, nx)."""
    (xmin, ymin), (xmax, ymax) = bounds_xy
    x0 = math.floor((xmin - pad) / resolution) * resolution
    y0 = math.floor((ymin - pad) / resolution) * resolution
    nx = int(math.ceil((xmax + pad - x0) / resolution))
    ny = int(math.ceil((ymax + pad - y0) / resolution))
    return (round(x0, 6), round(y0, 6)), (ny, nx)


def cell_centres(origin, shape, resolution):
    ny, nx = shape
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    return xs, ys


def rooms_mask(rooms: list[dict], origin, shape, resolution) -> np.ndarray:
    """bool [ny, nx]: cell centre inside any room polygon (world XY)."""
    xs, ys = cell_centres(origin, shape, resolution)
    mask = np.zeros(shape, bool)
    try:
        from matplotlib.path import Path as MPath

        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel()], 1)
        for r in rooms:
            mask |= MPath(np.asarray(r["polygon"])).contains_points(pts).reshape(shape)
    except Exception:  # slow fallback
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                mask[iy, ix] = any(point_in_polygon(r["polygon"], x, y) for r in rooms)
    return mask


def room_label_grid(rooms: list[dict], origin, shape, resolution) -> np.ndarray:
    """int16 [ny, nx]: index into `rooms` (-1 outside)."""
    xs, ys = cell_centres(origin, shape, resolution)
    lab = np.full(shape, -1, np.int16)
    from matplotlib.path import Path as MPath

    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    for i, r in enumerate(rooms):
        m = MPath(np.asarray(r["polygon"])).contains_points(pts).reshape(shape)
        lab[m & (lab < 0)] = i
    return lab


def finish(house_id, raw, origin, resolution, z_min, z_max, robot_radius, method, low=None, meta=None) -> Occupancy:
    """Inflate + distance transform."""
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(raw == FREE).astype(np.float32) * resolution
    inflated = ((raw != FREE) | (dist < robot_radius)).astype(np.uint8)
    inflated_low = inflated
    if low is not None and low.any():
        blocked = (raw != FREE) | (low > 0)
        d2 = distance_transform_edt(~blocked).astype(np.float32) * resolution
        inflated_low = (blocked | (d2 < robot_radius)).astype(np.uint8)
    return Occupancy(
        house_id=house_id,
        resolution=resolution,
        origin=tuple(origin),
        raw=raw.astype(np.uint8),
        inflated=inflated,
        dist=dist,
        low=None if low is None else low.astype(np.uint8),
        z_min=z_min,
        z_max=z_max,
        robot_radius=robot_radius,
        method=method,
        meta=meta or {},
        inflated_low=inflated_low,
    )


def best_yaw(occ: Occupancy, x: float, y: float, n: int = 16, max_m: float = 6.0) -> tuple[float, float]:
    """Heading with the longest straight free run for the robot centre. Returns (yaw, free_m)."""
    best = (0.0, -1.0)
    for k in range(n):
        yaw = -math.pi + 2 * math.pi * k / n
        d = occ.free_ray(x, y, yaw, max_m)
        if d > best[1] + 1e-6:
            best = (yaw, d)
    return best


def room_points(occ: Occupancy, rooms: list[dict]) -> list[dict]:
    """Per room: the free cell with maximum clearance (a good go_to / spawn target)."""
    lab = room_label_grid(rooms, occ.origin, occ.shape, occ.resolution)
    out = []
    for i, r in enumerate(rooms):
        m = (lab == i) & (occ.inflated == 0)
        if not m.any():
            out.append({"room": r["name"], "room_id": r["room_id"], "ok": False})
            continue
        d = np.where(m, occ.dist, -1)
        iy, ix = np.unravel_index(int(np.argmax(d)), d.shape)
        x, y = occ.cell_to_world(iy, ix)
        yaw, free = best_yaw(occ, x, y)
        out.append(
            {
                "room": r["name"],
                "room_id": r["room_id"],
                "ok": True,
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(yaw, 4),
                "clearance_m": round(float(occ.dist[iy, ix]), 3),
                "forward_free_m": round(free, 2),
                "free_cells": int(m.sum()),
            }
        )
    return out


def connectivity(occ: Occupancy, rooms: list[dict], start_xy) -> dict:
    """Which rooms are reachable for the robot centre from start_xy on the inflated grid."""
    from scipy.ndimage import label

    lab, n = label(occ.inflated == 0)
    iy, ix = occ.world_to_cell(*start_xy)
    comp = int(lab[iy, ix]) if occ.in_bounds(iy, ix) else 0
    rl = room_label_grid(rooms, occ.origin, occ.shape, occ.resolution)
    rooms_out = {}
    for i, r in enumerate(rooms):
        cells = (rl == i) & (occ.inflated == 0)
        reach = cells & (lab == comp) if comp > 0 else np.zeros_like(cells)
        rooms_out[r["name"]] = {
            "free_cells": int(cells.sum()),
            "reachable_cells": int(reach.sum()),
            "reachable": bool(reach.any()),
            "reachable_area_m2": round(float(reach.sum() * occ.resolution**2), 2),
        }
    return {
        "start": [float(start_xy[0]), float(start_xy[1])],
        "start_component": comp,
        "n_components": int(n),
        "component_area_m2": round(float((lab == comp).sum() * occ.resolution**2), 2) if comp > 0 else 0.0,
        "rooms": rooms_out,
        "all_rooms_reachable": all(v["reachable"] for v in rooms_out.values() if v["free_cells"] > 0),
    }


def choose_spawn(occ: Occupancy, rooms: list[dict], preferred: dict | None = None, min_clear: float = 0.55) -> dict:
    """Spawn pose: a free cell with >= min_clear m clearance and a long straight run ahead
    (so 'walk forward 2 m' works from the spawn). Prefer the ProcTHOR/Worldline agent start
    (`preferred` = {x, y, yaw}) when it qualifies; else the best-scoring cell in the house.
    """
    cands = []
    if preferred:
        px, py = preferred["x"], preferred["y"]
        c = occ.clearance(px, py)
        yaw, free = best_yaw(occ, px, py)
        pref_yaw = preferred.get("yaw")
        if pref_yaw is not None:
            fp = occ.free_ray(px, py, pref_yaw)
            if fp >= 2.5:
                yaw, free = pref_yaw, fp
        if c >= min_clear and free >= 2.5:
            return {"x": px, "y": py, "yaw": yaw, "clearance_m": round(c, 3), "forward_free_m": round(free, 2), "source": "procthor_agent_start"}
        cands.append({"x": px, "y": py, "clearance_m": c, "forward_free_m": free, "note": "procthor start rejected"})
    # global: sample well-cleared cells, score by clearance + forward run
    ok = (occ.inflated == 0) & (occ.dist >= min_clear)
    if not ok.any():
        ok = occ.inflated == 0
    idx = np.argwhere(ok)
    if len(idx) == 0:
        raise RuntimeError("no free cell for the robot in this house")
    stride = max(1, len(idx) // 600)
    best = None
    for iy, ix in idx[::stride]:
        x, y = occ.cell_to_world(int(iy), int(ix))
        yaw, free = best_yaw(occ, x, y)
        score = min(float(occ.dist[iy, ix]), 1.0) + 0.5 * min(free, 4.0)
        if best is None or score > best[0]:
            best = (score, x, y, yaw, float(occ.dist[iy, ix]), free)
    _, x, y, yaw, c, free = best
    room = None
    for r in rooms:
        if point_in_polygon(r["polygon"], x, y):
            room = r["name"]
    return {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4), "clearance_m": round(c, 3), "forward_free_m": round(free, 2), "room": room, "source": "max_score", "rejected": cands}


# ----------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------


def to_rgb(occ: Occupancy) -> np.ndarray:
    """RGB image (row 0 = max y, so +y is up): free white, inflation light grey, obstacle black,
    outside dark slate, low colliders orange."""
    img = np.full(occ.raw.shape + (3,), 255, np.uint8)
    img[(occ.inflated == 1) & (occ.raw == FREE)] = (200, 200, 200)
    if occ.low is not None:
        img[(occ.low == 1) & (occ.raw == FREE)] = (245, 160, 60)
    img[occ.raw == OUTSIDE] = (70, 80, 95)
    img[occ.raw == OBSTACLE] = (0, 0, 0)
    return img[::-1]


def save_png(occ: Occupancy, path: Path, scale: int = 4) -> None:
    from PIL import Image

    img = to_rgb(occ)
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)


def save_overlay_png(occ: Occupancy, house: dict, path: Path, extra_title: str = "") -> None:
    """Occupancy + room polygons/names + object AABBs + spawn + per-room points (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10 * occ.shape[0] / max(occ.shape[1], 1) + 0.6), dpi=110)
    ax.imshow(to_rgb(occ), extent=occ.extent, origin="upper", interpolation="nearest")
    for r in house.get("rooms", []):
        p = np.asarray(r["polygon"] + [r["polygon"][0]])
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.2)
        ax.text(*r["center"], f"{r['name']}\n({r['area_m2']:.1f} m2)", color="tab:blue", ha="center", va="center", fontsize=9, weight="bold")
    for o in house.get("objects", []):
        (x0, y0, _), (x1, y1, _) = o["aabb"]
        if o["category"] in ("Window", "Painting"):
            continue
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="tab:green" if o["is_static"] else "tab:purple", lw=0.6))
    for w in house.get("room_points", []):
        if w.get("ok"):
            ax.plot(w["x"], w["y"], "o", color="tab:orange", ms=6)
    sp = house.get("spawn")
    if sp:
        ax.plot(sp["x"], sp["y"], "*", color="red", ms=16)
        ax.arrow(sp["x"], sp["y"], 0.6 * math.cos(sp["yaw"]), 0.6 * math.sin(sp["yaw"]), color="red", width=0.03)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(f"{house.get('house_id', occ.house_id)}  occupancy {occ.resolution:.2f} m, z {occ.z_min}-{occ.z_max} m, "
                 f"r={occ.robot_radius} m ({occ.method}) {extra_title}", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Isaac-side generation (requires a running Isaac Sim with the house loaded and PhysX initialised)
# ----------------------------------------------------------------------------------------------


def raster_physx_overlap(root: str, origin, shape, resolution, z_min, z_max, exclude_prefixes=()) -> np.ndarray:
    """Exact rasterisation of PhysX colliders under `root`: one overlap_box per cell covering
    [z_min, z_max] (omni.physx scene query interface; the same shapes PhysX collides with the robot).
    Colliders outside `root` (e.g. the robot) are ignored."""
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    xs, ys = cell_centres(origin, shape, resolution)
    grid = np.zeros(shape, np.uint8)
    half = carb.Float3(resolution / 2, resolution / 2, (z_max - z_min) / 2)
    zc = (z_max + z_min) / 2
    rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
    hit = [False]

    def report(h) -> bool:
        p = h.collision
        if p.startswith(root) and not any(p.startswith(e) for e in exclude_prefixes):
            hit[0] = True
            return False  # stop at the first relevant hit
        return True

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            hit[0] = False
            sq.overlap_box(half, carb.Float3(float(x), float(y), zc), rot, report, False)
            if hit[0]:
                grid[iy, ix] = 1
    return grid


def raster_omap(origin, shape, resolution, z_min, z_max, seed_xy) -> tuple[np.ndarray | None, dict]:
    """isaacsim.asset.gen.omap 2D map (Isaac Sim 5.1, ext 2.0.29; API from
    isaacsim.asset.gen.omap include/.../MapGenerator.h and tests/test_occupancy.py:151-200).
    Returns (grid of 1 occupied / 0 free / 2 unknown on our frame, info)."""
    import omni.physx
    import omni.usd
    from isaacsim.asset.gen.omap.bindings import _omap

    ny, nx = shape
    gen = _omap.Generator(omni.physx.get_physx_interface(), omni.usd.get_context().get_stage_id())
    gen.update_settings(resolution, 1.0, 0.0, 0.5)
    ox, oy = seed_xy
    xmin, ymin = origin
    xmax, ymax = origin[0] + nx * resolution, origin[1] + ny * resolution
    gen.set_transform((ox, oy, 0.0), (xmin - ox, ymin - oy, z_min), (xmax - ox, ymax - oy, z_max))
    t0 = time.time()
    gen.generate2d()
    info = {"generate2d_s": round(time.time() - t0, 3)}
    dims = gen.get_dimensions()
    info.update({"dims": [int(dims[0]), int(dims[1]), int(dims[2])], "min_bound": list(gen.get_min_bound()), "max_bound": list(gen.get_max_bound())})
    buf = np.asarray(gen.get_buffer(), np.float32)
    info["buffer_len"] = int(buf.size)
    if buf.size == 0 or dims[0] * dims[1] != buf.size:
        return None, info
    # Occupied cell centres are returned in world coordinates, which pins down the orientation
    # without guessing the buffer layout.
    occ_pts = np.asarray([[p.x, p.y] for p in gen.get_occupied_positions()], np.float64)
    grid = np.full(shape, 2, np.uint8)
    free_pts = np.asarray([[p.x, p.y] for p in gen.get_free_positions()], np.float64)
    for pts, val in ((free_pts, 0), (occ_pts, 1)):
        if len(pts) == 0:
            continue
        ix = np.floor((pts[:, 0] - origin[0]) / resolution).astype(int)
        iy = np.floor((pts[:, 1] - origin[1]) / resolution).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid[iy[ok], ix[ok]] = val
    info.update({"n_occupied_pts": int(len(occ_pts)), "n_free_pts": int(len(free_pts))})
    return grid, info


def generate_occupancy(
    house,  # scenes.loader.HouseInfo
    *,
    resolution: float = DEFAULTS["resolution"],
    z_min: float = DEFAULTS["z_min"],
    z_max: float = DEFAULTS["z_max"],
    robot_radius: float = DEFAULTS["robot_radius"],
    low_z_min: float = DEFAULTS["low_z_min"],
    method: str = "auto",
    exclude_prefixes=(),
    save: bool = True,
) -> Occupancy:
    """Build the grid for a loaded house. PhysX must already hold the house colliders
    (e.g. after SimulationContext.reset() / timeline play + one step).

    method: "omap" | "overlap" | "auto" (both; omap is kept when it agrees with the exact PhysX
    overlap raster at IoU >= 0.9 on obstacles, else the overlap raster is used and the reason is
    recorded in meta["method_choice"]).
    """
    rooms = [r if isinstance(r, dict) else asdict(r) for r in house.rooms]
    origin, shape = make_grid_frame(house.bounds_xy, resolution)
    inside = rooms_mask(rooms, origin, shape, resolution) if rooms else np.ones(shape, bool)
    meta: dict = {"bounds_xy": house.bounds_xy}
    grids: dict[str, np.ndarray] = {}

    if method in ("overlap", "auto"):
        t0 = time.time()
        grids["overlap"] = raster_physx_overlap(house.root, origin, shape, resolution, z_min, z_max, exclude_prefixes)
        meta["overlap_s"] = round(time.time() - t0, 2)
    if method in ("omap", "auto"):
        seed = (house.spawn or {}).get("x"), (house.spawn or {}).get("y")
        if seed[0] is None:
            seed = house.rooms[0]["center"] if isinstance(house.rooms[0], dict) else house.rooms[0].center
        try:
            g, info = raster_omap(origin, shape, resolution, z_min, z_max, seed)
            meta["omap"] = info
            if g is not None:
                grids["omap"] = g
        except Exception as e:  # pragma: no cover - Isaac specific
            meta["omap"] = {"error": repr(e)}

    chosen = None
    if "omap" in grids and "overlap" in grids:
        a = grids["omap"] == 1
        b = grids["overlap"] == 1
        a_in, b_in = a & inside, b & inside
        iou = float((a_in & b_in).sum() / max((a_in | b_in).sum(), 1))
        meta["omap_vs_overlap"] = {
            "iou_obstacles_inside": round(iou, 4),
            "omap_only_cells": int((a_in & ~b_in).sum()),
            "overlap_only_cells": int((b_in & ~a_in).sum()),
            "omap_unknown_inside": int(((grids["omap"] == 2) & inside).sum()),
        }
        chosen = "omap" if iou >= 0.9 else "overlap"
        meta["method_choice"] = (
            f"omap kept (IoU {iou:.3f} >= 0.9 vs exact PhysX overlap raster)" if chosen == "omap"
            else f"omap rejected (IoU {iou:.3f} < 0.9 vs exact PhysX overlap raster); using overlap raster"
        )
    elif "omap" in grids:
        chosen = "omap"
    elif "overlap" in grids:
        chosen = "overlap"
        if method == "auto":
            meta["method_choice"] = "omap unavailable/failed; using PhysX overlap raster"
    else:
        raise RuntimeError(f"no occupancy method succeeded: {meta}")

    obst = grids[chosen] == 1
    raw = np.where(obst, OBSTACLE, FREE).astype(np.uint8)
    raw[~inside & ~obst] = OUTSIDE

    low = None
    if low_z_min < z_min:
        t0 = time.time()
        low = raster_physx_overlap(house.root, origin, shape, resolution, low_z_min, z_min, exclude_prefixes)
        low &= (~obst).astype(np.uint8)
        meta["low_s"] = round(time.time() - t0, 2)

    occ = finish(house.house_id, raw, origin, resolution, z_min, z_max, robot_radius, chosen, low, meta)
    if save:
        occ.save(house.assets_dir)
        for k, g in grids.items():  # keep both rasters for inspection
            np.save(house.assets_dir / f"raster_{k}.npy", g)
    return occ
