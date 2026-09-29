"""Occupancy grid, inflation, A* and path smoothing for the PathFollower.

Grid convention (body section of docs/contracts/m1.md):
  occ[iy, ix]; cell (0,0) has its lower-left corner at world `origin` = (x0, y0); x grows with ix, y with iy.
  cell centre: x = x0 + (ix + 0.5) * res, y = y0 + (iy + 0.5) * res.
  Values: bool, or ints (0 free, >0 occupied, <0 unknown), or floats as produced by Isaac's omap generator with
  update_settings(res, occupied=1.0, free=0.0, unknown=0.5) (research doc §4) -> >= 0.75 occupied, (0.25, 0.75) unknown.
  Unknown is treated as blocked unless unknown_is_free.
Loader accepts other layouts through hints in the get_occupancy reply: layout "xy" (occ[ix, iy]) and
origin_is_cell_center.

Planning: inflate by robot_radius (minus whatever inflation P1 already applied), A* (8-connected, octile heuristic,
clearance-weighted cost that keeps paths centred in doorways) on a coarse grid (plan_res, default 0.10 m) sampled at
block centres of the fine inflated grid, then greedy line-of-sight shortcutting checked on the FINE inflated grid, then
resampling every 0.10 m.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class PlanResult:
    ok: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (N,2) world xy, starts at start
    length: float = 0.0
    reason: str = ""
    goal_snapped: tuple | None = None
    start_snapped: tuple | None = None
    raw_cells: int = 0
    expansions: int = 0
    plan_ms: float = 0.0

    def brief(self, max_pts: int = 60) -> dict:
        p = self.path
        if len(p) > max_pts:
            idx = np.linspace(0, len(p) - 1, max_pts).round().astype(int)
            p = p[idx]
        return {"ok": self.ok, "reason": self.reason, "length_m": round(self.length, 3),
                "n": int(len(self.path)), "goal_snapped": self.goal_snapped, "start_snapped": self.start_snapped,
                "plan_ms": round(self.plan_ms, 1), "expansions": self.expansions,
                "path": [[round(float(a), 3), round(float(b), 3)] for a, b in p]}


class NavGrid:
    def __init__(self, occ: np.ndarray, res: float, origin: tuple[float, float], robot_radius: float = 0.25,
                 already_inflated_m: float = 0.0, unknown_is_free: bool = False, plan_res: float = 0.10,
                 meta: dict | None = None):
        self.res = float(res)
        self.origin = (float(origin[0]), float(origin[1]))
        self.robot_radius = float(robot_radius)
        self.already_inflated = float(already_inflated_m)
        self.meta = meta or {}
        self.raw = self._to_blocked(np.asarray(occ), unknown_is_free)   # True = obstacle
        self.H, self.W = self.raw.shape
        self.k = max(1, int(round(plan_res / self.res)))
        self.virtual: list[tuple[float, float, float]] = []
        self._recompute()

    # -- construction --------------------------------------------------------------------------
    @staticmethod
    def _to_blocked(a: np.ndarray, unknown_is_free: bool) -> np.ndarray:
        if a.dtype == bool:
            return a.copy()
        if np.issubdtype(a.dtype, np.floating):
            occ = a >= 0.75
            unk = (a > 0.25) & (a < 0.75)
        else:
            occ = a > 0
            unk = a < 0
        return occ | (unk & (not unknown_is_free))

    @classmethod
    def from_arrays(cls, occ, res, origin, **kw) -> "NavGrid":
        return cls(occ, res, origin, **kw)

    @classmethod
    def from_p1_reply(cls, rep: dict, robot_radius: float = 0.25, plan_res: float = 0.10) -> "NavGrid":
        """Build from a get_occupancy reply: {path|npz_path, resolution, origin, inflation_m, layout?, ...}.
        The npz may also carry `resolution` / `origin`; the reply wins."""
        src = rep.get("result", rep) if isinstance(rep.get("result"), dict) else rep
        path = src.get("path") or src.get("npz_path") or src.get("npz")
        arrs: dict = {}
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"occupancy file not found: {path}")
            if path.endswith(".npy"):
                arrs = {"occupancy": np.load(path)}
            else:
                with np.load(path, allow_pickle=False) as z:
                    arrs = {k: z[k] for k in z.files}
        elif "grid" in src:
            arrs = {"occupancy": np.asarray(src["grid"])}
        occ = None
        # P1 (m1.md §1.6) ships `occ` (raw, 1 = blocked) and `occ_inflated`; always use the RAW grid
        for key in ("occ", "occupancy", "grid", "omap", "map", "data"):
            if key in arrs:
                occ = arrs[key]
                break
        if occ is None:
            twod = [v for k, v in arrs.items() if getattr(v, "ndim", 0) == 2 and "inflat" not in k]
            if not twod:
                raise ValueError(f"no 2-D occupancy array in {path} (keys {list(arrs)})")
            occ = twod[0]
        res = float(src.get("resolution") or src.get("res") or _scalar(arrs.get("resolution")) or 0.05)
        origin = src.get("origin")
        if origin is None and "origin" in arrs:
            origin = arrs["origin"].tolist()
        if origin is None:
            origin = [0.0, 0.0]
        origin = [float(origin[0]), float(origin[1])]
        layout = str(src.get("layout") or _str(arrs.get("layout")) or "yx").lower()
        if layout.startswith("xy") or layout.startswith("grid[ix"):
            occ = occ.T
        if src.get("origin_is_cell_center"):
            origin = [origin[0] - res / 2, origin[1] - res / 2]
        infl = float(src.get("inflation_m") or src.get("inflation") or 0.0)
        meta = {k: v for k, v in src.items() if k not in ("grid",)}
        return cls(occ, res, origin, robot_radius=robot_radius, already_inflated_m=infl, plan_res=plan_res,
                   unknown_is_free=bool(src.get("unknown_is_free", False)), meta=meta)

    def _recompute(self) -> None:
        blocked = self.raw.copy()
        for (vx, vy, vr) in self.virtual:
            blocked |= self._disk_mask(vx, vy, vr)
        self.blocked_raw = blocked
        # clearance (m) from every cell centre to the nearest obstacle cell
        self.clear = ndimage.distance_transform_edt(~blocked) * self.res
        r_eff = max(0.0, self.robot_radius - self.already_inflated)
        self.r_eff = r_eff
        self.inflated = self.clear < max(r_eff, 0.5 * self.res)
        # coarse planning grid: sample the fine grid at block centres
        k = self.k
        c = k // 2
        self.cH, self.cW = (self.H - c + k - 1) // k, (self.W - c + k - 1) // k
        ys = np.arange(self.cH) * k + c
        xs = np.arange(self.cW) * k + c
        ys = ys[ys < self.H]
        xs = xs[xs < self.W]
        self.cH, self.cW = len(ys), len(xs)
        self._cy, self._cx = ys, xs
        self.c_blocked = self.inflated[np.ix_(ys, xs)]
        self.c_clear = self.clear[np.ix_(ys, xs)]

    def _disk_mask(self, x, y, r) -> np.ndarray:
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        cx = self.origin[0] + (xx + 0.5) * self.res
        cy = self.origin[1] + (yy + 0.5) * self.res
        return (cx - x) ** 2 + (cy - y) ** 2 <= r * r

    def add_virtual_obstacle(self, x: float, y: float, r: float) -> None:
        self.virtual.append((float(x), float(y), float(r)))
        self._recompute()

    def clear_virtual(self) -> None:
        if self.virtual:
            self.virtual = []
            self._recompute()

    def pop_virtual(self) -> tuple[float, float, float] | None:
        """Remove the newest virtual obstacle (x, y, r) and return it (None when there is none)."""
        if not self.virtual:
            return None
        v = self.virtual.pop()
        self._recompute()
        return v

    # -- coordinates ---------------------------------------------------------------------------
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((y - self.origin[1]) / self.res)), int(math.floor((x - self.origin[0]) / self.res))

    def cell_to_world(self, iy: int, ix: int) -> tuple[float, float]:
        return self.origin[0] + (ix + 0.5) * self.res, self.origin[1] + (iy + 0.5) * self.res

    def in_bounds(self, iy: int, ix: int) -> bool:
        return 0 <= iy < self.H and 0 <= ix < self.W

    def is_free(self, x: float, y: float, inflated: bool = True) -> bool:
        iy, ix = self.world_to_cell(x, y)
        if not self.in_bounds(iy, ix):
            return False
        return not (self.inflated[iy, ix] if inflated else self.blocked_raw[iy, ix])

    def clearance(self, x: float, y: float) -> float:
        iy, ix = self.world_to_cell(x, y)
        if not self.in_bounds(iy, ix):
            return 0.0
        return float(self.clear[iy, ix])

    def extent(self) -> list[float]:
        return [self.origin[0], self.origin[0] + self.W * self.res, self.origin[1], self.origin[1] + self.H * self.res]

    def _c_world(self, cy: int, cx: int) -> tuple[float, float]:
        return self.cell_to_world(int(self._cy[cy]), int(self._cx[cx]))

    def _world_to_c(self, x: float, y: float) -> tuple[int, int]:
        iy, ix = self.world_to_cell(x, y)
        cy = int(round((iy - self.k // 2) / self.k))
        cx = int(round((ix - self.k // 2) / self.k))
        return min(max(cy, 0), self.cH - 1), min(max(cx, 0), self.cW - 1)

    def _nearest_free_c(self, cy: int, cx: int, max_r_m: float) -> tuple[int, int] | None:
        rmax = int(math.ceil(max_r_m / (self.k * self.res)))
        best, bestd = None, 1e18
        for dy in range(-rmax, rmax + 1):
            for dx in range(-rmax, rmax + 1):
                y, x = cy + dy, cx + dx
                if 0 <= y < self.cH and 0 <= x < self.cW and not self.c_blocked[y, x]:
                    d = dy * dy + dx * dx
                    if d < bestd:
                        best, bestd = (y, x), d
        if best is None or math.sqrt(bestd) * self.k * self.res > max_r_m + 1e-9:
            return None
        return best

    # -- line of sight -------------------------------------------------------------------------
    def segment_free(self, a, b, grid: np.ndarray | None = None) -> bool:
        g = self.inflated if grid is None else grid
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        L = math.hypot(bx - ax, by - ay)
        n = max(2, int(math.ceil(L / (0.5 * self.res))) + 1)
        xs = np.linspace(ax, bx, n)
        ys = np.linspace(ay, by, n)
        iy = np.floor((ys - self.origin[1]) / self.res).astype(int)
        ix = np.floor((xs - self.origin[0]) / self.res).astype(int)
        if (iy < 0).any() or (ix < 0).any() or (iy >= self.H).any() or (ix >= self.W).any():
            return False
        return not g[iy, ix].any()

    def ray_free_distance(self, x: float, y: float, heading: float, max_d: float = 6.0,
                          inflated: bool = True) -> float:
        g = self.inflated if inflated else self.blocked_raw
        step = 0.5 * self.res
        d = 0.0
        while d < max_d:
            px, py = x + (d + step) * math.cos(heading), y + (d + step) * math.sin(heading)
            iy, ix = self.world_to_cell(px, py)
            if not self.in_bounds(iy, ix) or g[iy, ix]:
                return d
            d += step
        return max_d

    # -- A* --------------------------------------------------------------------------------------
    def plan(self, start, goal, snap_radius: float = 0.5, start_snap_radius: float = 0.8,
             w_clear: float = 2.0, clear_scale: float = 0.25, resample: float = 0.10) -> PlanResult:
        import time as _t

        t0 = _t.perf_counter()
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        sc = self._world_to_c(sx, sy)
        gc = self._world_to_c(gx, gy)
        res = PlanResult(ok=False)
        if self.c_blocked[sc]:
            alt = self._nearest_free_c(*sc, start_snap_radius)
            if alt is None:
                res.reason = "start_in_obstacle"
                return res
            sc = alt
            res.start_snapped = tuple(round(v, 3) for v in self._c_world(*sc))
        goal_pt = (gx, gy)
        if self.c_blocked[gc] or not self.is_free(gx, gy):
            alt = self._nearest_free_c(*gc, snap_radius)
            if alt is None:
                res.reason = "goal_in_obstacle"
                return res
            gc = alt
            goal_pt = self._c_world(*gc)
            res.goal_snapped = (round(goal_pt[0], 3), round(goal_pt[1], 3))
        cells, exp = self._astar(sc, gc, w_clear, clear_scale)
        res.expansions = exp
        if cells is None:
            res.reason = "no_path"
            res.plan_ms = (_t.perf_counter() - t0) * 1e3
            return res
        pts = [self._c_world(cy, cx) for cy, cx in cells]
        if res.start_snapped is None:
            pts[0] = (sx, sy)
        else:
            pts.insert(0, (sx, sy))
        if len(pts) == 1:
            pts.append(goal_pt)
        else:
            pts[-1] = goal_pt
        pts = np.asarray(pts, dtype=float)
        res.raw_cells = len(cells)
        pts = self.shortcut(pts)
        pts = resample_path(pts, resample)
        res.path = pts
        res.length = path_length(pts)
        res.ok = True
        res.plan_ms = (_t.perf_counter() - t0) * 1e3
        return res

    def _astar(self, s, g, w_clear, clear_scale):
        H, W = self.cH, self.cW
        blocked = self.c_blocked
        step = self.k * self.res
        mult = 1.0 + w_clear * np.exp(-np.clip(self.c_clear - self.r_eff, 0, None) / clear_scale)
        INF = float("inf")
        gcost = np.full((H, W), INF)
        parent = -np.ones((H, W), dtype=np.int64)
        closed = np.zeros((H, W), dtype=bool)
        gcost[s] = 0.0
        gy, gx = g
        D2 = math.sqrt(2.0)

        def h(y, x):
            dy, dx = abs(y - gy), abs(x - gx)
            return step * ((dx + dy) + (D2 - 2) * min(dx, dy))

        heap = [(h(*s), 0.0, s[0], s[1])]
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, D2), (-1, 1, D2), (1, -1, D2), (1, 1, D2)]
        exp = 0
        while heap:
            f, gc_, y, x = heapq.heappop(heap)
            if closed[y, x]:
                continue
            closed[y, x] = True
            exp += 1
            if (y, x) == (gy, gx):
                break
            for dy, dx, c in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or nx < 0 or ny >= H or nx >= W or blocked[ny, nx] or closed[ny, nx]:
                    continue
                if dy and dx and (blocked[y, nx] or blocked[ny, x]):  # no corner cutting
                    continue
                ng = gc_ + c * step * mult[ny, nx]
                if ng < gcost[ny, nx]:
                    gcost[ny, nx] = ng
                    parent[ny, nx] = y * W + x
                    heapq.heappush(heap, (ng + h(ny, nx), ng, ny, nx))
        if not closed[gy, gx]:
            return None, exp
        cells = []
        cur = gy * W + gx
        while cur >= 0:
            cy, cx = divmod(int(cur), W)
            cells.append((cy, cx))
            if (cy, cx) == tuple(s):
                break
            cur = parent[cy, cx]
        return cells[::-1], exp

    def shortcut(self, pts: np.ndarray) -> np.ndarray:
        """Greedy farthest-visible shortcutting on the fine inflated grid (start/end kept)."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not self.segment_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.asarray(out)


def _scalar(v):
    if v is None:
        return None
    try:
        return float(np.asarray(v).reshape(-1)[0])
    except Exception:
        return None


def _str(v):
    if v is None:
        return None
    try:
        return str(np.asarray(v).reshape(-1)[0])
    except Exception:
        return None


def path_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def resample_path(p: np.ndarray, ds: float) -> np.ndarray:
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return p[:1]
    n = max(2, int(math.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])], axis=1)
