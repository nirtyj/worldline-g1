"""Occupancy grid for the runtime side: free checks, stand clearance, all-pairs keypoint distances, paths.

The grid is the house agent's cached occupancy (docs/scenes.md §5; P1 `get_occupancy`, contract §1.6): 0.05 m cells,
`occ[iy, ix]` = 1 blocked (obstacles between z 0.02-1.60 m, low colliders, and outside every room), not inflated,
`origin` = world xy of the lower-left corner of cell (0, 0).

Planning uses the BODY's planner (`body.nav_grid.NavGrid`, imported, not copied), with the body's robot radius, so
a path length the runtime quotes is the path wl-body's `go_to` (astar backend) will plan. All-pairs keypoint
distances (the MAP `edges`, `list_locations`) use one scipy csgraph Dijkstra per source on the same 0.10 m planning
grid with 16-connected moves (<= 2.7 % longer than the any-angle shortest path; the body's clearance-centred path
is typically a few % longer still).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from body.nav_grid import NavGrid, PlanResult  # noqa: F401  (re-exported for services)

_OFFSETS_16 = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]   # half; the graph is undirected


@dataclass(frozen=True)
class OccupancyData:
    occ: np.ndarray                  # bool, True = blocked (raw, not inflated)
    res: float
    origin: tuple[float, float]
    raw_codes: np.ndarray | None = None   # uint8 0 free / 1 obstacle / 2 outside (house agent's `raw`), if known
    source: str = ""

    @classmethod
    def from_npz(cls, path: str | Path, source: str = "") -> "OccupancyData":
        with np.load(str(path), allow_pickle=False) as z:
            files = set(z.files)
            if "occ" in files:
                occ = z["occ"].astype(bool)
            elif "raw" in files:
                occ = z["raw"] != 0
                if "low" in files:
                    occ = occ | (z["low"] != 0)
            else:
                raise ValueError(f"{path}: no 'occ' or 'raw' array (keys {sorted(files)})")
            raw = z["raw"].astype(np.uint8) if "raw" in files else None
            res = float(np.asarray(z["resolution"]).reshape(-1)[0]) if "resolution" in files else 0.05
            origin = tuple(float(v) for v in np.asarray(z["origin"]).reshape(-1)[:2]) if "origin" in files \
                else (0.0, 0.0)
        return cls(occ=occ, res=res, origin=origin, raw_codes=raw, source=source or str(path))   # type: ignore[arg-type]

    @classmethod
    def from_p1_reply(cls, rep: dict) -> "OccupancyData":
        """P1 get_occupancy reply: {path, resolution, origin, ...}; the npz lives on the box (P5 runs there)."""
        src = rep.get("result", rep) if isinstance(rep.get("result"), dict) else rep
        d = cls.from_npz(src["path"], source=str(src.get("source") or src["path"]))
        res = float(src.get("resolution") or d.res)
        origin = tuple(float(v) for v in (src.get("origin") or d.origin))
        return cls(occ=d.occ, res=res, origin=origin, raw_codes=d.raw_codes, source=d.source)   # type: ignore[arg-type]


class WorldGrid:
    """Free space, clearance and distances for mapgen, navigation, locations and reachability."""

    def __init__(self, data: OccupancyData, robot_radius: float = 0.25, plan_res: float = 0.10):
        self.data = data
        self.robot_radius = float(robot_radius)
        self.nav = NavGrid(data.occ, data.res, data.origin, robot_radius=robot_radius, plan_res=plan_res)
        self.res = self.nav.res
        self.origin = self.nav.origin
        self._build_graph()

    # ------------------------------------------------------------------ basics
    def is_free(self, x: float, y: float) -> bool:
        """Free for the robot centre (inflated by the body's robot radius)."""
        return self.nav.is_free(x, y, inflated=True)

    def clearance(self, x: float, y: float) -> float:
        """Metres from (x, y) to the nearest blocked cell (0 outside the grid)."""
        return self.nav.clearance(x, y)

    def segment_free(self, a: Sequence[float], b: Sequence[float]) -> bool:
        return self.nav.segment_free(a, b)

    def is_outside(self, x: float, y: float) -> bool:
        if self.data.raw_codes is None:
            return False
        iy, ix = self.nav.world_to_cell(x, y)
        if not self.nav.in_bounds(iy, ix):
            return True
        return int(self.data.raw_codes[iy, ix]) == 2

    # ------------------------------------------------------------------ coarse graph
    def _build_graph(self) -> None:
        free = ~self.nav.c_blocked
        H, W = free.shape
        self.cH, self.cW = H, W
        step = self.nav.k * self.nav.res
        rows, cols, vals = [], [], []
        idx = np.arange(H * W).reshape(H, W)
        for dy, dx in _OFFSETS_16:
            y0, y1 = max(0, -dy), min(H, H - dy)
            x0, x1 = max(0, -dx), min(W, W - dx)
            a = free[y0:y1, x0:x1]
            b = free[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            ok = a & b
            # no corner cutting: every cell the move passes must be free
            if abs(dy) + abs(dx) >= 2:
                for iy, ix in _passes(dy, dx):
                    ok &= free[y0 + iy:y1 + iy, x0 + ix:x1 + ix]
            src = idx[y0:y1, x0:x1][ok]
            dst = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx][ok]
            rows.append(src)
            cols.append(dst)
            vals.append(np.full(src.shape, step * math.hypot(dy, dx)))
        r = np.concatenate(rows) if rows else np.zeros(0, int)
        c = np.concatenate(cols) if cols else np.zeros(0, int)
        v = np.concatenate(vals) if vals else np.zeros(0)
        n = H * W
        g = csr_matrix((v, (r, c)), shape=(n, n))
        self.graph = g.maximum(g.T)                       # undirected
        self.n_comp, self.labels = connected_components(self.graph, directed=False)
        self._free_c = free

    def node_of(self, x: float, y: float, snap_m: float = 0.6) -> tuple[int | None, float]:
        """Nearest free coarse node to (x, y) and the straight distance to it (m)."""
        cy, cx = self.nav._world_to_c(x, y)
        if not self._free_c[cy, cx]:
            alt = self.nav._nearest_free_c(cy, cx, snap_m)
            if alt is None:
                return None, math.inf
            cy, cx = alt
        wx, wy = self.nav._c_world(cy, cx)
        return cy * self.cW + cx, math.hypot(wx - x, wy - y)

    def component(self, x: float, y: float) -> int | None:
        n, _ = self.node_of(x, y)
        return None if n is None else int(self.labels[n])

    def distances(self, sources: Sequence[Sequence[float]], targets: Sequence[Sequence[float]] | None = None
                  ) -> np.ndarray:
        """Geodesic walking distances (m); rows = sources, cols = targets (default: the sources).
        inf where there is no path or a point is not on free space."""
        targets = sources if targets is None else targets
        s_nodes = [self.node_of(*p[:2]) for p in sources]
        t_nodes = [self.node_of(*p[:2]) for p in targets]
        out = np.full((len(sources), len(targets)), np.inf)
        valid = [i for i, (n, _) in enumerate(s_nodes) if n is not None]
        if not valid:
            return out
        d = dijkstra(self.graph, directed=False, indices=[s_nodes[i][0] for i in valid])
        d = np.atleast_2d(d)
        for row, i in enumerate(valid):
            si_off = s_nodes[i][1]
            for j, (tn, t_off) in enumerate(t_nodes):
                if tn is None:
                    continue
                dd = d[row, tn]
                if np.isfinite(dd):
                    if tn == s_nodes[i][0]:
                        out[i, j] = math.dist(sources[i][:2], targets[j][:2])
                    else:
                        out[i, j] = dd + si_off + t_off
        return out

    def distance(self, a: Sequence[float], b: Sequence[float]) -> float | None:
        d = float(self.distances([a], [b])[0, 0])
        return d if math.isfinite(d) else None

    # ------------------------------------------------------------------ paths (the body's planner)
    def plan(self, a: Sequence[float], b: Sequence[float]) -> PlanResult:
        return self.nav.plan(a[:2], b[:2])

    def path(self, a: Sequence[float], b: Sequence[float]) -> list[tuple[float, float]] | None:
        r = self.plan(a, b)
        if not r.ok:
            return None
        return [(float(x), float(y)) for x, y in r.path]

    def path_length(self, a: Sequence[float], b: Sequence[float]) -> float | None:
        r = self.plan(a, b)
        return r.length if r.ok else None

    # ------------------------------------------------------------------ helpers for mapgen / UI
    def best_free_in(self, mask_fn, component: int | None = None) -> tuple[float, float, float] | None:
        """The fine free cell with the largest clearance for which mask_fn(x, y) holds -> (x, y, clearance)."""
        clear = np.where(self.nav.inflated, 0.0, self.nav.clear)
        order = np.argsort(clear, axis=None)[::-1]
        for flat in order[:20000]:
            iy, ix = divmod(int(flat), self.nav.W)
            c = float(clear[iy, ix])
            if c <= 0:
                break
            x, y = self.nav.cell_to_world(iy, ix)
            if not mask_fn(x, y):
                continue
            if component is not None and self.component(x, y) != component:
                continue
            return x, y, c
        return None

    def free_cells(self, step: float = 0.25) -> list[list[int]]:
        """Free robot-centre cells on a `step` lattice as [ix, iz] with ix = round(x/step) (the THOR/UI `grid`)."""
        out = []
        x0, y0 = self.origin
        x1, y1 = x0 + self.nav.W * self.res, y0 + self.nav.H * self.res
        for ix in range(int(math.ceil(x0 / step)), int(math.floor(x1 / step)) + 1):
            for iy in range(int(math.ceil(y0 / step)), int(math.floor(y1 / step)) + 1):
                if self.is_free(ix * step, iy * step):
                    out.append([ix, iy])
        return out

    def extent(self) -> list[float]:
        return self.nav.extent()


def _passes(dy: int, dx: int) -> Iterable[tuple[int, int]]:
    """Intermediate cells a (dy, dx) move crosses (besides its ends)."""
    if abs(dy) == 1 and abs(dx) == 1:
        return [(dy, 0), (0, dx)]
    if abs(dy) == 2:     # (2, +-1): passes (1, 0) and (1, dx)
        sy = 1 if dy > 0 else -1
        return [(sy, 0), (sy, dx)]
    if abs(dx) == 2:
        sx = 1 if dx > 0 else -1
        return [(0, sx), (dy, sx)]
    return []
