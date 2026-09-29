"""House occupancy (P1 get_occupancy npz, docs/contracts/m1.md §1.6) -> ROS map values / map_server pgm+yaml.

Pure numpy + stdlib so both the ROS bridge (system Python 3.12) and the body venv (3.11) can import it.

Grid convention (P1 contract, same as nav_msgs/OccupancyGrid): occ[iy, ix], row 0 is the lowest y, cell (0,0) has its
lower-left CORNER at `origin` (x0, y0); x grows with the column. OccupancyGrid.data is row-major from (0,0), so the
array maps 1:1 (no flip). A pgm image is written top row first, so the pgm IS flipped (map_server flips it back).

Values: we publish the RAW grid (not inflated): 100 = blocked (obstacle or outside the house), 0 = free,
-1 = unknown (only for float omap grids with 0.25 < v < 0.75). Nav2's inflation layer adds the robot radius.
"""

from __future__ import annotations

import os

import numpy as np


def load_occupancy(path: str) -> dict:
    """Load a P1 occupancy npz. Returns {occ (raw array), resolution, origin, keys}."""
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        occ = None
        for k in ("occ", "raw", "occupancy", "grid", "omap"):
            if k in keys:
                occ = np.asarray(z[k])
                break
        if occ is None:
            cands = [k for k in keys if z[k].ndim == 2 and "inflat" not in k and k not in ("dist", "low")]
            if not cands:
                raise ValueError(f"no 2-D occupancy array in {path}: {keys}")
            occ = np.asarray(z[cands[0]])
        res = float(z["resolution"]) if "resolution" in keys else None
        origin = [float(v) for v in z["origin"][:2]] if "origin" in keys else None
    return {"occ": occ, "resolution": res, "origin": origin, "keys": keys}


def to_ros_values(occ: np.ndarray) -> np.ndarray:
    """Raw occupancy -> int8 OccupancyGrid values (100 blocked, 0 free, -1 unknown)."""
    a = np.asarray(occ)
    if a.dtype == bool:
        return np.where(a, 100, 0).astype(np.int8)
    if np.issubdtype(a.dtype, np.floating):
        out = np.zeros(a.shape, dtype=np.int8)
        out[a >= 0.75] = 100
        out[(a > 0.25) & (a < 0.75)] = -1
        return out
    out = np.where(a > 0, 100, 0).astype(np.int8)
    out[a < 0] = -1
    return out


def write_map_server_files(out_prefix: str, occ: np.ndarray, resolution: float, origin) -> tuple[str, str]:
    """Write <prefix>.pgm + <prefix>.yaml for nav2_map_server (mode trinary, negate 0)."""
    vals = to_ros_values(occ)
    img = np.full(vals.shape, 254, dtype=np.uint8)      # free (white)
    img[vals == 100] = 0                                # occupied (black)
    img[vals == -1] = 205                               # unknown (grey)
    img = img[::-1]                                     # pgm row 0 = top = max y
    pgm = out_prefix + ".pgm"
    yml = out_prefix + ".yaml"
    os.makedirs(os.path.dirname(os.path.abspath(pgm)), exist_ok=True)
    with open(pgm, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())
    with open(yml, "w") as f:
        f.write(f"image: {os.path.basename(pgm)}\nmode: trinary\nresolution: {float(resolution)}\n"
                f"origin: [{float(origin[0])}, {float(origin[1])}, 0.0]\nnegate: 0\n"
                f"occupied_thresh: 0.65\nfree_thresh: 0.25\n")
    return pgm, yml
