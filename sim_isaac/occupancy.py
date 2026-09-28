"""2-D occupancy grid from the PhysX scene (fallback when the scene plugin does not provide one).

Each cell is tested with a PhysX box overlap query covering [floor_z + z_min, floor_z + z_max]; hits on the robot's
own prims are ignored. Needs SimulationCfg(enable_scene_query_support=True). npz format: docs/contracts/m1.md 1.6.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np


def inflate(occ: np.ndarray, radius_m: float, res: float) -> np.ndarray:
    import cv2

    r = int(math.ceil(radius_m / res))
    if r <= 0:
        return occ.copy()
    k = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(k, (r, r), r, 1, -1)
    return cv2.dilate(occ.astype(np.uint8), k)


def generate_physx(bounds, floor_z: float, res: float = 0.05, z_min: float = 0.10, z_max: float = 1.60,
                   ignore_prefixes: tuple[str, ...] = ("/World/G1",), log=print) -> tuple[np.ndarray, list[float]]:
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    xmin, ymin, xmax, ymax = bounds
    nx = int(math.ceil((xmax - xmin) / res))
    ny = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((ny, nx), np.uint8)
    half = carb.Float3(res / 2, res / 2, (z_max - z_min) / 2)
    zc = floor_z + (z_min + z_max) / 2
    rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
    blocked = [False]

    def report(hit) -> bool:
        path = str(getattr(hit, "rigid_body", "") or getattr(hit, "collision", ""))
        if any(path.startswith(p) for p in ignore_prefixes):
            return True
        blocked[0] = True
        return False

    t0 = time.perf_counter()
    for r in range(ny):
        y = ymin + (r + 0.5) * res
        for c in range(nx):
            blocked[0] = False
            sq.overlap_box(half, carb.Float3(xmin + (c + 0.5) * res, y, zc), rot, report, False)
            if blocked[0]:
                occ[r, c] = 1
    log(f"[occupancy] {ny}x{nx} cells at {res} m in {time.perf_counter() - t0:.1f} s, "
        f"{int(occ.sum())} blocked")
    return occ, [float(xmin), float(ymin)]


def save(path: str, occ: np.ndarray, origin, res: float, robot_radius: float, source: str, **extra) -> dict:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    occ_inf = inflate(occ, robot_radius, res)
    np.savez_compressed(path, occ=occ.astype(np.uint8), occ_inflated=occ_inf.astype(np.uint8),
                        resolution=np.float64(res), origin=np.asarray(origin, np.float64),
                        robot_radius=np.float64(robot_radius), **extra)
    return {"path": path, "resolution": res, "origin": list(origin), "shape": list(occ.shape),
            "robot_radius": robot_radius, "source": source, "blocked_cells": int(occ.sum()),
            "blocked_cells_inflated": int(occ_inf.sum())}
