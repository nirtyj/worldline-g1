"""Where G6's box goes: across the robot's own planned route, at its narrowest point (a doorway or corridor).

The body plans go_to with A* on P1's occupancy (body/nav_grid.py). place_across_path plans the same way from the
robot to the goal, walks the path from `min_from_start` to `min_from_goal` before the end, and tries the narrowest
points first: the box (thin side along the path, long side across it) is drawn into a copy of the raw grid, and the
first spot where A* then finds NO path is chosen, so no other route to the goal exists and the walk has to end
`blocked`. When no spot cuts every route (e.g. two doors into the room), the narrowest spot the box spans is used and
the result says blocks_all_routes: false (the body may then walk around it after its replan, and G6 fails honestly).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from body.nav_grid import NavGrid


def rect_mask(grid: NavGrid, x: float, y: float, yaw: float, size_xy: tuple[float, float],
              margin: float = 0.02) -> np.ndarray:
    """Cells whose centres lie inside the yaw-rotated rectangle (size along x, y of the box) plus a margin."""
    yy, xx = np.mgrid[0:grid.H, 0:grid.W]
    cx = grid.origin[0] + (xx + 0.5) * grid.res - x
    cy = grid.origin[1] + (yy + 0.5) * grid.res - y
    c, s = math.cos(yaw), math.sin(yaw)
    u = c * cx + s * cy                      # along the box's x (the path)
    v = -s * cx + c * cy                     # across
    return (np.abs(u) <= size_xy[0] / 2 + margin) & (np.abs(v) <= size_xy[1] / 2 + margin)


def with_box(grid: NavGrid, x: float, y: float, yaw: float, size_xy: tuple[float, float]) -> NavGrid:
    raw = grid.raw | rect_mask(grid, x, y, yaw, size_xy)
    return NavGrid(raw, grid.res, grid.origin, robot_radius=grid.robot_radius,
                   already_inflated_m=grid.already_inflated)


def _headings(path: np.ndarray) -> np.ndarray:
    d = np.gradient(path, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def place_across_path(grid: NavGrid, start: tuple[float, float], goal: tuple[float, float],
                      size: tuple[float, float, float], min_from_start: float = 1.2, min_from_goal: float = 1.0,
                      max_tries: int = 40, goal_snap: float = 1.2) -> dict[str, Any]:
    plan = grid.plan(start, goal, snap_radius=goal_snap)
    if not plan.ok:
        raise ValueError(f"no path from {start} to {goal}: {plan.reason}")
    path = plan.path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    idx = [i for i in range(len(path)) if s[i] >= min_from_start and L - s[i] >= min_from_goal]
    if not idx:
        raise ValueError(f"the path is {L:.2f} m: too short for a box {min_from_start} m from the robot and "
                         f"{min_from_goal} m from the goal")
    head = _headings(path)
    clear = np.array([grid.clearance(float(path[i, 0]), float(path[i, 1])) for i in idx])
    order = [idx[j] for j in np.argsort(clear, kind="stable")]
    half = size[1] / 2.0
    spans = [i for i in order if grid.clearance(float(path[i, 0]), float(path[i, 1])) + 0.05 <= half]
    tried = []
    for i in (spans or order)[:max_tries]:
        x, y, yaw = float(path[i, 0]), float(path[i, 1]), float(head[i])
        g2 = with_box(grid, x, y, yaw, (size[0], size[1]))
        p2 = g2.plan(start, goal, snap_radius=goal_snap)
        tried.append(i)
        if not p2.ok:
            return _placed(x, y, yaw, grid, s[i], L, True, len(tried), plan, p2.reason)
    i = (spans or order)[0]
    return _placed(float(path[i, 0]), float(path[i, 1]), float(head[i]), grid, s[i], L, False, len(tried), plan,
                   "a detour exists")


def _placed(x, y, yaw, grid, s_i, L, blocks, tries, plan, why) -> dict[str, Any]:
    return {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4),
            "clearance_m": round(grid.clearance(x, y), 3), "along_m": round(float(s_i), 2),
            "path_m": round(float(L), 2), "blocks_all_routes": bool(blocks), "tries": tries,
            "replan_with_box": why, "path_start": [round(float(v), 3) for v in plan.path[0]],
            "path_end": [round(float(v), 3) for v in plan.path[-1]]}
