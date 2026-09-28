"""Walkability checks for a loaded house (Isaac side; PhysX must be initialised).

floor_check      downward PhysX raycasts in free cells: is there a floor collider at floor_z?
door_check       per Doorway/Doorframe object: clear width of the opening on the raw grid
blocker_check    collider-occupied cells that no visual geometry explains (invisible blockers)
settle_check     loose props: displacement after N seconds of physics (stability of the scene)
"""

from __future__ import annotations

import math

import numpy as np

from scenes.occupancy import FREE, OBSTACLE, Occupancy, cell_centres


def floor_check(house, occ: Occupancy, stride: int = 5) -> dict:
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    fz = house.floor_z
    floor_prefixes = (f"{house.root}/Geometry/floor",)
    n = ok = miss = other = 0
    miss_pts = []
    ys_idx, xs_idx = np.nonzero(occ.raw == FREE)
    sel = (ys_idx % stride == 0) & (xs_idx % stride == 0)
    for iy, ix in zip(ys_idx[sel], xs_idx[sel]):
        x, y = occ.cell_to_world(int(iy), int(ix))
        h = sq.raycast_closest(carb.Float3(x, y, fz + 0.3), carb.Float3(0, 0, -1), 1.0)
        n += 1
        if not h["hit"]:
            miss += 1
            if len(miss_pts) < 20:
                miss_pts.append([round(x, 2), round(y, 2)])
        elif h["collision"].startswith(floor_prefixes) and abs(h["position"][2] - fz) < 0.02:
            ok += 1
        else:
            other += 1
    return {
        "rays": n,
        "floor_hits": ok,
        "no_hit": miss,
        "other_hits": other,
        "floor_hit_frac": round(ok / max(n, 1), 4),
        "no_hit_samples": miss_pts,
        "spacing_m": stride * occ.resolution,
    }


def door_check(house, occ: Occupancy) -> list[dict]:
    out = []
    for o in house.objects:
        if o.category not in ("Doorway", "Doorframe", "Door"):
            continue
        (x0, y0, _), (x1, y1, _) = o.aabb
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        along_x = (x1 - x0) >= (y1 - y0)
        # walk along the wall direction from the centre over free raw cells
        def run(sign):
            d = 0.0
            while d < 3.0:
                nd = d + occ.resolution / 2
                px = cx + sign * nd if along_x else cx
                py = cy if along_x else cy + sign * nd
                iy, ix = occ.world_to_cell(px, py)
                if not occ.in_bounds(iy, ix) or occ.raw[iy, ix] != FREE:
                    break
                d = nd
            return d

        iy, ix = occ.world_to_cell(cx, cy)
        centre_free = occ.in_bounds(iy, ix) and occ.raw[iy, ix] == FREE
        width = (run(+1) + run(-1)) if centre_free else 0.0
        out.append(
            {
                "id": o.id,
                "asset_id": o.asset_id,
                "rooms": o.id.split("|")[1:3] if o.id.startswith("door|") else None,
                "centre": [round(cx, 3), round(cy, 3)],
                "centre_free": bool(centre_free),
                "clear_width_m": round(width, 3),
                "robot_passable": bool(width >= 2 * occ.robot_radius + 0.05),
                "centre_inflated_free": bool(occ.in_bounds(iy, ix) and occ.inflated[iy, ix] == 0),
            }
        )
    return out


def visual_mask(house, occ: Occupancy, dilate_m: float = 0.10) -> np.ndarray:
    """Cells covered by any visual (default-purpose) AABB overlapping the occupancy z band."""
    from pxr import UsdGeom

    from scenes.loader import _bbox_cache  # noqa: PLC0415

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cache = _bbox_cache()
    geo = stage.GetPrimAtPath(f"{house.root}/Geometry")
    xs, ys = cell_centres(occ.origin, occ.shape, occ.resolution)
    mask = np.zeros(occ.shape, bool)
    for c in geo.GetChildren():
        if not c.IsA(UsdGeom.Imageable) or c.GetName() == "floor":
            continue
        r = cache.ComputeWorldBound(c).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = r.GetMin(), r.GetMax()
        if mx[2] < occ.z_min - 0.10 or mn[2] > occ.z_max:  # 10 cm slack: props resting/tilting near z_min
            continue
        mx_ = (xs >= mn[0] - dilate_m) & (xs <= mx[0] + dilate_m)
        my_ = (ys >= mn[1] - dilate_m) & (ys <= mx[1] + dilate_m)
        mask[np.ix_(my_, mx_)] = True
    return mask


def blocker_check(house, occ: Occupancy) -> tuple[dict, np.ndarray]:
    vm = visual_mask(house, occ)
    unexplained = (occ.raw == OBSTACLE) & ~vm
    ys_idx, xs_idx = np.nonzero(unexplained)
    samples = [list(map(lambda v: round(v, 2), occ.cell_to_world(int(a), int(b)))) for a, b in list(zip(ys_idx, xs_idx))[:15]]
    return {
        "obstacle_cells": int((occ.raw == OBSTACLE).sum()),
        "unexplained_obstacle_cells": int(unexplained.sum()),
        "unexplained_area_m2": round(float(unexplained.sum() * occ.resolution**2), 4),
        "samples_xy": samples,
        "note": "obstacle cells (PhysX colliders in the z band) outside every visual AABB dilated by 0.10 m",
    }, unexplained


def body_poses(paths: list[str]) -> dict:
    import omni.physx

    px = omni.physx.get_physx_interface()
    out = {}
    for p in paths:
        t = px.get_rigidbody_transformation(p)
        if t and t.get("ret_val"):
            out[p] = np.asarray(t["position"], float)
    return out


def settle_check(house, sim, seconds: float = 5.0) -> dict:
    paths = [o.body_path for o in house.objects if o.body_path]
    before = body_poses(paths)
    n = int(round(seconds / sim.get_physics_dt()))
    for _ in range(n):
        sim.step(render=False)
    after = body_poses(paths)
    moved, fell = [], []
    for p, a in before.items():
        if p not in after:
            continue
        d = float(np.linalg.norm(after[p] - a))
        dz = float(after[p][2] - a[2])
        name = next((o.name for o in house.objects if o.body_path == p), p)
        if d > 0.05:
            moved.append({"name": name, "disp_m": round(d, 3), "dz_m": round(dz, 3)})
        if dz < -0.2:
            fell.append(name)
    return {"seconds": seconds, "tracked_bodies": len(before), "moved_gt_5cm": moved, "fell_gt_20cm": fell}


def awake_bodies(house, sim, seconds: float = 0.5, tol_m: float = 2e-4, tol_rad: float = 2e-3) -> dict:
    """Rigid bodies (props and articulation links) still moving after settling: they keep PhysX
    busy every step. Identifies what makes a house expensive to simulate."""
    import omni.physx
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    px = omni.physx.get_physx_interface()
    paths = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(house.root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]

    def snap():
        out = {}
        for p in paths:
            t = px.get_rigidbody_transformation(p)
            if t and t.get("ret_val"):
                out[p] = (np.asarray(t["position"], float), np.asarray(t["rotation"], float))
        return out

    a = snap()
    for _ in range(int(round(seconds / sim.get_physics_dt()))):
        sim.step(render=False)
    b = snap()
    movers = []
    for p, (pa, qa) in a.items():
        if p not in b:
            continue
        pb, qb = b[p]
        dp = float(np.linalg.norm(pb - pa))
        dq = float(2 * math.acos(min(1.0, abs(float(np.dot(qa, qb))))))
        if dp > tol_m or dq > tol_rad:
            owner = next((o.name for o in house.objects if p.startswith(o.prim_path + "/") or p == o.prim_path), p.rsplit("/", 1)[-1])
            movers.append({"body": p.replace(house.root + "/Geometry/", ""), "object": owner, "dp_m": round(dp, 5), "drot_rad": round(dq, 4)})
    movers.sort(key=lambda m: -(m["dp_m"] + m["drot_rad"]))
    return {"bodies": len(paths), "readable": len(a), "moving": len(movers), "window_s": seconds, "top": movers[:15]}
