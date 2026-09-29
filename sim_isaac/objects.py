"""ObjectTracker: live object poses, attach/detach (STEPPING STONE), scene reset and test-only object writes for P1
(docs/contracts/p1_m2b.md §3, §4, §8, §10.2).

Dynamic objects are the house props with a rigid body (`scenes/loader.py` ObjectInfo.body_path, non-articulated).
One PhysX rigid-body tensor view (omni.physics.tensors, the API Isaac Lab's RigidObject uses) reads and writes all
of them; static and articulated furniture keep their load-time pose.

Attach modes:
  follow       every physics step, before `sim.step`, the held body's pose is written as palm x grip offset and
               its velocity zeroed. Its colliders are disabled (USD `physics:collisionEnabled`), so it cannot push
               the hand, the robot or furniture. No stage structure changes.
  fixed_joint  a UsdPhysics.FixedJoint (excluded from the articulation) between `<arm>_wrist_yaw_link` and the
               body, created at attach and removed at detach; colliders disabled as above; gravity acts, so the
               object's mass hangs on the arm. Structural stage change: the tracker checks both tensor views after
               every attach/detach and switches the mode off for the run if one was invalidated.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sim_isaac.wire import (ARMS, ATTACH_MODES, GRIP_OFFSET, PALM_OFFSET, SNAP_M, FallDetector, OpError,
                            body_local_points, body_world_points, box_center, box_corners, compose, moved_box,
                            moved_point, object_record, parse_pose_arg, placed_pose, relative)

IDENT_Q = np.array([1.0, 0.0, 0.0, 0.0])


def palm_pose(st: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Palm frame (contract §4.1) from the wrist_yaw_link pose in a P1 state dict."""
    return compose(st[f"{arm}_wrist_pos"], st[f"{arm}_wrist_quat"], PALM_OFFSET[arm], IDENT_Q)


@dataclass
class Hold:
    arm: str
    mode: str
    off_p: np.ndarray          # body pose in the palm frame
    off_q: np.ndarray
    joint_path: str | None = None
    t_attach: float = 0.0


class ObjectTracker:
    def __init__(self, stage, objects: list[dict], floor_z: float, *, device: str = "cpu",
                 log: Callable[[str], None] = print, event: Callable[..., None] | None = None):
        self.stage = stage
        self.log = log
        self.event = event or (lambda *a, **k: None)
        self.device = device
        self.floor_z = float(floor_z)
        self.objs = {str(o["id"]): o for o in objects}
        dyn = [o for o in objects if o.get("body_path") and not o.get("articulated") and not o.get("is_static")]
        self.dyn_ids = [str(o["id"]) for o in dyn]
        self.held: dict[str, Hold] = {}
        self.fall = FallDetector(floor_z=self.floor_z)
        self.attach_count = self.detach_count = self.object_writes = 0
        self.view_errors: list[str] = []
        self.fixed_joint_disabled: str | None = None
        self._colliders: dict[str, list] = {}
        self._joint_n = 0
        self._cache: tuple[int, np.ndarray, np.ndarray] | None = None
        self.view = None
        self.index_of: dict[str, int] = {}
        if not self.dyn_ids:
            self.p0 = np.zeros((0, 3))
            self.q0 = np.zeros((0, 4))
            return
        import omni.physics.tensors.impl.api as physx
        import torch

        self.torch = torch
        self.sim_view = physx.create_simulation_view("torch")
        self.sim_view.set_subspace_roots("/")
        paths = [str(self.objs[i]["body_path"]) for i in self.dyn_ids]
        self.view = self.sim_view.create_rigid_body_view(paths)
        vp = list(getattr(self.view, "prim_paths", []) or [])
        if vp and len(vp) == len(paths):
            order = {p: k for k, p in enumerate(vp)}
            self.index_of = {oid: order[str(self.objs[oid]["body_path"])] for oid in self.dyn_ids}
        else:
            self.index_of = {oid: k for k, oid in enumerate(self.dyn_ids)}
        if self.view.count != len(paths):
            raise RuntimeError(f"rigid body view has {self.view.count} bodies for {len(paths)} paths")
        T = self.view.get_transforms().detach().cpu().numpy().astype(np.float64)
        self.p0 = T[:, 0:3].copy()
        self.q0 = T[:, [6, 3, 4, 5]].copy()          # PhysX xyzw -> wxyz
        self._n = self.view.count
        # body-frame box corners and root origin (index order): every read moves them in one batched einsum
        order = sorted(self.dyn_ids, key=lambda i: self.index_of[i])
        corners = np.stack([box_corners(self.objs[i]["aabb"]) for i in order])
        roots = np.array([[float(v) for v in self.objs[i]["pos"]] for i in order])[:, None, :]
        self._lc = body_local_points(corners, self.p0, self.q0)
        self._lroot = body_local_points(roots, self.p0, self.q0)
        self._all_idx = torch.arange(self._n, dtype=torch.int32, device=device)
        for oid in self.dyn_ids:        # props rest at their load pose: a later drop from there is an object_fell
            self.fall.seed(oid, float(self.objs[oid]["aabb"][0][2]))
        log(f"[objects] {len(self.dyn_ids)} dynamic props in one rigid-body view "
            f"({len(self.objs) - len(self.dyn_ids)} static/articulated keep their load pose)")

    # ------------------------------------------------------------------ reading
    def _read(self, step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(T (N,7) pos + quat wxyz, V (N,6) lin + ang) of every dynamic body; cached per physics step."""
        return self._read_all(step)[:2]

    def _read_all(self, step: int | None = None):
        """T, V, box lo/hi (N,3) and root origin (N,3) of every dynamic body (index order), cached per step."""
        if self.view is None:
            z = np.zeros((0, 3))
            return np.zeros((0, 7)), np.zeros((0, 6)), z, z, z
        if step is not None and self._cache is not None and self._cache[0] == step:
            return self._cache[1]
        T = self.view.get_transforms().detach().cpu().numpy().astype(np.float64)
        V = self.view.get_velocities().detach().cpu().numpy().astype(np.float64)
        T = np.concatenate([T[:, 0:3], T[:, [6, 3, 4, 5]]], axis=1)
        C = body_world_points(self._lc, T[:, 0:3], T[:, 3:7])
        root = body_world_points(self._lroot, T[:, 0:3], T[:, 3:7])[:, 0, :]
        out = (T, V, C.min(axis=1), C.max(axis=1), root)
        if step is not None:
            self._cache = (step, out)
        return out

    def _pose_of(self, oid: str, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self.index_of[oid]
        return T[k, 0:3], T[k, 3:7]

    def _box(self, oid: str, step: int | None = None) -> list:
        _T, _V, lo, hi, _r = self._read_all(step)
        k = self.index_of[oid]
        return [lo[k].tolist(), hi[k].tolist()]

    def record(self, oid: str, data) -> dict:
        o = self.objs[oid]
        held = self.held.get(oid)
        if oid in self.index_of:
            T, V, lo, hi, root = data
            k = self.index_of[oid]
            return object_record(oid, o.get("name") or oid, root[k], T[k, 3:7], (lo[k], hi[k]),
                                 held_by=held.arm if held else None, dynamic=True, lin_vel=V[k, 0:3])
        return object_record(oid, o.get("name") or oid, o["pos"], IDENT_Q, o["aabb"], held_by=None, dynamic=False)

    def get_objects(self, step: int, ids=None, dynamic_only: bool = False) -> list[dict]:
        data = self._read_all(step)
        want = [str(i) for i in ids] if ids else list(self.objs)
        unknown = [i for i in want if i not in self.objs]
        if unknown:
            raise OpError("unknown_object", f"{unknown[:5]}")
        if dynamic_only:
            want = [i for i in want if i in self.index_of]
        return [self.record(i, data) for i in want]

    def dynamic_records(self, step: int) -> list[dict]:
        data = self._read_all(step)
        return [self.record(i, data) for i in self.dyn_ids]

    def centres(self, step: int) -> dict[str, np.ndarray]:
        _T, _V, lo, hi, _r = self._read_all(step)
        out = {oid: box_center(o["aabb"]) for oid, o in self.objs.items()}
        for oid, k in self.index_of.items():
            out[oid] = (lo[k] + hi[k]) / 2.0
        return out

    def fell_events(self, step: int, t_sim: float) -> list[dict]:
        _T, V, lo, hi, _r = self._read_all(step)
        out = []
        for oid, k in self.index_of.items():
            ev = self.fall.update(oid, t_sim, float(lo[k, 2]), float(np.linalg.norm(V[k, 0:3])), oid in self.held,
                                  pos=(lo[k] + hi[k]) / 2.0)
            if ev:
                out.append(ev)
        return out

    # ------------------------------------------------------------------ writing
    def _write(self, rows: dict[str, tuple[np.ndarray, np.ndarray]], vel: dict[str, np.ndarray] | None = None) -> None:
        """Set poses (and velocities; zero when not given) of some dynamic bodies."""
        torch = self.torch
        T, V = self._read(None)
        buf = np.concatenate([T[:, 0:3], T[:, [4, 5, 6, 3]]], axis=1).astype(np.float32)   # back to xyzw
        vbuf = V.astype(np.float32)
        ks = []
        for oid, (p, q) in rows.items():
            k = self.index_of[oid]
            buf[k, 0:3] = p
            buf[k, 3:7] = [q[1], q[2], q[3], q[0]]
            vbuf[k] = 0.0
            if vel and oid in vel:
                vbuf[k, 0:3] = vel[oid]
            ks.append(k)
        idx = torch.tensor(ks, dtype=torch.int32, device=self.device)
        self.view.set_transforms(torch.from_numpy(buf).to(self.device), idx)
        self.view.set_velocities(torch.from_numpy(vbuf).to(self.device), idx)
        self._cache = None

    def _set_velocity(self, oid: str, v) -> None:
        torch = self.torch
        _, V = self._read(None)
        vbuf = V.astype(np.float32)
        k = self.index_of[oid]
        vbuf[k, 0:3] = v
        self.view.set_velocities(torch.from_numpy(vbuf).to(self.device),
                                 torch.tensor([k], dtype=torch.int32, device=self.device))
        self._cache = None

    def _collider_prims(self, oid: str) -> list:
        if oid not in self._colliders:
            from pxr import Usd, UsdPhysics

            o = self.objs[oid]
            prims = []
            for root in [o.get("prim_path")] + list(o.get("extra_prims") or []):
                p = self.stage.GetPrimAtPath(str(root)) if root else None
                if p is None or not p.IsValid():
                    continue
                prims += [x for x in Usd.PrimRange(p) if x.HasAPI(UsdPhysics.CollisionAPI)]
            self._colliders[oid] = prims
        return self._colliders[oid]

    def _set_colliders(self, oid: str, enabled: bool) -> int:
        from pxr import UsdPhysics

        n = 0
        for p in self._collider_prims(oid):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(bool(enabled))
            n += 1
        return n

    def views_ok(self, robot_view=None) -> bool:
        ok = True
        for name, v in (("objects", self.view), ("robot", robot_view)):
            if v is None:
                continue
            chk = getattr(v, "check", None)
            try:
                good = bool(chk()) if callable(chk) else True
            except Exception as e:  # noqa: BLE001
                good = False
                self.view_errors.append(f"{name}: check() raised {e!r}")
            if not good:
                ok = False
                self.view_errors.append(f"{name} view invalid at {time.strftime('%H:%M:%S')}")
        return ok

    # ------------------------------------------------------------------ per physics step
    def pre_step(self, st: dict) -> None:
        """follow mode: move every held body to its palm x offset (before sim.step)."""
        if not self.held:
            return
        rows = {}
        for oid, h in self.held.items():
            if h.mode != "follow":
                continue
            pp, pq = palm_pose(st, h.arm)
            rows[oid] = compose(pp, pq, h.off_p, h.off_q)
        if rows:
            self._write(rows)

    # ------------------------------------------------------------------ ops
    def _need_dynamic(self, oid: str) -> None:
        if oid not in self.objs:
            raise OpError("unknown_object", repr(oid))
        if oid not in self.index_of:
            raise OpError("not_movable", f"{oid} is static or articulated")

    def attach(self, st: dict, oid: str, arm: str, mode: str = "follow", offset=None, snap: bool = True,
               snap_m: float = SNAP_M, robot_view=None) -> dict:
        oid = str(oid)
        self._need_dynamic(oid)
        if arm not in ARMS:
            raise OpError("bad_arg", f"arm must be one of {ARMS}")
        if mode not in ATTACH_MODES:
            raise OpError("bad_arg", f"mode must be one of {ATTACH_MODES}")
        if mode == "fixed_joint" and self.fixed_joint_disabled:
            raise OpError("mode_disabled", f"fixed_joint disabled for this run: {self.fixed_joint_disabled}")
        cur = self.held.get(oid)
        if cur is not None and cur.arm != arm:
            raise OpError("held_by_other", f"{oid} is held by the {cur.arm} hand")
        busy = [i for i, h in self.held.items() if h.arm == arm and i != oid]
        if busy:
            raise OpError("hand_busy", f"the {arm} hand holds {busy[0]}")
        if cur is not None:                       # re-attach: release the previous hold first (joint, flags)
            self._release(oid, cur)
        off = np.asarray(offset if offset is not None else GRIP_OFFSET, dtype=np.float64)
        pp, pq = palm_pose(st, arm)
        grip, _ = compose(pp, pq, off, IDENT_Q)
        T, _V = self._read(None)
        p, q = self._pose_of(oid, T)
        centre = box_center(self._box(oid))
        dist = float(np.linalg.norm(centre - grip))
        snapped = bool(snap) and dist > float(snap_m)
        if snapped:
            p = p + (grip - centre)
        self._write({oid: (p, q)})
        n_col = self._set_colliders(oid, False)
        off_p, off_q = relative(pp, pq, p, q)
        hold = Hold(arm=arm, mode=mode, off_p=off_p, off_q=off_q, t_attach=time.monotonic())
        if mode == "fixed_joint":
            hold.joint_path = self._make_joint(oid, arm, off_p, off_q)
        self.held[oid] = hold
        self.attach_count += 1
        self.fall.reset(oid)
        if not self.views_ok(robot_view) and mode == "fixed_joint":
            self.fixed_joint_disabled = self.view_errors[-1]
        self.event("attach", id=oid, arm=arm, mode=mode, snapped=snapped)
        return {"id": oid, "arm": arm, "mode": mode, "held_by": arm, "snapped": snapped, "dist_m": round(dist, 4),
                "grip_point": [round(float(v), 4) for v in grip], "colliders_disabled": n_col,
                "stepping_stone": True}

    def _make_joint(self, oid: str, arm: str, off_p: np.ndarray, off_q: np.ndarray) -> str:
        from pxr import Gf, Sdf, UsdPhysics

        # joint frame on the wrist link = palm offset x (object pose in the palm frame); on the body: its origin
        lp0, lq0 = compose(PALM_OFFSET[arm], IDENT_Q, off_p, off_q)
        self._joint_n += 1
        path = f"/World/wl_grip/{arm}_{self._joint_n}"
        if not self.stage.GetPrimAtPath("/World/wl_grip").IsValid():
            self.stage.DefinePrim("/World/wl_grip", "Scope")
        j = UsdPhysics.FixedJoint.Define(self.stage, path)
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/G1/{arm}_wrist_yaw_link")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(str(self.objs[oid]["body_path"]))])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in lp0]))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(float(lq0[0]), float(lq0[1]), float(lq0[2]), float(lq0[3])))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateExcludeFromArticulationAttr().Set(True)
        return path

    def _release(self, oid: str, h: Hold) -> None:
        if h.joint_path:
            try:
                self.stage.RemovePrim(h.joint_path)
            except Exception as e:  # noqa: BLE001
                self.view_errors.append(f"remove joint {h.joint_path}: {e!r}")
        self._set_colliders(oid, True)

    def detach(self, oid: str, pose=None, robot_view=None) -> dict:
        oid = str(oid)
        self._need_dynamic(oid)
        h = self.held.pop(oid, None)
        if h is not None:
            self._release(oid, h)
            self.detach_count += 1
        placed = pose is not None
        k = self.index_of[oid]
        o = self.objs[oid]
        if placed:
            centre, yaw = parse_pose_arg(pose)
            p, q = placed_pose(o["aabb"], self.p0[k], self.q0[k], centre, yaw)
        else:
            T, _ = self._read(None)
            p, q = self._pose_of(oid, T)
        self._write({oid: (p, q)})
        self.fall.reset(oid)
        if h is not None and h.mode == "fixed_joint" and not self.views_ok(robot_view):
            self.fixed_joint_disabled = self.view_errors[-1]
        self.event("detach", id=oid, arm=h.arm if h else None, placed=placed)
        aabb = moved_box(o["aabb"], self.p0[k], self.q0[k], p, q)
        return {"id": oid, "was_held": h is not None, "placed": placed, "held_by": None,
                "pos": [round(float(v), 4) for v in moved_point(o["pos"], self.p0[k], self.q0[k], p, q)],
                "aabb": [[round(float(v), 4) for v in aabb[0]], [round(float(v), 4) for v in aabb[1]]]}

    def release_all(self) -> list[str]:
        out = []
        for oid in list(self.held):
            self.detach(oid, None)
            out.append(oid)
        return out

    def move(self, oid: str, pose, vel=None, by: str = "move_object") -> dict:
        oid = str(oid)
        self._need_dynamic(oid)
        if oid in self.held:
            self.detach(oid, None)
        centre, yaw = parse_pose_arg(pose)
        k = self.index_of[oid]
        o = self.objs[oid]
        p, q = placed_pose(o["aabb"], self.p0[k], self.q0[k], centre, yaw)
        self._write({oid: (p, q)}, {oid: np.asarray(vel, dtype=np.float64)} if vel is not None else None)
        self.object_writes += 1
        self.fall.reset(oid)
        self.event("object_moved", id=oid, by=by)
        aabb = moved_box(o["aabb"], self.p0[k], self.q0[k], p, q)
        return {"id": oid, "pos": [round(float(v), 4) for v in moved_point(o["pos"], self.p0[k], self.q0[k], p, q)],
                "aabb": [[round(float(v), 4) for v in aabb[0]], [round(float(v), 4) for v in aabb[1]]]}

    def push(self, oid: str, vel) -> dict:
        oid = str(oid)
        self._need_dynamic(oid)
        if oid in self.held:
            raise OpError("bad_arg", f"{oid} is held; detach it first")
        v = [float(x) for x in vel]
        if len(v) != 3:
            raise OpError("bad_arg", "vel needs [vx, vy, vz]")
        self._set_velocity(oid, v)
        self.object_writes += 1
        self.event("object_moved", id=oid, by="push_object")
        return {"id": oid, "vel": v}

    def reset(self, poses: dict | None = None) -> dict:
        released = self.release_all()
        rows = {oid: (self.p0[self.index_of[oid]], self.q0[self.index_of[oid]]) for oid in self.dyn_ids}
        if rows:
            self._write(rows)
        applied = []
        for oid, pose in (poses or {}).items():
            self.move(str(oid), pose, by="reset_scene")
            applied.append(str(oid))
        self.fall.reset()
        for oid in self.dyn_ids:
            if oid not in applied:
                self.fall.seed(oid, float(self.objs[oid]["aabb"][0][2]))
        return {"objects_reset": len(rows), "poses_applied": applied, "released": released}

    def hide_paths(self) -> list[str]:
        return [str(self.objs[i]["prim_path"]) for i in self.dyn_ids] + \
            [str(p) for i in self.dyn_ids for p in (self.objs[i].get("extra_prims") or [])]

    def held_map(self) -> dict[str, str | None]:
        out: dict[str, Any] = {"left": None, "right": None}
        for oid, h in self.held.items():
            out[h.arm] = oid
        return out

    def stats(self) -> dict:
        return {"dynamic_objects": len(self.dyn_ids), "held": self.held_map(), "attach_count": self.attach_count,
                "detach_count": self.detach_count, "object_writes": self.object_writes,
                "view_errors": self.view_errors[-5:], "n_view_errors": len(self.view_errors),
                "fixed_joint_disabled": self.fixed_joint_disabled}


def quat_angle_deg(q_a, q_b) -> float:
    d = abs(float(np.dot(np.asarray(q_a), np.asarray(q_b))))
    return math.degrees(2 * math.acos(min(1.0, d)))
