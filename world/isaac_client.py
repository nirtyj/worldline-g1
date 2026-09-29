"""IsaacGTWorldModel: the WorldModel backed by P1 wl-isaac (docs/contracts/m1.md §1.5-1.6).

    REQ -> REP 5600   get_scene_info, get_occupancy, render_topdown, ping, get_stats, band, reset_robot
                      (+ the M2b ops this module already speaks when a P1 build offers them: get_objects,
                       attach, detach; see M2B_OPS)
    SUB 5601          gt.pose at 50 Hz (robot pose, fallen, rtf)

The REQ client and the gt.pose subscriber are wl-body's (`body.p1_client.P1Rpc`, `PoseSub`), imported, not copied,
so both sides parse the contract identically.

What M1's P1 does NOT provide yet, and what this client does instead (labelled in every output):
  * live object poses: objects keep their `get_scene_info` poses (`pose_source: "scene"`); attach/detach move them
    locally (`"attach"`). With a P1 `get_objects` op, poses come from PhysX (`"sim"`).
  * attach / detach: `SimControl.attach/detach` raise NotSupported unless P1 lists the ops (ping `ops`), so the
    `kinematic_attach` executor reports `controller_unavailable` instead of pretending.
  * visibility from instance segmentation: GT geometry is used (world/perception.py, method `gt-geometric`).
  * the torso/camera pose: derived from the pelvis gt.pose at nominal waist angles.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from body.config import ep, ports as body_ports
from body.p1_client import P1Error, P1Rpc, PoseSub

from .gt_world import GTWorld
from .lite_world import find_house
from .mapgen import MapParams
from .model import NotSupported, RobotPose, xyyaw, xyz
from .nav_grid import OccupancyData
from .perception import CameraModel, EGO_D435
from .scene import SceneData

M2B_OPS = ("get_objects", "attach", "detach")
OBJECT_REFRESH_S = 0.1


class IsaacGTWorldModel(GTWorld):
    source = "isaac-gt"
    perception_source = "isaac-ground-truth"

    def __init__(self, *, port_offset: int | None = None, host: str = "127.0.0.1", scene_key: str | None = None,
                 map_params: MapParams | None = None, camera: CameraModel | str = EGO_D435,
                 user_surface: str | None = None, rpc_timeout_s: float = 5.0, pose_wait_s: float = 5.0,
                 local_houses: list[str | Path] | None = None):
        off = int(os.environ.get("WL_PORT_OFFSET", "0")) if port_offset is None else port_offset
        self.ports = body_ports(off)
        self.rpc = P1Rpc(ep(self.ports["p1_rep"], host), timeout_s=rpc_timeout_s)
        info = self._ok(self.rpc.call("get_scene_info", timeout_s=10.0), "get_scene_info")
        mp = map_params or MapParams()
        occ_rep = self._ok(self.rpc.call("get_occupancy", robot_radius=mp.robot_radius, timeout_s=15.0),
                           "get_occupancy")
        scene = SceneData.from_scene_info(info)
        occ = self._load_occupancy(occ_rep, scene.house_id, local_houses)
        topdown = None
        try:
            td = self.rpc.call("render_topdown", timeout_s=5.0)
            if td.get("ok"):
                topdown = topdown_from_render(td)
                self.topdown_render = td
        except P1Error:
            self.topdown_render = None
        super().__init__(scene, occ, scene_key=scene_key, map_params=mp, camera=camera, user_surface=user_surface,
                         topdown=topdown)
        self.pose_sub = PoseSub(ep(self.ports["p1_pose"], host))
        self.pose_sub.start()
        self._ops = self._probe_ops()
        self._obj_t = 0.0
        self._by_scene_id = {spec.scene_id: oid for oid, spec in self.map.objects.items()}
        t0 = time.monotonic()
        while self.pose_sub.latest() is None and time.monotonic() - t0 < pose_wait_s:
            time.sleep(0.02)

    # ------------------------------------------------------------------ plumbing
    @staticmethod
    def _ok(rep: dict, op: str) -> dict:
        if not rep.get("ok", True):
            raise P1Error(f"P1 {op}: {rep.get('error')}")
        return rep

    @staticmethod
    def _load_occupancy(rep: dict, house_id: str, local_houses) -> OccupancyData:
        path = rep.get("path") or rep.get("npz")
        if path and Path(path).exists():
            return OccupancyData.from_p1_reply(rep)
        # P5 off the box (laptop dev): use a local mirror of assets/houses/<id>/occupancy.npz
        d = find_house(house_id, local_houses)
        return OccupancyData.from_npz(d / "occupancy.npz", source=f"local-mirror:{d}")

    def _probe_ops(self) -> set[str]:
        ops: set[str] = set()
        try:
            rep = self.rpc.call("ping", timeout_s=2.0)
            ops |= set(rep.get("ops") or [])
        except P1Error:
            pass
        if "get_objects" not in ops:
            try:
                if self.rpc.call("get_objects", timeout_s=2.0).get("ok"):
                    ops.add("get_objects")
            except P1Error:
                pass
        return ops

    def close(self) -> None:
        self.pose_sub.stop()
        self.rpc.close()

    # ------------------------------------------------------------------ pose / localizer
    def robot_pose(self) -> RobotPose:
        p = self.pose_sub.latest()
        if p is None:
            sx, sy, syaw = self.scene_data.spawn
            return RobotPose(sx, sy, self.map.floor_z + 0.78, syaw, 0.0, "isaac-gt:none")
        vx, vy, wz = self.pose_sub.velocity()
        return RobotPose(p.x, p.y, p.z, p.yaw, p.t_wall or time.time(), "isaac-gt", fallen=p.fallen,
                         pelvis_z=p.pelvis_z, vx=vx, vy=vy, wz=wz)

    def pose_age_s(self) -> float:
        return self.pose_sub.age_s()

    def rtf(self) -> float | None:
        p = self.pose_sub.latest()
        return None if p is None or math.isnan(p.rtf) else p.rtf

    # ------------------------------------------------------------------ live object poses (M2b)
    def _refresh_objects(self) -> None:
        if "get_objects" not in self._ops or time.monotonic() - self._obj_t < OBJECT_REFRESH_S:
            return
        self._obj_t = time.monotonic()
        try:
            rep = self.rpc.call("get_objects", timeout_s=1.0)
        except P1Error:
            return
        for o in rep.get("objects") or []:
            oid = self._by_scene_id.get(str(o.get("id")))
            if oid is None or not o.get("aabb"):
                continue
            box = (tuple(float(v) for v in o["aabb"][0]), tuple(float(v) for v in o["aabb"][1]))
            self._set_object_box(oid, box, source="sim")

    def object(self, object_id: str):
        self._refresh_objects()
        return super().object(object_id)

    def detections(self, camera: str = "head", **kw):
        self._refresh_objects()
        return super().detections(camera, **kw)

    def truth(self):
        self._refresh_objects()
        return super().truth()

    # ------------------------------------------------------------------ SimControl
    def capabilities(self) -> dict[str, bool]:
        return {"teleport_robot": False, "attach": "attach" in self._ops, "detach": "detach" in self._ops,
                "band": True, "reset_robot": True, "object_poses": "get_objects" in self._ops}

    def attach(self, object_id: str, arm: str, mode: str = "follow") -> None:
        if "attach" not in self._ops:
            raise NotSupported("P1 has no `attach` op (M2b: attach {id, arm, mode})")
        spec = self.map.objects[object_id]
        self._ok(self.rpc.call("attach", id=spec.scene_id, arm=arm, mode=mode, timeout_s=2.0), "attach")
        self._attach_local(object_id, arm, mode)

    def detach(self, object_id: str, pose=None) -> None:
        pose = xyz(pose)
        if "detach" not in self._ops:
            raise NotSupported("P1 has no `detach` op (M2b: detach {id, pose?})")
        spec = self.map.objects[object_id]
        args: dict[str, Any] = {"id": spec.scene_id}
        if pose is not None:
            args["pose"] = [float(v) for v in pose]
        self._ok(self.rpc.call("detach", timeout_s=2.0, **args), "detach")
        self._detach_local(object_id, pose)

    def band(self, on: bool, ramp_s: float = 1.5) -> None:
        self._ok(self.rpc.call("band", on=bool(on), ramp_s=float(ramp_s)), "band")

    def reset_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        x, y, yaw = xyyaw(pose, y, yaw)
        self._ok(self.rpc.call("reset_robot", x=x, y=y, yaw=yaw, timeout_s=10.0), "reset_robot")

    def stats(self) -> dict:
        try:
            return self.rpc.call("get_stats", timeout_s=2.0)
        except P1Error as e:
            return {"ok": False, "error": str(e)}


def topdown_from_render(rep: dict) -> dict:
    """P1 render_topdown {center, meters_per_pixel, extent, width, height} -> THOR topdown {cx, cz, size, w, h}
    (ui/index.html:548-551: u = w/2 + (x - cx)k, v = h/2 - (z - cz)k, k = h / (2 size)). Image right = +x and
    image up = +y (contract §1.6), which is the THOR map's +x right / +z up with z = isaac y."""
    w, h = int(rep.get("width", 0)), int(rep.get("height", 0))
    mpp = float(rep.get("meters_per_pixel", 0.0))
    cx, cy = rep.get("center") or [0.0, 0.0]
    return {"cx": float(cx), "cz": float(cy), "size": h * mpp / 2, "w": w, "h": h, "path": rep.get("path"),
            "caption": "static render at P1 start-up (robot visible at spawn)"}
