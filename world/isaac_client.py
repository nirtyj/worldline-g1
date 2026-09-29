"""IsaacGTWorldModel: the WorldModel backed by P1 wl-isaac (docs/contracts/m1.md §1.5-1.6; M2b: p1_m2b.md).

    REQ -> REP 5600   M1: get_scene_info, get_occupancy, render_topdown, ping, get_stats, band, reset_robot
                      M2b (when ping lists them, p1_m2b.md §2): get_objects, attach, detach, release_all,
                      get_link_poses, detections, camera, set_render_rates, reset_scene, move_object,
                      render_topdown {mode: furniture}
    SUB 5601          gt.pose 50 Hz (robot pose, fallen, rtf; M2b + links {torso_link, left_palm, right_palm})
                      M2b: gt.objects 10 Hz, gt.event (robot_fell, object_fell, ...), sim.health 1 Hz (world/p1_stream)

The REQ client and the gt.pose subscriber are wl-body's (`body.p1_client.P1Rpc`, `PoseSub`), imported, not copied,
so both sides parse the contract identically.

What each P1 build gives, and what this client does without it (labelled in every output):

  P1.2 object poses   gt.objects / get_objects -> every object P1 reports is `pose_source: "sim"`, held objects
                      follow the palm in PhysX. Without: the get_scene_info poses (`"scene"`), attach/detach move
                      them locally (`"attach"`).
  P1.3 attach/detach  SimControl.attach/detach (STEPPING STONE). Without: NotSupported, so kinematic_attach reports
                      controller_unavailable instead of pretending.
  P1.4 / OD1 cameras  the `head` camera (5565, System 1 + scans) the profile's `head_sim` model describes; GR00T's
                      `ego_view` (5566) rendered only while enabled (`enable_camera`). An M1 P1 has no head camera:
                      the world then models the d435 camera it does render, and says so (`camera_note`).
  P1.5 link poses     the camera pose from the real torso link (waist pitch/roll included), palm positions for
                      `where`'s hand rule, grasp_state and GR00T's success check. Without: the pelvis at nominal
                      waist angles, and no palm (None).
  P1.6 detections     `detections(method="best")` (the observation service's glances and scans) uses P1's
                      instance-id segmentation of the live view (`instance_id_segmentation_fast`); GT geometry
                      (`gt-geometric`) otherwise, and always for other callers (it costs P1 an extra render).
  P1.8 top render     the furniture-only top render (no stale props).
  P1.9 health/events  sim.health -> world/sim_health.py (P1's level wins while fresh; else gt.pose's 1 s RTF through
                      the same thresholds); gt.event robot_fell / object_fell -> drain_events() for robot/.
"""

from __future__ import annotations

import collections
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from body.config import ep, ports as body_ports
from body.p1_client import P1Error, P1Rpc, PoseSub

from . import coords
from .gt_world import GTWorld
from .lite_world import find_house
from .mapgen import MapParams
from .model import Detection, GraspState, NotSupported, RobotPose, xyyaw, xyz
from .nav_grid import OccupancyData
from .p1_stream import P1Stream
from .perception import CAMERAS, EGO_D435, HEAD_SIM, P1_CAMERA, CameraModel, CameraPose
from .scene import SceneData
from .sim_health import RtfMonitor, SimHealthConfig

M2B_OPS = ("get_objects", "attach", "detach")
OBJECT_REFRESH_S = 0.1
STREAM_FRESH_S = 1.0          # gt.objects / sim.health / links newer than this are "live"
GRIP_QUIET_S = 0.3            # after our own attach/detach, P1's held_by needs this long to catch up
SEG_CACHE_S = 0.2             # one P1 segmentation render per camera per this interval at most
SEG_POSE_TOL = (0.02, math.radians(2.0))
BOX_TOL_M = 0.005


class IsaacGTWorldModel(GTWorld):
    source = "isaac-gt"
    perception_source = "isaac-ground-truth"

    def __init__(self, *, port_offset: int | None = None, host: str = "127.0.0.1", scene_key: str | None = None,
                 map_params: MapParams | None = None, camera: CameraModel | str = EGO_D435,
                 user_surface: str | None = None, rpc_timeout_s: float = 5.0, pose_wait_s: float = 5.0,
                 local_houses: list[str | Path] | None = None, sim_health_cfg: SimHealthConfig | None = None,
                 objects_hz_poll: float = 1.0 / OBJECT_REFRESH_S):
        off = int(os.environ.get("WL_PORT_OFFSET", "0")) if port_offset is None else port_offset
        self.ports = body_ports(off)
        self.host = host
        self.stream: P1Stream | None = None
        self.rpc = P1Rpc(ep(self.ports["p1_rep"], host), timeout_s=rpc_timeout_s)
        self._ops, self.p1_info = self._probe_ops()
        cam = CAMERAS[camera] if isinstance(camera, str) else camera
        self.camera_note = ""
        if cam.name == HEAD_SIM.name and not self.has_camera("head"):
            # an M1 P1 renders only the d435-mounted camera on 5565: model what System 1 really sees, and say so
            self.camera_note = "P1 has no head camera (M1 contract): perception models the d435 camera it renders"
            cam = EGO_D435
        info = self._ok(self.rpc.call("get_scene_info", timeout_s=10.0), "get_scene_info")
        mp = map_params or MapParams()
        occ_rep = self._ok(self.rpc.call("get_occupancy", robot_radius=mp.robot_radius, timeout_s=15.0),
                           "get_occupancy")
        scene = SceneData.from_scene_info(info)
        occ = self._load_occupancy(occ_rep, scene.house_id, local_houses)
        topdown = None
        self.topdown_render = None
        try:
            args = {"mode": "furniture"} if self.m2b else {}
            td = self.rpc.call("render_topdown", timeout_s=5.0, **args)
            if td.get("ok"):
                topdown = topdown_from_render(td)
                self.topdown_render = td
        except P1Error:
            pass
        super().__init__(scene, occ, scene_key=scene_key, map_params=mp, camera=cam, user_surface=user_surface,
                         topdown=topdown)
        self._by_scene_id = {spec.scene_id: oid for oid, spec in self.map.objects.items()}
        self._landmark_of = {sid: name for name, lm in self.map.landmarks.items() for sid in lm.scene_ids}
        self._obj_t = 0.0
        self._poll_s = 1.0 / max(objects_hz_poll, 0.1)
        self._grip_t = -1e9
        self._events: collections.deque[dict] = collections.deque(maxlen=500)
        self._seg_cache: dict[str, tuple[float, tuple, list[Detection]]] = {}
        self.seg_stats = {"calls": 0, "errors": 0, "last_error": None}
        self.rtf_monitor = RtfMonitor(sim_health_cfg, source="p1:gt.pose")
        self.pose_sub = PoseSub(ep(self.ports["p1_pose"], host), on_pose=self._on_pose)
        self.pose_sub.start()
        if self.m2b:
            self.stream = P1Stream(ep(self.ports["p1_pose"], host),
                                   {"gt.objects": self._on_objects, "gt.event": self._on_event,
                                    "sim.health": self._on_health})
            self.stream.start()
        if "get_objects" in self._ops:
            self._poll_objects(dynamic_only=False)          # every object once: all of them "sim" from here on
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

    def _probe_ops(self) -> tuple[set[str], dict]:
        """ping -> (ops, {p1_contract, cameras, topics}) (p1_m2b.md §2). An M1 P1 lists no ops; get_objects is
        then probed directly (older M2b builds)."""
        ops: set[str] = set()
        info: dict[str, Any] = {}
        try:
            rep = self.rpc.call("ping", timeout_s=2.0)
            ops |= set(rep.get("ops") or [])
            info = {k: rep.get(k) for k in ("p1_contract", "cameras", "topics", "house_id") if k in rep}
        except P1Error:
            pass
        if "get_objects" not in ops:
            try:
                if self.rpc.call("get_objects", timeout_s=2.0).get("ok"):
                    ops.add("get_objects")
            except P1Error:
                pass
        return ops, info

    @property
    def m2b(self) -> bool:
        return bool(self.p1_info.get("p1_contract"))

    def has_camera(self, name: str) -> bool:
        return name in (self.p1_info.get("cameras") or {})

    def close(self) -> None:
        self.pose_sub.stop()
        if self.stream is not None:
            self.stream.stop()
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

    def _on_pose(self, p) -> None:
        """PoseSub thread: gt.pose carries P1's 1 s RTF; P1.9's sim.health (when fresh) supersedes it."""
        if self.stream is not None and self.stream.age_s("sim.health") < 3.0:
            return
        r = getattr(p, "rtf", None)
        if r is not None and not math.isnan(r):
            self.rtf_monitor.update(float(r), source="p1:gt.pose")

    def _live_sim_health(self):
        return self.rtf_monitor.state()

    def _on_health(self, msg: dict) -> None:
        """sim.health (P1.9): P1's level is PLAN §3.5 over its own windows (unsafe = rtf_3s < 0.85, degraded =
        rtf_5s < 0.95); world applies it. rtf_1s is the number shown."""
        r1 = msg.get("rtf_1s")
        self.rtf_monitor.update(float(r1) if isinstance(r1, (int, float)) else None, source="p1:sim.health")
        level = str(msg.get("level") or "")
        if level in ("ok", "degraded", "unsafe"):
            r3, r5 = msg.get("rtf_3s"), msg.get("rtf_5s")
            if level == "unsafe":
                detail = f"sim below real time: UNSAFE, rtf_3s {r3:.2f} (< 0.85)" if isinstance(r3, (int, float)) \
                    else "sim below real time: UNSAFE"
            elif level == "degraded":
                detail = f"sim below real time: DEGRADED, rtf_5s {r5:.2f} (< 0.95)" if isinstance(r5, (int, float)) \
                    else "sim below real time: DEGRADED"
            else:
                detail = f"rtf {r1:.2f}" if isinstance(r1, (int, float)) else "ok"
            self.rtf_monitor.force(level, detail)

    # ------------------------------------------------------------------ link poses (P1.5)
    def _links(self) -> dict | None:
        p = self.pose_sub.latest()
        if p is None or self.pose_sub.age_s() > STREAM_FRESH_S:
            return None
        links = (getattr(p, "raw", None) or {}).get("links")
        return links if isinstance(links, dict) else None

    def link_pose(self, name: str) -> tuple[tuple[float, float, float], tuple[float, ...]] | None:
        lk = (self._links() or {}).get(name)
        if not isinstance(lk, dict) or not lk.get("pos"):
            return None
        return tuple(float(v) for v in lk["pos"][:3]), tuple(float(v) for v in (lk.get("quat_wxyz") or (1, 0, 0, 0)))

    def palm_position(self, arm: str) -> tuple[float, float, float] | None:
        lp = self.link_pose(f"{arm}_palm")
        return lp[0] if lp is not None else None

    def camera_pose(self, *, yaw_offset: float = 0.0, waist_yaw: float = 0.0, waist_pitch: float = 0.0,
                    pose: RobotPose | None = None, camera: str | None = None) -> CameraPose:
        """The live view from the real torso link (P1.5) when there is no view override; the nominal model
        (pelvis + nominal waist) for hypothetical views and on an M1 P1."""
        if yaw_offset or waist_yaw or waist_pitch or pose is not None:
            return super().camera_pose(yaw_offset=yaw_offset, waist_yaw=waist_yaw, waist_pitch=waist_pitch,
                                       pose=pose, camera=camera)
        torso = self.link_pose("torso_link")
        if torso is None:
            return super().camera_pose(camera=camera)
        cam = self.camera_model(camera)
        R_t = np.array(coords.quat_wxyz_to_R(torso[1]))
        R_m = np.array(coords.rot_z(cam.mount_yaw)) @ np.array(coords.rot_y(cam.mount_pitch))
        R = R_t @ R_m
        pos = np.asarray(torso[0]) + R_t @ np.asarray(cam.mount_xyz)
        fwd = R[:, 0]
        return CameraPose(pos=tuple(float(v) for v in pos), R=R, yaw=math.atan2(fwd[1], fwd[0]),   # type: ignore[arg-type]
                          pitch_down=math.asin(max(-1.0, min(1.0, -fwd[2]))))

    def grasp_state(self, object_id: str, arm: str) -> GraspState:
        g = super().grasp_state(object_id, arm)
        palm = self.palm_position(arm)
        o = self.object(object_id)
        if palm is None or o is None:
            return g
        from dataclasses import replace
        return replace(g, palm_dist_m=round(math.dist(o.pos, palm), 3), source="isaac-gt")

    # ------------------------------------------------------------------ live object poses (P1.2)
    def _objects_live(self) -> bool:
        return self.stream is not None and self.stream.age_s("gt.objects") < STREAM_FRESH_S

    def _apply_objects(self, objects: list, *, allow_held_sync: bool = True) -> None:
        quiet = time.monotonic() - self._grip_t < GRIP_QUIET_S
        with self._lock:
            for o in objects or []:
                oid = self._by_scene_id.get(str(o.get("id")))
                if oid is None or not o.get("aabb"):
                    continue
                box = (tuple(float(v) for v in o["aabb"][0]), tuple(float(v) for v in o["aabb"][1]))
                held_by = o.get("held_by")
                if allow_held_sync and not quiet:
                    if held_by in ("left", "right") and self._held.get(oid) != held_by:
                        support = self._boxes[oid][0][2]
                        self._held[oid] = held_by
                        self._grasp[oid] = GraspState(oid, held_by, True, round(box[0][2] - support, 3),
                                                      attach_mode="p1")
                        self.occluders.replace(oid, None)
                        self.rev += 1
                    elif held_by is None and oid in self._held:
                        self._held.pop(oid, None)
                        self._grasp.pop(oid, None)
                        self.rev += 1
                self._set_object_box(oid, box, source="sim", held_too=True)

    def _poll_objects(self, dynamic_only: bool = True) -> None:
        try:
            rep = self.rpc.call("get_objects", timeout_s=1.0, **({"dynamic_only": True} if dynamic_only else {}))
        except P1Error:
            return
        if rep.get("ok", True):
            self._apply_objects(rep.get("objects") or [])

    def _on_objects(self, msg: dict) -> None:
        self._apply_objects(msg.get("objects") or [])

    def _refresh_objects(self) -> None:
        if "get_objects" not in self._ops or self._objects_live():
            return
        if time.monotonic() - self._obj_t < self._poll_s:
            return
        self._obj_t = time.monotonic()
        self._poll_objects(dynamic_only=True)

    def _set_object_box(self, object_id: str, box, source: str = "sim", held_too: bool = False) -> None:
        """PhysX jitter below BOX_TOL_M is not a move (a move bumps `rev`, which re-derives every `where`)."""
        with self._lock:
            if object_id in self._held and not held_too:
                return
            old = self._boxes.get(object_id)
            if old is None or max(abs(a - b) for lo_hi in zip(old, box) for a, b in zip(*lo_hi)) > BOX_TOL_M:
                self._boxes[object_id] = box
                if object_id not in self._held:
                    self.occluders.replace(object_id, box)
                self.rev += 1
            self._pose_source[object_id] = source

    def _update_held(self, pose: RobotPose | None = None) -> None:
        """Held objects follow the real palm in PhysX when P1 streams poses; else the nominal hand offset."""
        if self._objects_live() or ("get_objects" in self._ops and self.palm_position("right") is not None):
            return
        super()._update_held(pose)

    def object(self, object_id: str):
        self._refresh_objects()
        return super().object(object_id)

    def truth(self):
        self._refresh_objects()
        return super().truth()

    # ------------------------------------------------------------------ detections (P1.6)
    def detections(self, camera: str = "head", *, yaw_offset: float = 0.0, waist_yaw: float = 0.0,
                   waist_pitch: float = 0.0, cam_pose: CameraPose | None = None, method: str | None = None):
        self._refresh_objects()
        geo = super().detections(camera, yaw_offset=yaw_offset, waist_yaw=waist_yaw, waist_pitch=waist_pitch,
                                 cam_pose=cam_pose)
        live_view = not (yaw_offset or waist_yaw or waist_pitch or cam_pose is not None)
        if method != "best" or not live_view or "detections" not in self._ops:
            return geo
        seg = self._segmentation(camera)
        return geo if seg is None else seg

    def _segmentation(self, camera: str | None) -> list[Detection] | None:
        cam = self.camera_model(camera)
        p1_name = P1_CAMERA.get(cam.name)
        if p1_name is None or not self.has_camera(p1_name):
            return None
        p = self.robot_pose()
        key = (round(p.x / SEG_POSE_TOL[0]), round(p.y / SEG_POSE_TOL[0]), round(p.yaw / SEG_POSE_TOL[1]), self.rev)
        hit = self._seg_cache.get(cam.name)
        if hit is not None and time.monotonic() - hit[0] < SEG_CACHE_S and hit[1] == key:
            return hit[2]
        self.seg_stats["calls"] += 1
        try:
            rep = self.rpc.call("detections", timeout_s=2.0, camera=p1_name, min_px=int(round(cam.min_pixels)),
                                max_range=max(cam.range_objects, cam.range_landmarks))
        except P1Error as e:
            self.seg_stats["errors"] += 1
            self.seg_stats["last_error"] = str(e)[:200]
            return None
        if not rep.get("ok", True):
            self.seg_stats["errors"] += 1
            self.seg_stats["last_error"] = str(rep.get("code") or rep.get("error"))[:200]
            return None
        out = self._from_segmentation(rep, cam)
        self._seg_cache[cam.name] = (time.monotonic(), key, out)
        return out

    def _from_segmentation(self, rep: dict, cam: CameraModel) -> list[Detection]:
        """P1's pixel counts per scene id -> Detections: visibility and pixels from the render, position and `where`
        from the world's GT (one object state for everyone). Held objects are always reported, like the geometry."""
        method = str(rep.get("method") or "instance_id_segmentation_fast")
        pose = self.robot_pose()
        cp = self.camera_pose(camera=cam.name if cam.name != self.cam.name else None)
        out: list[Detection] = []
        with self._lock:
            wh = self._wheres()
            for oid, arm in self._held.items():
                spec = self.map.objects[oid]
                b = self._boxes[oid]
                out.append(Detection(oid, "object", spec.type, spec.label, f"hand:{arm}",
                                     tuple((b[0][i] + b[1][i]) / 2 for i in range(3)), 0.4, 1000.0,   # type: ignore[arg-type]
                                     method=method))
            marks: dict[str, float] = {}
            for d in rep.get("detections") or []:
                sid, px = str(d.get("id")), float(d.get("px") or 0.0)
                oid = self._by_scene_id.get(sid)
                if oid is not None:
                    if oid in self._held:
                        continue
                    spec = self.map.objects[oid]
                    b = self._boxes[oid]
                    c = tuple((b[0][i] + b[1][i]) / 2 for i in range(3))
                    dist = float(d.get("dist_m") or math.dist(c, cp.pos))
                    if dist > cam.range_objects:
                        continue
                    out.append(Detection(oid, "object", spec.type, spec.label, wh[oid].where, c,   # type: ignore[arg-type]
                                         round(dist, 3), round(px, 1), method=method))
                    continue
                name = self._landmark_of.get(sid)
                if name is not None:
                    marks[name] = marks.get(name, 0.0) + px
            for name, px in marks.items():
                lm = self.map.landmarks[name]
                dist = math.dist(lm.center, cp.pos)
                if dist > cam.range_landmarks:
                    continue
                out.append(Detection(name, "landmark", lm.type, lm.label, None, lm.center, round(dist, 3),
                                     round(px, 1), near=lm.near, method=method))
        _ = pose
        return out

    # ------------------------------------------------------------------ cameras (P1.4 / OD1)
    def enable_camera(self, camera: str, on: bool, *, consumer: str = "runtime", ttl_s: float | None = None) -> dict:
        if "camera" not in self._ops:
            raise NotSupported("P1 has no `camera` op (p1_m2b.md §5.4)")
        name = P1_CAMERA.get(self.camera_model(camera).name, camera)
        args: dict[str, Any] = {"name": name, "on": bool(on), "consumer": consumer}
        if ttl_s is not None:
            args["ttl_s"] = float(ttl_s)
        return self._ok(self.rpc.call("camera", timeout_s=2.0, **args), "camera")

    def set_render_rates(self, head_hz: float, ego_hz: float) -> None:
        if "set_render_rates" not in self._ops:
            raise NotSupported("P1 has no `set_render_rates` op")
        self._ok(self.rpc.call("set_render_rates", timeout_s=2.0, head_hz=float(head_hz), ego_hz=float(ego_hz)),
                 "set_render_rates")

    # ------------------------------------------------------------------ sim events (P1.9)
    def _on_event(self, msg: dict) -> None:
        self._events.append(dict(msg))
        if msg.get("event") in ("object_fell", "object_moved", "reset_scene", "detach", "attach"):
            self._obj_t = 0.0                   # re-read poses on the next query when there is no stream

    def drain_events(self) -> list[dict]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    # ------------------------------------------------------------------ SimControl
    def capabilities(self) -> dict[str, Any]:
        ops = self._ops
        return {**super().capabilities(),
                "teleport_robot": False, "attach": "attach" in ops, "detach": "detach" in ops, "band": True,
                "reset_robot": True, "object_poses": "get_objects" in ops,
                "link_poses": self.link_pose("torso_link") is not None, "segmentation": "detections" in ops,
                "enable_camera": "camera" in ops, "reset_scene": "reset_scene" in ops,
                "move_object": "move_object" in ops, "p1_contract": self.p1_info.get("p1_contract"),
                "cameras": dict(self.p1_info.get("cameras") or {})}

    def attach(self, object_id: str, arm: str, mode: str = "follow") -> None:
        if "attach" not in self._ops:
            raise NotSupported("P1 has no `attach` op (M2b: attach {id, arm, mode})")
        spec = self.map.objects[object_id]
        self._grip_t = time.monotonic()
        self._ok(self.rpc.call("attach", id=spec.scene_id, arm=arm, mode=mode, timeout_s=2.0), "attach")
        self._attach_local(object_id, arm, mode)
        self._grip_t = time.monotonic()

    def detach(self, object_id: str, pose=None) -> None:
        pose = xyz(pose)
        if "detach" not in self._ops:
            raise NotSupported("P1 has no `detach` op (M2b: detach {id, pose?})")
        spec = self.map.objects[object_id]
        args: dict[str, Any] = {"id": spec.scene_id}
        if pose is not None:
            args["pose"] = [float(v) for v in pose]
        self._grip_t = time.monotonic()
        rep = self._ok(self.rpc.call("detach", timeout_s=2.0, **args), "detach")
        self._detach_local(object_id, pose)
        if rep.get("aabb"):
            box = (tuple(float(v) for v in rep["aabb"][0]), tuple(float(v) for v in rep["aabb"][1]))
            self._set_object_box(object_id, box, source="sim")
        self._grip_t = time.monotonic()

    def band(self, on: bool, ramp_s: float = 1.5) -> None:
        self._ok(self.rpc.call("band", on=bool(on), ramp_s=float(ramp_s)), "band")

    def reset_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        x, y, yaw = xyyaw(pose, y, yaw)
        self._ok(self.rpc.call("reset_robot", x=x, y=y, yaw=yaw, timeout_s=10.0), "reset_robot")

    def reset_scene(self, variant: str = "default", *, poses: dict | None = None, robot: Any = None) -> dict:
        """P1.7: every held object released, every dynamic object back at its load pose (+ `poses`, object centres
        by object id here, scene ids on the wire), optionally the robot too."""
        if "reset_scene" not in self._ops:
            raise NotSupported("P1 has no `reset_scene` op")
        args: dict[str, Any] = {"variant": variant}
        if poses:
            args["poses"] = {self.map.objects[k].scene_id if k in self.map.objects else k: list(v)
                             for k, v in poses.items()}
        if robot is not None:
            args["robot"] = robot if isinstance(robot, (bool, dict)) else dict(zip(("x", "y", "yaw"), xyyaw(robot)))
        rep = self._ok(self.rpc.call("reset_scene", timeout_s=30.0, **args), "reset_scene")
        with self._lock:
            self._held.clear()
            self._grasp.clear()
            self.rev += 1
        self._grip_t = -1e9
        self._poll_objects(dynamic_only=False)
        return rep

    def move_object(self, object_id: str, center, yaw: float | None = None) -> None:
        """Test-only fixture write (P1.7 move_object): the object's centre at `center`."""
        if "move_object" not in self._ops:
            raise NotSupported("P1 has no `move_object` op")
        pose = [float(v) for v in xyz(center)] + ([float(yaw)] if yaw is not None else [])
        rep = self._ok(self.rpc.call("move_object", timeout_s=2.0, id=self.map.objects[object_id].scene_id,
                                     pose=pose), "move_object")
        if rep.get("aabb"):
            box = (tuple(float(v) for v in rep["aabb"][0]), tuple(float(v) for v in rep["aabb"][1]))
            with self._lock:
                self._held.pop(object_id, None)
                self._grasp.pop(object_id, None)
            self._set_object_box(object_id, box, source="sim")

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
    mode = rep.get("mode")
    caption = ("furniture-only render at P1 start-up (props drawn from truth)" if mode == "furniture"
               else "static render at P1 start-up (robot visible at spawn)")
    return {"cx": float(cx), "cz": float(cy), "size": h * mpp / 2, "w": w, "h": h, "path": rep.get("path"),
            "caption": caption, "mode": mode or "full"}


__all__ = ["IsaacGTWorldModel", "topdown_from_render", "M2B_OPS"]
