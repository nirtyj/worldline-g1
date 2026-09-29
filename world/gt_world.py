"""GTWorld: the ground-truth WorldModel shared by LiteWorld (offline) and IsaacGTWorldModel (P1).

Both hold the same static map (mapgen), the same geometric `where`, the same camera perception, and an object-state
table. They differ only in where the robot pose and the object poses come from:

    LiteWorld            robot pose set by LiteBody; objects move by in-process attach/detach (SimControl)
    IsaacGTWorldModel    robot pose from P1 gt.pose (SUB 5601); objects: P1 `get_objects` if the P1 build has it
                         (M2b), else the static scene_info poses + the local attach/detach overlay (labelled
                         `pose_source: "scene"` / `"attach"`)

Outputs for agent/ (look data, perception, truth) are in Worldline's frame via world/coords.py.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

from . import coords, vocab
from .mapgen import MapParams, StaticMap, build_static_map
from .model import Detection, GraspState, NotSupported, ObjectState, RobotPose, TruthSnapshot, xyyaw, xyz
from .nav_grid import OccupancyData, WorldGrid
from .perception import (CAMERAS, EGO_D435, CameraModel, CameraPose, Occluders, Perception, Target, camera_pose,
                         is_closed_container, structure_mask)
from .scene import SceneData
from .where import WhereModel

HAND_OFFSET = {"left": (0.30, 0.20), "right": (0.30, -0.20)}   # held object centre in the pelvis frame (m)
HAND_Z = 0.95                                                  # world height of a held object's centre (m)


def _place_points(half: tuple[float, float], edge: float = 0.08) -> list[tuple[float, float]]:
    """Centre of a stretch first, then rings of points inside it (thor/world.py:95-104, verbatim rule)."""
    hx, hy = max(half[0] - edge, 0.0), max(half[1] - edge, 0.0)
    pts = [(0.0, 0.0)]
    for r in (0.15, 0.3, 0.45):
        for fx, fy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            dx, dy = max(-hx, min(hx, fx * r)), max(-hy, min(hy, fy * r))
            if (dx, dy) not in pts:
                pts.append((dx, dy))
    return pts


def _translate(box, c) -> tuple:
    (x0, y0, z0), (x1, y1, z1) = box
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    return ((c[0] - hx, c[1] - hy, c[2] - hz), (c[0] + hx, c[1] + hy, c[2] + hz))


class GTWorld:
    source = "isaac-gt"
    perception_source = "isaac-ground-truth"

    def __init__(self, scene: SceneData, occ: OccupancyData, *, scene_key: str | None = None,
                 map_params: MapParams | None = None, camera: CameraModel | str = EGO_D435,
                 user_surface: str | None = None, topdown: dict | None = None):
        self._lock = threading.RLock()
        self.scene_data = scene
        mp = map_params or MapParams()
        self.grid = WorldGrid(occ, robot_radius=mp.robot_radius)
        self.map: StaticMap = build_static_map(scene, self.grid, mp, scene_key=scene_key, source=self.source,
                                               topdown=topdown, user_surface=user_surface)
        self.scene = self.map.scene
        self.where_model = WhereModel(self.map)
        self.cam = CAMERAS[camera] if isinstance(camera, str) else camera
        self._boxes: dict[str, tuple] = {oid: spec.aabb for oid, spec in self.map.objects.items()}
        self._held: dict[str, str] = {}
        self._grasp: dict[str, GraspState] = {}
        self._pose_source: dict[str, str] = {oid: "scene" for oid in self._boxes}
        self.rev = 0                                   # bumps whenever an object moves or is held/released
        self._where_cache: tuple[int, dict] | None = None
        things = {t.scene_id: t.aabb for t in self.map.things
                  if not vocab.is_pickupable(t.thor_type) and t.thor_type not in vocab.NON_SOLID_TYPES}
        closed = [t.scene_id for t in self.map.things if is_closed_container(t.thor_type)]
        struct = structure_mask(occ.occ, occ.raw_codes, [t.aabb for t in self.map.things
                                                         if t.thor_type not in vocab.NON_SOLID_TYPES],
                                occ.res, occ.origin)
        self._occ_keys_static = list(things)
        self.occluders = Occluders.build({**things, **self._boxes}, closed_keys=closed, structure=struct,
                                         res=occ.res, origin=occ.origin)
        self.perceiver = Perception(self.cam, self.occluders, floor_z=self.map.floor_z)
        self._landmark_targets = [Target(l.name, "landmark", l.aabb, frozenset(l.scene_ids))
                                  for l in self.map.landmarks.values()]

    # ================================================================== to override
    def robot_pose(self) -> RobotPose:
        raise NotImplementedError

    # ================================================================== static map
    def static_map(self) -> StaticMap:
        return self.map

    def lookup_keypoints(self) -> dict:
        return self.map.lookup_keypoints()

    def path(self, a, b) -> list[tuple[float, float]] | None:
        return self.grid.path(xyyaw(a)[:2], xyyaw(b)[:2])

    def occupancy(self) -> dict:
        """The occupancy grid for the UI (ui/truth.py grid_view): resolution, origin, raw (0 free / 1 obstacle /
        2 outside), inflated (1 = robot centre may not go, the body's radius), source."""
        d = self.grid.data
        raw = d.raw_codes if d.raw_codes is not None else d.occ.astype(np.uint8)
        return {"resolution": d.res, "origin": list(d.origin), "raw": raw,
                "inflated": self.grid.nav.inflated.astype(np.uint8), "source": d.source,
                "robot_radius": self.grid.robot_radius}

    def latest_frame(self, camera: str = "head"):
        """api.types.CameraFrame of this camera, or None (GT worlds render no images; frames come from
        world/frames.py or viz.tap.FrameTap on Isaac)."""
        return None

    # ================================================================== objects
    def _update_held(self, pose: RobotPose | None = None) -> None:
        if not self._held:
            return
        pose = pose or self.robot_pose()
        for oid, arm in self._held.items():
            dx, dy = HAND_OFFSET[arm]
            hx, hy = coords.body_to_world(pose.xy_yaw, dx, dy)
            box = _translate(self._boxes[oid], (hx, hy, HAND_Z))
            self._boxes[oid] = box

    def _wheres(self) -> dict:
        with self._lock:
            if self._where_cache is None or self._where_cache[0] != self.rev:
                self._where_cache = (self.rev, self.where_model.all(self._boxes, self._held))
            return self._where_cache[1]

    def object(self, object_id: str) -> ObjectState | None:
        spec = self.map.objects.get(object_id)
        if spec is None:
            return None
        with self._lock:
            self._update_held()
            w = self._wheres()[object_id]
            return ObjectState(oid=object_id, type=spec.type, label=spec.label, box=self._boxes[object_id],
                               where=w.where, where_rule=w.rule, held_by=self._held.get(object_id),
                               pose_source=self._pose_source.get(object_id, "scene"))

    def objects(self) -> dict[str, ObjectState]:
        return {oid: self.object(oid) for oid in self.map.objects}   # type: ignore[misc]

    def hands(self) -> dict[str, str | None]:
        with self._lock:
            out: dict[str, str | None] = {"left": None, "right": None}
            for oid, arm in self._held.items():
                out[arm] = oid
            return out

    def grasp_state(self, object_id: str, arm: str) -> GraspState:
        with self._lock:
            g = self._grasp.get(object_id)
            held = self._held.get(object_id) == arm
            if g is not None and held:
                return g
            return GraspState(object_id, arm, held)

    def free_spot(self, surface: str, object_id: str, near=None,
                  within: Callable[[float, float, float], bool] | None = None):
        """A free place for object_id on this stretch: THOR's `_place_points` rings (thor/world.py:95-104), or,
        with `near` (the robot's pose), a dense grid nearest the robot first. The object's footprint must stay on
        the stretch and not overlap another object's (2 cm margin), and `within(x, y, z)` (the reach test from the
        current pose) must hold. Returns api Pose3D of the object's centre resting on the top."""
        from api.types import Pose3D
        s = self.map.surfaces.get(surface)
        spec = self.map.objects.get(object_id)
        if s is None or spec is None:
            return None
        near = xyz(near) if near is not None and not hasattr(near, "yaw") else (
            (near.x, near.y, 0.0) if near is not None else None)
        with self._lock:
            box = self._boxes[object_id]
            ex, ey, ez = (box[1][i] - box[0][i] for i in range(3))
            others = [b for oid, b in self._boxes.items() if oid != object_id and oid not in self._held]
            if near is None:
                pts = _place_points(s.half)
            else:
                # a G1 reaches ~0.1-0.2 m past the edge: a dense 4 cm grid over the stretch, nearest the robot
                # first (THOR's sparse rings rarely land inside a humanoid's reach)
                mx = max(0.0, s.half[0] - ex / 2 - 0.02)
                my = max(0.0, s.half[1] - ey / 2 - 0.02)
                xs = sorted({max(-mx, min(mx, i * 0.04)) for i in range(-int(mx / 0.04) - 1, int(mx / 0.04) + 2)})
                ys = sorted({max(-my, min(my, j * 0.04)) for j in range(-int(my / 0.04) - 1, int(my / 0.04) + 2)})
                pts = [(dx, dy) for dx in xs for dy in ys]
                pts.sort(key=lambda d: math.hypot(s.center[0] + d[0] - near[0], s.center[1] + d[1] - near[1]))
                pts = pts[:600]
            for dx, dy in pts:
                cx, cy = s.center[0] + dx, s.center[1] + dy
                cz = s.z_top + ez / 2 + 0.005
                lo = (cx - ex / 2 - 0.02, cy - ey / 2 - 0.02)
                hi = (cx + ex / 2 + 0.02, cy + ey / 2 + 0.02)
                clash = any(b[0][0] < hi[0] and b[1][0] > lo[0] and b[0][1] < hi[1] and b[1][1] > lo[1]
                            and b[1][2] > s.z_top - 0.02 and b[0][2] < s.z_top + 0.3 for b in others)
                if clash:
                    continue
                if within is not None and not within(cx, cy, cz):
                    continue
                return Pose3D(cx, cy, cz)
            return None

    def surface_has_room(self, surface: str, object_id: str) -> bool:
        """Any free spot on the stretch, reach ignored."""
        return self.free_spot(surface, object_id) is not None

    # ------------------------------------------------------------------ in-process object moves
    def _attach_local(self, object_id: str, arm: str, mode: str) -> None:
        with self._lock:
            if object_id not in self._boxes:
                raise KeyError(object_id)
            support = self._boxes[object_id][0][2]
            self._held[object_id] = arm
            self._pose_source[object_id] = "attach"
            self._update_held()
            self._grasp[object_id] = GraspState(object_id, arm, True, round(self._boxes[object_id][0][2] - support, 3),
                                                attach_mode=mode)
            self.occluders.replace(object_id, None)
            self.rev += 1

    def _detach_local(self, object_id: str, pose: tuple[float, float, float] | None) -> None:
        with self._lock:
            if object_id not in self._boxes:
                raise KeyError(object_id)
            self._update_held()
            self._held.pop(object_id, None)
            self._grasp.pop(object_id, None)
            if pose is not None:
                self._boxes[object_id] = _translate(self._boxes[object_id], pose)
            else:            # dropped: straight down to the floor
                b = self._boxes[object_id]
                c = ((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, self.map.floor_z + (b[1][2] - b[0][2]) / 2)
                self._boxes[object_id] = _translate(b, c)
            self._pose_source[object_id] = "attach"
            self.occluders.replace(object_id, self._boxes[object_id])
            self.rev += 1

    def _set_object_box(self, object_id: str, box, source: str = "sim") -> None:
        with self._lock:
            if object_id in self._held:
                return
            if box != self._boxes.get(object_id):
                self._boxes[object_id] = box
                self.occluders.replace(object_id, box)
                self.rev += 1
            self._pose_source[object_id] = source

    # ================================================================== perception
    def camera_pose(self, *, yaw_offset: float = 0.0, waist_yaw: float = 0.0, waist_pitch: float = 0.0,
                    pose: RobotPose | None = None) -> CameraPose:
        pose = pose or self.robot_pose()
        return camera_pose(self.cam, (pose.x, pose.y, pose.z), pose.yaw, waist_yaw=waist_yaw,
                           waist_pitch=waist_pitch, yaw_offset=yaw_offset)

    def _targets(self) -> list[Target]:
        ts = [Target(oid, "object", self._boxes[oid]) for oid in self.map.objects if oid not in self._held]
        return ts + self._landmark_targets

    def sightings(self, cam_pose: CameraPose | None = None) -> dict:
        with self._lock:
            self._update_held()
            cp = cam_pose or self.camera_pose()
            return self.perceiver.sight_all(cp, self._targets())

    def detections(self, camera: str = "head", *, yaw_offset: float = 0.0, waist_yaw: float = 0.0,
                   waist_pitch: float = 0.0, cam_pose: CameraPose | None = None) -> list[Detection]:
        """What the camera sees now (GT, method gt-geometric). Held objects are always reported (in hand)."""
        with self._lock:
            pose = self.robot_pose()
            self._update_held(pose)
            cp = cam_pose or self.camera_pose(yaw_offset=yaw_offset, waist_yaw=waist_yaw, waist_pitch=waist_pitch,
                                              pose=pose)
            sights = self.perceiver.sight_all(cp, self._targets())
            wh = self._wheres()
            out = []
            for oid, arm in self._held.items():
                spec = self.map.objects[oid]
                out.append(Detection(oid, "object", spec.type, spec.label, f"hand:{arm}",
                                     tuple((self._boxes[oid][0][i] + self._boxes[oid][1][i]) / 2 for i in range(3)),   # type: ignore[arg-type]
                                     0.4, 1000.0))
            for key, s in sights.items():
                if not s.visible:
                    continue
                if s.kind == "object":
                    spec = self.map.objects[key]
                    b = self._boxes[key]
                    out.append(Detection(key, "object", spec.type, spec.label, wh[key].where,
                                         tuple((b[0][i] + b[1][i]) / 2 for i in range(3)), round(s.dist, 3),   # type: ignore[arg-type]
                                         round(s.px, 1)))
                else:
                    lm = self.map.landmarks[key]
                    out.append(Detection(key, "landmark", lm.type, lm.label, None, lm.center, round(s.dist, 3),
                                         round(s.px, 1), near=lm.near))
            return out

    def view_spec(self, cam_pose: CameraPose) -> dict:
        return cam_pose.view_spec(self.cam, self.map.floor_z)

    def camera_pose_from_view(self, view) -> CameraPose:
        """A Worldline ViewSpec (dict or api.observation.ViewSpec) -> the camera pose it describes."""
        g = (lambda k, d=None: view.get(k, d)) if isinstance(view, dict) else (lambda k, d=None: getattr(view, k, d))
        x, y = coords.to_isaac_xy(float(g("x")), float(g("z")))
        yaw = coords.yaw_isaac_rad(float(g("yaw")))
        tilt = math.radians(float(g("tilt", 15.0) or 0.0))
        cam_h = g("cam_h")
        z = self.map.floor_z + (float(cam_h) if cam_h is not None else self.camera_pose().pos[2] - self.map.floor_z)
        R = np.array(coords.rot_z(yaw)) @ np.array(coords.rot_y(tilt))
        return CameraPose(pos=(x, y, z), R=R, yaw=yaw, pitch_down=tilt)

    def scan(self, views, *, at: str | None):
        """api WorldModel.scan: what these views (Worldline ViewSpecs) see, as api.observation.LookData.
        Used by lite for the PLAN's two-row waist scan without moving the body."""
        from api.observation import LookData, ViewSpec
        rows = [self.detections(cam_pose=self.camera_pose_from_view(v)) for v in views]
        d = self.look([v if isinstance(v, dict) else v.to_dict() for v in views], at=at, detections=rows)
        vs = [v if isinstance(v, ViewSpec) else ViewSpec(**{k: v[k] for k in ViewSpec.__dataclass_fields__ if k in v})
              for v in views]
        return LookData(at=d["at"], surfaces=d["surfaces"], landmarks=d["landmarks"], views=vs, hands=d["hands"])

    def look(self, views: list[dict] | None, *, at: str | None,
             detections: list[list[Detection]] | None = None) -> dict:
        """LookData dict (agent/state.apply_observation shape; Worldline frame):
        {at, surfaces:{where:[{id,type,brand,color,label,surface,pos:[x,z,h]}]}, landmarks:[{id,type,label,near,
        pos:[x,z]}], views:[ViewSpec...], hands:{left,right}}. `views=[]` (a glance) never marks anything absent.
        detections: one list per view (from `detections()`); default: the current view only."""
        dets = detections if detections is not None else [self.detections()]
        seen: dict[str, Detection] = {}
        marks: dict[str, Detection] = {}
        for row in dets:
            for d in row:
                if d.kind == "object":
                    if d.where and d.where.startswith("hand:"):
                        continue
                    seen[d.id] = d
                else:
                    marks[d.id] = d
        surfaces: dict[str, list[dict]] = {}
        for oid, d in sorted(seen.items()):
            surfaces.setdefault(d.where or "unknown", []).append(
                {"id": oid, "type": d.type, "brand": None, "color": None, "label": d.label,
                 "surface": d.where, "pos": coords.map_pos(d.pos[0], d.pos[1], d.pos[2] - self.map.floor_z)})
        landmarks = [{"id": n, "type": d.type, "label": d.label, "near": d.near,
                      "pos": coords.map_pos(d.pos[0], d.pos[1])} for n, d in sorted(marks.items())]
        return {"at": at, "surfaces": surfaces, "landmarks": landmarks, "views": list(views or []),
                "hands": self.hands()}

    def perception(self) -> dict:
        """THOR `ThorRobot.perception()` shape (thor/robot.py:146-182), Worldline frame, visible things only,
        confidence 1.0, `where="hand:<arm>"` directly for held objects (PLAN §4.4 state.py)."""
        pose = self.robot_pose()
        src = self.perception_source
        objects, landmarks = {}, {}
        for d in self.detections():
            pose_d = coords.thor_pose(d.pos[0], d.pos[1], d.pos[2] - self.map.floor_z)
            dist = round(math.hypot(d.pos[0] - pose.x, d.pos[1] - pose.y), 3)
            if d.kind == "object":
                objects[d.id] = {"type": d.type, "label": d.label, "where": d.where, "pose": pose_d,
                                 "distance_m": dist, "visible": True, "confidence": 1.0, "source": src,
                                 "method": d.method}
            else:
                landmarks[d.id] = {"type": d.type, "label": d.label, "near": d.near, "pose": pose_d,
                                   "distance_m": dist, "visible": True, "confidence": 1.0, "source": src}
        people = {}
        user = self.map.lookup_keypoints().get("people", {}).get("user")
        if user:
            k = self.map.keypoints.get(user["keypoint"])
            if k is not None:
                mx, mz = coords.to_map_xz(k.x, k.y)
                people["user"] = {"pose": {"x": round(mx, 3), "z": round(mz, 3)},
                                  "distance_m": round(math.hypot(k.x - pose.x, k.y - pose.y), 3),
                                  "confidence": 1.0, "source": src, **user}
        return {"objects": objects, "landmarks": landmarks, "people": people, "source": src}

    # ================================================================== truth (UI + eval only)
    def truth(self) -> TruthSnapshot:
        pose = self.robot_pose()
        with self._lock:
            self._update_held(pose)
            sights = self.perceiver.sight_all(self.camera_pose(pose=pose), self._targets())
            wh = self._wheres()
            objs = {}
            for oid, spec in self.map.objects.items():
                b = self._boxes[oid]
                c = [(b[0][i] + b[1][i]) / 2 for i in range(3)]
                mx, mz = coords.to_map_xz(c[0], c[1])
                s = sights.get(oid)
                objs[oid] = {"type": spec.type, "label": spec.label, "where": wh[oid].where,
                             "x": round(mx, 3), "y": round(c[2] - self.map.floor_z, 3), "z": round(mz, 3),
                             "visible": bool(oid in self._held or (s is not None and s.visible)),
                             "pose_source": self._pose_source.get(oid, "scene")}
            lms = {}
            for name, lm in self.map.landmarks.items():
                mx, mz = coords.to_map_xz(lm.center[0], lm.center[1])
                s = sights.get(name)
                lms[name] = {"type": lm.type, "label": lm.label, "near": lm.near, "x": round(mx, 3),
                             "y": round(lm.center[2] - self.map.floor_z, 3), "z": round(mz, 3),
                             "visible": bool(s is not None and s.visible)}
            things = {t.scene_id: vocab.snake(t.thor_type) for t in self.map.things
                      if t.thor_type not in vocab.NON_SOLID_TYPES}
            mx, mz = coords.to_map_xz(pose.x, pose.y)
            robot = {"x": round(mx, 3), "z": round(mz, 3), "yaw": round(coords.yaw_map_deg(pose.yaw), 1),
                     "pelvis_z": round(pose.pelvis_z, 3), "upright": not pose.fallen, "source": pose.source}
            return TruthSnapshot(t=time.time(), robot=robot, objects=objs, landmarks=lms, things=things,
                                 hands=self.hands(), source=self.source)

    # ================================================================== SimControl defaults
    def capabilities(self) -> dict[str, bool]:
        return {"teleport_robot": False, "attach": False, "detach": False, "band": False, "reset_robot": False,
                "object_poses": False}

    def teleport_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        raise NotSupported("teleport_robot")

    def attach(self, object_id: str, arm: str, mode: str = "follow") -> None:
        raise NotSupported("attach")

    def detach(self, object_id: str, pose: tuple[float, float, float] | None = None) -> None:
        raise NotSupported("detach")

    def band(self, on: bool, ramp_s: float = 1.5) -> None:
        raise NotSupported("band")

    def reset_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        raise NotSupported("reset_robot")

    def reset_scene(self, variant: str) -> None:
        raise NotSupported("reset_scene")

    def set_render_rates(self, head_hz: float, ego_hz: float) -> None:
        raise NotSupported("set_render_rates")

    # ================================================================== helpers for services
    def surface_of_keypoint(self, keypoint: str | None):
        return self.map.surface_served_from(keypoint)

    def keypoint_at(self, x: float, y: float, tol: float = 0.30) -> str | None:
        name, d = self.map.nearest_keypoint(x, y, tol)
        return name


def load_house_dir(path) -> tuple[SceneData, OccupancyData]:
    """A recorded house directory: house_info.json + occupancy.npz (assets/houses/<id>/ on the box)."""
    from pathlib import Path
    p = Path(path)
    return SceneData.from_json(p / "house_info.json"), OccupancyData.from_npz(p / "occupancy.npz")
