"""Ground-truth "what the head camera sees now" (PLAN §6.2.2 Visible, §5.10, §13 row 1-2).

METHOD (labelled `gt-geometric` in every detection; this is NOT instance segmentation):
  1. camera pose = gt.pose pelvis pose (x, y, z, yaw) * the torso offset at nominal waist (0, 0, 0) * the camera
     mount (docs/contracts/m1.md §1.4: d435_link on torso_link). A yaw/pitch override models scan views.
  2. frustum: the object's AABB corners are projected with the pinhole intrinsics; the clipped 2-D box area is the
     THOR-style "bbox pixels" (thor/world.py:_seen used the instance-detection box area >= 40 px^2).
  3. occlusion: 9 rays (centre + the 8 corners pulled 25 % toward the centre) from the camera to the object, tested
     against every other scene item's AABB (slab test, boxes shrunk 1 cm) and against wall/structure cells of the
     occupancy grid (blocked cells no item explains, treated as full height). Furniture that encloses the object
     (a shelf unit, a TV stand's lower shelf, a bed) does not hide it; a closed container (fridge, microwave,
     cabinet, drawer, ...) does.
  4. visible = at least one clear ray, visible pixels (bbox area x clear fraction) >= MIN_PIXELS (40 at 640x480,
     scaled with the resolution), and camera distance <= 2.5 m (objects) / 4.0 m (landmarks).

The real stack replaces this with instance-id pixel counts from Isaac (`instance_id_segmentation_fast`, M2b) or a
detector (§13); the output shape stays the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import coords, vocab

MIN_PIXELS = 40
REF_PIXELS = 640 * 480
TORSO_FROM_PELVIS = (-0.0039635, 0.0, 0.044)   # waist_roll_joint origin (g1 URDF; waist yaw/pitch joints at 0)


@dataclass(frozen=True)
class CameraModel:
    name: str
    width: int
    height: int
    vfov_deg: float
    mount_xyz: tuple[float, float, float]         # in torso_link
    mount_pitch: float                            # rad, positive = looking down (URDF rpy pitch about +y)
    mount_yaw: float = 0.0
    near: float = 0.05
    range_objects: float = 2.5
    range_landmarks: float = 4.0
    sim_added: bool = False                       # PLAN §1.3 #7: the `head` camera does not exist on a stock G1

    @property
    def fy(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    @property
    def fx(self) -> float:
        return self.fy                            # square pixels

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.width / 2) / self.fx))

    @property
    def min_pixels(self) -> float:
        return MIN_PIXELS * (self.width * self.height) / REF_PIXELS

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "CameraModel":
        return cls(name=name, width=int(d.get("width", 640)), height=int(d.get("height", 480)),
                   vfov_deg=float(d["vfov_deg"]), mount_xyz=tuple(float(v) for v in d["mount_xyz"]),   # type: ignore[arg-type]
                   mount_pitch=float(d.get("mount_pitch_rad", math.radians(d.get("mount_pitch_deg", 0.0)))),
                   mount_yaw=float(d.get("mount_yaw_rad", 0.0)), near=float(d.get("near", 0.05)),
                   range_objects=float(d.get("range_objects", 2.5)),
                   range_landmarks=float(d.get("range_landmarks", 4.0)), sim_added=bool(d.get("sim_added", False)))


# P1 today (docs/contracts/m1.md §1.4): d435_link on torso_link, ~48 deg down, VFOV 45, 640x480
EGO_D435 = CameraModel("ego_d435", 640, 480, 45.0, (0.0576235, 0.01753, 0.41987), 0.8307767239493009)
# PLAN §1.3 #7 / §7.2 target: sim-added head camera ~1.35 m standing, 15 deg down, 90 deg HFOV (VFOV 73.7 at 4:3)
HEAD_SIM = CameraModel("head_sim", 640, 480, 2 * math.degrees(math.atan(math.tan(math.radians(45)) * 3 / 4)),
                       (0.06, 0.0, 0.526), math.radians(15.0), sim_added=True)
CAMERAS = {c.name: c for c in (EGO_D435, HEAD_SIM)}


@dataclass(frozen=True)
class CameraPose:
    pos: tuple[float, float, float]               # world
    R: np.ndarray                                  # 3x3 world-from-camera (camera: x forward, y left, z up)
    yaw: float                                     # heading of the optical axis (Isaac rad)
    pitch_down: float                              # rad

    def view_spec(self, cam: CameraModel, floor_z: float = 0.0, range_m: float | None = None) -> dict:
        """Worldline's ViewSpec dict (api/observation.py ViewSpec; agent/state.in_view reads it)."""
        mx, mz = coords.to_map_xz(self.pos[0], self.pos[1])
        return {"x": round(mx, 3), "z": round(mz, 3), "yaw": round(coords.yaw_map_deg(self.yaw), 1),
                "tilt": round(math.degrees(self.pitch_down), 1), "fov": round(cam.hfov_deg, 1),
                "range": range_m if range_m is not None else cam.range_objects,
                "vfov": round(cam.vfov_deg, 1), "cam_h": round(self.pos[2] - floor_z, 3), "near": 0.3}


def camera_pose(cam: CameraModel, pelvis_xyz: Sequence[float], pelvis_yaw: float, *, waist_yaw: float = 0.0,
                waist_pitch: float = 0.0, yaw_offset: float = 0.0) -> CameraPose:
    """World pose of the camera from the pelvis pose (upright pelvis assumed: only its yaw is used).
    waist_yaw / waist_pitch model the PLAN's waist scan (M2b); yaw_offset models an in-place turn."""
    yaw_t = pelvis_yaw + waist_yaw + yaw_offset
    Rz = np.array(coords.rot_z(yaw_t))
    Ry_w = np.array(coords.rot_y(waist_pitch))
    R_torso = Rz @ Ry_w
    p_torso = np.asarray(pelvis_xyz, dtype=float) + Rz @ np.array(TORSO_FROM_PELVIS)
    R_mount = np.array(coords.rot_z(cam.mount_yaw)) @ np.array(coords.rot_y(cam.mount_pitch))
    R = R_torso @ R_mount
    pos = p_torso + R_torso @ np.array(cam.mount_xyz)
    fwd = R[:, 0]
    return CameraPose(pos=tuple(float(v) for v in pos), R=R, yaw=math.atan2(fwd[1], fwd[0]),   # type: ignore[arg-type]
                      pitch_down=math.asin(max(-1.0, min(1.0, -fwd[2]))))


@dataclass
class Target:
    key: str                                      # oid or landmark name
    kind: str                                     # "object" | "landmark"
    box: tuple[tuple[float, float, float], tuple[float, float, float]]
    exclude: frozenset[str] = frozenset()         # occluder keys never tested against this target


@dataclass
class Sighting:
    key: str
    kind: str
    visible: bool
    in_frustum: bool
    px: float                                     # estimated visible pixels
    bbox_px: float
    clear_frac: float
    dist: float
    bbox: tuple[float, float, float, float] | None
    reason: str = ""                              # why not visible: out_of_frustum | occluded | too_far | too_small


@dataclass
class Occluders:
    keys: list[str]
    lo: np.ndarray                                # (B, 3)
    hi: np.ndarray
    closed: np.ndarray                            # (B,) bool: closed container -> hides what is inside it
    structure: np.ndarray | None = None           # (H, W) bool wall cells
    res: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)
    wall_h: tuple[float, float] = (0.0, 2.6)

    @classmethod
    def build(cls, boxes: Mapping[str, tuple], closed_keys: Iterable[str] = (), structure: np.ndarray | None = None,
              res: float = 0.05, origin: tuple[float, float] = (0.0, 0.0), shrink: float = 0.01) -> "Occluders":
        keys = list(boxes)
        closed_keys = set(closed_keys)
        lo = np.array([[boxes[k][0][i] + shrink for i in range(3)] for k in keys]).reshape(-1, 3)
        hi = np.array([[boxes[k][1][i] - shrink for i in range(3)] for k in keys]).reshape(-1, 3)
        return cls(keys=keys, lo=lo, hi=hi, closed=np.array([k in closed_keys for k in keys], dtype=bool),
                   structure=structure, res=res, origin=origin)

    def replace(self, key: str, box: tuple | None, shrink: float = 0.01) -> None:
        """Move (or remove, box=None -> far away) one occluder, e.g. an object that was picked up."""
        i = self.keys.index(key) if key in self.keys else None
        if i is None:
            return
        if box is None:
            self.lo[i] = (1e6, 1e6, 1e6)
            self.hi[i] = (1e6 + 1e-3,) * 3
        else:
            self.lo[i] = [box[0][k] + shrink for k in range(3)]
            self.hi[i] = [box[1][k] - shrink for k in range(3)]


class Perception:
    def __init__(self, cam: CameraModel, occluders: Occluders, floor_z: float = 0.0):
        self.cam = cam
        self.occ = occluders
        self.floor_z = floor_z

    # ------------------------------------------------------------------ projection
    def project(self, pose: CameraPose, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points (N,3) -> (u, v, depth). depth = distance along the optical axis."""
        rel = (pts - np.asarray(pose.pos)) @ pose.R          # camera frame: x fwd, y left, z up
        d = rel[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cam.width / 2 - self.cam.fx * rel[:, 1] / d
            v = self.cam.height / 2 - self.cam.fy * rel[:, 2] / d
        return u, v, d

    def _ray_clear(self, pose: CameraPose, samples: np.ndarray, t: Target, target_center: np.ndarray) -> np.ndarray:
        o = np.asarray(pose.pos)
        D = samples - o                                       # (S, 3), segment o -> sample
        occ = self.occ
        if len(occ.keys):
            mask = np.ones(len(occ.keys), dtype=bool)
            for i, k in enumerate(occ.keys):
                if k == t.key or k in t.exclude:
                    mask[i] = False
            # open furniture that encloses the target never hides it; closed containers do
            inside = np.all((occ.lo <= target_center) & (target_center <= occ.hi), axis=1)
            mask &= ~(inside & ~occ.closed)
            lo, hi = occ.lo[mask], occ.hi[mask]
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / D[:, None, :]                     # (S, 1, 3)
                t1 = (lo[None] - o) * inv
                t2 = (hi[None] - o) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=2)      # (S, B)
            tmax = np.nanmin(np.maximum(t1, t2), axis=2)
            hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0) & (tmax > 0.0)
            clear = ~hit.any(axis=1)
        else:
            clear = np.ones(len(samples), dtype=bool)
        if occ.structure is not None and clear.any():
            for s in np.nonzero(clear)[0]:
                if self._wall_hit(o, samples[s]):
                    clear[s] = False
        return clear

    def _wall_hit(self, a: np.ndarray, b: np.ndarray) -> bool:
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(2, int(L / (0.5 * self.occ.res)) + 1)
        ts = np.linspace(0.0, 1.0, n)[1:-1]
        if len(ts) == 0:
            return False
        xs = a[0] + (b[0] - a[0]) * ts
        ys = a[1] + (b[1] - a[1]) * ts
        zs = a[2] + (b[2] - a[2]) * ts
        ix = np.floor((xs - self.occ.origin[0]) / self.occ.res).astype(int)
        iy = np.floor((ys - self.occ.origin[1]) / self.occ.res).astype(int)
        H, W = self.occ.structure.shape
        ok = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
        if not ok.all():
            return True
        # skip the last 5 cm next to the target (its own wall-mounted neighbours)
        near_end = (1.0 - ts) * L < 0.05
        walls = self.occ.structure[iy, ix] & ~near_end & (zs >= self.occ.wall_h[0]) & (zs <= self.occ.wall_h[1])
        return bool(walls.any())

    # ------------------------------------------------------------------ sightings
    def sight(self, pose: CameraPose, t: Target) -> Sighting:
        cam = self.cam
        lo = np.asarray(t.box[0], dtype=float)
        hi = np.asarray(t.box[1], dtype=float)
        c = (lo + hi) / 2
        dist = float(np.linalg.norm(c - np.asarray(pose.pos)))
        rng = cam.range_objects if t.kind == "object" else cam.range_landmarks
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        u, v, d = self.project(pose, corners)
        front = d > cam.near
        if not front.any():
            return Sighting(t.key, t.kind, False, False, 0.0, 0.0, 0.0, dist, None, "out_of_frustum")
        uu, vv = u[front], v[front]
        u0, u1 = float(np.clip(uu.min(), 0, cam.width)), float(np.clip(uu.max(), 0, cam.width))
        v0, v1 = float(np.clip(vv.min(), 0, cam.height)), float(np.clip(vv.max(), 0, cam.height))
        bbox_px = max(0.0, u1 - u0) * max(0.0, v1 - v0)
        if bbox_px <= 0.0:
            return Sighting(t.key, t.kind, False, False, 0.0, 0.0, 0.0, dist, None, "out_of_frustum")
        samples = np.vstack([c, c + 0.75 * (corners - c)])
        su, sv, sd = self.project(pose, samples)
        in_img = (sd > cam.near) & (su >= 0) & (su <= cam.width) & (sv >= 0) & (sv <= cam.height)
        if not in_img.any():
            # the box straddles the image border with no sample inside: count the centre ray only
            in_img = np.zeros(len(samples), dtype=bool)
            in_img[0] = True
        clear = np.zeros(len(samples), dtype=bool)
        clear[in_img] = self._ray_clear(pose, samples[in_img], t, c)
        frac = float(clear[in_img].mean()) if in_img.any() else 0.0
        px = bbox_px * frac
        bbox = (round(u0, 1), round(v0, 1), round(u1, 1), round(v1, 1))
        if not clear.any():
            return Sighting(t.key, t.kind, False, True, 0.0, bbox_px, 0.0, dist, bbox, "occluded")
        if dist > rng:
            return Sighting(t.key, t.kind, False, True, px, bbox_px, frac, dist, bbox, "too_far")
        if px < cam.min_pixels:
            return Sighting(t.key, t.kind, False, True, px, bbox_px, frac, dist, bbox, "too_small")
        return Sighting(t.key, t.kind, True, True, px, bbox_px, frac, dist, bbox, "")

    def sight_all(self, pose: CameraPose, targets: Iterable[Target]) -> dict[str, Sighting]:
        out = {}
        p = np.asarray(pose.pos)
        fwd = pose.R[:, 0]
        for t in targets:
            c = (np.asarray(t.box[0]) + np.asarray(t.box[1])) / 2
            rel = c - p
            half_diag = float(np.linalg.norm(np.asarray(t.box[1]) - np.asarray(t.box[0]))) / 2
            rng = self.cam.range_objects if t.kind == "object" else self.cam.range_landmarks
            if float(np.linalg.norm(rel)) > rng + half_diag + 0.5 or float(rel @ fwd) < -half_diag:
                out[t.key] = Sighting(t.key, t.kind, False, False, 0.0, 0.0, 0.0, float(np.linalg.norm(rel)),
                                      None, "out_of_frustum")
                continue
            out[t.key] = self.sight(pose, t)
        return out


def structure_mask(occ: np.ndarray, raw_codes: np.ndarray | None, boxes: Iterable[tuple], res: float,
                   origin: tuple[float, float], dilate_cells: int = 2) -> np.ndarray:
    """Wall/structure cells: blocked or outside cells that no scene item footprint explains."""
    from scipy import ndimage
    H, W = occ.shape
    explained = np.zeros((H, W), dtype=bool)
    for b in boxes:
        ix0 = max(0, int(math.floor((b[0][0] - origin[0]) / res)))
        ix1 = min(W - 1, int(math.floor((b[1][0] - origin[0]) / res)))
        iy0 = max(0, int(math.floor((b[0][1] - origin[1]) / res)))
        iy1 = min(H - 1, int(math.floor((b[1][1] - origin[1]) / res)))
        if ix1 >= ix0 and iy1 >= iy0:
            explained[iy0:iy1 + 1, ix0:ix1 + 1] = True
    if dilate_cells:
        explained = ndimage.binary_dilation(explained, iterations=dilate_cells)
    blocked = occ.copy()
    if raw_codes is not None:
        blocked = (raw_codes != 0)                        # the z-band obstacles + outside; `low` props never occlude
    return blocked & ~explained


def is_closed_container(thor_type: str) -> bool:
    return vocab.is_container(thor_type) and not vocab.is_pickupable(thor_type)
