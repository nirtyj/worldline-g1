"""P1 M2b wire helpers (docs/contracts/p1_m2b.md): camera specs, frame metadata, object records, health level, the
object-fall detector and argument parsing.

Pure numpy (no Isaac imports), so sim_isaac/app.py (Isaac, /work/envs/isaaclab) and tools/fake_p1.py (the kinematic
fake, any venv) build byte-identical messages from the same code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim_isaac.mathutil import quat_mul, quat_rotate, quat_rotate_inverse

CONTRACT = "m2b-1"
TOPICS = ["gt.pose", "gt.objects", "gt.event", "sim.health"]
EGO_PORT = 5566                      # base port of the ego_view PUB (contract §1); + port offset

# palm frame = <arm>_wrist_yaw_link x (0.0415, +-0.003, 0), identity rotation: <arm>_hand_palm_joint
# (SONIC main.urdf:808-809, g1_29dof_with_hand.urdf:834-838; merged into wrist_yaw_link by merge_fixed_joints=True)
PALM_OFFSET = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}
GRIP_OFFSET = (0.07, 0.0, 0.0)       # default attach grip point in the palm frame (contract §4.1)
SNAP_M = 0.12
ARMS = ("left", "right")
ATTACH_MODES = ("follow", "fixed_joint")

# health (PLAN §3.5): unsafe = RTF < 0.85 for 3 s, degraded = RTF < 0.95 for 5 s
DEGRADED_RTF, DEGRADED_WINDOW_S = 0.95, 5.0
UNSAFE_RTF, UNSAFE_WINDOW_S = 0.85, 3.0

STATIONARY_V, STATIONARY_W = 0.05, 0.05   # world/frames.py: speed < 0.05 m/s and |wz| < 0.05 rad/s


# ------------------------------------------------------------------------------------------------ small math
def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def matrix_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_about_y(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])


def quat_about_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def world_quat_from_ros_optical(q_xyzw) -> np.ndarray:
    """A camera orientation in ROS optical convention (x right, y down, z forward; quaternion x,y,z,w as in Isaac
    Lab's OffsetCfg(convention="ros")) -> the same camera in the 'world' convention (x forward, y left, z up), wxyz."""
    x, y, z, w = (float(v) for v in q_xyzw)
    R = matrix_from_quat(np.array([w, x, y, z]) / math.sqrt(w * w + x * x + y * y + z * z))
    return quat_from_matrix(np.column_stack([R[:, 2], -R[:, 0], -R[:, 1]]))


def forward_yaw_pitch(q_wxyz) -> tuple[float, float]:
    """(yaw ccw from +x, pitch_down) of the +x axis of a world-convention frame (rad)."""
    f = quat_rotate(np.asarray(q_wxyz, dtype=np.float64), np.array([1.0, 0.0, 0.0]))
    return math.atan2(f[1], f[0]), math.asin(max(-1.0, min(1.0, -f[2])))


def compose(p_a, q_a, p_b, q_b) -> tuple[np.ndarray, np.ndarray]:
    """T_a * T_b for (pos, quat_wxyz) pairs."""
    q_a = np.asarray(q_a, dtype=np.float64)
    return np.asarray(p_a, dtype=np.float64) + quat_rotate(q_a, np.asarray(p_b, dtype=np.float64)), \
        quat_mul(q_a, np.asarray(q_b, dtype=np.float64))


def relative(p_a, q_a, p_b, q_b) -> tuple[np.ndarray, np.ndarray]:
    """T_a^-1 * T_b."""
    q_a = np.asarray(q_a, dtype=np.float64)
    qi = np.array([q_a[0], -q_a[1], -q_a[2], -q_a[3]])
    return quat_rotate_inverse(q_a, np.asarray(p_b, dtype=np.float64) - np.asarray(p_a, dtype=np.float64)), \
        quat_mul(qi, np.asarray(q_b, dtype=np.float64))


def yaw_map_deg(yaw_isaac_rad: float) -> float:
    """world/coords.py yaw_map_deg: Isaac yaw (rad, ccw from +x) -> Worldline yaw (deg, cw from +y, [0, 360))."""
    a = (90.0 - math.degrees(yaw_isaac_rad)) % 360.0
    return 0.0 if abs(a - 360.0) < 1e-9 else a


def r(v, nd: int = 4):
    """Round a float or a sequence for the wire."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), nd) for x in v]
    return round(float(v), nd)


# ------------------------------------------------------------------------------------------------ cameras
@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    clipping: tuple[float, float]
    mount_xyz: tuple[float, float, float]          # in the parent link frame (m)
    mount_quat_wxyz: tuple[float, float, float, float]   # 'world' convention: x forward, y left, z up
    parent: str = "torso_link"
    port: str = "camera"                           # "camera" (5565) or "ego" (5566)
    default_on: bool = True
    default_hz: float = 30.0
    source: str = ""

    @property
    def key(self) -> str:
        return self.name

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.horizontal_aperture_mm / 2 / self.focal_length_mm))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.vertical_aperture_mm / 2 / self.focal_length_mm))

    @property
    def fx_px(self) -> float:
        return self.focal_length_mm / self.horizontal_aperture_mm * self.width

    @property
    def fy_px(self) -> float:
        return self.focal_length_mm / self.vertical_aperture_mm * self.height

    @property
    def pitch_down_deg(self) -> float:
        return math.degrees(forward_yaw_pitch(self.mount_quat_wxyz)[1])

    def world_pose(self, parent_pos, parent_quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
        return compose(parent_pos, parent_quat_wxyz, self.mount_xyz, self.mount_quat_wxyz)

    def info(self) -> dict:
        return {"name": self.name, "key": self.key, "width": self.width, "height": self.height,
                "hfov_deg": r(self.hfov_deg, 3), "vfov_deg": r(self.vfov_deg, 3), "fx_px": r(self.fx_px, 3),
                "fy_px": r(self.fy_px, 3), "focal_length_mm": self.focal_length_mm,
                "horizontal_aperture_mm": self.horizontal_aperture_mm,
                "vertical_aperture_mm": self.vertical_aperture_mm, "clipping": list(self.clipping),
                "parent": self.parent, "mount_xyz": list(self.mount_xyz),
                "mount_quat_wxyz": r(self.mount_quat_wxyz, 6), "pitch_down_deg": r(self.pitch_down_deg, 3),
                "source": self.source}


def _vfov_apertures(vfov_deg: float, width: int, height: int, f: float = 10.0) -> tuple[float, float]:
    va = 2.0 * f * math.tan(math.radians(vfov_deg) / 2.0)
    return va * width / height, va


# Arena G1 head camera (IsaacLab-Arena release/0.2.1 @ 8b4a3a47, isaaclab_arena/embodiments/g1/g1.py:103-106,
# 508-539): on head_link, offset (0.04485, 0, 0.35325), ROS-convention rotation xyzw below, focal 15 mm, Isaac Lab
# PinholeCameraCfg default horizontal aperture 20.955 (sensors_cfg.py:61), vertical = 20.955 * 480 / 640
# (sensors/camera/camera.py:172-173), clipping (0.1, 5), 640x480.
ARENA_CAM_OFFSET_XYZ = (0.04485, 0.0, 0.35325)
ARENA_CAM_ROT_ROS_XYZW = (-0.62721, 0.62721, -0.32651, 0.32651)
# head_link on torso_link: SONIC main.urdf:587-588 == Arena's rev_1_0 USD head_joint localPos0 (read with usd-core)
HEAD_LINK_IN_TORSO = (0.0039635, 0.0, -0.044)

HEAD = CameraSpec(
    name="head", width=640, height=480, focal_length_mm=10.0, horizontal_aperture_mm=20.0, vertical_aperture_mm=15.0,
    clipping=(0.05, 50.0), mount_xyz=(0.06, 0.0, 0.526), mount_quat_wxyz=tuple(quat_about_y(math.radians(15.0))),
    port="camera", default_on=True, default_hz=30.0,
    source="docs/M2.md P1.4; world/perception.py:78-79 HEAD_SIM (HFOV 90, 15 deg down, torso + (0.06, 0, 0.526))")

EGO_VIEW = CameraSpec(
    name="ego_view", width=640, height=480, focal_length_mm=15.0, horizontal_aperture_mm=20.955,
    vertical_aperture_mm=20.955 * 480 / 640, clipping=(0.1, 5.0),
    mount_xyz=tuple(float(a + b) for a, b in zip(HEAD_LINK_IN_TORSO, ARENA_CAM_OFFSET_XYZ)),
    mount_quat_wxyz=tuple(float(v) for v in world_quat_from_ros_optical(ARENA_CAM_ROT_ROS_XYZW)),
    port="ego", default_on=False, default_hz=30.0,
    source="IsaacLab-Arena 8b4a3a47 isaaclab_arena/embodiments/g1/g1.py:103-106,508-539 (head_link/RobotHeadCam); "
           "head_link = torso_link + (0.0039635, 0, -0.044) (main.urdf:587-588, Arena rev_1_0 USD head_joint)")

_D435_HA, _D435_VA = _vfov_apertures(45.0, 640, 480)
D435 = CameraSpec(
    name="d435", width=640, height=480, focal_length_mm=10.0, horizontal_aperture_mm=_D435_HA,
    vertical_aperture_mm=_D435_VA, clipping=(0.05, 50.0), mount_xyz=(0.0576235, 0.01753, 0.41987),
    mount_quat_wxyz=tuple(quat_about_y(0.8307767239493009)), port="camera", default_on=True, default_hz=30.0,
    source="docs/contracts/m1.md 1.4 (d435_link of g1_29dof_with_hand.urdf:614-619, VFOV 45)")

CAMERA_SPECS = {c.name: c for c in (HEAD, EGO_VIEW, D435)}


def with_resolution(spec: CameraSpec, width: int, height: int) -> CameraSpec:
    """Same camera at another resolution (square pixels kept: the vertical aperture follows the aspect ratio)."""
    if (width, height) == (spec.width, spec.height):
        return spec
    from dataclasses import replace
    return replace(spec, width=width, height=height,
                   vertical_aperture_mm=spec.horizontal_aperture_mm * height / width)


def cam_pose_wl(pos, quat_wxyz) -> list[float]:
    """[map_x, map_z, yaw_map_deg, pitch_down_deg] as world/frames.py IsaacFrames stamps it (P1.10)."""
    yaw, pitch = forward_yaw_pitch(quat_wxyz)
    return [round(float(pos[0]), 3), round(float(pos[1]), 3), round(yaw_map_deg(yaw), 1),
            round(math.degrees(pitch), 1)]


def is_stationary(lin_vel_w, ang_vel_w) -> bool:
    return math.hypot(float(lin_vel_w[0]), float(lin_vel_w[1])) < STATIONARY_V and abs(float(ang_vel_w[2])) < STATIONARY_W


def frame_meta(spec: CameraSpec, *, seq: int, render_seq: int, t_sim: float, t_capture: float, t_capture_mono: float,
               cam_pos, cam_quat, base_lin_vel_w, base_ang_vel_w, jpeg_q: int = 80) -> dict:
    """The extra keys of a 5565/5566 frame message (contract §5.2), everything except the image and t_pub."""
    return {"camera": spec.name, "seq": int(seq), "render_seq": int(render_seq), "t_sim": round(float(t_sim), 4),
            "t_capture": float(t_capture), "t_capture_mono": float(t_capture_mono),
            "w": spec.width, "h": spec.height, "hfov": round(spec.hfov_deg, 3), "vfov": round(spec.vfov_deg, 3),
            "cam_pos": r(cam_pos, 4), "cam_quat_wxyz": r(cam_quat, 5),
            "cam_pose_wl": cam_pose_wl(cam_pos, cam_quat),
            "stationary": is_stationary(base_lin_vel_w, base_ang_vel_w),
            "base_speed": round(math.hypot(float(base_lin_vel_w[0]), float(base_lin_vel_w[1])), 4),
            "base_wz": round(float(base_ang_vel_w[2]), 4), "jpeg_q": int(jpeg_q)}


def gear_sonic_message(key: str, b64: str, meta: dict, t_pub: float) -> dict:
    """gear_sonic sensor_server dict (image_publish_utils.py:165-185) + the M2b extras. `timestamps` carries the
    capture time (meta["t_capture"]); consumers that ignore extras see an M1-format message."""
    d = {"timestamps": {key: float(meta.get("t_capture", t_pub))}, "images": {key: b64}, key: b64}
    d.update(meta)
    d["t_pub"] = float(t_pub)
    return d


# ------------------------------------------------------------------------------------------------ objects
def box_corners(aabb) -> np.ndarray:
    lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def moved_box(aabb0, p0, q0, p, q) -> list[list[float]]:
    """World AABB of the load-time box `aabb0` after the body moved from pose (p0, q0) to (p, q)."""
    c = box_corners(aabb0)
    q0 = np.asarray(q0, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    loc = np.array([quat_rotate_inverse(q0, v - np.asarray(p0)) for v in c])
    w = np.array([quat_rotate(q, v) for v in loc]) + np.asarray(p, dtype=np.float64)
    return [w.min(axis=0).tolist(), w.max(axis=0).tolist()]


def moved_point(pt0, p0, q0, p, q) -> np.ndarray:
    loc = quat_rotate_inverse(np.asarray(q0, dtype=np.float64), np.asarray(pt0, dtype=np.float64) - np.asarray(p0))
    return quat_rotate(np.asarray(q, dtype=np.float64), loc) + np.asarray(p, dtype=np.float64)


def matrices_from_quats(Q) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3) rotation matrices."""
    Q = np.asarray(Q, dtype=np.float64)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def body_local_points(points_w, P0, Q0) -> np.ndarray:
    """World points (N, K, 3) at load -> the same points in each body's frame (N, K, 3)."""
    R0 = matrices_from_quats(Q0)
    return np.einsum("nji,nkj->nki", R0, np.asarray(points_w, dtype=np.float64) - np.asarray(P0)[:, None, :])


def body_world_points(points_b, P, Q) -> np.ndarray:
    """Body-frame points (N, K, 3) -> world, for body poses P (N, 3), Q (N, 4) wxyz. Vectorized moved_box/point."""
    return np.einsum("nij,nkj->nki", matrices_from_quats(Q), points_b) + np.asarray(P)[:, None, :]


def box_center(aabb) -> np.ndarray:
    return (np.asarray(aabb[0], dtype=np.float64) + np.asarray(aabb[1], dtype=np.float64)) / 2.0


def object_record(oid: str, name: str, pos, quat_wxyz, aabb, *, held_by: str | None, dynamic: bool,
                  lin_vel=(0.0, 0.0, 0.0)) -> dict:
    v = [0.0, 0.0, 0.0] if held_by else [float(x) for x in lin_vel]
    return {"id": oid, "name": name, "pos": r(pos, 4), "quat_wxyz": r(quat_wxyz, 5),
            "aabb": [r(aabb[0], 4), r(aabb[1], 4)], "held_by": held_by, "dynamic": bool(dynamic),
            "source": "sim" if dynamic else "static", "lin_vel": r(v, 3),
            "moving": bool(math.sqrt(sum(x * x for x in v)) > 0.02)}


def parse_pose_arg(pose) -> tuple[np.ndarray, float | None]:
    """detach/move_object `pose`: [x,y,z] | [x,y,z,yaw] | {"pos": [..], "yaw": ..} | {"x","y","z","yaw"}
    -> (centre xyz, yaw or None)."""
    if isinstance(pose, dict):
        if "pos" in pose:
            c = [float(v) for v in pose["pos"]][:3]
        else:
            c = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
        y = pose.get("yaw")
        return np.array(c), (None if y is None else float(y))
    vals = [float(v) for v in pose]
    if len(vals) not in (3, 4):
        raise ValueError(f"pose needs 3 or 4 numbers, got {len(vals)}")
    return np.array(vals[:3]), (vals[3] if len(vals) == 4 else None)


def placed_pose(aabb0, p0, q0, center, yaw: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Body pose that puts the load-time box's centre at `center`, orientation = load-time orientation rotated about
    world z by `yaw` (contract §4.2)."""
    q = np.asarray(q0, dtype=np.float64) if yaw is None else quat_mul(quat_about_z(float(yaw)), np.asarray(q0))
    c0 = box_center(aabb0)
    # the box centre is a body-fixed point: centre_new = p + R(q) R(q0)^-1 (c0 - p0)
    off = quat_rotate(q, quat_rotate_inverse(np.asarray(q0, dtype=np.float64), c0 - np.asarray(p0)))
    return np.asarray(center, dtype=np.float64) - off, q


class FallDetector:
    """object_fell (contract §10.2): a dynamic, non-held object came to rest (speed < v_rest for rest_s) with its box
    bottom at least drop_m lower than where it last rested."""

    def __init__(self, drop_m: float = 0.10, v_rest: float = 0.05, rest_s: float = 0.3, floor_z: float = 0.0,
                 floor_tol: float = 0.05):
        self.drop_m, self.v_rest, self.rest_s = drop_m, v_rest, rest_s
        self.floor_z, self.floor_tol = floor_z, floor_tol
        self.rest_z: dict[str, float] = {}
        self._still_since: dict[str, float] = {}

    def seed(self, oid: str, z_bottom: float) -> None:
        """Record a known resting height (objects at their load pose rest there)."""
        self.rest_z[oid] = float(z_bottom)

    def reset(self, oid: str | None = None) -> None:
        if oid is None:
            self.rest_z.clear()
            self._still_since.clear()
        else:
            self.rest_z.pop(oid, None)
            self._still_since.pop(oid, None)

    def update(self, oid: str, t: float, z_bottom: float, speed: float, held: bool, pos=None) -> dict | None:
        if held:
            self.rest_z.pop(oid, None)
            self._still_since.pop(oid, None)
            return None
        if speed >= self.v_rest:
            self._still_since.pop(oid, None)
            return None
        t0 = self._still_since.setdefault(oid, t)
        if t - t0 < self.rest_s:
            return None
        prev = self.rest_z.get(oid)
        self.rest_z[oid] = z_bottom
        if prev is not None and prev - z_bottom >= self.drop_m:
            return {"id": oid, "from_z": round(prev, 3), "to_z": round(z_bottom, 3), "drop_m": round(prev - z_bottom, 3),
                    "pos": r(pos, 3) if pos is not None else None,
                    "on_floor": bool(z_bottom - self.floor_z <= self.floor_tol)}
        return None


# ------------------------------------------------------------------------------------------------ health
def health_level(rtf_3s: float | None, rtf_5s: float | None) -> tuple[str, str | None]:
    if rtf_3s is not None and rtf_3s < UNSAFE_RTF:
        return "unsafe", f"rtf_3s {rtf_3s:.3f} < {UNSAFE_RTF}"
    if rtf_5s is not None and rtf_5s < DEGRADED_RTF:
        return "degraded", f"rtf_5s {rtf_5s:.3f} < {DEGRADED_RTF}"
    return "ok", None


# ------------------------------------------------------------------------------------------------ errors / consumers
def err(code: str, msg: str = "") -> dict:
    return {"ok": False, "code": code, "error": f"{code}: {msg}" if msg else code}


class OpError(Exception):
    """Raised inside an op handler; the server replies err(code, msg)."""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code, self.msg = code, msg

    def reply(self) -> dict:
        return err(self.code, self.msg)


@dataclass
class Consumers:
    """Who keeps a camera on (contract §5.4): consumer -> expiry (monotonic s) or None (no TTL)."""
    items: dict = field(default_factory=dict)

    def add(self, name: str, now: float, ttl_s: float | None = None) -> None:
        self.items[name] = None if not ttl_s else now + float(ttl_s)

    def remove(self, name: str | None) -> None:
        if name is None:
            self.items.clear()
        else:
            self.items.pop(name, None)

    def expire(self, now: float) -> list[str]:
        gone = [k for k, t in self.items.items() if t is not None and now >= t]
        for k in gone:
            del self.items[k]
        return gone

    def names(self) -> list[str]:
        return sorted(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


def next_due(t_sim: float, hz: float) -> float:
    """Next time on the sim-time grid k/hz strictly after t_sim (cameras of commensurate rates share renders)."""
    if hz <= 0:
        return math.inf
    return (math.floor(t_sim * hz + 1e-6) + 1) / hz
