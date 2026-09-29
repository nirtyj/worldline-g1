"""The only conversion point between Isaac/REP-103 world coordinates and Worldline's map frame (PLAN §6.2.3).

Isaac (and every service, the body and P1): x, y in metres on the floor, z up; yaw in radians, counter-clockwise
from +x (REP-103).

Worldline (agent/, layout, belief, UI, the THOR-era map dicts): the floor pair is called (x, z) and stored as
"xy": [x, z]; heights are "y"; yaw is in degrees, clockwise from +z, i.e. heading = atan2(dx, dz).

    map_x = isaac_x,  map_z = isaac_y,  height = isaac_z
    yaw_map_deg = (90 - yaw_isaac_ccw_deg) mod 360

Handedness is preserved: from above, facing +y (map yaw 0) has +x on the right, exactly as facing +z has +x on
the right in Unity/THOR, so layout.Line.along's right-hand rule `(facing[1], -facing[0])` keeps working.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def wrap_pi(a: float) -> float:
    """Wrap an angle (rad) to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def wrap_deg(a: float) -> float:
    """Wrap an angle (deg) to [0, 360)."""
    a = a % 360.0
    return 0.0 if abs(a - 360.0) < 1e-9 else a


def ang_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b (rad), in (-pi, pi]."""
    return wrap_pi(a - b)


# ---------------------------------------------------------------- positions
def to_map_xz(x: float, y: float) -> tuple[float, float]:
    """Isaac floor (x, y) -> Worldline map (x, z)."""
    return float(x), float(y)


def to_isaac_xy(map_x: float, map_z: float) -> tuple[float, float]:
    """Worldline map (x, z) -> Isaac floor (x, y)."""
    return float(map_x), float(map_z)


def thor_pose(x: float, y: float, z: float, nd: int = 3) -> dict:
    """Isaac (x, y, z-up) -> THOR-shaped pose dict {x, y (height), z (floor)} used by perception() and belief."""
    return {"x": round(float(x), nd), "y": round(float(z), nd), "z": round(float(y), nd)}


def map_pos(x: float, y: float, z: float | None = None, nd: int = 2) -> list[float]:
    """Isaac point -> Worldline look item "pos": [x, z] or [x, z, h] (h = height above the floor)."""
    out = [round(float(x), nd), round(float(y), nd)]
    if z is not None:
        out.append(round(float(z), nd))
    return out


# ---------------------------------------------------------------- yaw
def yaw_map_deg(yaw_isaac_rad: float) -> float:
    """Isaac yaw (rad, ccw from +x) -> Worldline yaw (deg, cw from +z/+y, [0, 360))."""
    return wrap_deg(90.0 - math.degrees(yaw_isaac_rad))


def yaw_isaac_rad(yaw_map: float) -> float:
    """Worldline yaw (deg, cw from +y) -> Isaac yaw (rad, ccw from +x, (-pi, pi])."""
    return wrap_pi(math.radians(90.0 - yaw_map))


def heading_map_deg(dx: float, dz: float) -> float:
    """Worldline heading of a floor vector (map dx, dz): atan2(dx, dz) in degrees, [0, 360)."""
    return wrap_deg(math.degrees(math.atan2(dx, dz)))


def heading_isaac_rad(dx: float, dy: float) -> float:
    """Isaac heading of a floor vector (dx, dy): atan2(dy, dx) in radians."""
    return math.atan2(dy, dx)


# ---------------------------------------------------------------- quaternions / rotations (no numpy needed)
def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_to_R(q: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """World-from-body rotation matrix of a unit quaternion [w, x, y, z] (P1 `quat_wxyz`)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def rot_z(yaw: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_y(pitch: float) -> tuple[tuple[float, float, float], ...]:
    c, s = math.cos(pitch), math.sin(pitch)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def mat_vec(m, v: Iterable[float]) -> tuple[float, float, float]:
    v = tuple(v)
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))   # type: ignore[return-value]


def mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def body_to_world(pose_xy_yaw: tuple[float, float, float], dx: float, dy: float) -> tuple[float, float]:
    """A point (dx forward, dy left) in a planar body frame -> world floor xy."""
    x, y, yaw = pose_xy_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * dx - s * dy, y + s * dx + c * dy


def world_to_body(pose_xy_yaw: tuple[float, float, float], px: float, py: float) -> tuple[float, float]:
    """A world floor point -> (forward, left) in the planar body frame."""
    x, y, yaw = pose_xy_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - x, py - y
    return c * dx + s * dy, -s * dx + c * dy
