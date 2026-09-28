"""Small numpy quaternion helpers. Quaternions are (w, x, y, z), like Isaac Lab and the Unitree IMU."""
from __future__ import annotations

import math

import numpy as np


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (body -> world when q is the body orientation)."""
    w = q[0]
    u = np.asarray(q[1:4], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q^-1 (world -> body)."""
    return quat_rotate(quat_conj(q), v)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float64)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:4])
    if s < 1e-9:
        return 2.0 * q[1:4]
    angle = 2.0 * math.atan2(s, q[0])
    return q[1:4] / s * angle


def up_z(q: np.ndarray) -> float:
    """z component of the body +z axis in world (1 = upright, 0 = lying)."""
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
