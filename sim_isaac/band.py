"""Elastic band: a virtual spring-damper holding the pelvis until the controller takes over.

Port of gear_sonic's MuJoCo ElasticBand ($WBC/gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:366-395),
applied like MuJoCo's xfrc_applied (world-frame wrench, base_sim.py:394-414).

Differences from MuJoCo (documented in docs/contracts/m1.md 1.7):
- the anchor is (x, y) of the pelvis at engage time and floor_z + height, not the fixed point (0, 0, 1);
- the orientation target is upright with the yaw at engage time (MuJoCo drives towards yaw 0);
- release can ramp the gains down linearly over ramp_s seconds.
"""
from __future__ import annotations

import numpy as np

from .mathutil import quat_conj, quat_from_yaw, quat_mul, quat_to_rotvec, yaw_from_quat


class ElasticBand:
    # gains from unitree_sdk2py_bridge.py:372-375
    KP_POS = 10000.0
    KD_POS = 1000.0
    KP_ANG = 1000.0
    KD_ANG = 10.0

    def __init__(self):
        self.enabled = False
        self.anchor = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.scale = 0.0
        self._ramp_rate = 0.0  # scale units per second while releasing

    def engage(self, pos: np.ndarray, quat_wxyz: np.ndarray, z_world: float) -> None:
        self.anchor = np.array([pos[0], pos[1], z_world], dtype=np.float64)
        self.target_quat = quat_from_yaw(yaw_from_quat(quat_wxyz))
        self.enabled = True
        self.scale = 1.0
        self._ramp_rate = 0.0

    def release(self, ramp_s: float = 0.0) -> None:
        if not self.enabled:
            return
        if ramp_s <= 0.0:
            self.enabled = False
            self.scale = 0.0
        else:
            self._ramp_rate = 1.0 / ramp_s

    def advance(self, dt: float, pos, quat_wxyz, lin_vel_w, ang_vel_w) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (force_w, torque_w) to apply at the pelvis, or None when the band is off."""
        if not self.enabled:
            return None
        if self._ramp_rate > 0.0:
            self.scale -= self._ramp_rate * dt
            if self.scale <= 0.0:
                self.enabled = False
                self.scale = 0.0
                self._ramp_rate = 0.0
                return None
        f = self.KP_POS * (self.anchor - np.asarray(pos)) - self.KD_POS * np.asarray(lin_vel_w)
        q_err = quat_mul(np.asarray(quat_wxyz, dtype=np.float64), quat_conj(self.target_quat))
        torque = -self.KP_ANG * quat_to_rotvec(q_err) - self.KD_ANG * np.asarray(ang_vel_w)
        return self.scale * f, self.scale * torque
