"""World <-> SONIC planner frame.

Why a frame conversion is needed (all verified in WBC @b042411):
- planner `movement` / `facing` are directions in the planner's "world" frame (docs/source/references/planner_onnx.md:42-50,
  74-77, 152-176).
- On (re)initialisation the planner context is rotated to ZERO yaw at the origin, whatever the robot's real heading
  (localmotion_kplanner.hpp:591-624 "Normalize coordinate frame so first context frame faces zero yaw").
- The policy maps that reference into the robot frame with apply_delta_heading = heading(init_base_quat) *
  heading(ref_first_frame)^-1, then * euler_z(delta_heading) (g1_deploy_onnx_ref.cpp:560-606). init_base_quat is the
  pelvis IMU quaternion captured on start (reinitialize_heading), delta_heading starts at 0.
- Both are published in g1_debug as `init_base_quat` / `delta_heading` (zmq_output_handler.hpp:43-45).

So  yaw_world = yaw_planner + theta0,  theta0 = yaw(init_base_quat) + delta_heading.
In sim the IMU quaternion P1 publishes on rt/lowstate is the pelvis world orientation, i.e. the same frame as gt.pose.

Fallback before g1_debug carries init_base_quat: theta0 = gt yaw captured when we send command{start}
(the robot is static on the band then).

On top of that a slow integral `bias` absorbs any steady heading tracking error measured with GT
(only updated while the robot is settled with a constant facing command).
"""

from __future__ import annotations

import math
import threading

from .wire import wrap, yaw_from_quat_wxyz


class PlannerFrame:
    def __init__(self, bias_ki: float = 0.4, bias_max_deg: float = 20.0):
        self._lock = threading.Lock()
        self.theta0: float | None = None
        self.delta_heading: float = 0.0
        self.source: str = "none"
        self.bias: float = 0.0
        self.bias_ki = bias_ki
        self.bias_max = math.radians(bias_max_deg)
        self.init_quat: list[float] | None = None

    # -- sources -------------------------------------------------------------------------------
    def set_fallback(self, gt_yaw: float) -> None:
        with self._lock:
            if self.source != "g1_debug":
                self.theta0 = float(gt_yaw)
                self.delta_heading = 0.0
                self.source = "gt_at_start"

    def update_from_debug(self, init_base_quat, delta_heading) -> bool:
        """Returns True if the frame changed (e.g. deploy restarted / heading re-initialised)."""
        if init_base_quat is None:
            return False
        q = [float(v) for v in init_base_quat]
        d = float(delta_heading or 0.0)
        with self._lock:
            changed = self.init_quat is None or any(abs(a - b) > 1e-6 for a, b in zip(q, self.init_quat)) \
                or abs(d - self.delta_heading) > 1e-6
            if changed:
                self.init_quat = q
                self.theta0 = yaw_from_quat_wxyz(q)
                self.delta_heading = d
                if self.source == "g1_debug":
                    self.bias = 0.0  # a new heading init invalidates the learned bias
                self.source = "g1_debug"
            return changed

    def reset(self) -> None:
        with self._lock:
            self.theta0 = None
            self.delta_heading = 0.0
            self.source = "none"
            self.bias = 0.0
            self.init_quat = None

    # -- conversion ----------------------------------------------------------------------------
    @property
    def known(self) -> bool:
        return self.theta0 is not None

    @property
    def offset(self) -> float:
        """planner -> world rotation angle (theta0 + delta_heading - bias)."""
        if self.theta0 is None:
            return 0.0
        return self.theta0 + self.delta_heading - self.bias

    def vec_world_to_planner(self, vx: float, vy: float) -> tuple[float, float]:
        a = -self.offset
        c, s = math.cos(a), math.sin(a)
        return c * vx - s * vy, s * vx + c * vy

    def yaw_world_to_planner(self, yaw_w: float) -> float:
        return wrap(yaw_w - self.offset)

    def facing_vec(self, yaw_w: float) -> tuple[float, float, float]:
        yp = self.yaw_world_to_planner(yaw_w)
        return math.cos(yp), math.sin(yp), 0.0

    # -- outer loop ----------------------------------------------------------------------------
    def observe_settled(self, commanded_yaw_w: float, gt_yaw: float, dt: float) -> None:
        """Integrate heading error while settled.

        yaw_p = cmd - (theta0 + delta - bias); robot settles at yaw_p + theta0 + delta + e_track = cmd + bias + e_track.
        Observed e_obs = gt - cmd = bias + e_track, so bias -= ki * e_obs * dt converges to bias = -e_track.
        """
        e = wrap(gt_yaw - commanded_yaw_w)
        if abs(e) > math.radians(30):  # not settled, or a gross frame error: don't learn from it
            return
        with self._lock:
            self.bias = max(-self.bias_max, min(self.bias_max, self.bias - self.bias_ki * e * dt))

    def to_dict(self) -> dict:
        return {"known": self.known, "source": self.source,
                "theta0_deg": None if self.theta0 is None else round(math.degrees(self.theta0), 2),
                "delta_heading": self.delta_heading, "bias_deg": round(math.degrees(self.bias), 2)}
