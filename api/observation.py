"""Observations (PLAN 5.10; doc 45).

``LookData`` is exactly what ``agent/state.BeliefState.apply_observation``
consumes (the ex-THOR look payload). A glance has ``views=[]`` and never marks
anything absent; a scan reports its views, and absence is asserted only for
positions inside the 3-D view frustum (``in_frustum``).

Frames here are Worldline's: (x, z) in metres, yaw in degrees clockwise from +z
(``atan2(dx, dz)``), tilt in degrees below horizontal.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

HEAD_HFOV_DEG = 90.0
HEAD_VFOV_DEG = 73.7          # 90 deg HFOV at 4:3
HEAD_TILT_DEG = 15.0
SCAN_PITCHED_TILT_DEG = 35.0  # the second scan row (waist pitch +20 deg)
SCAN_YAWS_DEG = (-35.0, 0.0, 35.0)
VIEW_RANGE_M = 2.5
FRUSTUM_MARGIN_DEG = 3.0
YAW_MARGIN_DEG = 5.0          # kept from Worldline's 2-D test
GLANCE_PREFIX = "obs-g"


@dataclass(frozen=True)
class ViewSpec:
    x: float
    z: float
    yaw: float
    tilt: float = HEAD_TILT_DEG
    fov: float = HEAD_HFOV_DEG
    range: float = VIEW_RANGE_M
    vfov: float | None = None        # vertical FOV (deg)
    cam_h: float | None = None       # camera height (m) at capture
    near: float = 0.3                # nothing closer than this is asserted absent

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class LookData:
    at: str | None
    surfaces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # {surface: [{id, type, brand, color, label, surface, pos: [x, z, h]}]}; h = object height (m)
    landmarks: list[dict[str, Any]] = field(default_factory=list)   # [{id, type, label, near, pos: [x, z]}]
    views: list[ViewSpec] = field(default_factory=list)            # [] for a glance
    hands: dict[str, str | None] = field(default_factory=lambda: {"left": None, "right": None})

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


@dataclass(frozen=True)
class RobotObservation:
    observation_id: str
    timestamp: float
    mode: Literal["glance", "scan"]
    look: LookData
    head_rev: int = 0
    ego_rev: int = 0
    panorama_revs: list[int] = field(default_factory=list)
    source: Literal["isaac-gt", "lite-gt", "perception"] = "lite-gt"

    def data(self) -> dict[str, Any]:
        """The ToolResult.data payload of an internal observe."""
        d = self.look.to_dict()
        d.update({"observation_id": self.observation_id, "mode": self.mode, "source": self.source,
                  "head_rev": self.head_rev})
        return d


def glance_id(rev: int) -> str:
    return f"{GLANCE_PREFIX}{int(rev)}"


def is_glance_id(observation_id: str | None) -> bool:
    return bool(observation_id) and str(observation_id).startswith(GLANCE_PREFIX)


def _get(view: Mapping[str, Any] | ViewSpec, key: str, default: Any = None) -> Any:
    if isinstance(view, ViewSpec):
        return getattr(view, key, default)
    return view.get(key, default)


def in_frustum(view: Mapping[str, Any] | ViewSpec, pos: tuple[float, ...] | list[float],
               h: float | None = None) -> bool:
    """Was this spot inside a view?

    2-D (Worldline's old rule): within range - 0.1 m and within fov/2 - 5 deg of yaw.
    3-D when the view has vfov and cam_h and a height is known (``h`` or pos[2]):
    also d >= near and the elevation angle atan2(cam_h - h, d) inside
    [tilt - vfov/2 + 3, tilt + vfov/2 - 3] degrees below horizontal."""
    x0, z0 = float(_get(view, "x")), float(_get(view, "z"))
    dx, dz = float(pos[0]) - x0, float(pos[1]) - z0
    d = math.hypot(dx, dz)
    if d > float(_get(view, "range", 1.5) or 1.5) - 0.1:
        return False
    off = (math.degrees(math.atan2(dx, dz)) - float(_get(view, "yaw")) + 180) % 360 - 180
    if abs(off) > float(_get(view, "fov", 90) or 90) / 2 - YAW_MARGIN_DEG:
        return False
    vfov, cam_h = _get(view, "vfov"), _get(view, "cam_h")
    height = h if h is not None else (float(pos[2]) if len(pos) > 2 and pos[2] is not None else None)
    if vfov is None or cam_h is None or height is None:
        return True
    if d < float(_get(view, "near", 0.3) or 0.0):
        return False
    below = math.degrees(math.atan2(float(cam_h) - float(height), max(d, 1e-6)))
    tilt = float(_get(view, "tilt", HEAD_TILT_DEG) or 0.0)
    half = float(vfov) / 2.0 - FRUSTUM_MARGIN_DEG
    return tilt - half <= below <= tilt + half


def scan_views(x: float, z: float, yaw: float, cam_h: float, *, rows: tuple[float, ...] = (HEAD_TILT_DEG,
               SCAN_PITCHED_TILT_DEG), yaws: tuple[float, ...] = SCAN_YAWS_DEG) -> list[ViewSpec]:
    """The two-row waist scan's views (PLAN 6.5), for lite and tests."""
    return [ViewSpec(x=x, z=z, yaw=(yaw + dy) % 360, tilt=t, fov=HEAD_HFOV_DEG, range=VIEW_RANGE_M,
                     vfov=HEAD_VFOV_DEG, cam_h=cam_h) for t in rows for dy in yaws]


__all__ = ["HEAD_HFOV_DEG", "HEAD_VFOV_DEG", "HEAD_TILT_DEG", "SCAN_PITCHED_TILT_DEG", "SCAN_YAWS_DEG",
           "VIEW_RANGE_M", "GLANCE_PREFIX", "ViewSpec", "LookData", "RobotObservation", "glance_id",
           "is_glance_id", "in_frustum", "scan_views"]
