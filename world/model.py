"""WorldModel, SimControl and Localizer: the semantic interfaces between services and ground truth (PLAN §6.2.1).

`WorldModel` is the ONLY ground-truth reader on the runtime side. Its methods are semantic ("what does the head
camera see", "where is alarm_clock_1", "a free spot on this surface") so a perception implementation can replace
the GT one (§13) without changing services or the harness. Actuation that only a simulator can do (teleport,
attach, band, resets) is in the separate `SimControl`, which raises `NotSupported` on the real robot and on P1
builds that lack the op (M2a: P1 has no attach/detach/object-pose ops yet; see the M2b list in the report).

Geometry here is Isaac/REP-103 (x, y floor; z up; yaw rad ccw from +x). Dict outputs that go to agent/ (look data,
perception, truth) are converted to Worldline's frame through world/coords.py and say so in their docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Arm = Literal["left", "right"]
Box = tuple[tuple[float, float, float], tuple[float, float, float]]


class NotSupported(RuntimeError):
    """This backend cannot do that (real robot, or a P1 build without the op)."""


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float                         # pelvis height, world
    yaw: float                       # rad
    t: float = 0.0                   # wall time of the sample
    source: str = "isaac-gt"         # "isaac-gt" | "lite" | "kiss-icp-ekf"
    fallen: bool = False
    pelvis_z: float = 0.78           # above the floor
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    @property
    def xy_yaw(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw

    @property
    def speed(self) -> float:
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    # duck-compatible with api.types.Pose3D (upright: yaw only)
    @property
    def qw(self) -> float:
        import math
        return math.cos(self.yaw / 2)

    @property
    def qx(self) -> float:
        return 0.0

    @property
    def qy(self) -> float:
        return 0.0

    @property
    def qz(self) -> float:
        import math
        return math.sin(self.yaw / 2)

    def pose2d(self):
        from api.types import Pose2D
        return Pose2D(self.x, self.y, self.yaw)


@dataclass(frozen=True)
class ObjectState:
    oid: str
    type: str                        # snake THOR type
    label: str
    box: Box                         # current AABB (world)
    where: str                       # world/where.py vocabulary
    where_rule: str = ""
    held_by: str | None = None       # "left" | "right"
    pose_source: str = "scene"       # "scene" (static scene_info), "sim" (live P1 pose), "attach" (follows a hand)

    @property
    def pos(self) -> tuple[float, float, float]:
        return tuple((self.box[0][i] + self.box[1][i]) / 2 for i in range(3))   # type: ignore[return-value]

    @property
    def bottom(self) -> float:
        return self.box[0][2]

    # duck-compatible with api.types.ObjectState
    @property
    def id(self) -> str:
        return self.oid

    @property
    def pose(self):
        from api.types import Pose3D
        return Pose3D(*self.pos)


@dataclass(frozen=True)
class Detection:
    """One thing the camera sees now (GT hint; `source` says so)."""
    id: str
    kind: Literal["object", "landmark"]
    type: str
    label: str
    where: str | None                 # objects: where vocabulary; landmarks: None
    pos: tuple[float, float, float]
    dist: float
    px: float                         # estimated visible pixels
    near: str | None = None           # landmarks: the keypoint to stand at
    method: str = "gt-geometric"

    @property
    def pose(self):                   # api.types.Detection compatibility
        from api.types import Pose3D
        return Pose3D(*self.pos)


@dataclass(frozen=True)
class GraspState:
    object_id: str
    arm: str
    held: bool
    lifted_m: float = 0.0             # height above its support at grasp time
    attach_mode: str | None = None    # "follow" | "fixed_joint" | "kinematic" | None
    source: str = "gt"


@dataclass
class TruthSnapshot:
    """UI + eval only; never passed to the planner. THOR `ThorWorld.objects()` shape, Worldline frame."""
    t: float
    robot: dict[str, Any]
    objects: dict[str, dict[str, Any]]
    landmarks: dict[str, dict[str, Any]]
    things: dict[str, str] = field(default_factory=dict)      # every scene item id -> snake type (vocabulary)
    hands: dict[str, str | None] = field(default_factory=dict)
    source: str = "isaac-gt"


@runtime_checkable
class WorldModel(Protocol):
    scene: str
    source: str                                        # "isaac-gt" | "lite-gt" | "perception"

    # static semantic map (doc §27)
    def static_map(self) -> Any: ...                   # world.mapgen.StaticMap
    def lookup_keypoints(self) -> dict: ...

    # dynamic state
    def robot_pose(self) -> RobotPose: ...
    def detections(self, camera: str = "head", *, yaw_offset: float = 0.0) -> list[Detection]: ...
    def look(self, views: list[dict] | None, *, at: str | None, detections: list[list[Detection]] | None = None
             ) -> dict: ...
    def object(self, object_id: str) -> ObjectState | None: ...
    def objects(self) -> dict[str, ObjectState]: ...
    def hands(self) -> dict[str, str | None]: ...
    def grasp_state(self, object_id: str, arm: str) -> GraspState: ...
    def free_spot(self, surface: str, object_id: str, near: tuple[float, float, float] | None = None,
                  within=None) -> tuple[float, float, float] | None: ...
    def path(self, a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]] | None: ...
    def perception(self) -> dict: ...

    # UI + eval only
    def truth(self) -> TruthSnapshot: ...


@runtime_checkable
class SimControl(Protocol):
    def teleport_robot(self, x: float, y: float, yaw: float) -> None: ...
    def attach(self, object_id: str, arm: str, mode: str = "follow") -> None: ...
    def detach(self, object_id: str, pose: tuple[float, float, float] | None = None) -> None: ...
    def band(self, on: bool, ramp_s: float = 1.5) -> None: ...
    def reset_robot(self, x: float, y: float, yaw: float) -> None: ...
    def capabilities(self) -> dict[str, bool]: ...


@runtime_checkable
class Localizer(Protocol):
    def pose(self) -> tuple[RobotPose, float, str]: ...     # pose, age_s, source
    def velocity(self) -> tuple[float, float]: ...          # m/s, rad/s


class WorldLocalizer:
    """Localizer over a WorldModel's GT pose (api.services.Localizer): pose() -> (Pose2D, age_s, source)."""

    def __init__(self, world: Any):
        self.world = world

    def pose(self):
        import time
        from api.types import Pose2D
        p = self.world.robot_pose()
        age_fn = getattr(self.world, "pose_age_s", None)
        age = float(age_fn()) if callable(age_fn) else max(0.0, time.time() - p.t) if p.t else 0.0
        return Pose2D(p.x, p.y, p.yaw), age, p.source

    def robot_pose(self) -> RobotPose:
        return self.world.robot_pose()

    def velocity(self) -> tuple[float, float]:
        p = self.world.robot_pose()
        return p.speed, p.wz


def xyyaw(pose_or_x: Any, y: float | None = None, yaw: float | None = None) -> tuple[float, float, float]:
    """Accept a Pose2D-like object (x, y, yaw), a tuple, or three numbers."""
    if y is not None:
        return float(pose_or_x), float(y), float(yaw or 0.0)
    if hasattr(pose_or_x, "x"):
        return float(pose_or_x.x), float(pose_or_x.y), float(getattr(pose_or_x, "yaw", 0.0) or 0.0)
    t = tuple(pose_or_x)
    return float(t[0]), float(t[1]), float(t[2]) if len(t) > 2 else 0.0


def xyz(p: Any) -> tuple[float, float, float] | None:
    if p is None:
        return None
    if hasattr(p, "x"):
        return float(p.x), float(p.y), float(getattr(p, "z", 0.0))
    t = tuple(p)
    return float(t[0]), float(t[1]), float(t[2]) if len(t) > 2 else 0.0
