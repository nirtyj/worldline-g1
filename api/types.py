"""Geometry and common value types shared by every process (PLAN 6.1).

Stdlib only. Services use REP-103 geometry: x, y in metres, yaw in radians
counter-clockwise from +x, Z up. The agent, layout and UI keep Worldline's
frame (map x, map z, yaw in degrees clockwise from +z); ``world/coords.py`` is
the only conversion point between the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

ARMS: tuple[str, str] = ("left", "right")
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


@dataclass(frozen=True)
class ServiceHealth:
    ok: bool
    state: str                     # "ok", "degraded", "unsafe", "down", "fault", "estop", ...
    detail: str = ""
    t: float = 0.0


@dataclass(frozen=True)
class CameraFrame:
    rev: int                       # render counter of the producing camera
    t_wall: float
    jpeg: bytes
    cam_pose_wl: tuple[float, float, float, float]   # (x, z, yaw_cw_deg, pitch_down_deg), Worldline frame
    stationary: bool = False
    camera: Literal["head", "ego", "top"] = "head"
    hfov: float | None = None
    vfov: float | None = None
    w: int = 0
    h: int = 0


@dataclass(frozen=True)
class Detection:
    """One object or landmark visible now on a camera (GT-backed in sim)."""
    id: str
    type: str
    label: str
    where: str | None              # surface, "hand:<arm>", "floor", container or None
    pose: Pose3D | None = None
    px: int = 0                    # instance-id pixels
    dist: float | None = None


@dataclass(frozen=True)
class ObjectState:
    id: str
    type: str
    pose: Pose3D | None
    where: str | None
    held_by: str | None = None     # "left" | "right" | None


@dataclass(frozen=True)
class GraspState:
    object_id: str
    arm: str
    lifted_m: float = 0.0          # above its support
    palm_dist_m: float | None = None
    contacts: int = 0
    held: bool = False


@dataclass(frozen=True)
class RobotProfile:
    """The numeric slots that may differ per profile (PLAN 1.3 #30, deviation D2).

    Everything textual in the tool descriptions and the SYSTEM prompt is one
    template; only these numbers are filled in. ``test_parity`` masks digits and
    compares the text across profiles.
    """
    name: Literal["lite", "bringup", "sonic", "full", "real_g1"] = "lite"
    robot: str = "unitree_g1"
    walk_speed_mps: float = 0.40
    t_pick_s: float = 12.0
    t_place_s: float = 8.0
    approach_max_m: float = 0.40
    reach_h_min_m: float = 0.55
    reach_h_max_m: float = 1.20
    wait_max_s: float = 60.0
    cancel_grace_body_s: float = 3.0
    cancel_grace_other_s: float = 1.5
    reach_fresh_s: float = 30.0
    time_scale: float = 1.0                     # eval timeout multiplier (PLAN 9.1)
    stepping_stones: tuple[str, ...] = ()       # executor names labelled STEPPING STONE in this profile

    @property
    def s_per_m(self) -> float:
        return 1.0 / self.walk_speed_mps if self.walk_speed_mps > 0 else 0.0

    def slots(self) -> dict[str, float]:
        """Values for ``str.format`` in tool descriptions and the SYSTEM template."""
        return {
            "v": self.walk_speed_mps, "s_per_m": self.s_per_m, "t_pick": self.t_pick_s,
            "t_place": self.t_place_s, "approach_max_m": self.approach_max_m,
            "h_min": self.reach_h_min_m, "h_max": self.reach_h_max_m, "wait_max": self.wait_max_s,
        }


# Per-profile numbers. The Isaac profiles carry R.7's live measurements (world-cal, 2026-09-29, docs/calibration.md
# §7, SONIC on H40): a navigate averages 0.291 m/s = 3.44 s/m (17 go_to legs; turns, stops and the final approach
# included), the arm script's pick phases take 16.5-18.0 s and its place phases 8.8-10.1 s. full's t_pick is an
# ESTIMATE, not a measurement: GR00T's zero-shot attempt (about 12.6 s to its timeout in wave 1) plus the 17 s
# scripted fallback. lite keeps M2a's numbers (its body is pure Python).
# time_scale multiplies every eval scenario timeout (PLAN 9.1): lite 1.0, bringup 1.3, sonic 2.0, full 2.5.
PROFILES: dict[str, RobotProfile] = {
    "lite": RobotProfile(name="lite", walk_speed_mps=0.45, t_pick_s=6.0, t_place_s=5.0, time_scale=1.0),
    "bringup": RobotProfile(name="bringup", walk_speed_mps=0.29, t_pick_s=8.0, t_place_s=6.0, time_scale=1.3,
                            stepping_stones=("kinematic_nav", "kinematic_attach")),
    "sonic": RobotProfile(name="sonic", walk_speed_mps=0.29, t_pick_s=17.0, t_place_s=10.0, time_scale=2.0,
                          stepping_stones=("sonic_arm_script",)),
    "full": RobotProfile(name="full", walk_speed_mps=0.29, t_pick_s=30.0, t_place_s=10.0, time_scale=2.5,
                         stepping_stones=("sonic_arm_script",)),
    "real_g1": RobotProfile(name="real_g1", walk_speed_mps=0.40, t_pick_s=15.0, t_place_s=10.0, time_scale=2.5),
}


# ----------------------------------------------------------------------
# The map the runtime consumes: RobotBridge.lookup_keypoints()
# (THOR-shaped keys unchanged, plus the G1 additions in PLAN 6.2.1)
# ----------------------------------------------------------------------
class KeypointInfo(TypedDict, total=False):
    desc: str
    xy: list[float]                 # [map_x, map_z], Worldline frame
    yaw: float                      # facing, degrees clockwise from +z (Worldline frame)
    room: str
    kind: str                       # "surface" | "room" | "start" | "person"


class SurfaceInfo(TypedDict, total=False):
    keypoints: list[str]            # stands that serve this surface
    desc: str
    height_m: float
    xy: list[float]


class RoomInfo(TypedDict, total=False):
    label: str
    spots: list[str]                # keypoints in the room
    keypoint: str                   # the room's own keypoint (PLAN 5.2 alias target)


class MapDict(TypedDict, total=False):
    scene: str                      # "procthor-train-40@a" style scene key
    keypoints: dict[str, KeypointInfo]
    edges: list[list[Any]]          # [[a, b, metres], ...]
    surfaces: dict[str, SurfaceInfo]
    people: dict[str, dict[str, Any]]   # {"user": {"deliver_to_surface", "keypoint"}}
    rooms: dict[str, RoomInfo]
    nav_speed_mps: float
    max_reach_height_m: float
    min_reach_height_m: float
    robot: str                      # "unitree_g1"
    profile: str
    executors: dict[str, str]       # {"navigate": "sonic_walk", "manipulate": "sonic_arm_script", ...}


def as_dict(obj: Any) -> Any:
    """``dataclasses.asdict`` that also accepts plain values (for data payloads)."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


__all__ = ["ARMS", "UNKNOWN", "Pose2D", "Pose3D", "ServiceHealth", "CameraFrame", "Detection",
           "ObjectState", "GraspState", "RobotProfile", "PROFILES", "KeypointInfo", "SurfaceInfo",
           "RoomInfo", "MapDict", "as_dict"]
