"""Typed service interfaces (PLAN 6). The runtime (agent/, brains/, llmkit/) talks ONLY
to a ``RobotBridge``; the world agent implements it (robot/bridge.py) on top of the
services below (services/, world/, robot/). Implementations may be richer; these
Protocols are the minimum both sides rely on.

Dispatch (``RobotBridge.start``). The harness creates an ``Execution`` (ids,
generation, epoch; args already alias-resolved by agent/validate.py) and calls
``robot.start(ex)``. ``start`` must not block; it returns an ``ExecutionHandle``
whose ``result()`` always resolves to a ``ToolResult`` built with
``api.results.finish``. It raises ``api.execution.Rejected("capability", code, msg)``
when a health check fails at dispatch time. Tools routed through ``start``:

  speak            args {text}. Resolves when the line has played (succeeded, data {utterance_id,
                   status: "queued"}) or was cut (cancelled, data {played}). Emits
                   speech_started / speech_ended / speech_cut on the sim EventLog (System 1 reads it).
  list_locations   args {query?}. data = asdict(ListLocationsResult); distances from the current pose.
  navigate         action "keypoint": args {location: <real keypoint>, timeout_s?}.
                   action "reposition": args {location: "reach_stance", anchor: <keypoint>,
                   stance: {x, y, yaw, ...}, reach_execution_id}. data = asdict(NavigateResult).
                   A keypoint navigate does NOT scan; the harness runs the arrival scan itself.
  check_reachability  args {object_type, object_id?, candidates: [ids], at: <keypoint|None>}.
                   Always succeeded when it ran; data = asdict(ReachabilityResult).
  manipulate       args {action, object_type, arm (resolved), object_id?, target?, goal?}.
                   data = asdict(ManipulationResult). Never moves the base on purpose.
  observe          args {mode: "glance"|"scan", why?}. data = RobotObservation.data()
                   (LookData fields + observation_id + mode). A glance has views=[].
  look             only with WL_LOOK_TOOL=1: same as observe(scan).

``wait_and_observe`` and ``recall`` run inside the harness (it starts an ``observe``).

Every ToolResult must carry ``observation_id`` (``glance_id()`` if nothing else ran).
Body executions honour cancel per PLAN 5.6; ``halt()`` returns within 30 ms.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .execution import Execution, ExecutionHandle
from .observation import LookData, RobotObservation, ViewSpec
from .results import NamedLocation, NavigateResult, ReachabilityResult, ManipulationResult
from .skills import SkillRegistry
from .types import (CameraFrame, Detection, GraspState, MapDict, ObjectState, Pose2D, Pose3D, RobotProfile,
                    ServiceHealth)

if TYPE_CHECKING:                                     # pragma: no cover
    pass

CAPABILITIES = ("navigation", "manipulation", "observation", "speech", "body")


# ----------------------------------------------------------------------
# The façade the runtime uses
# ----------------------------------------------------------------------
@runtime_checkable
class RobotBridge(Protocol):
    """robot/bridge.py: G1Robot (and LiteRobot). Duck-compatible with the old ThorRobot read surface."""

    profile: RobotProfile

    # ---- THOR-compatible read surface (agent/harness.py, fused_state.py) ----
    def lookup_keypoints(self) -> MapDict: ...
    def memory(self) -> list[dict[str, Any]]: ...               # [] (the runtime keeps its own memory)
    def base_state(self) -> dict[str, Any]: ...                 # {moving, at, between, xy:[x, z]}
    def gripper(self, arm: str) -> dict[str, Any]: ...          # {closed, width, force}
    def proprio(self, arm: str) -> dict[str, Any]: ...
    def telemetry(self) -> dict[str, Any]: ...
    # THOR shape {pose:{x,z,yaw,horizon,at,between}, velocity:{linear_mps, angular_dps}, moving,
    #             arms, grippers, active_skills:[{id, skill, status}], health:{ok, source}}
    # + body:{mode, lease, upright, rtf, carry, halt_epoch}
    def perception(self) -> dict[str, Any]: ...
    # THOR shape {objects:{id:{type,label,where,pose,distance_m,visible,confidence,source}},
    #             landmarks:{...}, people:{user:{...}}, source}; where="hand:<arm>" directly

    # ---- execution ----
    def start(self, execution: Execution) -> ExecutionHandle: ...   # may raise Rejected("capability", ...)
    def halt(self) -> dict[str, Any]: ...
    # {accepted, stopped (acked within 30 ms), at_rest, mode: "HOLD", body_epoch, source}
    def resume(self, control_epoch: int) -> None: ...          # clears the body's halt latch
    def estop(self, reason: str) -> dict[str, Any]: ...        # operator kill button only
    def active_executions(self) -> list[Execution]: ...
    async def shutdown(self) -> None: ...

    # ---- what validation and prompts need ----
    def capabilities(self) -> dict[str, ServiceHealth]: ...    # keys from CAPABILITIES
    def registry(self) -> SkillRegistry: ...                   # loaded at session start; frozen enum
    def timeout_s(self, tool: str, args: dict[str, Any]) -> float: ...
    # navigate: clamp(1.8*path_m/v + 12, 20, 240) from the current pose (aliases resolved first);
    # manipulate: skill.max_duration_s + 10; others: a service default
    def observation_id(self) -> str: ...                       # the latest glance record, "obs-g<rev>"
    def events(self) -> "asyncio.Queue[dict[str, Any]]": ...   # a new subscriber queue of robot events:
    # {"type": "safety_event", "kind": "fell"|"deploy_lost"|..., ...}, {"type": "capability_changed", ...},
    # {"type": "body_mode", "mode": ...}, {"type": "stale_result", "execution_id": ...}


# ----------------------------------------------------------------------
# Services behind the façade (PLAN 6.3-6.5)
# ----------------------------------------------------------------------
@runtime_checkable
class NavigationService(Protocol):
    async def list_locations(self, current_pose: Pose2D, query: str | None = None) -> list[NamedLocation]: ...
    async def navigate(self, location: str, *, execution: Execution,
                       timeout_s: float | None = None) -> ExecutionHandle: ...
    async def reposition(self, stance_w: Pose2D, *, anchor: str, execution: Execution) -> ExecutionHandle: ...
    def resolve(self, location: str) -> str: ...                # user / room aliases -> real keypoint
    async def status(self, execution_id: str) -> NavigateResult: ...
    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None: ...
    def timeout_s(self, from_pose: Pose2D, location: str) -> float: ...
    def at(self) -> tuple[str | None, list[str] | None]: ...    # keypoint within 0.30 m, or between [a, b]
    def health(self) -> ServiceHealth: ...


@runtime_checkable
class ManipulationService(Protocol):
    async def list_skills(self) -> SkillRegistry: ...
    async def check_reachability(self, object_type: str, object_id: str | None = None, *,
                                 candidates: list[str] = ..., at: str | None = None) -> ReachabilityResult: ...
    async def execute(self, action: Literal["pick", "place"], object_type: str, *, arm: str | None = None,
                      target: str | None = None, object_id: str | None = None,
                      execution: Execution) -> ExecutionHandle: ...
    async def status(self, execution_id: str) -> ManipulationResult: ...
    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None: ...
    def health(self, skill_id: str | None = None) -> ServiceHealth: ...


@runtime_checkable
class ObservationService(Protocol):
    async def observe(self, mode: Literal["glance", "scan"], *, at: str | None,
                      execution: Execution) -> RobotObservation: ...
    def perception(self) -> dict[str, Any]: ...
    def latest_frame(self, camera: str) -> CameraFrame | None: ...
    def glance_record(self) -> str: ...                         # "obs-g<rev>"


@runtime_checkable
class SpeechService(Protocol):
    async def speak(self, text: str, *, execution: Execution) -> ExecutionHandle: ...
    def cut_all(self) -> None: ...


@runtime_checkable
class NavExecutor(Protocol):
    name: Literal["sonic_walk", "kinematic_nav", "lite"]

    async def run(self, plan: Any, handle: ExecutionHandle) -> Any: ...
    async def cancel(self) -> None: ...


@runtime_checkable
class ManipExecutor(Protocol):
    backend: Literal["groot", "sonic_arm_script", "kinematic_attach", "lite"]

    async def run(self, job: Any, handle: ExecutionHandle) -> Any: ...
    async def cancel(self) -> None: ...


# ----------------------------------------------------------------------
# World (PLAN 6.2): the ONLY ground-truth readers on the runtime side live in world/
# ----------------------------------------------------------------------
@runtime_checkable
class WorldModel(Protocol):
    scene: str

    def static_map(self) -> Any: ...
    def lookup_keypoints(self) -> MapDict: ...
    def robot_pose(self) -> Pose3D: ...
    def detections(self, camera: str = "head") -> list[Detection]: ...
    def scan(self, views: list[ViewSpec], *, at: str | None) -> LookData: ...
    def object(self, object_id: str) -> ObjectState | None: ...
    def hands(self) -> dict[str, str | None]: ...
    def grasp_state(self, object_id: str, arm: str) -> GraspState: ...
    def free_spot(self, surface: str, object_id: str, near: Pose2D) -> Pose3D | None: ...
    def path(self, a: Pose2D, b: Pose2D) -> list[tuple[float, float]] | None: ...
    def latest_frame(self, camera: Literal["head", "ego", "top"]) -> CameraFrame | None: ...
    def truth(self) -> Any: ...                                 # UI + eval ONLY; never the runtime


class NotSupported(Exception):
    """SimControl on a real robot."""


@runtime_checkable
class SimControl(Protocol):
    def teleport_robot(self, pose: Pose2D) -> None: ...
    def attach(self, object_id: str, arm: str, mode: Literal["fixed_joint", "follow"]) -> None: ...
    def detach(self, object_id: str, pose: Pose3D | None) -> None: ...
    def band(self, on: bool, ramp_s: float = 1.5) -> None: ...
    def reset_robot(self, pose: Pose2D) -> None: ...
    def reset_scene(self, variant: str) -> None: ...
    def set_render_rates(self, head_hz: float, ego_hz: float) -> None: ...


@runtime_checkable
class Localizer(Protocol):
    def pose(self) -> tuple[Pose2D, float, str]: ...           # pose, age_s, source
    def velocity(self) -> tuple[float, float]: ...             # m/s, rad/s


@runtime_checkable
class FrameSource(Protocol):
    def latest(self, camera: str = "head") -> CameraFrame | None: ...


__all__ = ["CAPABILITIES", "RobotBridge", "NavigationService", "ManipulationService", "ObservationService",
           "SpeechService", "NavExecutor", "ManipExecutor", "WorldModel", "NotSupported", "SimControl",
           "Localizer", "FrameSource"]
