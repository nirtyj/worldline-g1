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
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

from .execution import Execution, ExecutionHandle
from .observation import LookData, RobotObservation, ViewSpec
from .results import NamedLocation, NavigateResult, ReachabilityResult, ManipulationResult
from .skills import SkillRegistry, SkillSpec
from .types import (CameraFrame, Detection, GraspState, MapDict, ObjectState, Pose2D, Pose3D, RobotProfile,
                    ServiceHealth)

if TYPE_CHECKING:                                     # pragma: no cover
    pass

# A pose argument that SimControl and the WorldModel accept: an api Pose2D (x, y, yaw), anything with .x/.y/.yaw,
# or an (x, y, yaw) tuple. Implementations normalise it with world.model.xyyaw.
PoseLike = Any

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
# Services behind the façade (PLAN 6.3-6.5). Implemented in services/ and wired by robot/bridge.py.
# Each service also has a non-blocking ``start(...)`` that the bridge dispatches to; the async
# ``navigate``/``execute``/``speak`` forms are the PLAN 6 names and just call ``start``.
# tests/contract/test_protocols.py checks every implementation against these Protocols (names and
# parameter names), so a drift fails a test instead of a live run.
# ----------------------------------------------------------------------
@runtime_checkable
class NavigationService(Protocol):
    async def list_locations(self, current_pose: PoseLike = None, query: str | None = None
                             ) -> list[NamedLocation]: ...
    async def navigate(self, location: str, *, execution: Execution,
                       timeout_s: float | None = None) -> ExecutionHandle: ...
    async def reposition(self, stance_w: Pose2D, *, anchor: str, execution: Execution) -> ExecutionHandle: ...
    def start(self, execution: Execution) -> ExecutionHandle: ...   # keypoint or reposition, by args
    def resolve(self, location: str) -> str: ...                # user / room aliases -> real keypoint
    async def status(self, execution_id: str) -> NavigateResult | None: ...
    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None: ...
    def timeout_s(self, from_pose: PoseLike = None, location: str = "") -> float: ...
    def at(self) -> tuple[str | None, list[str] | None]: ...    # keypoint within 0.30 m, or between [a, b]
    def health(self) -> ServiceHealth: ...


@runtime_checkable
class ManipulationService(Protocol):
    async def list_skills(self) -> SkillRegistry: ...
    def capability(self, action: str, object_type: str, arm: str | None = None) -> tuple[SkillSpec | None, str]: ...
    async def check_reachability(self, object_type: str, object_id: str | None = None, *,
                                 candidates: Any = (), at: str | None = None) -> ReachabilityResult: ...
    def start_reachability(self, execution: Execution) -> ExecutionHandle: ...
    async def execute(self, action: Literal["pick", "place"], object_type: str, *, arm: str | None = None,
                      target: str | None = None, object_id: str | None = None,
                      execution: Execution) -> ExecutionHandle: ...
    def start(self, execution: Execution) -> ExecutionHandle: ...
    async def status(self, execution_id: str) -> ManipulationResult | None: ...
    async def cancel(self, execution_id: str, reason: str = "cancelled") -> None: ...
    def health(self, skill_id: str | None = None) -> ServiceHealth: ...


@runtime_checkable
class ObservationService(Protocol):
    async def observe(self, mode: Literal["glance", "scan"], *, at: str | None = None,
                      execution: Execution | None = None) -> RobotObservation: ...
    def start(self, execution: Execution) -> ExecutionHandle: ...   # observe(glance|scan) as an execution
    def start_wait(self, execution: Execution, *, wake: asyncio.Event | None = None,
                   lease_held: bool = False) -> ExecutionHandle: ...  # wait_and_observe (service fallback)
    def perception(self) -> dict[str, Any]: ...
    def latest_frame(self, camera: str = "head") -> CameraFrame | None: ...
    def glance_record(self) -> str: ...                         # "obs-g<rev>"


@runtime_checkable
class SpeechService(Protocol):
    async def speak(self, text: str, *, execution: Execution) -> ExecutionHandle: ...
    def start(self, text: str, execution: Execution) -> ExecutionHandle: ...
    def cut_all(self) -> None: ...


# ----------------------------------------------------------------------
# Executors behind the services (PLAN 6.3, 6.4). Navigation's executor is the body itself
# (robot/body_client.SonicBody = sonic_walk over wl-body, robot/lite_body.LiteBody = lite): a
# ``BodyPort``. Manipulation's executors run one ManipJob (services/executors/).
# ----------------------------------------------------------------------
@runtime_checkable
class BodyOpHandle(Protocol):
    """An in-flight body motion (robot/body_client.BodyOp). ``result()`` always resolves to
    {"state": "succeeded"|"failed"|"canceled", "data": {...}} (docs/contracts/m1.md 3.4)."""
    op: str
    id: str

    @property
    def done(self) -> bool: ...
    async def result(self) -> dict[str, Any]: ...
    def cancel(self, reason: str = "cancelled") -> None: ...   # planner IDLE; never command{stop}


@runtime_checkable
class BodyPort(Protocol):
    """What services call to move the robot. ``halt`` latches HOLD (never command{stop}); ``estop``
    is the operator kill button only (the deploy exits; P2 restart)."""
    name: str                                     # executor name: "sonic_walk" | "kinematic_nav" | "lite"

    async def go_to(self, x: float, y: float, yaw: float | None = None, *, speed: float | None = None,
                    timeout_s: float | None = None, final_pos_tol: float | None = None,
                    final_yaw_tol_deg: float | None = None) -> BodyOpHandle: ...
    async def turn_to(self, yaw: float, *, tol_deg: float | None = None) -> BodyOpHandle: ...
    async def stop(self) -> BodyOpHandle: ...
    def halt(self, epoch: int | None = None) -> dict[str, Any]: ...
    def resume(self, epoch: int | None = None) -> None: ...
    def estop(self, reason: str) -> dict[str, Any]: ...
    def state(self) -> dict[str, Any]: ...        # {mode, active, pose, halt_epoch, latched, gt_pose{rtf}, ...}
    def health(self) -> ServiceHealth: ...


NavExecutor = BodyPort                            # PLAN 6.3's name for navigation's executor


@runtime_checkable
class ManipExecutor(Protocol):
    backend: str                                  # "groot" | "sonic_arm_script" | "kinematic_attach" | "lite"
    name: str                                     # the executor name results carry (api.skills.EXECUTOR_OF_BACKEND)

    async def run(self, job: Any, handle: Any) -> Any: ...     # ManipJob -> ManipOutcome (services/executors)
    async def cancel(self) -> None: ...
    def health(self) -> ServiceHealth: ...


# ----------------------------------------------------------------------
# World (PLAN 6.2): the ONLY ground-truth readers on the runtime side live in world/
# (world/gt_world.GTWorld -> LiteWorld, IsaacGTWorldModel). Geometry is REP-103 (x, y, yaw rad ccw);
# dicts that reach agent/ (look, perception, truth) are already in Worldline's frame (world/coords.py).
# ----------------------------------------------------------------------
@runtime_checkable
class RobotPoseLike(Protocol):
    """``WorldModel.robot_pose()``: an upright pose (world.model.RobotPose). Duck-compatible with
    ``Pose3D`` (qw..qz from yaw) and carries what services need beyond it."""
    x: float
    y: float
    z: float                                      # pelvis, world
    yaw: float                                    # rad, ccw from +x
    t: float                                      # sample time
    source: str                                   # "isaac-gt" | "lite" | a localizer name
    fallen: bool
    pelvis_z: float                               # above the floor
    wz: float                                     # yaw rate, rad/s

    @property
    def speed(self) -> float: ...                 # planar m/s


@runtime_checkable
class WorldModel(Protocol):
    scene: str
    source: str                                   # "isaac-gt" | "lite-gt" | "perception"

    # the static semantic map (doc 27)
    def static_map(self) -> Any: ...              # world.mapgen.StaticMap (keypoints, surfaces, grid, ...)
    def lookup_keypoints(self) -> MapDict: ...
    def keypoint_at(self, x: float, y: float, tol: float = 0.30) -> str | None: ...
    def path(self, a: PoseLike, b: PoseLike) -> list[tuple[float, float]] | None: ...

    # the robot and its head camera
    def robot_pose(self) -> RobotPoseLike: ...
    def camera_pose(self, **view: Any) -> Any: ...               # world.perception.CameraPose
    def view_spec(self, cam_pose: Any) -> dict[str, Any]: ...    # an api.observation.ViewSpec as a dict

    # what the robot can perceive (GT-backed hints in sim; a perception stack later, PLAN 13)
    def detections(self, camera: str = "head", **view: Any) -> list[Any]: ...   # world.model.Detection
    def look(self, views: list[dict] | None, *, at: str | None, detections: Any = None) -> dict[str, Any]: ...
    def scan(self, views: list[ViewSpec], *, at: str | None) -> LookData: ...
    def perception(self) -> dict[str, Any]: ...
    def latest_frame(self, camera: str = "head") -> CameraFrame | None: ...

    # objects and hands
    def object(self, object_id: str) -> Any: ...                # world.model.ObjectState (duck: api ObjectState)
    def objects(self) -> dict[str, Any]: ...
    def hands(self) -> dict[str, str | None]: ...
    def grasp_state(self, object_id: str, arm: str) -> Any: ...
    def free_spot(self, surface: str, object_id: str, near: PoseLike = None,
                  within: Callable[[float, float, float], bool] | None = None) -> Pose3D | None: ...
    # `within(x, y, z)`: the reach test from the current pose; lets place tell no_room_in_reach
    # (room exists, not reachable from here) from no_room_on_surface.

    def truth(self) -> Any: ...                                 # UI + eval ONLY; never the runtime


class NotSupported(RuntimeError):
    """This backend cannot do that: SimControl on a real robot, or a P1 build without the op.
    world.model.NotSupported is this class."""


@runtime_checkable
class SimControl(Protocol):
    """Actuation only a simulator can do. Raises NotSupported (world.model.NotSupported) where the
    backend lacks the op; ``capabilities()`` says which ops exist, e.g. the skill registry marks
    kinematic_attach healthy only when {"attach", "detach"} are available."""
    def capabilities(self) -> dict[str, bool]: ...              # {attach, detach, object_poses, band, teleport, ...}
    def teleport_robot(self, pose: PoseLike, y: float | None = None, yaw: float | None = None) -> None: ...
    def reset_robot(self, pose: PoseLike, y: float | None = None, yaw: float | None = None) -> None: ...
    def attach(self, object_id: str, arm: str, mode: str = "follow") -> None: ...   # "follow" | "fixed_joint"
    def detach(self, object_id: str, pose: Any = None) -> None: ...
    def band(self, on: bool, ramp_s: float = 1.5) -> None: ...
    def reset_scene(self, variant: str) -> None: ...
    def set_render_rates(self, head_hz: float, ego_hz: float) -> None: ...


@runtime_checkable
class Localizer(Protocol):
    def pose(self) -> tuple[Pose2D, float, str]: ...           # pose, age_s, source
    def velocity(self) -> tuple[float, float]: ...             # m/s, rad/s


@runtime_checkable
class FrameSource(Protocol):
    def latest(self, camera: str = "head") -> CameraFrame | None: ...
    def latest_frame(self, camera: str = "head") -> CameraFrame | None: ...


__all__ = ["CAPABILITIES", "PoseLike", "RobotBridge", "NavigationService", "ManipulationService",
           "ObservationService", "SpeechService", "BodyOpHandle", "BodyPort", "NavExecutor", "ManipExecutor",
           "RobotPoseLike", "WorldModel", "NotSupported", "SimControl", "Localizer", "FrameSource"]
