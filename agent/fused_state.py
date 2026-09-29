"""Fused runtime state and the adapters that feed it.

The runtime keeps one latest telemetry/perception frame and a bounded history
of semantic events.  Sampling a robot at 10 Hz therefore does not create ten
complete world snapshots per second and does not wake the planner unless
something relevant actually changed.

The adapter intentionally talks only to the public robot/sensor surface.  The
THOR robot implements ``telemetry`` and ``perception`` using simulator state;
a ROS-backed robot can implement the same two methods with real sensor and
perception messages.
"""

from __future__ import annotations

import collections
import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol


OBSERVATION_PERIOD_S = 0.1
EVENT_HISTORY_LIMIT = 512


@dataclass(frozen=True)
class Directive:
    """One structured System-1 instruction delivered to the runtime."""

    id: str
    t: float
    kind: str
    text: str
    source: str = "runtime-classifier"
    target: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    supersedes: str | None = None
    reply: str | None = None      # yes / no, when it answers the robot's yes/no question

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationFrame:
    """A timestamped sensor/perception sample from an observation adapter."""

    seq: int
    t: float
    source: str
    robot: dict[str, Any]
    perception: dict[str, Any]


@dataclass(frozen=True)
class RuntimeEvent:
    """A small semantic delta retained for planners, traces, and debugging."""

    seq: int
    t: float
    type: str
    priority: int
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationSource(Protocol):
    def sample(self, now: float) -> ObservationFrame: ...


class RobotObservationAdapter:
    """Turn a robot's telemetry/perception API into observation frames.

    ``telemetry`` and ``perception`` are optional to keep scripted test robots
    and the baseline compatible.  The fallbacks use the older robot API.
    """

    def __init__(self, robot: Any, source: str = "robot-adapter") -> None:
        self.robot = robot
        self.source = source
        self._seq = itertools.count(1)

    def sample(self, now: float) -> ObservationFrame:
        telemetry = getattr(self.robot, "telemetry", None)
        if callable(telemetry):
            robot_state = dict(telemetry())
        else:
            base = dict(self.robot.base_state())
            arms: dict[str, Any] = {}
            grippers: dict[str, Any] = {}
            for arm in ("left", "right"):
                proprio = getattr(self.robot, "proprio", None)
                arms[arm] = dict(proprio(arm)) if callable(proprio) else {}
                grippers[arm] = dict(self.robot.gripper(arm))
            robot_state = {
                "pose": {"at": base.get("at"), "between": base.get("between"),
                         "xy": base.get("xy")},
                "velocity": {"linear_mps": None, "angular_dps": None},
                "moving": bool(base.get("moving")),
                "arms": arms,
                "grippers": grippers,
                "active_skills": [],
                "health": {"ok": True},
            }

        perception_fn = getattr(self.robot, "perception", None)
        perception = dict(perception_fn()) if callable(perception_fn) else {"objects": {}, "people": {}}
        robot_state["t"] = round(now, 3)
        perception["observed_at"] = round(now, 3)
        return ObservationFrame(next(self._seq), round(now, 3), self.source, robot_state, perception)


def _stable(value: Any) -> Any:
    """Remove volatile timestamps before comparing two fused samples."""

    if isinstance(value, Mapping):
        return {k: _stable(v) for k, v in value.items() if k not in {"t", "observed_at"}}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, set):
        return sorted(_stable(v) for v in value)
    return value


def _fingerprint(value: Any) -> str:
    return json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), default=str)


class FusedRuntimeState:
    """Latest robot/world state plus bounded semantic event history."""

    def __init__(self, belief: Any, task: Any, event_limit: int = EVENT_HISTORY_LIMIT) -> None:
        self.belief = belief
        self.task = task
        self.robot: dict[str, Any] = {}
        self.perception: dict[str, Any] = {"objects": {}, "people": {}}
        self.revision = 0
        self.last_observation_seq = 0
        self.events: collections.deque[RuntimeEvent] = collections.deque(maxlen=event_limit)
        self._event_ids = itertools.count(1)
        self._robot_fp = ""
        self._world_fp = ""

    def record(self, type_: str, t: float, *, priority: int = 4, **data: Any) -> RuntimeEvent:
        event = RuntimeEvent(next(self._event_ids), round(t, 3), type_, priority, dict(data))
        self.events.append(event)
        return event

    def ingest(self, frame: ObservationFrame) -> set[str]:
        """Replace latest state and return the semantic components that changed."""

        robot_fp = _fingerprint(frame.robot)
        world_fp = _fingerprint(frame.perception)
        first = not self.robot
        changed: set[str] = set()
        if robot_fp != self._robot_fp:
            changed.add("robot")
        if world_fp != self._world_fp:
            changed.add("world")

        self.robot = dict(frame.robot)
        self.perception = dict(frame.perception)
        self.last_observation_seq = frame.seq
        self._robot_fp, self._world_fp = robot_fp, world_fp

        apply_perception = getattr(self.belief, "apply_perception", None)
        if callable(apply_perception):
            apply_perception(frame.perception, frame.robot, frame.t, frame.source)

        if changed:
            self.revision += 1
            if first:
                self.record("observation_initialized", frame.t, priority=3,
                            source=frame.source, components=sorted(changed))
            else:
                self.record("observation_changed", frame.t, priority=3,
                            source=frame.source, components=sorted(changed))
        return changed

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return [event.to_dict() for event in list(self.events)[-limit:]]

    def task_dict(self) -> dict[str, Any]:
        current = getattr(self.task, "current_directive", None)
        return {
            "intent_version": getattr(self.task, "intent_version", 0),
            "control_epoch": getattr(self.task, "control_epoch", 0),
            "paused": bool(getattr(self.task, "paused", False)),
            "goal": getattr(self.task, "goal", None),
            "current_directive": current.to_dict() if hasattr(current, "to_dict") else current,
        }

    def snapshot(self, *, active: list[dict[str, Any]] | None = None,
                 recent_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "observation_seq": self.last_observation_seq,
            "robot": self.robot,
            "world": self.perception,
            "task": self.task_dict(),
            "active_behaviors": active or [],
            "recent_actions": recent_actions or [],
            "recent_events": self.recent(),
        }
