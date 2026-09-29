"""Execution objects, ids, generations and epochs (PLAN 5.4, 5.5). Replaces ``sim/goals.py``.

Every tool call the harness accepts (and every rejected one) becomes an
``Execution``. Services receive it, run it, and hand back an
``ExecutionHandle`` whose ``result()`` always resolves to a frozen
``api.results.ToolResult``.

    ex = executions.create("navigate", {"location": "kitchen_counter_1a"}, generation=g,
                           control_epoch=e, source="brain")
    handle = robot.start(ex)             # may raise Rejected (capability stage)
    handle.cancel("correction")          # a request: idempotent, never blocks
    result = await handle.result()       # always resolves

Invariants (tests/unit/test_execution.py):
  I1 result() always resolves (cancel, halt, service timeout, crash -> failed/internal_error)
  I2 cancel before start -> cancelled, reason "cancelled before start"
  I3 terminal statuses are immutable; ToolResult is frozen; Execution.data is the mutable copy
  I4 generation and control_epoch never change after creation
  I5 a result finishing with an older generation is late (set by the harness in _finish)
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:                                   # pragma: no cover
    from .results import ToolResult

ExecStatus = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled",
                     "timed_out", "rejected", "dropped"]
ACTIVE_STATUSES: tuple[str, ...] = ("queued", "running", "cancelling")
TERMINAL_STATUSES: tuple[str, ...] = ("succeeded", "failed", "cancelled", "timed_out", "rejected", "dropped")
Source = Literal["brain", "harness", "persona"]

PREFIX: dict[str, str] = {"speak": "spk", "list_locations": "loc", "navigate": "nav",
                          "check_reachability": "rch", "manipulate": "man", "wait_and_observe": "wai",
                          "recall": "rec", "observe": "obs", "look": "obs"}


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


@dataclass(eq=False)
class Execution:
    """One tool call's lifecycle. Replaces Worldline's HistoryEntry.

    ``status``, ``data``, ``executor``, ``t_started``/``t_ended``, ``cancel_reason`` and
    ``result`` are real mutable fields (the harness and SpeechQueue assign them).
    ``generation`` and ``control_epoch`` must not change after creation (I4)."""
    execution_id: str
    tool_name: str
    args: dict[str, Any]                       # AFTER alias resolution (PLAN 5.2)
    generation: int                            # intent_version sampled BEFORE the brain call
    control_epoch: int
    action: str | None = None                  # manipulate: pick|place; navigate: keypoint|reposition; observe: glance|scan
    source: Source = "brain"
    tag: str | None = None
    resources: frozenset[str] = frozenset()
    status: ExecStatus = "queued"
    data: dict[str, Any] = field(default_factory=dict)
    executor: str | None = None                # sonic_walk | kinematic_nav | groot_arms | sonic_arm_script | ...
    t_created: float = 0.0
    t_started: float | None = None
    t_ended: float | None = None
    cancel_reason: str | None = None
    result: "ToolResult | None" = None         # the frozen envelope, once terminal

    # ---- read-only aliases for existing agent/ code ----
    @property
    def id(self) -> str:
        return self.execution_id

    @property
    def created_for(self) -> int:
        return self.generation

    @property
    def tool(self) -> str:
        return self.tool_name

    @property
    def finished(self) -> bool:
        return self.status not in ACTIVE_STATUSES

    @property
    def t_start(self) -> float:
        return self.t_started if self.t_started is not None else self.t_created

    @t_start.setter
    def t_start(self, v: float) -> None:
        self.t_started = v

    @property
    def t_end(self) -> float | None:
        return self.t_ended

    @t_end.setter
    def t_end(self, v: float | None) -> None:
        self.t_ended = v

    @property
    def late(self) -> bool:
        return bool(self.result.late) if self.result is not None else bool(self.data.get("late"))

    def to_dict(self) -> dict[str, Any]:
        return {"execution_id": self.execution_id, "tool": self.tool_name, "args": dict(self.args),
                "action": self.action, "generation": self.generation, "control_epoch": self.control_epoch,
                "source": self.source, "tag": self.tag, "resources": sorted(self.resources),
                "status": self.status, "executor": self.executor, "t_created": self.t_created,
                "t_start": self.t_start, "t_end": self.t_ended, "cancel_reason": self.cancel_reason,
                "late": self.late, "data": dict(self.data or {})}


class Rejected(Exception):
    """A call refused before it runs (replaces the THOR-era GoalRejected).

    ``stage`` is one of schema | enum | state | capability (PLAN 5.7)."""

    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message

    @property
    def reason(self) -> str:                    # GoalRejected compatibility
        return self.message

    def __repr__(self) -> str:
        return f"Rejected({self.stage!r}, {self.code!r}, {self.message!r})"


@runtime_checkable
class ExecutionHandle(Protocol):
    execution_id: str

    def status(self) -> ExecStatus: ...
    def cancel(self, reason: str = "cancelled") -> None: ...    # a request: idempotent, never blocks
    async def result(self) -> "ToolResult": ...                # ALWAYS resolves (shielded)

    @property
    def done(self) -> bool: ...

    @property
    def cancel_requested(self) -> bool: ...


class ResultHandle:
    """A ready-made ExecutionHandle for services and fakes (asyncio, stdlib only).

    The service calls ``resolve(result)`` exactly once; later calls are ignored (I3).
    ``cancel()`` records the request, marks the execution ``cancelling`` and calls
    ``on_cancel(reason)``; the service decides when the work actually stops."""

    def __init__(self, execution: Execution, on_cancel: Callable[[str], Any] | None = None) -> None:
        self.execution = execution
        self.execution_id = execution.execution_id
        self._on_cancel = on_cancel
        self._cancel_reason: str | None = None
        self._future: asyncio.Future | None = None
        self._cancel_evt: asyncio.Event | None = None
        self._resolved: "ToolResult | None" = None

    # lazily bound to the running loop, so a handle can be built outside one
    def _fut(self) -> asyncio.Future:
        if self._future is None:
            self._future = asyncio.get_running_loop().create_future()
            if self._resolved is not None:
                self._future.set_result(self._resolved)
        return self._future

    def _evt(self) -> asyncio.Event:
        if self._cancel_evt is None:
            self._cancel_evt = asyncio.Event()
            if self._cancel_reason is not None:
                self._cancel_evt.set()
        return self._cancel_evt

    def status(self) -> ExecStatus:
        return self.execution.status

    @property
    def done(self) -> bool:
        return self._resolved is not None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_reason is not None

    @property
    def cancel_reason(self) -> str | None:
        return self._cancel_reason

    def cancel(self, reason: str = "cancelled") -> None:
        if self.done or self._cancel_reason is not None:
            return
        self._cancel_reason = reason
        self.execution.cancel_reason = reason
        if self.execution.status in ("queued", "running"):
            self.execution.status = "cancelling"
        if self._cancel_evt is not None:
            self._cancel_evt.set()
        if self._on_cancel is not None:
            self._on_cancel(reason)

    async def wait_cancel(self) -> str:
        """For executors: returns the reason once cancel() is called."""
        await self._evt().wait()
        return self._cancel_reason or "cancelled"

    def resolve(self, result: "ToolResult") -> bool:
        """Deliver the terminal result. Returns False (and changes nothing) if already resolved."""
        if self._resolved is not None:
            return False
        self._resolved = result
        ex = self.execution
        ex.status = result.status
        ex.t_ended = result.t_end
        ex.result = result
        if self._future is not None and not self._future.done():
            self._future.set_result(result)
        return True

    async def result(self) -> "ToolResult":
        if self._resolved is not None:
            return self._resolved
        return await asyncio.shield(self._fut())

    def __repr__(self) -> str:
        return f"<ResultHandle {self.execution_id} {self.execution.status}>"


class ExecutionIds:
    """Session-unique ids: f"{PREFIX[tool]}-{seq:06d}" with one shared sequence."""

    def __init__(self, start: int = 1) -> None:
        self._seq = itertools.count(start)

    def next(self, tool: str) -> str:
        return f"{PREFIX.get(tool, 'exe')}-{next(self._seq):06d}"


class ExecutionManager:
    """Creates executions (stamping id, generation and epoch) and keeps them in order."""

    def __init__(self, clock: Any = None) -> None:
        self.ids = ExecutionIds()
        self.clock = clock
        self.all: list[Execution] = []
        self.by_id: dict[str, Execution] = {}

    def create(self, tool: str, args: dict[str, Any], *, generation: int, control_epoch: int,
               source: Source = "brain", tag: str | None = None, action: str | None = None,
               resources: frozenset[str] | None = None, status: ExecStatus = "queued",
               t: float | None = None) -> Execution:
        from .tools import resources_of
        now = t if t is not None else (self.clock.now() if self.clock is not None else 0.0)
        if action is None:
            action = default_action(tool, args)
        ex = Execution(execution_id=self.ids.next(tool), tool_name=tool, args=dict(args),
                       generation=generation, control_epoch=control_epoch, action=action, source=source,
                       tag=tag, resources=resources if resources is not None else resources_of(tool, args),
                       status=status, t_created=round(now, 3))
        self.all.append(ex)
        self.by_id[ex.execution_id] = ex
        return ex

    def get(self, execution_id: str) -> Execution | None:
        return self.by_id.get(execution_id)

    def active(self) -> list[Execution]:
        return [e for e in self.all if not e.finished]


def default_action(tool: str, args: dict[str, Any]) -> str | None:
    if tool == "manipulate":
        return str(args.get("action")) if args.get("action") else None
    if tool == "navigate":
        return "reposition" if args.get("location") == "reach_stance" or args.get("stance") else "keypoint"
    if tool == "observe":
        return str(args.get("mode") or "glance")
    return None


# ----------------------------------------------------------------------
# Generations and epochs (PLAN 5.5; doc 24, kept exactly as Worldline does it)
# ----------------------------------------------------------------------
Trigger = Literal["request_idle", "request_busy", "correction", "stop", "resume", "constraint",
                  "own_goal_dropped"]


@dataclass(frozen=True)
class GenerationRule:
    bump_generation: bool
    bump_epoch: bool
    cancels: str          # what the harness cancels; "" for nothing


GENERATION_RULES: dict[str, GenerationRule] = {
    "request_idle": GenerationRule(True, False, ""),
    "request_busy": GenerationRule(False, False, ""),
    "correction": GenerationRule(True, True, "older_generation"),   # + speech.drop_older_than; hands -> UNKNOWN
    "stop": GenerationRule(False, True, "body"),                    # halt() FIRST; speech.cut_all()
    "resume": GenerationRule(False, True, ""),                      # also clears the body's halt latch
    "constraint": GenerationRule(False, True, ""),
    "own_goal_dropped": GenerationRule(False, True, "persona"),
}


@dataclass
class Fence:
    """The current (generation, control_epoch). Anything tagged older is stale."""
    generation: int = 0
    control_epoch: int = 0

    def apply(self, trigger: Trigger) -> GenerationRule:
        rule = GENERATION_RULES[trigger]
        if rule.bump_generation:
            self.generation += 1
        if rule.bump_epoch:
            self.control_epoch += 1
        return rule

    def is_late(self, generation: int) -> bool:
        return generation < self.generation

    def decision_stale(self, generation: int, control_epoch: int) -> bool:
        """A planner decision made for (generation, epoch) is dropped if either moved."""
        return generation != self.generation or control_epoch != self.control_epoch


def should_cancel(ex: Execution, trigger: Trigger, fence: Fence) -> bool:
    """Which running executions a trigger cancels (after the fence was applied)."""
    if ex.finished:
        return False
    cancels = GENERATION_RULES[trigger].cancels
    if cancels == "older_generation":
        return ex.generation < fence.generation
    if cancels == "body":
        return "body" in ex.resources
    if cancels == "persona":
        return ex.source == "persona"
    return False


def is_stale_event(execution_id: str | None, executions: dict[str, Execution]) -> bool:
    """An event for an unknown or finished execution is a stale_result, never progress (PLAN 5.5)."""
    ex = executions.get(execution_id or "")
    return ex is None or ex.finished


__all__ = ["ExecStatus", "ACTIVE_STATUSES", "TERMINAL_STATUSES", "PREFIX", "Source", "Execution",
           "Rejected", "ExecutionHandle", "ResultHandle", "ExecutionIds", "ExecutionManager",
           "default_action", "is_terminal", "Trigger", "GenerationRule", "GENERATION_RULES", "Fence",
           "should_cancel", "is_stale_event"]
