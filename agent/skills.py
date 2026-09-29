"""Running executions with timeouts, and the speech queue.

Every tool the robot runs is started with ``robot.start(execution)`` (api/services.py
RobotBridge) and ends in a frozen ``ToolResult`` (api/results.py) with a lowercase
status: succeeded, failed, cancelled, timed_out or rejected.

``run_execution`` escalates on timeout (PLAN 5.6): ``cancel("timeout")``, then wait
``cancel_grace_s`` (3.0 s for body tools, 1.5 s for the rest), then ``robot.halt()``.
The result always resolves (invariant I1).
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable

from api.execution import Execution, Rejected
from api.results import ToolResult, finish, rejected
from api.summaries import summarize

CANCEL_GRACE_BODY_S = 3.0
CANCEL_GRACE_OTHER_S = 1.5
AFTER_HALT_S = 5.0            # a handle that ignores even a halt: give up and report internal_error


def cancel_grace(execution: Execution, profile: Any = None) -> float:
    body = "body" in execution.resources
    if profile is not None:
        return float(profile.cancel_grace_body_s if body else profile.cancel_grace_other_s)
    return CANCEL_GRACE_BODY_S if body else CANCEL_GRACE_OTHER_S


def _observation_id(robot: Any) -> str | None:
    fn = getattr(robot, "observation_id", None)
    try:
        return fn() if callable(fn) else None
    except Exception:
        return None


def start_execution(robot: Any, clock: Any, execution: Execution) -> tuple[Any, ToolResult | None]:
    """Start it; a capability rejection comes back as a rejected ToolResult instead of a handle."""
    try:
        handle = robot.start(execution)
    except Rejected as e:
        res = rejected(execution.tool_name, execution.args, stage=e.stage, code=e.code, message=e.message,
                       generation=execution.generation, control_epoch=execution.control_epoch,
                       source=execution.source, t=round(clock.now(), 3), execution_id=execution.execution_id,
                       observation_id=_observation_id(robot))
        return None, res
    if execution.status == "queued":
        execution.status = "running"
    if execution.t_started is None:
        execution.t_started = round(clock.now(), 3)
    return handle, None


async def run_execution(robot: Any, clock: Any, execution: Execution, timeout: float,
                        on_handle: Callable[[Any], None] | None = None, profile: Any = None) -> ToolResult:
    """Start one execution and await its result. On timeout: cancel, wait the grace, then halt."""
    handle, rej = start_execution(robot, clock, execution)
    if rej is not None:
        return rej
    if on_handle is not None:
        on_handle(handle)
    try:
        return await clock.wait_for(handle.result(), timeout)
    except asyncio.TimeoutError:
        pass
    handle.cancel("timeout")
    why = f"{execution.tool_name} took longer than {timeout:.0f} s"
    try:
        res = await clock.wait_for(handle.result(), cancel_grace(execution, profile))
    except asyncio.TimeoutError:
        robot.halt()
        why = f"{execution.tool_name} ignored cancel; halted"
        halted = True
        try:
            res = await clock.wait_for(handle.result(), AFTER_HALT_S)
        except asyncio.TimeoutError:
            return finish(execution, "failed", {"reason": "internal_error", "detail": why + "; no result",
                                                "halted_by_runtime": True},
                          t_end=round(clock.now(), 3), observation_id=_observation_id(robot))
    else:
        halted = False
    # halted_by_runtime: this halt is the runtime's own (not a user stop); the harness releases it (_finish)
    data = {**res.data, "reason": "timeout", "detail": why, **({"halted_by_runtime": True} if halted else {})}
    return dataclasses.replace(res, status="timed_out", data=data,
                               summary=summarize(execution.tool_name, "timed_out", data,
                                                 action=execution.action, args=execution.args))


# ----------------------------------------------------------------------
# Map helpers (the service's own timeout is preferred: RobotBridge.timeout_s)
# ----------------------------------------------------------------------
def path_length(map_: dict[str, Any], start: str | None, goal: str, blocked: set[tuple[str, str]]) -> float:
    if start is None:
        return 12.0
    if start == goal:
        return 0.0
    bad = {frozenset(e) for e in blocked}
    adj: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for a, b, length in map_.get("edges", []):
        if frozenset((a, b)) in bad:
            continue
        adj[a].append((b, length))
        adj[b].append((a, length))
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == goal:
            return d
        if d > dist.get(node, math.inf):
            continue
        for nxt, length in adj[node]:
            nd = d + length
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return 40.0     # no known path: the robot's planner may still find one


def navigate_timeout(map_: dict[str, Any], start: str | None, goal: str, blocked: set[tuple[str, str]],
                     robot: Any = None, args: dict[str, Any] | None = None) -> float:
    """PLAN 1.3 #22: clamp(1.8 * path_m / v + 12, 20, 240). NavigationService.timeout_s when the robot has it."""
    fn = getattr(robot, "timeout_s", None)
    if callable(fn):
        try:
            return float(fn("navigate", dict(args or {"location": goal})))
        except Exception:
            pass
    v = float(map_.get("nav_speed_mps") or 0.4)
    return max(20.0, min(240.0, 1.8 * path_length(map_, start, goal, blocked) / v + 12.0))


# ----------------------------------------------------------------------
@dataclass
class SpeechItem:
    entry: Execution                # the speak execution
    text: str
    created_for: int
    control_epoch: int = 0


class SpeechQueue:
    """FIFO speech, played through ``robot.start(speak execution)``. Lines are tagged with the
    generation they serve, so a correction can drop the ones that no longer apply, and with
    the control epoch, so a resume can fence lines from before a stop."""

    def __init__(self, robot: Any, clock: Any, on_change=None) -> None:
        self.robot = robot
        self.clock = clock
        self.items: collections.deque[SpeechItem] = collections.deque()
        self.current: tuple[SpeechItem, Any] | None = None
        self._evt = asyncio.Event()
        self._on_change = on_change
        self.on_result: Callable[[Execution, ToolResult], None] | None = None   # the harness records it

    def enqueue(self, item: SpeechItem) -> None:
        item.entry.status = "queued"
        item.entry.data = {"speech": "queued", "status": "queued", "utterance_id": item.entry.execution_id}
        self.items.append(item)
        self._evt.set()

    def enqueue_priority(self, item: SpeechItem) -> None:
        """Queue runtime safety speech ahead of ordinary planner speech."""
        item.entry.status = "queued"
        item.entry.data = {"speech": "queued", "status": "queued", "utterance_id": item.entry.execution_id}
        self.items.appendleft(item)
        self._evt.set()

    def busy(self) -> bool:
        return bool(self.items) or self.current is not None

    def _drop(self, item: SpeechItem, why: str) -> None:
        item.entry.status = "dropped"
        item.entry.t_end = self.clock.now()
        item.entry.data = {**(item.entry.data or {}), "speech": "dropped", "reason": why}
        self._record(item.entry, finish(item.entry, "dropped", item.entry.data, t_end=round(self.clock.now(), 3)))

    def _record(self, entry: Execution, result: ToolResult) -> None:
        entry.result = result
        if self.on_result is not None:
            self.on_result(entry, result)

    def drop_older_than(self, version: int) -> list[str]:
        dropped = []
        for item in list(self.items):
            if item.created_for < version:
                self.items.remove(item)
                self._drop(item, f"superseded (generation {version})")
                dropped.append(item.text)
        if self.current and self.current[0].created_for < version:
            self.current[1].cancel("superseded")
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    def drop_before_epoch(self, epoch: int, keep_tags: tuple[str, ...] = ("runtime:safety-ack",)) -> list[str]:
        """Fence speech produced by planner calls from an older control epoch (wired into resume).
        Safety acknowledgements are kept: they must be said exactly once."""
        dropped = []
        for item in list(self.items):
            if item.control_epoch < epoch and item.entry.tag not in keep_tags:
                self.items.remove(item)
                self._drop(item, f"fenced (epoch {epoch})")
                dropped.append(item.text)
        if (self.current and self.current[0].control_epoch < epoch
                and self.current[0].entry.tag not in keep_tags):
            self.current[1].cancel("fenced")
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    def cut_all(self) -> list[str]:
        dropped = []
        for item in list(self.items):
            self._drop(item, "cut")
            dropped.append(item.text)
        self.items.clear()
        if self.current:
            self.current[1].cancel("cut")
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    async def run(self) -> None:
        while True:
            await self._evt.wait()
            self._evt.clear()
            while self.items:
                item = self.items.popleft()
                entry = item.entry
                handle, rej = start_execution(self.robot, self.clock, entry)
                if rej is not None:
                    entry.status = "rejected"
                    entry.data = dict(rej.data)
                    entry.t_end = self.clock.now()
                    self._record(entry, rej)
                    continue
                entry.status = "running"
                entry.t_start = self.clock.now()
                self.current = (item, handle)
                words = len(item.text.split())
                try:
                    res = await self.clock.wait_for(handle.result(), 0.4 * words + 4.0)
                except asyncio.TimeoutError:
                    handle.cancel("timeout")
                    try:
                        res = await self.clock.wait_for(handle.result(), CANCEL_GRACE_OTHER_S)
                    except asyncio.TimeoutError:
                        res = finish(entry, "timed_out", {"reason": "timeout"}, t_end=round(self.clock.now(), 3))
                    else:
                        res = dataclasses.replace(res, status="timed_out", data={**res.data, "reason": "timeout"})
                entry.status = res.status
                entry.data = {**(entry.data or {}), **dict(res.data)}
                entry.t_end = self.clock.now()
                self._record(entry, res)
                self.current = None
                if self._on_change:
                    self._on_change()


__all__ = ["CANCEL_GRACE_BODY_S", "CANCEL_GRACE_OTHER_S", "cancel_grace", "start_execution", "run_execution",
           "path_length", "navigate_timeout", "SpeechItem", "SpeechQueue"]
