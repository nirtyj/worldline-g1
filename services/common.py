"""Shared plumbing for services: the execution runner (ResultHandle + task + always-resolving result), the halt gate
and a tiny event sink.

Every service method that starts work returns an `api.execution.ResultHandle` whose `result()` ALWAYS resolves
(invariant I1) to a frozen `api.results.ToolResult` built with `api.results.finish`, stamped with an observation id
(the latest glance unless the work ran its own observation).
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from api.execution import Execution, ResultHandle
from api.results import ToolResult, finish

log = logging.getLogger("services")


class WallClock:
    """Fallback clock (sim.clock.SimClock has the same surface)."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    async def sleep(self, s: float) -> None:
        await asyncio.sleep(max(0.0, s))

    async def wait_for(self, aw, timeout: float):
        return await asyncio.wait_for(aw, timeout)


class EventSink:
    """emit(type, **fields) -> dict. Wraps a sim.log.EventLog (System 1 reads speech_* from it) and/or a list of
    asyncio queues (RobotBridge.events())."""

    def __init__(self, log_: Any = None) -> None:
        self.log = log_
        self.queues: list[asyncio.Queue] = []

    def emit(self, type: str, **fields: Any) -> dict:
        ev = None
        if self.log is not None:
            try:
                ev = self.log.emit(type, **fields)
            except Exception:  # noqa: BLE001
                ev = None
        ev = dict(ev) if ev else {"type": type, **fields}
        for q in list(self.queues):
            try:
                q.put_nowait(dict(ev))
            except Exception:  # noqa: BLE001
                pass
        return ev

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.queues.append(q)
        return q


@dataclass
class HaltGate:
    """The runtime-side halt latch (PLAN §5.6): halt() bumps the epoch and latches; resume() clears it. Running
    body executions see `halted_since(epoch)` and end failed(halted)."""
    epoch: int = 0
    latched: bool = False
    history: list[tuple[float, str, int]] = field(default_factory=list)

    def halt(self, epoch: int | None = None) -> int:
        self.epoch = max(self.epoch + 1, int(epoch) if epoch is not None else 0)
        self.latched = True
        self.history.append((time.monotonic(), "halt", self.epoch))
        return self.epoch

    def resume(self, epoch: int | None = None) -> None:
        self.latched = False
        if epoch is not None:
            self.epoch = max(self.epoch, int(epoch))
        self.history.append((time.monotonic(), "resume", self.epoch))

    def halted_since(self, epoch: int) -> bool:
        return self.latched and self.epoch > epoch


def start_execution(execution: Execution, work: Callable[[ResultHandle], Awaitable[ToolResult]], *,
                    clock: Any, observation_id: Callable[[], str | None] | None = None,
                    on_cancel: Callable[[str], Any] | None = None, executor: str | None = None,
                    active: dict[str, Execution] | None = None) -> ResultHandle:
    """Run `work(handle)` as a task; its ToolResult resolves the handle. Crashes -> failed/internal_error; cancel
    before the task starts -> cancelled "cancelled before start" (I2)."""
    handle = ResultHandle(execution, on_cancel=on_cancel)
    if executor and execution.executor is None:
        execution.executor = executor
    if execution.status == "queued":
        execution.status = "running"
    if execution.t_started is None:
        execution.t_started = round(clock.now(), 3)
    if active is not None:
        active[execution.execution_id] = execution

    def obs() -> str | None:
        try:
            return observation_id() if observation_id else None
        except Exception:  # noqa: BLE001
            return None

    async def runner() -> None:
        try:
            await asyncio.sleep(0)
            if handle.cancel_requested:
                res = finish(execution, "cancelled", {"reason": "cancelled before start"},
                             t_end=round(clock.now(), 3), observation_id=obs(), executor=executor)
            else:
                res = await work(handle)
        except asyncio.CancelledError:
            res = finish(execution, "failed", {"reason": "shutdown"}, t_end=round(clock.now(), 3),
                         observation_id=obs(), executor=executor)
            handle.resolve(res)
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("execution %s crashed", execution.execution_id)
            res = finish(execution, "failed", {"reason": "internal_error", "detail": repr(e)[:300],
                                               "trace": traceback.format_exc()[-800:]},
                         t_end=round(clock.now(), 3), observation_id=obs(), executor=executor)
        if res.observation_id is None:
            import dataclasses
            res = dataclasses.replace(res, observation_id=obs())
        handle.resolve(res)
        if active is not None:
            active.pop(execution.execution_id, None)

    handle.task = asyncio.ensure_future(runner())       # type: ignore[attr-defined]
    return handle
