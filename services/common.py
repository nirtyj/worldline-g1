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
    """The runtime-side halt latch (PLAN §5.6) in the executions' control_epoch space: ONE epoch space with the body
    (M2b R.2). A halt latches at the control_epoch it fences, i.e. every running body execution's control_epoch is
    <= `epoch`; the body's halt lane gets the same number (robot/body_client.SonicBody maps it onto the wire with a
    per-session constant). There is no counter of its own any more.

        halt(epoch)              latch at `epoch` (while latched, never below the latched epoch)
        halted_since(ce)         latched and ce <= epoch: an execution of that control_epoch ends failed(halted)
        resume(epoch)            clear the latch; `epoch` = the control_epoch the next executions carry

    `halt()` without an epoch (tools and tests that do not track executions) latches one above the last known epoch.
    """
    epoch: int = 0                           # the last halt's control_epoch (0 before any)
    latched: bool = False
    resume_epoch: int | None = None          # the control_epoch the last resume opened
    history: list[tuple[float, str, int]] = field(default_factory=list)

    def halt(self, epoch: int | None = None) -> int:
        if epoch is None:
            e = max(self.epoch, self.resume_epoch or 0) + (0 if self.latched else 1)
        else:
            e = int(epoch)
        if self.latched:
            e = max(e, self.epoch)
        self.epoch = e
        self.latched = True
        self.history.append((time.monotonic(), "halt", self.epoch))
        return self.epoch

    def resume(self, epoch: int | None = None) -> None:
        self.latched = False
        if epoch is not None:
            self.resume_epoch = int(epoch)
        self.history.append((time.monotonic(), "resume", self.epoch if epoch is None else int(epoch)))

    def halted_since(self, epoch: int) -> bool:
        """True while latched for an execution of control_epoch `epoch` (it started at or before the halt)."""
        return self.latched and int(epoch) <= self.epoch


# ---------------------------------------------------------------------------------------------------------------
# The body's fences and leases, as services see them (docs/contracts/m1.md §3.10-§3.11). The BodyPort may be an M1
# body or the lite body, which have neither: these helpers make that a no-op.
# ---------------------------------------------------------------------------------------------------------------
# a `stale_command` whose data.why names a fence is a stale result, not a failure; the body may give the fence case
# its own reason (the body verifier's item: `stale_command` also means a t_wall-stale stream message)
FENCE_WHY = ("control_epoch", "generation", "resume_epoch")
STALE_REASONS = ("stale_command", "stale_fence", "stale_epoch", "stale_generation")


def refusal(reason: Any, data: dict | None = None) -> str | None:
    """What a body refusal means to the runtime: 'halted' | 'stale_result' | 'body_busy' | None (not a fence).
    Keyed on the reason, and on data.why for `stale_command`."""
    r = str(reason or "")
    d = data or {}
    if r == "halted":
        return "halted"
    if r == "body_busy":
        return "body_busy"
    if r in STALE_REASONS and (r != "stale_command" or str(d.get("why") or "") in FENCE_WHY):
        return "stale_result"
    return None


def body_fence(body: Any, execution: Execution) -> dict:
    """The body's fence fields for this execution ({} on a body without fences)."""
    fn = getattr(body, "fence", None)
    try:
        return dict(fn(execution)) if callable(fn) else {}
    except Exception:  # noqa: BLE001
        return {}


async def body_acquire(body: Any, execution: Execution, mode: str) -> dict:
    """Lease the body for a body execution (B.2): {ok, reason, meaning, lease}. A body without leases: ok."""
    fn = getattr(body, "acquire", None)
    if not callable(fn):
        return {"ok": True, "reason": None, "lease": None}
    try:
        rep = await fn(execution, mode)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "lease_error", "meaning": None, "detail": repr(e)[:200]}
    if not rep.get("ok") and "meaning" not in rep:
        rep = {**rep, "meaning": refusal(rep.get("reason"), rep.get("data"))}
    return rep


async def body_release(body: Any, execution: Execution) -> None:
    fn = getattr(body, "release", None)
    if callable(fn):
        try:
            await fn(execution.execution_id)
        except Exception:  # noqa: BLE001
            pass


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
