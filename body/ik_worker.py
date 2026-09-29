"""IK off the body's control thread (M2b wave 2, body-fix B-D1; docs/contracts/m1.md §3.9, §3.10).

The arm_script IK (body/arm_script.py: via points + the goal solve at an op start, and the 10 Hz world-goal re-solve
during a settle) runs in a worker PROCESS, never on the service's control thread and never under the arm lock.

Why a process and not a thread: `g1_kin.ik_palm` calls `np.linalg.solve` every iteration, and numpy releases and
re-takes the GIL around each of those calls. A thread that does that starves every other Python thread of the GIL for
the whole solve (the "convoy effect": each re-take counts as a GIL switch, so a waiting thread never gets to force
one). Live, before this module, halts sent during an unreachable-goal IK (198-233 ms) were acked after 176-221 ms
(verifier B-D1). On the laptop, a PULL thread next to a continuous IK loop in the same process waited 64-131 s for its
messages, and 0.09-0.15 ms with the IK in a worker process (`outputs/m2b_wave2/bodyfix/gil_probe.txt`). A worker
process shares no GIL with the halt lane.

    ik = IKWorker("process")      # BodyService: one spawn-context worker process, warmed up at setup()
    fut = ik.submit(fn, *args)    # concurrent.futures.Future; fn must be a module-level function (pickled by name)
    ik = IKWorker("inline")       # tests and tools: runs fn at once and returns a finished Future (same code path)

A broken worker (killed, crashed) is replaced once per submit; the failed Future carries the exception and the caller
answers `ik_unavailable`. `stats` counts submits, failures, restarts and the solve times the worker reports.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool
import threading
import time
from typing import Any, Callable

MODES = ("process", "inline")


def _warm() -> float:
    """Imported in the worker: loads numpy and body.g1_kin once, so the first real solve does not pay for it."""
    from . import arm_script  # noqa: F401  (imports g1_kin, joint_map, numpy)

    return time.monotonic()


class IKWorker:
    def __init__(self, mode: str = "process", log: Callable[[str], None] | None = None):
        if mode not in MODES:
            raise ValueError(f"IKWorker mode must be one of {MODES}")
        self.mode = mode
        self.log = log or (lambda *_: None)
        self._pool: cf.ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self.stats = {"mode": mode, "submits": 0, "failures": 0, "restarts": 0, "warm_s": None}

    # -- lifecycle --------------------------------------------------------------------------------------------
    def _new_pool(self) -> cf.ProcessPoolExecutor:
        return cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))

    def start(self) -> "IKWorker":
        """Start the worker process and warm it up in the background (never blocks the caller)."""
        if self.mode != "process":
            return self
        with self._lock:
            if self._pool is None:
                self._pool = self._new_pool()
        t0 = time.monotonic()
        fut = self.submit(_warm)

        def _done(f: cf.Future) -> None:
            try:
                f.result()
                self.stats["warm_s"] = round(time.monotonic() - t0, 2)
                self.log(f"[ik] worker process ready in {self.stats['warm_s']} s")
            except Exception as e:  # noqa: BLE001
                self.log(f"[ik] worker warm-up failed: {e!r}")

        fut.add_done_callback(_done)
        return self

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    # -- API --------------------------------------------------------------------------------------------------
    def submit(self, fn: Callable[..., Any], *args) -> cf.Future:
        self.stats["submits"] += 1
        if self.mode == "inline":
            fut: cf.Future = cf.Future()
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - the caller reads it from the Future, as from the process
                self.stats["failures"] += 1
                fut.set_exception(e)
            return fut
        for attempt in (0, 1):
            with self._lock:
                if self._pool is None:
                    self._pool = self._new_pool()
                pool = self._pool
            try:
                return pool.submit(fn, *args)
            except (BrokenProcessPool, RuntimeError) as e:
                self.stats["restarts"] += 1
                self.log(f"[ik] worker broken ({e!r}); restarting it")
                with self._lock:
                    if self._pool is pool:
                        self._pool = None
                pool.shutdown(wait=False, cancel_futures=True)
        fut = cf.Future()
        self.stats["failures"] += 1
        fut.set_exception(RuntimeError("ik worker unavailable"))
        return fut

    def note_failure(self, exc: BaseException) -> None:
        """A Future failed: count it; a broken pool is dropped so the next submit starts a fresh worker."""
        self.stats["failures"] += 1
        if isinstance(exc, BrokenProcessPool):
            with self._lock:
                pool, self._pool = self._pool, None
            if pool is not None:
                self.stats["restarts"] += 1
                pool.shutdown(wait=False, cancel_futures=True)

    def snapshot(self) -> dict:
        return dict(self.stats)
