"""Wall-clock pacing of the physics loop and real-time-factor (RTF) accounting.

The deploy runs on the wall clock (4 wall-clock threads, $WBC/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/
g1_deploy_onnx_ref.cpp:11-30) and has no sim-time mode, so sim time must track wall time. RtPacer keeps
t_sim ~= t_wall - t0: it sleeps when ahead and runs steps back-to-back when behind (catch-up). If it falls behind by
more than max_lag_s it gives up on catching up (an "overrun"), re-anchors the schedule, and that lost time shows up
as RTF < 1.
"""
from __future__ import annotations

import collections
import time

import numpy as np


class RollingRate:
    """Counts events and reports the rate over a trailing window (seconds)."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self.t = collections.deque()
        self.total = 0

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self.t.append(now)
        self.total += 1
        lo = now - self.window_s
        while self.t and self.t[0] < lo:
            self.t.popleft()

    def rate(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        lo = now - self.window_s
        while self.t and self.t[0] < lo:
            self.t.popleft()
        return len(self.t) / self.window_s


class DurationStats:
    """Keeps the last N durations (ms) for mean/p50/p99/max."""

    def __init__(self, n: int = 4000):
        self.buf = collections.deque(maxlen=n)
        self.count = 0
        self.max_all = 0.0

    def add(self, ms: float) -> None:
        self.buf.append(ms)
        self.count += 1
        if ms > self.max_all:
            self.max_all = ms

    def summary(self) -> dict:
        if not self.buf:
            return {"mean": None, "p50": None, "p99": None, "max": None, "max_all": None, "n": 0}
        a = np.fromiter(self.buf, dtype=np.float64)
        return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
                "p99": round(float(np.percentile(a, 99)), 3), "max": round(float(a.max()), 3),
                "max_all": round(self.max_all, 3), "n": self.count}


class RtPacer:
    def __init__(self, dt: float, enabled: bool = True, max_lag_s: float = 0.1):
        self.dt = dt
        self.enabled = enabled
        self.max_lag_s = max_lag_s
        self.t0_wall = None
        self.n_steps = 0
        self.sched_anchor_wall = None
        self.sched_anchor_step = 0
        self.overruns = 0
        self.lost_s = 0.0
        # (wall, sim) samples for windowed RTF
        self._hist = collections.deque()

    def start(self) -> None:
        now = time.perf_counter()
        self.t0_wall = now
        self.sched_anchor_wall = now
        self.sched_anchor_step = self.n_steps
        self._hist.clear()
        self._hist.append((now, self.n_steps * self.dt))

    @property
    def t_sim(self) -> float:
        return self.n_steps * self.dt

    def step_done(self) -> None:
        """Call once after every physics step."""
        self.n_steps += 1
        if self.n_steps % 10:
            return
        now = time.perf_counter()
        self._hist.append((now, self.n_steps * self.dt))
        lo = now - 10.5
        while len(self._hist) > 2 and self._hist[0][0] < lo:
            self._hist.popleft()

    def wait(self) -> None:
        """Sleep until the wall-clock deadline of the next step (no-op when pacing is off)."""
        if not self.enabled:
            return
        target = self.sched_anchor_wall + (self.n_steps - self.sched_anchor_step) * self.dt
        now = time.perf_counter()
        ahead = target - now
        if ahead > 0:
            # sleep most of it, spin the last ~0.3 ms for accuracy
            if ahead > 0.0008:
                time.sleep(ahead - 0.0003)
            while time.perf_counter() < target:
                pass
        elif -ahead > self.max_lag_s:
            # too far behind: drop the backlog instead of bursting
            self.overruns += 1
            self.lost_s += -ahead
            self.sched_anchor_wall = now
            self.sched_anchor_step = self.n_steps

    def rtf(self, window_s: float) -> float | None:
        if len(self._hist) < 2:
            return None
        now_w, now_s = self._hist[-1]
        lo = now_w - window_s
        # oldest sample inside the window
        for w, s in self._hist:
            if w >= lo:
                dw = now_w - w
                return (now_s - s) / dw if dw > 1e-6 else None
        return None

    def rtf_total(self) -> float | None:
        if self.t0_wall is None:
            return None
        dw = time.perf_counter() - self.t0_wall
        return self.t_sim_since_start / dw if dw > 0 else None

    @property
    def t_sim_since_start(self) -> float:
        return self.n_steps * self.dt - (self._start_sim if hasattr(self, "_start_sim") else 0.0)

    def mark_start_sim(self) -> None:
        self._start_sim = self.n_steps * self.dt
        self.start()

    def physics_hz(self, window_s: float = 1.0) -> float | None:
        r = self.rtf(window_s)
        return None if r is None else r / self.dt
