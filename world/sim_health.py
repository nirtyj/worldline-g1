"""Sim health from the real-time factor (PLAN §3.5; M2.md P1.9, R.6). World owns it because RTF is simulator truth.

    ok         RTF >= 0.95, or no RTF known (lite runs on its own clock; a stale P1 is navigation's problem)
    degraded   RTF < 0.95 held for `degraded_hold_s` (5 s)    -> `manipulate` is rejected at CAPABILITY
    unsafe     RTF < 0.85 held for `unsafe_hold_s` (3 s)      -> every body tool is rejected at CAPABILITY

The hold times keep a single render hitch (M1: worst 1 s window 0.82, 2-3 % of 1 s windows below 0.95 in a quiet
integrated run, docs/contracts/m1.md §1.10) from flapping capabilities; `hold_s = 0` makes the thresholds instant.
A state recovers once RTF has been back above its threshold for `recover_s`. When P1 sends its own verdict
(`sim.health {state}`), `force()` takes it as is.

RtfMonitor is fed from one thread (P1's SUB) and read from the event loop, so it locks. Time comes from `clock`
(monotonic seconds) so tests drive it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

STATES = ("ok", "degraded", "unsafe")


@dataclass(frozen=True)
class SimHealth:
    state: str                         # "ok" | "degraded" | "unsafe"
    rtf: float | None                  # the newest RTF sample (1 s window), None when unknown
    detail: str = ""
    source: str = "none"               # "p1:sim.health" | "p1:gt.pose" | "lite" | "fixture" | "none"
    since_s: float = 0.0               # how long this state has held
    age_s: float | None = None         # age of the newest RTF sample

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    @property
    def degraded(self) -> bool:
        return self.state in ("degraded", "unsafe")

    @property
    def unsafe(self) -> bool:
        return self.state == "unsafe"

    def to_dict(self) -> dict:
        return {"state": self.state, "rtf": None if self.rtf is None else round(self.rtf, 3), "detail": self.detail,
                "source": self.source, "since_s": round(self.since_s, 2),
                "age_s": None if self.age_s is None else round(self.age_s, 2)}


@dataclass(frozen=True)
class SimHealthConfig:
    degraded_below: float = 0.95
    unsafe_below: float = 0.85
    degraded_hold_s: float = 5.0
    unsafe_hold_s: float = 3.0
    recover_s: float = 2.0
    stale_s: float = 3.0               # no RTF sample for this long -> state "ok" with rtf None (unknown)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SimHealthConfig":
        d = dict(d or {})
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


class RtfMonitor:
    def __init__(self, cfg: SimHealthConfig | None = None, *, clock: Callable[[], float] = time.monotonic,
                 source: str = "none"):
        self.cfg = cfg or SimHealthConfig()
        self.clock = clock
        self.source = source
        self._lock = threading.Lock()
        self._rtf: float | None = None
        self._t_rtf: float | None = None
        self._below_deg: float | None = None     # since when RTF has been below degraded_below
        self._below_uns: float | None = None
        self._above_deg: float | None = None     # since when RTF has been back above degraded_below
        self._above_uns: float | None = None
        self._state = "ok"
        self._since = self.clock()
        self._forced: tuple[str, float, str] | None = None   # P1's own verdict: (state, t, detail)

    def update(self, rtf: float | None, *, source: str | None = None) -> SimHealth:
        """Feed one RTF sample (a 1 s window). None or NaN is ignored."""
        now = self.clock()
        with self._lock:
            if source:
                self.source = source
            if rtf is None or rtf != rtf:
                return self._snapshot(now)
            c = self.cfg
            self._rtf, self._t_rtf = float(rtf), now
            if rtf < c.degraded_below:
                self._below_deg = self._below_deg if self._below_deg is not None else now
                self._above_deg = None
            else:
                self._below_deg = None
                self._above_deg = self._above_deg if self._above_deg is not None else now
            if rtf < c.unsafe_below:
                self._below_uns = self._below_uns if self._below_uns is not None else now
                self._above_uns = None
            else:
                self._below_uns = None
                self._above_uns = self._above_uns if self._above_uns is not None else now
            self._step(now)
            return self._snapshot(now)

    def force(self, state: str, detail: str = "", *, source: str = "p1:sim.health") -> None:
        """P1's own verdict (if its sim.health carries a `state`); it wins while fresh (stale_s)."""
        if state not in STATES:
            return
        with self._lock:
            self._forced = (state, self.clock(), detail)
            self.source = source

    def _step(self, now: float) -> None:
        c = self.cfg
        want = self._state
        uns = self._below_uns is not None and now - self._below_uns >= c.unsafe_hold_s
        deg = self._below_deg is not None and now - self._below_deg >= c.degraded_hold_s
        if uns:
            want = "unsafe"
        elif self._state == "unsafe":
            if self._above_uns is not None and now - self._above_uns >= c.recover_s:
                want = "degraded" if (self._below_deg is not None) else "ok"
        elif deg:
            want = "degraded"
        elif self._state == "degraded":
            if self._above_deg is not None and now - self._above_deg >= c.recover_s:
                want = "ok"
        if want != self._state:
            self._state, self._since = want, now

    def _snapshot(self, now: float) -> SimHealth:
        c = self.cfg
        age = None if self._t_rtf is None else now - self._t_rtf
        if self._forced is not None and now - self._forced[1] <= c.stale_s:
            st, t, detail = self._forced
            return SimHealth(st, self._rtf, detail or f"P1 says {st}", self.source, now - t, age)
        if age is None or age > c.stale_s:
            return SimHealth("ok", None, "no RTF sample" if age is None else f"no RTF for {age:.1f} s",
                             self.source, 0.0, age)
        self._step(now)
        rtf = self._rtf
        if self._state == "unsafe":
            detail = f"sim below real time: UNSAFE, rtf {rtf:.2f} (< {c.unsafe_below} for {c.unsafe_hold_s:g} s)"
        elif self._state == "degraded":
            detail = f"sim below real time: DEGRADED, rtf {rtf:.2f} (< {c.degraded_below} for {c.degraded_hold_s:g} s)"
        else:
            detail = f"rtf {rtf:.2f}"
        return SimHealth(self._state, rtf, detail, self.source, now - self._since, age)

    def state(self) -> SimHealth:
        with self._lock:
            return self._snapshot(self.clock())


def fixed(state: str = "ok", rtf: float | None = None, source: str = "lite", detail: str = "") -> SimHealth:
    """A constant health (lite: no real-time constraint; tests: a fixture)."""
    return SimHealth(state, rtf, detail or ("no real-time constraint" if source == "lite" else state), source)


__all__ = ["STATES", "SimHealth", "SimHealthConfig", "RtfMonitor", "fixed"]
