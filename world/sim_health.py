"""Sim health from the real-time factor (PLAN §3.5; M2.md P1.9, R.6). World owns it because RTF is simulator truth.

    ok         the default, and whenever no RTF is known (lite runs on its own clock; a stale P1 is navigation's
               problem)
    degraded   rtf_5s < `degraded_below` (0.90) held for `degraded_hold_s` (3 s)   -> `manipulate` is rejected at
               CAPABILITY. Back to ok once rtf_5s >= `recover_above` (0.94) has held for `recover_s` (2 s)
    unsafe     rtf_3s < `unsafe_below` (0.85) held for `unsafe_hold_s` (2 s), or rtf_3s < `unsafe_now_below` (0.70)
               at once (a stalled sim does not wait)                          -> every body tool is rejected
               Leaves once rtf_3s >= `unsafe_recover_above` (0.90) has held for `recover_s`: to ok if the degraded
               window has recovered too, else to degraded

Hysteresis and dwell (PLAN §3.5, 2026-09-29 stance-fix live finding): the live house sits at rtf_5s 0.92-0.96, so
the old instant P1 level (degraded = rtf_5s < 0.95) flipped ok <-> degraded every few seconds; every flip was a
capability_changed that woke the planner, and picks right after a walk were refused during 1-2 s dips. The floor
is 0.90 because nothing in the timing evidence fails between 0.90 and 0.95 (outputs/m2b_finish/wrap/summary_all.txt:
stand/walk p10 down to 0.82 with 0 falls; the scripted pick held and carried at rtf_5s 0.93); the §0.10 b timing
gate (p10 >= 0.98) is the bar for a measured run, not a runtime capability.

Inputs: P1's sim.health (1 Hz) carries rtf_1s / rtf_3s / rtf_5s; world applies its own thresholds and dwell to the
windows and never adopts P1's instant `level` (docs/contracts/p1_m2b.md §10.1: world owns what it does with it).
Without windows (M1's gt.pose `rtf`, a 1 s window at 50 Hz) the 3 s / 5 s windows are the mean of the 1 s samples
over the last 3 s / 5 s. A P1 level with no numbers at all goes through the same dwell. `rtf` is the newest 1 s
sample (the number shown).

RtfMonitor is fed from one thread (P1's SUB) and read from the event loop, so it locks. Time comes from `clock`
(monotonic seconds) so tests drive it.
"""

from __future__ import annotations

import collections
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
    rtf_3s: float | None = None        # the unsafe window (P1's, or the mean of the 1 s samples)
    rtf_5s: float | None = None        # the degraded window

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
        def r(v):
            return None if v is None else round(v, 3)
        return {"state": self.state, "rtf": r(self.rtf), "rtf_3s": r(self.rtf_3s), "rtf_5s": r(self.rtf_5s),
                "detail": self.detail, "source": self.source, "since_s": round(self.since_s, 2),
                "age_s": None if self.age_s is None else round(self.age_s, 2)}


@dataclass(frozen=True)
class SimHealthConfig:
    degraded_below: float = 0.90       # rtf_5s below this ...
    degraded_hold_s: float = 3.0       # ... held this long -> degraded
    recover_above: float = 0.94        # degraded -> ok: rtf_5s at or above this ...
    recover_s: float = 2.0             # ... held this long (and unsafe -> below: rtf_3s >= unsafe_recover_above)
    unsafe_below: float = 0.85         # rtf_3s below this ...
    unsafe_hold_s: float = 2.0         # ... held this long -> unsafe
    unsafe_now_below: float = 0.70     # rtf_3s below this -> unsafe at once
    unsafe_recover_above: float = 0.90
    stale_s: float = 3.0               # no RTF sample for this long -> state "ok" with rtf None (unknown)
    window_3s: float = 3.0             # the windows built from 1 s samples when P1 sends none
    window_5s: float = 5.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "SimHealthConfig":
        d = dict(d or {})
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


def _held(since: float | None, now: float, hold_s: float) -> bool:
    return since is not None and now - since >= hold_s


class RtfMonitor:
    def __init__(self, cfg: SimHealthConfig | None = None, *, clock: Callable[[], float] = time.monotonic,
                 source: str = "none"):
        self.cfg = cfg or SimHealthConfig()
        self.clock = clock
        self.source = source
        self._lock = threading.Lock()
        self._rtf: float | None = None               # newest 1 s sample
        self._r3: float | None = None                # newest unsafe window
        self._r5: float | None = None                # newest degraded window
        self._t_rtf: float | None = None             # when the newest sample came
        self._hist: collections.deque[tuple[float, float]] = collections.deque()   # (t, rtf_1s), 1 s samples
        self._deg_low: float | None = None           # since when the degraded window has been below degraded_below
        self._deg_high: float | None = None          # ... at or above recover_above
        self._uns_low: float | None = None           # since when the unsafe window has been below unsafe_below
        self._uns_high: float | None = None          # ... at or above unsafe_recover_above
        self._uns_now = False                        # the newest unsafe window is below unsafe_now_below
        self._entered = ""                           # why the current non-ok state was entered
        self._state = "ok"
        self._since = self.clock()
        self.transitions: list[tuple[float, str, str]] = []   # (t, from, to): every level change after the dwell

    # ------------------------------------------------------------------ feed
    def update(self, rtf: float | None, *, rtf_3s: float | None = None, rtf_5s: float | None = None,
               level: str | None = None, source: str | None = None) -> SimHealth:
        """One sample: the 1 s RTF, and P1's windows / level when it sent them. None or NaN values are ignored;
        a sample with nothing usable changes nothing."""
        now = self.clock()

        def num(v):
            return float(v) if isinstance(v, (int, float)) and v == v else None
        r1, r3, r5 = num(rtf), num(rtf_3s), num(rtf_5s)
        lvl = level if level in STATES else None
        with self._lock:
            if source:
                self.source = source
            if r1 is None and r3 is None and r5 is None and lvl is None:
                return self._snapshot(now)
            c = self.cfg
            if self._t_rtf is not None and now - self._t_rtf > c.stale_s:
                self._reset(now)                     # P1 went quiet: what we knew before is not a verdict now
            self._t_rtf = now
            if r1 is not None:
                self._rtf = r1
                self._hist.append((now, r1))
            while self._hist and now - self._hist[0][0] > max(c.window_3s, c.window_5s):
                self._hist.popleft()
            if r3 is None and r5 is None and r1 is not None:        # no windows from P1: build them here
                r3, r5 = self._mean(now, c.window_3s), self._mean(now, c.window_5s)
            r3, r5 = (r3 if r3 is not None else r5), (r5 if r5 is not None else r3)
            self._r3, self._r5 = r3, r5
            # what this sample says: from the windows, else (P1's level and no numbers) from the level
            if r3 is not None and r5 is not None:
                deg_low, deg_high = r5 < c.degraded_below, r5 >= c.recover_above
                uns_low, uns_high = r3 < c.unsafe_below, r3 >= c.unsafe_recover_above
                self._uns_now = r3 < c.unsafe_now_below
            else:
                deg_low, deg_high = lvl in ("degraded", "unsafe"), lvl == "ok"
                uns_low, uns_high = lvl == "unsafe", lvl in ("ok", "degraded")
                self._uns_now = False
            self._deg_low = (self._deg_low if self._deg_low is not None else now) if deg_low else None
            self._deg_high = (self._deg_high if self._deg_high is not None else now) if deg_high else None
            self._uns_low = (self._uns_low if self._uns_low is not None else now) if uns_low else None
            self._uns_high = (self._uns_high if self._uns_high is not None else now) if uns_high else None
            self._step(now)
            return self._snapshot(now)

    def _mean(self, now: float, window_s: float) -> float | None:
        xs = [r for t, r in self._hist if now - t <= window_s]
        return sum(xs) / len(xs) if xs else None

    def _reset(self, now: float) -> None:
        self._hist.clear()
        self._deg_low = self._deg_high = self._uns_low = self._uns_high = None
        self._uns_now = False
        if self._state != "ok":
            self.transitions.append((now, self._state, "ok"))
            self._state, self._since, self._entered = "ok", now, ""

    def _step(self, now: float) -> None:
        c = self.cfg
        want, why = self._state, self._entered
        back = f"ok again at >= {c.recover_above:g} for {c.recover_s:g} s"
        held_uns = _held(self._uns_low, now, c.unsafe_hold_s)
        if self._state != "unsafe" and (held_uns or self._uns_now):
            want = "unsafe"
            why = f"< {c.unsafe_below:g} for {c.unsafe_hold_s:g} s" if held_uns else f"< {c.unsafe_now_below:g}"
        elif self._state == "unsafe":
            if _held(self._uns_high, now, c.recover_s):
                want, why = ("ok", "") if _held(self._deg_high, now, c.recover_s) else \
                    ("degraded", f"after UNSAFE; {back}")
        elif self._state == "ok":
            if _held(self._deg_low, now, c.degraded_hold_s):
                want, why = "degraded", f"< {c.degraded_below:g} for {c.degraded_hold_s:g} s; {back}"
        elif _held(self._deg_high, now, c.recover_s):                 # degraded
            want = "ok"
        if want != self._state:
            self.transitions.append((now, self._state, want))
            self._state, self._since = want, now
            self._entered = "" if want == "ok" else why

    # ------------------------------------------------------------------ read
    def _snapshot(self, now: float) -> SimHealth:
        c = self.cfg
        age = None if self._t_rtf is None else now - self._t_rtf
        if age is None or age > c.stale_s:
            return SimHealth("ok", None, "no RTF sample" if age is None else f"no RTF for {age:.1f} s",
                             self.source, 0.0, age)
        self._step(now)                              # a dwell completes between samples too
        rtf, r3, r5 = self._rtf, self._r3, self._r5

        def f(v):
            return "?" if v is None else f"{v:.2f}"
        if self._state == "unsafe":
            detail = f"sim below real time: UNSAFE, rtf_3s {f(r3)} ({self._entered})"
        elif self._state == "degraded":
            detail = f"sim below real time: DEGRADED, rtf_5s {f(r5)} ({self._entered})"
        else:
            detail = f"rtf {f(rtf if rtf is not None else r5)}"
        return SimHealth(self._state, rtf, detail, self.source, now - self._since, age, r3, r5)

    def state(self) -> SimHealth:
        with self._lock:
            return self._snapshot(self.clock())


def fixed(state: str = "ok", rtf: float | None = None, source: str = "lite", detail: str = "") -> SimHealth:
    """A constant health (lite: no real-time constraint; tests: a fixture)."""
    return SimHealth(state, rtf, detail or ("no real-time constraint" if source == "lite" else state), source)


__all__ = ["STATES", "SimHealth", "SimHealthConfig", "RtfMonitor", "fixed"]
