"""Synthetic RTF traces for the sim-health tests (world/sim_health.py, robot/health.py): a manual monotonic clock and
a sim.health trace shaped like the 2026-09-29 stance-fix live runs (outputs/stance_fix/: rtf_5s 0.92-0.96)."""

from __future__ import annotations

import math
import random


class Clock:
    """Monotonic seconds the test advances by hand."""

    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def live_trace(seconds: float, lo: float, hi: float, seed: int = 7) -> list[tuple[float, float, float]]:
    """(rtf_1s, rtf_3s, rtf_5s) once a second: rtf_5s wanders in [lo, hi] (a slow sine plus noise, crossing the
    middle every few seconds), rtf_3s a little wider, rtf_1s wider still."""
    rnd = random.Random(seed)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    out = []
    for i in range(int(seconds)):
        r5 = mid + amp * (0.7 * math.sin(i / 3.0) + 0.3 * rnd.uniform(-1, 1))
        r3 = min(1.02, max(0.5, r5 + rnd.uniform(-0.03, 0.03)))
        r1 = min(1.02, max(0.5, r5 + rnd.uniform(-0.08, 0.06)))
        out.append((r1, r3, min(hi, max(lo, r5))))
    return out
