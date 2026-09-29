"""One clock for the whole session.

Time is measured in seconds from the start of the session. Everything that waits
-- the robot, the runtime, the server -- waits through this clock, never through
``time.sleep`` or a bare ``asyncio.sleep``, so timeouts agree. The playground runs
at speed 1 because the models answer in wall time.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, TypeVar

T = TypeVar("T")


class SimClock:
    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self._t0 = time.monotonic()

    def now(self) -> float:
        """Sim seconds since the scenario started."""
        return (time.monotonic() - self._t0) * self.speed

    async def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` of sim time (yields even for 0)."""
        if seconds > 0:
            await asyncio.sleep(seconds / self.speed)
        else:
            await asyncio.sleep(0)

    async def sleep_until(self, t: float) -> None:
        await self.sleep(t - self.now())

    async def wait_for(self, aw: Awaitable[T], timeout: float) -> T:
        """``asyncio.wait_for`` with a timeout in sim seconds.

        Built on ``asyncio.timeout``: on Python 3.11, ``asyncio.wait_for`` loses a
        cancellation of the caller when the awaited thing finishes in the same loop
        iteration (fixed in 3.12), which left the runtime's think loop running after
        its TaskGroup was cancelled. Raises TimeoutError (== asyncio.TimeoutError)."""
        async with asyncio.timeout(timeout / self.speed):
            return await aw

    def timeout(self, seconds: float):
        """``asyncio.timeout`` with a delay in sim seconds (Python 3.11+)."""
        return asyncio.timeout(seconds / self.speed)
