"""A stand-in System 1 with fixed answers and no model, for testing the wiring.

    .venv-rt/bin/python -m ui.server --system1 tests.kept.system1_stub:create   (or SYSTEM1=tests.kept.system1_stub:create)

It labels "stop …" and "bring me the …" messages, leaves every other message to
the planner (route returns None), and reports one fixed observation once it has
been sent camera frames. It keeps what it was sent, so a test can check that the
fused state, frames and the robot's lines really reach System 1.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

OBSERVATION = "(test) the lights are on in this room"


class StubSystemOne:
    def __init__(self, on_status: Callable[[str, str], None]) -> None:
        self.status = "connecting"
        self._on_status = on_status
        self.contexts: list[dict[str, Any]] = []
        self.frames = 0
        self.robot_lines: list[str] = []

    async def run(self) -> None:
        self.status = "ready"
        self._on_status("ready", "test stand-in: fixed answers")
        while True:
            await asyncio.sleep(3600)

    async def route(self, text: str) -> dict[str, Any] | None:
        t = text.lower().strip()
        if t.startswith("stop"):
            return {"kind": "stop", "target": "", "replaces_task": False, "says_yes": None, "confidence": 0.99}
        if t.startswith("bring me the "):
            return {"kind": "request", "target": t[len("bring me the "):].rstrip(".!? "), "replaces_task": False,
                    "says_yes": None, "confidence": 0.9}
        return None

    async def update(self, context: dict[str, Any]) -> None:
        self.contexts.append(context)

    async def frame(self, jpeg: bytes, where: str | None) -> None:
        self.frames += 1

    async def robot_said(self, text: str) -> None:
        self.robot_lines.append(text)

    async def observe(self) -> list[dict[str, Any]]:
        if not self.frames:
            return []
        return [{"what": OBSERVATION, "where": None, "confidence": 0.6}]


def create(on_status: Callable[[str, str], None]) -> StubSystemOne:
    return StubSystemOne(on_status)
