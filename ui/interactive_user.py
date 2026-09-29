"""A user fed by a chat box instead of a script.

The runtime only ever calls ``await user.next()``. Each typed line is delivered at once
(a typed message has no speaking time) and recorded in the event log the same
way every utterance is recorded, so the page's feed sees it.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    t_start: float     # when the user started speaking
    t_end: float       # when the sentence was delivered to the runtime
    directive: dict[str, Any] | None = None  # structured System-1 metadata, when available


class InteractiveUser:
    def __init__(self, clock: Any, log: Any) -> None:
        self.clock = clock
        self.log = log
        self.delivered: dict[str, Utterance] = {}
        self._queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self._ids = itertools.count(1)

    # Runtime-facing API ------------------------------------------------
    async def next(self) -> Utterance:
        return await self._queue.get()

    # Chat-facing API ---------------------------------------------------
    def say(self, text: str, directive: dict[str, Any] | None = None) -> Utterance:
        text = text.strip()
        if not text:
            raise ValueError("empty message")
        now = round(self.clock.now(), 3)
        utt = Utterance(f"u{next(self._ids)}", text, now, now,
                        dict(directive) if directive is not None else None)
        self.log.emit("utterance_started", id=utt.id, text=text, t_start=now)
        self.delivered[utt.id] = utt
        self.log.emit("utterance", id=utt.id, text=text, t_start=now, kind=None)
        self._queue.put_nowait(utt)
        return utt
