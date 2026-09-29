"""Ground-truth event log.

ThorRobot and InteractiveUser write every event here; ui/server.py streams it to
the page. The runtime must NOT read it: it sees goal results and
``ThorRobot.events()`` only.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .clock import SimClock


def match_event(event: dict[str, Any], filt: dict[str, Any]) -> bool:
    """True if ``event`` matches every key of ``filt``.

    Keys match exactly, except: ``text_re`` is a case-insensitive regex search
    on the event's text; dotted keys (``args.to``) look inside nested dicts; a
    list value matches any of its members.
    """
    for key, want in filt.items():
        if key == "text_re":
            if not re.search(want, str(event.get("text", "")), re.I):
                return False
            continue
        value: Any = event
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(want, list):
            if value not in want:
                return False
        elif isinstance(value, list):
            if want not in value:          # a list-valued field matches if it contains the value
                return False
        elif value != want:
            return False
    return True


class EventLog:
    def __init__(self, clock: SimClock) -> None:
        self.clock = clock
        self.events: list[dict[str, Any]] = []
        self._subscribers: list[asyncio.Queue] = []

    def emit(self, type: str, **fields: Any) -> dict[str, Any]:
        event = {"t": round(self.clock.now(), 3), "type": type, **fields}
        self.events.append(event)
        for queue in list(self._subscribers):
            queue.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def find(self, type: str | None = None, **match: Any) -> list[dict[str, Any]]:
        filt = dict(match)
        if type is not None:
            filt["type"] = type
        return [e for e in self.events if match_event(e, filt)]

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event, default=str) + "\n")
