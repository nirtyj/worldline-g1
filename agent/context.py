"""ContextBus: the runtime's trace rows as context events (PLAN 5.9; doc 43).

``log(type, **fields)`` keeps Worldline's trace row (``{"t", "type", ...}``) for every
existing sink (episodes, narrator, UI, procedures, eval) and also emits an
``api.events.InteractionEvent`` whose ``event_type`` is the doc name where one
exists (``ALIASES``: heard -> user_utterance, decision -> tool_call, started ->
tool_started, result -> tool_result, rejected -> rejection, ...).

Compaction (doc 44) is windowing for now (30 conversation rows, 18 actions, 12 notes,
8 observations in agent/model.py); agent/compactor.py lands in M7 and is installed
only between think iterations.
"""

from __future__ import annotations

from typing import Any, Callable

from api.events import EventSeq, InteractionEvent, event_type_for

from .state import TraceLog

EVENT_LIMIT = 4000


class ContextBus(TraceLog):
    def __init__(self, clock: Any, fence: Callable[[], tuple[int, int]] | None = None) -> None:
        super().__init__(clock)
        self._fence = fence or (lambda: (0, 0))
        self._seq = EventSeq()
        self.events: list[InteractionEvent] = []
        self.event_sinks: list[Callable[[InteractionEvent], None]] = []

    def log(self, type: str, **fields: Any) -> None:
        super().log(type, **fields)
        row = self.rows[-1]
        gen, epoch = self._fence()
        ev = self._seq.make(event_type_for(type), row["t"], gen, epoch, {"row_type": type, **fields})
        self.events.append(ev)
        if len(self.events) > EVENT_LIMIT:
            del self.events[: len(self.events) - EVENT_LIMIT]
        for sink in list(self.event_sinks):
            try:
                sink(ev)
            except Exception:              # a display sink never breaks the runtime
                pass

    def recent_events(self, n: int = 50, types: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        evs = [e for e in self.events if types is None or e.event_type in types]
        return [e.to_dict() for e in evs[-n:]]


__all__ = ["ContextBus", "EVENT_LIMIT"]
