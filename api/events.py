"""Context events (PLAN 5.9; doc 43).

Every runtime trace row is also an ``InteractionEvent``. The doc's event types
are used where they exist; Worldline's own types sit beside them. Old trace row
names stay valid as aliases (``ALIASES``), so episodes, procedures and the eval's
``rows()`` keep reading ``row["type"]`` unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

DOC_EVENT_TYPES: tuple[str, ...] = (
    "user_utterance", "model_reasoning", "tool_call", "tool_started", "tool_result",
    "visual_observation", "speech_queued", "interruption",
)

WL_EVENT_TYPES: tuple[str, ...] = (
    "rejection", "stale_decision", "stale_result", "late_result",
    "speech_started", "speech_ended", "speech_cut", "speech_dropped",
    "system1_label", "system1_observation", "goal_check", "reconcile_started", "reconcile_done",
    "persona_goal", "persona_goal_end", "place_learned", "delivered", "memory_saved", "note_saved",
    "recall", "tool_state", "body_mode", "safety_event", "capability_changed", "narrate",
    "classified", "session_start",
)

EVENT_TYPES: tuple[str, ...] = DOC_EVENT_TYPES + WL_EVENT_TYPES

# Old trace row type -> event type. Row names not listed map to themselves.
ALIASES: dict[str, str] = {
    "heard": "user_utterance",
    "decision": "tool_call",
    "started": "tool_started",
    "result": "tool_result",
    "rejected": "rejection",
    "say_queued": "speech_queued",
    "safety_ack_queued": "speech_queued",
    "observation": "system1_observation",
    "observation_dropped": "system1_observation",
    "stop": "interruption",
    "correction": "interruption",
    "reconcile_start": "reconcile_started",
    "reconcile_done": "reconcile_done",
    "look": "visual_observation",
}

# Priorities for the fused-state deque (agent/fused_state.py; PLAN 4.4).
PRIORITY: dict[str, int] = {"safety_event": 0, "emergency_stop": 0, "task_corrected": 1, "reconciled": 1,
                            "capability_changed": 2, "utterance": 2, "directive": 2, "safety_ack": 2,
                            "body_mode": 3, "behavior_started": 3, "behavior_result": 3, "own_goal": 4}


def event_type_for(row_type: str) -> str:
    return ALIASES.get(row_type, row_type)


@dataclass(frozen=True)
class InteractionEvent:
    seq: int
    event_type: str
    timestamp: float
    generation: int
    control_epoch: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "event_type": self.event_type, "timestamp": self.timestamp,
                "generation": self.generation, "control_epoch": self.control_epoch,
                "payload": dict(self.payload)}


class EventSeq:
    """Monotonic sequence numbers for one session."""

    def __init__(self) -> None:
        self._seq = itertools.count(1)

    def make(self, event_type: str, timestamp: float, generation: int, control_epoch: int,
             payload: dict[str, Any] | None = None) -> InteractionEvent:
        return InteractionEvent(next(self._seq), event_type, round(float(timestamp), 3), int(generation),
                                int(control_epoch), dict(payload or {}))


__all__ = ["DOC_EVENT_TYPES", "WL_EVENT_TYPES", "EVENT_TYPES", "ALIASES", "PRIORITY", "event_type_for",
           "InteractionEvent", "EventSeq"]
