"""SpeechService (PLAN §6.5): text speech that takes 0.3 + 0.32 * words seconds (Worldline's THOR timing,
thor/robot.py:324-338). Emits `speech_started`, `speech_ended` and `speech_cut` on the event log, which System 1's
`robot_said` and the eval read. Real TTS comes later behind the same interface.

Envelope (api/services.py): played -> succeeded, data {status: "queued", utterance_id, speech: "played"};
cut -> cancelled, data {played: fraction}. The runtime's SpeechQueue does the FIFO; this service plays one line.
"""

from __future__ import annotations

from typing import Any

from api.execution import Execution, ResultHandle
from api.results import SpeakResult, finish

from .common import EventSink, start_execution

POLL_S = 0.05


class SpeechService:
    def __init__(self, clock: Any, events: EventSink, observation_id=None, *, base_s: float = 0.3,
                 per_word_s: float = 0.32):
        self.clock = clock
        self.events = events
        self.observation_id = observation_id
        self.base_s = base_s
        self.per_word_s = per_word_s
        self._playing: dict[str, ResultHandle] = {}

    def duration_s(self, text: str) -> float:
        return self.base_s + self.per_word_s * max(1, len(str(text).split()))

    async def speak(self, text: str, *, execution: Execution) -> ResultHandle:
        return self.start(text, execution)

    def start(self, text: str, execution: Execution) -> ResultHandle:
        text = str(text)

        async def work(h: ResultHandle):
            dur = self.duration_s(text)
            if self._playing:
                self.events.emit("speech_overlap", execution_id=execution.execution_id, text=text,
                                 with_executions=list(self._playing))
            self._playing[execution.execution_id] = h
            self.events.emit("speech_started", execution_id=execution.execution_id, goal=execution.execution_id,
                             text=text)
            t_end = self.clock.now() + dur
            try:
                while self.clock.now() < t_end:
                    if h.cancel_requested:
                        played = round(1.0 - (t_end - self.clock.now()) / dur, 2)
                        self.events.emit("speech_cut", execution_id=execution.execution_id, text=text, played=played)
                        return finish(execution, "cancelled",
                                      {**SpeakResult("queued", execution.execution_id, played).__dict__,
                                       "reason": h.cancel_reason or "cancelled"},
                                      t_end=round(self.clock.now(), 3), observation_id=self._obs())
                    await self.clock.sleep(min(POLL_S, max(0.0, t_end - self.clock.now())))
            finally:
                self._playing.pop(execution.execution_id, None)
            self.events.emit("speech_ended", execution_id=execution.execution_id, text=text)
            return finish(execution, "succeeded",
                          {**SpeakResult("queued", execution.execution_id).__dict__, "speech": "played"},
                          t_end=round(self.clock.now(), 3), observation_id=self._obs())

        return start_execution(execution, work, clock=self.clock, observation_id=self._obs)

    def _obs(self) -> str | None:
        return self.observation_id() if callable(self.observation_id) else None

    def cut_all(self) -> None:
        for h in list(self._playing.values()):
            h.cancel("cut")

    def busy(self) -> bool:
        return bool(self._playing)
