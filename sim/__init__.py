"""The pieces every part of the runtime shares.

Modules:
  clock     one clock for everything; every wait goes through it
  log       the ground-truth event log (the page reads it; the runtime never does)

Execution objects (the old ``goals``) are api/execution.py: Execution,
ExecutionHandle, ResultHandle, Rejected. The robot and its world live in
robot/, services/ and world/.
"""

from .clock import SimClock
from .log import EventLog

__all__ = ["SimClock", "EventLog"]
