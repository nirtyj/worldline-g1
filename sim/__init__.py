"""The pieces every part of the playground shares.

Modules:
  clock     one clock for everything; every wait goes through it
  log       the ground-truth event log (the page reads it; the runtime never does)
  goals     Goal, GoalResult, GoalRejected: how skills are started, cancelled, awaited

The robot and its world live in thor/.
"""

from .clock import SimClock
from .goals import Goal, GoalRejected, GoalResult
from .log import EventLog

__all__ = ["SimClock", "EventLog", "Goal", "GoalRejected", "GoalResult"]
