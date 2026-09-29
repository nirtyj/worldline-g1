"""The runtime: the harness between the brain (model) and the robot.

Layout:

  state.py    Fact, BeliefState, TaskState, ActionHandle, TraceLog
  harness.py  event loop, utterance handling, rules, executor
  skills.py   goals with timeouts, and the speech queue
  model.py    the planner: context builder and prompt for Claude

mutants.py   copies of this runtime with one safeguard removed each (the server
             starts one when its reset message names it in `agent`)

Run it: .venv-thor/bin/python ui/server.py, then open http://localhost:8765.
"""

from .harness import Runtime


def create_runtime(robot, user, brain, clock) -> Runtime:
    return Runtime(robot, user, brain, clock)
