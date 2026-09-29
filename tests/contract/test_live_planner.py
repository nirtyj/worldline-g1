"""Opt-in live check (``-m live``, needs a key): one real planner call on the new tool schemas.
Deselected by default (pyproject addopts); unit and contract tests never call a model.

    .venv-rt/bin/python -m pytest -m live tests/contract/test_live_planner.py
    LIVE_MODEL=claude-sonnet-5 .venv-rt/bin/python -m pytest -m live tests/contract/test_live_planner.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.live


async def test_one_live_planner_decision_uses_the_new_tools():
    from agent.harness import Runtime
    from agent.model import ReferenceBrain
    from api.tools import TOOL_SPECS, schema_issue
    from llmkit import client_from_options
    from sim.clock import SimClock
    from tests.kept.system1_live_check import load_env
    from tests.unit.fakes_rt import FakeRobot, FakeUser, ScriptBrain

    load_env()
    model = os.environ.get("LIVE_MODEL") or ("gemini-3.8-flash" if os.environ.get("GEMINI_API_KEY") else
                                             "claude-sonnet-5" if os.environ.get("ANTHROPIC_API_KEY") else None)
    if model is None:
        pytest.skip("no GEMINI_API_KEY or ANTHROPIC_API_KEY")
    clock = SimClock(1.0)
    robot = FakeRobot(clock)
    rt = Runtime(robot, FakeUser(clock), ScriptBrain([]), clock)
    rt.task.utterances.append(SimpleNamespace(id="u1", text="Bring me the alarm clock.", t_end=0.0, directive=None))
    rt.task.kinds["u1"] = "request"
    rt.task.intent_version = 1
    ctx = rt._ctx()
    brain = ReferenceBrain(client_from_options({"model": model}, model), rt.map)
    call = await brain.next_action(ctx)
    assert call.tool in TOOL_SPECS, call
    assert schema_issue(call.tool, call.args) is None, (call.tool, call.args)
