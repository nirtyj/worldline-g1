"""llmkit/brain.py on api/tools: schemas from the contract, frozen per session, provider rules,
multi-call replies, unknown tools, argument coercion. No network: a fake client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from api.tools import SchemaContext
from api.types import PROFILES
from llmkit.brain import ModelBrain, extra_calls, provider_of, tool_schemas
from llmkit.client import AnthropicClient, GeminiClient, OpenAIClient, ToolUse
from tests.unit.fakes_rt import house_map


class FakeClient:
    def __init__(self, uses):
        self.uses = list(uses)
        self.seen = []

    async def tool_call(self, system, text, tools):
        self.seen.append((system, text, tools))
        return self.uses.pop(0)


class Brain(ModelBrain):
    def render(self, ctx):
        return "CONTEXT"


def ctx(**kw):
    sc = SchemaContext.from_map(house_map(), ["alarm_clock", "apple"], PROFILES["sonic"].slots())
    return SimpleNamespace(tools_ctx=sc, map=house_map(), profile=PROFILES["sonic"].slots(), **kw)


def run(coro):
    return asyncio.run(coro)


def test_unknown_tool_becomes_wait_and_observe():
    c = FakeClient([ToolUse("fly", {}, 10.0, text="I think I should fly")])
    call = run(Brain(c, house_map()).next_action(ctx()))
    assert call.tool == "wait_and_observe" and call.args["timeout_s"] == 10.0
    assert call.args["reason"] == "I think I should fly" and "unknown tool 'fly'" in call.reason


def test_args_are_filtered_and_coerced():
    c = FakeClient([ToolUse("navigate", {"location": "kitchen", "speed": "fast", "timeout_s": "30"}, 5.0)])
    call = run(Brain(c, house_map()).next_action(ctx()))
    assert call.tool == "navigate" and call.args == {"location": "kitchen", "timeout_s": 30.0} and call.extra == []
    c = FakeClient([ToolUse("wait_and_observe", {"timeout_s": 5, "reason": "waiting"}, 5.0, text="hmm")])
    call = run(Brain(c, house_map()).next_action(ctx()))
    assert call.reason == "waiting"


def test_multi_call_replies_keep_the_first_and_carry_the_rest():
    raw = {"content": [{"type": "text", "text": "ok"},
                       {"type": "tool_use", "name": "speak", "input": {"text": "On it."}},
                       {"type": "tool_use", "name": "navigate", "input": {"location": "kitchen"}}]}
    use = ToolUse("speak", {"text": "On it."}, 5.0, raw=raw)
    call = run(Brain(FakeClient([use]), house_map()).next_action(ctx()))
    assert call.tool == "speak" and [(x.tool, x.args) for x in call.extra] == [("navigate", {"location": "kitchen"})]
    oa = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "speak", "arguments": '{"text": "hi"}'}},
        {"function": {"name": "recall", "arguments": '{"query": "mug"}'}}]}}]}
    assert extra_calls(SimpleNamespace(raw=oa)) == [("recall", {"query": "mug"})]
    assert extra_calls(SimpleNamespace(raw={"model": "gemini"})) == []


def test_tool_schemas_are_frozen_per_session_and_provider_specific():
    b = Brain(FakeClient([]), house_map())
    b.client = AnthropicClient("claude-x", api_key="unused")
    c = ctx()
    first = b.tools(c)
    assert b.tools(c) is first                                 # cached: byte-identical every turn
    assert all(t["parameters"]["additionalProperties"] is False for t in first)
    assert provider_of(GeminiClient("g", api_key="x")) == "gemini"
    assert provider_of(OpenAIClient("m", base_url="http://x")) == "openai" and provider_of(None) == "neutral"
    g = tool_schemas(c, "gemini")
    assert all("additionalProperties" not in t["parameters"] for t in g)


def test_tool_schemas_without_a_session_context_derive_from_the_map():
    names = [t["name"] for t in tool_schemas(SimpleNamespace(map={"keypoints": {"a": {}}}, belief={}))]
    assert names[:3] == ["speak", "list_locations", "navigate"] and "manipulate" in names


def test_reference_brain_system_uses_profile_slots():
    from agent.model import ReferenceBrain
    b = ReferenceBrain(FakeClient([]), house_map())
    lite = b.system_for(SimpleNamespace(profile=PROFILES["lite"].slots()))
    sonic = b.system_for(SimpleNamespace(profile=PROFILES["sonic"].slots()))
    assert "walks about 0.5 m/s" in lite or "walks about 0.4 m/s" in lite
    assert "walks about 0.4 m/s" in sonic and "a pick takes about 12 s" in sonic
    assert b.system_for(SimpleNamespace(profile={})) == b.system


def test_classify_still_forces_one_kind():
    c = FakeClient([ToolUse("classify", {"kind": "Correction"}, 3.0)])
    kind = run(Brain(c, house_map()).classify(SimpleNamespace(text="no, the apple"), ctx()))
    assert kind == "correction"
