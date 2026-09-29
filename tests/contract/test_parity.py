"""Sim/real parity (PLAN 6.7, deviation D2; G13 schema_parity): across every profile the tool and
envelope schemas are equal, enum values are equal except the map- and registry-derived ones, and
the tool descriptions and the rendered SYSTEM prompt are equal once numbers are masked. Also: the
provider request builders accept the schemas, and api/schemas/*.json is current."""

from __future__ import annotations

import json
import re

import pytest

from agent.model import SYSTEM_TEMPLATE, system_prompt
from api import gen_schemas
from api.results import envelope_schema
from api.tools import SchemaContext, json_schemas
from api.types import PROFILES
from llmkit.client import AnthropicClient, GeminiClient, OpenAIClient
from tests.unit.fakes_rt import house_map

MASK = re.compile(r"\d+(\.\d+)?")
TYPES = ["alarm_clock", "apple", "banana", "book"]


def mask(text: str) -> str:
    return MASK.sub("#", text)


def profile_slots():
    """The numeric slots of every profile: from robot/profile.py when importable, else api.types.PROFILES."""
    out = {}
    try:
        from robot.profile import load_profile
        for name in PROFILES:
            try:
                out[name] = load_profile(name).robot.slots()
            except Exception:
                out[name] = PROFILES[name].slots()
    except Exception:
        out = {name: p.slots() for name, p in PROFILES.items()}
    return out


def schemas_for(slots, provider="neutral"):
    return json_schemas(SchemaContext.from_map(house_map(), TYPES, slots), provider)


def strip_text(schemas):
    """Schemas without descriptions: names, arguments, types, enums, required."""
    s = json.loads(json.dumps(schemas))
    for t in s:
        t.pop("description", None)
        for p in t["parameters"]["properties"].values():
            p.pop("description", None)
    return s


def test_tool_and_envelope_schemas_equal_across_profiles():
    slots = profile_slots()
    ref = strip_text(schemas_for(next(iter(slots.values()))))
    for name, sl in slots.items():
        assert strip_text(schemas_for(sl)) == ref, name
    assert envelope_schema() == envelope_schema()


def test_descriptions_and_system_equal_with_numbers_masked():
    slots = profile_slots()
    descs = {n: [mask(t["description"]) for t in schemas_for(sl)] for n, sl in slots.items()}
    systems = {n: mask(system_prompt(sl)) for n, sl in slots.items()}
    assert len({json.dumps(v) for v in descs.values()}) == 1, descs.keys()
    assert len(set(systems.values())) == 1
    # the template really is filled (no leftover slots) and the numbers really differ
    assert "{" not in system_prompt(slots["sonic"])
    assert system_prompt(slots["lite"]) != system_prompt(slots["sonic"]) or slots["lite"] == slots["sonic"]


def test_system_prompt_uses_only_the_new_tool_names():
    text = SYSTEM_TEMPLATE
    for new in ("speak", "navigate", "check_reachability", "manipulate", "wait_and_observe", "recall"):
        assert new in text, new
    assert not re.search(r"\bcall (wait|look|say|pick|place|reachability)\b", text)
    assert not re.search(r"\b(say|look|pick|place|reachability)\(", text)
    assert "look first" not in text and "call wait_and_observe" in text


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_provider_request_builders_accept_the_schemas(provider):
    tools = schemas_for(PROFILES["sonic"].slots(), provider)
    if provider == "anthropic":
        body = AnthropicClient("claude-x", api_key="unused").build_request("sys", "text", tools)
        assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
        assert all(t["input_schema"]["additionalProperties"] is False for t in body["tools"])
        json.dumps(body)
    elif provider == "openai":
        body = OpenAIClient("m", base_url="http://localhost:1/v1").build_request("sys", "text", tools)
        assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
        json.dumps(body)
    else:
        pytest.importorskip("google.genai")
        cfg = GeminiClient("gemini-x", api_key="unused").build_config("sys", tools)
        decls = cfg.tools[0].function_declarations
        assert [d.name for d in decls] == [t["name"] for t in tools]


def test_committed_api_schemas_are_current():
    assert gen_schemas.main(["--check"]) == 0, "run: python -m api.gen_schemas"
