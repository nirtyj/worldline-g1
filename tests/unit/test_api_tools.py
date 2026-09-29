"""api/tools.py: the only tool definition. Schemas per provider, enums by argument name, optional
args not required, frozen per session, numeric-slot descriptions, SCHEMA-stage checks."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from api.tools import (BODY_TOOLS, INSTANT_TOOLS, SENSE_TOOLS, TOOL_SPECS, WAIT_TOOLS, SchemaContext, ToolCall,
                       coerce_args, json_schemas, render_description, resources_of, schema_issue, tool_specs)
from api.types import PROFILES
from tests.unit.fakes_rt import house_map

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["alarm_clock", "apple", "banana", "bottle"]


def ctx(profile: str = "sonic", **kw) -> SchemaContext:
    return SchemaContext.from_map(house_map(), TYPES, PROFILES[profile].slots(), **kw)


def by_name(schemas):
    return {s["name"]: s for s in schemas}


def test_the_seven_tools_in_order():
    names = [s["name"] for s in json_schemas(ctx())]
    assert names == ["speak", "list_locations", "navigate", "check_reachability", "manipulate",
                     "wait_and_observe", "recall"]
    assert set(BODY_TOOLS) == {"navigate", "manipulate"} and SENSE_TOOLS == ("check_reachability",)
    assert set(INSTANT_TOOLS) == {"speak", "list_locations", "recall"} and WAIT_TOOLS == ("wait_and_observe",)
    assert TOOL_SPECS["recall"].extension and not TOOL_SPECS["navigate"].extension


def test_enums_are_filled_by_argument_name():
    s = by_name(json_schemas(ctx()))
    loc = s["navigate"]["parameters"]["properties"]["location"]["enum"]
    m = house_map()
    assert set(m["keypoints"]) <= set(loc)
    assert "user" in loc and loc[-1] == "reach_stance"                 # reach_stance is appended last
    assert {"kitchen", "bedroom", "living_room"} <= set(loc)            # rooms are locations
    tgt = s["manipulate"]["parameters"]["properties"]["target"]["enum"]
    assert set(tgt) == set(m["surfaces"]) | {"user"}
    for tool in ("check_reachability", "manipulate"):
        assert s[tool]["parameters"]["properties"]["object_type"]["enum"] == sorted(TYPES)
        assert "enum" not in s[tool]["parameters"]["properties"]["object_id"]   # checked against belief instead
    assert s["manipulate"]["parameters"]["properties"]["action"]["enum"] == ["pick", "place"]
    assert s["manipulate"]["parameters"]["properties"]["arm"]["enum"] == ["left", "right"]


def test_optional_args_are_not_required_and_no_null_unions():
    s = by_name(json_schemas(ctx()))
    assert s["manipulate"]["parameters"]["required"] == ["action", "object_type"]
    assert s["navigate"]["parameters"]["required"] == ["location"]
    assert s["check_reachability"]["parameters"]["required"] == ["object_type"]
    assert "required" not in s["list_locations"]["parameters"]
    assert "required" not in s["wait_and_observe"]["parameters"]
    text = json.dumps(json_schemas(ctx()))
    assert '["string", "null"]' not in text and '"null"' not in text
    wo = s["wait_and_observe"]["parameters"]["properties"]["timeout_s"]
    assert (wo["minimum"], wo["maximum"]) == (0.0, 60.0)


def test_additional_properties_only_for_anthropic():
    for provider, want in (("anthropic", True), ("gemini", False), ("openai", False), ("neutral", False)):
        for sch in json_schemas(ctx(), provider):
            assert ("additionalProperties" in sch["parameters"]) is want, (provider, sch["name"])


def test_schemas_are_frozen_per_session_byte_for_byte():
    c = ctx()
    a = json.dumps(json_schemas(c), sort_keys=True)
    b = json.dumps(json_schemas(c), sort_keys=True)
    assert a == b
    # a skill becoming unhealthy never changes the enum: it is built from loaded skills only
    assert json.dumps(json_schemas(ctx()), sort_keys=True) == a


def test_descriptions_differ_only_in_numeric_slots_across_profiles():
    mask = lambda t: re.sub(r"\d+(\.\d+)?", "#", t)                 # noqa: E731 (PLAN 6.7)
    texts = {p: [mask(s["description"]) for s in json_schemas(ctx(p))] for p in PROFILES}
    first = next(iter(texts.values()))
    assert all(v == first for v in texts.values())
    lite, sonic = (by_name(json_schemas(ctx(p)))["navigate"]["description"] for p in ("lite", "sonic"))
    assert lite != sonic                                             # the numbers really differ
    assert "2.5 s per metre" in sonic and "0.40 m" in sonic


def test_description_templates_render_missing_slots_as_question_marks():
    assert "about ? s per metre" in render_description(TOOL_SPECS["navigate"], {})


def test_look_tool_only_with_the_escape_hatch():
    assert "look" not in [s["name"] for s in json_schemas(ctx())]
    assert "look" in [s["name"] for s in json_schemas(ctx(include_look=True))]
    assert "look" in tool_specs(include_look=True) and "look" not in tool_specs()


@pytest.mark.parametrize("tool,args,code", [
    ("fly", {}, "unknown_tool"),
    ("observe", {"mode": "scan"}, "unknown_tool"),                  # internal, never planner-visible
    ("speak", {"text": "  "}, "empty_text"),
    ("recall", {}, "empty_query"),
    ("navigate", {}, "missing_arg"),
    ("manipulate", {"action": "pick"}, "missing_arg"),
    ("navigate", {"location": 3}, "bad_arg_type"),
    ("wait_and_observe", {"timeout_s": 61}, "out_of_range"),
    ("wait_and_observe", {"timeout_s": "soon"}, "bad_arg_type"),
    ("navigate", {"location": "kitchen", "timeout_s": 0.5}, "out_of_range"),
])
def test_schema_issues(tool, args, code):
    issue = schema_issue(tool, args)
    assert issue is not None and issue.code == code, issue


def test_schema_ok_and_coercion():
    assert schema_issue("wait_and_observe", {}) is None
    assert schema_issue("wait_and_observe", coerce_args("wait_and_observe", {"timeout_s": "0"})) is None
    assert coerce_args("navigate", {"location": "kitchen", "bogus": 1, "timeout_s": None}) == {"location": "kitchen"}
    assert "unknown tool 'fly'; use one of: speak, list_locations" in schema_issue("fly", {}).message


def test_resources():
    assert resources_of("navigate") == {"body"} and resources_of("manipulate") == {"body"}
    assert resources_of("check_reachability") == {"sense"} and resources_of("speak") == {"speech"}
    assert resources_of("wait_and_observe") == {"wait"} and resources_of("list_locations") == frozenset()
    assert resources_of("observe", {"mode": "scan"}) == {"sense", "body"}       # a scan moves the waist
    assert resources_of("observe", {"mode": "glance"}) == {"sense"}


def test_toolcall_extra_defaults_empty():
    assert ToolCall("speak", {"text": "hi"}).extra == []


# ----------------------------------------------------------------------
# api/ is stdlib only and imports on 3.11 and 3.12
# ----------------------------------------------------------------------
STDLIB_OK = {"__future__", "asyncio", "dataclasses", "enum", "itertools", "math", "string", "typing", "json"}


def test_api_is_stdlib_only():
    for path in sorted((ROOT / "api").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:                    # relative: inside api/
                    continue
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            for m in mods:
                assert m in STDLIB_OK or m in sys.stdlib_module_names, f"{path.name} imports {m}"
                assert m not in ("numpy", "zmq", "msgpack", "agent", "world", "robot", "body", "services"), path


@pytest.mark.parametrize("version", ["3.11", "3.12"])
def test_api_imports_on(version):
    exe = shutil.which(f"python{version}")
    if exe is None:
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip(f"no python{version}")
        found = subprocess.run([uv, "python", "find", version], capture_output=True, text=True)
        if found.returncode != 0:
            pytest.skip(f"no python{version}")
        exe = found.stdout.strip()
    code = ("import api, api.types, api.tools, api.reasons, api.execution, api.results, api.summaries, "
            "api.observation, api.events, api.state_machine, api.skills, api.services, api.gen_schemas; "
            "from api.tools import json_schemas, SchemaContext; print(len(json_schemas(SchemaContext())))")
    out = subprocess.run([exe, "-E", "-s", "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "7"
