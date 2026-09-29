"""eval/suite.py against a scripted fake page (no server, no simulator, no LLM): scenarios load a house,
assert fixtures on truth, score on truth, record executors, and label passes that rest on sim shortcuts."""

from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from eval import scenes as sc
from eval import suite

ROOT = Path(__file__).resolve().parents[2]

H40_LAYOUT = {"user_surface": "kitchen_counter_1a", "things": ["alarm_clock", "book", "mug", "dresser", "bed"],
              "surfaces": {"kitchen_counter_1a": {"x": 7.27, "z": 5.99}, "bedroom_dresser_1b": {"x": 2.46, "z": 0.34}},
              "landmarks": {}}
H40_OBJECTS = {"alarm_clock_1": {"type": "alarm_clock", "where": "bedroom_dresser_1b"},
               "book_1": {"type": "book", "where": "bedroom_bed_1a"}, "book_2": {"type": "book", "where": "bedroom_bed_1c"},
               "mug_1": {"type": "mug", "where": "bedroom_dresser_1a"}, "mug_2": {"type": "mug", "where": "kitchen_counter_1b"}}
K10_LAYOUT = {"user_surface": "sink_basin_1", "things": ["spatula", "stove_burner", "counter_top"],
              "surfaces": {"counter_2a": {"x": 0.95, "z": -1.92}, "counter_2b": {"x": 0.95, "z": -0.95},
                           "counter_2c": {"x": 0.95, "z": 0.02}, "sink_basin_1": {"x": -0.7, "z": -0.65}},
              "landmarks": {"stove_1": {"type": "stove_burner", "label": "stove", "near": "counter_2b", "x": 0.91, "z": -1.30}}}


class FakePage:
    """Answers the suite's websocket messages like ui/server.py would, with a script per utterance."""

    def __init__(self, layout: dict, objects: dict, on_say: Callable[["FakePage", str], Any] | None = None) -> None:
        self.layout, self.objects0, self.on_say = layout, objects, on_say
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.objects: dict = {}
        self.t = 0.0
        self.scene = ""
        self.profile = ""
        self.ids = 0
        self.execs: list[dict] = []
        self.sent: list[dict] = []

    # the websocket the suite holds
    async def send(self, raw: str) -> None:
        m = json.loads(raw)
        self.sent.append(m)
        if m["type"] == "reset":
            self.scene, self.profile = m["scene"], m.get("profile") or "lite"
            self.objects = json.loads(json.dumps(self.objects0))
            self.execs = []
            await self.inbox.put(json.dumps({"type": "init", "config": {"scene": self.scene, "profile": self.profile},
                                             "layout": self.layout, "map": {"keypoints": {}},
                                             "truth": {"objects": self.objects}, "runtime": {"active": []}}))
        elif m["type"] == "say" and self.on_say:
            asyncio.get_running_loop().create_task(self.on_say(self, m["text"]))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self.inbox.get()

    # helpers for scripts
    async def frame(self, trace=(), events=(), calls=(), active=()) -> None:
        self.t += 0.5
        await self.inbox.put(json.dumps({"type": "frame", "t": self.t, "trace": list(trace), "events": list(events),
                                         "calls": list(calls), "truth": {"objects": self.objects},
                                         "runtime": {"active": list(active), "executions": list(self.execs)}}))

    async def run_tool(self, tool: str, executor: str, action: str | None = None, skill: str | None = None) -> None:
        self.ids += 1
        eid = f"{tool[:3]}-{self.ids:06d}"
        e = {"id": eid, "tool": tool, "action": action, "status": "running", "executor": executor, "skill": skill}
        self.execs.append(e)
        await self.frame(trace=[{"t": self.t, "type": "started", "tool": tool, "action": action, "execution_id": eid,
                                 "args": {"action": action} if action else {}}], active=[e])
        e["status"] = "succeeded"
        await self.frame(trace=[{"t": self.t, "type": "result", "tool": tool, "skill": tool, "action": action,
                                 "execution_id": eid, "status": "succeeded", "executor": executor,
                                 "data": {"executor": executor, **({"skill": skill} if skill else {})}}])


def deliver(nav: str, manip: str, skill: str, moves: dict[str, str]):
    async def on_say(page: FakePage, text: str) -> None:
        await page.frame(trace=[{"t": page.t, "type": "classified", "kind": "request", "directive": {"text": text}}])
        await page.run_tool("navigate", nav, "keypoint")
        await page.run_tool("manipulate", manip, "pick", skill)
        await page.run_tool("navigate", nav, "keypoint")
        await page.run_tool("manipulate", manip, "place", skill)
        for oid, where in moves.items():
            if oid.rsplit("_", 1)[0].replace("_", " ") in text.lower():     # "alarm_clock_1" <- "alarm clock"
                page.objects[oid]["where"] = where
        await page.frame(events=[{"t": page.t, "type": "speech_started", "text": "Here you go."}])
    return on_say


async def _play(page: FakePage, fn, profile: str, scale: float | None = 0.01, original: bool = False):
    run = suite.Run(page, profile, scale, original)
    reader = asyncio.create_task(run.reader())
    try:
        passed, note = await asyncio.wait_for(fn(run), 30)
    except suite.FixtureError as e:
        passed, note = False, f"fixture: {e}"
    reader.cancel()
    return suite.score(fn.__name__, passed, note, 1.0, run), run


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(suite, "LOAD_TIMEOUT_S", 5.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(suite.asyncio, "sleep", lambda s, *a: real_sleep(min(s, 0.05)))


def test_fetch_on_sonic_is_a_fallback_pass_labelled_attach_grasp():
    page = FakePage(H40_LAYOUT, H40_OBJECTS, deliver("sonic_walk", "sonic_arm_script", "sonic.script.pick.v0",
                                                   {"alarm_clock_1": "kitchen_counter_1a"}))
    res, run = asyncio.run(_play(page, suite.fetch_other_room, "sonic", scale=None))
    assert res["passed"] and res["fallback_pass"] and not res["target_pass"], res
    assert res["executors_used"]["nav"] == {"sonic_walk": 2} and res["executors_used"]["manip"] == {"sonic_arm_script": 2}
    assert res["executors_used"]["skills"] == {"sonic.script.pick.v0": 2}
    assert res["shortcuts"] == ["sonic_arm_script"] and "attach grasp" in res["honesty"][0]
    assert res["fixtures_ok"] and res["profile"] == "sonic" and res["scene"] == "procthor-train-40"
    assert res["time_limit_s"] >= 400 and res["time_limit"]["estimate_s"] > 60
    reset = next(m for m in page.sent if m["type"] == "reset")
    assert reset == {"type": "reset", "scene": "procthor-train-40", "forget": True, "profile": "sonic"}
    assert run.started("manipulate", action="pick") and run.started("manipulate", action="place")
    assert not run.started("manipulate", action="wave")
    assert "PASS*" in suite.line(res) and "attach grasp" in suite.line(res)


def test_fetch_on_full_with_groot_is_a_target_pass():
    page = FakePage(H40_LAYOUT, H40_OBJECTS, deliver("sonic_walk", "groot_sonic", "groot.pick.alarm_clock.v1",
                                                   {"alarm_clock_1": "kitchen_counter_1a"}))
    res, _ = asyncio.run(_play(page, suite.fetch_other_room, "full"))
    assert res["passed"] and res["target_pass"] and not res["fallback_pass"] and res["shortcuts"] == []


def test_bringup_kinematic_base_is_labelled_even_with_groot_style_manip():
    page = FakePage(H40_LAYOUT, H40_OBJECTS, deliver("kinematic_nav", "kinematic_attach", "bringup.attach.pick",
                                                   {"alarm_clock_1": "kitchen_counter_1a"}))
    res, _ = asyncio.run(_play(page, suite.fetch_other_room, "bringup"))
    assert res["fallback_pass"] and res["shortcuts"] == ["kinematic_attach", "kinematic_nav"]
    assert any("kinematic base" in h for h in res["honesty"])


def test_undelivered_object_fails_and_is_neither_target_nor_fallback(monkeypatch):
    monkeypatch.setattr(suite.Run, "limit", lambda self, name=None: 1.5)
    page = FakePage(H40_LAYOUT, H40_OBJECTS, deliver("sonic_walk", "sonic_arm_script", "s", {}))
    res, _ = asyncio.run(_play(page, suite.fetch_other_room, "sonic", scale=0.001))
    assert not res["passed"] and not res["fallback_pass"] and not res["target_pass"]
    assert "alarm_clock_1 ended on bedroom_dresser_1b" in res["note"]


def test_missing_fixture_object_fails_the_scenario_before_anything_is_said():
    objs = {k: v for k, v in H40_OBJECTS.items() if k != "alarm_clock_1"}
    page = FakePage(H40_LAYOUT, objs, deliver("sonic_walk", "sonic_arm_script", "s", {}))
    res, _ = asyncio.run(_play(page, suite.fetch_other_room, "sonic"))
    assert not res["passed"] and res["note"].startswith("fixture:") and res["fixtures_ok"] is False
    assert not any(m["type"] == "say" for m in page.sent)


def test_addition_runs_the_substitute_by_default_and_the_original_on_request(monkeypatch):
    async def on_say(page: FakePage, text: str) -> None:
        await page.run_tool("navigate", "sonic_walk", "keypoint")
        if "alarm clock" in text:
            page.objects["alarm_clock_1"]["where"] = "kitchen_counter_1a"
        if "mug" in text:
            page.objects["mug_1"]["where"] = "kitchen_counter_1a"
        await page.frame()
    res, _ = asyncio.run(_play(FakePage(H40_LAYOUT, H40_OBJECTS, on_say), suite.addition, "lite"))
    assert res["passed"] and res["binding"] == "substitute" and res["binding_status"] == "substituted", res
    page = FakePage(H40_LAYOUT, H40_OBJECTS, on_say)
    monkeypatch.setattr(suite.Run, "limit", lambda self, name=None: 1.5)
    res, _ = asyncio.run(_play(page, suite.addition, "lite", original=True))
    assert not res["passed"] and res["binding"] == "original"
    assert any(m.get("text") == "Also bring me a book." for m in page.sent)


def test_other_side_resolves_from_the_live_map_and_checks_the_golden_line():
    async def on_say(page: FakePage, text: str) -> None:
        call = {"n": 1, "via": "model", "input": "LAYOUT\ncounter_2c · counter_2b · stove_1 · counter_2a\n"
                                                 "stove_1 (stove) is between counter_2b and counter_2a"}
        await page.frame(calls=[call])
        await page.run_tool("manipulate", "sonic_arm_script", "pick", "sonic.script.pick.v0")
        page.objects["spatula_1"]["where"] = "counter_2a"
        await page.frame(trace=[{"t": page.t, "type": "goal_check", "goal": "other side of stove_1 from counter_2b",
                                 "ok": True}])
    page = FakePage(K10_LAYOUT, {"spatula_1": {"type": "spatula", "where": "counter_2b"}}, on_say)
    res, _ = asyncio.run(_play(page, suite.other_side, "sonic"))
    assert res["passed"], res
    assert "LAYOUT had 'stove_1 (stove) is between counter_2b and counter_2a': True" in res["note"]
    assert res["fallback_pass"] and res["house"] == "K10" and res["binding_status"] == "flagged"


def test_summary_counts_target_and_fallback_passes_per_executor():
    rows = [{"passed": True, "target_pass": True, "fallback_pass": False, "decisions": 3, "tokens_in": 10,
             "executors_used": {"nav": {"sonic_walk": 2}, "manip": {"groot_sonic": 2}}},
            {"passed": True, "target_pass": False, "fallback_pass": True, "decisions": 4, "tokens_in": 5,
             "executors_used": {"nav": {"sonic_walk": 2}, "manip": {"sonic_arm_script": 2}}},
            {"passed": False, "target_pass": False, "fallback_pass": False, "decisions": 1, "tokens_in": 0,
             "executors_used": {"nav": {}, "manip": {}}}]
    s = suite.summarize(rows, "full", "t")
    assert (s["passed"], s["target_passes"], s["fallback_passes"], s["total"]) == (2, 1, 1, 3)
    assert s["by_executor"]["sonic_walk"] == {"scenarios": 2, "passed": 2}
    assert s["by_executor"]["sonic_arm_script"] == {"scenarios": 1, "passed": 1}


def test_executors_from_frames_fill_in_missing_result_rows():
    frames = [{"runtime": {"executions": [{"id": "man-1", "tool": "manipulate", "status": "succeeded",
                                           "executor": "kinematic_attach", "skill": "bringup.attach.pick"},
                                          {"id": "nav-2", "tool": "navigate", "status": "running", "executor": "kinematic_nav"}]}}]
    used = sc.executors_used([], frames)
    assert used == {"nav": {}, "manip": {"kinematic_attach": 1}, "skills": {"bringup.attach.pick": 1}}
    assert sc.honesty({"nav": {}, "manip": {}}, grasp=True, profile="sonic")["shortcuts"] == ["sonic_arm_script"]
    assert sc.honesty({"nav": {}, "manip": {}}, grasp=True, profile="full")["shortcuts"] == []


# ---------------------------------------------------------------------------------------------- confinement
BANNED_EVAL = {"sim_isaac", "isaac_host", "scenes", "body", "viz", "zmq", "world", "robot", "services", "nav2", "sonic"}
BANNED_UI = {"sim_isaac", "isaac_host", "scenes", "body", "nav2", "sonic", "zmq"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def test_the_referee_reads_ground_truth_only_through_the_page_and_world():
    """eval/ never talks to the simulator, the body or the world model directly: it reads frame.truth,
    which ui/server.py builds from world.truth(). ui/ reads GT only through world/ (and frames via viz.tap)."""
    for p in sorted((ROOT / "eval").glob("*.py")):
        bad = _imports(p) & BANNED_EVAL
        assert not bad, f"{p.name} imports {bad}"
    for p in sorted((ROOT / "ui").glob("*.py")):
        bad = _imports(p) & BANNED_UI
        assert not bad, f"{p.name} imports {bad}"
        if "viz" in _imports(p):
            assert p.name == "cameras.py", f"{p.name}: only ui/cameras.py may use viz (FrameTap)"
    src = (ROOT / "ui" / "server.py").read_text()
    assert "tap.pose(" not in src and "gt_pub" not in src, "the robot pose on the page comes from world.truth()"


def test_only_target_executors_make_a_target_pass():
    assert sc.honesty({"nav": {"sonic_walk": 1}, "manip": {"groot_sonic": 1}})["fallback"] is False
    lite = sc.honesty({"nav": {"lite": 2}, "manip": {"lite": 2}}, grasp=True, profile="lite")
    assert lite["fallback"] and lite["shortcuts"] == ["lite"] and "no physics" in lite["labels"][0]
    odd = sc.honesty({"nav": {"sonic_walk": 1}, "manip": {"some_new_executor": 1}})
    assert odd["fallback"] and odd["shortcuts"] == ["some_new_executor"], "unknown executors are never target passes"


OLD_TOOLS = {"say", "look", "reachability", "pick", "place", "wait"}


def test_no_old_tool_names_in_ui_and_eval():
    """PLAN 10 test_tool_vocab for ui/ and eval/: no comparison of a tool name (.tool, tool_name, skill,
    started(...), 'tool' keys) against THOR's old tools. Actions ('pick'/'place' as manipulate's action) are fine."""
    tool_attr = {"tool", "tool_name", "skill"}

    def is_tool_expr(n):
        if isinstance(n, ast.Attribute) and n.attr in tool_attr:
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value in tool_attr:
            return True
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in tool_attr:
            return True
        if isinstance(n, ast.BoolOp):
            return any(is_tool_expr(v) for v in n.values)
        return isinstance(n, ast.Name) and n.id in ("tool",)

    def consts(n):
        return {c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}

    bad = []
    for p in sorted(list((ROOT / "ui").glob("*.py")) + list((ROOT / "eval").glob("*.py"))):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and (is_tool_expr(node.left) or any(is_tool_expr(c) for c in node.comparators)):
                hit = consts(node) & OLD_TOOLS
                if hit:
                    bad.append(f"{p.name}:{node.lineno} {sorted(hit)}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "started" \
                    and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in OLD_TOOLS:
                bad.append(f"{p.name}:{node.lineno} started({node.args[0].value!r})")
    assert not bad, bad
