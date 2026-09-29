"""Static checks over the runtime code (PLAN 10):

  test_tool_vocab      no comparison of .tool / tool_name / skill / started(...) with an old tool name
  test_status_vocab    no uppercase THOR status literals
  test_gt_confinement  agent/, brains/, llmkit/ never import world/, robot/, services/, body/, the sim
                       process packages, or sim.log (the ground-truth event log)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from api.tools import OLD_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRS = ("agent", "brains", "llmkit")
TOOL_FIELDS = {"tool", "tool_name", "skill"}
OLD_STATUSES = {"SUCCEEDED", "ABORTED", "CANCELED", "TIMEOUT", "REJECTED", "DROPPED", "FAILED"}
FORBIDDEN_IMPORTS = {"world", "robot", "services", "body", "isaac_host", "sim_isaac", "scenes", "viz", "nav2",
                     "sonic", "thor", "baseline"}


def files():
    for d in RUNTIME_DIRS:
        yield from sorted((ROOT / d).rglob("*.py"))


def _is_tool_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr in TOOL_FIELDS:
        return True
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in TOOL_FIELDS:
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in TOOL_FIELDS):
        return True
    return False


def _old_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value} & set(OLD_TOOL_NAMES)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: set[str] = set()
        for e in node.elts:
            out |= _old_names(e)
        return out
    return set()


def tool_vocab_violations() -> list[str]:
    out = []
    for path in files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                sides = [node.left, *node.comparators]
                if any(_is_tool_expr(s) for s in sides):
                    bad = set().union(*(_old_names(s) for s in sides))
                    if bad:
                        out.append(f"{path.relative_to(ROOT)}:{node.lineno} compares a tool with {sorted(bad)}")
            if (isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
                    and getattr(node.func, "id", getattr(node.func, "attr", "")) == "started" and node.args):
                bad = _old_names(node.args[0])
                if bad:
                    out.append(f"{path.relative_to(ROOT)}:{node.lineno} started({sorted(bad)})")
    return out


def test_tool_vocab():
    assert tool_vocab_violations() == []


def test_the_lint_catches_old_names(tmp_path):
    src = "if e.tool == 'say': pass\nif r.get('skill') in ('pick', 'x'): pass\nif row['tool'] == 'navigate': pass\n"
    tree = ast.parse(src)
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
            and any(_is_tool_expr(s) for s in [n.left, *n.comparators])
            and set().union(*(_old_names(s) for s in [n.left, *n.comparators]))]
    assert len(hits) == 2


def test_status_vocab():
    bad = []
    for path in files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in OLD_STATUSES:
                bad.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.value}")
    assert bad == []


def test_gt_confinement():
    bad = []
    for path in files():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            for n in names:
                if n.split(".")[0] in FORBIDDEN_IMPORTS or n == "sim.log" or n.startswith("sim.log."):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {n}")
    assert bad == []


@pytest.mark.parametrize("d", RUNTIME_DIRS + ("api",))
def test_no_thor_or_goals_left(d):
    for path in sorted((ROOT / d).rglob("*.py")):
        text = path.read_text()
        assert "ai2thor" not in text and "sim.goals" not in text and "from thor" not in text, path
