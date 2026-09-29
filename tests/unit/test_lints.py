"""Static checks over the runtime code (PLAN 10):

  test_tool_vocab      no comparison of .tool / tool_name / skill / started(...) with an old tool name
  test_status_vocab    no uppercase THOR status literals
  test_gt_confinement  agent/, brains/, llmkit/ never import world/, robot/, services/, body/, the sim
                       process packages, or sim.log (the ground-truth event log)
  test_gt_reads_live_in_world
                       outside world/ (agent/, brains/, llmkit/, robot/, services/, ui/) nothing reads P1's
                       ground truth directly: no import of P1's GT client (body.p1_client) or the sim process
                       packages, no `gt_pose` field access (.gt_pose, ["gt_pose"], .get("gt_pose")), no P1 GT topic
                       literal (gt.pose, gt.event(s), gt.objects, sim.health), and no viz.tap.FrameTap built without
                       gt_pose=False (the tap subscribes to gt.pose by default). World owns GT (PLAN §6.2): RTF is
                       WorldModel.sim_health, speed WorldModel.planar_speed, frames world.frames.IsaacFrames.
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


# ------------------------------------------------------------------ GT reads outside world/
GT_DIRS = RUNTIME_DIRS + ("robot", "services", "ui")
GT_IMPORTS = {"body.p1_client", "sim_isaac", "isaac_host", "scenes", "thor"}
GT_TOPICS = {"gt.pose", "gt.event", "gt.events", "gt.objects", "sim.health", "gt_pose"}


def _docstrings(tree: ast.AST) -> set[int]:
    """ids of string constants that are statements (docstrings, bare strings): prose, not reads."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.add(id(node.value))
    return out


def gt_read_violations(tree: ast.AST, where: str) -> list[str]:
    bad = []
    prose = _docstrings(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mods = [node.module or ""]
            mods += [f"{node.module}.{a.name}" for a in node.names]
        hit = next((m for m in mods if any(m == g or m.startswith(g + ".") for g in GT_IMPORTS)), None)
        if hit:
            bad.append(f"{where}:{line} imports {hit}")
        if isinstance(node, ast.Attribute) and node.attr == "gt_pose":
            bad.append(f"{where}:{line} reads .gt_pose")
        if isinstance(node, ast.Constant) and id(node) not in prose:
            v = node.value.decode("utf-8", "replace") if isinstance(node.value, bytes) else node.value
            if isinstance(v, str) and v in GT_TOPICS:
                bad.append(f"{where}:{line} uses the P1 GT name {v!r}")
        if isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn == "FrameTap":
                kw = {k.arg: k.value for k in node.keywords}
                off = kw.get("gt_pose")
                if not (isinstance(off, ast.Constant) and off.value is False):
                    bad.append(f"{where}:{line} builds a FrameTap that subscribes to gt.pose (pass gt_pose=False)")
    return bad


def test_gt_reads_live_in_world():
    bad = []
    for d in GT_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            bad += gt_read_violations(ast.parse(path.read_text()), str(path.relative_to(ROOT)))
    assert bad == []


def test_the_gt_lint_catches_the_m2a_findings():
    """The patterns the M2a verifier found (robot/body_client.py health and speed from the body's copy of gt.pose,
    robot/bridge.py telemetry, ui/cameras.py's FrameTap) and the allowed forms."""
    src = (
        'rtf = (st.get("gt_pose") or {}).get("rtf")\n'
        'gp = st["gt_pose"]\n'
        'v = state.gt_pose.speed\n'
        'from body.p1_client import PoseSub\n'
        'import body.p1_client\n'
        's.setsockopt(zmq.SUBSCRIBE, b"gt.pose")\n'
        'tap = FrameTap(port_offset=0)\n'
        'tap = viz.tap.FrameTap(port_offset=0, gt_pose=True)\n'
    )
    hits = gt_read_violations(ast.parse(src), "x.py")
    assert sorted(int(h.split(":")[1].split(" ")[0]) for h in hits) == list(range(1, 9)), hits
    ok = ('"""Docs may say gt.pose."""\n'
          'tap = FrameTap(port_offset=0, gt_pose=False)\n'
          'h = world.sim_health()\n'
          'v = world.planar_speed()\n'
          'from body.client import BodyClient\n')
    assert gt_read_violations(ast.parse(ok), "y.py") == []


@pytest.mark.parametrize("d", RUNTIME_DIRS + ("api",))
def test_no_thor_or_goals_left(d):
    for path in sorted((ROOT / d).rglob("*.py")):
        text = path.read_text()
        assert "ai2thor" not in text and "sim.goals" not in text and "from thor" not in text, path
