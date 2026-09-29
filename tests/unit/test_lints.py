"""Static checks over the runtime code (PLAN 10):

  test_tool_vocab      no comparison of .tool / tool_name / skill / started(...) with an old tool name
  test_status_vocab    no uppercase THOR status literals
  test_gt_confinement  agent/, brains/, llmkit/ never import world/, robot/, services/, body/, the sim
                       process packages, zmq (no raw sockets in the brain), or sim.log (the ground-truth event log),
                       statically or through importlib.import_module / __import__ with a constant name
  test_gt_reads_live_in_world
                       outside world/ (agent/, brains/, llmkit/, robot/, services/, ui/) nothing reads P1's
                       ground truth directly: no import of P1's GT client (body.p1_client) or the sim process
                       packages (also not by importlib.import_module / __import__), no `gt_pose` field access
                       (.gt_pose, ["gt_pose"], .get("gt_pose"), getattr), no P1 GT topic literal (gt.pose, gt.event(s),
                       gt.objects, sim.health; also when built from pieces: "gt" + ".pose", f-strings, str.join), no
                       raw socket on P1's GT ports (REP 5600, PUB 5601: the numbers, ":5600"/":5601" in an endpoint,
                       the port-table keys p1_rep / p1_pose), no hand-built P1 GT request ({"op": "get_objects"} and
                       the other GT reads), and no viz.tap.FrameTap built without gt_pose=False (the tap subscribes to
                       gt.pose by default). World owns GT (PLAN §6.2): RTF is WorldModel.sim_health, speed
                       WorldModel.planar_speed, frames world.frames.IsaacFrames. Rendered camera streams (head 5565,
                       viz 5602, ego_view 5566) and SONIC's joint telemetry (5557) are sensors, not ground truth.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from api.tools import OLD_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRS = ("agent", "brains", "llmkit")
TOOL_FIELDS = {"tool", "tool_name", "skill"}
OLD_STATUSES = {"SUCCEEDED", "ABORTED", "CANCELED", "TIMEOUT", "REJECTED", "DROPPED", "FAILED"}
FORBIDDEN_IMPORTS = {"world", "robot", "services", "body", "isaac_host", "sim_isaac", "scenes", "viz", "nav2",
                     "sonic", "thor", "baseline", "zmq"}
DYNAMIC_IMPORTERS = {"import_module", "__import__"}         # importlib.import_module(name), __import__(name)


def const_str(node: ast.AST) -> str | None:
    """The string a node always evaluates to, when it is built only from literals: "a", b"a", "a" + "b",
    f"{'a'}b" and "sep".join(["a", "b"]); None otherwise (a name, a call, a formatted variable)."""
    if isinstance(node, ast.Constant):
        v = node.value
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else v if isinstance(v, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = const_str(node.left), const_str(node.right)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = [const_str(v.value if isinstance(v, ast.FormattedValue) else v) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple))):
        sep = const_str(node.func.value)
        parts = [const_str(e) for e in node.args[0].elts]
        return sep.join(parts) if sep is not None and all(p is not None for p in parts) else None
    return None


def dynamic_imports(node: ast.AST) -> list[str]:
    """Module names imported through importlib.import_module("x") / __import__("x") with a literal name."""
    if not (isinstance(node, ast.Call) and node.args):
        return []
    fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
    name = const_str(node.args[0]) if fn in DYNAMIC_IMPORTERS else None
    return [name] if name else []


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


def confinement_violations(tree: ast.AST, where: str) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        names += dynamic_imports(node)
        for n in names:
            if n.split(".")[0] in FORBIDDEN_IMPORTS or n == "sim.log" or n.startswith("sim.log."):
                bad.append(f"{where}:{node.lineno} imports {n}")
    return bad


def test_gt_confinement():
    bad = []
    for path in files():
        bad += confinement_violations(ast.parse(path.read_text()), str(path.relative_to(ROOT)))
    assert bad == []


# ------------------------------------------------------------------ GT reads outside world/
GT_DIRS = RUNTIME_DIRS + ("robot", "services", "ui")
GT_IMPORTS = {"body.p1_client", "sim_isaac", "isaac_host", "scenes", "thor"}
GT_TOPICS = {"gt.pose", "gt.event", "gt.events", "gt.objects", "sim.health", "gt_pose"}
GT_PORTS = {5600, 5601}                                     # P1 REP (ops) and PUB gt.pose (body/config.py BASE_PORTS)
GT_PORT_KEYS = {"p1_rep", "p1_pose"}                        # their names in body/config.py BASE_PORTS / ports(off)
GT_PORT_IN_ENDPOINT = re.compile(r":560[01](?!\d)")
# P1 ops that return ground truth (docs/contracts/p1_m2b.md): world/ sends them, nothing else builds such a request
P1_GT_OPS = {"get_objects", "get_link_poses", "detections", "get_health", "get_stats", "scene_info", "get_occupancy",
             "get_robot_pose"}


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
        mods += dynamic_imports(node)
        hit = next((m for m in mods if any(m == g or m.startswith(g + ".") for g in GT_IMPORTS)), None)
        if hit:
            bad.append(f"{where}:{line} imports {hit}")
        if isinstance(node, ast.Attribute) and node.attr == "gt_pose":
            bad.append(f"{where}:{line} reads .gt_pose")
        v = const_str(node) if id(node) not in prose else None
        if v is not None and v in GT_TOPICS:
            bad.append(f"{where}:{line} uses the P1 GT name {v!r}")
        if v is not None and v in GT_PORT_KEYS:
            bad.append(f"{where}:{line} names P1's GT port {v!r}")
        # an endpoint string; an f-string's literal parts are Constant nodes of their own, so f"tcp://{h}:5601" counts
        if v is not None and GT_PORT_IN_ENDPOINT.search(v):
            bad.append(f"{where}:{line} connects to P1's GT port ({v!r})")
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in GT_PORTS:
            bad.append(f"{where}:{line} uses P1's GT port {node.value}")
        if isinstance(node, ast.Dict):
            for k, val in zip(node.keys, node.values):
                if k is not None and const_str(k) == "op" and const_str(val) in P1_GT_OPS:
                    bad.append(f"{where}:{line} builds the P1 GT request {{'op': {const_str(val)!r}}}")
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


def test_the_gt_lint_catches_raw_p1_sockets_and_dynamic_reads():
    """The M2b wave-1 verifier's gaps: a raw ZMQ socket on P1's GT ports, importlib/__import__ of a GT module,
    getattr and names built from pieces, and a hand-built P1 GT request. Camera streams and the body stay allowed."""
    src = (
        's.connect("tcp://127.0.0.1:5601")\n'                        # 1 the gt.pose PUB
        's.connect(f"tcp://{host}:5600")\n'                          # 2 the P1 REP, an f-string
        'sock.connect(ep(5601 + off))\n'                              # 3 the port number
        's.connect(ep(ports["p1_rep"]))\n'                            # 4 the port-table key
        'm = importlib.import_module("body.p1_client")\n'             # 5 a dynamic import
        'm = __import__("sim_isaac.app")\n'                           # 6
        'v = getattr(state, "gt_" + "pose")\n'                        # 7 a name built from pieces
        's.setsockopt(zmq.SUBSCRIBE, "".join(["gt", ".pose"]).encode())\n'   # 8
        'req.send_json({"op": "get_objects"})\n'                      # 9 a hand-built GT request
        't = f"{\'gt\'}.objects"\n'                                   # 10 a constant f-string
    )
    hits = gt_read_violations(ast.parse(src), "x.py")
    assert sorted({int(h.split(":")[1].split(" ")[0]) for h in hits}) == list(range(1, 11)), hits
    ok = ('m = importlib.import_module(spec.partition(":")[0])\n'    # a configured factory, not a literal
          'tap = FrameTap(port_offset=off, gt_pose=False)\n'          # frames only (5602)
          'sensors = ZmqSensors(f"tcp://{host}:{5566 + off}", f"tcp://{host}:{5557 + off}")\n'   # ego_view, g1_debug
          'body = BodyClient(port_offset=off)\n'                      # the body (5610-5612)
          'req = {"op": "arm_script", "phase": "grasp"}\n'            # a body op
          'port = 56010\n'
          'note = "P1.2 get_objects missing"\n')                      # prose inside a message
    assert gt_read_violations(ast.parse(ok), "y.py") == []


def test_the_runtime_confinement_catches_dynamic_imports_and_sockets():
    src = ('w = importlib.import_module("world.isaac_client")\n'
           'import zmq\n'
           'r = __import__("robot" + ".bridge")\n'
           'from sim.log import SimLog\n')
    hits = confinement_violations(ast.parse(src), "agent/x.py")
    assert sorted({int(h.split(":")[1].split(" ")[0]) for h in hits}) == [1, 2, 3, 4], hits
    ok = 'b = importlib.import_module(name)\nfrom api.types import PROFILES\nimport json\n'
    assert confinement_violations(ast.parse(ok), "agent/y.py") == []


@pytest.mark.parametrize("d", RUNTIME_DIRS + ("api",))
def test_no_thor_or_goals_left(d):
    for path in sorted((ROOT / d).rglob("*.py")):
        text = path.read_text()
        assert "ai2thor" not in text and "sim.goals" not in text and "from thor" not in text, path
