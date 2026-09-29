"""The M2 bring-up scripts (docs/bringup.md): scripts/m2_up.sh, m2_down.sh, m2_p5.sh, m2_venv.sh, m2_smoke.sh and
scripts/p5_probe.py.

  test_syntax             every stack script parses (bash -n)
  test_help               each M2 script documents its options and exits 0 on --help
  test_up_dry_run         m2_up.sh resolves the house from the scene with the runtime's own code, shifts the page port
                          by the offset, passes --viz to P1, and plans the GR00T link check only for `full`
  test_up_rejects         bad profiles, viz levels and offsets are refused before anything starts
  test_down_idempotent    m2_down.sh on a session that is not running is a no-op
  test_probe_unreachable  p5_probe.py reports a page that is not served, and exits 1
  test_p5_lifecycle       the real page server (lite, scripted planner, System 1 stub) in a private tmux server:
                          start -> ready; start again -> left alone; wrong scene -> not ready; restart keeps the
                          options; stop -> port closed; stop again -> no-op
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
M2 = ["m2_up.sh", "m2_down.sh", "m2_p5.sh", "m2_venv.sh", "m2_smoke.sh"]


def _env(tmp: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE", "WL_PORT_OFFSET")}
    env.update(WL=str(ROOT), PY_RT=sys.executable, M2_STATE=str(tmp / "state"), LOGD=str(tmp / "logs"),
               WORLDLINE_RUNS=str(tmp / "runs"), SECRETS=str(tmp / "no-secrets.env"))
    env.update(extra)
    return env


def _run(args: list[str], env: dict[str, str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_syntax():
    for name in M2 + ["m1_up.sh", "m1_down.sh"]:
        r = subprocess.run(["bash", "-n", str(SCRIPTS / name)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"


def test_help(tmp_path):
    want = {"m2_up.sh": ["--profile", "--scene", "--port-offset", "--viz", "--planner", "--system1", "groot_link.sh"],
            "m2_down.sh": ["--p5-only", "m1_down.sh", "HOLD"],
            "m2_p5.sh": ["start", "stop", "restart", "status", "SimClock(1.0)"],
            "m2_venv.sh": ["--recreate", "--check"],
            "m2_smoke.sh": ["--label", "eval.offline_episode", "tunnel.sh"]}
    for name, words in want.items():
        r = _run([str(SCRIPTS / name), "--help"], _env(tmp_path))
        assert r.returncode == 0, (name, r.stderr)
        for w in words:
            assert w in r.stdout, (name, w)


def test_up_dry_run(tmp_path):
    r = _run([str(SCRIPTS / "m2_up.sh"), "--dry-run", "--profile", "full", "--scene", "procthor-train-40@a",
              "--port-offset", "300", "--viz", "min", "--session", "t-m2"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "house=procthor-train-40 " in out                           # "<house>@<variant>" -> the house P1 loads
    assert "--house procthor-train-40 --port-offset 300 --session t-m2" in out
    assert "ISAAC_ARGS='--viz min'" in out
    assert "--port 9065 --profile full --scene procthor-train-40@a" in out   # 8765 + offset
    assert "nav_backend=astar" in out                                   # Nav2 deferred (PLAN §0.8)
    assert "groot:" in out                                              # full plans the link check (or its skip note)
    r = _run([str(SCRIPTS / "m2_up.sh"), "--dry-run", "--p5-port", "8795",
              "--planner", "brains.scripted:create", "--system1", "tests.kept.system1_stub:create"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "--port 8795 --profile sonic --scene procthor-train-40 --planner brains.scripted:create " \
           "--system1 tests.kept.system1_stub:create" in r.stdout
    assert "groot:" not in r.stdout and "--viz" not in r.stdout.split("ISAAC_ARGS=")[1].split(")")[0]


def test_up_rejects(tmp_path):
    for bad in (["--profile", "lite"], ["--viz", "ultra"], ["--port-offset", "abc"], ["--bogus"]):
        r = _run([str(SCRIPTS / "m2_up.sh"), "--dry-run", *bad], _env(tmp_path))
        assert r.returncode == 2, (bad, r.stdout, r.stderr)


@pytest.fixture
def tmux_dir():
    if not shutil.which("tmux"):
        pytest.skip("tmux not installed")
    d = tempfile.mkdtemp(prefix="wlt", dir="/tmp")     # a private tmux server; its socket path must stay short
    yield d
    subprocess.run(["tmux", "kill-server"], env={**os.environ, "TMUX_TMPDIR": d}, capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


def test_down_idempotent(tmp_path, tmux_dir):
    r = _run([str(SCRIPTS / "m2_down.sh"), "--session", "t-none"], _env(tmp_path, TMUX_TMPDIR=tmux_dir))
    assert r.returncode == 0, r.stderr
    assert "nothing to stop" in r.stdout


def test_probe_unreachable():
    port = _free_port()
    r = subprocess.run([sys.executable, str(SCRIPTS / "p5_probe.py"), "--port", str(port), "--timeout", "1"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 1
    res = json.loads(r.stdout)
    assert res["ok"] is False and res["reason"].startswith("page not served")


def test_p5_lifecycle(tmp_path, tmux_dir):
    port = _free_port()
    env = _env(tmp_path, TMUX_TMPDIR=tmux_dir)
    p5 = [str(SCRIPTS / "m2_p5.sh")]
    opts = ["--session", "t-p5", "--port", str(port), "--profile", "lite", "--scene", "procthor-train-40",
            "--planner", "brains.scripted:create", "--system1", "tests.kept.system1_stub:create", "--timeout", "90"]
    try:
        r = _run(p5 + ["start", *opts], env, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        ready = json.loads((tmp_path / "state" / "p5-t-p5.ready.json").read_text())
        assert ready["ok"] and ready["scene"] == "procthor-train-40" and ready["profile"] == "lite"
        assert ready["http"] == "HTTP 200" and ready["error"] is None
        assert ready["system1"]["status"] == "ready"                     # the stub loaded
        st = (tmp_path / "state" / "p5-t-p5.env").read_text()
        assert f"P5_PORT={port}" in st and "P5_PLANNER=brains.scripted:create" in st

        r = _run(p5 + ["start", "--session", "t-p5"], env)               # options come from the state file
        assert r.returncode == 0 and "already up" in r.stdout, r.stdout + r.stderr

        r = subprocess.run([sys.executable, str(SCRIPTS / "p5_probe.py"), "--port", str(port), "--scene",
                            "procthor-train-15", "--timeout", "2"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 1 and "waiting for procthor-train-15" in json.loads(r.stdout)["reason"]

        r = _run(p5 + ["restart", "--session", "t-p5"], env, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "P5 stopped" in r.stdout and "--profile lite --scene procthor-train-40" in r.stdout
        r = _run(p5 + ["status", "--session", "t-p5"], env)
        assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1])["ok"]
        log = Path(st.split("P5_LOG=")[1].split()[0]).parent
        assert list(log.glob("t-p5-*-p5.log"))
    finally:
        r = _run(p5 + ["stop", "--session", "t-p5"], env)
    assert r.returncode == 0 and "port" in r.stdout and "closed" in r.stdout, r.stdout + r.stderr
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0
    r = _run(p5 + ["stop", "--session", "t-p5"], env)
    assert r.returncode == 0 and "nothing to stop" in r.stdout
