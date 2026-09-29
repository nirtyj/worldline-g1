import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest


@pytest.fixture(autouse=True)
def _isolated_runs(tmp_path, monkeypatch):
    """Nothing these tests run writes to the repo's runs/ (memory, episodes, procedures, logs)."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("WORLDLINE_RUNS", str(runs))
    for mod, attr, sub in (("agent.memory", "ROOT", "memory"), ("agent.episodes", "ROOT", "episodes"),
                           ("agent.procedures", "STORE", "procedures.json")):
        m = sys.modules.get(mod)
        if m is None:
            try:
                m = __import__(mod, fromlist=["_"])
            except Exception:  # noqa: BLE001  (agent/ not importable: nothing to redirect)
                continue
        if hasattr(m, attr):
            monkeypatch.setattr(m, attr, runs / sub)
    yield runs
