"""Episodic memory: what happened, session by session.

Every session appends structured events to runs/episodes/<scene>/<stamp>.jsonl
as they happen (a crash still leaves a record): what the user said and how it
was labelled, each decision, each action and its outcome, where the robot
learned things are, deliveries, corrections, stops and own goals. The recall
tool answers "what did we do" from these, and the procedural graph
(agent/procedures.py) learns from them.

    log = EpisodeLog("procthor-train-40")
    tracer.sinks.append(log.write)          # every trace row of interest lands on disk
    episodes = load_episodes("procthor-train-40")   # oldest first, each a list of rows
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("WORLDLINE_RUNS") or Path(__file__).resolve().parents[1] / "runs") / "episodes"

KEEP = {
    "session_start", "heard", "classified", "decision", "started", "result", "rejected",
    "stale_decision", "stop", "correction", "say_queued", "repeat_say_dropped",
    "unprompted_say_dropped", "persona_goal", "persona_goal_end", "note_saved",
    "place_learned", "delivered", "recall", "decision_limit", "rejection_limit", "brain_error",
    "observation",
    # G1 / Ludi API additions (PLAN 4.4)
    "tool_result", "rejection", "visual_observation", "goal_check", "body_mode", "safety_event", "stale_result",
    "late_result", "resume", "capability_changed",
}


class EpisodeLog:
    def __init__(self, scene: str | None, root: Path | None = None) -> None:
        root = ROOT if root is None else root                  # looked up at call time (tests relocate it)
        self.path = (root / scene / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl") if scene else None
        self._f: Any = None

    def write(self, row: dict[str, Any]) -> None:
        if self.path is None or row.get("type") not in KEEP:
            return
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("a")
        self._f.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def load_episodes(scene: str | None = None, root: Path | None = None, limit: int = 200) -> list[list[dict[str, Any]]]:
    """Past sessions, oldest first: all scenes when ``scene`` is None."""
    root = ROOT if root is None else root
    if not root.exists():
        return []
    files = sorted((root / scene).glob("*.jsonl") if scene else root.glob("*/*.jsonl"),
                   key=lambda p: p.name)[-limit:]
    out = []
    for f in files:
        rows = []
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        if rows:
            out.append(rows)
    return out


def session_wall(rows: list[dict[str, Any]]) -> float | None:
    """Wall-clock time a session started (from its session_start row)."""
    for r in rows:
        if r.get("type") == "session_start":
            return r.get("wall")
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One session in a few fields: what the user asked for (requests and corrections,
    not questions or chat) and what got delivered."""
    asked = {r.get("id") for r in rows if r.get("type") == "classified" and r.get("kind") in ("request", "correction")}
    return {"wall": session_wall(rows),
            "requests": [r.get("text") for r in rows if r.get("type") == "heard" and r.get("id") in asked],
            "delivered": [f"{r.get('object')} to {r.get('surface')}" for r in rows if r.get("type") == "delivered"],
            "decisions": sum(1 for r in rows if r.get("type") == "decision"),
            "recalls": sum(1 for r in rows if r.get("type") == "recall")}
