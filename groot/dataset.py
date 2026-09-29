"""Read episodes of the Arena LeRobot v2.1 dataset (nvidia/Arena-G1-Static-PickNPlace-Task, CC-BY-4.0).

Box-side helper for the open-loop check and the latency bench. Needs `pyarrow` (parquet) and the `ffmpeg` CLI (video
decode to raw RGB); neither is a runtime dependency, so this module imports pyarrow lazily.

Layout (`lerobot/meta/info.json`): `data/chunk-000/episode_{i:06d}.parquet` with `observation.state` / `action`
(43 f32, joint_order.LEROBOT_43_JOINTS), `teleop.navigate_command` (3), `teleop.base_height_command` (1),
`annotation.human.task_description` (int, a task_index into meta/tasks.jsonl); and
`videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4` (640x480 h264 yuv420p, 50 fps).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    index: int
    state: np.ndarray            # (N, 43) f32, LeRobot 43 order
    action: np.ndarray           # (N, 43) f32
    navigate: np.ndarray         # (N, 3)
    base_height: np.ndarray      # (N, 1)
    task_index: int
    task: str
    frames: np.ndarray | None    # (N, 480, 640, 3) uint8 RGB, or None when not decoded

    @property
    def n(self) -> int:
        return int(self.state.shape[0])


def load_tasks(root: Path) -> dict[int, str]:
    out = {}
    for line in (Path(root) / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[int(d["task_index"])] = d["task"]
    return out


def decode_video(path: Path, hw: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Every frame of an mp4 as uint8 RGB (N, H, W, 3), via the ffmpeg CLI."""
    h, w = hw
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], check=True, capture_output=True).stdout
    n = len(raw) // (h * w * 3)
    if n * h * w * 3 != len(raw):
        raise ValueError(f"{path}: {len(raw)} bytes is not a whole number of {w}x{h} RGB frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3)


def load_episode(root: str | Path, index: int, *, frames: bool = True, chunk: int = 0) -> Episode:
    import pyarrow.parquet as pq

    root = Path(root)
    t = pq.read_table(root / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet").to_pydict()
    state = np.asarray(t["observation.state"], dtype=np.float32)
    action = np.asarray(t["action"], dtype=np.float32)
    ann = sorted(set(int(x) for x in t["annotation.human.task_description"]))
    if len(ann) != 1:
        raise ValueError(f"episode {index}: {len(ann)} different task annotations {ann}")
    tasks = load_tasks(root)
    vid = None
    if frames:
        vid = decode_video(root / "videos" / f"chunk-{chunk:03d}" / "observation.images.ego_view"
                           / f"episode_{index:06d}.mp4")
        if vid.shape[0] < state.shape[0]:
            raise ValueError(f"episode {index}: {vid.shape[0]} frames for {state.shape[0]} rows")
        vid = vid[: state.shape[0]]
    return Episode(index, state, action, np.asarray(t["teleop.navigate_command"], dtype=np.float32),
                   np.asarray(t["teleop.base_height_command"], dtype=np.float32), ann[0], tasks.get(ann[0], ""), vid)
