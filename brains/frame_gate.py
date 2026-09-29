"""Which head-camera frames are worth sending to System 1.

THOR bumps its frame counter on every controller step, including steps that
only read metadata, so a robot standing still produces a stream of identical
pictures. System 1 should see a frame only when it shows something new:

  scene changed   the robot hasn't moved, but part of the picture has (someone
                  moved something, a door opened)
  new view        the robot moved 1 m or turned 30 degrees since the last frame
                  sent, or the picture changed a lot (a doorway, a corner)
  first           nothing sent yet

Everything else is skipped: standing still ("still"), moving so little that the
view is mostly the same ("similar view"), or a change the robot made itself
("own action": the caller says a pick or place is running; the new picture
becomes the reference, so the result isn't reported as a change later either).
Sent frames are what System 1's observe() looks at, so skipping them also skips
observation calls.

The comparison is a 32x24 grayscale thumbnail: cheap (well under a millisecond)
and blind to JPEG noise. A cell counts as changed when its brightness moves by
more than CELL_DELTA; a scene change needs at least SCENE_CELLS of the cells to
change, so a small object appearing or leaving is enough, and a lighting
flicker is not.

    gate = FrameGate()
    d = gate.decide(frame_rgb, (x, z, yaw_deg, horizon_deg), now)
    if d.send: ...send it...; the counts are in gate.counts
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

THUMB_W, THUMB_H = 32, 24
CELL_DELTA = 0.08          # brightness change (0..1) for a cell to count as changed
SCENE_CELLS = 0.004        # fraction of changed cells that makes a scene change (3 of 768: a mug at 2 m)
VIEW_CELLS = 0.30          # fraction of changed cells that makes a new view before 1 m or 30 degrees
MOVE_M = 0.05              # below this the robot hasn't moved
TURN_DEG = 3.0             # below this it hasn't turned
NEW_VIEW_M = 1.0           # moved this far since the last frame sent: a new view whatever the pixels say
NEW_VIEW_DEG = 30.0        # turned this far: a new view
MIN_INTERVAL_S = 1.0       # never more than one frame a second


def thumbnail(frame: np.ndarray, w: int = THUMB_W, h: int = THUMB_H) -> np.ndarray:
    """An RGB (H, W, 3) uint8 frame as an (h, w) grayscale thumbnail in 0..1, by block averaging."""
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    H, W = a.shape
    a = a[: H - H % h, : W - W % w]                      # crop to a multiple of the grid
    return a.reshape(h, a.shape[0] // h, w, a.shape[1] // w).mean(axis=(1, 3)) / 255.0


def _turn(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class Decision:
    send: bool
    reason: str               # first, scene changed, new view, still, similar view, own action, too soon
    changed: float            # fraction of thumbnail cells that changed since the last frame sent
    moved_m: float
    turned_deg: float


class FrameGate:
    def __init__(self, scene_cells: float = SCENE_CELLS, view_cells: float = VIEW_CELLS,
                 min_interval: float = MIN_INTERVAL_S) -> None:
        self.scene_cells = scene_cells
        self.view_cells = view_cells
        self.min_interval = min_interval
        self.counts: Counter[str] = Counter()
        self._thumb: np.ndarray | None = None
        self._pose: tuple[float, float, float, float] | None = None
        self._t = -math.inf
        self.skipped_since_send: Counter[str] = Counter()

    def reset(self) -> None:
        """Forget the last frame sent (a new session, or System 1 reconnected)."""
        self._thumb, self._pose, self._t = None, None, -math.inf

    def decide(self, frame: np.ndarray, pose: Sequence[float], now: float, expected: bool = False) -> Decision:
        """``expected``: the robot is changing the scene itself (a pick or place is running)."""
        thumb = thumbnail(frame)
        x, z, yaw, horizon = (float(v) for v in pose)
        if self._thumb is None or self._pose is None or self._thumb.shape != thumb.shape:
            return self._send(thumb, (x, z, yaw, horizon), now, "first", 1.0, 0.0, 0.0)
        changed = float(np.mean(np.abs(thumb - self._thumb) > CELL_DELTA))
        px, pz, pyaw, phor = self._pose
        moved = math.hypot(x - px, z - pz)
        turned = max(_turn(yaw, pyaw), abs(horizon - phor))
        if expected:                                  # its own arm at work: follow along, don't report
            self._thumb, self._pose = thumb, (x, z, yaw, horizon)
            return self._skip("own action", changed, moved, turned)
        if now - self._t < self.min_interval:
            return self._skip("too soon", changed, moved, turned)
        if moved < MOVE_M and turned < TURN_DEG:
            if changed >= self.scene_cells:
                return self._send(thumb, (x, z, yaw, horizon), now, "scene changed", changed, moved, turned)
            return self._skip("still", changed, moved, turned)
        if moved >= NEW_VIEW_M or turned >= NEW_VIEW_DEG or changed >= self.view_cells:
            return self._send(thumb, (x, z, yaw, horizon), now, "new view", changed, moved, turned)
        return self._skip("similar view", changed, moved, turned)

    def _send(self, thumb: np.ndarray, pose: tuple[float, float, float, float], now: float, reason: str,
              changed: float, moved: float, turned: float) -> Decision:
        self._thumb, self._pose, self._t = thumb, pose, now
        self.counts[reason] += 1
        self.skipped_since_send = Counter()
        return Decision(True, reason, changed, moved, turned)

    def _skip(self, reason: str, changed: float, moved: float, turned: float) -> Decision:
        self.counts[reason] += 1
        self.skipped_since_send[reason] += 1
        return Decision(False, reason, changed, moved, turned)

    def summary(self) -> str:
        """For logs and the page: what was sent and what was skipped, e.g. 'sent 4 (...), skipped 57 (...)'."""
        sent = {k: v for k, v in self.counts.items() if k in ("first", "scene changed", "new view")}
        skipped = {k: v for k, v in self.counts.items() if k not in sent}
        fmt = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))   # noqa: E731
        return (f"sent {sum(sent.values())} ({fmt(sent) or 'none'}), "
                f"skipped {sum(skipped.values())} ({fmt(skipped) or 'none'})")
