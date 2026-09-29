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

    gate = FrameGate()                                  # THOR constants
    gate = FrameGate(config=G1_HEAD)                    # the G1 head camera (PLAN 8.3)
    d = gate.decide(frame_rgb, (x, z, yaw_deg, horizon_deg), now, stationary=frame.stationary)
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


@dataclass(frozen=True)
class FrameGateConfig:
    """The gate's thresholds. The defaults are Worldline's THOR constants, so the old tests pass;
    the G1 presets are below (PLAN 8.3)."""
    new_view_deg: float = NEW_VIEW_DEG
    cell_delta: float = CELL_DELTA
    scene_cells: float = SCENE_CELLS
    view_cells: float = VIEW_CELLS
    move_m: float = NEW_VIEW_M
    min_interval_s: float = MIN_INTERVAL_S
    still_move_m: float = MOVE_M
    still_turn_deg: float = TURN_DEG
    scene_only_when_stationary: bool = False   # G1: sway and bob while walking are not "scene changed"
    align_cells: int = 0                       # G1: compare at the best integer shift within +-align_cells


DEFAULT = FrameGateConfig()
# 90 deg HFOV head camera on the torso: 25 deg = 0.28 x HFOV is a new view.
G1_HEAD = FrameGateConfig(new_view_deg=25.0, cell_delta=0.10, scene_cells=0.008, move_m=1.0, min_interval_s=1.0,
                          scene_only_when_stationary=True, align_cells=2)
# ~58 deg HFOV ego camera (D435 mount).
G1_EGO = FrameGateConfig(new_view_deg=16.0, cell_delta=0.10, scene_cells=0.012, move_m=1.0, min_interval_s=1.0,
                         scene_only_when_stationary=True, align_cells=2)
PRESETS = {"default": DEFAULT, "g1_head": G1_HEAD, "g1_ego": G1_EGO}


def changed_fraction(a: np.ndarray, b: np.ndarray, cell_delta: float, align_cells: int = 0) -> float:
    """Fraction of cells whose brightness moved by more than cell_delta; with alignment, the
    smallest fraction over integer shifts within +-align_cells (the border cells are ignored)."""
    if align_cells <= 0:
        return float(np.mean(np.abs(a - b) > cell_delta))
    k = int(align_cells)
    h, w = a.shape
    core = a[k:h - k, k:w - k]
    best = 1.0
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            other = b[k + dy:h - k + dy, k + dx:w - k + dx]
            best = min(best, float(np.mean(np.abs(core - other) > cell_delta)))
    return best


class FrameGate:
    def __init__(self, scene_cells: float | None = None, view_cells: float | None = None,
                 min_interval: float | None = None, config: FrameGateConfig | None = None) -> None:
        cfg = config or DEFAULT
        self.config = cfg
        self.scene_cells = cfg.scene_cells if scene_cells is None else scene_cells
        self.view_cells = cfg.view_cells if view_cells is None else view_cells
        self.min_interval = cfg.min_interval_s if min_interval is None else min_interval
        self.counts: Counter[str] = Counter()
        self._thumb: np.ndarray | None = None
        self._pose: tuple[float, float, float, float] | None = None
        self._t = -math.inf
        self.skipped_since_send: Counter[str] = Counter()

    @classmethod
    def preset(cls, name: str) -> "FrameGate":
        return cls(config=PRESETS[name])

    def reset(self) -> None:
        """Forget the last frame sent (a new session, or System 1 reconnected)."""
        self._thumb, self._pose, self._t = None, None, -math.inf

    def decide(self, frame: np.ndarray, pose: Sequence[float], now: float, expected: bool = False,
               stationary: bool | None = None) -> Decision:
        """``expected``: the robot is changing the scene itself (a manipulate is running).
        ``stationary``: from the frame (base speed, camera rate, body HOLD); None = unknown."""
        cfg = self.config
        thumb = thumbnail(frame)
        x, z, yaw, horizon = (float(v) for v in pose)
        if self._thumb is None or self._pose is None or self._thumb.shape != thumb.shape:
            return self._send(thumb, (x, z, yaw, horizon), now, "first", 1.0, 0.0, 0.0)
        changed = changed_fraction(thumb, self._thumb, cfg.cell_delta, cfg.align_cells)
        px, pz, pyaw, phor = self._pose
        moved = math.hypot(x - px, z - pz)
        turned = max(_turn(yaw, pyaw), abs(horizon - phor))
        if expected:                                  # its own arm at work: follow along, don't report
            self._thumb, self._pose = thumb, (x, z, yaw, horizon)
            return self._skip("own action", changed, moved, turned)
        if now - self._t < self.min_interval:
            return self._skip("too soon", changed, moved, turned)
        if moved < cfg.still_move_m and turned < cfg.still_turn_deg:
            may_report = not (cfg.scene_only_when_stationary and stationary is False)
            if changed >= self.scene_cells and may_report:
                return self._send(thumb, (x, z, yaw, horizon), now, "scene changed", changed, moved, turned)
            return self._skip("still" if may_report else "similar view", changed, moved, turned)
        if moved >= cfg.move_m or turned >= cfg.new_view_deg or changed >= self.view_cells:
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
