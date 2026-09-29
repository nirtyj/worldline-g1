"""The robot's own map of the house, the way a robot vacuum shows it.

Everything here comes from the robot itself, never from the house layout:

  free      the nav stack's free-space grid (the cells it plans paths on)
  trail     the path it drove: its pose (odometry) whenever it moved or turned
  explored  floor its head camera has covered: grid cells within reach of the
            detector, inside the camera's view, with a clear line of sight over
            free cells (a look turns the head, so it covers a wider fan)
  sightings where it stood when it found something (a verified place_learned row)

The page draws the house outline too, but only as a visual aid: the robot has no
walls or room shapes, just this grid, its named spots and what it has seen.
"""

from __future__ import annotations

import math
from typing import Any

VIEW_M = 1.5          # the detector's range for objects (thor/world.py REACH_M)
HALF_FOV = 45.0       # the head camera sees 90°
LOOK_HALF = 85.0      # a look turns the head ±40° on top of that


class RobotMap:
    def __init__(self, free: set[tuple[int, int]], step: float) -> None:
        self.free, self.step = free, step
        self.trail: list[list[float]] = []
        self.explored: set[tuple[int, int]] = set()
        self.sightings: list[dict[str, Any]] = []
        self._new: dict[str, list[Any]] = {"trail": [], "explored": [], "sightings": []}

    def pose(self, t: float, x: float, z: float, yaw: float, looking: bool) -> None:
        last = self.trail[-1] if self.trail else None
        moved = (last is None or abs(last[1] - x) >= 0.05 or abs(last[2] - z) >= 0.05
                 or abs((last[3] - yaw + 180) % 360 - 180) >= 5)
        if moved:
            p = [round(t, 2), round(x, 2), round(z, 2), round(yaw, 1)]
            self.trail.append(p)
            self._new["trail"].append(p)
        if moved or looking:
            self._cover(x, z, yaw, LOOK_HALF if looking else HALF_FOV)

    def found(self, t: float, obj: str, place: str, x: float, z: float) -> None:
        s = {"t": round(t, 2), "object": obj, "place": place, "x": round(x, 2), "z": round(z, 2)}
        self.sightings.append(s)
        self._new["sightings"].append(s)

    def full(self) -> dict[str, Any]:
        """Everything so far, for a page that just connected (other pages keep getting take_new)."""
        return {"trail": self.trail, "explored": [list(c) for c in self.explored], "sightings": self.sightings}

    def take_new(self) -> dict[str, Any]:
        out, self._new = self._new, {"trail": [], "explored": [], "sightings": []}
        return out

    # ------------------------------------------------------------------
    def _cover(self, x: float, z: float, yaw: float, half: float) -> None:
        s, r = self.step, int(VIEW_M / self.step) + 1
        cx, cz = round(x / s), round(z / s)
        for ix in range(cx - r, cx + r + 1):
            for iz in range(cz - r, cz + r + 1):
                c = (ix, iz)
                if c in self.explored or c not in self.free:
                    continue
                dx, dz = ix * s - x, iz * s - z
                d = math.hypot(dx, dz)
                if d > VIEW_M:
                    continue
                if d > s:                                   # its own cell always counts
                    heading = math.degrees(math.atan2(dx, dz))   # yaw 0 faces +z, as in thor/robot.py
                    if abs((heading - yaw + 180) % 360 - 180) > half or not self._clear(x, z, ix * s, iz * s):
                        continue
                self.explored.add(c)
                self._new["explored"].append([ix, iz])

    def _clear(self, x0: float, z0: float, x1: float, z1: float) -> bool:
        """Line of sight over free cells only: walls and furniture block the view of the floor."""
        n = max(2, int(math.hypot(x1 - x0, z1 - z0) / (self.step / 2)))
        for i in range(1, n):
            f = i / n
            if (round((x0 + (x1 - x0) * f) / self.step), round((z0 + (z1 - z0) * f) / self.step)) not in self.free:
                return False
        return True
