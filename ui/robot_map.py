"""The robot's own map of the house, the way a robot vacuum shows it.

Everything here comes from the robot itself, never from the house layout:

  free      the nav stack's free-space grid (the cells it plans paths on), from the
            occupancy grid's inflated layer, shown at 0.25 m (ui.truth.display_cells)
  trail     the path it walked: its pose (odometry) whenever it moved or turned
  explored  floor its head camera has covered: floor cells within the detector's range,
            inside the camera's view, beyond the near edge of the view frustum on the
            floor, with a clear line of sight over floor cells. A scan twists the waist,
            so it covers a wider fan, one pass per scan row (each row has its own near edge)
  sightings where it stood when it found something (a verified place_learned row)

The page draws the house outline too, but only as a visual aid: the robot has no walls
or room shapes, just this grid, its named spots and what it has seen.

G1 numbers (PLAN 4.4, 5.10, 6.5): the head camera sees 90 deg across (HFOV) and 73.7 deg
up and down (VFOV), from about 1.35 m. A scan is two rows of waist yaw -35/0/+35 deg:
pitch 0 (camera tilted 15 deg down) and pitch +20 (35 deg down). Objects count as seen
up to 2.5 m (VIEW_M).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

VIEW_M = 2.5          # the GT detector's range for objects (world/where.py: 2.5 m)
HALF_FOV = 45.0       # the head camera sees 90 deg across
SCAN_YAW = 35.0       # a scan twists the waist +-35 deg on top of that
CAM_H = 1.35          # head camera height (m)
VFOV = 73.7           # 90 deg HFOV at 4:3
TILT = 15.0           # camera pitch while walking and in scan row 1 (deg down)
SCAN_TILTS = (15.0, 35.0)   # scan rows: waist pitch 0, then +20 deg down


def near_edge_m(cam_h: float, tilt_deg: float, vfov_deg: float) -> float:
    """Distance on the floor to the bottom edge of the view: nothing closer is in the picture."""
    low = math.radians(tilt_deg + vfov_deg / 2)
    if low >= math.pi / 2:
        return 0.0
    return cam_h / math.tan(low)


class RobotMap:
    def __init__(self, free: Iterable[tuple[int, int]], step: float, floor: Iterable[tuple[int, int]] | None = None,
                 *, view_m: float = VIEW_M, half_fov: float = HALF_FOV, scan_yaw: float = SCAN_YAW,
                 cam_h: float = CAM_H, vfov: float = VFOV, tilt: float = TILT,
                 scan_tilts: tuple[float, ...] = SCAN_TILTS) -> None:
        self.free = set(free)
        self.floor = set(floor) if floor is not None else set(self.free)   # line of sight crosses floor, not just nav cells
        self.step = step
        self.view_m, self.half_fov, self.scan_yaw = view_m, half_fov, scan_yaw
        self.cam_h, self.vfov, self.tilt, self.scan_tilts = cam_h, vfov, tilt, scan_tilts
        self.trail: list[list[float]] = []
        self.explored: set[tuple[int, int]] = set()
        self.sightings: list[dict[str, Any]] = []
        self._new: dict[str, list[Any]] = {"trail": [], "explored": [], "sightings": []}

    def pose(self, t: float, x: float, z: float, yaw: float, looking: bool) -> None:
        """One odometry sample. `looking`: a scan (waist twist) is running now."""
        last = self.trail[-1] if self.trail else None
        moved = (last is None or abs(last[1] - x) >= 0.05 or abs(last[2] - z) >= 0.05
                 or abs((last[3] - yaw + 180) % 360 - 180) >= 5)
        if moved:
            p = [round(t, 2), round(x, 2), round(z, 2), round(yaw, 1)]
            self.trail.append(p)
            self._new["trail"].append(p)
        if looking:
            for tilt in self.scan_tilts:          # each scan row: its own fan and its own near edge
                self._cover(x, z, yaw, self.half_fov + self.scan_yaw, near_edge_m(self.cam_h, tilt, self.vfov))
        elif moved:
            self._cover(x, z, yaw, self.half_fov, near_edge_m(self.cam_h, self.tilt, self.vfov))

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

    def frustum(self, x: float, z: float, yaw: float, looking: bool) -> dict[str, Any]:
        """The view fan the page draws: per row its half angle, near edge and range."""
        rows = ([{"half": self.half_fov + self.scan_yaw, "near": round(near_edge_m(self.cam_h, t, self.vfov), 2),
                  "tilt": t} for t in self.scan_tilts] if looking else
                [{"half": self.half_fov, "near": round(near_edge_m(self.cam_h, self.tilt, self.vfov), 2), "tilt": self.tilt}])
        return {"x": round(x, 2), "z": round(z, 2), "yaw": round(yaw, 1), "range": self.view_m, "rows": rows}

    # ------------------------------------------------------------------
    def _cover(self, x: float, z: float, yaw: float, half: float, near: float) -> None:
        s, r = self.step, int(self.view_m / self.step) + 1
        cx, cz = round(x / s), round(z / s)
        for ix in range(cx - r, cx + r + 1):
            for iz in range(cz - r, cz + r + 1):
                c = (ix, iz)
                if c in self.explored or c not in self.floor:
                    continue
                dx, dz = ix * s - x, iz * s - z
                d = math.hypot(dx, dz)
                if d > self.view_m:
                    continue
                if c != (cx, cz):                                # its own cell always counts
                    if d < near:
                        continue
                    heading = math.degrees(math.atan2(dx, dz))   # yaw 0 faces +z, clockwise (Worldline frame)
                    if abs((heading - yaw + 180) % 360 - 180) > half or not self._clear(x, z, ix * s, iz * s):
                        continue
                self.explored.add(c)
                self._new["explored"].append([ix, iz])

    def _clear(self, x0: float, z0: float, x1: float, z1: float) -> bool:
        """Line of sight over floor cells only: walls and furniture block the view of the floor."""
        n = max(2, int(math.hypot(x1 - x0, z1 - z0) / (self.step / 2)))
        for i in range(1, n):
            f = i / n
            if (round((x0 + (x1 - x0) * f) / self.step), round((z0 + (z1 - z0) * f) / self.step)) not in self.floor:
                return False
        return True
