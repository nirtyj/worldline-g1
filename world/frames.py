"""Camera frame sources for the runtime side (api.services.FrameSource: latest(camera) / latest_frame(camera)).

    IsaacFrames   P1's frames through viz's FrameTap (docs/viz.md §9.2; imported, not copied): the head camera
                  (P1 PUB 5565, today the d435-mounted camera, contract §1.4) and the chase/top views (PUB 5602).
                  Each frame is stamped with the GT camera pose of the world model at receive time, in
                  Worldline's frame (cam_pose_wl = (x, z, yaw_cw_deg, pitch_down_deg)).
    NoFrames      the lite profile (GT worlds render no images; the UI shows no camera pane). PLAN §8.6's icon
                  sprites would slot in here.

`stationary` is true when the GT base speed is below 0.05 m/s and the yaw rate below 0.05 rad/s (FrameGate's
new-view/changed-scene logic needs it, PLAN §8.3).
"""

from __future__ import annotations

import math
import time
from typing import Any

from . import coords


class NoFrames:
    def latest(self, camera: str = "head"):
        return None

    def latest_frame(self, camera: str = "head"):
        return None

    def close(self) -> None:
        pass


class IsaacFrames:
    NAMES = {"head": "head", "ego": "head", "chase": "chase", "top": "top"}

    def __init__(self, world: Any = None, port_offset: int | None = None, tap: Any = None):
        if tap is None:
            from viz.tap import FrameTap
            tap = FrameTap(port_offset=port_offset)
        self.tap = tap
        self.world = world

    def latest(self, camera: str = "head"):
        from api.types import CameraFrame
        name = self.NAMES.get(camera, camera)
        jpeg = self.tap.jpeg(name)
        if jpeg is None:
            return None
        meta = self.tap.meta(name)
        pose_wl = (0.0, 0.0, 0.0, 0.0)
        stationary = False
        hfov = vfov = None
        if self.world is not None and name == "head":
            cp = self.world.camera_pose()
            mx, mz = coords.to_map_xz(cp.pos[0], cp.pos[1])
            pose_wl = (round(mx, 3), round(mz, 3), round(coords.yaw_map_deg(cp.yaw), 1),
                       round(math.degrees(cp.pitch_down), 1))
            rp = self.world.robot_pose()
            stationary = rp.speed < 0.05 and abs(rp.wz) < 0.05
            hfov, vfov = self.world.cam.hfov_deg, self.world.cam.vfov_deg
        return CameraFrame(rev=int(self.tap.rev(name)), t_wall=float(meta.get("t_wall") or time.time()), jpeg=jpeg,
                           cam_pose_wl=pose_wl, stationary=stationary, camera="head" if name == "head" else "top",
                           hfov=hfov, vfov=vfov, w=int(meta.get("w") or meta.get("width") or 0),
                           h=int(meta.get("h") or meta.get("height") or 0))

    latest_frame = latest

    def close(self) -> None:
        close = getattr(self.tap, "close", None)
        if callable(close):
            close()
