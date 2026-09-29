"""Camera frame sources for the runtime side (api.services.FrameSource: latest(camera) / latest_frame(camera)).

    IsaacFrames   P1's frames through viz's FrameTap (docs/viz.md §9.2; imported, not copied): the head camera
                  (P1 PUB 5565, today the d435-mounted camera, contract §1.4) and the chase/top views (PUB 5602).
                  Each frame is stamped with the GT camera pose of the world model at receive time, in
                  Worldline's frame (cam_pose_wl = (x, z, yaw_cw_deg, pitch_down_deg)).
    LiteFrames    the lite profile: a SCHEMATIC head-camera image drawn from the GT detections of the head
                  camera's frustum (a box and a label per visible object or landmark, placed by projecting its
                  centre with the camera model). Not a render; the image says so. It gives System 1's frame path
                  (frame gate, frame(), observe()) and the page's head pane something real to run on offline.
                  A simplified stand-in for PLAN §8.6's icon sprites.
    NoFrames      no images at all.

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


class LiteFrames:
    """Schematic head frames for the lite profile (see the module docstring). ``latest(camera)`` re-draws at most
    every ``min_period_s`` (wall), and only bumps ``rev`` when what is drawn changed."""

    LABEL = "lite schematic: GT detections, not a camera render"

    def __init__(self, world: Any, *, width: int = 320, height: int = 240, min_period_s: float = 0.2):
        self.world = world
        self.w, self.h = width, height
        self.min_period_s = min_period_s
        self._t = -1e9
        self._key: Any = None
        self._frame = None
        self.rev = 0

    def latest(self, camera: str = "head"):
        if camera != "head":
            return None
        now = time.monotonic()
        if self._frame is not None and now - self._t < self.min_period_s:
            return self._frame
        self._t = now
        try:
            return self._draw()
        except Exception:  # noqa: BLE001  (a frame is a nicety; never break a caller over it)
            return self._frame

    latest_frame = latest

    def _draw(self):
        from api.types import CameraFrame
        world = self.world
        rp = world.robot_pose()
        cp = world.camera_pose(pose=rp)
        dets = world.detections()
        key = (round(cp.pos[0], 2), round(cp.pos[1], 2), round(cp.yaw, 2), round(cp.pitch_down, 2),
               tuple(sorted((d.id, d.where, tuple(round(v, 2) for v in d.pos)) for d in dets)))
        stationary = rp.speed < 0.05 and abs(rp.wz) < 0.05
        mx, mz = coords.to_map_xz(cp.pos[0], cp.pos[1])
        pose_wl = (round(mx, 3), round(mz, 3), round(coords.yaw_map_deg(cp.yaw), 1),
                   round(math.degrees(cp.pitch_down), 1))
        if key != self._key or self._frame is None:
            self._key = key
            self.rev += 1
            jpeg = self._render(cp, dets)
        else:
            jpeg = self._frame.jpeg
        cam = getattr(world, "cam", None)
        self._frame = CameraFrame(rev=self.rev, t_wall=time.time(), jpeg=jpeg, cam_pose_wl=pose_wl,
                                  stationary=stationary, camera="head",
                                  hfov=getattr(cam, "hfov_deg", None), vfov=getattr(cam, "vfov_deg", None),
                                  w=self.w, h=self.h)
        return self._frame

    def _render(self, cp: Any, dets: list) -> bytes:
        import io

        import numpy as np
        from PIL import Image, ImageDraw
        cam = self.world.cam
        f = cam.fy * self.h / max(1, cam.height)          # the camera model's focal length at this image size
        img = Image.new("RGB", (self.w, self.h), (214, 222, 230))
        dr = ImageDraw.Draw(img)
        horizon = self.h / 2 - f * math.tan(cp.pitch_down)
        dr.rectangle([0, max(0, int(horizon)), self.w, self.h], fill=(196, 182, 160))
        R = np.asarray(cp.R)
        c = np.asarray(cp.pos, dtype=float)
        items = []
        for d in dets:
            v = R.T @ (np.asarray(d.pos, dtype=float) - c)       # camera frame: x forward, y left, z up
            if v[0] <= 0.05:
                continue
            u = self.w / 2 - f * v[1] / v[0]
            y = self.h / 2 - f * v[2] / v[0]
            items.append((float(v[0]), u, y, d))
        for depth, u, y, d in sorted(items, key=lambda t: -t[0]):   # far first
            half = max(4.0, min(60.0, (18.0 if d.kind == "object" else 40.0) * f / 200.0 / depth))
            h = abs(hash(d.type)) % 360
            col = _hsv(h, 0.55 if d.kind == "object" else 0.25, 0.85 if d.kind == "object" else 0.6)
            dr.rectangle([u - half, y - half, u + half, y + half], fill=col, outline=(40, 40, 40))
            dr.text((u - half, y + half + 1), d.label[:18], fill=(20, 20, 20))
        dr.rectangle([0, 0, self.w, 12], fill=(40, 40, 40))
        dr.text((3, 1), self.LABEL, fill=(255, 210, 120))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return buf.getvalue()

    def close(self) -> None:
        pass


def _hsv(h_deg: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

