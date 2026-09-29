"""Camera frame sources for the runtime side (api.services.FrameSource: latest(camera) / latest_frame(camera)).

    IsaacFrames   P1's head camera on PUB 5565 (docs/contracts/p1_m2b.md §5.2; M1: the d435-mounted camera), read
                  with its own CONFLATE SUB so the frame metadata survives (P1.10): `cam_pose_wl`, `stationary`,
                  `hfov`, `vfov` come from the render that made the frame. A frame without them (an M1 P1) is
                  stamped with the world model's GT camera pose at receive time, as before, and says so
                  (`pose_stamp`). The chase/top views (PUB 5602) come through viz's FrameTap (docs/viz.md §9.2).
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

import base64
import math
import threading
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


class CameraSub(threading.Thread):
    """The newest frame of one P1 camera port (gear_sonic `sensor_server` msgpack + p1_m2b.md §5.2 metadata)."""

    def __init__(self, endpoint: str, key: str | None = None):
        super().__init__(name=f"camera-sub-{endpoint.rsplit(':', 1)[-1]}", daemon=True)
        self.endpoint = endpoint
        self.key = key
        self._lock = threading.Lock()
        self._frame: tuple[bytes, dict, int, float] | None = None       # (jpeg as sent, meta, rev, t_rx)
        self._running = True
        self.errors = 0

    def run(self) -> None:
        import msgpack
        import zmq
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(s, zmq.POLLIN)
        rev = 0
        while self._running:
            if not poller.poll(200):
                continue
            try:
                raw = s.recv(zmq.NOBLOCK)
                d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                imgs = d.get("images") or {}
                key = self.key or d.get("camera") or next(iter(imgs), None)
                b64 = imgs.get(key) if key in imgs else d.get(key) if key else None
                if b64 is None and imgs:
                    b64 = next(iter(imgs.values()))
                if b64 is None:
                    continue
                jpeg = base64.b64decode(b64) if isinstance(b64, str) else bytes(b64)
            except Exception:  # noqa: BLE001  (a bad message must not kill the SUB)
                self.errors += 1
                continue
            rev += 1
            meta = {k: d.get(k) for k in META_KEYS if k in d}
            meta.setdefault("t_capture", (d.get("timestamps") or {}).get(key))
            with self._lock:
                self._frame = (jpeg, meta, rev, time.time())
        s.close(0)

    def latest(self) -> tuple[bytes, dict, int, float] | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False


META_KEYS = ("camera", "seq", "render_seq", "t_sim", "t_capture", "t_capture_mono", "w", "h", "hfov", "vfov",
             "cam_pos", "cam_quat_wxyz", "cam_pose_wl", "stationary", "base_speed", "base_wz")


class IsaacFrames:
    NAMES = {"head": "head", "ego": "head", "chase": "chase", "top": "top"}
    MAX_STALE_S = 2.0

    def __init__(self, world: Any = None, port_offset: int | None = None, tap: Any = None, host: str = "127.0.0.1",
                 head_sub: Any = None):
        import os
        off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0) if port_offset is None else int(port_offset)
        self.port_offset = off
        self.world = world
        self._tap = tap
        self._fixed: tuple[int, bytes] | None = None          # (rev, R/B-fixed JPEG) of the head frame
        if head_sub is None and tap is None:
            head_sub = CameraSub(f"tcp://{host}:{5565 + off}")
            head_sub.start()
        self.head_sub = head_sub

    @property
    def tap(self):
        """viz FrameTap for the chase/top views, built on first use, never subscribing to gt.pose."""
        if self._tap is None:
            from viz.tap import FrameTap
            self._tap = FrameTap(port_offset=self.port_offset, gt_pose=False)
        return self._tap

    def latest(self, camera: str = "head"):
        name = self.NAMES.get(camera, camera)
        if name == "head" and self.head_sub is not None:
            return self._head()
        return self._from_tap(name)

    latest_frame = latest

    def _head(self):
        from api.types import CameraFrame
        got = self.head_sub.latest()
        if got is None:
            return None
        raw, meta, rev, t_rx = got
        if time.time() - t_rx > self.MAX_STALE_S:
            return None
        if self._fixed is None or self._fixed[0] != rev:
            from viz.common import swap_rb_jpeg          # cv2-encoded RGB (contract §5.2): fix R/B for viewers
            self._fixed = (rev, swap_rb_jpeg(raw, 85))
        jpeg = self._fixed[1]
        wl = meta.get("cam_pose_wl")
        if isinstance(wl, (list, tuple)) and len(wl) == 4:                    # P1.10: from the render itself
            pose_wl = tuple(float(v) for v in wl)
            stationary = bool(meta.get("stationary"))
            hfov, vfov = meta.get("hfov"), meta.get("vfov")
        else:                                                                 # M1 P1: stamped at receive time
            pose_wl, stationary, hfov, vfov = self._stamp()
        return CameraFrame(rev=int(rev), t_wall=float(meta.get("t_capture") or t_rx), jpeg=jpeg,
                           cam_pose_wl=pose_wl, stationary=stationary, camera="head",   # type: ignore[arg-type]
                           hfov=None if hfov is None else float(hfov), vfov=None if vfov is None else float(vfov),
                           w=int(meta.get("w") or 0), h=int(meta.get("h") or 0))

    def pose_stamp(self) -> str:
        """Where the head frames' camera poses come from: "render" (P1.10) or "receive-time" (M1 P1)."""
        got = self.head_sub.latest() if self.head_sub is not None else None
        return "render" if got is not None and got[1].get("cam_pose_wl") else "receive-time"

    def _stamp(self):
        if self.world is None:
            return (0.0, 0.0, 0.0, 0.0), False, None, None
        cp = self.world.camera_pose()
        mx, mz = coords.to_map_xz(cp.pos[0], cp.pos[1])
        pose_wl = (round(mx, 3), round(mz, 3), round(coords.yaw_map_deg(cp.yaw), 1),
                   round(math.degrees(cp.pitch_down), 1))
        rp = self.world.robot_pose()
        stationary = rp.speed < 0.05 and abs(rp.wz) < 0.05
        return pose_wl, stationary, self.world.cam.hfov_deg, self.world.cam.vfov_deg

    def _from_tap(self, name: str):
        from api.types import CameraFrame
        jpeg = self.tap.jpeg(name)
        if jpeg is None:
            return None
        meta = self.tap.meta(name)
        pose_wl, stationary, hfov, vfov = ((0.0, 0.0, 0.0, 0.0), False, None, None)
        if name == "head":
            pose_wl, stationary, hfov, vfov = self._stamp()
        return CameraFrame(rev=int(self.tap.rev(name)), t_wall=float(meta.get("t_wall") or time.time()), jpeg=jpeg,
                           cam_pose_wl=pose_wl, stationary=stationary, camera="head" if name == "head" else "top",
                           hfov=hfov, vfov=vfov, w=int(meta.get("w") or meta.get("width") or 0),
                           h=int(meta.get("h") or meta.get("height") or 0))

    def close(self) -> None:
        if self.head_sub is not None and hasattr(self.head_sub, "stop"):
            self.head_sub.stop()
        close = getattr(self._tap, "close", None)
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

