"""Small camera helpers for evidence images (Isaac Sim 5.1, omni.replicator.core render products).

    cam = Camera(stage, "/World/Cams/eye", (640, 480), hfov_deg=70)
    cam.look_at((x, y, 1.2), (tx, ty, 1.0))
    rgb = cam.grab(sim)            # HxWx3 uint8, after a few render-only app updates

`TopDown.for_bounds(...)` places a narrow-FOV perspective camera straight above the house and
records the floor-plane pixel<->world mapping in its meta (image up = +y world, right = +x).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    from pxr import Gf

    e = np.asarray(eye, float)
    f = np.asarray(target, float) - e
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    if abs(np.dot(f, u)) > 0.999:  # looking straight down/up: use +y as the image-up hint
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    u2 = np.cross(r, f)
    # USD cameras look down -Z with +Y up; rows are the camera axes in world (row-vector convention)
    return Gf.Matrix4d(
        r[0], r[1], r[2], 0.0,
        u2[0], u2[1], u2[2], 0.0,
        -f[0], -f[1], -f[2], 0.0,
        e[0], e[1], e[2], 1.0,
    )


class Camera:
    def __init__(self, stage, path: str, resolution=(640, 480), hfov_deg: float = 70.0, clip=(0.05, 200.0)):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.stage = stage
        self.path = path
        self.resolution = tuple(resolution)
        self.hfov_deg = hfov_deg
        cam = UsdGeom.Camera.Define(stage, path)
        ha = 20.955
        w, h = self.resolution
        cam.CreateHorizontalApertureAttr().Set(ha)
        cam.CreateVerticalApertureAttr().Set(ha * h / w)
        cam.CreateFocalLengthAttr().Set(ha / 2.0 / math.tan(math.radians(hfov_deg) / 2.0))
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(*clip))
        self._xf = UsdGeom.Xformable(cam.GetPrim())
        self._xf.ClearXformOpOrder()
        self._op = self._xf.AddTransformOp()
        self.rp = rep.create.render_product(path, self.resolution)
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot.attach([self.rp])

    def look_at(self, eye, target, up=(0.0, 0.0, 1.0)) -> None:
        self._op.Set(_look_at_matrix(eye, target, up))

    def latest(self) -> np.ndarray | None:
        d = self.annot.get_data()
        if d is None or getattr(d, "size", 0) == 0:
            return None
        return np.asarray(d)[..., :3].copy()

    def grab(self, sim, n_updates: int = 12) -> np.ndarray | None:
        """Render-only updates (SimulationContext.render() does not step physics)."""
        for _ in range(n_updates):
            sim.render()
        return self.latest()

    def destroy(self) -> None:
        try:
            self.annot.detach([self.rp])
            self.rp.destroy()
        except Exception:
            pass


class TopDown(Camera):
    @classmethod
    def for_bounds(cls, stage, path, bounds_xy, px: int = 1024, hfov_deg: float = 20.0, margin: float = 0.6, floor_z: float = 0.0):
        (x0, y0), (x1, y1) = bounds_xy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        span = max(x1 - x0, y1 - y0) + 2 * margin
        height = span / (2 * math.tan(math.radians(hfov_deg) / 2))
        cam = cls(stage, path, (px, px), hfov_deg=hfov_deg, clip=(1.0, height + 50))
        cam.look_at((cx, cy, floor_z + height), (cx, cy, floor_z))
        cam.meta = {
            "center_xy": [cx, cy],
            "height_above_floor_m": height,
            "hfov_deg": hfov_deg,
            "px": px,
            "floor_z": floor_z,
            "m_per_px_at_floor": span / px,
            "mapping": "x = cx + (u - px/2 + 0.5) * m_per_px ; y = cy - (v - px/2 + 0.5) * m_per_px  (u right, v down; exact on the floor plane only)",
        }
        return cam

    def save_meta(self, path: Path) -> None:
        path.write_text(json.dumps(self.meta, indent=2))


def save_rgb(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)
