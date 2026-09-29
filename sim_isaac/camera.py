"""Cameras for wl-isaac: the head (ego) camera, its gear_sonic-format ZMQ publisher, a top-down renderer and an
optional third-person chase camera.

Wire format of the ego stream is byte-compatible with the MuJoCo sim publisher:
  $WBC/gear_sonic/utils/mujoco_sim/image_publish_utils.py:165-185 (message dict: images/timestamps + a top-level
  copy under the camera name), sensor_server.py:57-77 (msgpack use_bin_type=True, JPEG q80 via cv2.imencode,
  base64 str). The RGB array is passed to cv2.imencode unchanged, as MuJoCo does, so ImageUtils.decode_image
  (cv2.imdecode) returns RGB again.
"""
from __future__ import annotations

import base64
import math
import queue
import threading
import time

import numpy as np


def _pinhole_cfg(width: int, height: int, vfov_deg: float, near: float = 0.05, far: float = 50.0):
    import isaaclab.sim as sim_utils

    f = 10.0
    va = 2.0 * f * math.tan(math.radians(vfov_deg) / 2.0)
    ha = va * width / height  # square pixels
    return sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=ha, vertical_aperture=va,
                                      clipping_range=(near, far))


def spawn_camera_prim(prim_path: str, width: int, height: int, vfov_deg: float, xyz, quat_world_wxyz,
                      near: float = 0.05, far: float = 50.0, apertures: tuple[float, float, float] | None = None) -> str:
    """Create a USD camera at prim_path. quat_world_wxyz is the mount orientation in the 'world' convention
    (x forward, z up), converted here to the USD/OpenGL camera convention (-z forward, y up).
    apertures = (focal_length, horizontal_aperture, vertical_aperture) overrides the vfov-derived pinhole."""
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    q = convert_camera_frame_orientation_convention(
        torch.tensor([list(quat_world_wxyz)], dtype=torch.float32), origin="world", target="opengl")[0]
    if apertures is None:
        cfg = _pinhole_cfg(width, height, vfov_deg, near, far)
    else:
        import isaaclab.sim as sim_utils

        f, ha, va = apertures
        cfg = sim_utils.PinholeCameraCfg(focal_length=float(f), horizontal_aperture=float(ha),
                                         vertical_aperture=float(va), clipping_range=(near, far))
    cfg.func(prim_path, cfg, translation=tuple(float(x) for x in xyz), orientation=tuple(float(x) for x in q))
    return prim_path


def spawn_spec_camera(prim_path: str, spec) -> str:
    """A sim_isaac.wire.CameraSpec camera (explicit focal length and apertures, e.g. Arena's head camera)."""
    return spawn_camera_prim(prim_path, spec.width, spec.height, spec.vfov_deg, spec.mount_xyz, spec.mount_quat_wxyz,
                             near=spec.clipping[0], far=spec.clipping[1],
                             apertures=(spec.focal_length_mm, spec.horizontal_aperture_mm, spec.vertical_aperture_mm))


class RgbCapture:
    """A replicator render product + rgb annotator on an existing camera prim."""

    def __init__(self, cam_path: str, width: int, height: int):
        import omni.replicator.core as rep

        self.cam_path, self.width, self.height = cam_path, width, height
        rp = rep.create.render_product(cam_path, resolution=(width, height))
        self._rp = rp
        self.rp_path = rp if isinstance(rp, str) else rp.path
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annot.attach(self.rp_path)
        self.enabled = True

    def set_enabled(self, on: bool) -> bool:
        """Enable/disable rendering of this render product (skips its GPU work while disabled)."""
        if bool(on) == self.enabled:
            return True
        try:
            self._rp.hydra_texture.set_updates_enabled(bool(on))
            self.enabled = bool(on)
            return True
        except Exception:  # noqa: BLE001  (older API / str render product)
            return False

    def disable(self) -> bool:
        return self.set_enabled(False)

    def enable(self) -> float:
        """Re-arm a disabled render product: a product that was ever disabled only delivers annotator data again
        after its annotator is re-attached (docs/viz.md §4 finding 3). Returns the sim-thread cost in ms."""
        t0 = time.perf_counter()
        if not self.enabled:
            try:
                self.annot.detach([self.rp_path])
            except Exception:  # noqa: BLE001
                pass
            self._rp.hydra_texture.set_updates_enabled(True)
            self.annot.attach([self.rp_path])
            self.enabled = True
        return (time.perf_counter() - t0) * 1e3

    def read(self) -> np.ndarray | None:
        d = self.annot.get_data()
        if isinstance(d, dict):
            d = d.get("data")
        if d is None:
            return None
        a = np.asarray(d)
        if a.size == 0 or a.ndim != 3:
            return None
        return a[..., :3]

    def read_copy(self) -> np.ndarray | None:
        """A contiguous copy of the annotator's buffer (RGBA or RGB) before the next render overwrites it: one memcpy
        (~0.1 ms at 640x480); the strided RGB slice costs ~1.7 ms (docs/viz.md §7.2) and is left to the worker."""
        d = self.annot.get_data()
        if isinstance(d, dict):
            d = d.get("data")
        if d is None:
            return None
        a = np.asarray(d)
        if a.size == 0 or a.ndim != 3:
            return None
        return np.array(a, copy=True, order="C")

    def destroy(self) -> None:
        try:
            self.annot.detach([self.rp_path])
        except Exception:  # noqa: BLE001
            pass
        try:
            self._rp.destroy()
        except Exception:  # noqa: BLE001
            pass


def encode_b64_jpeg(rgb: np.ndarray, quality: int = 80) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf).decode("utf-8")


class FramePublisher:
    """Encodes frames on a worker thread (cv2 releases the GIL) and publishes them on a ZMQ PUB socket.

    mode="gear_sonic": the MuJoCo sensor_server message (see module docstring), one frame per message. With `extra`
    (the M2b frame metadata, sim_isaac.wire.frame_meta) the message also carries those keys and `timestamps` is the
    capture time (docs/contracts/p1_m2b.md §5.2).
    mode="multipart": [topic, msgpack{seq, t_sim, t_wall, jpeg(bytes), ...extra}] for non-gear_sonic consumers.
    Frames may be RGB or RGBA (alpha dropped here, off the sim thread).
    """

    def __init__(self, ctx, port: int, name: str = "ego_view", mode: str = "gear_sonic", topic: bytes = b"",
                 bind_host: str = "127.0.0.1", jpeg_q: int = 80):
        import zmq

        self.name, self.mode, self.topic, self.jpeg_q = name, mode, topic, int(jpeg_q)
        self.sock = ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 20)   # sensor_server.py:45-46
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://{bind_host}:{port}")
        self.port = port
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.sent = 0
        self.dropped = 0
        self.encode_ms = []
        self._send_times: list[float] = []
        self._stop = threading.Event()
        self.last_jpeg: bytes | None = None
        self.th = threading.Thread(target=self._run, name=f"pub-{name}", daemon=True)
        self.th.start()

    def submit(self, rgb: np.ndarray, t_sim: float, seq: int, extra: dict | None = None) -> None:
        try:
            self.q.put_nowait((rgb, t_sim, seq, extra))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        import msgpack
        import zmq

        while not self._stop.is_set():
            try:
                rgb, t_sim, seq, extra = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            b64 = encode_b64_jpeg(rgb, self.jpeg_q)
            now = time.time()
            if self.mode == "gear_sonic" and extra:
                from sim_isaac.wire import gear_sonic_message
                parts = [msgpack.packb(gear_sonic_message(self.name, b64, extra, now), use_bin_type=True)]
            elif self.mode == "gear_sonic":
                msg = {"timestamps": {self.name: now}, "images": {self.name: b64}, self.name: b64,
                       "t_sim": t_sim, "seq": seq}
                parts = [msgpack.packb(msg, use_bin_type=True)]
            else:
                body = {"seq": seq, "t_sim": t_sim, "t_wall": now, "jpeg": base64.b64decode(b64)}
                if extra:
                    body.update(extra)
                parts = [self.topic, msgpack.packb(body, use_bin_type=True)]
            self.last_jpeg = base64.b64decode(b64) if self.mode == "gear_sonic" else body["jpeg"]
            try:
                self.sock.send_multipart(parts, flags=zmq.NOBLOCK)
                self.sent += 1
                self._send_times.append(time.perf_counter())
                if len(self._send_times) > 600:
                    del self._send_times[:300]
            except zmq.Again:
                self.dropped += 1
            self.encode_ms.append((time.perf_counter() - t0) * 1e3)
            if len(self.encode_ms) > 600:
                del self.encode_ms[:300]

    def rate(self, window_s: float = 2.0) -> float:
        now = time.perf_counter()
        return sum(1 for t in list(self._send_times) if t >= now - window_s) / window_s

    def close(self):
        self._stop.set()
        self.th.join(timeout=1.0)
        self.sock.close(0)


class ChaseCamera:
    """Third-person camera that follows the pelvis (sim-only visualisation, not a robot sensor)."""

    def __init__(self, stage, prim_path: str, width: int = 640, height: int = 480, vfov_deg: float = 55.0,
                 back: float = 2.2, up: float = 1.3, look_z: float = 0.7):
        self.stage, self.prim_path = stage, prim_path
        self.back, self.up, self.look_z = back, up, look_z
        spawn_camera_prim(prim_path, width, height, vfov_deg, (0, 0, 2), (1, 0, 0, 0), near=0.05, far=100.0)
        self.cap = RgbCapture(prim_path, width, height)
        self._yaw_f = None

    def update_pose(self, base_pos, yaw: float) -> None:
        """Place the camera behind the robot (smoothed yaw) looking at it."""
        import torch
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
        from pxr import Gf, UsdGeom

        if self._yaw_f is None:
            self._yaw_f = yaw
        d = math.atan2(math.sin(yaw - self._yaw_f), math.cos(yaw - self._yaw_f))
        self._yaw_f += 0.15 * d
        cy, sy = math.cos(self._yaw_f), math.sin(self._yaw_f)
        eye = np.array([base_pos[0] - self.back * cy, base_pos[1] - self.back * sy, base_pos[2] + self.up])
        tgt = np.array([base_pos[0], base_pos[1], self.look_z])
        fwd = tgt - eye
        fwd /= np.linalg.norm(fwd)
        yaw_c = math.atan2(fwd[1], fwd[0])
        pitch_c = -math.asin(max(-1.0, min(1.0, fwd[2])))
        # world-convention quaternion: yaw about z then pitch about y
        qy = np.array([math.cos(yaw_c / 2), 0, 0, math.sin(yaw_c / 2)])
        qp = np.array([math.cos(pitch_c / 2), 0, math.sin(pitch_c / 2), 0])
        from .mathutil import quat_mul
        qw = quat_mul(qy, qp)
        q = convert_camera_frame_orientation_convention(
            torch.tensor([qw.tolist()], dtype=torch.float32), origin="world", target="opengl")[0].tolist()
        prim = self.stage.GetPrimAtPath(self.prim_path)
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        t_op = ops.get("xformOp:translate")
        o_op = ops.get("xformOp:orient")
        if t_op is None or o_op is None:
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp()
            o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(*[float(x) for x in eye]))
        val = Gf.Quatd(q[0], q[1], q[2], q[3]) if "Double" in str(o_op.GetPrecision()) else Gf.Quatf(*q)
        o_op.Set(val)


def render_topdown(sim, stage, extent: tuple[float, float, float, float], path: str, px_per_m: float = 40.0,
                   max_px: int = 1600, height_m: float = 60.0, hide_paths: list[str] | None = None) -> dict:
    """Render a top-down image of the scene with a temporary perspective camera high above the centre.

    Perspective from `height_m` above: objects at floor level map exactly; tall objects are displaced outwards by
    about (distance from centre) * h / height_m. `hide_paths`: prims made invisible for this render only (the
    furniture-only mode hides the dynamic props and the robot, docs/contracts/p1_m2b.md §9).
    """
    if hide_paths:
        from pxr import UsdGeom

        hidden = []
        for p in hide_paths:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                img_api = UsdGeom.Imageable(prim)
                if img_api.ComputeVisibility() != UsdGeom.Tokens.invisible:
                    img_api.MakeInvisible()
                    hidden.append(img_api)
        try:
            info = render_topdown(sim, stage, extent, path, px_per_m, max_px, height_m)
        finally:
            for img_api in hidden:
                img_api.MakeVisible()
        info["hidden"] = len(hidden)
        return info
    import cv2

    xmin, ymin, xmax, ymax = extent
    wx, wy = xmax - xmin, ymax - ymin
    scale = min(px_per_m, max_px / max(wx, wy))
    w = int(round(wx * scale)) // 2 * 2
    h = int(round(wy * scale)) // 2 * 2
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    vfov = math.degrees(2 * math.atan((wy / 2) / height_m))
    path_prim = "/World/wl_topdown_cam"
    # looking straight down with image-up = world +y: world-convention camera pitched +90 deg (x fwd -> -z),
    # then yawed so that image up is +y.
    qy = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])       # yaw +90: forward = +y
    qp = np.array([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])       # pitch +90: forward -> down
    from .mathutil import quat_mul
    q = quat_mul(qy, qp)
    if not stage.GetPrimAtPath(path_prim).IsValid():
        spawn_camera_prim(path_prim, w, h, vfov, (cx, cy, height_m), q, near=1.0, far=height_m + 20.0)
    cap = RgbCapture(path_prim, w, h)
    img = None
    try:
        for _ in range(6):
            sim.render()
            img = cap.read()
        if img is None:
            raise RuntimeError("top-down render returned no data")
        cv2.imwrite(path, img[..., ::-1])
    finally:
        cap.destroy()
    return {"path": path, "width": w, "height": h, "center": [cx, cy], "meters_per_pixel": 1.0 / scale,
            "extent": [xmin, ymin, xmax, ymax]}
