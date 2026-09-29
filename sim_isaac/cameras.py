"""CameraRig: P1's robot cameras (docs/contracts/p1_m2b.md §5, owner decision OD1).

    head      System 1 / scans / the UI head pane / M1 tools: torso + (0.06, 0, 0.526), 15 deg down, HFOV 90,
              640x480, PUB 5565 key "head", on by default at --camera-hz
    ego_view  GR00T: Arena's G1 head camera exactly (sim_isaac.wire.EGO_VIEW, cited there), 640x480, PUB 5566 key
              "ego_view", OFF until a consumer enables it (`camera` / `set_render_rates` REP ops)
    d435      M1's camera (optional, --stream-camera d435)

Every camera is one Replicator render product + rgb annotator (sim_isaac.camera.RgbCapture) on a USD camera prim
under the torso link. Rendering is shared: one `sim.render()` renders EVERY enabled product (docs/viz.md §4 finding
2), so the rig renders when any enabled camera is due on its sim-time grid (k / hz) and reads the ones that are due.
A disabled product costs nothing; enabling re-arms it (annotator detach/enable/attach, finding 3) and skips the
first `warmup_frames` frames while the RTX denoiser settles (finding 6).

Per frame the sim thread does: one annotator read + one contiguous RGBA copy (~0.1 ms) + the camera pose from the
torso link pose of the physics step that preceded the render. JPEG encoding and the send run on the camera's
FramePublisher worker thread.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from sim_isaac.wire import Consumers, OpError, frame_meta, next_due, r


def prim_path_of(spec) -> str:
    return f"/World/G1/{spec.parent}/wl_cam_{spec.name}"


class RigCamera:
    def __init__(self, spec, prim_path: str, cap, pub, hz: float):
        self.spec = spec
        self.prim_path = prim_path
        self.cap = cap
        self.pub = pub
        self.hz = float(hz)
        self.consumers = Consumers()
        self.on = True           # render products start enabled; the rig disables the ones nobody wants
        self.next_t = 0.0
        self.seq = 0
        self.warmup_left = 0
        self.rearm_ms: float | None = None
        self.skipped = 0
        self.seg = None          # segment.SegAnnotator while `detections` keeps it attached
        self.seg_until = 0.0
        self.last_meta: dict | None = None

    def info(self, port: int | None) -> dict:
        d = self.spec.info()
        d.update({"on": self.on, "hz": self.hz, "port": port, "consumers": self.consumers.names(),
                  "frames": self.seq, "warmup_skipped": self.skipped,
                  "pub_hz": round(self.pub.rate(), 1) if self.pub else 0.0,
                  "dropped": self.pub.dropped if self.pub else 0,
                  "rearm_ms": None if self.rearm_ms is None else round(self.rearm_ms, 1),
                  "seg_attached": self.seg is not None})
        return d


class CameraRig:
    def __init__(self, sim, stage, specs: list, ports: dict[str, int], zctx, *, hz: dict[str, float],
                 jpeg_q: int = 80, warmup_frames: int = 4, seg_keep_s: float = 3.0,
                 log: Callable[[str], None] = print, event: Callable[..., None] | None = None):
        from sim_isaac.camera import FramePublisher, RgbCapture

        self.sim, self.stage, self.log = sim, stage, log
        self.event = event or (lambda *a, **k: None)
        self.warmup_frames = int(warmup_frames)
        self.seg_keep_s = float(seg_keep_s)
        self.ports = ports
        self.cams: dict[str, RigCamera] = {}
        self.render_seq = 0
        self.render_calls = 0
        for spec in specs:          # prims already spawned (before sim.reset) at prim_path_of(spec)
            path = prim_path_of(spec)
            cap = RgbCapture(path, spec.width, spec.height)
            pub = FramePublisher(zctx, ports[spec.port], spec.key, jpeg_q=jpeg_q)
            self.cams[spec.name] = RigCamera(spec, path, cap, pub, hz.get(spec.name, spec.default_hz))
            log(f"[cameras] {spec.name}: {spec.width}x{spec.height} HFOV {spec.hfov_deg:.2f} VFOV {spec.vfov_deg:.2f} "
                f"pitch {spec.pitch_down_deg:.2f} deg down, {spec.parent} + {tuple(round(v, 5) for v in spec.mount_xyz)}"
                f", PUB :{ports[spec.port]} key {spec.key!r}, {'on' if spec.default_on else 'off'} "
                f"@ {self.cams[spec.name].hz:g} Hz")

    # ------------------------------------------------------------------ start-up
    def finish_warmup(self, t_sim: float) -> None:
        """After the host's start-up warm-up renders (all products enabled): keep the default-on cameras, disable the
        rest (zero render cost until a consumer enables them)."""
        for c in self.cams.values():
            if c.spec.default_on:
                c.consumers.add("default", time.monotonic())
                c.next_t = next_due(t_sim, c.hz)
            else:
                c.cap.disable()
                c.on = False

    # ------------------------------------------------------------------ loop
    def due(self, t_sim: float) -> list[RigCamera]:
        return [c for c in self.cams.values() if c.on and t_sim + 1e-9 >= c.next_t]

    def render(self) -> int:
        self.sim.render()
        self.render_seq += 1
        self.render_calls += 1
        return self.render_seq

    def harvest(self, cams: list[RigCamera], st: dict, t_sim: float, render_seq: int) -> None:
        """Read the cameras that were due in this render and publish them with their metadata."""
        now_w, now_m = time.time(), time.monotonic()
        for c in cams:
            c.next_t = next_due(t_sim, c.hz)
            if c.warmup_left > 0:
                c.warmup_left -= 1
                c.skipped += 1
                continue
            self._publish(c, st, t_sim, render_seq, now_w, now_m)

    def _publish(self, c: RigCamera, st: dict, t_sim: float, render_seq: int, now_w: float, now_m: float) -> bool:
        img = c.cap.read_copy()
        if img is None:
            return False
        c.seq += 1
        pos, quat = c.spec.world_pose(st["torso_pos"], st["torso_quat"])
        meta = frame_meta(c.spec, seq=c.seq, render_seq=render_seq, t_sim=t_sim, t_capture=now_w, t_capture_mono=now_m,
                          cam_pos=pos, cam_quat=quat, base_lin_vel_w=st["lin_w"], base_ang_vel_w=st["ang_w"],
                          jpeg_q=c.pub.jpeg_q)
        c.last_meta = meta
        c.pub.submit(img, t_sim, c.seq, meta)
        return True

    def housekeeping(self, t_sim: float) -> None:
        """Consumer TTLs and the segmentation annotator's keep-alive (call a few times per second)."""
        now = time.monotonic()
        for c in self.cams.values():
            gone = c.consumers.expire(now)
            if gone:
                self.log(f"[cameras] {c.spec.name}: consumer(s) {gone} expired (ttl)")
                self._apply(c, t_sim, reason="ttl")
                self.event("camera", name=c.spec.name, on=c.on, hz=c.hz, consumers=c.consumers.names(),
                           expired=gone)
            if c.seg is not None and now > c.seg_until:
                c.seg.detach()
                c.seg = None

    # ------------------------------------------------------------------ control
    def camera(self, name: str, t_sim: float, *, on: bool | None = None, hz: float | None = None,
               consumer: str | None = None, ttl_s: float | None = None) -> dict:
        c = self.cams.get(name)
        if c is None:
            raise OpError("unknown_camera", f"{name!r} (have {sorted(self.cams)})")
        before = (c.on, c.hz, tuple(c.consumers.names()))
        if hz is not None:
            hz = float(hz)
            if not hz > 0:
                raise OpError("bad_arg", "hz must be > 0 (use on:false to stop a camera)")
            c.hz = hz
            c.next_t = next_due(t_sim, hz)
        if on is True:
            c.consumers.add(str(consumer or "anon"), time.monotonic(), ttl_s)
        elif on is False:
            c.consumers.remove(None if consumer is None else str(consumer))
        self._apply(c, t_sim, reason="op")
        if (c.on, c.hz, tuple(c.consumers.names())) != before:
            self.event("camera", name=name, on=c.on, hz=c.hz, consumers=c.consumers.names())
        return self.cam_reply(c)

    def set_render_rates(self, t_sim: float, head_hz: float | None, ego_hz: float | None) -> dict:
        out = {}
        for name, hz in (("head", head_hz), ("ego_view", ego_hz)):
            if hz is None or name not in self.cams:
                continue
            if float(hz) > 0:
                self.camera(name, t_sim, on=True, hz=float(hz),
                            consumer="default" if name == "head" else "set_render_rates")
            else:
                self.camera(name, t_sim, on=False)
            out[name] = self.cam_reply(self.cams[name])
        return out

    def _apply(self, c: RigCamera, t_sim: float, reason: str) -> bool:
        """Switch the render product to match its consumers; True when on/off changed."""
        want = bool(c.consumers)
        if want and not c.on:
            c.rearm_ms = c.cap.enable()
            c.on = True
            c.warmup_left = self.warmup_frames
            c.next_t = next_due(t_sim, c.hz)
        elif not want and c.on:
            c.cap.disable()
            c.on = False
            if c.seg is not None:
                c.seg.detach()
                c.seg = None
        else:
            return False
        self.log(f"[cameras] {c.spec.name} {'ON' if c.on else 'OFF'} ({reason}) hz={c.hz:g} "
                 f"consumers={c.consumers.names()}" + (f" rearm {c.rearm_ms:.1f} ms" if c.on else ""))
        return True

    def cam_reply(self, c: RigCamera) -> dict:
        return {"name": c.spec.name, "on": c.on, "hz": c.hz, "consumers": c.consumers.names(),
                "port": self.ports[c.spec.port], "key": c.spec.key, "warmup_frames": self.warmup_frames}

    # ------------------------------------------------------------------ detections (P1.6)
    def detect(self, name: str, st: dict, t_sim: float, index, *, min_px: int = 40, max_range: float | None = None,
               ids: set | None = None, bbox: bool = True, objects_pos: dict | None = None,
               held: dict | None = None) -> dict:
        from sim_isaac.segment import SegAnnotator, count_instances

        c = self.cams.get(name)
        if c is None:
            raise OpError("unknown_camera", f"{name!r}")
        if not c.on:
            raise OpError("camera_off", f"{name} is off; enable it with the `camera` op first")
        t0 = time.perf_counter()
        if c.seg is None:
            try:
                c.seg = SegAnnotator(c.cap.rp_path)
            except Exception as e:  # noqa: BLE001
                raise OpError("segmentation_unavailable", f"{type(e).__name__}: {e}") from e
        c.seg_until = time.monotonic() + self.seg_keep_s
        seg, id_to_path = None, {}
        rs = self.render_seq
        for _ in range(4):   # the first render after attaching the annotator may come back empty
            rs = self.render()
            seg, id_to_path = c.seg.read()
            if seg is not None and id_to_path:
                break
        if seg is None:
            raise OpError("segmentation_unavailable", "annotator returned no data after 4 renders")
        now_w, now_m = time.time(), time.monotonic()
        self._publish(c, st, t_sim, rs, now_w, now_m)
        dets, other = count_instances(seg, id_to_path, index, min_px=int(min_px), want_bbox=bool(bbox), ids=ids)
        pos, _q = c.spec.world_pose(st["torso_pos"], st["torso_quat"])
        out = []
        for d in dets:
            p = (objects_pos or {}).get(d["id"])
            dist = None if p is None else float(np.linalg.norm(np.asarray(p) - pos))
            if max_range is not None and dist is not None and dist > float(max_range):
                continue
            d["dist_m"] = r(dist, 3)
            d["held_by"] = (held or {}).get(d["id"])
            out.append(d)
        meta = c.last_meta or {}
        return {"camera": name, "t_sim": round(t_sim, 4), "render_seq": rs, "frame_seq": c.seq,
                "w": c.spec.width, "h": c.spec.height, "method": "instance_id_segmentation_fast",
                "min_px": int(min_px), "detections": out, "other_px": other,
                "cam_pose_wl": meta.get("cam_pose_wl"), "cam_pos": meta.get("cam_pos"),
                "ms": round((time.perf_counter() - t0) * 1e3, 2)}

    # ------------------------------------------------------------------ reporting
    def pose_of(self, name: str, st: dict) -> tuple[np.ndarray, np.ndarray] | None:
        c = self.cams.get(name)
        return None if c is None else c.spec.world_pose(st["torso_pos"], st["torso_quat"])

    def info(self) -> list[dict]:
        return [c.info(self.ports[c.spec.port]) for c in self.cams.values()]

    def on_map(self) -> dict[str, bool]:
        return {n: c.on for n, c in self.cams.items()}

    def stats(self) -> dict:
        return {n: {"on": c.on, "hz": c.hz, "frames": c.seq, "pub_hz": round(c.pub.rate(), 1),
                    "dropped": c.pub.dropped, "consumers": c.consumers.names(), "seg_attached": c.seg is not None}
                for n, c in self.cams.items()}

    def close(self) -> None:
        for c in self.cams.values():
            if c.seg is not None:
                c.seg.detach()
            c.pub.close()
