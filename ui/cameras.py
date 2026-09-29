"""Where the page's camera panes come from.

On Isaac profiles the frames come from viz's FrameTap (docs/viz.md 9.2): the head
camera on SUB 5565 (gear_sonic msgpack, R/B fixed by the tap), and the chase / top /
overview cameras on SUB 5602 (VizCams, or P1's own --tp-camera as "chase"). The top
view's meta carries `extent` [xmin, ymin, xmax, ymax] (Isaac xy = Worldline x, z), which
the page uses to lay the image under the ground-truth map.

On `lite` the world model renders its own frames (`world.latest_frame(camera)`), so
`WorldCameras` wraps that. Both sources have the same four calls: names, rev, jpeg, meta.

Frames are pictures, not ground truth (GT confinement: on the runtime side only world/ reads ground truth).
`TapCameras` builds its FrameTap with `gt_pose=False`, so the page never subscribes to gt.pose (5601), and
`meta()` drops the pose fields some publishers attach to a frame (`GT_META`). The robot pose the page draws comes
from `world.truth()`.

Captions say which camera a pane shows. The head pane on the G1 profiles is the sim-added head camera
(PLAN 12.2: "camera: head (sim-added)"); the ego pane is GR00T's `ego_view` (OD1: Arena's G1 head camera, rendered
only while a consumer enables it).
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

CAPTIONS = {
    "head": "head camera · what System 1 sees",
    "ego": "camera: ego_view (Arena G1 head camera) · what GR00T sees",
    "chase": "chase camera · sim viz, not a robot sensor",
    "top": "top view · sim render (ground truth)",
    "overview": "overview · sim render (ground truth)",
}
# The panes the page offers, in tab order. Anything else a source publishes is offered after them.
PANES = ("head", "chase", "top", "ego")
# Most frames per second the page gets for each pane (the THOR page sent head <= 5 Hz, top <= 1 Hz).
MAX_HZ = {"head": 5.0, "chase": 5.0, "ego": 2.0, "top": 1.0, "overview": 1.0}
# Frame metadata that is ground truth (the robot's or a camera's sim pose, GT-derived motion flags). A publisher
# may attach them (P1 frame.tp: base_pos/yaw -> robot; VizCams: cam_pose); the page never passes them on.
GT_META = frozenset({"robot", "cam_pose", "cam_pose_wl", "base_pos", "yaw", "pose", "stationary"})
SIM_ADDED_CAMERAS = frozenset({"head_sim"})         # camera models a stock G1 does not have (world/perception.py)


def head_caption(meta: dict[str, Any] | None = None, camera: dict[str, Any] | None = None,
                 profile: str | None = None) -> str:
    """The head pane's caption. `camera` is the robot map's {name, sim_added} for the profile's perception camera
    (robot.lookup_keypoints()["camera"]: head_sim or ego_d435); a frame's own `sim_added` flag wins, and its
    `camera` names the model only when the map gives none (CameraFrame.camera is the pane name, "head"). The
    sim-added head camera is labelled as such (PLAN 12.2, docs/parity.md). On lite the frame is a schematic of that
    camera's GT detections, not a render."""
    meta, camera = dict(meta or {}), dict(camera or {})
    name = str(camera.get("name") or (meta.get("camera") if meta.get("camera") != "head" else "") or "")
    sim_added = meta.get("sim_added")
    if sim_added is None:
        sim_added = camera.get("sim_added") if camera.get("sim_added") is not None else name in SIM_ADDED_CAMERAS
    if sim_added:
        cap = "camera: head (sim-added) · what System 1 sees"
    elif name:
        cap = f"camera: head ({name}) · what System 1 sees"
    else:
        cap = CAPTIONS["head"]
    if profile == "lite" and meta.get("source") == "world":
        cap += " · lite schematic, not a render"
    return cap


class CameraSource(Protocol):
    def names(self) -> list[str]: ...
    def rev(self, name: str) -> int: ...
    def jpeg(self, name: str) -> bytes | None: ...
    def meta(self, name: str) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _order(names: list[str]) -> list[str]:
    return [n for n in PANES if n in names] + sorted(n for n in names if n not in PANES)


class TapCameras:
    """viz.tap.FrameTap as a CameraSource. Pass a tap (tests pass a fake), or a port offset."""

    def __init__(self, tap: Any = None, port_offset: int | None = None) -> None:
        if tap is None:
            from viz.tap import FrameTap          # pyzmq + msgpack + Pillow; only on the Isaac profiles
            tap = FrameTap(port_offset=port_offset, gt_pose=False)     # frames only: no gt.pose (5601)
        self.tap = tap

    def names(self) -> list[str]:
        try:
            got = list(self.tap.streams())
        except Exception:  # noqa: BLE001
            got = []
        return _order(got)

    def rev(self, name: str) -> int:
        return int(self.tap.rev(name) or 0)

    def jpeg(self, name: str) -> bytes | None:
        return self.tap.jpeg(name)

    def meta(self, name: str) -> dict[str, Any]:
        m = {k: v for k, v in (self.tap.meta(name) or {}).items() if k not in GT_META}
        m.setdefault("source", "viz.tap")
        age = getattr(self.tap, "age_s", None)
        if callable(age):
            a = age(name)
            m["age_s"] = None if a is None else round(float(a), 2)
        return m

    def close(self) -> None:
        close = getattr(self.tap, "close", None)
        if callable(close):
            close()


class WorldCameras:
    """world.latest_frame(camera) -> CameraFrame (api/types.py) as a CameraSource (lite, or a
    FrameSource from robot.factory.build)."""

    def __init__(self, world: Any, cameras: tuple[str, ...] = ("head", "top", "ego")) -> None:
        self.world, self.cameras = world, cameras

    def _frame(self, name: str) -> Any:
        fn = getattr(self.world, "latest_frame", None) or getattr(self.world, "latest", None)
        if not callable(fn):
            return None
        try:
            return fn(name)
        except Exception:  # noqa: BLE001
            return None

    def names(self) -> list[str]:
        return _order([c for c in self.cameras if self._frame(c) is not None])

    def rev(self, name: str) -> int:
        f = self._frame(name)
        return int(getattr(f, "rev", 0) or 0) if f is not None else 0

    def jpeg(self, name: str) -> bytes | None:
        f = self._frame(name)
        return getattr(f, "jpeg", None) if f is not None else None

    def meta(self, name: str) -> dict[str, Any]:
        f = self._frame(name)
        if f is None:
            return {}
        d = asdict(f) if is_dataclass(f) else dict(getattr(f, "__dict__", {}))
        d.pop("jpeg", None)
        d["source"] = "world"
        return d

    def close(self) -> None:
        pass


class NoCameras:
    def names(self) -> list[str]:
        return []

    def rev(self, name: str) -> int:
        return 0

    def jpeg(self, name: str) -> bytes | None:
        return None

    def meta(self, name: str) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class CameraFeed:
    """Decides which frames go to the page: a pane is sent when its rev changed, at most
    MAX_HZ times a second, and never twice for the same (seq, t_wall) (VizCams re-sends the
    last top snapshot every 2 s)."""

    def __init__(self, source: CameraSource, camera: dict[str, Any] | None = None, profile: str | None = None) -> None:
        self.source = source
        self.camera = dict(camera or {})              # the robot map's {name, sim_added} for the head camera
        self.profile = profile
        self._sent: dict[str, tuple[int, float, Any]] = {}

    def reset(self) -> None:
        self._sent = {}

    def due(self, force: bool = False, now: float | None = None) -> list[dict[str, Any]]:
        now = time.monotonic() if now is None else now
        out = []
        for name in self.source.names():
            rev = self.source.rev(name)
            last_rev, last_t, last_key = self._sent.get(name, (-1, -1e9, None))
            if not force and (rev == last_rev or now - last_t < 1.0 / MAX_HZ.get(name, 2.0)):
                continue
            meta = self.source.meta(name)
            key = (meta.get("seq"), meta.get("t_wall"), meta.get("source"))
            if not force and key == last_key and key[0] is not None:
                self._sent[name] = (rev, last_t, last_key)          # a re-send of the frame we already have
                continue
            jpeg = self.source.jpeg(name)
            if not jpeg:
                continue
            self._sent[name] = (rev, now, key)
            caption = head_caption(meta, self.camera, self.profile) if name == "head" else CAPTIONS.get(name, name)
            meta = {k: v for k, v in meta.items() if k in ("t_sim", "t_wall", "w", "h", "seq", "source", "extent",
                                                           "age_s", "snapshot", "hfov", "vfov", "camera", "rev")}
            meta["caption"] = caption
            out.append({"type": "camera", "which": name, "jpeg": jpeg, "meta": meta})
        return out


def thumbnail(jpeg: bytes, width: int = 160) -> bytes:
    """A small JPEG for the scan strip (Pillow); the frame itself when it cannot be decoded."""
    try:
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(jpeg))
        im.thumbnail((width, width * 3 // 4))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=70)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return jpeg


class ScanThumbs:
    """Head-camera thumbnails of each scan, sent once the scan's result arrives (PLAN 9.4: "Scan thumbnails appear
    after each observation"). While an `observe(scan)` runs (the arrival scan, or a scan inside wait_and_observe),
    every new head frame is kept; the result row closes the scan and names the spot and what it saw. The frames are
    the camera's pictures; nothing here reads ground truth."""

    MAX_FRAMES = 24          # kept while a scan runs
    MAX_THUMBS = 6           # sent per scan, evenly spread over the scan
    KEEP = 4                 # recent scans a new page gets

    def __init__(self) -> None:
        self.frames: list[tuple[int, float, bytes]] = []
        self.scanning = False
        self.recent: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.frames, self.scanning, self.recent = [], False, []

    @staticmethod
    def _is_scan(e: dict[str, Any]) -> bool:
        return e.get("tool") == "observe" and (e.get("action") == "scan" or (e.get("args") or {}).get("mode") == "scan")

    def update(self, frame: dict[str, Any], source: CameraSource, caption: str = "") -> list[dict[str, Any]]:
        """One page frame (the Session's `frame` message) and the camera source -> scan messages to send."""
        rt = frame.get("runtime") or {}
        active = list(rt.get("active") or []) + list((frame.get("truth") or {}).get("executions") or [])
        now = float(frame.get("t") or 0.0)
        running = any(self._is_scan(e) for e in active)
        done = [r for r in frame.get("trace") or [] if r.get("type") == "result" and self._is_scan(r)]
        if running or done or self.scanning:
            self._grab(source, now)
        self.scanning = running
        out = []
        for r in done:
            out.append(self._close(r, caption))
        return [m for m in out if m is not None]

    def _grab(self, source: CameraSource, now: float) -> None:
        try:
            if "head" not in source.names():
                return
            rev = source.rev("head")
            if self.frames and self.frames[-1][0] == rev:
                return
            jpeg = source.jpeg("head")
        except Exception:  # noqa: BLE001
            return
        if jpeg:
            self.frames = (self.frames + [(rev, now, jpeg)])[-self.MAX_FRAMES:]

    def _close(self, r: dict[str, Any], caption: str) -> dict[str, Any] | None:
        frames, self.frames = self.frames, []
        if not frames:
            return None
        n = len(frames)
        pick = sorted({round(i * (n - 1) / max(1, self.MAX_THUMBS - 1)) for i in range(min(n, self.MAX_THUMBS))})
        d = r.get("data") or {}
        import base64
        msg = {"type": "scan", "execution_id": r.get("execution_id"),
               "observation_id": r.get("observation_id") or d.get("observation_id"),
               "at": d.get("at"), "sees": list(d.get("saw") or []), "t": r.get("t"), "status": r.get("status"),
               "executor": d.get("scan_executor"), "note": d.get("scan_note"), "caption": caption,
               "thumbs": [{"rev": frames[i][0], "t": frames[i][1],
                           "jpeg": base64.b64encode(thumbnail(frames[i][2])).decode()} for i in pick]}
        self.recent = (self.recent + [msg])[-self.KEEP:]
        return msg
