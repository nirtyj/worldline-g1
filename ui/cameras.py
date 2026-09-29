"""Where the page's camera panes come from.

On Isaac profiles the frames come from viz's FrameTap (docs/viz.md 9.2): the head
camera on SUB 5565 (gear_sonic msgpack, R/B fixed by the tap), and the chase / top /
overview cameras on SUB 5602 (VizCams, or P1's own --tp-camera as "chase"). The top
view's meta carries `extent` [xmin, ymin, xmax, ymax] (Isaac xy = Worldline x, z), which
the page uses to lay the image under the ground-truth map.

On `lite` the world model renders its own frames (`world.latest_frame(camera)`), so
`WorldCameras` wraps that. Both sources have the same four calls: names, rev, jpeg, meta.

Frames are pictures, not ground truth; the robot pose the page draws comes from
`world.truth()`, never from the tap's gt.pose copy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

CAPTIONS = {
    "head": "head camera · what System 1 sees",
    "ego": "ego camera · what GR00T sees",
    "chase": "chase camera · sim viz, not a robot sensor",
    "top": "top view · sim render (ground truth)",
    "overview": "overview · sim render (ground truth)",
}
# The panes the page offers, in tab order. Anything else a source publishes is offered after them.
PANES = ("head", "chase", "top", "ego")
# Most frames per second the page gets for each pane (the THOR page sent head <= 5 Hz, top <= 1 Hz).
MAX_HZ = {"head": 5.0, "chase": 5.0, "ego": 2.0, "top": 1.0, "overview": 1.0}


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
            tap = FrameTap(port_offset=port_offset)
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
        m = dict(self.tap.meta(name) or {})
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

    def __init__(self, source: CameraSource) -> None:
        self.source = source
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
            meta = {k: v for k, v in meta.items() if k in ("t_sim", "t_wall", "w", "h", "seq", "source", "extent",
                                                           "age_s", "snapshot", "hfov", "vfov", "camera", "rev")}
            meta["caption"] = CAPTIONS.get(name, name)
            out.append({"type": "camera", "which": name, "jpeg": jpeg, "meta": meta})
        return out
