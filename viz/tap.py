"""FrameTap: the latest sim frames + gt.pose for any Python UI, in a background thread (no asyncio needed).

For the Worldline ui/server.py (websockets-based) or any other consumer:

    from viz.tap import FrameTap
    tap = FrameTap(port_offset=0)          # SUB head 5565, frames 5602, gt.pose 5601
    tap = FrameTap(gt_pose=False)          # frames only: no SUB on 5601 (the runtime side, where only world/
                                           # may read ground truth: ui/cameras.py uses this)
    ...
    jpeg = tap.jpeg("head")                # standard-colour JPEG bytes (R/B fixed) or None
    jpeg = tap.jpeg("chase")               # VizCams chase / top / overview (P1 frame.tp -> "chase", R/B fixed)
    meta = tap.meta("top")                 # {"t_sim", "extent": [xmin,ymin,xmax,ymax], "robot": {...}, ...}
    rev = tap.rev("chase")                 # bumps on every new frame (send only on change)
    pose = tap.pose()                      # latest gt.pose dict (contract fields)
    tap.close()

The head frame (and P1's frame.tp) is re-encoded (R/B swap, ~5 ms) lazily on the first jpeg() call after a new
frame.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import zmq

from viz.common import decode_head, ep, frame_from_msg, ports, split_msg, swap_rb_jpeg


class FrameTap:
    def __init__(self, port_offset: int | None = None, head_swap_rb: bool = True, gt_pose: bool = True,
                 **port_overrides: int):
        self.p = ports(port_offset, **port_overrides)
        self.head_swap_rb = head_swap_rb
        self.gt_pose = gt_pose                                           # False: never subscribe to gt.pose
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[bytes, dict, int, float]] = {}   # name -> (jpeg, meta, rev, t_rx)
        self._swap: dict[str, bool] = {"head": head_swap_rb}             # name -> JPEG is cv2-encoded RGB
        self._fixed: dict[str, tuple[int, bytes]] = {}                   # name -> (rev, R/B-fixed JPEG)
        self._pose: dict | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="viz.tap", daemon=True)
        self._t.start()

    def jpeg(self, name: str) -> bytes | None:
        with self._lock:
            f = self._frames.get(name)
        if f is None:
            return None
        if self._swap.get(name):
            fx = self._fixed.get(name)
            if fx is None or fx[0] != f[2]:
                fx = self._fixed[name] = (f[2], swap_rb_jpeg(f[0], 85))
            return fx[1]
        return f[0]

    def meta(self, name: str) -> dict:
        with self._lock:
            f = self._frames.get(name)
        return dict(f[1]) if f else {}

    def rev(self, name: str) -> int:
        with self._lock:
            f = self._frames.get(name)
        return f[2] if f else 0

    def age_s(self, name: str) -> float | None:
        with self._lock:
            f = self._frames.get(name)
        return time.time() - f[3] if f else None

    def streams(self) -> list[str]:
        with self._lock:
            return sorted(self._frames)

    def pose(self) -> dict | None:
        with self._lock:
            return dict(self._pose) if self._pose else None

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)

    def _put(self, name: str, jpeg: bytes, meta: dict) -> None:
        with self._lock:
            rev = self._frames[name][2] + 1 if name in self._frames else 1
            self._frames[name] = (jpeg, meta, rev, time.time())

    def _run(self) -> None:
        ctx = zmq.Context.instance()
        socks = {}
        subs = [("head", self.p["head"], [b""], True), ("frames", self.p["frames"], [b"frame."], False)]
        if self.gt_pose:
            subs.append(("gt", self.p["gt_pub"], [b"gt.pose"], False))
        for name, port, topics, conflate in subs:
            s = ctx.socket(zmq.SUB)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVHWM, 20)
            if conflate:
                s.setsockopt(zmq.CONFLATE, 1)
            for t in topics:
                s.setsockopt(zmq.SUBSCRIBE, t)
            s.connect(ep(port))
            socks[s] = name
        poller = zmq.Poller()
        for s in socks:
            poller.register(s, zmq.POLLIN)
        while not self._stop.is_set():
            for s, _ in poller.poll(200):
                while True:
                    try:
                        frames = s.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    kind = socks[s]
                    try:
                        if kind == "head":
                            jpeg, meta = decode_head(frames[-1])
                            if jpeg:
                                self._put("head", jpeg, meta)
                        elif kind == "frames":
                            got = frame_from_msg(*split_msg(frames))
                            if got is not None:
                                name, jpeg, msg, swap = got
                                meta: dict[str, Any] = {k: msg.get(k) for k in ("t_sim", "t_wall", "w", "h", "robot",
                                                                                  "seq", "snapshot", "source")}
                                meta["extent"] = (msg.get("cam_pose") or {}).get("extent")
                                meta["cam_pose"] = msg.get("cam_pose")
                                self._swap[name] = swap
                                self._put(name, jpeg, meta)
                        else:
                            topic, msg = split_msg(frames)
                            if topic == "gt.pose" and isinstance(msg, dict):
                                with self._lock:
                                    self._pose = msg
                    except Exception:  # noqa: BLE001  (a bad message must not kill the tap)
                        pass
        for s in socks:
            s.close(0)
