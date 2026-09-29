"""P1's M2b topics on PUB 5601 besides gt.pose (docs/contracts/p1_m2b.md §1, §3.2, §10):

    gt.objects   10 Hz   {seq, t_sim, t_wall, objects: [OBJ...]}   dynamic and held objects only (P1.2)
    gt.event     sporadic {event, t_sim, t_wall, ...}                 robot_fell, object_fell, attach, detach, camera,
                                                                     reset_scene, object_moved, band, ... (P1.9)
    sim.health   1 Hz    {rtf_1s, rtf_3s, rtf_5s, rtf_10s, level, ...}                                      (P1.9)

One SUB thread, three prefix subscriptions (ZMQ matches prefixes, so b"gt.event" also receives an M2b "gt.events"
if a P1 ever sends it). Handlers run on this thread; world/isaac_client.py locks what they touch. gt.pose stays on
wl-body's PoseSub (body.p1_client), shared with the body so both parse it identically.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import zmq

from body.wire import loads_any

TOPICS = (b"gt.objects", b"gt.event", b"sim.health")


class P1Stream(threading.Thread):
    def __init__(self, endpoint: str, handlers: dict[str, Callable[[dict], Any]], ctx: zmq.Context | None = None):
        super().__init__(name="p1-m2b-sub", daemon=True)
        self.endpoint = endpoint
        self.handlers = dict(handlers)
        self._ctx = ctx or zmq.Context.instance()
        self._running = True
        self.counts: dict[str, int] = {}
        self.last_mono: dict[str, float] = {}
        self.errors = 0

    def run(self) -> None:
        s = self._ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 200)
        for t in TOPICS:
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(s, zmq.POLLIN)
        while self._running:
            if not poller.poll(100):
                continue
            while True:
                try:
                    frames = s.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(frames) < 2:
                    continue
                topic = frames[0].decode("utf-8", "replace")
                try:
                    msg = loads_any(frames[-1])
                except Exception:  # noqa: BLE001
                    self.errors += 1
                    continue
                if not isinstance(msg, dict):
                    continue
                key = "gt.event" if topic.startswith("gt.event") else topic
                self.counts[key] = self.counts.get(key, 0) + 1
                self.last_mono[key] = time.monotonic()
                fn = self.handlers.get(key)
                if fn is None:
                    continue
                try:
                    fn(msg)
                except Exception:  # noqa: BLE001  (a bad message must not kill the stream)
                    self.errors += 1
        s.close(0)

    def age_s(self, topic: str) -> float:
        t = self.last_mono.get(topic)
        return float("inf") if t is None else time.monotonic() - t

    def stop(self) -> None:
        self._running = False


__all__ = ["TOPICS", "P1Stream"]
