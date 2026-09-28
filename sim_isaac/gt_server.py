"""ZMQ transport for wl-isaac: REP control/ground-truth server and the gt.* PUB stream (docs/contracts/m1.md 1.5-1.6).

Requests are JSON (first byte '{') or msgpack maps; the reply uses the same encoding. Handlers run on the main
(physics) thread between physics steps, so they can touch the simulation safely.
"""
from __future__ import annotations

import json
import time
import traceback
from typing import Any, Callable

import numpy as np


def _plain(o: Any):
    """Make numpy scalars/arrays JSON/msgpack friendly."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class GtServer:
    def __init__(self, ctx, rep_port: int, pub_port: int, bind_host: str = "127.0.0.1", log=print):
        import zmq

        self.zmq = zmq
        self.log = log
        self.rep = ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.rep.bind(f"tcp://{bind_host}:{rep_port}")
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.setsockopt(zmq.SNDHWM, 200)
        self.pub.bind(f"tcp://{bind_host}:{pub_port}")
        self.handlers: dict[str, Callable[[dict], dict]] = {}
        self.requests = 0
        self.slow_ms = 0.0

    def register(self, op: str, fn: Callable[[dict], dict]) -> None:
        self.handlers[op] = fn

    def poll(self, max_requests: int = 2) -> int:
        """Serve up to max_requests pending requests without blocking."""
        import msgpack

        n = 0
        while n < max_requests and self.rep.poll(0):
            raw = self.rep.recv()
            t0 = time.perf_counter()
            is_json = raw[:1] == b"{"
            try:
                req = json.loads(raw) if is_json else msgpack.unpackb(raw, raw=False)
                if not isinstance(req, dict) or "op" not in req:
                    raise ValueError("request must be a map with 'op'")
                fn = self.handlers.get(req["op"])
                if fn is None:
                    rep = {"ok": False, "error": f"unknown op {req['op']!r}", "ops": sorted(self.handlers)}
                else:
                    rep = fn(req) or {}
                    rep.setdefault("ok", True)
            except Exception as e:  # noqa: BLE001
                self.log(f"[gt_server] error: {e}\n{traceback.format_exc()}")
                rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            rep = _plain(rep)
            self.rep.send(json.dumps(rep).encode() if is_json else msgpack.packb(rep, use_bin_type=True))
            self.requests += 1
            self.slow_ms = max(self.slow_ms, (time.perf_counter() - t0) * 1e3)
            n += 1
        return n

    def publish(self, topic: str, payload: dict) -> None:
        import msgpack

        try:
            self.pub.send_multipart([topic.encode(), msgpack.packb(_plain(payload), use_bin_type=True)],
                                    flags=self.zmq.NOBLOCK)
        except self.zmq.Again:
            pass

    def close(self):
        self.rep.close(0)
        self.pub.close(0)
