"""Clients for P1 (wl-isaac): REQ -> REP 5600 and SUB gt.pose on 5601 (docs/contracts/m1.md)."""

from __future__ import annotations

import collections
import math
import threading
import time

import numpy as np
import zmq

from .wire import Pose, dumps_json, loads_any, parse_pose, split_topic


class P1Error(RuntimeError):
    pass


class P1Rpc:
    """Lazy-pirate REQ client. Requests are JSON {"op": <op>, **args, "args": {...}} so a server may read the
    arguments either at the top level or under "args". Replies may be JSON or msgpack."""

    def __init__(self, endpoint: str, timeout_s: float = 5.0, ctx: zmq.Context | None = None):
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_s * 1000)
        self._ctx = ctx or zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0

    def _connect(self) -> zmq.Socket:
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.connect(self.endpoint)
        return s

    def call(self, op: str, timeout_s: float | None = None, **args) -> dict:
        req = {"op": op, **args, "args": args}
        with self._lock:
            if self._sock is None:
                self._sock = self._connect()
            s = self._sock
            if timeout_s is not None:
                s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
            self.calls += 1
            try:
                s.send(dumps_json(req))
                raw = s.recv()
            except zmq.ZMQError as e:
                self.failures += 1
                s.close(0)
                self._sock = None  # a REQ socket is wedged after a timeout: recreate
                raise P1Error(f"P1 {op} failed: {e}") from e
            finally:
                if self._sock is not None and timeout_s is not None:
                    self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        rep = loads_any(raw)
        if not isinstance(rep, dict):
            rep = {"ok": True, "result": rep}
        return rep

    def try_call(self, op: str, **args) -> dict | None:
        try:
            return self.call(op, **args)
        except P1Error:
            return None

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                self._sock.close(0)
                self._sock = None


class PoseSub(threading.Thread):
    """Keeps the latest gt.pose and a short history; computes receive rate and a finite-difference velocity
    (used when P1 does not fill base_lin_vel_w)."""

    TOPIC = b"gt.pose"

    def __init__(self, endpoint: str, history_s: float = 10.0, ctx: zmq.Context | None = None, on_pose=None):
        super().__init__(name="gt-pose-sub", daemon=True)
        self.endpoint = endpoint
        self._ctx = ctx or zmq.Context.instance()
        self._lock = threading.Lock()
        self._latest: Pose | None = None
        self._hist: collections.deque[Pose] = collections.deque(maxlen=int(history_s * 250))
        self._running = True
        self.count = 0
        self.parse_errors = 0
        self.on_pose = on_pose

    def run(self) -> None:
        s = self._ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 100)
        s.setsockopt(zmq.SUBSCRIBE, self.TOPIC)
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
                payload = split_topic(frames, self.TOPIC)
                if payload is None:
                    continue
                try:
                    p = parse_pose(payload, time.monotonic())
                except Exception:
                    self.parse_errors += 1
                    continue
                with self._lock:
                    self._latest = p
                    self._hist.append(p)
                    self.count += 1
                if self.on_pose:
                    try:
                        self.on_pose(p)
                    except Exception:
                        pass
        s.close(0)

    def stop(self) -> None:
        self._running = False

    def latest(self) -> Pose | None:
        with self._lock:
            return self._latest

    def age_s(self) -> float:
        p = self.latest()
        return float("inf") if p is None else time.monotonic() - p.recv_mono

    def history(self, window_s: float | None = None) -> list[Pose]:
        with self._lock:
            h = list(self._hist)
        if window_s is None:
            return h
        now = time.monotonic()
        return [p for p in h if now - p.recv_mono <= window_s]

    def rate_hz(self, window_s: float = 2.0) -> float:
        h = self.history(window_s)
        if len(h) < 2:
            return 0.0
        return (len(h) - 1) / max(1e-6, h[-1].recv_mono - h[0].recv_mono)

    def velocity(self, window_s: float = 0.3) -> tuple[float, float, float]:
        """(vx, vy, wz) world frame: P1's fields if present, else finite differences over window_s."""
        p = self.latest()
        if p is None:
            return 0.0, 0.0, 0.0
        if "base_lin_vel_w" in p.raw:
            return p.vx, p.vy, p.wz
        h = self.history(window_s)
        if len(h) < 2:
            return 0.0, 0.0, 0.0
        a, b = h[0], h[-1]
        dt = max(1e-3, (b.t_sim - a.t_sim) if b.t_sim > a.t_sim else (b.recv_mono - a.recv_mono))
        dyaw = math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))
        return (b.x - a.x) / dt, (b.y - a.y) / dt, dyaw / dt

    def speed(self) -> float:
        vx, vy, _ = self.velocity()
        return math.hypot(vx, vy)


def wait_for_pose(sub: PoseSub, timeout_s: float) -> Pose | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        p = sub.latest()
        if p is not None:
            return p
        time.sleep(0.02)
    return None
