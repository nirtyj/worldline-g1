"""g1_debug subscriber (deploy PUB 5557).

Wire: one frame [b"g1_debug"][msgpack map] (output_interface/zmq_output_handler.hpp:1-60, 13-17).
Fields used: index, base_quat, init_base_quat + delta_heading (conditional, present once the heading state exists,
i.e. after start -> CONTROL; :43-45), last_action (29, MuJoCo order, :37), token_state (:40), base_trans_target (:48).

IMPORTANT: g1_debug is only published from the CONTROL state (g1_deploy_onnx_ref.cpp:4002-4010, inside
`case ProgramState::CONTROL`). Before start the deploy only re-publishes [b"robot_config"][msgpack] every ~2 s
(zmq_output_handler.hpp:211-230; called from INIT and WAIT_FOR_CONTROL at g1_deploy_onnx_ref.cpp:3845, 3857 and once
at construction :2592). So "deploy alive" = robot_config or g1_debug seen recently; "in control" = fresh g1_debug
carrying init_base_quat. The INIT ramp to the default pose takes 3 s after lowstate appears (duration_, :181) and ends
with "Init Done" on stdout (:2787-2790); send start only after that.

This is also E4 evidence: the control-loop rate (~50 Hz) and whether the leg targets (last_action[0:12]) change.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
import zmq

TOPIC = b"g1_debug"
CONFIG_TOPIC = b"robot_config"


class DeployMonitor(threading.Thread):
    def __init__(self, endpoint: str, ctx: zmq.Context | None = None, on_debug=None):
        super().__init__(name="g1-debug-sub", daemon=True)
        self.endpoint = endpoint
        self._ctx = ctx or zmq.Context.instance()
        self._lock = threading.Lock()
        self._running = True
        self.latest: dict | None = None
        self.latest_mono = 0.0
        self._ts: collections.deque[float] = collections.deque(maxlen=500)
        self._legs: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=500)
        self._trans_target: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=500)
        self.count = 0
        self.parse_errors = 0
        self.on_debug = on_debug
        self.first_heading_mono: float | None = None
        self.config: dict | None = None
        self.config_mono = 0.0
        self.first_seen_mono: float | None = None
        self.config_count = 0

    def run(self) -> None:
        import msgpack

        s = self._ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 50)
        s.setsockopt(zmq.SUBSCRIBE, TOPIC)
        s.setsockopt(zmq.SUBSCRIBE, CONFIG_TOPIC)
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
                f0 = frames[0]
                is_cfg = f0.startswith(CONFIG_TOPIC)
                topic = CONFIG_TOPIC if is_cfg else TOPIC
                if not f0.startswith(topic):
                    continue
                raw = frames[-1] if len(frames) > 1 else f0[len(topic):]
                try:
                    d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                except Exception:
                    self.parse_errors += 1
                    continue
                now = time.monotonic()
                if self.first_seen_mono is None:
                    self.first_seen_mono = now
                if is_cfg:
                    with self._lock:
                        self.config = d
                        self.config_mono = now
                        self.config_count += 1
                    continue
                with self._lock:
                    self.latest = d
                    self.latest_mono = now
                    self.count += 1
                    self._ts.append(now)
                    la = d.get("last_action")
                    if la is not None and len(la) >= 12:
                        self._legs.append((now, np.asarray(la[:12], dtype=float)))
                    tt = d.get("base_trans_target")
                    if tt is not None:
                        self._trans_target.append((now, np.asarray(tt, dtype=float)))
                    if d.get("init_base_quat") is not None and self.first_heading_mono is None:
                        self.first_heading_mono = now
                if self.on_debug:
                    try:
                        self.on_debug(d)
                    except Exception:
                        pass
        s.close(0)

    def stop(self) -> None:
        self._running = False

    def age_s(self) -> float:
        """Age of the last g1_debug (CONTROL only)."""
        return float("inf") if self.latest is None else time.monotonic() - self.latest_mono

    def alive_age_s(self) -> float:
        """Age of the last message of either topic (robot_config comes every ~2 s before CONTROL)."""
        a = self.age_s()
        c = float("inf") if self.config is None else time.monotonic() - self.config_mono
        return min(a, c)

    def alive(self, max_age_s: float = 5.0) -> bool:
        return self.alive_age_s() < max_age_s

    def alive_for_s(self) -> float:
        return 0.0 if self.first_seen_mono is None else time.monotonic() - self.first_seen_mono

    def rate_hz(self, window_s: float = 2.0) -> float:
        now = time.monotonic()
        with self._lock:
            ts = [t for t in self._ts if now - t <= window_s]
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])

    def heading(self) -> tuple[list | None, float]:
        d = self.latest or {}
        return d.get("init_base_quat"), float(d.get("delta_heading") or 0.0)

    def in_control(self, max_age_s: float = 1.0) -> bool:
        """init_base_quat appears only once the heading state exists (after start, in CONTROL)."""
        return self.age_s() < max_age_s and (self.latest or {}).get("init_base_quat") is not None

    def leg_action_stats(self, window_s: float = 2.0) -> dict:
        now = time.monotonic()
        with self._lock:
            legs = [a for t, a in self._legs if now - t <= window_s]
        if len(legs) < 2:
            return {"n": len(legs), "std_mean": 0.0, "changes": 0}
        arr = np.stack(legs)
        changes = int(np.sum(np.any(np.abs(np.diff(arr, axis=0)) > 1e-6, axis=1)))
        return {"n": len(legs), "std_mean": float(arr.std(axis=0).mean()), "std_max": float(arr.std(axis=0).max()),
                "changes": changes, "change_frac": changes / (len(legs) - 1)}

    def snapshot(self) -> dict:
        d = self.latest or {}
        return {"endpoint": self.endpoint, "age_s": round(self.age_s(), 3) if self.latest else None,
                "alive": self.alive(), "alive_for_s": round(self.alive_for_s(), 1), "config_count": self.config_count,
                "rate_hz": round(self.rate_hz(), 2), "count": self.count, "in_control": self.in_control(),
                "index": d.get("index"), "has_init_base_quat": d.get("init_base_quat") is not None,
                "delta_heading": d.get("delta_heading"), "legs": self.leg_action_stats()}
