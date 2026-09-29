"""The test-only P1 ops (sim_isaac/test_ops.py) on tools/fake_p1.FakeP1, for offline tests and rehearsals.

Same op names, arguments and replies as the real P1 with --test-ops (the parsing is sim_isaac.test_ops's, shared);
the physics is the fake's:
  push_robot    an impulse >= FALL_IMPULSE_NS with the band off collapses the kinematic robot (fallen, like the
                deploy's damping); a smaller one only logs (the fake has no balance to disturb)
  rtf_throttle  pins FakeP1.rtf_override to the target (gt.pose and sim.health report it) until duration_s ends
  spawn_box     draws the box footprint into the fake's physics grid (it blocks the robot but is NOT in the published
                map, exactly like the real box); clear_box / ttl_s restores the grid

    python -m tools.hooks.fake_ops --port-offset 200      # a fake P1 that answers the test ops
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from sim_isaac import test_ops as T
from sim_isaac.wire import OpError
from tools.fake_p1 import FakeP1

FALL_IMPULSE_NS = 60.0


class HookableFakeP1(FakeP1):
    def __init__(self, *args, box_size=(0.3, 1.6, 1.2), **kw):
        super().__init__(*args, **kw)
        self.box_size = T.parse_size(box_size)
        self.push = self.throttle = self.box = None
        self.counts = {op: 0 for op in T.OPS}
        self._rtf_before = self.rtf_override
        self._occ_before: np.ndarray | None = None
        self._timers: list[threading.Timer] = []

    def _later(self, s: float, fn) -> None:
        t = threading.Timer(s, fn)
        t.daemon = True
        t.start()
        self._timers.append(t)

    def stop(self) -> None:
        for t in self._timers:
            t.cancel()
        super().stop()

    def _op(self, op: str, a: dict) -> dict:
        if op in T.OPS:
            with self._lock:
                try:
                    return {"ok": True, **getattr(self, f"_t_{op}")(a)}
                except OpError as e:
                    return e.reply()
        rep = super()._op(op, a)
        if op == "ping" and rep.get("ok"):
            rep["ops"] = sorted(set(rep.get("ops") or []) | set(T.OPS))
        return rep

    # -- ops ------------------------------------------------------------------------------------
    def _t_push_robot(self, a: dict) -> dict:
        p = T.parse_push(a, self.yaw)
        p.update(t_sim_start=round(self.t_sim, 4), until_t_sim=self.t_sim + p["duration_s"], robot_yaw=self.yaw)
        self.counts["push_robot"] += 1
        fell = p["impulse_ns"] >= FALL_IMPULSE_NS and not self.band_on and not self.collapsed
        if fell:
            self.collapsed = True
        self._event("test_op", op="push_robot", state="started", force_w=p["force_w"], impulse_ns=p["impulse_ns"],
                    fake_fell=fell)
        return {"push": p, "label": "test-only", "fake_fell": fell}

    def _t_rtf_throttle(self, a: dict) -> dict:
        th = T.parse_throttle(a)
        self.counts["rtf_throttle"] += 1
        if th is None:
            was = self.throttle
            self._end_throttle("request")
            return {"throttle": None, "was": None if was is None else was["target"], "label": "test-only"}
        if self.throttle is None:
            self._rtf_before = self.rtf_override
        th["extra_s"] = T.throttle_extra_s(1.0 / self.physics_hz, th["target"])
        th["until_wall"] = time.perf_counter() + th["duration_s"]
        self.throttle = th
        self.rtf_override = th["target"]
        self._later(th["duration_s"], lambda: self._end_throttle("duration"))
        self._event("test_op", op="rtf_throttle", state="started", target=th["target"])
        return {"throttle": {k: v for k, v in th.items() if k != "until_wall"}, "label": "test-only"}

    def _end_throttle(self, by: str) -> None:
        with self._lock:
            if self.throttle is None or (by == "duration" and time.perf_counter() < self.throttle["until_wall"]):
                return
            self.throttle = None
            self.rtf_override = self._rtf_before
            self._event("test_op", op="rtf_throttle", state="ended", by=by)

    def _t_spawn_box(self, a: dict) -> dict:
        b = T.parse_box(a, self.box_size)
        self._clear_box("respawn")
        self._occ_before = self.phys_occ.copy()
        H, W_ = self.phys_occ.shape
        yy, xx = np.mgrid[0:H, 0:W_]
        cx = self.origin[0] + (xx + 0.5) * self.res - b["x"]
        cy = self.origin[1] + (yy + 0.5) * self.res - b["y"]
        c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
        inside = (np.abs(c * cx + s * cy) <= self.box_size[0] / 2) & (np.abs(-s * cx + c * cy) <= self.box_size[1] / 2)
        self.phys_occ[inside] = 1
        b.update(size=list(self.box_size), until_wall=time.perf_counter() + b["ttl_s"], cells=int(inside.sum()))
        self.box = b
        self.counts["spawn_box"] += 1
        self._later(b["ttl_s"], lambda: self._expire_box(b))
        self._event("test_op", op="spawn_box", x=b["x"], y=b["y"], yaw=b["yaw"], cells=b["cells"])
        return {"box": {k: v for k, v in b.items() if k != "until_wall"}, "label": "test-only"}

    def _expire_box(self, b: dict) -> None:
        with self._lock:
            if self.box is b:
                self._clear_box("ttl")

    def _clear_box(self, by: str) -> bool:
        if self.box is None:
            return False
        if self._occ_before is not None:
            self.phys_occ = self._occ_before
        self._occ_before, self.box = None, None
        self._event("test_op", op="clear_box", by=by)
        return True

    def _t_clear_box(self, a: dict) -> dict:
        self.counts["clear_box"] += 1
        return {"cleared": self._clear_box("request"), "label": "test-only"}

    def _t_test_ops_status(self, a: dict) -> dict:
        return T.status_reply(self.push, self.throttle, self.box, self.box_size, self.counts, t_sim=self.t_sim,
                              fake=True)


def main(argv=None) -> int:
    import argparse
    import signal
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=200)
    ap.add_argument("--out", default="/tmp/fake_p1_hooks")
    ap.add_argument("--house-dir", default=None)
    a = ap.parse_args(argv)
    p1 = HookableFakeP1(port_offset=a.port_offset, out_dir=a.out, house_dir=a.house_dir, rtf=1.0).start()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(f"[fake_ops] fake P1 with the test ops on offset {a.port_offset}", flush=True)
    while not stop.is_set() and not p1.shutdown_requested:
        stop.wait(0.2)
    p1.stop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
