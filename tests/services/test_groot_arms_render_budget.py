"""groot_arms' render budget (docs/groot_serving.md §9; SONIC's timing gate with the GR00T client active): P1 renders
every enabled camera product at every render call, so a session (a) enables ego_view at camera_warm_hz until its
first frame, then streams it at camera_hz, (b) drops the head camera to session_head_hz for the session and restores
the rate it had, and (c) with frame_sync waits for each fresh frame instead of polling, so every get_action gets a
new frame and nothing counts as stale. Also ZmqSensors.wait_frame, without sockets."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

pytest.importorskip("groot.actions")

from services.executors.groot_arms import GrootArmExecutor, ZmqSensors, _groot_helpers  # noqa: E402
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from tests.services.test_groot_arms import FakeWorld, Sink, fast_cfg, make_job  # noqa: E402


class RateWorld(FakeWorld):
    """FakeWorld whose `camera` op keeps each camera's rate, as P1 does (p1_m2b.md §5.4): the reply carries it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hz = {"head": 30.0, "ego_view": 30.0}
        self.head_log: list[tuple[float, float]] = []            # (t, hz) after every head call

    def enable_camera(self, camera, on, *, consumer="runtime", ttl_s=None, hz=None):
        super().enable_camera(camera, on, consumer=consumer, ttl_s=ttl_s, hz=hz)
        if hz is not None and on:
            self.hz[camera] = float(hz)
        if camera == "head":
            self.head_log.append((time.monotonic(), self.hz["head"]))
        return {"name": camera, "on": on, "hz": self.hz[camera]}


class PacedSensors:
    """The arm body's g1_debug, and ego frames that arrive at `hz` (a camera at the inference rate); each frame's
    first pixel carries its sequence number, so a test can tell which frame an observation was built from."""

    def __init__(self, body: FakeArmBody, hz: float):
        self.body, self.period = body, 1.0 / hz
        self.cond = threading.Condition()
        self.seq, self.t = 0, 0.0
        self._run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self) -> None:
        while self._run:
            time.sleep(self.period)
            with self.cond:
                self.seq += 1
                self.t = time.monotonic()
                self.cond.notify_all()

    def ego_frame(self):
        with self.cond:
            if self.seq == 0:
                return None, 0.0
            f = np.full((480, 640, 3), 90, dtype=np.uint8)
            f[0, 0, 0] = self.seq % 256
            return f, self.t

    def wait_frame(self, after: float, timeout_s: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.seq > 0 and self.t > after, timeout=timeout_s)

    def debug_state(self):
        return self.body.debug_state()

    def close(self) -> None:
        self._run = False


def _rig(world: FakeWorld, sensors=None, **cfg):
    srv = FakePolicyServer(close_after=None).start()
    body = FakeArmBody()
    world.couple(body)
    seen: list[int] = []
    h = _groot_helpers()
    build = h["build_obs"]

    def build_obs(frame, *a, **kw):
        seen.append(int(np.asarray(frame)[0, 0, 0]))
        return build(frame, *a, **kw)
    h["build_obs"] = build_obs
    h["build_obs_warmup"] = build
    exe = GrootArmExecutor(world, arm=body, sensors=sensors or body, cfg=fast_cfg(srv, **cfg), events=Sink(),
                           helpers=h)
    return srv, body, exe, seen


def _healthy(exe: GrootArmExecutor, timeout: float = 5.0) -> None:
    t_end = time.monotonic() + timeout
    while not exe.health().ok and time.monotonic() < t_end:
        time.sleep(0.05)
    assert exe.health().ok


def test_the_session_warms_the_ego_camera_then_streams_at_the_inference_rate_with_the_head_lowered():
    world = RateWorld("never", enable_camera=True)
    srv, body, exe, _ = _rig(world, camera_hz=2.5, camera_warm_hz=10.0, session_head_hz=2.5, max_duration_s=1.2)
    try:
        _healthy(exe)
        job, handle = make_job()
        out = asyncio.run(exe.run(job, handle))
    finally:
        exe.close(); body.close(); srv.stop()
    ego_on = [c for c in world.cam_calls if c[0] == "ego_view" and c[1]]
    assert ego_on[0][5] == 10.0 and ego_on[1][5] == 2.5                    # warm-up rate, then the stream rate
    assert {c[5] for c in ego_on[1:]} == {2.5}                              # the lease renewals keep 2.5
    head = [c for c in world.cam_calls if c[0] == "head"]
    assert all(c[1] and c[2] == "default" for c in head)                   # only the rate changes: its own consumer
    assert [c[5] for c in head] == [None, 2.5, 30.0]                        # read the rate, lower it, restore it
    first_chunk = body.messages("chunk", "man-1")[0]["t"]
    t_lowered = world.head_log[1][0]
    ego_off = next(c for c in world.cam_calls if c[0] == "ego_view" and not c[1])
    assert ego_on[1][4] < t_lowered < first_chunk                           # lowered before GR00T streams
    assert world.head_log[-1][1] == 30.0 and world.head_log[-1][0] >= ego_off[4]   # restored after ego_view went off
    r = out.data["render"]
    assert r == {"ego_hz": 2.5, "warm_hz": 10.0, "session_head_hz": 2.5, "head_restored_hz": 30.0,
                 "frame_sync": False, "frame_waits": 0, "frame_wait_s": 0.0}


def test_without_a_session_head_rate_the_head_camera_is_untouched():
    world = RateWorld("never", enable_camera=True)
    srv, body, exe, _ = _rig(world, max_duration_s=0.6)
    try:
        _healthy(exe)
        job, handle = make_job()
        out = asyncio.run(exe.run(job, handle))
    finally:
        exe.close(); body.close(); srv.stop()
    assert not [c for c in world.cam_calls if c[0] == "head"]
    assert [c[5] for c in world.cam_calls if c[0] == "ego_view" and c[1]][0] == 30.0   # the old single rate
    assert out.data["render"]["session_head_hz"] is None


def test_frame_sync_gives_every_inference_a_fresh_frame_and_counts_no_stale_observations():
    world = FakeWorld("never")
    body = FakeArmBody()
    sensors = PacedSensors(body, hz=10.0)                   # a 10 Hz camera, inference every 0.1 s at most
    srv = FakePolicyServer(close_after=None).start()
    world.couple(body)
    seen: list[int] = []
    h = _groot_helpers()
    build = h["build_obs"]

    def build_obs(frame, *a, **kw):
        seen.append(int(np.asarray(frame)[0, 0, 0]))
        return build(frame, *a, **kw)
    h["build_obs"], h["build_obs_warmup"] = build_obs, build
    exe = GrootArmExecutor(world, arm=body, sensors=sensors, cfg=fast_cfg(srv, frame_sync=True, camera_hz=10.0,
                                                                           replan_s=0.05, max_duration_s=1.5),
                           events=Sink(), helpers=h)
    try:
        _healthy(exe)
        job, handle = make_job()
        out = asyncio.run(exe.run(job, handle))
    finally:
        exe.close(); sensors.close(); body.close(); srv.stop()
    d = out.data
    assert d["inferences"] >= 5
    assert len(seen) == len(set(seen)), f"a frame was sent twice: {seen}"   # on demand: one inference per frame
    assert d["obs_stale"] == 0
    assert d["render"]["frame_sync"] is True and d["render"]["frame_waits"] >= 1


def test_zmq_sensors_wait_frame_wakes_on_a_newer_frame_and_times_out_without_one():
    s = ZmqSensors("tcp://127.0.0.1:1", "tcp://127.0.0.1:2")     # nothing listens: only the hand-off is tested
    try:
        t0 = time.monotonic()
        assert s.wait_frame(t0, 0.05) is False

        def later():
            time.sleep(0.1)
            with s._lock:
                s._seq += 1
                s._frame = (b"jpeg", time.monotonic())
                s._new_frame.notify_all()
        threading.Thread(target=later, daemon=True).start()
        assert s.wait_frame(t0, 2.0) is True
        assert 0.08 <= time.monotonic() - t0 < 1.0
        assert s.wait_frame(time.monotonic(), 0.05) is False                 # that frame is not newer than now
    finally:
        s.close()
