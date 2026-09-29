"""groot_arms' ZMQ adapters on real sockets (what runs live in wave 2): BodyArmPort over body.client.BodyClient
against a ROUTER/PUB wrapper of the chunk-mode reference body (docs/contracts/m1.md §3.4 envelope,
docs/contracts/arm_chunk.md), and ZmqSensors against P1's ego_view PUB format (p1_m2b.md §5.2) and SONIC's
g1_debug (sonic_deploy.md §4)."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time

import numpy as np
import pytest
import zmq

pytest.importorskip("groot.actions")

from body.client import BodyClient  # noqa: E402
from services.executors.groot_arms import BodyArmPort, GrootArmExecutor, ZmqSensors, _groot_helpers  # noqa: E402
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from tests.services.test_groot_arms import FakeWorld, fast_cfg, make_job  # noqa: E402

OFF = 530                                   # ports 6140 (ROUTER) / 6141 (PUB); no other test uses this offset


class WireBody:
    """wl-body's ROUTER 5610 / PUB 5611 in front of FakeArmBody: `arm` requests go to the fake, its events and a 20 Hz
    body.state (with arm.modes) go out on the PUB."""

    def __init__(self, body: FakeArmBody, modes=("target", "chunk"), off: int = OFF):
        self.body, self.modes, self.mute = body, list(modes), False
        self.ctx = zmq.Context.instance()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://127.0.0.1:{5610 + off}")
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://127.0.0.1:{5611 + off}")
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        body.subscribe(lambda ev: self._send(b"body.event", ev))
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._serve, self._state)]
        for t in self.threads:
            t.start()

    def _send(self, topic: bytes, msg: dict) -> None:
        with self.lock:
            self.pub.send_multipart([topic, json.dumps(msg).encode()])

    def _state(self) -> None:
        while not self.stop_evt.is_set():
            self._send(b"body.state", {"in_control": True, "fault": None, "deploy": {"alive": True},
                                       "arm": {"state": "stream" if self.body.cur else "off", "modes": self.modes}})
            time.sleep(0.05)

    def _serve(self) -> None:
        while not self.stop_evt.is_set():
            if not self.router.poll(50):
                continue
            ident, raw = self.router.recv_multipart()
            req = json.loads(raw)
            if self.mute:                     # a hung body: never answers
                continue
            rep = self.body.arm(req.get("args") or {}, req.get("id")) if req.get("op") == "arm" else \
                {"ok": False, "state": "rejected", "error": f"unknown op {req.get('op')!r}"}
            self.router.send_multipart([ident, json.dumps({"id": req.get("id"), **rep}).encode()])

    def close(self) -> None:
        self.stop_evt.set()
        for t in self.threads:
            t.join(timeout=1.0)
        self.router.close(0)
        self.pub.close(0)


@pytest.fixture
def wire():
    body = FakeArmBody()
    w = WireBody(body)
    client = BodyClient(port_offset=OFF).connect(wait_s=5.0)
    port = BodyArmPort(client, timeout_s=0.5)
    yield body, w, client, port
    port.close()
    client.close()
    w.close()
    body.close()


def test_a_session_over_the_real_body_wire(wire):
    body, w, client, port = wire
    world = FakeWorld("lift")
    world.couple(body)
    srv = FakePolicyServer(close_after=2).start()
    exe = GrootArmExecutor(world, arm=port, sensors=body, cfg=fast_cfg(srv), helpers=_groot_helpers())
    try:
        assert port.supports_chunk() is True
        t_end = time.monotonic() + 5.0
        while not exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        job, h = make_job("wire-1")
        out = asyncio.run(exe.run(job, h))
        assert out.status == "succeeded", out.detail
        chunks = [m for m in body.messages("chunk", "wire-1") if m["reply"]["ok"]]
        assert chunks and chunks[0]["chunk"]["upper_body"] == (40, 17) and chunks[0]["chunk"]["order"] == "wire"
        assert body.messages("start", "wire-1")[0]["op_id"] == "arm-wire-1"        # the op id crossed the wire
        assert out.data["clamped_frac_source"] == "body"                           # arm.progress came back on 5611
        assert body.messages("end", "wire-1")[-1]["args"]["hold_on_end"] == "target"
    finally:
        exe.close()
        srv.stop()


def test_health_says_so_when_the_body_has_no_chunk_mode(wire):
    body, w, client, port = wire
    w.modes = ["target"]                                   # the v0.5 op without B.8
    time.sleep(0.15)
    exe = GrootArmExecutor(FakeWorld(), arm=port, sensors=body, helpers=_groot_helpers())
    try:
        assert port.supports_chunk() is False
        h = exe.health()
        assert not h.ok and "chunk mode" in h.detail and "B.8" in h.detail
    finally:
        exe.close()


def test_a_hung_body_times_out_and_the_socket_recovers(wire):
    body, w, client, port = wire
    base = dict(stream="t", session_id="t", generation=1, control_epoch=0, mode="chunk")
    w.mute = True
    t0 = time.monotonic()
    rep = port.arm(base, "arm-t")
    assert rep["error"] == "body_timeout" and 0.4 <= time.monotonic() - t0 < 1.0
    w.mute = False
    assert port.arm(base, "arm-t2")["state"] == "accepted"     # a fresh DEALER after the timeout


def _jpeg_as_p1_sends(rgb: np.ndarray) -> str:
    """P1 hands the RGB array to cv2.imencode, which takes it as BGR: the JPEG's true colours are reversed."""
    import base64

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., ::-1])).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def test_zmq_sensors_read_p1_ego_view_and_g1_debug():
    import msgpack

    ctx = zmq.Context.instance()
    cam, dbg = ctx.socket(zmq.PUB), ctx.socket(zmq.PUB)
    cp = cam.bind_to_random_port("tcp://127.0.0.1")
    dp = dbg.bind_to_random_port("tcp://127.0.0.1")
    s = ZmqSensors(f"tcp://127.0.0.1:{cp}", f"tcp://127.0.0.1:{dp}", "ego_view", swap_rb=True).start()
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:, :320] = (200, 30, 30)                           # red left half, blue right half
    rgb[:, 320:] = (30, 30, 200)
    b64 = _jpeg_as_p1_sends(rgb)
    q = [0.01 * i for i in range(29)]
    try:
        t_end = time.monotonic() + 3.0
        frame = state = None
        t_cap = None
        while (frame is None or state is None) and time.monotonic() < t_end:
            t_cap = time.monotonic() - 0.04
            cam.send(msgpack.packb({"timestamps": {"ego_view": time.time()}, "images": {"ego_view": b64},
                                    "ego_view": b64, "camera": "ego_view", "t_capture_mono": t_cap},
                                   use_bin_type=True))
            dbg.send(b"g1_debug" + msgpack.packb({"body_q": q, "left_hand_q": [0.0] * 7,
                                                  "right_hand_q": [0.0] * 7}))       # one frame: topic + payload
            time.sleep(0.05)
            frame, t_f = s.ego_frame()
            state, t_d = s.debug_state()
        assert frame is not None and frame.shape == (480, 640, 3) and frame.dtype == np.uint8
        assert abs(int(frame[240, 100, 0]) - 200) < 12 and abs(int(frame[240, 500, 2]) - 200) < 12   # RGB, not BGR
        assert t_f <= time.monotonic() and time.monotonic() - t_f < 0.5                  # P1's capture time
        assert state["body_q"] == q and time.monotonic() - t_d < 0.5
        dbg.send_multipart([b"g1_debug", msgpack.packb({"body_q": q[::-1]})])           # multipart form too
        t_end = time.monotonic() + 2.0
        while s.debug_state()[0]["body_q"] != q[::-1] and time.monotonic() < t_end:
            time.sleep(0.02)
        assert s.debug_state()[0]["body_q"] == q[::-1]
    finally:
        s.close()
        cam.close(0)
        dbg.close(0)
