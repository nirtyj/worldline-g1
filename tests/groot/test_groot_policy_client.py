"""groot.policy_client + groot.wire against an in-process REP server that speaks the upstream wire.

`RefSerializer` is a verbatim copy of Isaac-GR00T @ 4b1dca9d `gr00t/policy/server_client.py:30-60` (MsgSerializer),
with ModalityConfig replaced by a stand-in class, so these tests pin byte compatibility without importing gr00t.
"""

from __future__ import annotations

import ast
import io
import subprocess
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest
import zmq

from groot import joint_order as jo
from groot import wire
from groot.actions import to_arm_chunk
from groot.obs import ARENA_PROMPT, build_observation
from groot.policy_client import PolicyClient, PolicyError, PolicyTimeout, normalize_endpoint

ROOT = Path(__file__).resolve().parents[2]


class FakeModalityConfig:
    def __init__(self, delta_indices, modality_keys):
        self.delta_indices, self.modality_keys = delta_indices, modality_keys


class RefSerializer:                                  # server_client.py:30-60
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=RefSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=RefSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return FakeModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, FakeModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": dict(vars(obj))}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class FakeServer:
    """PolicyServer.run() (server_client.py:132-164) with a scripted policy. `delay_s` delays get_action."""

    def __init__(self, delay_s: float = 0.0, first_only: bool = False):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.endpoint = f"tcp://127.0.0.1:{self.port}"
        self.delay_s, self.first_only = delay_s, first_only
        self.requests: list[dict] = []
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_action(self, observation, options=None):
        if self.delay_s and (not self.first_only or sum(r.get("endpoint") == "get_action"
                                                         for r in self.requests) == 1):
            time.sleep(self.delay_s)
        for k, v in observation["state"].items():
            assert v.dtype == np.float32 and v.ndim == 3, k
        if observation["language"]["annotation.human.task_description"] == [["explode"]]:
            raise ValueError("strict check failed")
        act = {k: np.full((1, 40, jo.GROOT_KEY_DIMS[k]), 0.1 * i, np.float32)
               for i, k in enumerate(jo.GROOT_ACTION_KEYS)}
        return act, {"note": "fake"}

    def run(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while self.running:
            if not dict(poller.poll(50)):
                continue
            request = RefSerializer.from_bytes(self.sock.recv())
            self.requests.append(request)
            try:
                ep = request.get("endpoint", "get_action")
                if ep == "ping":
                    result = {"status": "ok", "message": "Server is running"}
                elif ep == "get_action":
                    result = self.get_action(**request.get("data", {}))
                elif ep == "get_modality_config":
                    result = {"video": FakeModalityConfig([0], ["ego_view"])}
                elif ep == "reset":
                    result = {}
                else:
                    raise ValueError(f"Unknown endpoint: {ep}")
                self.sock.send(RefSerializer.to_bytes(result))
            except Exception as e:            # noqa: BLE001 (upstream replies {"error": str(e)})
                self.sock.send(RefSerializer.to_bytes({"error": str(e)}))

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.sock.close(linger=0)
        self.ctx.term()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def _obs(prompt=ARENA_PROMPT):
    return build_observation(np.full((480, 640, 3), 7, np.uint8), np.zeros(29), np.zeros(7), np.zeros(7), prompt)


def test_wire_is_byte_compatible_with_upstream():
    obs = _obs()
    ours = wire.packb({"endpoint": "get_action", "data": {"observation": obs, "options": None}})
    ref = RefSerializer.to_bytes({"endpoint": "get_action", "data": {"observation": obs, "options": None}})
    assert ours == ref
    back = RefSerializer.from_bytes(ours)
    assert np.array_equal(back["data"]["observation"]["video"]["ego_view"], obs["video"]["ego_view"])
    act = ({"left_arm": np.ones((1, 40, 7), np.float32)}, {})
    dec = wire.unpackb(RefSerializer.to_bytes(act))
    assert dec[0]["left_arm"].dtype == np.float32 and dec[0]["left_arm"].shape == (1, 40, 7)
    mc = wire.unpackb(RefSerializer.to_bytes({"video": FakeModalityConfig([0], ["ego_view"])}))
    assert mc == {"video": {"delta_indices": [0], "modality_keys": ["ego_view"]}}
    assert wire.unpackb(wire.packb({"x": np.float32(1.5)})) == {"x": 1.5}
    with pytest.raises(TypeError):
        wire.packb({"x": object()})
    with pytest.raises(ValueError):                  # object arrays never travel (no pickle)
        wire.packb({"x": np.array([{"a": 1}], dtype=object)})


def test_normalize_endpoint():
    assert normalize_endpoint("127.0.0.1:5550") == "tcp://127.0.0.1:5550"
    assert normalize_endpoint(" tcp://h:1 ") == "tcp://h:1"


def test_ping_and_get_action_roundtrip(server):
    with PolicyClient(server.endpoint, timeout_s=2.0) as c:
        assert c.ping()
        assert c.last_latency_s is not None and c.last_latency_s < 2.0
        act = c.get_action(_obs())
        assert c.last_info == {"note": "fake"}
        assert set(act) == set(jo.GROOT_ACTION_KEYS)
        assert act["left_arm"].shape == (1, 40, 7) and act["left_arm"].dtype == np.float32
        assert c.last_request_bytes > 480 * 640 * 3                 # the raw frame is on the wire
        chunk = to_arm_chunk(act, time.monotonic())
        assert chunk.T == 40
        req = server.requests[-1]
        assert req["endpoint"] == "get_action" and "api_token" not in req
        sent = req["data"]["observation"]
        assert sent["video"]["ego_view"].shape == (1, 1, 480, 640, 3)
        assert sent["language"] == {"annotation.human.task_description": [[ARENA_PROMPT]]}
        assert server.requests[0] == {"endpoint": "ping"}          # no "data" for input-less endpoints
        assert c.get_modality_config()["video"]["modality_keys"] == ["ego_view"]
        assert c.reset() == {}
        assert c.stats["ok"] == 4 and c.stats["timeouts"] == 0


def test_server_error_raises_and_socket_stays_usable(server):
    with PolicyClient(server.endpoint, timeout_s=2.0) as c:
        with pytest.raises(PolicyError, match="strict check failed"):
            c.get_action(_obs("explode"))
        with pytest.raises(PolicyError, match="Unknown endpoint"):
            c.call("nope", requires_input=False)
        assert c.ping()
        assert c.stats["errors"] == 2 and c.stats["socket_resets"] == 0


def test_timeout_recreates_socket_then_recovers():
    s = FakeServer(delay_s=0.6, first_only=True)
    try:
        with PolicyClient(s.endpoint, timeout_s=0.2) as c:
            t0 = time.monotonic()
            with pytest.raises(PolicyTimeout):
                c.get_action(_obs())
            assert time.monotonic() - t0 < 0.5
            assert c.stats["timeouts"] == 1 and c.stats["socket_resets"] == 1
            # a wedged REQ would raise "Operation cannot be accomplished in current state" here
            assert c.ping(timeout_s=2.0)
            act = c.get_action(_obs(), timeout_s=2.0)
            assert act["right_hand"].shape == (1, 40, 7)
    finally:
        s.close()


def test_dead_endpoint_fails_fast_and_keeps_working():
    ctx = zmq.Context()
    probe = ctx.socket(zmq.REP)
    port = probe.bind_to_random_port("tcp://127.0.0.1")
    probe.close(linger=0)                             # a free port with nothing listening
    ctx.term()
    c = PolicyClient(f"127.0.0.1:{port}", timeout_s=0.15)
    t0 = time.monotonic()
    assert c.ping() is False and c.ping() is False
    assert time.monotonic() - t0 < 1.0
    with pytest.raises(PolicyTimeout):
        c.get_action(_obs())
    assert c.stats["timeouts"] == 3 and c.stats["socket_resets"] == 3
    c.close()
    c.close()                                         # idempotent
    with pytest.raises(RuntimeError):
        c.ping()


def test_groot_package_never_imports_gr00t_or_body():
    for path in sorted((ROOT / "groot").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for n in names:
                top = n.split(".")[0]
                assert top not in ("gr00t", "torch", "body", "world", "services", "robot"), f"{path.name}: {n}"


@pytest.mark.parametrize("script", ["scripts/groot_server.sh", "scripts/groot_link.sh"])
def test_scripts_parse(script):
    r = subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
