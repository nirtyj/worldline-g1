"""A fake GR00T PolicyServer for offline tests: a real ZMQ REP socket on 127.0.0.1 that speaks the upstream wire
(Isaac-GR00T @ 4b1dca9d `gr00t/policy/server_client.py`, as groot/wire.py and groot/policy_client.py document it):
request `{"endpoint", "data"}`, `ping` -> `{"status": "ok"}`, `get_action {observation, options}` -> `[action, info]`,
any failure -> `{"error": str}`.

The action is the Arena N1.7 G1 layout (groot.joint_order.GROOT_ACTION_KEYS, each (1, 40, D) float32): arms near
SONIC's stand pose with a small reach, hands open until `close_after` calls, then Arena's closed pose. Behaviour per
call is set by `mode` (or a callable of the call index):

    ok       a well-formed chunk
    nan      NaN in the left arm (the client must drop it)
    oob      every arm joint at 3.2 rad, beyond every G1 arm limit (the body clamps; >= 50 % of the values)
    error    `{"error": ...}` (a server-side exception, e.g. a strict observation check)

`latency_s` delays every reply (`first_latency_s` the first get_action: the N1.7 warm-up), `die()` closes the socket so
requests time out (the server is gone), `revive()` binds the same port again.

    srv = FakePolicyServer().start()
    PolicyClient(srv.endpoint).get_action(obs)
    srv.die(); srv.revive(); srv.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
import zmq

from groot import joint_order as jo
from groot.wire import packb, unpackb

# Arena's closed hand in GR00T order (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2), HF Static
# dataset_statistics (docs/groot_arms_design.md §2.5); the right hand is mirrored.
ARENA_CLOSED_GROOT = {"left": (-0.6, -1.2, -0.6, -1.2, 0.0, 0.7, 0.7), "right": (0.6, 1.2, 0.6, 1.2, 0.0, -0.7, -0.7)}


def make_action(T: int = 40, closed: bool = False, mode: str = "ok", phase: float = 0.0) -> dict[str, np.ndarray]:
    t = np.arange(T)[:, None] * jo.DT
    stand = np.asarray(jo.STAND_Q29, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for key in jo.GROOT_ACTION_KEYS:
        d = jo.GROOT_KEY_DIMS[key]
        if key in ("left_arm", "right_arm"):
            idx = [jo.MUJOCO_JOINTS.index(n) for n in jo.GROOT_KEY_JOINTS[key]]
            v = np.broadcast_to(stand[idx], (T, d)).copy()
            v[:, 3] += 0.15 * np.sin(2 * np.pi * 0.5 * (t[:, 0] + phase))      # a slow elbow reach
            if mode == "oob":
                v[:] = 3.2
            if mode == "nan" and key == "left_arm":
                v[5, 2] = np.nan
        elif key in ("left_hand", "right_hand"):
            side = key.split("_")[0]
            v = np.broadcast_to(np.asarray(ARENA_CLOSED_GROOT[side]) if closed else np.zeros(d), (T, d)).copy()
        elif key == "base_height_command":
            v = np.full((T, d), 0.75)
        else:                                          # waist, navigate_command: logged by the client, never executed
            v = np.zeros((T, d))
        out[key] = v.astype(np.float32)[None]
    return out


class FakePolicyServer:
    def __init__(self, *, port: int | None = None, latency_s: float = 0.03, first_latency_s: float | None = None,
                 mode: str | Callable[[int], str] = "ok", close_after: int | None = 2, horizon: int = 40):
        self.ctx = zmq.Context()
        self.port = port
        self.latency_s = latency_s
        self.first_latency_s = first_latency_s
        self.mode = mode
        self.close_after = close_after
        self.horizon = horizon
        self.calls = 0                     # get_action calls answered (or failed on purpose)
        self.pings = 0
        self.prompts: list[str] = []
        self.states: list[dict] = []
        self.t_calls: list[float] = []
        self.dead = False
        self._sock: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._run = False

    @property
    def endpoint(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    def start(self) -> "FakePolicyServer":
        s = self.ctx.socket(zmq.REP)
        s.setsockopt(zmq.LINGER, 0)
        if self.port is None:
            self.port = s.bind_to_random_port("tcp://127.0.0.1")
        else:
            s.bind(self.endpoint)
        self._sock = s
        self._run = True
        self.dead = False
        self._thread = threading.Thread(target=self._loop, name="fake-policy-server", daemon=True)
        self._thread.start()
        return self

    def _mode(self, i: int) -> str:
        return self.mode(i) if callable(self.mode) else self.mode

    def _handle(self, req: dict) -> Any:
        ep = req.get("endpoint", "get_action")
        if ep == "ping":
            self.pings += 1
            return {"status": "ok", "message": "Server is running"}
        if ep == "get_action":
            i = self.calls
            self.calls += 1
            self.t_calls.append(time.monotonic())
            delay = self.first_latency_s if (i == 0 and self.first_latency_s is not None) else self.latency_s
            if delay:
                time.sleep(delay)
            obs = (req.get("data") or {}).get("observation") or {}
            lang = (obs.get("language") or {}).get(jo.LANGUAGE_KEY)
            if lang:
                self.prompts.append(str(lang[0][0]))
            self.states.append({k: np.asarray(v) for k, v in (obs.get("state") or {}).items()})
            mode = self._mode(i)
            if mode == "error":
                return {"error": "fake server: observation check failed"}
            closed = self.close_after is not None and i >= self.close_after
            return [make_action(self.horizon, closed=closed, mode=mode, phase=0.4 * i), {}]
        if ep in ("reset", "kill"):
            return {}
        if ep == "get_modality_config":
            return {}
        return {"error": f"Unknown endpoint: {ep}"}

    def _loop(self) -> None:
        s = self._sock
        while self._run:
            try:
                if not s.poll(50):
                    continue
                req = unpackb(s.recv())
            except zmq.ZMQError:
                return
            try:
                rep = self._handle(req)
            except Exception as e:  # noqa: BLE001 - the upstream server replies {"error"} to any exception
                rep = {"error": str(e)}
            if not self._run:                  # died while "computing": no reply, like a crashed server
                return
            try:
                s.send(packb(rep))
            except zmq.ZMQError:
                return

    def die(self) -> None:
        """The server process is gone: no replies, the port unbound."""
        self._run = False
        self.dead = True
        if self._thread is not None:
            self._thread.join(timeout=2.0 + (self.latency_s or 0.0))
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def revive(self) -> "FakePolicyServer":
        return self.start()

    def stop(self) -> None:
        self.die()
        self.ctx.term()
