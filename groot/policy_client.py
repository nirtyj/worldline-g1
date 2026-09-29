"""PolicyClient: the runtime's client of the Isaac-GR00T PolicyServer (P4, 127.0.0.1:5550).

The wire is upstream's (Isaac-GR00T @ 4b1dca9d, `gr00t/policy/server_client.py`):
- transport: ZMQ REQ -> REP; the server binds `tcp://{host}:{port}` (:84-86) and serves one request at a time (:132-164);
- request: msgpack map `{"endpoint": name, "data": {...}}` (+ `"api_token"` when the server has one; ours does not),
  `"data"` omitted for endpoints without input (:210-225); encoding in groot/wire.py (:30-60);
- endpoints (:90-99): `ping` -> `{"status": "ok", "message": ...}`, `get_action(observation, options)` ->
  `[action, info]` (a msgpack array: `BasePolicy.get_action` returns a tuple, gr00t/policy/policy.py:80-105),
  `reset(options)`, `get_modality_config`, `kill` (never sent from here);
- errors: the server replies `{"error": str}` for any exception, including a strict-mode observation check failure
  (:159-164); an unauthorized request gets `{"error": "Unauthorized: ..."}` (:141-145).

Timeouts: a REQ socket that sent a request and never got the reply is wedged (it refuses the next send). Upstream's
client recreates the socket on `zmq.Again` (:227-235); this client does the same on every timeout or ZMQ error, with
LINGER 0 so a dead socket never blocks close. The server does not know the client gave up: it still finishes the old
request (and its reply is dropped by ZMQ), so the next call can wait up to one extra inference.

One PolicyClient per thread (ZMQ sockets are not thread-safe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
import zmq

from . import DEFAULT_ENDPOINT
from .wire import packb, unpackb


class PolicyError(RuntimeError):
    """The server answered with an error (bad observation, unknown endpoint, unauthorized, ...)."""


class PolicyTimeout(TimeoutError):
    """No reply within the timeout (server down, busy, or the link is broken). The socket has been recreated."""


def normalize_endpoint(endpoint: str) -> str:
    """'127.0.0.1:5550' | 'tcp://127.0.0.1:5550' -> 'tcp://127.0.0.1:5550'."""
    e = endpoint.strip()
    if "://" not in e:
        e = "tcp://" + e
    return e


class PolicyClient:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout_s: float = 1.5, *, api_token: str | None = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.endpoint = normalize_endpoint(endpoint)
        self.timeout_s = float(timeout_s)
        self.api_token = api_token
        self._ctx = zmq.Context()
        self._sock: zmq.Socket | None = None
        self.stats = {"calls": 0, "ok": 0, "timeouts": 0, "errors": 0, "socket_resets": 0}
        self.last_latency_s: float | None = None
        self.last_info: Any = None
        self.last_request_bytes = 0
        self.last_reply_bytes = 0
        self._open()

    # -- socket ---------------------------------------------------------------------------------------------
    def _open(self) -> None:
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        s.connect(self.endpoint)
        self._sock = s

    def _reset_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self.stats["socket_resets"] += 1
        self._open()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if not self._ctx.closed:
            self._ctx.term()

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- calls ----------------------------------------------------------------------------------------------
    def call(self, endpoint: str, data: dict | None = None, *, requires_input: bool = True,
             timeout_s: float | None = None) -> Any:
        """One request/reply. Raises PolicyTimeout (socket recreated) or PolicyError."""
        if self._sock is None:
            raise RuntimeError("PolicyClient is closed")
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        if self.api_token:
            request["api_token"] = self.api_token
        payload = packb(request)
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        self.stats["calls"] += 1
        t0 = time.monotonic()
        try:
            self._sock.send(payload)
            if not self._sock.poll(int(timeout * 1000), zmq.POLLIN):
                self.stats["timeouts"] += 1
                self._reset_socket()
                raise PolicyTimeout(f"{endpoint}: no reply from {self.endpoint} within {timeout:.2f} s")
            reply = self._sock.recv(zmq.NOBLOCK)
        except zmq.ZMQError as e:
            self.stats["errors"] += 1
            self._reset_socket()
            raise PolicyTimeout(f"{endpoint}: ZMQ error on {self.endpoint}: {e}") from e
        self.last_latency_s = time.monotonic() - t0
        self.last_request_bytes, self.last_reply_bytes = len(payload), len(reply)
        if reply == b"ERROR":
            self.stats["errors"] += 1
            raise PolicyError(f"{endpoint}: server error (wrong policy server?)")
        resp = unpackb(reply)
        if isinstance(resp, dict) and "error" in resp:
            self.stats["errors"] += 1
            raise PolicyError(f"{endpoint}: {resp['error']}")
        self.stats["ok"] += 1
        return resp

    def ping(self, timeout_s: float | None = None) -> bool:
        try:
            resp = self.call("ping", requires_input=False, timeout_s=timeout_s)
        except (PolicyTimeout, PolicyError):
            return False
        return isinstance(resp, dict) and resp.get("status") == "ok"

    def get_action(self, obs: dict, options: dict | None = None, *, timeout_s: float | None = None
                   ) -> dict[str, np.ndarray]:
        """obs (groot.obs.build_observation) -> {action key: float32 array (B, T, D)}. The info dict goes to last_info."""
        resp = self.call("get_action", {"observation": obs, "options": options}, timeout_s=timeout_s)
        if isinstance(resp, (list, tuple)) and len(resp) == 2 and isinstance(resp[0], dict):
            action, self.last_info = resp
        elif isinstance(resp, dict):                   # a server that returns the action dict alone
            action, self.last_info = resp, {}
        else:
            raise PolicyError(f"get_action: unexpected reply type {type(resp).__name__}")
        out = {}
        for k, v in action.items():
            if not isinstance(v, np.ndarray):
                raise PolicyError(f"get_action: key {k!r} is {type(v).__name__}, not an array")
            out[str(k)] = v
        return out

    def get_modality_config(self, timeout_s: float | None = None) -> dict:
        return self.call("get_modality_config", requires_input=False, timeout_s=timeout_s)

    def reset(self, options: dict | None = None, timeout_s: float | None = None) -> Any:
        return self.call("reset", {"options": options}, timeout_s=timeout_s)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m groot.policy_client", description="Talk to a GR00T PolicyServer")
    ap.add_argument("cmd", choices=["ping", "modality"])
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--timeout", type=float, default=1.5)
    a = ap.parse_args(argv)
    with PolicyClient(a.endpoint, timeout_s=a.timeout) as c:
        if a.cmd == "ping":
            ok = c.ping()
            lat = None if c.last_latency_s is None else round(c.last_latency_s * 1000, 2)
            print(json.dumps({"endpoint": c.endpoint, "ok": ok, "latency_ms": lat if ok else None}))
            return 0 if ok else 1
        print(json.dumps(c.get_modality_config(), indent=1, default=str))
        return 0


if __name__ == "__main__":
    sys.exit(_main())
