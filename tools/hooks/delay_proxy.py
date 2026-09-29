"""A ZMQ REQ/REP delay proxy for G4 `stale_chunk_after_correction` (PLAN 3.3 port 5551, PLAN 9.3 G4). TEST-ONLY.

    python -m tools.hooks.delay_proxy --listen 5551 --upstream 5550 --ctl 5549 [--ms 0] [--where reply]

ROUTER on --listen (the runtime's GrootArmClient connects here: P5 started with WL_GROOT_ENDPOINT=tcp://127.0.0.1:5551),
DEALER to --upstream (the GR00T PolicyServer, REP 5550). The standard broker envelope passes through untouched, so the
proxy is transparent to upstream's msgpack wire (groot/policy_client.py). Every reply (or request, --where request) is
held `ms` before it is forwarded: with 600 ms a chunk computed before a correction arrives after the cancel ack, which
must then be dropped as stale (docs/contracts/arm_chunk.md; G4: zero stale-session chunks published).

Control: REP on --ctl, JSON {"op": "set", "ms": N} | {"op": "get"} | {"op": "stop"} -> {"ok", "ms", "where", stats}.
ms = 0 is a plain pass-through, so the proxy can stay in the loop for a whole stack-suite run and only G4 turns the
delay on (tools/hooks `delay-proxy set --ms 600`, then `--ms 0`). With the PolicyServer down a request is dropped (the
client times out, as on a direct link) instead of being queued for a server that comes back (G5/G13 kill it).
"""
from __future__ import annotations

import argparse
import collections
import json
import signal
import sys
import threading
import time
from typing import Any

import zmq


class DelayProxy:
    def __init__(self, listen: str, upstream: str, ctl: str | None = None, delay_ms: float = 0.0,
                 where: str = "reply", ctx: zmq.Context | None = None, log=print):
        if where not in ("reply", "request"):
            raise ValueError("where must be reply or request")
        self.listen, self.upstream, self.ctl_ep = listen, upstream, ctl
        self.delay_ms = float(delay_ms)
        self.where = where
        self.ctx = ctx or zmq.Context.instance()
        self.log = log
        self.stats = {"requests": 0, "replies": 0, "held": 0, "max_hold_ms": 0.0, "delay_changes": 0,
                      "dropped": 0}
        self.t_start = time.time()
        self._stop = threading.Event()
        self._front = self._back = self._ctl = None

    def _bind(self) -> None:
        z = zmq
        self._front = self.ctx.socket(z.ROUTER)
        self._front.setsockopt(z.LINGER, 0)
        self._front.bind(self.listen)
        self._back = self.ctx.socket(z.DEALER)
        self._back.setsockopt(z.LINGER, 0)
        self._back.setsockopt(z.IMMEDIATE, 1)      # upstream down: drop (the client times out), never queue stale
        self._back.connect(self.upstream)         # requests for a PolicyServer that comes back later
        if self.ctl_ep:
            self._ctl = self.ctx.socket(z.REP)
            self._ctl.setsockopt(z.LINGER, 0)
            self._ctl.bind(self.ctl_ep)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, Any]:
        return {"ms": self.delay_ms, "where": self.where, "listen": self.listen, "upstream": self.upstream,
                "uptime_s": round(time.time() - self.t_start, 1), **self.stats}

    def _control(self, raw: bytes) -> dict[str, Any]:
        try:
            req = json.loads(raw)
            op = req.get("op", "get")
            if op == "set":
                ms = float(req.get("ms", 0.0))
                if not 0.0 <= ms <= 10000.0:
                    return {"ok": False, "error": f"ms must be in [0, 10000], got {ms}"}
                self.delay_ms = ms
                self.stats["delay_changes"] += 1
                self.log(f"[delay_proxy] delay {ms:.0f} ms ({self.where})")
            elif op == "stop":
                self._stop.set()
            elif op != "get":
                return {"ok": False, "error": f"unknown op {op!r}"}
            return {"ok": True, **self.snapshot()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e)}

    def run(self) -> None:
        self._bind()
        poller = zmq.Poller()
        poller.register(self._front, zmq.POLLIN)
        poller.register(self._back, zmq.POLLIN)
        if self._ctl is not None:
            poller.register(self._ctl, zmq.POLLIN)
        pend_req: collections.deque = collections.deque()     # (due_mono, t_in, frames)
        pend_rep: collections.deque = collections.deque()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                due = [q[0][0] for q in (pend_req, pend_rep) if q]
                wait_ms = 50 if not due else max(0, min(50, int((min(due) - now) * 1000)))
                for sock, _ in poller.poll(wait_ms):
                    if sock is self._front:
                        fr = self._front.recv_multipart()
                        self.stats["requests"] += 1
                        d = self.delay_ms / 1000.0 if self.where == "request" else 0.0
                        pend_req.append((time.monotonic() + d, time.monotonic(), fr))
                    elif sock is self._back:
                        fr = self._back.recv_multipart()
                        self.stats["replies"] += 1
                        d = self.delay_ms / 1000.0 if self.where == "reply" else 0.0
                        pend_rep.append((time.monotonic() + d, time.monotonic(), fr))
                    elif sock is self._ctl:
                        self._ctl.send(json.dumps(self._control(self._ctl.recv())).encode())
                now = time.monotonic()
                for q, out in ((pend_req, self._back), (pend_rep, self._front)):
                    while q and q[0][0] <= now:
                        _, t_in, fr = q.popleft()
                        held = (now - t_in) * 1e3
                        if held > 1.0:
                            self.stats["held"] += 1
                        self.stats["max_hold_ms"] = round(max(self.stats["max_hold_ms"], held), 1)
                        try:
                            out.send_multipart(fr, flags=zmq.NOBLOCK)
                        except zmq.Again:                       # no upstream connected (P4 down)
                            self.stats["dropped"] += 1
        finally:
            for s in (self._front, self._back, self._ctl):
                if s is not None:
                    s.close(0)


def ctl_call(ctl: str, msg: dict, timeout_s: float = 2.0, ctx: zmq.Context | None = None) -> dict:
    """One control request; {"ok": False, "error": "no reply ..."} when the proxy is not running."""
    ctx = ctx or zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    try:
        s.connect(ctl)
        s.send(json.dumps(msg).encode())
        return json.loads(s.recv())
    except zmq.ZMQError as e:
        return {"ok": False, "error": f"no reply from the delay proxy at {ctl}: {e}"}
    finally:
        s.close(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=5551)
    ap.add_argument("--upstream", type=int, default=5550)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--ctl", type=int, default=5549)
    ap.add_argument("--ms", type=float, default=0.0)
    ap.add_argument("--where", choices=["reply", "request"], default="reply")
    a = ap.parse_args(argv)
    p = DelayProxy(f"tcp://127.0.0.1:{a.listen}", f"tcp://{a.upstream_host}:{a.upstream}",
                   f"tcp://127.0.0.1:{a.ctl}", a.ms, a.where,
                   log=lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True))
    signal.signal(signal.SIGTERM, lambda *_: p.stop())
    signal.signal(signal.SIGINT, lambda *_: p.stop())
    print(f"[delay_proxy] 127.0.0.1:{a.listen} -> {a.upstream_host}:{a.upstream}, ctl {a.ctl}, "
          f"{a.ms:.0f} ms on the {a.where} (TEST-ONLY)", flush=True)
    p.run()
    print(f"[delay_proxy] stopped: {json.dumps(p.snapshot())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
