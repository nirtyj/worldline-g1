"""Fake body ROUTER for viz tests (no Isaac, no SONIC): records every request and answers like body/service.py.

    python viz/tests/fake_body.py --port 5710 [--no-velocity] [--log /tmp/router.jsonl]

With velocity (default) it follows body/velocity.py route_velocity: a `velocity` message whose `stream` is not the
active velocity op starts a new op (reply "accepted", op counted), one with the active stream updates it (reply
"done", no op), `end: true` ends it. Any other motion op (stand, walk, go_to, turn_to, stop) pre-empts the active
one. --no-velocity answers `velocity` with "unknown op 'velocity'", like a body without the streaming op.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import zmq


class FakeBody:
    def __init__(self, port: int, velocity: bool = True, log_path: str | None = None):
        self.port, self.velocity = port, velocity
        self.reqs: list[dict] = []          # every request: {t, op, args, id, reply}
        self.ops_started: list[dict] = []   # requests that started a body op (reply "accepted")
        self.active: dict | None = None     # {"op", "id", "stream"?}
        self._log = open(log_path, "a", buffering=1) if log_path else None
        self._stop = threading.Event()
        self._sock = zmq.Context.instance().socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://127.0.0.1:{port}")
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _reply(self, req: dict) -> dict:
        op, args = req.get("op"), req.get("args") or {}
        rid = req.get("id")
        if op in ("status", "ping"):
            return {"id": rid, "ok": True, "state": "done", "data": {"active": self.active, "note": "fake body"}}
        if op == "velocity":
            if not self.velocity:
                return {"id": rid, "ok": False, "state": "rejected", "error": "unknown op 'velocity'"}
            a = self.active
            if a and a["op"] == "velocity" and a.get("stream") == str(args.get("stream")):
                if args.get("end"):
                    self.active = None
                    return {"id": rid, "ok": True, "state": "done", "data": {"id": a["id"], "ended": True}}
                return {"id": rid, "ok": True, "state": "done", "data": {"id": a["id"], "accepted": True}}
            self.active = None if args.get("end") else {"op": "velocity", "id": rid, "stream": str(args.get("stream"))}
            self.ops_started.append({"op": op, "args": args, "id": rid})
            return {"id": rid, "ok": True, "state": "accepted", "data": {"stream": args.get("stream")}}
        if op in ("stand", "walk", "go_to", "turn_to", "stop", "clear_fault"):
            self.active = None if op in ("stop", "stand", "clear_fault") else {"op": op, "id": rid}
            self.ops_started.append({"op": op, "args": args, "id": rid})
            return {"id": rid, "ok": True, "state": "accepted"}
        return {"id": rid, "ok": False, "state": "rejected", "error": f"unknown op {op!r}"}

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._sock.poll(50):
                continue
            frames = self._sock.recv_multipart()
            try:
                req = json.loads(frames[-1])
            except Exception:  # noqa: BLE001
                continue
            rep = self._reply(req)
            rec = {"t": time.time(), "op": req.get("op"), "args": req.get("args"), "id": req.get("id"),
                   "reply": rep.get("state"), "frames": len(frames)}
            self.reqs.append(rec)
            if self._log:
                self._log.write(json.dumps(rec) + "\n")
            self._sock.send_multipart(frames[:-1] + [json.dumps(rep).encode()])

    def close(self) -> None:
        self._stop.set()
        self._th.join(2)
        self._sock.close(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    fb = FakeBody(a.port, velocity=not a.no_velocity, log_path=a.log)
    print(f"fake body ROUTER on {a.port} (velocity {'on' if fb.velocity else 'off'})", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fb.close()
