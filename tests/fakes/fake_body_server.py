"""A scripted fake of wl-body (P3) on the M1 wire (docs/contracts/m1.md §3.4): ROUTER body_ctl + PUB body_evt.

It lets robot/body_client.SonicBody run over the real body.client.BodyClient without SONIC or Isaac:
    go_to / turn_to / walk   accepted, then `succeeded` after `motion_s` (or `fail_reason` -> failed); a new
                             motion pre-empts the active one (canceled {reason: preempted})
    stop                     the active motion ends canceled {reason: stop}; stop succeeds
    status, ping             body.state / ok
    shutdown_control         {confirm: true} -> ok (recorded)
    reject_reason            a motion op is refused synchronously (reply ok=false, error=<reason>)
body.state is published at 20 Hz so BodyClient.connect() returns quickly.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import zmq


class FakeBodyServer:
    def __init__(self, port_offset: int = 450, motion_s: float = 0.6):
        self.off = port_offset
        self.motion_s = motion_s
        self.fail_reason: str | None = None
        self.reject_reason: str | None = None
        self.reply_delay_s = 0.0
        self.requests: list[dict] = []
        self.active: dict | None = None
        self.pose = [0.0, 0.0, 0.0]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.ctx = zmq.Context.instance()

    def start(self) -> "FakeBodyServer":
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://127.0.0.1:{5610 + self.off}")
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://127.0.0.1:{5611 + self.off}")
        self._t = [threading.Thread(target=self._serve, daemon=True), threading.Thread(target=self._tick, daemon=True)]
        for t in self._t:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in self._t:
            t.join(timeout=1.0)
        self.router.close(0)
        self.pub.close(0)

    # ------------------------------------------------------------------
    def _event(self, op_id: str, op: str, state: str, data: dict | None = None) -> None:
        with self._lock:
            self.pub.send_multipart([b"body.event", json.dumps({"id": op_id, "op": op, "state": state,
                                                                "data": data or {}, "t_wall": time.time()}).encode()])

    def _tick(self) -> None:
        while not self._stop.is_set():
            a = self.active
            if a is not None and time.monotonic() >= a["t_end"]:
                self.active = None
                if self.fail_reason:
                    self._event(a["id"], a["op"], "failed", {"reason": self.fail_reason, "walked_m": 0.4})
                else:
                    self._event(a["id"], a["op"], "succeeded", {"path_len_m": 2.0, "walked_m": 2.0,
                                                                  "final_pose": list(self.pose), "replans": 0})
            st = {"in_control": True, "fault": None, "active": {"id": a["id"], "op": a["op"]} if a else None,
                  "pose": {"x": self.pose[0], "y": self.pose[1], "yaw": self.pose[2]},
                  "gt_pose": {"rate_hz": 50.0, "age_s": 0.01, "rtf": 1.0}, "deploy": {"alive": True}}
            with self._lock:
                self.pub.send_multipart([b"body.state", json.dumps(st).encode()])
            time.sleep(0.05)

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self.router, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(20):
                continue
            ident, raw = self.router.recv_multipart()[:2]
            req = json.loads(raw)
            self.requests.append(req)
            op, op_id, args = req.get("op"), req.get("id") or uuid.uuid4().hex[:8], req.get("args") or {}
            if self.reply_delay_s:
                time.sleep(self.reply_delay_s)
            rep = self._handle(op, op_id, args)
            self.router.send_multipart([ident, json.dumps({"id": op_id, **rep}).encode()])

    def _handle(self, op: str, op_id: str, args: dict) -> dict:
        if op in ("go_to", "turn_to", "walk"):
            if self.reject_reason:
                self._event(op_id, op, "failed", {"reason": self.reject_reason})
                return {"ok": False, "state": "rejected", "error": self.reject_reason}
            if self.active is not None:
                prev = self.active
                self.active = None
                self._event(prev["id"], prev["op"], "canceled", {"reason": "preempted", "by": op_id})
            self.active = {"id": op_id, "op": op, "t_end": time.monotonic() + self.motion_s, "args": args}
            self._event(op_id, op, "accepted", {})
            return {"ok": True, "state": "accepted"}
        if op == "stop":
            if self.active is not None:
                prev = self.active
                self.active = None
                self._event(prev["id"], prev["op"], "canceled", {"reason": "stop", "walked_m": 0.3})
            self._event(op_id, op, "succeeded", {"stopped": True, "stop_time_s": 0.5})
            return {"ok": True, "state": "done", "data": {"stopped": True}}
        if op in ("status", "ping"):
            return {"ok": True, "state": "done", "data": {"in_control": True}}
        if op == "shutdown_control":
            return {"ok": bool(args.get("confirm")), "state": "done"}
        return {"ok": False, "state": "rejected", "error": f"unknown op {op}"}
