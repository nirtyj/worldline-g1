"""BodyClient: the Python API for driving the G1 through wl-body (ROUTER 5610 / PUB 5611).

    from body.client import BodyClient
    with BodyClient() as bc:                       # port offset from WL_PORT_OFFSET (default 0)
        bc.stand()                                 # blocks until succeeded/failed; returns OpHandle
        bc.walk(vx=0.5, duration_s=5)
        h = bc.go_to(3.0, 1.0, yaw=1.57, wait=False)   # async handle
        h.wait(120); h.state, h.result
        bc.stop()
        bc.pose(); bc.status(); ts, img = bc.camera_frame()

Every motion returns an OpHandle whose terminal state is one of succeeded | failed | canceled (rejections surface
as failed with data.reason). Nothing here fakes a result: the state is what wl-body reported.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

import zmq

from .config import ep, ports as _ports, port_offset_from_env
from .wire import decode_camera_message, dumps_json

TERMINAL = ("succeeded", "failed", "canceled")


class OpHandle:
    def __init__(self, client: "BodyClient", op_id: str, op: str, args: dict):
        self.client = client
        self.id = op_id
        self.op = op
        self.args = args
        self.state = "pending"
        self.reply: dict | None = None
        self.result: dict | None = None
        self.events: list[dict] = []
        self._done = threading.Event()
        self._cbs = []
        self.t_submit = time.time()
        self.t_done: float | None = None

    def _on_event(self, ev: dict) -> None:
        self.events.append(ev)
        st = ev.get("state")
        if st in TERMINAL or st == "accepted":
            if self.state not in TERMINAL:
                self.state = st
        if st in TERMINAL and not self._done.is_set():
            self.result = ev.get("data") or {}
            self.t_done = time.time()
            self._done.set()
        for cb in list(self._cbs):
            try:
                cb(self, ev)
            except Exception:
                pass

    def on_event(self, cb) -> "OpHandle":
        self._cbs.append(cb)
        return self

    def done(self) -> bool:
        return self._done.is_set()

    @property
    def ok(self) -> bool:
        return self.state == "succeeded"

    @property
    def reason(self) -> str | None:
        return (self.result or {}).get("reason")

    def wait(self, timeout: float | None = None, poll_s: float = 2.0) -> "OpHandle":
        t_end = None if timeout is None else time.monotonic() + timeout
        while not self._done.is_set():
            rem = poll_s if t_end is None else min(poll_s, t_end - time.monotonic())
            if rem <= 0:
                raise TimeoutError(f"op {self.op} {self.id} not finished (state={self.state})")
            if self._done.wait(rem):
                break
            # safety net: the event may have been missed (reconnect); ask the service
            try:
                st = self.client.status(op_id=self.id)
                rec = (st or {}).get("op")
                if rec and rec.get("state") in TERMINAL and not self._done.is_set():
                    self._on_event({"id": self.id, "op": self.op, "state": rec["state"],
                                    "data": rec.get("result") or {}, "via": "status"})
            except Exception:
                pass
        return self

    def cancel(self, wait: bool = True, timeout: float = 5.0) -> "OpHandle":
        """Cancel = body stop (planner IDLE). This op ends 'canceled'; returns the stop op's handle."""
        return self.client.stop(wait=wait, timeout=timeout)

    def summary(self) -> dict:
        return {"id": self.id, "op": self.op, "args": self.args, "state": self.state, "result": self.result,
                "duration_s": None if self.t_done is None else round(self.t_done - self.t_submit, 3)}

    def __repr__(self) -> str:
        return f"OpHandle({self.op} {self.id} {self.state} {self.reason or ''})"


class BodyClient:
    def __init__(self, port_offset: int | None = None, host: str = "127.0.0.1", rpc_timeout_s: float = 10.0,
                 ctx: zmq.Context | None = None):
        off = port_offset_from_env() if port_offset is None else port_offset
        self.ports = _ports(off)
        self.host = host
        self.rpc_timeout_ms = int(rpc_timeout_s * 1000)
        self.ctx = ctx or zmq.Context.instance()
        self._lock = threading.Lock()
        self._handles: dict[str, OpHandle] = {}
        self._listeners = []
        self.last_state: dict | None = None
        self.last_state_mono = 0.0
        self._running = False
        self._dealer: zmq.Socket | None = None
        self._evt_thread: threading.Thread | None = None
        self._pose_sub = None
        self._cam: zmq.Socket | None = None

    # -- lifecycle -------------------------------------------------------------------------------
    def connect(self, wait_s: float = 10.0) -> "BodyClient":
        self._running = True
        self._evt_ready = threading.Event()
        self._evt_thread = threading.Thread(target=self._evt_loop, name="body-evt", daemon=True)
        self._evt_thread.start()
        self._dealer = self._new_dealer()
        # the first body.state proves both that the service is up and that our SUB is live (no slow-joiner loss)
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if self.last_state is not None:
                return self
            time.sleep(0.05)
        raise TimeoutError(f"no body.state on {ep(self.ports['body_evt'], self.host)} within {wait_s}s")

    def _new_dealer(self) -> zmq.Socket:
        s = self.ctx.socket(zmq.DEALER)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, self.rpc_timeout_ms)
        s.connect(ep(self.ports["body_ctl"], self.host))
        return s

    def close(self) -> None:
        self._running = False
        if self._evt_thread:
            self._evt_thread.join(timeout=1.0)
        for s in (self._dealer, self._cam):
            if s is not None:
                s.close(0)
        if self._pose_sub is not None:
            self._pose_sub.stop()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def add_listener(self, fn) -> None:
        """fn(event_dict) for every body.event (all ops)."""
        self._listeners.append(fn)

    def _evt_loop(self) -> None:
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SUBSCRIBE, b"body.")
        s.connect(ep(self.ports["body_evt"], self.host))
        poller = zmq.Poller()
        poller.register(s, zmq.POLLIN)
        while self._running:
            if not poller.poll(100):
                continue
            try:
                topic, payload = s.recv_multipart(zmq.NOBLOCK)[:2]
            except (zmq.Again, ValueError):
                continue
            try:
                msg = json.loads(payload)
            except Exception:
                continue
            if topic == b"body.state":
                self.last_state = msg
                self.last_state_mono = time.monotonic()
                continue
            h = self._handles.get(msg.get("id"))
            if h is not None:
                h._on_event(msg)
            for fn in list(self._listeners):
                try:
                    fn(msg)
                except Exception:
                    pass
        s.close(0)

    # -- RPC ------------------------------------------------------------------------------------
    def request(self, op: str, args: dict | None = None, op_id: str | None = None) -> dict:
        req = {"id": op_id or uuid.uuid4().hex[:12], "op": op, "args": args or {}}
        with self._lock:
            if self._dealer is None:
                self._dealer = self._new_dealer()
            try:
                self._dealer.send(dumps_json(req))
                while True:
                    rep = json.loads(self._dealer.recv())
                    if rep.get("id") == req["id"]:
                        return rep
            except zmq.Again:
                self._dealer.close(0)
                self._dealer = None
                raise TimeoutError(f"body did not answer {op} within {self.rpc_timeout_ms} ms")

    def submit(self, op: str, args: dict | None = None) -> OpHandle:
        op_id = f"{op}-{uuid.uuid4().hex[:8]}"
        h = OpHandle(self, op_id, op, args or {})
        self._handles[op_id] = h  # register before sending: events can beat the reply
        rep = self.request(op, args, op_id=op_id)
        h.reply = rep
        if not rep.get("ok") and not h.done():
            # rejected / failed at start: the service also publishes a failed event; don't wait for it
            h._on_event({"id": op_id, "op": op, "state": "failed",
                         "data": {"reason": rep.get("error"), **(rep.get("data") or {})}, "via": "reply"})
        return h

    def run(self, op: str, args: dict | None = None, wait: bool = True, timeout: float | None = None) -> OpHandle:
        h = self.submit(op, args)
        if wait:
            h.wait(timeout)
        return h

    # -- motions --------------------------------------------------------------------------------
    def stand(self, release_band: bool = True, settle_s: float = 2.0, verify_s: float = 3.0, wait: bool = True,
              timeout: float | None = 90.0) -> OpHandle:
        return self.run("stand", {"release_band": release_band, "settle_s": settle_s, "verify_s": verify_s},
                        wait, timeout)

    def walk(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0, duration_s: float = 2.0,
             wait: bool = True, timeout: float | None = None) -> OpHandle:
        return self.run("walk", {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "duration_s": duration_s}, wait,
                        timeout if timeout is not None else duration_s + 30.0)

    def go_to(self, x: float, y: float, yaw: float | None = None, timeout_s: float | None = None,
              speed: float | None = None, wait: bool = True, **kw) -> OpHandle:
        args = {"x": x, "y": y, **kw}
        if yaw is not None:
            args["yaw"] = yaw
        if timeout_s is not None:
            args["timeout_s"] = timeout_s
        if speed is not None:
            args["speed"] = speed
        return self.run("go_to", args, wait, (timeout_s or 180.0) + 30.0)

    def turn_to(self, yaw: float, tol_deg: float | None = None, relative: bool = False, wait: bool = True,
                timeout: float | None = 60.0) -> OpHandle:
        args = {"yaw": yaw, "relative": relative}
        if tol_deg is not None:
            args["tol_deg"] = tol_deg
        return self.run("turn_to", args, wait, timeout)

    def stop(self, wait: bool = True, timeout: float = 10.0) -> OpHandle:
        return self.run("stop", {}, wait, timeout)

    # -- queries --------------------------------------------------------------------------------
    def status(self, op_id: str | None = None) -> dict:
        rep = self.request("status", {"id": op_id} if op_id else {})
        return rep.get("data") or {}

    def ping(self) -> dict:
        return self.request("ping")

    def pose(self, direct: bool = True, timeout: float = 2.0):
        """Latest ground-truth pose. direct=True subscribes to gt.pose (5601) itself; else uses body.state."""
        if not direct:
            return (self.last_state or {}).get("pose")
        if self._pose_sub is None:
            from .p1_client import PoseSub
            self._pose_sub = PoseSub(ep(self.ports["p1_pose"], self.host), ctx=self.ctx)
            self._pose_sub.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            p = self._pose_sub.latest()
            if p is not None:
                return p
            time.sleep(0.02)
        return None

    def camera_frame(self, timeout: float = 2.0, name: str = "ego_view"):
        """(timestamps, HxWx3 uint8 RGB) of the latest head-camera frame (P1 PUB 5565, sensor_server format)."""
        if self._cam is None:
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.CONFLATE, 1)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.connect(ep(self.ports["camera"], self.host))
            self._cam = s
        if not self._cam.poll(int(timeout * 1000)):
            return None, None
        ts, imgs = decode_camera_message(self._cam.recv())
        if not imgs:
            return ts, None
        return ts, imgs.get(name, next(iter(imgs.values())))
