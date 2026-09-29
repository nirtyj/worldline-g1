"""BodyClient: the Python API for driving the G1 through wl-body (ROUTER 5610 / PUB 5611).

    from body.client import BodyClient
    with BodyClient() as bc:                       # port offset from WL_PORT_OFFSET (default 0)
        bc.stand()                                 # blocks until succeeded/failed; returns OpHandle
        bc.walk(vx=0.5, duration_s=5)
        h = bc.go_to(3.0, 1.0, yaw=1.57, wait=False)   # async handle
        h.wait(120); h.state, h.result
        bc.stop()
        bc.pose(); bc.status(); ts, img = bc.camera_frame()
        r = bc.halt(epoch=7)                       # halt lane (PUSH 5612) -> body.halted{7} within 30 ms: r["acked"]
        bc.resume(8)
        bc.approach(2.1, 1.3, yaw=0.0)             # short strafing reposition (0.1-0.4 m)

Every motion returns an OpHandle whose terminal state is one of succeeded | failed | canceled (rejections surface
as failed with data.reason). Nothing here fakes a result: the state is what wl-body reported.
Fences (docs/contracts/m1.md §3.11): pass execution_id / generation / control_epoch to any motion as keyword
arguments, or set them once with set_fence(); body topics other than body.event/body.state (body.halted, body.mode,
body.fault, body.stale_command, body.lease, body.session, body.resumed) go to add_topic_listener(fn(topic, msg)).
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
        self._halt_push: zmq.Socket | None = None
        self._halt_lock = threading.Lock()
        self._halted: dict[int, tuple[dict, float]] = {}       # epoch -> (body.halted payload, perf_counter at recv)
        self._halt_cv = threading.Condition()
        self._topic_listeners = []
        self.fence: dict = {}                                   # default fence fields for every motion (set_fence)
        self.session: str | None = None
        self._hb_thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------------
    def connect(self, wait_s: float = 10.0) -> "BodyClient":
        self._running = True
        self._evt_ready = threading.Event()
        self._evt_thread = threading.Thread(target=self._evt_loop, name="body-evt", daemon=True)
        self._evt_thread.start()
        self._dealer = self._new_dealer()
        # the halt lane PUSH is connected now so a halt never pays a connect: ZMQ queues on the pipe at once
        self._halt_push = self.ctx.socket(zmq.PUSH)
        self._halt_push.setsockopt(zmq.LINGER, 0)
        self._halt_push.setsockopt(zmq.SNDHWM, 100)
        self._halt_push.connect(ep(self.ports["body_halt"], self.host))
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
        if self._hb_thread:
            self._hb_thread.join(timeout=1.0)
        for s in (self._dealer, self._cam, self._halt_push):
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

    def add_topic_listener(self, fn) -> None:
        """fn(topic: str, msg: dict) for every body.* topic other than body.event and body.state."""
        self._topic_listeners.append(fn)

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
            if topic != b"body.event":
                t_recv = time.perf_counter()
                if topic == b"body.halted":
                    try:
                        ep_ = int(msg.get("requested_epoch", msg.get("epoch")))
                    except (TypeError, ValueError):
                        ep_ = None
                    if ep_ is not None:
                        with self._halt_cv:
                            self._halted.setdefault(ep_, (msg, t_recv))
                            if len(self._halted) > 200:
                                self._halted.pop(next(iter(self._halted)))
                            self._halt_cv.notify_all()
                for fn in list(self._topic_listeners):
                    try:
                        fn(topic.decode(), msg)
                    except Exception:
                        pass
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
        args = {**self.fence, **(args or {})}
        h = OpHandle(self, op_id, op, args)
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
              timeout: float | None = 90.0, **kw) -> OpHandle:
        return self.run("stand", {"release_band": release_band, "settle_s": settle_s, "verify_s": verify_s, **kw},
                        wait, timeout)

    def walk(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0, duration_s: float = 2.0,
             wait: bool = True, timeout: float | None = None, **kw) -> OpHandle:
        return self.run("walk", {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "duration_s": duration_s, **kw}, wait,
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
                timeout: float | None = 60.0, **kw) -> OpHandle:
        args = {"yaw": yaw, "relative": relative, **kw}
        if tol_deg is not None:
            args["tol_deg"] = tol_deg
        return self.run("turn_to", args, wait, timeout)

    def stop(self, wait: bool = True, timeout: float = 10.0, arms: bool = False) -> OpHandle:
        """Planner IDLE (legs). arms=True also ends an active arm stream (blend back to SONIC's own arms)."""
        return self.run("stop", {"arms": True} if arms else {}, wait, timeout)

    def approach(self, x: float, y: float, yaw: float | None = None, v: float | None = None,
                 tol: tuple[float, float] | None = None, wait: bool = True, timeout: float | None = 60.0,
                 **kw) -> OpHandle:
        """Short strafing reposition (0.1-0.4 m; contract §3.13). tol = (pos_m, yaw_deg), default (0.05, 5)."""
        args = {"x": x, "y": y, **kw}
        if yaw is not None:
            args["yaw"] = yaw
        if v is not None:
            args["v"] = v
        if tol is not None:
            args["tol"] = [float(tol[0]), float(tol[1])]
        return self.run("approach", args, wait, timeout)

    def recover(self, wait: bool = True, timeout: float | None = 90.0, **kw) -> OpHandle:
        """Soft recovery after a fall in sim (PLAN §7.3.3 path A; contract §3.12)."""
        return self.run("recover", kw, wait, timeout)

    # -- halt lane, fences, leases, runtime sessions (contract §3.10-§3.12) ------------------------
    def halt(self, epoch: int, timeout_s: float = 0.03, reason: str | None = None) -> dict:
        """Fire-and-forget PUSH {op: halt, epoch, t_wall} on the halt lane (5612), then wait up to timeout_s for
        body.halted{epoch} on 5611. Returns {acked, epoch, rtt_ms (send -> body.halted received by our SUB thread),
        wait_ms (send -> return), body: <body.halted payload or None>}. Never command{stop}."""
        epoch = int(epoch)
        msg = {"op": "halt", "epoch": epoch, "t_wall": time.time()}
        if reason:
            msg["reason"] = reason
        with self._halt_cv:
            earlier = self._halted.pop(epoch, None)       # an ack of an earlier attempt of this epoch
        t0 = time.perf_counter()
        with self._halt_lock:
            if self._halt_push is None:
                raise RuntimeError("BodyClient.halt before connect()")
            try:
                self._halt_push.send(dumps_json(msg), zmq.NOBLOCK)
            except zmq.Again:                    # the lane queue is full: the body has been gone for a while
                return {"acked": earlier is not None, "epoch": epoch, "rtt_ms": None, "wait_ms": 0.0,
                        "body": None if earlier is None else earlier[0], "error": "lane_queue_full"}
        got = self.wait_halted(epoch, timeout_s)
        t1 = time.perf_counter()
        body, t_recv = got if got is not None else (None, None)
        out = {"acked": body is not None or earlier is not None, "epoch": epoch,
               "rtt_ms": None if t_recv is None else round((t_recv - t0) * 1e3, 3),
               "wait_ms": round((t1 - t0) * 1e3, 3), "body": body}
        if body is None and earlier is not None:
            out["body"], out["from_earlier_attempt"] = earlier[0], True
        return out

    def wait_halted(self, epoch: int, timeout_s: float):
        """(body.halted payload, perf_counter at receipt) for `epoch` (received since the last halt() of that epoch),
        or None after timeout_s. The body answers a re-send of a latched epoch with `kind: repeat`."""
        t_end = time.perf_counter() + max(0.0, timeout_s)
        with self._halt_cv:
            while True:
                got = self._halted.get(int(epoch))
                if got is not None:
                    return got
                left = t_end - time.perf_counter()
                if left <= 0:
                    return None
                self._halt_cv.wait(left)

    def resume(self, epoch: int) -> dict:
        """Clear the halt latch (ROUTER op; reply {ok, data: body.resumed payload} or error stale_command)."""
        return self.request("resume", {"epoch": int(epoch)})

    def set_fence(self, execution_id: str | None = None, generation: int | None = None,
                  control_epoch: int | None = None) -> None:
        """Default fence fields merged into every motion submitted from now on (None clears a field)."""
        for k, v in (("execution_id", execution_id), ("generation", generation), ("control_epoch", control_epoch)):
            if v is None:
                self.fence.pop(k, None)
            else:
                self.fence[k] = v

    def acquire(self, execution_id: str, generation: int, control_epoch: int, mode: str = "ANY") -> dict:
        return self.request("acquire", {"execution_id": execution_id, "generation": int(generation),
                                        "control_epoch": int(control_epoch), "mode": mode,
                                        **({"session": self.session} if self.session else {})})

    def release(self, execution_id: str) -> dict:
        return self.request("release", {"execution_id": execution_id})

    def hello(self, session: str | None = None, watchdog_s: float = 1.0, heartbeat_s: float | None = 0.25) -> dict:
        """Register a runtime session: the body holds the robot if it hears no ping for watchdog_s while moving.
        heartbeat_s starts a background ping thread (None: the caller pings with ping_session())."""
        self.session = session or f"rt-{uuid.uuid4().hex[:8]}"
        rep = self.request("hello", {"session": self.session, "watchdog_s": watchdog_s})
        if heartbeat_s and self._hb_thread is None:
            self._hb_thread = threading.Thread(target=self._heartbeat, args=(heartbeat_s,), name="body-hb",
                                               daemon=True)
            self._hb_thread.start()
        return rep

    def ping_session(self) -> dict:
        return self.request("ping", {"session": self.session})

    def _heartbeat(self, period: float) -> None:
        s = self.ctx.socket(zmq.DEALER)            # own socket: the shared DEALER is not touched from this thread
        s.setsockopt(zmq.LINGER, 0)
        s.connect(ep(self.ports["body_ctl"], self.host))
        try:
            while self._running and self.session is not None:
                s.send(dumps_json({"id": f"hb-{uuid.uuid4().hex[:6]}", "op": "ping",
                                   "args": {"session": self.session}}))
                t_end = time.monotonic() + period
                while self._running and time.monotonic() < t_end:
                    if s.poll(20):
                        s.recv()
        finally:
            s.close(0)

    def bye(self) -> dict | None:
        sess, self.session = self.session, None
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=1.0)
            self._hb_thread = None
        return None if sess is None else self.request("bye", {"session": sess})

    def arm_stream(self, stream: str | None = None, **defaults) -> "ArmStream":
        """Stream arm/hand joint targets into SONIC (op `arm`, body/arm.py). defaults (watchdog_s, hold_s,
        blend_s, max_vel, waist, vel, preempt) are sent with the first message."""
        return ArmStream(self, stream or f"arm-{uuid.uuid4().hex[:8]}", defaults)

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


class ArmStream:
    """Client side of the `arm` op. Send targets at up to 50 Hz; stop sending (watchdog 0.3 s -> hold -> blend) or
    call end() to give the arms back to SONIC.

        with BodyClient() as bc:
            arm = bc.arm_stream()
            arm.send(upper_body={"right_elbow_joint": 1.2}, right_hand=0.0)   # dict by joint name, or a 17-list
            ...                                                                 # (joint_map.UPPER_BODY_MUJOCO_JOINTS)
            arm.end(); arm.handle.wait(5)                                       # succeeded once blended back

    `handle` is the op's OpHandle (accepted -> progress -> succeeded | canceled | failed). Updates only get a reply.
    """

    def __init__(self, client: BodyClient, stream: str, defaults: dict):
        self.client = client
        self.stream = stream
        self.defaults = defaults
        self.handle: OpHandle | None = None
        self.last_reply: dict | None = None
        self.sent = 0
        self.rejected = 0

    def send(self, upper_body=None, left_hand=None, right_hand=None, upper_body_vel=None, **kw) -> dict:
        args = {"stream": self.stream, "t_wall": time.time(), **kw}
        for k, v in (("upper_body", upper_body), ("left_hand", left_hand), ("right_hand", right_hand),
                     ("upper_body_vel", upper_body_vel)):
            if v is not None:
                args[k] = _plain(v)
        if self.handle is None or self.handle.done():
            args = {**self.defaults, **args}
            op_id = f"arm-{uuid.uuid4().hex[:8]}"
            h = OpHandle(self.client, op_id, "arm", {k: v for k, v in args.items() if not isinstance(v, (list, dict))})
            self.client._handles[op_id] = h
            rep = self.client.request("arm", args, op_id=op_id)
            h.reply = rep
            if rep.get("ok"):
                self.handle = h
            elif not h.done():
                h._on_event({"id": op_id, "op": "arm", "state": "failed",
                             "data": {"reason": rep.get("error"), **(rep.get("data") or {})}, "via": "reply"})
        else:
            rep = self.client.request("arm", args)
        self.sent += 1
        if not rep.get("ok"):
            self.rejected += 1
        self.last_reply = rep
        return rep

    def end(self) -> dict:
        return self.client.request("arm", {"stream": self.stream, "end": True})


def _plain(v):
    if isinstance(v, dict):
        return {str(k): float(x) for k, x in v.items()}
    if isinstance(v, (int, float)):
        return float(v)
    return [float(x) for x in v]
