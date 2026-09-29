"""The halt lane (M2b B.1; docs/contracts/m1.md §3.10; PLAN §5.6, §6.6): PULL 5612, handled on its own thread.

    PUSH -> PULL 5612, one JSON frame:  {"op": "halt", "epoch": int, "t_wall": sender time.time(), "reason"?: str}
                                        {"op": "resume", "epoch": int}
    PUB 5611 [b"body.halted", JSON]:    {epoch, t_mono, arms_latched, handle_ms, ...}   (BodyService.halt)
             [b"body.resumed", JSON]:   {epoch, was_latched, ...}                       (BodyService.resume)

Why a thread of its own: the control loop can be busy for tens of milliseconds (an A* plan at a go_to start, a P1
call), and a halt must never wait for it. This thread only does thread-safe things: SonicMux.latch (IDLE on the wire
at once), ArmChannel.latch under the service's arm lock, the fence latch, and the body.halted publish under the
event-socket lock. Cancelling the active leg motion (its `canceled` terminal event) is left to the control loop,
which runs it before it handles the next request or tick (<= 20 ms); the mux latch has already made that motion's
commands ineffective. The socket is bound by BodyService.setup() on the control thread and used only here afterwards.
"""

from __future__ import annotations

import json
import threading
import time

import zmq


class HaltLane(threading.Thread):
    def __init__(self, svc, sock: zmq.Socket):
        super().__init__(name="halt-lane", daemon=True)
        self.svc = svc
        self.sock = sock
        self._running = True
        self.stats = {"received": 0, "halts": 0, "resumes": 0, "bad": 0}

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while self._running:
            if not poller.poll(100):
                continue
            while True:
                try:
                    raw = self.sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                t_recv = time.monotonic()
                self.stats["received"] += 1
                try:
                    msg = json.loads(raw.decode("utf-8"))
                    op = str(msg.get("op", "halt"))
                    epoch = msg["epoch"]
                except Exception as e:  # malformed: count it, never guess an epoch
                    self.stats["bad"] += 1
                    self.svc.log(f"[halt-lane] bad message {raw[:120]!r}: {e!r}")
                    continue
                try:
                    if op == "halt":
                        self.stats["halts"] += 1
                        self.svc.halt(epoch, reason=str(msg.get("reason") or "halt"), source="lane", t_recv=t_recv,
                                      t_wall_sent=msg.get("t_wall"))
                    elif op == "resume":
                        self.stats["resumes"] += 1
                        self.svc.resume_from_lane(epoch, t_recv=t_recv)
                    else:
                        self.stats["bad"] += 1
                        self.svc.log(f"[halt-lane] unknown op {op!r}")
                except Exception as e:
                    self.svc.log(f"[halt-lane] {op} failed: {e!r}")

    def stop(self) -> None:
        self._running = False
