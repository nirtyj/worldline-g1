"""The runtime's view of the body: `BodyPort` (what services call) and `SonicBody`, which wraps wl-body's
`body.client.BodyClient` (ROUTER 5610 / PUB 5611 / PUSH 5612, docs/contracts/m1.md §3.4-§3.13; imported, not copied).

SonicBody speaks the M2b body surface (contract v0.6) when the body publishes it (`body.state.fences`), and falls back
to the M1 surface otherwise (an M1 body, or the M1 test fake), labelled INTERIM in every receipt:

    halt(epoch)        B.1 halt lane: PUSH {op: halt, epoch} on 5612 and wait for `body.halted{epoch}` on 5611 within
                       the 30 ms budget (measured on the box: receipt p50 0.6 ms). `stopped` = the body latched it in
                       time. An unacked halt is re-sent every 100 ms by robot/health.py (send_halt). Never
                       command{stop}. (M1: a `stop` request, INTERIM.)
    one epoch space    the body fences by the executions' own `control_epoch` (and `generation`); SonicBody adds one
                       per-session constant to each (`epoch_base`, `gen_base`, from the `hello` reply: above every
                       epoch and generation the body has seen), so a restarted runtime is never stale. The runtime's
                       HaltGate has no counter of its own any more (services/common.py).
    fences             `fence(execution)` -> {execution_id, generation, control_epoch, session} (body numbers); every
                       gated op (go_to, turn_to, approach, scan, arm_script, arm) carries them.
    leases (B.2)       acquire(execution, mode) / release(execution_id) around each body execution: LOCOMOTION for
                       navigate, ARM_STREAM (groot_arms) / ARM_SCRIPT (sonic_arm_script, scan) for the arm.
    session (B.3)      `hello{session, watchdog_s 1.0}` + a 0.25 s heartbeat thread: a dead runtime makes the body
                       hold the robot (an internal halt, reason runtime_lost). `bye` on close.
    events (B.3)       body.mode / body.fault / body.halted / body.resumed / body.stale_command / body.lease /
                       body.session are pushed by the body and handed to G1Robot (attach_events) without polling.
    ops                approach (B.6: the reach_stance reposition), scan (B.5: the waist scan), arm_script (B.7), and
                       arm end / release for a script session.

Refusals (reply `ok: false`, or a `failed` terminal event) keep the body's reason; `refusal(reason, data)` says what
it means for the runtime: `halted`, `stale_result` (a fence: `stale_command` with data.why control_epoch |
generation | resume_epoch), `body_busy`, else None (an ordinary failure).

Every BodyOp resolves to {"state": "succeeded"|"failed"|"canceled", "data": {...}} (the body's terminal event).

GT confinement (PLAN §6.2): wl-body's `body.state` carries its own copy of P1's pose stream. SonicBody passes on only
the body's control fields (BODY_STATE_KEYS) and never judges health or rest from simulator truth: the real-time
factor is world/'s (WorldModel.sim_health, applied by robot/health.py), and a halt receipt's `at_rest` comes from
`speed_fn` (the factory passes WorldModel.planar_speed).
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import threading
import time
import uuid
from typing import Any, Callable

from api.types import ServiceHealth
from services.common import FENCE_WHY, STALE_REASONS, refusal  # noqa: F401  (re-exported)

TERMINAL = ("succeeded", "failed", "canceled")
HALT_MARGIN_S = 0.003          # of the 30 ms halt budget, kept for the bridge around SonicBody.halt
# The body.state fields the runtime may use (docs/contracts/m1.md §3.4): control, faults, the active op, the deploy
# and mux links, the arm channel. Pose and timing copies of P1's stream are not passed on (world/ owns them).
BODY_STATE_KEYS = ("in_control", "control_started", "fault", "active", "deploy", "mux", "nav", "tick", "mode",
                   "lease", "halt_epoch", "latched", "arm", "carry", "ops_extra", "recoveries")
LEASE_MODES = ("ANY", "LOCOMOTION", "ARM_STREAM", "ARM_SCRIPT", "MANIP")


class BodyOp:
    """An in-flight body motion. `result()` always resolves (terminal event, or a synthesized failure)."""

    def __init__(self, op: str, op_id: str, cancel_fn=None, loop: asyncio.AbstractEventLoop | None = None):
        self.op = op
        self.id = op_id
        self._cancel_fn = cancel_fn
        self._loop = loop or asyncio.get_event_loop()
        self._fut: asyncio.Future = self._loop.create_future()
        self.cancel_requested: str | None = None
        self.events: list[dict] = []
        self.reply: dict | None = None                  # the body's synchronous reply (plan data of a script)
        self.t0 = time.monotonic()
        self._progress: list[Callable[[dict], None]] = []

    # -- producer side (may be called from any thread) --
    def _set(self, state: str, data: dict | None) -> None:
        if not self._fut.done():
            self._fut.set_result({"state": state, "data": dict(data or {})})

    def finish(self, state: str, data: dict | None = None) -> None:
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            self._set(state, data)
        else:
            self._loop.call_soon_threadsafe(self._set, state, data)

    def _progress_event(self, ev: dict) -> None:
        for cb in list(self._progress):
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                pass

    # -- consumer side --
    def on_progress(self, cb: Callable[[dict], None]) -> None:
        """cb(event) for every `progress` event of this op (called from the body client's thread)."""
        self._progress.append(cb)

    @property
    def done(self) -> bool:
        return self._fut.done()

    async def result(self) -> dict:
        return await asyncio.shield(self._fut)

    def cancel(self, reason: str = "cancelled") -> None:
        """Request a stop (planner IDLE; an arm script ends where it is). Idempotent; the op then ends `canceled`."""
        if self.done or self.cancel_requested:
            return
        self.cancel_requested = reason
        if self._cancel_fn is not None:
            self._cancel_fn(reason)

    def __repr__(self) -> str:
        return f"<BodyOp {self.op} {self.id} {'done' if self.done else 'running'}>"


# BodyPort (what services call) is the api contract; re-exported here under its old name.
from api.services import BodyPort  # noqa: E402,F401


class SonicBody:
    """BodyPort over wl-body (P3). Executor name `sonic_walk`: every motion is SONIC's."""

    name = "sonic_walk"

    def __init__(self, client: Any = None, *, port_offset: int | None = None, host: str = "127.0.0.1",
                 halt_budget_ms: float = 30.0, sim_control: Any = None, connect_wait_s: float = 10.0,
                 speed_fn: Any = None, session: str | None = None, session_watchdog_s: float = 1.0,
                 heartbeat_s: float = 0.25, hello: bool = True):
        if client is None:
            from body.client import BodyClient
            client = BodyClient(port_offset=port_offset, host=host).connect(wait_s=connect_wait_s)
        self.client = client
        self.halt_budget_s = halt_budget_ms / 1000.0
        self.sim_control = sim_control                  # estop engages the band first (sim)
        self.speed_fn = speed_fn                        # () -> planar m/s | None (WorldModel.planar_speed)
        self._halt_box: dict[int, dict] = {}            # M1 path: epoch -> the stop reply (re-sends, late acks)
        self.halt_epoch = 0                             # runtime space: the last halt's control_epoch
        self.latched = False
        self.estopped = False
        self._ops: dict[str, BodyOp] = {}
        self._threads: list[threading.Thread] = []
        self._closed = False
        # one epoch space: body number = runtime number + base (set from the `hello` reply)
        self._flock = threading.Lock()
        self.epoch_base = 0
        self.gen_base = 0
        self._halt_wire: int | None = None              # the body's latched epoch after our last halt
        self._resume_wire: int | None = None            # the epoch of our last resume on the body
        self.session: str | None = None
        self.session_watchdog_s = float(session_watchdog_s)
        self.hello_info: dict = {}
        self._lease: dict | None = None                 # the lease this runtime holds {lease_id, owner, mode, ...}
        self.halts: collections.deque = collections.deque(maxlen=200)    # halt receipts (evidence)
        # body.* topics -> G1Robot (attach_events); buffered until someone listens
        self._sink: Callable[[str, dict], Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._evbuf: collections.deque = collections.deque(maxlen=500)
        self.topic_counts: collections.Counter = collections.Counter()
        self.body_halts: collections.deque = collections.deque(maxlen=50)     # body.halted payloads seen
        add = getattr(client, "add_topic_listener", None)
        if callable(add):
            add(self._on_topic)
        if hello and self.surface == "m2b":
            self._hello(session, session_watchdog_s, heartbeat_s)

    # ------------------------------------------------------------------ surface
    @property
    def surface(self) -> str:
        """'m2b' when the body publishes the v0.6 surface (halt lane, fences, leases, sessions), else 'm1'."""
        st = getattr(self.client, "last_state", None) or {}
        return "m2b" if isinstance(st.get("fences"), dict) else "m1"

    def supports(self, what: str) -> bool:
        """Body features: 'scan' / 'arm_script' (B.5 / B.7 arm ops), 'approach' (B.6), 'halt_lane', 'leases',
        'session' (B.1-B.3), 'arm' (the arm op)."""
        st = getattr(self.client, "last_state", None) or {}
        if what in ("scan", "arm_script"):
            return what in (st.get("ops_extra") or [])
        if what == "approach":
            return "approach_est" in st
        if what == "arm":
            return isinstance(st.get("arm"), dict)
        return self.surface == "m2b"

    # ------------------------------------------------------------------ one epoch space
    def wire_epoch(self, control_epoch: int) -> int:
        return self.epoch_base + int(control_epoch)

    def runtime_epoch(self, wire: int) -> int:
        return int(wire) - self.epoch_base

    def fence(self, execution: Any = None, *, execution_id: str | None = None, generation: int | None = None,
              control_epoch: int | None = None) -> dict:
        """The body's fence fields for an execution (or explicit fields), {} on an M1 body."""
        if execution is not None:
            execution_id = execution.execution_id
            generation = execution.generation
            control_epoch = execution.control_epoch
        if self.surface != "m2b":
            return {}
        out: dict[str, Any] = {}
        if execution_id:
            out["execution_id"] = str(execution_id)
        with self._flock:
            if generation is not None:
                out["generation"] = self.gen_base + int(generation)
            if control_epoch is not None:
                out["control_epoch"] = self.epoch_base + int(control_epoch)
        if self.session:
            out["session"] = self.session
        return out

    def _rebase_above(self, wire_floor: int, control_epoch: int) -> None:
        """Move the epoch map so that `control_epoch` lands above `wire_floor` on the body."""
        with self._flock:
            if self.epoch_base + int(control_epoch) <= int(wire_floor):
                self.epoch_base = int(wire_floor) - int(control_epoch) + 1

    def _body_fences(self) -> dict:
        return dict(((getattr(self.client, "last_state", None) or {}).get("fences")) or {})

    # ------------------------------------------------------------------ runtime session (B.3)
    def _hello(self, session: str | None, watchdog_s: float, heartbeat_s: float) -> None:
        sess = session or f"rt-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        try:
            rep = self.client.hello(sess, watchdog_s=watchdog_s, heartbeat_s=heartbeat_s)
        except Exception as e:  # noqa: BLE001
            rep = {"ok": False, "error": repr(e)}
        if not rep.get("ok"):
            self.hello_info = {"ok": False, "error": rep.get("error")}
            try:
                self.client.session = None              # stops the heartbeat thread
            except Exception:  # noqa: BLE001
                pass
            return
        d = dict(rep.get("data") or {})
        self.session = sess
        seen = [int(d[k]) for k in ("halt_epoch", "resume_epoch", "epoch_seen") if isinstance(d.get(k), int)]
        gf = d.get("generation_floor")
        with self._flock:
            self.epoch_base = (max(seen) + 1) if seen else 0
            self.gen_base = (int(gf) + 1) if isinstance(gf, int) else 0
        self.hello_info = {"ok": True, "session": sess, "watchdog_s": d.get("watchdog_s", watchdog_s),
                           "epoch_base": self.epoch_base, "gen_base": self.gen_base,
                           **{k: d.get(k) for k in ("mode", "latched", "halt_epoch", "resume_epoch", "epoch_seen",
                                                    "generation_floor", "lease")}}
        if d.get("latched") and isinstance(d.get("halt_epoch"), int):
            # a latch left by an earlier runtime (its stop, or its heartbeat lapsing): this session starts fresh
            rep = self._oneshot("resume", {"epoch": int(d["halt_epoch"])}, 2.0) or {}
            self._resume_wire = int(d["halt_epoch"])
            self.hello_info["startup_resume"] = {"epoch": int(d["halt_epoch"]), "ok": bool(rep.get("ok")),
                                                 "error": rep.get("error")}

    # ------------------------------------------------------------------ events (B.3)
    def attach_events(self, fn: Callable[[str, dict], Any], loop: asyncio.AbstractEventLoop | None = None) -> None:
        """fn(topic, msg) on `loop` for every body.* topic but body.event / body.state (G1Robot.events())."""
        self._loop = loop or asyncio.get_running_loop()
        self._sink = fn
        while self._evbuf:
            topic, msg = self._evbuf.popleft()
            self._loop.call_soon_threadsafe(self._deliver, topic, msg)

    def _on_topic(self, topic: str, msg: dict) -> None:
        """BodyClient's SUB thread: bookkeeping, then hand the topic to the runtime's loop."""
        self.topic_counts[topic] += 1
        if topic == "body.halted":
            self.body_halts.append(dict(msg))
            ep = msg.get("halt_epoch", msg.get("epoch"))
            if isinstance(ep, int) and msg.get("latched"):
                self._halt_wire = max(self._halt_wire if self._halt_wire is not None else ep, ep)
        elif topic == "body.lease" and msg.get("event") in ("revoked", "released"):
            lease = msg.get("lease") or {}
            if self._lease is not None and lease.get("lease_id") == self._lease.get("lease_id"):
                self._lease = None
        loop, sink = self._loop, self._sink
        if sink is None or loop is None or loop.is_closed():
            self._evbuf.append((topic, dict(msg)))
            return
        try:
            loop.call_soon_threadsafe(self._deliver, topic, dict(msg))
        except RuntimeError:
            self._evbuf.append((topic, dict(msg)))

    def _deliver(self, topic: str, msg: dict) -> None:
        if self._sink is not None:
            try:
                self._sink(topic, msg)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ helpers
    def _wrap(self, handle: Any, cancel_fn: Callable[[str], None] | None = None) -> BodyOp:
        loop = asyncio.get_running_loop()
        op = BodyOp(handle.op, handle.id, cancel_fn=cancel_fn or (lambda reason: self._fire_stop()), loop=loop)
        op.reply = getattr(handle, "reply", None)

        def on_event(h, ev):
            op.events.append(ev)
            st = ev.get("state")
            if st == "progress":
                op._progress_event(ev)
            elif st in TERMINAL:
                op.finish(st, ev.get("data") or {})

        handle.on_event(on_event)
        for ev in list(getattr(handle, "events", []) or []):      # events that beat the callback
            if ev.get("state") == "progress":
                op._progress_event(ev)
        if handle.done():                               # the reply or the event beat the callback
            op.finish(handle.state if handle.state in TERMINAL else "failed", handle.result or {})
        self._ops[handle.id] = op
        asyncio.ensure_future(self._watch(handle, op))
        return op

    async def _watch(self, handle: Any, op: BodyOp) -> None:
        """Safety net: an event can be missed (reconnect); poll the service like OpHandle.wait does."""
        while not op.done:
            await asyncio.sleep(1.0)
            if op.done or self._closed:
                return
            if handle.done():
                op.finish(handle.state if handle.state in TERMINAL else "failed", handle.result or {})
                return
            try:
                st = await asyncio.to_thread(self.client.status, handle.id)
                rec = (st or {}).get("op")
                if rec and rec.get("state") in TERMINAL:
                    op.finish(rec["state"], rec.get("result") or {})
            except Exception:  # noqa: BLE001
                pass

    def _oneshot(self, op: str, args: dict, timeout_s: float) -> dict | None:
        """One request on a private DEALER socket owned by the calling thread (the shared client socket is never
        touched from a helper thread, so close() cannot race it)."""
        import zmq
        from body.config import ep
        s = self.client.ctx.socket(zmq.DEALER)
        s.setsockopt(zmq.LINGER, 0)
        try:
            s.connect(ep(self.client.ports["body_ctl"], self.client.host))
            rid = f"{op}-{uuid.uuid4().hex[:8]}"
            s.send(json.dumps({"id": rid, "op": op, "args": args}).encode())
            t_end = time.monotonic() + timeout_s
            while True:
                left = t_end - time.monotonic()
                if left <= 0 or not s.poll(int(left * 1000) + 1):
                    return None
                rep = json.loads(s.recv())
                if rep.get("id") == rid:
                    return rep
        except Exception:  # noqa: BLE001
            return None
        finally:
            s.close(0)

    def _spawn(self, fn) -> threading.Thread:
        t = threading.Thread(target=fn, daemon=True)
        self._threads = [x for x in self._threads if x.is_alive()] + [t]
        t.start()
        return t

    def _fire_stop(self) -> None:
        self._spawn(lambda: self._oneshot("stop", {}, 2.0))

    def _gate(self) -> str | None:
        if self.estopped:
            return "estop"
        if self.latched:
            return "halted"
        return None

    def _failed(self, op: str, reason: str, **data: Any) -> BodyOp:
        o = BodyOp(op, f"{op}-{uuid.uuid4().hex[:8]}", loop=asyncio.get_running_loop())
        o.finish("failed", {"reason": reason, **data})
        return o

    async def _submit(self, op: str, args: dict, fence: dict | None = None,
                      cancel_fn: Callable[[Any], Callable[[str], None]] | None = None) -> BodyOp:
        why = self._gate()
        if why:
            return self._failed(op, why)
        h = await asyncio.to_thread(self.client.submit, op, {**args, **(fence or {})})
        return self._wrap(h, cancel_fn(h) if cancel_fn is not None else None)

    # ------------------------------------------------------------------ motions
    async def go_to(self, x: float, y: float, yaw: float | None = None, *, speed: float | None = None,
                    timeout_s: float | None = None, final_pos_tol: float | None = None,
                    final_yaw_tol_deg: float | None = None, fence: dict | None = None) -> BodyOp:
        args: dict[str, Any] = {"x": float(x), "y": float(y)}
        if yaw is not None:
            args["yaw"] = float(yaw)
        if speed is not None:
            args["speed"] = float(speed)
        if timeout_s is not None:
            args["timeout_s"] = float(timeout_s)
        if final_pos_tol is not None:
            args["final_pos_tol"] = float(final_pos_tol)
        if final_yaw_tol_deg is not None:
            args["final_yaw_tol_deg"] = float(final_yaw_tol_deg)
        return await self._submit("go_to", args, fence)

    async def turn_to(self, yaw: float, *, tol_deg: float | None = None, fence: dict | None = None) -> BodyOp:
        args: dict[str, Any] = {"yaw": float(yaw), "relative": False}
        if tol_deg is not None:
            args["tol_deg"] = float(tol_deg)
        return await self._submit("turn_to", args, fence)

    async def approach(self, x: float, y: float, yaw: float | None = None, *, v: float | None = None,
                       tol: tuple[float, float] | None = None, timeout_s: float | None = None,
                       fence: dict | None = None) -> BodyOp:
        """B.6 strafing reposition (0.1-0.4 m, contract §3.13): SLOW_WALK re-aimed on ground truth, learnt stops."""
        if not self.supports("approach"):
            return self._failed("approach", "not_supported", detail="the body has no approach op (B.6)")
        args: dict[str, Any] = {"x": float(x), "y": float(y)}
        if yaw is not None:
            args["yaw"] = float(yaw)
        if v is not None:
            args["v"] = float(v)
        if tol is not None:
            args["tol"] = [float(tol[0]), float(tol[1])]
        if timeout_s is not None:
            args["timeout_s"] = float(timeout_s)
        return await self._submit("approach", args, fence)

    async def scan(self, yaw_deg: tuple[float, ...] | list[float] = (-35.0, 0.0, 35.0), *,
                   move_s: float | None = None, hold_s: float | None = None, fence: dict | None = None,
                   **extra: Any) -> BodyOp:
        """B.5 waist scan (yaw only, contract §3.9 `scan`): `progress {kind: scan.hold, i, yaw, ...}` per hold."""
        if not self.supports("scan"):
            return self._failed("scan", "not_supported", detail="the body has no scan op (B.5)")
        args: dict[str, Any] = {"yaw_deg": [float(v) for v in yaw_deg], **extra}
        if move_s is not None:
            args["move_s"] = float(move_s)
        if hold_s is not None:
            args["hold_s"] = float(hold_s)
        return await self._submit("scan", args, fence,
                                  cancel_fn=lambda h: (lambda reason: self._fire_arm_end(h.id, None, fence, reason)))

    async def arm_script(self, phase: str, arm: str, *, target_w: Any = None, fence: dict | None = None,
                         **extra: Any) -> BodyOp:
        """B.7 one scripted arm phase (pregrasp | grasp | lift | carry | lower | release | retract; contract §3.9).
        A cancel ends the script where the arm is (`arm {end, hold_on_end: measured}`)."""
        if not self.supports("arm_script"):
            return self._failed("arm_script", "not_supported", detail="the body has no arm_script op (B.7)")
        args: dict[str, Any] = {"phase": str(phase), "arm": str(arm), **extra}
        if target_w is not None:
            args["target_w"] = [float(v) for v in target_w]
        return await self._submit("arm_script", args, fence,
                                  cancel_fn=lambda h: (lambda reason: self._fire_arm_end(h.id, "measured", fence,
                                                                                          reason)))

    def arm_end(self, stream: str, hold_on_end: str | None, fence: dict | None = None,
                reason: str | None = None) -> dict:
        """End the arm session of `stream` now (a script or scan: its op ends `succeeded`, ended_by client) into
        `hold_on_end` (None: the session's own). Blocking; the reply."""
        args: dict[str, Any] = {"stream": stream, "end": True, **(fence or {}), "t_wall": time.time()}
        if hold_on_end is not None:
            args["hold_on_end"] = hold_on_end
        if reason:
            args["reason"] = str(reason)[:120]
        return self._oneshot("arm", args, 2.0) or {"ok": False, "error": "no reply"}

    def _fire_arm_end(self, stream: str, hold_on_end: str | None, fence: dict | None, reason: str) -> None:
        self._spawn(lambda: self.arm_end(stream, hold_on_end, fence, reason))

    def arm_release(self, fence: dict | None = None, stream: str | None = None) -> dict:
        """Blend a hold (a finished session's `target` / `measured`, or the halt latch's pose) back to SONIC's own
        arms. The body releases a hold only for the stream that left it (or any stream for the latch)."""
        hold = self.arm_state().get("hold") or {}
        args = {"stream": stream or hold.get("stream") or f"rt-release-{uuid.uuid4().hex[:6]}", "release": True,
                **(fence or {}), "t_wall": time.time()}
        return self._oneshot("arm", args, 2.0) or {"ok": False, "error": "no reply"}

    def arm_state(self) -> dict:
        """body.state.arm (ArmChannel.snapshot): mode, hold, carry {engaged, ...}, latched, ops."""
        return dict(((getattr(self.client, "last_state", None) or {}).get("arm")) or {})

    async def stop(self) -> BodyOp:
        h = await asyncio.to_thread(self.client.submit, "stop", {})
        return self._wrap(h)

    # ------------------------------------------------------------------ leases (B.2)
    async def acquire(self, execution: Any, mode: str = "ANY") -> dict:
        """Lease the body for this execution: {ok, reason, lease, data}. An M1 body has no leases (ok, lease None).
        A lease this runtime still holds for an execution that ended (a lost release) is released first."""
        if self.surface != "m2b":
            return {"ok": True, "reason": None, "lease": None, "note": "M1 body: no leases"}
        f = self.fence(execution)
        mine = self._lease
        if mine is not None and mine.get("owner") != f.get("execution_id"):
            await self.release(str(mine.get("owner")))
        rep = await asyncio.to_thread(self._oneshot, "acquire", {**f, "mode": mode if mode in LEASE_MODES else "ANY"},
                                      2.0)
        rep = rep or {"ok": False, "error": "no_reply", "data": {}}
        data = dict(rep.get("data") or {})
        if rep.get("ok"):
            self._lease = dict(data.get("lease") or {})
            return {"ok": True, "reason": None, "lease": self._lease, "data": data}
        return {"ok": False, "reason": rep.get("error"), "lease": None, "data": data,
                "meaning": refusal(rep.get("error"), data)}

    async def release(self, execution_id: str) -> dict:
        if self.surface != "m2b":
            return {"ok": True, "released": None}
        rep = await asyncio.to_thread(self._oneshot, "release", {"execution_id": str(execution_id)}, 2.0) or {}
        if self._lease is not None and self._lease.get("owner") == execution_id and \
                (rep.get("ok") or rep.get("error") == "not_owner"):
            self._lease = None
        return {"ok": bool(rep.get("ok")), "released": (rep.get("data") or {}).get("released"),
                "error": rep.get("error")}

    @property
    def lease(self) -> dict | None:
        return dict(self._lease) if self._lease else None

    # ------------------------------------------------------------------ halt / resume / estop
    def halt(self, epoch: int | None = None) -> dict:
        """Latch HOLD within the 30 ms budget (PLAN §5.6): the B.1 halt lane (M1 body: a `stop`, INTERIM). `epoch`
        is the runtime control_epoch the halt fences. Never command{stop}."""
        t0 = time.monotonic()
        self.halt_epoch = int(epoch) if epoch is not None else self.halt_epoch + 1
        self.latched = True
        budget = self.halt_budget_s - (time.monotonic() - t0) - HALT_MARGIN_S
        if self.surface != "m2b":
            acked = self.send_halt(self.halt_epoch, budget)
            return {"accepted": True, "stopped": bool(acked), "at_rest": self._at_rest(), "mode": "HOLD",
                    "body_epoch": self.halt_epoch, "epoch": self.halt_epoch, "source": "sonic-mux",
                    "via": "body stop op (INTERIM: this body has no halt lane)"}
        r = self._lane_halt(self.halt_epoch, budget)
        body = r.get("body") or {}
        stopped = bool(r.get("acked")) and body.get("kind") != "stale" and bool(body.get("latched", True))
        rec = {"accepted": True, "stopped": stopped, "at_rest": self._at_rest(), "mode": "HOLD",
               "body_epoch": body.get("epoch", self.wire_epoch(self.halt_epoch)), "epoch": self.halt_epoch,
               "source": "halt-lane", "via": "B.1 halt lane (PUSH 5612 -> body.halted)", "rtt_ms": r.get("rtt_ms"),
               "body_kind": body.get("kind"), "handle_ms": body.get("handle_ms"),
               "arms_latched": body.get("arms_latched"), "body_active": body.get("active")}
        self.halts.append({"t_wall": time.time(), **rec})
        return rec

    def _lane_halt(self, epoch: int, wait_s: float) -> dict:
        wire = self.wire_epoch(epoch)
        try:
            r = self.client.halt(wire, timeout_s=max(0.0, wait_s), reason="runtime")
        except Exception as e:  # noqa: BLE001
            return {"acked": False, "epoch": wire, "error": repr(e)}
        body = r.get("body") or {}
        if r.get("acked") and body.get("kind") == "stale":
            # below the body's last resume (a tool or an implicit resume went above us): move above it, re-send
            f = self._body_fences()
            floor = max([v for v in (f.get("resume_epoch"), f.get("halt_epoch"), f.get("epoch_seen"),
                                     self._resume_wire, body.get("halt_epoch")) if isinstance(v, int)] or [wire])
            self._rebase_above(floor, epoch)
            wire = self.wire_epoch(epoch)
            try:
                r = self.client.halt(wire, timeout_s=max(0.005, wait_s), reason="runtime")
            except Exception as e:  # noqa: BLE001
                return {"acked": False, "epoch": wire, "error": repr(e)}
            r["rebased"] = True
            body = r.get("body") or {}
        if r.get("acked") and body.get("latched", True):
            ep = body.get("halt_epoch", body.get("epoch", wire))
            if isinstance(ep, int):
                self._halt_wire = ep
        return r

    def send_halt(self, epoch: int, wait_s: float) -> bool:
        """One halt attempt for `epoch` (runtime space); True once the body has acked it. robot/health.py
        HaltResender calls this every 100 ms until True."""
        if self.surface == "m2b":
            r = self._lane_halt(int(epoch), wait_s)
            return bool(r.get("acked")) and (r.get("body") or {}).get("kind") != "stale"
        box = self._halt_box.setdefault(int(epoch), {})
        if box.get("acked"):
            return True
        done = threading.Event()

        def run():
            rep = self._oneshot("stop", {}, 2.0)
            if (rep or {}).get("ok"):
                box["acked"] = True
            done.set()

        self._spawn(run)
        done.wait(max(0.0, wait_s))
        return bool(box.get("acked"))

    def halt_acked(self, epoch: int) -> bool:
        if self.surface == "m2b":
            return self._halt_wire is not None and self._halt_wire >= self.wire_epoch(epoch)
        return bool(self._halt_box.get(int(epoch), {}).get("acked"))

    def _at_rest(self) -> bool | None:
        try:
            v = self.speed_fn() if callable(self.speed_fn) else None
        except Exception:  # noqa: BLE001
            v = None
        return None if v is None else bool(v < 0.05)

    def resume(self, epoch: int | None = None, *, release_arms: bool = False) -> dict:
        """Clear the latch. `epoch` = the runtime control_epoch the next executions carry: if it would not be above
        the body's halt epoch, the epoch map moves up (one epoch space, a constant shift). The body gets resume{its
        halt epoch}, so the next halt is never taken for a late re-send. `release_arms`: also blend the halt
        latch's arm pose back to SONIC's own arms (nothing in the hands)."""
        self.latched = False
        if self.surface != "m2b":
            if epoch is not None:
                self.halt_epoch = max(self.halt_epoch, int(epoch))
            return {"ok": True, "via": "M1 body: runtime latch only"}
        f = self._body_fences()
        cands = [v for v in (self._halt_wire, f.get("halt_epoch")) if isinstance(v, int)]
        out: dict[str, Any] = {"ok": True}
        if cands:
            wire_h = max(cands)
            if epoch is not None:
                self._rebase_above(wire_h, int(epoch))
            rep = self._oneshot("resume", {"epoch": wire_h}, 2.0) or {"ok": False, "error": "no reply"}
            if not rep.get("ok") and rep.get("error") == "stale_command":
                h2 = (rep.get("data") or {}).get("halt_epoch")        # the body latched above us (its watchdog)
                if isinstance(h2, int):
                    wire_h = h2
                    if epoch is not None:
                        self._rebase_above(wire_h, int(epoch))
                    rep = self._oneshot("resume", {"epoch": wire_h}, 2.0) or {"ok": False, "error": "no reply"}
            self._resume_wire = wire_h
            out = {"ok": bool(rep.get("ok")), "body_epoch": wire_h, "error": rep.get("error"),
                   "was_latched": (rep.get("data") or {}).get("was_latched")}
        elif epoch is not None:
            self._rebase_above(self._resume_wire if self._resume_wire is not None else -1, int(epoch))
        if release_arms:
            arm = self.arm_state()
            if (arm.get("hold") or {}).get("kind") == "latched" or arm.get("latched"):
                ce = self.fence(control_epoch=int(epoch)) if epoch is not None else {}
                out["arms"] = self.arm_release(ce)
        return out

    def estop(self, reason: str) -> dict:
        """Operator kill button only (PLAN §5.6): band on (sim), then shutdown_control (command{stop}); the deploy
        exits and only a supervised P2 restart recovers."""
        self.estopped = True
        band = None
        if self.sim_control is not None:
            try:
                self.sim_control.band(True)
                band = True
            except Exception as e:  # noqa: BLE001
                band = f"failed: {e!r}"
        rep = self._oneshot("shutdown_control", {"confirm": True}, 5.0) or {"ok": False, "error": "no reply"}
        return {"accepted": bool(rep.get("ok")), "reason": reason, "band": band, "mode": "ESTOP",
                "deploy": "exits; P2 restart required"}

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        raw = dict(self.client.last_state or {})
        st = {k: raw[k] for k in BODY_STATE_KEYS if k in raw}
        age = time.monotonic() - float(getattr(self.client, "last_state_mono", 0.0) or 0.0)
        st["age_s"] = round(age, 3)
        st["body_mode"] = raw.get("mode")
        st["mode"] = _mode(st, self.latched, self.estopped, raw.get("mode") if self.surface == "m2b" else None)
        st["body_halt_epoch"] = raw.get("halt_epoch")
        st["body_latched"] = raw.get("latched")
        st["halt_epoch"] = self.halt_epoch
        st["latched"] = self.latched
        st["carry"] = ((raw.get("arm") or {}).get("carry")) or {"engaged": False}
        st["surface"] = self.surface
        st["session"] = self.session
        return st

    def health(self) -> ServiceHealth:
        """The body's own health (link, fault, deploy, control). The simulator's RTF is not the body's: world/
        judges it (WorldModel.sim_health) and robot/health.py applies it to the capabilities."""
        st = self.state()
        if self.estopped or st.get("body_mode") == "ESTOP":
            return ServiceHealth(False, "estop", "operator kill button; P2 restart required")
        if st.get("age_s", 99) > 1.5:
            return ServiceHealth(False, "down", f"no body.state for {st.get('age_s')} s")
        if st.get("fault"):
            return ServiceHealth(False, "fault", str(st.get("fault")))
        dep = st.get("deploy") or {}
        if dep and not dep.get("alive", True):
            return ServiceHealth(False, "down", "deploy not alive")
        if not st.get("in_control", True):
            return ServiceHealth(False, "down", "SONIC not in control")
        return ServiceHealth(True, "ok")

    def close(self) -> None:
        self._closed = True
        if self._lease is not None:
            _quiet(self._oneshot, "release", {"execution_id": str(self._lease.get("owner"))}, 1.0)
        if self.session is not None:
            _quiet(self.client.bye)
        for t in self._threads:
            t.join(timeout=2.5)
        _quiet(self.client.close)


def _quiet(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:  # noqa: BLE001
        return None


def _mode(st: dict, latched: bool, estopped: bool, body_mode: str | None = None) -> str:
    """The runtime's body mode: the body's own (B.3: HOLD / LOCOMOTION / ARM_STREAM / TRANSITION / FAULT / ESTOP /
    OFF) when it publishes one, else derived from the M1 state."""
    if estopped:
        return "ESTOP"
    if body_mode:
        return str(body_mode)
    if st.get("fault"):
        return "FAULT"
    act = st.get("active") or {}
    op = act.get("op") if isinstance(act, dict) else None
    if op in ("go_to", "walk", "velocity", "turn_to", "approach"):
        return "LOCOMOTION"
    if not st.get("in_control", True):
        return "OFF"
    return "HOLD"
