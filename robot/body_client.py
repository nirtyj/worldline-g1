"""The runtime's view of the body: `BodyPort` (what services call) and `SonicBody`, which wraps wl-body's
`body.client.BodyClient` (ROUTER 5610 / PUB 5611, docs/contracts/m1.md §3.4-3.5; imported, not copied).

M1's body offers stand / walk / go_to / turn_to / stop / status / velocity. PLAN §6.6's full BodyClient (leases,
arm_script, waist scan, vla_start/stop, the halt lane on PUSH 5612 with a `body.halted{epoch}` ack, carry lock)
is M2b/M3 work on the body side; until then:

    halt()     = a `stop` request (planner IDLE, never command{stop}) with a 30 ms reply budget. `stopped` is True
                 only when the body acknowledged within the budget (PLAN §5.6: "stopped" means latched); an
                 unacknowledged halt is re-sent every 100 ms by robot/health.py until the body acks it.
    leases     = none: one motion at a time is the body's own rule (a new motion pre-empts the active one); the
                 runtime's validator (C7) keeps one `body` execution at a time.
    reposition = go_to with a tighter final tolerance (labelled interim in results).

Every BodyOp resolves to {"state": "succeeded"|"failed"|"canceled", "data": {...}} (the body's terminal event).

GT confinement (PLAN §6.2): wl-body's `body.state` carries its own copy of P1's pose stream. SonicBody passes on only
the body's control fields (BODY_STATE_KEYS) and never judges health or rest from simulator truth: the real-time
factor is world/'s (WorldModel.sim_health, applied by robot/health.py), and a halt receipt's `at_rest` comes from
`speed_fn` (the factory passes WorldModel.planar_speed).
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

from api.types import ServiceHealth

TERMINAL = ("succeeded", "failed", "canceled")
HALT_MARGIN_S = 0.003          # of the 30 ms halt budget, kept for the bridge around SonicBody.halt
# The body.state fields the runtime may use (docs/contracts/m1.md §3.4): control, faults, the active op, the deploy
# and mux links. Pose and timing copies of P1's stream are not passed on (world/ owns them).
BODY_STATE_KEYS = ("in_control", "control_started", "fault", "active", "deploy", "mux", "nav", "tick", "mode",
                   "lease", "halt_epoch", "latched", "arm", "carry")


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
        self.t0 = time.monotonic()

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

    # -- consumer side --
    @property
    def done(self) -> bool:
        return self._fut.done()

    async def result(self) -> dict:
        return await asyncio.shield(self._fut)

    def cancel(self, reason: str = "cancelled") -> None:
        """Request a stop (planner IDLE). Idempotent; the op then ends `canceled`."""
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
                 speed_fn: Any = None):
        if client is None:
            from body.client import BodyClient
            client = BodyClient(port_offset=port_offset, host=host).connect(wait_s=connect_wait_s)
        self.client = client
        self.halt_budget_s = halt_budget_ms / 1000.0
        self.sim_control = sim_control                  # estop engages the band first (sim)
        self.speed_fn = speed_fn                        # () -> planar m/s | None (WorldModel.planar_speed)
        self._halt_box: dict[int, dict] = {}            # epoch -> the stop reply (for re-sends and late acks)
        self.halt_epoch = 0
        self.latched = False
        self.estopped = False
        self._ops: dict[str, BodyOp] = {}
        self._threads: list[threading.Thread] = []
        self._closed = False

    # ------------------------------------------------------------------ helpers
    def _wrap(self, handle: Any) -> BodyOp:
        loop = asyncio.get_running_loop()
        op = BodyOp(handle.op, handle.id, cancel_fn=lambda reason: self._fire_stop(), loop=loop)

        def on_event(h, ev):
            op.events.append(ev)
            st = ev.get("state")
            if st in TERMINAL:
                op.finish(st, ev.get("data") or {})

        handle.on_event(on_event)
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
        import json

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

    async def _submit(self, op: str, args: dict) -> BodyOp:
        why = self._gate()
        if why:
            loop = asyncio.get_running_loop()
            o = BodyOp(op, f"{op}-{uuid.uuid4().hex[:8]}", loop=loop)
            o.finish("failed", {"reason": why})
            return o
        h = await asyncio.to_thread(self.client.submit, op, args)
        return self._wrap(h)

    # ------------------------------------------------------------------ motions
    async def go_to(self, x: float, y: float, yaw: float | None = None, *, speed: float | None = None,
                    timeout_s: float | None = None, final_pos_tol: float | None = None,
                    final_yaw_tol_deg: float | None = None) -> BodyOp:
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
        return await self._submit("go_to", args)

    async def turn_to(self, yaw: float, *, tol_deg: float | None = None) -> BodyOp:
        args: dict[str, Any] = {"yaw": float(yaw), "relative": False}
        if tol_deg is not None:
            args["tol_deg"] = float(tol_deg)
        return await self._submit("turn_to", args)

    async def stop(self) -> BodyOp:
        h = await asyncio.to_thread(self.client.submit, "stop", {})
        return self._wrap(h)

    # ------------------------------------------------------------------ halt / resume / estop
    def halt(self, epoch: int | None = None) -> dict:
        """Latch HOLD: a body `stop` (planner IDLE) within a 30 ms reply budget. Never command{stop}."""
        t0 = time.monotonic()
        self.halt_epoch = int(epoch) if epoch is not None else self.halt_epoch + 1
        self.latched = True
        # the whole call, the bridge's bookkeeping included, stays inside the 30 ms budget (PLAN §5.6)
        acked = self.send_halt(self.halt_epoch, self.halt_budget_s - (time.monotonic() - t0) - HALT_MARGIN_S)
        return {"accepted": True, "stopped": bool(acked), "at_rest": self._at_rest(), "mode": "HOLD",
                "body_epoch": self.halt_epoch, "source": "sonic-mux",
                "via": "body stop op (M1 body has no halt lane)"}

    def send_halt(self, epoch: int, wait_s: float) -> bool:
        """One halt attempt for `epoch` (a body `stop`); True once the body has acked this epoch's halt, including
        a late reply to an earlier attempt. robot/health.py HaltResender calls this every 100 ms until True."""
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
        return bool(self._halt_box.get(int(epoch), {}).get("acked"))

    def _at_rest(self) -> bool | None:
        try:
            v = self.speed_fn() if callable(self.speed_fn) else None
        except Exception:  # noqa: BLE001
            v = None
        return None if v is None else bool(v < 0.05)

    def resume(self, epoch: int | None = None) -> None:
        self.latched = False
        if epoch is not None:
            self.halt_epoch = max(self.halt_epoch, int(epoch))

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
        st["mode"] = _mode(st, self.latched, self.estopped)
        st["halt_epoch"] = self.halt_epoch
        st["latched"] = self.latched
        return st

    def health(self) -> ServiceHealth:
        """The body's own health (link, fault, deploy, control). The simulator's RTF is not the body's: world/
        judges it (WorldModel.sim_health) and robot/health.py applies it to the capabilities."""
        st = self.state()
        if self.estopped:
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
        for t in self._threads:
            t.join(timeout=2.5)
        _quiet(self.client.close)


def _quiet(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:  # noqa: BLE001
        return None


def _mode(st: dict, latched: bool, estopped: bool) -> str:
    if estopped:
        return "ESTOP"
    if st.get("fault"):
        return "FAULT"
    act = st.get("active") or {}
    op = act.get("op") if isinstance(act, dict) else None
    if op in ("go_to", "walk", "velocity", "turn_to"):
        return "LOCOMOTION"
    if not st.get("in_control", True):
        return "OFF"
    return "HOLD"
