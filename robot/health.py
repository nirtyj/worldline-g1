"""Robot-side health: what may run now, who hears when that changes, and the halt re-send (PLAN §3.5, §5.2, §5.6).

    CapabilityPolicy   the capabilities the validator and G1Robot.start() check, from the services' own health plus
                       the simulator's real-time health (world/sim_health.py, read through WorldModel.sim_health; the
                       level there already has its dwell and hysteresis, so a 1-2 s dip after a walk rejects nothing):
                           DEGRADED (rtf_5s < 0.90 held 3 s; ok again at    manipulate rejected: "policy unavailable:
                                     >= 0.94 for 2 s)                        sim below real time ..." (R.6)
                           UNSAFE   (rtf_3s < 0.85 held 2 s, or < 0.70;     every body tool rejected: navigate,
                                     out at >= 0.90 for 2 s)                 manipulate and the scan (which turns the
                                                                             body); speech, glances,
                                                                             check_reachability and waits still run
    HealthMonitor      the capability_changed producer (PLAN §5.2): polls the capabilities and every loaded skill's
                       health every `period_s`, and on a change of (ok, state) emits
                           {"type": "capability_changed", "capability", "skill_id"?, "ok", "state", "detail", "was"}
                       on the robot's event sink. A sim level change is emitted once, after world's dwell. The harness
                       logs it; it wakes the planner (with the detail in a NOTE) only when the change matters: UNSAFE,
                       a running execution that uses it, or an open request / own goal / question (agent/harness.py).
                       The object_type enum never changes (Invariant 9); only CAPABILITY answers do.
                       It also forwards the simulator's events (WorldModel.drain_events, P1 gt.event): robot_fell ->
                       safety_event {kind: "fell", source: "sim"} (the harness stops), object_fell -> object_fell,
                       anything else -> sim_event.
    HaltResender       PLAN §5.6: a halt that the body did not ack within 30 ms is re-sent every 100 ms until the
                       body acks it (BodyPort.send_halt), the operator resumes, or a newer halt replaces it.
                       Emits `halt_acked {epoch, attempts, latency_ms}` or, after `max_s`,
                       `safety_event {kind: "halt_unacked"}` (the deploy's 1 s planner timeout is the last backstop).

Nothing here reads simulator truth itself: the RTF verdict is world/'s.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from api.types import ServiceHealth

BODY_TOOLS = ("navigate", "manipulate")          # plus observe(scan): it turns the body


@dataclass(frozen=True)
class Gate:
    """A CAPABILITY rejection: (code, message) for api.execution.Rejected("capability", ...)."""
    code: str
    message: str


def _sim(world: Any):
    fn = getattr(world, "sim_health", None)
    try:
        return fn() if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


class CapabilityPolicy:
    def __init__(self, world: Any, *, nav: Any, manip: Any, body: Any, observation_detail: Callable[[], str]):
        self.world = world
        self.nav = nav
        self.manip = manip
        self.body = body
        self.observation_detail = observation_detail

    def sim(self) -> ServiceHealth:
        h = _sim(self.world)
        if h is None:
            return ServiceHealth(True, "ok", "no sim health")
        return ServiceHealth(h.ok, h.state, h.detail)

    def capabilities(self) -> dict[str, ServiceHealth]:
        sim = _sim(self.world)
        nav = self.nav.health()
        manip = self.manip.health()
        body = self.body.health()
        if sim is not None and sim.unsafe:
            nav = ServiceHealth(False, "unsafe", sim.detail) if nav.ok else nav
            body = ServiceHealth(False, "unsafe", sim.detail) if body.ok else body
        if sim is not None and sim.degraded and manip.ok:
            manip = ServiceHealth(False, sim.state, sim.detail)
        return {"navigation": nav, "manipulation": manip,
                "observation": ServiceHealth(True, "ok", self.observation_detail()),
                "speech": ServiceHealth(True, "ok"), "body": body,
                "sim": self.sim()}

    def gate(self, tool: str, args: dict) -> Gate | None:
        """The sim-health rejection for this call at dispatch, or None (services check their own health)."""
        sim = _sim(self.world)
        if sim is None or sim.ok:
            return None
        scan = tool in ("observe", "look") and (args.get("mode") == "scan" or tool == "look")
        if sim.unsafe:
            if tool == "navigate":
                return Gate("nav_unhealthy", f"navigation stack unavailable ({sim.detail})")
            if tool == "manipulate":
                return Gate("policy_unavailable", f"policy unavailable: {sim.detail}")
            if scan:
                return Gate("controller_unavailable", f"no scan: the body holds still ({sim.detail})")
        if sim.degraded and tool == "manipulate":
            return Gate("policy_unavailable", f"policy unavailable: {sim.detail}")
        return None


class HealthMonitor:
    """The capability_changed producer. `start()` needs a running event loop (G1Robot.events() calls it)."""

    def __init__(self, capabilities: Callable[[], dict[str, ServiceHealth]], registry: Any, emit: Callable[..., Any],
                 *, period_s: float = 0.25, sim_events: Callable[[], list[dict]] | None = None):
        self.capabilities = capabilities
        self.registry = registry
        self.emit = emit
        self.period_s = period_s
        self.sim_events = sim_events
        self._last: dict[str, ServiceHealth] = {}
        self._task: asyncio.Task | None = None
        self.changes: list[dict] = []

    def snapshot(self) -> dict[str, ServiceHealth]:
        out: dict[str, ServiceHealth] = {}
        try:
            out.update(self.capabilities())
        except Exception as e:  # noqa: BLE001
            out["body"] = ServiceHealth(False, "down", f"health check failed: {e!r}")
        try:
            for s in self.registry.skills():
                out[f"skill:{s.skill_id}"] = self.registry.healthy(s.skill_id)
        except Exception:  # noqa: BLE001
            pass
        return out

    def forward_sim_events(self) -> int:
        try:
            evs = self.sim_events() if callable(self.sim_events) else []
        except Exception:  # noqa: BLE001
            evs = []
        for ev in evs:
            kind = str(ev.get("event") or "")
            fields = {k: v for k, v in ev.items() if k not in ("event", "type")}
            if kind in ("robot_fell", "fallen"):
                self.emit("safety_event", kind="fell", source="sim", sim_event=kind, **fields)
            elif kind == "object_fell":
                self.emit("object_fell", **fields)
            else:
                self.emit("sim_event", event=kind, **fields)
        return len(evs)

    def poll(self) -> list[dict]:
        """One check; returns (and emits) the changes. The first poll only records the baseline."""
        self.forward_sim_events()
        now = self.snapshot()
        first = not self._last
        out = []
        for key, h in now.items():
            old = self._last.get(key)
            if first or old is None or (old.ok, old.state) == (h.ok, h.state):
                continue
            cap, _, skill = key.partition(":")
            what = f"skill {skill}" if skill else cap
            detail = (f"{what} available again" + (f" ({h.detail})" if h.detail else "")) if h.ok else \
                f"{what} unavailable: {h.detail or h.state}"
            ev = {"capability": "manipulation" if skill else cap, "ok": h.ok, "state": h.state, "detail": detail,
                  "was": {"ok": old.ok, "state": old.state}}
            if skill:
                ev["skill_id"] = skill
            out.append(ev)
        self._last = now
        for ev in out:
            self.changes.append(ev)
            self.emit("capability_changed", **ev)
        return out

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self.poll()

        async def loop():
            while True:
                await asyncio.sleep(self.period_s)
                self.poll()
        self._task = asyncio.ensure_future(loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


class HaltResender:
    """Re-sends an unacknowledged halt every `period_s` (PLAN §5.6: 100 ms) on a thread, so it runs whether or not
    the caller has an event loop. `emit(type, **fields)` is marshalled onto the loop that started it, if any."""

    def __init__(self, body: Any, emit: Callable[..., Any], *, period_s: float = 0.1, attempt_wait_s: float = 0.03,
                 max_s: float = 10.0):
        self.body = body
        self.emit = emit
        self.period_s = period_s
        self.attempt_wait_s = attempt_wait_s
        self.max_s = max_s
        self._epoch: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log: list[dict] = []

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def supported(self) -> bool:
        return callable(getattr(self.body, "send_halt", None))

    def start(self, epoch: int) -> bool:
        """Begin re-sending the halt for `epoch` (a newer epoch replaces an older one). False if unsupported."""
        if not self.supported():
            return False
        self.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self._epoch = int(epoch)
        self._stop = threading.Event()
        stop = self._stop
        t = threading.Thread(target=self._run, args=(int(epoch), stop, loop), name=f"halt-resend-{epoch}",
                             daemon=True)
        self._thread = t
        t.start()
        return True

    def cancel(self) -> None:
        """resume() or a newer halt: stop re-sending (the thread exits within one period)."""
        self._stop.set()

    def _post(self, loop, type_: str, **fields) -> None:
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(lambda: self.emit(type_, **fields))
                return
            except RuntimeError:
                pass
        self.emit(type_, **fields)

    def _run(self, epoch: int, stop: threading.Event, loop) -> None:
        t0 = time.monotonic()
        attempts = 0
        while not stop.is_set():
            if stop.wait(self.period_s):
                break
            attempts += 1
            try:
                acked = bool(self.body.send_halt(epoch, self.attempt_wait_s))
            except Exception:  # noqa: BLE001
                acked = False
            if acked:
                rec = {"epoch": epoch, "attempts": attempts, "latency_ms": round((time.monotonic() - t0) * 1000, 1)}
                self.log.append({"event": "halt_acked", **rec})
                self._post(loop, "halt_acked", **rec)
                return
            if time.monotonic() - t0 >= self.max_s:
                rec = {"epoch": epoch, "attempts": attempts, "waited_s": round(time.monotonic() - t0, 2)}
                self.log.append({"event": "halt_unacked", **rec})
                self._post(loop, "safety_event", kind="halt_unacked", **rec)
                return


__all__ = ["BODY_TOOLS", "Gate", "CapabilityPolicy", "HealthMonitor", "HaltResender"]
