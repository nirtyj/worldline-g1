"""groot_arms against the body's own chunk-mode ArmChannel (docs/contracts/arm_chunk.md; body B.8), in process.

The runtime side of the chunk wire was built against tests/fakes/fake_arm_body.py (the contract's reference). The
body owner implemented B.8 in body/arm.py on the body branch (master, "body-arm: arm channel wave 1"). This test runs
the REAL GrootArmExecutor (real groot/ helpers, the fake PolicyServer on a real ZMQ socket) against the REAL
ArmChannel ticking at 50 Hz in wall time, with the body's own test plant (body/tests/arm_sim.py: a SONIC-like arm
response to the override) standing in for SONIC and g1_debug. It skips until body/arm.py with chunk mode is on this
branch (the wave-2 merge); the integrator ran it on a trial merge (docs/M2b_wave1.md §3).

Pinned: the start / chunk / keepalive / end messages are accepted as sent (no bad_args, bad_chunk, mode_mismatch or
stale_command); chunks are applied (not expired) with the executor's lead; the body's arm.progress reaches the
executor (clamped_frac from the body); the session ends with exactly one terminal event and the hold the executor
asked for; a body halt latch ends the session failed(halted) with no chunk applied after it; cancel ends it at once.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("groot.actions")
arm_mod = pytest.importorskip("body.arm")
if "chunk" not in getattr(arm_mod, "MODES", ()):
    pytest.skip("body/arm.py has no chunk mode (B.8) on this branch", allow_module_level=True)
arm_sim = pytest.importorskip("body.tests.arm_sim")

from services.common import HaltGate  # noqa: E402
from services.executors.groot_arms import GrootArmExecutor, _groot_helpers  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from tests.services.test_groot_arms import FakeWorld, Sink, fast_cfg, make_job  # noqa: E402


class RealtimeBody:
    """ArmPort + SensorPort over the body's ArmChannel: handle() and a 50 Hz tick serialised on one lock (the
    BodyService runs both on its service thread), events delivered like BodyArmPort's body.event listener."""

    def __init__(self):
        self.clock = time.monotonic
        self.mux = arm_sim.Mux()
        self.mux.clock = self.clock
        self.dep = arm_sim.Deploy(self.clock)
        self.plant = arm_sim.Plant(self.mux, self.dep)
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.replies: list[tuple[float, dict, dict]] = []
        self._subs: list = []
        self.ch = arm_mod.ArmChannel(SimpleNamespace(), self.mux, self.dep, self._emit, log=lambda *_: None,
                                     record=lambda *_: None, pose=lambda: None, clock=self.clock)
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="arm-tick")
        self._t.start()

    def _emit(self, op_id, state, data):
        ev = {"op": "arm", "id": op_id, "state": state, "data": data, "t": self.clock()}
        self.events.append(ev)
        for cb in list(self._subs):
            cb(ev)

    def _loop(self):
        nxt = self.clock()
        while self._run:
            nxt += 0.02
            with self.lock:
                now = self.clock()
                self.ch.tick(now)
                self.plant.step(now)
            time.sleep(max(0.0, nxt - self.clock()))

    # ArmPort
    def arm(self, args: dict, op_id: str | None = None) -> dict:
        with self.lock:
            try:
                rep = self.ch.handle(op_id or "arm-x", dict(args), (True, None, {}))
            except arm_mod.ArmError as e:
                rep = {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
            self.replies.append((self.clock(), dict(args), rep))
            return rep

    def subscribe(self, cb):
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def supports_chunk(self) -> bool:
        return "chunk" in arm_mod.MODES

    # SensorPort (GR00T's inputs: a grey ego frame, g1_debug from the plant)
    def ego_frame(self):
        return np.full((480, 640, 3), 90, np.uint8), self.clock()

    def debug_state(self):
        with self.lock:
            d = {k: list(v) for k, v in self.dep.latest.items()}
        return d, self.clock()

    def latch(self, epoch: int):
        with self.lock:
            return self.ch.latch(epoch, "halt")

    def close(self):
        self._run = False
        self._t.join(timeout=1.0)

    def rejected(self) -> list[str]:
        return [r.get("error") for _, _, r in self.replies if not r.get("ok")]

    def terminal(self, sid: str) -> list[dict]:
        return [e for e in self.events if e["state"] in ("succeeded", "failed", "canceled")
                and (e["data"] or {}).get("session_id") == sid]


class Rig:
    def __init__(self, server: dict | None = None, cfg: dict | None = None, gate: HaltGate | None = None):
        self.srv = FakePolicyServer(**(server or {"close_after": None})).start()
        self.body = RealtimeBody()
        self.world = FakeWorld("never")
        self.sink = Sink()
        self.gate = gate
        c = {"max_duration_s": 2.5, "stall_fail_s": 3.0, "hands_open_s": 0.1, "lead_s": 0.15, **(cfg or {})}
        self.exe = GrootArmExecutor(self.world, arm=self.body, sensors=self.body, cfg=fast_cfg(self.srv, **c),
                                    gate=gate, events=self.sink, helpers=_groot_helpers())

    def __enter__(self):
        t_end = time.monotonic() + 5.0
        while not self.exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        assert self.exe.health().ok, self.exe.health().detail
        return self

    def __exit__(self, *exc):
        self.exe.close()
        self.body.close()
        self.srv.stop()


def test_a_session_runs_on_the_body_arm_channel():
    with Rig() as r:
        job, h = make_job("man-b1")
        out = asyncio.run(r.exe.run(job, h))
        d = out.data
        assert out.status == "failed" and out.reason == "timeout", out.detail          # nothing lifts zero-shot
        assert not set(r.body.rejected()) & {"bad_args", "bad_chunk", "mode_mismatch", "stale_command"}, \
            r.body.rejected()
        assert r.body.ch.stats["chunks_applied"] >= 3 and d["chunks_sent"] >= 3
        assert not any(k.startswith("body:") for k in d["chunks_dropped"]), d["chunks_dropped"]
        assert d["clamped_frac_source"] == "body" and d["clamped_frac"] is not None
        prog = [e["data"] for e in r.body.events if e["state"] == "progress"]
        assert prog and prog[-1]["kind"] == "chunk" and prog[-1]["lead_s"] == pytest.approx(0.15)
        term = r.body.terminal("man-b1")
        assert len(term) == 1 and term[0]["state"] == "succeeded" and term[0]["data"]["ended_by"] == "client"
        assert term[0]["data"]["hold"] == "measured" and d["hold_on_end"] == "measured"
        assert r.body.ch.state()["mode"] == "hold"                                   # HOLD at the measured pose
        # the result carries the body's own counters from its terminal event (the executor waits for it)
        b = d["body"]
        assert b["source"] == "terminal" and b["chunks"]["applied"] == r.body.ch.stats["chunks_applied"]
        assert d["slew_frac"] == b["slew_frac_total"] and d["clamped_frac"] == b["clamped_frac_total"]
        assert b["max_step_rad"] is not None and b["lead_s"] == pytest.approx(0.15)


def test_a_body_halt_latch_ends_the_session_halted():
    gate = HaltGate()
    with Rig(gate=gate, cfg={"max_duration_s": 6.0}) as r:
        job, h = make_job("man-b2", gate_epoch=gate.epoch)

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 4.0
            while r.body.ch.stats["chunks_applied"] < 2 and time.monotonic() < t_end:
                await asyncio.sleep(0.02)
            epoch = gate.halt()                   # G1Robot.halt: the runtime gate, then the body's halt lane latch
            r.body.latch(max(epoch, job.control_epoch))
            n_at_latch = r.body.ch.stats["chunks_applied"]
            out = await task
            return out, n_at_latch

        out, n = asyncio.run(main())
        assert out.status == "failed" and out.reason == "halted", out.detail
        assert r.body.ch.stats["chunks_applied"] == n                               # nothing applied after the latch
        t_ack = r.exe.last_session.t_ack
        assert [t for t, a, _ in r.body.replies if "chunk" in a and t > t_ack] == []   # nothing SENT after the ack
        term = r.body.terminal("man-b2")
        assert len(term) == 1 and term[0]["state"] == "canceled" and term[0]["data"]["ended_by"] == "halt"
        assert r.body.ch.state()["mode"] == "latched"


def test_cancel_ends_the_session_at_once():
    with Rig(cfg={"max_duration_s": 6.0}) as r:
        job, h = make_job("man-b3")

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 4.0
            while r.body.ch.stats["chunks_applied"] < 2 and time.monotonic() < t_end:
                await asyncio.sleep(0.02)
            h.cancel("user")
            t_c = time.monotonic()
            out = await task
            return out, time.monotonic() - t_c

        out, dt = asyncio.run(main())
        assert out.status == "cancelled" and dt < 1.0, (out.detail, dt)
        t_ack = r.exe.last_session.t_ack
        sent_after = [t for t, a, rep in r.body.replies if "chunk" in a and t > t_ack]   # whatever the body said
        assert sent_after == []
        term = r.body.terminal("man-b3")
        assert len(term) == 1 and term[0]["data"]["ended_by"] == "client" and term[0]["data"]["hold"] == "stand"
