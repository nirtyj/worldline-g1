"""groot_arms against the REAL PolicyServer and the real Arena N1.7 G1 checkpoint (opt-in: `-m box`).

The body and the world are still the fakes (tests/fakes/fake_arm_body.py, the FakeWorld of test_groot_arms.py): this
checks the executor's side of the loop with real inference latency and the real action layout, not a grasp. Run on
the dev box after `bash scripts/groot_server.sh start` (or anywhere with a tunnel):

    GROOT_ENDPOINT=tcp://127.0.0.1:5550 GROOT_LIVE_OUT=outputs/m2b_wave1/groot_rt/live.json \
    GROOT_KILL_CMD="bash scripts/groot_server.sh stop" .venv-rt/bin/python -m pytest -m box \
        tests/services/test_groot_arms_live.py

GROOT_KILL_CMD is optional: with it, the last test stops the server in the middle of a session (F7). Every number the
tests measured goes to GROOT_LIVE_OUT (JSON).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("groot.policy_client")

from groot.policy_client import PolicyClient  # noqa: E402
from services.common import HaltGate  # noqa: E402
from services.executors.groot_arms import GrootArmExecutor, GrootArmsConfig, _groot_helpers  # noqa: E402
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.services.test_groot_arms import FakeWorld, Sink, make_job  # noqa: E402

pytestmark = pytest.mark.box

ENDPOINT = os.environ.get("GROOT_ENDPOINT", "tcp://127.0.0.1:5550")
OUT = os.environ.get("GROOT_LIVE_OUT")
KILL = os.environ.get("GROOT_KILL_CMD")
REPORT: dict = {"endpoint": ENDPOINT, "t_wall": time.time(), "runs": {}}


def _save(name: str, rec: dict) -> None:
    REPORT["runs"][name] = rec
    if OUT:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps(REPORT, indent=1, default=str))


@pytest.fixture(scope="module", autouse=True)
def server():
    c = PolicyClient(ENDPOINT, timeout_s=30.0)
    try:
        if not c.ping(timeout_s=5.0):
            pytest.skip(f"no PolicyServer at {ENDPOINT}")
    finally:
        c.close()


def _rig(gate=None, **cfg):
    body = FakeArmBody()
    world = FakeWorld("never")
    world.couple(body)
    c = GrootArmsConfig(endpoint=ENDPOINT, **{"max_duration_s": 6.0, **cfg})
    exe = GrootArmExecutor(world, arm=body, sensors=body, cfg=c, gate=gate, events=Sink(), helpers=_groot_helpers())
    t_end = time.monotonic() + 60.0
    while not exe.health().ok and time.monotonic() < t_end:
        time.sleep(0.2)
    return exe, body


def _summary(out, body, s) -> dict:
    d = out.data
    prog = [e["data"] for e in body.events if e["state"] == "progress" and e["data"].get("session_id") == s.id]
    return {"status": out.status, "reason": out.reason, "inferences": d["inferences"], "chunks_sent": d["chunks_sent"],
            "chunks_dropped": d["chunks_dropped"], "latency_ms": d["latency_ms"], "clamped_frac": d["clamped_frac"],
            "stall_s_max": d["stall_s_max"], "hold_on_end": d["hold_on_end"], "keepalives": d["keepalives"],
            "body_latency_ms": [p.get("latency_ms") for p in prog if p.get("latency_ms") is not None][-10:],
            "detail": out.detail}


def test_live_session_runs_and_reports_honestly():
    exe, body = _rig()
    try:
        assert exe.health().ok, exe.health().detail
        job, h = make_job("live-run")
        out = asyncio.run(exe.run(job, h))
        rec = _summary(out, body, exe.last_session)
        _save("session", rec)
        assert out.status in ("failed", "succeeded") and out.reason in (None, "timeout", "grasp_missed"), rec
        assert rec["inferences"] >= 10 and rec["chunks_sent"] >= 5, rec
        assert rec["latency_ms"]["p95"] < 1500, rec
    finally:
        exe.close()
        body.close()


@pytest.mark.parametrize("delay", [0.6, 1.3, 2.1])
def test_live_cancel(delay):
    exe, body = _rig()
    try:
        job, h = make_job(f"live-cancel-{delay}")

        async def main():
            task = asyncio.ensure_future(exe.run(job, h))
            await asyncio.sleep(delay)
            t = time.monotonic()
            h.cancel("correction")
            out = await task
            return out, t, time.monotonic()

        out, t_cancel, t_done = asyncio.run(main())
        s = exe.last_session
        exe.last_client.join(3.0)
        late = [m["t"] for m in body.messages("chunk", s.id) if m["t"] > s.t_ack]     # every chunk SENT after it
        _save(f"cancel_{delay}", {"status": out.status, "terminal_s": round(t_done - t_cancel, 3),
                                  "ack_ms": round((s.t_ack - t_cancel) * 1000, 2), "chunks_before": len(
                                      body.chunk_times(s.id)), "chunks_after_ack": len(late),
                                  "stale_dropped": s.dropped.get("stale_session", 0)})
        assert out.status == "cancelled" and late == [] and t_done - t_cancel <= 1.3
    finally:
        exe.close()
        body.close()


@pytest.mark.parametrize("delay", [0.6, 1.3, 2.1])
def test_live_halt(delay):
    gate = HaltGate()
    exe, body = _rig(gate=gate)
    try:
        job, h = make_job(f"live-halt-{delay}", gate_epoch=gate.epoch)

        async def main():
            task = asyncio.ensure_future(exe.run(job, h))
            await asyncio.sleep(delay)
            t = time.monotonic()
            body.halt(gate.halt())
            out = await task
            return out, t, time.monotonic()

        out, t_halt, t_done = asyncio.run(main())
        s = exe.last_session
        exe.last_client.join(3.0)
        late = [m["t"] for m in body.messages("chunk", s.id) if m["t"] > s.t_ack]     # every chunk SENT after it
        _save(f"halt_{delay}", {"status": out.status, "reason": out.reason, "terminal_s": round(t_done - t_halt, 3),
                                "chunks_before": len(body.chunk_times(s.id)), "chunks_after_ack": len(late),
                                "body_hold": body.hold["mode"] if body.hold else None})
        assert out.status == "failed" and out.reason == "halted" and late == []
    finally:
        exe.close()
        body.close()


@pytest.mark.skipif(not KILL, reason="GROOT_KILL_CMD not set (F7 stops the server)")
def test_live_f7_server_stopped_mid_session():
    exe, body = _rig(max_duration_s=30.0)
    try:
        job, h = make_job("live-f7")

        async def main():
            task = asyncio.ensure_future(exe.run(job, h))
            await asyncio.sleep(2.0)
            t = time.monotonic()
            await asyncio.to_thread(subprocess.run, KILL, shell=True, check=False, capture_output=True)
            t_killed = time.monotonic()
            out = await task
            return out, t, t_killed, time.monotonic()

        out, t_kill, t_killed, t_done = asyncio.run(main())
        health = exe.health()
        _save("f7", {"status": out.status, "reason": out.reason, "since_kill_cmd_s": round(t_done - t_kill, 3),
                     "since_kill_returned_s": round(t_done - t_killed, 3), "kill_cmd_s": round(t_killed - t_kill, 3),
                     "hold": body.hold["mode"] if body.hold else None, "health_after": [health.ok, health.detail],
                     "policy_errors": out.data["policy_errors"], "detail": out.detail})
        assert out.status == "failed" and out.reason == "policy_unavailable", out.detail
        assert t_done - t_killed <= 3.0 and body.hold["mode"] == "measured" and not health.ok
    finally:
        exe.close()
        body.close()
