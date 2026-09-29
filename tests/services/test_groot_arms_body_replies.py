"""groot_arms on the as-built body wire (docs/contracts/arm_chunk.md §8, docs/contracts/m1.md §3.9-§3.11): which body
refusals end a GR00T session and with which reason, how a session the body ended on its own is mapped, the body's
own counters in the result, and the warm-up observation kept out of recorded evidence.

    stale_command   keyed on data.why: the fence (control_epoch / resume_epoch -> halted, generation -> superseded)
                    ends the session; a single late message (t_wall older than the watchdog, no data.why) is counted
                    and the session goes on
    body_busy       (a B.2 lease held by another execution) ends it as body_busy
    3 in a row      refusals of any kind, an unknown code too, end it as controller_unavailable
    ended_by        the body's terminal event while the executor streams: watchdog -> policy_stall, stop -> halted,
                    halt -> halted, fault -> fell, preempted/taken_over -> body_busy, anything else ->
                    controller_unavailable

Every fatal case also checks that no chunk message of the session reached the body after the executor's ack,
whatever the body answered (the wave-1 verifier: count every chunk SENT, not only the accepted ones).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("groot.actions")

from services.executors.groot_arms import (ENDED_BY, GrootArmExecutor, _groot_helpers, body_reject_key,  # noqa: E402
                                           body_verdict)
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from tests.services.test_groot_arms import FakeWorld, Sink, fast_cfg, make_job  # noqa: E402


# ================================================================================================ the table
@pytest.mark.parametrize("rep,want", [
    ({"ok": True, "state": "done"}, None),
    ({"ok": False, "error": "stale_command", "data": {"id": "arm-1", "t_wall": 1.0}}, None),        # one late message
    ({"ok": False, "error": "stale_command", "data": {"why": "control_epoch", "halt_epoch": 3}}, "halted"),
    ({"ok": False, "error": "stale_command", "data": {"why": "resume_epoch", "halt_epoch": 3}}, "halted"),
    ({"ok": False, "error": "stale_command", "data": {"why": "generation", "generation_floor": 4}}, "superseded"),
    ({"ok": False, "error": "body_busy", "data": {"lease": {"owner": "nav-7"}}}, "body_busy"),
    ({"ok": False, "error": "halted", "data": {"halt_epoch": 2}}, "halted"),
    ({"ok": False, "error": "stale_session", "data": {"ended": True}}, "controller_unavailable"),
    ({"ok": False, "error": "arm_preempted"}, "body_busy"),
    ({"ok": False, "error": "arm_stopped"}, "halted"),
    ({"ok": False, "error": "fault:fallen"}, "fell"),
    ({"ok": False, "error": "fault:deploy_lost"}, "controller_unavailable"),
    ({"ok": False, "error": "body_timeout"}, "controller_unavailable"),
    ({"ok": False, "error": "something_new"}, None),                                  # counted; 3 in a row end it
])
def test_body_verdict_table(rep, want):
    v = body_verdict(rep)
    assert (v[0] if v else None) == want


def test_reject_keys_tell_the_two_stale_commands_apart():
    assert body_reject_key({"error": "stale_command", "data": {"t_wall": 1.0}}) == "body:stale_command(t_wall)"
    assert body_reject_key({"error": "stale_command", "data": {"why": "generation"}}) == \
        "body:stale_command(generation)"
    assert body_reject_key({"error": "body_busy"}) == "body:body_busy"


def test_ended_by_mapping():
    assert ENDED_BY["watchdog"] == "policy_stall" and ENDED_BY["stop"] == "halted" and ENDED_BY["halt"] == "halted"
    assert ENDED_BY["fault"] == "fell" and ENDED_BY["preempted"] == "body_busy"
    assert "client" not in ENDED_BY and "script" not in ENDED_BY        # -> controller_unavailable (the default)


# ================================================================================================ live sessions
class InjectingBody:
    """ArmPort + SensorPort over FakeArmBody that answers chosen chunk messages itself (a synthetic refusal) and
    records every message it was sent, with its time, whatever the answer."""

    def __init__(self, body: FakeArmBody, inject):
        self.body, self.inject = body, inject
        self.sent: list[tuple[float, str, dict, dict]] = []        # (t, kind, args, reply)
        self.n_chunks = 0

    def arm(self, args: dict, op_id: str | None = None) -> dict:
        t = time.monotonic()
        kind = "end" if args.get("end") else "keepalive" if args.get("keepalive") else \
            "chunk" if "chunk" in args else "start"
        rep = None
        if kind == "chunk":
            self.n_chunks += 1
            rep = self.inject(self.n_chunks, args)
        if rep is None:
            rep = self.body.arm(args, op_id)
        self.sent.append((t, kind, args, rep))
        return rep

    def subscribe(self, cb):
        return self.body.subscribe(cb)

    def ego_frame(self):
        return self.body.ego_frame()

    def debug_state(self):
        return self.body.debug_state()

    def chunks_after(self, t_ack: float) -> list[float]:
        return [t for t, k, _, _ in self.sent if k == "chunk" and t > t_ack]


class Rig:
    def __init__(self, inject, cfg: dict | None = None):
        self.srv = FakePolicyServer(close_after=None).start()
        self.fake = FakeArmBody()
        self.body = InjectingBody(self.fake, inject)
        self.world = FakeWorld("never")
        c = {"max_duration_s": 2.5, **(cfg or {})}
        self.exe = GrootArmExecutor(self.world, arm=self.body, sensors=self.body, cfg=fast_cfg(self.srv, **c),
                                    events=Sink(), helpers=_groot_helpers())

    def __enter__(self):
        t_end = time.monotonic() + 5.0
        while not self.exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        assert self.exe.health().ok, self.exe.health().detail
        return self

    def __exit__(self, *exc):
        self.exe.close()
        self.fake.close()
        self.srv.stop()

    def run(self, eid: str):
        job, h = make_job(eid)
        out = asyncio.run(self.exe.run(job, h))
        self.exe.last_client.join(2.0)
        return out, self.exe.last_session


def _rej(err: str, **data) -> dict:
    return {"ok": False, "state": "rejected", "error": err, "data": data}


def test_a_late_message_is_counted_and_the_session_goes_on():
    with Rig(lambda n, a: _rej("stale_command", id="x", t_wall=a["t_wall"] - 5.0) if n == 2 else None) as r:
        out, s = r.run("man-late")
        assert out.status == "failed" and out.reason == "timeout", out.detail       # the normal zero-shot end
        assert out.data["chunks_dropped"].get("body:stale_command(t_wall)") == 1
        assert out.data["chunks_sent"] >= 3                                        # chunks 3.. went on


@pytest.mark.parametrize("why,reason", [("control_epoch", "halted"), ("resume_epoch", "halted"),
                                        ("generation", "superseded")])
def test_a_fenced_stream_ends_the_session(why, reason):
    with Rig(lambda n, a: _rej("stale_command", why=why) if n >= 2 else None, cfg={"max_duration_s": 6.0}) as r:
        t0 = time.monotonic()
        out, s = r.run(f"man-fence-{why}")
        assert out.status == "failed" and out.reason == reason, out.detail
        assert f"stale_command ({why})" in out.detail and time.monotonic() - t0 < 4.0
        assert out.data["chunks_dropped"].get(f"body:stale_command({why})") == 1
        assert r.body.chunks_after(s.t_ack) == []                                  # every chunk sent, counted


def test_a_lease_held_by_another_execution_ends_the_session_body_busy():
    with Rig(lambda n, a: _rej("body_busy", lease={"owner": "nav-9"}) if n >= 2 else None,
             cfg={"max_duration_s": 6.0}) as r:
        out, s = r.run("man-busy")
        assert out.status == "failed" and out.reason == "body_busy", out.detail
        assert r.body.chunks_after(s.t_ack) == []


def test_three_unknown_refusals_in_a_row_end_the_session():
    with Rig(lambda n, a: _rej("mystery") if n >= 2 else None, cfg={"max_duration_s": 8.0}) as r:
        out, s = r.run("man-unknown")
        assert out.status == "failed" and out.reason == "controller_unavailable", out.detail
        assert "3 arm messages in a row" in out.detail and out.data["chunks_dropped"]["body:mystery"] == 3
        assert r.body.chunks_after(s.t_ack) == []


@pytest.mark.parametrize("by,state,reason", [("watchdog", "failed", "policy_stall"), ("stop", "canceled", "halted"),
                                             ("client", "succeeded", "controller_unavailable"),
                                             ("taken_over", "canceled", "body_busy")])
def test_a_session_the_body_ends_is_mapped_by_ended_by(by, state, reason):
    with Rig(lambda n, a: None, cfg={"max_duration_s": 6.0}) as r:
        job, h = make_job(f"man-end-{by}")

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 4.0
            while len(r.fake.chunk_times(job.execution_id)) < 2 and time.monotonic() < t_end:
                await asyncio.sleep(0.02)
            with r.fake.lock:                                  # the body ends the session on its own
                ev = r.fake._end(r.fake.cur, state, by, "measured", "client_silent" if by == "watchdog" else None)
            r.fake._publish(ev)
            return await task

        out = asyncio.run(main())
        r.exe.last_client.join(2.0)
        s = r.exe.last_session
        assert out.status == "failed" and out.reason == reason, out.detail
        assert f"ended_by {by}" in out.detail
        assert r.body.chunks_after(s.t_ack) == []
        assert out.data["body"]["source"] == "terminal" and out.data["body"]["ended_by"] == by
        assert r.body.sent[-1][1] != "end"                     # the body already ended it: no `end` sent


def test_the_result_carries_the_bodys_terminal_counters():
    with Rig(lambda n, a: None) as r:
        out, s = r.run("man-counters")
        b = out.data["body"]
        assert b["source"] == "terminal" and b["ended_by"] == "client" and b["hold"] == "measured"
        assert b["chunks"]["applied"] == out.data["chunks_sent"] and "clamped_frac_total" in b
        assert out.data["clamped_frac"] == b["clamped_frac_total"]


# ================================================================================================ warm-up
def test_the_warmup_observation_is_not_recorded(tmp_path: Path):
    """frame_01 of the wave-1 live smoke was the black warm-up frame: the recorder must only see session
    observations."""
    from tools.groot_live_smoke import ChainRecorder
    srv = FakePolicyServer(close_after=None).start()
    fake = FakeArmBody()
    rec = ChainRecorder(tmp_path, save_frames=(1,))
    h = rec.wrap(_groot_helpers())
    exe = GrootArmExecutor(FakeWorld("never"), arm=fake, sensors=fake, cfg=fast_cfg(srv, max_duration_s=1.0),
                           events=Sink(), helpers=h)
    try:
        t_end = time.monotonic() + 5.0
        while not exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        assert exe.health().ok and exe.policy_health.warm and srv.calls >= 1       # the warm-up call ran
        assert rec.n_obs == 0 and rec.frames == []                                 # ... and was not recorded
        job, hd = make_job("man-rec")
        asyncio.run(exe.run(job, hd))
        exe.last_client.join(2.0)
        assert rec.n_obs >= 1 and rec.frames == ["frame_01.png"]
        from PIL import Image
        img = np.asarray(Image.open(tmp_path / "frame_01.png"))
        assert img.mean() > 50                                                     # the session's frame, not black
    finally:
        exe.close()
        fake.close()
        srv.stop()
