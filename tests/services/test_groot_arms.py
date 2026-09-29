"""groot_arms (R.4): GrootArmExecutor against a fake PolicyServer (a real ZMQ REP socket speaking the upstream wire,
tests/fakes/fake_policy_server.py), a fake wl-body arm channel in chunk mode (tests/fakes/fake_arm_body.py, the
docs/contracts/arm_chunk.md reference) and a fake GT world. groot_srv's real `groot/` helpers build the observations
and map the chunks. Everything is offline and runs in wall time, like the executor.

Covered: the success path from ground truth and the carry hand-over; honest zero-shot failures (grasp_missed,
object_dropped, timeout); F7 (a dead server: rejected for new calls; a server dying mid-call: failed
policy_unavailable within 3 s, then HOLD, then recovery); cancel 3/3 and halt 3/3 with no chunk published after the
ack; stale-session drops on both sides; NaN and out-of-bounds chunks; stance, view and camera handling; the
registry, the `full` profile and the retired groot_sonic.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

pytest.importorskip("groot.actions")                   # groot_srv's model-side helpers

from api.execution import Execution, ResultHandle  # noqa: E402
from api.skills import BACKEND_ORDER, DEPRECATED_EXECUTORS, EXECUTOR_OF_BACKEND, StaticSkillRegistry  # noqa: E402
from api.types import ServiceHealth  # noqa: E402
from services.common import HaltGate  # noqa: E402
from services.executors.groot_arms import (CARRY_LABEL, GrootArmExecutor, GrootArmOutcome,  # noqa: E402
                                           GrootArmsConfig, _groot_helpers, create, hand_closure)
from services.executors.kinematic_attach import ManipJob  # noqa: E402
from services.skills import load_skill_specs  # noqa: E402
from tests.fakes.fake_arm_body import FakeArmBody  # noqa: E402
from tests.fakes.fake_policy_server import FakePolicyServer  # noqa: E402
from world.model import Detection, ObjectState, RobotPose  # noqa: E402

APPLE = "groot.pick.apple.arena_static_experimental.v0"
ANY = "groot.pick.any.arena_static_experimental.v0"
SKILLS = {s.skill_id: s for s in load_skill_specs()}


# ================================================================================================ fakes
class FakeWorld:
    """The WorldModel surface groot_arms reads. The apple rests on a 0.95 m counter; `scenario` decides what the
    measured left hand does to it (coupled through FakeArmBody.on_tick):

        never   nothing moves (the zero-shot expectation)
        lift    once the hand is >= 0.6 closed (of Arena's closed pose) for 0.2 s, the apple rises 8 cm and stays
        drop    as lift, then it falls back onto the counter 0.4 s later
    """

    Z0 = 0.95

    def __init__(self, scenario: str = "never", *, speed: float = 0.0, ego_px: float | None = None,
                 enable_camera: bool = False, palm: bool = False):
        self.scenario, self.speed, self.ego_px = scenario, speed, ego_px
        self.can_enable, self.palm = enable_camera, palm
        self.box = ((0.95, 0.10, self.Z0), (1.03, 0.18, self.Z0 + 0.08))
        self.fallen = False
        self.cam_calls: list[tuple] = []
        self.t_closed: float | None = None
        self.t_lifted: float | None = None

    def robot_pose(self):
        return RobotPose(0.55, 0.14, 0.787, 0.0, time.time(), "fake", fallen=self.fallen, pelvis_z=0.787, vx=self.speed)

    def object(self, oid):
        if oid != "apple_1":
            return None
        return ObjectState("apple_1", "apple", "Apple", self.box, "on kitchen_counter_1a", pose_source="sim")

    def capabilities(self):
        return {"detections:ego_view": self.ego_px is not None, "enable_camera": self.can_enable}

    def detections(self, camera="head", *, method=None, **kw):
        c = tuple((self.box[0][i] + self.box[1][i]) / 2 for i in range(3))
        return [Detection("apple_1", "object", "apple", "Apple", "on kitchen_counter_1a", c, 0.5, float(self.ego_px),
                          method="instance_id_segmentation_fast" if method == "best" else "gt-geometric")]

    def palm_position(self, arm):
        return tuple((self.box[0][i] + self.box[1][i]) / 2 for i in range(3)) if self.palm else None

    def enable_camera(self, camera, on, *, consumer="runtime", ttl_s=None):
        self.cam_calls.append((camera, on, consumer, ttl_s, time.monotonic()))
        return {"name": camera, "on": on}

    def hands(self):
        return {"left": None, "right": None}

    def _set_z(self, z: float) -> None:
        (x0, y0, _), (x1, y1, _) = self.box
        self.box = ((x0, y0, z), (x1, y1, z + 0.08))

    def couple(self, body: FakeArmBody) -> None:
        def tick(b: FakeArmBody, now: float) -> None:
            if self.scenario == "never":
                return
            closed = hand_closure("left", b.hand_q["left"]) >= 0.6
            if closed and self.t_closed is None:
                self.t_closed = now
            if self.t_lifted is None and self.t_closed is not None and now - self.t_closed >= 0.2:
                self.t_lifted = now
                self._set_z(self.Z0 + 0.08)
            if self.scenario == "drop" and self.t_lifted is not None and now - self.t_lifted >= 0.4:
                self._set_z(self.Z0)
        body.on_tick(tick)


class Sink:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, type, **fields):
        self.events.append((type, fields))
        return {"type": type, **fields}

    def types(self) -> set[str]:
        return {t for t, _ in self.events}


def fast_cfg(srv: FakePolicyServer, **over) -> GrootArmsConfig:
    cfg = GrootArmsConfig(endpoint=srv.endpoint, timeout_s=0.6, ping_timeout_s=0.3, ping_period_s=0.5,
                          warmup_timeout_s=3.0, replan_s=0.2, hold_s=0.4, grasp_missed_s=0.6, stall_event_s=0.5,
                          stall_fail_s=1.5, oob_window_s=0.5, hands_open_s=0.05, settle_s=0.3, max_duration_s=6.0,
                          camera_refresh_s=0.3)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_job(eid: str = "man-1", *, skill: str = APPLE, otype: str = "apple", gate_epoch: int = 0):
    """A pick job the way ManipulationService builds it (with the fence fields) and its ResultHandle."""
    ex = Execution(execution_id=eid, tool_name="manipulate", args={}, generation=3, control_epoch=2)
    job = ManipJob("pick", "apple_1", "left", skill, epoch=gate_epoch)
    for k, v in dict(execution_id=eid, generation=3, control_epoch=2, object_type=otype, skill=SKILLS[skill]).items():
        setattr(job, k, v)
    return job, ResultHandle(ex)


class Rig:
    """One executor over the fakes. Use as a context manager (it stops the threads and the server)."""

    def __init__(self, scenario="never", *, server: dict | None = None, world: dict | None = None,
                 cfg: dict | None = None, gate: HaltGate | None = None):
        self.srv = FakePolicyServer(**(server or {})).start()
        self.body = FakeArmBody()
        self.world = FakeWorld(scenario, **(world or {}))
        self.world.couple(self.body)
        self.sink = Sink()
        self.gate = gate
        self.exe = GrootArmExecutor(self.world, arm=self.body, sensors=self.body, cfg=fast_cfg(self.srv, **(cfg or {})),
                                    gate=gate, events=self.sink, helpers=_groot_helpers())

    def __enter__(self) -> "Rig":
        return self

    def __exit__(self, *exc) -> None:
        self.exe.close()
        self.body.close()
        self.srv.stop()

    def healthy(self, timeout: float = 5.0) -> ServiceHealth:
        t_end = time.monotonic() + timeout
        h = self.exe.health()
        while not h.ok and time.monotonic() < t_end:
            time.sleep(0.05)
            h = self.exe.health()
        return h

    def job(self, eid: str = "man-1", **kw):
        return make_job(eid, **kw)

    def run(self, eid: str = "man-1", **kw) -> GrootArmOutcome:
        job, h = self.job(eid, **kw)
        return asyncio.run(self.exe.run(job, h))


def _no_chunk_after(body: FakeArmBody, sid: str, t_ack: float) -> list[float]:
    return [t for t in body.chunk_times(sid) if t > t_ack]


# ================================================================================================ outcomes
def test_success_from_ground_truth_hands_over_to_the_carry_hold():
    with Rig("lift", server={"close_after": 2}) as r:
        assert r.healthy().ok
        out = r.run()
        d = out.data
        assert isinstance(out, GrootArmOutcome) and out.status == "succeeded" and out.reason is None, out.detail
        assert out.phase == "verify" and out.holding is True
        assert d["executor"] == "groot_arms" and d["label"] == d["skill_label"] == "experimental"
        assert d["inferences"] >= 3 and d["chunks_sent"] >= 2 and d["session_id"] == "man-1"
        assert 0 < d["latency_ms"]["p50"] <= d["latency_ms"]["p95"] and d["latency_ms"]["n"] == d["inferences"]
        assert d["hold_on_end"] == "target" and d["carry"] == CARRY_LABEL and "INTERIM" in d["carry"]
        assert d["attempts"] == [dict(d["attempts"][0], executor="groot_arms", skill=APPLE, status="succeeded")]
        assert d["gt"]["lift_max_m"] >= 0.05 and "INTERIM" in d["gt"]["palm"]
        assert d["base_shift_m"] == 0.0 and d["clamped_frac_source"] == "body"
        assert [p["phase"] for p in out.phases] == ["stance_check", "view_check", "enter", "execute", "carry_lock"]
        assert "experimental" in out.detail and "zero-shot" in out.detail
        # the wire (docs/contracts/arm_chunk.md §1): a start with open hands, time-stamped chunks, an end with a hold
        start = r.body.messages("start", "man-1")[0]
        assert start["op_id"] == "arm-man-1" and start["reply"]["state"] == "accepted"
        a = start["args"]
        assert (a["stream"], a["session_id"], a["execution_id"], a["generation"], a["control_epoch"], a["mode"]) == \
               ("man-1", "man-1", "man-1", 3, 2, "chunk")
        assert a["left_hand"] == a["right_hand"] == [0.0] * 7 and a["hold_on_end"] == "measured"
        chunks = [m for m in r.body.messages("chunk", "man-1") if m["reply"]["ok"]]
        c = chunks[0]["chunk"]
        assert c["order"] == "wire" and c["dt"] == pytest.approx(0.02) and c["upper_body"] == (40, 17)
        assert c["left_hand"] == c["right_hand"] == (40, 7) and isinstance(c["t0_mono"], float)
        assert [m["seq"] for m in chunks] == sorted(m["seq"] for m in chunks)
        end = r.body.messages("end", "man-1")[-1]
        assert end["args"]["hold_on_end"] == "target" and end["reply"]["ok"]
        assert r.body.hold["mode"] == "target"
        assert r.srv.prompts and set(r.srv.prompts) == {"move the apple to the plate"}
        assert {"manip.phase", "groot.inference", "arm.progress", "policy.health"} <= r.sink.types()


def test_zero_shot_grasp_missed_is_reported_honestly():
    with Rig("never", server={"close_after": 1}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "grasp_missed", out.detail
        assert out.holding is False and out.data["hold_on_end"] == "measured"
        assert out.data["gt"]["lift_max_m"] < 0.01 and out.data["chunks_sent"] >= 1
        assert r.body.messages("end", "man-1")[-1]["args"]["reason"] == "grasp_missed"


def test_object_dropped_after_a_lift():
    with Rig("drop", server={"close_after": 1}, cfg={"hold_s": 5.0}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "object_dropped", out.detail
        assert out.data["gt"]["lift_max_m"] >= 0.05 and out.data["hold_on_end"] == "measured"


def test_timeout_when_the_policy_never_grasps():
    with Rig("never", server={"close_after": None}, cfg={"max_duration_s": 1.2}) as r:
        assert r.healthy().ok
        t0 = time.monotonic()
        out = r.run()
        assert out.status == "failed" and out.reason == "timeout", out.detail
        assert 1.2 <= time.monotonic() - t0 < 3.0 and out.data["chunks_sent"] >= 2


# ================================================================================================ F7: policy down
def test_f7_dead_server_is_rejected_for_new_calls_and_the_enum_stays_frozen():
    with Rig("never") as r:
        assert r.healthy().ok
        ok = ServiceHealth(True, "ok")
        reg = StaticSkillRegistry(SKILLS.values(), vocab=["apple", "mug"], backend_order=BACKEND_ORDER["full"],
                                  health_fn=lambda s: r.exe.health() if s.backend == "groot" else ok)
        types = reg.loaded_object_types()
        assert reg.select_healthy("pick", "apple", "left")[0].skill_id == APPLE
        r.srv.die()
        t_end = time.monotonic() + 3.0
        while r.exe.health().ok and time.monotonic() < t_end:
            time.sleep(0.05)
        h = r.exe.health()
        assert not h.ok and r.srv.endpoint in h.detail
        skill, why = reg.select_healthy("pick", "apple", "left")
        assert skill is not None and skill.backend == "sonic_arm_script"          # groot_then_script falls through
        only_groot = StaticSkillRegistry(SKILLS.values(), vocab=["apple", "mug"], backend_order=("groot",),
                                         health_fn=lambda s: r.exe.health())
        skill, why = only_groot.select_healthy("pick", "apple", "left")
        assert skill is None and why.startswith(APPLE) and "PolicyServer" in why      # -> "policy unavailable: ..."
        assert reg.loaded_object_types() == types                                     # health never changes the enum
        out = r.run("man-2")
        assert out.status == "failed" and out.reason == "policy_unavailable" and out.phase == "select_skill"
        assert r.body.log == []                                                       # the body was never touched


def test_f7_server_dies_mid_call_fails_within_3s_then_holds_then_recovers():
    # production timeouts: 1.5 s get_action, 0.5 s ping
    with Rig("never", server={"close_after": None},
             cfg={"timeout_s": 1.5, "ping_timeout_s": 0.5, "max_duration_s": 20.0, "stall_fail_s": 10.0}) as r:
        assert r.healthy().ok
        job, h = r.job()

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            await asyncio.sleep(0.8)
            t_die = time.monotonic()
            await asyncio.to_thread(r.srv.die)
            out = await task
            return out, time.monotonic() - t_die

        out, dt = asyncio.run(main())
        assert out.status == "failed" and out.reason == "policy_unavailable", out.detail
        assert dt <= 3.0, f"policy_unavailable after {dt:.2f} s"
        assert out.data["policy_errors"] >= 2 and out.data["chunks_sent"] >= 1
        assert out.data["hold_on_end"] == "measured" and r.body.hold["mode"] == "measured"   # then HOLD
        assert not r.exe.health().ok                                                        # new calls: rejected
        r.srv.revive()
        assert r.healthy(timeout=4.0).ok                                                    # back on a good ping


# ================================================================================================ cancel and halt
@pytest.mark.parametrize("delay", [0.35, 0.7, 1.1])
def test_cancel_stops_the_stream_at_once(delay):
    with Rig("never", server={"close_after": None}) as r:
        assert r.healthy().ok
        job, h = r.job(f"man-c{delay}")

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            await asyncio.sleep(delay)
            t_cancel = time.monotonic()
            h.cancel("correction")
            out = await task
            return out, t_cancel, time.monotonic()

        out, t_cancel, t_done = asyncio.run(main())
        s = r.exe.last_session
        assert out.status == "cancelled" and out.reason == "correction", out.detail
        assert t_done - t_cancel <= 1.3                                  # PLAN §5.6 worst case, not holding
        assert s.t_ack is not None and s.t_ack - t_cancel < 0.5
        assert len(r.body.chunk_times(s.id)) >= 1                        # the stream was live
        r.exe.last_client.join(2.0)                                      # let an in-flight inference come back
        assert _no_chunk_after(r.body, s.id, s.t_ack) == []              # nothing published after the cancel ack
        assert out.data["hold_on_end"] == "stand" and r.body.messages("end", s.id)[-1]["args"]["hold_on_end"] == "stand"


def test_cancel_while_holding_keeps_the_object():
    with Rig("lift", server={"close_after": 0}, cfg={"hold_s": 30.0}) as r:
        assert r.healthy().ok
        job, h = r.job()

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 4.0
            while r.world.t_lifted is None and time.monotonic() < t_end:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            h.cancel("correction")
            return await task

        out = asyncio.run(main())
        assert out.status == "cancelled" and out.holding is True
        assert out.data["hold_on_end"] == "target" and r.body.hold["mode"] == "target"


@pytest.mark.parametrize("variant", ["latch", "no_latch", "latch_late"])
def test_halt_stops_the_stream_at_once(variant):
    gate = HaltGate()
    with Rig("never", server={"close_after": None}, gate=gate) as r:
        assert r.healthy().ok
        job, h = r.job(f"man-h-{variant}", gate_epoch=gate.epoch)

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            await asyncio.sleep(1.2 if variant == "latch_late" else 0.5)
            t_halt = time.monotonic()
            epoch = gate.halt()                          # G1Robot.halt: the runtime latch first,
            if variant != "no_latch":
                r.body.halt(epoch)                       # then the body's B.1 latch (absent on M1's body)
            out = await task
            return out, t_halt, time.monotonic()

        out, t_halt, t_done = asyncio.run(main())
        s = r.exe.last_session
        assert out.status == "failed" and out.reason == "halted", out.detail
        assert t_done - t_halt < 0.5 and s.t_ack is not None
        r.exe.last_client.join(2.0)
        assert _no_chunk_after(r.body, s.id, s.t_ack) == []
        assert r.body.hold["mode"] == "measured"         # the hands were never opened by a blend
        if variant == "no_latch":                        # M1's body: the runtime ends the session itself
            end = r.body.messages("end", s.id)[-1]
            assert end["args"]["hold_on_end"] == "measured" and end["args"]["reason"] == "halted"


# ================================================================================================ stale sessions
def test_a_result_that_returns_after_cancel_is_dropped_as_stale():
    with Rig("never", server={"close_after": None, "latency_s": 0.4}) as r:
        assert r.healthy().ok
        job, h = r.job()

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 3.0
            while r.srv.calls < 3 and time.monotonic() < t_end:       # warm-up + two session inferences
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)                                  # the third one is in flight now
            h.cancel("correction")
            return await task

        out = asyncio.run(main())
        s = r.exe.last_session
        r.exe.last_client.join(2.0)
        assert out.status == "cancelled"
        assert s.dropped["stale_session"] == 1 and s.inferences == s.chunks_sent + 1
        assert _no_chunk_after(r.body, s.id, s.t_ack) == []


def test_the_body_drops_stale_sessions_and_the_executor_stops():
    with Rig("never", server={"close_after": None}) as r:
        assert r.healthy().ok
        job, h = r.job()

        async def main():
            task = asyncio.ensure_future(r.exe.run(job, h))
            t_end = time.monotonic() + 3.0
            while len(r.body.chunk_times("man-1")) < 2 and time.monotonic() < t_end:
                await asyncio.sleep(0.01)
            r.body.forget_session()                                   # e.g. wl-body restarted
            return await task

        out = asyncio.run(main())
        assert out.status == "failed" and out.reason == "controller_unavailable", out.detail
        assert "stale_session" in out.detail and out.data["chunks_dropped"].get("body:stale_session") == 1
    # the fence itself, on the reference body: an older generation or another session id is refused
    body = FakeArmBody()
    try:
        base = dict(stream="s", session_id="s", generation=5, control_epoch=1, mode="chunk")
        assert body.arm(base)["state"] == "accepted"
        assert body.arm({**base, "generation": 4, "keepalive": True})["error"] == "stale_session"
        assert body.arm({**base, "session_id": "s-old", "keepalive": True})["error"] == "stale_session"
        assert body.arm({**base, "end": True, "hold_on_end": "stand"})["ok"]
        assert body.arm({**base, "keepalive": True})["error"] == "stale_session"   # an ended session stays ended
    finally:
        body.close()


# ================================================================================================ bad chunks
def test_nan_chunks_are_never_sent_and_end_in_policy_stall():
    with Rig("never", server={"mode": "nan"}, cfg={"stall_fail_s": 1.0}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "policy_stall", out.detail
        assert out.data["chunks_sent"] == 0 and out.data["chunks_dropped"]["nan"] >= 2
        assert r.body.messages("chunk", "man-1") == [] and out.data["keepalives"] >= 1
        assert r.body.messages("end", "man-1")[-1]["reply"]["ok"]      # the keepalives kept the session alive


def test_out_of_bounds_chunks_end_in_policy_out_of_bounds():
    with Rig("never", server={"mode": "oob"}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "policy_out_of_bounds", out.detail
        assert out.data["clamped_frac"] >= 0.2 and out.data["clamped_frac_source"] == "body"


# ================================================================================================ stance, view, camera
def test_stance_and_view_checks_fail_without_touching_the_body():
    with Rig("never", world={"speed": 0.3}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "base_moving" and out.phase == "stance_check"
        assert r.body.log == []
    with Rig("never", world={"ego_px": 50.0}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.status == "failed" and out.reason == "not_in_ego_view" and out.phase == "view_check"
        assert out.data["view_check"]["method"] == "instance_id_segmentation_fast" and r.body.log == []


def test_view_check_skipped_is_labelled_and_the_camera_is_leased():
    with Rig("never", server={"close_after": None}, world={"enable_camera": True},
             cfg={"max_duration_s": 1.0}) as r:
        assert r.healthy().ok
        out = r.run()
        assert out.data["view_check"]["ran"] is False
        assert any("view check skipped" in n and "INTERIM" in n for n in out.data["notes"])
        on = [c for c in r.world.cam_calls if c[1]]
        off = [c for c in r.world.cam_calls if not c[1]]
        assert len(on) >= 2 and len(off) == 1 and off[0][4] > on[-1][4]         # enabled, renewed, released
        assert {(c[0], c[2], c[3]) for c in r.world.cam_calls} == {("ego_view", "man-1", 10.0)}


# ================================================================================================ registry, profile
def test_registry_backend_and_the_experimental_skills():
    from groot.obs import ARENA_PROMPT
    from robot.profile import load_profile

    assert EXECUTOR_OF_BACKEND["groot"] == "groot_arms" and DEPRECATED_EXECUTORS == {"groot_sonic": "groot_arms"}
    assert BACKEND_ORDER["full"] == ("groot", "sonic_arm_script", "kinematic_attach")
    for sid in (APPLE, ANY):
        s = SKILLS[sid]
        assert s.backend == "groot" and s.executor == "groot_arms" and s.label == "experimental"
        assert s.status == "available" and s.embodiment_tag == "new_embodiment" and s.arms == ("left",)
        assert s.checkpoint.startswith("nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace@")
        assert s.policy_endpoint == "tcp://127.0.0.1:5550" and s.success["kind"] == "gt_lifted"
    assert SKILLS[APPLE].prompt_template == ARENA_PROMPT                 # the checkpoint's trained instruction
    assert SKILLS["groot.pick.bottle.cloudwalk.v0"].status == "planned"  # retired with groot_sonic
    reg = StaticSkillRegistry(SKILLS.values(), vocab=["apple", "alarm_clock"], backend_order=BACKEND_ORDER["full"])
    assert reg.select("pick", "apple", "left").skill_id == APPLE
    assert reg.select("pick", "alarm_clock", "left").skill_id == ANY
    assert reg.select("pick", "alarm_clock", "right").backend == "sonic_arm_script"   # the checkpoint is left-handed
    exe = GrootArmExecutor(FakeWorld(), arm=None, sensors=None, helpers=_groot_helpers())
    job = ManipJob("pick", "alarm_clock_1", "left", ANY)
    job.object_type = "alarm_clock"
    assert exe._prompt(SKILLS[ANY], job) == "move the alarm clock to the plate"
    prof = load_profile("full")
    assert prof.manip_executors == ("groot_arms", "sonic_arm_script", "kinematic_attach")
    assert prof.manip_policy == "groot_then_script"
    assert GrootArmsConfig.from_dict(prof.raw["groot_arms"]).endpoint == "tcp://127.0.0.1:5550"


def test_create_from_the_registry_context(monkeypatch):
    from robot.profile import load_profile

    class Ctx:
        name, world, body, clock, gate, events = "groot_arms", FakeWorld(), object(), None, HaltGate(), None
        profile, port_offset, extras = load_profile("full"), 100, {}

    monkeypatch.setenv("WL_GROOT_ENDPOINT", "tcp://127.0.0.1:15550")
    exe = create(Ctx())
    try:
        assert exe.backend == "groot" and exe.name == "groot_arms"
        assert exe.cfg.endpoint == "tcp://127.0.0.1:15550" and exe.cfg.camera_port == 5566
        h = exe.health()                                  # a body without wl-body's client (lite): honest, down
        assert not h.ok and "not wired" in h.detail and "arm" in h.detail
    finally:
        exe.close()


def test_groot_sonic_is_retired():
    from services.executors.groot_sonic import GrootSonicExecutor

    ex = GrootSonicExecutor()
    h = ex.health()
    assert not h.ok and "retired" in h.detail and "groot_arms" in h.detail
    out = asyncio.run(ex.run(ManipJob("pick", "x", "left", "s"), None))
    assert out.status == "failed" and out.reason == "policy_unavailable"


# ================================================================================================ reference body
def test_fake_arm_body_plays_by_time_cross_fades_and_clamps():
    body = FakeArmBody()
    evs: list[dict] = []
    body.subscribe(evs.append)
    try:
        from groot import joint_order as jo
        stand = [jo.STAND_Q29[i] for i in jo.SONIC_WIRE_FROM_MUJOCO]
        base = dict(stream="t", session_id="t", generation=1, control_epoch=0, mode="chunk")
        assert body.arm(base)["state"] == "accepted"
        t0 = time.monotonic()
        rows = np.tile(stand, (40, 1))
        ch = {"seq": 1, "t0_mono": t0, "dt": 0.02, "order": "wire", "upper_body": rows.tolist(),
              "left_hand": [[0.0] * 7] * 40, "right_hand": [[0.0] * 7] * 40}
        assert body.arm({**base, "chunk": ch})["ok"]
        assert body.arm({**base, "chunk": ch})["data"]["dropped"] == "out_of_order"
        bad = dict(ch, seq=2, upper_body=rows[:, :16].tolist())
        assert body.arm({**base, "chunk": bad})["error"] == "bad_chunk"
        oob = rows.copy()
        oob[:, [i for i, n in enumerate(jo.SONIC_UPPER_JOINTS) if n not in jo.WAIST_JOINTS]] = 3.2
        assert body.arm({**base, "chunk": dict(ch, seq=3, t0_mono=time.monotonic(), upper_body=oob.tolist())})["ok"]
        time.sleep(1.2)                                                   # past the end of the chunk
        prog = [e["data"] for e in evs if e["state"] == "progress"]
        assert prog[-1]["chunk_seq"] == 3 and prog[-1]["cross_fades"] == 1
        assert prog[-1]["stall_s"] > 0.2 and prog[-1]["k"] == 39          # holds the last row, counts the stall
        assert prog[-1]["clamped_frac"] == pytest.approx(0.5, abs=0.01)  # 14 arm values of 28 clamped
        late = dict(ch, seq=4, t0_mono=time.monotonic() - 1.0)
        assert body.arm({**base, "chunk": late})["data"]["dropped"] == "expired"
    finally:
        body.close()
