"""G1Robot (RobotBridge) on lite: the THOR-compatible read surface, dispatch of every tool, halt/resume, events,
and a scripted F1 fetch (bring the alarm clock) that ends with truth on the user surface. Every ToolResult is
validated against the envelope schema and carries an observation id."""

import pytest

from api.results import validate_envelope
from api.services import RobotBridge
from tests.services.conftest import Stack, run


def test_bridge_is_a_robot_bridge_and_thor_shaped():
    s = Stack()
    r = s.robot
    assert isinstance(r, RobotBridge)
    m = r.lookup_keypoints()
    assert m is r.lookup_keypoints()                          # stable object
    assert m["profile"] == "lite" and m["robot"] == "unitree_g1"
    assert m["executors"] == {"navigate": "lite", "manipulate": "lite", "observe": "turn_in_place"}
    assert m["nav_speed_mps"] == r.profile.walk_speed_mps
    assert r.memory() == []
    bs = r.base_state()
    assert set(bs) == {"moving", "at", "between", "xy"} and bs["at"] == "start"
    t = r.telemetry()
    assert set(t) >= {"pose", "velocity", "moving", "arms", "grippers", "active_skills", "health", "body"}
    assert set(t["pose"]) >= {"x", "z", "yaw", "horizon", "at", "between"}
    assert set(t["body"]) >= {"mode", "lease", "upright", "rtf", "carry", "halt_epoch"}
    assert r.gripper("left") == {"closed": False, "width": 0.085, "force": 0.0}
    p = r.perception()
    assert set(p) >= {"objects", "landmarks", "people", "source"}
    caps = r.capabilities()
    assert set(caps) == {"navigation", "manipulation", "observation", "speech", "body"}
    assert all(h.ok for h in caps.values())
    assert r.observation_id().startswith("obs-g")
    assert "lite" in r.profile.stepping_stones


def test_timeouts_per_tool():
    s = Stack()
    r = s.robot
    assert r.timeout_s("manipulate", {"action": "pick", "object_type": "alarm_clock"}) == pytest.approx(18.0)
    assert r.timeout_s("speak", {"text": "one two three"}) == pytest.approx(5.2)
    assert r.timeout_s("check_reachability", {"object_type": "apple"}) == 12.0
    assert 20.0 <= r.timeout_s("navigate", {"location": "bedroom_bed_1b"}) <= 240.0


def test_speak_emits_speech_events_and_cut():
    s = Stack()

    async def main():
        q = s.robot.events()
        r = await s.run("speak", {"text": "On my way to the bedroom."})
        assert r.status == "succeeded" and r.data["status"] == "queued" and r.data["utterance_id"]
        assert validate_envelope(r) == []
        types = []
        while not q.empty():
            types.append(q.get_nowait()["type"])
        assert types[:2] == ["speech_started", "speech_ended"]
        assert s.log.find("speech_started")                   # System 1's robot_said reads the EventLog
        h = s.robot.start(s.ex("speak", {"text": "a long sentence that will be cut off before its end"}))
        await s.clock.sleep(0.5)
        s.robot.speech.cut_all()
        r2 = await h.result()
        assert r2.status == "cancelled" and 0 < r2.data["played"] < 1
    run(main())


def test_halt_is_fast_and_resume_clears():
    s = Stack()

    async def main():
        h = s.robot.start(s.ex("navigate", {"location": "kitchen"}))
        await s.clock.sleep(2.0)
        rec = s.robot.halt()
        assert rec["accepted"] and rec["stopped"] and rec["latency_ms"] < 30 and rec["body_epoch"] == 1
        assert (await h.result()).data["reason"] == "halted"
        assert s.robot.telemetry()["body"]["latched"] is True
        s.robot.resume(3)
        assert s.robot.telemetry()["body"]["latched"] is False
        assert s.robot.active_executions() == []
    run(main())


def test_estop_marks_the_body():
    s = Stack()
    res = s.robot.estop("operator")
    assert res["mode"] == "ESTOP"
    assert s.robot.capabilities()["body"].state == "estop"


def test_scripted_f1_fetch_delivers_on_the_user_surface():
    """F1 on lite (house 38): navigate -> arrival scan -> reachability -> reach_stance -> reachability -> pick ->
    navigate(user) -> place(user) -> verify. Mirrors what the harness does, through RobotBridge.start only."""
    s = Stack()
    results = []

    async def call(tool, args, **kw):
        r = await s.run(tool, args, **kw)
        results.append(r)
        return r

    async def main():
        await call("speak", {"text": "Sure, I'll bring you the alarm clock."})
        nav = await call("navigate", {"location": "bedroom_bed_1b"})
        assert nav.status == "succeeded"
        scan = await call("observe", {"mode": "scan"}, source="harness", tag="auto:arrival")
        assert "alarm_clock_1" in {i["id"] for items in scan.data["surfaces"].values() for i in items}
        reach = await call("check_reachability", {"object_type": "alarm_clock"})
        if reach.data["reason"] == "needs_reposition":
            rep = await call("navigate", {"location": "reach_stance", "anchor": "bedroom_bed_1b",
                                          "stance": reach.data["stance"]})
            assert rep.status == "succeeded"
            reach = await call("check_reachability", {"object_type": "alarm_clock"})
        assert reach.data["reachable"], reach.summary
        pick = await call("manipulate", {"action": "pick", "object_type": "alarm_clock",
                                         "object_id": reach.data["object_id"], "arm": reach.data["preferred_arm"]
                                         if reach.data["preferred_arm"] in ("left", "right") else None})
        assert pick.status == "succeeded"
        glance = await call("observe", {"mode": "glance"}, source="harness", tag="auto:verify pick")
        assert glance.data["hands"][pick.data["arm"]] == "alarm_clock_1"
        home = await call("navigate", {"location": "user"})
        assert home.status == "succeeded"
        place = await call("manipulate", {"action": "place", "object_type": "alarm_clock", "target": "user"})
        assert place.status == "succeeded"
        await call("speak", {"text": "Here it is."})
        truth = s.world.truth()
        assert truth.objects["alarm_clock_1"]["where"] == s.world.map.user_surface
        for r in results:
            assert validate_envelope(r) == [], (r.tool, validate_envelope(r))
            assert r.observation_id, r.tool
            assert r.generation == 1 and r.control_epoch == 0
    run(main())


def test_factory_rejects_real_g1_in_m2a():
    from robot.factory import build
    with pytest.raises(NotImplementedError):
        build("real_g1", "procthor-train-38")
