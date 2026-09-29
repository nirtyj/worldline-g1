"""ObservationService: glance records, the interim turn-in-place scan, wait_and_observe outcomes."""

import asyncio
import math

import pytest

from api.observation import is_glance_id
from api.results import validate_envelope
from tests.services.conftest import Stack, run
from world import coords


def test_glance_record_is_cheap_and_bumps_on_change():
    s = Stack()
    k = s.put_at("bedroom_bed_1b")

    async def main():
        g1 = s.robot.observation_id()
        assert is_glance_id(g1)
        assert s.robot.observation_id() == g1                 # nothing changed
        s.world.set_robot_pose(k.x, k.y, k.yaw + math.pi)     # look away
        await s.clock.sleep(0.2)
        g2 = s.robot.observation_id()
        assert g2 != g1
        assert s.robot.obs.glance() is not None and s.robot.obs.glance()["views"] == []
    run(main())


def test_scan_turns_in_place_and_reports_views():
    s = Stack()
    k = s.put_at("bedroom_bed_1b")
    s.robot.nav._last_at = "bedroom_bed_1b"

    async def main():
        r = await s.run("observe", {"mode": "scan"})
        assert validate_envelope(r) == []
        d = r.data
        assert d["mode"] == "scan" and d["scan_executor"] == "turn_in_place" and "INTERIM" in d["scan_note"]
        assert len(d["views"]) == 3 and d["method"] == "gt-geometric" and d["source"] == "lite-gt"
        yaws = sorted(v["yaw"] for v in d["views"])
        want = sorted(coords.yaw_map_deg(k.yaw + math.radians(dy)) for dy in (-35, 0, 35))
        for a, b in zip(yaws, want):
            assert abs((a - b + 180) % 360 - 180) < 7.0
        assert "alarm_clock_1" in {it["id"] for it in d["surfaces"].get("bedroom_bed_1b", [])}
        assert r.observation_id == r.execution_id                 # a real observation stamps itself
        # back facing the surface
        assert abs(coords.ang_diff(s.world.robot_pose().yaw, k.yaw)) < math.radians(7)
        assert "alarm_clock_1" in s.robot.obs.seen_here("bedroom_bed_1b")
        # the turns went through the body
        assert sum(1 for e in s.body.log if e["op"] == "turn_to") >= 3
    run(main())


def test_virtual_scan_executor_does_not_move():
    s = Stack(overrides={"scan_executor": "virtual"})
    k = s.put_at("bedroom_bed_1b")

    async def main():
        r = await s.run("observe", {"mode": "scan"})
        assert r.data["scan_executor"].startswith("virtual") and len(r.data["views"]) == 3
        assert s.world.robot_pose().yaw == pytest.approx(k.yaw)
        assert not s.body.log
    run(main())


def test_glance_observe_has_no_views():
    s = Stack()
    s.put_at("bedroom_bed_1b")

    async def main():
        r = await s.run("observe", {"mode": "glance"})
        assert r.data["views"] == [] and r.data["mode"] == "glance"
        assert r.data["hands"] == {"left": None, "right": None}
    run(main())


def test_wait_and_observe_unchanged_then_changed():
    s = Stack()
    s.put_at("bedroom_bed_1b")
    s.robot.nav._last_at = "bedroom_bed_1b"

    async def main():
        first = await s.run("wait_and_observe", {"timeout_s": 0})
        assert first.data["observed"] == "scan"               # no lease, at a keypoint, never scanned here
        second = await s.run("wait_and_observe", {"timeout_s": 0})
        assert second.status == "succeeded" and second.data["status"] == "unchanged"
        assert second.data["observed"] == "glance"            # the last scan here is < 5 s old
        assert second.summary == "looked; nothing new"
        assert validate_envelope(second) == []
        # someone puts the alarm clock on the floor in view: changed
        c = s.world.object("alarm_clock_1").pos
        await s.clock.sleep(6.0)
        s.world.move_object("alarm_clock_1", (c[0], c[1] + 0.35, 0.05))
        third = await s.run("wait_and_observe", {"timeout_s": 0})
        assert third.status == "succeeded" and third.data["status"] == "changed"
        assert "alarm_clock_1" in third.data["summary"]
    run(main())


def test_wait_and_observe_times_out_and_wakes():
    s = Stack()
    s.put_at("bedroom_bed_1b")

    async def main():
        await s.run("wait_and_observe", {"timeout_s": 0})
        r = await s.run("wait_and_observe", {"timeout_s": 2.0})
        assert r.status == "timed_out" and r.data["status"] == "timed_out"
        wake = asyncio.Event()
        wake.why = "the user spoke"
        h = s.robot.obs.start_wait(s.ex("wait_and_observe", {"timeout_s": 30.0}), wake=wake)
        await s.clock.sleep(1.0)
        wake.set()
        r2 = await h.result()
        assert r2.status == "succeeded" and r2.data["status"] == "changed" and r2.data["summary"] == "the user spoke"
        h3 = s.robot.start(s.ex("wait_and_observe", {"timeout_s": 30.0}))
        await s.clock.sleep(1.0)
        h3.cancel("correction")
        r3 = await h3.result()
        assert r3.status == "cancelled"
    run(main())


def test_wait_during_a_body_execution_only_glances():
    s = Stack()
    s.put_at("start")

    async def main():
        h = s.robot.start(s.ex("navigate", {"location": "kitchen"}))
        await s.clock.sleep(1.0)
        r = await s.run("wait_and_observe", {"timeout_s": 0})
        assert r.data["observed"] == "glance"                 # never requests the body, never body_busy
        h.cancel("x")
        await h.result()
    run(main())
