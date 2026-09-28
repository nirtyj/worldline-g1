"""End-to-end: BodyClient -> wl-body -> (fake) deploy -> (fake) P1, all over the real ZMQ contract.

The fakes mirror the deploy's zmq_manager decoding / planner-frame / timeout / stop semantics and P1's REP + gt.pose,
so these tests exercise the wire format, the planner frame conversion, pre-emption and the watchdogs.
Assertions use ground truth read from the fake P1, never the body's own claims.
"""

import math
import threading
import time

import numpy as np
import pytest

from body.client import BodyClient
from body.config import BodyConfig
from body.service import BodyService
from body.wire import wrap
from tools.fake_deploy import FakeDeploy
from tools.fake_p1 import FakeP1, SPAWN


class Stack:
    def __init__(self, off, tmp):
        self.p1 = FakeP1(off, str(tmp / "p1"), log=lambda *_: None).start()
        self.dep = FakeDeploy(off, log=lambda *_: None).start()
        self.svc = BodyService(BodyConfig(port_offset=off), log_dir=str(tmp / "body"), log=None)
        self.th = threading.Thread(target=self.svc.run, daemon=True)
        self.th.start()
        self.bc = BodyClient(port_offset=off).connect(15)
        t0 = time.monotonic()
        while self.svc.pose_sub.latest() is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)

    def stand(self):
        h = self.bc.stand(release_band=True, settle_s=0.5, verify_s=1.0)
        assert h.ok, h.result
        return h

    def gt(self):
        return self.p1.x, self.p1.y, self.p1.yaw

    def close(self):
        self.bc.close()
        self.svc.stop()
        self.th.join(5)
        self.dep.stop()
        self.p1.stop()


@pytest.fixture
def stack(port_offset, tmp_path):
    s = Stack(port_offset, tmp_path)
    yield s
    s.close()


def test_main_flow(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    # motion before stand is rejected, reported as failed, and nothing moves
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.state == "failed" and h.reason == "not_standing"

    h = stack.stand()
    assert p1.band_on is False and not p1.collapsed
    st = bc.status()
    assert st["in_control"] and st["mux"]["frame"]["source"] == "g1_debug"
    assert st["mux"]["frame"]["theta0_deg"] == pytest.approx(math.degrees(SPAWN[2]), abs=0.5)

    # turn in place to face +x (world)
    h = bc.turn_to(0.0)
    assert h.ok, h.result
    x, y, yaw = stack.gt()
    assert abs(math.degrees(wrap(yaw))) <= 15 and math.hypot(x - SPAWN[0], y - SPAWN[1]) < 0.3

    # walk forward: from (1.8, 1.4) along +x through the kitchen door region
    x0, y0, _ = stack.gt()
    h = bc.walk(vx=0.5, duration_s=4.5)
    assert h.ok, h.result
    x1, y1, yaw1 = stack.gt()
    assert x1 - x0 >= 1.8 and abs(y1 - y0) < 0.3 and abs(math.degrees(wrap(yaw1))) < 15

    # strafe left (+y world when facing +x)
    h = bc.walk(vy=0.3, duration_s=2.0)
    assert h.ok, h.result
    x2, y2, _ = stack.gt()
    assert y2 - y1 >= 0.35 and abs(x2 - x1) < 0.25

    # go_to 3 waypoints spanning kitchen -> living room -> dining room with furniture in between
    wps = [(5.5, 1.5, None), (8.0, 4.5, math.pi / 2), (3.3, 4.5, math.pi)]
    for (gx, gy, gyaw) in wps:
        h = bc.go_to(gx, gy, yaw=gyaw, timeout_s=120)
        assert h.ok, h.result
        x, y, yaw = stack.gt()
        assert math.hypot(x - gx, y - gy) <= 0.3                    # GT, not the body's word
        if gyaw is not None:
            assert abs(math.degrees(wrap(yaw - gyaw))) <= 15
        assert h.result["pos_err"] == pytest.approx(math.hypot(x - gx, y - gy), abs=0.1)

    # stop mid-walk: the walk ends 'canceled', the stop succeeds within 1.5 s and the robot is at rest
    bc.turn_to(0.0)
    hw = bc.walk(vx=0.5, duration_s=10.0, wait=False)
    time.sleep(1.5)
    assert math.hypot(p1.vx, p1.vy) > 0.3
    hs = bc.stop()
    hw.wait(5)
    assert hw.state == "canceled" and hw.result["reason"] == "stop"
    assert hs.ok and hs.result["stop_time_s"] <= 1.5
    time.sleep(0.5)
    assert math.hypot(p1.vx, p1.vy) < 0.05 and not p1.collapsed

    # pre-emption: a new command cancels the running one
    hg = bc.go_to(8.0, 1.0, wait=False)
    time.sleep(1.0)
    ht = bc.turn_to(math.pi / 2)
    hg.wait(5)
    assert hg.state == "canceled" and hg.result["reason"] == "preempted" and hg.result["by"] == ht.id
    assert ht.ok

    # keepalive and safety invariants seen by the deploy
    assert dep.planner_rate() >= 10.0
    assert dep.stats["planner_timeouts"] == 0
    assert dep.stats["stop"] == 0 and dep.stats["bad_planner"] == 0 and dep.stats["bad_command"] == 0
    assert dep.state == "CONTROL"
    assert p1.root_writes == 0


def test_watchdogs_fall_and_shutdown(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    stack.stand()
    bc.turn_to(0.0)
    # P1 stalls -> the running op fails with pose_stale and the mux holds IDLE
    hw = bc.walk(vx=0.4, duration_s=8.0, wait=False)
    time.sleep(1.0)
    p1.pose_paused = True
    hw.wait(5)
    assert hw.state == "failed" and hw.result["reason"] == "pose_stale"
    time.sleep(0.6)
    assert dep.mode == 0 and math.hypot(p1.vx, p1.vy) < 0.1
    p1.pose_paused = False
    time.sleep(0.5)

    # a fall is detected from GT, the op fails, and the fault latches
    hw = bc.walk(vx=0.4, duration_s=8.0, wait=False)
    time.sleep(0.8)
    p1.collapsed = True
    hw.wait(5)
    assert hw.state == "failed" and hw.result["reason"] == "fallen"
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.state == "failed" and h.reason.startswith("fault:")
    # episode reset (P1) + clear_fault -> the robot is usable again
    p1._op("reset_robot", {"x": 2.0, "y": 1.5, "yaw": 0.0})
    assert bc.request("clear_fault")["ok"]
    time.sleep(0.3)
    h = bc.walk(vx=0.4, duration_s=1.0)
    assert h.ok, h.result

    # shutdown_control needs confirmation; with the band on the deploy exits and nothing collapses
    assert not bc.request("shutdown_control")["ok"]
    p1._op("band", {"on": True})
    rep = bc.request("shutdown_control", {"confirm": True})
    assert rep["ok"]
    time.sleep(0.5)
    assert dep.stats["stop"] >= 1 and dep.state == "STOPPED" and not p1.collapsed


def test_stuck_detection_replans(stack):
    bc, p1 = stack.bc, stack.p1
    stack.stand()
    # an obstacle the map does not know about, right in the doorway path kitchen -> living room
    p1._op("spawn_obstacle", {"x": 4.0, "y": 1.5, "r": 0.45})
    h = bc.go_to(5.5, 1.5, timeout_s=120)
    x, y, _ = stack.gt()
    if h.ok:
        assert math.hypot(x - 5.5, y - 1.5) <= 0.3
        assert h.result["replans"] >= 1 and h.result["stuck_events"]
    else:
        assert h.reason and ("stuck" in h.reason or "no_path" in h.reason or "timeout" in h.reason)
        assert math.hypot(x - 5.5, y - 1.5) > 0.3 or h.reason != "final_error"
