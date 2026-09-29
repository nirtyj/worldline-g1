"""B.6 op `approach` over the wire (fakes): short strafing repositions judged on the fake P1's ground truth.

The fake deploy is not SONIC (first-order velocity response, tau 0.35 s, 92 % of the commanded speed, a +3 deg yaw
bias), so this checks the pulse logic, the learnt glide, the turn-first path, the rejections and the fences; the
precision on SONIC is measured live (tools/approach_test.py, docs/contracts/m1.md §3.13)."""

import math
import time

import pytest

from body.wire import wrap

from .test_integration_fakes import Stack


@pytest.fixture
def stack(port_offset, tmp_path):
    s = Stack(port_offset, tmp_path)
    yield s
    s.close()


def test_approach_short_repositions(stack):
    bc, p1, dep = stack.bc, stack.p1, stack.dep
    stack.stand()
    assert bc.turn_to(0.0).ok
    errs = []
    # (dx, dy in the robot frame at the start, yaw change deg): forward, strafe left/right, back, diagonal, a turn
    for dx, dy, dyaw in ((0.30, 0.0, 0), (0.0, 0.25, 0), (0.0, -0.35, 0), (-0.15, 0.0, 0), (0.2, 0.2, 0),
                         (0.1, 0.0, 30)):
        x0, y0, yaw0 = p1.x, p1.y, p1.yaw
        c, s = math.cos(yaw0), math.sin(yaw0)
        gx, gy = x0 + c * dx - s * dy, y0 + s * dx + c * dy
        gyaw = wrap(yaw0 + math.radians(dyaw))
        h = bc.approach(gx, gy, yaw=gyaw)
        assert h.ok, h.result
        e = math.hypot(p1.x - gx, p1.y - gy)
        ey = abs(math.degrees(wrap(p1.yaw - gyaw)))
        errs.append((e, ey))
        assert e <= 0.05 + 0.01 and ey <= 5.0 + 0.5, (dx, dy, dyaw, e, ey, h.result)   # GT, not the body's word
        assert h.result["pos_err"] == pytest.approx(e, abs=0.02)
        assert h.result["moves"] and h.result["pose_source"] == "gt.pose"
        if dyaw:
            assert h.result["turns"] >= 1
    st = bc.status()
    assert set(st["approach_est"]["t_stop"]) == {"fwd", "back", "left", "right"}
    assert dep.stats["stop"] == 0 and dep.stats["planner_timeouts"] == 0


def test_approach_rejections_and_fences(stack):
    bc, p1 = stack.bc, stack.p1
    h = bc.approach(p1.x + 0.2, p1.y)
    assert h.state == "failed" and h.reason == "not_standing"
    stack.stand()
    h = bc.approach(p1.x + 1.0, p1.y)
    assert h.state == "failed" and h.reason == "too_far" and h.result["distance_m"] == pytest.approx(1.0, abs=0.02)
    h = bc.approach(p1.x + 0.2, p1.y, tol=(0.05, 5.0), extra_bad=None)
    assert h.ok or h.reason == "final_error"
    h = bc.request("approach", {"x": p1.x, "tol": 3})
    assert not h["ok"] and h["error"] == "bad_args"
    # fenced like every motion: halted while latched
    assert bc.halt(4, timeout_s=0.5)["acked"]
    h = bc.approach(p1.x + 0.2, p1.y, control_epoch=4)
    assert h.state == "failed" and h.reason == "halted"
    assert bc.resume(5)["ok"]
    # a halt mid-approach cancels it
    hw = bc.approach(p1.x + 0.4, p1.y, wait=False, max_dist=0.6)
    time.sleep(0.4)
    assert bc.halt(6, timeout_s=0.5)["acked"]
    hw.wait(3)
    assert hw.state == "canceled" and hw.result["reason"] == "halt"
    assert bc.resume(7)["ok"]
