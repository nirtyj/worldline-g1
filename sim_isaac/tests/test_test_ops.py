"""sim_isaac/test_ops.py without Isaac: argument parsing, the throttle arithmetic on RtPacer's schedule, expiries, the
push wrench on a fake articulation view, and the ops' replies (TEST-ONLY P1 ops behind --test-ops)."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sim_isaac import test_ops as T
from sim_isaac.rt_pacer import RtPacer
from sim_isaac.wire import OpError


def test_push_direction_robot_frame_world_vector_and_degrees():
    yaw = math.radians(30)
    left = T.push_direction("left", yaw)
    assert left == pytest.approx([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)])
    assert T.push_direction("forward", 0.0) == pytest.approx([1.0, 0.0])
    assert T.push_direction([0.0, -2.0], 1.0) == pytest.approx([0.0, -1.0])       # world, normalised, yaw ignored
    assert T.push_direction(90, 0.3) == pytest.approx([0.0, 1.0], abs=1e-12)
    assert T.push_direction("180", 0.0) == pytest.approx([-1.0, 0.0], abs=1e-12)
    for bad in ("sideways", [0, 0], None):
        with pytest.raises(OpError):
            T.push_direction(bad, 0.0)


def test_parse_push_defaults_are_plans_g7_push_and_limits_hold():
    p = T.parse_push({}, 0.0)
    assert p["force_n"] == 250.0 and p["duration_s"] == 0.5 and p["impulse_ns"] == 125.0
    assert p["force_w"] == pytest.approx([0.0, 250.0, 0.0], abs=1e-9)                 # left of a robot facing +x
    with pytest.raises(OpError):
        T.parse_push({"force_n": 5000}, 0.0)
    with pytest.raises(OpError):
        T.parse_push({"duration_s": 10}, 0.0)
    with pytest.raises(OpError):
        T.parse_push({"force_n": "a lot"}, 0.0)


def test_throttle_holds_the_pacer_at_the_target_rtf():
    """RtPacer.wait with the anchor moved by throttle_extra_s every step: sim/wall settles at the target."""
    dt = 0.005
    p = RtPacer(dt, enabled=True)
    p.start()
    extra = T.throttle_extra_s(dt, 0.8)
    assert extra == pytest.approx(dt * 0.25)
    t0 = time.perf_counter()
    for _ in range(100):
        p.step_done()
        p.sched_anchor_wall += extra
        p.wait()
    wall = time.perf_counter() - t0
    assert 0.6 <= 100 * dt / wall <= 0.83                   # never faster than the target; a loaded host oversleeps
    assert p.overruns == 0
    assert T.throttle_extra_s(dt, 1.0) == 0.0


def test_parse_throttle_and_box():
    assert T.parse_throttle({"target": 0.9}) == {"target": 0.9, "duration_s": 120.0}
    assert T.parse_throttle({"target": 1.0}) is None and T.parse_throttle({"off": True}) is None
    with pytest.raises(OpError):
        T.parse_throttle({"target": 0.2})
    size = T.parse_size("0.3,1.6,1.2")
    assert size == (0.3, 1.6, 1.2)
    b = T.parse_box({"pose": {"x": 1, "y": 2, "yaw": 0.5}}, size)
    assert (b["x"], b["y"], b["yaw"], b["ttl_s"]) == (1.0, 2.0, 0.5, 600.0)
    assert T.parse_box({"x": 1, "y": 2, "size": [0.3, 1.6, 1.2]}, size)["yaw"] == 0.0
    with pytest.raises(OpError):
        T.parse_box({"pose": {"x": 1, "y": 2}, "size": [1, 1, 1]}, size)             # no run-time resizing
    with pytest.raises(OpError):
        T.parse_box({"pose": {"y": 2}}, size)
    pose = T.box_root_pose(1.0, 2.0, math.pi / 2, 0.1, size)
    assert pose[2] == pytest.approx(0.1 + 0.6) and pose[3] == pytest.approx(math.cos(math.pi / 4))


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def zeros_like(a):
        return np.zeros_like(a)

    @staticmethod
    def tensor(v, dtype=None, device=None):
        return np.asarray(v, dtype=np.float32)


class FakeView:
    def __init__(self):
        self.calls = []

    def apply_forces_and_torques_at_position(self, force_data, torque_data, position_data, indices, is_global):
        self.calls.append((force_data.copy(), is_global))


class FakeBox:
    def __init__(self):
        self.poses = []

    def write_root_pose_to_sim(self, pose):
        self.poses.append(np.asarray(pose).reshape(-1).tolist())


def fake_app():
    events = []
    pacer = RtPacer(0.005, enabled=False)
    pacer.start()
    app = SimpleNamespace(torch=FakeTorch, force_buf=np.zeros((5, 3)), torque_buf=np.zeros((5, 3)), pelvis_id=0,
                          view=FakeView(), view_idx=None, pacer=pacer, dt=0.005, floor_z=0.0,
                          sim=SimpleNamespace(device="cpu"), last_pose={"yaw": 0.0},
                          _event=lambda ev, **kw: events.append({"event": ev, **kw}))
    return app, events


def test_ops_on_a_fake_app_push_throttle_box_and_status():
    app, events = fake_app()
    ops = T.TestOps(app, "0.3,1.6,1.2", log=lambda *_: None)
    ops.box_obj, ops.park = FakeBox(), T.box_root_pose(60, 60, 0.0, 0.0, ops.box_size)
    assert ops.op_test_ops_status({})["active"] is False

    # push: a world-frame force on the pelvis every step until duration_s of sim time has passed
    rep = ops.op_push_robot({"force_n": 200, "dir": "left", "duration_s": 0.02})
    assert rep["push"]["force_w"] == pytest.approx([0.0, 200.0, 0.0], abs=1e-9)
    for _ in range(6):
        ops.pre_step({})
        app.pacer.step_done()
    assert len(app.view.calls) == 4 and all(c[1] for c in app.view.calls)            # 0.02 s / 0.005 s, global
    assert app.view.calls[0][0][0] == pytest.approx([0.0, 200.0, 0.0])
    assert ops.push is None and events[-1]["op"] == "push_robot" and events[-1]["state"] == "ended"

    # throttle: the anchor moves by the extra per step; it ends by request (or after duration_s)
    ops.op_rtf_throttle({"target": 0.9, "duration_s": 30})
    a0 = app.pacer.sched_anchor_wall
    ops.before_wait()
    assert app.pacer.sched_anchor_wall - a0 == pytest.approx(0.005 * (1 / 0.9 - 1))
    assert ops.op_test_ops_status({})["throttle"]["target"] == 0.9
    ops.throttle["until_wall"] = time.perf_counter() - 1                               # its duration is over
    ops.before_wait()
    assert ops.throttle is None and events[-1]["state"] == "ended"
    assert ops.op_rtf_throttle({"off": True})["throttle"] is None

    # box: written standing on the floor, parked again by clear_box or its ttl
    rep = ops.op_spawn_box({"pose": {"x": 2.0, "y": 3.0, "yaw": 0.0}, "ttl_s": 5})
    assert ops.box_obj.poses[-1][:3] == pytest.approx([2.0, 3.0, 0.6])
    assert ops.op_test_ops_status({})["active"] is True
    assert ops.op_clear_box({})["cleared"] is True and ops.box_obj.poses[-1][:2] == [60, 60]
    ops.op_spawn_box({"pose": {"x": 2.0, "y": 3.0}, "ttl_s": 5})
    ops.box["until_wall"] = time.perf_counter() - 1
    ops.before_wait()
    assert ops.box is None and events[-1]["event"] == "test_op" and events[-1]["by"] == "ttl"
    st = ops.op_test_ops_status({})
    assert st["active"] is False and st["counts"]["spawn_box"] == 2 and st["label"] == "test-only"


def test_spawn_box_without_the_box_is_refused():
    app, _ = fake_app()
    ops = T.TestOps(app, (0.3, 1.6, 1.2), log=lambda *_: None)
    with pytest.raises(OpError):
        ops.op_spawn_box({"pose": {"x": 0, "y": 0}})
