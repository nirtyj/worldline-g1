import math
import time

import pytest
import zmq

from body.frames import PlannerFrame
from body.sonic_mux import PlannerCmd, SonicMux
from body.wire import LocomotionMode, decode_command, decode_planner, quat_wxyz_from_yaw


def test_frame_from_debug_and_conversion():
    f = PlannerFrame()
    assert not f.known
    f.set_fallback(0.3)
    assert f.source == "gt_at_start" and f.offset == pytest.approx(0.3)
    assert f.update_from_debug(quat_wxyz_from_yaw(1.2), 0.0)
    assert f.source == "g1_debug" and f.offset == pytest.approx(1.2)
    # world +y with theta0 = 1.2 -> planner angle pi/2 - 1.2
    mx, my = f.vec_world_to_planner(0.0, 1.0)
    assert math.atan2(my, mx) == pytest.approx(math.pi / 2 - 1.2)
    fx, fy, _ = f.facing_vec(1.2)
    assert (fx, fy) == pytest.approx((1.0, 0.0))  # facing the start heading == planner +x
    f.set_fallback(0.0)  # a fallback must not override g1_debug
    assert f.offset == pytest.approx(1.2)


def test_frame_bias_converges_sign():
    f = PlannerFrame(bias_ki=2.0)
    f.update_from_debug(quat_wxyz_from_yaw(0.0), 0.0)
    e_track = math.radians(4)
    cmd = 0.5
    for _ in range(400):
        # robot settles at yaw_p + theta0 + e_track, with yaw_p = cmd - offset
        yaw_p = f.yaw_world_to_planner(cmd)
        gt = yaw_p + f.theta0 + e_track
        f.observe_settled(cmd, gt, 0.02)
    assert f.bias == pytest.approx(-e_track, abs=math.radians(0.3))


def _sub(ctx, endpoint):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.connect(endpoint)
    return s


def test_mux_keepalive_rate_start_and_watchdog(port_offset):
    ctx = zmq.Context.instance()
    endpoint = f"tcp://127.0.0.1:{5556 + port_offset}"
    f = PlannerFrame()
    mux = SonicMux(endpoint, f, keepalive_hz=50, stale_s=0.3, log=lambda *_: None, ctx=ctx)
    mux.start()
    sub = _sub(ctx, endpoint)
    try:
        time.sleep(0.3)
        f.set_fallback(0.0)
        mux.start_control()
        mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (0.0, 1.0), math.pi / 2, 0.5))
        msgs = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            if sub.poll(50):
                msgs.append(sub.recv())
        planners = [decode_planner(m) for m in msgs if m.startswith(b"planner")]
        cmds = [decode_command(m) for m in msgs if m.startswith(b"command")]
        assert cmds and all(c["start"] and c["planner"] and not c["stop"] for c in cmds)
        assert len(planners) >= 35  # ~50 Hz for ~1 s (>= 10 Hz required)
        # set() was 0-0.3 s ago -> walking; then the watchdog must fall back to IDLE since nobody refreshed it
        walk = [p for p in planners if p["mode"] == LocomotionMode.SLOW_WALK]
        idle = [p for p in planners if p["mode"] == LocomotionMode.IDLE]
        assert walk and idle and mux.stats["stale_holds"] > 0
        w = walk[0]
        assert w["movement"][:2] == pytest.approx([0.0, 1.0], abs=1e-5)  # theta0=0 -> world == planner
        assert w["speed"] == pytest.approx(0.5)
        assert idle[-1]["facing"][:2] == pytest.approx([0.0, 1.0], abs=1e-5)  # IDLE keeps the last facing
        assert idle[-1]["movement"] == [0.0, 0.0, 0.0]
        assert mux.stats["command_stop_sent"] == 0
    finally:
        sub.close(0)
        mux.close()


def test_mux_deadband_keeps_identical_vectors(port_offset):
    ctx = zmq.Context.instance()
    endpoint = f"tcp://127.0.0.1:{5556 + port_offset}"
    f = PlannerFrame()
    f.set_fallback(0.4)
    mux = SonicMux(endpoint, f, keepalive_hz=50, stale_s=5.0, dir_deadband_deg=2.0, log=lambda *_: None, ctx=ctx)
    mux.start()
    sub = _sub(ctx, endpoint)
    try:
        time.sleep(0.3)
        seen = []
        for k in range(20):
            jitter = math.radians(0.5) * (1 if k % 2 else -1)
            mux.set(PlannerCmd(LocomotionMode.SLOW_WALK, (math.cos(1.0 + jitter), math.sin(1.0 + jitter)), 1.0 + jitter, 0.4))
            time.sleep(0.03)
            while sub.poll(0):
                m = sub.recv()
                if m.startswith(b"planner"):
                    seen.append(m[len(b"planner") + 1280:])
        walk_payloads = {p for p in seen if p[:4] == (1).to_bytes(4, "little")}
        assert len(walk_payloads) == 1  # tiny jitter never changes the bytes -> no forced re-plan in the deploy
    finally:
        sub.close(0)
        mux.close()


def test_mux_refuses_second_binder(port_offset):
    ctx = zmq.Context.instance()
    endpoint = f"tcp://127.0.0.1:{5556 + port_offset}"
    a = SonicMux(endpoint, PlannerFrame(), log=lambda *_: None, ctx=ctx)
    a.bind()
    b = SonicMux(endpoint, PlannerFrame(), log=lambda *_: None, ctx=ctx)
    with pytest.raises(zmq.ZMQError):
        b.bind()
    a.close()
