"""Unit tests for body/velocity.py (op velocity / Nav2 cmd_vel -> SONIC planner commands) and the go_to backend
selection in body/nav2_backend.py. No sockets."""

import math
import time

from body.config import BodyConfig
from body.nav2_backend import AStarGoToMotion, Nav2GoToMotion, select_motion
from body.path_follower import GoToMotion
from body.velocity import VelocityCommander
from body.wire import LocomotionMode, Pose


class FakeMux:
    def __init__(self):
        self.cmd = None
        self.last_facing_w = None

    def set(self, cmd):
        self.cmd = cmd
        if cmd.facing_w is not None:
            self.last_facing_w = cmd.facing_w

    def hold(self, facing_w=None, owner=""):
        from body.sonic_mux import PlannerCmd
        self.set(PlannerCmd(LocomotionMode.IDLE, (0.0, 0.0), facing_w, -1.0, -1.0, owner))


def pose(x=0.0, y=0.0, yaw=0.0):
    return Pose({"base_pos": [x, y, 0.78], "yaw": yaw}, time.monotonic())


def test_forward_in_world_frame_and_clamps():
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    assert vc.update({"vx": 0.5, "vy": 0.0, "wz": 0.0})
    now = time.monotonic()
    assert vc.apply(pose(yaw=math.pi / 2), now) == "walk"
    c = mux.cmd
    assert c.mode == LocomotionMode.SLOW_WALK
    assert abs(c.move_w[0]) < 1e-9 and abs(c.move_w[1] - 0.5) < 1e-9        # body x = world +y at yaw 90 deg
    assert abs(c.speed - 0.5) < 1e-9
    vc.update({"vx": 0.1, "vy": 0.0, "wz": 0.0})                             # below SLOW_WALK floor -> 0.2
    vc.apply(pose(yaw=math.pi / 2), now + 0.02)
    assert abs(mux.cmd.speed - cfg.v_min) < 1e-9
    vc.update({"vx": 0.0, "vy": 0.7, "wz": 0.0})                             # strafe capped at 0.4
    vc.apply(pose(yaw=0.0), now + 0.04)
    assert abs(mux.cmd.speed - cfg.v_strafe_max) < 1e-9
    assert mux.cmd.move_w[1] > 0.69                                          # body +y = world +y at yaw 0


def test_deadband_turn_in_place_and_facing_integration():
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    vc.update({"vx": 0.02, "vy": 0.0, "wz": 0.5})
    t = time.monotonic()
    vc.apply(pose(yaw=0.0), t)
    for k in range(1, 11):                                                    # 0.2 s at 50 Hz (simulated clock)
        vc.update({"vx": 0.02, "vy": 0.0, "wz": 0.5})
        vc.t_cmd = t + 0.02 * k
        st = vc.apply(pose(yaw=0.0), t + 0.02 * k)
    assert st == "turn" and mux.cmd.mode == LocomotionMode.IDLE
    assert abs(mux.cmd.facing_w - 0.1) < 1e-6                                 # 0.5 rad/s * 0.2 s
    for k in range(11, 200):                                                  # anti-windup: lead <= 25 deg
        vc.update({"vx": 0.0, "vy": 0.0, "wz": 0.5})
        vc.t_cmd = t + 0.02 * k
        vc.apply(pose(yaw=0.0), t + 0.02 * k)
    assert abs(mux.cmd.facing_w - math.radians(cfg.facing_lead_max_deg)) < 1e-6


def test_watchdog_goes_idle_and_stale_messages_dropped():
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    assert vc.update({"vx": 0.4, "t_wall": time.time()})
    t = time.monotonic()
    assert vc.apply(pose(yaw=0.3), t) == "walk"
    assert vc.apply(pose(yaw=0.35), t + 0.31) == "watchdog"
    assert mux.cmd.mode == LocomotionMode.IDLE and abs(mux.cmd.facing_w - 0.35) < 1e-9
    assert vc.stats["watchdog_trips"] == 1
    assert not vc.update({"vx": 0.4, "t_wall": time.time() - 1.0})          # stale -> dropped
    assert vc.stats["stale_dropped"] == 1


class _Svc:
    def __init__(self, backend, ready):
        self.cfg = BodyConfig()
        self.cfg.nav_backend = backend
        self._ready = ready
        self.logs = []

    def nav2_link(self):
        svc = self

        class L:
            def ready(self, timeout_s=0.5):
                return svc._ready, "" if svc._ready else "bridge_unreachable"
        return L()

    def log(self, m):
        self.logs.append(m)


def test_backend_selection():
    assert select_motion(_Svc("nav2", True), "go_to", {}, GoToMotion)[0] is Nav2GoToMotion
    cls, info = select_motion(_Svc("nav2", False), "go_to", {}, GoToMotion)
    assert cls is AStarGoToMotion and info["backend_fallback"] == "bridge_unreachable"
    assert select_motion(_Svc("astar", True), "go_to", {}, GoToMotion)[0] is AStarGoToMotion
    assert select_motion(_Svc("nav2", True), "go_to", {"backend": "astar"}, GoToMotion)[0] is AStarGoToMotion
    assert select_motion(_Svc("nav2", True), "walk", {}, "W") == ("W", {})
