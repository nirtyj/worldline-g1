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


def _run(vc, t0, k0, k1, cmd, yaw_fn, dt=0.02):
    """Feed `cmd` and apply at 50 Hz (simulated clock) for ticks k0..k1-1; returns the facings sent."""
    out = []
    for k in range(k0, k1):
        vc.update(cmd)
        vc.t_cmd = t0 + dt * k
        vc.apply(pose(yaw=yaw_fn(k)), t0 + dt * k)
        out.append(vc.mux.cmd.facing_w)
    return out


def test_turn_in_place_is_stepped_not_ramped():
    """IDLE facing changes are discrete steps (walk diagnosis / Motion.turn_cmd): >= 0.4 s apart unless a full 30 deg
    step is pending, <= 30 deg from the last COMMANDED facing, and only while the body is within 15 deg."""
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    t = time.monotonic()
    turn = {"vx": 0.02, "vy": 0.0, "wz": 0.5}
    f = _run(vc, t, 0, 100, turn, lambda k: 0.0)                               # 2 s, the body does not turn
    assert mux.cmd.mode == LocomotionMode.IDLE and vc.state == "turn"
    changes = [k for k in range(1, len(f)) if abs(f[k] - f[k - 1]) > 1e-9]
    assert 2 <= len(changes) <= 4                                            # a ramp would change every tick
    assert all(b - a >= 20 for a, b in zip(changes, changes[1:]))            # >= 0.4 s apart
    assert max(f) < math.radians(15) + math.radians(30) + 1e-6              # frozen once the body lags > 15 deg
    lag_cmd = vc.cmd_facing
    assert abs(vc.want - lag_cmd) <= math.radians(30) + 1e-9                 # anti-windup on the COMMANDED facing
    # the body catches up (within 15 deg): a full 30 deg step goes out at once
    f2 = _run(vc, t, 100, 101, turn, lambda k: lag_cmd - math.radians(10))
    assert abs(f2[-1] - (lag_cmd + math.radians(30))) < 1e-6
    # rotation request ends: the remainder (if any) goes out once, then the facing is held
    vc.want = vc.cmd_facing + math.radians(5)
    stop = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    f3 = _run(vc, t, 101, 130, stop, lambda k: f2[-1] - math.radians(2))
    assert vc.state == "hold" and len(set(round(x, 9) for x in f3)) == 1
    assert abs(f3[-1] - (f2[-1] + math.radians(5))) < 1e-6


def test_walking_facing_is_continuous_and_gated_by_catchup():
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    t = time.monotonic()
    walk = {"vx": 0.4, "vy": 0.0, "wz": 0.3}
    f = _run(vc, t, 0, 26, walk, lambda k: 0.0)                               # 0.5 s, body on its facing
    assert mux.cmd.mode == LocomotionMode.SLOW_WALK
    assert abs(f[-1] - 0.3 * 0.02 * 25) < 1e-6                               # integrated wz, every tick
    # the body lags more than 15 deg behind the command: the facing waits, the demand stays <= 30 deg ahead
    held = vc.cmd_facing
    f2 = _run(vc, t, 26, 200, walk, lambda k: held - math.radians(20))
    assert all(abs(x - held) < 1e-9 for x in f2)
    assert abs(wrap_deg(vc.want - held) - 30.0) < 1e-6
    assert vc.stats["catchup_waits"] > 0


def wrap_deg(a):
    return math.degrees((a + math.pi) % (2 * math.pi) - math.pi)


def test_watchdog_goes_idle_and_stale_messages_dropped():
    cfg, mux = BodyConfig(), FakeMux()
    vc = VelocityCommander(cfg, mux, "t")
    assert vc.update({"vx": 0.4, "t_wall": time.time()})
    t = time.monotonic()
    assert vc.apply(pose(yaw=0.3), t) == "walk"
    assert vc.apply(pose(yaw=0.35), t + 0.31) == "watchdog"
    # IDLE holding the last COMMANDED facing (0.3), not the measured yaw (0.35)
    assert mux.cmd.mode == LocomotionMode.IDLE and abs(mux.cmd.facing_w - 0.3) < 1e-9
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


# -- the Nav2 link never blocks the 50 Hz body loop (verifier finding 2) ------------------------------------------
import json  # noqa: E402
import threading  # noqa: E402

import zmq  # noqa: E402

from body.nav2_backend import DOWN_CACHE_S, Nav2Link  # noqa: E402


class FakeBridge(threading.Thread):
    """REP socket answering like nav2/ros_bridge.py (echoes rid); per-op delays."""

    def __init__(self, port, delays=None, replies=None):
        super().__init__(daemon=True)
        self.port, self.delays, self.replies = port, delays or {}, replies or {}
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.run_ = True
        self.ops = []

    def run(self):
        while self.run_:
            if not self.sock.poll(50):
                continue
            req = json.loads(self.sock.recv())
            op = req.get("op")
            self.ops.append(op)
            time.sleep(self.delays.get(op, 0.0))
            rep = dict(self.replies.get(op, {"ok": True}))
            rep["rid"] = req.get("rid")
            self.sock.send(json.dumps(rep).encode())
        self.sock.close(0)
        self.ctx.term()

    def stop(self):
        self.run_ = False
        self.join(2.0)


def test_nav2_link_ping_is_bounded_and_down_is_cached(port_offset):
    port = 5620 + port_offset
    link = Nav2Link(f"tcp://127.0.0.1:{port}")
    t0 = time.monotonic()
    ok, why = link.ready()                        # nothing listening: ZMQ_IMMEDIATE -> fails at once
    assert not ok and "bridge_unreachable" in why and time.monotonic() - t0 < 0.06
    t0 = time.monotonic()
    ok, why = link.ready()
    assert not ok and "cached" in why and time.monotonic() - t0 < 0.005
    link._down = None
    br = FakeBridge(port, delays={"ping": 0.3}, replies={"ping": {"ok": True, "nav2_ready": True}})
    br.start()
    try:
        time.sleep(0.3)                           # let the DEALER connect
        t0 = time.monotonic()
        ok, why = link.ready()                    # a hung bridge: 50 ms, then cached
        assert not ok and 0.04 < time.monotonic() - t0 < 0.1
        assert link.ready()[0] is False and "cached" in link.ready()[1]
        time.sleep(0.4)
        link._down = None
        br.delays = {}
        ok, _ = link.ready()
        assert ok, "late reply of the timed-out ping must not be taken for the new one"
    finally:
        br.stop()
        link.close()
    assert DOWN_CACHE_S >= 2.0


class _Ctx:
    def __init__(self, link):
        self.cfg = BodyConfig()
        self.mux = FakeMux()
        self.link = link
        self.events = []

    def nav2_link(self):
        return self.link

    def emit(self, op_id, state, data):
        self.events.append((state, data))

    def velocity(self):
        return 0.0, 0.0, 0.0


def test_nav2_goto_ticks_never_block_on_a_slow_bridge(port_offset):
    """goto reply after 0.3 s and status replies after 0.2 s: start() and every tick() return within a few ms."""
    port = 5620 + port_offset
    br = FakeBridge(port, delays={"goto": 0.3, "status": 0.2},
                    replies={"goto": {"ok": True, "plan": {"length_m": 3.0}, "goal": {"x": 3.0, "y": 0.0}},
                             "status": {"ok": True, "state": "active", "feedback": {"number_of_recoveries": 0}},
                             "cancel": {"ok": True, "state": "canceling"}})
    br.start()
    link = Nav2Link(f"tcp://127.0.0.1:{port}")
    try:
        time.sleep(0.3)
        m = Nav2GoToMotion("g1", {"x": 3.0, "y": 0.0}, _Ctx(link))
        t0 = time.monotonic()
        link.ready()                              # select_motion's ping, then start(): <= ~50 ms together
        out = m.start(pose())
        assert time.monotonic() - t0 < 0.06 and out.get("goto_pending") and m.phase == "starting"
        worst, t_end = 0.0, time.monotonic() + 1.5
        while time.monotonic() < t_end:
            t1 = time.monotonic()
            assert m.tick(pose(), t1) is None
            worst = max(worst, time.monotonic() - t1)
            time.sleep(0.02)
        assert m.phase == "navigate" and m.nav_state == "active"
        assert worst < 0.01, f"a tick blocked {worst * 1e3:.1f} ms"
        t1 = time.monotonic()
        m.on_cancel(pose(), "stop")
        assert time.monotonic() - t1 < 0.04
        time.sleep(0.5)
        assert "cancel" in br.ops
    finally:
        br.stop()
        link.close()
