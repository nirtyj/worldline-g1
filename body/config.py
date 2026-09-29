"""Ports and tunables for wl-body.

All contract ports shift together by one offset so a whole test stack can run beside the
integrated one (build phase: +100 for real components; the body's own fakes default to +200
so they never collide with another agent's +100 P1/deploy under test).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# Contract ports for the integrated M1 stack (docs/contracts/m1.md).
BASE_PORTS = {
    "p1_rep": 5600,       # P1 REP  (JSON/msgpack ops)
    "p1_pose": 5601,      # P1 PUB  topic "gt.pose"
    "camera": 5565,       # P1 PUB  gear_sonic sensor_server msgpack (ego_view)
    "p1_frames": 5602,    # P1 PUB  optional third-person camera, multipart [b"frame.tp", msgpack{jpeg,...}]
    "sonic_in": 5556,     # SonicMux PUB bind; deploy zmq_manager SUB connects (command/planner/pose)
    "sonic_debug": 5557,  # deploy PUB g1_debug (zmq_output_handler.hpp)
    "body_ctl": 5610,     # body ROUTER
    "body_evt": 5611,     # body PUB events/state
    "body_halt": 5612,    # body PULL halt lane: {op: halt|resume, epoch, t_wall} (M2b B.1, contract §3.10)
    "fake_link": 5690,    # tools/fake_deploy.py -> tools/fake_p1.py twist link (mocks only)
    "nav_bridge": 5620,   # nav2/ros_bridge.py REP (go_to backend nav2: goto / cancel / status); ROS side
}


def port_offset_from_env(default: int = 0) -> int:
    try:
        return int(os.environ.get("WL_PORT_OFFSET", default))
    except ValueError:
        return default


def ports(offset: int | None = None) -> dict:
    off = port_offset_from_env() if offset is None else offset
    return {k: v + off for k, v in BASE_PORTS.items()}


def ep(port: int, host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{port}"


@dataclass
class BodyConfig:
    port_offset: int = 0
    host: str = "127.0.0.1"
    # SonicMux
    keepalive_hz: float = 50.0          # >= 10 Hz required (zmq_manager.hpp PLANNER_TIMEOUT = 1 s)
    cmd_stale_s: float = 0.30           # control loop silent longer than this -> mux holds IDLE
    dir_deadband_deg: float = 2.0       # suppress planner re-plans on tiny direction changes
    speed_deadband: float = 0.03
    # control loop
    control_hz: float = 50.0
    state_pub_hz: float = 5.0
    # speeds (keyboard_handler.hpp: SLOW_WALK clamped to 0.2..0.8; keyboard.md: strafe ~0.4)
    v_min: float = 0.2
    v_max: float = 0.8
    v_default: float = 0.45
    v_strafe_max: float = 0.4
    # stand / fall detection (G1 pelvis ~0.78 m standing)
    pelvis_z_min: float = 0.55
    pelvis_z_max: float = 1.00
    # watchdogs
    pose_stale_s: float = 0.5
    debug_stale_s: float = 1.0
    # navigation
    robot_radius: float = 0.25
    plan_res: float = 0.10
    pos_tol: float = 0.15               # arrival radius for the follower
    final_pos_tol: float = 0.25         # success tolerance after settle; E3 allows 0.30
    approach_tol: float = 0.12          # after settle, a slow holonomic approach runs while the error is above this
                                        # (SONIC glides 0.1-0.2 m after IDLE: stops take 0.4-1.1 s, sonic_deploy.md §6)
    yaw_tol_deg: float = 6.0
    final_yaw_tol_deg: float = 12.0     # E3 allows 15
    lookahead_m: float = 0.6
    a_dec: float = 0.35
    stuck_window_s: float = 3.0
    stuck_min_progress_m: float = 0.08
    max_replans: int = 2
    settle_s: float = 0.8
    stop_v_eps: float = 0.05
    turn_style: str = "idle"            # "idle": IDLE + facing (keyboard Q/E); "slowwalk": fallback
    walk_hold_line: bool = True         # forward walks with yaw_rate 0: pure pursuit on the start line (WalkMotion)
    walk_lookahead_m: float = 1.0
    walk_ct_max_deg: float = 30.0
    turn_push: float = 0.6              # FacingServo residual push (motions.py; sonic_deploy.md §0.5: 0.6 measured on P1)
    # Outer-loop heading bias integrator [1/s]. 0 = off (M1 default): on P1 the planner frame from g1_debug equals the
    # GT frame (P1's IMU quaternion is the GT pelvis orientation), and the only steady heading error is SONIC's
    # IDLE-turn shortfall, which is planner behaviour. Learnt as a frame bias it would also rotate every later walk
    # direction (by up to the 20 deg clamp; inferred from frames.py, not measured); FacingServo corrects turns
    # explicitly instead.
    heading_bias_ki: float = 0.0
    heading_bias_max_deg: float = 20.0
    # go_to backend (docs/nav2.md): "nav2" (ROS 2 Nav2 through nav2/ros_bridge.py) or "astar" (path_follower.py)
    nav_backend: str = field(default_factory=lambda: os.environ.get("NAV_BACKEND", "nav2").strip().lower())
    nav_fallback: bool = field(default_factory=lambda: os.environ.get("NAV_FALLBACK", "1") not in ("0", "false"))
    nav2_goto_timeout_s: float = 8.0    # bridge goto = ComputePathToPose pre-check + NavigateToPose accept
    # streaming velocity op / Nav2 cmd_vel -> SONIC (body/velocity.py)
    vel_watchdog_s: float = 0.30        # no fresh velocity message for this long -> IDLE
    vel_deadband: float = 0.05          # |v| below this: no translation (turn in place / stand)
    vel_wz_deadband: float = 0.02
    facing_lead_max_deg: float = 25.0   # facing setpoint (integrated wz) stays within this of the GT yaw
    # arm channel (op `arm`, body/arm.py; measured in docs/arm_tracking.md)
    arm_watchdog_s: float = 0.30        # no message for this long -> hold the last pose
    arm_hold_s: float = 1.0             # ... for this long, then blend back to SONIC's own arms
    arm_blend_s: float = 1.5
    arm_max_vel: float = 6.0            # slew limit on the sent targets [rad/s]
    arm_servo_ki: float = 2.0           # integral outer loop on measured arm joints [1/s]; 0 = off. ki 2 took static
                                        # palm errors from 17-75 mm to 2-23 mm; ki 4 was no better and tilted more
    arm_servo_delay_s: float = 0.15     # error vs the target this long ago (SONIC's lag), so the loop ignores lag
    arm_servo_max: float = 0.4          # |correction| per joint [rad]
    arm_ik_worker: str = "process"      # arm_script IK: "process" (a spawn-context worker, body/ik_worker.py) | "inline"
    gil_switch_s: float = 0.001         # sys.setswitchinterval in body.service main (CPython default 0.005)
    # M2b body wave (docs/contracts/m1.md §3.10-§3.14)
    deploy_lost_s: float = 1.0          # g1_debug older than this while in control -> fault deploy_lost (PLAN §6.6 says
                                        # 300 ms; 1.0 s = the existing deploy_stale watchdog, so a loaded box's jitter
                                        # does not engage the band)
    fault_band_on: bool = True          # on a fall or deploy_lost: P1 band{on} at once (sim; PLAN §7.3.3 path A step 1)
    session_watchdog_s: float = 1.0     # default runtime-ping watchdog of a `hello` session (PLAN §6.6: 1.0 s)
    approach_op_v: float = 0.2          # op approach: SLOW_WALK speed forward/back (its floor, keyboard_handler.hpp:267-272)
    approach_op_v_lat: float = 0.3      # op approach: sideways (live: 0.2 m/s hardly steps sideways, body/approach.py)
    approach_op_max_dist: float = 0.6   # op approach: longer moves are go_to's job (rejected too_far)
    approach_op_t_stop: float = 0.4     # op approach: initial travel-after-IDLE time constant [s] per direction (learnt)

    @property
    def ports(self) -> dict:
        return ports(self.port_offset)

    def ep(self, name: str) -> str:
        return ep(self.ports[name], self.host)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ports"] = self.ports
        return d
