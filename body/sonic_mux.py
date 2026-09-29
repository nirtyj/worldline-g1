"""SonicMux: the ONLY binder of SONIC's input PUB (contract port 5556).

gear_sonic_deploy in zmq_manager mode SUBSCRIBES (connects) to host:5556 on topics command / planner / pose
(zmq_manager.hpp:1-39, 96-144), so exactly one process may bind it. Everything that wants to move the robot goes
through this object; it owns the single PUB socket and a single sender thread (ZMQ sockets are not thread safe).

Semantics honoured (WBC @b042411):
- Planner keepalive: the deploy clears each planner message after use and falls back to IDLE when no message arrived
  for 1 s (zmq_manager.hpp:581-642, PLANNER_TIMEOUT). We therefore re-send the current command at `keepalive_hz`
  (default 50 Hz, >= 10 Hz required) for as long as the robot should be under planner control.
- The deploy re-plans whenever mode/facing/height change or (moving) when speed/direction change
  (g1_deploy_onnx_ref.cpp:3659-3732). A deadband keeps the exact previous vectors for tiny changes so noise does not
  force a re-plan every tick.
- Watchdog: if the control loop stops refreshing the command for `stale_s`, the mux sends IDLE (hold) with the last
  facing instead of repeating a stale walk command.
- command{start:1, stop:0, planner:1} starts control in PLANNER mode (zmq_manager.hpp:232-300, 525-570).
- command{stop:1} makes the deploy exit through Stop() -> CreateDampingCommand() (kp=0, kd=8) and the robot
  collapses (g1_deploy_onnx_ref.cpp:2718-2745, main loop :4513-4530). So it is ONLY used by shutdown_control()
  (m1_down.sh, after P1's band is on). A user stop is planner IDLE (hold()).
- Upper-body overlay (body/arm.py): while set_upper() is active every planner message also carries
  upper_body_position[17] + upper_body_velocity[17] (wire order, body/joint_map.py) and, if given,
  left/right_hand_joints[7] (zmq_manager.hpp:885-980). The deploy re-evaluates the override on every consumed
  message (:581-596), so the overlay is independent of the planner command: motions keep the legs, the arm channel
  the arms. The overlay is NOT subject to the stale-command watchdog above: if the control loop stalls the arms keep
  their last target (the arm channel's own watchdog runs in that loop); if this process dies the deploy's 1 s
  planner timeout drops the override (:618-628).
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
from dataclasses import dataclass, field, asdict

import zmq

from .frames import PlannerFrame
from .wire import LocomotionMode, build_command_message, build_planner_message, wrap


@dataclass
class PlannerCmd:
    mode: int = LocomotionMode.IDLE
    move_w: tuple = (0.0, 0.0)        # world-frame direction of travel (magnitude ignored)
    facing_w: float | None = None     # world yaw to face; None = keep the last facing sent
    speed: float = -1.0               # m/s; <= 0 -> mode default (planner_onnx.md:79-92)
    height: float = -1.0
    owner: str = ""                   # op id that produced it (evidence)

    def moving(self) -> bool:
        return self.mode not in LocomotionMode.STATIC and math.hypot(*self.move_w) > 1e-6


class SonicMux:
    def __init__(self, endpoint: str, frame: PlannerFrame, keepalive_hz: float = 50.0, stale_s: float = 0.3,
                 dir_deadband_deg: float = 2.0, speed_deadband: float = 0.03, log=print,
                 cmd_log_path: str | None = None, ctx: zmq.Context | None = None):
        if keepalive_hz < 10.0:
            raise ValueError("keepalive must be >= 10 Hz (deploy planner timeout is 1 s)")
        self.endpoint = endpoint
        self.frame = frame
        self.period = 1.0 / keepalive_hz
        self.stale_s = stale_s
        self.dir_db = math.radians(dir_deadband_deg)
        self.speed_db = speed_deadband
        self.log = log
        self._ctx = ctx or zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._cmd: PlannerCmd | None = None
        self._cmd_t = 0.0
        self._queue: collections.deque[tuple[bytes, str]] = collections.deque()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_sent: dict | None = None          # planner-frame values last sent
        self._upper: tuple | None = None             # (pos17, vel17, left7|None, right7|None), wire order
        self._upper_t = 0.0
        self._last_facing_w: float | None = None
        self.control_started = False
        self.control_stopped = False
        self.t_start_sent: float | None = None
        self.stats = {"planner_sent": 0, "command_start_sent": 0, "command_stop_sent": 0, "stale_holds": 0,
                      "frame_unknown_holds": 0, "replan_changes": 0, "upper_sent": 0, "hands_sent": 0,
                      "by_mode": collections.Counter()}
        self._rate_ts: collections.deque[float] = collections.deque(maxlen=200)
        self.changes: collections.deque[dict] = collections.deque(maxlen=5000)  # command changes (evidence)
        self._cmd_log = open(cmd_log_path, "a", buffering=1) if cmd_log_path else None

    # -- lifecycle -----------------------------------------------------------------------------
    def bind(self) -> None:
        s = self._ctx.socket(zmq.PUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDHWM, 20)
        s.bind(self.endpoint)  # raises ZMQError(EADDRINUSE) if someone else already binds it: fatal by design
        self._sock = s
        self.log(f"[mux] bound PUB {self.endpoint}")

    def start(self) -> None:
        if self._sock is None:
            self.bind()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="sonic-mux", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None
        if self._cmd_log:
            self._cmd_log.close()

    # -- API (thread safe) ---------------------------------------------------------------------
    def set(self, cmd: PlannerCmd) -> None:
        with self._lock:
            self._cmd = cmd
            self._cmd_t = time.monotonic()

    def set_upper(self, pos17, vel17=None, left7=None, right7=None) -> None:
        """Overlay the upper-body/hand override on every planner message from now on (wire order, see joint_map)."""
        if len(pos17) != 17 or (vel17 is not None and len(vel17) != 17):
            raise ValueError("upper body vectors must have 17 values")
        for h in (left7, right7):
            if h is not None and len(h) != 7:
                raise ValueError("hand vectors must have 7 values")
        ov = (tuple(float(v) for v in pos17), tuple(float(v) for v in (vel17 if vel17 is not None else [0.0] * 17)),
              None if left7 is None else tuple(float(v) for v in left7),
              None if right7 is None else tuple(float(v) for v in right7))
        with self._lock:
            self._upper = ov
            self._upper_t = time.monotonic()

    def clear_upper(self) -> None:
        with self._lock:
            self._upper = None

    def upper(self) -> tuple | None:
        with self._lock:
            return self._upper

    def hold(self, facing_w: float | None = None, owner: str = "") -> None:
        self.set(PlannerCmd(LocomotionMode.IDLE, (0.0, 0.0), facing_w, -1.0, -1.0, owner))

    def start_control(self, repeats: int = 3) -> None:
        """Queue command{start:1, stop:0, planner:1}. Repeats are harmless: start is only acted on while
        !operator_state.start (zmq_manager.hpp:525), and start/stop are OR-accumulated (:748-760)."""
        msg = build_command_message(start=True, stop=False, planner=True)
        with self._lock:
            for _ in range(repeats):
                self._queue.append((msg, "start"))
        self.control_started = True
        self.control_stopped = False
        self.t_start_sent = time.monotonic()

    def shutdown_control(self, hold_s: float = 0.5, repeats: int = 3) -> None:
        """IDLE for hold_s, then command{stop}. The deploy exits and damps (robot collapses unless the band holds it)."""
        self.hold(self._last_facing_w, owner="shutdown")
        time.sleep(hold_s)
        msg = build_command_message(start=False, stop=True, planner=True)
        with self._lock:
            for _ in range(repeats):
                self._queue.append((msg, "stop"))
        t0 = time.monotonic()
        while self._queue and time.monotonic() - t0 < 1.0:
            time.sleep(0.01)
        self.control_started = False
        self.control_stopped = True

    @property
    def last_facing_w(self) -> float | None:
        return self._last_facing_w

    def current(self) -> tuple[PlannerCmd | None, float]:
        with self._lock:
            return self._cmd, self._cmd_t

    def rate_hz(self) -> float:
        ts = list(self._rate_ts)
        now = time.monotonic()
        ts = [t for t in ts if now - t < 2.0]
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def snapshot(self) -> dict:
        cmd, t = self.current()
        return {
            "endpoint": self.endpoint,
            "control_started": self.control_started,
            "control_stopped": self.control_stopped,
            "rate_hz": round(self.rate_hz(), 2),
            "cmd": None if cmd is None else {**asdict(cmd), "age_s": round(time.monotonic() - t, 3)},
            "upper": None if self._upper is None else {"age_s": round(time.monotonic() - self._upper_t, 3),
                                                       "hands": [self._upper[2] is not None, self._upper[3] is not None]},
            "last_sent": self._last_sent,
            "stats": {**{k: v for k, v in self.stats.items() if k != "by_mode"},
                      "by_mode": {LocomotionMode.NAMES.get(k, str(k)): v for k, v in self.stats["by_mode"].items()}},
            "frame": self.frame.to_dict(),
        }

    # -- sender thread -------------------------------------------------------------------------
    def _effective(self) -> PlannerCmd:
        with self._lock:
            cmd, t = self._cmd, self._cmd_t
        now = time.monotonic()
        if cmd is None:
            return PlannerCmd(LocomotionMode.IDLE, (0.0, 0.0), self._last_facing_w, owner="default")
        if now - t > self.stale_s and cmd.mode != LocomotionMode.IDLE:
            self.stats["stale_holds"] += 1
            return PlannerCmd(LocomotionMode.IDLE, (0.0, 0.0), self._last_facing_w, owner="watchdog")
        return cmd

    def _build(self, cmd: PlannerCmd) -> tuple[bytes, dict]:
        mode = int(cmd.mode)
        if mode not in LocomotionMode.STATIC and not cmd.moving():
            # Upstream clients never send a non-static mode with zero movement; the keyboard switches to IDLE
            # (keyboard_handler.hpp:681-686). SLOW_WALK + speed -1 replans every second (g1_deploy_onnx_ref.cpp:3729)
            # and fell / overshot in the reference runs (docs/walk_diagnosis.md).
            mode = LocomotionMode.IDLE
        if not self.frame.known and (cmd.moving() or cmd.facing_w is not None):
            # Cannot express a world direction yet; before start the planner frame is "current heading".
            if cmd.moving():
                self.stats["frame_unknown_holds"] += 1
            mode = LocomotionMode.IDLE if cmd.moving() else mode
            facing = (1.0, 0.0, 0.0)
            movement = (0.0, 0.0, 0.0)
            speed = -1.0
        else:
            fw = cmd.facing_w if cmd.facing_w is not None else self._last_facing_w
            if fw is None:
                facing = (1.0, 0.0, 0.0)  # planner frame zero == heading at init (localmotion_kplanner.hpp:346-351)
            else:
                facing = self.frame.facing_vec(fw)
                self._last_facing_w = fw
            if cmd.moving():
                mx, my = self.frame.vec_world_to_planner(*cmd.move_w)
                n = math.hypot(mx, my)
                movement = (mx / n, my / n, 0.0)
                speed = float(cmd.speed)
            else:
                movement = (0.0, 0.0, 0.0)
                speed = -1.0  # static modes force -1 anyway (zmq_manager.hpp:607-609)
        facing, movement, speed, changed = self._deadband(mode, facing, movement, speed)
        with self._lock:
            ov = self._upper
        if ov is None:
            msg = build_planner_message(mode, movement, facing, speed, float(cmd.height))
        else:
            msg = build_planner_message(mode, movement, facing, speed, float(cmd.height),
                                        upper_body_position=ov[0], upper_body_velocity=ov[1],
                                        left_hand_position=ov[2], right_hand_position=ov[3])
            self.stats["upper_sent"] += 1
            if ov[2] is not None or ov[3] is not None:
                self.stats["hands_sent"] += 1
        sent = {"mode": mode, "movement": [round(v, 5) for v in movement], "facing": [round(v, 5) for v in facing],
                "speed": round(speed, 4), "owner": cmd.owner}
        if changed:
            self.stats["replan_changes"] += 1
            rec = {"t_wall": time.time(), "t_mono": time.monotonic(), **sent,
                   "move_w": list(cmd.move_w), "facing_w": self._last_facing_w,
                   "frame_offset_deg": round(math.degrees(self.frame.offset), 3)}
            self.changes.append(rec)
            if self._cmd_log:
                self._cmd_log.write(json.dumps(rec) + "\n")
        return msg, sent

    def _deadband(self, mode, facing, movement, speed):
        last = self._last_sent
        if last is None or last["mode"] != mode:
            return facing, movement, speed, True
        changed = False
        lf, lm = last["facing_t"], last["movement_t"]
        if _ang(facing, lf) < self.dir_db:
            facing = lf
        else:
            changed = True
        moving_now = abs(movement[0]) + abs(movement[1]) > 0
        moving_last = abs(lm[0]) + abs(lm[1]) > 0
        if moving_now and moving_last and _ang(movement, lm) < self.dir_db:
            movement = lm
        elif moving_now != moving_last or (moving_now and moving_last):
            changed = True
        if abs(speed - last["speed"]) < self.speed_db:
            speed = last["speed"]
        else:
            changed = True
        return facing, movement, speed, changed

    def _loop(self) -> None:
        nxt = time.monotonic()
        while self._running:
            try:
                while True:
                    with self._lock:
                        item = self._queue.popleft() if self._queue else None
                    if item is None:
                        break
                    raw, kind = item
                    self._sock.send(raw)
                    self.stats[f"command_{kind}_sent"] += 1
                    self.log(f"[mux] sent command {kind}")
                cmd = self._effective()
                msg, sent = self._build(cmd)
                self._sock.send(msg)
                self._last_sent = {**sent, "facing_t": tuple(sent["facing"]), "movement_t": tuple(sent["movement"])}
                # keep the exact tuples used on the wire for the deadband
                self._last_sent["facing_t"] = tuple(_unpack_vec(msg, "facing"))
                self._last_sent["movement_t"] = tuple(_unpack_vec(msg, "movement"))
                self.stats["planner_sent"] += 1
                self.stats["by_mode"][sent["mode"]] += 1
                self._rate_ts.append(time.monotonic())
            except Exception as e:  # never let the keepalive die silently
                self.log(f"[mux] ERROR in sender loop: {e!r}")
            nxt += self.period
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.monotonic()


def _ang(a, b) -> float:
    return abs(wrap(math.atan2(a[1], a[0]) - math.atan2(b[1], b[0])))


def _unpack_vec(msg: bytes, name: str):
    # movement at payload offset 4, facing at 16 (mode i32, movement f32[3], facing f32[3]); keep full f32 precision
    import struct
    base = len(b"planner") + 1280
    off = base + (4 if name == "movement" else 16)
    return struct.unpack_from("<fff", msg, off)
