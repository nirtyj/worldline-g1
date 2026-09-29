"""Arm/hand channel: streams upper-body and Dex3 joint targets INTO SONIC through the planner message (op `arm`).

Owner decision 2026-09-29 (recommendation (b), docs/arena_vs_sonic.md §4.2): GR00T drives the arms and hands by
joint targets; SONIC stays the only body controller and keeps the legs and balance. The targets ride on every
planner message SonicMux sends (`upper_body_position[17]`, `upper_body_velocity[17]`, `left/right_hand_joints[7]`,
zmq_manager.hpp:885-980), so the arm channel composes with whatever motion owns the legs (IDLE hold, walk, velocity,
turn_to, go_to): motions set mode/movement/facing/speed, this channel sets the upper-body overlay.

What the deploy does with them (WBC @ b042411):
- Every consumed planner message sets `has_upper_body_control_ = upper_body_position present` and
  `has_hand_joints_ = hand fields present` (zmq_manager.hpp:581-596); a message without them releases the override
  at once. After 1 s without any planner message both are cleared (:618-628).
- The 17 targets REPLACE the upper-body joints of all 10 future reference frames the encoder sees
  (motion_joint_positions_10frame_step5 / _velocities_, g1_deploy_onnx_ref.cpp:738-800, 861-867;
  policy/release/observation_config.yaml encoder mode g1). SONIC then *tracks* them with its whole-body policy:
  there is no direct PD path for the arms, so tracking error and lag are the policy's (docs/arm_tracking.md).
  `upper_body_velocity` replaces the reference joint velocities the same way; when it was never sent the deploy
  uses zeros (input_interface.hpp:380-390), so this channel always sends it (zeros unless `vel` says otherwise) and a
  stale velocity from an earlier client can never persist.
- Hands go straight to rt/dex3/<side>/cmd with kp 1.5 / kd 0.1, clipped to `max_close_ratio` and to +-0.25 rad from
  the measured position per write (dex3_hands.hpp:114-190). Without hand fields the deploy commands its default
  fist (input_interface.hpp:333-362).

Channel states: off -> stream -> (watchdog) hold -> blend -> off.
- stream: the latest client target (clamped to the URDF limits minus 0.02 rad, slew-limited to `max_vel`).
- hold: no message for `watchdog_s` (0.3 s): keep sending the last pose for `hold_s` (1.0 s). A new message from
  the owner resumes the stream.
- blend: min-jerk from the held pose to the policy-default arms over `blend_s` (1.5 s), then the override is
  dropped (off). "Policy-default" = SONIC's own planner reference for the upper body, read live from g1_debug
  `body_q_target` (the current reference motion frame without the override, output_interface.hpp:175-192); the
  hands blend to the deploy's default fist. Dropping the override when both are equal is seamless.
- A client `end` goes straight to blend; `stop {arms: true}` too. A fall (fault) or a service shutdown drops the
  override at once (no blend).

Ownership (one owner at a time), keyed by the message's `stream` id:
1. With the channel off, any stream may start it (the robot must be standing and not faulted). Its first message
   creates the op record (events accepted/progress/succeeded|canceled|failed), later messages are stream updates
   (reply only, no events), exactly like the `velocity` op.
2. While the owner is streaming, a message from another stream is rejected `arm_busy` unless it carries
   `preempt: true`; then the new stream takes over from the pose being sent (no blend) and the old op ends
   `canceled` (reason preempted). The old stream's messages are rejected `arm_preempted` while the new owner
   streams; once the arms are free it may start a new session.
3. Once the owner has stopped streaming (hold or blend), any stream may take over without `preempt`.
4. The legs are never owned here: motion ops (walk/velocity/turn_to/go_to/stop) keep the legs, and pre-empting a
   motion does not touch the arms.

Servo (default on, `servo_ki` 2.0; 0 turns it off): SONIC tracks the override with a pose-dependent steady-state
error of 0.1-0.3 rad on some joints and ~150 ms of lag (docs/arm_tracking.md). The channel closes an integral loop on
the measured arm joints (g1_debug body_q, 14 arm joints, not the waist):
corr += ki * dt * (target(t - servo_delay_s) - q_measured), |corr| <= servo_max (0.4 rad), sent = target + corr.
Comparing against the target `servo_delay_s` (0.15 s) ago keeps the integrator from chasing the lag. Measured: static
palm error 17-75 mm -> 2-23 mm; carrying while walking 41-45 mm -> 9-17 mm RMS. The lag itself is not removed: a
client that knows its future targets (a GR00T chunk) should send them ~0.15 s ahead.

Waist: the 17-vector includes waist yaw/roll/pitch. By default (`waist: "ref"`) the channel sends SONIC's own
reference waist (g1_debug body_q_target, i.e. what SONIC would do without the override) and only the 14 arm
joints are the client's. `waist: "cmd"` sends the client's waist values instead (default when the client gives
waist joints by name or a 17-list with `waist: "cmd"`).
"""

from __future__ import annotations

import collections
import math
import time
from typing import Sequence

from . import joint_map as jm

N = jm.N_UPPER
WAIST_IDX = (0, 1, 2)              # in UPPER_BODY_MUJOCO_JOINTS (mj17) order
ARM_IDX = tuple(range(3, 17))
_LIM = [(jm.JOINT_LIMITS[n][0] + 0.02, jm.JOINT_LIMITS[n][1] - 0.02) for n in jm.UPPER_BODY_MUJOCO_JOINTS]
_MJ17_IDX = {n: k for k, n in enumerate(jm.UPPER_BODY_MUJOCO_JOINTS)}


class ArmError(Exception):
    def __init__(self, reason: str, data: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.data = data or {}


def _minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _vec17(v, base: Sequence[float], what: str) -> tuple[list[float], bool]:
    """Parse a dict {joint: q} (overlaid on base) or a 17-list in UPPER_BODY_MUJOCO_JOINTS order.
    Returns (mj17, waist_given)."""
    if isinstance(v, dict):
        try:
            out = jm.mj17_from_named({str(k): float(x) for k, x in v.items()}, base)
        except KeyError as e:
            raise ArmError("bad_args", {"arg": what, "unknown_joint": str(e.args[0]),
                                        "joints": list(jm.UPPER_BODY_MUJOCO_JOINTS)})
        waist = any(k in v for k in jm.WAIST_JOINTS)
    elif isinstance(v, (list, tuple)):
        if len(v) != N:
            raise ArmError("bad_args", {"arg": what, "error": f"need 17 values in order {jm.UPPER_BODY_MUJOCO_JOINTS}",
                                        "got": len(v)})
        out = [float(x) for x in v]
        waist = True
    else:
        raise ArmError("bad_args", {"arg": what, "error": "dict {joint: rad} or a 17-list"})
    if not all(math.isfinite(x) for x in out):
        raise ArmError("bad_args", {"arg": what, "error": "non-finite value"})
    return out, waist


def _hand(v, side: str, what: str) -> list[float]:
    if isinstance(v, (int, float)):
        return jm.hand_closure(side, float(v))
    if isinstance(v, (list, tuple)) and len(v) == jm.N_HAND and all(math.isfinite(float(x)) for x in v):
        q, _ = jm.clamp_hand(side, [float(x) for x in v])
        return q
    raise ArmError("bad_args", {"arg": what, "error": "7 Dex3 values (thumb_0..index_1) or a closure 0..1"})


class ArmChannel:
    """Owned by BodyService; handle() runs on the service thread for op `arm`, tick() every control tick."""

    def __init__(self, cfg, mux, deploy, emit, log=print, record=None):
        self.cfg = cfg
        self.mux = mux
        self.deploy = deploy
        self.emit = emit                # emit(op_id, state, data)
        self.record = record            # record(op_id, op, args): creates the op record
        self.log = log
        self.state = "off"
        self.owner: str | None = None   # stream id
        self.op_id: str | None = None
        self.preempted: set[str] = set()
        self.stats = {"messages": 0, "stale_dropped": 0, "watchdog_trips": 0, "resumed": 0, "clamped": 0,
                      "slew_limited_ticks": 0, "ticks": 0, "sessions": 0, "preemptions": 0, "rejected_busy": 0}
        self._reset_session()

    # -- session state ----------------------------------------------------------------------------
    def _reset_session(self) -> None:
        self.target: list[float] | None = None          # mj17, client target (clamped)
        self.target_vel: list[float] | None = None      # mj17, client velocity (vel="cmd") or estimate
        self.sent: list[float] | None = None            # mj17, last value sent (after slew limit)
        self.hands: dict[str, list[float] | None] = {"left": None, "right": None}
        self.hands_sent: dict[str, list[float] | None] = {"left": None, "right": None}
        self.waist_mode = "ref"
        self.vel_mode = "zero"
        self.watchdog_s = float(getattr(self.cfg, "arm_watchdog_s", 0.3))
        self.hold_s = float(getattr(self.cfg, "arm_hold_s", 1.0))
        self.blend_s = float(getattr(self.cfg, "arm_blend_s", 1.5))
        self.max_vel = float(getattr(self.cfg, "arm_max_vel", 6.0))
        # joint servo (outer loop on g1_debug body_q, see module docstring "Servo"); 0 = off
        self.servo_ki = float(getattr(self.cfg, "arm_servo_ki", 2.0))
        self.servo_delay_s = float(getattr(self.cfg, "arm_servo_delay_s", 0.15))
        self.servo_max = float(getattr(self.cfg, "arm_servo_max", 0.4))
        self.corr = [0.0] * N
        self._t_servo: float | None = None
        self._hist: collections.deque = collections.deque(maxlen=100)     # (t_mono, target mj17), ~2 s
        self.t_msg: float | None = None
        self.t_tick: float | None = None
        self.t_hold: float | None = None
        self.t_blend: float | None = None
        self.blend_from: list[float] | None = None
        self.blend_hands_from: dict[str, list[float] | None] = {"left": None, "right": None}
        self.ended_by: str | None = None
        self.t_start = time.monotonic()
        self._prev_t: tuple[float, list[float]] | None = None   # (t, target) for velocity estimation
        self.session = {"messages": 0, "watchdog_trips": 0, "clamped": 0, "slew_limited_ticks": 0}
        self._t_progress = time.monotonic()

    # -- reference (SONIC's own upper body, g1_debug) ----------------------------------------------
    def _debug(self) -> dict:
        return self.deploy.latest or {}

    def reference_mj17(self) -> list[float]:
        d = self._debug()
        for key in ("body_q_target", "body_q"):
            v = d.get(key)
            if v is not None and len(v) == 29:
                return jm.mj17_from_mujoco(v)
        return list(jm.DEFAULT_ANGLES[12:29])

    def measured_mj17(self) -> list[float] | None:
        v = self._debug().get("body_q")
        return jm.mj17_from_mujoco(v) if v is not None and len(v) == 29 else None

    # -- API: op `arm` ----------------------------------------------------------------------------
    def handle(self, op_id: str, args: dict, can_start: tuple[bool, str | None, dict]) -> dict:
        """One `arm` message. Raises ArmError(bad_args) before changing any state."""
        stream = str(args.get("stream") or op_id)
        now = time.monotonic()
        if args.get("end"):
            if self.owner == stream and self.state in ("stream", "hold"):
                self._start_blend(now, "client")
                return {"ok": True, "state": "done", "data": {"id": self.op_id, "arm": self.state}}
            return {"ok": False, "state": "rejected", "error": "not_owner",
                    "data": {"owner": self.owner, "arm": self.state}}
        if stream in self.preempted and self.owner != stream:
            if self.state == "stream":      # told once it lost the arms; it may start again once they are free
                return {"ok": False, "state": "rejected", "error": "arm_preempted", "data": {"owner": self.owner}}
            self.preempted.discard(stream)
        new_session = self.owner != stream or self.state == "off"
        if new_session:
            ok, reason, info = can_start
            if not ok:
                self.emit_reject(op_id, args, reason, info)
                return {"ok": False, "state": "rejected", "error": reason, "data": info}
            if self.state == "stream" and self.owner is not None and not args.get("preempt"):
                self.stats["rejected_busy"] += 1
                return {"ok": False, "state": "rejected", "error": "arm_busy",
                        "data": {"owner": self.owner, "op": self.op_id, "hint": "preempt: true takes over"}}
        base = self.target if (not new_session and self.target is not None) else \
            (self.sent if self.sent is not None else self.reference_mj17())
        parsed = self._parse(args, base)          # validate before touching any state
        if self._stale(args):
            self.stats["stale_dropped"] += 1
            return {"ok": False, "state": "rejected", "error": "stale_command",
                    "data": {"id": None if new_session else self.op_id, "t_wall": args.get("t_wall")}}
        if new_session:
            self._begin(op_id, stream, args, now)
        self._apply(parsed, args, now)
        return {"ok": True, "state": "accepted" if new_session else "done",
                "data": {"id": self.op_id, "arm": self.state, "stream": self.owner}}

    def emit_reject(self, op_id, args, reason, info) -> None:
        if self.record:
            self.record(op_id, "arm", args)
        self.emit(op_id, "failed", {"reason": reason, **(info or {})})

    def _stale(self, args: dict) -> bool:
        tw = args.get("t_wall")
        if tw is None:
            return False
        try:
            return time.time() - float(tw) > self.watchdog_s
        except (TypeError, ValueError):
            return False

    def _parse(self, args: dict, base: list[float]) -> dict:
        out = {}
        ub = args.get("upper_body")
        if ub is not None:
            out["target"], out["waist_given"] = _vec17(ub, base, "upper_body")
        ubv = args.get("upper_body_vel")
        if ubv is not None:
            out["vel"], _ = _vec17(ubv, [0.0] * N, "upper_body_vel")
        for side in ("left", "right"):
            v = args.get(f"{side}_hand")
            if v is not None:
                out[side] = _hand(v, side, f"{side}_hand")
        for k, lo, hi in (("watchdog_s", 0.05, 2.0), ("hold_s", 0.0, 30.0), ("blend_s", 0.2, 10.0),
                          ("max_vel", 0.5, 20.0), ("servo_ki", 0.0, 10.0), ("servo_delay_s", 0.0, 0.5),
                          ("servo_max", 0.0, 0.8)):
            if args.get(k) is not None:
                v = float(args[k])
                if not math.isfinite(v):
                    raise ArmError("bad_args", {"arg": k})
                out[k] = min(max(v, lo), hi)
        w = args.get("waist")
        if w is not None and w not in ("ref", "cmd"):
            raise ArmError("bad_args", {"arg": "waist", "error": "ref | cmd"})
        vm = args.get("vel")
        if vm is not None and vm not in ("zero", "est", "cmd"):
            raise ArmError("bad_args", {"arg": "vel", "error": "zero | est | cmd"})
        return out

    def _begin(self, op_id: str, stream: str, args: dict, now: float) -> None:
        prev_sent, prev_hands = self.sent, dict(self.hands_sent)
        if self.op_id is not None and self.state != "off":
            # take-over: the old op ends now; the new stream continues from the pose being sent
            if self.state == "stream":
                self.stats["preemptions"] += 1
                if len(self.preempted) > 100:
                    self.preempted.clear()
                self.preempted.add(self.owner)
                self._finish("canceled", {"reason": "preempted", "by": stream})
            else:
                self._finish("canceled", {"reason": "taken_over", "by": stream})
        self._reset_session()
        self.preempted.discard(stream)
        self.owner, self.op_id = stream, op_id
        self.state = "stream"
        self.stats["sessions"] += 1
        # continuity: start from what is being sent, else SONIC's own reference (never from zeros)
        self.sent = prev_sent if prev_sent is not None else self.reference_mj17()
        self.hands_sent = prev_hands
        if self.record:
            self.record(op_id, "arm", args)
        self.log(f"[arm] session {op_id} stream={stream} start")
        self.emit(op_id, "accepted", {"args": _brief_args(args), "stream": stream,
                                      "start_mj17": [round(v, 4) for v in self.sent]})

    def _apply(self, p: dict, args: dict, now: float) -> None:
        for k in ("watchdog_s", "hold_s", "blend_s", "max_vel", "servo_ki", "servo_delay_s", "servo_max"):
            if k in p:
                setattr(self, k, p[k])
        if args.get("vel") is not None:
            self.vel_mode = args["vel"]
        elif "vel" in p:
            self.vel_mode = "cmd"
        if "target" in p:
            if args.get("waist") is not None:
                self.waist_mode = args["waist"]
            elif p["waist_given"] and isinstance(args.get("upper_body"), dict):
                self.waist_mode = "cmd"
            tgt, n = jm.clamp_mj17(p["target"], margin=0.02)
            if n:
                self.stats["clamped"] += 1
                self.session["clamped"] += 1
            # velocity estimate from consecutive client targets (sender clock when given)
            t = float(args["t_wall"]) if args.get("t_wall") is not None else time.time()
            if "vel" in p:
                self.target_vel = p["vel"]
            elif self._prev_t is not None and t - self._prev_t[0] > 1e-3:
                dt = t - self._prev_t[0]
                raw = [(a - b) / dt for a, b in zip(tgt, self._prev_t[1])]
                if self.target_vel is None or dt > 0.2:
                    self.target_vel = raw
                else:
                    self.target_vel = [0.5 * a + 0.5 * b for a, b in zip(raw, self.target_vel)]
            else:
                self.target_vel = [0.0] * N
            self._prev_t = (t, tgt)
            self.target = tgt
            self._hist.append((now, tgt))
        elif self.target is None:
            self.target = list(self.sent)
        for side in ("left", "right"):
            if side in p:
                self.hands[side] = p[side]
        self.t_msg = now
        self.stats["messages"] += 1
        self.session["messages"] += 1
        if self.state in ("hold", "blend"):
            self.stats["resumed"] += 1
            self.log(f"[arm] {self.op_id} resumed from {self.state}")
            self.state = "stream"
            self.t_hold = self.t_blend = None
            self.ended_by = None
        # hand the new target to the mux now (its next send is <= 20 ms away) instead of at the next control tick
        self.tick(now)

    # -- ends --------------------------------------------------------------------------------------
    def end(self, reason: str) -> bool:
        """stop {arms: true}: blend back now. Returns True if a session was active."""
        if self.state in ("stream", "hold"):
            self._start_blend(time.monotonic(), reason)
            return True
        return self.state == "blend"

    def abort(self, reason: str) -> None:
        """Fall / shutdown: drop the override immediately."""
        if self.state == "off":
            return
        self.log(f"[arm] abort ({reason}): override dropped")
        self.mux.clear_upper()
        if self.op_id is not None:
            self._finish("failed" if reason == "fallen" else "canceled", {"reason": reason})
        self.state = "off"
        self.owner = None
        self.sent = None
        self.hands_sent = {"left": None, "right": None}

    def _start_blend(self, now: float, why: str) -> None:
        self.state = "blend"
        self.ended_by = self.ended_by or why
        self.t_blend = now
        self.blend_from = list(self.sent) if self.sent is not None else self.reference_mj17()
        self.blend_hands_from = {s: (list(self.hands_sent[s]) if self.hands_sent[s] is not None else None)
                                 for s in ("left", "right")}
        self.log(f"[arm] {self.op_id} blend back to SONIC's reference ({why}, {self.blend_s:.2f} s)")

    def _finish(self, state: str, data: dict) -> None:
        op = self.op_id
        self.op_id = None
        if op is not None:
            self.emit(op, state, {**data, "stream": self.owner, "ended_by": self.ended_by,
                                  "duration_s": round(time.monotonic() - self.t_start, 3), **self.session})

    # -- 50 Hz --------------------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        if self.state == "off":
            return
        # never below the nominal period: an extra tick right after a message must not throttle the slew limit
        dt = 0.02 if self.t_tick is None else min(0.05, max(0.02, now - self.t_tick))
        self.t_tick = now
        self.stats["ticks"] += 1
        if self.op_id is not None and now - self._t_progress >= 1.0:
            self._t_progress = now
            self.emit(self.op_id, "progress", {"arm": self.state, "msg_age_s": None if self.t_msg is None else
                                               round(now - self.t_msg, 3), **self.session})
        ref = self.reference_mj17()
        hands_des = {s: self.hands[s] for s in ("left", "right")}
        if self.state == "stream" and (self.t_msg is None or now - self.t_msg > self.watchdog_s):
            self.state = "hold"
            self.t_hold = now
            self.ended_by = "watchdog"
            self.stats["watchdog_trips"] += 1
            self.session["watchdog_trips"] += 1
            self.log(f"[arm] {self.op_id} watchdog: no message for {self.watchdog_s:.2f} s -> hold {self.hold_s:.1f} s")
        if self.state == "hold" and now - self.t_hold >= self.hold_s:
            self._start_blend(now, "watchdog")
        vel = [0.0] * N
        if self.state == "stream":
            des = list(self.target)
            if self.servo_ki > 0.0:
                self._servo(now, dt)
                for k in ARM_IDX:
                    lo, hi = _LIM[k]
                    des[k] = min(max(des[k] + self.corr[k], lo), hi)
            if self.vel_mode in ("est", "cmd") and self.target_vel is not None:
                vel = list(self.target_vel)
        elif self.state == "hold":
            des = list(self.sent)
            hands_des = dict(self.hands_sent)
        else:  # blend
            a = _minjerk((now - self.t_blend) / self.blend_s)
            des = [(1 - a) * f + a * r for f, r in zip(self.blend_from, ref)]
            for s in ("left", "right"):
                f = self.blend_hands_from[s]
                if f is not None:
                    d = jm.DEX3_CLOSED[s]
                    hands_des[s] = [(1 - a) * x + a * y for x, y in zip(f, d)]
            if a >= 1.0:
                self.mux.clear_upper()
                self.log(f"[arm] {self.op_id} blended back; override released")
                self._finish("canceled" if self.ended_by == "stop" else "succeeded", {"reason": self.ended_by})
                self.state = "off"
                self.owner = None
                self.sent = None
                self.hands_sent = {"left": None, "right": None}
                return
        if self.waist_mode == "ref":
            for k in WAIST_IDX:
                des[k] = ref[k]
        # slew limit (protects against jumps; GR00T chunks are smooth, 1 Hz x 0.3 rad needs only 1.9 rad/s)
        step = self.max_vel * dt
        prev = self.sent if self.sent is not None else des
        out, limited = [], False
        for p_, d_ in zip(prev, des):
            dd = d_ - p_
            if dd > step:
                dd, limited = step, True
            elif dd < -step:
                dd, limited = -step, True
            out.append(p_ + dd)
        if limited:
            self.stats["slew_limited_ticks"] += 1
            self.session["slew_limited_ticks"] += 1
            vel = [(o - p_) / dt for o, p_ in zip(out, prev)] if self.vel_mode != "zero" else vel
        if self.state != "stream":
            vel = [0.0] * N
        self.sent = out
        self.hands_sent = {s: (list(hands_des[s]) if hands_des[s] is not None else None) for s in ("left", "right")}
        if self.hands_sent["left"] is not None or self.hands_sent["right"] is not None:
            # always both: the deploy keeps the last value it ever received per hand and would reuse a stale one
            # for a hand that is missing from the message (input_interface.hpp:341-362)
            for s in ("left", "right"):
                if self.hands_sent[s] is None:
                    self.hands_sent[s] = list(jm.DEX3_CLOSED[s])
        self.mux.set_upper(jm.wire_from_mj17(out), jm.wire_from_mj17(vel),
                           self.hands_sent["left"], self.hands_sent["right"])

    def _servo(self, now: float, dt: float) -> None:
        """Integral outer loop on the measured arm joints (g1_debug body_q). The error is taken against the target
        `servo_delay_s` ago (SONIC's measured lag, docs/arm_tracking.md), so the integrator removes the steady-state
        bias without chasing the tracking delay. Frozen while g1_debug is stale; |corr| <= servo_max; anti-windup at
        the joint limits (the corrected target is clamped there and the integrator stops pushing outwards)."""
        # true elapsed time: tick() also runs on every message, so the nominal-period dt would double-count
        dt = 0.0 if self._t_servo is None else min(0.05, max(0.0, now - self._t_servo))
        self._t_servo = now
        if self.deploy.age_s() > 0.1 or dt <= 0.0:
            return
        qm = self.measured_mj17()
        if qm is None or not self._hist:
            return
        t_d = now - self.servo_delay_s
        tgt_d = self._hist[0][1]
        for t_h, v in self._hist:
            if t_h <= t_d:
                tgt_d = v
            else:
                break
        for k in ARM_IDX:
            e = tgt_d[k] - qm[k]
            c = self.corr[k] + self.servo_ki * dt * e
            c = min(max(c, -self.servo_max), self.servo_max)
            lo, hi = _LIM[k]
            if (self.target[k] + c > hi and c > self.corr[k]) or (self.target[k] + c < lo and c < self.corr[k]):
                continue
            self.corr[k] = c

    def progress(self) -> dict:
        return self.snapshot()

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {"state": self.state, "owner": self.owner, "op": self.op_id,
                "msg_age_s": None if self.t_msg is None else round(now - self.t_msg, 3),
                "waist": self.waist_mode, "vel": self.vel_mode, "watchdog_s": self.watchdog_s,
                "hold_s": self.hold_s, "blend_s": self.blend_s, "max_vel": self.max_vel,
                "servo": {"ki": self.servo_ki, "delay_s": self.servo_delay_s, "max": self.servo_max,
                          "corr_max_abs": round(max(abs(c) for c in self.corr), 4)},
                "hands": [s for s in ("left", "right") if self.hands_sent.get(s) is not None],
                "sent_mj17": None if self.sent is None else [round(v, 4) for v in self.sent],
                "stats": dict(self.stats)}


def _brief_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in ("upper_body", "upper_body_vel", "left_hand", "right_hand")}
