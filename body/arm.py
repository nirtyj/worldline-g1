"""Arm/hand channel: streams upper-body and Dex3 joint targets INTO SONIC through the planner message.

Ops served here (BodyService routes them; the handlers all take `(op_id, args, can_start)` like `handle`):
    arm          handle()             v0.5 single-target stream (mode "target", the default) and B.8 chunk mode
                                      (mode "chunk", docs/contracts/arm_chunk.md v0.1, deviations in docs/arm_tracking.md
                                      §8.4)
    arm_script   handle_arm_script()  B.7 scripted pregrasp/grasp/lift/carry/lower/release/retract (body/arm_script.py)
    scan         handle_scan()        B.5 waist-yaw scan (body/scan.py)
and the halt-lane interface (B.1, called by BodyService's halt lane):
    latch(epoch, reason) -> {latched, arms: held|free, owner, ...}  lock-free: freeze the wire, queue the latch
    unlatch(epoch)                                                  resume: the arms as the halt found them
    state() -> {mode: off|stream|chunk|hold|blend|latched|script, arms: held|free, owner, session_id, op_id, ...}

Owner decision 2026-09-29 (recommendation (b), docs/arena_vs_sonic.md §4.2): GR00T drives the arms and hands by
joint targets; SONIC stays the only body controller and keeps the legs and balance. The targets ride on every
planner message SonicMux sends (`upper_body_position[17]`, `upper_body_velocity[17]`, `left/right_hand_joints[7]`,
zmq_manager.hpp:885-980), so the channel composes with whatever motion owns the legs (IDLE hold, walk, velocity,
turn_to, go_to): motions set mode/movement/facing/speed, this channel sets the upper-body overlay.

What the deploy does with them (WBC @ b042411):
- Every consumed planner message sets `has_upper_body_control_ = upper_body_position present` and
  `has_hand_joints_ = hand fields present` (zmq_manager.hpp:581-596); a message without them releases the override
  at once. After 1 s without any planner message both are cleared (:618-628).
- The 17 targets REPLACE the upper-body joints of all 10 future reference frames the encoder sees
  (g1_deploy_onnx_ref.cpp:738-800, 861-867). SONIC then *tracks* them with its whole-body policy: there is no direct
  PD path for the arms, so tracking error and lag are the policy's (docs/arm_tracking.md). `upper_body_velocity`
  replaces the reference joint velocities the same way; the channel always sends it (zeros unless `vel` says
  otherwise), so a stale velocity from an earlier client can never persist (input_interface.hpp:380-390).
- Hands go straight to rt/dex3/<side>/cmd with kp 1.5 / kd 0.1, clipped to `max_close_ratio` and to +-0.25 rad from
  the measured position per write (dex3_hands.hpp:114-190). Without hand fields the deploy commands its default
  fist (input_interface.hpp:333-362), and it reuses the last value for a hand missing from a message, so both hands
  are always sent together.

Channel phases (`state().mode` in brackets):
- off                   no override.
- active [stream|chunk|script]   a session streams: a v0.5 target stream, a GR00T chunk session, or an internal
                        script (arm_script / scan).
- hold [hold|latched]   a fixed pose is re-sent on every planner message (body/carry.py `Hold`): a session's
                        `hold_on_end` ("target" = CarryLock after a grasp, "measured"), a v0.5 stream's watchdog hold,
                        or the halt latch.
- blend [blend]         min-jerk from the pose being sent to SONIC's own reference arms (g1_debug `body_q_target`,
                        the planner motion without the override) over `blend_s`, hands to the deploy's default fist;
                        then the override is dropped (seamless: the last override equals the reference).

Every 50 Hz tick (and ONLY the tick: a message just sets the target, so the slew limit is exactly `max_vel`
whatever the message rate) computes the pose, adds the servo correction, rate-limits it and hands it to the mux.

Ownership (one owner at a time), keyed by the message's `stream` id (contract m1.md §3.9, unchanged):
1. With nobody streaming (off, hold, blend) any stream may start a session (the robot must be standing and not
   faulted); it continues from the pose being sent (a take-over ends a v0.5 op still in hold/blend `canceled`,
   reason `taken_over`). The first message creates the op record; later messages are updates (reply only).
2. While a session streams, another stream is rejected `arm_busy` unless it sends `preempt: true`: the new stream
   takes over from the pose being sent, the old op ends `canceled` (ended_by `preempted`) and the old stream gets
   `arm_preempted` while the new owner streams.
3. `stop {arms: true}` (`end("stop")`) blends back and the stopped stream is rejected `arm_stopped` from then on,
   during and after the blend, until it sends `restart: true` or a new stream id is used (G0 defect D1).
4. The legs are never owned here.

Ends and terminal states (exactly one terminal event per op): client `end` -> succeeded (v0.5 target: after the
blend-back; `hold_on_end: target|measured` holds instead); watchdog (client silent) -> **failed**, reason
`client_silent` (G0: a crashed client must not look like success); `stop {arms}` -> canceled (reason `stop`); halt
-> canceled (ended_by `halt`); preempt / take-over -> canceled; fall -> failed (ended_by `fault`); a script that
ran to the end -> succeeded.

Halt latch (B.1; G0 defect D2; wave-2 fixes B-D1..B-D3): `latch(epoch)` runs on the halt lane's thread and never
takes the arm lock (only a wire lock held for microseconds): it sets the latch flag, freezes the override on the wire
at the pose being sent (no chunk row / script step after the ack) and queues the rest for the next tick (<= 20 ms),
which acts on who owns the arms: a session moving them ends (canceled, ended_by halt) into a `latched` hold = the
MEASURED arm pose (servo preloaded: no step on the wire), the waist as the session sent it and the hands' last
TARGET (never the measured q, never opened); a hold (CarryLock, a target / measured hold) stays exactly as it is; a
blend back to SONIC pauses; SONIC's free arms stay free (no override is added). While latched every `arm` /
`arm_script` / `scan` message is rejected `halted`; any message whose control_epoch <= halt_epoch is rejected `halted`
for good. `unlatch(epoch)` gives the arms back as the halt found them (free stays free, the same hold, the blend
continues); a `latched` hold stays until a new owner with control_epoch > halt_epoch takes it over, or `end` /
`release` / `stop {arms}` blends it back to SONIC. latch() does no I/O: events are published by the next tick.

Servo (default on, `servo_ki` 2.0; 0 turns it off): SONIC tracks the override with a pose-dependent steady-state
error of 0.1-0.3 rad on some joints and ~150 ms of lag (docs/arm_tracking.md). The channel closes an integral loop
on the measured arm joints (g1_debug body_q; the 14 arm joints, plus waist yaw during a scan):
    corr += w * ki * dt * (y_ref - q_measured),  |corr| <= servo_max (0.4 rad),  sent = target + corr
`servo_model` picks y_ref:
    "gated" (default)  y_ref = SONIC's nominal response to the target (dead time `servo_dead_s` 0.09 s, then first
                       order `servo_tau_s` 0.085 s: G0's step dead time / t50), and w = exp(-|dy_ref/dt| / servo_v0)
                       with servo_v0 = 0.3 rad/s, so the integrator only learns while the reference is (nearly)
                       still. This removes the step overshoot of the G0 servo (the integrator wound up during every
                       transient) without losing the static accuracy (docs/arm_tracking.md §8.2).
    "delay"            G0: y_ref = the target `servo_delay_s` (0.15 s) ago, w = 1.
The lag itself is not removed: a client that knows its future targets (a GR00T chunk) plays them ~0.15 s ahead
(chunk `lead_s`).

Waist: the 17-vector includes waist yaw/roll/pitch. By default (`waist: "ref"`) the channel sends SONIC's own
reference waist and only the 14 arm joints are the client's. `waist: "cmd"` sends the client's waist values instead
(default when a v0.5 client names waist joints). Scans command waist yaw only (G0: SONIC does not move waist pitch).
"""

from __future__ import annotations

import collections
import math
import threading
import time
from typing import Any, Callable, Sequence

from . import joint_map as jm
from .carry import Hold, carry_info
from .ik_worker import IKWorker

N = jm.N_UPPER
WAIST_IDX = (0, 1, 2)              # in UPPER_BODY_MUJOCO_JOINTS (mj17) order
YAW_IDX = 0
ARM_IDX = tuple(range(3, 17))
SIDES = ("left", "right")
_LIM = [(jm.JOINT_LIMITS[n][0] + 0.02, jm.JOINT_LIMITS[n][1] - 0.02) for n in jm.UPPER_BODY_MUJOCO_JOINTS]
_MJ17_IDX = {n: k for k, n in enumerate(jm.UPPER_BODY_MUJOCO_JOINTS)}
_HAND_LIM = {s: [jm.JOINT_LIMITS[n] for n in jm.HAND_JOINTS[s]] for s in SIDES}

MODES = ("target", "chunk")                      # arm op modes (body.state.arm.modes; groot_arms checks "chunk")
OPS = ("arm", "arm_script", "scan")              # ops whose handlers live here
HOLD_ON_END = ("target", "measured", "stand")
CHUNK_VALUES = 28                                # 14 arm + 14 hand values per tick (the waist is the channel's)
KIND_MODE = {"target": "stream", "chunk": "chunk", "script": "script"}


class ArmError(Exception):
    def __init__(self, reason: str, data: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.data = data or {}


def _minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _lerp(a: Sequence[float], b: Sequence[float], w: float) -> list[float]:
    return [x + (y - x) * w for x, y in zip(a, b)]


def _vec17(v, base: Sequence[float], what: str) -> tuple[list[float], bool]:
    """Parse a dict {joint: q} (overlaid on base) or a 17-list in UPPER_BODY_MUJOCO_JOINTS order.
    Returns (mj17, waist_given)."""
    if isinstance(v, dict):
        try:
            out = jm.mj17_from_named({str(k): float(x) for k, x in v.items()}, base)
        except KeyError as e:
            raise ArmError("bad_args", {"arg": what, "unknown_joint": str(e.args[0]),
                                        "joints": list(jm.UPPER_BODY_MUJOCO_JOINTS)})
        except (TypeError, ValueError):
            raise ArmError("bad_args", {"arg": what, "error": "joint values must be numbers"})
        waist = any(k in v for k in jm.WAIST_JOINTS)
    elif isinstance(v, (list, tuple)):
        if len(v) != N:
            raise ArmError("bad_args", {"arg": what, "error": f"need 17 values in order {jm.UPPER_BODY_MUJOCO_JOINTS}",
                                        "got": len(v)})
        try:
            out = [float(x) for x in v]
        except (TypeError, ValueError):
            raise ArmError("bad_args", {"arg": what, "error": "joint values must be numbers"})
        waist = True
    else:
        raise ArmError("bad_args", {"arg": what, "error": "dict {joint: rad} or a 17-list"})
    if not all(math.isfinite(x) for x in out):
        raise ArmError("bad_args", {"arg": what, "error": "non-finite value"})
    return out, waist


def _hand(v, side: str, what: str) -> list[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return jm.hand_closure(side, float(v))
    try:
        if isinstance(v, (list, tuple)) and len(v) == jm.N_HAND and all(math.isfinite(float(x)) for x in v):
            q, _ = jm.clamp_hand(side, [float(x) for x in v])
            return q
    except (TypeError, ValueError):
        pass
    raise ArmError("bad_args", {"arg": what, "error": "7 Dex3 values (thumb_0..index_1) or a closure 0..1"})


def _num(args: dict, key: str, lo: float, hi: float, default: float) -> float:
    v = args.get(key)
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ArmError("bad_args", {"arg": key, "error": "number"})
    if isinstance(v, bool) or not math.isfinite(f):
        raise ArmError("bad_args", {"arg": key, "error": "finite number"})
    return min(max(f, lo), hi)


def _int(args: dict, key: str) -> int | None:
    v = args.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise ArmError("bad_args", {"arg": key, "error": "integer"})
    return int(v)


def _fill_hands(h: dict) -> dict:
    """Both hands or none: the deploy reuses the last value it received for a hand missing from a message
    (input_interface.hpp:341-362), so a missing hand is sent as the deploy's default fist (what it would command)."""
    if h.get("left") is None and h.get("right") is None:
        return {"left": None, "right": None}
    return {s: (list(h[s]) if h.get(s) is not None else list(jm.DEX3_CLOSED[s])) for s in SIDES}


def _brief_args(args: dict) -> dict:
    return {k: v for k, v in args.items()
            if k not in ("upper_body", "upper_body_vel", "left_hand", "right_hand", "chunk")}


# ------------------------------------------------------------------------------------------------ chunks
class _Chunk:
    """One GR00T action chunk in mj17 order (arm_chunk.md §1.2). Row k belongs to t0 + k*dt."""

    __slots__ = ("seq", "t0", "dt", "T", "ub", "lh", "rh", "waist", "inference_ms")

    def __init__(self, seq, t0, dt, ub, lh, rh, waist, inference_ms):
        self.seq, self.t0, self.dt = seq, t0, dt
        self.ub, self.lh, self.rh = ub, lh, rh
        self.T = len(ub)
        self.waist = waist
        self.inference_ms = inference_ms

    def at(self, t: float) -> tuple[list[float], list[float], list[float]]:
        """Linear interpolation between the two rows around t; the first / last row outside the chunk."""
        x = (t - self.t0) / self.dt
        if x <= 0.0:
            return self.ub[0], self.lh[0], self.rh[0]
        if x >= self.T - 1:
            return self.ub[-1], self.lh[-1], self.rh[-1]
        i = int(x)
        f = x - i
        return _lerp(self.ub[i], self.ub[i + 1], f), _lerp(self.lh[i], self.lh[i + 1], f), \
            _lerp(self.rh[i], self.rh[i + 1], f)

    def k(self, t: float) -> int:
        return int(min(max(round((t - self.t0) / self.dt), 0), self.T - 1))

    def stall(self, t: float) -> float:
        return max(0.0, t - (self.t0 + (self.T - 1) * self.dt))


class _Base:
    """What a chunk session plays before its first chunk: the pose it started from (continuity) and the start hands
    blended in over hands_blend_s. Also the 'old chunk' the first chunk cross-fades from."""

    def __init__(self, pose, hands0: dict, hands1: dict, t0: float, dur: float):
        self.pose, self.hands0, self.hands1, self.t0, self.dur = list(pose), hands0, hands1, t0, dur

    def hands_at(self, t: float) -> dict:
        a = 1.0 if self.dur <= 0 else min(1.0, max(0.0, (t - self.t0) / self.dur))
        out = {}
        for s in SIDES:
            h0, h1 = self.hands0.get(s), self.hands1.get(s)
            if h1 is None:
                out[s] = h0
            else:
                out[s] = _lerp(h0 if h0 is not None else jm.DEX3_CLOSED[s], h1, a)
        return out

    def at(self, t: float):
        h = self.hands_at(t)
        return self.pose, h["left"], h["right"]


def _parse_chunk(ch) -> _Chunk:
    if not isinstance(ch, dict):
        raise ArmError("bad_chunk", {"error": "chunk must be an object"})
    need = ("seq", "t0_mono", "dt", "order", "upper_body", "left_hand", "right_hand")
    missing = [k for k in need if ch.get(k) is None]
    if missing:
        raise ArmError("bad_chunk", {"error": "missing fields", "missing": missing})
    if isinstance(ch["seq"], bool) or not isinstance(ch["seq"], (int, float)) or float(ch["seq"]) != int(ch["seq"]):
        raise ArmError("bad_chunk", {"arg": "seq", "error": "integer"})
    seq = int(ch["seq"])
    try:
        t0, dt = float(ch["t0_mono"]), float(ch["dt"])
    except (TypeError, ValueError):
        raise ArmError("bad_chunk", {"error": "t0_mono and dt must be numbers"})
    if not (math.isfinite(t0) and math.isfinite(dt) and 0.005 <= dt <= 0.1):
        raise ArmError("bad_chunk", {"arg": "dt", "error": "0.005 <= dt <= 0.1, finite t0_mono", "dt": dt})
    order = ch["order"]
    if order not in ("wire", "mj17"):
        raise ArmError("bad_chunk", {"arg": "order", "error": 'required: "wire" or "mj17"'})
    waist = ch.get("waist") or "ref"
    if waist not in ("ref", "cmd"):
        raise ArmError("bad_chunk", {"arg": "waist", "error": "ref | cmd"})
    ub, lh, rh = ch["upper_body"], ch["left_hand"], ch["right_hand"]
    if not all(isinstance(a, (list, tuple)) for a in (ub, lh, rh)):
        raise ArmError("bad_chunk", {"error": "upper_body, left_hand, right_hand must be arrays of rows"})
    T = len(ub)
    if not (1 <= T <= 64) or len(lh) != T or len(rh) != T:
        raise ArmError("bad_chunk", {"error": "1 <= T <= 64 rows, the same T for upper_body and both hands",
                                     "T": [len(ub), len(lh), len(rh)]})
    try:
        rows = [[float(x) for x in r] for r in ub]
        L = [[float(x) for x in r] for r in lh]
        R = [[float(x) for x in r] for r in rh]
    except (TypeError, ValueError):
        raise ArmError("bad_chunk", {"error": "rows must be arrays of numbers"})
    if any(len(r) != N for r in rows) or any(len(r) != jm.N_HAND for r in L + R):
        raise ArmError("bad_chunk", {"error": "rows: upper_body 17, hands 7 values"})
    if not all(math.isfinite(x) for r in rows + L + R for x in r):
        raise ArmError("bad_chunk", {"error": "non-finite value"})
    if order == "wire":
        rows = [jm.mj17_from_wire(r) for r in rows]
    inf = ch.get("inference_ms")
    try:
        inf = None if inf is None else round(float(inf), 1)
    except (TypeError, ValueError):
        inf = None
    return _Chunk(seq, t0, dt, rows, L, R, waist, inf)


# ------------------------------------------------------------------------------------------------ sessions
class _Session:
    """One owner's session: a v0.5 target stream, a chunk session or a script. Created by the first message."""

    def __init__(self, kind: str, op_id: str, stream: str, fence: dict, now: float):
        self.kind, self.op_id, self.stream = kind, op_id, stream
        self.session_id = fence.get("session_id")
        self.generation = fence.get("generation")
        self.control_epoch = fence.get("control_epoch")
        self.execution_id = fence.get("execution_id")
        self.t_start = self.t_msg = self.t_progress = now
        self.watchdog_s, self.hold_s, self.blend_s, self.max_vel = 0.3, 1.0, 1.5, 6.0
        self.hold_on_end: str | None = None
        self.waist_mode, self.vel_mode = "ref", "zero"
        self.ended_by: str | None = None
        self.t_hold: float | None = None
        self.counters = {"messages": 0, "watchdog_trips": 0, "clamped": 0, "slew_limited_ticks": 0}
        self.last_ref: list[float] | None = None         # the last pose before servo / slew (a "target" hold keeps it)
        self.last_hands: dict = {"left": None, "right": None}
        self.last_waist = "ref"
        self.waist_now = "ref"                             # the waist mode of the last tick: ref | cmd | yaw
        self.carry_prev: str | None = None                # CarryLock arm of the hold this session took over
        self.max_step = 0.0                               # rad: the largest per-tick change of a sent arm value
        # target (v0.5)
        self.target: list[float] | None = None
        self.target_vel: list[float] | None = None
        self.prev_t: tuple[float, list[float]] | None = None
        self.hands: dict = {"left": None, "right": None}
        # chunk (B.8)
        self.lead_s, self.xfade = 0.15, 5
        self.base: _Base | None = None
        self.chunk: _Chunk | None = None
        self.old: Any = None
        self.xf_i = 0
        self.chunks = {"received": 0, "applied": 0,
                       "dropped": {"expired": 0, "out_of_order": 0, "invalid": 0, "stale_session": 0, "halted": 0}}
        self.cross_fades = 0
        self.latency_ms: float | None = None
        self.inference_ms: float | None = None
        self.k: int | None = None
        self.stall_s = self.stall_max = 0.0
        self.win = [0, 0, 0]                              # clamped values, slew-limited values, ticks (progress window)
        self.tot = [0, 0, 0]                              # the same since the session start
        # script
        self.plan: Any = None

    def fence(self) -> dict:
        return {"session_id": self.session_id, "generation": self.generation, "control_epoch": self.control_epoch,
                "execution_id": self.execution_id}

    def session_brief(self) -> dict:
        c = self.chunk
        return {"session_id": self.session_id, "chunk_seq": 0 if c is None else c.seq, "k": self.k,
                "T": None if c is None else c.T, "stall_s": round(self.stall_s, 3)}


class ScriptPending:
    """An `arm_script` whose IK is in the worker. BodyService keeps the ROUTER envelope and calls poll() every loop
    (and at most `timeout_s` later answers `ik_timeout`); poll() returns None until the reply is ready."""

    timeout_s = 5.0

    def __init__(self, ch: "ArmChannel", op_id: str, args: dict, fence: dict, prep: dict, fut, t0: float):
        self.ch, self.op_id, self.args, self.fence, self.prep, self.fut, self.t0 = ch, op_id, args, fence, prep, fut, t0

    def finish(self, can_start) -> dict:
        return self.ch._finish_script(self, can_start)

    def poll(self, can_start_fn: Callable[[], tuple]) -> dict | None:
        if self.fut.done():
            return self.finish(can_start_fn())
        if self.ch.clock() - self.t0 > self.timeout_s:
            self.fut.cancel()
            return {"ok": False, "state": "rejected", "error": "ik_timeout",
                    "data": {"waited_s": round(self.ch.clock() - self.t0, 2), "ik": self.ch.ik.snapshot()}}
        return None


class ArmChannel:
    """Owned by BodyService; handle*() run on the service thread for ops arm / arm_script / scan, tick() every
    control tick (50 Hz). latch()/unlatch()/state() may be called from any thread."""

    def __init__(self, cfg, mux, deploy, emit, log=print, record=None, pose: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic, ik: IKWorker | None = None):
        self.cfg = cfg
        self.mux = mux
        self.deploy = deploy
        self.emit = emit                # emit(op_id, state, data): publishes body.event (service thread only)
        self.record = record            # record(op_id, op, args): creates the op record
        self.log = log
        self.clock = clock
        self._pose_fn = pose            # () -> GT Pose (body.wire.Pose) or None; BodyService: pose_sub.latest
        self._pose_sub = None
        self._lock = threading.RLock()
        # the wire lock: held only for a few microseconds, around every write of the override to the mux (with the
        # `sent` / `hands_sent` / phase-off bookkeeping) and by latch(). latch() never takes self._lock, so a halt never
        # waits for a handler or a tick (B-D1); the tick checks for a queued latch before it commits a pose.
        self._wire_lock = threading.Lock()
        self._latch_req: dict | None = None               # queued by latch() (lane), applied by the next tick
        self.ik = ik if ik is not None else IKWorker("inline")
        self._pending: list[tuple[str, str, dict]] = []   # events queued until flush() on the service thread
        self._logq: list[str] = []
        self.sess: _Session | None = None
        self.phase = "off"              # off | active | hold | blend
        self.hold: Hold | None = None
        self.blend: dict | None = None
        self.latched = False
        self.halt_epoch: int | None = None
        self.latch_info: dict | None = None
        self.sent: list[float] | None = None                # mj17, last value sent (after servo + slew)
        self.hands_sent: dict = {"left": None, "right": None}
        self.preempted: collections.OrderedDict = collections.OrderedDict()
        self.stopped: collections.OrderedDict = collections.OrderedDict()
        self.ended_sessions: collections.OrderedDict = collections.OrderedDict()
        self.t_tick: float | None = None
        self.stats = {"messages": 0, "stale_dropped": 0, "watchdog_trips": 0, "resumed": 0, "clamped": 0,
                      "slew_limited_ticks": 0, "ticks": 0, "sessions": 0, "preemptions": 0, "rejected_busy": 0,
                      "takeovers": 0, "rejected_stopped": 0, "rejected_halted": 0, "stale_session": 0,
                      "latches": 0, "latches_applied": 0, "ticks_skipped_latch": 0, "chunks_applied": 0, "scripts": 0,
                      "scripts_async": 0, "max_step_rad": 0.0}
        self._defaults()
        self._reset_servo()

    # -- configuration ---------------------------------------------------------------------------------------
    def _cfg(self, name: str, default):
        return type(default)(getattr(self.cfg, name, default))

    def _defaults(self) -> None:
        self.sv = {"ki": self._cfg("arm_servo_ki", 2.0), "model": str(getattr(self.cfg, "arm_servo_model", "gated")),
                   "delay_s": self._cfg("arm_servo_delay_s", 0.15), "dead_s": self._cfg("arm_servo_dead_s", 0.09),
                   "tau_s": self._cfg("arm_servo_tau_s", 0.085), "v0": self._cfg("arm_servo_v0", 0.3),
                   "max": self._cfg("arm_servo_max", 0.4), "ki_waist": self._cfg("arm_servo_ki_waist", 5.0)}
        self.blend_s_default = self._cfg("arm_blend_s", 1.5)
        self.max_vel_default = self._cfg("arm_max_vel", 6.0)

    def _reset_servo(self) -> None:
        self.ref_last: list[float] | None = None            # mj17 servo reference of the last tick (pre-correction)
        self.corr = [0.0] * N
        self.corr_cap: list[float] | None = None            # per-joint |corr| bound after a preload (_preload)
        self._ym: list[float] | None = None
        self._hist: collections.deque = collections.deque(maxlen=120)     # (t, servo reference mj17), ~2.4 s

    # -- reference / measurement (g1_debug) ------------------------------------------------------------------
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

    def measured_hands(self) -> dict:
        d = self._debug()
        out = {}
        for s in SIDES:
            v = d.get(f"{s}_hand_q")
            out[s] = [float(x) for x in v] if v is not None and len(v) == jm.N_HAND else None
        return out

    def gt_pose(self, wait_s: float = 0.0):
        """Ground-truth pelvis pose (body.wire.Pose) from BodyService's pose subscriber when it was passed in
        (`pose=`), else from a subscriber of our own on gt.pose (started on first use)."""
        if self._pose_fn is not None:
            return self._pose_fn()
        if self._pose_sub is None:
            ports = getattr(self.cfg, "ports", None)
            if not ports or "p1_pose" not in ports:
                return None
            from .config import ep
            from .p1_client import PoseSub

            self._pose_sub = PoseSub(ep(ports["p1_pose"], getattr(self.cfg, "host", "127.0.0.1")))
            self._pose_sub.start()
        t_end = time.monotonic() + wait_s
        p = self._pose_sub.latest()
        while p is None and time.monotonic() < t_end:
            time.sleep(0.01)
            p = self._pose_sub.latest()
        return p

    def continuity_pose(self) -> list[float]:
        """What a new session starts from (its servo reference): the reference being played before the servo
        correction (the correction carries over a take-over, so starting from the corrected `sent` would count it
        twice), else the pose being sent (a blend: no correction), else SONIC's own reference (never zeros)."""
        if self.ref_last is not None and self.phase in ("active", "hold"):
            return list(self.ref_last)
        return list(self.sent) if self.sent is not None else self.reference_mj17()

    # -- events (queued; flushed on the service thread) ----------------------------------------------------------
    def _emit(self, op_id: str | None, state: str, data: dict) -> None:
        if op_id is not None:
            self._pending.append((op_id, state, data))

    def _say(self, msg: str) -> None:
        self._logq.append(msg)

    def flush(self) -> None:
        """Publish queued events and log lines. Service thread only (the event socket is not thread safe)."""
        with self._lock:
            ev, self._pending = self._pending, []
            lg, self._logq = self._logq, []
        for m in lg:
            try:
                self.log(m)
            except Exception:
                pass
        for op_id, state, data in ev:
            self.emit(op_id, state, data)

    @staticmethod
    def _bounded_add(d: collections.OrderedDict, key, value=True, cap: int = 500) -> None:
        d[key] = value
        d.move_to_end(key)
        while len(d) > cap:
            d.popitem(last=False)

    # -- API: op `arm` ---------------------------------------------------------------------------------------
    def handle(self, op_id: str, args: dict, can_start: tuple[bool, str | None, dict]) -> dict:
        """One `arm` message (mode "target" or "chunk"). Raises ArmError(bad_args | bad_chunk) before changing any
        state."""
        try:
            with self._lock:
                return self._handle(op_id, args or {}, can_start)
        finally:
            self.flush()

    def _rej(self, error: str, data: dict | None = None) -> dict:
        return {"ok": False, "state": "rejected", "error": error, "data": data or {}}

    def _fence_of(self, args: dict) -> dict:
        sid = args.get("session_id")
        if sid is not None and not isinstance(sid, str):
            raise ArmError("bad_args", {"arg": "session_id", "error": "string"})
        eid = args.get("execution_id")
        return {"session_id": sid, "generation": _int(args, "generation"),
                "control_epoch": _int(args, "control_epoch"), "execution_id": None if eid is None else str(eid)}

    def _halt_check(self, fence: dict, args: dict) -> dict | None:
        ce = fence["control_epoch"]
        if self.latched:
            self.stats["rejected_halted"] += 1
            return self._rej("halted", {"halt_epoch": self.halt_epoch, "latched": True,
                                        "hint": "the arms are latched by a halt until resume"})
        if self.halt_epoch is not None and ce is not None and ce <= self.halt_epoch:
            self.stats["rejected_halted"] += 1
            return self._rej("halted", {"halt_epoch": self.halt_epoch, "control_epoch": ce})
        if self.hold is not None and self.hold.kind == "latched" and ce is None \
                and not (args.get("end") or args.get("release")):
            self.stats["rejected_halted"] += 1
            return self._rej("halted", {"halt_epoch": self.halt_epoch, "hint": "after resume the latched pose is taken "
                                        "over only with control_epoch > halt_epoch, or ended with end/release"})
        return None

    def _session_fence(self, stream: str, mode: str, fence: dict, args: dict) -> dict | None:
        """The B.8 session fence (arm_chunk.md §4) on EVERY message that acts on a chunk session's stream: a chunk,
        a keepalive, an `end` or a `release`, with or without `mode: "chunk"` (wave 2: `release: true` and an `end`
        without the mode skipped it). Also on `release` / `end` of a hold that a chunk session left: they must come
        from that session (a newer session's hold is not an older one's to release). Any other message of an ended
        session is `stale_session`, so a message in flight when the runtime cancelled can never touch the arms."""
        s = self.sess
        sid, gen, ce = fence["session_id"], fence["generation"], fence["control_epoch"]
        acts = mode == "chunk" or bool(args.get("end") or args.get("keepalive") or args.get("release"))
        if s is not None and s.stream == stream and s.kind == "chunk":
            if not acts:
                return None                       # a v0.5 target message: mode_mismatch, below
            if sid is None:
                raise ArmError("bad_args", {"arg": "session_id", "error": "every message of a chunk session carries "
                                                                          "session_id, generation, control_epoch"})
            if sid != s.session_id or (gen is not None and gen < s.generation) or \
                    (ce is not None and ce < s.control_epoch):
                s.chunks["dropped"]["stale_session"] += 1
                self.stats["stale_session"] += 1
                return self._rej("stale_session", {"owner_session": s.session_id, "generation": s.generation,
                                                   "control_epoch": s.control_epoch})
            return None
        if sid is None:
            return None
        h = self.hold
        hf = h.fence if (h is not None and h.stream == stream and h.fence) else None
        if hf and hf.get("session_id") is not None and (s is None or s.stream != stream):
            if sid != hf["session_id"] or (gen is not None and hf.get("generation") is not None and
                                           gen < hf["generation"]) or \
                    (ce is not None and hf.get("control_epoch") is not None and ce < hf["control_epoch"]):
                self.stats["stale_session"] += 1
                return self._rej("stale_session", {"hold_session": hf["session_id"], "hold": h.kind})
            if args.get("release") or args.get("end"):
                return None                       # the session that left this hold releases it
        if (s is None or s.stream != stream or s.session_id != sid) and sid in self.ended_sessions:
            self.stats["stale_session"] += 1
            return self._rej("stale_session", {"ended": True, "ended_by": self.ended_sessions[sid]})
        return None

    def _handle(self, op_id: str, args: dict, can_start) -> dict:
        self._apply_latch()
        now = self.clock()
        stream = str(args.get("stream") or op_id)
        mode = args.get("mode") or "target"
        if mode not in MODES:
            raise ArmError("bad_args", {"arg": "mode", "error": "target | chunk"})
        fence = self._fence_of(args)
        rej = self._halt_check(fence, args)
        if rej is not None:
            return rej
        if mode == "chunk":
            missing = [k for k in ("session_id", "generation", "control_epoch") if fence[k] is None]
            if missing:
                raise ArmError("bad_args", {"arg": missing, "error": "chunk mode needs session_id, generation, "
                                                                     "control_epoch"})
        rej = self._session_fence(stream, mode, fence, args)
        if rej is not None:
            return rej
        if args.get("release"):
            return self._release(stream, args, now)
        s = self.sess
        own = s is not None and s.stream == stream
        if mode == "chunk":
            if own and s.kind != "chunk":
                return self._rej("mode_mismatch", {"session_mode": s.kind})
        elif own and s.kind != "target" and not (args.get("end") or args.get("keepalive")):
            return self._rej("mode_mismatch", {"session_mode": s.kind})
        if stream in self.stopped:
            if not args.get("restart"):
                self.stats["rejected_stopped"] += 1
                return self._rej("arm_stopped", {"hint": "stop {arms: true} ended this stream; send restart: true or "
                                                         "use a new stream id"})
            self.stopped.pop(stream, None)
        if args.get("end"):
            return self._end_msg(stream, args, now)
        if args.get("keepalive"):
            if not own:
                return self._rej("not_owner", {"owner": None if s is None else s.stream, "arm": self._mode()})
            s.t_msg = now
            return {"ok": True, "state": "done", "data": {"id": s.op_id, "arm": self._mode(),
                                                          **({"session": s.session_brief()} if s.kind == "chunk"
                                                             else {})}}
        if stream in self.preempted and not own:
            if self.phase == "active":      # told once it lost the arms; it may start again once they are free
                return self._rej("arm_preempted", {"owner": None if s is None else s.stream})
            self.preempted.pop(stream, None)
        if not own:
            ok, reason, info = can_start
            if not ok:
                self.emit_reject(op_id, args, reason, info)
                return {"ok": False, "state": "rejected", "error": reason, "data": info}
            if self.phase == "active" and not args.get("preempt"):
                self.stats["rejected_busy"] += 1
                return self._rej("arm_busy", {"owner": s.stream, "op": s.op_id, "kind": s.kind,
                                              "hint": "preempt: true takes over"})
        if mode == "chunk":
            return self._chunk_msg(op_id, stream, args, fence, now, own)
        return self._target_msg(op_id, stream, args, fence, now, own)

    # .. v0.5 target stream ......................................................................................
    def _target_msg(self, op_id, stream, args, fence, now, own) -> dict:
        s = self.sess
        base = s.target if (own and s.target is not None) else self.continuity_pose()
        p = self._parse_target(args, base)          # validate before touching any state
        wd = p.get("watchdog_s", s.watchdog_s if own else self._cfg("arm_watchdog_s", 0.3))
        if self._stale(args, wd):
            self.stats["stale_dropped"] += 1
            return self._rej("stale_command", self._stale_data(s.op_id if own else None, args, wd))
        if not own:
            s = self._begin("target", op_id, stream, args, fence, now)
            s.watchdog_s = self._cfg("arm_watchdog_s", 0.3)
            s.hold_s = self._cfg("arm_hold_s", 1.0)
            s.blend_s, s.max_vel = self.blend_s_default, self.max_vel_default
            new = True
        else:
            new = False
            if self.phase in ("hold", "blend"):    # the owner resumes from its watchdog hold / its end blend
                self.stats["resumed"] += 1
                self._say(f"[arm] {s.op_id} resumed from {self.phase}")
                self.phase, self.hold, self.blend = "active", None, None
                s.ended_by = None
        self._apply_target(s, p, args, now)
        return {"ok": True, "state": "accepted" if new else "done",
                "data": {"id": s.op_id, "arm": self._mode(), "stream": s.stream}}

    def _stale(self, args: dict, watchdog_s: float) -> bool:
        tw = args.get("t_wall")
        if tw is None:
            return False
        try:
            return time.time() - float(tw) > watchdog_s
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _stale_data(op_id, args: dict, watchdog_s: float) -> dict:
        """`stale_command` for a message older than the watchdog. `why: "t_wall"` tells it apart from the fences'
        stale_command (why: control_epoch | generation | resume_epoch, body/fence.py)."""
        try:
            age = round(time.time() - float(args.get("t_wall")), 3)
        except (TypeError, ValueError):
            age = None
        return {"why": "t_wall", "id": op_id, "t_wall": args.get("t_wall"), "age_s": age, "watchdog_s": watchdog_s}

    def _servo_args(self, args: dict, out: dict) -> None:
        for k, lo, hi in (("servo_ki", 0.0, 10.0), ("servo_ki_waist", 0.0, 10.0), ("servo_delay_s", 0.0, 0.5),
                          ("servo_max", 0.0, 0.8),
                          ("servo_dead_s", 0.0, 0.5), ("servo_tau_s", 0.01, 1.0), ("servo_v0", 0.01, 10.0)):
            if args.get(k) is not None:
                out[k] = _num(args, k, lo, hi, 0.0)
        m = args.get("servo_model")
        if m is not None:
            if m not in ("gated", "delay", "fopdt"):
                raise ArmError("bad_args", {"arg": "servo_model", "error": "gated | delay | fopdt"})
            out["servo_model"] = m

    def _apply_servo_args(self, p: dict) -> None:
        for k, sk in (("servo_ki", "ki"), ("servo_ki_waist", "ki_waist"), ("servo_delay_s", "delay_s"),
                      ("servo_max", "max"), ("servo_dead_s", "dead_s"),
                      ("servo_tau_s", "tau_s"), ("servo_v0", "v0"), ("servo_model", "model")):
            if k in p:
                self.sv[sk] = p[k]

    def _parse_target(self, args: dict, base: list[float]) -> dict:
        out: dict = {}
        ub = args.get("upper_body")
        if ub is not None:
            out["target"], out["waist_given"] = _vec17(ub, base, "upper_body")
        ubv = args.get("upper_body_vel")
        if ubv is not None:
            out["vel"], _ = _vec17(ubv, [0.0] * N, "upper_body_vel")
        for side in SIDES:
            v = args.get(f"{side}_hand")
            if v is not None:
                out[side] = _hand(v, side, f"{side}_hand")
        for k, lo, hi in (("watchdog_s", 0.05, 2.0), ("hold_s", 0.0, 30.0), ("blend_s", 0.2, 10.0),
                          ("max_vel", 0.5, 20.0)):
            if args.get(k) is not None:
                out[k] = _num(args, k, lo, hi, 0.0)
        self._servo_args(args, out)
        w = args.get("waist")
        if w is not None and w not in ("ref", "cmd"):
            raise ArmError("bad_args", {"arg": "waist", "error": "ref | cmd"})
        vm = args.get("vel")
        if vm is not None and vm not in ("zero", "est", "cmd"):
            raise ArmError("bad_args", {"arg": "vel", "error": "zero | est | cmd"})
        return out

    def _apply_target(self, s: _Session, p: dict, args: dict, now: float) -> None:
        for k in ("watchdog_s", "hold_s", "blend_s", "max_vel"):
            if k in p:
                setattr(s, k, p[k])
        self._apply_servo_args(p)
        if args.get("vel") is not None:
            s.vel_mode = args["vel"]
        elif "vel" in p:
            s.vel_mode = "cmd"
        if "target" in p:
            if args.get("waist") is not None:
                s.waist_mode = args["waist"]
            elif p["waist_given"] and isinstance(args.get("upper_body"), dict):
                s.waist_mode = "cmd"
            tgt, n = jm.clamp_mj17(p["target"], margin=0.02)
            if n:
                self.stats["clamped"] += 1
                s.counters["clamped"] += 1
            t = float(args["t_wall"]) if args.get("t_wall") is not None else time.time()
            if "vel" in p:
                s.target_vel = p["vel"]
            elif s.prev_t is not None and t - s.prev_t[0] > 1e-3:
                dt = t - s.prev_t[0]
                raw = [(a - b) / dt for a, b in zip(tgt, s.prev_t[1])]
                s.target_vel = raw if (s.target_vel is None or dt > 0.2) else \
                    [0.5 * a + 0.5 * b for a, b in zip(raw, s.target_vel)]
            else:
                s.target_vel = [0.0] * N
            s.prev_t = (t, tgt)
            s.target = tgt
        elif s.target is None:
            s.target = self.continuity_pose()
        for side in SIDES:
            if side in p:
                s.hands[side] = p[side]
        s.t_msg = now
        self.stats["messages"] += 1
        s.counters["messages"] += 1

    # .. B.8 chunk session ........................................................................................
    def _chunk_msg(self, op_id, stream, args, fence, now, own) -> dict:
        s = self.sess
        chunk = None
        if args.get("chunk") is not None:
            try:
                chunk = _parse_chunk(args["chunk"])
            except ArmError:
                if own:
                    s.chunks["received"] += 1
                    s.chunks["dropped"]["invalid"] += 1
                raise
        p: dict = {}
        if not own:
            p = self._parse_chunk_start(args)
        wd = p.get("watchdog_s", s.watchdog_s if own else 2.0)
        if self._stale(args, wd):
            self.stats["stale_dropped"] += 1
            return self._rej("stale_command", self._stale_data(s.op_id if own else None, args, wd))
        new = not own
        if new:
            start_pose = self.continuity_pose()
            hands0 = dict(self.hands_sent)
            s = self._begin("chunk", op_id, stream, args, fence, now)
            s.watchdog_s, s.lead_s, s.xfade = p["watchdog_s"], p["lead_s"], p["xfade"]
            s.hold_on_end, s.max_vel, s.blend_s = p["hold_on_end"], p["max_vel"], p["blend_s"]
            self._apply_servo_args(p)
            s.base = _Base(start_pose, hands0, p["hands"], now, p["hands_blend_s"])
        s.t_msg = now
        s.counters["messages"] += 1
        self.stats["messages"] += 1
        data: dict = {"id": s.op_id}
        if chunk is not None:
            why = self._apply_chunk(s, chunk, now)
            if why:
                data["dropped"] = why
        data["session"] = s.session_brief()
        data["arm"] = self._mode()
        return {"ok": True, "state": "accepted" if new else "done", "data": data}

    def _parse_chunk_start(self, args: dict) -> dict:
        out = {"watchdog_s": _num(args, "watchdog_s", 0.5, 5.0, 2.0),
               "lead_s": _num(args, "lead_s", 0.0, 0.3, self._cfg("arm_lead_s", 0.15)),
               "hands_blend_s": _num(args, "hands_blend_s", 0.0, 5.0, 0.3),
               "max_vel": _num(args, "max_vel", 0.5, 20.0, self.max_vel_default),
               "blend_s": _num(args, "blend_s", 0.2, 10.0, self.blend_s_default)}
        xf = _int(args, "xfade_ticks")
        out["xfade"] = 5 if xf is None else min(max(xf, 1), 50)
        hoe = args.get("hold_on_end") or "measured"
        if hoe not in HOLD_ON_END:
            raise ArmError("bad_args", {"arg": "hold_on_end", "error": " | ".join(HOLD_ON_END)})
        out["hold_on_end"] = hoe
        out["hands"] = {s: (_hand(args[f"{s}_hand"], s, f"{s}_hand") if args.get(f"{s}_hand") is not None else None)
                        for s in SIDES}
        self._servo_args(args, out)
        return out

    def _apply_chunk(self, s: _Session, c: _Chunk, now: float) -> str | None:
        s.chunks["received"] += 1
        cur = s.chunk
        if cur is not None and (c.seq <= cur.seq or c.t0 < cur.t0):
            s.chunks["dropped"]["out_of_order"] += 1
            return "out_of_order"
        if round((now + s.lead_s - c.t0) / c.dt) >= c.T:
            s.chunks["dropped"]["expired"] += 1
            return "expired"
        s.old = cur if cur is not None else s.base
        if cur is not None:
            s.cross_fades += 1
        s.xf_i = 0
        s.chunk = c
        s.chunks["applied"] += 1
        self.stats["chunks_applied"] += 1
        s.latency_ms = round((now - c.t0) * 1000.0, 1)
        s.inference_ms = c.inference_ms
        return None

    # .. begin / end ..............................................................................................
    def _begin(self, kind: str, op_id: str, stream: str, args: dict, fence: dict, now: float,
               op_name: str = "arm") -> _Session:
        prev = self.sess
        continuing = self.phase != "off"
        if prev is not None:
            if self.phase == "active":
                self.stats["preemptions"] += 1
                if prev.kind != "script":
                    self._bounded_add(self.preempted, prev.stream, cap=100)
                self._end_session(prev, "canceled", "preempted", "none", reason="preempted", extra={"by": stream})
            else:                           # a v0.5 op in its watchdog hold / end blend
                self.stats["takeovers"] += 1
                self._end_session(prev, "canceled", "taken_over", "none", reason="taken_over", extra={"by": stream})
        elif self.hold is not None or self.blend is not None:
            self.stats["takeovers"] += 1
        carry_prev = self.hold.carry_arm if (self.hold is not None and self.hold.kind == "target") else None
        self.hold = self.blend = None
        self._defaults()                    # per-session settings (v0.5); the servo state carries over a take-over
        if not continuing:
            self._reset_servo()
            self.sent = self.reference_mj17()
            self.hands_sent = {"left": None, "right": None}
        self.preempted.pop(stream, None)
        s = _Session(kind, op_id, stream, fence, now)
        s.carry_prev = carry_prev
        s.last_ref = list(self.sent)
        s.last_hands = dict(self.hands_sent)
        self.sess, self.phase = s, "active"
        self.stats["sessions"] += 1
        if self.record:
            self.record(op_id, op_name, _brief_args(args))
        self._say(f"[arm] session {op_id} {op_name}/{kind} stream={stream}"
                  f"{' session=' + str(s.session_id) if s.session_id else ''} start")
        self._emit(op_id, "accepted", {"args": _brief_args(args), "stream": stream, "kind": kind, **s.fence(),
                                       "start_mj17": [round(v, 4) for v in self.sent]})
        return s

    def emit_reject(self, op_id, args, reason, info, op_name: str = "arm") -> None:
        if self.record:
            self.record(op_id, op_name, _brief_args(args))
        self._emit(op_id, "failed", {"reason": reason, **(info or {})})

    def _terminal_data(self, s: _Session, ended_by: str, hold: str, reason: str | None, now: float) -> dict:
        d = {"stream": s.stream, "kind": s.kind, "ended_by": ended_by, "hold": hold, "reason": reason, **s.fence(),
             "duration_s": round(now - s.t_start, 3), **s.counters, "max_step_rad": round(s.max_step, 4)}
        if s.kind == "chunk":
            d.update({"chunks": {"received": s.chunks["received"], "applied": s.chunks["applied"],
                                 "dropped": dict(s.chunks["dropped"])},
                      "clamped_frac_total": round(s.tot[0] / (CHUNK_VALUES * s.tot[2]), 4) if s.tot[2] else 0.0,
                      "slew_frac_total": round(s.tot[1] / (CHUNK_VALUES * s.tot[2]), 4) if s.tot[2] else 0.0,
                      "stall_s_max": round(s.stall_max, 3), "cross_fades": s.cross_fades, "lead_s": s.lead_s,
                      "chunk_seq": 0 if s.chunk is None else s.chunk.seq})
        if s.kind == "script" and s.plan is not None:
            try:
                d.update(s.plan.result())
            except Exception as e:  # noqa: BLE001 - a result must never crash the loop
                d["result_error"] = repr(e)
        return d

    def _end_session(self, s: _Session, state: str, ended_by: str, hold: str, reason: str | None = None,
                     extra: dict | None = None, label: str | None = None) -> None:
        """End session s: exactly one terminal event, then the channel continues as `hold` says: target | measured
        (a Hold), stand (blend back, no op), none (the caller sets up what follows), drop (override off now)."""
        if s is not self.sess:
            return
        now = self.clock()
        self.sess = None
        data = self._terminal_data(s, ended_by, label or hold, reason, now)
        if extra:
            data.update(extra)
        if s.session_id:
            self._bounded_add(self.ended_sessions, s.session_id, ended_by)
        if ended_by == "stop" and s.kind != "script":
            self._bounded_add(self.stopped, s.stream, cap=100)
        self._say(f"[arm] {s.op_id} {state} (ended_by {ended_by}, hold {label or hold})")
        self._emit(s.op_id, state, data)
        if hold == "target":
            self.hold = Hold("target", list(s.last_ref if s.last_ref is not None else self.continuity_pose()),
                             dict(s.last_hands), s.last_waist, s.stream, s.op_id, f"{s.kind}_target", now,
                             carry_arm=self._carry_arm(s), fence=s.fence())
            self.phase, self.blend = "hold", None
        elif hold == "measured":
            qm = self.measured_mj17()
            src = "g1_debug.body_q"
            if qm is None or self.deploy.age_s() > 0.1:
                qm, src = self.continuity_pose(), "sent"
            pose, wmode = self._hold_waist(qm, s)
            self.hold = Hold("measured", pose, dict(s.last_hands), wmode, s.stream, s.op_id, src, now,
                             fence=s.fence())
            self.phase, self.blend = "hold", None
            self._preload(pose)
        elif hold == "stand":
            self._start_blend(now, ended_by, blend_s=s.blend_s)
        elif hold == "drop":
            self._drop()
        if s.kind == "script":
            self.corr[YAW_IDX] = 0.0                    # a scan's waist-yaw correction never outlives it

    def _hold_waist(self, arms: list[float], s: _Session | None) -> tuple[list[float], str]:
        """A measured / latched hold's pose: `arms` for the arm joints, and the waist exactly as the session was
        sending it (B-low: holding the MEASURED waist while the session sent SONIC's reference stepped the waist by
        0.12-0.14 rad at every arm/chunk halt). `ref` stays SONIC's live reference, `cmd` / `yaw` keep the values on the
        wire (servo correction included: the hold runs no waist servo)."""
        pose = list(arms)
        w = getattr(s, "waist_now", None) or (s.last_waist if s is not None else "ref")
        sent = self.sent if self.sent is not None else pose
        if w == "cmd":
            pose[0:3] = sent[0:3]
        elif w == "yaw":
            pose[YAW_IDX] = sent[YAW_IDX]
        else:
            w = "ref"
        return pose, w

    @staticmethod
    def _carry_arm(s: _Session) -> str | None:
        """Which hand a `target` hold carries with: the arm_script's arm, the hold a scan took over, else the hand the
        client itself commanded (chunk rows: both; v0.5: the ones it sent) that is closed the most (>= 0.3)."""
        if s.kind == "script":
            a = getattr(s.plan, "arm", None)
            return a if a in SIDES else s.carry_prev
        given = s.last_hands if s.kind == "chunk" else {k: v for k, v in s.hands.items() if v is not None}
        best, cb = None, 0.3
        for side, h in given.items():
            if h is not None:
                c = jm.hand_closure_of(side, h)
                if c >= cb:
                    best, cb = side, c
        return best

    def _start_blend(self, now: float, why: str, op_sess: _Session | None = None, blend_s: float | None = None) -> None:
        # from the pose on the wire (the correction is baked in there and the servo stops for the blend)
        frm = list(self.sent) if self.sent is not None else self.reference_mj17()
        self._reset_servo()
        self.phase = "blend"
        self.hold = None
        self.blend = {"from": frm, "hands": dict(self.hands_sent), "t0": now, "why": why,
                      "dur": float(blend_s if blend_s is not None else
                                   (op_sess.blend_s if op_sess is not None else self.blend_s_default)),
                      "sess": op_sess}
        if op_sess is not None:
            op_sess.ended_by = op_sess.ended_by or why
        self._say(f"[arm] {op_sess.op_id if op_sess else '-'} blend back to SONIC's reference ({why}, "
                  f"{self.blend['dur']:.2f} s)")

    def _preload(self, q_hold: list[float], idx: tuple = ARM_IDX) -> None:
        """A measured / latched hold makes the measured pose the servo reference. Preload the correction so the wire
        does not move at that instant (corr = sent - measured): a joint at rest stays exactly where it is (its
        unconverged tracking error is kept, not released), and a joint still moving towards the old target is pulled
        back to the halt point by the servo afterwards, without a step on the wire. The bound on |corr| starts at the
        preload and can only shrink back to servo_max."""
        if self.sent is None or float(self.sv["ki"]) <= 0.0:    # servo off: nothing would pull the arm back
            return
        mx = float(self.sv["max"])
        cap = list(self.corr_cap) if self.corr_cap is not None else [mx] * N
        for k in idx:
            c = min(max(self.sent[k] - q_hold[k], -0.8), 0.8)
            self.corr[k] = c
            cap[k] = max(mx, abs(c))
        self.corr_cap = cap
        self._ym = list(q_hold)
        self._hist.clear()

    def _drop(self) -> None:
        with self._wire_lock:                    # atomic against latch()'s freeze: never an override left behind
            self.mux.clear_upper()
            self.phase, self.hold, self.blend = "off", None, None
            self.sent = None
            self.hands_sent = {"left": None, "right": None}
        self._reset_servo()

    # .. end / release ............................................................................................
    def _end_msg(self, stream: str, args: dict, now: float) -> dict:
        s = self.sess
        hoe = args.get("hold_on_end")
        if hoe is not None and hoe not in HOLD_ON_END:
            raise ArmError("bad_args", {"arg": "hold_on_end", "error": " | ".join(HOLD_ON_END)})
        reason = args.get("reason")
        if s is not None and s.stream == stream:
            op = s.op_id
            if s.kind == "target" and hoe in (None, "stand"):
                # the client ended it: `succeeded` once blended back, also when the end comes during the watchdog's
                # hold or blend (B-low: that reported failed / client_silent although the client was there)
                s.ended_by = "client"
                if self.phase in ("active", "hold"):       # v0.5: blend back, the op ends succeeded afterwards
                    self._start_blend(now, "client", op_sess=s, blend_s=_num(args, "blend_s", 0.2, 10.0, s.blend_s))
                return {"ok": True, "state": "done", "data": {"id": op, "arm": self._mode(), "hold": "stand"}}
            hold = hoe or s.hold_on_end or ("stand" if s.kind == "target" else "measured")
            if hold == "stand" and args.get("blend_s") is not None:
                s.blend_s = _num(args, "blend_s", 0.2, 10.0, s.blend_s)
            self._end_session(s, "succeeded", "client", hold, reason=None if reason is None else str(reason)[:200])
            return {"ok": True, "state": "done", "data": {"id": op, "hold": hold, "arm": self._mode(),
                                                          **({"session_id": s.session_id} if s.session_id else {})}}
        if s is None and self.hold is not None and (self.hold.stream == stream or self.hold.kind == "latched"):
            return self._release(stream, args, now)
        return self._rej("not_owner", {"owner": None if s is None else s.stream, "arm": self._mode()})

    def _release(self, stream: str, args: dict, now: float) -> dict:
        s, h = self.sess, self.hold
        if s is not None and s.stream == stream:
            return self._end_msg(stream, {**args, "release": False, "end": True, "hold_on_end": "stand"}, now)
        if s is None and h is not None and (h.stream == stream or h.kind == "latched"):
            kind = h.kind
            self._start_blend(now, "release", blend_s=_num(args, "blend_s", 0.2, 10.0, self.blend_s_default))
            return {"ok": True, "state": "done", "data": {"released": kind, "arm": self._mode()}}
        return self._rej("not_owner", {"hold": None if h is None else h.kind,
                                       "hold_stream": None if h is None else h.stream, "arm": self._mode()})

    def end(self, reason: str) -> bool:
        """stop {arms: true}: blend back now (a hold too). Returns True if the arms were driven. While latched the
        latch wins (resume first)."""
        try:
            with self._lock:
                self._apply_latch()
                now = self.clock()
                if self.latched:
                    self._say(f"[arm] stop {{arms}} ignored: latched (halt epoch {self.halt_epoch})")
                    return False
                s = self.sess
                if s is not None:
                    if s.kind == "target":
                        if self.phase != "blend" or (self.blend and self.blend.get("why") != reason):
                            self._start_blend(now, reason, op_sess=s)
                        s.ended_by = reason
                        self._bounded_add(self.stopped, s.stream, cap=100)
                    else:
                        self._end_session(s, "canceled", reason, "stand", reason=reason)
                    return True
                if self.hold is not None:
                    self._start_blend(now, reason)
                    return True
                return self.phase == "blend"
        finally:
            self.flush()

    def abort(self, reason: str) -> None:
        """Fall / shutdown: drop the override immediately (a halt epoch fence stays)."""
        try:
            with self._lock:
                self._apply_latch()
                if self.phase == "off" and self.sess is None:
                    return
                self._say(f"[arm] abort ({reason}): override dropped")
                s = self.sess
                if s is not None:
                    fault = reason == "fallen"
                    self._end_session(s, "failed" if fault else "canceled", "fault" if fault else reason, "drop",
                                      reason=reason, label="none")
                self._drop()
        finally:
            self.flush()

    # -- B.1 halt latch (called by the body's halt lane) ---------------------------------------------------------
    def latch(self, epoch: int, reason: str = "halt") -> dict:
        """The halt lane's call, from any thread. It NEVER takes the arm lock (B-D1: a handler or a tick holding it
        must not delay a halt) and does no I/O: under the few-microsecond wire lock it sets the latch (every arm /
        arm_script / scan message is `halted` from now on), re-sends the override being sent with velocity 0 (the
        wire is frozen: the tick running right now cannot commit another pose) and queues the latch for the next tick
        (<= 20 ms), which applies it under the arm lock (`_apply_latch`). What that does depends on who owns the arms
        (B-D3): a session moving them -> the measured arm pose, the session's waist, the hands' last TARGET (B-D2);
        a hold (CarryLock, a target / measured hold) -> kept exactly; a blend -> paused; SONIC's free arms -> left
        free. Returns at once: {latched (arms held), arms: held|free, owner, pending, froze_wire, latch_ms, ...}."""
        t0 = time.perf_counter()
        now = self.clock()
        epoch = int(epoch)
        with self._wire_lock:
            self.halt_epoch = epoch if self.halt_epoch is None else max(self.halt_epoch, epoch)
            self.stats["latches"] += 1
            again = self.latched
            self.latched = True
            if self._latch_req is not None:
                self._latch_req["epoch"] = self.halt_epoch
            elif not again:
                self._latch_req = {"epoch": self.halt_epoch, "reason": reason, "t_mono": now}
            s, phase = self.sess, self.phase
            held = phase != "off"
            froze = held and self.sent is not None
            if froze:
                fh = _fill_hands(self.hands_sent)
                self.mux.set_upper(jm.wire_from_mj17(self.sent), [0.0] * N, fh["left"], fh["right"])
        return {"latched": held, "arms": "held" if held else "free", "already": again,
                "owner": None if s is None else s.stream, "op": None if s is None else s.op_id, "phase": phase,
                "pending": not again, "froze_wire": froze, "t_mono": now, "epoch": self.halt_epoch,
                "latch_ms": round((time.perf_counter() - t0) * 1e3, 3)}

    def _apply_latch(self) -> None:
        """Carry out a latch that the lane queued. Arm lock held (the tick, a handler, unlatch, end, abort)."""
        with self._wire_lock:
            req, self._latch_req = self._latch_req, None
        if req is None:
            return
        now = self.clock()
        s = self.sess
        self.stats["latches_applied"] += 1
        info = {"epoch": req["epoch"], "reason": req["reason"], "t_mono": req["t_mono"], "t_applied": now,
                "apply_ms": round((now - req["t_mono"]) * 1e3, 2), "ended_op": None, "hold": None,
                "pose_source": None, "hands_source": None}
        if self.phase == "active" and s is not None:
            # a session was moving the arms: stop where the arm IS (the measured pose, servo preloaded so the wire
            # does not step), keep the waist as the session sent it and the hands' last TARGET (never the measured
            # q: repeated halts must not ratchet a grip open, and nothing in the hand is dropped)
            qm = self.measured_mj17()
            if qm is not None and self.deploy.age_s() <= 0.1:
                src = "g1_debug.body_q"
            else:
                qm, src = self.continuity_pose(), "sent"
            pose, wmode = self._hold_waist(qm, s)
            hands = {side: (None if self.hands_sent.get(side) is None else list(self.hands_sent[side]))
                     for side in SIDES}
            info.update(case="session", arms="held", ended_op=s.op_id, pose_source=src, waist=wmode,
                        hands_source="last_target" if any(hands.values()) else "none (deploy default)")
            self._end_session(s, "canceled", "halt", "none", reason=req["reason"], label="measured")
            self.blend = None
            self.hold = Hold("latched", pose, hands, wmode, None, info["ended_op"], src, now, req["epoch"])
            self.phase = "hold"
            self._preload(pose)
        elif self.phase == "hold" and self.hold is not None:
            h = self.hold
            if s is not None:                      # a v0.5 op in its watchdog hold: the op ends, its pose stays
                info["ended_op"] = s.op_id
                h.kind, h.carry_arm, h.fence = "target", self._carry_arm(s), s.fence()
                self._end_session(s, "canceled", "halt", "none", reason=req["reason"], label="target")
                self.hold = h
            info.update(case="hold", arms="held", hold=h.kind, carry=h.is_carry(), pose_source=f"hold:{h.source}",
                        hands_source="hold")
        elif self.phase == "blend" and self.blend is not None:
            b = self.blend
            b["paused_at"] = now                   # frozen where it is; resume continues it (the arms go free)
            if s is not None and b.get("sess") is s:
                info["ended_op"] = s.op_id
                b["sess"] = None
                self._end_session(s, "canceled", "halt", "none", reason=req["reason"], label="stand")
                self.phase, self.blend = "blend", b
            info.update(case="blend", arms="held", pose_source="blend (paused)", hands_source="blend (paused)")
        else:
            info.update(case="off", arms="free")   # SONIC's own arms: a halt leaves them free (B-D3)
        self.latch_info = info
        self._say(f"[arm] LATCHED (halt epoch {req['epoch']}, {req['reason']}): {info['case']}, arms {info['arms']}"
                  f"{', ended ' + info['ended_op'] if info['ended_op'] else ''} ({info['apply_ms']} ms after the lane)")

    def unlatch(self, epoch: int) -> None:
        """resume{epoch}: open the latch and give the arms back exactly as the halt found them: free arms stay free,
        a hold (CarryLock) is the same hold, a paused blend continues. A session the halt stopped is over: its latched
        pose is kept until a new owner (control_epoch > halt_epoch) takes it over, or `end` / `release` / `stop {arms}`
        blends it back to SONIC."""
        with self._lock:
            self._apply_latch()
            if not self.latched:
                return
            if self.halt_epoch is not None and int(epoch) < self.halt_epoch:
                self._say(f"[arm] resume epoch {epoch} < halt epoch {self.halt_epoch}: ignored")
                return
            with self._wire_lock:
                self.latched = False
            b = self.blend
            if b is not None and b.get("paused_at") is not None:
                b["t0"] += self.clock() - b["paused_at"]
                b["paused_at"] = None
            self._say(f"[arm] unlatched (resume epoch {epoch}); arms {self._mode()}")

    def state(self) -> dict:
        with self._lock:
            s = self.sess
            return {"mode": self._mode(), "owner": None if s is None else s.stream,
                    "session_id": None if s is None else s.session_id, "op_id": None if s is None else s.op_id,
                    "kind": None if s is None else s.kind, "latched": self.latched, "halt_epoch": self.halt_epoch,
                    "hold": None if self.hold is None else self.hold.kind,
                    "arms": "free" if self.phase == "off" else "held"}

    def _mode(self) -> str:
        if self.latched and (self.phase != "off" or self._latch_req is not None):
            return "latched"                      # arms held by the latch (free arms under a latch read "off")
        if self.phase == "active" and self.sess is not None:
            return KIND_MODE[self.sess.kind]
        return self.phase

    # -- scripts (arm_script / scan): the plan objects live in body/arm_script.py and body/scan.py ---------------
    def handle_arm_script(self, op_id: str, args: dict, can_start: tuple[bool, str | None, dict]):
        """op `arm_script`. The IK runs in the IK worker (a separate process on the service, body/ik_worker.py), so
        this returns a `ScriptPending` that BodyService polls every loop and answers when the solve is back (the reply
        is the same as before: `accepted` with the plan, or a rejection); with an inline worker (tests) the solve is
        done at once and the reply dict comes back directly."""
        from . import arm_script

        args = args or {}
        try:
            with self._lock:
                self._apply_latch()
                now = self.clock()
                fence = self._fence_of(args)
                rej = self._script_gate(op_id, "arm_script", args, fence, can_start)
                if rej is not None:
                    return rej
                prep = arm_script.prepare(self, args, now)        # raises ArmError before any state change
        finally:
            self.flush()
        fut = self.ik.submit(arm_script.solve, prep["job"])
        pending = ScriptPending(self, op_id, args, fence, prep, fut, now)
        if fut.done():
            return pending.finish(can_start)
        self.stats["scripts_async"] += 1
        return pending

    def handle_scan(self, op_id: str, args: dict, can_start: tuple[bool, str | None, dict]) -> dict:
        from . import scan

        args = args or {}
        try:
            with self._lock:
                self._apply_latch()
                now = self.clock()
                fence = self._fence_of(args)
                rej = self._script_gate(op_id, "scan", args, fence, can_start)
                if rej is not None:
                    return rej
                plan = scan.build(self, args, now)                # raises ArmError before any state change
                return self._start_script(op_id, "scan", args, fence, plan, now)
        finally:
            self.flush()

    def _script_gate(self, op_id: str, op: str, args: dict, fence: dict, can_start) -> dict | None:
        rej = self._halt_check(fence, args)
        if rej is not None:
            return rej
        ok, reason, info = can_start
        if not ok:
            self.emit_reject(op_id, args, reason, info, op_name=op)
            return {"ok": False, "state": "rejected", "error": reason, "data": info}
        s = self.sess
        if self.phase == "active" and s is not None and not args.get("preempt"):
            self.stats["rejected_busy"] += 1
            return self._rej("arm_busy", {"owner": s.stream, "op": s.op_id, "kind": s.kind,
                                          "hint": "preempt: true takes over"})
        return None

    def _start_script(self, op_id: str, op: str, args: dict, fence: dict, plan, now: float) -> dict:
        stream = str(args.get("stream") or op_id)
        s = self._begin("script", op_id, stream, args, fence, now, op_name=op)
        s.plan = plan
        s.max_vel = _num(args, "max_vel", 0.5, 20.0, self.max_vel_default)
        s.blend_s = _num(args, "blend_s", 0.2, 10.0, self.blend_s_default)
        p: dict = {}
        self._servo_args(args, p)
        self._apply_servo_args(p)
        s.hold_on_end = plan.hold_on_end
        self.stats["scripts"] += 1
        self._emit(op_id, "progress", {"kind": f"{op}.plan", **plan.brief()})
        return {"ok": True, "state": "accepted", "data": {"id": op_id, "arm": self._mode(), **plan.brief()}}

    def _finish_script(self, pend: "ScriptPending", can_start) -> dict:
        """The IK is back: re-check the gate (a halt, a fault or another owner may have come meanwhile), then start
        the session from the pose being sent now."""
        from . import arm_script

        try:
            with self._lock:
                self._apply_latch()
                now = self.clock()
                try:
                    sol = pend.fut.result()
                except Exception as e:  # noqa: BLE001 - a dead worker is an answer, not a crash
                    self.ik.note_failure(e)
                    return self._rej("ik_unavailable", {"error": repr(e), "ik": self.ik.snapshot()})
                rej = self._script_gate(pend.op_id, "arm_script", pend.args, pend.fence, can_start)
                if rej is not None:
                    return rej
                try:
                    plan = arm_script.finish(self, pend.prep, sol)
                except ArmError as e:
                    return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
                rep = self._start_script(pend.op_id, "arm_script", pend.args, pend.fence, plan, now)
                rep["data"]["ik_wait_ms"] = round((now - pend.t0) * 1e3, 1)
                return rep
        finally:
            self.flush()

    # -- 50 Hz --------------------------------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        try:
            with self._lock:
                self._tick(now)
        finally:
            self.flush()

    def _tick(self, now: float) -> None:
        self._apply_latch()
        if self.phase == "off":
            self.t_tick = None
            return
        dt = 0.02 if self.t_tick is None else min(0.05, max(0.005, now - self.t_tick))
        self.t_tick = now
        self.stats["ticks"] += 1
        ref = self.reference_mj17()
        s = self.sess
        # -- watchdogs and session ends -------------------------------------------------------------------------
        if s is not None and self.phase == "active":
            if s.kind in ("target", "chunk") and now - s.t_msg > s.watchdog_s:
                s.counters["watchdog_trips"] += 1
                self.stats["watchdog_trips"] += 1
                if s.kind == "target":
                    self._say(f"[arm] {s.op_id} watchdog: no message for {s.watchdog_s:.2f} s -> hold "
                              f"{s.hold_s:.1f} s")
                    s.ended_by, s.t_hold = "watchdog", now
                    self.hold = Hold("watchdog", list(s.target if s.target is not None else self.continuity_pose()),
                                     dict(s.hands), s.waist_mode, s.stream, s.op_id, "target", now)
                    self.phase = "hold"
                else:
                    self._end_session(s, "failed", "watchdog", s.hold_on_end or "measured", reason="client_silent")
                    s = None
            elif s.kind == "script" and s.plan is not None and now - s.t_start >= s.plan.duration_s:
                self._end_session(s, "succeeded", "script", s.plan.hold_on_end, reason=None)
                s = None
        if s is not None and self.phase == "hold" and self.hold is not None and self.hold.kind == "watchdog" \
                and now - s.t_hold >= s.hold_s:
            self._start_blend(now, "watchdog", op_sess=s)
        if self.phase == "off":
            return
        # -- the pose for this tick ------------------------------------------------------------------------------
        vel = [0.0] * N
        servo_idx: tuple = ARM_IDX
        waist_mode = "ref"
        s = self.sess
        blending = self.phase == "blend"
        if blending:
            b = self.blend
            tb = b["paused_at"] if b.get("paused_at") is not None else now      # a halt pauses a blend
            a = _minjerk((tb - b["t0"]) / b["dur"])
            des = [(1 - a) * f + a * r for f, r in zip(b["from"], ref)]
            hands = {}
            for side in SIDES:
                f = b["hands"].get(side)
                hands[side] = None if f is None else _lerp(f, jm.DEX3_CLOSED[side], a)
            waist_mode = "cmd"                       # the blend already ends on the reference waist
            if a >= 1.0:
                op_sess = b.get("sess")
                self._say(f"[arm] {op_sess.op_id if op_sess else '-'} blended back; override released")
                if op_sess is not None and op_sess is self.sess:
                    why = op_sess.ended_by or b["why"]
                    state = {"client": "succeeded", "watchdog": "failed"}.get(why, "canceled")
                    reason = {"watchdog": "client_silent", "client": None}.get(why, why)
                    self._end_session(op_sess, state, why, "none", reason=reason, label="stand")
                self._drop()
                return
        elif self.phase == "hold":
            h = self.hold
            des, hands, waist_mode = list(h.pose), dict(h.hands), h.waist_mode
        else:  # active
            if s.kind == "target":
                des = list(s.target if s.target is not None else self.continuity_pose())
                hands = dict(s.hands)
                waist_mode = s.waist_mode
                if s.vel_mode in ("est", "cmd") and s.target_vel is not None:
                    vel = list(s.target_vel)
            elif s.kind == "chunk":
                des, hands, waist_mode = self._chunk_row(s, now)
            else:
                des, hands, waist_mode, servo_idx = self._script_row(s, now, ref)
        if not blending:
            if waist_mode == "ref":
                des[0:3] = ref[0:3]
            elif waist_mode == "yaw":
                des[1:3] = ref[1:3]
            if s is not None and self.phase == "active":
                s.last_ref, s.last_hands, s.last_waist = list(des), dict(hands), \
                    ("cmd" if waist_mode == "cmd" else "ref")
                s.waist_now = waist_mode
            self._hist.append((now, list(des)))
            self.ref_last = list(des)
            self._servo(now, dt, des, servo_idx)
            for k in servo_idx:
                lo, hi = _LIM[k]
                des[k] = min(max(des[k] + self.corr[k], lo), hi)
        # -- slew limit (protects against jumps; the only place the sent pose advances) -------------------------
        max_vel = s.max_vel if (s is not None and not blending) else self.max_vel_default
        step = max_vel * dt
        prev = self.sent if self.sent is not None else des
        out, n_lim, big = [], 0, 0.0
        for k, (p_, d_) in enumerate(zip(prev, des)):
            dd = d_ - p_
            if dd > step:
                dd = step
                n_lim += 1
            elif dd < -step:
                dd = -step
                n_lim += 1
            out.append(p_ + dd)
            if k >= 3:
                big = max(big, abs(dd))
        if n_lim:
            self.stats["slew_limited_ticks"] += 1
            if s is not None:
                s.counters["slew_limited_ticks"] += 1
            if s is not None and s.kind == "target" and s.vel_mode != "zero" and self.phase == "active":
                vel = [(o - p_) / dt for o, p_ in zip(out, prev)]
        if not (self.phase == "active" and s is not None and s.kind == "target"):
            vel = [0.0] * N
        self.stats["max_step_rad"] = round(max(self.stats["max_step_rad"], big), 5)
        if s is not None and self.phase == "active":
            s.max_step = max(s.max_step, big)
            if s.kind == "chunk":
                n_arm_lim = sum(1 for k in ARM_IDX if abs(out[k] - des[k]) > 1e-12)
                s.win[1] += n_arm_lim
                s.tot[1] += n_arm_lim
        hs = {side: (list(hands[side]) if hands.get(side) is not None else None) for side in SIDES}
        fh = _fill_hands(hs)
        with self._wire_lock:
            if self._latch_req is not None:
                # a halt arrived during this tick: its frozen pose stays on the wire, the next tick applies it
                self.stats["ticks_skipped_latch"] += 1
                return
            self.sent = out
            self.hands_sent = fh if fh["left"] is not None else hs
            self.mux.set_upper(jm.wire_from_mj17(out), jm.wire_from_mj17(vel), fh["left"], fh["right"])
        if s is not None and self.phase == "active":
            self._progress(s, now)

    def _chunk_row(self, s: _Session, now: float):
        t = now + s.lead_s
        c = s.chunk
        if c is None:
            row, lh, rh = s.base.at(now)
        else:
            row, lh, rh = c.at(t)
            s.k, s.stall_s = c.k(t), c.stall(t)
            s.stall_max = max(s.stall_max, s.stall_s)
            if s.old is not None and s.xf_i < s.xfade:
                w = (s.xf_i + 1) / s.xfade
                orow, olh, orh = s.old.at(t if isinstance(s.old, _Chunk) else now)
                row = _lerp(orow, row, w)
                lh = _lerp(olh if olh is not None else jm.DEX3_CLOSED["left"], lh, w)
                rh = _lerp(orh if orh is not None else jm.DEX3_CLOSED["right"], rh, w)
                s.xf_i += 1
                if s.xf_i >= s.xfade:
                    s.old = None
        waist_mode = "cmd" if (c is not None and c.waist == "cmd") else "ref"
        des, n = jm.clamp_mj17(row, margin=0.02)
        n_arm = sum(1 for k in ARM_IDX if des[k] != float(row[k]))
        hands = {}
        n_hand = 0
        for side, h in (("left", lh), ("right", rh)):
            if h is None:
                hands[side] = None
                continue
            q, m = jm.clamp_hand(side, h)
            hands[side] = q
            n_hand += m
        if c is not None:
            s.win[0] += n_arm + n_hand
            s.tot[0] += n_arm + n_hand
            s.win[2] += 1
            s.tot[2] += 1
            if n_arm + n_hand:
                s.counters["clamped"] += 1
                self.stats["clamped"] += 1
        return list(des), hands, waist_mode

    def _script_row(self, s: _Session, now: float, ref: list[float]):
        plan = s.plan
        t = now - s.t_start
        des, hands = plan.sample(t, ref)
        try:
            for payload in plan.on_tick(t, now, self) or ():
                self._emit(s.op_id, "progress", payload)
        except Exception as e:  # noqa: BLE001 - measurement must never stop the arm
            self._say(f"[arm] {s.op_id} {getattr(plan, 'op', 'script')} measurement error: {e!r}")
        return list(des), dict(hands), plan.waist_mode, plan.servo_idx

    def _progress(self, s: _Session, now: float) -> None:
        if s.kind == "chunk":
            if now - s.t_progress < 0.2:
                return
            s.t_progress = now
            w = s.win
            data = {"kind": "chunk", "session_id": s.session_id, "stream": s.stream,
                    "chunk_seq": 0 if s.chunk is None else s.chunk.seq, "k": s.k,
                    "T": None if s.chunk is None else s.chunk.T, "stall_s": round(s.stall_s, 3),
                    "clamped_frac": round(w[0] / (CHUNK_VALUES * w[2]), 4) if w[2] else 0.0,
                    "clamped_frac_total": round(s.tot[0] / (CHUNK_VALUES * s.tot[2]), 4) if s.tot[2] else 0.0,
                    "slew_frac": round(w[1] / (CHUNK_VALUES * w[2]), 4) if w[2] else 0.0,
                    "latency_ms": s.latency_ms, "inference_ms": s.inference_ms, "lead_s": s.lead_s,
                    "cross_fades": s.cross_fades, "max_step_rad": round(s.max_step, 4),
                    "chunks": {"received": s.chunks["received"], "applied": s.chunks["applied"],
                               "dropped": dict(s.chunks["dropped"])}}
            s.win = [0, 0, 0]
        elif s.kind == "target":
            if now - s.t_progress < 1.0:
                return
            s.t_progress = now
            data = {"arm": self._mode(), "msg_age_s": round(now - s.t_msg, 3), **s.counters}
        else:
            hz = float(getattr(s.plan, "progress_hz", 2.0) or 0.0)
            if hz <= 0 or now - s.t_progress < 1.0 / hz:
                return
            s.t_progress = now
            data = {"kind": getattr(s.plan, "op", "script"), **s.plan.progress(now - s.t_start)}
        self._emit(s.op_id, "progress", data)

    def _hist_at(self, t: float) -> list[float]:
        h = self._hist
        for th, v in reversed(h):
            if th <= t:
                return v
        return h[0][1]

    def _servo(self, now: float, dt: float, des_ref: list[float], idx: tuple) -> None:
        """Integral outer loop on the measured joints (module docstring "Servo"). Frozen while g1_debug is stale;
        |corr| <= servo_max; anti-windup at the joint limits (the corrected target is clamped there and the
        integrator stops pushing outwards)."""
        sv = self.sv
        ki = float(sv["ki"])
        if ki <= 0.0 or not self._hist:
            return
        if self.deploy.age_s() > 0.1:
            return
        qm = self.measured_mj17()
        if qm is None:
            return
        model = sv["model"]
        if model == "delay":
            r = self._hist_at(now - float(sv["delay_s"]))
            w = None
        else:
            u = self._hist_at(now - float(sv["dead_s"]))
            prev = self._ym if self._ym is not None else list(qm)
            a = 1.0 - math.exp(-dt / max(1e-3, float(sv["tau_s"])))
            r = [y + (uu - y) * a for y, uu in zip(prev, u)]
            self._ym = r
            if model == "gated":
                v0 = max(1e-3, float(sv["v0"]))
                w = [math.exp(-abs(y1 - y0) / dt / v0) for y1, y0 in zip(r, prev)]
            else:
                w = None
        mx = float(sv["max"])
        cap = self.corr_cap
        for k in idx:
            e = r[k] - qm[k]
            g = float(sv["ki_waist"]) if k == YAW_IDX else ki
            c = self.corr[k] + (1.0 if w is None else w[k]) * g * dt * e
            m = mx if cap is None else cap[k]
            c = min(max(c, -m), m)
            if cap is not None:
                cap[k] = max(mx, min(cap[k], abs(c)))
            lo, hi = _LIM[k]
            if (des_ref[k] + c > hi and c > self.corr[k]) or (des_ref[k] + c < lo and c < self.corr[k]):
                continue
            self.corr[k] = c

    # -- status ---------------------------------------------------------------------------------------------------
    def progress(self) -> dict:
        return self.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            now = self.clock()
            s = self.sess
            legacy = {"active": "stream", "hold": "hold", "blend": "blend", "off": "off"}[self.phase]
            snap = {"state": legacy, "mode": self._mode(), "modes": list(MODES), "ops": list(OPS),
                    "owner": None if s is None else s.stream, "op": None if s is None else s.op_id,
                    "kind": None if s is None else s.kind,
                    "msg_age_s": None if s is None else round(now - s.t_msg, 3),
                    "waist": None if s is None else s.waist_mode, "vel": None if s is None else s.vel_mode,
                    "watchdog_s": None if s is None else s.watchdog_s, "hold_s": None if s is None else s.hold_s,
                    "blend_s": self.blend_s_default if s is None else s.blend_s,
                    "max_vel": self.max_vel_default if s is None else s.max_vel,
                    "servo": {**self.sv, "corr_max_abs": round(max(abs(c) for c in self.corr[3:]), 4),
                              "corr_waist_yaw": round(self.corr[YAW_IDX], 4)},
                    "latched": self.latched, "halt_epoch": self.halt_epoch, "latch": self.latch_info,
                    "latch_pending": self._latch_req is not None, "arms": "free" if self.phase == "off" else "held",
                    "ik": self.ik.snapshot(),
                    "hold": None if self.hold is None else self.hold.brief(now),
                    "carry": carry_info(self.hold, self.measured_mj17(), now),
                    "hands": [side for side in SIDES if self.hands_sent.get(side) is not None],
                    "sent_mj17": None if self.sent is None else [round(v, 4) for v in self.sent],
                    "stats": dict(self.stats)}
            if s is not None:
                snap["session"] = {**s.fence(), **(s.session_brief() if s.kind == "chunk" else {}),
                                   "lead_s": s.lead_s if s.kind == "chunk" else None,
                                   "hold_on_end": s.hold_on_end}
                if s.kind == "script" and s.plan is not None:
                    snap["script"] = {"op": getattr(s.plan, "op", None), **s.plan.progress(now - s.t_start)}
            return snap
