"""GrootArmExecutor (R.4; backend `groot`, executor `groot_arms`): GR00T N1.7 drives the G1's arms and Dex3 hands
through the body `arm` op in chunk mode, and SONIC keeps the legs (owner decision (b), PLAN §0.7;
docs/groot_arms_design.md §5; the wire: docs/contracts/arm_chunk.md).

Label: every result is **experimental**. The policy is the off-the-shelf nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace
checkpoint, trained on one apple-to-plate scene with the left hand, so in a house it is expected to move the arms
plausibly and usually not to finish the grasp (PLAN §0.8). What this executor must get right is the plumbing and the
control semantics: lease and session fence, cancel, halt, stale chunks, the policy going down, and an honest outcome.

    run(job)   stance_check  base at rest (waits settle_s for SONIC's glide) and upright; else failed(base_moving |
                             fell) with the body untouched. The service already checked the drift since
                             check_reachability
               camera        the ego camera on (world.enable_camera(consumer=<execution id>, ttl_s, hz), renewed
                             while the session runs; P1 renders ego_view only while a consumer holds it, and its
                             `detections` op answers `camera_off` otherwise, so this comes before the view check);
                             waits up to camera_wait_s for the first frame (P1 skips 4 warm-up frames: enabled at
                             camera_warm_hz, 10 Hz in `full`, that is ~0.4 s), then streams at camera_hz
               render budget P1 renders EVERY enabled camera product at every render call (docs/viz.md §4), so an
                             enabled ego_view is rendered at the head camera's rate whatever its own rate is. With
                             session_head_hz > 0 the head camera drops to that rate for the session (restored after):
                             `full` renders head + ego_view once per inference (2.5 Hz) and the client waits for
                             each fresh frame (frame_sync), the measured fix for SONIC's timing gate
                             (docs/groot_serving.md §9)
               view_check    >= view_min_px (200) of the object in the GR00T camera, from world.detections(camera=
                             "ego_view") when the world advertises it (capabilities()["detections:ego_view"]);
                             else skipped, and the result says so (INTERIM)
               enter         the `arm` session opens with open hands (0.3 s blend), the arms on SONIC's own
                             reference
               execute       GrootArmClient thread: latest ego frame + g1_debug -> observation -> PolicyServer (REQ,
                             1.5 s) -> chunk -> `arm`; the ground-truth outcome at 10 Hz through the WorldModel
               end           `arm` end with hold_on_end = target (success: the carry hold = the body's CarryLock
                             B.7), stand (cancel with nothing in hand), measured (every failure, halt, policy down)

    outcome                       when
    succeeded                     gt_lifted: >= min_lift_m above its support (and within max_dist_m of the palm once
                                  the world has palm poses; before that, within reach_max_m of the pelvis) for hold_s
    failed(grasp_missed)          the hand is >= 0.6 closed (of Arena's closed pose) and nothing is lifted, for 1.5 s
    failed(object_dropped)        lifted before, now back within 2 cm of its support (or > 0.20 m from the palm)
    failed(policy_stall)          no chunk rows left for stall_fail_s (5 s; an event at 2 s)
    failed(policy_unavailable)    2 consecutive PolicyServer errors (a timed-out get_action is followed by a short
                                  ping, so a dead server is detected in <= 2.4 s). The executor then reports itself
                                  down, so new calls are rejected at CAPABILITY ("policy unavailable: ...") while the
                                  object_type enum stays frozen; the next good ping brings it back
    failed(policy_out_of_bounds)  > 20 % of the arm/hand values clamped, sustained over 1 s
    failed(halted | fell), cancelled, failed(timeout) at skill.max_duration_s

Fences (PLAN §5.5): session = stream = execution id, and every message carries generation and control_epoch. A result
that comes back after the session was fenced is dropped (`stale_session`) and never sent. The fence check and the
send share one lock, and cancel waits for a send in flight before it acknowledges, so no chunk is published after
the cancel ack.

Body refusals (the as-built wire, docs/contracts/arm_chunk.md §8): body_verdict() decides which end the session.
`stale_command` is keyed on data.why: the fence (control_epoch / resume_epoch -> halted, generation -> superseded)
is fatal, a single late message (t_wall older than the watchdog, no data.why) is only counted. `body_busy` (B.2
lease), `halted` (B.1 latch), `stale_session`, `arm_preempted` ... are fatal (BODY_FATAL); three refusals in a row
of any kind are too. A session the body ends on its own maps its ended_by through ENDED_BY (watchdog ->
policy_stall, halt/stop -> halted, fault -> fell, preempted/taken_over -> body_busy, anything else ->
controller_unavailable). After `end` the executor waits terminal_wait_s for the body's terminal event, so the
result carries the body's own counters (data.body: chunks applied, clamped_frac_total, slew_frac_total,
max_step_rad).

GT confinement: every ground-truth read is a WorldModel call (object, robot_pose, detections, palm_position when the
world has it). GR00T's inputs are sensors, not ground truth: the ego camera frame and SONIC's g1_debug (5557).

Timing is wall time (time.monotonic), not the session SimClock: the body, the deploy and the PolicyServer are
wall-clock processes, and the live stack runs SimClock(1.0).

The model-side helpers are owner groot_srv's `groot/` package (policy_client.PolicyClient, obs.build_observation,
actions.to_arm_chunk / clamp_stats). They are imported when the executor is built; without them it reports itself
down and every run fails policy_unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from api.types import ServiceHealth

from .kinematic_attach import ManipJob, ManipOutcome

EXECUTOR = "groot_arms"
BACKEND = "groot"
LABEL = "experimental"
CHECKPOINT = "nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5550"
ENDPOINT_ENV = "WL_GROOT_ENDPOINT"
CARRY_LABEL = "CarryLock (body B.7: the arm op's target hold with the hand closed, m1.md §3.9)"
N_UPPER, N_HAND = 17, 7

# Arena's closed Dex3 pose in Dex3 order (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1): the
# checkpoint's hand actions are open (0) or this pose (HF Static dataset_statistics min/max, docs/groot_arms_design.md
# §2.5). grasp_missed measures closure against it rather than the deploy's fist: Arena's "closed" is about half a fist,
# so a fist-relative 0.6 would never trigger.
ARENA_CLOSED_DEX3 = {"left": (0.0, 0.7, 0.7, -0.6, -1.2, -0.6, -1.2),
                     "right": (0.0, -0.7, -0.7, 0.6, 1.2, 0.6, 1.2)}

# body replies that end the session (the arm op is no longer ours) -> the outcome's reason
BODY_FATAL = {"stale_session": "controller_unavailable", "arm_preempted": "body_busy", "not_owner": "body_busy",
              "arm_busy": "body_busy", "body_busy": "body_busy",       # B.2: a lease held by another execution
              "halted": "halted", "arm_stopped": "halted",               # the B.1 latch; body stop {arms: true}
              "not_standing": "controller_unavailable",
              "mode_mismatch": "controller_unavailable", "bad_args": "controller_unavailable",
              "body_timeout": "controller_unavailable"}
# The body's `stale_command` means two things (docs/contracts/m1.md §3.9, §3.11), told apart by data.why:
#   the FENCE (data.why = control_epoch | resume_epoch | generation): every message of this execution is refused
#       from now on (a halt ended its epoch, or a correction's newer generation superseded it) -> fatal;
#   one late MESSAGE (no data.why; data.t_wall older than the session watchdog): only that message was dropped, the
#       next one carries a fresh t_wall -> counted, not fatal.
STALE_WHY = {"control_epoch": "halted", "resume_epoch": "halted", "generation": "superseded"}
MAX_BODY_REJECTS = 3        # this many refusals in a row of any kind (an unknown code too) end the session
# The body ended the session on its own (terminal event while the executor still streams; data.ended_by, m1.md
# §3.9) -> the outcome's reason. watchdog = the body heard nothing from P5 for watchdog_s (terminal `failed`,
# reason client_silent, and body.fault{policy_lost}); stop = `stop {arms: true}` from someone else. Anything
# else (client, script, an unknown value) means the arm op is not doing what this session asked.
ENDED_BY = {"halt": "halted", "stop": "halted", "watchdog": "policy_stall", "fault": "fell",
            "preempted": "body_busy", "taken_over": "body_busy"}


def body_verdict(rep: dict) -> tuple[str, str] | None:
    """(reason, detail) when a body reply to a session message ends the session, None when it does not."""
    if rep.get("ok"):
        return None
    err = str(rep.get("error") or "rejected")
    data = rep.get("data") if isinstance(rep.get("data"), dict) else {}
    if err == "stale_command":
        why = data.get("why")
        if why in STALE_WHY:
            return STALE_WHY[why], f"the body fenced the arm stream: stale_command ({why})"
        return None
    if err.startswith("fault:"):
        return ("fell" if err == "fault:fallen" else "controller_unavailable"), f"the body is in fault: {err}"
    if err in BODY_FATAL:
        return BODY_FATAL[err], f"the body rejected the arm stream: {err}"
    return None


def body_reject_key(rep: dict) -> str:
    """The `chunks_dropped` counter key of a refused message: body:<error>, with the stale_command kind."""
    err = str(rep.get("error") or "rejected")
    if err == "stale_command":
        why = (rep.get("data") or {}).get("why") if isinstance(rep.get("data"), dict) else None
        return f"body:stale_command({why or 't_wall'})"
    return f"body:{err}"


# ================================================================================================ config
@dataclass
class GrootArmsConfig:
    """Profile section `groot_arms:` (config/profiles/full.yaml). The endpoint is WL_GROOT_ENDPOINT, else this, else
    tcp://127.0.0.1:5550; on the main box it is a local tunnel to the dev box's PolicyServer (OD3)."""
    endpoint: str = DEFAULT_ENDPOINT
    timeout_s: float = 1.5               # one get_action
    ping_timeout_s: float = 0.5          # the ping after a failed get_action, and the health pings
    ping_period_s: float = 5.0
    warmup: bool = True                  # one get_action with warmup_timeout_s after the first good ping (the first
    warmup_timeout_s: float = 20.0       # N1.7 call took 3.8 s, docs/arena_vs_sonic.md §2.5)
    camera: str = "ego_view"             # the GR00T camera (OD1: Arena's head camera, 640x480)
    camera_port: int = 5566              # P1 PUB of ego_view (docs/contracts/p1_m2b.md §5.2); + the port offset
    camera_swap_rb: bool = True          # P1 hands RGB to cv2.imencode, so a plain JPEG decode is BGR (m1.md §1.4)
    camera_ttl_s: float = 10.0           # enable_camera consumer lease; refreshed every camera_refresh_s
    camera_refresh_s: float = 3.0
    camera_hz: float = 30.0              # passed with every enable: P1 keeps a camera's rate across off/on
                                         # (docs/contracts/p1_m2b.md §5.4), so a session must say which it needs
    camera_warm_hz: float = 0.0          # the rate the session enables ego_view at, until its first frame (P1 skips
                                         # 4 warm-up frames per enable: 1.6 s at 2.5 Hz); 0 = camera_hz
    camera_wait_s: float = 1.5           # the first ego frame after enabling (measured 0.18-0.19 s, §13.1)
    session_head_hz: float = 0.0         # the head camera's rate while the session holds ego_view (P1 renders every
                                         # enabled product at every render call), restored afterwards; 0 = leave it
    frame_sync: bool = False             # when due and the newest frame is older than frame_max_age_s, wait for the
                                         # next frame (a camera at the inference rate) instead of polling
    debug_port: int = 5557               # SONIC g1_debug PUB
    frame_max_age_s: float = 0.15
    state_max_age_s: float = 0.06
    replan_s: float = 0.4                # 2.5 Hz
    min_rows_left: int = 12
    expire_margin_s: float = 0.1
    max_errors: int = 2
    keepalive_s: float = 0.5
    watchdog_s: float = 2.0              # the body's session watchdog (docs/contracts/arm_chunk.md §1.1)
    lead_s: float = 0.15                 # play rows this far ahead: SONIC's ~0.15 s arm lag (docs/arm_tracking.md
                                         # §0 on the body branch; the body's chunk-mode default is also 0.15)
    hands_open_s: float = 0.3
    view_min_px: float = 200.0
    settle_s: float = 1.0
    rest_speed: float = 0.05
    rest_wz: float = 0.1
    gt_hz: float = 10.0
    min_lift_m: float = 0.05
    max_palm_dist_m: float = 0.12
    hold_s: float = 1.0
    drop_below_m: float = 0.02
    drop_palm_dist_m: float = 0.20
    reach_max_m: float = 0.9
    closure_thresh: float = 0.6
    grasp_missed_s: float = 1.5
    stall_event_s: float = 2.0
    stall_fail_s: float = 5.0
    oob_frac: float = 0.2
    oob_window_s: float = 1.0
    arm_timeout_s: float = 0.5           # one `arm` message round trip
    terminal_wait_s: float = 0.3         # after `end`: wait this long for the body's terminal event (its counters)
    max_duration_s: float | None = None  # None: the skill's
    progress_hz: float = 5.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "GrootArmsConfig":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d or d[f.name] is None:
                continue
            v = d[f.name]
            kw[f.name] = v if f.name in ("endpoint", "camera") else (
                bool(v) if f.name in ("warmup", "camera_swap_rb", "frame_sync") else
                int(v) if f.name in ("camera_port", "debug_port", "min_rows_left", "max_errors") else float(v))
        cfg = cls(**kw)
        env = os.environ.get(ENDPOINT_ENV)
        if env:
            cfg.endpoint = env
        return cfg


# ================================================================================================ ports
@runtime_checkable
class PolicyPort(Protocol):
    """groot.policy_client.PolicyClient (one per thread: ZMQ sockets are not thread-safe)."""
    def ping(self, timeout_s: float | None = None) -> bool: ...
    def get_action(self, obs: dict) -> dict: ...
    def close(self) -> None: ...


@runtime_checkable
class ArmPort(Protocol):
    """The body `arm` op (docs/contracts/arm_chunk.md). `arm` is thread-safe and returns the body's reply
    {ok, state, error?, data?}; `subscribe(cb)` delivers every `arm` body.event (progress and terminal) and returns an
    unsubscribe callable."""
    def arm(self, args: dict, op_id: str | None = None) -> dict: ...
    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]: ...


@runtime_checkable
class SensorPort(Protocol):
    """GR00T's inputs: the latest ego frame (HxWx3 uint8 RGB) and the latest g1_debug dict, each with the
    time.monotonic() it was received."""
    def ego_frame(self) -> tuple[np.ndarray | None, float]: ...
    def debug_state(self) -> tuple[dict | None, float]: ...
    # optional: wait_frame(after: float, timeout_s: float) -> bool, True once a frame captured after `after` arrived
    # (frame_sync; ZmqSensors has it, a port without it is polled)


@dataclass
class GrootArmOutcome(ManipOutcome):
    """ManipOutcome + the typed GR00T data (inferences, chunks, latency, attempts ...) for the ManipulationResult."""
    data: dict = field(default_factory=dict)


def _groot_helpers() -> dict[str, Any]:
    """groot_srv's model-side helpers; raises ImportError when the package is missing."""
    from groot.actions import clamp_stats, to_arm_chunk
    from groot.obs import DEFAULT_PROMPT, build_observation
    from groot.policy_client import PolicyClient
    return {"policy": lambda endpoint, timeout_s: PolicyClient(endpoint, timeout_s=timeout_s),
            "build_obs": build_observation, "to_chunk": to_arm_chunk, "clamp_stats": clamp_stats,
            "prompt": DEFAULT_PROMPT}


def hand_closure(side: str, q7: Any) -> float:
    """Closure of a measured Dex3 hand as a fraction of Arena's closed pose (0 open, 1 Arena-closed), mean over the
    six joints that close (thumb_0 is 0 in both poses)."""
    ref = ARENA_CLOSED_DEX3[side]
    q = [float(v) for v in q7]
    return sum(min(max(q[j] / ref[j], 0.0), 1.5) for j in range(1, 7)) / 6.0


def _p(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _rows(a: Any) -> list[list[float]]:
    return np.round(np.asarray(a, dtype=np.float64), 5).tolist()


# ================================================================================================ session
class Session:
    """One manipulate execution's GR00T session: the fence, the counters, and the hand-off between the client thread
    and the executor's event loop (everything the loop reads from the thread goes through `lock` or a deque)."""

    def __init__(self, job: ManipJob, handle: Any, skill: Any, prompt: str, cfg: GrootArmsConfig):
        ex = getattr(handle, "execution", None)
        self.id = (getattr(job, "execution_id", "") or getattr(ex, "execution_id", "")
                   or f"groot-{uuid.uuid4().hex[:8]}")
        self.generation = int(getattr(job, "generation", 0) or getattr(ex, "generation", 0) or 0)
        self.control_epoch = int(getattr(job, "control_epoch", 0) or getattr(ex, "control_epoch", 0) or 0)
        self.job, self.skill, self.prompt, self.cfg = job, skill, prompt, cfg
        self.op_id = f"arm-{self.id}"
        self.send_lock = threading.Lock()
        self.lock = threading.Lock()
        self.fenced: str | None = None
        self.t_fence: float | None = None
        self.t_ack: float | None = None
        self.verdict: tuple[str, str, str] | None = None        # (status, reason, detail) from the client thread
        self.seq = 0
        self.inferences = 0
        self.chunks_sent = 0
        self.keepalives = 0
        self.dropped: collections.Counter = collections.Counter()
        self.latencies_ms: list[float] = []
        self.errors = 0                                          # consecutive
        self.errors_total = 0
        self.rejects_run = 0                                     # consecutive body refusals
        self.last_error: str | None = None
        self.obs_stale = 0
        self.frame_waits = 0                                     # frame_sync: waits for a fresh frame, and their time
        self.frame_wait_s = 0.0
        self.last_frame_t: float | None = None                   # capture time of the last frame sent to the policy
        self.head_prev_hz: float | None = None                   # session_head_hz: the head rate to restore
        self.last_obs_age: tuple[float, float] | None = None
        self.last_t0: float | None = None
        self.last_T = 0
        self.last_dt = 0.02
        self.t_last_msg = time.monotonic()
        self.t_exec: float | None = None
        self.client_clamp: collections.deque = collections.deque(maxlen=64)   # (t_sent, clamped_frac) per chunk
        self.client_clamp_all: list[float] = []
        self.body_progress: dict | None = None
        self.t_body_progress: float | None = None
        self.body_clamp: collections.deque = collections.deque(maxlen=64)     # (t, clamped_frac) from arm.progress
        self.body_slew: list[float] = []                                      # slew_frac per arm.progress window
        self.body_terminal: dict | None = None
        self.stall_max = 0.0
        self.sent_log: list[tuple[float, int]] = []              # (t_mono, seq) of every chunk the body accepted
        self.events: collections.deque = collections.deque()     # (type, fields) for the loop to emit

    # -- fence -----------------------------------------------------------------------------------------------
    def fence(self, reason: str) -> bool:
        """Stop publishing. The first reason wins. Returns True the first time."""
        with self.lock:
            if self.fenced is not None:
                return False
            self.fenced = reason
            self.t_fence = time.monotonic()
            return True

    def drain(self) -> float:
        """Wait for a send in flight (the fence is set): from the returned time on no chunk can be published."""
        with self.send_lock:
            if self.t_ack is None:
                self.t_ack = time.monotonic()
            return self.t_ack

    def publish(self, arm: ArmPort, msg: dict) -> dict | None:
        """Send one session message unless the session is fenced (None). The fence check and the send are atomic."""
        with self.send_lock:
            if self.fenced is not None:
                return None
            rep = arm.arm(msg)
            self.t_last_msg = time.monotonic()
            return rep

    def keepalive(self, arm: ArmPort) -> None:
        """Refresh the body's session watchdog when nothing was sent for keepalive_s. Called by the client thread
        between inferences AND by the executor's loop, because a blocked get_action (1.5 s) plus the ping after it
        (0.5 s) would otherwise outlast the 2 s watchdog."""
        if time.monotonic() - self.t_last_msg < self.cfg.keepalive_s:
            return
        rep = self.publish(arm, {**self.base(), "keepalive": True})
        if rep is not None and rep.get("ok"):
            self.keepalives += 1
            self.body_ok()
        elif rep is not None:
            self.body_rejected(rep)

    def body_ok(self) -> None:
        with self.lock:
            self.rejects_run = 0

    def body_rejected(self, rep: dict) -> None:
        """Count a refused session message; end the session when the refusal is fatal (body_verdict), or after
        MAX_BODY_REJECTS refusals in a row (an unknown code must not leave the arms to stall silently)."""
        key = body_reject_key(rep)
        self.dropped[key] += 1
        v = body_verdict(rep)
        with self.lock:
            self.rejects_run += 1
            if v is None and self.rejects_run >= MAX_BODY_REJECTS:
                v = ("controller_unavailable", f"the body refused {self.rejects_run} arm messages in a row "
                                               f"(last: {key[5:]})")
            if v is None:
                return
            if self.verdict is None:
                self.verdict = ("failed", v[0], v[1])
        self.fence(key)

    # -- derived ---------------------------------------------------------------------------------------------
    def base(self) -> dict:
        return {"stream": self.id, "session_id": self.id, "execution_id": self.id, "generation": self.generation,
                "control_epoch": self.control_epoch, "mode": "chunk", "t_wall": time.time()}

    def stall_s(self, now: float) -> float:
        """Time past the last row of the last chunk the body accepted (before the first chunk: time since execute
        began), the larger of this estimate and the body's own."""
        if self.last_t0 is not None:
            s = max(0.0, now + self.cfg.lead_s - (self.last_t0 + (self.last_T - 1) * self.last_dt))
        elif self.t_exec is not None:
            s = max(0.0, now - self.t_exec)
        else:
            s = 0.0
        bp = self.body_progress or {}
        if isinstance(bp.get("stall_s"), (int, float)) and self.t_body_progress is not None \
                and now - self.t_body_progress < 0.5:
            s = max(s, float(bp["stall_s"]))
        self.stall_max = max(self.stall_max, s)
        return s

    def oob(self, now: float) -> float | None:
        """The clamped fraction if it stayed above oob_frac over the whole oob window (body numbers when the body
        reports them, else the client's per-chunk estimate), else None."""
        w = self.cfg.oob_window_s
        samples = [(t, f) for t, f in (self.body_clamp or self.client_clamp) if now - t <= w]
        if len(samples) < 2 or samples[-1][0] - samples[0][0] < 0.6 * w:
            return None
        if all(f > self.cfg.oob_frac for _, f in samples):
            return max(f for _, f in samples)
        return None

    def body_summary(self) -> dict | None:
        """The body's own numbers for this session: its terminal event when it came, else the last arm.progress."""
        term = (self.body_terminal or {}).get("data") or {}
        bp = self.body_progress or {}
        src = term if term else bp
        if not src:
            return None
        out = {"source": "terminal" if term else "progress"}
        if term:
            out.update(state=(self.body_terminal or {}).get("state"), ended_by=term.get("ended_by"),
                       hold=term.get("hold"), reason=term.get("reason"), duration_s=term.get("duration_s"))
        for k in ("chunks", "clamped_frac_total", "slew_frac_total", "stall_s_max", "max_step_rad", "cross_fades",
                  "lead_s", "chunk_seq"):
            if src.get(k) is not None:
                out[k] = src[k]
        if not term and bp.get("slew_frac") is not None:
            out["slew_frac_last_window"] = bp["slew_frac"]
        return out

    def slew_frac(self) -> float | None:
        """The body's slew-limited share of the arm values (slew_frac_total of the terminal event; before it, the
        mean of the 5 Hz progress windows)."""
        term = (self.body_terminal or {}).get("data") or {}
        if isinstance(term.get("slew_frac_total"), (int, float)):
            return round(float(term["slew_frac_total"]), 4)
        if self.body_slew:
            return round(float(np.mean(self.body_slew)), 4)
        return None

    def clamped_frac(self) -> tuple[float | None, str]:
        term = (self.body_terminal or {}).get("data") or {}
        if isinstance(term.get("clamped_frac_total"), (int, float)):
            return round(float(term["clamped_frac_total"]), 4), "body"
        bp = self.body_progress or {}
        if isinstance(bp.get("clamped_frac_total"), (int, float)):
            return round(float(bp["clamped_frac_total"]), 4), "body"
        if self.client_clamp_all:
            return round(float(np.mean(self.client_clamp_all)), 4), "client (groot.actions.clamp_stats, 0.02 rad)"
        return None, "none"

    def emit(self, type: str, **fields: Any) -> None:
        self.events.append((type, fields))


# ================================================================================================ client thread
class GrootArmClient(threading.Thread):
    """P5 -> PolicyServer -> body `arm` op, for one session (docs/groot_arms_design.md §5.2)."""

    def __init__(self, s: Session, policy: PolicyPort, sensors: SensorPort, arm: ArmPort, helpers: dict,
                 on_down: Callable[[str], None]):
        super().__init__(name=f"groot-client-{s.id}", daemon=True)
        self.s, self.policy, self.sensors, self.arm, self.h = s, policy, sensors, arm, helpers
        self.on_down = on_down
        self.cfg = s.cfg

    def _due(self, now: float) -> bool:
        s = self.s
        if s.last_t0 is None:
            return True
        if now - s.last_t0 >= self.cfg.replan_s:
            return True
        return (s.last_t0 + s.last_T * s.last_dt - now) / s.last_dt < self.cfg.min_rows_left

    def _keepalive(self) -> None:
        self.s.keepalive(self.arm)

    def _error(self, e: Exception) -> bool:
        """Count one policy error; True when the policy is to be declared unavailable."""
        s = self.s
        s.errors += 1
        s.errors_total += 1
        s.last_error = f"{type(e).__name__}: {e}"[:200]
        return s.errors >= self.cfg.max_errors

    def _unavailable(self) -> None:
        s = self.s
        detail = (f"PolicyServer {self.cfg.endpoint}: {s.errors} consecutive errors "
                  f"(get_action timeout {self.cfg.timeout_s:.1f} s): {s.last_error}")
        with s.lock:
            if s.verdict is None:
                s.verdict = ("failed", "policy_unavailable", detail)
        s.fence("policy_unavailable")
        self.on_down(detail)

    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001 - the thread must always leave a verdict
            with self.s.lock:
                if self.s.verdict is None:
                    self.s.verdict = ("failed", "internal_error", f"groot client crashed: {e!r}"[:300])
            self.s.fence("internal_error")
        finally:
            try:
                self.policy.close()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self) -> None:
        s, cfg = self.s, self.cfg
        while s.fenced is None:
            now = time.monotonic()
            if not self._due(now):
                self._keepalive()
                time.sleep(0.01)
                continue
            frame, t_frame = self.sensors.ego_frame()
            if cfg.frame_sync and (frame is None or time.monotonic() - t_frame > cfg.frame_max_age_s
                                   or (s.last_frame_t is not None and t_frame <= s.last_frame_t)):
                frame, t_frame = self._fresh_frame(t_frame if frame is not None else 0.0)
                if s.fenced is not None:
                    return
            dbg, t_dbg = self.sensors.debug_state()
            now = time.monotonic()
            f_age = now - t_frame if frame is not None else math.inf
            d_age = now - t_dbg if dbg is not None else math.inf
            if f_age > cfg.frame_max_age_s or d_age > cfg.state_max_age_s or dbg.get("body_q") is None:
                s.obs_stale += 1
                s.last_obs_age = (round(f_age, 3), round(d_age, 3))
                self._keepalive()
                time.sleep(0.02)
                continue
            hands = [dbg.get(f"{side}_hand_q") for side in ("left", "right")]
            try:
                obs = self.h["build_obs"](frame, dbg["body_q"], *[[0.0] * N_HAND if q is None else q for q in hands],
                                          s.prompt)
            except ValueError as e:
                s.dropped["obs_invalid"] += 1
                s.last_error = f"observation: {e}"[:200]
                self._keepalive()
                time.sleep(0.02)
                continue
            t_req = time.monotonic()
            try:
                action = self.policy.get_action(obs)
            except Exception as e:  # noqa: BLE001 - PolicyTimeout, PolicyError, ZMQ errors
                s.emit("groot.inference", ok=False, error=f"{type(e).__name__}: {e}"[:200],
                       latency_ms=round((time.monotonic() - t_req) * 1000.0, 1))
                # the second error: a timed-out server gets one short ping, so a dead one is known in <= 2.4 s
                down = self._error(e)
                if not down and s.fenced is None and not self._ping():
                    down = self._error(TimeoutError(f"no answer to ping ({cfg.ping_timeout_s:.1f} s)"))
                if down:
                    self._unavailable()
                    return
                self._keepalive()
                continue
            lat = (time.monotonic() - t_req) * 1000.0
            s.errors = 0
            s.inferences += 1
            s.last_frame_t = t_frame
            s.latencies_ms.append(lat)
            if s.fenced is not None:
                s.dropped["stale_session"] += 1
                s.emit("groot.inference", ok=True, latency_ms=round(lat, 1), dropped="stale_session")
                return
            self._handle_action(action, t_dbg, lat)

    def _fresh_frame(self, after: float) -> tuple[np.ndarray | None, float]:
        """frame_sync: wait for a frame captured after `after` (the camera renders at about the inference rate, so
        the newest frame is often most of a period old), up to one camera period + 0.3 s, keeping the body's session
        alive; returns the newest frame either way (an old one then counts as stale in the loop)."""
        s, cfg = self.s, self.cfg
        wait = getattr(self.sensors, "wait_frame", None)
        t0 = time.monotonic()
        t_end = t0 + 1.0 / max(cfg.camera_hz, 0.5) + 0.3
        s.frame_waits += 1
        while s.fenced is None and time.monotonic() < t_end:
            if callable(wait):
                if wait(after, min(0.1, max(t_end - time.monotonic(), 0.0))):
                    break
            else:
                _, t = self.sensors.ego_frame()
                if t > after:
                    break
                time.sleep(0.01)
            self._keepalive()
        s.frame_wait_s += time.monotonic() - t0
        return self.sensors.ego_frame()

    def _ping(self) -> bool:
        try:
            return bool(self.policy.ping(timeout_s=self.cfg.ping_timeout_s))
        except TypeError:                                   # a PolicyPort without the timeout argument
            return bool(self.policy.ping())
        except Exception:  # noqa: BLE001
            return False

    def _drop(self, why: str, lat: float) -> None:
        self.s.dropped[why] += 1
        self.s.emit("groot.inference", ok=True, latency_ms=round(lat, 1), dropped=why)
        self._keepalive()

    def _handle_action(self, action: dict, t_obs: float, lat: float) -> None:
        s, cfg = self.s, self.cfg
        try:
            finite = all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in action.values())
        except (TypeError, ValueError):
            finite = False
        if not finite:
            return self._drop("nan", lat)
        try:
            chunk = self.h["to_chunk"](action, t0_mono=t_obs)
        except (ValueError, TypeError) as e:
            s.last_error = f"chunk: {e}"[:200]
            return self._drop("invalid", lat)
        T, dt = int(chunk.upper_body.shape[0]), float(chunk.dt)
        now = time.monotonic()
        if now + cfg.lead_s - t_obs > T * dt - cfg.expire_margin_s:        # the body's index runs lead_s ahead
            return self._drop("expired", lat)
        frac = None
        if self.h.get("clamp_stats") is not None:
            try:
                frac = float(self.h["clamp_stats"](chunk, margin=0.02)["clamped_frac"])
            except Exception:  # noqa: BLE001
                frac = None
        seq = s.seq + 1
        msg = {**s.base(), "chunk": {"seq": seq, "t0_mono": float(t_obs), "dt": dt,
                                     "order": getattr(chunk, "order", "wire"), "upper_body": _rows(chunk.upper_body),
                                     "left_hand": _rows(chunk.left_hand), "right_hand": _rows(chunk.right_hand),
                                     "inference_ms": round(lat, 1)}}
        rep = s.publish(self.arm, msg)
        if rep is None:
            return self._drop("stale_session", lat)
        if not rep.get("ok"):
            s.body_rejected(rep)
            s.emit("groot.inference", ok=True, latency_ms=round(lat, 1), dropped=body_reject_key(rep))
            return
        s.body_ok()
        s.seq = seq
        body_drop = (rep.get("data") or {}).get("dropped")
        if body_drop:                                       # the body counted it (expired / out_of_order)
            s.dropped[f"body:{body_drop}"] += 1
        else:
            s.chunks_sent += 1
            s.last_t0, s.last_T, s.last_dt = float(t_obs), T, dt
            s.sent_log.append((time.monotonic(), seq))
            if frac is not None:
                s.client_clamp.append((time.monotonic(), frac))
                s.client_clamp_all.append(frac)
        s.emit("groot.inference", ok=True, latency_ms=round(lat, 1), chunk_idx=seq, dropped=body_drop)


# ================================================================================================ GT outcome
class GtJudge:
    """The ground-truth outcome at 10 Hz, through the WorldModel only (PLAN §6.4 verify; design §5.5)."""

    def __init__(self, world: Any, object_id: str, arm: str, cfg: GrootArmsConfig, success: dict | None = None):
        sc = dict(success or {})
        self.world, self.oid, self.arm, self.cfg = world, object_id, arm, cfg
        self.min_lift = float(sc.get("min_lift_m", cfg.min_lift_m))
        self.max_dist = float(sc.get("max_dist_m", cfg.max_palm_dist_m))
        self.hold_s = float(sc.get("hold_s", cfg.hold_s))
        o = world.object(object_id)
        self.z0 = float(o.box[0][2]) if o is not None else None
        self.pose_source = getattr(o, "pose_source", None) if o is not None else None
        self.palm_fn = getattr(world, "palm_position", None)
        self.lift_since: float | None = None
        self.closed_since: float | None = None
        self.ever_lifted = False
        self.lifted_now = False
        self.lift_max = 0.0
        self.palm_min: float | None = None
        self.palm_available = False
        self.samples = 0

    def _palm(self) -> tuple[float, float, float] | None:
        if not callable(self.palm_fn):
            return None
        try:
            p = self.palm_fn(self.arm)
        except Exception:  # noqa: BLE001 - NotSupported before P1.5
            return None
        if p is None:
            return None
        self.palm_available = True
        return float(p[0]), float(p[1]), float(p[2])

    def step(self, now: float, closure: float | None) -> tuple[str, str, str] | None:
        cfg = self.cfg
        pose = self.world.robot_pose()
        if getattr(pose, "fallen", False):
            return "failed", "fell", "the robot fell during the skill"
        o = self.world.object(self.oid)
        if o is None or self.z0 is None:
            return None
        self.samples += 1
        c = o.pos
        lift = float(o.box[0][2]) - self.z0
        self.lift_max = max(self.lift_max, lift)
        palm = self._palm()
        palm_d = math.dist(c, palm) if palm is not None else None
        if palm_d is not None:
            self.palm_min = palm_d if self.palm_min is None else min(self.palm_min, palm_d)
            near = palm_d <= self.max_dist
        else:                               # INTERIM (no palm poses before P1.5): lifted and still within reach
            near = math.hypot(c[0] - pose.x, c[1] - pose.y) <= cfg.reach_max_m
        self.lifted_now = lift >= self.min_lift and near
        if self.lifted_now:
            self.ever_lifted = True
            self.lift_since = self.lift_since if self.lift_since is not None else now
            if now - self.lift_since >= self.hold_s:
                return ("succeeded", "gt_lifted", f"lifted {lift:.3f} m for {self.hold_s:.1f} s"
                        + (f", {palm_d:.3f} m from the palm" if palm_d is not None else ""))
        else:
            self.lift_since = None
        if self.ever_lifted and not self.lifted_now and (
                lift < cfg.drop_below_m or (palm_d is not None and palm_d > cfg.drop_palm_dist_m)):
            return "failed", "object_dropped", f"lifted {self.lift_max:.3f} m, now {lift:.3f} m above its support"
        closed = closure is not None and closure >= cfg.closure_thresh
        missed = closed and not self.lifted_now and (palm_d is None or palm_d > self.max_dist)
        if missed:
            self.closed_since = self.closed_since if self.closed_since is not None else now
            if now - self.closed_since >= cfg.grasp_missed_s:
                return ("failed", "grasp_missed", f"hand {closure:.2f} closed for {cfg.grasp_missed_s:.1f} s, object "
                        f"lifted {lift:.3f} m" + (f", {palm_d:.3f} m from the palm" if palm_d is not None else ""))
        else:
            self.closed_since = None
        return None

    def summary(self) -> dict:
        return {"predicate": "gt_lifted", "min_lift_m": self.min_lift, "max_dist_m": self.max_dist,
                "hold_s": self.hold_s, "lift_max_m": round(self.lift_max, 3), "lifted_now": self.lifted_now,
                "palm": "world.palm_position" if self.palm_available else "unavailable (P1.5 link poses): lift "
                "within reach_max_m of the pelvis (INTERIM)",
                "palm_min_m": None if self.palm_min is None else round(self.palm_min, 3),
                "pose_source": self.pose_source, "samples": self.samples}


# ================================================================================================ health
class PolicyHealth:
    """Background pings of the PolicyServer (and one warm-up get_action). Health must be cheap, so capability checks
    read this cached state. A session that sees 2 consecutive errors marks it down at once; only a ping that STARTED
    after that can bring it back."""

    def __init__(self, make_policy: Callable[[float], PolicyPort] | None, cfg: GrootArmsConfig,
                 warmup: Callable[[PolicyPort], None] | None = None):
        self.make_policy, self.cfg, self._warmup = make_policy, cfg, warmup
        self.lock = threading.Lock()
        self.state = "starting"
        self.detail = f"PolicyServer {cfg.endpoint}: first ping pending"
        self.warm = not cfg.warmup
        self.gen = 0
        self.t_ok: float | None = None
        self.transitions: collections.deque = collections.deque()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def ensure_started(self) -> None:
        if self._thread is None and self.make_policy is not None:
            self._thread = threading.Thread(target=self._run, name="groot-health", daemon=True)
            self._thread.start()

    def _set(self, state: str, detail: str, gen: int | None = None) -> None:
        with self.lock:
            if gen is not None and gen != self.gen:
                return                                   # a ping that started before a mark_down
            if state != self.state:
                self.transitions.append({"ok": state == "ok", "state": state, "detail": detail})
            self.state, self.detail = state, detail
            if state == "ok":
                self.t_ok = time.monotonic()

    def mark_down(self, detail: str) -> None:
        with self.lock:
            self.gen += 1
        self._set("down", detail)
        self._wake.set()

    def check_now(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        client = None
        ep = self.cfg.endpoint
        while not self._stop.is_set():
            with self.lock:
                gen = self.gen
            why = f"PolicyServer {ep}: no answer to ping ({self.cfg.ping_timeout_s:.1f} s)"
            try:
                if client is None:
                    client = self.make_policy(self.cfg.ping_timeout_s)
                ok = bool(client.ping(timeout_s=self.cfg.ping_timeout_s))
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"PolicyServer {ep}: {type(e).__name__}: {e}"[:300]
            if ok and not self.warm and self._warmup is not None:
                try:
                    self._warmup(client)
                    self.warm = True
                except Exception as e:  # noqa: BLE001
                    ok, why = False, f"PolicyServer {ep}: warm-up get_action failed: {type(e).__name__}: {e}"[:300]
            if ok:
                self._set("ok", f"PolicyServer {ep} answers", gen)
            else:
                self._set("down", why, gen)
            self._wake.wait(self.cfg.ping_period_s)
            self._wake.clear()
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.warmup_timeout_s + 1.0)


# ================================================================================================ executor
class GrootArmExecutor:
    """ManipExecutor (api/services.py) for backend `groot`. Built by `create(ctx)` from the executor registry
    (services/executors/registry.py); tests build it directly with fakes for every port."""

    backend = BACKEND

    def __init__(self, world: Any, *, arm: ArmPort | None, sensors: SensorPort | None,
                 cfg: GrootArmsConfig | None = None, gate: Any = None, events: Any = None, name: str = EXECUTOR,
                 helpers: dict | None = None, skill_lookup: Callable[[str], Any] | None = None):
        self.world, self.arm, self.sensors = world, arm, sensors
        self.cfg = cfg or GrootArmsConfig()
        self.gate, self.events, self.name = gate, events, name
        self.skill_lookup = skill_lookup
        self.missing: list[str] = []
        if helpers is None:
            try:
                helpers = _groot_helpers()
            except ImportError as e:
                helpers = {}
                self.missing.append(f"groot package not importable ({e})")
        self.h = helpers
        if arm is None:
            self.missing.append("the body has no wl-body `arm` op (lite profile or no BodyClient)")
        if sensors is None:
            self.missing.append("no GR00T sensors (ego_view frame + g1_debug)")
        make = None
        if "policy" in self.h:
            make = lambda t: self.h["policy"](self.cfg.endpoint, t)   # noqa: E731
        self.policy_health = PolicyHealth(make, self.cfg, self._warmup if self.cfg.warmup else None)
        self._cancel = False
        self.last_session: Session | None = None
        self.last_client: GrootArmClient | None = None
        self._closed = False

    # -- health ----------------------------------------------------------------------------------------------
    def health(self) -> ServiceHealth:
        if self.missing:
            return ServiceHealth(False, "down", f"{self.name} not wired: " + "; ".join(self.missing))
        chunk = getattr(self.arm, "supports_chunk", None)
        if callable(chunk) and chunk() is False:
            return ServiceHealth(False, "down", "the body's arm op has no chunk mode yet (B.8, "
                                                "docs/contracts/arm_chunk.md)")
        ph = self.policy_health
        ph.ensure_started()
        with ph.lock:
            state, detail = ph.state, ph.detail
        if state == "ok":
            return ServiceHealth(True, "ok", f"{self.name} ({LABEL}): {CHECKPOINT} at {self.cfg.endpoint}")
        return ServiceHealth(False, state, detail)

    def _warmup(self, client: PolicyPort) -> None:
        """One get_action on a black frame at zero state. It builds its observation with helpers["build_obs_warmup"]
        when given (a recorder wraps "build_obs" and must not count this synthetic observation as evidence)."""
        build = self.h.get("build_obs_warmup") or self.h["build_obs"]
        obs = build(np.zeros((480, 640, 3), np.uint8), [0.0] * 29, [0.0] * N_HAND, [0.0] * N_HAND,
                    self.h.get("prompt") or "move the apple to the plate")   # any non-empty prompt
        client.get_action(obs, timeout_s=self.cfg.warmup_timeout_s)

    def close(self) -> None:
        self._closed = True
        self.policy_health.close()
        for x in (self.sensors, self.arm):
            c = getattr(x, "close", None)
            if callable(c):
                try:
                    c()
                except Exception:  # noqa: BLE001
                    pass

    async def cancel(self) -> None:
        """ManipulationService calls this on its own timeout. The running session fences at the next tick."""
        self._cancel = True
        s = self.last_session
        if s is not None and s.fence("cancelled"):
            await asyncio.to_thread(s.drain)

    # -- events ----------------------------------------------------------------------------------------------
    def _emit(self, type: str, **fields: Any) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(type, **fields)
        except Exception:  # noqa: BLE001
            pass

    def _flush(self, s: Session | None) -> None:
        ph = self.policy_health
        while ph.transitions:
            t = ph.transitions.popleft()
            self._emit("policy.health", endpoint=self.cfg.endpoint, checkpoint=CHECKPOINT, executor=self.name, **t)
        if s is None:
            return
        while s.events:
            type_, f = s.events.popleft()
            self._emit(type_, session=s.id, execution_id=s.id, executor=self.name, **f)

    def _phase(self, s: Session, phases: list[dict], phase: str, t0: float, **extra: Any) -> None:
        phases.append({"phase": phase, "s": round(time.monotonic() - t0, 3), **extra})
        self._emit("manip.phase", execution_id=s.id, phase=phase, skill=getattr(s.skill, "skill_id", s.job.skill_id),
                   executor=self.name, label=LABEL)

    # -- helpers ---------------------------------------------------------------------------------------------
    def _skill(self, job: ManipJob) -> Any:
        sk = getattr(job, "skill", None)
        if sk is None and self.skill_lookup is not None:
            sk = self.skill_lookup(job.skill_id)
        if sk is None:
            try:
                from services.skills import load_skill_specs
                sk = next((x for x in load_skill_specs() if x.skill_id == job.skill_id), None)
            except Exception:  # noqa: BLE001
                sk = None
        return sk

    def _prompt(self, skill: Any, job: ManipJob) -> str:
        otype = getattr(job, "object_type", "") or ""
        if not otype:
            o = self.world.object(job.object_id)
            otype = getattr(o, "type", "") if o is not None else ""
        tpl = (getattr(skill, "prompt_template", "") or self.h.get("prompt") or "").strip()
        try:
            return tpl.format(label=otype.replace("_", " "), type=otype, arm=job.arm)
        except (KeyError, IndexError, ValueError):
            return tpl

    def _closure(self, arm: str) -> float | None:
        dbg, _ = self.sensors.debug_state()
        q = (dbg or {}).get(f"{arm}_hand_q")
        if q is None or len(q) != N_HAND:
            return None
        return hand_closure(arm, q)

    def _body_event(self, s: Session):
        def cb(ev: dict) -> None:
            d = ev.get("data") or {}
            if ev.get("id") != s.op_id and d.get("session_id") != s.id:
                return
            st = ev.get("state")
            if st == "progress":
                s.body_progress = d
                s.t_body_progress = time.monotonic()
                if isinstance(d.get("clamped_frac"), (int, float)):
                    s.body_clamp.append((s.t_body_progress, float(d["clamped_frac"])))
                if isinstance(d.get("slew_frac"), (int, float)):
                    s.body_slew.append(float(d["slew_frac"]))
                s.emit("arm.progress", chunk_idx=d.get("chunk_seq"), k=d.get("k"), stall_s=d.get("stall_s"),
                       clamped_frac=d.get("clamped_frac"), latency_ms=d.get("latency_ms"),
                       cross_fades=d.get("cross_fades"), inferences=s.inferences, dropped=dict(s.dropped),
                       source="body")
            elif st in ("succeeded", "canceled", "failed"):
                s.body_terminal = ev
        return cb

    async def _stance(self, s: Session) -> tuple[str, str] | None:
        cfg = self.cfg
        t_end = time.monotonic() + cfg.settle_s
        while True:
            p = self.world.robot_pose()
            if getattr(p, "fallen", False):
                return "fell", "the robot is down"
            if p.speed <= cfg.rest_speed and abs(p.wz) <= cfg.rest_wz:
                return None
            if time.monotonic() >= t_end:
                return "base_moving", f"base still moving after {cfg.settle_s:.1f} s ({p.speed:.2f} m/s, " \
                                      f"{math.degrees(p.wz):.0f} deg/s)"
            await asyncio.sleep(0.05)

    def _caps(self) -> dict:
        try:
            return dict(self.world.capabilities()) if hasattr(self.world, "capabilities") else {}
        except Exception:  # noqa: BLE001
            return {}

    def _view(self, s: Session) -> dict:
        cam = self.cfg.camera
        if not self._caps().get(f"detections:{cam}"):
            return {"ran": False, "note": f"view check skipped: the world has no {cam} detections (P1.6 / R.3); "
                                          "INTERIM"}
        try:
            dets = self.world.detections(camera=cam, method="best")
        except TypeError:                                   # a world without `method`
            dets = self.world.detections(camera=cam)
        mine = [d for d in dets if d.id == s.job.object_id]
        px = max((float(d.px) for d in mine), default=0.0)
        method = getattr(mine[0], "method", None) if mine else getattr(dets[0], "method", None) if dets else None
        return {"ran": True, "camera": cam, "px": round(px, 1), "min_px": self.cfg.view_min_px, "method": method,
                "ok": px >= self.cfg.view_min_px}

    async def _camera(self, s: Session, on: bool, notes: list[str] | None = None, hz: float | None = None) -> None:
        """P1 renders ego_view only while a consumer holds it (docs/contracts/p1_m2b.md §5.4): the session is the
        consumer, with a TTL so a crashed runtime cannot leave the camera rendering."""
        if not self._caps().get("enable_camera"):
            if on and notes is not None:
                notes.append(f"{self.cfg.camera} render toggle not available (world has no enable_camera); P1 "
                             f"renders it or not by its own config (INTERIM)")
            return
        kw: dict[str, Any] = {"consumer": s.id, "ttl_s": self.cfg.camera_ttl_s}
        if on:
            kw["hz"] = hz or self.cfg.camera_hz
        try:
            try:
                await asyncio.to_thread(self.world.enable_camera, self.cfg.camera, on, **kw)
            except TypeError:                               # a world whose enable_camera has no `hz`
                kw.pop("hz", None)
                await asyncio.to_thread(self.world.enable_camera, self.cfg.camera, on, **kw)
        except Exception as e:  # noqa: BLE001 - NotSupported on a P1 without the op, a P1 timeout
            if notes is not None:
                notes.append(f"{self.cfg.camera} enable_camera({on}) failed: {e}"[:200])

    async def _head_rate(self, s: Session, on: bool, notes: list[str] | None = None) -> None:
        """session_head_hz: the head camera at that rate while the session renders ego_view, then back to the rate it
        had. It keeps its own `default` consumer (no TTL), so only the rate changes; a runtime that dies mid-session
        leaves the lower rate until the next session or a restart of P1 (docs/groot_serving.md §9)."""
        hz = float(self.cfg.session_head_hz or 0.0)
        if hz <= 0 or not self._caps().get("enable_camera"):
            return
        try:
            if on:
                rep = await asyncio.to_thread(self.world.enable_camera, "head", True, consumer="default")
                prev = (rep or {}).get("hz") if isinstance(rep, dict) else None
                if isinstance(prev, (int, float)) and prev > 0 and s.head_prev_hz is None:
                    s.head_prev_hz = float(prev)
                await asyncio.to_thread(self.world.enable_camera, "head", True, consumer="default", hz=hz)
            elif s.head_prev_hz is not None:
                await asyncio.to_thread(self.world.enable_camera, "head", True, consumer="default",
                                        hz=s.head_prev_hz)
        except Exception as e:  # noqa: BLE001 - a world without the op, a P1 timeout: the rate stays as it was
            if notes is not None:
                notes.append(f"head camera rate {'set' if on else 'restore'} failed: {e}"[:200])

    async def _first_frame(self, t_on: float) -> float | None:
        """Seconds from enabling the camera to the first ego frame captured after it, None if none came within
        camera_wait_s (the client then counts stale observations and the session ends policy_stall)."""
        t_end = t_on + self.cfg.camera_wait_s
        while True:
            try:
                frame, t = self.sensors.ego_frame()
            except Exception:  # noqa: BLE001
                frame, t = None, 0.0
            if frame is not None and t >= t_on - 0.05:
                return round(max(t - t_on, 0.0), 3)
            if time.monotonic() >= t_end:
                return None
            await asyncio.sleep(0.02)

    async def _send(self, msg: dict, op_id: str | None = None) -> dict:
        try:
            return await asyncio.to_thread(self.arm.arm, msg, op_id)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "state": "rejected", "error": "body_timeout", "data": {"detail": repr(e)[:200]}}

    # -- run -------------------------------------------------------------------------------------------------
    async def run(self, job: ManipJob, handle: Any) -> GrootArmOutcome:
        t_run = time.monotonic()
        self._cancel = False
        cfg = self.cfg
        skill = self._skill(job)
        prompt = self._prompt(skill, job)
        s = Session(job, handle, skill, prompt, cfg)
        self.last_session = s
        phases: list[dict] = []
        notes = [f"{LABEL}: off-the-shelf {CHECKPOINT} (Arena apple-to-plate, left hand); a house grasp is not "
                 f"expected zero-shot"]
        extra: dict[str, Any] = {"view_check": None, "hold_on_end": None}

        def done(status: str, reason: str | None, phase: str, detail: str = "",
                 judge: GtJudge | None = None) -> GrootArmOutcome:
            return self._outcome(s, status, reason, phase, detail, phases, notes, extra, judge, t_run)

        if job.action != "pick":
            return done("failed", "policy_unavailable", "select_skill",
                        f"{self.name} has no {job.action} skill (GR00T place is future work; place is scripted)")
        h = self.health()
        self._flush(None)
        if not h.ok:
            return done("failed", "policy_unavailable", "select_skill", h.detail)
        # 1. stance
        t0 = time.monotonic()
        why = await self._stance(s)
        self._phase(s, phases, "stance_check", t0, ok=why is None)
        if why is not None:
            return done("failed", why[0], "stance_check", why[1])
        pose0 = self.world.robot_pose()
        extra["pose0"] = (pose0.x, pose0.y)
        # 2. the ego camera on (P1's `detections` answers camera_off otherwise), then the view check
        t0 = time.monotonic()
        await self._camera(s, True, notes, hz=cfg.camera_warm_hz or None)
        judge: GtJudge | None = None
        verdict: tuple[str, str | None, str] | None = None
        opened = False
        ended: str | None = None
        t_exec = time.monotonic()
        try:
            first = await self._first_frame(t0) if self.sensors is not None else None
            extra["camera_first_frame_s"] = first
            if first is None:
                notes.append(f"no {cfg.camera} frame within {cfg.camera_wait_s:.1f} s of enabling it")
            vc = self._view(s)
            extra["view_check"] = vc
            self._phase(s, phases, "view_check", t0, camera_first_frame_s=first,
                        **{k: v for k, v in vc.items() if k != "note"})
            if not vc["ran"]:
                notes.append(vc["note"])
            elif not vc["ok"]:
                return done("failed", "not_in_ego_view", "view_check",
                            f"{vc['px']:.0f} px of {job.object_id} in {cfg.camera} (< {cfg.view_min_px:.0f})")
            if self._fenced_by(handle, job, s):
                return self._early_fence(s, done)
            # the stream rate (camera_warm_hz only covered P1's warm-up frames) and the session's head rate
            if cfg.camera_warm_hz and cfg.camera_warm_hz != cfg.camera_hz:
                await self._camera(s, True, notes)
            await self._head_rate(s, True, notes)
            # 3. enter: the arm session opens with open hands
            t0 = time.monotonic()
            unsub = self.arm.subscribe(self._body_event(s))
            try:
                start = {**s.base(), "hold_on_end": "measured", "watchdog_s": cfg.watchdog_s, "lead_s": cfg.lead_s,
                         "left_hand": [0.0] * N_HAND, "right_hand": [0.0] * N_HAND,
                         "hands_blend_s": cfg.hands_open_s}
                rep = await self._send(start, s.op_id)
                s.t_last_msg = time.monotonic()
                if not rep.get("ok"):
                    key = body_reject_key(rep)[5:]
                    v = body_verdict(rep)
                    self._phase(s, phases, "enter", t0, ok=False, error=key)
                    return done("failed", v[0] if v else "controller_unavailable", "enter",
                                f"the body refused the arm session: {key}")
                opened = True
                await asyncio.sleep(cfg.hands_open_s)
                self._phase(s, phases, "enter", t0, ok=True, hands="open")
                # 4. execute
                t_exec = time.monotonic()
                s.t_exec = t_exec
                judge = GtJudge(self.world, job.object_id, job.arm, cfg, getattr(skill, "success", None))
                if judge.pose_source == "scene":
                    notes.append("the object's pose is static scene data (P1.2 get_objects missing): the GT "
                                 "outcome cannot see a lift")
                client = GrootArmClient(s, self.h["policy"](cfg.endpoint, cfg.timeout_s), self.sensors, self.arm,
                                        self.h, self.policy_health.mark_down)
                self.last_client = client
                client.start()
                verdict = await self._monitor(s, job, handle, judge, skill)
            finally:
                if opened:
                    ended = await self._end(s, judge, verdict)
                else:
                    s.fence("not_opened")
                unsub()
        finally:
            await self._camera(s, False)
            await self._head_rate(s, False, notes)
            self._flush(s)
        status, reason, detail = verdict
        self._phase(s, phases, "execute", t_exec, inferences=s.inferences, chunks_sent=s.chunks_sent,
                    chunks_dropped=dict(s.dropped), latency_ms={"p50": _p(s.latencies_ms, 50),
                                                                "p95": _p(s.latencies_ms, 95)},
                    clamped_frac=s.clamped_frac()[0])
        self._phase(s, phases, "carry_lock" if status == "succeeded" else "end", time.monotonic(), hold=ended)
        if status == "succeeded":
            return done(status, None, "verify", detail, judge)
        return done(status, reason, "execute", detail, judge)

    def _fenced_by(self, handle: Any, job: ManipJob, s: Session) -> str | None:
        if getattr(handle, "cancel_requested", False) or self._cancel:
            s.fence("cancelled")
        elif self.gate is not None and self.gate.halted_since(job.epoch):
            s.fence("halted")
        return s.fenced

    @staticmethod
    def _early_fence(s: Session, done: Callable[..., GrootArmOutcome]) -> GrootArmOutcome:
        s.drain()
        if s.fenced == "halted":
            return done("failed", "halted", "view_check", "halted before the arm session opened")
        return done("cancelled", "cancelled", "view_check", "cancelled before the arm session opened")

    async def _monitor(self, s: Session, job: ManipJob, handle: Any, judge: GtJudge, skill: Any
                       ) -> tuple[str, str | None, str]:
        cfg = self.cfg
        max_s = float(cfg.max_duration_s or getattr(skill, "max_duration_s", 25.0) or 25.0)
        deadline = time.monotonic() + max_s
        next_gt = time.monotonic()
        next_prog = time.monotonic()
        next_cam = time.monotonic() + cfg.camera_refresh_s
        cam_task: asyncio.Future | None = None
        ka_task: asyncio.Future | None = None
        stall_warned = False
        watch = None
        if hasattr(handle, "wait_cancel"):
            async def _watch() -> None:                    # fence the moment cancel() is called, not at the next tick
                await handle.wait_cancel()
                s.fence("cancelled")
            watch = asyncio.ensure_future(_watch())
        try:
            while True:
                self._flush(s)
                now = time.monotonic()
                if self._fenced_by(handle, job, s) is not None or s.verdict is not None:
                    break
                if s.body_terminal is not None:
                    return self._body_ended(s)
                if now >= next_gt:
                    next_gt = now + 1.0 / cfg.gt_hz
                    v = judge.step(now, self._closure(job.arm))
                    if v is not None:
                        return v
                stall = s.stall_s(now)
                if stall > cfg.stall_event_s and not stall_warned:
                    stall_warned = True
                    s.emit("groot.stall", stall_s=round(stall, 2))
                if stall > cfg.stall_fail_s:
                    why = f"no chunk rows left for {stall:.1f} s (inferences {s.inferences}, sent {s.chunks_sent}, " \
                          f"dropped {dict(s.dropped)}, obs_stale {s.obs_stale}"
                    why += f", last obs age frame/state {s.last_obs_age} s)" if s.last_obs_age else ")"
                    return "failed", "policy_stall", why
                frac = s.oob(now)
                if frac is not None:
                    return "failed", "policy_out_of_bounds", f"{frac:.0%} of the arm/hand targets clamped over " \
                                                             f"{cfg.oob_window_s:.1f} s"
                if now >= next_prog and (s.t_body_progress is None or now - s.t_body_progress > 0.5):
                    next_prog = now + 1.0 / cfg.progress_hz                # before B.8: the client's own view
                    s.emit("arm.progress", chunk_idx=s.seq, k=None if s.last_t0 is None else
                           int(min(max(round((now - s.last_t0) / s.last_dt), 0), max(s.last_T - 1, 0))),
                           stall_s=round(stall, 3), clamped_frac=s.client_clamp[-1][1] if s.client_clamp else None,
                           latency_ms=round(s.latencies_ms[-1], 1) if s.latencies_ms else None, cross_fades=None,
                           inferences=s.inferences, dropped=dict(s.dropped), source="client")
                if now - s.t_last_msg >= cfg.keepalive_s and (ka_task is None or ka_task.done()):
                    ka_task = asyncio.ensure_future(asyncio.to_thread(s.keepalive, self.arm))
                if now >= next_cam and (cam_task is None or cam_task.done()):
                    next_cam = now + cfg.camera_refresh_s          # renew the enable_camera lease (ttl_s)
                    cam_task = asyncio.ensure_future(self._camera(s, True))
                if now >= deadline:
                    return "failed", "timeout", f"no GT success within max_duration_s {max_s:.0f} s"
                await asyncio.sleep(0.02)
        finally:
            if watch is not None:
                watch.cancel()
            pending = [x for x in (cam_task, ka_task) if x is not None and not x.done()]
            if pending:
                await asyncio.wait(pending, timeout=1.0)
        # a halt or a cancel wins over what it causes: the body's latch can refuse the client's next chunk (stale
        # session / halted) a few ms before this loop sees the runtime gate
        if (self.gate is not None and self.gate.halted_since(job.epoch)) or s.fenced == "halted":
            return "failed", "halted", "halted: the stream stopped at once"
        if getattr(handle, "cancel_requested", False) or self._cancel or s.fenced == "cancelled":
            return "cancelled", (getattr(handle, "cancel_reason", None) or "cancelled"), \
                "cancelled: the stream stopped at once"
        if s.verdict is not None:
            return s.verdict
        return "failed", "internal_error", f"the session stopped ({s.fenced})"

    def _body_ended(self, s: Session) -> tuple[str, str, str]:
        ev = s.body_terminal or {}
        d = ev.get("data") or {}
        by = str(d.get("ended_by") or d.get("reason") or ev.get("state"))
        s.fence(f"body:{by}")
        reason = ENDED_BY.get(by, "controller_unavailable")
        why = f", reason {d['reason']}" if d.get("reason") and d.get("reason") != by else ""
        return "failed", reason, f"the body ended the arm session ({ev.get('state')}, ended_by {by}{why})"

    async def _end(self, s: Session, judge: GtJudge | None, verdict: tuple[str, str | None, str] | None
                   ) -> str | None:
        """Fence, wait for a send in flight (the ack), then end the body session with the right hold."""
        status, reason = (verdict[0], verdict[1]) if verdict else ("failed", "internal_error")
        s.fence(reason or status)
        await asyncio.to_thread(s.drain)
        hold = None
        if s.body_terminal is None and not (s.fenced or "").startswith("body:"):
            in_hand = bool(judge and judge.lifted_now)
            if status == "succeeded":
                hold = "target"
            elif status == "cancelled":
                hold = "target" if in_hand else "stand"
            else:
                hold = "measured"
            rep = await self._send({**s.base(), "end": True, "hold_on_end": hold, "reason": reason or status})
            if not rep.get("ok"):
                s.dropped[f"end:{rep.get('error')}"] += 1   # e.g. `halted` once the B.1 latch holds: harmless
        # the body's terminal event (published by its next 50 Hz tick) carries the session's own counters:
        # chunks applied, clamped_frac_total, slew_frac_total, max_step_rad (docs/contracts/m1.md §3.9)
        t_end = time.monotonic() + self.cfg.terminal_wait_s
        while s.body_terminal is None and time.monotonic() < t_end:
            await asyncio.sleep(0.01)
        return hold if hold is not None else ((s.body_terminal or {}).get("data") or {}).get("hold")

    def _outcome(self, s: Session, status: str, reason: str | None, phase: str, detail: str, phases: list[dict],
                 notes: list[str], extra: dict, judge: GtJudge | None, t_run: float, **_: Any) -> GrootArmOutcome:
        self.last_session = s
        pose = self.world.robot_pose()
        p0 = extra.get("pose0")
        shift = math.hypot(pose.x - p0[0], pose.y - p0[1]) if p0 else 0.0
        hold = next((ph.get("hold") for ph in reversed(phases) if "hold" in ph), None)
        clamped, clamped_src = s.clamped_frac()
        attempt = {"executor": self.name, "skill": getattr(s.skill, "skill_id", s.job.skill_id), "label": LABEL,
                   "status": status, "reason": reason, "duration_s": round(time.monotonic() - t_run, 2),
                   "inferences": s.inferences, "chunks_sent": s.chunks_sent}
        data = {
            "executor": self.name, "label": LABEL, "skill_label": getattr(s.skill, "label", LABEL),
            "checkpoint": getattr(s.skill, "checkpoint", None) or CHECKPOINT, "endpoint": self.cfg.endpoint,
            "prompt": s.prompt, "session_id": s.id, "generation": s.generation, "control_epoch": s.control_epoch,
            "inferences": s.inferences, "chunks_sent": s.chunks_sent, "chunks_dropped": dict(s.dropped),
            "keepalives": s.keepalives, "latency_ms": {"p50": _p(s.latencies_ms, 50), "p95": _p(s.latencies_ms, 95),
                                                       "n": len(s.latencies_ms)},
            "clamped_frac": clamped, "clamped_frac_source": clamped_src, "slew_frac": s.slew_frac(),
            "body": s.body_summary(), "stall_s_max": round(s.stall_max, 2),
            "base_shift_m": round(shift, 3), "obs_stale": s.obs_stale, "policy_errors": s.errors_total,
            "hold_on_end": hold, "carry": CARRY_LABEL if status == "succeeded" else None,
            "view_check": extra.get("view_check"), "camera_first_frame_s": extra.get("camera_first_frame_s"),
            "render": {"ego_hz": self.cfg.camera_hz, "warm_hz": self.cfg.camera_warm_hz or self.cfg.camera_hz,
                       "session_head_hz": self.cfg.session_head_hz or None, "head_restored_hz": s.head_prev_hz,
                       "frame_sync": self.cfg.frame_sync, "frame_waits": s.frame_waits,
                       "frame_wait_s": round(s.frame_wait_s, 2)},
            "gt": judge.summary() if judge is not None else None,
            "cancel_ack_ms": None if s.t_ack is None or s.t_fence is None else round((s.t_ack - s.t_fence) * 1000, 2),
            "notes": list(notes), "attempts": [attempt],
        }
        head = f"{self.name} ({LABEL}, {CHECKPOINT} zero-shot)"
        stats = f"{s.inferences} inferences, {s.chunks_sent} chunks" + (
            f", p50 {data['latency_ms']['p50']:.0f} ms" if s.latencies_ms else "")
        text = f"{head}: {stats}" + (f"; {detail}" if detail else "")
        holding = bool(judge and judge.lifted_now) if judge is not None else False
        return GrootArmOutcome(status, reason, holding, phase, detail=text[:400], phases=list(phases), data=data)


# ================================================================================================ adapters
class BodyArmPort:
    """ArmPort over wl-body (ROUTER 5610 / PUB 5611): a private DEALER socket with a short reply timeout (the chunk
    stream never queues behind a go_to on the shared client socket), and the BodyClient's event listener."""

    def __init__(self, client: Any, timeout_s: float = 0.5):
        self.client = client
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._sock = None
        self._subs: list[Callable[[dict], None]] = []
        client.add_listener(self._on_event)

    @classmethod
    def of(cls, body: Any, timeout_s: float = 0.5) -> "BodyArmPort | None":
        client = getattr(body, "client", None)
        if client is None or not hasattr(client, "add_listener") or not hasattr(client, "ports"):
            return None
        return cls(client, timeout_s)

    def _open(self):
        import zmq
        from body.config import ep
        s = self.client.ctx.socket(zmq.DEALER)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(ep(self.client.ports["body_ctl"], self.client.host))
        self._sock = s
        return s

    def arm(self, args: dict, op_id: str | None = None) -> dict:
        rid = op_id or f"arm-{uuid.uuid4().hex[:10]}"
        with self._lock:
            sock = self._sock or self._open()
            sock.send(json.dumps({"id": rid, "op": "arm", "args": args}, separators=(",", ":")).encode())
            t_end = time.monotonic() + self.timeout_s
            while True:
                left = t_end - time.monotonic()
                if left <= 0 or not sock.poll(int(left * 1000) + 1):
                    sock.close(0)
                    self._sock = None
                    return {"id": rid, "ok": False, "state": "rejected", "error": "body_timeout"}
                rep = json.loads(sock.recv())
                if rep.get("id") == rid:
                    return rep

    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def _on_event(self, ev: dict) -> None:
        if ev.get("op") != "arm":
            return
        for cb in list(self._subs):
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                pass

    def supports_chunk(self) -> bool | None:
        """From body.state.arm.modes (B.8 lists "chunk"); None while the body has published no arm state."""
        st = getattr(self.client, "last_state", None) or {}
        arm = st.get("arm")
        if not isinstance(arm, dict):
            return None if not st else False
        return "chunk" in (arm.get("modes") or [])

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                self._sock.close(0)
                self._sock = None


class ZmqSensors:
    """SensorPort over ZMQ: the ego camera (P1 PUB 5566, gear_sonic sensor_server msgpack: {"images": {name: b64
    JPEG}, "t_capture_mono", ...}, docs/contracts/p1_m2b.md §5.2) and SONIC's g1_debug (deploy PUB 5557). The newest
    message of each is kept with its time (the frame's capture time when P1 sends it, else the receive time); a frame
    is decoded (Pillow) only when asked for."""

    def __init__(self, camera_ep: str, debug_ep: str, camera_key: str = "ego_view", swap_rb: bool = True):
        self.camera_ep, self.debug_ep, self.key, self.swap_rb = camera_ep, debug_ep, camera_key, swap_rb
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame: tuple[Any, float] | None = None
        self._decoded: tuple[int, np.ndarray] | None = None
        self._seq = 0
        self._debug: tuple[dict, float] | None = None
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self) -> "ZmqSensors":
        if not self._running:
            self._running = True
            for fn in (self._cam_loop, self._dbg_loop):
                t = threading.Thread(target=fn, daemon=True, name=f"groot-{fn.__name__}")
                t.start()
                self._threads.append(t)
        return self

    def _sub(self, endpoint: str, topic: bytes, conflate: bool):
        import zmq
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        if conflate:
            s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.SUBSCRIBE, topic)
        s.connect(endpoint)
        return s

    def _cam_loop(self) -> None:
        import msgpack
        s = self._sub(self.camera_ep, b"", True)
        while self._running:
            if not s.poll(100):
                continue
            try:
                d = msgpack.unpackb(s.recv(), raw=False)
            except Exception:  # noqa: BLE001
                continue
            img = (d.get("images") or {}).get(self.key)
            if img is None:
                continue
            now = time.monotonic()
            t_cap = d.get("t_capture_mono")                 # P1's capture time (p1_m2b.md §5.2), same box
            t = float(t_cap) if isinstance(t_cap, (int, float)) and t_cap <= now + 0.01 else now
            with self._lock:
                self._seq += 1
                self._frame = (img, t)
                self._new_frame.notify_all()
        s.close(0)

    def _dbg_loop(self) -> None:
        import msgpack
        s = self._sub(self.debug_ep, b"g1_debug", False)
        while self._running:
            if not s.poll(100):
                continue
            try:
                frames = s.recv_multipart()
                raw = frames[-1] if len(frames) > 1 else frames[0][len(b"g1_debug"):]
                d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception:  # noqa: BLE001
                continue
            with self._lock:
                self._debug = (d, time.monotonic())
        s.close(0)

    def ego_frame(self) -> tuple[np.ndarray | None, float]:
        self.start()
        with self._lock:
            fr, seq = self._frame, self._seq
            cached = self._decoded
        if fr is None:
            return None, 0.0
        if cached is not None and cached[0] == seq:
            return cached[1], fr[1]
        from PIL import Image
        raw = fr[0]
        data = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if self.swap_rb:
            rgb = np.ascontiguousarray(rgb[..., ::-1])
        with self._lock:
            self._decoded = (seq, rgb)
        return rgb, fr[1]

    def wait_frame(self, after: float, timeout_s: float) -> bool:
        """Block until a frame captured after `after` (monotonic) is held, or timeout_s; True when one is."""
        self.start()
        with self._lock:
            return self._new_frame.wait_for(lambda: self._frame is not None and self._frame[1] > after,
                                            timeout=max(0.0, timeout_s))

    def debug_state(self) -> tuple[dict | None, float]:
        self.start()
        with self._lock:
            return (None, 0.0) if self._debug is None else self._debug

    def close(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=0.5)


def create(ctx: Any) -> GrootArmExecutor:
    """The executor registry's factory (services/executors/registry.py: "groot_arms" -> backend "groot").

    Profile config: `groot_arms:` in config/profiles/<p>.yaml (StackProfile.raw), plus ctx.extras["groot_arms"].
    Nothing connects here: the sensors subscribe on the first session and the PolicyServer is pinged in the
    background once something asks for health."""
    raw = dict((getattr(getattr(ctx, "profile", None), "raw", None) or {}).get("groot_arms") or {})
    raw.update(dict((getattr(ctx, "extras", None) or {}).get("groot_arms") or {}))
    cfg = GrootArmsConfig.from_dict(raw)
    off = int(getattr(ctx, "port_offset", 0) or 0)
    arm = BodyArmPort.of(getattr(ctx, "body", None), cfg.arm_timeout_s)
    sensors = None
    if arm is not None:
        host = getattr(arm.client, "host", "127.0.0.1")
        sensors = ZmqSensors(f"tcp://{host}:{cfg.camera_port + off}", f"tcp://{host}:{cfg.debug_port + off}",
                             cfg.camera, cfg.camera_swap_rb)
    return GrootArmExecutor(ctx.world, arm=arm, sensors=sensors, cfg=cfg, gate=getattr(ctx, "gate", None),
                            events=getattr(ctx, "events", None), name=EXECUTOR)


__all__ = ["EXECUTOR", "BACKEND", "LABEL", "CHECKPOINT", "DEFAULT_ENDPOINT", "CARRY_LABEL", "ARENA_CLOSED_DEX3",
           "BODY_FATAL", "STALE_WHY", "ENDED_BY", "MAX_BODY_REJECTS", "body_verdict", "body_reject_key",
           "GrootArmsConfig", "PolicyPort", "ArmPort", "SensorPort", "GrootArmOutcome", "Session", "GrootArmClient",
           "GtJudge", "PolicyHealth", "GrootArmExecutor", "BodyArmPort", "ZmqSensors", "hand_closure", "create"]
