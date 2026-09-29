# Arm op, chunk mode (body B.8): the wire between `groot_arms` (P5) and wl-body (P3)

Status: **v0.2, 2026-09-29 (M2b wave 2, owner body-fix): the implemented wire.** v0.1 was written by the runtime owner
of `groot_arms` (R.4) before the body existed; the body implemented it in `body/arm.py` (body wave, master
`4b081c2` ... `84ddbfe`) with the differences that wave 1 listed in v0.1 §8. v0.2 folds those differences into the
text, adds the wave-2 body fixes (the halt latch keeps the hands' last **target**, the session fence covers every
message of a session, `stale_command` says why), and is what `body/arm.py` does. §8 lists every change from v0.1.

The runtime side is built against it: `services/executors/groot_arms.py` (owner ops-groot in wave 2) sends exactly
these messages; `tests/services/test_groot_arms_body_arm.py` runs the real executor against the real `ArmChannel`
(50 Hz, the body's own test plant `body/tests/arm_sim.py`), and `tests/fakes/fake_arm_body.py` is the in-process
reference body of v0.1 that the executor's offline tests use. `body/tests/test_arm_channel.py` replays the fake body's
sequences against the real channel (session start, chunks with lead 0, a late chunk, an out-of-order seq, a stale
session, a NaN chunk, cancel, halt) and pins every v0.2 item below.

Sources: `docs/groot_arms_design.md` §5.2-§5.5 (what GR00T needs), `docs/contracts/m1.md` §3.4 (envelopes), §3.9
(the `arm` op, holds, CarryLock), §3.10 (halt lane), §3.11 (fences), PLAN §5.5-§5.6 (fences, halt).

---

## 0. Summary

Chunk mode is **the same op `arm`, the same `stream` ownership and the same preempt rules** as the single-target
stream of v0.5. A session only adds a mode flag, three fence fields and timestamped action chunks. Every chunk is
played by P3 at 50 Hz, indexed by time since the observation it was computed from, so P5's asyncio, network and
inference jitter only change *which* row is played, never *when*.

```
P5 groot_arms ── GrootArmClient thread ──────────────────────────────────────────────────────────────────────┐
   │ SUB ego_view (P1 5566) + SUB g1_debug (5557)  →  obs  →  PolicyServer REQ (1.5 s)  →  action (40 x 50 Hz)   │
   │ t0_mono = receive time of that g1_debug                                                                   │
   └─ DEALER 5610  op "arm" {stream, session_id, generation, control_epoch, mode:"chunk", chunk{seq, t0_mono,   │
                             dt, order, upper_body T×17, left_hand T×7, right_hand T×7}}                        │
P3 wl-body ArmChannel (chunk mode), every 50 Hz tick:                                                          │
   x = (now + lead_s − t0_mono) / dt → rows ⌊x⌋, ⌊x⌋+1 interpolated (cross-fade 5 ticks on a new chunk; hold the│
   last row and count stall once x ≥ T − 1) → servo → clamp (URDF − 0.02 rad) → slew limit → SonicMux: every    │
   planner message carries upper_body_position (wire order, joint_map.wire_from_mj17) + both Dex3 hands → P2   │
   PUB 5611 body.event {op:"arm", state:"progress", data:{kind:"chunk", chunk_seq, k, stall_s, clamped_frac …}} ─┘
```

**Clock rule.** `t0_mono` is `time.monotonic()` (CLOCK_MONOTONIC) on the host that runs P5. P5 and P3 run on the
same host (the main box), so P3 compares it with its own `time.monotonic()`. The PolicyServer may be remote (OD3: dev
box over an SSH tunnel); that only adds latency, which the time index absorbs. If P5 and P3 ever run on different
hosts this contract does not hold (a `t0_wall` would be needed; not specified).

---

## 1. Messages (op `arm`, DEALER → ROUTER 5610; envelope as `docs/contracts/m1.md` §3.4)

Every message of a chunk session carries these fields, **including** `keepalive`, `end` and `release` (§1.3, §4).
The first one opens the session and creates the op record (reply `accepted`, event `accepted`), exactly like the
first message of a v0.5 stream. Later ones are stream updates: reply `done`, no event.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `stream` | str | yes | Ownership key, as in v0.5. `groot_arms` uses `stream = session_id` |
| `mode` | `"chunk"` | yes on the start and on chunks | Chunk mode. v0.5 messages have no `mode` (= `"target"`). A session's mode is fixed: a v0.5 target message on a chunk session's stream is rejected `mode_mismatch` |
| `session_id` | str | yes | The fence id. `groot_arms` uses the `manipulate` execution id. Missing on a message acting on a chunk session → `bad_args` |
| `generation` | int | yes | The execution's generation (PLAN §5.5). Missing on a chunk-mode message → `bad_args` |
| `control_epoch` | int | yes | The execution's control epoch (PLAN §5.5); compared with the halt latch (§4). Missing on a chunk-mode message → `bad_args` |
| `execution_id` | str | no | For B.2 leases; `groot_arms` sends it (= `session_id`) |
| `t_wall` | float | yes | Sender `time.time()`: a message older than the session's `watchdog_s` is dropped `stale_command` with `data.why: "t_wall"` (only that message; the session goes on) |

The body's fences (m1.md §3.11) run first on every `arm` message: while latched, `halted`; after a resume a
`control_epoch <= halt_epoch` is `stale_command` with `data.why: "control_epoch"`; an older `generation` is
`stale_command` with `data.why: "generation"`; another execution than a held lease's owner is `body_busy`. The
fence answers carry no event (the reply only). `groot_arms` keys `stale_command` on `data.why`: the fence reasons
end its session, `t_wall` is only counted.

### 1.1 Session start (first message)

| Field | Default | Meaning |
|---|---|---|
| `hold_on_end` | `"measured"` | What the channel does when the session ends without an explicit `hold_on_end` (the watchdog): §5 |
| `watchdog_s` | 2.0 (0.5-5.0) | No message of this session for this long ⇒ the session ends **`failed`**, reason `client_silent`, `ended_by: "watchdog"`, into `hold_on_end` (§5), and the body publishes `body.fault{kind: policy_lost}` (an event, not a FAULT mode). 2.0 s, not the design's 1.0 s: a policy request may block 1.5 s (REQ timeout), and the runtime sends a `keepalive` after every inference that produced no chunk |
| `lead_s` | **0.15** (0-0.3) | Play rows this far ahead of time: SONIC's arm lag is ~0.15 s (`docs/arm_tracking.md` §0, G0 condition 2). An explicit value wins; `groot_arms` sends 0.15 (`GrootArmsConfig.lead_s`) and counts it in its own expiry check |
| `xfade_ticks` | 5 (1-50) | Cross-fade length on a new chunk (100 ms) |
| `max_vel` | the channel's `arm_max_vel` (6 rad/s) | Slew limit on the arm joints, as v0.5 |
| `blend_s` | 1.5 | Blend time for `hold_on_end: stand` and `release` |
| `left_hand`, `right_hand` | – | Optional start targets (7 Dex3 values each, or a closure 0..1), blended in over `hands_blend_s`. `groot_arms` sends open hands (all 0): the Arena demos start with open hands |
| `hands_blend_s` | 0.3 | Blend time for the start hands |
| `preempt` | false | v0.5 semantics (§4) |
| servo args | the channel's | `servo_ki`, `servo_model`, ... as the v0.5 `arm` op (m1.md §3.9 "Servo") |
| `chunk` | – | Optional: the first chunk may ride on the start message |

Until the first chunk plays, the channel sends the pose it was sending when the session started (the continuity
rule of every take-over: SONIC's own reference when nothing drove the arms, else the held pose), plus the start
hands. The first chunk cross-fades in from there.

### 1.2 Chunk

`chunk` is an object:

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | Monotonic per session (1, 2, ...). A chunk whose `seq` is not above the playing one's is dropped `out_of_order` |
| `t0_mono` | float | CLOCK_MONOTONIC time that row 0 belongs to: the receive time of the `g1_debug` sample the observation was built from (design §5.2 step 1). Also dropped `out_of_order` when it is older than the playing chunk's `t0_mono` |
| `dt` | float | Row period. 0.02 (50 Hz, the N1.7 action horizon's rate). Accepted range 0.005-0.1 |
| `order` | `"wire"` \| `"mj17"` | Joint order of `upper_body`. **`"wire"`** = SONIC's IsaacLab-interleaved order (`joint_map.UPPER_BODY_JOINTS`: waist yaw/roll/pitch, then left/right pairs joint by joint), what `groot.actions.to_arm_chunk` produces. `"mj17"` = the v0.5 API order (`joint_map.UPPER_BODY_MUJOCO_JOINTS` = `g1_debug.body_q[12:29]`). Required, no default: an order mix-up is a rejection, not a silent swap of the arms. P3 converts `"wire"` with `joint_map.mj17_from_wire` and sends with `wire_from_mj17` |
| `upper_body` | T×17 float | Absolute joint targets (rad), in `order` |
| `left_hand`, `right_hand` | T×7 float | Absolute Dex3 targets in **Dex3 order** (`thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1`), both always present (the deploy reuses a stale value for a missing hand, `input_interface.hpp:341-362`). GR00T's `index…thumb` order is converted by name in `groot/`, never in P3 |
| `waist` | `"ref"` \| `"cmd"` | Optional, default `"ref"`: the waist entries are replaced by SONIC's live reference waist. GR00T's waist output is identically 0 in the Arena data and is never executed (design §2.5) |
| `inference_ms` | float | Optional telemetry (echoed in `arm.progress`) |

1 ≤ T ≤ 64, every row finite, shapes consistent. Otherwise the whole message is rejected `bad_chunk` with no state
change and the session counts it `dropped.invalid` (the runtime also drops NaN chunks before sending, §6).

### 1.3 Keepalive, end, release

All three carry the session fields; while the session runs, a message on its stream without `session_id` is
`bad_args`, one with another `session_id` or a lower `generation` / `control_epoch` is `stale_session` (§4).

| Message | Effect |
|---|---|
| `{..., keepalive: true}` | Refreshes the session watchdog only. Playback and stall accounting continue |
| `{..., end: true, hold_on_end: "target"\|"measured"\|"stand", reason}` | Ends the session now (terminal event `succeeded`, `ended_by: "client"`, `hold`: the mode applied, `reason` echoed). §5 says what each mode holds |
| `{..., release: true, blend_s}` | While the session runs: the same as `end` with `hold_on_end: "stand"`. After it ended into a `target` / `measured` hold on this stream: min-jerk blend to SONIC's reference over `blend_s` (1.5 s), then the override is dropped. A release that carries a `session_id` must be the session that left the hold (another one, or a lower `generation` / `control_epoch` → `stale_session`); one without session fields is accepted from the same `stream` (v0.5); `stop {arms: true}` releases any holder |

A new stream may also take over a hold without `preempt` (v0.5 rule 1), continuing from the pose being sent: that is
how a scripted place (`sonic_arm_script`, B.7) takes the carried object over.

### 1.4 Replies

| `state` | `error` | When |
|---|---|---|
| `accepted` | – | Session opened (first message) |
| `done` | – | Update applied, or a chunk counted as dropped (`data.dropped`: `expired` \| `out_of_order`; the session continues) |
| `rejected` | `bad_args` / `bad_chunk` / `mode_mismatch` | Malformed (missing session fields are `bad_args`); no state change |
| `rejected` | `arm_busy` | Another stream is streaming and `preempt` is not set (v0.5) |
| `rejected` | `arm_preempted` / `not_owner` | This stream lost the channel, or has no session (v0.5) |
| `rejected` | `stale_session` | `session_id`, `generation` or `control_epoch` older than the session that owns this stream (`data.owner_session`), a message of a session that already ended (`data: {ended: true, ended_by}`; the body keeps the last 500 ended ids), or a release/end of a hold that another session left (`data.hold_session`). Such a message never touches playback, so a chunk in flight when the runtime cancelled can never restart the arms |
| `rejected` | `halted` | The halt latch is set and the message has no newer `control_epoch` (§4) |
| `rejected` | `stale_command` | The fences (`data.why`: `control_epoch` \| `resume_epoch` \| `generation`), or this message's `t_wall` is older than the watchdog (`data.why: "t_wall"`, `age_s`, `watchdog_s`) |
| `rejected` | `body_busy` | A B.2 lease is held by another execution |
| `rejected` | `not_standing` / `fault:<kind>` | As v0.5 (on a session start) |

`data.session` in every reply to a chunk message: `{session_id, chunk_seq (the playing chunk), k, T, stall_s}`.

---

## 2. Playback, every 50 Hz tick

```
x      = (now + lead_s − t0_mono) / dt                  # latency compensation: rows in the past are skipped
row    = linear interpolation of rows ⌊x⌋ and ⌊x⌋+1     # row 0 before the chunk, the last row after it
k      = round(x) clamped to [0, T − 1]                 # reported
stall_s = max(0, now + lead_s − (t0_mono + (T − 1)·dt))  # > 0 once the chunk has run out: its last row is held
if a new chunk arrived i < xfade_ticks ticks ago:
    row = (1 − (i+1)/xfade_ticks) · old(now) + ((i+1)/xfade_ticks) · row    # old = the previous chunk (its last row
                                                                            # once it ran out) or the start pose
waist  = SONIC's reference waist unless the chunk says waist: "cmd"
target = clamp(row, URDF limits − 0.02 rad)             # joint_map.clamp_mj17 / clamp_hand; counts clamped values
sent   = slew_limit(target + servo correction, max_vel · tick)   # arm joints only; counts slew-limited arm values
mux.set_upper(wire_from_mj17(sent), velocity = 0, both hands)    # every planner message, whatever the legs do
```

- A chunk that is entirely in the past on arrival (`round(x) ≥ T`) is dropped `expired` and the old one keeps
  playing.
- The servo (the integral loop on the measured arm joints, m1.md §3.9) runs in chunk mode too, against SONIC's
  nominal response to the played rows (`servo_model: gated`, the default).
- `clamped_frac` = clamped values / (ticks × 28) over the progress window: the 14 arm values and 14 hand values of
  every tick (the waist is the channel's, not the client's, and is not counted); `clamped_frac_total` since the
  session start. The same count as `groot.actions.clamp_stats`, so the runtime's own per-chunk estimate and the
  body's number agree. `slew_frac` = slew-limited **arm** values / (ticks × 28) (the body never slew-limits the
  hands: the deploy limits each hand write to ±0.25 rad from the measured position).
- The legs are not touched: a motion op still owns mode, movement and facing. `manipulate` never runs with `navigate`
  (the runtime's rule C7), so in practice the legs hold IDLE.

---

## 3. Events: `arm.progress` and the terminal event (PUB 5611, `body.event` envelope)

`arm.progress` is `{op: "arm", state: "progress", id: <op id>, data: {...}}` at **5 Hz** while a chunk session
runs (v0.5 target sessions keep their 1 Hz progress):

| `data` field | Meaning |
|---|---|
| `kind` | `"chunk"` |
| `session_id`, `stream` | Echo |
| `chunk_seq` | `seq` of the chunk being played (0 before the first) |
| `k`, `T` | Row index and length of that chunk |
| `stall_s` | §2; 0 while rows remain |
| `clamped_frac`, `clamped_frac_total`, `slew_frac` | §2 |
| `max_step_rad` | The largest change of a sent arm value in one tick, since the session start |
| `latency_ms` | Age of the newest chunk on arrival: `(t_arrival − t0_mono)·1000` |
| `inference_ms` | Echo of the newest chunk's value |
| `lead_s`, `cross_fades` | Settings and count |
| `chunks` | `{received, applied, dropped: {expired, out_of_order, invalid, stale_session, halted}}` |

Terminal event (exactly one per session): `succeeded` (ended by the client), `failed` (the watchdog: reason
`client_silent`; a fall: `ended_by: "fault"`) or `canceled` (preempted, taken over, `stop {arms: true}`, halt).
`data`: `stream, kind: "chunk", ended_by` (`client` \| `watchdog` \| `preempted` \| `taken_over` \| `stop` \| `halt`
\| `fault`), `hold` (`target` \| `measured` \| `stand` \| `none`), `reason` (the client's, when given; else the
body's: `client_silent`, `halt`, `stop`, ...), `session_id, generation, control_epoch, execution_id, duration_s,
messages, watchdog_trips, clamped, slew_limited_ticks, max_step_rad`, the `chunks` counters, `clamped_frac_total,
slew_frac_total, stall_s_max, cross_fades, lead_s, chunk_seq`; a preempt / take-over adds `by`.

`body.state.arm` lists `modes: ["target", "chunk"]`. `groot_arms` reads it: a live body that does not list `"chunk"`
makes the executor unhealthy ("body arm op has no chunk mode (B.8)").

---

## 4. Fences, ownership, halt

- **Ownership** is v0.5's, keyed by `stream` (m1.md §3.9 rules 1-4: `arm_busy`, `preempt: true`, `arm_preempted`,
  take-over of a hold without preempt, legs never owned).
- **Session fence.** The session that owns a stream stores `(session_id, generation, control_epoch)`. **Every**
  message that acts on that stream (a chunk, a keepalive, an `end`, a `release`; with or without `mode: "chunk"`)
  must carry `session_id` (else `bad_args`); another `session_id`, or a lower `generation` or `control_epoch`, is
  `stale_session` and never touches playback. A message of a session that has ended is `stale_session` too (a new
  session on the same stream needs a new `session_id`). A hold that a session left keeps its fence: its release must
  come from that session.
- **`stop {arms: true}`** ends a chunk session at once (`canceled`, `ended_by: "stop"`, hold `stand`: blend back);
  its later messages are `stale_session`. `groot_arms` does not send it; it maps the terminal to `failed(halted)`.
- **Halt (B.1, m1.md §3.10).** The body's halt lane latches at once, on its own thread, without waiting for the arm
  handler or the tick: the wire freezes at the pose being sent (no chunk row after the ack) and every later `arm`
  message is `halted`. The next tick (≤ 20 ms) ends the session `canceled`, `ended_by: "halt"`, `hold: "measured"`,
  and holds: the **measured** arm pose (the servo is preloaded, so the wire does not step), the **waist as the
  session sent it** (SONIC's reference for the default `waist: "ref"`: no waist step), and the hands' **last
  target** (never the measured q, never opened: an object in the hand stays squeezed however often the robot is
  halted). Until `resume`, every message without a newer `control_epoch` is rejected `halted` (one with a newer epoch
  resumes implicitly, m1.md §3.10); afterwards a message with `control_epoch ≤ halt_epoch` is `stale_command`
  (`why: control_epoch`, the body's fences, m1.md §3.11) for good. A new session with a newer `control_epoch` takes
  the held pose over; `end` / `release` / `stop {arms}` blends it back.
  The comparison needs one epoch space: the halt's epoch must be the executions' `control_epoch` (R.2, runtime side).
- **Faults.** A fall drops the override at once; the session ends `failed`, `ended_by: "fault"`.

---

## 5. `hold_on_end`

| Mode | What the channel keeps sending after the session ends | Used by `groot_arms` for |
|---|---|---|
| `target` | The last played row (arms and hands, the waist as it was sent) on every planner message, walking included, with the servo on, until `release`, a take-over, `stop {arms: true}` or a fault. With a hand closed ≥ 0.3 this **is CarryLock** (B.7): `body.state.arm.carry.engaged` with the live palm error. A halt keeps it exactly as it is (m1.md §3.9) | Success: the object is in hand (GT). `groot_arms` labels the carry `arm_op_hold (INTERIM; CarryLock is body B.7)`; B.7 is built, so that label can now read CarryLock (runtime owner) |
| `measured` | The measured arm pose at the end tick (servo preloaded), the waist as the session sent it, and the hands' last target. Released like `target` | Failures, halt, policy down: the robot stops where it is and nothing in the hands is dropped (PLAN §5.6 "then HOLD") |
| `stand` | Min-jerk blend to SONIC's reference over `blend_s` (1.5 s), hands to the deploy's default fist, then the override is dropped | Cancel with nothing in hand |

---

## 6. What the runtime side guarantees (`services/executors/groot_arms.py`, owner ops-groot)

- One session per `manipulate` execution; `stream = session_id = execution_id`; `generation` and `control_epoch` are
  the execution's; every message (chunks, keepalives, `end`) carries them (`Session.base()`).
- Observation freshness: frame at most 150 ms old, `g1_debug` at most 60 ms old; otherwise no request (counted
  `obs_stale`). `t0_mono` = the receive time of that `g1_debug`.
- A new inference when none is in flight and 0.4 s have passed since the last chunk's `t0_mono` (2.5 Hz) or fewer than
  12 rows of it remain.
- Before sending: a result is dropped `stale_session` if the session was fenced while the request was in flight,
  `expired` if `now + lead_s − t0_mono > T·dt − 0.1 s`, `invalid` if a value is not finite or a shape is wrong.
  Out-of-limit values are sent; P3 clamps them and `clamped_frac > 0.2` for 1 s ends the skill
  `failed(policy_out_of_bounds)`.
- A `keepalive` whenever nothing was sent for 0.5 s, from the executor's event loop as well as from the client
  thread, so the body's 2.0 s watchdog only fires when P5 is gone.
- **No chunk after the cancel ack.** The fence check and the send happen under one lock; cancel sets the fence and
  then waits for any send in flight, and only then acknowledges. The tests count every chunk SENT after the ack,
  whatever the body answered (wave-1 verifier item, fixed by ops-groot in wave 2).
- Body refusals: `body_verdict()` ends the session on the fatal ones (`halted`, `stale_session`, `arm_preempted`,
  `body_busy`, `stale_command` with a fence `why`, `fault:*`, three refusals in a row); a `stale_command` with
  `why: "t_wall"` is only counted. A session the body ended on its own maps its `ended_by` (watchdog →
  `policy_stall`, halt / stop → `halted`, fault → `fell`, preempted / taken_over → `body_busy`).
- Ends: `target` on success, `stand` on a cancel with nothing in hand (`target` if GT says it is in hand),
  `measured` on every failure and on halt; after `end` it waits for the body's terminal event and reports the
  body's own counters (`data.body`).

---

## 7. Tests and open items

- Body: `body/tests/test_arm_channel.py` (tick by tick on the SONIC-like plant: the contract sequence of §1-§3, lead,
  cross-fade, stall, clamp, preempt / take-over / CarryLock, fault, the halt latch: the wire does not step, the hands
  keep their target under a lagging hand, the waist does not step, the session fence on end / release / keepalive,
  `why: "t_wall"`), `body/tests/test_halt.py` (over ZMQ with the real service).
- Runtime: `tests/services/test_groot_arms_body_arm.py` (the real executor against the real channel: a full
  session, a body halt latch, a cancel), `test_groot_arms_body_replies.py` (the refusal table), `test_groot_arms.py`
  (against `tests/fakes/fake_arm_body.py`).
- Live: synthetic chunk sessions (`tools/arm_wave_test.py chunk`, `tools/halt_test.py --chunk`, `docs/arm_tracking.md`
  §8.3, §9) and real GR00T chunks (`tools/groot_live_smoke.py --g2 --arm body`, ops-groot).
- [u] `lead_s` 0 vs 0.15 with real GR00T chunks (the G2 A/B).
- [u] The 5-tick cross-fade and the 6 rad/s slew limit against GR00T's chunk-to-chunk jumps (the dev-box smoke saw a
  max arm step of 0.127 rad per 20 ms in 1 of 30 chunks, `docs/M2b_wave1.md` §3): G2 measures `slew_frac`.
- `tests/fakes/fake_arm_body.py` still implements v0.1 (lead 0 by default, hands held at their last target on a halt,
  a watchdog end `succeeded`); the executor's offline tests do not depend on the differences.

---

## 8. Changes from v0.1

| # | v0.1 | v0.2 (as built) | Since |
|---|---|---|---|
| 1 | `lead_s` default 0.0 | **0.15** (G0 condition 2); an explicit value wins | body wave |
| 2 | watchdog end `succeeded` | **`failed`**, reason `client_silent`, `ended_by: "watchdog"`; `body.fault{policy_lost}` event | body wave |
| 3 | `k = round(...)`, one row per tick | rows **interpolated** at the continuous index; `k` is reported rounded | body wave |
| 4 | `slew_frac` over every value | slew-limited **arm** values over ticks × 28 (hands are never slew-limited) | body wave |
| 5 | `stop {arms}` not specified | ends a chunk session `canceled`, `ended_by: "stop"`, blend back; later messages `stale_session` | body wave |
| 6 | halt: hands "keep their last target" | the body wave held the **measured** hands (repeated halts ratcheted a grip open, verifier B-D2); wave 2 is back to the **last target**, and the waist stays as the session sent it (the measured waist stepped it 0.12-0.14 rad) | wave 2 |
| 7 | missing fence fields `bad_chunk` | `bad_args` | body wave |
| 8 | ended sessions `stale_session` | the same, kept in a bounded map (500), `data: {ended: true, ended_by}` | body wave |
| 9 | the session fence on chunk messages | on **every** message acting on the session's stream (`keepalive`, `end`, `release`, with or without `mode`), and on the release of a hold the session left (verifier B-low: `release: true` and a non-chunk `end` skipped it) | wave 2 |
| 10 | `stale_command` for a late message | the same, with `data.why: "t_wall"` (+ `age_s`, `watchdog_s`), told apart from the fences' `stale_command` (`why: control_epoch \| resume_epoch \| generation`) | wave 2 |
| 11 | halt handled "at once" | the halt lane never takes the arm lock: the wire freezes on the lane thread and the next tick (≤ 20 ms) ends the session and sets up the hold (verifier B-D1: a halt waited 176-221 ms behind an `arm_script` IK) | wave 2 |
| 12 | terminal `data` | adds `kind, generation, control_epoch, execution_id, messages, watchdog_trips, clamped, slew_limited_ticks, max_step_rad, slew_frac_total, cross_fades, lead_s, chunk_seq`; `arm.progress` adds `max_step_rad` | body wave |
