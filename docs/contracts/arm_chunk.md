# Arm op, chunk mode (body B.8): the wire between `groot_arms` (P5) and wl-body (P3)

Status: **v0.1, 2026-09-29.** Written by the runtime owner of `groot_arms` (R.4) for the body owner (B.8, `docs/M2.md`
§7.2). The runtime side is built against it: `services/executors/groot_arms.py` sends exactly these messages, and
`tests/fakes/fake_arm_body.py` is an in-process implementation of this document that the offline tests run against.
**As built (integration, wave 1):** the body owner implemented chunk mode in `body/arm.py` on the body branch
(master `4b081c2`, not yet merged into this branch). §8 lists the differences found; `groot_arms` was run against
that ArmChannel on a trial merge (`tests/services/test_groot_arms_body_arm.py`, 3/3). No chunk has reached SONIC on a
box yet.

Sources: `docs/groot_arms_design.md` §5.2-§5.5 (what GR00T needs), the `arm` op v0.5 as it stands in the G0 working
tree (`body/arm.py`, `body/client.py`, `docs/contracts/m1.md` §3.9; read, not modified), `docs/contracts/m1.md`
§3.4 (envelopes, the `velocity` streaming pattern), PLAN §5.5-§5.6 (fences, halt).

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
   k = round((now + lead_s − t0_mono) / dt) → row k (cross-fade 5 ticks on a new chunk; hold the last row and  │
   count stall once k ≥ T) → clamp (URDF − 0.02 rad) → slew limit → SonicMux: planner IDLE + upper_body_position│
   (wire order, joint_map.wire_from_mj17) + both Dex3 hands → P2                                              │
   PUB 5611 body.event {op:"arm", state:"progress", data:{kind:"chunk", chunk_seq, k, stall_s, clamped_frac …}} ─┘
```

**Clock rule.** `t0_mono` is `time.monotonic()` (CLOCK_MONOTONIC) on the host that runs P5. P5 and P3 run on the
same host (the main box), so P3 compares it with its own `time.monotonic()`. The PolicyServer may be remote (OD3: dev
box over an SSH tunnel); that only adds latency, which the time index absorbs. If P5 and P3 ever run on different
hosts this contract does not hold (use `t0_wall` instead; not specified in v0.1).

---

## 1. Messages (op `arm`, DEALER → ROUTER 5610; envelope as `docs/contracts/m1.md` §3.4)

Every message of a chunk session carries these fields. The first one opens the session and creates the op record
(reply `accepted`, event `accepted`), exactly like the first message of a v0.5 stream. Later ones are stream updates:
reply `done`, no event.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `stream` | str | yes | Ownership key, as in v0.5. `groot_arms` uses `stream = session_id` |
| `mode` | `"chunk"` | yes | Chunk mode. v0.5 messages have no `mode` (= `"target"`). A session's mode is fixed: a message whose mode differs from its session's is rejected `mode_mismatch` |
| `session_id` | str | yes | The fence id. `groot_arms` uses the `manipulate` execution id |
| `generation` | int | yes | The execution's generation (PLAN §5.5) |
| `control_epoch` | int | yes | The execution's control epoch (PLAN §5.5); compared with the halt latch (§4) |
| `execution_id` | str | no | For B.2 leases; `groot_arms` sends it (= `session_id`) |
| `t_wall` | float | yes | Sender `time.time()`; v0.5 staleness rule: older than `watchdog_s` ⇒ `stale_command` |

### 1.1 Session start (first message)

| Field | Default | Meaning |
|---|---|---|
| `hold_on_end` | `"measured"` | What the channel does when the session ends without an explicit `hold_on_end` (watchdog, fault excepted): §5 |
| `watchdog_s` | 2.0 (0.5-5.0) | No message of this session for this long ⇒ the session ends `ended_by: "watchdog"` into `hold_on_end` (§5). 2.0 s, not the design's 1.0 s: a policy request may block 1.5 s (REQ timeout), and the runtime sends a `keepalive` after every inference that produced no chunk |
| `lead_s` | 0.0 (0-0.3) | Play rows this far ahead of time. `docs/arm_tracking.md` measured about 0.15 s of SONIC arm lag; a lead is the G2 A/B, off by default |
| `xfade_ticks` | 5 | Cross-fade length on a new chunk (100 ms) |
| `max_vel` | the channel's `arm_max_vel` | Slew limit (rad/s), as v0.5 |
| `left_hand`, `right_hand` | – | Optional start targets (7 Dex3 values each), blended in over `hands_blend_s`. `groot_arms` sends open hands (all 0) here: the Arena demos start with open hands |
| `hands_blend_s` | 0.3 | Blend time for the start hands |
| `preempt` | false | v0.5 semantics (§4) |
| `chunk` | – | Optional: the first chunk may ride on the start message |

Until the first chunk plays, the channel sends SONIC's own reference upper body (`g1_debug.body_q_target`), the
same continuity rule as a v0.5 session start, plus the start hands. The first chunk cross-fades in from there.

### 1.2 Chunk

`chunk` is an object:

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | Monotonic per session (1, 2, ...). A chunk whose `seq` is not above the last applied one is dropped `out_of_order` |
| `t0_mono` | float | CLOCK_MONOTONIC time that row 0 belongs to: the receive time of the `g1_debug` sample the observation was built from (design §5.2 step 1). Also dropped `out_of_order` when it is older than the applied chunk's `t0_mono` |
| `dt` | float | Row period. 0.02 (50 Hz, the N1.7 action horizon's rate). Accepted range 0.005-0.1 |
| `order` | `"wire"` \| `"mj17"` | Joint order of `upper_body`. **`"wire"`** = SONIC's IsaacLab-interleaved order (`joint_map.UPPER_BODY_JOINTS`: waist yaw/roll/pitch, then left/right pairs joint by joint), what `groot.actions.to_arm_chunk` produces. `"mj17"` = the v0.5 API order (`joint_map.UPPER_BODY_MUJOCO_JOINTS` = `g1_debug.body_q[12:29]`). The field is required: there is no default, so an order mix-up is a rejection, not a silent swap of left and right arm values. P3 converts `"wire"` to its internal mj17 with `joint_map.mj17_from_wire` and sends with `wire_from_mj17`, as today |
| `upper_body` | T×17 float | Absolute joint targets (rad), in `order` |
| `left_hand`, `right_hand` | T×7 float | Absolute Dex3 targets in **Dex3 order** (`thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1`), both always present (the deploy reuses a stale value for a missing hand, `input_interface.hpp:341-362`). GR00T's `index…thumb` order is converted by name in `groot/`, never in P3 |
| `waist` | `"ref"` \| `"cmd"` | Optional, default `"ref"`: the waist entries are replaced by SONIC's reference waist (v0.5 `waist` rule). GR00T's waist output is identically 0 in the Arena data and is never executed (design §2.5) |
| `inference_ms` | float | Optional telemetry (echoed in `arm.progress`) |

1 ≤ T ≤ 64, every row finite, shapes consistent. Otherwise the whole message is rejected `bad_chunk` with no state
change (the runtime also drops NaN chunks before sending, §6).

### 1.3 Keepalive, end, release

| Message | Effect |
|---|---|
| `{..., keepalive: true}` | Refreshes the session watchdog only. Playback and stall accounting continue |
| `{..., end: true, hold_on_end: "target"\|"measured"\|"stand", reason}` | Ends the session now (terminal event `succeeded`, `ended_by: "client"`, `hold`: the mode applied, `reason` echoed). §5 says what each mode holds |
| `{stream, release: true, blend_s}` | Ends a `target`/`measured` hold left by this stream: min-jerk blend to SONIC's reference over `blend_s` (1.5 s), then drop the override (v0.5 blend). `stop {arms: true}` does the same for any holder |

A new stream may also take over a hold without `preempt` (v0.5 rule 3), continuing from the pose being sent: that is
how a scripted place (`sonic_arm_script`, B.7) takes the carried object over.

### 1.4 Replies

| `state` | `error` | When |
|---|---|---|
| `accepted` | – | Session opened (first message) |
| `done` | – | Update applied, or a chunk counted as dropped (`data.dropped`: `expired` \| `out_of_order`; the session continues) |
| `rejected` | `bad_args` / `bad_chunk` / `mode_mismatch` | Malformed; no state change |
| `rejected` | `arm_busy` | Another stream is streaming and `preempt` is not set (v0.5) |
| `rejected` | `arm_preempted` / `not_owner` | This stream lost the channel, or has no session (v0.5) |
| `rejected` | `stale_session` | `session_id`, `generation` or `control_epoch` older than the session that owns this stream, or a message for a session that already ended (§4). Also published as `body.stale_command` once B.2 exists |
| `rejected` | `halted` | `control_epoch ≤ halt_epoch` after a halt (B.1, §4) |
| `rejected` | `not_standing` / `fault:<kind>` | As v0.5 |

`data.session` in every reply to a chunk message: `{session_id, chunk_seq (applied), k, T, stall_s}`.

---

## 2. Playback, every 50 Hz tick

```
k      = round((now + lead_s − t0_mono) / dt)          # latency compensation: rows in the past are skipped
row    = chunk[min(max(k, 0), T − 1)]
stall_s = max(0, now + lead_s − (t0_mono + (T − 1)·dt))  # > 0 once the chunk has run out: hold its last row
if a new chunk arrived i < xfade_ticks ticks ago:
    row = (1 − (i+1)/xfade_ticks) · old_chunk(now) + ((i+1)/xfade_ticks) · row     # old_chunk holds its last row too
waist  = SONIC's reference waist unless the chunk says waist: "cmd"
target = clamp(row, URDF limits − 0.02 rad)            # joint_map.clamp_mj17 / clamp_hand; counts clamped values
sent   = slew_limit(target, max_vel · tick)            # v0.5 limiter; counts limited values
mux.set_upper(wire_from_mj17(sent), velocity = 0, both hands)   # every planner message, IDLE legs unless a motion runs
```

- A chunk that is entirely in the past on arrival (`k ≥ T`) is dropped `expired` and the old one keeps playing.
- The v0.5 servo (the integral loop on the measured arm joints) stays available in chunk mode. It is compared against
  the played row `servo_delay_s` ago, as in v0.5.
- `clamped_frac` = clamped values / (ticks × 28) over the progress window: the 14 arm values and 14 hand values of
  every tick (the waist is the channel's, not the client's, and is not counted), and cumulative since the session
  start. This is the same count as `groot.actions.clamp_stats`, so the runtime's own per-chunk estimate and the
  body's number agree. `slew_frac` likewise.
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
| `latency_ms` | Age of the newest chunk on arrival: `(t_arrival − t0_mono)·1000` |
| `inference_ms` | Echo of the newest chunk's value |
| `lead_s`, `cross_fades` | Settings and count |
| `chunks` | `{received, applied, dropped: {expired, out_of_order, invalid, stale_session, halted}}` |

Terminal event (exactly one per session): `succeeded` (ended by the client, or by the watchdog), `canceled`
(preempted, taken over, `stop {arms: true}`, halt) or `failed` (fault). `data`: `ended_by` (`client` \| `watchdog`
\| `preempted` \| `taken_over` \| `stop` \| `halt` \| `fault`), `hold` (`target` \| `measured` \| `stand` \|
`none`), `reason` (the client's, when given), `session_id`, the `chunks` counters, `clamped_frac_total`,
`stall_s_max`, `duration_s`.

`body.state.arm` also lists `modes: ["target", "chunk"]` once B.8 is in. `groot_arms` reads it: a live body that does
not list `"chunk"` makes the executor unhealthy ("body arm op has no chunk mode (B.8)").

---

## 4. Fences, ownership, halt

- **Ownership** is v0.5's, keyed by `stream`: rules 1-4 of `docs/contracts/m1.md` §3.9 apply unchanged (`arm_busy`,
  `preempt: true`, `arm_preempted`, take-over after the owner stops streaming, legs never owned).
- **Session fence.** The session that owns a stream stores `(session_id, generation, control_epoch)`. A message on
  that stream with another `session_id`, or a lower `generation` or `control_epoch`, is rejected `stale_session` and
  never touches playback. A message for a session that has ended is rejected `stale_session` too, so a chunk that
  was in flight when the runtime cancelled can never restart the arms. (A new session on the same stream needs a
  new `session_id`.)
- **Halt (B.1).** On a halt at `epoch = n` the channel latches at once: the chunk timeline is dropped, the
  **measured** upper body (`g1_debug.body_q[12:29]`) is held, the hands keep their last target (**never opened**:
  an object in hand stays in hand), and the session ends `canceled`, `ended_by: "halt"`, `hold: "measured"`.
  Until `resume`, every message with `control_epoch ≤ n` is rejected `halted`. Before B.1 exists, the M1 halt is a
  body `stop` (legs only): the runtime then sends `end {hold_on_end: "measured", reason: "halted"}` itself.
  The comparison needs one epoch space: the halt's `n` must be the harness's control epoch (PLAN §5.5: a stop bumps
  it, `halt()` first). M2a's `HaltGate` counts its own epochs, so R.2 has to align them; until then the latch still
  ends the session, and later messages of it are refused `stale_session`.
- **Faults.** A fall drops the override at once, as in v0.5 (`abort`); the session ends `failed`, `ended_by:
  "fault"`.

---

## 5. `hold_on_end`

| Mode | What the channel keeps sending after the session ends | Used by `groot_arms` for |
|---|---|---|
| `target` | The last played row (arms and hands) on every planner message, including during later walks, until `release`, a take-over, `stop {arms: true}` or a fault. This is CarryLock's behaviour (B.7) on the arm op; B.7 may later move it into its own op | Success: the object is in hand (GT). **INTERIM** until CarryLock exists: `groot_arms` labels the carry `carry: "arm_op_hold (INTERIM; CarryLock is B.7)"` |
| `measured` | The measured upper body at the end tick, and the hands' last target. Released like `target` | Failures, halt, policy down: the robot stops where it is and nothing in the hands is dropped (PLAN §5.6 "then HOLD") |
| `stand` | Min-jerk blend to SONIC's reference over `blend_s` (1.5 s), then the override is dropped (v0.5 `end`) | Cancel with nothing in hand |

---

## 6. What the runtime side guarantees (`services/executors/groot_arms.py`)

- One session per `manipulate` execution; `stream = session_id = execution_id`; `generation` and `control_epoch` are
  the execution's.
- Observation freshness: frame at most 150 ms old, `g1_debug` at most 60 ms old; otherwise no request (counted
  `obs_stale`). `t0_mono` = the receive time of that `g1_debug`.
- A new inference when none is in flight and 0.4 s have passed since the last chunk's `t0_mono` (2.5 Hz) or fewer than
  12 rows of it remain.
- Before sending: a result is dropped `stale_session` if the session was fenced while the request was in flight,
  `expired` if `now − t0_mono > T·dt − 0.1 s`, `invalid` if a value is not finite or a shape is wrong. Out-of-limit
  values are sent; P3 clamps them and `clamped_frac > 0.2` for 1 s ends the skill `failed(policy_out_of_bounds)`.
- A `keepalive` whenever nothing was sent for 0.5 s, from the executor's event loop as well as from the client
  thread: a blocked `get_action` (1.5 s) plus the ping after it (0.5 s) must not outlast the 2.0 s watchdog, which
  should only fire when P5 is gone. (The offline F7 test caught exactly this before the loop sent keepalives.)
- **No chunk after the cancel ack.** The fence check and the send happen under one lock; cancel sets the fence and
  then waits for any send in flight, and only then acknowledges. The fake body records receive times, and the tests
  assert that no chunk arrives after the ack.
- Ends: `target` on success, `stand` on a cancel with nothing in hand (`target` if GT says it is in hand),
  `measured` on every failure and on halt.

---

## 7. Tests and open items

- `tests/fakes/fake_arm_body.py` implements §1-§5 in process (a 50 Hz playback thread, the same index, cross-fade,
  clamp, fences, events). Its clamp uses `groot.joint_order.URDF_LIMITS` minus 0.02 rad (the body's own table is
  `body/joint_map.py`, which is in the G0 working tree, not in this branch). The body's B.8 tests should replay the
  same sequences against the real channel: the session start, 3 chunks with `lead_s` 0, a late chunk (`expired`),
  an out-of-order `seq`, a stale `session_id`, a NaN chunk, cancel, halt.
- [u] `lead_s`: 0 or about 0.15 s (G2 A/B, together with the v0.5 servo).
- [u] The 5-tick cross-fade and the 6 rad/s slew limit against GR00T's chunk-to-chunk jumps (G2 measures
  `slew_frac`).
- [u] Whether `hold_on_end: "target"` stays on the arm op or moves to a separate CarryLock op in B.7. If it moves,
  `groot_arms` changes one call; the wire above stays.

---

## 8. As built by the body (master `4b081c2` `body/arm.py`), reconciled in wave 1

Read in `body/arm.py` on master (read-only) and exercised by `tests/services/test_groot_arms_body_arm.py`, which runs
the real `GrootArmExecutor` (real `groot/` helpers, the fake PolicyServer) against the real `ArmChannel` ticking at
50 Hz in wall time over the body's own test plant (`body/tests/arm_sim.py`). The test skips on this branch (no
`body/arm.py`) and runs after the wave-2 merge. On a trial merge (this branch + master's `body/`, `tools/`): 3 passed
(a full session, a body halt latch, a cancel).

| Item | Contract v0.1 | Body as built | Runtime now |
|---|---|---|---|
| `lead_s` default | 0.0 | `arm_lead_s` 0.15 when the start omits it (G0: SONIC's ~0.15 s arm lag, `docs/arm_tracking.md` §0) | `groot_arms` sends `lead_s: 0.15` (`GrootArmsConfig.lead_s`, `config/profiles/full.yaml`); its own expiry check adds `lead_s`. 0 vs 0.15 stays the G2 A/B |
| Chunk session watchdog | ends into `hold_on_end`, `ended_by: "watchdog"` | the same; terminal `failed`, reason `client_silent`; B.3 publishes `body.fault{policy_lost}` (an event, not FAULT) | `watchdog` → `failed(policy_stall)` (unchanged) |
| Halt latch | messages with `control_epoch <= halt_epoch` → `halted` | the same, plus: every `arm` message while latched → `halted`; the active session ends `canceled`, `ended_by: "halt"`, hold measured | `halted` / `ended_by: halt` → `failed(halted)` (unchanged). The runtime's HaltGate epochs are still not the executions' `control_epoch` (R.2) |
| Required fields | `session_id`, `generation`, `control_epoch` | missing ones → `bad_args` (not `bad_chunk`) | always sent |
| Ended session ids | `stale_session` | kept in a bounded map, `stale_session {ended: true, ended_by}` | `stale_session` → `failed(controller_unavailable)` |
| `stop {arms: true}` | not specified | ends the session `canceled` (reason `stop`); the stream is then refused `arm_stopped` until `restart: true` | not sent by `groot_arms`; a refused `arm_stopped` ends the session `failed(halted)` (`BODY_FATAL`) |
| Progress | `arm.progress` 5 Hz | 5 Hz `progress` with `kind: "chunk"`, `clamped_frac`, `clamped_frac_total`, `slew_frac`, `lead_s`, `max_step_rad`, `chunks{...}` | read as specified |

Open for wave 2: replay the fake body's sequences against the real channel in `body/tests` (the body owner's half of
§7); measure `slew_frac` on real GR00T chunks (the dev-box smoke saw a max arm step of 0.127 rad per 20 ms in 1 of 30
chunks, `docs/M2b_wave1.md` §3).
