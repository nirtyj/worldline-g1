"""Live tests of the body wave's arm side (owner body-arm) on the M1 stack: B.8 chunk mode with cancel and halt, B.7
arm_script + CarryLock (a scripted pick toward a real object, then a 2 m carry walk), B.5 waist scan.

    # box, M1 stack up, body running this code (scripts/m1_restart_body.sh)
    .venv/bin/python -m tools.arm_wave_test chunk --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-chunk
    .venv/bin/python -m tools.arm_wave_test pick  --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-pick
    .venv/bin/python -m tools.arm_wave_test scan  --out outputs/body_wave/$(date +%Y%m%d-%H%M%S)-scan
    python -m tools.arm_wave_test chunk --fake --sessions 3 --cancels 1 --halts 1 --out /tmp/aw   # plumbing (fakes)

chunk  Emulates groot_arms (services/executors/groot_arms.py, docs/contracts/arm_chunk.md): per session the start
       message (open hands, hold_on_end measured, lead_s 0.15), then every 0.4 s an "inference": the observation time
       t0 = receive time of the newest g1_debug, 150 ms of simulated inference, then a 40-row chunk (50 Hz, SONIC wire
       order) of a smooth synthetic reach (IK keyframes: out, sway, close the hand, back) with a per-chunk random offset
       (sigma 0.015 rad) standing in for GR00T's chunk-to-chunk disagreement. Sessions end `stand` or `target` (the
       next one takes the hold over); `--cancels` sessions are cancelled mid-motion (end hold_on_end measured) and
       `--halts` are halted on the body's halt lane (PUSH 5612, then resume and release). After every cancel/halt ack
       three in-flight chunks with a +0.4 rad "poison" on the moving elbow are sent: the body must reject them and the
       wire must never show them. Measured: falls (GT), the wire (a SUB on SONIC's input 5556: every planner message's
       upper_body_position), max joint step per planner message and per body tick (the body's max_step_rad), around
       chunk boundaries and elsewhere, g1_debug steps, clamped / slew fractions, ack times.
pick   Scene objects from P1: the object (default the bedroom dresser's remote in procthor-train-38) and its support;
       a stance `--gap` in front of the support edge with the object ahead of the right shoulder; go_to; then
       `--trials` x (pregrasp, grasp, retract-to-pregrasp), each grasp's palm error to the grasp point (the object's
       top centre + `--grasp-above`) measured by the tool itself (FK of g1_debug body_q on the GT pelvis pose, 50 Hz)
       and by the body; then pregrasp, grasp, lift, carry (CarryLock) and a 2 m walk (turn 180 deg, walk 0.3 m/s):
       palm drift against the held pose, falls; then release and retract.
scan   The waist scan standing in open space (where the robot is) and in front of the support (`--at-counter`):
       achieved yaw per hold, arm joint deviation ("only the waist moves"), base drift, falls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import traceback

import numpy as np
import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body import g1_kin as K  # noqa: E402
from body import joint_map as jm  # noqa: E402
from body.arm_script import pelvis_to_world, world_to_pelvis  # noqa: E402
from body.client import BodyClient  # noqa: E402
from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402
from body.wire import decode_planner, loads_any, split_topic, wrap  # noqa: E402

MJ17 = jm.UPPER_BODY_MUJOCO_JOINTS
ARM_K = list(range(3, 17))
DT = 0.02


def _r(v, n=4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, n) for x in v]
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


def _pct(a, q):
    return None if len(a) == 0 else round(float(np.percentile(np.asarray(a, float), q)), 4)


def _minjerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


# ================================================================================================= monitors
class Monitor:
    """SUBs on the wire (SONIC input 5556, planner topic), g1_debug (5557) and gt.pose (5601); every sample is stamped
    with this process's time.monotonic()."""

    def __init__(self, P: dict, ctx: zmq.Context):
        self.P, self.ctx = P, ctx
        self.lock = threading.Lock()
        self.wire: list = []           # (t, mode, mj17 | None, left | None, right | None)
        self.dbg: list = []            # (t, q29, lh, rh)
        self.gt: list = []             # (t, x, y, z, yaw, pelvis_z, fallen, quat)
        self.dbg_latest = None
        self.dbg_t = None
        self.running = True
        self.th = [threading.Thread(target=f, daemon=True) for f in (self._wire, self._dbg, self._gt)]
        for t in self.th:
            t.start()

    def _sub(self, port, topic):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 5000)
        s.setsockopt(zmq.SUBSCRIBE, topic)
        s.connect(ep(port))
        return s

    def _wire(self):
        s = self._sub(self.P["sonic_in"], b"planner")
        while self.running:
            if not s.poll(100):
                continue
            raw = s.recv()
            t = time.monotonic()
            try:
                p = decode_planner(raw)
            except Exception:
                continue
            up = p.get("upper_body_position")
            with self.lock:
                self.wire.append((t, int(p["mode"]), None if up is None else jm.mj17_from_wire(up),
                                  p.get("left_hand_joints"), p.get("right_hand_joints")))
        s.close(0)

    def _dbg(self):
        import msgpack
        s = self._sub(self.P["sonic_debug"], b"g1_debug")
        while self.running:
            if not s.poll(100):
                continue
            raw = s.recv()
            t = time.monotonic()
            try:
                d = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False, strict_map_key=False)
            except Exception:
                continue
            if d.get("body_q") is None or len(d["body_q"]) != 29:
                continue
            with self.lock:
                self.dbg.append((t, list(d["body_q"]), d.get("left_hand_q"), d.get("right_hand_q")))
                self.dbg_latest, self.dbg_t = d, t
        s.close(0)

    def _gt(self):
        s = self._sub(self.P["p1_pose"], b"gt.pose")
        while self.running:
            if not s.poll(100):
                continue
            frames = s.recv_multipart()
            t = time.monotonic()
            pl = split_topic(frames, b"gt.pose")
            if pl is None:
                continue
            try:
                d = loads_any(pl)
            except Exception:
                continue
            pos = d.get("base_pos") or [0, 0, 0]
            with self.lock:
                self.gt.append((t, float(pos[0]), float(pos[1]), float(pos[2]), float(d.get("yaw") or 0.0),
                                float(d.get("pelvis_z") if d.get("pelvis_z") is not None else pos[2]),
                                bool(d.get("fallen", False)), list(d.get("base_quat_wxyz") or [1, 0, 0, 0])))
        s.close(0)

    def falls(self, t0=None, t1=None) -> int:
        with self.lock:
            g = [x for x in self.gt if (t0 is None or x[0] >= t0) and (t1 is None or x[0] <= t1)]
        n, prev = 0, False
        for x in g:
            f = x[6] or x[5] < 0.55
            n += f and not prev
            prev = f
        return n

    def pose_at(self, t):
        with self.lock:
            g = list(self.gt)
        best = min(g, key=lambda x: abs(x[0] - t)) if g else None
        return best

    def last_gt(self):
        with self.lock:
            return self.gt[-1] if self.gt else None

    def dbg_between(self, t0, t1):
        with self.lock:
            return [x for x in self.dbg if t0 <= x[0] <= t1]

    def wire_between(self, t0, t1):
        with self.lock:
            return [x for x in self.wire if t0 <= x[0] <= t1]

    def stop(self):
        self.running = False
        for t in self.th:
            t.join(1.0)


class Pose_:
    """body.wire.Pose look-alike for the frame helpers (x, y, z, quat)."""

    def __init__(self, g):
        self.x, self.y, self.z, self.yaw, self.quat = g[1], g[2], g[3], g[4], g[7]


def palm_world(q29, g, side):
    return pelvis_to_world(K.points(K.named_from_q29(q29))[f"{side}_palm"], Pose_(g))


# ================================================================================================= runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.off = port_offset_from_env() if a.port_offset is None else a.port_offset
        self.P = _ports(self.off)
        self.ctx = zmq.Context.instance()
        self.out = a.out
        os.makedirs(self.out, exist_ok=True)
        self.events: list = []
        self.notes: dict = {"errors": []}
        self.fake = None
        if a.fake:
            self._start_fakes()
        self.mon = Monitor(self.P, self.ctx)
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0, ctx=self.ctx)
        self.bc = BodyClient(port_offset=self.off, ctx=self.ctx).connect(20)
        self.bc.add_listener(lambda ev: self.events.append({"t": time.monotonic(), **ev}))
        t0 = time.monotonic()
        while (self.mon.dbg_t is None or self.mon.last_gt() is None) and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        st = self.bc.status()
        self.notes["status_start"] = {k: st.get(k) for k in ("fault", "in_control", "mode", "arm", "halt", "fences")}
        self.notes["p1_stats_start"] = self.p1.try_call("get_stats")
        self.notes["load_start"] = os.getloadavg()
        if st.get("fault"):
            raise RuntimeError(f"body fault {st['fault']}: reset the robot first")
        if not st.get("in_control"):
            h = self.bc.stand(timeout=120)
            if not h.ok:
                raise RuntimeError(f"stand failed: {h.result}")
        self.epoch = self._epoch(st) + 1
        self.generation = max(1, int((st.get("fences") or {}).get("generation_floor") or 1))
        self.t_start = time.monotonic()

    def _start_fakes(self):
        from body.config import BodyConfig
        from body.service import BodyService
        from tools.fake_deploy import FakeDeploy
        from tools.fake_p1 import FakeP1

        tmp = os.path.join(self.out, "fake")
        self.fake = {"p1": FakeP1(self.off, os.path.join(tmp, "p1"), log=lambda *_: None).start(),
                     "dep": FakeDeploy(self.off, log=lambda *_: None).start()}
        svc = BodyService(BodyConfig(port_offset=self.off), log_dir=os.path.join(tmp, "body"), log=None)
        th = threading.Thread(target=svc.run, daemon=True)
        th.start()
        self.fake["svc"] = svc

    @staticmethod
    def _epoch(st: dict) -> int:
        cands = [0]
        for path in (("fences", "halt_epoch"), ("fences", "epoch_seen"), ("fences", "resume_epoch"), ("halt", "last", "epoch"),
                     ("arm", "halt_epoch")):
            v = st
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, int):
                cands.append(v)
        return max(cands)

    def ev_terminal(self, op_id):
        for e in self.events:
            if e.get("id") == op_id and e.get("state") in ("succeeded", "failed", "canceled"):
                return e
        return None

    def wait_terminal(self, op_id, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            e = self.ev_terminal(op_id)
            if e is not None:
                return e
            time.sleep(0.02)
        return None

    def arm(self, args, op_id=None):
        return self.bc.request("arm", args, op_id=op_id)

    def ref_mj17(self):
        d = self.mon.dbg_latest or {}
        v = d.get("body_q_target") or d.get("body_q")
        return jm.mj17_from_mujoco(v)

    def q_mj17(self):
        return jm.mj17_from_mujoco(self.mon.dbg_latest["body_q"])

    def save(self, name, obj):
        with open(os.path.join(self.out, name), "w") as f:
            json.dump(obj, f, indent=1, default=_jd)

    def close(self):
        self.notes["p1_stats_end"] = self.p1.try_call("get_stats")
        self.notes["load_end"] = os.getloadavg()
        with self.mon.lock:
            np.savez_compressed(os.path.join(self.out, "raw.npz"),
                                wire_t=np.array([w[0] for w in self.mon.wire]),
                                wire_mode=np.array([w[1] for w in self.mon.wire]),
                                wire_mj17=np.array([w[2] if w[2] is not None else [np.nan] * 17 for w in self.mon.wire]),
                                dbg_t=np.array([d[0] for d in self.mon.dbg]),
                                dbg_q=np.array([d[1] for d in self.mon.dbg]),
                                gt=np.array([g[:7] for g in self.mon.gt], dtype=float))
        self.save("events.json", self.events)
        self.save("notes.json", self.notes)
        self.mon.stop()
        self.bc.close()


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ================================================================================================= chunk sessions
class Reach:
    """A smooth synthetic reach (absolute mj17 targets as a function of the time since the session start), from the
    measured arm pose at the start: out to an
    IK keyframe over 1.5 s, a Hann-windowed 0.5 Hz sway (shoulder pitch 0.15 rad, elbow 0.10 rad) with the hand
    closing to 0.6,
    back to SONIC's reference arms from 4.0 s, hand open again."""

    def __init__(self, ref17, side: str, i: int):
        self.ref, self.side = list(ref17), side
        seed = K.named_from_mj17(ref17)
        sgn = -1.0 if side == "right" else 1.0
        q, err = K.ik_palm(side, (0.26, 0.20 * sgn, 0.02 + 0.03 * (i % 3)), seed, q_rest=seed,
                           lock=(f"{side}_wrist_pitch_joint", f"{side}_wrist_yaw_joint"))
        self.goal = list(ref17)
        for n in K.ARM_CHAIN[side]:
            self.goal[MJ17.index(n)] = q[n]
        self.ik_err = err
        self.k_sp = MJ17.index(f"{side}_shoulder_pitch_joint")
        self.k_el = MJ17.index(f"{side}_elbow_joint")
        self.duration = 5.5

    def at(self, t):
        if t < 1.5:
            a = _minjerk(t / 1.5)
            q = [r + (g - r) * a for r, g in zip(self.ref, self.goal)]
        elif t < 4.0:
            q = list(self.goal)
            tau = t - 1.5
            s = math.sin(2 * math.pi * 0.5 * tau) * math.sin(math.pi * tau / 2.5) ** 2   # Hann-windowed: smooth ends
            q[self.k_sp] += 0.15 * s
            q[self.k_el] += 0.10 * s
        else:
            a = _minjerk((t - 4.0) / 1.5)
            q = [g + (r - g) * a for r, g in zip(self.ref, self.goal)]
        c = 0.6 * _minjerk((t - 2.0) / 0.8) * (1.0 - _minjerk((t - 4.2) / 0.5))
        hands = {"left": [0.0] * 7, "right": [0.0] * 7}
        hands[self.side] = jm.hand_closure(self.side, c)
        return q, hands


def chunk_test(R: Runner) -> dict:
    a = R.a
    rng = random.Random(a.seed)
    n_s = a.sessions
    kinds = ["normal"] * n_s
    idx = list(range(1, n_s))
    rng.shuffle(idx)
    for j in idx[:a.cancels]:
        kinds[j] = "cancel"
    for j in idx[a.cancels:a.cancels + a.halts]:
        kinds[j] = "halt"
    results = []
    hold_next = False
    for i in range(n_s):
        sid = f"barm-{int(time.time())}-{i}"
        side = "right" if i % 2 == 0 else "left"
        kind = kinds[i]
        end_hold = "target" if (kind == "normal" and i % 3 == 1 and i < n_s - 1) else "stand"
        print(f"[arm_wave] chunk session {i} {kind} {side} end={end_hold if kind == 'normal' else kind}", flush=True)
        try:
            results.append(_chunk_session(R, rng, i, sid, side, kind, end_hold))
        except Exception as e:
            R.notes["errors"].append(f"session {i}: {traceback.format_exc()}")
            results.append({"i": i, "sid": sid, "kind": kind, "error": repr(e)})
        time.sleep(0.3 if end_hold == "target" and kind == "normal" else 2.0)
    return summarize_chunk(R, results)


def _chunk_session(R: Runner, rng, i, sid, side, kind, end_hold) -> dict:
    a = R.a
    base = {"stream": sid, "session_id": sid, "execution_id": sid, "generation": R.generation,
            "control_epoch": R.epoch, "mode": "chunk"}
    # a policy's first rows start from the observed arm state (GR00T's state input), not from SONIC's reference: a
    # session that takes over a hold starts where the arm physically is
    ref = R.ref_mj17()
    qm = R.q_mj17()
    start = ref[:3] + qm[3:]
    traj = Reach(start, side, i)
    t_open = time.monotonic()
    rep = R.arm({**base, "t_wall": time.time(), "hold_on_end": "measured", "watchdog_s": 2.0, "lead_s": a.lead,
                 "left_hand": [0.0] * 7, "right_hand": [0.0] * 7, "hands_blend_s": 0.3}, op_id=f"arm-{sid}")
    traj_step = max(max(abs(x - y) for x, y in zip(traj.at(k * DT)[0][3:], traj.at((k + 1) * DT)[0][3:]))
                    for k in range(int(traj.duration / DT)))
    out = {"i": i, "sid": sid, "kind": kind, "side": side, "end": end_hold, "start_reply": rep.get("state"),
           "traj_max_step_rad": round(traj_step, 4),
           "start_error": rep.get("error"), "ik_err_m": _r(traj.ik_err), "chunks": [], "late": [],
           "epoch": R.epoch, "t_open": t_open}
    if not rep.get("ok"):
        return out
    time.sleep(0.3)
    t_sess0 = time.monotonic()
    seq = 0
    t_last_obs = -1e9
    stop_at = {"cancel": 2.6, "halt": 2.6}.get(kind, traj.duration)
    ack = None
    while True:
        now = time.monotonic()
        if now - t_sess0 >= stop_at:
            break
        if now - t_last_obs < a.replan_s:
            time.sleep(0.005)
            continue
        t_obs = R.mon.dbg_t                                   # observation = the newest g1_debug receive time
        t_last_obs = t_obs
        time.sleep(a.inference_s)                             # simulated inference
        seq += 1
        off = [rng.gauss(0.0, a.noise) for _ in range(17)]
        rows, lh, rh = [], [], []
        for k in range(40):
            q, hands = traj.at(t_obs - t_sess0 + k * DT)
            rows.append(jm.wire_from_mj17([v + (o if kk >= 3 else 0.0) for kk, (v, o) in enumerate(zip(q, off))]))
            lh.append(hands["left"])
            rh.append(hands["right"])
        msg = {**base, "t_wall": time.time(), "chunk": {"seq": seq, "t0_mono": t_obs, "dt": DT, "order": "wire",
                                                         "upper_body": rows, "left_hand": lh, "right_hand": rh,
                                                         "inference_ms": round(a.inference_s * 1000, 1)}}
        t_send = time.monotonic()
        rep = R.arm(msg)
        out["chunks"].append({"seq": seq, "t_obs": t_obs, "t_send": t_send, "t_reply": time.monotonic(),
                              "ok": rep.get("ok"), "error": rep.get("error"),
                              "dropped": (rep.get("data") or {}).get("dropped")})
    q_at_stop = R.q_mj17()
    hands_at_stop = {s: list((R.mon.dbg_latest or {}).get(f"{s}_hand_q") or []) for s in ("left", "right")}
    if kind == "normal":
        rep = R.arm({**base, "t_wall": time.time(), "end": True, "hold_on_end": end_hold, "reason": "done"})
        out["end_reply"] = rep.get("state") if rep.get("ok") else rep.get("error")
    elif kind == "cancel":
        t_c = time.monotonic()
        rep = R.arm({**base, "t_wall": time.time(), "end": True, "hold_on_end": "measured", "reason": "cancelled"})
        ack = time.monotonic()
        out["cancel"] = {"reply": rep.get("state") if rep.get("ok") else rep.get("error"),
                         "ack_ms": round((ack - t_c) * 1e3, 2)}
    else:
        h = R.bc.halt(R.epoch, timeout_s=0.1, reason="arm_wave_test")
        ack = time.monotonic()
        out["halt"] = {"acked": h.get("acked"), "rtt_ms": h.get("rtt_ms"), "wait_ms": h.get("wait_ms"),
                       "arms_latched": (h.get("body") or {}).get("arms_latched"),
                       "arm": (h.get("body") or {}).get("arm"), "handle_ms": (h.get("body") or {}).get("handle_ms")}
    out["t_stop"] = time.monotonic()
    if ack is not None:
        # in-flight chunks after the ack, poisoned: +0.4 rad on the moving arm's elbow
        k_el = traj.k_el
        poison = []
        for j in range(3):
            q, hands = traj.at(time.monotonic() - t_sess0)
            q = list(q)
            q[k_el] += 0.4
            seq += 1
            rep = R.arm({**base, "t_wall": time.time(),
                         "chunk": {"seq": seq, "t0_mono": time.monotonic(), "dt": DT, "order": "wire",
                                   "upper_body": [jm.wire_from_mj17(q)] * 40, "left_hand": [hands["left"]] * 40,
                                   "right_hand": [hands["right"]] * 40}})
            poison.append({"seq": seq, "ok": rep.get("ok"), "error": rep.get("error")})
            time.sleep(0.05)
        out["late"] = poison
        time.sleep(1.0)
        w_before = R.mon.wire_between(ack - 0.3, ack)
        w_after = R.mon.wire_between(ack, ack + 1.0)
        el_b = [w[2][k_el] for w in w_before if w[2] is not None]
        el_a = [w[2][k_el] for w in w_after if w[2] is not None]
        el_ack = el_b[-1] if el_b else None
        out["after_ack"] = {"wire_elbow_at_ack": _r(el_ack),
                            "wire_elbow_max_rise": _r(max(el_a) - el_ack) if el_a and el_ack is not None else None,
                            "poison_applied": bool(el_a and el_ack is not None and max(el_a) - el_ack > 0.2),
                            "n_wire": len(el_a)}
        if kind == "halt":
            q_after = R.q_mj17()
            hq = {s: list((R.mon.dbg_latest or {}).get(f"{s}_hand_q") or []) for s in ("left", "right")}
            w_last = w_after[-1] if w_after else None
            out["after_ack"].update({
                "measured_arm_drift_rad_1s": _r(max(abs(q_after[k] - q_at_stop[k]) for k in ARM_K)),
                "wire_hands_vs_measured_at_halt_rad": None if w_last is None or w_last[3] is None else _r(max(
                    max(abs(x - y) for x, y in zip(w_last[3], hands_at_stop["left"])),
                    max(abs(x - y) for x, y in zip(w_last[4], hands_at_stop["right"])))),
                "hand_closure_at_halt": {s: _r(jm.hand_closure_of(s, hands_at_stop[s]), 3) for s in ("left", "right")
                                         if len(hands_at_stop[s]) == 7},
                "hand_closure_1s_later": {s: _r(jm.hand_closure_of(s, hq[s]), 3) for s in ("left", "right")
                                          if len(hq[s]) == 7}})
            st = R.bc.status()
            out["after_ack"]["arm_mode_latched"] = (st.get("arm") or {}).get("mode")
            rep = R.bc.resume(R.epoch)
            out["resume"] = rep.get("ok")
            R.epoch += 1
            rep = R.arm({"stream": f"{sid}-release", "end": True, "control_epoch": R.epoch, "t_wall": time.time()})
            out["release_reply"] = rep.get("ok"), (rep.get("data") or {}).get("released"), rep.get("error")
    ev = R.wait_terminal(f"arm-{sid}", 5.0)
    out["terminal"] = None if ev is None else {"state": ev["state"], **{k: (ev.get("data") or {}).get(k) for k in (
        "ended_by", "hold", "reason", "chunks", "clamped_frac_total", "slew_frac_total", "stall_s_max",
        "max_step_rad", "cross_fades", "duration_s", "lead_s")}}
    return out


def _steps(samples, t_idx=0, v_idx=2, win=None):
    """Max per-sample arm-joint step between consecutive samples (optionally only where the later sample falls in
    one of the windows)."""
    best = 0.0
    prev = None
    for s in samples:
        v = s[v_idx]
        if v is None:
            prev = None
            continue
        if prev is not None and (win is None or any(a <= s[t_idx] <= b for a, b in win)):
            best = max(best, max(abs(v[k] - prev[k]) for k in ARM_K))
        prev = v
    return best


def summarize_chunk(R: Runner, results: list) -> dict:
    t0, t1 = R.t_start, time.monotonic()
    wire = R.mon.wire_between(t0, t1)
    dbg = [(d[0], None, jm.mj17_from_mujoco(d[1])) for d in R.mon.dbg_between(t0, t1)]
    bounds = []
    for r in results:
        for c in r.get("chunks", []):
            if c.get("ok") and not c.get("dropped"):
                bounds.append((c["t_reply"], c["t_reply"] + 0.16))
    ok_sessions = [r for r in results if r.get("start_reply") == "accepted"]
    terms = [r.get("terminal") or {} for r in results]
    cancels = [r for r in results if r["kind"] == "cancel"]
    halts = [r for r in results if r["kind"] == "halt"]

    def no_chunk_after_ack(r):
        late_ok = all(not x.get("ok") for x in r.get("late", []))
        wire_ok = not (r.get("after_ack") or {}).get("poison_applied", True)
        return bool(late_ok and wire_ok)

    S = {
        "sessions": len(results), "sessions_opened": len(ok_sessions),
        "falls": R.mon.falls(t0, t1),
        "chunks_sent": sum(len(r.get("chunks", [])) for r in results),
        "chunks_applied_body": sum(((t.get("chunks") or {}).get("applied") or 0) for t in terms),
        "chunks_dropped_body": _sum_dropped(terms),
        "terminal_states": [(t.get("state"), t.get("ended_by"), t.get("hold")) for t in terms],
        "wire_max_step_rad": {"all": round(_steps(wire), 4), "chunk_boundaries": round(_steps(wire, win=bounds), 4)},
        "trajectory_max_step_rad": max([r.get("traj_max_step_rad") or 0.0 for r in results] or [0.0]),
        "body_max_step_rad_per_tick": max([t.get("max_step_rad") or 0.0 for t in terms] or [0.0]),
        "measured_max_step_rad": {"all": round(_steps(dbg), 4), "chunk_boundaries": round(_steps(dbg, win=bounds), 4)},
        "slew_limit_rad_per_tick": 0.12,
        "clamped_frac_total_max": max([t.get("clamped_frac_total") or 0.0 for t in terms] or [0.0]),
        "slew_frac_total_max": max([t.get("slew_frac_total") or 0.0 for t in terms] or [0.0]),
        "stall_s_max": max([t.get("stall_s_max") or 0.0 for t in terms] or [0.0]),
        "cancel": {"n": len(cancels), "ok": sum(no_chunk_after_ack(r) and (r.get("terminal") or {}).get("state")
                                               == "succeeded" for r in cancels),
                   "ack_ms": [(r.get("cancel") or {}).get("ack_ms") for r in cancels]},
        "halt": {"n": len(halts), "ok": sum(no_chunk_after_ack(r) and (r.get("halt") or {}).get("acked") and
                                           (r.get("terminal") or {}).get("ended_by") == "halt" for r in halts),
                 "rtt_ms": [(r.get("halt") or {}).get("rtt_ms") for r in halts],
                 "arms_latched": [(r.get("halt") or {}).get("arms_latched") for r in halts],
                 "arm_latch_ms": [((r.get("halt") or {}).get("arm") or {}).get("latch_ms") for r in halts],
                 "measured_arm_drift_rad_1s": [(r.get("after_ack") or {}).get("measured_arm_drift_rad_1s")
                                               for r in halts],
                 "hand_closure_at_halt_vs_1s": [((r.get("after_ack") or {}).get("hand_closure_at_halt"),
                                                 (r.get("after_ack") or {}).get("hand_closure_1s_later")) for r in halts]},
        "late_chunk_replies": [[x.get("error") for x in r.get("late", [])] for r in cancels + halts],
        "rtf": _rtf(R),
    }
    R.save("sessions.json", results)
    return S


def _sum_dropped(terms):
    out = {}
    for t in terms:
        for k, v in (((t.get("chunks") or {}).get("dropped")) or {}).items():
            out[k] = out.get(k, 0) + int(v or 0)
    return out


def _rtf(R):
    st = R.p1.try_call("get_stats") or {}
    return {k: st.get(k) for k in ("rtf_total", "rtf_10s", "rtf_1s_min", "rtf_1s_below_0p95_frac", "overruns")}


# ================================================================================================= pick + carry
def scene_live(R: Runner) -> dict:
    """P1's scene (get_scene_info: load-time boxes) with every dynamic object's LIVE box (get_objects): an earlier run
    (a GR00T staging, a pick, a push) may have moved it since the load."""
    scene = R.p1.call("get_scene_info")
    live = {o["id"]: o for o in (R.p1.try_call("get_objects", dynamic_only=True) or {}).get("objects") or []}
    moved = {}
    for o in scene.get("objects") or []:
        lo = live.get(o.get("id"))
        if lo and lo.get("aabb"):
            c0 = np.mean(np.asarray(o["aabb"], float), axis=0)
            c1 = np.mean(np.asarray(lo["aabb"], float), axis=0)
            if float(np.linalg.norm(c1 - c0)) > 0.02:
                moved[o["id"]] = _r(float(np.linalg.norm(c1 - c0)), 3)
            o["aabb"], o["held_by"] = lo["aabb"], lo.get("held_by")
    scene["_moved_since_load"] = moved
    return scene


def find_object(scene: dict, oid: str | None):
    objs = scene.get("objects") or []
    by_id = {o.get("id"): o for o in objs}
    if oid:
        return by_id[oid], objs
    return by_id["RemoteControl|surface|2|30"], objs


def support_of(obj, objs):
    """The furniture whose footprint holds the object and whose top is at the object's bottom."""
    (ox0, oy0, oz0), (ox1, oy1, oz1) = obj["aabb"]
    cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    best = None
    for o in objs:
        if o is obj or "surface" in str(o.get("id")):
            continue
        (x0, y0, z0), (x1, y1, z1) = o["aabb"]
        if x0 <= cx <= x1 and y0 <= cy <= y1 and abs(z1 - oz0) < 0.06:
            if best is None or (x1 - x0) * (y1 - y0) < (best["aabb"][1][0] - best["aabb"][0][0]) * \
                    (best["aabb"][1][1] - best["aabb"][0][1]):
                best = o
    return best


# --faces: the robot's facing at the stance -> the support AABB side it stands at (that side's outward normal)
FACE_NORMAL = {"+x": (-1.0, 0.0), "-x": (1.0, 0.0), "+y": (0.0, -1.0), "-y": (0.0, 1.0)}


def stance_for(obj, sup, gap: float, lateral: float, side: str, face: str | None = None):
    """Stand `gap` from the support edge nearest to the object, facing it, the object `lateral` to the arm's side.
    `face` ('+x', '-x', '+y', '-y': the robot's facing) picks the edge instead: an L-shaped counter's AABB puts the
    nearest edge on the wall side (H40 CounterTop|6|1: every object on its east leg is nearest the wall)."""
    (x0, y0, _), (x1, y1, _) = sup["aabb"]
    (ox0, oy0, _), (ox1, oy1, _) = obj["aabb"]
    cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    edges = [(cx - x0, (-1.0, 0.0)), (x1 - cx, (1.0, 0.0)), (cy - y0, (0.0, -1.0)), (y1 - cy, (0.0, 1.0))]
    if face and face != "auto":
        d, n = next(e for e in edges if e[1] == FACE_NORMAL[face])
    else:
        d, n = min(edges, key=lambda e: e[0])
    f = np.array([-n[0], -n[1]])                      # facing: towards the support
    left = np.array([-f[1], f[0]])
    sgn = -1.0 if side == "right" else 1.0
    xy = np.array([cx, cy]) - (d + gap) * f - sgn * lateral * left
    return float(xy[0]), float(xy[1]), math.atan2(f[1], f[0]), d


def script(R: Runner, args: dict, timeout: float = 20.0) -> dict:
    op_id = f"as-{args.get('phase', 'scan')}-{int(time.time() * 1000) % 10 ** 8}"
    op = "scan" if "phase" not in args else "arm_script"
    t0 = time.monotonic()
    rep = R.bc.request(op, {**args, "control_epoch": R.epoch, "generation": R.generation}, op_id=op_id)
    if not rep.get("ok"):
        return {"op_id": op_id, "reply": rep, "t0": t0}
    ev = R.wait_terminal(op_id, timeout)
    return {"op_id": op_id, "reply_state": rep.get("state"), "plan": rep.get("data"), "t0": t0,
            "t1": time.monotonic(), "terminal": None if ev is None else {"state": ev["state"], **(ev.get("data") or {})}}


def palm_err_tool(R: Runner, t0, t1, side, goal_w):
    errs = []
    for d in R.mon.dbg_between(t0, t1):
        g = R.mon.pose_at(d[0])
        if g is None or abs(g[0] - d[0]) > 0.05:
            continue
        errs.append(float(np.linalg.norm(palm_world(d[1], g, side) - np.asarray(goal_w))))
    return errs


def pick_test(R: Runner) -> dict:
    a = R.a
    scene = scene_live(R)
    obj, objs = find_object(scene, a.object)
    sup = support_of(obj, objs)
    if sup is None:
        raise RuntimeError(f"no support found for {obj['id']}")
    (ox0, oy0, oz0), (ox1, oy1, oz1) = obj["aabb"]
    grasp_w = [(ox0 + ox1) / 2, (oy0 + oy1) / 2, oz1 + a.grasp_above]
    S = {"object": obj["id"], "support": sup["id"], "grasp_w": _r(grasp_w), "object_aabb": _r(obj["aabb"]),
         "moved_since_load": scene.get("_moved_since_load")}
    print(f"[arm_wave] pick {obj['id']} on {sup['id']}", flush=True)
    # go_to (A*-safe), raise the hand over the support first with --rise-gap, then `approach` (body B.6) in
    _stance_at(R, obj, sup, grasp_w, S, (a.faces or "auto").split(",")[0].strip())
    g = R.mon.last_gt()
    S["pose_at_stance"] = _r([g[1], g[2], math.degrees(g[4])], 3)
    ext = {}
    if a.clear:                                      # hand clearance over the object + the collision-checked path
        ext = {"clear_z": oz1, "avoid_boxes": [sup["aabb"], obj["aabb"]]}
    trials = []
    for k in range(a.trials):
        tr = {"k": k}
        tr["pregrasp"] = script(R, {"phase": "pregrasp", "arm": a.arm, "target_w": grasp_w, **ext})
        tr["grasp"] = gr = script(R, {"phase": "grasp", "arm": a.arm, "target_w": grasp_w, "closure": 0.6,
                                      "settle_s": a.settle, **ext})
        term = gr.get("terminal") or {}
        if term.get("state") == "succeeded":
            m1 = gr["t1"] - 0.05
            e = palm_err_tool(R, m1 - min(1.0, a.settle - 0.3), m1, a.arm, (gr.get("plan") or {}).get("goal_w")
                              or grasp_w)
            tr["tool_palm_err_w_m"] = {"median": _pct(e, 50), "p90": _pct(e, 90), "max": _r(max(e) if e else None),
                                       "n": len(e)}
            tr["_tool_errs"] = e
        if k < a.trials - 1:
            tr["retract"] = script(R, {"phase": "pregrasp", "arm": a.arm, "target_w": grasp_w, "hold_on_end": "target",
                                       **ext})
        trials.append(tr)
    allerr = [x for tr in trials for x in tr.pop("_tool_errs", [])]
    S["grasp_palm_err_w_m_tool"] = {"median": _pct(allerr, 50), "p90": _pct(allerr, 90),
                                    "max": _r(max(allerr) if allerr else None), "n": len(allerr)}
    S["grasp_palm_err_w_m_body"] = [((tr["grasp"].get("terminal") or {}).get("palm_err_w_m")) for tr in trials]
    S["grasp_palm_err_b_m_body"] = [((tr["grasp"].get("terminal") or {}).get("palm_err_b_m")) for tr in trials]
    S["ik_err_m"] = [((tr["grasp"].get("plan") or {}).get("ik_err_m")) for tr in trials]
    if a.attach != "none":
        # P1.3 attach (STEPPING STONE: the object follows the palm; a Dex3 grasp is not reliable yet), with the hand
        # at the last grasp, gated like sonic_arm_script's: the palm (FK of the measured joints on the GT pelvis)
        # within 0.10 m of the grasp point, else no attach (P1 would snap an object from anywhere to the hand)
        with R.mon.lock:
            d = R.mon.dbg[-1] if R.mon.dbg else None
        g = R.mon.last_gt()
        gap = None if d is None or g is None else float(np.linalg.norm(palm_world(d[1], g, a.arm) - grasp_w))
        last_ok = bool(trials) and (trials[-1]["grasp"].get("terminal") or {}).get("state") == "succeeded"
        if last_ok and gap is not None and gap < 0.10:
            try:
                rep = R.p1.call("attach", id=obj["id"], arm=a.arm, mode=a.attach)
            except Exception as e:  # noqa: BLE001  (P1Error: the op's code and message)
                rep = {"error": repr(e)}
        else:
            rep = {"error": "not attached: " + ("the last grasp did not succeed" if not last_ok
                                                else f"palm {gap} m from the grasp point (gate 0.10 m)")}
        S["attach"] = {k: rep.get(k) for k in ("ok", "held_by", "mode", "snapped", "dist_m", "error")}
        S["attach"].update({"label": "STEPPING STONE", "palm_to_grasp_point_m": _r(gap, 3)})
    # carry: lift, tuck, CarryLock, turn, 2 m walk
    lift = script(R, {"phase": "lift", "arm": a.arm, "lift_m": 0.06})
    # step back from the support with the lifted arm held (CarryLock), then tuck: lowering the hand next to the
    # support's front edge hits it (live 20260929-080443 / -082958: the fist stopped on the dresser edge)
    g = R.mon.last_gt()
    bx, by = g[1] - a.back * math.cos(g[4]), g[2] - a.back * math.sin(g[4])
    hb = R.bc.approach(float(bx), float(by), yaw=g[4], tol=(0.05, 5.0), timeout=45)
    S["step_back"] = {"state": hb.state, "reason": hb.reason, **{k: (hb.result or {}).get(k) for k in ("pos_err",)}}
    carry = script(R, {"phase": "carry", "arm": a.arm})
    st = R.bc.status()
    S["carry_lock"] = (st.get("arm") or {}).get("carry")
    hold_pose = ((st.get("arm") or {}).get("hold") or {}).get("pose_mj17")
    t_w0 = time.monotonic()
    g0 = R.mon.last_gt()
    if a.carry_to:
        # an A* go_to with the arm held (the override rides on every planner message): a blind straight walk from the
        # support can run into furniture (H40 dresser: an obstacle 1.4 m straight behind the stance)
        cx, cy = (float(v) for v in a.carry_to.split(","))
        ht = R.bc.go_to(cx, cy, timeout_s=120)
        hw = ht
    else:
        ht = R.bc.turn_to(wrap(g0[4] + math.pi), timeout=40)
        hw = R.bc.walk(vx=a.walk_v, duration_s=a.walk_s)
    t_w1 = time.monotonic()
    g1 = R.mon.last_gt()
    drift = []
    if hold_pose is not None:
        want = K.points({**K.named_from_mj17(hold_pose)})
        for d in R.mon.dbg_between(t_w0, t_w1):
            qm = jm.mj17_from_mujoco(d[1])
            got_n = K.named_from_mj17(qm)
            want_n = K.named_from_mj17(hold_pose)
            for wj in jm.WAIST_JOINTS:
                want_n[wj] = got_n[wj]
            pw, pg = K.points(want_n)[f"{a.arm}_palm"], K.points(got_n)[f"{a.arm}_palm"]
            drift.append(float(np.linalg.norm(pw - pg)))
    with R.mon.lock:
        gts = [g for g in R.mon.gt if t_w0 <= g[0] <= t_w1]
    path_m = sum(math.hypot(b[1] - q[1], b[2] - q[2]) for q, b in zip(gts, gts[1:]))
    S["carry_walk"] = {"mode": "go_to " + a.carry_to if a.carry_to else "turn 180 + walk",
                       "gt_path_m": _r(path_m, 3),
                       "turn": {"state": ht.state, "yaw_err_deg": (ht.result or {}).get("yaw_err_deg")},
                       "walk": {"state": hw.state, "walked_m": (hw.result or {}).get("walked_m"),
                                "displacement_m": (hw.result or {}).get("displacement_m")},
                       "gt_displacement_m": _r(math.hypot(g1[1] - g0[1], g1[2] - g0[2]), 3),
                       "falls": R.mon.falls(t_w0, t_w1),
                       "palm_drift_m": {"rms": _r(math.sqrt(sum(x * x for x in drift) / len(drift))) if drift else None,
                                        "p90": _pct(drift, 90), "max": _r(max(drift) if drift else None),
                                        "n": len(drift)},
                       "carry_after": (R.bc.status().get("arm") or {}).get("carry")}
    if a.attach != "none" and S["attach"].get("held_by"):
        # did the object come along? P1's live box vs the palm (FK of the measured joints on the GT pelvis)
        o = next((x for x in (R.p1.try_call("get_objects", ids=[obj["id"]]) or {}).get("objects") or []), None)
        with R.mon.lock:
            d = R.mon.dbg[-1] if R.mon.dbg else None
        g = R.mon.last_gt()
        if o is not None and d is not None and g is not None:
            c = np.mean(np.asarray(o["aabb"], float), axis=0)
            S["carry_walk"]["object_after"] = {
                "held_by": o.get("held_by"), "centre_w": _r(c, 3),
                "centre_to_palm_m": _r(float(np.linalg.norm(c - palm_world(d[1], g, a.arm))), 3),
                "height_above_floor_m": _r(float(c[2]), 3),
                "in_hand": o.get("held_by") == a.arm and float(np.linalg.norm(c - palm_world(d[1], g, a.arm))) < 0.15}
    S["lift"], S["carry"] = lift.get("terminal"), carry.get("terminal")
    rel = script(R, {"phase": "release", "arm": a.arm})
    ret = script(R, {"phase": "retract", "arm": a.arm})
    S["release"], S["retract"] = (rel.get("terminal") or {}).get("state"), (ret.get("terminal") or {}).get("state")
    if a.attach != "none" and S["attach"].get("held_by"):
        # put the object back where it stood (detach with a pose): the next test and the house start as they were
        (x0, y0, z0), (x1, y1, z1) = obj["aabb"]
        rep = R.p1.try_call("detach", id=obj["id"], pose=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        S["detach_put_back"] = {k: (rep or {}).get(k) for k in ("ok", "pos", "error", "code")}
    S["trials"] = trials
    S["falls_total"] = R.mon.falls(R.t_start, time.monotonic())
    S["rtf"] = _rtf(R)
    return S


# ================================================================================================= grasp accuracy (B-D4)
def grasp_candidates(scene: dict, a) -> list[dict]:
    """Small objects on furniture tops the G1 palm can reach from an A*-safe stance: top 0.6-1.1 m, object <= 0.15 m
    tall and <= 0.3 m across, its centre <= --max-edge m from the support edge nearest to it."""
    objs = scene.get("objects") or []
    out = []
    for o in objs:
        if "surface" not in str(o.get("id")):
            continue
        (x0, y0, z0), (x1, y1, z1) = o["aabb"]
        if z1 - z0 > 0.15 or max(x1 - x0, y1 - y0) > 0.3:
            continue
        sup = support_of(o, objs)
        if sup is None:
            continue
        top = sup["aabb"][1][2]
        if not (0.6 <= top <= 1.1):
            continue
        try:
            _, _, _, d = stance_for(o, sup, a.gap, a.lateral, a.arm)
        except Exception:
            continue
        if d > a.max_edge:
            continue
        out.append({"id": o["id"], "support": sup["id"], "top_z": round(top, 3), "edge_dist_m": round(d, 3),
                    "size": [round(x1 - x0, 3), round(y1 - y0, 3), round(z1 - z0, 3)]})
    return sorted(out, key=lambda c: (c["support"], c["edge_dist_m"]))


def _err_b(R: Runner, t0, t1, side, goal_w) -> list:
    """Palm - goal in the GT pelvis frame (x forward, y left, z up), 50 Hz."""
    out = []
    for d in R.mon.dbg_between(t0, t1):
        g = R.mon.pose_at(d[0])
        if g is None or abs(g[0] - d[0]) > 0.05:
            continue
        pw = palm_world(d[1], g, side)
        out.append(world_to_pelvis(pw, Pose_(g)) - world_to_pelvis(goal_w, Pose_(g)))
    return out


def _stance_at(R: Runner, obj, sup, grasp_w, S: dict, face: str | None = None) -> None:
    a = R.a
    gap = a.rise_gap if a.rise_gap else a.gap
    sx, sy, syaw, d_edge = stance_for(obj, sup, gap, a.lateral, a.arm, face)
    S["edge_dist_m"], S["stance"], S["face"] = _r(d_edge), _r([sx, sy, math.degrees(syaw)], 3), face or "auto"
    h = R.bc.go_to(sx, sy, yaw=syaw, timeout_s=120)
    S["go_to"] = {"state": h.state, "reason": h.reason, "pos_err": (h.result or {}).get("pos_err")}
    time.sleep(1.0)
    if a.rise_gap:
        # raise the hand above the support before stepping in: from the A* stance the open hand, sweeping up and
        # forward from the arm's rest pose, catches the support's front edge (live 20260929-112144: the fingers stuck
        # under the dresser top, SONIC stepped back 0.22 m). Raise it `rise_gap` from the edge, then `approach` in
        # with the arm held (the override rides on every planner message while the legs walk).
        g = R.mon.last_gt()
        top_b = sup["aabb"][1][2] - g[5]
        sgn = -1.0 if a.arm == "right" else 1.0
        rz = script(R, {"phase": "carry", "arm": a.arm, "carry_b": [0.22, sgn * a.lateral, top_b + a.raise_above],
                        "avoid_boxes": [sup["aabb"]], "hold_on_end": "target"})
        term = rz.get("terminal") or {}
        S["raise"] = {"reply": (rz.get("reply") or {}).get("error") or rz.get("reply_state"),
                      "state": term.get("state"), "path": (rz.get("plan") or {}).get("path"),
                      "palm_err_b_m": term.get("palm_err_b_m"), "pelvis_shift_m": term.get("pelvis_shift_m")}
    f = np.array([math.cos(syaw), math.sin(syaw)])
    left = np.array([-f[1], f[0]])
    sgn = -1.0 if a.arm == "right" else 1.0
    fx, fy = np.array(grasp_w[:2]) - a.reach * f - sgn * a.lateral * left
    S["approach"] = []
    for _ in range(3):
        g = R.mon.last_gt()
        off = world_to_pelvis(grasp_w, Pose_(g))
        if abs(off[0] - a.reach) < 0.025 and abs(off[1] - sgn * a.lateral) < 0.03:
            break
        ha = R.bc.approach(float(fx), float(fy), yaw=syaw, tol=(0.025, 3.0), timeout=45)
        S["approach"].append({"state": ha.state, "reason": ha.reason,
                              **{k: (ha.result or {}).get(k) for k in ("pos_err", "yaw_err_deg", "attempts")}})
        time.sleep(0.8)
    g = R.mon.last_gt()
    S["grasp_b_at_stance"] = _r(world_to_pelvis(grasp_w, Pose_(g)))


def _one_grasp(R: Runner, grasp_w, extra: dict) -> dict:
    a = R.a
    tr = {"pregrasp": script(R, {"phase": "pregrasp", "arm": a.arm, "target_w": grasp_w, **extra})}
    tr["pregrasp_path"] = ((tr["pregrasp"].get("plan") or {}).get("path"))
    pre = tr["pregrasp"].get("terminal") or {}
    if pre.get("state") != "succeeded":
        tr["skipped"] = f"pregrasp {pre.get('state') or tr['pregrasp'].get('reply')}"
        return tr
    gr = script(R, {"phase": "grasp", "arm": a.arm, "target_w": grasp_w, "closure": a.closure,
                    "settle_s": a.settle, **{k: v for k, v in extra.items() if k in ("preshape", "clear_z",
                                                                                     "avoid_boxes")}})
    plan = gr.get("plan") or {}
    tr["grasp"] = {"plan": {k: plan.get(k) for k in ("goal_b", "goal_w", "ik_err_m", "ik_ms", "ik_seed", "move_s",
                                                     "clear")},
                   "reply": gr.get("reply")}
    term = gr.get("terminal") or {}
    tr["grasp"]["state"] = term.get("state")
    if term.get("state") == "succeeded":
        m1 = gr["t1"] - 0.05
        m0 = m1 - min(1.0, a.settle - 0.3)
        goal_w = plan.get("goal_w") or grasp_w           # what the script commanded (raised for hand clearance)
        e = palm_err_tool(R, m0, m1, a.arm, goal_w)
        e0 = palm_err_tool(R, m0, m1, a.arm, grasp_w)
        eb = _err_b(R, m0, m1, a.arm, goal_w)
        tr["tool_palm_err_w_m"] = {"median": _pct(e, 50), "p90": _pct(e, 90), "max": _r(max(e) if e else None),
                                   "n": len(e)}
        tr["tool_palm_err_vs_grasp_point_m"] = {"median": _pct(e0, 50), "p90": _pct(e0, 90), "n": len(e0)}
        tr["goal_minus_grasp_point_m"] = _r(np.asarray(goal_w) - np.asarray(grasp_w))
        tr["err_pelvis_frame_mean_m"] = _r(np.mean(eb, axis=0)) if eb else None
        tr["body_palm_err_w_m"], tr["body_palm_err_b_m"] = term.get("palm_err_w_m"), term.get("palm_err_b_m")
        tr["pelvis_shift_m"], tr["track_updates"] = term.get("pelvis_shift_m"), term.get("track_updates")
        tr["_errs"] = e
    return tr


def grasp_at(R: Runner, scene: dict, oid: str, face: str | None = None) -> dict:
    a = R.a
    obj, objs = find_object(scene, oid)
    sup = support_of(obj, objs)
    (ox0, oy0, oz0), (ox1, oy1, oz1) = obj["aabb"]
    grasp_w = [(ox0 + ox1) / 2, (oy0 + oy1) / 2, oz1 + a.grasp_above]
    S = {"object": oid, "support": None if sup is None else sup["id"], "object_aabb": _r(obj["aabb"]),
         "grasp_w": _r(grasp_w)}
    print(f"[arm_wave] grasp {oid} on {S['support']}", flush=True)
    if sup is None:
        S["error"] = "no support"
        return S
    _stance_at(R, obj, sup, grasp_w, S, face)
    extra = {"approach": a.approach, "preshape": a.preshape}
    if a.above_m is not None:
        extra["above_m"] = a.above_m
    if a.clear:
        extra["clear_z"] = oz1                        # the hand must clear the object's top (body: hand clearance)
        extra["avoid_boxes"] = [sup["aabb"], obj["aabb"]]
    S["trials"] = []
    for k in range(a.trials):
        tr = _one_grasp(R, grasp_w, extra)
        tr["k"] = k
        S["trials"].append(tr)
        print(f"[arm_wave]   grasp {k}: {tr.get('tool_palm_err_w_m')} err_b {tr.get('err_pelvis_frame_mean_m')} "
              f"{tr.get('skipped') or ''}", flush=True)
        if k < a.trials - 1 and tr.get("grasp", {}).get("state") == "succeeded":
            script(R, {"phase": "pregrasp", "arm": a.arm, "target_w": grasp_w, "hold_on_end": "target", **extra})
    errs = [x for tr in S["trials"] for x in tr.pop("_errs", [])]
    S["palm_err_w_m_tool"] = {"median": _pct(errs, 50), "p90": _pct(errs, 90), "max": _r(max(errs) if errs else None),
                              "n": len(errs)}
    S["_errs"] = errs
    if a.free_air:
        # the same pelvis-frame goal with the support out of reach: step back, then grasp in free air (no contact)
        goals = [tr["grasp"]["plan"]["goal_b"] for tr in S["trials"] if (tr.get("grasp") or {}).get("plan", {}).get(
            "goal_b")]
        script(R, {"phase": "retract", "arm": a.arm})
        g = R.mon.last_gt()
        hb = R.bc.approach(g[1] - 0.3 * math.cos(g[4]), g[2] - 0.3 * math.sin(g[4]), yaw=g[4], tol=(0.05, 5.0),
                           timeout=45)
        S["free_air"] = {"step_back": hb.state, "trials": []}
        if goals:
            gb = np.mean(np.asarray(goals, float), axis=0)
            for k in range(min(2, a.trials)):
                g = R.mon.last_gt()
                gw = pelvis_to_world(gb, Pose_(g)).tolist()
                tr = _one_grasp(R, gw, extra)
                tr.pop("_errs", None)
                S["free_air"]["trials"].append({"goal_b": _r(gb), **tr})
                print(f"[arm_wave]   free-air {k}: {tr.get('tool_palm_err_w_m')} err_b "
                      f"{tr.get('err_pelvis_frame_mean_m')}", flush=True)
    script(R, {"phase": "retract", "arm": a.arm})
    return S


def grasp_test(R: Runner) -> dict:
    a = R.a
    scene = scene_live(R)
    if a.list or not a.objects:
        cands = grasp_candidates(scene, a)
        print(json.dumps(cands, indent=1), flush=True)
        if a.list or not cands:
            return {"candidates": cands}
        seen, ids = set(), []
        for c in cands:                                   # one object per support, nearest the edge first
            if c["support"] not in seen:
                seen.add(c["support"])
                ids.append(c["id"])
        ids = ids[:a.n_objects]
    else:
        ids = a.objects.split(",")
    faces = [f.strip() for f in (a.faces or "").split(",") if f.strip()]
    faces += ["auto"] * (len(ids) - len(faces))
    S = {"variant": {k: getattr(a, k) for k in ("approach", "preshape", "reach", "lateral", "gap", "grasp_above",
                                                  "closure", "settle", "above_m", "arm", "clear", "rise_gap",
                                                  "raise_above")},
         "moved_since_load": scene.get("_moved_since_load"),
         "objects": [grasp_at(R, scene, oid, face) for oid, face in zip(ids, faces)]}
    errs = [x for o in S["objects"] for x in o.pop("_errs", [])]
    per = [tr["tool_palm_err_w_m"]["p90"] for o in S["objects"] for tr in o.get("trials", [])
           if tr.get("tool_palm_err_w_m")]
    S["pooled"] = {"grasps": len(per), "objects": sum(1 for o in S["objects"] if any(
                       tr.get("tool_palm_err_w_m") for tr in o.get("trials", []))),
                   "supports": len({o.get("support") for o in S["objects"] if any(
                       tr.get("tool_palm_err_w_m") for tr in o.get("trials", []))}),
                   "objects_tried": len(S["objects"]),
                   "palm_err_w_m": {"median": _pct(errs, 50), "p90": _pct(errs, 90),
                                    "max": _r(max(errs) if errs else None), "n": len(errs)},
                   "per_grasp_p90_m": per, "grasps_p90_under_3cm": sum(1 for x in per if x < 0.03)}
    S["pooled"]["pass_p90_under_3cm"] = bool(errs) and _pct(errs, 90) < 0.03 and len(per) >= 6 and \
        S["pooled"]["supports"] >= 2
    S["falls_total"] = R.mon.falls(R.t_start, time.monotonic())
    S["rtf"] = _rtf(R)
    return S


# ================================================================================================= scan
def scan_once(R: Runner, label: str, extra: dict | None = None) -> dict:
    t0 = time.monotonic()
    g0 = R.mon.last_gt()
    q0 = R.q_mj17()
    r = script(R, {"yaw_deg": R.a.yaw_deg, **(extra or {})}, timeout=20.0)
    t1 = time.monotonic()
    g1 = R.mon.last_gt()
    dev = 0.0
    legs = 0.0
    for d in R.mon.dbg_between(t0, t1):
        qm = jm.mj17_from_mujoco(d[1])
        dev = max(dev, max(abs(qm[k] - q0[k]) for k in ARM_K))
    term = r.get("terminal") or {}
    return {"label": label, "args": extra, "arms": term.get("arms"), "state": term.get("state"), "holds": term.get("holds"),
            "yaw_err_deg_max": term.get("yaw_err_deg_max"), "body_arm_dev_rad_max": term.get("arm_dev_rad_max"),
            "tool_arm_dev_rad_max": _r(dev), "waist_roll_pitch_dev_rad_max": term.get("waist_roll_pitch_dev_rad_max"),
            "base_shift_m": _r(math.hypot(g1[1] - g0[1], g1[2] - g0[2]), 4), "falls": R.mon.falls(t0, t1),
            "duration_s": _r(t1 - t0, 2), "reply": r.get("reply")}


def scan_test(R: Runner) -> dict:
    a = R.a
    S = {"standing": scan_once(R, "standing")}
    if a.scan_hold_variants:
        # arms held by the servo at SONIC's reference pose (as under CarryLock)
        rep_ = R.arm({"stream": "barm-scan-hold", "upper_body": R.ref_mj17(), "control_epoch": R.epoch})
        R.arm({"stream": "barm-scan-hold", "end": True, "hold_on_end": "target", "control_epoch": R.epoch})
        time.sleep(1.5)
        S["standing_hold"] = scan_once(R, "standing, arms held", {"arms": "hold"})
        R.bc.stop(arms=True)                                   # release whatever holds the arms now
        S["hold_start_reply"] = rep_.get("state") or rep_.get("error")
        time.sleep(2.0)
    if a.at_counter:
        scene = R.p1.call("get_scene_info")
        obj, objs = find_object(scene, a.object)
        sup = support_of(obj, objs)
        sx, sy, syaw, _ = stance_for(obj, sup, a.gap, a.lateral, a.arm)
        h = R.bc.go_to(sx, sy, yaw=syaw, timeout_s=120)
        S["go_to"] = {"state": h.state, "pos_err": (h.result or {}).get("pos_err"), "support": sup["id"]}
        time.sleep(1.5)
        S["at_counter"] = scan_once(R, f"at {sup['id']}")
    S["rtf"] = _rtf(R)
    return S


# ================================================================================================= main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["chunk", "pick", "scan", "grasp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--fake", action="store_true", help="run against in-process fakes (fake P1 + fake deploy + body)")
    ap.add_argument("--seed", type=int, default=7)
    # chunk
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--cancels", type=int, default=3)
    ap.add_argument("--halts", type=int, default=3)
    ap.add_argument("--lead", type=float, default=0.15)
    ap.add_argument("--replan-s", type=float, default=0.4)
    ap.add_argument("--inference-s", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=0.015)
    # pick / scan
    ap.add_argument("--object", default=None)
    ap.add_argument("--arm", default="right", choices=["left", "right"])
    ap.add_argument("--gap", type=float, default=0.26, help="go_to stance: distance to the support edge (A*-safe)")
    ap.add_argument("--reach", type=float, default=0.34, help="final stance: the grasp point this far ahead (approach)")
    ap.add_argument("--back", type=float, default=0.25, help="step back from the support before the carry tuck")
    ap.add_argument("--lateral", type=float, default=0.20)
    ap.add_argument("--grasp-above", type=float, default=0.03)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--walk-v", type=float, default=0.3)
    ap.add_argument("--walk-s", type=float, default=7.0)
    ap.add_argument("--attach", default="none", choices=["none", "fixed_joint", "follow"],
                    help="pick: P1.3 attach (STEPPING STONE) after the last grasp; checked after the carry walk, then "
                    "the object is put back where it stood")
    ap.add_argument("--carry-to", default=None, help="pick: x,y: the carry walk is a go_to there (A*) instead of a "
                    "180 deg turn and a blind --walk-s walk")
    ap.add_argument("--yaw-deg", type=float, nargs="+", default=[-35.0, 0.0, 35.0])
    ap.add_argument("--at-counter", action="store_true")
    # grasp (B-D4: accuracy over objects / surfaces)
    ap.add_argument("--objects", default=None, help="comma-separated object ids (default: one per support, auto)")
    ap.add_argument("--n-objects", type=int, default=2)
    ap.add_argument("--list", action="store_true", help="grasp: only list the candidate objects")
    ap.add_argument("--max-edge", type=float, default=0.16, help="grasp: object centre at most this from the edge")
    ap.add_argument("--approach", default="front", choices=["front", "above"])
    ap.add_argument("--faces", default=None, help="grasp/pick: per object (comma list, aligned with --objects), the "
                    "robot's facing at the stance: auto (the support edge nearest the object) | +x | -x | +y | -y")
    ap.add_argument("--preshape", type=float, default=0.0)
    ap.add_argument("--above-m", type=float, default=None)
    ap.add_argument("--closure", type=float, default=0.6)
    ap.add_argument("--free-air", action="store_true", help="grasp: repeat the goal in free air (no contact)")
    ap.add_argument("--clear", action="store_true", help="grasp: pass clear_z = the object's top (hand clearance) and "
                    "avoid_boxes = the support + the object")
    ap.add_argument("--rise-gap", type=float, default=0.0, help="grasp: go_to this far from the edge, raise the hand "
                    "above the support there (carry phase), then approach in with the arm held (0: off)")
    ap.add_argument("--raise-above", type=float, default=0.10, help="grasp: raised palm height above the support top")
    ap.add_argument("--scan-hold-variants", action="store_true")
    a = ap.parse_args(argv)
    if a.fake and a.port_offset is None:
        a.port_offset = 300
    R = Runner(a)
    rc = 0
    try:
        S = {"chunk": chunk_test, "pick": pick_test, "scan": scan_test, "grasp": grasp_test}[a.mode](R)
    except Exception:
        R.notes["errors"].append(traceback.format_exc())
        S = {"error": traceback.format_exc()}
        rc = 1
    S["mode"], S["args"], S["duration_s"] = a.mode, vars(a), round(time.monotonic() - R.t_start, 1)
    S["errors"] = R.notes["errors"]
    R.save("summary.json", S)
    R.close()
    print(json.dumps({k: v for k, v in S.items() if k not in ("trials", "args", "errors")}, default=_jd)[:4000])
    if R.fake:
        R.fake["svc"].stop()
        R.fake["dep"].stop()
        R.fake["p1"].stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
