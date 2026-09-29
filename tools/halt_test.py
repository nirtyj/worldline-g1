"""B.1 live exit test (docs/M2.md §7.2, contract §3.10): halts through the halt lane mid-walk, mid-arm-stream,
during an arm_script IK, during chunk sessions, on a CarryLock hold, and with SONIC's free arms (M2b wave 2, body-fix
B-D1..B-D3).

    .venv/bin/python -m tools.halt_test [--walk 20] [--arm 5] [--script 20] [--chunk 10] [--carry 10] [--free-walks 3]
                                        [--out outputs/body_wave/halt-<ts>] [--port-offset N]

    --script N      halts 4-30 ms after an arm_script whose goal is out of reach (1.4 m above the pelvis: the longest
                    IK, 150-250 ms): the halt must not wait for it (B-D1). Nothing moves; the arms must stay free.
    --chunk N       halts mid-way through a synthetic chunk session (groot_arms' wire: mode chunk, 40-row chunks at
                    2.5 Hz, lead 0.15, hands closing to --hand-closure); 3 poisoned chunks (+0.4 rad elbow) after the
                    ack must be rejected and never reach the wire. Checked: the hand command after the halt = the
                    last hand target (B-D2), the waist on the wire does not step (B-low), the session ends canceled.
    --carry N       a CarryLock hold (a target stream closing the right hand to --carry-closure, ended hold_on_end
                    target), then N halt + resume cycles: the hand command on the wire is unchanged and CarryLock
                    stays engaged (B-D2).
    --free-walks N  with no arm op: N walks before and N after a halt + resume while standing; the arms must stay
                    SONIC's (no override on the wire) and swing as before (B-D3).

Runs against a standing M1 stack, only through BodyClient (halt = PUSH 5612, receipt = body.halted on 5611).
Per halt it records, from sources other than the body's own claims:
- receipt: client send -> body.halted received by the client's SUB thread (rtt_ms), and the body's handle_ms;
- ground truth (gt.pose 5601): speed at the halt, time until |v| < 0.05 m/s for 0.3 s, travel after the halt, pelvis_z
  min, fallen;
- what SONIC was sent (a read-only SUB on the SONIC input 5556, as the deploy sees it): planner mode after the halt,
  command{stop} count (must stay 0), the upper-body / hand override;
- arms (mid-arm-stream halts; g1_debug 5557): both palms (FK of the measured joints, pelvis frame) over the 2 s
  after the halt vs at the halt (drift), measured and commanded hand closure before/after (never opened).
Pass (per the B.1 exit and the wave-2 bars): every receipt < 30 ms, body handling p99 < 10 ms, upright, at rest
within 1.5 s, 0 falls; arm halts: arms held at the measured pose (drift reported), hand command = the last target;
free arms stay free. Writes trials.json, summary.json, raw.npz.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import threading
import time

import numpy as np
import zmq

from body import g1_kin as K
from body import joint_map as jm
from body.client import BodyClient
from body.config import ep, port_offset_from_env, ports as _ports
from body.wire import LocomotionMode, decode_command, decode_planner, wrap


class Tap:
    """Read-only taps: gt.pose (5601), g1_debug (5557), SONIC input (5556: planner + command, as the deploy sees it)."""

    def __init__(self, P: dict, host: str = "127.0.0.1"):
        self.P, self.host = P, host
        self.ctx = zmq.Context.instance()
        self.lock = threading.Lock()
        self.pose = collections.deque(maxlen=60000)     # (t_mono, x, y, yaw, vx, vy, pelvis_z, fallen)
        self.dbg = collections.deque(maxlen=60000)      # (t_mono, body_q29, left_hand_q7, right_hand_q7)
        self.plan = collections.deque(maxlen=60000)     # (t_mono, mode, upper17|None, left7|None, right7|None)
        self.commands = []                              # (t_mono, decoded command)
        self._running = True
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._pose, self._dbg, self._sonic)]
        for t in self.threads:
            t.start()

    def _sub(self, port, topics):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 2000)
        for t in topics:
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(ep(port, self.host))
        return s

    def _pose(self):
        import msgpack
        s = self._sub(self.P["p1_pose"], [b"gt.pose"])
        while self._running:
            if not s.poll(100):
                continue
            fr = s.recv_multipart()
            d = msgpack.unpackb(fr[-1], raw=False)
            v = d.get("base_lin_vel_w") or [0, 0, 0]
            p = d.get("base_pos") or [0, 0, 0]
            with self.lock:
                self.pose.append((time.monotonic(), p[0], p[1], float(d.get("yaw", 0.0)), v[0], v[1],
                                  float(d.get("pelvis_z", p[2])), bool(d.get("fallen", False))))
        s.close(0)

    def _dbg(self):
        import msgpack
        s = self._sub(self.P["sonic_debug"], [b"g1_debug"])
        while self._running:
            if not s.poll(100):
                continue
            raw = s.recv_multipart()[-1]
            if raw.startswith(b"g1_debug"):
                raw = raw[len(b"g1_debug"):]
            try:
                d = msgpack.unpackb(raw, raw=False)
            except Exception:
                continue
            q = d.get("body_q")
            if q is None or len(q) != 29:
                continue
            with self.lock:
                self.dbg.append((time.monotonic(), list(q), d.get("left_hand_q"), d.get("right_hand_q")))
        s.close(0)

    def _sonic(self):
        s = self._sub(self.P["sonic_in"], [b"planner", b"command"])
        while self._running:
            if not s.poll(100):
                continue
            msg = s.recv()
            now = time.monotonic()
            try:
                if msg.startswith(b"planner"):
                    p = decode_planner(msg)
                    with self.lock:
                        self.plan.append((now, p["mode"], p.get("upper_body_position"), p.get("left_hand_joints"),
                                          p.get("right_hand_joints")))
                elif msg.startswith(b"command"):
                    with self.lock:
                        self.commands.append((now, decode_command(msg)))
            except Exception:
                pass
        s.close(0)

    def window(self, name: str, t0: float, t1: float) -> list:
        with self.lock:
            return [r for r in getattr(self, name) if t0 <= r[0] <= t1]

    def last(self, name: str):
        with self.lock:
            d = getattr(self, name)
            return d[-1] if d else None

    def stop_count(self) -> int:
        with self.lock:
            return sum(1 for _, c in self.commands if c.get("stop"))

    def close(self):
        self._running = False
        for t in self.threads:
            t.join(timeout=1.0)


def rest_time(poses: list, t_halt: float, v_eps: float = 0.05, hold_s: float = 0.3) -> float | None:
    """Seconds from the halt until |v| < v_eps continuously for hold_s (GT), else None."""
    t_ok = None
    for r in poses:
        if r[0] < t_halt:
            continue
        if math.hypot(r[4], r[5]) < v_eps:
            if t_ok is None:
                t_ok = r[0]
            if r[0] - t_ok >= hold_s:
                return round(t_ok - t_halt, 3)
        else:
            t_ok = None
    return None


def rest_time_m1(poses: list, t_halt: float, v_eps: float = 0.05, n: int = 5) -> float | None:
    """The M1 E3 stop criterion (tools/m1_drive_test.py t_stop_mid_walk): the first of n consecutive gt.pose samples
    (50 Hz, 0.1 s) with |v| < v_eps, seconds after the halt."""
    a = [r for r in poses if r[0] >= t_halt]
    v = [math.hypot(r[4], r[5]) for r in a]
    for i in range(len(v) - n + 1):
        if all(x < v_eps for x in v[i:i + n]):
            return round(a[i][0] - t_halt, 3)
    return None


def palms(q29) -> dict:
    return K.points(K.named_from_q29(q29))


def drift_profile(dbg: list, t_h: float) -> dict:
    """Palm drift (FK of the measured arms, pelvis frame) after a halt: the max over the 2 s after the halt vs the
    pose at the halt, the drift at +0.25/0.5/1/2 s, and the max over +0.5..2 s vs the pose at +0.5 s (frozen after
    the arm's own lag has played out)."""
    rows = [r for r in dbg if t_h <= r[0] <= t_h + 2.05]
    if not rows:
        return {"arm_metrics": "no g1_debug"}
    P = [(r[0] - t_h, palms(r[1])) for r in rows]
    p0 = P[0][1]
    out = {"palm_drift_mm": {}, "palm_drift_at_mm": {}, "palm_drift_after_0p5s_mm": {}}
    for s in ("left", "right"):
        k = f"{s}_palm"
        d = [(t, float(np.linalg.norm(p[k] - p0[k])) * 1e3) for t, p in P]
        out["palm_drift_mm"][s] = round(max(x for _, x in d), 1)
        out["palm_drift_at_mm"][s] = {str(tt): round(min(d, key=lambda z: abs(z[0] - tt))[1], 1)
                                      for tt in (0.25, 0.5, 1.0, 2.0)}
        late = [(t, p) for t, p in P if t >= 0.5]
        if late:
            ref = late[0][1][k]
            out["palm_drift_after_0p5s_mm"][s] = round(max(float(np.linalg.norm(p[k] - ref)) for _, p in late) * 1e3, 1)
    return out


def pct(v: list, p: float):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def stats(v: list) -> dict:
    vv = [x for x in v if x is not None]
    return {"n": len(vv), "p50": pct(vv, 50), "p95": pct(vv, 95), "max": None if not vv else round(max(vv), 3),
            "min": None if not vv else round(min(vv), 3)}


class HaltTest:
    def __init__(self, a):
        off = a.port_offset if a.port_offset is not None else port_offset_from_env()
        self.a = a
        self.off = off
        self.script_warm = self.carry = self.free = None
        self.P = _ports(off)
        self.bc = BodyClient(port_offset=off).connect(15)
        self.tap = Tap(self.P)
        self.topics = []
        self.bc.add_topic_listener(lambda t, m: self.topics.append((time.monotonic(), t, m)))
        self.events = []
        self.bc.add_listener(lambda ev: self.events.append((time.monotonic(), ev)))
        self.epoch = int(a.epoch0)
        self.trials: list[dict] = []
        os.makedirs(a.out, exist_ok=True)

    def log(self, msg):
        print(f"[halt_test {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def pose(self):
        r = self.tap.last("pose")
        return None if r is None else {"t": r[0], "x": r[1], "y": r[2], "yaw": r[3], "v": math.hypot(r[4], r[5]),
                                       "pelvis_z": r[6], "fallen": r[7]}

    def resume(self):
        rep = self.bc.resume(self.epoch + 1)
        self.epoch += 2
        return rep

    # -- one halt, measured -------------------------------------------------------------------------
    def halt_and_measure(self, kind: str, extra: dict | None = None, watch_s: float = 3.0,
                         during=None) -> dict:
        ep_ = self.epoch
        p0 = self.pose()
        stops0 = self.tap.stop_count()
        t_send = time.monotonic()
        r = self.bc.halt(ep_, timeout_s=0.5)
        t_end = t_send + watch_s
        while time.monotonic() < t_end:
            if during is not None:
                during()
            time.sleep(0.02)
        poses = self.tap.window("pose", t_send - 0.5, t_end)
        after = [x for x in poses if x[0] >= t_send]
        plan_after = self.tap.window("plan", t_send, t_end)
        t_rest = rest_time_m1(poses, t_send)
        t_still = rest_time(poses, t_send)
        zmin = min((x[6] for x in after), default=None)
        fell = any(x[7] for x in after) or (zmin is not None and zmin < 0.55)
        p_end = after[-1] if after else None
        travel = None if p0 is None or p_end is None else round(math.hypot(p_end[1] - p0["x"], p_end[2] - p0["y"]), 3)
        v_late = [math.hypot(x[4], x[5]) for x in after if x[0] >= t_send + 1.5]
        idle_frac = None if not plan_after else round(
            sum(1 for x in plan_after if x[1] == LocomotionMode.IDLE) / len(plan_after), 4)
        first_idle = next((round((x[0] - t_send) * 1e3, 2) for x in plan_after if x[1] == LocomotionMode.IDLE), None)
        body = r.get("body") or {}
        rec = {"kind": kind, "epoch": ep_, "t_send_mono": t_send, "acked": r["acked"], "rtt_ms": r["rtt_ms"],
               "wait_ms": r["wait_ms"], "t_still_0p3_s": t_still,
               "handle_ms": body.get("handle_ms"), "body_kind": body.get("kind"),
               "arms_latched": body.get("arms_latched"), "arm_latch": body.get("arm"),
               "speed_at_halt": None if p0 is None else round(p0["v"], 3), "t_rest_s": t_rest,
               "at_rest_1p5": t_rest is not None and t_rest <= 1.5,
               "v_max_after_1p5s": None if not v_late else round(max(v_late), 3),
               "travel_after_halt_m": travel, "pelvis_z_min": None if zmin is None else round(zmin, 4),
               "fell": fell, "command_stop_sent": self.tap.stop_count() - stops0,
               "planner_first_idle_ms": first_idle, "planner_idle_frac": idle_frac,
               "pose_at_halt": p0, **(extra or {})}
        return rec

    # -- walks -------------------------------------------------------------------------------------
    def walk_trials(self, n: int) -> None:
        p = self.pose()
        h0 = p["yaw"]
        for i in range(n):
            heading = wrap(h0 + (math.pi if i % 2 else 0.0))
            ht = self.bc.turn_to(heading, timeout=40)
            if not ht.ok:
                self.log(f"turn_to failed: {ht.reason}")
            hw = self.bc.walk(vx=self.a.vx, duration_s=10.0, wait=False)
            delay = random.uniform(1.0, 2.0)
            t0 = time.monotonic()
            while time.monotonic() - t0 < delay:
                time.sleep(0.02)
            rec = self.halt_and_measure("walk", {"i": i, "walk_s_before_halt": round(delay, 2), "walk_op": hw.id})
            hw.wait(3.0)
            rec["walk_terminal"] = {"state": hw.state, "reason": hw.reason,
                                    "halt_epoch": (hw.result or {}).get("halt_epoch")}
            self.trials.append(rec)
            self.log(f"walk {i}: rtt {rec['rtt_ms']} ms (body {rec['handle_ms']} ms), v {rec['speed_at_halt']} -> "
                     f"rest {rec['t_rest_s']} s, travel {rec['travel_after_halt_m']} m, z_min {rec['pelvis_z_min']}, "
                     f"fell {rec['fell']}, op {hw.state}/{hw.reason}")
            rep = self.resume()
            if not rep.get("ok"):
                self.log(f"resume failed: {rep}")
            if rec["fell"]:
                self.log("FALL: stopping the walk trials")
                break

    # -- arms --------------------------------------------------------------------------------------
    def arm_trials(self, n: int) -> None:
        seed = K.named_from_mj17(self.bc_ref_mj17())
        qL, eL = K.ik_palm("left", (0.22, 0.20, -0.035), seed, seed)
        qR, eR = K.ik_palm("right", (0.22, -0.20, -0.035), qL, seed)
        base = {k: v for k, v in qR.items() if k not in jm.WAIST_JOINTS}
        self.log(f"arm pose: palms forward at table height (IK err {eL:.3f} / {eR:.3f} m), hands closure "
                 f"{self.a.hand_closure}")
        for i in range(n):
            stream = f"halt-test-arm-{i}-{int(time.time())}"
            arm = self.bc.arm_stream(stream=stream)
            ce = self.epoch                     # > the last halt epoch (resume bumped it): takes the latched hold over
            t0 = time.monotonic()
            closure = self.a.hand_closure

            def target(t):
                q = dict(base)
                q["right_shoulder_pitch_joint"] = base["right_shoulder_pitch_joint"] + 0.15 * math.sin(
                    2 * math.pi * 0.5 * t)
                q["left_shoulder_pitch_joint"] = base["left_shoulder_pitch_joint"] + 0.15 * math.sin(
                    2 * math.pi * 0.5 * t + 1.0)
                return q

            rejected = collections.Counter()

            def send():
                t = time.monotonic() - t0
                rep = arm.send(upper_body=target(t), left_hand=closure, right_hand=closure, control_epoch=ce)
                if not rep.get("ok"):
                    rejected[rep.get("error")] += 1

            ok0 = arm.send(upper_body=target(0.0), left_hand=closure, right_hand=closure, control_epoch=ce,
                           hold_s=1.0, blend_s=1.5)
            if not ok0.get("ok"):
                self.log(f"arm {i}: stream start rejected {ok0}")
            nxt = time.monotonic()
            while time.monotonic() - t0 < self.a.arm_lead_s:
                send()
                nxt += 0.02
                time.sleep(max(0.0, nxt - time.monotonic()))
            dbg_before = self.tap.last("dbg")
            plan_before = self.tap.last("plan")
            t_h = time.monotonic()
            state = {"nxt": time.monotonic()}

            def during():
                if time.monotonic() >= state["nxt"]:
                    send()                      # the client keeps streaming (it does not know about the halt yet)
                    state["nxt"] += 0.02

            rec = self.halt_and_measure("arm", {"i": i, "stream": stream, "arm_op": getattr(arm.handle, "id", None)},
                                        watch_s=self.a.arm_watch_s, during=during)
            rec["stream_rejected_after_halt"] = dict(rejected)
            rec.update(self.arm_metrics(t_h, dbg_before, plan_before))
            if arm.handle is not None:
                if not arm.handle.done():
                    try:
                        arm.handle.wait(2.0)
                    except TimeoutError:
                        pass
                rec["arm_terminal"] = {"state": arm.handle.state, "reason": arm.handle.reason,
                                       "ended_by": (arm.handle.result or {}).get("ended_by")}
            self.trials.append(rec)
            self.log(f"arm {i}: rtt {rec['rtt_ms']} ms, arms_latched {rec['arms_latched']}, palm drift 2 s "
                     f"L {rec.get('palm_drift_mm', {}).get('left')} R {rec.get('palm_drift_mm', {}).get('right')} mm, "
                     f"hand closure meas {rec.get('hand_closure_meas')}, cmd {rec.get('hand_closure_cmd')}, "
                     f"op {rec.get('arm_terminal')}, fell {rec['fell']}")
            rep = self.resume()
            if not rep.get("ok"):
                self.log(f"resume failed: {rep}")
            if rec["fell"]:
                break
        # give the arms back to SONIC (blend) and check the override is released
        self.bc.stop(arms=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0 and (self.bc.status().get("mux") or {}).get("upper") is not None:
            time.sleep(0.2)
        self.arms_released = (self.bc.status().get("mux") or {}).get("upper") is None

    # -- B-D1: halts during an arm_script IK ----------------------------------------------------------------------
    def script_trials(self, n: int) -> None:
        bc2 = BodyClient(port_offset=self.off).connect(15)
        try:
            p = self.pose()
            far = [p["x"] + 0.3 * math.cos(p["yaw"]), p["y"] + 0.3 * math.sin(p["yaw"]), p["pelvis_z"] + 1.4]
            t0 = time.perf_counter()
            warm = bc2.request("arm_script", {"phase": "grasp", "arm": "right", "target_w": far})
            self.script_warm = {"reply": warm.get("error") or warm.get("state"), "ms": round((time.perf_counter() - t0)
                                                                                            * 1e3, 1),
                                "ik_ms": (warm.get("data") or {}).get("ik_ms")}
            self.log(f"script warm-up: {self.script_warm}")
            for i in range(n):
                out = {}
                ep_ = self.epoch

                def go():
                    t1 = time.perf_counter()
                    out["t_req"] = time.monotonic()
                    out["rep"] = bc2.request("arm_script", {"phase": "grasp", "arm": "right", "target_w": far,
                                                            "control_epoch": ep_})
                    out["ms"] = round((time.perf_counter() - t1) * 1e3, 1)

                th = threading.Thread(target=go)
                th.start()
                time.sleep(random.uniform(0.004, 0.030))
                t_lead = time.monotonic()
                rec = self.halt_and_measure("script", {"i": i}, watch_s=1.0)
                th.join(10)
                r = out.get("rep") or {}
                rec.update({"script_reply": r.get("error") or r.get("state"), "script_ms": out.get("ms"),
                            "halt_after_request_ms": round((t_lead - out.get("t_req", t_lead)) * 1e3, 1),
                            "ik_ms": (r.get("data") or {}).get("ik_ms"),
                            "override_on_wire": any(x[2] is not None for x in self.tap.window(
                                "plan", rec["t_send_mono"], rec["t_send_mono"] + 1.0))})
                rec["overlapped"] = rec["script_reply"] == "halted"
                self.trials.append(rec)
                self.log(f"script {i}: rtt {rec['rtt_ms']} ms (body {rec['handle_ms']} ms), halt "
                         f"{rec['halt_after_request_ms']} ms after the request, script {rec['script_reply']} after "
                         f"{rec['script_ms']} ms, override {rec['override_on_wire']}")
                rep = self.resume()
                if not rep.get("ok"):
                    self.log(f"resume failed: {rep}")
        finally:
            bc2.close()

    # -- B-D1/B-D2/B-low: halts during chunk sessions ---------------------------------------------------------------
    def chunk_trials(self, n: int) -> None:
        k_el = jm.UPPER_BODY_MUJOCO_JOINTS.index("right_elbow_joint")
        k_sl = jm.UPPER_BODY_MUJOCO_JOINTS.index("left_shoulder_pitch_joint")
        k_sr = jm.UPPER_BODY_MUJOCO_JOINTS.index("right_shoulder_pitch_joint")
        for i in range(n):
            sid = f"halt-test-chunk-{i}-{int(time.time() * 1000) % 10 ** 8}"
            ce = self.epoch
            base = {"stream": sid, "session_id": sid, "execution_id": sid, "generation": 1, "control_epoch": ce,
                    "mode": "chunk"}
            q0 = self.bc_ref_mj17()
            closure = self.a.hand_closure
            t_start = time.monotonic()

            def rows(t0: float, T: int = 40, poison: float = 0.0):
                ub, lh, rh = [], [], []
                for k in range(T):
                    t = t0 + k * 0.02 - t_start
                    q = list(q0)
                    q[k_sl] += 0.15 * math.sin(2 * math.pi * 0.5 * t)
                    q[k_sr] += 0.15 * math.sin(2 * math.pi * 0.5 * t + 1.0)
                    q[k_el] += poison
                    ub.append(q)
                    c = min(closure, closure * max(0.0, t) / 1.0)
                    lh.append(jm.hand_closure("left", c))
                    rh.append(jm.hand_closure("right", c))
                return ub, lh, rh

            rep0 = self.bc.request("arm", {**base, "t_wall": time.time(), "hold_on_end": "measured", "lead_s": 0.15,
                                           "left_hand": [0.0] * 7, "right_hand": [0.0] * 7})
            if not rep0.get("ok"):
                self.log(f"chunk {i}: session start rejected {rep0}")
                self.resume()
                continue
            op_id = (rep0.get("data") or {}).get("id")
            seq = [0]
            sent_after_ack, rejected_after = [], collections.Counter()
            last_hand = {}

            def send_chunk(poison: float = 0.0, record_after: bool = False):
                d = self.tap.last("dbg")
                t0 = d[0] if d is not None else time.monotonic()
                ub, lh, rh = rows(t0, poison=poison)
                seq[0] += 1
                r = self.bc.request("arm", {**base, "t_wall": time.time(), "chunk": {
                    "seq": seq[0], "t0_mono": t0, "dt": 0.02, "order": "mj17", "upper_body": ub, "left_hand": lh,
                    "right_hand": rh, "inference_ms": 150.0}})
                if record_after:
                    sent_after_ack.append(round(time.monotonic(), 4))
                    if not r.get("ok"):
                        rejected_after[r.get("error")] += 1
                return r

            halt_at = random.uniform(1.6, 3.0)
            nxt = time.monotonic()
            while time.monotonic() - t_start < halt_at:
                send_chunk()
                nxt += 0.4
                time.sleep(max(0.0, nxt - time.monotonic()))
            plan_before = self.tap.last("plan")
            dbg_before = self.tap.last("dbg")
            rec = self.halt_and_measure("chunk", {"i": i, "session_id": sid, "arm_op": op_id}, watch_s=0.05)
            for _ in range(3):                             # in flight when the halt came: must never play
                send_chunk(poison=0.4, record_after=True)
                time.sleep(0.05)
            time.sleep(max(0.0, self.a.arm_watch_s - 0.2))
            t_h = rec["t_send_mono"]
            rec.update(self.arm_metrics(t_h, dbg_before, plan_before))
            plan_after = self.tap.window("plan", t_h, t_h + 1.0)
            ups = [x for x in plan_after if x[2] is not None]
            pb = plan_before[2] if plan_before is not None else None
            if ups and pb is not None:
                pb17 = jm.mj17_from_wire(pb)
                a17 = [jm.mj17_from_wire(x[2]) for x in ups]
                rec["waist_step_after_halt_rad"] = round(max(max(abs(q[k] - pb17[k]) for k in range(3)) for q in a17), 4)
                rec["poison_on_wire"] = any(q[k_el] > pb17[k_el] + 0.3 for q in a17)
                rec["hand_cmd_equal_last_target"] = all(
                    x[3] is not None and plan_before[3] is not None and
                    max(abs(u - v) for u, v in zip(x[3], plan_before[3])) < 1e-4 and
                    max(abs(u - v) for u, v in zip(x[4], plan_before[4])) < 1e-4 for x in ups)
            rec["chunks_sent_after_ack"] = len(sent_after_ack)
            rec["chunks_rejected_after_ack"] = dict(rejected_after)
            ev = self.find_terminal(op_id)
            rec["arm_terminal"] = None if ev is None else {"state": ev.get("state"),
                                                           "ended_by": (ev.get("data") or {}).get("ended_by"),
                                                           "hold": (ev.get("data") or {}).get("hold")}
            rec["arm_latch_applied"] = (self.bc.status().get("arm") or {}).get("latch")
            self.trials.append(rec)
            self.log(f"chunk {i}: rtt {rec['rtt_ms']} ms (body {rec['handle_ms']} ms), after the ack "
                     f"{rec['chunks_sent_after_ack']} sent / rejected {rec['chunks_rejected_after_ack']}, poison on "
                     f"wire {rec.get('poison_on_wire')}, waist step {rec.get('waist_step_after_halt_rad')} rad, hand "
                     f"cmd = last target {rec.get('hand_cmd_equal_last_target')}, op {rec['arm_terminal']}")
            rep = self.resume()
            if not rep.get("ok"):
                self.log(f"resume failed: {rep}")
            if rec["fell"]:
                break
        self.release_arms()

    # -- B-D2: CarryLock through repeated halts -------------------------------------------------------------------
    def carry_trials(self, n: int) -> None:
        stream = f"halt-test-carry-{int(time.time())}"
        ce = self.epoch
        for _ in range(40):                             # close the right hand on "an object" at SONIC's arm pose
            self.bc.request("arm", {"stream": stream, "right_hand": self.a.carry_closure, "left_hand": 0.0,
                                    "control_epoch": ce, "t_wall": time.time()})
            time.sleep(0.05)
        time.sleep(0.5)
        r = self.bc.request("arm", {"stream": stream, "end": True, "hold_on_end": "target", "control_epoch": ce})
        time.sleep(1.5)

        def carry_state():
            a = self.bc.status().get("arm") or {}
            c = a.get("carry") or {}
            return {"engaged": c.get("engaged"), "hold": (a.get("hold") or {}).get("kind"), "mode": a.get("mode"),
                    "closure": (a.get("hold") or {}).get("closure")}

        def wire_hands():
            x = self.tap.last("plan")
            return None if x is None or x[3] is None else (list(x[3]), list(x[4]))

        self.carry = {"end_reply": r.get("state") or r.get("error"), "before": carry_state(), "cycles": []}
        h0 = wire_hands()
        d0 = self.tap.last("dbg")
        self.carry["hand_cmd_closure_before"] = None if h0 is None else round(jm.hand_closure_of("right", h0[1]), 4)
        self.carry["hand_meas_closure_before"] = None if d0 is None or d0[3] is None else round(
            jm.hand_closure_of("right", d0[3]), 4)
        for i in range(n):
            rec = self.halt_and_measure("carry", {"i": i}, watch_s=0.6)
            latched = carry_state()
            h1 = wire_hands()
            self.resume()
            time.sleep(0.4)
            after = carry_state()
            h2 = wire_hands()
            same = h0 is not None and h1 is not None and h2 is not None and all(
                max(abs(u - v) for u, v in zip(a, b)) < 1e-5 for a, b in ((h0[0], h1[0]), (h0[1], h1[1]),
                                                                         (h0[0], h2[0]), (h0[1], h2[1])))
            rec.update({"carry_latched": latched, "carry_after_resume": after, "hand_cmd_unchanged": same,
                        "hand_cmd_closure": None if h2 is None else round(jm.hand_closure_of("right", h2[1]), 4)})
            self.trials.append(rec)
            self.carry["cycles"].append({"i": i, "engaged_latched": latched["engaged"],
                                         "engaged_after": after["engaged"], "hand_cmd_unchanged": same,
                                         "hand_cmd_closure": rec["hand_cmd_closure"]})
            self.log(f"carry {i}: rtt {rec['rtt_ms']} ms, CarryLock latched {latched['engaged']} / after resume "
                     f"{after['engaged']}, hand cmd unchanged {same} (closure {rec['hand_cmd_closure']})")
        d1 = self.tap.last("dbg")
        self.carry["hand_meas_closure_after"] = None if d1 is None or d1[3] is None else round(
            jm.hand_closure_of("right", d1[3]), 4)
        self.release_arms()

    # -- B-D3: free arms through a halt ----------------------------------------------------------------------------
    def free_walk_trials(self, n: int) -> None:
        self.release_arms()
        st = self.bc.status().get("arm") or {}
        self.free = {"arm_mode_before": st.get("mode"), "before": [], "after": []}

        def walk_once(label: str, i: int) -> dict:
            p = self.pose()
            heading = wrap(p["yaw"] + (math.pi if i % 2 else 0.0))
            self.bc.turn_to(heading, timeout=40)
            t0 = time.monotonic()
            hw = self.bc.walk(vx=self.a.vx, duration_s=3.0)
            t1 = time.monotonic()
            dbg = self.tap.window("dbg", t0 + 1.0, t1 - 0.3)
            plan = self.tap.window("plan", t0, t1)
            sp = [(r[1][jm.MJ["left_shoulder_pitch_joint"]], r[1][jm.MJ["right_shoulder_pitch_joint"]]) for r in dbg]
            swing = None if not sp else [round(float(np.ptp([x[k] for x in sp])), 4) for k in (0, 1)]
            out = {"i": i, "walk": hw.state, "shoulder_pitch_swing_rad": swing,
                   "override_frac": None if not plan else round(sum(1 for x in plan if x[2] is not None) / len(plan), 4)}
            self.log(f"free walk {label} {i}: {out}")
            return out

        for i in range(n):
            self.free["before"].append(walk_once("before", i))
        rec = self.halt_and_measure("free", {"i": 0}, watch_s=1.0)
        rec["arm_mode_latched"] = (self.bc.status().get("arm") or {}).get("mode")
        rec["override_on_wire_latched"] = any(x[2] is not None for x in self.tap.window(
            "plan", rec["t_send_mono"], rec["t_send_mono"] + 1.0))
        self.trials.append(rec)
        self.resume()
        time.sleep(0.5)
        self.free["arm_mode_after_resume"] = (self.bc.status().get("arm") or {}).get("mode")
        for i in range(n):
            self.free["after"].append(walk_once("after", i))
        self.free["halt"] = {k: rec.get(k) for k in ("rtt_ms", "handle_ms", "arms_latched", "arm_mode_latched",
                                                     "override_on_wire_latched")}

    def release_arms(self) -> None:
        """Give the arms back to SONIC (stop {arms}) and wait until the override is off the wire."""
        self.bc.stop(arms=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6.0 and (self.bc.status().get("mux") or {}).get("upper") is not None:
            time.sleep(0.2)

    def find_terminal(self, op_id, timeout: float = 2.0):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            for _, ev in list(self.events):
                if ev.get("id") == op_id and ev.get("state") in ("succeeded", "failed", "canceled"):
                    return ev
            time.sleep(0.05)
        return None

    def bc_ref_mj17(self):
        d = self.tap.last("dbg")
        return jm.mj17_from_mujoco(d[1]) if d is not None else list(jm.DEFAULT_ANGLES[12:29])

    def arm_metrics(self, t_h: float, dbg_before, plan_before) -> dict:
        dbg = self.tap.window("dbg", t_h, t_h + 2.2)
        if not dbg:
            return {"arm_metrics": "no g1_debug"}
        # the halt pose = the measured pose at the halt (first sample at/after the send)
        p_h = palms(dbg[0][1])
        out = drift_profile(dbg, t_h)
        # hands: measured closure before the halt and 2 s after; commanded (SONIC input) before and after
        last = dbg[-1]

        def clo(q7, side):
            return None if q7 is None or len(q7) != 7 else round(jm.hand_closure_of(side, q7), 3)

        out["hand_closure_meas"] = {s: [clo(dbg_before[2 if s == "left" else 3], s) if dbg_before else None,
                                        clo(last[2 if s == "left" else 3], s)] for s in ("left", "right")}
        plan_after = self.tap.window("plan", t_h + 1.8, t_h + 2.2)
        pa = plan_after[-1] if plan_after else None
        out["hand_closure_cmd"] = {s: [clo(plan_before[3 if s == "left" else 4], s) if plan_before else None,
                                       clo(pa[3 if s == "left" else 4], s) if pa else None] for s in ("left", "right")}
        # opened = the measured hand opened, or the command after the halt is below where the hand actually was
        # before it (a command below the pre-halt target but at the measured closure is the latch holding the
        # measured q, which does not open the hand)
        m, c = out["hand_closure_meas"], out["hand_closure_cmd"]
        out["hands_opened"] = any(
            (m[s][0] is not None and m[s][1] is not None and m[s][1] < m[s][0] - 0.05) or
            (m[s][0] is not None and c[s][1] is not None and c[s][1] < m[s][0] - 0.05) for s in ("left", "right"))
        out["hand_cmd_below_target"] = {s: c[s][0] is not None and c[s][1] is not None and c[s][1] < c[s][0] - 0.05
                                        for s in ("left", "right")}
        # measured palms at +2 s vs the commanded (latched) override at +2 s
        if pa is not None and pa[2] is not None:
            q29 = jm.mujoco_from_upper(pa[2], last[1])
            pc = palms(q29)
            pm = palms(last[1])
            out["palm_vs_latched_cmd_mm"] = {s: round(float(np.linalg.norm(pm[f"{s}_palm"] - pc[f"{s}_palm"])) * 1e3, 1)
                                             for s in ("left", "right")}
        return out

    # -- summary ------------------------------------------------------------------------------------
    def summary(self) -> dict:
        tr = self.trials
        walk = [t for t in tr if t["kind"] == "walk"]
        arm = [t for t in tr if t["kind"] == "arm"]
        script = [t for t in tr if t["kind"] == "script"]
        chunk = [t for t in tr if t["kind"] == "chunk"]
        carry = [t for t in tr if t["kind"] == "carry"]
        free = [t for t in tr if t["kind"] == "free"]
        hm = [t["handle_ms"] for t in tr if t["handle_ms"] is not None]
        s = {"n_walk": len(walk), "n_arm": len(arm), "n_script": len(script), "n_chunk": len(chunk),
             "n_carry": len(carry), "n_free": len(free),
             "rtt_ms": stats([t["rtt_ms"] for t in tr]), "handle_ms": stats([t["handle_ms"] for t in tr]),
             "handle_ms_p99": pct(hm, 99),
             "by_kind": {k: {"rtt_ms": stats([t["rtt_ms"] for t in tr if t["kind"] == k]),
                             "handle_ms": stats([t["handle_ms"] for t in tr if t["kind"] == k]),
                             "handle_ms_p99": pct([t["handle_ms"] for t in tr if t["kind"] == k and
                                                   t["handle_ms"] is not None], 99)}
                         for k in ("walk", "arm", "script", "chunk", "carry", "free") if any(t["kind"] == k for t in tr)},
             "script": {"warm": getattr(self, "script_warm", None),
                        "overlapped": sum(1 for t in script if t.get("overlapped")),
                        "replies": dict(collections.Counter(t.get("script_reply") for t in script)),
                        "script_ms": stats([t.get("script_ms") for t in script]),
                        "override_on_wire": sum(1 for t in script if t.get("override_on_wire"))},
             "chunk": {"arms_latched": sum(1 for t in chunk if t["arms_latched"]),
                       "chunks_sent_after_ack": sum(t.get("chunks_sent_after_ack", 0) for t in chunk),
                       "chunks_rejected_after_ack": dict(sum((collections.Counter(t.get("chunks_rejected_after_ack")
                                                                                  or {}) for t in chunk),
                                                             collections.Counter())),
                       "poison_on_wire": sum(1 for t in chunk if t.get("poison_on_wire")),
                       "waist_step_after_halt_rad": stats([t.get("waist_step_after_halt_rad") for t in chunk]),
                       "hand_cmd_equal_last_target": sum(1 for t in chunk if t.get("hand_cmd_equal_last_target")),
                       "canceled_halt": sum(1 for t in chunk if (t.get("arm_terminal") or {}).get("ended_by") == "halt"),
                       "palm_drift_mm": stats([max(t["palm_drift_mm"].values()) for t in chunk
                                               if "palm_drift_mm" in t])},
             "carry": getattr(self, "carry", None),
             "free": getattr(self, "free", None),
             "acked": sum(1 for t in tr if t["acked"]),
             "rtt_under_30ms": sum(1 for t in tr if t["rtt_ms"] is not None and t["rtt_ms"] < 30.0),
             "falls": sum(1 for t in tr if t["fell"]),
             "t_rest_s": stats([t["t_rest_s"] for t in walk]),
             "t_still_0p3_s": stats([t.get("t_still_0p3_s") for t in walk]),
             "at_rest_within_1p5s": sum(1 for t in walk if t["at_rest_1p5"]),
             "speed_at_halt": stats([t["speed_at_halt"] for t in walk]),
             "travel_after_halt_m": stats([t["travel_after_halt_m"] for t in walk]),
             "pelvis_z_min": min((t["pelvis_z_min"] for t in tr if t["pelvis_z_min"] is not None), default=None),
             "command_stop_sent": sum(t["command_stop_sent"] for t in tr),
             "planner_first_idle_ms": stats([t["planner_first_idle_ms"] for t in tr]),
             "walk_ops_canceled_halt": sum(1 for t in walk if (t.get("walk_terminal") or {}).get("state") == "canceled"
                                           and (t.get("walk_terminal") or {}).get("reason") == "halt"),
             "arms_latched": sum(1 for t in arm if t["arms_latched"]),
             "palm_drift_mm": stats([max(t["palm_drift_mm"].values()) for t in arm if "palm_drift_mm" in t]),
             "palm_drift_after_0p5s_mm": stats([max(t["palm_drift_after_0p5s_mm"].values()) for t in arm
                                                if t.get("palm_drift_after_0p5s_mm")]),
             "palm_vs_latched_cmd_mm": stats([max(t["palm_vs_latched_cmd_mm"].values()) for t in arm
                                              if "palm_vs_latched_cmd_mm" in t]),
             "hands_opened": sum(1 for t in arm if t.get("hands_opened")),
             "arm_ops_canceled_halt": sum(1 for t in arm if (t.get("arm_terminal") or {}).get("state") == "canceled"),
             "arms_released_after": getattr(self, "arms_released", None)}
        s["pass"] = {
            "receipt_under_30ms": s["rtt_under_30ms"] == len(tr) and s["acked"] == len(tr),
            "handle_p99_under_10ms": s["handle_ms_p99"] is not None and s["handle_ms_p99"] < 10.0,
            "zero_falls": s["falls"] == 0,
            "at_rest_1p5s": s["at_rest_within_1p5s"] == len(walk),
            "upright": s["pelvis_z_min"] is not None and s["pelvis_z_min"] >= 0.55,
            "no_command_stop": s["command_stop_sent"] == 0,
            "arms_latched": s["arms_latched"] == len(arm),
            "hands_not_opened": s["hands_opened"] == 0,
        }
        if script:
            s["pass"]["script_arms_free"] = s["script"]["override_on_wire"] == 0 and \
                not any(t["arms_latched"] for t in script)
        if chunk:
            c = s["chunk"]
            s["pass"]["chunk_latched_and_fenced"] = c["arms_latched"] == len(chunk) and c["poison_on_wire"] == 0 and \
                c["chunks_rejected_after_ack"].get("halted", 0) == c["chunks_sent_after_ack"] and \
                c["canceled_halt"] == len(chunk)
            s["pass"]["chunk_hand_cmd_is_last_target"] = c["hand_cmd_equal_last_target"] == len(chunk)
            s["pass"]["chunk_no_waist_step"] = (c["waist_step_after_halt_rad"]["max"] or 0.0) < 0.02
        if carry:
            cy = (getattr(self, "carry", None) or {}).get("cycles") or []
            s["pass"]["carrylock_kept"] = bool(cy) and all(x["engaged_latched"] and x["engaged_after"] and
                                                           x["hand_cmd_unchanged"] for x in cy)
        if free:
            f = getattr(self, "free", None) or {}
            s["pass"]["free_arms_stay_free"] = not any(t["arms_latched"] or t.get("override_on_wire_latched")
                                                       for t in free) and \
                all((w.get("override_frac") or 0.0) == 0.0 for w in f.get("after", [])) and \
                f.get("arm_mode_after_resume") == "off"
        s["all_pass"] = all(s["pass"].values())
        return s

    def save(self) -> dict:
        s = self.summary()
        with open(os.path.join(self.a.out, "trials.json"), "w") as f:
            json.dump(self.trials, f, indent=1, default=str)
        with open(os.path.join(self.a.out, "summary.json"), "w") as f:
            json.dump(s, f, indent=1, default=str)
        with open(os.path.join(self.a.out, "topics.jsonl"), "w") as f:
            for t, top, m in self.topics:
                f.write(json.dumps({"t_mono": t, "topic": top, **m}, default=str) + "\n")
        with open(os.path.join(self.a.out, "events.jsonl"), "w") as f:
            for t, ev in self.events:
                f.write(json.dumps({"t_mono": t, **ev}, default=str) + "\n")
        pose = np.array(list(self.tap.pose), dtype=float) if self.tap.pose else np.zeros((0, 8))
        dbg_t = np.array([r[0] for r in self.tap.dbg])
        dbg_q = np.array([r[1] for r in self.tap.dbg]) if self.tap.dbg else np.zeros((0, 29))
        np.savez_compressed(os.path.join(self.a.out, "raw.npz"), pose=pose, dbg_t=dbg_t, dbg_q=dbg_q)
        return s


def analyze(out: str) -> dict:
    """Recompute the ground-truth metrics of a finished run from raw.npz (+ trials.json), e.g. after a metric change.
    Runs before t_send_mono was recorded use the last gt.pose sample before the halt (<= 20 ms early)."""
    tr = json.load(open(os.path.join(out, "trials.json")))
    z = np.load(os.path.join(out, "raw.npz"))
    poses = [tuple(r) for r in z["pose"]]
    dbg = [(float(t), list(q), None, None) for t, q in zip(z["dbg_t"], z["dbg_q"])]
    for t in tr:
        th = t.get("t_send_mono") or (t.get("pose_at_halt") or {}).get("t")
        if th is None:
            continue
        w = [p for p in poses if th - 0.5 <= p[0] <= th + 3.0]
        t["t_rest_s"] = rest_time_m1(w, th)
        t["t_still_0p3_s"] = rest_time(w, th)
        t["at_rest_1p5"] = t["t_rest_s"] is not None and t["t_rest_s"] <= 1.5
        if t["kind"] == "arm":
            t.update(drift_profile(dbg, th))
            m, c = t.get("hand_closure_meas") or {}, t.get("hand_closure_cmd") or {}
            if m and c:
                t["hands_opened"] = any(
                    (m[s][0] is not None and m[s][1] is not None and m[s][1] < m[s][0] - 0.05) or
                    (m[s][0] is not None and c[s][1] is not None and c[s][1] < m[s][0] - 0.05)
                    for s in ("left", "right"))
                t["hand_cmd_below_target"] = {s: c[s][0] is not None and c[s][1] is not None and
                                              c[s][1] < c[s][0] - 0.05 for s in ("left", "right")}
    fake = argparse.Namespace(out=out)
    h = HaltTest.__new__(HaltTest)
    h.a, h.trials, h.arms_released = fake, tr, json.load(open(os.path.join(out, "summary.json"))).get(
        "arms_released_after")
    s = h.summary()
    s["reanalyzed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out, "trials.json"), "w") as f:
        json.dump(tr, f, indent=1, default=str)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1, default=str)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analyze", help="recompute the metrics of a finished run dir (raw.npz + trials.json)")
    ap.add_argument("--walk", type=int, default=20)
    ap.add_argument("--arm", type=int, default=5)
    ap.add_argument("--script", type=int, default=0, help="halts during an unreachable arm_script IK (B-D1)")
    ap.add_argument("--chunk", type=int, default=0, help="halts during synthetic chunk sessions")
    ap.add_argument("--carry", type=int, default=0, help="halt + resume cycles on a CarryLock hold (B-D2)")
    ap.add_argument("--free-walks", type=int, default=0, help="walks before/after a halt with free arms (B-D3)")
    ap.add_argument("--carry-closure", type=float, default=0.99)
    ap.add_argument("--vx", type=float, default=0.45)
    ap.add_argument("--epoch0", type=int, default=int(time.time()) % 100000 * 10)
    ap.add_argument("--hand-closure", type=float, default=0.6)
    ap.add_argument("--arm-lead-s", type=float, default=3.0, help="stream this long before the halt")
    ap.add_argument("--arm-watch-s", type=float, default=2.5)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=f"outputs/body_wave/halt-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args(argv)
    if a.analyze:
        s = analyze(a.analyze)
        print(json.dumps(s, indent=1, default=str))
        return 0 if s["all_pass"] else 1
    random.seed(a.seed)
    t = HaltTest(a)
    t0 = time.monotonic()
    while (t.tap.last("pose") is None or t.tap.last("dbg") is None or t.tap.last("plan") is None) \
            and time.monotonic() - t0 < 5.0:
        time.sleep(0.05)
    st = t.bc.status()
    if not st.get("in_control") or st.get("fault") or st.get("latched"):
        t.log(f"not ready: in_control {st.get('in_control')} fault {st.get('fault')} latched {st.get('latched')}")
        return 2
    t.log(f"start: mode {st.get('mode')}, pose {st.get('pose')}, epoch0 {t.epoch}")
    try:
        if a.free_walks:
            t.free_walk_trials(a.free_walks)
        if a.walk:
            t.walk_trials(a.walk)
        if a.script:
            t.script_trials(a.script)
        if a.chunk:
            t.chunk_trials(a.chunk)
        if a.carry:
            t.carry_trials(a.carry)
        if a.arm:
            t.arm_trials(a.arm)
    finally:
        s = t.save()
        t.tap.close()
        t.bc.close()
    t.log(f"summary: {json.dumps(s)}")
    t.log(f"wrote {a.out}")
    return 0 if s["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
