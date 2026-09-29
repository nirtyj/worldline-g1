"""B.1 live exit test (docs/M2.md §7.2, contract §3.10): halts through the halt lane mid-walk and mid-arm-stream.

    .venv/bin/python -m tools.halt_test [--walk 20] [--arm 5] [--out outputs/body_wave/halt-<ts>] [--port-offset N]

Runs against a standing M1 stack, only through BodyClient (halt = PUSH 5612, receipt = body.halted on 5611).
Per halt it records, from sources other than the body's own claims:
- receipt: client send -> body.halted received by the client's SUB thread (rtt_ms), and the body's handle_ms;
- ground truth (gt.pose 5601): speed at the halt, time until |v| < 0.05 m/s for 0.3 s, travel after the halt, pelvis_z
  min, fallen;
- what SONIC was sent (a read-only SUB on the SONIC input 5556, as the deploy sees it): planner mode after the halt,
  command{stop} count (must stay 0), the upper-body / hand override;
- arms (mid-arm-stream halts; g1_debug 5557): both palms (FK of the measured joints, pelvis frame) over the 2 s
  after the halt vs at the halt (drift), measured and commanded hand closure before/after (never opened).
Pass (per the B.1 exit): receipt < 30 ms, upright, at rest within 1.5 s, 0 falls; arm halts: arms frozen at the
measured pose (drift reported), hands not opened. Writes trials.json, summary.json, raw.npz.
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
        s = {"n_walk": len(walk), "n_arm": len(arm),
             "rtt_ms": stats([t["rtt_ms"] for t in tr]), "handle_ms": stats([t["handle_ms"] for t in tr]),
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
            "zero_falls": s["falls"] == 0,
            "at_rest_1p5s": s["at_rest_within_1p5s"] == len(walk),
            "upright": s["pelvis_z_min"] is not None and s["pelvis_z_min"] >= 0.55,
            "no_command_stop": s["command_stop_sent"] == 0,
            "arms_latched": s["arms_latched"] == len(arm),
            "hands_not_opened": s["hands_opened"] == 0,
        }
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
        if a.walk:
            t.walk_trials(a.walk)
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
