"""SONIC arm/hand tracking through the arm channel (op `arm`, body/arm.py), on the live M1 stack.

Recommendation (b) (owner decision 2026-09-29): GR00T outputs arm + hand joint targets, streamed into SONIC's planner
message (upper_body_position[17] + left/right_hand_joints[7]); SONIC keeps legs and balance. This tool measures how
well SONIC tracks such targets, standing and walking, and writes plots + metrics.json (docs/arm_tracking.md).

    # on the box, M1 stack up (body restarted with the arm channel):
    .venv/bin/python -m tools.arm_track_test --out outputs/arm_track/$(date +%Y%m%d-%H%M%S) --video
    .venv/bin/python -m tools.arm_track_test --analyze outputs/arm_track/<ts>          # re-run the analysis only
    python -m tools.arm_track_test --fake --quick --out /tmp/arm-fake                  # plumbing check (fakes)

Phases (--phases, default all): baseline, static, steps, sine, grasp, hands, walk.
  baseline  4 s without override: SONIC's own tracking of its reference arms (g1_debug body_q vs body_q_target)
  static    min-jerk (1.5 s) between poses, 4 s holds: SONIC reference, policy default angles, both palms forward at
            table height (0.75 m), right palm to a counter (0.865 m), arms up, default
  steps     0.3 rad steps (slew-limited to 6 rad/s) on right elbow / right shoulder pitch / left elbow: dead time,
            rise time, overshoot
  sine      +-0.3 rad (limits respected by moving the centre) at 0.2 / 0.5 / 1.0 Hz on each of the 7 arm joints,
            left arm, right arm, then both; 2.0 Hz on 3 right-arm joints (bandwidth bracket); vel=est A/B on the
            right elbow
  grasp     scripted reach-grasp-lift-place (IK keyframes, right arm) with Dex3 open/close
  hands     both Dex3 hands open -> 50 % -> closed -> open
  walk      SLOW_WALK 0.3 m/s (walk op, 8 s) and a 180 deg turn_to: baseline without override, then with a static
            carry pose, a 0.5 Hz sine, and the grasp script replay

Measurement: g1_debug (deploy PUB 5557, 50 Hz): body_q (lowstate as the deploy consumed it, MuJoCo order),
last_action (the policy's joint targets = rt/lowcmd q), body_q_target (SONIC's own reference, without the override),
left/right_hand_q (Dex3 state). gt.pose (P1 PUB 5601) for the pelvis. P1 `record` (200 Hz motor_q / q_target)
during the step block and P1 get_joint_state at the end of each static hold, as cross-checks. Commands are logged
at the client's send time, so latencies are end to end (client -> body -> mux -> deploy -> policy -> PD -> joint).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
from body.client import BodyClient  # noqa: E402
from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402
from body.wire import loads_any, split_topic, wrap, yaw_from_quat_wxyz  # noqa: E402

ARM = jm.ARM_JOINTS                                  # 14: left 7, right 7 (MuJoCo order)
ARM_MJ = np.array([jm.MJ[n] for n in ARM])            # indices into the 29-vector
SHORT = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll", "wr_pitch", "wr_yaw"]
SIDE_IDX = {"left": list(range(7)), "right": list(range(7, 14))}
DT = 0.02


def jidx(side: str, short: str) -> int:
    return SIDE_IDX[side][SHORT.index(short)]


def aname(i: int) -> str:
    return ("L_" if i < 7 else "R_") + SHORT[i % 7]


# ======================================================================================================
# capture
# ======================================================================================================
class Capture:
    """g1_debug + gt.pose subscribers; every sample is stamped with this process's monotonic clock."""

    def __init__(self, P: dict, ctx: zmq.Context):
        self.P, self.ctx = P, ctx
        self.lock = threading.Lock()
        self.dbg = {k: [] for k in ("t", "idx", "q", "act", "ref", "lh", "rh", "lha", "rha", "bquat")}
        self.gt = {k: [] for k in ("t", "x", "y", "z", "yaw", "quat", "pelvis_z", "fallen", "rtf", "t_sim")}
        self.running = True
        self.latest_dbg: dict | None = None
        self.threads = [threading.Thread(target=self._dbg_loop, daemon=True),
                        threading.Thread(target=self._gt_loop, daemon=True)]
        for th in self.threads:
            th.start()

    def _dbg_loop(self):
        import msgpack
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 1000)
        s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        s.connect(ep(self.P["sonic_debug"]))
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
            self.latest_dbg = d
            n7 = [math.nan] * 7
            with self.lock:
                D = self.dbg
                D["t"].append(t)
                D["idx"].append(int(d.get("index", -1)))
                D["q"].append(d["body_q"])
                D["act"].append(d.get("last_action") or [math.nan] * 29)
                D["ref"].append(d.get("body_q_target") or [math.nan] * 29)
                D["lh"].append(d.get("left_hand_q") or n7)
                D["rh"].append(d.get("right_hand_q") or n7)
                D["lha"].append(d.get("last_left_hand_action") or n7)
                D["rha"].append(d.get("last_right_hand_action") or n7)
                D["bquat"].append(d.get("base_quat") or [1, 0, 0, 0])
        s.close(0)

    def _gt_loop(self):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 1000)
        s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        s.connect(ep(self.P["p1_pose"]))
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
            quat = d.get("base_quat_wxyz") or [1, 0, 0, 0]
            with self.lock:
                G = self.gt
                G["t"].append(t)
                G["x"].append(float(pos[0]))
                G["y"].append(float(pos[1]))
                G["z"].append(float(pos[2]))
                G["yaw"].append(float(d["yaw"]) if d.get("yaw") is not None else yaw_from_quat_wxyz(quat))
                G["quat"].append([float(v) for v in quat])
                G["pelvis_z"].append(float(d.get("pelvis_z") if d.get("pelvis_z") is not None else pos[2]))
                G["fallen"].append(bool(d.get("fallen", False)))
                G["rtf"].append(float(d["rtf"]) if d.get("rtf") is not None else math.nan)
                G["t_sim"].append(float(d.get("t_sim") or 0.0))
        s.close(0)

    def ref_q29(self) -> list[float] | None:
        d = self.latest_dbg
        return None if d is None else list(d.get("body_q_target") or d["body_q"])

    def q29(self) -> list[float] | None:
        d = self.latest_dbg
        return None if d is None else list(d["body_q"])

    def last_gt(self) -> dict | None:
        with self.lock:
            if not self.gt["t"]:
                return None
            return {k: v[-1] for k, v in self.gt.items()}

    def arrays(self) -> dict:
        with self.lock:
            out = {f"dbg_{k}": np.asarray(v, dtype=float) for k, v in self.dbg.items()}
            out.update({f"gt_{k}": np.asarray(v, dtype=float) for k, v in self.gt.items()})
        return out

    def stop(self):
        self.running = False
        for th in self.threads:
            th.join(1.0)


# ======================================================================================================
# trajectories
# ======================================================================================================
def minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def clamp14(v: np.ndarray, margin: float = 0.03) -> np.ndarray:
    lo = np.array([jm.JOINT_LIMITS[n][0] + margin for n in ARM])
    hi = np.array([jm.JOINT_LIMITS[n][1] - margin for n in ARM])
    return np.clip(v, lo, hi)


def poses(ref14: np.ndarray, ref_named: dict) -> dict[str, np.ndarray]:
    """Named arm poses (14, ARM order). IK in the pelvis frame; the waist is taken from SONIC's reference."""
    default = np.array([jm.DEFAULT_ANGLES[jm.MJ[n]] for n in ARM])
    out = {"sonic_ref": ref14.copy(), "default": default}
    seed = dict(ref_named)
    q = dict(seed)
    # both palms at table height (0.75 m world = pelvis z 0.785 - 0.035), 22 cm forward, 20 cm to the side
    qL, eL = K.ik_palm("left", (0.22, 0.20, -0.035), q, seed)
    qR, eR = K.ik_palm("right", (0.22, -0.20, -0.035), qL, seed)
    out["reach_table"] = np.array([qR[n] for n in ARM])
    # right palm down-forward to a counter top (0.865 m world), left arm at default
    qC, eC = K.ik_palm("right", (0.34, -0.20, 0.08), seed, seed)
    rc = default.copy()
    rc[7:] = [qC[n] for n in ARM[7:]]
    out["counter_right"] = rc
    up = default.copy()
    for side, sgn in (("left", 1), ("right", -1)):
        up[jidx(side, "sh_pitch")] = -2.5
        up[jidx(side, "sh_roll")] = 0.3 * sgn
        up[jidx(side, "sh_yaw")] = 0.0
        up[jidx(side, "elbow")] = 1.2
        for w in ("wr_roll", "wr_pitch", "wr_yaw"):
            up[jidx(side, w)] = 0.0
    out["arms_up"] = up
    out = {k: clamp14(v) for k, v in out.items()}
    out["_ik_err_m"] = {"reach_table_left": eL, "reach_table_right": eR, "counter_right": eC}
    return out


def sine_center(default14: np.ndarray) -> np.ndarray:
    c = default14.copy()
    c[jidx("left", "sh_roll")] = 0.35     # arms away from the torso so +-0.3 rad roll never hits it
    c[jidx("right", "sh_roll")] = -0.35
    return c


# ======================================================================================================
# the run
# ======================================================================================================
class Run:
    def __init__(self, a):
        self.a = a
        self.off = port_offset_from_env() if a.port_offset is None else a.port_offset
        self.P = _ports(self.off)
        self.ctx = zmq.Context.instance()
        self.out = a.out
        os.makedirs(self.out, exist_ok=True)
        self.cmd = {"t": [], "tw": [], "q": [], "q_sent": [], "lh": [], "rh": [], "seg": [], "ok": []}
        self.segments: list[dict] = []
        self.events: list[dict] = []
        self.notes: dict = {"errors": []}
        self.video_procs: list = []
        self.p1_records: list[dict] = []
        self.js_snapshots: list[dict] = []
        self.q = 1.0 if not a.quick else 0.25    # duration scale in --quick mode
        self.lead = float(a.lead)

    # -- setup ---------------------------------------------------------------------------------------------
    def start(self):
        self.cap = Capture(self.P, self.ctx)
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0, ctx=self.ctx)
        self.bc = BodyClient(port_offset=self.off, ctx=self.ctx).connect(20)
        self.bc.add_listener(lambda ev: self.events.append({"t": time.monotonic(), **ev}))
        t0 = time.monotonic()
        while (self.cap.latest_dbg is None or self.cap.last_gt() is None) and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        st = self.bc.status()
        self.notes["status_start"] = {k: st.get(k) for k in ("fault", "in_control", "active", "pose", "gt_pose",
                                                              "arm")}
        self.notes["p1_stats_start"] = self.p1.try_call("get_stats")
        self.notes["load_start"] = os.getloadavg()
        if not st.get("in_control"):
            h = self.bc.stand(timeout=120)
            if not h.ok:
                raise RuntimeError(f"stand failed: {h.result}")
        if st.get("fault"):
            raise RuntimeError(f"body fault {st['fault']}: reset the robot first")
        ref29 = self.cap.ref_q29()
        self.ref14 = np.array([ref29[i] for i in ARM_MJ])
        self.P_ = poses(self.ref14, K.named_from_q29(ref29))
        self.notes["ik_err_m"] = self.P_.pop("_ik_err_m")
        self.notes["poses"] = {k: [round(float(x), 4) for x in v] for k, v in self.P_.items()}
        self.notes["palm_targets_pelvis"] = {"reach_table": [0.22, 0.20, -0.035], "counter_right": [0.34, -0.20, 0.08]}
        self.arm = None
        self.cur = self.ref14.copy()
        self.hands = {"left": None, "right": None}

    # -- streaming -----------------------------------------------------------------------------------------
    def begin_stream(self, **kw):
        # always explicit, so a run records exactly what it measured (the body's default is servo_ki 2.0)
        kw = {"servo_ki": self.a.servo_ki, "servo_delay_s": self.a.servo_delay, "servo_max": self.a.servo_max, **kw}
        self.arm = self.bc.arm_stream(**kw)
        self.cur = np.array([self.cap.ref_q29()[i] for i in ARM_MJ])
        self.hands = {"left": None, "right": None}

    def end_stream(self, wait_s: float = 6.0):
        if self.arm is None:
            return
        try:
            self.arm.end()
            if self.arm.handle is not None:
                self.arm.handle.wait(wait_s)
        except Exception as e:
            self.notes["errors"].append(f"end_stream: {e!r}")
        self.arm = None

    def seg(self, name: str, kind: str, **meta) -> dict:
        s = {"name": name, "kind": kind, "t0": time.monotonic(), "t1": None, **meta}
        self.segments.append(s)
        print(f"[arm_track] {name}", flush=True)
        return s

    def close_seg(self, s: dict):
        s["t1"] = time.monotonic()
        g = self.cap.last_gt()
        if g is not None and (g["fallen"] or g["pelvis_z"] < 0.55):
            raise RuntimeError(f"robot fell during {s['name']}")

    def stream(self, fn, dur: float, s: dict | None = None, until=None, extra_args: dict | None = None):
        """fn(t) -> (arms14 or None, left_hand or None, right_hand or None) at 50 Hz for dur s (or until
        until() is true)."""
        t0 = time.monotonic()
        k = 0
        segi = len(self.segments) - 1
        first = True
        while True:
            now = time.monotonic()
            t = now - t0
            if (dur is not None and t > dur) or (until is not None and until()):
                break
            arms, lh, rh = fn(t)
            if arms is not None:
                self.cur = np.asarray(arms, float)
            if lh is not None:
                self.hands["left"] = list(lh)
            if rh is not None:
                self.hands["right"] = list(rh)
            sent = self.cur
            if self.lead > 0:
                # client-side lead: a GR00T chunk is a known future trajectory, so send the target `lead` s ahead
                # (errors are still computed against the desired value at t)
                a2, _, _ = fn(t + self.lead)
                sent = self.cur if a2 is None else np.asarray(a2, float)
            args = dict(extra_args or {}) if first else {}
            first = False
            rep = self.arm.send(upper_body={n: float(v) for n, v in zip(ARM, sent)},
                                left_hand=self.hands["left"], right_hand=self.hands["right"], **args)
            C = self.cmd
            C["t"].append(time.monotonic())
            C["tw"].append(time.time())
            C["q"].append(self.cur.copy())
            C["q_sent"].append(np.array(sent, float))
            C["lh"].append(self.hands["left"] if self.hands["left"] is not None else [math.nan] * 7)
            C["rh"].append(self.hands["right"] if self.hands["right"] is not None else [math.nan] * 7)
            C["seg"].append(segi)
            C["ok"].append(bool(rep.get("ok")))
            if not rep.get("ok"):
                self.notes["errors"].append(f"arm send rejected: {rep}")
                if rep.get("error") in ("not_standing", "arm_preempted") or str(rep.get("error", "")).startswith(
                        "fault"):
                    raise RuntimeError(f"arm stream rejected: {rep}")
            k += 1
            nxt = t0 + k * DT
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            elif d < -0.1:
                k = int((time.monotonic() - t0) / DT)

    def move_to(self, target14, T: float = 1.5, lh=None, rh=None):
        p0 = self.cur.copy()
        p1 = np.asarray(target14, float)
        self.stream(lambda t: (p0 + (p1 - p0) * minjerk(t / T), lh, rh), T)

    def hold(self, T: float, lh=None, rh=None):
        p = self.cur.copy()
        self.stream(lambda t: (p, lh, rh), T)

    # -- recorder ------------------------------------------------------------------------------------------
    def video(self, label: str, dur: float):
        if not self.a.video:
            return
        py = os.path.join(ROOT, "viz/.venv/bin/python")
        if not os.path.exists(py):
            self.notes["errors"].append("viz/.venv missing: no video")
            return
        vdir = os.path.join(self.out, "video")
        os.makedirs(vdir, exist_ok=True)
        cmd = [py, os.path.join(ROOT, "viz/recorder.py"), "--duration", f"{dur:.0f}", "--label", f"arm-{label}",
               "--out", vdir, "--port-offset", str(self.off)]
        logf = open(os.path.join(vdir, f"recorder-{label}.log"), "w")
        self.video_procs.append(subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT))
        # recorder start-up (subscriptions) before the motion; keep an active arm stream alive meanwhile (a plain
        # sleep here tripped the 0.3 s watchdog in the first runs: hold, then resume)
        if self.arm is not None:
            self.hold(1.5)
        else:
            time.sleep(1.5)

    # -- phases ---------------------------------------------------------------------------------------------
    def ph_baseline(self):
        self.end_stream()
        s = self.seg("baseline_no_override", "baseline")
        time.sleep(4.0 * self.q)
        self.close_seg(s)

    def ph_static(self):
        P = self.P_
        self.video("static", 45 * self.q + 8)
        self.begin_stream()
        self.hold(0.5)
        for name in ("sonic_ref", "default", "reach_table", "counter_right", "arms_up", "default"):
            s = self.seg(f"static_{name}", "move", pose=name)
            self.move_to(P[name], 1.5 * max(self.q, 0.6))
            self.close_seg(s)
            s = self.seg(f"hold_{name}", "static", pose=name, target=P[name].tolist())
            self.hold(4.0 * self.q)
            self.close_seg(s)
            self.js_snapshots.append({"seg": len(self.segments) - 1, "t": time.monotonic(),
                                      "rep": self.p1.try_call("get_joint_state")})

    def ph_steps(self):
        base = self.P_["default"]
        if self.arm is None:
            self.begin_stream()
        self.move_to(base, 1.5)
        rec = self.p1.try_call("record", on=True, path=os.path.join(os.path.abspath(self.out), "p1_steps.npz"))
        for side, j, d in (("right", "elbow", 0.3), ("right", "sh_pitch", -0.3), ("left", "elbow", 0.3)):
            i = jidx(side, j)
            for sgn in (1, -1):
                tgt = self.cur.copy()
                tgt[i] += sgn * d
                s = self.seg(f"step_{side}_{j}_{'+' if sgn > 0 else '-'}", "step", joint=i, delta=sgn * d,
                             target=tgt.tolist())
                self.stream(lambda t, tgt=tgt: (tgt, None, None), 2.5 * max(self.q, 0.6))
                self.close_seg(s)
        rep = self.p1.try_call("record", on=False)
        self.p1_records.append({"block": "steps", "on": rec, "off": rep})

    def _sine(self, center: np.ndarray, idx: list[int], f: float, amp: float, label: str, fade: float = 1.0,
              extra_args=None, kind="sine", **meta):
        ncyc = {0.2: 2, 0.5: 4, 1.0: 6, 2.0: 8}.get(f, 4)
        if self.a.quick:
            ncyc = max(1, ncyc // 2)
        c = center.copy()
        for i in idx:   # move the centre so centre +- amp stays inside the limits (never clip the sine)
            lo, hi = jm.JOINT_LIMITS[ARM[i]]
            c[i] = min(max(c[i], lo + 0.05 + amp), hi - 0.05 - amp)
        if np.max(np.abs(self.cur - c)) > 1e-3:
            self.move_to(c, 1.0)
        T = fade + ncyc / f + fade

        def fn(t):
            if t < fade:
                env = minjerk(t / fade)
            elif t > T - fade:
                env = minjerk((T - t) / fade)
            else:
                env = 1.0
            v = c.copy()
            v[idx] += env * amp * math.sin(2 * math.pi * f * t)
            return v, None, None

        s = self.seg(label, kind, joints=idx, f=f, amp=amp, center=c.tolist(), fade=fade, n_cycles=ncyc, **meta)
        s["t_sine0"] = time.monotonic()
        s["win"] = [s["t_sine0"] + fade + 0.3, s["t_sine0"] + fade + ncyc / f]
        self.stream(fn, T, extra_args=extra_args)
        self.close_seg(s)

    def ph_sine(self):
        if self.arm is None:
            self.begin_stream()
        center = sine_center(self.P_["default"])
        self.move_to(center, 1.5)
        freqs = tuple(float(f) for f in self.a.sine_freqs.split(","))
        joints = SHORT if not self.a.quick else ["sh_pitch", "elbow"]
        first = True
        for side in self.a.sine_sides.split(","):
            for j in joints:
                idx = [jidx("left", j), jidx("right", j)] if side == "both" else [jidx(side, j)]
                for f in freqs:
                    if first:
                        self.video("sine_example", (10 + 8 + 6 + 9) * self.q + 6)
                        first = False
                    self._sine(center, idx, f, 0.3, f"sine_{side}_{j}_{f:g}Hz", side=side, joint=j)
        if not self.a.sine_extras:
            return
        for j in ("sh_pitch", "elbow", "wr_pitch"):
            self._sine(center, [jidx("right", j)], 2.0, 0.3, f"sine_right_{j}_2Hz", side="right", joint=j)
        # velocity feed-forward A/B (vel=est sends d(target)/dt as upper_body_velocity)
        for f in (0.5, 1.0):
            self._sine(center, [jidx("right", "elbow")], f, 0.3, f"sineVel_right_elbow_{f:g}Hz",
                       extra_args={"vel": "est"}, kind="sine_vel", side="right", joint="elbow")
        self.arm.send(vel="zero")

    def grasp_script(self, side: str = "right"):
        """Keyframes (palm in the pelvis frame): pregrasp above, approach, close, lift, hold, place, open, retreat."""
        ref_named = K.named_from_q29(self.cap.q29())
        seed = dict(ref_named)
        for n, v in zip(ARM, self.P_["default"]):
            seed[n] = v
        y = -0.20 if side == "right" else 0.20
        kf = [("pregrasp", (0.30, y, 0.08), 0.0, 1.5), ("approach", (0.32, y, 0.00), 0.0, 1.2),
              ("close", (0.32, y, 0.00), 1.0, 1.0), ("lift", (0.28, y, 0.15), 1.0, 1.5),
              ("hold", (0.28, y, 0.15), 1.0, 1.5), ("place", (0.32, y, 0.00), 1.0, 1.5),
              ("open", (0.32, y, 0.00), 0.0, 0.8), ("retreat", (0.26, y, 0.08), 0.0, 1.0)]
        out, q = [], seed
        for name, p, closure, T in kf:
            q, err = K.ik_palm(side, p, q, seed)
            v = self.cur.copy()
            v[SIDE_IDX[side]] = [q[n] for n in ARM[SIDE_IDX[side][0]:SIDE_IDX[side][-1] + 1]]
            out.append((name, clamp14(v), closure, T, p, err))
        return out

    def play_grasp(self, side: str = "right", label: str = "grasp", until=None, walking: bool = False):
        kf = self.grasp_script(side)
        self.notes.setdefault("grasp_keyframes", [{"name": n, "palm_target": list(p), "ik_err_m": round(e, 4),
                                                   "closure": c, "T": T} for n, _, c, T, p, e in kf])
        p0 = self.cur.copy()
        h0 = jm.hand_closure_of(side, self.hands[side]) if self.hands[side] is not None else 0.0
        # one continuous piecewise min-jerk trajectory (like a GR00T chunk), so a client lead works across keyframes
        pts, t_acc = [(0.0, p0, h0)], 0.0
        for name, v, closure, T, p, e in kf:
            t_acc += T * max(self.q, 0.6)
            pts.append((t_acc, v.copy(), closure))

        def fn(t):
            if t >= pts[-1][0]:
                a14, c = pts[-1][1], pts[-1][2]
            else:
                k = next(i for i in range(1, len(pts)) if t < pts[i][0])
                (ta, va, ca), (tb, vb, cb) = pts[k - 1], pts[k]
                u = minjerk((t - ta) / (tb - ta))
                a14, c = va + (vb - va) * u, ca + (cb - ca) * u
            hand = jm.hand_closure(side, c)
            return a14, (hand if side == "left" else None), (hand if side == "right" else None)

        s = self.seg(label, "grasp", side=side, walking=walking)
        self.stream(fn, pts[-1][0], until=until)
        self.close_seg(s)

    def ph_grasp(self):
        if self.arm is None:
            self.begin_stream()
        self.video("grasp", 16 * self.q + 12)
        self.move_to(self.P_["default"], 1.5, rh=jm.hand_closure("right", 0.0))
        self.play_grasp("right")
        self.move_to(self.P_["default"], 1.5)

    def ph_hands(self):
        if self.arm is None:
            self.begin_stream()
        for frac in (0.0, 0.5, 1.0, 0.0):
            s = self.seg(f"hands_{frac:g}", "hands", closure=frac)
            L, R = jm.hand_closure("left", frac), jm.hand_closure("right", frac)
            self.stream(lambda t, L=L, R=R: (None, L, R), 2.5 * max(self.q, 0.6))
            self.close_seg(s)
            self.js_snapshots.append({"seg": len(self.segments) - 1, "t": time.monotonic(),
                                      "rep": self.p1.try_call("get_joint_state")})

    def _walk_leg(self, name: str, arm_fn, dur: float = 8.0, vx: float = 0.3):
        s = self.seg(name, "walk", vx=vx, duration_s=dur, arms=arm_fn is not None)
        h = self.bc.walk(vx=vx, duration_s=dur * self.q, wait=False)
        if arm_fn is None:
            h.wait(dur + 30)
        else:
            self.stream(arm_fn, None, until=h.done)
        s["op"] = h.summary()
        self.close_seg(s)
        return h

    def _turn(self, name: str, arm_fn):
        g = self.cap.last_gt()
        s = self.seg(name, "turn", arms=arm_fn is not None)
        h = self.bc.turn_to(wrap(g["yaw"] + math.pi), wait=False)
        if arm_fn is None:
            h.wait(60)
        else:
            self.stream(arm_fn, None, until=h.done)
        s["op"] = h.summary()
        self.close_seg(s)
        return h

    def ph_walk(self):
        self.end_stream()
        info = self.p1.try_call("get_scene_info") or {}
        sp = info.get("spawn") or {}
        g = self.cap.last_gt()
        self.notes["walk_start"] = {"pose": {k: g[k] for k in ("x", "y", "yaw")}, "spawn": sp}
        if sp and (math.hypot(g["x"] - sp["x"], g["y"] - sp["y"]) > 0.6 or
                   abs(wrap(g["yaw"] - sp["yaw"])) > math.radians(25)):
            h = self.bc.go_to(sp["x"], sp["y"], yaw=sp["yaw"], timeout_s=90)
            self.notes["walk_start"]["go_to_spawn"] = h.summary()
        self.video("walk", 110 * self.q + 10)
        # baseline: no arm override
        self._walk_leg("walk_fwd_no_override", None)
        self._turn("turn_no_override", None)
        # with arms: carry pose (both palms forward at table height), hands closed halfway
        self.begin_stream()
        self.move_to(self.P_["reach_table"], 1.5, lh=jm.hand_closure("left", 0.5), rh=jm.hand_closure("right", 0.5))
        carry = self.cur.copy()
        s = self.seg("pre_walk_carry_hold", "static", pose="reach_table", target=carry.tolist())
        self.hold(2.0)
        self.close_seg(s)
        self._walk_leg("walk_back_carry", lambda t: (carry, None, None))
        s = self.seg("post_walk_carry_hold", "static", pose="reach_table", target=carry.tolist())
        self.hold(2.0)
        self.close_seg(s)
        center = sine_center(self.P_["default"])
        self.move_to(center, 1.5)
        idx = [jidx("right", "sh_pitch"), jidx("left", "elbow")]

        def sine_fn(t, t_start=[None]):
            v = center.copy()
            env = minjerk(t / 1.0)
            v[idx] += env * 0.3 * math.sin(2 * math.pi * 0.5 * t)
            return v, None, None

        t_turn0 = time.monotonic()

        def from_now():   # continuous sine phase across the turn and the walk; uses the stream's t (lead works)
            off = time.monotonic() - t_turn0
            return lambda t: sine_fn(t + off)

        self._turn("turn_sine", from_now())
        self.segments[-1].update({"joints": idx, "f": 0.5, "amp": 0.3, "t_sine0": t_turn0,
                                  "win": [t_turn0 + 1.3, self.segments[-1]["t1"]]})
        self._walk_leg("walk_fwd_sine", from_now())
        self.segments[-1].update({"joints": idx, "f": 0.5, "amp": 0.3, "t_sine0": t_turn0,
                                  "win": [self.segments[-1]["t0"], self.segments[-1]["t1"]]})
        self.move_to(self.P_["default"], 1.0)
        self._turn("turn_default_pose", lambda t: (self.P_["default"], None, None))
        # walk back while replaying the grasp script
        h = self.bc.walk(vx=0.3, duration_s=8.0 * self.q, wait=False)
        self.play_grasp("right", label="walk_back_grasp_script", walking=True)
        self.hold(0.1)
        self.stream(lambda t: (self.cur, None, None), None, until=h.done)
        self.segments[-1]["op"] = h.summary()
        self.move_to(self.P_["default"], 1.0, rh=jm.hand_closure("right", 0.0))
        self.end_stream()

    # -- main -----------------------------------------------------------------------------------------------
    def run(self, phases: list[str]) -> int:
        self.start()
        self.notes["phases"] = phases
        self.notes["config"] = {"servo_ki": self.a.servo_ki, "servo_delay_s": self.a.servo_delay,
                                "servo_max": self.a.servo_max, "lead_s": self.lead, "sine_freqs": self.a.sine_freqs,
                                "sine_sides": self.a.sine_sides, "chase": self.a.chase}
        self.viz_prev = None
        if self.a.chase:
            st = self.p1.try_call("viz_stats") or {}
            self.viz_prev = st.get("level")
            self.notes["viz_level_set"] = self.p1.try_call("viz_level", level="low")
            time.sleep(3.0)
        self.notes["t_start_wall"] = time.time()
        rc = 0
        try:
            for ph in phases:
                getattr(self, f"ph_{ph}")()
            self.end_stream()
            s = self.seg("final_no_override", "baseline")
            time.sleep(2.0)
            self.close_seg(s)
        except Exception as e:
            rc = 1
            self.notes["errors"].append(f"ABORTED: {e!r}\n{traceback.format_exc()}")
            print(f"[arm_track] ABORTED: {e!r}", flush=True)
            try:
                self.end_stream()
                self.bc.stop(arms=True, wait=False)
            except Exception:
                pass
        finally:
            self.notes["t_end_wall"] = time.time()
            if self.a.chase and self.viz_prev is not None:
                for p in self.video_procs:
                    try:
                        p.wait(timeout=150)
                    except Exception:
                        pass
                self.notes["viz_level_restored"] = self.p1.try_call("viz_level", level=self.viz_prev)
            try:
                self.notes["status_end"] = {k: v for k, v in self.bc.status().items()
                                            if k in ("fault", "in_control", "pose", "gt_pose", "arm", "mux")}
            except Exception as e:
                self.notes["status_end"] = repr(e)
            self.notes["p1_stats_end"] = self.p1.try_call("get_stats")
            self.notes["load_end"] = os.getloadavg()
            time.sleep(0.5)
            self.cap.stop()
            self.save()
            for p in self.video_procs:
                try:
                    p.wait(timeout=120)
                except Exception:
                    p.kill()
            self.bc.close()
        return rc

    def save(self):
        arr = self.cap.arrays()
        C = self.cmd
        arr.update({"cmd_t": np.asarray(C["t"]), "cmd_tw": np.asarray(C["tw"]), "cmd_q": np.asarray(C["q"]),
                    "cmd_q_sent": np.asarray(C["q_sent"]),
                    "cmd_lh": np.asarray(C["lh"], float), "cmd_rh": np.asarray(C["rh"], float),
                    "cmd_seg": np.asarray(C["seg"]), "cmd_ok": np.asarray(C["ok"])})
        np.savez_compressed(os.path.join(self.out, "raw.npz"), **arr)
        meta = {"segments": self.segments, "notes": self.notes, "p1_records": self.p1_records,
                "js_snapshots": self.js_snapshots, "port_offset": self.off, "fake": bool(self.a.fake),
                "quick": bool(self.a.quick), "events": [e for e in self.events if e.get("op") in ("arm", "walk",
                                                                                                    "turn_to", "go_to")]}
        with open(os.path.join(self.out, "run.json"), "w") as f:
            json.dump(meta, f, indent=1, default=_jd)


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ======================================================================================================
# analysis
# ======================================================================================================
def _fit_sine(t, y, f):
    """y ~ a sin + b cos + c + d t. Returns (amplitude, phase, residual rms). phase: y = A sin(wt + phase)."""
    w = 2 * math.pi * f
    M = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t), t - t.mean()], 1)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = coef[0], coef[1]
    res = y - M @ coef
    return float(math.hypot(a, b)), float(math.atan2(b, a)), float(np.sqrt(np.mean(res ** 2)))


def _zoh(tc, vc, t):
    i = np.searchsorted(tc, t, side="right") - 1
    i = np.clip(i, 0, len(tc) - 1)
    return vc[i]


def _win(t, a, b):
    return (t >= a) & (t <= b)


def _tilt(quat):
    w, x, y, z = quat.T
    # angle between body z and world z: cos = 1 - 2(x^2 + y^2)
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def analyze(out: str, plots: bool = True) -> dict:
    R = dict(np.load(os.path.join(out, "raw.npz")))
    meta = json.load(open(os.path.join(out, "run.json")))
    segs = meta["segments"]
    t = R["dbg_t"]
    q = R["dbg_q"]
    act = R["dbg_act"]
    ref = R["dbg_ref"]
    qa, aa, ra = q[:, ARM_MJ], act[:, ARM_MJ], ref[:, ARM_MJ]
    tc, cq = R["cmd_t"], R["cmd_q"]
    gt_t = R["gt_t"]
    M: dict = {"run": {"out": out, "fake": meta.get("fake"), "quick": meta.get("quick"),
                       "n_g1_debug": int(len(t)), "n_cmd": int(len(tc)), "n_gt": int(len(gt_t)),
                       "g1_debug_hz": float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 else None,
                       "cmd_hz": float((len(tc) - 1) / (tc[-1] - tc[0])) if len(tc) > 1 else None,
                       "cmd_rejected": int((~R["cmd_ok"].astype(bool)).sum()) if len(tc) else 0,
                       "errors": meta["notes"].get("errors")}}
    rtf = R["gt_rtf"][np.isfinite(R["gt_rtf"])]
    M["run"]["rtf"] = {"median": float(np.median(rtf)) if len(rtf) else None,
                       "p5": float(np.percentile(rtf, 5)) if len(rtf) else None}
    M["run"]["load"] = [meta["notes"].get("load_start"), meta["notes"].get("load_end")]
    M["run"]["ik_err_m"] = meta["notes"].get("ik_err_m")

    def cmd_at(tt):
        return _zoh(tc, cq, tt) if len(tc) else np.full((len(tt), 14), np.nan)

    def palm_err(q29_meas, arms_cmd):
        """mm, pelvis frame: FK(measured) vs FK(commanded arms + measured waist), per side (palm, wrist)."""
        out_ = {"left_palm": [], "right_palm": [], "left_wrist": [], "right_wrist": []}
        for qm, ac in zip(q29_meas, arms_cmd):
            nm = K.named_from_q29(qm)
            nc = dict(nm)
            nc.update({n: float(v) for n, v in zip(ARM, ac)})
            pm, pc = K.points(nm), K.points(nc)
            for k in out_:
                out_[k].append(1000 * float(np.linalg.norm(pm[k] - pc[k])))
        return {k: np.asarray(v) for k, v in out_.items()}

    # ---- baseline: SONIC vs its own reference ------------------------------------------------------------
    base = []
    for s in segs:
        if s["kind"] == "baseline" and s.get("t1"):
            m = _win(t, s["t0"] + 0.5, s["t1"])
            if m.sum() < 5:
                continue
            e = qa[m] - ra[m]
            pe = palm_err(q[m][::5], ra[m][::5])
            base.append({"seg": s["name"], "mean_err_rad": dict(zip(map(aname, range(14)), np.round(e.mean(0), 4).tolist())),
                         "max_abs_mean_err_rad": float(np.abs(e.mean(0)).max()),
                         "palm_err_mm": {k: round(float(np.mean(v)), 1) for k, v in pe.items()}})
    M["baseline_sonic_own_reference"] = base

    # ---- static holds -------------------------------------------------------------------------------------
    static = []
    for s in segs:
        if s["kind"] != "static" or not s.get("t1"):
            continue
        m = _win(t, max(s["t1"] - 2.0, 0.5 * (s["t0"] + s["t1"])), s["t1"])   # last 2 s (at most half the hold)
        if m.sum() < 5:
            continue
        tgt = np.asarray(s["target"])
        e = qa[m] - tgt
        ea = aa[m] - tgt
        pe = palm_err(q[m][::5], np.repeat(tgt[None], len(q[m][::5]), 0))
        # settle time after the move ended (= hold start): last sample outside a 0.05 rad band on any arm joint
        mm = _win(t, s["t0"], s["t1"])
        bad = np.where(np.abs(qa[mm] - tgt).max(1) >= 0.05)[0]
        if not len(bad):
            settle = 0.0
        elif bad[-1] + 1 < int(mm.sum()):
            settle = float(t[mm][bad[-1] + 1] - s["t0"])
        else:
            settle = None     # never inside the band during the hold
        static.append({"seg": s["name"], "pose": s.get("pose"),
                       "ss_err_rad": dict(zip(map(aname, range(14)), np.round(e.mean(0), 4).tolist())),
                       "ss_err_policy_action_rad": dict(zip(map(aname, range(14)), np.round(ea.mean(0), 4).tolist())),
                       "max_abs_ss_err_rad": float(np.abs(e.mean(0)).max()),
                       "rms_ss_err_rad": float(np.sqrt(np.mean(e.mean(0) ** 2))),
                       "worst_joint": aname(int(np.argmax(np.abs(e.mean(0))))),
                       "palm_err_mm": {k: round(float(np.mean(v)), 1) for k, v in pe.items()},
                       "settle_s_0p05rad": settle})
    M["static"] = static

    # ---- P1 get_joint_state vs g1_debug (cross-check) ----------------------------------------------------
    xc = []
    for snap in meta.get("js_snapshots", []):
        rep = snap.get("rep") or {}
        if not rep.get("names"):
            continue
        nm = dict(zip(rep["names"], rep["q"]))
        i = int(np.searchsorted(t, snap["t"]) - 1)
        if i < 0:
            continue
        d_arm = [nm[n] - q[i, jm.MJ[n]] for n in ARM if n in nm]
        d_lh = [nm[n] - R["dbg_lh"][i, k] for k, n in enumerate(jm.LEFT_HAND_JOINTS) if n in nm]
        d_rh = [nm[n] - R["dbg_rh"][i, k] for k, n in enumerate(jm.RIGHT_HAND_JOINTS) if n in nm]
        xc.append({"seg": segs[snap["seg"]]["name"], "arm_max_abs_diff": float(np.max(np.abs(d_arm))) if d_arm else None,
                   "hand_max_abs_diff": float(np.max(np.abs(d_lh + d_rh))) if d_lh else None})
    M["crosscheck_p1_get_joint_state_vs_g1_debug"] = xc

    # ---- steps --------------------------------------------------------------------------------------------
    steps = []
    for s in segs:
        if s["kind"] != "step" or not s.get("t1"):
            continue
        i, d = int(s["joint"]), float(s["delta"])
        mc = _win(tc, s["t0"] - 0.1, s["t1"])
        if mc.sum() < 3:
            continue
        ts = s["t0"]                           # first command of the new target is sent at ~t0
        m = _win(t, ts - 0.2, s["t1"])
        y0 = float(np.median(qa[_win(t, ts - 0.2, ts), i])) if _win(t, ts - 0.2, ts).any() else float(qa[m][0, i])
        yy, tt = qa[m, i] - y0, t[m] - ts
        ya = aa[m, i] - float(np.median(aa[_win(t, ts - 0.2, ts), i]))

        def cross(v, frac):
            k = np.where((tt > 0) & (np.sign(d) * v >= frac * abs(d)))[0]
            return float(tt[k[0]]) if len(k) else None

        final = float(np.mean(yy[tt > tt.max() - 0.5]))
        settle = None
        band = np.abs(yy - d) <= 0.05 * abs(d) + 0.01
        if band.any():
            bad = np.where(~band & (tt > 0))[0]
            settle = float(tt[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(tt) else None
        steps.append({"seg": s["name"], "joint": aname(i), "delta": d, "dead_time_10pct_s": cross(yy, 0.1),
                      "t50_s": cross(yy, 0.5), "rise_10_90_s": (lambda a, b: None if a is None or b is None else b - a)(
                          cross(yy, 0.1), cross(yy, 0.9)),
                      "policy_action_10pct_s": cross(ya, 0.1), "policy_action_50pct_s": cross(ya, 0.5),
                      "overshoot_pct": float(100 * max(0.0, (np.max(np.sign(d) * yy) - abs(d)) / abs(d))),
                      "final_err_rad": final - d, "settle_5pct_s": settle})
    M["steps"] = steps
    # P1 `record` cross-check of the step timing (motor_q from the sim itself, ~40 Hz)
    p1s = []
    for rec in meta.get("p1_records", []):
        path = (rec.get("off") or {}).get("path")
        if path and not os.path.exists(path):
            path = os.path.join(out, os.path.basename(path))     # pulled to another machine
        if not path or not os.path.exists(path):
            continue
        try:
            Z = np.load(path)
            tw = Z["t_wall"]
            mq = Z["motor_q"]
            # map wall time -> monotonic with the command log's pairs
            off_ = float(np.median(R["cmd_tw"] - R["cmd_t"]))
            tm = tw - off_
            for s in segs:
                if s["kind"] != "step":
                    continue
                i, d = int(s["joint"]), float(s["delta"])
                j = int(ARM_MJ[i])
                m = _win(tm, s["t0"] - 0.2, s["t1"])
                if m.sum() < 10:
                    continue
                y0 = float(np.median(mq[_win(tm, s["t0"] - 0.2, s["t0"]), j]))
                yy, tt = mq[m, j] - y0, tm[m] - s["t0"]
                k = np.where((tt > 0) & (np.sign(d) * yy >= 0.1 * abs(d)))[0]
                k5 = np.where((tt > 0) & (np.sign(d) * yy >= 0.5 * abs(d)))[0]
                p1s.append({"seg": s["name"], "dead_time_10pct_s": float(tt[k[0]]) if len(k) else None,
                            "t50_s": float(tt[k5[0]]) if len(k5) else None, "rate_hz": float(1 / np.median(np.diff(tm)))})
        except Exception as e:
            p1s.append({"error": repr(e)})
    M["steps_p1_record"] = p1s

    # ---- sines ----------------------------------------------------------------------------------------------
    sines = []
    for s in segs:
        if s["kind"] not in ("sine", "sine_vel", "walk", "turn") or "win" not in s or not s.get("t1"):
            continue
        a_, b_ = s["win"]
        if b_ - a_ < 1.0:
            continue
        m = _win(t, a_, b_)
        mc = _win(tc, a_, b_)
        if m.sum() < 20 or mc.sum() < 20:
            continue
        f = float(s["f"])
        for i in s["joints"]:
            Ac, pc_, _ = _fit_sine(tc[mc], cq[mc, i], f)
            Am, pm_, resm = _fit_sine(t[m], qa[m, i], f)
            Aa, pa_, _ = _fit_sine(t[m], aa[m, i], f)
            lag = wrap(pc_ - pm_)
            lag_a = wrap(pc_ - pa_)
            lat = lag / (2 * math.pi * f)
            e = qa[m, i] - cmd_at(t[m])[:, i]
            # lag-compensated error: command delayed by the measured latency (linear interpolation)
            cdel = np.interp(t[m] - lat, tc, cq[:, i])
            e2 = qa[m, i] - cdel
            # cross-talk: motion of the other arm joints (not excited) around their commanded values
            others = [k for k in range(14) if k not in s["joints"]]
            ct = np.sqrt(np.mean((qa[m][:, others] - cmd_at(t[m])[:, others]
                                  - np.mean(qa[m][:, others] - cmd_at(t[m])[:, others], 0)) ** 2, 0))
            sines.append({"seg": s["name"], "kind": s["kind"], "joint": aname(i), "side": s.get("side"),
                          "f": f, "amp_cmd": Ac, "gain": Am / Ac if Ac > 1e-6 else None,
                          "gain_policy_action": Aa / Ac if Ac > 1e-6 else None,
                          "phase_lag_deg": math.degrees(lag), "latency_ms": 1000 * lat,
                          "latency_policy_action_ms": 1000 * lag_a / (2 * math.pi * f),
                          "rms_err_rad": float(np.sqrt(np.mean(e ** 2))),
                          "rms_err_lag_comp_rad": float(np.sqrt(np.mean((e2 - e2.mean()) ** 2))),
                          "bias_rad": float(np.mean(e)),
                          "fit_resid_rms_rad": resm,
                          "crosstalk_max_rms_rad": float(ct.max()) if len(ct) else None,
                          "crosstalk_joint": aname(others[int(np.argmax(ct))]) if len(ct) else None})
    M["sine"] = sines
    # bandwidth per joint (one arm at a time and both): -3 dB crossing of the measured gain
    bw = {}
    for kind in ("one_arm", "both"):
        for i in range(14):
            rows = [r for r in sines if r["kind"] == "sine" and r["joint"] == aname(i)
                    and ((r["side"] == "both") == (kind == "both"))]
            if not rows:
                continue
            rows.sort(key=lambda r: r["f"])
            fs = [r["f"] for r in rows]
            gs = [r["gain"] for r in rows]
            f3 = None
            for (f0, g0), (f1, g1) in zip(zip(fs, gs), zip(fs[1:], gs[1:])):
                if g0 >= 0.707 > g1:
                    f3 = math.exp(math.log(f0) + (math.log(f1) - math.log(f0)) * (g0 - 0.707) / (g0 - g1))
                    break
            if f3 is None and gs and gs[0] < 0.707:
                f3 = f"<{fs[0]:g}"
            if f3 is None:
                f3 = f">{fs[-1]:g}"
            bw[f"{kind}:{aname(i)}"] = {"f_3db_hz": f3, "gain_by_f": dict(zip(map(str, fs), [round(g, 3) for g in gs])),
                                        "latency_ms_by_f": dict(zip(map(str, fs), [round(r["latency_ms"], 1) for r in rows]))}
    M["bandwidth"] = bw

    # ---- grasp script ---------------------------------------------------------------------------------------
    gr = []
    for s in segs:
        if s["kind"] != "grasp" or not s.get("t1"):
            continue
        m = _win(t, s["t0"], s["t1"])
        if m.sum() < 20:
            continue
        side = s.get("side", "right")
        cm = cmd_at(t[m])
        pe = palm_err(q[m], cm)
        # latency-compensated palm error (median sine-derived latency ~ cross-correlation over the whole script)
        lags = np.arange(0, 0.4, 0.01)
        idx = SIDE_IDX[side]
        best = min(lags, key=lambda L: np.mean((qa[m][:, idx] - np.stack([np.interp(t[m] - L, tc, cq[:, k]) for k in idx], 1)) ** 2))
        cml = np.stack([np.interp(t[m] - best, tc, cq[:, k]) for k in range(14)], 1)
        pe2 = palm_err(q[m], cml)
        hq = R["dbg_rh" if side == "right" else "dbg_lh"][m]
        hc = _zoh(tc, R["cmd_rh" if side == "right" else "cmd_lh"], t[m])
        clo_m = np.array([jm.hand_closure_of(side, v) for v in hq])
        clo_c = np.array([jm.hand_closure_of(side, v) if np.all(np.isfinite(v)) else np.nan for v in hc])
        gr.append({"seg": s["name"], "side": side, "palm_err_mm": {"rms": float(np.sqrt(np.mean(pe[f"{side}_palm"] ** 2))),
                                                                  "max": float(pe[f"{side}_palm"].max()),
                                                                  "p95": float(np.percentile(pe[f"{side}_palm"], 95))},
                   "best_lag_s": float(best),
                   "palm_err_lag_comp_mm": {"rms": float(np.sqrt(np.mean(pe2[f"{side}_palm"] ** 2))),
                                            "max": float(pe2[f"{side}_palm"].max()),
                                            "p95": float(np.percentile(pe2[f"{side}_palm"], 95))},
                   "wrist_err_mm_rms": float(np.sqrt(np.mean(pe[f"{side}_wrist"] ** 2))),
                   "joint_rms_err_rad": dict(zip([aname(k) for k in idx],
                                                 np.round(np.sqrt(np.mean((qa[m][:, idx] - cm[:, idx]) ** 2, 0)), 4).tolist())),
                   "closure_max_measured": float(np.nanmax(clo_m)), "closure_cmd_max": float(np.nanmax(clo_c))})
    M["grasp"] = gr

    # ---- hands ----------------------------------------------------------------------------------------------
    hands = []
    for s in segs:
        if s["kind"] != "hands" or not s.get("t1"):
            continue
        m = _win(t, s["t1"] - 1.0, s["t1"])
        mm = _win(t, s["t0"], s["t1"])
        row = {"seg": s["name"], "closure_cmd": s["closure"]}
        for side, key in (("left", "dbg_lh"), ("right", "dbg_rh")):
            cmdv = np.array(jm.hand_closure(side, s["closure"]))
            hm = R[key][m]
            e = hm.mean(0) - cmdv
            clo = jm.hand_closure_of(side, hm.mean(0))
            # time to reach 90 % of the commanded change in closure
            c_series = np.array([jm.hand_closure_of(side, v) for v in R[key][mm]])
            c0 = c_series[0]
            dc = s["closure"] - c0
            t90 = None
            if abs(dc) > 0.05:
                k = np.where(np.sign(dc) * (c_series - c0) >= 0.9 * abs(dc))[0]
                t90 = float(t[mm][k[0]] - s["t0"]) if len(k) else None
            act_ = R["dbg_lha" if side == "left" else "dbg_rha"][m].mean(0)
            row[side] = {"closure_measured": round(float(clo), 3), "closure_err": round(float(clo - s["closure"]), 3),
                         "joint_err_rad": dict(zip(jm.DEX3_SUFFIX, np.round(e, 3).tolist())),
                         "max_abs_joint_err_rad": float(np.abs(e).max()), "t90_s": t90,
                         "deploy_hand_action_matches_cmd": bool(np.allclose(act_, cmdv, atol=0.02))}
        hands.append(row)
    M["hands"] = hands

    # ---- pelvis / base per segment, falls -------------------------------------------------------------------
    pel = []
    gq = R["gt_quat"] if len(gt_t) else np.zeros((0, 4))
    tilt = _tilt(gq) if len(gq) else np.zeros(0)
    for s in segs:
        if not s.get("t1"):
            continue
        m = _win(gt_t, s["t0"], s["t1"])
        if m.sum() < 5:
            continue
        x, y, yaw = R["gt_x"][m], R["gt_y"][m], R["gt_yaw"][m]
        pz = R["gt_pelvis_z"][m]
        pel.append({"seg": s["name"], "kind": s["kind"], "pelvis_z_mean": float(pz.mean()), "pelvis_z_std": float(pz.std()),
                    "pelvis_z_min": float(pz.min()), "tilt_deg_max": float(tilt[m].max()), "tilt_deg_mean": float(tilt[m].mean()),
                    "base_disp_m": float(math.hypot(x[-1] - x[0], y[-1] - y[0])),
                    "base_path_m": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
                    "yaw_change_deg": float(math.degrees(wrap(yaw[-1] - yaw[0]))),
                    "fallen": bool(R["gt_fallen"][m].any() or pz.min() < 0.55)})
    M["pelvis"] = pel
    M["falls"] = int(sum(p["fallen"] for p in pel))
    walking_segs = {s["name"] for s in segs if s.get("walking") or "walk" in s["name"]}
    standing = [p for p in pel if p["kind"] in ("static", "sine", "sine_vel", "step", "grasp", "hands", "move")
                and p["seg"] not in walking_segs]
    base_rows = [p for p in pel if p["kind"] == "baseline"]
    M["pelvis_summary"] = {
        "standing_with_arms": {"tilt_deg_max": max((p["tilt_deg_max"] for p in standing), default=None),
                               "pelvis_z_std_max": max((p["pelvis_z_std"] for p in standing), default=None),
                               "base_disp_m_max": max((p["base_disp_m"] for p in standing), default=None),
                               "base_disp_m_total": float(sum(p["base_disp_m"] for p in standing)),
                               "worst_disp_seg": max(standing, key=lambda p: p["base_disp_m"])["seg"] if standing else None},
        "baseline_no_override": {"tilt_deg_max": max((p["tilt_deg_max"] for p in base_rows), default=None),
                                 "pelvis_z_std_max": max((p["pelvis_z_std"] for p in base_rows), default=None),
                                 "base_disp_m_max": max((p["base_disp_m"] for p in base_rows), default=None)}}
    if len(gt_t):
        m = _win(gt_t, segs[0]["t0"], segs[-1]["t1"] or gt_t[-1])
        M["pelvis_summary"]["whole_run"] = {"tilt_deg_max": float(tilt[m].max()), "pelvis_z_min": float(R["gt_pelvis_z"][m].min())}

    # ---- walking --------------------------------------------------------------------------------------------
    walk = []
    for s in segs:
        if s["kind"] not in ("walk", "turn") or not s.get("t1"):
            continue
        p = next((r for r in pel if r["seg"] == s["name"]), None)
        m = _win(t, s["t0"] + 0.5, s["t1"])
        row = {"seg": s["name"], "kind": s["kind"], "arms": s.get("arms"), "op": (s.get("op") or {}).get("state"),
               "result": {k: v for k, v in ((s.get("op") or {}).get("result") or {}).items()
                          if k in ("displacement_m", "displacement_body0", "yaw_change_deg", "max_cross_track_m",
                                   "reason", "final_yaw_err_deg", "duration_s")},
               "pelvis": p}
        if s.get("arms") and m.sum() > 10 and len(tc):
            cm = cmd_at(t[m])
            e = qa[m] - cm
            row["arm_rms_err_rad_max"] = float(np.sqrt(np.mean(e ** 2, 0)).max())
            row["arm_rms_err_rad_mean"] = float(np.sqrt(np.mean(e ** 2, 0)).mean())
            row["arm_rms_err_worst_joint"] = aname(int(np.argmax(np.sqrt(np.mean(e ** 2, 0)))))
            pe = palm_err(q[m][::5], cm[::5])
            row["palm_err_mm_rms"] = {k: round(float(np.sqrt(np.mean(v ** 2))), 1) for k, v in pe.items() if "palm" in k}
        walk.append(row)
    M["walk"] = walk

    M["summary"] = summarize(M)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1, default=_jd)
    if plots:
        try:
            make_plots(out, R, meta, M)
        except Exception:
            M["run"]["plot_error"] = traceback.format_exc()
            with open(os.path.join(out, "metrics.json"), "w") as f:
                json.dump(M, f, indent=1, default=_jd)
    return M


def summarize(M: dict) -> dict:
    S: dict = {}
    st = [r for r in M["static"] if r["pose"] not in ("sonic_ref",)]
    if st:
        palms = [v for r in st for k, v in r["palm_err_mm"].items() if "palm" in k]
        S["static_palm_err_mm"] = {"median": float(np.median(palms)), "max": float(np.max(palms))}
        S["static_max_abs_joint_err_rad"] = float(max(r["max_abs_ss_err_rad"] for r in st))
        S["static_worst"] = max(st, key=lambda r: r["max_abs_ss_err_rad"])["worst_joint"]
    one = [r for r in M["sine"] if r["kind"] == "sine" and r["side"] in ("left", "right")]
    for f in (0.2, 0.5, 1.0, 2.0):
        rows = [r for r in one if r["f"] == f]
        if rows:
            S[f"sine_{f:g}Hz"] = {"gain_median": float(np.median([r["gain"] for r in rows])),
                                   "gain_min": float(np.min([r["gain"] for r in rows])),
                                   "latency_ms_median": float(np.median([r["latency_ms"] for r in rows])),
                                   "latency_ms_max": float(np.max([r["latency_ms"] for r in rows])),
                                   "latency_policy_action_ms_median": float(np.median([r["latency_policy_action_ms"] for r in rows])),
                                   "rms_err_rad_median": float(np.median([r["rms_err_rad"] for r in rows])),
                                   "rms_err_lag_comp_rad_median": float(np.median([r["rms_err_lag_comp_rad"] for r in rows])),
                                   "n": len(rows)}
    both = [r for r in M["sine"] if r["kind"] == "sine" and r["side"] == "both"]
    if both:
        S["sine_both_arms"] = {"gain_median": float(np.median([r["gain"] for r in both])),
                               "latency_ms_median": float(np.median([r["latency_ms"] for r in both]))}
    vel = [r for r in M["sine"] if r["kind"] == "sine_vel"]
    if vel:
        S["vel_feedforward_ab"] = [{"f": r["f"], "gain_est": r["gain"], "latency_ms_est": r["latency_ms"],
                                    "gain_zero": next((x["gain"] for x in one if x["joint"] == r["joint"] and x["f"] == r["f"]), None),
                                    "latency_ms_zero": next((x["latency_ms"] for x in one if x["joint"] == r["joint"] and x["f"] == r["f"]), None)}
                                   for r in vel]
    f3 = [v["f_3db_hz"] for k, v in M["bandwidth"].items() if k.startswith("one_arm")]
    S["bandwidth_f3db_one_arm"] = f3
    if M["steps"]:
        S["step_dead_time_s"] = [r["dead_time_10pct_s"] for r in M["steps"]]
        S["step_t50_s"] = [r["t50_s"] for r in M["steps"]]
    if M["grasp"]:
        S["grasp"] = [{"seg": g["seg"], "palm_rms_mm": round(g["palm_err_mm"]["rms"], 1),
                       "palm_max_mm": round(g["palm_err_mm"]["max"], 1),
                       "palm_rms_lag_comp_mm": round(g["palm_err_lag_comp_mm"]["rms"], 1),
                       "best_lag_s": g["best_lag_s"]} for g in M["grasp"]]
    if M["hands"]:
        S["hand_closure_err"] = {side: [r[side]["closure_err"] for r in M["hands"]] for side in ("left", "right")}
    S["falls"] = M["falls"]
    S["pelvis"] = M.get("pelvis_summary")
    return S


def make_plots(out: str, R: dict, meta: dict, M: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = meta["segments"]
    t, q, act = R["dbg_t"], R["dbg_q"], R["dbg_act"]
    qa, aa = q[:, ARM_MJ], act[:, ARM_MJ]
    tc, cq = R["cmd_t"], R["cmd_q"]
    t00 = segs[0]["t0"] if segs else (t[0] if len(t) else 0)
    tag = " [FAKE]" if meta.get("fake") else ""

    # 1. bode-like: gain + latency vs f per joint
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), dpi=100, sharex=True)
    for col, side in enumerate(("left", "right", "both")):
        for i in (SIDE_IDX["left"] + SIDE_IDX["right"] if side == "both" else SIDE_IDX[side]):
            rows = sorted([r for r in M["sine"] if r["kind"] == "sine" and r["joint"] == aname(i) and r["side"] == side],
                          key=lambda r: r["f"])
            if not rows:
                continue
            fs = [r["f"] for r in rows]
            axs[0, col].plot(fs, [r["gain"] for r in rows], "o-", label=aname(i))
            axs[1, col].plot(fs, [r["latency_ms"] for r in rows], "o-", label=aname(i))
        axs[0, col].axhline(0.707, color="k", ls=":", lw=1)
        axs[0, col].set_title(f"{side} arm sines: gain (measured / commanded){tag}")
        axs[1, col].set_title("end-to-end latency (phase lag / 2 pi f)")
        axs[1, col].set_xlabel("f [Hz]")
        axs[0, col].set_xscale("log")
        axs[0, col].set_xticks([0.2, 0.5, 1.0, 2.0])
        axs[0, col].set_xticklabels(["0.2", "0.5", "1", "2"])
        axs[0, col].minorticks_off()
        axs[0, col].legend(fontsize=7, ncol=2)
        axs[0, col].grid(alpha=0.3)
        axs[1, col].grid(alpha=0.3)
    axs[1, 0].set_ylabel("ms")
    axs[0, 0].set_ylabel("gain")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bode.png"))
    plt.close(fig)

    # 2. example sines (right shoulder pitch 0.2/0.5/1/2 Hz, right elbow 1 Hz)
    names = [s for s in segs if s["kind"] == "sine" and s.get("side") == "right" and s.get("joint") in ("sh_pitch",)]
    names += [s for s in segs if s["kind"] == "sine" and s.get("side") == "right" and s.get("joint") == "elbow" and s["f"] == 1.0]
    if names:
        fig, axs = plt.subplots(len(names), 1, figsize=(13, 2.4 * len(names)), dpi=100)
        axs = np.atleast_1d(axs)
        for ax, s in zip(axs, names):
            i = s["joints"][0]
            mc = _win(tc, s["t0"], s["t1"])
            m = _win(t, s["t0"], s["t1"])
            ax.plot(tc[mc] - s["t0"], cq[mc, i], "k-", lw=1.2, label="command (client)")
            ax.plot(t[m] - s["t0"], aa[m, i], "C1-", lw=1, label="policy action (lowcmd q)")
            ax.plot(t[m] - s["t0"], qa[m, i], "C0-", lw=1.2, label="measured q")
            ax.axvspan(s["win"][0] - s["t0"], s["win"][1] - s["t0"], color="0.9")
            r = next((r for r in M["sine"] if r["seg"] == s["name"] and r["joint"] == aname(i)), None)
            if r:
                ax.set_title(f"{s['name']}: gain {r['gain']:.2f}, latency {r['latency_ms']:.0f} ms, "
                             f"rms err {r['rms_err_rad']:.3f} rad{tag}", fontsize=9)
            ax.set_ylabel("rad")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        axs[-1].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "sine_examples.png"))
        plt.close(fig)

    # 3. static poses: steady-state error per joint
    st = M["static"]
    if st:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100, gridspec_kw={"width_ratios": [3, 1]})
        w = 0.8 / len(st)
        for k, r in enumerate(st):
            vals = list(r["ss_err_rad"].values())
            axs[0].bar(np.arange(14) + k * w, vals, w, label=r["seg"])
        axs[0].set_xticks(np.arange(14) + 0.4)
        axs[0].set_xticklabels([aname(i) for i in range(14)], rotation=60, fontsize=8)
        axs[0].axhline(0, color="k", lw=0.8)
        axs[0].set_ylabel("steady-state error (measured - command) [rad]")
        axs[0].set_title(f"static holds: steady-state joint error{tag}")
        axs[0].legend(fontsize=7)
        axs[0].grid(alpha=0.3, axis="y")
        labels = [r["seg"].replace("hold_", "") for r in st]
        axs[1].barh(np.arange(len(st)) - 0.2, [r["palm_err_mm"]["left_palm"] for r in st], 0.4, label="left palm")
        axs[1].barh(np.arange(len(st)) + 0.2, [r["palm_err_mm"]["right_palm"] for r in st], 0.4, label="right palm")
        axs[1].axvline(20, color="r", ls=":", lw=1)
        axs[1].set_yticks(np.arange(len(st)))
        axs[1].set_yticklabels(labels, fontsize=8)
        axs[1].set_xlabel("palm position error [mm] (pelvis frame, FK)")
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "static_poses.png"))
        plt.close(fig)

    # 4. steps
    ss = [s for s in segs if s["kind"] == "step"]
    if ss:
        fig, axs = plt.subplots(2, 3, figsize=(15, 6), dpi=100)
        for ax, s in zip(axs.flat, ss):
            i = int(s["joint"])
            m = _win(t, s["t0"] - 0.3, s["t1"])
            mc = _win(tc, s["t0"] - 0.3, s["t1"])
            ax.step(tc[mc] - s["t0"], cq[mc, i], "k-", where="post", label="command")
            ax.plot(t[m] - s["t0"], aa[m, i], "C1.-", ms=3, lw=1, label="policy action")
            ax.plot(t[m] - s["t0"], qa[m, i], "C0.-", ms=3, lw=1.2, label="measured")
            r = next((r for r in M["steps"] if r["seg"] == s["name"]), {})
            ax.set_title(f"{s['name']}: t10 {r.get('dead_time_10pct_s')} t50 {r.get('t50_s')}", fontsize=8)
            ax.grid(alpha=0.3)
        axs.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "steps.png"))
        plt.close(fig)

    # 5. grasp: palm position + closure
    gs = [s for s in segs if s["kind"] == "grasp"]
    if gs:
        fig, axs = plt.subplots(len(gs), 2, figsize=(15, 3.6 * len(gs)), dpi=100, squeeze=False)
        for row, s in enumerate(gs):
            side = s.get("side", "right")
            m = _win(t, s["t0"], s["t1"])
            cm = _zoh(tc, cq, t[m])
            P_m, P_c = [], []
            for qm, ac in zip(q[m], cm):
                nm = K.named_from_q29(qm)
                nc = dict(nm)
                nc.update({n: float(v) for n, v in zip(ARM, ac)})
                P_m.append(K.points(nm)[f"{side}_palm"])
                P_c.append(K.points(nc)[f"{side}_palm"])
            P_m, P_c = np.array(P_m), np.array(P_c)
            tt = t[m] - s["t0"]
            for k, lab in enumerate("xyz"):
                axs[row, 0].plot(tt, P_c[:, k], f"C{k}--", lw=1, label=f"{lab} cmd")
                axs[row, 0].plot(tt, P_m[:, k], f"C{k}-", lw=1.4, label=f"{lab} meas")
            axs[row, 0].set_title(f"{s['name']}: {side} palm in the pelvis frame (FK){tag}", fontsize=9)
            axs[row, 0].set_ylabel("m")
            axs[row, 0].legend(fontsize=7, ncol=3)
            axs[row, 0].grid(alpha=0.3)
            err = 1000 * np.linalg.norm(P_m - P_c, axis=1)
            axs[row, 1].plot(tt, err, "C3-", label="palm error [mm]")
            key = "dbg_rh" if side == "right" else "dbg_lh"
            ckey = "cmd_rh" if side == "right" else "cmd_lh"
            hc = _zoh(tc, R[ckey], t[m])
            ax2 = axs[row, 1].twinx()
            ax2.plot(tt, [jm.hand_closure_of(side, v) if np.all(np.isfinite(v)) else np.nan for v in hc], "k--", lw=1,
                     label="closure cmd")
            ax2.plot(tt, [jm.hand_closure_of(side, v) for v in R[key][m]], "k-", lw=1.2, label="closure meas")
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend(fontsize=7, loc="upper right")
            axs[row, 1].axhline(20, color="r", ls=":", lw=1)
            axs[row, 1].set_ylabel("mm")
            axs[row, 1].legend(fontsize=7, loc="upper left")
            axs[row, 1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "grasp.png"))
        plt.close(fig)

    # 6. hands
    hs = [s for s in segs if s["kind"] == "hands"]
    if hs:
        fig, axs = plt.subplots(2, 1, figsize=(13, 6), dpi=100, sharex=True)
        a0, b0 = hs[0]["t0"], hs[-1]["t1"]
        m = _win(t, a0, b0)
        for ax, side, key, ckey in ((axs[0], "left", "dbg_lh", "cmd_lh"), (axs[1], "right", "dbg_rh", "cmd_rh")):
            hc = _zoh(tc, R[ckey], t[m])
            for k, nme in enumerate(jm.DEX3_SUFFIX):
                ax.plot(t[m] - a0, hc[:, k], f"C{k}--", lw=0.8)
                ax.plot(t[m] - a0, R[key][m][:, k], f"C{k}-", lw=1.2, label=nme)
            ax.set_title(f"{side} Dex3: command (dashed) vs measured{tag}", fontsize=9)
            ax.set_ylabel("rad")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7, ncol=7)
        axs[1].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "hands.png"))
        plt.close(fig)

    # 7. pelvis over the whole run with segment shading
    gt_t = R["gt_t"]
    if len(gt_t):
        fig, axs = plt.subplots(3, 1, figsize=(15, 8), dpi=100, sharex=True)
        tt = gt_t - t00
        axs[0].plot(tt, R["gt_pelvis_z"], "C0-", lw=1)
        axs[0].set_ylabel("pelvis z [m]")
        axs[1].plot(tt, _tilt(R["gt_quat"]), "C1-", lw=1)
        axs[1].set_ylabel("tilt [deg]")
        axs[2].plot(tt, R["gt_x"] - R["gt_x"][0], label="x")
        axs[2].plot(tt, R["gt_y"] - R["gt_y"][0], label="y")
        axs[2].set_ylabel("base disp [m]")
        axs[2].legend(fontsize=8)
        colors = {"static": "0.92", "sine": "#e8f0ff", "walk": "#ffe8e8", "turn": "#fff3d6", "grasp": "#e8ffe8",
                  "hands": "#f3e8ff", "step": "#e0f7f7", "baseline": "#ffffff"}
        for s in segs:
            if s.get("t1") is None:
                continue
            for ax in axs:
                ax.axvspan(s["t0"] - t00, s["t1"] - t00, color=colors.get(s["kind"], "0.97"), lw=0)
        for ax in axs:
            ax.grid(alpha=0.3)
        axs[0].set_title(f"pelvis and base over the run (shaded: segments; blue sines, red walks, yellow turns){tag}")
        axs[2].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "pelvis.png"))
        plt.close(fig)

    # 8. walking: top view + arm error during walks
    ws = [s for s in segs if s["kind"] in ("walk", "turn")]
    if ws and len(gt_t):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100, gridspec_kw={"width_ratios": [1, 2]})
        for k, s in enumerate(ws):
            m = _win(gt_t, s["t0"], s["t1"])
            axs[0].plot(R["gt_x"][m], R["gt_y"][m], "-", lw=1.5, label=s["name"])
        axs[0].set_aspect("equal")
        axs[0].legend(fontsize=7)
        axs[0].set_title(f"base trajectory during walks/turns{tag}")
        axs[0].grid(alpha=0.3)
        for s in ws:
            if not s.get("arms"):
                continue
            m = _win(t, s["t0"], s["t1"])
            cm = _zoh(tc, cq, t[m])
            e = np.abs(qa[m] - cm).max(1)
            axs[1].plot(t[m] - t00, e, lw=1, label=s["name"])
        gsw = [s for s in segs if s["name"] == "walk_back_grasp_script"]
        for s in gsw:
            m = _win(t, s["t0"], s["t1"])
            cm = _zoh(tc, cq, t[m])
            axs[1].plot(t[m] - t00, np.abs(qa[m] - cm).max(1), lw=1, label=s["name"])
        axs[1].set_ylabel("max |arm joint error| [rad]")
        axs[1].set_xlabel("s since start")
        axs[1].legend(fontsize=7)
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "walk.png"))
        plt.close(fig)


# ======================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--phases", default="baseline,static,steps,sine,grasp,hands,walk")
    ap.add_argument("--video", action="store_true", help="record chase/head/top videos (viz/recorder.py)")
    ap.add_argument("--analyze", default=None, help="only (re-)analyse an existing run dir")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quick", action="store_true", help="short durations (plumbing check)")
    ap.add_argument("--fake", action="store_true", help="run against in-process fakes (fake P1 + fake deploy)")
    ap.add_argument("--servo-ki", type=float, default=2.0, help="arm channel servo gain (0 = off, body/arm.py)")
    ap.add_argument("--servo-delay", type=float, default=0.15)
    ap.add_argument("--servo-max", type=float, default=0.4)
    ap.add_argument("--lead", type=float, default=0.0, help="send pre-planned targets this many s ahead")
    ap.add_argument("--sine-freqs", default="0.2,0.5,1.0")
    ap.add_argument("--sine-sides", default="left,right,both")
    ap.add_argument("--no-sine-extras", dest="sine_extras", action="store_false",
                    help="skip the 2 Hz and vel=est sines")
    ap.add_argument("--chase", action="store_true", help="turn P1's VizCams to 'low' for the chase video, restore after")
    a = ap.parse_args(argv)
    if a.analyze:
        M = analyze(a.analyze, plots=not a.no_plots)
        print(json.dumps(M["summary"], indent=1, default=_jd))
        return 0
    a.out = a.out or os.path.join(ROOT, "outputs/arm_track", time.strftime("%Y%m%d-%H%M%S"))
    stack = None
    if a.fake:
        from body.config import BodyConfig
        from body.service import BodyService
        from tools.fake_deploy import FakeDeploy
        from tools.fake_p1 import FakeP1
        off = a.port_offset if a.port_offset is not None else 200
        a.port_offset = off
        p1 = FakeP1(off, os.path.join(a.out, "fake_p1"), log=lambda *_: None).start()
        dep = FakeDeploy(off, log=lambda *_: None).start()
        svc = BodyService(BodyConfig(port_offset=off), log_dir=os.path.join(a.out, "body"), log=None)
        th = threading.Thread(target=svc.run, daemon=True)
        th.start()
        stack = (p1, dep, svc, th)
        time.sleep(1.0)
    try:
        rc = Run(a).run([p for p in a.phases.split(",") if p])
    finally:
        if stack:
            p1, dep, svc, th = stack
            svc.stop()
            th.join(5)
            dep.stop()
            p1.stop()
    M = analyze(a.out, plots=not a.no_plots)
    print(json.dumps(M["summary"], indent=1, default=_jd))
    print(f"[arm_track] out: {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
