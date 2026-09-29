"""Dex3 hand probe on the live M1 stack, measured directly in P1 (REP get_joint_state).

Drives both Dex3 hands through the body's arm channel (op `arm`: body -> SonicMux -> deploy -> rt/dex3/<side>/cmd
-> P1) and polls P1 `get_joint_state` (~40 Hz) for the simulated hand joint positions and P1's current hand targets
(the bridge's q_t = what the deploy wrote, after its +-0.25 rad per-write clamp, dex3_hands.hpp:114-190). The arms
are held where they were measured at the start (streamed as the upper-body target), so SONIC keeps standing.

    # box, M1 stack up
    .venv/bin/python -m sim_isaac.tools.hand_step_probe --out outputs/body_wave/hands/<run> [--frames]
    .venv/bin/python -m sim_isaac.tools.hand_step_probe --analyze outputs/body_wave/hands/<run>   # re-analyse + plots
    .venv/bin/python -m sim_isaac.tools.hand_step_probe --compare <run1> <run2> ...             # drift across runs

Phases:
  rest     the hands as the stack left them (the deploy's default fist: thumb_0/1 commanded 0), before any command
  steps    per Dex3 joint, both hands at once: base (mid-range, the other 6 joints open) -> +0.3 -> base -> -0.3 ->
           base, `--hold` s each. t_band = time from the client's send until |q - goal| <= 0.05 rad for the rest of
           the hold (end to end: body, deploy clamp, DDS, P1 PD); t_band_p1 = the same from the moment P1's target
           for that joint first moved
  closure  open -> 0.5 -> 1.0 -> 0 (body/joint_map.hand_closure: the deploy's fist scaled), `--closure-hold` s each;
           closure ratio = hand_closure_of(mean of the last 0.5 s) / commanded
  frames   (--frames) palms raised in front of the head camera; ego frames open / fist / thumbs rotated
  end      stream ended (the body blends the hands back to the deploy's default fist), rest pose again
Pass (sim_isaac owner's exit test): every transition t_band < 0.5 s and steady-state |err| <= 0.05 rad, closure ratio
>= 0.9 at 0.5 and 1.0 on both hands, thumb_0/1 at rest within 0.05 rad of their command (0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body import joint_map as bjm  # noqa: E402

SIDES = ("left", "right")
SUF = bjm.DEX3_SUFFIX
HAND = {s: list(bjm.HAND_JOINTS[s]) for s in SIDES}
ARM = list(bjm.ARM_JOINTS)
STEP = 0.3
BAND = 0.05
T_PASS = 0.5
CLOSURE_PASS = 0.9
# palms in front of the head camera (d435 on torso_link), 34 cm apart so the hands cannot touch: IK on main.urdf
# (body/g1_kin.ik_palm, palms at pelvis (0.28, +-0.17, 0.10), +-20-24 deg off the image centre); only for --frames
FRAME_POSE = {"shoulder_pitch": -0.20, "shoulder_roll": 0.17, "shoulder_yaw": -0.03, "elbow": 0.27,
              "wrist_roll": 0.01, "wrist_pitch": -0.12, "wrist_yaw": 0.0}


def step_base(side: str, k: int) -> float:
    lo, hi = bjm.JOINT_LIMITS[HAND[side][k]]
    return float(np.clip(0.5 * (lo + hi), lo + STEP + BAND, hi - STEP - BAND))


def frame_pose() -> dict:
    out = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        for k, v in FRAME_POSE.items():
            out[f"{side}_{k}_joint"] = v * (sgn if k in ("shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw")
                                            else 1.0)
    return out


def minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


# ======================================================================================================
# live run
# ======================================================================================================
class Poller(threading.Thread):
    """P1 get_joint_state as fast as P1 answers (capped at `hz`); samples stamped with this process's clock."""

    def __init__(self, p1, hz: float = 40.0):
        super().__init__(daemon=True, name="p1-poll")
        self.p1, self.dt = p1, 1.0 / hz
        self.rows: list[tuple] = []
        self.names = None
        self.idx = None
        self.stop_ev = threading.Event()
        self.errors = 0

    def run(self):
        while not self.stop_ev.is_set():
            t0 = time.monotonic()
            try:
                r = self.p1.call("get_joint_state", timeout_s=2.0)
            except Exception:  # noqa: BLE001
                self.errors += 1
                time.sleep(0.1)
                continue
            t1 = time.monotonic()
            if self.idx is None:
                self.names = r["names"]
                pos = {n: i for i, n in enumerate(self.names)}
                self.idx = np.array([pos[n] for n in HAND["left"] + HAND["right"]])
                self.arm_idx = np.array([pos[n] for n in ARM])
            q = np.asarray(r["q"], float)
            qt = np.asarray(r["q_target"], float)
            self.rows.append((0.5 * (t0 + t1), q[self.idx], qt[self.idx], q[self.arm_idx]))
            d = self.dt - (time.monotonic() - t0)
            if d > 0:
                time.sleep(d)

    def latest(self):
        return self.rows[-1] if self.rows else None


class Probe:
    def __init__(self, a):
        import zmq

        from body.client import BodyClient
        from body.config import ep, port_offset_from_env, ports as _ports
        from body.p1_client import P1Rpc
        self.a = a
        self.off = port_offset_from_env() if a.port_offset is None else a.port_offset
        P = _ports(self.off)
        self.ctx = zmq.Context.instance()
        self.p1 = P1Rpc(ep(P["p1_rep"]), timeout_s=5.0, ctx=self.ctx)
        self.bc = BodyClient(port_offset=self.off, ctx=self.ctx).connect(20)
        self.poll = Poller(P1Rpc(ep(P["p1_rep"]), timeout_s=5.0, ctx=self.ctx), a.poll_hz)
        self.cmd: list[tuple] = []          # (t_mono, left7, right7, seg)
        self.segments: list[dict] = []
        self.notes: dict = {"errors": [], "port_offset": self.off}
        self.arm = None
        self.hands = {"left": list(bjm.DEX3_CLOSED["left"]), "right": list(bjm.DEX3_CLOSED["right"])}
        self.upper = None

    # -- helpers -----------------------------------------------------------------------------------------
    def seg(self, name: str, kind: str, **meta) -> dict:
        s = {"name": name, "kind": kind, "t0": time.monotonic(), "t1": None, **meta}
        self.segments.append(s)
        print(f"[hand_probe] {name}", flush=True)
        return s

    def stream(self, dur: float, hands_fn=None, upper_fn=None):
        """Send hands (and the held arms) at 50 Hz for dur s. hands_fn(t) -> (left7, right7) or None."""
        t0 = time.monotonic()
        k = 0
        segi = len(self.segments) - 1
        while True:
            t = time.monotonic() - t0
            if t > dur:
                break
            if hands_fn is not None:
                h = hands_fn(t)
                if h is not None:
                    self.hands = {"left": list(h[0]), "right": list(h[1])}
            if upper_fn is not None:
                self.upper = upper_fn(t)
            rep = self.arm.send(upper_body=self.upper, left_hand=self.hands["left"], right_hand=self.hands["right"])
            self.cmd.append((time.monotonic(), list(self.hands["left"]), list(self.hands["right"]), segi))
            if not rep.get("ok"):
                self.notes["errors"].append(f"arm send rejected: {rep}")
                raise RuntimeError(f"arm stream rejected: {rep}")
            k += 1
            d = t0 + k * 0.02 - time.monotonic()
            if d > 0:
                time.sleep(d)
            elif d < -0.1:
                k = int((time.monotonic() - t0) / 0.02)

    def hold_hands(self, left, right, dur):
        self.stream(dur, lambda t: (left, right))

    def snapshot(self) -> dict:
        r = self.p1.call("get_joint_state")
        pos = {n: i for i, n in enumerate(r["names"])}
        return {s: {"q": [round(r["q"][pos[n]], 4) for n in HAND[s]],
                    "q_target": [round(r["q_target"][pos[n]], 4) for n in HAND[s]],
                    "kp": [round(r["kp"][pos[n]], 3) for n in HAND[s]],
                    "kd": [round(r["kd"][pos[n]], 3) for n in HAND[s]]} for s in SIDES}

    # -- run ---------------------------------------------------------------------------------------------
    def run(self):
        a = self.a
        os.makedirs(a.out, exist_ok=True)
        st = self.bc.status()
        self.notes["status_start"] = {k: st.get(k) for k in ("fault", "in_control", "active", "arm")}
        self.notes["p1_stats_start"] = self.p1.try_call("get_stats")
        self.notes["load_start"] = os.getloadavg()
        if not st.get("in_control") or st.get("fault"):
            raise RuntimeError(f"robot not standing under SONIC control: {self.notes['status_start']}")
        self.notes["rest_start"] = self.snapshot()
        self.poll.start()
        time.sleep(1.0)
        s = self.seg("rest", "rest")
        time.sleep(1.0)
        s["t1"] = time.monotonic()
        r = self.p1.call("get_joint_state")
        pos = {n: i for i, n in enumerate(r["names"])}
        arm0 = {n: float(r["q"][pos[n]]) for n in ARM}
        self.upper = dict(arm0)
        self.notes["arm_start"] = arm0
        if a.frames:
            self.bc.camera_frame(timeout=3.0)   # subscribe now: a later first frame must not stall the 50 Hz stream
        self.arm = self.bc.arm_stream(stream=f"hand-probe-{int(time.time())}")
        try:
            # open both hands
            s = self.seg("open", "open")
            self.hold_hands(bjm.DEX3_OPEN, bjm.DEX3_OPEN, 2.0)
            s["t1"] = time.monotonic()
            if "steps" in a.phases:
                self.ph_steps()
            if "closure" in a.phases:
                self.ph_closure()
            if a.frames:
                self.ph_frames(arm0)
            s = self.seg("open_end", "open")
            self.hold_hands(bjm.DEX3_OPEN, bjm.DEX3_OPEN, 1.5)
            s["t1"] = time.monotonic()
        finally:
            try:
                self.arm.end()
                if self.arm.handle is not None:
                    self.arm.handle.wait(8.0)
                    self.notes["arm_op"] = self.arm.handle.summary()
            except Exception as e:  # noqa: BLE001
                self.notes["errors"].append(f"end: {e!r}")
        s = self.seg("rest_end", "rest")
        time.sleep(3.0)
        s["t1"] = time.monotonic()
        self.poll.stop_ev.set()
        self.poll.join(3.0)
        self.notes["rest_end"] = self.snapshot()
        self.notes["p1_stats_end"] = self.p1.try_call("get_stats")
        self.notes["status_end"] = {k: self.bc.status().get(k) for k in ("fault", "in_control", "active")}
        self.notes["poll_errors"] = self.poll.errors
        self.save()
        self.bc.close()

    def ph_steps(self):
        hold = self.a.hold
        for k, suf in enumerate(SUF):
            base = {s: [0.0] * 7 for s in SIDES}
            for s in SIDES:
                base[s][k] = step_base(s, k)
            sg = self.seg(f"base_{suf}", "settle", joint=k)
            self.hold_hands(base["left"], base["right"], hold)
            sg["t1"] = time.monotonic()
            for d in (+STEP, 0.0, -STEP, 0.0):
                goal = {s: list(base[s]) for s in SIDES}
                for s in SIDES:
                    goal[s][k] = base[s][k] + d
                sg = self.seg(f"step_{suf}_{'+' if d > 0 else '-' if d < 0 else '0'}", "step", joint=k, delta=d,
                              goal={s: goal[s][k] for s in SIDES})
                self.hold_hands(goal["left"], goal["right"], hold)
                sg["t1"] = time.monotonic()

    def ph_closure(self):
        for frac in (0.0, 0.5, 1.0, 0.0):
            sg = self.seg(f"closure_{frac:g}", "closure", closure=frac)
            self.hold_hands(bjm.hand_closure("left", frac), bjm.hand_closure("right", frac), self.a.closure_hold)
            sg["t1"] = time.monotonic()

    def ph_frames(self, arm0: dict):
        from PIL import Image
        fp = frame_pose()
        up0 = dict(self.upper)
        sg = self.seg("frames_raise", "move")
        self.stream(2.5, None, lambda t: {n: up0[n] + (fp[n] - up0[n]) * minjerk(t / 2.0) for n in ARM})
        sg["t1"] = time.monotonic()
        vdir = os.path.join(self.a.out, "frames")
        os.makedirs(vdir, exist_ok=True)
        shots = [("open", bjm.DEX3_OPEN, bjm.DEX3_OPEN),
                 ("fist", bjm.hand_closure("left", 1.0), bjm.hand_closure("right", 1.0)),
                 ("thumbs_rotated", [0.6, 0.6, 0.0, 0, 0, 0, 0], [-0.6, -0.6, 0.0, 0, 0, 0, 0]),
                 ("thumbs_closed", [0.0, 0.5, 1.2, 0, 0, 0, 0], [0.0, -0.5, -1.2, 0, 0, 0, 0])]
        self.notes["frames"] = []
        for name, L, R in shots:
            sg = self.seg(f"frame_{name}", "frame")
            self.hold_hands(L, R, 2.0)
            _, img = self.bc.camera_frame(timeout=0.1)
            js = self.snapshot()
            if img is not None:
                p = os.path.join(vdir, f"ego_{name}.png")
                Image.fromarray(img).save(p)
                self.notes["frames"].append({"name": name, "path": p, "hands": js})
            self.hold_hands(L, R, 0.2)
            sg["t1"] = time.monotonic()
        sg = self.seg("frames_lower", "move")
        self.stream(2.5, lambda t: (bjm.DEX3_OPEN, bjm.DEX3_OPEN),
                    lambda t: {n: fp[n] + (arm0[n] - fp[n]) * minjerk(t / 2.0) for n in ARM})
        sg["t1"] = time.monotonic()

    def save(self):
        P = self.poll.rows
        np.savez_compressed(
            os.path.join(self.a.out, "probe.npz"),
            t=np.array([r[0] for r in P]), q=np.array([r[1] for r in P]), qt=np.array([r[2] for r in P]),
            arm_q=np.array([r[3] for r in P]),
            ct=np.array([c[0] for c in self.cmd]), cl=np.array([c[1] for c in self.cmd]),
            cr=np.array([c[2] for c in self.cmd]), cseg=np.array([c[3] for c in self.cmd]))
        with open(os.path.join(self.a.out, "run.json"), "w") as f:
            json.dump({"segments": self.segments, "notes": self.notes, "args": vars(self.a)}, f, indent=1,
                      default=str)


# ======================================================================================================
# analysis
# ======================================================================================================
def _band_time(t, q, goal, t_ref, t_end):
    """Seconds after t_ref from which |q - goal| <= BAND until t_end (None if never)."""
    m = (t >= t_ref) & (t <= t_end)
    if not m.any():
        return None
    tt, e = t[m], np.abs(q[m] - goal)
    out = np.where(e > BAND)[0]
    if len(out) == 0:
        return float(tt[0] - t_ref)
    if out[-1] == len(e) - 1:
        return None
    return float(tt[out[-1] + 1] - t_ref)


def analyze(out: str) -> dict:
    run = json.load(open(os.path.join(out, "run.json")))
    D = np.load(os.path.join(out, "probe.npz"))
    t, q, qt = D["t"], D["q"], D["qt"]
    segs = run["segments"]
    M: dict = {"steps": [], "closure": [], "rest": {}, "pass": {}}
    for s in segs:
        if s["kind"] != "step" or not s.get("t1"):
            continue
        k = s["joint"]
        row = {"seg": s["name"], "joint": SUF[k], "delta": s["delta"]}
        for si, side in enumerate(SIDES):
            j = si * 7 + k
            goal = s["goal"][side]
            m = (t >= s["t0"]) & (t <= s["t1"])
            q0 = float(q[m][0, j]) if m.any() else math.nan
            # when P1's target for this joint first moved (by > 0.01 rad from its value before the send)
            before = np.where(t < s["t0"])[0]
            ref = qt[before[-1], j] if len(before) else (qt[m][0, j] if m.any() else math.nan)
            tm = np.where(m & (np.abs(qt[:, j] - ref) > 0.01))[0]
            t_p1 = float(t[tm[0]]) if len(tm) else s["t0"]
            last = m & (t >= s["t1"] - 0.3)
            ss = float(np.mean(q[last, j]) - goal) if last.any() else math.nan
            sgn = 1.0 if goal >= q0 else -1.0
            over = float(max(0.0, np.max(sgn * (q[m, j] - goal)))) if m.any() else math.nan
            tb = _band_time(t, q[:, j], goal, s["t0"], s["t1"])
            tb1 = _band_time(t, q[:, j], goal, t_p1, s["t1"])
            row[side] = {"goal": round(goal, 3), "start": round(q0, 3), "t_band_s": None if tb is None else round(tb, 3),
                         "t_band_p1_s": None if tb1 is None else round(tb1, 3),
                         "p1_target_delay_s": round(t_p1 - s["t0"], 3), "ss_err_rad": round(ss, 4),
                         "overshoot_rad": round(over, 4),
                         "p1_target_end": round(float(qt[last, j].mean()), 4) if last.any() else None,
                         "pass": tb is not None and tb < T_PASS and abs(ss) <= BAND}
        M["steps"].append(row)
    for s in segs:
        if s["kind"] != "closure" or not s.get("t1"):
            continue
        row = {"seg": s["name"], "closure_cmd": s["closure"]}
        last = (t >= s["t1"] - 0.5) & (t <= s["t1"])
        for si, side in enumerate(SIDES):
            qm = q[last, si * 7:(si + 1) * 7].mean(0)
            c = bjm.hand_closure_of(side, qm)
            cmdv = np.array(bjm.hand_closure(side, s["closure"]))
            row[side] = {"closure_measured": round(float(c), 3),
                         "ratio": round(float(c / s["closure"]), 3) if s["closure"] > 0 else None,
                         "joint_err_rad": dict(zip(SUF, np.round(qm - cmdv, 3).tolist())),
                         "max_abs_joint_err_rad": round(float(np.abs(qm - cmdv).max()), 3)}
        M["closure"].append(row)
    for key in ("rest_start", "rest_end"):
        M["rest"][key] = run["notes"].get(key)
    # pass/fail
    st = [r[s] for r in M["steps"] for s in SIDES]
    M["pass"]["steps_n"] = len(st)
    M["pass"]["steps_ok"] = sum(1 for x in st if x["pass"])
    tb = [x["t_band_s"] for x in st if x["t_band_s"] is not None]
    M["pass"]["t_band_max_s"] = max(tb) if tb else None
    M["pass"]["ss_err_max_rad"] = max(abs(x["ss_err_rad"]) for x in st) if st else None
    M["pass"]["overshoot_max_rad"] = max(x["overshoot_rad"] for x in st) if st else None
    cl = [(r["closure_cmd"], r[s]["ratio"]) for r in M["closure"] for s in SIDES if r["closure_cmd"] > 0]
    M["pass"]["closure_ratio_min"] = min(c[1] for c in cl) if cl else None
    # at rest the deploy commands its default fist: thumb_0/1 = 0 (joint_map.DEX3_DEPLOY_DEFAULT_*)
    for key in ("rest_start", "rest_end"):
        rs = run["notes"].get(key) or {}
        thumbs = [abs(rs[s]["q"][i]) for s in SIDES for i in (0, 1)] if rs else []
        M["pass"][f"{key}_thumb01_abs_max_rad"] = round(max(thumbs), 4) if thumbs else None
    M["pass"]["all"] = bool(st and M["pass"]["steps_ok"] == len(st)
                            and (not cl or M["pass"]["closure_ratio_min"] >= CLOSURE_PASS)
                            and (M["pass"]["rest_start_thumb01_abs_max_rad"] or 0.0) <= BAND)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1)
    try:
        plots(out, run, D, M)
    except Exception as e:  # noqa: BLE001
        print(f"[hand_probe] plots failed: {e!r}")
    return M


def plots(out, run, D, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, q, qt = D["t"], D["q"], D["qt"]
    ct, cl, cr = D["ct"], D["cl"], D["cr"]
    t0 = t[0]
    fig, ax = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    for si, side in enumerate(SIDES):
        c = cl if si == 0 else cr
        for k in range(7):
            a = ax[k, si]
            a.plot(ct - t0, c[:, k], color="0.6", lw=1.0, label="client command")
            a.plot(t - t0, qt[:, si * 7 + k], color="tab:orange", lw=0.9, label="P1 target (deploy write)")
            a.plot(t - t0, q[:, si * 7 + k], color="tab:blue", lw=1.2, label="P1 measured")
            a.set_ylabel(SUF[k], fontsize=8)
            a.grid(alpha=0.3)
            if k == 0:
                a.set_title(f"{side} hand")
    ax[0, 0].legend(fontsize=7, loc="best")
    for s in run["segments"]:
        if s["kind"] == "closure":
            for a in ax.flat:
                a.axvspan(s["t0"] - t0, (s["t1"] or s["t0"]) - t0, color="#f3e8ff", alpha=0.4, lw=0)
    ax[-1, 0].set_xlabel("s")
    ax[-1, 1].set_xlabel("s")
    p = M["pass"]
    fig.suptitle(f"Dex3 in P1 ({os.path.basename(out.rstrip('/'))}): steps ok {p['steps_ok']}/{p['steps_n']}, "
                 f"t_band max {p['t_band_max_s']} s, closure ratio min {p['closure_ratio_min']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hands_timeseries.png"), dpi=90)
    plt.close(fig)
    # step zoom: every step transition aligned at the send
    fig, ax = plt.subplots(2, 7, figsize=(20, 6), sharey=False)
    for s in run["segments"]:
        if s["kind"] != "step" or not s.get("t1"):
            continue
        k = s["joint"]
        for si, side in enumerate(SIDES):
            m = (t >= s["t0"] - 0.1) & (t <= s["t1"])
            if not m.any():
                continue
            goal = s["goal"][side]
            a = ax[si, k]
            up = goal > q[m, si * 7 + k][0]
            # clipped to the axes, so a joint that never gets there shows as a line on the edge
            a.plot(t[m] - s["t0"], np.clip(q[m, si * 7 + k] - goal, -0.44, 0.44), lw=1.0,
                   color="tab:red" if up else "tab:green")
            a.axhspan(-BAND, BAND, color="0.9")
            a.axvline(T_PASS, color="0.5", ls="--", lw=0.8)
            a.set_title(f"{side[0].upper()} {SUF[k]}", fontsize=9)
            a.set_ylim(-0.45, 0.45)
            a.grid(alpha=0.3)
    ax[1, 0].set_xlabel("s after send")
    ax[0, 0].set_ylabel("q - goal (rad)")
    p = M["pass"]
    fig.suptitle(f"{os.path.basename(out.rstrip('/'))}: 0.3 rad steps (red up, green down), P1 measured minus goal "
                 f"(clipped at +-0.44); band +-0.05 rad; dashed 0.5 s. ok {p['steps_ok']}/{p['steps_n']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "hands_steps.png"), dpi=90)
    plt.close(fig)


def compare(dirs: list[str]) -> dict:
    """Drift across runs: rest pose at start/end and per-joint step results."""
    rows = []
    for d in dirs:
        M = json.load(open(os.path.join(d, "metrics.json")))
        rows.append((d, M))
    out = {"runs": [d for d, _ in rows], "rest_start": {}, "rest_end": {}, "pass": [M["pass"] for _, M in rows]}
    for key in ("rest_start", "rest_end"):
        for side in SIDES:
            qs = np.array([M["rest"][key][side]["q"] for _, M in rows if M["rest"].get(key)])
            if len(qs):
                out[key][side] = {"per_run": qs.round(4).tolist(),
                                  "spread_rad": dict(zip(SUF, np.round(qs.max(0) - qs.min(0), 4).tolist()))}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--phases", default="steps,closure")
    ap.add_argument("--hold", type=float, default=1.2, help="hold per step (s)")
    ap.add_argument("--closure-hold", type=float, default=2.5)
    ap.add_argument("--poll-hz", type=float, default=40.0)
    ap.add_argument("--frames", action="store_true", help="ego frames with the palms raised in front of the camera")
    ap.add_argument("--analyze", metavar="DIR")
    ap.add_argument("--compare", nargs="+", metavar="DIR")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
        return
    if a.analyze:
        M = analyze(a.analyze)
        print(json.dumps(M["pass"], indent=1))
        return
    if not a.out:
        ap.error("--out is required for a live run")
    pr = Probe(a)
    pr.run()
    M = analyze(a.out)
    print("HAND_PROBE " + json.dumps(M["pass"]), flush=True)


if __name__ == "__main__":
    main()
