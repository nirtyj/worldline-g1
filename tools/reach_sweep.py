"""Reach re-calibration (owner reach): where the G1's palm really gets to through the body's `arm_script` op, live on
SONIC, and what that says about config/g1.yaml's workspace buffers.

    # box, stack up, under the stack lock, the robot standing in open space
    .venv/bin/python -m tools.reach_sweep sweep  --out outputs/reach_recal/sweep-$(date +%Y%m%d-%H%M%S)
    .venv/bin/python -m tools.reach_sweep turned --out outputs/reach_recal/turned-$(date +%Y%m%d-%H%M%S)
    .venv/bin/python -m tools.reach_sweep analyze outputs/reach_recal/sweep-* [--turned DIR] --out DIR

sweep   Per arm, per grasp-point height h (above the floor) and lateral offset towards the arm's own side, targets
        forward of the pelvis from --fwd-lo in --fwd-step steps. Every row starts from SONIC's own arms (a `retract`
        ending `stand`, then the pelvis at rest): that REST pelvis pose is the frame the targets are fixed in, in the
        world, exactly as services/reachability.py judges an object from the resting robot before the arm moves. Each
        target is one `arm_script grasp` with sonic_arm_script's own grasp arguments (closure 0.6, settle_s 2.0,
        max_ik_err_m 0.03; services/executors/sonic_arm_script.py ArmScriptConfig), no clear_z / avoid_boxes (the
        runtime passes none; free air). After the op (the settle's 2 s), the ACHIEVED palm is read from P1
        (`get_link_poses <arm>_palm`, the same point as body/g1_kin.py "palm"). A row stops extending forward after
        two consecutive failed points. Recorded per point: the body's reply (IK found?, ik_err_m, the palm goal it
        commanded in the world and pelvis frames), the achieved palm, its error to the target (world, and as a vector
        in the rest frame), the time for the palm to settle (gt.pose palm trace, 50 Hz: within 1 cm of where it
        ends, and within the tolerance of the target, to stay), pelvis shift (xy) and height change from the rest
        pose, pelvis tilt (max during the op and at its end), feet (ankle links) moved (a step), band / fallen /
        body fault.
turned  The far-stance posture (services/reachability.py find_far_stance: facing an edge normal turned up to
        far_turn_max_deg 45): the rest pelvis yawed by --turns from the virtual edge's normal towards the reaching
        arm, at --heights; targets on the edge normal through the arm's shoulder line, stepping further past a
        virtual edge that lies --standoff (stance_clearance_m 0.20) in front of the pelvis along that normal. Free
        air, same op and read-back as sweep.

Safety (the robot must never fall): a monitor on gt.pose checks every sample. Instability = pelvis tilt >= --tilt-ok
(5 deg), the elastic band on, fallen, pelvis below 0.70 m, a body fault, or the pelvis > --shift-abort from the rest
pose: the op is ended at once (`stop {arms: true}`: the script ends and the arm blends back to SONIC), the row ends,
and with tilt >= --tilt-abort (8 deg), band, fall or fault the whole run stops.

Reachable (analyze): the op succeeded, the achieved palm within --tol (2.5 cm) of the target, tilt < 5 deg all op
long, no fault / band.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GRASP_ARGS = {"closure": 0.6, "settle_s": 2.0, "max_ik_err_m": 0.03}   # sonic_arm_script ArmScriptConfig (grasp)
ANKLES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _r(v, n=4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, n) for x in v]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, n) if math.isfinite(f) else None


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(q) -> float:
    """Angle of the pelvis z axis from the world vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(quat_R(q)[2, 2])))))


def yaw_of(q) -> float:
    R = quat_R(q)
    return math.atan2(R[1, 0], R[0, 0])


def heading_frame(pose: dict) -> tuple[np.ndarray, np.ndarray]:
    """The rest frame targets are fixed in: the pelvis position and its HEADING (yaw only), as
    services/reachability.py judges an object (coords.world_to_body with x, y, yaw; heights in the world)."""
    yaw = yaw_of(pose["quat"])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([pose["x"], pose["y"], pose["z"]])


def to_world(p_b, frame) -> np.ndarray:
    R, t = frame
    return R @ np.asarray(p_b, float) + t


def to_body(p_w, frame) -> np.ndarray:
    R, t = frame
    return R.T @ (np.asarray(p_w, float) - t)


# ====================================================================== monitor
class Monitor:
    """gt.pose at 50 Hz: (recv_mono, t_sim, x, y, z, qw, qx, qy, qz, band, fallen, lpx, lpy, lpz, rpx, rpy, rpz,
    tilt_deg), plus the safety verdict."""

    def __init__(self, endpoint: str, ctx=None):
        from body.p1_client import PoseSub
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.sub = PoseSub(endpoint, ctx=ctx, on_pose=self._on)
        self.sub.start()

    def _on(self, p) -> None:
        d = p.raw
        lk = d.get("links") or {}
        lp = (lk.get("left_palm") or {}).get("pos") or [math.nan] * 3
        rp = (lk.get("right_palm") or {}).get("pos") or [math.nan] * 3
        row = (p.recv_mono, p.t_sim, p.x, p.y, p.z, *[float(v) for v in p.quat], float(bool(d.get("band"))),
               float(p.fallen), *[float(v) for v in lp], *[float(v) for v in rp], tilt_deg(p.quat))
        with self.lock:
            self.rows.append(row)

    def latest(self):
        with self.lock:
            return self.rows[-1] if self.rows else None

    def arr(self, t0=None, t1=None) -> np.ndarray:
        with self.lock:
            a = np.asarray(self.rows, float) if self.rows else np.zeros((0, 18))
        if len(a) and t0 is not None:
            a = a[a[:, 0] >= t0]
        if len(a) and t1 is not None:
            a = a[a[:, 0] <= t1]
        return a

    def wait(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = self.latest()
            if r is not None:
                return r
            time.sleep(0.05)
        raise RuntimeError("no gt.pose")


PALM_COL = {"left": slice(11, 14), "right": slice(14, 17)}


# ====================================================================== the live session
class Abort(Exception):
    pass


class Sweep:
    def __init__(self, a):
        from body.client import BodyClient
        from body.config import ep, ports as _ports
        from body.p1_client import P1Rpc
        import zmq
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.P = _ports(a.port_offset)
        self.ctx = zmq.Context.instance()
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0, ctx=self.ctx)
        self.mon = Monitor(ep(self.P["p1_pose"]), ctx=self.ctx)
        self.bc = BodyClient(port_offset=a.port_offset, ctx=self.ctx).connect(20)
        self.mon.wait()
        st = self.bc.status()
        self.start = {"status": {k: st.get(k) for k in ("mode", "in_control", "fault", "fences", "lease", "latched")},
                      "p1_stats": self.p1.try_call("get_stats"), "load": os.getloadavg(), "t_wall": time.time(),
                      "args": vars(a), "grasp_args": GRASP_ARGS}
        if st.get("fault"):
            raise RuntimeError(f"body fault {st['fault']}: not starting")
        if not st.get("in_control"):
            raise RuntimeError("body not in control: not starting (stand it first)")
        if st.get("lease"):
            raise RuntimeError(f"a lease is held ({st['lease']}): not starting")
        self.points_f = open(self.out / "points.jsonl", "a")
        self.rows_f = open(self.out / "rows.jsonl", "a")
        self.stream = f"reach-{uuid.uuid4().hex[:6]}"
        self.aborted: str | None = None
        self.n_points = 0
        self.origin: tuple[float, float] | None = None

    # ------------------------------------------------------------------ helpers
    def body_fault(self) -> str | None:
        st = self.bc.last_state or {}
        return st.get("fault")

    def pose_now(self) -> dict:
        r = self.mon.latest()
        return {"x": r[2], "y": r[3], "z": r[4], "quat": list(r[5:9]), "yaw": yaw_of(r[5:9]), "tilt": r[17]}

    def links(self, arm: str | None = None) -> dict:
        want = ["pelvis", *ANKLES] + ([f"{arm}_palm"] if arm else ["left_palm", "right_palm"])
        rep = self.p1.call("get_link_poses", links=want)
        return rep

    def unsafe(self, rest: dict | None) -> tuple[str | None, bool]:
        """(why, severe) from the latest gt.pose and body state."""
        r = self.mon.latest()
        if r is None:
            return "no_gt", True
        if r[9] > 0.5:
            return "band", True
        if r[10] > 0.5:
            return "fallen", True
        if r[4] < 0.70:
            return f"pelvis_z {r[4]:.3f}", True
        f = self.body_fault()
        if f:
            return f"fault {f}", True
        if r[17] >= self.a.tilt_abort:
            return f"tilt {r[17]:.1f} deg", True
        if r[17] >= self.a.tilt_ok:
            return f"tilt {r[17]:.1f} deg", False
        if rest is not None:
            sh = math.hypot(r[2] - rest["x"], r[3] - rest["y"])
            if sh > self.a.shift_abort:
                return f"pelvis shift {sh:.3f} m", False
        return None, False

    def stop_arms(self) -> str:
        """`stop {arms: true}`: the script / hold ends and the arms blend back to SONIC. The stream is stopped for
        good (arm_stopped), so the next ops use a new stream id."""
        s = self.bc.stop(arms=True, timeout=10)
        self.stream = f"reach-{uuid.uuid4().hex[:6]}"
        return s.state

    def arms_home(self, arm: str, label: str) -> dict:
        """Give the arm back to SONIC: `arm {release}` of the script's `target` hold (a min-jerk blend to SONIC's own
        arms, 1.5 s, the override dropped), then wait until the pelvis and the palm are at rest."""
        t0 = time.monotonic()
        st = self.bc.status()
        arm_st = st.get("arm") or {}
        res = {"label": label, "arm_mode_before": arm_st.get("mode")}
        if arm_st.get("mode") not in ("off", None):
            rep = self.bc.request("arm", {"stream": self.stream, "release": True})
            res["release"] = rep.get("ok"), rep.get("error")
            if not rep.get("ok"):
                res["stop_arms"] = self.stop_arms()
        # the blend back and the pelvis at rest
        res.update(self.wait_rest(arm))
        res["s"] = round(time.monotonic() - t0, 2)
        return res

    def wait_rest(self, arm: str, min_s: float = 1.8, max_s: float = 8.0) -> dict:
        t0 = time.monotonic()
        time.sleep(min_s)
        while time.monotonic() - t0 < max_s:
            a = self.mon.arr(time.monotonic() - 0.5)
            st = self.bc.status()
            mode = (st.get("arm") or {}).get("mode")
            if len(a) > 10 and mode in ("off", None):
                pv = np.ptp(a[:, 2:4], axis=0).max()
                pm = np.ptp(a[:, PALM_COL[arm]], axis=0).max()
                if pv < 0.01 and pm < 0.01:
                    return {"rest_wait_s": round(time.monotonic() - t0, 2), "arm_mode": mode}
            time.sleep(0.2)
        return {"rest_wait_s": round(time.monotonic() - t0, 2), "rest_timeout": True,
                "arm_mode": (self.bc.status().get("arm") or {}).get("mode")}

    def rest_frame(self, arm: str) -> dict:
        lp = self.links(arm)["links"]
        p = self.pose_now()
        return {**p, "ankles": {k: lp[k]["pos"] for k in ANKLES if k in lp}, "t_wall": time.time()}

    # ------------------------------------------------------------------ one point
    def point(self, arm: str, tgt_w: np.ndarray, rest: dict, meta: dict) -> dict:
        frame = heading_frame(rest)
        args = {"phase": "grasp", "arm": arm, "target_w": [float(v) for v in tgt_w], "hold_on_end": "target",
                "stream": self.stream, **GRASP_ARGS}
        t0 = time.monotonic()
        h = self.bc.submit("arm_script", args)
        t_rep = time.monotonic()
        plan = (h.reply or {}).get("data") or {}
        rec = {**meta, "arm": arm, "target_w": _r(tgt_w), "target_b_rest": _r(to_body(tgt_w, frame)),
               "rest": {k: _r(rest[k]) for k in ("x", "y", "z", "yaw", "tilt")},
               "reply_ok": bool((h.reply or {}).get("ok")), "reply_error": (h.reply or {}).get("error"),
               "ik_err_m": _r(plan.get("ik_err_m")), "ik_ms": plan.get("ik_ms"), "ik_seed": plan.get("ik_seed"),
               "goal_w": _r(plan.get("goal_w")), "goal_b": _r(plan.get("goal_b")), "move_s": plan.get("move_s"),
               "op_duration_s": plan.get("duration_s"), "reply_s": round(t_rep - t0, 3)}
        rec["ik_found"] = rec["reply_ok"] or (rec["reply_error"] not in (None, "ik_unreachable", "ik_timeout",
                                                                           "ik_unavailable"))
        why = None
        severe = False
        if rec["reply_ok"]:
            while not h.done():
                w, sv = self.unsafe(rest)
                if w is not None:
                    why, severe = w, sv
                    self.stop_arms()
                    break
                if time.monotonic() - t0 > 30:
                    why = "op_timeout"
                    self.stop_arms()
                    break
                time.sleep(0.02)
            try:
                h.wait(5)
            except TimeoutError:
                pass
        t1 = time.monotonic()
        res = h.result or {}
        rec["state"] = h.state
        rec["reason"] = res.get("reason") or res.get("error") or rec["reply_error"]
        if not rec["reply_ok"] and res.get("ik_err_m") is not None:
            rec["ik_err_m"] = _r(res.get("ik_err_m"))
        rec["ik_found"] = rec["reply_ok"] or (rec["reason"] != "ik_unreachable")
        for k in ("palm_err_w_m", "palm_err_b_m"):
            if isinstance(res.get(k), dict):
                rec[k] = res[k]
        for k in ("pelvis_shift_m", "track_updates", "track_submits", "track_errors", "palm_final_w", "palm_final_b"):
            if res.get(k) is not None:
                rec["body_" + k] = res[k]
        rec["stopped"] = why
        # the achieved palm (P1), after the settle
        lp = self.links(arm)
        L = lp["links"]
        palm = np.asarray(L[f"{arm}_palm"]["pos"], float)
        pel = L.get("pelvis")
        if pel is None:                                   # a P1 without the pelvis body: gt.pose's base
            r = self.mon.latest()
            pel = {"pos": [r[2], r[3], r[4]], "quat_wxyz": list(r[5:9])}
        err = palm - tgt_w
        rec["t_sim"] = lp.get("t_sim")
        rec["palm_w"] = _r(palm)
        rec["palm_b_rest"] = _r(to_body(palm, frame))
        now = heading_frame({"x": pel["pos"][0], "y": pel["pos"][1], "z": pel["pos"][2], "quat": pel["quat_wxyz"]})
        rec["palm_b_now"] = _r(to_body(palm, now))
        rec["err_m"] = _r(float(np.linalg.norm(err)))
        rec["err_b_rest"] = _r(frame[0].T @ err)                  # (fwd, left, up) of the miss, rest frame
        rec["err_to_goal_m"] = None if rec["goal_w"] is None else \
            _r(float(np.linalg.norm(palm - np.asarray(rec["goal_w"], float))))
        rec["pelvis_end"] = _r(pel["pos"])
        rec["pelvis_shift_m"] = _r(math.hypot(pel["pos"][0] - rest["x"], pel["pos"][1] - rest["y"]))
        rec["pelvis_dz_m"] = _r(pel["pos"][2] - rest["z"])
        rec["pelvis_dyaw_deg"] = _r(math.degrees(math.atan2(math.sin(yaw_of(pel["quat_wxyz"]) - rest["yaw"]),
                                                            math.cos(yaw_of(pel["quat_wxyz"]) - rest["yaw"]))), 2)
        rec["tilt_end_deg"] = _r(tilt_deg(pel["quat_wxyz"]), 2)
        a = self.mon.arr(t0, t1 + 0.05)
        rec["tilt_max_deg"] = _r(float(a[:, 17].max()) if len(a) else None, 2)
        rec["band_seen"] = bool(len(a) and a[:, 9].max() > 0.5)
        rec["fallen_seen"] = bool(len(a) and a[:, 10].max() > 0.5)
        rec["fault"] = self.body_fault()
        ank = rest.get("ankles") or {}
        rec["feet_moved_m"] = _r(max((math.hypot(L[k]["pos"][0] - ank[k][0], L[k]["pos"][1] - ank[k][1])
                                      for k in ANKLES if k in L and k in ank), default=None))
        # palm settling from the gt.pose trace
        if len(a) > 5:
            tt = a[:, 0] - t0
            pp = a[:, PALM_COL[arm]]
            fin = pp[-1]
            d_fin = np.linalg.norm(pp - fin, axis=1)
            d_tgt = np.linalg.norm(pp - tgt_w, axis=1)

            def settle(d, lim):
                bad = np.nonzero(d > lim)[0]
                if not len(bad):
                    return 0.0
                k = bad[-1] + 1
                return None if k >= len(d) else round(float(tt[k]), 2)
            rec["t_settle_1cm_s"] = settle(d_fin, 0.01)
            rec["t_in_tol_s"] = settle(d_tgt, self.a.tol)
            rec["palm_trace_n"] = int(len(a))
        rec["t_op_s"] = round(t1 - t0, 2)
        stable = bool((rec["tilt_max_deg"] or 0) < self.a.tilt_ok and not rec["fault"] and not rec["band_seen"]
                      and not rec["fallen_seen"] and why is None)
        rec["reachable"] = bool(rec["state"] == "succeeded" and rec["err_m"] is not None and rec["err_m"] <= self.a.tol
                                and stable)
        # the pick pipeline's own gates (sonic_arm_script): the grasp's IK within max_ik_err_m (the op ran) and the
        # GT palm within attach_max_m 0.10 of the grasp point
        rec["gate_ok"] = bool(rec["state"] == "succeeded" and rec["err_m"] is not None
                              and rec["err_m"] <= self.a.gate_m and stable)
        self.points_f.write(json.dumps(rec) + "\n")
        self.points_f.flush()
        self.n_points += 1
        print(f"[reach] {arm[0]} h {meta.get('h'):.2f} lat {meta.get('lat_mag'):.2f} fwd {meta.get('fwd', 0):.3f} "
              f"{rec['state'][:4]:4s} ik {rec['ik_err_m']} err {rec['err_m']} tilt {rec['tilt_max_deg']} "
              f"shift {rec['pelvis_shift_m']} feet {rec['feet_moved_m']} t {rec['t_op_s']} "
              f"{'OK' if rec['reachable'] else ('g ' if rec['gate_ok'] else '--')} {rec['reason'] or ''} {why or ''}",
              flush=True)
        if why is not None:
            if severe:
                self.aborted = why
                raise Abort(why)
            rec["_row_stop"] = why
        return rec

    # ------------------------------------------------------------------ the pick's pregrasp
    def pregrasp(self, arm: str, tgt_w: np.ndarray, rest: dict) -> dict:
        """sonic_arm_script's pregrasp (settle_s 1.2, max_ik_err_m 0.03; the body's standoff 0.10 back along the
        horizontal approach and 0.05 up), with the same safety watch as a grasp."""
        args = {"phase": "pregrasp", "arm": arm, "target_w": [float(v) for v in tgt_w], "hold_on_end": "target",
                "stream": self.stream, "settle_s": 1.2, "max_ik_err_m": 0.03}
        t0 = time.monotonic()
        h = self.bc.submit("arm_script", args)
        plan = (h.reply or {}).get("data") or {}
        why, severe = None, False
        if (h.reply or {}).get("ok"):
            while not h.done():
                w, sv = self.unsafe(rest)
                if w is not None:
                    why, severe = w, sv
                    self.stop_arms()
                    break
                if time.monotonic() - t0 > 30:
                    why = "op_timeout"
                    self.stop_arms()
                    break
                time.sleep(0.02)
            try:
                h.wait(5)
            except TimeoutError:
                pass
        res = h.result or {}
        a = self.mon.arr(t0, time.monotonic())
        out = {"state": h.state, "reason": res.get("reason") or res.get("error") or (h.reply or {}).get("error"),
               "ik_err_m": _r(plan.get("ik_err_m") if plan.get("ik_err_m") is not None else res.get("ik_err_m")),
               "goal_b": _r(plan.get("goal_b")), "palm_err_w_m": res.get("palm_err_w_m"),
               "t_s": round(time.monotonic() - t0, 2), "stopped": why,
               "tilt_max_deg": _r(float(a[:, 17].max()) if len(a) else None, 2)}
        if why is not None and severe:
            self.aborted = why
            raise Abort(why)
        return out

    def record_fail(self, meta: dict, arm: str, tgt_w: np.ndarray, rest: dict, reason: str) -> dict:
        frame = heading_frame(rest)
        rec = {**meta, "arm": arm, "target_w": _r(tgt_w), "target_b_rest": _r(to_body(tgt_w, frame)),
               "rest": {k: _r(rest[k]) for k in ("x", "y", "z", "yaw", "tilt")}, "state": "failed",
               "reason": reason, "reply_ok": False, "ik_found": None, "err_m": None, "reachable": False,
               "gate_ok": False, "stopped": (meta.get("pre") or {}).get("stopped"),
               "tilt_max_deg": (meta.get("pre") or {}).get("tilt_max_deg")}
        self.points_f.write(json.dumps(rec) + "\n")
        self.points_f.flush()
        self.n_points += 1
        print(f"[reach] {arm[0]} h {meta.get('h'):.2f} lat {meta.get('lat_mag'):.2f} fwd {meta.get('fwd', 0):.3f} "
              f"FAIL {reason}", flush=True)
        return rec

    # ------------------------------------------------------------------ a row
    def row(self, arm: str, h: float, lat_mag: float, fwds, meta_extra: dict | None = None,
            target_fn=None) -> dict:
        """Targets fixed in the rest frame; stop after two consecutive failures (or an instability)."""
        home = self.arms_home(arm, "row_start")
        rest = self.rest_frame(arm)
        if self.origin is None:
            self.origin = (rest["x"], rest["y"])
        drift = math.hypot(rest["x"] - self.origin[0], rest["y"] - self.origin[1])
        if drift > self.a.drift_abort:
            self.aborted = f"drifted {drift:.2f} m from the sweep start"
            raise Abort(self.aborted)
        frame = heading_frame(rest)
        side = -1.0 if arm == "right" else 1.0
        z_b = self.a.floor + h - rest["z"]
        recs, fails = [], 0
        stop = None
        for f in fwds:
            if target_fn is None:
                tb = np.array([f, side * lat_mag, z_b])
            else:
                tb = target_fn(f, rest, z_b)
            tw = to_world(tb, frame)
            meta = {"h": h, "lat_mag": lat_mag, "lat": round(side * lat_mag, 3), "fwd": round(float(f), 4),
                    **(meta_extra or {})}
            rec = self.point(arm, tw, rest, meta)
            recs.append(rec)
            if rec.get("_row_stop"):
                stop = rec["_row_stop"]
                break
            # prune: two consecutive points that fail even the pipeline's gate (so the rows also cover the band
            # between the tolerance's envelope and the IK / gate envelope; beyond the IK boundary the body rejects
            # without moving)
            fails = 0 if (rec["reachable"] or rec["gate_ok"]) else fails + 1
            if fails >= 2:
                stop = "two_fails"
                break
        ok = [r["fwd"] for r in recs if r["reachable"]]
        gok = [r["fwd"] for r in recs if r["gate_ok"]]
        row = {"arm": arm, "h": h, "lat_mag": lat_mag, **(meta_extra or {}), "n": len(recs), "stop": stop,
               "fwd_max_ok": max(ok) if ok else None, "fwd_ok": ok, "fwd_gate_ok": gok, "home": home,
               "rest": {k: _r(rest[k]) for k in ("x", "y", "z", "yaw", "tilt")}}
        self.rows_f.write(json.dumps(row) + "\n")
        self.rows_f.flush()
        return row

    def finish(self, extra: dict) -> None:
        end = {"p1_stats": self.p1.try_call("get_stats"), "load": os.getloadavg(), "t_wall": time.time(),
               "status": {k: (self.bc.status() or {}).get(k) for k in ("mode", "in_control", "fault", "latched")}}
        a = self.mon.arr()
        np.savez_compressed(self.out / "gt_trace.npz", rows=a,
                            cols=np.array(["recv_mono", "t_sim", "x", "y", "z", "qw", "qx", "qy", "qz", "band",
                                           "fallen", "lpx", "lpy", "lpz", "rpx", "rpy", "rpz", "tilt_deg"]))
        rep = {"start": self.start, "end": end, "aborted": self.aborted, "points": self.n_points,
               "tilt_max_deg_all": _r(float(a[:, 17].max()) if len(a) else None, 2),
               "band_seen": bool(len(a) and a[:, 9].max() > 0.5), "fallen_seen": bool(len(a) and a[:, 10].max() > 0.5),
               **extra}
        (self.out / "session.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
        self.points_f.close()
        self.rows_f.close()
        print("[reach] done " + json.dumps({k: rep[k] for k in ("aborted", "points", "tilt_max_deg_all", "band_seen",
                                                                "fallen_seen")}), flush=True)


def _frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def cmd_sweep(a) -> int:
    S = Sweep(a)
    heights = [float(v) for v in a.heights.split(",")]
    lats = [float(v) for v in a.lats.split(",")]
    arms = [s for s in a.arms.split(",") if s]
    fwds = _frange(a.fwd_lo, a.fwd_hi, a.fwd_step)
    done = set()
    donef = S.out / "rows.jsonl"
    if a.resume and donef.exists():
        for ln in donef.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["arm"], r["h"], r["lat_mag"]))
    last_arm = None
    try:
        for h in heights:
            for arm in arms:
                for lat in lats:
                    if (arm, h, lat) in done:
                        continue
                    if last_arm and last_arm != arm:
                        S.arms_home(last_arm, "switch_arm")
                    last_arm = arm
                    S.row(arm, h, lat, fwds)
    except Abort as e:
        print(f"[reach] ABORT {e}", flush=True)
        try:
            S.stop_arms()
        except Exception:  # noqa: BLE001
            pass
    except KeyboardInterrupt:
        S.aborted = "interrupted"
        S.stop_arms()
    finally:
        try:
            if last_arm:
                S.arms_home(last_arm, "end")
        except Exception:  # noqa: BLE001
            pass
        S.finish({"kind": "sweep"})
    return 0


def cmd_turned(a) -> int:
    """Far-stance posture: the virtual edge's normal is the rest heading turned by -turn towards the arm (the robot
    stands turned `turn` off the edge normal, the far search's yaw), the pelvis a.standoff from the edge along it;
    targets on the edge normal through the arm's shoulder line (lateral a.shoulder_lat on the arm's side, measured
    ALONG the edge), stepping past the edge."""
    S = Sweep(a)
    heights = [float(v) for v in a.heights.split(",")]
    turns = [float(v) for v in a.turns.split(",")]
    arms = [s for s in a.arms.split(",") if s]
    pasts = _frange(a.past_lo, a.past_hi, a.fwd_step)
    last_arm = None
    try:
        for arm in arms:
            side = -1.0 if arm == "right" else 1.0
            for turn in turns:
                for h in heights:
                    # robot turned `turn` deg AWAY from the arm's side relative to the edge normal, i.e. the edge
                    # normal n (pointing from the robot to the edge) is the heading rotated by side*turn: the
                    # object ahead-and-to-the-arm's-side, the reach across a corner
                    th = math.radians(turn) * side
                    n_b = np.array([math.cos(th), math.sin(th)])
                    t_b = np.array([-n_b[1], n_b[0]])          # along the edge (left of n)

                    def tfn(past, rest, z_b, n_b=n_b, t_b=t_b):
                        # the point where the pelvis' perpendicular meets the edge, then along the edge by the
                        # shoulder offset (on the arm's side), then `past` beyond the edge
                        along = float(np.dot([0.0, side * a.shoulder_lat], t_b))
                        xy = n_b * (a.standoff + past) + t_b * along
                        return np.array([xy[0], xy[1], z_b])
                    if last_arm and last_arm != arm:
                        S.arms_home(last_arm, "switch_arm")
                    last_arm = arm
                    S.row(arm, h, 0.0, pasts, meta_extra={"turn_deg": turn, "standoff": a.standoff,
                                                         "kind": "turned"}, target_fn=tfn)
    except Abort as e:
        print(f"[reach] ABORT {e}", flush=True)
        try:
            S.stop_arms()
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if last_arm:
                S.arms_home(last_arm, "end")
        except Exception:  # noqa: BLE001
            pass
        S.finish({"kind": "turned"})
    return 0


def cmd_refine(a) -> int:
    """The boundary at 1 cm, pick-faithful and repeated: for rows of a sweep (--base DIR), targets at the sweep's
    last in-tolerance forward reach + --offsets, each tried --trials times as the runtime's pick does it: from SONIC's
    own arms at rest, sonic_arm_script's pregrasp, then its grasp (same args as the sweep), the palm read from P1."""
    S = Sweep(a)
    rows = [json.loads(ln) for ln in (Path(a.base) / "rows.jsonl").read_text().splitlines() if ln.strip()]
    base = {(r["arm"], round(r["h"], 3), round(r["lat_mag"], 3)): r for r in rows}
    heights = [float(v) for v in a.heights.split(",")]
    lats = [float(v) for v in a.lats.split(",")]
    arms = [s_ for s_ in a.arms.split(",") if s_]
    offs = [float(v) for v in a.offsets.split(",")]
    last_arm = None
    try:
        for h in heights:
            for arm in arms:
                side = -1.0 if arm == "right" else 1.0
                for lat in lats:
                    r = base.get((arm, round(h, 3), round(lat, 3)))
                    ok = sorted(r["fwd_ok"]) if r else []
                    if not ok:
                        continue
                    f0 = ok[0]                          # the contiguous in-tolerance run from the inside
                    for f in ok[1:]:
                        if f - f0 > 0.026:
                            break
                        f0 = f
                    for off in offs:
                        for k in range(a.trials):
                            if last_arm and last_arm != arm:
                                S.arms_home(last_arm, "switch_arm")
                            last_arm = arm
                            S.arms_home(arm, "trial_start")
                            rest = S.rest_frame(arm)
                            frame = heading_frame(rest)
                            fwd = round(f0 + off, 4)
                            tw = to_world([fwd, side * lat, a.floor + h - rest["z"]], frame)
                            meta = {"h": h, "lat_mag": lat, "lat": round(side * lat, 3), "fwd": fwd, "kind": "refine",
                                    "offset": off, "trial": k, "sweep_fwd_ok": f0}
                            pre = S.pregrasp(arm, tw, rest)
                            meta["pre"] = pre
                            if pre["state"] != "succeeded" or pre.get("stopped"):
                                S.record_fail(meta, arm, tw, rest, f"pregrasp:{pre.get('stopped') or pre['reason']}")
                                continue
                            S.point(arm, tw, rest, meta)
    except Abort as e:
        print(f"[reach] ABORT {e}", flush=True)
        try:
            S.stop_arms()
        except Exception:  # noqa: BLE001
            pass
    except KeyboardInterrupt:
        S.aborted = "interrupted"
        S.stop_arms()
    finally:
        try:
            if last_arm:
                S.arms_home(last_arm, "end")
        except Exception:  # noqa: BLE001
            pass
        S.finish({"kind": "refine", "base": a.base})
    return 0


# ====================================================================== analysis
def load_points(dirs) -> list[dict]:
    out = []
    for d in dirs:
        p = Path(d) / "points.jsonl"
        if p.exists():
            out += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return out


def fit_sphere(pts: np.ndarray) -> dict:
    from world.workspace_cal import fit_sphere as fs
    return fs(pts)


WS_KEYS = ("shoulder_z_m", "shoulder_lat_m", "shoulder_fwd_m", "arm_reach_m", "reach_fwd_m", "reach_lat_max_m",
           "stance_margin_m", "stance_clearance_m", "obj_z_min_m", "obj_z_max_m", "grasp_above_top_m",
           "far_stand_off_m", "far_turn_max_deg")


def load_ws(path: str | Path = Path(ROOT) / "config" / "g1.yaml") -> dict:
    """config/g1.yaml `workspace` (the calibrated arm, not lite_world): yaml when importable, else a line parse."""
    txt = Path(path).read_text()
    try:
        import yaml
        ws = dict((yaml.safe_load(txt) or {}).get("workspace") or {})
        ws.pop("lite_world", None)
        return {k: ws[k] for k in WS_KEYS if k in ws}
    except ImportError:
        pass
    out, inside = {}, False
    for ln in txt.splitlines():
        if ln.startswith("workspace:"):
            inside = True
            continue
        if inside and ln and not ln.startswith(" "):
            break
        if inside and ln.startswith("  lite_world"):
            break
        s = ln.split("#", 1)[0].strip()
        if inside and ":" in s:
            k, v = [x.strip() for x in s.split(":", 1)]
            if k in WS_KEYS and v:
                out[k] = json.loads(v) if v.startswith("[") else (v if v in ("true", "false") else float(v))
    return out


def cfg_fwd_max(ws: dict, h: float, lat_mag: float, marg: float = 0.0, r: float | None = None) -> float | None:
    """The forward reach services/reachability.py allows at grasp height h (above the floor, floor 0) and lateral
    |lat| on the arm's side: the window's forward max and the shoulder sphere (radius r, default arm_reach_m), both
    less `marg` (a stance's stance_margin_m)."""
    r = float(ws["arm_reach_m"]) if r is None else r
    h2 = (r - marg) ** 2 - (lat_mag - float(ws["shoulder_lat_m"])) ** 2 - (h - float(ws["shoulder_z_m"])) ** 2
    if h2 < 0 or lat_mag > float(ws["reach_lat_max_m"]) - marg + 1e-9:
        return None
    return min(float(ws["reach_fwd_m"][1]) - marg, float(ws.get("shoulder_fwd_m", 0.0)) + math.sqrt(h2))


def cfg_horiz(ws: dict, h: float, r: float | None = None, fwd_hi: float | None = None,
              lat_max: float | None = None) -> float:
    """services/reachability.py horizontal_reach(z): the farthest pelvis-to-palm horizontal distance in the window and
    the sphere at height h."""
    r = float(ws["arm_reach_m"]) if r is None else r
    hi = float(ws["reach_fwd_m"][1]) if fwd_hi is None else fwd_hi
    lm = float(ws["reach_lat_max_m"]) if lat_max is None else lat_max
    best = 0.0
    for i in range(int(round(lm / 0.02)) + 1):
        lat = i * 0.02
        h2 = r ** 2 - (lat - float(ws["shoulder_lat_m"])) ** 2 - (h - float(ws["shoulder_z_m"])) ** 2
        if h2 < 0:
            continue
        f = min(hi, float(ws.get("shoulder_fwd_m", 0.0)) + math.sqrt(h2))
        if f >= float(ws["reach_fwd_m"][0]):
            best = max(best, math.hypot(f, lat))
    return best


def shoulder_r(ws: dict, fwd: float, lat_mag: float, h: float) -> float:
    return math.sqrt((fwd - float(ws.get("shoulder_fwd_m", 0.0))) ** 2 + (lat_mag - float(ws["shoulder_lat_m"])) ** 2
                     + (h - float(ws["shoulder_z_m"])) ** 2)


def analyze(a) -> int:
    allp = load_points(a.dirs)
    pts = [p for p in allp if p.get("kind") not in ("turned", "refine")]
    refine = [p for p in allp if p.get("kind") == "refine"]
    tol = a.tol
    ws = load_ws(a.config) if a.config else load_ws()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def stable(p):
        return bool((p.get("tilt_max_deg") or 0) < 5.0 and not p.get("fault") and not p.get("band_seen")
                    and not p.get("fallen_seen") and not p.get("stopped"))

    def ok(p, t=tol):
        return bool(p["state"] == "succeeded" and p.get("err_m") is not None and p["err_m"] <= t and stable(p))

    crit = {"tol": lambda p: ok(p), "gate": lambda p: ok(p, a.gate_m), "ik": lambda p: bool(p.get("reply_ok"))}
    rep: dict = {"n_points": len(pts), "tol_m": tol, "gate_m": a.gate_m, "config_workspace": ws, "arms": {},
                 "criteria": {"tol": f"op succeeded, P1 palm within {tol} m of the target, tilt < 5 deg, no fault",
                              "gate": f"sonic_arm_script's own gates: the IK accepted (max_ik_err_m 0.03) and the P1 "
                                      f"palm within attach_max_m {a.gate_m} m, tilt < 5 deg, no fault",
                              "ik": "the body's IK accepted the goal (ik_err <= 0.03), wherever the palm went"}}
    radii_all: dict = {k: [] for k in crit}
    for arm in sorted({p["arm"] for p in pts}):
        P = [p for p in pts if p["arm"] == arm]
        hs = sorted({p["h"] for p in P})
        lats = sorted({p["lat_mag"] for p in P})
        table, bnd, radii = {}, {k: [] for k in crit}, {k: [] for k in crit}
        for h in hs:
            row = {}
            for l in lats:
                R = sorted([p for p in P if p["h"] == h and p["lat_mag"] == l], key=lambda p: p["fwd"])
                if not R:
                    continue
                cell = {"n": len(R), "fwds": [p["fwd"] for p in R]}
                for k, pred in crit.items():
                    far = _run_max(R, pred)
                    nxt = next((p["fwd"] for p in R if far is not None and p["fwd"] > far + 1e-9), None)
                    cell[f"fwd_max_{k}"] = far
                    cell[f"fwd_fail_{k}"] = nxt                   # the first failing point past it (upper bound)
                    if far is not None and nxt is not None:       # a boundary the arm set (not the grid's end)
                        q = next(p for p in R if p["fwd"] == far)
                        bnd[k].append(q["target_b_rest"])
                        rr = (shoulder_r(ws, far, l, h), shoulder_r(ws, nxt, l, h))
                        radii[k].append({"h": h, "lat": l, "r_ok": round(rr[0], 4), "r_fail": round(rr[1], 4)})
                cfg = cfg_fwd_max(ws, h, l)
                cfg_st = cfg_fwd_max(ws, h, l, marg=float(ws.get("stance_margin_m", 0.02)))
                cell["cfg_fwd_max"] = None if cfg is None else round(cfg, 3)
                cell["cfg_stance_fwd_max"] = None if cfg_st is None else round(cfg_st, 3)
                if cfg is not None and cell["fwd_max_tol"] is not None:
                    cell["margin_tol_vs_cfg"] = round(cell["fwd_max_tol"] - cfg, 3)
                if cfg is not None and cell["fwd_max_gate"] is not None:
                    cell["margin_gate_vs_cfg"] = round(cell["fwd_max_gate"] - cfg, 3)
                # palm errors along the row (cm), for the table
                cell["err_cm"] = [None if p.get("err_m") is None or p["state"] != "succeeded"
                                  else round(p["err_m"] * 100, 1) for p in R]
                row[str(l)] = cell
            hz = {k: [math.hypot(v[f"fwd_max_{k}"], float(l)) for l, v in row.items()
                      if v[f"fwd_max_{k}"] is not None] for k in crit}
            table[str(h)] = {"best_fwd": {k: max([v[f"fwd_max_{k}"] for v in row.values()
                                                  if v[f"fwd_max_{k}"] is not None], default=None) for k in crit},
                             "horiz_max": {k: _r(max(hz[k]), 3) if hz[k] else None for k in crit},
                             "cfg_horiz": round(cfg_horiz(ws, h), 3),
                             "cfg_past_edge": round(max(0.0, cfg_horiz(ws, h) - float(ws["stance_clearance_m"])), 3),
                             "by_lat": row}
            for k in crit:
                hm = table[str(h)]["horiz_max"][k]
                table[str(h)][f"past_edge_{k}"] = None if hm is None else round(hm - float(ws["stance_clearance_m"]), 3)
        # past an edge the pelvis stands `stance_clearance_m` from, facing the edge normal turned by `turn`
        # towards the other arm (the object ahead and to this arm's side): max over the reached points of
        # fwd cos(turn) + |lat| sin(turn), minus the clearance; the config's the same over its window and sphere
        # (all of it, and only the grid's points: fwd >= the sweep's first target)
        sc = float(ws["stance_clearance_m"])
        pe = {}
        for h in hs:
            Q = [p for p in P if p["h"] == h]
            row = {}
            for th in (0, 15, 25, 35, 45):
                c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
                m = {k: max([p["fwd"] * c + p["lat_mag"] * s_ for p in Q if crit[k](p)], default=None) for k in crit}
                cf_all, cf_grid = None, None
                for f in np.arange(float(ws["reach_fwd_m"][0]), float(ws["reach_fwd_m"][1]) + 1e-9, 0.005):
                    for l in np.arange(0.0, float(ws["reach_lat_max_m"]) + 1e-9, 0.005):
                        if shoulder_r(ws, f, l, h) <= float(ws["arm_reach_m"]):
                            v = f * c + l * s_
                            cf_all = v if cf_all is None else max(cf_all, v)
                for p in Q:
                    if shoulder_r(ws, p["fwd"], p["lat_mag"], h) <= float(ws["arm_reach_m"]) and \
                            p["fwd"] <= float(ws["reach_fwd_m"][1]) + 1e-9:
                        v = p["fwd"] * c + p["lat_mag"] * s_
                        cf_grid = v if cf_grid is None else max(cf_grid, v)
                row[str(th)] = {**{k: None if v is None else round(v - sc, 3) for k, v in m.items()},
                                "cfg": None if cf_all is None else round(cf_all - sc, 3),
                                "cfg_grid": None if cf_grid is None else round(cf_grid - sc, 3)}
            pe[str(h)] = row
        good = [p for p in P if ok(p)]
        spheres = {k: (fit_sphere(np.asarray(bnd[k], float)) if len(bnd[k]) >= 8 else None) for k in crit}
        rstats = {k: {"r_ok": _pct_stats([x["r_ok"] for x in radii[k]]),
                      "r_fail": _pct_stats([x["r_fail"] for x in radii[k]]), "rows": radii[k]} for k in crit}
        for k in crit:
            radii_all[k] += radii[k]
        rep["arms"][arm] = {
            "table": table, "past_edge_by_turn": pe, "sphere_fit_rest_frame": spheres, "radius_about_cfg_shoulder": rstats,
            "n": len(P), "n_reachable_tol": len(good), "n_gate_ok": sum(ok(p, a.gate_m) for p in P),
            "n_ik_rejected": sum(p.get("reason") == "ik_unreachable" for p in P),
            "err_reachable_m": _pct_stats([p["err_m"] for p in good]),
            "err_all_succeeded_m": _pct_stats([p["err_m"] for p in P if p["state"] == "succeeded"]),
            "err_b_rest_mean_reachable": _r(np.mean([p["err_b_rest"] for p in good], axis=0)) if good else None,
            "t_settle_1cm_s": _pct_stats([p.get("t_settle_1cm_s") for p in good]),
            "t_in_tol_s": _pct_stats([p.get("t_in_tol_s") for p in good]),
            "t_op_s": _pct_stats([p.get("t_op_s") for p in P if p["state"] == "succeeded"]),
            "pelvis_shift_m": _pct_stats([p.get("pelvis_shift_m") for p in P if p["state"] == "succeeded"]),
            "pelvis_dz_m": _pct_stats([p.get("pelvis_dz_m") for p in P if p["state"] == "succeeded"]),
            "tilt_max_deg": _pct_stats([p.get("tilt_max_deg") for p in P]),
            "tilt_max_deg_reachable": _pct_stats([p.get("tilt_max_deg") for p in good]),
            "feet_moved_m": _pct_stats([p.get("feet_moved_m") for p in P]),
            "unstable_points": [{k: p.get(k) for k in ("h", "lat_mag", "fwd", "tilt_max_deg", "stopped", "fault",
                                                       "band_seen", "feet_moved_m")} for p in P if not stable(p)],
            "reasons": _count([p.get("reason") or ("ok" if ok(p) else ("gate" if ok(p, a.gate_m) else "err>gate"))
                               for p in P]),
        }
    rep["radius_about_cfg_shoulder_both_arms"] = {k: {"r_ok": _pct_stats([x["r_ok"] for x in v]),
                                                      "r_fail": _pct_stats([x["r_fail"] for x in v])}
                                                  for k, v in radii_all.items()}
    if refine:
        rep["refine"] = refine_summary(refine, ws, ok, a.gate_m)
    rep["proposal"] = propose(rep, ws)
    turned = [p for p in allp if p.get("kind") == "turned"] + \
        [p for p in load_points(a.turned or []) if p.get("kind") == "turned"]
    if turned:
        tt: dict = {}
        for p in turned:
            k = f"{p['arm']} turn {p['turn_deg']:g} h {p['h']:.2f}"
            tt.setdefault(k, []).append(p)
        rep["turned"] = {}
        for k, R in tt.items():
            R.sort(key=lambda p: p["fwd"])
            p0 = R[0]
            rep["turned"][k] = {
                "arm": p0["arm"], "turn_deg": p0["turn_deg"], "h": p0["h"], "standoff": p0.get("standoff"),
                "past_edge_max_tol": _run_max(R, crit["tol"]), "past_edge_max_gate": _run_max(R, crit["gate"]),
                "past_edge_max_ik": _run_max(R, crit["ik"]),
                "cfg_past_edge_at_h": round(max(0.0, cfg_horiz(ws, p0["h"]) - float(ws["stance_clearance_m"])), 3),
                "cfg_past_edge_on_line": max([p["fwd"] for p in R if _cfg_ok(ws, p)], default=None),
                "pts": [{"past": p["fwd"], "target_b": p["target_b_rest"], "cfg_ok": _cfg_ok(ws, p),
                         "err_cm": None if p.get("err_m") is None
                         else round(p["err_m"] * 100, 1), "ik_err": p.get("ik_err_m"), "tilt": p.get("tilt_max_deg"),
                         "state": p["state"], "reason": p.get("reason")} for p in R]}
    (out / "envelope.json").write_text(json.dumps(rep, indent=1) + "\n")
    _print_tables(rep)
    try:
        _plots(pts, rep, out, ok, a.gate_m, ws)
    except Exception as e:  # noqa: BLE001
        print("plots failed:", e)
    return 0


def _cfg_ok(ws: dict, p: dict) -> bool:
    """Would services/reachability.py call this target (pelvis frame at rest) in the window and the arm's sphere?"""
    f, l = p["target_b_rest"][0], abs(p["target_b_rest"][1])
    lo, hi = ws["reach_fwd_m"]
    return bool(lo <= f <= hi and l <= float(ws["reach_lat_max_m"]) and
                shoulder_r(ws, f, l, p["h"]) <= float(ws["arm_reach_m"]))


def refine_summary(R: list[dict], ws: dict, ok, gate_m: float) -> dict:
    """Per (arm, h, lat): every refine target's trials (pregrasp + grasp from rest), the fraction within the tolerance
    and within the gate, and the boundary = the largest forward reach at which every trial at it and at every nearer
    refine target passed."""
    cells: dict = {}
    for p in R:
        cells.setdefault((p["arm"], p["h"], p["lat_mag"]), []).append(p)
    out = {"rows": [], "radius_tol_all_trials": None}
    radii = []
    for (arm, h, l), P in sorted(cells.items()):
        fw = sorted({p["fwd"] for p in P})
        per = []
        bt = bg = None
        brk_t = brk_g = False
        for f in fw:
            T = [p for p in P if p["fwd"] == f]
            nt, ng = sum(ok(p) for p in T), sum(ok(p, gate_m) for p in T)
            per.append({"fwd": f, "n": len(T), "tol_ok": nt, "gate_ok": ng,
                        "err_cm": [None if p.get("err_m") is None else round(p["err_m"] * 100, 1) for p in T],
                        "tilt": [p.get("tilt_max_deg") for p in T], "reasons": [p.get("reason") for p in T],
                        "pre_ik": [(p.get("pre") or {}).get("ik_err_m") for p in T]})
            if not brk_t and nt == len(T):
                bt = f
            else:
                brk_t = True
            if not brk_g and ng == len(T):
                bg = f
            else:
                brk_g = True
        cfg = cfg_fwd_max(ws, h, l)
        row = {"arm": arm, "h": h, "lat": l, "fwd_all_trials_tol": bt, "fwd_all_trials_gate": bg,
               "cfg_fwd_max": None if cfg is None else round(cfg, 3),
               "margin_tol_vs_cfg": None if bt is None or cfg is None else round(bt - cfg, 3),
               "margin_gate_vs_cfg": None if bg is None or cfg is None else round(bg - cfg, 3), "targets": per}
        if bt is not None:
            row["r_tol"] = round(shoulder_r(ws, bt, l, h), 4)
            radii.append(row["r_tol"])
        if bg is not None:
            row["r_gate"] = round(shoulder_r(ws, bg, l, h), 4)
        out["rows"].append(row)
    out["radius_tol_all_trials"] = _pct_stats(radii)
    out["radius_gate_all_trials"] = _pct_stats([r["r_gate"] for r in out["rows"] if "r_gate" in r])
    out["margin_tol_vs_cfg"] = _pct_stats([r["margin_tol_vs_cfg"] for r in out["rows"]])
    out["margin_gate_vs_cfg"] = _pct_stats([r["margin_gate_vs_cfg"] for r in out["rows"]])
    return out


def propose(rep: dict, ws: dict) -> dict:
    """Limits from the measured envelope, per criterion and buffer: arm_reach_m = the p10 (over rows whose boundary
    the arm set, both arms) of the radius about the configured shoulder centre that the palm reached, minus the
    buffer; reach_fwd_m max = the p10 over heights (in the obj_z band's grasp heights) of the best forward reach,
    minus the buffer."""
    out = {}
    zlo = float(ws["obj_z_min_m"]) + 0.05          # grasp point ~ centre + half height + 3 cm: the band's grasp heights
    zhi = float(ws["obj_z_max_m"]) + 0.08
    for k in ("tol", "gate", "ik"):
        rr = rep["radius_about_cfg_shoulder_both_arms"][k]["r_ok"]
        best = [t["best_fwd"][k] for arm in rep["arms"].values() for h, t in arm["table"].items()
                if zlo <= float(h) <= zhi and t["best_fwd"][k] is not None]
        p10f = float(np.percentile(best, 10)) if best else None
        out[k] = {"radius_p10": rr.get("p10"), "radius_median": rr.get("median"), "best_fwd_p10": _r(p10f, 3),
                  "best_fwd_max": _r(max(best), 3) if best else None,
                  "by_buffer": {str(b): {"arm_reach_m": None if rr.get("p10") is None else round(rr["p10"] - b, 3),
                                         "reach_fwd_max": None if p10f is None else round(p10f - b, 3)}
                                for b in (0.0, 0.01, 0.02)}}
    return out


def _run_max(R, pred):
    far = None
    for p in R:
        if pred(p):
            far = p["fwd"]
        elif far is not None:
            break
    return far


def _pct_stats(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": int(a.size), "median": _r(np.median(a)), "p10": _r(np.percentile(a, 10)),
            "p90": _r(np.percentile(a, 90)), "max": _r(a.max()), "min": _r(a.min())}


def _count(v):
    out: dict = {}
    for x in v:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _print_tables(rep):
    for arm, R in rep["arms"].items():
        print(f"== {arm} arm: {R['n_reachable_tol']}/{R['n']} within tol, {R['n_gate_ok']} within the gate, "
              f"{R['n_ik_rejected']} IK-rejected")
        for k in ("tol", "gate"):
            print(f"  sphere fit ({k}): {json.dumps(R['sphere_fit_rest_frame'][k])}")
            print(f"  radius about the config shoulder ({k}): ok {R['radius_about_cfg_shoulder'][k]['r_ok']} "
                  f"fail {R['radius_about_cfg_shoulder'][k]['r_fail']}")
        hs = list(R["table"].keys())
        lats = sorted({l for h in hs for l in R["table"][h]["by_lat"].keys()}, key=float)
        print("  max forward reach (tol / gate) [config]; per height: best, horizontal reach, past a 0.20 m edge")
        print("  h \\ lat " + " ".join(f"{float(l):>15.2f}" for l in lats))
        for h in hs:
            t = R["table"][h]
            cells = []
            for l in lats:
                v = t["by_lat"].get(l)
                if v is None:
                    cells.append(" " * 15)
                    continue
                f = lambda x: " -  " if x is None else f"{x:.3f}"[1:]   # noqa: E731
                cells.append(f"{f(v['fwd_max_tol'])}/{f(v['fwd_max_gate'])}[{f(v['cfg_fwd_max'])}]".rjust(15))
            print(f"  {float(h):4.2f}   " + " ".join(cells) +
                  f"  best {t['best_fwd']['tol']}/{t['best_fwd']['gate']} horiz {t['horiz_max']['tol']}/"
                  f"{t['horiz_max']['gate']} [cfg {t['cfg_horiz']}] past-edge {t['past_edge_tol']}/"
                  f"{t['past_edge_gate']} [cfg {t['cfg_past_edge']}]")
    if rep.get("refine"):
        print("refine (pregrasp + grasp from rest, all trials): fwd tol / gate [config] margin")
        for r in rep["refine"]["rows"]:
            print(f"  {r['arm']:5s} h {r['h']:.2f} lat {r['lat']:.2f}: {r['fwd_all_trials_tol']} / "
                  f"{r['fwd_all_trials_gate']} [{r['cfg_fwd_max']}] {r['margin_tol_vs_cfg']} / {r['margin_gate_vs_cfg']}"
                  f"  " + " ".join(f"{t['fwd']}:{t['tol_ok']}/{t['gate_ok']}/{t['n']}{t['err_cm']}" for t in r["targets"]))
        print("  radius tol", rep["refine"]["radius_tol_all_trials"], "gate", rep["refine"]["radius_gate_all_trials"])
        print("  margin vs config tol", rep["refine"]["margin_tol_vs_cfg"], "gate", rep["refine"]["margin_gate_vs_cfg"])
    for arm, R in rep["arms"].items():
        print(f"{arm}: past a {rep['config_workspace'].get('stance_clearance_m')} m edge by turn (tol / gate "
              f"[config over the grid | config all])")
        for h, row in R["past_edge_by_turn"].items():
            print(f"  h {float(h):.2f}  " + "  ".join(f"{th}deg {v['tol']}/{v['gate']} [{v['cfg_grid']}|{v['cfg']}]"
                                                   for th, v in row.items()))
    print("proposal", json.dumps(rep["proposal"], indent=1))
    for k, v in (rep.get("turned") or {}).items():
        print(f"{k}: past the edge tol {v['past_edge_max_tol']} gate {v['past_edge_max_gate']} ik "
              f"{v['past_edge_max_ik']} [cfg on this line {v['cfg_past_edge_on_line']}, cfg best {v['cfg_past_edge_at_h']}]")


def _plots(pts, rep, out: Path, ok, gate_m, ws):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for arm, R in rep["arms"].items():
        P = [p for p in pts if p["arm"] == arm]
        hs = sorted({p["h"] for p in P})
        nc = 5
        nr = int(math.ceil(len(hs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.4 * nr), sharex=True, sharey=True, squeeze=False)
        for ax in axes.flat[len(hs):]:
            ax.axis("off")
        for ax, h in zip(axes.flat, hs):
            for p in [q for q in P if q["h"] == h]:
                if ok(p):
                    c, m = "#2a9d4b", "o"
                elif ok(p, gate_m):
                    c, m = "#e0a526", "o"
                elif p["state"] == "succeeded":
                    c, m = "#c0392b", "o"
                else:
                    c, m = "#888888", "x"
                ax.scatter(p["fwd"], p["lat_mag"], c=c, s=30, marker=m)
            ls = np.linspace(0, 0.40, 41)
            cf = [cfg_fwd_max(ws, h, l) for l in ls]
            ax.plot([np.nan if v is None else v for v in cf], ls, "k--", lw=1)
            ax.set_title(f"{arm}, grasp point h = {h:.2f} m", fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("forward of the pelvis (m)")
        for ax in axes[:, 0]:
            ax.set_ylabel("lateral, arm side (m)")
        fig.suptitle(f"{arm} arm, live arm_script grasp: green palm <= {rep['tol_m'] * 100:.1f} cm, amber <= "
                     f"{gate_m * 100:.0f} cm (pipeline gate), red further, grey x IK rejected; dashed = config reach")
        fig.tight_layout()
        fig.savefig(out / f"grid_{arm}.png", dpi=85)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, arm in zip(axes, sorted(rep["arms"])):
        T = rep["arms"][arm]["table"]
        hs = sorted(float(h) for h in T)
        for l in ("0.0", "0.1", "0.2", "0.3", "0.4"):
            xs = [T[str(h)]["by_lat"].get(l, {}).get("fwd_max_tol") for h in hs]
            xg = [T[str(h)]["by_lat"].get(l, {}).get("fwd_max_gate") for h in hs]
            xc = [T[str(h)]["by_lat"].get(l, {}).get("cfg_fwd_max") for h in hs]
            line, = ax.plot([np.nan if v is None else v for v in xs], hs, "-o", ms=3, label=f"lat {l} tol")
            ax.plot([np.nan if v is None else v for v in xg], hs, ":", color=line.get_color(), lw=1)
            ax.plot([np.nan if v is None else v for v in xc], hs, "--", color=line.get_color(), lw=0.8, alpha=0.6)
        ax.set_title(f"{arm}: max forward reach (solid: within tol; dotted: pipeline gate; dashed: config)",
                     fontsize=9)
        ax.set_xlabel("forward of the pelvis (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("grasp point height above the floor (m)")
    fig.tight_layout()
    fig.savefig(out / "reach_vs_height.png", dpi=100)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, R in rep["arms"].items():
        T = R["table"]
        hs = sorted(float(h) for h in T)
        ax.plot([T[str(h)]["horiz_max"]["tol"] or np.nan for h in hs], hs, "-o", ms=3, label=f"{arm} tol")
        ax.plot([T[str(h)]["horiz_max"]["gate"] or np.nan for h in hs], hs, ":o", ms=2, label=f"{arm} gate")
    hs = sorted({float(h) for R in rep["arms"].values() for h in R["table"]})
    ax.plot([cfg_horiz(ws, h) for h in hs], hs, "k--", label="config horizontal_reach")
    ax.set_xlabel("horizontal pelvis-to-palm reach (m); past an edge = this - stance_clearance_m 0.20")
    ax.set_ylabel("grasp point height (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "horizontal_reach.png", dpi=100)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in sorted(rep["arms"]):
        e = [p["err_m"] * 100 for p in pts if p["arm"] == arm and p["state"] == "succeeded"
             and p.get("err_m") is not None]
        ax.hist(e, bins=np.arange(0, 20.5, 0.5), alpha=0.6, label=arm)
    ax.axvline(rep["tol_m"] * 100, color="r", ls="--")
    ax.set_xlabel("achieved palm (P1) to target (cm), every op that ran")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "err_hist.png", dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sweep", "turned", "refine"):
        s = sub.add_parser(name)
        s.add_argument("--port-offset", type=int, default=int(os.environ.get("WL_PORT_OFFSET", 0) or 0))
        s.add_argument("--out", required=True)
        s.add_argument("--arms", default="right,left")
        s.add_argument("--floor", type=float, default=0.0)
        s.add_argument("--tol", type=float, default=0.025)
        s.add_argument("--tilt-ok", type=float, default=5.0)
        s.add_argument("--tilt-abort", type=float, default=8.0)
        s.add_argument("--shift-abort", type=float, default=0.15)
        s.add_argument("--drift-abort", type=float, default=0.50)
        s.add_argument("--gate-m", type=float, default=0.10, help="sonic_arm_script attach_max_m")
        s.add_argument("--fwd-step", type=float, default=0.025)
        s.add_argument("--resume", action="store_true")
    w = sub.choices["sweep"]
    w.add_argument("--heights", default="0.95,1.00,0.90,1.05,0.85,1.10,0.80,1.15,0.75,1.20,0.70,1.25,1.30")
    w.add_argument("--lats", default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    w.add_argument("--fwd-lo", type=float, default=0.30)
    w.add_argument("--fwd-hi", type=float, default=0.65)
    t = sub.choices["turned"]
    t.add_argument("--heights", default="0.85,1.00,1.15")
    t.add_argument("--turns", default="0,25,45")
    t.add_argument("--standoff", type=float, default=0.20)
    t.add_argument("--shoulder-lat", type=float, default=0.136)
    t.add_argument("--past-lo", type=float, default=0.10)
    t.add_argument("--past-hi", type=float, default=0.45)
    r = sub.choices["refine"]
    r.add_argument("--base", required=True, help="the sweep dir whose rows.jsonl gives the boundaries")
    r.add_argument("--heights", default="0.95,0.85,1.05,1.15")
    r.add_argument("--lats", default="0.0,0.10,0.20,0.30")
    r.add_argument("--offsets", default="0.0,0.01,0.02")
    r.add_argument("--trials", type=int, default=2)
    z = sub.add_parser("analyze")
    z.add_argument("dirs", nargs="+")
    z.add_argument("--turned", nargs="*", default=None)
    z.add_argument("--tol", type=float, default=0.025)
    z.add_argument("--gate-m", type=float, default=0.10)
    z.add_argument("--config", default=None)
    z.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sweep":
        return cmd_sweep(a)
    if a.cmd == "turned":
        return cmd_turned(a)
    if a.cmd == "refine":
        return cmd_refine(a)
    return analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
