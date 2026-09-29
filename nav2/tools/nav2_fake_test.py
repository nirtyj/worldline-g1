"""End-to-end Nav2 test through BodyClient only (body venv). Run by nav2/tools/run_fake_e2e.sh against the fakes; the
same script runs unchanged against the real stack (P1 Isaac + SONIC deploy) because it only uses the contract ports.

    .venv/bin/python -m nav2.tools.nav2_fake_test --port-offset 400 --out DIR --body-log DIR/body --bridge-log DIR/nav2 \
        [--house-dir assets/houses/procthor-train-38] [--tests goals,unreachable,cancel,watchdog] [--no-stand]

Tests (pass criteria in brackets; GT = gt.pose, independent of what the body reports):
  goals        go_to through Nav2 to >= 3 goals in >= 2 rooms  [body succeeded, backend nav2, GT pos err <= 0.30 m,
               GT yaw err <= 15 deg when a yaw is given, no fall]
  unreachable  a free goal the robot cannot reach (a free pocket not connected at the robot radius) -> failed no_path
               synchronously [reply within 5 s, no motion]; plus a goal inside furniture -> goal_in_obstacle or no_path
               (Smac's goal tolerance > 0 reports NO_VALID_PATH, not GOAL_OCCUPIED)
  cancel       go_to far away, cancel (= body stop) after ~1 m  [go_to canceled, Nav2 goal canceled, no /cmd_vel
               forwarded after the cancel, GT speed < 0.05 m/s within 2.0 s]
  watchdog     stream op velocity (pure wz 0.5 for 1.5 s, then vx 0.4 for 2 s), then stop sending  [planner goes IDLE
               within watchdog_s + 60 ms of the last message, op succeeded ended_by=watchdog, robot stops]
Artifacts: trajectory.png (all runs over the map + Nav2 plans), goal_<k>.png (per goal: map zoom + cmd_vel / planner /
GT speed traces), watchdog.png, metrics.json, pose.csv, events.jsonl, phases.json (for the CPU split).
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

from body.client import BodyClient  # noqa: E402
from body.config import ep, ports as _ports  # noqa: E402
from body.p1_client import P1Rpc, PoseSub  # noqa: E402
from body.wire import wrap  # noqa: E402


# ------------------------------------------------------------------------------------------------------------------
class Recorder:
    """50 Hz GT log from our own gt.pose SUB."""

    def __init__(self, sub: PoseSub):
        self.sub = sub
        self.rows: list[tuple] = []
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        last = None
        while self._run:
            p = self.sub.latest()
            if p is not None and p is not last:
                last = p
                self.rows.append((time.time(), p.x, p.y, p.yaw, p.vx, p.vy, p.wz, p.pelvis_z, int(p.fallen)))
            time.sleep(0.01)

    def stop(self):
        self._run = False
        self._th.join(1.0)

    def arr(self, t0=None, t1=None) -> np.ndarray:
        a = np.array(self.rows) if self.rows else np.zeros((0, 9))
        if len(a) and t0 is not None:
            a = a[a[:, 0] >= t0]
        if len(a) and t1 is not None:
            a = a[a[:, 0] <= t1]
        return a


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_map(house_dir: str | None, p1: P1Rpc):
    rep = p1.call("get_occupancy", timeout_s=30.0, robot_radius=0.30)
    with np.load(rep["path"]) as z:
        occ = (z["occ"] > 0)
    return occ, float(rep["resolution"]), [float(v) for v in rep["origin"]]


def free_pockets(occ, res, origin, start_xy, r=0.30, min_margin=0.45):
    """Free cells with clearance > r + 0.03 that are NOT connected to the start at radius r, whose nearest
    reachable cell is > min_margin away (so Smac's 0.25 m goal tolerance cannot reach into the pocket)."""
    from scipy import ndimage
    d = ndimage.distance_transform_edt(~occ) * res
    walk = d > r
    lab, n = ndimage.label(walk)
    if n == 0:
        return [], walk, d
    sizes = ndimage.sum(walk, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1          # the house's walkable area = the largest component
    reach = lab == main
    dist_to_reach = ndimage.distance_transform_edt(~reach) * res
    cand = (d > r + 0.03) & ~reach & (dist_to_reach > min_margin)
    ys, xs = np.nonzero(cand)
    pts = [(origin[0] + (x + 0.5) * res, origin[1] + (y + 0.5) * res, float(dist_to_reach[y, x]), float(d[y, x]))
           for y, x in zip(ys, xs)]
    pts.sort(key=lambda p: -p[2])
    return pts, reach, d


def furniture_goal(house_info: dict, occ, res, origin):
    """Centre of a large object whose centre cell is blocked (goal_in_obstacle test)."""
    best = None
    for o in house_info.get("objects", []):
        ab = o.get("aabb")
        if not ab:
            continue
        (x0, y0, _), (x1, y1, _) = ab
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        area = (x1 - x0) * (y1 - y0)
        ix, iy = int((cx - origin[0]) / res), int((cy - origin[1]) / res)
        if 0 <= iy < occ.shape[0] and 0 <= ix < occ.shape[1] and occ[iy, ix] and area > 0.6:
            if best is None or area > best[2]:
                best = (cx, cy, area, o.get("category") or o.get("id"))
    return best


# ------------------------------------------------------------------------------------------------------------------
class Nav2Test:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.P = _ports(a.port_offset)
        self.bc = BodyClient(port_offset=a.port_offset).connect(15.0)
        self.events: list[dict] = []
        self.bc.add_listener(self.events.append)
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0)
        self.sub = PoseSub(ep(self.P["p1_pose"]))
        self.sub.start()
        self.rec = Recorder(self.sub)
        self.phases: list[dict] = []
        self.results: dict = {}
        self.runs: list[dict] = []       # for plotting
        hi_path = os.path.join(a.house_dir, "house_info.json") if a.house_dir else None
        self.house = json.load(open(hi_path)) if hi_path and os.path.exists(hi_path) else {}
        self.occ, self.res, self.origin = load_map(a.house_dir, self.p1)

    def phase(self, name: str, t0: float, t1: float, **kw):
        self.phases.append({"name": name, "t0": t0, "t1": t1, **kw})

    def pose(self):
        for _ in range(100):
            p = self.sub.latest()
            if p is not None:
                return p
            time.sleep(0.02)
        raise RuntimeError("no gt.pose")

    def settle_speed(self, t_from: float, window: float = 2.0) -> float | None:
        """First time after t_from when GT speed stays < 0.05 m/s for 0.25 s (s after t_from)."""
        a = self.rec.arr(t_from, t_from + window + 1.0)
        if not len(a):
            return None
        sp = np.hypot(a[:, 4], a[:, 5])
        for i in range(len(a)):
            j = np.searchsorted(a[:, 0], a[i, 0] + 0.25)
            if j <= len(a) and (sp[i:j] < 0.05).all() and j > i:
                return float(a[i, 0] - t_from)
        return None

    # -- tests ----------------------------------------------------------------------------------------
    def stand(self):
        st = self.bc.status()
        if st.get("in_control") and not self.a.force_stand:
            return {"skipped": "already in control"}
        t0 = time.time()
        h = self.bc.stand(timeout=120)
        self.phase("stand", t0, time.time())
        if not h.ok:
            raise RuntimeError(f"stand failed: {h.result}")
        return {"ok": True, "duration_s": round(time.time() - t0, 2)}

    def goals(self) -> dict:
        rp = {r["room"]: r for r in self.house.get("room_points", []) if r.get("ok")}
        spawn = self.house.get("spawn") or {}
        # >= 3 goals across rooms; alternate yaw given / not given
        plan = []
        order = [k for k in ("kitchen", "bedroom", "living_room") if k in rp] or list(rp)
        yaws = [None, math.pi, math.pi / 2]
        for k, room in enumerate(order):
            g = rp[room]
            plan.append({"room": room, "x": g["x"], "y": g["y"], "yaw": yaws[k % len(yaws)]})
        if spawn and "kitchen" in rp:
            plan.append({"room": "kitchen", "x": rp["kitchen"]["x"], "y": rp["kitchen"]["y"] + 1.2, "yaw": 0.0})
        out = []
        for k, g in enumerate(plan):
            p0 = self.pose()
            t0 = time.time()
            h = self.bc.go_to(g["x"], g["y"], yaw=g["yaw"], timeout_s=self.a.goal_timeout)
            t1 = time.time()
            p1 = self.pose()
            r = h.result or {}
            gt_err = math.hypot(p1.x - g["x"], p1.y - g["y"])
            if r.get("goal"):   # a snapped goal is what the body aimed for; report both
                gt_err_snap = math.hypot(p1.x - r["goal"][0], p1.y - r["goal"][1])
            else:
                gt_err_snap = gt_err
            yaw_err = None if g["yaw"] is None else math.degrees(wrap(g["yaw"] - p1.yaw))
            acc = [e for e in h.events if e.get("state") == "accepted"]
            plan_path = ((acc[0].get("data") or {}).get("plan") or {}).get("path") if acc else None
            ok = (h.state == "succeeded" and r.get("backend") == "nav2" and gt_err_snap <= 0.30
                  and (yaw_err is None or abs(yaw_err) <= 15.0) and not p1.fallen)
            rec = {"k": k, "room": g["room"], "goal": [g["x"], g["y"]], "goal_yaw": g["yaw"], "state": h.state,
                   "reason": h.reason, "backend": r.get("backend"), "body_pos_err": r.get("pos_err"),
                   "body_yaw_err_deg": r.get("yaw_err_deg"), "gt_pos_err": round(gt_err, 3),
                   "gt_pos_err_vs_body_goal": round(gt_err_snap, 3),
                   "gt_yaw_err_deg": None if yaw_err is None else round(yaw_err, 2),
                   "path_len_m": r.get("path_len_m"), "walked_m": r.get("walked_m"),
                   "duration_s": round(t1 - t0, 2), "start": [round(p0.x, 3), round(p0.y, 3)],
                   "recoveries": r.get("replans"), "reapproach": r.get("approach_attempts"),
                   "nav2": r.get("nav2"), "velocity": r.get("velocity"), "pass": bool(ok), "id": h.id}
            print(f"[goals] {k} {g['room']} -> {h.state} {h.reason or ''} gt_err={gt_err:.3f} "
                  f"yaw_err={rec['gt_yaw_err_deg']} t={t1 - t0:.1f}s walked={r.get('walked_m')}", flush=True)
            out.append(rec)
            self.runs.append({"kind": "goal", "k": k, "t0": t0, "t1": t1, "goal": g, "plan": plan_path,
                              "id": h.id, "state": h.state})
            self.phase(f"goal{k}", t0, t1, room=g["room"])
        rooms = {o["room"] for o in out if o["pass"]}
        return {"pass": sum(o["pass"] for o in out) >= 3 and len(rooms) >= 2 and all(o["pass"] for o in out),
                "n_pass": sum(o["pass"] for o in out), "n": len(out), "rooms_reached": sorted(rooms), "goals": out}

    def unreachable(self) -> dict:
        p = self.pose()
        pockets, _, _ = free_pockets(self.occ, self.res, self.origin, (p.x, p.y))
        out = {}
        if pockets:
            x, y, margin, clear = pockets[0]
            t0 = time.time()
            h = self.bc.go_to(x, y, timeout_s=30)
            t1 = time.time()
            p1 = self.pose()
            moved = math.hypot(p1.x - p.x, p1.y - p.y)
            out["pocket"] = {"goal": [round(x, 3), round(y, 3)], "pocket_margin_m": round(margin, 3),
                             "clearance_m": round(clear, 3), "state": h.state, "reason": h.reason,
                             "reply_state": (h.reply or {}).get("state"), "t_reply_s": round(t1 - t0, 3),
                             "moved_m": round(moved, 3), "nav2": (h.result or {}).get("error_name"),
                             "pass": h.state == "failed" and h.reason == "no_path" and t1 - t0 < 5.0 and moved < 0.05}
            self.runs.append({"kind": "unreachable", "t0": t0, "t1": t1, "goal": {"x": x, "y": y}, "plan": None})
            self.phase("unreachable", t0, t1)
            print(f"[unreachable] pocket {out['pocket']}", flush=True)
        else:
            out["pocket"] = {"pass": False, "reason": "no disconnected free pocket in this house"}
        fg = furniture_goal(self.house, self.occ, self.res, self.origin)
        if fg:
            t0 = time.time()
            h = self.bc.go_to(fg[0], fg[1], timeout_s=30)
            t1 = time.time()
            out["furniture"] = {"goal": [round(fg[0], 3), round(fg[1], 3)], "object": fg[3], "state": h.state,
                                "reason": h.reason, "t_reply_s": round(t1 - t0, 3),
                                "nav2": (h.result or {}).get("error_name"),
                                "pass": h.state == "failed" and h.reason in ("goal_in_obstacle", "no_path")}
            print(f"[unreachable] furniture {out['furniture']}", flush=True)
            self.runs.append({"kind": "unreachable", "t0": t0, "t1": t1, "goal": {"x": fg[0], "y": fg[1]},
                              "plan": None, "state": h.state})
        # a goal outside the map (Nav2 GOAL_OUTSIDE_MAP)
        t0 = time.time()
        h = self.bc.go_to(self.origin[0] - 5.0, self.origin[1] - 5.0, timeout_s=30)
        out["outside_map"] = {"goal": [self.origin[0] - 5.0, self.origin[1] - 5.0], "state": h.state,
                              "reason": h.reason, "t_reply_s": round(time.time() - t0, 3),
                              "nav2": (h.result or {}).get("error_name"),
                              "pass": h.state == "failed" and h.reason == "no_path"}
        print(f"[unreachable] outside_map {out['outside_map']}", flush=True)
        variants = [v for k, v in out.items() if isinstance(v, dict) and "pass" in v
                    and not (k == "pocket" and "reason" in v and "no disconnected" in str(v.get("reason")))]
        out["pass"] = bool(variants) and all(v["pass"] for v in variants) and any(
            v.get("reason") == "no_path" for v in variants)
        return out

    def cancel(self) -> dict:
        p0 = self.pose()
        rp = {r["room"]: r for r in self.house.get("room_points", []) if r.get("ok")}
        # the room point farthest from here
        far = max(rp.values(), key=lambda r: math.hypot(r["x"] - p0.x, r["y"] - p0.y))
        t0 = time.time()
        h = self.bc.go_to(far["x"], far["y"], timeout_s=120, wait=False)
        moved = 0.0
        while time.time() - t0 < 25 and not h.done():
            p = self.pose()
            moved = math.hypot(p.x - p0.x, p.y - p0.y)
            if moved >= 1.0:
                break
            time.sleep(0.05)
        p_c = self.pose()
        v_at_cancel = p_c.speed
        t_cancel = time.time()
        sh = h.cancel(wait=True, timeout=10)
        t_stop = time.time()
        try:
            h.wait(5)
        except TimeoutError:
            pass
        time.sleep(2.5)
        t_settle = self.settle_speed(t_cancel, 3.0)
        # bridge side: goal state + /cmd_vel forwarded after the cancel
        st = {}
        try:
            import zmq
            s = zmq.Context.instance().socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, 2000)
            s.connect(ep(self.P["nav_bridge"]))
            s.send(json.dumps({"op": "status", "id": h.id}).encode())
            st = json.loads(s.recv())
            s.close(0)
        except Exception as e:  # noqa: BLE001
            st = {"error": repr(e)}
        cmd = load_jsonl(os.path.join(self.a.bridge_log, "cmd_vel.jsonl"))
        fwd_after = [c for c in cmd if c.get("goal") == h.id and c.get("fwd") and c["t_wall"] > t_cancel + 0.05]
        out = {"goal": [far["x"], far["y"]], "moved_before_cancel_m": round(moved, 3),
               "v_at_cancel": round(v_at_cancel, 3), "goto_state": h.state, "goto_reason": h.reason,
               "stop_state": sh.state, "stop_time_s_body": (sh.result or {}).get("stop_time_s"),
               "gt_settle_s": None if t_settle is None else round(t_settle, 3),
               "nav2_goal_state": st.get("state"), "cmd_vel_forwarded_after_cancel": len(fwd_after),
               "cancel_rpc_s": round(t_stop - t_cancel, 3)}
        out["pass"] = (h.state == "canceled" and sh.state == "succeeded" and st.get("state") == "canceled"
                       and len(fwd_after) == 0 and t_settle is not None and t_settle <= 2.0)
        self.runs.append({"kind": "cancel", "t0": t0, "t1": time.time(), "goal": {"x": far["x"], "y": far["y"]},
                          "plan": None, "t_cancel": t_cancel, "id": h.id, "state": h.state})
        self.phase("cancel", t0, time.time())
        print(f"[cancel] {out}", flush=True)
        return out

    def _bridge(self, req: dict, timeout_s: float = 5.0) -> dict:
        import zmq
        s = zmq.Context.instance().socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        s.connect(ep(self.P["nav_bridge"]))
        try:
            s.send(json.dumps(req).encode())
            return json.loads(s.recv())
        finally:
            s.close(0)

    def _far_room_point(self):
        p0 = self.pose()
        rp = [r for r in self.house.get("room_points", []) if r.get("ok")]
        return max(rp, key=lambda r: math.hypot(r["x"] - p0.x, r["y"] - p0.y))

    def timeout(self) -> dict:
        """go_to with timeout_s=4 to a far goal: body fails 'timeout' and cancels the Nav2 goal."""
        far = self._far_room_point()
        t0 = time.time()
        h = self.bc.go_to(far["x"], far["y"], timeout_s=4.0)
        t1 = time.time()
        time.sleep(1.5)
        st = self._bridge({"op": "status", "id": h.id})
        t_settle = self.settle_speed(t1, 3.0)
        out = {"state": h.state, "reason": h.reason, "duration_s": round(t1 - t0, 2),
               "nav2_goal_state": st.get("state"), "gt_settle_s": t_settle}
        out["pass"] = (h.state == "failed" and h.reason == "timeout" and st.get("state") == "canceled"
                       and 3.9 <= t1 - t0 <= 6.0)
        self.runs.append({"kind": "timeout", "t0": t0, "t1": t1, "goal": {"x": far["x"], "y": far["y"]},
                          "plan": None, "id": h.id, "state": h.state})
        self.phase("timeout", t0, t1)
        print(f"[timeout] {out}", flush=True)
        return out

    def escape(self) -> dict:
        """FAKE ONLY (uses P1 reset_robot): put the robot inside the inscribed zone (clearance 0.18-0.26 m < robot
        radius 0.30), where Nav2 cannot plan (START_OCCUPIED); go_to must step out (escape) and then succeed."""
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~self.occ) * self.res
        p0 = self.pose()
        rp = [r for r in self.house.get("room_points", []) if r.get("ok")]
        tgt = min(rp, key=lambda r: math.hypot(r["x"] - p0.x, r["y"] - p0.y))
        ys, xs = np.nonzero((d > 0.18) & (d < 0.26))
        X = self.origin[0] + (xs + 0.5) * self.res
        Y = self.origin[1] + (ys + 0.5) * self.res
        # a spot near the robot, in the same walkable component (reachable after the escape)
        dist = np.hypot(X - p0.x, Y - p0.y)
        i = int(np.argmin(np.abs(dist - 1.5)))
        pick = (float(X[i]), float(Y[i]), float(d[ys[i], xs[i]]))
        rep = self.p1.call("reset_robot", x=pick[0], y=pick[1], yaw=p0.yaw, band=False)
        time.sleep(1.0)
        t0 = time.time()
        h = self.bc.go_to(tgt["x"], tgt["y"], timeout_s=90)
        t1 = time.time()
        r = h.result or {}
        p1 = self.pose()
        out = {"reset_to": [round(pick[0], 3), round(pick[1], 3)], "clearance_m": round(pick[2], 3),
               "reset_reply_ok": rep.get("ok"), "goal": [tgt["x"], tgt["y"]], "state": h.state, "reason": h.reason,
               "escapes": r.get("escapes"), "gt_pos_err": round(math.hypot(p1.x - tgt["x"], p1.y - tgt["y"]), 3),
               "duration_s": round(t1 - t0, 2)}
        out["pass"] = h.state == "succeeded" and bool(r.get("escapes"))
        self.runs.append({"kind": "escape", "t0": t0, "t1": t1, "goal": {"x": tgt["x"], "y": tgt["y"]},
                          "plan": None, "id": h.id, "state": h.state})
        self.phase("escape", t0, t1)
        print(f"[escape] {out}", flush=True)
        return out

    def stuck(self) -> dict:
        """FAKE ONLY (uses P1 spawn_obstacle, physics only, NOT in the map): block a doorway on the planned path.
        Nav2's progress checker fails the controller (FAILED_TO_MAKE_PROGRESS), the BT recovers (clear costmaps,
        wait 2 s, replan the same path), gives up; the body retries once, then fails with reason 'stuck'."""
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~self.occ) * self.res
        far = self._far_room_point()
        pl = self._bridge({"op": "plan", "x": far["x"], "y": far["y"]})
        path = np.array(pl.get("path") or [])
        p0 = self.pose()
        best = None
        for q in path:
            if math.hypot(q[0] - p0.x, q[1] - p0.y) < 1.2:
                continue
            ix, iy = int((q[0] - self.origin[0]) / self.res), int((q[1] - self.origin[1]) / self.res)
            c = float(d[iy, ix])
            if best is None or c < best[2]:
                best = (float(q[0]), float(q[1]), c)
        r_obs = best[2] + 0.25
        self.p1.call("spawn_obstacle", x=best[0], y=best[1], r=r_obs)
        t0 = time.time()
        h = self.bc.go_to(far["x"], far["y"], timeout_s=240)
        t1 = time.time()
        r = h.result or {}
        out = {"obstacle": [round(best[0], 3), round(best[1], 3), round(r_obs, 3)], "goal": [far["x"], far["y"]],
               "state": h.state, "reason": h.reason, "nav2": (r.get("nav2") or {}).get("error_name"),
               "nav2_retries": r.get("nav2_retries"), "recoveries": r.get("replans"),
               "duration_s": round(t1 - t0, 2)}
        out["pass"] = h.state == "failed" and h.reason == "stuck"
        self.runs.append({"kind": "stuck", "t0": t0, "t1": t1, "goal": {"x": far["x"], "y": far["y"]},
                          "plan": pl.get("path"), "id": h.id, "state": h.state, "obstacle": out["obstacle"]})
        self.phase("stuck", t0, t1)
        print(f"[stuck] {out}", flush=True)
        return out

    def watchdog(self) -> dict:
        p0 = self.pose()
        stream = f"wd-{int(time.time())}"
        sends = []
        t0 = time.time()
        first = None
        seq = [(1.5, 0.0, 0.0, 0.5), (2.0, 0.4, 0.0, 0.0)]   # turn in place, then walk: ends in SLOW_WALK
        for dur, vx, vy, wz in seq:
            te = time.time() + dur
            while time.time() < te:
                ts = time.time()
                rep = self.bc.request("velocity", {"vx": vx, "vy": vy, "wz": wz, "stream": stream, "t_wall": ts})
                sends.append((ts, vx, vy, wz, rep.get("state")))
                if first is None:
                    first = rep
                time.sleep(max(0.0, 0.05 - (time.time() - ts)))
        t_last = sends[-1][0]
        op_id = first.get("id")
        # wait for the op to end
        term = None
        te = time.time() + 6.0
        while time.time() < te and term is None:
            for e in list(self.events):
                if e.get("id") == op_id and e.get("state") in ("succeeded", "failed", "canceled"):
                    term = e
            time.sleep(0.05)
        time.sleep(0.5)
        p1 = self.pose()
        pc = load_jsonl(os.path.join(self.a.body_log, "planner_cmds.jsonl"))
        idle = [c for c in pc if c.get("mode") == 0 and c["t_wall"] > t_last and c.get("owner") == op_id]
        first_idle = idle[0]["t_wall"] if idle else None
        lat = None if first_idle is None else first_idle - t_last
        t_settle = self.settle_speed(t_last, 3.0)
        data = (term or {}).get("data") or {}
        wd = (data.get("velocity") or {}).get("watchdog_s", 0.3)
        out = {"op": op_id, "first_reply": first.get("state"), "n_sent": len(sends),
               "update_states": sorted({s[4] for s in sends[1:]}), "terminal": (term or {}).get("state"),
               "ended_by": data.get("ended_by"), "watchdog_s": wd,
               "idle_latency_s": None if lat is None else round(lat, 3),
               "gt_settle_after_last_msg_s": None if t_settle is None else round(t_settle, 3),
               "moved_m": round(math.hypot(p1.x - p0.x, p1.y - p0.y), 3),
               "yaw_change_deg": round(math.degrees(wrap(p1.yaw - p0.yaw)), 1),
               "velocity_stats": data.get("velocity")}
        out["pass"] = (first.get("state") == "accepted" and out["update_states"] == ["done"]
                       and term is not None and term.get("state") == "succeeded" and out["ended_by"] == "watchdog"
                       and lat is not None and wd - 0.02 <= lat <= wd + 0.06 and out["moved_m"] > 0.4)
        self.runs.append({"kind": "watchdog", "t0": t0, "t1": time.time(), "t_last": t_last, "sends": sends,
                          "idle_t": first_idle, "op": op_id})
        self.phase("watchdog", t0, time.time())
        print(f"[watchdog] {json.dumps({k: v for k, v in out.items() if k != 'velocity_stats'})}", flush=True)
        return out

    # -- artifacts ------------------------------------------------------------------------------------
    def plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import ndimage

        occ, res, (x0, y0) = self.occ, self.res, self.origin
        H, W = occ.shape
        ext = [x0, x0 + W * res, y0, y0 + H * res]
        d = ndimage.distance_transform_edt(~occ) * res
        bg = np.full(occ.shape + (3,), 1.0)
        bg[(d <= 0.30) & ~occ] = (0.85, 0.88, 0.95)       # inscribed (robot centre cannot go)
        bg[(d > 0.30) & (d <= 0.45)] = (0.94, 0.95, 0.98)  # inflation band
        bg[occ] = (0.25, 0.25, 0.28)
        cmd = load_jsonl(os.path.join(self.a.bridge_log, "cmd_vel.jsonl"))
        pc = load_jsonl(os.path.join(self.a.body_log, "planner_cmds.jsonl"))
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(9, 10))
        ax.imshow(bg, origin="lower", extent=ext, interpolation="nearest")
        for r in self.house.get("rooms", []):
            poly = r.get("polygon")
            if poly:
                P = np.array(poly + [poly[0]])
                ax.plot(P[:, 0], P[:, 1], ":", color="0.6", lw=0.8)
                c = np.mean(np.array(poly), axis=0)
                ax.text(c[0], c[1], r.get("name") or r.get("id"), color="0.45", fontsize=8, ha="center")
        for i, run in enumerate(self.runs):
            if run["kind"] == "watchdog":
                continue
            col = colors[i % 10]
            a = self.rec.arr(run["t0"], run["t1"])
            if run.get("plan"):
                P = np.array(run["plan"])
                ax.plot(P[:, 0], P[:, 1], "--", color=col, lw=1.0, alpha=0.8)
            if len(a):
                ax.plot(a[:, 1], a[:, 2], "-", color=col, lw=2.0,
                        label=f"{run['kind']} {run.get('k', '')} ({run.get('state', '')})")
                ax.plot(a[0, 1], a[0, 2], "o", color=col, ms=5)
            g = run["goal"]
            ax.plot(g["x"], g["y"], "x" if run["kind"] == "unreachable" else "*", color=col, ms=14, mew=2)
            if run.get("obstacle"):
                ox, oy, orr = run["obstacle"]
                ax.add_patch(plt.Circle((ox, oy), orr, color="red", alpha=0.35, label="spawned obstacle (physics)"))
            if run.get("t_cancel"):
                b = self.rec.arr(run["t_cancel"] - 0.02, run["t_cancel"] + 0.05)
                if len(b):
                    ax.plot(b[0, 1], b[0, 2], "s", color="red", ms=8, label="cancel")
        ax.set_title(f"Nav2 (FAKE P1 + FAKE deploy, kinematic G1) in {self.house.get('house_id', '?')}\n"
                     "solid = GT trajectory, dashed = Nav2 plan (Smac 2D), * goal, x unreachable goal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.a.out, "trajectory.png"), dpi=130)
        plt.close(fig)
        # per goal traces
        for run in [r for r in self.runs if r["kind"] in ("goal", "cancel")]:
            a = self.rec.arr(run["t0"], run["t1"] + 1.0)
            if not len(a):
                continue
            t0 = run["t0"]
            fig = plt.figure(figsize=(13, 8))
            axm = fig.add_subplot(1, 2, 1)
            axm.imshow(bg, origin="lower", extent=ext, interpolation="nearest")
            if run.get("plan"):
                P = np.array(run["plan"])
                axm.plot(P[:, 0], P[:, 1], "--", color="C1", lw=1.2, label="Nav2 plan")
            axm.plot(a[:, 1], a[:, 2], "-", color="C0", lw=2, label="GT")
            step = max(1, len(a) // 25)
            axm.quiver(a[::step, 1], a[::step, 2], np.cos(a[::step, 3]), np.sin(a[::step, 3]), color="C0",
                       scale=25, width=0.004)
            g = run["goal"]
            axm.plot(g["x"], g["y"], "*", color="C3", ms=15)
            pad = 0.8
            axm.set_xlim(min(a[:, 1].min(), g["x"]) - pad, max(a[:, 1].max(), g["x"]) + pad)
            axm.set_ylim(min(a[:, 2].min(), g["y"]) - pad, max(a[:, 2].max(), g["y"]) + pad)
            axm.set_aspect("equal")
            axm.legend(fontsize=8)
            axm.set_title(f"{run['kind']} {run.get('k', '')}: {run.get('state')}")
            c = [x for x in cmd if x.get("goal") == run.get("id")]
            p = [x for x in pc if x.get("owner") == run.get("id")]
            ax1 = fig.add_subplot(3, 2, 2)
            if c:
                tc = np.array([x["t_wall"] - t0 for x in c])
                ax1.plot(tc, [x["vx"] for x in c], label="cmd vx")
                ax1.plot(tc, [x["vy"] for x in c], label="cmd vy")
                ax1.plot(tc, [x["wz"] for x in c], label="cmd wz")
            ax1.set_ylabel("Nav2 /cmd_vel")
            ax1.legend(fontsize=7, ncol=3)
            ax1.grid(alpha=0.3)
            ax2 = fig.add_subplot(3, 2, 4, sharex=ax1)
            if p:
                tp = np.array([x["t_wall"] - t0 for x in p])
                ax2.step(tp, [x["speed"] if x["mode"] == 1 else 0.0 for x in p], where="post", label="planner speed")
                ax2.step(tp, [x["mode"] * 0.1 for x in p], where="post", label="mode x0.1 (1=SLOW_WALK)")
                mv = [math.degrees(math.atan2(*x["move_w"][::-1])) if x["mode"] == 1 else np.nan for x in p]
                ax2b = ax2.twinx()
                ax2b.plot(tp, [math.degrees(x["facing_w"]) if x.get("facing_w") is not None else np.nan for x in p], "C2.", ms=2,
                          label="facing_w deg")
                ax2b.plot(tp, mv, "C3.", ms=2, label="move dir deg")
                ax2b.set_ylabel("deg")
                ax2b.legend(fontsize=7, loc="lower right")
            ax2.set_ylabel("SONIC planner msgs\n(changes only)")
            ax2.legend(fontsize=7, loc="upper right")
            ax2.grid(alpha=0.3)
            ax3 = fig.add_subplot(3, 2, 6, sharex=ax1)
            ta = a[:, 0] - t0
            ax3.plot(ta, np.hypot(a[:, 4], a[:, 5]), label="GT speed")
            ax3.plot(ta, a[:, 6], label="GT wz")
            if run.get("t_cancel"):
                ax3.axvline(run["t_cancel"] - t0, color="r", ls="--", label="cancel")
            ax3.set_xlabel("t [s]")
            ax3.legend(fontsize=7)
            ax3.grid(alpha=0.3)
            fig.tight_layout()
            name = f"goal_{run['k']}.png" if run["kind"] == "goal" else "cancel.png"
            fig.savefig(os.path.join(self.a.out, name), dpi=110)
            plt.close(fig)
        wd = [r for r in self.runs if r["kind"] == "watchdog"]
        if wd:
            run = wd[0]
            t0 = run["t0"]
            a = self.rec.arr(t0 - 0.5, run["t1"])
            p = [x for x in pc if x.get("owner") == run["op"]]
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            S = np.array([s[:4] for s in run["sends"]])
            axs[0].plot(S[:, 0] - t0, S[:, 1], ".", ms=3, label="velocity vx sent (20 Hz)")
            axs[0].plot(S[:, 0] - t0, S[:, 3], ".", ms=3, label="velocity wz sent")
            axs[0].axvline(run["t_last"] - t0, color="k", ls=":", label="last message")
            if run["idle_t"]:
                axs[0].axvline(run["idle_t"] - t0, color="r", ls="--",
                               label=f"planner IDLE (+{run['idle_t'] - run['t_last']:.3f} s)")
            axs[0].legend(fontsize=8)
            if p:
                tp = np.array([x["t_wall"] - t0 for x in p])
                axs[1].step(tp, [x["mode"] for x in p], where="post", label="planner mode (0 IDLE, 1 SLOW_WALK)")
                axs[1].step(tp, [x["speed"] for x in p], where="post", label="planner speed")
                axs[1].legend(fontsize=8)
            axs[2].plot(a[:, 0] - t0, np.hypot(a[:, 4], a[:, 5]), label="GT speed")
            axs[2].plot(a[:, 0] - t0, a[:, 6], label="GT wz")
            axs[2].legend(fontsize=8)
            axs[2].set_xlabel("t [s]")
            for x in axs:
                x.grid(alpha=0.3)
                x.axvline(run["t_last"] - t0, color="k", ls=":")
            fig.suptitle("velocity op watchdog (FAKE stack)")
            fig.tight_layout()
            fig.savefig(os.path.join(self.a.out, "watchdog.png"), dpi=110)
            plt.close(fig)

    def save(self):
        self.rec.stop()
        a = self.rec.arr()
        np.savetxt(os.path.join(self.a.out, "pose.csv"), a, delimiter=",",
                   header="t_wall,x,y,yaw,vx,vy,wz,pelvis_z,fallen", comments="", fmt="%.5f")
        with open(os.path.join(self.a.out, "events.jsonl"), "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        json.dump(self.phases, open(os.path.join(self.a.out, "phases.json"), "w"), indent=1)
        runs = [{k: v for k, v in r.items() if k != "sends"} for r in self.runs]
        json.dump(runs, open(os.path.join(self.a.out, "runs.json"), "w"), indent=1, default=str)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--body-log", required=True)
    ap.add_argument("--bridge-log", required=True)
    ap.add_argument("--house-dir", default="/work/worldline-g1/assets/houses/procthor-train-38")
    ap.add_argument("--tests", default="goals,unreachable,cancel,timeout,escape,watchdog,stuck",
                    help="escape and stuck use fake-P1-only ops (reset_robot / spawn_obstacle)")
    ap.add_argument("--goal-timeout", type=float, default=150.0)
    ap.add_argument("--force-stand", action="store_true")
    ap.add_argument("--label", default="FAKE P1 + FAKE deploy")
    a = ap.parse_args(argv)
    t = Nav2Test(a)
    metrics = {"label": a.label, "p1_is_fake": "FAKE" in a.label, "t_start": time.time(),
               "status0": {k: t.bc.status().get(k) for k in ("nav_backend", "in_control")}}
    try:
        metrics["stand"] = t.stand()
        for name in a.tests.split(","):
            metrics[name] = getattr(t, name)()
    finally:
        metrics["t_end"] = time.time()
        t.save()
        try:
            t.plots()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            metrics["plot_error"] = repr(e)
        metrics["pass"] = all(metrics.get(n, {}).get("pass", False) for n in a.tests.split(","))
        json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=str)
        print(f"[nav2_fake_test] PASS={metrics['pass']} " +
              " ".join(f"{n}={metrics.get(n, {}).get('pass')}" for n in a.tests.split(",")), flush=True)
    t.bc.close()
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
