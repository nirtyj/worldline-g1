"""M1 drive test: E2/E3 (+ E4 evidence) driven ONLY through BodyClient, asserted on ground truth (gt.pose).

    /work/worldline-g1/.venv/bin/python -m tools.m1_drive_test [--port-offset 0] [--out DIR] [--stand-s 60]

Records into --out (default /work/worldline-g1/outputs/m1/run-<ts>/):
  head_camera.mp4   ego_view from P1's camera PUB (sensor_server format), with a test label overlay
  topdown.mp4       ground-truth top-down schematic (occupancy + trajectory + robot), drawn from gt.pose
  trajectory.png    trajectory over the occupancy map with rooms, planned paths, waypoints, falls
  gait.png          foot contacts and leg action targets (E4)
  metrics.json      rates, RTF, per-test pass/fail with GT errors, times, falls, E4 evidence
  pose.csv, events.jsonl, planner_cmds (from body status), debug_legs.npz
A test passes only if the body reports succeeded AND the ground truth agrees. Failures are recorded as failed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import threading
import time
import traceback

import numpy as np
import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402
from body.config import ep, ports as _ports, port_offset_from_env  # noqa: E402
from body.nav_grid import NavGrid  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402
from body.wire import decode_camera_message, loads_any, split_topic, wrap, yaw_from_quat_wxyz  # noqa: E402

DEG = math.pi / 180.0


# ------------------------------------------------------------------------------------------------
# recorders
# ------------------------------------------------------------------------------------------------
class Label:
    def __init__(self):
        self.text = "init"

    def set(self, t: str):
        self.text = t


class PoseRecorder(threading.Thread):
    def __init__(self, endpoint, ctx):
        super().__init__(daemon=True)
        self.endpoint, self.ctx = endpoint, ctx
        self.rows: list[tuple] = []
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 1000)
        s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        s.connect(self.endpoint)
        while self.running:
            if not s.poll(100):
                continue
            payload = split_topic(s.recv_multipart(), b"gt.pose")
            if payload is None:
                continue
            try:
                d = loads_any(payload)
            except Exception:
                continue
            pos = d.get("base_pos") or [0, 0, 0]
            q = d.get("base_quat_wxyz") or [1, 0, 0, 0]
            yaw = d.get("yaw")
            yaw = float(yaw) if yaw is not None else yaw_from_quat_wxyz(q)
            v = d.get("base_lin_vel_w") or [float("nan")] * 3
            w = d.get("base_ang_vel_w") or [float("nan")] * 3
            fc = d.get("foot_contact") or {}
            row = (time.time(), float(d.get("t_sim", float("nan"))), float(pos[0]), float(pos[1]), float(pos[2]),
                   yaw, float(v[0]), float(v[1]), float(w[2]), float(d.get("pelvis_z", pos[2])),
                   int(bool(d.get("fallen", False))), int(bool(fc.get("left", False))),
                   int(bool(fc.get("right", False))), float(d.get("rtf", float("nan"))))
            with self.lock:
                self.rows.append(row)
        s.close(0)

    COLS = ["t_wall", "t_sim", "x", "y", "z", "yaw", "vx", "vy", "wz", "pelvis_z", "fallen", "contact_l",
            "contact_r", "rtf"]

    def arr(self, t0=None, t1=None) -> np.ndarray:
        with self.lock:
            a = np.array(self.rows, dtype=float) if self.rows else np.zeros((0, len(self.COLS)))
        if len(a) and t0 is not None:
            a = a[a[:, 0] >= t0]
        if len(a) and t1 is not None:
            a = a[a[:, 0] <= t1]
        return a

    def latest(self):
        with self.lock:
            return self.rows[-1] if self.rows else None


class CameraRecorder(threading.Thread):
    def __init__(self, endpoint, ctx, path, fps, label: Label, enabled=True):
        super().__init__(daemon=True)
        self.endpoint, self.ctx, self.path, self.fps, self.label = endpoint, ctx, path, fps, label
        self.running = True
        self.enabled = enabled
        self.frames = 0
        self.t_first = self.t_last = None
        self.lat_ms: list[float] = []
        self.shape = None
        self.error = None

    def run(self):
        if not self.enabled:
            return
        import cv2
        import imageio.v2 as imageio

        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 60)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(self.endpoint)
        w = None
        try:
            while self.running:
                if not s.poll(200):
                    continue
                ts, imgs = decode_camera_message(s.recv())
                if not imgs:
                    continue
                img = imgs.get("ego_view", next(iter(imgs.values())))
                now = time.time()
                if ts:
                    self.lat_ms.append((now - float(next(iter(ts.values())))) * 1e3)
                if w is None:
                    self.shape = img.shape
                    w = imageio.get_writer(self.path, fps=self.fps, codec="libx264", quality=7,
                                           macro_block_size=8)
                frame = np.ascontiguousarray(img).copy()
                cv2.putText(frame, f"{self.label.text}  {time.strftime('%H:%M:%S')}", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                w.append_data(frame)  # RGB (cv2 round-trip of an RGB source, see body/wire.py)
                self.frames += 1
                self.t_first = self.t_first or now
                self.t_last = now
        except Exception as e:
            self.error = repr(e)
        finally:
            if w is not None:
                w.close()
            s.close(0)

    def rate(self):
        if not self.frames or self.t_last == self.t_first:
            return 0.0
        return (self.frames - 1) / (self.t_last - self.t_first)


class DebugRecorder(threading.Thread):
    def __init__(self, endpoint, ctx):
        super().__init__(daemon=True)
        self.endpoint, self.ctx = endpoint, ctx
        self.t: list[float] = []
        self.legs: list[list[float]] = []
        self.index: list[int] = []
        self.running = True
        self.config_seen = 0

    def run(self):
        import msgpack

        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 500)
        s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        s.setsockopt(zmq.SUBSCRIBE, b"robot_config")
        s.connect(self.endpoint)
        while self.running:
            if not s.poll(100):
                continue
            f = s.recv_multipart()[0]
            if f.startswith(b"robot_config"):
                self.config_seen += 1
                continue
            try:
                d = msgpack.unpackb(f[len(b"g1_debug"):], raw=False, strict_map_key=False)
            except Exception:
                continue
            la = d.get("last_action") or []
            self.t.append(time.time())
            self.legs.append([float(v) for v in la[:12]] if len(la) >= 12 else [float("nan")] * 12)
            self.index.append(int(d.get("index", -1)))
        s.close(0)

    def window(self, t0, t1):
        t = np.array(self.t)
        m = (t >= t0) & (t <= t1)
        return t[m], np.array(self.legs)[m] if len(self.legs) else np.zeros((0, 12))


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def poly_xy(poly) -> np.ndarray:
    pts = []
    for p in poly:
        if isinstance(p, dict):
            pts.append([float(p.get("x", 0.0)), float(p.get("y", p.get("z", 0.0)))])
        else:
            pts.append([float(p[0]), float(p[1])])
    return np.array(pts)


def room_centres(nav: NavGrid, rooms: list[dict]) -> list[dict]:
    from matplotlib.path import Path

    H, W = nav.H, nav.W
    xs = nav.origin[0] + (np.arange(W) + 0.5) * nav.res
    ys = nav.origin[1] + (np.arange(H) + 0.5) * nav.res
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    out = []
    for r in rooms:
        poly = r.get("polygon") or r.get("floor_polygon") or r.get("floorPolygon")
        if not poly:
            continue
        P = poly_xy(poly)
        inside = Path(P).contains_points(pts).reshape(H, W)
        free = inside & ~nav.inflated
        if not free.any():
            continue
        c = np.where(free, nav.clear, -1)
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        x, y = nav.cell_to_world(iy, ix)
        out.append({"room_id": r.get("id") or r.get("room_id"), "name": r.get("name") or r.get("type") or
                    r.get("roomType"), "x": x, "y": y, "clearance": float(nav.clear[iy, ix]), "polygon": P})
    return out


def room_of(rooms_c: list[dict], x, y):
    from matplotlib.path import Path

    for r in rooms_c:
        if Path(r["polygon"]).contains_point((x, y)):
            return r["room_id"]
    return None


def best_heading(nav: NavGrid, x, y, n=72, max_d=6.0):
    best = (0.0, 0.0)
    for k in range(n):
        h = -math.pi + 2 * math.pi * k / n
        d = nav.ray_free_distance(x, y, h, max_d)
        if d > best[1]:
            best = (h, d)
    return best


def speed_series(a: np.ndarray) -> np.ndarray:
    """speed from P1 velocities if present, else finite differences."""
    if len(a) < 2:
        return np.zeros(len(a))
    v = np.hypot(a[:, 6], a[:, 7])
    if np.all(np.isfinite(v)):
        return v
    dt = np.diff(a[:, 0])
    dt[dt <= 0] = 1e-3
    d = np.hypot(np.diff(a[:, 2]), np.diff(a[:, 3])) / dt
    return np.concatenate([[d[0]], d])


def contact_alternation(a: np.ndarray) -> dict:
    """Count single-support phases and how many consecutive ones alternate L/R."""
    if len(a) < 3:
        return {"single_support_phases": 0, "alternations": 0, "alternation_ratio": None}
    L, R = a[:, 11].astype(int), a[:, 12].astype(int)
    state = np.where((L == 1) & (R == 0), 1, np.where((R == 1) & (L == 0), 2, 0))
    phases = []
    for s in state:
        if s and (not phases or phases[-1] != s):
            phases.append(int(s))
        elif s == 0 and phases and phases[-1] != 0:
            pass
    # collapse repeats separated by double support
    seq = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    alt = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return {"single_support_phases": len(seq), "alternations": alt,
            "alternation_ratio": None if len(seq) < 2 else round(alt / (len(seq) - 1), 3),
            "contact_l_frac": round(float(L.mean()), 3), "contact_r_frac": round(float(R.mean()), 3)}


# ------------------------------------------------------------------------------------------------
class DriveTest:
    def __init__(self, args):
        self.args = args
        self.off = port_offset_from_env() if args.port_offset is None else args.port_offset
        self.P = _ports(self.off)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.out = args.out or os.path.join("/work/worldline-g1/outputs/m1", f"run-{ts}")
        os.makedirs(self.out, exist_ok=True)
        self.ctx = zmq.Context.instance()
        self.label = Label()
        self.results: list[dict] = []
        self.events: list[dict] = []
        self.segments: list[dict] = []
        self.plans: list[dict] = []
        self.falls: list[dict] = []
        self.record_stop = None
        self.t_start = time.time()
        self.log_f = open(os.path.join(self.out, "drive_test.log"), "a", buffering=1)

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")

    # -- infra ----------------------------------------------------------------------------------
    def start(self):
        self.p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=10.0, ctx=self.ctx)
        self.stats0 = self.p1.try_call("get_stats") or {}
        self.poses = PoseRecorder(ep(self.P["p1_pose"]), self.ctx)
        self.poses.start()
        # one video frame per published frame, so play back at the publisher's rate (P1 render_hz, default 30)
        fps = float(self.args.cam_fps or self.stats0.get("render_hz") or 30.0)
        self.cam = CameraRecorder(ep(self.P["camera"]), self.ctx, os.path.join(self.out, "head_camera.mp4"),
                                  fps, self.label, enabled=not self.args.no_video)
        self.cam.start()
        self.dbg = DebugRecorder(ep(self.P["sonic_debug"]), self.ctx)
        self.dbg.start()
        self.bc = BodyClient(port_offset=self.off, ctx=self.ctx).connect(20)
        self.bc.add_listener(self._on_event)
        t0 = time.time()
        while self.poses.latest() is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self.poses.latest() is None:
            raise RuntimeError("no gt.pose")
        self.scene = self.p1.try_call("get_scene_info") or {}
        occ = self.p1.try_call("get_occupancy") or {}
        self.nav = NavGrid.from_p1_reply(occ, robot_radius=self.args.robot_radius) if occ.get("ok", True) and \
            (occ.get("path") or occ.get("npz_path") or occ.get("grid") is not None) else None
        self.rooms = room_centres(self.nav, self.scene.get("rooms") or []) if self.nav is not None else []
        self.topdown_png = None
        rep = self.p1.try_call("render_topdown", path=os.path.join(self.out, "p1_topdown.png"))
        if rep and rep.get("ok", True) and rep.get("path") and os.path.exists(rep["path"]):
            self.topdown_png = rep
        # P1 50 Hz trace of lowcmd targets / contacts / root pose (m1.md §1.6 `record`) = E4 evidence
        self.record_path = os.path.join(self.out, "p1_record.npz")
        self.record_on = self.p1.try_call("record", on=True, path=self.record_path)
        self.log(f"[setup] offset={self.off} rooms={[r['name'] for r in self.rooms]} "
                 f"nav={'yes' if self.nav is not None else 'NO'} fake={bool(self.stats0.get('fake'))}")

    def _on_event(self, ev):
        self.events.append(ev)
        if ev.get("state") == "accepted" and ev.get("op") == "go_to":
            plan = (ev.get("data") or {}).get("plan")
            if plan:
                self.plans.append({"id": ev["id"], **plan})
        if ev.get("state") == "progress" and (ev.get("data") or {}).get("phase") == "replanned":
            self.plans.append({"id": ev["id"], **ev["data"]["plan"]})
        if ev.get("state") == "failed" and (ev.get("data") or {}).get("reason") == "fallen":
            self.falls.append({"t": time.time(), "op": ev.get("op"), "pose": ev["data"].get("pose")})

    def pose(self):
        r = self.poses.latest()
        return {"x": r[2], "y": r[3], "yaw": r[5], "pelvis_z": r[9], "fallen": bool(r[10]), "t": r[0]}

    def fallen_between(self, t0, t1) -> bool:
        a = self.poses.arr(t0, t1)
        return bool(len(a) and (a[:, 10].max() > 0 or a[:, 9].min() < self.args.pelvis_z_min))

    # -- test bookkeeping ------------------------------------------------------------------------
    def run_test(self, name, fn):
        self.label.set(name)
        t0 = time.time()
        self.log(f"=== {name}")
        rec = {"name": name, "t_start": t0}
        try:
            res = fn()
            rec.update(res)
        except Exception as e:
            rec.update({"pass": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]})
        rec["t_end"] = time.time()
        rec["duration_s"] = round(rec["t_end"] - t0, 2)
        if self.fallen_between(t0, rec["t_end"]):
            rec["pass"] = False
            rec["fell"] = True
        self.results.append(rec)
        self.segments.append({"name": name, "t0": t0, "t1": rec["t_end"]})
        self.log(f"=== {name}: {'PASS' if rec.get('pass') else 'FAIL'} "
                 f"{json.dumps({k: v for k, v in rec.items() if k not in ('trace', 'body')}, default=str)[:600]}")
        return rec

    # -- tests ----------------------------------------------------------------------------------
    def t_stand(self):
        st = self.bc.status()
        stand = None
        if not st.get("in_control"):
            h = self.bc.stand(release_band=True, settle_s=2.0, verify_s=3.0, timeout=120)
            stand = h.summary()
            if not h.ok:
                return {"pass": False, "body": stand, "reason": h.reason}
        t0 = time.time()
        z = []
        while time.time() - t0 < self.args.stand_s:
            p = self.pose()
            z.append(p["pelvis_z"])
            if p["fallen"]:
                break
            time.sleep(0.2)
        a = self.poses.arr(t0, time.time())
        zmin, zmax = float(a[:, 9].min()), float(a[:, 9].max())
        drift = float(np.hypot(a[-1, 2] - a[0, 2], a[-1, 3] - a[0, 3]))
        ok = (not a[:, 10].any()) and zmin >= self.args.pelvis_z_min and zmax <= self.args.pelvis_z_max \
            and time.time() - t0 >= self.args.stand_s - 0.5
        return {"pass": bool(ok), "stand_op": stand, "hold_s": round(time.time() - t0, 2),
                "pelvis_z_min": round(zmin, 4), "pelvis_z_max": round(zmax, 4),
                "pelvis_z_mean": round(float(a[:, 9].mean()), 4), "xy_drift_m": round(drift, 4)}

    def _ensure_room(self, need_m: float) -> dict:
        """Make sure there is need_m of free space ahead-ish: turn to the best heading, or go to a better spot."""
        p = self.pose()
        if self.nav is None:
            return {"heading": p["yaw"], "free_m": None, "moved": False}
        h, d = best_heading(self.nav, p["x"], p["y"])
        moved = False
        if d < need_m and self.rooms:
            cands = sorted(((best_heading(self.nav, r["x"], r["y"])[1], r) for r in self.rooms),
                           key=lambda t: -t[0])
            for dd, r in cands:
                if dd >= need_m:
                    self.log(f"[setup] relocating to {r['name']} ({r['x']:.2f},{r['y']:.2f}) for {need_m} m of space")
                    hh = self.bc.go_to(r["x"], r["y"], timeout_s=180)
                    moved = hh.summary()
                    p = self.pose()
                    h, d = best_heading(self.nav, p["x"], p["y"])
                    break
        return {"heading": h, "free_m": d, "moved": moved}

    def t_turn90(self):
        p = self.pose()
        sign = 1.0
        if self.nav is not None:
            dl = self.nav.ray_free_distance(p["x"], p["y"], p["yaw"] + math.pi / 2)
            dr = self.nav.ray_free_distance(p["x"], p["y"], p["yaw"] - math.pi / 2)
            sign = 1.0 if dl >= dr else -1.0
        target = wrap(p["yaw"] + sign * math.pi / 2)
        h = self.bc.turn_to(target, timeout=60)
        q = self.pose()
        err = math.degrees(wrap(target - q["yaw"]))
        disp = math.hypot(q["x"] - p["x"], q["y"] - p["y"])
        ok = h.ok and abs(err) <= 15.0 and disp <= 0.3
        return {"pass": bool(ok), "target_deg": round(math.degrees(target), 2), "turned_deg":
                round(math.degrees(wrap(q["yaw"] - p["yaw"])), 2), "yaw_err_deg": round(err, 2),
                "displacement_m": round(disp, 3), "body": h.summary()}

    def t_walk_forward(self):
        room = self._ensure_room(2.6)
        p = self.pose()
        pre = None
        if abs(wrap(room["heading"] - p["yaw"])) > 5 * DEG:
            pre = self.bc.turn_to(room["heading"], timeout=60).summary()
        p = self.pose()
        dur = self.args.walk_dist / self.args.walk_speed / 0.9 + 0.6
        h = self.bc.walk(vx=self.args.walk_speed, duration_s=dur)
        q = self.pose()
        dx, dy = q["x"] - p["x"], q["y"] - p["y"]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        fwd, lat = c * dx + s * dy, -s * dx + c * dy
        ok = h.ok and fwd >= 2.0 and abs(lat) <= 0.5
        return {"pass": bool(ok), "forward_m": round(fwd, 3), "lateral_m": round(lat, 3),
                "yaw_change_deg": round(math.degrees(wrap(q["yaw"] - p["yaw"])), 2), "duration_cmd_s": dur,
                "free_ahead_m": room["free_m"], "pre_turn": pre, "relocated": room["moved"], "body": h.summary()}

    def t_strafe(self):
        p = self.pose()
        side = 1.0
        if self.nav is not None:
            dl = self.nav.ray_free_distance(p["x"], p["y"], p["yaw"] + math.pi / 2)
            dr = self.nav.ray_free_distance(p["x"], p["y"], p["yaw"] - math.pi / 2)
            side = 1.0 if dl >= dr else -1.0
            if max(dl, dr) < 1.0:
                room = self._ensure_room(1.5)
                self.bc.turn_to(wrap(room["heading"] - side * math.pi / 2), timeout=60)
                p = self.pose()
        h = self.bc.walk(vy=side * 0.3, duration_s=3.0)
        q = self.pose()
        dx, dy = q["x"] - p["x"], q["y"] - p["y"]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        fwd, lat = c * dx + s * dy, -s * dx + c * dy
        dyaw = math.degrees(wrap(q["yaw"] - p["yaw"]))
        ok = h.ok and side * lat >= 0.5 and abs(dyaw) <= 20.0 and abs(fwd) <= 0.5
        return {"pass": bool(ok), "side": "left" if side > 0 else "right", "lateral_m": round(lat, 3),
                "forward_m": round(fwd, 3), "yaw_change_deg": round(dyaw, 2), "body": h.summary()}

    def t_stop_mid_walk(self):
        room = self._ensure_room(2.5)
        p = self.pose()
        if abs(wrap(room["heading"] - p["yaw"])) > 5 * DEG:
            self.bc.turn_to(room["heading"], timeout=60)
        hw = self.bc.walk(vx=self.args.walk_speed, duration_s=12.0, wait=False)
        time.sleep(2.2)
        t_cmd = time.time()
        v_before = float(speed_series(self.poses.arr(t_cmd - 0.3, t_cmd)).mean()) if len(
            self.poses.arr(t_cmd - 0.3, t_cmd)) else float("nan")
        hs = self.bc.stop(timeout=10)
        try:
            hw.wait(5)
        except TimeoutError:
            pass
        time.sleep(2.0)
        a = self.poses.arr(t_cmd, time.time())
        v = speed_series(a)
        stopped_idx = None
        for i in range(len(v)):
            if np.all(v[i:i + 5] < self.args.stop_v_eps):
                stopped_idx = i
                break
        t_stop_gt = None if stopped_idx is None else float(a[stopped_idx, 0] - t_cmd)
        upright = not a[:, 10].any() and a[:, 9].min() >= self.args.pelvis_z_min
        ok = hs.ok and hw.state == "canceled" and t_stop_gt is not None and t_stop_gt <= 1.5 and upright
        return {"pass": bool(ok), "v_before_stop": round(v_before, 3), "stop_time_gt_s": t_stop_gt,
                "stop_time_body_s": (hs.result or {}).get("stop_time_s"), "walk_state": hw.state,
                "upright_after": bool(upright), "body_stop": hs.summary(), "body_walk": hw.summary()}

    def _pick_waypoints(self):
        if self.args.waypoints:
            wps = []
            for part in self.args.waypoints.split(";"):
                v = [float(t) for t in part.split(",")]
                wps.append((v[0], v[1], None if len(v) < 3 else v[2] * DEG))
            return wps, "manual"
        if self.nav is None or len(self.rooms) < 2:
            raise RuntimeError("need occupancy + >= 2 rooms from get_scene_info to pick waypoints")
        p = self.pose()
        start_room = room_of(self.rooms, p["x"], p["y"])
        reach = []
        for r in self.rooms:
            res = self.nav.plan((p["x"], p["y"]), (r["x"], r["y"]))
            if res.ok and math.hypot(r["x"] - p["x"], r["y"] - p["y"]) > 1.0:
                reach.append((res.length, r))
        reach.sort(key=lambda t: t[0])
        others = [r for _, r in reach if r["room_id"] != start_room]
        if not others:
            raise RuntimeError("no reachable room other than the start room")
        seq = [others[0]]
        rest = [r for r in others[1:]]
        seq.append(rest[0] if rest else next((r for _, r in reach if r["room_id"] == start_room),
                                              {"x": p["x"], "y": p["y"], "room_id": start_room, "name": "start"}))
        if len(rest) >= 2:
            seq.append(rest[1])
        else:
            seq.append({"x": p["x"], "y": p["y"], "room_id": start_room, "name": "start"})
        wps = []
        prev = (p["x"], p["y"])
        for r in seq:
            yaw = math.atan2(r["y"] - prev[1], r["x"] - prev[0])
            wps.append((r["x"], r["y"], yaw))
            prev = (r["x"], r["y"])
        return wps, [r.get("name") for r in seq]

    def t_goto(self):
        wps, how = self._pick_waypoints()
        legs = []
        rooms_visited = set()
        allok = True
        for i, (x, y, yaw) in enumerate(wps):
            self.label.set(f"go_to #{i + 1} ({x:.1f},{y:.1f})")
            p0 = self.pose()
            h = self.bc.go_to(x, y, yaw=yaw, timeout_s=self.args.goto_timeout)
            q = self.pose()
            perr = math.hypot(q["x"] - x, q["y"] - y)
            yerr = None if yaw is None else math.degrees(wrap(yaw - q["yaw"]))
            ok = h.ok and perr <= 0.3 and (yerr is None or abs(yerr) <= 15.0)
            allok &= ok
            room = room_of(self.rooms, q["x"], q["y"]) if self.rooms else None
            if room:
                rooms_visited.add(room)
            res = h.result or {}
            straight = math.hypot(x - p0["x"], y - p0["y"])
            legs.append({"i": i + 1, "goal": [round(x, 3), round(y, 3)],
                         "goal_yaw_deg": None if yaw is None else round(math.degrees(yaw), 2),
                         "pass": bool(ok), "state": h.state, "reason": h.reason, "pos_err_gt": round(perr, 4),
                         "yaw_err_gt_deg": None if yerr is None else round(yerr, 2), "room": room,
                         "path_len_m": res.get("path_len_m"), "straight_m": round(straight, 3),
                         "detour_ratio": None if not res.get("path_len_m") or straight < 1e-6 else
                         round(res["path_len_m"] / straight, 3), "replans": res.get("replans"),
                         "walked_m": res.get("walked_m"), "duration_s": res.get("duration_s"),
                         "body": {k: v for k, v in res.items() if k not in ("plans",)}})
            if h.state == "failed" and h.reason == "fallen":
                break
        ok = allok and len(wps) >= 3 and len(rooms_visited) >= 2
        return {"pass": bool(ok), "waypoints_from": how, "rooms_visited": sorted(rooms_visited), "legs": legs}

    # -- evidence --------------------------------------------------------------------------------
    def e4(self) -> dict:
        walk_windows = [(s["t0"], s["t1"]) for s in self.segments if s["name"] in ("E3_walk_forward", "E3_goto",
                                                                                    "E3_strafe", "E3_stop")]
        dbg_t = np.array(self.dbg.t)
        rate = None
        if len(dbg_t) > 10:
            rate = round(float((len(dbg_t) - 1) / (dbg_t[-1] - dbg_t[0])), 2)
        legs_changes, legs_std = [], []
        contacts = []
        for t0, t1 in walk_windows:
            t, L = self.dbg.window(t0, t1)
            if len(L) > 2:
                ch = np.any(np.abs(np.diff(L, axis=0)) > 1e-6, axis=1)
                legs_changes.append(float(ch.mean()))
                legs_std.append(float(np.nanstd(L, axis=0).mean()))
            contacts.append(contact_alternation(self.poses.arr(t0, t1)))
        st = self.bc.status()
        mux = st.get("mux") or {}
        stats1 = self.p1.try_call("get_stats") or {}
        return {
            "g1_debug_rate_hz": rate,
            "g1_debug_msgs": len(dbg_t),
            "leg_target_change_frac_walking": None if not legs_changes else round(float(np.mean(legs_changes)), 3),
            "leg_target_std_walking": None if not legs_std else round(float(np.mean(legs_std)), 4),
            "planner_msgs_by_mode": (mux.get("stats") or {}).get("by_mode"),
            "planner_keepalive_hz": mux.get("rate_hz"),
            "planner_command_stop_sent": (mux.get("stats") or {}).get("command_stop_sent"),
            "planner_frame": mux.get("frame"),
            "foot_contacts_walking": contacts,
            "p1_stats_start": self.stats0, "p1_stats_end": stats1,
            "root_writes_reported_by_p1": {"start": self.stats0.get("root_writes"), "end": stats1.get("root_writes")},
            "lowcmd_rx_hz_p1": stats1.get("lowcmd_rx_hz"),
            "lowcmd_leg_change_hz_p1": stats1.get("lowcmd_leg_change_hz"),
            "p1_record": self.record_stop,
            "deploy": st.get("deploy"),
        }

    def write_outputs(self):
        self.record_stop = self.p1.try_call("record", on=False, path=self.record_path) if self.record_on else None
        a = self.poses.arr()
        with open(os.path.join(self.out, "pose.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(PoseRecorder.COLS)
            w.writerows(a.tolist())
        with open(os.path.join(self.out, "events.jsonl"), "w") as f:
            for ev in self.events:
                f.write(json.dumps(ev, default=str) + "\n")
        np.savez(os.path.join(self.out, "debug_legs.npz"), t=np.array(self.dbg.t), legs=np.array(self.dbg.legs),
                 index=np.array(self.dbg.index))
        rtf = a[:, 13] if len(a) else np.zeros(0)
        rtf = rtf[np.isfinite(rtf)]
        dur = (a[-1, 0] - a[0, 0]) if len(a) > 1 else 0.0
        e4 = self.e4()
        tests = {r["name"]: {k: v for k, v in r.items() if k != "trace"} for r in self.results}
        metrics = {
            "run_dir": self.out, "t_start": self.t_start, "t_end": time.time(), "port_offset": self.off,
            "p1_is_fake": bool(self.stats0.get("fake")), "house": self.scene.get("house_id"),
            "rates": {"gt_pose_hz": None if dur <= 0 else round((len(a) - 1) / dur, 2),
                      "camera_hz": round(self.cam.rate(), 2), "camera_frames": self.cam.frames,
                      "camera_latency_ms_p50": None if not self.cam.lat_ms else round(float(np.median(self.cam.lat_ms)), 1),
                      "g1_debug_hz": e4["g1_debug_rate_hz"], "planner_keepalive_hz": e4["planner_keepalive_hz"],
                      "physics_hz_p1": e4["p1_stats_end"].get("physics_hz_1s"),
                      "physics_hz_total_p1": e4["p1_stats_end"].get("physics_hz_total"),
                      "render_hz_p1": e4["p1_stats_end"].get("render_hz"),
                      "camera_pub_hz_p1": e4["p1_stats_end"].get("camera_pub_hz"),
                      "lowstate_pub_hz_p1": e4["p1_stats_end"].get("lowstate_pub_hz"),
                      "lowcmd_rx_hz_p1": e4["p1_stats_end"].get("lowcmd_rx_hz")},
            "rtf": {"gt_pose_mean": None if not len(rtf) else round(float(rtf.mean()), 4),
                    "gt_pose_min": None if not len(rtf) else round(float(rtf.min()), 4),
                    "gt_pose_p05": None if not len(rtf) else round(float(np.percentile(rtf, 5)), 4),
                    "p1_rtf_total": e4["p1_stats_end"].get("rtf_total"),
                    "p1_rtf_10s_end": e4["p1_stats_end"].get("rtf_10s")},
            "falls": self.falls, "fell_any": bool(len(a) and a[:, 10].any()),
            "tests": tests,
            "summary": {r["name"]: bool(r.get("pass")) for r in self.results},
            "all_pass": all(bool(r.get("pass")) for r in self.results) and bool(self.results),
            "e4": e4,
            "camera_error": self.cam.error,
        }
        with open(os.path.join(self.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        try:
            self.plot_trajectory(a)
            self.plot_gait(a)
        except Exception:
            self.log(f"[plot] failed: {traceback.format_exc()}")
        if not self.args.no_video:
            try:
                self.render_topdown_video(a)
            except Exception:
                self.log(f"[topdown] failed: {traceback.format_exc()}")
        return metrics

    def plot_trajectory(self, a):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
        if self.topdown_png and self.topdown_png.get("extent"):
            try:
                bg = plt.imread(self.topdown_png["path"])
                x0, y0, x1, y1 = self.topdown_png["extent"]  # [xmin,ymin,xmax,ymax], image up = +y (m1.md §1.6)
                ax.imshow(bg, extent=[x0, x1, y0, y1], origin="upper", alpha=0.55, zorder=0)
            except Exception:
                pass
        if self.nav is not None:
            img = np.zeros(self.nav.raw.shape)
            img[self.nav.inflated] = 0.35
            img[self.nav.blocked_raw] = 1.0
            ax.imshow(np.ma.masked_where(img == 0, img), origin="lower", extent=self.nav.extent(), cmap="Greys",
                      vmin=0, vmax=1, alpha=0.8 if self.topdown_png else 1.0, zorder=1)
        for r in self.rooms:
            P = np.vstack([r["polygon"], r["polygon"][:1]])
            ax.plot(P[:, 0], P[:, 1], "-", color="tab:blue", lw=0.8, alpha=0.6)
            ax.text(r["x"], r["y"], r["name"], fontsize=8, color="tab:blue", ha="center", alpha=0.8)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for k, s in enumerate(self.segments):
            m = (a[:, 0] >= s["t0"]) & (a[:, 0] <= s["t1"]) if len(a) else []
            if len(a) and m.any():
                ax.plot(a[m, 2], a[m, 3], "-", lw=1.6, color=colors[k % 10], label=s["name"])
        for pl in self.plans:
            P = np.array(pl.get("path") or [])
            if len(P):
                ax.plot(P[:, 0], P[:, 1], "--", lw=0.8, color="tab:orange", alpha=0.8)
        for r in self.results:
            for leg in r.get("legs", []) or []:
                gx, gy = leg["goal"]
                ax.plot(gx, gy, "*", ms=12, color="green" if leg["pass"] else "red")
                if leg["goal_yaw_deg"] is not None:
                    yy = leg["goal_yaw_deg"] * DEG
                    ax.arrow(gx, gy, 0.35 * math.cos(yy), 0.35 * math.sin(yy), width=0.02, color="green")
                ax.text(gx + 0.1, gy + 0.1, f"#{leg['i']} err {leg['pos_err_gt']:.2f}m", fontsize=7)
        if len(a):
            ax.plot(a[0, 2], a[0, 3], "o", color="k", label="start")
            f = a[:, 10] > 0
            if f.any():
                ax.plot(a[f, 2], a[f, 3], "x", color="red", ms=10, label="fallen")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m] (Isaac world)")
        ax.set_ylabel("y [m]")
        fake = " [FAKE P1/deploy]" if self.stats0.get("fake") else ""
        ax.set_title(f"M1 drive test {os.path.basename(self.out)}{fake}: GT trajectory, dashed = A* plans")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "trajectory.png"))
        plt.close(fig)

    def plot_gait(self, a):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(12, 7), dpi=100, sharex=True)
        t0 = a[0, 0] if len(a) else time.time()
        if len(a):
            axs[0].plot(a[:, 0] - t0, speed_series(a), lw=0.8)
            axs[0].set_ylabel("|v| [m/s]")
            axs[1].plot(a[:, 0] - t0, a[:, 11] + 0.05, lw=0.6, label="left")
            axs[1].plot(a[:, 0] - t0, a[:, 12] - 0.05, lw=0.6, label="right")
            axs[1].set_ylabel("foot contact")
            axs[1].legend(fontsize=7)
        if self.dbg.t:
            L = np.array(self.dbg.legs)
            T = np.array(self.dbg.t) - t0
            for j, nm in ((0, "L hip pitch"), (3, "L knee"), (6, "R hip pitch"), (9, "R knee")):
                axs[2].plot(T, L[:, j], lw=0.6, label=nm)
            axs[2].legend(fontsize=7)
        axs[2].set_ylabel("g1_debug last_action")
        axs[2].set_xlabel("t [s]")
        for s in self.segments:
            for ax in axs:
                ax.axvspan(s["t0"] - t0, s["t1"] - t0, alpha=0.06)
            axs[0].text(s["t0"] - t0, axs[0].get_ylim()[1] * 0.9, s["name"], fontsize=6, rotation=90)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "gait.png"))
        plt.close(fig)

    def render_topdown_video(self, a, fps=10):
        import cv2
        import imageio.v2 as imageio

        if self.nav is None or len(a) < 2:
            return
        scale = max(1, int(round(0.02 / self.nav.res))) if self.nav.res < 0.02 else 1
        px_per_m = 1.0 / self.nav.res * scale
        base = np.full((self.nav.H * scale, self.nav.W * scale, 3), 245, np.uint8)
        infl = np.kron(self.nav.inflated, np.ones((scale, scale), bool))
        raw = np.kron(self.nav.blocked_raw, np.ones((scale, scale), bool))
        base[infl] = (200, 200, 200)
        base[raw] = (60, 60, 60)
        base = base[::-1].copy()  # y up

        def to_px(x, y):
            return (int((x - self.nav.origin[0]) * px_per_m),
                    int(base.shape[0] - (y - self.nav.origin[1]) * px_per_m))

        for pl in self.plans:
            P = pl.get("path") or []
            for p, q in zip(P[:-1], P[1:]):
                cv2.line(base, to_px(*p), to_px(*q), (255, 160, 0), 1)
        for r in self.rooms:
            cv2.putText(base, str(r["name"]), to_px(r["x"] - 0.4, r["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (40, 90, 200), 1)
        up = max(1, int(720 / base.shape[0])) if base.shape[0] < 720 else 1
        w = imageio.get_writer(os.path.join(self.out, "topdown.mp4"), fps=fps, codec="libx264", quality=7,
                               macro_block_size=8)
        ts = np.arange(a[0, 0], a[-1, 0], 1.0 / fps)
        idx = np.searchsorted(a[:, 0], ts).clip(0, len(a) - 1)
        trail = []
        for t, i in zip(ts, idx):
            fr = base.copy()
            x, y, yaw = a[i, 2], a[i, 3], a[i, 5]
            trail.append(to_px(x, y))
            if len(trail) > 1:
                cv2.polylines(fr, [np.array(trail, np.int32)], False, (0, 120, 255), 2)
            c = to_px(x, y)
            cv2.circle(fr, c, int(0.2 * px_per_m), (255, 0, 0) if not a[i, 10] else (0, 0, 255), 2)
            cv2.line(fr, c, (int(c[0] + 0.35 * px_per_m * math.cos(yaw)), int(c[1] - 0.35 * px_per_m * math.sin(yaw))),
                     (255, 0, 0), 2)
            seg = next((s["name"] for s in self.segments if s["t0"] <= t <= s["t1"]), "")
            cv2.putText(fr, f"{seg}  t={t - a[0, 0]:.1f}s  z={a[i, 9]:.2f}  GT top-down", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if up > 1:
                fr = cv2.resize(fr, (fr.shape[1] * up, fr.shape[0] * up), interpolation=cv2.INTER_NEAREST)
            w.append_data(fr)
        w.close()

    def close(self):
        self.label.set("done")
        for r in (self.poses, self.cam, self.dbg):
            r.running = False
        self.cam.join(5)
        self.bc.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stand-s", type=float, default=60.0)
    ap.add_argument("--tests", default="stand,turn,walk,strafe,stop,goto")
    ap.add_argument("--waypoints", default=None, help="x,y,yaw_deg;x,y,yaw_deg;... (default: room centres)")
    ap.add_argument("--walk-speed", type=float, default=0.5)
    ap.add_argument("--walk-dist", type=float, default=2.3)
    ap.add_argument("--goto-timeout", type=float, default=150.0)
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--pelvis-z-min", type=float, default=0.55)
    ap.add_argument("--pelvis-z-max", type=float, default=1.0)
    ap.add_argument("--stop-v-eps", type=float, default=0.05)
    ap.add_argument("--cam-fps", type=float, default=None, help="mp4 fps (default: P1 get_stats render_hz or 30)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)
    dt = DriveTest(args)
    dt.start()
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    table = {"stand": ("E2_stand", dt.t_stand), "turn": ("E3_turn90", dt.t_turn90),
             "walk": ("E3_walk_forward", dt.t_walk_forward), "strafe": ("E3_strafe", dt.t_strafe),
             "stop": ("E3_stop", dt.t_stop_mid_walk), "goto": ("E3_goto", dt.t_goto)}
    try:
        for t in tests:
            name, fn = table[t]
            rec = dt.run_test(name, fn)
            if rec.get("fell"):
                dt.log("robot fell: stopping the sequence (reported as failed)")
                break
            if name == "E2_stand" and not rec.get("pass") and not dt.bc.status().get("in_control"):
                dt.log("stand failed and SONIC is not in control: stopping")
                break
    finally:
        m = dt.write_outputs()
        dt.close()
    dt.log(f"RESULT all_pass={m['all_pass']} summary={m['summary']} out={dt.out}")
    return 0 if m["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
