"""Fake P1 (wl-isaac) for testing wl-body and the runtime without Isaac.

Implements the P1 side of docs/contracts/m1.md and docs/contracts/p1_m2b.md with a kinematic robot in a synthetic
3-room house (or a real house's occupancy + objects with --house-dir):
  REP  p1_rep  : ping, get_pose, get_scene_info, get_occupancy, band, reset_robot, get_stats, render_topdown,
                 record, shutdown, plus test-only spawn_obstacle {x,y,r} (blocks the fake physics but NOT the published
                 map); M2b: get_objects, attach, detach, release_all, get_cameras, camera, set_render_rates,
                 get_link_poses, detections, reset_scene, move_object, set_object_pose, push_object, get_health
  PUB  p1_pose : multipart [b"gt.pose", msgpack] at 50 Hz (+ M2b "links", "waist_q"); [b"gt.objects", ..] at 10 Hz;
                 [b"gt.event", ..] (band, reset_robot, robot_fell, object_fell, attach, detach, camera, ...);
                 [b"sim.health", ..] at 1 Hz
  PUB  camera  : head camera, gear_sonic sensor_server msgpack, key "head" + the M2b frame metadata, at 10 Hz
  PUB  ego     : ego_view (5566 + offset), same format, key "ego_view", only while a consumer enabled it
  SUB  fake_link: [b"twist", msgpack {vx, vy, wz, damping}] from tools/fake_deploy.py (stands in for rt/lowcmd)

Physics: 200 Hz integration of the commanded world twist, disk-vs-occupancy collision (robot radius 0.15 m) with
axis sliding; band on => robot held; damping (deploy command{stop}) with band off => the robot "collapses"
(fallen=True, pelvis_z=0.3) exactly like the real deploy's CreateDampingCommand would cause.
Objects: dynamic props slide when pushed (friction 2 m/s^2), drop to the next support (furniture top or floor) when
they leave it, and follow a palm while attached. Images are synthetic; `detections` is a frustum count (no
occlusion), labelled method "fake-frustum". Everything the wire carries is built with sim_isaac.wire, the module the
real P1 uses.

Run:  python -m tools.fake_p1 --port-offset 200 --out /tmp/fake_p1
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import queue
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from body.config import ep, ports as _ports
from body.wire import dumps_json, loads_any, quat_wxyz_from_yaw, wrap
from sim_isaac import wire as W
from sim_isaac.mathutil import quat_from_yaw, quat_mul

RES = 0.05
ORIGIN = (-0.5, -0.5)
SIZE = (11.0, 7.0)   # x, y extent (m)

ROOMS = [
    {"id": "room_kitchen", "name": "kitchen", "type": "Kitchen", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
    {"id": "room_dining", "name": "dining room", "type": "DiningRoom", "polygon": [[0, 3], [4, 3], [4, 6], [0, 6]]},
    {"id": "room_living", "name": "living room", "type": "LivingRoom", "polygon": [[4, 0], [10, 0], [10, 6], [4, 6]]},
]

# furniture boxes: id, category, room, (xmin, ymin, xmax, ymax), height
FURNITURE = [
    ("CounterTop|1", "CounterTop", "room_kitchen", (0.0, 0.0, 0.6, 2.2), 0.9),
    ("Fridge|1", "Fridge", "room_kitchen", (3.2, 0.0, 3.9, 0.7), 1.8),
    ("DiningTable|1", "DiningTable", "room_dining", (1.2, 4.2, 2.8, 5.2), 0.75),
    ("Sofa|1", "Sofa", "room_living", (8.9, 0.6, 9.9, 3.2), 0.8),
    ("CoffeeTable|1", "CoffeeTable", "room_living", (6.6, 2.3, 7.6, 3.5), 0.45),
    ("TVStand|1", "TVStand", "room_living", (5.0, 5.4, 7.0, 5.9), 0.6),
]
# small objects on furniture (dynamic props: they move when pushed, attached or reset)
OBJECTS = [
    ("Apple|1", "Apple", "room_kitchen", (0.3, 1.0, 0.95)),
    ("Mug|1", "Mug", "room_dining", (2.0, 4.7, 0.8)),
    ("RemoteControl|1", "RemoteControl", "room_living", (7.1, 2.9, 0.5)),
]
SPAWN = (1.8, 1.4, 1.1)  # kitchen, facing roughly +y (non-zero yaw catches planner-frame bugs)

# fake arm geometry: palms by the hips, fingers pointing 60 deg below the heading (pelvis frame)
FAKE_PALM_B = {"left": (0.10, 0.22, -0.12), "right": (0.10, -0.22, -0.12)}
FAKE_PALM_PITCH = math.radians(60.0)
TORSO_FROM_PELVIS = (-0.0039635, 0.0, 0.044)
G = 9.81
FRICTION_DECEL = 2.0


def build_occupancy() -> np.ndarray:
    W_ = int(round(SIZE[0] / RES))
    H = int(round(SIZE[1] / RES))
    occ = np.zeros((H, W_), dtype=np.uint8)

    def box(x0, y0, x1, y1):
        ix0 = int(math.floor((x0 - ORIGIN[0]) / RES))
        ix1 = int(math.ceil((x1 - ORIGIN[0]) / RES))
        iy0 = int(math.floor((y0 - ORIGIN[1]) / RES))
        iy1 = int(math.ceil((y1 - ORIGIN[1]) / RES))
        occ[max(iy0, 0):max(iy1, 0), max(ix0, 0):max(ix1, 0)] = 1

    t = 0.1
    # outer walls
    box(-t, -t, 10 + t, 0)
    box(-t, 6, 10 + t, 6 + t)
    box(-t, -t, 0, 6 + t)
    box(10, -t, 10 + t, 6 + t)
    # x = 4 wall with doors y in [1.0, 2.0] and [4.0, 5.0]
    box(4 - t / 2, 0, 4 + t / 2, 1.0)
    box(4 - t / 2, 2.0, 4 + t / 2, 4.0)
    box(4 - t / 2, 5.0, 4 + t / 2, 6)
    # y = 3 wall (kitchen/dining) with door x in [1.6, 2.6]
    box(0, 3 - t / 2, 1.6, 3 + t / 2)
    box(2.6, 3 - t / 2, 4, 3 + t / 2)
    for _, _, _, (x0, y0, x1, y1), _ in FURNITURE:
        box(x0, y0, x1, y1)
    # outside the house is unknown/blocked
    ys = ORIGIN[1] + (np.arange(H) + 0.5) * RES
    xs = ORIGIN[0] + (np.arange(W_) + 0.5) * RES
    outside = (xs[None, :] < -t) | (xs[None, :] > 10 + t) | (ys[:, None] < -t) | (ys[:, None] > 6 + t)
    occ[outside] = 1
    return occ


def scene_info() -> dict:
    objs = []
    for oid, cat, room, (x0, y0, x1, y1), h in FURNITURE:
        objs.append({"id": oid, "name": f"{cat.lower()}_1", "category": cat, "room_id": room,
                     "pos": [(x0 + x1) / 2, (y0 + y1) / 2, h / 2], "aabb": [[x0, y0, 0.0], [x1, y1, h]],
                     "is_static": True, "articulated": False, "prim_path": f"/World/House/Geometry/{cat}_1",
                     "body_path": None})
    for oid, cat, room, (x, y, z) in OBJECTS:
        objs.append({"id": oid, "name": f"{cat.lower()}_1", "category": cat, "room_id": room, "pos": [x, y, z],
                     "aabb": [[x - 0.05, y - 0.05, z - 0.05], [x + 0.05, y + 0.05, z + 0.05]],
                     "is_static": False, "articulated": False, "prim_path": f"/World/House/Geometry/{cat}_1",
                     "body_path": f"/World/House/Geometry/{cat}_1"})
    return {"house_id": "fake-3room", "rooms": ROOMS, "objects": objs,
            "spawn": {"x": SPAWN[0], "y": SPAWN[1], "yaw": SPAWN[2]}, "frame": "world, z-up, meters"}


def _jpeg_b64(rgb: np.ndarray, q: int = 80) -> str:
    """The P1 convention: cv2.imencode of an RGB array (cv2.imdecode returns RGB). PIL fallback without cv2."""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return base64.b64encode(buf).decode("utf-8")
    except ImportError:
        import io

        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb[..., ::-1])).save(b, format="JPEG", quality=q)
        return base64.b64encode(b.getvalue()).decode("utf-8")


def _put_text(img: np.ndarray, text: str) -> None:
    try:
        import cv2
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except ImportError:
        img[8:36, 8:8 + min(620, 8 * len(text))] = 255


class FakeObject:
    def __init__(self, o: dict):
        self.id, self.name = str(o["id"]), str(o.get("name") or o["id"])
        self.aabb0 = [list(map(float, o["aabb"][0])), list(map(float, o["aabb"][1]))]
        self.pos0 = [float(v) for v in o["pos"]]
        self.p0 = W.box_center(self.aabb0)          # the fake's body origin = the box centre
        self.q0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.p = self.p0.copy()
        self.q = self.q0.copy()
        self.v = np.zeros(3)
        self.held: str | None = None
        self.mode: str | None = None
        self.off_p = np.zeros(3)
        self.off_q = np.array([1.0, 0.0, 0.0, 0.0])

    def aabb(self):
        return W.moved_box(self.aabb0, self.p0, self.q0, self.p, self.q)

    def record(self) -> dict:
        return W.object_record(self.id, self.name, W.moved_point(self.pos0, self.p0, self.q0, self.p, self.q), self.q,
                               self.aabb(), held_by=self.held, dynamic=True, lin_vel=self.v)


class FakeP1:
    def __init__(self, port_offset: int = 200, out_dir: str = "/tmp/fake_p1", pose_hz: float = 50.0,
                 cam_hz: float = 10.0, physics_hz: float = 200.0, ctx: zmq.Context | None = None,
                 robot_radius: float = 0.15, log=print, house_dir: str | None = None, objects_hz: float = 10.0,
                 health_hz: float = 1.0, rtf: float | None = None):
        self.P = _ports(port_offset)
        self.P["ego"] = W.EGO_PORT + port_offset
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ctx = ctx or zmq.Context.instance()
        self.pose_hz, self.cam_hz, self.physics_hz = pose_hz, cam_hz, physics_hz
        self.objects_hz, self.health_hz = objects_hz, health_hz
        self.robot_radius = robot_radius
        self.log = log
        self.res, self.origin = RES, ORIGIN
        self.info = scene_info()
        if house_dir:
            # a real MolmoSpaces house prepared by scenes/ (occupancy.npz + house_info.json): real geometry,
            # still a kinematic robot
            import json as _json
            with np.load(os.path.join(house_dir, "occupancy.npz")) as z:
                self.occ = (z["occ"] > 0).astype(np.uint8)
                self.res = float(z["resolution"])
                self.origin = (float(z["origin"][0]), float(z["origin"][1]))
            hi = _json.load(open(os.path.join(house_dir, "house_info.json")))
            rooms = [{"id": r.get("room_id", r.get("id")), "name": r.get("name"), "type": r.get("type"),
                      "polygon": r.get("polygon")} for r in hi.get("rooms", [])]
            self.info = {"house_id": hi.get("house_id"), "rooms": rooms, "objects": hi.get("objects", []),
                         "spawn": hi.get("spawn"), "frame": "world, z-up, meters", "source": house_dir}
        else:
            self.occ = build_occupancy()
        sp = self.info["spawn"]
        self.spawn = (float(sp["x"]), float(sp["y"]), float(sp.get("yaw", 0.0)))
        self.phys_occ = self.occ.copy()
        self.occ_path = os.path.join(out_dir, "occupancy.npz")
        # npz layout of docs/contracts/m1.md §1.6 get_occupancy: occ (raw), occ_inflated, resolution, origin
        from scipy import ndimage
        infl = ndimage.distance_transform_edt(self.occ == 0) * self.res < 0.25
        np.savez(self.occ_path, occ=self.occ, occ_inflated=infl.astype(np.uint8), resolution=np.float32(self.res),
                 origin=np.array(self.origin), robot_radius=np.float32(0.25))
        self._lock = threading.RLock()
        self.x, self.y, self.yaw = self.spawn
        self.vx = self.vy = self.wz = 0.0
        self.band_on = True
        self.collapsed = False
        self.pelvis_z = 0.78
        self.t_sim = 0.0
        self.twist = (0.0, 0.0, 0.0)
        self.twist_mono = 0.0
        self.link_count = 0
        self.collisions = 0
        self.root_writes = 0
        self.phase = 0.0
        self.contact = (True, True)
        self._running = False
        self._threads: list[threading.Thread] = []
        self.step_ms = 0.0
        self.pose_sent = 0
        self.cam_sent = 0
        self.pose_paused = False  # test hook: simulate a P1 stall (gt.pose stops)
        self.shutdown_requested = False
        # ---- M2b state (docs/contracts/p1_m2b.md)
        objs = self.info.get("objects") or []
        self.static_objs = {str(o["id"]): o for o in objs
                            if not (o.get("body_path") and not o.get("articulated") and not o.get("is_static"))}
        self.dyn: dict[str, FakeObject] = {str(o["id"]): FakeObject(o) for o in objs
                                           if str(o["id"]) not in self.static_objs}
        self.fall = W.FallDetector(floor_z=0.0)
        self._seed_fall()
        self.events: list[dict] = []
        self._evq: queue.Queue = queue.Queue()
        self.attach_count = self.detach_count = self.object_writes = 0
        # The RTF gt.pose and sim.health report. None: measured, t_sim over wall time since start (the CLI default).
        # A number pins it: a kinematic fake has no real-time budget, so on a loaded test machine the measured ratio
        # reads as a DEGRADED sim that is not there (sim_isaac/tests pin 1.0). Tests change it at run time.
        self.rtf_override: float | None = rtf
        self.render_seq = 0
        self.cams = {"head": {"spec": W.HEAD, "on": True, "hz": float(cam_hz), "consumers": W.Consumers(),
                              "seq": 0},
                     "ego_view": {"spec": W.EGO_VIEW, "on": False, "hz": float(cam_hz), "consumers": W.Consumers(),
                                  "seq": 0}}
        self.cams["head"]["consumers"].add("default", time.monotonic())
        self.last_health: dict = {}
        self._fallen_prev = False

    def _seed_fall(self) -> None:
        for o in self.dyn.values():
            self.fall.seed(o.id, o.aabb()[0][2])

    # -- collision ------------------------------------------------------------------------------
    def _blocked(self, x: float, y: float) -> bool:
        r = self.robot_radius
        RES, ORIGIN = self.res, self.origin
        ix0 = int(math.floor((x - r - ORIGIN[0]) / RES))
        ix1 = int(math.floor((x + r - ORIGIN[0]) / RES))
        iy0 = int(math.floor((y - r - ORIGIN[1]) / RES))
        iy1 = int(math.floor((y + r - ORIGIN[1]) / RES))
        H, W_ = self.phys_occ.shape
        if ix0 < 0 or iy0 < 0 or ix1 >= W_ or iy1 >= H:
            return True
        sub = self.phys_occ[iy0:iy1 + 1, ix0:ix1 + 1]
        if not sub.any():
            return False
        yy, xx = np.nonzero(sub)
        cx = ORIGIN[0] + (ix0 + xx + 0.5) * RES
        cy = ORIGIN[1] + (iy0 + yy + 0.5) * RES
        return bool(np.any((cx - x) ** 2 + (cy - y) ** 2 <= (r + RES / 2) ** 2))

    # -- threads --------------------------------------------------------------------------------
    def start(self) -> "FakeP1":
        self._running = True
        for fn, name in ((self._physics, "fp1-physics"), (self._rep, "fp1-rep"), (self._link, "fp1-link"),
                         (self._camera, "fp1-camera")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)

    def _physics(self) -> None:
        pub = self.ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(ep(self.P["p1_pose"]))
        dt = 1.0 / self.physics_hz
        every = max(1, int(round(self.physics_hz / self.pose_hz)))
        every_obj = max(1, int(round(self.physics_hz / self.objects_hz))) if self.objects_hz > 0 else 0
        k = 0
        nxt = time.monotonic()
        t_wall0 = time.monotonic()
        next_health = time.monotonic() + (1.0 / self.health_hz if self.health_hz > 0 else 1e9)

        def send(topic: str, msg: dict) -> None:
            pub.send_multipart([topic.encode(), msgpack.packb(msg, use_bin_type=True)])

        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                self._step(dt)
                k += 1
                if k % every == 0 and not self.pose_paused:
                    msg = self._pose_msg(time.monotonic() - t_wall0)
                    send("gt.pose", msg)
                    self.pose_sent += 1
                    if msg["fallen"] != self._fallen_prev:
                        self._fallen_prev = msg["fallen"]
                        self._event("fallen" if msg["fallen"] else "recovered", pelvis_z=msg["pelvis_z"])
                        if msg["fallen"]:
                            self._event("robot_fell", pelvis_z=msg["pelvis_z"], tilt_deg=90.0,
                                        base_pos=msg["base_pos"])
                        else:
                            self._event("robot_recovered", pelvis_z=msg["pelvis_z"])
                if every_obj and k % every_obj == 0 and self.dyn:
                    send("gt.objects", {"seq": k // every_obj, "t_sim": round(self.t_sim, 4), "t_wall": time.time(),
                                        "objects": [o.record() for o in self.dyn.values()]})
                    for o in self.dyn.values():
                        ev = self.fall.update(o.id, self.t_sim, o.aabb()[0][2], float(np.linalg.norm(o.v)),
                                              o.held is not None, pos=W.box_center(o.aabb()))
                        if ev:
                            self._event("object_fell", **ev)
                if time.monotonic() >= next_health:
                    next_health = time.monotonic() + 1.0 / self.health_hz
                    self.last_health = self._health(time.monotonic() - t_wall0)
                    send("sim.health", self.last_health)
                self._expire_consumers()
            while True:
                try:
                    ev = self._evq.get_nowait()
                except queue.Empty:
                    break
                send("gt.event", ev)
            self.step_ms = (time.perf_counter() - t0) * 1e3
            nxt += dt
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()
        pub.close(0)

    def _event(self, event: str, /, **kw) -> None:
        ev = {"t_sim": round(self.t_sim, 4), "t_wall": time.time(), "event": event, **kw}
        self.events.append(ev)
        self._evq.put(ev)

    def _step(self, dt: float) -> None:
        self.t_sim += dt
        self._step_objects(dt)
        if self.collapsed:
            self.vx = self.vy = self.wz = 0.0
            self.pelvis_z = max(0.3, self.pelvis_z - 2.0 * dt)
            return
        tvx, tvy, twz = self.twist
        if time.monotonic() - self.twist_mono > 0.2:
            tvx = tvy = twz = 0.0
        if self.band_on:
            tvx = tvy = twz = 0.0
        self.vx, self.vy, self.wz = tvx, tvy, twz
        nx, ny = self.x + self.vx * dt, self.y + self.vy * dt
        if self._blocked(nx, ny):
            self.collisions += 1
            if not self._blocked(nx, self.y):
                ny = self.y
                self.vy = 0.0
            elif not self._blocked(self.x, ny):
                nx = self.x
                self.vx = 0.0
            else:
                nx, ny = self.x, self.y
                self.vx = self.vy = 0.0
        self.x, self.y = nx, ny
        self.yaw = wrap(self.yaw + self.wz * dt)
        moving = math.hypot(self.vx, self.vy) > 0.05 or abs(self.wz) > 0.2
        if moving:
            self.phase += 2 * math.pi * 1.8 * dt
            s = math.sin(self.phase)
            self.contact = (s > -0.35, s < 0.35)
        else:
            self.contact = (True, True)

    # -- kinematic objects ----------------------------------------------------------------------
    def _support_top(self, o: FakeObject, bottom: float) -> float:
        """Highest static top under the object's centre that is not above its bottom (floor = 0)."""
        c = W.box_center(o.aabb())
        top = 0.0
        for s in self.static_objs.values():
            lo, hi = s["aabb"]
            if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and hi[2] <= bottom + 0.02:
                top = max(top, float(hi[2]))
        return top

    def _step_objects(self, dt: float) -> None:
        for o in self.dyn.values():
            if o.held:
                pp, pq = self._palm(o.held)
                o.p, o.q = W.compose(pp, pq, o.off_p, o.off_q)
                o.v[:] = 0.0
                continue
            if not o.v.any():
                continue
            bottom = o.aabb()[0][2]
            sp = math.hypot(o.v[0], o.v[1])
            if sp > 0:
                dec = min(sp, FRICTION_DECEL * dt)
                o.v[0:2] *= (sp - dec) / sp
            o.p[0:2] += o.v[0:2] * dt
            top = self._support_top(o, bottom)
            if bottom > top + 1e-4 or o.v[2] != 0.0:
                o.v[2] -= G * dt
                nb = bottom + o.v[2] * dt
                if nb <= top:
                    o.p[2] += top - bottom
                    o.v[:] = 0.0
                else:
                    o.p[2] += o.v[2] * dt
            if np.linalg.norm(o.v) < 1e-3:
                o.v[:] = 0.0

    # -- fake robot links -----------------------------------------------------------------------
    def _torso(self) -> tuple[np.ndarray, np.ndarray]:
        q = quat_from_yaw(self.yaw)
        p = np.array([self.x, self.y, self.pelvis_z]) + W.quat_rotate(q, np.array(TORSO_FROM_PELVIS))
        return p, q

    def _palm(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        q = quat_mul(quat_from_yaw(self.yaw), W.quat_about_y(FAKE_PALM_PITCH))
        p = np.array([self.x, self.y, self.pelvis_z]) + W.quat_rotate(quat_from_yaw(self.yaw),
                                                                         np.array(FAKE_PALM_B[arm]))
        return p, q

    def _links(self) -> dict:
        tp, tq = self._torso()
        out = {"torso_link": {"pos": tp.tolist(), "quat_wxyz": tq.tolist()}}
        for arm in W.ARMS:
            p, q = self._palm(arm)
            out[f"{arm}_palm"] = {"pos": p.tolist(), "quat_wxyz": q.tolist()}
        return out

    def _pose_msg(self, t_wall_rel: float) -> dict:
        rtf = self.rtf_override if self.rtf_override is not None else self.t_sim / max(1e-6, t_wall_rel)
        return {"t_sim": self.t_sim, "t_wall": time.time(), "rtf": rtf,
                "base_pos": [self.x, self.y, self.pelvis_z], "base_quat_wxyz": quat_wxyz_from_yaw(self.yaw),
                "base_lin_vel_w": [self.vx, self.vy, 0.0], "base_ang_vel_w": [0.0, 0.0, self.wz],
                "yaw": self.yaw, "pelvis_z": self.pelvis_z, "fallen": self.collapsed or self.pelvis_z < 0.45,
                "foot_contact": {"left": self.contact[0], "right": self.contact[1]}, "band": self.band_on,
                "lowcmd_age_s": None if not self.twist_mono else time.monotonic() - self.twist_mono,
                "links": self._links(), "waist_q": [0.0, 0.0, 0.0]}

    def _health(self, t_wall_rel: float) -> dict:
        rtf = self.rtf_override if self.rtf_override is not None else min(1.0, self.t_sim / max(1e-6, t_wall_rel))
        level, why = W.health_level(rtf, rtf)
        return {"seq": int(self.t_sim), "t_sim": round(self.t_sim, 3), "t_wall": time.time(), "rtf_1s": rtf,
                "rtf_3s": rtf, "rtf_5s": rtf, "rtf_10s": rtf, "level": level, "level_reason": why,
                "physics_hz_1s": self.physics_hz, "render_hz": self.cam_hz, "step_ms_p99": self.step_ms,
                "overruns": 0, "lost_s": 0.0, "hitches_gt25ms": 0, "heartbeat_pubs": 0, "band": self.band_on,
                "fallen": self.collapsed, "held": self._held_map(),
                "cameras": {n: {"on": c["on"], "hz": c["hz"], "pub_hz": c["hz"] if c["on"] else 0.0}
                            for n, c in self.cams.items()}, "fake": True}

    def _link(self) -> None:
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SUBSCRIBE, b"twist")
        s.bind(ep(self.P["fake_link"]))
        while self._running:
            if not s.poll(100):
                continue
            _, payload = s.recv_multipart()
            d = msgpack.unpackb(payload, raw=False)
            with self._lock:
                self.link_count += 1
                self.twist = (float(d["vx"]), float(d["vy"]), float(d["wz"]))
                self.twist_mono = time.monotonic()
                if d.get("damping") and not self.band_on and not self.collapsed:
                    self.collapsed = True
                    self.log("[fake_p1] deploy damping with band off -> robot collapses")
        s.close(0)

    # -- cameras --------------------------------------------------------------------------------
    def _image(self, name: str, x: float, y: float, yaw: float) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:240] = (180, 200, 230) if name == "head" else (200, 180, 150)
        img[240:] = (90, 80, 70)
        off = int((yaw % (2 * math.pi)) / (2 * math.pi) * 640)
        for k in range(8):
            cx = (off + k * 80) % 640
            img[120:360, cx:cx + 6] = ((k * 30) % 255, 60, 200 - k * 20)
        _put_text(img, f"FAKE P1 {name} x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}")
        return img

    def _frame(self, name: str, render_seq: int) -> bytes:
        c = self.cams[name]
        spec = c["spec"]
        with self._lock:
            x, y, yaw = self.x, self.y, self.yaw
            tp, tq = self._torso()
            v, wz, t_sim = (self.vx, self.vy, 0.0), (0.0, 0.0, self.wz), self.t_sim
        pos, quat = spec.world_pose(tp, tq)
        c["seq"] += 1
        meta = W.frame_meta(spec, seq=c["seq"], render_seq=render_seq, t_sim=t_sim, t_capture=time.time(),
                            t_capture_mono=time.monotonic(), cam_pos=pos, cam_quat=quat, base_lin_vel_w=v,
                            base_ang_vel_w=wz)
        b64 = _jpeg_b64(self._image(name, x, y, yaw))
        return msgpack.packb(W.gear_sonic_message(spec.key, b64, meta, time.time()), use_bin_type=True)

    def _camera(self) -> None:
        socks = {}
        for name, port in (("head", self.P["camera"]), ("ego_view", self.P["ego"])):
            s = self.ctx.socket(zmq.PUB)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.SNDHWM, 20)
            s.bind(ep(port))
            socks[name] = s
        nxt = {n: 0.0 for n in socks}
        while self._running:
            now = time.monotonic()
            rendered = False
            for name, s in socks.items():
                c = self.cams[name]
                if not c["on"] or now < nxt[name]:
                    continue
                if not rendered:
                    self.render_seq += 1
                    rendered = True
                s.send(self._frame(name, self.render_seq))
                nxt[name] = now + 1.0 / max(0.1, c["hz"])
                if name == "head":
                    self.cam_sent += 1
            time.sleep(0.005)
        for s in socks.values():
            s.close(0)

    def _expire_consumers(self) -> None:
        now = time.monotonic()
        for name, c in self.cams.items():
            if c["consumers"].expire(now):
                self._apply_cam(name, "ttl")

    def _apply_cam(self, name: str, reason: str) -> None:
        c = self.cams[name]
        want = bool(c["consumers"])
        if want != c["on"]:
            c["on"] = want
            self._event("camera", name=name, on=want, hz=c["hz"], consumers=c["consumers"].names(), reason=reason)

    def _cam_reply(self, name: str) -> dict:
        c = self.cams[name]
        return {"name": name, "on": c["on"], "hz": c["hz"], "consumers": c["consumers"].names(),
                "port": self.P["camera"] if name == "head" else self.P["ego"], "key": name, "warmup_frames": 0}

    # -- REP ------------------------------------------------------------------------------------
    def _rep(self) -> None:
        s = self.ctx.socket(zmq.REP)
        s.setsockopt(zmq.LINGER, 0)
        s.bind(ep(self.P["p1_rep"]))
        while self._running:
            if not s.poll(100):
                continue
            raw = s.recv()
            try:
                req = loads_any(raw)
                rep = self._op(req.get("op"), {**req, **(req.get("args") or {})})
            except W.OpError as e:
                rep = e.reply()
            except Exception as e:
                rep = {"ok": False, "error": f"internal: {e!r}", "code": "internal"}
            s.send(dumps_json(rep))
        s.close(0)

    M1_OPS = ("ping", "get_pose", "get_scene_info", "get_occupancy", "band", "reset_robot", "get_stats",
              "render_topdown", "record", "shutdown", "spawn_obstacle")
    M2B_OPS = ("get_objects", "attach", "detach", "release_all", "get_cameras", "camera", "set_render_rates",
               "get_link_poses", "detections", "reset_scene", "move_object", "set_object_pose", "push_object",
               "get_health")

    def _op(self, op: str, a: dict) -> dict:
        with self._lock:
            if op == "ping":
                return {"ok": True, "t_wall": time.time(), "fake": True, "ops": sorted(self.M1_OPS + self.M2B_OPS),
                        "p1_contract": W.CONTRACT, "cameras": {n: c["on"] for n, c in self.cams.items()},
                        "topics": W.TOPICS, "house_id": self.info.get("house_id"), "band": self.band_on}
            if op == "get_pose":
                return {"ok": True, **self._pose_msg(self.t_sim)}
            if op == "get_scene_info":
                return {"ok": True, **self.info}
            if op == "get_occupancy":
                return {"ok": True, "path": self.occ_path, "resolution": self.res, "origin": list(self.origin),
                        "shape": list(self.occ.shape), "robot_radius": 0.25, "source": "fake"}
            if op == "band":
                on = bool(a.get("on"))
                self.band_on = on
                self.log(f"[fake_p1] band {'ON' if on else 'OFF'} ramp_s={a.get('ramp_s', 0)}")
                self._event("band", on=on, ramp_s=float(a.get("ramp_s", 0.0)))
                return {"ok": True, "band": on}
            if op == "shutdown":
                self.shutdown_requested = True
                return {"ok": True}
            if op == "record":
                return {"ok": True, "path": a.get("path"), "samples": 0, "fake": True}
            if op == "reset_robot":
                self._reset_robot(a)
                return {"ok": True, "root_writes": self.root_writes}
            if op == "get_stats":
                return {"ok": True, "rtf_1s": 1.0, "rtf_10s": 1.0, "physics_hz_1s": self.physics_hz,
                        "render_hz": self.cam_hz, "camera_pub_hz": self.cam_hz, "lowstate_pub_hz": self.physics_hz,
                        "step_ms": {"mean": self.step_ms}, "lowcmd_fresh_hz": None, "twist_msgs": self.link_count,
                        "collisions": self.collisions, "root_writes": self.root_writes, "fake": True,
                        "cameras": {n: {"on": c["on"], "hz": c["hz"], "frames": c["seq"]} for n, c in self.cams.items()},
                        "held": self._held_map(), "attach_count": self.attach_count,
                        "detach_count": self.detach_count, "object_writes": self.object_writes}
            if op == "render_topdown":
                mode = str(a.get("mode") or "full")
                if mode not in ("full", "furniture"):
                    raise W.OpError("bad_arg", f"mode must be full|furniture, got {mode!r}")
                path = a.get("path") or os.path.join(self.out_dir, "topdown.png" if mode == "full"
                                                     else "topdown_furniture.png")
                hidden = self._render(path, props=(mode == "full"))
                H, W_ = self.occ.shape
                return {"ok": True, "path": path, "mode": mode, "hidden": hidden,
                        "extent": [self.origin[0], self.origin[1], self.origin[0] + W_ * self.res,
                                   self.origin[1] + H * self.res]}  # [xmin,ymin,xmax,ymax]
            if op == "spawn_obstacle":
                x, y, r = float(a["x"]), float(a["y"]), float(a.get("r", 0.2))
                H, W_ = self.phys_occ.shape
                yy, xx = np.mgrid[0:H, 0:W_]
                cx = self.origin[0] + (xx + 0.5) * self.res
                cy = self.origin[1] + (yy + 0.5) * self.res
                self.phys_occ[(cx - x) ** 2 + (cy - y) ** 2 <= r * r] = 1
                return {"ok": True}
            if op in self.M2B_OPS:
                return {"ok": True, **getattr(self, f"_op_{op}")(a)}
            return {"ok": False, "error": f"unknown op {op}", "code": "unknown_op"}

    def _reset_robot(self, a: dict) -> None:
        self.x, self.y = float(a.get("x", self.spawn[0])), float(a.get("y", self.spawn[1]))
        self.yaw = float(a.get("yaw", self.spawn[2]))
        self.collapsed = False
        self.pelvis_z = 0.78
        self.root_writes += 1
        if "band" in a:          # the real P1 engages the band by default; the fake keeps M1's behaviour (unchanged)
            self.band_on = bool(a["band"])
        self.log(f"[fake_p1] reset_robot -> {self.x:.2f},{self.y:.2f},{self.yaw:.2f} (logged)")
        self._event("reset_robot", x=self.x, y=self.y, yaw=self.yaw, band=self.band_on, root_writes=self.root_writes)

    # -- M2b ops ----------------------------------------------------------------------------------
    def _held_map(self) -> dict:
        out = {"left": None, "right": None}
        for o in self.dyn.values():
            if o.held:
                out[o.held] = o.id
        return out

    def _dyn(self, oid) -> FakeObject:
        oid = str(oid)
        if oid in self.dyn:
            return self.dyn[oid]
        if oid in self.static_objs:
            raise W.OpError("not_movable", f"{oid} is static or articulated")
        raise W.OpError("unknown_object", repr(oid))

    def _static_record(self, o: dict) -> dict:
        return W.object_record(str(o["id"]), str(o.get("name") or o["id"]), o["pos"], [1.0, 0.0, 0.0, 0.0],
                               o["aabb"], held_by=None, dynamic=False)

    def _op_get_objects(self, a: dict) -> dict:
        ids = [str(i) for i in a["ids"]] if a.get("ids") else None
        recs = [o.record() for o in self.dyn.values()]
        if not a.get("dynamic_only"):
            recs += [self._static_record(o) for o in self.static_objs.values()]
        if ids is not None:
            unknown = [i for i in ids if i not in self.dyn and i not in self.static_objs]
            if unknown:
                raise W.OpError("unknown_object", f"{unknown[:5]}")
            recs = [r for r in recs if r["id"] in ids]
        return {"t_sim": round(self.t_sim, 4), "t_wall": time.time(), "seq": int(self.t_sim * self.physics_hz),
                "pose_source": "sim", "objects": recs}

    def _op_attach(self, a: dict) -> dict:
        for k in ("id", "arm"):
            if a.get(k) is None:
                raise W.OpError("bad_arg", f"missing {k}")
        o = self._dyn(a["id"])
        arm, mode = str(a["arm"]), str(a.get("mode") or "follow")
        if arm not in W.ARMS:
            raise W.OpError("bad_arg", f"arm must be one of {W.ARMS}")
        if mode not in W.ATTACH_MODES:
            raise W.OpError("bad_arg", f"mode must be one of {W.ATTACH_MODES}")
        if o.held and o.held != arm:
            raise W.OpError("held_by_other", f"{o.id} is held by the {o.held} hand")
        other = self._held_map().get(arm)
        if other and other != o.id:
            raise W.OpError("hand_busy", f"the {arm} hand holds {other}")
        off = np.asarray(a.get("offset") or W.GRIP_OFFSET, dtype=np.float64)
        pp, pq = self._palm(arm)
        grip, _ = W.compose(pp, pq, off, [1.0, 0.0, 0.0, 0.0])
        centre = W.box_center(o.aabb())
        dist = float(np.linalg.norm(centre - grip))
        snapped = bool(a.get("snap", True)) and dist > float(a.get("snap_m", W.SNAP_M))
        if snapped:
            o.p = o.p + (grip - centre)
        o.off_p, o.off_q = W.relative(pp, pq, o.p, o.q)
        o.held, o.mode = arm, mode
        o.v[:] = 0.0
        self.attach_count += 1
        self.fall.reset(o.id)
        self._event("attach", id=o.id, arm=arm, mode=mode, snapped=snapped)
        return {"id": o.id, "arm": arm, "mode": mode, "held_by": arm, "snapped": snapped, "dist_m": round(dist, 4),
                "grip_point": [round(float(v), 4) for v in grip], "stepping_stone": True}

    def _op_detach(self, a: dict) -> dict:
        if a.get("id") is None:
            raise W.OpError("bad_arg", "missing id")
        o = self._dyn(a["id"])
        was = o.held is not None
        arm = o.held
        o.held = o.mode = None
        if was:
            self.detach_count += 1
        placed = a.get("pose") is not None
        if placed:
            centre, yaw = W.parse_pose_arg(a["pose"])
            o.p, o.q = W.placed_pose(o.aabb0, o.p0, o.q0, centre, yaw)
        o.v[:] = 0.0
        o.v[2] = -1e-6          # let it settle onto a support (or fall) under the fake gravity
        self.fall.reset(o.id)
        self._event("detach", id=o.id, arm=arm, placed=placed)
        rec = o.record()
        return {"id": o.id, "was_held": was, "placed": placed, "held_by": None, "pos": rec["pos"],
                "aabb": rec["aabb"]}

    def _op_release_all(self, a: dict) -> dict:
        out = []
        for o in self.dyn.values():
            if o.held:
                self._op_detach({"id": o.id})
                out.append(o.id)
        return {"released": out}

    def _op_get_cameras(self, a: dict) -> dict:
        out = []
        for name, c in self.cams.items():
            d = c["spec"].info()
            d.update(self._cam_reply(name))
            d["frames"] = c["seq"]
            out.append(d)
        return {"cameras": out, "stream_camera": "head", "render_hz": self.cam_hz}

    def _op_camera(self, a: dict) -> dict:
        name = str(a.get("name"))
        if name not in self.cams:
            raise W.OpError("unknown_camera", f"{name!r} (have {sorted(self.cams)})")
        c = self.cams[name]
        if a.get("hz") is not None:
            hz = float(a["hz"])
            if not hz > 0:
                raise W.OpError("bad_arg", "hz must be > 0 (use on:false to stop a camera)")
            c["hz"] = hz
        on = a.get("on")
        if on is True:
            c["consumers"].add(str(a.get("consumer") or "anon"), time.monotonic(), a.get("ttl_s"))
        elif on is False:
            cons = a.get("consumer")
            c["consumers"].remove(None if cons is None else str(cons))
        self._apply_cam(name, "op")
        return self._cam_reply(name)

    def _op_set_render_rates(self, a: dict) -> dict:
        out = {}
        for name, key in (("head", "head_hz"), ("ego_view", "ego_hz")):
            hz = a.get(key)
            if hz is None:
                continue
            if float(hz) > 0:
                self._op_camera({"name": name, "on": True, "hz": float(hz),
                                 "consumer": "default" if name == "head" else "set_render_rates"})
            else:
                self._op_camera({"name": name, "on": False})
            out[name] = self._cam_reply(name)
        return out

    def _op_get_link_poses(self, a: dict) -> dict:
        links = self._links()
        for name in self.cams:
            p, q = self.cams[name]["spec"].world_pose(*self._torso())
            links[f"cam:{name}"] = {"pos": p.tolist(), "quat_wxyz": q.tolist()}
        want = a.get("links") or ["torso_link", "left_palm", "right_palm"]
        return {"t_sim": round(self.t_sim, 4), "links": {k: links[k] for k in want if k in links},
                "unknown": [k for k in want if k not in links], "available": sorted(links)}

    def _op_detections(self, a: dict) -> dict:
        """Frustum count on the fake camera (no occlusion): the shape of the real reply, method 'fake-frustum'."""
        name = str(a.get("camera") or "head")
        if name not in self.cams:
            raise W.OpError("unknown_camera", repr(name))
        c = self.cams[name]
        if not c["on"]:
            raise W.OpError("camera_off", f"{name} is off; enable it with the `camera` op first")
        spec = c["spec"]
        cp, cq = spec.world_pose(*self._torso())
        min_px = int(a.get("min_px", 40))
        max_range = a.get("max_range")
        ids = set(str(i) for i in a["ids"]) if a.get("ids") else None
        dets = []
        boxes = [(o.id, o.name, o.aabb(), o.held) for o in self.dyn.values()] + \
            [(str(s["id"]), str(s.get("name") or s["id"]), s["aabb"], None) for s in self.static_objs.values()]
        for oid, name_, aabb, held in boxes:
            if ids is not None and oid not in ids:
                continue
            corners = W.box_corners(aabb)
            loc = np.array([W.quat_rotate_inverse(cq, v - cp) for v in corners])
            if (loc[:, 0] <= spec.clipping[0]).any():
                continue
            u = spec.width / 2 - spec.fx_px * loc[:, 1] / loc[:, 0]
            v = spec.height / 2 - spec.fy_px * loc[:, 2] / loc[:, 0]
            u0, u1 = float(np.clip(u.min(), 0, spec.width - 1)), float(np.clip(u.max(), 0, spec.width - 1))
            v0, v1 = float(np.clip(v.min(), 0, spec.height - 1)), float(np.clip(v.max(), 0, spec.height - 1))
            px = int(max(0.0, u1 - u0) * max(0.0, v1 - v0))
            dist = float(np.linalg.norm(W.box_center(aabb) - cp))
            if px < min_px or (max_range is not None and dist > float(max_range)):
                continue
            dets.append({"id": oid, "name": name_, "px": px, "bbox": [int(u0), int(v0), int(u1), int(v1)],
                         "dist_m": round(dist, 3), "held_by": held})
        dets.sort(key=lambda d: -d["px"])
        self.render_seq += 1
        return {"camera": name, "t_sim": round(self.t_sim, 4), "render_seq": self.render_seq, "frame_seq": c["seq"],
                "w": spec.width, "h": spec.height, "method": "fake-frustum", "min_px": min_px, "detections": dets,
                "other_px": {"robot": 0, "structure": 0, "background": 0},
                "cam_pose_wl": W.cam_pose_wl(cp, cq), "cam_pos": W.r(cp, 4), "ms": 0.0}

    def _op_reset_scene(self, a: dict) -> dict:
        t0 = time.perf_counter()
        variant = str(a.get("variant") or "default")
        if variant != "default":
            raise W.OpError("unknown_variant", f"{variant!r}: only 'default'; pass placements in `poses`")
        released = self._op_release_all({})["released"]
        for o in self.dyn.values():
            o.p, o.q, o.v = o.p0.copy(), o.q0.copy(), np.zeros(3)
        applied = []
        for oid, pose in (a.get("poses") or {}).items():
            self._op_move_object({"id": oid, "pose": pose, "_by": "reset_scene"})
            applied.append(str(oid))
        self.fall.reset()
        self._seed_fall()
        robot = a.get("robot")
        if robot:
            rr = {"x": self.spawn[0], "y": self.spawn[1], "yaw": self.spawn[2]} if robot is True else dict(robot)
            rr["band"] = bool(a.get("band", True))
            self._reset_robot(rr)
        ms = round((time.perf_counter() - t0) * 1e3, 2)
        self._event("reset_scene", variant=variant, objects_reset=len(self.dyn), robot_reset=bool(robot), ms=ms)
        return {"variant": variant, "objects_reset": len(self.dyn), "poses_applied": applied, "released": released,
                "robot_reset": bool(robot), "ms": ms, "object_writes": self.object_writes,
                "root_writes": self.root_writes}

    def _op_move_object(self, a: dict) -> dict:
        if a.get("id") is None or a.get("pose") is None:
            raise W.OpError("bad_arg", "missing id or pose")
        o = self._dyn(a["id"])
        if o.held:
            self._op_detach({"id": o.id})
        centre, yaw = W.parse_pose_arg(a["pose"])
        o.p, o.q = W.placed_pose(o.aabb0, o.p0, o.q0, centre, yaw)
        o.v = np.asarray(a["vel"], dtype=np.float64) if a.get("vel") is not None else np.zeros(3)
        if not o.v.any():
            o.v[2] = -1e-6      # settle / fall onto whatever is below
        self.object_writes += 1
        self.fall.reset(o.id)
        self._event("object_moved", id=o.id, by=a.get("_by") or a.get("op") or "move_object")
        rec = o.record()
        return {"id": o.id, "pos": rec["pos"], "aabb": rec["aabb"]}

    def _op_set_object_pose(self, a: dict) -> dict:
        return self._op_move_object(a)

    def _op_push_object(self, a: dict) -> dict:
        if a.get("id") is None or a.get("vel") is None:
            raise W.OpError("bad_arg", "missing id or vel")
        o = self._dyn(a["id"])
        if o.held:
            raise W.OpError("bad_arg", f"{o.id} is held; detach it first")
        v = [float(x) for x in a["vel"]]
        if len(v) != 3:
            raise W.OpError("bad_arg", "vel needs [vx, vy, vz]")
        o.v = np.array(v)
        self.object_writes += 1
        self._event("object_moved", id=o.id, by="push_object")
        return {"id": o.id, "vel": v}

    def _op_get_health(self, a: dict) -> dict:
        return dict(self.last_health) if self.last_health else self._health(max(1e-6, self.t_sim))

    def _render(self, path: str, props: bool = True) -> int:
        """Occupancy picture; `props` draws the dynamic objects (full) or hides them (furniture)."""
        img = np.where(self.occ[::-1] > 0, 40, 235).astype(np.uint8)
        img = np.repeat(img[..., None], 3, axis=2)
        H = img.shape[0]
        if props:
            for o in self.dyn.values():
                c = W.box_center(o.aabb())
                ci = int((c[0] - self.origin[0]) / self.res)
                ri = H - 1 - int((c[1] - self.origin[1]) / self.res)
                img[max(0, ri - 2):ri + 3, max(0, ci - 2):ci + 3] = (0, 0, 255)
        try:
            import cv2
            cv2.imwrite(path, img)
        except ImportError:
            from PIL import Image
            Image.fromarray(img[..., ::-1]).save(path)
        return 0 if props else len(self.dyn) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=200)
    ap.add_argument("--out", default="/tmp/fake_p1")
    ap.add_argument("--house-dir", default=None, help="real house: dir with occupancy.npz + house_info.json")
    args = ap.parse_args(argv)
    p1 = FakeP1(args.port_offset, args.out, house_dir=args.house_dir).start()
    print(f"[fake_p1] up on offset {args.port_offset}: {p1.P}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.wait(1.0) and not p1.shutdown_requested:
        if int(time.monotonic()) % 5:
            continue
        print(f"[fake_p1] pose={p1.x:.2f},{p1.y:.2f},{math.degrees(p1.yaw):.0f}deg band={p1.band_on} "
              f"fallen={p1.collapsed} twist_msgs={p1.link_count} collisions={p1.collisions} "
              f"held={p1._held_map()}", flush=True)
    p1.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
