"""Fake P1 (wl-isaac) for testing wl-body without Isaac.

Implements the P1 side of docs/contracts/m1.md with a kinematic robot in a synthetic 3-room house:
  REP  p1_rep  : ping, get_pose, get_scene_info, get_occupancy, band, reset_robot, get_stats, render_topdown,
                 plus test-only spawn_obstacle {x,y,r} (blocks the fake physics but NOT the published map)
  PUB  p1_pose : multipart [b"gt.pose", msgpack] at 50 Hz
  PUB  camera  : gear_sonic sensor_server msgpack {"timestamps", "images": {"ego_view": b64 JPEG}} at 10 Hz
  SUB  fake_link: [b"twist", msgpack {vx, vy, wz, damping}] from tools/fake_deploy.py (stands in for rt/lowcmd)

Physics: 200 Hz integration of the commanded world twist, disk-vs-occupancy collision (robot radius 0.15 m) with
axis sliding; band on => robot held; damping (deploy command{stop}) with band off => the robot "collapses"
(fallen=True, pelvis_z=0.3) exactly like the real deploy's CreateDampingCommand would cause.

Run:  python -m tools.fake_p1 --port-offset 200 --out /tmp/fake_p1
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import threading
import time

import msgpack
import numpy as np
import zmq

from body.config import ep, ports as _ports
from body.wire import dumps_json, encode_camera_message, loads_any, quat_wxyz_from_yaw, wrap

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
# small objects on furniture (for get_scene_info only)
OBJECTS = [
    ("Apple|1", "Apple", "room_kitchen", (0.3, 1.0, 0.95)),
    ("Mug|1", "Mug", "room_dining", (2.0, 4.7, 0.8)),
    ("RemoteControl|1", "RemoteControl", "room_living", (7.1, 2.9, 0.5)),
]
SPAWN = (1.8, 1.4, 1.1)  # kitchen, facing roughly +y (non-zero yaw catches planner-frame bugs)


def build_occupancy() -> np.ndarray:
    W = int(round(SIZE[0] / RES))
    H = int(round(SIZE[1] / RES))
    occ = np.zeros((H, W), dtype=np.uint8)

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
    xs = ORIGIN[0] + (np.arange(W) + 0.5) * RES
    outside = (xs[None, :] < -t) | (xs[None, :] > 10 + t) | (ys[:, None] < -t) | (ys[:, None] > 6 + t)
    occ[outside] = 1
    return occ


def scene_info() -> dict:
    objs = []
    for oid, cat, room, (x0, y0, x1, y1), h in FURNITURE:
        objs.append({"id": oid, "category": cat, "room_id": room, "pos": [(x0 + x1) / 2, (y0 + y1) / 2, h / 2],
                     "aabb": [[x0, y0, 0.0], [x1, y1, h]]})
    for oid, cat, room, (x, y, z) in OBJECTS:
        objs.append({"id": oid, "category": cat, "room_id": room, "pos": [x, y, z],
                     "aabb": [[x - 0.05, y - 0.05, z - 0.05], [x + 0.05, y + 0.05, z + 0.05]]})
    return {"house_id": "fake-3room", "rooms": ROOMS, "objects": objs,
            "spawn": {"x": SPAWN[0], "y": SPAWN[1], "yaw": SPAWN[2]}, "frame": "world, z-up, meters"}


class FakeP1:
    def __init__(self, port_offset: int = 200, out_dir: str = "/tmp/fake_p1", pose_hz: float = 50.0,
                 cam_hz: float = 10.0, physics_hz: float = 200.0, ctx: zmq.Context | None = None,
                 robot_radius: float = 0.15, log=print, house_dir: str | None = None):
        self.P = _ports(port_offset)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ctx = ctx or zmq.Context.instance()
        self.pose_hz, self.cam_hz, self.physics_hz = pose_hz, cam_hz, physics_hz
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
        self._lock = threading.Lock()
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

    # -- collision ------------------------------------------------------------------------------
    def _blocked(self, x: float, y: float) -> bool:
        r = self.robot_radius
        RES, ORIGIN = self.res, self.origin
        ix0 = int(math.floor((x - r - ORIGIN[0]) / RES))
        ix1 = int(math.floor((x + r - ORIGIN[0]) / RES))
        iy0 = int(math.floor((y - r - ORIGIN[1]) / RES))
        iy1 = int(math.floor((y + r - ORIGIN[1]) / RES))
        H, W = self.phys_occ.shape
        if ix0 < 0 or iy0 < 0 or ix1 >= W or iy1 >= H:
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
        k = 0
        nxt = time.monotonic()
        t_wall0 = time.monotonic()
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                self._step(dt)
                k += 1
                if k % every == 0 and not self.pose_paused:
                    msg = self._pose_msg(time.monotonic() - t_wall0)
                    pub.send_multipart([b"gt.pose", msgpack.packb(msg, use_bin_type=True)])
                    self.pose_sent += 1
            self.step_ms = (time.perf_counter() - t0) * 1e3
            nxt += dt
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()
        pub.close(0)

    def _step(self, dt: float) -> None:
        self.t_sim += dt
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

    def _pose_msg(self, t_wall_rel: float) -> dict:
        return {"t_sim": self.t_sim, "t_wall": time.time(), "rtf": self.t_sim / max(1e-6, t_wall_rel),
                "base_pos": [self.x, self.y, self.pelvis_z], "base_quat_wxyz": quat_wxyz_from_yaw(self.yaw),
                "base_lin_vel_w": [self.vx, self.vy, 0.0], "base_ang_vel_w": [0.0, 0.0, self.wz],
                "yaw": self.yaw, "pelvis_z": self.pelvis_z, "fallen": self.collapsed or self.pelvis_z < 0.45,
                "foot_contact": {"left": self.contact[0], "right": self.contact[1]}, "band": self.band_on,
                "lowcmd_age_s": None if not self.twist_mono else time.monotonic() - self.twist_mono}

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

    def _camera(self) -> None:
        import cv2

        pub = self.ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.setsockopt(zmq.SNDHWM, 20)
        pub.bind(ep(self.P["camera"]))
        period = 1.0 / self.cam_hz
        while self._running:
            with self._lock:
                x, y, yaw = self.x, self.y, self.yaw
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:240] = (180, 200, 230)
            img[240:] = (90, 80, 70)
            off = int((yaw % (2 * math.pi)) / (2 * math.pi) * 640)
            for k in range(8):
                cx = (off + k * 80) % 640
                img[120:360, cx:cx + 6] = ((k * 30) % 255, 60, 200 - k * 20)
            cv2.putText(img, f"FAKE P1 x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            pub.send(encode_camera_message({"ego_view": img}, time.time()))
            self.cam_sent += 1
            time.sleep(period)
        pub.close(0)

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
            except Exception as e:
                rep = {"ok": False, "error": repr(e)}
            s.send(dumps_json(rep))
        s.close(0)

    def _op(self, op: str, a: dict) -> dict:
        with self._lock:
            if op == "ping":
                return {"ok": True, "t_wall": time.time(), "fake": True}
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
                return {"ok": True, "band": on}
            if op == "shutdown":
                self.shutdown_requested = True
                return {"ok": True}
            if op == "record":
                return {"ok": True, "path": a.get("path"), "samples": 0, "fake": True}
            if op == "reset_robot":
                self.x, self.y = float(a.get("x", self.spawn[0])), float(a.get("y", self.spawn[1]))
                self.yaw = float(a.get("yaw", self.spawn[2]))
                self.collapsed = False
                self.pelvis_z = 0.78
                self.root_writes += 1
                self.log(f"[fake_p1] reset_robot -> {self.x:.2f},{self.y:.2f},{self.yaw:.2f} (logged)")
                return {"ok": True, "root_writes": self.root_writes}
            if op == "get_stats":
                return {"ok": True, "rtf_1s": 1.0, "rtf_10s": 1.0, "physics_hz_1s": self.physics_hz,
                        "render_hz": self.cam_hz, "camera_pub_hz": self.cam_hz, "lowstate_pub_hz": self.physics_hz,
                        "step_ms": {"mean": self.step_ms}, "lowcmd_fresh_hz": None, "twist_msgs": self.link_count,
                        "collisions": self.collisions, "root_writes": self.root_writes, "fake": True}
            if op == "render_topdown":
                path = a.get("path") or os.path.join(self.out_dir, "topdown.png")
                self._render(path)
                H, W = self.occ.shape
                return {"ok": True, "path": path, "extent": [self.origin[0], self.origin[1],
                                                             self.origin[0] + W * self.res,
                                                             self.origin[1] + H * self.res]}  # [xmin,ymin,xmax,ymax]
            if op == "spawn_obstacle":
                x, y, r = float(a["x"]), float(a["y"]), float(a.get("r", 0.2))
                H, W = self.phys_occ.shape
                yy, xx = np.mgrid[0:H, 0:W]
                cx = self.origin[0] + (xx + 0.5) * self.res
                cy = self.origin[1] + (yy + 0.5) * self.res
                self.phys_occ[(cx - x) ** 2 + (cy - y) ** 2 <= r * r] = 1
                return {"ok": True}
            return {"ok": False, "error": f"unknown op {op}"}

    def _render(self, path: str) -> None:
        import cv2

        img = np.where(self.occ[::-1] > 0, 40, 235).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, img)


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
              f"fallen={p1.collapsed} twist_msgs={p1.link_count} collisions={p1.collisions}", flush=True)
    p1.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
