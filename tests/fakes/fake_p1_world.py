"""A minimal fake P1 (wl-isaac) for world/isaac_client.py tests: REP (JSON) + PUB on 5601, serving a recorded house.

M1 contract ops (docs/contracts/m1.md §1.6): ping, get_pose, get_scene_info, get_occupancy, render_topdown, band,
reset_robot, get_stats; PUB gt.pose.

With `m2b=True` it speaks the M2b wire of docs/contracts/p1_m2b.md (v1, "m2b-1"), kinematically:
    ping                ops, p1_contract, cameras {head, ego_view}, topics                              §2
    get_objects         {t_sim, t_wall, seq, pose_source, objects: [OBJ]} (ids, dynamic_only)          §3.1
    PUB gt.objects      dynamic + held objects at `objects_hz` (10)                                       §3.2
    attach / detach / release_all   STEPPING STONE; a held object's centre follows the palm grip point    §4
    camera, get_cameras, set_render_rates                                                                 §5.4
    gt.pose links       torso_link, left_palm, right_palm (+ waist_q); get_link_poses                     §6
    detections          instance-id pixel counts: whatever the test put in `seg_px` {scene_id: px}      §7
                        (method "fake-segmentation"; camera_off when the camera is off)
    reset_scene, move_object, set_object_pose, push_object                                                §8
    render_topdown      {mode}                                                                            §9
    PUB sim.health      at `health_hz` (1): rtf_1s/3s/5s/10s = `rtf`, level from PLAN §3.5 or `level`   §10.1
    gt.event            emit_event(...) plus attach/detach/camera/reset_scene/object_moved              §10.2
Optionally (`frames=True`) the head camera on 5565 in the gear_sonic format with the §5.2 metadata (a flat image).
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from pathlib import Path

import msgpack
import zmq

HOUSES = Path(__file__).resolve().parent / "houses"
TORSO_FROM_PELVIS = (-0.0039635, 0.0, 0.044)
PALM_IN_PELVIS = {"left": (0.32, 0.18, 0.20), "right": (0.32, -0.18, 0.20)}   # (fwd, left, up) from the pelvis
GRIP_OFFSET = 0.07


def _rot(yaw: float, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


class FakeP1World:
    def __init__(self, house: str = "procthor-train-38", port_offset: int = 400, m2b: bool = False,
                 objects_hz: float = 10.0, health_hz: float = 1.0, frames: bool = False, links: bool = True):
        self.dir = HOUSES / house
        self.info = json.loads((self.dir / "house_info.json").read_text())
        self.info.setdefault("bounds", [self.info["bounds_xy"][0][0], self.info["bounds_xy"][0][1],
                                        self.info["bounds_xy"][1][0], self.info["bounds_xy"][1][1]])
        self.off = port_offset
        self.m2b = m2b
        self.objects_hz = objects_hz
        self.health_hz = health_hz
        self.frames = frames
        self.links = links
        sp = self.info["spawn"]
        self.pose = [float(sp["x"]), float(sp["y"]), 0.78, float(sp["yaw"])]
        self.objects = {o["id"]: {"name": o.get("name", ""), "pos": list(o["pos"]),
                                  "aabb": [list(o["aabb"][0]), list(o["aabb"][1])], "held_by": None,
                                  "dynamic": bool(not o.get("is_static", True) and o.get("body_path")
                                                  and not o.get("articulated")),
                                  "vel": [0.0, 0.0, 0.0]} for o in self.info["objects"]}
        self._load = {k: json.loads(json.dumps(v)) for k, v in self.objects.items()}
        self.calls: list[str] = []
        self.requests: list[dict] = []
        self.band = True
        self.rtf = 1.0
        self.level: str | None = None          # None: from rtf (PLAN §3.5); else forced
        self.seg_px: dict[str, int] = {}       # detections: scene id -> pixels (what the "render" shows)
        self.cameras = {"head": {"on": True, "hz": 30.0, "consumers": {"default"}},
                        "ego_view": {"on": False, "hz": 30.0, "consumers": set()}}
        self.object_writes = 0
        self.torso_pitch = 0.0                 # rad, + = leaning forward (the camera looks further down)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.ctx = zmq.Context.instance()

    # ------------------------------------------------------------------
    def start(self) -> "FakeP1World":
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.rep.bind(f"tcp://127.0.0.1:{5600 + self.off}")
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://127.0.0.1:{5601 + self.off}")
        self._threads = [threading.Thread(target=self._serve, daemon=True),
                         threading.Thread(target=self._publish, daemon=True)]
        if self.frames:
            self.cam_pub = self.ctx.socket(zmq.PUB)
            self.cam_pub.setsockopt(zmq.LINGER, 0)
            self.cam_pub.bind(f"tcp://127.0.0.1:{5565 + self.off}")
            self._threads.append(threading.Thread(target=self._publish_frames, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self.rep.close(0)
        self.pub.close(0)
        if self.frames:
            self.cam_pub.close(0)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.pose = [x, y, 0.78, yaw]

    # ------------------------------------------------------------------ geometry
    def palm(self, arm: str) -> tuple[float, float, float]:
        x, y, z, yaw = self.pose
        d = _rot(yaw, PALM_IN_PELVIS[arm])
        return (x + d[0], y + d[1], z + d[2])

    def grip_point(self, arm: str) -> tuple[float, float, float]:
        x, y, z, yaw = self.pose
        f, l, u = PALM_IN_PELVIS[arm]
        d = _rot(yaw, (f + GRIP_OFFSET, l, u))
        return (x + d[0], y + d[1], z + d[2])

    def _center(self, o) -> list[float]:
        lo, hi = o["aabb"]
        return [(lo[i] + hi[i]) / 2 for i in range(3)]

    def _put(self, o, c) -> None:
        lo, hi = o["aabb"]
        h = [(hi[i] - lo[i]) / 2 for i in range(3)]
        o["aabb"] = [[c[i] - h[i] for i in range(3)], [c[i] + h[i] for i in range(3)]]
        o["pos"] = list(c)

    def _follow(self) -> None:
        for o in self.objects.values():
            if o["held_by"]:
                self._put(o, list(self.grip_point(o["held_by"])))

    def _obj(self, sid: str) -> dict:
        o = self.objects[sid]
        return {"id": sid, "name": o["name"], "pos": list(o["pos"]), "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "aabb": [list(o["aabb"][0]), list(o["aabb"][1])], "held_by": o["held_by"], "dynamic": o["dynamic"],
                "source": "sim" if o["dynamic"] else "static", "lin_vel": list(o["vel"]),
                "moving": math.hypot(*o["vel"]) > 0.02}

    def _links(self) -> dict:
        x, y, z, yaw = self.pose
        t = _rot(yaw, TORSO_FROM_PELVIS)
        q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        p = self.torso_pitch / 2                    # torso = yaw about z, then pitch about the body y (a waist lean)
        qt = [q[0] * math.cos(p), -q[3] * math.sin(p), q[0] * math.sin(p), q[3] * math.cos(p)]
        return {"torso_link": {"pos": [x + t[0], y + t[1], z + t[2]], "quat_wxyz": qt},
                "left_palm": {"pos": list(self.palm("left")), "quat_wxyz": q},
                "right_palm": {"pos": list(self.palm("right")), "quat_wxyz": q}}

    def health(self) -> dict:
        lvl = self.level or ("unsafe" if self.rtf < 0.85 else "degraded" if self.rtf < 0.95 else "ok")
        return {"seq": 0, "t_sim": 0.0, "t_wall": time.time(), "rtf_1s": self.rtf, "rtf_3s": self.rtf,
                "rtf_5s": self.rtf, "rtf_10s": self.rtf, "level": lvl, "band": self.band, "fallen": False,
                "held": {a: next((k for k, o in self.objects.items() if o["held_by"] == a), None)
                         for a in ("left", "right")},
                "cameras": {n: {"on": c["on"], "hz": c["hz"], "pub_hz": c["hz"] if c["on"] else 0.0}
                            for n, c in self.cameras.items()}}

    def emit_event(self, event: str, **fields) -> None:
        msg = {"event": event, "t_sim": 0.0, "t_wall": time.time(), **fields}
        with self._lock:
            self.pub.send_multipart([b"gt.event", msgpack.packb(msg, use_bin_type=True)])

    # ------------------------------------------------------------------
    def _publish(self) -> None:
        seq = 0
        t_obj = t_health = 0.0
        while not self._stop.is_set():
            x, y, z, yaw = self.pose
            msg = {"seq": seq, "t_sim": seq * 0.02, "t_wall": time.time(), "rtf": self.rtf, "base_pos": [x, y, z],
                   "base_quat_wxyz": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                   "base_lin_vel_w": [0.0, 0.0, 0.0], "base_ang_vel_w": [0.0, 0.0, 0.0], "yaw": yaw,
                   "pelvis_z": z, "fallen": False, "foot_contact": {"left": True, "right": True},
                   "band": self.band, "lowcmd_age_s": 0.01}
            if self.m2b and self.links:
                msg["links"] = self._links()
                msg["waist_q"] = [0.0, 0.0, 0.0]
            with self._lock:
                self._follow()
                self.pub.send_multipart([b"gt.pose", msgpack.packb(msg, use_bin_type=True)])
                now = time.monotonic()
                if self.m2b and self.objects_hz > 0 and now - t_obj >= 1.0 / self.objects_hz:
                    t_obj = now
                    objs = [self._obj(k) for k, o in self.objects.items() if o["dynamic"] or o["held_by"]]
                    self.pub.send_multipart([b"gt.objects", msgpack.packb(
                        {"seq": seq, "t_sim": seq * 0.02, "t_wall": time.time(), "objects": objs}, use_bin_type=True)])
                if self.m2b and self.health_hz > 0 and now - t_health >= 1.0 / self.health_hz:
                    t_health = now
                    self.pub.send_multipart([b"sim.health", msgpack.packb(self.health(), use_bin_type=True)])
            seq += 1
            time.sleep(0.02)

    def _publish_frames(self) -> None:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 140, 160)).save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        seq = 0
        while not self._stop.is_set():
            head = self.cameras["head"]
            if head["on"]:
                x, y, z, yaw = self.pose
                t = time.time()
                d = {"timestamps": {"head": t}, "images": {"head": b64}, "head": b64, "camera": "head", "seq": seq,
                     "render_seq": seq, "t_sim": seq / 30.0, "t_capture": t, "t_capture_mono": time.monotonic(),
                     "t_pub": t, "w": 640, "h": 480, "hfov": 90.0, "vfov": 73.74,
                     "cam_pos": [x, y, 1.36], "cam_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                     "cam_pose_wl": [round(x, 3), round(y, 3), round((90.0 - math.degrees(yaw)) % 360.0, 1), 15.0],
                     "stationary": True, "base_speed": 0.0, "base_wz": 0.0, "jpeg_q": 80}
                self.cam_pub.send(msgpack.packb(d, use_bin_type=True))
                seq += 1
            time.sleep(1.0 / 30.0)

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self.rep, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(50):
                continue
            raw = self.rep.recv()
            try:
                req = json.loads(raw)
            except Exception:
                req = msgpack.unpackb(raw, raw=False)
            op = req.get("op")
            args = {**req, **(req.get("args") or {})}
            self.calls.append(op)
            self.requests.append({"op": op, **{k: v for k, v in args.items() if k not in ("op", "args")}})
            try:
                with self._lock:
                    rep = self._op(op, args)
            except Exception as e:  # noqa: BLE001
                rep = {"ok": False, "error": f"internal: {e!r}", "code": "internal"}
            self.rep.send(json.dumps(rep).encode())

    OPS_M1 = ["ping", "get_pose", "get_scene_info", "get_occupancy", "render_topdown", "band", "reset_robot",
              "get_stats"]
    OPS_M2B = ["get_objects", "attach", "detach", "release_all", "get_link_poses", "detections", "camera",
               "get_cameras", "set_render_rates", "reset_scene", "move_object", "set_object_pose", "push_object",
               "get_health"]

    def ops(self) -> list[str]:
        return sorted(self.OPS_M1 + (self.OPS_M2B if self.m2b else []))

    @staticmethod
    def _err(code: str, text: str = "") -> dict:
        return {"ok": False, "code": code, "error": f"{code}: {text}" if text else code}

    def _op(self, op: str, a: dict) -> dict:
        if op not in self.ops():
            return {**self._err("unknown_op", repr(op)), "ops": self.ops() if self.m2b else None}
        if op == "ping":
            rep = {"ok": True, "house_id": self.info["house_id"], "band": self.band}
            if self.m2b:
                rep.update(ops=self.ops(), p1_contract="m2b-1",
                           cameras={n: c["on"] for n, c in self.cameras.items()},
                           topics=["gt.pose", "gt.objects", "gt.event", "sim.health"])
            return rep
        if op == "get_pose":
            return {"ok": True, "base_pos": self.pose[:3], "yaw": self.pose[3]}
        if op == "get_scene_info":
            d = dict(self.info)
            d["objects"] = [{**o, "pos": self._load[o["id"]]["pos"], "aabb": self._load[o["id"]]["aabb"]}
                            for o in self.info["objects"]]
            return {"ok": True, **d}
        if op == "get_occupancy":
            return {"ok": True, "path": str(self.dir / "occupancy.npz"), "resolution": 0.05,
                    "origin": [-0.3, -0.3], "robot_radius": a.get("robot_radius", 0.25), "source": "scene:omap"}
        if op == "render_topdown":
            b = self.info["bounds"]
            rep = {"ok": True, "path": "/nonexistent/topdown.png", "width": 400, "height": 480,
                   "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "meters_per_pixel": 0.025,
                   "extent": [b[0], b[1], b[2], b[3]]}
            if self.m2b:
                rep.update(mode=a.get("mode") or "full", hidden=12 if a.get("mode") == "furniture" else 0)
            return rep
        if op == "band":
            self.band = bool(a.get("on", True))
            return {"ok": True, "band": self.band}
        if op == "reset_robot":
            self.pose = [float(a["x"]), float(a["y"]), 0.78, float(a.get("yaw", 0.0))]
            return {"ok": True, "pose": self.pose}
        if op == "get_stats":
            return {"ok": True, "rtf_1s": self.rtf, "rtf_10s": self.rtf, "object_writes": self.object_writes}
        # ---------------------------------------------------------------- M2b
        if op == "get_objects":
            ids = a.get("ids")
            keys = [k for k in self.objects if (ids is None or k in ids)
                    and (not a.get("dynamic_only") or self.objects[k]["dynamic"] or self.objects[k]["held_by"])]
            return {"ok": True, "t_sim": 0.0, "t_wall": time.time(), "seq": 0, "pose_source": "sim",
                    "objects": [self._obj(k) for k in keys]}
        if op == "attach":
            sid, arm = a.get("id"), a.get("arm")
            if sid not in self.objects:
                return self._err("unknown_object", str(sid))
            o = self.objects[sid]
            if arm not in ("left", "right"):
                return self._err("bad_arg", f"arm {arm!r}")
            if not o["dynamic"]:
                return self._err("not_movable", sid)
            if any(k != sid and x["held_by"] == arm for k, x in self.objects.items()):
                return self._err("hand_busy", arm)
            if o["held_by"] not in (None, arm):
                return self._err("held_by_other", sid)
            gp = self.grip_point(arm)
            dist = math.dist(self._center(o), gp)
            o["held_by"] = arm
            self._put(o, list(gp))
            self.pub.send_multipart([b"gt.event", msgpack.packb({"event": "attach", "id": sid, "arm": arm,
                                                                 "mode": a.get("mode", "follow")}, use_bin_type=True)])
            return {"ok": True, "id": sid, "arm": arm, "mode": a.get("mode", "follow"), "held_by": arm,
                    "snapped": dist > 0.0, "dist_m": round(dist, 3), "grip_point": list(gp), "stepping_stone": True}
        if op == "detach":
            sid = a.get("id")
            if sid not in self.objects:
                return self._err("unknown_object", str(sid))
            o = self.objects[sid]
            was = o["held_by"] is not None
            o["held_by"] = None
            if a.get("pose"):
                self._put(o, [float(v) for v in a["pose"][:3]])
            elif was:                                                  # released in the air: it falls to the floor
                c = self._center(o)
                h = (o["aabb"][1][2] - o["aabb"][0][2]) / 2
                self._put(o, [c[0], c[1], self.info.get("floor_z", 0.0) + h])
            self.pub.send_multipart([b"gt.event", msgpack.packb({"event": "detach", "id": sid,
                                                                 "placed": bool(a.get("pose"))}, use_bin_type=True)])
            return {"ok": True, "id": sid, "was_held": was, "placed": bool(a.get("pose")), "pos": o["pos"],
                    "aabb": o["aabb"], "held_by": None}
        if op == "release_all":
            rel = [k for k, o in self.objects.items() if o["held_by"]]
            for k in rel:
                self.objects[k]["held_by"] = None
            return {"ok": True, "released": rel}
        if op == "get_link_poses":
            links = self._links()
            want = a.get("links") or list(links)
            return {"ok": True, "t_sim": 0.0, "links": {n: links[n] for n in want if n in links},
                    "available": list(links), "unknown": [n for n in want if n not in links]}
        if op == "detections":
            cam = a.get("camera") or "head"
            if cam not in self.cameras:
                return self._err("unknown_camera", cam)
            if not self.cameras[cam]["on"]:
                return self._err("camera_off", cam)
            min_px = int(a.get("min_px", 40))
            dets = [{"id": k, "name": self.objects[k]["name"] if k in self.objects else k, "px": int(px),
                     "bbox": [0, 0, 1, 1], "dist_m": 1.0, "held_by": (self.objects.get(k) or {}).get("held_by")}
                    for k, px in sorted(self.seg_px.items(), key=lambda kv: -kv[1]) if px >= min_px]
            return {"ok": True, "camera": cam, "t_sim": 0.0, "render_seq": 1, "frame_seq": 1, "w": 640, "h": 480,
                    "method": "fake-segmentation", "min_px": min_px, "detections": dets,
                    "other_px": {"robot": 0, "structure": 0, "background": 0}, "cam_pose_wl": [0, 0, 0, 0], "ms": 1.0}
        if op == "camera":
            name = a.get("name")
            if name not in self.cameras:
                return self._err("unknown_camera", str(name))
            c = self.cameras[name]
            consumer = a.get("consumer") or "anon"
            if a.get("on") is True:
                c["consumers"].add(consumer)
            elif a.get("on") is False:
                if a.get("consumer"):
                    c["consumers"].discard(consumer)
                else:
                    c["consumers"].clear()
            if a.get("hz"):
                c["hz"] = float(a["hz"])
            c["on"] = bool(c["consumers"])
            self.pub.send_multipart([b"gt.event", msgpack.packb({"event": "camera", "name": name, "on": c["on"],
                                                                 "hz": c["hz"], "consumers": sorted(c["consumers"])},
                                                                use_bin_type=True)])
            return {"ok": True, "name": name, "on": c["on"], "hz": c["hz"], "consumers": sorted(c["consumers"]),
                    "port": (5565 if name == "head" else 5566) + self.off, "warmup_frames": 4}
        if op == "get_cameras":
            return {"ok": True, "cameras": [{"name": n, "on": c["on"], "hz": c["hz"], "consumers": sorted(c["consumers"])}
                                            for n, c in self.cameras.items()], "stream_camera": "head", "render_hz": 30}
        if op == "set_render_rates":
            for n, key in (("head", "head_hz"), ("ego_view", "ego_hz")):
                if key not in a:
                    continue
                c, hz = self.cameras[n], float(a[key])
                if hz <= 0:
                    c["consumers"].clear()
                    c["on"] = False
                else:
                    c["hz"] = hz
            return {"ok": True, "head": {"hz": self.cameras["head"]["hz"], "on": self.cameras["head"]["on"]},
                    "ego_view": {"hz": self.cameras["ego_view"]["hz"], "on": self.cameras["ego_view"]["on"]}}
        if op == "reset_scene":
            rel = [k for k, o in self.objects.items() if o["held_by"]]
            n = 0
            for k, o in self.objects.items():
                o["held_by"] = None
                if o["dynamic"]:
                    o["aabb"] = json.loads(json.dumps(self._load[k]["aabb"]))
                    o["pos"] = list(self._load[k]["pos"])
                    n += 1
            for k, p in (a.get("poses") or {}).items():
                if k in self.objects:
                    self._put(self.objects[k], [float(v) for v in p[:3]])
            if isinstance(a.get("robot"), dict):
                r = a["robot"]
                self.pose = [float(r["x"]), float(r["y"]), 0.78, float(r.get("yaw", 0.0))]
            return {"ok": True, "variant": a.get("variant", "default"), "objects_reset": n,
                    "poses_applied": len(a.get("poses") or {}), "robot_reset": bool(a.get("robot")),
                    "released": rel, "ms": 5.0, "object_writes": self.object_writes, "root_writes": 0}
        if op in ("move_object", "set_object_pose"):
            sid = a.get("id")
            if sid not in self.objects:
                return self._err("unknown_object", str(sid))
            o = self.objects[sid]
            o["held_by"] = None
            self._put(o, [float(v) for v in a["pose"][:3]])
            self.object_writes += 1
            return {"ok": True, "id": sid, "pos": o["pos"], "aabb": o["aabb"]}
        if op == "push_object":
            sid = a.get("id")
            if sid not in self.objects:
                return self._err("unknown_object", str(sid))
            self.objects[sid]["vel"] = [float(v) for v in a.get("vel") or [0, 0, 0]]
            self.object_writes += 1
            return {"ok": True, "id": sid, "vel": self.objects[sid]["vel"]}
        if op == "get_health":
            return {"ok": True, **self.health()}
        return {"ok": False, "error": "unhandled"}

    def move_object(self, scene_id: str, center: list[float]) -> None:
        with self._lock:
            self._put(self.objects[scene_id], list(center))
