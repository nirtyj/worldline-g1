"""A minimal fake P1 (wl-isaac) for world/isaac_client.py tests: REP (JSON) + PUB gt.pose, serving a recorded house.

M1 contract ops (docs/contracts/m1.md §1.6): ping, get_pose, get_scene_info, get_occupancy, render_topdown, band,
reset_robot, get_stats. With `m2b=True` it also serves the ops this repo asks P1 to add in M2b (report):
    get_objects  -> {objects: [{id, pos, aabb, held_by}]}           live object poses (PhysX)
    attach       {id, arm, mode}                                     object follows the palm
    detach       {id, pose?: [x, y, z]}                              object released at pose (or dropped)
and lists every op in ping's `ops`.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import msgpack
import zmq

HOUSES = Path(__file__).resolve().parent / "houses"


class FakeP1World:
    def __init__(self, house: str = "procthor-train-38", port_offset: int = 400, m2b: bool = False):
        self.dir = HOUSES / house
        self.info = json.loads((self.dir / "house_info.json").read_text())
        self.info.setdefault("bounds", [self.info["bounds_xy"][0][0], self.info["bounds_xy"][0][1],
                                        self.info["bounds_xy"][1][0], self.info["bounds_xy"][1][1]])
        self.off = port_offset
        self.m2b = m2b
        sp = self.info["spawn"]
        self.pose = [float(sp["x"]), float(sp["y"]), 0.78, float(sp["yaw"])]
        self.objects = {o["id"]: {"pos": list(o["pos"]), "aabb": [list(o["aabb"][0]), list(o["aabb"][1])],
                                  "held_by": None} for o in self.info["objects"]}
        self.calls: list[str] = []
        self.band = True
        self._stop = threading.Event()
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
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self.rep.close(0)
        self.pub.close(0)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.pose = [x, y, 0.78, yaw]

    # ------------------------------------------------------------------
    def _publish(self) -> None:
        seq = 0
        while not self._stop.is_set():
            x, y, z, yaw = self.pose
            msg = {"seq": seq, "t_sim": seq * 0.02, "t_wall": time.time(), "rtf": 1.0, "base_pos": [x, y, z],
                   "base_quat_wxyz": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                   "base_lin_vel_w": [0.0, 0.0, 0.0], "base_ang_vel_w": [0.0, 0.0, 0.0], "yaw": yaw,
                   "pelvis_z": z, "fallen": False, "foot_contact": {"left": True, "right": True},
                   "band": self.band, "lowcmd_age_s": 0.01}
            self.pub.send_multipart([b"gt.pose", msgpack.packb(msg, use_bin_type=True)])
            seq += 1
            time.sleep(0.02)

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
            try:
                rep = self._op(op, args)
            except Exception as e:  # noqa: BLE001
                rep = {"ok": False, "error": repr(e)}
            self.rep.send(json.dumps(rep).encode())

    def _op(self, op: str, a: dict) -> dict:
        ops = ["ping", "get_pose", "get_scene_info", "get_occupancy", "render_topdown", "band", "reset_robot",
               "get_stats"]
        if self.m2b:
            ops += ["get_objects", "attach", "detach"]
        if op not in ops:
            return {"ok": False, "error": f"unknown op {op!r}"}
        if op == "ping":
            rep = {"ok": True, "house_id": self.info["house_id"], "band": self.band}
            if self.m2b:
                rep["ops"] = ops
            return rep
        if op == "get_pose":
            return {"ok": True, "base_pos": self.pose[:3], "yaw": self.pose[3]}
        if op == "get_scene_info":
            d = dict(self.info)
            d["objects"] = [{**o, "pos": self.objects[o["id"]]["pos"], "aabb": self.objects[o["id"]]["aabb"]}
                            for o in self.info["objects"]]
            return {"ok": True, **d}
        if op == "get_occupancy":
            return {"ok": True, "path": str(self.dir / "occupancy.npz"), "resolution": 0.05,
                    "origin": [-0.3, -0.3], "robot_radius": a.get("robot_radius", 0.25), "source": "scene:omap"}
        if op == "render_topdown":
            b = self.info["bounds"]
            return {"ok": True, "path": "/nonexistent/topdown.png", "width": 400, "height": 480,
                    "center": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "meters_per_pixel": 0.025,
                    "extent": [b[0], b[1], b[2], b[3]]}
        if op == "band":
            self.band = bool(a.get("on", True))
            return {"ok": True, "band": self.band}
        if op == "reset_robot":
            self.pose = [float(a["x"]), float(a["y"]), 0.78, float(a.get("yaw", 0.0))]
            return {"ok": True, "pose": self.pose}
        if op == "get_stats":
            return {"ok": True, "rtf_1s": 1.0, "rtf_10s": 1.0}
        if op == "get_objects":
            return {"ok": True, "objects": [{"id": k, **v} for k, v in self.objects.items()]}
        if op == "attach":
            self.objects[a["id"]]["held_by"] = a["arm"]
            return {"ok": True}
        if op == "detach":
            o = self.objects[a["id"]]
            o["held_by"] = None
            if a.get("pose"):
                c = a["pose"]
                lo, hi = o["aabb"]
                h = [(hi[i] - lo[i]) / 2 for i in range(3)]
                o["aabb"] = [[c[i] - h[i] for i in range(3)], [c[i] + h[i] for i in range(3)]]
                o["pos"] = list(c)
            return {"ok": True}
        return {"ok": False, "error": "unhandled"}

    def move_object(self, scene_id: str, center: list[float]) -> None:
        o = self.objects[scene_id]
        lo, hi = o["aabb"]
        h = [(hi[i] - lo[i]) / 2 for i in range(3)]
        o["aabb"] = [[center[i] - h[i] for i in range(3)], [center[i] + h[i] for i in range(3)]]
        o["pos"] = list(center)
