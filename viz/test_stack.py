#!/usr/bin/env python3
"""VIZ TEST ONLY: a stand-in for P1 so the viz stack (VizCams, server, recorder) can be tested before M1 lands.

!!! The robot is moved KINEMATICALLY (root pose + joint angles written every step, gravity off, no collisions).
!!! This is NOT locomotion and proves nothing about SONIC; every gt.pose carries note="VIZ TEST: kinematic".

Runs in /work/envs/isaaclab (Isaac Sim 5.1 + Isaac Lab 2.3.2), headless with cameras:

    /work/envs/isaaclab/bin/python viz/test_stack.py --port-offset 100 --level low [--duration 120]
    /work/envs/isaaclab/bin/python viz/test_stack.py --bench            # RTF cost of viz levels, then exit
    /work/envs/isaaclab/bin/python viz/test_stack.py --selftest         # top-view mapping + frame latency check
    /work/envs/isaaclab/bin/python viz/test_stack.py --settle-sweep     # top snapshot noise vs settle renders

RTX preset: "balanced" by default, like sim_isaac/app.py (Isaac Lab applies no preset at all when cameras are on and
--rendering_mode is not given, and those frames are as speckled as "performance").

What it publishes (ports = contract ports + offset, docs/contracts/m1.md section 0):
  head camera   PUB 5565+off  gear_sonic msgpack, d435 mount on torso_link, 640x480, VFOV 45 deg, cv2-encoded RGB
                              exactly like the contract (so R/B appear swapped to standard decoders)
  gt.pose       PUB 5601+off  multipart [b"gt.pose", msgpack] at 50 Hz sim, contract fields + "note"
  P1 REP        REP 5600+off  ping, get_pose, get_scene_info, get_occupancy (npz), render_topdown (cached, made with
                              sim_isaac.camera.render_topdown), get_stats, viz_level, viz_stats
  VizCams       PUB 5602+off  frame.chase / frame.top / frame.overview
  fake body     ROUTER 5610+off / PUB 5611+off, same wire format as body/service.py: ops stand|walk|go_to|
                turn_to|stop|status|ping; events [b"body.event", JSON]; [b"body.state", JSON] at 5 Hz. The
                commands move the kinematic robot; after 20 s without an op the demo loop resumes.

Scene: scenes/loader.py's house if it exists and loads (--house), else a "viz test flat": 10 x 8 m floor, walls,
a partition with two doors, furniture boxes, and a ceiling over the right half (to check the top-view ceiling cut).
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description="VIZ TEST ONLY (kinematic robot)")
ap.add_argument("--port-offset", type=int, default=100)
ap.add_argument("--level", default="low", choices=["off", "min", "low", "high"])
ap.add_argument("--viz-hz", type=float, default=10.0)
ap.add_argument("--viz-render", default="auto", choices=["auto", "own", "piggyback"])
ap.add_argument("--head-hz", type=float, default=30.0, help="head camera rate (sim time), like P1 --camera-hz")
ap.add_argument("--head", default="640x480", help="head camera WxH or 'none'")
ap.add_argument("--physics-hz", type=float, default=200.0)
ap.add_argument("--physx-device", default="cpu")
ap.add_argument("--rt-pace", action="store_true", help="sleep to keep sim time <= wall time (like P1 --rt-pace)")
ap.add_argument("--duration", type=float, default=None, help="sim seconds, then exit")
ap.add_argument("--house", default="auto", help="auto | empty | <house id for scenes.loader>")
ap.add_argument("--speed", type=float, default=0.5, help="demo loop speed m/s")
ap.add_argument("--bench", action="store_true", help="measure RTF per viz level (unpaced), write bench.json, exit")
ap.add_argument("--bench-secs", type=float, default=20.0, help="sim seconds per bench phase")
ap.add_argument("--bench-phases", default="off/auto,low/auto,high/auto,low/own,high/own,off/auto",
                help="comma list of level/render_mode phases")
ap.add_argument("--selftest", action="store_true", help="check top-view mapping + frame latency, then exit")
ap.add_argument("--top-settle", type=int, default=None, help="renders before a top snapshot is read (VizCams default 4)")
ap.add_argument("--settle-sweep", action="store_true", help="top snapshot noise vs settle renders, then exit")
ap.add_argument("--out", default=str(REPO / "outputs" / "viz_test"))
AppLauncher.add_app_launcher_args(ap)
ap.set_defaults(rendering_mode="balanced")  # = sim_isaac/app.py; unset gives noisy frames (see docstring)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------------------- Isaac imports
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
import msgpack  # noqa: E402
import cv2  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from viz.isaac_cams import VizCams, _look_at_rows  # noqa: E402

NOTE = "VIZ TEST: kinematic, not locomotion"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
OFF = args.port_offset
P = {"head": 5565 + OFF, "gt_rep": 5600 + OFF, "gt_pub": 5601 + OFF, "frames": 5602 + OFF,
     "body_ctl": 5610 + OFF, "body_evt": 5611 + OFF}


def log(msg: str) -> None:
    print(f"[viz_test] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- scene
def quat_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


BOXES = []  # (name, center xyz, size xyz, color) -- also used for the fake occupancy / scene info


def build_flat(stage) -> dict:
    """10 x 8 m flat: outer walls, partition at x=0.5 with doors at y=+-2, furniture, ceiling over the right half."""
    cfg = sim_utils.GroundPlaneCfg(color=(0.32, 0.30, 0.28))
    cfg.func("/World/ground", cfg)
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.95, 1.0))
    dome.func("/World/Light/dome", dome)
    sun = sim_utils.DistantLightCfg(intensity=2200.0, angle=1.0)
    sun.func("/World/Light/sun", sun, orientation=(0.88, 0.2, 0.35, 0.25))
    lamp = sim_utils.SphereLightCfg(intensity=60000.0, radius=0.15, color=(1.0, 0.92, 0.8))
    lamp.func("/World/Light/lamp", lamp, translation=(2.75, 0.0, 2.2))
    H, T = 2.4, 0.12
    wall = (0.78, 0.78, 0.74)
    walls = [
        ("wall_s", (0.0, -4.0, H / 2), (10.0 + T, T, H), wall), ("wall_n", (0.0, 4.0, H / 2), (10.0 + T, T, H), wall),
        ("wall_w", (-5.0, 0.0, H / 2), (T, 8.0, H), wall), ("wall_e", (5.0, 0.0, H / 2), (T, 8.0, H), wall),
        # partition x=0.5 with door gaps y in [1.4, 2.6] and [-2.6, -1.4]
        ("part_n", (0.5, 3.3, H / 2), (T, 1.4, H), wall), ("part_m", (0.5, 0.0, H / 2), (T, 2.8, H), wall),
        ("part_s", (0.5, -3.3, H / 2), (T, 1.4, H), wall),
    ]
    furniture = [
        ("table", (-2.0, 0.0, 0.375), (1.2, 0.8, 0.75), (0.55, 0.35, 0.2)),
        ("sofa", (2.2, 0.0, 0.4), (1.6, 0.8, 0.8), (0.2, 0.35, 0.65)),  # clear of the loop (x=3.5) and the partition
        ("fridge", (4.4, 3.4, 0.95), (0.8, 0.8, 1.9), (0.9, 0.9, 0.92)),
        ("shelf", (-4.6, 0.0, 0.9), (0.5, 1.6, 1.8), (0.35, 0.5, 0.3)),
        ("bed", (2.9, -3.1, 0.3), (2.0, 1.4, 0.6), (0.7, 0.3, 0.35)),
        ("plant", (-4.4, 3.4, 0.6), (0.5, 0.5, 1.2), (0.2, 0.6, 0.25)),
    ]
    ceiling = [("ceiling_right", (2.75, 0.0, 2.55), (4.5, 8.0, 0.1), (0.9, 0.9, 0.9))]
    for name, c, s, col in walls + furniture + ceiling:
        cube = sim_utils.CuboidCfg(size=s, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col),
                                   collision_props=sim_utils.CollisionPropertiesCfg())
        cube.func(f"/World/Flat/{name}", cube, translation=c)
        if not name.startswith("ceiling"):
            BOXES.append((name, c, s, col))
    rooms = [{"id": "left", "name": "living", "type": "LivingRoom",
              "polygon": [[-5, -4], [0.5, -4], [0.5, 4], [-5, 4]]},
             {"id": "right", "name": "bedroom", "type": "Bedroom (ceiling)",
              "polygon": [[0.5, -4], [5, -4], [5, 4], [0.5, 4]]}]
    objects = [{"id": n, "category": n, "room_id": "left" if c[0] < 0.5 else "right", "pos": list(c),
                "aabb": [[c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2],
                         [c[0] + s[0] / 2, c[1] + s[1] / 2, c[2] + s[2] / 2]]} for n, c, s, _ in furniture]
    return {"house_id": "viz-test-flat", "floor_z": 0.0, "bounds": [-5.1, -4.1, 5.1, 4.1], "rooms": rooms,
            "objects": objects, "spawn": {"x": -3.5, "y": -2.0, "yaw": 0.0}}


HOUSE_OCC = None  # occupancy npz of a loaded house (assets/houses/<id>/occupancy.npz)


def try_house(stage, house_id: str) -> dict | None:
    """scenes.loader.load_house (M1 scene agent) if present; returns a get_scene_info-shaped dict."""
    global HOUSE_OCC
    loader = REPO / "scenes" / "loader.py"
    if house_id == "empty" or not loader.exists():
        return None
    hid = "procthor-train-40" if house_id == "auto" else house_id
    try:
        from scenes.loader import load_house  # noqa: WPS433

        info = load_house(stage, hid)
    except Exception as e:  # noqa: BLE001
        log(f"scenes.loader could not load {hid} ({e!r}); using the viz test flat")
        return None
    # the house is on the stage now: from here on never fall back to the flat (it would overlap the house)
    if hasattr(info, "to_scene_info"):
        d = info.to_scene_info()
    elif hasattr(info, "scene_info"):
        d = info.scene_info()
    else:
        d = {k: getattr(info, k, None) for k in ("house_id", "floor_z", "rooms", "objects", "spawn", "room_points")}
    b = d.get("bounds") or d.get("bounds_xy") or getattr(info, "bounds_xy", None)
    if b is not None and len(b) == 2:
        b = [b[0][0], b[0][1], b[1][0], b[1][1]]
    d["bounds"] = [float(v) for v in b]
    d["floor_z"] = float(d.get("floor_z") or 0.0)
    d["room_points"] = d.get("room_points") or getattr(info, "room_points", []) or []
    sp = d.get("spawn") or {"x": (d["bounds"][0] + d["bounds"][2]) / 2, "y": (d["bounds"][1] + d["bounds"][3]) / 2}
    d["spawn"] = {"x": float(sp["x"]), "y": float(sp["y"]), "yaw": float(sp.get("yaw", 0.0) or 0.0)}
    occ = d.get("occupancy_npz") or getattr(info, "occupancy_npz", None)
    if not occ and getattr(info, "assets_dir", None) and (Path(info.assets_dir) / "occupancy.npz").exists():
        occ = str(Path(info.assets_dir) / "occupancy.npz")
    HOUSE_OCC = occ
    log(f"house {hid} loaded via scenes.loader: bounds {d['bounds']} rooms {len(d.get('rooms') or [])} "
        f"occupancy {HOUSE_OCC}")
    return d


_GRID = None  # (blocked bool[R, C], res, ox, oy) from the house occupancy npz


def _grid():
    global _GRID
    if _GRID is None and HOUSE_OCC:
        z = np.load(HOUSE_OCC)
        blocked = np.asarray(z["inflated"] if "inflated" in z.files else z["occ_inflated"]) > 0
        res = float(np.asarray(z["resolution"]).reshape(-1)[0])
        ox, oy = (float(v) for v in np.asarray(z["origin"]).reshape(-1)[:2])
        _GRID = (blocked, res, ox, oy)
    return _GRID


def grid_path(a: tuple[float, float], b: tuple[float, float], every: int = 6) -> list[tuple[float, float]] | None:
    """BFS (8-connected) on the inflated occupancy grid from world point a to b; every `every`-th cell centre plus
    b itself. None if there is no grid or no path (b blocked / unreachable). VIZ TEST helper, not the body planner."""
    from collections import deque as _dq

    g = _grid()
    if g is None:
        return None
    blocked, res, ox, oy = g
    R, C = blocked.shape

    def cell(x, y):
        return int(np.clip((y - oy) / res, 0, R - 1)), int(np.clip((x - ox) / res, 0, C - 1))

    s0, s1 = cell(*a), cell(*b)
    if blocked[s1]:
        return None
    prev = {s0: None}
    q = _dq([s0])
    while q:
        cur = q.popleft()
        if cur == s1:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < R and 0 <= n[1] < C and not blocked[n] and n not in prev:
                prev[n] = cur
                q.append(n)
    if s1 not in prev:
        return None
    out, cur = [], s1
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    cells = out[::-1]
    return [(ox + (c + 0.5) * res, oy + (r + 0.5) * res) for r, c in cells[::every]] + [tuple(b)]


def house_tour(scene: dict) -> list[tuple[float, float]] | None:
    """Closed loop spawn -> every room point -> spawn, BFS on the inflated occupancy grid (goes through doors)."""
    if not HOUSE_OCC:
        return None
    sp = scene["spawn"]
    stops = [(sp["x"], sp["y"])] + [(p["x"], p["y"]) for p in scene.get("room_points", []) if p.get("ok", True)]
    if len(stops) < 2:
        return None
    pts = []
    for a, b in zip(stops, stops[1:] + stops[:1]):
        path = grid_path(a, b)
        if path is not None:
            pts += path[:-1]
    if len(pts) < 4:
        return None
    # light smoothing so the kinematic heading does not zig-zag along grid steps
    arr = np.array(pts)
    for _ in range(3):
        arr[1:-1] = 0.25 * arr[:-2] + 0.5 * arr[1:-1] + 0.25 * arr[2:]
    return [tuple(p) for p in arr]


# ---------------------------------------------------------------------------------------------- sim setup
# scene queries on, like sim_isaac/app.py:119: VizCams' chase camera raycasts need them (Isaac Lab's default is off)
sim = SimulationContext(SimulationCfg(dt=1.0 / args.physics_hz, device=args.physx_device, render_interval=1,
                                      enable_scene_query_support=True))
stage = omni.usd.get_context().get_stage()
scene = try_house(stage, args.house)
if scene is None:
    scene = build_flat(stage)
    house_mode = False
else:
    house_mode = True
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))  # as sim_isaac/scene.py lights
    light.func("/World/Light/viz_dome", light)
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    sun.func("/World/Light/viz_sun", sun)
FLOOR = float(scene["floor_z"])

g1_usds = sorted(glob.glob("/work/worldline-g1/assets/g1/*.usd"))
if g1_usds:
    usd_path = g1_usds[0]
    log(f"G1 asset: {usd_path} (M1 build)")
else:
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: WPS433

    usd_path = G1_29DOF_CFG.spawn.usd_path
    log(f"G1 asset: {usd_path} (Isaac Lab G1_29DOF_CFG)")
robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(scene["spawn"]["x"], scene["spawn"]["y"], FLOOR + 0.78)),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=400.0, damping=20.0)},
)
robot = Articulation(robot_cfg)

# head camera (the P1 ego camera stand-in): d435_link offset on torso_link (contract 1.4)
torso = None
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/Robot") and prim.GetName() == "torso_link":
        torso = prim.GetPath().pathString
        break
head = None
if args.head != "none":
    hw, hh = (int(v) for v in args.head.split("x"))
    head_path = f"{torso or '/World/Robot'}/viz_test_head_cam"
    cam = UsdGeom.Camera.Define(stage, head_path)
    fl = 18.0
    vap = 2 * fl * math.tan(math.radians(45.0) / 2)
    cam.CreateFocalLengthAttr().Set(fl)
    cam.CreateVerticalApertureAttr().Set(vap)
    cam.CreateHorizontalApertureAttr().Set(vap * hw / hh)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 50.0))
    # camera looks along the link's +x, up +z, then pitched down 0.8308 rad about y (URDF d435_link rpy)
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])  # rows: cam x, y, z axes in link frame
    p = 0.8307767
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    axes = (Ry @ base.T).T
    m = np.eye(4)
    m[:3, :3] = axes
    m[3, :3] = [0.0576235, 0.01753, 0.41987]
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*[float(v) for v in m.flatten()]))
    head_rp = rep.create.render_product(head_path, (hw, hh), name="viz_test_head")
    head_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    head_annot.attach([head_rp.path])
    head = (head_rp, head_annot)
    log(f"head camera {hw}x{hh} @ {args.head_hz:g} Hz under {torso}")

# selftest marker (top-view mapping + latency)
marker = None
if args.selftest:
    mk = sim_utils.CuboidCfg(size=(0.45, 0.45, 0.45),
                             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0),
                                                                         emissive_color=(1.0, 0.0, 1.0)))
    mk.func("/World/viz_marker", mk, translation=(0.0, -3.4, 1.2))  # above furniture, below the 2.0 m ceiling cut
    marker = UsdGeom.Xformable(stage.GetPrimAtPath("/World/viz_marker"))

sim.reset()
log(f"sim ready: dt={sim.get_physics_dt()} device={args.physx_device} joints={len(robot.joint_names)}")

# ---------------------------------------------------------------------------------------------- kinematic "robot"
J = {n: i for i, n in enumerate(robot.joint_names)}
q_default = robot.data.default_joint_pos.clone()


def jidx(name: str) -> int | None:
    return J.get(name)


class Kinematic:
    """Demo loop (rounded rectangle through both doors) + fake body ops. Positions in world metres."""

    def __init__(self):
        sp = scene["spawn"]
        self.x, self.y, self.yaw = float(sp["x"]), float(sp["y"]), float(sp.get("yaw", 0.0))
        self.phase = 0.0
        self.v = 0.0            # current speed for the gait amplitude
        self.mode = "loop"
        self.op = None          # active fake body op dict
        self.idle_since = 0.0
        if house_mode:
            self.path = house_tour(scene)
            if self.path is None:
                cx, cy = self.x, self.y
                self.path = [(cx + 0.8 * math.cos(a) - 0.8, cy + 0.8 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64)]
            log(f"kinematic tour: {len(self.path)} waypoints, {sum(self._seglens()):.1f} m")
        else:
            self.path = self._rounded_rect(-3.5, -2.0, 3.5, 2.0, 1.0)
        self.s = self._nearest_s(self.x, self.y)
        self.x, self.y = self._at(self.s)
        self.yaw = self._tangent(self.s)

    @staticmethod
    def _rounded_rect(x0, y0, x1, y1, r):
        pts = []
        for cx, cy, a0 in ((x1 - r, y0 + r, -math.pi / 2), (x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, math.pi / 2),
                           (x0 + r, y0 + r, math.pi)):
            for a in np.linspace(a0, a0 + math.pi / 2, 12):
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    def _seglens(self):
        pts = self.path + [self.path[0]]
        return [math.dist(pts[i], pts[i + 1]) for i in range(len(self.path))]

    def _at(self, s):
        L = self._seglens()
        s %= sum(L)
        pts = self.path + [self.path[0]]
        for i, l in enumerate(L):
            if s <= l:
                f = s / l if l > 0 else 0
                return pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
            s -= l
        return pts[0]

    def _tangent(self, s):
        a, b = self._at(s), self._at(s + 0.3)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _nearest_s(self, x, y):
        best, bs, acc = 1e9, 0.0, 0.0
        L = self._seglens()
        for k in np.linspace(0, sum(L), 400):
            px, py = self._at(k)
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best:
                best, bs = d, k
        return bs

    def update(self, dt: float, t: float, events: list):
        v = 0.0
        if self.mode == "loop":
            tgt_s = self.s + args.speed * dt
            nx, ny = self._at(tgt_s)
            if math.hypot(nx - self.x, ny - self.y) > 0.3:  # off the path: walk back to it first
                self._move_towards(self._at(self.s), args.speed, dt)
                v = args.speed
            else:
                self.x, self.y, self.s = nx, ny, tgt_s
                self.yaw = self._turn(self.yaw, self._tangent(self.s), 1.5, dt)
                v = args.speed
        elif self.mode == "op" and self.op is not None:
            v = self._run_op(dt, t, events)
        else:  # hold
            if t - self.idle_since > 20.0 and not args.selftest:
                self.mode = "loop"
                self.s = self._nearest_s(self.x, self.y)
        self.v += (v - self.v) * min(1.0, dt / 0.3)
        self.phase += 2 * math.pi * (1.1 + 1.4 * self.v) * dt if self.v > 0.02 else 0.0

    @staticmethod
    def _turn(cur, want, rate, dt):
        d = wrap(want - cur)
        return wrap(cur + max(-rate * dt, min(rate * dt, d)))

    def _move_towards(self, xy, speed, dt):
        dx, dy = xy[0] - self.x, xy[1] - self.y
        d = math.hypot(dx, dy)
        want = math.atan2(dy, dx)
        self.yaw = self._turn(self.yaw, want, 1.5, dt)
        if abs(wrap(want - self.yaw)) < 0.6 and d > 1e-3:
            step = min(d, speed * dt)
            self.x += dx / d * step
            self.y += dy / d * step
        return d

    def start_op(self, op: dict, t: float, events: list):
        if self.op is not None:
            events.append(("canceled", self.op, {"reason": "preempted" if op["op"] != "stop" else "stop"}))
        if op["op"] in ("stop", "stand"):
            self.op = None
            self.mode = "hold"
            self.idle_since = t
            if op["op"] == "stand":
                op["t0"] = t
                events.append(("succeeded", op, {"note": NOTE}))
            else:
                events.append(("succeeded", op, {}))
            return
        op["t0"] = t
        op["last_progress"] = t
        if op["op"] == "go_to" and house_mode:  # follow a grid path through doors (not through walls)
            a = op.get("args") or {}
            path = grid_path((self.x, self.y), (float(a["x"]), float(a["y"])), every=4)
            if path is None:
                events.append(("failed", op, {"reason": "no_path", "note": NOTE}))
                self.op, self.mode, self.idle_since = None, "hold", t
                return
            op["path"], op["wp"] = path, 0
        self.op, self.mode = op, "op"

    def _run_op(self, dt, t, events):
        op = self.op
        a = op.get("args") or {}
        name = op["op"]
        v = 0.0
        done = False
        if name == "walk":
            vx, vy, wz = float(a.get("vx", 0)), float(a.get("vy", 0)), float(a.get("yaw_rate", 0))
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (c * vx - s * vy) * dt
            self.y += (s * vx + c * vy) * dt
            self.yaw = wrap(self.yaw + wz * dt)
            v = math.hypot(vx, vy) + 0.3 * abs(wz)
            done = t - op["t0"] >= float(a.get("duration_s", 2.0))
        elif name == "go_to":
            path = op.get("path")
            if path is not None and op["wp"] < len(path) - 1:  # intermediate waypoints: no stop, 0.15 m capture
                if self._move_towards(path[op["wp"]], float(a.get("speed", 0.5)), dt) < 0.15:
                    op["wp"] += 1
                d = 1.0
            else:
                d = self._move_towards((float(a["x"]), float(a["y"])), float(a.get("speed", 0.5)), dt)
            v = 0.5
            if d < 0.05:
                if "yaw" in a and a["yaw"] is not None:
                    self.yaw = self._turn(self.yaw, float(a["yaw"]), 1.0, dt)
                    done = abs(wrap(float(a["yaw"]) - self.yaw)) < 0.02
                else:
                    done = True
        elif name == "turn_to":
            want = float(a.get("yaw", 0.0)) + (op.get("yaw0", self.yaw) if a.get("relative") else 0.0)
            op.setdefault("yaw0", self.yaw)
            self.yaw = self._turn(self.yaw, want, 0.8, dt)
            v = 0.2
            done = abs(wrap(want - self.yaw)) < 0.02
        if t - op["last_progress"] >= 1.0:
            op["last_progress"] = t
            events.append(("progress", op, {"pose": [round(self.x, 3), round(self.y, 3), round(self.yaw, 3)]}))
        if done:
            events.append(("succeeded", op, {"pose": [round(self.x, 3), round(self.y, 3), round(self.yaw, 3)],
                                             "duration_s": round(t - op["t0"], 2)}))
            self.op, self.mode, self.idle_since = None, "hold", t
        return v

    def joints(self) -> torch.Tensor:
        q = q_default.clone()
        amp = min(1.0, self.v / 0.4)
        ph = self.phase
        vals = {
            "left_hip_pitch_joint": -0.25 - 0.45 * amp * math.sin(ph),
            "right_hip_pitch_joint": -0.25 + 0.45 * amp * math.sin(ph),
            "left_knee_joint": 0.45 + 0.55 * amp * max(0.0, math.cos(ph)),
            "right_knee_joint": 0.45 + 0.55 * amp * max(0.0, -math.cos(ph)),
            "left_shoulder_pitch_joint": 0.25 + 0.35 * amp * math.sin(ph),
            "right_shoulder_pitch_joint": 0.25 - 0.35 * amp * math.sin(ph),
            "left_elbow_joint": 0.9, "right_elbow_joint": 0.9,
            "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
        }
        vals["left_ankle_pitch_joint"] = -0.5 * (vals["left_hip_pitch_joint"] + vals["left_knee_joint"]) + 0.05
        vals["right_ankle_pitch_joint"] = -0.5 * (vals["right_hip_pitch_joint"] + vals["right_knee_joint"]) + 0.05
        for n, v in vals.items():
            i = jidx(n)
            if i is not None:
                q[:, i] = v
        return q

    def base_z(self) -> float:
        return FLOOR + 0.755 + 0.015 * math.cos(2 * self.phase) * min(1.0, self.v / 0.4)


kin = Kinematic()

# P1's cached top-down render (contract 1.6 render_topdown), made with P1's own function (imported, not copied)
TOPDOWN = None
try:
    from sim_isaac.camera import render_topdown as _p1_render_topdown  # noqa: E402

    TOPDOWN = _p1_render_topdown(sim, stage, tuple(scene["bounds"]), str(OUT / f"_topdown_{scene['house_id']}.png"))
    log(f"P1 render_topdown cached: {TOPDOWN}")
except Exception as e:  # noqa: BLE001
    log(f"P1 render_topdown unavailable: {e!r}")

# ---------------------------------------------------------------------------------------------- IO: ZMQ
ctx = zmq.Context.instance()


def bind(kind, port):
    s = ctx.socket(kind)
    s.setsockopt(zmq.LINGER, 0)
    try:
        s.bind(f"tcp://127.0.0.1:{port}")
    except zmq.ZMQError as e:
        log(f"cannot bind port {port}: {e} (another stack on this offset? use --port-offset 300)")
        simulation_app.close()
        sys.exit(3)
    return s


gt_pub = bind(zmq.PUB, P["gt_pub"])
rep_sock = bind(zmq.REP, P["gt_rep"])
body_router = bind(zmq.ROUTER, P["body_ctl"])
body_pub = bind(zmq.PUB, P["body_evt"])
body_pub.setsockopt(zmq.SNDHWM, 1000)
bounds = scene["bounds"]
_top_over = {"every_s": 0.5, "long": 768} if args.selftest else {}
if args.top_settle is not None:
    _top_over["settle"] = args.top_settle
cams = VizCams(sim, "/World/Robot", bounds, pub=f"tcp://127.0.0.1:{P['frames']}", hz=args.viz_hz,
               level=args.level, floor_z=FLOOR, render=args.viz_render,
               cams={"top": _top_over} if _top_over else None)

# head publisher thread: encode exactly like the contract (cv2.imencode on the RGB array, q80, base64)
head_q: queue.Queue = queue.Queue(maxsize=4)


def head_worker():
    s = ctx.socket(zmq.PUB)
    s.setsockopt(zmq.SNDHWM, 20)
    s.setsockopt(zmq.LINGER, 0)
    s.bind(f"tcp://127.0.0.1:{P['head']}")
    seq = 0
    while True:
        item = head_q.get()
        if item is None:
            break
        rgb, t_sim = item
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buf).decode("utf-8")
        seq += 1
        d = {"timestamps": {"ego_view": time.time()}, "images": {"ego_view": b64}, "ego_view": b64,
             "t_sim": t_sim, "seq": seq}
        try:
            s.send(msgpack.packb(d, use_bin_type=True), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
    s.close(0)


head_thread = threading.Thread(target=head_worker, daemon=True)
head_thread.start()


# fake occupancy from the boxes (contract npz format)
def make_occupancy(res=0.05, radius=0.25):
    x0, y0, x1, y1 = bounds
    cols, rows = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    occ = np.zeros((rows, cols), np.uint8)
    xs = x0 + (np.arange(cols) + 0.5) * res
    ys = y0 + (np.arange(rows) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    for _, c, s, _ in BOXES:
        if c[2] - s[2] / 2 > 1.6:
            continue
        occ[(np.abs(X - c[0]) <= s[0] / 2) & (np.abs(Y - c[1]) <= s[1] / 2)] = 1
    r = int(math.ceil(radius / res))
    infl = occ.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                infl |= np.roll(np.roll(occ, dy, 0), dx, 1)
    path = OUT / "occupancy.npz"
    np.savez(path, occ=occ, occ_inflated=infl, resolution=res, origin=np.array([x0, y0]), robot_radius=radius)
    return {"path": str(path), "resolution": res, "origin": [x0, y0], "shape": [rows, cols],
            "robot_radius": radius, "source": "scene"}


occ_info = make_occupancy() if not house_mode else (
    {"path": HOUSE_OCC, "source": "scene"} if HOUSE_OCC else None)

# ---------------------------------------------------------------------------------------------- loop state
dt = sim.get_physics_dt()
t_sim = 0.0
wall_hist: list[tuple[float, float]] = []
stats = {"render_ms": [], "head_frames": 0, "steps": 0, "t0": time.perf_counter()}
pose_seq = 0
body_seq = 0
events: list = []
last_pose_msg: dict = {}


def body_event(state, op, data):
    global body_seq
    body_seq += 1
    ev = {"id": op.get("id"), "op": op.get("op"), "state": state, "data": data, "seq": body_seq, "t_wall": time.time()}
    body_pub.send_multipart([b"body.event", json.dumps(ev).encode()])


def body_status():
    return {"t_wall": time.time(), "fault": None, "control_started": True, "in_control": True,
            "active": None if kin.op is None else {"id": kin.op.get("id"), "op": kin.op["op"], "phase": "run",
                                                   "elapsed_s": round(t_sim - kin.op["t0"], 2)},
            "mode": kin.mode, "pose": {"x": round(kin.x, 3), "y": round(kin.y, 3), "yaw": round(kin.yaw, 3)},
            "note": "VIZ TEST fake body (kinematic)"}


def poll_body():
    while True:
        try:
            frames = body_router.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return
        ident = frames[0]
        env = [ident, b""] if len(frames) >= 3 and frames[1] == b"" else [ident]
        try:
            req = json.loads(frames[-1])
        except Exception as e:  # noqa: BLE001
            body_router.send_multipart(env + [json.dumps({"ok": False, "state": "rejected", "error": str(e)}).encode()])
            continue
        op = {"id": req.get("id") or f"op-{time.time_ns()}", "op": req.get("op"), "args": req.get("args") or {}}
        if op["op"] in ("status", "ping"):
            rep_ = {"id": op["id"], "ok": True, "state": "done", "data": body_status()}
        elif op["op"] in ("stand", "walk", "go_to", "turn_to", "stop"):
            if op["op"] == "go_to" and ("x" not in op["args"] or "y" not in op["args"]):
                rep_ = {"id": op["id"], "ok": False, "state": "rejected", "error": "go_to needs x, y"}
            else:
                rep_ = {"id": op["id"], "ok": True, "state": "accepted"}
                body_event("accepted", op, {})
                kin.start_op(op, t_sim, events)
        else:
            rep_ = {"id": op["id"], "ok": False, "state": "rejected", "error": f"unknown op {op['op']}"}
        body_router.send_multipart(env + [json.dumps(rep_).encode()])


def rtf_window(win=1.0):
    if len(wall_hist) < 2:
        return None
    w1, s1 = wall_hist[-1]
    for w0, s0 in wall_hist:
        if w1 - w0 <= win:
            return (s1 - s0) / (w1 - w0) if w1 > w0 else None
    return None


def poll_rep():
    while True:
        try:
            raw = rep_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            req = json.loads(raw)
            op = req.get("op")
            if op == "ping":
                r = {"ok": True, "t_sim": t_sim, "t_wall": time.time(), "pid": os.getpid(),
                     "house_id": scene["house_id"], "band": False, "note": NOTE}
            elif op == "get_pose":
                r = {"ok": True, **last_pose_msg}
            elif op == "get_scene_info":
                r = {"ok": True, **scene}
            elif op == "get_occupancy":
                r = {"ok": True, **occ_info} if occ_info else {"ok": False, "error": "no occupancy in house mode"}
            elif op == "get_stats":
                rm = stats["render_ms"][-300:]
                r = {"ok": True, "rtf_1s": rtf_window(1.0), "rtf_10s": rtf_window(10.0), "t_sim": t_sim,
                     "render_ms": {"mean": float(np.mean(rm)) if rm else None}, "physx_device": args.physx_device,
                     "viz": cams.stats(), "note": NOTE}
            elif op in ("viz_level", "viz_stats"):
                r = cams.handle_op(req)
            elif op == "render_topdown":
                if TOPDOWN is None:
                    r = {"ok": False, "error": "no top-down render"}
                else:
                    import shutil

                    r = {"ok": True, **TOPDOWN}
                    path = req.get("path") or (req.get("args") or {}).get("path")
                    if path and path != TOPDOWN["path"]:
                        shutil.copyfile(TOPDOWN["path"], path)
                        r["path"] = path
            else:
                r = {"ok": False, "error": f"op {op!r} not in the viz test stand-in"}
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": repr(e)}
        rep_sock.send(json.dumps(r, default=float).encode())


def pub_pose():
    global pose_seq
    pose_seq += 1
    q = quat_yaw(kin.yaw)
    ph = kin.phase
    moving = kin.v > 0.05
    msg = {"seq": pose_seq, "t_sim": round(t_sim, 4), "t_wall": time.time(), "rtf": rtf_window(1.0),
           "base_pos": [kin.x, kin.y, kin.base_z()], "base_quat_wxyz": q,
           "base_lin_vel_w": [kin.v * math.cos(kin.yaw), kin.v * math.sin(kin.yaw), 0.0], "base_ang_vel_w": [0, 0, 0],
           "yaw": kin.yaw, "pelvis_z": kin.base_z() - FLOOR, "fallen": False,
           "foot_contact": {"left": (not moving) or math.sin(ph) < 0.3, "right": (not moving) or math.sin(ph) > -0.3},
           "band": False, "lowcmd_age_s": None, "note": NOTE}
    last_pose_msg.clear()
    last_pose_msg.update(msg)
    gt_pub.send_multipart([b"gt.pose", msgpack.packb(msg, use_bin_type=True)])


def step_once(render_head: bool):
    """One physics step of the stand-in P1 loop. Returns the head render ms (0 if none)."""
    global t_sim
    kin.update(dt, t_sim, events)
    for st, op, data in events:
        body_event(st, op, data)
    events.clear()
    pose = torch.tensor([[kin.x, kin.y, kin.base_z(), *quat_yaw(kin.yaw)]], dtype=torch.float32, device=sim.device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
    q = kin.joints()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    if marker is not None:
        mx = -4.0 + (t_sim * 1.5) % 8.0
        marker.GetOrderedXformOps()[0].Set(Gf.Vec3d(mx, -3.4, 1.2))
    sim.step(render=False)
    robot.update(dt)
    t_sim += dt
    ms = 0.0
    if render_head:
        t0 = time.perf_counter()
        sim.render()
        ms = (time.perf_counter() - t0) * 1000.0
        if head is not None:
            data = head[1].get_data()
            if data is not None and getattr(data, "size", 0) and data.ndim == 3:
                try:
                    head_q.put_nowait((np.ascontiguousarray(data[:, :, :3]), round(t_sim, 4)))
                    stats["head_frames"] += 1
                except queue.Full:
                    pass
    cams.step(t_sim, [kin.x, kin.y, kin.base_z()], quat_yaw(kin.yaw))
    return ms


def run(seconds: float | None, pace: bool, tag: str = "") -> dict:
    """Run the P1-like loop for `seconds` of sim time (None = until the app closes)."""
    global t_sim
    head_period = 1.0 / args.head_hz if args.head_hz > 0 else None
    next_head = t_sim
    next_pose = t_sim
    next_state = time.time()
    t_start_sim, w_start = t_sim, time.perf_counter()
    render_ms = []
    step_ms = []
    last_log = time.time()
    anchor_w, anchor_s = time.perf_counter(), t_sim
    while simulation_app.is_running() and (seconds is None or t_sim - t_start_sim < seconds - 1e-9):
        t0 = time.perf_counter()
        do_head = head_period is not None and t_sim + dt >= next_head - 1e-9
        if do_head:
            next_head += head_period
        ms = step_once(do_head)
        if ms:
            render_ms.append(ms)
            stats["render_ms"].append(ms)
        stats["steps"] += 1
        if t_sim >= next_pose - 1e-9:
            next_pose += 0.02
            pub_pose()
        poll_rep()
        poll_body()
        if time.time() >= next_state:
            next_state += 0.2
            body_pub.send_multipart([b"body.state", json.dumps(body_status()).encode()])
        now = time.perf_counter()
        step_ms.append((now - t0) * 1000.0)
        if stats["steps"] % 10 == 0:
            wall_hist.append((now, t_sim))
            if len(wall_hist) > 3000:
                del wall_hist[:1000]
        if pace:
            target = anchor_w + (t_sim - anchor_s)
            ahead = target - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
            elif -ahead > 0.1:
                anchor_w, anchor_s = time.perf_counter(), t_sim
        if time.time() - last_log > 10.0:
            last_log = time.time()
            vs = cams.stats()
            log(f"{tag}t_sim {t_sim:7.2f}  rtf_1s {rtf_window(1.0) or 0:.2f}  head {stats['head_frames']}  "
                f"viz sent {vs['sent']} forced {vs['forced_renders']} piggy {vs['piggyback_captures']}  "
                f"mode {kin.mode}  pos ({kin.x:+.2f},{kin.y:+.2f})")
    wall = time.perf_counter() - w_start
    sim_s = t_sim - t_start_sim
    return {"sim_s": round(sim_s, 3), "wall_s": round(wall, 3), "rtf": round(sim_s / wall, 3) if wall > 0 else None,
            "wall_ms_per_sim_s": round(wall / sim_s * 1000.0, 1) if sim_s > 0 else None,
            "step_ms_mean": round(float(np.mean(step_ms)), 3) if step_ms else None,
            "head_render_ms_mean": round(float(np.mean(render_ms)), 2) if render_ms else None,
            "head_renders": len(render_ms)}


def bench() -> None:
    log("BENCH: warm-up 5 s (shader compile, render products)")
    for lvl in ("high", "low"):
        cams.set_level(lvl)
        run(3.0, pace=False, tag="warmup ")
    phases = [tuple(p.split("/")) for p in args.bench_phases.split(",")]
    results = []
    for lvl, mode in phases:
        cams.render_mode = mode
        cams.set_level(lvl)
        before = cams.stats()
        r = run(args.bench_secs, pace=False, tag=f"[{lvl}/{mode}] ")
        after = cams.stats()
        r.update({"level": lvl, "render_mode": mode, "viz_frames": after["sent"] - before["sent"],
                  "forced_renders": after["forced_renders"] - before["forced_renders"],
                  "capture_ms_mean": after["capture_ms_mean"], "encode_ms_mean": after["encode_ms_mean"],
                  "cams": after["cams"]})
        results.append(r)
        log(f"BENCH {lvl:>4}/{mode:<4}: RTF {r['rtf']:.3f}  {r['wall_ms_per_sim_s']} wall-ms per sim-s  "
            f"viz frames {r['viz_frames']} forced {r['forced_renders']}  head render {r['head_render_ms_mean']} ms")
    base = [r for r in results if r["level"] == "off"]
    base_ms = float(np.median([r["wall_ms_per_sim_s"] for r in base]))
    base_rms = float(np.median([r["head_render_ms_mean"] for r in base]))
    for r in results:
        r["overhead_ms_per_sim_s"] = round(r["wall_ms_per_sim_s"] - base_ms, 1)
        r["overhead_pct"] = round(100.0 * (r["wall_ms_per_sim_s"] - base_ms) / base_ms, 1)
    # medians per level/mode: phases are interleaved and repeated so other agents' load hits every level alike
    by_level = {}
    for key in dict.fromkeys(f"{r['level']}/{r['render_mode']}" for r in results):
        rs = [r for r in results if f"{r['level']}/{r['render_mode']}" == key]
        wm = float(np.median([r["wall_ms_per_sim_s"] for r in rs]))
        hm = float(np.median([r["head_render_ms_mean"] for r in rs]))
        by_level[key] = {"n": len(rs), "rtf_median": round(float(np.median([r["rtf"] for r in rs])), 3),
                         "wall_ms_per_sim_s_median": round(wm, 1), "overhead_ms_per_sim_s": round(wm - base_ms, 1),
                         "overhead_pct": round(100.0 * (wm - base_ms) / base_ms, 1),
                         "render_ms_median": round(hm, 2), "render_ms_added": round(hm - base_rms, 2),
                         "viz_frames_per_phase": [r["viz_frames"] for r in rs]}
        log(f"BENCH median {key:>10}: RTF {by_level[key]['rtf_median']:.3f}  "
            f"{by_level[key]['wall_ms_per_sim_s_median']} wall-ms/sim-s ({by_level[key]['overhead_pct']:+.1f} %)  "
            f"render {hm:.2f} ms ({hm - base_rms:+.2f})")
    out = {"physics_hz": args.physics_hz, "head": args.head, "head_hz": args.head_hz, "viz_hz": args.viz_hz,
           "physx_device": args.physx_device, "scene": scene["house_id"], "rendering_mode": args.rendering_mode,
           "note": NOTE, "by_level": by_level, "phases": results,
           "gpu": os.popen("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader").read().strip(),
           "load": os.getloadavg()}
    name = "bench.json" if not (OUT / "bench.json").exists() else f"bench_{time.strftime('%H%M%S')}.json"
    (OUT / name).write_text(json.dumps(out, indent=2))
    log(f"BENCH written to {OUT / name}")


def selftest() -> None:
    """Top frames: find the magenta marker, map its pixel centroid to world x with the frame's extent and compare
    with the marker's position at the frame's t_sim. Checks the ortho mapping and that frames are not stale."""
    import io

    from PIL import Image

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame.top")
    sub.connect(f"tcp://127.0.0.1:{P['frames']}")
    cams.set_level("high")
    results = []
    run(2.0, pace=False, tag="selftest warmup ")
    while sub.poll(0):
        sub.recv_multipart()
    t_end = t_sim + 12.0
    while t_sim < t_end:
        run(0.5, pace=False, tag="selftest ")
        while sub.poll(0):
            topic, payload = sub.recv_multipart()
            m = msgpack.unpackb(payload, raw=False)
            im = np.asarray(Image.open(io.BytesIO(m["jpeg"])).convert("RGB")).astype(int)
            mask = (im[:, :, 0] > 170) & (im[:, :, 2] > 170) & (im[:, :, 1] < 90)
            if mask.sum() < 20:
                results.append({"t_sim": m["t_sim"], "found": False})
                continue
            vs, us = np.nonzero(mask)
            ext = m["cam_pose"]["extent"]
            x = ext[0] + (us.mean() + 0.5) / m["w"] * (ext[2] - ext[0])
            y = ext[3] - (vs.mean() + 0.5) / m["h"] * (ext[3] - ext[1])
            exp_x = -4.0 + (m["t_sim"] * 1.5) % 8.0
            if abs(exp_x - (-4.0)) < 0.5 or abs(exp_x - 4.0) < 0.5 or abs(exp_x - 0.5) < 0.45:
                continue  # near the wrap, or passing through the partition wall
            results.append({"t_sim": m["t_sim"], "found": True, "x": round(float(x), 3), "y": round(float(y), 3),
                            "exp_x": round(exp_x, 3), "exp_y": -3.4, "err_x": round(float(x - exp_x), 3),
                            "err_y": round(float(y + 3.4), 3), "px": int(mask.sum()), "forced": m.get("forced")})
    found = [r for r in results if r.get("found")]
    ex = np.array([r["err_x"] for r in found]) if found else np.array([np.nan])
    ey = np.array([r["err_y"] for r in found]) if found else np.array([np.nan])
    out = {"frames": len(results), "found": len(found), "err_x_mean_m": round(float(np.mean(ex)), 3),
           "err_x_absmax_m": round(float(np.max(np.abs(ex))), 3), "err_y_mean_m": round(float(np.mean(ey)), 3),
           "err_y_absmax_m": round(float(np.max(np.abs(ey))), 3),
           "implied_lag_ms": round(float(-np.mean(ex)) / 1.5 * 1000.0, 1),
           "marker_speed_mps": 1.5, "render_mode": cams.render_mode, "samples": results[:40]}
    out["pass"] = bool(found) and out["err_x_absmax_m"] < 0.2 and out["err_y_absmax_m"] < 0.2
    (OUT / "selftest.json").write_text(json.dumps(out, indent=2))
    log(f"SELFTEST {'PASS' if out['pass'] else 'FAIL'}: {len(found)}/{len(results)} frames, "
        f"err_x mean {out['err_x_mean_m']} max {out['err_x_absmax_m']} m, err_y max {out['err_y_absmax_m']} m, "
        f"implied lag {out['implied_lag_ms']} ms -> {OUT / 'selftest.json'}")


def settle_sweep() -> None:
    """Top snapshot quality vs `settle` (renders between re-arm and read), robot held still. Reference = the same
    view read after 40 settle renders. Metrics per settle: mean abs difference to the reference (0-255) and the
    Laplacian std (speckle). Writes settle_sweep.json + one JPEG per settle value."""
    import io

    from PIL import Image

    cams.set_level("low")
    top = cams._cams["top"]
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"frame.top")
    sub.connect(f"tcp://127.0.0.1:{P['frames']}")
    kin.mode, kin.v = "hold", 0.0
    run(3.0, pace=False, tag="sweep warmup ")

    def grab(settle: int) -> np.ndarray:
        while sub.poll(0):
            sub.recv_multipart()
        top.settle = settle
        top.next_due = t_sim  # due now
        t_end = t_sim + 5.0
        while t_sim < t_end:
            kin.idle_since = t_sim
            run(0.1, pace=False)
            if sub.poll(0):
                _, payload = sub.recv_multipart()
                m = msgpack.unpackb(payload, raw=False)
                return np.asarray(Image.open(io.BytesIO(m["jpeg"])).convert("RGB"))
        raise RuntimeError(f"no top frame for settle {settle}")

    grab(40)
    ref = grab(40).astype(np.float32)
    res = []
    for settle in (1, 2, 3, 4, 6, 8, 12):
        for rep_i in range(3):
            im = grab(settle)
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            r = {"settle": settle, "rep": rep_i, "mad_vs_ref": round(float(np.abs(im.astype(np.float32) - ref).mean()), 2),
                 "lap_std": round(float(cv2.Laplacian(g, cv2.CV_32F).std()), 2), "rearm_ms": round(top.rearm_ms, 1)}
            res.append(r)
            if rep_i == 0:
                Image.fromarray(im).save(OUT / f"settle_{settle:02d}.jpg", quality=92)
            log(f"SWEEP settle {settle:2d} rep {rep_i}: mad {r['mad_vs_ref']:6.2f}  lap_std {r['lap_std']:7.2f}")
    g = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = {"scene": scene["house_id"], "rendering_mode": args.rendering_mode, "top": [top.w, top.h],
           "ref_settle": 40, "ref_lap_std": round(float(cv2.Laplacian(g, cv2.CV_32F).std()), 2), "samples": res}
    Image.fromarray(ref.astype(np.uint8)).save(OUT / "settle_ref.jpg", quality=92)
    (OUT / "settle_sweep.json").write_text(json.dumps(out, indent=2))
    log(f"SWEEP written to {OUT / 'settle_sweep.json'} (ref lap_std {out['ref_lap_std']})")


READY = {"ports": P, "level": args.level, "scene": scene["house_id"], "robot_usd": usd_path, "note": NOTE,
         "rendering_mode": args.rendering_mode}
log("WL_VIZ_TEST_READY " + json.dumps(READY))
try:
    if args.bench:
        bench()
    elif args.selftest:
        selftest()
    elif args.settle_sweep:
        settle_sweep()
    else:
        r = run(args.duration, pace=args.rt_pace)
        log(f"done: {r}  viz {json.dumps(cams.stats())}")
finally:
    head_q.put(None)
    cams.close()
    sys.stdout.flush()
    os._exit(0)  # simulation_app.close() can hang for minutes after replicator use; nothing to save here
