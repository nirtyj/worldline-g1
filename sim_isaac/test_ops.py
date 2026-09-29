"""Test-only P1 ops for the G1 stack scenarios (PLAN 9.3 G6/G7/G9; eval/stack_suite.py, docs/eval_hooks.md).

Registered only with `python -m sim_isaac.app ... --test-ops`: the demo stack never exposes them. Every op is a fault
injection for eval, logged as a gt.event `test_op`, and each one ends by itself (a push after its duration, a throttle
after duration_s, a spawned box after ttl_s), so an eval that dies mid-scenario cannot leave the sim degraded.

    push_robot {force_n=250, dir="left", duration_s=0.5}
        A world-frame force on the pelvis for duration_s of SIM time, applied every physics step like the band's
        wrench (base_sim.py xfrc_applied). dir: left | right | forward | back (the robot's frame at the push), a
        world [dx, dy], or a world heading in degrees. 250 N for 0.5 s is PLAN G7's "250 N lateral push -> fall".
    rtf_throttle {target=0.9, duration_s=120}      target 1.0 (or off: true) ends it
        Holds the sim below real time: every physics step's wall deadline moves dt * (1/target - 1) later
        (RtPacer's schedule anchor), so the pacer sleeps longer and RTF settles at `target` with no overruns.
        P1's own sim.health then reports degraded (< 0.95) / unsafe (< 0.85) as it would for a slow sim (P1.9).
    spawn_box {pose: {x, y, yaw}, size?, ttl_s=600}  /  clear_box
        One kinematic box (created at start-up with --test-box-size, parked far outside the house) is moved to the
        pose, standing on the floor: an obstacle the robot collides with but that is in no occupancy map, so a walk
        into it gets stuck (body/path_follower.py) and navigation ends `blocked`. clear_box parks it again.
        A `size` other than the start-up size is refused (bad_arg): resizing a collider at run time is a structural
        stage change (objects.py fixed_joint notes).
    test_ops_status
        What is active now: {push, throttle, box} (None when idle) and the box size.

The argument parsing and the math are pure functions (tested in sim_isaac/tests/test_test_ops.py without Isaac);
TestOps binds them to the running App (sim_isaac/app.py).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from sim_isaac.wire import OpError

OPS = ("push_robot", "rtf_throttle", "spawn_box", "clear_box", "test_ops_status")
MAX_FORCE_N = 1000.0
MAX_PUSH_S = 2.0
MIN_RTF, MAX_THROTTLE_S = 0.5, 900.0
MAX_BOX_TTL_S = 1800.0
PARK_OFFSET_M = 60.0                 # the parked box sits this far from the spawn (x and y), outside every house
ROBOT_DIRS = {"left": math.pi / 2, "right": -math.pi / 2, "forward": 0.0, "back": math.pi}


# ------------------------------------------------------------------------------------------------ pure helpers
def _num(req: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = req.get(key, default)
    try:
        f = float(default if v is None else v)
    except (TypeError, ValueError):
        raise OpError("bad_arg", f"{key} must be a number, got {v!r}") from None
    if not (lo <= f <= hi) or f != f:
        raise OpError("bad_arg", f"{key} must be in [{lo}, {hi}], got {f}")
    return f


def push_direction(direction: Any, yaw: float) -> np.ndarray:
    """Unit world-frame xy direction of a push: a robot-frame word, a world [dx, dy], or a world heading (deg)."""
    if isinstance(direction, str):
        d = direction.strip().lower()
        if d in ROBOT_DIRS:
            a = yaw + ROBOT_DIRS[d]
            return np.array([math.cos(a), math.sin(a)])
        try:
            direction = float(d)
        except ValueError:
            raise OpError("bad_arg", f"dir must be one of {sorted(ROBOT_DIRS)}, [dx, dy] or degrees; got {direction!r}") \
                from None
    if isinstance(direction, (int, float)):
        a = math.radians(float(direction))
        return np.array([math.cos(a), math.sin(a)])
    try:
        v = np.asarray(direction, dtype=np.float64).reshape(-1)[:2]
    except (TypeError, ValueError):
        raise OpError("bad_arg", f"dir: cannot read {direction!r}") from None
    n = float(np.linalg.norm(v)) if v.size == 2 else 0.0
    if n < 1e-9:
        raise OpError("bad_arg", f"dir must be a non-zero [dx, dy], got {direction!r}")
    return v / n


def parse_push(req: dict, yaw: float) -> dict:
    """push_robot's arguments -> {force_n, duration_s, dir, force_w: [fx, fy, 0]} (world frame, newtons)."""
    f = _num(req, "force_n", 250.0, 0.0, MAX_FORCE_N)
    dur = _num(req, "duration_s", 0.5, 0.005, MAX_PUSH_S)
    raw = req.get("dir", "left")
    u = push_direction(raw, yaw)
    return {"force_n": f, "duration_s": dur, "dir": raw, "force_w": [float(f * u[0]), float(f * u[1]), 0.0],
            "impulse_ns": round(f * dur, 2)}


def throttle_extra_s(dt: float, target: float) -> float:
    """Extra wall time per physics step that holds the RTF at `target` (1.0 -> 0)."""
    return 0.0 if target >= 1.0 else dt * (1.0 / target - 1.0)


def parse_throttle(req: dict) -> dict | None:
    """rtf_throttle's arguments -> {target, duration_s}, or None to end the throttle (target >= 1 or off)."""
    if req.get("off") or (req.get("target") is not None and float(req["target"]) >= 1.0):
        return None
    return {"target": _num(req, "target", 0.9, MIN_RTF, 0.999),
            "duration_s": _num(req, "duration_s", 120.0, 1.0, MAX_THROTTLE_S)}


def parse_size(size: Any) -> tuple[float, float, float]:
    """'0.3,1.6,1.2' | [sx, sy, sz] -> (sx, sy, sz) in metres (each 0.05-3 m)."""
    vals = [v for v in str(size).split(",")] if isinstance(size, str) else list(size or [])
    try:
        out = tuple(float(v) for v in vals)
    except (TypeError, ValueError):
        raise OpError("bad_arg", f"size must be three numbers, got {size!r}") from None
    if len(out) != 3 or not all(0.05 <= v <= 3.0 for v in out):
        raise OpError("bad_arg", f"size must be three sides of 0.05-3 m, got {size!r}")
    return out  # type: ignore[return-value]


def parse_box(req: dict, box_size: tuple[float, float, float]) -> dict:
    """spawn_box's arguments -> {x, y, yaw, ttl_s} (world frame; the box stands on the floor)."""
    pose = req.get("pose") if isinstance(req.get("pose"), dict) else req
    try:
        x, y = float(pose["x"]), float(pose["y"])
        yaw = float(pose.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError):
        raise OpError("bad_arg", "spawn_box needs pose {x, y, yaw?} (world metres, radians)") from None
    if req.get("size") is not None:
        want = parse_size(req["size"])
        if any(abs(a - b) > 1e-3 for a, b in zip(want, box_size)):
            raise OpError("bad_arg", f"the test box is {list(box_size)} m (P1 --test-box-size); got {list(want)}")
    return {"x": x, "y": y, "yaw": yaw, "ttl_s": _num(req, "ttl_s", 600.0, 1.0, MAX_BOX_TTL_S)}


def box_root_pose(x: float, y: float, yaw: float, floor_z: float, size: tuple[float, float, float]) -> list[float]:
    """[x, y, z, qw, qx, qy, qz] of the box standing on the floor, rotated by yaw about z."""
    return [x, y, floor_z + size[2] / 2.0, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


# ------------------------------------------------------------------------------------------------ bound to P1
class TestOps:
    """The test ops on a running App (sim_isaac/app.py). spawn_prims runs before sim.reset(); pre_step after the band
    wrench every physics step; before_wait right before the pacer's sleep."""

    __test__ = False                  # not a pytest class

    def __init__(self, app: Any, box_size: Any = (0.3, 1.6, 1.2), log: Callable[[str], None] = print):
        self.app = app
        self.log = log
        self.box_size = parse_size(box_size)
        self.box_obj = None
        self.park: list[float] | None = None
        self.push: dict | None = None
        self.throttle: dict | None = None
        self.box: dict | None = None
        self.counts = {op: 0 for op in OPS}
        self._fbuf = self._tbuf = None

    # -- set-up -----------------------------------------------------------------------------------
    def spawn_prims(self, sim_utils: Any, spawn_xyz: tuple[float, float, float]) -> None:
        """The kinematic test box (Isaac Lab RigidObject), parked outside the house. Before sim.reset()."""
        from isaaclab.assets import RigidObject, RigidObjectCfg
        sx, sy, floor_z = spawn_xyz
        self.park = box_root_pose(sx + PARK_OFFSET_M, sy + PARK_OFFSET_M, 0.0, floor_z, self.box_size)
        cfg = RigidObjectCfg(
            prim_path="/World/WlTestBox",
            spawn=sim_utils.CuboidCfg(
                size=self.box_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self.park[:3]), rot=tuple(self.park[3:])))
        self.box_obj = RigidObject(cfg)
        self.log(f"[test_ops] test box {list(self.box_size)} m parked at {[round(v, 2) for v in self.park[:3]]}")

    def register(self, gt: Any) -> None:
        for op in OPS:
            gt.register(op, getattr(self, f"op_{op}"))
        self.log(f"[test_ops] TEST-ONLY P1 ops registered: {', '.join(OPS)}")

    # -- per step ---------------------------------------------------------------------------------
    def pre_step(self, st: dict) -> None:
        p = self.push
        if p is None:
            return
        app = self.app
        if app.pacer.t_sim >= p["until_t_sim"]:
            self.push = None
            self._event("push_robot", state="ended", t_sim_start=p["t_sim_start"])
            return
        torch = app.torch
        if self._fbuf is None:
            self._fbuf = torch.zeros_like(app.force_buf)
            self._tbuf = torch.zeros_like(app.torque_buf)
        self._fbuf[app.pelvis_id] = torch.tensor(p["force_w"], dtype=torch.float32, device=self._fbuf.device)
        app.view.apply_forces_and_torques_at_position(
            force_data=self._fbuf, torque_data=self._tbuf, position_data=None, indices=app.view_idx, is_global=True)

    def before_wait(self) -> None:
        th = self.throttle
        if th is not None:
            if time.perf_counter() >= th["until_wall"]:
                self.throttle = None
                self._event("rtf_throttle", state="ended", target=th["target"])
            else:
                self.app.pacer.sched_anchor_wall += th["extra_s"]
        b = self.box
        if b is not None and time.perf_counter() >= b["until_wall"]:
            self._park("ttl")

    # -- ops --------------------------------------------------------------------------------------
    def op_push_robot(self, req: dict) -> dict:
        app = self.app
        lp = getattr(app, "last_pose", None) or {}
        yaw = float(lp["yaw"]) if "yaw" in lp else float(app._read_state()["yaw"])
        p = parse_push(req, yaw)
        t0 = app.pacer.t_sim
        p.update(t_sim_start=round(t0, 4), until_t_sim=t0 + p["duration_s"], robot_yaw=round(yaw, 4))
        self.push = p
        self.counts["push_robot"] += 1
        self._event("push_robot", state="started", **{k: v for k, v in p.items() if k != "until_t_sim"})
        return {"push": {k: v for k, v in p.items()}, "label": "test-only"}

    def op_rtf_throttle(self, req: dict) -> dict:
        th = parse_throttle(req)
        self.counts["rtf_throttle"] += 1
        if th is None:
            was, self.throttle = self.throttle, None
            self._event("rtf_throttle", state="ended", by="request", was=None if was is None else was["target"])
            return {"throttle": None, "was": None if was is None else was["target"], "label": "test-only"}
        th["extra_s"] = throttle_extra_s(self.app.dt, th["target"])
        th["until_wall"] = time.perf_counter() + th["duration_s"]
        self.throttle = th
        self._event("rtf_throttle", state="started", target=th["target"], duration_s=th["duration_s"],
                    extra_ms_per_step=round(th["extra_s"] * 1e3, 4))
        return {"throttle": {k: v for k, v in th.items() if k != "until_wall"}, "label": "test-only"}

    def _write_box(self, pose7: list[float]) -> None:
        torch = self.app.torch
        self.box_obj.write_root_pose_to_sim(torch.tensor([pose7], dtype=torch.float32, device=self.app.sim.device))

    def op_spawn_box(self, req: dict) -> dict:
        if self.box_obj is None:
            raise OpError("unavailable", "no test box (P1 started without --test-ops)")
        b = parse_box(req, self.box_size)
        pose7 = box_root_pose(b["x"], b["y"], b["yaw"], self.app.floor_z, self.box_size)
        self._write_box(pose7)
        b.update(pose=[round(v, 4) for v in pose7], size=list(self.box_size),
                 until_wall=time.perf_counter() + b["ttl_s"], t_sim=round(self.app.pacer.t_sim, 3))
        self.box = b
        self.counts["spawn_box"] += 1
        self._event("spawn_box", x=b["x"], y=b["y"], yaw=b["yaw"], size=list(self.box_size), ttl_s=b["ttl_s"])
        return {"box": {k: v for k, v in b.items() if k != "until_wall"}, "label": "test-only"}

    def _park(self, by: str) -> dict | None:
        was, self.box = self.box, None
        if self.box_obj is not None and self.park is not None and was is not None:
            self._write_box(self.park)
            self._event("clear_box", by=by, x=was["x"], y=was["y"])
        return was

    def op_clear_box(self, req: dict) -> dict:
        self.counts["clear_box"] += 1
        was = self._park("request")
        return {"cleared": was is not None, "label": "test-only"}

    def op_test_ops_status(self, req: dict) -> dict:
        return status_reply(self.push, self.throttle, self.box, self.box_size, self.counts,
                            t_sim=self.app.pacer.t_sim, rtf_1s=self.app.pacer.rtf(1.0))

    def _event(self, op: str, **kw) -> None:
        self.app._event("test_op", op=op, **kw)


def status_reply(push: dict | None, throttle: dict | None, box: dict | None, box_size, counts: dict,
                 **extra) -> dict:
    """test_ops_status: what is active now; `active` is False only when nothing is."""
    now = time.perf_counter()
    th = None if throttle is None else {"target": throttle["target"],
                                        "left_s": round(max(0.0, throttle["until_wall"] - now), 1)}
    bx = None if box is None else {"x": box["x"], "y": box["y"], "yaw": box["yaw"],
                                   "left_s": round(max(0.0, box["until_wall"] - now), 1)}
    pu = None if push is None else {"force_w": push["force_w"], "until_t_sim": round(push["until_t_sim"], 3)}
    return {"active": bool(pu or th or bx), "push": pu, "throttle": th, "box": bx, "box_size": list(box_size),
            "counts": dict(counts), "label": "test-only", **{k: v for k, v in extra.items()}}
