"""VizCams: extra sim cameras for humans (chase, top-down, overview) -> JPEG -> ZMQ PUB 5602.

Runs INSIDE the Isaac process (P1, /work/envs/isaaclab). Attach it with two lines:

    from viz.isaac_cams import VizCams
    cams = VizCams(sim, "/World/Robot", house_bounds, pub="tcp://127.0.0.1:5602", hz=10, level="low")
    ...
    cams.step(t_sim, base_pos, base_quat_wxyz)     # every physics step (cheap when nothing is due)

What it renders (all prims live under /World/VizCams, nothing else on the stage is touched):
  chase     perspective, ~2.5 m behind / 1.6 m above the pelvis, looks at the pelvis, smoothed. When walls or
            furniture block the view (PhysX "fat ray", robot colliders ignored) it swings to the freest azimuth
            (behind, +-35, +-70, +-110 deg) and pulls in, staying high so it looks down over the robot
  top       orthographic, straight down over the whole house; the near clipping plane sits just under the
            ceiling (floor_z + ceiling_cut) so roofs/ceilings are cut away
  overview  optional, fixed perspective looking straight down over the house centre (same ceiling cut)

How it renders, and what it costs (Isaac Sim 5.1 / Replicator 1.12.27 APIs, measured on the box's L40S; the
experiments are in docs/viz.md "Render-product findings"):
  * One Replicator render product per camera: `omni.replicator.core.create.render_product(cam, (w, h), name=...)`
    (omni.replicator.core-1.12.27 scripts/create.py:1486) with the "rgb" annotator
    (`rep.AnnotatorRegistry.get_annotator("rgb")`, `.attach([rp.path])`, `.get_data()` -> uint8 HxWx4), the same
    pattern Isaac Lab's Camera sensor uses (isaaclab/sensors/camera/camera.py:439-488, 514). Rendering is an app
    update: Isaac Lab `SimulationContext.render()` -> `app.update()` with `/app/player/playSimulations=False`
    (isaaclab/sim/simulation_context.py:585-627). With Isaac Lab's kit settings (`app.asyncRendering=false`,
    `omni.replicator.asyncRendering=false`, IsaacLab/apps/isaaclab.python.headless.rendering.kit:79) the data read
    right after a render belongs to that render (measured: a marker moved before each render is seen at the new
    position, zero frames of lag).
  * An enabled render product is rendered by EVERY app update, including the host's own renders (P1 renders its
    head camera at --camera-hz, 30 Hz). Measured cost per render on the L40S: ~2.2 ms for 640x360, ~6 ms for
    1280x720, mostly independent of scene content (a blank view saves only ~0.7 ms).
  * `rp.hydra_texture.set_updates_enabled(False)` (HydraTexture, scripts/utils/viewport_manager.py:48-81) removes
    that cost, and is what NVIDIA's "capture at a custom FPS" snippet does
    (isaacsim.replicator.examples/tests/test_sdg_useful_snippets_timeline_based.py:84-86, 118-129). BUT that snippet
    re-arms the capture with `rep.orchestrator.step_async(...)`; driven from Isaac Lab's loop with plain
    `sim.render()`, a render product that was ever disabled never delivers annotator data again (measured, 6+
    renders), and `rep.orchestrator.step()` blocks indefinitely in this loop. `isaacsim.sensors.camera.Camera
    (frequency=...)`/`.pause()` only throttle data *acquisition* (isaacsim.sensors.camera/camera.py:439-481).
  * So there are two kinds of cameras here:
      live      (chase, overview): the render product stays enabled; a frame is READ at the viz rate right after a
                host render (piggyback), and VizCams calls `sim.render()` itself only if the host has not rendered
                within 1.5 viz periods (render="auto"; "own" always renders itself). Cost = per-render cost x the
                host's render rate, independent of the viz rate.
      snapshot  (top): the render product is disabled between captures (zero cost); a capture re-arms it with
                annotator detach -> enable -> attach (fresh data on the first render, ~16-20 ms of main-thread work
                at 512-768 px), keeps it enabled for `settle` renders (host renders when available) so the RTX
                preset's temporal DLSS/denoiser converges, reads it, then disables it again. `settle` 1 gives a
                speckled image under the "balanced" preset (docs/viz.md: settle sweep). Rate: every `every_s`
                (low 10 s, high 3 s). Between snapshots the UI/recorder draw the live robot pose from gt.pose over
                the last snapshot.
  * JPEG encoding and the ZMQ send run on a worker thread (cv2.imencode releases the GIL); the physics thread only
    copies the annotator buffer (~0.3 ms for 640x360).
  * The PUB socket is bound in the constructor (fails fast on a busy port) and then used only by the worker thread.

Wire format on the PUB socket (one message per frame), multipart:
    [b"frame.<cam>", msgpack({
        "topic": "frame.<cam>", "seq": int, "t_sim": float, "t_wall": float, "w": int, "h": int,
        "jpeg": bytes (RGB JPEG, standard channel order),
        "cam_pose": {"pos": [x,y,z], "quat_wxyz": [w,x,y,z], "projection": "perspective"|"orthographic",
                     "hfov_deg": float|None, "vfov_deg": float|None,
                     "extent": [xmin,ymin,xmax,ymax] | None,   # top/overview: world rect of the image,
                                                               # image +x = world +x, image up = world +y
                     "target": [x,y,z] | None},
        "robot": {"pos": [x,y,z], "yaw": float} | None,
        "level": "low"|"high", "render_ms": float, "v": 1})]

Viz levels: "off" (no render products at all, zero cost), "min", "low", "high" (see LEVELS). Switch at runtime with
`cams.set_level(...)` from the sim thread, or route the REP op through `cams.handle_op({"op": "viz_level", ...})`.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

LEVELS: dict[str, dict[str, dict[str, Any]]] = {
    "off": {},
    # chase "hz": None -> the constructor's `hz`; "every_s" = snapshot interval of the top camera
    # min: the cheapest way to SEE the robot walk (for RTF-critical SONIC runs); top comes from P1 render_topdown
    "min": {
        "chase": {"w": 480, "h": 270, "hz": None, "q": 80},
    },
    "low": {
        "chase": {"w": 640, "h": 360, "hz": None, "q": 80},
        "top": {"long": 512, "every_s": 10.0, "q": 85, "settle": 4},
    },
    "high": {
        "chase": {"w": 960, "h": 540, "hz": None, "q": 85},
        "top": {"long": 1024, "every_s": 3.0, "q": 85, "settle": 4},
        "overview": {"w": 960, "h": 540, "hz": 5.0, "q": 80},
    },
}
SNAPSHOT_CAMS = {"top"}


# ------------------------------------------------------------------------------------------------ math helpers
def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _look_at_rows(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 row-major USD matrix (row vectors: X, Y, Z axes, translation) for a camera at `eye` looking at
    `target`. USD cameras look down local -Z with +Y up."""
    f = target - eye
    n = np.linalg.norm(f)
    f = f / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    upv = np.asarray(up, dtype=float)
    r = np.cross(f, upv)
    if np.linalg.norm(r) < 1e-6:  # looking straight up/down: pick +x as image right
        r = np.array([1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, u, -f, eye
    return m


def _quat_wxyz_from_rows(m: np.ndarray) -> list[float]:
    # rows are the camera axes in world -> rotation matrix R has those as columns
    R = m[:3, :3].T
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [round(float(v), 6) for v in (w, x, y, z)]


def _even(v: float, mult: int = 8) -> int:
    return max(mult, int(round(v / mult)) * mult)


def parse_bounds(house_bounds: Any, floor_z: float = 0.0) -> tuple[float, float, float, float]:
    """Accepts [xmin,ymin,xmax,ymax] (P1 get_scene_info), ((xmin,ymin,zmin),(xmax,ymax,zmax)), a dict with
    'bounds', or an object with a .bounds attribute (scenes.loader HouseInfo)."""
    b = house_bounds
    if b is None:
        raise ValueError("house_bounds is required (xmin, ymin, xmax, ymax)")
    if hasattr(b, "bounds"):
        b = b.bounds
    if isinstance(b, dict):
        b = b.get("bounds", b)
    b = list(b)
    if len(b) == 2 and hasattr(b[0], "__len__"):
        (x0, y0, *_), (x1, y1, *_) = b
    elif len(b) == 4:
        x0, y0, x1, y1 = b
    elif len(b) == 6:
        x0, y0, _, x1, y1, _ = b
    else:
        raise ValueError(f"cannot parse house_bounds {house_bounds!r}")
    return float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1))


# ------------------------------------------------------------------------------------------------ one camera
@dataclass
class _Cam:
    name: str
    path: str
    w: int
    h: int
    hz: float
    q: int
    projection: str = "perspective"
    hfov_deg: float | None = None
    vfov_deg: float | None = None
    extent: list[float] | None = None
    rp: Any = None
    annot: Any = None
    xform_op: Any = None
    next_due: float = -1.0
    pending: bool = False
    req_update: int = -1
    req_t_sim: float = 0.0
    req_wall: float = 0.0
    pose_rows: np.ndarray | None = None
    target: list[float] | None = None
    seq: int = 0
    n_captured: int = 0
    n_forced: int = 0
    n_empty: int = 0
    snapshot: bool = False      # disabled between captures, re-armed per capture (see module docstring)
    settle: int = 1             # renders between request and read (snapshots: >1 lets DLSS/denoiser converge)
    rearm_ms: float = 0.0
    t_captures: list[float] = field(default_factory=list)


class VizCams:
    """See the module docstring. All methods must be called from the thread that steps the simulation."""

    def __init__(
        self,
        sim: Any = None,
        robot_prim_path: str = "/World/Robot",
        house_bounds: Any = None,
        pub: str | Any = "tcp://127.0.0.1:5602",
        hz: float = 10.0,
        level: str = "low",
        floor_z: float = 0.0,
        ceiling_cut: float = 2.0,
        chase_back: float = 2.5,
        chase_up: float = 1.6,
        chase_max_z: float | None = None,
        chase_hfov_deg: float = 70.0,
        chase_tau_yaw: float = 0.6,
        chase_tau_pos: float = 0.15,
        chase_avoid_walls: bool = True,
        overview: dict | None = None,
        overview_tilt_deg: float = 0.0,
        margin: float = 0.5,
        render: str = "auto",
        pose_fn: Callable[[], tuple[Sequence[float], Sequence[float]]] | None = None,
        root: str = "/World/VizCams",
        cams: dict[str, dict] | None = None,
        verbose: bool = True,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        if render not in ("auto", "own", "piggyback"):
            raise ValueError("render must be auto|own|piggyback")
        self.sim = sim
        self.robot_prim_path = robot_prim_path.rstrip("/")
        self.floor_z = float(floor_z)
        self.bounds = parse_bounds(house_bounds, floor_z)
        self.hz = float(hz)
        self.level = "off"
        self.ceiling_cut = float(ceiling_cut)
        self.chase_back, self.chase_up = float(chase_back), float(chase_up)
        self.chase_max_z = float(chase_max_z) if chase_max_z is not None else self.floor_z + self.ceiling_cut + 0.2
        self.chase_hfov_deg = float(chase_hfov_deg)
        self.chase_tau_yaw, self.chase_tau_pos = float(chase_tau_yaw), float(chase_tau_pos)
        self.chase_avoid_walls = chase_avoid_walls
        self.overview_cfg = overview
        self.overview_tilt_deg = float(overview_tilt_deg)
        self.margin = float(margin)
        self.render_mode = render
        self.pose_fn = pose_fn
        self.root = root.rstrip("/")
        self.cam_overrides = cams or {}
        self.verbose = verbose

        self._cams: dict[str, _Cam] = {}
        self._updates = 0
        self._t_last = None
        self._t_pose = None     # t_sim of the last pose sample (chase smoothing interval)
        self._chase_state: dict[str, Any] | None = None
        self._pose: tuple[np.ndarray, float] | None = None  # (pos, yaw) of the latest step
        self._stats = {"capture_ms": [], "forced_renders": 0, "dropped": 0, "encode_ms": [], "sent": 0,
                       "piggyback": 0, "step_calls": 0, "chase_ms": [], "chase_free": [], "rays": 0, "ray_hits": 0}
        self._lock = threading.Lock()

        import omni.kit.app  # noqa: WPS433  (Isaac-only import)
        import omni.usd

        self._app = omni.kit.app.get_app()
        self._stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom

        self._mpu = float(UsdGeom.GetStageMetersPerUnit(self._stage) or 1.0)
        self._upd_sub = self._app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="viz_cams.update_counter")

        # worker thread: JPEG encode + ZMQ PUB. The socket is bound here (fail fast on a busy port) and then used
        # only by the worker thread (a ZMQ socket may migrate between threads; thread start is a full barrier).
        import zmq

        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._pub_addr = pub if isinstance(pub, str) else "<socket>"
        if isinstance(pub, str):
            self._sock = zmq.Context.instance().socket(zmq.PUB)
            self._sock.setsockopt(zmq.SNDHWM, 16)
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.bind(pub)
            self._own_sock = True
        else:  # an already bound PUB socket handed in (must not be used by any other thread afterwards)
            self._sock, self._own_sock = pub, False
        self._worker = threading.Thread(target=self._encode_loop, name="viz_cams.encode", daemon=True)
        self._worker.start()

        if self.chase_avoid_walls and self._scene_queries_enabled() is False:
            print("[viz_cams] WARNING: PhysX scene queries are off (Isaac Lab SimulationCfg.enable_scene_query_support "
                  "defaults to False; sim_isaac/app.py sets True): chase wall avoidance disabled", flush=True)
            self.chase_avoid_walls = False
        self.set_level(level)

    # -------------------------------------------------------------------------------------------- public API
    def step(self, t_sim: float, base_pos: Sequence[float] | None = None,
             base_quat_wxyz: Sequence[float] | None = None) -> None:
        """Call every physics step (after `sim.step`). Returns at once unless a viz frame is due or pending.

        The pose is only read when a capture is requested (10 Hz for the chase): the chase smoothing is an exact
        first-order filter for any sample interval, and the camera is placed once per request. Re-placing it every
        physics step until the host renders (<= 1/30 s later, <= 2 cm of robot motion) cost a raycast + a USD write
        per step for no visible gain."""
        self._stats["step_calls"] += 1
        if not self._cams:
            return
        t_sim = float(t_sim)
        if self._t_last is not None and t_sim + 1e-6 < self._t_last:  # sim reset: re-arm schedules
            for c in self._cams.values():
                c.next_due, c.pending = -1.0, False
            self._chase_state = None
            self._t_pose = None
        self._t_last = t_sim

        due = [c for c in self._cams.values() if not c.pending and t_sim >= c.next_due]
        if not due and not any(c.pending for c in self._cams.values()):
            return  # the common case (199 of 200 physics steps at 200 Hz / 10 Hz viz)

        if due:
            if base_pos is None:
                got = self._pose_from_fallback()
                if got is not None:
                    base_pos, base_quat_wxyz = got
            if base_pos is not None:
                pos = np.asarray(base_pos, dtype=float)[:3]
                yaw = yaw_from_quat_wxyz(base_quat_wxyz) if base_quat_wxyz is not None else (
                    self._pose[1] if self._pose else 0.0)
                dt = 0.0 if self._t_pose is None else max(0.0, t_sim - self._t_pose)
                self._t_pose = t_sim
                self._pose = (pos, yaw)
                if "chase" in self._cams:
                    self._update_chase_state(pos, yaw, dt)

        # 1) harvest pending captures that have been rendered (settle times) since they were requested
        # (the host rendered after our previous step() call and before this one, i.e. at ~t_sim)
        for c in self._cams.values():
            if c.pending and self._updates >= c.req_update + c.settle:
                self._harvest(c, forced=False, t_render=t_sim)

        # 2) request captures that are due (places the chase camera / re-arms a snapshot)
        for c in due:
            if not c.pending:
                period = 1.0 / c.hz
                c.next_due = t_sim + period if c.next_due < 0 or t_sim - c.next_due > period else c.next_due + period
                self._request(c, t_sim)

        # 3) force a render if the host is not rendering (render="auto"), or always ("own")
        pend = [c for c in self._cams.values() if c.pending]
        if not pend:
            return
        force = self.render_mode == "own"
        if self.render_mode == "auto":
            force = any(t_sim - c.req_t_sim > min(1.5 / c.hz, 0.25) for c in pend)
        if force:
            t0 = time.perf_counter()
            n = max(1, max(c.req_update + c.settle - self._updates for c in pend))
            for _ in range(n):
                self._render_now()
            self._stats["forced_renders"] += n
            for c in pend:
                if self._updates >= c.req_update + c.settle:
                    self._harvest(c, forced=True, t_render=t_sim, render_ms=(time.perf_counter() - t0) * 1000.0)

    def set_level(self, level: str) -> None:
        """off | low | high. Recreates render products (call from the sim thread)."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        if level == self.level and self._cams:
            return
        self._destroy_cams()
        self.level = level
        spec = LEVELS[level]
        for name, cfg in spec.items():
            cfg = {**cfg, **self.cam_overrides.get(name, {})}
            if name == "overview" and self.overview_cfg is False:
                continue
            self._create_cam(name, cfg)
        if level == "low" and self.overview_cfg:  # explicit overview request on low
            self._create_cam("overview", {**LEVELS["high"]["overview"], **self.overview_cfg})
        if self.verbose:
            desc = ", ".join(f"{c.name} {c.w}x{c.h} " + (f"snapshot every {1 / c.hz:g}s (settle {c.settle})"
                                                          if c.snapshot else f"@{c.hz:g}Hz")
                             for c in self._cams.values()) or "none"
            print(f"[viz_cams] level={level}: {desc} -> {self._pub_addr} (render={self.render_mode})", flush=True)

    def handle_op(self, req: dict) -> dict:
        """For P1's REP: {"op": "viz_level", "level": ...} | {"op": "viz_stats"}."""
        op = req.get("op")
        args = {**req.get("args", {}), **{k: v for k, v in req.items() if k not in ("op", "args")}}
        if op == "viz_level":
            self.set_level(str(args.get("level", "low")))
            return {"ok": True, "level": self.level}
        if op == "viz_stats":
            return {"ok": True, **self.stats()}
        return {"ok": False, "error": f"unknown viz op {op!r}"}

    def stats(self) -> dict:
        with self._lock:
            cap = list(self._stats["capture_ms"][-500:])
            enc = list(self._stats["encode_ms"][-500:])
            chm = list(self._stats["chase_ms"][-500:])
            chf = list(self._stats["chase_free"][-500:])
        out = {
            "level": self.level, "render_mode": self.render_mode,
            "forced_renders": self._stats["forced_renders"], "piggyback_captures": self._stats["piggyback"],
            "dropped": self._stats["dropped"], "sent": self._stats["sent"], "app_updates": self._updates,
            "capture_ms_mean": round(float(np.mean(cap)), 3) if cap else None,
            "capture_ms_p99": round(float(np.percentile(cap, 99)), 3) if cap else None,
            "encode_ms_mean": round(float(np.mean(enc)), 3) if enc else None,
            # chase placement (raycasts included) and how often the chosen view was not fully free
            "chase_place_ms_mean": round(float(np.mean(chm)), 3) if chm else None,
            "chase_place_ms_p99": round(float(np.percentile(chm, 99)), 3) if chm else None,
            "chase_blocked_frac": round(float(np.mean(np.asarray(chf) < 0.999)), 3) if chf else None,
            "chase_avoid_walls": self.chase_avoid_walls, "rays": self._stats["rays"],
            "ray_hits": self._stats["ray_hits"],
            "cams": {},
        }
        for c in self._cams.values():
            ts = c.t_captures[-50:]
            rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else None
            out["cams"][c.name] = {"w": c.w, "h": c.h, "hz_target": round(c.hz, 3), "snapshot": c.snapshot,
                                   "settle": c.settle,
                                   "rearm_ms": round(c.rearm_ms, 1), "captured": c.n_captured,
                                   "forced": c.n_forced, "empty": c.n_empty,
                                   "hz_sim": round(rate, 2) if rate else None}
        return out

    def close(self) -> None:
        self._destroy_cams()
        self._upd_sub = None
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    # -------------------------------------------------------------------------------------------- internals
    def _on_update(self, _event: Any) -> None:
        self._updates += 1

    def _render_now(self) -> None:
        if self.sim is not None and hasattr(self.sim, "render"):
            self.sim.render()
        else:  # same as Isaac Lab SimulationContext.render(): update the app without stepping physics
            import carb

            s = carb.settings.get_settings()
            s.set("/app/player/playSimulations", False)
            self._app.update()
            s.set("/app/player/playSimulations", True)

    def _pose_from_fallback(self):
        if self.pose_fn is not None:
            try:
                return self.pose_fn()
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[viz_cams] pose_fn failed: {e}", flush=True)
                return None
        # USD read: only correct when physics writes back to USD (Fabric off). P1 should pass the pose.
        try:
            from pxr import Usd, UsdGeom

            prim = self._stage.GetPrimAtPath(self.robot_prim_path)
            for child in ("pelvis",):
                p2 = self._stage.GetPrimAtPath(f"{self.robot_prim_path}/{child}")
                if p2 and p2.IsValid():
                    prim = p2
                    break
            if not prim or not prim.IsValid():
                return None
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            q = m.ExtractRotationQuat()
            im = q.GetImaginary()
            return (t[0], t[1], t[2]), (q.GetReal(), im[0], im[1], im[2])
        except Exception:  # noqa: BLE001
            return None

    def _create_cam(self, name: str, cfg: dict) -> None:
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        x0, y0, x1, y1 = self.bounds
        m = self.margin
        bw, bh = (x1 - x0) + 2 * m, (y1 - y0) + 2 * m
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        snapshot = name in SNAPSHOT_CAMS and not cfg.get("live", False)
        if snapshot:
            hz = 1.0 / max(0.05, float(cfg.get("every_s", 10.0)))
        else:
            hz = float(cfg.get("hz") or self.hz)
            if name != "chase":
                hz = min(hz, self.hz)
        c = _Cam(name=name, path=f"{self.root}/{name}", w=0, h=0, hz=hz, q=int(cfg.get("q", 80)), snapshot=snapshot,
                 settle=max(1, int(cfg.get("settle", 4 if snapshot else 1))))

        cam = UsdGeom.Camera.Define(self._stage, c.path)
        prim = cam.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        c.xform_op = xf.AddTransformOp()
        s = 1.0 / self._mpu  # metres -> stage units

        if name == "top":
            long_px = int(cfg.get("long", 512))
            if bw >= bh:
                c.w, c.h = _even(long_px), _even(long_px * bh / bw)
            else:
                c.w, c.h = _even(long_px * bw / bh), _even(long_px)
            mpp = max(bw / c.w, bh / c.h)
            ww, hh = c.w * mpp, c.h * mpp
            c.projection = "orthographic"
            c.extent = [cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2]
            z_cam = self.floor_z + self.ceiling_cut + 30.0
            cam.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
            # USD apertures are in tenths of a scene unit (UsdGeomCamera docs; GfCamera::APERTURE_UNIT = 0.1)
            cam.CreateHorizontalApertureAttr().Set(float(ww * s * 10.0))
            cam.CreateVerticalApertureAttr().Set(float(hh * s * 10.0))
            cam.CreateFocalLengthAttr().Set(50.0)
            near = z_cam - (self.floor_z + self.ceiling_cut)
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(float(near * s), float((z_cam - self.floor_z + 5.0) * s)))
            rows = np.eye(4)
            rows[3, :3] = [cx, cy, z_cam]
            c.pose_rows = rows
            c.target = [cx, cy, self.floor_z]
        else:
            if name == "chase":
                c.w, c.h = int(cfg.get("w", 640)), int(cfg.get("h", 360))
                hfov = float(cfg.get("hfov_deg", self.chase_hfov_deg))
                near, far = 0.05, 60.0
            else:  # overview: straight down (optionally tilted) above the house centre
                c.w, c.h = int(cfg.get("w", 960)), int(cfg.get("h", 540))
                hfov = float(cfg.get("hfov_deg", 60.0))
            fl = 18.0
            hap = 2.0 * fl * math.tan(math.radians(hfov) / 2.0)
            vap = hap * c.h / c.w
            c.hfov_deg = round(hfov, 3)
            c.vfov_deg = round(math.degrees(2.0 * math.atan(vap / (2.0 * fl))), 3)
            cam.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
            cam.CreateFocalLengthAttr().Set(fl)
            cam.CreateHorizontalApertureAttr().Set(float(hap))
            cam.CreateVerticalApertureAttr().Set(float(vap))
            if name == "overview":
                ocfg = self.overview_cfg if isinstance(self.overview_cfg, dict) else {}
                if "eye" in ocfg and "target" in ocfg:
                    eye, tgt = np.asarray(ocfg["eye"], float), np.asarray(ocfg["target"], float)
                    near, far = float(ocfg.get("near", 0.1)), float(ocfg.get("far", 200.0))
                else:
                    # height so the whole (margined) house fits both FOVs
                    th = math.tan(math.radians(c.hfov_deg) / 2.0)
                    tv = math.tan(math.radians(c.vfov_deg) / 2.0)
                    hgt = max(bw / 2.0 / th, bh / 2.0 / tv) * 1.05
                    tilt = math.radians(self.overview_tilt_deg)
                    tgt = np.array([cx, cy, self.floor_z])
                    eye = tgt + np.array([0.0, -math.sin(tilt) * hgt, math.cos(tilt) * hgt])
                    near = max(0.1, eye[2] - (self.floor_z + self.ceiling_cut)) * math.cos(tilt)
                    far = hgt * 1.5 + 5.0
                c.pose_rows = _look_at_rows(eye, tgt, up=(0.0, 1.0, 0.0) if self.overview_tilt_deg < 1 else (0, 0, 1))
                c.target = [float(v) for v in tgt]
                if abs(self.overview_tilt_deg) < 1e-3:
                    # straight down: the image is an axis-aligned world rectangle at floor height
                    d = eye[2] - self.floor_z
                    ww, hh = 2 * d * math.tan(math.radians(c.hfov_deg) / 2), 2 * d * math.tan(math.radians(c.vfov_deg) / 2)
                    c.extent = [eye[0] - ww / 2, eye[1] - hh / 2, eye[0] + ww / 2, eye[1] + hh / 2]
            cam.CreateClippingRangeAttr().Set(Gf.Vec2f(float(near * s), float(far * s)))

        if c.pose_rows is not None:
            self._set_rows(c, c.pose_rows)
        rp = rep.create.render_product(c.path, (c.w, c.h), name=f"viz_{name}")
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp.path])
        if c.snapshot:  # zero render cost until the first capture re-arms it
            rp.hydra_texture.set_updates_enabled(False)
        c.rp, c.annot = rp, annot
        self._cams[name] = c

    def _destroy_cams(self) -> None:
        for c in list(self._cams.values()):
            try:
                c.annot.detach([c.rp.path])
            except Exception:  # noqa: BLE001
                pass
            try:
                c.rp.destroy()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._stage.RemovePrim(c.path)
            except Exception:  # noqa: BLE001
                pass
        self._cams.clear()

    def _set_rows(self, c: _Cam, rows: np.ndarray) -> None:
        from pxr import Gf

        r = rows.copy()
        r[3, :3] = r[3, :3] / self._mpu
        c.xform_op.Set(Gf.Matrix4d(*[float(v) for v in r.flatten()]))

    def _request(self, c: _Cam, t_sim: float) -> None:
        if c.name == "chase":
            self._apply_chase_pose(c)
        if c.snapshot:
            # a disabled render product only delivers annotator data again after the annotator is re-attached
            t0 = time.perf_counter()
            c.annot.detach([c.rp.path])
            c.rp.hydra_texture.set_updates_enabled(True)
            c.annot.attach([c.rp.path])
            c.rearm_ms = (time.perf_counter() - t0) * 1000.0
        c.pending = True
        c.req_update = self._updates
        c.req_t_sim = t_sim
        c.req_wall = time.time()

    def _harvest(self, c: _Cam, forced: bool, t_render: float, render_ms: float = 0.0) -> None:
        t0 = time.perf_counter()
        data = c.annot.get_data()
        if c.snapshot:
            c.rp.hydra_texture.set_updates_enabled(False)
        c.pending = False
        if data is None or getattr(data, "size", 0) == 0 or data.ndim != 3:
            c.n_empty += 1
            return
        rgb = np.ascontiguousarray(data[:, :, :3])
        c.seq += 1
        c.n_captured += 1
        c.n_forced += int(forced)
        if not forced:
            self._stats["piggyback"] += 1
        c.t_captures.append(t_render)
        if len(c.t_captures) > 200:
            del c.t_captures[:100]
        rows = c.pose_rows if c.pose_rows is not None else np.eye(4)
        robot = None
        if self._pose is not None:
            robot = {"pos": [round(float(v), 4) for v in self._pose[0]], "yaw": round(float(self._pose[1]), 4)}
        meta = {
            "topic": f"frame.{c.name}", "seq": c.seq, "t_sim": round(t_render, 4), "t_wall": time.time(),
            "t_request": round(c.req_t_sim, 4),
            "w": int(rgb.shape[1]), "h": int(rgb.shape[0]), "level": self.level,
            "render_ms": round(render_ms, 3), "forced": forced, "snapshot": c.snapshot,
            "rearm_ms": round(c.rearm_ms, 1) if c.snapshot else None, "settle": c.settle, "v": 1,
            "cam_pose": {
                "pos": [round(float(v), 4) for v in rows[3, :3]], "quat_wxyz": _quat_wxyz_from_rows(rows),
                "projection": c.projection, "hfov_deg": c.hfov_deg, "vfov_deg": c.vfov_deg,
                "extent": [round(float(v), 4) for v in c.extent] if c.extent else None,
                "target": c.target,
            },
            "robot": robot,
        }
        with self._lock:
            self._stats["capture_ms"].append((time.perf_counter() - t0) * 1000.0 + render_ms
                                             + (c.rearm_ms if c.snapshot else 0.0))
            if len(self._stats["capture_ms"]) > 2000:
                del self._stats["capture_ms"][:1000]
        try:
            self._q.put_nowait((rgb, meta, c.q))
        except queue.Full:
            self._stats["dropped"] += 1

    # chase camera --------------------------------------------------------------------------------------------
    def _update_chase_state(self, pos: np.ndarray, yaw: float, dt: float) -> None:
        st = self._chase_state
        if st is None or dt <= 0.0:
            if st is None:
                self._chase_state = {"target": pos.copy(), "yaw": yaw, "dist": self.chase_back}
            return
        a_pos = 1.0 - math.exp(-dt / max(1e-3, self.chase_tau_pos))
        a_yaw = 1.0 - math.exp(-dt / max(1e-3, self.chase_tau_yaw))
        st["target"] = st["target"] + a_pos * (pos - st["target"])
        st["yaw"] = _wrap(st["yaw"] + a_yaw * _wrap(yaw - st["yaw"]))

    # candidate camera azimuths relative to "straight behind" (deg), scanned when the view is blocked
    CHASE_OFFSETS_DEG = (0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0)

    def _view_free(self, tgt: np.ndarray, az: float) -> float:
        """Free fraction (0..1) of the sight line from the robot's chest to a camera at azimuth `az` (world yaw of
        robot -> camera) at the full chase distance. A "fat ray": three parallel rays 0.22 m apart, so a door frame
        or a fridge corner next to the centre line also counts."""
        o = tgt + np.array([0.0, 0.0, 0.3])
        eye = tgt + np.array([math.cos(az) * self.chase_back, math.sin(az) * self.chase_back, self.chase_up])
        eye[2] = min(eye[2], self.chase_max_z)
        v = eye - o
        side = np.array([-math.sin(az), math.cos(az), 0.0]) * 0.22
        free = 1.0
        for k in (0.0, 1.0, -1.0):
            h = self._ray_free_dist(o + k * side, v)
            if h is not None:
                free = min(free, h)
        return free

    def _apply_chase_pose(self, c: _Cam) -> None:
        """Place the chase camera. Default: straight behind the (smoothed) heading. With chase_avoid_walls, when the
        sight line is blocked (or every 5th frame, to drift back behind), scan CHASE_OFFSETS_DEG and move toward the
        freest azimuth (behind preferred, with hysteresis; offset smoothed), then pull the camera in to the free
        distance. The height stays at chase_up (clamped under the ceiling cut), so a pulled-in camera looks down over
        the robot instead of into a wall."""
        st = self._chase_state
        if st is None:
            return
        t0 = time.perf_counter()
        tgt = st["target"] + np.array([0.0, 0.0, 0.05])
        base_az = st["yaw"] + math.pi
        off = st.setdefault("off", 0.0)
        free = 1.0
        if self.chase_avoid_walls:
            st["n"] = st.get("n", 0) + 1
            goal = st.get("off_goal", 0.0)
            cur = self._view_free(tgt, base_az + goal)
            if cur < 0.9 or st["n"] % 5 == 0:
                def score(o_rad: float, fr: float) -> float:
                    return fr - 0.2 * abs(o_rad)          # 70 deg costs 0.24 of free fraction

                best, best_s = goal, score(goal, cur) + 0.1   # hysteresis for the current goal
                for deg in self.CHASE_OFFSETS_DEG:
                    o_rad = math.radians(deg)
                    if abs(o_rad - goal) < 1e-6:
                        continue
                    fr = self._view_free(tgt, base_az + o_rad)
                    if score(o_rad, fr) > best_s:
                        best, best_s = o_rad, score(o_rad, fr)
                st["off_goal"] = goal = best
            a = 1.0 - math.exp(-(1.0 / max(self.hz, 1e-3)) / 0.35)
            off = st["off"] = off + a * (goal - off)
            free = cur if abs(off - goal) < 1e-3 else self._view_free(tgt, base_az + off)
            if not self.chase_avoid_walls:  # raycasts unavailable (set by _ray_free_dist)
                off = st["off"] = 0.0
                free = 1.0
        az = base_az + off
        want = self.chase_back if free >= 1.0 else max(0.6, min(self.chase_back, free * self.chase_back - 0.25))
        # pull in fast, back out slowly
        st["dist"] = want if want < st["dist"] else st["dist"] + 0.15 * (want - st["dist"])
        eye = tgt + np.array([math.cos(az) * st["dist"], math.sin(az) * st["dist"], self.chase_up])
        eye[2] = min(eye[2], self.chase_max_z)
        rows = _look_at_rows(eye, tgt)
        c.pose_rows = rows
        c.target = [round(float(v), 4) for v in tgt]
        self._set_rows(c, rows)
        st["free"] = free
        with self._lock:
            self._stats["chase_ms"].append((time.perf_counter() - t0) * 1000.0)
            self._stats["chase_free"].append(free)
            for k in ("chase_ms", "chase_free"):
                if len(self._stats[k]) > 2000:
                    del self._stats[k][:1000]

    def _scene_queries_enabled(self) -> bool | None:
        """physxScene:enableSceneQuerySupport of the stage's PhysicsScene (None = unknown / not authored). Without
        it raycast_all silently reports no hits."""
        try:
            from pxr import PhysxSchema, UsdPhysics

            for prim in self._stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    attr = PhysxSchema.PhysxSceneAPI(prim).GetEnableSceneQuerySupportAttr()
                    return bool(attr.Get()) if attr and attr.HasAuthoredValue() else None
        except Exception:  # noqa: BLE001
            return None
        return None

    def _ray_free_dist(self, origin: np.ndarray, vec: np.ndarray) -> float | None:
        """Fraction (0..1) of `vec` that is free of non-robot colliders, None if free or unavailable."""
        try:
            from omni.physx import get_physx_scene_query_interface

            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                return None
            d = vec / dist
            best = [None]
            robot = self.robot_prim_path

            def report(hit) -> bool:
                path = str(getattr(hit, "collision", "") or getattr(hit, "rigid_body", ""))
                if path.startswith(robot) or path.startswith(self.root):
                    return True
                if best[0] is None or hit.distance < best[0]:
                    best[0] = float(hit.distance)
                return True

            s = 1.0 / self._mpu
            get_physx_scene_query_interface().raycast_all(
                tuple(float(v) * s for v in origin), tuple(float(v) for v in d), dist * s, report)
            self._stats["rays"] += 1
            if best[0] is None:
                return None
            self._stats["ray_hits"] += 1
            return best[0] / s / dist
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"[viz_cams] raycast failed, chase wall avoidance off: {e!r}", flush=True)
            self.chase_avoid_walls = False
            return None

    # encode thread -------------------------------------------------------------------------------------------
    def _encode_loop(self) -> None:
        import msgpack
        import zmq

        try:
            import cv2  # noqa: F401
            enc = "cv2"
        except Exception:  # noqa: BLE001
            enc = "pil"
        sock, own = self._sock, self._own_sock
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            rgb, meta, q = item
            t0 = time.perf_counter()
            try:
                if enc == "cv2":
                    import cv2

                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                           [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
                    jpeg = buf.tobytes()
                else:
                    import io

                    from PIL import Image

                    b = io.BytesIO()
                    Image.fromarray(rgb).save(b, format="JPEG", quality=int(q))
                    jpeg = b.getvalue()
                meta["jpeg"] = jpeg
                meta["encode_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                sock.send_multipart([meta["topic"].encode(), msgpack.packb(meta, use_bin_type=True)],
                                    flags=zmq.NOBLOCK)
                with self._lock:
                    self._stats["encode_ms"].append(meta["encode_ms"])
                    if len(self._stats["encode_ms"]) > 2000:
                        del self._stats["encode_ms"][:1000]
                self._stats["sent"] += 1
            except zmq.Again:
                self._stats["dropped"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[viz_cams] encode/send failed: {e}", flush=True)
        if own:
            sock.close(0)
