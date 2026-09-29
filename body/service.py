"""wl-body service (P3): ROUTER 5610 (commands) / PUB 5611 (events + state), 50 Hz control loop.

Request  (DEALER or REQ -> ROUTER 5610, one JSON frame):
    {"id": "<client id, optional>", "op": "stand|walk|go_to|turn_to|velocity|arm|stop|status|ping|reload_map|
                                          clear_fault|shutdown_control", "args": {...}}
Reply    (same envelope back): {"id", "ok": bool, "state": "accepted|rejected|failed|done", "error"?: str, "data"?: {}}
Events   (PUB 5611, multipart [b"body.event", JSON]):
    {"id", "op", "state": "accepted|progress|succeeded|failed|canceled", "data": {...}, "seq", "t_wall"}
State    (PUB 5611, multipart [b"body.state", JSON]) at 5 Hz: control/deploy/P1 health, active op, pose, frame.

Rules: one active motion at a time; a new motion op pre-empts the current one (it ends "canceled" with
reason "preempted"); stop cancels (reason "stop") and runs a StopMotion (planner IDLE until |v| < 0.05).
Watchdogs: gt.pose stale > pose_stale_s, g1_debug stale > debug_stale_s, op timeout, fall (gt fallen or pelvis_z below
band) -> the op fails, the mux holds IDLE, and after a fall the service latches fault="fallen" (clear_fault after a
P1 reset). The mux itself holds IDLE if this loop stalls for > cmd_stale_s. A user stop never sends command{stop}.
The arm channel (op `arm`, body/arm.py) is separate from the one active motion: it overlays upper-body and hand
targets on every planner message, so it composes with any leg motion; `stop {arms: true}` also ends it.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import uuid

import zmq

from .arm import ArmChannel, ArmError
from .config import BodyConfig, ep
from .deploy_monitor import DeployMonitor
from .frames import PlannerFrame
from .motions import MOTIONS, Motion, MotionError, StopMotion
from .nav_grid import NavGrid
from .p1_client import P1Error, P1Rpc, PoseSub
from .nav2_backend import Nav2Link, select_motion
from .path_follower import GoToMotion
from .sonic_mux import SonicMux
from .velocity import VelocityMotion, route_velocity
from .wire import LocomotionMode, dumps_json, wrap

ALL_MOTIONS = {**MOTIONS, GoToMotion.op: GoToMotion, VelocityMotion.op: VelocityMotion}
TERMINAL = ("succeeded", "failed", "canceled")


class MotionCtx:
    def __init__(self, svc: "BodyService"):
        self.svc = svc
        self.cfg = svc.cfg
        self.mux = svc.mux
        self.frame = svc.frame
        self.deploy = svc.deploy
        self.pose_sub = svc.pose_sub

    def velocity(self):
        return self.pose_sub.velocity()

    def emit(self, op_id: str, state: str, data: dict) -> None:
        self.svc.emit(op_id, state, data)

    def p1_call(self, op: str, **args):
        return self.svc.p1.try_call(op, **args)

    def get_nav(self) -> NavGrid | None:
        return self.svc.get_nav()

    def nav2_link(self) -> Nav2Link:
        return self.svc.nav2_link()


class BodyService:
    def __init__(self, cfg: BodyConfig, log_dir: str | None = None, log=None):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._logf = open(os.path.join(log_dir, "service.log"), "a", buffering=1) if log_dir else None
        self._evf = open(os.path.join(log_dir, "events.jsonl"), "a", buffering=1) if log_dir else None
        self._ext_log = log
        self.frame = PlannerFrame(cfg.heading_bias_ki, cfg.heading_bias_max_deg)
        P = cfg.ports
        self.mux = SonicMux(ep(P["sonic_in"], cfg.host), self.frame, cfg.keepalive_hz, cfg.cmd_stale_s,
                            cfg.dir_deadband_deg, cfg.speed_deadband, log=self.log,
                            cmd_log_path=os.path.join(log_dir, "planner_cmds.jsonl") if log_dir else None,
                            ctx=self.ctx)
        self.p1 = P1Rpc(ep(P["p1_rep"], cfg.host), timeout_s=5.0, ctx=self.ctx)
        self.pose_sub = PoseSub(ep(P["p1_pose"], cfg.host), ctx=self.ctx)
        self.deploy = DeployMonitor(ep(P["sonic_debug"], cfg.host), ctx=self.ctx, on_debug=self._on_debug)
        self.mctx = MotionCtx(self)
        self.arm = ArmChannel(cfg, self.mux, self.deploy, self.emit, log=self.log, record=self._record)
        self.active: Motion | None = None
        self.ops: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.fault: str | None = None
        self.nav: NavGrid | None = None
        self.nav_info: dict | None = None
        self._nav2: Nav2Link | None = None   # nav2/ros_bridge.py REP (go_to backend nav2), created lazily
        self._seq = itertools.count(1)
        self._running = False
        self._last_progress = 0.0
        self._facing_same_since: float | None = None
        self._facing_last: float | None = None
        self.t_start = time.monotonic()
        self.tick_stats = {"ticks": 0, "overruns": 0, "max_tick_ms": 0.0}

    # -- logging / events ------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self._logf:
            self._logf.write(line + "\n")
        if self._ext_log:
            self._ext_log(line)

    def emit(self, op_id: str, state: str, data: dict | None = None) -> None:
        rec = self.ops.get(op_id)
        ev = {"id": op_id, "op": rec["op"] if rec else None, "state": state, "data": data or {},
              "seq": next(self._seq), "t_wall": time.time()}
        if rec is not None:
            rec["events"] += 1
            if state in TERMINAL or state == "accepted":
                rec["state"] = state
            if state in TERMINAL:
                rec["t_end"] = time.time()
                rec["result"] = data
        try:
            self.evt.send_multipart([b"body.event", dumps_json(ev)])
        except Exception as e:
            self.log(f"[evt] send failed: {e!r}")
        if self._evf:
            self._evf.write(dumps_json(ev).decode() + "\n")
        if state != "progress":
            self.log(f"[op {op_id} {ev['op']}] {state} {json.dumps(_short(data))}")

    # -- callbacks -------------------------------------------------------------------------------
    def _on_debug(self, d: dict) -> None:
        if self.frame.update_from_debug(d.get("init_base_quat"), d.get("delta_heading")):
            self.log(f"[frame] planner frame from g1_debug: {self.frame.to_dict()}")

    # -- nav ------------------------------------------------------------------------------------
    def nav2_link(self) -> Nav2Link:
        if self._nav2 is None:
            self._nav2 = Nav2Link(ep(self.cfg.ports["nav_bridge"], self.cfg.host), ctx=self.ctx)
        return self._nav2

    def get_nav(self, reload: bool = False) -> NavGrid | None:
        if self.nav is not None and not reload:
            return self.nav
        try:
            # docs/contracts/m1.md §1.6: npz `occ` is raw (not inflated); the body inflates it itself
            rep = self.p1.call("get_occupancy", timeout_s=60.0, robot_radius=self.cfg.robot_radius)
        except P1Error as e:
            self.log(f"[nav] get_occupancy failed: {e}")
            return self.nav
        if rep.get("ok") is False:
            self.log(f"[nav] get_occupancy error: {rep}")
            return self.nav
        try:
            self.nav = NavGrid.from_p1_reply(rep, robot_radius=self.cfg.robot_radius, plan_res=self.cfg.plan_res)
            self.nav_info = {"shape": [self.nav.H, self.nav.W], "res": self.nav.res, "origin": self.nav.origin,
                             "robot_radius": self.nav.robot_radius, "p1_inflation": self.nav.already_inflated,
                             "free_frac": round(float((~self.nav.inflated).mean()), 3),
                             "path": rep.get("path") or rep.get("npz_path")}
            self.log(f"[nav] occupancy loaded {self.nav_info}")
        except Exception as e:
            self.log(f"[nav] could not load occupancy: {e!r}")
        return self.nav

    # -- request handling ------------------------------------------------------------------------
    def _handle(self, frames: list[bytes]) -> None:
        ident = frames[0]
        envelope = [ident, b""] if len(frames) >= 3 and frames[1] == b"" else [ident]
        try:
            req = json.loads(frames[-1].decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
        except Exception as e:
            self.ctl.send_multipart(envelope + [dumps_json({"ok": False, "state": "rejected",
                                                            "error": f"bad_request: {e}"})])
            return
        op_id = str(req.get("id") or uuid.uuid4().hex[:12])
        op = str(req.get("op", ""))
        args = req.get("args") or {}
        try:
            rep = self._dispatch(op_id, op, args)
        except Exception as e:
            self.log(f"[ctl] op {op} crashed: {traceback.format_exc()}")
            rep = {"ok": False, "state": "failed", "error": f"internal: {e!r}"}
        rep["id"] = op_id
        self.ctl.send_multipart(envelope + [dumps_json(rep)])

    def _dispatch(self, op_id: str, op: str, args: dict) -> dict:
        if op == "ping":
            return {"ok": True, "state": "done", "data": {"t_wall": time.time()}}
        if op == "status":
            data = self.status()
            if args.get("id"):
                data["op"] = self.ops.get(str(args["id"]))
            return {"ok": True, "state": "done", "data": data}
        if op == "reload_map":
            nav = self.get_nav(reload=True)
            return {"ok": nav is not None, "state": "done", "data": {"nav": self.nav_info}}
        if op == "clear_fault":
            prev, self.fault = self.fault, None
            return {"ok": True, "state": "done", "data": {"cleared": prev}}
        if op == "shutdown_control":
            if not args.get("confirm"):
                return {"ok": False, "state": "rejected", "error": "shutdown_control needs args.confirm=true "
                        "(sends command{stop}: the deploy damps and exits; put P1's band on first)"}
            self._cancel_active("shutdown")
            self.mux.shutdown_control()
            return {"ok": True, "state": "done", "data": {"mux": self.mux.snapshot()}}
        if op == "arm":
            try:
                return self.arm.handle(op_id, args, self._arm_can_start())
            except ArmError as e:
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
        if op == "stop" and args.get("arms"):
            self.arm.end("stop")
        if op == "velocity":
            rep = route_velocity(self, args)   # stream updates (and Nav2 /cmd_vel) never start a new op
            if rep is not None:
                return rep
        if op not in ALL_MOTIONS:
            return {"ok": False, "state": "rejected", "error": f"unknown op {op!r}"}
        return self._start_motion(op_id, op, args)

    def _arm_can_start(self) -> tuple[bool, str | None, dict]:
        if self.fault:
            return False, f"fault:{self.fault}", {"hint": "reset the robot in P1, then clear_fault"}
        if not (self.mux.control_started and self.deploy.in_control()):
            return False, "not_standing", {"hint": "call stand first", "deploy": self.deploy.snapshot()}
        return True, None, {}

    def _record(self, op_id: str, op: str, args: dict) -> None:
        self.ops[op_id] = {"id": op_id, "op": op, "args": args, "state": "received", "t_start": time.time(),
                           "t_end": None, "result": None, "events": 0}
        while len(self.ops) > 300:
            self.ops.popitem(last=False)

    def _reject(self, op_id: str, reason: str, data: dict | None = None) -> dict:
        self.emit(op_id, "failed", {"reason": reason, **(data or {})})
        return {"ok": False, "state": "rejected", "error": reason, "data": data or {}}

    def _start_motion(self, op_id: str, op: str, args: dict) -> dict:
        self._record(op_id, op, args)
        cls = ALL_MOTIONS[op]
        pose = self.pose_sub.latest()
        if pose is None or self.pose_sub.age_s() > self.cfg.pose_stale_s:
            return self._reject(op_id, "no_pose", {"gt_pose_age_s": _r(self.pose_sub.age_s())})
        if self.fault and op not in ("stop",):
            return self._reject(op_id, f"fault:{self.fault}", {"hint": "reset the robot in P1, then clear_fault"})
        if cls.needs_control and not (self.mux.control_started and self.deploy.in_control()):
            return self._reject(op_id, "not_standing", {"hint": "call stand first",
                                                        "deploy": self.deploy.snapshot()})
        if cls.needs_control and not self.frame.known:
            return self._reject(op_id, "planner_frame_unknown")
        cls, backend_info = select_motion(self, op, args, cls)   # go_to: NAV_BACKEND nav2 | astar
        if self.active is not None:
            self._cancel_active("stop" if op == "stop" else "preempted", by=op_id)
        m = cls(op_id, args, self.mctx)
        if backend_info:
            m.backend_info = backend_info
        try:
            data = m.start(pose)
        except MotionError as e:
            self.mux.hold(pose.yaw, owner=op_id)
            self.emit(op_id, "failed", {"reason": e.reason, **e.data})
            return {"ok": False, "state": "failed", "error": e.reason, "data": _short(e.data)}
        self.active = m
        self._last_progress = time.monotonic()
        self.emit(op_id, "accepted", {"args": args, **getattr(m, "backend_info", {}), **(data or {})})
        return {"ok": True, "state": "accepted", "data": _short(data or {})}

    def _cancel_active(self, reason: str, by: str | None = None) -> None:
        m = self.active
        if m is None:
            return
        self.active = None
        pose = self.pose_sub.latest()
        try:
            data = m.on_cancel(pose, reason)
        except Exception as e:
            data = {"reason": reason, "on_cancel_error": repr(e)}
        if by:
            data["by"] = by
        data["reason"] = reason
        self.mux.hold(pose.yaw if pose is not None else self.mux.last_facing_w, owner=f"cancel:{m.id}")
        self.emit(m.id, "canceled", data)

    def _finish(self, state: str, data: dict) -> None:
        m = self.active
        self.active = None
        # hold at the facing the motion last commanded (not the current yaw) so a settled robot stays put
        cmd, _ = self.mux.current()
        facing = cmd.facing_w if (cmd is not None and cmd.facing_w is not None) else self.mux.last_facing_w
        if state == "failed":
            p = self.pose_sub.latest()
            facing = p.yaw if p is not None else facing
        self.mux.hold(facing, owner=f"done:{m.id}")
        self.emit(m.id, state, data)

    # -- control tick ----------------------------------------------------------------------------
    def _tick(self, now: float) -> None:
        pose = self.pose_sub.latest()
        age = self.pose_sub.age_s()
        m = self.active
        if pose is not None and age < self.cfg.pose_stale_s:
            fallen = pose.fallen or (self.mux.control_started and pose.pelvis_z < self.cfg.pelvis_z_min)
            if fallen and self.fault != "fallen":
                self.fault = "fallen"
                self.log(f"[watchdog] FALL detected: {pose.brief()}")
                if m is not None:
                    self._finish("failed", {"reason": "fallen", "pose": pose.brief(), **_safe(m, pose)})
                    m = None
                self.mux.hold(pose.yaw, owner="fall")
                self.arm.abort("fallen")
        try:
            self.arm.tick(now)
        except Exception:
            self.log(f"[arm] tick crashed: {traceback.format_exc()}")
            self.arm.abort("internal")
        if m is None:
            self._learn_bias(pose, age)
            return
        if pose is None or age > self.cfg.pose_stale_s:
            self._finish("failed", {"reason": "pose_stale", "gt_pose_age_s": _r(age)})
            return
        if m.needs_control and self.deploy.age_s() > self.cfg.debug_stale_s:
            self._finish("failed", {"reason": "deploy_stale", "g1_debug_age_s": _r(self.deploy.age_s()),
                                    **_safe(m, pose)})
            return
        if m.elapsed() > m.timeout_s:
            extra = m.on_timeout(pose) if hasattr(m, "on_timeout") else {}
            self._finish("failed", {"reason": "timeout", "timeout_s": m.timeout_s, **m._summary(pose), **extra})
            return
        try:
            res = m.tick(pose, now)
        except MotionError as e:
            self._finish("failed", {"reason": e.reason, **e.data})
            return
        except Exception as e:
            self.log(f"[tick] {m.op} crashed: {traceback.format_exc()}")
            self._finish("failed", {"reason": f"internal: {e!r}"})
            return
        if res is not None:
            self._finish(res[0], res[1])
            return
        if now - self._last_progress >= m.progress_period:
            self._last_progress = now
            self.emit(m.id, "progress", m.progress(pose))
        self._learn_bias(pose, age)

    def _learn_bias(self, pose, age) -> None:
        """Outer heading loop: while IDLE with an unchanged facing for > 1.5 s and settled, integrate the error."""
        if pose is None or age > self.cfg.pose_stale_s or not self.frame.known or not self.deploy.in_control():
            self._facing_same_since = None
            return
        cmd, _ = self.mux.current()
        if cmd is None or cmd.mode != LocomotionMode.IDLE or cmd.facing_w is None:
            self._facing_same_since = None
            return
        now = time.monotonic()
        if self._facing_last is None or abs(wrap(cmd.facing_w - self._facing_last)) > 1e-6:
            self._facing_last = cmd.facing_w
            self._facing_same_since = now
            return
        if self._facing_same_since is None:
            self._facing_same_since = now
            return
        vx, vy, wz = self.pose_sub.velocity()
        if now - self._facing_same_since > 1.5 and math.hypot(vx, vy) < 0.05 and abs(wz) < 0.1:
            self.frame.observe_settled(cmd.facing_w, pose.yaw, 1.0 / self.cfg.control_hz)

    # -- status ----------------------------------------------------------------------------------
    def status(self) -> dict:
        pose = self.pose_sub.latest()
        m = self.active
        return {
            "t_wall": time.time(), "uptime_s": round(time.monotonic() - self.t_start, 1),
            "fault": self.fault,
            "control_started": self.mux.control_started,
            "in_control": self.deploy.in_control(),
            "active": None if m is None else {"id": m.id, "op": m.op, "phase": m.phase,
                                              "elapsed_s": round(m.elapsed(), 2)},
            "pose": None if pose is None else pose.brief(),
            "gt_pose": {"age_s": _r(self.pose_sub.age_s()), "rate_hz": round(self.pose_sub.rate_hz(), 2),
                        "rtf": None if pose is None else pose.rtf},
            "deploy": self.deploy.snapshot(),
            "mux": self.mux.snapshot(),
            "nav": self.nav_info,
            "nav_backend": self.cfg.nav_backend,
            "arm": self.arm.snapshot(),
            "tick": self.tick_stats,
            "ports": self.cfg.ports,
        }

    # -- main loop -------------------------------------------------------------------------------
    def setup(self) -> None:
        P = self.cfg.ports
        self.ctl = self.ctx.socket(zmq.ROUTER)
        self.ctl.setsockopt(zmq.LINGER, 0)
        self.ctl.bind(ep(P["body_ctl"], self.cfg.host))
        self.evt = self.ctx.socket(zmq.PUB)
        self.evt.setsockopt(zmq.LINGER, 0)
        self.evt.setsockopt(zmq.SNDHWM, 1000)
        self.evt.bind(ep(P["body_evt"], self.cfg.host))
        self.mux.start()
        self.pose_sub.start()
        self.deploy.start()
        self.log(f"[body] up: ROUTER {P['body_ctl']} PUB {P['body_evt']} SONIC-in PUB {P['sonic_in']} "
                 f"P1 REQ {P['p1_rep']} gt.pose SUB {P['p1_pose']} g1_debug SUB {P['sonic_debug']}")

    def run(self) -> None:
        self.setup()
        self._running = True
        poller = zmq.Poller()
        poller.register(self.ctl, zmq.POLLIN)
        period = 1.0 / self.cfg.control_hz
        state_period = 1.0 / self.cfg.state_pub_hz
        nxt = time.monotonic()
        nxt_state = nxt
        while self._running:
            timeout_ms = max(0.0, (nxt - time.monotonic()) * 1000.0)
            socks = dict(poller.poll(timeout_ms))
            if self.ctl in socks:
                for _ in range(20):
                    try:
                        frames = self.ctl.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._handle(frames)
            now = time.monotonic()
            if now >= nxt:
                t0 = time.perf_counter()
                self._tick(now)
                dt_ms = (time.perf_counter() - t0) * 1e3
                self.tick_stats["ticks"] += 1
                self.tick_stats["max_tick_ms"] = round(max(self.tick_stats["max_tick_ms"], dt_ms), 2)
                nxt += period
                if now - nxt > 5 * period:
                    self.tick_stats["overruns"] += 1
                    nxt = now + period
            if now >= nxt_state:
                nxt_state = now + state_period
                try:
                    self.evt.send_multipart([b"body.state", dumps_json(self.status())])
                except Exception as e:
                    self.log(f"[state] {e!r}")
        self._shutdown()

    def stop(self) -> None:
        self._running = False

    def _shutdown(self) -> None:
        self.log("[body] shutting down (planner IDLE hold; command{stop} is NOT sent)")
        if self.active is not None:
            self._cancel_active("service_shutdown")
        self.arm.abort("service_shutdown")
        p = self.pose_sub.latest()
        self.mux.hold(p.yaw if p else self.mux.last_facing_w, owner="shutdown")
        time.sleep(0.1)
        self.mux.close()
        self.pose_sub.stop()
        self.deploy.stop()
        self.p1.close()
        if self._nav2 is not None:
            self._nav2.close()
        for s in (self.ctl, self.evt):
            s.close(0)
        if self._logf:
            self._logf.close()
        if self._evf:
            self._evf.close()


def _r(v):
    return None if v is None or v == float("inf") else round(float(v), 3)


def _safe(m: Motion, pose) -> dict:
    try:
        return m._summary(pose)
    except Exception:
        return {}


def _short(d, maxlen: int = 400):
    try:
        s = json.dumps(d, default=str)
    except Exception:
        return str(d)[:maxlen]
    if len(s) <= maxlen:
        return d
    if isinstance(d, dict):
        return {k: (v if len(json.dumps(v, default=str)) < 200 else "...") for k, v in d.items()}
    return s[:maxlen]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="wl-body service (P3)")
    ap.add_argument("--port-offset", type=int, default=None, help="shift all contract ports (env WL_PORT_OFFSET)")
    ap.add_argument("--log-dir", default=os.environ.get("WL_BODY_LOG_DIR"))
    ap.add_argument("--keepalive-hz", type=float, default=50.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--turn-style", choices=["idle", "slowwalk"], default="idle")
    ap.add_argument("--v-default", type=float, default=0.45)
    ap.add_argument("--pelvis-z-min", type=float, default=0.55)
    ap.add_argument("--turn-push", type=float, default=None, help="FacingServo residual push (default 0.6)")
    ap.add_argument("--heading-bias-ki", type=float, default=None, help="planner-frame bias integrator (default 0 = off)")
    ap.add_argument("--nav-backend", choices=["nav2", "astar"], default=None,
                    help="go_to backend (env NAV_BACKEND, default nav2; falls back to astar if Nav2 is down)")
    args = ap.parse_args(argv)
    from .config import port_offset_from_env

    cfg = BodyConfig(port_offset=args.port_offset if args.port_offset is not None else port_offset_from_env(),
                     keepalive_hz=args.keepalive_hz, control_hz=args.control_hz, robot_radius=args.robot_radius,
                     turn_style=args.turn_style, v_default=args.v_default, pelvis_z_min=args.pelvis_z_min)
    if args.nav_backend:
        cfg.nav_backend = args.nav_backend
    if args.turn_push is not None:
        cfg.turn_push = args.turn_push
    if args.heading_bias_ki is not None:
        cfg.heading_bias_ki = args.heading_bias_ki
    log_dir = args.log_dir or f"/tmp/wl-body-{time.strftime('%Y%m%d-%H%M%S')}"
    svc = BodyService(cfg, log_dir=log_dir)

    def _sig(*_):
        svc.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        svc.run()
    except zmq.ZMQError as e:
        print(f"[body] FATAL: {e} (is another body / SonicMux already bound?)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
