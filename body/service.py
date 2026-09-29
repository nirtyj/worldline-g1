"""wl-body service (P3): ROUTER 5610 (commands) / PUB 5611 (events + state) / PULL 5612 (halt lane), 50 Hz loop.

Request  (DEALER or REQ -> ROUTER 5610, one JSON frame):
    {"id": "<client id, optional>", "op": "stand|walk|go_to|turn_to|velocity|approach|arm|stop|halt|resume|acquire|
                                          release|hello|bye|recover|status|ping|reload_map|clear_fault|
                                          shutdown_control", "args": {...},
     "execution_id"?, "generation"?, "control_epoch"?, "session"?}     (fence fields: top level or in args)
Reply    (same envelope back): {"id", "ok": bool, "state": "accepted|rejected|failed|done", "error"?: str, "data"?: {}}
Events   (PUB 5611, multipart [b"body.event", JSON]):
    {"id", "op", "state": "accepted|progress|succeeded|failed|canceled", "data": {...}, "seq", "t_wall"}
State    (PUB 5611, multipart [b"body.state", JSON]) at 5 Hz: mode, speed, control/deploy/P1 health, active op, pose,
         frame, arm, latch/lease/fences, runtime sessions.
Topics   (PUB 5611, multipart [b"body.<topic>", JSON], docs/contracts/m1.md §3.10-§3.12): body.halted, body.resumed,
         body.mode, body.fault, body.stale_command, body.lease, body.session.
Halt     (PUSH -> PULL 5612, body/halt.py): {"op": "halt", "epoch", "t_wall"} -> SonicMux IDLE at the current facing at
         once, ArmChannel.latch, the fence latch, body.halted{epoch, t_mono, arms_latched} (on the lane thread);
         the active leg motion ends `canceled` (reason halt) on the control thread before its next request or tick.

Rules: one active motion at a time; a new motion op pre-empts the current one (it ends "canceled" with
reason "preempted"); stop cancels (reason "stop") and runs a StopMotion (planner IDLE until |v| < 0.05).
Fences (body/fence.py): the halt latch, stale generations/epochs (`stale_command`, published as body.stale_command)
and leases (`body_busy`) gate every command that moves the robot; stop/halt/resume/status are never gated.
Watchdogs: gt.pose stale > pose_stale_s, g1_debug stale > debug_stale_s, op timeout, fall (gt fallen or pelvis_z below
band) -> the op fails, the mux holds IDLE, and after a fall the service latches fault="fallen", engages P1's band and
publishes body.fault{fell} (recover op = soft recovery, or clear_fault after a P1 reset). g1_debug lost while in
control -> fault deploy_lost (cleared by itself once the deploy is back in control). A registered runtime session that
stops pinging while the body moves -> an internal halt (reason runtime_lost). The mux itself holds IDLE if this loop
stalls for > cmd_stale_s. Nothing here sends command{stop} except shutdown_control.
The arm channel (op `arm`, body/arm.py) is separate from the one active motion: it overlays upper-body and hand
targets on every planner message, so it composes with any leg motion; `stop {arms: true}` also ends it.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import inspect
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

from .approach import ApproachMotion
from .arm import ArmChannel, ArmError
from .config import BodyConfig, ep
from .deploy_monitor import DeployMonitor
from .fence import Fence, FenceReject, Fences
from .frames import PlannerFrame
from .halt import HaltLane
from .motions import MOTIONS, Motion, MotionError, StopMotion
from .nav_grid import NavGrid
from .p1_client import P1Error, P1Rpc, PoseSub
from .nav2_backend import Nav2Link, select_motion
from .path_follower import GoToMotion
from .sonic_mux import SonicMux
from .velocity import VelocityMotion, route_velocity
from .wire import LocomotionMode, dumps_json, wrap

ALL_MOTIONS = {**MOTIONS, GoToMotion.op: GoToMotion, VelocityMotion.op: VelocityMotion,
               ApproachMotion.op: ApproachMotion}
TERMINAL = ("succeeded", "failed", "canceled")
LEG_OPS = ("walk", "velocity", "turn_to", "go_to", "approach")      # mode LOCOMOTION
TRANSITION_OPS = ("stand", "recover")                              # mode TRANSITION
ARM_ACTIVE = ("stream", "chunk", "script")                         # ArmChannel modes that are ARM_STREAM
FAULT_OK_OPS = ("stop", "recover", "stand")                        # may start while a fault is latched (stand: only
                                                                   # deploy_lost, see _start_motion)
# ops gated by the fences (halt latch, stale generation/epoch, lease). Never gated: stop, halt, resume, status, ping,
# hello, bye, release, clear_fault, reload_map, shutdown_control, recover. (arm / arm_script / scan are gated in
# _dispatch without events; the arm channel keeps its own session fences on top.)
GATED_OPS = ("stand", "walk", "turn_to", "go_to", "velocity", "approach")
# ArmChannel handlers body-arm provides with the `arm` op's signature handler(op_id, args, can_start) -> reply.
# Registered when the channel has them (docs/contracts/m1.md §3.9; body-arm owns them).
ARM_EXTRA_OPS = {"arm_script": "handle_arm_script", "scan": "handle_scan"}


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
        # locks: the halt lane (its own thread) shares the event socket and the op records. The arm channel locks
        # itself: its latch()/unlatch()/state() may be called from any thread (body/arm.py), so no lock here, and a
        # slow arm handler on the control thread can never delay a halt through this service.
        self._evt_lock = threading.RLock()      # self.evt sends, self.ops records, events.jsonl
        self._halt_lock = threading.Lock()      # serializes halt / resume (lane, ROUTER, watchdog)
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
        arm_kw = {"pose": self.pose_sub.latest} if "pose" in inspect.signature(ArmChannel).parameters else {}
        self.arm = ArmChannel(cfg, self.mux, self.deploy, self.emit, log=self.log, record=self._record, **arm_kw)
        self.fences = Fences()
        self.active: Motion | None = None
        self.ops: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.fault: str | None = None
        self.fault_info: dict | None = None
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
        # M2b: halt lane, modes, faults (docs/contracts/m1.md §3.10-§3.12)
        self.halt_lane: HaltLane | None = None
        self._halt_pending: collections.deque[tuple[int, str]] = collections.deque()
        self.halt_info: dict | None = None
        self.halt_ms: collections.deque[float] = collections.deque(maxlen=200)
        self.mode = "OFF"
        self._mode_t = time.monotonic()
        self.mode_changes = 0
        self.recoveries = 0
        self.approach_est: dict | None = None     # ApproachMotion's learnt glide / minimum step (last op)
        self._deploy_ok_since: float | None = None
        self._p1_async = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="p1-async")
        self.extra_ops = {op: getattr(self.arm, name) for op, name in ARM_EXTRA_OPS.items()
                          if callable(getattr(self.arm, name, None))}

    # -- logging / events ------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self._logf:
            self._logf.write(line + "\n")
        if self._ext_log:
            self._ext_log(line)

    def emit(self, op_id: str, state: str, data: dict | None = None) -> None:
        with self._evt_lock:
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
        if ev["op"] == "arm" and state in TERMINAL:
            self._maybe_policy_lost(op_id, data or {})

    def publish(self, topic: str, payload: dict) -> None:
        """A body.<topic> message on 5611 (thread safe; also written to events.jsonl with its topic)."""
        payload = {**payload, "seq": next(self._seq)}
        payload.setdefault("t_wall", time.time())
        raw = dumps_json(payload)
        with self._evt_lock:
            try:
                self.evt.send_multipart([f"body.{topic}".encode(), raw])
            except Exception as e:
                self.log(f"[evt] {topic} send failed: {e!r}")
            if self._evf:
                self._evf.write(dumps_json({"topic": f"body.{topic}", **payload}).decode() + "\n")

    def _maybe_policy_lost(self, op_id: str, data: dict) -> None:
        """body.fault{policy_lost}: a chunk (GR00T) session of the arm op ended by its own watchdog, i.e. the policy
        client stopped sending. An event only: the arm's hold_on_end applies and the mode is not FAULT."""
        chunk = data.get("kind") == "chunk" or data.get("mode") == "chunk" or data.get("session_id") is not None
        if chunk and data.get("ended_by") == "watchdog":
            self.publish("fault", {"kind": "policy_lost", "op_id": op_id, "session_id": data.get("session_id"),
                                   "stream": data.get("stream"), "hold": data.get("hold"), "latched": False})

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

    # -- halt lane (B.1): called from the lane thread, the ROUTER op and the session watchdog -----
    def halt(self, epoch, reason: str = "halt", source: str = "lane", t_recv: float | None = None,
             t_wall_sent=None, internal: bool = False) -> dict:
        """Latch HOLD now and publish body.halted. Never command{stop}. Thread safe; no I/O wait before the publish."""
        t_recv = time.monotonic() if t_recv is None else t_recv
        with self._halt_lock:
            try:
                req_epoch = int(epoch)
            except (TypeError, ValueError):
                self.log(f"[halt] bad epoch {epoch!r} from {source}")
                return {"ok": False, "error": "bad_args", "epoch": epoch}
            kind, ep_ = self.fences.latch(req_epoch, reason, internal=internal)
            arms = None
            active = self.active
            pose = self.pose_sub.latest()
            fresh = pose is not None and self.pose_sub.age_s() < self.cfg.pose_stale_s
            facing = pose.yaw if fresh else self.mux.last_facing_w
            if kind == "new":
                self.mux.latch(facing, owner=f"halt:{ep_}")
                arms = self._arm_latch(ep_, reason)
                self._halt_pending.append((ep_, reason))
                self.halt_info = {"epoch": ep_, "reason": reason, "source": source, "t_wall": time.time(),
                                  "facing_deg": None if facing is None else round(math.degrees(facing), 2),
                                  "arm": arms, "active": None if active is None else active.id}
            elif kind == "repeat" and self.halt_info is not None:
                arms = self.halt_info.get("arm")
            t_pub = time.monotonic()
            handle_ms = (t_pub - t_recv) * 1e3
            vx, vy, _ = self.pose_sub.velocity()
            payload = {"epoch": ep_, "requested_epoch": req_epoch, "kind": kind, "reason": reason, "source": source,
                       "t_mono": t_pub, "t_recv_mono": t_recv, "handle_ms": round(handle_ms, 3),
                       "t_wall_sent": t_wall_sent, "arms_latched": bool(arms and arms.get("latched")),
                       "arm": arms, "latched": self.fences.latched, "halt_epoch": self.fences.halt_epoch,
                       "facing_deg": None if facing is None else round(math.degrees(facing), 2),
                       "speed_mps": round(math.hypot(vx, vy), 3), "pose_source": "gt.pose",
                       "active": None if active is None else {"id": active.id, "op": active.op}}
            self.publish("halted", payload)
            if kind == "new":
                self.halt_ms.append(handle_ms)
        self.log(f"[halt] {kind} epoch={ep_} ({source}, {reason}) in {handle_ms:.2f} ms; arms={arms}; "
                 f"active={None if active is None else active.id}")
        return payload

    def _arm_latch(self, epoch: int, reason: str) -> dict:
        fn = getattr(self.arm, "latch", None)
        if not callable(fn):
            return {"latched": False, "why": "ArmChannel.latch not in this build (body-arm, B.1 arm side)"}
        t0 = time.perf_counter()
        try:
            r = fn(epoch, reason)
        except Exception as e:
            self.log(f"[halt] ArmChannel.latch failed: {traceback.format_exc()}")
            return {"latched": False, "error": repr(e)}
        out = dict(r or {})
        out.setdefault("latched", True)
        out["latch_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return out

    def resume(self, epoch, source: str = "op") -> dict:
        """resume{epoch}: clear the latch (FenceReject stale_command if epoch < halt_epoch). Publishes body.resumed."""
        with self._halt_lock:
            epoch = int(epoch)
            was = self.fences.unlatch(epoch)
            arm = None
            if was:
                self.mux.unlatch(owner=f"resume:{epoch}")
                fn = getattr(self.arm, "unlatch", None)
                if callable(fn):
                    try:
                        fn(epoch)
                        arm = "unlatched"
                    except Exception as e:
                        self.log(f"[resume] ArmChannel.unlatch failed: {traceback.format_exc()}")
                        arm = f"error: {e!r}"
            payload = {"epoch": epoch, "was_latched": was, "halt_epoch": self.fences.halt_epoch, "source": source,
                       "arm": arm, "t_mono": time.monotonic()}
            self.publish("resumed", payload)
        self.log(f"[resume] epoch={epoch} ({source}) was_latched={was}")
        return payload

    def resume_from_lane(self, epoch, t_recv: float | None = None) -> None:
        try:
            self.resume(epoch, source="lane")
        except FenceReject as e:
            self.publish("stale_command", {"op": "resume", "reason": e.reason, **e.data, "source": "lane"})

    def _process_halts(self) -> None:
        """Control thread: end the leg motion a halt interrupted, and fence off its leases."""
        while self._halt_pending:
            epoch, reason = self._halt_pending.popleft()
            if self.active is not None:
                self._cancel_active("halt", extra={"halt_epoch": epoch, "halt_reason": reason})
            lease = self.fences.revoke(max_epoch=epoch)
            if lease is not None:
                self.publish("lease", {"event": "revoked", "reason": "halt", "halt_epoch": epoch,
                                       "lease": lease.to_dict()})

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
        self._process_halts()
        op_id = str(req.get("id") or uuid.uuid4().hex[:12])
        op = str(req.get("op", ""))
        args = req.get("args") or {}
        try:
            rep = self._dispatch(op_id, op, args, req)
        except Exception as e:
            self.log(f"[ctl] op {op} crashed: {traceback.format_exc()}")
            rep = {"ok": False, "state": "failed", "error": f"internal: {e!r}"}
        rep["id"] = op_id
        self.ctl.send_multipart(envelope + [dumps_json(rep)])

    def _dispatch(self, op_id: str, op: str, args: dict, req: dict | None = None) -> dict:
        try:
            fence = Fence.parse(req, args)
        except FenceReject as e:
            return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
        back = self.fences.touch(fence.session)
        if back is not None:
            self.publish("session", {"event": "back", "session": back["session"], "pings": back["pings"]})
        if op == "ping":
            data = {"t_wall": time.time()}
            if fence.session is not None:
                data["session"] = fence.session
                data["registered"] = fence.session in self.fences.sessions
            return {"ok": True, "state": "done", "data": data}
        if op == "status":
            data = self.status()
            if args.get("id"):
                data["op"] = self.ops.get(str(args["id"]))
            return {"ok": True, "state": "done", "data": data}
        if op == "halt":
            if args.get("epoch") is None:
                return {"ok": False, "state": "rejected", "error": "bad_args", "data": {"need": "epoch"}}
            payload = self.halt(args["epoch"], reason=str(args.get("reason") or "halt"), source="op",
                                t_wall_sent=args.get("t_wall"))
            if payload.get("ok") is False:
                return {"ok": False, "state": "rejected", "error": payload.get("error"), "data": payload}
            self._process_halts()
            return {"ok": True, "state": "done", "data": payload}
        if op == "resume":
            if args.get("epoch") is None:
                return {"ok": False, "state": "rejected", "error": "bad_args", "data": {"need": "epoch"}}
            try:
                return {"ok": True, "state": "done", "data": self.resume(args["epoch"], source="op")}
            except FenceReject as e:
                self._stale(op_id, op, fence, e)
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
            except (TypeError, ValueError):
                return {"ok": False, "state": "rejected", "error": "bad_args", "data": {"arg": "epoch"}}
        if op == "acquire":
            return self._acquire(op_id, args, fence)
        if op == "release":
            try:
                lease = self.fences.release(fence.execution_id, args.get("lease_id"))
            except FenceReject as e:
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
            if lease is not None:
                self._end_owner_motion(lease.owner, "released")
                self.publish("lease", {"event": "released", "lease": lease.to_dict()})
            return {"ok": True, "state": "done", "data": {"released": None if lease is None else lease.to_dict()}}
        if op == "hello":
            if fence.session is None:
                return {"ok": False, "state": "rejected", "error": "bad_args", "data": {"need": "session"}}
            wd = min(max(float(args.get("watchdog_s", self.cfg.session_watchdog_s)), 0.2), 10.0)
            s = self.fences.hello(fence.session, wd, {k: v for k, v in args.items() if k in ("client", "pid")})
            self.publish("session", {"event": "hello", "session": fence.session, "watchdog_s": wd})
            snap = self.fences.snapshot()
            return {"ok": True, "state": "done", "data": {
                "session": s["session"], "watchdog_s": wd, "mode": self.mode,
                **{k: snap[k] for k in ("latched", "halt_epoch", "resume_epoch", "epoch_seen", "generation_floor",
                                        "lease")}}}
        if op == "bye":
            return {"ok": True, "state": "done", "data": {"removed": self.fences.bye(str(fence.session))}}
        if op == "reload_map":
            nav = self.get_nav(reload=True)
            return {"ok": nav is not None, "state": "done", "data": {"nav": self.nav_info}}
        if op == "clear_fault":
            prev = self._clear_fault("clear_fault")
            return {"ok": True, "state": "done", "data": {"cleared": prev}}
        if op == "shutdown_control":
            if not args.get("confirm"):
                return {"ok": False, "state": "rejected", "error": "shutdown_control needs args.confirm=true "
                        "(sends command{stop}: the deploy damps and exits; put P1's band on first)"}
            self._cancel_active("shutdown")
            self.mux.shutdown_control()
            return {"ok": True, "state": "done", "data": {"mux": self.mux.snapshot()}}
        if op == "arm" or op in self.extra_ops:
            try:
                self._gate(op_id, op, args, fence, events=False)
            except FenceReject as e:
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
            handler = self.arm.handle if op == "arm" else self.extra_ops[op]
            try:
                return handler(op_id, args, self._arm_can_start())
            except ArmError as e:
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
        if op == "stop" and args.get("arms"):
            self.arm.end("stop")
        if op == "velocity":
            if args.get("goal_id") is None and args.get("stream") is not None and \
                    isinstance(self.active, VelocityMotion) and self.active.stream == str(args["stream"]):
                try:   # a stream update: fenced like its op, but no events (updates are not ops)
                    self._gate(op_id, op, args, fence, events=False)
                except FenceReject as e:
                    return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
            rep = route_velocity(self, args)   # stream updates (and Nav2 /cmd_vel) never start a new op
            if rep is not None:
                return rep
        if op not in ALL_MOTIONS:
            return {"ok": False, "state": "rejected", "error": f"unknown op {op!r}"}
        if op in GATED_OPS:
            try:
                self._gate(op_id, op, args, fence, events=True)
            except FenceReject as e:
                return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
        return self._start_motion(op_id, op, args, fence)

    # -- fences / leases (B.2) -------------------------------------------------------------------
    def _gate(self, op_id: str, op: str, args: dict, fence: Fence, events: bool) -> None:
        """Raise FenceReject if the fences refuse this command; otherwise record the fence (and an implicit resume).
        events=True: a refused op gets its record and `failed` event like any rejection (stream updates do not)."""
        try:
            implicit = self.fences.check(op, fence)
        except FenceReject as e:
            if e.stale:
                self._stale(op_id, op, fence, e)
            if events:
                self._record(op_id, op, args, fence)
                self.emit(op_id, "failed", {"reason": e.reason, **e.data})
            raise
        if implicit:
            self.resume(fence.control_epoch, source=f"implicit:{op}")
        self.fences.accept(fence)

    def _stale(self, op_id: str, op: str, fence: Fence, e: FenceReject) -> None:
        self.publish("stale_command", {"id": op_id, "op": op, "reason": e.reason, **fence.to_dict(), **e.data})

    def _acquire(self, op_id: str, args: dict, fence: Fence) -> dict:
        try:
            lease, old, implicit = self.fences.acquire(fence, args.get("mode", "ANY"))
        except FenceReject as e:
            if e.stale:
                self._stale(op_id, "acquire", fence, e)
            return {"ok": False, "state": "rejected", "error": e.reason, "data": e.data}
        if implicit:
            self.resume(fence.control_epoch, source="implicit:acquire")
        self.fences.accept(fence)
        if old is not None:
            self._end_owner_motion(old.owner, "superseded")
            self.publish("lease", {"event": "revoked", "reason": "superseded", "lease": old.to_dict(),
                                   "by": lease.owner})
        self.publish("lease", {"event": "acquired", "lease": lease.to_dict()})
        return {"ok": True, "state": "done", "data": {"lease": lease.to_dict(),
                                                      "superseded": None if old is None else old.to_dict()}}

    def _end_owner_motion(self, owner: str, reason: str) -> None:
        m = self.active
        if m is not None and getattr(m, "fence", None) is not None and m.fence.execution_id == owner:
            self._cancel_active(reason)

    def _arm_can_start(self) -> tuple[bool, str | None, dict]:
        if self.fault:
            return False, f"fault:{self.fault}", {"hint": "reset the robot in P1, then clear_fault"}
        if not (self.mux.control_started and self.deploy.in_control()):
            return False, "not_standing", {"hint": "call stand first", "deploy": self.deploy.snapshot()}
        return True, None, {}

    def _record(self, op_id: str, op: str, args: dict, fence: Fence | None = None) -> None:
        with self._evt_lock:
            self.ops[op_id] = {"id": op_id, "op": op, "args": args, "state": "received", "t_start": time.time(),
                               "t_end": None, "result": None, "events": 0,
                               "fence": fence.to_dict() if fence is not None and fence.fenced else None}
            while len(self.ops) > 300:
                self.ops.popitem(last=False)

    def _reject(self, op_id: str, reason: str, data: dict | None = None) -> dict:
        self.emit(op_id, "failed", {"reason": reason, **(data or {})})
        return {"ok": False, "state": "rejected", "error": reason, "data": data or {}}

    def _start_motion(self, op_id: str, op: str, args: dict, fence: Fence | None = None) -> dict:
        self._record(op_id, op, args, fence)
        cls = ALL_MOTIONS[op]
        pose = self.pose_sub.latest()
        if pose is None or self.pose_sub.age_s() > self.cfg.pose_stale_s:
            return self._reject(op_id, "no_pose", {"gt_pose_age_s": _r(self.pose_sub.age_s())})
        if self.fault and (op not in FAULT_OK_OPS or (op == "stand" and self.fault != "deploy_lost")):
            return self._reject(op_id, f"fault:{self.fault}", {"hint": "recover (soft recovery), or reset the robot "
                                                               "in P1 and clear_fault"})
        if cls.needs_control and not (self.mux.control_started and self.deploy.in_control()):
            return self._reject(op_id, "not_standing", {"hint": "call stand first",
                                                        "deploy": self.deploy.snapshot()})
        if cls.needs_control and not self.frame.known:
            return self._reject(op_id, "planner_frame_unknown")
        cls, backend_info = select_motion(self, op, args, cls)   # go_to: NAV_BACKEND nav2 | astar
        if self.active is not None:
            self._cancel_active("stop" if op == "stop" else "preempted", by=op_id)
        m = cls(op_id, args, self.mctx)
        m.fence = fence if fence is not None and fence.fenced else None
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
        extra = {"fence": m.fence.to_dict()} if m.fence is not None else {}
        self.emit(op_id, "accepted", {"args": args, **getattr(m, "backend_info", {}), **(data or {}), **extra})
        return {"ok": True, "state": "accepted", "data": _short(data or {})}

    def _cancel_active(self, reason: str, by: str | None = None, extra: dict | None = None) -> None:
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
        data.update(extra or {})
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
        if m.op == "recover" and state == "succeeded":
            self.recoveries += 1
            self._clear_fault("recover", data={"recoveries": self.recoveries})
        self.emit(m.id, state, data)

    # -- faults (B.3) ----------------------------------------------------------------------------
    def _raise_fault(self, kind: str, fault: str, data: dict) -> None:
        """Latch `fault`, publish body.fault{kind}, engage P1's band (sim), drop the arm override. Never command{stop}."""
        self.fault = fault
        self.fault_info = {"kind": kind, "fault": fault, "t_wall": time.time(), **data}
        band = None
        if self.cfg.fault_band_on:
            band = "requested"
            self._p1_async.submit(self._band_on_async, kind)
        self.arm.abort("fallen" if kind == "fell" else kind)
        self.publish("fault", {"kind": kind, "fault": fault, "band": band, "latched": True, **data})
        self.log(f"[fault] {kind}: {json.dumps(_short(data))} (band {band})")

    def _band_on_async(self, why: str) -> None:
        rep = self.p1.try_call("band", on=True)
        self.log(f"[fault] P1 band on ({why}): {rep}")

    def _clear_fault(self, by: str, data: dict | None = None) -> str | None:
        prev, info = self.fault, self.fault_info
        self.fault, self.fault_info = None, None
        if prev is not None:
            self.publish("fault", {"kind": (info or {}).get("kind", prev), "fault": prev, "cleared": True, "by": by,
                                   "latched": False, **(data or {})})
        return prev

    def _check_faults(self, pose, age: float, now: float) -> None:
        m = self.active
        if pose is not None and age < self.cfg.pose_stale_s:
            fallen = pose.fallen or (self.mux.control_started and pose.pelvis_z < self.cfg.pelvis_z_min)
            if fallen and self.fault != "fallen" and not (m is not None and m.op == "recover"):
                self.log(f"[watchdog] FALL detected: {pose.brief()}")
                if m is not None:
                    self._finish("failed", {"reason": "fallen", "pose": pose.brief(), **_safe(m, pose)})
                self.mux.hold(pose.yaw, owner="fall")
                self._raise_fault("fell", "fallen", {"pose": pose.brief()})
        # deploy lost: we started control, it was in control, and g1_debug stopped (not after shutdown_control)
        if self.mux.control_started and not self.mux.control_stopped and self.deploy.first_heading_mono is not None:
            dage = self.deploy.age_s()
            if dage > self.cfg.deploy_lost_s:
                self._deploy_ok_since = None
                if self.fault is None:
                    if m is not None and m.needs_control:
                        self._finish("failed", {"reason": "deploy_stale", "g1_debug_age_s": _r(dage),
                                                **_safe(m, pose)})
                    self._raise_fault("deploy_lost", "deploy_lost", {"g1_debug_age_s": _r(dage),
                                                                      "deploy_alive": self.deploy.alive()})
            elif self.fault == "deploy_lost" and self.deploy.in_control():
                if self._deploy_ok_since is None:
                    self._deploy_ok_since = now
                elif now - self._deploy_ok_since >= 1.0:
                    self._clear_fault("deploy_back", {"g1_debug_rate_hz": round(self.deploy.rate_hz(), 1)})

    # -- runtime sessions (B.3) ------------------------------------------------------------------
    def _check_sessions(self) -> None:
        for s in self.fences.expired():
            moving = (self.active is not None and self.active.op in LEG_OPS) or \
                self._arm_state().get("mode") in ARM_ACTIVE
            self.publish("session", {"event": "lost", "session": s["session"], "age_s": s["age_s"],
                                     "watchdog_s": s["watchdog_s"], "held": moving})
            self.log(f"[session] runtime {s['session']} silent {s['age_s']} s (watchdog {s['watchdog_s']} s); "
                     f"{'holding' if moving else 'idle'}")
            if moving:
                self.halt(-1, reason="runtime_lost", source="watchdog", internal=True)
                self._process_halts()
            lease = self.fences.lease
            if lease is not None and lease.session in (None, s["session"]):
                self.fences.revoke()
                self.publish("lease", {"event": "revoked", "reason": "runtime_lost", "lease": lease.to_dict()})

    # -- modes (B.3) -----------------------------------------------------------------------------
    def _arm_state(self) -> dict:
        fn = getattr(self.arm, "state", None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                return {"mode": "unknown"}
        return {"mode": fn if isinstance(fn, str) else "off", "owner": getattr(self.arm, "owner", None),
                "op_id": getattr(self.arm, "op_id", None)}

    def compute_mode(self, arm: dict | None = None) -> str:
        if self.mux.control_stopped:
            return "ESTOP"
        if self.fault:
            return "FAULT"
        m = self.active
        if m is not None and m.op in TRANSITION_OPS:
            return "TRANSITION"
        if not (self.mux.control_started and self.deploy.in_control()):
            return "OFF"
        if m is not None and m.op in LEG_OPS:
            return "LOCOMOTION"
        if (arm if arm is not None else self._arm_state()).get("mode") in ARM_ACTIVE:
            return "ARM_STREAM"
        return "HOLD"

    def _update_mode(self, now: float) -> None:
        arm = self._arm_state()
        mode = self.compute_mode(arm)
        if mode != self.mode:
            prev, self.mode = self.mode, mode
            dur = now - self._mode_t
            self._mode_t = now
            self.mode_changes += 1
            m = self.active
            self.publish("mode", {"mode": mode, "prev": prev, "prev_s": round(dur, 3), "t_mono": now,
                                  "active": None if m is None else {"id": m.id, "op": m.op},
                                  "arm_mode": arm.get("mode"), "fault": self.fault, "latched": self.fences.latched})

    # -- control tick ----------------------------------------------------------------------------
    def _tick(self, now: float) -> None:
        self._process_halts()
        pose = self.pose_sub.latest()
        age = self.pose_sub.age_s()
        self._check_faults(pose, age, now)
        self._check_sessions()
        m = self.active
        try:
            self.arm.tick(now)
        except Exception:
            self.log(f"[arm] tick crashed: {traceback.format_exc()}")
            self.arm.abort("internal")
        if m is not None:
            self._tick_motion(m, pose, age, now)
        else:
            self._learn_bias(pose, age)
        self._update_mode(now)

    def _tick_motion(self, m: Motion, pose, age: float, now: float) -> None:
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
        vx, vy, wz = self.pose_sub.velocity()
        arm = self.arm.snapshot()
        fences = self.fences.snapshot()
        with self._halt_lock:                   # the lane thread appends to halt_ms
            hm = sorted(self.halt_ms)
        return {
            "t_wall": time.time(), "uptime_s": round(time.monotonic() - self.t_start, 1),
            "mode": self.mode, "mode_for_s": round(time.monotonic() - self._mode_t, 2),
            "speed": round(math.hypot(vx, vy), 4), "yaw_rate": round(wz, 4), "pose_source": "gt.pose",
            "fault": self.fault, "fault_info": self.fault_info,
            "latched": fences["latched"], "halt_epoch": fences["halt_epoch"], "lease": fences["lease"],
            "fences": fences, "recoveries": self.recoveries,
            "halt": {"last": self.halt_info, "n": len(hm),
                     "handle_ms_p50": None if not hm else round(hm[len(hm) // 2], 3),
                     "handle_ms_max": None if not hm else round(hm[-1], 3),
                     "lane": None if self.halt_lane is None else self.halt_lane.stats},
            "control_started": self.mux.control_started,
            "in_control": self.deploy.in_control(),
            "active": None if m is None else {"id": m.id, "op": m.op, "phase": m.phase,
                                              "elapsed_s": round(m.elapsed(), 2),
                                              "fence": None if getattr(m, "fence", None) is None else
                                              m.fence.to_dict()},
            "pose": None if pose is None else pose.brief(),
            "gt_pose": {"age_s": _r(self.pose_sub.age_s()), "rate_hz": round(self.pose_sub.rate_hz(), 2),
                        "rtf": None if pose is None else pose.rtf},
            "deploy": self.deploy.snapshot(),
            "mux": self.mux.snapshot(),
            "nav": self.nav_info,
            "nav_backend": self.cfg.nav_backend,
            "arm": arm,
            "approach_est": self.approach_est,
            "ops_extra": sorted(self.extra_ops),
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
        halt_sock = self.ctx.socket(zmq.PULL)
        halt_sock.setsockopt(zmq.LINGER, 0)
        halt_sock.bind(ep(P["body_halt"], self.cfg.host))
        self.mux.start()
        self.pose_sub.start()
        self.deploy.start()
        self.halt_lane = HaltLane(self, halt_sock)
        self.halt_lane.start()
        self.log(f"[body] up: ROUTER {P['body_ctl']} PUB {P['body_evt']} PULL halt {P['body_halt']} "
                 f"SONIC-in PUB {P['sonic_in']} P1 REQ {P['p1_rep']} gt.pose SUB {P['p1_pose']} "
                 f"g1_debug SUB {P['sonic_debug']}; extra arm ops {sorted(self.extra_ops)}")

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
                    raw = dumps_json(self.status())
                    with self._evt_lock:
                        self.evt.send_multipart([b"body.state", raw])
                except Exception as e:
                    self.log(f"[state] {e!r}")
        self._shutdown()

    def stop(self) -> None:
        self._running = False

    def _shutdown(self) -> None:
        self.log("[body] shutting down (planner IDLE hold; command{stop} is NOT sent)")
        if self.halt_lane is not None:
            self.halt_lane.stop()
            self.halt_lane.join(timeout=1.0)
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
        self._p1_async.shutdown(wait=False)
        if self._nav2 is not None:
            self._nav2.close()
        for s in (self.ctl, self.evt):
            s.close(0)
        if self.halt_lane is not None:
            self.halt_lane.sock.close(0)
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
    ap.add_argument("--no-fault-band", action="store_true", help="do not engage P1's band on a fall / deploy_lost")
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
    if args.no_fault_band:
        cfg.fault_band_on = False
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
