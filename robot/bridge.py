"""G1Robot: the RobotBridge façade the runtime talks to (api/services.py RobotBridge; PLAN §6.7).

Duck-compatible with the old ThorRobot read surface (lookup_keypoints, memory, base_state, gripper, proprio,
telemetry, perception) plus the new execution surface (start(execution) -> ExecutionHandle, halt, resume, estop,
capabilities, registry, timeout_s, observation_id, events). Everything physical goes through services/, everything
ground-truth through world/; agent/ never imports either.

Dispatch (`start`) per tool (api/services.py docstring):
    speak               SpeechService
    list_locations      LocationsService (instant)
    navigate            NavigationService (keypoint | reposition); CAPABILITY: navigation health
    check_reachability  ManipulationService.start_reachability
    manipulate          ManipulationService.start; CAPABILITY: a healthy skill for (action, type, arm)
    observe / look      ObservationService.start (glance | scan)
    wait_and_observe    ObservationService.start_wait (the harness may also run its own)
`start` never blocks; it raises api.execution.Rejected("capability", code, message) when a health check fails.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import math
import time
from typing import Any

from api.execution import Execution, Rejected
from api.results import ListLocationsResult, finish
from api.types import ServiceHealth
from world import coords

from services.common import EventSink, HaltGate, start_execution
from services.executors import ExecutorContext, build_executor
from services.locations import LocationsService
from services.manipulation import ManipConfig, ManipulationService
from services.navigation import NavConfig, NavigationService
from services.observation import ObservationService, ScanConfig
from services.reachability import G1Workspace, ReachabilityModel
from services.skills import build_registry
from services.speech import SpeechService

from .health import CapabilityPolicy, HaltResender, HealthMonitor
from .profile import StackProfile

ARMS = ("left", "right")


class G1Robot:
    def __init__(self, world: Any, body: Any, *, profile: StackProfile, clock: Any, log: Any = None,
                 frames: Any = None, registry: Any = None):
        self.world = world
        self.body = body
        self.clock = clock
        self.stack_profile = profile
        self.profile = profile.robot
        self.frames = frames
        self.sink = EventSink(log)
        self.gate = HaltGate()
        g1 = profile.g1
        ws = dict(g1.get("workspace") or {})
        walking = dict(g1.get("walking") or {})
        walking["nav_speed_mps"] = self.profile.walk_speed_mps
        self.locations = LocationsService(world, ws)
        self.nav = NavigationService(world, body, clock, NavConfig.from_dicts(walking, ws), gate=self.gate,
                                     events=self.sink, observation_id=self.observation_id, locations=self.locations)
        scan = dict(g1.get("scan") or {})
        scan["executor"] = profile.scan_executor
        self.obs = ObservationService(world, body, clock, ScanConfig.from_dict(scan), nav=self.nav, frames=frames,
                                      events=self.sink, gate=self.gate)
        manip_cfg = dict(g1.get("manipulation") or {})
        self.executors = self._executors(manip_cfg)
        # a skill's health is its executor's health (the registry never changes the enum; PLAN §5.2)
        self.skill_registry = registry or build_registry(
            profile.manip_executors, world, health={b: ex.health for b, ex in self.executors.items()})
        self.reach = ReachabilityModel(world, G1Workspace.from_dict(ws), registry=self.skill_registry,
                                       observation=self.obs, nav=self.nav, body=body)
        self.manip = ManipulationService(world, self.skill_registry, self.reach, self.executors, clock,
                                         ManipConfig.from_dict(manip_cfg), nav=self.nav, observation=self.obs,
                                         gate=self.gate, events=self.sink, policy=profile.manip_policy, body=body)
        self.speech = SpeechService(clock, self.sink, observation_id=self.observation_id)
        self.policy = CapabilityPolicy(world, nav=self.nav, manip=self.manip, body=body,
                                       observation_detail=lambda: f"{self.world.source} "
                                                                  f"{getattr(self.world.cam, 'name', '')}")
        self.monitor = HealthMonitor(self.capabilities, self.skill_registry, self.sink.emit,
                                     sim_events=getattr(world, "drain_events", None))
        self.halt_resender = HaltResender(body, self.sink.emit)
        self._active: dict[str, Execution] = {}
        self._handles: dict[str, Any] = {}
        self._map: dict | None = None
        self._epoch_seen = 0                             # the newest control_epoch seen (one epoch space, R.2)
        self._own_halts: collections.deque = collections.deque(maxlen=50)
        self._last_fell = -1e9
        self._body_events_attached = False
        self.body_events: collections.Counter = collections.Counter()

    def _executors(self, manip_cfg: dict) -> dict[str, Any]:
        """Profile executor names -> executors by registry backend (services/executors/registry.py). The first
        executor a profile lists for a backend wins."""
        out: dict[str, Any] = {}
        for name in self.stack_profile.manip_executors:
            ctx = ExecutorContext(name=name, world=self.world, body=self.body, clock=self.clock, gate=self.gate,
                                  events=self.sink, profile=self.stack_profile, manip_cfg=manip_cfg,
                                  port_offset=self.stack_profile.port_offset)
            ex = build_executor(ctx)
            out.setdefault(ex.backend, ex)
        return out

    # ================================================================== THOR-compatible read surface
    def lookup_keypoints(self) -> dict:
        if self._map is None:
            m = copy.deepcopy(self.world.lookup_keypoints())
            m["profile"] = self.stack_profile.name
            m["nav_speed_mps"] = self.profile.walk_speed_mps
            m["max_reach_height_m"] = self.profile.reach_h_max_m
            m["min_reach_height_m"] = self.profile.reach_h_min_m
            m["executors"] = self.executor_names()
            sim_added = bool(getattr(self.world.cam, "sim_added", False))
            # PLAN §12.2: the sim-added head camera is labelled wherever it shows ("camera: head (sim-added)")
            m["camera"] = {"name": getattr(self.world.cam, "name", "head"), "sim_added": sim_added,
                           "caption": "camera: head (sim-added)" if sim_added else
                           f"camera: {getattr(self.world.cam, 'name', 'head')}"}
            self._map = m
        return self._map

    def executor_names(self) -> dict[str, str]:
        skills = self.skill_registry.skills()
        manip = next((s.executor for s in skills if self.skill_registry.healthy(s.skill_id).ok),
                     skills[0].executor if skills else "none")
        return {"navigate": self.nav.executor, "manipulate": manip,
                "observe": self.stack_profile.scan_executor}

    def memory(self) -> list[dict]:
        return []

    def base_state(self) -> dict:
        p = self.world.robot_pose()
        at, between = self.nav.at()
        moving = self.nav.moving or p.speed > 0.05
        mx, mz = coords.to_map_xz(p.x, p.y)
        return {"moving": moving, "at": None if moving else at, "between": between,
                "xy": [round(mx, 3), round(mz, 3)]}

    def gripper(self, arm: str) -> dict:
        """From the GT attach state (Dex3 q in M2b). A gripper feels, it can't name."""
        if self.world.hands().get(arm):
            return {"closed": True, "width": 0.06, "force": 8.0}
        return {"closed": False, "width": 0.085, "force": 0.0}

    def proprio(self, arm: str) -> dict:
        held = self.world.hands().get(arm)
        return {"joints": [0.0] * 7, "gripper": "closed" if held else "open", "posture": "holding" if held else "home"}

    def telemetry(self) -> dict:
        p = self.world.robot_pose()
        at, between = self.nav.at()
        moving = self.nav.moving or p.speed > 0.05
        mx, mz = coords.to_map_xz(p.x, p.y)
        cp = self.world.camera_pose(pose=p)
        st = self.body.state() if hasattr(self.body, "state") else {}
        hands = self.world.hands()
        sim = self.world.sim_health() if hasattr(self.world, "sim_health") else None
        rtf = sim.rtf if sim is not None else None
        active = [{"id": e.execution_id, "skill": e.tool_name, "status": e.status}
                  for e in sorted(self._active.values(), key=lambda e: e.execution_id)]
        bh = self.body.health()
        return {
            "pose": {"x": round(mx, 3), "z": round(mz, 3), "yaw": round(coords.yaw_map_deg(p.yaw), 2),
                     "horizon": round(math.degrees(cp.pitch_down), 2), "at": None if moving else at,
                     "between": between},
            "velocity": {"linear_mps": round(p.speed, 3), "angular_dps": round(math.degrees(p.wz), 1)},
            "moving": moving,
            "arms": {a: self.proprio(a) for a in ARMS},
            "grippers": {a: self.gripper(a) for a in ARMS},
            "active_skills": active,
            "health": {"ok": bh.ok, "source": self.world.source, "state": bh.state, "detail": bh.detail},
            "body": {"mode": st.get("mode", "HOLD"), "lease": (st.get("lease") or {}).get("owner")
                     if isinstance(st.get("lease"), dict) else st.get("lease"), "upright": not p.fallen,
                     "rtf": rtf, "carry": any(hands.values()), "halt_epoch": self.gate.epoch,
                     "latched": self.gate.latched, "executor": self.nav.executor,
                     "carry_lock": bool((st.get("carry") or {}).get("engaged")),
                     "pelvis_z": round(p.pelvis_z, 3), "sim": sim.to_dict() if sim is not None else None,
                     "halt_resend": self.halt_resender.active},
        }

    def perception(self) -> dict:
        return self.obs.perception()

    # ================================================================== execution
    def start(self, execution: Execution):
        tool = execution.tool_name
        a = execution.args
        self._epoch_seen = max(self._epoch_seen, int(execution.control_epoch))
        g = self.policy.gate(tool, a)                      # R.6: sim DEGRADED / UNSAFE (world's RTF verdict)
        if g is not None:
            raise Rejected("capability", g.code, g.message)
        if tool == "speak":
            h = self.speech.start(str(a.get("text", "")), execution)
        elif tool == "list_locations":
            h = self._start_locations(execution)
        elif tool == "navigate":
            health = self.nav.health()
            if not health.ok:
                raise Rejected("capability", "nav_unhealthy",
                               f"navigation stack unavailable ({health.state}: {health.detail})")
            h = self.nav.start(execution)
        elif tool == "check_reachability":
            h = self.manip.start_reachability(execution)
        elif tool == "manipulate":
            action = str(a.get("action") or execution.action or "pick")
            skill, why = self.manip.capability(action, str(a.get("object_type")), a.get("arm"))
            if skill is None:
                raise Rejected("capability", "policy_unavailable", f"policy unavailable: {why}")
            if self.world.robot_pose().fallen:
                raise Rejected("capability", "controller_unavailable", "the robot is down")
            h = self.manip.start(execution)
        elif tool in ("observe", "look"):
            h = self.obs.start(execution)
        elif tool == "wait_and_observe":
            lease_held = any("body" in e.resources for e in self._active.values())
            h = self.obs.start_wait(execution, lease_held=lease_held)
        else:
            raise Rejected("schema", "unknown_tool", f"unknown tool {tool!r}")
        self._track(execution, h)
        return h

    def _track(self, execution: Execution, h: Any) -> None:
        self._active[execution.execution_id] = execution
        self._handles[execution.execution_id] = h

        async def drop():
            try:
                await h.result()
            finally:
                self._active.pop(execution.execution_id, None)
                self._handles.pop(execution.execution_id, None)
        asyncio.ensure_future(drop())

    def _start_locations(self, execution: Execution):
        async def work(h):
            locs = self.locations.list(None, execution.args.get("query"))
            return finish(execution, "succeeded", ListLocationsResult(locs), t_end=round(self.clock.now(), 3),
                          observation_id=self.observation_id())
        return start_execution(execution, work, clock=self.clock, observation_id=self.observation_id)

    def halt(self, control_epoch: int | None = None) -> dict:
        """Latch HOLD within 30 ms (PLAN §5.6). One epoch space (M2b R.2): the halt fences `control_epoch` (the
        harness passes its current one; without it, the newest control_epoch any execution started with), the
        runtime gate latches at it and the body's halt lane gets the same number. Running body executions end
        failed(halted). If the body did not ack within the budget, robot/health.py re-sends it every 100 ms."""
        t0 = time.perf_counter()
        epoch = self.gate.halt(self._epoch_now() if control_epoch is None else int(control_epoch))
        receipt = dict(self.body.halt(epoch))
        receipt.setdefault("accepted", True)
        if receipt.get("at_rest") is None:
            v = self.world.planar_speed() if hasattr(self.world, "planar_speed") else None
            receipt["at_rest"] = None if v is None else bool(v < 0.05)
        receipt["epoch"] = epoch
        receipt["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        receipt["resend"] = bool(not receipt.get("stopped") and self.halt_resender.start(epoch))
        self._own_halts.append(epoch)
        self.sink.emit("halted", epoch=epoch, stopped=receipt.get("stopped"), via=receipt.get("via"),
                       latency_ms=receipt["latency_ms"], rtt_ms=receipt.get("rtt_ms"))
        return receipt

    def _epoch_now(self) -> int:
        """The newest control_epoch this robot has seen (executions started, resumes, the running body ones)."""
        act = [int(e.control_epoch) for e in self._active.values() if "body" in e.resources]
        return max([self._epoch_seen, *act])

    def resume(self, control_epoch: int | None = None) -> None:
        """Clear the latch. `control_epoch` is the epoch the next executions carry (above the halt's). The halt
        latch's arm pose goes back to SONIC's own arms unless a hand holds something (then the next place takes it
        over)."""
        self.halt_resender.cancel()
        self.gate.resume(control_epoch)
        if control_epoch is not None:
            self._epoch_seen = max(self._epoch_seen, int(control_epoch))
        hands = {}
        try:
            hands = self.world.hands()
        except Exception:  # noqa: BLE001
            pass
        try:
            rep = self.body.resume(control_epoch, release_arms=not any(hands.values()))
        except TypeError:                                   # a body without release_arms (older fakes)
            rep = self.body.resume(control_epoch)
        self.sink.emit("resumed", epoch=control_epoch, body=rep if isinstance(rep, dict) else None)

    def estop(self, reason: str) -> dict:
        """Operator kill button only (any profile). The deploy exits on a real stack; P2 restart required."""
        epoch = self.gate.halt(self._epoch_now())
        res = dict(self.body.estop(reason))
        res["body_epoch"] = epoch
        self.sink.emit("safety_event", kind="estop", reason=reason)
        return res

    # ================================================================== body events (B.3), pushed by the body
    def _on_body_topic(self, topic: str, msg: dict) -> None:
        """A body.* topic (on the runtime's loop, SonicBody.attach_events) -> robot events, no polling:
            body.mode            -> body_mode {mode, prev, ...}
            body.fault           -> safety_event {kind: fell | deploy_lost, source: body}; policy_lost -> body_fault
                                    (the GR00T session's own hold applies; not a stop); cleared -> body_fault
            body.halted          -> a halt this runtime did not send (the body's runtime-session watchdog): the gate
                                    latches at that epoch, running body executions end failed(halted), and
                                    safety_event {kind: runtime_lost} lets the harness resume above it
            body.stale_command   -> stale_result {execution_id, why}
            body.resumed / lease / session -> body_event (trace)"""
        self.body_events[topic] += 1
        t = topic.split(".", 1)[-1]
        now = time.monotonic()
        if t == "mode":
            self.sink.emit("body_mode", mode=msg.get("mode"), prev=msg.get("prev"), prev_s=msg.get("prev_s"),
                           fault=msg.get("fault"), latched=msg.get("latched"), arm_mode=msg.get("arm_mode"),
                           source="body")
        elif t == "fault":
            kind = str(msg.get("kind") or "")
            if msg.get("cleared"):
                self.sink.emit("body_fault", kind=kind or msg.get("fault"), cleared=True, by=msg.get("by"))
            elif kind in ("fell", "deploy_lost"):
                if kind == "fell" and now - self._last_fell < 3.0:
                    return                                      # P1's robot_fell already raised it
                if kind == "fell":
                    self._last_fell = now
                self.sink.emit("safety_event", kind=kind, source="body", fault=msg.get("fault"))
            else:
                self.sink.emit("body_fault", kind=kind, session_id=msg.get("session_id"), hold=msg.get("hold"))
        elif t == "halted":
            src = str(msg.get("source") or "")
            ep = msg.get("epoch")
            if src == "watchdog" and isinstance(ep, int) and msg.get("kind") == "new":
                wire = int(ep)
                rt = self.body.runtime_epoch(wire) if hasattr(self.body, "runtime_epoch") else wire
                self.gate.halt(max(rt, self._epoch_now()))
                self.sink.emit("safety_event", kind="runtime_lost", source="body", reason=msg.get("reason"),
                               body_epoch=wire, epoch=self.gate.epoch, recoverable=True)
            else:
                self.sink.emit("body_event", topic="halted", epoch=ep, kind=msg.get("kind"), source=src,
                               handle_ms=msg.get("handle_ms"), arms_latched=msg.get("arms_latched"))
        elif t == "stale_command":
            self.sink.emit("stale_result", execution_id=msg.get("execution_id"), op=msg.get("op"),
                           why=msg.get("why"), reason=msg.get("reason"), source="body")
        else:
            brief = {k: msg.get(k) for k in ("event", "reason", "epoch", "was_latched", "session", "source")
                     if msg.get(k) is not None}
            lease = msg.get("lease")
            if isinstance(lease, dict):
                brief["owner"] = lease.get("owner")
                brief["mode"] = lease.get("mode")
            self.sink.emit("body_event", topic=t, **brief)

    def active_executions(self) -> list[Execution]:
        return list(self._active.values())

    async def shutdown(self) -> None:
        self.monitor.stop()
        self.halt_resender.cancel()
        for h in list(self._handles.values()):
            h.cancel("shutdown")
        for h in list(self._handles.values()):
            try:
                await asyncio.wait_for(h.result(), 3.0)
            except Exception:  # noqa: BLE001
                pass
        for ex in self.executors.values():
            close = getattr(ex, "close", None)
            if callable(close):
                try:
                    res = close()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:  # noqa: BLE001
                    pass
        close = getattr(self.body, "close", None)
        if callable(close):
            close()

    # ================================================================== validation and prompt helpers
    def capabilities(self) -> dict[str, ServiceHealth]:
        """Service health with the sim's real-time verdict applied (robot/health.py CapabilityPolicy)."""
        return self.policy.capabilities()

    def registry(self):
        return self.skill_registry

    def timeout_s(self, tool: str, args: dict) -> float:
        if tool == "navigate":
            if args.get("location") == "reach_stance" or args.get("stance"):
                return self.nav.reposition_timeout_s() + 5.0
            t = self.nav.timeout_s(self.world.robot_pose(), self.nav.resolve(str(args.get("location", ""))))
            if args.get("timeout_s"):
                t = min(t, float(args["timeout_s"]))
            return t
        if tool == "manipulate":
            action, otype = str(args.get("action") or "pick"), str(args.get("object_type"))
            arm = args.get("arm") if args.get("arm") in ARMS else None
            s = self.skill_registry.select(action, otype, arm)
            if s is None:
                return 20.0
            fb = self.manip.fallback_for(s, action, otype, arm)       # groot_then_script: room for the fallback
            return s.timeout_s() + (fb.max_duration_s + self.manip.cfg.verify_hold_s if fb is not None else 0.0)
        if tool == "check_reachability":
            return 12.0
        if tool in ("observe", "look"):
            return 30.0
        if tool == "wait_and_observe":
            return float(args.get("timeout_s") or 10.0) + 30.0
        if tool == "speak":
            return 0.4 * max(1, len(str(args.get("text", "")).split())) + 4.0
        return 10.0

    def observation_id(self) -> str:
        return self.obs.glance_record()

    def events(self) -> asyncio.Queue:
        """A new subscriber queue of robot events. Subscribing (inside the runtime's loop) also starts the
        capability monitor, the capability_changed producer (robot/health.py)."""
        q = self.sink.subscribe()
        try:
            loop = asyncio.get_running_loop()
            self.monitor.start()
            attach = getattr(self.body, "attach_events", None)
            if callable(attach) and not self._body_events_attached:
                attach(self._on_body_topic, loop)
                self._body_events_attached = True
        except RuntimeError:
            pass
        return q
