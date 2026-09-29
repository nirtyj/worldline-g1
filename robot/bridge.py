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
import copy
import math
import time
from typing import Any

from api.execution import Execution, Rejected
from api.results import ListLocationsResult, finish
from api.types import ServiceHealth
from world import coords

from services.common import EventSink, HaltGate, start_execution
from services.executors import GrootSonicExecutor, KinematicAttachExecutor, SonicArmScriptExecutor
from services.locations import LocationsService
from services.manipulation import ManipConfig, ManipulationService
from services.navigation import NavConfig, NavigationService
from services.observation import ObservationService, ScanConfig
from services.reachability import G1Workspace, ReachabilityModel
from services.skills import build_registry
from services.speech import SpeechService

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
        self.skill_registry = registry or build_registry(profile.manip_executors, world)
        manip_cfg = dict(g1.get("manipulation") or {})
        self.executors = self._executors(manip_cfg)
        self.reach = ReachabilityModel(world, G1Workspace.from_dict(ws), registry=self.skill_registry,
                                       observation=self.obs, nav=self.nav, body=body)
        self.manip = ManipulationService(world, self.skill_registry, self.reach, self.executors, clock,
                                         ManipConfig.from_dict(manip_cfg), nav=self.nav, observation=self.obs,
                                         gate=self.gate, events=self.sink)
        self.speech = SpeechService(clock, self.sink, observation_id=self.observation_id)
        self._active: dict[str, Execution] = {}
        self._handles: dict[str, Any] = {}
        self._map: dict | None = None

    def _executors(self, manip_cfg: dict) -> dict[str, Any]:
        pick = manip_cfg.get("pick_phases_s")
        place = manip_cfg.get("place_phases_s")
        out: dict[str, Any] = {}
        for name in self.stack_profile.manip_executors:
            if name in ("lite", "kinematic_attach"):
                ex = KinematicAttachExecutor(self.world, self.clock, pick_phases=pick, place_phases=place,
                                             gate=self.gate, name=name,
                                             attach_mode="kinematic" if name == "lite" else "follow")
                out[ex.backend] = ex
            elif name == "sonic_arm_script":
                out["sonic_arm_script"] = SonicArmScriptExecutor()
            elif name == "groot_sonic":
                out["groot"] = GrootSonicExecutor()
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
            m["camera"] = {"name": getattr(self.world.cam, "name", "head"),
                           "sim_added": bool(getattr(self.world.cam, "sim_added", False))}
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
        rtf = (st.get("gt_pose") or {}).get("rtf")
        if rtf is None and hasattr(self.world, "rtf"):
            rtf = self.world.rtf()
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
            "body": {"mode": st.get("mode", "HOLD"), "lease": None, "upright": not p.fallen,
                     "rtf": rtf, "carry": any(hands.values()), "halt_epoch": self.gate.epoch,
                     "latched": self.gate.latched, "executor": self.nav.executor,
                     "pelvis_z": round(p.pelvis_z, 3)},
        }

    def perception(self) -> dict:
        return self.obs.perception()

    # ================================================================== execution
    def start(self, execution: Execution):
        tool = execution.tool_name
        a = execution.args
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

    def halt(self) -> dict:
        """Latch HOLD within 30 ms (PLAN §5.6). Running body executions end failed(halted)."""
        t0 = time.perf_counter()
        epoch = self.gate.halt()
        receipt = dict(self.body.halt(epoch))
        receipt.setdefault("accepted", True)
        receipt["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        self.sink.emit("halted", epoch=epoch, stopped=receipt.get("stopped"))
        return receipt

    def resume(self, control_epoch: int | None = None) -> None:
        self.gate.resume(control_epoch)
        self.body.resume(self.gate.epoch)

    def estop(self, reason: str) -> dict:
        """Operator kill button only (any profile). The deploy exits on a real stack; P2 restart required."""
        epoch = self.gate.halt()
        res = dict(self.body.estop(reason))
        res["body_epoch"] = epoch
        self.sink.emit("safety_event", kind="estop", reason=reason)
        return res

    def active_executions(self) -> list[Execution]:
        return list(self._active.values())

    async def shutdown(self) -> None:
        for h in list(self._handles.values()):
            h.cancel("shutdown")
        for h in list(self._handles.values()):
            try:
                await asyncio.wait_for(h.result(), 3.0)
            except Exception:  # noqa: BLE001
                pass
        close = getattr(self.body, "close", None)
        if callable(close):
            close()

    # ================================================================== validation and prompt helpers
    def capabilities(self) -> dict[str, ServiceHealth]:
        return {"navigation": self.nav.health(), "manipulation": self.manip.health(),
                "observation": ServiceHealth(True, "ok", f"{self.world.source} {getattr(self.world.cam, 'name', '')}"),
                "speech": ServiceHealth(True, "ok"), "body": self.body.health()}

    def registry(self):
        return self.skill_registry

    def timeout_s(self, tool: str, args: dict) -> float:
        if tool == "navigate":
            if args.get("location") == "reach_stance" or args.get("stance"):
                return self.nav.cfg.reposition_timeout_s + 5.0
            t = self.nav.timeout_s(self.world.robot_pose(), self.nav.resolve(str(args.get("location", ""))))
            if args.get("timeout_s"):
                t = min(t, float(args["timeout_s"]))
            return t
        if tool == "manipulate":
            s = self.skill_registry.select(str(args.get("action") or "pick"), str(args.get("object_type")),
                                           args.get("arm") if args.get("arm") in ARMS else None)
            return s.timeout_s() if s else 20.0
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
        return self.sink.subscribe()
