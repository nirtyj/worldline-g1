"""A minimal in-process robot for runtime tests: implements api.services.RobotBridge.

Not the world agent's lite world (robot/, world/): just enough house, physics and
ground truth to drive agent/harness.py through the new API. Durations are in sim
seconds; tests run a fast SimClock.

    robot = FakeRobot(clock)
    user = FakeUser(clock)
    rt = Runtime(robot, user, PolicyBrain("alarm_clock"), clock)
"""

from __future__ import annotations

import asyncio
import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from api.execution import Execution, Rejected, ResultHandle
from api.observation import scan_views
from api.results import (ListLocationsResult, ManipulationResult, NamedLocation, NavigateResult,
                         ReachabilityResult, finish)
from api.skills import SkillSpec, StaticSkillRegistry
from api.tools import ToolCall
from api.types import PROFILES, ServiceHealth

PICKUPABLE = ("alarm_clock", "apple", "banana", "book", "bottle", "mug", "spatula")


def house_map() -> dict[str, Any]:
    kps = {
        "start": {"desc": "where the robot started", "xy": [0.0, 0.0], "room": "living_room", "kind": "start"},
        "living_room": {"desc": "the middle of the living room", "xy": [0.5, 0.5], "room": "living_room",
                        "kind": "room"},
        "living_room_table_1a": {"desc": "in front of the table", "xy": [1.0, 0.0], "room": "living_room",
                                 "kind": "surface"},
        "kitchen": {"desc": "the middle of the kitchen", "xy": [4.0, 0.5], "room": "kitchen", "kind": "room"},
        "kitchen_counter_1a": {"desc": "left end of the counter", "xy": [4.0, 1.0], "room": "kitchen",
                               "kind": "surface"},
        "kitchen_counter_1b": {"desc": "right end of the counter", "xy": [5.0, 1.0], "room": "kitchen",
                               "kind": "surface"},
        "bedroom": {"desc": "the middle of the bedroom", "xy": [0.5, 4.0], "room": "bedroom", "kind": "room"},
        "bedroom_dresser_1a": {"desc": "in front of the dresser", "xy": [0.0, 5.0], "room": "bedroom",
                               "kind": "surface"},
    }
    edges = []
    for a, b in itertools.combinations(kps, 2):
        edges.append([a, b, round(math.dist(kps[a]["xy"], kps[b]["xy"]) * 1.2, 2)])
    return {
        "scene": "fake-house@a",
        "keypoints": kps,
        "edges": edges,
        # keypoint == surface stretch (THOR and mapgen convention, PLAN 6.2.1 / test_mapgen)
        "surfaces": {
            "living_room_table_1a": {"keypoints": ["living_room_table_1a"], "desc": "the living room table",
                                     "height_m": 0.62, "xy": [1.0, -0.6]},
            "kitchen_counter_1a": {"keypoints": ["kitchen_counter_1a"], "desc": "the kitchen counter, left end",
                                   "height_m": 0.9, "xy": [4.0, 1.6]},
            "kitchen_counter_1b": {"keypoints": ["kitchen_counter_1b"], "desc": "the kitchen counter, right end",
                                   "height_m": 0.9, "xy": [5.0, 1.6]},
            "bedroom_dresser_1a": {"keypoints": ["bedroom_dresser_1a"], "desc": "the dresser", "height_m": 0.85,
                                   "xy": [0.0, 5.6]},
        },
        "people": {"user": {"deliver_to_surface": "living_room_table_1a", "keypoint": "living_room_table_1a"}},
        "rooms": {"living_room": {"label": "living room", "spots": ["start", "living_room_table_1a"],
                                  "keypoint": "living_room"},
                  "kitchen": {"label": "kitchen", "spots": ["kitchen_counter_1a", "kitchen_counter_1b"],
                              "keypoint": "kitchen"},
                  "bedroom": {"label": "bedroom", "spots": ["bedroom_dresser_1a"], "keypoint": "bedroom"}},
        "nav_speed_mps": 0.45, "max_reach_height_m": 1.2, "min_reach_height_m": 0.55,
        "robot": "unitree_g1", "profile": "lite",
        "executors": {"navigate": "lite", "manipulate": "lite"},
    }


@dataclass
class Obj:
    id: str
    type: str
    where: str                         # surface or hand:<arm>
    pos: tuple[float, float, float]
    needs_reposition: bool = False


@dataclass
class Utterance:
    id: str
    text: str
    t_end: float
    directive: dict[str, Any] | None = None


class FakeUser:
    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.q: asyncio.Queue = asyncio.Queue()
        self._ids = itertools.count(1)

    def say(self, text: str, directive: dict[str, Any] | None = None) -> Utterance:
        u = Utterance(f"u{next(self._ids)}", text, self.clock.now(), directive)
        self.q.put_nowait(u)
        return u

    async def next(self) -> Utterance:
        return await self.q.get()


class FakeRobot:
    """RobotBridge over a dict world. Fault injection: ``blocked``, ``fail_picks``, ``hang``,
    ``policy_down``, ``nav_s`` (seconds per metre), ``ignore_cancel``."""

    def __init__(self, clock: Any, *, nav_s_per_m: float = 0.5, speech_s_per_word: float = 0.05) -> None:
        self.clock = clock
        self.profile = PROFILES["lite"]
        self.map = house_map()
        self.at: str | None = "start"
        self.between: list[str] | None = None
        self.xy = tuple(self.map["keypoints"]["start"]["xy"])
        self.moving = False
        self.in_stance = False
        self.hands: dict[str, str | None] = {"left": None, "right": None}
        self.objects: dict[str, Obj] = {
            "alarm_clock_1": Obj("alarm_clock_1", "alarm_clock", "bedroom_dresser_1a", (0.1, 5.6, 0.9)),
            "apple_1": Obj("apple_1", "apple", "kitchen_counter_1a", (4.2, 1.6, 0.95)),
            "book_1": Obj("book_1", "book", "kitchen_counter_1b", (4.8, 1.6, 0.93)),
        }
        self.nav_s_per_m = nav_s_per_m
        self.speech_s_per_word = speech_s_per_word
        self.halt_epoch = 0
        self.halts: list[float] = []
        self.halt_epochs: list[int | None] = []
        self.resumes: list[int] = []
        self.estops: list[str] = []
        self.started: list[Execution] = []
        self.said: list[str] = []
        self.blocked: set[str] = set()
        self.fail_picks = 0
        self.ignore_cancel = False
        self.policy_down = False
        self.rev = 0
        self._active: dict[str, tuple[Execution, ResultHandle, asyncio.Task | None]] = {}
        self._queues: list[asyncio.Queue] = []
        self._registry = StaticSkillRegistry(
            [SkillSpec("lite.pick.v0", "pick", ("@vocab:pickupable",), backend="lite", label="stepping_stone",
                       max_duration_s=6.0),
             SkillSpec("lite.place.v0", "place", ("@vocab:pickupable",), backend="lite", label="stepping_stone",
                       max_duration_s=5.0)],
            vocab=PICKUPABLE, backend_order=("lite",),
            health_fn=lambda s: ServiceHealth(not self.policy_down, "down" if self.policy_down else "ok",
                                              "policy server not answering" if self.policy_down else ""))

    # ---------------- read surface ----------------
    def lookup_keypoints(self) -> dict[str, Any]:
        return self.map

    def memory(self) -> list[dict[str, Any]]:
        return []

    def base_state(self) -> dict[str, Any]:
        return {"moving": self.moving, "at": None if self.moving else self.at, "between": self.between,
                "xy": list(self.xy)}

    def gripper(self, arm: str) -> dict[str, Any]:
        held = self.hands[arm] is not None
        return {"closed": held, "width": 0.06 if held else 0.085, "force": 8.0 if held else 0.0}

    def proprio(self, arm: str) -> dict[str, Any]:
        return {"joints": [0.0] * 7, "gripper": "closed" if self.hands[arm] else "open", "posture": "home"}

    def telemetry(self) -> dict[str, Any]:
        return {"pose": {"x": self.xy[0], "z": self.xy[1], "yaw": 0.0, "horizon": 15.0,
                         "at": None if self.moving else self.at, "between": self.between},
                "velocity": {"linear_mps": 0.45 if self.moving else 0.0, "angular_dps": 0.0},
                "moving": self.moving, "arms": {a: self.proprio(a) for a in ("left", "right")},
                "grippers": {a: self.gripper(a) for a in ("left", "right")},
                "active_skills": [{"id": k, "skill": v[0].tool_name, "status": v[0].status}
                                  for k, v in self._active.items()],
                "health": {"ok": True, "source": "fake"},
                "body": {"mode": "LOCOMOTION" if self.moving else "HOLD", "lease": None, "upright": True,
                         "rtf": 1.0, "carry": any(self.hands.values()), "halt_epoch": self.halt_epoch}}

    def perception(self) -> dict[str, Any]:
        return {"objects": {}, "landmarks": {}, "people": {}, "source": "fake"}

    def truth(self) -> dict[str, Any]:
        return {"objects": {k: {"type": o.type, "where": o.where} for k, o in self.objects.items()},
                "robot": {"at": self.at, "hands": dict(self.hands)}}

    # ---------------- contract extras ----------------
    def capabilities(self) -> dict[str, ServiceHealth]:
        return {"navigation": ServiceHealth(True, "ok"), "manipulation": ServiceHealth(True, "ok"),
                "observation": ServiceHealth(True, "ok"), "speech": ServiceHealth(True, "ok"),
                "body": ServiceHealth(True, "ok")}

    def registry(self) -> StaticSkillRegistry:
        return self._registry

    def timeout_s(self, tool: str, args: dict[str, Any]) -> float:
        if tool == "navigate":
            if args.get("location") == "reach_stance":
                return 8.0
            d = self._dist(self.at, args.get("location"))
            return max(5.0, 1.8 * d * self.nav_s_per_m + 3.0)
        if tool == "manipulate":
            s = self._registry.get(args.get("skill_id") or "")
            return s.timeout_s() if s else 15.0
        return 5.0

    def observation_id(self) -> str:
        return f"obs-g{self.rev}"

    def events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return q

    def emit(self, ev: dict[str, Any]) -> None:
        for q in self._queues:
            q.put_nowait(ev)

    def active_executions(self) -> list[Execution]:
        return [v[0] for v in self._active.values()]

    async def shutdown(self) -> None:
        for _, h, task in list(self._active.values()):
            if task is not None:
                task.cancel()

    # ---------------- control ----------------
    def halt(self, control_epoch: int | None = None) -> dict[str, Any]:
        self.halt_epoch += 1                        # the fake's own counter: running motions see it change
        self.halts.append(self.clock.now())
        self.halt_epochs.append(control_epoch)      # the control_epoch the harness fenced (one epoch space)
        self.moving = False
        return {"accepted": True, "stopped": True, "at_rest": True, "mode": "HOLD", "body_epoch": self.halt_epoch,
                "epoch": control_epoch, "source": "lite"}

    def resume(self, control_epoch: int) -> None:
        self.resumes.append(control_epoch)

    def estop(self, reason: str) -> dict[str, Any]:
        self.estops.append(reason)
        return self.halt()

    def start(self, execution: Execution) -> ResultHandle:
        tool = execution.tool_name
        if tool == "manipulate" and self.policy_down:
            raise Rejected("capability", "policy_unavailable", "policy unavailable: fake policy down")
        self.started.append(execution)
        h = ResultHandle(execution)
        runner = {"speak": self._speak, "list_locations": self._list_locations, "navigate": self._navigate,
                  "check_reachability": self._reach, "manipulate": self._manipulate,
                  "observe": self._observe, "look": self._observe}.get(tool)
        if runner is None:
            raise Rejected("capability", "unknown_tool", f"no service for {tool}")
        task = asyncio.get_running_loop().create_task(self._guard(execution, h, runner))
        self._active[execution.execution_id] = (execution, h, task)
        return h

    async def _guard(self, ex: Execution, h: ResultHandle, runner: Any) -> None:
        try:
            res = await runner(ex, h)
        except Exception as e:                       # I1: a crash still resolves
            res = finish(ex, "failed", {"reason": "internal_error", "detail": repr(e)}, t_end=self.clock.now(),
                         observation_id=self.observation_id())
        if res.observation_id is None:
            import dataclasses
            res = dataclasses.replace(res, observation_id=self.observation_id())
        h.resolve(res)
        self._active.pop(ex.execution_id, None)

    # ---------------- helpers ----------------
    def _dist(self, a: str | None, b: str | None) -> float:
        kps = self.map["keypoints"]
        if a not in kps or b not in kps:
            return 3.0
        return math.dist(kps[a]["xy"], kps[b]["xy"]) * 1.2

    def _served(self) -> list[str]:
        return [s for s, info in self.map["surfaces"].items() if self.at in info["keypoints"]]

    async def _sleep_or_stop(self, seconds: float, h: ResultHandle, epoch: int, body: bool = True) -> str | None:
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            if body and self.halt_epoch != epoch:
                return "halted"
            if h.cancel_requested and not self.ignore_cancel:
                return "cancelled"
            await self.clock.sleep(0.02)
        return None

    def _look(self, mode: str) -> dict[str, Any]:
        self.rev += 1
        surfaces: dict[str, list[dict[str, Any]]] = {}
        for s in self._served():
            surfaces[s] = [{"id": o.id, "type": o.type, "brand": None, "color": None, "label": o.type.replace("_", " "),
                            "surface": s, "pos": list(o.pos)} for o in self.objects.values() if o.where == s]
        views = []
        if mode == "scan" and self.at in self.map["keypoints"]:
            x, z = self.map["keypoints"][self.at]["xy"]
            views = [v.to_dict() for v in scan_views(x, z, 0.0, cam_h=1.35)]
            for v in views:          # wide views so the fake's surfaces count as covered
                v["range"] = 3.0
                v["fov"] = 360.0
                v["vfov"] = None
        return {"at": None if self.moving else self.at, "surfaces": surfaces, "landmarks": [], "views": views,
                "hands": dict(self.hands), "mode": mode, "observation_id": f"obs-{mode[0]}{self.rev}",
                "source": "lite-gt"}

    # ---------------- services ----------------
    async def _speak(self, ex: Execution, h: ResultHandle):
        text = str(ex.args.get("text", ""))
        dur = 0.1 + self.speech_s_per_word * len(text.split())
        stop = await self._sleep_or_stop(dur, h, self.halt_epoch, body=False)
        if stop:
            return finish(ex, "cancelled", {"reason": "cut", "played": 0.5, "utterance_id": ex.execution_id},
                          t_end=self.clock.now())
        self.said.append(text)
        return finish(ex, "succeeded", {"utterance_id": ex.execution_id, "status": "queued", "speech": "played"},
                      t_end=self.clock.now())

    async def _list_locations(self, ex: Execution, h: ResultHandle):
        q = str(ex.args.get("query") or "").lower()
        locs = []
        for k, info in self.map["keypoints"].items():
            if q and q not in k:
                continue
            locs.append(NamedLocation(k, round(self._dist(self.at, k), 2), "room" if info.get("kind") == "room"
                                      else ("start" if k == "start" else "surface"), room=info.get("room")))
        locs.sort(key=lambda x: x.distance_m or 0.0)
        return finish(ex, "succeeded", ListLocationsResult(locs), t_end=self.clock.now())

    async def _navigate(self, ex: Execution, h: ResultHandle):
        epoch = self.halt_epoch
        a = ex.args
        if ex.action == "reposition":
            self.moving = True
            stop = await self._sleep_or_stop(0.3, h, epoch)
            self.moving = False
            if stop:
                return finish(ex, "cancelled" if stop == "cancelled" else "failed",
                              NavigateResult(ex.execution_id, "cancelled", "reach_stance", stop, at=self.at,
                                             executor="lite", kind="reposition"), t_end=self.clock.now())
            self.in_stance = True
            return finish(ex, "succeeded", NavigateResult(ex.execution_id, "succeeded", "reach_stance", None,
                                                          at=self.at, executor="lite", kind="reposition",
                                                          walked_m=0.2), t_end=self.clock.now())
        to = a["location"]
        start = self.at
        d = self._dist(start, to)
        if to in self.blocked:
            return finish(ex, "failed", NavigateResult(ex.execution_id, "failed", to, "blocked", at=start,
                                                       blocked_edge=[start or "?", to], executor="lite"),
                          t_end=self.clock.now())
        self.moving, self.in_stance = True, False
        self.between = [start or "start", to]
        stop = await self._sleep_or_stop(d * self.nav_s_per_m, h, epoch)
        self.moving = False
        if stop:
            self.at = None
            status = "cancelled" if stop == "cancelled" else "failed"
            return finish(ex, status, NavigateResult(ex.execution_id, status, to, stop, at=None,
                                                     between=[start or "start", to], executor="lite",
                                                     path_len_m=d), t_end=self.clock.now())
        self.at, self.between = to, None
        self.xy = tuple(self.map["keypoints"][to]["xy"])
        return finish(ex, "succeeded", NavigateResult(ex.execution_id, "succeeded", to, None, at=to, executor="lite",
                                                      path_len_m=round(d, 2), walked_m=round(d, 2),
                                                      duration_s=round(d * self.nav_s_per_m, 2)),
                      t_end=self.clock.now())

    async def _reach(self, ex: Execution, h: ResultHandle):
        a = ex.args
        ot, cands = a["object_type"], list(a.get("candidates") or [])
        await self.clock.sleep(0.05)

        def res(reachable: bool, visible: bool, arm: str, reason: str | None, oid: str | None = None, **kw):
            return finish(ex, "succeeded", ReachabilityResult(reachable, visible, arm, reason, object_type=ot,
                                                              object_id=oid, at=self.at, skill_id="lite.pick.v0",
                                                              **kw), t_end=self.clock.now())
        if self.moving:
            return res(False, False, "none", "base_moving")
        objs = [self.objects[c] for c in cands if c in self.objects and self.objects[c].type == ot]
        if not objs:
            return res(False, False, "none", "not_found")
        served = set(self._served())
        o = next((x for x in objs if x.where in served), objs[0])
        if o.where.startswith("hand"):
            return res(False, True, "none", "in_hand", o.id)
        if o.where not in served:
            return res(False, False, "none", "not_seen_here", o.id)
        if any(self.hands.values()):
            return res(False, True, "none", "hand_full", o.id)
        if o.needs_reposition and not self.in_stance:
            return res(False, True, "none", "needs_reposition", o.id, suggest_location="reach_stance",
                       stance={"x": o.pos[0], "y": o.pos[1] - 0.5, "yaw": 1.57, "dx": 0.22, "dy": 0.0, "dyaw": 0.0},
                       distance_m=0.62)
        return res(True, True, "right", None, o.id, distance_m=0.41)

    async def _manipulate(self, ex: Execution, h: ResultHandle):
        epoch = self.halt_epoch
        a = ex.args
        arm, ot = a.get("arm") or "right", a["object_type"]
        oid = a.get("object_id")
        stop = await self._sleep_or_stop(1.0, h, epoch)

        def res(status: str, reason: str | None, holding: bool | None, **kw):
            return finish(ex, status, ManipulationResult(ex.execution_id, status, a.get("skill_id") or "lite.pick.v0",
                                                         ot, reason, action=ex.action or "pick", arm=arm,
                                                         object_id=oid, target=a.get("target"), holding=holding,
                                                         executor="lite", **kw), t_end=self.clock.now())
        if ex.action == "pick":
            if stop:
                return res("cancelled" if stop == "cancelled" else "failed", stop, False)
            if self.fail_picks > 0:
                self.fail_picks -= 1
                return res("failed", "grasp_failed", False)
            o = self.objects.get(oid or "")
            if o is None:
                return res("failed", "not_found", False)
            o.where = f"hand:{arm}"
            self.hands[arm] = o.id
            return res("succeeded", None, True)
        # place
        o = self.objects.get(oid or "")
        if stop:
            return res("cancelled" if stop == "cancelled" else "failed", stop, True)
        if o is None or self.hands.get(arm) != o.id:
            return res("failed", "nothing_in_hand", False)
        target = a.get("target")
        if target not in self._served():
            return res("failed", "target_not_here", True)
        s = self.map["surfaces"][target]
        o.where = target
        o.pos = (s["xy"][0], s["xy"][1], s["height_m"] + 0.05)
        self.hands[arm] = None
        return res("succeeded", None, False, surface=target)

    async def _observe(self, ex: Execution, h: ResultHandle):
        mode = ex.args.get("mode", "glance")
        await self.clock.sleep(0.2 if mode == "scan" else 0.02)
        if h.cancel_requested:
            return finish(ex, "cancelled", {"reason": "cancelled", "mode": mode}, t_end=self.clock.now())
        data = self._look(mode)
        return finish(ex, "succeeded", data, t_end=self.clock.now(), observation_id=data["observation_id"])


# ----------------------------------------------------------------------
# Brains for runtime tests
# ----------------------------------------------------------------------
# PolicyBrain now lives in brains/scripted.py (the offline planner stub); re-exported for the tests.
from brains.scripted import PolicyBrain  # noqa: E402,F401


class ScriptBrain:
    """Returns the given calls in order (then wait_and_observe(0)); classifies by keyword."""

    def __init__(self, calls: list[ToolCall | Any], kinds: dict[str, str] | None = None, delay: float = 0.0,
                 clock: Any = None) -> None:
        self.calls = list(calls)
        self.kinds = dict(kinds or {})
        self.contexts: list[Any] = []
        self.delay = delay
        self.clock = clock

    async def classify(self, utt: Any, ctx: Any) -> str:
        return self.kinds.get(utt.text, "request")

    async def next_action(self, ctx: Any) -> ToolCall:
        self.contexts.append(ctx)
        if self.delay and self.clock is not None:
            await self.clock.sleep(self.delay)
        if self.calls:
            c = self.calls.pop(0)
            return c(ctx) if callable(c) else c
        return ToolCall("wait_and_observe", {"timeout_s": 0})
