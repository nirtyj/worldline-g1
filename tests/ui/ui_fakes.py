"""Offline stand-ins for the page's tests: a world model, a robot facade, a runtime, a planner
and a viz FrameTap. No simulator, no LLM, no network beyond 127.0.0.1.

The shapes follow PLAN 5-6 (robot facade: lookup_keypoints/base_state/telemetry/
active_executions/capabilities/halt/estop; world model: static_map/truth/latest_frame).
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any

# Stand-in JPEG payloads (the page forwards bytes; nothing here decodes them).
JPEG_A = b"\xff\xd8\xff\xe0 fake head frame A \xff\xd9"
JPEG_B = b"\xff\xd8\xff\xe0 fake head frame B \xff\xd9"


class FakeOccupancy:
    """scenes.occupancy.Occupancy's shape: 4 m x 3 m, 0.05 m cells, a wall down the middle with a door."""

    def __init__(self) -> None:
        self.resolution = 0.05
        self.origin = (0.0, 0.0)
        ny, nx = 60, 80
        raw = [[0] * nx for _ in range(ny)]
        for iy in range(ny):
            for ix in range(nx):
                if ix in (0, nx - 1) or iy in (0, ny - 1):
                    raw[iy][ix] = 2                     # outside
                elif 39 <= ix <= 40 and not (25 <= iy <= 34):
                    raw[iy][ix] = 1                     # wall with a 0.5 m door
        infl = [[0] * nx for _ in range(ny)]
        r = 6                                           # 0.30 m
        for iy in range(ny):
            for ix in range(nx):
                if raw[iy][ix]:
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            y, x = iy + dy, ix + dx
                            if 0 <= y < ny and 0 <= x < nx and dx * dx + dy * dy <= r * r:
                                infl[y][x] = 1
        self.raw, self.inflated, self.inflated_low = raw, infl, infl
        self.method = "fake"


@dataclass
class FakeStaticMap:
    occupancy: Any
    rooms: dict[str, Any]


class FakeWorld:
    scene = "procthor-train-40"

    def __init__(self) -> None:
        self.occ = FakeOccupancy()
        self.pose = [1.0, 1.5, 90.0]                    # x, z, yaw (Worldline frame)
        self.objects = {
            "alarm_clock_1": {"type": "alarm_clock", "label": "alarm clock", "where": "bedroom_dresser_1a",
                              "pos": [0.8, 2.6, 1.05], "visible": True},
            "book_1": {"type": "book", "label": "book", "where": "bedroom_bed_1a", "pos": [0.5, 0.5, 0.55]},
        }
        self.held: dict[str, str | None] = {"left": None, "right": None}
        self.closed = False

    def static_map(self) -> FakeStaticMap:
        return FakeStaticMap(self.occ, {
            "bedroom": {"label": "bedroom", "polygon": [[0, 0], [2, 0], [2, 3], [0, 3]], "center": [1.0, 1.5]},
            "kitchen": {"label": "kitchen", "polygon": [[2, 0], [4, 0], [4, 3], [2, 3]]},
        })

    def truth(self) -> dict[str, Any]:
        return {"source": "isaac-gt", "t": 12.5,
                "robot": {"x": self.pose[0], "z": self.pose[1], "yaw": self.pose[2], "pelvis_z": 0.78, "upright": True},
                "objects": dict(self.objects), "hands": dict(self.held)}

    def latest_frame(self, camera: str) -> Any:
        return None

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeExecution:
    execution_id: str
    tool_name: str
    args: dict
    generation: int = 1
    control_epoch: int = 0
    source: str = "brain"
    status: str = "running"
    data: dict = field(default_factory=dict)
    executor: str | None = None
    action: str | None = None
    t_created: float = 0.0
    t_started: float | None = 0.5

    @property
    def tool(self) -> str:
        return self.tool_name

    @property
    def t_start(self) -> float:
        return self.t_started if self.t_started is not None else self.t_created

    @property
    def finished(self) -> bool:
        return self.status not in ("queued", "running", "cancelling")


class FakeRobot:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        self.execs: list[FakeExecution] = []
        self.halts = 0
        self.estops: list[str] = []
        self.shut = False

    def lookup_keypoints(self) -> dict[str, Any]:
        return {
            "scene": "procthor-train-40",
            "keypoints": {"start": {"desc": "start", "xy": [1.0, 1.5], "room": "bedroom"},
                          "bedroom_dresser_1a": {"desc": "dresser 1, part a in the bedroom", "xy": [0.8, 2.3],
                                                 "room": "bedroom", "yaw": 0.0},
                          "kitchen_dining_table_1b": {"desc": "dining table 1, part b in the kitchen", "xy": [3.0, 1.0],
                                                      "room": "kitchen", "yaw": 180.0}},
            "edges": [["start", "bedroom_dresser_1a", 1.2]],
            "surfaces": {"bedroom_dresser_1a": {"keypoints": ["bedroom_dresser_1a"], "desc": "dresser 1a",
                                                "height_m": 0.97, "xy": [0.8, 2.7]},
                         "kitchen_dining_table_1b": {"keypoints": ["kitchen_dining_table_1b"], "desc": "table 1b",
                                                     "height_m": 0.78, "xy": [3.0, 0.6]}},
            "people": {"user": {"deliver_to_surface": "kitchen_dining_table_1b", "keypoint": "kitchen_dining_table_1b"}},
            "rooms": {"bedroom": {"label": "bedroom", "spots": ["start", "bedroom_dresser_1a"]},
                      "kitchen": {"label": "kitchen", "spots": ["kitchen_dining_table_1b"]}},
            "nav_speed_mps": 0.4, "robot": "unitree_g1", "profile": "sonic",
        }

    def base_state(self) -> dict[str, Any]:
        return {"moving": False, "at": "start", "between": None, "xy": [1.0, 1.5]}

    def telemetry(self) -> dict[str, Any]:
        return {"pose": {"x": self.world.pose[0], "z": self.world.pose[1], "yaw": self.world.pose[2], "horizon": 15.0,
                         "at": "start", "between": None},
                "moving": False, "health": {"ok": True, "source": "fake"},
                "body": {"mode": "HOLD", "lease": None, "upright": True, "rtf": 1.01, "carry": None,
                         "nav": {"route": [[1.0, 1.5], [2.0, 1.5]]}}}

    def capabilities(self) -> dict[str, Any]:
        return {"navigation": {"ok": True, "state": "ok"}, "manipulation": {"ok": True, "state": "ok"}}

    def active_executions(self) -> list[FakeExecution]:
        return [e for e in self.execs if not e.finished]

    def halt(self) -> dict[str, Any]:
        self.halts += 1
        return {"accepted": True, "stopped": True, "at_rest": True, "mode": "HOLD", "source": "fake"}

    def estop(self, reason: str) -> dict[str, Any]:
        self.estops.append(reason)
        return {"accepted": True, "mode": "ESTOP"}

    async def shutdown(self) -> None:
        self.shut = True


class FakeTracer:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []


class FakeTask:
    intent_version = 1
    control_epoch = 0
    paused = False


class FakeBelief:
    def to_dict(self) -> dict[str, Any]:
        return {"robot": {"at": "start"}, "hands": {"left": {"holding": None, "verified": True},
                                                    "right": {"holding": None, "verified": True}},
                "objects": {"alarm_clock_1": {"type": "alarm_clock", "where": "bedroom_dresser_1a", "verified": True,
                                              "source": "look", "t": 3.0}},
                "landmarks": {}, "looked": {"bedroom_dresser_1a": {"t": 3.0}}}


class FakeRuntime:
    """Just enough of agent.harness.Runtime for the page: it hears the user, keeps a history
    of executions and a trace, and answers the System 1 context."""

    def __init__(self, robot: FakeRobot, user: Any, brain: Any, clock: Any) -> None:
        self.robot, self.user, self.brain, self.clock = robot, user, brain, clock
        self.tracer = FakeTracer()
        self.task = FakeTask()
        self.belief = FakeBelief()
        self.history: list[FakeExecution] = []
        self.heard: list[Any] = []
        self.stops: list[tuple[str, str]] = []
        self.persona_level = None
        self.step_mode = False
        self._ids = itertools.count(1)

    async def run(self) -> None:
        while True:
            utt = await self.user.next()
            self.heard.append(utt)
            self.tracer.rows.append({"t": self.clock.now(), "type": "classified", "id": utt.id, "kind": "request",
                                     "directive": utt.directive})
            e = FakeExecution(f"nav-{next(self._ids):06d}", "navigate", {"location": "bedroom_dresser_1a"},
                              executor="sonic_walk", action="keypoint")
            self.history.append(e)
            self.robot.execs.append(e)
            self.tracer.rows.append({"t": self.clock.now(), "type": "started", "tool": "navigate",
                                     "args": e.args, "action": "keypoint", "version": 1})

    def emergency_stop(self, text: str, source: str = "keyword") -> None:
        self.stops.append((text, source))
        self.robot.halt()
        self.tracer.rows.append({"t": self.clock.now(), "type": "stop", "reason": source, "canceled": [], "cut": []})

    def system1_context(self) -> dict[str, Any]:
        return {"at": "start", "moving": False, "running": [e.tool for e in self.history if not e.finished]}

    def add_observations(self, items: list, source: str = "system1") -> None:
        pass

    def set_persona(self, level: str) -> None:
        self.persona_level = level

    def set_step_mode(self, on: bool) -> None:
        self.step_mode = on

    def stepping(self) -> None:
        return None

    def step(self) -> None:
        self.tracer.rows.append({"t": self.clock.now(), "type": "step_released"})


class FakePlanner:
    def stats(self) -> dict[str, Any]:
        return {"calls": 0}


class FakeTap:
    """viz.tap.FrameTap's read API with scripted frames."""

    def __init__(self) -> None:
        self.frames: dict[str, tuple[bytes, dict, int]] = {}

    def put(self, name: str, jpeg: bytes, meta: dict | None = None) -> None:
        rev = self.frames[name][2] + 1 if name in self.frames else 1
        self.frames[name] = (jpeg, dict(meta or {}), rev)

    def streams(self) -> list[str]:
        return sorted(self.frames)

    def rev(self, name: str) -> int:
        return self.frames[name][2] if name in self.frames else 0

    def jpeg(self, name: str) -> bytes | None:
        return self.frames[name][0] if name in self.frames else None

    def meta(self, name: str) -> dict:
        return dict(self.frames[name][1]) if name in self.frames else {}

    def age_s(self, name: str) -> float | None:
        return 0.1 if name in self.frames else None

    def close(self) -> None:
        pass


class FakeClock:
    def __init__(self, speed: float = 1.0) -> None:
        self.t = 0.0

    def now(self) -> float:
        self.t += 0.01
        return self.t

    async def sleep(self, s: float) -> None:
        await asyncio.sleep(0)


class FakeLog:
    def __init__(self, clock: Any) -> None:
        self.clock, self.events = clock, []

    def emit(self, type: str, **fields: Any) -> dict[str, Any]:
        e = {"t": round(self.clock.now(), 3), "type": type, **fields}
        self.events.append(e)
        return e


def make_deps(built: dict[str, Any] | None = None) -> Any:
    """ui.server.Deps wired to the fakes. `built` collects what the fake factory made."""
    from ui.server import Deps

    built = built if built is not None else {}

    def build(profile: str, scene: str, clock: Any, log: Any) -> tuple[Any, Any, Any]:
        w = FakeWorld()
        r = FakeRobot(w)
        built.update(world=w, robot=r, profile=profile, scene=scene)
        return w, r, None

    def create_runtime(spec: str, robot: Any, user: Any, brain: Any, clock: Any) -> FakeRuntime:
        rt = FakeRuntime(robot, user, brain, clock)
        built["runtime"] = rt
        return rt

    @dataclass
    class Info:
        scenario_id: str
        map: dict
        meanings: Any
        options: dict
        clock: Any

    return Deps(build=build, create_planner=lambda info: FakePlanner(), create_runtime=create_runtime,
                brain_info=Info, composite=lambda planner: planner, clock=FakeClock, event_log=FakeLog,
                is_stop=lambda text: text.strip().lower() in ("stop", "stop!"),
                frame_gate=lambda: None)
