"""The page's ground-truth panels on the REAL world model (world.lite_world.LiteWorld over the recorded
procthor-train-40), with only the robot facade and the runtime faked: the adapters in ui/truth.py must read
world/'s actual TruthSnapshot, StaticMap.occupancy and Room objects."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOUSE = ROOT / "tests" / "fakes" / "houses" / "procthor-train-40"


@pytest.fixture(scope="module")
def world():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    if not (HOUSE / "occupancy.npz").exists():
        pytest.skip("no recorded house")
    try:
        from world.lite_world import LiteWorld
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"world/ not importable: {e}")
    return LiteWorld(HOUSE)


class WorldRobot:
    """A facade stub that only answers from the world model (the real one lives in robot/)."""

    def __init__(self, world):
        self.world = world

    def lookup_keypoints(self):
        return self.world.lookup_keypoints()

    def base_state(self):
        p = self.world.robot_pose()
        return {"moving": False, "at": "start", "between": None, "xy": [p.x, p.y]}

    def telemetry(self):
        return {"body": {"mode": "HOLD"}}

    def active_executions(self):
        return []

    async def shutdown(self):
        pass


def test_session_on_the_real_world_model(world):
    from ui.server import Session
    from ui_fakes import make_deps

    deps = make_deps()
    deps.build = lambda profile, scene, clock, log: (world, WorldRobot(world), None)

    async def go():
        s = Session("procthor-train-40", "agent", "gemini-3.8-flash", "lite", deps)
        await s.start()
        init = s.init_message()
        frame = s.frame()
        await s.stop()
        return init, frame

    init, frame = asyncio.run(go())
    json.dumps(init)
    lay = init["layout"]
    assert lay["user_surface"] == "kitchen_counter_1a"
    assert set(lay["rooms"]) == {"bedroom", "kitchen", "living_room"}
    assert len(lay["rooms"]["bedroom"]["polygon"]) >= 4
    assert lay["occupancy"]["extent"] == [-0.3, -0.3, 9.0, 9.0] and lay["occupancy"]["resolution"] == 0.05
    assert 400 < len(lay["grid"]) < 600, "about 29 m2 of nav cells at 0.25 m"
    assert "fridge_1" in lay["landmarks"] and lay["landmarks"]["fridge_1"]["x"] is not None
    assert {"alarm_clock", "book", "dresser"} <= set(lay["things"]) and "banana" not in lay["things"]
    assert lay["surfaces"]["bedroom_dresser_1b"]["height"] == 0.97
    t = init["truth"]
    assert t["source"] == "lite-gt" and t["label"].startswith("sim ground truth")
    assert t["objects"]["alarm_clock_1"]["where"] == "bedroom_dresser_1b"
    assert (t["robot"]["x"], t["robot"]["z"]) == (5.75, 4.25)
    assert t["robot"]["yaw"] == pytest.approx(180.0), "spawn yaw -90 deg (Isaac) faces -y: Worldline 180"
    assert t["arms"] == {"left": {"holding": None, "phase": "free"}, "right": {"holding": None, "phase": "free"}}
    assert frame["robot_map"]["explored"], "the first pose covers floor in front of the robot"
    assert frame["view"]["yaw"] == pytest.approx(180.0)
