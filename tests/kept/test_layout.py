"""agent/layout.py: what is next to what, from the robot's map and the landmarks it has seen.

The fixture is Kitchen 10 (FloorPlan10), where "move the spatula to the other side of
the stove" went to the wrong counter: counter 1 runs along one wall as
fridge · toaster · counter_1b · counter_1a · stove · counter_2, facing the sink run.

    .venv-rt/bin/python -m pytest tests/kept/test_layout.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent import layout
from agent.harness import Runtime
from agent.model import ReferenceBrain
from agent.state import BeliefState, Fact, LandmarkBelief, ObjectBelief
from llmkit.brain import tool_schemas

STANDS = {"start": (0.0, -1.25), "counter_1a": (0.25, -0.5), "counter_1b": (0.25, 0.25), "counter_2": (0.25, -2.0),
          "counter_3a": (-1.0, -2.0), "counter_3b": (0.0, -0.25), "shelf_1": (-1.75, 2.25), "sink_basin_1": (0.0, -0.75)}
CENTRES = {"counter_1a": (0.95, -0.55), "counter_1b": (0.95, 0.15), "counter_2": (0.95, -2.05), "counter_3a": (-0.95, -1.17),
           "counter_3b": (-0.95, -0.13), "shelf_1": (-1.64, 2.64), "sink_basin_1": (-0.7, -0.65)}
MAP = {"keypoints": {k: {"xy": list(v)} for k, v in STANDS.items()},
       "surfaces": {k: {"xy": list(v), "keypoints": [k], "desc": f"the {k.replace('_', ' ')}"} for k, v in CENTRES.items()}}
SEEN = {"stove_1": {"label": "stove", "near": "start", "pos": [0.96, -1.3]},
        "toaster_1": {"label": "toaster", "near": "counter_1b", "pos": [0.98, 0.33]},
        "fridge_1": {"label": "fridge", "near": "counter_1b", "pos": [0.97, 1.25]}}


def names(line: layout.Line) -> list[str]:
    return [t.name for t in line.things]


class LayoutTest(unittest.TestCase):
    def test_the_counter_run_left_to_right_as_you_face_it(self):
        lines = layout.build(MAP, SEEN)
        self.assertEqual(names(lines[0]), ["fridge_1", "toaster_1", "counter_1b", "counter_1a", "stove_1", "counter_2"])
        self.assertEqual(names(lines[1]), ["sink_basin_1", "counter_3b"])     # counter_3a is faced from its end

    def test_the_other_side_of_the_stove(self):
        text = layout.render(layout.build(MAP, SEEN), MAP)
        self.assertIn("stove_1 (stove) is between counter_1a and counter_2", text)
        self.assertIn("line 1 and line 2 face each other", text)
        about = layout.about("counter_1a", layout.build(MAP, SEEN), MAP)
        self.assertIn("stove_1 (stove), then counter_2 on its right", about[0])

    def test_without_landmarks_only_the_surfaces(self):
        lines = layout.build(MAP, {})
        self.assertEqual(names(lines[0]), ["counter_1b", "counter_1a", "counter_2"])
        self.assertNotIn("stove", layout.render(lines, MAP))              # never seen: not placed

    def test_a_landmark_without_a_position_is_left_out(self):
        lines = layout.build(MAP, {"stove_1": {"label": "stove", "near": "start", "pos": None}})
        self.assertNotIn("stove_1", names(lines[0]))

    def test_rooms_keep_their_lines_apart(self):
        m = {"keypoints": {"a": {"xy": [0, 0], "room": "kitchen"}, "b": {"xy": [1, 0], "room": "kitchen"},
                           "c": {"xy": [2, 0], "room": "living"}},
             "surfaces": {k: {"xy": [x, 0.7], "keypoints": [k]} for k, x in (("a", 0), ("b", 1), ("c", 2))},
             "rooms": {"kitchen": {"label": "kitchen"}, "living": {"label": "living room"}}}
        lines = layout.build(m, {})
        self.assertEqual([names(l) for l in lines], [["a", "b"]])          # c is alone in its room
        self.assertIn("kitchen, line 1: a · b", layout.render(lines, m))

    def test_layout_questions_get_the_whole_layout(self):
        lines = layout.build(MAP, SEEN)
        out = layout.about("layout of counters and stove", lines, MAP)
        self.assertTrue(out[0].startswith("Layout"))
        self.assertTrue(any("between counter_1a and counter_2" in o for o in out))
        self.assertEqual(layout.about("mug", lines, MAP), [])

    def test_the_planner_gets_it(self):
        ctx = SimpleNamespace(map=MAP, belief={"landmarks": SEEN})
        self.assertIn("stove_1 (stove) is between counter_1a and counter_2", ReferenceBrain._layout(ctx))
        self.assertEqual(ReferenceBrain._layout(SimpleNamespace(map={"keypoints": {}, "surfaces": {}}, belief={})), "")



class RelationTest(unittest.TestCase):
    def setUp(self):
        self.lines = layout.build(MAP, SEEN)

    def want(self, text: str, origin: str | None = None):
        goal = layout.parse(text, self.lines, MAP, {"cup_1"})
        self.assertIsNotNone(goal, text)
        return layout.targets(goal, self.lines, MAP, origin)[0]

    def test_each_relation_on_kitchen_10(self):
        self.assertEqual(self.want("other side of stove_1 from counter_1a"), ["counter_2"])
        self.assertEqual(self.want("the other side of the stove", origin="counter_1a"), ["counter_2"])
        self.assertEqual(self.want("other side of stove_1", origin="counter_2"), ["counter_1a", "counter_1b"])
        self.assertEqual(self.want("left of counter_1a"), ["counter_1b"])
        self.assertEqual(self.want("right of the toaster"), ["counter_1b", "counter_1a", "counter_2"])
        self.assertEqual(self.want("next to stove_1"), ["counter_1a", "counter_2"])
        self.assertEqual(self.want("between counter_1b and counter_2"), ["counter_1a"])
        self.assertEqual(self.want("across from counter_1a"), ["sink_basin_1", "counter_3b"])
        self.assertEqual(self.want("on counter_2"), ["counter_2"])
        self.assertIsNone(self.want("in cup_1"))                           # can't put things in containers yet

    def test_what_it_cant_parse(self):
        for text in ("somewhere nice", "next to the microwave", "other side of stove_1 from the garden", "on"):
            self.assertIsNone(layout.parse(text, self.lines, MAP), text)

    def test_check(self):
        goal = layout.parse("other side of stove_1", self.lines, MAP)
        miss = layout.check(goal, "counter_1b", self.lines, MAP, origin="counter_1a")
        self.assertEqual((miss.ok, miss.expected), (False, ["counter_2"]))
        self.assertIn("counter_2 would be", miss.why)
        self.assertTrue(layout.check(goal, "counter_2", self.lines, MAP, origin="counter_1a").ok)
        self.assertIsNone(layout.check(goal, "counter_2", self.lines, MAP).ok)   # no side to start from

    def test_recall_answers_relations(self):
        self.assertIn("from counter_1a: counter_2", layout.about("what is on the other side of the stove", self.lines, MAP)[0])
        self.assertEqual(layout.about("left of counter_1a", self.lines, MAP)[0], "spots left of counter_1a: counter_1b")


class GoalCheckTest(unittest.TestCase):
    """Runtime._check_goal: where the object really is, against the relation the place asked for."""

    def runtime(self, where: str):
        b = BeliefState()
        b.objects["spatula_1"] = ObjectBelief(id="spatula_1", type="spatula", brand=None, color=None,
                                              where=Fact(where, "look", 10.0, True))
        for k, lm in SEEN.items():
            b.landmarks[k] = LandmarkBelief(k, lm["label"], Fact(lm["near"], "camera", 1.0, True), tuple(lm["pos"]))
        rows = []
        rt = SimpleNamespace(map=MAP, belief=b, _note=None, _goal_missed=set(), _picked_from={"spatula_1": "counter_1a"},
                             tracer=SimpleNamespace(log=lambda kind, **kw: rows.append({"type": kind, **kw})))
        rt._layout = lambda: Runtime._layout(rt)
        return rt, rows

    def test_a_miss_goes_back_to_the_planner(self):
        rt, rows = self.runtime("counter_1b")
        Runtime._check_goal(rt, "spatula_1", "other side of stove_1")
        self.assertEqual((rows[0]["ok"], rows[0]["expected"]), (False, ["counter_2"]))
        self.assertIn("don't say it's done", rt._note)
        self.assertIn("spatula_1", rt._goal_missed)                     # so it may pick it up again

    def test_a_hit_is_quiet(self):
        rt, rows = self.runtime("counter_2")
        rt._goal_missed.add("spatula_1")
        Runtime._check_goal(rt, "spatula_1", "other side of stove_1")
        self.assertEqual((rows[0]["ok"], rt._note, rt._goal_missed), (True, None, set()))

    def test_goal_is_optional_for_the_planner(self):
        # G1 port (PLAN 5.11): place is manipulate(action="place", ...); goal stays optional.
        ctx = SimpleNamespace(map={"keypoints": {"a": {}}}, belief={"objects": {"x_1": {}}})
        place = next(t for t in tool_schemas(ctx) if t["name"] == "manipulate")["parameters"]
        self.assertEqual((place["required"], "goal" in place["properties"]), (["action", "object_type"], True))


if __name__ == "__main__":
    unittest.main()
