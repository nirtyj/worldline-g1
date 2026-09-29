"""The chat's progress lines (agent/narrator.py) and the runtime's filter for System 1
observations that only repeat what it just confirmed (Runtime._echoes_belief).

    .venv-rt/bin/python -m pytest tests/kept/test_narrator.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.harness import Runtime
from agent.narrator import Narrator
from agent.state import BeliefState, Fact, ObjectBelief

MAP = {"keypoints": {"start": {}, "counter_1a": {}, "counter_1b": {}, "sink_basin_1": {}}}


def belief(**where: tuple[str, float, bool]) -> BeliefState:
    b = BeliefState()
    for oid, (w, t, verified) in where.items():
        b.objects[oid] = ObjectBelief(id=oid, type=oid.rsplit("_", 1)[0], brand=None, color=None,
                                      where=Fact(w, "look", t, verified))
    return b


def lines(n: Narrator) -> list[str]:
    return [text for _, text in n._out]


class NarratorTest(unittest.TestCase):
    def setUp(self):
        self.b = belief(spatula_1=("counter_1a", 90.0, True))
        self.n = Narrator(MAP, self.b, "sink_basin_1")

    def ask(self, text: str, target: str = "spatula") -> None:
        self.n.row({"type": "heard", "id": "u1", "text": text})
        self.n.row({"type": "classified", "id": "u1", "kind": "request", "directive": {"target": {"object": target}}})

    def test_putting_it_down_is_not_finding_it(self):
        self.ask("move the spatula to counter 1b")
        self.n.row({"type": "place_learned", "object": "spatula_1", "place": "hand:right", "was": "counter_1a"})
        self.n.row({"type": "place_learned", "object": "spatula_1", "place": "counter_1b", "was": "hand:right"})
        out = lines(self.n)
        self.assertIn("Got the spatula", out)
        self.assertIn("The spatula is on the counter now", out)
        self.assertFalse([l for l in out if l.startswith("Found")], out)

    def test_carrying_it_is_not_looking_for_it(self):
        self.ask("put the spatula on counter 1b")
        self.b.objects["spatula_1"].where = Fact("hand:right", "look", 95.0, True)
        self.n.row({"type": "started", "tool": "navigate", "args": {"location": "counter_1b"}})   # G1: to -> location
        self.assertIn("Heading to the counter, with the spatula", lines(self.n))

    def test_a_real_find_still_says_found(self):
        self.ask("bring me the spatula")
        self.n.row({"type": "place_learned", "object": "spatula_1", "place": "counter_1b", "was": None})
        self.assertIn("Found the spatula on the counter", lines(self.n))

    def test_delivering_leaves_it_to_the_delivered_line(self):
        self.ask("bring me the spatula")
        self.n.row({"type": "place_learned", "object": "spatula_1", "place": "sink_basin_1", "was": "hand:right"})
        self.assertFalse([l for l in lines(self.n) if "spatula is on" in l or l.startswith("Found")])

    def test_looking_for_only_when_it_doesnt_know_where(self):
        self.ask("move the spatual to the other side of the stove")
        self.assertIn("Working on it: “move the spatual to the other side of the stove”", lines(self.n))
        self.b.objects["spatula_1"].where = Fact("UNKNOWN", "look_absent", 100.0, False)
        self.ask("find the spatula")
        self.assertIn("Working on it: looking for the spatula", lines(self.n))
        self.ask("bring me a banana", target="banana")                 # never seen: nothing in belief
        self.assertIn("Working on it: looking for the banana", lines(self.n))


class EchoTest(unittest.TestCase):
    def echo(self, b: BeliefState, text: str, where: str | None, now: float) -> str | None:
        return Runtime._echoes_belief(SimpleNamespace(belief=b), text, where, now)

    def test_what_it_just_placed_is_an_echo(self):
        b = belief(spatula_1=("counter_1a", 101.7, True))
        self.assertEqual(self.echo(b, "the spatula is on the counter", "counter_1a", 102.5), "spatula_1")
        self.assertEqual(self.echo(b, "Spatulas on the counter", "counter_1a", 102.5), "spatula_1")

    def test_news_is_kept(self):
        b = belief(spatula_1=("counter_1a", 101.7, True), paper_towel_roll_1=("counter_1a", 60.0, True))
        self.assertIsNone(self.echo(b, "the spatula is on the counter", "counter_1b", 102.5))    # another spot
        self.assertIsNone(self.echo(b, "the spatula is on the counter", "counter_1a", 130.0))    # not recent
        self.assertIsNone(self.echo(b, "a paper towel roll on the counter", "counter_1a", 102.5))  # not just confirmed
        self.assertIsNone(self.echo(b, "the fridge door is open", "counter_1a", 102.5))          # no object named
        b.objects["spatula_1"].where = Fact("counter_1a", "skill", 101.7, False)
        self.assertIsNone(self.echo(b, "the spatula is on the counter", "counter_1a", 102.5))    # unverified

    def test_what_it_holds_is_an_echo(self):
        b = belief(spatula_1=("hand:right", 93.0, True))
        self.assertEqual(self.echo(b, "the robot is holding a spatula", "counter_1a", 94.0), "spatula_1")


if __name__ == "__main__":
    unittest.main()
