"""Skills: typed wrappers over the robot API, with timeouts, and the speech queue.

Every skill returns an Outcome(status, data). Statuses: SUCCEEDED, ABORTED,
CANCELED, TIMEOUT, REJECTED.

Pick and place are plain goals: the robot runs the grasp chunk by chunk and
honours cancel only between chunks (see thor/robot.py).
"""

from __future__ import annotations

import asyncio
import collections
import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from sim.goals import GoalRejected



@dataclass
class Outcome:
    status: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "SUCCEEDED"


# ----------------------------------------------------------------------
# Plain goals
# ----------------------------------------------------------------------
async def run_goal(robot: Any, clock: Any, skill: str, args: dict[str, Any], timeout: float,
                   on_goal=None) -> Outcome:
    """Send one goal and await it. On timeout: cancel, wait a moment, then halt."""
    try:
        goal = robot.send_goal(skill, **args)
    except GoalRejected as e:
        return Outcome("REJECTED", {"reason": e.reason})
    if on_goal is not None:
        on_goal(goal)
    try:
        res = await clock.wait_for(goal.result(), timeout)
        return Outcome(res.status, dict(res.data))
    except asyncio.TimeoutError:
        goal.cancel()
        try:
            res = await clock.wait_for(goal.result(), 1.5)
            return Outcome("TIMEOUT", {"reason": f"{skill} took longer than {timeout:.0f} s", **res.data})
        except asyncio.TimeoutError:
            robot.halt()
            res = await goal.result()
            return Outcome("TIMEOUT", {"reason": f"{skill} ignored cancel; halted", **res.data})


def path_length(map_: dict[str, Any], start: str | None, goal: str, blocked: set[tuple[str, str]]) -> float:
    if start is None:
        return 12.0
    if start == goal:
        return 0.0
    bad = {frozenset(e) for e in blocked}
    adj: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for a, b, length in map_["edges"]:
        if frozenset((a, b)) in bad:
            continue
        adj[a].append((b, length))
        adj[b].append((a, length))
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == goal:
            return d
        if d > dist.get(node, math.inf):
            continue
        for nxt, length in adj[node]:
            nd = d + length
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return 40.0     # no known path: the robot's planner may still find one


def navigate_timeout(map_: dict[str, Any], start: str | None, goal: str, blocked: set[tuple[str, str]]) -> float:
    # generous: a slow render stretches a step, and a timeout halts the robot mid-route
    return path_length(map_, start, goal, blocked) / float(map_["nav_speed_mps"]) * 2.0 + 10.0


# ----------------------------------------------------------------------
@dataclass
class SpeechItem:
    entry: Any                      # the HistoryEntry
    text: str
    created_for: int
    control_epoch: int = 0


class SpeechQueue:
    """FIFO speech. Lines are tagged with the intent version they serve, so a
    correction can drop the ones that no longer apply."""

    def __init__(self, robot: Any, clock: Any, on_change=None) -> None:
        self.robot = robot
        self.clock = clock
        self.items: collections.deque[SpeechItem] = collections.deque()
        self.current: tuple[SpeechItem, Any] | None = None
        self._evt = asyncio.Event()
        self._on_change = on_change

    def enqueue(self, item: SpeechItem) -> None:
        item.entry.status = "queued"
        self.items.append(item)
        self._evt.set()

    def enqueue_priority(self, item: SpeechItem) -> None:
        """Queue runtime safety speech ahead of ordinary planner speech."""
        item.entry.status = "queued"
        self.items.appendleft(item)
        self._evt.set()

    def busy(self) -> bool:
        return bool(self.items) or self.current is not None

    def drop_older_than(self, version: int) -> list[str]:
        dropped = []
        for item in list(self.items):
            if item.created_for < version:
                self.items.remove(item)
                item.entry.status = "DROPPED"
                item.entry.t_end = self.clock.now()
                dropped.append(item.text)
        if self.current and self.current[0].created_for < version:
            self.current[1].cancel()
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    def drop_before_epoch(self, epoch: int) -> list[str]:
        """Fence speech produced by planner calls from an older control epoch."""
        dropped = []
        for item in list(self.items):
            if item.control_epoch < epoch:
                self.items.remove(item)
                item.entry.status = "DROPPED"
                item.entry.t_end = self.clock.now()
                dropped.append(item.text)
        if self.current and self.current[0].control_epoch < epoch:
            self.current[1].cancel()
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    def cut_all(self) -> list[str]:
        dropped = []
        for item in list(self.items):
            item.entry.status = "DROPPED"
            item.entry.t_end = self.clock.now()
            dropped.append(item.text)
        self.items.clear()
        if self.current:
            self.current[1].cancel()
            dropped.append(f"(cut) {self.current[0].text}")
        return dropped

    async def run(self) -> None:
        while True:
            await self._evt.wait()
            self._evt.clear()
            while self.items:
                item = self.items.popleft()
                try:
                    goal = self.robot.send_goal("say", text=item.text)
                except GoalRejected as e:
                    item.entry.status = "REJECTED"
                    item.entry.data = {"reason": e.reason}
                    continue
                item.entry.status = "running"
                item.entry.t_start = self.clock.now()
                self.current = (item, goal)
                words = len(item.text.split())
                try:
                    res = await self.clock.wait_for(goal.result(), 0.4 * words + 4.0)
                    item.entry.status = res.status
                    item.entry.data = dict(res.data)
                except asyncio.TimeoutError:
                    goal.cancel()
                    item.entry.status = "TIMEOUT"
                item.entry.t_end = self.clock.now()
                self.current = None
                if self._on_change:
                    self._on_change()
