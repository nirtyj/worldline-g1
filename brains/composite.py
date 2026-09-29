"""The planner behind the one Brain interface the runtime knows.

    classify     System 1's label when it gave a valid, confident one; otherwise the planner
    next_action  always the planner (System 2)

System 1's label travels with the utterance (its directive: kind, target,
confidence, source "system1"), so the runtime still sees one classify call and
doesn't know which one answered. It adds preview() for the page's step mode and
stats() for the page's header.
"""

from __future__ import annotations

from typing import Any

from .interface import KINDS, BrainInput, ToolCall

MIN_CONFIDENCE = 0.5           # below this, System 1's label is only a hint and the planner decides


class CompositeBrain:
    def __init__(self, planner: Any) -> None:
        self.planner = planner

    async def classify(self, utterance: Any, ctx: BrainInput) -> str:
        d = getattr(utterance, "directive", None) or {}
        try:
            confident = float(d.get("confidence", 0)) >= MIN_CONFIDENCE
        except (TypeError, ValueError):
            confident = False
        if d.get("source") == "system1" and d.get("kind") in KINDS and confident:
            return d["kind"]
        return await self.planner.classify(utterance, ctx)

    async def next_action(self, ctx: BrainInput) -> ToolCall:
        return await self.planner.next_action(ctx)

    def preview(self, ctx: BrainInput) -> str | None:
        """The text the planner would get for this context (the page's step mode shows it)."""
        fn = getattr(self.planner, "render", None)
        return fn(ctx) if callable(fn) else None

    def stats(self) -> dict[str, Any]:
        fn = getattr(self.planner, "stats", None)
        return fn() if callable(fn) else {}
