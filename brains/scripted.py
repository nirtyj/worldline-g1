"""Scripted planners: System 2 without a model, for offline runs, the page without API keys, and tests.

    PolicyBrain       fetch a fixed ``want`` and deliver it to the user, choosing each tool call from BELIEF and
                      the latest results only (what a model sees): acknowledge, search surfaces in map order,
                      navigate to the object's stand, check_reachability, reposition on ``needs_reposition``,
                      pick, navigate(user), place, say it is done. Gives up after two failed manipulations.
                      (Moved from tests/unit/fakes_rt.py, unchanged; the tests import it from there.)
    ScriptedPlanner   PolicyBrain whose ``want`` comes from the newest request: System 1's target when it gave
                      one, else "bring me the X". Its place names the goal ("on <user surface>"), so the runtime's
                      layout goal check runs. ``create(info)`` is the page's planner factory:

        SYSTEM1=tests.kept.system1_stub:create python -m ui.server --profile lite --planner brains.scripted:create

Nothing here reads the simulator: only ``BrainInput`` (belief, map, history, results), like a model.
"""

from __future__ import annotations

import re
from typing import Any

from api.tools import ToolCall


class PolicyBrain:
    """A tiny deterministic planner: fetch ``want`` and deliver it to the user, using only BELIEF
    and the latest results (what a model would see). Records every context it was given."""

    def __init__(self, want: str | None = "alarm_clock", classify_as: dict[str, str] | None = None) -> None:
        self.want = want
        self.classify_as = dict(classify_as or {})
        self.contexts: list[Any] = []
        self.calls: list[ToolCall] = []
        self.acked = False
        self.done_said = False
        self.searched: list[str] = []
        self.tried: list[str] = []
        self.failures = 0
        self.counted: set[Any] = set()
        self.absent: set[str] = set()

    async def classify(self, utt: Any, ctx: Any) -> str:
        t = utt.text.lower()
        if t in self.classify_as:
            return self.classify_as[t]
        if t.startswith("stop"):
            return "stop"
        if "carry on" in t or "go ahead" in t:
            return "resume"
        if t.startswith("no,") or "instead" in t:
            return "correction"
        if t.endswith("?"):
            return "question"
        return "request"

    def _call(self, tool: str, **args: Any) -> ToolCall:
        c = ToolCall(tool, args)
        self.calls.append(c)
        return c

    async def next_action(self, ctx: Any) -> ToolCall:
        self.contexts.append(ctx)
        return self.decide(ctx)

    def decide(self, ctx: Any) -> ToolCall:
        b = ctx.belief
        m = ctx.map
        user = m["people"]["user"]
        want = self.want
        if want is None or ctx.paused or ctx.own_goal:
            return self._call("wait_and_observe", timeout_s=0)
        if not self.acked:
            self.acked = True
            return self._call("speak", text="On it.")
        if any(getattr(e, "tool", None) in ("navigate", "manipulate") for e in (getattr(ctx, "active", None) or [])):
            # a body action is running (the harness asked for another reason, e.g. its arrival scan's result):
            # wait for it instead of spending the search list on body_busy rejections (live F1, main box)
            return self._call("wait_and_observe", timeout_s=0)
        hands = b["hands"]
        mine = [oid for oid, o in b["objects"].items() if o["type"] == want]
        if any(h["holding"] == "UNKNOWN" or not h["verified"] for h in hands.values()):
            return self._call("wait_and_observe", timeout_s=0)
        for oid in mine:
            o = b["objects"][oid]
            if o["where"] == user["deliver_to_surface"] and o["verified"]:
                if not self.done_said:
                    self.done_said = True
                    return self._call("speak", text="Here it is.")
                return self._call("wait_and_observe", timeout_s=0)
        held = next((oid for oid in mine if str(b["objects"][oid]["where"]).startswith("hand:")), None)
        at = b["robot"]["at"]
        last_manip = next((e for e in reversed(ctx.history) if e.tool == "manipulate" and e.status != "rejected"), None)
        if last_manip is not None and last_manip.status in ("failed", "timed_out") and \
                getattr(last_manip, "execution_id", None) not in self.counted:
            self.counted.add(getattr(last_manip, "execution_id", None))   # each failed manipulate counts once, however
            self.failures += 1                                            # often the planner is asked after it
            if self.failures >= 2:
                self.want = None                              # rule 7: tell the user, don't loop
                return self._call("speak", text=f"The {last_manip.action} didn't work ({last_manip.data.get('reason')}).")
        if held:
            if at != user["keypoint"]:
                return self._call("navigate", location="user")
            return self._call("manipulate", action="place", object_type=want, target="user")
        known = [oid for oid in mine if b["objects"][oid]["where"] in m["surfaces"]
                 and b["objects"][oid]["where"] not in self.absent]
        if known:
            oid = known[0]
            surf = b["objects"][oid]["where"]
            kps = m["surfaces"][surf]["keypoints"]
            if at not in kps:
                return self._call("navigate", location=kps[0])
            last = next((e for e in reversed(ctx.history) if e.tool not in ("speak", "list_locations", "recall")
                         and e.status != "rejected"), None)
            if last is not None and last.tool == "check_reachability" and last.status == "succeeded":
                d = last.data
                if d.get("reachable"):
                    return self._call("manipulate", action="pick", object_type=want)
                if d.get("reason") == "needs_reposition":
                    return self._call("navigate", location="reach_stance")
                if d.get("reason") in ("not_seen_here", "not_found"):
                    self.absent.add(surf)                     # stale belief (e.g. memory): keep searching
                    self.searched.append(surf)
                    return self._search(ctx) or self._give_up(want)
                sug = d.get("suggest_location")
                if sug and sug != "reach_stance" and sug != at and sug not in self.tried:
                    self.tried.append(sug)
                    return self._call("navigate", location=sug)
                self.want = None                              # give up once, then stay quiet
                return self._call("speak", text=f"I can't reach it ({d.get('reason')}).")
            return self._call("check_reachability", object_type=want)
        return self._search(ctx) or self._give_up(want)

    def _search(self, ctx: Any) -> ToolCall | None:
        """Surfaces not looked at yet (this session), in map order. A surface whose navigate was rejected (the
        body was busy, a validation rule) was not searched: it stays on the list."""
        b, m = ctx.belief, ctx.map
        last_nav: dict[str, str] = {}
        for e in getattr(ctx, "history", None) or []:
            if getattr(e, "tool", None) == "navigate":
                last_nav[str((e.args or {}).get("location"))] = str(e.status)
        self.searched = [x for x in self.searched
                         if last_nav.get((m["surfaces"].get(x) or {}).get("keypoints", [x])[0]) != "rejected"]
        for s, info in m["surfaces"].items():
            looked = b["looked"].get(s) or b["looked"].get(info["keypoints"][0])
            fresh = looked is not None and looked.get("source") != "memory"
            if not fresh and s not in self.searched:
                self.searched.append(s)
                return self._call("navigate", location=info["keypoints"][0])
        return None

    def _give_up(self, want: str) -> ToolCall:
        self.want = None
        return self._call("speak", text=f"I couldn't find the {want}.")


REQUEST_RE = re.compile(r"\b(?:bring|get|fetch|grab|find)\s+(?:me\s+)?(?:the|a|an|my)\s+([a-z][a-z \-]*?)[.!?]*$", re.I)
REQUEST_KINDS = ("request", "correction")


def type_of(phrase: str, vocabulary: tuple[str, ...] | list[str] = ()) -> str | None:
    """"alarm clock" -> "alarm_clock", checked against the session's object_type enum when there is one."""
    t = re.sub(r"[^a-z]+", "_", phrase.lower()).strip("_")
    if not t:
        return None
    if not vocabulary or t in vocabulary:
        return t
    if t.endswith("s") and t[:-1] in vocabulary:
        return t[:-1]
    return None


class ScriptedPlanner(PolicyBrain):
    def __init__(self, want: str | None = None, *, goal_on_place: bool = True) -> None:
        super().__init__(want)
        self.goal_on_place = goal_on_place
        self._taken: set[str] = set()                 # utterance ids already turned into a want

    def _new_want(self, ctx: Any) -> str | None:
        vocab = tuple(getattr(ctx.tools_ctx, "skill_types", ()) or ())
        for u in reversed(ctx.utterances or []):
            if u.id in self._taken:
                return None
            if ctx.kinds.get(u.id) not in REQUEST_KINDS:
                continue
            self._taken.add(u.id)
            target = ((getattr(u, "directive", None) or {}).get("target") or {}).get("object")
            m = REQUEST_RE.search(u.text.strip())
            for phrase in (target, m.group(1) if m else None):
                t = type_of(phrase or "", vocab)
                if t:
                    return t
            return None
        return None

    async def next_action(self, ctx: Any) -> ToolCall:
        want = self._new_want(ctx)
        if want is not None and want != self.want:
            self.want = want
            self.acked = self.done_said = False
            self.searched, self.tried, self.failures, self.absent = [], [], 0, set()
        call = await super().next_action(ctx)
        if (self.goal_on_place and call.tool == "manipulate" and call.args.get("action") == "place"
                and "goal" not in call.args):
            surface = ((ctx.map.get("people") or {}).get("user") or {}).get("deliver_to_surface")
            if surface:
                call.args = {**call.args, "goal": f"on {surface}"}
        return call

    def stats(self) -> dict[str, Any]:
        return {"planner": "scripted", "calls": len(self.calls), "want": self.want}


def create(info: Any = None) -> ScriptedPlanner:
    """The page's planner factory (ui.server --planner brains.scripted:create). ``--opt want=alarm_clock``
    style options are accepted through ``info.options``; ``goal=off`` leaves place goals out."""
    opts = dict(getattr(info, "options", None) or {})
    return ScriptedPlanner(opts.get("want") or None, goal_on_place=str(opts.get("goal", "on")) != "off")


__all__ = ["PolicyBrain", "ScriptedPlanner", "create", "type_of"]
