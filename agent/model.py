"""The planner: context builder and prompt for the planning model: Gemini 3.8 Flash
(GEMINI_API_KEY, the default) or Claude (ANTHROPIC_API_KEY).

SYSTEM is one template (SYSTEM_TEMPLATE); only numeric slots (speeds, durations,
heights) differ per profile, filled from BrainInput.profile (PLAN 5.13, 1.3 #30).
robot_state and perception are never rendered (Invariant 5).

The playground builds it with create_brain(info); info.options picks the model.
Beyond a plain one-tool-per-turn loop, it:

- classifies obvious utterances with rules and asks the model only for the rest,
  which saves a model round trip on "stop", "go ahead" and answers to its own question
- states the procedure and the harness's rules in the prompt, so fewer calls are rejected
- renders belief as short readable lines with provenance instead of raw JSON
- marks late results and unplayed lines, and says which user lines still need a reply
- tells the model which request is current (intent version) and what is running

The client, tool schemas and stats come from llmkit/.
"""

from __future__ import annotations

import re
from typing import Any

from api.results import is_fallback
from api.types import PROFILES
from brains.interface import UNKNOWN, BrainInfo, BrainInput
from llmkit import ModelBrain, client_from_options

from . import layout

REFERENCE_DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_TEMPLATE = """You are the planner for a home robot: a Unitree G1 humanoid with two arms and \
three-finger hands, a camera on its torso (no neck) and a speaker. It walks about {v:.1f} m/s and turns in \
place; a pick takes about {t_pick:.0f} s and a place about {t_place:.0f} s. It holds one object at a time. \
It can reach surfaces {h_min:.2f}-{h_max:.2f} m high. A harness runs your decisions. Each turn you call \
exactly one tool.

THE WORLD
The robot walks between named locations: keypoints (each stands at a stretch of a surface), rooms, and \
"user". Objects sit on surfaces; each surface is served from one or more keypoints. To deliver to a person, \
place the object on their surface (see PEOPLE). "Me"/"you" is the user (PEOPLE: user). Objects inside a \
fridge, cabinet or drawer can't be reached. In a house with several rooms, spots start with their room \
(kitchen_counter_1a, bedroom_bed_1a) and ROOMS lists the spots in each room; think about which room a \
thing usually lives in before searching. list_locations gives walking distances from where the robot is.

WHAT YOU KNOW
You have the map (keypoints, surfaces, people), not the room's contents. BELIEF lists only what the robot \
has seen: "seen" this session (an observation, or the camera while walking), or "remembered" from an \
earlier session (it may have moved since). Objects can be picked up. Landmarks are fixed things \
(microwave, stove, fridge, toaster...): they can't be picked up, and they tell you where places are \
("next to the toaster" means near that landmark's keypoint). LAYOUT, when shown, is worked out from the \
map: each line of surfaces left to right as you face it, what each landmark sits between, and which lines \
face each other. Use it for "next to", "left of" or "the other side of"; don't guess from names. When a \
request says where something should end up, pass that relation as goal on the place ("other side of \
stove_1 from kitchen_counter_1a"); the runtime checks where it really landed. If the check fails, fix it or \
tell the user; never say it's done when it isn't. The robot can't put things into containers (a cup, a \
bowl) yet: say so. The robot can't open, switch on or put things inside appliances yet; say so if asked. \
LOOKED AT says which spots the robot has checked and when.

FINDING THINGS
Something you need isn't in BELIEF, is UNKNOWN, or isn't where BELIEF says: search for it yourself. Think \
where that kind of thing is usually kept, then walk to the likely spots you haven't looked at recently \
(LOOKED AT), nearest first; the robot looks around each time it arrives, and the camera also adds what it \
passes. Say once that you're looking. Ask the user only when you've checked the likely spots and still \
can't find it, or when they obviously know better ("my phone": ask where they left it after one quick look \
nearby).

YOUR OWN GOALS
When nobody needs anything, the robot's persona may give you an OWN GOAL: look around from where you are, \
look at a spot you've never checked, go back to the user, or ask the user whether you may look around. Use \
only the tools the goal lists, never manipulate, and stay quiet apart from at most one short line (if the \
user just agreed to it, a brief thanks). Work on the goal's own target: the persona hands you the next spot \
when this one is done. The user always comes first: anything they say replaces the own goal. When there is \
no own goal and nothing to do, call wait_and_observe.

NOTES are things the user told you about the home in earlier conversations. Trust them like memory: a \
good first place to look, not proof.

NOTICED lists what System 1, a fast camera model, saw that the object list doesn't hold: a door left open, \
a spill, what a room looks like. They are unverified hints: use them to decide where to look or what to \
mention, and confirm with wait_and_observe before acting on them.

MEMORY AND RECALL
BELIEF shows only what matters right now. Everything else the robot has seen, where things usually are, \
what the user told you, and what was asked or done in earlier sessions is in memory: call recall(query) to \
ask, e.g. recall("mug"), recall("kitchen"), recall("what did the user ask for last time"). It answers \
instantly and nothing moves. Use it before searching, and to answer questions about the past. "usually \
<spot>" is where a thing has been seen most often: search there first. LEARNED FROM PAST TASKS, when \
present, says what worked before at this step.

HOW TO DO A DELIVERY
1. Acknowledge the request in one short sentence.
2. navigate to the object's surface keypoint; the robot looks around when it arrives (a remembered place \
may be stale).
3. check_reachability(object_type, object_id) from there. If it says needs_reposition: \
navigate(location="reach_stance"), then check_reachability again. Then manipulate(action="pick", \
object_type, arm=preferred_arm).
4. After every manipulate the harness looks to verify. Trust BELIEF, not a tool's status.
5. navigate to "user", manipulate(action="place", object_type, target="user"), then speak to say it's done.

RULES (the harness enforces them; a rejection comes back in NOTE and ACTIONS)
- A hand marked UNKNOWN or unverified: call wait_and_observe(timeout_s=0) first. Its "unchanged" result \
means "looked, nothing new".
- Holding something the current request doesn't want: put it back where it came from first (navigate \
there if needed, then place).
- Pick only right after a successful check_reachability for the same object from exactly where the robot \
stands; only speak, list_locations or recall may come in between.
- Ask the user only when the choice matters and the request doesn't settle it (two different items that \
fit, like two mugs for "my mug"). Identical items: take either.
- Object not where expected: search the other likely spots (FINDING THINGS); tell the user only if it isn't found.
- Out of reach: if check_reachability suggests another location, try it once; then tell the user.
- A reason too_low or too_high means the robot can't reach that height: tell the user.
- When a result says [fallback], it still counts; don't mention it to the user.
- Never say something is done before BELIEF shows it.
- The user just tells you something ("my keys are usually on the shelf"): acknowledge it in a few words; \
don't start a task they didn't ask for.
- Answer questions right away (the task keeps running). Estimate time from distances: the robot walks \
about {v:.1f} m/s ({s_per_m:.1f} s per metre); a pick takes about {t_pick:.0f} s, a place about {t_place:.0f} s.
- Lines marked "late" in ACTIONS are results from an earlier request. They tell you about the world, not \
about progress on the current request.
- After the user says stop, only talk until they say to continue.
- If nothing needs doing now (an action is running, or you're waiting for the user), call wait_and_observe.

Speak in short, natural sentences. Don't narrate every step."""


def system_prompt(slots: dict[str, float] | None = None) -> str:
    """The one SYSTEM template with the profile's numeric slots filled (PLAN 1.3 #30, deviation D2)."""
    from api.tools import render_description, ToolSpec
    s = dict(PROFILES["lite"].slots())
    s.update(slots or {})
    return render_description(ToolSpec("system", "instant", frozenset(), SYSTEM_TEMPLATE, {}), s)


SYSTEM = system_prompt()


def create_brain(info: BrainInfo) -> "ReferenceBrain":
    return ReferenceBrain(client_from_options(info.options, REFERENCE_DEFAULT_MODEL), info.map)


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.0f} h" if seconds < 172800 else f"{seconds / 86400:.0f} days"


def render_map(m: dict[str, Any]) -> str:
    rooms = m.get("rooms") or {}
    if rooms:
        lines = ["ROOMS (spots in each; navigate to the room name to go to the room)"] + [
            f"  {r.get('label') or name}: {', '.join(r.get('spots') or []) or '(no spots)'}"
            for name, r in rooms.items()]
        lines.append("KEYPOINTS")
    else:
        lines = ["KEYPOINTS"]
    lines += [f"  {k}: {v.get('desc') or k}" for k, v in m["keypoints"].items()]
    lines.append("SURFACES (reached from)")
    hi, lo = m.get("max_reach_height_m", 99.0), m.get("min_reach_height_m", -1.0)
    lines += [f"  {s}: {i.get('desc') or s}, {i['height_m']} m high, from {', '.join(i['keypoints'])}"
              + ("  (too high to reach)" if i["height_m"] > hi else "")
              + ("  (too low to reach)" if i["height_m"] < lo else "")
              for s, i in m["surfaces"].items()]
    lines.append("PEOPLE (deliver to)")
    lines += [f"  {p}: {i.get('deliver_to_surface')} at keypoint {i.get('keypoint')} (navigate to \"{p}\")"
              for p, i in (m.get("people") or {}).items()]
    return "\n".join(lines)


class ReferenceBrain(ModelBrain):
    system = SYSTEM

    def system_for(self, ctx: BrainInput) -> str:
        """The SYSTEM template with this profile's numeric slots (the only per-profile difference)."""
        return system_prompt(ctx.profile) if getattr(ctx, "profile", None) else self.system

    def render_classify(self, utterance: Any, ctx: BrainInput) -> str:
        return ("CONVERSATION (oldest first)\n" + self._conversation(ctx) +
                f"\n\nLATEST UTTERANCE\n{utterance.text}\n\nCall classify.")

    # ------------------------------------------------------------------
    # Classification: rules first, the model for the rest
    # ------------------------------------------------------------------
    async def classify(self, utterance: Any, ctx: BrainInput) -> str:
        text = utterance.text.strip().lower()
        if re.match(r"^\W*(stop|freeze|halt)\b", text):
            return "stop"
        if ctx.paused and re.search(r"\b(go ahead|continue|carry on|keep going|go on)\b", text):
            return "resume"
        if self._robot_asked_last(ctx) and len(text.split()) <= 5 and re.match(r"^(the |that )?\w+( one)?\W*$", text):
            return "answer"
        return await super().classify(utterance, ctx)

    @staticmethod
    def _robot_asked_last(ctx: BrainInput) -> bool:
        says = [e for e in ctx.history if e.tool == "speak" and e.status not in ("dropped", "rejected")]
        return bool(says) and says[-1].args.get("text", "").rstrip().endswith("?")

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def render(self, ctx: BrainInput) -> str:
        return "\n\n".join([
            f"TIME {ctx.now:.1f} s. Current request version: {ctx.intent_version}."
            + (f" Goal: {ctx.task_state['goal'].rstrip('.')}." if ctx.task_state.get("goal") else "")
            + (" The user said STOP: talk only, until they say to continue." if ctx.paused else ""),
            "MAP\n" + render_map(ctx.map),
        ] + ([f"LAYOUT (worked out from the map and the landmarks seen)\n{lay}"] if (lay := self._layout(ctx)) else []) + [
            "CONVERSATION (oldest first)\n" + self._conversation(ctx),
            # Belief is the fused view the planner acts on. Raw telemetry and the
            # perception frame stay out of the prompt: they double its size, change
            # every frame, and would leak ground truth past the look/verify rules.
            "BELIEF\n" + self._belief(ctx),
            "LOOKED AT\n" + self._looked(ctx),
        ] + ([self._notes(ctx)] if ctx.notes else [])
          + ([self._noticed(ctx)] if ctx.observations else []) + [
            "ACTIONS (latest last)\n" + self._actions(ctx),
            "RUNNING NOW\n" + (", ".join(f"{e.execution_id} {e.tool}({self._args(e.args)})"
                                         for e in ctx.active if e.tool != "speak") or "nothing"),
        ] + ([f"LEARNED FROM PAST TASKS (suggestions, not rules)\n{ctx.guidance}"] if ctx.guidance else [])
          + ([self._own_goal(ctx)] if ctx.own_goal else [])
          + ([f"NOTE\n{ctx.note}"] if ctx.note else []) + ["Choose the next step: call one tool."])

    @staticmethod
    def _layout(ctx: BrainInput) -> str:
        return layout.render(layout.build(ctx.map, (ctx.belief or {}).get("landmarks")), ctx.map)

    @staticmethod
    def _own_goal(ctx: BrainInput) -> str:
        g = ctx.own_goal or {}
        return (f"OWN GOAL (from the persona, drive: {g.get('drive')}; started {ctx.now - g.get('t_start', ctx.now):.0f} s ago)\n"
                f"{g.get('text')}\nTools allowed: {', '.join(g.get('tools') or [])}. It ends by itself when done.")

    @staticmethod
    def _notes(ctx: BrainInput) -> str:
        return "NOTES (what the user told you before)\n" + "\n".join(
            f"- {_ago(n['ago_s'])} ago: \"{n['text']}\"" + (f" (about the {n['about']})" if n.get("about") else "")
            for n in ctx.notes[-12:])

    @staticmethod
    def _noticed(ctx: BrainInput) -> str:
        return "NOTICED (by System 1, unverified)\n" + "\n".join(
            f"- {_ago(o['ago_s'])} ago" + (f" at {o['where']}" if o.get("where") else "") + f": {o['text']}"
            for o in ctx.observations[-8:])

    @staticmethod
    def _looked(ctx: BrainInput) -> str:
        looked = (ctx.belief or {}).get("looked") or {}
        surfaces = ctx.map["surfaces"]

        def seen(s: str) -> dict[str, Any] | None:     # a surface counts as looked at from any of its stands
            keys = [k for k in [s, *(surfaces[s].get("keypoints") or [])] if k in looked]
            return max((looked[k] for k in keys), key=lambda f: f["t"]) if keys else None
        spots = list(surfaces)
        done = sorted((k for k in spots if seen(k)), key=lambda k: -seen(k)["t"])
        rows = [f"{k} {_ago(ctx.now - seen(k)['t'])} ago" + (" (remembered)" if seen(k)["source"] == "memory" else "")
                for k in done]
        never = [k for k in spots if not seen(k)]
        return ((" · ".join(rows) if rows else "no spot looked at yet")
                + (f"\nnever: {', '.join(never)}" if never else ""))

    HIDDEN_ARGS = ("candidates", "stance", "reach_execution_id", "skill_id", "why", "at")

    @classmethod
    def _args(cls, args: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in args.items() if k not in cls.HIDDEN_ARGS)

    def _conversation(self, ctx: BrainInput) -> str:
        rows = []
        says = [e for e in ctx.history if e.tool == "speak"]
        for u in ctx.utterances:
            replied = any(e.t_start >= u.t_end - 1e-6 and e.status not in ("dropped", "rejected") for e in says)
            kind = ctx.kinds.get(u.id, "?")
            # The runtime halts and acks a stop from the partial transcript, before the
            # final one lands: list the stop just ahead of its ack, already answered.
            ack = next((e for e in says if e.tag == "runtime:safety-ack"
                        and abs(e.t_start - u.t_end) < 3.0), None) if kind == "stop" else None
            sort_t = ack.t_start - 1e-3 if ack else u.t_end
            replied = replied or ack is not None
            target = ((getattr(u, "directive", None) or {}).get("target") or {})
            kind += "".join(f", {k}={v}" for k, v in target.items() if v)     # what the classifier grounded
            flag = "" if replied else "   <- not replied to yet"
            rows.append((sort_t, f"[{sort_t:6.1f}] USER ({kind}): {u.text}{flag}"))
        for e in says:
            if e.status == "rejected":
                continue
            state = {"dropped": " (dropped, never played)", "cancelled": " (cut off)",
                     "queued": " (queued)", "running": " (playing)"}.get(e.status, "")
            rows.append((e.t_start, f"[{e.t_start:6.1f}] ROBOT: {e.args.get('text', '')}{state}"))
        rows.sort(key=lambda r: r[0])
        return "\n".join(r[1] for r in rows[-30:]) or "(nothing yet)"

    @staticmethod
    def _focus(ctx: BrainInput) -> set[str]:
        """Objects worth showing now: what this session has seen in the robot's room (all of it
        in a single room), anything held, and anything the conversation or own goal mentions.
        The rest stays in memory, one recall away."""
        b = ctx.belief
        objects = b.get("objects") or {}
        kps = ctx.map.get("keypoints") or {}
        here = (b.get("robot") or {}).get("at")
        room_here = (kps.get(here) or {}).get("room")
        text = " ".join(u.text for u in ctx.utterances[-3:]).lower()
        if ctx.own_goal:
            text += " " + str(ctx.own_goal.get("text", "")).lower()
        words = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", text)}
        keep = set()
        for oid, o in objects.items():
            kind_words = set(str(o.get("type", "")).split("_")) | set(str(o.get("label") or "").split())
            where = str(o.get("where"))
            if kind_words & words or where.startswith("hand") or oid in text:
                keep.add(oid)
            elif o.get("source") != "memory" and (not room_here or (kps.get(where) or {}).get("room") == room_here):
                keep.add(oid)
        return keep

    @classmethod
    def _belief(cls, ctx: BrainInput) -> str:
        b, now = ctx.belief, ctx.now
        robot = b.get("robot") or {}
        where = robot.get("at") or (f"between {' and '.join(robot['between'])}" if robot.get("between") else "unknown")
        lines = [f"robot at: {where}"]
        for arm, h in (b.get("hands") or {}).items():
            held = h.get("holding")
            if held == UNKNOWN:
                state = "UNKNOWN (look first)"
            elif held is None:
                state = "empty"
            else:
                state = f"holding {held}"
            if not h.get("verified"):
                state += " (unverified)"
            if h.get("hint"):
                state += f"; {h['hint']}"
            lines.append(f"{arm} hand: {state}")
        objects = b.get("objects") or {}
        focus = cls._focus(ctx)
        lines.append("objects (what matters now; the rest is in memory):" if focus else
                     "objects: none relevant in view (recall, or search: see FINDING THINGS)")
        for oid in sorted(focus):
            o = objects[oid]
            if o.get("source") == "memory":
                src = f"remembered, last seen {_ago(now - o['t'])} ago, unverified"
            elif o.get("verified"):
                src = f"seen, t={o.get('t')}"
            elif o.get("source") == "look_absent":
                src = "not there on the last look"
            else:
                src = f"{o.get('source')}, unverified, t={o.get('t')}"
            extra = f"; usually {o['usual']}" if o.get("usual") and o.get("usual") != o.get("where") else ""
            if o.get("mem_status") in ("missed", "moved"):
                extra += f"; memory: {o['mem_status']}"
            lines.append(f"  {oid} ({o.get('label') or o.get('type')}): {o.get('where')} [{src}{extra}]")
        rest = [oid for oid in objects if oid not in focus]
        if rest:
            kps = ctx.map.get("keypoints") or {}
            by_room: dict[str, int] = {}
            for oid in rest:
                room = (kps.get(str(objects[oid].get("where"))) or {}).get("room") or "elsewhere"
                by_room[room] = by_room.get(room, 0) + 1
            lines.append(f"  (+{len(rest)} more in memory: " + ", ".join(f"{r} {n}" for r, n in sorted(by_room.items()))
                         + "; call recall to ask about them)")
        marks = b.get("landmarks") or {}
        if marks:
            lines.append("landmarks (fixed, can't be picked up):")
            for k, lm in marks.items():
                src = (f"remembered, last seen {_ago(now - lm['t'])} ago" if lm["source"] == "memory" else "seen")
                lines.append(f"  {k} ({lm['label']}): near {lm['near']} [{src}]")
        if b.get("blocked"):
            lines.append("blocked edges: " + "; ".join("-".join(e) for e in b["blocked"]))
        return "\n".join(lines)

    def _actions(self, ctx: BrainInput) -> str:
        """[t] g{gen} nav-000012 navigate(location=...) -> succeeded [sonic_walk] <summary> (PLAN 5.13)."""
        rows = []
        for e in [e for e in ctx.history if e.tool != "speak"][-18:]:
            d = e.data or {}
            res = e.result
            late = " (late: from an earlier request)" if (d.get("late") or (res is not None and res.late)) else ""
            who = " [harness]" if e.source == "harness" else (" [persona]" if e.source == "persona" else "")
            executor = d.get("executor")
            skill = d.get("skill")
            # the manipulate summary already names the skill (and [fallback]); others get their executor
            tags = [executor] if (executor and e.tool != "manipulate") else []
            if is_fallback(executor) and e.tool != "manipulate":
                tags.append("fallback")
            tag = f" [{', '.join(tags)}]" if tags else ""
            if res is None or e.tool == "recall":
                what = self._data(e.tool, d)
            elif e.tool == "observe" and res.ok:
                what = f"{res.summary}; hands={d.get('hands')}"
            else:
                what = res.summary
            name = e.tool if e.tool != "observe" else f"look({e.action})"
            call = f"{name}({self._args(e.args)})" if e.tool != "observe" else name
            rows.append(f"[{e.t_start:6.1f}] g{e.generation} {e.execution_id} {call} -> {e.status}"
                        f"{tag}{late}{who} {what}".rstrip())
        return "\n".join(rows) or "(none)"

    @staticmethod
    def _data(tool: str, d: dict[str, Any]) -> str:
        if tool == "recall":
            return "answer: " + str(d.get("answer", "")).replace("\n", " | ")[:700]
        if tool in ("observe", "look"):
            seen = {s: [o["id"] for o in objs] for s, objs in (d.get("surfaces") or {}).items()}
            return f"at={d.get('at')} sees={seen} hands={d.get('hands')}"
        keep = {k: v for k, v in d.items() if k in ("reason", "at", "between", "reachable", "preferred_arm",
                                                   "suggest_location", "holding", "blocked_edge", "status")}
        return " ".join(f"{k}={v}" for k, v in keep.items())
