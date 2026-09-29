"""The plumbing every model-backed brain shares: tool schemas, one forced tool
call per decision, argument filtering, and latency/token stats.

A brain subclasses ModelBrain and supplies the prompt and the context:

    class MyBrain(ModelBrain):
        system = "You control a home robot..."
        def render(self, ctx): return "..."             # BrainInput -> text for next_action
        def render_classify(self, utt, ctx): return "..."

Change prompts and context in the brain (agent/model.py), not here.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

from brains.interface import KINDS, TOOLS, BrainInput, ToolCall

from .client import LLMError, make_client, provider_for

CLASSIFY_TOOL = {
    "name": "classify",
    "description": "Report the kind of the latest utterance.",
    "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": list(KINDS)}},
                   "required": ["kind"]},
}

CLASSIFY_SYSTEM = """You label what a user just said to a home robot. Kinds:
request     a new task ("bring me the mug", "what's on the table?")
correction  changes the current task, e.g. a different object or place ("no, the alarm clock instead")
addition    adds a task and keeps the current one ("also grab a napkin")
question    needs an answer and changes nothing ("how long will it take?")
stop        halt now ("stop!")
resume      carry on after a stop ("okay, go ahead")
answer      answers the robot's own question ("the blue one")
chitchat    nothing to do
constraint  changes how the current task should be done without replacing it
observation reports something about the world that may affect the task
Call classify with the kind of the latest utterance."""


def tool_schemas(ctx: BrainInput) -> list[dict[str, Any]]:
    """The robot's tools as JSON schemas, with allowed values filled in from the
    map (keypoints) and belief (object ids)."""
    keypoints = sorted(ctx.map["keypoints"])
    object_ids = sorted((ctx.belief.get("objects") or {}).keys())
    out = []
    for name, spec in TOOLS.items():
        props: dict[str, Any] = {}
        required: list[str] = []
        for arg, a in spec["args"].items():
            schema = {k: v for k, v in a.items() if k in ("type", "description", "enum")}
            if arg == "to":
                schema["enum"] = keypoints
            elif arg == "object" and object_ids:
                schema["enum"] = object_ids
            props[arg] = schema
            if not a.get("optional"):
                required.append(arg)
        if name == "wait":
            props["reason"] = {"type": "string", "description": "Why, in a few words."}
        params: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            params["required"] = required
        out.append({"name": name, "description": spec["description"], "parameters": params})
    return out


def client_from_options(options: dict[str, str], default_model: str) -> Any:
    """Build a client from BrainInfo.options (set by ui/server.py): provider, model, base_url, max_tokens,
    timeout, temperature (a number, or 'none' to leave it unset)."""
    opts = dict(options)
    provider = opts.pop("provider", None) or provider_for(opts.get("model") or default_model)
    model = opts.pop("model", default_model if provider == "anthropic" else None)
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set: add it to .env and restart the server")
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set: add it to .env and restart the server")
    kw: dict[str, Any] = {}
    if provider == "gemini" and "thinking_budget" in opts:
        t = opts.pop("thinking_budget")
        kw["thinking_budget"] = None if t.lower() == "none" else int(t)
    if "base_url" in opts:
        kw["base_url"] = opts.pop("base_url")
    for key, cast in (("max_tokens", int), ("timeout", float)):
        if key in opts:
            kw[key] = cast(opts.pop(key))
    if "temperature" in opts:
        t = opts.pop("temperature")
        kw["temperature"] = None if t.lower() == "none" else float(t)
    return make_client(provider, model, **kw)


class ModelBrain:
    system = "You control a home robot. Call exactly one tool."
    classify_system = CLASSIFY_SYSTEM

    def __init__(self, client: Any, map_: dict[str, Any]) -> None:
        self.client = client
        self.map = map_
        self.latencies: list[float] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0

    # Override these two in a brain.
    def render(self, ctx: BrainInput) -> str:
        raise NotImplementedError

    def render_classify(self, utterance: Any, ctx: BrainInput) -> str:
        return f"LATEST UTTERANCE\n{utterance.text}\n\nCall classify."

    def tools(self, ctx: BrainInput) -> list[dict[str, Any]]:
        return tool_schemas(ctx)

    async def _call(self, system: str, text: str, tools: list[dict[str, Any]]):
        try:
            use = await self.client.tool_call(system, text, tools)
        except LLMError:
            self.errors += 1
            raise
        self.latencies.append(use.latency_ms)
        self.tokens_in += use.input_tokens
        self.tokens_out += use.output_tokens
        return use

    async def classify(self, utterance: Any, ctx: BrainInput) -> str:
        use = await self._call(self.classify_system, self.render_classify(utterance, ctx), [CLASSIFY_TOOL])
        kind = str(use.args.get("kind", "")).strip().lower()
        return kind if kind in KINDS else "chitchat"

    async def next_action(self, ctx: BrainInput) -> ToolCall:
        use = await self._call(self.system, self.render(ctx), self.tools(ctx))
        if use.name not in TOOLS:
            return ToolCall("wait", reason=f"model called unknown tool {use.name!r}")
        args = {k: v for k, v in use.args.items() if k in TOOLS[use.name]["args"]}
        reason = use.args.get("reason") if use.name == "wait" else (use.text or None)
        return ToolCall(use.name, args, reason=reason)

    def stats(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        out: dict[str, Any] = {"calls": len(lat), "errors": self.errors,
                               "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
        if lat:
            out["latency_ms_p50"] = statistics.median(lat)
            out["latency_ms_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return out
