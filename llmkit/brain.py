"""The plumbing every model-backed brain shares: tool schemas, one forced tool
call per decision, argument filtering, and latency/token stats.

Tool schemas come from api/tools.py (the only definition), enums filled by
argument name once per session and frozen (PLAN 5.2, Invariant 9).

A brain subclasses ModelBrain and supplies the prompt and the context:

    class MyBrain(ModelBrain):
        system = "You control a home robot..."
        def render(self, ctx): return "..."             # BrainInput -> text for next_action
        def render_classify(self, utt, ctx): return "..."

Change prompts and context in the brain (agent/model.py), not here.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any

from api.tools import TOOL_SPECS, UNKNOWN_TOOL_FALLBACK, SchemaContext, ToolCall, coerce_args, json_schemas
from api.types import PROFILES
from brains.interface import KINDS, BrainInput

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


def schema_context(ctx: Any) -> SchemaContext:
    """The session's frozen SchemaContext (BrainInput.tools_ctx), else one derived from the map."""
    sc = getattr(ctx, "tools_ctx", None)
    if isinstance(sc, SchemaContext):
        return sc
    m = getattr(ctx, "map", None) or {}
    slots = getattr(ctx, "profile", None) or PROFILES["lite"].slots()
    return SchemaContext.from_map(m, m.get("skill_types") or [], slots)


def provider_of(client: Any) -> str:
    name = type(client).__name__ if client is not None else ""
    return {"AnthropicClient": "anthropic", "GeminiClient": "gemini", "OpenAIClient": "openai"}.get(name, "neutral")


def tool_schemas(ctx: Any, provider: str = "neutral") -> list[dict[str, Any]]:
    """The robot's tools as JSON schemas (api.tools.json_schemas), enums filled by argument name
    from the session's SchemaContext: locations, surfaces and the registry's object types."""
    return json_schemas(schema_context(ctx), provider)  # type: ignore[arg-type]


def extra_calls(use: Any) -> list[tuple[str, dict[str, Any]]]:
    """Calls beyond the first in one reply (Anthropic content blocks, OpenAI tool_calls)."""
    raw = getattr(use, "raw", None) or {}
    out: list[tuple[str, dict[str, Any]]] = []
    blocks = raw.get("content") if isinstance(raw, dict) else None
    if isinstance(blocks, list):
        uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        out = [(b.get("name", ""), dict(b.get("input") or {})) for b in uses[1:]]
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices:
        calls = ((choices[0] or {}).get("message") or {}).get("tool_calls") or []
        for c in calls[1:]:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            out.append((fn.get("name", ""), dict(args) if isinstance(args, dict) else {}))
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
        self._tools: list[dict[str, Any]] = []
        self._tools_key: Any = None

    # Override these two in a brain.
    def render(self, ctx: BrainInput) -> str:
        raise NotImplementedError

    def render_classify(self, utterance: Any, ctx: BrainInput) -> str:
        return f"LATEST UTTERANCE\n{utterance.text}\n\nCall classify."

    def tools(self, ctx: BrainInput) -> list[dict[str, Any]]:
        """Frozen per session (Invariant 9): rendered once per schema_rev and reused byte for byte."""
        sc = schema_context(ctx)
        key = (sc.schema_rev, sc.locations, sc.surfaces, sc.skill_types, tuple(sorted(sc.slots.items())),
               sc.include_look)
        if self._tools_key != key:
            self._tools_key = key
            self._tools = tool_schemas(ctx, provider_of(self.client))
        return self._tools

    def system_for(self, ctx: BrainInput) -> str:
        return self.system

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
        use = await self._call(self.system_for(ctx), self.render(ctx), self.tools(ctx))
        include_look = schema_context(ctx).include_look
        known = set(TOOL_SPECS) | ({"look"} if include_look else set())
        if use.name not in known:
            # An unknown tool name becomes wait_and_observe(timeout_s=10, reason=<model text>) (PLAN 5.2).
            return ToolCall(UNKNOWN_TOOL_FALLBACK, {"timeout_s": 10.0,
                                                   "reason": (use.text or f"model called unknown tool {use.name!r}")[:200]},
                            reason=f"model called unknown tool {use.name!r}")
        args = coerce_args(use.name, use.args, include_look)
        reason = args.get("reason") if use.name == "wait_and_observe" else (use.text or None)
        # More than one call in a reply: keep the first; the harness rejects the rest (doc 21).
        extra = [ToolCall(n, coerce_args(n, a, include_look)) for n, a in extra_calls(use)]
        return ToolCall(use.name, args, reason=reason, extra=extra)

    def stats(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        out: dict[str, Any] = {"calls": len(lat), "errors": self.errors,
                               "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
        if lat:
            out["latency_ms_p50"] = statistics.median(lat)
            out["latency_ms_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return out
