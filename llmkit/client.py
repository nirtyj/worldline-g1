"""Minimal tool-calling clients for Anthropic, Gemini and OpenAI-compatible servers.

No dependencies: plain HTTPS through urllib, run in a worker thread so the
simulation keeps ticking while a request is in flight.

    client = make_client(provider="anthropic", model="claude-haiku-4-5-20251001")
    call = await client.tool_call(system, user_text, tools)   # -> ToolUse(name, args, ...)

``tools`` is a list of {"name", "description", "parameters": <JSON schema>}.
The call forces exactly one tool call:
  Anthropic  tool_choice {"type": "any", "disable_parallel_tool_use": true}
  OpenAI     tool_choice "required", parallel_tool_calls false (vLLM: serve the
             model with tool calling enabled, e.g. --enable-auto-tool-choice
             --tool-call-parser <parser>)
  Gemini     function calling mode ANY (through the google-genai SDK), e.g.
             gemini-3.8-flash

Environment: ANTHROPIC_API_KEY for Anthropic; GEMINI_API_KEY for Gemini;
OPENAI_API_KEY (optional for a local vLLM) and OPENAI_BASE_URL for
OpenAI-compatible servers.

The request and response shapes follow the public API docs.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMError(Exception):
    pass


@dataclass
class ToolUse:
    name: str
    args: dict[str, Any]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    text: str = ""                       # any text the model produced alongside the call
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class _Base:
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] = staticmethod(_post_json)

    def __init__(self, model: str, max_tokens: int = 512, temperature: float | None = 0.0,
                 timeout: float = 30.0, retries: int = 3) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    async def _send(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.to_thread(self.transport, url, headers, body, self.timeout)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500] if hasattr(e, "read") else ""
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < self.retries:
                    retry_after = e.headers.get("retry-after") if e.headers else None
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    await asyncio.sleep(wait + random.random() * 0.3)
                    delay *= 2
                    continue
                raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"cannot reach {url}: {e}") from e
        raise LLMError("unreachable")


class AnthropicClient(_Base):
    def __init__(self, model: str, api_key: str | None = None, url: str | None = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.url = url or os.environ.get("ANTHROPIC_URL", ANTHROPIC_URL)

    def build_request(self, system: str, user_text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_text}],
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    @staticmethod
    def parse_response(resp: dict[str, Any]) -> tuple[str, dict[str, Any], str, int, int]:
        blocks = resp.get("content") or []
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not uses:
            raise LLMError(f"no tool call in response (stop_reason={resp.get('stop_reason')}): {text[:200]!r}")
        use = uses[0]
        usage = resp.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0) or 0) \
            + int(usage.get("cache_creation_input_tokens", 0) or 0)
        return use["name"], dict(use.get("input") or {}), text, tokens_in, int(usage.get("output_tokens", 0))

    async def tool_call(self, system: str, user_text: str, tools: list[dict[str, Any]]) -> ToolUse:
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        body = self.build_request(system, user_text, tools)
        headers = {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION}
        t0 = time.monotonic()
        resp = await self._send(self.url, headers, body)
        latency = (time.monotonic() - t0) * 1000
        name, args, text, tin, tout = self.parse_response(resp)
        return ToolUse(name, args, latency, tin, tout, text, resp)


class OpenAIClient(_Base):
    """OpenAI-compatible chat completions: OpenAI, vLLM, SGLang, llama.cpp server..."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def build_request(self, system: str, user_text: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["parameters"]}} for t in tools],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    @staticmethod
    def parse_response(resp: dict[str, Any]) -> tuple[str, dict[str, Any], str, int, int]:
        choices = resp.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {str(resp)[:200]}")
        msg = choices[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        text = (msg.get("content") or "").strip()
        if not calls:
            raise LLMError(f"no tool call in response: {text[:200]!r}")
        fn = calls[0].get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError as e:
            raise LLMError(f"tool arguments are not JSON: {raw_args[:200]!r}") from e
        usage = resp.get("usage") or {}
        return fn.get("name", ""), args, text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    async def tool_call(self, system: str, user_text: str, tools: list[dict[str, Any]]) -> ToolUse:
        body = self.build_request(system, user_text, tools)
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.monotonic()
        resp = await self._send(self.base_url + "/chat/completions", headers, body)
        latency = (time.monotonic() - t0) * 1000
        name, args, text, tin, tout = self.parse_response(resp)
        return ToolUse(name, args, latency, tin, tout, text, resp)


class GeminiClient(_Base):
    """Gemini through the google-genai SDK, forced to call exactly one function.

    ``thinking_budget`` 0 keeps latency low (Gemini thinks by default); None
    leaves the model's own thinking on."""

    def __init__(self, model: str, api_key: str | None = None, thinking_budget: int | None = 0, **kw: Any) -> None:
        super().__init__(model, **kw)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.thinking_budget = thinking_budget
        self._client: Any = None

    def build_config(self, system: str, tools: list[dict[str, Any]]) -> Any:
        from google.genai import types
        cfg: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
            "tools": [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"]) for t in tools])],
            "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**cfg)

    @staticmethod
    def parse_response(resp: Any) -> tuple[str, dict[str, Any], str, int, int]:
        calls = resp.function_calls or []
        text = ""
        try:
            parts = resp.candidates[0].content.parts or []
            text = " ".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()
        except (AttributeError, IndexError, TypeError):
            pass
        if not calls:
            reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
            raise LLMError(f"no function call in response (finish_reason={reason}): {text[:200]!r}")
        usage = resp.usage_metadata
        tin = int(getattr(usage, "prompt_token_count", 0) or 0)
        tout = int(getattr(usage, "candidates_token_count", 0) or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0)
        return calls[0].name, dict(calls[0].args or {}), text, tin, tout

    async def tool_call(self, system: str, user_text: str, tools: list[dict[str, Any]]) -> ToolUse:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set: add it to .env and restart the server")
        from google import genai
        from google.genai import errors
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        config = self.build_config(system, tools)
        delay = 1.0
        t0 = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=self.model, contents=user_text, config=config),
                    self.timeout)
                break
            except errors.APIError as e:
                if getattr(e, "code", None) in (429, 500, 502, 503, 504) and attempt < self.retries:
                    await asyncio.sleep(delay + random.random() * 0.3)
                    delay *= 2
                    continue
                raise LLMError(f"Gemini {getattr(e, 'code', '?')}: {str(e)[:400]}") from e
            except asyncio.TimeoutError as e:
                if attempt < self.retries:
                    continue
                raise LLMError(f"Gemini took longer than {self.timeout:.0f} s") from e
        latency = (time.monotonic() - t0) * 1000
        name, args, text, tin, tout = self.parse_response(resp)
        return ToolUse(name, args, latency, tin, tout, text, {"model": self.model})


def provider_for(model: str | None) -> str:
    """The provider a model name implies: gemini-* is Gemini, claude-* is Anthropic."""
    if model and model.startswith("gemini"):
        return "gemini"
    return "anthropic"


def make_client(provider: str = "anthropic", model: str | None = None, **kw: Any) -> _Base:
    if provider == "gemini":
        return GeminiClient(model or "gemini-3.8-flash", **kw)
    if provider == "anthropic":
        return AnthropicClient(model or "claude-haiku-4-5-20251001", **kw)
    if provider in ("openai", "vllm", "local"):
        if not model:
            raise LLMError("an OpenAI-compatible server needs --opt model=<served model name>")
        return OpenAIClient(model, **kw)
    raise LLMError(f"unknown provider {provider!r}; use anthropic, gemini or openai")
