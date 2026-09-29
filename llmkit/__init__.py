"""Model access: clients for Anthropic, Gemini and OpenAI-compatible servers, and
the ModelBrain base class. Prompts and context live in the brain that
subclasses it (agent/model.py)."""

from .brain import CLASSIFY_SYSTEM, CLASSIFY_TOOL, ModelBrain, client_from_options, tool_schemas
from .client import AnthropicClient, GeminiClient, LLMError, OpenAIClient, ToolUse, make_client

__all__ = ["ModelBrain", "tool_schemas", "client_from_options", "CLASSIFY_TOOL", "CLASSIFY_SYSTEM",
           "make_client", "AnthropicClient", "GeminiClient", "OpenAIClient", "ToolUse", "LLMError"]
