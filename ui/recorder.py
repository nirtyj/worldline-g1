"""Records every model call a brain makes: the exact input, the response,
timing, tokens, and which request version was current before and after.

It wraps two methods on the brain instance, without changing the brain:

  _call     the single place a ModelBrain talks to the model (llmkit/brain.py)
  classify  so utterances the brain labels by rule, with no model call
            (agent/model.py's fast path for "stop", "go ahead", answers),
            show up too

Each finished call is also appended to a plain-text log.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable


class CallRecorder:
    def __init__(self, brain: Any, clock: Any, version: Callable[[], int], log_path: Path | None) -> None:
        self.clock = clock
        self.version = version
        self.calls: list[dict[str, Any]] = []
        self.log_path = log_path
        self._rev = 0
        self._systems_written: set[str] = set()
        self._model_calls = 0
        if self._is_model_call(getattr(brain, "_call", None)):
            self._wrap_call(brain)
        if hasattr(brain, "classify"):
            self._wrap_classify(brain)

    # ------------------------------------------------------------------
    def changed_since(self, rev: int) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["rev"] > rev]

    @property
    def rev(self) -> int:
        return self._rev

    def _touch(self, rec: dict[str, Any]) -> None:
        self._rev += 1
        rec["rev"] = self._rev

    @staticmethod
    def _is_model_call(fn: Any) -> bool:
        """llmkit's ModelBrain._call(system, text, tools) coroutine; a scripted brain may have an
        unrelated helper of the same name, which must not be wrapped."""
        if fn is None or not inspect.iscoroutinefunction(fn):
            return False
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            return False
        return params[:3] == ["system", "text", "tools"]

    # ------------------------------------------------------------------
    def _wrap_call(self, brain: Any) -> None:
        original = brain._call

        async def recorded(system: str, text: str, tools: list[dict[str, Any]]):
            self._model_calls += 1
            names = [t.get("name") for t in tools]
            rec: dict[str, Any] = {
                "n": len(self.calls) + 1, "via": "model",
                "purpose": "classify" if names == ["classify"] else "next_action",
                "status": "pending", "t_start": round(self.clock.now(), 2), "t_end": None,
                "version_start": self.version(), "version_end": None,
                "system": system, "input": text, "tools": names, "tool_schemas": tools,
                "response": None, "error": None, "latency_s": None, "tokens_in": None, "tokens_out": None,
            }
            self.calls.append(rec)
            self._touch(rec)
            wall0 = time.monotonic()
            try:
                use = await original(system, text, tools)
            except BaseException as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}")
                self._finish(rec, wall0)
                raise
            rec.update(status="ok", response={"tool": use.name, "args": use.args, "text": use.text},
                       tokens_in=use.input_tokens, tokens_out=use.output_tokens)
            self._finish(rec, wall0)
            return use

        brain._call = recorded

    def _wrap_classify(self, brain: Any) -> None:
        original = brain.classify

        async def classify(utterance: Any, ctx: Any) -> str:
            before = self._model_calls
            kind = await original(utterance, ctx)
            if self._model_calls == before:        # decided by a rule, not the model
                now = round(self.clock.now(), 2)
                rec = {"n": len(self.calls) + 1, "via": "rule", "purpose": "classify", "status": "ok",
                       "t_start": now, "t_end": now, "version_start": self.version(),
                       "version_end": self.version(), "system": None,
                       "input": f"LATEST UTTERANCE\n{utterance.text}", "tools": [], "tool_schemas": [],
                       "response": {"tool": "classify", "args": {"kind": kind}, "text": ""},
                       "error": None, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0}
                self.calls.append(rec)
                self._touch(rec)
                self._write(rec)
            return kind

        brain.classify = classify

    def record_external(self, rec: dict[str, Any]) -> None:
        """A call made outside the planner, e.g. System 1's."""
        rec = {"n": len(self.calls) + 1, "status": "ok", "error": None, "tokens_in": None, "tokens_out": None,
               "tools": [], "tool_schemas": [], "system": None, **rec}
        self.calls.append(rec)
        self._touch(rec)
        self._write(rec)

    def _finish(self, rec: dict[str, Any], wall0: float) -> None:
        rec["t_end"] = round(self.clock.now(), 2)
        rec["latency_s"] = round(time.monotonic() - wall0, 2)
        rec["version_end"] = self.version()
        self._touch(rec)
        self._write(rec)

    # ------------------------------------------------------------------
    def _write(self, rec: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        lines = []
        head = (f"==== #{rec['n']}  {rec['purpose']}  "
                f"{'(by rule, no model call)' if rec['via'] == 'rule' else ''}"
                f"sim t={rec['t_start']}->{rec['t_end']} s  latency {rec['latency_s']} s  "
                f"request v{rec['version_start']}")
        if rec["version_end"] != rec["version_start"]:
            head += f" -> v{rec['version_end']} (the answer came back STALE)"
        if rec["tokens_in"]:
            head += f"  tokens {rec['tokens_in']} in / {rec['tokens_out']} out"
        lines.append(head)
        system = rec.get("system")
        if system:
            if system in self._systems_written:
                lines.append("---- SYSTEM PROMPT: same as before, not repeated")
            else:
                self._systems_written.add(system)
                lines += ["---- SYSTEM PROMPT", system]
        lines += ["---- INPUT", rec["input"]]
        if rec["tools"]:
            lines.append("---- TOOLS OFFERED: " + ", ".join(rec["tools"]))
        lines.append("---- RESPONSE")
        if rec["error"]:
            lines.append("ERROR " + rec["error"])
        elif rec["response"]:
            r = rec["response"]
            args = ", ".join(f"{k}={v!r}" for k, v in (r.get("args") or {}).items())
            lines.append(f"{r['tool']}({args})")
            if r.get("text"):
                lines.append("text: " + r["text"])
        lines.append("")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write("\n".join(lines) + "\n")
