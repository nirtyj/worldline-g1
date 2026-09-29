"""The page's GR00T strip (PLAN 9.4 and F2; docs/groot_arms_design.md 5): one `manipulate` session through
`groot_arms` at a time, built from what the page already receives. No simulator, policy or body access here.

Fed by (field names as `services/executors/groot_arms.py` emits them into the robot's event sink, v0.1 of
docs/contracts/arm_chunk.md; whatever arrives is shown, nothing is invented):

  trace rows   `started` tool=manipulate (args.skill_id, execution_id)   -> opens a session when the skill is GR00T's
               `result`  tool=manipulate                                 -> the honest outcome (ManipulationResult
                                                                            data: status, reason, holding, gt, label,
                                                                            inferences, chunks_dropped, latency_ms,
                                                                            clamped_frac, stall_s_max, cancel_ack_ms)
  events       `manip.phase`     {execution_id, phase, skill, executor, label}
               `groot.inference` {session, ok, latency_ms, chunk_idx?, dropped?: reason, error?}
               `arm.progress`    {session, chunk_idx, k, stall_s, clamped_frac, latency_ms, cross_fades, inferences,
                                  dropped: {reason: n}, source: body | client}. From the body (B.8) latency_ms is the
                                  chunk's age on arrival; from the client (before B.8) it is the last inference's.
               `groot.stall`     {session, stall_s}
               `policy.health`   {ok, state, detail, endpoint, checkpoint, executor}
  capabilities the robot facade's health entries (`manipulation`), from the frame's stack, until an event arrives

The session id is the execution id (arm_chunk.md: session_id = the manipulate execution id). Every GR00T skill is
labelled `experimental` (owner decision, PLAN 0.8): the off-the-shelf N1.7 checkpoint is not expected to finish a
grasp in a house, so the outcome is shown as the result says it, with the ground-truth judge's numbers next to it.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable

LABEL = "experimental"
GROOT_EXECUTORS = frozenset({"groot_arms", "groot_sonic"})
PROGRESS_EVENTS = frozenset({"arm.progress", "arm_progress"})
INFERENCE_EVENTS = frozenset({"groot.inference", "vla.inference", "groot_inference"})
PHASE_EVENTS = frozenset({"manip.phase", "manip_phase"})
POLICY_EVENTS = frozenset({"policy.health", "policy_health"})
KEEP_S = 30.0                  # a finished session stays on the strip this long (sim seconds)


def is_groot(skill: Any = None, executor: Any = None) -> bool:
    return str(executor or "") in GROOT_EXECUTORS or str(skill or "").startswith("groot.")


def _num(v: Any) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def _sid(e: dict[str, Any]) -> str | None:
    s = e.get("session") or e.get("session_id") or e.get("execution_id")
    return str(s) if s else None


class GrootStrip:
    def __init__(self) -> None:
        self.s: dict[str, Any] | None = None
        self.policy: dict[str, Any] = {}

    # ------------------------------------------------------------------ inputs
    def _open(self, sid: str, t: float, skill: Any = None, executor: Any = None, args: dict | None = None) -> None:
        a = dict(args or {})
        self.s = {"session": sid, "skill": skill or a.get("skill_id"), "executor": executor or "groot_arms",
                  "action": a.get("action"), "object": a.get("object_id") or a.get("object_type"), "label": LABEL,
                  "status": "running", "phase": None, "t_start": t, "t_end": None, "chunk_idx": None, "k": None,
                  "inferences": 0, "latencies": [], "latency_last_ms": None, "latency_p50_result": None,
                  "chunk_age_ms": None, "clamped_frac": None, "stall_s": None, "stall_warned": False,
                  "cross_fades": None, "dropped": {}, "errors": 0, "progress_source": None, "outcome": None}

    def _mine(self, sid: str | None) -> bool:
        return self.s is not None and sid is not None and sid == self.s["session"]

    def trace(self, rows: Iterable[dict[str, Any]]) -> None:
        for r in rows:
            if r.get("tool") != "manipulate":
                continue
            args = r.get("args") or {}
            data = r.get("data") or {}
            skill = args.get("skill_id") or data.get("skill")
            executor = r.get("executor") or data.get("executor")
            sid = r.get("execution_id")
            if r.get("type") == "started" and sid and is_groot(skill, executor):
                self._open(str(sid), float(r.get("t") or 0.0), skill, executor, args)
            elif r.get("type") == "result" and sid and (self._mine(str(sid)) or is_groot(skill, executor)):
                if not self._mine(str(sid)):
                    self._open(str(sid), float(r.get("t_start") or r.get("t") or 0.0), skill, executor,
                               {**args, **data})
                self._result(r, data)

    def _result(self, r: dict[str, Any], data: dict[str, Any]) -> None:
        s = self.s
        assert s is not None
        s["status"] = str(r.get("status") or "")
        s["t_end"] = float(r.get("t") or r.get("t_end") or 0.0)
        s["skill"] = data.get("skill") or s["skill"]
        s["executor"] = data.get("executor") or r.get("executor") or s["executor"]
        s["label"] = data.get("label") or s["label"]
        s["phase"] = data.get("phase") or s["phase"]
        if data.get("inferences") is not None:
            s["inferences"] = int(data["inferences"])
        if data.get("chunks_dropped"):
            s["dropped"] = dict(data["chunks_dropped"])
        if _num(data.get("clamped_frac")) is not None:
            s["clamped_frac"] = _num(data.get("clamped_frac"))
        if _num(data.get("stall_s_max")) is not None:
            s["stall_s"] = _num(data.get("stall_s_max"))
        lat = data.get("latency_ms")
        if isinstance(lat, dict) and _num(lat.get("p50")) is not None:
            s["latency_p50_result"] = _num(lat.get("p50"))
        holding = data.get("holding")
        reason = data.get("reason")
        gt = data.get("gt") if isinstance(data.get("gt"), dict) else None
        text = s["status"] + (f" · {reason}" if reason else "")
        if holding is not None:
            text += f" · in hand (GT): {'yes' if holding else 'no'}"
        if gt and _num(gt.get("lift_max_m")) is not None:
            text += f" · lifted at most {float(gt['lift_max_m']) * 100:.0f} cm"
        s["outcome"] = {"status": s["status"], "reason": reason, "holding": holding, "gt": gt,
                        "late": bool(r.get("late")), "summary": r.get("summary"), "text": text,
                        "cancel_ack_ms": data.get("cancel_ack_ms"), "carry": data.get("carry")}

    def events(self, evs: Iterable[dict[str, Any]]) -> None:
        for e in evs:
            ty = str(e.get("type") or "")
            if ty in POLICY_EVENTS:
                state = e.get("state")
                ok = e.get("ok") if e.get("ok") is not None else (state == "ok" if state else None)
                self.policy = {k: v for k, v in {"ok": ok, "state": state, "detail": e.get("detail"),
                                                  "endpoint": e.get("endpoint") or e.get("port"),
                                                  "checkpoint": e.get("checkpoint"), "source": "policy.health"}.items()
                               if v is not None}
                continue
            sid = _sid(e)
            if ty in PHASE_EVENTS:
                if not self._mine(sid) and sid and is_groot(e.get("skill"), e.get("executor")):
                    self._open(sid, float(e.get("t") or 0.0), e.get("skill"), e.get("executor"))
                if self._mine(sid):
                    self.s["phase"] = e.get("phase") or self.s["phase"]      # type: ignore[index]
                    self.s["label"] = e.get("label") or self.s["label"]      # type: ignore[index]
            elif ty in PROGRESS_EVENTS and self._mine(sid):
                self._progress(e)
            elif ty in INFERENCE_EVENTS and self._mine(sid):
                s = self.s
                assert s is not None
                if e.get("ok", True):
                    s["inferences"] += 1
                    x = _num(e.get("latency_ms"))
                    if x is not None:
                        s["latencies"] = (s["latencies"] + [x])[-200:]
                        s["latency_last_ms"] = x
                    why = e.get("dropped")
                    if isinstance(why, str) and why:
                        s["dropped"][why] = s["dropped"].get(why, 0) + 1
                else:
                    s["errors"] += 1
                if e.get("chunk_idx") is not None:
                    s["chunk_idx"] = e.get("chunk_idx")
            elif ty == "groot.stall" and self._mine(sid):
                self.s["stall_s"] = _num(e.get("stall_s"))                   # type: ignore[index]
                self.s["stall_warned"] = True                                # type: ignore[index]

    def _progress(self, e: dict[str, Any]) -> None:
        s = self.s
        assert s is not None
        for k in ("chunk_idx", "k", "cross_fades"):
            if e.get(k) is not None:
                s[k] = e.get(k)
        for k in ("stall_s", "clamped_frac"):
            if _num(e.get(k)) is not None:
                s[k] = _num(e.get(k))
        if e.get("inferences") is not None:                    # the client's own count: authoritative
            s["inferences"] = int(e["inferences"])
        if isinstance(e.get("dropped"), dict):
            s["dropped"] = dict(e["dropped"])
        s["progress_source"] = e.get("source") or s["progress_source"]
        if e.get("source") == "body" and _num(e.get("latency_ms")) is not None:
            s["chunk_age_ms"] = _num(e.get("latency_ms"))      # B.8: the chunk's age on arrival at the body

    # ------------------------------------------------------------------ output
    def snapshot(self, now: float, services: dict[str, Any] | None = None) -> dict[str, Any] | None:
        s = self.s
        if s is None or (s["t_end"] is not None and now - s["t_end"] > KEEP_S):
            return None
        lat = s["latencies"]
        policy = dict(self.policy)
        h = (services or {}).get("manipulation")
        if not policy and isinstance(h, dict):
            policy = {"ok": h.get("ok"), "state": h.get("state"), "detail": h.get("detail"), "source": "manipulation"}
        out = {k: v for k, v in s.items() if k not in ("latencies", "latency_p50_result")}
        p50 = round(statistics.median(lat), 1) if lat else s["latency_p50_result"]
        out.update(latency_ms={"p50": p50, "last": s["latency_last_ms"], "n": len(lat)},
                   running=s["t_end"] is None, age_s=round(max(0.0, now - (s["t_end"] or s["t_start"])), 1),
                   stall=bool(s["stall_warned"] or (s["stall_s"] or 0.0) > 0.0), policy=policy)
        return out
