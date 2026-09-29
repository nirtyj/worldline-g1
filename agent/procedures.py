"""Procedural memory: a graph of what usually works next, learned from past episodes.

After the Procedural Graphs idea (Lu et al., 2026): procedural knowledge kept as
(step, relation, step) edges instead of being left implicit in a long history.

  nodes   abstract steps with their outcome: "navigate:ok", "look", "verify",
          "check_reachability:not_seen_here", "manipulate:pick:ok", "manipulate:place:ok",
          "speak", "ask", "recall", "rejected:manipulate" ...
  edges   "a -> b": how often b followed a in tasks that ended well (ok) or not (fail)
  rules   short learned triplets, e.g. {"after": "manipulate:place:ok", "then": "speak", "why": ...},
          proposed by a model that compares failed tasks with successful ones.
          A rule is used only once it is "kept": it has to survive a scenario-suite
          run without making the results worse (eval/evolve.py).

A task is one request: from a request or correction until the next one. It ended
well if something was delivered to the user.

The graph never commands anything. Before each decision the planner gets a few
lines of guidance for the step it is at; the runtime's rules still decide.

    graph = ProceduralGraph.load()
    graph.learn(load_episodes())              # recount from every episode on disk
    graph.guidance(["navigate:ok", "check_reachability:not_seen_here"])   # -> text or ""

Command line (see eval/evolve.py for the whole loop):
    python -m agent.procedures learn          # recount and print the graph
    python -m agent.procedures propose        # ask the refiner model for candidate rules
    python -m agent.procedures trial | keep | reject | show
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

from .episodes import load_episodes

STORE = Path(os.environ.get("WORLDLINE_RUNS") or Path(__file__).resolve().parents[1] / "runs") / "procedures.json"
MIN_SUPPORT = 2          # an edge needs this many tasks behind it before it is suggested


def step_of(row: dict[str, Any]) -> str | None:
    """A trace row as an abstract step, or None if it isn't one.

    check_reachability:ok|<reason>, manipulate:pick:ok|<status>, navigate:ok|<status>,
    look (the robot looked: an arrival scan or wait_and_observe), verify (a harness check
    after a manipulate or a cancel), speak / ask, recall, list_locations, rejected:<tool>."""
    t = row.get("type")
    if t == "result":
        tool, status, d = row.get("tool") or row.get("skill"), str(row.get("status") or ""), row.get("data") or {}
        ok = status == "succeeded"
        if tool == "check_reachability":
            return "check_reachability:ok" if d.get("reachable") else f"check_reachability:{d.get('reason') or 'no'}"
        if tool in ("observe", "look"):
            why = str(row.get("why") or "")
            if why == "wait_and_observe":
                return None                          # the wait_and_observe row itself is the step
            if why == "arrival" or row.get("source") != "harness":
                return "look"
            return "verify"
        if tool == "wait_and_observe":
            return "look"
        if tool == "manipulate":
            return f"manipulate:{row.get('action') or d.get('action') or 'pick'}:{'ok' if ok else status}"
        if tool in ("recall", "list_locations"):
            return tool
        return f"{tool}:{'ok' if ok else status}"
    if t == "rejected":
        return f"rejected:{row.get('tool')}"
    if t == "say_queued":
        return "ask" if str(row.get("text", "")).rstrip().endswith("?") else "speak"
    if t == "recall":
        return "recall"
    if t in ("stop", "correction"):
        return t
    return None


def tasks_of(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split one episode into tasks: [{"text", "steps", "ok"}]."""
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    heard: dict[str, str] = {}
    deliver_to = next((r.get("deliver_to") for r in rows if r.get("type") == "session_start"), None)
    for r in rows:
        if r.get("type") == "heard":
            heard[r.get("id")] = r.get("text", "")
        if r.get("type") == "classified" and r.get("kind") in ("request", "correction"):
            if cur is not None:
                out.append(cur)
            cur = {"text": heard.get(r.get("id"), ""), "steps": [], "ok": False, "physical": False}
            continue
        if cur is None:
            continue
        if r.get("type") == "delivered":
            cur["ok"] = True
        elif (r.get("type") == "result" and r.get("tool") == "manipulate" and r.get("action") == "place"
              and r.get("status") == "succeeded"
              and deliver_to is None and not any(x.get("type") == "delivered" for x in rows)):
            cur["ok"] = True              # an older episode without delivery events: a good place counts
        step = step_of(r)
        if step:
            cur["steps"].append(step)
            if step.split(":")[0] in ("navigate", "manipulate"):
                cur["physical"] = True
    if cur is not None:
        out.append(cur)
    return [t for t in out if t["physical"]]          # only tasks that moved something count


class ProceduralGraph:
    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self.path = STORE if path is None else path
        d = data or {}
        self.nodes: dict[str, int] = dict(d.get("nodes") or {})
        self.edges: dict[str, dict[str, int]] = dict(d.get("edges") or {})
        self.rules: list[dict[str, Any]] = list(d.get("rules") or [])
        self.tasks: dict[str, int] = dict(d.get("tasks") or {"ok": 0, "fail": 0})
        self.learned_wall: float | None = d.get("learned_wall")

    @classmethod
    def load(cls, path: Path | None = None) -> "ProceduralGraph":
        path = STORE if path is None else path
        try:
            return cls(json.loads(path.read_text()), path)
        except (OSError, ValueError):
            return cls(None, path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.learned_wall = round(time.time())
        self.path.write_text(json.dumps({"nodes": self.nodes, "edges": self.edges, "rules": self.rules,
                                         "tasks": self.tasks, "learned_wall": self.learned_wall}, indent=1))

    # ------------------------------------------------------------------
    def learn(self, episodes: list[list[dict[str, Any]]]) -> None:
        """Recount nodes and edges from scratch (rules are kept)."""
        nodes: collections.Counter[str] = collections.Counter()
        edges: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"ok": 0, "fail": 0})
        tasks = {"ok": 0, "fail": 0}
        for rows in episodes:
            for task in tasks_of(rows):
                outcome = "ok" if task["ok"] else "fail"
                tasks[outcome] += 1
                seq = ["start"] + task["steps"]
                seen: set[str] = set()
                for a, b in zip(seq, seq[1:]):
                    nodes[b] += 1
                    key = f"{a} -> {b}"
                    if key not in seen:                     # count each edge once per task
                        edges[key][outcome] += 1
                        seen.add(key)
        self.nodes, self.edges, self.tasks = dict(nodes), {k: dict(v) for k, v in edges.items()}, tasks
        self.learned_wall = round(time.time())

    def guidance(self, steps: list[str], limit: int = 3) -> str:
        """A few lines for the step the current task is at. Empty when nothing is learned."""
        at = steps[-1] if steps else "start"
        options = []
        for key, c in self.edges.items():
            a, b = key.split(" -> ")
            n = c["ok"] + c["fail"]
            if a == at and n >= MIN_SUPPORT:
                options.append((c["ok"] / n, n, b, c["ok"]))
        options.sort(key=lambda o: (-o[0], -o[1]))
        lines = []
        if options:
            shown = ", ".join(f"{b} ({ok} of {n} tasks went well)" for _, n, b, ok in options[:limit])
            lines.append(f"after {at}, past tasks continued with: {shown}")
        for r in self.rules:
            if r.get("status") in ("kept", "trial") and r.get("after") in (at, "any"):
                lines.append(f"learned rule: after {r['after']}, {r['then']} ({r.get('why', '')})".rstrip(" ()"))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def failure_digest(self, episodes: list[list[dict[str, Any]]], limit: int = 12) -> str:
        """Failed, slow and quick tasks side by side, for the refiner model. A slow task
        succeeded with more steps than usual: there is something to learn from it too."""
        all_tasks = [t for rows in episodes for t in tasks_of(rows)]
        good = [t for t in all_tasks if t["ok"]]
        lengths = sorted(len(t["steps"]) for t in good)
        median = lengths[len(lengths) // 2] if lengths else 0
        fmt = lambda t: f'"{t["text"]}" ({len(t["steps"])} steps): ' + " → ".join(t["steps"][:30])
        bad = [fmt(t) for t in all_tasks if not t["ok"]]
        slow = [fmt(t) for t in good if len(t["steps"]) > median + 2]
        quick = [fmt(t) for t in good if len(t["steps"]) <= median]
        return ("FAILED TASKS\n" + ("\n".join(bad[-limit:]) or "(none)") +
                f"\n\nSLOW TASKS (succeeded, but with more than {median + 2} steps)\n" + ("\n".join(slow[-limit:]) or "(none)") +
                "\n\nQUICK TASKS\n" + ("\n".join(quick[-limit:]) or "(none)"))


PROPOSE_TOOL = {
    "name": "propose_rules",
    "description": "Propose up to three short procedural rules that would turn the failed tasks into successful ones.",
    "parameters": {"type": "object", "properties": {"rules": {"type": "array", "items": {
        "type": "object",
        "properties": {"after": {"type": "string", "description": "The step the rule applies after, exactly as written in the traces (or 'any')."},
                       "then": {"type": "string", "description": "What to do next, in a few words."},
                       "why": {"type": "string", "description": "The evidence, in a few words."}},
        "required": ["after", "then", "why"]}}}, "required": ["rules"]},
}
PROPOSE_SYSTEM = """You improve a home robot's procedures. You get step traces of tasks that failed, \
tasks that succeeded slowly, and tasks that succeeded quickly. Steps look like navigate:ok, look (the \
robot looked around), verify (an automatic look after a pick or place), check_reachability:not_seen_here, \
manipulate:pick:ok, manipulate:place:ok, speak, ask, recall, rejected:manipulate. Propose at most three rules of the form "after <step>, <do this>" that \
would have prevented the failures or removed the wasted steps of the slow tasks, without hurting the \
quick ones. Use step names exactly as they appear. Only propose rules the traces support; if there is \
nothing to fix, propose nothing."""


async def propose(graph: ProceduralGraph, episodes: list[list[dict[str, Any]]], model: str = "gemini-3.8-flash") -> list[dict[str, Any]]:
    from llmkit.client import LLMError, make_client
    client = make_client("gemini", model, thinking_budget=None, max_tokens=4096, timeout=90.0)
    for attempt in range(3):                      # a long thought can cut the call short: try again
        try:
            call = await client.tool_call(PROPOSE_SYSTEM, graph.failure_digest(episodes), [PROPOSE_TOOL])
            break
        except LLMError as e:
            if attempt == 2 or "MALFORMED" not in str(e):
                raise
    new = []
    for r in (call.args.get("rules") or [])[:3]:
        rule = {"id": f"r{int(time.time())}{len(graph.rules)}", "after": str(r.get("after", "any")),
                "then": str(r.get("then", "")).strip(), "why": str(r.get("why", "")).strip(),
                "status": "candidate", "proposed_wall": round(time.time())}
        if rule["then"]:
            graph.rules.append(rule)
            new.append(rule)
    return new


def _load_env() -> None:
    import os
    import re
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"'))


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn, show and evolve the procedural graph.")
    ap.add_argument("cmd", choices=["learn", "show", "propose", "trial", "keep", "reject"])
    args = ap.parse_args()
    g = ProceduralGraph.load()
    episodes = load_episodes()
    if args.cmd in ("learn", "propose"):
        g.learn(episodes)
    if args.cmd == "propose":
        _load_env()
        for r in asyncio.run(propose(g, episodes)):
            print(f"candidate {r['id']}: after {r['after']}, {r['then']}  ({r['why']})")
    for r in g.rules:
        if args.cmd == "trial" and r["status"] == "candidate":
            r["status"] = "trial"
        elif args.cmd == "keep" and r["status"] == "trial":
            r["status"] = "kept"
        elif args.cmd == "reject" and r["status"] == "trial":
            r["status"] = "rejected"
    g.save()
    print(f"tasks: {g.tasks}  nodes: {len(g.nodes)}  edges: {len(g.edges)}")
    for key, c in sorted(g.edges.items(), key=lambda kv: -(kv[1]['ok'] + kv[1]['fail']))[:25]:
        print(f"  {key:<48} ok {c['ok']:>3}  fail {c['fail']:>3}")
    for r in g.rules:
        print(f"  rule [{r['status']}] after {r['after']}: {r['then']}  ({r['why']})")


if __name__ == "__main__":
    main()
