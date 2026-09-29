"""The M2a offline episode: Worldline end to end on a laptop, with no simulator, no LLM and no keys.

    .venv-rt/bin/python -m eval.offline_episode                               # H40, "Bring me the alarm clock."
    .venv-rt/bin/python -m eval.offline_episode --out docs/traces/m2a_offline_f1_h40 --speed 40

What runs (every layer is the real M2a code; only the model, System 1 and the body are stand-ins):

    eval/suite.py Run (the referee: scores on frame.truth, like every eval scenario)
      -> websocket -> ui/server.py Hub + Session (the page's server)
      -> System 1 = tests/kept/system1_stub.py (labels "bring me the ..." as a request; sees the head frames;
         reports one fixed observation)
      -> agent/ Runtime (harness, validation, belief, arrival scans, goal check, memory, narrator)
      -> CompositeBrain(brains/scripted.py ScriptedPlanner): one tool call per turn, from belief only
      -> robot.factory.build("lite") = robot/bridge.py G1Robot -> services/ (navigation, observation,
         reachability, manipulation, speech)
      -> robot/lite_body.py LiteBody + world/lite_world.py LiteWorld over the recorded MolmoSpaces house
         (procthor-train-40 = Worldline's H40), world/frames.py LiteFrames (schematic head frames)

It passes when the eval's fetch scenario passes (alarm_clock_1 on the user's surface in world truth, robot idle)
AND the trace shows the F1 shape (PLAN 2.2), in order: navigate -> arrival scan -> check_reachability [-> reposition
-> check again] -> manipulate(pick), labelled -> verify glance -> navigate(user) -> manipulate(place) -> goal_check
ok (after the place's result) and delivered (after navigate(user), for that same place execution) AND every result
row carries the full envelope (PLAN 5.3: execution_id, tool, status, observation_id, generation, control_epoch,
t_start, t_end, late), with speak, recall and rejections recorded as result rows too. On lite everything is a
STEPPING STONE (executor "lite"): a labelled fallback pass, never a target pass.

The same checks run against a live page server (M2b: the box's `sonic` profile through `00_infra/tunnel.sh 8765`):

    .venv-rt/bin/python -m eval.offline_episode --url ws://127.0.0.1:8765/ws --profile sonic

There the page's own planner and System 1 are used (whatever the server was started with), and a pass is a target
pass only if every navigate/manipulate result came from a target executor (api.results.is_target).

Writes <out>.jsonl (every trace row and sim event the page received, in order) and <out>.summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIO = "fetch_other_room"          # eval/scenes.yaml: H40, alarm_clock_1, "Bring me the alarm clock."
SYSTEM1_STUB = "tests.kept.system1_stub:create"
PLANNER = "brains.scripted:create"
# The envelope every trace `result` row carries (PLAN 5.3 ToolResult; the harness logs one row per ToolResult), and
# the row's `kind`: a tool that ran, a line of speech, recall or a rejection (api.results.RESULT_ROW_KINDS when the
# runtime defines it; these literals are its fallback).
ENVELOPE_KEYS = ("execution_id", "tool", "status", "observation_id", "generation", "control_epoch", "t_start",
                 "t_end", "late")
ENVELOPE_STATUSES = ("succeeded", "failed", "cancelled", "timed_out", "rejected")
ROW_KINDS = ("tool", "speech", "recall", "rejection")
try:
    from api.results import RESULT_ROW_KINDS as ROW_KINDS  # type: ignore  # noqa: F811
except ImportError:
    pass
KIND_OF = {"speak": "speech", "recall": "recall", "rejection": "rejection"}


# ----------------------------------------------------------------------------------------------------------
# What the trace must show
# ----------------------------------------------------------------------------------------------------------
def _results(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in trace if r.get("type") == "result"]


def _is(tool: str, **kw: Any) -> Callable[[dict[str, Any]], bool]:
    def pred(r: dict[str, Any]) -> bool:
        if r.get("tool") != tool:
            return False
        for k, v in kw.items():
            got = r.get(k) if k in r else (r.get("data") or {}).get(k)
            if callable(v):
                if not v(got, r):
                    return False
            elif got != v:
                return False
        return True
    return pred


def labelled(ex: Any, r: dict[str, Any]) -> bool:
    """A result names its executor, and a STEPPING STONE says so in the line the planner reads ([fallback])."""
    from api.results import is_target
    return bool(ex) and (is_target(ex) or "fallback" in str(r.get("summary") or ""))


def f1_steps(user_keypoint: str, user_surface: str) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    """The F1 fetch (PLAN 2.2) as an ordered list of trace-row predicates (other rows may sit in between)."""
    return [
        ("System 1 labelled the request", lambda r: r.get("type") == "classified" and r.get("kind") == "request"
         and (r.get("directive") or {}).get("source") == "system1"),
        ("navigate to a stand (executor named, labelled)",
         lambda r: r.get("type") == "result" and _is("navigate", status="succeeded", executor=labelled)(r)),
        # a real scan: data.mode "scan" (a scan whose first turn failed falls back to a glance, data.mode "glance")
        ("arrival scan", lambda r: r.get("type") == "result" and _is("observe", status="succeeded", action="scan",
                                                                     mode="scan")(r)),
        ("check_reachability sees the alarm clock",
         lambda r: r.get("type") == "result" and _is("check_reachability", status="succeeded",
                                                      object_id="alarm_clock_1")(r)),
        ("check_reachability says reachable",
         lambda r: r.get("type") == "result" and _is("check_reachability", reachable=True)(r)),
        ("pick (executor and skill named, labelled)",
         lambda r: r.get("type") == "result" and _is("manipulate", status="succeeded", action="pick",
                                                      executor=labelled)(r)),
        ("verify glance after the pick",
         lambda r: r.get("type") == "result" and _is("observe", status="succeeded", action="glance")(r)),
        ("navigate to the user",
         lambda r: r.get("type") == "result" and _is("navigate", status="succeeded", at=user_keypoint)(r)),
        ("place on the user's surface (labelled)",
         lambda r: r.get("type") == "result" and _is("manipulate", status="succeeded", action="place",
                                                      executor=labelled)(r)),
        # the goal check runs after the place's result and its verify glance (PLAN 5.12)
        ("goal check ok", lambda r: r.get("type") == "goal_check" and r.get("ok") is True
         and r.get("object") == "alarm_clock_1"),
        # the harness logs `delivered` while it applies that place's result, just before the result row, so
        # delivered is ordered after navigate(user) and must name the same place execution
        ("delivered", lambda r: r.get("type") == "delivered" and r.get("object") == "alarm_clock_1"
         and r.get("surface") == user_surface,
         {"after": "navigate to the user", "same_execution_as": "place on the user's surface (labelled)"}),
    ]


Step = tuple  # (name, pred) or (name, pred, {"after": step name, "same_execution_as": step name})


def check_sequence(trace: list[dict[str, Any]], steps: list[Step]) -> list[dict[str, Any]]:
    """Each step matched in order (a subsequence of the trace); one entry per step with the row it matched.

    A step is searched from the row after the previous matched step, or, with {"after": name}, from the row
    after that earlier step (so it may sit before the previous step, never before its anchor). With
    {"same_execution_as": name} the row's execution_id must be that step's. A step whose anchor is missing fails."""
    out: list[dict[str, Any]] = []
    rows: dict[str, int] = {}
    cursor = 0
    for step in steps:
        name, pred = step[0], step[1]
        opts = step[2] if len(step) > 2 else {}
        start = cursor
        if opts.get("after") is not None:
            anchor = rows.get(opts["after"])
            start = None if anchor is None else anchor + 1
        eid = None
        if opts.get("same_execution_as") is not None:
            k = rows.get(opts["same_execution_as"])
            eid = trace[k].get("execution_id") if k is not None else None
            if eid is None:
                start = None
        j = None if start is None else next(
            (k for k in range(start, len(trace)) if pred(trace[k]) and (eid is None or trace[k].get("execution_id") == eid)),
            None)
        out.append({"step": name, "ok": j is not None, "row": trace[j].get("i") if j is not None else None,
                    "t": trace[j].get("t") if j is not None else None})
        if j is not None:
            rows[name] = j
            cursor = max(cursor, j + 1)
    return out


def manipulations(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every manipulate that ran, with each attempt behind its result (PLAN 6.4 groot_then_script: the GR00T attempt
    and the labelled fallback are both in data.attempts): which executor did the manipulation, and which tried."""
    out = []
    for r in _results(trace):
        if r.get("tool") != "manipulate" or r.get("kind") == "rejection" or r.get("status") == "rejected":
            continue
        d = r.get("data") or {}
        out.append({"execution_id": r.get("execution_id"), "action": r.get("action") or d.get("action"),
                    "status": r.get("status"), "reason": d.get("reason"), "executor": r.get("executor"),
                    "skill": d.get("skill"), "arm": d.get("arm"), "stepping_stone": d.get("stepping_stone"),
                    "attempts": [{k: a.get(k) for k in ("executor", "skill", "label", "status", "reason", "duration_s")}
                                 for a in d.get("attempts") or []],
                    "fallback_from": d.get("fallback_from")})
    return out


def manipulation_line(m: dict[str, Any]) -> str:
    """'pick failed(ik_unreachable) by sonic_arm_script [STEPPING STONE]: groot_arms failed(grasp_missed) ->
    sonic_arm_script failed(ik_unreachable)'."""
    tries = " -> ".join(f"{a.get('executor')} {a.get('status')}" + (f"({a.get('reason')})" if a.get("reason") else "")
                        for a in m["attempts"]) or "-"
    what = f"{m['action']} {m['status']}" + (f"({m['reason']})" if m.get("reason") else "")
    stone = " [STEPPING STONE]" if m.get("stepping_stone") else ""
    return f"{what} by {m['executor']}{stone}, {m.get('arm') or '?'} arm: {tries}"


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def envelope_problems(r: dict[str, Any]) -> list[str]:
    """What is missing or malformed in one trace `result` row's envelope (PLAN 5.3) and its `kind`, as short
    messages."""
    out = [f"no {k}" for k in ENVELOPE_KEYS + ("kind",) if k not in r]
    st = r.get("status")
    schema_reject = st == "rejected" and "schema" in (r.get("stage"), (r.get("data") or {}).get("stage"))
    if "execution_id" in r and not (isinstance(r["execution_id"], str) and r["execution_id"]) and not schema_reject:
        out.append("execution_id empty")
    if "tool" in r and not (isinstance(r["tool"], str) and r["tool"]):
        out.append("tool empty")
    if "status" in r and st not in ENVELOPE_STATUSES:
        out.append(f"status {st!r}")
    if "observation_id" in r and not (isinstance(r["observation_id"], str) and r["observation_id"]):
        out.append("observation_id empty")
    for k in ("generation", "control_epoch"):
        if k in r and not (_int(r[k]) and r[k] >= 0):
            out.append(f"{k} {r[k]!r}")
    for k in ("t_start", "t_end"):
        if k in r and not _num(r[k]):
            out.append(f"{k} {r[k]!r}")
    if _num(r.get("t_start")) and _num(r.get("t_end")) and r["t_end"] < r["t_start"] - 1e-6:
        out.append("t_end before t_start")
    if "late" in r and not isinstance(r["late"], bool):
        out.append(f"late {r['late']!r}")
    if r.get("kind") is not None and r.get("kind") not in ROW_KINDS:
        out.append(f"kind {r.get('kind')!r}")
    return out


def contract_checks(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The envelope contract on every result row, and the rows that must exist as results (PLAN 5.3, 5.4)."""
    res = _results(trace)
    by_eid: dict[str, list[dict[str, Any]]] = {}
    for r in res:
        if r.get("execution_id"):
            by_eid.setdefault(str(r["execution_id"]), []).append(r)
    bad_env = [(r.get("execution_id") or r.get("tool"), p) for r in res if (p := envelope_problems(r))]
    upper = [r.get("execution_id") for r in res if str(r.get("status")) != str(r.get("status")).lower()]
    # a rejection never ran an executor (CAPABILITY: "policy unavailable: ..."); its own check is the kind check
    manip = [r for r in res if r.get("tool") == "manipulate" and r.get("kind") != "rejection"
             and r.get("status") != "rejected"]
    # speak, recall and rejections are ToolResults too: each gets a `result` row with a `kind`
    wanted = [("speak", r.get("execution_id")) for r in trace if r.get("type") == "say_queued"]
    wanted += [("recall", r.get("execution_id")) for r in trace if r.get("type") == "recall"]
    wanted += [("rejection", r.get("execution_id")) for r in trace if r.get("type") == "rejected"]
    missing_rows, no_kind = [], []
    for what, eid in wanted:
        rows = by_eid.get(str(eid), []) if eid else []
        if what == "rejection":
            rows = [x for x in rows if x.get("status") == "rejected"]
        else:
            rows = [x for x in rows if x.get("tool") == what]
        if not rows:
            missing_rows.append((what, eid))
        elif not any(x.get("kind") == KIND_OF[what] for x in rows):
            no_kind.append((what, eid, [x.get("kind") for x in rows]))
    late_rows = [r.get("execution_id") for r in trace if r.get("type") == "late_result"]
    unmarked = [e for e in late_rows if not any(x.get("late") is True for x in by_eid.get(str(e), []))]
    return [
        {"check": "every result row carries the full envelope (" + ", ".join(ENVELOPE_KEYS) + ") and a kind",
         "ok": bool(res) and not bad_env, "detail": {"rows": len(res), "bad": bad_env[:5], "n_bad": len(bad_env)}},
        {"check": "every result carries an observation_id", "ok": all(r.get("observation_id") for r in res),
         "detail": [r.get("execution_id") for r in res if not r.get("observation_id")][:5]},
        {"check": "lowercase envelope statuses", "ok": not upper, "detail": upper[:5]},
        {"check": "speak, recall and rejections are result rows (kind speech / recall / rejection)",
         "ok": not missing_rows and not no_kind,
         "detail": {"expected": len(wanted), "missing": missing_rows[:5], "no_kind": no_kind[:5]}},
        {"check": "late results are marked late on their result row", "ok": not unmarked, "detail": unmarked[:5]},
        {"check": "every manipulate result names its executor and skill",
         "ok": all((r.get("data") or {}).get("skill") and r.get("executor") for r in manip),
         "detail": [(r.get("execution_id"), r.get("executor"), (r.get("data") or {}).get("skill")) for r in manip]},
    ]


# ----------------------------------------------------------------------------------------------------------
# Running it
# ----------------------------------------------------------------------------------------------------------
async def run_episode(*, speed: float = 40.0, scenario: str = SCENARIO, system1: str = SYSTEM1_STUB,
                      planner: str = PLANNER, timeout_s: float = 240.0, url: str | None = None,
                      profile: str = "lite") -> dict[str, Any]:
    """One episode over a real websocket; returns {score, steps, checks, trace, events, ...}.

    Without ``url`` it starts its own page server in process (lite, the scripted planner, the System 1 stub).
    With ``url`` it drives an already running page server (any profile) and only referees."""
    from websockets.asyncio.client import connect

    from eval import suite

    hub = server = ticker = None
    if url is None:
        from sim.clock import SimClock
        from ui.server import Deps, Hub, _import, serve_hub
        deps = Deps(create_planner=lambda info: _import(planner)(info), clock=lambda _s: SimClock(speed)).resolve()
        hub = Hub(suite.BIND.scene(suite.BIND.scenario(scenario)["house"]), profile, deps=deps, cameras="auto",
                  system1=system1)
        server = await serve_hub(hub, "127.0.0.1", 0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws"
        hub.start_system1()
        ticker = asyncio.create_task(hub.ticker())
    events: list[dict[str, Any]] = []
    t0 = time.monotonic()
    try:
        async with connect(url, max_size=2 ** 24) as ws:
            run = suite.Run(ws, profile)
            orig_ingest = run.ingest

            def ingest(m: dict[str, Any]) -> None:
                if m.get("type") == "init":
                    events.clear()
                if m.get("type") in ("init", "frame"):
                    events.extend(m.get("events") or [])
                orig_ingest(m)
            run.ingest = ingest                                   # type: ignore[method-assign]
            reader = asyncio.create_task(run.reader())
            fn = dict(suite.SCENARIOS)[scenario]
            try:
                passed, note = await asyncio.wait_for(fn(run), timeout_s)
            except asyncio.TimeoutError:
                passed, note = False, f"no result within {timeout_s:.0f} s wall"
            await run.until(lambda: any(r.get("type") == "delivered" for r in run.trace), 5.0)
            await asyncio.sleep(0.5)                              # the last frames (goal_check, final glance)
            score = suite.score(scenario, passed, note, time.monotonic() - t0, run)
            reader.cancel()
            s1 = {"status": hub.s1_status if hub else ((run.frame or {}).get("system1") or {}).get("status"),
                  "frames": getattr(hub.system1, "frames", None) if hub else None,
                  "contexts": len(getattr(hub.system1, "contexts", []) or []) if hub else None,
                  "robot_lines": list(getattr(hub.system1, "robot_lines", []) or []) if hub else None,
                  "observe_calls": len(run.s1_calls("observe")), "route_calls": len(run.s1_calls("route"))}
            layout = (run.init or {}).get("layout") or {}
            m = (run.init or {}).get("map") or {}
            user = (m.get("people") or {}).get("user") or {}
            trace, init_cfg = list(run.trace), (run.init or {}).get("config") or {}
    finally:
        if hub is not None:
            ticker.cancel()
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
    steps = check_sequence(trace, f1_steps(user.get("keypoint", ""), layout.get("user_surface")
                                           or user.get("deliver_to_surface", "")))
    goal_given = any(r.get("type") == "decision" and r.get("tool") == "manipulate"
                     and (r.get("args") or {}).get("action") == "place" and (r.get("args") or {}).get("goal")
                     for r in trace)
    for st in steps:                    # a planner that names no goal (a model may not, for "bring me X") has
        if st["step"] == "goal check ok" and not st["ok"] and not goal_given:   # nothing to check: n/a, not a miss
            st.update(ok=True, note="n/a: no place named a goal; 'delivered' is the check")
    checks = contract_checks(trace)
    noticed = [r for r in trace if r.get("type") in ("observation", "observation_dropped")]
    s1["noticed_rows"] = len(noticed)
    checks.append({"check": "System 1 got gated head frames, and its observation reached the runtime",
                   "ok": (s1["frames"] is None or bool(s1["frames"])) and s1["observe_calls"] > 0 and bool(noticed),
                   "detail": s1})
    from api.results import is_target
    used = [ex for g in ("nav", "manip") for ex in (score["executors_used"].get(g) or {})]
    all_target = bool(used) and all(is_target(ex) for ex in used)
    checks.append({"check": "a target pass only if every body result came from a target executor",
                   "ok": score["target_pass"] == (bool(score["passed"]) and all_target)
                   and score["fallback_pass"] == (bool(score["passed"]) and not all_target),
                   "detail": {"executors": score["executors_used"], "shortcuts": score["shortcuts"],
                              "honesty": score["honesty"], "target_pass": score["target_pass"]}})
    ok = bool(score["passed"]) and all(s["ok"] for s in steps) and all(c["ok"] for c in checks)
    live = hub is None
    return {"ok": ok, "scenario": scenario, "profile": profile, "url": url if live else None,
            "config": init_cfg, "speed": None if live else speed, "system1": None if live else system1,
            "planner": None if live else planner, "user": user, "score": score, "steps": steps, "checks": checks,
            "manipulate": manipulations(trace), "wall_s": round(time.monotonic() - t0, 1), "trace": trace,
            "events": events}


def write(result: dict[str, Any], out: Path) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    jl = out.with_suffix(".jsonl")
    rows = [{"src": "trace", **r} for r in result["trace"]] + [{"src": "event", **e} for e in result["events"]]
    rows.sort(key=lambda r: (float(r.get("t") or 0.0), 0 if r["src"] == "event" else 1, r.get("i") or 0))
    with jl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, sort_keys=True) + "\n")
    summ = out.with_suffix(".summary.json")
    brief = {k: v for k, v in result.items() if k not in ("trace", "events")}
    brief["timeline"] = timeline(result["trace"])
    summ.write_text(json.dumps(brief, indent=1, default=str) + "\n")
    return jl, summ


def timeline(trace: list[dict[str, Any]]) -> list[str]:
    """One line per decision-relevant row: the story of the episode."""
    out = []
    for r in trace:
        t = f"{float(r.get('t') or 0):7.1f}s"
        ty = r.get("type")
        if ty == "classified":
            d = r.get("directive") or {}
            out.append(f"{t}  heard     {d.get('text')!r} -> {r.get('kind')} (by {d.get('source') or 'planner'})")
        elif ty == "decision":
            call = r.get("call") or {}
            out.append(f"{t}  decide    {call.get('tool') or r.get('tool')} {json.dumps(call.get('args') or r.get('args') or {})}")
        elif ty == "started":
            out.append(f"{t}  start     {r.get('tool')}{'(' + str(r.get('action')) + ')' if r.get('action') else ''} "
                       f"{r.get('execution_id') or ''}")
        elif ty == "result":
            out.append(f"{t}  result    {r.get('tool')} {r.get('status')}: {r.get('summary')}")
        elif ty in ("goal_check", "delivered", "rejected", "late_result", "stop"):
            keep = {k: v for k, v in r.items() if k not in ("t", "i", "type", "seq")}
            out.append(f"{t}  {ty:<9} {json.dumps(keep, default=str)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=40.0, help="sim clock speed (the scripted planner is instant)")
    ap.add_argument("--out", default=None, help="output stem (default: $WORLDLINE_RUNS/m2a/offline_<stamp>)")
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--url", default=None, help="referee a running page server instead (e.g. the box's, M2b)")
    ap.add_argument("--profile", default="lite", help="the profile to load (with --url: the server's profile)")
    ap.add_argument("--timeout", type=float, default=240.0, help="wall seconds for the scenario")
    ap.add_argument("--keep-memory", action="store_true",
                    help="use the real runs/ (memory, episodes); default: a throwaway runs dir")
    args = ap.parse_args()
    if not args.url and not args.keep_memory and not os.environ.get("WORLDLINE_RUNS"):
        os.environ["WORLDLINE_RUNS"] = tempfile.mkdtemp(prefix="wl-offline-")
        _isolate_runs(Path(os.environ["WORLDLINE_RUNS"]))
    result = asyncio.run(run_episode(speed=args.speed, scenario=args.scenario, url=args.url, profile=args.profile,
                                     timeout_s=args.timeout))
    stem = Path(args.out) if args.out else Path(os.environ.get("WORLDLINE_RUNS") or ROOT / "runs") / "m2a" / \
        f"offline_{time.strftime('%Y%m%d-%H%M%S')}"
    jl, summ = write(result, stem)
    for line in timeline(result["trace"]):
        print(line)
    print()
    for s in result["steps"]:
        print(f"  {'ok ' if s['ok'] else 'MISSING'}  {s['step']}")
    for c in result["checks"]:
        print(f"  {'ok ' if c['ok'] else 'FAIL'}  {c['check']}")
    for m in result.get("manipulate") or []:
        print(f"  manipulate {manipulation_line(m)}")
    sc = result["score"]
    verdict = "PASS*" if sc["fallback_pass"] else ("PASS" if sc["passed"] else "FAIL")
    print(f"\n{verdict} {result['scenario']} on {result['profile']}: {sc['note']} · {sc['decisions']} decisions · "
          f"{result['wall_s']} s wall · honesty: {'; '.join(sc['honesty']) or '-'}")
    print(f"trace: {jl}\nsummary: {summ}")
    return 0 if result["ok"] else 1


def _isolate_runs(root: Path) -> None:
    """Memory, episodes and the procedural graph go to the throwaway dir too (they don't read WORLDLINE_RUNS
    at import time everywhere), so an offline run starts from an empty house memory, like eval's forget."""
    try:
        import agent.episodes
        import agent.memory
        import agent.procedures
        agent.episodes.ROOT = root / "episodes"
        agent.memory.ROOT = root / "memory"
        agent.procedures.STORE = root / "procedures.json"
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
