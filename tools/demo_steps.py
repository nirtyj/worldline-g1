"""The demo's steps (config/demo_steps.yaml), their PASS rules and the line timing: one copy, shared by
scripts/demo.sh (which says the lines through tools/say.py) and the Worldline UI's Demo panel (ui/demo.py, which
says them through the chat's own say path). docs/demo.md.

    python -m tools.demo_steps list                     # the steps, as scripts/demo.sh --list prints them
    python -m tools.demo_steps bash                     # step_def(), STEP_IDS, NSTEPS, SAY_OPTS, READY_S, BETWEEN_S
                                                        # for scripts/demo.sh to eval
    python -m tools.demo_steps results RUN_DIR TOTAL_S  # the PASS/FAIL table from RUN_DIR/steps.tsv + stepN.json
    python -m tools.demo_steps json                     # the steps as JSON (what the Demo panel gets)

A step's evidence `d` is what tools/say.py --json writes, and what ui/demo.py builds the same way:
{status: ok|max_s, sent: [{text, trace_at, answering?}], trace: [trace rows], end: {robot, hands, paused, system1}}.
`trace_at` indexes `trace`: the rows seen when that line was sent.

The line timing (say_lines) is tools/say.py's: a plain line waits until the runtime has been idle for idle_s after
the previous line (twice that while nothing has happened yet), at most max_s; "@N text" is sent N s after the previous
line, and the line before it does not wait for idle; an --if-asked pair is answered once.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

ROOT = Path(__file__).resolve().parent.parent
STEPS_FILE = ROOT / "config" / "demo_steps.yaml"
TIMED = re.compile(r"@(\d+(?:\.\d+)?)\s+(.*)")


def split_line(raw: str) -> tuple[float | None, str]:
    """ "@6 stop" -> (6.0, "stop"); "stop" -> (None, "stop")."""
    m = TIMED.match(raw)
    return (float(m.group(1)), m.group(2)) if m else (None, raw)


# ---------------------------------------------------------------------------------------------- the steps
@dataclass
class Step:
    id: int
    title: str
    shows: str
    lines: list[str]
    max_s: float
    rule: dict[str, Any]
    pass_text: str = ""
    if_asked: list[tuple[str, str]] = field(default_factory=list)
    needs: str | None = None

    def line_text(self, n: int) -> str:
        """Line n (1-based) as it is said: without its "@N " timing."""
        return split_line(self.lines[n - 1])[1]

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "shows": self.shows, "lines": list(self.lines),
                "max_s": self.max_s, "if_asked": [list(p) for p in self.if_asked], "needs": self.needs,
                "rule": dict(self.rule), "pass_text": self.pass_text}


@dataclass
class Demo:
    steps: list[Step]
    idle_s: float = 5.0
    idle_ignores_wait: bool = True
    ready_s: float = 240.0
    between_s: float = 2.0
    source: str = ""

    def step(self, sid: int) -> Step:
        for s in self.steps:
            if s.id == sid:
                return s
        raise KeyError(f"no step {sid}")

    @property
    def ids(self) -> list[int]:
        return [s.id for s in self.steps]


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"{what} must be a positive number, not {v!r}")
    return v


def load(path: Path | str | None = None) -> Demo:
    """config/demo_steps.yaml, checked: unique ids, known rules with their parameters, lines, regexes."""
    import yaml

    path = Path(path) if path else STEPS_FILE
    doc = yaml.safe_load(path.read_text()) or {}
    dflt = doc.get("defaults") or {}
    steps: list[Step] = []
    for i, raw in enumerate(doc.get("steps") or []):
        where = f"{path.name} step #{i + 1}"
        sid = raw.get("id")
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise ValueError(f"{where}: id must be a positive integer, not {sid!r}")
        lines = raw.get("lines")
        if not lines or not isinstance(lines, list) or not all(isinstance(x, str) and x.strip() for x in lines):
            raise ValueError(f"{where}: lines must be a non-empty list of strings")
        rule = dict(raw.get("pass") or {})
        name = rule.get("rule")
        if name not in RULES:
            raise ValueError(f"{where}: unknown pass rule {name!r} (known: {', '.join(sorted(RULES))})")
        missing = [p for p in RULE_PARAMS[name] if p not in rule]
        if missing:
            raise ValueError(f"{where}: rule {name} needs {', '.join(missing)}")
        if "line" in rule and not (isinstance(rule["line"], int) and 1 <= rule["line"] <= len(lines)):
            raise ValueError(f"{where}: rule line {rule['line']!r} is not one of the step's {len(lines)} lines")
        if "pattern" in rule:
            re.compile(rule["pattern"])
        asked = []
        for pair in raw.get("if_asked") or []:
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                raise ValueError(f"{where}: if_asked entries are [RE, REPLY] pairs")
            re.compile(pair[0])
            asked.append((pair[0], pair[1]))
        steps.append(Step(id=sid, title=str(raw.get("title") or f"step {sid}"), shows=str(raw.get("shows") or ""),
                          lines=list(lines), max_s=_num(raw.get("max_s", 300), f"{where}: max_s"), rule=rule,
                          pass_text=str(raw.get("pass_text") or ""), if_asked=asked, needs=raw.get("needs")))
    if not steps:
        raise ValueError(f"{path}: no steps")
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: step ids repeat: {ids}")
    try:
        source = str(path.resolve().relative_to(ROOT))
    except ValueError:
        source = str(path)
    return Demo(steps=steps, idle_s=_num(dflt.get("idle_s", 5), "idle_s"),
                idle_ignores_wait=bool(dflt.get("idle_ignores_wait", True)),
                ready_s=_num(dflt.get("ready_s", 240), "ready_s"),
                between_s=float(dflt.get("between_s", 2)), source=source)


# ---------------------------------------------------------------------------------------------- PASS rules
def after(d: dict, k: int = 0) -> list[dict]:
    """Trace rows from the k-th line sent on."""
    sent = d.get("sent") or []
    return d["trace"][sent[k]["trace_at"]:] if len(sent) > k else []


def heard_t(d: dict, text: str) -> float | None:
    for r in d["trace"]:
        if r.get("type") == "heard" and r.get("text", "").strip().lower() == text.strip().lower():
            return r.get("t", 0.0)
    return None


def speech(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("type") == "result" and r.get("kind") == "speech"]


def said(rows: list[dict]) -> str:
    return " | ".join(str(r.get("text")) for r in speech(rows))


def res(rows: list[dict], tool: str) -> list[dict]:
    return [r for r in rows if r.get("type") == "result" and r.get("tool") == tool]


def ok(r: dict) -> bool:
    return str(r.get("status")).lower() == "succeeded"


def since(rows: list[dict], t: float | None) -> list[dict]:
    return [r for r in rows if t is not None and (r.get("t_start") or r.get("t") or 0) >= t]


def _at(r: dict, place: str) -> bool:
    return place in str((r.get("data") or {}).get("location"))


def note_saved(d: dict, step: Step) -> tuple[bool, str]:
    rows = after(d)
    notes = [r for r in rows if r.get("type") == "note_saved"]
    return bool(notes), f"note saved: {notes[0].get('text')!r}" if notes else "no note_saved row"


def answered_while_walking(d: dict, step: Step, location: str, line: int) -> tuple[bool, str]:
    rows = after(d)
    nav = [r for r in res(rows, "navigate") if ok(r) and _at(r, location)]
    tq = heard_t(d, step.line_text(line))
    t_arr = nav[0]["t"] if nav else None
    ans = [r for r in speech(rows) if tq is not None and (r.get("t_start") or 0) >= tq]
    while_walking = [r for r in ans if t_arr is None or (r.get("t_start") or 0) <= t_arr]
    txt = said(ans)[:160]
    if nav and while_walking:
        return True, f"{nav[0]['summary'][:70]}; answered while walking: {said(while_walking)[:120]!r}"
    return False, f"navigate ok={bool(nav)}; answer={txt!r} ({'while walking' if while_walking else 'not while walking'})"


def stop_resume(d: dict, step: Step, location: str) -> tuple[bool, str]:
    rows = after(d)
    stops = [r for r in rows if r.get("type") == "stop" and not r.get("already_paused")]
    resumes = [r for r in rows if r.get("type") == "resume" and stops and r["t"] >= stops[0]["t"]]
    nav = [r for r in res(rows, "navigate") if ok(r) and resumes and r["t"] >= resumes[0]["t"] and _at(r, location)]
    msg = f"stop={len(stops)} (canceled {stops[0].get('canceled') if stops else '-'}), resume={len(resumes)}, " \
          f"arrived after resume: {nav[0]['summary'][:60] if nav else 'no'}"
    return bool(stops and resumes and nav), msg


def said_rule(d: dict, step: Step, pattern: str) -> tuple[bool, str]:
    rows = after(d)
    sp = speech(rows)
    hit = [r for r in sp if re.search(pattern, str(r.get("text")), re.I)]
    return bool(hit), f"said: {said(hit or sp)[:200]!r}"


def nav_after_line(d: dict, step: Step, location: str, line: int) -> tuple[bool, str]:
    rows = after(d)
    tc = heard_t(d, step.line_text(line))
    lab = [r.get("kind") for r in rows if r.get("type") == "classified" and tc is not None and r.get("t", 0) >= tc]
    nav = [r for r in res(since(rows, tc), "navigate") if ok(r) and _at(r, location)]
    return bool(nav), f"labelled {lab[:1]}; {nav[0]['summary'][:70] if nav else location.replace('_', ' ') + ' not reached'}"


def reach_refusal(d: dict, step: Step, verdict: str) -> tuple[bool, str]:
    rows = after(d)
    cr = [r for r in res(rows, "check_reachability") if verdict in str(r.get("summary"))]
    sp = [r for r in speech(rows) if cr and (r.get("t_start") or 0) >= cr[0]["t"] - 0.5]
    return bool(cr and sp), (f"{cr[0]['summary'][:110]}; said {said(sp)[:100]!r}" if cr else
                             f"no {verdict}; said {said(rows)[:120]!r}")


def pick_holding(d: dict, step: Step) -> tuple[bool, str]:
    rows = after(d)
    picks = [r for r in res(rows, "manipulate") if r.get("action") == "pick"]
    good = [r for r in picks if ok(r) and (r.get("data") or {}).get("holding")]
    fb = [r for r in rows if r.get("type") == "manip.fallback"]
    hands = (d.get("end") or {}).get("hands") or {}
    held = {a: v.get("holding") for a, v in hands.items() if v.get("holding") not in (None, "nothing", "")}
    path = " -> ".join(r.get("tool") + (f"({(r.get('args') or {}).get('location')})" if r.get("tool") == "navigate" else "")
                       for r in rows if r.get("type") == "decision" and r.get("tool") not in ("speak", "wait_and_observe", "recall"))
    ex = good[0].get("summary", "")[:90] if good else (picks[-1].get("summary", "")[:120] if picks else "no pick")
    return bool(good), f"{ex}; fallbacks {[f.get('from_executor') + ':' + str(f.get('reason')) for f in fb]}; end hands {held}; path {path}"


RULES: dict[str, Callable[..., tuple[bool, str]]] = {
    "note_saved": note_saved, "answered_while_walking": answered_while_walking, "stop_resume": stop_resume,
    "said": said_rule, "nav_after_line": nav_after_line, "reach_refusal": reach_refusal, "pick_holding": pick_holding}
RULE_PARAMS: dict[str, tuple[str, ...]] = {
    "note_saved": (), "answered_while_walking": ("location", "line"), "stop_resume": ("location",),
    "said": ("pattern",), "nav_after_line": ("location", "line"), "reach_refusal": ("verdict",), "pick_holding": ()}


def judge(step: Step, d: dict) -> tuple[bool, str]:
    """The step's PASS rule over its evidence."""
    params = {k: v for k, v in step.rule.items() if k != "rule"}
    return RULES[step.rule["rule"]](d, step, **params)


def verdict(step: Step, d: dict | None, rc: int = 0) -> tuple[bool, str]:
    """PASS/FAIL and the evidence line, as scripts/demo.sh's table has it: a failed say (rc) or no evidence fails; a
    judge that raises fails; a line that hit max_s only adds a note."""
    if d is None or rc != 0:
        return False, f"tools.say rc {rc}"
    try:
        good, why = judge(step, d)
    except Exception as e:  # noqa: BLE001
        good, why = False, f"judge error {e!r}"
    if d.get("status") == "max_s":
        why += " (a line hit --max-s)"
    return bool(good), why


def table(rows: list[tuple[int, str, bool, int, str]], total_s: int) -> str:
    """scripts/demo.sh's results table: rows of (step id, title, passed, seconds, evidence)."""
    out = ["| step | behaviour | result | s | evidence |\n|---|---|---|---|---|"]
    for n, title, good, s, why in rows:
        out.append(f"| {n} | {title} | {'PASS' if good else 'FAIL'} | {s} | {why.replace('|', '/')} |")
    out.append(f"\n{sum(bool(r[2]) for r in rows)}/{len(rows)} steps passed; total {total_s // 60} min {total_s % 60} s")
    return "\n".join(out)


def results(run: Path, total_s: int, demo: Demo) -> tuple[str, int]:
    """The markdown table of a scripts/demo.sh run dir (steps.tsv: "N rc seconds" per step run), and the exit code."""
    steps: dict[int, tuple[int, int]] = {}
    for line in (run / "steps.tsv").read_text().split("\n"):
        if line.strip():
            n, rc, s = line.split()
            steps[int(n)] = (int(rc), int(s))
    rows = []
    for n, (rc, s) in steps.items():
        p = run / f"step{n}.json"
        d = json.loads(p.read_text()) if p.exists() else None
        good, why = verdict(demo.step(n), d, rc)
        rows.append((n, demo.step(n).title, good, s, why))
    return table(rows, total_s), (0 if all(r[2] for r in rows) else 1)


# ---------------------------------------------------------------------------------------------- saying the lines
class Channel(Protocol):
    """Where the lines go and what comes back: the page websocket (tools/say.py) or the page server itself."""

    async def say(self, text: str) -> None: ...

    def count(self) -> int: ...                              # trace rows so far (the step's own numbering)

    def rows(self, start: int) -> list[dict]: ...            # trace rows from `start` on

    def busy(self, ignore_wait: bool) -> bool: ...           # something runs (speech too) or the planner thinks


async def say_lines(ch: Channel, lines: list[str], *, idle_s: float, max_s: float, ignore_wait: bool,
                    if_asked: list[tuple[str, str]] | tuple = (), poll_s: float = 0.3,
                    on: Callable[..., Any] | None = None,
                    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> tuple[list[dict], str]:
    """Say `lines` with tools/say.py's timing. Returns (sent, status): sent = [{text, wall, trace_at, answering?}],
    status "ok" or "max_s" (a line was still busy after max_s; the next line went anyway).
    `on(event, **info)` hears: wait (a timed line: delay), sent (text, answering?), idle (waiting for idle after
    text), max_s (text)."""
    note = on or (lambda *_a, **_k: None)
    sent: list[dict] = []
    status = "ok"
    seen = ch.count()
    asked = [False] * len(if_asked)
    for i, raw in enumerate(lines):
        delay, line = split_line(raw)
        if delay is not None:
            note("wait", text=line, delay=delay)
            await sleep(delay)
            seen = ch.count()                        # say.py prints these rows while it waits; they are not scanned
        sent.append({"text": line, "wall": round(time.time(), 3), "trace_at": ch.count()})
        note("sent", text=line)
        await ch.say(line)
        if i + 1 < len(lines) and lines[i + 1].startswith("@"):
            continue                                 # timed follow-up: don't wait for idle
        note("idle", text=line)
        t0 = time.monotonic()
        idle_since, acted = None, False
        while time.monotonic() - t0 < max_s:
            await sleep(poll_s)
            new = ch.rows(seen)
            seen += len(new)
            for r in new:
                acted = acted or r.get("type") in ("decision", "result")
                if r.get("type") == "result" and r.get("kind") == "speech":
                    for k, (pat, reply) in enumerate(if_asked):
                        if not asked[k] and re.search(pat, str(r.get("text") or ""), re.I):
                            asked[k] = True
                            sent.append({"text": reply, "wall": round(time.time(), 3), "trace_at": ch.count(),
                                         "answering": pat})
                            note("sent", text=reply, answering=pat)
                            await ch.say(reply)
                            idle_since = None
            if ch.busy(ignore_wait):
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= (idle_s if acted else idle_s * 2):
                    break
        else:
            status = "max_s"
            note("max_s", text=line)
    return sent, status


# ---------------------------------------------------------------------------------------------- scripts/demo.sh
def _num_text(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def bash(demo: Demo) -> str:
    """Shell code for scripts/demo.sh: step_def N (sets NAME, MAXS, LINES, EXTRA; returns 1 for no such step),
    STEP_IDS, NSTEPS, and the tools.say options every step uses."""
    q = shlex.quote
    out = ["step_def() {", "  EXTRA=()", '  case "$1" in']
    for s in demo.steps:
        body = f"NAME={q(s.title)}; MAXS={_num_text(s.max_s)}; LINES=({' '.join(q(x) for x in s.lines)})"
        if s.if_asked:
            extra = " ".join(f"--if-asked {q(p)} {q(r)}" for p, r in s.if_asked)
            body += f"\n       EXTRA=({extra})"
        out.append(f"    {s.id}) {body};;")
    out += ["    *) return 1;;", "  esac", "}"]
    opts = (["--idle-ignores-wait"] if demo.idle_ignores_wait else []) + ["--idle-s", _num_text(demo.idle_s)]
    out += [f"STEP_IDS=({' '.join(str(i) for i in demo.ids)})", f"NSTEPS={len(demo.steps)}",
            f"SAY_OPTS=({' '.join(opts)})", f"READY_S={_num_text(demo.ready_s)}",
            f"BETWEEN_S={_num_text(demo.between_s)}"]
    return "\n".join(out)


def listing(demo: Demo) -> str:
    """What scripts/demo.sh --list prints: id, title padded to 30, then each line in quotes."""
    return "\n".join(f"{s.id:d}  {s.title:<30}" + "".join(f' "{x}"' for x in s.lines) for s in demo.steps)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    try:
        demo = load()
    except Exception as e:  # noqa: BLE001
        print(f"[demo_steps] cannot read {STEPS_FILE}: {e}", file=sys.stderr)
        return 2
    if cmd == "bash":
        print(bash(demo))
    elif cmd == "list":
        print(listing(demo))
    elif cmd == "json":
        print(json.dumps({"source": demo.source, "idle_s": demo.idle_s, "ready_s": demo.ready_s,
                          "steps": [s.public() for s in demo.steps]}, indent=1))
    elif cmd == "results" and len(rest) == 2:
        text, rc = results(Path(rest[0]), int(rest[1]), demo)
        print(text)
        return rc
    else:
        print(f"usage: python -m tools.demo_steps list|bash|json|results RUN_DIR TOTAL_S (not {' '.join(argv)})",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
