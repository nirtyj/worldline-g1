"""tools/demo_steps.py: the demo's one step list (config/demo_steps.yaml) and its PASS rules, shared by scripts/demo.sh
and the Worldline UI's Demo panel (docs/demo.md).

  test_config_loads            the shipped file: nine steps, the rules and their parameters check out
  test_bash_*                  the shell code scripts/demo.sh evals: step_def per step (NAME, MAXS, LINES, EXTRA),
                               unknown steps return 1, the say options; --list's format
  test_load_rejects            unknown rules, missing parameters, a rule line past the step's lines, repeated ids
  test_rules_*                 each PASS rule on a small trace: pass, and the fail it reports
  test_verdict                 a failed say or missing evidence fails; a judge error fails; max_s only adds a note
  test_results_table           scripts/demo.sh's table and exit code from a run dir
  test_demo_sh_*               scripts/demo.sh reads the list: --list prints it, a step not in it exits 2, and the script
                               keeps no step list or judge of its own
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import demo_steps as ds

BASH = shutil.which("bash")
ROOT = Path(__file__).resolve().parents[2]
DEMO_SH = ROOT / "scripts" / "demo.sh"


def _yaml(tmp: Path, text: str) -> Path:
    p = tmp / "steps.yaml"
    p.write_text(text)
    return p


def test_config_loads():
    demo = ds.load()
    assert demo.ids == list(range(1, 10))
    assert demo.idle_s == 5 and demo.idle_ignores_wait and demo.ready_s == 240 and demo.between_s == 2
    assert demo.groot == "off", "the Demo panel keeps scripts/demo.sh's default: --groot off"
    assert demo.source == "config/demo_steps.yaml"
    s2 = demo.step(2)
    assert s2.lines == ["go to the bedroom dresser", "@6 where are you going?"] and s2.max_s == 150
    assert s2.line_text(2) == "where are you going?"
    s8 = demo.step(8)
    assert s8.if_asked == [("which (one|bottle|of)", "the white one")] and s8.needs == "empty hands"
    for s in demo.steps:
        assert s.title and s.shows and s.pass_text, s.id
        assert s.rule["rule"] in ds.RULES
    pub = demo.step(1).public()
    assert set(pub) >= {"id", "title", "shows", "lines", "max_s", "pass_text", "rule"}
    with pytest.raises(KeyError):
        demo.step(42)


def _bash_eval(code: str, script: str) -> str:
    return subprocess.run([BASH, "-c", code + "\n" + script], capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_bash_step_def_matches_the_file():
    demo = ds.load()
    code = ds.bash(demo)
    out = _bash_eval(code, 'for n in "${STEP_IDS[@]}"; do step_def $n; printf "%s|%s|" "$NAME" "$MAXS"; '
                           'printf "%s;" "${LINES[@]}"; printf "|"; printf "%s;" "${EXTRA[@]}"; echo; done; '
                           'step_def 99; echo "rc=$? n=$NSTEPS opts=${SAY_OPTS[*]} ready=$READY_S between=$BETWEEN_S"')
    rows = out.strip().split("\n")
    assert rows[-1] == "rc=1 n=9 opts=--idle-ignores-wait --idle-s 5 ready=240 between=2"
    got = {i + 1: r for i, r in enumerate(rows[:-1])}
    assert got[1] == "memory note|60|my keys are usually on the kitchen counter;|;"
    assert got[3] == "stop / resume while walking|180|go to the dining table;@6 stop;@5 okay, carry on;|;"
    assert got[8] == "pick (reach stance + arm)|330|pick up the white bottle;|--if-asked;which (one|bottle|of);the white one;"
    assert got[9] == "question + chitchat|60|what are you holding?;thanks;|;"


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_bash_quotes_awkward_text(tmp_path):
    demo = ds.load(_yaml(tmp_path, """
steps:
  - id: 4
    title: it's "quoted" $HOME `x`
    shows: s
    lines: ["don't $(touch /nope) go", "@2 a;b"]
    max_s: 7.5
    pass: {rule: said, pattern: x}
"""))
    out = _bash_eval(ds.bash(demo), 'step_def 4; printf "%s\\n" "$NAME" "$MAXS" "${LINES[@]}"')
    assert out.split("\n")[:4] == ["it's \"quoted\" $HOME `x`", "7.5", "don't $(touch /nope) go", "@2 a;b"]


def test_listing_format():
    first = ds.listing(ds.load()).split("\n")[0]
    assert first == '1  memory note                    "my keys are usually on the kitchen counter"'


@pytest.mark.parametrize("body, err", [
    ("pass: {rule: nope}", "unknown pass rule"),
    ("pass: {rule: said}", "needs pattern"),
    ("pass: {rule: nav_after_line, location: x, line: 3}", "not one of the step's 1 lines"),
    ("pass: {rule: said, pattern: '('}", "missing"),
])
def test_load_rejects(tmp_path, body, err):
    p = _yaml(tmp_path, f"steps:\n  - id: 1\n    title: t\n    lines: [a]\n    {body}\n")
    with pytest.raises(Exception, match=err):
        ds.load(p)


def test_load_groot_setting(tmp_path):
    step = "  - id: 1\n    lines: [a]\n    pass: {rule: note_saved}\n"
    assert ds.load(_yaml(tmp_path, "defaults: {groot: on}\nsteps:\n" + step)).groot == "on"
    with pytest.raises(ValueError, match="groot must be off or on"):
        ds.load(_yaml(tmp_path, "defaults: {groot: maybe}\nsteps:\n" + step))


def test_load_rejects_repeated_ids_and_no_lines(tmp_path):
    step = "  - id: 1\n    lines: [a]\n    pass: {rule: note_saved}\n"
    with pytest.raises(ValueError, match="repeat"):
        ds.load(_yaml(tmp_path, "steps:\n" + step + step))
    with pytest.raises(ValueError, match="lines"):
        ds.load(_yaml(tmp_path, "steps:\n  - id: 1\n    lines: []\n    pass: {rule: note_saved}\n"))


# ---------------------------------------------------------------------------------------------- the rules
def _d(sent, trace, **kw):
    return {"status": "ok", "sent": [{"text": t, "trace_at": i} for t, i in sent], "trace": trace, **kw}


def sp(t, text):
    return {"type": "result", "kind": "speech", "t": t, "t_start": t, "text": text}


def nav(t, loc, status="succeeded"):
    return {"type": "result", "tool": "navigate", "t": t, "status": status, "data": {"location": loc},
            "summary": f"arrived at {loc}"}


def heard(t, text):
    return {"type": "heard", "t": t, "text": text}


def test_rules_note_and_said():
    demo = ds.load()
    d = _d([("my keys…", 0)], [{"type": "note_saved", "t": 1, "text": "keys on the counter"}])
    assert ds.judge(demo.step(1), d) == (True, "note saved: 'keys on the counter'")
    assert ds.judge(demo.step(1), _d([("x", 0)], [])) == (False, "no note_saved row")
    d = _d([("where are my keys?", 1)], [sp(0, "the counter, earlier"), sp(2, "On the kitchen COUNTER.")])
    assert ds.judge(demo.step(4), d) == (True, "said: 'On the kitchen COUNTER.'")
    ok, why = ds.judge(demo.step(4), _d([("x", 0)], [sp(1, "no idea")]))
    assert not ok and why == "said: 'no idea'"


def test_rules_answered_while_walking():
    step = ds.load().step(2)
    trace = [heard(1, "go to the bedroom dresser"), heard(7, "Where are you going?"), sp(8, "To the dresser."),
             nav(20, "bedroom_dresser_1a")]
    ok, why = ds.judge(step, _d([("go…", 0), ("where…", 1)], trace))
    assert ok and why.startswith("arrived at bedroom_dresser_1a; answered while walking")
    late = [heard(7, "where are you going?"), nav(20, "bedroom_dresser_1a"), sp(21, "I was going to the dresser.")]
    ok, why = ds.judge(step, _d([("go…", 0)], late))
    assert not ok and "not while walking" in why


def test_rules_stop_resume():
    step = ds.load().step(3)
    trace = [{"type": "stop", "t": 6, "canceled": ["navigate"]}, {"type": "resume", "t": 11},
             nav(20, "kitchen_dining_table_1a")]
    ok, why = ds.judge(step, _d([("go", 0)], trace))
    assert ok and why.startswith("stop=1 (canceled ['navigate']), resume=1, arrived after resume: arrived at")
    early = [nav(3, "kitchen_dining_table_1a"), {"type": "stop", "t": 6, "already_paused": True}, {"type": "resume", "t": 11}]
    assert not ds.judge(step, _d([("go", 0)], early))[0]


def test_rules_correction_reach_pick():
    demo = ds.load()
    trace = [heard(5, "no, go to the kitchen counter instead"), {"type": "classified", "t": 5.2, "kind": "correction"},
             nav(15, "kitchen_counter_1a")]
    assert ds.judge(demo.step(6), _d([("go", 0)], trace)) == (True, "labelled ['correction']; arrived at kitchen_counter_1a")
    assert ds.judge(demo.step(6), _d([("go", 0)], trace[:2]))[1] == "labelled ['correction']; kitchen counter not reached"
    cr = {"type": "result", "tool": "check_reachability", "t": 4, "summary": "bowl_1 beyond_reach from every stance"}
    ok, why = ds.judge(demo.step(7), _d([("can…", 0)], [cr, sp(3.8, "I can't reach the bowl.")]))
    assert ok and why.startswith("bowl_1 beyond_reach")
    ok, why = ds.judge(demo.step(7), _d([("can…", 0)], [sp(3, "Sure.")]))
    assert not ok and why == "no beyond_reach; said 'Sure.'"
    pick = {"type": "result", "tool": "manipulate", "action": "pick", "t": 50, "status": "succeeded",
            "data": {"holding": True}, "summary": "picked bottle_1 [fallback]"}
    rows = [{"type": "decision", "tool": "navigate", "args": {"location": "table_1a"}}, {"type": "decision", "tool": "recall"},
            {"type": "manip.fallback", "from_executor": "groot_arms",
                                                       "reason": "timeout"}, pick]
    ok, why = ds.judge(demo.step(8), _d([("pick", 0)], rows, end={"hands": {"left": {"holding": "bottle_1"}}}))
    assert ok and why == ("picked bottle_1 [fallback]; fallbacks ['groot_arms:timeout']; end hands {'left': 'bottle_1'}; "
                          "path navigate(table_1a)")


def test_verdict():
    step = ds.load().step(1)
    good = _d([("x", 0)], [{"type": "note_saved", "text": "n"}])
    assert ds.verdict(step, None, 0) == (False, "tools.say rc 0")
    assert ds.verdict(step, good, 3) == (False, "tools.say rc 3")
    assert ds.verdict(step, {**good, "status": "max_s"}) == (True, "note saved: 'n' (a line hit --max-s)")
    ok, why = ds.verdict(ds.load().step(2), {"sent": [{"trace_at": 0}], "trace": [nav(1, "dresser")] + [{"type": "heard"}]})
    assert ok is False and why.startswith("navigate ok=True")
    ok, why = ds.verdict(step, {"sent": [{"text": "x"}], "trace": []})          # no trace_at: the judge raises
    assert not ok and why.startswith("judge error")


def test_results_table(tmp_path):
    (tmp_path / "steps.tsv").write_text("1 0 27\n4 0 30\n5 2 9\n")
    (tmp_path / "step1.json").write_text(json.dumps(_d([("x", 0)], [{"type": "note_saved", "text": "n|m"}])))
    (tmp_path / "step4.json").write_text(json.dumps(_d([("x", 0)], [sp(1, "no idea")])))
    text, rc = ds.results(tmp_path, 125, ds.load())
    assert rc == 1
    assert text.split("\n") == [
        "| step | behaviour | result | s | evidence |", "|---|---|---|---|---|",
        "| 1 | memory note | PASS | 27 | note saved: 'n/m' |",
        "| 4 | recall a note | FAIL | 30 | said: 'no idea' |",
        "| 5 | spatial memory | FAIL | 9 | tools.say rc 2 |",
        "", "1/3 steps passed; total 2 min 5 s"]
    (tmp_path / "steps.tsv").write_text("1 0 27\n")
    assert ds.results(tmp_path, 27, ds.load())[1] == 0


def test_cli(tmp_path, capsys):
    assert ds.main(["list"]) == 0 and capsys.readouterr().out.startswith("1  memory note")
    assert ds.main(["json"]) == 0 and len(json.loads(capsys.readouterr().out)["steps"]) == 9
    assert ds.main(["nope"]) == 2


# ---------------------------------------------------------------------------------------------- scripts/demo.sh
def _demo_sh(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "WL": str(ROOT), "PY_RT": sys.executable}
    return subprocess.run([BASH, str(DEMO_SH), *args], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_demo_sh_lists_the_shared_steps():
    subprocess.run([BASH, "-n", str(DEMO_SH)], check=True)
    out = _demo_sh("--list")
    assert out.returncode == 0 and out.stdout.rstrip("\n") == ds.listing(ds.load())
    assert _demo_sh("--help").stdout.startswith("# The Worldline-on-G1 demo")


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_demo_sh_rejects_a_step_not_in_the_list():
    out = _demo_sh("--only", "3,42", "--no-lock")          # checked before anything runs (no run dir, no lock)
    assert out.returncode == 2 and "no step 42" in out.stderr


def test_demo_sh_keeps_no_step_list_or_judge_of_its_own():
    text = DEMO_SH.read_text()
    assert "-m tools.demo_steps bash" in text and "-m tools.demo_steps results" in text
    for gone in ("NSTEPS=9", "def j8", "JUDGE = ", "NAMES = ", "<<'EOF'", "my keys are usually"):
        assert gone not in text, gone
