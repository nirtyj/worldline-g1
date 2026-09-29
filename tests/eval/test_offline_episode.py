"""The M2a integration episode (eval/offline_episode.py): the page's server, the real runtime, the scripted planner,
the System 1 stub and the whole lite stack (G1Robot -> services -> LiteBody + LiteWorld over the recorded H40) run
"Bring me the alarm clock." over a real websocket, and the eval's referee scores it on world truth.

No simulator, no LLM, no network beyond 127.0.0.1."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("websockets")
pytest.importorskip("PIL")

from eval import offline_episode as ep  # noqa: E402


def test_the_offline_f1_episode_delivers_and_shows_the_f1_shape(tmp_path):
    try:
        res = asyncio.run(asyncio.wait_for(ep.run_episode(speed=40.0), 180))
    except FileNotFoundError as e:                      # no recorded house data on this machine
        pytest.skip(f"no recorded house: {e}")
    missing = [s["step"] for s in res["steps"] if not s["ok"]]
    failed = [(c["check"], c["detail"]) for c in res["checks"] if not c["ok"]]
    assert res["score"]["passed"], (res["score"]["note"], ep.timeline(res["trace"])[-25:])
    assert not missing, (missing, ep.timeline(res["trace"])[-40:])
    assert not failed, failed
    assert res["ok"]
    # honesty: a lite pass is a labelled fallback pass, never a target pass (PLAN 2.3, 12.2)
    assert res["score"]["fallback_pass"] and not res["score"]["target_pass"]
    assert res["score"]["executors_used"]["manip"] == {"lite": 2}
    # which executor manipulated, attempt by attempt (E1 reports it; on full: the GR00T attempt, then the fallback)
    man = res["manipulate"]
    assert [(m["action"], m["status"], m["executor"]) for m in man] == [("pick", "succeeded", "lite"),
                                                                        ("place", "succeeded", "lite")], man
    assert man[0]["attempts"] and man[0]["attempts"][0]["executor"] == "lite", man[0]
    assert ep.manipulation_line(man[0]).startswith("pick succeeded by lite"), ep.manipulation_line(man[0])
    # the recorded trace is complete and ordered
    jl, summ = ep.write(res, tmp_path / "episode")
    rows = jl.read_text().splitlines()
    assert len(rows) == len(res["trace"]) + len(res["events"]) and summ.exists()
    assert '"type": "goal_check"' in jl.read_text() and '"type": "delivered"' in jl.read_text()


def test_the_scripted_planner_takes_its_want_from_the_request():
    from types import SimpleNamespace

    from brains.scripted import ScriptedPlanner, type_of

    assert type_of("alarm clock", ("alarm_clock", "mug")) == "alarm_clock"
    assert type_of("mugs", ("mug",)) == "mug" and type_of("unicorn", ("mug",)) is None
    p = ScriptedPlanner()
    utt = SimpleNamespace(id="u1", text="Bring me the alarm clock.", directive=None)
    ctx = SimpleNamespace(utterances=[utt], kinds={"u1": "request"},
                          tools_ctx=SimpleNamespace(skill_types=("alarm_clock", "mug")))
    assert p._new_want(ctx) == "alarm_clock"
    assert p._new_want(ctx) is None                    # the same utterance is not taken twice
    utt2 = SimpleNamespace(id="u2", text="No, the mug instead.", directive={"target": {"object": "mug"}})
    ctx.utterances.append(utt2)
    ctx.kinds["u2"] = "correction"
    assert p._new_want(ctx) == "mug"                   # System 1's target wins over the text
