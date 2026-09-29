"""The page on the whole lite stack as ui/server.py builds it: robot.factory.build("lite", house) ->
LiteWorld (recorded procthor-train-40) + LiteBody + G1Robot facade + the real agent/ runtime, with the
runtime agent's scripted PolicyBrain instead of an LLM. The page sees the walk, the reposition, the pick
(labelled lite) and the object in the robot's hand in world truth."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
try:
    import robot.factory  # noqa: F401
    from sim.clock import SimClock
    from tests.unit.fakes_rt import PolicyBrain
except Exception as e:  # noqa: BLE001
    pytest.skip(f"robot/ or the runtime fakes not importable: {e}", allow_module_level=True)

from ui.server import Deps, Session  # noqa: E402


def test_page_session_on_the_lite_stack():
    deps = Deps(create_planner=lambda info: PolicyBrain("alarm_clock"), clock=lambda speed: SimClock(20.0)).resolve()

    async def go():
        s = Session("procthor-train-40", "agent", "gemini-3.8-flash", "lite", deps)
        await s.start()
        assert s.error is None, s.error
        init = s.init_message()
        s.hear("Bring me the alarm clock.")
        trace, held = [], None
        for _ in range(400):
            await asyncio.sleep(0.05)
            f = s.frame()
            trace += f["trace"]
            held = f["truth"]["arms"]["right"]["holding"] or f["truth"]["arms"]["left"]["holding"]
            if held and any(r.get("type") == "result" and r.get("tool") == "manipulate" for r in trace):
                break
        await s.stop()
        return init, trace, held, f

    init, trace, held, f = asyncio.run(asyncio.wait_for(go(), 90))
    lay = init["layout"]
    assert lay["scene"] == "procthor-train-40" and lay["occupancy"] and len(lay["grid"]) > 400
    assert init["truth"]["source"] == "lite-gt" and init["body"]["mode"] == "HOLD"
    assert init["stack"]["profile"] == "lite" and init["config"]["profile_fallbacks"] == []
    results = [r for r in trace if r.get("type") == "result"]
    nav = [r for r in results if r["tool"] == "navigate" and r["status"] == "succeeded"]
    assert nav and all(r.get("executor") == "lite" for r in nav)
    picks = [r for r in results if r["tool"] == "manipulate" and r.get("action") == "pick"]
    assert picks and picks[-1]["status"] == "succeeded" and picks[-1]["executor"] == "lite"
    assert held == "alarm_clock_1" and f["truth"]["objects"]["alarm_clock_1"]["where"].startswith("hand:")
