"""The offline episode's referee on synthetic traces: the F1 order (goal check after the place, delivered after
navigate(user) and for that place), the full envelope on every result row, and the rows that must exist as results
(speak, recall, rejections). No server, no runtime."""

from __future__ import annotations

from eval import offline_episode as ep

USER_KP, USER_SURF = "kitchen_counter_1a", "kitchen_counter_1a"


def row(i: int, **kw):
    base = {"i": i, "t": float(i)}
    if kw.get("type") == "result":
        base.update(kind="tool", execution_id=f"x-{i:06d}", status="succeeded", observation_id=f"obs-g{i}",
                    generation=1, control_epoch=0, t_start=float(i) - 0.5, t_end=float(i), late=False, data={})
    base.update(kw)
    return base


def f1_trace(order: str = "normal") -> list[dict]:
    """A minimal F1 trace; `order` moves the goal check or the delivery out of place."""
    lab = {"executor": "lite", "summary": "picked alarm_clock_1 [lite.pick.v0, fallback]"}
    rows = [
        row(0, type="classified", kind="request", directive={"source": "system1"}),
        row(1, type="result", tool="navigate", execution_id="nav-000001", data={"executor": "lite"},
            executor="lite", summary="arrived (lite, fallback)"),
        row(2, type="result", tool="observe", action="scan", execution_id="obs-000002"),
        row(3, type="result", tool="check_reachability", execution_id="rch-000003",
            data={"object_id": "alarm_clock_1", "reachable": False, "reason": "needs_reposition"}),
        row(3, type="result", tool="check_reachability", execution_id="rch-000033",
            data={"object_id": "alarm_clock_1", "reachable": True}),
        row(4, type="result", tool="manipulate", action="pick", execution_id="man-000004",
            data={"skill": "lite.pick.v0", "executor": "lite"}, **lab),
        row(5, type="result", tool="observe", action="glance", execution_id="obs-000005"),
        row(6, type="result", tool="navigate", execution_id="nav-000006", data={"at": USER_KP, "executor": "lite"},
            executor="lite", summary="arrived (lite, fallback)"),
        row(7, type="delivered", object="alarm_clock_1", surface=USER_SURF, execution_id="man-000008"),
        row(8, type="result", tool="manipulate", action="place", execution_id="man-000008",
            data={"skill": "lite.place.v0", "executor": "lite"}, **lab),
        row(9, type="goal_check", object="alarm_clock_1", ok=True),
    ]
    if order == "goal_first":           # the goal check before the place's result
        rows[9], rows[10] = rows[10], rows[9]
    elif order == "delivered_early":    # delivered before navigate(user)
        rows.insert(3, rows.pop(8))
    elif order == "delivered_other":    # delivered for another execution
        rows[8] = dict(rows[8], execution_id="man-000004")
    return rows


def steps_ok(trace):
    return {s["step"]: s["ok"] for s in ep.check_sequence(trace, ep.f1_steps(USER_KP, USER_SURF))}


def test_the_f1_shape_in_order_passes_and_delivered_may_precede_its_own_result_row():
    got = steps_ok(f1_trace())
    assert all(got.values()), got


def test_goal_check_before_the_place_result_fails():
    got = steps_ok(f1_trace("goal_first"))
    assert got["place on the user's surface (labelled)"] and not got["goal check ok"], got


def test_delivered_before_navigate_to_the_user_fails():
    got = steps_ok(f1_trace("delivered_early"))
    assert not got["delivered"], got


def test_delivered_must_name_the_place_execution():
    got = steps_ok(f1_trace("delivered_other"))
    assert not got["delivered"], got


def test_a_step_whose_anchor_is_missing_fails():
    trace = [r for r in f1_trace() if r.get("tool") != "navigate" or r["i"] != 6]
    got = steps_ok(trace)
    assert not got["navigate to the user"] and not got["delivered"], got


def test_envelope_problems_on_one_row():
    good = row(5, type="result", tool="speak", kind="speech")
    assert ep.envelope_problems(good) == []
    assert "no generation" in ep.envelope_problems({k: v for k, v in good.items() if k != "generation"})
    assert "no kind" in ep.envelope_problems({k: v for k, v in good.items() if k != "kind"})
    assert "generation True" in ep.envelope_problems(dict(good, generation=True))
    assert "t_end before t_start" in ep.envelope_problems(dict(good, t_start=9.0, t_end=1.0))
    assert "late 0" in ep.envelope_problems(dict(good, late=0))
    assert "observation_id empty" in ep.envelope_problems(dict(good, observation_id=None))
    assert "kind 'banana'" in ep.envelope_problems(dict(good, kind="banana"))
    assert "status 'SUCCEEDED'" in ep.envelope_problems(dict(good, status="SUCCEEDED"))
    schema = dict(good, status="rejected", kind="rejection", execution_id=None, data={"stage": "schema"})
    assert ep.envelope_problems(schema) == []
    state = dict(schema, data={"stage": "state"})
    assert "execution_id empty" in ep.envelope_problems(state)


def test_contract_checks_want_speak_recall_and_rejections_as_result_rows():
    trace = [row(1, type="say_queued", execution_id="spk-000001", text="On it."),
             row(2, type="result", tool="speak", execution_id="spk-000001", kind="speech"),
             row(3, type="recall", execution_id="rec-000003"),
             row(4, type="rejected", execution_id="man-000004", stage="state"),
             row(5, type="result", tool="manipulate", execution_id="man-000004", status="rejected", kind="tool",
                 data={"stage": "state", "skill": "s"}, executor="lite")]
    checks = {c["check"].split(" (")[0]: c for c in ep.contract_checks(trace)}
    rows_check = checks["speak, recall and rejections are result rows"]
    assert not rows_check["ok"]
    assert rows_check["detail"]["missing"] == [("recall", "rec-000003")]
    assert rows_check["detail"]["no_kind"] == [("rejection", "man-000004", ["tool"])]
    trace[4]["kind"] = "rejection"
    trace.append(row(6, type="result", tool="recall", execution_id="rec-000003", kind="recall"))
    checks = {c["check"].split(" (")[0]: c for c in ep.contract_checks(trace)}
    assert checks["speak, recall and rejections are result rows"]["ok"]
    assert checks["every result row carries the full envelope"]["ok"], checks["every result row carries the full envelope"]


def test_late_results_must_be_marked_on_their_row():
    trace = [row(1, type="result", tool="manipulate", execution_id="man-000001", status="cancelled",
                 data={"skill": "s"}, executor="lite"),
             row(2, type="late_result", execution_id="man-000001")]
    checks = {c["check"]: c for c in ep.contract_checks(trace)}
    assert not checks["late results are marked late on their result row"]["ok"]
    trace[0]["late"] = True
    checks = {c["check"]: c for c in ep.contract_checks(trace)}
    assert checks["late results are marked late on their result row"]["ok"]
