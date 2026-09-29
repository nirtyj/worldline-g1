"""world/workspace_cal.py: the IK envelope and its sphere, and the coverage of household pickables (R.7 / M3.5).
Offline; the live side is world/live_cal.py (run on the box)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from world import workspace_cal as wc


def test_fit_sphere_recovers_a_sphere_and_drops_a_split_row():
    rng = np.random.default_rng(0)
    c, r = np.array([0.004, -0.136, 0.299]), 0.4255
    d = rng.normal(size=(80, 3))
    d[:, 0] = np.abs(d[:, 0])
    pts = c + r * d / np.linalg.norm(d, axis=1)[:, None] + rng.normal(scale=0.003, size=(80, 3))
    pts = np.vstack([pts, c + np.array([0.14, 0.0, 0.0])])        # an IK split row: reach far short of the sphere
    f = wc.fit_sphere(pts)
    assert np.allclose(f["centre_b"], c, atol=0.01) and f["radius_m"] == pytest.approx(r, abs=0.01)
    assert f["residual_m"]["rms"] < 0.006 and len(f["dropped"]) == 1


def test_grasp_ok_uses_arm_scripts_ik_settings():
    spec = wc.EnvelopeSpec()
    z = 1.05 - spec.pelvis_z_m
    ok, eg, ep = wc.grasp_ok(spec, 0.30, -0.20, z)                  # the B.7 dresser grasp's neighbourhood
    assert ok and eg < 0.01 and ep is not None and ep <= 0.02
    far, eg2, _ = wc.grasp_ok(spec, 0.55, -0.15, z)
    assert not far and eg2 > 0.05


def test_one_envelope_row_bisects_the_forward_reach():
    spec = wc.EnvelopeSpec(heights_m=(1.05,), lats_m=(0.15,), fwd_step_m=0.02)
    row = wc._row((spec, 1.05, 0.15))
    assert row["lat"] == -0.15 and 0.38 <= row["fwd_max"] <= 0.46 and row["fwd_min"] <= 0.2


def test_coverage_judges_with_the_calibrated_arm():
    pytest.importorskip("scipy")
    rows = wc.coverage("procthor-train-40")
    by = {r.object_id: r for r in rows}
    assert by["alarm_clock_1"].verdict == "one_approach"                    # F1's object: one reach stance
    assert by["book_1"].verdict == "too_low"
    assert by["alarm_clock_1"].second["reachable"] is True
    s = wc.coverage_summary(rows, {"obj_z_min_m": 0.70, "obj_z_max_m": 1.25})
    assert s["pickables_on_surfaces"] == len(rows) and 0 < s["frac_in_band"] < 1
    assert sum(s["verdicts"].values()) == len(rows)


def test_a_two_step_reach_stance_would_reach_more_than_one_straight_approach():
    """The request behind stance_via two_step (navigation: A* next to the stance, then the approach)."""
    pytest.importorskip("scipy")
    from dataclasses import replace
    from services.reachability import G1Workspace, ReachabilityModel
    from world.lite_world import LiteWorld
    from world.workspace_cal import _At, _SeenAll
    from services.skills import build_registry
    from robot.profile import load_profile
    ws = G1Workspace.from_dict({k: v for k, v in load_profile("sonic").g1["workspace"].items() if k != "lite_world"})
    w = LiteWorld("procthor-train-15")
    reg = build_registry(["lite"], w)
    o = w.object("egg_1")                                           # 0.32 m from a standable spot, ~1 m from its stand
    k = w.static_map().keypoints[o.where]
    w.set_robot_pose(k.x, k.y, k.yaw)
    one = ReachabilityModel(w, ws, registry=reg, observation=_SeenAll(w), nav=_At())
    two = ReachabilityModel(w, replace(ws, stance_via="two_step", approach_max_m=2.0), registry=reg,
                            observation=_SeenAll(w), nav=_At())
    assert one.find_stance(o.pos[:2], None, one.grasp_z(o)) is None
    st = two.find_stance(o.pos[:2], None, two.grasp_z(o))
    assert st is not None and st["distance_m"] > 0.6 and two.stance_ok(st["x"], st["y"])
