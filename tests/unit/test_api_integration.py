"""The M2a integration changes to api/ (the other agents' change requests): one definition of a target executor,
per-profile eval time scales, `detail` on navigate / reachability results, and the service Protocols' shapes."""

from __future__ import annotations

import dataclasses

from api import results as R
from api import services as S
from api.types import PROFILES


def test_target_and_fallback_executors_are_disjoint_and_complete():
    every = set(R.NAV_EXECUTORS) | set(R.MANIP_EXECUTORS)
    assert set(R.TARGET_EXECUTORS) <= every
    for ex in every:
        assert R.is_target(ex) != R.is_fallback(ex), ex        # each known executor is exactly one of them
    assert R.is_target("sonic_walk") and R.is_target("groot_arms")
    for ex in ("lite", "kinematic_nav", "kinematic_attach", "sonic_arm_script", None, "", "something_new"):
        assert not R.is_target(ex), ex                          # unknown or missing is never a target


def test_eval_time_scales_live_in_api_profiles():
    assert {k: PROFILES[k].time_scale for k in ("lite", "bringup", "sonic", "full")} == \
        {"lite": 1.0, "bringup": 1.3, "sonic": 2.0, "full": 2.5}                          # PLAN 9.1
    from eval import scenes
    b = scenes.load()
    for name in ("lite", "bringup", "sonic", "full"):
        assert b.profile(name)["time_scale"] == PROFILES[name].time_scale


def test_detail_on_navigate_and_reachability_results():
    for cls in (R.NavigateResult, R.ReachabilityResult):
        f = {x.name: x for x in dataclasses.fields(cls)}
        assert "detail" in f and f["detail"].default is None


def test_sim_control_and_world_model_protocols_carry_what_services_use():
    assert "capabilities" in dir(S.SimControl) and "free_spot" in dir(S.WorldModel)
    import inspect
    assert "within" in inspect.signature(S.WorldModel.free_spot).parameters
    sig = inspect.signature(S.SimControl.teleport_robot).parameters
    assert list(sig)[1:] == ["pose", "y", "yaw"]                # Pose2D, anything with x/y/yaw, or three numbers
    assert issubclass(S.NotSupported, RuntimeError)
