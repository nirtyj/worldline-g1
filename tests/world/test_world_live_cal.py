"""world/live_cal.py and world/vis_audit.py: the pure parts (the live runs themselves need the box; they were also run
once on tools/fake_p1 + tools/fake_deploy + the body service). No ZMQ here."""

from __future__ import annotations

import json

import numpy as np
import pytest

from world import live_cal as lc
from world import vis_audit as va


def test_rtf_windows_is_sim_time_over_receive_time():
    t = np.arange(0.0, 5.0, 0.02)
    sim = t.copy()
    sim[t > 2.5] -= 0.15                                  # one 150 ms stall (the P1 render hitch)
    a = np.zeros((len(t), 10))
    a[:, 0], a[:, 1] = t, sim
    r = lc.rtf_windows(a)
    assert r["n"] == 4 and r["min"] == pytest.approx(0.85, abs=0.03) and r["total"] == pytest.approx(0.97, abs=0.01)


def test_arm_targets_sit_on_the_sphere_and_inside_it():
    ws = {"arm_reach_m": 0.405, "shoulder_z_m": 1.083, "shoulder_lat_m": 0.136, "shoulder_fwd_m": 0.0,
          "reach_fwd_m": [0.15, 0.43]}
    tg = lc.arm_targets(ws, [1.10], [0.15], [0.0, 0.04])
    assert len(tg) == 2 and all(lat == -0.15 for _, lat, _, _ in tg)
    (f0, _, _, m0), (f1, _, _, m1) = tg
    assert (m0, m1) == (0.0, 0.04) and f0 > f1 and f0 <= 0.45


def _write(tmp, name, rep):
    d = tmp / name
    d.mkdir()
    (d / "live_cal.json").write_text(json.dumps(rep))
    return str(d)


def test_summarize_keeps_only_ops_that_passed_the_timing_gate(tmp_path):
    walk = {"cmd_mps": 0.45, "steady_speed_mps": 0.57, "steady_disp_speed_mps": 0.56, "start_latency_s": 0.2,
            "stop_after_cmd_s": 1.0, "glide_after_cmd_m": 0.3}
    run = {"walk": [dict(walk, valid=True), dict(walk, steady_speed_mps=0.1, valid=False)],
           "stop": [{"gt_to_rest_s": 0.9, "body_stop_time_s": 0.9, "glide_m": 0.2, "valid": True}],
           "turn": [{"cmd_deg": 90.0, "deg_per_s": 18.0, "final_err_deg": 1.0, "valid": True}],
           "go_to": [{"state": "succeeded", "path_len_m": 5.0, "duration_s": 17.0, "eff_speed_mps": 0.294,
                      "gt_pos_err_m": 0.1, "gt_yaw_err_deg": 2.0, "valid": True},
                     {"state": "succeeded", "path_len_m": 5.0, "duration_s": 50.0, "eff_speed_mps": 0.1,
                      "gt_pos_err_m": 0.1, "gt_yaw_err_deg": 2.0, "valid": False}],
           "approach": [{"leg": "to_counter", "state": "succeeded", "duration_s": 12.0, "gt_pos_err_m": 0.03,
                         "gt_clearance_m": 0.2, "valid": True}]}
    d1 = _write(tmp_path, "walk", {"gate": {"valid": False}, "runs": [run]})
    arm = {"free_air": [{"fwd": 0.3, "lat": -0.15, "h": 1.0, "margin": 0.0, "state": "succeeded",
                         "palm_err_w_m": {"p90": 0.02}, "pelvis_z": 0.784, "valid": True}],
           "sequence": [{"t_pick_s": 17.0, "t_place_s": 9.6,
                         "phases": [{"phase": "pregrasp", "duration_s": 7.0, "valid": True}]}]}
    d2 = _write(tmp_path, "arm", {"gate": {"valid": True}, "arm": arm})
    s = lc.summarize([d1, d2])
    assert s["walk"]["0.45"]["steady_speed_mps"]["n"] == 1 and s["walk"]["0.45"]["steady_speed_mps"]["mean"] == 0.57
    assert s["go_to"]["succeeded"] == 1 and s["go_to"]["effective_speed_mps"] == pytest.approx(5.0 / 17.0, abs=1e-3)
    assert s["arm_sequence"]["t_pick_s"]["mean"] == 17.0 and s["arm"]["pelvis_z_m"]["mean"] == 0.784
    s_all = lc.summarize([d1, d2], valid_only=False)
    assert s_all["go_to"]["succeeded"] == 2 and s_all["walk"]["0.45"]["steady_speed_mps"]["n"] == 2


def test_vis_audit_summary_tallies_what_the_images_support(tmp_path):
    audit = {"views": 2, "agree_views": 1, "per_view": [
        {"i": 0, "keypoint": "a", "agree": True, "p1_only": [], "geo_only": []},
        {"i": 1, "keypoint": "b", "agree": False, "p1_only": [{"id": "remote_control_2", "px": 2788,
                                                               "geo_reason": "occluded"}],
         "geo_only": [{"id": "vase_3", "seg_px": 0, "geo_px": 15308.0}, {"id": "cd_1", "seg_px": 0, "geo_px": 10.0}]}]}
    (tmp_path / "audit.json").write_text(json.dumps(audit))
    (tmp_path / "verdicts.json").write_text(json.dumps({"views": {"01": {"items": {"remote_control_2": "seg",
                                                                                   "vase_3": "seg"}}}}))
    s = va.summarize(tmp_path)
    assert s["disagreements"] == 3 and s["image_supports"] == {"seg": 2, "geo": 0, "unclear": 1}
    assert s["per_view"][1]["verdicts"]["cd_1"] == "unclear"
