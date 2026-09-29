"""brains/frame_gate.py G1 presets (PLAN 8.3): sway, bob and exposure drift stay quiet; a mug-sized
blob appearing while stationary is a scene change; a 30 deg waist turn is a new view."""

from __future__ import annotations

import numpy as np

from brains.frame_gate import DEFAULT, G1_EGO, G1_HEAD, FrameGate, FrameGateConfig

H, W = 480, 640


def scene(shift_px: int = 0, gain: float = 1.0, blob: bool = False) -> np.ndarray:
    """A smooth synthetic room: soft gradients and a few large low-contrast shapes."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx = xx - shift_px
    img = 110 + 40 * np.sin(xx / 97.0) + 30 * np.cos(yy / 73.0) + 20 * np.sin((xx + yy) / 150.0)
    if blob:                                               # a mug about 1 m away: ~48 px
        img[300:348, 300:348] = 245
    img = np.clip(img * gain, 0, 255)
    return np.repeat(img[..., None], 3, axis=2).astype(np.uint8)


def test_default_config_is_the_thor_constants():
    assert (DEFAULT.new_view_deg, DEFAULT.cell_delta, DEFAULT.scene_cells, DEFAULT.move_m) == (30, 0.08, 0.004, 1.0)
    assert FrameGate().config == DEFAULT and FrameGate.preset("g1_head").config == G1_HEAD
    assert (G1_HEAD.new_view_deg, G1_HEAD.cell_delta, G1_HEAD.scene_cells) == (25.0, 0.10, 0.008)
    assert (G1_EGO.new_view_deg, G1_EGO.scene_cells) == (16.0, 0.012)


def test_sway_bob_and_exposure_drift_stay_quiet():
    g = FrameGate(config=G1_HEAD)
    assert g.decide(scene(), (0, 0, 0, 15), 0.0, stationary=True).reason == "first"
    t = 2.0
    for shift, bob, gain, sway in ((3, 0.03, 1.08, 0.5), (-3, -0.03, 0.92, -0.5), (2, 0.02, 1.05, 0.3)):
        d = g.decide(scene(shift_px=shift, gain=gain), (0, bob, sway, 15), t, stationary=True)
        assert not d.send and d.reason in ("still", "similar view"), d
        t += 2.0


def test_a_mug_appearing_while_stationary_is_a_scene_change():
    g = FrameGate(config=G1_HEAD)
    g.decide(scene(), (0, 0, 0, 15), 0.0, stationary=True)
    d = g.decide(scene(blob=True), (0, 0, 0, 15), 2.0, stationary=True)
    assert d.send and d.reason == "scene changed", d


def test_the_same_change_while_walking_is_not_a_scene_change():
    g = FrameGate(config=G1_HEAD)
    g.decide(scene(), (0, 0, 0, 15), 0.0, stationary=False)
    d = g.decide(scene(blob=True), (0, 0.02, 0, 15), 2.0, stationary=False)
    assert not d.send and d.reason == "similar view"


def test_a_30_degree_waist_turn_is_a_new_view():
    g = FrameGate(config=G1_HEAD)
    g.decide(scene(), (0, 0, 0, 15), 0.0, stationary=True)
    d = g.decide(scene(shift_px=200), (0, 0, 30, 15), 2.0, stationary=True)
    assert d.send and d.reason == "new view"


def test_old_call_signature_still_works():
    g = FrameGate()
    assert g.decide(scene(), (0, 0, 0, 15), 0.0).reason == "first"
    assert g.decide(scene(), (0, 0, 0, 15), 0.5).reason == "too soon"
    cfg = FrameGateConfig(min_interval_s=0.0)
    assert FrameGate(config=cfg).min_interval == 0.0
