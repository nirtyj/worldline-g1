"""groot.open_loop end to end against the fake PolicyServer, with a synthetic episode (no parquet, no video).

The fake policy answers with the recorded actions of the episode it was built from, so the `main` variant must
score zero error, the hold baseline must not, and the SONIC mapping asserts inside run() must hold.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from groot import joint_order as jo
from groot import open_loop
from groot.dataset import Episode

from .test_groot_policy_client import FakeServer

N = 90


def _episode() -> Episode:
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, N)[:, None]
    state = (0.3 * np.sin(2 * np.pi * t + rng.uniform(0, 3, 43))).astype(np.float32)
    action = np.roll(state, -5, axis=0)
    return Episode(0, state, action, np.zeros((N, 3), np.float32), np.full((N, 1), 0.72, np.float32), 3,
                   "Pick up the apple.", rng.integers(0, 255, (N, 480, 640, 3), dtype=np.uint8))


class ReplayServer(FakeServer):
    """Answers get_action with the episode's recorded actions from the row whose state matches the observation."""

    def __init__(self, ep: Episode):
        self.ep = ep
        super().__init__()

    def get_action(self, observation, options=None):
        la = observation["state"]["left_arm"][0, 0]
        rows = jo.groot_keys_from_lerobot43(self.ep.state)["left_arm"]
        t = int(np.argmin(np.abs(rows - la).sum(1)))
        keys = jo.groot_keys_from_lerobot43(self.ep.action[t: t + 40])
        keys["base_height_command"] = self.ep.base_height[t: t + 40]
        keys["navigate_command"] = self.ep.navigate[t: t + 40]
        out = {}
        for k in jo.GROOT_ACTION_KEYS:
            a = np.asarray(keys[k], np.float32)
            a = np.concatenate([a, np.repeat(a[-1:], 40 - len(a), 0)]) if len(a) < 40 else a
            out[k] = a[None]
        return out, {}


def test_open_loop_run_with_replay_policy(tmp_path, monkeypatch):
    ep = _episode()
    monkeypatch.setattr(open_loop, "load_episode", lambda root, i, frames=True: ep)
    srv = ReplayServer(ep)
    try:
        args = argparse.Namespace(dataset="unused", episodes=[0], stride=20, endpoint=srv.endpoint, timeout=5.0,
                                  ckpt="", urdf="", prompt="move the apple to the plate",
                                  variants="main,baseline_hold,neg_swap_arms", plots=False, out=str(tmp_path))
        res = open_loop.run(args)
    finally:
        srv.close()
    main = res["variants"]["main"]
    for k in ("left_arm", "right_arm", "left_hand", "right_hand", "waist", "navigate_command"):
        assert main[k]["mse"] == pytest.approx(0.0, abs=1e-10), k
    assert res["variants"]["baseline_hold"]["left_arm"]["mse"] > 1e-4
    assert res["queries_main"] == len(range(0, N, 20))
    assert res["clamp"]["margin_0.0"]["targets"] == res["queries_main"] * 40 * 28
    assert res["latency_ms_during_eval"]["n"] == res["queries_main"]
    assert set(res["dropped_ranges"]) == {"waist", "base_height_command", "navigate_command"}
    assert json.loads((tmp_path / "open_loop.json").read_text())["episodes"] == [0]
