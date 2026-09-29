"""Against the real PolicyServer (opt-in: `-m box`; GROOT_ENDPOINT, default tcp://127.0.0.1:5550).

Run on the dev box after `bash scripts/groot_server.sh start`, or anywhere with a tunnel to it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from groot import DEFAULT_ENDPOINT, joint_order as jo
from groot.actions import clamp_stats, to_arm_chunk
from groot.obs import ARENA_PROMPT, build_observation
from groot.policy_client import PolicyClient, PolicyError

pytestmark = pytest.mark.box

ENDPOINT = os.environ.get("GROOT_ENDPOINT", DEFAULT_ENDPOINT)


@pytest.fixture(scope="module")
def client():
    c = PolicyClient(ENDPOINT, timeout_s=30.0)
    if not c.ping(timeout_s=5.0):
        c.close()
        pytest.skip(f"no PolicyServer at {ENDPOINT}")
    yield c
    c.close()


def test_live_modality_config_is_the_arena_g1_schema(client):
    mc = client.get_modality_config()
    assert mc["video"]["modality_keys"] == [jo.VIDEO_KEY]
    assert mc["state"]["modality_keys"] == list(jo.GROOT_STATE_KEYS)
    assert mc["action"]["modality_keys"] == list(jo.GROOT_ACTION_KEYS)
    assert mc["action"]["delta_indices"] == list(range(jo.ACTION_HORIZON))
    assert mc["language"]["modality_keys"] == [jo.LANGUAGE_KEY]
    assert all(c["rep"] == "ABSOLUTE" for c in mc["action"]["action_configs"])


def test_live_get_action_shapes_and_mapping(client):
    q = np.asarray(jo.STAND_Q29)
    obs = build_observation(np.full((480, 640, 3), 90, np.uint8), q, np.zeros(7), np.zeros(7), ARENA_PROMPT)
    act = client.get_action(obs)
    for k in jo.GROOT_ACTION_KEYS:
        assert act[k].shape == (1, jo.ACTION_HORIZON, jo.GROOT_KEY_DIMS[k]) and act[k].dtype == np.float32, k
    chunk = to_arm_chunk(act, time.monotonic())
    assert np.isfinite(chunk.upper_body).all()
    assert clamp_stats(chunk)["targets"] == jo.ACTION_HORIZON * 28


def test_live_strict_server_rejects_a_bad_observation(client):
    obs = build_observation(np.zeros((480, 640, 3), np.uint8), np.zeros(29), np.zeros(7), np.zeros(7), "x")
    obs["state"]["waist"] = obs["state"]["waist"].astype(np.float64)
    with pytest.raises(PolicyError):
        client.get_action(obs)
    assert client.ping()
