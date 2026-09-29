"""groot.obs.build_observation and groot.actions.to_arm_chunk: shapes, dtypes, joint placement by name, the waist
hold, the dropped navigation/height keys and the URDF clamp statistics."""

from __future__ import annotations

import numpy as np
import pytest

from groot import joint_order as jo
from groot.actions import ArmChunk, clamp_stats, to_arm_chunk
from groot.obs import ARENA_PROMPT, build_observation

IMG = np.zeros((480, 640, 3), dtype=np.uint8)


def _obs(**kw):
    args = dict(ego_rgb_uint8_HxWx3=IMG, body_q_mj29=np.arange(29, dtype=float) / 100,
                left_hand_q7=np.arange(7, dtype=float) / 10, right_hand_q7=-np.arange(7, dtype=float) / 10,
                prompt=ARENA_PROMPT)
    args.update(kw)
    return build_observation(**args)


def test_observation_structure_matches_strict_server_check():
    o = _obs()
    assert set(o) == {"video", "state", "language"}
    v = o["video"]["ego_view"]
    assert v.dtype == np.uint8 and v.shape == (1, 1, 480, 640, 3) and v.flags["C_CONTIGUOUS"]
    assert list(o["state"]) == ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
    for k, a in o["state"].items():
        assert a.dtype == np.float32 and a.shape == (1, 1, jo.GROOT_KEY_DIMS[k]), k
    assert o["language"] == {"annotation.human.task_description": [[ARENA_PROMPT]]}


def test_observation_state_values_by_name():
    o = _obs()
    assert np.allclose(o["state"]["left_arm"][0, 0], np.arange(15, 22) / 100)
    assert np.allclose(o["state"]["right_arm"][0, 0], np.arange(22, 29) / 100)
    assert np.allclose(o["state"]["waist"][0, 0], np.arange(12, 15) / 100)
    # Dex3 slots (thumb_0..index_1) -> GR00T order (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2)
    assert np.allclose(o["state"]["left_hand"][0, 0], np.array([5, 6, 3, 4, 0, 1, 2]) / 10)
    assert np.allclose(o["state"]["right_hand"][0, 0], -np.array([5, 6, 3, 4, 0, 1, 2]) / 10)


@pytest.mark.parametrize("bad", [
    dict(ego_rgb_uint8_HxWx3=IMG.astype(np.float32)),
    dict(ego_rgb_uint8_HxWx3=np.zeros((480, 640), dtype=np.uint8)),
    dict(ego_rgb_uint8_HxWx3=np.zeros((240, 320, 3), dtype=np.uint8)),
    dict(body_q_mj29=np.zeros(28)),
    dict(left_hand_q7=np.zeros(6)),
    dict(right_hand_q7=np.array([0, 0, 0, np.nan, 0, 0, 0])),
    dict(prompt=""),
])
def test_observation_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        _obs(**bad)


def test_observation_any_size_when_allowed():
    o = build_observation(np.zeros((240, 320, 3), np.uint8), np.zeros(29), np.zeros(7), np.zeros(7), "x",
                          expect_hw=None)
    assert o["video"]["ego_view"].shape == (1, 1, 240, 320, 3)


def _action(T=40, batch=True, prefix=""):
    """Distinct value per (key, joint): 10*key_id + joint index + t/1000."""
    t = np.arange(T)[:, None] / 1000.0
    a = {}
    for kid, key in enumerate(jo.GROOT_ACTION_KEYS):
        d = jo.GROOT_KEY_DIMS[key]
        v = (10 * (kid + 1) + np.arange(d)[None, :] + t).astype(np.float32)
        a[prefix + key] = v[None] if batch else v
    return a


def test_to_arm_chunk_places_every_joint_by_name():
    act = _action()
    c = to_arm_chunk(act, t0_mono=123.0)
    assert c.T == 40 and c.dt == pytest.approx(0.02) and c.t0_mono == 123.0 and c.duration_s == pytest.approx(0.8)
    assert c.upper_body.shape == (40, 17) and c.left_hand.shape == (40, 7) and c.right_hand.shape == (40, 7)
    la, ra = act["left_arm"][0], act["right_arm"][0]
    for i, n in enumerate(jo.SONIC_UPPER_JOINTS):
        if n in jo.WAIST_JOINTS:
            assert np.all(c.upper_body[:, i] == 0.0), n             # stand waist held, GR00T's waist unused
        elif n in jo.LEFT_ARM_JOINTS:
            assert np.allclose(c.upper_body[:, i], la[:, jo.LEFT_ARM_JOINTS.index(n)]), n
        else:
            assert np.allclose(c.upper_body[:, i], ra[:, jo.RIGHT_ARM_JOINTS.index(n)]), n
    # wire slot 3/4 = left/right shoulder pitch
    assert np.allclose(c.upper_body[:, 3], la[:, 0]) and np.allclose(c.upper_body[:, 4], ra[:, 0])
    assert np.allclose(c.upper_body_mj17[:, 3:10], la) and np.allclose(c.upper_body_mj17[:, 10:17], ra)
    for side in jo.SIDES:
        g = act[f"{side}_hand"][0]
        h = getattr(c, f"{side}_hand")
        for j, n in enumerate(jo.DEX3_HAND_JOINTS[side]):
            assert np.allclose(h[:, j], g[:, jo.GROOT_HAND_JOINTS[side].index(n)]), n


def test_to_arm_chunk_drops_navigation_and_height():
    act = _action()
    c = to_arm_chunk(act, 0.0)
    assert set(c.dropped) == {"waist", "base_height_command", "navigate_command"}
    assert c.dropped["navigate_command"].shape == (40, 3) and c.dropped["base_height_command"].shape == (40, 1)
    assert np.allclose(c.dropped["waist"], act["waist"][0])
    executed = np.concatenate([c.upper_body.ravel(), c.left_hand.ravel(), c.right_hand.ravel()])
    for key in ("navigate_command", "base_height_command", "waist"):
        assert not np.isin(act[key][0].ravel(), executed).any(), key


def test_to_arm_chunk_variants_and_waist_hold():
    c1 = to_arm_chunk(_action(batch=False, prefix="action."), 0.0, waist_hold=(0.1, -0.05, 0.2))
    c2 = to_arm_chunk(_action(), 0.0, waist_hold=(0.1, -0.05, 0.2))
    assert np.allclose(c1.upper_body, c2.upper_body)
    assert np.allclose(c1.upper_body[:, :3], [0.1, -0.05, 0.2])
    minimal = {k: v for k, v in _action().items() if k in ("left_arm", "right_arm", "left_hand", "right_hand")}
    assert to_arm_chunk(minimal, 0.0).dropped == {}


@pytest.mark.parametrize("mutate", [
    lambda a: a.pop("left_hand"),
    lambda a: a.__setitem__("right_arm", a["right_arm"][:, :20]),
    lambda a: a.__setitem__("left_arm", np.concatenate([a["left_arm"]] * 2)),
    lambda a: a.__setitem__("left_arm", a["left_arm"][..., :6]),
    lambda a: a["right_hand"].__setitem__((0, 3, 2), np.nan),
])
def test_to_arm_chunk_rejects_bad_chunks(mutate):
    a = _action()
    mutate(a)
    with pytest.raises(ValueError):
        to_arm_chunk(a, 0.0)


def test_chunk_time_index_and_named_targets():
    c = to_arm_chunk(_action(), t0_mono=10.0)
    assert c.index_at(9.0) == 0 and c.index_at(10.0) == 0 and c.index_at(10.021) == 1
    assert c.index_at(10.5) == 25 and c.index_at(99.0) == 39
    named = c.arm_targets_named(5)
    assert set(named) == set(jo.LEFT_ARM_JOINTS + jo.RIGHT_ARM_JOINTS)
    assert named["left_elbow_joint"] == pytest.approx(float(c.upper_body_mj17[5, 6]))
    assert set(c.arm_targets_named(0, include_waist=True)) == set(jo.MJ17_JOINTS)
    with pytest.raises(ValueError):
        ArmChunk(np.zeros((3, 17)), np.zeros((2, 7)), np.zeros((3, 7)))


def test_clamp_stats_and_clamped_copy():
    act = {k: np.zeros((1, 40, jo.GROOT_KEY_DIMS[k]), np.float32) for k in jo.GROOT_ACTION_KEYS}
    c = to_arm_chunk(act, 0.0)
    s = clamp_stats(c)
    assert s["targets"] == 40 * 28 and s["clamped"] == 0 and s["clamped_frac"] == 0.0
    # left elbow 2.2 rad (> 2.0944) on 10 rows; right thumb_2 +0.1 (upper limit 0.0) on 4 rows
    act["left_arm"][0, :10, 3] = 2.2
    act["right_hand"][0, :4, 6] = 0.1                   # GR00T slot 6 = thumb_2
    c = to_arm_chunk(act, 0.0)
    s = clamp_stats(c)
    assert s["arm_clamped"] == 10 and s["hand_clamped"] == 4 and s["clamped"] == 14
    assert s["clamped_frac"] == pytest.approx(14 / (40 * 28))
    assert s["per_joint"] == {"left_elbow_joint": 10, "right_hand_thumb_2_joint": 4}
    assert s["max_violation_rad"] == pytest.approx(2.2 - 2.0944, abs=1e-6)
    # an elbow at 2.08 is inside the URDF range but outside it minus the body's 0.02 margin
    act["left_arm"][0, :10, 3] = 2.08
    s2 = clamp_stats(to_arm_chunk(act, 0.0), margin=0.02)
    assert s2["arm_clamped"] == 10
    cc, _ = to_arm_chunk(act, 0.0).clamped(margin=0.02)
    assert np.all(cc.upper_body_mj17[:, 6] <= 2.0944 - 0.02 + 1e-12)
    assert np.all(cc.right_hand[:, 2] <= 0.0)


# The checkpoint's action normalization band (statistics.json, new_embodiment/action, q01/q99; use_percentiles).
# decode clips the model output to [-1, 1] before unnormalizing (Isaac-GR00T @ 4b1dca9d gr00t/data/utils.py:150),
# so every target it can emit lies in [q01, q99].
CKPT_ACTION_Q01_Q99 = {
    "left_arm": ([-0.7372, -0.2056, -0.3444, -0.4802, -0.3368, -0.588, -0.9391],
                 [-0.0337, 0.6631, 0.4394, 0.8202, 1.1451, 0.9451, 0.2177]),
    "right_arm": ([-0.8034, -0.6453, -0.4363, 0.0072, -0.0019, -0.2574, -0.1273],
                  [-0.2612, -0.3513, -0.2358, 1.1324, 0.4071, 0.5929, 0.3471]),
    "left_hand": ([-0.6, -1.2, -0.6, -1.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7]),
    "right_hand": ([0.0] * 7, [0.0] * 7),
}


def test_checkpoint_output_band_lies_inside_the_urdf_limits():
    """Why the clamp fraction of this checkpoint is 0 by construction: its [q01, q99] band is inside the URDF limits
    (arms even with the body's 0.02 rad margin). A fine-tuned checkpoint must be re-checked (new statistics.json)."""
    for key, (q01, q99) in CKPT_ACTION_Q01_Q99.items():
        names = jo.GROOT_KEY_JOINTS[key]
        lo, hi = jo.limits_array(names, margin=0.02 if key.endswith("arm") else 0.0)
        assert np.all(np.asarray(q01) >= lo - 1e-9) and np.all(np.asarray(q99) <= hi + 1e-9), key
    for i in (0, 1):                                   # a chunk at the band's lower / upper edge clamps nothing
        act = {k: np.asarray(v[i], np.float32)[None, None].repeat(40, 1) for k, v in CKPT_ACTION_Q01_Q99.items()}
        assert clamp_stats(to_arm_chunk(act, 0.0), margin=0.02)["clamped"] == 0


def test_request_image_area256_is_the_exact_area_resize():
    """area256 = cv2.INTER_AREA 640x480 -> 256x192 (the server's first resize after its letterbox, docs/groot_serving.md
    §9). Checked here against an exact rational area average on crops; the box check compares with cv2 itself."""
    from fractions import Fraction

    from groot.obs import area_downscale_2p5, request_frame

    def exact(a):
        h, w, c = a.shape

        def wts(n, m):
            return [[max(Fraction(0), min(Fraction(5, 2) * (j + 1), i + 1) - max(Fraction(5, 2) * j, i)) / Fraction(5, 2)
                     for i in range(n)] for j in range(m)]
        wr, wc = wts(h, 2 * h // 5), wts(w, 2 * w // 5)
        o = np.zeros((2 * h // 5, 2 * w // 5, c), np.uint8)
        for y in range(o.shape[0]):
            for x in range(o.shape[1]):
                for ch in range(c):
                    v = sum(wr[y][i] * wc[x][k] * int(a[i, k, ch]) for i in range(h) if wr[y][i]
                            for k in range(w) if wc[x][k])
                    o[y, x, ch] = int((2 * v + 1) // 2)
        return o
    rng = np.random.default_rng(3)
    for _ in range(2):
        a = rng.integers(0, 256, (15, 20, 3), dtype=np.uint8)
        assert np.array_equal(area_downscale_2p5(a), exact(a))
    assert area_downscale_2p5(np.full((480, 640, 3), 255, np.uint8)).min() == 255
    full = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert request_frame(full, "full") is full
    o = _obs(ego_rgb_uint8_HxWx3=full, request_image="area256")
    v = o["video"]["ego_view"]
    assert v.dtype == np.uint8 and v.shape == (1, 1, 192, 256, 3) and v.flags["C_CONTIGUOUS"]
    assert np.array_equal(v[0, 0], area_downscale_2p5(full))
    with pytest.raises(ValueError):
        _obs(request_image="jpeg")
    with pytest.raises(ValueError):
        build_observation(np.zeros((240, 320, 3), np.uint8), np.zeros(29), np.zeros(7), np.zeros(7), "x",
                          expect_hw=None, request_image="area256")
