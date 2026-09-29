"""brains/frame_gate.py: which frames System 1 gets.

    .venv-rt/bin/python -m pytest tests/kept/test_frame_gate.py
"""

from __future__ import annotations

import unittest

import numpy as np

from brains.frame_gate import FrameGate, thumbnail

RNG = np.random.default_rng(5)


def room(seed: int = 0) -> np.ndarray:
    """A 480x640 'room': smooth gradients plus a few boxes, so views differ."""
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:480, 0:640]
    img = np.stack([(x / 640 * 200 + r.integers(0, 40)), (y / 480 * 180 + r.integers(0, 40)),
                    np.full_like(x, 120 + r.integers(0, 60))], axis=-1).astype(np.float32)
    for _ in range(6):
        x0, y0 = r.integers(0, 560), r.integers(0, 400)
        img[y0:y0 + 80, x0:x0 + 80] = r.integers(0, 255, 3)
    return img.clip(0, 255).astype(np.uint8)


POSE = (1.0, 2.0, 90.0, 30.0)


class ThumbnailTest(unittest.TestCase):
    def test_shape_range_and_odd_sizes(self):
        t = thumbnail(room())
        self.assertEqual(t.shape, (24, 32))
        self.assertTrue(0.0 <= t.min() and t.max() <= 1.0)
        self.assertEqual(thumbnail(np.zeros((301, 401, 3), np.uint8)).shape, (24, 32))
        self.assertEqual(thumbnail(np.full((48, 64), 255, np.uint8)).max(), 1.0)      # grayscale input


class GateTest(unittest.TestCase):
    def test_first_frame_is_sent_and_a_still_robot_sends_nothing_more(self):
        g = FrameGate()
        f = room()
        self.assertEqual(g.decide(f, POSE, 0.0).reason, "first")
        for t in range(1, 30):
            d = g.decide(f.copy(), POSE, float(t))
            self.assertFalse(d.send)
            self.assertEqual(d.reason, "still")
        self.assertEqual(g.counts["still"], 29)

    def test_jpeg_like_noise_and_flicker_are_ignored(self):
        g = FrameGate()
        f = room()
        g.decide(f, POSE, 0.0)
        noisy = (f.astype(np.int16) + RNG.integers(-6, 7, f.shape)).clip(0, 255).astype(np.uint8)
        self.assertEqual(g.decide(noisy, POSE, 2.0).reason, "still")
        dimmer = (f * 0.96).astype(np.uint8)                                   # a 4% lighting change
        self.assertEqual(g.decide(dimmer, POSE, 3.0).reason, "still")

    def test_a_small_object_moved_while_standing_still_is_a_scene_change(self):
        g = FrameGate()
        f = room()
        g.decide(f, POSE, 0.0)
        g2 = f.copy()
        g2[300:360, 400:450] = (250, 20, 20)                                 # a red mug appears (50x60 px)
        d = g.decide(g2, POSE, 5.0)
        self.assertTrue(d.send)
        self.assertEqual(d.reason, "scene changed")
        self.assertEqual(g.decide(g2, POSE, 6.0).reason, "still")           # and only once

    def test_moving_a_little_is_a_similar_view_turning_or_moving_far_is_new(self):
        g = FrameGate()
        f = room()
        g.decide(f, POSE, 0.0)
        shifted = np.roll(f, 8, axis=1)                                       # crept forward 20 cm
        self.assertEqual(g.decide(shifted, (1.2, 2.0, 90.0, 30.0), 2.0).reason, "similar view")
        self.assertEqual(g.decide(room(1), (1.3, 2.0, 90.0, 30.0), 3.0).reason, "similar view")  # 0.3 m, a bit different
        self.assertEqual(g.decide(255 - f, (1.3, 2.0, 90.0, 30.0), 4.0).reason, "new view")  # through a doorway
        self.assertEqual(g.decide(room(1), (1.3, 2.0, 135.0, 30.0), 5.0).reason, "new view")  # turned 45 degrees
        self.assertEqual(g.decide(room(1), (2.5, 2.0, 135.0, 30.0), 6.0).reason, "new view")  # moved 1.2 m
        self.assertEqual(g.decide(room(1), (2.5, 2.0, 135.0, 60.0), 7.0).reason, "new view")  # looked down 30

    def test_the_robots_own_action_is_not_a_scene_change(self):
        g = FrameGate()
        f = room()
        g.decide(f, POSE, 0.0)
        picked = f.copy()
        picked[300:360, 400:450] = (0, 0, 0)                                  # the mug is gone: it picked it up
        self.assertEqual(g.decide(picked, POSE, 5.0, expected=True).reason, "own action")
        self.assertEqual(g.decide(picked, POSE, 6.0).reason, "still")        # and not reported afterwards
        moved = picked.copy()
        moved[0:60, 0:60] = (255, 255, 255)                                   # then someone else changes something
        self.assertEqual(g.decide(moved, POSE, 7.0).reason, "scene changed")

    def test_yaw_wraps_around(self):
        g = FrameGate()
        f = room()
        g.decide(f, (0, 0, 359.0, 0), 0.0)
        self.assertEqual(g.decide(f, (0, 0, 1.0, 0), 2.0).reason, "still")  # 2 degrees, not 358

    def test_at_most_one_frame_a_second(self):
        g = FrameGate()
        g.decide(room(0), POSE, 0.0)
        d = g.decide(room(1), (3.0, 3.0, 0.0, 30.0), 0.4)
        self.assertEqual((d.send, d.reason), (False, "too soon"))
        self.assertTrue(g.decide(room(1), (3.0, 3.0, 0.0, 30.0), 1.1).send)

    def test_counts_summary_and_reset(self):
        g = FrameGate()
        f = room()
        g.decide(f, POSE, 0.0)
        for t in range(1, 4):
            g.decide(f, POSE, float(t))
        self.assertEqual(dict(g.skipped_since_send), {"still": 3})
        self.assertEqual(g.summary(), "sent 1 (first 1), skipped 3 (still 3)")
        g2 = f.copy()
        g2[0:100, 0:100] = 0
        g.decide(g2, POSE, 10.0)
        self.assertEqual(dict(g.skipped_since_send), {})
        g.reset()
        self.assertEqual(g.decide(f, POSE, 11.0).reason, "first")


if __name__ == "__main__":
    unittest.main()
