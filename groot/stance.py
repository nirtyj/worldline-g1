"""Where a target lands in P1's `ego_view` (Arena's G1 head camera) for a given stance: the pinhole model behind the
GR00T stance (M2b wave 2, W2.5; docs/groot_serving.md §8).

The camera is fixed to the torso: `torso_link` + (0.0488135, 0, 0.30925) m, pitched 35.00 deg down, f 15 mm on a
20.955 x 15.71625 mm aperture at 640x480 (fx = fy = 458.12 px), clipping 0.1-5 m (docs/contracts/p1_m2b.md §5.1;
torso = pelvis + (-0.0039635, 0, 0.044) at zero waist). With SONIC standing (pelvis 0.787 m, waist at 0) that is
pelvis + (0.0449, 0, 0.353): 1.14 m above the floor. Arena's training frames show the apple in the lower part of this
view, left of centre (HF dataset episode 0, frame 60: its centre near u 105, v 330), so a stance should put the target
below the image's upper third (v >= 160), the bar of docs/M2b_wave1.md W2.5.

    ego_view_uv(forward, left, z)   -> (u, v) pixel of a point `forward` m ahead of the pelvis, `left` m to its left,
                                       `z` m above the floor (pelvis at pelvis_z); None behind the camera
    in_lower_two_thirds(v)          -> v >= 160
"""

from __future__ import annotations

import math

W, H = 640, 480
FX = FY = 15.0 * W / 20.955                      # 458.12 px
CX, CY = W / 2.0, H / 2.0
PITCH_DOWN = math.radians(35.0)
CAM_IN_PELVIS = (0.0488135 - 0.0039635, 0.0, 0.30925 + 0.044)   # (0.04485, 0, 0.35325) m at zero waist
PELVIS_Z = 0.787                                 # SONIC standing (docs/arm_tracking.md; p1_m2b.md §5.1)
UPPER_THIRD_V = H / 3.0


def ego_view_uv(forward: float, left: float, z: float, pelvis_z: float = PELVIS_Z) -> tuple[float, float] | None:
    dx = forward - CAM_IN_PELVIS[0]
    dy = left - CAM_IN_PELVIS[1]
    dz = z - (pelvis_z + CAM_IN_PELVIS[2])
    depth = dx * math.cos(PITCH_DOWN) - dz * math.sin(PITCH_DOWN)          # along the optical axis
    up = dx * math.sin(PITCH_DOWN) + dz * math.cos(PITCH_DOWN)             # image up
    if depth <= 0.0:
        return None
    return CX - FX * dy / depth, CY - FY * up / depth


def in_lower_two_thirds(v: float) -> bool:
    return v >= UPPER_THIRD_V
