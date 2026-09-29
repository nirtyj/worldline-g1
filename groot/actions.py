"""GR00T action chunk -> what the body `arm` op executes on SONIC (decision (b), docs/groot_arms_design.md §5.2).

The checkpoint returns, per key, float32 (B, 40, D) ABSOLUTE targets at 50 Hz: left_arm 7, right_arm 7,
left_hand 7, right_hand 7 (GR00T hand order), waist 3, base_height_command 1, navigate_command 3.

What is executed and what is not:
- arms  -> the 14 arm entries of SONIC's 17-D upper body, placed BY NAME;
- hands -> Dex3 order, both hands always (the deploy commands a fist for a missing hand, input_interface.hpp:341-362);
- waist -> NOT GR00T's: the 3 waist entries hold `waist_hold` (default SONIC's stand waist, 0/0/0). The Arena
  checkpoint's waist action is identically 0 (its `statistics.json` action.waist min = max = 0) and SONIC owns balance;
- base_height_command, navigate_command -> DROPPED (never sent to SONIC; `manipulate` never moves the base,
  PLAN #17/#26). They and GR00T's waist are returned in `ArmChunk.dropped` for telemetry (`groot_nav_pred`).

Order on the way out: `ArmChunk.upper_body` is SONIC's interleaved wire order (joint_order.SONIC_UPPER_JOINTS), the
order the deploy consumes. The body `arm` op takes a 17-list in **mj17** order (body/arm.py `_vec17`) or a dict
{joint: rad}; use `upper_body_mj17` or `arm_targets_named(k)` for it, never `upper_body` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import joint_order as jo

ARM_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand")
TELEMETRY_KEYS = ("waist", "base_height_command", "navigate_command")


@dataclass(frozen=True, eq=False)
class ArmChunk:
    """A time-stamped chunk of upper-body targets. Row k is the target for t0_mono + k * dt."""

    upper_body: np.ndarray                 # (T, 17) SONIC wire order (joint_order.SONIC_UPPER_JOINTS), rad
    left_hand: np.ndarray                  # (T, 7) Dex3 order (joint_order.DEX3_HAND_JOINTS["left"]), rad
    right_hand: np.ndarray                 # (T, 7) Dex3 order
    dt: float = jo.DT
    t0_mono: float = 0.0                   # time.monotonic() of the observation the chunk was computed from
    dropped: Mapping[str, np.ndarray] = field(default_factory=dict)   # GR00T waist / base height / nav, (T, D)

    def __post_init__(self):
        T = self.upper_body.shape[0]
        if self.upper_body.shape != (T, jo.N_UPPER) or self.left_hand.shape != (T, jo.N_HAND) \
                or self.right_hand.shape != (T, jo.N_HAND):
            raise ValueError(f"inconsistent chunk shapes {self.upper_body.shape} {self.left_hand.shape} "
                             f"{self.right_hand.shape}")

    @property
    def T(self) -> int:
        return int(self.upper_body.shape[0])

    @property
    def duration_s(self) -> float:
        return self.T * self.dt

    @property
    def upper_body_mj17(self) -> np.ndarray:
        """(T, 17) in the body `arm` op's list order (waist, left arm, right arm; MuJoCo order)."""
        return jo.wire_to_mj17(self.upper_body)

    def index_at(self, t_mono: float) -> int:
        """Row to play at monotonic time t (clamped to [0, T-1]); P3 indexes by time, not by count."""
        return int(min(max(round((t_mono - self.t0_mono) / self.dt), 0), self.T - 1))

    def arm_targets_named(self, k: int, include_waist: bool = False) -> dict[str, float]:
        """Row k as {joint: rad} (the order-proof form of the `arm` op). Arms only by default: a waist name in the
        dict switches the op to the client's waist (body/arm.py `waist: "cmd"`)."""
        names = jo.SONIC_UPPER_JOINTS
        row = self.upper_body[k]
        return {n: float(v) for n, v in zip(names, row) if include_waist or n not in jo.WAIST_JOINTS}

    def clamped(self, margin: float = 0.0, limits: Mapping[str, tuple[float, float]] = jo.URDF_LIMITS
                ) -> tuple["ArmChunk", dict]:
        """A copy clamped to the URDF limits shrunk by `margin`, plus clamp statistics (clamp_stats)."""
        stats = clamp_stats(self, margin=margin, limits=limits)
        lo_u, hi_u = jo.limits_array(jo.SONIC_UPPER_JOINTS, limits, margin)
        hands = {}
        for side in jo.SIDES:
            lo, hi = jo.limits_array(jo.DEX3_HAND_JOINTS[side], limits, 0.0)   # the body clamps hands without margin
            hands[side] = np.clip(getattr(self, f"{side}_hand"), lo, hi)
        c = ArmChunk(np.clip(self.upper_body, lo_u, hi_u), hands["left"], hands["right"], self.dt, self.t0_mono,
                     self.dropped)
        return c, stats


def _get(action: Mapping[str, np.ndarray], key: str) -> np.ndarray | None:
    for k in (key, f"action.{key}"):
        if k in action:
            return np.asarray(action[k])
    return None


def _as_TD(a: np.ndarray, key: str, dim: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError(f"{key}: batch size {a.shape[0]} (one robot, B must be 1)")
        a = a[0]
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"{key}: expected (T, {dim}) or (1, T, {dim}), got {np.shape(a)}")
    return a


def to_arm_chunk(action: Mapping[str, np.ndarray], t0_mono: float, *,
                 waist_hold: Sequence[float] = jo.STAND_WAIST, dt: float = jo.DT) -> ArmChunk:
    """PolicyClient.get_action() output -> ArmChunk (arms + hands executed; waist held; nav/height dropped).

    Raises ValueError on a missing key, a shape mismatch or a non-finite value (the caller drops the chunk).
    """
    parts: dict[str, np.ndarray] = {}
    for key in ARM_KEYS:
        a = _get(action, key)
        if a is None:
            raise ValueError(f"action has no {key!r} (keys: {sorted(action)})")
        parts[key] = _as_TD(a, key, jo.GROOT_KEY_DIMS[key])
    T = parts["left_arm"].shape[0]
    if T < 1 or any(p.shape[0] != T for p in parts.values()):
        raise ValueError(f"action keys disagree on the horizon: { {k: p.shape[0] for k, p in parts.items()} }")
    if not all(np.isfinite(p).all() for p in parts.values()):
        raise ValueError("non-finite value in the action chunk")
    waist = np.asarray(waist_hold, dtype=np.float64).reshape(-1)
    if waist.size != 3:
        raise ValueError("waist_hold needs 3 values (yaw, roll, pitch)")
    # mj17 by name: waist (held), left arm, right arm -> SONIC wire order
    mj17 = np.concatenate([np.broadcast_to(waist, (T, 3)),
                           jo.reorder(parts["left_arm"], jo.GROOT_KEY_JOINTS["left_arm"], jo.LEFT_ARM_JOINTS),
                           jo.reorder(parts["right_arm"], jo.GROOT_KEY_JOINTS["right_arm"], jo.RIGHT_ARM_JOINTS)],
                          axis=1)
    upper = jo.mj17_to_wire(mj17)
    left = jo.groot_to_dex3_hand("left", parts["left_hand"])
    right = jo.groot_to_dex3_hand("right", parts["right_hand"])
    dropped = {}
    for key in TELEMETRY_KEYS:
        a = _get(action, key)
        if a is not None:
            dropped[key] = _as_TD(a, key, jo.GROOT_KEY_DIMS[key])
    return ArmChunk(np.ascontiguousarray(upper), np.ascontiguousarray(left), np.ascontiguousarray(right),
                    float(dt), float(t0_mono), dropped)


def clamp_stats(chunk: ArmChunk, margin: float = 0.0,
                limits: Mapping[str, tuple[float, float]] = jo.URDF_LIMITS) -> dict:
    """How much of a chunk lies outside the URDF limits. Arms use `margin` (the body clamps at 0.02 rad), hands none.

    Returns {"targets", "clamped", "clamped_frac", "arm_clamped", "hand_clamped", "per_joint": {joint: count},
    "max_violation_rad"}; `targets` counts every (row, joint) value, arms 14 + hands 14 per row (the held waist is
    ours, not GR00T's, and is not counted).
    """
    per_joint: dict[str, int] = {}
    arm_c = hand_c = 0
    worst = 0.0
    arm_idx = [i for i, n in enumerate(jo.SONIC_UPPER_JOINTS) if n not in jo.WAIST_JOINTS]
    arm_names = [jo.SONIC_UPPER_JOINTS[i] for i in arm_idx]
    blocks = [(chunk.upper_body[:, arm_idx], arm_names, margin, "arm")]
    for side in jo.SIDES:
        blocks.append((getattr(chunk, f"{side}_hand"), list(jo.DEX3_HAND_JOINTS[side]), 0.0, "hand"))
    total = 0
    for vals, names, m, kind in blocks:
        lo, hi = jo.limits_array(names, limits, m)
        over = np.maximum(vals - hi, 0.0) + np.maximum(lo - vals, 0.0)
        bad = over > 0.0
        total += vals.size
        n_bad = int(bad.sum())
        if kind == "arm":
            arm_c += n_bad
        else:
            hand_c += n_bad
        if n_bad:
            worst = max(worst, float(over.max()))
            for j, n in enumerate(names):
                c = int(bad[:, j].sum())
                if c:
                    per_joint[n] = per_joint.get(n, 0) + c
    clamped = arm_c + hand_c
    return {"targets": total, "clamped": clamped, "clamped_frac": clamped / total if total else 0.0,
            "arm_clamped": arm_c, "hand_clamped": hand_c, "per_joint": per_joint,
            "max_violation_rad": worst, "margin_rad": margin}
