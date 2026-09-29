"""B.5 waist scan through the arm channel: op `scan` (handler `ArmChannel.handle_scan`).

    {yaw_deg: [-35, 0, 35], move_s (0.8), hold_s (0.8), measure_s (0.3), return_zero (true), servo_waist (true),
     yaw_ff (1.0), hold_on_end, preempt, stream, execution_id, generation, control_epoch}

Only waist YAW is commanded (G0, docs/arm_tracking.md: SONIC does not move waist pitch under the override at all and
follows waist yaw with a gain of about 0.78; roll and pitch stay SONIC's reference). The 17-vector is built only
through joint_map (the channel's `wire_from_mj17`): waist yaw from the scan, waist roll/pitch from SONIC's live
reference (`waist: "yaw"`), the 14 arm joints frozen at the pose being sent when the scan starts (a CarryLock hold,
or SONIC's own reference arms when nothing was driving them) with the servo holding them, the hands as they were.
For each yaw: a min-jerk move over `move_s`, a hold of `hold_s`, and at the end of the hold one
`progress` event `{kind: "scan.hold", i, yaw, pitch, yaw_cmd, pitch_cmd, yaw_deg, yaw_cmd_deg, pitch_deg,
yaw_err_deg}` with the ACHIEVED waist yaw / pitch (g1_debug body_q, mean over the last `measure_s`). The waist then
returns to 0 and the scan ends `succeeded` into `hold_on_end`: `target` (the held arms, waist back to SONIC) when the
scan took over a hold, else `stand` (blend back, override dropped). The servo also runs on waist yaw
(`servo_waist`), so the achieved yaw converges on the command within the hold.

Result: `holds [...]`, `yaw_err_deg_max`, `arm_dev_rad_max` (largest deviation of a measured arm joint from its value at
the start: "only the waist moves"), `waist_roll_pitch_dev_rad_max`, `base_shift_m` (GT pelvis xy, start to end).
"""

from __future__ import annotations

import math

import numpy as np

from .arm import ARM_IDX, YAW_IDX, ArmError, _minjerk, _num


class ScanPlan:
    op = "scan"
    waist_mode = "yaw"
    progress_hz = 0.0              # scan.hold events come from on_tick

    def __init__(self, q0: list[float], hands0: dict, yaws: list[float], move_s: float, hold_s: float,
                 measure_s: float, return_zero: bool, servo_waist: bool, yaw_ff: float, hold_on_end: str, pose0):
        self.q0 = list(q0)
        self.q0[YAW_IDX] = 0.0
        self.hands0 = dict(hands0)
        self.yaws, self.move_s, self.hold_s, self.measure_s = yaws, move_s, hold_s, min(measure_s, hold_s)
        self.yaw_ff = yaw_ff
        self.servo_idx = ARM_IDX + ((YAW_IDX,) if servo_waist else ())
        self.hold_on_end = hold_on_end
        self.pose0 = pose0
        # schedule: (t_start, y_from, y_to, t_hold_end, i) per hold; then the return move
        self.segs = []
        t, y = 0.0, 0.0
        for i, yw in enumerate(yaws):
            self.segs.append((t, y, yw, t + move_s + hold_s, i))
            t, y = t + move_s + hold_s, yw
        self.t_ret = t
        self.y_last = y
        self.duration_s = t + (move_s + 0.3 if return_zero and abs(y) > 1e-6 else 0.1)
        self.return_zero = return_zero
        self.samples: list[list] = [[] for _ in yaws]         # (yaw, pitch) measured in each hold's window
        self.holds: list[dict] = []
        self.emitted = set()
        self.arm0: list[float] | None = None
        self.arm_dev = 0.0
        self.wrp0: list[float] | None = None
        self.wrp_dev = 0.0
        self.pose_last = None

    def yaw_cmd(self, t: float) -> float:
        for t0, y0, y1, t_end, _ in self.segs:
            if t < t_end:
                return y0 + (y1 - y0) * _minjerk((t - t0) / self.move_s)
        if not self.return_zero:
            return self.y_last
        return self.y_last * (1.0 - _minjerk((t - self.t_ret) / self.move_s))

    def sample(self, t: float, ref: list[float]):
        q = list(self.q0)
        q[YAW_IDX] = self.yaw_cmd(t) * self.yaw_ff
        return q, self.hands0

    def on_tick(self, t: float, now: float, ch) -> list:
        qm = ch.measured_mj17()
        if qm is None or ch.deploy.age_s() > 0.1:
            return []
        if self.arm0 is None:
            self.arm0 = [qm[k] for k in ARM_IDX]
            self.wrp0 = [qm[1], qm[2]]
        self.arm_dev = max(self.arm_dev, max(abs(qm[k] - a) for k, a in zip(ARM_IDX, self.arm0)))
        self.wrp_dev = max(self.wrp_dev, abs(qm[1] - self.wrp0[0]), abs(qm[2] - self.wrp0[1]))
        pose = ch.gt_pose()
        if pose is not None:
            self.pose_last = pose
        out = []
        for t0, y0, y1, t_end, i in self.segs:
            if t_end - self.measure_s <= t < t_end:
                self.samples[i].append((qm[YAW_IDX], qm[2]))
            elif t >= t_end and i not in self.emitted:
                self.emitted.add(i)
                sm = self.samples[i]
                yaw = float(np.mean([a for a, _ in sm])) if sm else float("nan")
                pitch = float(np.mean([b for _, b in sm])) if sm else float("nan")
                h = {"kind": "scan.hold", "i": i, "yaw": round(yaw, 4), "pitch": round(pitch, 4), "yaw_cmd": round(y1, 4),
                     "pitch_cmd": 0.0, "yaw_deg": round(math.degrees(yaw), 2), "yaw_cmd_deg": round(math.degrees(y1), 2),
                     "pitch_deg": round(math.degrees(pitch), 2), "yaw_err_deg": round(math.degrees(yaw - y1), 2),
                     "n": len(sm), "t": round(t, 2)}
                self.holds.append(h)
                out.append(h)
        return out

    def progress(self, t: float) -> dict:
        return {"t": round(t, 2), "duration_s": round(self.duration_s, 2), "yaw_cmd": round(self.yaw_cmd(t), 4),
                "holds_done": len(self.holds)}

    def brief(self) -> dict:
        return {"yaw_deg": [round(math.degrees(y), 1) for y in self.yaws], "move_s": self.move_s, "hold_s": self.hold_s,
                "duration_s": round(self.duration_s, 2), "hold_on_end": self.hold_on_end, "yaw_ff": self.yaw_ff,
                "servo_waist": YAW_IDX in self.servo_idx}

    def result(self) -> dict:
        errs = [abs(h["yaw_err_deg"]) for h in self.holds if h["n"]]
        shift = None
        if self.pose0 is not None and self.pose_last is not None:
            shift = round(math.hypot(self.pose_last.x - self.pose0.x, self.pose_last.y - self.pose0.y), 4)
        return {**self.brief(), "holds": list(self.holds), "yaw_err_deg_max": round(max(errs), 2) if errs else None,
                "arm_dev_rad_max": round(self.arm_dev, 4), "waist_roll_pitch_dev_rad_max": round(self.wrp_dev, 4),
                "base_shift_m": shift}


def build(ch, args: dict, now: float) -> ScanPlan:
    yd = args.get("yaw_deg", [-35.0, 0.0, 35.0])
    try:
        yaws = [math.radians(float(y)) for y in yd]
    except (TypeError, ValueError):
        raise ArmError("bad_args", {"arg": "yaw_deg", "error": "a list of degrees"})
    if not (1 <= len(yaws) <= 12) or not all(math.isfinite(y) and abs(y) <= math.radians(60.0) for y in yaws):
        raise ArmError("bad_args", {"arg": "yaw_deg", "error": "1-12 finite values within +-60 deg"})
    if args.get("pitch_deg") not in (None, 0, 0.0, [0], [0.0]):
        raise ArmError("bad_args", {"arg": "pitch_deg", "error": "not commanded: SONIC does not move waist pitch under "
                                                                 "the override (docs/arm_tracking.md)"})
    took_hold = ch.hold is not None and ch.hold.kind in ("target", "measured")
    hold = args.get("hold_on_end") or ("target" if took_hold else "stand")
    if hold not in ("target", "measured", "stand"):
        raise ArmError("bad_args", {"arg": "hold_on_end", "error": "target | measured | stand"})
    hands0 = {s: (None if ch.hands_sent.get(s) is None else list(ch.hands_sent[s])) for s in ("left", "right")}
    return ScanPlan(ch.continuity_pose(), hands0, yaws, _num(args, "move_s", 0.2, 5.0, 0.8),
                    _num(args, "hold_s", 0.2, 10.0, 0.8), _num(args, "measure_s", 0.05, 2.0, 0.3),
                    bool(args.get("return_zero", True)), bool(args.get("servo_waist", True)),
                    _num(args, "yaw_ff", 0.5, 2.0, 1.0), hold, ch.gt_pose())
