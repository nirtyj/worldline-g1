"""In-process harness for ArmChannel tests: a fake clock, a recording mux, a g1_debug source and a SONIC-like arm
plant (dead time + first-order lag + a per-joint steady-state bias, on the upper body the override commands), so the
channel's 50 Hz behaviour is tested tick by tick and deterministically (no threads, no ZMQ)."""

from __future__ import annotations

import collections
import math
from types import SimpleNamespace

from body import joint_map as jm
from body.arm import ArmChannel
from tools.fake_deploy import REF_Q29

DT = 0.02


class Clock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t


class Mux:
    """Records what SonicMux would put on every planner message (wire order in, mj17 kept for the tests)."""

    def __init__(self):
        self.upper = None                      # (mj17, vel_mj17, left7|None, right7|None) or None
        self.log: list = []                    # (t, mj17 | None, left, right)
        self.cleared = 0
        self.clock = None

    def set_upper(self, pos17, vel17=None, left7=None, right7=None):
        assert len(pos17) == 17
        self.upper = (jm.mj17_from_wire(pos17), jm.mj17_from_wire(vel17 or [0.0] * 17),
                      None if left7 is None else list(left7), None if right7 is None else list(right7))
        self.log.append((None if self.clock is None else self.clock(), self.upper[0], self.upper[2], self.upper[3]))

    def clear_upper(self):
        self.upper = None
        self.cleared += 1
        self.log.append((None if self.clock is None else self.clock(), None, None, None))


class Deploy:
    """g1_debug as the channel reads it: body_q, body_q_target (SONIC's reference), left/right_hand_q."""

    def __init__(self, clock: Clock):
        self.clock = clock
        self.latest = {"body_q": list(REF_Q29), "body_q_target": list(REF_Q29),
                       "left_hand_q": list(jm.DEX3_CLOSED["left"]), "right_hand_q": list(jm.DEX3_CLOSED["right"])}
        self.t_latest = clock()
        self.stale = False

    def age_s(self) -> float:
        return 5.0 if self.stale else self.clock() - self.t_latest


class Plant:
    """SONIC's arm response to the override (docs/arm_tracking.md §3.2): each upper-body joint follows the commanded
    value after `dead_s` with a first-order lag `tau_s`, plus `bias[k]` (pose-independent here). Without an override
    the joints follow SONIC's reference. Hands: first order, 0.1 s. `yaw_gain` scales the waist-yaw response (G0:
    ~0.78)."""

    def __init__(self, mux: Mux, dep: Deploy, dead_s: float = 0.09, tau_s: float = 0.085, yaw_gain: float = 1.0):
        self.mux, self.dep = mux, dep
        self.dead_s, self.tau_s, self.yaw_gain = dead_s, tau_s, yaw_gain
        self.bias = [0.0] * 17
        self.buf: collections.deque = collections.deque()

    def step(self, t: float, dt: float = DT) -> None:
        up = self.mux.upper
        ref = jm.mj17_from_mujoco(self.dep.latest["body_q_target"])
        cmd = list(up[0]) if up is not None else ref
        self.buf.append((t, cmd, up))
        while len(self.buf) > 1 and self.buf[1][0] <= t - self.dead_s:
            self.buf.popleft()
        u = self.buf[0][1] if self.buf[0][0] <= t - self.dead_s else ref
        q = self.dep.latest["body_q"]
        a = 1.0 - math.exp(-dt / self.tau_s)
        for k in range(17):
            tgt = u[k] + self.bias[k]
            if k == 0:
                tgt = ref[0] + (u[0] - ref[0]) * self.yaw_gain + self.bias[0]
            q[12 + k] += (tgt - q[12 + k]) * a
        hands = self.buf[0][2] if self.buf[0][0] <= t - self.dead_s else None
        b = 1.0 - math.exp(-dt / 0.1)
        for side, idx in (("left", 2), ("right", 3)):
            h = hands[idx] if hands is not None and hands[idx] is not None else jm.DEX3_CLOSED[side]
            hq = self.dep.latest[f"{side}_hand_q"]
            for j in range(7):
                hq[j] += (h[j] - hq[j]) * b
        self.dep.t_latest = t


class Rig:
    def __init__(self, cfg: dict | None = None, pose=None, plant_kw: dict | None = None):
        self.clock = Clock()
        self.mux = Mux()
        self.mux.clock = self.clock
        self.dep = Deploy(self.clock)
        self.plant = Plant(self.mux, self.dep, **(plant_kw or {}))
        self.events: list[dict] = []
        self.records: dict = {}
        self.pose = pose
        self.ch = ArmChannel(SimpleNamespace(**(cfg or {})), self.mux, self.dep, self._emit, log=lambda *_: None,
                             record=self._record, pose=lambda: self.pose, clock=self.clock)

    def _emit(self, op_id, state, data):
        self.events.append({"id": op_id, "state": state, "data": data, "t": self.clock()})

    def _record(self, op_id, op, args):
        self.records[op_id] = op

    def arm(self, args: dict, op_id: str = "op", ok_start=(True, None, {})) -> dict:
        return self.ch.handle(op_id, dict(args), ok_start)

    def run(self, secs: float, each=None) -> None:
        """Advance the clock tick by tick: each(t) (messages), then the channel tick, then the plant."""
        n = int(round(secs / DT))
        for _ in range(n):
            self.clock.t += DT
            if each is not None:
                each(self.clock.t)
            self.ch.tick(self.clock.t)
            self.plant.step(self.clock.t)

    def terminal(self, op_id: str) -> dict | None:
        for e in self.events:
            if e["id"] == op_id and e["state"] in ("succeeded", "failed", "canceled"):
                return e
        return None

    def sent(self) -> list[float]:
        return list(self.mux.upper[0])

    def q(self) -> list[float]:
        return jm.mj17_from_mujoco(self.dep.latest["body_q"])


def chunk(seq: int, t0: float, rows: list[list[float]], hands: float | list | None = 0.0, dt: float = 0.02,
          order: str = "mj17", **kw) -> dict:
    """A chunk message body; rows in mj17 (or wire when order='wire'); hands a closure or a 7-list for every row."""
    T = len(rows)

    def hand(side):
        if isinstance(hands, list):
            return [list(hands)] * T
        return [jm.hand_closure(side, float(hands or 0.0))] * T

    return {"seq": seq, "t0_mono": t0, "dt": dt, "order": order, "upper_body": [list(r) for r in rows],
            "left_hand": hand("left"), "right_hand": hand("right"), **kw}


def base(session: str = "s1", gen: int = 1, epoch: int = 1, **kw) -> dict:
    import time as _t

    return {"stream": session, "session_id": session, "execution_id": session, "generation": gen,
            "control_epoch": epoch, "mode": "chunk", "t_wall": _t.time(), **kw}
