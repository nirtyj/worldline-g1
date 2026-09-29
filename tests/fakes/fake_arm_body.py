"""An in-process wl-body arm channel in chunk mode (docs/contracts/arm_chunk.md), plus the sensors GR00T reads, for
offline tests of services/executors/groot_arms.py. It implements the contract, not the SONIC physics:

    ArmPort     arm(args, op_id) -> reply; subscribe(cb) -> unsubscribe. Sessions keyed by stream, fenced by
                (session_id, generation, control_epoch); chunks validated (shape, finite, order), dropped as
                out_of_order / expired, played by a 50 Hz thread: k = round((now + lead_s - t0_mono) / dt), a 5-tick
                cross-fade, the last row held with stall_s counting, a clamp at groot.joint_order.URDF_LIMITS - 0.02 rad
                (arms) and the limits (hands) counted over the 28 GR00T values per tick; the session watchdog;
                hold_on_end; the B.1 halt latch (halt(epoch): hold measured, reject control_epoch <= epoch).
                Events: `arm.progress` at 5 Hz and one terminal event per session, in the body.event envelope.
    SensorPort  ego_frame() -> a 480x640 frame stamped now; debug_state() -> g1_debug {body_q (29, MuJoCo), left/right
                _hand_q (Dex3)} whose upper body and hands follow what was played with a 0.1 s first-order lag.

Every message is recorded with its receive time (`log`), so tests can prove that nothing was published after a
cancel ack. `on_tick(fn)` lets a fake world react to the measured hands (a lift after a grasp).
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any, Callable

import numpy as np

from groot import joint_order as jo

HZ = 50.0
N_UPPER, N_HAND = jo.N_UPPER, jo.N_HAND
ARM_WIRE_IDX = [i for i, n in enumerate(jo.SONIC_UPPER_JOINTS) if n not in jo.WAIST_JOINTS]
_UB_LO, _UB_HI = jo.limits_array(jo.SONIC_UPPER_JOINTS, jo.URDF_LIMITS, 0.02)
_HAND_LIM = {s: jo.limits_array(jo.DEX3_HAND_JOINTS[s], jo.URDF_LIMITS, 0.0) for s in jo.SIDES}
STAND_WIRE = np.asarray([jo.STAND_Q29[i] for i in jo.SONIC_WIRE_FROM_MUJOCO], dtype=np.float64)


class FakeArmBody:
    def __init__(self, *, frames_fresh: bool = True, debug_fresh: bool = True, watchdog_default_s: float = 2.0):
        self.lock = threading.RLock()
        self.subs: list[Callable[[dict], None]] = []
        self.log: list[dict] = []                          # every message: {t, kind, session_id, seq?, reply}
        self.events: list[dict] = []
        self.frames_fresh, self.debug_fresh = frames_fresh, debug_fresh
        self.watchdog_default_s = watchdog_default_s
        self.cur: dict | None = None                       # the session that owns the channel
        self.ended: set[str] = set()
        self.hold: dict | None = None                      # {mode, upper, hands} after a session ended
        self.halt_epoch: int | None = None
        self.sent_upper = STAND_WIRE.copy()                # what the "mux" sends, wire order
        self.sent_hands = {s: np.zeros(N_HAND) for s in jo.SIDES}
        self.q29 = np.asarray(jo.STAND_Q29, dtype=np.float64).copy()
        self.hand_q = {s: np.zeros(N_HAND) for s in jo.SIDES}
        self.tick_hooks: list[Callable[["FakeArmBody", float], None]] = []
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="fake-arm-body", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ ports
    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        self.subs.append(cb)
        return lambda: self.subs.remove(cb) if cb in self.subs else None

    def on_tick(self, fn: Callable[["FakeArmBody", float], None]) -> None:
        self.tick_hooks.append(fn)

    def ego_frame(self) -> tuple[np.ndarray | None, float]:
        now = time.monotonic()
        return np.full((480, 640, 3), 90, dtype=np.uint8), now if self.frames_fresh else now - 5.0

    def debug_state(self) -> tuple[dict | None, float]:
        now = time.monotonic()
        with self.lock:
            d = {"body_q": self.q29.tolist(), "left_hand_q": self.hand_q["left"].tolist(),
                 "right_hand_q": self.hand_q["right"].tolist(), "body_q_target": self.q29.tolist()}
        return d, (now - 0.005) if self.debug_fresh else now - 5.0

    def chunk_times(self, session_id: str | None = None) -> list[float]:
        return [m["t"] for m in self.log if m["kind"] == "chunk" and m["reply"].get("ok")
                and (session_id is None or m["session_id"] == session_id)]

    def messages(self, kind: str | None = None, session_id: str | None = None) -> list[dict]:
        return [m for m in self.log if (kind is None or m["kind"] == kind)
                and (session_id is None or m["session_id"] == session_id)]

    # ------------------------------------------------------------------ the op
    def arm(self, args: dict, op_id: str | None = None) -> dict:
        now = time.monotonic()
        kind = ("end" if args.get("end") else "keepalive" if args.get("keepalive") else
                "chunk" if "chunk" in args else "start" if args.get("release") is None else "release")
        ch = args.get("chunk") or {}
        meta = {k: ch.get(k) for k in ("seq", "t0_mono", "dt", "order", "inference_ms")}
        meta.update({k: np.shape(ch[k]) for k in ("upper_body", "left_hand", "right_hand") if k in ch})
        with self.lock:
            rep, evs = self._arm(dict(args), op_id or f"arm-{uuid.uuid4().hex[:8]}", now)
            self.log.append({"t": now, "kind": kind, "session_id": args.get("session_id"), "seq": ch.get("seq"),
                             "op_id": op_id, "reply": rep, "chunk": meta if ch else None,
                             "args": {k: v for k, v in args.items() if k != "chunk"}})
        for ev in evs:
            self._publish(ev)
        return rep

    @staticmethod
    def _rej(err: str, **data: Any) -> dict:
        return {"ok": False, "state": "rejected", "error": err, "data": data}

    def _arm(self, a: dict, op_id: str, now: float) -> tuple[dict, list[dict]]:
        if a.get("mode") != "chunk":
            return self._rej("bad_args", error="this fake implements chunk mode only"), []
        if self.halt_epoch is not None and int(a.get("control_epoch", -1)) <= self.halt_epoch:
            return self._rej("halted", halt_epoch=self.halt_epoch), []
        sid, stream = a.get("session_id"), a.get("stream")
        cur = self.cur
        if cur is not None and cur["stream"] == stream:
            if sid != cur["session_id"] or int(a.get("generation", 0)) < cur["generation"] \
                    or int(a.get("control_epoch", 0)) < cur["control_epoch"]:
                return self._rej("stale_session", owner=cur["session_id"]), []
        elif sid in self.ended:
            return self._rej("stale_session", ended=True), []
        elif cur is not None and not a.get("preempt"):
            return self._rej("arm_busy", owner=cur["stream"]), []
        evs: list[dict] = []
        if cur is None or cur["stream"] != stream:
            if a.get("end") or a.get("keepalive"):
                return self._rej("not_owner"), []
            if cur is not None:                                   # preempted
                evs.append(self._terminal(cur, "canceled", "preempted", "none"))
            cur = self.cur = self._new(a, op_id, now)
            self.hold = None
            evs.append(self._ev(cur, "accepted", {"stream": stream, "session_id": sid}))
            rep_state = "accepted"
        else:
            rep_state = "done"
        cur["t_msg"] = now
        if a.get("end"):
            hold = str(a.get("hold_on_end") or cur["hold_on_end"])
            evs.append(self._end(cur, "succeeded", "client", hold, a.get("reason")))
            return {"ok": True, "state": "done", "data": {"id": cur["op_id"], "hold": hold}}, evs
        if a.get("keepalive"):
            return {"ok": True, "state": rep_state, "data": {"id": cur["op_id"]}}, evs
        data: dict[str, Any] = {"id": cur["op_id"]}
        ch = a.get("chunk")
        if ch is not None:
            why = self._apply(cur, ch, now)
            if why in ("bad_chunk",):
                cur["dropped"]["invalid"] += 1
                return self._rej("bad_chunk"), evs
            if why:
                data["dropped"] = why
        data["session"] = {"session_id": cur["session_id"], "chunk_seq": cur["chunk_seq"], "stall_s": cur["stall_s"]}
        return {"ok": True, "state": rep_state, "data": data}, evs

    def _new(self, a: dict, op_id: str, now: float) -> dict:
        hands = {s: a.get(f"{s}_hand") for s in jo.SIDES}
        return {"stream": a.get("stream"), "session_id": a.get("session_id"), "op_id": op_id,
                "generation": int(a.get("generation", 0)), "control_epoch": int(a.get("control_epoch", 0)),
                "hold_on_end": str(a.get("hold_on_end") or "measured"),
                "watchdog_s": float(a.get("watchdog_s") or self.watchdog_default_s),
                "lead_s": float(a.get("lead_s") or 0.0), "xfade": int(a.get("xfade_ticks") or 5),
                "start_hands": {s: (np.asarray(h, float) if h is not None else None) for s, h in hands.items()},
                "chunk": None, "old": None, "xfade_i": 0, "chunk_seq": 0, "t_msg": now, "t_start": now,
                "stall_s": 0.0, "stall_max": 0.0, "cross_fades": 0, "received": 0, "applied": 0,
                "dropped": {"expired": 0, "out_of_order": 0, "invalid": 0, "stale_session": 0, "halted": 0},
                "clamp_win": [], "clamped_total": 0, "values_total": 0, "latency_ms": None, "inference_ms": None,
                "t_prog": now}

    def _apply(self, cur: dict, ch: dict, now: float) -> str | None:
        cur["received"] += 1
        try:
            ub = np.asarray(ch["upper_body"], dtype=np.float64)
            lh = np.asarray(ch["left_hand"], dtype=np.float64)
            rh = np.asarray(ch["right_hand"], dtype=np.float64)
            T = ub.shape[0]
            dt, t0, seq = float(ch["dt"]), float(ch["t0_mono"]), int(ch["seq"])
            order = ch["order"]
        except (KeyError, TypeError, ValueError, IndexError):
            return "bad_chunk"
        if order not in ("wire", "mj17") or not (1 <= T <= 64) or ub.shape != (T, N_UPPER) \
                or lh.shape != (T, N_HAND) or rh.shape != (T, N_HAND) or not (0.005 <= dt <= 0.1) \
                or not (np.isfinite(ub).all() and np.isfinite(lh).all() and np.isfinite(rh).all()):
            return "bad_chunk"
        if order == "mj17":
            ub = jo.mj17_to_wire(ub)
        old = cur["chunk"]
        if seq <= cur["chunk_seq"] or (old is not None and t0 < old["t0"]):
            cur["dropped"]["out_of_order"] += 1
            return "out_of_order"
        if round((now + cur["lead_s"] - t0) / dt) >= T:
            cur["dropped"]["expired"] += 1
            return "expired"
        cur["old"], cur["xfade_i"] = old, 0
        if old is not None:
            cur["cross_fades"] += 1
        cur["chunk"] = {"ub": ub, "hands": {"left": lh, "right": rh}, "t0": t0, "dt": dt, "T": T}
        cur["chunk_seq"] = seq
        cur["applied"] += 1
        cur["latency_ms"] = round((now - t0) * 1000.0, 1)
        cur["inference_ms"] = ch.get("inference_ms")
        return None

    @staticmethod
    def _row(c: dict, now: float, lead: float) -> tuple[np.ndarray, dict, int, float]:
        k = int(round((now + lead - c["t0"]) / c["dt"]))
        kk = min(max(k, 0), c["T"] - 1)
        stall = max(0.0, now + lead - (c["t0"] + (c["T"] - 1) * c["dt"]))
        return c["ub"][kk], {s: c["hands"][s][kk] for s in jo.SIDES}, kk, stall

    # ------------------------------------------------------------------ 50 Hz
    def _loop(self) -> None:
        period = 1.0 / HZ
        nxt = time.monotonic()
        while self._running:
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
            now = time.monotonic()
            evs: list[dict] = []
            with self.lock:
                self._tick(now, evs)
            for ev in evs:
                self._publish(ev)
            for fn in list(self.tick_hooks):
                try:
                    fn(self, now)
                except Exception:  # noqa: BLE001
                    pass

    def _tick(self, now: float, evs: list[dict]) -> None:
        cur = self.cur
        if cur is not None:
            if now - cur["t_msg"] > cur["watchdog_s"]:
                evs.append(self._end(cur, "succeeded", "watchdog", cur["hold_on_end"], None))
            else:
                c = cur["chunk"]
                if c is None:
                    up = STAND_WIRE.copy()
                    a = min(1.0, (now - cur["t_start"]) / 0.3)
                    hands = {s: (a * h + (1 - a) * self.sent_hands[s]) if h is not None else self.sent_hands[s]
                             for s, h in cur["start_hands"].items()}
                else:
                    up, hands, k, stall = self._row(c, now, cur["lead_s"])
                    cur["stall_s"] = round(stall, 3)
                    cur["stall_max"] = max(cur["stall_max"], stall)
                    if cur["old"] is not None and cur["xfade_i"] < cur["xfade"]:
                        w = (cur["xfade_i"] + 1) / cur["xfade"]
                        oup, ohands, _, _ = self._row(cur["old"], now, cur["lead_s"])
                        up = (1 - w) * oup + w * up
                        hands = {s: (1 - w) * ohands[s] + w * hands[s] for s in jo.SIDES}
                        cur["xfade_i"] += 1
                    up = np.array(up, dtype=np.float64)
                    up[[0, 1, 2]] = STAND_WIRE[[0, 1, 2]]          # waist: "ref"
                    cl = np.clip(up, _UB_LO, _UB_HI)
                    n = int(np.count_nonzero(cl[ARM_WIRE_IDX] != up[ARM_WIRE_IDX]))
                    hc = {}
                    for s in jo.SIDES:
                        lo, hi = _HAND_LIM[s]
                        hc[s] = np.clip(hands[s], lo, hi)
                        n += int(np.count_nonzero(hc[s] != hands[s]))
                    up, hands = cl, hc
                    cur["clamp_win"].append(n)
                    cur["clamped_total"] += n
                    cur["values_total"] += 28
                self.sent_upper = np.asarray(up, dtype=np.float64)
                self.sent_hands = {s: np.asarray(hands[s], dtype=np.float64) for s in jo.SIDES}
                if now - cur["t_prog"] >= 1.0 / 5.0:
                    cur["t_prog"] = now
                    evs.append(self._progress(cur))
        elif self.hold is not None:
            self.sent_upper, self.sent_hands = self.hold["upper"], self.hold["hands"]
        # the "robot": the measured state follows what is sent (first-order lag, tau 0.1 s)
        a = 1.0 - math.exp(-(1.0 / HZ) / 0.1)
        for i, m in enumerate(jo.SONIC_WIRE_FROM_MUJOCO):
            self.q29[m] += a * (self.sent_upper[i] - self.q29[m])
        for s in jo.SIDES:
            self.hand_q[s] += a * (self.sent_hands[s] - self.hand_q[s])

    def _progress(self, cur: dict) -> dict:
        win = cur["clamp_win"]
        frac = (sum(win) / (28.0 * len(win))) if win else 0.0
        cur["clamp_win"] = []
        c = cur["chunk"]
        k = None if c is None else int(min(max(round((time.monotonic() + cur["lead_s"] - c["t0"]) / c["dt"]), 0),
                                           c["T"] - 1))
        return self._ev(cur, "progress", {
            "kind": "chunk", "session_id": cur["session_id"], "stream": cur["stream"], "chunk_seq": cur["chunk_seq"],
            "k": k, "T": None if c is None else c["T"], "stall_s": cur["stall_s"], "clamped_frac": round(frac, 4),
            "clamped_frac_total": round(cur["clamped_total"] / cur["values_total"], 4) if cur["values_total"] else 0.0,
            "latency_ms": cur["latency_ms"], "inference_ms": cur["inference_ms"], "lead_s": cur["lead_s"],
            "cross_fades": cur["cross_fades"],
            "chunks": {"received": cur["received"], "applied": cur["applied"], "dropped": dict(cur["dropped"])}})

    # ------------------------------------------------------------------ ends
    def _terminal(self, cur: dict, state: str, by: str, hold: str, reason: str | None = None) -> dict:
        self.ended.add(cur["session_id"])
        return self._ev(cur, state, {
            "session_id": cur["session_id"], "stream": cur["stream"], "ended_by": by, "hold": hold, "reason": reason,
            "chunks": {"received": cur["received"], "applied": cur["applied"], "dropped": dict(cur["dropped"])},
            "clamped_frac_total": round(cur["clamped_total"] / cur["values_total"], 4) if cur["values_total"] else 0.0,
            "stall_s_max": round(cur["stall_max"], 3), "duration_s": round(time.monotonic() - cur["t_start"], 3)})

    def _end(self, cur: dict, state: str, by: str, hold: str, reason: str | None) -> dict:
        ev = self._terminal(cur, state, by, hold, reason)
        if hold == "target":
            self.hold = {"mode": hold, "upper": self.sent_upper.copy(),
                         "hands": {s: self.sent_hands[s].copy() for s in jo.SIDES}}
        elif hold == "measured":
            self.hold = {"mode": hold, "upper": np.asarray([self.q29[i] for i in jo.SONIC_WIRE_FROM_MUJOCO]),
                         "hands": {s: self.sent_hands[s].copy() for s in jo.SIDES}}   # hands never opened
        else:
            self.hold = {"mode": "stand", "upper": STAND_WIRE.copy(),
                         "hands": {s: self.sent_hands[s].copy() for s in jo.SIDES}}
        self.cur = None
        return ev

    def halt(self, epoch: int) -> None:
        """The B.1 latch: drop the timeline, hold the measured upper body, never open the hands."""
        with self.lock:
            self.halt_epoch = int(epoch)
            ev = self._end(self.cur, "canceled", "halt", "measured", "halted") if self.cur is not None else None
        if ev is not None:
            self._publish(ev)

    def resume(self) -> None:
        with self.lock:
            self.halt_epoch = None

    def forget_session(self) -> None:
        """As if wl-body restarted: the session is gone but not marked ended (its next message is stale)."""
        with self.lock:
            if self.cur is not None:
                self.ended.add(self.cur["session_id"])
                self.cur = None

    # ------------------------------------------------------------------ events
    def _ev(self, cur: dict, state: str, data: dict) -> dict:
        self._seq += 1
        return {"id": cur["op_id"], "op": "arm", "state": state, "data": data, "seq": self._seq,
                "t_wall": time.time()}

    def _publish(self, ev: dict) -> None:
        self.events.append(ev)
        for cb in list(self.subs):
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
