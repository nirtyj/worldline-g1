"""B.2/B.3 live checks (docs/contracts/m1.md §3.11-§3.12) on a standing stack, through BodyClient only.

    .venv/bin/python -m tools.modes_test [--out outputs/body_wave/modes-<ts>] [--no-recover] [--port-offset N]

1. modes: walk 2 s, an arm stream 2 s then `end`, an approach of 0.2 m; records every body.mode event and body.state
   (mode, speed, pose_source) and checks the sequence HOLD -> LOCOMOTION -> HOLD -> ARM_STREAM -> HOLD ->
   LOCOMOTION -> HOLD.
2. leases: acquire by A; a walk from B and an unfenced walk are body_busy; a stale generation is dropped and
   published as body.stale_command (and nothing moves, gt.pose); release.
3. runtime session: hello (watchdog 1.0 s, no heartbeat), a 6 s walk, no pings: the body halts itself
   (body.halted{reason: runtime_lost}); reports last ping -> body.halted, the walk's terminal event, the time to rest
   (M1 E3 criterion); then resume.
4. recover (skipped with --no-recover): `recover {force: true}` while standing (PLAN §7.3.3 path A without a fall:
   band on, P1 reset_robot in place, stabilize, band release, watch); reports the steps, pelvis_z and 0 falls.
5. --fall: a SYNTHETIC fall (test harness only): P1 `band {on, z: 0.45}` pulls the pelvis down below the fall band;
   the body must detect it (body.fault{fell}), re-engage the band at 0.80 m, go to FAULT, reject motions
   (`fault:fallen`); then `recover` (not forced) must bring it back to HOLD and a short walk must succeed.
   `falls` in the summary then counts that one intended fall.
Writes summary.json, topics.jsonl, raw.npz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from body.client import BodyClient
from body.config import port_offset_from_env, ports as _ports

from .halt_test import Tap, rest_time_m1


def fall_step(bc: BodyClient, tap: Tap, topics: list, log) -> dict:
    """Step 5: the synthetic fall (band pulled down to 0.45 m) -> body.fault{fell} -> band -> FAULT -> recover."""
    from body.config import ep
    from body.p1_client import P1Rpc
    p1 = P1Rpc(ep(bc.ports["p1_rep"], bc.host), timeout_s=5.0)
    t0 = time.monotonic()
    rep = p1.call("band", on=True, z=0.45)
    t_fault = None
    while time.monotonic() - t0 < 6.0 and t_fault is None:
        f = [tt for tt, top, m in topics if top == "body.fault" and tt >= t0 and m.get("kind") == "fell"]
        t_fault = f[0] if f else None
        time.sleep(0.02)
    w = tap.window("pose", t0, time.monotonic())
    zmin = min((r[6] for r in w), default=None)
    time.sleep(2.5)
    band_after = (p1.try_call("ping") or {}).get("band")
    st = bc.status()
    hw = bc.walk(vx=0.3, duration_s=1.0)
    w2 = tap.window("pose", t0, time.monotonic())
    z_after = w2[-1][6] if w2 else None
    tr = time.monotonic()
    hr = bc.recover(timeout=90)
    st2 = bc.status()
    hw2 = bc.walk(vx=0.3, duration_s=1.0) if hr.ok else None
    modes = [m["mode"] for tt, top, m in topics if top == "body.mode" and tt >= t0]
    out = {"pull_down": rep.get("ok"), "pelvis_z_min": None if zmin is None else round(zmin, 4),
           "t_fault_s": None if t_fault is None else round(t_fault - t0, 3), "band_after_fault": band_after,
           "pelvis_z_held_by_band": None if z_after is None else round(z_after, 4),
           "mode_after_fault": st.get("mode"), "fault": st.get("fault"), "walk_while_faulted": hw.reason,
           "recover": {"state": hr.state, "reason": hr.reason, "steps": (hr.result or {}).get("steps"),
                       "duration_s": round(time.monotonic() - tr, 2)},
           "mode_after_recover": st2.get("mode"), "fault_after_recover": st2.get("fault"),
           "walk_after_recover": None if hw2 is None else hw2.state, "modes": modes}
    out["pass"] = bool(t_fault is not None and band_after and st.get("mode") == "FAULT" and
                       hw.reason == "fault:fallen" and hr.ok and st2.get("mode") == "HOLD" and
                       hw2 is not None and hw2.ok)
    p1.close()
    log(f"fall: {out}")
    return {"fall": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--no-recover", action="store_true")
    ap.add_argument("--fall", action="store_true", help="also run the synthetic fall + recover (step 5)")
    ap.add_argument("--only-fall", action="store_true", help="run step 5 only")
    ap.add_argument("--out", default=f"outputs/body_wave/modes-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args(argv)
    off = a.port_offset if a.port_offset is not None else port_offset_from_env()
    os.makedirs(a.out, exist_ok=True)
    bc = BodyClient(port_offset=off).connect(15)
    tap = Tap(_ports(off))
    topics = []
    bc.add_topic_listener(lambda t, m: topics.append((time.monotonic(), t, m)))
    states = []
    t0 = time.monotonic()
    while tap.last("pose") is None and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    st = bc.status()
    if not st.get("in_control") or st.get("fault") or st.get("latched"):
        print(f"[modes_test] not ready: {st.get('mode')} fault {st.get('fault')} latched {st.get('latched')}")
        return 2
    out: dict = {"start": {"mode": st["mode"], "pose": st["pose"]}}

    def log(msg):
        print(f"[modes_test {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def sample(dur):
        t_end = time.monotonic() + dur
        while time.monotonic() < t_end:
            s = bc.last_state or {}
            states.append((time.monotonic(), s.get("mode"), s.get("speed"), s.get("pose_source")))
            time.sleep(0.1)

    def modes_since(t):
        return [m["mode"] for tt, top, m in topics if top == "body.mode" and tt >= t]

    try:
        if a.only_fall:
            out.update(fall_step(bc, tap, topics, log))
            out["command_stop_sent"] = tap.stop_count()
            out["all_pass"] = out["fall"]["pass"] and out["command_stop_sent"] == 0
            return 0 if out["all_pass"] else 1
        # 1. modes --------------------------------------------------------------------------------
        t1 = time.monotonic()
        h = bc.walk(vx=0.4, duration_s=2.0, wait=False)
        sample(1.5)
        h.wait(20)
        sample(0.5)
        arm = bc.arm_stream()
        ts = time.monotonic()
        while time.monotonic() - ts < 2.0:
            arm.send(upper_body={"right_elbow_joint": 1.0 + 0.2 * math.sin(time.monotonic() - ts)})
            time.sleep(0.02)
        sample(0.2)
        arm.end()
        if arm.handle is not None:
            arm.handle.wait(6)
        sample(0.5)
        p = tap.last("pose")
        ha = bc.approach(p[1] + 0.2 * math.cos(p[3]), p[2] + 0.2 * math.sin(p[3]), yaw=p[3], wait=True, timeout=60)
        sample(0.5)
        seq = modes_since(t1)
        want = ["LOCOMOTION", "HOLD", "ARM_STREAM", "HOLD", "LOCOMOTION", "HOLD"]
        ok_seq = seq == want
        loco_speed = [s[2] for s in states if s[1] == "LOCOMOTION" and s[2] is not None]
        out["modes"] = {"sequence": seq, "expected": want, "pass": ok_seq, "walk": h.state, "arm": arm.handle.state
                        if arm.handle else None, "approach": {"state": ha.state, "pos_err": (ha.result or {}).get(
                            "pos_err"), "attempts": (ha.result or {}).get("attempts")},
                        "state_speed_max_locomotion": None if not loco_speed else round(max(loco_speed), 3),
                        "pose_source": sorted({s[3] for s in states if s[3]})}
        log(f"modes: {seq} (pass {ok_seq}), speed max in LOCOMOTION {out['modes']['state_speed_max_locomotion']}")

        # 2. leases -------------------------------------------------------------------------------
        tl = time.monotonic()
        r_a = bc.acquire("modes-A", 5, 1, mode="LOCOMOTION")
        hb = bc.walk(vx=0.4, duration_s=1.0, execution_id="modes-B", generation=5, control_epoch=1)
        hu = bc.walk(vx=0.4, duration_s=1.0)
        p0 = tap.last("pose")
        hs = bc.walk(vx=0.4, duration_s=1.0, execution_id="modes-old", generation=4, control_epoch=1)
        time.sleep(0.8)
        p1 = tap.last("pose")
        stale_pub = [m for tt, top, m in topics if top == "body.stale_command" and tt >= tl]
        r_rel = bc.release("modes-A")
        out["leases"] = {"acquire": r_a.get("ok"), "other_owner": hb.reason, "unfenced": hu.reason, "stale": hs.reason,
                         "stale_published": len(stale_pub), "moved_m": round(math.hypot(p1[1] - p0[1], p1[2] - p0[2]), 4),
                         "release": r_rel.get("ok")}
        out["leases"]["pass"] = (out["leases"]["acquire"] and hb.reason == "body_busy" and hu.reason == "body_busy"
                                 and hs.reason == "stale_command" and len(stale_pub) >= 1 and
                                 out["leases"]["moved_m"] < 0.03 and out["leases"]["release"])
        log(f"leases: {out['leases']}")

        # 3. runtime session watchdog ------------------------------------------------------------------
        rep = bc.hello("modes-rt", watchdog_s=1.0, heartbeat_s=None)
        t_ping = time.monotonic()
        bc.ping_session()
        t_last_ping = time.monotonic()
        hw = bc.walk(vx=0.4, duration_s=6.0, wait=False)
        try:
            hw.wait(8)
        except TimeoutError:
            pass
        halted = [(tt, m) for tt, top, m in topics if top == "body.halted" and tt >= t_ping and
                  m.get("reason") == "runtime_lost"]
        lost = [(tt, m) for tt, top, m in topics if top == "body.session" and tt >= t_ping and m.get("event") == "lost"]
        time.sleep(1.5)
        t_h = halted[0][0] if halted else None
        rest = None if t_h is None else rest_time_m1(tap.window("pose", t_h - 0.2, t_h + 3.0), t_h)
        sth = bc.status()
        out["session"] = {"hello": rep.get("ok"), "walk": hw.state, "walk_reason": hw.reason,
                          "walk_halt_reason": (hw.result or {}).get("halt_reason"),
                          "last_ping_to_lost_s": None if not lost else round(lost[0][0] - t_last_ping, 3),
                          "last_ping_to_halted_s": None if t_h is None else round(t_h - t_last_ping, 3),
                          "halted_epoch": None if not halted else halted[0][1].get("epoch"),
                          "rest_after_halt_s": rest, "latched": sth.get("latched"), "mode": sth.get("mode")}
        rr = bc.resume(sth.get("halt_epoch"))
        bc.bye()
        out["session"]["resume"] = rr.get("ok")
        out["session"]["pass"] = (hw.state == "canceled" and (hw.result or {}).get("halt_reason") == "runtime_lost"
                                  and t_h is not None and 0.9 <= t_h - t_last_ping <= 1.3 and rest is not None
                                  and rest <= 1.5 and rr.get("ok"))
        log(f"session: {out['session']}")

        # 4. recover (forced, no fall) ------------------------------------------------------------------
        if not a.no_recover:
            tr = time.monotonic()
            hr = bc.recover(force=True, timeout=90)
            w = tap.window("pose", tr, time.monotonic())
            zmin = min((r[6] for r in w), default=None)
            out["recover"] = {"state": hr.state, "reason": hr.reason, "steps": (hr.result or {}).get("steps"),
                              "pelvis_z_min": None if zmin is None else round(zmin, 4),
                              "fallen": any(r[7] for r in w), "duration_s": round(time.monotonic() - tr, 2),
                              "mode_after": bc.status().get("mode")}
            out["recover"]["pass"] = hr.ok and not out["recover"]["fallen"] and out["recover"]["mode_after"] == "HOLD"
            log(f"recover: {out['recover']}")
        out["command_stop_sent"] = tap.stop_count()
        out["falls"] = sum(1 for r in tap.pose if r[7])
        out["all_pass"] = all(v.get("pass", True) for v in out.values() if isinstance(v, dict)) and \
            out["command_stop_sent"] == 0 and out["falls"] == 0
        if a.fall:
            out.update(fall_step(bc, tap, topics, log))
            out["command_stop_sent"] = tap.stop_count()
            out["all_pass"] = out["all_pass"] and out["fall"]["pass"] and out["command_stop_sent"] == 0
    finally:
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)
        with open(os.path.join(a.out, "topics.jsonl"), "w") as f:
            for tt, top, m in topics:
                f.write(json.dumps({"t_mono": tt, "topic": top, **m}, default=str) + "\n")
        pose = np.array(list(tap.pose), dtype=float) if tap.pose else np.zeros((0, 8))
        np.savez_compressed(os.path.join(a.out, "raw.npz"), pose=pose)
        tap.close()
        bc.close()
    log(f"summary: {json.dumps(out, default=str)}")
    return 0 if out.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
