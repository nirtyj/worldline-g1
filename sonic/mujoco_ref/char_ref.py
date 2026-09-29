"""Characterise the SONIC planner's response to zmq_manager planner commands in the MuJoCo reference loop.

Same plumbing and hand-over as drive_ref.py (messages built with gear_sonic's build_planner_message,
gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:30-158, WBC @ b042411). Answers what the body layer (P3) needs:

  A. open-loop in-place turns: mode IDLE + new facing (keyboard Q/E do this, keyboard_handler.hpp:556-566) for
     +-45/+-90/+-135 deg. Reports the achieved yaw change (ground truth), the planner's own target yaw change
     (g1_debug.base_quat_target, zmq_output_handler.hpp:18-75) and the residual error.
  B. closed-loop in-place turn: same command, then push the facing command further by the residual ground-truth
     error once the robot has settled (what body/motions.py TurnToMotion could do). Reports final error and time.
  C. in-place turn with mode SLOW_WALK and zero movement.
  D. walk start/stop cycles (SLOW_WALK 0.3-0.7 m/s -> IDLE): falls, distance, stop time per cycle.
  E. curved walk: SLOW_WALK with movement = facing rotating at a constant yaw rate (pure-pursuit-like).

Run through run_ref_loop.sh --driver char_ref.py. Writes drive_result.json like drive_ref.py, plus char.json.
"""

from __future__ import annotations

import json
import math
import time

from drive_ref import IDLE, SLOW_WALK, Driver, base_args, wrap


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Char(Driver):
    def __init__(self, a):
        super().__init__(a)
        self.char: dict = {"turn_open_loop": [], "turn_closed_loop": [], "turn_slowwalk": [], "walk_cycles": [], "curve": [],
                           "gentle": {"turns": [], "cycles": [], "curves": [], "stand": None}, "body_turn": []}

    # ------------------------------------------------------------------ helpers
    def planner_target_yaw(self):
        d = self.debug_last
        if not d or "base_quat_target" not in d:
            return None
        return quat_yaw(d["base_quat_target"])

    def facing_cmd(self, yaw_world: float):
        return self.world_to_planner([math.cos(yaw_world), math.sin(yaw_world)])

    def watch(self, secs: float):
        """Sample the pose for secs; returns (min pelvis z, falls delta, samples)."""
        f0 = self.pose()["falls"]
        zs, samples = [], []
        t_end = time.time() + secs
        while time.time() < t_end:
            p = self.pose()
            zs.append(p["pelvis_z"])
            samples.append((time.time(), p["yaw"], p["base_pos"][:2]))
            time.sleep(0.05)
        return min(zs), self.pose()["falls"] - f0, samples

    def settled(self, secs=0.5, wz_eps=0.08, v_eps=0.08, timeout=6.0):
        """Wait until |yaw rate| < wz_eps and |v_xy| < v_eps for secs; returns the time waited."""
        t0 = time.time()
        t_ok = None
        while time.time() - t0 < timeout:
            p = self.pose()
            wz = abs(p["base_ang_vel_b"][2])
            v = math.hypot(*p["base_lin_vel_w"][:2])
            if wz < wz_eps and v < v_eps:
                t_ok = t_ok or time.time()
                if time.time() - t_ok >= secs:
                    return time.time() - t0
            else:
                t_ok = None
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------ A
    def turn_open_loop(self, delta_deg: float, secs: float = 6.5, mode=IDLE):
        p0 = self.pose()
        y0 = p0["yaw"]
        ty0 = self.planner_target_yaw()
        tgt = wrap(y0 + math.radians(delta_deg))
        mv = [0.0, 0.0, 0.0]
        self.set_cmd(mode=mode, movement=mv, facing=self.facing_cmd(tgt), speed=-1.0)
        zmin, nf, samples = self.watch(secs)
        p1 = self.pose()
        ty1 = self.planner_target_yaw()
        achieved = math.degrees(wrap(p1["yaw"] - y0))
        # time to reach 90 % of the final change
        t90 = next((t - samples[0][0] for t, yw, _ in samples if abs(math.degrees(wrap(yw - y0))) >= 0.9 * abs(achieved)), None)
        rec = {"mode": "IDLE" if mode == IDLE else "SLOW_WALK", "cmd_deg": delta_deg, "achieved_deg": round(achieved, 1),
               "planner_target_deg": None if ty0 is None or ty1 is None else round(math.degrees(wrap(ty1 - ty0)), 1),
               "residual_err_deg": round(delta_deg - achieved, 1), "t90_s": None if t90 is None else round(t90, 2),
               "translation_m": round(math.hypot(p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]), 3),
               "pelvis_z_min": round(zmin, 3), "falls": nf}
        self.log("turn_result", **rec)
        return rec

    # ------------------------------------------------------------------ B
    def turn_closed_loop(self, delta_deg: float, tol_deg: float = 3.0, max_iter: int = 4, timeout: float = 15.0):
        p0 = self.pose()
        y0 = p0["yaw"]
        f0 = p0["falls"]
        tgt = wrap(y0 + math.radians(delta_deg))
        cmd = tgt
        t0 = time.time()
        iters = []
        zmin = 9.0
        for it in range(max_iter + 1):
            self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(cmd), speed=-1.0)
            time.sleep(1.0)
            waited = self.settled(timeout=max(0.5, timeout - (time.time() - t0)))
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            err = wrap(tgt - p["yaw"])
            iters.append({"iter": it, "cmd_offset_deg": round(math.degrees(wrap(cmd - tgt)), 1), "err_deg": round(math.degrees(err), 2),
                          "settle_wait_s": None if waited is None else round(waited, 2)})
            if abs(math.degrees(err)) <= tol_deg or time.time() - t0 > timeout:
                break
            cmd = wrap(cmd + err)  # push the facing command past the target by the residual error
        p1 = self.pose()
        rec = {"cmd_deg": delta_deg, "final_err_deg": round(math.degrees(wrap(tgt - p1["yaw"])), 2), "time_s": round(time.time() - t0, 2),
               "iterations": iters, "translation_m": round(math.hypot(p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]), 3),
               "pelvis_z_min": round(zmin, 3), "falls": p1["falls"] - f0}
        # leave the facing command at the ACHIEVED yaw so the next test starts from a consistent state
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(p1["yaw"]), speed=-1.0)
        self.log("closed_loop_turn_result", **rec)
        return rec

    # ------------------------------------------------------------------ G (gentle envelope)
    def turn_ramp(self, delta_deg: float, rate_dps: float = 30.0, tol_deg: float = 3.0, max_iter: int = 4, timeout: float = 20.0):
        """IDLE turn with the facing command ramped at rate_dps (like repeated keyboard Q/E, +-30 deg per press), then
        closed-loop correction on ground-truth yaw with gain 0.6 on the residual."""
        p0 = self.pose()
        y0, f0 = p0["yaw"], p0["falls"]
        tgt = wrap(y0 + math.radians(delta_deg))
        t0 = time.time()
        zmin = 9.0
        dur = abs(delta_deg) / rate_dps
        while (t := time.time() - t0) < dur:
            f = y0 + math.copysign(math.radians(rate_dps) * t, delta_deg)
            self.set_cmd_quiet(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(f), speed=-1.0)
            zmin = min(zmin, self.pose()["pelvis_z"])
            time.sleep(0.1)
        cmd = tgt
        iters = []
        for it in range(max_iter + 1):
            self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(cmd), speed=-1.0)
            time.sleep(0.8)
            waited = self.settled(timeout=max(0.5, timeout - (time.time() - t0)))
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            err = wrap(tgt - p["yaw"])
            iters.append({"iter": it, "cmd_offset_deg": round(math.degrees(wrap(cmd - tgt)), 1), "err_deg": round(math.degrees(err), 2),
                          "settle_wait_s": None if waited is None else round(waited, 2)})
            if abs(math.degrees(err)) <= tol_deg or time.time() - t0 > timeout:
                break
            cmd = wrap(cmd + 0.6 * err)
        p1 = self.pose()
        rec = {"cmd_deg": delta_deg, "rate_dps": rate_dps, "final_err_deg": round(math.degrees(wrap(tgt - p1["yaw"])), 2),
               "time_s": round(time.time() - t0, 2), "iterations": iters,
               "translation_m": round(math.hypot(p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]), 3),
               "pelvis_z_min": round(zmin, 3), "falls": p1["falls"] - f0}
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(p1["yaw"]), speed=-1.0)
        self.log("ramp_turn_result", **rec)
        return rec

    # ------------------------------------------------------------------ H: wl-body TurnToMotion emulation
    def turn_body_style(self, delta_deg: float, tol_deg: float = 6.0, timeout: float = 20.0, push: float = 0.0):
        """Emulates body/motions.py TurnToMotion + Motion.turn_cmd (commit 9185dd7): the facing command advances in
        <= 30 deg steps from the last COMMANDED facing, only once the robot is within 15 deg of it; the final command is
        the target; success needs |err| < tol_deg (cfg.yaw_tol_deg = 6). push > 0 adds the residual correction
        measured in section B/G (once settled, move the command past the target by push * residual)."""
        STEP, CATCH = math.radians(30.0), math.radians(15.0)
        p0 = self.pose()
        tgt = wrap(p0["yaw"] + math.radians(delta_deg))
        prev = p0["yaw"]
        t0 = time.time()
        t_in_tol = None
        offset = 0.0
        t_settle_ok = None
        zmin = 9.0
        errs = []
        while time.time() - t0 < timeout:
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            yaw = p["yaw"]
            err = wrap(tgt - yaw)
            errs.append((round(time.time() - t0, 2), round(math.degrees(err), 2)))
            if abs(err) < math.radians(tol_deg):
                t_in_tol = t_in_tol or time.time() - t0
                break
            want = wrap(tgt + offset)
            d = wrap(want - prev)
            cmd = want
            if abs(d) > STEP:
                cmd = wrap(prev + math.copysign(STEP, d)) if abs(wrap(prev - yaw)) < CATCH else prev
            prev = cmd
            self.set_cmd_quiet(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(cmd), speed=-1.0)
            # residual push: once the final command is the target and the robot has settled short of it
            if push > 0 and abs(wrap(cmd - want)) < 1e-6:
                wz = abs(p["base_ang_vel_b"][2])
                if wz < 0.08:
                    t_settle_ok = t_settle_ok or time.time()
                    if time.time() - t_settle_ok > 0.5:
                        offset = wrap(offset + push * err)
                        t_settle_ok = None
                else:
                    t_settle_ok = None
            time.sleep(0.1)
        p1 = self.pose()
        rec = {"cmd_deg": delta_deg, "push": push, "tol_deg": tol_deg, "reached_tol": t_in_tol is not None,
               "t_reach_s": None if t_in_tol is None else round(t_in_tol, 2),
               "final_err_deg": round(math.degrees(wrap(tgt - p1["yaw"])), 2), "offset_deg": round(math.degrees(offset), 1),
               "err_trace_1s": [e for e in errs if abs(e[0] - round(e[0])) < 0.05][:25],
               "pelvis_z_min": round(zmin, 3), "falls": p1["falls"] - p0["falls"]}
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(p1["yaw"]), speed=-1.0)
        time.sleep(1.5)
        self.log("body_turn_result", **{k: v for k, v in rec.items() if k != "err_trace_1s"})
        return rec

    # ------------------------------------------------------------------ D
    def walk_cycle(self, speed: float, secs: float = 4.0, rest: float = 3.0):
        p0 = self.pose()
        hdg = p0["yaw"]
        fwd = [math.cos(hdg), math.sin(hdg)]
        self.set_cmd(mode=SLOW_WALK, movement=self.world_to_planner(fwd), facing=self.world_to_planner(fwd), speed=speed)
        zmin, nf, samples = self.watch(secs)
        pmid = self.pose()
        v_end = math.hypot(*pmid["base_lin_vel_w"][:2])
        t_stop = time.time()
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(hdg), speed=-1.0)
        stop_s = None
        t_ok = None
        while time.time() - t_stop < rest:
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            if math.hypot(*p["base_lin_vel_w"][:2]) < 0.1:
                t_ok = t_ok or time.time()
                if stop_s is None and time.time() - t_ok >= 0.3:
                    stop_s = t_ok - t_stop
            else:
                t_ok = None
            time.sleep(0.05)
        p1 = self.pose()
        dx, dy = p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]
        # steady-state speed: displacement over the last 2 s of the walk window
        ss = [s for s in samples if s[0] >= samples[-1][0] - 2.0]
        v_ss = math.hypot(ss[-1][2][0] - ss[0][2][0], ss[-1][2][1] - ss[0][2][1]) / max(1e-3, ss[-1][0] - ss[0][0]) if len(ss) > 2 else None
        rec = {"speed_cmd": speed, "walk_s": secs, "along_m": round(dx * fwd[0] + dy * fwd[1], 3), "lateral_m": round(-dx * fwd[1] + dy * fwd[0], 3),
               "v_steady_mps": None if v_ss is None else round(v_ss, 3), "v_at_stop_cmd": round(v_end, 3),
               "stop_s": None if stop_s is None else round(stop_s, 2), "dyaw_deg": round(math.degrees(wrap(p1["yaw"] - hdg)), 1),
               "pelvis_z_min": round(zmin, 3), "falls": nf + (p1["falls"] - pmid["falls"])}
        self.log("walk_cycle_result", **rec)
        return rec

    # ------------------------------------------------------------------ E
    def curve(self, speed: float, yaw_rate_dps: float, secs: float):
        p0 = self.pose()
        y0 = p0["yaw"]
        f0 = p0["falls"]
        t0 = time.time()
        zmin = 9.0
        errs = []
        yaw_acc, y_prev = 0.0, y0
        while time.time() - t0 < secs:
            t = time.time() - t0
            yc = wrap(y0 + math.radians(yaw_rate_dps) * t)
            d = [math.cos(yc), math.sin(yc)]
            self.set_cmd_quiet(mode=SLOW_WALK, movement=self.world_to_planner(d), facing=self.world_to_planner(d), speed=speed)
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            yaw_acc += wrap(p["yaw"] - y_prev)
            y_prev = p["yaw"]
            if t > 2.0:
                errs.append(abs(math.degrees(wrap(yc - p["yaw"]))))
            time.sleep(0.1)
        p1 = self.pose()
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(p1["yaw"]), speed=-1.0)
        time.sleep(2.5)
        p2 = self.pose()
        rec = {"speed_cmd": speed, "yaw_rate_cmd_dps": yaw_rate_dps, "secs": secs,
               "yaw_change_deg": round(math.degrees(yaw_acc), 1), "yaw_change_cmd_deg": round(yaw_rate_dps * secs, 1),
               "heading_lag_deg_mean": round(sum(errs) / len(errs), 1) if errs else None, "heading_lag_deg_max": round(max(errs), 1) if errs else None,
               "path_chord_m": round(math.hypot(p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]), 3),
               "pelvis_z_min": round(zmin, 3), "falls": p2["falls"] - f0}
        self.log("curve_result", **rec)
        return rec

    def set_cmd_quiet(self, **kw):
        with self.cmd_lock:
            self.cmd.update(kw)

    # ------------------------------------------------------------------ scenario
    def run(self):
        a = self.a
        if not self.startup():
            return self.finish()
        mz, nf, _ = self.watch(5.0)
        self.check("stand_5s", nf == 0 and mz > 0.6, pelvis_z_min=round(mz, 3), falls=nf)

        sec = set(a.sections.split(","))
        # A. open-loop IDLE turns
        for dd in ((90, -90, 45, -45, 135, -135) if "turns" in sec else ()):
            r = self.turn_open_loop(dd)
            self.char["turn_open_loop"].append(r)
        if "turns" in sec: self.check("turns_open_loop_no_falls", all(r["falls"] == 0 for r in self.char["turn_open_loop"]),
                   residual_err_deg={r["cmd_deg"]: r["residual_err_deg"] for r in self.char["turn_open_loop"]})
        # C. SLOW_WALK + zero movement turn
        for dd in ((90, -90) if "slowwalk" in sec else ()):
            r = self.turn_open_loop(dd, mode=SLOW_WALK)
            self.char["turn_slowwalk"].append(r)
        if "slowwalk" in sec:
            self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(self.pose()["yaw"]), speed=-1.0)
            time.sleep(2.0)
        if "slowwalk" in sec: self.check("turns_slowwalk_no_falls", all(r["falls"] == 0 for r in self.char["turn_slowwalk"]),
                   residual_err_deg={r["cmd_deg"]: r["residual_err_deg"] for r in self.char["turn_slowwalk"]},
                   translation_m={r["cmd_deg"]: r["translation_m"] for r in self.char["turn_slowwalk"]})
        # B. closed-loop turns
        for dd in ((90, -90, 180) if "closed" in sec else ()):
            r = self.turn_closed_loop(dd)
            self.char["turn_closed_loop"].append(r)
        if "closed" in sec: self.check("turn_closed_loop_within_3deg", all(abs(r["final_err_deg"]) <= 3.0 and r["falls"] == 0 for r in self.char["turn_closed_loop"]),
                   final_err_deg={r["cmd_deg"]: r["final_err_deg"] for r in self.char["turn_closed_loop"]},
                   time_s={r["cmd_deg"]: r["time_s"] for r in self.char["turn_closed_loop"]})
        # D. walk start/stop cycles (alternate direction by turning 180 deg in between every 2 cycles)
        for i, spd in enumerate(a.cycle_speeds if "cycles" in sec else ()):
            r = self.walk_cycle(spd)
            self.char["walk_cycles"].append(r)
            if i % 2 == 1:
                self.char["turn_closed_loop"].append(self.turn_closed_loop(180, tol_deg=5.0))
        wc = self.char["walk_cycles"]
        if "cycles" in sec: self.check("walk_cycles_no_falls", all(r["falls"] == 0 for r in wc), n=len(wc), falls=sum(r["falls"] for r in wc),
                   stop_s=[r["stop_s"] for r in wc], v_steady=[r["v_steady_mps"] for r in wc])
        # E. curved walks
        for spd, rate in (((0.5, 20.0), (0.5, -20.0), (0.4, 35.0)) if "curves" in sec else ()):
            self.char["curve"].append(self.curve(spd, rate, 6.0))
        if "curves" in sec: self.check("curves_no_falls", all(r["falls"] == 0 for r in self.char["curve"]),
                   heading_lag_mean={r["yaw_rate_cmd_dps"]: r["heading_lag_deg_mean"] for r in self.char["curve"]})
        # H. wl-body TurnToMotion emulation, as committed (push 0) and with the residual push (0.6)
        if "bodyturn" in sec:
            for push in (0.0, 0.6):
                for dd in (90, -90, 180, 45, -45):
                    self.char["body_turn"].append(self.turn_body_style(dd, push=push))
            bt = self.char["body_turn"]
            self.check("body_turn_as_committed_reaches_6deg", all(r["reached_tol"] for r in bt if r["push"] == 0.0),
                       results={r["cmd_deg"]: (r["reached_tol"], r["t_reach_s"], r["final_err_deg"]) for r in bt if r["push"] == 0.0})
            self.check("body_turn_with_push_reaches_6deg", all(r["reached_tol"] for r in bt if r["push"] > 0),
                       results={r["cmd_deg"]: (r["reached_tol"], r["t_reach_s"], r["final_err_deg"]) for r in bt if r["push"] > 0})
        # G. gentle envelope: what P3 should use by default
        if "gentle" in sec:
            g = self.char["gentle"]
            for dd in (90, -90, 180, -180, 45):
                g["turns"].append(self.turn_ramp(dd, rate_dps=a.gentle_turn_dps))
            for i in range(a.gentle_cycles):
                g["cycles"].append(self.walk_cycle(a.gentle_speed, secs=5.0))
                if i % 2 == 1:
                    g["turns"].append(self.turn_ramp(180, rate_dps=a.gentle_turn_dps))
            for rate in (15.0, -15.0):
                g["curves"].append(self.curve(a.gentle_speed, rate, 8.0))
            self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.facing_cmd(self.pose()["yaw"]), speed=-1.0)
            zmin, nf_s, _ = self.watch(a.gentle_stand_s)
            g["stand"] = {"secs": a.gentle_stand_s, "pelvis_z_min": round(zmin, 3), "falls": nf_s}
            nf_g = sum(r["falls"] for r in g["turns"] + g["cycles"] + g["curves"]) + nf_s
            self.check("gentle_envelope_no_falls", nf_g == 0, falls=nf_g,
                       turn_final_err_deg=[r["final_err_deg"] for r in g["turns"]], turn_time_s=[r["time_s"] for r in g["turns"]],
                       cycle_stop_s=[r["stop_s"] for r in g["cycles"]], curve_lag=[r["heading_lag_deg_mean"] for r in g["curves"]],
                       stand=g["stand"])
        nf = self.pose()["falls"] - self.falls_at_release
        self.check("no_falls_since_release", nf == 0, falls=nf)
        json.dump(self.char, open(self.out / "char.json", "w"), indent=1)
        if a.send_stop:
            self.send_stop_and_check()
        return self.finish()


def main():
    ap = base_args()
    ap.add_argument("--sections", default="turns,slowwalk,closed,cycles,curves", help="comma list of scenario sections to run")
    ap.add_argument("--gentle-speed", type=float, default=0.4)
    ap.add_argument("--gentle-turn-dps", type=float, default=30.0)
    ap.add_argument("--gentle-cycles", type=int, default=6)
    ap.add_argument("--gentle-stand-s", type=float, default=60.0)
    ap.add_argument("--cycle-speeds", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 0.5, 0.3, 0.3, 0.7, 0.7, 0.5, 0.5])
    a = ap.parse_args()
    r = Char(a).run()
    raise SystemExit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
