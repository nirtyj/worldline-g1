"""Drive the SONIC deploy through zmq_manager (command/planner topics) against the MuJoCo reference sim.

The messages are built only with gear_sonic's own builders
(gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:30-158, WBC @ b042411). Ground truth comes from sim_ref.py's
control socket. The script runs a fixed scenario with pass/fail checks and writes drive_result.json +
drive_trace.jsonl + deploy_g1_debug.jsonl into --out.

Scenario (timings are wall clock; the deploy runs on the wall clock):
  start (command start=1 planner=1) -> planner keepalive IDLE -> band release -> stand
  -> walk forward -> idle -> turn in place +90 deg -> strafe left -> walk + stop mid-walk
  -> planner silence (1 s timeout -> IDLE) -> [optional] command stop=1 (the deploy must exit)

movement/facing are sent in the PLANNER frame (+X = robot heading when the planner was initialised,
localmotion_kplanner.hpp:332-352, 591-624). yaw0 is the world yaw at start, used to convert.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message

IDLE, SLOW_WALK, WALK = 0, 1, 2  # localmotion_kplanner.hpp:78-82


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Driver:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://127.0.0.1:{a.zmq_port}")
        self.pub_lock = threading.Lock()  # deploy SUB connects (zmq_packed_message_subscriber.hpp:201)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVHWM, 1000)
        self.sub.connect(f"tcp://127.0.0.1:{a.zmq_out_port}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        self.ctl_lock = threading.Lock()
        self.ctl = None
        self._new_ctl()
        # planner command state (planner frame)
        self.cmd = dict(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=[1.0, 0.0, 0.0], speed=-1.0, height=-1.0)
        self.cmd_lock = threading.Lock()
        self.keepalive_on = False
        self.planner_sent = 0
        self.stop_flag = False
        self.yaw0 = 0.0
        self.falls_at_release = 0
        self.debug_count = 0
        self.debug_first_t = None
        self.debug_last = None
        self.results: list[dict] = []
        self.trace_f = open(self.out / "drive_trace.jsonl", "w")
        self.dbg_f = open(self.out / "deploy_g1_debug.jsonl", "w")
        threading.Thread(target=self._keepalive, daemon=True).start()
        threading.Thread(target=self._debug_reader, daemon=True).start()
        threading.Thread(target=self._tracer, daemon=True).start()
        time.sleep(0.5)  # ZMQ slow joiner: let the deploy's SUB connect before the first message

    # ---------------------------------------------------------------- plumbing
    def _new_ctl(self):
        if self.ctl is not None:
            self.ctl.close(0)
        self.ctl = self.ctx.socket(zmq.REQ)
        self.ctl.setsockopt(zmq.LINGER, 0)
        self.ctl.setsockopt(zmq.RCVTIMEO, 2000)
        self.ctl.setsockopt(zmq.SNDTIMEO, 2000)
        self.ctl.connect(f"tcp://127.0.0.1:{self.a.ctl_port}")

    def send(self, msg: bytes):
        with self.pub_lock:
            self.pub.send(msg)

    def sim(self, op, **kw):
        with self.ctl_lock:
            try:
                self.ctl.send(json.dumps({"op": op, **kw}).encode())
                return json.loads(self.ctl.recv())
            except zmq.Again:
                self._new_ctl()
                raise RuntimeError(f"sim control socket timeout on {op}")

    def pose(self):
        return self.sim("pose")

    def set_cmd(self, **kw):
        with self.cmd_lock:
            self.cmd.update(kw)
        self.log("planner_cmd", **kw)

    def _keepalive(self):
        period = 1.0 / self.a.planner_hz
        while not self.stop_flag:
            if self.keepalive_on:
                with self.cmd_lock:
                    c = dict(self.cmd)
                self.send(build_planner_message(c["mode"], c["movement"], c["facing"], c["speed"], c["height"]))
                self.planner_sent += 1
            time.sleep(period)

    def _debug_reader(self):
        while not self.stop_flag:
            if not self.sub.poll(200):
                continue
            raw = self.sub.recv()
            t = time.time()
            payload = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
            self.debug_count += 1
            if self.debug_first_t is None:
                self.debug_first_t = t
            self.debug_last = payload
            if self.debug_count % 5 == 0:  # 10 Hz subset to disk
                rec = {"t_wall": t, "index": payload.get("index")}
                for k in ("base_quat", "base_trans_target", "base_quat_target", "init_base_quat", "delta_heading"):
                    if k in payload:
                        rec[k] = payload[k]
                if "body_q_target" in payload:
                    rec["body_q_target_legs"] = [round(x, 4) for x in payload["body_q_target"][:12]]
                if "last_action" in payload:
                    rec["last_action_legs"] = [round(x, 4) for x in payload["last_action"][:12]]
                self.dbg_f.write(json.dumps(rec) + "\n")

    def _tracer(self):
        while not self.stop_flag:
            try:
                p = self.pose()
                with self.cmd_lock:
                    p["cmd"] = dict(self.cmd)
                p["planner_sent"] = self.planner_sent
                p["debug_count"] = self.debug_count
                self.trace_f.write(json.dumps(p) + "\n")
            except Exception:
                pass
            time.sleep(0.1)

    def log(self, event, **kw):
        rec = {"t_wall": time.time(), "event": event, **kw}
        self.trace_f.write(json.dumps(rec) + "\n")
        print(f"[drive {time.strftime('%H:%M:%S')}] {event} {kw}", flush=True)

    def check(self, name, ok, **data):
        r = {"test": name, "pass": bool(ok), **data}
        self.results.append(r)
        print(f"[drive] {'PASS' if ok else 'FAIL'} {name} {data}", flush=True)
        return ok

    # ---------------------------------------------------------------- frames
    def world_to_planner(self, v_world):
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        return [c * v_world[0] - s * v_world[1], s * v_world[0] + c * v_world[1], 0.0]

    def planner_yaw(self, yaw_world):
        return wrap(yaw_world - self.yaw0)

    def upright_window(self, secs, zmin=0.55, zmax=1.0):
        """Sample the pose for `secs`; return (ok, min_z, max_z, falls_delta)."""
        f0 = self.pose()["falls"]
        zs = []
        t_end = time.time() + secs
        while time.time() < t_end:
            zs.append(self.pose()["pelvis_z"])
            time.sleep(0.1)
        f1 = self.pose()["falls"]
        return (min(zs) > zmin and max(zs) < zmax and f1 == f0), min(zs), max(zs), f1 - f0

    # ---------------------------------------------------------------- scenario
    def run(self):
        a = self.a
        self.log("wait_sim")
        t0 = time.time()
        while True:
            try:
                if self.sim("ping")["ok"]:
                    break
            except Exception:
                pass
            if time.time() - t0 > 30:
                raise SystemExit("sim not reachable")
        p = self.pose()
        self.log("sim_pose", **p)

        # 1) command start (after the deploy printed "Init Done"; the orchestrator waits for that)
        self.yaw0 = p["yaw"]
        self.keepalive_on = True  # IDLE, facing +X planner frame
        t_start = time.time()
        while self.debug_count == 0 and time.time() - t_start < 20:
            self.send(build_command_message(start=True, stop=False, planner=True))
            time.sleep(0.2)
        started = self.debug_count > 0
        self.check("control_started", started, secs=round(time.time() - t_start, 2), yaw0_deg=round(math.degrees(self.yaw0), 1))
        if not started:
            return self.finish()
        time.sleep(2.0)
        dbg_rate = self.debug_count / max(1e-3, time.time() - self.debug_first_t)
        self.check("g1_debug_rate", 40 <= dbg_rate <= 60, hz=round(dbg_rate, 1))

        # 2) band release (upstream procedure: start first, then '9')
        if a.band_lower > 0:
            # hand-over: lower the band until the feet carry the weight, then release (no free fall)
            for i in range(1, 21):
                self.sim("band_length", length=-a.band_lower * i / 20)
                time.sleep(0.1)
            time.sleep(1.0)
        pz = self.pose()["pelvis_z"]
        self.sim("band", on=False)
        self.falls_at_release = self.pose()["falls"]
        self.log("band_released", pelvis_z=round(pz, 3), band_lower=a.band_lower)
        time.sleep(3.0)
        st = self.sim("stats")
        self.check("lowcmd_rate", st["lowcmd_hz_2s"] > 300 and 35 <= st["lowcmd_leg_target_change_hz_2s"] <= 65,
                   msg_hz=st["lowcmd_hz_2s"], leg_target_change_hz=st["lowcmd_leg_target_change_hz_2s"],
                   mode_machine=st["lowcmd_mode_machine"], rtf=round(st["rtf"], 3))

        # 3) stand
        ok, zmin, zmax, nf = self.upright_window(a.stand_secs)
        p = self.pose()
        self.check("stand", ok, secs=a.stand_secs, pelvis_z_min=round(zmin, 3), pelvis_z_max=round(zmax, 3), falls=nf,
                   drift_xy=round(math.hypot(p["base_pos"][0], p["base_pos"][1]), 3))

        # 4) walk forward (planner-frame +X of the CURRENT heading)
        p0 = self.pose()
        hdg = p0["yaw"]
        fwd_w = [math.cos(hdg), math.sin(hdg)]
        self.set_cmd(mode=SLOW_WALK, movement=self.world_to_planner(fwd_w), facing=self.world_to_planner(fwd_w), speed=a.walk_speed)
        ok_w, zmin, _, nf = self.upright_window(a.walk_secs)
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], speed=-1.0)
        time.sleep(2.5)
        p1 = self.pose()
        dx, dy = p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]
        along = dx * fwd_w[0] + dy * fwd_w[1]
        lateral = -dx * fwd_w[1] + dy * fwd_w[0]
        self.check("walk_forward", ok_w and along >= a.walk_min_m, along_m=round(along, 3), lateral_m=round(lateral, 3),
                   dyaw_deg=round(math.degrees(wrap(p1["yaw"] - hdg)), 1), pelvis_z_min=round(zmin, 3), falls=nf)
        # frame check: displacement direction in world vs commanded world heading
        self.check("frame_world_heading", abs(math.degrees(wrap(math.atan2(dy, dx) - hdg))) < 15,
                   disp_dir_deg=round(math.degrees(math.atan2(dy, dx)), 1), cmd_heading_deg=round(math.degrees(hdg), 1))

        # 5) turn in place +90 deg (IDLE + new facing, like keyboard Q/E: keyboard_handler.hpp:556-566)
        p0 = self.pose()
        tgt = wrap(p0["yaw"] + math.radians(90))
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.world_to_planner([math.cos(tgt), math.sin(tgt)]), speed=-1.0)
        ok_t, zmin, _, nf = self.upright_window(a.turn_secs)
        p1 = self.pose()
        dyaw = math.degrees(wrap(p1["yaw"] - p0["yaw"]))
        moved = math.hypot(p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1])
        self.check("turn_in_place_90", ok_t and abs(dyaw - 90) <= 15, dyaw_deg=round(dyaw, 1), translation_m=round(moved, 3),
                   pelvis_z_min=round(zmin, 3), falls=nf)

        # 6) strafe left (movement perpendicular to facing; keyboard ','/'.': keyboard_handler.hpp:597-610)
        p0 = self.pose()
        hdg = p0["yaw"]
        left_w = [-math.sin(hdg), math.cos(hdg)]
        face_w = [math.cos(hdg), math.sin(hdg)]
        self.set_cmd(mode=SLOW_WALK, movement=self.world_to_planner(left_w), facing=self.world_to_planner(face_w), speed=a.strafe_speed)
        ok_s, zmin, _, nf = self.upright_window(a.strafe_secs)
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], speed=-1.0)
        time.sleep(2.0)
        p1 = self.pose()
        dx, dy = p1["base_pos"][0] - p0["base_pos"][0], p1["base_pos"][1] - p0["base_pos"][1]
        lat = dx * left_w[0] + dy * left_w[1]
        fwd = dx * face_w[0] + dy * face_w[1]
        self.check("strafe_left", ok_s and lat >= a.strafe_min_m, lateral_m=round(lat, 3), forward_m=round(fwd, 3),
                   dyaw_deg=round(math.degrees(wrap(p1["yaw"] - hdg)), 1), falls=nf)

        # 7) stop mid-walk: walk, then IDLE; time until |v_xy| < 0.1 m/s and stays upright
        p0 = self.pose()
        hdg = p0["yaw"]
        fwd_w = [math.cos(hdg), math.sin(hdg)]
        self.set_cmd(mode=SLOW_WALK, movement=self.world_to_planner(fwd_w), facing=self.world_to_planner(fwd_w), speed=a.walk_speed)
        time.sleep(3.0)
        v_before = math.hypot(*self.pose()["base_lin_vel_w"][:2])
        t_stop = time.time()
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], speed=-1.0)
        t_settle = None
        vs = []
        while time.time() - t_stop < 4.0:
            p = self.pose()
            v = math.hypot(*p["base_lin_vel_w"][:2])
            vs.append(v)
            # "stopped" = speed stays < 0.1 m/s for 0.3 s
            if v < 0.1:
                if t_settle is None:
                    t_settle = time.time()
                elif time.time() - t_settle > 0.3:
                    break
            else:
                t_settle = None
            time.sleep(0.02)
        stop_s = (t_settle - t_stop) if t_settle else None
        ok_u, zmin, _, nf = self.upright_window(3.0)
        self.check("stop_mid_walk", stop_s is not None and stop_s <= 1.5 and ok_u, v_before=round(v_before, 3),
                   stop_s=None if stop_s is None else round(stop_s, 2), pelvis_z_min=round(zmin, 3), falls=nf)

        # 8) planner silence -> 1 s timeout -> IDLE, robot must keep standing (zmq_manager.hpp:582-630)
        self.set_cmd(mode=SLOW_WALK, movement=self.world_to_planner(fwd_w), facing=self.world_to_planner(fwd_w), speed=a.walk_speed)
        time.sleep(2.0)
        self.keepalive_on = False
        t_sil = time.time()
        self.log("planner_silence")
        ok_u, zmin, _, nf = self.upright_window(4.0)
        v_end = math.hypot(*self.pose()["base_lin_vel_w"][:2])
        self.check("planner_timeout_idle", ok_u and v_end < 0.15, v_end=round(v_end, 3), pelvis_z_min=round(zmin, 3), falls=nf)
        self.set_cmd(mode=IDLE, movement=[0.0, 0.0, 0.0], facing=self.cmd["facing"], speed=-1.0)
        self.keepalive_on = True
        time.sleep(1.0)

        # 9) long stand (optional)
        if a.long_stand_secs > 0:
            ok, zmin, zmax, nf = self.upright_window(a.long_stand_secs)
            self.check("long_stand", ok, secs=a.long_stand_secs, pelvis_z_min=round(zmin, 3), pelvis_z_max=round(zmax, 3), falls=nf)

        # no upstream fall-reset (root teleport) may have happened since the band release
        nf = self.pose()["falls"] - self.falls_at_release
        self.check("no_falls_since_release", nf == 0, falls=nf)

        # 10) command stop=1 must terminate the deploy (zmq_manager.hpp:343-362 -> main() exits)
        if a.send_stop:
            n0 = self.debug_count
            for _ in range(3):
                self.send(build_command_message(start=False, stop=True, planner=True))
                time.sleep(0.1)
            time.sleep(2.0)
            n1 = self.debug_count
            time.sleep(1.0)
            self.check("zmq_stop_ends_control", self.debug_count == n1, g1_debug_after_stop=self.debug_count - n0)
        return self.finish()

    def finish(self):
        self.stop_flag = True
        time.sleep(0.3)
        st = {}
        try:
            st = self.sim("stats")
        except Exception:
            pass
        res = {"pass": all(r["pass"] for r in self.results) and len(self.results) > 0, "tests": self.results,
               "sim_stats": st, "planner_msgs_sent": self.planner_sent, "g1_debug_msgs": self.debug_count,
               "yaw0_deg": math.degrees(self.yaw0), "args": vars(self.a)}
        json.dump(res, open(self.out / "drive_result.json", "w"), indent=1)
        print("DRIVE_RESULT " + json.dumps({"pass": res["pass"], "tests": {r["test"]: r["pass"] for r in self.results}}), flush=True)
        self.trace_f.close()
        self.dbg_f.close()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmq-port", type=int, default=5656)
    ap.add_argument("--zmq-out-port", type=int, default=5657)
    ap.add_argument("--ctl-port", type=int, default=5712)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner-hz", type=float, default=20.0)
    ap.add_argument("--band-lower", type=float, default=0.18, help="lower the band by this many m before release (0 = upstream drop from 1.0 m)")
    ap.add_argument("--stand-secs", type=float, default=15.0)
    ap.add_argument("--walk-speed", type=float, default=0.5)
    ap.add_argument("--walk-secs", type=float, default=6.0)
    ap.add_argument("--walk-min-m", type=float, default=2.0)
    ap.add_argument("--turn-secs", type=float, default=7.0)
    ap.add_argument("--strafe-speed", type=float, default=0.3)
    ap.add_argument("--strafe-secs", type=float, default=4.0)
    ap.add_argument("--strafe-min-m", type=float, default=0.5)
    ap.add_argument("--long-stand-secs", type=float, default=0.0)
    ap.add_argument("--send-stop", action="store_true")
    a = ap.parse_args()
    r = Driver(a).run()
    raise SystemExit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
