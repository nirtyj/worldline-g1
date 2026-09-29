"""Live GR00T smoke on the stack (M2b wave 1 integration; E5's plumbing half before the body's chunk mode is merged).

Drives ONE GrootArmExecutor session on the live stack with the robot standing at a counter stance, then a halted
session, then halts mid-walk through the body. Run on the box with the runtime venv while `scripts/m2_up.sh --profile
sonic` is up and a PolicyServer answers (scripts/groot_server.sh):

    .venv-rt/bin/python -m tools.groot_live_smoke --scene procthor-train-40 --stand kitchen_counter_1b \\
        --object pepper_shaker_1 --out outputs/m2b_wave1/integ/live/<ts>

What is real and what is not:
    real      P1 (Isaac, the M2b wire: live object poses, ego_view on 5566 enabled by the session, P1 detections),
              SONIC (g1_debug on 5557: GR00T's joint state), wl-body (navigate, reach stance, halt), the world model,
              the PolicyServer (N1.7, nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace) and every groot/ helper.
    arm op    `--arm record` (default): the chunks stop at the body boundary. The `arm` op is not on this branch
              (body B.4/B.8 live on the body branch), so they go to the contract's reference body
              (tests/fakes/fake_arm_body.py), which plays them by time and reports clamped_frac but moves nothing.
              `--arm body` streams into wl-body's `arm` op (after the wave-2 merge; groot_arms reports itself down
              when body.state.arm.modes has no "chunk").
Every chunk and the frames it was computed from are saved: frame_<n>.png (the exact RGB array sent to the policy),
chunks.npz (per inference: t0, latency, the raw GR00T arm/hand actions and the ArmChunk sent, SONIC wire order),
smoke.json (numbers). Nothing here is a success claim: the checkpoint is not expected to grasp zero-shot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

SAVE_FRAMES = (1, 6, 11)


class ChainRecorder:
    """Wraps groot_arms' helpers: every observation (frame + state) and every chunk that leaves groot/."""

    def __init__(self, out: Path, save_frames: tuple[int, ...] = SAVE_FRAMES):
        self.out, self.save_frames = out, save_frames
        self.n_obs = 0
        self.obs: list[dict] = []
        self.chunks: list[dict] = []
        self.frames: list[str] = []

    def wrap(self, h: dict) -> dict:
        build, to_chunk = h["build_obs"], h["to_chunk"]

        def build_obs(frame, body_q, lh, rh, prompt, **kw):
            self.n_obs += 1
            n = self.n_obs
            if n in self.save_frames:
                from PIL import Image
                p = self.out / f"frame_{n:02d}.png"
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(p)
                self.frames.append(p.name)
            self.obs.append({"n": n, "t": time.monotonic(), "body_q": [round(float(v), 4) for v in body_q],
                             "left_hand_q": [round(float(v), 4) for v in lh],
                             "right_hand_q": [round(float(v), 4) for v in rh]})
            return build(frame, body_q, lh, rh, prompt, **kw)

        def chunk(action, t0_mono, **kw):
            c = to_chunk(action, t0_mono=t0_mono, **kw)
            self.chunks.append({"t0": float(t0_mono), "t_rx": time.monotonic(), "n_obs": self.n_obs,
                                "raw": {k: np.asarray(v, dtype=np.float32)[0] for k, v in action.items()},
                                "upper_body": np.asarray(c.upper_body, dtype=np.float32),
                                "left_hand": np.asarray(c.left_hand, dtype=np.float32),
                                "right_hand": np.asarray(c.right_hand, dtype=np.float32)})
            return c

        return {**h, "build_obs": build_obs, "to_chunk": chunk}

    def save(self) -> dict:
        if not self.chunks:
            return {"chunks": 0}
        keys = sorted(self.chunks[0]["raw"])
        arr = {f"raw_{k}": np.stack([c["raw"][k] for c in self.chunks]) for k in keys}
        arr.update(t0=np.array([c["t0"] for c in self.chunks]), t_rx=np.array([c["t_rx"] for c in self.chunks]),
                   upper_body_wire=np.stack([c["upper_body"] for c in self.chunks]),
                   left_hand=np.stack([c["left_hand"] for c in self.chunks]),
                   right_hand=np.stack([c["right_hand"] for c in self.chunks]))
        np.savez_compressed(self.out / "chunks.npz", **arr)
        (self.out / "obs_states.json").write_text(json.dumps(self.obs))
        return {"chunks": len(self.chunks), "frames": self.frames, "keys": keys}


async def _run_tool(robot, em, tool: str, args: dict, epoch: int) -> Any:
    from types import SimpleNamespace

    from api.execution import Rejected
    ex = em.create(tool, args, generation=1, control_epoch=epoch)
    try:
        return await robot.start(ex).result()
    except Rejected as e:                  # CAPABILITY (R.6 sim health, policy down): recorded, not fatal
        return SimpleNamespace(status="rejected", summary=f"rejected ({e.stage}): {e.message}",
                               data={"reason": e.code, "detail": e.message})


def _brief(res) -> dict:
    d = dict(res.data or {})
    keep = ("reason", "at", "executor", "distance_m", "duration_s", "reachable", "preferred_arm", "object_id",
            "detail", "mode", "method", "sees", "stance")
    return {"status": res.status, "summary": res.summary, **{k: d.get(k) for k in keep if k in d}}


async def main_async(a: argparse.Namespace) -> dict:
    from api.execution import ExecutionManager
    from robot.factory import build
    from services.executors.groot_arms import (BodyArmPort, GrootArmExecutor, GrootArmsConfig, ZmqSensors,
                                               _groot_helpers)
    from services.executors.kinematic_attach import ManipJob
    from services.skills import load_skill_specs
    from sim.clock import SimClock
    from api.execution import Execution, ResultHandle

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = SimClock(1.0)
    world, robot, _ = build("sonic", a.scene, clock, frames=None)
    em = ExecutionManager(clock)
    rep: dict[str, Any] = {"scene": a.scene, "p1": dict(world.p1_info), "t_start": time.time(),
                           "arm_port": a.arm, "labels": {"groot_arms": "experimental",
                                                         "arm_op": "reference body (tests/fakes/fake_arm_body.py): "
                                                         "chunks stop at the body boundary" if a.arm == "record"
                                                         else "wl-body arm op (chunk mode)"}}
    rep["sim_health_start"] = world.sim_health().__dict__ if hasattr(world, "sim_health") else None
    off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
    try:
        # 1. the counter stance
        steps = []
        if a.stand:
            r = await _run_tool(robot, em, "navigate", {"location": a.stand}, 0)
            steps.append({"navigate": _brief(r)})
            r = await _run_tool(robot, em, "observe", {"mode": "scan", "why": "smoke"}, 0)
            steps.append({"scan": _brief(r)})
            obj = world.object(a.object)
            r = await _run_tool(robot, em, "check_reachability", {"object_type": obj.type, "object_id": a.object}, 0)
            steps.append({"check_reachability": _brief(r)})
            if r.data.get("reason") == "needs_reposition":
                r2 = await _run_tool(robot, em, "navigate", {"location": "reach_stance", "anchor": a.stand,
                                                              "stance": r.data["stance"]}, 0)
                steps.append({"reach_stance": _brief(r2)})
                r = await _run_tool(robot, em, "check_reachability",
                                    {"object_type": obj.type, "object_id": a.object}, 0)
                steps.append({"check_reachability": _brief(r)})
        rep["stance"] = steps
        p = world.robot_pose()
        o = world.object(a.object)
        rep["pose_at_stance"] = {"x": round(p.x, 3), "y": round(p.y, 3), "yaw_deg": round(math.degrees(p.yaw), 1),
                                 "object_center": [round(v, 3) for v in o.pos], "object_where": o.where,
                                 "object_pose_source": o.pose_source,
                                 "dist_xy_m": round(math.hypot(o.pos[0] - p.x, o.pos[1] - p.y), 3)}

        # 2. the executor over the live world + sensors; chunks to the body boundary
        sensors = ZmqSensors(f"tcp://127.0.0.1:{5566 + off}", f"tcp://127.0.0.1:{5557 + off}", "ego_view").start()
        if a.arm == "body":
            arm = BodyArmPort.of(robot.body)
            ref_body = None
        else:
            from tests.fakes.fake_arm_body import FakeArmBody
            ref_body = arm = FakeArmBody()
        cfg = GrootArmsConfig(endpoint=a.endpoint, max_duration_s=a.max_s, view_min_px=a.view_min_px,
                              lead_s=0.15, camera_hz=30.0)
        if a.arm == "record":
            # the chunks never reach the hands, so the measured Dex3 hands stay in the deploy's default fist (it
            # commands that without hand fields): grasp_missed would fire on the fist, not on anything GR00T did
            cfg.closure_thresh = 1e9
            rep["labels"]["grasp_judge"] = "grasp_missed off: the real hands are not driven in record mode"
        rec = ChainRecorder(out)
        helpers = rec.wrap(_groot_helpers())
        from services.common import EventSink
        from sim.log import EventLog
        log = EventLog(clock)
        exe = GrootArmExecutor(world, arm=arm, sensors=sensors, cfg=cfg, gate=robot.gate, events=EventSink(log),
                               helpers=helpers)
        t_end = time.monotonic() + 30.0
        while not exe.health().ok and time.monotonic() < t_end:
            await asyncio.sleep(0.2)
        rep["executor_health"] = exe.health().__dict__
        skills = {s.skill_id: s for s in load_skill_specs()}
        skill = skills[a.skill]
        otype = world.object(a.object).type

        def job(eid: str) -> tuple[ManipJob, ResultHandle]:
            ex = Execution(execution_id=eid, tool_name="manipulate", args={}, generation=1, control_epoch=0)
            j = ManipJob("pick", a.object, a.arm_side, skill.skill_id, epoch=robot.gate.epoch)
            for k, v in dict(execution_id=eid, generation=1, control_epoch=0, object_type=otype, skill=skill).items():
                setattr(j, k, v)
            return j, ResultHandle(ex)

        # 2a. one full session
        seq0 = sensors._seq
        t0 = time.monotonic()
        j, h = job("man-smoke-1")
        o1 = await exe.run(j, h)
        dt = time.monotonic() - t0
        fr = sensors._seq - seq0
        rep["session"] = {"status": o1.status, "reason": o1.reason, "phase": o1.phase, "detail": o1.detail,
                          "phases": o1.phases, "data": o1.data, "wall_s": round(dt, 2),
                          "ego_view_frames_received": fr, "ego_view_hz": round(fr / dt, 2) if dt else None}
        rep["chain"] = rec.save()
        if ref_body is not None:
            rep["session"]["body_boundary"] = {
                "chunks_accepted": len([m for m in ref_body.log if m["kind"] == "chunk" and m["reply"].get("ok")]),
                "messages": len(ref_body.log), "hold": ref_body.hold and ref_body.hold.get("mode")}
        # latency from the executor (the policy round trip incl. the 0.9 MB request)
        # 2b. a halted session: G1Robot.halt() (runtime gate + the body's stop, INTERIM halt lane) mid-stream
        j, h = job("man-smoke-halt")
        task = asyncio.ensure_future(exe.run(j, h))
        t_end = time.monotonic() + 15.0
        while (exe.last_session is None or exe.last_session.id != "man-smoke-halt"
               or exe.last_session.chunks_sent < 2) and time.monotonic() < t_end and not task.done():
            await asyncio.sleep(0.02)
        t_h = time.monotonic()
        receipt = robot.halt()
        o2 = await task
        s2 = exe.last_session
        after = [m["t"] for m in (ref_body.log if ref_body is not None else [])
                 if m["kind"] == "chunk" and m["session_id"] == "man-smoke-halt" and m["reply"].get("ok")
                 and s2 is not None and s2.t_ack is not None and m["t"] > s2.t_ack]
        rep["halted_session"] = {"status": o2.status, "reason": o2.reason, "receipt": receipt,
                                 "terminal_after_halt_s": round(time.monotonic() - t_h, 3),
                                 "chunks_sent_before": s2.chunks_sent if s2 else None,
                                 "chunks_after_ack": len(after), "cancel_ack_ms": o2.data.get("cancel_ack_ms"),
                                 "hold_on_end": o2.data.get("hold_on_end")}
        robot.resume(robot.gate.epoch)
        exe.close()
        if ref_body is not None:
            ref_body.close()

        # 3. halt mid-walk through the body (the INTERIM lane: body `stop`, 30 ms reply budget)
        halts = []
        for i, (dest, back) in enumerate(a.walks):
            for loc in (dest, back):
                ex = em.create("navigate", {"location": loc}, generation=1, control_epoch=robot.gate.epoch)
                hh = robot.start(ex)
                await asyncio.sleep(a.halt_after_s)
                v0 = world.planar_speed()
                rc = robot.halt()
                r = await hh.result()
                await asyncio.sleep(1.5)
                v15 = world.planar_speed()
                halts.append({"to": loc, "speed_at_halt": round(v0, 3), "receipt": rc, "result": r.status,
                              "reason": (r.data or {}).get("reason"), "speed_1_5s": round(v15, 3),
                              "fallen": world.robot_pose().fallen})
                robot.resume(robot.gate.epoch)
                # finish the walk so the next trial starts from a stand
                r = await _run_tool(robot, em, "navigate", {"location": loc}, robot.gate.epoch)
        rep["halts"] = halts
        rep["events"] = [e for e in log.events if e.get("type") in ("policy.health", "groot.stall")][:20]
    finally:
        rep["sim_health_end"] = world.sim_health().__dict__ if hasattr(world, "sim_health") else None
        try:
            rep["p1_stats"] = {k: v for k, v in world.stats().items()
                               if k in ("rtf_total", "rtf_worst_1s", "heartbeat_pubs", "cameras", "render_calls")}
        except Exception as e:  # noqa: BLE001
            rep["p1_stats"] = {"error": repr(e)}
        await robot.shutdown()
        world.close()
    rep["t_end"] = time.time()
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="procthor-train-40")
    ap.add_argument("--stand", default="kitchen_counter_1b")
    ap.add_argument("--object", default="pepper_shaker_1")
    ap.add_argument("--skill", default="groot.pick.any.arena_static_experimental.v0")
    ap.add_argument("--arm-side", default="left", help="the GR00T checkpoint is left-handed")
    ap.add_argument("--arm", choices=("record", "body"), default="record")
    ap.add_argument("--endpoint", default=os.environ.get("WL_GROOT_ENDPOINT", "tcp://127.0.0.1:5550"))
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--view-min-px", type=float, default=200.0)
    ap.add_argument("--halt-after-s", type=float, default=2.5)
    ap.add_argument("--walk", action="append", default=None, help="DEST:BACK pairs for the halt trials")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.walks = [tuple(w.split(":", 1)) for w in (a.walk or ["kitchen_counter_1a:kitchen_counter_1b"])]
    rep = asyncio.run(main_async(a))
    out = Path(a.out)
    (out / "smoke.json").write_text(json.dumps(rep, indent=1, default=lambda x: repr(x)))
    s = rep.get("session") or {}
    d = s.get("data") or {}
    print(json.dumps({"session": {k: s.get(k) for k in ("status", "reason", "wall_s", "ego_view_hz")},
                      "latency_ms": d.get("latency_ms"), "clamped_frac": d.get("clamped_frac"),
                      "clamped_frac_source": d.get("clamped_frac_source"), "inferences": d.get("inferences"),
                      "chunks_sent": d.get("chunks_sent"), "view_check": d.get("view_check"),
                      "camera_first_frame_s": d.get("camera_first_frame_s"),
                      "halted_session": rep.get("halted_session"),
                      "halts": [(x["result"], x["reason"], (x["receipt"] or {}).get("latency_ms"),
                                 (x["receipt"] or {}).get("stopped"), x["speed_1_5s"]) for x in rep.get("halts", [])]},
                     default=repr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
