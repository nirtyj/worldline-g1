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
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SAVE_FRAMES = (1, 6, 11)


class ChainRecorder:
    """Wraps groot_arms' helpers: every observation (frame + state) and every chunk that leaves groot/.

    Frames: without `session`, observations number 1, 2, ... and `save_frames` picks which to save (frame_NN.png).
    With `session` set (G2 mode), each session numbers its own and every `every`-th one from its first is saved as
    frames/<session>_obsNN.png. The executor's warm-up observation is never counted (it uses build_obs_warmup)."""

    def __init__(self, out: Path, save_frames: tuple[int, ...] = SAVE_FRAMES, every: int = 5):
        self.out, self.save_frames, self.every = out, save_frames, every
        self.n_obs = 0
        self.session: str | None = None
        self.n_sess: dict[str, int] = {}
        self.obs: list[dict] = []
        self.chunks: list[dict] = []
        self.frames: list[str] = []

    def _save_frame(self, frame) -> None:
        from PIL import Image
        if self.session is None:
            if self.n_obs not in self.save_frames:
                return
            p = self.out / f"frame_{self.n_obs:02d}.png"
        else:
            k = self.n_sess[self.session]
            if (k - 1) % self.every:
                return
            (self.out / "frames").mkdir(exist_ok=True)
            p = self.out / "frames" / f"{self.session}_obs{k:02d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(p)
        self.frames.append(str(p.relative_to(self.out)))

    def wrap(self, h: dict) -> dict:
        build, to_chunk = h["build_obs"], h["to_chunk"]

        def build_obs(frame, body_q, lh, rh, prompt, **kw):
            self.n_obs += 1
            n = self.n_obs
            if self.session is not None:
                self.n_sess[self.session] = self.n_sess.get(self.session, 0) + 1
            self._save_frame(frame)
            self.obs.append({"n": n, "session": self.session, "t": time.monotonic(),
                             "body_q": [round(float(v), 4) for v in body_q],
                             "left_hand_q": [round(float(v), 4) for v in lh],
                             "right_hand_q": [round(float(v), 4) for v in rh], "prompt": prompt})
            return build(frame, body_q, lh, rh, prompt, **kw)

        def chunk(action, t0_mono, **kw):
            c = to_chunk(action, t0_mono=t0_mono, **kw)
            self.chunks.append({"t0": float(t0_mono), "t_rx": time.monotonic(), "n_obs": self.n_obs,
                                "session": self.session or "",
                                "raw": {k: np.asarray(v, dtype=np.float32)[0] for k, v in action.items()},
                                "upper_body": np.asarray(c.upper_body, dtype=np.float32),
                                "left_hand": np.asarray(c.left_hand, dtype=np.float32),
                                "right_hand": np.asarray(c.right_hand, dtype=np.float32)})
            return c

        # the executor's warm-up get_action (a black frame at zero state) builds its observation with the
        # unwrapped function: it is not evidence (wave 1's frame_01 was that black frame)
        return {**h, "build_obs": build_obs, "build_obs_warmup": build, "to_chunk": chunk}

    def save(self) -> dict:
        if not self.chunks:
            return {"chunks": 0}
        keys = sorted(self.chunks[0]["raw"])
        arr = {f"raw_{k}": np.stack([c["raw"][k] for c in self.chunks]) for k in keys}
        arr.update(t0=np.array([c["t0"] for c in self.chunks]), t_rx=np.array([c["t_rx"] for c in self.chunks]),
                   session=np.array([c["session"] for c in self.chunks]),
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
        after = [m["t"] for m in (ref_body.log if ref_body is not None else [])       # every chunk SENT after the
                 if m["kind"] == "chunk" and m["session_id"] == "man-smoke-halt"         # ack, whatever the reply
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


# ================================================================================================ G2 (M2b wave 2)
G2_PLAN = "NCNHNCHNCH"          # 10 sessions: N runs to its end, C is cancelled, H is halted on the body's lane
N_ARM = 14                      # the arm joints of the 17-D upper body (waist excluded)


class RecordingArm:
    """ArmPort wrapper around BodyArmPort: every message sent to wl-body with its send and reply times and the
    body's answer, WHATEVER the answer (the fence tests count every chunk sent after an ack, accepted or not)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.lock = threading.Lock()
        self.log: list[dict] = []

    def arm(self, args: dict, op_id: str | None = None) -> dict:
        t0 = time.monotonic()
        rep = self.inner.arm(args, op_id)
        t1 = time.monotonic()
        kind = ("end" if args.get("end") else "keepalive" if args.get("keepalive") else "chunk" if "chunk" in args
                else "release" if args.get("release") else "start")
        ch = args.get("chunk") or {}
        d = rep.get("data") if isinstance(rep.get("data"), dict) else {}
        with self.lock:
            self.log.append({"t_send": t0, "t_reply": t1, "kind": kind, "session_id": args.get("session_id"),
                             "seq": ch.get("seq"), "t0_mono": ch.get("t0_mono"), "ok": bool(rep.get("ok")),
                             "state": rep.get("state"), "error": rep.get("error"), "why": d.get("why"),
                             "dropped": d.get("dropped")})
        return rep

    def subscribe(self, cb):
        return self.inner.subscribe(cb)

    def supports_chunk(self):
        return self.inner.supports_chunk()

    def close(self) -> None:
        self.inner.close()

    def of(self, sid: str) -> list[dict]:
        with self.lock:
            return [m for m in self.log if m["session_id"] == sid]


class TrajRecorder(threading.Thread):
    """The measured upper body (g1_debug body_q[12:29], MuJoCo order: waist 3, left arm 7, right arm 7) and both
    Dex3 hands, every new g1_debug sample while a session is tagged."""

    def __init__(self, sensors: Any):
        super().__init__(daemon=True, name="g2-traj")
        self.sensors = sensors
        self.session: str | None = None
        self.samples: dict[str, list[tuple]] = {}
        self._run = True
        self._last: float | None = None

    def run(self) -> None:
        while self._run:
            d, t = self.sensors.debug_state()
            sid = self.session
            if d is not None and sid and t != self._last and d.get("body_q") is not None:
                self._last = t
                q = [float(v) for v in d["body_q"]]
                self.samples.setdefault(sid, []).append(
                    (t, q[12:29], [float(v) for v in d.get("left_hand_q") or [0.0] * 7],
                     [float(v) for v in d.get("right_hand_q") or [0.0] * 7]))
            time.sleep(0.004)

    def stop(self) -> None:
        self._run = False

    def arrays(self, sid: str) -> dict[str, np.ndarray]:
        rows = self.samples.get(sid) or []
        if not rows:
            return {}
        return {"t": np.array([r[0] for r in rows]), "q17": np.array([r[1] for r in rows]),
                "left_hand": np.array([r[2] for r in rows]), "right_hand": np.array([r[3] for r in rows])}


def ego_view_check(world: Any, sensors: Any, object_id: str, out_png: Path | None, consumer: str = "g2-view",
                   settle_s: float = 0.5) -> dict:
    """The GR00T view check the W2.5 bar uses, from P1's instance segmentation of ego_view (docs/contracts/p1_m2b.md
    §7): the target's pixels and bbox, and whether its bbox centre sits below the image's upper third (v >= 160 of
    480: Arena's training frames show the apple in the lower part). Saves the frame it judged."""
    out: dict[str, Any] = {"camera": "ego_view", "object": object_id}
    try:
        world.enable_camera("ego_view", True, consumer=consumer, ttl_s=10.0, hz=30.0)
    except Exception as e:  # noqa: BLE001 - an M1 P1 has no camera op (its 5565 stream is always on)
        out["enable_camera"] = f"unavailable: {e}"[:160]
    try:
        time.sleep(settle_s)
        sid = world.map.objects[object_id].scene_id
        try:
            rep = world.rpc.call("detections", timeout_s=3.0, camera="ego_view", min_px=1, bbox=True)
            if rep.get("ok") is False:
                raise RuntimeError(rep.get("code") or rep.get("error"))
        except Exception as e:  # noqa: BLE001 - an M1 P1 has no `detections` op (P1.6)
            rep = None
            out.update(px=None, ok=None, detections=f"unavailable: {e}"[:160])
        mine = [d for d in (rep or {}).get("detections") or [] if str(d.get("id")) == str(sid)]
        if rep is not None:
            out.update(method=rep.get("method"), render_seq=rep.get("render_seq"), cam_pose_wl=rep.get("cam_pose_wl"))
        if rep is None:
            pass
        elif mine:
            d = mine[0]
            u0, v0, u1, v1 = (float(x) for x in d["bbox"])
            uc, vc = (u0 + u1) / 2.0, (v0 + v1) / 2.0
            out.update(px=int(d["px"]), bbox=[int(u0), int(v0), int(u1), int(v1)], centre_uv=[round(uc, 1),
                       round(vc, 1)], dist_m=d.get("dist_m"), lower_two_thirds=bool(vc >= 160.0),
                       ok=bool(d["px"] >= 200 and vc >= 160.0))
        else:
            out.update(px=0, bbox=None, centre_uv=None, lower_two_thirds=False, ok=False)
        if out_png is not None:
            frame, _ = sensors.ego_frame()
            if frame is not None:
                from PIL import Image, ImageDraw
                img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
                dr = ImageDraw.Draw(img)
                dr.line([(0, 160), (639, 160)], fill=(255, 255, 0))
                if out.get("bbox"):
                    dr.rectangle(out["bbox"], outline=(255, 0, 0), width=2)
                img.save(out_png)
                out["frame"] = out_png.name
        return out
    finally:
        try:
            world.enable_camera("ego_view", False, consumer=consumer)
        except Exception:  # noqa: BLE001
            pass


def _steps(q: np.ndarray) -> np.ndarray:
    """Largest arm-joint change between consecutive measured samples (rad), per sample."""
    if len(q) < 2:
        return np.zeros(0)
    arms = np.asarray(q)[:, 3:]                                  # drop the waist: 14 arm joints
    return np.abs(np.diff(arms, axis=0)).max(axis=1)


def _palms(q17: np.ndarray) -> dict[str, np.ndarray]:
    from body.g1_kin import named_from_mj17, points
    pts = [points(named_from_mj17(v)) for v in q17]
    return {k: np.array([p[k] for p in pts]) for k in ("left_palm", "right_palm")}


def profile_groot_cfg(profile: str, **over: Any):
    """GrootArmsConfig from the profile's `groot_arms:` section (as services.executors.groot_arms.create reads it),
    then the given overrides."""
    from robot.profile import load_profile
    from services.executors.groot_arms import GrootArmsConfig
    cfg = GrootArmsConfig.from_dict(dict((load_profile(profile).raw or {}).get("groot_arms") or {}))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def one_epoch_space(robot: Any) -> bool:
    """True on the M2b façade (R.2): SonicBody.fence maps an execution's control_epoch/generation onto the body's
    numbers and HaltGate works in control_epoch space. False on the M1 façade (raw numbers, its own gate counter)."""
    return hasattr(robot.gate, "resume_epoch") and callable(getattr(robot.body, "fence", None))


def session_job(robot: Any, sid: str, ce: int, skill: Any, object_id: str, object_type: str, arm: str = "left"):
    """(Execution, ManipJob) for one groot_arms session, fenced the way ManipulationService fences it: the body's
    numbers from SonicBody.fence on the M2b façade, else the execution's own."""
    from api.execution import Execution
    from services.executors.kinematic_attach import ManipJob
    ex = Execution(execution_id=sid, tool_name="manipulate", args={}, generation=1, control_epoch=ce)
    one = one_epoch_space(robot)
    fence = dict(robot.body.fence(ex)) if one else {}
    j = ManipJob("pick", object_id, arm, skill.skill_id, epoch=ce if one else robot.gate.epoch)
    for k, v in dict(execution_id=sid, generation=int(fence.get("generation", 1)),
                     control_epoch=int(fence.get("control_epoch", ce)), object_type=object_type, skill=skill).items():
        setattr(j, k, v)
    return ex, j


def g2_session_report(kind: str, out, s, sends: list[dict], t_trig: float | None, receipt: dict | None,
                      traj: dict, t_start: float, t_done: float) -> dict:
    d = out.data or {}
    chunks = [m for m in sends if m["kind"] == "chunk"]
    r: dict[str, Any] = {
        "kind": {"N": "run", "C": "cancel", "H": "halt"}[kind], "status": out.status, "reason": out.reason,
        "detail": out.detail, "wall_s": round(t_done - t_start, 2), "inferences": d.get("inferences"),
        "chunks_sent": d.get("chunks_sent"), "chunk_msgs": len(chunks),
        "chunks_accepted": sum(1 for m in chunks if m["ok"] and not m["dropped"]),
        "chunks_dropped": d.get("chunks_dropped"), "latency_ms": d.get("latency_ms"),
        "clamped_frac": d.get("clamped_frac"), "clamped_frac_source": d.get("clamped_frac_source"),
        "slew_frac": d.get("slew_frac"), "body": d.get("body"), "gt": d.get("gt"), "hold_on_end":
        d.get("hold_on_end"), "view_check": d.get("view_check"), "camera_first_frame_s": d.get("camera_first_frame_s"),
        "stall_s_max": d.get("stall_s_max"), "prompt": d.get("prompt"), "session_id": d.get("session_id"),
        "control_epoch": d.get("control_epoch")}
    if s is not None and s.t_ack is not None:
        r["ack_after_fence_ms"] = None if s.t_fence is None else round((s.t_ack - s.t_fence) * 1000, 2)
        r["chunks_sent_after_ack"] = sum(1 for m in chunks if m["t_send"] > s.t_ack)       # every one, whatever reply
    if t_trig is not None:
        r["trigger_to_ack_ms"] = None if s is None or s.t_ack is None else round((s.t_ack - t_trig) * 1000, 2)
        r["trigger_to_result_ms"] = round((t_done - t_trig) * 1000, 1)
        late = [m for m in chunks if m["t_send"] >= t_trig]
        r["chunks_sent_after_trigger"] = [{"dt_ms": round((m["t_send"] - t_trig) * 1000, 1), "ok": m["ok"],
                                           "error": m["error"], "dropped": m["dropped"]} for m in late]
        r["chunks_accepted_after_trigger"] = sum(1 for m in late if m["ok"] and not m["dropped"])
    if receipt is not None:
        body = receipt.get("body") or {}
        # BodyClient.halt (lane) -> {acked, rtt_ms, wait_ms, body}; G1Robot.halt (R.2) -> {stopped, rtt_ms, ...}
        r["halt_receipt"] = {"acked": receipt.get("acked", receipt.get("stopped")), "rtt_ms": receipt.get("rtt_ms"),
                             "wait_ms": receipt.get("wait_ms"), "latency_ms": receipt.get("latency_ms"),
                             "handle_ms": body.get("handle_ms"), "kind": body.get("kind"),
                             "arms_latched": body.get("arms_latched"), "arm": body.get("arm"),
                             "epoch": receipt.get("epoch"),
                             "raw": {k: v for k, v in receipt.items() if k != "body"}}
    if traj:
        t, q = traj["t"], traj["q17"]
        st = _steps(q)
        r["measured"] = {"samples": int(len(t)), "rate_hz": round((len(t) - 1) / (t[-1] - t[0]), 1)
                         if len(t) > 1 and t[-1] > t[0] else None,
                         "max_arm_step_rad": round(float(st.max()), 4) if len(st) else None,
                         "left_hand_closure_max": round(float(np.abs(traj["left_hand"]).max()), 3)}
        if t_trig is not None and len(st):
            tt = t[1:]
            before = st[(tt >= t_trig - 1.0) & (tt < t_trig)]
            after = st[(tt >= t_trig) & (tt <= t_trig + 0.5)]
            r["measured"]["max_arm_step_1s_before_rad"] = round(float(before.max()), 4) if len(before) else None
            r["measured"]["max_arm_step_0p5s_after_rad"] = round(float(after.max()), 4) if len(after) else None
        try:
            pal = _palms(q)
            lp = pal["left_palm"]
            r["measured"]["left_palm_pelvis_m"] = {"start": np.round(lp[0], 3).tolist(),
                                                   "end": np.round(lp[-1], 3).tolist(),
                                                   "min": np.round(lp.min(axis=0), 3).tolist(),
                                                   "max": np.round(lp.max(axis=0), 3).tolist()}
        except Exception as e:  # noqa: BLE001 - FK is evidence, not a verdict
            r["measured"]["palm_error"] = repr(e)
    return r


async def g2_async(a: argparse.Namespace) -> dict:
    import random

    from api.execution import ExecutionManager, ResultHandle
    from robot.factory import build
    from services.common import EventSink
    from services.executors.groot_arms import BodyArmPort, GrootArmExecutor, ZmqSensors, _groot_helpers
    from services.skills import load_skill_specs
    from sim.clock import SimClock
    from sim.log import EventLog

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = SimClock(1.0)
    world, robot, _ = build(a.profile, a.scene, clock, frames=None)
    em = ExecutionManager(clock)
    bc = robot.body.client
    off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
    rng = random.Random(a.seed)
    rep: dict[str, Any] = {
        "mode": "g2", "scene": a.scene, "profile": a.profile, "p1": dict(world.p1_info), "t_start": time.time(),
        "plan": a.plan, "object": a.object, "skill": a.skill, "endpoint": a.endpoint,
        "labels": {"groot_arms": "experimental (off-the-shelf Arena N1.7 G1 checkpoint, zero-shot)",
                   "arm_op": "wl-body arm op, chunk mode (B.8): REAL chunks into SONIC's upper-body override",
                   "halt": "the body's B.1 halt lane (BodyClient.halt, PUSH 5612) + the runtime HaltGate",
                   "grasp": "GT judge (gt_lifted / grasp_missed) through the WorldModel; success is not expected"}}
    rep["sim_health_start"] = world.sim_health().__dict__ if hasattr(world, "sim_health") else None
    stats0 = world.stats()
    sensors = traj = exe = arm = None
    poses: list[tuple] = []
    pose_task = None
    try:
        # 0. staging (P1.7 move_object, test-only) and the stance
        steps: list[dict] = []
        if a.stage:
            c = [float(v) for v in a.stage.split(",")]
            world.move_object(a.object, c[:3], c[3] if len(c) > 3 else None)
            await asyncio.sleep(1.0)
            steps.append({"stage": {"object": a.object, "center": c, "now": [round(v, 3) for v in
                                                                                 world.object(a.object).pos]}})
        if a.stand:
            r = await _run_tool(robot, em, "navigate", {"location": a.stand}, 0)
            steps.append({"navigate": _brief(r)})
        if a.approach:
            x, y, yaw_deg = (float(v) for v in a.approach.split(","))
            h = await asyncio.to_thread(bc.approach, x, y, math.radians(yaw_deg), None, (0.03, 3.0), True, 90.0)
            steps.append({"approach": {"goal": [x, y, yaw_deg], "ok": h.ok, "reason": h.reason,
                                       "pos_err": (h.result or {}).get("pos_err"),
                                       "yaw_err_deg": (h.result or {}).get("yaw_err_deg")}})
        elif a.reach:
            obj = world.object(a.object)
            for _ in range(2):
                r = await _run_tool(robot, em, "check_reachability", {"object_type": obj.type,
                                                                        "object_id": a.object}, 0)
                steps.append({"check_reachability": _brief(r)})
                if r.data.get("reason") != "needs_reposition":
                    break
                r2 = await _run_tool(robot, em, "navigate", {"location": "reach_stance", "anchor": a.stand,
                                                              "stance": r.data["stance"]}, 0)
                steps.append({"reach_stance": _brief(r2)})
        await asyncio.sleep(1.0)
        p = world.robot_pose()
        o = world.object(a.object)
        from world import coords
        f, l = coords.world_to_body((p.x, p.y, p.yaw), o.pos[0], o.pos[1])
        rep["stance"] = {"steps": steps, "pose": {"x": round(p.x, 3), "y": round(p.y, 3),
                                                  "yaw_deg": round(math.degrees(p.yaw), 1)},
                         "object_center": [round(v, 3) for v in o.pos], "object_where": o.where,
                         "object_body_frame": {"forward": round(f, 3), "left": round(l, 3),
                                               "height_above_floor": round(o.pos[2], 3)}}

        # 1. the executor: real chunks into the real arm op, every message recorded
        sensors = ZmqSensors(f"tcp://127.0.0.1:{a.camera_port + off}", f"tcp://127.0.0.1:{5557 + off}",
                             a.camera_key).start()
        rep["camera"] = {"port": a.camera_port, "key": a.camera_key,
                         "label": "Arena ego_view (P1 5566, OD1)" if a.camera_port == 5566 else
                         "INTERIM: not Arena's ego_view (an M1 P1 stream); the policy's visual input is "
                         "off-distribution"}
        rep["view"] = await asyncio.to_thread(ego_view_check, world, sensors, a.object, out / "view_check.png")
        port = BodyArmPort.of(robot.body)
        if port is None:
            raise SystemExit("the body has no wl-body client (profile sonic/full on a live stack)")
        arm = RecordingArm(port)
        events: list[dict] = []
        arm.subscribe(lambda ev: events.append({**ev, "t_rx": time.monotonic()}))
        # the profile's own groot_arms section (config/profiles/full.yaml: the render budget of §9 included), so G2
        # runs the session exactly as `full` does; only the run's knobs are overridden
        cfg = profile_groot_cfg(a.profile, endpoint=a.endpoint, max_duration_s=a.max_s, view_min_px=a.view_min_px,
                                lead_s=a.lead_s)
        rep["groot_cfg"] = {k: getattr(cfg, k) for k in ("camera_hz", "camera_warm_hz", "session_head_hz", "frame_sync", "request_image",
                                                         "replan_s", "lead_s", "timeout_s", "endpoint")}
        rec = ChainRecorder(out, every=a.frame_every)
        helpers = rec.wrap(_groot_helpers())
        log = EventLog(clock)
        exe = GrootArmExecutor(world, arm=arm, sensors=sensors, cfg=cfg, gate=robot.gate, events=EventSink(log),
                               helpers=helpers)
        t_h = time.monotonic()
        while not exe.health().ok and time.monotonic() < t_h + 60.0:
            await asyncio.sleep(0.2)
        rep["executor_health"] = {**exe.health().__dict__, "wait_s": round(time.monotonic() - t_h, 2)}
        if not exe.health().ok:
            raise SystemExit(f"groot_arms is down: {exe.health().detail}")
        skill = {x.skill_id: x for x in load_skill_specs()}[a.skill]
        otype = world.object(a.object).type
        traj = TrajRecorder(sensors)
        traj.start()

        async def sample_poses():
            while True:
                q = world.robot_pose()
                poses.append((time.monotonic(), q.x, q.y, q.yaw, getattr(q, "pelvis_z", None), bool(q.fallen),
                              float(q.speed)))
                await asyncio.sleep(0.1)
        pose_task = asyncio.ensure_future(sample_poses())

        # 2. the sessions. Epochs: the M2b façade (R.2: SonicBody.fence, HaltGate in control_epoch space) maps an
        # execution's runtime control_epoch onto the body's space; the M1 façade sends it raw and its HaltGate counts
        # its own epochs. Either way a halt fences this session's control_epoch and the next session carries a newer
        # one (the body refuses control_epoch <= its halt epoch after the resume: stale_command).
        one_space = one_epoch_space(robot)
        rep["epoch_space"] = "one (R.2: SonicBody.fence)" if one_space else "M1 façade (raw control_epoch)"
        acquire = getattr(robot.body, "acquire", None) if one_space else None
        release_fn = getattr(robot.body, "release", None) if one_space else None
        ce = max(int(getattr(robot.gate, "epoch", 0)), int(getattr(robot.gate, "resume_epoch", 0) or 0)) + 1
        sessions = []
        for i, kind in enumerate(a.plan):
            sid = f"g2-{i + 1:02d}-{kind}"
            ex, j = session_job(robot, sid, ce, skill, a.object, otype, a.arm_side)
            h = ResultHandle(ex)
            lease = None
            if callable(acquire):
                lease = await acquire(ex, "ARM_STREAM")
            rec.session, traj.session = sid, sid
            t_start = time.monotonic()
            task = asyncio.ensure_future(exe.run(j, h))
            t_trig = receipt = None
            halt_epoch = None
            if kind in "CH":
                t_end = time.monotonic() + 20.0
                while not task.done() and time.monotonic() < t_end and (
                        exe.last_session is None or exe.last_session.id != sid
                        or exe.last_session.chunks_sent < a.trigger_chunks):
                    await asyncio.sleep(0.02)
                await asyncio.sleep(rng.uniform(0.0, 0.4))
                if not task.done():
                    t_trig = time.monotonic()
                    if kind == "C":
                        h.cancel("g2 cancel")
                    elif one_space:
                        receipt = await asyncio.to_thread(robot.halt, ce)    # gate latch at ce, then the B.1 lane
                        halt_epoch = ce
                    else:
                        halt_epoch = robot.gate.halt()             # the runtime latch first, then the body's lane
                        receipt = await asyncio.to_thread(bc.halt, halt_epoch, 0.03, "g2")
            o = await task
            t_done = time.monotonic()
            await asyncio.sleep(0.4)                               # late replies / the terminal event
            rec.session = traj.session = None
            s = exe.last_session if exe.last_session is not None and exe.last_session.id == sid else None
            if exe.last_client is not None:
                exe.last_client.join(2.0)
            row = g2_session_report(kind, o, s, arm.of(sid), t_trig, receipt, traj.arrays(sid), t_start, t_done)
            row["session"] = sid
            tr0 = traj.arrays(sid)
            row["t_trigger_rel"] = None if t_trig is None or not tr0 else round(t_trig - float(tr0["t"][0]), 3)
            row["body_events"] = [{"state": e["state"], "ended_by": (e.get("data") or {}).get("ended_by"),
                                   "hold": (e.get("data") or {}).get("hold")}
                                  for e in events if (e.get("data") or {}).get("session_id") == sid
                                  and e.get("state") in ("succeeded", "failed", "canceled")]
            win = [x for x in poses if t_start - 0.5 <= x[0] <= time.monotonic()]
            row["fell"] = any(x[5] for x in win)
            row["pelvis_z_min"] = round(min((x[4] for x in win if x[4] is not None), default=float("nan")), 3)
            row["base_drift_m"] = round(math.hypot(win[-1][1] - win[0][1], win[-1][2] - win[0][2]), 3) if win else None
            row["fence"] = {"control_epoch": j.control_epoch, "generation": j.generation, "runtime_ce": ce,
                            "lease": None if lease is None else {k: lease.get(k) for k in ("ok", "reason")}}
            if callable(release_fn):
                try:
                    await release_fn(sid)
                except Exception:  # noqa: BLE001
                    pass
            if kind == "H" and halt_epoch is not None:
                if one_space:
                    ce = halt_epoch + 1
                    rr = await asyncio.to_thread(robot.resume, ce)
                    row["resume"] = {"control_epoch": ce, "body": rr if isinstance(rr, dict) else None}
                else:
                    robot.gate.resume(halt_epoch)
                    rr = await asyncio.to_thread(bc.resume, halt_epoch)
                    row["resume"] = {"ok": rr.get("ok"), "error": rr.get("error")}
                    ce = halt_epoch + 1
            # arms back to SONIC's own reference between sessions (stop {arms: true}: blend 1.5 s)
            hs = await asyncio.to_thread(bc.stop, True, 10.0, True)
            row["release"] = {"ok": hs.ok, "reason": hs.reason}
            await asyncio.sleep(a.rest_s)
            sessions.append(row)
            print(json.dumps({k: row.get(k) for k in ("session", "status", "reason", "chunks_sent",
                                                      "chunks_sent_after_ack", "chunks_accepted_after_trigger",
                                                      "trigger_to_ack_ms", "slew_frac", "clamped_frac", "fell")},
                             default=repr), flush=True)
        rep["sessions"] = sessions
        rep["chain"] = rec.save()
        # 3. trajectories: measured (g1_debug) per session, the chunks as sent (chunks.npz), palms by FK
        arrs = {}
        for row in sessions:
            tr = traj.arrays(row["session"])
            for k, v in tr.items():
                arrs[f"{row['session']}__{k}"] = v
            if tr:
                for k, v in _palms(tr["q17"]).items():
                    arrs[f"{row['session']}__{k}"] = v
        np.savez_compressed(out / "trajectories.npz", **arrs)
        try:
            rep["trajectories_png"] = plot_trajectories(out, sessions)
        except Exception as e:  # noqa: BLE001 - a plot is evidence, never a verdict
            rep["trajectories_png"] = f"failed: {e!r}"
        rep["summary"] = g2_summary(sessions)
    finally:
        if pose_task is not None:
            pose_task.cancel()
        if traj is not None:
            traj.stop()
        if exe is not None:
            exe.close()
        rep["sim_health_end"] = world.sim_health().__dict__ if hasattr(world, "sim_health") else None
        try:
            st1 = world.stats()
            rep["p1_stats"] = {k: st1.get(k) for k in ("rtf_total", "rtf_worst_1s", "heartbeat_pubs", "cameras",
                                                        "render_calls", "overruns", "hitches_gt25ms")}
            rep["p1_stats"]["heartbeat_pubs_delta"] = (st1.get("heartbeat_pubs") or 0) - (stats0.get("heartbeat_pubs")
                                                                                          or 0)
        except Exception as e:  # noqa: BLE001
            rep["p1_stats"] = {"error": repr(e)}
        await robot.shutdown()
        world.close()
    rep["t_end"] = time.time()
    return rep


def plot_trajectories(out: Path, rows: list[dict]) -> str | None:
    """trajectories.png (PIL only: the runtime venv has no matplotlib): one panel per session, the left palm in the
    pelvis frame (FK of the measured arm, g1_debug) as solid x/y/z lines and the palm of the rows GR00T sent (FK of
    each chunk's rows at their own times, t0 + k dt: SENT, not necessarily played; a halt or cancel drops them) as
    dots; a grey line marks the cancel/halt trigger."""
    from PIL import Image, ImageDraw
    try:
        tr = np.load(out / "trajectories.npz")
        ch = np.load(out / "chunks.npz") if (out / "chunks.npz").exists() else None
    except Exception:  # noqa: BLE001
        return None
    from body.g1_kin import named_from_mj17, points
    from groot import joint_order as jo
    W, H, M = 900, 170, 40
    img = Image.new("RGB", (W, H * max(1, len(rows))), "white")
    dr = ImageDraw.Draw(img)
    cols = {0: (200, 30, 30), 1: (30, 150, 30), 2: (30, 60, 200)}
    for i, row in enumerate(rows):
        sid = row["session"]
        y0 = i * H
        dr.rectangle([M, y0 + 18, W - 10, y0 + H - 18], outline=(180, 180, 180))
        dr.text((M, y0 + 2), f"{sid}  {row['status']}({row['reason']})  chunks {row.get('chunks_sent')}  "
                             f"slew {row.get('slew_frac')}  left palm x/y/z (red/green/blue), pelvis frame, "
                             f"-0.1..0.5 m", fill=(0, 0, 0))
        key = f"{sid}__t"
        if key not in tr.files:
            continue
        t, lp = tr[key], tr[f"{sid}__left_palm"]
        ts = float(t[0])
        span = max(float(t[-1]) - ts, 1.0)

        def X(tt):
            return M + (W - 10 - M) * (float(tt) - ts) / span

        def Y(v):
            return y0 + H - 18 - (H - 36) * (float(v) + 0.1) / 0.6
        for d in range(3):
            pts = [(X(tt), Y(v)) for tt, v in zip(t, lp[:, d])]
            if len(pts) > 1:
                dr.line(pts, fill=cols[d], width=2)
        if ch is not None and "session" in ch.files:
            for c in np.where(ch["session"] == sid)[0]:
                ub = jo.wire_to_mj17(ch["upper_body_wire"][c])
                t0 = float(ch["t0"][c])
                for k in range(0, ub.shape[0], 4):
                    p = points(named_from_mj17(ub[k]))["left_palm"]
                    tt = t0 + k * 0.02
                    if ts <= tt <= ts + span:
                        for d in range(3):
                            x, y = X(tt), Y(p[d])
                            dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=cols[d])
        if row.get("t_trigger_rel") is not None:
            x = X(ts + row["t_trigger_rel"])
            dr.line([(x, y0 + 18), (x, y0 + H - 18)], fill=(120, 120, 120), width=1)
    path = out / "trajectories.png"
    img.save(path)
    return path.name


def g2_summary(rows: list[dict]) -> dict:
    by = {k: [r for r in rows if r["kind"] == k] for k in ("run", "cancel", "halt")}
    ok_c = [r for r in by["cancel"] if r["status"] == "cancelled" and r.get("chunks_sent_after_ack") == 0]
    ok_h = [r for r in by["halt"] if r["status"] == "failed" and r["reason"] == "halted"
            and r.get("chunks_sent_after_ack") == 0 and r.get("chunks_accepted_after_trigger") == 0
            and (r.get("halt_receipt") or {}).get("acked")]
    slew = [r["slew_frac"] for r in rows if isinstance(r.get("slew_frac"), (int, float))]
    clamp = [r["clamped_frac"] for r in rows if isinstance(r.get("clamped_frac"), (int, float))]
    lat = [r["latency_ms"]["p50"] for r in rows if (r.get("latency_ms") or {}).get("p50") is not None]
    return {"sessions": len(rows), "falls": sum(1 for r in rows if r.get("fell")),
            "cancel_ok": f"{len(ok_c)}/{len(by['cancel'])}", "halt_ok": f"{len(ok_h)}/{len(by['halt'])}",
            "chunks_sent_total": sum(r.get("chunks_sent") or 0 for r in rows),
            "chunks_sent_after_ack_total": sum(r.get("chunks_sent_after_ack") or 0 for r in rows),
            "chunks_accepted_after_trigger_total": sum(r.get("chunks_accepted_after_trigger") or 0
                                                       for r in rows if r["kind"] != "run"),
            "slew_frac": {"mean": round(float(np.mean(slew)), 4) if slew else None,
                          "max": round(float(np.max(slew)), 4) if slew else None},
            "clamped_frac": {"mean": round(float(np.mean(clamp)), 4) if clamp else None,
                             "max": round(float(np.max(clamp)), 4) if clamp else None},
            "latency_p50_ms_median": round(float(np.median(lat)), 1) if lat else None,
            "outcomes": {f"{r['status']}({r['reason']})": sum(1 for x in rows if (x["status"], x["reason"]) ==
                                                                (r["status"], r["reason"])) for r in rows},
            "gt_lift_max_m": max(((r.get("gt") or {}).get("lift_max_m") or 0.0) for r in rows) if rows else None,
            "grasp_success": sum(1 for r in rows if r["status"] == "succeeded")}


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
    g = ap.add_argument_group("G2 (--g2, with --arm body): N sessions of real chunks into the real arm op")
    g.add_argument("--g2", action="store_true")
    g.add_argument("--plan", default=G2_PLAN, help="one letter per session: N run, C cancel, H halt")
    g.add_argument("--profile", default="full", help="the robot façade's profile (full: GR00T skills' stance)")
    g.add_argument("--stage", default=None, help="x,y,z[,yaw]: move the object there first (P1 move_object)")
    g.add_argument("--approach", default=None, help="x,y,yaw_deg: the body's approach op to this stance")
    g.add_argument("--reach", action="store_true", help="check_reachability + navigate(reach_stance)")
    g.add_argument("--trigger-chunks", type=int, default=3, help="cancel/halt once this many chunks were sent")
    g.add_argument("--lead-s", type=float, default=0.15)
    g.add_argument("--rest-s", type=float, default=2.5)
    g.add_argument("--frame-every", type=int, default=5)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--camera-port", type=int, default=5566, help="GR00T's view: P1 ego_view 5566 (M1 P1: 5565)")
    g.add_argument("--camera-key", default="ego_view")
    a = ap.parse_args(argv)
    a.walks = [tuple(w.split(":", 1)) for w in (a.walk or ["kitchen_counter_1a:kitchen_counter_1b"])]
    if a.g2:
        if a.arm != "body":
            ap.error("--g2 needs --arm body")
        rep = asyncio.run(g2_async(a))
        out = Path(a.out)
        (out / "g2.json").write_text(json.dumps(rep, indent=1, default=lambda x: repr(x)))
        print(json.dumps({"summary": rep.get("summary"), "view": rep.get("view"),
                          "stance": (rep.get("stance") or {}).get("object_body_frame")}, default=repr))
        return 0
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
