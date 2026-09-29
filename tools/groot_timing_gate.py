"""SONIC's timing gate with the GR00T client active (M2b wave 2, W2.6; docs/walk_diagnosis.md "Rules"; PLAN §0.10 b, d).

OD3 puts the PolicyServer on the dev box, so what the GR00T client adds to the MAIN box is its own work: P1 renders
the Arena `ego_view` at 30 Hz (a second 640x480 render product), the client decodes the JPEG, builds the observation
and ships a 0.92 MB request through the SSH tunnel (ssh encrypts it) about 2.5 times a second, and while standing the
body's `arm` op plays the returned chunks into SONIC's upper-body override. This tool measures SONIC's timing gate
with and without that load, standing and walking, in interleaved rounds (slow drifts on the box do not favour one
phase):

    stand_base    standing, no GR00T client
    stand_groot   standing, one real groot_arms session (real chunks into the real arm op; the grasp judge is off:
                  this is a load run, not a pick) for --stand-s
    walk_base     SONIC walking the rtf_cameras pattern (walk 0.4 m/s 4 s, 2 x 90 deg, walk back, 2 x 90 deg)
    walk_groot    the same walk with the GR00T client loop running (ego_view on, observation + get_action at the
                  executor's 2.5 Hz, the chunk built and dropped: `manipulate` never runs while walking, rule C7)

Per phase: gt.pose RTF over 1 s windows (p10 / mean / min / share < 0.98; the bar is p10 >= 0.98, PLAN §0.10 b),
P1 heartbeat publishes during the phase (> 0 while moving = INVALID, walk_diagnosis rule 3), P1 overruns and
hitches, the leg-target change rate, the deploy's g1_debug cadence as seen from P5 (a proxy: the bridge-side
`irregular` share of rule 3 is not exposed by P1), get_action latency through the link, and falls. The GPU is checked
for other compute processes before and after (rule 1). Deliberate GR00T-load measurement (PLAN §0.10 d).

    WL_PORT_OFFSET=0 .venv-rt/bin/python -m tools.groot_timing_gate --out outputs/m2b_wave2/opsg/timing-<ts> \\
        [--rounds 2] [--stand-s 20] [--walk-s 40] [--endpoint tcp://127.0.0.1:5550] [--no-home]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

PHASES = ("stand_base", "stand_groot", "walk_base", "walk_groot")


def gpu_apps() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e!r}"]


class DebugCadence(threading.Thread):
    """Receive times of the deploy's g1_debug (PUB 5557), the only per-tick SONIC signal P5 sees."""

    def __init__(self, endpoint: str):
        super().__init__(daemon=True, name="g1-debug-cadence")
        self.endpoint = endpoint
        self.t: list[float] = []
        self.lock = threading.Lock()
        self._run = True

    def run(self) -> None:
        import zmq
        s = zmq.Context.instance().socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        s.connect(self.endpoint)
        while self._run:
            if s.poll(100):
                while True:
                    try:
                        s.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    with self.lock:
                        self.t.append(time.monotonic())
        s.close(0)

    def window(self, t0: float, t1: float) -> dict:
        with self.lock:
            t = np.array([x for x in self.t if t0 <= x <= t1])
        if len(t) < 3:
            return {"n": int(len(t))}
        d = np.diff(t) * 1000.0
        edges = np.arange(t[0], t[-1], 0.02)
        counts = np.histogram(t, bins=edges)[0] if len(edges) > 1 else np.zeros(0)
        return {"n": int(len(t)), "rate_hz": round((len(t) - 1) / (t[-1] - t[0]), 2),
                "gap_ms_p50": round(float(np.percentile(d, 50)), 2), "gap_ms_p99": round(float(np.percentile(d, 99)), 2),
                "gap_ms_max": round(float(d.max()), 2), "gaps_over_40ms": int((d > 40.0).sum()),
                "windows_not_one_msg": round(float((counts != 1).mean()), 4) if len(counts) else None,
                "note": "P5-side receive cadence of g1_debug: a proxy, not the bridge's `irregular` share"}


class ClientLoad(threading.Thread):
    """The GR00T client's main-box work without the arm op: the latest ego_view frame + g1_debug -> observation ->
    get_action through the link -> ArmChunk (dropped), at most every replan_s (the executor's 2.5 Hz)."""

    def __init__(self, sensors: Any, endpoint: str, prompt: str, replan_s: float = 0.4, timeout_s: float = 1.5):
        super().__init__(daemon=True, name="groot-client-load")
        from groot.actions import to_arm_chunk
        from groot.obs import build_observation
        from groot.policy_client import PolicyClient
        self.sensors, self.prompt, self.replan_s = sensors, prompt, replan_s
        self.client = PolicyClient(endpoint, timeout_s=timeout_s)
        self.build, self.to_chunk = build_observation, to_arm_chunk
        self.lat: list[float] = []
        self.errors: list[str] = []
        self.stale = 0
        self._run = True

    def run(self) -> None:
        try:
            while self._run:
                t0 = time.monotonic()
                frame, tf = self.sensors.ego_frame()
                dbg, td = self.sensors.debug_state()
                now = time.monotonic()
                if frame is None or dbg is None or now - tf > 0.15 or now - td > 0.06:
                    self.stale += 1
                    time.sleep(0.02)
                    continue
                obs = self.build(frame, dbg["body_q"], dbg.get("left_hand_q") or [0.0] * 7,
                                 dbg.get("right_hand_q") or [0.0] * 7, self.prompt)
                t_req = time.monotonic()
                try:
                    act = self.client.get_action(obs)
                    self.lat.append((time.monotonic() - t_req) * 1000.0)
                    self.to_chunk(act, t0_mono=td)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"{type(e).__name__}: {e}"[:160])
                left = self.replan_s - (time.monotonic() - t0)
                if left > 0:
                    time.sleep(left)
        finally:
            self.client.close()

    def stop(self) -> None:
        self._run = False


def lat_stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1),
            "max": round(float(max(v)), 1)}


async def main_async(a: argparse.Namespace) -> dict:
    from api.execution import Execution, ResultHandle
    from body.client import BodyClient
    from body.config import ep
    from body.p1_client import PoseSub
    from robot.factory import build
    from services.executors.groot_arms import (BodyArmPort, GrootArmExecutor, GrootArmsConfig, ZmqSensors,
                                               _groot_helpers)
    from services.executors.kinematic_attach import ManipJob
    from services.skills import load_skill_specs
    from sim.clock import SimClock
    from sim_isaac.tools.rtf_cameras import summarize, walk_pattern

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
    clock = SimClock(1.0)
    world, robot, _ = build("full", a.scene, clock, frames=None)
    rpc = world.rpc
    bc = BodyClient(port_offset=off).connect(10)
    sub = PoseSub(ep(5601 + off))
    samples: list = []
    lock = threading.Lock()

    def on_pose(p):
        with lock:
            samples.append((p.recv_mono, p.t_sim, p.rtf, p.fallen))
    sub.on_pose = on_pose
    sub.start()
    cad = DebugCadence(ep(5557 + off))
    cad.start()
    sensors = ZmqSensors(f"tcp://127.0.0.1:{5566 + off}", f"tcp://127.0.0.1:{5557 + off}", "ego_view").start()
    res: dict[str, Any] = {"tool": "groot_timing_gate", "scene": a.scene, "endpoint": a.endpoint,
                           "rounds": a.rounds, "stand_s": a.stand_s, "walk_s": a.walk_s, "gpu_apps_before": gpu_apps(),
                           "label": "deliberate GR00T-load measurement (PLAN §0.10 d)", "runs": []}
    skill = {x.skill_id: x for x in load_skill_specs()}[a.skill]
    cfg = GrootArmsConfig(endpoint=a.endpoint, max_duration_s=a.stand_s, view_min_px=0.0, lead_s=0.15,
                          camera_hz=30.0, closure_thresh=1e9)        # a load run: the grasp judge is off
    exe = GrootArmExecutor(world, arm=BodyArmPort.of(robot.body), sensors=sensors, cfg=cfg, gate=robot.gate,
                           helpers=_groot_helpers())
    t_h = time.monotonic()
    while not exe.health().ok and time.monotonic() < t_h + 60:
        await asyncio.sleep(0.2)
    res["executor_health"] = exe.health().__dict__
    try:
        if not a.no_home:
            sp = (rpc.call("get_scene_info") or {}).get("spawn") or {}
            if sp:
                h = await asyncio.to_thread(bc.go_to, float(sp["x"]), float(sp["y"]), float(sp.get("yaw", 0.0)), 120)
                res["home"] = {"ok": h.ok, "reason": h.reason}
        n = 0
        for rnd in range(a.rounds):
            for phase in PHASES:
                await asyncio.sleep(a.settle_s)
                run: dict[str, Any] = {"phase": phase, "round": rnd}
                load = None
                s0 = rpc.call("get_stats")
                with lock:
                    samples.clear()
                t0 = time.monotonic()
                if phase == "stand_base":
                    await asyncio.sleep(a.stand_s)
                elif phase == "stand_groot":
                    n += 1
                    sid = f"tg-{n:02d}"
                    ex = Execution(execution_id=sid, tool_name="manipulate", args={}, generation=1,
                                   control_epoch=robot.gate.epoch + 1)
                    j = ManipJob("pick", a.object, "left", skill.skill_id, epoch=robot.gate.epoch)
                    for k, v in dict(execution_id=sid, generation=1, control_epoch=robot.gate.epoch + 1,
                                     object_type=world.object(a.object).type, skill=skill).items():
                        setattr(j, k, v)
                    o = await exe.run(j, ResultHandle(ex))
                    run["session"] = {"status": o.status, "reason": o.reason,
                                      "inferences": o.data.get("inferences"), "chunks_sent": o.data.get("chunks_sent"),
                                      "latency_ms": o.data.get("latency_ms"), "slew_frac": o.data.get("slew_frac"),
                                      "clamped_frac": o.data.get("clamped_frac")}
                    await asyncio.to_thread(bc.stop, True, 10.0, True)      # arms back to SONIC (blend 1.5 s)
                else:
                    if phase == "walk_groot":
                        await asyncio.to_thread(world.enable_camera, "ego_view", True, consumer="timing-gate",
                                                ttl_s=30.0, hz=30.0)
                        await asyncio.sleep(0.5)
                        load = ClientLoad(sensors, a.endpoint, skill.prompt_template.format(label="potato"))
                        load.start()
                    ops: list = []
                    t_end = time.monotonic() + a.walk_s
                    await asyncio.to_thread(walk_pattern, bc, t_end, ops)
                    run["walk_ops"] = ops
                    if load is not None:
                        load.stop()
                        load.join(3.0)
                        await asyncio.to_thread(world.enable_camera, "ego_view", False, consumer="timing-gate")
                        run["client"] = {"latency_ms": lat_stats(load.lat), "errors": load.errors[:5],
                                         "n_errors": len(load.errors), "obs_stale": load.stale}
                t1 = time.monotonic()
                s1 = rpc.call("get_stats")
                with lock:
                    run.update(summarize(list(samples), s0, s1))
                run["g1_debug"] = cad.window(t0, t1)
                moving = phase.startswith("walk")
                run["valid"] = not (moving and (run.get("heartbeat_pubs") or 0) > 0)
                run["pass_p10"] = bool((run.get("rtf1s_p10") or 0) >= 0.98)
                res["runs"].append(run)
                print(f"[timing_gate] round {rnd} {phase:11s} p10 {run.get('rtf1s_p10')} mean {run.get('rtf1s_mean')} "
                      f"min {run.get('rtf1s_min')} hb {run.get('heartbeat_pubs')} fallen {run.get('fallen')} "
                      f"lat {(run.get('client') or run.get('session') or {}).get('latency_ms')}", flush=True)
                (out / "timing_gate.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
                if run.get("fallen"):
                    raise RuntimeError("robot fell")
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        res["gpu_apps_after"] = gpu_apps()
        exe.close()
        try:
            bc.stop()
        except Exception:  # noqa: BLE001
            pass
        bc.close()
        sub.stop()
        cad._run = False
        await robot.shutdown()
        world.close()
    agg: dict[str, Any] = {}
    for ph in PHASES:
        rows = [r for r in res["runs"] if r["phase"] == ph and "rtf1s_p10" in r]
        if rows:
            agg[ph] = {"p10_min": min(r["rtf1s_p10"] for r in rows),
                       "p10_mean": round(float(np.mean([r["rtf1s_p10"] for r in rows])), 4),
                       "mean": round(float(np.mean([r["rtf1s_mean"] for r in rows])), 4),
                       "min": min(r["rtf1s_min"] for r in rows),
                       "heartbeat_pubs": sum(r.get("heartbeat_pubs") or 0 for r in rows),
                       "valid": all(r["valid"] for r in rows), "pass": all(r["pass_p10"] for r in rows),
                       "fallen": any(r.get("fallen") for r in rows)}
    res["summary"] = agg
    (out / "timing_gate.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="procthor-train-40")
    ap.add_argument("--endpoint", default=os.environ.get("WL_GROOT_ENDPOINT", "tcp://127.0.0.1:5550"))
    ap.add_argument("--skill", default="groot.pick.any.arena_static_experimental.v0")
    ap.add_argument("--object", default="potato_1", help="only names the session's prompt and GT object")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--stand-s", type=float, default=20.0)
    ap.add_argument("--walk-s", type=float, default=40.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--no-home", action="store_true")
    a = ap.parse_args(argv)
    res = asyncio.run(main_async(a))
    print("TIMING_GATE " + json.dumps(res.get("summary"), default=str), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())
