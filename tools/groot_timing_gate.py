"""SONIC's timing gate with the GR00T client active on the MAIN box (M2b; docs/walk_diagnosis.md "Rules"; PLAN §0.10
b, d; PLAN §0.12: the PolicyServer runs on the DEV box, reached through the OD3 link, so what GR00T adds to the main
box is the client side: P1 rendering the Arena `ego_view`, the client decoding the frame and building/serializing a
0.92 MB request, the link's ssh encrypting it, and while standing the body's `arm` op playing the chunks). Phases, in
interleaved rounds (slow drifts on the box do not favour one phase):

    stand_base     standing, no GR00T client, the stack's camera rates (head 30 Hz)
    stand_lowhead  standing, no GR00T client, the head camera at --head-hz (the session's head rate, isolated)
    stand_client   standing, the GR00T client loop only (ego_view on at --ego-hz, the head at --head-hz,
                   observation + get_action at the executor's 2.5 Hz on real frames, frame-synced, the chunk built and
                   dropped): the client-side load without arm motion. Started before the phase's window opens, so the
                   window is the steady state (the warm-up burst is in stand_groot)
    stand_groot    standing, one real groot_arms session (real chunks into the real arm op, the executor's own camera
                   handling with session_head_hz = --head-hz; the grasp judge is off: a load run, not a pick)
    walk_base      SONIC walking the rtf_cameras pattern (walk 0.4 m/s 4 s, 2 x 90 deg, walk back, 2 x 90 deg)
    walk_groot     the same walk with the GR00T client loop running (informational: `manipulate` never runs while
                   walking, rule C7, so GR00T never infers while walking in the product)

Why the head rate matters: P1 renders EVERY enabled camera product at every render call (docs/viz.md §4), so an
enabled ego_view is rendered at the head camera's rate (30 Hz in the stack) whatever its own rate is, and each render
blocks P1's physics loop for its duration. A render longer than the slack before the deploy's next leg target leaves a
20 ms sim window without a new target (and the next one with two): `empties_per_render` below.

Per phase, the walk_diagnosis gate (rule 3: INVALID if irregular > 0.15, RTF < 0.98, or heartbeats during motion):
  - RTF: gt.pose rtf over 1 s windows (p10 / mean / min / share < 0.98; the bar is p10 >= 0.98, PLAN §0.10 b);
  - irregular leg-target windows: P1's `record` trace (one sample per gt.pose tick = 20 ms of sim time, the leg
    targets P1 applied); `no_new_leg_target_frac` = the share of consecutive 20 ms sim windows whose 12 leg targets
    did not change (exact), `irregular_est` = 2 x that (every empty window is followed by a double when the deploy
    and sim rates match; a window with two new targets is invisible in the trace), gated at 0.15; with P1's render
    count in the phase, `empties_per_render`, and the P1 loop's wall time per 20 ms sim window (trace t_wall);
  - heartbeats: P1 lowstate heartbeat publishes in the phase (> 0 fails the gate), with the P1 render hitches
    (get_stats hitches_last) that fell in the phase, so a heartbeat can be traced to its cause;
  - the deploy's own loop timing ("Loop timing" lines, one per 50 ticks, g1_deploy_onnx_ref.cpp ~4080-4100):
    Obs 2 Motor Command and Policy (TensorRT) p50/p90/max in us, LowState age in ms (the walk_diagnosis numbers:
    quiet obs->cmd p90 0.4-0.5 ms, TensorRT p90 0.1 ms; contended 3-7 ms and 4 ms);
  - also P1 overruns/lost_s/hitches, the g1_debug cadence at the client, get_action latency (through the link),
    falls, the GPU's compute processes before and after (rule 1: a local PolicyServer would be one), and where the
    PolicyServer is (a live local server, else the OD3 link: host, the tunnel ssh's CPUs and nice).
The tool pins itself (--cpus, default 4-15: never the deploy's 0-3, like P5 under m2_p5.sh); run it with the
client's thread caps (OMP_NUM_THREADS=1 etc., scripts/groot_gate.sh). Deliberate GR00T-load measurement (PLAN
§0.10 d).

    WL_PORT_OFFSET=0 .venv-rt/bin/python -m tools.groot_timing_gate --out outputs/m2b_finish/gmain/timing-<ts> \\
        [--rounds 3] [--phases stand_base,stand_lowhead,stand_client,stand_groot] [--stand-s 20] [--walk-s 40] \\
        [--ego-hz 2.5] [--ego-warm-hz 10] [--head-hz 2.5] [--no-frame-sync] [--cpus 4-15] \\
        [--endpoint tcp://127.0.0.1:5550] [--deploy-log PATH] [--no-home] [--dodge-stall]
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

PHASES = ("stand_base", "stand_lowhead", "stand_client", "stand_groot", "walk_base", "walk_groot")
CLIENT_PHASES = ("stand_client", "walk_groot")          # the client loop runs through the whole phase window
LOWHEAD_PHASES = ("stand_lowhead", "stand_client", "stand_groot", "walk_groot")   # the head at --head-hz
GATE = {"rtf_p10_min": 0.98, "irregular_max": 0.15, "heartbeats_max": 0}
LOOP_TIMING = {"obs2cmd_us": r"Obs 2 Motor Command: (\d+)us", "policy_us": r"Policy: (\d+)us",
               "obs_us": r"Obs: (\d+)us", "lowstate_age_ms": r"LowState age: ([\d.]+)ms"}
N_LEG = 12                  # the first 12 motors (MuJoCo order) are the legs


def newest_deploy_log(session: str | None = None) -> str | None:
    pats = [f"/work/logs/wl/{session}-*-deploy.log"] if session else []
    for pat in pats + ["/work/logs/wl/*-deploy.log"]:
        c = sorted(glob.glob(pat), key=os.path.getmtime)
        if c:
            return c[-1]
    return None


class DeployLog:
    """The deploy's "Loop timing" lines appended between two byte offsets of its log (one line per 50 ticks)."""

    def __init__(self, path: str | None):
        self.path = path

    def offset(self) -> int | None:
        try:
            return os.path.getsize(self.path) if self.path else None
        except OSError:
            return None

    def window(self, o0: int | None, o1: int | None) -> dict:
        if not self.path or o0 is None or o1 is None or o1 <= o0:
            return {"n": 0, "log": self.path}
        with open(self.path, "rb") as f:
            f.seek(o0)
            txt = f.read(o1 - o0).decode(errors="replace")
        lines = [l for l in txt.splitlines() if "Loop timing" in l]
        out: dict[str, Any] = {"n": len(lines)}
        for k, pat in LOOP_TIMING.items():
            v = np.array([float(x) for x in re.findall(pat, "\n".join(lines))])
            if v.size:
                out[k] = {"p50": round(float(np.percentile(v, 50)), 2), "p90": round(float(np.percentile(v, 90)), 2),
                          "max": round(float(v.max()), 2)}
        return out


def leg_target_windows(path: str, wait_s: float = 5.0) -> dict:
    """walk_diagnosis rule 3's `irregular` from P1's record trace (t_sim, q_target per gt.pose tick = 20 ms sim)."""
    t_end = time.monotonic() + wait_s
    while not os.path.exists(path) and time.monotonic() < t_end:   # P1 writes it off its physics thread
        time.sleep(0.1)
    if not os.path.exists(path):
        return {"error": f"no trace at {path}"}
    z = np.load(path, allow_pickle=False)
    ts, qt = np.asarray(z["t_sim"], dtype=np.float64), np.asarray(z["q_target"], dtype=np.float64)
    if ts.size < 10:
        return {"n": int(ts.size)}
    d_sim = np.diff(ts)
    ok20 = np.abs(d_sim - 0.02) < 0.004                  # consecutive 20 ms sim samples
    changed = np.any(qt[1:, :N_LEG] != qt[:-1, :N_LEG], axis=1)
    zero = float(np.mean(~changed[ok20])) if ok20.any() else None
    out = {"n_windows": int(ok20.sum()), "n_samples": int(ts.size),
           "sim_span_s": round(float(ts[-1] - ts[0]), 2),
           "no_new_leg_target_frac": None if zero is None else round(zero, 4),
           "no_new_leg_target_n": int((~changed[ok20]).sum()),
           "irregular_est": None if zero is None else round(min(1.0, 2.0 * zero), 4),
           "note": "20 ms sim windows (P1 record trace) with no new leg target; irregular_est = 2 x that"}
    if "t_wall" in z.files and ok20.any():                 # P1's loop: wall time per 20 ms sim window
        dw = np.diff(np.asarray(z["t_wall"], dtype=np.float64))[ok20] * 1000.0
        out["wall_ms_per_window"] = {"p1": round(float(np.percentile(dw, 1)), 1),
                                     "p50": round(float(np.percentile(dw, 50)), 1),
                                     "p99": round(float(np.percentile(dw, 99)), 1), "max": round(float(dw.max()), 1),
                                     "over_30ms_frac": round(float((dw > 30.0).mean()), 4)}
    return out


class StallClock:
    """P1's periodic head-render stall (docs/calibration.md "Timing gate": one render of 45-160 ms every 29.9 s of sim
    time, a lowstate heartbeat each; an isaac-side issue present with no GR00T load). A phase shorter than the period
    can be placed between two stalls, so its heartbeats measure the load and not the stall. The anchor is the last
    hitch that looks like the stall (render >= min_render_ms with the head camera), refined at every phase."""

    def __init__(self, period_s: float = 29.9, min_render_ms: float = 40.0, tol_s: float = 1.5):
        self.period, self.min_render, self.tol = period_s, min_render_ms, tol_s
        self.anchor: float | None = None
        self.seen: list[float] = []

    def observe(self, stats: dict) -> None:
        for h in stats.get("hitches_last") or []:
            t, r = h.get("t_sim"), h.get("render_ms") or 0.0
            if t is None or r < self.min_render or "head" not in (h.get("cams") or []):
                continue
            t = float(t)
            if self.anchor is None:
                self.anchor = t
            else:
                k = round((t - self.anchor) / self.period)
                if abs(t - (self.anchor + k * self.period)) <= self.tol and t >= self.anchor:
                    self.anchor = t                     # on the beat: re-anchor (absorbs a small period error)
            if t not in self.seen:
                self.seen.append(t)

    def next_after(self, t: float) -> float | None:
        if self.anchor is None:
            return None
        k = int((t - self.anchor) // self.period) + 1
        return self.anchor + k * self.period

    def wait_s(self, t_now: float, dur_s: float, lead_s: float = 1.0, margin_s: float = 1.0) -> float:
        """Sim seconds to wait so [start, start + dur] holds no predicted stall (0 if it fits now)."""
        nxt = self.next_after(t_now)
        if nxt is None or dur_s + lead_s + margin_s > self.period:
            return 0.0
        prev = nxt - self.period
        if t_now >= prev + lead_s and t_now + dur_s + margin_s <= nxt:
            return 0.0
        return max(0.0, nxt + lead_s - t_now)


def hitches_in(stats: dict, t_sim0: float | None, t_sim1: float | None) -> list[dict]:
    if t_sim0 is None or t_sim1 is None:
        return []
    return [h for h in (stats.get("hitches_last") or []) if t_sim0 <= float(h.get("t_sim", -1)) <= t_sim1]


def gpu_timeslice() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "compute-policy", "-l"], capture_output=True, text=True,
                              timeout=10).stdout.strip()[-200:] or None
    except Exception:  # noqa: BLE001
        return None


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def _affinity(pid: Any) -> str | None:
    try:
        return subprocess.run(["taskset", "-cp", str(pid)], capture_output=True, text=True,
                              timeout=5).stdout.strip().split(": ")[-1] or None
    except Exception:  # noqa: BLE001
        return None


def server_placement(port: int = 5550) -> dict:
    """Where the PolicyServer runs: a LIVE local server (its state file names a running pid), else the OD3 link
    (scripts/groot_link.sh state: the dev host, and the tunnel ssh's CPUs and nice). A stale local state file (a
    stopped server) is reported as stale, not as the placement."""
    out: dict[str, Any] = {}
    try:
        d = json.loads(Path(f"/work/logs/groot/server-{port}.json").read_text())
    except (OSError, ValueError):
        d = None
    if d and _pid_alive(d.get("pid")):
        return {"where": "local"} | {k: d.get(k) for k in ("pid", "host", "taskset", "nice", "threads", "started_utc",
                                                          "groot_sha", "ckpt")} | {"affinity_now": _affinity(d["pid"])}
    if d:
        out["stale_local_state"] = {k: d.get(k) for k in ("pid", "started_utc", "host")}
    link: dict[str, Any] = {"where": "link"}
    try:
        for line in Path(f"/work/logs/groot/link-{port}.env").read_text().splitlines():
            k, _, v = line.partition("=")
            if k == "LINK_HOST":
                link["dev_host"] = "set" if v else None           # the IP stays out of committed evidence
    except OSError:
        link["state"] = "no link state file"
    try:
        pids = subprocess.run(["pgrep", "-x", "ssh"], capture_output=True, text=True, timeout=5).stdout.split()
        for pid in pids:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if f"-L 127.0.0.1:{port}:127.0.0.1:{port} " in cmd:
                ni = subprocess.run(["ps", "-o", "ni=", "-p", pid], capture_output=True, text=True).stdout.strip()
                link.update(ssh_pid=int(pid), ssh_cpus=_affinity(pid), ssh_nice=ni)
    except Exception as e:  # noqa: BLE001
        link["ssh"] = repr(e)[:120]
    return out | link


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

    def __init__(self, sensors: Any, endpoint: str, prompt: str, replan_s: float = 0.4, timeout_s: float = 1.5,
                 max_frame_age_s: float = 0.15, frame_sync: bool = True, camera_hz: float = 2.5):
        super().__init__(daemon=True, name="groot-client-load")
        from groot.actions import to_arm_chunk
        from groot.obs import build_observation
        from groot.policy_client import PolicyClient
        self.sensors, self.prompt, self.replan_s, self.max_frame_age_s = sensors, prompt, replan_s, max_frame_age_s
        self.frame_sync, self.camera_hz = frame_sync, camera_hz
        self.client = PolicyClient(endpoint, timeout_s=timeout_s)
        self.build, self.to_chunk = build_observation, to_arm_chunk
        self.lat: list[float] = []
        self.frame_age_ms: list[float] = []
        self.wait_ms: list[float] = []
        self.errors: list[str] = []
        self.stale = 0
        self.last_tf: float | None = None
        self._run = True

    def run(self) -> None:
        try:
            while self._run:
                t0 = time.monotonic()
                frame, tf = self.sensors.ego_frame()
                if self.frame_sync and (frame is None or t0 - tf > self.max_frame_age_s
                                        or (self.last_tf is not None and tf <= self.last_tf)):
                    # on demand: the camera renders at about the inference rate; wait for its next frame
                    wait = getattr(self.sensors, "wait_frame", None)
                    if callable(wait):
                        wait(tf if frame is not None else 0.0, 1.0 / max(self.camera_hz, 0.5) + 0.3)
                    self.wait_ms.append((time.monotonic() - t0) * 1000.0)
                    frame, tf = self.sensors.ego_frame()
                dbg, td = self.sensors.debug_state()
                now = time.monotonic()
                if frame is None or dbg is None or now - tf > self.max_frame_age_s or now - td > 0.06:
                    self.stale += 1
                    time.sleep(0.02)
                    continue
                self.last_tf = tf
                self.frame_age_ms.append((now - tf) * 1000.0)
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


class HeartbeatClock(threading.Thread):
    """P1's cumulative lowstate heartbeat count from `get_health` (the cached sim.health, no computation in P1's loop)
    at `hz`, so each heartbeat gets a time (+- 1/hz) to line up with a session's phases."""

    def __init__(self, endpoint: str, hz: float = 10.0):
        super().__init__(daemon=True, name="heartbeat-clock")
        self.endpoint, self.dt = endpoint, 1.0 / hz
        self.rows: list[tuple[float, int, float]] = []          # (monotonic, heartbeat_pubs, t_sim)
        self.lock = threading.Lock()
        self._run = True

    def run(self) -> None:
        from body.p1_client import P1Rpc
        rpc = P1Rpc(self.endpoint, timeout_s=1.0)
        while self._run:
            try:
                h = rpc.call("get_health")
                with self.lock:
                    self.rows.append((time.monotonic(), int(h.get("heartbeat_pubs") or 0), float(h.get("t_sim") or 0)))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.dt)

    def beats(self, t0: float, t1: float) -> list[dict]:
        """Heartbeat increments inside [t0, t1] (monotonic), relative to t0."""
        with self.lock:
            r = [x for x in self.rows if t0 - self.dt <= x[0] <= t1]
        return [{"t_rel_s": round(b[0] - t0, 2), "n": b[1] - a[1], "t_sim": round(b[2], 2)}
                for a, b in zip(r, r[1:]) if b[1] > a[1]]


class PhaseLog:
    """The executor's manip.phase events with their monotonic times (EventSink interface: emit(type, **fields))."""

    def __init__(self):
        self.rows: list[dict] = []

    def emit(self, type: str, **fields: Any) -> dict:
        ev = {"type": type, "t": time.monotonic(), **{k: fields.get(k) for k in ("phase", "execution_id")}}
        if type in ("manip.phase", "manip.started", "manip.progress_first"):
            self.rows.append(ev)
        return ev

    def of(self, sid: str, t0: float) -> list[dict]:
        return [{"phase": r["phase"], "t_rel_s": round(r["t"] - t0, 2)} for r in self.rows if r["execution_id"] == sid]


def parse_cpus(spec: str) -> set[int]:
    """'4-15' / '4,6,8-11' -> {4, 5, ...} (taskset -c syntax)."""
    out: set[int] = set()
    for part in (x.strip() for x in spec.split(",") if x.strip()):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out:
        raise ValueError(f"empty CPU list {spec!r}")
    return out


def lat_stats(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1),
            "max": round(float(max(v)), 1)}


async def main_async(a: argparse.Namespace) -> dict:
    from api.execution import ResultHandle
    from body.client import BodyClient
    from body.config import ep
    from body.p1_client import PoseSub
    from robot.factory import build
    from services.executors.groot_arms import (BodyArmPort, GrootArmExecutor, GrootArmsConfig, ZmqSensors,
                                               _groot_helpers)
    from services.skills import load_skill_specs
    from sim.clock import SimClock
    from sim_isaac.tools.rtf_cameras import summarize, walk_pattern
    from tools.groot_live_smoke import one_epoch_space, session_job

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

    t_now = [0.0]

    def on_pose(p):
        t_now[0] = p.t_sim
        with lock:
            samples.append((p.recv_mono, p.t_sim, p.rtf, p.fallen))
    sub.on_pose = on_pose
    sub.start()
    cad = DebugCadence(ep(5557 + off))
    cad.start()
    sensors = ZmqSensors(f"tcp://127.0.0.1:{a.camera_port + off}", f"tcp://127.0.0.1:{5557 + off}",
                         a.camera_key).start()
    phases = tuple(p for p in a.phases.split(",") if p)
    bad = [p for p in phases if p not in PHASES]
    if bad:
        raise SystemExit(f"unknown phases {bad}; known: {PHASES}")
    dlog = DeployLog(a.deploy_log or newest_deploy_log(a.session))
    res: dict[str, Any] = {"tool": "groot_timing_gate", "scene": a.scene, "endpoint": a.endpoint,
                           "camera": {"port": a.camera_port, "key": a.camera_key}, "phases": phases,
                           "rounds": a.rounds, "stand_s": a.stand_s, "walk_s": a.walk_s, "gpu_apps_before": gpu_apps(),
                           "deploy_log": dlog.path, "server": server_placement(), "gate": GATE,
                           "tag": a.tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "label": "deliberate GR00T-load measurement (PLAN §0.10 d)", "runs": []}
    skill = {x.skill_id: x for x in load_skill_specs()}[a.skill]
    replan_s = 1.0 / a.replan_hz
    cfg = GrootArmsConfig(endpoint=a.endpoint, max_duration_s=a.stand_s, view_min_px=0.0, lead_s=0.15,
                          camera_hz=a.ego_hz, camera_warm_hz=a.ego_warm_hz, session_head_hz=a.head_hz,
                          frame_sync=a.frame_sync, replan_s=replan_s,
                          closure_thresh=1e9)   # a load run: grasp judge off
    res["knobs"] = {"ego_hz": a.ego_hz, "ego_warm_hz": a.ego_warm_hz, "head_hz_session": a.head_hz,
                    "replan_hz": a.replan_hz, "frame_sync": a.frame_sync, "cpus": a.cpus,
                    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "thread_caps": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                   "MKL_NUM_THREADS")},
                    "gpu_timeslice": gpu_timeslice()}
    # other runtimes on the body (a live P5 planner scans on its own and holds the arms: rule 1's "quiet box")
    try:
        st = bc.status()
        sess = (((st.get("result") or st).get("fences") or {}).get("sessions")) or []
        res["body_sessions_at_start"] = [x.get("session") for x in sess]
    except Exception as e:  # noqa: BLE001
        res["body_sessions_at_start"] = f"status failed: {e!r}"[:160]
    cams0 = (rpc.call("get_stats") or {}).get("cameras") or {}
    head_hz0 = (cams0.get("head") or {}).get("hz")
    res["head_camera"] = {"stack_hz": head_hz0, "session_hz": a.head_hz or None}

    def head(hz: float | None) -> dict | None:
        """The head camera's rate (its `default` consumer keeps it on); None leaves it."""
        if not hz:
            return None
        return rpc.call("camera", name="head", hz=float(hz))
    plog = PhaseLog()
    exe = GrootArmExecutor(world, arm=BodyArmPort.of(robot.body), sensors=sensors, cfg=cfg, gate=robot.gate,
                           helpers=_groot_helpers(), events=plog)
    hbc = HeartbeatClock(ep(5600 + off))
    hbc.start()
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

        def ego(on: bool, hz: float | None = None) -> str | None:
            try:
                if on:
                    world.enable_camera("ego_view", True, consumer="timing-gate", ttl_s=max(a.walk_s, a.stand_s) + 30,
                                        hz=float(hz or a.ego_hz))
                else:
                    world.enable_camera("ego_view", False, consumer="timing-gate")
                return None
            except Exception as e:  # noqa: BLE001 - an M1 P1: no ego_view render product to add
                return f"enable_camera unavailable: {e}"[:160]

        async def client_on(run: dict) -> ClientLoad:
            """As a session does it: ego_view on at the warm-up rate until its first frame, then the stream rate and
            the session's head rate; the client loop starts on the first fresh frame."""
            t_on = time.monotonic()
            err = await asyncio.to_thread(ego, True, a.ego_warm_hz or a.ego_hz)
            if err:
                run["ego_view"] = err
            first = None
            while time.monotonic() < t_on + 3.0:
                _, tf = sensors.ego_frame()
                if tf >= t_on - 0.05:
                    first = round(tf - t_on, 3)
                    break
                await asyncio.sleep(0.02)
            if a.ego_warm_hz and a.ego_warm_hz != a.ego_hz:
                await asyncio.to_thread(ego, True, a.ego_hz)
            run["head_set"] = await asyncio.to_thread(head, a.head_hz)
            run["ego_first_frame_s"] = first
            load = ClientLoad(sensors, a.endpoint, skill.prompt_template.format(label="potato"), replan_s=replan_s,
                              max_frame_age_s=0.15 if a.frame_sync else max(0.15, 1.5 / a.ego_hz),
                              frame_sync=a.frame_sync, camera_hz=a.ego_hz)
            load.start()
            return load

        async def client_off(run: dict, load: ClientLoad) -> None:
            load.stop()
            load.join(3.0)
            await asyncio.to_thread(ego, False)
            await asyncio.to_thread(head, head_hz0 if a.head_hz else None)
            run["client"] = {"latency_ms": lat_stats(load.lat), "frame_age_ms": lat_stats(load.frame_age_ms),
                             "frame_wait_ms": lat_stats(load.wait_ms), "errors": load.errors[:5],
                             "n_errors": len(load.errors), "obs_stale": load.stale, "inferences": len(load.lat)}

        stall = StallClock(a.stall_period)
        if a.dodge_stall:                           # find the stall's beat before the first phase (<= 1.5 periods)
            t_w = time.monotonic() + 1.5 * a.stall_period
            while time.monotonic() < t_w:
                stall.observe(rpc.call("get_stats"))
                if stall.anchor is not None and len(stall.seen) >= 1 and t_now[0] > stall.anchor + 0.5:
                    break
                await asyncio.sleep(1.0)
            res["stall_anchor_t_sim"] = stall.anchor
        for rnd in range(a.rounds):
            for phase in phases:
                await asyncio.sleep(a.settle_s)
                dur = a.stand_s if phase.startswith("stand") else a.walk_s
                waited = 0.0
                if a.dodge_stall:
                    stall.observe(rpc.call("get_stats"))
                    waited = stall.wait_s(t_now[0], dur + (0.0 if phase.endswith("_base") else 3.0))
                    if waited > 0:
                        await asyncio.sleep(waited)
                run: dict[str, Any] = {"phase": phase, "round": rnd, "t_sim_start": round(t_now[0], 2),
                                       "stall_wait_s": round(waited, 2),
                                       "stall_next_predicted": stall.next_after(t_now[0]) if a.dodge_stall else None}
                load = None
                if phase in CLIENT_PHASES:                  # the steady state: the load is on before the window
                    load = await client_on(run)
                    await asyncio.sleep(1.0)
                elif phase == "stand_lowhead":
                    run["head_set"] = await asyncio.to_thread(head, a.head_hz)
                    await asyncio.sleep(1.0)
                trace = str((out / f"trace_r{rnd}_{phase}.npz").resolve())
                rec = rpc.call("record", on=True, path=trace)
                s0 = rpc.call("get_stats")
                off0 = dlog.offset()
                with lock:
                    samples.clear()
                t0 = time.monotonic()
                if phase in ("stand_base", "stand_lowhead", "stand_client"):
                    await asyncio.sleep(a.stand_s)
                elif phase == "stand_groot":
                    n += 1
                    sid = f"tg-{n:02d}"
                    ce = max(int(robot.gate.epoch), int(getattr(robot.gate, "resume_epoch", 0) or 0)) + 1
                    ex, j = session_job(robot, sid, ce, skill, a.object, world.object(a.object).type)
                    acq = getattr(robot.body, "acquire", None) if one_epoch_space(robot) else None
                    if callable(acq):
                        await acq(ex, "ARM_STREAM")
                    o = await exe.run(j, ResultHandle(ex))
                    rel = getattr(robot.body, "release", None) if callable(acq) else None
                    if callable(rel):
                        await rel(sid)
                    run["session"] = {"status": o.status, "reason": o.reason,
                                      "inferences": o.data.get("inferences"), "chunks_sent": o.data.get("chunks_sent"),
                                      "latency_ms": o.data.get("latency_ms"), "slew_frac": o.data.get("slew_frac"),
                                      "clamped_frac": o.data.get("clamped_frac")}
                    await asyncio.to_thread(bc.stop, True, 10.0, True)      # arms back to SONIC (blend 1.5 s)
                else:
                    ops: list = []
                    t_end = time.monotonic() + a.walk_s
                    await asyncio.to_thread(walk_pattern, bc, t_end, ops)
                    run["walk_ops"] = ops
                t1 = time.monotonic()
                s1 = rpc.call("get_stats")
                off1 = dlog.offset()
                rec_off = rpc.call("record", on=False)
                if load is not None:
                    await client_off(run, load)
                elif phase == "stand_lowhead":
                    await asyncio.to_thread(head, head_hz0)
                with lock:
                    run.update(summarize(list(samples), s0, s1))
                run["g1_debug"] = cad.window(t0, t1)
                run["deploy_loop"] = dlog.window(off0, off1)
                run["legs"] = (await asyncio.to_thread(leg_target_windows, trace)) if rec.get("recording") else {
                    "error": f"P1 record refused: {rec}"[:200]}
                run["legs"]["record_samples"] = rec_off.get("samples")
                rc0, rc1 = s0.get("render_calls"), s1.get("render_calls")
                if isinstance(rc0, int) and isinstance(rc1, int):
                    run["legs"]["renders"] = rc1 - rc0
                    ne = run["legs"].get("no_new_leg_target_n")
                    if ne is not None and rc1 > rc0:
                        run["legs"]["empties_per_render"] = round(ne / (rc1 - rc0), 3)
                run["cameras_during"] = {k: {kk: v.get(kk) for kk in ("on", "hz", "pub_hz")}
                                         for k, v in (s1.get("cameras") or {}).items()}
                run["hitches_in_phase"] = hitches_in(s1, s0.get("t_sim"), s1.get("t_sim"))
                run["heartbeat_times"] = hbc.beats(t0, t1)
                if phase == "stand_groot":
                    run["session_phases"] = plog.of(sid, t0)
                stall.observe(s1)
                run["periodic_stall_in_phase"] = [t for t in stall.seen
                                                  if (s0.get("t_sim") or 0) <= t <= (s1.get("t_sim") or 0)]
                moving = phase.startswith("walk")
                hb = run.get("heartbeat_pubs") or 0
                irr = run["legs"].get("irregular_est")
                run["valid"] = not (moving and hb > 0)
                run["pass_p10"] = bool((run.get("rtf1s_p10") or 0) >= GATE["rtf_p10_min"])
                run["pass_irregular"] = None if irr is None else bool(irr <= GATE["irregular_max"])
                run["pass_heartbeats"] = bool(hb <= GATE["heartbeats_max"])
                run["gate_pass"] = bool(run["pass_p10"] and run["pass_irregular"] is not False
                                        and run["pass_heartbeats"] and not run.get("fallen"))
                res["runs"].append(run)
                dl = run["deploy_loop"]
                print(f"[timing_gate] round {rnd} {phase:12s} p10 {run.get('rtf1s_p10')} mean {run.get('rtf1s_mean')} "
                      f"min {run.get('rtf1s_min')} irr~{irr} hb {hb} hitch {run.get('hitches_gt25ms')} "
                      f"o2c_p90 {(dl.get('obs2cmd_us') or {}).get('p90')}us trt_p90 {(dl.get('policy_us') or {}).get('p90')}us "
                      f"fallen {run.get('fallen')} gate {run['gate_pass']} "
                      f"lat {(run.get('client') or run.get('session') or {}).get('latency_ms')}", flush=True)
                (out / "timing_gate.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
                if run.get("fallen"):
                    raise RuntimeError("robot fell")
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)
    finally:
        res["gpu_apps_after"] = gpu_apps()
        if head_hz0 is not None and a.head_hz:
            try:
                res["head_camera"]["restored"] = head(head_hz0)
            except Exception as e:  # noqa: BLE001
                res["head_camera"]["restored"] = repr(e)[:160]
        exe.close()
        try:
            bc.stop()
        except Exception:  # noqa: BLE001
            pass
        bc.close()
        sub.stop()
        cad._run = False
        hbc._run = False
        await robot.shutdown()
        world.close()
    agg: dict[str, Any] = {}

    def pool(rows: list, key: str, stat: str) -> float | None:
        v = [((r.get("deploy_loop") or {}).get(key) or {}).get(stat) for r in rows]
        v = [x for x in v if x is not None]
        return round(float(max(v)), 2) if v else None

    for ph in phases:
        rows = [r for r in res["runs"] if r["phase"] == ph and "rtf1s_p10" in r]
        if rows:
            irr = [r["legs"].get("irregular_est") for r in rows if (r.get("legs") or {}).get("irregular_est") is not None]
            lat = [x for r in rows for x in [((r.get("client") or r.get("session") or {}).get("latency_ms") or {})]
                   if isinstance(x, dict) and x.get("p50") is not None]
            agg[ph] = {"n": len(rows), "p10_min": min(r["rtf1s_p10"] for r in rows),
                       "p10_mean": round(float(np.mean([r["rtf1s_p10"] for r in rows])), 4),
                       "mean": round(float(np.mean([r["rtf1s_mean"] for r in rows])), 4),
                       "min": min(r["rtf1s_min"] for r in rows),
                       "irregular_est_max": round(max(irr), 4) if irr else None,
                       "irregular_est_mean": round(float(np.mean(irr)), 4) if irr else None,
                       "irregular_est": irr or None,
                       "renders_per_s": [round(r["legs"]["renders"] / max(r.get("wall_s") or 1, 1e-3), 2)
                                         for r in rows if isinstance((r.get("legs") or {}).get("renders"), int)] or None,
                       "empties_per_render": [r["legs"].get("empties_per_render") for r in rows] or None,
                       "heartbeat_pubs": sum(r.get("heartbeat_pubs") or 0 for r in rows),
                       "runs_with_heartbeats": sum(1 for r in rows if (r.get("heartbeat_pubs") or 0) > 0),
                       "hitches_gt25ms": sum(r.get("hitches_gt25ms") or 0 for r in rows),
                       "overruns": sum(r.get("overruns") or 0 for r in rows),
                       "lost_s": round(sum(r.get("lost_s") or 0 for r in rows), 3),
                       "obs2cmd_us_p90_max": pool(rows, "obs2cmd_us", "p90"),
                       "obs2cmd_us_max": pool(rows, "obs2cmd_us", "max"),
                       "policy_us_p90_max": pool(rows, "policy_us", "p90"),
                       "policy_us_max": pool(rows, "policy_us", "max"),
                       "lowstate_age_ms_p90_max": pool(rows, "lowstate_age_ms", "p90"),
                       "p10": [r["rtf1s_p10"] for r in rows],
                       "get_action_ms_p50": [x["p50"] for x in lat] or None,
                       "get_action_ms_p95": [x.get("p95") for x in lat] or None,
                       "valid": all(r["valid"] for r in rows), "pass_p10": all(r["pass_p10"] for r in rows),
                       "gate_pass_runs": sum(1 for r in rows if r.get("gate_pass")),
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
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stand-s", type=float, default=20.0)
    ap.add_argument("--walk-s", type=float, default=40.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--phases", default=",".join(PHASES), help=f"comma list, order kept per round: {PHASES}")
    ap.add_argument("--session", default=os.environ.get("WL_SESSION", "wl-m2"), help="stack session (its deploy log)")
    ap.add_argument("--deploy-log", default="", help="the deploy's log (default: the session's newest *-deploy.log)")
    ap.add_argument("--tag", default="", help="free text stored in the result (e.g. the server placement)")
    ap.add_argument("--ego-hz", type=float, default=2.5, help="ego_view stream rate while GR00T holds it (the "
                    "inference rate: one frame per get_action)")
    ap.add_argument("--ego-warm-hz", type=float, default=10.0, help="ego_view rate until its first frame (P1 skips "
                    "4 warm-up frames per enable); 0 = --ego-hz")
    ap.add_argument("--head-hz", type=float, default=2.5, help="the head camera's rate while the client runs "
                    "(session_head_hz; restored after each phase); 0 = leave the stack's rate")
    ap.add_argument("--no-frame-sync", dest="frame_sync", action="store_false",
                    help="poll the newest frame instead of waiting for a fresh one")
    ap.add_argument("--cpus", default="4-15" if hasattr(os, "sched_setaffinity") else "",
                    help="CPU list this process (the client) runs on; never the deploy's 0-3; '' = leave")
    ap.add_argument("--replan-hz", type=float, default=2.5, help="GR00T inference rate (the executor's 2.5 Hz)")
    ap.add_argument("--dodge-stall", action="store_true",
                    help="place each phase between two of P1's periodic 29.9 s head-render stalls (StallClock)")
    ap.add_argument("--stall-period", type=float, default=29.9)
    ap.add_argument("--camera-port", type=int, default=5566, help="GR00T's view: P1 ego_view 5566 (M1 P1: 5565)")
    ap.add_argument("--camera-key", default="ego_view")
    a = ap.parse_args(argv)
    if a.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, parse_cpus(a.cpus))       # before any thread starts: they inherit it
    res = asyncio.run(main_async(a))
    print("TIMING_GATE " + json.dumps(res.get("summary"), default=str), flush=True)
    return 0 if "error" not in res else 1


if __name__ == "__main__":
    raise SystemExit(main())
