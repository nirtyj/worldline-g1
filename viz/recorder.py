#!/usr/bin/env python3
"""Record what the sim shows, for humans (and Claude) to check that the robot really walks.

Subscribes to the head camera (5565, gear_sonic format), the viz frames (5602: VizCams frame.chase/top/overview, or
P1's own --tp-camera frame.tp, recorded as chase), gt.pose (5601) and the body events (5611), and writes one
directory per run:

    <out>/<YYYYmmdd-HHMMSS>[-label]/
        head.mp4  chase.mp4  top.mp4  [overview.mp4]    one video per stream (top has trajectory + robot overlay)
        composite.mp4       2x2 grid (head | chase / top | overview-or-telemetry) + a header with t_sim, pose, RTF,
                            pelvis_z, fallen, body state and the last command
        telemetry.jsonl     every gt.pose, body event, frame arrival (meta only) and annotation, with wall time
        contact_sheet.png   ~12 evenly spaced composite frames, labelled with time, for a quick look
        last.jpg            the last composite frame, full size
        summary.json        counts, delivered fps per stream, duration, distance travelled, fallen, stop reason

Videos are sampled on the WALL clock at --fps (default 10): each tick writes the latest frame of every stream (a
stream that has not delivered yet is black), so all videos stay aligned. t_sim and RTF are in the overlay.

CLI (run in viz/.venv on the box):
    python viz/recorder.py --duration 30                       # record now, stop after 30 s
    python viz/recorder.py --until-event succeeded,failed,fallen   # stop 2 s after a body.event state / a fall
    python viz/recorder.py --control tcp://127.0.0.1:5630      # record now; stop via the control socket or Ctrl-C
    python viz/recorder.py --control tcp://127.0.0.1:5630 --idle   # daemon: wait for start/stop (many runs)
    python viz/recorder.py ctl tcp://127.0.0.1:5630 start|stop|status|quit|note "text"
Ports: --port-offset N (or WL_PORT_OFFSET) shifts every port; --head/--frames/--gt/--gt-rep/--body-ctl/--body-evt
override one (viz/server.py passes all of them explicitly to the recorders it spawns).

Importable:  from viz.recorder import Recorder;  r = Recorder(out_root, ports); r.start(); ...; r.stop()
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.common import (  # noqa: E402
    TopMap, decode_head, decode_jpeg, draw_robot_overlay, dumps, ep, font, frame_from_msg, occupancy_rgba,
    pose_summary, ports, same_frame, split_msg,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs" / "recordings"
TILE = (640, 360)
HEADER_H = 58
STREAMS = ("head", "chase", "top", "overview")
TOP_LONG = 768            # top.mp4 long side: fixed, so a small first image (P1 render_topdown) cannot shrink it
MIN_LONG = {"chase": 640, "overview": 640}   # a level switch (min 480x270 -> low 640x360) keeps native size
TEL_LINE_H = 20           # telemetry panel line height (15 px font)


class _Stream:
    def __init__(self, name: str):
        self.name = name
        self.jpeg: bytes | None = None
        self.meta: dict = {}
        self.rx = 0             # messages received
        self.rx_t: deque = deque(maxlen=60)
        self.img = None         # decoded PIL image of the latest frame
        self.img_rx = -1        # rx count the decoded image belongs to
        self.size: tuple[int, int] | None = None   # fixed output size for this stream's video
        self.writer = None
        self.written = 0
        self.first_tick: int | None = None
        self.swap_rb = False    # JPEG is cv2-encoded RGB (head camera, P1 frame.tp)

    def fps(self) -> float | None:
        if len(self.rx_t) < 3:
            return None
        dt = self.rx_t[-1] - self.rx_t[0]
        return (len(self.rx_t) - 1) / dt if dt > 0 else None


class Recorder:
    def __init__(self, out_root: str | Path = DEFAULT_OUT, port_map: dict | None = None, fps: float = 10.0,
                 label: str = "", head_swap_rb: bool = True, duration: float | None = None,
                 until_event: list[str] | None = None, post_roll: float = 2.0, crf: int = 23,
                 occupancy_npz: str | None = None, top_long: int = TOP_LONG, verbose: bool = True):
        self.out_root = Path(out_root)
        self.ports = port_map or ports()
        self.fps = float(fps)
        self.label = label
        self.head_swap_rb = head_swap_rb
        self.duration = duration
        self.until = [u.strip() for u in (until_event or []) if u.strip()]
        self.post_roll = post_roll
        self.crf = crf
        self.occupancy_npz = occupancy_npz
        self.top_long = int(top_long)
        self.verbose = verbose

        self.dir: Path | None = None
        self.streams = {n: _Stream(n) for n in STREAMS}
        self.pose: dict | None = None
        self.pose_n = 0
        self.traj: deque = deque(maxlen=20000)
        self.body: dict = {}          # latest body state/event summary
        self.body_log: deque = deque(maxlen=32)   # progress events of one op collapse into one entry
        self.last_cmd: dict | None = None
        self.target: list[float] | None = None
        self.fallen_ever = False
        self.dist = 0.0
        self.stop_reason: str | None = None
        self._stop_at: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self._tel = None
        self._tick = 0
        self._t0 = 0.0
        self._thumbs: list[tuple[float, Any]] = []
        self._tick_ms: list[float] = []
        self._thumb_every = 1
        self._composite = None
        self._occ = None
        self._td = None               # (PIL image, extent) of P1's cached render_topdown, if P1 has one
        self.scene_bounds: list[float] | None = None
        self.summary: dict | None = None

    # ------------------------------------------------------------------------------------------------ control
    def start(self) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.dir = self.out_root / (f"{ts}-{self.label}" if self.label else ts)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._tel = open(self.dir / "telemetry.jsonl", "w", buffering=1)
        self._t0 = time.time()
        self._log({"k": "start", "ports": self._used_ports(), "fps": self.fps, "until": self.until,
                   "duration": self.duration, "dir": str(self.dir)})
        for fn, name in ((self._rx_loop, "rec.rx"), (self._tick_loop, "rec.tick"), (self._scene_loop, "rec.scene")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self._say(f"recording -> {self.dir}")
        return self.dir

    def _used_ports(self) -> dict:
        """The ports this recorder connects to (summary/telemetry; body_ctl is recorded for reference only)."""
        return {k: self.ports[k] for k in ("head", "frames", "gt_pub", "gt_rep", "body_ctl", "body_evt") if k in self.ports}

    def _scene_loop(self) -> None:
        """Background: scene bounds + occupancy for the top-view fallback (never blocks the recording)."""
        self._fetch_scene()
        if self.occupancy_npz:
            try:
                self._occ = occupancy_rgba(self.occupancy_npz)
            except Exception as e:  # noqa: BLE001
                self._say(f"occupancy not loaded: {e}")

    def _fetch_scene(self) -> None:
        """Ask P1's REP for the house bounds (get_scene_info), its cached top-down render (render_topdown, contract
        1.6: pre-rendered at start-up, the reply only copies it) and the occupancy (get_occupancy), so the top view
        has a background before the first VizCams top frame arrives, or at all when VizCams is off / at "min"."""
        from PIL import Image

        self.scene_bounds = None
        s = zmq.Context.instance().socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(ep(self.ports["gt_rep"]))
        td_path = f"/tmp/viz_rec_topdown_{self.ports['gt_rep']}.png"
        try:
            for op in ("get_scene_info", "render_topdown", "get_occupancy"):
                s.send_string(json.dumps({"op": op, **({"path": td_path} if op == "render_topdown" else {}),
                                          "args": {}}))
                if not s.poll(3000 if op == "get_scene_info" else 15000):
                    self._say(f"P1 REP {self.ports['gt_rep']}: no reply to {op}; top view falls back to the trajectory")
                    return
                rep = json.loads(s.recv())
                self._log({"k": "p1", "op": op, "ok": rep.get("ok"),
                           **({"bounds": rep.get("bounds") or rep.get("bounds_xy"), "house_id": rep.get("house_id")}
                              if op == "get_scene_info" else {"path": rep.get("path"), "extent": rep.get("extent")})})
                if op == "render_topdown" and rep.get("ok", True) and rep.get("extent") and rep.get("path") \
                        and os.path.exists(rep["path"]):
                    try:
                        self._td = (Image.open(rep["path"]).convert("RGB"), [float(v) for v in rep["extent"]])
                    except Exception as e:  # noqa: BLE001
                        self._say(f"P1 top-down not loaded: {e}")
                if op == "get_scene_info" and rep.get("ok", True):
                    b = rep.get("bounds") or rep.get("bounds_xy")
                    if b is not None and len(b) == 2:
                        b = [b[0][0], b[0][1], b[1][0], b[1][1]]
                    self.scene_bounds = [float(v) for v in b] if b else None
                if op == "get_occupancy" and rep.get("ok", True) and rep.get("path") and not self.occupancy_npz:
                    if os.path.exists(rep["path"]):
                        self.occupancy_npz = rep["path"]
        except Exception as e:  # noqa: BLE001
            self._say(f"scene query failed: {e}")
        finally:
            s.close(0)

    def _top_fallback(self):
        """Top view without a VizCams top frame: P1's cached top-down render, else occupancy, else the scene bounds,
        with the trajectory + robot (+ go_to target) drawn live from gt.pose."""
        from PIL import Image

        if self._td is not None:
            return self._top_overlay(self._td[0], {"cam_pose": {"extent": self._td[1]}, "fallback": "p1_topdown"},
                                     occ=False)
        ext = None
        if self._occ is not None:
            ext = self._occ[1]
        elif self.scene_bounds:
            b = self.scene_bounds
            ext = [b[0] - 0.5, b[1] - 0.5, b[2] + 0.5, b[3] + 0.5]
        else:
            with self._lock:
                pts = list(self.traj)
            if not pts:
                return None
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ext = [min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3]
        ew, eh = ext[2] - ext[0], ext[3] - ext[1]
        W = 640
        H = max(2, int(W * eh / ew) // 2 * 2)
        im = Image.new("RGB", (W, H), (24, 26, 30))
        return self._top_overlay(im, {"cam_pose": {"extent": ext}, "fallback": True})

    def stop(self, reason: str = "stopped") -> dict:
        if self.stop_reason is None:
            self.stop_reason = reason
        self._stop.set()
        for t in self._threads:
            t.join(timeout=30)
        if self.summary is None:
            self.summary = self._finalize()
        self._done.set()
        return self.summary

    def wait(self) -> dict:
        """Block until a stop condition (duration / until-event / stop())."""
        while not self._stop.is_set():
            time.sleep(0.1)
        return self.stop(self.stop_reason or "stopped")

    def annotate(self, text: str | None = None, **kw) -> None:
        """Record an external note, e.g. the command the UI just sent: annotate(cmd={"op": "walk", ...})."""
        rec = {"k": "note", **({"text": text} if text else {}), **kw}
        if "cmd" in kw and isinstance(kw["cmd"], dict):
            self._set_cmd(kw["cmd"])
        self._log(rec)

    def _set_cmd(self, cmd: dict, accepted: bool = False) -> None:
        """Last command (header + panel) and the go_to target (top view). Sources: UI notes, and the body's
        'accepted' event, which carries data.args for every op (body/service.py _start_motion), so commands from
        the CLI, /api/cmd or BodyClient show up too. The target stays after the go_to ends (final pose vs target)
        and is cleared when the body accepts any other motion op."""
        with self._lock:
            self.last_cmd = cmd
            if cmd.get("op") == "go_to":
                a = cmd.get("args") or {}
                try:
                    self.target = [float(a["x"]), float(a["y"])]
                except (KeyError, TypeError, ValueError):
                    pass
            elif accepted:
                self.target = None

    def status(self) -> dict:
        el = time.time() - self._t0 if self._t0 else 0.0
        return {
            "recording": bool(self.dir) and not self._stop.is_set(), "dir": str(self.dir) if self.dir else None,
            "elapsed_s": round(el, 1), "ticks": self._tick, "pose_msgs": self.pose_n,
            "streams": {n: {"rx": s.rx, "fps": round(s.fps(), 1) if s.fps() else None} for n, s in
                        self.streams.items()},
            "stop_reason": self.stop_reason,
        }

    # ------------------------------------------------------------------------------------------------ internals
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[recorder] {msg}", flush=True)

    def _log(self, rec: dict) -> None:
        rec.setdefault("tw", round(time.time(), 4))
        if self._tel is not None:
            with self._lock:
                try:
                    self._tel.write(dumps(rec) + "\n")
                except ValueError:
                    pass

    def _trigger(self, reason: str) -> None:
        if self._stop_at is None:
            self._stop_at = time.time() + self.post_roll
            self.stop_reason = reason
            self._say(f"stop condition: {reason} (post-roll {self.post_roll:.1f} s)")

    def _rx_loop(self) -> None:
        ctx = zmq.Context.instance()
        socks = {}

        def sub(name: str, port: int, topics: list[bytes], conflate: bool = False):
            s = ctx.socket(zmq.SUB)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVHWM, 50)
            if conflate:
                s.setsockopt(zmq.CONFLATE, 1)
            for t in topics:
                s.setsockopt(zmq.SUBSCRIBE, t)
            s.connect(ep(port))
            socks[s] = name

        sub("head", self.ports["head"], [b""], conflate=True)
        sub("frames", self.ports["frames"], [b"frame."])
        sub("gt", self.ports["gt_pub"], [b"gt."])
        sub("body", self.ports["body_evt"], [b""])
        poller = zmq.Poller()
        for s in socks:
            poller.register(s, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                for s, _ in poller.poll(100):
                    name = socks[s]
                    while True:
                        try:
                            frames = s.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            self._on_msg(name, frames)
                        except Exception as e:  # noqa: BLE001
                            self._say(f"bad {name} message: {e}")
                if self.duration and time.time() - self._t0 >= self.duration:
                    self.stop_reason = self.stop_reason or f"duration {self.duration:g}s"
                    self._stop.set()
                if self._stop_at is not None and time.time() >= self._stop_at:
                    self._stop.set()
        finally:
            for s in socks:
                s.close(0)

    def _on_msg(self, name: str, frames: list[bytes]) -> None:
        now = time.time()
        if name == "head":
            jpeg, meta = decode_head(frames[-1])
            if jpeg is None:
                return
            st = self.streams["head"]
            with self._lock:
                st.jpeg, st.meta = jpeg, meta
                st.rx += 1
                st.rx_t.append(now)
            self._log({"k": "frame", "stream": "head", "t_sim": meta.get("t_sim"), "seq": meta.get("seq"),
                       "bytes": len(jpeg)})
            return
        topic, msg = split_msg(frames)
        if name == "frames":
            got = frame_from_msg(topic, msg)
            if got is None:
                return
            cam, jpeg, msg, swap = got
            st = self.streams.get(cam)
            if st is None:
                st = self.streams[cam] = _Stream(cam)
            if st.jpeg is not None and same_frame(st.meta, msg):
                return   # VizCams re-send of the snapshot we already hold
            with self._lock:
                st.jpeg, st.meta, st.swap_rb = jpeg, msg, swap
                st.rx += 1
                st.rx_t.append(now)
            self._log({"k": "frame", "stream": cam, "src": msg.get("source"), "t_sim": msg.get("t_sim"),
                       "seq": msg.get("seq"), "bytes": len(jpeg), "w": msg.get("w"), "h": msg.get("h")})
        elif name == "gt":
            if topic == "gt.pose" and isinstance(msg, dict):
                p = pose_summary(msg)
                with self._lock:
                    self.pose, self.pose_n = p, self.pose_n + 1
                    if p and p.get("base_pos"):
                        x, y = p["base_pos"][0], p["base_pos"][1]
                        if not self.traj or math.hypot(x - self.traj[-1][0], y - self.traj[-1][1]) > 0.02:
                            if self.traj:
                                self.dist += math.hypot(x - self.traj[-1][0], y - self.traj[-1][1])
                            self.traj.append((x, y))
                    if p and p.get("fallen"):
                        self.fallen_ever = True
                self._log({"k": "pose", **msg})
                if p and p.get("fallen") and "fallen" in self.until:
                    self._trigger("fallen")
            else:
                self._log({"k": "gt", "topic": topic, "msg": msg})
        elif name == "body":
            if not isinstance(msg, dict):
                msg = {"raw": str(msg)[:200]}
            if topic == "body.state":  # 5 Hz health + active op (body/service.py status())
                act = msg.get("active") or {}
                with self._lock:
                    self.body = {"op": act.get("op"), "phase": act.get("phase"), "fault": msg.get("fault"),
                                 "in_control": msg.get("in_control"), "event": self.body.get("event")}
                if self._tick % 5 == 0:
                    self._log({"k": "body_state", "msg": msg})
                return
            ev = {"t": round(now - self._t0, 1), "id": msg.get("id"), "op": msg.get("op"),
                  "state": msg.get("state"), "n": 1}
            with self._lock:
                self.body["event"] = ev
                last = self.body_log[-1] if self.body_log else None
                if (ev["state"] == "progress" and last is not None and last.get("state") == "progress"
                        and last.get("id") == ev["id"]):
                    last["n"] += 1          # keep accepted / succeeded visible: one line per run of progress
                    last["t"] = ev["t"]
                else:
                    self.body_log.append(ev)
            self._log({"k": "body", "topic": topic, "msg": msg})
            state = str(msg.get("state") or "")
            op = str(msg.get("op") or "")
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            if state == "accepted":
                self._set_cmd({"op": op, "args": data.get("args") if isinstance(data.get("args"), dict) else {},
                               "id": msg.get("id")}, accepted=True)
            for u in self.until:
                if u and u != "fallen" and (u == state or u == f"{op}:{state}" or u == topic):
                    self._trigger(f"event {op}:{state}")

    # ------------------------------------------------------------------------------------------------ video
    def _open_writer(self, path: Path, size: tuple[int, int]):
        import imageio_ffmpeg

        gen = imageio_ffmpeg.write_frames(
            str(path), size, fps=self.fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=2, ffmpeg_log_level="error",
            output_params=["-preset", "veryfast", "-crf", str(self.crf), "-movflags", "+faststart"])
        gen.send(None)
        return gen

    def _stream_image(self, st: _Stream):
        """Latest decoded frame of a stream (decode only when a new frame arrived)."""
        with self._lock:
            jpeg, rx, meta = st.jpeg, st.rx, dict(st.meta)
        if jpeg is None:
            return None, meta
        if st.img_rx != rx:
            try:
                st.img = decode_jpeg(jpeg, swap_rb=(st.name == "head" and self.head_swap_rb) or st.swap_rb)
                st.img_rx = rx
            except Exception as e:  # noqa: BLE001
                self._say(f"decode {st.name}: {e}")
        return st.img, meta

    def _top_overlay(self, im, meta: dict, occ: bool = True):
        ext = (meta.get("cam_pose") or {}).get("extent")
        if not ext:
            return im
        im = im.copy()
        tm = TopMap(ext, im.width, im.height)
        with self._lock:
            traj, pose, tgt = list(self.traj), self.pose, self.target
        if occ and self._occ is not None:
            occ_im, occ_ext = self._occ
            # paste the occupancy map scaled into the frame's extent
            u0, v0 = tm.to_px(occ_ext[0], occ_ext[3])
            u1, v1 = tm.to_px(occ_ext[2], occ_ext[1])
            if u1 - u0 > 2 and v1 - v0 > 2:
                o = occ_im.resize((int(u1 - u0), int(v1 - v0)))
                im.paste(o, (int(u0), int(v0)), o)
        draw_robot_overlay(im, tm, traj, pose, tgt)
        return im

    def _video_size(self, name: str, wh: tuple[int, int]) -> tuple[int, int]:
        """Fixed size of a stream's video, chosen at its first frame. top: long side = top_long whatever the first
        image is (P1's render_topdown fallback is ~350 px, VizCams snapshots 512-1024 px); chase/overview: at least
        MIN_LONG; everything: at most 1280 wide. Later frames of another size are letterboxed into it."""
        w, h = wh
        if name == "top":
            s = self.top_long / max(w, h)
        else:
            s = max(1.0, MIN_LONG.get(name, 0) / max(w, h))
        if w * s > 1280:
            s = 1280 / w
        w, h = int(round(w * s)), int(round(h * s))
        return max(2, w - w % 2), max(2, h - h % 2)

    def _fit(self, im, size):
        """Letterbox `im` into `size` (w, h)."""
        from PIL import Image

        W, H = size
        if im.size == (W, H):
            return im
        s = min(W / im.width, H / im.height)
        nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
        out = Image.new("RGB", (W, H), (12, 12, 14))
        out.paste(im.resize((nw, nh)), ((W - nw) // 2, (H - nh) // 2))
        return out

    def _tick_loop(self) -> None:
        period = 1.0 / self.fps
        nxt = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < nxt:
                time.sleep(min(period, nxt - now))
                continue
            nxt += period
            if now - nxt > 1.0:  # fell far behind (slow disk/CPU): skip ahead, do not burst
                nxt = now + period
            t0 = time.perf_counter()
            with self._lock:
                started = self.pose is not None or any(st.jpeg is not None for st in self.streams.values())
            if not started and now - self._t0 < 3.0:  # no data yet: do not open the videos with black frames
                continue
            try:
                self._write_tick()
            except Exception as e:  # noqa: BLE001
                self._say(f"tick failed: {e}")
            self._tick_ms.append((time.perf_counter() - t0) * 1000.0)
            self._tick += 1

    def _write_tick(self) -> None:
        from PIL import Image

        tiles = {}
        for name, st in list(self.streams.items()):
            im, meta = self._stream_image(st)
            if name == "top":
                # a real top frame (+ live overlay), else occupancy/bounds + trajectory, so top.mp4 is useful
                # even when VizCams is off
                im = self._top_overlay(im, meta) if im is not None else self._top_fallback()
            if im is None:
                continue
            if st.size is None:
                st.size = self._video_size(name, im.size)
                st.first_tick = self._tick
                st.writer = self._open_writer(self.dir / f"{name}.mp4", st.size)
                black = np.zeros((st.size[1], st.size[0], 3), np.uint8)
                for _ in range(self._tick):  # keep this video aligned with the others
                    st.writer.send(black)
                st.written += self._tick
            fr = im if im.size == st.size else self._fit(im, st.size)
            st.writer.send(np.asarray(fr, dtype=np.uint8))
            st.written += 1
            tiles[name] = im
        comp = self._compose(tiles)
        if not hasattr(self, "_comp_writer"):
            self._comp_writer = self._open_writer(self.dir / "composite.mp4", comp.size)
            self._comp_written = 0
        self._comp_writer.send(np.asarray(comp, dtype=np.uint8))
        self._comp_written += 1
        self._composite = comp
        if self._tick % self._thumb_every == 0:
            self._thumbs.append((time.time() - self._t0, comp.resize((480, int(480 * comp.height / comp.width)),
                                                                     Image.BILINEAR)))
            if len(self._thumbs) > 96:
                self._thumbs = self._thumbs[::2]
                self._thumb_every *= 2

    def _compose(self, tiles: dict):
        from PIL import Image, ImageDraw

        W, H = TILE
        comp = Image.new("RGB", (2 * W, 2 * H + HEADER_H), (18, 18, 22))
        slots = [("head", 0, 0), ("chase", W, 0), ("top", 0, H)]
        br = "overview" if "overview" in tiles else None
        if br:
            slots.append(("overview", W, H))
        d = ImageDraw.Draw(comp)
        f_small, f_big = font(15), font(19)
        for name, x, y in slots:
            if name in tiles:
                comp.paste(self._fit(tiles[name], TILE), (x, y + HEADER_H))
            else:
                d.text((x + W // 2 - 90, y + HEADER_H + H // 2 - 10), f"no {name} frames yet", fill=(120, 120, 130),
                       font=f_big)
            st = self.streams[name]
            fps = st.fps()
            age = time.time() - st.rx_t[-1] if st.rx_t else None
            cap = f"{name.upper()}  {fps:.1f} fps" if fps else name.upper()
            snap = bool(st.meta.get("snapshot")) if st.meta else False
            if name == "top" and st.jpeg is None and name in tiles:
                cap = ("TOP  P1 top-down render, robot live" if self._td is not None
                       else "TOP  occupancy + trajectory (no top frame yet)")
            elif snap and age is not None:
                cap += f"  snapshot {age:.0f}s old, robot live"
            elif age is not None and age > 1.5:
                cap += f"  STALE {age:.0f}s"
            tsim = st.meta.get("t_sim") if st.meta else None
            if tsim is not None:
                cap += f"  t_sim {float(tsim):.2f}"
            d.rectangle([x, y + HEADER_H, x + 8 + int(d.textlength(cap, font=f_small)) + 8, y + HEADER_H + 22],
                        fill=(0, 0, 0))
            d.text((x + 8, y + HEADER_H + 3), cap, fill=(230, 230, 230), font=f_small)
        if not br:
            self._telemetry_panel(d, W, H + HEADER_H, W, H, f_small, f_big)
        # header band: line 1 pose/time, line 2 body state + last command (left) and the source's note (right,
        # e.g. "VIZ TEST: kinematic"), so no overlay ever covers a tile or the telemetry panel
        with self._lock:
            p, body, cmd = self.pose, dict(self.body), self.last_cmd
        el = time.time() - self._t0
        parts = [f"rec {el:6.1f}s", time.strftime("%H:%M:%S")]
        if p:
            if p.get("t_sim") is not None:
                parts.append(f"t_sim {p['t_sim']:.2f}")
            if p.get("rtf") is not None:
                parts.append(f"RTF {p['rtf']:.2f}")
            if p.get("base_pos"):
                bp = p["base_pos"]
                parts.append(f"pos ({bp[0]:+.2f}, {bp[1]:+.2f})")
            if p.get("yaw") is not None:
                parts.append(f"yaw {math.degrees(p['yaw']):+5.0f}°")
            if p.get("pelvis_z") is not None:
                parts.append(f"pelvis_z {p['pelvis_z']:.2f}")
            parts.append("FALLEN" if p.get("fallen") else "upright")
        else:
            parts.append("no gt.pose")
        d.text((10, 6), "   ".join(parts), fill=(255, 90, 90) if p and p.get("fallen") else (235, 235, 240),
               font=f_big)
        right_edge = 2 * W - 6
        if p and p.get("note"):
            note = str(p["note"])[:60]
            tw = d.textlength(note, font=f_small)
            d.rectangle([right_edge - tw - 16, 30, right_edge, 53], fill=(120, 60, 0))
            d.text((right_edge - tw - 8, 33), note, fill=(255, 220, 150), font=f_small)
            right_edge -= tw + 28
        line2 = ""
        if body:
            line2 = f"body: {body.get('op') or 'idle'}" + (f"/{body['phase']}" if body.get("phase") else "")
            if body.get("fault"):
                line2 += f" FAULT {body['fault']}"
            ev = body.get("event")
            if ev:
                line2 += f"  [{ev.get('op')}:{ev.get('state')}]"
        if cmd:
            line2 += f"   cmd: {cmd.get('op')} {json.dumps(cmd.get('args') or {}, separators=(',', ':'))[:48]}"
        line2 = line2.strip()
        while line2 and d.textlength(line2, font=f_small) > right_edge - 10:
            line2 = line2[:-2]
        if line2:
            d.text((10, 33), line2, fill=(150, 220, 255), font=f_small)
        return comp

    def _telemetry_panel(self, d, x, y, W, H, f_small, f_big) -> dict:
        """Bottom-right tile when there is no overview. Lines are fitted to the tile: the pose block, the last
        command and the go_to target stay; the body-event list shows as many of the newest events as fit (runs of
        'progress' events are one line with a count). Returns the layout for tests."""
        with self._lock:
            p, log, cmd, tgt = self.pose, [dict(e) for e in self.body_log], self.last_cmd, self.target
        fixed = []
        if p:
            fc = p.get("foot_contact") or {}
            fixed += [
                f"t_sim     {p.get('t_sim') or 0:.2f} s      RTF {p.get('rtf') or 0:.2f}",
                f"base_pos  {', '.join(f'{v:+.2f}' for v in (p.get('base_pos') or []))}",
                f"yaw       {math.degrees(p.get('yaw') or 0):+.1f} deg",
                f"pelvis_z  {p.get('pelvis_z') or 0:.3f} m   fallen {bool(p.get('fallen'))}",
                f"feet      L {'on' if fc.get('left') else '--'}   R {'on' if fc.get('right') else '--'}"
                f"   band {p.get('band')}",
                f"distance  {self.dist:.2f} m   fallen ever {self.fallen_ever}",
            ]
        else:
            fixed.append("no gt.pose yet")
        if cmd:
            fixed.append(f"last cmd  {cmd.get('op')} {json.dumps(cmd.get('args') or {}, separators=(',', ':'))[:44]}")
        if tgt:
            bp = (p or {}).get("base_pos")
            dist = f"   {math.hypot(tgt[0] - bp[0], tgt[1] - bp[1]):.2f} m away" if bp else ""
            fixed.append(f"target    ({tgt[0]:+.2f}, {tgt[1]:+.2f}){dist}")
        ev_lines = []
        for e in log:
            n = f" x{e['n']}" if e.get("n", 1) > 1 else ""
            eid = str(e.get("id") or "").rsplit("-", 1)[-1][:10]     # ui-go_to-1a2b3c4d -> 1a2b3c4d
            ev_lines.append(f"  {e['t']:6.1f}s  {e.get('op') or ''}  {e.get('state') or ''}{n}  {eid}"[:60])
        top_y = y + 10
        first = top_y + 28                       # first line under the title
        bottom = y + H - 6                       # last pixel a line may use
        room = (bottom - first) // TEL_LINE_H    # lines that fit
        n_ev = max(1, room - len(fixed) - 1)     # "body events:" takes one line
        if len(fixed) + 1 + n_ev > room:          # (never with the fixed block above: 8 + 1 + 6 = 15 = room)
            fixed = fixed[: max(0, room - 1 - n_ev)]
        shown = ev_lines[-n_ev:] if ev_lines else ["  (none)"]
        hidden = max(0, len(ev_lines) - n_ev)
        lines = fixed + [f"body events:{f'  ({hidden} older)' if hidden else ''}"] + shown
        d.text((x + 16, top_y), "TELEMETRY", fill=(150, 220, 255), font=f_big)
        yy = first
        for ln in lines:
            d.text((x + 16, yy), ln, fill=(200, 200, 210), font=f_small)
            yy += TEL_LINE_H
        return {"lines": lines, "last_line_bottom": yy - TEL_LINE_H + 18, "tile_bottom": y + H,
                "events_shown": len(shown), "events_hidden": hidden}

    def _finalize(self) -> dict:
        from PIL import Image, ImageDraw

        for st in self.streams.values():
            if st.writer is not None:
                try:
                    st.writer.close()
                except Exception as e:  # noqa: BLE001
                    self._say(f"close {st.name}: {e}")
        if hasattr(self, "_comp_writer"):
            self._comp_writer.close()
        sheet_path = None
        if self._thumbs:
            n = min(12, len(self._thumbs))
            idx = [round(i * (len(self._thumbs) - 1) / max(1, n - 1)) for i in range(n)]
            pics = [self._thumbs[i] for i in idx]
            tw, th = pics[0][1].size
            cols = 4 if n > 6 else min(n, 3)
            rows = math.ceil(n / cols)
            sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (10, 10, 12))
            d = ImageDraw.Draw(sheet)
            for k, (t, im) in enumerate(pics):
                cx, cy = (k % cols) * tw, (k // cols) * (th + 22)
                sheet.paste(im, (cx, cy + 22))
                d.text((cx + 6, cy + 3), f"#{k + 1}  t_rec {t:5.1f}s", fill=(230, 230, 230), font=font(14))
            sheet_path = self.dir / "contact_sheet.png"
            sheet.save(sheet_path)
        if self._composite is not None:
            self._composite.save(self.dir / "last.jpg", quality=90)
        dur = time.time() - self._t0
        streams = {}
        for n, st in self.streams.items():
            f = self.dir / f"{n}.mp4"
            streams[n] = {"rx": st.rx, "rx_fps": round(st.rx / dur, 2) if dur > 0 else None,
                          "video": str(f) if st.writer is not None else None, "size": st.size,
                          "frames_written": st.written, "first_tick": st.first_tick}
        with self._lock:
            pose = self.pose
        summary = {
            "dir": str(self.dir), "duration_s": round(dur, 2), "fps": self.fps, "ticks": self._tick,
            "composite_frames": getattr(self, "_comp_written", 0), "stop_reason": self.stop_reason,
            "tick_ms": ({"mean": round(float(np.mean(self._tick_ms)), 1),
                         "p95": round(float(np.percentile(self._tick_ms, 95)), 1)} if self._tick_ms else None),
            "streams": streams, "pose_msgs": self.pose_n, "distance_m": round(self.dist, 3),
            "fallen_ever": self.fallen_ever, "last_pose": pose, "contact_sheet": str(sheet_path) if sheet_path else None,
            "ports": self._used_ports(),
        }
        self._log({"k": "stop", **{k: v for k, v in summary.items() if k != "streams"}})
        with open(self.dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        if self._tel:
            self._tel.close()
            self._tel = None
        self._say(f"done: {self.dir} ({dur:.1f} s, {summary['composite_frames']} composite frames, "
                  "streams " + ", ".join(f"{k}:{v['rx']}" for k, v in streams.items()) + f", reason: {self.stop_reason})")
        return summary


# ---------------------------------------------------------------------------------------------------- control
def ctl(addr: str, op: str, timeout_s: float = 60.0, **kw) -> dict:
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.connect(addr)
    try:
        s.send_string(json.dumps({"op": op, **kw}))
        return json.loads(s.recv_string())
    finally:
        s.close(0)


def serve(args, port_map: dict) -> int:
    """Run with a control REP socket. Without --idle a recording starts immediately."""
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(args.control)
    cur: Recorder | None = None
    last: dict | None = None
    quit_ = threading.Event()

    def new() -> Recorder:
        return Recorder(args.out, port_map, fps=args.fps, label=args.label, head_swap_rb=not args.no_head_swap,
                        duration=args.duration, until_event=args.until_event, post_roll=args.post_roll,
                        crf=args.crf, occupancy_npz=args.occupancy, top_long=args.top_long)

    signal.signal(signal.SIGINT, lambda *_: quit_.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_.set())
    if not args.idle:
        cur = new()
        cur.start()
    print(f"[recorder] control on {args.control} ({'idle' if args.idle else 'recording'})", flush=True)
    while not quit_.is_set():
        if cur is not None and cur._stop.is_set() and cur.summary is None:
            last = cur.stop()
            cur = None
            if not args.idle:
                break
        if not rep.poll(100):
            continue
        try:
            req = json.loads(rep.recv_string())
        except Exception as e:  # noqa: BLE001
            rep.send_string(json.dumps({"ok": False, "error": str(e)}))
            continue
        op = req.get("op")
        try:
            if op == "start":
                if cur is None:
                    cur = new()
                    if req.get("label"):
                        cur.label = str(req["label"])
                    cur.start()
                rep.send_string(dumps({"ok": True, **cur.status()}))
            elif op == "stop":
                if cur is not None:
                    last = cur.stop("stop via control")
                    cur = None
                rep.send_string(dumps({"ok": True, "summary": last}))
                if not args.idle:
                    break
            elif op == "status":
                rep.send_string(dumps({"ok": True, **(cur.status() if cur else {"recording": False}),
                                       "last": last}))
            elif op in ("note", "annotate"):
                if cur is not None:
                    cur.annotate(req.get("text"), **{k: v for k, v in req.items() if k not in ("op", "text")})
                rep.send_string(dumps({"ok": True}))
            elif op == "quit":
                if cur is not None:
                    last = cur.stop("quit")
                    cur = None
                rep.send_string(dumps({"ok": True, "summary": last}))
                break
            else:
                rep.send_string(dumps({"ok": False, "error": f"unknown op {op!r}"}))
        except Exception as e:  # noqa: BLE001
            rep.send_string(dumps({"ok": False, "error": str(e)}))
    if cur is not None:
        cur.stop("signal")
    rep.close(0)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "ctl":
        if len(argv) < 3:
            print("usage: recorder.py ctl ADDR start|stop|status|quit|note TEXT [--label L]")
            return 2
        addr, op = argv[1], argv[2]
        kw: dict[str, Any] = {}
        if op in ("note", "annotate"):
            kw["text"] = " ".join(argv[3:])
        elif len(argv) > 4 and argv[3] == "--label":
            kw["label"] = argv[4]
        print(json.dumps(ctl(addr, op, **kw), indent=2))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--label", default="")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many wall seconds")
    ap.add_argument("--until-event", type=lambda s: s.split(","), default=None,
                    help="comma list: body.event state (succeeded), op:state (go_to:succeeded), or 'fallen'")
    ap.add_argument("--post-roll", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--control", default=None, help="REP address for start/stop/status/note/quit")
    ap.add_argument("--idle", action="store_true", help="with --control: wait for 'start' (daemon, many runs)")
    ap.add_argument("--no-head-swap", action="store_true",
                    help="do not swap R/B of the head JPEG (the contract publishes cv2-encoded RGB -> swapped)")
    ap.add_argument("--occupancy", default=None, help="P1 occupancy npz to draw under the top view")
    ap.add_argument("--top-long", type=int, default=TOP_LONG, help="top.mp4 long side in px (fixed for the run)")
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--head", type=int, default=None, help="head camera SUB port (5565)")
    ap.add_argument("--frames", type=int, default=None, help="viz frames SUB port (5602)")
    ap.add_argument("--gt", type=int, default=None, help="gt.pose SUB port (5601)")
    ap.add_argument("--gt-rep", type=int, default=None, help="P1 REP port for scene/top-down/occupancy (5600)")
    ap.add_argument("--body-ctl", type=int, default=None, help="body ROUTER port (5610; only recorded in summary)")
    ap.add_argument("--body-evt", type=int, default=None, help="body events SUB port (5611)")
    args = ap.parse_args(argv)
    port_map = ports(args.port_offset, head=args.head, frames=args.frames, gt_pub=args.gt, gt_rep=args.gt_rep,
                     body_ctl=args.body_ctl, body_evt=args.body_evt)
    if args.control:
        return serve(args, port_map)
    rec = Recorder(args.out, port_map, fps=args.fps, label=args.label, head_swap_rb=not args.no_head_swap,
                   duration=args.duration, until_event=args.until_event, post_roll=args.post_roll, crf=args.crf,
                   occupancy_npz=args.occupancy, top_long=args.top_long)

    def _sig(*_):
        rec.stop_reason = rec.stop_reason or "signal"
        rec._stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    rec.start()
    summary = rec.wait()
    print(json.dumps({k: summary[k] for k in ("dir", "duration_s", "composite_frames", "stop_reason",
                                              "contact_sheet")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
