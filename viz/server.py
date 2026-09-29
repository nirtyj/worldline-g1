#!/usr/bin/env python3
"""Live sim viewer + drive panel for the browser (runs on the box in viz/.venv, bound to 127.0.0.1).

    python viz/server.py [--port-offset 0] [--http 8765]
    # laptop:  00_infra/tunnel.sh 8765   then open http://localhost:8765

Sources (docs/contracts/m1.md): head camera SUB 5565 (gear_sonic msgpack), extra frames SUB 5602 (VizCams
frame.chase/top/overview, P1's own --tp-camera frame.tp shown as the chase pane, or any frame.<name>), gt.pose and
gt.objects SUB 5601, P1 REP 5600 (get_scene_info, get_occupancy,
render_topdown, get_stats, viz_level), body DEALER -> ROUTER 5610 and SUB 5611 (body.event, body.state).

Items (the sim's loose props, docs/contracts/p1_m2b.md §3): seeded from get_scene_info (its dynamic, non-articulated
objects at their load pose), then kept live from gt.objects (10 Hz, dynamic + held objects). The pages get ONE
compact full list {"type": "objects", "rev", "src", "t_sim", "objects": [{id, name, x, y, z, held_by}]} on connect
and then at most ITEMS_HZ (2 Hz), only when something moved by >= 1 cm or changed hands; /api/state carries it too.

Browser transport: ONE WebSocket (/ws) carries every pane as binary JPEG messages
    [uint32 BE header length][JSON header {"s": stream, "seq", "t_sim", "t_wall", "w", "h", "extent"?, "robot"?}][JPEG]
plus JSON text messages (hello, telemetry at 10 Hz, command replies, recorder status). Frames are passed through
as published (no re-encode) except the head camera and P1's frame.tp, whose R/B are swapped back (both are
cv2-encoded RGB). Flow control: the client acks every displayed frame and the server keeps at most --ws-window (4) unacked
frames per stream per client, always sending the newest one, so a slow SSH tunnel drops frames instead of building
latency (the window must cover fps x RTT: the Brev tunnel RTT is ~170 ms).
A single WebSocket avoids the browsers' 6-connections-per-host limit that several MJPEG <img> streams hit.

Also served (handy for curl, other UIs, and the future Worldline ui/server.py):
    GET /stream/<name>.mjpg   multipart/x-mixed-replace MJPEG (?fps=N caps the rate)
    GET /frame/<name>.jpg     latest frame of a stream
    GET /api/state            JSON: pose, body state, stream rates, scene, recorder, items (the full list)
    POST /api/cmd             {"op": "walk", "args": {...}} -> body reply
    POST /api/record          {"action": "start"|"stop"}
    GET /occupancy.png, /topdown.png, /recordings/<run>/<file>

Driving with held keys (page W/A/S/D/Q/E): the page sends {"type": "drive", "vx", "vy", "wz"} on every change
(chords debounced 40 ms) and as a 5 Hz heartbeat while keys are held, and {"type": "drive", "stop": true} on
release. The server turns that into ONE body op per key press, not one per change or per keepalive:
  velocity mode  the body's streaming op `velocity` (body/velocity.py): the first message starts it with a fresh
                 `stream` id, later ones (10 Hz from the server, box-local) update it without new ops or events,
                 release sends `end`; the body's own watchdog (watchdog_s 0.5) stops the robot if the server dies.
  walk mode      bodies without `velocity` (reply "unknown op"): one `walk` per distinct velocity, duration_s 10,
                 re-sent only after 8 s of the same keys; release sends `stop`.
--drive auto (default) tries velocity at the first key press and falls back to walk for the rest of the run.
Deadman: no heartbeat for 0.6 s (tab hidden, tunnel stalled, WebSocket closed) ends the drive like a release.
Any other command from the same page (stop, stand, go_to, turn_to) supersedes the drive without an extra message.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import struct
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio
from aiohttp import WSMsgType, web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.common import (  # noqa: E402
    decode_head, dumps, ep, frame_from_msg, is_dynamic_prop, item_summary, occupancy_rgba, pose_summary, ports,
    same_frame, split_msg, swap_rb_jpeg,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REC_ROOT = REPO / "outputs" / "recordings"
MAX_INFLIGHT = 4   # unacked frames per stream per client; >= fps x RTT (tunnel RTT ~170 ms -> 4 allows ~23 fps)
TEL_HZ = 10.0
ITEMS_HZ = 2.0            # items (gt.objects) -> pages: at most this often, and only on a change
UI_OPS = {"stand", "walk", "go_to", "turn_to", "stop", "status", "ping", "clear_fault", "velocity"}
DRIVE_TICK_S = 0.1        # velocity mode: server -> body stream rate (box-local, no tunnel jitter)
DRIVE_DEADMAN_S = 0.6     # no page heartbeat (5 Hz) for this long -> end the drive
DRIVE_WATCHDOG_S = 0.5    # body-side velocity watchdog we request (the server streams at 10 Hz)
WALK_HOLD_S = 10.0        # walk mode: duration_s of each walk ...
WALK_RESEND_S = 8.0       # ... re-sent only after this long with unchanged keys


class DriveSession:
    """Held-key drive of one page (see the module docstring)."""

    def __init__(self, cid: str):
        self.cid = cid
        self.n = 0
        self.v: tuple[float, float, float] | None = None   # wanted (vx, vy, wz); None = released
        self.t_hb = 0.0            # monotonic time of the last page message
        self.active = False        # our velocity/walk op is (believed) running on the body
        self.mode: str | None = None
        self.stream: str | None = None
        self.sent_v: tuple[float, float, float] | None = None
        self.t_sent = 0.0
        self.failed_v: tuple[float, float, float] | None = None   # its start failed: no retry until it changes
        self.epoch = 0             # bumped by any other command from this page (it supersedes the drive)
        self.busy = False
        self.ops = 0               # body ops started by this session (for tests / stats)
        self.msgs = 0              # body messages sent (starts + stream updates + ends)


class Stream:
    def __init__(self, name: str):
        self.name = name
        self.jpeg: bytes | None = None      # display-ready JPEG
        self.hdr: dict = {}
        self.seq = 0
        self.rx_t: deque = deque(maxlen=40)
        self.last_rx = 0.0
        self.cond = asyncio.Condition()

    def fps(self) -> float | None:
        if len(self.rx_t) < 3:
            return None
        dt = self.rx_t[-1] - self.rx_t[0]
        return round((len(self.rx_t) - 1) / dt, 2) if dt > 0 else None

    async def publish(self, jpeg: bytes, hdr: dict) -> None:
        now = time.time()
        async with self.cond:
            self.jpeg, self.hdr = jpeg, hdr
            self.seq += 1
            self.rx_t.append(now)
            self.last_rx = now
            self.cond.notify_all()


class Client:
    def __init__(self, ws: web.WebSocketResponse):
        self.ws = ws
        self.id = uuid.uuid4().hex[:6]
        self.sent: dict[str, int] = {}
        self.inflight: dict[str, int] = {}
        self.subs: set[str] | None = None   # None = all streams
        self.wake = asyncio.Event()
        self.outbox: deque = deque()
        self.frames_sent = 0
        self.drive = DriveSession(self.id)


class Hub:
    def __init__(self, args: argparse.Namespace, port_map: dict):
        self.args = args
        self.p = port_map
        self.ctx = zmq.asyncio.Context.instance()
        self.streams: dict[str, Stream] = {}
        self.clients: set[Client] = set()
        self.pose: dict | None = None
        self.pose_raw: dict | None = None
        self.pose_t = 0.0
        self.pose_rx: deque = deque(maxlen=60)
        self.traj: deque = deque(maxlen=4000)
        self.body_state: dict | None = None
        self.body_state_t = 0.0
        self.body_events: deque = deque(maxlen=30)
        self.velocity_ok: bool | None = None if args.drive == "auto" else (args.drive == "velocity")
        self.scene: dict | None = None
        self.occ_png: bytes | None = None
        self.occ_meta: dict | None = None
        self.occ_npz: str | None = None
        self.topdown_png: bytes | None = None
        self.topdown_meta: dict | None = None
        self.p1_stats: dict | None = None
        self.p1_ok = False
        self.items: dict[str, dict] = {}          # id -> item_summary (module docstring: Items)
        self.items_rev = 0                        # bumped on every change the pages have to see
        self.items_t = 0.0                        # wall time of the last gt.objects
        self.items_t_sim: float | None = None
        self.items_rx: deque = deque(maxlen=30)
        self._items_msg: tuple[int, dict] | None = None
        self.dealer = None
        self.pending: dict[str, tuple[asyncio.Future, Client | None]] = {}
        self.rec_proc: asyncio.subprocess.Process | None = None
        self.rec_ctl: str | None = None
        self.rec_status: dict = {"recording": False}
        self.rec_last: dict | None = None
        self.head_raw: tuple[bytes, dict] | None = None
        self.head_raw_seq = 0
        self.head_event = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.closed_drivers: list[Client] = []    # pages that closed while driving (the loop ends their drive)
        self.t_start = time.time()

    def stream(self, name: str) -> Stream:
        st = self.streams.get(name)
        if st is None:
            st = self.streams[name] = Stream(name)
        return st

    def wake_all(self) -> None:
        for c in self.clients:
            c.wake.set()

    # ------------------------------------------------------------------------------------------ ZMQ inputs
    def _sub(self, port: int, topics: list[bytes], conflate: bool = False):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 20)
        if conflate:
            s.setsockopt(zmq.CONFLATE, 1)
        for t in topics:
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(ep(port))
        return s

    async def head_loop(self) -> None:
        s = self._sub(self.p["head"], [b""], conflate=True)
        while True:
            raw = await s.recv()
            try:
                jpeg, meta = decode_head(raw)
            except Exception:  # noqa: BLE001
                continue
            if jpeg:
                self.head_raw = (jpeg, meta)
                self.head_raw_seq += 1
                self.head_event.set()

    async def head_convert_loop(self) -> None:
        """Swap R/B of the head JPEG (off the event loop), rate-capped; the latest frame wins."""
        loop = asyncio.get_running_loop()
        min_dt = 1.0 / max(1.0, self.args.head_fps)
        last = 0.0
        done_seq = 0
        while True:
            await self.head_event.wait()
            self.head_event.clear()
            wait = min_dt - (time.time() - last)
            if wait > 0:
                await asyncio.sleep(wait)
            if self.head_raw_seq == done_seq or self.head_raw is None:
                continue
            done_seq = self.head_raw_seq
            jpeg, meta = self.head_raw
            last = time.time()
            if self.args.head_swap_rb:
                try:
                    jpeg = await loop.run_in_executor(None, swap_rb_jpeg, jpeg, 80)
                except Exception:  # noqa: BLE001
                    continue
            hdr = {"s": "head", "t_sim": meta.get("t_sim"), "t_wall": meta.get("t_wall")}
            await self.stream("head").publish(jpeg, hdr)
            self.wake_all()

    async def frames_loop(self) -> None:
        s = self._sub(self.p["frames"], [b"frame."])
        loop = asyncio.get_running_loop()
        while True:
            frames = await s.recv_multipart()
            topic, msg = split_msg(frames)
            got = frame_from_msg(topic, msg)
            if got is None:
                continue
            name, jpeg, msg, swap = got
            held = self.streams.get(name)
            if held is not None and held.jpeg is not None and same_frame(held.hdr, msg):
                continue  # VizCams re-send of the snapshot we already have
            if swap:  # P1 frame.tp: cv2-encoded RGB (10 Hz, ~4 ms per swap, off the event loop)
                try:
                    jpeg = await loop.run_in_executor(None, swap_rb_jpeg, jpeg, 85)
                except Exception:  # noqa: BLE001
                    continue
            cp = msg.get("cam_pose") or {}
            hdr = {"s": name, "seq": msg.get("seq"), "t_sim": msg.get("t_sim"), "t_wall": msg.get("t_wall"),
                   "w": msg.get("w"), "h": msg.get("h"), "extent": cp.get("extent"), "robot": msg.get("robot"),
                   "snapshot": bool(msg.get("snapshot")), "src": msg.get("source"), "source": msg.get("source")}
            await self.stream(name).publish(jpeg, hdr)
            self.wake_all()

    async def gt_loop(self) -> None:
        s = self._sub(self.p["gt_pub"], [b"gt."])
        while True:
            frames = await s.recv_multipart()
            topic, msg = split_msg(frames)
            if topic == "gt.objects" and isinstance(msg, dict):
                self.on_objects(msg)
                continue
            if topic != "gt.pose" or not isinstance(msg, dict):
                continue
            p = pose_summary(msg)
            if not p:
                continue
            now = time.time()
            self.pose, self.pose_raw, self.pose_t = p, msg, now
            self.pose_rx.append(now)
            bp = p.get("base_pos")
            if bp:
                if not self.traj or (bp[0] - self.traj[-1][0]) ** 2 + (bp[1] - self.traj[-1][1]) ** 2 > 0.03 ** 2:
                    self.traj.append((round(bp[0], 3), round(bp[1], 3)))

    # ------------------------------------------------------------------------------------------ items
    def on_objects(self, msg: dict) -> bool:
        """One gt.objects message: update the items; True when one moved by >= 1 cm, appeared or changed hands."""
        now = time.time()
        changed = not self.items_t                # the first one: the pages' source changes to live poses
        self.items_t, self.items_t_sim = now, msg.get("t_sim")
        self.items_rx.append(now)
        fz = float((self.scene or {}).get("floor_z") or 0.0)
        for o in msg.get("objects") or []:
            it = item_summary(o, fz)
            if it is not None and self.items.get(it["id"]) != it:
                self.items[it["id"]] = it
                changed = True
        if changed:
            self.items_rev += 1
        return changed

    def seed_items(self, info: dict) -> None:
        """get_scene_info: the house's props at their load pose (live gt.objects values win). A new house drops the
        old one's items; a scene that lists no props (the viz test flat, an empty scene) keeps what gt.objects gave."""
        fz = float(info.get("floor_z") or 0.0)
        seeded = {}
        for o in info.get("objects") or []:
            if is_dynamic_prop(o):
                it = item_summary(o, fz)
                if it is not None:
                    seeded[it["id"]] = it
        if not seeded:
            return
        live = {k: v for k, v in self.items.items() if k in seeded} if self.items_t else {}
        new = {**seeded, **live}
        if new != self.items:
            self.items = new
            self.items_rev += 1

    def items_source(self) -> str | None:
        if self.items_t:
            return "gt.objects"
        return "get_scene_info" if self.items else None

    def items_msg(self) -> dict:
        """The compact full list the pages draw (cached per revision)."""
        if self._items_msg is None or self._items_msg[0] != self.items_rev:
            self._items_msg = (self.items_rev, {
                "type": "objects", "rev": self.items_rev, "src": self.items_source(), "t_sim": self.items_t_sim,
                "objects": [self.items[k] for k in sorted(self.items)]})
        return self._items_msg[1]

    def items_rate(self) -> float | None:
        rx = self.items_rx
        if len(rx) < 3 or rx[-1] <= rx[0]:
            return None
        return round((len(rx) - 1) / (rx[-1] - rx[0]), 1)

    async def items_loop(self) -> None:
        """Forward item changes to the pages: at most ITEMS_HZ, one compact full list, only after a change."""
        sent = self.items_rev
        while True:
            await asyncio.sleep(1.0 / ITEMS_HZ)
            if self.items_rev == sent:
                continue
            sent = self.items_rev
            msg = self.items_msg()
            for c in self.clients:
                c.outbox.append(msg)
                c.wake.set()

    async def body_evt_loop(self) -> None:
        s = self._sub(self.p["body_evt"], [b""])
        while True:
            frames = await s.recv_multipart()
            topic, msg = split_msg(frames)
            if not isinstance(msg, dict):
                continue
            if topic == "body.state":
                self.body_state, self.body_state_t = msg, time.time()
            else:
                ev = {"topic": topic, "t": time.time(), **{k: msg.get(k) for k in ("id", "op", "state", "data")}}
                self.body_events.append(ev)
                for c in self.clients:
                    c.outbox.append({"type": "body_event", **ev})
                    c.wake.set()

    async def dealer_loop(self) -> None:
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.connect(ep(self.p["body_ctl"]))
        while True:
            frames = await self.dealer.recv_multipart()
            try:
                rep = json.loads(frames[-1])
            except Exception:  # noqa: BLE001
                continue
            fut_client = self.pending.pop(rep.get("id"), None)
            if fut_client:
                fut, _ = fut_client
                if not fut.done():
                    fut.set_result(rep)

    async def body_cmd(self, op: str, args: dict | None, client: Client | None = None,
                       timeout: float = 5.0, note: bool = True) -> dict:
        if op not in UI_OPS:
            return {"ok": False, "error": f"op {op!r} not allowed from the UI"}
        if client is not None and op not in ("status", "ping", "velocity"):
            client.drive.epoch += 1          # this command supersedes the page's held-key drive
            client.drive.v, client.drive.active = None, False
        req = {"id": f"ui-{op}-{uuid.uuid4().hex[:8]}", "op": op, "args": args or {}}
        fut = asyncio.get_running_loop().create_future()
        self.pending[req["id"]] = (fut, client)
        await self.dealer.send(json.dumps(req).encode())
        if note and op not in ("status", "ping"):
            asyncio.create_task(self.rec_note(cmd={"op": op, "args": args or {}, "id": req["id"]}))
        try:
            rep = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self.pending.pop(req["id"], None)
            rep = {"id": req["id"], "ok": False, "error": f"no reply from body on {self.p['body_ctl']} in {timeout}s"}
        return {"op": op, "args": args or {}, **rep}

    # ------------------------------------------------------------------------------------------ held-key drive
    def drive_msg(self, c: Client, m: dict) -> None:
        ds = c.drive
        ds.t_hb = time.monotonic()
        if m.get("stop"):
            ds.v = None
        else:
            def num(k: str, lim: float) -> float:
                try:
                    return round(max(-lim, min(lim, float(m.get(k) or 0.0))), 3)
                except (TypeError, ValueError):
                    return 0.0
            ds.v = (num("vx", 1.0), num("vy", 1.0), num("wz", 1.5))
        if ds.v != ds.sent_v or ds.v is None:
            asyncio.create_task(self.drive_step(c))   # act on a change at once; the tick loop does the rest

    async def drive_loop(self) -> None:
        while True:
            await asyncio.sleep(DRIVE_TICK_S)
            for c in list(self.clients) + list(self.closed_drivers):
                if c.drive.v is not None or c.drive.active:
                    asyncio.create_task(self.drive_step(c))
            self.closed_drivers = [c for c in self.closed_drivers if c.drive.active or c.drive.busy]

    def _drive_tell(self, c: Client, **kw) -> None:
        c.outbox.append({"type": "drive", **kw})
        c.wake.set()

    async def drive_step(self, c: Client) -> None:
        ds = c.drive
        if ds.busy:
            return
        ds.busy = True
        try:
            await self._drive_step(c, ds)
        finally:
            ds.busy = False

    async def _drive_step(self, c: Client, ds: DriveSession) -> None:
        now = time.monotonic()
        want = ds.v
        if want is not None and now - ds.t_hb > DRIVE_DEADMAN_S:
            ds.v = want = None
            self._drive_tell(c, state="deadman", mode=ds.mode,
                             info=f"no key heartbeat for {DRIVE_DEADMAN_S} s: drive ended")
        epoch = ds.epoch
        if want is None:                                   # released
            if ds.active:
                ds.active = False
                if ds.mode == "velocity":
                    await self.body_cmd("velocity", {"stream": ds.stream, "end": True, "t_wall": time.time()},
                                        timeout=2.0, note=False)
                else:
                    await self.body_cmd("stop", {}, timeout=2.0, note=False)
                ds.msgs += 1
                await self.rec_note(cmd={"op": "drive", "args": {"released": True}})
                self._drive_tell(c, state="ended", mode=ds.mode)
            ds.sent_v = ds.failed_v = None
            return
        if want == ds.failed_v:
            return
        changed = want != ds.sent_v
        vx, vy, wz = want
        if self.velocity_ok is not False:
            if not changed and ds.active and now - ds.t_sent < DRIVE_TICK_S * 0.8:
                return
            if not ds.active:
                ds.n += 1
                ds.stream = f"ui-{c.id}-{ds.n}"
            args = {"vx": vx, "vy": vy, "wz": wz, "stream": ds.stream, "t_wall": time.time(),
                    "watchdog_s": DRIVE_WATCHDOG_S}
            rep = await self.body_cmd("velocity", args, timeout=2.0, note=False)
            if changed and rep.get("ok"):   # note the recorder only about what the body took, and only on change
                asyncio.create_task(self.rec_note(cmd={"op": "velocity", "args": args, "id": rep.get("id")}))
            ds.msgs += 1
            err = str(rep.get("error") or "")
            if not rep.get("ok") and "unknown op" in err and self.args.drive == "auto":
                self.velocity_ok = False            # this body has no streaming op: walk mode for the rest of the run
                self._drive_tell(c, state="mode", mode="walk",
                                 info="body has no 'velocity' op: using walk (one op per key change)")
            elif not rep.get("ok"):
                ds.failed_v, ds.active = want, False
                self._drive_tell(c, state="failed", mode="velocity", error=err or rep.get("state"))
                return
            else:
                self.velocity_ok = True
                if ds.epoch != epoch:                 # another command superseded us while this was in flight
                    ds.active = False
                    return
                if rep.get("state") == "accepted" or not ds.active:
                    ds.ops += int(rep.get("state") == "accepted")
                    self._drive_tell(c, state="started", mode="velocity", stream=ds.stream, v=list(want))
                ds.active, ds.mode, ds.sent_v, ds.t_sent = True, "velocity", want, now
                return
        # walk mode
        if not changed and ds.active and now - ds.t_sent < WALK_RESEND_S:
            return
        rep = await self.body_cmd("walk", {"vx": vx, "vy": vy, "yaw_rate": wz, "duration_s": WALK_HOLD_S},
                                  timeout=2.0)
        ds.msgs += 1
        if not rep.get("ok"):
            ds.failed_v, ds.active = want, False
            self._drive_tell(c, state="failed", mode="walk", error=str(rep.get("error") or rep.get("state")))
            return
        ds.ops += 1
        if ds.epoch != epoch:
            ds.active = False
            return
        ds.active, ds.mode, ds.sent_v, ds.t_sent = True, "walk", want, now
        self._drive_tell(c, state="started", mode="walk", v=list(want))

    # ------------------------------------------------------------------------------------------ P1 REP
    async def p1(self, op: str, timeout: float = 5.0, **args) -> dict | None:
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(ep(self.p["gt_rep"]))
        try:
            await s.send(json.dumps({"op": op, **args, "args": args}).encode())
            if await s.poll(int(timeout * 1000)) == 0:
                self.p1_ok = False
                return None
            rep = json.loads(await s.recv())
            self.p1_ok = True
            return rep
        except Exception:  # noqa: BLE001
            return None
        finally:
            s.close(0)

    async def refresh_scene(self) -> None:
        info = await self.p1("get_scene_info")
        if info and info.get("ok", True):
            self.scene = {k: info.get(k) for k in ("house_id", "floor_z", "bounds", "rooms", "spawn")}
            self.scene["n_objects"] = len(info.get("objects") or [])
            self.seed_items(info)
        occ = await self.p1("get_occupancy", timeout=30.0)
        if occ and occ.get("ok", True) and occ.get("path") and os.path.exists(occ["path"]):
            try:
                loop = asyncio.get_running_loop()
                im, ext = await loop.run_in_executor(None, occupancy_rgba, occ["path"])
                import io

                b = io.BytesIO()
                im.save(b, format="PNG")
                self.occ_png, self.occ_meta, self.occ_npz = b.getvalue(), {"extent": ext, "w": im.width,
                                                                            "h": im.height}, occ["path"]
            except Exception as e:  # noqa: BLE001
                print(f"[viz] occupancy load failed: {e}", flush=True)
        td_path = f"/tmp/viz_topdown_{self.p['gt_rep']}.png"
        td = await self.p1("render_topdown", timeout=30.0, path=td_path)
        if td and td.get("ok", True) and td.get("path") and os.path.exists(td["path"]):
            self.topdown_png = Path(td["path"]).read_bytes()
            self.topdown_meta = {"extent": td.get("extent"), "w": td.get("width"), "h": td.get("height")}
        for c in self.clients:
            c.outbox.append(self.hello())
            c.wake.set()

    async def p1_stats_loop(self) -> None:
        await asyncio.sleep(1.0)
        await self.refresh_scene()
        n = 0
        while True:
            st = await self.p1("get_stats", timeout=2.0)
            if st and st.get("ok", True):
                self.p1_stats = {k: st.get(k) for k in ("rtf_1s", "rtf_10s", "render_hz", "camera_pub_hz",
                                                        "physics_hz_1s", "overruns", "physx_device")}
                if self.scene is None and n % 5 == 0:
                    await self.refresh_scene()
            n += 1
            await asyncio.sleep(2.0)

    # ------------------------------------------------------------------------------------------ recorder
    async def rec_note(self, **kw) -> None:
        if not self.rec_ctl or not self.rec_status.get("recording"):
            return
        await self._rec_req({"op": "note", **kw}, timeout=2.0)

    async def _rec_req(self, req: dict, timeout: float = 5.0) -> dict | None:
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.rec_ctl)
        try:
            await s.send(dumps(req).encode())
            if await s.poll(int(timeout * 1000)) == 0:
                return None
            return json.loads(await s.recv())
        finally:
            s.close(0)

    async def rec_start(self, label: str = "ui") -> dict:
        if self.rec_proc and self.rec_proc.returncode is None:
            return {"ok": True, **self.rec_status}
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        self.rec_ctl = f"tcp://127.0.0.1:{port}"
        # every port explicitly: the recorder must use this server's sources (offset or per-port overrides), never
        # its own WL_PORT_OFFSET/default, or a test UI would query the production P1 REP on 5600
        cmd = [sys.executable, str(HERE / "recorder.py"), "--control", self.rec_ctl, "--label", label,
               "--head", str(self.p["head"]), "--frames", str(self.p["frames"]), "--gt", str(self.p["gt_pub"]),
               "--gt-rep", str(self.p["gt_rep"]), "--body-ctl", str(self.p["body_ctl"]),
               "--body-evt", str(self.p["body_evt"]), "--out", str(self.args.rec_out)]
        if not self.args.head_swap_rb:
            cmd.append("--no-head-swap")
        if self.occ_npz:
            cmd += ["--occupancy", self.occ_npz]
        self.rec_proc = await asyncio.create_subprocess_exec(*cmd)
        for _ in range(50):
            await asyncio.sleep(0.1)
            st = await self._rec_req({"op": "status"}, timeout=0.5)
            if st and st.get("ok"):
                self.rec_status = st
                return {"ok": True, **st}
        return {"ok": False, "error": "recorder did not start"}

    async def rec_stop(self) -> dict:
        if not self.rec_proc or self.rec_proc.returncode is not None:
            return {"ok": False, "error": "not recording", "last": self.rec_last}
        rep = await self._rec_req({"op": "stop"}, timeout=90.0)
        try:
            await asyncio.wait_for(self.rec_proc.wait(), 10.0)
        except asyncio.TimeoutError:
            self.rec_proc.terminate()
        self.rec_status = {"recording": False}
        self.rec_last = (rep or {}).get("summary")
        return {"ok": bool(rep), "summary": self.rec_last}

    async def rec_status_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.rec_proc and self.rec_proc.returncode is None and self.rec_ctl:
                st = await self._rec_req({"op": "status"}, timeout=1.0)
                if st:
                    self.rec_status = st
            elif self.rec_proc and self.rec_proc.returncode is not None and self.rec_status.get("recording"):
                self.rec_status = {"recording": False}

    # ------------------------------------------------------------------------------------------ state
    def hello(self) -> dict:
        return {"type": "hello", "ports": self.p, "scene": self.scene,
                "occupancy": ({"url": "/occupancy.png", **self.occ_meta} if self.occ_png else None),
                "topdown": ({"url": "/topdown.png", **self.topdown_meta} if self.topdown_png else None),
                "traj": list(self.traj)[-2000:], "streams": sorted(self.streams),
                "rec_root": str(self.args.rec_out)}

    def telemetry(self) -> dict:
        now = time.time()
        bs = self.body_state
        body = None
        if bs is not None:
            act = bs.get("active") or {}
            body = {"age_s": round(now - self.body_state_t, 2), "fault": bs.get("fault"),
                    "in_control": bs.get("in_control"), "control_started": bs.get("control_started"),
                    "op": act.get("op"), "phase": act.get("phase"), "id": act.get("id")}
        pose_rate = None
        if len(self.pose_rx) > 3 and self.pose_rx[-1] > self.pose_rx[0]:
            pose_rate = round((len(self.pose_rx) - 1) / (self.pose_rx[-1] - self.pose_rx[0]), 1)
        return {
            "type": "tel", "t": now, "pose": self.pose,
            "pose_age_s": round(now - self.pose_t, 2) if self.pose_t else None, "pose_hz": pose_rate,
            "body": body, "p1": self.p1_stats, "p1_ok": self.p1_ok,
            "streams": {n: {"fps": s.fps(), "age_s": round(now - s.last_rx, 2) if s.last_rx else None,
                            "seq": s.seq, "snapshot": bool(s.hdr.get("snapshot"))} for n, s in self.streams.items()},
            "rec": self.rec_status,
            "items": {"n": len(self.items), "rev": self.items_rev, "src": self.items_source(),
                      "hz": self.items_rate(), "age_s": round(now - self.items_t, 2) if self.items_t else None},
        }


# ---------------------------------------------------------------------------------------------- web handlers
def make_app(hub: Hub) -> web.Application:
    app = web.Application(client_max_size=2 ** 20)

    async def index(_req):
        return web.FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    async def ws_handler(req: web.Request):
        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=2 ** 20, compress=False)
        await ws.prepare(req)
        c = Client(ws)
        hub.clients.add(c)
        c.outbox.append(hub.hello())
        if hub.items:
            c.outbox.append(hub.items_msg())
        sender = asyncio.create_task(client_sender(hub, c))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except Exception:  # noqa: BLE001
                    continue
                t = m.get("type")
                if t == "ack":
                    s = m.get("s")
                    c.inflight[s] = max(0, c.inflight.get(s, 0) - 1)
                    c.wake.set()
                elif t == "sub":
                    c.subs = set(m.get("streams") or []) or None
                    c.wake.set()
                elif t == "drive":
                    hub.drive_msg(c, m)
                elif t == "cmd":
                    asyncio.create_task(_cmd_reply(hub, c, m))
                elif t == "record":
                    asyncio.create_task(_rec_reply(hub, c, m))
                elif t == "viz_level":
                    asyncio.create_task(_level_reply(hub, c, m))
                elif t == "refresh":
                    asyncio.create_task(hub.refresh_scene())
                elif t == "clear_traj":
                    hub.traj.clear()
        finally:
            hub.clients.discard(c)
            sender.cancel()
            if c.drive.v is not None or c.drive.active:   # page gone while driving: end it now
                c.drive.v = None
                hub.closed_drivers.append(c)
                asyncio.create_task(hub.drive_step(c))
        return ws

    async def mjpeg(req: web.Request):
        name = req.match_info["name"]
        st = hub.stream(name)
        fps_cap = float(req.query.get("fps", "0") or 0)
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=vizframe",
            "Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-Accel-Buffering": "no"})
        await resp.prepare(req)
        seen, last = 0, 0.0
        try:
            while True:
                async with st.cond:
                    await st.cond.wait_for(lambda: st.seq != seen and st.jpeg is not None)
                    jpeg, seen = st.jpeg, st.seq
                if fps_cap > 0:
                    wait = 1.0 / fps_cap - (time.time() - last)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    last = time.time()
                await resp.write(b"--vizframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def frame(req: web.Request):
        st = hub.streams.get(req.match_info["name"])
        if not st or st.jpeg is None:
            raise web.HTTPNotFound(text="no frame yet")
        return web.Response(body=st.jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    async def api_state(_req):
        tel = hub.telemetry()
        items = {**tel["items"], "objects": hub.items_msg()["objects"]}
        return web.json_response({**tel, "items": items, "hello": hub.hello(),
                                   "body_events": list(hub.body_events)[-10:], "rec_last": hub.rec_last,
                                   "drive": {"velocity_ok": hub.velocity_ok, "mode_arg": hub.args.drive,
                                             "sessions": [{"client": cl.id, "active": cl.drive.active,
                                                           "mode": cl.drive.mode, "ops": cl.drive.ops,
                                                           "msgs": cl.drive.msgs} for cl in hub.clients]}},
                                  dumps=dumps)

    async def api_cmd(req: web.Request):
        m = await req.json()
        return web.json_response(await hub.body_cmd(m.get("op", ""), m.get("args") or {}), dumps=dumps)

    async def api_record(req: web.Request):
        m = await req.json()
        rep = await (hub.rec_start(m.get("label") or "api") if m.get("action") == "start" else hub.rec_stop())
        return web.json_response(rep, dumps=dumps)

    async def occ(_req):
        if not hub.occ_png:
            raise web.HTTPNotFound()
        return web.Response(body=hub.occ_png, content_type="image/png")

    async def topdown(_req):
        if not hub.topdown_png:
            raise web.HTTPNotFound()
        return web.Response(body=hub.topdown_png, content_type="image/png")

    async def recordings(_req):
        root = Path(hub.args.rec_out)
        runs = []
        if root.exists():
            for d in sorted(root.iterdir(), reverse=True)[:30]:
                if d.is_dir():
                    runs.append({"name": d.name, "files": sorted(f.name for f in d.iterdir() if f.is_file())})
        return web.json_response({"root": str(root), "runs": runs})

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/stream/{name}.mjpg", mjpeg)
    app.router.add_get("/frame/{name}.jpg", frame)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/cmd", api_cmd)
    app.router.add_post("/api/record", api_record)
    app.router.add_get("/api/recordings", recordings)
    app.router.add_get("/occupancy.png", occ)
    app.router.add_get("/topdown.png", topdown)
    Path(hub.args.rec_out).mkdir(parents=True, exist_ok=True)
    app.router.add_static("/recordings/", str(hub.args.rec_out), show_index=True)
    return app


async def _cmd_reply(hub: Hub, c: Client, m: dict) -> None:
    rep = await hub.body_cmd(str(m.get("op", "")), m.get("args") or {}, client=c)
    c.outbox.append({"type": "reply", "req": m.get("req"), **rep})
    c.wake.set()


async def _rec_reply(hub: Hub, c: Client, m: dict) -> None:
    rep = await (hub.rec_start(m.get("label") or "ui") if m.get("action") == "start" else hub.rec_stop())
    for cl in hub.clients:
        cl.outbox.append({"type": "record", "action": m.get("action"), **rep})
        cl.wake.set()


async def _level_reply(hub: Hub, c: Client, m: dict) -> None:
    rep = await hub.p1("viz_level", level=m.get("level", "low"))
    c.outbox.append({"type": "viz_level", **(rep or {"ok": False, "error": "no reply from P1 REP"})})
    c.wake.set()


async def client_sender(hub: Hub, c: Client) -> None:
    """The only writer of this client's socket: queued JSON, telemetry at TEL_HZ, newest frames with flow control."""
    ws = c.ws
    next_tel = 0.0
    try:
        while not ws.closed:
            try:
                await asyncio.wait_for(c.wake.wait(), timeout=1.0 / TEL_HZ)
            except asyncio.TimeoutError:
                pass
            c.wake.clear()
            while c.outbox:
                await ws.send_str(dumps(c.outbox.popleft()))
            now = time.time()
            if now >= next_tel:
                next_tel = now + 1.0 / TEL_HZ
                await ws.send_str(dumps(hub.telemetry()))
            for name, st in list(hub.streams.items()):
                if c.subs is not None and name not in c.subs:
                    continue
                if st.jpeg is None or st.seq == c.sent.get(name) or c.inflight.get(name, 0) >= hub.args.ws_window:
                    continue
                hdr = json.dumps({**st.hdr, "s": name, "seq_srv": st.seq}).encode()
                await ws.send_bytes(struct.pack(">I", len(hdr)) + hdr + st.jpeg)
                c.sent[name] = st.seq
                c.inflight[name] = c.inflight.get(name, 0) + 1
                c.frames_sent += 1
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[viz] client {c.id} sender stopped: {e}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    port_map = ports(args.port_offset, head=args.head, frames=args.frames, gt_pub=args.gt, gt_rep=args.gt_rep,
                     body_ctl=args.body_ctl, body_evt=args.body_evt, http=args.http)
    hub = Hub(args, port_map)
    for fn in (hub.head_loop, hub.head_convert_loop, hub.frames_loop, hub.gt_loop, hub.body_evt_loop,
               hub.dealer_loop, hub.p1_stats_loop, hub.rec_status_loop, hub.drive_loop, hub.items_loop):
        hub.tasks.append(asyncio.create_task(fn(), name=fn.__name__))
    app = make_app(hub)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, port_map["http"])
    await site.start()
    print(f"[viz] http://{args.host}:{port_map['http']}  sources: head {port_map['head']}, frames "
          f"{port_map['frames']}, gt {port_map['gt_pub']}/{port_map['gt_rep']}, body {port_map['body_ctl']}/"
          f"{port_map['body_evt']}; recordings -> {args.rec_out}", flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    for c in list(hub.clients):          # end held-key drives, then close the pages (no 60 s graceful wait)
        if c.drive.active:
            c.drive.v = None
            await hub.drive_step(c)
        await c.ws.close()
    if hub.rec_proc and hub.rec_proc.returncode is None:
        print("[viz] stopping the recorder", flush=True)
        await hub.rec_stop()
    for t in hub.tasks:
        t.cancel()
    await runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port-offset", type=int, default=None, help="shift every port (build-phase tests: 100)")
    ap.add_argument("--http", type=int, default=None, help="HTTP port (default 8765 + offset)")
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--gt", type=int, default=None)
    ap.add_argument("--gt-rep", type=int, default=None)
    ap.add_argument("--body-ctl", type=int, default=None)
    ap.add_argument("--body-evt", type=int, default=None)
    ap.add_argument("--head-fps", type=float, default=15.0, help="cap for the (re-encoded) head stream")
    ap.add_argument("--no-head-swap", dest="head_swap_rb", action="store_false",
                    help="the head JPEG already has standard channel order (contract: it does not)")
    ap.add_argument("--rec-out", default=str(REC_ROOT))
    ap.add_argument("--drive", choices=["auto", "velocity", "walk"], default="auto",
                    help="held-key driving: body op 'velocity' (streaming), 'walk', or auto-detect (module docstring)")
    ap.add_argument("--ws-window", type=int, default=MAX_INFLIGHT,
                    help="unacked frames per stream per browser (flow control; >= fps x RTT)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
