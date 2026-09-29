"""viz tests without Isaac: frame formats (P1 frame.tp aliasing + R/B swap, VizCams pass-through, head decode, snapshot
re-send dedupe), the recorder (telemetry panel fits its tile, top.mp4 size, go_to target from body events), the P1
hook's argument guard, and viz/server.py end to end (HTTP frames, server-spawned recorder ports, held-key driving
against viz/tests/fake_body.py in both body modes, items from a fake P1's get_scene_info and gt.objects).

    cd /work/worldline-g1 && viz/.venv/bin/python viz/tests/test_frames.py      # plain runner, prints ok per test
    cd /work/worldline-g1 && viz/.venv/bin/python -m pytest -q viz/tests         # same tests (box.sh setup installs pytest)
Set VIZ_TEST_OUT=<dir> to keep the rendered composite of test_telemetry_panel_fits for a look.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import socket
import threading
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import msgpack
import numpy as np
import zmq
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from viz.common import decode_head, decode_jpeg, frame_from_msg, same_frame, split_msg  # noqa: E402
from viz.recorder import TILE, Recorder  # noqa: E402
from viz.tap import FrameTap  # noqa: E402
from viz.tests.fake_body import FakeBody  # noqa: E402

RED = (230, 20, 20)


def _rgb(w: int = 64, h: int = 48) -> np.ndarray:
    a = np.zeros((h, w, 3), np.uint8)
    a[:, :] = RED
    return a


def _jpeg_standard(rgb: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format="JPEG", quality=90)
    return b.getvalue()


def _jpeg_cv2_rgb(rgb: np.ndarray) -> bytes:
    """What cv2.imencode(".jpg", rgb) produces: cv2 reads channel 0 as B, so the stored colour is rgb[..., ::-1]."""
    return _jpeg_standard(np.ascontiguousarray(rgb[..., ::-1]))


def _center(im) -> tuple[int, int, int]:
    return im.getpixel((im.width // 2, im.height // 2))


def _is_red(px) -> bool:
    return px[0] > 180 and px[2] < 80


def _p1_tp_parts(seq: int = 1) -> list[bytes]:
    """sim_isaac/camera.py FramePublisher(mode="multipart", topic=b"frame.tp") message."""
    body = {"seq": seq, "t_sim": 1.25, "t_wall": time.time(), "jpeg": _jpeg_cv2_rgb(_rgb()),
            "base_pos": [1.0, 2.0, 0.75], "yaw": 0.5}
    return [b"frame.tp", msgpack.packb(body, use_bin_type=True)]


def _vizcams_parts(seq: int = 1) -> list[bytes]:
    body = {"topic": "frame.chase", "seq": seq, "t_sim": 1.0, "t_wall": time.time(), "w": 64, "h": 48,
            "jpeg": _jpeg_standard(_rgb()), "cam_pose": {"extent": None}, "robot": {"pos": [0, 0, 0.75], "yaw": 0.0},
            "v": 1}
    return [b"frame.chase", msgpack.packb(body, use_bin_type=True)]


def test_p1_tp_is_chase_and_swapped():
    name, jpeg, meta, swap = frame_from_msg(*split_msg(_p1_tp_parts()))
    assert name == "chase" and swap and meta["source"] == "tp"
    assert meta["robot"] == {"pos": [1.0, 2.0, 0.75], "yaw": 0.5}
    assert not _is_red(_center(decode_jpeg(jpeg)))          # as published: R/B swapped
    assert _is_red(_center(decode_jpeg(jpeg, swap_rb=True)))


def test_vizcams_passthrough():
    name, jpeg, meta, swap = frame_from_msg(*split_msg(_vizcams_parts()))
    assert name == "chase" and not swap and meta["source"] == "chase"
    assert _is_red(_center(decode_jpeg(jpeg)))


def test_head_gear_sonic():
    b64 = base64.b64encode(_jpeg_cv2_rgb(_rgb())).decode()
    raw = msgpack.packb({"timestamps": {"ego_view": 1.0}, "images": {"ego_view": b64}, "ego_view": b64,
                         "t_sim": 2.0, "seq": 7}, use_bin_type=True)
    jpeg, meta = decode_head(raw)
    assert meta == {"t_wall": 1.0, "t_sim": 2.0, "seq": 7}
    assert _is_red(_center(decode_jpeg(jpeg, swap_rb=True)))


def test_recorder_colours_tp():
    rec = Recorder(out_root="/tmp", verbose=False)
    rec._on_msg("frames", _p1_tp_parts())
    assert "tp" not in rec.streams
    im, meta = rec._stream_image(rec.streams["chase"])
    assert _is_red(_center(im)) and meta["source"] == "tp"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pub_until(pub, parts_fn, cond, timeout=5.0) -> bool:
    t_end = time.time() + timeout
    seq = 0
    while time.time() < t_end:
        seq += 1
        pub.send_multipart(parts_fn(seq))
        time.sleep(0.05)
        if cond():
            return True
    return False


def test_tap_fixes_tp():
    port = _free_port()
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    tap = FrameTap(port_offset=0, frames=port, head=_free_port(), gt_pub=_free_port())
    try:
        assert _pub_until(pub, _p1_tp_parts, lambda: tap.rev("chase") > 0)
        assert _is_red(_center(decode_jpeg(tap.jpeg("chase"))))
        assert tap.meta("chase")["source"] == "tp"
    finally:
        tap.close()
        pub.close(0)


def test_server_http_tp():
    """viz/server.py end to end: a P1-style frame.tp publisher -> GET /frame/chase.jpg is red, /api/state lists it."""
    frames, http = _free_port(), _free_port()
    others = {k: _free_port() for k in ("head", "gt", "gt-rep", "body-ctl", "body-evt")}
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{frames}")
    cmd = [sys.executable, str(REPO / "viz" / "server.py"), "--frames", str(frames), "--http", str(http),
           "--rec-out", "/tmp/viz_test_rec"] + sum(([f"--{k}", str(v)] for k, v in others.items()), [])
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def got() -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{http}/frame/chase.jpg", timeout=0.5) as r:
                got.jpeg = r.read()
            return True
        except Exception:  # noqa: BLE001
            return False

    try:
        assert _pub_until(pub, _p1_tp_parts, got, timeout=15.0), "server never served /frame/chase.jpg"
        assert _is_red(_center(decode_jpeg(got.jpeg)))
        with urllib.request.urlopen(f"http://127.0.0.1:{http}/api/state", timeout=2) as r:
            st = json.loads(r.read())
        assert "chase" in st["streams"] and "tp" not in st["streams"]
    finally:
        srv.terminate()
        srv.wait(10)
        pub.close(0)


# ------------------------------------------------------------------------------------------------ recorder units
def _body_ev(op: str, state: str, oid: str, data: dict | None = None) -> list[bytes]:
    return [b"body.event", json.dumps({"id": oid, "op": op, "state": state, "data": data or {}}).encode()]


def test_recorder_target_and_cmd_from_accepted_event():
    """A go_to from any client (CLI, /api/cmd, BodyClient) sets the target: the body's 'accepted' carries data.args."""
    rec = Recorder(out_root="/tmp", verbose=False)
    rec._on_msg("body", _body_ev("go_to", "accepted", "bc-1", {"args": {"x": 1.7, "y": 4.0}, "backend": "astar"}))
    assert rec.target == [1.7, 4.0]
    assert rec.last_cmd["op"] == "go_to" and rec.last_cmd["args"] == {"x": 1.7, "y": 4.0}
    rec._on_msg("body", _body_ev("go_to", "succeeded", "bc-1"))
    assert rec.target == [1.7, 4.0]                                     # kept after the go_to: final pose vs target
    rec._on_msg("body", _body_ev("walk", "accepted", "bc-2", {"args": {"vx": 0.4}}))
    assert rec.last_cmd["op"] == "walk" and rec.target is None          # another motion op clears it


def test_recorder_progress_collapses():
    rec = Recorder(out_root="/tmp", verbose=False)
    rec._on_msg("body", _body_ev("go_to", "accepted", "g1", {"args": {"x": 1, "y": 2}}))
    for _ in range(9):
        rec._on_msg("body", _body_ev("go_to", "progress", "g1"))
    rec._on_msg("body", _body_ev("go_to", "succeeded", "g1"))
    states = [(e["state"], e["n"]) for e in rec.body_log]
    assert states == [("accepted", 1), ("progress", 9), ("succeeded", 1)], states


def test_recorder_ports_flags():
    """--gt-rep / --body-ctl exist and win over WL_PORT_OFFSET (the server passes them to spawned recorders)."""
    from viz.common import ports
    old = os.environ.get("WL_PORT_OFFSET")
    os.environ["WL_PORT_OFFSET"] = "0"
    try:
        p = ports(None, head=5865, frames=5902, gt_pub=5901, gt_rep=5900, body_ctl=5910, body_evt=5911)
        assert p["gt_rep"] == 5900 and p["body_ctl"] == 5910
    finally:
        if old is None:
            os.environ.pop("WL_PORT_OFFSET", None)
        else:
            os.environ["WL_PORT_OFFSET"] = old
    import viz.recorder as R
    src = Path(R.__file__).read_text()
    assert '"--gt-rep"' in src and '"--body-ctl"' in src


def test_top_video_size_fixed():
    """top.mp4 has the same size whether the first image is P1's 346 px render_topdown or a 512 px VizCams snapshot."""
    rec = Recorder(out_root="/tmp", verbose=False)
    assert rec._video_size("top", (346, 346)) == (768, 768)
    assert rec._video_size("top", (512, 512)) == (768, 768)
    assert rec._video_size("top", (1024, 640)) == (768, 480)
    assert rec._video_size("chase", (480, 270)) == (640, 360)     # level min first, then low: native 640x360 fits
    assert rec._video_size("chase", (960, 540)) == (960, 540)
    assert rec._video_size("head", (640, 480)) == (640, 480)


def test_telemetry_panel_fits():
    """Full panel (pose, last cmd, target, many events incl. long progress runs) + a VIZ TEST note: every line inside
    the 360 px tile, the newest events (accepted/succeeded) shown, the note in the header band (not on a tile)."""
    from PIL import Image, ImageDraw

    from viz.common import font

    rec = Recorder(out_root="/tmp", verbose=False)
    rec._t0 = time.time() - 30
    rec.pose = {"t_sim": 123.45, "rtf": 0.97, "base_pos": [1.73, 3.97, 0.78], "yaw": 1.2, "pelvis_z": 0.781,
                "fallen": False, "foot_contact": {"left": True, "right": False}, "band": False,
                "note": "VIZ TEST: kinematic, not locomotion"}
    for k in range(3):
        rec._on_msg("body", _body_ev("walk", "accepted", f"w{k}", {"args": {"vx": 0.4}}))
        rec._on_msg("body", _body_ev("walk", "canceled", f"w{k}"))
    rec.annotate(cmd={"op": "go_to", "args": {"x": 1.7, "y": 4.0}, "id": "ui-go_to-1a2b3c4d"})
    rec._on_msg("body", _body_ev("go_to", "accepted", "ui-go_to-1a2b3c4d", {"args": {"x": 1.7, "y": 4.0}}))
    for _ in range(25):
        rec._on_msg("body", _body_ev("go_to", "progress", "ui-go_to-1a2b3c4d"))
    rec._on_msg("body", _body_ev("go_to", "succeeded", "ui-go_to-1a2b3c4d"))
    W, H = TILE
    comp = rec._compose({})
    probe = Image.new("RGB", comp.size)
    lay = rec._telemetry_panel(ImageDraw.Draw(probe), W, H + 58, W, H, font(15), font(19))
    assert lay["last_line_bottom"] <= lay["tile_bottom"], lay
    assert "succeeded" in lay["lines"][-1] and any("progress x25" in ln for ln in lay["lines"]), lay["lines"]
    assert any("accepted" in ln and "go_to" in ln for ln in lay["lines"]), lay["lines"]
    assert any(ln.startswith("target") for ln in lay["lines"]) and any(ln.startswith("last cmd") for ln in lay["lines"])
    # the note box (120, 60, 0) is in the header band and nowhere in the telemetry tile
    px = np.asarray(comp)
    note = np.all(px == (120, 60, 0), axis=-1)
    assert note[:58].any() and not note[58:].any()
    out = os.environ.get("VIZ_TEST_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        comp.save(Path(out) / "telemetry_panel.png")


def test_snapshot_resend_dedupe():
    parts = _vizcams_parts(3)
    body = msgpack.unpackb(parts[1], raw=False)
    body["topic"], body["snapshot"] = "frame.top", True
    msg = [b"frame.top", msgpack.packb(body, use_bin_type=True)]
    rec = Recorder(out_root="/tmp", verbose=False)
    rec._on_msg("frames", list(msg))
    rec._on_msg("frames", list(msg))       # VizCams re-send (same seq, t_wall)
    assert rec.streams["top"].rx == 1
    body2 = dict(body, t_wall=body["t_wall"] + 10.0)   # a new snapshot after set_level (seq restarts at the same value)
    rec._on_msg("frames", [b"frame.top", msgpack.packb(body2, use_bin_type=True)])
    assert rec.streams["top"].rx == 2
    assert same_frame({"seq": 1, "t_wall": 5.0, "source": "top"}, {"seq": 1, "t_wall": 5.0, "source": "top"})
    assert not same_frame({"seq": 1, "t_wall": 5.0, "source": "top"}, {"seq": 1, "t_wall": 6.0, "source": "top"})


def test_p1_hook_args_guard():
    """add_p1_args / viz_enabled: --viz with --tp-camera exits before Kit starts (both bind 5602)."""
    import argparse

    from viz.isaac_cams import add_p1_args, viz_enabled

    ap = argparse.ArgumentParser()
    ap.add_argument("--tp-camera", action="store_true")
    add_p1_args(ap)
    assert viz_enabled(ap.parse_args([])) is False
    assert viz_enabled(ap.parse_args(["--viz", "min"])) is True
    assert viz_enabled(ap.parse_args(["--tp-camera"])) is False
    try:
        viz_enabled(ap.parse_args(["--viz", "low", "--tp-camera"]))
    except SystemExit as e:
        assert "5602" in str(e)
    else:
        raise AssertionError("--viz with --tp-camera must exit")


# ------------------------------------------------------------------------------------------------ server end to end
def _free_offset() -> int:
    """A port offset whose viz ports (head..http) are all free."""
    for _ in range(50):
        off = random.randint(20000, 50000)
        ok = True
        for base in (5565, 5600, 5601, 5602, 5610, 5611, 8765):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", base + off))
                except OSError:
                    ok = False
                    break
        if ok:
            return off
    raise RuntimeError("no free offset")


def _start_server(args: list[str], env: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(REPO / "viz" / "server.py"), *args], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _http_json(url: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_http(url: str, timeout: float = 15.0) -> None:
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            _http_json(url, timeout=0.5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise AssertionError(f"server never answered {url}")


def test_server_recorder_uses_server_ports():
    """POST /api/record on a server at a port offset: the spawned recorder asks THIS offset's P1 REP (never 5600)."""
    tmp_dir = "/tmp/viz_test_rec_ports"
    off = _free_offset()
    seen: list[str] = []
    rep = zmq.Context.instance().socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(f"tcp://127.0.0.1:{5600 + off}")
    stop = threading.Event()

    def fake_p1():
        while not stop.is_set():
            if rep.poll(50):
                req = json.loads(rep.recv())
                seen.append(req.get("op"))
                ok = req.get("op") == "get_scene_info"
                rep.send_string(json.dumps({"ok": ok, "bounds": [0, 0, 8, 6], "house_id": "fake"} if ok
                                           else {"ok": False, "error": "fake"}))

    th = threading.Thread(target=fake_p1, daemon=True)
    th.start()
    env = dict(os.environ, WL_PORT_OFFSET="0")     # the recorder must not fall back to this
    srv = _start_server(["--port-offset", str(off), "--rec-out", tmp_dir], env=env)
    base = f"http://127.0.0.1:{8765 + off}"
    try:
        _wait_http(base + "/api/state")
        r = _http_json(base + "/api/record", {"action": "start", "label": "ports"}, timeout=10)
        assert r.get("ok"), r
        t_end = time.time() + 10
        while time.time() < t_end and seen.count("get_scene_info") < 2:   # server + recorder
            time.sleep(0.1)
        r = _http_json(base + "/api/record", {"action": "stop"}, timeout=60)
        ports_used = r["summary"]["ports"]
        assert ports_used["gt_rep"] == 5600 + off and ports_used["body_ctl"] == 5610 + off, ports_used
        assert ports_used["head"] == 5565 + off and ports_used["frames"] == 5602 + off, ports_used
        assert seen.count("get_scene_info") >= 2, seen
    finally:
        srv.terminate()
        srv.wait(10)
        stop.set()
        th.join(2)
        rep.close(0)


async def _drive_script(url: str, steps: list[tuple[float, dict]]) -> None:
    """Send WS messages at the given times (s from start); drive messages are repeated at 5 Hz like the page."""
    import aiohttp

    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url) as ws:
            async def drain():
                async for _ in ws:
                    pass
            dr = asyncio.create_task(drain())
            t0 = time.monotonic()
            held: dict | None = None
            next_hb = 0.0
            i = 0
            while i < len(steps) or held is not None:
                now = time.monotonic() - t0
                if i < len(steps) and now >= steps[i][0]:
                    m = steps[i][1]
                    i += 1
                    if m.get("type") == "hold":
                        held = m["v"]
                        await ws.send_str(json.dumps({"type": "drive", **held}))
                        next_hb = now + 0.2
                    elif m.get("type") == "release":
                        held = None
                        await ws.send_str(json.dumps({"type": "drive", "stop": True}))
                    elif m.get("type") == "silence":   # keys held but heartbeats stop (tab frozen / tunnel stall)
                        held = None
                    else:
                        await ws.send_str(json.dumps(m))
                    continue
                if held is not None and now >= next_hb:
                    await ws.send_str(json.dumps({"type": "drive", **held}))
                    next_hb += 0.2
                if i >= len(steps) and held is None:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(1.2)
            dr.cancel()


def _drive_run(velocity: bool, steps: list[tuple[float, dict]]) -> FakeBody:
    fb_port, http = _free_port(), _free_port()
    fb = FakeBody(fb_port, velocity=velocity)
    others = {k: _free_port() for k in ("head", "frames", "gt", "gt-rep", "body-evt")}
    srv = _start_server(["--http", str(http), "--body-ctl", str(fb_port), "--rec-out", "/tmp/viz_test_rec"]
                        + sum(([f"--{k}", str(v)] for k, v in others.items()), []))
    try:
        _wait_http(f"http://127.0.0.1:{http}/api/state")
        asyncio.run(_drive_script(f"ws://127.0.0.1:{http}/ws", steps))
    finally:
        srv.terminate()
        srv.wait(10)
        fb.close()
    return fb


W_ = {"vx": 0.4, "vy": 0.0, "wz": 0.0}
WA = {"vx": 0.4, "vy": 0.25, "wz": 0.0}


def test_server_drive_velocity_one_op_per_press():
    """Streaming body: hold W 1.5 s, chord to W+A 1 s, release -> ONE body op, stream updates at ~10 Hz, one end."""
    fb = _drive_run(True, [(0.0, {"type": "hold", "v": W_}), (1.5, {"type": "hold", "v": WA}),
                           (2.5, {"type": "release"})])
    vel = [r for r in fb.reqs if r["op"] == "velocity"]
    assert [o["op"] for o in fb.ops_started] == ["velocity"], fb.ops_started
    assert len({r["args"]["stream"] for r in vel}) == 1
    assert vel[-1]["args"].get("end") is True and not any(r["args"].get("end") for r in vel[:-1])
    assert not [r for r in fb.reqs if r["op"] in ("walk", "stop")]
    upd = [r for r in vel if not r["args"].get("end")]
    rate = (len(upd) - 1) / (upd[-1]["t"] - upd[0]["t"])
    assert 7.0 <= rate <= 12.0, rate
    assert {(r["args"]["vx"], r["args"]["vy"]) for r in upd} == {(0.4, 0.0), (0.4, 0.25)}
    assert all(r["args"]["watchdog_s"] == 0.5 and "t_wall" in r["args"] for r in upd)


def test_server_drive_deadman_and_supersede():
    """Heartbeats stop -> the server ends the stream within ~0.6 s; a page 'stop' command supersedes the drive
    without an extra end message."""
    fb = _drive_run(True, [(0.0, {"type": "hold", "v": W_}), (0.5, {"type": "silence"})])
    vel = [r for r in fb.reqs if r["op"] == "velocity"]
    assert vel[-1]["args"].get("end") is True
    t_last_upd = max(r["t"] for r in vel if not r["args"].get("end"))
    assert vel[-1]["t"] - vel[0]["t"] < 0.5 + 0.6 + 0.35, vel[-1]["t"] - vel[0]["t"]
    assert vel[-1]["t"] - t_last_upd < 0.3
    fb = _drive_run(True, [(0.0, {"type": "hold", "v": W_}), (0.8, {"type": "silence"}),
                           (0.8, {"type": "cmd", "op": "stop", "args": {}})])
    ops = [r["op"] for r in fb.reqs]
    assert "stop" in ops
    after = ops[ops.index("stop") + 1:]
    assert "velocity" not in after, after


def test_server_drive_walk_fallback():
    """Body without 'velocity': one walk per distinct key set (no 2 s keepalive walks), stop on release."""
    fb = _drive_run(False, [(0.0, {"type": "hold", "v": W_}), (2.5, {"type": "hold", "v": WA}),
                            (4.5, {"type": "release"})])
    ops = [r["op"] for r in fb.reqs]
    assert ops[0] == "velocity" and fb.reqs[0]["reply"] == "rejected"
    walks = [r for r in fb.reqs if r["op"] == "walk"]
    assert len(walks) == 2, [(r["t"], r["args"]) for r in walks]
    assert walks[0]["args"] == {"vx": 0.4, "vy": 0.0, "yaw_rate": 0.0, "duration_s": 10.0}
    assert walks[1]["args"]["vy"] == 0.25
    assert ops[-1] == "stop" and ops.count("velocity") == 1


# ------------------------------------------------------------------------------------------------ items (gt.objects)
def test_item_summary():
    """P1 object records -> the viewer's compact items: AABB centre, height above the floor, 1 cm rounding."""
    from viz.common import is_dynamic_prop, item_summary

    rec = {"id": "Vase|3", "name": "vase_3", "pos": [9, 9, 9], "aabb": [[1.0, 2.0, 0.85], [1.2, 2.3, 1.1]],
           "held_by": None, "dynamic": True, "quat_wxyz": [1, 0, 0, 0]}
    assert item_summary(rec, floor_z=0.1) == {"id": "Vase|3", "name": "vase_3", "x": 1.1, "y": 2.15, "z": 0.75,
                                              "held_by": None}
    assert item_summary({"id": "m", "pos": [0.5, -1.0, 0.2], "held_by": "left"}) == {
        "id": "m", "name": "m", "x": 0.5, "y": -1.0, "z": 0.2, "held_by": "left"}
    assert item_summary({"id": "x", "held_by": "left"}) is None and item_summary({"name": "no id"}) is None
    assert item_summary({"id": "bad", "pos": [float("nan"), 0, 0]}) is None
    assert item_summary({"id": "odd", "pos": [0, 0], "held_by": "tail"})["held_by"] is None
    assert is_dynamic_prop({"body_path": "/W/v", "is_static": False, "articulated": False})
    assert not is_dynamic_prop({"body_path": "/W/f", "is_static": True})
    assert not is_dynamic_prop({"body_path": "/W/d", "is_static": False, "articulated": True})
    assert not is_dynamic_prop({"id": "flat_table"})                      # the viz test flat: furniture only


def _fake_p1_scene(port: int, stop: threading.Event, floor_z: float = 0.1) -> threading.Thread:
    """P1 REP stand-in: get_scene_info with two loose props and one piece of furniture; every other op fails."""
    rep = zmq.Context.instance().socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(f"tcp://127.0.0.1:{port}")
    objects = [
        {"id": "Vase|3", "name": "vase_3", "category": "Vase", "pos": [1.1, 2.15, floor_z + 0.75],
         "aabb": [[1.0, 2.0, floor_z + 0.75], [1.2, 2.3, floor_z + 1.0]], "is_static": False, "articulated": False,
         "body_path": "/World/House/Vase_3"},
        {"id": "Mug|1", "name": "mug_1", "category": "Mug", "pos": [3.0, 1.0, floor_z + 0.8],
         "aabb": [[2.95, 0.95, floor_z + 0.8], [3.05, 1.05, floor_z + 0.9]], "is_static": False,
         "articulated": False, "body_path": "/World/House/Mug_1"},
        {"id": "Table|1", "name": "dining_table_1", "category": "DiningTable", "pos": [1.0, 2.0, floor_z],
         "aabb": [[0.5, 1.5, floor_z], [1.5, 2.5, floor_z + 0.75]], "is_static": True, "articulated": False,
         "body_path": None},
    ]

    def run():
        try:
            while not stop.is_set():
                if rep.poll(50):
                    req = json.loads(rep.recv())
                    if req.get("op") == "get_scene_info":
                        rep.send_string(json.dumps({"ok": True, "house_id": "fake", "floor_z": floor_z,
                                                    "bounds": [0, 0, 6, 4], "rooms": [], "objects": objects}))
                    else:
                        rep.send_string(json.dumps({"ok": False, "error": "fake"}))
        finally:
            rep.close(0)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _gt_objects(seq: int, vase_xy: tuple[float, float], floor_z: float = 0.1, mug_held: bool = True) -> list[bytes]:
    """P1 gt.objects (p1_m2b.md §3.2): the dynamic and held objects, full records."""
    vx, vy = vase_xy
    objs = [{"id": "Vase|3", "name": "vase_3", "pos": [vx, vy, floor_z + 0.75], "quat_wxyz": [1, 0, 0, 0],
             "aabb": [[vx - 0.1, vy - 0.15, floor_z + 0.75], [vx + 0.1, vy + 0.15, floor_z + 1.0]],
             "held_by": None, "dynamic": True, "source": "sim", "lin_vel": [0, 0, 0], "moving": False},
            {"id": "Mug|1", "name": "mug_1", "pos": [2.0, 1.5, floor_z + 1.0], "quat_wxyz": [1, 0, 0, 0],
             "aabb": [[1.95, 1.45, floor_z + 0.95], [2.05, 1.55, floor_z + 1.05]],
             "held_by": "left" if mug_held else None, "dynamic": True, "source": "sim", "lin_vel": [0, 0, 0],
             "moving": False}]
    body = {"seq": seq, "t_sim": round(seq * 0.1, 2), "t_wall": time.time(), "objects": objs}
    return [b"gt.objects", msgpack.packb(body, use_bin_type=True)]


def _gt_pose(seq: int) -> list[bytes]:
    body = {"seq": seq, "t_sim": round(seq * 0.02, 3), "t_wall": time.time(), "base_pos": [2.0, 1.6, 0.78],
            "base_quat_wxyz": [1, 0, 0, 0], "yaw": 0.0, "pelvis_z": 0.78, "fallen": False}
    return [b"gt.pose", msgpack.packb(body, use_bin_type=True)]


async def _collect_objects(url: str, script) -> list[tuple[float, dict]]:
    """Connect a page, run `script(t0)` (a coroutine publishing on the fake P1), return every 'objects' message."""
    import aiohttp

    got: list[tuple[float, dict]] = []
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url) as ws:
            async def drain():
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d.get("type") == "objects":
                            got.append((time.monotonic(), d))
            dr = asyncio.create_task(drain())
            await script()
            await asyncio.sleep(1.0)
            dr.cancel()
    return got


def test_server_forwards_objects():
    """viz/server.py items: seeded from get_scene_info (loose props only, load poses), sent on connect; gt.objects at
    20 Hz with a prop moving every tick reaches the page as compact full lists at <= 2 Hz; sub-cm jitter sends
    nothing; held_by and heights above the floor survive; /api/state and the telemetry carry the items."""
    gt, gt_rep, http = _free_port(), _free_port(), _free_port()
    others = {k: _free_port() for k in ("head", "frames", "body-ctl", "body-evt")}
    stop = threading.Event()
    th = _fake_p1_scene(gt_rep, stop)
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f"tcp://127.0.0.1:{gt}")
    srv = _start_server(["--http", str(http), "--gt", str(gt), "--gt-rep", str(gt_rep), "--rec-out",
                         "/tmp/viz_test_rec"] + sum(([f"--{k}", str(v)] for k, v in others.items()), []))
    base = f"http://127.0.0.1:{http}"
    try:
        _wait_http(base + "/api/state")
        t_end = time.time() + 15
        while time.time() < t_end and _http_json(base + "/api/state")["items"]["n"] < 2:
            time.sleep(0.1)
        st = _http_json(base + "/api/state")
        assert st["items"]["src"] == "get_scene_info" and st["items"]["n"] == 2, st["items"]
        seeded = {o["name"]: o for o in st["items"]["objects"]}
        assert set(seeded) == {"vase_3", "mug_1"}                        # the table is furniture, not an item
        assert seeded["vase_3"] == {"id": "Vase|3", "name": "vase_3", "x": 1.1, "y": 2.15, "z": 0.75,
                                    "held_by": None}

        phases: dict[str, float] = {}

        async def script():
            await asyncio.sleep(0.5)
            phases["move"] = time.monotonic()
            for k in range(40):                                    # 2 s at 20 Hz, +2 cm per tick
                pub.send_multipart(_gt_objects(k, (1.1 + 0.02 * k, 2.15)))
                pub.send_multipart(_gt_pose(k))
                await asyncio.sleep(0.05)
            phases["jitter"] = time.monotonic()
            for k in range(40, 70):                                # 1.5 s of sub-cm jitter at the final pose
                pub.send_multipart(_gt_objects(k, (1.1 + 0.02 * 39 + (0.002 if k % 2 else -0.002), 2.15)))
                await asyncio.sleep(0.05)
            phases["end"] = time.monotonic()

        got = asyncio.run(_collect_objects(f"ws://127.0.0.1:{http}/ws", script))
        assert got, "no objects message"
        first = got[0][1]
        assert first["src"] == "get_scene_info" and [o["name"] for o in first["objects"]] == ["mug_1", "vase_3"], \
            first                                                    # on connect, sorted by id ("Mug|1" < "Vase|3")
        live = [(t, m) for t, m in got if m["src"] == "gt.objects"]
        assert live, [m["src"] for _, m in got]
        # throttle: <= 2 Hz while the prop moved on every one of 40 ticks
        in_move = [t for t, _ in live if t <= phases["jitter"] + 0.6]
        assert len(in_move) <= 2.0 * (phases["jitter"] + 0.6 - phases["move"]) + 1, len(in_move)
        gaps = [b - a for a, b in zip(in_move, in_move[1:])]
        assert all(g >= 0.35 for g in gaps), gaps
        # jitter under 1 cm is no change: nothing sent once the final pose is out
        assert not [t for t, _ in live if t > phases["jitter"] + 0.6], [round(t - phases["jitter"], 2) for t, _ in live]
        last = {o["name"]: o for o in live[-1][1]["objects"]}
        assert last["vase_3"]["x"] == round(1.1 + 0.02 * 39, 2) and last["vase_3"]["z"] == 0.75, last["vase_3"]
        assert last["mug_1"]["held_by"] == "left" and last["mug_1"]["z"] == 0.95, last["mug_1"]
        assert set(last["vase_3"]) == {"id", "name", "x", "y", "z", "held_by"}
        st = _http_json(base + "/api/state")
        assert st["items"]["src"] == "gt.objects" and st["items"]["n"] == 2 and st["items"]["age_s"] is not None
        assert {o["name"]: o for o in st["items"]["objects"]} == last
        assert st["pose"]["base_pos"][:2] == [2.0, 1.6]
    finally:
        srv.terminate()
        srv.wait(10)
        stop.set()
        th.join(2)
        pub.close(0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
