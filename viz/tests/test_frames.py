"""Frame-format tests for viz (no Isaac): P1 frame.tp aliasing + R/B swap, VizCams pass-through, head decode, and an
end-to-end check through viz/server.py over HTTP.

    cd /work/worldline-g1 && viz/.venv/bin/python -m pytest -q viz/tests      # or: viz/.venv/bin/python viz/tests/test_frames.py
"""

from __future__ import annotations

import base64
import io
import json
import socket
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

from viz.common import decode_head, decode_jpeg, frame_from_msg, split_msg  # noqa: E402
from viz.recorder import Recorder  # noqa: E402
from viz.tap import FrameTap  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
