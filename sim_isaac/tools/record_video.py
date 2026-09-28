"""Record wl-isaac camera streams to mp4 (evidence helper, E5).

    /work/envs/isaaclab/bin/python -m sim_isaac.tools.record_video --port-offset 0 --seconds 60 \
        --out-dir /work/worldline-g1/outputs/m1/run-<ts> [--ego] [--tp]

--ego  SUB tcp://127.0.0.1:5565 (gear_sonic ego_view format; the base64 JPEG decodes to RGB, docs/contracts/m1.md 1.4)
--tp   SUB tcp://127.0.0.1:5602 topic frame.tp (needs the app's --tp-camera; raw JPEG bytes; like the ego stream
       the JPEG was encoded from the RGB array, so cv2.imdecode returns RGB)
Frames are written at the nominal rate (--fps); the frame's sim time is burned into the corner.
"""
from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np


def main():
    import cv2
    import msgpack
    import zmq

    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ego", action="store_true")
    ap.add_argument("--tp", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--tp-fps", type=float, default=10.0)
    ap.add_argument("--topdown", action="store_true",
                    help="top-down mp4 drawn from gt.pose over the cached top-down render (no extra render cost)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = zmq.Context.instance()
    socks = {}
    if a.ego:
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.setsockopt(zmq.RCVHWM, 10)
        s.connect(f"tcp://127.0.0.1:{5565 + a.port_offset}")
        socks["ego"] = s
    if a.tp:
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"frame.tp")
        s.setsockopt(zmq.RCVHWM, 10)
        s.connect(f"tcp://127.0.0.1:{5602 + a.port_offset}")
        socks["tp"] = s
    td = None
    if a.topdown:
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 20000)
        req.connect(f"tcp://127.0.0.1:{5600 + a.port_offset}")
        req.send(msgpack.packb({"op": "render_topdown", "path": str(out / "topdown.png")}))
        info = msgpack.unpackb(req.recv(), raw=False)
        if not info.get("ok"):
            raise SystemExit(f"render_topdown failed: {info}")
        bg = cv2.imread(info["path"])
        scale = 1.0 / info["meters_per_pixel"]
        x0, y0, x1, y1 = info["extent"]
        td = {"bg": bg, "scale": scale, "x0": x0, "y1": y1, "trail": [], "next": 0.0}
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        s.connect(f"tcp://127.0.0.1:{5601 + a.port_offset}")
        socks["topdown"] = s
    if not socks:
        ap.error("pass --ego, --tp and/or --topdown")
    poller = zmq.Poller()
    for s in socks.values():
        poller.register(s, zmq.POLLIN)
    writers, counts = {}, {k: 0 for k in socks}
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        for s, _ in poller.poll(200):
            name = next(k for k, v in socks.items() if v is s)
            if name == "topdown":
                _, body = s.recv_multipart()
                m = msgpack.unpackb(body, raw=False)
                x, y = m["base_pos"][0], m["base_pos"][1]
                px = (int(round((x - td["x0"]) * td["scale"])), int(round((td["y1"] - y) * td["scale"])))
                td["trail"].append(px)
                if m["t_sim"] < td["next"]:
                    continue
                td["next"] = m["t_sim"] + 1.0 / a.tp_fps
                bgr = td["bg"].copy()
                if len(td["trail"]) > 1:
                    cv2.polylines(bgr, [np.array(td["trail"], np.int32)], False, (0, 0, 255), 2)
                hx = px[0] + int(12 * np.cos(m["yaw"]))
                hy = px[1] - int(12 * np.sin(m["yaw"]))
                cv2.circle(bgr, px, 7, (0, 200, 0) if not m["fallen"] else (0, 0, 255), -1)
                cv2.line(bgr, px, (hx, hy), (0, 0, 0), 2)
                cv2.putText(bgr, f"t {m['t_sim']:6.1f}s z {m['pelvis_z']:.2f} {'BAND' if m['band'] else ''}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                t_sim, fps = None, a.tp_fps
            elif name == "ego":
                m = msgpack.unpackb(s.recv(), raw=False)
                rgb = cv2.imdecode(np.frombuffer(base64.b64decode(m["images"]["ego_view"]), np.uint8),
                                   cv2.IMREAD_COLOR)
                bgr = rgb[..., ::-1].copy()  # decoded array is RGB (MuJoCo convention); VideoWriter wants BGR
                t_sim = m.get("t_sim")
                fps = a.fps
            else:
                _, body = s.recv_multipart()
                m = msgpack.unpackb(body, raw=False)
                rgb = cv2.imdecode(np.frombuffer(m["jpeg"], np.uint8), cv2.IMREAD_COLOR)
                bgr = rgb[..., ::-1].copy()
                t_sim = m.get("t_sim")
                fps = a.tp_fps
            if t_sim is not None:
                cv2.putText(bgr, f"t_sim {t_sim:7.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if name not in writers:
                h, w = bgr.shape[:2]
                writers[name] = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writers[name].write(bgr)
            counts[name] += 1
    for w in writers.values():
        w.release()
    print("RECORDED", {k: {"frames": v, "path": str(out / f"{k}.mp4")} for k, v in counts.items()}, flush=True)


if __name__ == "__main__":
    main()
