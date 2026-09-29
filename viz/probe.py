#!/usr/bin/env python3
"""Listen to every viz-relevant port for a few seconds; print message rates and save one frame per stream.

    python viz/probe.py [--port-offset N] [--seconds 5] [--save DIR]

Head frames are saved both as published (<dir>/head_raw.jpg, R/B swapped per the contract) and fixed (head.jpg);
P1's frame.tp is saved fixed as chase.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.common import decode_head, ep, frame_from_msg, ports, pose_summary, split_msg, swap_rb_jpeg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    p = ports(a.port_offset)
    ctx = zmq.Context.instance()
    socks = {}
    for name, port, topics in (("head", p["head"], [b""]), ("frames", p["frames"], [b"frame."]),
                               ("gt", p["gt_pub"], [b"gt."]), ("body", p["body_evt"], [b""])):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        for t in topics:
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(ep(port))
        socks[s] = name
    poller = zmq.Poller()
    for s in socks:
        poller.register(s, zmq.POLLIN)
    counts: dict[str, int] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    sample: dict[str, object] = {}
    saved: dict[str, str] = {}
    out = Path(a.save) if a.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        for s, _ in poller.poll(100):
            frames = s.recv_multipart()
            src = socks[s]
            now = time.time()
            if src == "head":
                key = "head"
                jpeg, meta = decode_head(frames[-1])
                if out and jpeg and key not in saved:
                    (out / "head_raw.jpg").write_bytes(jpeg)
                    (out / "head.jpg").write_bytes(swap_rb_jpeg(jpeg))
                    saved[key] = str(out / "head.jpg")
                sample[key] = {**meta, "bytes": len(jpeg or b"")}
            else:
                topic, msg = split_msg(frames)
                key = topic or src
                got = frame_from_msg(topic, msg) if isinstance(msg, dict) and "jpeg" in msg else None
                if got is not None:
                    name, jpeg, msg, swap = got   # P1 frame.tp -> "chase", R/B fixed below
                    if out and key not in saved:
                        f = out / f"{name}.jpg"
                        f.write_bytes(swap_rb_jpeg(jpeg, 90) if swap else jpeg)
                        saved[key] = str(f)
                    msg["jpeg_bytes"] = len(jpeg)
                sample[key] = pose_summary(msg) if key == "gt.pose" else msg
            counts[key] = counts.get(key, 0) + 1
            first.setdefault(key, now)
            last[key] = now
    print(f"ports: {json.dumps(p)}")
    for k in sorted(counts):
        dt = last[k] - first[k]
        rate = (counts[k] - 1) / dt if dt > 0 else 0.0
        print(f"  {k:<16} {counts[k]:5d} msgs  {rate:6.1f} Hz")
    for k, v in sample.items():
        print(f"  sample {k}: {json.dumps(v, default=str)[:400]}")
    for k, f in saved.items():
        print(f"  saved {k} -> {f}")
    if not counts:
        print("  nothing received")
    return 0 if counts else 1


if __name__ == "__main__":
    sys.exit(main())
