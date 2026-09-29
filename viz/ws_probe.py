#!/usr/bin/env python3
"""Measure what a browser gets from viz/server.py's WebSocket (run on the LAPTOP through the tunnel).

    uv run --with aiohttp python viz/ws_probe.py --url ws://localhost:8765/ws --seconds 10 --save outputs/viz_ws

Acks every frame like the page does (so flow control behaves the same), counts frames and bytes per stream,
reports publish->laptop latency (box clock offset estimated over HTTP), saves the first frame of each stream, and
optionally sends a body command (--cmd '{"op":"status"}').
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import aiohttp


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/ws")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--cmd", action="append", default=[], help='JSON {"op":..., "args":...}; repeatable')
    a = ap.parse_args()
    out = Path(a.save) if a.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    nbytes: dict[str, int] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    tel = 0
    lat: dict[str, list] = {}
    msgs: list[dict] = []
    async with aiohttp.ClientSession() as sess:
        # clock offset box - laptop from /api/state "t" (NTP-style, best of 5 by round-trip time)
        http = a.url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws", 1)[0]
        best = None
        for _ in range(5):
            t1 = time.time()
            async with sess.get(f"{http}/api/state") as r:
                srv = (await r.json())["t"]
            t2 = time.time()
            if best is None or t2 - t1 < best[0]:
                best = (t2 - t1, srv - (t1 + t2) / 2)
        rtt, offset = best
        print(f"http RTT {rtt * 1000:.0f} ms, clock offset box-laptop {offset * 1000:+.0f} ms")
        async with sess.ws_connect(a.url, max_msg_size=2 ** 22) as ws:
            t0 = time.time()
            for c in a.cmd:
                m = json.loads(c)
                await ws.send_str(json.dumps({"type": "cmd", "op": m["op"], "args": m.get("args", {}), "req": 1}))
            while time.time() - t0 < a.seconds:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                now = time.time()
                if msg.type == aiohttp.WSMsgType.BINARY:
                    buf = msg.data
                    n = struct.unpack(">I", buf[:4])[0]
                    hdr = json.loads(buf[4:4 + n])
                    s = hdr["s"]
                    jpeg = buf[4 + n:]
                    counts[s] = counts.get(s, 0) + 1
                    if hdr.get("t_wall"):
                        lat.setdefault(s, []).append(now + offset - float(hdr["t_wall"]))
                    nbytes[s] = nbytes.get(s, 0) + len(buf)
                    first.setdefault(s, now)
                    last[s] = now
                    if out and counts[s] == 1:
                        (out / f"ws_{s}.jpg").write_bytes(jpeg)
                    await ws.send_str(json.dumps({"type": "ack", "s": s}))
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    if m.get("type") == "tel":
                        tel += 1
                    else:
                        msgs.append(m)
                else:
                    break
            dur = time.time() - t0
    print(f"{a.url}: {dur:.1f} s")
    for s in sorted(counts):
        span = last[s] - first[s]
        fps = (counts[s] - 1) / span if span > 0 else 0.0
        ls = sorted(lat.get(s, []))
        lt = f"  latency p50 {ls[len(ls) // 2] * 1000:5.0f} ms  p90 {ls[int(len(ls) * 0.9)] * 1000:5.0f} ms" if ls else ""
        print(f"  {s:<9} {counts[s]:4d} frames  {fps:5.1f} fps  {nbytes[s] / dur / 1024:7.1f} KiB/s{lt}")
    print(f"  telemetry messages: {tel} ({tel / dur:.1f}/s)")
    for m in msgs[:12]:
        print("  msg:", json.dumps(m)[:300])


if __name__ == "__main__":
    asyncio.run(main())
