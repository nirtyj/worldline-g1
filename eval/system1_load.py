"""Does System 1 stay fast while the robot moves? About two minutes of driving against a real
Gemini Live connection (GEMINI_API_KEY from .env; some cents a run): a frame and a state every
second to the observer, an observation every 5 s, and a message every few seconds to the router.

    .venv/bin/python -m eval.system1_load [seconds]
    .venv/bin/python -m eval.system1_load 120 --jev    # label with Jev, observe with Gemini

Frames: captured G1 head frames runs/frames/g1_*.jpg (PLAN 8.6, recorded in M2 from the eval houses),
else the M1 head-camera stills under outputs/ (the d435-mount camera), else a stub.

Reports route latency (the target is under 1 s; the runtime's budget is 2 s), observation
latency, fresh sessions, time the router was unavailable, and tokens billed.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

from brains.system1_jev import JevSystemOne
from brains.system1 import MODEL, SystemOne
from tests.system1_live_check import jpeg_from, load_env

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [("hold on", "stop"), ("where are you going?", "question"), ("okay go ahead", "resume"),
            ("no, the apple instead", "correction"), ("also grab the bread", "addition"),
            ("thanks!", "chitchat"), ("stop", "stop"), ("don't go into the bedroom", "constraint")]


def g1_frames() -> list[bytes]:
    """G1 head-camera JPEGs: runs/frames/g1_*.jpg, else the M1 stills in outputs/, else a stub."""
    jpgs = sorted((ROOT / "runs" / "frames").glob("g1_*.jpg"))
    if jpgs:
        return [p.read_bytes() for p in jpgs]
    stills = sorted((ROOT / "outputs").glob("m1*/**/*ego*.png")) + sorted((ROOT / "outputs").glob("m1*/**/*eye*.png"))
    return [jpeg_from(p) for p in stills[:40]] or [b"\xff\xd8"]


async def main(seconds: float) -> int:
    load_env()
    frames = g1_frames()
    if "--jev" in sys.argv:
        s1 = JevSystemOne(os.environ.get("TYPESAFE_API_KEY", ""), os.environ.get("GEMINI_API_KEY", ""),
                          model=os.environ.get("SYSTEM1_MODEL", MODEL), route_timeout=8.0)
    else:
        s1 = SystemOne(os.environ.get("GEMINI_API_KEY", ""), model=os.environ.get("SYSTEM1_MODEL", MODEL),
                       route_timeout=8.0)                    # measure the real latency
    down, t_down = 0.0, None
    task = asyncio.create_task(s1.run())
    t0 = time.monotonic()
    while s1.status != "ready":
        if time.monotonic() - t0 > 20:
            print("not ready:", s1.detail)
            return 1
        await asyncio.sleep(0.1)
    lat, right, obs_lat = [], 0, []
    t0 = time.monotonic()
    i = 0
    observing: asyncio.Task | None = None

    async def observe() -> None:
        t = time.monotonic()
        await s1.observe()
        obs_lat.append(time.monotonic() - t)

    while time.monotonic() - t0 < seconds:
        i += 1
        if s1.router.status != "ready" and t_down is None:
            t_down = time.monotonic()
        if s1.router.status == "ready" and t_down is not None:
            down, t_down = down + time.monotonic() - t_down, None
        await s1.frame(frames[i % len(frames)], f"spot_{i % 7}")
        await s1.update({"at": f"spot_{i % 7}", "moving": True, "goal": "bring the user the mug", "tick": i // 3})
        if i % 5 == 0 and (observing is None or observing.done()):
            observing = asyncio.create_task(observe())
        if i % 4 == 0 and s1.router.status == "ready":
            text, want = MESSAGES[(i // 4) % len(MESSAGES)]
            t = time.monotonic()
            r = await s1.route(text)
            lat.append(time.monotonic() - t)
            right += bool(r and r["kind"] == want)
        await asyncio.sleep(max(0.0, t0 + i - time.monotonic()))     # about one tick a second
    elapsed = time.monotonic() - t0
    task.cancel()
    lat.sort()
    p90 = lat[int(len(lat) * 0.9)] if lat else 0
    print(f"labels {s1.router.model}, observations {s1.observer.model}; {elapsed:.0f}s, {i} frames")
    print(f"routes: {len(lat)}, right {right}; latency median {statistics.median(lat):.2f}s p90 {p90:.2f}s "
          f"max {lat[-1]:.2f}s; under 1 s: {sum(x < 1 for x in lat)}/{len(lat)}")
    if obs_lat:
        print(f"observations: {len(obs_lat)}, latency median {statistics.median(obs_lat):.2f}s max {max(obs_lat):.2f}s")
    r, o = s1.router.stats, s1.observer.stats
    print(f"router: {r.rotations} fresh sessions, context now {r.context_tokens}, tokens {r.total_tokens}, "
          f"unavailable {down:.1f}s ({100 * down / elapsed:.1f}%)")
    print(f"observer: {o.rotations} fresh sessions, context now {o.context_tokens}, tokens {o.total_tokens}")
    errs = [e for e in (r.last_error, o.last_error) if e]
    print("errors:", errs or "none")
    return 0


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a != "--jev"]
    raise SystemExit(asyncio.run(main(float(nums[0]) if nums else 120.0)))
