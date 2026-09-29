"""A real Gemini Live session through brains/system1.py: labels and latency for a few messages,
then one observation from a camera frame. Uses GEMINI_API_KEY from .env (a few cents).

    .venv-thor/bin/python -m tests.system1_live_check [path/to/frame.png|jpg]
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
from pathlib import Path

from brains.system1 import MODEL, ROUTE_TIMEOUT_S, SystemOne

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [
    ("Bring me the mug from the kitchen.", "request"),
    ("stop", "stop"),
    ("hold on a sec", "stop"),
    ("okay, go ahead", "resume"),
    ("how long will it take?", "question"),
    ("no, the apple instead", "correction"),
    ("also grab the bread", "addition"),
    ("don't go into the bedroom", "constraint"),
    ("my keys are usually on the counter", "observation"),
    ("thanks!", "chitchat"),
]


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


def jpeg_from(path: Path) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def main(frame_path: Path | None) -> int:
    load_env()
    s1 = SystemOne(os.environ.get("GEMINI_API_KEY", ""),
                   lambda st, d: print(f"  [status] {st} {d[:120]}"), model=os.environ.get("SYSTEM1_MODEL", MODEL))
    task = asyncio.create_task(s1.run())
    t0 = time.monotonic()
    while s1.status != "ready" and time.monotonic() - t0 < 20:
        if s1.status == "error" and not s1.api_key:
            return 2
        await asyncio.sleep(0.1)
    if s1.status != "ready":
        print("not ready after 20 s")
        return 1
    print(f"connected in {time.monotonic() - t0:.2f}s")
    await s1.update({"at": "kitchen_counter_1", "room": "kitchen", "holding": {}, "in_view": ["mug_1", "apple_2"],
                     "goal": None, "stopped": False, "running": [], "robot_last_said": None})
    ok = 0
    lat = []
    for text, expected in MESSAGES:
        if expected == "resume":
            await s1.robot_said("Stopped. Say go ahead when you want me to continue.")
        t = time.monotonic()
        r = await s1.route(text)
        dt = time.monotonic() - t
        lat.append(dt)
        kind = r["kind"] if r else None
        ok += kind == expected
        print(f"  {dt:5.2f}s  {text!r:42} -> {kind!s:12} (expected {expected}) {'' if r is None else r}")
    await s1.robot_said("Want me to check the kitchen?")
    r = await s1.route("sure, have a look")
    print(f"  yes/no answer -> {r}")
    if frame_path is not None:
        await s1.frame(jpeg_from(frame_path), "kitchen_counter_1")
        t = time.monotonic()
        items = await s1.observe()
        print(f"  observe {time.monotonic() - t:.2f}s -> {items}")
    lat.sort()
    print(f"routed {ok}/{len(MESSAGES)} as expected; latency median {lat[len(lat) // 2]:.2f}s, "
          f"max {lat[-1]:.2f}s (budget {ROUTE_TIMEOUT_S}s); stats {s1.stats}")
    task.cancel()
    return 0


if __name__ == "__main__":
    default = ROOT.parent / ".playwright-mcp" / "thor_frame1.png"
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else (default if default.exists() else None)
    raise SystemExit(asyncio.run(main(arg)))
