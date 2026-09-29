"""Serve the page against the offline fakes, for looking at it in a browser (no sim, no LLM).

    uv run --python 3.11 --with websockets --with numpy --with pillow python tests/ui/run_fake_page.py --port 8790

The fake robot walks a small loop, a navigate execution runs with the sonic_walk executor and a
manipulate runs with the sonic_arm_script fallback, and a fake FrameTap publishes head/chase/top
frames (drawn with Pillow when it is installed), so every pane has something to show.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from ui.cameras import TapCameras  # noqa: E402
from ui.server import Hub, serve_hub  # noqa: E402
from ui_fakes import JPEG_A, FakeExecution, FakeTap, make_deps  # noqa: E402


def picture(text: str, w: int = 320, h: int = 240, hue: int = 0) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return JPEG_A
    im = Image.new("RGB", (w, h), (30 + hue % 60, 40, 60))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, w - 10, h - 10], outline=(200, 200, 120), width=3)
    d.text((20, h // 2), text, fill=(240, 240, 240))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=80)
    return out.getvalue()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--seconds", type=float, default=600)
    args = ap.parse_args()
    built: dict = {}
    tap = FakeTap()
    hub = Hub("procthor-train-40", "sonic", deps=make_deps(built), cameras=TapCameras(tap), system1="off")
    server = await serve_hub(hub, "127.0.0.1", args.port)
    await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
    world, robot = built["world"], built["robot"]
    rt = built["runtime"]
    nav = FakeExecution("nav-000001", "navigate", {"location": "bedroom_dresser_1a"}, executor="sonic_walk", action="keypoint")
    man = FakeExecution("man-000002", "manipulate", {"action": "pick", "object_type": "alarm_clock"},
                        executor="sonic_arm_script", action="pick", data={"skill": "sonic.script.pick.v0"}, status="succeeded")
    rt.history += [man, nav]
    robot.execs.append(nav)
    print(f"fake page on http://127.0.0.1:{args.port}", flush=True)
    t, i = 0.0, 0
    while t < args.seconds:
        a = t * 0.3
        world.pose = [2.0 + 1.2 * math.cos(a), 1.5 + 0.9 * math.sin(a), (90 - math.degrees(a + math.pi / 2)) % 360]
        if i % 2 == 0:
            tap.put("head", picture(f"head t={t:.1f}", hue=i), {"t_sim": t, "seq": i})
            tap.put("chase", picture(f"chase t={t:.1f}", 480, 270, hue=i + 20), {"t_sim": t, "seq": i, "source": "tp"})
        if i % 10 == 0:
            tap.put("top", picture("top view", 400, 300), {"seq": i // 10, "t_wall": float(i // 10), "source": "top",
                                                           "extent": [0.0, 0.0, 4.0, 3.0]})
        await hub.tick()
        await asyncio.sleep(0.2)
        t += 0.2
        i += 1
    server.close()


if __name__ == "__main__":
    asyncio.run(main())
