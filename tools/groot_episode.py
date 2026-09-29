"""One live Worldline "bring me the X" episode on the `full` stack, with the object staged at the GR00T stance (M2b wave
2, W2.7). Runs on the box next to the page (P5 ui.server --profile full), with the runtime venv:

    .venv-rt/bin/python -m tools.groot_episode --object potato_1 --label potato \\
        --stage-at kitchen_dining_table_1a --depth 0.07 --out outputs/m2b_wave2/opsg/episode-<ts>

What it does: loads the scene on the page (reset, forget), stages the object DEPTH m behind the front edge of a
surface in front of its stand (P1 `move_object`, test-only, AFTER the page's reset so the reset cannot undo it), says
"Bring me the <label>." and follows the episode until the object is delivered to the user surface, or the runtime is
idle again, or --timeout. The page's planner and System 1 are whatever P5 was started with (scripts/m2_p5.sh restart
--planner brains.scripted:create --system1 tests.kept.system1_stub:create for the scripted run).

The point is the manipulate path of the `full` profile (policy groot_then_script): groot_arms (the off-the-shelf
Arena N1.7 checkpoint, label experimental; a zero-shot grasp is not expected) first, then the labelled fallback
(sonic_arm_script STEPPING STONE, else kinematic_attach). Written: episode.jsonl (the page's trace, its model calls and
what the robot said), episode.json (the manipulate results with every attempt, the timeline, the verdict).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def stage(scene: str, oid: str, surface: str, depth: float, left: float, off: int) -> dict:
    """Put `oid` depth m behind `surface`'s front edge, `left` m to the left of its stand, through P1 directly."""
    from body.config import ep
    from body.p1_client import P1Rpc
    from robot.factory import build
    from sim.clock import SimClock
    world, _, _ = build("sonic", scene, SimClock(1.0), frames=None)     # the map (stands, edges) and scene ids only
    try:
        m = world.static_map()
        kp, surf = m.keypoint(surface), m.surfaces[surface]
        c, s = math.cos(kp.yaw), math.sin(kp.yaw)
        fx = float(surf.stand_off_m) + depth
        o = world.object(oid)
        z = float(o.pos[2]) - float(o.box[0][2]) + float(surf.z_top)                 # its centre above the top
        ctr = [kp.x + c * fx - s * left, kp.y + s * fx + c * left, z + 0.005]
        rep = P1Rpc(ep(5600 + off), timeout_s=3.0).call("move_object", id=m.objects[oid].scene_id, pose=ctr)
        return {"object": oid, "surface": surface, "depth_behind_edge_m": depth, "left_m": left,
                "center": [round(v, 3) for v in ctr], "p1": {k: rep.get(k) for k in ("id", "pos", "aabb")}}
    finally:
        world.close()


def manip_rows(trace: list[dict]) -> list[dict]:
    keep = ("executor", "skill", "label", "stepping_stone", "reason", "detail", "attempts", "fallback_from",
            "groot", "inferences", "chunks_dropped", "clamped_frac", "slew_frac", "holding")
    out = []
    for r in trace:
        if r.get("type") == "result" and r.get("tool") == "manipulate":
            d = r.get("data") or {}
            out.append({"t": r.get("t"), "status": r.get("status"), "summary": r.get("summary"),
                        **{k: d.get(k) for k in keep if k in d}})
    return out


async def run(a: argparse.Namespace) -> dict:
    from websockets.asyncio.client import connect

    from eval import suite
    off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
    url = a.url or f"ws://127.0.0.1:{8765 + off}/ws"
    res: dict[str, Any] = {"url": url, "scene": a.scene, "object": a.object, "say": f"Bring me the {a.label}.",
                           "t_start": time.time()}
    async with connect(url, max_size=2 ** 24) as ws:
        r = suite.Run(ws, a.profile)
        reader = asyncio.create_task(r.reader())
        try:
            await r.load(a.scene, forget=True)
            res["config"] = (r.init or {}).get("config")
            res["user_surface"] = r.user_surface
            if a.stage_at:
                res["staged"] = await asyncio.to_thread(stage, a.scene, a.object, a.stage_at, a.depth, a.left, off)
                await asyncio.sleep(1.5)                               # gt.objects at 10 Hz -> the page's world
            res["where_before"] = r.where(a.object)
            t0 = time.monotonic()
            await r.say(res["say"])
            delivered = await r.until(lambda: r.where(a.object) == r.user_surface and r.idle(), a.timeout)
            if not delivered:                                          # or it gave up: idle after at least one pick
                await r.until(lambda: r.idle() and r.started("manipulate", action="pick"), 5.0)
            res["wall_s"] = round(time.monotonic() - t0, 1)
            await asyncio.sleep(1.0)
            res["delivered"] = bool(delivered)
            res["where_after"] = r.where(a.object)
            res["manipulate"] = manip_rows(r.trace)
            res["executors"] = r.executors()
            res["metrics"] = r.metrics()
            from eval.offline_episode import timeline
            res["timeline"] = timeline(r.trace)
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            r.dump(out / "episode.jsonl")
        finally:
            reader.cancel()
    res["t_end"] = time.time()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--scene", default="procthor-train-40")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--object", default="potato_1")
    ap.add_argument("--label", default="potato")
    ap.add_argument("--stage-at", default="kitchen_dining_table_1a", help="surface keypoint ('' = no staging)")
    ap.add_argument("--depth", type=float, default=0.07, help="m behind the surface's front edge")
    ap.add_argument("--left", type=float, default=0.0, help="m left of the stand along the edge")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    res = asyncio.run(run(a))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episode.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res.get(k) for k in ("delivered", "where_before", "where_after", "wall_s", "staged")},
                     default=str))
    for m in res.get("manipulate") or []:
        print(json.dumps({k: m.get(k) for k in ("status", "summary", "executor", "reason", "fallback_from")},
                         default=str)[:600])
    print("\n".join(res.get("timeline") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
