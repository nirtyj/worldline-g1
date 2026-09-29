"""GR00T stance probe (M2b wave 2, W2.5): where must the G1 stand so the target sits where Arena's training frames put
the apple, in the lower two-thirds of `ego_view`?

The off-the-shelf checkpoint (nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace, label experimental) saw one framing: the
apple on a surface in front of the robot, left of centre, in the lower part of the Arena head camera (its frame 60 of
episode 0 has the apple's centre at about v = 330 of 480). P1's `ego_view` is that camera exactly (p1_m2b.md §5.1:
1.12-1.14 m high at SONIC's stand, 35 deg down, VFOV 55.3 deg), so the target's image row depends only on its height
and its distance from the pelvis. This tool measures it: per surface, the robot walks to the stand, optionally the
target is staged a few cm behind the front edge (P1 `move_object`, test-only), then for each (forward, left) of a grid
the body's `approach` op (B.6) puts the pelvis so the object sits at that offset in the body frame, facing the
surface, and P1's instance segmentation of ego_view (`detections {camera: ego_view, bbox}`) gives the target's pixels
and bbox. The bar (docs/M2b_wave1.md W2.5): >= 200 px with the bbox centre below the image's upper third (v >= 160).
Each view is saved with the 1/3 line and the bbox drawn. Worldline feasibility of each stance is reported too: the
runtime's reach stance (services/reachability.py find_stance) needs >= 0.25 m of raw clearance around the pelvis.

    WL_PORT_OFFSET=0 .venv-rt/bin/python -m tools.groot_stance_probe --out outputs/m2b_wave2/opsg/stance-<ts> \\
        --surface kitchen_dining_table_1a:potato_1:0.08 --surface kitchen_counter_1b:mug_2:0.08:0.10 \\
        --surface bedroom_dresser_1b:mug_1:0.08:0.10 --forward 0.30,0.35,0.40 --left 0.10,0.20

`--surface KEYPOINT:OBJECT[:DEPTH[:LEFT]]`: stage OBJECT DEPTH m behind the keypoint's front edge and LEFT m to the
robot's left of the keypoint (omit DEPTH to leave it where it is).
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


async def main_async(a: argparse.Namespace) -> dict:
    from api.execution import ExecutionManager
    from groot.stance import ego_view_uv
    from robot.factory import build
    from services.executors.groot_arms import ZmqSensors
    from sim.clock import SimClock
    from tools.groot_live_smoke import _brief, _run_tool, ego_view_check
    from world import coords

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    off = int(os.environ.get("WL_PORT_OFFSET", "0") or 0)
    clock = SimClock(1.0)
    world, robot, _ = build(a.profile, a.scene, clock, frames=None)
    em = ExecutionManager(clock)
    bc = robot.body.client
    grid = world.static_map().grid
    sensors = ZmqSensors(f"tcp://127.0.0.1:{5566 + off}", f"tcp://127.0.0.1:{5557 + off}", "ego_view").start()
    fwds = [float(v) for v in a.forward.split(",")]
    lefts = [float(v) for v in a.left.split(",")]
    res: dict[str, Any] = {"tool": "groot_stance_probe", "scene": a.scene, "forward": fwds, "left": lefts,
                           "bar": "px >= 200 and bbox centre v >= 160 (lower two-thirds of 480)", "surfaces": [],
                           "t_start": time.time()}
    try:
        for spec in a.surface:
            parts = spec.split(":")
            kp_name, oid = parts[0], parts[1]
            depth = float(parts[2]) if len(parts) > 2 and parts[2] else None
            left0 = float(parts[3]) if len(parts) > 3 else 0.0
            kp = world.static_map().keypoint(kp_name)
            surf = world.static_map().surfaces.get(kp_name)
            row: dict[str, Any] = {"keypoint": kp_name, "object": oid, "views": []}
            r = await _run_tool(robot, em, "navigate", {"location": kp_name}, 0)
            row["navigate"] = _brief(r)
            psi = float(kp.yaw)                                       # the stand faces the surface
            c, s = math.cos(psi), math.sin(psi)
            if depth is not None:
                edge = float(getattr(surf, "stand_off_m", 0.35) or 0.35)
                fx, ly = edge + depth, left0
                o = world.object(oid)
                ctr = [kp.x + c * fx - s * ly, kp.y + s * fx + c * ly, o.pos[2]]
                world.move_object(oid, ctr)
                await asyncio.sleep(1.0)
                row["staged"] = {"depth_behind_edge_m": depth, "left_m": left0, "center": [round(v, 3) for v in ctr],
                                 "now": [round(v, 3) for v in world.object(oid).pos]}
            o = world.object(oid)
            row["object_center"] = [round(v, 3) for v in o.pos]
            row["surface_height_m"] = getattr(surf, "z_top", None)
            for f in fwds:
                for l in lefts:
                    sx, sy = o.pos[0] - (c * f - s * l), o.pos[1] - (s * f + c * l)
                    clear = float(grid.clearance(sx, sy))
                    v: dict[str, Any] = {"forward": f, "left": l, "stance": [round(sx, 3), round(sy, 3),
                                                                             round(math.degrees(psi), 1)],
                                         "clearance_m": round(clear, 3),
                                         "worldline_feasible": bool(clear >= 0.25 and grid.is_free(sx, sy))}
                    if clear < a.min_clearance:
                        v["skipped"] = f"clearance {clear:.2f} m < {a.min_clearance:.2f} m"
                        row["views"].append(v)
                        continue
                    h = await asyncio.to_thread(bc.approach, sx, sy, psi, None, (0.03, 3.0), True, 90.0)
                    await asyncio.sleep(a.settle_s)
                    p = world.robot_pose()
                    bf, bl = coords.world_to_body((p.x, p.y, p.yaw), o.pos[0], o.pos[1])
                    pz = getattr(p, "pelvis_z", None) or 0.787
                    pred = ego_view_uv(bf, bl, o.pos[2], pz)
                    v.update(approach={"ok": h.ok, "reason": h.reason, "pos_err": (h.result or {}).get("pos_err")},
                             achieved={"forward": round(bf, 3), "left": round(bl, 3),
                                       "yaw_deg": round(math.degrees(p.yaw), 1), "pelvis_z": round(pz, 3)},
                             predicted_uv=None if pred is None else [round(pred[0], 1), round(pred[1], 1)],
                             fallen=bool(p.fallen))
                    png = out / f"{kp_name}_{oid}_f{int(f * 100):02d}_l{int(round(l * 100)):+03d}.png"
                    v["view"] = await asyncio.to_thread(ego_view_check, world, sensors, oid, png)
                    row["views"].append(v)
                    print(json.dumps({"kp": kp_name, "f": f, "l": l, "achieved": v.get("achieved"),
                                      "px": v["view"].get("px"), "centre_uv": v["view"].get("centre_uv"),
                                      "predicted_uv": v.get("predicted_uv"),
                                      "ok": v["view"].get("ok"), "clear": v["clearance_m"]}), flush=True)
                    (out / "stance_probe.json").write_text(json.dumps(res | {"current": row}, indent=1,
                                                                      default=str) + "\n")
            # back to the stand (a clean start for the next surface's walk)
            r = await _run_tool(robot, em, "navigate", {"location": kp_name}, 0)
            row["back"] = _brief(r)
            res["surfaces"].append(row)
            (out / "stance_probe.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    finally:
        sensors.close()
        await robot.shutdown()
        world.close()
    ok_by = {}
    for row in res["surfaces"]:
        for v in row["views"]:
            key = (v["forward"], v["left"])
            ok_by.setdefault(key, []).append(bool((v.get("view") or {}).get("ok")))
    res["summary"] = {f"f{k[0]:.2f}_l{k[1]:+.2f}": f"{sum(x)}/{len(x)}" for k, x in ok_by.items()}
    res["t_end"] = time.time()
    (out / "stance_probe.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="procthor-train-40")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--surface", action="append", required=True, help="KEYPOINT:OBJECT[:DEPTH[:LEFT]]")
    ap.add_argument("--forward", default="0.30,0.35,0.40")
    ap.add_argument("--left", default="0.10,0.20")
    ap.add_argument("--min-clearance", type=float, default=0.14, help="skip stances closer to furniture (m)")
    ap.add_argument("--settle-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    res = asyncio.run(main_async(a))
    print("STANCE_PROBE " + json.dumps(res.get("summary")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
