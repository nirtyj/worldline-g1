"""Live test of the page reset on an Isaac profile (M2b finish, owner body-regress; world-cal's wave-2 change to
ui/server.py: a page or eval `reset` runs P1 `reset_scene` + a body `stand` before the new runtime starts).

Per reset it first disturbs the house the way a session would leave it: the robot walks `--walk-away` m from the
spawn (BodyClient go_to), and two small objects are teleported to the floor next to the spawn (P1 `move_object`,
test-only). Then it sends what the page's reset button sends (`{type: reset, scene, profile}` on P5's websocket, so
`reset_sim` defaults to the robot too) and waits for the new session's `init`. Checked, from sources other than
the server's own claims where possible:

    objects   every dynamic object's live pose (P1 get_objects) vs its load pose (get_scene_info `pos`): the moved
              objects and all the others, right after the init and again --settle-s later (props settle a few mm);
    robot     gt.pose vs the scene spawn (position, yaw), pelvis height, not fallen, standing under SONIC
              (body.state in_control, no fault, no halt latch) for --watch-s after the init;
    session   init.config.sim_reset (ok, objects_reset, robot_reset, the stand), a fresh runtime (no trace rows,
              no events from before the reset), System 1's status in meta, the time from the reset to the init.

    WL_PORT_OFFSET=0 .venv-rt/bin/python -m tools.page_reset_test --url ws://127.0.0.1:8766/ws \\
        --out outputs/m2b_finish/bregress/reset-<ts> [--resets 3] [--walk-away 1.2] [--settle-s 3] [--watch-s 5]

Pass per reset: sim_reset ok, every moved object back within --tol-m (default 0.03 m) of its load pose, the robot
within 0.10 m / 5 deg of the spawn, standing and not fallen through the watch, a fresh session. Writes resets.json.
Runs with the stack lock held; P5 must be up (any brains).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402
from body.config import ep, port_offset_from_env  # noqa: E402
from body.p1_client import P1Rpc, PoseSub  # noqa: E402

# (object ids moved per reset, cycled): small H40 props on the counter, dining table and dresser
MOVE_SETS = [("Mug|surface|6|11", "Bowl|surface|6|6"), ("AlarmClock|surface|2|25", "Vase|surface|6|15"),
             ("WineBottle|surface|6|17", "Mug|surface|2|28")]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


async def recv_until(ws, pred, timeout: float) -> tuple[dict | None, list]:
    seen, t_end = [], time.monotonic() + timeout
    while time.monotonic() < t_end:
        try:
            raw = await asyncio.wait_for(ws.recv(), max(0.05, t_end - time.monotonic()))
        except (asyncio.TimeoutError, TimeoutError):
            break
        if isinstance(raw, (bytes, bytearray)):
            continue
        try:
            m = json.loads(raw)
        except ValueError:
            continue
        if m.get("type") in ("notice", "loading", "init"):
            seen.append({"t": time.monotonic(), "type": m["type"], "text": m.get("text"), "scene": m.get("scene")})
        if pred(m):
            return m, seen
    return None, seen


def centre(o: dict) -> list[float]:
    (x0, y0, z0), (x1, y1, z1) = o["aabb"]
    return [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]


def residuals(p1: P1Rpc, load: dict) -> dict:
    """AABB centre, live (get_objects) vs load (get_scene_info), per dynamic object: the measure P1's own reset check
    uses (sim_isaac/tools/m2b_probe.py check_reset). A live record's `pos` is the prim root, not the box centre."""
    live = {o["id"]: o for o in (p1.call("get_objects") or {}).get("objects") or []}
    out = {}
    for oid, o in live.items():
        if not o.get("dynamic") or oid not in load:
            continue
        out[oid] = round(math.dist(centre(o), centre(load[oid])), 4)
    return out


async def one_reset(k: int, a, p1: P1Rpc, bc: BodyClient, sub: PoseSub, scene_info: dict, ws, scene: str,
                    profile: str) -> dict:
    load = {o["id"]: o for o in scene_info["objects"] if o.get("aabb")}
    sp = scene_info["spawn"]
    r: dict = {"k": k}
    # --- disturb: walk away, move two props to the floor next to the spawn
    fx, fy = math.cos(sp["yaw"]), math.sin(sp["yaw"])
    h = await asyncio.to_thread(bc.go_to, sp["x"] + a.walk_away * fx, sp["y"] + a.walk_away * fy,
                                wrap(sp["yaw"] + math.pi / 2), 90)
    g = sub.latest()
    r["walk_away"] = {"state": h.state, "reason": h.reason,
                      "dist_from_spawn_m": round(math.hypot(g.x - sp["x"], g.y - sp["y"]), 3)}
    moved = MOVE_SETS[k % len(MOVE_SETS)]
    r["moved"] = {}
    lx, ly = -fy, fx
    for j, oid in enumerate(moved):
        o = load.get(oid)
        if o is None:
            r["moved"][oid] = {"error": "not in the scene"}
            continue
        (x0, y0, z0), (x1, y1, z1) = o["aabb"]
        side = 1.0 if j == 0 else -1.0
        tgt = [sp["x"] - 0.5 * fx + side * 0.7 * lx, sp["y"] - 0.5 * fy + side * 0.7 * ly,
               scene_info.get("floor_z", 0.0) + (z1 - z0) / 2 + 0.01]
        rep = p1.call("move_object", id=oid, pose=tgt)
        r["moved"][oid] = {"to": [round(v, 3) for v in tgt], "reply_pos": rep.get("pos")}
    time.sleep(1.0)
    before = residuals(p1, load)
    r["before"] = {"moved": {oid: before.get(oid) for oid in moved},
                   "max_other": max((v for kk, v in before.items() if kk not in moved), default=None)}
    # --- the page's reset button
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "reset", "scene": scene, "profile": profile}))
    init, seen = await recv_until(ws, lambda m: m.get("type") == "init", a.timeout)
    t1 = time.monotonic()
    r["reset_to_init_s"] = round(t1 - t0, 2)
    r["messages"] = [{**m, "t": round(m["t"] - t0, 2)} for m in seen]
    if init is None:
        r["error"] = "no init after the reset"
        return r
    cfg = init.get("config") or {}
    r["sim_reset"] = cfg.get("sim_reset")
    r["session"] = {"error": init.get("error"), "trace_rows": len(init.get("trace") or []),
                    "events": len(init.get("events") or []),
                    "event_types": sorted({e.get("type") for e in init.get("events") or []}),
                    "system1": (init.get("meta") or {}).get("system1"), "scene": cfg.get("scene"),
                    "profile": cfg.get("profile")}
    after = residuals(p1, load)
    r["after"] = {"moved": {oid: after.get(oid) for oid in moved},
                  "max_all": max(after.values(), default=None),
                  "worst": sorted(after.items(), key=lambda kv: -kv[1])[:3]}
    # --- robot: at the spawn, standing, through the watch
    t_w = time.monotonic()
    zmin, fallen = 9.9, False
    while time.monotonic() - t_w < a.watch_s:
        g = sub.latest()
        zmin, fallen = min(zmin, g.pelvis_z), fallen or g.fallen
        await asyncio.sleep(0.02)
    g = sub.latest()
    st = await asyncio.to_thread(bc.status)
    r["robot"] = {"pos_err_m": round(math.hypot(g.x - sp["x"], g.y - sp["y"]), 3),
                  "yaw_err_deg": round(math.degrees(abs(wrap(g.yaw - sp["yaw"]))), 1),
                  "pelvis_z_min": round(zmin, 3), "fallen": fallen, "in_control": st.get("in_control"),
                  "fault": st.get("fault"), "mode": st.get("mode"), "latched": st.get("latched")}
    time.sleep(max(0.0, a.settle_s - a.watch_s))
    settled = residuals(p1, load)
    r["settled"] = {"moved": {oid: settled.get(oid) for oid in moved}, "max_all": max(settled.values(), default=None),
                    "worst": sorted(settled.items(), key=lambda kv: -kv[1])[:3]}
    sr = r["sim_reset"] or {}
    r["pass"] = bool(sr.get("ok") and all((after.get(o) or 9) <= a.tol_m for o in moved)
                     and r["robot"]["pos_err_m"] <= 0.10 and r["robot"]["yaw_err_deg"] <= 5.0
                     and not fallen and st.get("in_control") and not st.get("fault") and not st.get("latched")
                     and r["session"]["trace_rows"] == 0 and not r["session"]["error"])
    return r


async def main_async(a) -> dict:
    from websockets.asyncio.client import connect

    off = port_offset_from_env() if a.port_offset is None else a.port_offset
    p1 = P1Rpc(ep(5600 + off), timeout_s=20.0)
    sub = PoseSub(ep(5601 + off))
    sub.start()
    bc = BodyClient(port_offset=off).connect(10)
    si = p1.call("get_scene_info")
    res: dict = {"tool": "page_reset_test", "url": a.url, "spawn": si.get("spawn"), "resets": []}
    try:
        async with connect(a.url, max_size=2 ** 25, open_timeout=10) as ws:
            first, _ = await recv_until(ws, lambda m: m.get("type") == "init", 60)
            cfg = (first or {}).get("config") or {}
            scene, profile = a.scene or cfg.get("scene"), a.profile or cfg.get("profile")
            res.update({"scene": scene, "profile": profile})
            for k in range(a.resets):
                r = await one_reset(k, a, p1, bc, sub, si, ws, scene, profile)
                res["resets"].append(r)
                print(f"[page_reset] {k}: pass {r.get('pass')} init {r.get('reset_to_init_s')} s sim_reset "
                      f"{(r.get('sim_reset') or {}).get('ok')} ({(r.get('sim_reset') or {}).get('s')} s) moved after "
                      f"{(r.get('after') or {}).get('moved')} max_all {(r.get('after') or {}).get('max_all')} robot "
                      f"{r.get('robot')}", flush=True)
                Path(a.out, "resets.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    finally:
        sub.stop()
        bc.close()
    res["passed"] = sum(1 for r in res["resets"] if r.get("pass"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8766/ws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--resets", type=int, default=3)
    ap.add_argument("--walk-away", type=float, default=1.2)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--watch-s", type=float, default=5.0)
    ap.add_argument("--tol-m", type=float, default=0.03)
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = asyncio.run(main_async(a))
    Path(a.out, "resets.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("PAGE_RESET " + json.dumps({"passed": res["passed"], "of": len(res["resets"])}), flush=True)
    return 0 if res["passed"] == len(res["resets"]) == a.resets else 1


if __name__ == "__main__":
    sys.exit(main())
