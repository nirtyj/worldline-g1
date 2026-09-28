"""Read-only compatibility probe of a running P1 with wl-body's own client code (no band/reset/motion ops).

    python -m tools.probe_p1 --port-offset 100 [--seconds 5] [--occupancy]

Checks: gt.pose via body.p1_client.PoseSub (rate, fields, parse), camera via body.wire.decode_camera_message,
REP ping / get_scene_info / get_stats via body.p1_client.P1Rpc, and (house scenes) NavGrid.from_p1_reply on the
scene's occupancy npz + A* between room points. --occupancy calls get_occupancy on P1 (the empty scene generates it
with PhysX overlaps once, which can stall a run; houses just return a cached file). Prints one JSON document.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402
from body.nav_grid import NavGrid  # noqa: E402
from body.p1_client import P1Rpc, PoseSub  # noqa: E402
from body.wire import decode_camera_message  # noqa: E402

POSE_FIELDS = ["t_sim", "t_wall", "rtf", "base_pos", "base_quat_wxyz", "base_lin_vel_w", "base_ang_vel_w", "yaw",
               "pelvis_z", "fallen", "foot_contact"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--occupancy", action="store_true")
    a = ap.parse_args(argv)
    P = _ports(port_offset_from_env() if a.port_offset is None else a.port_offset)
    ctx = zmq.Context.instance()
    out: dict = {"ports": P}
    sub = PoseSub(ep(P["p1_pose"]), ctx=ctx)
    sub.start()
    cam = ctx.socket(zmq.SUB)
    cam.setsockopt(zmq.LINGER, 0)
    cam.setsockopt(zmq.SUBSCRIBE, b"")
    cam.connect(ep(P["camera"]))
    rpc = P1Rpc(ep(P["p1_rep"]), timeout_s=5.0, ctx=ctx)
    t0 = time.monotonic()
    out["ping"] = rpc.try_call("ping")
    cams = []
    while time.monotonic() - t0 < a.seconds:
        if cam.poll(100):
            ts, imgs = decode_camera_message(cam.recv())
            cams.append({"t": time.time(), "names": list(imgs), "shape": [list(v.shape) for v in imgs.values()],
                         "latency_ms": round((time.time() - float(next(iter(ts.values())))) * 1e3, 1) if ts else None})
    p = sub.latest()
    out["gt_pose"] = {"received": sub.count, "rate_hz": round(sub.rate_hz(), 2), "parse_errors": sub.parse_errors,
                      "missing_fields": None if p is None else [f for f in POSE_FIELDS if f not in p.raw],
                      "extra_fields": None if p is None else sorted(set(p.raw) - set(POSE_FIELDS)),
                      "latest": None if p is None else p.brief(), "rtf": None if p is None else p.rtf,
                      "velocity_from_body": [round(v, 4) for v in sub.velocity()]}
    out["camera"] = {"frames": len(cams), "rate_hz": round(len(cams) / a.seconds, 2),
                     "first": cams[0] if cams else None}
    stats = rpc.try_call("get_stats") or {}
    out["get_stats"] = {k: stats.get(k) for k in ("rtf_1s", "rtf_10s", "rtf_total", "physics_hz_1s", "render_hz",
                                                   "camera_pub_hz", "lowstate_pub_hz", "gt_pose_hz",
                                                   "lowcmd_fresh_hz", "lowcmd_leg_change_hz", "root_writes",
                                                   "physx_device", "house", "camera")}
    info = rpc.try_call("get_scene_info") or {}
    out["scene"] = {"house_id": info.get("house_id"), "rooms": [r.get("name") for r in info.get("rooms") or []],
                    "n_objects": len(info.get("objects") or []), "spawn": info.get("spawn"),
                    "occupancy_npz": info.get("occupancy_npz"), "keys": sorted(info)}
    occ_rep = None
    if a.occupancy:
        occ_rep = rpc.try_call("get_occupancy", robot_radius=0.25)
    elif info.get("occupancy_npz") and os.path.exists(str(info["occupancy_npz"])):
        occ_rep = {"ok": True, "path": info["occupancy_npz"]}
    if occ_rep and occ_rep.get("ok", True) and occ_rep.get("path"):
        nav = NavGrid.from_p1_reply(occ_rep, robot_radius=0.25)
        plans = []
        pts = [(r["x"], r["y"], r.get("room")) for r in (info.get("room_points") or []) if r.get("ok", True)]
        if p is not None and pts:
            for (x, y, name) in pts:
                res = nav.plan((p.x, p.y), (x, y))
                plans.append({"to": name, "ok": res.ok, "reason": res.reason, "length_m": round(res.length, 2),
                              "straight_m": round(math.hypot(x - p.x, y - p.y), 2), "plan_ms": round(res.plan_ms, 1)})
        out["nav"] = {"shape": [nav.H, nav.W], "res": nav.res, "origin": nav.origin,
                      "free_frac_inflated": round(float((~nav.inflated).mean()), 3),
                      "robot_cell_free": None if p is None else nav.is_free(p.x, p.y), "plans_from_robot": plans}
    sub.stop()
    cam.close(0)
    rpc.close()
    print(json.dumps(out, indent=1, default=str))
    ok = bool(out["ping"]) and sub.count > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
