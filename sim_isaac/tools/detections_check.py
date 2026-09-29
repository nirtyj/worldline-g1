"""P1.6 exit check: P1 `detections` (instance-id segmentation of the head render) against world's gt-geometric
visibility (world/perception.py through IsaacGTWorldModel, camera head_sim) on sampled views.

Runs where world/ runs (the laptop's .venv-rt, P1's REP/PUB ports forwarded with 00_infra/tunnel.sh), against a
P1 WITHOUT a controller: each view teleports the robot with `reset_robot` (band on, joints at the default pose), so
the head camera sits exactly where world's model puts it.

    .venv-rt/bin/python -m sim_isaac.tools.detections_check --port-offset 100 --views 50 --out DIR [--seed 0]

Views: world's surface keypoints (the stands the planner walks to) with yaw jitter of +-25 deg, cycled until
--views. Per view: world's visible objects (kind object, gt-geometric) vs P1's objects with px >= --min-px within
2.5 m (world's object range). Agreement: exact set equality per view, per-object decisions over the union, and the
same for landmarks (furniture; visible in P1 if any of its scene prims is). --overlay N saves N head frames with
P1's boxes (green = both, red = P1 only) and world-only labels (yellow) for hand checks.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.config import ep  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402


def head_frame(sub, after_render_seq: int, timeout_s: float = 3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if sub.poll(100):
            m = msgpack.unpackb(sub.recv(), raw=False)
            if m.get("render_seq", 0) >= after_render_seq:
                return m
    return None


def draw(msg: dict, p1_dets: list, world_only: list, both: set, path: Path) -> None:
    from PIL import Image, ImageDraw

    key = msg.get("camera") or next(iter(msg["images"]))
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(msg["images"][key]))).convert("RGB"))[..., ::-1]
    im = Image.fromarray(np.ascontiguousarray(arr))
    d = ImageDraw.Draw(im)
    for det in p1_dets:
        u0, v0, u1, v1 = det["bbox"]
        c = (0, 220, 0) if det["id"] in both else (230, 40, 40)
        d.rectangle([u0, v0, u1, v1], outline=c, width=2)
        d.text((u0 + 2, v0 + 2), f"{det.get('name')} {det['px']}", fill=c)
    y = 4
    for name in world_only:
        d.text((4, y), f"world only: {name}", fill=(240, 220, 0))
        y += 12
    im.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--views", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--range", type=float, default=2.5)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--overlay", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from world.isaac_client import IsaacGTWorldModel

    w = IsaacGTWorldModel(port_offset=a.port_offset, camera="head_sim", rpc_timeout_s=10.0)
    rpc = P1Rpc(ep(5600 + a.port_offset), timeout_s=10.0)
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(ep(5565 + a.port_offset))
    rng = random.Random(a.seed)
    kps = [k for k in w.map.keypoints.values() if k.kind == "surface"]
    rng.shuffle(kps)
    sid_to_oid = {s.scene_id: oid for oid, s in w.map.objects.items()}
    lm_sids = {name: set(lm.scene_ids) for name, lm in w.map.landmarks.items()}
    views = []
    agg = {"obj_both": 0, "obj_p1_only": 0, "obj_world_only": 0, "lm_both": 0, "lm_p1_only": 0, "lm_world_only": 0}
    try:
        for i in range(a.views):
            k = kps[i % len(kps)]
            yaw = k.yaw + math.radians(rng.uniform(-25, 25))
            rep = rpc.call("reset_robot", x=k.x, y=k.y, yaw=yaw)
            if not rep.get("ok", True):
                raise SystemExit(f"reset_robot failed: {rep}")
            time.sleep(a.settle_s)
            wd = w.detections(camera="head")
            w_obj = {d.id for d in wd if d.kind == "object"}
            w_lm = {d.id for d in wd if d.kind == "landmark"}
            det = rpc.call("detections", camera="head", min_px=a.min_px, timeout_s=20.0)
            if not det.get("ok", True):
                raise SystemExit(f"detections failed: {det}")
            p1_in_range = [d for d in det["detections"] if d.get("dist_m") is None or d["dist_m"] <= a.range]
            p_sids = {d["id"] for d in p1_in_range}
            p_obj = {sid_to_oid[s] for s in p_sids if s in sid_to_oid}
            all_sids = {d["id"] for d in det["detections"]}
            p_lm = {n for n, sids in lm_sids.items() if sids & all_sids}
            v = {"i": i, "keypoint": k.name, "x": round(k.x, 3), "y": round(k.y, 3), "yaw_deg": round(math.degrees(yaw), 1),
                 "world_objects": sorted(w_obj), "p1_objects": sorted(p_obj), "world_landmarks": sorted(w_lm),
                 "p1_landmarks": sorted(p_lm), "objects_equal": w_obj == p_obj, "landmarks_equal": w_lm == p_lm,
                 "p1_px": {sid_to_oid.get(d["id"], d["id"]): d["px"] for d in det["detections"]},
                 "det_ms": det.get("ms"), "cam_pose_wl": det.get("cam_pose_wl")}
            agg["obj_both"] += len(w_obj & p_obj)
            agg["obj_p1_only"] += len(p_obj - w_obj)
            agg["obj_world_only"] += len(w_obj - p_obj)
            agg["lm_both"] += len(w_lm & p_lm)
            agg["lm_p1_only"] += len(p_lm - w_lm)
            agg["lm_world_only"] += len(w_lm - p_lm)
            if i < a.overlay:
                m = head_frame(sub, int(det.get("render_seq", 0)))
                if m is not None:
                    both_sids = {w.map.objects[o].scene_id for o in (w_obj & p_obj)}
                    oids = {d["id"]: d for d in det["detections"] if d["id"] in sid_to_oid}
                    draw(m, list(oids.values()), sorted(w_obj - p_obj), both_sids, out / f"view_{i:02d}.png")
                    v["overlay"] = f"view_{i:02d}.png"
            views.append(v)
            print(f"[detections_check] {i:2d} {k.name:32s} world {sorted(w_obj)} p1 {sorted(p_obj)} "
                  f"{'=' if v['objects_equal'] else '!='}", flush=True)
    finally:
        sub.close(0)
        rpc.close()
        w.close()
    n = len(views)
    obj_dec = agg["obj_both"] + agg["obj_p1_only"] + agg["obj_world_only"]
    lm_dec = agg["lm_both"] + agg["lm_p1_only"] + agg["lm_world_only"]
    res = {"views": n, "min_px": a.min_px, "range_m": a.range, "seed": a.seed,
           "objects_view_agreement": round(sum(v["objects_equal"] for v in views) / max(1, n), 4),
           "landmarks_view_agreement": round(sum(v["landmarks_equal"] for v in views) / max(1, n), 4),
           "object_decisions": agg, "object_jaccard": round(agg["obj_both"] / obj_dec, 4) if obj_dec else None,
           "landmark_jaccard": round(agg["lm_both"] / lm_dec, 4) if lm_dec else None,
           "views_with_objects": sum(1 for v in views if v["world_objects"] or v["p1_objects"]),
           "per_view": views}
    (out / "detections_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("DETECTIONS_CHECK " + json.dumps({k: v for k, v in res.items() if k != "per_view"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
