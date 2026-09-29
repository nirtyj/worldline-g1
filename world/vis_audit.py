"""P1.6 hand audit (PLAN §0.10 a): 20 head views where P1's instance-id segmentation (`detections`, the source of
truth for glances, scans and the GR00T view check) and world's gt-geometric visibility (world/perception.py) are drawn
on the SAME rendered frame, for a person (or a model reading the images) to judge which one the pixels support.

    python -m world.vis_audit --port-offset 100 --house procthor-train-40 --views 20 --out DIR [--seed 0]

Runs against a P1 WITHOUT a controller (band on; a stiff DDS peer holds the torso upright, as sim_isaac's
tools/detections_check.py does): each view teleports the robot with `reset_robot` to a surface stand with the yaw
jittered +-25 deg, reads P1's real head-camera pose (`get_link_poses cam:head`), and evaluates world's geometry at
that pose, so the two methods look through one camera. Per view it saves `view_NN.png`:

  green box    object both methods call visible (P1's segmentation bbox, its pixel count)
  red box      P1 only (segmentation sees >= min_px pixels; world's geometry says occluded / too small / too far)
  yellow box   gt-geometric only (world's projected AABB; P1's segmentation has < min_px pixels of it)
  labels       world's reason for every red box; P1's pixel count for every yellow box (0 = none at all)

and `audit.json` with both sets per view. The verdicts (per disagreement: which method the image supports) are
recorded by hand into `verdicts.json` next to it; `--summarize` folds them into the audit's summary.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

from body.config import ep
from body.p1_client import P1Rpc


def camera_pose_from_link(link: dict):
    """P1 `get_link_poses` cam:<name> (x forward, y left, z up) -> world.perception.CameraPose."""
    from sim_isaac.wire import matrix_from_quat
    from world.perception import CameraPose

    R = matrix_from_quat(link["quat_wxyz"])
    fwd = R[:, 0]
    return CameraPose(pos=tuple(float(v) for v in link["pos"]), R=R, yaw=math.atan2(fwd[1], fwd[0]),
                      pitch_down=math.asin(max(-1.0, min(1.0, -fwd[2]))))


def head_frame(sub, after_render_seq: int, timeout_s: float = 3.0) -> dict | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if sub.poll(100):
            m = msgpack.unpackb(sub.recv(), raw=False)
            if m.get("render_seq", 0) >= after_render_seq:
                return m
    return None


def draw(msg: dict, both: list, p1_only: list, geo_only: list, path: Path, title: str) -> None:
    from PIL import Image, ImageDraw

    key = msg.get("camera") or next(iter(msg["images"]))
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(msg["images"][key]))).convert("RGB"))[..., ::-1]
    im = Image.fromarray(np.ascontiguousarray(arr))
    d = ImageDraw.Draw(im)
    for items, colour in ((both, (0, 220, 0)), (p1_only, (235, 40, 40)), (geo_only, (250, 220, 0))):
        for it in items:
            b = it.get("bbox")
            if not b:
                continue
            u0, v0, u1, v1 = (max(-1, min(640, float(x))) for x in b)
            d.rectangle([u0, v0, u1, v1], outline=colour, width=2)
            d.text((u0 + 2, max(0, v0 - 11)), it["label"], fill=colour)
    d.text((4, 466), title, fill=(255, 255, 255))
    im.save(path)


def run(a) -> dict:
    from world.isaac_client import IsaacGTWorldModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
    views = []
    try:
        for i in range(a.views):
            k = kps[i % len(kps)]
            yaw = k.yaw + math.radians(rng.uniform(-25, 25))
            rep = rpc.call("reset_robot", x=k.x, y=k.y, yaw=yaw)
            if not rep.get("ok", True):
                raise SystemExit(f"reset_robot failed: {rep}")
            time.sleep(a.settle_s)
            lp = rpc.call("get_link_poses", links=["cam:head"])["links"]
            cp = camera_pose_from_link(lp["cam:head"])
            sights = {kk: s for kk, s in w.sightings(cp).items() if s.kind == "object"}
            geo = {kk for kk, s in sights.items() if s.visible and s.dist <= a.range}
            det = rpc.call("detections", camera="head", min_px=1, bbox=True, timeout_s=20.0)
            if not det.get("ok", True):
                raise SystemExit(f"detections failed: {det}")
            px = {sid_to_oid[d["id"]]: d for d in det["detections"] if d["id"] in sid_to_oid}
            seg = {o for o, d in px.items() if d["px"] >= a.min_px and (d.get("dist_m") is None or d["dist_m"] <= a.range)}
            both, p1_only, geo_only = [], [], []
            for o in sorted(seg & geo):
                both.append({"id": o, "label": f"{o} {px[o]['px']}px", "bbox": px[o].get("bbox")})
            for o in sorted(seg - geo):
                s = sights.get(o)
                why = (s.reason or ("visible>range" if s.visible else "?")) if s else "not a target"
                p1_only.append({"id": o, "label": f"{o} {px[o]['px']}px (geo: {why})", "bbox": px[o].get("bbox"),
                                "geo_reason": why, "px": px[o]["px"]})
            for o in sorted(geo - seg):
                s = sights[o]
                n = px[o]["px"] if o in px else 0
                geo_only.append({"id": o, "label": f"{o} geo {s.px:.0f}px, seg {n}px", "bbox": s.bbox,
                                 "seg_px": n, "geo_px": round(s.px, 1)})
            m = head_frame(sub, int(det.get("render_seq", 0)))
            v = {"i": i, "keypoint": k.name, "yaw_deg": round(math.degrees(yaw), 1), "cam_pos": list(cp.pos),
                 "cam_pitch_down_deg": round(math.degrees(cp.pitch_down), 2),
                 "both": [x["id"] for x in both], "p1_only": [{k2: x[k2] for k2 in ("id", "px", "geo_reason")}
                                                             for x in p1_only],
                 "geo_only": [{k2: x[k2] for k2 in ("id", "seg_px", "geo_px")} for x in geo_only],
                 "agree": not p1_only and not geo_only, "det_ms": det.get("ms")}
            if m is not None:
                title = (f"view {i:02d} {k.name} yaw {math.degrees(yaw):.0f}  green both / red seg only / "
                         f"yellow geo only")
                draw(m, both, p1_only, geo_only, out / f"view_{i:02d}.png", title)
                v["image"] = f"view_{i:02d}.png"
            views.append(v)
            print(f"[vis_audit] {i:2d} {k.name:30s} both {v['both']} seg-only {[x['id'] for x in p1_only]} "
                  f"geo-only {[x['id'] for x in geo_only]}", flush=True)
    finally:
        sub.close(0)
        rpc.close()
        w.close()
    res = {"house": a.house, "views": len(views), "min_px": a.min_px, "range_m": a.range, "seed": a.seed,
           "agree_views": sum(v["agree"] for v in views),
           "disagreements": sum(len(v["p1_only"]) + len(v["geo_only"]) for v in views), "per_view": views}
    (out / "audit.json").write_text(json.dumps(res, indent=1) + "\n")
    print("VIS_AUDIT " + json.dumps({k: v for k, v in res.items() if k != "per_view"}), flush=True)
    return res


def summarize(out: Path) -> dict:
    """Fold verdicts.json ({"views": {"NN": {"items": {oid: "seg"|"geo"|"unclear"}, "note": str}}}) into a summary:
    per disagreement, which method the rendered image supports."""
    audit = json.loads((out / "audit.json").read_text())
    ver = json.loads((out / "verdicts.json").read_text())
    tally = {"seg": 0, "geo": 0, "unclear": 0}
    rows = []
    for v in audit["per_view"]:
        vv = (ver.get("views") or {}).get(f"{v['i']:02d}") or {}
        items = vv.get("items") or {}
        dis = [x["id"] for x in v["p1_only"]] + [x["id"] for x in v["geo_only"]]
        for o in dis:
            tally[items.get(o, "unclear")] = tally.get(items.get(o, "unclear"), 0) + 1
        rows.append({"i": v["i"], "keypoint": v["keypoint"], "agree": v["agree"], "disagreements": dis,
                     "verdicts": {o: items.get(o, "unclear") for o in dis}, "note": vv.get("note", "")})
    summ = {"views": audit["views"], "agree_views": audit["agree_views"], "disagreements": sum(tally.values()),
            "image_supports": tally, "per_view": rows}
    (out / "audit_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    return summ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=100)
    ap.add_argument("--house", default="procthor-train-40")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--range", type=float, default=2.5)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true", help="fold DIR/verdicts.json into DIR/audit_summary.json")
    a = ap.parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(Path(a.out)), indent=1))
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
