"""Live checks of the P1 M2b wire (docs/contracts/p1_m2b.md §13) against a running P1, over its contract ports only.

    /work/worldline-g1/.venv/bin/python -m sim_isaac.tools.m2b_probe --port-offset 0 --out DIR <check> [...]

checks:
  smoke     ping ops/contract/cameras, get_cameras, head frames (rate, metadata, PNG), ego_view enable -> first frame
            latency, rate, PNG, disable; gt.pose links, gt.objects rate, sim.health, get_link_poses, top renders,
            one head `detections`
  attach    N attach/detach cycles of one dynamic object (--mode follow|fixed_joint): held at the grip point, placed
            back at its load centre; P1 tensor-view errors (get_stats) and REP/gt.pose liveness after every cycle
  latency   push_object and move_object on one object: time until get_objects (polled) and gt.objects (SUB) show
            the move (> 2 cm)
  reset     move a few objects, then reset_scene (optionally with the robot): P1 time, round trip, residuals
  carry     (SONIC standing, body up) attach, walk forward, check the object stayed at the palm, detach ahead
Writes <check>.json into --out and prints a one-line summary. Needs numpy, pyzmq, msgpack, pillow.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
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
from sim_isaac import wire as W  # noqa: E402


def centre(o: dict) -> np.ndarray:
    return W.box_center(o["aabb"])


class Probe:
    def __init__(self, off: int, out: Path):
        self.off = off
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.ctx = zmq.Context.instance()
        self.rpc = P1Rpc(ep(5600 + off), timeout_s=10.0)

    def call(self, op: str, **kw) -> dict:
        t0 = time.monotonic()
        rep = self.rpc.call(op, **kw)
        rep["_rtt_ms"] = round((time.monotonic() - t0) * 1e3, 2)
        return rep

    def sub(self, port: int, topics=(b"",), conflate: bool = False):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        if conflate:
            s.setsockopt(zmq.CONFLATE, 1)
        for t in topics:
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(ep(port))
        return s

    @staticmethod
    def recv(s, timeout_s: float):
        if not s.poll(int(timeout_s * 1000)):
            return None, None
        fr = s.recv_multipart()
        return (fr[0].decode() if len(fr) > 1 else ""), msgpack.unpackb(fr[-1], raw=False)

    def collect(self, s, seconds: float, pred=lambda t, m: True) -> list:
        out, t0 = [], time.monotonic()
        while time.monotonic() - t0 < seconds:
            t, m = self.recv(s, 0.1)
            if m is not None and pred(t, m):
                out.append((time.monotonic(), t, m))
        return out

    def save_frame(self, msg: dict, name: str) -> str:
        """PNG in true colours. The JPEG is cv2.imencode of an RGB array (P1 convention): a standard decoder sees
        R and B swapped, cv2.imdecode returns RGB."""
        import io

        from PIL import Image

        key = msg.get("camera") or next(iter(msg["images"]))
        im = np.asarray(Image.open(io.BytesIO(base64.b64decode(msg["images"][key]))).convert("RGB"))[..., ::-1]
        path = str(self.out / f"{name}.png")
        Image.fromarray(np.ascontiguousarray(im)).save(path)
        return path

    def objects(self, **kw) -> dict:
        return {o["id"]: o for o in self.call("get_objects", **kw)["objects"]}

    def pose(self) -> dict:
        return self.call("get_pose")

    def write(self, name: str, d: dict) -> None:
        (self.out / f"{name}.json").write_text(json.dumps(d, indent=1, default=str) + "\n")

    def pick_object(self, want: str | None) -> str:
        objs = self.objects(dynamic_only=True)
        if want:
            if want in objs:
                return want
            hits = [i for i, o in objs.items() if want.lower() in (o.get("name") or "").lower() or want in i]
            if hits:
                return sorted(hits)[0]
            raise SystemExit(f"no dynamic object matches {want!r}")
        p = self.pose()["base_pos"]
        return min(objs, key=lambda i: float(np.linalg.norm(centre(objs[i])[:2] - np.asarray(p[:2]))))


# ------------------------------------------------------------------------------------------------ checks
def check_smoke(pr: Probe, a) -> dict:
    res: dict = {}
    ping = pr.call("ping")
    res["ping"] = {k: ping.get(k) for k in ("p1_contract", "cameras", "topics", "house_id", "_rtt_ms")}
    res["ping"]["n_ops"] = len(ping.get("ops") or [])
    res["ping"]["ops"] = ping.get("ops")
    cams = pr.call("get_cameras")
    res["cameras"] = cams.get("cameras")
    # head
    s = pr.sub(5565 + pr.off)
    frames = pr.collect(s, 3.0)
    s.close(0)
    if frames:
        m = frames[-1][2]
        res["head"] = {"frames_3s": len(frames), "rx_hz": round((len(frames) - 1) / max(1e-6, frames[-1][0]
                                                                                       - frames[0][0]), 2),
                       "keys": sorted(m.keys()), "camera": m.get("camera"), "hfov": m.get("hfov"),
                       "vfov": m.get("vfov"), "cam_pose_wl": m.get("cam_pose_wl"), "cam_pos": m.get("cam_pos"),
                       "stationary": m.get("stationary"),
                       "age_ms": round((time.time() - m["t_capture"]) * 1e3, 1),
                       "png": pr.save_frame(m, "head")}
    else:
        res["head"] = {"frames_3s": 0}
    # ego_view: enable -> first frame latency -> rate -> PNG -> disable
    s = pr.sub(W.EGO_PORT + pr.off)
    time.sleep(0.3)
    before = pr.collect(s, 0.5)
    t_on = time.monotonic()
    rep = pr.call("camera", name="ego_view", on=True, consumer="m2b_probe", ttl_s=20.0)
    first = pr.collect(s, 2.0)
    frames = first + pr.collect(s, 3.0)
    res["ego_view"] = {"frames_before_enable": len(before), "enable_reply": rep,
                       "first_frame_after_s": round(first[0][0] - t_on, 3) if first else None,
                       "frames": len(frames)}
    if frames:
        m = frames[-1][2]
        res["ego_view"].update({"rx_hz": round((len(frames) - 1) / max(1e-6, frames[-1][0] - frames[0][0]), 2),
                                "camera": m.get("camera"), "hfov": m.get("hfov"), "vfov": m.get("vfov"),
                                "cam_pose_wl": m.get("cam_pose_wl"), "cam_pos": m.get("cam_pos"),
                                "age_ms": round((time.time() - m["t_capture"]) * 1e3, 1),
                                "png": pr.save_frame(m, "ego_view")})
    res["ego_view"]["disable_reply"] = pr.call("camera", name="ego_view", on=False, consumer="m2b_probe")
    after = pr.collect(s, 1.0)
    res["ego_view"]["frames_1s_after_disable"] = len(after)
    s.close(0)
    # gt.* topics
    s = pr.sub(5601 + pr.off, [b"gt.pose", b"gt.objects", b"sim.health"])
    msgs = pr.collect(s, 3.0)
    s.close(0)
    by = {}
    for tm, t, m in msgs:
        by.setdefault(t, []).append((tm, m))
    res["topics"] = {t: {"n": len(v), "hz": round((len(v) - 1) / max(1e-6, v[-1][0] - v[0][0]), 2)}
                     for t, v in by.items()}
    if by.get("gt.pose"):
        res["gt_pose_links"] = by["gt.pose"][-1][1].get("links")
        res["gt_pose_waist_q"] = by["gt.pose"][-1][1].get("waist_q")
    if by.get("gt.objects"):
        res["gt_objects_n"] = len(by["gt.objects"][-1][1]["objects"])
    if by.get("sim.health"):
        res["sim_health"] = by["sim.health"][-1][1]
    objs = pr.call("get_objects")
    res["get_objects"] = {"n": len(objs["objects"]), "dynamic": sum(o["dynamic"] for o in objs["objects"]),
                          "rtt_ms": objs["_rtt_ms"]}
    res["link_poses"] = pr.call("get_link_poses", links=["torso_link", "left_palm", "right_palm", "cam:head",
                                                         "cam:ego_view", "head_link"])
    res["topdown"] = {m: pr.call("render_topdown", mode=m) for m in ("full", "furniture")}
    det = pr.call("detections", camera="head")
    res["detections_head"] = {k: det.get(k) for k in ("ok", "error", "method", "ms", "_rtt_ms", "other_px",
                                                       "cam_pose_wl")}
    res["detections_head"]["detections"] = [(d["id"], d["px"], d.get("dist_m")) for d in det.get("detections", [])]
    st = pr.call("get_stats")
    res["stats"] = {k: st.get(k) for k in ("rtf_1s", "rtf_10s", "render_hz", "render_ms", "cameras", "render_calls",
                                            "n_view_errors", "dynamic_objects", "gt_slowest_ms")}
    res["pass"] = bool(res["head"].get("frames_3s") and res["ego_view"].get("frames")
                       and res["ego_view"]["frames_before_enable"] == 0 and res["ego_view"]["frames_1s_after_disable"]
                       <= 1 and res.get("gt_pose_links") and res.get("sim_health") and det.get("ok"))
    return res


def check_attach(pr: Probe, a) -> dict:
    oid = pr.pick_object(a.id)
    o0 = pr.objects(ids=[oid])[oid]
    c0 = centre(o0)
    cycles, fails = [], []
    st0 = pr.call("get_stats")
    t_start = time.monotonic()
    for i in range(a.n):
        cyc: dict = {"i": i}
        rep = pr.call("attach", id=oid, arm=a.arm, mode=a.mode)
        cyc["attach_ok"], cyc["attach_ms"] = rep.get("ok"), rep["_rtt_ms"]
        if not rep.get("ok"):
            cyc["error"] = rep.get("error")
            fails.append(cyc)
            cycles.append(cyc)
            if rep.get("code") == "mode_disabled":
                break
            continue
        time.sleep(a.hold_s)
        o = pr.objects(ids=[oid])[oid]
        links = pr.pose().get("links") or {}
        palm = links.get(f"{a.arm}_palm")
        grip = W.compose(palm["pos"], palm["quat_wxyz"], W.GRIP_OFFSET, [1, 0, 0, 0])[0] if palm else None
        cyc["held_by"] = o["held_by"]
        cyc["grip_err_m"] = None if grip is None else round(float(np.linalg.norm(centre(o) - grip)), 4)
        rep = pr.call("detach", id=oid, pose=c0.tolist())
        cyc["detach_ok"], cyc["detach_ms"] = rep.get("ok"), rep["_rtt_ms"]
        time.sleep(a.settle_s)
        o = pr.objects(ids=[oid])[oid]
        cyc["place_err_m"] = round(float(np.linalg.norm(centre(o) - c0)), 4)
        cyc["held_after"] = o["held_by"]
        ok = (cyc["attach_ok"] and cyc["detach_ok"] and cyc["held_by"] == a.arm and cyc["held_after"] is None
              and (cyc["grip_err_m"] is None or cyc["grip_err_m"] < a.grip_tol)
              and cyc["place_err_m"] < a.place_tol)
        if (i + 1) % 10 == 0 or not ok:
            st = pr.call("get_stats")
            cyc["n_view_errors"] = st.get("n_view_errors")
            cyc["fixed_joint_disabled"] = st.get("fixed_joint_disabled")
            ok = ok and not st.get("n_view_errors")
        cyc["ok"] = bool(ok)
        if not ok:
            fails.append(cyc)
        cycles.append(cyc)
    st1 = pr.call("get_stats")
    s = pr.sub(5601 + pr.off, [b"gt.pose"])
    alive = pr.collect(s, 1.0)
    s.close(0)
    grip = [c["grip_err_m"] for c in cycles if c.get("grip_err_m") is not None]
    place = [c["place_err_m"] for c in cycles if c.get("place_err_m") is not None]
    return {"object": oid, "name": o0.get("name"), "arm": a.arm, "mode": a.mode, "n": a.n,
            "ok_cycles": sum(1 for c in cycles if c.get("ok")), "fails": fails[:10],
            "grip_err_m": {"max": max(grip) if grip else None, "mean": float(np.mean(grip)) if grip else None},
            "place_err_m": {"max": max(place) if place else None, "mean": float(np.mean(place)) if place else None},
            "attach_ms_max": max((c.get("attach_ms", 0) for c in cycles), default=None),
            "detach_ms_max": max((c.get("detach_ms", 0) for c in cycles), default=None),
            "wall_s": round(time.monotonic() - t_start, 1),
            "view_errors": st1.get("view_errors"), "n_view_errors": st1.get("n_view_errors"),
            "fixed_joint_disabled": st1.get("fixed_joint_disabled"),
            "attach_count": [st0.get("attach_count"), st1.get("attach_count")],
            "rtf_10s_end": st1.get("rtf_10s"), "gt_pose_msgs_1s_after": len(alive),
            "pass": sum(1 for c in cycles if c.get("ok")) == a.n and not st1.get("n_view_errors")
            and len(alive) > 30}


def check_latency(pr: Probe, a) -> dict:
    oid = pr.pick_object(a.id)
    runs = []
    s = pr.sub(5601 + pr.off, [b"gt.objects"])
    time.sleep(0.3)
    for k in range(a.n):
        for how in ("push", "move"):
            o0 = pr.objects(ids=[oid])[oid]
            c0 = centre(o0)
            while pr.recv(s, 0.0)[1] is not None:
                pass
            if how == "push":
                sign = 1 if k % 2 == 0 else -1
                rep = pr.call("push_object", id=oid, vel=[sign * a.push_v, 0.0, 0.0])
            else:
                tgt = c0 + np.array([0.0, 0.05 if k % 2 == 0 else -0.05, 0.0])
                rep = pr.call("move_object", id=oid, pose=tgt.tolist())
            t0 = time.monotonic()
            t_get = t_sub = None
            while time.monotonic() - t0 < 2.0 and (t_get is None or t_sub is None):
                if t_get is None:
                    o = pr.objects(ids=[oid])[oid]
                    if float(np.linalg.norm(centre(o) - c0)) > 0.02:
                        t_get = time.monotonic() - t0
                t, m = pr.recv(s, 0.02)
                if t_sub is None and m is not None:
                    for x in m["objects"]:
                        if x["id"] == oid and float(np.linalg.norm(centre(x) - c0)) > 0.02:
                            t_sub = time.monotonic() - t0
            runs.append({"how": how, "op_ok": rep.get("ok"), "op_ms": rep["_rtt_ms"],
                         "get_objects_s": None if t_get is None else round(t_get, 3),
                         "gt_objects_s": None if t_sub is None else round(t_sub, 3)})
            time.sleep(1.5)     # let a pushed object come to rest
    s.close(0)
    pr.call("reset_scene")
    g = [r["get_objects_s"] for r in runs if r["get_objects_s"] is not None]
    t = [r["gt_objects_s"] for r in runs if r["gt_objects_s"] is not None]
    return {"object": oid, "runs": runs, "get_objects_s_max": max(g) if g else None,
            "gt_objects_s_max": max(t) if t else None,
            "pass": len(g) == len(runs) and len(t) == len(runs) and max(g) < 0.5 and max(t) < 0.5}


def check_reset(pr: Probe, a) -> dict:
    objs0 = pr.objects(dynamic_only=True)
    ids = sorted(objs0)[: a.n_moved]
    for i, oid in enumerate(ids):
        c = centre(objs0[oid]) + np.array([0.1, 0.1 * (i % 3 - 1), 0.0])
        pr.call("move_object", id=oid, pose=c.tolist())
    time.sleep(0.5)
    t0 = time.monotonic()
    rep = pr.call("reset_scene", robot=bool(a.robot))
    wall = time.monotonic() - t0
    time.sleep(1.0)
    objs1 = pr.objects(dynamic_only=True)
    resid = {oid: round(float(np.linalg.norm(centre(objs1[oid]) - centre(objs0[oid]))), 4) for oid in objs0}
    big = {k: v for k, v in resid.items() if v > 0.02}
    return {"reply": rep, "round_trip_s": round(wall, 3), "moved": ids, "residual_max_m": max(resid.values()),
            "residual_gt_2cm": big, "pass": bool(rep.get("ok")) and wall < 30.0 and not big}


def check_carry(pr: Probe, a) -> dict:
    from body.client import BodyClient

    bc = BodyClient(port_offset=pr.off).connect(10)
    oid = pr.pick_object(a.id)
    o0 = pr.objects(ids=[oid])[oid]
    res: dict = {"object": oid}
    res["attach"] = pr.call("attach", id=oid, arm=a.arm, mode=a.mode)
    time.sleep(1.0)
    p0 = pr.pose()
    h = bc.walk(vx=0.35, duration_s=a.walk_s)
    res["walk"] = {"ok": h.ok, "reason": h.reason}
    time.sleep(1.0)
    p1 = pr.pose()
    o = pr.objects(ids=[oid])[oid]
    palm = p1["links"][f"{a.arm}_palm"]
    grip = W.compose(palm["pos"], palm["quat_wxyz"], W.GRIP_OFFSET, [1, 0, 0, 0])[0]
    res["walked_m"] = round(float(np.linalg.norm(np.asarray(p1["base_pos"][:2]) - np.asarray(p0["base_pos"][:2]))), 3)
    res["grip_err_m"] = round(float(np.linalg.norm(centre(o) - grip)), 4)
    res["held_by"] = o["held_by"]
    res["fallen"] = p1["fallen"]
    res["detach"] = pr.call("detach", id=oid, pose=centre(o0).tolist())
    bc.close()
    res["pass"] = bool(res["attach"].get("ok") and h.ok and res["grip_err_m"] < 0.03 and not res["fallen"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    sp = ap.add_subparsers(dest="check", required=True)
    sp.add_parser("smoke")
    p = sp.add_parser("attach")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--mode", default="follow", choices=list(W.ATTACH_MODES))
    p.add_argument("--arm", default="right", choices=list(W.ARMS))
    p.add_argument("--id", default=None)
    p.add_argument("--hold-s", type=float, default=0.3)
    p.add_argument("--settle-s", type=float, default=0.4)
    p.add_argument("--grip-tol", type=float, default=0.03)
    p.add_argument("--place-tol", type=float, default=0.02)
    p = sp.add_parser("latency")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--id", default=None)
    p.add_argument("--push-v", type=float, default=0.6)
    p = sp.add_parser("reset")
    p.add_argument("--n-moved", type=int, default=5)
    p.add_argument("--robot", action="store_true")
    p = sp.add_parser("carry")
    p.add_argument("--id", default=None)
    p.add_argument("--arm", default="right", choices=list(W.ARMS))
    p.add_argument("--mode", default="follow", choices=list(W.ATTACH_MODES))
    p.add_argument("--walk-s", type=float, default=4.0)
    a = ap.parse_args(argv)
    pr = Probe(a.port_offset, Path(a.out))
    fn = {"smoke": check_smoke, "attach": check_attach, "latency": check_latency, "reset": check_reset,
          "carry": check_carry}[a.check]
    res = fn(pr, a)
    name = a.check if a.check != "attach" else f"attach_{a.mode}"
    pr.write(name, res)
    print(f"M2B_PROBE {name} pass={res.get('pass')} -> {pr.out / (name + '.json')}", flush=True)
    return 0 if res.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
