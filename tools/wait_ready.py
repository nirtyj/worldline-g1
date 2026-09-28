"""Readiness probes used by scripts/m1_up.sh. Exit 0 when ready, 1 on timeout; prints one JSON line.

    python -m tools.wait_ready p1        [--timeout 900]     # P1 REP answers ping
    python -m tools.wait_ready gtpose    [--min-hz 40]       # gt.pose flowing on 5601 at >= min-hz
    python -m tools.wait_ready camera                        # one decodable sensor_server frame on 5565
    python -m tools.wait_ready lowstate                      # P1 get_stats reports lowstate publishing (> 0 Hz)
    python -m tools.wait_ready deploy                        # deploy alive on 5557 (robot_config or g1_debug)
    python -m tools.wait_ready deploy_log --file LOG         # the deploy printed "Init Done" (INIT ramp finished)
    python -m tools.wait_ready control                       # g1_debug with init_base_quat (deploy in CONTROL)
    python -m tools.wait_ready body                          # body.state heartbeat on 5611
All ports follow --port-offset / WL_PORT_OFFSET.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import zmq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402
from body.wire import decode_camera_message, split_topic  # noqa: E402


def _sub(ctx, port, topics):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    for t in topics:
        s.setsockopt(zmq.SUBSCRIBE, t)
    s.connect(ep(port))
    return s


def probe(what: str, P: dict, a, ctx) -> tuple[bool, dict]:
    if what == "p1":
        rep = P1Rpc(ep(P["p1_rep"]), timeout_s=2.0, ctx=ctx).try_call("ping")
        return bool(rep and rep.get("ok", True)), {"reply": rep}
    if what == "gtpose":
        s = _sub(ctx, P["p1_pose"], [b"gt.pose"])
        ts = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            if s.poll(100) and split_topic(s.recv_multipart(), b"gt.pose") is not None:
                ts.append(time.monotonic())
        s.close(0)
        hz = 0.0 if len(ts) < 2 else (len(ts) - 1) / (ts[-1] - ts[0])
        return hz >= a.min_hz, {"hz": round(hz, 1)}
    if what == "camera":
        s = _sub(ctx, P["camera"], [b""])
        ok, info = False, {}
        if s.poll(3000):
            ts, imgs = decode_camera_message(s.recv())
            ok = bool(imgs)
            info = {"names": list(imgs), "shape": [list(v.shape) for v in imgs.values()][:1]}
        s.close(0)
        return ok, info
    if what == "lowstate":
        rep = P1Rpc(ep(P["p1_rep"]), timeout_s=2.0, ctx=ctx).try_call("get_stats") or {}
        hz = rep.get("lowstate_pub_hz", rep.get("lowstate_hz"))  # m1.md §1.6 get_stats
        return bool(hz and float(hz) > 0), {"lowstate_hz": hz, "stats": rep}
    if what in ("deploy", "control"):
        import msgpack

        s = _sub(ctx, P["sonic_debug"], [b"g1_debug", b"robot_config"])
        ok, info = False, {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0 and not ok:
            if not s.poll(100):
                continue
            f = s.recv_multipart()[0]
            if f.startswith(b"robot_config"):
                info["robot_config"] = True
                ok = what == "deploy"
            elif f.startswith(b"g1_debug"):
                d = msgpack.unpackb(f[len(b"g1_debug"):], raw=False, strict_map_key=False)
                info["g1_debug"] = True
                info["in_control"] = d.get("init_base_quat") is not None
                ok = what == "deploy" or info["in_control"]
        s.close(0)
        return ok, info
    if what == "deploy_log":
        if not a.file or not os.path.exists(a.file):
            return False, {"file": a.file, "exists": False}
        with open(a.file, "rb") as f:
            data = f.read()
        return a.pattern.encode() in data, {"file": a.file}
    if what == "body":
        s = _sub(ctx, P["body_evt"], [b"body.state"])
        ok = s.poll(1500) > 0
        info = json.loads(s.recv_multipart()[1]) if ok else {}
        s.close(0)
        return ok, {"in_control": info.get("in_control"), "fault": info.get("fault")}
    raise SystemExit(f"unknown probe {what}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--min-hz", type=float, default=40.0)
    ap.add_argument("--file", default=None)
    ap.add_argument("--pattern", default="Init Done")
    a = ap.parse_args(argv)
    P = _ports(port_offset_from_env() if a.port_offset is None else a.port_offset)
    ctx = zmq.Context.instance()
    t0 = time.monotonic()
    ok, info = False, {}
    while time.monotonic() - t0 < a.timeout:
        try:
            ok, info = probe(a.what, P, a, ctx)
        except Exception as e:
            ok, info = False, {"error": repr(e)}
        if ok:
            break
        time.sleep(1.0)
    print(json.dumps({"probe": a.what, "ready": ok, "waited_s": round(time.monotonic() - t0, 1), **info},
                     default=str), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
