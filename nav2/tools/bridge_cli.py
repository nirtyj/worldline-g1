#!/usr/bin/env python3
"""Tiny client for nav2/ros_bridge.py REP (nav_bridge 5620 + offset). Works in any Python with pyzmq.

    python3 nav2/tools/bridge_cli.py [--port-offset N] ping | stats | status ID | cancel ID | plan X Y | reload_map
    python3 nav2/tools/bridge_cli.py --port-offset N wait [--timeout 120]     # until ping says nav2_ready
Prints JSON; exit 0 iff ok (wait: iff nav2_ready)."""
import argparse
import json
import os
import sys
import time

import zmq


def call(port: int, req: dict, timeout_s: float = 3.0) -> dict:
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.connect(f"tcp://127.0.0.1:{port}")
    try:
        s.send(json.dumps(req).encode())
        return json.loads(s.recv())
    except zmq.Again:
        return {"ok": False, "error": f"no reply from :{port} within {timeout_s} s"}
    finally:
        s.close(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=int(os.environ.get("WL_PORT_OFFSET", "0")))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("cmd")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    port = 5620 + a.port_offset
    if a.cmd == "wait":
        t0, rep = time.monotonic(), {}
        while time.monotonic() - t0 < a.timeout:
            rep = call(port, {"op": "ping"}, 1.0)
            if rep.get("nav2_ready"):
                print(json.dumps({"ready_after_s": round(time.monotonic() - t0, 1), **rep}))
                return 0
            time.sleep(1.0)
        print(json.dumps({"ready": False, **rep}))
        return 1
    req = {"op": a.cmd}
    if a.cmd in ("status", "cancel"):
        req["id"] = a.rest[0]
    elif a.cmd == "plan":
        req.update(x=float(a.rest[0]), y=float(a.rest[1]))
    rep = call(port, req, 10.0)
    print(json.dumps(rep, indent=1))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
