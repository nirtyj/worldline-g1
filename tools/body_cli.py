"""Command line for wl-body (and a raw P1 REP call), used by m1_up.sh / m1_down.sh and by hand.

    python -m tools.body_cli stand [--no-release-band] [--verify-s 3]
    python -m tools.body_cli walk --vx 0.5 [--vy 0] [--yaw-rate 0] --duration 4
    python -m tools.body_cli turn_to --yaw-deg 90 [--relative]
    python -m tools.body_cli go_to --x 3 --y 1 [--yaw-deg 90]
    python -m tools.body_cli stop | status | ping
    python -m tools.body_cli shutdown_control --confirm      # command{stop}: deploy damps & exits (band first!)
    python -m tools.body_cli p1 band on=true                  # raw P1 REP op with key=value args (JSON values)
Prints JSON; exit 0 iff the op succeeded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402
from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402
from body.p1_client import P1Rpc  # noqa: E402


def _kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stand")
    s.add_argument("--no-release-band", action="store_true")
    s.add_argument("--settle-s", type=float, default=2.0)
    s.add_argument("--verify-s", type=float, default=3.0)
    s = sp.add_parser("walk")
    s.add_argument("--vx", type=float, default=0.0)
    s.add_argument("--vy", type=float, default=0.0)
    s.add_argument("--yaw-rate", type=float, default=0.0)
    s.add_argument("--duration", type=float, default=2.0)
    s = sp.add_parser("turn_to")
    s.add_argument("--yaw-deg", type=float, required=True)
    s.add_argument("--relative", action="store_true")
    s = sp.add_parser("go_to")
    s.add_argument("--x", type=float, required=True)
    s.add_argument("--y", type=float, required=True)
    s.add_argument("--yaw-deg", type=float, default=None)
    s.add_argument("--speed", type=float, default=None)
    for c in ("stop", "status", "ping"):
        sp.add_parser(c)
    s = sp.add_parser("shutdown_control")
    s.add_argument("--confirm", action="store_true")
    s = sp.add_parser("p1")
    s.add_argument("op")
    s.add_argument("kv", nargs="*")
    a = ap.parse_args(argv)
    off = port_offset_from_env() if a.port_offset is None else a.port_offset

    if a.cmd == "p1":
        rep = P1Rpc(ep(_ports(off)["p1_rep"]), timeout_s=a.timeout or 10.0).try_call(a.op, **_kv(a.kv))
        print(json.dumps(rep, default=str))
        return 0 if rep is not None and rep.get("ok", True) else 1

    bc = BodyClient(port_offset=off).connect(10)
    try:
        if a.cmd in ("status", "ping"):
            print(json.dumps(bc.status() if a.cmd == "status" else bc.ping(), default=str, indent=1))
            return 0
        if a.cmd == "shutdown_control":
            rep = bc.request("shutdown_control", {"confirm": bool(a.confirm)})
            print(json.dumps(rep, default=str))
            return 0 if rep.get("ok") else 1
        if a.cmd == "stand":
            h = bc.stand(release_band=not a.no_release_band, settle_s=a.settle_s, verify_s=a.verify_s,
                         timeout=a.timeout or 120)
        elif a.cmd == "walk":
            h = bc.walk(a.vx, a.vy, a.yaw_rate, a.duration)
        elif a.cmd == "turn_to":
            h = bc.turn_to(math.radians(a.yaw_deg), relative=a.relative)
        elif a.cmd == "go_to":
            h = bc.go_to(a.x, a.y, yaw=None if a.yaw_deg is None else math.radians(a.yaw_deg), speed=a.speed)
        else:
            h = bc.stop()
        print(json.dumps(h.summary(), default=str))
        return 0 if h.ok else 1
    finally:
        bc.close()


if __name__ == "__main__":
    raise SystemExit(main())
