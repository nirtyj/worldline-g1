"""Fake P1 + fake SONIC deploy for Nav2 tests (body venv):  kinematic G1 in a REAL house occupancy.

    cd /work/worldline-g1 && .venv/bin/python -m nav2.tools.fake_stack --port-offset 400 \
        --house-dir /work/worldline-g1/assets/houses/procthor-train-38 --out DIR

Reuses the body agent's mocks unchanged: tools/fake_deploy.py consumes the SONIC input PUB (5556 + offset) exactly
like gear_sonic_deploy's zmq_manager (1280-byte header, planner frame = GT yaw at start, 1 s planner timeout, g1_debug
with init_base_quat), and integrates SonicMux's planner commands into a twist (first-order speed, heading P-control
with a +3 deg bias); tools/fake_p1.py integrates that twist at 200 Hz with disk-vs-occupancy collisions and publishes
gt.pose (50 Hz) + get_occupancy / get_scene_info from the house dir. Everything is labelled fake.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.fake_deploy import FakeDeploy  # noqa: E402
from tools.fake_p1 import FakeP1  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=400)
    ap.add_argument("--house-dir", default="/work/worldline-g1/assets/houses/procthor-train-38")
    ap.add_argument("--out", default=f"/tmp/nav2-fakes-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--yaw-bias-deg", type=float, default=3.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    p1 = FakeP1(a.port_offset, os.path.join(a.out, "fake_p1"), house_dir=a.house_dir).start()
    dep = FakeDeploy(a.port_offset, yaw_bias_deg=a.yaw_bias_deg).start()
    print(f"[fake_stack] FAKE P1 + FAKE deploy up, offset {a.port_offset}, house {a.house_dir}, "
          f"spawn {p1.spawn}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.wait(5.0) and not p1.shutdown_requested:
        print(f"[fake_stack] pose={p1.x:.2f},{p1.y:.2f} yaw={p1.yaw:.2f} band={p1.band_on} deploy={dep.state} "
              f"planner_hz={dep.planner_rate():.1f} collisions={p1.collisions}", flush=True)
    dep.stop()
    p1.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
