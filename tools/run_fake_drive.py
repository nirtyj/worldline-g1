"""Run the full M1 drive test against the fakes in one process (fake P1 + fake deploy + real wl-body).

    python -m tools.run_fake_drive --port-offset 200 --stand-s 10 --out /tmp/m1-fake

Everything goes over the real ZMQ contract; only P1 and the deploy are mocks. Outputs are labelled fake
(metrics.json p1_is_fake=true, plot title), so they can never be mistaken for Isaac/SONIC evidence.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.config import BodyConfig  # noqa: E402
from body.service import BodyService  # noqa: E402
from tools import m1_drive_test  # noqa: E402
from tools.fake_deploy import FakeDeploy  # noqa: E402
from tools.fake_p1 import FakeP1  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=200)
    ap.add_argument("--stand-s", type=float, default=10.0)
    ap.add_argument("--out", default=f"/tmp/m1-fake-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--house-dir", default=None, help="real house dir (occupancy.npz + house_info.json)")
    ap.add_argument("--tests", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    p1 = FakeP1(a.port_offset, os.path.join(a.out, "fake_p1"), house_dir=a.house_dir).start()
    dep = FakeDeploy(a.port_offset).start()
    svc = BodyService(BodyConfig(port_offset=a.port_offset), log_dir=os.path.join(a.out, "body"))
    th = threading.Thread(target=svc.run, daemon=True)
    th.start()
    time.sleep(1.0)
    args = ["--port-offset", str(a.port_offset), "--stand-s", str(a.stand_s), "--out", a.out]
    if a.no_video:
        args.append("--no-video")
    if a.tests:
        args += ["--tests", a.tests]
    try:
        rc = m1_drive_test.main(args)
    finally:
        svc.stop()
        th.join(5)
        dep.stop()
        p1.stop()
    print(f"[run_fake_drive] deploy stats: {dict(dep.stats)} modes={dict(dep.modes_seen)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
