"""One-line-per-second monitor of the M1 stack from body.state (5611) + gt.pose (5601).

    python -m tools.m1_monitor [--port-offset N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from body.client import BodyClient  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--period", type=float, default=1.0)
    a = ap.parse_args(argv)
    while True:
        try:
            bc = BodyClient(port_offset=a.port_offset).connect(30)
            break
        except TimeoutError:
            print(f"{time.strftime('%H:%M:%S')} waiting for body.state ...", flush=True)
    try:
        while True:
            st = bc.last_state or {}
            p = st.get("pose") or {}
            act = st.get("active") or {}
            dep = st.get("deploy") or {}
            mux = st.get("mux") or {}
            gt = st.get("gt_pose") or {}
            print(f"{time.strftime('%H:%M:%S')} ctl={int(bool(st.get('in_control')))} fault={st.get('fault')} "
                  f"pose=({p.get('x', float('nan')):.2f},{p.get('y', float('nan')):.2f},"
                  f"{p.get('yaw', float('nan')):.2f}) z={p.get('pelvis_z', float('nan')):.2f} v={p.get('v', 0):.2f} "
                  f"rtf={gt.get('rtf')} gt={gt.get('rate_hz')}Hz dbg={dep.get('rate_hz')}Hz mux={mux.get('rate_hz')}Hz "
                  f"op={act.get('op')}:{act.get('phase')}", flush=True)
            time.sleep(a.period)
    except KeyboardInterrupt:
        return 0
    finally:
        bc.close()


if __name__ == "__main__":
    raise SystemExit(main())
