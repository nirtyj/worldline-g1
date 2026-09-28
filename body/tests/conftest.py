import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Where the upstream WBC clone lives (byte-equality test of the wire format).
WBC_CANDIDATES = [
    os.environ.get("WBC_DIR", ""),
    "/work/repos/GR00T-WholeBodyControl",
    "/private/tmp/claude-501/-Users-nirty-workspace-ludo-interview/0060a66e-327a-4be6-9a73-d1dfd593f49d/scratchpad/repos/"
    "GR00T-WholeBodyControl-sonicagent",
]


def wbc_dir():
    for d in WBC_CANDIDATES:
        if d and os.path.exists(os.path.join(d, "gear_sonic/utils/teleop/zmq/zmq_planner_sender.py")):
            return d
    return None


def _free_block(n: int = 12) -> int:
    """Find a port offset such that every contract port + offset is free (tests run beside other agents)."""
    from body.config import BASE_PORTS

    import random

    for _ in range(200):
        off = random.randrange(20000, 40000, 100)
        ok = True
        for p in BASE_PORTS.values():
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p + off))
            except OSError:
                ok = False
            finally:
                s.close()
            if not ok:
                break
        if ok:
            return off
    raise RuntimeError("no free port block")


@pytest.fixture
def port_offset():
    return _free_block()
