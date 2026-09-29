"""wl-body for the Nav2 fake E2E, with test-only config overrides (body venv).

    WL_FAKE_HEADING_BIAS_KI=0.4 .venv/bin/python -m nav2.tools.fake_body --port-offset 900 --log-dir DIR [body.service args]

Runs body.service.main unchanged; env overrides are applied to the BodyConfig before the service is built:
  WL_FAKE_HEADING_BIAS_KI  planner-frame heading-bias integrator [1/s] (body/config.py heading_bias_ki; M1 default 0,
                           the Nav2 backend was first measured with 0.4). The verifier's narrow-passage lockup depended
                           on it, so the E2E runs the goal set with both values (nav2/tools/run_fake_e2e.sh --ki).
Fakes only: body/config.py stays the single source of the real defaults.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import body.service as service  # noqa: E402


def main(argv=None) -> int:
    ki = os.environ.get("WL_FAKE_HEADING_BIAS_KI")
    orig = service.BodyService.__init__

    def init(self, cfg, *a, **kw):
        if ki not in (None, ""):
            cfg.heading_bias_ki = float(ki)
        print(f"[fake_body] heading_bias_ki={cfg.heading_bias_ki} (env WL_FAKE_HEADING_BIAS_KI={ki!r})", flush=True)
        orig(self, cfg, *a, **kw)

    service.BodyService.__init__ = init
    return service.main(argv)


if __name__ == "__main__":
    sys.exit(main())
