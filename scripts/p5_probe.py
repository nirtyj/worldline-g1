#!/usr/bin/env python3
"""Readiness probe for P5 (ui.server), used by scripts/m2_p5.sh and scripts/m2_up.sh. Prints one JSON line; exit 0
when the page is ready, 1 when not (within --timeout wall seconds).

    .venv-rt/bin/python scripts/p5_probe.py --port 8765 [--scene procthor-train-40] [--profile sonic] [--timeout 300]

Ready means:
  1. GET / answers 200 with the page (ui/index.html);
  2. the websocket /ws sends an `init` whose config names that scene and profile (either may be omitted) and has no
     `error`, i.e. the session built its world, robot, planner and runtime. `loading` messages are waited through.
The line reports what the lead needs at a glance: the scene and profile, the stepping stones this profile loads,
System 1's status, whether the Gemini key is visible to the server, and how long the probe waited.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def http_ok(port: int, host: str = "127.0.0.1", timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=timeout_s) as r:
            body = r.read(4096).decode("utf-8", "replace")
            return r.status == 200 and "<html" in body.lower(), f"HTTP {r.status}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"


async def ws_init(port: int, scene: str | None, profile: str | None, deadline: float,
                  host: str = "127.0.0.1") -> dict[str, Any]:
    """The first `init` for the wanted scene/profile, or the reason there was none before the deadline."""
    from websockets.asyncio.client import connect

    last: dict[str, Any] = {"reason": "no websocket"}
    while time.monotonic() < deadline:
        try:
            async with connect(f"ws://{host}:{port}/ws", max_size=2 ** 24, open_timeout=3) as ws:
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), max(0.1, deadline - time.monotonic()))
                    m = json.loads(raw)
                    if m.get("type") == "loading":
                        last = {"reason": f"loading {m.get('scene')} ({m.get('profile')})"}
                    elif m.get("type") == "init":
                        cfg = m.get("config") or {}
                        if (scene and cfg.get("scene") != scene) or (profile and cfg.get("profile") != profile):
                            last = {"reason": f"init for {cfg.get('scene')} ({cfg.get('profile')}), waiting for "
                                              f"{scene or 'any'} ({profile or 'any'})"}
                            continue
                        return {"init": m}
        except (asyncio.TimeoutError, TimeoutError):
            if last.get("reason") == "no websocket":        # keep what the server said last (loading, wrong init)
                last = {"reason": "no init before the deadline"}
        except OSError as e:
            last = {"reason": f"{type(e).__name__}: {e}"}
        except Exception as e:  # noqa: BLE001  (websockets' own errors: handshake, closed)
            last = {"reason": f"{type(e).__name__}: {e}"}
        await asyncio.sleep(0.5)
    return last


def probe(port: int, scene: str | None = None, profile: str | None = None, timeout_s: float = 30.0,
          host: str = "127.0.0.1") -> dict[str, Any]:
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    out: dict[str, Any] = {"ok": False, "port": port, "scene": scene, "profile": profile}
    ok_http, why = False, ""
    while True:
        ok_http, why = http_ok(port, host)
        if ok_http or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    out["http"] = why
    if not ok_http:
        out.update(reason=f"page not served: {why}", waited_s=round(time.monotonic() - t0, 1))
        return out
    got = asyncio.run(ws_init(port, scene, profile, deadline, host))
    out["waited_s"] = round(time.monotonic() - t0, 1)
    if "init" not in got:
        out["reason"] = got.get("reason")
        return out
    m = got["init"]
    cfg, meta = m.get("config") or {}, m.get("meta") or {}
    err = m.get("error")
    out.update(scene=cfg.get("scene"), profile=cfg.get("profile"), error=str(err)[:300] if err else None,
               stepping_stones=cfg.get("stepping_stones"), system1=meta.get("system1"),
               gemini_key=meta.get("gemini_key"), ok=not err)
    if err:
        out["reason"] = "the session reported an error"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--timeout", type=float, default=30.0, help="wall seconds to wait for readiness")
    a = ap.parse_args(argv)
    res = probe(a.port, a.scene, a.profile, a.timeout, a.host)
    print(json.dumps(res, default=str), flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
