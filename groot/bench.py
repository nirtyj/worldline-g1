"""Latency sampler for the GR00T PolicyServer: ping and get_action round trips, measured at the client.

    # dev box, right after `scripts/groot_server.sh start` (the first get_action includes CUDA warm-up):
    python -m groot.bench --first-call --n 100 --dataset <…/lerobot> --episode 0 --frame 60 --out bench.json
    # save that observation for a machine without the dataset (the laptop through tunnel.sh):
    python -m groot.bench --n 0 --dataset <…/lerobot> --save-obs obs_ep0_f60.npz
    python -m groot.bench --n 20 --obs-npz obs_ep0_f60.npz --endpoint tcp://127.0.0.1:5550 --out laptop.json

The observation is a real 640x480 dataset frame + state (build_observation), so the payload is the production one
(about 0.9 MB per request: the frame travels as raw uint8, the stock wire has no compression; 0.15 MB with
--request-image area256, the server's own first resize done by the client, docs/groot_serving.md §9).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np

from . import DEFAULT_ENDPOINT, joint_order as jo
from .actions import to_arm_chunk
from .obs import DEFAULT_PROMPT, REQUEST_IMAGES, build_observation
from .policy_client import PolicyClient


def _pct(a: list[float]) -> dict:
    x = np.asarray(a, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def load_obs_inputs(args) -> dict:
    if args.obs_npz:
        z = np.load(args.obs_npz, allow_pickle=False)
        return {"ego": z["ego"], "q29": z["q29"], "lh": z["lh"], "rh": z["rh"], "prompt": str(z["prompt"])}
    if args.dataset:
        from .dataset import load_episode
        ep = load_episode(args.dataset, args.episode, frames=True)
        f = min(args.frame, ep.n - 1)
        q29, lh, rh = jo.body_from_lerobot43(ep.state[f])
        return {"ego": ep.frames[f], "q29": q29, "lh": lh, "rh": rh, "prompt": args.prompt}
    rng = np.random.default_rng(0)
    return {"ego": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), "q29": np.asarray(jo.STAND_Q29),
            "lh": np.zeros(7), "rh": np.zeros(7), "prompt": args.prompt}


def run(args) -> dict:
    inp = load_obs_inputs(args)
    if args.save_obs:
        np.savez(args.save_obs, ego=inp["ego"], q29=inp["q29"], lh=inp["lh"], rh=inp["rh"], prompt=inp["prompt"])
    res: dict = {"endpoint": args.endpoint, "host": socket.gethostname(), "platform": platform.platform(),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "timeout_s": args.timeout,
                 "obs_source": args.obs_npz or (f"{args.dataset} ep {args.episode} frame {args.frame}"
                                                if args.dataset else "synthetic"),
                 "prompt": inp["prompt"]}
    if args.n <= 0 and not args.first_call:
        return res
    t_b = time.perf_counter()
    obs = build_observation(inp["ego"], inp["q29"], inp["lh"], inp["rh"], inp["prompt"],
                            request_image=args.request_image)
    res["request_image"] = args.request_image
    res["build_ms"] = (time.perf_counter() - t_b) * 1000.0
    with PolicyClient(args.endpoint, timeout_s=args.timeout) as c:
        t0 = time.monotonic()
        ok = c.ping(timeout_s=max(args.timeout, 5.0))
        res["ping_ok"] = ok
        res["first_ping_ms"] = (time.monotonic() - t0) * 1000.0
        if not ok:
            return res
        if args.first_call:
            t0 = time.monotonic()
            act = c.get_action(obs, timeout_s=max(args.timeout, 120.0))
            res["first_call_ms"] = (time.monotonic() - t0) * 1000.0
            res["action_shapes"] = {k: list(v.shape) for k, v in act.items()}
        for _ in range(args.warmup):
            c.get_action(obs)
        pings, acts, fails = [], [], 0
        for _ in range(args.pings):
            t0 = time.monotonic()
            if c.ping():
                pings.append((time.monotonic() - t0) * 1000.0)
        for _ in range(args.n):
            t0 = time.monotonic()
            try:
                act = c.get_action(obs)
            except Exception as e:                       # noqa: BLE001 (count and continue: this is a sampler)
                fails += 1
                res.setdefault("errors", []).append(str(e)[:200])
                continue
            acts.append((time.monotonic() - t0) * 1000.0)
            if args.interval > 0:
                time.sleep(args.interval)
        res["ping_ms"] = _pct(pings)
        res["get_action_ms"] = _pct(acts)
        res["get_action_failures"] = fails
        res["request_bytes"], res["reply_bytes"] = c.last_request_bytes, c.last_reply_bytes
        res["client_stats"] = dict(c.stats)
        if acts:
            ch = to_arm_chunk(act, time.monotonic())
            res["last_chunk"] = {"T": ch.T, "dt": ch.dt, "dropped_keys": sorted(ch.dropped)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m groot.bench")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=50, help="timed get_action calls")
    ap.add_argument("--pings", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.0, help="sleep between calls [s]")
    ap.add_argument("--first-call", action="store_true", help="time the first get_action (fresh server)")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=60)
    ap.add_argument("--obs-npz", default="")
    ap.add_argument("--save-obs", default="")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--request-image", default="full", choices=REQUEST_IMAGES,
                    help="full: the 640x480 frame; area256: the server's first resize done here (groot.obs)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    res = run(args)
    txt = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    print(txt)
    return 0 if res.get("ping_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
