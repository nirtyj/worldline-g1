"""Open-loop check of the whole runtime-side GR00T path against recorded demos (dev box).

For each episode of nvidia/Arena-G1-Static-PickNPlace-Task and each query step t (every `--stride` rows):

    dataset row t (LeRobot 43, GR00T order) --joint_order--> body q 29 (MuJoCo) + hands (Dex3)   [what g1_debug gives]
    --groot.obs.build_observation--> PolicyClient.get_action --> 40-step chunk per key
    compare with the recorded actions t .. t+39 (per key MSE / MAE, raw and in the checkpoint's normalized units)
    --groot.actions.to_arm_chunk--> SONIC upper body + Dex3 hands --> fraction outside the URDF limits

The episodes are the checkpoint's own training demos (its model card names this dataset), so a low error proves the
plumbing (keys, joint orders, Dex3 reorder, normalization, wire) rather than generalization. Controls:
    baseline_hold    no model: every step predicts the current state (what a dead policy would score)
    prompt_dataset   the sentence in the released tasks.jsonl instead of Arena's eval instruction
    neg_swap_arms    left/right arm states swapped before build_observation (a joint-order bug)
    neg_hand_order   hands handed over in GR00T order instead of Dex3 order (a missed reorder)

Run (box, /work/groot/venv: numpy, pyzmq, msgpack, pyarrow, matplotlib; ffmpeg CLI):
    python -m groot.open_loop --dataset /work/groot/datasets/Arena-G1-Static-PickNPlace-Task/lerobot \
        --episodes 0 1 2 3 4 --ckpt <ckpt dir> --out /work/groot/eval/openloop-<ts>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import DEFAULT_ENDPOINT, joint_order as jo
from .actions import clamp_stats, to_arm_chunk
from .dataset import Episode, load_episode
from .obs import ARENA_PROMPT, build_observation
from .policy_client import PolicyClient
from .urdf import BOX_URDF, load_limits

JOINT_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
ALL_KEYS = JOINT_KEYS + ("base_height_command", "navigate_command")
VARIANTS = ("main", "prompt_dataset", "baseline_hold", "neg_swap_arms", "neg_hand_order")


def gt_keys(ep: Episode, t: int, horizon: int) -> dict[str, np.ndarray]:
    """Recorded actions t .. t+horizon-1 (truncated at the episode end) per GR00T key."""
    a = ep.action[t: t + horizon]
    out = jo.groot_keys_from_lerobot43(a)
    out["base_height_command"] = ep.base_height[t: t + horizon]
    out["navigate_command"] = ep.navigate[t: t + horizon]
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def norm_half_range(stats: dict) -> dict[str, np.ndarray]:
    """(q99 - q01) / 2 per action key: 1.0 in these units = the full [-1, 1] normalized range / 2."""
    out = {}
    for k, v in stats.items():
        hr = (np.asarray(v["q99"], dtype=np.float64) - np.asarray(v["q01"], dtype=np.float64)) / 2.0
        out[k] = np.maximum(hr, 1e-8)
    return out


class Acc:
    """Squared / absolute error accumulators per key."""

    def __init__(self):
        self.se: dict[str, float] = {}
        self.ae: dict[str, float] = {}
        self.nse: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.first10_se: dict[str, float] = {}
        self.first10_n: dict[str, int] = {}

    def add(self, key: str, pred: np.ndarray, gt: np.ndarray, half_range: np.ndarray | None):
        m = min(pred.shape[0], gt.shape[0])
        d = pred[:m] - gt[:m]
        self.se[key] = self.se.get(key, 0.0) + float((d ** 2).sum())
        self.ae[key] = self.ae.get(key, 0.0) + float(np.abs(d).sum())
        self.n[key] = self.n.get(key, 0) + d.size
        f = d[:10]
        self.first10_se[key] = self.first10_se.get(key, 0.0) + float((f ** 2).sum())
        self.first10_n[key] = self.first10_n.get(key, 0) + f.size
        if half_range is not None:
            self.nse[key] = self.nse.get(key, 0.0) + float(((d / half_range) ** 2).sum())

    def summary(self) -> dict:
        out = {}
        for k in self.n:
            n = max(self.n[k], 1)
            out[k] = {"mse": self.se[k] / n, "mae": self.ae[k] / n, "values": self.n[k],
                      "mse_first10": self.first10_se[k] / max(self.first10_n[k], 1)}
            if k in self.nse:
                out[k]["nmse"] = self.nse[k] / n
        return out


def _obs_for(variant: str, ep: Episode, t: int, prompt_main: str):
    q29, lh, rh = jo.body_from_lerobot43(ep.state[t])
    prompt = ep.task if variant == "prompt_dataset" else prompt_main
    if variant == "neg_swap_arms":
        q29 = q29.copy()
        la = jo.index_map(jo.MUJOCO_JOINTS, jo.LEFT_ARM_JOINTS)
        ra = jo.index_map(jo.MUJOCO_JOINTS, jo.RIGHT_ARM_JOINTS)
        q29[la], q29[ra] = q29[ra].copy(), q29[la].copy()
    if variant == "neg_hand_order":
        lh = jo.reorder(ep.state[t], jo.LEROBOT_43_JOINTS, jo.GROOT_HAND_JOINTS["left"])
        rh = jo.reorder(ep.state[t], jo.LEROBOT_43_JOINTS, jo.GROOT_HAND_JOINTS["right"])
    return build_observation(ep.frames[t], q29, lh, rh, prompt)


def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = None
    if args.ckpt:
        s = json.loads((Path(args.ckpt) / "statistics.json").read_text())
        stats = norm_half_range(s[jo.EMBODIMENT_TAG]["action"])
    limits = load_limits(args.urdf) if args.urdf and Path(args.urdf).exists() else dict(jo.URDF_LIMITS)
    limits_src = args.urdf if args.urdf and Path(args.urdf).exists() else "groot.joint_order.URDF_LIMITS"
    variants = [v for v in args.variants.split(",") if v]
    accs = {v: Acc() for v in variants}
    per_ep: dict[int, dict[str, dict]] = {}
    lat_ms: list[float] = []
    clamp = {m: {"targets": 0, "clamped": 0, "arm_clamped": 0, "hand_clamped": 0, "per_joint": {},
                 "max_violation_rad": 0.0} for m in (0.0, 0.02)}
    dropped_ranges: dict[str, list[float]] = {}
    plots: dict[int, dict] = {}
    req_bytes = rep_bytes = 0
    n_queries = 0
    client = PolicyClient(args.endpoint, timeout_s=args.timeout)
    try:
        if not client.ping(timeout_s=5.0):
            raise SystemExit(f"no PolicyServer at {args.endpoint}")
        for ei in args.episodes:
            ep = load_episode(args.dataset, ei, frames=True)
            ep_acc = {v: Acc() for v in variants}
            qs = list(range(0, ep.n, args.stride))
            plots[ei] = {"t": [], "pred": [], "n": ep.n, "gt_left_arm": jo.groot_keys_from_lerobot43(ep.action)[
                "left_arm"].tolist(), "gt_left_hand": jo.groot_keys_from_lerobot43(ep.action)["left_hand"].tolist()}
            for t in qs:
                gt = gt_keys(ep, t, jo.ACTION_HORIZON)
                for v in variants:
                    if v == "baseline_hold":
                        st = jo.groot_keys_from_lerobot43(ep.state[t])
                        pred = {k: np.repeat(np.asarray(st[k], dtype=np.float64)[None], jo.ACTION_HORIZON, 0)
                                for k in JOINT_KEYS}
                    else:
                        obs = _obs_for(v, ep, t, args.prompt)
                        t0 = time.monotonic()
                        raw = client.get_action(obs)
                        if v == "main":
                            lat_ms.append((time.monotonic() - t0) * 1000.0)
                            req_bytes, rep_bytes = client.last_request_bytes, client.last_reply_bytes
                            n_queries += 1
                        pred = {k: np.asarray(raw[k], dtype=np.float64)[0] for k in ALL_KEYS if k in raw}
                        if v == "main":
                            missing = [k for k in ALL_KEYS if k not in raw]
                            if missing:
                                raise SystemExit(f"server reply lacks keys {missing} (got {sorted(raw)})")
                            shapes = {k: tuple(raw[k].shape) for k in ALL_KEYS}
                            for k in ALL_KEYS:
                                if shapes[k] != (1, jo.ACTION_HORIZON, jo.GROOT_KEY_DIMS[k]):
                                    raise SystemExit(f"bad shape {k}: {shapes[k]}")
                            chunk = to_arm_chunk(raw, t0)
                            # the SONIC mapping must carry exactly GR00T's arm/hand values (by name)
                            assert np.allclose(chunk.upper_body_mj17[:, 3:10], pred["left_arm"])
                            assert np.allclose(chunk.upper_body_mj17[:, 10:17], pred["right_arm"])
                            assert np.allclose(jo.dex3_to_groot_hand("left", chunk.left_hand), pred["left_hand"])
                            for m in clamp:
                                cs = clamp_stats(chunk, margin=m, limits=limits)
                                c = clamp[m]
                                for f in ("targets", "clamped", "arm_clamped", "hand_clamped"):
                                    c[f] += cs[f]
                                c["max_violation_rad"] = max(c["max_violation_rad"], cs["max_violation_rad"])
                                for j, n in cs["per_joint"].items():
                                    c["per_joint"][j] = c["per_joint"].get(j, 0) + n
                            for k, a in chunk.dropped.items():
                                r = dropped_ranges.setdefault(k, [float("inf"), float("-inf")])
                                r[0], r[1] = min(r[0], float(a.min())), max(r[1], float(a.max()))
                            plots[ei]["t"].append(t)
                            plots[ei]["pred"].append({"left_arm": pred["left_arm"].tolist(),
                                                      "left_hand": pred["left_hand"].tolist()})
                    for k in pred:
                        hr = stats.get(k) if stats else None
                        accs[v].add(k, pred[k], gt[k], hr)
                        ep_acc[v].add(k, pred[k], gt[k], hr)
            per_ep[ei] = {v: {k: round(s["mse"], 6) for k, s in a.summary().items()} for v, a in ep_acc.items()}
            per_ep[ei]["_meta"] = {"rows": ep.n, "queries": len(qs), "task_index": ep.task_index, "task": ep.task}
            print(f"[open_loop] episode {ei}: {ep.n} rows, {len(qs)} queries; main left_arm mse "
                  f"{per_ep[ei].get('main', {}).get('left_arm')}", flush=True)
    finally:
        client.close()
    lat = np.asarray(lat_ms)
    res = {
        "what": "GR00T N1.7 open-loop check vs recorded Arena demos (training episodes; plumbing check, not skill)",
        "label": "experimental",
        "endpoint": args.endpoint, "ckpt": args.ckpt, "dataset": args.dataset, "episodes": args.episodes,
        "stride": args.stride, "horizon": jo.ACTION_HORIZON, "prompt_main": args.prompt,
        "variants": {v: a.summary() for v, a in accs.items()},
        "per_episode": per_ep,
        "clamp": {f"margin_{m}": {**c, "clamped_frac": c["clamped"] / c["targets"] if c["targets"] else 0.0}
                  for m, c in clamp.items()},
        "clamp_limits_source": limits_src,
        "dropped_ranges": dropped_ranges,
        "latency_ms_during_eval": {"n": int(lat.size), "p50": float(np.percentile(lat, 50)) if lat.size else None,
                                   "p95": float(np.percentile(lat, 95)) if lat.size else None,
                                   "mean": float(lat.mean()) if lat.size else None,
                                   "max": float(lat.max()) if lat.size else None},
        "payload_bytes": {"request": req_bytes, "reply": rep_bytes},
        "queries_main": n_queries,
        "client_stats": client.stats,
    }
    (out_dir / "open_loop.json").write_text(json.dumps(res, indent=1))
    (out_dir / "open_loop_traces.json").write_text(json.dumps(plots))
    if args.plots:
        _plot(plots, out_dir)
    return res


def _plot(plots: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ei, p in plots.items():
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        names = list(jo.LEFT_ARM_JOINTS) + ["left_hand_index_0", "left_hand_index_1", "left_hand_thumb_1",
                                            "left_hand_thumb_2", "left_hand_middle_0"]
        hand_cols = [0, 1, 5, 6, 2]
        gt_a = np.asarray(p["gt_left_arm"])
        gt_h = np.asarray(p["gt_left_hand"])
        for i, ax in enumerate(axes.flat[: len(names)]):
            gt = gt_a[:, i] if i < 7 else gt_h[:, hand_cols[i - 7]]
            ax.plot(np.arange(len(gt)) / jo.CONTROL_HZ, gt, color="k", lw=1.5, label="recorded action")
            for t, pr in zip(p["t"], p["pred"]):
                arr = np.asarray(pr["left_arm"])[:, i] if i < 7 else np.asarray(pr["left_hand"])[:, hand_cols[i - 7]]
                ax.plot((t + np.arange(len(arr))) / jo.CONTROL_HZ, arr, color="tab:orange", lw=1)
            ax.set_title(names[i].replace("_joint", ""), fontsize=9)
        for ax in axes.flat[len(names):]:
            ax.axis("off")
        axes.flat[0].legend(["recorded", "GR00T chunks"], fontsize=8)
        fig.suptitle(f"Arena G1 static apple, episode {ei}: GR00T N1.7 40-step chunks (orange) vs recorded (black)")
        fig.supxlabel("time [s]")
        fig.tight_layout()
        fig.savefig(out_dir / f"open_loop_ep{ei:03d}.png", dpi=90)
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m groot.open_loop")
    ap.add_argument("--dataset", required=True, help="…/Arena-G1-Static-PickNPlace-Task/lerobot")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--ckpt", default="", help="checkpoint dir (statistics.json for normalized errors)")
    ap.add_argument("--urdf", default=BOX_URDF)
    ap.add_argument("--prompt", default=ARENA_PROMPT)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res = run(args)
    short = {v: {k: round(s["mse"], 5) for k, s in d.items()} for v, d in res["variants"].items()}
    print(json.dumps({"mse": short, "clamp": {m: c["clamped_frac"] for m, c in res["clamp"].items()},
                      "latency_ms": res["latency_ms_during_eval"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
