"""Is `request_image="area256"` the same model input as the full 640x480 frame? (docs/groot_serving.md §9)

Runs on the DEV box in the PolicyServer's own env (Isaac-GR00T @ 4b1dca9d, `/work/arena/gr00t_n17/.venv`, py3.10),
because it applies the checkpoint's own eval image transform: `build_image_transformations_albumentations` with the
values of `$CKPT/processor_config.json`, through `apply_with_replay` as `Gr00tN1d7Processor` does for inference
(`processing_gr00t_n1d7.py:404-412`). For each test frame it compares

    transform(full 640x480 frame)   vs   transform(groot.obs.area_downscale_2p5(frame))      (the model's input)
    groot.obs.area_downscale_2p5    vs   cv2.resize(frame, (256, 192), interpolation=INTER_AREA)

and writes the max abs difference and the number of differing values. No GPU, no server: CPU only.

    cd /work/worldline-g1 && /work/arena/gr00t_n17/.venv/bin/python -m tools.groot_request_check \\
        --groot-dir /work/arena/gr00t_n17 --ckpt <CKPT> --npz /work/groot/eval/obs_ep0_f60.npz --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np


def frames(npz: list[str], pngs: list[str], n_random: int) -> dict:
    out = {}
    for p in npz:
        z = np.load(p, allow_pickle=False)
        out[Path(p).name] = np.asarray(z["ego"], dtype=np.uint8)
    for pat in pngs:
        for p in sorted(glob.glob(pat))[:6]:
            from PIL import Image
            a = np.asarray(Image.open(p).convert("RGB"))
            if a.shape[:2] == (480, 640):
                out[Path(p).name] = a
    rng = np.random.default_rng(0)
    for i in range(n_random):
        out[f"random_{i}"] = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:480, 0:640]
    out["gradient"] = np.stack([(xx * 255 // 639), (yy * 255 // 479), ((xx + yy) % 256)], -1).astype(np.uint8)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groot-dir", default="/work/arena/gr00t_n17")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", action="append", default=[], help="an obs npz with an `ego` frame (groot.bench)")
    ap.add_argument("--png", action="append", default=[], help="glob of 640x480 frames (e.g. G2 frames)")
    ap.add_argument("--random", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.groot_dir)
    import cv2
    from PIL import Image

    from gr00t.model.gr00t_n1d7.image_augmentations import (apply_with_replay,
                                                            build_image_transformations_albumentations)
    from groot.obs import area_downscale_2p5

    pc = json.loads((Path(a.ckpt) / "processor_config.json").read_text())
    kw = pc.get("processor_kwargs", pc)
    if not kw.get("use_albumentations", True):
        raise SystemExit("this checkpoint's processor does not use the albumentations transforms")
    _, ev = build_image_transformations_albumentations(
        kw.get("image_target_size"), kw.get("image_crop_size"), kw.get("random_rotation_angle"),
        kw.get("color_jitter_params"), kw.get("shortest_image_edge"), kw.get("crop_fraction"))

    def model_input(img: np.ndarray) -> np.ndarray:
        t, _ = apply_with_replay(ev, [Image.fromarray(img)])
        return t[0].numpy()

    rows = []
    for name, f in frames(a.npz, a.png, a.random).items():
        t0 = time.perf_counter()
        small = area_downscale_2p5(f)
        t_ds = (time.perf_counter() - t0) * 1000.0
        ref = cv2.resize(f, (256, 192), interpolation=cv2.INTER_AREA)
        m_full, m_small = model_input(f), model_input(small)
        d_in = np.abs(m_full.astype(np.int16) - m_small.astype(np.int16))
        d_cv = np.abs(small.astype(np.int16) - ref.astype(np.int16))
        rows.append({"frame": name, "model_input_shape": list(m_full.shape),
                     "model_input_max_abs_diff": int(d_in.max()), "model_input_n_diff": int((d_in > 0).sum()),
                     "vs_cv2_inter_area_max_abs_diff": int(d_cv.max()), "vs_cv2_n_diff": int((d_cv > 0).sum()),
                     "downscale_ms": round(t_ds, 2)})
        print(json.dumps(rows[-1]), flush=True)
    res = {"tool": "groot_request_check", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "processor": {k: kw.get(k) for k in ("use_albumentations", "shortest_image_edge", "crop_fraction",
                                                "letter_box_transform", "image_target_size", "image_crop_size")},
           "cv2": cv2.__version__, "frames": rows,
           "identical": all(r["model_input_max_abs_diff"] == 0 for r in rows),
           "identical_to_cv2": all(r["vs_cv2_inter_area_max_abs_diff"] == 0 for r in rows)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "request_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("REQUEST_CHECK identical=%s identical_to_cv2=%s" % (res["identical"], res["identical_to_cv2"]))
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
