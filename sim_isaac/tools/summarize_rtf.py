"""Summarise measure_rtf.sh outputs into rtf_summary.json + rtf_summary.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(out: str) -> None:
    d = Path(out)
    rows = []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("rtf_summary"):
            continue
        s = json.loads(f.read_text())
        env = (d / f"{f.stem}.env.txt").read_text() if (d / f"{f.stem}.env.txt").exists() else ""
        others = [ln for ln in env.splitlines() if "," in ln and "MiB" in ln]
        rows.append({
            "config": f.stem, "house": s.get("house"), "physx": s.get("physx_device"), "rt_pace": s.get("rt_pace"),
            "camera": s.get("camera"), "camera_hz": s.get("camera_hz_target"),
            "rtf_total": s.get("rtf_total"), "rtf_10s": s.get("rtf_10s"), "rtf_1s_min": s.get("rtf_1s_min"),
            "rtf_1s_below_0p95_frac": s.get("rtf_1s_below_0p95_frac"),
            "physics_hz_10s": s.get("physics_hz_10s"), "step_ms_mean": (s.get("step_ms") or {}).get("mean"),
            "step_ms_p99": (s.get("step_ms") or {}).get("p99"), "render_ms_mean": (s.get("render_ms") or {}).get("mean"),
            "render_ms_p99": (s.get("render_ms") or {}).get("p99"), "camera_pub_hz": s.get("camera_pub_hz"),
            "overruns": s.get("overruns"), "lost_s": s.get("lost_s"), "heartbeat_pubs": s.get("heartbeat_pubs"),
            "lowcmd_fresh_hz": s.get("lowcmd_fresh_hz"), "gpu_mem_mib": s.get("gpu_mem_mib"),
            "process_cpu_pct": s.get("process_cpu_pct"), "breakdown_ms": s.get("step_breakdown_ms"),
            "concurrent_gpu_apps": others,
        })
    (d / "rtf_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    cols = ["config", "rtf_total", "rtf_10s", "rtf_1s_min", "rtf_1s_below_0p95_frac", "physics_hz_10s", "step_ms_mean",
            "step_ms_p99", "render_ms_mean", "render_ms_p99", "camera_pub_hz", "overruns", "gpu_mem_mib",
            "process_cpu_pct"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    (d / "rtf_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/work/worldline-g1/outputs/m1/isaac/rtf")
