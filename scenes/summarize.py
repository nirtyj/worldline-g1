"""Collect outputs/m1/house/*/metrics.json into summary.json + summary.md (no Isaac).

    python -m scenes.summarize [/work/worldline-g1/outputs/m1/house]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def row(name: str, m: dict) -> dict:
    r, c = m.get("rtf_physics_only", {}), m.get("rtf_physics_cam30", {})
    ph = m.get("loader_stats", {}).get("physics", {})
    return {
        "run": name,
        "house": m.get("house_id"),
        "physx_threads": m.get("physx_threads"),
        "locked_joints": ph.get("locked_joints"),
        "kinematic_props": ph.get("kinematic_props"),
        "slept": bool(m.get("sleep_house")),
        "awake_after_settle": len(m.get("physx_awake_after_settle", [])) if "physx_awake_after_settle" in m else None,
        "rtf_physics": r.get("rtf"),
        "step_ms_p50": r.get("step_ms_p50"),
        "step_ms_p95": r.get("step_ms_p95"),
        "rtf_cam30": c.get("rtf"),
        "render_ms_p50": c.get("render_ms_p50"),
        "others_cpu_pct": (r.get("cpu") or {}).get("others_cpu_pct_of_box"),
        "load_to_physics_ready_s": m.get("times_s", {}).get("load_total_to_physics_ready"),
        "load_house_s": m.get("times_s", {}).get("load_house"),
        "gpu_mem_mib": m.get("gpu_mem_mib"),
        "rooms": m.get("counts", {}).get("rooms"),
        "objects": m.get("counts", {}).get("objects"),
        "all_rooms_reachable": m.get("connectivity", {}).get("all_rooms_reachable"),
        "occ_method": m.get("occupancy", {}).get("method"),
        "omap_iou": (m.get("occupancy", {}).get("omap_vs_overlap") or {}).get("iou_obstacles_inside"),
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/worldline-g1/outputs/m1/house")
    rows = []
    for d in sorted(root.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            rows.append(row(d.name, json.loads(f.read_text())))
    (root / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = ["run", "physx_threads", "slept", "awake_after_settle", "rtf_physics", "step_ms_p50", "rtf_cam30", "render_ms_p50", "others_cpu_pct", "load_to_physics_ready_s", "all_rooms_reachable", "occ_method", "omap_iou"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in cols) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
