"""Print a generated house summary (no Isaac): spawn, rooms, go_to points, doors, grid.

    python -m scenes.info procthor-train-38 [--json]
"""

from __future__ import annotations

import argparse
import json
import math

from scenes.catalog import parse_house_id
from scenes.occupancy import occupancy_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("house")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ref = parse_house_id(a.house)
    info = json.loads((ref.assets_dir / "house_info.json").read_text())
    occ = occupancy_summary(ref.house_id)
    if a.json:
        print(json.dumps({"scene_info_keys": list(info), "spawn": info["spawn"], "room_points": info["room_points"], "occupancy": occ}, indent=2))
        return 0
    sp = info["spawn"]
    print(f"{ref.house_id}  usd={info['usd_path']}")
    print(f"  spawn: x={sp['x']:.2f} y={sp['y']:.2f} yaw={math.degrees(sp['yaw']):.0f}deg room={sp.get('room')} "
          f"clearance={sp['clearance_m']:.2f} m forward_free={sp['forward_free_m']:.2f} m ({sp['source']})")
    print("  rooms:")
    for r in info["rooms"]:
        c = info["connectivity"]["rooms"].get(r["name"], {})
        print(f"    {r['name']:<14} {r['type']:<11} {r['area_m2']:6.1f} m2  reachable {c.get('reachable_area_m2', 0):5.1f} m2")
    print("  go_to points (max clearance per room):")
    for p in info["room_points"]:
        if p.get("ok"):
            print(f"    {p['room']:<14} x={p['x']:.2f} y={p['y']:.2f} clearance={p['clearance_m']:.2f} m")
    print(f"  objects: {len(info['objects'])}  grid: {occ['shape']} @ {occ['resolution']} m, origin {occ['origin']}, "
          f"r={occ['robot_radius']} m, method={occ['method']}  npz={occ['npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
