#!/usr/bin/env python3
"""Summarise nav2/tools/cpu_monitor.py output per group and per test phase (stdlib only).

    python3 nav2/tools/cpu_summary.py --cpu cpu.csv --phases phases.json --out cpu_summary.json
Phases come from nav2_fake_test.py (goal0.., unreachable, cancel, watchdog, stand); "navigating" = all goal phases,
"idle" = samples outside every phase. cpu_pct: 100 = one core."""
import argparse
import csv
import json
import statistics


def stats(v):
    if not v:
        return None
    v = sorted(v)
    return {"n": len(v), "mean": round(statistics.fmean(v), 2), "p50": round(v[len(v) // 2], 2),
            "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 2), "max": round(v[-1], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.cpu)))
    phases = json.load(open(a.phases))
    groups = sorted({r["group"] for r in rows})
    out = {"note": "cpu_pct: 100 = one core; per-process CPU from /proc/<pid>/stat utime+stime deltas, 1 s samples",
           "groups": {}}
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        by = {"all": [], "idle": [], "navigating": []}
        for r in rs:
            t, c = float(r["t_wall"]), float(r["cpu_pct"])
            by["all"].append(c)
            ph = [p for p in phases if p["t0"] <= t <= p["t1"]]
            if not ph:
                by["idle"].append(c)
            for p in ph:
                key = "navigating" if p["name"].startswith("goal") else p["name"]
                by.setdefault(key, []).append(c)
        out["groups"][g] = {k: stats(v) for k, v in by.items()}
        out["groups"][g]["rss_mb_max"] = max(float(r["rss_mb"]) for r in rs)
    nav = [g for g in ("nav2", "bridge", "launch") if g in out["groups"]]
    tot = {}
    for key in ("navigating", "idle"):
        s = 0.0
        for g in nav:
            st = out["groups"][g].get(key)
            s += st["mean"] if st else 0.0
        tot[key] = round(s, 2)
    out["ros_side_total_mean_pct"] = tot
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({g: {k: (v or {}).get("mean") for k, v in d.items() if isinstance(v, dict) or v is None}
                      for g, d in out["groups"].items()}, indent=1))
    print("ROS side total mean %:", tot)


if __name__ == "__main__":
    main()
