#!/usr/bin/env python3
"""Sample the CPU use of processes matching command-line patterns from /proc (stdlib only, ~0.1% CPU itself).

    python3 nav2/tools/cpu_monitor.py --out cpu.csv --period 1.0 \
        --group nav2=component_container,controller_server,planner_server,bt_navigator,behavior_server,velocity_smoother,lifecycle_manager \
        --group bridge=ros_bridge.py --group launch="ros2 launch" --group body=body.service --group fakes=fake_stack

One row per sample: t_wall, group, pids, cpu_pct (100 = one core), rss_mb. Processes are (re)discovered every sample,
so groups can start later. Stop with SIGINT/SIGTERM; nav2/tools/cpu_summary.py splits the csv by test phase."""
import argparse
import os
import signal
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def ticks_rss(pid: int):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return int(parts[11]) + int(parts[12]), int(parts[21]) * PAGE
    except (OSError, IndexError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--group", action="append", default=[], help="name=pattern[,pattern...]")
    ap.add_argument("--per-process", action="store_true", help="also one row per process (group/<exe name>)")
    a = ap.parse_args()
    groups = []
    for g in a.group:
        name, _, pats = g.partition("=")
        groups.append((name, [p for p in pats.split(",") if p]))
    me = os.getpid()
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    prev: dict[int, tuple[float, int]] = {}
    with open(a.out, "w", buffering=1) as f:
        f.write("t_wall,group,pids,cpu_pct,rss_mb\n")
        while not stop[0]:
            t = time.time()
            mono = time.monotonic()
            per = {name: [0.0, [], 0] for name, _ in groups}
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                pid = int(d)
                if pid == me:
                    continue
                cl = cmdline(pid)
                if not cl:
                    continue
                for name, pats in groups:
                    if any(p in cl for p in pats) and "cpu_monitor.py" not in cl and "tmux" not in cl \
                            and not cl.startswith("bash") and "tee " not in cl:
                        tr = ticks_rss(pid)
                        if tr is None:
                            break
                        ticks, rss = tr
                        if pid in prev:
                            m0, t0 = prev[pid]
                            c = (ticks - t0) / CLK / max(1e-6, mono - m0) * 100.0
                            per[name][0] += c
                            if a.per_process:
                                exe = os.path.basename(cl.split()[0])
                                if exe.startswith("python"):
                                    exe = os.path.basename(cl.split()[1]) if len(cl.split()) > 1 else exe
                                f.write(f"{t:.3f},{name}/{exe},{pid},{c:.2f},{rss / 1e6:.1f}\n")
                        prev[pid] = (mono, ticks)
                        per[name][1].append(pid)
                        per[name][2] += rss
                        break
            for name, (cpu, pids, rss) in per.items():
                if pids:
                    f.write(f"{t:.3f},{name},{' '.join(map(str, pids))},{cpu:.2f},{rss / 1e6:.1f}\n")
            time.sleep(max(0.0, a.period - (time.monotonic() - mono)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
