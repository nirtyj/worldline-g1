"""Live smoke of every fault-injection hook on the running stack (docs/eval_hooks.md §2). TEST-ONLY; hold the stack lock.

    .venv-rt/bin/python -m tools.hooks.live_smoke --out outputs/m2b_finish/hooks/smoke-<ts> [--only push,throttle,...]

Steps (each one's hook replies, P1 health samples and body status go to <out>/smoke.json):
  preflight   nothing active, the body standing in control
  nudge       a small push (60 N, 0.2 s) standing: balance holds (informational: SONIC's own push recovery)
  push        PLAN G7's push (250 N lateral, 0.5 s) standing: a fall is detected (P1 fallen, body fault `fallen`),
              then `recover` (the body's path A, operator-triggered) stands it again
  throttle    RTF 0.9 for 25 s standing: P1 sim.health rtf/level sampled at 1 Hz, the robot stays up; then off
  box         spawn-box across the route to --toward; a body go_to along that route must end failed (stuck /
              off_path) with >= 1 replan and no fall; clear-box; go_to back to the start
  deploy      kill-deploy: the body's fault `deploy_lost` and the band (no fall); then `recover` (operator path B:
              run_deploy.sh start + stand) with its timing
  proxy       the delay proxy between a PolicyServer ping and 5550: RTT at 0 ms and at 600 ms (needs P4 up)
  final       `status`: hooks_active false, the body standing in control
Every step runs even when an earlier one failed (after a `recover`), and the last step always runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = ("preflight", "nudge", "push", "throttle", "box", "deploy", "proxy", "final")


def hook(*args: str, timeout: float = 400.0) -> dict[str, Any]:
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-m", "tools.hooks", *args], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    try:
        out = json.loads(lines[-1]) if lines else {"ok": False, "error": (p.stdout + p.stderr)[-400:]}
    except ValueError:
        out = {"ok": False, "error": lines[-1][:400]}
    out["_rc"], out["_s"] = p.returncode, round(time.monotonic() - t0, 2)
    print(f"  hooks {' '.join(args)} -> rc {p.returncode} in {out['_s']} s", flush=True)
    return out


class Probe:
    def __init__(self, offset: int) -> None:
        from body.client import BodyClient
        from body.config import ep, ports
        from body.p1_client import P1Rpc
        self.p1 = P1Rpc(ep(ports(offset)["p1_rep"]), timeout_s=5.0)
        self.bc = BodyClient(port_offset=offset).connect(10)

    def body(self) -> dict[str, Any]:
        s = self.bc.status()
        return {k: s.get(k) for k in ("mode", "fault", "recoveries", "in_control", "latched")} | \
            {"pelvis_z": (s.get("pose") or {}).get("pelvis_z") if isinstance(s.get("pose"), dict) else None}

    def pose(self) -> dict[str, Any]:
        p = self.p1.call("get_pose")
        return {"x": round(p["base_pos"][0], 3), "y": round(p["base_pos"][1], 3), "yaw": round(p.get("yaw", 0.0), 3),
                "pelvis_z": round(p.get("pelvis_z", 0.0), 3), "fallen": p.get("fallen")}

    def health(self) -> dict[str, Any]:
        h = self.p1.call("get_health")
        return {k: h.get(k) for k in ("rtf_1s", "rtf_3s", "rtf_5s", "level", "overruns", "fallen", "band")}

    def watch(self, seconds: float, every: float = 0.1) -> dict[str, Any]:
        """Poll the pose: the lowest pelvis, whether (and when) it fell."""
        t0, zmin, t_fall = time.monotonic(), 9.0, None
        while time.monotonic() - t0 < seconds:
            p = self.pose()
            zmin = min(zmin, p["pelvis_z"])
            if p["fallen"] and t_fall is None:
                t_fall = round(time.monotonic() - t0, 2)
            time.sleep(every)
        return {"fell": t_fall is not None, "t_fall_s": t_fall, "pelvis_z_min": round(zmin, 3)}

    def wait_body(self, pred, timeout: float) -> tuple[bool, float, dict]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            b = self.body()
            if pred(b):
                return True, round(time.monotonic() - t0, 2), b
            time.sleep(0.1)
        return False, round(time.monotonic() - t0, 2), self.body()

    def standing(self) -> bool:
        b, p = self.body(), self.pose()
        return bool(b.get("in_control")) and not b.get("fault") and not p["fallen"] and p["pelvis_z"] > 0.6


def run(steps: list[str], a: argparse.Namespace) -> dict[str, Any]:
    pr = Probe(a.port_offset)
    res: dict[str, Any] = {"t_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}}

    def rec(name: str, ok: Any, **kw) -> None:
        res["steps"][name] = {"ok": ok, **kw}
        print(f"[{name}] {'OK' if ok else 'FAIL' if ok is False else '--'} {json.dumps(kw, default=str)[:600]}",
              flush=True)

    def ensure_standing(why: str) -> dict | None:
        if pr.standing():
            return None
        r = hook("recover")
        print(f"  (recover before {why}: {r.get('path')} ok={r.get('ok')})", flush=True)
        return r

    for step in steps:
        try:
            if step == "preflight":
                st = hook("status")
                rec(step, pr.standing() and not st.get("hooks_active") and st["p1"].get("ok", True) is not False,
                    status=st, pose=pr.pose(), health=pr.health())
            elif step == "nudge":
                pre = ensure_standing(step)
                r = hook("push", "--force-n", str(a.nudge_n), "--duration-s", "0.2", "--dir", "left",
                         "--watch-s", "4")
                rec(step, r.get("ok") and not r.get("fell"), push=r, body=pr.body(), pre_recover=pre,
                    label="informational: balance after a small push")
                if r.get("fell"):
                    res["steps"][step]["recover"] = hook("recover")
            elif step == "push":
                pre = ensure_standing(step)
                p0 = pr.pose()
                r = hook("push", "--force-n", "250", "--duration-s", "0.5", "--dir", "left", "--watch-s", "5")
                ok_f, t_f, b_f = pr.wait_body(lambda b: b.get("fault") == "fallen", 5.0)
                rv = hook("recover", "--timeout", "90")
                ok_up = pr.standing()
                rec(step, bool(r.get("fell")) and ok_f and rv.get("ok") and ok_up, push=r, pose_before=p0,
                    body_fault=b_f, fault_after_s=t_f, recover=rv, standing_after=ok_up, pose_after=pr.pose(),
                    pre_recover=pre)
            elif step == "throttle":
                pre = ensure_standing(step)
                h0 = pr.health()
                r = hook("throttle", "--rtf", "0.9", "--duration-s", str(a.throttle_s + 30))
                samples = []
                t0 = time.monotonic()
                while time.monotonic() - t0 < a.throttle_s:
                    samples.append({"t": round(time.monotonic() - t0, 1), **pr.health()})
                    time.sleep(1.0)
                up = hook("unthrottle")
                time.sleep(7.0)
                h1 = pr.health()
                late = [s for s in samples if s["t"] >= 8.0]
                r5 = [s["rtf_5s"] for s in late if s.get("rtf_5s") is not None]
                ok = (r.get("ok") and up.get("ok") and bool(r5) and all(0.85 <= v <= 0.93 for v in r5)
                      and all(s["level"] == "degraded" for s in late) and h1.get("level") == "ok" and pr.standing())
                rec(step, ok, throttle=r, unthrottle=up, before=h0, samples=samples, after=h1,
                    rtf_5s_range=[min(r5), max(r5)] if r5 else None, standing_after=pr.standing(), pre_recover=pre)
            elif step == "box":
                pre = ensure_standing(step)
                p0 = pr.pose()
                sb = hook("spawn-box", "--toward", a.toward, "--ttl-s", "300")
                placed = sb.get("placed") or {}
                walk = None
                if sb.get("ok") and placed:
                    gx, gy = placed["path_end"]
                    h = pr.bc.go_to(gx, gy, timeout_s=90.0)
                    walk = {"state": h.state, "reason": h.reason,
                            "replans": (h.result or {}).get("replans"), "result": {
                                k: (h.result or {}).get(k) for k in ("reason", "replans", "walked_m", "final_err_m",
                                                                     "stuck_events", "pose")}}
                p_mid = pr.pose()
                cb = hook("clear-box")
                back = pr.bc.go_to(p0["x"], p0["y"], yaw=p0["yaw"], timeout_s=90.0)
                ok = (sb.get("ok") and walk is not None and walk["state"] == "failed"
                      and str(walk["reason"]) in ("stuck", "off_path") and not p_mid["fallen"] and cb.get("ok"))
                rec(step, ok, spawn=sb, walk=walk, pose_at_box=p_mid, clear=cb,
                    back={"state": back.state, "reason": back.reason}, standing_after=pr.standing(), pre_recover=pre)
            elif step == "deploy":
                pre = ensure_standing(step)
                k = hook("kill-deploy")
                ok_l, t_l, b_l = pr.wait_body(lambda b: b.get("fault") == "deploy_lost", 5.0)
                w = pr.watch(4.0)
                rv = hook("recover", "--wait-init", "240", "--timeout", "120", timeout=500)
                up = pr.standing()
                rec(step, k.get("ok") and ok_l and not w["fell"] and rv.get("ok") and up, kill=k,
                    fault_after_s=t_l, body_fault=b_l, while_down=w, recover=rv, standing_after=up,
                    body_after=pr.body(), pre_recover=pre)
            elif step == "proxy":
                def ping(ep: str) -> dict:
                    t0 = time.monotonic()
                    p = subprocess.run([sys.executable, "-m", "groot.policy_client", "ping", "--endpoint", ep,
                                        "--timeout", "3"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
                    return {"rc": p.returncode, "s": round(time.monotonic() - t0, 3),
                            "out": (p.stdout + p.stderr).strip().splitlines()[-1:]}
                direct = ping("tcp://127.0.0.1:5550")
                st = hook("delay-proxy", "start", "--ms", "0")
                p0 = ping("tcp://127.0.0.1:5551")
                s6 = hook("delay-proxy", "set", "--ms", "600")
                p6 = ping("tcp://127.0.0.1:5551")
                s0 = hook("delay-proxy", "set", "--ms", "0")
                p00 = ping("tcp://127.0.0.1:5551")
                stat = hook("delay-proxy", "status")
                sp = hook("delay-proxy", "stop")
                ok = (direct["rc"] == 0 and st.get("ok") and p0["rc"] == 0 and p6["rc"] == 0 and p00["rc"] == 0
                      and p6["s"] - p0["s"] >= 0.55 and sp.get("ok"))
                rec(step, ok, direct=direct, ping_0ms=p0, ping_600ms=p6, ping_0ms_again=p00, proxy=stat,
                    start=st, set_600=s6, set_0=s0, stop=sp, added_s=round(p6["s"] - p0["s"], 3))
            elif step == "final":
                pre = ensure_standing(step)
                st = hook("status")
                rec(step, pr.standing() and not st.get("hooks_active"), status=st, pose=pr.pose(),
                    health=pr.health(), pre_recover=pre)
        except Exception as e:  # noqa: BLE001  (a broken step is a failed step; the rest still run)
            rec(step, False, error=f"{type(e).__name__}: {e}")
    pr.bc.close()
    res["passed"] = sorted(k for k, v in res["steps"].items() if v["ok"])
    res["failed"] = sorted(k for k, v in res["steps"].items() if v["ok"] is False)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--port-offset", type=int, default=0)
    ap.add_argument("--toward", default="alarm_clock_1")
    ap.add_argument("--nudge-n", type=float, default=60.0)
    ap.add_argument("--throttle-s", type=float, default=25.0)
    a = ap.parse_args(argv)
    steps = [s for s in STEPS if not a.only or s in a.only.split(",") or s == "final"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(steps, a)
    (out / "smoke.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"passed {res['passed']} failed {res['failed']} -> {out / 'smoke.json'}", flush=True)
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
