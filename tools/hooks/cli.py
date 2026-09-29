"""Box-side fault injection and recovery for eval/stack_suite.py --hook (docs/eval_hooks.md). TEST-ONLY.

    python -m tools.hooks [--port-offset N] [--session wl-m2] CMD ...

  push [--force-n 250] [--dir left] [--duration-s 0.5] [--watch-s 0]    P1 push_robot (G7); --watch-s reports a fall
  throttle --rtf 0.9 [--duration-s 240]  |  unthrottle                  P1 rtf_throttle (G9)
  spawn-box [--toward OBJ | --toward-xy X,Y | --x X --y Y --yaw R] [--ttl-s 600]   P1 spawn_box (G6); --toward puts it
                                         across the robot's own A* route at its narrowest point (tools/hooks/obstacle)
  clear-box                                                              P1 clear_box
  kill-deploy                            SIGKILL the SONIC deploy of the stack session (G8a: a crash, never 'o' /
                                         command{stop}); the body's deploy_lost fault and the band catch it
  restart-deploy [--wait-init 180] [--no-stand]   the deploy again in the stack's tmux window (run_deploy.sh, as
                                         m1_up.sh starts it), then the body `stand`: an operator path B
  recover [--timeout 120]                back to standing: deploy dead -> restart-deploy (path B); fault fallen -> the
                                         body's `recover` (path A), else P1 reset_robot + clear_fault + stand
  delay-proxy start|set|stop|status [--ms N]      tools/hooks/delay_proxy.py in tmux 'hooks-delay' (G4)
  p5-endpoint proxy|direct               restart P5 (scripts/m2_p5.sh restart) with WL_GROOT_ENDPOINT on the proxy
                                         (tcp://127.0.0.1:5551) or back on the PolicyServer (unset)
  status                                 what is active: P1 test ops, body mode/fault, deploy pid, delay proxy
  clear-all                              unthrottle, clear-box, delay 0 (never touches the deploy)

Every command prints one JSON line last ({"ok": ..., ...}) and exits 0 iff ok. The P1 ops exist only when P1 runs
with --test-ops (scripts/m2_up.sh --isaac-args "--test-ops"); without it they fail `unknown_op`.
Never sends command{stop}; never starts a second deploy (restart-deploy refuses while one is alive).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from body.config import ep, port_offset_from_env, ports as _ports  # noqa: E402

LOG_ROOT = Path(os.environ.get("WL_LOG_ROOT", "/work/logs/wl"))
DEPLOY_BIN_NAME = "g1_deploy_onnx_ref"
PROXY_SESSION = "hooks-delay"
PROXY_PORTS = {"listen": 5551, "upstream": 5550, "ctl": 5549}


def say(msg: str) -> None:
    print(f"[hooks {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ P1 and body
class Stack:
    """Lazy clients of one stack (port offset + tmux session)."""

    def __init__(self, offset: int, session: str):
        self.offset, self.session = offset, session
        self.P = _ports(offset)
        self._p1 = self._body = None

    def p1(self, op: str, timeout_s: float = 10.0, **args) -> dict:
        from body.p1_client import P1Rpc
        if self._p1 is None:
            self._p1 = P1Rpc(ep(self.P["p1_rep"]), timeout_s=timeout_s)
        rep = self._p1.call(op, timeout_s=timeout_s, **args)
        return rep if isinstance(rep, dict) else {"ok": False, "error": f"bad reply {rep!r}"}

    def body(self):
        from body.client import BodyClient
        if self._body is None:
            self._body = BodyClient(port_offset=self.offset).connect(wait_s=10.0)
        return self._body

    def close(self) -> None:
        if self._body is not None:
            self._body.close()


def _ok(rep: dict) -> bool:
    return bool(rep) and rep.get("ok", True) is not False


def _pose(st: Stack) -> dict:
    return st.p1("get_pose")


# ------------------------------------------------------------------------------------------------ the deploy
def deploy_pid(session: str, debug_port: int) -> int | None:
    """The live deploy of THIS stack: a g1_deploy_onnx_ref whose command line binds this stack's g1_debug port
    (run_deploy.sh always passes --zmq-out-port), from run_deploy.sh's pid file or pgrep. A deploy of another
    stack (another port offset) is never returned, so kill-deploy cannot hit it."""
    pf = LOG_ROOT / f"{session}.pid"
    cands: list[int] = []
    try:
        cands.append(int(pf.read_text().strip()))
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["pgrep", "-f", DEPLOY_BIN_NAME], capture_output=True, text=True, timeout=5).stdout
        cands += [int(x) for x in out.split() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        pass
    for pid in dict.fromkeys(cands):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if DEPLOY_BIN_NAME in cmd and "pgrep" not in cmd and f"--zmq-out-port {debug_port} " in cmd + " ":
            return pid
    return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_deploy(st: Stack, a: argparse.Namespace) -> dict:
    pid = deploy_pid(st.session, st.P["sonic_debug"])
    if pid is None:
        return {"ok": False, "error": f"no {DEPLOY_BIN_NAME} running (session {st.session})"}
    t0 = time.time()
    os.kill(pid, signal.SIGKILL)                  # a crash: no 'o', no command{stop}
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.05)
    gone = not _alive(pid)
    say(f"deploy pid {pid} SIGKILLed ({'gone' if gone else 'STILL ALIVE'})")
    return {"ok": gone, "killed_pid": pid, "t_wall": round(t0, 3), "signal": "SIGKILL", "label": "test-only"}


def _deploy_log(session: str) -> str:
    try:
        return (LOG_ROOT / f"{session}.logpath").read_text().strip()
    except OSError:
        return str(LOG_ROOT / f"{session}-deploy-restart.log")


def restart_deploy(st: Stack, a: argparse.Namespace) -> dict:
    pid = deploy_pid(st.session, st.P["sonic_debug"])
    if pid is not None:
        return {"ok": False, "error": f"a deploy is alive (pid {pid}): one deploy per box; kill it first"}
    t0 = time.monotonic()
    # the dead deploy's window (its pane exec'd the binary, so it normally closed with it)
    subprocess.run(["tmux", "kill-window", "-t", f"={st.session}:=deploy"], capture_output=True, timeout=10)
    cmd = ["bash", str(ROOT / "sonic" / "run_deploy.sh"), "start", "--session", st.session, "--window", "deploy",
           "--log", _deploy_log(st.session), "--zmq-port", str(st.P["sonic_in"]),
           "--zmq-out-port", str(st.P["sonic_debug"]), "--wait-init", str(int(a.wait_init))]
    if a.taskset:
        cmd += ["--taskset", a.taskset]
    say("restarting the deploy: " + " ".join(cmd[2:]))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.wait_init + 60)
    t_init = time.monotonic() - t0
    out = {"ok": p.returncode == 0, "path": "B (operator: tools/hooks restart-deploy)", "t_init_s": round(t_init, 1),
           "run_deploy": (p.stdout + p.stderr).strip().splitlines()[-2:], "new_pid": deploy_pid(st.session, st.P["sonic_debug"])}
    if p.returncode != 0 or a.no_stand:
        out["total_s"] = round(time.monotonic() - t0, 1)
        return out
    out.update(_stand(st, a.timeout))
    out["ok"] = out["ok"] and out["stand"]["state"] == "succeeded"
    out["total_s"] = round(time.monotonic() - t0, 1)
    return out


def _stand(st: Stack, timeout: float) -> dict:
    t0 = time.monotonic()
    h = st.body().stand(wait=True, timeout=timeout)
    res = dict(getattr(h, "result", None) or {})
    return {"stand": {"state": getattr(h, "state", None), "reason": res.get("reason") or res.get("error"),
                      "band_released": res.get("band_released"), "s": round(time.monotonic() - t0, 1)}}


def recover(st: Stack, a: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    if deploy_pid(st.session, st.P["sonic_debug"]) is None:
        a.no_stand = False
        out = restart_deploy(st, a)
        out["why"] = "deploy not running"
        return out
    b = st.body()
    s0 = b.status()
    fault = s0.get("fault")
    steps: list[dict] = []
    out: dict[str, Any] = {"fault_before": fault, "mode_before": s0.get("mode"), "in_control": s0.get("in_control")}
    if fault == "fallen":
        h = b.recover(wait=True, timeout=a.timeout)
        res = dict(getattr(h, "result", None) or {})
        steps.append({"op": "recover", "state": getattr(h, "state", None), "reason": res.get("reason"),
                      "path": res.get("path"), "label": res.get("label")})
        if getattr(h, "state", None) == "succeeded":
            out.update(ok=True, path="A (body recover, operator-triggered)", steps=steps,
                       recoveries=b.status().get("recoveries"), total_s=round(time.monotonic() - t0, 1))
            return out
        p = _pose(st)
        bp = p.get("base_pos") or [None, None]
        rr = st.p1("reset_robot", x=bp[0], y=bp[1], yaw=p.get("yaw", 0.0), band=True)
        steps.append({"op": "p1.reset_robot", "ok": _ok(rr)})
        cf = b.request("clear_fault")
        steps.append({"op": "clear_fault", "ok": cf.get("ok"), "cleared": (cf.get("data") or {}).get("cleared")})
        out["path"] = "reset (P1 reset_robot + clear_fault + stand)"
    elif fault in (None, "deploy_lost") and s0.get("in_control") and s0.get("mode") not in ("ESTOP", "FAULT"):
        out.update(ok=True, path="none needed", steps=steps, total_s=round(time.monotonic() - t0, 1))
        return out
    else:
        out["path"] = f"stand ({fault or s0.get('mode')})"
    sd = _stand(st, a.timeout)
    steps.append({"op": "stand", **sd["stand"]})
    out.update(ok=sd["stand"]["state"] == "succeeded", steps=steps, total_s=round(time.monotonic() - t0, 1))
    return out


# ------------------------------------------------------------------------------------------------ P1 test ops
def push(st: Stack, a: argparse.Namespace) -> dict:
    d: Any = a.dir
    if "," in str(d):
        d = [float(v) for v in str(d).split(",")]
    rep = st.p1("push_robot", force_n=a.force_n, dir=d, duration_s=a.duration_s)
    out = {"ok": _ok(rep), "p1": rep}
    if not _ok(rep) or a.watch_s <= 0:
        return out
    t0 = time.monotonic()
    zmin, t_fall = 9.9, None
    while time.monotonic() - t0 < a.watch_s:
        p = _pose(st)
        zmin = min(zmin, float(p.get("pelvis_z") or 9.9))
        if p.get("fallen") and t_fall is None:
            t_fall = round(time.monotonic() - t0, 2)
        time.sleep(0.1)
    out.update(fell=t_fall is not None, t_fall_s=t_fall, pelvis_z_min=round(zmin, 3))
    return out


def throttle(st: Stack, a: argparse.Namespace) -> dict:
    rep = st.p1("rtf_throttle", target=a.rtf, duration_s=a.duration_s)
    return {"ok": _ok(rep), "p1": rep}


def unthrottle(st: Stack, a: argparse.Namespace) -> dict:
    rep = st.p1("rtf_throttle", off=True)
    return {"ok": _ok(rep), "p1": rep}


def spawn_box(st: Stack, a: argparse.Namespace) -> dict:
    placed = None
    if a.toward or a.toward_xy:
        from body.nav_grid import NavGrid
        from tools.hooks.obstacle import place_across_path
        status = st.p1("test_ops_status")
        if not _ok(status):
            return {"ok": False, "error": f"P1 has no test ops: {status.get('error')}"}
        if a.toward_xy:
            gx, gy = (float(v) for v in a.toward_xy.split(","))
        else:
            objs = st.p1("get_objects", ids=[a.toward]).get("objects") or []
            if not objs:
                return {"ok": False, "error": f"no object {a.toward!r}"}
            gx, gy = float(objs[0]["pos"][0]), float(objs[0]["pos"][1])
        bp = _pose(st).get("base_pos") or [0.0, 0.0]
        grid = NavGrid.from_p1_reply(st.p1("get_occupancy", timeout_s=30.0, robot_radius=0.25), robot_radius=0.25)
        placed = place_across_path(grid, (float(bp[0]), float(bp[1])), (gx, gy), tuple(status["box_size"]),
                                   min_from_start=a.min_from_robot)
        x, y, yaw = placed["x"], placed["y"], placed["yaw"]
    elif a.x is not None and a.y is not None:
        x, y, yaw = a.x, a.y, a.yaw
    else:
        return {"ok": False, "error": "spawn-box needs --toward OBJ, --toward-xy X,Y or --x/--y"}
    rep = st.p1("spawn_box", pose={"x": x, "y": y, "yaw": yaw}, ttl_s=a.ttl_s)
    return {"ok": _ok(rep), "placed": placed, "p1": rep}


def clear_box(st: Stack, a: argparse.Namespace) -> dict:
    rep = st.p1("clear_box")
    return {"ok": _ok(rep), "p1": rep}


# ------------------------------------------------------------------------------------------------ delay proxy
def _ctl_ep() -> str:
    return f"tcp://127.0.0.1:{PROXY_PORTS['ctl']}"


def delay_proxy(st: Stack, a: argparse.Namespace) -> dict:
    from tools.hooks.delay_proxy import ctl_call
    if a.action == "status":
        rep = ctl_call(_ctl_ep(), {"op": "get"})
        return {"ok": True, "running": _ok(rep), "proxy": rep}
    if a.action == "set":
        rep = ctl_call(_ctl_ep(), {"op": "set", "ms": a.ms})
        return {"ok": _ok(rep), "proxy": rep}
    if a.action == "stop":
        rep = ctl_call(_ctl_ep(), {"op": "stop"}, timeout_s=1.0)
        time.sleep(0.3)
        subprocess.run(["tmux", "kill-session", "-t", f"={PROXY_SESSION}"], capture_output=True, timeout=10)
        return {"ok": True, "stopped": _ok(rep), "last": rep}
    # start
    if _ok(ctl_call(_ctl_ep(), {"op": "get"}, timeout_s=0.5)):
        return {"ok": False, "error": "a delay proxy already answers on the control port"}
    cmd = (f"cd {ROOT} && exec {sys.executable} -u -m tools.hooks.delay_proxy --listen {PROXY_PORTS['listen']} "
           f"--upstream {PROXY_PORTS['upstream']} --ctl {PROXY_PORTS['ctl']} --ms {a.ms} "
           f"2>&1 | tee -a {LOG_ROOT}/hooks-delay-proxy.log")
    subprocess.run(["tmux", "new-session", "-d", "-s", PROXY_SESSION, "bash", "--noprofile", "--norc", "-c", cmd],
                   check=True, timeout=10)
    for _ in range(40):
        rep = ctl_call(_ctl_ep(), {"op": "get"}, timeout_s=0.25)
        if _ok(rep):
            return {"ok": True, "proxy": rep, "tmux": PROXY_SESSION}
        time.sleep(0.25)
    return {"ok": False, "error": "the delay proxy did not answer within 10 s", "tmux": PROXY_SESSION}


def p5_endpoint(st: Stack, a: argparse.Namespace) -> dict:
    """P5 reads WL_GROOT_ENDPOINT when it builds the groot_arms executor; m2_p5.sh opens its window in the stack's
    tmux session, which inherits the session environment set here."""
    if a.to == "proxy":
        env = ["tmux", "set-environment", "-t", f"={st.session}", "WL_GROOT_ENDPOINT",
               f"tcp://127.0.0.1:{PROXY_PORTS['listen']}"]
    else:
        env = ["tmux", "set-environment", "-t", f"={st.session}", "-u", "WL_GROOT_ENDPOINT"]
    subprocess.run(env, check=True, capture_output=True, timeout=10)
    t0 = time.monotonic()
    p = subprocess.run(["bash", str(ROOT / "scripts" / "m2_p5.sh"), "restart", "--session", st.session],
                       capture_output=True, text=True, timeout=420)
    return {"ok": p.returncode == 0, "endpoint": a.to, "s": round(time.monotonic() - t0, 1),
            "m2_p5": (p.stdout + p.stderr).strip().splitlines()[-1:]}


# ------------------------------------------------------------------------------------------------ status
def status(st: Stack, a: argparse.Namespace) -> dict:
    from tools.hooks.delay_proxy import ctl_call
    out: dict[str, Any] = {"ok": True}
    try:
        out["p1"] = st.p1("test_ops_status", timeout_s=3.0)
    except Exception as e:  # noqa: BLE001
        out["p1"] = {"ok": False, "error": repr(e)}
    try:
        s = st.body().status()
        out["body"] = {k: s.get(k) for k in ("mode", "fault", "recoveries", "in_control", "latched", "halt_epoch")}
    except Exception as e:  # noqa: BLE001
        out["body"] = {"error": repr(e)}
    pid = deploy_pid(st.session, st.P["sonic_debug"])
    out["deploy"] = {"pid": pid, "alive": pid is not None}
    px = ctl_call(_ctl_ep(), {"op": "get"}, timeout_s=0.5)
    out["delay_proxy"] = px if _ok(px) else None
    out["hooks_active"] = bool((out["p1"] or {}).get("active")) or bool(_ok(px) and px.get("ms"))
    return out


def clear_all(st: Stack, a: argparse.Namespace) -> dict:
    from tools.hooks.delay_proxy import ctl_call
    r1 = st.p1("rtf_throttle", off=True)
    r2 = st.p1("clear_box")
    px = ctl_call(_ctl_ep(), {"op": "get"}, timeout_s=0.5)
    r3 = ctl_call(_ctl_ep(), {"op": "set", "ms": 0}) if _ok(px) else None
    s = st.p1("test_ops_status")
    return {"ok": _ok(r1) and _ok(r2) and (r3 is None or _ok(r3)) and not s.get("active"),
            "unthrottle": _ok(r1), "clear_box": _ok(r2), "delay_proxy": None if r3 is None else r3.get("ms"),
            "active_after": s.get("active")}


# ------------------------------------------------------------------------------------------------ main
COMMANDS = {"push": push, "throttle": throttle, "unthrottle": unthrottle, "spawn-box": spawn_box,
            "clear-box": clear_box, "kill-deploy": kill_deploy, "restart-deploy": restart_deploy, "recover": recover,
            "delay-proxy": delay_proxy, "p5-endpoint": p5_endpoint, "status": status, "clear-all": clear_all}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.hooks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port-offset", type=int, default=None)
    ap.add_argument("--session", default="wl-m2", help="the stack's tmux session (m2_up.sh --session)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("push")
    s.add_argument("--force-n", type=float, default=250.0)
    s.add_argument("--dir", default="left", help="left|right|forward|back, degrees (world), or dx,dy (world)")
    s.add_argument("--duration-s", type=float, default=0.5)
    s.add_argument("--watch-s", type=float, default=0.0)
    s = sp.add_parser("throttle")
    s.add_argument("--rtf", type=float, default=0.9)
    s.add_argument("--duration-s", type=float, default=240.0)
    sp.add_parser("unthrottle")
    s = sp.add_parser("spawn-box")
    s.add_argument("--toward", default=None, help="object id: across the robot's A* route to it")
    s.add_argument("--toward-xy", default=None)
    s.add_argument("--x", type=float, default=None)
    s.add_argument("--y", type=float, default=None)
    s.add_argument("--yaw", type=float, default=0.0)
    s.add_argument("--min-from-robot", type=float, default=1.2)
    s.add_argument("--ttl-s", type=float, default=600.0)
    sp.add_parser("clear-box")
    sp.add_parser("kill-deploy")
    for name in ("restart-deploy", "recover"):
        s = sp.add_parser(name)
        s.add_argument("--wait-init", type=float, default=180.0)
        s.add_argument("--taskset", default=os.environ.get("DEPLOY_TASKSET", "0-3"))
        s.add_argument("--timeout", type=float, default=120.0)
        s.add_argument("--no-stand", action="store_true")
    s = sp.add_parser("delay-proxy")
    s.add_argument("action", choices=["start", "set", "stop", "status"])
    s.add_argument("--ms", type=float, default=0.0)
    s = sp.add_parser("p5-endpoint")
    s.add_argument("to", choices=["proxy", "direct"])
    sp.add_parser("status")
    sp.add_parser("clear-all")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    off = port_offset_from_env() if a.port_offset is None else a.port_offset
    st = Stack(off, a.session)
    t0 = time.time()
    try:
        out = COMMANDS[a.cmd](st, a)
    except Exception as e:  # noqa: BLE001  (one JSON line whatever happens)
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        st.close()
    out = {"cmd": a.cmd, **out, "t_wall": round(t0, 3)}
    print(json.dumps(out, default=str), flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
