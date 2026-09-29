"""The --hook commands eval/stack_suite.py runs against the live stack (docs/eval_hooks.md). TEST-ONLY.

    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8766/ws --profile full --hooks box      # on the box
    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8766/ws --profile full --hooks laptop   # via ssh.sh

Each hook is one shell command (stack_suite's HookInjector formats the {placeholders}: push_robot {newtons},
throttle_rtf {rtf}, delay_proxy_on {ms}). Every main-box command is `python -m tools.hooks ...` (tools/hooks/cli.py),
which prints one JSON line; the P1 ones need P1 started with --test-ops (scripts/m2_up.sh --isaac-args "--test-ops").

    fault injections (PLAN 9.3)                 fixtures (restore the stack; never a source of verdicts)
    kill_policy     G5, G13   P4 unreachable           restore_policy   P4 answers again (see below)
    spawn_box       G6        box across the route     clear_box        the box parked again
    push_robot      G7        250 N lateral, 0.5 s     recover_robot    path A (body recover), else reset + stand
    kill_deploy     G8a       SIGKILL of the deploy    restore_deploy   tools.hooks recover: path B when it is dead
    throttle_rtf    G9        P1 rtf_throttle          restore_rtf      rtf_throttle off
    delay_proxy_on  G4        proxy reply delay        delay_proxy_off  0 ms
                                                       reset_fixup      a page reset that left the robot down: home
                                                                        (P1 reset_scene + stand), then a page reload
                                                                        that keeps the robot (stack_suite._after_load)
                                                       clear_all        at the end of every run: nothing left active

Where P4 (the GR00T PolicyServer) is, PLAN 0.12: on the DEV box, reached from the main box through the OD3 link
(scripts/groot_link.sh, main 127.0.0.1:5550 -> dev 127.0.0.1:5550). The main box's link key is port-forward only, so
a suite running ON the main box cannot stop the dev-box server:
  --hooks box     policy="link": kill_policy cuts the link (`tools.hooks policy cut`: groot_link.sh down, then a ping
                  must fail); restore_policy brings it back (groot_link.sh ensure, then a ping must answer). What the
                  runtime sees is the same as a dead server (no reply on 127.0.0.1:5550); the result says which it was.
  --hooks laptop  kill_policy stops the server process on the dev box (groot_server.sh stop, through ssh.sh with
                  BREV_NAME=ludo-g1-arena); restore_policy starts it again (--warm) and then checks the link from the
                  main box (`tools.hooks policy restore`).
policy="local" is PLAN 0.11's layout (the server on the main box): groot_server.sh stop / start --warm there.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

WL_BOX = "/work/worldline-g1"
# the laptop's Brev scripts (ludo_robotics_prep_g1/00_infra, next to this repo); WL_INFRA overrides
INFRA = os.environ.get("WL_INFRA", str(Path(__file__).resolve().parents[2] / "ludo_robotics_prep_g1" / "00_infra"))
MAIN_BOX, DEV_BOX = "ludo-g1-brev2", "ludo-g1-arena"
FAULTS = ("kill_policy", "spawn_box", "push_robot", "kill_deploy", "throttle_rtf", "delay_proxy_on")
FIXTURES = ("restore_policy", "clear_box", "recover_robot", "restore_deploy", "restore_rtf", "delay_proxy_off",
            "reset_fixup", "clear_all")
POLICY_MODES = ("link", "local", "none")


def box_hooks(wl: str = WL_BOX, py: str = ".venv-rt/bin/python", session: str = "wl-m2", policy: str = "link",
              policy_port: int = 5550, toward: str = "alarm_clock_1", push_s: float = 0.5,
              throttle_s: float = 300.0) -> dict[str, str]:
    """The hook map as run ON the main box (the suite itself on the box, the page at 127.0.0.1). policy: link (P4
    behind the OD3 link, PLAN 0.12), local (P4 on this box, PLAN 0.11) or none (G5/G13 SKIPPED live)."""
    if policy not in POLICY_MODES:
        raise ValueError(f"policy must be one of {POLICY_MODES}, got {policy!r}")
    h = f"cd {wl} && {py} -m tools.hooks --session {session}"
    out = {
        "spawn_box": f"{h} spawn-box --toward {toward}",
        "clear_box": f"{h} clear-box",
        "push_robot": f"{h} push --force-n {{newtons}} --dir left --duration-s {push_s}",
        "recover_robot": f"{h} recover",
        "reset_fixup": f"{h} home",
        "kill_deploy": f"{h} kill-deploy",
        "restore_deploy": f"{h} recover",
        "throttle_rtf": f"{h} throttle --rtf {{rtf}} --duration-s {throttle_s}",
        "restore_rtf": f"{h} unthrottle",
        "delay_proxy_on": f"{h} delay-proxy set --ms {{ms}}",
        "delay_proxy_off": f"{h} delay-proxy set --ms 0",
        "clear_all": f"{h} clear-all",
    }
    if policy in ("link", "local"):
        out["kill_policy"] = f"{h} policy cut --via {policy} --port {policy_port}"
        out["restore_policy"] = f"{h} policy restore --via {policy} --port {policy_port}"
    return out


def via_ssh(hooks: dict[str, str], brev_name: str = MAIN_BOX, infra: str = INFRA) -> dict[str, str]:
    """The same commands from the laptop, each through 00_infra/ssh.sh (the {placeholders} survive the quoting)."""
    return {k: ssh_cmd(v, brev_name, infra) for k, v in hooks.items()}


def ssh_cmd(cmd: str, brev_name: str, infra: str = INFRA) -> str:
    return f"BREV_NAME={shlex.quote(brev_name)} {shlex.quote(infra + '/ssh.sh')} {shlex.quote(cmd)}"


def laptop_hooks(infra: str = INFRA, wl: str = WL_BOX, policy_port: int = 5550, **kw) -> dict[str, str]:
    """The suite on the laptop: the main-box hooks through ssh.sh, and P4 stopped / started on the DEV box itself
    (PLAN 0.12), then the link checked from the main box."""
    main = box_hooks(wl=wl, policy="none", **kw)
    out = via_ssh(main, MAIN_BOX, infra)
    dev = f"cd {wl} && bash scripts/groot_server.sh"
    session = kw.get("session", "wl-m2")
    check = (f"cd {wl} && .venv-rt/bin/python -m tools.hooks --session {session} policy restore --via link "
             f"--port {policy_port}")
    out["kill_policy"] = ssh_cmd(f"{dev} stop --port {policy_port}", DEV_BOX, infra)
    out["restore_policy"] = (ssh_cmd(f"{dev} start --port {policy_port} --warm", DEV_BOX, infra) + " && "
                             + ssh_cmd(check, MAIN_BOX, infra))
    return out


def preset(name: str, **kw) -> dict[str, str]:
    if name == "box":
        return box_hooks(**kw)
    if name == "laptop":
        kw.pop("policy", None)
        return laptop_hooks(**kw)
    if name in ("none", ""):
        return {}
    raise ValueError(f"unknown hooks preset {name!r} (box | laptop | none)")
