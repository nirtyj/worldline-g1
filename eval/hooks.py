"""The --hook commands eval/stack_suite.py runs against the live stack (docs/eval_hooks.md). TEST-ONLY.

    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8766/ws --profile full --hooks box      # on the box
    .venv-rt/bin/python -m eval.stack_suite --url ws://127.0.0.1:8766/ws --profile full --hooks laptop   # via ssh.sh

Each hook is one shell command (stack_suite's HookInjector formats the {placeholders}: push_robot {newtons},
throttle_rtf {rtf}, delay_proxy_on {ms}). Every box-side command is `python -m tools.hooks ...` (tools/hooks/cli.py),
which prints one JSON line; the P1 ones need P1 started with --test-ops (scripts/m2_up.sh --isaac-args "--test-ops").

    fault injections (PLAN 9.3)                 fixtures (restore the stack; never a source of verdicts)
    kill_policy     G5, G13   groot_server.sh stop     restore_policy   groot_server.sh start --warm
    spawn_box       G6        box across the route     clear_box        the box parked again
    push_robot      G7        250 N lateral, 0.5 s     recover_robot    path A (body recover), else reset + stand
    kill_deploy     G8a       SIGKILL of the deploy    restore_deploy   tools.hooks recover: path B when it is dead
    throttle_rtf    G9        P1 rtf_throttle          restore_rtf      rtf_throttle off
    delay_proxy_on  G4        proxy reply delay        delay_proxy_off  0 ms
                                                       clear_all        at the end of every run: nothing left active

The GR00T PolicyServer runs on the main box (PLAN 0.11), so kill_policy/restore_policy are main-box commands.
"""
from __future__ import annotations

import os
import shlex

WL_BOX = "/work/worldline-g1"
INFRA = os.environ.get("WL_INFRA", "/Users/nirty/workspace/ludo-interview/ludo_robotics_prep_g1/00_infra")
FAULTS = ("kill_policy", "spawn_box", "push_robot", "kill_deploy", "throttle_rtf", "delay_proxy_on")
FIXTURES = ("restore_policy", "clear_box", "recover_robot", "restore_deploy", "restore_rtf", "delay_proxy_off",
            "clear_all")


def box_hooks(wl: str = WL_BOX, py: str = ".venv-rt/bin/python", session: str = "wl-m2",
              policy_port: int = 5550, toward: str = "alarm_clock_1", push_s: float = 0.5,
              throttle_s: float = 300.0) -> dict[str, str]:
    """The hook map as run ON the box (the suite itself on the box, the page at 127.0.0.1)."""
    h = f"cd {wl} && {py} -m tools.hooks --session {session}"
    return {
        "kill_policy": f"cd {wl} && bash scripts/groot_server.sh stop --port {policy_port}",
        "restore_policy": f"cd {wl} && bash scripts/groot_server.sh start --port {policy_port} --warm",
        "spawn_box": f"{h} spawn-box --toward {toward}",
        "clear_box": f"{h} clear-box",
        "push_robot": f"{h} push --force-n {{newtons}} --dir left --duration-s {push_s}",
        "recover_robot": f"{h} recover",
        "kill_deploy": f"{h} kill-deploy",
        "restore_deploy": f"{h} recover",
        "throttle_rtf": f"{h} throttle --rtf {{rtf}} --duration-s {throttle_s}",
        "restore_rtf": f"{h} unthrottle",
        "delay_proxy_on": f"{h} delay-proxy set --ms {{ms}}",
        "delay_proxy_off": f"{h} delay-proxy set --ms 0",
        "clear_all": f"{h} clear-all",
    }


def via_ssh(hooks: dict[str, str], brev_name: str = "ludo-g1-brev2", infra: str = INFRA) -> dict[str, str]:
    """The same commands from the laptop, each through 00_infra/ssh.sh (the {placeholders} survive the quoting)."""
    pre = f"BREV_NAME={shlex.quote(brev_name)} {shlex.quote(infra + '/ssh.sh')}"
    return {k: f"{pre} {shlex.quote(v)}" for k, v in hooks.items()}


def preset(name: str, **kw) -> dict[str, str]:
    if name == "box":
        return box_hooks(**kw)
    if name == "laptop":
        return via_ssh(box_hooks(**kw))
    if name in ("none", ""):
        return {}
    raise ValueError(f"unknown hooks preset {name!r} (box | laptop | none)")
