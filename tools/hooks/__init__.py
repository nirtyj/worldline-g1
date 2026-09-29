"""Fault injection for the G1 stack scenarios (eval/stack_suite.py --hook; docs/eval_hooks.md). TEST-ONLY.

cli.py          box-side commands (python -m tools.hooks ...): P1 test ops, deploy kill/restart, recovery, delay proxy
delay_proxy.py  ZMQ REQ/REP delay proxy for the GR00T link (G4)
obstacle.py     where G6's box goes: across the robot's own route, at its narrowest point
fake_ops.py     the same P1 test ops on tools/fake_p1.FakeP1 (offline tests and rehearsals)
"""
