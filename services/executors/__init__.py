"""Manipulation executors behind ManipulationService (PLAN §6.4, §12.2).

    kinematic_attach   STEPPING STONE: timed phases + SimControl.attach/detach (LiteWorld in process; P1 in M2b)
    lite               the same executor labelled `lite` for the lite profile
    sonic_arm_script   stub (M3: BodyServer arm_script + attach)
    groot_arms         owner groot_rt (services/executors/groot_arms.py `create`), experimental
    groot_sonic        stub (the retired token route)

registry.py is the one place a profile's executor names become executors (build_executor, register_executor).
"""

from .groot_sonic import GrootSonicExecutor
from .kinematic_attach import KinematicAttachExecutor, ManipJob, ManipOutcome
from .registry import ExecutorContext, UnavailableExecutor, backend_of, build_executor, register_executor
from .sonic_arm_script import SonicArmScriptExecutor

__all__ = ["KinematicAttachExecutor", "ManipJob", "ManipOutcome", "GrootSonicExecutor", "SonicArmScriptExecutor",
           "ExecutorContext", "UnavailableExecutor", "backend_of", "build_executor", "register_executor"]
