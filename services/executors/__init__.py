"""Manipulation executors behind ManipulationService (PLAN §6.4, §12.2).

    kinematic_attach   STEPPING STONE: timed phases + SimControl.attach/detach (LiteWorld in process; P1 in M2b)
    lite               the same executor labelled `lite` for the lite profile
    sonic_arm_script   stub (M3: BodyServer arm_script + attach)
    groot_sonic        stub (M4: BodyServer vla_start + PolicyServer)
"""

from .groot_sonic import GrootSonicExecutor
from .kinematic_attach import KinematicAttachExecutor, ManipJob
from .sonic_arm_script import SonicArmScriptExecutor

__all__ = ["KinematicAttachExecutor", "ManipJob", "GrootSonicExecutor", "SonicArmScriptExecutor"]
