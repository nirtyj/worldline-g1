"""SonicArmScriptExecutor: STUB until M3 (PLAN §6.4, §6.6 ArmScript + CarryLock).

SONIC reaches with planner-mode upper_body_position targets (17-D, built only by body/joint_map.py) and the object
is attached at the grasp (STEPPING STONE). It needs the BodyServer `arm_script` op (pregrasp/grasp/lift/lower/
release/retract), `carry`, and P1 attach/detach; none exist in M1, so it is unhealthy and runs fail
controller_unavailable.
"""

from __future__ import annotations

from typing import Any

from api.types import ServiceHealth

from .kinematic_attach import ManipJob, ManipOutcome


class SonicArmScriptExecutor:
    backend = "sonic_arm_script"
    name = "sonic_arm_script"

    def health(self) -> ServiceHealth:
        return ServiceHealth(False, "planned", "needs the BodyServer arm_script op and a GT attach (M3)")

    async def run(self, job: ManipJob, handle: Any) -> ManipOutcome:
        return ManipOutcome("failed", "controller_unavailable", False, "execute", detail=self.health().detail)

    async def cancel(self) -> None:
        return None
