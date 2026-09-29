"""GrootSonicExecutor: RETIRED (owner decision (b), PLAN §0.7, 2026-09-29). Kept only as a deprecated alias.

The token route (GR00T -> SONIC motion tokens through BodyServer vla_start/VlaStreamer, PLAN M4, embodiment
UNITREE_G1_SONIC) is replaced by `groot_arms` (services/executors/groot_arms.py): GR00T N1.7 outputs arm and Dex3
hand joint targets, streamed through the body `arm` op in chunk mode, and SONIC keeps the legs.
`api.skills.EXECUTOR_OF_BACKEND["groot"]` is `groot_arms`.

A profile that still lists `groot_sonic` gets this stub: always down, so its skills are rejected at CAPABILITY
("policy unavailable: ...") while the object_type enum stays frozen, and a run that slips through fails
policy_unavailable. It never talks to a body or a policy server. Use `groot_arms` instead.
"""

from __future__ import annotations

from typing import Any

from api.types import ServiceHealth

from .kinematic_attach import ManipJob, ManipOutcome

REPLACED_BY = "groot_arms"


class GrootSonicExecutor:
    backend = "groot"
    name = "groot_sonic"
    deprecated = True

    def __init__(self, *, policy_port: int = 5550, body: Any = None):
        self.policy_port = policy_port
        self.body = body

    def health(self) -> ServiceHealth:
        return ServiceHealth(False, "down", f"groot_sonic is retired (the SONIC-token route of PLAN M4); the groot "
                                            f"backend runs {REPLACED_BY}: list {REPLACED_BY} in the profile")

    async def run(self, job: ManipJob, handle: Any) -> ManipOutcome:
        return ManipOutcome("failed", "policy_unavailable", False, "select_skill", detail=self.health().detail)

    async def cancel(self) -> None:
        return None
