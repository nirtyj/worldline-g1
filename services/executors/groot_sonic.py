"""GrootSonicExecutor: STUB until M4 (PLAN §6.4 "The GR00T execution layer", §7.4).

The target chain is ManipulationService -> SkillRegistry -> GrootSonicExecutor -> BodyClient.vla_start ->
VlaStreamer -> PolicyServer (5550) -> SonicMux -> deploy. M1's body has no vla_start/vla_stop ops and no
PolicyServer runs in M2a, so this executor reports itself unhealthy (the registry keeps the type in the frozen enum;
calls are rejected at CAPABILITY with `policy unavailable: ...`) and any run ends failed(policy_unavailable).
"""

from __future__ import annotations

from typing import Any

from api.types import ServiceHealth

from .kinematic_attach import ManipJob, ManipOutcome


class GrootSonicExecutor:
    backend = "groot"
    name = "groot_sonic"

    def __init__(self, *, policy_port: int = 5550, body: Any = None):
        self.policy_port = policy_port
        self.body = body

    def health(self) -> ServiceHealth:
        return ServiceHealth(False, "down", f"GR00T executor not available: needs BodyServer vla_start and a "
                                            f"PolicyServer on {self.policy_port} (M4)")

    async def run(self, job: ManipJob, handle: Any) -> ManipOutcome:
        return ManipOutcome("failed", "policy_unavailable", False, "enter", detail=self.health().detail)

    async def cancel(self) -> None:
        return None
