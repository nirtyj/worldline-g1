"""Joint limits straight from a URDF file (stdlib XML), for the box-side checks and for pinning joint_order.URDF_LIMITS.

On both boxes the URDF P1 simulates is /work/worldline-g1/assets/g1/g1_sonic_dex3.urdf (43 actuated joints: SONIC's
main.urdf body + the Dex3 hands of g1_29dof_with_hand.urdf; built by sim_isaac/g1_asset.py).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

BOX_URDF = "/work/worldline-g1/assets/g1/g1_sonic_dex3.urdf"


def load_limits(path: str | Path = BOX_URDF) -> dict[str, tuple[float, float]]:
    """{joint name: (lower, upper)} for every revolute/prismatic joint with a <limit>."""
    root = ET.parse(str(path)).getroot()
    out: dict[str, tuple[float, float]] = {}
    for j in root.iter("joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        lim = j.find("limit")
        if lim is None or lim.get("lower") is None or lim.get("upper") is None:
            continue
        out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out
