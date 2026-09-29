"""Profiles (PLAN §6.7): which world, body, camera, scan and manipulation executors a session uses, plus the G1
constants (config/g1.yaml) and stack ports (config/stack.yaml). The numeric prompt slots come from
api.types.PROFILES (RobotProfile); `stepping_stones` is filled from what this profile actually loads.

    lite      LiteWorld + LiteBody                         executors: lite
    bringup   Isaac GT + wl-body (SONIC until M2.2's kinematic backend) + kinematic_attach
    sonic     Isaac GT + wl-body SONIC walking + (sonic_arm_script | kinematic_attach)
    full      + groot_sonic (stub until M4)
    real_g1   parity skeleton (not buildable in M2a)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.results import STEPPING_STONE_EXECUTORS
from api.types import PROFILES, RobotProfile

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


@dataclass
class StackProfile:
    name: str
    robot: RobotProfile
    world: str                           # lite | isaac | perception
    body: str                            # lite | sonic | kinematic
    camera: str                          # ego_d435 | head_sim
    scan_executor: str                   # turn_in_place | virtual | waist
    manip_executors: tuple[str, ...]
    manip_policy: str
    nav_executor: str
    frames: str                          # none | isaac
    g1: dict = field(default_factory=dict)
    stack: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def port_offset(self) -> int:
        return int(os.environ.get("WL_PORT_OFFSET", self.stack.get("port_offset", 0)) or 0)

    def camera_config(self) -> dict:
        return dict((self.g1.get("cameras") or {}).get(self.camera) or {})

    def scene_config(self, scene: str) -> dict:
        return dict((self.stack.get("scenes") or {}).get(scene) or {})


def load_profile(name: str, *, config_dir: str | Path | None = None, overrides: dict | None = None) -> StackProfile:
    cdir = Path(config_dir or CONFIG)
    path = cdir / "profiles" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown profile {name!r} (no {path})")
    raw = _yaml(path)
    raw.update(overrides or {})
    g1 = _yaml(cdir / "g1.yaml")
    stack = _yaml(cdir / "stack.yaml")
    manip = raw.get("manipulation") or {}
    executors = tuple(manip.get("executors") or ("lite",))
    nav_ex = str(raw.get("nav_executor") or ("lite" if raw.get("body") == "lite" else "sonic_walk"))
    base = PROFILES.get(name, RobotProfile(name=name))                 # type: ignore[arg-type]
    stones = tuple(dict.fromkeys(list(base.stepping_stones) + [e for e in (nav_ex, *executors)
                                                              if e in STEPPING_STONE_EXECUTORS]))
    robot = dataclasses.replace(base, stepping_stones=stones,
                                approach_max_m=float((g1.get("workspace") or {}).get("approach_max_m",
                                                                                     base.approach_max_m)),
                                reach_h_min_m=float((g1.get("workspace") or {}).get("obj_z_min_m", base.reach_h_min_m)),
                                reach_h_max_m=float((g1.get("workspace") or {}).get("obj_z_max_m", base.reach_h_max_m)))
    return StackProfile(name=name, robot=robot, world=str(raw.get("world", "lite")), body=str(raw.get("body", "lite")),
                        camera=str(raw.get("perception_camera", "head_sim")),
                        scan_executor=str(raw.get("scan_executor") or (g1.get("scan") or {}).get("executor")
                                          or "turn_in_place"),
                        manip_executors=executors, manip_policy=str(manip.get("policy") or "first_healthy"),
                        nav_executor=nav_ex, frames=str(raw.get("frames", "none")), g1=g1, stack=stack, raw=raw)
