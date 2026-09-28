"""Pluggable scene for wl-isaac: an empty ground plane, or a house from the house agent's scenes/loader.py.

House plugin contract (docs/contracts/m1.md 1.8): `scenes/loader.py: load_house(stage, house_id) -> HouseInfo`.
P1 reads (duck-typed, all optional): house_id, floor_z, spawn, bounds, rooms, objects, occupancy_npz, and/or a
to_scene_info() method returning the full get_scene_info dict.
"""
from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ground material used in SONIC training: $WBC/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:319-330
GROUND_FRICTION = 1.0


@dataclass
class SceneInfo:
    house_id: str
    floor_z: float = 0.0
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw
    bounds: tuple[float, float, float, float] = (-15.0, -15.0, 15.0, 15.0)
    rooms: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    occupancy_npz: str | None = None
    source: str = "empty"
    raw: Any = None

    def to_dict(self) -> dict:
        d = None
        if self.raw is not None and hasattr(self.raw, "to_scene_info"):
            try:
                d = dict(self.raw.to_scene_info())
            except Exception as e:  # noqa: BLE001
                d = {"to_scene_info_error": str(e)}
        base = {"house_id": self.house_id, "floor_z": self.floor_z, "bounds": list(self.bounds),
                "rooms": self.rooms, "objects": self.objects,
                "spawn": {"x": self.spawn[0], "y": self.spawn[1], "yaw": self.spawn[2]}, "source": self.source}
        if d:
            base.update({k: v for k, v in d.items() if v is not None})
        return base


def _spawn_tuple(s) -> tuple[float, float, float]:
    if s is None:
        return (0.0, 0.0, 0.0)
    if isinstance(s, dict):
        return (float(s.get("x", 0.0)), float(s.get("y", 0.0)), float(s.get("yaw", 0.0)))
    s = list(s)
    return (float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.0)


def _add_ground(size: float = 30.0) -> None:
    import isaaclab.sim as sim_utils

    cfg = sim_utils.GroundPlaneCfg(
        size=(size, size),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=GROUND_FRICTION, dynamic_friction=GROUND_FRICTION),
    )
    cfg.func("/World/ground", cfg)


def add_default_lights() -> None:
    import isaaclab.sim as sim_utils

    dome = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    dome.func("/World/wl_dome_light", dome)
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    sun.func("/World/wl_sun", sun, orientation=(0.8536, 0.1464, 0.3536, 0.3536))


def build_scene(stage, house_id: str, log=print, default_lights: bool = True, loader_kwargs: dict | None = None
                ) -> SceneInfo:
    if house_id in ("", "empty", "none", "flat"):
        _add_ground()
        if default_lights:
            add_default_lights()
        return SceneInfo(house_id="empty", source="empty")

    loader_path = REPO_ROOT / "scenes" / "loader.py"
    if not loader_path.exists():
        raise RuntimeError(f"--house {house_id}: scenes/loader.py not found at {loader_path} "
                           "(the house agent's loader is required for houses; use --house empty)")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    loader = importlib.import_module("scenes.loader")
    log(f"[scene] loading house {house_id!r} via {loader_path}")
    kw = dict(loader_kwargs or {})
    try:
        info = loader.load_house(stage, house_id, **kw)
    except TypeError:
        if not kw:
            raise
        log(f"[scene] load_house does not accept {sorted(kw)}; loading with defaults")
        info = loader.load_house(stage, house_id)
    floor_z = float(getattr(info, "floor_z", 0.0) or 0.0)
    b = getattr(info, "bounds", None)
    bounds = tuple(float(x) for x in b) if b is not None else (-15.0, -15.0, 15.0, 15.0)
    si = SceneInfo(
        house_id=str(getattr(info, "house_id", house_id)),
        floor_z=floor_z,
        spawn=_spawn_tuple(getattr(info, "spawn", None)),
        bounds=bounds,
        rooms=list(getattr(info, "rooms", []) or []),
        objects=list(getattr(info, "objects", []) or []),
        occupancy_npz=getattr(info, "occupancy_npz", None),
        source="scenes.loader",
        raw=info,
    )
    if default_lights and not getattr(info, "has_lights", False):
        add_default_lights()
    return si


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
