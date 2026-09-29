"""build(profile, scene, clock, log) -> (world, robot, frames)  (PLAN §6.7; ui/server.py Session.start calls it).

    scene   "procthor-train-38", or "<house>@<variant>" (the variant is part of the memory key; the house before
            "@" selects the MolmoSpaces house). config/stack.yaml `scenes:` may map a scene key to a house and set
            per-scene overrides such as `user_surface`.

    lite      LiteWorld(recorded house) + LiteBody (kinematic)            frames: none
    bringup   IsaacGTWorldModel(P1 5600/5601) + SonicBody(wl-body 5610)   frames: viz FrameTap
    sonic     same                                                        (+ scan/manip executors per profile)
    full      same (+ groot_sonic stub)
"""

from __future__ import annotations

from typing import Any

from world.mapgen import MapParams
from world.perception import CameraModel

from .bridge import G1Robot
from .profile import StackProfile, load_profile


def parse_scene(scene: str, profile: StackProfile) -> tuple[str, dict]:
    cfg = profile.scene_config(scene)
    house = cfg.get("house") or scene.split("@", 1)[0]
    return str(house), cfg


def camera_of(profile: StackProfile) -> CameraModel:
    d = profile.camera_config()
    if not d:
        from world.perception import CAMERAS
        return CAMERAS[profile.camera]
    return CameraModel.from_dict(profile.camera, d)


def build(profile: str | StackProfile, scene: str | None = None, clock: Any = None, log: Any = None, **kw: Any
          ) -> tuple[Any, G1Robot, Any]:
    prof = profile if isinstance(profile, StackProfile) else load_profile(profile)
    if clock is None:
        from sim.clock import SimClock
        clock = SimClock(1.0)
    scene = scene or prof.stack.get("default_scene") or "procthor-train-38"
    house, scfg = parse_scene(scene, prof)
    mp = MapParams.from_dict(prof.g1.get("mapgen"))
    cam = camera_of(prof)
    user_surface = kw.get("user_surface", scfg.get("user_surface"))
    if prof.world == "lite":
        from world.frames import NoFrames
        from world.lite_world import LiteWorld

        from .lite_body import LiteBody
        world = LiteWorld(kw.get("house_dir") or house, scene_key=scene, map_params=mp, camera=cam,
                          user_surface=user_surface)
        body = LiteBody(world, clock, prof.g1.get("walking"))
        frames = NoFrames()
    elif prof.world == "isaac":
        from world.frames import IsaacFrames
        from world.isaac_client import IsaacGTWorldModel

        from .body_client import SonicBody
        world = IsaacGTWorldModel(port_offset=prof.port_offset, scene_key=scene, map_params=mp, camera=cam,
                                  user_surface=user_surface)
        body = kw.get("body") or SonicBody(port_offset=prof.port_offset, sim_control=world)
        frames = kw.get("frames") or (IsaacFrames(world, port_offset=prof.port_offset) if prof.frames == "isaac"
                                      else None)
    else:
        raise NotImplementedError(f"profile {prof.name}: world {prof.world!r} needs world/perception.py's "
                                  f"PerceptionWorldModel (PLAN §13); not in M2a")
    robot = G1Robot(world, body, profile=prof, clock=clock, log=log, frames=frames)
    return world, robot, frames
