"""LiteWorld: the offline world for tests and the `lite` profile (PLAN §6.7, §11 M1.2).

Built from a recorded house (house_info.json + occupancy.npz, exactly what P1 serves through get_scene_info /
get_occupancy), so the map, `where`, visibility and reachability are the same code as on Isaac. The robot pose is
set by the kinematic LiteBody (robot/lite_body.py); objects move by in-process attach/detach. Everything it
reports carries `source: "lite-gt"` / `"lite-ground-truth"`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .gt_world import GTWorld, load_house_dir
from .mapgen import MapParams
from .model import RobotPose, xyyaw, xyz
from .perception import CameraModel, EGO_D435

HOUSES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fakes" / "houses"


def find_house(house_id: str, roots: list[str | Path] | None = None) -> Path:
    """Locate a recorded house directory: $WL_HOUSES_DIR, /work/worldline-g1/assets/houses, tests/fakes/houses."""
    import os
    cands = [Path(r) for r in (roots or [])]
    if os.environ.get("WL_HOUSES_DIR"):
        cands.append(Path(os.environ["WL_HOUSES_DIR"]))
    cands += [Path(__file__).resolve().parents[1] / "assets" / "houses", Path("/work/worldline-g1/assets/houses"),
              HOUSES_DIR]
    for c in cands:
        d = c / house_id
        if (d / "house_info.json").exists() and (d / "occupancy.npz").exists():
            return d
    raise FileNotFoundError(f"no recorded house {house_id!r} (house_info.json + occupancy.npz) under "
                            f"{[str(c) for c in cands]}; pull it with 00_infra/sync_wl.sh pull "
                            f"/work/worldline-g1/assets/houses/{house_id}")


class LiteWorld(GTWorld):
    source = "lite-gt"
    perception_source = "lite-ground-truth"

    def __init__(self, house: str | Path, *, scene_key: str | None = None, map_params: MapParams | None = None,
                 camera: CameraModel | str = EGO_D435, user_surface: str | None = None,
                 start: tuple[float, float, float] | None = None, pelvis_z: float = 0.78):
        d = Path(house) if Path(house).is_dir() else find_house(str(house))
        scene, occ = load_house_dir(d)
        super().__init__(scene, occ, scene_key=scene_key, map_params=map_params, camera=camera,
                         user_surface=user_surface)
        self.house_dir = d
        sx, sy, syaw = start or scene.spawn
        self._pose_lock = threading.Lock()
        self._pelvis_z = pelvis_z
        self._pose = RobotPose(sx, sy, scene.floor_z + pelvis_z, syaw, time.time(), "lite",
                               pelvis_z=pelvis_z)
        self.teleports = 0

    # ------------------------------------------------------------------ pose (driven by LiteBody)
    def robot_pose(self) -> RobotPose:
        with self._pose_lock:
            return self._pose

    def set_robot_pose(self, x: float, y: float, yaw: float, *, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
                       fallen: bool = False) -> None:
        pz = 0.3 if fallen else self._pelvis_z
        with self._pose_lock:
            self._pose = RobotPose(x, y, self.map.floor_z + pz, yaw, time.time(), "lite", fallen=fallen,
                                   pelvis_z=pz, vx=vx, vy=vy, wz=wz)

    # ------------------------------------------------------------------ SimControl (in-process)
    def capabilities(self) -> dict[str, bool]:
        return {"teleport_robot": True, "attach": True, "detach": True, "band": False, "reset_robot": True,
                "object_poses": True}

    def teleport_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        self.teleports += 1
        self.set_robot_pose(*xyyaw(pose, y, yaw))

    def reset_robot(self, pose, y: float | None = None, yaw: float | None = None) -> None:
        self.set_robot_pose(*xyyaw(pose, y, yaw))

    def attach(self, object_id: str, arm: str, mode: str = "kinematic") -> None:
        self._attach_local(object_id, arm, mode)

    def detach(self, object_id: str, pose=None) -> None:
        self._detach_local(object_id, xyz(pose))

    def move_object(self, object_id: str, center: tuple[float, float, float]) -> None:
        """Test/fixture helper: put an object somewhere (e.g. 0.25 m outside a skill stance for G14)."""
        from .gt_world import _translate
        self._set_object_box(object_id, _translate(self._boxes[object_id], center), source="fixture")
