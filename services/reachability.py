"""check_reachability (doc §8-§9; PLAN §6.4): judged FROM THE CURRENT POSE ONLY, with G1 geometry.

Checks, in THOR's order (thor/robot.py:434-459) so prompts and eval semantics carry over, plus the G1 ones:

  1. base_moving          the body is walking / turning (or navigate is running)
  2. not_found            the object_id / candidates name nothing that exists
  3. in_hand              already held
  4. not_seen_here        not visible now and not in the last scan here -> visible=false (positions never leak;
                          also the answer for a type with no visible instance, so "no banana" is never revealed)
  5. inside_or_on_<x>     its where is not a map surface (fridge, chair, floor, ...)
  6. too_high / too_low   object centre above obj_z_max_m / below obj_z_min_m (squat mode deferred)
  7. in the reach window  forward reach_fwd_m and |lateral| <= reach_lat_max_m in the CURRENT pelvis frame, and
                          inside the skill's stance tolerance when it has one (GR00T skills):
       - no -> a free stance within approach_max_m that satisfies both? -> needs_reposition (+ stance, world pose
         and delta, suggest_location="reach_stance"); none -> too_far (+ suggest_location: the stand of the same
         furniture from which it is in reach, else its surface)
       - yes but farther than arm_reach_m from the nearer shoulder -> out_of_workspace (the IK stand-in)
  8. hand_full            max_held (1) objects already held
  9. no_skill             no loaded skill handles pick of this type (with the preferred arm)

preferred_arm: the sign of the lateral offset in the current pelvis frame (left = +y), "either" inside the dead
band, restricted to the selected skill's arms. `reachable=true` always means "manipulable from exactly here".
All geometry is GT (`source: isaac-gt|lite-gt`; §13 row 4 swaps in detected poses + IK).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Iterable

from api.results import ReachabilityResult
from world import coords


@dataclass(frozen=True)
class G1Workspace:
    shoulder_z_m: float = 1.08
    shoulder_lat_m: float = 0.10
    arm_reach_m: float = 0.65
    obj_z_min_m: float = 0.55
    obj_z_max_m: float = 1.20
    approach_max_m: float = 0.40
    reach_fwd_m: tuple[float, float] = (0.20, 0.55)
    reach_lat_max_m: float = 0.40
    either_deadband_m: float = 0.10
    stance_fwd_m: tuple[float, float] = (0.30, 0.45)
    stance_lat_m: tuple[float, ...] = (0.0, 0.10, -0.10, 0.15, -0.15)
    stance_clearance_m: float = 0.25
    stance_margin_m: float = 0.02       # a reposition lands within a few cm: keep the stance inside the reach
    max_held: int = 1
    moving_speed_mps: float = 0.05

    @classmethod
    def from_dict(cls, d: dict | None) -> "G1Workspace":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if isinstance(f.default, tuple):
                kw[f.name] = tuple(float(x) for x in v)
            elif isinstance(f.default, int) and not isinstance(f.default, bool):
                kw[f.name] = int(v)
            else:
                kw[f.name] = float(v)
        return cls(**kw)


class ReachabilityModel:
    def __init__(self, world: Any, workspace: G1Workspace | None = None, *, registry: Any = None,
                 observation: Any = None, nav: Any = None, body: Any = None):
        self.world = world
        self.ws = workspace or G1Workspace()
        self.registry = registry
        self.observation = observation
        self.nav = nav
        self.body = body

    # ------------------------------------------------------------------ helpers
    def _moving(self) -> bool:
        p = self.world.robot_pose()
        if p.speed > self.ws.moving_speed_mps or abs(p.wz) > 0.1:
            return True
        if self.nav is not None and getattr(self.nav, "moving", False):
            return True
        if self.body is not None:
            st = self.body.state() or {}
            if (st.get("active") or {}).get("op") in ("go_to", "walk", "turn_to", "velocity"):
                return True
        return False

    def _visible_now(self) -> set[str]:
        return {d.id for d in self.world.detections() if d.kind == "object"}

    def _seen_here(self, at: str | None) -> set[str]:
        return self.observation.seen_here(at) if self.observation is not None else set()

    def in_body_frame(self, x: float, y: float, pose=None) -> tuple[float, float]:
        pose = pose or self.world.robot_pose()
        return coords.world_to_body((pose.x, pose.y, pose.yaw), x, y)

    def in_window(self, fwd: float, lat: float, skill: Any = None, pose=None, obj=None) -> bool:
        lo, hi = self.ws.reach_fwd_m
        if not (lo <= fwd <= hi and abs(lat) <= self.ws.reach_lat_max_m):
            return False
        st = dict(getattr(skill, "stance", None) or {})
        if st and "stand_off_m" in st:
            tol_m, tol_deg = (st.get("tol") or [0.05, 5.0])[:2]
            if abs(fwd - float(st["stand_off_m"])) > tol_m or abs(lat - float(st.get("lateral_m", 0.0))) > tol_m:
                return False
            if obj is not None and pose is not None:
                want = math.atan2(obj[1] - pose.y, obj[0] - pose.x) + float(st.get("yaw_to_object", 0.0))
                if abs(math.degrees(coords.ang_diff(want, pose.yaw))) > tol_deg and abs(lat) > tol_m:
                    return False
        return True

    def shoulder_dist(self, fwd: float, lat: float, z: float) -> float:
        side = 1.0 if lat >= 0 else -1.0
        sz = self.world.static_map().floor_z + self.ws.shoulder_z_m
        return math.sqrt(fwd ** 2 + (lat - side * self.ws.shoulder_lat_m) ** 2 + (z - sz) ** 2)

    def preferred_arm(self, lat: float, skill: Any = None) -> str:
        if lat > self.ws.either_deadband_m:
            arm = "left"
        elif lat < -self.ws.either_deadband_m:
            arm = "right"
        else:
            arm = "either"
        arms = tuple(getattr(skill, "arms", ("left", "right")) or ("left", "right"))
        if arm == "either":
            return "either" if set(arms) >= {"left", "right"} else arms[0]
        return arm if arm in arms else (arms[0] if abs(lat) <= self.ws.reach_lat_max_m else "none")

    # ------------------------------------------------------------------ stance search (needs_reposition)
    def find_stance(self, obj_xy: tuple[float, float], skill: Any = None, z: float | None = None) -> dict | None:
        """A free stance within approach_max_m of here from which the object is in the window (and within
        arm_reach_m of the nearer shoulder when its height z is given)."""
        p = self.world.robot_pose()
        grid = self.world.static_map().grid
        ws = self.ws
        st = dict(getattr(skill, "stance", None) or {})
        fwds = [float(st["stand_off_m"])] if "stand_off_m" in st else \
            [ws.stance_fwd_m[0] + i * 0.05 for i in range(int(round((ws.stance_fwd_m[1] - ws.stance_fwd_m[0]) / 0.05)) + 1)]
        lats = [float(st.get("lateral_m", 0.0))] if "stand_off_m" in st else list(ws.stance_lat_m)
        to_obj = math.atan2(obj_xy[1] - p.y, obj_xy[0] - p.x)
        yaws = [p.yaw]
        for y in (to_obj, self._keypoint_yaw()):
            if y is not None and all(abs(coords.ang_diff(y, q)) > math.radians(3) for q in yaws):
                yaws.append(y)
        best = None
        for yaw in yaws:
            c, s = math.cos(yaw), math.sin(yaw)
            for f in fwds:
                for l in lats:
                    if z is not None and self.shoulder_dist(f, l, z) > ws.arm_reach_m - ws.stance_margin_m:
                        continue
                    sx = obj_xy[0] - (c * f - s * l)
                    sy = obj_xy[1] - (s * f + c * l)
                    d = math.hypot(sx - p.x, sy - p.y)
                    if d > ws.approach_max_m + 1e-9:
                        continue
                    if grid.clearance(sx, sy) < ws.stance_clearance_m or not grid.is_free(sx, sy):
                        continue
                    if grid.nav.c_blocked[grid.nav._world_to_c(sx, sy)]:
                        continue                  # the planner would snap this goal to another cell
                    if not grid.segment_free((p.x, p.y), (sx, sy)):
                        continue
                    cost = d + 0.3 * abs(coords.ang_diff(yaw, p.yaw)) + 0.2 * abs(l)
                    if best is None or cost < best[0]:
                        dx, dy = sx - p.x, sy - p.y
                        bf, bl = coords.world_to_body((p.x, p.y, p.yaw), sx, sy)
                        best = (cost, {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(coords.wrap_pi(yaw), 4),
                                       "dx": round(dx, 3), "dy": round(dy, 3),
                                       "dyaw": round(coords.ang_diff(yaw, p.yaw), 4),
                                       "forward": round(bf, 3), "left": round(bl, 3), "distance_m": round(d, 3)})
        return best[1] if best else None

    def _keypoint_yaw(self) -> float | None:
        at = self.nav.at()[0] if self.nav is not None else None
        k = self.world.static_map().keypoints.get(at) if at else None
        return k.yaw if k is not None else None

    def suggest_keypoint(self, obj_xy: tuple[float, float], z: float, where: str) -> str | None:
        """too_far: the stand of the same furniture from which the object is in the window, else its surface."""
        m = self.world.static_map()
        s = m.surfaces.get(where)
        if s is None:
            return None
        cands = [x for x in m.surfaces.values() if x.furniture_id == s.furniture_id]
        best = None
        for x in cands:
            k = m.keypoints[x.name]
            fwd, lat = coords.world_to_body((k.x, k.y, k.yaw), *obj_xy)
            ok = self.ws.reach_fwd_m[0] <= fwd <= self.ws.reach_fwd_m[1] + self.ws.approach_max_m and \
                abs(lat) <= self.ws.reach_lat_max_m + self.ws.approach_max_m
            d = math.hypot(k.x - obj_xy[0], k.y - obj_xy[1])
            key = (0 if ok else 1, d)
            if best is None or key < best[0]:
                best = (key, x.name)
        return best[1] if best else where

    # ------------------------------------------------------------------ the check
    def check(self, object_type: str, object_id: str | None = None, *, candidates: Iterable[str] = (),
              at: str | None = None) -> ReachabilityResult:
        m = self.world.static_map()
        if at is None and self.nav is not None:
            at = self.nav.at()[0]

        def res(reachable=False, visible=False, arm="none", reason=None, oid=None, **kw) -> ReachabilityResult:
            return ReachabilityResult(reachable=reachable, visible=visible, preferred_arm=arm, reason=reason,   # type: ignore[arg-type]
                                      object_type=object_type, object_id=oid, at=at, **kw)

        # 1. base moving
        if self._moving():
            return res(reason="base_moving", oid=object_id)
        # 2. which instance
        cands = [object_id] if object_id else [c for c in candidates if c]
        objs = self.world.objects()
        visible = self._visible_now()
        seen = self._seen_here(at)
        if cands:
            known = [c for c in cands if c in objs]
            if not known:
                return res(reason="not_found", oid=object_id or cands[0])
            same_type = [c for c in known if objs[c].type == object_type] or known
            ordered = sorted(same_type, key=lambda c: (c not in visible, c not in seen))
        else:
            of_type = [oid for oid, o in objs.items() if o.type == object_type]
            ordered = sorted(of_type, key=lambda c: (c not in visible, c not in seen, c))
            ordered = [c for c in ordered if c in visible or c in seen or objs[c].held_by] or ordered[:1]
            if not ordered:
                # no instance of that type anywhere: answer as unseen (never reveal absence, eval missing_object)
                return res(reason="not_seen_here")
        oid = ordered[0]
        o = objs[oid]
        # 3. in hand
        if o.held_by:
            return res(reason="in_hand", visible=True, oid=oid)
        # 4. not seen here
        is_vis = oid in visible
        if not is_vis and oid not in seen:
            return res(reason="not_seen_here", oid=oid)
        # 5. inside or on something that is not a surface
        if o.where not in m.surfaces:
            return res(reason=f"inside_or_on_{o.where}", visible=is_vis, oid=oid)
        x, y, z = o.pos
        h = z - m.floor_z
        pose = self.world.robot_pose()
        dist = math.hypot(x - pose.x, y - pose.y)
        common = {"distance_m": round(dist, 3), "height_m": round(h, 3)}
        # 6. height
        if h > self.ws.obj_z_max_m:
            return res(reason="too_high", visible=is_vis, oid=oid, **common)
        if h < self.ws.obj_z_min_m:
            return res(reason="too_low", visible=is_vis, oid=oid, **common)
        skill = self.registry.select("pick", object_type, None) if self.registry is not None else None
        # 7. reach window from exactly here
        fwd, lat = self.in_body_frame(x, y, pose)
        if not self.in_window(fwd, lat, skill, pose, (x, y)):
            stance = self.find_stance((x, y), skill, z)
            if stance is not None:
                return res(reason="needs_reposition", visible=is_vis, oid=oid, suggest_location="reach_stance",
                           stance=stance, skill_id=getattr(skill, "skill_id", None), **common)
            sug = self.suggest_keypoint((x, y), z, o.where)
            return res(reason="too_far", visible=is_vis, oid=oid, suggest_location=sug if sug != at else None,
                       **common)
        if self.shoulder_dist(fwd, lat, z) > self.ws.arm_reach_m:
            return res(reason="out_of_workspace", visible=is_vis, oid=oid, **common)
        # 8. hands
        held = [v for v in self.world.hands().values() if v]
        if len(held) >= self.ws.max_held:
            return res(reason="hand_full", visible=is_vis, oid=oid, **common)
        # 9. a skill for it
        arm = self.preferred_arm(lat, skill)
        if skill is None or arm == "none":
            return res(reason="no_skill", visible=is_vis, oid=oid, **common)
        return res(reachable=True, visible=is_vis, arm=arm, oid=oid, skill_id=skill.skill_id,
                   stance={"x": round(pose.x, 3), "y": round(pose.y, 3), "yaw": round(pose.yaw, 4),
                           "forward": round(fwd, 3), "left": round(lat, 3)}, **common)
