"""check_reachability (doc §8-§9; PLAN §6.4): judged FROM THE CURRENT POSE ONLY, with G1 geometry.

Checks, in THOR's order (thor/robot.py:434-459) so prompts and eval semantics carry over, plus the G1 ones:

  1. base_moving          the body is walking / turning (or navigate is running)
  2. not_found            the object_id / candidates name nothing that exists; without candidates, nothing of that
                          type is visible, in the last scan here or in a hand. Same answer (and no object id)
                          whether or not an instance exists elsewhere, so "no banana" is never revealed
  3. in_hand              already held
  4. not_seen_here        a candidate that is not visible now and not in the last scan here -> visible=false
                          (positions never leak)
  5. inside_or_on_<x>     its where is not a map surface (fridge, chair, floor, ...)
  6. too_high / too_low   object centre above obj_z_max_m / below obj_z_min_m (squat mode deferred)
  7. in the reach window  forward reach_fwd_m and |lateral| <= reach_lat_max_m in the CURRENT pelvis frame, and
                          inside the skill's stance tolerance when it has one (GR00T skills):
       - no -> a stance within approach_max_m that satisfies both? -> needs_reposition (+ stance, world pose
         and delta, suggest_location="reach_stance"); none -> too_far (+ suggest_location: the stand of the same
         furniture from which it is in reach, else its surface)
       - yes but the grasp point is outside the arm's sphere (below) -> needs_reposition when a nearby stance
         reaches it, else out_of_workspace
       - neither, and no spot the robot can stand on (stance_clearance_m from every obstacle) lies within the arm's
         horizontal reach of the object -> beyond_reach (+ detail): no stand, stance or reposition can help (H15's
         apple, 0.44 m deep on a 0.59 m counter, 15 cm from the wall, is 0.64 m from any such spot vs 0.50 m)
  8. hand_full            max_held (1) objects already held
  9. no_skill             no loaded skill handles pick of this type (with the preferred arm)

preferred_arm: the sign of the lateral offset in the current pelvis frame (left = +y), "either" inside the dead
band, restricted to the selected skill's arms. `reachable=true` always means "manipulable from exactly here".
All geometry is GT (`source: isaac-gt|lite-gt`; §13 row 4 swaps in detected poses + IK).

The arm (R.7 / M3.5 calibration, world/workspace_cal.py; config/g1.yaml `workspace`): the palm reaches a sphere
around the shoulder, centre (shoulder_fwd_m, +-shoulder_lat_m, shoulder_z_m above the floor) in the pelvis frame,
radius arm_reach_m. That sphere is the fit of body/arm_script.py's own IK envelope (grasp and pregrasp solvable with
the wrist pitch and yaw locked; residual rms 4 mm), shrunk by the live palm check's margin. The point it must reach
is the grasp point sonic_arm_script uses: the object's AABB top centre + grasp_above_top_m (B.7's hovering top
grasp). Heights (step 6) stay on the object's centre, the numbers the planner prompt quotes.

Reach stances (`stance_via`): `approach` (the body's strafing reposition, B.6, R.2) stands the pelvis
stance_clearance_m from the nearest obstacle (raw occupancy) on a straight segment from here; `go_to` (A*: the
interim reposition) needs the planner's free space (body radius inflated); `two_step` (not wired in navigation yet: an
A* go_to next to the stance, then the approach) drops the straight segment, so approach_max_m can grow (R.7 coverage:
48 % of in-band household pickables reachable instead of 23 %, of a 54 % physical bound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Iterable

from api.results import ReachabilityResult
from world import coords


@dataclass(frozen=True)
class G1Workspace:
    """Defaults = config/g1.yaml `workspace` (the R.7 calibration); the config is what the runtime loads."""
    shoulder_z_m: float = 1.089         # reach-sphere centre above the floor (pelvis 0.79 + 0.299)
    shoulder_lat_m: float = 0.136       # ... and its lateral offset (the arm's side)
    shoulder_fwd_m: float = 0.0         # ... and its forward offset (pelvis frame)
    arm_reach_m: float = 0.40           # reach-sphere radius (IK fit 0.4255 minus the live margin)
    grasp_above_top_m: float = 0.03     # grasp point = AABB top centre + this (sonic_arm_script, B.7)
    grasp_ref: str = "top"              # "top": the grasp point above; "centre": the object's centre (M2a)
    obj_z_min_m: float = 0.55
    obj_z_max_m: float = 1.20
    approach_max_m: float = 0.40
    reach_fwd_m: tuple[float, float] = (0.15, 0.43)
    reach_lat_max_m: float = 0.40
    either_deadband_m: float = 0.10
    stance_fwd_m: tuple[float, float] = (0.22, 0.36)
    stance_lat_m: tuple[float, ...] = (0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15)
    stance_clearance_m: float = 0.25
    stance_via: str = "go_to"           # "approach" (body B.6) | "go_to" (A*, the interim reposition) | "two_step"
    stance_search: str = "window"       # "window": the whole reach window, any direction; "band": M2a's search
    stance_margin_m: float = 0.02       # a reposition lands within a few cm: keep the stance inside the reach
    max_held: int = 1
    moving_speed_mps: float = 0.05
    lite_world: tuple = ()              # INTERIM overrides for LiteWorld (config `workspace.lite_world`), see for_world

    @classmethod
    def from_dict(cls, d: dict | None) -> "G1Workspace":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name == "lite_world":
                kw[f.name] = tuple(sorted((str(k), tuple(x) if isinstance(x, list) else x)
                                          for k, x in dict(v or {}).items()))
            elif isinstance(f.default, tuple):
                kw[f.name] = tuple(float(x) for x in v)
            elif isinstance(f.default, str):
                kw[f.name] = str(v)
            elif isinstance(f.default, int) and not isinstance(f.default, bool):
                kw[f.name] = int(v)
            else:
                kw[f.name] = float(v)
        if kw.get("stance_search", "window") not in ("window", "band"):
            raise ValueError(f"workspace.stance_search must be window or band, not {kw['stance_search']!r}")
        if kw.get("grasp_ref", "top") not in ("top", "centre"):
            raise ValueError(f"workspace.grasp_ref must be top or centre, not {kw['grasp_ref']!r}")
        if kw.get("stance_via", "go_to") not in ("approach", "go_to", "two_step"):
            raise ValueError(f"workspace.stance_via must be approach, go_to or two_step, not {kw['stance_via']!r}")
        return cls(**kw)

    def for_world(self, world: Any) -> "G1Workspace":
        """The workspace this world's reachability uses. The calibrated G1 arm everywhere, except that a LiteWorld
        (source lite-gt: the `lite` profile and the laptop test stacks) takes the INTERIM `lite_world` overrides
        (M2a's optimistic arm) while tests that pick an object the calibrated arm cannot reach are migrated (the R.7
        result lists them). The overrides are labelled in `lite_arm_note`."""
        if not self.lite_world or getattr(world, "source", None) != "lite-gt":
            return self
        from dataclasses import asdict
        d = {k: v for k, v in asdict(self).items() if k != "lite_world"}
        d.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in self.lite_world})
        return G1Workspace.from_dict(d)


class ReachabilityModel:
    def __init__(self, world: Any, workspace: G1Workspace | None = None, *, registry: Any = None,
                 observation: Any = None, nav: Any = None, body: Any = None):
        self.world = world
        base = workspace or G1Workspace()
        self.ws = base.for_world(world)
        # honesty label: which arm this reachability judges with
        self.lite_arm_note = ("lite world: M2a's optimistic arm (INTERIM, config workspace.lite_world), not the "
                              "calibrated G1 arm" if self.ws is not base else None)
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
        """Distance from the nearer arm's reach-sphere centre to a palm target (pelvis-frame fwd/lat, world z)."""
        side = 1.0 if lat >= 0 else -1.0
        sz = self.world.static_map().floor_z + self.ws.shoulder_z_m
        return math.sqrt((fwd - self.ws.shoulder_fwd_m) ** 2 + (lat - side * self.ws.shoulder_lat_m) ** 2
                         + (z - sz) ** 2)

    def grasp_z(self, obj: Any) -> float:
        """The height the palm must reach for this object: its AABB top + grasp_above_top_m (its centre if it has
        no box)."""
        box = getattr(obj, "box", None)
        if box is not None and self.ws.grasp_ref == "top":
            return float(box[1][2]) + self.ws.grasp_above_top_m
        return float(obj.pos[2])

    def horizontal_reach(self, z: float) -> float:
        """The farthest horizontal pelvis-to-palm distance inside the reach window and the sphere at height z (m);
        0 if the sphere does not reach that height."""
        ws = self.ws
        dz = z - (self.world.static_map().floor_z + ws.shoulder_z_m)
        best = 0.0
        n = int(round(ws.reach_lat_max_m / 0.02))
        for i in range(n + 1):
            lat = i * 0.02
            h2 = ws.arm_reach_m ** 2 - (lat - ws.shoulder_lat_m) ** 2 - dz ** 2
            if h2 < 0:
                continue
            fwd = min(ws.reach_fwd_m[1], ws.shoulder_fwd_m + math.sqrt(h2))
            if fwd >= ws.reach_fwd_m[0]:
                best = max(best, math.hypot(fwd, lat))
        return best

    def stance_ok(self, sx: float, sy: float, pose: Any = None) -> bool:
        """Can the robot take a reach stance at (sx, sy) from `pose` (default: where it is now) by this workspace's
        stance rule (`stance_via`)? The one rule for find_stance and for navigate(reach_stance)'s own check."""
        return self._stance_ok(self.world.static_map().grid, pose or self.world.robot_pose(), sx, sy)

    def _stance_ok(self, grid: Any, p: Any, sx: float, sy: float) -> bool:
        ws = self.ws
        if ws.stance_via == "approach":
            # the body's approach: one straight segment on ground truth; the pelvis stance_clearance_m from the
            # furniture (raw occupancy: the toes are 0.13 m ahead of the pelvis, B.7 stood at 0.19 m), and nothing
            # closer than that minus 5 cm on the way
            if grid.clearance(sx, sy) < ws.stance_clearance_m or grid.is_outside(sx, sy):
                return False
            n = max(2, int(math.hypot(sx - p.x, sy - p.y) / 0.02) + 1)
            floor = min(ws.stance_clearance_m, grid.clearance(p.x, p.y)) - 0.05
            return all(grid.clearance(p.x + (sx - p.x) * t, p.y + (sy - p.y) * t) >= floor
                       for t in (k / (n - 1) for k in range(n)))
        if ws.stance_via == "two_step":
            # A* go_to to a free spot next to the stance, then the approach (no straight line from here needed):
            # the stance as for `approach`, and a free cell of the robot's component within 0.3 m of it
            if grid.clearance(sx, sy) < ws.stance_clearance_m or grid.is_outside(sx, sy):
                return False
            comp = grid.component(p.x, p.y)
            for r in (0.1, 0.2, 0.3):
                for k in range(12):
                    a = k * math.pi / 6
                    qx, qy = sx + r * math.cos(a), sy + r * math.sin(a)
                    if grid.is_free(qx, qy) and (comp is None or grid.component(qx, qy) == comp):
                        return True
            return False
        if grid.clearance(sx, sy) < ws.stance_clearance_m or not grid.is_free(sx, sy):
            return False
        if grid.nav.c_blocked[grid.nav._world_to_c(sx, sy)]:
            return False                      # the planner would snap this goal to another cell
        return bool(grid.segment_free((p.x, p.y), (sx, sy)))

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
    def _find_stance_band(self, obj_xy: tuple[float, float], skill: Any = None, z: float | None = None) -> dict | None:
        """M2a's search (stance_search "band", the lite world's INTERIM arm): the object straight ahead in the
        stance_fwd_m band at the stance_lat_m offsets, facing as now, at the object or as the keypoint does."""
        p = self.world.robot_pose()
        grid = self.world.static_map().grid
        ws = self.ws
        st = dict(getattr(skill, "stance", None) or {})
        fwds = [float(st["stand_off_m"])] if "stand_off_m" in st else \
            [ws.stance_fwd_m[0] + i * 0.02 for i in range(int(round((ws.stance_fwd_m[1] - ws.stance_fwd_m[0]) / 0.02)) + 1)]
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
                    if not self._stance_ok(grid, p, sx, sy):
                        continue
                    cost = d + 0.3 * abs(coords.ang_diff(yaw, p.yaw)) + 0.2 * abs(l)
                    if best is None or cost < best[0]:
                        dx, dy = sx - p.x, sy - p.y
                        bf, bl = coords.world_to_body((p.x, p.y, p.yaw), sx, sy)
                        best = (cost, {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(coords.wrap_pi(yaw), 4),
                                       "dx": round(dx, 3), "dy": round(dy, 3),
                                       "dyaw": round(coords.ang_diff(yaw, p.yaw), 4),
                                       "forward": round(bf, 3), "left": round(bl, 3), "distance_m": round(d, 3),
                                       "object_fwd": round(f, 3), "object_left": round(l, 3)})
        return best[1] if best else None

    def find_stance(self, obj_xy: tuple[float, float], skill: Any = None, z: float | None = None) -> dict | None:
        """A stance within approach_max_m of here from which the object is in the window (and, when the palm
        height z is given, inside the arm's sphere with stance_margin_m to spare).

        Candidates: the object at any (forward, lateral) of the reach window on a 2 x 5 cm grid, seen from any
        direction around it (every 10 deg), so a deep object can sit diagonally on the reaching arm's side (the
        sphere is centred 0.14 m out on that side: the palm gets ~0.5 m from the pelvis there, ~0.43 m straight
        ahead). A skill with its own stance (GR00T: stand_off_m, lateral_m, yaw_to_object) keeps exactly that
        offset. Cost: the walk, the turn, a lateral offset, and the distance outside the comfortable band
        stance_fwd_m (tracking sags where the arm is nearly straight)."""
        if self.ws.stance_search == "band":
            return self._find_stance_band(obj_xy, skill, z)
        p = self.world.robot_pose()
        grid = self.world.static_map().grid
        ws = self.ws
        st = dict(getattr(skill, "stance", None) or {})
        if "stand_off_m" in st:
            cands = [(float(st["stand_off_m"]), float(st.get("lateral_m", 0.0)))]
        else:
            lo, hi = ws.reach_fwd_m
            n_f = int(round((hi - lo) / 0.02))
            n_l = int(math.floor(ws.reach_lat_max_m / 0.05 + 1e-9))
            e = 1e-3                        # inside the window's edges: the check at the stance must agree
            lat_in = ws.reach_lat_max_m - e
            cands = [(min(max(lo + i * 0.02, lo + e), hi - e), max(-lat_in, min(lat_in, j * 0.05)))
                     for i in range(n_f + 1) for j in range(-n_l, n_l + 1)]
            if z is not None:
                cands = [(f, l) for f, l in cands if self.shoulder_dist(f, l, z) <= ws.arm_reach_m - ws.stance_margin_m]
        f_lo, f_hi = ws.stance_fwd_m
        yaw_obj = float(st.get("yaw_to_object", 0.0))
        best = None
        for k in range(36):
            beta = -math.pi + k * math.pi / 18            # direction stance -> object (world)
            cb, sb = math.cos(beta), math.sin(beta)
            for f, l in cands:
                r = math.hypot(f, l)
                # rounded as the stance is handed on, so navigation's check sees the point judged here (5 cm grid)
                sx, sy = round(obj_xy[0] - r * cb, 3), round(obj_xy[1] - r * sb, 3)
                d = math.hypot(sx - p.x, sy - p.y)
                if d > ws.approach_max_m + 1e-9:
                    continue
                yaw = coords.wrap_pi(beta - math.atan2(l, f) - yaw_obj)
                cost = d + 0.3 * abs(coords.ang_diff(yaw, p.yaw)) + 0.2 * abs(l) + max(0.0, f - f_hi, f_lo - f)
                if best is not None and cost >= best[0]:
                    continue
                if not self._stance_ok(grid, p, sx, sy):
                    continue
                dx, dy = sx - p.x, sy - p.y
                bf, bl = coords.world_to_body((p.x, p.y, p.yaw), sx, sy)
                best = (cost, {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(yaw, 4),
                               "dx": round(dx, 3), "dy": round(dy, 3),
                               "dyaw": round(coords.ang_diff(yaw, p.yaw), 4),
                               "forward": round(bf, 3), "left": round(bl, 3), "distance_m": round(d, 3),
                               "object_fwd": round(f, 3), "object_left": round(l, 3)})
        return best[1] if best else None

    def nearest_stand_m(self, obj_xy: tuple[float, float], r_max: float = 1.2, step: float = 0.02) -> float | None:
        """Distance from the object to the nearest spot the pelvis may stand on for a reach (stance_clearance_m of
        raw clearance, inside the house; with stance_via go_to also the planner's free space in this component),
        to `step` m; None if none within r_max."""
        grid = self.world.static_map().grid
        p = self.world.robot_pose()
        comp = grid.component(p.x, p.y) if self.ws.stance_via == "go_to" else None
        r = step
        while r <= r_max + 1e-9:
            n = max(12, int(2 * math.pi * r / 0.03))
            for k in range(n):
                a = 2 * math.pi * k / n
                sx, sy = obj_xy[0] + r * math.cos(a), obj_xy[1] + r * math.sin(a)
                if grid.clearance(sx, sy) < self.ws.stance_clearance_m or grid.is_outside(sx, sy):
                    continue
                if self.ws.stance_via == "go_to" and not (grid.is_free(sx, sy) and
                                                          (comp is None or grid.component(sx, sy) == comp)):
                    continue
                return round(r, 3)
            r += step
        return None

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
            # No candidates from belief: only what the robot perceives now (visible, the last scan here, its
            # hands) can bind an instance. Nothing perceivable -> not_found (PLAN 6.4 step 2: "no instance of that
            # type in candidates"), the same answer whether or not one exists elsewhere, with no object id, so
            # neither an unseen instance's id nor an absence (eval missing_object) leaks.
            of_type = [oid for oid, o in objs.items() if o.type == object_type]
            ordered = sorted(of_type, key=lambda c: (c not in visible, c not in seen, c))
            ordered = [c for c in ordered if c in visible or c in seen or objs[c].held_by]
            if not ordered:
                return res(reason="not_found")
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
        gz = self.grasp_z(o)                    # the palm target's height (grasp point)
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
            stance = self.find_stance((x, y), skill, gz)
            if stance is not None:
                return res(reason="needs_reposition", visible=is_vis, oid=oid, suggest_location="reach_stance",
                           stance=stance, skill_id=getattr(skill, "skill_id", None), **common)
            gap = self.nearest_stand_m((x, y))
            reach = self.horizontal_reach(gz)
            if gap is not None and gap > reach + 0.02:
                return res(reason="beyond_reach", visible=is_vis, oid=oid, detail=(
                    f"{oid} is {gap:.2f} m from the nearest spot the robot can stand; at {gz - m.floor_z:.2f} m the "
                    f"arm reaches {reach:.2f} m"), **common)
            sug = self.suggest_keypoint((x, y), z, o.where)
            return res(reason="too_far", visible=is_vis, oid=oid, suggest_location=sug if sug != at else None,
                       **common)
        if self.shoulder_dist(fwd, lat, gz) > self.ws.arm_reach_m:
            # in the window but beyond the arm from here: a nearby stance may fix it (a reposition), else no pose
            # near here can reach it
            stance = self.find_stance((x, y), skill, gz)
            if stance is not None:
                return res(reason="needs_reposition", visible=is_vis, oid=oid, suggest_location="reach_stance",
                           stance=stance, skill_id=getattr(skill, "skill_id", None), **common)
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
