"""G1 workspace calibration (M3.5, R.7): where the scripted grasp can put the palm, and which household pickables that
makes reachable. Offline; the live palm check is world/live_cal.py `arm`.

    python -m world.workspace_cal envelope [--workers 8] [--out DIR]     IK envelope + sphere fit (JSON)
    python -m world.workspace_cal coverage [--houses H ...] [--out DIR]  pickables reachable from a keypoint or after
                                                                         one reach_stance, with config/g1.yaml

**Envelope.** The IK is the body's own (`body/g1_kin.py`, read-only here) with exactly `body/arm_script.py`'s
settings: the arm starts from SONIC's standing arms (`joint_map.DEFAULT_ANGLES`), wrist pitch and yaw locked at
that pose (SONIC barely tracks them, docs/arm_tracking.md §3.2), the null space pulled to it, waist at 0 (SONIC's
reference waist; waist pitch is never commanded). A grasp point (pelvis frame, x forward, y left, z up) is in the
envelope when the `grasp` goal solves to within `grasp_tol_m` (1 cm: half the body's 2 cm `ik_unreachable` bar, so
IK error and SONIC's 2-4 cm tracking error do not add up to a miss) and the `pregrasp` goal (10 cm back along the
horizontal approach, 5 cm up, arm_script defaults) to within the body's 2 cm. Sampled on a grid of grasp-point
heights above the floor (pelvis at `pelvis_z_m`) and lateral offsets on the arm's own side, bisecting the forward
reach. The boundary is fitted with a sphere (`services/reachability.py`'s shoulder model: centre forward, lateral,
height; radius) whose residuals say how well that model holds.

**Coverage.** Every pickable that starts on a map surface, in the recorded houses, through the runtime's own
`ReachabilityModel` (services/reachability.py) with the calibrated config: the robot at the object's surface
keypoint (its stand), `check_reachability`; on `needs_reposition` it moves to the suggested stance (exactly; the
body's `approach` lands within 5 cm, docs/contracts/m1.md §3.13) and checks again. Verdicts: `keypoint`,
`one_approach`, or the reason (`too_far`, `beyond_reach`, `too_low`, `too_high`, `out_of_workspace`, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HOUSES = ("procthor-train-40", "procthor-train-15", "ithor-FloorPlan10", "procthor-train-38")


# ====================================================================== IK envelope
@dataclass(frozen=True)
class EnvelopeSpec:
    arm: str = "right"
    pelvis_z_m: float = 0.79                 # standing pelvis above the floor (stand ops: 0.775-0.797 m)
    heights_m: tuple[float, ...] = tuple(round(0.60 + 0.05 * i, 2) for i in range(16))   # grasp point above floor
    lats_m: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)    # towards the arm's side
    fwd_lo_m: float = 0.10
    fwd_hi_m: float = 0.60
    fwd_step_m: float = 0.01
    grasp_tol_m: float = 0.01
    pregrasp_tol_m: float = 0.02
    standoff_m: float = 0.10                 # arm_script pregrasp defaults
    above_m: float = 0.05


def _ik_setup(arm: str):
    from body import g1_kin as K
    from body import joint_map as jm
    q = dict(zip(jm.MUJOCO_JOINTS, jm.DEFAULT_ANGLES))
    seed = {n: float(q[n]) for n in jm.UPPER_BODY_MUJOCO_JOINTS}
    lock = tuple(f"{arm}_{j}_joint" for j in ("wrist_pitch", "wrist_yaw"))
    return K, seed, lock


def ik_error(arm: str, p_b: Sequence[float]) -> float:
    """The palm IK error (m) for a pelvis-frame goal, with arm_script's IK settings."""
    K, seed, lock = _ik_setup(arm)
    return float(K.ik_palm(arm, [float(v) for v in p_b], seed, q_rest=seed, lock=lock)[1])


def grasp_ok(spec: EnvelopeSpec, fwd: float, lat: float, z: float) -> tuple[bool, float, float | None]:
    """(in the envelope, grasp IK error, pregrasp IK error) for a grasp point in the pelvis frame; lat is signed
    (left +y), so the right arm's own side is lat < 0."""
    eg = ik_error(spec.arm, (fwd, lat, z))
    if eg > spec.grasp_tol_m:
        return False, eg, None
    d = np.array([fwd, lat], float)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-9 else np.array([1.0, 0.0])
    ep = ik_error(spec.arm, (fwd - spec.standoff_m * d[0], lat - spec.standoff_m * d[1], z + spec.above_m))
    return ep <= spec.pregrasp_tol_m, eg, ep


def _row(args: tuple[EnvelopeSpec, float, float]) -> dict:
    spec, h, lat_mag = args
    side = -1.0 if spec.arm == "right" else 1.0
    lat = side * lat_mag
    z = h - spec.pelvis_z_m
    fwds = np.arange(spec.fwd_lo_m, spec.fwd_hi_m + 1e-9, spec.fwd_step_m)
    ok = [bool(grasp_ok(spec, float(f), lat, z)[0]) for f in fwds]
    feas = [float(f) for f, k in zip(fwds, ok) if k]
    # the reach is the far end of the feasible run that contains the nearest feasible point
    far = None
    if feas:
        i0 = ok.index(True)
        i = i0
        while i + 1 < len(ok) and ok[i + 1]:
            i += 1
        far = round(float(fwds[i]), 3)
    return {"h": h, "lat": round(lat, 3), "fwd_min": round(min(feas), 3) if feas else None,
            "fwd_max": far, "n_feasible": len(feas)}


def sample_envelope(spec: EnvelopeSpec | None = None, workers: int = 1) -> dict:
    spec = spec or EnvelopeSpec()
    jobs = [(spec, h, l) for h in spec.heights_m for l in spec.lats_m]
    t0 = time.monotonic()
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_row, jobs)
    else:
        rows = [_row(j) for j in jobs]
    return {"spec": asdict(spec), "rows": rows, "s": round(time.monotonic() - t0, 1)}


def boundary_points(env: dict) -> np.ndarray:
    """(fwd_max, lat, z) of every sampled row with a reach, pelvis frame."""
    pz = env["spec"]["pelvis_z_m"]
    pts = [(r["fwd_max"], r["lat"], r["h"] - pz) for r in env["rows"] if r["fwd_max"] is not None]
    return np.asarray(pts, float)


def fit_sphere(pts: np.ndarray, outlier_k: float = 4.0) -> dict:
    """Least-squares sphere |p - c| = r through the boundary points, robust to the IK's odd local minimum (a split
    feasible run shortens one row's reach): fit, drop points whose residual exceeds `outlier_k` x the median absolute
    residual (and 2 cm), refit. Returns centre, radius, residual statistics (m) and the dropped points."""
    fit = _fit_sphere(pts)
    res = np.abs(np.linalg.norm(pts - np.array(fit["centre_b"]), axis=1) - fit["radius_m"])
    keep = res <= max(0.02, outlier_k * float(np.median(res)))
    if keep.all() or keep.sum() < 8:
        return {**fit, "dropped": []}
    out = _fit_sphere(pts[keep])
    return {**out, "dropped": [[round(float(v), 3) for v in p] for p in pts[~keep]]}


def _fit_sphere(pts: np.ndarray) -> dict:
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(max(1e-9, sol[3] + float(c @ c)))
    for _ in range(50):
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        res = dist - r
        J = np.c_[-d / dist[:, None], -np.ones(len(pts))]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c, r = c + step[:3], r + float(step[3])
        if float(np.abs(step).max()) < 1e-7:
            break
    res = np.linalg.norm(pts - c, axis=1) - r
    return {"centre_b": [round(float(v), 4) for v in c], "radius_m": round(float(r), 4),
            "residual_m": {"rms": round(float(np.sqrt((res ** 2).mean())), 4),
                           "p90_abs": round(float(np.percentile(np.abs(res), 90)), 4),
                           "max_abs": round(float(np.abs(res).max()), 4)}, "n": int(len(pts))}


def envelope_summary(env: dict) -> dict:
    """Per grasp height: the best forward reach and the lateral range with a reach of at least 0.25 m."""
    by_h: dict[float, list[dict]] = {}
    for r in env["rows"]:
        by_h.setdefault(r["h"], []).append(r)
    out = []
    for h in sorted(by_h):
        rows = [r for r in by_h[h] if r["fwd_max"] is not None]
        best = max((r["fwd_max"] for r in rows), default=None)
        lat_ok = [abs(r["lat"]) for r in rows if r["fwd_max"] >= 0.25]
        out.append({"h": h, "fwd_max_best": best, "lat_max_at_0.25": max(lat_ok) if lat_ok else None,
                    "fwd_max_by_lat": {str(abs(r["lat"])): r["fwd_max"] for r in by_h[h]}})
    return {"by_height": out}


# ====================================================================== coverage in the houses
class _SeenAll:
    """ObservationService stand-in for the coverage run: the robot has looked here (no not_seen_here)."""

    def __init__(self, world):
        self.world = world

    def seen_here(self, keypoint):
        return set(self.world.objects())


class _At:
    """NavigationService stand-in: `at()` is the keypoint the robot was put at."""

    def __init__(self):
        self.kp: str | None = None
        self.moving = False

    def at(self):
        return self.kp, None


@dataclass
class CoverageRow:
    house: str
    object_id: str
    type: str
    surface: str
    height_m: float                 # centre above the floor
    verdict: str                    # keypoint | one_approach | other_stand | <reason>
    gap_m: float | None = None      # object -> nearest spot the pelvis may stand (stance_clearance_m from obstacles)
    reach_m: float = 0.0            # the arm's largest horizontal reach at the object's grasp height
    first: dict = field(default_factory=dict)
    second: dict | None = None


def _res(r) -> dict:
    return {k: v for k, v in {"reachable": r.reachable, "reason": r.reason, "arm": r.preferred_arm,
                              "distance_m": getattr(r, "distance_m", None),
                              "suggest": getattr(r, "suggest_location", None),
                              "stance": getattr(r, "stance", None)}.items() if v is not None}


def coverage(house: str, *, profile: str = "sonic", config_dir: str | Path | None = None) -> list[CoverageRow]:
    from robot.profile import load_profile
    from services.reachability import G1Workspace, ReachabilityModel
    from services.skills import build_registry
    from world.lite_world import LiteWorld
    from world.mapgen import MapParams

    prof = load_profile(profile, config_dir=config_dir)
    # the Isaac map (R.7 stands), not the lite world's INTERIM M2a stands (mapgen.lite_world)
    mp = MapParams.from_dict({k: v for k, v in (prof.g1.get("mapgen") or {}).items() if k != "lite_world"})
    w = LiteWorld(house, map_params=mp, user_surface=prof.scene_config(house).get("user_surface"))
    reg = build_registry(["lite"], w)
    nav = _At()
    # the calibrated G1 arm, not the lite world's INTERIM one (workspace.lite_world): this is about the robot
    ws = {k: v for k, v in (prof.g1.get("workspace") or {}).items() if k != "lite_world"}
    model = ReachabilityModel(w, G1Workspace.from_dict(ws), registry=reg, observation=_SeenAll(w), nav=nav)
    m = w.static_map()
    rows = []
    for oid, o in sorted(w.objects().items()):
        if o.where not in m.surfaces:
            continue
        h = round(o.pos[2] - m.floor_z, 3)

        def at(kp: str, pose=None):
            k = m.keypoints[kp]
            nav.kp = kp
            if pose is None:
                w.set_robot_pose(k.x, k.y, k.yaw)
            else:
                w.set_robot_pose(*pose)
            return model.check(o.type, oid, at=kp)

        r1 = at(o.where)
        row = CoverageRow(house, oid, o.type, o.where, h, "", first=_res(r1), gap_m=model.nearest_stand_m(o.pos[:2]),
                          reach_m=round(model.horizontal_reach(model.grasp_z(o)), 3))
        if r1.reachable:
            row.verdict = "keypoint"
        elif r1.reason == "needs_reposition" and r1.stance:
            st = r1.stance
            r2 = at(o.where, (st["x"], st["y"], st["yaw"]))
            row.second = _res(r2)
            row.verdict = "one_approach" if r2.reachable else f"after_approach:{r2.reason}"
        elif r1.reason == "too_far" and r1.suggest_location and r1.suggest_location in m.keypoints:
            r2 = at(r1.suggest_location)
            if r2.reason == "needs_reposition" and r2.stance:
                st = r2.stance
                r2 = at(r1.suggest_location, (st["x"], st["y"], st["yaw"]))
            row.second = _res(r2)
            row.verdict = "other_stand" if r2.reachable else f"too_far:{r2.reason}"
        else:
            row.verdict = str(r1.reason)
        rows.append(row)
    return rows


def coverage_summary(rows: list[CoverageRow], ws: dict) -> dict:
    zmin, zmax = float(ws.get("obj_z_min_m", 0.0)), float(ws.get("obj_z_max_m", 9.0))
    band = [r for r in rows if zmin <= r.height_m <= zmax]
    ok = ("keypoint", "one_approach")

    def frac(rs, verdicts):
        return round(sum(r.verdict in verdicts for r in rs) / len(rs), 3) if rs else None

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    # the upper bound any stand / stance scheme could reach: some spot the pelvis may stand is within the arm's
    # horizontal reach of the object
    phys = [r for r in band if r.gap_m is not None and r.gap_m <= r.reach_m]
    return {"pickables_on_surfaces": len(rows), "in_height_band": len(band),
            "physically_reachable_in_band": len(phys),
            "frac_in_band_upper_bound": round(len(phys) / len(band), 3) if band else None,
            "reachable_keypoint_or_one_approach": sum(r.verdict in ok for r in rows),
            "frac_all": frac(rows, ok), "frac_in_band": frac(band, ok),
            "frac_in_band_incl_other_stand": frac(band, ok + ("other_stand",)),
            "verdicts": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


# ====================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("envelope")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--arm", default="right")
    e.add_argument("--pelvis-z", type=float, default=EnvelopeSpec.pelvis_z_m)
    e.add_argument("--out", default=None)
    c = sub.add_parser("coverage")
    c.add_argument("--houses", nargs="*", default=list(HOUSES))
    c.add_argument("--profile", default="sonic")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "envelope":
        env = sample_envelope(EnvelopeSpec(arm=a.arm, pelvis_z_m=a.pelvis_z), workers=a.workers)
        env["sphere"] = fit_sphere(boundary_points(env))
        env["summary"] = envelope_summary(env)
        for r in env["summary"]["by_height"]:
            print(f"h {r['h']:.2f}  fwd_max {r['fwd_max_best']}  |lat| with >= 0.25 m: {r['lat_max_at_0.25']}  "
                  + " ".join(f"{k}:{v}" for k, v in r["fwd_max_by_lat"].items()))
        print("sphere", json.dumps(env["sphere"]), f"({env['s']} s)")
        if out:
            (out / f"envelope_{a.arm}.json").write_text(json.dumps(env, indent=1) + "\n")
        return 0
    from robot.profile import load_profile
    ws = load_profile(a.profile).g1.get("workspace") or {}
    report: dict[str, Any] = {"profile": a.profile, "workspace": ws, "houses": {}}
    allrows: list[CoverageRow] = []
    for h in a.houses:
        rows = coverage(h, profile=a.profile)
        allrows += rows
        report["houses"][h] = {"summary": coverage_summary(rows, ws), "rows": [asdict(r) for r in rows]}
        print(h, json.dumps(report["houses"][h]["summary"]))
    report["summary"] = coverage_summary(allrows, ws)
    print("ALL", json.dumps(report["summary"]))
    if out:
        (out / "coverage.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
