"""The eval's bindings on the G1 stack: eval/scenes.yaml plus the rules that use it (PLAN 9.1).

  load()                      the houses, profiles and 17 scenario bindings
  Bindings.time_limit()       a scenario's wall-clock limit on a profile (THOR limit x time scale, never
                              below twice the humanoid estimate from walk distance, scans, picks and places)
  executors_used()            which executors and skills produced the results (trace rows + executions)
  honesty()                   labels for a pass that rests on a sim shortcut (attach grasp, kinematic base)
  check_fixtures()            the house is as scenes.yaml says at scenario start (asserted on world truth)
  other_side()                resolves the other_side scenario from the live map and truth

Everything here reads the page's messages (init.layout, init.map, frame.truth), whose ground truth comes from
world/ (world.truth(), world.static_map()); nothing here imports the simulator, world/ or zmq.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

HERE = Path(__file__).resolve().parent
SCENES_YAML = HERE / "scenes.yaml"
STEPPING_STONES = frozenset({"kinematic_nav", "kinematic_attach", "sonic_arm_script"})
# The target executors (PLAN 2.1): a pass counts as a target pass only if every body result came from these.
# One definition, api.results.TARGET_EXECUTORS; the literal is the fallback when api/ is not importable.
try:
    from api.results import TARGET_EXECUTORS as _API_TARGETS  # type: ignore
    TARGET_EXECUTORS = frozenset(_API_TARGETS)
except Exception:  # noqa: BLE001
    TARGET_EXECUTORS = frozenset({"sonic_walk", "groot_sonic"})
SHORTCUT_LABELS = {
    "kinematic_nav": "kinematic base (teleported along the nav path; no gait)",
    "kinematic_attach": "attach grasp (kinematic; no arm motion)",
    "sonic_arm_script": "attach grasp (SONIC arm-script reach + ground-truth attach)",
    "lite": "lite body (pure Python; no physics)",
}
NAV_TOOLS = ("navigate",)
MANIP_TOOLS = ("manipulate",)


def stepping_stones() -> frozenset[str]:
    try:
        from api.results import STEPPING_STONE_EXECUTORS  # type: ignore
        return frozenset(STEPPING_STONE_EXECUTORS) | STEPPING_STONES
    except Exception:  # noqa: BLE001
        return STEPPING_STONES


@dataclass
class Bindings:
    data: dict[str, Any]

    @property
    def houses(self) -> dict[str, dict[str, Any]]:
        return self.data["houses"]

    @property
    def scenarios(self) -> dict[str, dict[str, Any]]:
        return self.data["scenarios"]

    def house(self, key: str) -> dict[str, Any]:
        return self.houses[key]

    def scene(self, key: str) -> str:
        return self.houses[key]["scene"]

    def scenario(self, name: str) -> dict[str, Any]:
        return self.scenarios[name]

    def house_of(self, name: str) -> dict[str, Any]:
        return self.houses[self.scenarios[name]["house"]]

    # ------------------------------------------------------------------ profiles and time
    def profile(self, name: str) -> dict[str, float]:
        p = dict(self.data["profiles"].get(name) or self.data["profiles"]["lite"])
        try:                                                  # api/ is the one source of the numbers
            from api.types import PROFILES  # type: ignore
            rp = PROFILES.get(name)
            if rp is not None:
                p.update(walk_mps=rp.walk_speed_mps, t_pick_s=rp.t_pick_s, t_place_s=rp.t_place_s)
                if getattr(rp, "time_scale", 1.0) not in (None, 1.0):
                    p["time_scale"] = rp.time_scale
        except Exception:  # noqa: BLE001
            pass
        return p

    def walk_m(self, house: dict[str, Any], legs: Iterable[str]) -> float:
        table = house.get("walk_m") or {}
        total = 0.0
        for leg in legs:
            a, _, b = leg.partition(">")
            d = table.get(f"{a}>{b}", table.get(f"{b}>{a}"))
            if d is None:
                raise KeyError(f"no walk distance for leg {leg!r} in {house.get('scene')}")
            total += float(d)
        return total

    def humanoid_estimate(self, name: str, profile: str) -> float:
        sc, est = self.scenarios[name], self.data["estimate"]
        p = self.profile(profile)
        legs = sc.get("legs") or []
        work = sc.get("work") or {}
        walk = self.walk_m(self.house_of(name), legs)
        return (est["planner_s"] + 1.25 * walk / max(p["walk_mps"], 0.05) + len(legs) * est["stop_s"]
                + work.get("scans", 0) * est["scan_s"] + work.get("picks", 0) * (p["t_pick_s"] + est["check_s"])
                + work.get("places", 0) * p["t_place_s"])

    def time_limit(self, name: str, profile: str, scale: float | None = None) -> tuple[float, dict[str, float]]:
        """(limit_s, how): max(THOR limit x scale, 2 x humanoid estimate) for body scenarios; talk-only
        scenarios keep THOR's limit (the planner and speech run at the same speed on every profile)."""
        sc = self.scenarios[name]
        base = float(sc["thor_limit_s"])
        s = float(scale if scale is not None else self.profile(profile)["time_scale"])
        if not sc.get("legs"):
            return base, {"thor_s": base, "scale": 1.0, "estimate_s": 0.0}
        est = self.humanoid_estimate(name, profile)
        limit = max(base * s, 2.0 * est)
        return round(limit, 1), {"thor_s": base, "scale": s, "estimate_s": round(est, 1)}

    def wait_scale(self, profile: str, scale: float | None = None) -> float:
        """For the short body-dependent waits inside a script (e.g. 'until the first navigate starts')."""
        return float(scale if scale is not None else self.profile(profile)["time_scale"])

    # ------------------------------------------------------------------ what a scenario says and waits for
    def second_request(self, name: str, original: bool = False) -> tuple[str | None, list[str]]:
        sc = self.scenarios[name]
        use = "original" if original else sc.get("use", "original")
        part = sc.get(use) or sc.get("original") or {}
        return part.get("then", sc.get("then")), list(part.get("second") or [])


def load(path: str | Path | None = None) -> Bindings:
    data = yaml.safe_load(Path(path or SCENES_YAML).read_text())
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("eval/scenes.yaml: expected version: 1")
    return Bindings(data)


# ----------------------------------------------------------------------------------------------
# executors and honesty labels
# ----------------------------------------------------------------------------------------------
def _executions(frame: dict[str, Any] | None) -> list[dict[str, Any]]:
    rt = (frame or {}).get("runtime") or {}
    return list(rt.get("executions") or []) + list(rt.get("recent") or [])


def executors_used(trace: list[dict[str, Any]], frames: Iterable[dict[str, Any]] = ()) -> dict[str, dict[str, int]]:
    """{nav: {executor: n}, manip: {executor: n}, skills: {skill: n}} over the results of this scenario.

    Result rows carry `executor` (and `data.skill` for manipulate); executions seen in frames fill in any
    result row the trace window missed (keyed by execution_id so nothing is counted twice)."""
    seen: dict[str, tuple[str, str | None, str | None]] = {}
    for r in trace:
        if r.get("type") != "result":
            continue
        tool = r.get("tool") or r.get("skill")
        d = r.get("data") or {}
        ex = r.get("executor") or d.get("executor")
        key = r.get("execution_id") or f"row{id(r)}"
        seen[key] = (tool, ex, d.get("skill"))
    for f in frames:
        for e in _executions(f):
            if e.get("id") in seen or str(e.get("status", "")).lower() in ("queued", "running", "cancelling"):
                continue
            seen[e["id"]] = (e.get("tool"), e.get("executor"), e.get("skill") or (e.get("data") or {}).get("skill"))
    out: dict[str, dict[str, int]] = {"nav": {}, "manip": {}, "skills": {}}
    for tool, ex, skill in seen.values():
        if not ex and not skill:
            continue
        group = "nav" if tool in NAV_TOOLS else "manip" if tool in MANIP_TOOLS else None
        if group and ex:
            out[group][ex] = out[group].get(ex, 0) + 1
        if skill:
            out["skills"][skill] = out["skills"].get(skill, 0) + 1
    return out


def honesty(used: dict[str, dict[str, int]], *, grasp: bool = False, profile: str = "") -> dict[str, Any]:
    """Which sim shortcuts a result rests on (PLAN 12.2). A pass with any of them is a fallback pass,
    never a target pass. If a grasp scenario reports no manipulation executor at all, the profile's
    documented grasp executor is assumed (bringup: kinematic attach; sonic: arm script + attach)."""
    stones = stepping_stones() | {"lite"}
    used_stones = sorted({ex for g in ("nav", "manip") for ex in (used.get(g) or {})
                          if ex in stones or ex not in TARGET_EXECUTORS})
    if grasp and not used.get("manip"):
        assumed = {"bringup": "kinematic_attach", "sonic": "sonic_arm_script", "lite": "lite"}.get(profile)
        if assumed:
            used_stones = sorted(set(used_stones) | {assumed})
    return {"shortcuts": used_stones, "labels": [SHORTCUT_LABELS.get(s, s) for s in used_stones],
            "fallback": bool(used_stones)}


# ----------------------------------------------------------------------------------------------
# fixtures, asserted on world truth at scenario start
# ----------------------------------------------------------------------------------------------
def types_present(truth: dict[str, Any], layout: dict[str, Any] | None = None) -> set[str]:
    """Every type in the house (truth objects + world.truth().things, decor included), as words:
    'alarm_clock' and 'AlarmClock' both give 'alarm clock'."""
    import re
    raw = [str(v.get("type") or "") for v in ((truth or {}).get("objects") or {}).values()]
    raw += list((layout or {}).get("things") or [])
    raw += [str(v.get("type") or "") for v in ((layout or {}).get("landmarks") or {}).values()]
    out: set[str] = set()
    for t in raw:
        w = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t).replace("_", " ").lower().strip()
        if not w:
            continue
        out |= {w, w.split()[-1], w.replace(" ", "")}
    extra = {"phone": "cell phone", "keys": "key chain", "remote": "remote control", "sponge": "dish sponge",
             "tv": "television", "bin": "garbage can", "plant": "house plant", "stove": "stove burner"}
    out |= {k for k, v in extra.items() if v in out}
    return out


def check_fixtures(bind: Bindings, house_key: str, layout: dict[str, Any], truth: dict[str, Any],
                   need: Iterable[str] = ()) -> tuple[bool, list[str]]:
    """(ok, notes). Hard: every object the scenario needs exists; absent types are absent. Soft (noted, not
    failing): an object is not where scenes.yaml says, or the world's user surface differs from the binding."""
    h = bind.house(house_key)
    objs = (truth or {}).get("objects") or {}
    notes: list[str] = []
    ok = True
    for oid in need:
        if oid not in objs:
            ok = False
            notes.append(f"{oid} is not in the house")
    present = types_present(truth, layout)
    for t in h.get("absent_types") or []:
        if t.replace("_", " ") in present:
            ok = False
            notes.append(f"{t} should be absent but the house has one")
    for oid in need:
        want = ((h.get("objects") or {}).get(oid) or {}).get("surface")
        got = (objs.get(oid) or {}).get("where")
        if want and got and got != want:
            notes.append(f"{oid} starts on {got}, scenes.yaml says {want}")
    us = (layout or {}).get("user_surface")
    if us and h.get("user_surface") and us != h["user_surface"]:
        notes.append(f"user surface is {us}, scenes.yaml says {h['user_surface']}")
    return ok, notes


# ----------------------------------------------------------------------------------------------
# other_side: which surface is on the other side of the landmark, from the live map
# ----------------------------------------------------------------------------------------------
def other_side(layout: dict[str, Any], landmark_xz: tuple[float, float], start_surface: str,
               max_off_line_m: float = 0.6) -> str | None:
    """The nearest surface past the landmark, seen from the start surface: the landmark must project between
    the two surface centres, within `max_off_line_m` of the line joining them."""
    surfs = (layout or {}).get("surfaces") or {}
    if start_surface not in surfs:
        return None
    s = surfs[start_surface]
    ax, az = float(s["x"]), float(s["z"])
    lx, lz = landmark_xz
    best, bd = None, math.inf
    for name, t in surfs.items():
        if name == start_surface:
            continue
        bx, bz = float(t["x"]), float(t["z"])
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        if L2 < 1e-6:
            continue
        u = ((lx - ax) * dx + (lz - az) * dz) / L2
        if not 0.0 < u < 1.0:
            continue
        off = abs((lx - ax) * dz - (lz - az) * dx) / math.sqrt(L2)
        if off > max_off_line_m:
            continue
        d = math.sqrt(L2)
        if d < bd:
            best, bd = name, d
    return best


def golden_lines(landmark: str, a: str, b: str, label: str = "stove") -> tuple[str, str]:
    return (f"{landmark} ({label}) is between {a} and {b}", f"{landmark} ({label}) is between {b} and {a}")
