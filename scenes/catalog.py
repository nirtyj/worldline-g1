"""House ids, on-disk locations and ProcTHOR room metadata. Pure Python (no Isaac import).

House id forms accepted by `parse_house_id` (canonical form first):
    procthor-train-40   | procthor-10k-train-40 | train_40 | procthor-10k-train/40
    ithor-FloorPlan10   | FloorPlan10 | FloorPlan10_physics

Index mapping (verified): MolmoSpaces exports `prior.load_dataset("procthor-10k")[split][idx]`
as `{split}_{idx}` (molmospaces molmo_spaces/housegen/exporter.py:78, 92, 498) and the USD
converter keeps the MJCF stem as the folder name (molmo_spaces_isaac/assets/house_converter.py:872-876).
Worldline's `procthor-train-40` is `prior.load_dataset("procthor-10k")["train"][40]`
(ludo-runtime thor/procthor.py:36-49). So `procthor-train-40` == MolmoSpaces `train_40`;
loader.verify_procthor_mapping() checks it by comparing the object ids of the USD metadata
with the ProcTHOR JSON copied into scenes/data/procthor/.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Pinned MolmoSpaces USD versions (molmo_spaces_isaac/downloader/main.py:17-50 @713fd12).
SOURCE_VERSIONS = {
    "objects/thor": "20260128",
    "ithor": "20260121",
    "procthor-10k-train": "20260128",
    "procthor-10k-val": "20260128",
    "procthor-10k-test": "20260128",
}

MOLMOSPACES_ROOT = Path(os.environ.get("WL_MOLMOSPACES_ROOT", "/work/assets/molmospaces"))
HOUSE_ASSETS_ROOT = Path(os.environ.get("WL_HOUSE_ASSETS", "/work/worldline-g1/assets/houses"))
DATA_DIR = Path(__file__).resolve().parent / "data"

# Houses Worldline uses on AI2-THOR (ludo-runtime thor/procthor.py:21-26) + the iTHOR kitchen of eval 17.
DEFAULT_HOUSES = [
    "procthor-train-40",
    "procthor-train-15",
    "procthor-train-38",
    "procthor-train-59",
    "ithor-FloorPlan10",
]

_PROCTHOR_RE = re.compile(r"^(?:procthor-(?:10k-)?)?(train|val|test)[-_/](\d+)$")
_PROCTHOR_RE2 = re.compile(r"^procthor-10k-(train|val|test)/(\d+)$")
_ITHOR_RE = re.compile(r"^(?:ithor[-_/])?FloorPlan(\d+)(?:_physics)?$", re.IGNORECASE)


@dataclass(frozen=True)
class HouseRef:
    house_id: str  # canonical, e.g. procthor-train-40 / ithor-FloorPlan10
    kind: str  # "procthor" | "ithor"
    source: str  # MolmoSpaces source, e.g. procthor-10k-train / ithor
    scene_dir: str  # folder inside the source, e.g. train_40 / FloorPlan10_physics
    archive: str  # archive name in the MolmoSpaces manifest
    split: str | None = None
    index: int | None = None

    @property
    def usd_dir(self) -> Path:
        return MOLMOSPACES_ROOT / "usd" / "scenes" / self.source / self.scene_dir

    @property
    def scene_usd(self) -> Path:
        return self.usd_dir / "scene.usda"

    @property
    def metadata_json(self) -> Path:
        return self.usd_dir / "scene_metadata.json"

    @property
    def assets_dir(self) -> Path:
        """Our generated per-house assets (occupancy, meta, renders)."""
        return HOUSE_ASSETS_ROOT / self.house_id


def parse_house_id(house_id: str) -> HouseRef:
    s = house_id.strip()
    m = _PROCTHOR_RE.match(s) or _PROCTHOR_RE2.match(s)
    if m:
        split, idx = m.group(1), int(m.group(2))
        source = f"procthor-10k-{split}"
        return HouseRef(
            house_id=f"procthor-{split}-{idx}",
            kind="procthor",
            source=source,
            scene_dir=f"{split}_{idx}",
            archive=f"{source}_{split}_{idx}.tar.zst",
            split=split,
            index=idx,
        )
    m = _ITHOR_RE.match(s)
    if m:
        n = int(m.group(1))
        return HouseRef(
            house_id=f"ithor-FloorPlan{n}",
            kind="ithor",
            source="ithor",
            scene_dir=f"FloorPlan{n}_physics",
            archive=f"ithor_FloorPlan{n}_physics.tar.zst",
            index=n,
        )
    raise ValueError(f"unknown house id {house_id!r} (expected e.g. procthor-train-40 or ithor-FloorPlan10)")


# --------------------------------------------------------------------------------------------
# ProcTHOR house JSON (room types + polygons). Copied from Worldline's cache
# (ludo-runtime runs/procthor/train-<i>.json, written by thor/procthor.py:36-49 from `prior`).
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=16)
def procthor_json(ref: HouseRef) -> dict | None:
    if ref.kind != "procthor":
        return None
    p = DATA_DIR / "procthor" / f"{ref.split}-{ref.index}.json"
    if p.exists():
        return json.loads(p.read_text())
    try:  # optional: fetch like Worldline does (needs `prior` + network once)
        import prior  # type: ignore

        house = dict(prior.load_dataset("procthor-10k")[ref.split][ref.index])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(house))
        return house
    except Exception:
        return None


def snake(name: str) -> str:
    """CamelCase THOR type -> snake_case (same rule as ludo-runtime thor/procthor.py:58)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def procthor_rooms(ref: HouseRef) -> list[dict]:
    """Rooms from the ProcTHOR JSON, named like Worldline (thor/procthor.py:52-67).

    Returns [{room_id:int, thor_id:'room|6', type:'Kitchen', name:'kitchen', polygon_thor:[(x,z),...]}].
    Polygons are in THOR coordinates (x, z); `thor_xz_to_world` converts to the USD/world frame.
    """
    house = procthor_json(ref)
    if not house:
        return []
    counts: dict[str, int] = {}
    for r in house.get("rooms", []):
        counts[r["roomType"]] = counts.get(r["roomType"], 0) + 1
    seen: dict[str, int] = {}
    out = []
    for r in house.get("rooms", []):
        base = snake(r["roomType"])
        seen[r["roomType"]] = seen.get(r["roomType"], 0) + 1
        name = base if counts[r["roomType"]] == 1 else f"{base}_{seen[r['roomType']]}"
        rid_s = r["id"].split("|")[-1]
        out.append(
            {
                "room_id": int(rid_s) if rid_s.isdigit() else 0,
                "thor_id": r["id"],
                "type": r["roomType"],
                "name": name,
                "polygon_thor": [(p["x"], p["z"]) for p in r["floorPolygon"]],
            }
        )
    return out


def procthor_objects(ref: HouseRef) -> dict[str, dict]:
    """THOR object id -> {assetId, position(x,y,z), type} from the ProcTHOR JSON (children included)."""
    house = procthor_json(ref)
    out: dict[str, dict] = {}
    if not house:
        return out

    def walk(objs):
        for o in objs:
            out[o["id"]] = {
                "assetId": o.get("assetId"),
                "position": o.get("position"),
                "type": o["id"].split("|")[0],
            }
            walk(o.get("children", []) or [])

    walk(house.get("objects", []))
    # doors/windows live in their own lists; their assetPosition is wall-relative, so no position
    for key in ("doors", "windows"):
        for o in house.get(key, []) or []:
            out[o["id"]] = {"assetId": o.get("assetId"), "position": None, "type": key[:-1]}
    return out


def procthor_agent_pose(ref: HouseRef) -> dict | None:
    house = procthor_json(ref)
    if not house:
        return None
    return (house.get("metadata", {}).get("agentPoses") or {}).get("default") or house.get("metadata", {}).get("agent")


def point_in_polygon(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def polygon_area(poly: list[tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0
