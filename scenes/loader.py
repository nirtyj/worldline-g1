"""Load an AllenAI MolmoSpaces house (ProcTHOR-10K / iTHOR, USD for Isaac Sim 5.1) into a stage.

    from scenes.loader import load_house
    info = load_house(sim_or_stage, "procthor-train-40", root="/World/House")

Works in a standalone Isaac Sim 5.1 app (isaacsim.SimulationApp + isaacsim.core.api) and in
Isaac Lab (pass the SimulationContext or its stage). Everything that needs Isaac (pxr, omni.*)
is imported inside functions, so `import scenes.loader` is safe before the app starts.

What it does (each step cites the upstream code it relies on):
  1. References <molmospaces>/usd/scenes/<source>/<scene>/scene.usda at `root` and loads its
     payload. The file is opened through the per-file symlink tree so the scene's relative
     references `../../../../objects/thor/...` resolve (molmospaces_resources behaviors.py
     LinkStrategy.PER_FILE docstring; Geometry.usda references).
  2. Collision groups: MolmoSpaces puts static structure in `structural_cls_group` and moving
     parts of articulated furniture in `articulable_dynamic_cls_group`, and the structural group
     *filters* both (house_converter.py:809-838). Isaac Lab's InteractiveScene.filter_collisions()
     (isaaclab/scene/interactive_scene.py:214-215, always on CPU physics) calls the cloner, which
     sets physxScene:invertCollisionGroupFilter = True (isaacsim.core.cloner cloner.py:446). With
     the inversion the structural group collides ONLY with those two groups, so the robot falls
     through the floor and walks through walls. `load_house` (and `fix_collision_filter`, to be
     called again after any InteractiveScene is built) resets the flag to False and reports it.
     With a plain SimulationContext nothing inverts it. See docs/scenes.md.
  3. Walkability: the only floor collider is the invisible (purpose=guide) plane `Geometry/floor`
     at z=0 (house_converter.py:790-806; PhysX treats it as infinite); room floor meshes are
     visual only (house_converter.py:496-503 applies colliders to non-visual geoms only). It gets
     the same physics material SONIC trained on: static=dynamic friction 1.0, combine multiply
     (GR00T-WholeBodyControl gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:320-330).
     Walls with doorways are split into several convex colliders (collision_0..2), so convexHull
     approximation does not close doorways (verified by the occupancy connectivity test).
     Rug/mat categories (none in ProcTHOR-10K) are made non-colliding.
  4. Semantic labels from the per-object metadata (`scene_metadata.json`: object_id, category,
     is_static, room_id; molmo_spaces_isaac/assets/utils/data.py:104 MetadataObjInfo), applied with
     isaacsim.core.utils.semantics.add_labels(prim, [label], instance_name="class") (Isaac Sim
     5.1 semantics.py:218). Labels are snake_case THOR types ("alarm_clock"), plus "wall",
     "floor". iTHOR archives ship no metadata: categories come from the referenced THOR asset id
     (molmo_spaces_isaac resources/asset_id_to_object_type.json) or the prim-name lemma.
  5. Rooms: ProcTHOR room types + floor polygons from the ProcTHOR JSON (scenes/data/procthor),
     THOR (x, z) -> world (x, y) (molmo_spaces/housegen/utils.py:107-116 unity_to_mj_pos).
  6. Object AABBs from visual geometry via isaacsim.core.utils.bounds (create_bbox_cache /
     compute_aabb; bounds.py:99 = default purpose only); all MolmoSpaces colliders are
     purpose=guide (measured: 1893/1893 in train_40), so AABBs are visual extents.
  7. Physics cost: `lock_joints` (default) holds doors/drawers at their rest pose, loose props get
     angular damping + a higher sleep threshold, and `sleep_house()` (call after the sim has
     started and settled) puts the house to sleep until touched. Measured house-only step with
     PhysX numThreads=0: 0.33-0.36 ms on all four ProcTHOR houses (docs/scenes.md §7).
     `dynamic_objects="kinematic"` exists but measured slower (0.58 -> 2.70 ms/step on train-59;
     cause not investigated).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scenes.catalog import (
    HouseRef,
    parse_house_id,
    polygon_area,
    point_in_polygon,
    procthor_agent_pose,
    procthor_objects,
    procthor_rooms,
    snake,
)

RUG_CATEGORIES = {"Rug", "Mat", "FloorMat", "BathMat", "Carpet"}
STRUCTURAL_CATEGORIES = {"Wall", "Floor", "Structure", "Decal"}

# iTHOR custom (non-asset) geometry: prim-name lemma -> THOR-like type. Only used for iTHOR scenes,
# which carry no scene_metadata.json in the MolmoSpaces 20260121 archive.
ITHOR_LEMMA_TO_TYPE = {
    "cabinet": "Cabinet",
    "drawer": "Drawer",
    "knob": "StoveKnob",
    "oven": "Stove",
    "dishwasher": "Dishwasher",
    "tap": "Faucet",
    "faucet": "Faucet",
    "sink": "Sink",
    "window": "Window",
    "windowstructure": "Window",
    "floor": "Floor",
    "standardcounterheightwidth": "CounterTop",
    "standardislandheight": "CounterTop",
    "standarduppercabinetheightwidth": "Cabinet",
    "standarddoor": "Door",
    "standarddoorframe": "Doorframe",
    "standardknob": "DoorKnob",
    "standardwallsize": "Wall",
    "mesh": "Structure",
    "quaddecalspawnplane": "Decal",
    # WordNet-ish lemmas used by MolmoSpaces for THOR assets
    "refrigerator": "Fridge",
    "microwaveoven": "Microwave",
    "Irishpotato": "Potato",
    "cellulartelephone": "CellPhone",
    "cookingpan": "Pan",
    "trashcan": "GarbageCan",
    "ashcan": "GarbageCan",
    "papertowel": "PaperTowelRoll",
    "soapdispenser": "SoapBottle",
    "coffeemaker": "CoffeeMachine",
    "sponge": "DishSponge",
}

ITHOR_ROOM_TYPES = [(1, 30, "Kitchen"), (201, 230, "LivingRoom"), (301, 330, "Bedroom"), (401, 430, "Bathroom")]


# ----------------------------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------------------------


@dataclass
class RoomInfo:
    room_id: int
    name: str  # Worldline naming: kitchen, living_room, bedroom_2 ...
    type: str  # ProcTHOR roomType: Kitchen, LivingRoom, Bedroom, Bathroom
    polygon: list  # [[x, y], ...] world
    area_m2: float
    center: list  # [x, y] vertex mean (as ludo-runtime thor/procthor.py:62-64)

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(self.polygon, x, y)


@dataclass
class ObjectInfo:
    id: str  # THOR object id, e.g. "Chair|6|3|1" (ProcTHOR); synthesized for iTHOR
    name: str  # unique snake name: snake(category)_n numbered in sorted-id order
    category: str  # THOR type, e.g. "AlarmClock"
    label: str  # semantic class label applied = snake(category)
    room_id: int | None
    room: str | None
    prim_path: str  # object root Xform
    body_path: str | None  # prim with RigidBodyAPI (track this for live pose of loose props)
    pos: list  # root Xform world translation at load
    aabb: list  # [[xmin, ymin, zmin], [xmax, ymax, zmax]] world, visual geometry
    is_static: bool
    articulated: bool
    asset_id: str
    extra_prims: list = field(default_factory=list)  # iTHOR multi-body objects


@dataclass
class HouseInfo:
    house_id: str
    kind: str
    source: str
    usd_path: str
    root: str
    floor_z: float
    bounds_xy: list  # [[xmin, ymin], [xmax, ymax]]
    rooms: list  # [RoomInfo]
    objects: list  # [ObjectInfo]
    spawn: dict | None = None  # {x, y, yaw, clearance_m, forward_free_m, room, source}
    room_points: list = field(default_factory=list)  # per-room max-clearance go_to targets
    connectivity: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    assets_dir: Path | None = None

    # -- convenience / P1 contract (docs/contracts/m1.md §1.8) --
    @property
    def bounds(self) -> list:
        (x0, y0), (x1, y1) = self.bounds_xy
        return [x0, y0, x1, y1]

    @property
    def occupancy_npz(self) -> str | None:
        p = self.assets_dir / "occupancy.npz" if self.assets_dir else None
        return str(p) if p is not None and p.exists() else None

    def to_scene_info(self) -> dict:
        """The full REP `get_scene_info` reply body (contract §1.6), plus extra keys."""
        return {
            "house_id": self.house_id,
            "floor_z": self.floor_z,
            "bounds": self.bounds,
            "rooms": [{"id": r.room_id, "name": r.name, "type": r.type, "polygon": r.polygon, "area_m2": r.area_m2, "center": r.center} for r in self.rooms],
            "objects": [
                {"id": o.id, "name": o.name, "category": o.category, "label": o.label, "room_id": o.room_id, "room": o.room,
                 "pos": o.pos, "aabb": o.aabb, "is_static": o.is_static, "articulated": o.articulated, "prim_path": o.prim_path, "body_path": o.body_path}
                for o in self.objects
            ],
            "spawn": {k: self.spawn[k] for k in ("x", "y", "yaw")} if self.spawn else None,
            "spawn_detail": self.spawn,
            "room_points": self.room_points,
            "occupancy_npz": self.occupancy_npz,
            "source": {"usd": self.usd_path, "kind": self.kind, "molmospaces_source": self.source},
        }

    def room_at(self, x: float, y: float) -> str | None:
        for r in self.rooms:
            if r.contains(x, y):
                return r.name
        return None

    def room_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.rooms]

    def objects_of(self, category: str) -> list[ObjectInfo]:
        return [o for o in self.objects if o.category == category or o.label == category]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assets_dir"] = str(self.assets_dir) if self.assets_dir else None
        return d

    def save_json(self, path: Path | None = None) -> Path:
        path = path or (self.assets_dir / "house_info.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path


def load_house_info(house_id: str) -> HouseInfo:
    """Read the cached house_info.json written by scenes.test_house (no Isaac needed)."""
    ref = parse_house_id(house_id)
    d = json.loads((ref.assets_dir / "house_info.json").read_text())
    d["rooms"] = [RoomInfo(**r) for r in d["rooms"]]
    d["objects"] = [ObjectInfo(**o) for o in d["objects"]]
    d["assets_dir"] = ref.assets_dir
    return HouseInfo(**d)


# ----------------------------------------------------------------------------------------------
# Isaac helpers
# ----------------------------------------------------------------------------------------------


def _resolve_stage(stage_or_sim):
    from pxr import Usd

    if stage_or_sim is None:
        import omni.usd

        return omni.usd.get_context().get_stage()
    if isinstance(stage_or_sim, Usd.Stage):
        return stage_or_sim
    st = getattr(stage_or_sim, "stage", None)
    if isinstance(st, Usd.Stage):
        return st
    import omni.usd

    return omni.usd.get_context().get_stage()


def fix_collision_filter(stage=None, invert: bool = False) -> list[str]:
    """Force physxScene:invertCollisionGroupFilter = `invert` (False) on every PhysicsScene.
    Returns the scenes that were changed. Call after building an Isaac Lab InteractiveScene."""
    from pxr import PhysxSchema, UsdPhysics

    stage = _resolve_stage(stage)
    changed = []
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Scene):
            continue
        api = PhysxSchema.PhysxSceneAPI.Apply(p)
        attr = api.GetInvertCollisionGroupFilterAttr()
        cur = attr.Get() if attr and attr.HasAuthoredValue() else False
        if bool(cur) != invert:
            api.CreateInvertCollisionGroupFilterAttr().Set(invert)
            changed.append(str(p.GetPath()))
    return changed


def collision_filter_report(stage=None) -> dict:
    from pxr import PhysxSchema, UsdPhysics

    stage = _resolve_stage(stage)
    out = {}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            attr = PhysxSchema.PhysxSceneAPI(p).GetInvertCollisionGroupFilterAttr()
            out[str(p.GetPath())] = bool(attr.Get()) if attr and attr.HasAuthoredValue() else False
    return out


def _apply_label(prim, label: str) -> str:
    try:
        from isaacsim.core.utils.semantics import add_labels

        add_labels(prim, labels=[label], instance_name="class")
        return "add_labels"
    except Exception:
        from pxr import UsdSemantics

        api = UsdSemantics.LabelsAPI.Apply(prim, "class")
        api.CreateLabelsAttr().Set([label])
        return "UsdSemantics.LabelsAPI"


def _floor_material(stage, root: str, static_f=1.0, dynamic_f=1.0):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    path = f"{root}_PhysicsMaterials/floor"
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(static_f)
    api.CreateDynamicFrictionAttr().Set(dynamic_f)
    api.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set("multiply")
    px.CreateRestitutionCombineModeAttr().Set("multiply")
    return mat


def _bind_physics_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _find_rigid_body(prim):
    from pxr import Usd, UsdPhysics

    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
    return None


def _is_articulated(prim) -> bool:
    from pxr import Usd, UsdPhysics

    return any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in Usd.PrimRange(prim))


def _world_translation(prim) -> list:
    from pxr import Usd, UsdGeom

    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    return [round(float(t[0]), 4), round(float(t[1]), 4), round(float(t[2]), 4)]


def _asset_id_to_type() -> dict:
    import importlib.util

    spec = importlib.util.find_spec("molmo_spaces_isaac")
    if spec is None or not spec.submodule_search_locations:
        return {}
    p = Path(list(spec.submodule_search_locations)[0]) / "resources" / "asset_id_to_object_type.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _referenced_asset_id(prim) -> str:
    """THOR asset id from the object's reference, e.g. .../objects/thor/Fridge_10_mesh/Fridge_10_mesh.usda -> Fridge_10."""
    for spec in prim.GetPrimStack():
        rl = spec.referenceList
        for ref in list(rl.explicitItems) + list(rl.prependedItems) + list(rl.appendedItems) + list(rl.addedItems):
            ap = ref.assetPath
            if "objects/thor/" in ap:
                stem = Path(ap).stem
                return re.sub(r"_(mesh|prim)$", "", stem)
    return ""


_NAME_RE = re.compile(r"^(?P<lemma>.+?)_(?P<hash>[0-9a-f]{32})_(?P<count>\d+)_(?P<body>\d+)_(?P<room>\d+)")


# ----------------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------------


def load_house(
    stage_or_sim,
    house_id: str,
    root: str = "/World/House",
    *,
    apply_labels: bool = True,
    physics_fixes: bool = True,
    dynamic_objects: str = "keep",  # "keep" | "kinematic" (measured slower: see docs/scenes.md)
    lock_joints: bool = True,
    prop_angular_damping: float | None = 1.0,
    floor_friction: float = 1.0,
) -> HouseInfo:
    """Reference the house at `root`, fix physics for walking, label semantics, and return
    its rooms/objects/spawn. Must run after the Isaac app has started."""
    from pxr import Sdf, UsdGeom

    t0 = time.time()
    ref: HouseRef = parse_house_id(house_id)
    usd = ref.scene_usd
    if not usd.exists():
        raise FileNotFoundError(
            f"{usd} not found. Download it first on the box:\n"
            f"  /work/envs/isaaclab/bin/python -m scenes.download {ref.house_id}"
        )
    stage = _resolve_stage(stage_or_sim)
    warnings: list[str] = []
    stats: dict = {}

    # 1. reference + load payload ------------------------------------------------------------
    if not stage.GetPrimAtPath(root.rsplit("/", 1)[0] or "/").IsValid():
        stage.DefinePrim(root.rsplit("/", 1)[0], "Xform")
    house = stage.DefinePrim(root, "Xform")
    house.GetReferences().AddReference(str(usd))  # keep the symlink path (relative refs)
    stage.Load(Sdf.Path(root))
    geo = stage.GetPrimAtPath(f"{root}/Geometry")
    if not geo.IsValid():
        raise RuntimeError(f"{usd}: no Geometry scope under {root} after referencing (payload not loaded?)")
    children = list(geo.GetChildren())
    empty = [c.GetName() for c in children if c.GetTypeName() == "Xform" and not c.GetChildren() and _referenced_asset_id(c)]
    if empty:
        warnings.append(f"{len(empty)} object prims are empty (unresolved THOR asset references?): {empty[:5]}")
    stats["reference_s"] = round(time.time() - t0, 3)
    stats["geometry_children"] = len(children)

    # 2. collision groups ------------------------------------------------------------------------
    changed = fix_collision_filter(stage)
    if changed:
        warnings.append(f"invertCollisionGroupFilter was True on {changed}; reset to False (MolmoSpaces collision groups)")
    stats["collision_filter"] = collision_filter_report(stage)

    # 3. objects -----------------------------------------------------------------------------------
    t1 = time.time()
    if ref.kind == "procthor":
        objects, n_labels = _procthor_objects(stage, ref, root, children, apply_labels, warnings)
    else:
        objects, n_labels = _ithor_objects(stage, ref, root, children, apply_labels, warnings)
    # walls / floors
    for c in children:
        n = c.GetName()
        if apply_labels and (n.startswith("wall_") and "_visual" in n):
            _apply_label(c, "wall")
            n_labels += 1
        elif apply_labels and n.startswith("room_") and "_visual" in n:
            _apply_label(c, "floor")
            n_labels += 1
    stats["labels_applied"] = n_labels
    stats["objects_s"] = round(time.time() - t1, 3)

    # 4. rooms -----------------------------------------------------------------------------------
    rooms = _rooms(ref, stage, root, objects)
    for o in objects:
        rname = next((r.name for r in rooms if r.room_id == o.room_id), None)
        if rname is None:
            cx = (o.aabb[0][0] + o.aabb[1][0]) / 2
            cy = (o.aabb[0][1] + o.aabb[1][1]) / 2
            rname = next((r.name for r in rooms if r.contains(cx, cy)), None)
        o.room = rname

    # 5. physics fixes ---------------------------------------------------------------------------
    floor_z = 0.0
    floor = stage.GetPrimAtPath(f"{root}/Geometry/floor")
    if floor.IsValid():
        floor_z = _world_translation(floor)[2]
    else:
        warnings.append("no Geometry/floor collider plane found")
    if physics_fixes:
        stats["physics"] = _physics_fixes(stage, root, children, objects, dynamic_objects, floor_friction, warnings)
        if lock_joints:
            stats["physics"]["locked_joints"] = lock_furniture_joints(stage, root)
        if prop_angular_damping is not None and dynamic_objects == "keep":
            stats["physics"]["damped_props"] = damp_props(stage, objects, prop_angular_damping)

    # 6. bounds ----------------------------------------------------------------------------------
    if rooms:
        xs = [p[0] for r in rooms for p in r.polygon]
        ys = [p[1] for r in rooms for p in r.polygon]
        bounds = [[min(xs), min(ys)], [max(xs), max(ys)]]
    else:
        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        r = cache.ComputeWorldBound(house).ComputeAlignedRange()
        bounds = [[r.GetMin()[0], r.GetMin()[1]], [r.GetMax()[0], r.GetMax()[1]]]

    info = HouseInfo(
        house_id=ref.house_id,
        kind=ref.kind,
        source=ref.source,
        usd_path=str(usd),
        root=root,
        floor_z=floor_z,
        bounds_xy=[[round(v, 4) for v in b] for b in bounds],
        rooms=rooms,
        objects=objects,
        warnings=warnings,
        stats=stats,
        assets_dir=ref.assets_dir,
    )

    # 7. spawn (needs the occupancy grid; use the cached one when present) ------------------------
    try:
        from scenes.occupancy import get_occupancy

        occ = get_occupancy(ref.house_id)
        finalize_with_occupancy(info, occ, save=False)
    except FileNotFoundError:
        pa = procthor_agent_pose(ref)
        if pa:
            yaw = math.radians(90.0 - pa["rotation"]["y"])  # THOR yaw (deg, +z fwd, cw) -> world yaw
            info.spawn = {"x": pa["position"]["x"], "y": pa["position"]["z"], "yaw": round(math.atan2(math.sin(yaw), math.cos(yaw)), 4), "source": "procthor_agent_start_unchecked"}
        info.warnings.append("no cached occupancy: spawn not validated (run scenes.test_house once)")
    stats["load_s"] = round(time.time() - t0, 3)
    return info


def finalize_with_occupancy(info: HouseInfo, occ, save: bool = True) -> HouseInfo:
    """Spawn pose, per-room go_to points and room connectivity from the occupancy grid."""
    from scenes.occupancy import choose_spawn, connectivity, room_points

    rooms = info.room_dicts()
    pref = None
    ref = parse_house_id(info.house_id)
    pa = procthor_agent_pose(ref)
    if pa:
        yaw = math.radians(90.0 - pa["rotation"]["y"])
        pref = {"x": pa["position"]["x"], "y": pa["position"]["z"], "yaw": math.atan2(math.sin(yaw), math.cos(yaw))}
    info.spawn = choose_spawn(occ, rooms, pref)
    info.spawn["room"] = info.room_at(info.spawn["x"], info.spawn["y"])
    info.room_points = room_points(occ, rooms)
    info.connectivity = connectivity(occ, rooms, (info.spawn["x"], info.spawn["y"]))
    if not info.connectivity.get("all_rooms_reachable", False):
        bad = [k for k, v in info.connectivity["rooms"].items() if not v["reachable"]]
        info.warnings.append(f"rooms not reachable from spawn with robot radius {occ.robot_radius} m: {bad}")
    if save:
        info.save_json()
    return info


# ----------------------------------------------------------------------------------------------
# Objects
# ----------------------------------------------------------------------------------------------


def _aabb(cache, prim) -> list:
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    mn, mx = r.GetMin(), r.GetMax()
    return [[round(float(mn[i]), 4) for i in range(3)], [round(float(mx[i]), 4) for i in range(3)]]


def _bbox_cache():
    try:  # Isaac Sim 5.1 isaacsim.core.utils.bounds.create_bbox_cache (default purpose only)
        from isaacsim.core.utils.bounds import create_bbox_cache

        return create_bbox_cache(use_extents_hint=False)
    except Exception:
        from pxr import Usd, UsdGeom

        return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)


def _name_objects(objs: list[ObjectInfo]) -> None:
    """snake(category)_n numbered in sorted-id order (plan_design_contract-first.md §7 naming rule)."""
    counts: dict[str, int] = {}
    for o in sorted(objs, key=lambda o: o.id):
        s = snake(o.category)
        counts[s] = counts.get(s, 0) + 1
        o.name = f"{s}_{counts[s]}"


def _procthor_objects(stage, ref, root, children, apply_labels, warnings):
    meta = json.loads(ref.metadata_json.read_text())["objects"]
    by_name = {c.GetName(): c for c in children}
    cache = _bbox_cache()
    objs: list[ObjectInfo] = []
    n_labels = 0
    for body_name, m in meta.items():
        prim = by_name.get(body_name)
        if prim is None:  # the converter skips a few asset ids (house_converter.py:94-100)
            warnings.append(f"metadata object {m['object_id']} ({m['asset_id']}) has no prim")
            continue
        label = snake(m["category"])
        if apply_labels:
            _apply_label(prim, label)
            n_labels += 1
        rb = _find_rigid_body(prim)
        objs.append(
            ObjectInfo(
                id=m["object_id"],
                name="",
                category=m["category"],
                label=label,
                room_id=int(m["room_id"]) if m.get("room_id") is not None else None,
                room=None,
                prim_path=str(prim.GetPath()),
                body_path=str(rb.GetPath()) if (rb is not None and not m["is_static"]) else None,
                pos=_world_translation(prim),
                aabb=_aabb(cache, prim),
                is_static=bool(m["is_static"]),
                articulated=_is_articulated(prim),
                asset_id=m["asset_id"],
            )
        )
    _name_objects(objs)
    return objs, n_labels


def _ithor_objects(stage, ref, root, children, apply_labels, warnings):
    a2t = _asset_id_to_type()
    cache = _bbox_cache()
    groups: dict[tuple, list] = {}
    for c in children:
        if c.GetTypeName() != "Xform":
            continue
        n = c.GetName()
        m = _NAME_RE.match(re.sub(r"^tn__", "", n))
        if not m:
            continue
        lemma = re.sub(r"\d+$", "", m.group("lemma"))
        groups.setdefault((lemma, m.group("hash"), m.group("count")), []).append(c)
    objs: list[ObjectInfo] = []
    n_labels = 0
    for (lemma, h, count), prims in sorted(groups.items()):
        prims.sort(key=lambda p: p.GetName())
        head = prims[0]
        asset_id = _referenced_asset_id(head)
        cat = a2t.get(asset_id) or ITHOR_LEMMA_TO_TYPE.get(lemma) or lemma[:1].upper() + lemma[1:]
        label = snake(cat)
        if apply_labels:
            for p in prims:
                _apply_label(p, label)
                n_labels += 1
        if cat in STRUCTURAL_CATEGORIES:
            continue
        rb = _find_rigid_body(head)
        from pxr import Gf

        rng = Gf.Range3d()
        for p in prims:
            rng.UnionWith(cache.ComputeWorldBound(p).ComputeAlignedRange())
        aabb = [[round(float(rng.GetMin()[i]), 4) for i in range(3)], [round(float(rng.GetMax()[i]), 4) for i in range(3)]] if not rng.IsEmpty() else _aabb(cache, head)
        from pxr import UsdPhysics

        static = rb is None or bool(UsdPhysics.RigidBodyAPI(rb).GetKinematicEnabledAttr().Get())
        objs.append(
            ObjectInfo(
                id=f"{cat}|{int(count)}|{h[:8]}",
                name="",
                category=cat,
                label=label,
                room_id=0,
                room=None,
                prim_path=str(head.GetPath()),
                body_path=None if static else str(rb.GetPath()),
                pos=_world_translation(head),
                aabb=aabb,
                is_static=static,
                articulated=any(_is_articulated(p) for p in prims) or len(prims) > 1,
                asset_id=asset_id or lemma,
                extra_prims=[str(p.GetPath()) for p in prims[1:]],
            )
        )
    warnings.append("iTHOR scene: no scene_metadata.json in the MolmoSpaces archive; categories from asset ids / name lemmas")
    _name_objects(objs)
    return objs, n_labels


# ----------------------------------------------------------------------------------------------
# Rooms
# ----------------------------------------------------------------------------------------------


def _rooms(ref, stage, root, objects) -> list[RoomInfo]:
    out: list[RoomInfo] = []
    if ref.kind == "procthor":
        for r in procthor_rooms(ref):
            poly = [[float(x), float(z)] for x, z in r["polygon_thor"]]  # THOR (x, z) -> world (x, y)
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            out.append(RoomInfo(r["room_id"], r["name"], r["type"], poly, round(polygon_area(poly), 3), [round(cx, 3), round(cy, 3)]))
        return out
    # iTHOR: one room = the floor extent
    from pxr import Gf, UsdGeom

    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    rng = Gf.Range3d()
    geo = stage.GetPrimAtPath(f"{root}/Geometry")
    for c in geo.GetChildren():
        if c.GetName().startswith("floor_"):
            rng.UnionWith(cache.ComputeWorldBound(c).ComputeAlignedRange())
    if rng.IsEmpty():
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(root)).ComputeAlignedRange()
    (x0, y0), (x1, y1) = (rng.GetMin()[0], rng.GetMin()[1]), (rng.GetMax()[0], rng.GetMax()[1])
    rtype = next((t for a, b, t in ITHOR_ROOM_TYPES if a <= (ref.index or 0) <= b), "Room")
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    poly = [[round(float(a), 4), round(float(b), 4)] for a, b in poly]
    out.append(RoomInfo(0, snake(rtype), rtype, poly, round(polygon_area(poly), 3), [round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3)]))
    return out


# ----------------------------------------------------------------------------------------------
# Physics fixes
# ----------------------------------------------------------------------------------------------


def _physics_fixes(stage, root, children, objects, dynamic_objects, floor_friction, warnings) -> dict:
    from pxr import Usd, UsdPhysics

    out = {"floor_material": None, "rugs_disabled": 0, "kinematic_props": 0}
    mat = _floor_material(stage, root, floor_friction, floor_friction)
    n_floor = 0
    for c in children:
        n = c.GetName()
        if n == "floor" or n.startswith("floor_"):
            for p in Usd.PrimRange(c):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    _bind_physics_material(p, mat)
                    n_floor += 1
    out["floor_material"] = {"path": str(mat.GetPath()), "static": floor_friction, "dynamic": floor_friction, "combine": "multiply", "bound_colliders": n_floor}
    if n_floor == 0:
        warnings.append("no floor collider found to bind the floor physics material")
    for o in objects:
        if o.category in RUG_CATEGORIES:
            for p in Usd.PrimRange(stage.GetPrimAtPath(o.prim_path)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            out["rugs_disabled"] += 1
    if dynamic_objects == "kinematic":
        for o in objects:
            if o.is_static or o.body_path is None or o.articulated:
                continue
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(o.body_path)).CreateKinematicEnabledAttr().Set(True)
            out["kinematic_props"] += 1
    return out


def lock_furniture_joints(stage, root: str = "/World/House") -> int:
    """Lock every revolute/prismatic joint of the house at its authored rest pose (q = 0, which the
    converter maps to the MJCF initial qpos, e.g. an open door: house_converter.py
    set_articulated_object_init_qpos). Doors cannot swing into doorways when bumped and drawers
    cannot drift open (measured on train-59: 6 dresser drawers kept sliding, keeping the island
    awake). Re-open for manipulation work by reloading with lock_joints=False."""
    from pxr import Usd, UsdPhysics

    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(p)
        elif p.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.PrismaticJoint(p)
        else:
            continue
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        q = 0.0
        if lo is not None and hi is not None and lo <= hi:
            q = min(max(0.0, float(lo)), float(hi))
        j.CreateLowerLimitAttr().Set(q)
        j.CreateUpperLimitAttr().Set(q)
        n += 1
    return n


def damp_props(stage, objects, angular_damping: float = 1.0) -> int:
    """Angular damping on loose props (PhysX default 0.05). Pens/pencils otherwise keep rolling
    in place and never sleep (measured on train-15: 3 bodies awake -> 2.3 ms/step vs 0.5)."""
    from pxr import PhysxSchema

    n = 0
    for o in objects:
        if o.is_static or o.body_path is None or o.articulated:
            continue
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(o.body_path))
        api.CreateAngularDampingAttr().Set(float(angular_damping))
        # mass-normalised kinetic energy below which a body may sleep (PhysX default 5e-5):
        # slowly rolling pens/eggs otherwise keep narrow-phase against large static meshes alive
        api.CreateSleepThresholdAttr().Set(5e-3)
        n += 1
    return n


def sleep_house(stage_or_sim=None, root: str = "/World/House") -> dict:
    """Put every house rigid body and articulation to sleep (omni.physx simulation interface
    put_to_sleep). Call after physics has started (and ideally after ~1 s of settling). Bodies
    wake up again when the robot touches them. Some assets never fall asleep on their own and
    cost PhysX time every step (measured: train-15's toilet articulation + toilet paper + a pen
    kept 1.86 ms/step; train-38 with nothing awake: 0.32 ms/step)."""
    import omni.physx
    import omni.usd
    from pxr import PhysicsSchemaTools, Usd, UsdPhysics

    stage = _resolve_stage(stage_or_sim)
    si = omni.physx.get_physx_simulation_interface()
    sid = omni.usd.get_context().get_stage_id()
    n = awake_before = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not (p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)):
            continue
        pid = PhysicsSchemaTools.sdfPathToInt(p.GetPath())
        try:
            if si.is_sleeping(sid, pid) is False:
                awake_before += 1
            si.put_to_sleep(sid, pid)
            n += 1
        except Exception:
            pass
    return {"bodies_and_articulations": n, "awake_before": awake_before}


def awake_report(stage_or_sim=None, root: str = "/World/House") -> list[str]:
    """Paths of house bodies/articulations PhysX reports awake."""
    import omni.physx
    import omni.usd
    from pxr import PhysicsSchemaTools, Usd, UsdPhysics

    stage = _resolve_stage(stage_or_sim)
    si = omni.physx.get_physx_simulation_interface()
    sid = omni.usd.get_context().get_stage_id()
    out = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                if si.is_sleeping(sid, PhysicsSchemaTools.sdfPathToInt(p.GetPath())) is False:
                    out.append(str(p.GetPath()))
            except Exception:
                pass
    return out


def unload_house(stage_or_sim, root: str = "/World/House") -> None:
    stage = _resolve_stage(stage_or_sim)
    for p in (root, f"{root}_PhysicsMaterials"):
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)


def verify_procthor_mapping(info: HouseInfo) -> dict:
    """Check `procthor-train-N` == MolmoSpaces `train_N`: object ids of the USD metadata vs the
    ProcTHOR JSON Worldline uses, and their XY positions (THOR x,z -> world x,y)."""
    ref = parse_house_id(info.house_id)
    if ref.kind != "procthor":
        return {"applicable": False}
    j = procthor_objects(ref)
    if not j:
        return {"applicable": True, "json_available": False}
    usd_ids = {o.id for o in info.objects}
    # doors/windows are "door|a|b"/"window|a|n" in both; objects keep THOR ids
    matched = sorted(usd_ids & set(j))
    errs = []
    for o in info.objects:
        if o.id in j and j[o.id].get("position"):
            p = j[o.id]["position"]
            errs.append(math.hypot(o.pos[0] - p["x"], o.pos[1] - p["z"]))
    errs.sort()
    return {
        "applicable": True,
        "json_available": True,
        "usd_objects": len(usd_ids),
        "json_objects": len(j),
        "matched_ids": len(matched),
        "matched_frac_of_usd": round(len(matched) / max(len(usd_ids), 1), 3),
        "usd_only": sorted(usd_ids - set(j))[:20],
        "json_only": sorted(set(j) - usd_ids)[:20],
        "xy_err_median_m": round(errs[len(errs) // 2], 4) if errs else None,
        "xy_err_p90_m": round(errs[int(len(errs) * 0.9)], 4) if errs else None,
        "n_pos_compared": len(errs),
    }
