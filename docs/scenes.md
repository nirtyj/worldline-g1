# Scenes: MolmoSpaces houses in Isaac Sim 5.1 (M1, house component)

Owner: house agent. Code: `scenes/`. Generated assets: `/work/worldline-g1/assets/houses/<id>/`.
Evidence: `/work/worldline-g1/outputs/m1/house/` (pulled to `outputs/m1/house/`, not in git).

**Recommendation for M1: `procthor-train-38`.** It has the most walkable space in every room,
two interior doorways at least 1.6 m wide, a spawn with 5.4 m of straight free floor ahead, and
furniture in the middle of each room, which exercises path planning. `procthor-train-40`
(Worldline's main eval house) also works, but its living room is cramped: 1.3 m² of it is
reachable at a 0.30 m robot radius, and the kitchen to living-room doorway is 0.925 m wide.

## 1. Source, versions, licences, sizes

| Item | Value |
|---|---|
| Upstream | allenai/molmospaces @ `713fd12` (`/work/repos/molmospaces`), HF dataset `allenai/molmospaces`, prefix `isaac` |
| Pinned USD versions | objects/thor `20260128`, procthor-10k-train `20260128`, ithor `20260121` (`molmo_spaces_isaac/downloader/main.py:17-50`) |
| Licences | Code Apache-2.0. Data CC BY 4.0 (THOR/ProcTHOR/iTHOR subsets). Objaverse subsets ODC-BY (not used). Research and education use per Ai2 Responsible Use (`molmospaces/README.md:373-379`) |
| On disk (`/work/assets/molmospaces`, 3.1 GB) | THOR object USDs 3.0 GB (eager, all 382 archives). 4 ProcTHOR houses 150 MB (each archive has `train_N` and `train_N_ceiling`; train_40 is 8.3 MB per variant, train_59 is 52 MB). FloorPlan10 3.1 MB |
| Full split, not downloaded | procthor-10k-train is 10,000 archives, 644,790 MB per the manifest |

### Install (done on the box)

MolmoSpaces' `[sim]` extra pins Isaac Lab 2.3.1 (`molmo_spaces_isaac/pyproject.toml`), so the
package is installed without its dependencies:

```bash
uv pip install --python /work/envs/isaaclab/bin/python "molmospaces-resources==0.0.3a2" tyro msgspec \
    "lmdb~=1.7.5" "zstandard~=0.25.0" p_tqdm "tinyobjloader>=2.0.0rc13"   # numpy-stl, mujoco, scipy: already there
uv pip install --python /work/envs/isaaclab/bin/python --no-deps -e /work/repos/molmospaces/molmo_spaces_isaac
```

- `pip freeze` before and after (`/work/logs/wl/house-freeze-{before,after}.txt`) shows **only
  additions**: 12 packages plus `molmo_spaces_isaac`. Isaac Lab, Isaac Sim, torch and numpy are
  unchanged.
- `usd-exchange` is deliberately left out. The MJCF→USD converters (`ms-convert-*`) need it, but
  loading does not, and a second USD build must not shadow Kit's `pxr`.
- `molmospaces-resources` is the downloader's real dependency. It is missing from
  `molmo_spaces_isaac/pyproject.toml`; the root `pyproject.toml:72` pins it.

### Download

`ms-download --scenes procthor-10k-train` would install the whole split
(`downloader/main.py:132-134`). `scenes/download.py` builds the same `ResourceManager`
(`downloader/main.py:115-128`) but installs one archive per house:

```bash
/work/envs/isaaclab/bin/python -m scenes.download procthor-train-40 procthor-train-15 procthor-train-38 procthor-train-59 ithor-FloorPlan10
```

It writes `/work/assets/molmospaces/downloads.json` (versions, archive names, licences). The THOR
objects took 59 s, and the houses 4 to 64 s each (about 2.8 minutes in all).

## 2. House ids and the Worldline mapping (verified)

- `procthor-train-N` is MolmoSpaces `train_N`.
  - MolmoSpaces exports `prior.load_dataset("procthor-10k")[split][N]` as `{split}_{N}` (`molmo_spaces/housegen/exporter.py:78,92,498`).
  - Worldline loads the same item (`ludo-runtime thor/procthor.py:36-49`).
- Aliases accepted by `scenes.catalog.parse_house_id`:
  - `train_40`, `procthor-10k-train-40`, `procthor-10k-train/40`
  - `FloorPlan10`, `ithor-FloorPlan10`
- **The mapping check passes on all four houses** (`verify_procthor_mapping`, in each `metrics.json` as `mapping_check`):
  - 100% of USD object ids match ProcTHOR JSON ids. The JSON is Worldline's cache, copied to `scenes/data/procthor/`.
  - The XY position error, THOR (x, z) against world (x, y), has a median of 0.0 to 0.1 mm and a p90 of 2.5 to 18 mm.
- The coordinate transform is THOR (x, y_up, z) → world (x, z, y) (`molmo_spaces/housegen/utils.py:107-116`). So room polygons need no fitted transform.

## 3. Picking houses

The criteria were: a kitchen plus at least 2 more rooms, all rooms reachable at a 0.30 m robot
radius, interior doorways wider than 0.65 m, a spawn with at least 2.5 m of straight run, and
cheap physics. Numbers come from the final runs in `outputs/m1/house/<id>/metrics.json` (PhysX
`numThreads=0`, house put to sleep; see section 7).

| House | Rooms: area / reachable area (m²) | Interior doorway clear width (m) | Objects static/dyn/artic | Spawn (x, y, yaw), room, straight run | Physics-only RTF (step p50) | + 640×480 camera at 30 Hz RTF (render p50) |
|---|---|---|---|---|---|---|
| **procthor-train-38** | kitchen 43.8/26.8, bedroom 36.5/20.4, living 29.2/13.4 | 1.625, 1.675 | 42/35/8 | (7.75, 5.75, +90°) living room, 5.4 m | 13.9 (0.35 ms) | 2.17 (12.4 ms) |
| procthor-train-40 | kitchen 36.0/19.4, bedroom 21.0/8.5, living 6.0/**1.3** | **0.925**, 1.675 | 28/39/9 | (5.75, 4.25, −90°) kitchen, 3.9 m | 14.7 (0.34 ms) | 2.18 (11.8 ms) |
| procthor-train-15 | bedroom 50.2/29.3, kitchen 37.7/20.0, living 28.2/16.0, bathroom 25.1/16.2 | 1.675, 1.725, 3.425 | 40/60/15 | (9.25, 2.75, −22.5°) living room, 3.1 m | 13.7 (0.36 ms) | 2.12 (12.5 ms) |
| procthor-train-59 | bedroom 37.0/19.8, kitchen 31.7/16.0, bathroom 21.1/14.3, living 15.9/4.6 | 3.425, **0.825**, 1.675 | 42/43/11 | (2.75, 6.25, 0°) kitchen, 6.0 m | 14.1 (0.35 ms) | 2.38 (10.9 ms) |
| ithor-FloorPlan10 | kitchen 35.4/14.6 (single room) | – | 27/36/30 | (−3.125, −1.575, +90°), 4.4 m | 14.1 (0.34 ms) | 2.11 (12.4 ms) |

- Each ProcTHOR house also has an exterior door (`door|1|*`). It is closed, as in the ProcTHOR JSON (`openness 0`).
- Go-to points with the most clearance in each room are in `house_info.json` → `room_points` and are printed by `python -m scenes.info <id>`. For train-38:
  - kitchen (4.13, 6.08)
  - bedroom (7.13, 2.18)
  - living room (8.28, 5.93)

  Going from the spawn to the kitchen, then the bedroom, then back to the living room passes through both interior doorways.

## 4. API (Isaac side and pure Python)

```python
# inside Isaac (after SimulationApp; P1 = sim_isaac/app.py)
from scenes.loader import load_house, sleep_house, fix_collision_filter
info = load_house(sim_or_stage, "procthor-train-38", root="/World/House")   # before the robot is spawned
# ... spawn robot, sim.reset(), let it settle ~1 s (band on) ...
sleep_house(stage, info.root)          # house bodies asleep until touched (RTF, section 7)
fix_collision_filter(stage)            # again if an Isaac Lab InteractiveScene was built (section 6)
info.spawn        # {"x","y","yaw","clearance_m","forward_free_m","room","source"}
info.to_scene_info()                   # the REP get_scene_info body (docs/contracts/m1.md §1.6/§1.8)
info.bounds, info.floor_z, info.occupancy_npz

# anywhere (numpy/scipy only), e.g. body/ or P1's REP handler
from scenes.occupancy import get_occupancy, occupancy_reply
occ = get_occupancy("procthor-train-38")            # Occupancy: raw/inflated/inflated_low/dist/low, world<->cell
reply = occupancy_reply("procthor-train-38", robot_radius=0.25)   # re-inflates without Isaac, writes occupancy_r0.25.npz
from scenes.loader import load_house_info           # cached HouseInfo from house_info.json, no Isaac
```

`load_house` options (defaults shown):

| Option | Default | What it does |
|---|---|---|
| `apply_labels` | True | Semantic labels |
| `physics_fixes` | True | Floor material, rugs |
| `lock_joints` | True | Doors and drawers held at their rest pose, so a bumped door cannot swing into a doorway |
| `prop_angular_damping` | 1.0 | Also sets sleep threshold 5e-3 on loose props |
| `dynamic_objects` | `"keep"` | `"kinematic"` measured **slower**: with joints locked, the step went from 0.58 to 2.70 ms (train-59) and from 2.28 to 4.77 ms (train-15) |

`HouseInfo` holds:

- `rooms`: `RoomInfo` room_id, name, type, polygon, area_m2, center
- `objects`: `ObjectInfo` id, name, category, label, room_id, room, prim_path, body_path, pos, aabb, is_static, articulated, asset_id
- `spawn`, `room_points`, `connectivity`, `warnings`, `stats`

Generating and refreshing the per-house assets takes about 30 s per house. It also produces the evidence:

```bash
bash /work/worldline-g1/scenes/run_house_test.sh procthor-train-38 [more ids]    # logs /work/logs/wl/house-test-<id>.log
/work/envs/isaaclab/bin/python -m pytest -q scenes/tests/test_pure.py            # 6 pure-Python tests incl. cached houses
```

## 5. Occupancy grid

Stored in `/work/worldline-g1/assets/houses/<id>/occupancy.{npz,png}` and `occupancy_meta.json`.

**Grid layout**

- Resolution 5 cm. Obstacles are any house collider between z = 0.10 and 1.60 m.
- The inflation radius is 0.30 m, for the G1 footprint.
- Arrays are indexed `[iy, ix]`, row = y. `origin` is the world XY of the lower-left **corner** of cell (0, 0), and `x = x0 + (ix + 0.5)·res`. This matches the P1 contract.

**Layers in the npz**

| Layer | Meaning |
|---|---|
| `raw` | 0 free, 1 obstacle, 2 outside every room polygon |
| `inflated` | `raw` inflated by the robot radius |
| `low` | Colliders between 0.02 and 0.10 m, such as chair and lamp feet, dog-bed rims and fridge plinths |
| `inflated_low` | `raw` + `low`, inflated. **Recommended for walking.** Adding `low` never disconnected a room in any of the 5 houses |
| `dist` | Metres to the nearest `raw != 0` cell |
| `occ`, `occ_inflated` | Contract aliases: `occ` = blocked, including `low`; `occ_inflated` = `inflated_low` |

**How the grid is generated**

- Two independent rasters are computed:
  1. `isaacsim.asset.gen.omap` `Generator.generate2d` (Isaac Sim 5.1 ext 2.0.29, API from its `MapGenerator.h` and `tests/test_occupancy.py`)
  2. An exact PhysX `overlap_box` per cell
- omap is kept when the two agree (obstacle IoU ≥ 0.9); otherwise the overlap raster is used. IoU was 0.990 to 1.000 on all five houses, so all five use omap.
- Each raster takes under 0.2 s.
- omap needs the extension enabled (`enable_extension("isaacsim.asset.gen.omap")`) and PhysX loaded (after `sim.reset()`).
- **5.1 caveats:**
  - omap's buffer is x-flipped relative to its bounds (`utils.compute_coordinates`), so cells are placed by `get_occupied_positions()` / `get_free_positions()` in world coordinates.
  - Its lattice is anchored at the seed point, so the seed must be snapped onto the grid lattice. Unsnapped, IoU on FloorPlan10 was 0.83; snapped, it was 1.00.
- The floor is excluded by `z_min = 0.10`, so there is no floor bug.

**Spawn**

- The ProcTHOR/Worldline agent start is used if it has at least 0.55 m clearance and a straight run of at least 2.5 m. All four ProcTHOR houses qualify.
- Otherwise the best cell by clearance plus straight run is used.
- Yaw is the longest free run among 16 headings.

## 6. Walkability and physics caveats (handled)

- **Isaac Lab collision-group inversion** (MolmoSpaces README: "collision groups are reverted by default when using IsaacLab's InteractiveScene").
  - MolmoSpaces' `structural_cls_group` filters itself and `articulable_dynamic_cls_group` (`house_converter.py:809-838`).
  - `InteractiveScene.filter_collisions` (`isaaclab/scene/interactive_scene.py:214-215`, always on CPU physics) sets `physxScene:invertCollisionGroupFilter = True` (`isaacsim.core.cloner cloner.py:446`).
  - With the inversion, the structural group collides *only* with those groups: the robot falls through the floor and walks through walls.
  - `load_house` resets the flag. If P1 builds an `InteractiveScene`, it should use `filter_collisions=False` or call `fix_collision_filter()` afterwards. With a plain `SimulationContext` the flag stays False; this was checked after `reset()` in every run.
- **Floor**
  - The only floor collider is an invisible plane at z = 0 (`Geometry/floor`, purpose guide, `house_converter.py:790-806`). Room floor meshes are visual only.
  - It gets SONIC's training ground material: friction 1.0/1.0, combine `multiply` (`gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:320-330`).
  - Downward raycasts in free cells hit the floor at z = 0 in 99.5 to 100% of cells, with zero misses in all houses. The remainder hit low props.
- **No invisible blockers.** Every obstacle cell lies inside a visual AABB (+0.10 m): 0 unexplained cells in all 5 houses.
  - All 1,893 colliders in train_40 are `purpose=guide`, so AABBs are visual extents.
  - Walls with doorways are split into 3 convex pieces, so the convexHull approximation keeps doorways open. Clear widths are in the table in section 3.
- **Semantic labels.** `add_labels(prim, [snake(category)], "class")` (Isaac Sim 5.1 `semantics.py:218`) is applied to every object root, plus `wall` and `floor` on the visual meshes: 96 to 155 prims per house.
- **Joints and props.** Unlocked, these ProcTHOR articulations never sleep and cost 2 to 6 ms per step:
  - dresser drawers drifting open (train-59)
  - a toilet lid (train-15)
  - rolling pens

  `lock_joints`, prop damping and `sleep_house()` fix this (section 7).

## 7. Load time and RTF (house only, no robot)

Load time for one house, warm shader cache:

- App start: 10 to 14 s
- `load_house`: 0.3 to 2.3 s
- First renders: 0.6 to 2.6 s
- Physics ready: 12.5 to 17.8 s total

The **first** launch with a new house compiled RTX materials for 106 s (train_40, first run of the day).

RTF with physics at 200 Hz on CPU PhysX, house only:

| Setting (train-38 unless noted) | Step p50 | Physics-only RTF | + camera at 30 Hz RTF |
|---|---|---|---|
| Kit default `numThreads=8` | 0.69 ms | 6.6 | 1.77 |
| `numThreads=1/2/4` | 0.58–0.69 ms | 6.6–8.1 | 1.78–2.04 |
| **`numThreads=0`** (PhysX on the calling thread) | **0.33–0.35 ms** | **13.9–14.9** | 2.2–2.5 |
| train-15 or train-59, nothing locked or slept, 8 threads | 1.8–4.3 ms | 1.1–2.6 | 0.7–1.4 |
| train-15 or train-59, locked + `sleep_house`, 0 threads | 0.36 ms | 13.2–13.8 | 2.0–2.2 |

- **Advice to P1 (measured with the house only):**
  - Set `/persistent/physics/numThreads = 0`. PhysX worker threads spin: at 8 threads the process used 5.8 CPU-ms per 0.69 ms step, which competes with the deploy's real-time threads.
  - Call `sleep_house()` after settling.
  - The 640×480 camera, at 10 to 12 ms per frame (about 0.33 s wall per simulated second), dominates the house cost. That leaves about 0.6 s per simulated second for the robot and the bridge at RTF 1.
- **Test conditions.**
  - The box was shared during the tests. Other agents used 12 to 86% of all CPUs, recorded as `cpu` in each metrics block. The RTF numbers are therefore lower bounds.
  - The GPU footprint of the test process was 3.2 to 3.8 GB.
- The Kit setting is "persistent", but the test runs did not write it back: `user.config.json` still says 8.
- At 640×480 the RTX DLSS upscaler renders internally at 320×240 (Kit warning). For image quality, P1 may want to set the DLSS/antialiasing mode explicitly.

## 8. Object categories vs AI2-THOR types (Worldline compatibility)

- **Categories are AI2-THOR object types.** They are the ProcTHOR JSON types verbatim: `AlarmClock`, `CounterTop`, `Fridge`, `DiningTable`, `GarbageCan`, `HousePlant`, ... There are 36 to 56 per house, plus `Doorway`, `Doorframe` and `Window` from the door and window lists.
  - Labels and `ObjectInfo.name` are snake_case (`alarm_clock_1`), numbered in sorted-id order. This is the naming rule in the design doc.
- **Differences Worldline will see (to handle in M2):**
  - AI2-THOR *runtime* sub-objects do not exist, because MolmoSpaces works from the JSON. Worldline's `SURFACE_TYPES` and `LANDMARK_TYPES` (`thor/world.py:61-74`) use `SinkBasin` (here `Sink`), `Shelf` (here `ShelvingUnit`) and `StoveBurner` (none of the 4 houses has a stove; FloorPlan10 has a `Stove` built from iTHOR geometry). Add aliases `Sink→SinkBasin` and `ShelvingUnit→Shelf`.
  - Thin props are missing from the USD:
    - `CreditCard`: 2 in train-15, 2 in train-38, 1 in train-59
    - `Watch`: 2 in train-15
    - `KeyChain`: 1 in train-59
    - `Plunger`: its metadata entry has no prim in train-15 and train-59
    - train-40 is complete
  - iTHOR archives ship **no** `scene_metadata.json`:
    - Categories come from the referenced THOR asset id (`molmo_spaces_isaac/resources/asset_id_to_object_type.json`), or from the prim-name lemma for iTHOR's own geometry: cabinets, drawers, `Stove`, `StoveKnob`, `Dishwasher`, `CounterTop`.
    - Ids are synthesized (`Fridge|1|08880dea`), and the scene is one room.
- The static/dynamic split and room ids come from the metadata (`is_static`, `room_id`). Receptacle ("on the counter") relations are not in the data; derive them geometrically from the AABBs in M2.

## 9. Known issues

- Loose props are dynamic. A few settle by 5 to 18 cm in the first seconds (an egg, a spray bottle in train-15). Their live pose is at `ObjectInfo.body_path` (PhysX), not the USD xform.
- The ceiling variant (`train_N_ceiling`) is downloaded but not used. The top-down render needs no ceiling.
- `sleep_house` does not stick on 11 iTHOR FloorPlan10 bodies (cabinet and drawer roots). That scene is cheap anyway (0.34 ms/step).
- The spawn comes from the cached occupancy. On a fresh box, run `scenes/run_house_test.sh <id>` once (about 30 s). Otherwise `load_house` falls back to the unchecked ProcTHOR start and warns.
- `occupancy.png` shows the `inflated` layer. The overlay (`outputs/.../occupancy_overlay.png`) adds rooms, AABBs, spawn and room points.

## 10. Evidence (`/work/worldline-g1/outputs/m1/house/<id>/`)

| File | Contents |
|---|---|
| `metrics.json` | Counts, categories, mapping check, occupancy meta, connectivity, floor/door/blocker/settle/awake checks, RTF with CPU contention, times, GPU MiB |
| `topdown.png` + `topdown_meta.json` | 1024 px, 20° FOV camera at 36 m, floor-plane pixel↔world mapping |
| `eye_spawn.png`, `eye_kitchen.png`, `eye_<room>.png` | 1.2 m eye-height views |
| `rtf_camera_last_frame.png` | Last frame of the 640×480 camera used in the RTF run |
| `occupancy.png`, `occupancy_overlay.png`, `house_info.json` | Grid images and the cached house description |

- `summary.md` / `summary.json` (`python -m scenes.summarize`) compare every run, including the thread, lock, sleep and kinematic experiments (`*-thr*`, `*-locked*`, `*-slp-*`, `*-kinematic`).

## 11. What M2 needs from scenes

- Type aliases for Worldline, and fixtures (`scenes/<scene>/fixtures.yaml`), for example placing `alarm_clock_1` and removing `Banana`.
- Receptacle relations computed from AABBs.
- An instance-id → object id map for visibility, from the labels plus `instance_id_segmentation_fast`.
- Unlocking specific joints (fridge, drawers) for manipulation: `lock_joints=False` or per object.
- Re-validating `sleep_house` with the robot touching props.
