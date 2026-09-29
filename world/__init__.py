"""world/: the ONLY ground-truth readers on the runtime side (PLAN §6.2, §13).

Everything that looks at simulator truth (object poses, visibility, receptacles, rooms, the occupancy grid,
the robot's GT pose) lives here, behind the semantic `WorldModel` protocol (world/model.py), so a perception
stack can replace it later without touching agent/, brains/ or llmkit/ (which must never import this package).

Modules
    coords      the only Isaac <-> Worldline frame conversion (map_x = isaac_x, map_z = isaac_y, yaw cw from +y)
    vocab       MolmoSpaces category -> AI2-THOR type aliases, surface / landmark / container / pickupable sets
    scene       scene data (house_info / get_scene_info) + room geometry
    nav_grid    occupancy grid, all-pairs geodesics (keypoint edges), A* paths (body.nav_grid planner)
    mapgen      Worldline's MAP (lookup_keypoints() shape) from scene data + occupancy
    where       geometric "where" (parent receptacle) of every object
    perception  GT "what the head camera sees now": frustum + AABB-ray occlusion
    model       WorldModel / SimControl / Localizer protocols and the shared dataclasses
    lite_world  offline world (recorded house_info + occupancy), for tests and the `lite` profile
    isaac_client  IsaacGTWorldModel: P1 REP 5600 + SUB 5601
    frames      camera frame sources (P1 head/ego frames via viz.tap.FrameTap; lite: none)
"""
