"""services/: the typed services behind the Ludi tools (PLAN §6; api/services.py protocols).

    locations      list_locations (walking distance from the current GT pose)
    navigation     NavigationService: keypoint -> pose -> BodyPort.go_to; reach_stance reposition; envelopes
    observation    ObservationService: glance records, scans (the waist scan; turn_in_place on bodies without it), waits
    reachability   ReachabilityModel: G1 geometry from the current pose, THOR-ordered reasons
    manipulation   ManipulationService: skill selection, stance check, executors, verification
    skills         the registry from config/skills.yaml (frozen enum, per-backend health)
    speech         SpeechService (timed text speech, speech_* events)
    executors/     kinematic_attach / lite / sonic_arm_script (STEPPING STONE), groot_arms (experimental)
    common         execution runner, halt gate, event sink

Services read the world only through a WorldModel (world/) and move the robot only through a BodyPort (robot/).
"""
