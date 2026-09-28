"""wl-body (P3): the body service that drives SONIC's planner for the M1 stack.

Modules:
  config         ports and tunables (all ports shift together with --port-offset / WL_PORT_OFFSET)
  wire           gear_sonic ZMQ wire format (command/planner) + gt.pose / camera / REP decoding
  frames         world <-> SONIC planner frame (PlannerFrame)
  sonic_mux      SonicMux: the ONLY binder of the SONIC input PUB (5556), keepalive >= 10 Hz
  p1_client      P1 (wl-isaac) REQ client (5600) and gt.pose subscriber (5601)
  deploy_monitor g1_debug subscriber (5557)
  nav_grid       occupancy grid, inflation, A* (scipy csgraph), line-of-sight smoothing
  path_follower  pure-pursuit follower, arrival and stuck detection (GoTo motion)
  motions        stand / walk / turn_to / stop motions
  service        ROUTER 5610 / PUB 5611 body service (python -m body.service)
  client         BodyClient (blocking + async handles + events + camera)
"""

__version__ = "0.1.0"
