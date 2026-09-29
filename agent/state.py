"""Runtime state: beliefs with provenance, the task version, action handles,
and a trace log.

Every belief says where it came from and whether an observation confirmed
it. After a cancel, whatever the cancelled action touched is UNKNOWN until
the robot looks. A late result (from an action serving an old intent
version) is recorded as an unverified hint, never as progress.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from api.observation import in_frustum
from api.results import envelope_status
from api.types import ARMS, UNKNOWN


@dataclass
class Fact:
    value: Any                 # an object id, a keypoint, None, or UNKNOWN
    source: str                # look, gripper, odometry, memory, skill, late_result, cancel ...
    t: float                   # sim time of the information
    verified: bool             # confirmed by an observation


@dataclass
class ObjectBelief:
    id: str
    type: str
    brand: str | None
    color: str | None
    where: Fact                # a surface, "hand:<arm>", "floor" or UNKNOWN
    x: float | None = None
    depth: float | None = None
    seen_from: str | None = None
    pose: dict[str, float] | None = None
    distance_m: float | None = None
    visible: bool | None = None
    confidence: float | None = None
    observed_at: float | None = None
    usual: str | None = None               # from spatial memory: where it is most often
    mem_status: str | None = None          # seen, missed, moved, held (as memory had it)
    history: list[dict[str, Any]] | None = None   # its last few sightings, from memory

    @property
    def label(self) -> str:
        return " ".join(p for p in (self.color, self.brand, self.type) if p)


@dataclass
class LandmarkBelief:
    """A fixed thing the robot has seen (microwave, stove, fridge): it can't be
    picked up, and it doesn't move, so a sighting stays good."""
    id: str
    label: str
    near: Fact                 # the keypoint to stand at
    pos: tuple[float, float] | None = None


def in_view(view: dict[str, Any], pos: tuple[float, ...], h: float | None = None) -> bool:
    """Was this spot inside one of an observation's views?

    The 2-D test (heading, field of view, range) is Worldline's. When the view has
    ``vfov``/``cam_h`` and a height is known (``h`` or ``pos[2]``), the elevation must
    also be inside the vertical field of view (api/observation.in_frustum, PLAN 5.10):
    out-of-frame spots are never asserted absent."""
    return in_frustum(view, pos, h)


class BeliefState:
    def __init__(self) -> None:
        self.robot_at = Fact(None, "init", 0.0, False)
        self.between: tuple[str, str] | None = None
        self.holding: dict[str, Fact] = {a: Fact(None, "init", 0.0, False) for a in ARMS}
        self.hints: dict[str, str] = {}              # arm -> object a late result says is held
        self.objects: dict[str, ObjectBelief] = {}
        self.people: dict[str, dict[str, Any]] = {}
        self.landmarks: dict[str, LandmarkBelief] = {}
        self.looked: dict[str, Fact] = {}            # keypoint -> when the robot last looked there
        self.blocked: set[tuple[str, str]] = set()
        self.rev = 0                                  # bumps on every change

    def _bump(self) -> None:
        self.rev += 1

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def load_memory(self, entries: list[dict[str, Any]]) -> None:
        for m in entries:
            t = float(m.get("last_seen_t", -600))
            pos = m.get("pos")
            if m.get("kind") == "landmark":
                self.landmarks[m["id"]] = LandmarkBelief(m["id"], m.get("label", m.get("type", "")),
                                                         Fact(m.get("near"), "memory", t, False),
                                                         tuple(pos) if pos else None)
            elif m.get("kind") == "looked":
                self.looked[m["id"]] = Fact(True, "memory", t, False)
            else:
                ob = ObjectBelief(
                    id=m["id"], type=m.get("type", "object"), brand=m.get("brand"), color=m.get("color"),
                    where=Fact(m.get("surface") or UNKNOWN, "memory", t, False))
                if pos:
                    ob.pose = {"x": pos[0], "z": pos[1]}
                ob.usual, ob.mem_status, ob.history = m.get("usual"), m.get("status"), m.get("history")
                self.objects[m["id"]] = ob
        self._bump()

    def set_pose(self, at: str | None, between: list[str] | None, source: str, t: float) -> None:
        self.robot_at = Fact(at, source, t, True)
        self.between = tuple(between) if between else None
        self._bump()

    def set_hand(self, arm: str, value: Any, source: str, t: float, verified: bool) -> None:
        self.holding[arm] = Fact(value, source, t, verified)
        if verified:
            self.hints.pop(arm, None)
        self._bump()

    def mark_hand_unknown(self, arm: str, source: str, t: float, hint: str | None = None) -> None:
        self.holding[arm] = Fact(UNKNOWN, source, t, False)
        if hint:
            self.hints[arm] = hint
        self._bump()

    def apply_observation(self, data: dict[str, Any], t: float,
                          surface_xy: dict[str, tuple[float, float]] | None = None,
                          surface_h: dict[str, float] | None = None) -> None:
        """An observation (glance or scan) is ground truth for what the camera saw and for both hands.

        Something believed to be somewhere a view covered (inside its 3-D frustum), but
        not seen, is UNKNOWN (look_absent). Somewhere no view covered stays as it was:
        not checked isn't the same as missing. A glance reports no views, so it never
        marks anything absent."""
        at = data.get("at")
        views = data.get("views")
        if at is not None:
            self.robot_at = Fact(at, "look", t, True)
            self.between = None
            self.looked[at] = Fact(True, "look", t, True)
        seen_ids: set[str] = set()
        for surface, items in (data.get("surfaces") or {}).items():
            for v in items:
                oid = v["id"]
                seen_ids.add(oid)
                ob = self.objects.get(oid)
                if ob is None:
                    ob = ObjectBelief(id=oid, type=v.get("type", "object"), brand=v.get("brand"),
                                      color=v.get("color"), where=Fact(surface, "look", t, True))
                    self.objects[oid] = ob
                ob.type, ob.brand, ob.color = v.get("type", ob.type), v.get("brand"), v.get("color")
                ob.where = Fact(surface, "look", t, True)
                ob.x, ob.depth, ob.seen_from = v.get("x"), v.get("depth"), at
                pos = v.get("pos")
                if pos:
                    ob.pose = {"x": pos[0], "z": pos[1]}
                    if len(pos) > 2 and pos[2] is not None:
                        ob.pose["h"] = pos[2]
        for lm in data.get("landmarks") or []:
            self.landmarks[lm["id"]] = LandmarkBelief(lm["id"], lm.get("label", lm.get("type", "")),
                                                      Fact(lm.get("near"), "look", t, True),
                                                      tuple(lm["pos"][:2]) if lm.get("pos") else None)
        for oid, ob in self.objects.items():
            where = ob.where.value
            if oid in seen_ids or not isinstance(where, str) or where == UNKNOWN or where.startswith("hand"):
                continue
            if views is None:                 # a robot that doesn't report its views: the old rule
                gone = where in (data.get("surfaces") or {})
            else:
                if ob.pose and ob.pose.get("x") is not None:
                    pos = (ob.pose["x"], ob.pose["z"])
                    h = ob.pose.get("h")
                else:
                    pos = (surface_xy or {}).get(where)
                    h = None
                if h is None:
                    h = (surface_h or {}).get(where)
                gone = pos is not None and any(in_view(v, pos, h) for v in views)
            if gone:
                ob.where = Fact(UNKNOWN, "look_absent", t, False)
        for arm, oid in (data.get("hands") or {}).items():
            if arm not in self.holding:
                continue
            self.holding[arm] = Fact(oid, "look", t, True)
            self.hints.pop(arm, None)
            if oid is not None:
                ob = self.objects.get(oid)
                if ob is not None:
                    ob.where = Fact(f"hand:{arm}", "look", t, True)
        # Objects we believed were in a hand that the observation shows empty.
        for oid, ob in self.objects.items():
            where = str(ob.where.value)
            if where.startswith("hand:"):
                arm = where.split(":", 1)[1]
                if arm in self.holding and self.holding[arm].value != oid:
                    ob.where = Fact(UNKNOWN, "look_absent", t, False)
        self._bump()

    # the THOR-era name
    apply_look = apply_observation

    def apply_perception(self, perception: dict[str, Any], robot: dict[str, Any],
                         t: float, source: str) -> None:
        """Fuse a semantic perception frame without treating it as action truth.

        THOR supplies every object with confidence 1.0.  A real adapter may
        supply only tracked objects and lower confidences.  Continuous
        perception refreshes object beliefs, while hand identity remains under
        the stricter pick/place/look reconciliation path.
        """
        changed = False
        grippers = robot.get("grippers") or {}
        closed = [arm for arm, g in grippers.items() if g.get("closed")]
        for oid, raw in (perception.get("objects") or {}).items():
            if raw.get("visible") is False:
                continue                # a camera only reports what is in view
            item = dict(raw)
            confidence = float(item.get("confidence", 0.0))
            where = item.get("where") or UNKNOWN
            # G1: the world model reports "hand:<arm>" directly (two hands can be closed at once).
            # The old one-closed-gripper rule stays as the fallback for a bare "hand".
            if where == "hand" and len(closed) == 1:
                where = f"hand:{closed[0]}"      # the camera sees it held; the closed gripper says which
            verified = confidence >= 0.5
            item_source = str(item.get("source") or source)
            ob = self.objects.get(oid)
            if ob is None:
                ob = ObjectBelief(
                    id=oid, type=item.get("type", "object"), brand=item.get("brand"),
                    color=item.get("color"), where=Fact(where, item_source, t, verified),
                )
                self.objects[oid] = ob
                changed = True

            # An interrupted manipulation deliberately leaves hand state
            # UNKNOWN until reconciliation.  Do not let a background frame
            # silently clear that safety fence.
            guarded_hand = (str(where).startswith("hand:") and
                            any(not fact.verified and fact.value == UNKNOWN
                                for fact in self.holding.values()))
            before = (ob.type, ob.where.value, ob.where.verified)
            ob.type = item.get("type", ob.type)
            ob.brand = item.get("brand", ob.brand)
            ob.color = item.get("color", ob.color)
            if not guarded_hand:
                ob.where = Fact(where, item_source, t, verified)
            ob.pose = dict(item.get("pose") or {}) or None
            ob.distance_m = item.get("distance_m")
            ob.visible = item.get("visible")
            ob.confidence = confidence
            ob.observed_at = float(item.get("observed_at", t))
            # pose, distance and visibility change every frame; only a change of
            # place (or a first sighting) is a change of belief
            changed |= before != (ob.type, ob.where.value, ob.where.verified)

        for name, raw in (perception.get("landmarks") or {}).items():
            if not raw.get("visible"):
                continue
            pose = raw.get("pose") or {}
            old = self.landmarks.get(name)
            if old is None or old.near.value != raw.get("near") or old.near.source == "memory":
                changed = True
            self.landmarks[name] = LandmarkBelief(name, raw.get("label", name), Fact(raw.get("near"), "camera", t, True),
                                                  (pose.get("x"), pose.get("z")) if pose else None)

        people = {name: {k: v for k, v in info.items() if k != "distance_m"}
                  for name, info in (perception.get("people") or {}).items()}
        if people != self.people:
            self.people = people
            changed = True
        if changed:
            self._bump()

    def apply_navigate(self, status: str, data: dict[str, Any], t: float) -> None:
        if envelope_status(status) == "succeeded":
            self.set_pose(data.get("at"), None, "odometry", t)
        else:
            self.set_pose(data.get("at"), data.get("between"), "odometry", t)
        if data.get("blocked_edge"):
            self.blocked.add(tuple(data["blocked_edge"]))
            self._bump()

    def claim_pick(self, arm: str, oid: str, t: float) -> None:
        """A pick reported success: unverified until the robot looks."""
        self.holding[arm] = Fact(oid, "skill", t, False)
        ob = self.objects.get(oid)
        if ob is not None:
            ob.where = Fact(f"hand:{arm}", "skill", t, False)
        self._bump()

    def claim_place(self, arm: str, oid: str, surface: str | None, t: float) -> None:
        self.holding[arm] = Fact(None, "skill", t, False)
        ob = self.objects.get(oid)
        if ob is not None:
            ob.where = Fact(surface or UNKNOWN, "skill", t, False)
        self._bump()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def hand_known_empty(self, arm: str) -> bool:
        f = self.holding[arm]
        return f.value is None and f.verified

    def hand_holds(self, arm: str, oid: str) -> bool:
        f = self.holding[arm]
        return f.value == oid and f.verified

    # ------------------------------------------------------------------
    # Rendering for the brain (brains.interface.BELIEF_EXAMPLE)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        hands = {}
        for arm in ARMS:
            f = self.holding[arm]
            hands[arm] = {"holding": f.value, "verified": f.verified, "source": f.source, "t": round(f.t, 2)}
            if arm in self.hints:
                hands[arm]["hint"] = f"a late result says {self.hints[arm]} (unverified)"
        objects = {}
        for oid, ob in self.objects.items():
            objects[oid] = {"type": ob.type, "brand": ob.brand, "color": ob.color, "label": ob.label,
                            "where": ob.where.value, "x": ob.x, "depth": ob.depth,
                            "seen_from": ob.seen_from, "verified": ob.where.verified,
                            "source": ob.where.source, "t": round(ob.where.t, 2),
                            "pose": ob.pose, "distance_m": ob.distance_m,
                            "visible": ob.visible, "confidence": ob.confidence,
                            "usual": ob.usual, "mem_status": ob.mem_status, "history": ob.history,
                            "observed_at": round(ob.observed_at, 2) if ob.observed_at is not None else None}
        return {
            "robot": {"at": self.robot_at.value, "between": list(self.between) if self.between else None},
            "hands": hands,
            "objects": objects,
            "people": dict(self.people),
            "landmarks": {k: {"label": lm.label, "near": lm.near.value, "verified": lm.near.verified,
                              "source": lm.near.source, "t": round(lm.near.t, 2),
                              "pos": [round(v, 2) for v in lm.pos] if lm.pos and None not in lm.pos else None}
                          for k, lm in sorted(self.landmarks.items())},
            "looked": {k: {"t": round(f.t, 2), "source": f.source} for k, f in self.looked.items()},
            "blocked": [list(e) for e in sorted(self.blocked)],
        }


@dataclass
class TaskState:
    intent_version: int = 0
    control_epoch: int = 0
    paused: bool = False
    goal: str | None = None
    utterances: list[Any] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)
    directives: list[Any] = field(default_factory=list)
    current_directive: Any = None


@dataclass
class ActionHandle:
    """A running execution, as the harness tracks it. Wraps the service's ExecutionHandle."""
    entry_id: str                    # the execution_id
    created_for: int                 # the generation (intent_version) this action serves
    skill: str                       # the tool name
    args: dict[str, Any]
    resources: frozenset[str]        # {"body"}, {"sense"}, {"sense", "body"} (a scan), {"wait"}
    task: asyncio.Task | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    handle: Any = None               # the api ExecutionHandle, once started
    source: str = "brain"
    cancel_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def goal(self) -> Any:            # the THOR-era name
        return self.handle

    def cancel(self, reason: str = "cancelled") -> None:
        """Request a cancel. The service stops at its next safe point; await ``task``."""
        if self.done or self.cancel_event.is_set():
            return
        self.cancel_reason = reason
        self.cancel_event.set()
        if self.handle is not None:
            self.handle.cancel(reason)


class TraceLog:
    """The runtime's own decision log; ui/server.py streams its rows to the page's live feed."""

    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.rows: list[dict[str, Any]] = []
        self.sinks: list[Any] = []           # e.g. the episode log, which writes rows to disk

    def log(self, type: str, **fields: Any) -> None:
        row = {"t": round(self.clock.now(), 3), "type": type, **fields}
        self.rows.append(row)
        for sink in self.sinks:
            sink(row)
