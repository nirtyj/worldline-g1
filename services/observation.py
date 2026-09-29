"""ObservationService (PLAN §6.5, §5.10): glances, scans, wait_and_observe, and the 10 Hz perception hints.

    glance_record()   "obs-g<rev>": the latest GT perception snapshot of the head camera (views=[]). Cheap (no
                      motion, no render); stamped on every ToolResult (doc §42/§51). rev bumps when what is seen
                      changes.
    observe(glance)   the current view only; LookData with views=[] (never marks anything absent).
    observe(scan)     executor `waist` (M2b B.5, the target): the body's `scan` op turns the waist YAW to -35, 0,
                      +35 deg with the base still (one row: SONIC does not move the waist pitch under the override,
                      so the PLAN's second row is not possible; docs/arm_tracking.md §8.5), and the live view is
                      sampled in the middle of each hold (the head camera rides on the torso). The scan leases the
                      body (ARM_SCRIPT) and carries the execution's fence. While carrying (CarryLock, or anything
                      in a hand) it is a glance instead: SONIC turns the shoulders with the waist and moves the
                      palms 6-10 cm. Executor `turn_in_place` (the M2a INTERIM, also the fallback for a body without
                      the op): in-place body turn_to steps to yaw0-35, yaw0, yaw0+35 deg (then back), a settle and a
                      snapshot at each. Executor `virtual` (lite/tests only) computes the same views without moving.
                      The scan executor is named in every result (`scan_executor`).
    wait_and_observe  PLAN §5.1: observe first (scan only if no body lease is held, the robot is at a keypoint
                      and the last scan there is older than 5 s; else a glance); `changed` if the look differs
                      from the previous one there; timeout 0 -> `unchanged`; else wait for a wake (event) or a
                      perception change until timeout -> `timed_out`.

Everything reported carries the world's source label (`isaac-gt` / `lite-gt`) and the method (`gt-geometric`).
"""

from __future__ import annotations

import asyncio
import collections
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from api.execution import Execution, ResultHandle
from api.observation import LookData, RobotObservation, ViewSpec, glance_id
from api.results import WaitResult, finish
from world import coords

from .common import EventSink, HaltGate, body_acquire, body_fence, body_release, start_execution


@dataclass(frozen=True)
class ScanConfig:
    executor: str = "turn_in_place"            # waist (M2b B.5) | turn_in_place (M2a INTERIM) | virtual (lite/tests)
    yaws_deg: tuple[float, ...] = (-35.0, 0.0, 35.0)
    settle_s: float = 0.4
    turn_tol_deg: float = 6.0
    min_interval_s: float = 5.0
    glance_period_s: float = 0.1
    waist_move_s: float = 0.8                  # the body's scan op: min-jerk move per yaw (its default)
    waist_hold_s: float = 1.0                  # hold per yaw; the view is sampled in its middle (body default 0.8)
    waist_grace_s: float = 3.0                 # after the last hold: the return move and the terminal event
    waist_rest_s: float = 2.0                  # then wait (at most this) for the base to be still again: SONIC sways
                                               # the pelvis while the waist returns, and a reachability check right
                                               # after the scan would read it as base_moving (live, main box)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScanConfig":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for k in ("executor",):
            if k in d:
                kw[k] = str(d[k])
        for k in ("settle_s", "turn_tol_deg", "min_interval_s", "glance_period_s", "waist_move_s", "waist_hold_s",
                  "waist_grace_s", "waist_rest_s"):
            if k in d:
                kw[k] = float(d[k])
        if "yaws_deg" in d:
            kw["yaws_deg"] = tuple(float(v) for v in d["yaws_deg"])
        return cls(**kw)


def _snapshot(look: dict) -> dict[str, str]:
    """{object id: where} of everything a look saw (hands included)."""
    out = {it["id"]: sname for sname, items in (look.get("surfaces") or {}).items() for it in items}
    for arm, oid in (look.get("hands") or {}).items():
        if oid:
            out[oid] = f"hand:{arm}"
    return out


def describe_changes(before: dict[str, str], after: dict[str, str], views_cover: Callable[[str], bool] | None = None
                     ) -> list[str]:
    """THOR-style change lines: 'saw book_2 on bedroom_desk_1', 'alarm_clock_1 no longer on bedroom_dresser_1a'."""
    out = []
    for oid, w in sorted(after.items()):
        if before.get(oid) != w:
            out.append(f"saw {oid} {'in' if w.startswith('hand') else 'on'} {w}")
    for oid, w in sorted(before.items()):
        if oid not in after and (views_cover is None or views_cover(oid)):
            out.append(f"{oid} no longer on {w}")
    return out


class ObservationService:
    def __init__(self, world: Any, body: Any, clock: Any, cfg: ScanConfig | None = None, *, nav: Any = None,
                 frames: Any = None, events: EventSink | None = None, gate: HaltGate | None = None):
        self.world = world
        self.body = body
        self.clock = clock
        self.cfg = cfg or ScanConfig()
        self.nav = nav
        self.frames = frames
        self.events = events or EventSink()
        self.gate = gate or HaltGate()
        self._glance_rev = 0
        self._glance_t = -math.inf
        self._glance_snap: dict[str, str] | None = None
        self._glances: collections.OrderedDict[int, dict] = collections.OrderedDict()
        self._scans: dict[str | None, tuple[float, set[str]]] = {}
        self._last: dict[str | None, dict[str, str]] = {}
        self._obs: collections.OrderedDict[str, RobotObservation] = collections.OrderedDict()
        self._extra: dict[str, dict] = {}
        self._seq = 0
        self.head_rev = 0

    # ------------------------------------------------------------------ cheap reads
    def perception(self) -> dict:
        return self.world.perception()

    def latest_frame(self, camera: str = "head"):
        if self.frames is not None:
            fn = getattr(self.frames, "latest", None) or getattr(self.frames, "latest_frame", None)
            return fn(camera) if fn else None
        return self.world.latest_frame(camera)

    def _here(self) -> str | None:
        if self.nav is not None:
            return self.nav.at()[0]
        kp, _ = self.world.static_map().nearest_keypoint(*self.world.robot_pose().xy_yaw[:2], 0.30)
        return kp

    def glance_record(self) -> str:
        now = self.clock.now()
        if now - self._glance_t >= self.cfg.glance_period_s:
            self._glance_t = now
            look = self.world.look([], at=self._here())
            snap = _snapshot(look)
            if snap != self._glance_snap or self._glance_rev == 0:
                self._glance_rev += 1
                self._glance_snap = snap
                self._glances[self._glance_rev] = look
                while len(self._glances) > 50:
                    self._glances.popitem(last=False)
        return glance_id(self._glance_rev)

    def glance(self, rev: int | None = None) -> dict | None:
        return self._glances.get(self._glance_rev if rev is None else rev)

    def seen_here(self, keypoint: str | None) -> set[str]:
        """What the last scan at this keypoint saw (plus later glances there): reachability's not_seen_here gate."""
        rec = self._scans.get(keypoint)
        return set(rec[1]) if rec else set()

    def last_scan_age(self, keypoint: str | None) -> float:
        rec = self._scans.get(keypoint)
        return math.inf if rec is None else self.clock.now() - rec[0]

    def get(self, observation_id: str) -> RobotObservation | None:
        return self._obs.get(observation_id)

    def extra_of(self, observation_id: str) -> dict:
        """Labels of an observation: camera, method, source, scan_executor (+ scan notes)."""
        return dict(self._extra.get(observation_id, {}))

    def data_of(self, obs: RobotObservation) -> dict:
        d = obs.data()
        d.update(self.extra_of(obs.observation_id))
        return d

    # ------------------------------------------------------------------ observe
    async def observe(self, mode: str, *, at: str | None = None, execution: Execution | None = None,
                      handle: ResultHandle | None = None) -> RobotObservation:
        at = at if at is not None else self._here()
        extra: dict[str, Any] = {"camera": getattr(getattr(self.world, "cam", None), "name", "head"),
                                 "method": "gt-geometric", "source": self.world.source}
        if mode == "scan":
            rows, views, info = await self._scan(handle, execution)
            extra.update(info)
            if rows is None:                          # the body could not turn: a glance instead (labelled)
                mode = "glance"
        if mode != "scan":
            rows, views = [self._live_detections()], []
        methods = sorted({getattr(d, "method", "gt-geometric") for row in rows for d in row})
        if methods:
            extra["method"] = "+".join(methods)
        look_d = self.world.look(views, at=at, detections=rows)
        snap = _snapshot(look_d)
        seen = {oid for oid, w in snap.items() if not w.startswith("hand")}
        now = self.clock.now()
        if mode == "scan":
            self._scans[at] = (now, seen)
        elif at in self._scans:                       # a glance adds to the last scan of this spot (THOR)
            t, prev = self._scans[at]
            self._scans[at] = (t, prev | seen)
        self._seq += 1
        oid = execution.execution_id if execution is not None else f"obs-{self._seq:06d}"
        look = LookData(at=look_d["at"], surfaces=look_d["surfaces"], landmarks=look_d["landmarks"],
                        views=[ViewSpec(**{k: v[k] for k in ViewSpec.__dataclass_fields__ if k in v})
                               for v in look_d["views"]], hands=look_d["hands"])
        src = "lite-gt" if self.world.source.startswith("lite") else "isaac-gt"
        obs = RobotObservation(observation_id=oid, timestamp=round(now, 3), mode=mode, look=look,   # type: ignore[arg-type]
                               head_rev=self.head_rev, source=src)                                  # type: ignore[arg-type]
        self._obs[oid] = obs
        self._extra[oid] = extra
        while len(self._obs) > 100:
            old, _ = self._obs.popitem(last=False)
            self._extra.pop(old, None)
        self.events.emit("visual_observation", observation_id=oid, mode=mode, at=at, sees=sorted(seen),
                         views=len(look.views), **{k: v for k, v in extra.items() if k != "source"})
        return obs

    def _live_detections(self) -> list:
        """What the camera sees from where the robot really stands: the world's most faithful source (on Isaac,
        P1's instance-id segmentation when it has it, P1.6; GT geometry otherwise)."""
        try:
            return self.world.detections(method="best")
        except TypeError:                              # a world without the `method` keyword
            return self.world.detections()

    def _body_supports(self, what: str) -> bool:
        fn = getattr(self.body, "supports", None)
        try:
            return bool(fn(what)) if callable(fn) else False
        except Exception:  # noqa: BLE001
            return False

    def _carrying(self) -> bool:
        """CarryLock engaged on the body, or anything in a hand (GT attach): no waist scan then."""
        try:
            if any(self.world.hands().values()):
                return True
        except Exception:  # noqa: BLE001
            pass
        fn = getattr(self.body, "arm_state", None)
        try:
            return bool(((fn() if callable(fn) else {}).get("carry") or {}).get("engaged"))
        except Exception:  # noqa: BLE001
            return False

    async def _scan(self, handle: ResultHandle | None, execution: Execution | None = None):
        """Returns (rows, views, info) or (None, None, info) when the scan could not run."""
        c = self.cfg
        p0 = self.world.robot_pose()
        yaw0 = p0.yaw
        at = self._here()
        k = self.world.static_map().keypoints.get(at) if at else None
        if k is not None:
            yaw0 = k.yaw
        if c.executor == "virtual":
            views, rows = [], []
            for dy in c.yaws_deg:
                cp = self.world.camera_pose(yaw_offset=coords.wrap_pi(yaw0 - p0.yaw + math.radians(dy)))
                views.append(self.world.view_spec(cp))
                rows.append(self.world.detections(cam_pose=cp))
            return rows, views, {"scan_executor": "virtual (no motion; lite/tests only)"}
        if c.executor == "waist":
            if self._body_supports("scan"):
                return await self._waist_scan(handle, execution)
            interim = "INTERIM: in-place turns (the body has no waist `scan` op, B.5)"
        else:
            interim = "INTERIM: in-place turns (profile scan_executor turn_in_place; the target is `waist`)"
        info: dict[str, Any] = {"scan_executor": "turn_in_place", "scan_note": interim}
        epoch = execution.control_epoch if execution is not None else self.gate.epoch
        rows, views = [], []
        for i, dy in enumerate(c.yaws_deg):
            if (handle is not None and handle.cancel_requested) or self.gate.halted_since(epoch):
                info["scan_interrupted"] = "cancel" if handle is not None and handle.cancel_requested else "halt"
                break
            target = coords.wrap_pi(yaw0 + math.radians(dy))
            cur = self.world.robot_pose().yaw
            if abs(coords.ang_diff(target, cur)) > math.radians(c.turn_tol_deg):
                op = await self.body.turn_to(target, tol_deg=c.turn_tol_deg)
                res = await self._await_turn(op, handle, epoch)
                if (handle is not None and handle.cancel_requested) or self.gate.halted_since(epoch):
                    info["scan_interrupted"] = "cancel" if handle is not None and handle.cancel_requested else "halt"
                    break
                if res.get("state") != "succeeded":
                    info.setdefault("scan_turn_failures", []).append(
                        {"yaw_deg": dy, "state": res.get("state"), "reason": (res.get("data") or {}).get("reason")})
                    if i == 0 and not rows:
                        info["scan_fallback"] = "glance"
                        return None, None, info
            await self.clock.sleep(c.settle_s)
            cp = self.world.camera_pose()
            views.append(self.world.view_spec(cp))
            rows.append(self._live_detections())          # the body really turned: the live view
        # face the surface again
        if abs(coords.ang_diff(yaw0, self.world.robot_pose().yaw)) > math.radians(c.turn_tol_deg) and \
                not self.gate.halted_since(epoch):
            op = await self.body.turn_to(yaw0, tol_deg=c.turn_tol_deg)
            await op.result()
        if not rows:
            return None, None, info
        return rows, views, info

    async def _waist_scan(self, handle: ResultHandle | None, execution: Execution | None):
        """B.5 through the body's `scan` op: waist yaw to each of yaws_deg (move, hold), the live view sampled in the
        middle of each hold, the achieved yaws from the body's `scan.hold` events. Wall time (the body is)."""
        c = self.cfg
        info: dict[str, Any] = {"scan_executor": "waist",
                                "scan_note": "waist yaw only (one row); SONIC also turns the shoulders with the waist"}
        if self._carrying():
            info["scan_fallback"] = "glance"
            info["scan_note"] = ("no waist scan while carrying: SONIC turns the shoulders with the waist and the palms "
                                 "move 6-10 cm (docs/arm_tracking.md §8.5)")
            return None, None, info
        ce = execution.control_epoch if execution is not None else self.gate.epoch
        if self.gate.latched:
            info["scan_interrupted"] = "halt"
            return None, None, info
        lease = await body_acquire(self.body, execution, "ARM_SCRIPT") if execution is not None else {"ok": True}
        if not lease.get("ok"):
            info["scan_fallback"] = "glance"
            info["scan_lease"] = {"refused": lease.get("reason"), "meaning": lease.get("meaning")}
            return None, None, info
        rows, views, holds = [], [], []
        op = None
        try:
            fence = body_fence(self.body, execution) if execution is not None else {}
            op = await self.body.scan(list(c.yaws_deg), move_s=c.waist_move_s, hold_s=c.waist_hold_s, fence=fence)
            t0 = time.monotonic()
            op.on_progress(lambda ev: holds.append(dict(ev.get("data") or {}))
                           if (ev.get("data") or {}).get("kind") == "scan.hold" else None)
            period = c.waist_move_s + c.waist_hold_s
            for i, dy in enumerate(c.yaws_deg):
                t_mid = t0 + i * period + c.waist_move_s + 0.5 * c.waist_hold_s
                while time.monotonic() < t_mid and not op.done:
                    if (handle is not None and handle.cancel_requested) or self.gate.halted_since(ce):
                        break
                    await asyncio.sleep(0.02)
                if (handle is not None and handle.cancel_requested) or self.gate.halted_since(ce):
                    info["scan_interrupted"] = "cancel" if handle is not None and handle.cancel_requested else "halt"
                    if info["scan_interrupted"] == "cancel":
                        op.cancel("cancel")                      # the scan ends where it is (arm end)
                    break
                if op.done:
                    break
                cp = self.world.camera_pose()                    # the live torso: the head camera turned with it
                views.append(self.world.view_spec(cp))
                rows.append(self._live_detections())
            try:
                res = await asyncio.wait_for(op.result(), max(0.5, (t0 + len(c.yaws_deg) * period + c.waist_grace_s)
                                                             - time.monotonic()))
            except asyncio.TimeoutError:
                res = {"state": "failed", "data": {"reason": "no terminal event"}}
        finally:
            if execution is not None:
                await body_release(self.body, execution)
        info["scan_rest_s"] = await self._wait_rest(c.waist_rest_s)
        d = res.get("data") or {}
        info["scan_body"] = {"state": res.get("state"), "reason": d.get("reason"),
                             **{k: d.get(k) for k in ("yaw_err_deg_max", "arm_dev_rad_max", "base_shift_m",
                                                      "waist_roll_pitch_dev_rad_max") if k in d},
                             "op": getattr(op, "id", None)}
        info["scan_holds"] = [{k: h.get(k) for k in ("i", "yaw_deg", "yaw_cmd_deg", "pitch_deg", "yaw_err_deg")}
                              for h in holds]
        if not rows:
            if res.get("state") == "failed" and not info.get("scan_interrupted"):
                info["scan_fallback"] = "glance"
            return None, None, info
        return rows, views, info

    async def _wait_rest(self, max_s: float, speed: float = 0.05, wz: float = 0.1, n: int = 3) -> float | None:
        """Seconds until the base was still for n samples (50 ms apart), None if not within max_s (wall time)."""
        t0 = time.monotonic()
        run = 0
        while time.monotonic() - t0 < max_s:
            p = self.world.robot_pose()
            run = run + 1 if (p.speed <= speed and abs(p.wz) <= wz) else 0
            if run >= n:
                return round(time.monotonic() - t0, 2)
            await asyncio.sleep(0.05)
        return None

    async def _await_turn(self, op: Any, handle: ResultHandle | None, epoch: int, grace_s: float = 1.5) -> dict:
        """A scan turn that stops at once on cancel or halt: the in-flight body op is cancelled (body stop: planner
        IDLE, never command{stop}), so a cancelled scan ends within the harness's grace instead of running on until
        the runtime halts the body (docs/bringup.md §7 item 1)."""
        while not getattr(op, "done", True):
            if (handle is not None and handle.cancel_requested) or self.gate.halted_since(epoch):
                if not self.gate.halted_since(epoch):
                    try:
                        op.cancel("cancel")
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    return await self.clock.wait_for(op.result(), grace_s)
                except asyncio.TimeoutError:
                    return {"state": "failed", "data": {"reason": "no terminal event after stop"}}
            await self.clock.sleep(0.05)
        return await op.result()

    def start(self, execution: Execution) -> ResultHandle:
        """The internal `observe` tool (and `look` with WL_LOOK_TOOL=1)."""
        mode = str(execution.args.get("mode") or execution.action or ("scan" if execution.tool_name == "look"
                                                                     else "glance"))

        async def work(h: ResultHandle):
            obs = await self.observe(mode, at=self._here(), execution=execution, handle=h)
            data = self.data_of(obs)
            status = "cancelled" if h.cancel_requested and data.get("scan_interrupted") == "cancel" else "succeeded"
            return finish(execution, status, data, t_end=round(self.clock.now(), 3),
                          observation_id=obs.observation_id)

        return start_execution(execution, work, clock=self.clock, observation_id=self.glance_record)

    # ------------------------------------------------------------------ wait_and_observe (PLAN §5.1)
    def start_wait(self, execution: Execution, *, wake: asyncio.Event | None = None, lease_held: bool = False,
                   changed_fn: Callable[[RobotObservation], list[str]] | None = None) -> ResultHandle:
        timeout_s = float(execution.args.get("timeout_s", 10.0) if execution.args.get("timeout_s") is not None
                          else 10.0)

        async def work(h: ResultHandle):
            at = self._here()
            stationary = self.world.robot_pose().speed < 0.05 and not (self.nav is not None and self.nav.moving)
            scan = (not lease_held and at is not None and stationary
                    and self.last_scan_age(at) > self.cfg.min_interval_s)
            before = dict(self._last.get(at, {}))
            obs = await self.observe("scan" if scan else "glance", at=at, execution=execution, handle=h)
            snap = _snapshot(obs.look.to_dict())
            if changed_fn is not None:
                changes = list(changed_fn(obs))
            else:
                changes = describe_changes(before, snap) if (before or at in self._last) else []
                if obs.mode == "glance":            # a glance never asserts absence
                    changes = [c for c in changes if "no longer" not in c]
            self._last[at] = snap if obs.mode == "scan" else {**before, **snap}
            data = lambda st, summ, ch: {**WaitResult(st, obs.observation_id, summ, ch).__dict__,   # noqa: E731
                                         "look": self.data_of(obs), "observed": obs.mode}
            if h.cancel_requested:
                return finish(execution, "cancelled", data("cancelled", None, []), t_end=round(self.clock.now(), 3),
                              observation_id=obs.observation_id)
            if changes:
                return finish(execution, "changed", data("changed", "; ".join(changes), changes),
                              t_end=round(self.clock.now(), 3), observation_id=obs.observation_id)
            if timeout_s <= 0:
                return finish(execution, "unchanged", data("unchanged", "looked; nothing new", []),
                              t_end=round(self.clock.now(), 3), observation_id=obs.observation_id)
            t_end = self.clock.now() + timeout_s
            base = self._visible_snapshot()
            while self.clock.now() < t_end:
                if h.cancel_requested:
                    return finish(execution, "cancelled", data("cancelled", None, []),
                                  t_end=round(self.clock.now(), 3), observation_id=obs.observation_id)
                if wake is not None and wake.is_set():
                    why = getattr(wake, "why", None) or "woken"
                    return finish(execution, "changed", data("changed", str(why), [str(why)]),
                                  t_end=round(self.clock.now(), 3), observation_id=obs.observation_id)
                now_snap = self._visible_snapshot()
                if now_snap != base:
                    ch = [c for c in describe_changes(base, now_snap) if "no longer" not in c]
                    if ch:
                        return finish(execution, "changed", data("changed", "; ".join(ch), ch),
                                      t_end=round(self.clock.now(), 3), observation_id=self.glance_record())
                    base = now_snap
                await self.clock.sleep(0.1)
            return finish(execution, "timed_out", data("timed_out", None, []), t_end=round(self.clock.now(), 3),
                          observation_id=obs.observation_id)

        return start_execution(execution, work, clock=self.clock, observation_id=self.glance_record)

    def _visible_snapshot(self) -> dict[str, str]:
        return {d.id: d.where or "" for d in self.world.detections() if d.kind == "object"}
