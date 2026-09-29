"""LiteBody: a kinematic stand-in for wl-body, in process, for offline runs and the `lite` profile (PLAN §6.6).

It speaks the same BodyPort as SonicBody (go_to / turn_to / stop / halt / resume / estop / state / health) and
moves LiteWorld's robot pose along the body's own planner path (body.nav_grid via world.grid) at humanoid speed:
turn in place first when the heading is far off, walk the path with the facing on its tangent, glide to a stop over
`lite_stop_s` when stopped, turn to the goal yaw, settle. Terminal results mirror the body's contract (m1.md §3.4):
succeeded / failed {reason} / canceled {reason: stop | preempted}.

Fault injection (tests, G-scenarios): `inject(fall_after_s=, stuck_after_s=, deploy_down=, stop_ignored=,
final_error_m=)`; faults latch like the real body (`fault="fallen"` until `clear_fault()`).

Executor name `lite` (everything physical is simulated; results say so).
"""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from typing import Any

from api.types import ServiceHealth
from world import coords

from .body_client import BodyOp


@dataclass
class LiteFaults:
    fall_after_s: float | None = None        # fall this long into the next motion
    stuck_after_s: float | None = None       # stop making progress this long into the next walk
    deploy_down: bool = False                # every motion fails deploy_not_running
    stop_ignored: bool = False               # a stop/cancel does nothing (tests the harness's halt escalation)
    final_error_m: float = 0.0               # arrive this far short of the goal


MAX_DT_S = 1.0                               # a longer stall than this (a debugger, a GC pause) is not integrated


class LiteBody:
    name = "lite"

    def __init__(self, world: Any, clock: Any, walking: dict | None = None, *, tick_s: float = 0.05):
        w = dict(walking or {})
        self.world = world
        self.clock = clock
        self.cruise = float(w.get("cruise_mps", 0.45))
        self.turn_rate = math.radians(float(w.get("lite_turn_dps", 45.0)))
        self.stop_s = float(w.get("lite_stop_s", 0.6))
        self.final_pos_tol = float(w.get("final_pos_tol_m", 0.25))
        self.tick = tick_s
        self.faults = LiteFaults()
        self.fault: str | None = None
        self.halt_epoch = 0
        self.latched = False
        self.estopped = False
        self._active: dict | None = None
        self._task: asyncio.Task | None = None
        self.log: list[dict] = []

    # ------------------------------------------------------------------ fault injection
    def inject(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self.faults, k, v)

    def clear_fault(self) -> None:
        self.fault = None
        p = self.world.robot_pose()
        self.world.set_robot_pose(p.x, p.y, p.yaw)

    # ------------------------------------------------------------------ motions
    def _gate(self) -> str | None:
        if self.estopped:
            return "estop"
        if self.fault:
            return f"fault:{self.fault}"
        if self.faults.deploy_down:
            return "deploy_not_running"
        if self.latched:
            return "halted"
        return None

    def _start(self, op: str, coro_fn, args: dict) -> BodyOp:
        loop = asyncio.get_running_loop()
        o = BodyOp(op, f"{op}-{uuid.uuid4().hex[:8]}", loop=loop)
        why = self._gate()
        if why:
            o.finish("failed", {"reason": why})
            return o
        # a new motion pre-empts the active one (the body's rule); it starts once the old one has settled
        prev = None
        if self._active is not None and not self._active["op"].done:
            self._active["stop"] = "preempted"
            prev = self._active["op"]
        state = {"op": o, "stop": None, "args": args, "prev": prev}
        o._cancel_fn = lambda reason: state.__setitem__("stop", "stop")
        self._active = state
        self._task = asyncio.ensure_future(self._run(coro_fn, state))
        self.log.append({"op": op, "args": args, "t": self.clock.now()})
        return o

    async def _run(self, coro_fn, state: dict) -> None:
        o: BodyOp = state["op"]
        if state.get("prev") is not None:
            await state["prev"].result()
        t0 = self.clock.now()
        p0 = self.world.robot_pose()
        try:
            res_state, data = await coro_fn(state)
        except Exception as e:  # noqa: BLE001
            res_state, data = "failed", {"reason": f"internal_error: {e!r}"}
        p = self.world.robot_pose()
        base = {"final_pose": [round(p.x, 3), round(p.y, 3), round(p.yaw, 4)],
                "start_pose": [round(p0.x, 3), round(p0.y, 3), round(p0.yaw, 4)],
                "displacement_m": round(math.hypot(p.x - p0.x, p.y - p0.y), 3),
                "yaw_change_deg": round(math.degrees(coords.ang_diff(p.yaw, p0.yaw)), 1),
                "duration_s": round(self.clock.now() - t0, 3), "executor": self.name}
        self.world.set_robot_pose(p.x, p.y, p.yaw, fallen=p.fallen)
        o.finish(res_state, {**base, **data})

    async def go_to(self, x: float, y: float, yaw: float | None = None, *, speed: float | None = None,
                    timeout_s: float | None = None, final_pos_tol: float | None = None,
                    final_yaw_tol_deg: float | None = None) -> BodyOp:
        args = {"x": x, "y": y, "yaw": yaw, "speed": speed, "timeout_s": timeout_s, "final_pos_tol": final_pos_tol}
        return self._start("go_to", self._go_to, args)

    async def turn_to(self, yaw: float, *, tol_deg: float | None = None) -> BodyOp:
        return self._start("turn_to", self._turn_to, {"yaw": yaw, "tol_deg": tol_deg})

    async def stop(self) -> BodyOp:
        if self._active is not None and not self._active["op"].done and not self.faults.stop_ignored:
            self._active["stop"] = "stop"
        loop = asyncio.get_running_loop()
        o = BodyOp("stop", f"stop-{uuid.uuid4().hex[:8]}", loop=loop)
        active = self._active

        async def settle():
            if active is not None and not self.faults.stop_ignored:
                await active["op"].result()
            p = self.world.robot_pose()
            o.finish("succeeded", {"stopped": True, "v_final": 0.0, "final_pose": [p.x, p.y, p.yaw]})
        asyncio.ensure_future(settle())
        return o

    # ------------------------------------------------------------------ motion bodies
    def _stopping(self, state: dict) -> str | None:
        if self.faults.stop_ignored:
            return None
        if self.latched:
            return "halted"
        return state.get("stop")

    def _dt(self, state: dict) -> float:
        """Sim seconds since the last motion step of this op (capped at MAX_DT_S). Motion integrates over the
        elapsed sim time, not a fixed tick, so a loaded event loop at a high clock speed does not make the body
        lag the sim clock that every timeout is measured on."""
        now = self.clock.now()
        last = state.get("t_step")
        state["t_step"] = now
        if last is None:
            return self.tick
        return min(max(0.0, now - last), MAX_DT_S)

    async def _turn(self, state: dict, target: float, tol: float = math.radians(3.0)) -> str | None:
        state["t_step"] = self.clock.now()
        while True:
            why = self._stopping(state) or self._fault_now(state)
            if why:
                return why
            p = self.world.robot_pose()
            err = coords.ang_diff(target, p.yaw)
            if abs(err) <= tol:
                self.world.set_robot_pose(p.x, p.y, target)
                return None
            step = math.copysign(min(abs(err), self.turn_rate * self._dt(state)), err)
            self.world.set_robot_pose(p.x, p.y, p.yaw + step, wz=math.copysign(self.turn_rate, err))
            await self.clock.sleep(self.tick)

    def _fault_now(self, state: dict) -> str | None:
        f = self.faults
        el = self.clock.now() - state.setdefault("t0", self.clock.now())
        if f.fall_after_s is not None and el >= f.fall_after_s:
            p = self.world.robot_pose()
            self.world.set_robot_pose(p.x, p.y, p.yaw, fallen=True)
            self.fault = "fallen"
            f.fall_after_s = None
            return "fallen"
        return None

    async def _glide(self, v: float, pts, s: float, total: float) -> float:
        """Decelerate linearly to rest over stop_s (sim time) along the path; returns the new arc length."""
        t0 = self.clock.now()
        last = t0
        while True:
            now = self.clock.now()
            el = min(now - t0, self.stop_s)
            dt = min(max(0.0, now - last), MAX_DT_S)
            last = now
            v_i = v * (1.0 - el / self.stop_s) if self.stop_s > 0 else 0.0
            s = min(total, s + v_i * dt)
            x, y = _interp(pts, s)
            p = self.world.robot_pose()
            self.world.set_robot_pose(x, y, p.yaw, vx=v_i * math.cos(p.yaw), vy=v_i * math.sin(p.yaw))
            if el >= self.stop_s:
                break
            await self.clock.sleep(self.tick)
        p = self.world.robot_pose()
        self.world.set_robot_pose(p.x, p.y, p.yaw)
        return s

    async def _go_to(self, state: dict):
        a = state["args"]
        state["t0"] = self.clock.now()
        p = self.world.robot_pose()
        plan = self.world.grid.plan((p.x, p.y), (a["x"], a["y"]))
        if not plan.ok:
            return "failed", {"reason": plan.reason or "no_path", "path_len_m": 0.0}
        pts = [(float(px), float(py)) for px, py in plan.path]
        if self.faults.final_error_m > 0 and len(pts) >= 2:
            pts = _trim(pts, self.faults.final_error_m)
        seg = _cum(pts)
        total = seg[-1]
        v = min(float(a.get("speed") or self.cruise), 0.8)
        data = {"path_len_m": round(total, 3), "replans": 0, "stuck_events": 0, "goal": [a["x"], a["y"], a["yaw"]],
                "plans": [plan.brief(20)]}
        # turn in place first when far off the path heading
        if total > 0.05:
            h0 = math.atan2(pts[min(len(pts) - 1, 3)][1] - pts[0][1], pts[min(len(pts) - 1, 3)][0] - pts[0][0])
            if abs(coords.ang_diff(h0, p.yaw)) > math.radians(60):
                why = await self._turn(state, h0, math.radians(10))
                if why:
                    return self._end(why, data)
        s = 0.0
        last_prog_t, last_prog_s = self.clock.now(), 0.0
        timeout = float(a.get("timeout_s") or 180.0)
        state["t_step"] = self.clock.now()
        while s < total - 1e-6:
            why = self._stopping(state) or self._fault_now(state)
            if why:
                if why != "fallen":
                    s = await self._glide(v, pts, s, total)
                data["walked_m"] = round(s, 3)
                return self._end(why, data)
            if self.clock.now() - state["t0"] > timeout:
                data["walked_m"] = round(s, 3)
                return "failed", {**data, "reason": "timeout"}
            stuck = self.faults.stuck_after_s is not None and self.clock.now() - state["t0"] >= self.faults.stuck_after_s
            dt = self._dt(state)
            if not stuck:
                s = min(total, s + v * dt)
            if s - last_prog_s > 0.08:
                last_prog_t, last_prog_s = self.clock.now(), s
            elif self.clock.now() - last_prog_t > 3.0:
                data["walked_m"] = round(s, 3)
                data["stuck_events"] = 1
                return "failed", {**data, "reason": "stuck"}
            x, y = _interp(pts, s)
            hx, hy = _interp(pts, min(total, s + 0.3))
            yaw = math.atan2(hy - y, hx - x) if math.hypot(hx - x, hy - y) > 1e-3 else self.world.robot_pose().yaw
            self.world.set_robot_pose(x, y, yaw, vx=v * math.cos(yaw), vy=v * math.sin(yaw))
            await self.clock.sleep(self.tick)
        data["walked_m"] = round(total, 3)
        if a.get("yaw") is not None:
            why = await self._turn(state, float(a["yaw"]))
            if why:
                return self._end(why, data)
        await self.clock.sleep(min(0.3, self.stop_s))
        p = self.world.robot_pose()
        err = math.hypot(p.x - a["x"], p.y - a["y"])
        yaw_err = abs(math.degrees(coords.ang_diff(float(a["yaw"]), p.yaw))) if a.get("yaw") is not None else 0.0
        data.update(pos_err=round(err, 3), yaw_err_deg=round(yaw_err, 1))
        tol = float(a.get("final_pos_tol") or self.final_pos_tol)
        if err > tol:
            return "failed", {**data, "reason": "final_error"}
        return "succeeded", data

    def _end(self, why: str, data: dict):
        if why in ("stop", "preempted"):
            return "canceled", {**data, "reason": why}
        if why == "halted":
            return "canceled", {**data, "reason": "stop", "halted": True}
        return "failed", {**data, "reason": why}

    async def _turn_to(self, state: dict):
        a = state["args"]
        state["t0"] = self.clock.now()
        tol = math.radians(float(a.get("tol_deg") or 6.0))
        why = await self._turn(state, float(a["yaw"]), min(tol, math.radians(3.0)))
        p = self.world.robot_pose()
        data = {"target_deg": round(math.degrees(float(a["yaw"])), 1),
                "yaw_err_deg": round(math.degrees(coords.ang_diff(float(a["yaw"]), p.yaw)), 1)}
        if why:
            return self._end(why, data)
        await self.clock.sleep(0.2)
        return "succeeded", data

    # ------------------------------------------------------------------ halt / resume / estop / state
    def halt(self, epoch: int | None = None) -> dict:
        self.halt_epoch = int(epoch) if epoch is not None else self.halt_epoch + 1
        self.latched = True
        p = self.world.robot_pose()
        return {"accepted": True, "stopped": True, "at_rest": p.speed < 0.05, "mode": "HOLD",
                "body_epoch": self.halt_epoch, "source": "lite"}

    def resume(self, epoch: int | None = None) -> None:
        self.latched = False
        if epoch is not None:
            self.halt_epoch = max(self.halt_epoch, int(epoch))

    def estop(self, reason: str) -> dict:
        self.estopped = True
        self.latched = True
        return {"accepted": True, "reason": reason, "mode": "ESTOP", "source": "lite"}

    def state(self) -> dict:
        p = self.world.robot_pose()
        act = self._active
        active = None
        if act is not None and not act["op"].done:
            active = {"id": act["op"].id, "op": act["op"].op}
        mode = ("ESTOP" if self.estopped else "FAULT" if self.fault else
                "LOCOMOTION" if active else "HOLD")
        return {"mode": mode, "in_control": not self.estopped, "fault": self.fault, "active": active,
                "pose": {"x": p.x, "y": p.y, "yaw": p.yaw, "speed": p.speed}, "halt_epoch": self.halt_epoch,
                "latched": self.latched, "deploy": {"alive": not self.faults.deploy_down},
                "age_s": 0.0, "source": "lite"}

    def health(self) -> ServiceHealth:
        if self.estopped:
            return ServiceHealth(False, "estop", "operator kill button")
        if self.faults.deploy_down:
            return ServiceHealth(False, "down", "deploy not running (injected)")
        if self.fault:
            return ServiceHealth(False, "fault", self.fault)
        return ServiceHealth(True, "ok")

    def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()


def _cum(pts) -> list[float]:
    out = [0.0]
    for i in range(1, len(pts)):
        out.append(out[-1] + math.dist(pts[i - 1], pts[i]))
    return out


def _interp(pts, s: float) -> tuple[float, float]:
    if len(pts) == 1:
        return pts[0]
    acc = 0.0
    for i in range(1, len(pts)):
        d = math.dist(pts[i - 1], pts[i])
        if acc + d >= s:
            f = 0.0 if d < 1e-9 else (s - acc) / d
            return (pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]))
        acc += d
    return pts[-1]


def _trim(pts, cut: float):
    total = _cum(pts)[-1]
    keep = max(0.0, total - cut)
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        d = math.dist(pts[i - 1], pts[i])
        if acc + d >= keep:
            out.append(_interp(pts, keep))
            return out
        out.append(pts[i])
        acc += d
    return out
