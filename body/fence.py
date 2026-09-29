"""Fences, the halt latch, leases and runtime sessions (M2b B.1-B.3; docs/contracts/m1.md §3.10-§3.12; PLAN §5.5,
§5.6, §6.6).

Bookkeeping only, no sockets and no motion: BodyService applies the side effects (SonicMux latch, ArmChannel latch,
events). The control thread and the halt-lane thread both call into it, so every method takes `lock`.

Rules (the contract states them for clients):
- Fence fields `execution_id`, `generation`, `control_epoch` (and the optional heartbeat `session`) ride on any
  request, in `args` or at the top level of the envelope (`args` wins). They are optional: M1 tools send none.
- Halt latch: halt{epoch} latches. While latched, a gated command whose control_epoch is missing or <= halt_epoch is
  rejected `halted`; one with a higher control_epoch clears the latch (an implicit resume: the runtime only uses a
  newer epoch after a resume or a correction). resume{epoch >= halt_epoch} clears it explicitly.
  After the resume, a fenced command with control_epoch <= halt_epoch is stale (`stale_command`): every execution of
  that epoch was ended by the halt. A halt whose epoch is <= the last resume epoch is a late re-send: ignored.
- Generation: a fenced command whose generation is below the highest generation the body has accepted is stale
  (`stale_command`, published as body.stale_command). A correction bumps the generation and ends older executions.
- Leases: acquire{execution_id, generation, control_epoch, mode}. While a lease is held, gated commands must carry
  execution_id == lease.owner, else `body_busy` (M1 tools without an execution_id too). An acquire from another owner
  is `body_busy`, unless its generation is newer: then the old lease is revoked (`superseded`). A halt revokes every
  lease of its epoch or older; release{execution_id} or a lost runtime session ends one.
- Runtime sessions: hello{session, watchdog_s} registers one; `ping{session}` or any request carrying `session`
  refreshes it. A session silent for watchdog_s is lost once (the service then holds the robot, §3.12).
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import asdict, dataclass

LEASE_MODES = ("ANY", "LOCOMOTION", "ARM_STREAM", "ARM_SCRIPT", "MANIP")


class FenceReject(Exception):
    """A gated command refused by a fence. `stale` ones are also published as body.stale_command."""

    def __init__(self, reason: str, data: dict | None = None, stale: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.data = data or {}
        self.stale = stale


def _int_or_none(v, key: str):
    if v is None:
        return None
    if isinstance(v, bool):
        raise FenceReject("bad_args", {"arg": key, "error": "integer"})
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise FenceReject("bad_args", {"arg": key, "error": "integer"})
    if f != int(f):
        raise FenceReject("bad_args", {"arg": key, "error": "integer"})
    return int(f)


@dataclass(frozen=True)
class Fence:
    execution_id: str | None = None
    generation: int | None = None
    control_epoch: int | None = None
    session: str | None = None

    @classmethod
    def parse(cls, req: dict | None, args: dict | None) -> "Fence":
        req, args = req or {}, args or {}

        def pick(k):
            return args[k] if args.get(k) is not None else req.get(k)

        eid = pick("execution_id")
        sess = pick("session")
        return cls(None if eid is None else str(eid), _int_or_none(pick("generation"), "generation"),
                   _int_or_none(pick("control_epoch"), "control_epoch"), None if sess is None else str(sess))

    @property
    def fenced(self) -> bool:
        return self.execution_id is not None or self.generation is not None or self.control_epoch is not None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Lease:
    lease_id: str
    owner: str               # execution_id
    generation: int
    control_epoch: int
    mode: str = "ANY"
    session: str | None = None
    t_wall: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Fences:
    def __init__(self):
        self.lock = threading.RLock()
        self.gen_floor: int | None = None       # highest generation accepted
        self.epoch_seen: int | None = None      # highest control_epoch accepted
        self.halt_epoch: int | None = None
        self.resume_epoch: int | None = None
        self.latched = False
        self.latch_reason: str | None = None
        self.t_latch_mono: float | None = None
        self.lease: Lease | None = None
        self.sessions: dict[str, dict] = {}
        self.stats = collections.Counter()

    # -- halt latch ----------------------------------------------------------------------------------
    def latch(self, epoch: int, reason: str = "halt", internal: bool = False) -> tuple[str, int]:
        """Returns (kind, epoch): kind 'new' (latched now), 'repeat' (already latched at >= epoch: re-ack) or 'stale'
        (a late re-send of a halt that a resume already cleared: ignored). An internal halt (runtime lost) latches
        at max(epoch, every epoch the body has seen) and is never stale."""
        with self.lock:
            if internal:
                epoch = max([e for e in (epoch, self.halt_epoch, self.resume_epoch, self.epoch_seen)
                             if e is not None])
            elif self.resume_epoch is not None and epoch <= self.resume_epoch:
                self.stats["halts_stale"] += 1
                return "stale", epoch
            if self.latched and self.halt_epoch is not None and epoch <= self.halt_epoch:
                self.stats["halts_repeat"] += 1
                return "repeat", self.halt_epoch
            self.halt_epoch = epoch if self.halt_epoch is None else max(self.halt_epoch, epoch)
            self.latched = True
            self.latch_reason = reason
            self.t_latch_mono = time.monotonic()
            self.stats["halts"] += 1
            return "new", self.halt_epoch

    def unlatch(self, epoch: int) -> bool:
        """resume{epoch}: clears the latch (returns whether it was latched). An epoch older than the halt is stale."""
        with self.lock:
            if self.halt_epoch is not None and epoch < self.halt_epoch:
                raise FenceReject("stale_command", {"why": "resume_epoch", "epoch": epoch,
                                                    "halt_epoch": self.halt_epoch}, stale=True)
            was = self.latched
            self.latched = False
            self.latch_reason = None
            self.resume_epoch = epoch if self.resume_epoch is None else max(self.resume_epoch, epoch)
            self.epoch_seen = epoch if self.epoch_seen is None else max(self.epoch_seen, epoch)
            self.stats["resumes"] += 1
            return was

    # -- gate ---------------------------------------------------------------------------------------
    def check(self, op: str, fence: Fence, lease_exempt: bool = False) -> bool:
        """Raise FenceReject if the command may not run. Returns True when its control_epoch clears the latch
        (implicit resume). Changes nothing: the caller runs accept() (and the resume) once the command is taken."""
        with self.lock:
            implicit = False
            ce = fence.control_epoch
            if self.latched:
                if ce is None or ce <= self.halt_epoch:
                    self.stats["rejected_halted"] += 1
                    raise FenceReject("halted", {"halt_epoch": self.halt_epoch, "control_epoch": ce,
                                                 "hint": "resume{epoch} (or a command with a newer control_epoch)"})
                implicit = True
            elif ce is not None and self.halt_epoch is not None and ce <= self.halt_epoch:
                self.stats["stale_epoch"] += 1
                raise FenceReject("stale_command", {"why": "control_epoch", "control_epoch": ce,
                                                    "halt_epoch": self.halt_epoch}, stale=True)
            g = fence.generation
            if g is not None and self.gen_floor is not None and g < self.gen_floor:
                self.stats["stale_generation"] += 1
                raise FenceReject("stale_command", {"why": "generation", "generation": g,
                                                    "generation_floor": self.gen_floor}, stale=True)
            if self.lease is not None and not lease_exempt and fence.execution_id != self.lease.owner:
                self.stats["busy"] += 1
                raise FenceReject("body_busy", {"lease": self.lease.to_dict(), "execution_id": fence.execution_id})
            return implicit

    def accept(self, fence: Fence) -> None:
        with self.lock:
            if fence.generation is not None:
                self.gen_floor = fence.generation if self.gen_floor is None else max(self.gen_floor, fence.generation)
            if fence.control_epoch is not None:
                self.epoch_seen = fence.control_epoch if self.epoch_seen is None else \
                    max(self.epoch_seen, fence.control_epoch)

    # -- leases -------------------------------------------------------------------------------------
    def acquire(self, fence: Fence, mode: str = "ANY") -> tuple[Lease, Lease | None, bool]:
        """Returns (lease, superseded_lease_or_None, implicit_resume). Raises FenceReject."""
        if fence.execution_id is None or fence.generation is None or fence.control_epoch is None:
            raise FenceReject("bad_args", {"need": "execution_id, generation, control_epoch"})
        mode = str(mode or "ANY").upper()
        if mode not in LEASE_MODES:
            raise FenceReject("bad_args", {"arg": "mode", "modes": list(LEASE_MODES)})
        with self.lock:
            implicit = self.check("acquire", fence, lease_exempt=True)
            cur = self.lease
            if cur is not None and cur.owner == fence.execution_id:
                cur.mode = mode
                return cur, None, implicit
            old = None
            if cur is not None:
                if fence.generation > cur.generation:
                    old = cur
                    self.stats["superseded"] += 1
                else:
                    self.stats["busy"] += 1
                    raise FenceReject("body_busy", {"lease": cur.to_dict()})
            self.lease = Lease(f"lease-{uuid.uuid4().hex[:8]}", fence.execution_id, fence.generation,
                               fence.control_epoch, mode, fence.session, time.time())
            self.stats["acquired"] += 1
            return self.lease, old, implicit

    def release(self, execution_id: str | None = None, lease_id: str | None = None) -> Lease | None:
        with self.lock:
            cur = self.lease
            if cur is None:
                return None
            if (execution_id is not None and execution_id != cur.owner) or \
                    (lease_id is not None and lease_id != cur.lease_id) or (execution_id is None and lease_id is None):
                raise FenceReject("not_owner", {"lease": cur.to_dict()})
            self.lease = None
            self.stats["released"] += 1
            return cur

    def revoke(self, max_epoch: int | None = None) -> Lease | None:
        """Drop the lease (every lease if max_epoch is None, else only one whose control_epoch <= max_epoch)."""
        with self.lock:
            cur = self.lease
            if cur is None or (max_epoch is not None and cur.control_epoch > max_epoch):
                return None
            self.lease = None
            self.stats["revoked"] += 1
            return cur

    # -- runtime sessions ---------------------------------------------------------------------------
    def hello(self, session: str, watchdog_s: float, info: dict | None = None) -> dict:
        with self.lock:
            now = time.monotonic()
            s = {"session": session, "watchdog_s": float(watchdog_s), "t_hello": now, "t_last": now, "pings": 0,
                 "lost": False, "lost_count": 0, "info": info or {}}
            self.sessions[session] = s
            return dict(s)

    def touch(self, session: str | None) -> dict | None:
        """A ping or any request carrying `session`. Returns the session if it was lost and is back."""
        if session is None:
            return None
        with self.lock:
            s = self.sessions.get(session)
            if s is None:
                return None
            s["t_last"] = time.monotonic()
            s["pings"] += 1
            if s["lost"]:
                s["lost"] = False
                return dict(s)
            return None

    def bye(self, session: str) -> bool:
        with self.lock:
            return self.sessions.pop(session, None) is not None

    def expired(self) -> list[dict]:
        """Sessions that just went silent for longer than their watchdog (each reported once per loss)."""
        out = []
        with self.lock:
            now = time.monotonic()
            for s in self.sessions.values():
                if not s["lost"] and now - s["t_last"] > s["watchdog_s"]:
                    s["lost"] = True
                    s["lost_count"] += 1
                    out.append({**s, "age_s": round(now - s["t_last"], 3)})
        return out

    # -- state --------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            now = time.monotonic()
            return {"latched": self.latched, "halt_epoch": self.halt_epoch, "resume_epoch": self.resume_epoch,
                    "latch_reason": self.latch_reason,
                    "latched_for_s": None if not self.latched or self.t_latch_mono is None else
                    round(now - self.t_latch_mono, 3),
                    "generation_floor": self.gen_floor, "epoch_seen": self.epoch_seen,
                    "lease": None if self.lease is None else self.lease.to_dict(),
                    "sessions": [{"session": s["session"], "watchdog_s": s["watchdog_s"], "lost": s["lost"],
                                  "age_s": round(now - s["t_last"], 3), "pings": s["pings"]}
                                 for s in self.sessions.values()],
                    "stats": dict(self.stats)}
