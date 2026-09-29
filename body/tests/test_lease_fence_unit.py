"""body/fence.py on its own: the halt latch, stale generations/epochs, leases and runtime sessions (contract §3.10-§3.12).
No sockets; the integration tests (test_halt.py, test_lease.py, test_modes.py) run the same rules over the wire."""

import time

import pytest

from body.fence import Fence, FenceReject, Fences


def F(eid=None, gen=None, ep=None, session=None):
    return Fence(eid, gen, ep, session)


def test_parse_top_level_and_args():
    f = Fence.parse({"execution_id": "top", "generation": 2, "control_epoch": 3}, {"execution_id": "arg"})
    assert (f.execution_id, f.generation, f.control_epoch) == ("arg", 2, 3) and f.fenced
    assert not Fence.parse({}, {}).fenced
    assert Fence.parse({}, {"generation": 4.0}).generation == 4
    for bad in ("x", 1.5, True):
        with pytest.raises(FenceReject) as e:
            Fence.parse({}, {"control_epoch": bad})
        assert e.value.reason == "bad_args"


def test_halt_latch_rejects_old_epochs_then_resume():
    f = Fences()
    assert f.latch(5, "halt") == ("new", 5)
    assert f.latch(5, "halt") == ("repeat", 5)            # a re-send: re-ack, no new latch
    assert f.latch(3, "halt") == ("repeat", 5)
    for fence in (F(), F("e1", 1, 5), F("e1", 1, 4)):      # no epoch (an M1 tool) or <= halt_epoch
        with pytest.raises(FenceReject) as e:
            f.check("walk", fence)
        assert e.value.reason == "halted" and not e.value.stale
    with pytest.raises(FenceReject) as e:
        f.unlatch(4)                                       # a resume older than the halt
    assert e.value.reason == "stale_command" and e.value.stale
    assert f.unlatch(6) is True and not f.latched
    assert f.latch(6, "halt")[0] == "stale"                 # a late re-send after the resume is ignored
    assert f.latch(5, "halt")[0] == "stale"
    with pytest.raises(FenceReject) as e:
        f.check("walk", F("e1", 1, 5))                     # that epoch's executions were ended by the halt
    assert e.value.reason == "stale_command" and e.value.data["why"] == "control_epoch"
    assert f.check("walk", F()) is False                   # unfenced M1 tools work again after the resume
    assert f.check("walk", F("e2", 1, 7)) is False


def test_implicit_resume_by_newer_epoch():
    f = Fences()
    f.latch(2, "halt")
    assert f.check("walk", F("e", 1, 3)) is True           # the caller resumes, then accepts
    assert f.latched                                       # check() itself changes nothing


def test_internal_halt_latches_above_everything_seen():
    f = Fences()
    f.accept(F("e", 3, 9))
    f.latch(4, "halt")
    f.unlatch(9)
    kind, ep = f.latch(-1, "runtime_lost", internal=True)
    assert kind == "new" and ep == 9 and f.latched
    with pytest.raises(FenceReject):
        f.check("walk", F("e", 3, 9))
    assert f.unlatch(9) is True


def test_generation_floor():
    f = Fences()
    f.accept(F("e", 4, 1))
    with pytest.raises(FenceReject) as e:
        f.check("go_to", F("old", 3, 1))
    assert e.value.reason == "stale_command" and e.value.stale and e.value.data["generation_floor"] == 4
    assert f.check("go_to", F("new", 4, 1)) is False
    assert f.check("go_to", F("newer", 5, 1)) is False


def test_lease_busy_supersede_release_revoke():
    f = Fences()
    with pytest.raises(FenceReject) as e:
        f.acquire(F("a", None, 1))
    assert e.value.reason == "bad_args"
    lease, old, _ = f.acquire(F("a", 1, 1), "LOCOMOTION")
    assert lease.owner == "a" and old is None and lease.mode == "LOCOMOTION"
    again, old, _ = f.acquire(F("a", 1, 1), "ARM_STREAM")      # re-acquire by the owner: same lease
    assert again.lease_id == lease.lease_id and again.mode == "ARM_STREAM" and old is None
    with pytest.raises(FenceReject) as e:
        f.acquire(F("b", 1, 1))
    assert e.value.reason == "body_busy" and e.value.data["lease"]["owner"] == "a"
    for fence in (F("b", 1, 1), F()):                          # a second owner, or an M1 tool, while leased
        with pytest.raises(FenceReject) as e:
            f.check("walk", fence)
        assert e.value.reason == "body_busy"
    assert f.check("walk", F("a", 1, 1)) is False
    assert f.check("stop", F(), lease_exempt=True) is False
    new, old, _ = f.acquire(F("c", 2, 2))                      # a newer generation supersedes
    assert new.owner == "c" and old.owner == "a"
    with pytest.raises(FenceReject) as e:
        f.release("a")
    assert e.value.reason == "not_owner"
    assert f.release("c").owner == "c" and f.lease is None
    assert f.release("c") is None
    f.acquire(F("d", 2, 3))
    assert f.revoke(max_epoch=2) is None and f.lease is not None   # a halt at epoch 2 leaves an epoch-3 lease
    assert f.revoke(max_epoch=3).owner == "d" and f.lease is None
    with pytest.raises(FenceReject) as e:
        f.acquire(F("x", 2, 1, None), "FLY")
    assert e.value.reason == "bad_args"


def test_sessions_watchdog():
    f = Fences()
    f.hello("rt1", 0.1)
    assert f.expired() == []
    time.sleep(0.15)
    lost = f.expired()
    assert [s["session"] for s in lost] == ["rt1"] and lost[0]["age_s"] >= 0.1
    assert f.expired() == []                                   # reported once per loss
    back = f.touch("rt1")
    assert back is not None and back["session"] == "rt1"
    assert f.touch("rt1") is None and f.touch("unknown") is None and f.touch(None) is None
    assert f.bye("rt1") and not f.bye("rt1")
    snap = f.snapshot()
    assert snap["sessions"] == [] and snap["latched"] is False
