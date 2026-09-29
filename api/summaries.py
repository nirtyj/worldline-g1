"""One-line summaries the planner reads (PLAN 5.3), built from result data.

    arrived at bedroom_dresser_1a (6.2 m, 17 s, sonic_walk); looked: sees alarm_clock_1, book_2
    alarm_clock_1 visible, reachable with the right arm (0.41 m)
    book_1 not reachable from here: too_far (0.9 m); try bedroom_bed_1b
    alarm_clock_1 visible but not reachable from this exact pose: needs_reposition (0.22 m); ...
    picked alarm_clock_1 with the right hand [sonic.script.pick.v0, fallback], 9.1 s; verifying
    rejected (state): pick needs a successful check_reachability for alarm_clock right before it
"""

from __future__ import annotations

from typing import Any

from .reasons import AREA_OF_TOOL, hint

FALLBACK_TAG = "fallback"
TOO_FAR_NO_SUGGESTION = "no stand the robot knows reaches it; try another stand of that surface once, or tell the user"
# too_far after the robot's outline-wide search (services/reachability.py find_far_stance) found no stance anywhere
TOO_FAR_NOWHERE = "no spot around that surface reaches it; tell the user"


def _m(v: Any) -> str:
    try:
        return f"{float(v):.1f} m"
    except (TypeError, ValueError):
        return "? m"


def _fallback(executor: Any) -> bool:
    from .results import is_fallback
    return is_fallback(executor)


def _obj(d: dict[str, Any]) -> str:
    return str(d.get("object_id") or d.get("object_type") or d.get("object") or "the object")


def _sees(d: dict[str, Any]) -> str:
    ids = sorted({v.get("id") for items in (d.get("surfaces") or {}).values() for v in items if v.get("id")})
    return ("sees " + ", ".join(ids)) if ids else "sees nothing new"


def rejection_summary(stage: str, message: str) -> str:
    return f"rejected ({stage}): {message}"


def summarize(tool: str, status: str, data: dict[str, Any], *, action: str | None = None,
              args: dict[str, Any] | None = None) -> str:
    d = dict(data or {})
    a = dict(args or {})
    reason = d.get("reason")
    if status == "rejected":
        return rejection_summary(str(d.get("stage", "state")), str(reason or "rejected"))
    tail = ""
    if reason and status != "succeeded":
        h = hint(str(reason), AREA_OF_TOOL.get(tool))
        tail = f": {reason}" + (f" ({h})" if h else "")

    if tool == "speak":
        if status == "succeeded":
            return "said it" if d.get("speech") != "queued" else "queued to say"
        if status == "cancelled":
            played = d.get("played")
            return "speech cut" + (f" after {float(played):.0%}" if isinstance(played, (int, float)) else "") + tail
        return f"speak {status}{tail}"

    if tool == "list_locations":
        locs = d.get("locations") or []
        names = ", ".join(f"{x.get('name')} ({_m(x.get('distance_m'))})" if x.get("distance_m") is not None
                          else str(x.get("name")) for x in locs[:8])
        more = f" (+{len(locs) - 8} more)" if len(locs) > 8 else ""
        return f"{len(locs)} locations: {names}{more}" if locs else "no locations match"

    if tool == "navigate":
        loc = d.get("location") or a.get("location") or "?"
        ex = d.get("executor")
        tag = f", {ex}" + (f", {FALLBACK_TAG}" if _fallback(ex) else "") if ex else ""
        kind = d.get("kind") or action
        if status == "succeeded":
            if kind == "reposition":
                how = f" ({tag[2:]})" if tag else ""
                return (f"repositioned {float(d.get('walked_m') or 0.0):.2f} m to the reach stance{how}; "
                        f"check reachability again")
            looked = d.get("look") or {}
            seen = f"; looked: {_sees(looked)}" if looked else ""
            return (f"arrived at {d.get('at') or loc} ({_m(d.get('path_len_m') or d.get('walked_m'))}, "
                    f"{float(d.get('duration_s') or 0.0):.0f} s{tag}){seen}")
        where = f"; at {d['at']}" if d.get("at") else (f"; between {d['between'][0]} and {d['between'][1]}"
                                                         if d.get("between") else "")
        return f"navigate to {loc} {status}{tail}{where}"

    if tool == "check_reachability":
        o = _obj(d)
        if status != "succeeded":
            return f"check_reachability {status}{tail}"
        dist = f" ({_m(d.get('distance_m'))})" if d.get("distance_m") is not None else ""
        if d.get("reachable"):
            arm = d.get("preferred_arm")
            arm_txt = "either arm" if arm == "either" else f"the {arm} arm"
            return f"{o} visible, reachable with {arm_txt}{dist}"
        r = str(d.get("reason") or "not reachable")
        why = f": {d['detail']}" if d.get("detail") and r in ("needs_reposition", "too_far", "beyond_reach") else ""
        if r == "needs_reposition":
            # a far stance says where and how far (e.g. "reach stance on the other side of kitchen_dining_table_1
            # (-y side), 2.6 m walk")
            return (f"{o} visible but not reachable from this exact pose: needs_reposition{dist}{why}; "
                    f"navigate(location='reach_stance'), then check again")
        vis = "visible but " if d.get("visible") else ""
        sug = f"; try {d['suggest_location']}" if d.get("suggest_location") and r == "too_far" else ""
        h = hint(r)
        if r == "too_far" and not sug:
            # no stand to suggest: never "navigate to the suggested location" (there is none); after the
            # outline-wide search (a detail says why) not "another stand" either
            h = TOO_FAR_NOWHERE if why else TOO_FAR_NO_SUGGESTION
        return f"{o} {vis}not reachable from here: {r}{dist}{why}{sug}" + (f" ({h})" if h and not sug else "")

    if tool == "manipulate":
        act = d.get("action") or action or a.get("action") or "pick"
        o = _obj(d) if (d.get("object_id") or d.get("object_type")) else _obj(a)
        skill = d.get("skill")
        ex = d.get("executor")
        exp = "experimental" if d.get("skill_label") == "experimental" and "experimental" not in str(skill) else None
        fb = d.get("fallback_from") if isinstance(d.get("fallback_from"), dict) else {}
        after = f"after {fb.get('executor')} {fb.get('reason')}" if fb else None     # groot_then_script
        label = ", ".join(x for x in (skill, exp, FALLBACK_TAG if _fallback(ex) else None, after) if x)
        tag = f" [{label}]" if label else ""
        dur = f", {float(d['duration_s']):.1f} s" if d.get("duration_s") else ""
        if status == "succeeded":
            if act == "pick":
                arm = d.get("arm")
                return f"picked {o} with the {arm} hand{tag}{dur}; verifying" if arm else f"picked {o}{tag}{dur}; verifying"
            where = d.get("surface") or d.get("target") or "the surface"
            return f"placed {o} on {where}{tag}{dur}; verifying"
        holding = d.get("holding")
        hold = "" if holding is None else ("; still holding it" if holding else "; hand empty")
        return f"{act} {o} {status}{tag}{tail}{hold}"

    if tool == "wait_and_observe":
        st = d.get("status")
        if st == "changed":
            return f"changed: {d.get('summary') or ', '.join(d.get('changes') or []) or 'something changed'}"
        if st == "unchanged":
            return "looked; nothing new"
        if status == "timed_out":
            return f"waited {float(a.get('timeout_s', 10) or 0):.0f} s; nothing changed"
        return f"wait {status}{tail}"

    if tool == "recall":
        return str(d.get("answer") or "nothing remembered")[:300]

    if tool in ("observe", "look"):
        if status != "succeeded":
            return f"{d.get('mode') or action or 'look'} {status}{tail}"
        return f"{d.get('mode') or action or 'look'} at {d.get('at') or 'here'}: {_sees(d)}"

    return f"{tool} {status}{tail}"


__all__ = ["summarize", "rejection_summary", "FALLBACK_TAG", "TOO_FAR_NO_SUGGESTION"]
