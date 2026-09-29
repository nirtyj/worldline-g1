"""Generate api/schemas/*.json from the contract (PLAN 4.2: schemas/*.json, generated).

    python -m api.gen_schemas            # write
    python -m api.gen_schemas --check    # exit 1 if the committed files are stale

The tool schemas use symbolic enums (``<location>``, ``<surface>``, ``<object_type>``) and the
``sonic`` profile's numeric slots; the real enums are filled once per session from the map
and the loaded registry (PLAN 5.2). The typed-result schemas are derived from the dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import typing
from pathlib import Path
from typing import Any

from . import results as R
from .tools import SchemaContext, json_schemas
from .types import PROFILES

OUT = Path(__file__).resolve().parent / "schemas"
SYMBOLIC = SchemaContext(locations=("<location>", "user"), surfaces=("<surface>",), skill_types=("<object_type>",),
                         slots=PROFILES["sonic"].slots())
TYPED = {"SpeakResult": R.SpeakResult, "NamedLocation": R.NamedLocation,
         "ListLocationsResult": R.ListLocationsResult, "NavigateResult": R.NavigateResult,
         "ReachabilityResult": R.ReachabilityResult, "ManipulationResult": R.ManipulationResult,
         "WaitResult": R.WaitResult, "RecallResult": R.RecallResult}


def _type_schema(tp: Any) -> dict[str, Any]:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is type(None):
        return {"type": "null"}
    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        subs = [_type_schema(a) for a in args]
        types = []
        for s in subs:
            t = s.get("type")
            types.extend(t if isinstance(t, list) else [t] if t else [])
        return {"type": sorted(set(types))} if types else {}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(args)}
    if origin in (list, tuple):
        return {"type": "array"}
    if origin is dict or tp is dict:
        return {"type": "object"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if dataclasses.is_dataclass(tp):
        return {"$ref": f"#/$defs/{tp.__name__}"}
    return {}


def dataclass_schema(cls: type) -> dict[str, Any]:
    hints = typing.get_type_hints(cls, vars(R), vars(typing))
    props, required = {}, []
    for f in dataclasses.fields(cls):
        props[f.name] = _type_schema(hints[f.name])
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            required.append(f.name)
    return {"type": "object", "title": cls.__name__, "properties": props, "required": required}


def build() -> dict[str, Any]:
    return {
        "tools.neutral.json": json_schemas(SYMBOLIC, "neutral"),
        "tools.anthropic.json": json_schemas(SYMBOLIC, "anthropic"),
        "tool_result.json": R.envelope_schema(),
        "typed_results.json": {"$defs": {k: dataclass_schema(v) for k, v in TYPED.items()}},
    }


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def main(argv: list[str]) -> int:
    files = build()
    if "--check" in argv:
        stale = [n for n, obj in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != dump(obj)]
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        print("api/schemas up to date")
        return 0
    OUT.mkdir(exist_ok=True)
    for name, obj in files.items():
        (OUT / name).write_text(dump(obj))
        print("wrote", OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
