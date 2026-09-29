"""Scene vocabulary: MolmoSpaces categories -> AI2-THOR types -> Worldline names (docs/scenes.md §8).

MolmoSpaces categories ARE AI2-THOR object types (the ProcTHOR JSON types verbatim), with a few differences that
Worldline sees: `Sink` (THOR runtime sub-object `SinkBasin`), `ShelvingUnit` (THOR `Shelf` levels), `Stove`
(iTHOR geometry; THOR `StoveBurner`s). Those are aliased here, so the THOR-era tables below apply unchanged.

The tables are copied from ludo-runtime `thor/world.py:61-77` (SURFACE_TYPES, LANDMARK_TYPES, GROUPED_LANDMARKS,
HELD_CONTAINERS) so the planner prompt, layout relations and eval keep their names.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

# MolmoSpaces category -> AI2-THOR type (only where they differ)
CATEGORY_ALIASES = {
    "Sink": "SinkBasin",
    "ShelvingUnit": "Shelf",
    "Stove": "StoveBurner",
}

# THOR type -> what people call it (thor/world.py:61-66)
SURFACE_TYPES = {
    "CounterTop": "counter", "DiningTable": "dining table", "CoffeeTable": "coffee table",
    "SideTable": "side table", "Desk": "desk", "Dresser": "dresser", "Shelf": "shelf",
    "TVStand": "TV stand", "Sofa": "sofa", "Bed": "bed", "ArmChair": "armchair",
    "Ottoman": "ottoman", "SinkBasin": "sink",
}
# THOR type -> label (thor/world.py:67-73)
LANDMARK_TYPES = {
    "Microwave": "microwave", "Toaster": "toaster", "StoveBurner": "stove",
    "CoffeeMachine": "coffee machine", "Fridge": "fridge", "GarbageCan": "bin",
    "Television": "TV", "Toilet": "toilet", "Bathtub": "bathtub", "ShowerHead": "shower",
    "FloorLamp": "floor lamp", "HousePlant": "plant",
    "LaundryHamper": "laundry hamper", "Safe": "safe", "Desktop": "computer",
}
GROUPED_LANDMARKS = {"StoveBurner"}          # several THOR objects, one name
HELD_CONTAINERS = {"Bowl", "Plate", "Mug", "Cup", "Pan", "Pot"}   # an apple in a bowl is "on" the bowl's surface
# Things you can put something INTO (where = the snake type, "inside_or_on_<x>" for reachability)
CONTAINER_TYPES = {"Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "Dishwasher", "GarbageCan", "Toaster",
                   "LaundryHamper", "Box", "WashingMachine", "Bathtub"}
# Never occluders for perception (openings, not solids) and never "things" for the truth vocabulary
NON_SOLID_TYPES = {"Doorway", "Doorframe", "Window", "Door", "DoorKnob", "StoveKnob", "Faucet"}

_VOCAB_FILE = Path(__file__).resolve().parents[1] / "config" / "vocab" / "pickupable_types.yaml"


def thor_type(category: str) -> str:
    """MolmoSpaces category -> AI2-THOR type."""
    return CATEGORY_ALIASES.get(category, category)


def snake(name: str) -> str:
    """CreditCard -> credit_card, TVStand -> tv_stand (thor/world.py:118-120, verbatim rule)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def words(name: str) -> str:
    return snake(name).replace("_", " ")


@functools.lru_cache(maxsize=4)
def pickupable_types(path: str | None = None) -> frozenset[str]:
    """The fixed THOR pickupable vocabulary (snake_case), from config/vocab/pickupable_types.yaml."""
    p = Path(path or os.environ.get("WL_PICKUPABLE_VOCAB", "") or _VOCAB_FILE)
    text = p.read_text()
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        types = data.get("types", []) if isinstance(data, dict) else data
    except ImportError:                                   # tiny fallback parser: "  - name  # comment"
        types = [ln.split("#", 1)[0].strip()[1:].strip() for ln in text.splitlines()
                 if ln.split("#", 1)[0].strip().startswith("-")]
    return frozenset(str(t).strip() for t in types if str(t).strip())


def is_pickupable(ttype: str) -> bool:
    return snake(ttype) in pickupable_types()


def is_surface(ttype: str) -> bool:
    return ttype in SURFACE_TYPES


def is_landmark(ttype: str) -> bool:
    return ttype in LANDMARK_TYPES and not is_pickupable(ttype)


def is_container(ttype: str) -> bool:
    return ttype in CONTAINER_TYPES


def landmark_base(ttype: str) -> str:
    """StoveBurner -> stove, CoffeeMachine -> coffee_machine, Television -> tv (thor/world.py:441)."""
    return snake(LANDMARK_TYPES[ttype]).replace(" ", "_")


def surface_base(ttype: str) -> str:
    """CounterTop -> counter, DiningTable -> dining_table, TVStand -> tv_stand (thor/world.py:384)."""
    return "counter" if SURFACE_TYPES.get(ttype) == "counter" else snake(ttype)
