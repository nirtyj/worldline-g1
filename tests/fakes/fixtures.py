"""Shared pytest fixtures for tests/world and tests/services (imported by their conftest.py)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOUSES_DIR = Path(__file__).resolve().parent / "houses"
RECORDED = sorted(p.name for p in HOUSES_DIR.iterdir() if (p / "house_info.json").exists())
PROCTHOR = [h for h in RECORDED if h.startswith("procthor")]


@functools.lru_cache(maxsize=16)
def _lite(house: str, camera: str = "head_sim"):
    from world.lite_world import LiteWorld
    return LiteWorld(HOUSES_DIR / house, camera=camera)


def fresh_lite(house: str = "procthor-train-38", camera: str = "head_sim", **kw):
    from world.lite_world import LiteWorld
    return LiteWorld(HOUSES_DIR / house, camera=camera, **kw)


@pytest.fixture
def lite38():
    return fresh_lite("procthor-train-38")


@pytest.fixture(params=PROCTHOR)
def lite_house(request):
    """A fresh LiteWorld per recorded ProcTHOR house (38, 40, 15)."""
    return fresh_lite(request.param)


@pytest.fixture
def cached_map():
    """Static maps are immutable: share them between tests."""
    return lambda house: _lite(house).map
