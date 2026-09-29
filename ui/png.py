"""A tiny PNG encoder (stdlib only) for the page's ground-truth occupancy picture.

The UI draws the sim's occupancy grid under the "reality" map. The grid is small
(a few hundred cells a side), so an 8-bit palette PNG built with zlib is enough
and keeps Pillow out of the UI's hard dependencies.
"""

from __future__ import annotations

import struct
import zlib
from typing import Sequence


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_palette_png(rows: Sequence[bytes | bytearray], width: int,
                       palette: Sequence[tuple[int, int, int, int]]) -> bytes:
    """rows: one bytes object per image row (top row first), each `width` palette indices.

    palette: up to 256 RGBA entries; the alpha channel goes into a tRNS chunk.
    """
    if not rows:
        raise ValueError("empty image")
    if len(palette) == 0 or len(palette) > 256:
        raise ValueError("palette needs 1..256 entries")
    height = len(rows)
    raw = bytearray()
    for r in rows:
        if len(r) != width:
            raise ValueError(f"row of {len(r)} bytes, expected {width}")
        raw.append(0)                       # filter: none
        raw += r
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)   # 8-bit indexed colour
    plte = b"".join(struct.pack("BBB", *c[:3]) for c in palette)
    trns = bytes(c[3] if len(c) > 3 else 255 for c in palette)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"PLTE", plte)
            + _chunk(b"tRNS", trns) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b""))


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR (used by tests and to caption the image)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])
