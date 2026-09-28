"""Fast CDR codec for the fixed-layout unitree_hg messages.

cyclonedds-python (de)serialises IDL structs in pure Python: a LowState_ write + CRC costs ~1-3 ms and every
received LowCmd_ ~0.2 ms of GIL time, which at 200 Hz lowstate + 1500 Hz of incoming lowcmd/dex3 cmd pushes the
physics loop below real time. The unitree_hg messages we exchange have a fixed layout (fixed arrays, or sequences
whose length we fix at 7), so we:

- locate every scalar field's byte offset once, by serialising a template with unique sentinel values;
- keep one serialised buffer per writer and patch floats in place with numpy (vectorised);
- (LowState_ CRC: see dds_bridge.LowStateCrc; the CRC covers the C struct layout, which differs from the CDR layout);
- read samples as raw CDR bytes (`ddspy_take`) and pull floats out with numpy.

Every codec is verified at start-up against the library serializer/deserializer (byte-for-byte / value-for-value);
if a check fails, the slow library path is used instead and a warning is logged.
"""
from __future__ import annotations

import ctypes
import struct
from typing import Any, Callable

import numpy as np


def _get(obj, path):
    for p in path:
        obj = obj[p] if isinstance(p, int) else getattr(obj, p)
    return obj


def _set(obj, path, v):
    parent = _get(obj, path[:-1])
    last = path[-1]
    if isinstance(last, int):
        parent[last] = v
    else:
        setattr(parent, last, v)


def _pad4(b: bytes) -> bytes:
    return b.ljust((len(b) + 3) & ~3, b"\0")


def locate(factory: Callable[[], Any], fields: dict[str, list[tuple]], kind: str = "f") -> dict[str, np.ndarray]:
    """Return {group: array of 4-byte word indices} for each field path in the serialised sample."""
    s = factory()
    sentinels: dict[str, list] = {}
    k = 0
    for g, paths in fields.items():
        vals = []
        for p in paths:
            v = (1000.0 + 0.5 * k) if kind == "f" else (0x5A5A0000 + k)
            _set(s, p, v)
            vals.append(v)
            k += 1
        sentinels[g] = vals
    ser = _pad4(s.serialize())
    out = {}
    for g, vals in sentinels.items():
        idx = []
        for v in vals:
            b = struct.pack("<f" if kind == "f" else "<I", v)
            i = ser.find(b)
            if i < 0 or ser.find(b, i + 1) >= 0 or i % 4:
                raise RuntimeError(f"cannot locate field {g} (sentinel {v}) in serialised {type(s).__name__}")
            idx.append(i // 4)
        out[g] = np.asarray(idx, dtype=np.int64)
    return out


class FastWriter:
    """Holds one serialised sample; float fields are patched in place, then written with ddspy_write."""

    def __init__(self, participant, topic_name: str, idl_type, factory: Callable[[], Any],
                 float_fields: dict[str, list[tuple]], uint_fields: dict[str, list[tuple]] | None = None,
                 log=print):
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        self.name = topic_name
        self.topic = Topic(participant, topic_name, idl_type)
        self.writer = DataWriter(participant, self.topic)
        self.factory = factory
        self.float_fields = float_fields
        self.uint_fields = uint_fields or {}
        self.template = factory()
        self.buf = bytearray(_pad4(self.template.serialize()))
        self.f32 = np.frombuffer(self.buf, dtype="<f4")
        self.u32 = np.frombuffer(self.buf, dtype="<u4")
        self.fidx = locate(factory, float_fields, "f")
        self.uidx = locate(factory, self.uint_fields, "I") if self.uint_fields else {}
        self.fast = True
        self.slow_crc = False  # set by the owner for LowState_ (library CRC in the slow path)
        self._slow_vals: dict = {}
        from cyclonedds._clayer import ddspy_write
        self._ddspy_write = ddspy_write
        self.log = log

    def set(self, group: str, values) -> None:
        self.f32[self.fidx[group]] = values
        if not self.fast:
            self._slow_vals[group] = np.asarray(values, dtype=np.float64).reshape(-1)

    def set_u(self, group: str, values) -> None:
        self.u32[self.uidx[group]] = values
        if not self.fast:
            self._slow_vals[group] = np.asarray(values).reshape(-1)

    def write(self) -> None:
        if not self.fast:
            return self._write_slow()
        ret = self._ddspy_write(self.writer._ref, bytes(self.buf))
        if ret < 0:
            raise RuntimeError(f"ddspy_write({self.name}) failed: {ret}")

    def _write_slow(self) -> None:
        """Library path (used only if verify() failed): build the sample and let cyclonedds serialise it."""
        s = self.factory()
        for g, paths in list(self.float_fields.items()) + list(self.uint_fields.items()):
            if g == "crc" or g not in self._slow_vals:
                continue
            is_u = g in self.uint_fields
            for p, v in zip(paths, self._slow_vals[g]):
                _set(s, p, int(v) if is_u else float(v))
        if self.slow_crc:
            from unitree_sdk2py.utils.crc import CRC
            s.crc = CRC().Crc(s)
        self.writer.write(s)

    def verify(self, fill: Callable[[Any, "FastWriter"], None],
               crc: tuple[Callable[[Any], int], Callable[["FastWriter"], int]] | None = None) -> bool:
        """fill(sample, self) must set the same random values on a library sample and on this buffer.
        crc = (library_crc(sample), fast_crc(self)) for messages that carry a CRC."""
        s = self.factory()
        fill(s, self)
        if crc is not None:
            s.crc = crc[0](s)
            self.set_u("crc", crc[1](self))
        ref = _pad4(s.serialize())
        ok = ref == bytes(self.buf)
        if not ok:
            diff = [i for i in range(min(len(ref), len(self.buf))) if ref[i] != self.buf[i]][:8]
            self.log(f"[fastdds] {self.name}: fast serialisation MISMATCH (len {len(ref)} vs {len(self.buf)}, "
                     f"first diffs at {diff}); using the library path")
        # reset to the template values
        self.buf[:] = _pad4(self.template.serialize())
        self.fast = ok
        return ok


class FastReader:
    """Takes the newest sample as raw CDR and extracts float fields with numpy (falls back to deserialize())."""

    def __init__(self, participant, topic_name: str, idl_type, factory: Callable[[], Any],
                 float_fields: dict[str, list[tuple]], log=print):
        from cyclonedds._clayer import ddspy_take
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        self.name = topic_name
        self.idl_type = idl_type
        self.topic = Topic(participant, topic_name, idl_type)
        self.reader = DataReader(participant, self.topic)  # default QoS: KEEP_LAST(1), like unitree_sdk2py readers
        self.float_fields = float_fields
        tmpl = _pad4(factory().serialize())
        self.len = len(tmpl)
        self.hdr = tmpl[:4]
        self.fidx = locate(factory, float_fields, "f")
        self._take = ddspy_take
        self.count = 0
        self.slow_count = 0
        self.log = log
        self._warned = False
        self.force_slow = False

    def take_latest(self, max_n: int = 16):
        """Return ({group: float64 array}, source_timestamp_ns) for the newest valid sample, or None."""
        ret = self._take(self.reader._ref, max_n)
        if isinstance(ret, int) or not ret:
            return None
        data, info = None, None
        for d, inf in ret:
            if inf.valid_data:
                data, info = d, inf
                self.count += 1
        if data is None:
            return None
        return self.parse(bytes(data)), getattr(info, "source_timestamp", None)

    def parse(self, data: bytes) -> dict:
        b = _pad4(data)
        if not self.force_slow and len(b) == self.len and b[:4] == self.hdr:
            f = np.frombuffer(b, dtype="<f4")
            return {g: f[i].astype(np.float64) for g, i in self.fidx.items()}
        # different encoding/length: slow path
        self.slow_count += 1
        if not self._warned:
            self._warned = True
            self.log(f"[fastdds] {self.name}: sample len {len(b)} hdr {b[:4].hex()} != template "
                     f"({self.len}, {self.hdr.hex()}); using deserialize()")
        s = self.idl_type.deserialize(data)
        out = {}
        for g, paths in self.float_fields.items():
            vals = []
            for p in paths:
                try:
                    vals.append(float(_get(s, p)))
                except (IndexError, AttributeError):
                    vals.append(0.0)
            out[g] = np.asarray(vals, dtype=np.float64)
        return out

    def verify(self, factory: Callable[[], Any]) -> bool:
        """Serialise a random library sample and check that parse() recovers every float field."""
        rng = np.random.default_rng(0)
        s = factory()
        ref = {}
        for g, paths in self.float_fields.items():
            vals = rng.uniform(-3, 3, len(paths)).astype(np.float32)
            for p, v in zip(paths, vals):
                _set(s, p, float(v))
            ref[g] = vals.astype(np.float64)
        slow_before = self.slow_count
        got = self.parse(s.serialize())
        ok = self.slow_count == slow_before and all(np.array_equal(got[g], ref[g]) for g in ref)
        if not ok:
            self.log(f"[fastdds] {self.name}: fast parse MISMATCH; falling back to deserialize()")
            self.force_slow = True
        return ok
