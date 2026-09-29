"""msgpack + numpy codec of the Isaac-GR00T PolicyServer, re-implemented without importing `gr00t`.

Upstream (Isaac-GR00T @ 4b1dca9d, `gr00t/policy/server_client.py:30-60`, `MsgSerializer`):
- `to_bytes(x)   = msgpack.packb(x, default=encode_custom_classes)`
- `from_bytes(b) = msgpack.unpackb(b, object_hook=decode_custom_classes)`
- an `np.ndarray` travels as `{"__ndarray_class__": True, "as_npy": <np.save(..., allow_pickle=False) bytes>}`
- a `ModalityConfig` travels as `{"__ModalityConfig_class__": True, "as_json": {...}}`; here it decodes to that plain
  dict (the runtime has no ModalityConfig class).

No pickle anywhere: `np.load(..., allow_pickle=False)` refuses object arrays in both directions.
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np

NDARRAY_TAG = "__ndarray_class__"
MODALITY_CONFIG_TAG = "__ModalityConfig_class__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {NDARRAY_TAG: True, "as_npy": buf.getvalue()}
    if isinstance(obj, np.generic):          # numpy scalars (upstream would fail on these; we send plain values)
        return obj.item()
    raise TypeError(f"cannot encode {type(obj).__name__} for the GR00T wire")


def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if NDARRAY_TAG in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if MODALITY_CONFIG_TAG in obj:
        return dict(obj["as_json"])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode)
