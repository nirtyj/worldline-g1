"""Instance-id visibility for P1 `detections` (docs/contracts/p1_m2b.md §7).

Replicator's `instance_id_segmentation_fast` annotator gives, per render, an HxW uint32 image of instance ids and
`info.idToLabels` = {id: prim path}. This module maps each instance prim to a scene object (the longest object
prim path that prefixes it), the robot (`/World/G1`) or structure, and counts pixels per object. The mapping and
the counting are pure numpy (tested offline); `SegAnnotator` is the thin Isaac wrapper.
"""
from __future__ import annotations

import numpy as np

ROBOT_ROOT = "/World/G1"


class PrimIndex:
    """prim path -> scene object (id, name), by longest prefix; results cached per instance path."""

    def __init__(self, objects: list[dict], robot_root: str = ROBOT_ROOT):
        self.by_path: dict[str, tuple[str, str]] = {}
        for o in objects:
            oid, name = str(o.get("id")), str(o.get("name") or o.get("id"))
            for p in [o.get("prim_path")] + list(o.get("extra_prims") or []):
                if p:
                    self.by_path[str(p).rstrip("/")] = (oid, name)
        self.robot_root = robot_root.rstrip("/")
        self._cache: dict[str, tuple[str, str | None, str | None]] = {}

    def classify(self, path: str) -> tuple[str, str | None, str | None]:
        """-> ("object", id, name) | ("robot", None, None) | ("structure", None, None) | ("background", ..)."""
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        p = str(path or "").rstrip("/")
        out: tuple[str, str | None, str | None]
        if not p or p in ("BACKGROUND", "UNLABELLED", "/"):
            out = ("background", None, None)
        elif p == self.robot_root or p.startswith(self.robot_root + "/"):
            out = ("robot", None, None)
        else:
            out = ("structure", None, None)
            q = p
            while q:
                if q in self.by_path:
                    oid, name = self.by_path[q]
                    out = ("object", oid, name)
                    break
                cut = q.rfind("/")
                if cut <= 0:
                    break
                q = q[:cut]
        self._cache[path] = out
        return out


def count_instances(seg: np.ndarray, id_to_path: dict, index: PrimIndex, *, min_px: int = 40,
                    want_bbox: bool = True, ids: set | None = None) -> tuple[list[dict], dict]:
    """Pixel counts per scene object from one instance-id image.

    seg: HxW integer ids; id_to_path: {id (int or str): prim path}. Returns (detections sorted by px desc with
    px >= min_px, other_px {robot, structure, background})."""
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    flat = seg.reshape(-1).astype(np.int64, copy=False)
    uniq, counts = np.unique(flat, return_counts=True)
    per_obj: dict[str, dict] = {}
    other = {"robot": 0, "structure": 0, "background": 0}
    obj_ids_of: dict[str, list[int]] = {}
    for iid, n in zip(uniq.tolist(), counts.tolist()):
        path = id_to_path.get(iid, id_to_path.get(str(iid)))
        kind, oid, name = index.classify(path) if path is not None else ("background", None, None)
        if kind != "object":
            other[kind] += int(n)
            continue
        d = per_obj.setdefault(oid, {"id": oid, "name": name, "px": 0})
        d["px"] += int(n)
        obj_ids_of.setdefault(oid, []).append(int(iid))
    dets = [d for d in per_obj.values() if d["px"] >= min_px and (ids is None or d["id"] in ids)]
    if want_bbox and dets:
        w = seg.shape[1]
        for d in dets:
            mask = np.isin(flat, obj_ids_of[d["id"]])
            idx = np.flatnonzero(mask)
            rows, cols = idx // w, idx % w
            d["bbox"] = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
    dets.sort(key=lambda d: -d["px"])
    return dets, other


class SegAnnotator:
    """`instance_id_segmentation_fast` on an existing render product (Isaac only)."""

    NAME = "instance_id_segmentation_fast"

    def __init__(self, rp_path: str):
        import omni.replicator.core as rep

        self.rp_path = rp_path
        self.annot = rep.AnnotatorRegistry.get_annotator(self.NAME, init_params={"colorize": False}, device="cpu")
        self.annot.attach([rp_path])
        self.attached = True

    def read(self) -> tuple[np.ndarray | None, dict]:
        d = self.annot.get_data()
        if isinstance(d, dict):
            data, info = d.get("data"), d.get("info") or {}
        else:
            data, info = d, {}
        if data is None:
            return None, {}
        a = np.asarray(data)
        if a.size == 0:
            return None, {}
        m = info.get("idToLabels") or {}
        return a, {int(k) if str(k).isdigit() else k: v for k, v in m.items()}

    def detach(self) -> None:
        if not self.attached:
            return
        try:
            self.annot.detach([self.rp_path])
        except Exception:  # noqa: BLE001
            pass
        self.attached = False
