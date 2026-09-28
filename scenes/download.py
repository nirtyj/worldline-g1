"""Selective MolmoSpaces USD downloader (only the houses we use, not the 645 GB split).

`ms-download --type usd --scenes procthor-10k-train` installs EVERY house of the split
(molmo_spaces_isaac/downloader/main.py:132-134 -> ResourceManager.install_all_for_data_type),
which is 10 000 archives / ~645 GB for procthor-10k-train (manifest
`mjthor_resource_file_to_size_mb.json`, measured 2026-09-28). This script builds the same
ResourceManager as `ms-download` (downloader/main.py:115-128: HF repo `allenai/molmospaces`,
prefix `isaac`, same pinned versions) but calls `install_packages` for one archive per house.

Scenes are ON_DEMAND/PER_FILE and thor objects are EAGER/GLOBAL
(molmospaces_resources/behaviors.py DATA_TYPE_DEFAULTS + SOURCE_OVERRIDES), so `setup()`
pulls all THOR object USDs (~1.05 GB, referenced by the scenes through relative paths
`../../objects/thor/...`) and only the manifests for scenes.

Run with the Isaac Lab env (no Isaac app needed):
    /work/envs/isaaclab/bin/python -m scenes.download procthor-train-40 ithor-FloorPlan10
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from scenes.catalog import (
    MOLMOSPACES_ROOT,
    SOURCE_VERSIONS,
    HouseRef,
    parse_house_id,
)

LICENSES = {
    # molmospaces README.md:373-379 (code Apache-2.0, data CC BY 4.0; Objaverse subsets ODC-BY).
    "code": "Apache-2.0 (allenai/molmospaces)",
    "procthor-10k-train": "CC BY 4.0 (MolmoSpaces data); ProcTHOR-10K houses by AI2",
    "ithor": "CC BY 4.0 (MolmoSpaces data); iTHOR scenes by AI2",
    "objects/thor": "CC BY 4.0 (MolmoSpaces data); AI2-THOR object assets",
}


def _du_mb(path: Path) -> float:
    """Disk usage of a directory tree in MB, following the per-file symlinks into the cache."""
    try:
        out = subprocess.run(["du", "-sLm", str(path)], capture_output=True, text=True, check=True)
        return float(out.stdout.split()[0])
    except Exception:
        return -1.0


def make_manager(root: Path, sources: dict[str, str]):
    from huggingface_hub import get_token  # HF_HOME=/work/hf-cache holds the token on the box
    from molmospaces_resources import HFRemoteStorage, ResourceManager

    return ResourceManager(
        remote_storage=HFRemoteStorage(repo_id="allenai/molmospaces", repo_prefix="isaac", token=get_token()),
        data_type_to_source_to_version={"objects": {"thor": SOURCE_VERSIONS["objects/thor"]}, "scenes": sources},
        symlink_dir=root / "usd",
        cache_dir=root / "cache" / "usd",
        force_install=True,
    )


def download(house_ids: list[str], root: Path = MOLMOSPACES_ROOT) -> dict:
    refs: list[HouseRef] = [parse_house_id(h) for h in house_ids]
    sources = {r.source: SOURCE_VERSIONS[r.source] for r in refs}
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mgr = make_manager(root, sources)
    mgr.setup()  # thor objects (eager) + scene manifests
    t_setup = time.time() - t0

    report = {"root": str(root), "setup_s": round(t_setup, 1), "licenses": LICENSES, "houses": {}}
    manifests = {}
    for src in sources:
        cache = mgr.cache_path("scenes", src)
        manifests[src] = json.loads((cache / "mjthor_resource_file_to_size_mb.json").read_text())
    for r in refs:
        t1 = time.time()
        mgr.install_packages("scenes", {r.source: [r.archive]})
        scene_dir = root / "usd" / "scenes" / r.source / r.scene_dir
        report["houses"][r.house_id] = {
            "source": r.source,
            "version": SOURCE_VERSIONS[r.source],
            "archive": r.archive,
            "archive_mb": manifests[r.source].get(r.archive),
            "extracted_mb": _du_mb(scene_dir),
            "scene_usd": str(scene_dir / "scene.usda"),
            "has_metadata": (scene_dir / "scene_metadata.json").exists(),
            "install_s": round(time.time() - t1, 1),
            "license": LICENSES[r.source],
        }
    report["objects_thor_mb"] = _du_mb(root / "usd" / "objects" / "thor")
    report["objects_thor_version"] = SOURCE_VERSIONS["objects/thor"]
    out = root / "downloads.json"
    prev = json.loads(out.read_text()) if out.exists() else {"houses": {}}
    prev["houses"].update(report["houses"])
    report["houses"] = prev["houses"]
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("houses", nargs="+", help="house ids, e.g. procthor-train-40 ithor-FloorPlan10")
    ap.add_argument("--root", type=Path, default=MOLMOSPACES_ROOT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    rep = download(args.houses, args.root)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
