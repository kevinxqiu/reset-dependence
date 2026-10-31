#!/usr/bin/env python3
"""Select a stratified 60-body panel from the non-evolved UNIMAL init bodies.

Substrate: `assets/generated_morphologies/` holds 410 morphologies that were
randomly sampled from the UNIMAL parametric search space BEFORE DERL's
fitness-based selection. They are NOT organized by evolutionary family (no
floor/vt/mvt prefix); the files are named gen-0001.xml through gen-0410.xml and
carry raw structural metadata in unimal_init/gen-NNNN.pkl.

This script picks a 60-body panel stratified by (num_limbs x torso_mode)
for the NEMS (Non-Evolved Morphology Substrate) experiment. The panel is:
  - 5 bodies per (num_limbs, torso_mode) cell
  - num_limbs in {3, 4, 5, 6, 7, 8} (6 levels), torso_mode in
    {vertical, horizontal_y} (2 levels) = 12 cells x 5 = 60 bodies
  - Deterministic (numpy.random.default_rng(seed=42)) so the panel freezes
    BEFORE any training runs, preventing cherry-pick.

Design note: limb-only stratification is insufficient because the 3-limb
bucket collapses to all horizontal_y under pure random sampling. 2D
stratification on (num_limbs x torso_mode)
forces balanced coverage on both axes.

Outputs:
  records/nems_panel_bodies.json  (frozen body list + features)
  records/nems_panel_bodies.txt   (newline-separated names for
                                          --morphology-list in SLURM)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def default_generated_morphology_dir() -> Path:
    fallback = REPO / "assets/generated_morphologies"
    for path in (
        os.environ.get("RFC_NEMS_ROOT"),
        fallback,
    ):
        if path is None:
            continue
        candidate = Path(path)
        if (candidate / "xml").is_dir() and (candidate / "unimal_init").is_dir():
            return candidate
    return fallback


GEN_DIR = default_generated_morphology_dir()
XML_DIR = GEN_DIR / "xml"
META_DIR = GEN_DIR / "unimal_init"
OUT_JSON = REPO / "records/nems_panel_bodies.json"
OUT_TXT = REPO / "records/nems_panel_bodies.txt"

N_PER_CELL = 5
TARGET_LIMBS = (3, 4, 5, 6, 7, 8)  # skip 9-limb (n=17 too few, often pathological)
TARGET_TORSOS = ("vertical", "horizontal_y")
SEED = 42


def extract_features(pkl_path: Path) -> dict:
    """Pull structural features from a gen-NNNN.pkl record."""
    with pkl_path.open("rb") as f:
        d = pickle.load(f)
    bp = d.get("body_params", {})
    lm = d.get("limb_metadata", {})
    # Compute kinematic tree depth as max |ancestors| in limb_metadata
    # (fallback: num_limbs if metadata is flat)
    tree_depth = 1
    for limb_id, meta in lm.items():
        if isinstance(meta, dict):
            # Each limb has parent_name; depth = chain length to torso
            # We approximate by counting limbs whose parent is not torso
            pass
    # Simpler proxy: limb_list nesting depth
    limb_list = d.get("limb_list", [])
    if limb_list:
        tree_depth = max(len(level) for level in limb_list) if limb_list else 1
    return {
        "name": pkl_path.stem,
        "num_limbs": int(d.get("num_limbs", 0)),
        "num_torso": int(d.get("num_torso", 0)),
        "contact_pairs": int(len(d.get("contact_pairs", []))),
        "torso_mode": str(bp.get("torso_mode", "unknown")),
        "torso_density": int(bp.get("torso_density", 0)),
        "limb_density": int(bp.get("limb_density", 0)),
        "tree_depth_proxy": int(tree_depth),
    }


def main() -> int:
    records = [extract_features(p) for p in sorted(META_DIR.glob("gen-*.pkl"))]
    if not records:
        print(f"ERROR: no gen-*.pkl records in {META_DIR}", file=sys.stderr)
        return 1

    # Verify corresponding XML files exist
    missing = [r["name"] for r in records if not (XML_DIR / f"{r['name']}.xml").is_file()]
    if missing:
        print(f"WARNING: {len(missing)} bodies missing XML; dropping them.")
        records = [r for r in records if r["name"] not in missing]

    # 2D bucket by (num_limbs, torso_mode)
    by_cell: dict[tuple, list] = defaultdict(list)
    for r in records:
        by_cell[(r["num_limbs"], r["torso_mode"])].append(r)

    rng = np.random.default_rng(SEED)
    panel: list[dict] = []
    for nl in TARGET_LIMBS:
        for tm in TARGET_TORSOS:
            bucket = by_cell.get((nl, tm), [])
            if len(bucket) < N_PER_CELL:
                print(
                    f"WARNING: cell ({nl},{tm}) only has {len(bucket)} "
                    f"bodies; using all.",
                    file=sys.stderr,
                )
                picks = bucket
            else:
                idx = rng.choice(len(bucket), size=N_PER_CELL, replace=False)
                picks = [bucket[i] for i in sorted(idx)]
            for p in picks:
                p["stratum"] = f"limbs_{nl}_{tm}"
                panel.append(p)

    print(f"Selected {len(panel)} bodies from {len(records)} total.")
    cell_sizes = {f"{nl}x{tm}": len(by_cell[(nl, tm)]) for nl in TARGET_LIMBS for tm in TARGET_TORSOS}
    print(f"Per-cell pool sizes: {cell_sizes}")
    print()
    for p in panel:
        print(
            f"  {p['name']:10s} limbs={p['num_limbs']} torso={p['torso_mode']:12s} "
            f"contacts={p['contact_pairs']} depth={p['tree_depth_proxy']}"
        )

    out = {
        "seed": SEED,
        "n_per_cell": N_PER_CELL,
        "target_limbs": list(TARGET_LIMBS),
        "target_torsos": list(TARGET_TORSOS),
        "n_panel": len(panel),
        "panel": panel,
        "substrate": "UNIMAL-init (pre-DERL-selection)",
        "xml_dir": str(XML_DIR),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    with OUT_TXT.open("w") as f:
        for p in panel:
            f.write(p["name"] + "\n")

    print()
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
