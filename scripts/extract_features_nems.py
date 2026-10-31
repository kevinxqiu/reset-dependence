#!/usr/bin/env python3
"""Extract structural features for the 60 NEMS panel bodies.

Reads the pre-registered NEMS panel manifest
(`records/nems_panel_bodies.json`) and extracts the same structural
feature schema as `scripts/extract_morph_features.py` (which targets the
100-body UNIMAL corpus). Output matches `results/morph_features.json` so
that `scripts/structural_feature_regression.py` can ingest NEMS features
directly for zero-shot evaluation.

NEMS XMLs are expected under `assets/generated_morphologies/xml/` or under
RFC_NEMS_XML_DIR. They are not distributed with this repository. They use the
same UNIMAL parametric generator (`torso/0` naming, identical geom / joint /
body schema). No body-level exclusions are applied; missing bodies are reported
but the script writes a partial result.

Usage:
    python scripts/extract_features_nems.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from extract_morph_features import extract_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "records" / "nems_panel_bodies.json"


def default_nems_xml_dir() -> Path:
    fallback = REPO / "assets/generated_morphologies/xml"
    for path in (
        os.environ.get("RFC_NEMS_XML_DIR"),
        fallback,
    ):
        if path is None:
            continue
        candidate = Path(path)
        if candidate.is_dir():
            return candidate
    return fallback


XML_DIR = default_nems_xml_dir()
OUT = REPO / "results" / "morph_features_nems.json"


def main() -> None:
    with MANIFEST.open() as f:
        manifest = json.load(f)

    bodies = [entry["name"] for entry in manifest["panel"]]
    print(f"NEMS panel: {len(bodies)} bodies")

    features: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for name in bodies:
        xml_path = XML_DIR / f"{name}.xml"
        if not xml_path.exists():
            missing.append(name)
            continue
        features[name] = extract_features(xml_path)

    if missing:
        print(f"  WARNING: {len(missing)} bodies missing XML: {missing[:5]}...")

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        json.dump(features, f, indent=2)
    print(f"Extracted features for {len(features)} bodies; wrote {OUT}")


if __name__ == "__main__":
    main()
