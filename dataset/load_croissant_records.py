#!/usr/bin/env python3
"""Load every record of every record set in croissant.json from local archives.

The contentUrl of runs.zip, logs.zip and panels.zip points to the dataset
record. To test record loading before (or without) downloading, this script
writes a copy of the Croissant file whose contentUrl entries point to the
archives in --archives, then iterates over all records with mlcroissant and
prints the count per record set. The sha256 values are left as they are, so
mlcroissant also checks the archives against the published hashes.

    python dataset/load_croissant_records.py --archives <dir with the three archives>
    python dataset/load_croissant_records.py --archives <dir> \
        --expect runs=3306,morphology_features=160,humanoid_fall_rule_cells=1800,humanoid_fall_rule_progress=10800,humanoid_time_limit_cells=900,humanoid_time_limit_progress=5400,ant_reset_init_cells=160,ant_breadth_cells=1800,evogym_time_limit_cells=1200

Needs mlcroissant 1.1.1 with pandas < 3 (see requirements-croissant.txt).
Exits 1 if a record set fails to load or a count differs from --expect.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent


def local_copy(jsonld: Path, archives: Path, out_dir: Path) -> Path:
    meta = json.loads(jsonld.read_text())
    for d in meta.get("distribution", []):
        if d.get("@type") == "cr:FileObject" and "containedIn" not in d:
            p = (archives / d["name"]).resolve()
            if not p.is_file():
                raise SystemExit(f"ERROR: {p} not found")
            d["contentUrl"] = str(p)
    out = out_dir / "croissant.local.json"
    out.write_text(json.dumps(meta, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archives", type=Path, required=True, help="directory holding runs.zip, logs.zip and panels.zip")
    ap.add_argument("--jsonld", type=Path, default=HERE / "croissant.json")
    ap.add_argument("--expect", default="", help="comma-separated record_set=count pairs to check")
    args = ap.parse_args()

    import mlcroissant as mlc
    import pandas as pd

    print(f"mlcroissant {version('mlcroissant')}, pandas {pd.__version__}, python {sys.version.split()[0]}")
    expect = dict(kv.split("=") for kv in args.expect.split(",") if kv)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        ds = mlc.Dataset(jsonld=str(local_copy(args.jsonld, args.archives, Path(tmp))))
        for rs in ds.metadata.record_sets:
            try:
                n = sum(1 for _ in ds.records(record_set=rs.uuid))
            except Exception as e:  # report and continue with the next record set
                print(f"FAIL {rs.uuid}: {type(e).__name__}: {e}")
                ok = False
                continue
            want = expect.get(rs.uuid)
            flag = "" if want is None else (" (expected)" if int(want) == n else f" (EXPECTED {want})")
            ok &= want is None or int(want) == n
            print(f"{rs.uuid}: {n} records{flag}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
