#!/usr/bin/env python3
"""Check a recomputation of the paper's numbers against the recorded values.

    python scripts/camera_ready_numbers.py --out-json numbers.json --out-md numbers.md
    python scripts/check_numbers.py numbers.json

Every value must match records/camera_ready_numbers.json to a relative tolerance
of 1e-9, which absorbs last-digit differences between NumPy and SciPy versions.
The MuJoCo re-run rows (physics.*) also allow 1e-4 m in their height
differences, which change slightly between MuJoCo versions; their counts and
steps must match exactly. Rows whose inputs were not provided (physics.* and
dup.* without the body files, physics.* also with --no-physics, logs.* without
logs.zip, panels.* without panels.zip) are counted as skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPTIONAL = ("physics.", "dup.", "logs.", "panels.")


def same(a, b, abs_tol: float) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=abs_tol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, abs_tol) for x, y in zip(a, b))
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("new", type=Path, help="JSON written by camera_ready_numbers.py --out-json")
    ap.add_argument("--recorded", type=Path, default=REPO / "records/camera_ready_numbers.json")
    args = ap.parse_args()

    old = json.loads(args.recorded.read_text())["numbers"]
    new = json.loads(args.new.read_text())["numbers"]
    diff, skipped, missing = [], [], []
    for key, rec in old.items():
        if key not in new:
            (skipped if key.startswith(OPTIONAL) else missing).append(key)
            continue
        tol = 1e-4 if key.startswith("physics.") else 1e-12
        if not (same(rec["value"], new[key]["value"], tol) and same(rec["ci"], new[key]["ci"], tol)):
            diff.append(key)
    extra = [key for key in new if key not in old]

    compared = len(old) - len(skipped) - len(missing)
    print(f"{compared} values compared: {compared - len(diff)} match, {len(diff)} differ; "
          f"{len(skipped)} skipped (inputs not provided), {len(missing)} missing, {len(extra)} not recorded")
    for key in diff[:50]:
        print(f"  differs: {key}: recorded {old[key]['value']!r} {old[key]['ci']!r}, "
              f"now {new[key]['value']!r} {new[key]['ci']!r}")
    for key in missing[:50]:
        print(f"  missing: {key}")
    for key in extra[:50]:
        print(f"  not recorded: {key}")
    return 1 if diff or missing or extra else 0


if __name__ == "__main__":
    raise SystemExit(main())
