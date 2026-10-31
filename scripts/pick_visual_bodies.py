#!/usr/bin/env python3
"""Deterministic selection of the 9 UNIMAL bodies used for visual figures.

For each of the 3 families (Floor, VT, MVT), pick the bodies at the P25,
P50, and P75 of Delta_reset = PSR_epi - PSR_RF, using seed-averaged corpus
PSRs evaluated at tau_eval=0.20. Tie-break on smallest morph_idx. The frozen
output JSON is included in the code repository as figure-selection metadata.

Writes records/visual_bodies_frozen.json.

Run once at the start of Phase 1; downstream scripts load the frozen JSON.
Re-running this script is safe: the output is fully deterministic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "results"
OUT = REPO / "records/visual_bodies_frozen.json"

TAUS = {"episodic": "0p20", "reset_free": "0p00"}
SEEDS = (42, 43, 44)
FAMILIES = ("floor", "vt", "mvt")
PERCENTILES = (25, 50, 75)


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def load_psr_table() -> dict[str, dict[str, float]]:
    """Return {morph_name: {'episodic': psr, 'reset_free': psr}}."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for protocol, tau in TAUS.items():
        for seed in SEEDS:
            path = CORPUS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with path.open() as f:
                data = json.load(f)
            for morph, rec in data.items():
                psr = rec["eval_tau020"]["per_step_reward"]
                out[morph][protocol].append(float(psr))
    # Seed-average
    flat: dict[str, dict[str, float]] = {}
    for morph, d in out.items():
        flat[morph] = {p: float(np.mean(v)) for p, v in d.items()}
    return flat


def load_morph_idx() -> dict[str, int]:
    """Pull morph_idx from one of the corpus files (stable across files)."""
    path = CORPUS / "mjx_ppo_0p20_seed42" / "results.json"
    with path.open() as f:
        data = json.load(f)
    return {m: int(rec["morph_idx"]) for m, rec in data.items()}


def pick_by_percentile(
    candidates: list[tuple[str, float, int]],
) -> dict[int, str]:
    """For each of (25, 50, 75), pick the body at that percentile of delta_reset.

    `candidates` is a list of (morph_name, delta_reset, morph_idx). Ties at a
    percentile resolve to the smallest morph_idx (deterministic).
    """
    sorted_cs = sorted(candidates, key=lambda x: (x[1], x[2]))
    picks: dict[int, str] = {}
    for p in PERCENTILES:
        # numpy percentile with method='lower' returns one of the existing
        # values; convert that back to the body whose delta matches.
        target = float(np.percentile([c[1] for c in sorted_cs], p, method="lower"))
        for name, delta, _ in sorted_cs:
            if np.isclose(delta, target):
                picks[p] = name
                break
    return picks


def main() -> int:
    psr = load_psr_table()
    idx_map = load_morph_idx()

    per_family: dict[str, list[tuple[str, float, int]]] = {f: [] for f in FAMILIES}
    for morph, d in psr.items():
        fam = family_of(morph)
        if fam not in per_family:
            continue
        delta = d["episodic"] - d["reset_free"]
        per_family[fam].append((morph, delta, idx_map[morph]))

    report: dict[str, dict] = {}
    for fam in FAMILIES:
        picks = pick_by_percentile(per_family[fam])
        report[fam] = {
            "n_candidates": len(per_family[fam]),
            "picks": {str(p): picks[p] for p in PERCENTILES},
            "delta_resets": {
                str(p): float(
                    next(c[1] for c in per_family[fam] if c[0] == picks[p])
                )
                for p in PERCENTILES
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT}")
    print()
    for fam in FAMILIES:
        print(f"  {fam}:")
        for p in PERCENTILES:
            name = report[fam]["picks"][str(p)]
            delta = report[fam]["delta_resets"][str(p)]
            print(f"    P{p:<3d} delta_reset={delta:+.3f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
