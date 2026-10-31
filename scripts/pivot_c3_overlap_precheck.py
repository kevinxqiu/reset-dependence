"""
C.3 pre-check: do top-10 by episodic PSR and top-10 by R_obs overlap?

design review point: if the two top-10 lists overlap heavily, C.3 downstream demo
is pointless. This script answers in 30 seconds from existing data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]


def main() -> None:
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            with open(RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json") as f:
                data[(tau, seed)] = json.load(f)
    morphs = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    n = len(morphs)
    assert n == 100

    # Episodic PSR = tau=0.20 policies evaluated under eval_tau020, seed-averaged
    psr_epi = np.array(
        [
            float(
                np.mean(
                    [
                        data[("0p20", s)][m]["eval_tau020"]["per_step_reward"]
                        for s in SEEDS
                    ]
                )
            )
            for m in morphs
        ]
    )

    # R_obs = 1 - seed-averaged fall_rate of tau=0.00 policies under eval_tau020
    fall_rate_rf = np.array(
        [
            float(
                np.mean(
                    [data[("0p00", s)][m]["eval_tau020"]["fall_rate"] for s in SEEDS]
                )
            )
            for m in morphs
        ]
    )
    r_obs = 1.0 - fall_rate_rf

    morphs_arr = np.array(morphs)

    print("=== C.3 Overlap Pre-check ===")
    print(f"n morphs = {n}")
    for k in (5, 10, 15, 20, 30):
        epi_top = set(morphs_arr[np.argsort(-psr_epi)[:k]])
        robs_top = set(morphs_arr[np.argsort(-r_obs)[:k]])
        overlap = epi_top & robs_top
        print(
            f"top-{k:2d}: epi_psr n={len(epi_top)}, r_obs n={len(robs_top)}, "
            f"intersection={len(overlap)}/{k} ({len(overlap) / k * 100:.0f}%)"
        )

    # Per-family breakdown for top-10
    k = 10
    epi_top10 = morphs_arr[np.argsort(-psr_epi)[:k]]
    robs_top10 = morphs_arr[np.argsort(-r_obs)[:k]]
    print(f"\nTop-{k} by episodic PSR:")
    for m in epi_top10:
        i = morphs.index(m)
        fam = m.split("-")[0]
        print(f"  [{fam}] {m[:50]:50s}  psr_epi={psr_epi[i]:.3f}  r_obs={r_obs[i]:.3f}")

    print(f"\nTop-{k} by R_obs:")
    for m in robs_top10:
        i = morphs.index(m)
        fam = m.split("-")[0]
        print(f"  [{fam}] {m[:50]:50s}  psr_epi={psr_epi[i]:.3f}  r_obs={r_obs[i]:.3f}")

    # Also check: the union is the set we'd need to train if we ran C.3 as planned
    union = set(epi_top10) | set(robs_top10)
    print(f"\nUnion of top-10 lists: {len(union)} unique morphs")
    print(f"If each morph runs 3 seeds = {len(union) * 3} training runs")


if __name__ == "__main__":
    main()
