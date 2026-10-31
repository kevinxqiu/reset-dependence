#!/usr/bin/env python3
"""Select the 30-body boundary-free panel.

The boundary-free panel uses family-by-Delta-reset stratification with
seed=42. This preserves coverage of bodies that are easy, moderate, and hard
under reset-free training while keeping the panel small enough for the
persistent no-boundary follow-up.

This script:
1. Regenerates the panel using deterministic family x Delta-reset quintile
   stratification.
2. Writes records/boundary_free_bodies.json.

Run once to create the fixed panel; the script is retained for reproducibility.

Usage:
    python scripts/pick_boundary_free_bodies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
FEATURES_JSON = RESULTS_DIR / "morph_features.json"
OUT_JSON = REPO / "records/boundary_free_bodies.json"

SEED = 42
PER_FAMILY = 10
N_QUINTILES = 5
PER_QUINTILE = PER_FAMILY // N_QUINTILES  # 2

FAMILIES = ("floor", "vt", "mvt")
SEEDS = (42, 43, 44)


def load_psr(train_tau: str, seed: int, body: str, eval_key: str) -> float:
    path = RESULTS_DIR / f"mjx_ppo_{train_tau}_seed{seed}" / "results.json"
    with open(path) as f:
        data = json.load(f)
    blk = data.get(body, {}).get(eval_key)
    if not isinstance(blk, dict) or "per_step_reward" not in blk:
        return float("nan")
    return float(blk["per_step_reward"])


def mean_psr(train_tau: str, body: str, eval_key: str) -> float:
    vals = [load_psr(train_tau, s, body, eval_key) for s in SEEDS]
    return float(np.nanmean(vals))


def build_panel() -> list[dict]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    bodies = sorted(feats.keys())

    eval_key = "eval_tau020"
    delta_reset: dict[str, float] = {}
    for body in bodies:
        psr_epi = mean_psr("0p20", body, eval_key)
        psr_rf = mean_psr("0p00", body, eval_key)
        if np.isnan(psr_epi) or np.isnan(psr_rf):
            continue
        delta_reset[body] = psr_epi - psr_rf

    by_family: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for body in delta_reset:
        fam = body.split("-")[0]
        if fam in by_family:
            by_family[fam].append(body)

    rng = np.random.default_rng(SEED)
    panel: list[dict] = []
    for fam in FAMILIES:
        candidates = by_family[fam]
        if len(candidates) < PER_FAMILY:
            raise RuntimeError(
                f"family {fam} has {len(candidates)} bodies, need {PER_FAMILY}"
            )
        ranked = sorted(candidates, key=lambda b: delta_reset[b])
        quintiles = np.array_split(ranked, N_QUINTILES)
        for q_idx, q_bodies in enumerate(quintiles):
            q_list = list(q_bodies)
            pick = rng.choice(q_list, size=PER_QUINTILE, replace=False)
            for body in pick:
                panel.append(
                    {
                        "body": body,
                        "family": fam,
                        "quintile": q_idx,
                        "delta_reset": float(delta_reset[body]),
                    }
                )
    panel.sort(key=lambda d: (d["family"], d["quintile"], d["body"]))
    return panel


def main() -> int:
    panel = build_panel()

    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "per_family": PER_FAMILY,
                "per_quintile": PER_QUINTILE,
                "n_quintiles": N_QUINTILES,
                "eval_key_for_delta_reset": "eval_tau020",
                "delta_reset_formula": (
                    "PSR(train_tau=0p20) - PSR(train_tau=0p00), both at eval_tau020"
                ),
                "panel": panel,
            },
            f,
            indent=2,
        )
    print(
        f"Wrote {OUT_JSON} with {len(panel)} bodies "
        f"({PER_FAMILY} per family x {len(FAMILIES)} families)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
