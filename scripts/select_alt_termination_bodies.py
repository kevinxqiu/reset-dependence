#!/usr/bin/env python3
"""Deterministic selector for the 30-body alt-termination panel.

Loads per-body Delta_reset from the corpus (seed-averaged), partitions
each family into 5 quintiles, and selects 2 bodies per quintile with
numpy Generator(seed=42). Writes the panel to
``records/alt_termination_bodies.json``.

Run once, at spec-freeze time. The output JSON is itself the frozen
panel; the script is retained for reproducibility.

See ``records/alt_termination_spec.md`` for the full
pre-registration.

Usage:
    python scripts/select_alt_termination_bodies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
FEATURES_JSON = RESULTS_DIR / "morph_features.json"
SEED = 42
PER_FAMILY = 10
N_QUINTILES = 5
PER_QUINTILE = PER_FAMILY // N_QUINTILES  # 2

FAMILIES = ("floor", "vt", "mvt")

TRAIN_TAUS = ("0p00", "0p20", "0p50")
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


def main():
    with open(FEATURES_JSON) as f:
        feats = json.load(f)

    bodies = sorted(feats.keys())

    # Delta_reset = PSR_episodic - PSR_reset_free, common eval_tau020
    # (matches scripts/asymptotic_gate_analysis.py definition)
    eval_key = "eval_tau020"
    delta_reset = {}
    for body in bodies:
        psr_epi = mean_psr("0p20", body, eval_key)
        psr_rf = mean_psr("0p00", body, eval_key)
        if np.isnan(psr_epi) or np.isnan(psr_rf):
            continue
        delta_reset[body] = psr_epi - psr_rf

    # Group by family (family encoded in body name prefix)
    by_family: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for body in delta_reset:
        fam = body.split("-")[0]
        if fam in by_family:
            by_family[fam].append(body)

    rng = np.random.default_rng(SEED)
    panel = []

    for fam in FAMILIES:
        candidates = by_family[fam]
        if len(candidates) < PER_FAMILY:
            raise RuntimeError(
                f"family {fam} has only {len(candidates)} bodies, need {PER_FAMILY}"
            )
        # Rank by Delta_reset, split into N_QUINTILES bins
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

    # Deterministic order (family, quintile, body) for reproducibility
    panel.sort(key=lambda d: (d["family"], d["quintile"], d["body"]))

    out_path = Path("records/alt_termination_bodies.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "per_family": PER_FAMILY,
                "per_quintile": PER_QUINTILE,
                "n_quintiles": N_QUINTILES,
                "eval_key_for_delta_reset": eval_key,
                "delta_reset_formula": "PSR(train_tau=0p20) - PSR(train_tau=0p00), both at eval_tau020",
                "panel": panel,
            },
            f,
            indent=2,
        )
    print(
        f"Wrote {out_path} with {len(panel)} bodies "
        f"({PER_FAMILY} per family x {len(FAMILIES)} families)"
    )


if __name__ == "__main__":
    main()
