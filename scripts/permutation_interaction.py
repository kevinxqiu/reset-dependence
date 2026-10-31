#!/usr/bin/env python3
"""Morphology-level permutation test for protocol x family interaction.

The blocked Type II ANOVA is a weak point of the paper's main
interaction test because morphology is nested in family, making the family
main effect aliased with morphology dummies. This script adds a non-parametric,
nesting-free complementary test by permuting family labels across morphologies.

Two permutation schemes are reported:

  (A) One-way ANOVA on per-morphology RF/Epi ratios, family labels
      shuffled across the 100 morphologies preserving family sizes.
      Observed F = 6.46 (paper); null distribution from B permutations.
      This is fully nesting-free: there is no morphology block, just
      100 ratios grouped by family.

  (B) Two-way blocked ANOVA on per-seed PSR (the primary interaction
      F-statistic from the paper), family labels shuffled across
      morphologies preserving family sizes (so the nesting is exact
      under the permuted labels). Observed F_int = 12.54. Fewer
      permutations because each fit is slower.

For each scheme we report:
  - observed F
  - permutation p = (1 + #{F_perm >= F_obs}) / (B + 1)
  - mean and 95th percentile of the permutation null distribution

Writes: records/permutation_interaction_report.json

Usage:
    python scripts/permutation_interaction.py                    # both
    python scripts/permutation_interaction.py --fast-only        # A only
    python scripts/permutation_interaction.py --B-fast 20000 \
                                              --B-blocked 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

# One BLAS thread by default: this script fits thousands of small regressions, and
# multithreaded BLAS spends most of their time on thread overhead (several times slower
# on many-core machines). Set the variables before running to override.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f_oneway
from statsmodels.formula.api import ols

REPO = Path(__file__).resolve().parents[1]

# Morphology is nested in family, so in the blocked models the family main effect is
# aliased with the morphology fixed effects and the design is rank-deficient by
# construction; the tests of interest are unaffected. Recent statsmodels versions warn
# on every such fit, so that one expected warning is silenced here.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
RESULTS = REPO / "results"
OUT = REPO / "records" / "permutation_interaction_report.json"

SEEDS = (42, 43, 44)
TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}


def family_of(name: str) -> str | None:
    if name.startswith("floor"):
        return "floor"
    if name.startswith("mvt"):
        return "mvt"
    if name.startswith("vt"):
        return "vt"
    return None


def load_per_seed_df() -> pd.DataFrame:
    """Load the 900-observation per-seed PSR dataframe."""
    rows = []
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with path.open() as f:
                data = json.load(f)
            for body, rec in data.items():
                psr = rec.get("eval_tau020", {}).get("per_step_reward")
                if psr is None:
                    continue
                fam = family_of(body)
                if fam is None:
                    continue
                rows.append(
                    {
                        "morph": body,
                        "family": fam,
                        "protocol": proto,
                        "seed": seed,
                        "psr": float(psr),
                    }
                )
    return pd.DataFrame(rows)


def load_ratio_df() -> pd.DataFrame:
    """Per-morphology seed-averaged RF/Epi ratios, one row per body."""
    agg: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with path.open() as f:
                data = json.load(f)
            for body, rec in data.items():
                psr = rec.get("eval_tau020", {}).get("per_step_reward")
                if psr is None:
                    continue
                agg[(body, proto)].append(float(psr))
    bodies = sorted({b for b, _ in agg.keys()})
    rows = []
    for b in bodies:
        fam = family_of(b)
        if fam is None:
            continue
        if (b, "episodic") not in agg or (b, "reset_free") not in agg:
            continue
        epi = np.mean(agg[(b, "episodic")])
        rf = np.mean(agg[(b, "reset_free")])
        if epi <= 0:
            continue
        rows.append({"morph": b, "family": fam, "rf_epi": rf / epi})
    return pd.DataFrame(rows)


def f_oneway_on_ratios(df: pd.DataFrame) -> float:
    groups = [df[df["family"] == f]["rf_epi"].values for f in ("floor", "vt", "mvt")]
    f, _ = f_oneway(*groups)
    return float(f)


def permute_family_labels(
    morphs: list[str], family_counts: dict[str, int], rng: np.random.Generator
) -> dict[str, str]:
    """Shuffle family labels across morphologies preserving family sizes."""
    labels = []
    for fam, n in family_counts.items():
        labels.extend([fam] * n)
    rng.shuffle(labels)
    return dict(zip(morphs, labels, strict=True))


def scheme_A_ratio_permutation(B: int, seed: int = 42) -> dict:
    """Fast scheme: one-way ANOVA on RF/Epi ratios, family shuffled."""
    df = load_ratio_df()
    n = len(df)
    family_counts = df["family"].value_counts().to_dict()
    morphs = df["morph"].tolist()
    f_obs = f_oneway_on_ratios(df)
    rng = np.random.default_rng(seed)
    f_null = np.empty(B)
    t0 = time.time()
    for i in range(B):
        perm_map = permute_family_labels(morphs, family_counts, rng)
        df_p = df.copy()
        df_p["family"] = df_p["morph"].map(perm_map)
        f_null[i] = f_oneway_on_ratios(df_p)
    elapsed = time.time() - t0
    p = (1 + int(np.sum(f_null >= f_obs))) / (B + 1)
    return {
        "scheme": "A: one-way ANOVA on RF/Epi ratios, family shuffled",
        "n_bodies": n,
        "family_counts": family_counts,
        "F_observed": f_obs,
        "B": B,
        "p_permutation": float(p),
        "null_mean": float(np.mean(f_null)),
        "null_p95": float(np.percentile(f_null, 95)),
        "null_max": float(np.max(f_null)),
        "elapsed_s": round(elapsed, 1),
    }


def _interaction_f(df: pd.DataFrame) -> float:
    """Type II interaction F for protocol:family in the blocked model."""
    model = ols("psr ~ C(protocol) * C(family) + C(morph)", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    return float(aov.loc["C(protocol):C(family)", "F"])


def scheme_B_blocked_permutation(B: int, seed: int = 42) -> dict:
    """Slow scheme: full blocked ANOVA refit under permuted family labels."""
    df = load_per_seed_df()
    morphs = sorted(df["morph"].unique())
    morph_to_family = {m: family_of(m) for m in morphs}
    family_counts = pd.Series(list(morph_to_family.values())).value_counts().to_dict()
    f_obs = _interaction_f(df)
    rng = np.random.default_rng(seed)
    f_null = np.empty(B)
    t0 = time.time()
    for i in range(B):
        perm_map = permute_family_labels(morphs, family_counts, rng)
        df_p = df.copy()
        df_p["family"] = df_p["morph"].map(perm_map)
        f_null[i] = _interaction_f(df_p)
        if (i + 1) % max(1, B // 20) == 0:
            print(f"  [blocked] {i + 1}/{B} ({(time.time() - t0):.1f}s elapsed)")
    elapsed = time.time() - t0
    p = (1 + int(np.sum(f_null >= f_obs))) / (B + 1)
    return {
        "scheme": "B: two-way blocked ANOVA F_int, family shuffled",
        "n_observations": len(df),
        "family_counts": family_counts,
        "F_observed": f_obs,
        "B": B,
        "p_permutation": float(p),
        "null_mean": float(np.mean(f_null)),
        "null_p95": float(np.percentile(f_null, 95)),
        "null_max": float(np.max(f_null)),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B-fast", type=int, default=10_000)
    # B=5000 matches the paper's reported blocked permutation (see
    # paper/sections/results.tex sec 4.4 and appendix sec:per_seed_appendix).
    # Monte Carlo SE on p=0.005 at B=5000 is ~0.001; increase to B=20000+
    # if quoting a more precise p_perm.
    parser.add_argument("--B-blocked", type=int, default=5_000)
    parser.add_argument("--fast-only", action="store_true")
    args = parser.parse_args()

    out = {}
    print("Scheme A: one-way ANOVA on RF/Epi ratios, family shuffled")
    a = scheme_A_ratio_permutation(args.B_fast)
    print(
        f"  F_obs = {a['F_observed']:.2f}, "
        f"null mean = {a['null_mean']:.2f}, null p95 = {a['null_p95']:.2f}, "
        f"p_perm = {a['p_permutation']:.4g} (B={a['B']}, {a['elapsed_s']}s)"
    )
    out["scheme_A"] = a

    if not args.fast_only:
        print(
            f"\nScheme B: two-way blocked ANOVA F_int, family shuffled (B={args.B_blocked})"
        )
        b = scheme_B_blocked_permutation(args.B_blocked)
        print(
            f"  F_obs = {b['F_observed']:.2f}, "
            f"null mean = {b['null_mean']:.2f}, null p95 = {b['null_p95']:.2f}, "
            f"p_perm = {b['p_permutation']:.4g} (B={b['B']}, {b['elapsed_s']}s)"
        )
        out["scheme_B"] = b

    # Embed a hash of the corpus per-seed input so downstream consumers
    # (e.g. scripts/reproduce_paper_stats.py) can detect stale cache.
    import hashlib  # noqa: PLC0415

    try:
        input_df = load_per_seed_df().sort_values(["morph", "protocol", "seed"])[
            ["morph", "protocol", "seed", "psr"]
        ]
        payload = input_df.to_csv(index=False).encode("utf-8")
        out["corpus_input_sha256_16"] = hashlib.sha256(payload).hexdigest()[:16]
        out["corpus_n_obs"] = int(len(input_df))
    except Exception as e:  # noqa: BLE001
        out["corpus_input_sha256_16"] = None
        out["corpus_input_hash_error"] = f"{type(e).__name__}: {e}"

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
