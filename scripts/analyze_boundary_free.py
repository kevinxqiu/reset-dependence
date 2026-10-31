#!/usr/bin/env python3
"""Pre-specified analysis of the boundary-free panel.

Implements the boundary-free audit gates documented in the supplement:

1. QC gate (cell): results.json exists, no NaN PSR, eval PSR finite,
   implementation_reset_rate <= 0.30.
2. QC gate (panel): at least 80/90 cells valid, each family >= 8/10 bodies
   with >= 2/3 valid seeds.
3. Reliability gate: sigma_seed(TRF) <= 2 * sigma_seed(tau=0) across bodies.
4. Primary statistics (on 30-body x 4-protocol x 3-seed matrix, where TRF
   is boundary-free and the other three are corpus tau in {0.00, 0.20, 0.50}):
   a. Paired Delta_F = median_body_F(PSR_TRF - PSR_tau0) with 95% bootstrap
      CI (10,000 resamples over bodies within family, seed=42).
   b. Spearman rho(TRF, tau=0) with 95% bootstrap CI.
   c. Two-way ANOVA: reward ~ C(protocol) * C(family) + C(morph),
      extract TRF protocol contrast + interaction with family.
5. Inclusion gates:
   - main_text: rho CI upper < 0.95 AND >=2/3 families Delta_F CI excludes 0
     AND reliability gate passes.
   - appendix_only: fails main but at least 1 family Delta_F CI excludes 0
     AND rho CI upper < 0.99.
   - drop: rho CI lower > 0.95 AND all Delta_F CIs contain 0.
   - reframe: >45/90 runs with implementation_reset_rate > 0.30.

Usage:
    python scripts/analyze_boundary_free.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

REPO = Path(__file__).resolve().parents[1]

# Morphology is nested in family, so in the blocked models the family main effect is
# aliased with the morphology fixed effects and the design is rank-deficient by
# construction; the tests of interest are unaffected. Recent statsmodels versions warn
# on every such fit, so that one expected warning is silenced here.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
CORPUS_ROOT = REPO / "results"
TRF_ROOT = REPO / "results/boundary_free"
PANEL_JSON = REPO / "records/boundary_free_bodies.json"
EVAL_KEY = "eval_tau020"
SEEDS = (42, 43, 44)
EVAL_KEYS_CHECKED = ("eval_tau020", "eval_tau050", "eval_neutral")
PSR_FIELDS = ("per_step_reward", "fitness", "distance")

# Corpus hyperparameters that the TRF cell must match.
EXPECTED_HPARAMS = {
    "num_envs": 128,
    "unroll_length": 16,
    "n_epochs": 10,
    "minibatch_size": 256,
    "timesteps": 5000000,
    "threshold": 0.0,
    "max_steps": 5000001,
    "termination_mode": "height",
}

# Cell-level QC thresholds.
MAX_RESET_RATE_CELL = 0.30

# Panel-level QC thresholds.
MIN_VALID_CELLS = 80  # out of 90
MIN_BODIES_PER_FAMILY = 8  # out of 10
MIN_VALID_SEEDS_PER_BODY = 2  # out of 3

# Reliability threshold.
RELIABILITY_RATIO_MAX = 2.0

# Reframe threshold: strictly more than 45/90 runs with rate > 0.30.
REFRAME_MIN_BAD_RUNS = 46

# Bootstrap count.
B = 10_000


def load_corpus_cell(threshold_str: str, seed: int) -> dict:
    """Load corpus results.json for mjx_ppo_{threshold}_seed{seed}/."""
    path = CORPUS_ROOT / f"mjx_ppo_{threshold_str}_seed{seed}" / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_trf_cell(seed: int) -> dict:
    """Load TRF results.json for boundary_free/seed{seed}/."""
    path = TRF_ROOT / f"seed{seed}" / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_psr(records: dict, body: str, eval_key: str = EVAL_KEY) -> float:
    """Return eval_key PSR; NaN if any PSR field in any eval block is NaN."""
    rec = records.get(body, {})
    for ek in EVAL_KEYS_CHECKED:
        blk = rec.get(ek)
        if not isinstance(blk, dict):
            return float("nan")
        for f in PSR_FIELDS:
            if f not in blk or not np.isfinite(blk[f]):
                return float("nan")
    v = rec.get(eval_key, {}).get("per_step_reward")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def get_reset_rate(records: dict, body: str) -> float:
    """Return implementation_reset_rate; NaN if missing or non-finite."""
    rec = records.get(body, {})
    v = rec.get("implementation_reset_rate")
    if v is None:
        return float("nan")
    v = float(v)
    return v if np.isfinite(v) else float("nan")


def verify_trf_metadata(records: dict) -> list[str]:
    """Return list of bodies failing TRF pre-reg hyperparameter check."""
    bad = []
    for body, rec in records.items():
        for k, v in EXPECTED_HPARAMS.items():
            if rec.get(k) != v:
                bad.append(f"{body}: {k}={rec.get(k)!r}, expected {v!r}")
                break
    return bad


def load_trf_long_df(panel_bodies: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Long-form: (body, family, seed, psr, reset_rate). TRF-only."""
    rows = []
    violations: list[str] = []
    for seed in SEEDS:
        recs = load_trf_cell(seed)
        violations.extend(
            f"[seed={seed}] {v}" for v in verify_trf_metadata(recs)
        )
        for entry in panel_bodies:
            body, family = entry["body"], entry["family"]
            rows.append(
                dict(
                    body=body,
                    family=family,
                    protocol="TRF",
                    seed=seed,
                    psr=get_psr(recs, body),
                    reset_rate=get_reset_rate(recs, body),
                )
            )
    return pd.DataFrame(rows), violations


def load_corpus_long_df(panel_bodies: list[dict]) -> pd.DataFrame:
    """Corpus records for the same 30 bodies under three taus."""
    rows = []
    for entry in panel_bodies:
        body, family = entry["body"], entry["family"]
        for protocol, tau in [("tau0", "0p00"), ("tau020", "0p20"), ("tau050", "0p50")]:
            for seed in SEEDS:
                recs = load_corpus_cell(tau, seed)
                rows.append(
                    dict(
                        body=body,
                        family=family,
                        protocol=protocol,
                        seed=seed,
                        psr=get_psr(recs, body),
                        reset_rate=float("nan"),
                    )
                )
    return pd.DataFrame(rows)


def cell_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Return TRF df with qc_pass column per row."""
    psr_ok = np.isfinite(df["psr"])
    rr_ok = df["reset_rate"].fillna(np.inf) <= MAX_RESET_RATE_CELL
    df = df.copy()
    df["qc_pass"] = psr_ok & rr_ok
    return df


def panel_qc(trf_df: pd.DataFrame) -> dict:
    """Apply panel-level QC: at least 80/90 cells valid AND each family >=8/10
    bodies with >=2/3 valid seeds."""
    total = len(trf_df)
    valid = int(trf_df["qc_pass"].sum())
    per_body_valid = trf_df.groupby(["family", "body"])["qc_pass"].sum()
    family_body_counts = {}
    for family in ("floor", "vt", "mvt"):
        bodies = [
            b for (fam, b), cnt in per_body_valid.items()
            if fam == family and cnt >= MIN_VALID_SEEDS_PER_BODY
        ]
        family_body_counts[family] = len(bodies)
    families_ok = all(
        cnt >= MIN_BODIES_PER_FAMILY for cnt in family_body_counts.values()
    )
    return dict(
        total_cells=total,
        valid_cells=valid,
        invalid_cells=total - valid,
        cell_threshold=MIN_VALID_CELLS,
        family_body_counts=family_body_counts,
        families_ok=families_ok,
        qc_pass=bool(valid >= MIN_VALID_CELLS and families_ok),
    )


def reframe_check(trf_df: pd.DataFrame) -> dict:
    """Reframe gate: strictly more than 45/90 runs have reset_rate > 0.30."""
    rates = trf_df["reset_rate"].values
    finite = rates[np.isfinite(rates)]
    bad_runs = int(np.sum(finite > MAX_RESET_RATE_CELL))
    return dict(
        total_runs=int(len(finite)),
        bad_runs=bad_runs,
        threshold=REFRAME_MIN_BAD_RUNS,
        reframe_fires=bool(bad_runs >= REFRAME_MIN_BAD_RUNS),
    )


def reliability_check(trf_df: pd.DataFrame, tau0_df: pd.DataFrame) -> dict:
    """sigma_seed(TRF) <= 2 * sigma_seed(tau=0) averaged over bodies."""
    trf_sigma = (
        trf_df.dropna(subset=["psr"])
        .groupby("body")["psr"]
        .std(ddof=1)
    )
    tau0_sigma = (
        tau0_df[tau0_df["protocol"] == "tau0"]
        .dropna(subset=["psr"])
        .groupby("body")["psr"]
        .std(ddof=1)
    )
    shared = sorted(set(trf_sigma.index) & set(tau0_sigma.index))
    if not shared:
        return dict(error="no shared bodies for reliability")
    trf_vals = np.array([trf_sigma[b] for b in shared])
    tau0_vals = np.array([tau0_sigma[b] for b in shared])
    finite = np.isfinite(trf_vals) & np.isfinite(tau0_vals) & (tau0_vals > 0)
    if not finite.any():
        return dict(error="no bodies with finite non-zero sigma")
    ratios = trf_vals[finite] / tau0_vals[finite]
    median_ratio = float(np.median(ratios))
    return dict(
        n_bodies=int(finite.sum()),
        median_ratio=median_ratio,
        threshold=RELIABILITY_RATIO_MAX,
        passes=bool(median_ratio <= RELIABILITY_RATIO_MAX),
    )


def per_body_psr(df: pd.DataFrame, protocol: str) -> dict[str, float]:
    """Seed-averaged PSR per body for a given protocol."""
    sub = df[df["protocol"] == protocol].dropna(subset=["psr"])
    return sub.groupby("body")["psr"].mean().to_dict()


def family_delta_ci(
    trf_df: pd.DataFrame, tau0_df: pd.DataFrame, family: str
) -> dict:
    """Paired Delta_F = median_body(PSR_TRF - PSR_tau0) with 95% bootstrap CI."""
    trf = trf_df[trf_df["family"] == family].dropna(subset=["psr"])
    tau0 = tau0_df[
        (tau0_df["family"] == family) & (tau0_df["protocol"] == "tau0")
    ].dropna(subset=["psr"])
    trf_mean = trf.groupby("body")["psr"].mean()
    tau0_mean = tau0.groupby("body")["psr"].mean()
    shared = sorted(set(trf_mean.index) & set(tau0_mean.index))
    if not shared:
        return dict(error=f"family {family}: no shared bodies")
    delta = np.array([trf_mean[b] - tau0_mean[b] for b in shared])
    if len(delta) == 0:
        return dict(error=f"family {family}: empty delta")
    rng = np.random.default_rng(42)
    boots = np.array(
        [np.median(rng.choice(delta, size=len(delta), replace=True)) for _ in range(B)]
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    median = float(np.median(delta))
    return dict(
        family=family,
        n_bodies=len(shared),
        median_delta=median,
        ci_lo=float(lo),
        ci_hi=float(hi),
        excludes_zero=bool((lo > 0.0) or (hi < 0.0)),
    )


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho with 10000-bootstrap 95% CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(error=f"n={n}, need >=3", n=n)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return dict(error="bootstrap no finite samples", rho=rho, n=n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rho=rho, ci_lo=float(lo), ci_hi=float(hi), n=n)


def twoway_anova(full_df: pd.DataFrame) -> dict:
    """reward ~ C(protocol) * C(family) + C(morph), extract TRF-related terms."""
    df = full_df.dropna(subset=["psr"]).rename(columns={"body": "morph"}).copy()
    if df.empty:
        return dict(error="no valid rows")
    model = ols("psr ~ C(protocol) * C(family) + C(morph)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    try:
        interaction = anova.loc["C(protocol):C(family)"]
        return dict(
            interaction_f=float(interaction["F"]),
            interaction_p=float(interaction["PR(>F)"]),
            df_num=float(interaction["df"]),
            df_denom=float(anova.loc["Residual"]["df"]),
            anova_table=anova.reset_index()
            .rename(columns={"index": "source"})
            .to_dict(orient="records"),
        )
    except KeyError:
        return dict(error="no protocol:family interaction row")


def apply_gates(rank_corr: dict, family_deltas: dict, reliability: dict, reframe: dict) -> dict:
    """Decide outcome using the pre-specified gates."""
    if reframe["reframe_fires"]:
        return dict(
            decision="reframe",
            reason=(
                f"{reframe['bad_runs']}/{reframe['total_runs']} runs have "
                f"implementation_reset_rate > 0.30 (threshold {reframe['threshold']})"
            ),
        )

    rho_lo = rank_corr.get("ci_lo", -1.0)
    rho_hi = rank_corr.get("ci_hi", 1.0)
    rho = rank_corr.get("rho", 0.0)

    families_excluding_zero = [
        f for f, d in family_deltas.items() if d.get("excludes_zero", False)
    ]
    n_families_excl = len(families_excluding_zero)
    all_include_zero = all(
        not d.get("excludes_zero", False) for d in family_deltas.values()
    )
    reliability_passes = reliability.get("passes", False)

    if rho_lo > 0.95 and all_include_zero:
        decision = "drop"
        reason = (
            f"rank CI lower={rho_lo:.3f} > 0.95, all Delta_F CIs contain 0 "
            f"(redundant with tau=0)"
        )
    elif (
        rho_hi < 0.95
        and n_families_excl >= 2
        and reliability_passes
    ):
        decision = "main_text"
        reason = (
            f"rank CI upper={rho_hi:.3f} < 0.95, {n_families_excl}/3 families "
            f"with Delta_F CI excluding 0, reliability passes"
        )
    elif n_families_excl >= 1 and rho_hi < 0.99:
        decision = "appendix_only"
        reason = (
            f"rank CI upper={rho_hi:.3f} < 0.99, {n_families_excl}/3 family "
            f"Delta_F CI excludes 0"
        )
    else:
        decision = "drop"
        reason = (
            f"rank rho={rho:.3f} CI=[{rho_lo:.3f}, {rho_hi:.3f}], "
            f"{n_families_excl}/3 family CIs excluding 0, "
            f"reliability_passes={reliability_passes}"
        )
    return dict(
        decision=decision,
        reason=reason,
        spearman_rho=rho,
        spearman_ci_lo=rho_lo,
        spearman_ci_hi=rho_hi,
        families_excluding_zero=families_excluding_zero,
        reliability_passes=reliability_passes,
    )


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    if not TRF_ROOT.is_dir():
        # Missing data is not a gate verdict: stop before overwriting the recorded analysis.
        print(f"ERROR: {TRF_ROOT} not found; unzip runs.zip from the dataset first", file=sys.stderr)
        sys.exit(1)
    if not PANEL_JSON.exists():
        print(f"ERROR: panel JSON not found at {PANEL_JSON}", file=sys.stderr)
        sys.exit(2)
    with open(PANEL_JSON) as f:
        panel = json.load(f)["panel"]

    trf_df, violations = load_trf_long_df(panel)
    corpus_df = load_corpus_long_df(panel)
    trf_df = cell_qc(trf_df)

    qc = panel_qc(trf_df)
    reframe = reframe_check(trf_df)
    reliability = reliability_check(trf_df, corpus_df)

    print(
        f"Panel: {len(panel)} bodies x 1 protocol x 3 seeds = {qc['total_cells']} cells; "
        f"valid={qc['valid_cells']}/{qc['total_cells']} (threshold {qc['cell_threshold']})"
    )
    print(f"Per-family valid body counts: {qc['family_body_counts']}")
    print(
        f"Reframe check: {reframe['bad_runs']}/{reframe['total_runs']} runs with "
        f"reset_rate > 0.30 (fires if >= {reframe['threshold']})"
    )

    if violations:
        print(f"\nMETADATA VIOLATIONS ({len(violations)}):")
        for v in violations[:10]:
            print(f"  {v}")

    if violations:
        decision = dict(
            decision="drop",
            reason=f"{len(violations)} metadata violations",
        )
    elif not qc["qc_pass"]:
        decision = dict(
            decision="drop",
            reason=(
                f"panel QC fail: valid_cells={qc['valid_cells']}/{qc['total_cells']}, "
                f"family_counts={qc['family_body_counts']}"
            ),
        )
    elif reframe["reframe_fires"]:
        decision = dict(
            decision="reframe",
            reason=(
                f"{reframe['bad_runs']}/{reframe['total_runs']} runs with "
                f"reset_rate > 0.30 (threshold {reframe['threshold']})"
            ),
        )
    else:
        # Corpus completeness gate: require all 30 bodies x tau=0 x 3 seeds
        # to have finite PSR (otherwise Delta/rho paired comparison silently
        # shrinks). Report missing cells explicitly.
        corpus_tau0 = corpus_df[corpus_df["protocol"] == "tau0"]
        total_corpus_cells = len(corpus_tau0)
        valid_corpus_cells = int(corpus_tau0["psr"].apply(np.isfinite).sum())
        corpus_complete = valid_corpus_cells >= 88  # allow 2 missing out of 90
        if not corpus_complete:
            decision = dict(
                decision="qc_fail",
                reason=(
                    f"corpus tau=0 incomplete: "
                    f"{valid_corpus_cells}/{total_corpus_cells} valid; "
                    "need >= 88 (allow 2 missing). Paired Delta/rho requires "
                    "matched corpus data."
                ),
            )
        else:
            trf_psr = per_body_psr(trf_df, "TRF")
            tau0_psr = per_body_psr(corpus_df, "tau0")
            shared = sorted(set(trf_psr) & set(tau0_psr))
            x = np.array([trf_psr[b] for b in shared])
            y = np.array([tau0_psr[b] for b in shared])
            rank_corr = spearman_with_ci(x, y)
            family_deltas = {
                fam: family_delta_ci(trf_df, corpus_df, fam)
                for fam in ("floor", "vt", "mvt")
            }
            anova_result = twoway_anova(pd.concat([trf_df, corpus_df], ignore_index=True))
            decision = apply_gates(rank_corr, family_deltas, reliability, reframe)
            decision["rank_correlation"] = rank_corr
            decision["family_deltas"] = family_deltas
            decision["anova"] = anova_result
            decision["corpus_completeness"] = dict(
                total=total_corpus_cells,
                valid=valid_corpus_cells,
                shared_bodies_with_trf=len(shared),
            )

    report = dict(
        qc=qc,
        reliability=reliability,
        reframe=reframe,
        metadata_violations=violations,
        decision=decision,
    )

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o).__name__}")

    out_path = Path("records/boundary_free_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"\nWrote {out_path}")
    print(f"\nDecision: {decision['decision']}")
    print(f"Reason: {decision.get('reason', 'n/a')}")


if __name__ == "__main__":
    main()
