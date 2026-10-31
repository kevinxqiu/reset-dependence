#!/usr/bin/env python3
"""Pre-specified analysis of the boundary duration sweep at tau=0.

Implements the boundary-duration audit gates documented in the supplement:

1. QC gate (cell): no NaN PSR, finite eval PSR, reset_rate <= 0.50.
2. QC gate (panel, per B value): at least 80/90 cells valid.
3. Reliability gate (per new B value): sigma_seed(B) <= 2 * sigma_seed(B=1000).
4. Primary statistics on 30-body x 4-B x 3-seed matrix:
   a. Pairwise rho(PSR_1000, PSR_B) for B in {10k, 100k, 5M} with 95% CI.
   b. Monotonicity: rho_10k > rho_100k > rho_5M (point estimates) AND
      at least one adjacent pair is non-overlapping (sharper decrease).
   c. Per-body Delta(B) = PSR_1000 - PSR_B vs log(B); report slope.
5. Gates:
   - main_text: >=2/3 rho CI upper < 0.95 AND monotonic AND reliability passes.
   - appendix_only: >=1/3 rho CI upper < 0.95 AND reliability passes.
   - drop: all rho CIs contain 0.95.
   - qc_fail: reliability fails.

Usage:
    python scripts/analyze_boundary_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# One BLAS thread by default: this script fits thousands of small regressions, and
# multithreaded BLAS spends most of their time on thread overhead (several times slower
# on many-core machines). Set the variables before running to override.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy import stats

CORPUS_ROOT = Path("results")
TRF_ROOT = Path("results/boundary_free")
SWEEP_ROOT = Path("results/boundary_sweep")
PANEL_JSON = Path("records/boundary_free_bodies.json")

EVAL_KEY = "eval_tau020"
SEEDS = (42, 43, 44)
EVAL_KEYS_CHECKED = ("eval_tau020", "eval_tau050", "eval_neutral")
PSR_FIELDS = ("per_step_reward", "fitness", "distance")

# The 4 boundary cells, all at tau=0. (label, B value, source)
# B=100001 was removed per pre-smoke clarification (it equals B=5M because
# per-env steps = 5M/128 = 39056 < 100001, so truncation never fires).
# Replaced with B=3001 for a true log-spaced intermediate.
CELLS = [
    ("B1000", 1000, "corpus"),
    ("B3001", 3001, "sweep"),
    ("B10001", 10001, "sweep"),
    ("B5M", 5_000_001, "trf"),
]

EXPECTED_HPARAMS_COMMON = {
    "num_envs": 128,
    "unroll_length": 16,
    "n_epochs": 10,
    "minibatch_size": 256,
    "timesteps": 5000000,
    "threshold": 0.0,
}

MAX_RESET_RATE_CELL = 0.50
MIN_VALID_CELLS_PER_B = 80  # out of 90
RELIABILITY_RATIO_MAX = 2.0
B = 10_000  # bootstrap resamples


def _load_cell(label: str, source: str, seed: int) -> tuple[dict, Path]:
    if source == "corpus":
        path = CORPUS_ROOT / f"mjx_ppo_0p00_seed{seed}" / "results.json"
    elif source == "trf":
        path = TRF_ROOT / f"seed{seed}" / "results.json"
    elif source == "sweep":
        path = SWEEP_ROOT / f"{label}_seed{seed}" / "results.json"
    else:
        raise ValueError(source)
    if not path.exists():
        return {}, path
    with open(path) as f:
        return json.load(f), path


def get_psr(records: dict, body: str) -> float:
    rec = records.get(body, {})
    for ek in EVAL_KEYS_CHECKED:
        blk = rec.get(ek)
        if not isinstance(blk, dict):
            return float("nan")
        for f in PSR_FIELDS:
            if f not in blk or not np.isfinite(blk[f]):
                return float("nan")
    v = rec.get(EVAL_KEY, {}).get("per_step_reward")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def get_reset_rate(records: dict, body: str) -> float:
    rec = records.get(body, {})
    v = rec.get("implementation_reset_rate")
    if v is None:
        return float("nan")
    v = float(v)
    return v if np.isfinite(v) else float("nan")


def verify_metadata(records: dict, expected: dict, B_expected: int, is_corpus: bool) -> list[str]:
    bad = []
    for body, rec in records.items():
        for k, v in expected.items():
            if rec.get(k) != v:
                bad.append(f"{body}: {k}={rec.get(k)!r}, expected {v!r}")
                break
        # Corpus has no max_steps field.
        if not is_corpus:
            ms = rec.get("max_steps")
            if ms != B_expected:
                bad.append(f"{body}: max_steps={ms!r}, expected {B_expected!r}")
    return bad


def load_all_cells(panel_bodies: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Load 4 cells x 30 bodies x 3 seeds. Annotate rows with metadata_ok
    (non-corpus cells with hparam mismatches fail metadata_ok; corpus cells
    are exempt from fields not logged in the original corpus records)."""
    rows = []
    violations = {}
    bad_bodies_per_cell: dict[tuple, set] = {}
    for label, B_val, source in CELLS:
        for seed in SEEDS:
            recs, path = _load_cell(label, source, seed)
            if not recs:
                violations[(label, seed)] = f"missing: {path}"
                bad_bodies_per_cell[(label, seed)] = set()
                continue
            bad = set()
            is_corpus = source == "corpus"
            for body, rec in recs.items():
                mismatch = False
                # The original corpus predates per-row hparam logging, so
                # required hparams may be absent. Accept corpus rows as long
                # as threshold matches; the corpus-wide hparams are pinned in
                # the corpus SLURM scripts.
                if is_corpus:
                    if rec.get("threshold") != EXPECTED_HPARAMS_COMMON["threshold"]:
                        mismatch = True
                else:
                    for k, v in EXPECTED_HPARAMS_COMMON.items():
                        if rec.get(k) != v:
                            mismatch = True
                            break
                    if not mismatch and rec.get("max_steps") != B_val:
                        mismatch = True
                if mismatch:
                    bad.add(body)
            bad_bodies_per_cell[(label, seed)] = bad
            if bad:
                violations[(label, seed)] = [
                    f"{b}: metadata mismatch" for b in list(bad)[:3]
                ]
            for entry in panel_bodies:
                body, family = entry["body"], entry["family"]
                rr = get_reset_rate(recs, body) if not is_corpus else float("nan")
                rows.append(
                    dict(
                        body=body,
                        family=family,
                        B=int(B_val),
                        B_label=label,
                        source=source,
                        seed=seed,
                        psr=get_psr(recs, body),
                        reset_rate=rr,
                        metadata_ok=body not in bad,
                    )
                )
    return pd.DataFrame(rows), violations


def cell_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Cell QC: psr finite AND metadata_ok AND (corpus | finite reset_rate <= MAX).
    Corpus rows are exempt from reset_rate checks; non-corpus rows require
    finite reset_rate per the pre-registered diagnostics."""
    df = df.copy()
    psr_ok = np.isfinite(df["psr"])
    is_corpus = df["source"] == "corpus"
    rr_finite = np.isfinite(df["reset_rate"])
    rr_ok_nonc = rr_finite & (df["reset_rate"] <= MAX_RESET_RATE_CELL)
    df["qc_pass"] = psr_ok & df["metadata_ok"] & (is_corpus | rr_ok_nonc)
    return df


def panel_qc_per_B(df: pd.DataFrame, label: str) -> dict:
    """Per-B panel QC: >=80/90 valid AND each family has >=8/10 bodies with
    >=2/3 valid seeds. Matches boundary-free family QC logic."""
    cell = df[df["B_label"] == label]
    total_cells = len(cell)
    valid_cells = int(cell["qc_pass"].sum())
    per_body_valid_seeds = cell.groupby(["family", "body"])["qc_pass"].sum()
    per_family_body_counts = {}
    for family in ("floor", "vt", "mvt"):
        bodies_ok = [
            b for (fam, b), cnt in per_body_valid_seeds.items()
            if fam == family and cnt >= 2
        ]
        per_family_body_counts[family] = len(bodies_ok)
    families_ok = all(c >= 8 for c in per_family_body_counts.values())
    return dict(
        cell=label,
        total_cells=total_cells,
        valid_cells=valid_cells,
        cell_threshold=MIN_VALID_CELLS_PER_B,
        per_family_body_counts=per_family_body_counts,
        families_ok=families_ok,
        passes=bool(valid_cells >= MIN_VALID_CELLS_PER_B and families_ok),
    )


def reliability_check(df: pd.DataFrame, new_label: str) -> dict:
    """sigma_seed(new) <= 2 * sigma_seed(corpus B=1000) median across bodies."""
    new_sigma = (
        df[(df["B_label"] == new_label) & df["qc_pass"]]
        .groupby("body")["psr"]
        .std(ddof=1)
    )
    corpus_sigma = (
        df[(df["B_label"] == "B1000") & df["qc_pass"]]
        .groupby("body")["psr"]
        .std(ddof=1)
    )
    shared = sorted(set(new_sigma.index) & set(corpus_sigma.index))
    if not shared:
        return dict(error="no shared bodies", passes=False)
    new_vals = np.array([new_sigma[b] for b in shared])
    corp_vals = np.array([corpus_sigma[b] for b in shared])
    finite = np.isfinite(new_vals) & np.isfinite(corp_vals) & (corp_vals > 0)
    if not finite.any():
        return dict(error="no finite non-zero sigma", passes=False)
    ratios = new_vals[finite] / corp_vals[finite]
    med = float(np.median(ratios))
    return dict(
        n_bodies=int(finite.sum()),
        median_ratio=med,
        threshold=RELIABILITY_RATIO_MAX,
        passes=bool(med <= RELIABILITY_RATIO_MAX),
    )


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(error=f"n={n}", n=n)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return dict(error="no finite bootstrap samples", rho=rho, n=n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rho=rho, ci_lo=float(lo), ci_hi=float(hi), n=n)


def pairwise_rho(df: pd.DataFrame) -> dict:
    """rho(PSR_1000, PSR_B) for B in {B3001, B10001, B5M}, seed-averaged per body."""
    corpus_psr = (
        df[(df["B_label"] == "B1000") & df["qc_pass"]]
        .groupby("body")["psr"]
        .mean()
    )
    out = {}
    for label in ("B3001", "B10001", "B5M"):
        new_psr = (
            df[(df["B_label"] == label) & df["qc_pass"]]
            .groupby("body")["psr"]
            .mean()
        )
        shared = sorted(set(corpus_psr.index) & set(new_psr.index))
        if not shared:
            out[label] = dict(error="no shared bodies")
            continue
        x = np.array([corpus_psr[b] for b in shared])
        y = np.array([new_psr[b] for b in shared])
        out[label] = spearman_with_ci(x, y)
    return out


def monotonicity_check(rhos: dict) -> dict:
    """Check rho_3001 > rho_10001 > rho_5M as point estimates, and at least
    one strict decrease has non-overlapping CIs."""
    seq = [rhos.get(k, {}) for k in ("B3001", "B10001", "B5M")]
    points = [r.get("rho") for r in seq]
    if any(p is None for p in points):
        return dict(error="missing rho values", monotonic=False)
    pe_ok = points[0] > points[1] > points[2]
    # Non-overlapping: rho_i's CI_lo > rho_{i+1}'s CI_hi
    def _ci_gap(a, b):
        if "ci_lo" not in a or "ci_hi" not in b:
            return False
        return a["ci_lo"] > b["ci_hi"]
    strict_01 = _ci_gap(seq[0], seq[1])
    strict_12 = _ci_gap(seq[1], seq[2])
    return dict(
        points=points,
        point_estimates_monotonic=bool(pe_ok),
        strict_3001_to_10001=bool(strict_01),
        strict_10001_to_5M=bool(strict_12),
        any_strict=bool(strict_01 or strict_12),
        pass_gate=bool(pe_ok and (strict_01 or strict_12)),
    )


def per_body_delta_vs_logB(df: pd.DataFrame, valid_Bs: list[str]) -> dict:
    """For each body, compute Delta(B) = PSR_1000 - PSR_B; fit
    Delta ~ log10(B) + body (body fixed effects) using statsmodels, then
    compute body-clustered bootstrap CI on the log10(B) slope (B=10000).
    This matches the pre-specified boundary-duration statistic."""
    rows = []
    for label, B_val, _ in CELLS[1:]:  # skip B=1000
        if label not in valid_Bs:
            continue
        for body, sub in df[(df["B_label"] == label) & df["qc_pass"]].groupby("body"):
            mean_psr = sub["psr"].mean()
            corpus_psr = df[
                (df["B_label"] == "B1000") & (df["body"] == body) & df["qc_pass"]
            ]["psr"].mean()
            if not np.isfinite(mean_psr) or not np.isfinite(corpus_psr):
                continue
            rows.append(
                dict(
                    body=body,
                    B=B_val,
                    logB=np.log10(B_val),
                    delta=corpus_psr - mean_psr,
                )
            )
    if len(rows) < 10:
        return dict(error=f"only {len(rows)} valid rows; skipping regression")
    tbl = pd.DataFrame(rows)
    # Delta ~ log10(B) + C(body): body fixed effects remove per-body intercepts.
    from statsmodels.formula.api import ols
    model = ols("delta ~ logB + C(body)", data=tbl).fit()
    slope = float(model.params["logB"])
    slope_se = float(model.bse["logB"])
    # Body-clustered bootstrap on slope.
    unique_bodies = tbl["body"].unique().tolist()
    rng = np.random.default_rng(42)
    boot_slopes = []
    for _ in range(B):
        sampled = rng.choice(unique_bodies, size=len(unique_bodies), replace=True)
        sub = pd.concat(
            [tbl[tbl["body"] == b].assign(bid=i) for i, b in enumerate(sampled)],
            ignore_index=True,
        )
        try:
            m = ols("delta ~ logB + C(bid)", data=sub).fit()
            s = float(m.params["logB"])
            if np.isfinite(s):
                boot_slopes.append(s)
        except Exception:
            continue
    if not boot_slopes:
        return dict(
            n_rows=len(rows),
            slope=slope,
            slope_se=slope_se,
            error="bootstrap failed",
        )
    lo, hi = np.percentile(boot_slopes, [2.5, 97.5])
    return dict(
        n_rows=len(rows),
        n_bodies=len(unique_bodies),
        slope=slope,
        slope_se=slope_se,
        slope_ci_lo=float(lo),
        slope_ci_hi=float(hi),
        slope_ci_excludes_zero=bool((lo > 0.0) or (hi < 0.0)),
    )


def apply_gates(
    rhos: dict,
    monotonic: dict,
    reliabilities: dict,
    valid_Bs: list[str],
    panel_qc_per_new_B: dict,
) -> dict:
    """Main-text inclusion requires all 3 non-corpus
    cells pass reliability + panel QC (3/3). Appendix-only allows partial
    data; QC fail if <2 remain."""
    all_new_Bs = ("B3001", "B10001", "B5M")
    dropped_Bs = [b for b in all_new_Bs if b not in valid_Bs]

    if len(valid_Bs) < 2:
        return dict(
            decision="qc_fail",
            reason=f"<2 valid B values remain; dropped={dropped_Bs}",
            dropped_Bs=dropped_Bs,
            valid_Bs=valid_Bs,
        )

    n_rhos_below_095 = sum(
        1
        for label, r in rhos.items()
        if label in valid_Bs
        and isinstance(r, dict)
        and r.get("ci_hi", 1.0) < 0.95
    )

    # Main-text requires ALL 3 non-corpus cells valid (all pass QC and
    # reliability) AND >= 2 rho CIs < 0.95 AND monotonic.
    if (
        len(valid_Bs) == 3
        and n_rhos_below_095 >= 2
        and monotonic.get("pass_gate", False)
    ):
        decision = "main_text"
        reason = (
            f"all 3 non-corpus Bs passed QC+reliability, "
            f"{n_rhos_below_095}/3 rho CIs below 0.95, monotonic, "
            f"strict: 3k->10k={monotonic.get('strict_3001_to_10001')}, "
            f"10k->5M={monotonic.get('strict_10001_to_5M')}"
        )
    elif n_rhos_below_095 >= 1:
        decision = "appendix_only"
        reason = (
            f"{n_rhos_below_095}/{len(valid_Bs)} rho CIs below 0.95, "
            f"{len(dropped_Bs)} B(s) dropped ({dropped_Bs}); "
            "main-text requires all 3 non-corpus valid"
        )
    else:
        decision = "drop"
        reason = "all rho CIs contain 0.95 (no boundary effect detected)"
    return dict(
        decision=decision,
        reason=reason,
        dropped_Bs=dropped_Bs,
        valid_Bs=valid_Bs,
    )


def main():
    if not PANEL_JSON.exists():
        print(f"ERROR: panel JSON not found at {PANEL_JSON}", file=sys.stderr)
        sys.exit(2)
    with open(PANEL_JSON) as f:
        panel = json.load(f)["panel"]

    df, violations = load_all_cells(panel)
    df = cell_qc(df)

    print(f"Loaded {len(df)} rows across 4 B-values x 30 bodies x 3 seeds")
    if violations:
        print(f"\nData-source violations ({len(violations)}):")
        for k, v in list(violations.items())[:10]:
            print(f"  {k}: {v}")

    per_cell_valid = df.groupby("B_label")["qc_pass"].sum()
    print(f"\nValid cells per B:\n{per_cell_valid}")

    # Panel-level QC per pre-reg: >=80/90 valid per B AND >=8/10 bodies
    # per family with >=2/3 valid seeds.
    panel_qc_per_new_B = {}
    for label in ("B3001", "B10001", "B5M"):
        panel_qc_per_new_B[label] = panel_qc_per_B(df, label)
    print("\nPanel QC per new B:")
    for k, v in panel_qc_per_new_B.items():
        print(f"  {k}: valid_cells={v['valid_cells']}/90, "
              f"per_family={v['per_family_body_counts']}, passes={v['passes']}")

    # Reliability gate per new/non-corpus cell.
    reliabilities = {}
    for label in ("B3001", "B10001", "B5M"):
        reliabilities[label] = reliability_check(df, label)
    print("\nReliability gates:")
    for k, v in reliabilities.items():
        print(f"  {k}: {v}")

    # valid_Bs = new cells that passed BOTH panel QC and reliability.
    valid_Bs = []
    for label in ("B3001", "B10001", "B5M"):
        if not panel_qc_per_new_B[label]["passes"]:
            continue
        if not reliabilities[label].get("passes", False):
            continue
        valid_Bs.append(label)
    print(f"\nValid B values for analysis: {valid_Bs}")

    rhos = pairwise_rho(df)
    print("\nPairwise rho (vs corpus B=1000):")
    for k, v in rhos.items():
        print(f"  {k}: {v}")

    monotonic = monotonicity_check(rhos)
    print(f"\nMonotonicity: {monotonic}")

    dose_resp = per_body_delta_vs_logB(df, valid_Bs)
    print(f"\nPer-body Delta(B) ~ log(B) regression: {dose_resp}")

    decision = apply_gates(rhos, monotonic, reliabilities, valid_Bs, panel_qc_per_new_B)
    print(f"\nDecision: {decision['decision']}")
    print(f"Reason: {decision['reason']}")

    report = dict(
        per_cell_valid=per_cell_valid.to_dict(),
        panel_qc_per_new_B=panel_qc_per_new_B,
        reliability=reliabilities,
        valid_Bs=valid_Bs,
        pairwise_rho=rhos,
        monotonicity=monotonic,
        dose_response=dose_resp,
        decision=decision,
        metadata_violations={str(k): v for k, v in violations.items()},
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
        raise TypeError(type(o).__name__)

    out_path = Path("records/boundary_sweep_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
