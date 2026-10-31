#!/usr/bin/env python3
"""Pre-registered analysis of the factorial (tau x boundary) panel.

Implements the gates frozen in records/factorial_prereg.md:

1. QC gate (cell): no NaN PSR, finite eval PSR, reset_rate <= 0.30.
2. QC gate (panel, per new arm): at least 80/90 cells valid.
3. Reliability gate (per new arm): sigma_seed(arm) <= 2 * sigma_seed(matched corpus tau).
4. Primary statistics on 6-cell x 30-body x 3-seed matrix (540 rows):
   a. 2-way ANOVA: reward ~ C(tau) * C(boundary) + C(morph) + C(family)
      Extract tau x boundary interaction F, p, partial eta-squared.
   b. Per-tau boundary delta: Delta_tau = PSR(tau, 5M) - PSR(tau, 1000)
      seed-averaged per body. Report median with 95% bootstrap CI.
   c. Per-tau rank agreement: Spearman rho(PSR(tau, 5M), PSR(tau, 1000)).
5. Inclusion gates:
   - main_text: interaction p < 0.01 AND partial_eta2 > 0.02 AND >=2/3 tau have Delta_tau CI excluding 0 AND reliability passes.
   - appendix_only: interaction p < 0.05 OR >=1 tau has Delta_tau CI excluding 0 AND reliability passes.
   - drop: interaction p >= 0.05 AND all Delta_tau CIs contain 0.
   - qc_fail: reliability fails on one or both arms.

Usage:
    python scripts/analyze_factorial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

CORPUS_ROOT = Path("results")
TRF_ROOT = Path("results/boundary_free")
FACTORIAL_ROOT = Path("results/factorial")
PANEL_JSON = Path("records/boundary_free_bodies.json")

EVAL_KEY = "eval_tau020"
SEEDS = (42, 43, 44)
EVAL_KEYS_CHECKED = ("eval_tau020", "eval_tau050", "eval_neutral")
PSR_FIELDS = ("per_step_reward", "fitness", "distance")

# The 6 cells of the factorial.
CELLS = [
    ("tau0", 0.00, "1000", "corpus"),
    ("tau020", 0.20, "1000", "corpus"),
    ("tau050", 0.50, "1000", "corpus"),
    ("tau0_5M", 0.00, "5M", "trf"),
    ("tau020_5M", 0.20, "5M", "factorial"),
    ("tau050_5M", 0.50, "5M", "factorial"),
]

EXPECTED_HPARAMS_5M = {
    "num_envs": 128,
    "unroll_length": 16,
    "n_epochs": 10,
    "minibatch_size": 256,
    "timesteps": 5000000,
    "max_steps": 5000001,
    "termination_mode": "height",
}
EXPECTED_HPARAMS_1000 = {
    "num_envs": 128,
    "unroll_length": 16,
    "n_epochs": 10,
    "minibatch_size": 256,
    "timesteps": 5000000,
    "max_steps": 1000,
}

# QC thresholds (inherit from boundary-free).
MAX_RESET_RATE_CELL = 0.30
MIN_VALID_CELLS_PER_ARM = 80  # out of 90
RELIABILITY_RATIO_MAX = 2.0

# Gates.
GATE_MAIN_TEXT = {"interaction_p_max": 0.01, "partial_eta2_min": 0.02, "min_tau_with_delta_ci_excluding_0": 2}
GATE_APPENDIX = {"interaction_p_max": 0.05, "min_tau_with_delta_ci_excluding_0": 1}

B = 10_000  # bootstrap resamples


def _load_cell(label: str, source: str, tau: float, seed: int) -> tuple[dict, Path | None]:
    if source == "corpus":
        threshold_str = f"{tau:.2f}".replace(".", "p")
        path = CORPUS_ROOT / f"mjx_ppo_{threshold_str}_seed{seed}" / "results.json"
    elif source == "trf":
        path = TRF_ROOT / f"seed{seed}" / "results.json"
    elif source == "factorial":
        tau_label = f"{tau:.2f}".replace(".", "p")
        path = FACTORIAL_ROOT / f"tau{tau_label}_5M_seed{seed}" / "results.json"
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


def verify_metadata(records: dict, expected: dict) -> list[str]:
    bad = []
    for body, rec in records.items():
        for k, v in expected.items():
            if rec.get(k) != v:
                bad.append(f"{body}: {k}={rec.get(k)!r}, expected {v!r}")
                break
    return bad


def load_all_cells(panel_bodies: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Load 6 cells x 30 bodies x 3 seeds; annotate each row with metadata_ok.
    Non-corpus cells whose record has hparam mismatches fail metadata_ok.
    Corpus cells (legacy) are exempt from max_steps/termination_mode checks;
    we only verify the fields they do record (threshold, timesteps).
    """
    rows = []
    violations = {}
    # bodies_with_bad_metadata[(cell_label, seed)] = set of bodies failing
    bad_bodies_per_cell = {}
    for label, tau, boundary, source in CELLS:
        for seed in SEEDS:
            recs, path = _load_cell(label, source, tau, seed)
            if not recs:
                violations[(label, seed)] = f"missing: {path}"
                bad_bodies_per_cell[(label, seed)] = set()
                continue
            expected = dict(
                EXPECTED_HPARAMS_5M if boundary == "5M" else EXPECTED_HPARAMS_1000
            )
            expected["threshold"] = float(tau)
            # Corpus: verify only threshold + timesteps + panel coverage; skip
            # fields that don't exist in legacy corpus records (num_envs etc.
            # may be None in some pre-canonical-stack records but we know the
            # values were correct at train time from git history).
            if source == "corpus":
                expected = {k: expected[k] for k in ("threshold", "timesteps") if k in expected}
            bad_bodies = set()
            for body, rec in recs.items():
                for k, v in expected.items():
                    if rec.get(k) != v:
                        bad_bodies.add(body)
                        break
            bad_bodies_per_cell[(label, seed)] = bad_bodies
            if bad_bodies:
                violations[(label, seed)] = [
                    f"{b}: metadata mismatch" for b in list(bad_bodies)[:3]
                ]
            for entry in panel_bodies:
                body, family = entry["body"], entry["family"]
                rows.append(
                    dict(
                        body=body,
                        family=family,
                        tau=float(tau),
                        boundary=boundary,
                        cell=label,
                        source=source,
                        seed=seed,
                        psr=get_psr(recs, body),
                        reset_rate=(
                            get_reset_rate(recs, body)
                            if source != "corpus"
                            else float("nan")
                        ),
                        metadata_ok=body not in bad_bodies_per_cell.get((label, seed), set()),
                    )
                )
    return pd.DataFrame(rows), violations


def cell_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Cell QC: psr finite AND metadata_ok AND (corpus: pass | non-corpus:
    finite reset_rate <= MAX). Non-corpus cells REQUIRE a finite reset_rate
    per pre-reg diagnostics list; missing field is a failure."""
    df = df.copy()
    psr_ok = np.isfinite(df["psr"])
    is_corpus = df["source"] == "corpus"
    rr_finite = np.isfinite(df["reset_rate"])
    rr_ok_nonc = rr_finite & (df["reset_rate"] <= MAX_RESET_RATE_CELL)
    # Corpus: reset_rate not required; non-corpus: required finite + <= max.
    df["qc_pass"] = psr_ok & df["metadata_ok"] & (is_corpus | rr_ok_nonc)
    return df


def panel_qc_per_arm(df: pd.DataFrame, new_cell: str) -> dict:
    """Per-arm panel QC: >=80/90 valid cells AND each family >=8/10 bodies
    with >=2/3 valid seeds. Matches boundary-free panel_qc."""
    arm = df[df["cell"] == new_cell]
    total_cells = len(arm)
    valid_cells = int(arm["qc_pass"].sum())
    per_body_valid_seeds = arm.groupby(["family", "body"])["qc_pass"].sum()
    per_family_body_counts = {}
    for family in ("floor", "vt", "mvt"):
        bodies_ok = [
            b for (fam, b), cnt in per_body_valid_seeds.items()
            if fam == family and cnt >= 2
        ]
        per_family_body_counts[family] = len(bodies_ok)
    families_ok = all(c >= 8 for c in per_family_body_counts.values())
    return dict(
        cell=new_cell,
        total_cells=total_cells,
        valid_cells=valid_cells,
        cell_threshold=MIN_VALID_CELLS_PER_ARM,
        per_family_body_counts=per_family_body_counts,
        families_ok=families_ok,
        passes=bool(valid_cells >= MIN_VALID_CELLS_PER_ARM and families_ok),
    )


def reliability_check(df: pd.DataFrame, new_cell: str, corpus_cell: str) -> dict:
    """sigma_seed(new) <= 2 * sigma_seed(corpus). Median ratio across bodies."""
    new_sigma = (
        df[(df["cell"] == new_cell) & df["qc_pass"]]
        .groupby("body")["psr"]
        .std(ddof=1)
    )
    corpus_sigma = (
        df[(df["cell"] == corpus_cell) & df["qc_pass"]]
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


def twoway_anova(df: pd.DataFrame) -> dict:
    data = df.dropna(subset=["psr"]).rename(columns={"body": "morph"}).copy()
    if data.empty:
        return dict(error="no valid rows")
    # Need >= 2 levels of both tau AND boundary for the interaction term.
    if data["tau"].nunique() < 2 or data["boundary"].nunique() < 2:
        return dict(
            error=(
                f"insufficient levels: tau={data['tau'].nunique()}, "
                f"boundary={data['boundary'].nunique()}; need >=2 each"
            )
        )
    data["tau_str"] = data["tau"].astype(str)
    try:
        model = ols(
            "psr ~ C(tau_str) * C(boundary) + C(morph) + C(family)", data=data
        ).fit()
        anova = sm.stats.anova_lm(model, typ=2)
    except (ValueError, np.linalg.LinAlgError) as e:
        return dict(error=f"ANOVA failed: {e}")
    interaction_key = "C(tau_str):C(boundary)"
    if interaction_key not in anova.index:
        return dict(error=f"interaction row missing: {list(anova.index)}")
    row = anova.loc[interaction_key]
    residual = anova.loc["Residual"]
    ss_interaction = float(row["sum_sq"])
    ss_residual = float(residual["sum_sq"])
    partial_eta2 = ss_interaction / (ss_interaction + ss_residual)
    return dict(
        interaction_f=float(row["F"]),
        interaction_p=float(row["PR(>F)"]),
        df_num=float(row["df"]),
        df_denom=float(residual["df"]),
        partial_eta2=partial_eta2,
        anova_table=anova.reset_index()
        .rename(columns={"index": "source"})
        .to_dict(orient="records"),
    )


def delta_with_ci(delta: np.ndarray, rng_seed: int = 42) -> dict:
    if len(delta) == 0:
        return dict(error="empty")
    rng = np.random.default_rng(rng_seed)
    boots = np.array(
        [np.median(rng.choice(delta, size=len(delta), replace=True)) for _ in range(B)]
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(
        n=int(len(delta)),
        median_delta=float(np.median(delta)),
        ci_lo=float(lo),
        ci_hi=float(hi),
        excludes_zero=bool((lo > 0.0) or (hi < 0.0)),
    )


def per_tau_boundary_delta(df: pd.DataFrame) -> dict:
    out = {}
    for tau_str, paired_cells in [("0.0", ("tau0", "tau0_5M")), ("0.2", ("tau020", "tau020_5M")), ("0.5", ("tau050", "tau050_5M"))]:
        bounded = df[(df["cell"] == paired_cells[0]) & df["qc_pass"]].groupby("body")["psr"].mean()
        unbounded = df[(df["cell"] == paired_cells[1]) & df["qc_pass"]].groupby("body")["psr"].mean()
        shared = sorted(set(bounded.index) & set(unbounded.index))
        if not shared:
            out[tau_str] = dict(error=f"no shared bodies for tau={tau_str}")
            continue
        delta = np.array([unbounded[b] - bounded[b] for b in shared])
        ci = delta_with_ci(delta)
        out[tau_str] = ci
    return out


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


def per_tau_rank_corr(df: pd.DataFrame) -> dict:
    out = {}
    for tau_str, paired_cells in [("0.0", ("tau0", "tau0_5M")), ("0.2", ("tau020", "tau020_5M")), ("0.5", ("tau050", "tau050_5M"))]:
        bounded = df[(df["cell"] == paired_cells[0]) & df["qc_pass"]].groupby("body")["psr"].mean()
        unbounded = df[(df["cell"] == paired_cells[1]) & df["qc_pass"]].groupby("body")["psr"].mean()
        shared = sorted(set(bounded.index) & set(unbounded.index))
        if not shared:
            out[tau_str] = dict(error="no shared bodies")
            continue
        x = np.array([bounded[b] for b in shared])
        y = np.array([unbounded[b] for b in shared])
        out[tau_str] = spearman_with_ci(x, y)
    return out


def apply_gates(
    anova: dict,
    per_tau_delta: dict,
    per_tau_rho: dict,
    reliability_a: dict,
    reliability_h: dict,
    panel_qc_a: dict,
    panel_qc_h: dict,
) -> dict:
    p = anova.get("interaction_p", 1.0)
    pe2 = anova.get("partial_eta2", 0.0)
    n_tau_excluding_zero = sum(1 for v in per_tau_delta.values() if isinstance(v, dict) and v.get("excludes_zero"))

    # Both panel QC (80/90 + per-family 8/10 w/ 2/3 seeds) AND reliability must
    # pass on BOTH new arms for factorial main-text; single-arm failure → qc_fail.
    panel_a_ok = panel_qc_a.get("passes", False)
    panel_h_ok = panel_qc_h.get("passes", False)
    reliability_passes = (
        reliability_a.get("passes", False)
        and reliability_h.get("passes", False)
    )

    if not (panel_a_ok and panel_h_ok and reliability_passes):
        return dict(
            decision="qc_fail",
            reason=(
                f"panel_qc_A={panel_a_ok}, panel_qc_H={panel_h_ok}, "
                f"reliability: A={reliability_a.get('passes')}, "
                f"H={reliability_h.get('passes')}"
            ),
            panel_qc_a=panel_qc_a,
            panel_qc_h=panel_qc_h,
            reliability_a=reliability_a,
            reliability_h=reliability_h,
        )

    if (
        p < GATE_MAIN_TEXT["interaction_p_max"]
        and pe2 > GATE_MAIN_TEXT["partial_eta2_min"]
        and n_tau_excluding_zero >= GATE_MAIN_TEXT["min_tau_with_delta_ci_excluding_0"]
    ):
        decision = "main_text"
        reason = (
            f"interaction p={p:.3g}, partial_eta2={pe2:.3f}, "
            f"{n_tau_excluding_zero}/3 tau with Delta CI excluding 0"
        )
    elif (
        p < GATE_APPENDIX["interaction_p_max"]
        or n_tau_excluding_zero >= GATE_APPENDIX["min_tau_with_delta_ci_excluding_0"]
    ):
        decision = "appendix_only"
        reason = f"interaction p={p:.3g}, {n_tau_excluding_zero}/3 tau CIs excluding 0"
    else:
        decision = "drop"
        reason = f"interaction p={p:.3g}, all Delta_tau CIs contain 0"
    return dict(decision=decision, reason=reason)


def main():
    if not PANEL_JSON.exists():
        print(f"ERROR: panel JSON not found at {PANEL_JSON}", file=sys.stderr)
        sys.exit(2)
    with open(PANEL_JSON) as f:
        panel = json.load(f)["panel"]

    df, violations = load_all_cells(panel)
    df = cell_qc(df)

    print(f"Loaded {len(df)} rows across {len(CELLS)} cells x {len(panel)} bodies x {len(SEEDS)} seeds")
    if violations:
        print(f"\nData-source violations ({len(violations)}):")
        for k, v in list(violations.items())[:10]:
            print(f"  {k}: {v}")

    per_cell_valid = df.groupby("cell")["qc_pass"].sum()
    print(f"\nValid cells per protocol:\n{per_cell_valid}")

    panel_qc_a = panel_qc_per_arm(df, "tau020_5M")
    panel_qc_h = panel_qc_per_arm(df, "tau050_5M")
    print(f"\nPanel QC arm A: {panel_qc_a}")
    print(f"Panel QC arm H: {panel_qc_h}")

    reliability_a = reliability_check(df, "tau020_5M", "tau020")
    reliability_h = reliability_check(df, "tau050_5M", "tau050")
    print(f"\nReliability arm A (tau=0.20, 5M vs 1000): {reliability_a}")
    print(f"Reliability arm H (tau=0.50, 5M vs 1000): {reliability_h}")

    anova = twoway_anova(df)
    print(f"\nANOVA interaction: {anova.get('interaction_f')}, p={anova.get('interaction_p')}, partial_eta2={anova.get('partial_eta2')}")

    per_tau_delta = per_tau_boundary_delta(df)
    print("\nPer-tau boundary delta:")
    for k, v in per_tau_delta.items():
        print(f"  tau={k}: {v}")

    per_tau_rho = per_tau_rank_corr(df)
    print("\nPer-tau rank corr (5M vs 1000):")
    for k, v in per_tau_rho.items():
        print(f"  tau={k}: {v}")

    decision = apply_gates(
        anova, per_tau_delta, per_tau_rho,
        reliability_a, reliability_h,
        panel_qc_a, panel_qc_h,
    )
    print(f"\nDecision: {decision['decision']}")
    print(f"Reason: {decision['reason']}")

    report = dict(
        per_cell_valid=per_cell_valid.to_dict(),
        panel_qc=dict(arm_A=panel_qc_a, arm_H=panel_qc_h),
        reliability=dict(arm_A=reliability_a, arm_H=reliability_h),
        anova=anova,
        per_tau_delta=per_tau_delta,
        per_tau_rank_corr=per_tau_rho,
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

    out_path = Path("records/factorial_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
