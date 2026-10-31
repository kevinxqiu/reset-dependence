#!/usr/bin/env python3
"""Structural-feature regression for protocol sensitivity (M1 / M2 / M3).

Fits structural-feature predictors of per-body protocol sensitivity on the
100-body UNIMAL corpus and evaluates whether those predictors explain the
observed strict-vs-reset-free gap.

Target (paired delta, avoids ratio small-denominator issues):
    Delta_str_rf(m) = PSR_strict(m; tau_eval=0.20) - PSR_rf(m; tau_eval=0.20)

Models:
    M1: Delta ~ PSR_epi                             (baseline-skill only)
    M2: Delta ~ init_height + n_first_level_limbs + tree_depth
               + n_joints + n_bodies                (pre-registered M2)
    M3: Delta ~ PSR_epi + M2 features               (combined)

The script:
  - Fits all three on the 100-body UNIMAL corpus.
  - Reports in-sample R^2, adjusted R^2, AIC, Spearman rho between predicted
    and observed Delta, bootstrap 95% CIs.
  - Serialises coefficients + feature means/stds to
    `records/structural_regression_coefs.json`.
  - Optional zero-shot mode: --features <features.json> --results-dir
    <dir with tau0p00/0p20/0p50 seed runs> reports Spearman rho of prediction
    vs observation on the held-out body set.

Usage:
    # Fit + in-sample report (UNIMAL)
    python scripts/structural_feature_regression.py

    # Fit on UNIMAL, then evaluate zero-shot on a held-out body set
    python scripts/structural_feature_regression.py \\
        --zero-shot-features results/morph_features_heldout.json \\
        --zero-shot-results  results/heldout_panel \\
        --zero-shot-tau-labels tau0p00,tau0p20,tau0p50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
NOTES = REPO / "records"
OUT_COEFS = NOTES / "structural_regression_coefs.json"
OUT_REPORT = NOTES / "structural_regression_report.json"

SEEDS_UNIMAL = (42, 43, 44)

# Feature set used by the structural model reported in the appendix.
M2_FEATURES = (
    "init_height",
    "n_first_level_limbs",
    "tree_depth",
    "n_joints",
    "n_bodies",
)

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42


def load_corpus_psr(tau_label: str, seeds=SEEDS_UNIMAL) -> dict[str, float]:
    """Seed-averaged PSR at eval_tau020 for a given training tau.

    Divergent cells (NaN / Inf PSR, e.g. MuJoCo simulator blow-up) are dropped
    from the seed average. A body with one bad seed uses the other two. A body
    with all seeds divergent is omitted from the returned dict.
    """
    vals: dict[str, list[float]] = {}
    for s in seeds:
        path = RESULTS / f"mjx_ppo_{tau_label}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get("eval_tau020", {}).get("per_step_reward")
            if psr is None:
                continue
            psr_f = float(psr)
            if np.isfinite(psr_f):
                vals.setdefault(body, []).append(psr_f)
    return {b: float(np.mean(v)) for b, v in vals.items() if v}


def load_arbitrary_psr(
    results_dir: Path, tau_labels: tuple[str, str, str]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Generic PSR loader for any `{results_dir}/{body}_{tau_label}_seed{s}/results.json`.

    Supports held-out layouts where bodies are first-level directories and
    results files contain eval_tau020. Falls back to a two-level `{tau}/{body}`
    layout if the first probe fails. Seed-averages across whatever seeds are
    present (minimum 1).
    """
    out: list[dict[str, float]] = [{}, {}, {}]
    for i, tau in enumerate(tau_labels):
        agg: dict[str, list[float]] = {}
        # Layout A (corpus-style): {results_dir}/{stem}_{tau}_seed{s}/results.json
        for d in sorted(results_dir.glob(f"*_{tau}_seed*")):
            if not d.is_dir():
                continue
            rp = d / "results.json"
            if not rp.exists():
                continue
            with rp.open() as f:
                data = json.load(f)
            for body, rec in data.items():
                psr = rec.get("eval_tau020", {}).get("per_step_reward")
                if psr is None:
                    continue
                psr_f = float(psr)
                if np.isfinite(psr_f):
                    agg.setdefault(body, []).append(psr_f)
        if not agg:
            # Layout B (per-tau subdir): {results_dir}/{tau}_seed{s}/results.json
            for d in sorted(results_dir.glob(f"{tau}_seed*")):
                if not d.is_dir():
                    continue
                rp = d / "results.json"
                if not rp.exists():
                    continue
                with rp.open() as f:
                    data = json.load(f)
                for body, rec in data.items():
                    psr = rec.get("eval_tau020", {}).get("per_step_reward")
                    if psr is None:
                        continue
                    psr_f = float(psr)
                    if np.isfinite(psr_f):
                        agg.setdefault(body, []).append(psr_f)
        out[i] = {b: float(np.mean(v)) for b, v in agg.items() if v}
    return out[0], out[1], out[2]


def build_design(
    bodies: list[str],
    features: dict[str, dict[str, float]],
    psr_epi: dict[str, float],
    model: str,
) -> np.ndarray:
    """Return design matrix with intercept column for the requested model.

    Bodies missing from `psr_epi` get NaN for the PSR_epi predictor; the
    downstream finite-mask in `zero_shot_eval` will drop those rows for M1/M3.
    M2 is morphology-only so does not consult `psr_epi` and is unaffected.
    """
    rows = []
    for b in bodies:
        row = [1.0]  # intercept
        if model in ("M1", "M3"):
            row.append(psr_epi.get(b, float("nan")))
        if model in ("M2", "M3"):
            for fn in M2_FEATURES:
                row.append(float(features[b][fn]))
        rows.append(row)
    return np.array(rows)


def column_names(model: str) -> list[str]:
    cols = ["intercept"]
    if model in ("M1", "M3"):
        cols.append("PSR_epi")
    if model in ("M2", "M3"):
        cols.extend(M2_FEATURES)
    return cols


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Ordinary least squares with diagnostics.

    Returns beta and a dict with R^2, adj R^2, AIC, residual std, and the
    raw predictions (so downstream callers can Spearman-correlate without
    re-running).
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n, p = X.shape
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else float("nan")
    # AIC under Gaussian assumption
    sigma2 = ss_res / n
    aic = n * np.log(sigma2) + 2 * p if sigma2 > 0 else float("nan")
    return beta, {
        "n": int(n),
        "p": int(p),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "aic": float(aic),
        "resid_std": float(np.std(resid)),
        "y_hat": y_hat.tolist(),
    }


def standardise_features(
    X: np.ndarray, feature_cols: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise selected columns; returns (X_std, means, stds) for the selected columns only."""
    X_std = X.copy()
    means = X[:, feature_cols].mean(axis=0)
    stds = X[:, feature_cols].std(axis=0, ddof=0)
    stds_safe = np.where(stds > 0, stds, 1.0)
    X_std[:, feature_cols] = (X[:, feature_cols] - means) / stds_safe
    return X_std, means, stds_safe


def bootstrap_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(B, n))
    rhos = np.empty(B)
    for i in range(B):
        ii = idx[i]
        rho, _ = spearmanr(y_true[ii], y_pred[ii])
        rhos[i] = rho if not np.isnan(rho) else 0.0
    point, _ = spearmanr(y_true, y_pred)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fit_all_models(
    bodies: list[str],
    features: dict[str, dict[str, float]],
    psr_epi: dict[str, float],
    psr_strict: dict[str, float],
    psr_rf: dict[str, float],
) -> dict:
    """Fit M1, M2, M3 on the supplied body set; return standardised coefficients + diagnostics."""
    y = np.array([psr_strict[b] - psr_rf[b] for b in bodies])
    models = {}
    for mname in ("M1", "M2", "M3"):
        X_raw = build_design(bodies, features, psr_epi, mname)
        cols = column_names(mname)
        # Standardise all non-intercept columns so coefficient magnitudes
        # are directly comparable.
        feat_cols = list(range(1, X_raw.shape[1]))
        X_std, means, stds = standardise_features(X_raw, feat_cols)
        beta, diag = fit_ols(X_std, y)
        point_rho, lo_rho, hi_rho = bootstrap_spearman(y, np.array(diag["y_hat"]))
        pearson_r, pearson_p = pearsonr(y, np.array(diag["y_hat"]))
        models[mname] = {
            "columns": cols,
            "beta": beta.tolist(),
            "feature_means": [0.0] + means.tolist(),
            "feature_stds": [1.0] + stds.tolist(),
            "in_sample": {
                **{k: v for k, v in diag.items() if k != "y_hat"},
                "spearman_rho": point_rho,
                "spearman_ci": [lo_rho, hi_rho],
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
            },
        }
    return {
        "bodies": bodies,
        "n": len(bodies),
        "target": "Delta_str_rf = PSR_strict - PSR_rf (both eval_tau020)",
        "features_M2": list(M2_FEATURES),
        "models": models,
        "y_observed": y.tolist(),
    }


def predict(
    coefs: dict,
    bodies: list[str],
    features: dict[str, dict[str, float]],
    psr_epi: dict[str, float],
    model: str,
) -> np.ndarray:
    """Apply a fitted model to a held-out body set."""
    X_raw = build_design(bodies, features, psr_epi, model)
    means = np.array(coefs["models"][model]["feature_means"])
    stds = np.array(coefs["models"][model]["feature_stds"])
    X_std = (X_raw - means) / stds
    beta = np.array(coefs["models"][model]["beta"])
    return X_std @ beta


def zero_shot_eval(
    coefs: dict,
    bodies: list[str],
    features: dict[str, dict[str, float]],
    psr_epi: dict[str, float],
    psr_strict: dict[str, float],
    psr_rf: dict[str, float],
) -> dict:
    """Evaluate all three fitted models out-of-sample.

    Drops bodies with NaN / non-finite PSR in any of the three protocols (e.g.
    MuJoCo simulator divergences). The pre-registration explicitly calls for
    this: "If a body is NaN due to training divergence, we flag it and drop
    that body's contributions." All statistics are computed on the finite
    subset; the dropped body names are recorded in the output for audit.
    """
    # Per-model filtering: M2 only needs str+rf; M1 and M3 also need epi.
    # This matches the pre-reg directive to drop only divergent cells, not
    # whole bodies when the needed protocol has a finite seed-average.
    epi_arr = np.array([psr_epi.get(b, np.nan) for b in bodies])
    str_arr = np.array([psr_strict.get(b, np.nan) for b in bodies])
    rf_arr = np.array([psr_rf.get(b, np.nan) for b in bodies])
    needs_epi = {"M1": True, "M2": False, "M3": True}

    out: dict = {
        "n_total_bodies": len(bodies),
        "models": {},
    }
    for mname in ("M1", "M2", "M3"):
        if needs_epi[mname]:
            keep = np.isfinite(epi_arr) & np.isfinite(str_arr) & np.isfinite(rf_arr)
        else:
            keep = np.isfinite(str_arr) & np.isfinite(rf_arr)
        kept = [b for b, k in zip(bodies, keep, strict=True) if k]
        dropped = [b for b, k in zip(bodies, keep, strict=True) if not k]
        if not kept:
            out["models"][mname] = {
                "n_kept": 0,
                "bodies_dropped": dropped,
                "note": f"{mname} has no bodies with required finite PSR",
            }
            continue
        y = str_arr[keep] - rf_arr[keep]
        y_pred_full = predict(coefs, bodies, features, psr_epi, mname)
        y_pred = y_pred_full[keep]
        rho_pt, lo, hi = bootstrap_spearman(y, y_pred)
        pearson_r, pearson_p = pearsonr(y, y_pred)
        X_cal = np.column_stack([np.ones_like(y_pred), y_pred])
        beta_cal, *_ = np.linalg.lstsq(X_cal, y, rcond=None)
        out["models"][mname] = {
            "n_kept": len(kept),
            "bodies_dropped": dropped,
            "y_observed": y.tolist(),
            "y_predicted": y_pred.tolist(),
            "spearman_rho": rho_pt,
            "spearman_ci": [lo, hi],
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "calibration_intercept": float(beta_cal[0]),
            "calibration_slope": float(beta_cal[1]),
        }
    return out


def family_of(name: str) -> str | None:
    if name.startswith("floor"):
        return "floor"
    if name.startswith("mvt"):
        return "mvt"
    if name.startswith("vt"):
        return "vt"
    return None


def leave_one_family_out(
    bodies: list[str],
    features: dict[str, dict[str, float]],
    psr_epi: dict[str, float],
    psr_strict: dict[str, float],
    psr_rf: dict[str, float],
) -> dict:
    """Leave-one-family-out cross-validation for M1/M2/M3.

    Tests the DERL family confound directly: for each held-out family,
    fit each model on the other two families, predict the held-out family.
    If M2 still correlates with the observed Delta_str_rf on the unseen family,
    structural features generalise across DERL evolutionary lineages.
    """
    fams = ["floor", "vt", "mvt"]
    out: dict = {"folds": {}, "description": "Fit on two families, test on the third."}
    for held in fams:
        train_bodies = [b for b in bodies if family_of(b) != held]
        test_bodies = [b for b in bodies if family_of(b) == held]
        fit = fit_all_models(train_bodies, features, psr_epi, psr_strict, psr_rf)
        y_test = np.array([psr_strict[b] - psr_rf[b] for b in test_bodies])
        fold: dict = {
            "held_family": held,
            "n_train": len(train_bodies),
            "n_test": len(test_bodies),
            "models": {},
        }
        for mname in ("M1", "M2", "M3"):
            y_pred = predict(fit, test_bodies, features, psr_epi, mname)
            rho_pt, lo, hi = bootstrap_spearman(y_test, y_pred)
            pearson_r, pearson_p = pearsonr(y_test, y_pred)
            X_cal = np.column_stack([np.ones_like(y_pred), y_pred])
            beta_cal, *_ = np.linalg.lstsq(X_cal, y_test, rcond=None)
            fold["models"][mname] = {
                "spearman_rho": rho_pt,
                "spearman_ci": [lo, hi],
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "calibration_slope": float(beta_cal[1]),
            }
        out["folds"][held] = fold
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zero-shot-features",
        type=Path,
        default=None,
        help="features JSON for held-out body set (same schema as morph_features.json)",
    )
    parser.add_argument(
        "--zero-shot-results",
        type=Path,
        default=None,
        help="directory containing results for the held-out set",
    )
    parser.add_argument(
        "--zero-shot-tau-labels",
        default="tau0p00,tau0p20,tau0p50",
        help="comma-separated tau labels (order: rf,epi,strict) for the held-out results",
    )
    args = parser.parse_args()

    # --- UNIMAL fit ---
    feats = json.loads((RESULTS / "morph_features.json").read_text())
    psr_epi = load_corpus_psr("0p20")
    psr_strict = load_corpus_psr("0p50")
    psr_rf = load_corpus_psr("0p00")
    bodies = sorted(set(feats) & set(psr_epi) & set(psr_strict) & set(psr_rf))
    print(f"UNIMAL fit: n={len(bodies)}")

    coefs = fit_all_models(bodies, feats, psr_epi, psr_strict, psr_rf)

    # Console summary
    print("\n=== In-sample diagnostics (100-body UNIMAL) ===")
    print(
        f"{'model':<5} {'R^2':>7} {'adj R^2':>9} {'AIC':>9} "
        f"{'rho':>7} {'rho 95% CI':>16}"
    )
    for mname in ("M1", "M2", "M3"):
        m = coefs["models"][mname]["in_sample"]
        ci = f"[{m['spearman_ci'][0]:.2f}, {m['spearman_ci'][1]:.2f}]"
        print(
            f"{mname:<5} {m['r2']:>7.3f} {m['adj_r2']:>9.3f} {m['aic']:>9.1f} "
            f"{m['spearman_rho']:>7.3f} {ci:>16}"
        )

    print("\n=== Standardised coefficients ===")
    for mname in ("M1", "M2", "M3"):
        md = coefs["models"][mname]
        print(f"{mname}:")
        for c, b in zip(md["columns"], md["beta"], strict=False):
            print(f"    {c:<22}  {b:+.4f}")

    OUT_COEFS.parent.mkdir(exist_ok=True)
    with OUT_COEFS.open("w") as f:
        json.dump(coefs, f, indent=2)
    print(f"\nWrote {OUT_COEFS}")

    # --- Leave-one-family-out CV ---
    lofo = leave_one_family_out(bodies, feats, psr_epi, psr_strict, psr_rf)
    print("\n=== Leave-one-family-out CV (fit on 2 families, test on held-out) ===")
    print(f"{'held':<8} {'n_test':>7} {'model':<5} {'rho':>7} {'rho 95% CI':>16}")
    for held in ("floor", "vt", "mvt"):
        f = lofo["folds"][held]
        for mname in ("M1", "M2", "M3"):
            m = f["models"][mname]
            ci = f"[{m['spearman_ci'][0]:.2f}, {m['spearman_ci'][1]:.2f}]"
            print(
                f"{held:<8} {f['n_test']:>7} {mname:<5} "
                f"{m['spearman_rho']:>7.3f} {ci:>16}"
            )

    report = {"unimal_in_sample": coefs, "leave_one_family_out": lofo}

    # --- Zero-shot (optional) ---
    if args.zero_shot_features and args.zero_shot_results:
        zs_feats = json.loads(args.zero_shot_features.read_text())
        labels = tuple(args.zero_shot_tau_labels.split(","))
        assert len(labels) == 3, "expected three tau labels: rf,epi,strict"
        rf_zs, epi_zs, str_zs = load_arbitrary_psr(args.zero_shot_results, labels)
        # Pass ALL feature-matched bodies; per-model NaN filtering happens
        # inside zero_shot_eval. Pre-intersecting would undercount M2
        # (morphology-only) coverage for bodies missing epi seed-averages.
        zs_bodies = sorted(
            b for b in zs_feats if b in (set(rf_zs) | set(str_zs) | set(epi_zs))
        )
        if not zs_bodies:
            print("\nZero-shot: no bodies common to features + results. Skipping.")
        else:
            print(f"\n=== Zero-shot evaluation (candidates n={len(zs_bodies)}) ===")
            zs = zero_shot_eval(coefs, zs_bodies, zs_feats, epi_zs, str_zs, rf_zs)
            print(
                f"{'model':<5} {'n_kept':>7} {'rho':>7} {'rho 95% CI':>16} "
                f"{'slope':>7} {'pearson r':>10}"
            )
            for mname in ("M1", "M2", "M3"):
                m = zs["models"][mname]
                if m.get("n_kept", 0) == 0:
                    print(f"{mname:<5} {'0':>7} (no bodies with required finite PSR)")
                    continue
                ci = f"[{m['spearman_ci'][0]:.2f}, {m['spearman_ci'][1]:.2f}]"
                print(
                    f"{mname:<5} {m['n_kept']:>7} {m['spearman_rho']:>7.3f} {ci:>16} "
                    f"{m['calibration_slope']:>7.3f} {m['pearson_r']:>10.3f}"
                )
            report["zero_shot"] = {
                "bodies_candidates": zs_bodies,
                "tau_labels": list(labels),
                **zs,
            }

    with OUT_REPORT.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
