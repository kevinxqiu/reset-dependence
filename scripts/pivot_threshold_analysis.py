#!/usr/bin/env python3
"""Phase 1: Protocol-Conditioned Morphology Response Analysis.

Fits a hierarchical model PSR(m, tau) = alpha(m) + beta(m) * tau to the
protocol-morph matrix and compares against null models and baselines.

Sub-analyses:
  1a. Hierarchical response model (mixed-effects)
  1b. Null model comparisons (permuted, global slope, synthetic)
  1c. Interpolation baselines (leave-one-protocol-out)
  1d. Pairwise correlation structure with bootstrap CIs
  1e. Probe and feature connections with alpha(m) and beta(m)
  1f. Selection utility on full 100-morph pool

Develop against existing 3-protocol corpus (tau=0.00, 0.20, 0.50).
Rerun on full 5-threshold data when Phase 0 completes.

Usage:
  python scripts/pivot_threshold_analysis.py                    # 3-threshold dev
  python scripts/pivot_threshold_analysis.py --n_thresholds 5   # after Phase 0
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
THRESHOLDS_3 = [0.00, 0.20, 0.50]
THRESHOLDS_5 = [0.00, 0.10, 0.20, 0.35, 0.50]
EVAL_PROTOCOL = "eval_tau020"
FAMILY_MAP = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}
N_BOOTSTRAP = 5000
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus(thresholds, seeds, result_dir, eval_protocol=EVAL_PROTOCOL):
    """Load the protocol-morph matrix into a tidy DataFrame."""
    rows = []
    for tau in thresholds:
        tau_str = f"{tau:.2f}".replace(".", "p")
        for seed in seeds:
            path = os.path.join(
                result_dir, f"mjx_ppo_{tau_str}_seed{seed}", "results.json"
            )
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for morph_name, result in data.items():
                fam_key = morph_name.split("-")[0]
                rows.append(
                    {
                        "morph": morph_name,
                        "family": FAMILY_MAP.get(fam_key, fam_key),
                        "tau": tau,
                        "seed": seed,
                        "psr": result[eval_protocol]["per_step_reward"],
                        "fitness": result[eval_protocol]["fitness"],
                        "fall_rate": result[eval_protocol]["fall_rate"],
                        "distance": result[eval_protocol]["distance"],
                        "mean_ep_len": result[eval_protocol]["mean_ep_len"],
                    }
                )
    df = pd.DataFrame(rows)
    print(
        f"Loaded {len(df)} rows: {df['morph'].nunique()} morphs, "
        f"{df['tau'].nunique()} thresholds, {df['seed'].nunique()} seeds, "
        f"eval={eval_protocol}"
    )
    return df


def fit_params_silent(df, exclude_tau=None, random_slope=True):
    """Fit MixedLM and return per-morph alpha/beta. Silent (no printing).

    Args:
        df: tidy DataFrame with columns morph, tau, psr
        exclude_tau: if not None, exclude rows where tau == exclude_tau (for nested holdout)
        random_slope: if True, use (1+tau|morph). If False, use (1|morph) only (global slope null).

    Returns:
        DataFrame with morph, family, alpha, beta. If random_slope=False, beta is the global slope.
    """
    import statsmodels.formula.api as smf

    fit_df = df if exclude_tau is None else df[df["tau"] != exclude_tau]
    re_formula = "~tau" if random_slope else None
    model = smf.mixedlm(
        "psr ~ tau", data=fit_df, groups=fit_df["morph"], re_formula=re_formula
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True)

    re = result.random_effects
    global_beta = float(result.fe_params["tau"])
    rows = []
    for morph in sorted(re.keys()):
        blup = re[morph]
        alpha = float(result.fe_params["Intercept"] + blup["Group"])
        if random_slope:
            beta = float(result.fe_params["tau"] + blup.get("tau", 0.0))
        else:
            beta = global_beta
        fam_key = morph.split("-")[0]
        rows.append(
            {
                "morph": morph,
                "family": FAMILY_MAP.get(fam_key, fam_key),
                "alpha": alpha,
                "beta": beta,
            }
        )
    return pd.DataFrame(rows)


def load_probes(path):
    """Load geometric proxy data."""
    with open(path) as f:
        data = json.load(f)
    probe_keys = [
        "P1_passive_final",
        "P2_passive_max",
        "P3_random_final",
        "P4_random_max",
        "P5_settle_success",
    ]
    rows = []
    for morph, vals in data["per_morph_proxies"].items():
        row = {"morph": morph}
        for k in probe_keys:
            row[k] = vals[k]
        rows.append(row)
    return pd.DataFrame(rows)


def load_features(path):
    """Load structural morphology features."""
    with open(path) as f:
        data = json.load(f)
    feature_keys = [
        "n_bodies",
        "n_joints",
        "n_motors",
        "n_geoms",
        "init_height",
        "tree_depth",
        "total_volume",
        "n_first_level_limbs",
    ]
    rows = []
    for morph, vals in data.items():
        row = {"morph": morph}
        for k in feature_keys:
            row[k] = vals[k]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1a. Hierarchical response model
# ---------------------------------------------------------------------------
def fit_hierarchical_model(df):
    """Fit PSR ~ tau with random intercept + slope per morph.

    Uses statsmodels MixedLM. Falls back to per-morph OLS if statsmodels
    is unavailable.

    Returns DataFrame with columns: morph, family, alpha, beta
    """
    print("\n" + "=" * 70)
    print("1a. HIERARCHICAL RESPONSE MODEL")
    print("=" * 70)

    try:
        import statsmodels.formula.api as smf

        # Fit mixed-effects model: PSR ~ tau + (1 + tau | morph)
        model = smf.mixedlm("psr ~ tau", data=df, groups=df["morph"], re_formula="~tau")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(reml=True)

        print("\nFixed effects:")
        print(f"  Intercept (mu_alpha): {result.fe_params['Intercept']:.4f}")
        print(f"  Slope (mu_beta):      {result.fe_params['tau']:.4f}")
        print("\nRandom effects variance:")
        print(f"  Var(alpha): {result.cov_re.iloc[0, 0]:.6f}")
        if result.cov_re.shape[0] > 1:
            print(f"  Var(beta):  {result.cov_re.iloc[1, 1]:.6f}")
            print(f"  Cov(a,b):   {result.cov_re.iloc[0, 1]:.6f}")

        # Extract per-morph BLUPs
        re = result.random_effects
        morph_params = []
        for morph in sorted(re.keys()):
            blup = re[morph]
            alpha = result.fe_params["Intercept"] + blup["Group"]
            beta = result.fe_params["tau"] + blup.get("tau", 0.0)
            fam_key = morph.split("-")[0]
            morph_params.append(
                {
                    "morph": morph,
                    "family": FAMILY_MAP.get(fam_key, fam_key),
                    "alpha": alpha,
                    "beta": beta,
                }
            )
        params_df = pd.DataFrame(morph_params)
        method = "mixed-effects"

    except (ImportError, Exception) as e:
        print(f"  Mixed-effects failed ({e}), falling back to per-morph OLS")
        params_df = _fit_per_morph_ols(df)
        method = "per-morph-OLS"

    # Summary statistics
    print(f"\nMethod: {method}")
    print(f"Per-morph parameter distributions (n={len(params_df)}):")
    print(
        f"  alpha: mean={params_df['alpha'].mean():.4f}, "
        f"std={params_df['alpha'].std():.4f}, "
        f"range=[{params_df['alpha'].min():.4f}, {params_df['alpha'].max():.4f}]"
    )
    print(
        f"  beta:  mean={params_df['beta'].mean():.4f}, "
        f"std={params_df['beta'].std():.4f}, "
        f"range=[{params_df['beta'].min():.4f}, {params_df['beta'].max():.4f}]"
    )

    # Family breakdown
    print("\nFamily means:")
    for fam in ["Floor", "VT", "MVT"]:
        sub = params_df[params_df["family"] == fam]
        if len(sub) == 0:
            continue
        print(
            f"  {fam:5s} (n={len(sub):2d}): "
            f"alpha={sub['alpha'].mean():.4f} +/- {sub['alpha'].std():.4f}, "
            f"beta={sub['beta'].mean():.4f} +/- {sub['beta'].std():.4f}"
        )

    # Within- vs between-family variance
    grand_alpha = params_df["alpha"].var()
    grand_beta = params_df["beta"].var()
    between_alpha = params_df.groupby("family")["alpha"].mean().var()
    between_beta = params_df.groupby("family")["beta"].mean().var()
    print("\nVariance decomposition:")
    print(
        f"  alpha: between-family={between_alpha:.6f}, "
        f"total={grand_alpha:.6f}, "
        f"ratio={between_alpha / grand_alpha:.3f}"
    )
    print(
        f"  beta:  between-family={between_beta:.6f}, "
        f"total={grand_beta:.6f}, "
        f"ratio={between_beta / grand_beta:.3f}"
    )

    return params_df


def _fit_per_morph_ols(df):
    """Fallback: fit independent OLS per morph."""
    rows = []
    for morph, grp in df.groupby("morph"):
        tau = grp["tau"].values.reshape(-1, 1)
        psr = grp["psr"].values
        reg = LinearRegression().fit(tau, psr)
        fam_key = morph.split("-")[0]
        rows.append(
            {
                "morph": morph,
                "family": FAMILY_MAP.get(fam_key, fam_key),
                "alpha": reg.intercept_,
                "beta": reg.coef_[0],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1b. Null model comparisons
# ---------------------------------------------------------------------------
def null_model_comparisons(df, params_df, n_perm=1000):
    """Test whether body-specific slopes are real vs. artifacts.

    Null 1: Permute morph labels within each (tau, seed) group.
            (Note: this destroys both intercepts and slopes, so it tests "any
            morph-specific structure" not "slope-specific structure" alone.)
    Null 2: Properly fit global-slope mixed-effects model (random intercept only).
            Compare in-sample fit of body-specific vs global-slope model.
    """
    print("\n" + "=" * 70)
    print("1b. NULL MODEL COMPARISONS")
    print("=" * 70)

    rng = np.random.RandomState(RNG_SEED)
    observed_beta_var = params_df["beta"].var()

    # --- Null 1: Permuted morph labels ---
    print(f"\nNull 1: Permuted morph labels ({n_perm} permutations)")
    print("  (Note: destroys all morph structure, conservative null)")
    print(f"  Observed Var(beta) = {observed_beta_var:.6f}")

    null_vars = []
    for _ in range(n_perm):
        df_perm = df.copy()
        for (tau, seed), grp in df_perm.groupby(["tau", "seed"]):
            perm_idx = rng.permutation(grp.index)
            df_perm.loc[grp.index, "morph"] = df_perm.loc[perm_idx, "morph"].values
        perm_params = _fit_per_morph_ols(df_perm)
        null_vars.append(perm_params["beta"].var())

    null_vars = np.array(null_vars)
    n_at_or_above = np.sum(null_vars >= observed_beta_var)
    # Use (b+1)/(n+1) per Phipson & Smyth (2010) - never exactly 0
    p_value = (n_at_or_above + 1) / (n_perm + 1)
    print(
        f"  Null Var(beta):  mean={null_vars.mean():.6f}, "
        f"95th={np.percentile(null_vars, 95):.6f}"
    )
    print(f"  p-value (one-sided, (b+1)/(n+1)): {p_value:.4f}")
    print(f"  Observed / Null mean: {observed_beta_var / null_vars.mean():.2f}x")

    # --- Null 2: Properly fit global-slope mixed model ---
    print("\nNull 2: Global-slope mixed model (random intercept only)")
    print("  PSR ~ tau + (1|morph) vs. body-specific PSR ~ tau + (1+tau|morph)")
    global_params = fit_params_silent(df, random_slope=False)
    global_alphas = global_params.set_index("morph")["alpha"]
    global_beta = global_params["beta"].iloc[0]
    body_alphas = params_df.set_index("morph")["alpha"]
    body_betas = params_df.set_index("morph")["beta"]

    seed_avg = df.groupby(["morph", "tau"])["psr"].mean().reset_index()

    body_errors = []
    global_errors = []
    for _, row in seed_avg.iterrows():
        m, tau, psr = row["morph"], row["tau"], row["psr"]
        body_pred = body_alphas[m] + body_betas[m] * tau
        global_pred = global_alphas[m] + global_beta * tau
        body_errors.append((psr - body_pred) ** 2)
        global_errors.append((psr - global_pred) ** 2)

    mse_specific = float(np.mean(body_errors))
    mse_global = float(np.mean(global_errors))
    ss_total = float(np.sum((seed_avg["psr"] - seed_avg["psr"].mean()) ** 2))
    r2_specific = 1 - np.sum(body_errors) / ss_total
    r2_global = 1 - np.sum(global_errors) / ss_total
    print(f"  MSE body-specific: {mse_specific:.6f}  (R^2={r2_specific:.4f})")
    print(f"  MSE global-slope:  {mse_global:.6f}  (R^2={r2_global:.4f})")
    print(
        f"  Body-specific improvement: {(mse_global - mse_specific) / mse_global * 100:.1f}%"
    )
    print(f"  Global beta estimate: {global_beta:.4f}")

    return {
        "observed_beta_var": float(observed_beta_var),
        "null_beta_var_mean": float(null_vars.mean()),
        "null_beta_var_95th": float(np.percentile(null_vars, 95)),
        "null1_p_value": float(p_value),
        "null1_ratio": float(observed_beta_var / null_vars.mean()),
        "null2_mse_specific": mse_specific,
        "null2_mse_global": mse_global,
        "null2_r2_specific": float(r2_specific),
        "null2_r2_global": float(r2_global),
        "global_beta": float(global_beta),
    }


# ---------------------------------------------------------------------------
# 1c. Interpolation baselines (leave-one-protocol-out)
# ---------------------------------------------------------------------------
def interpolation_baselines(df, params_df):
    """Leave-one-protocol-out prediction comparison.

    Baseline 1: tau-only regression (no body info)
    Baseline 2: body intercept + global slope
    Baseline 3: body intercept + body-specific slope (full model)
    """
    print("\n" + "=" * 70)
    print("1c. INTERPOLATION BASELINES (leave-one-protocol-out)")
    print("=" * 70)

    seed_avg = df.groupby(["morph", "tau"])["psr"].mean().reset_index()
    thresholds = sorted(seed_avg["tau"].unique())
    morphs = sorted(seed_avg["morph"].unique())

    results_per_fold = []

    for held_out_tau in thresholds:
        train = seed_avg[seed_avg["tau"] != held_out_tau]
        test = seed_avg[seed_avg["tau"] == held_out_tau]

        # Baseline 1: tau-only (no body info)
        reg1 = LinearRegression().fit(
            train["tau"].values.reshape(-1, 1), train["psr"].values
        )
        pred1 = reg1.predict(test["tau"].values.reshape(-1, 1))

        # Baseline 2: body intercept + global slope
        # Fit per-morph intercept and one global slope from training data
        morph_intercepts = {}
        all_residuals_tau = []
        all_residuals_psr = []
        for morph in morphs:
            mt = train[train["morph"] == morph]
            if len(mt) == 0:
                continue
            morph_intercepts[morph] = mt["psr"].mean()
            for _, row in mt.iterrows():
                all_residuals_tau.append(row["tau"])
                all_residuals_psr.append(row["psr"] - morph_intercepts[morph])

        # Fit global slope on de-meaned data
        if len(all_residuals_tau) > 1:
            global_reg = LinearRegression().fit(
                np.array(all_residuals_tau).reshape(-1, 1),
                np.array(all_residuals_psr),
            )
            g_slope = global_reg.coef_[0]
        else:
            g_slope = 0.0

        # For held-out morphs, recalculate intercept from train data
        pred2 = []
        for _, row in test.iterrows():
            m_int = morph_intercepts.get(row["morph"], train["psr"].mean())
            # Adjust intercept: morph mean was computed at train taus, predict at test tau
            train_tau_mean = train[train["morph"] == row["morph"]]["tau"].mean()
            alpha_adj = m_int - g_slope * train_tau_mean
            pred2.append(alpha_adj + g_slope * row["tau"])
        pred2 = np.array(pred2)

        # Baseline 3: body-specific intercept + slope (from per-morph OLS on train)
        pred3 = []
        for _, row in test.iterrows():
            mt = train[train["morph"] == row["morph"]]
            if len(mt) >= 2:
                reg_m = LinearRegression().fit(
                    mt["tau"].values.reshape(-1, 1), mt["psr"].values
                )
                pred3.append(reg_m.predict([[row["tau"]]])[0])
            else:
                pred3.append(mt["psr"].mean() if len(mt) > 0 else train["psr"].mean())
        pred3 = np.array(pred3)

        actual = test["psr"].values
        ss_tot = np.sum((actual - actual.mean()) ** 2)

        def r2(pred):
            ss_res = np.sum((actual - pred) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fold_result = {
            "held_out_tau": held_out_tau,
            "r2_tau_only": r2(pred1),
            "r2_global_slope": r2(pred2),
            "r2_body_specific": r2(pred3),
            "rmse_tau_only": np.sqrt(np.mean((actual - pred1) ** 2)),
            "rmse_global_slope": np.sqrt(np.mean((actual - pred2) ** 2)),
            "rmse_body_specific": np.sqrt(np.mean((actual - pred3) ** 2)),
        }
        results_per_fold.append(fold_result)
        print(f"\n  Held-out tau={held_out_tau:.2f}:")
        print(
            f"    Tau-only:      R^2={fold_result['r2_tau_only']:.4f}  "
            f"RMSE={fold_result['rmse_tau_only']:.4f}"
        )
        print(
            f"    Global slope:  R^2={fold_result['r2_global_slope']:.4f}  "
            f"RMSE={fold_result['rmse_global_slope']:.4f}"
        )
        print(
            f"    Body-specific: R^2={fold_result['r2_body_specific']:.4f}  "
            f"RMSE={fold_result['rmse_body_specific']:.4f}"
        )

    # Average across folds
    avg = {
        k: np.mean([f[k] for f in results_per_fold])
        for k in ["r2_tau_only", "r2_global_slope", "r2_body_specific"]
    }
    print("\n  Mean across folds:")
    print(f"    Tau-only:      R^2={avg['r2_tau_only']:.4f}")
    print(f"    Global slope:  R^2={avg['r2_global_slope']:.4f}")
    print(f"    Body-specific: R^2={avg['r2_body_specific']:.4f}")

    verdict = "PASS" if avg["r2_body_specific"] > avg["r2_global_slope"] else "FAIL"
    print(f"\n  Body-specific beats global slope: {verdict}")

    return {"folds": results_per_fold, "averages": avg, "verdict": verdict}


# ---------------------------------------------------------------------------
# 1d. Pairwise correlation structure
# ---------------------------------------------------------------------------
def pairwise_correlations(df, n_bootstrap=N_BOOTSTRAP):
    """Spearman rank correlations between all protocol pairs with bootstrap CIs."""
    print("\n" + "=" * 70)
    print("1d. PAIRWISE CORRELATION STRUCTURE")
    print("=" * 70)

    seed_avg = df.groupby(["morph", "tau"])["psr"].mean().reset_index()
    thresholds = sorted(seed_avg["tau"].unique())
    morphs = sorted(seed_avg["morph"].unique())
    n_morphs = len(morphs)

    # Build morph x tau matrix
    psr_matrix = pd.pivot_table(
        seed_avg, values="psr", index="morph", columns="tau"
    ).loc[morphs]

    rng = np.random.RandomState(RNG_SEED)

    results = []
    print(f"\n  Spearman rho (seed-averaged PSR, n={n_morphs} morphs):\n")
    header = "  " + " " * 12 + "".join(f"tau={t:.2f}   " for t in thresholds)
    print(header)

    for i, t1 in enumerate(thresholds):
        row_str = f"  tau={t1:.2f}  "
        for j, t2 in enumerate(thresholds):
            x = psr_matrix[t1].values
            y = psr_matrix[t2].values
            rho, pval = stats.spearmanr(x, y)

            if i < j:
                # Bootstrap CI
                boot_rhos = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, n_morphs, n_morphs)
                    r, _ = stats.spearmanr(x[idx], y[idx])
                    boot_rhos.append(r)
                ci_lo = np.percentile(boot_rhos, 2.5)
                ci_hi = np.percentile(boot_rhos, 97.5)

                results.append(
                    {
                        "tau_1": t1,
                        "tau_2": t2,
                        "rho": float(rho),
                        "p_value": float(pval),
                        "ci_lo": float(ci_lo),
                        "ci_hi": float(ci_hi),
                        "tau_distance": abs(t2 - t1),
                    }
                )
                row_str += f"{rho:.3f}      "
            elif i == j:
                row_str += "1.000      "
            else:
                row_str += "  -        "
        print(row_str)

    # Test monotone decrease with threshold distance
    if len(results) >= 3:
        distances = [r["tau_distance"] for r in results]
        rhos = [r["rho"] for r in results]
        dist_rho, dist_p = stats.spearmanr(distances, rhos)
        monotone = dist_rho < 0
        print(
            f"\n  Correlation vs. threshold distance: rho={dist_rho:.3f}, p={dist_p:.4f}"
        )
        print(f"  Monotone decrease: {'YES' if monotone else 'NO'}")
    else:
        monotone = None

    # Within-protocol seed reliability (ICC proxy)
    print("\n  Within-protocol seed reliability:")
    for tau in thresholds:
        tau_data = df[df["tau"] == tau].pivot_table(
            values="psr", index="morph", columns="seed"
        )
        if tau_data.shape[1] >= 2:
            # Average pairwise Spearman across seeds
            seed_cols = tau_data.columns
            seed_rhos = []
            for si in range(len(seed_cols)):
                for sj in range(si + 1, len(seed_cols)):
                    r, _ = stats.spearmanr(
                        tau_data[seed_cols[si]], tau_data[seed_cols[sj]]
                    )
                    seed_rhos.append(r)
            mean_seed_rho = np.mean(seed_rhos)
            print(
                f"    tau={tau:.2f}: mean seed-to-seed rho={mean_seed_rho:.3f} "
                f"(range [{min(seed_rhos):.3f}, {max(seed_rhos):.3f}])"
            )

    return {
        "pairs": results,
        "monotone_decrease": bool(monotone) if monotone is not None else None,
    }


# ---------------------------------------------------------------------------
# 1e. Probe and feature connections
# ---------------------------------------------------------------------------
def probe_feature_connections(params_df, probe_path, feature_path):
    """Correlate alpha(m) and beta(m) with probes and structural features."""
    print("\n" + "=" * 70)
    print("1e. PROBE AND FEATURE CONNECTIONS")
    print("=" * 70)

    probes_df = load_probes(probe_path)
    features_df = load_features(feature_path)

    # Merge
    merged = params_df.merge(probes_df, on="morph").merge(features_df, on="morph")
    print(f"\n  Merged dataset: {len(merged)} morphs")

    probe_cols = [c for c in probes_df.columns if c != "morph"]
    feat_cols = [c for c in features_df.columns if c != "morph"]

    results = {"alpha_correlations": {}, "beta_correlations": {}}

    for target_name, target_col in [("alpha", "alpha"), ("beta", "beta")]:
        print(f"\n  Correlations with {target_name}:")
        print(f"    {'Predictor':<25s} {'Spearman rho':>12s} {'p-value':>10s}")
        print(f"    {'-' * 47}")

        target = merged[target_col].values

        for cols, label in [(probe_cols, "probe"), (feat_cols, "feature")]:
            for col in cols:
                x = merged[col].values
                valid = ~(np.isnan(x) | np.isnan(target))
                if valid.sum() < 10:
                    continue
                rho, pval = stats.spearmanr(x[valid], target[valid])
                marker = (
                    " ***"
                    if pval < 0.001
                    else " **"
                    if pval < 0.01
                    else " *"
                    if pval < 0.05
                    else ""
                )
                print(f"    {col:<25s} {rho:>12.3f} {pval:>10.4f}{marker}")
                results[f"{target_name}_correlations"][col] = {
                    "rho": float(rho),
                    "p_value": float(pval),
                    "type": label,
                }

    # LOO R^2 for predicting alpha and beta from (probes, features, both)
    print("\n  LOO-morph R^2 for predicting alpha and beta:")
    for target_name in ["alpha", "beta"]:
        target = merged[target_name].values
        print(f"\n    {target_name}:")
        for predictor_set, pred_label in [
            (probe_cols, "probes"),
            (feat_cols, "features"),
            (probe_cols + feat_cols, "both"),
        ]:
            X = merged[predictor_set].values
            valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(target)
            X_v = X[valid]
            y_v = target[valid]

            if len(y_v) < 10:
                print(f"      {pred_label}: insufficient data")
                continue

            # LOO prediction
            preds = np.zeros(len(y_v))
            for i in range(len(y_v)):
                mask = np.ones(len(y_v), dtype=bool)
                mask[i] = False
                reg = LinearRegression().fit(X_v[mask], y_v[mask])
                preds[i] = reg.predict(X_v[i : i + 1])[0]

            ss_res = np.sum((y_v - preds) ** 2)
            ss_tot = np.sum((y_v - y_v.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            print(f"      {pred_label:<12s}: LOO R^2 = {r2:.4f} (n={len(y_v)})")
            results[f"{target_name}_loo_r2_{pred_label}"] = float(r2)

    return results


# ---------------------------------------------------------------------------
# 1f. Selection utility
# ---------------------------------------------------------------------------
def selection_utility(df, top_k_values=None):
    """Test whether body-specific slope info improves morphology selection.

    Uses NESTED HOLDOUT: for each deploy_tau, refits alpha/beta WITHOUT deploy_tau
    so model-based strategies don't see the held-out target during scoring.

    Strategies compared (all scored on PSR at deploy_tau over the FULL 100-morph pool):
      - select_tau=X: rank by raw PSR at training tau X (X != deploy_tau valid)
      - alpha: rank by held-out alpha (intercept)
      - beta_robust: rank by |held-out beta| (smallest = most robust)
      - model_predict: rank by held-out alpha + deploy_tau * beta (natural score)

    Bootstrap: re-sample morphs (with replacement) and re-select each strategy
    on the resampled set using the held-out params (which are not refit per-resample
    for speed; this captures sampling uncertainty given honest params).
    """
    print("\n" + "=" * 70)
    print("1f. SELECTION UTILITY (with nested protocol holdout)")
    print("=" * 70)

    if top_k_values is None:
        top_k_values = [5, 10, 20]

    seed_avg = df.groupby(["morph", "tau"])["psr"].mean().reset_index()
    thresholds = sorted(seed_avg["tau"].unique())
    morphs = sorted(seed_avg["morph"].unique())

    psr_matrix = pd.pivot_table(
        seed_avg, values="psr", index="morph", columns="tau"
    ).loc[morphs]

    rng = np.random.RandomState(RNG_SEED)
    n_boot = N_BOOTSTRAP
    morph_arr = np.array(morphs)

    results = []

    for deploy_tau in thresholds:
        # NESTED HOLDOUT: refit alpha, beta WITHOUT deploy_tau in the fit data
        print(f"\n  Refitting params excluding tau={deploy_tau:.2f}...", flush=True)
        holdout_params = fit_params_silent(df, exclude_tau=deploy_tau).set_index(
            "morph"
        )
        holdout_alpha = holdout_params["alpha"]
        holdout_beta = holdout_params["beta"]
        deploy_psr = psr_matrix[deploy_tau]

        # Pre-compute strategy scores (vectorized for speed)
        # Lower = better for beta_robust; Higher = better for others
        strategy_scores = {}
        for select_tau in thresholds:
            strategy_scores[f"select_tau={select_tau:.2f}"] = (
                psr_matrix[select_tau],
                False,
            )
        strategy_scores["alpha"] = (holdout_alpha, False)
        strategy_scores["beta_robust"] = (holdout_beta.abs(), True)  # ascend=True
        strategy_scores["model_predict"] = (
            holdout_alpha + deploy_tau * holdout_beta,
            False,
        )

        for k in top_k_values:
            oracle_ranking = deploy_psr.sort_values(ascending=False)
            oracle_set = set(oracle_ranking.index[:k])
            oracle_mean = float(oracle_ranking.values[:k].mean())

            print(f"\n  Deploy at tau={deploy_tau:.2f}, top-{k}:")
            print(f"    Oracle mean PSR: {oracle_mean:.4f}")
            print(
                f"    {'Strategy':<22s} {'Mean PSR':>10s} {'%Oracle':>10s} "
                f"{'Overlap':>9s}  {'95% CI':>20s}"
            )

            # Pre-compute bootstrap samples ONCE per (deploy_tau, k)
            # All strategies use the SAME bootstrap samples for fair comparison
            boot_idx = rng.randint(0, len(morph_arr), size=(n_boot, len(morph_arr)))

            for strat_name, (score, ascending) in strategy_scores.items():
                # Point estimate
                ranking = score.sort_values(ascending=ascending)
                sel_set = set(ranking.index[:k])
                deploy_mean = float(deploy_psr.loc[list(sel_set)].mean())
                regret = deploy_mean / oracle_mean if oracle_mean > 0 else 0.0
                overlap = len(sel_set & oracle_set) / k

                # Bootstrap: resample morphs, re-select using SAME held-out params
                boot_regrets = np.empty(n_boot)
                for b in range(n_boot):
                    boot_morphs = morph_arr[boot_idx[b]]
                    # Bootstrap oracle (best-k achievable on resampled set)
                    b_deploy = deploy_psr.loc[boot_morphs].values
                    b_oracle_mean = np.mean(np.sort(b_deploy)[-k:])
                    # Bootstrap strategy: re-select using held-out params on resampled morphs
                    b_score = score.loc[boot_morphs].values
                    if ascending:
                        sel_idx = np.argsort(b_score)[:k]
                    else:
                        sel_idx = np.argsort(-b_score)[:k]
                    b_sel_mean = np.mean(b_deploy[sel_idx])
                    boot_regrets[b] = (
                        b_sel_mean / b_oracle_mean if b_oracle_mean > 0 else 0.0
                    )

                ci_lo = float(np.percentile(boot_regrets, 2.5))
                ci_hi = float(np.percentile(boot_regrets, 97.5))

                print(
                    f"    {strat_name:<22s} {deploy_mean:>10.4f} "
                    f"{regret * 100:>9.1f}% {overlap:>9.2f}  "
                    f"[{ci_lo * 100:>5.1f}%, {ci_hi * 100:>5.1f}%]"
                )

                results.append(
                    {
                        "deploy_tau": float(deploy_tau),
                        "top_k": int(k),
                        "strategy": strat_name,
                        "deploy_mean_psr": deploy_mean,
                        "oracle_mean_psr": oracle_mean,
                        "regret_ratio": float(regret),
                        "overlap": float(overlap),
                        "regret_ci_lo": ci_lo,
                        "regret_ci_hi": ci_hi,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_thresholds",
        type=int,
        default=3,
        choices=[3, 5],
        help="Number of thresholds (3=dev, 5=full)",
    )
    parser.add_argument(
        "--result_dir",
        default="results",
        help="Base results directory",
    )
    parser.add_argument(
        "--probe_path",
        default="results/pivot/a2_geom_proxy.json",
    )
    parser.add_argument(
        "--feature_path",
        default="results/morph_features.json",
    )
    parser.add_argument(
        "--out_dir",
        default="results/pivot",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=1000,
        help="Permutations for null test",
    )
    parser.add_argument(
        "--eval",
        default="eval_tau020",
        choices=["eval_tau020", "eval_tau050", "eval_neutral"],
        help="Which evaluation protocol to use as the response variable",
    )
    args = parser.parse_args()

    thresholds = THRESHOLDS_3 if args.n_thresholds == 3 else THRESHOLDS_5
    print("Protocol-Conditioned Morphology Response Analysis")
    print(f"  Thresholds: {thresholds}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Eval protocol: {args.eval}")
    print()

    # Load data
    df = load_corpus(thresholds, SEEDS, args.result_dir, eval_protocol=args.eval)
    if df.empty:
        print("ERROR: No data loaded. Check result paths.")
        sys.exit(1)

    # 1a. Hierarchical model
    params_df = fit_hierarchical_model(df)

    # 1b. Null model comparisons
    null_results = null_model_comparisons(df, params_df, n_perm=args.n_perm)

    # 1c. Interpolation baselines
    interp_results = interpolation_baselines(df, params_df)

    # 1d. Pairwise correlations
    corr_results = pairwise_correlations(df)

    # 1e. Probe and feature connections
    probe_results = probe_feature_connections(
        params_df, args.probe_path, args.feature_path
    )

    # 1f. Selection utility (with nested holdout, no params_df argument)
    selection_results = selection_utility(df)

    # Save all results
    os.makedirs(args.out_dir, exist_ok=True)
    output = {
        "config": {
            "thresholds": thresholds,
            "seeds": SEEDS,
            "eval_protocol": args.eval,
            "n_morphs": int(df["morph"].nunique()),
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": args.n_perm,
        },
        "morph_params": params_df.to_dict(orient="records"),
        "null_comparisons": null_results,
        "interpolation_baselines": interp_results,
        "pairwise_correlations": corr_results,
        "probe_feature_connections": probe_results,
        "selection_utility": selection_results,
    }

    eval_tag = args.eval.replace("eval_", "")
    out_path = os.path.join(args.out_dir, f"threshold_model_{eval_tag}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
