#!/usr/bin/env python3
"""Recompute the reviewed version's corpus statistics.

Loads the 900-run corpus (100 morphologies x 3 protocols x 3 seeds) from
``results/mjx_ppo_{0p00,0p20,0p50}_seed{42,43,44}/results.json`` and the
structural features from ``results/morph_features.json``, then computes and
prints every statistic cited in the paper, alongside the paper's claimed
value for direct comparison.

Writes a machine-readable artifact to
``records/paper_stats_reproduction.json`` so that reviewers can diff
against the paper text without re-running.

This script is the single reproduction entry-point. Individual
special-purpose scripts (``compute_regret_common_eval.py`` for the bootstrap
regret CIs) remain available as focused tools.

Usage:
    python scripts/reproduce_paper_stats.py
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

REPO = Path(__file__).resolve().parents[1]

# Morphology is nested in family, so in the blocked models the family main effect is
# aliased with the morphology fixed effects and the design is rank-deficient by
# construction; the tests of interest are unaffected. Recent statsmodels versions warn
# on every such fit, so that one expected warning is silenced here.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
RESULTS = REPO / "results"
FEATURES = RESULTS / "morph_features.json"
OUT_JSON = REPO / "records" / "paper_stats_reproduction.json"

SEEDS = (42, 43, 44)
TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}
EVAL_KEY = "eval_tau020"
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42


def family_of(name: str) -> str | None:
    if name.startswith("floor"):
        return "floor"
    if name.startswith("mvt"):
        return "mvt"
    if name.startswith("vt"):
        return "vt"
    return None


def load_per_seed(
    train_tau: str, eval_key: str = EVAL_KEY, metric: str = "per_step_reward"
):
    """Return {body: {seed: value}} for the given training tau / eval key."""
    per_seed: defaultdict[str, dict[int, float]] = defaultdict(dict)
    for s in SEEDS:
        path = RESULTS / f"mjx_ppo_{train_tau}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            v = rec.get(eval_key, {}).get(metric)
            if v is not None:
                per_seed[body][s] = float(v)
    return dict(per_seed)


def seed_avg(per_seed: dict[str, dict[int, float]]) -> dict[str, float]:
    return {b: float(np.mean(list(v.values()))) for b, v in per_seed.items()}


def section_header(name: str) -> None:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")


def main() -> None:
    out: dict[str, Any] = {
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "eval_key": EVAL_KEY,
    }

    # Load PSR (all three training protocols, all three seeds)
    psr = {p: load_per_seed(t) for p, t in TAUS.items()}
    common = sorted(set(psr["episodic"]) & set(psr["strict"]) & set(psr["reset_free"]))
    for b in common:
        assert len(psr["episodic"][b]) == 3, f"Missing seed for {b}"
    print(f"Loaded {len(common)} bodies with full coverage")

    family = {b: family_of(b) for b in common}
    psr_mean = {p: seed_avg(psr[p]) for p in psr}
    out["n_bodies"] = len(common)

    # ------------------------------------------------------------------
    # Section: Table 1 (family means + RF/Epi, Str/Epi ratios)
    # ------------------------------------------------------------------
    section_header("A. Table 1: family means and per-morph ratios")
    table1: dict[str, Any] = {}
    for fam in ("floor", "vt", "mvt"):
        bodies_f = [b for b in common if family[b] == fam]
        e = float(np.mean([psr_mean["episodic"][b] for b in bodies_f]))
        s = float(np.mean([psr_mean["strict"][b] for b in bodies_f]))
        r = float(np.mean([psr_mean["reset_free"][b] for b in bodies_f]))
        rfepi = float(
            np.mean(
                [psr_mean["reset_free"][b] / psr_mean["episodic"][b] for b in bodies_f]
            )
        )
        stepi = float(
            np.mean([psr_mean["strict"][b] / psr_mean["episodic"][b] for b in bodies_f])
        )
        table1[fam] = {
            "n": len(bodies_f),
            "epi": e,
            "strict": s,
            "rf": r,
            "rf_over_epi": rfepi,
            "str_over_epi": stepi,
        }
        print(
            f"  {fam.upper():5s} n={len(bodies_f):3d}  "
            f"Epi={e:.3f}  Str={s:.3f}  RF={r:.3f}  "
            f"RF/Epi={rfepi:.2f}  Str/Epi={stepi:.2f}"
        )
    out["table1_family_means"] = table1

    # ------------------------------------------------------------------
    # Section: one-way ANOVA on RF/Epi ratios (paper F=6.46, p=0.002)
    # ------------------------------------------------------------------
    section_header("B. One-way ANOVA on RF/Epi ratios by family")
    groups = []
    for fam in ("floor", "vt", "mvt"):
        bodies_f = [b for b in common if family[b] == fam]
        groups.append(
            np.array(
                [psr_mean["reset_free"][b] / psr_mean["episodic"][b] for b in bodies_f]
            )
        )
    F, p_a = stats.f_oneway(*groups)
    H, p_kw = stats.kruskal(*groups)
    print(f"  ANOVA F = {F:.3f}, p = {p_a:.5f}  (paper: F=6.46, p=0.002)")
    print(f"  Kruskal-Wallis H = {H:.3f}, p = {p_kw:.5f}  (paper: H=11.8, p=0.003)")
    out["oneway_rf_over_epi"] = {
        "anova_f": float(F),
        "anova_p": float(p_a),
        "kruskal_h": float(H),
        "kruskal_p": float(p_kw),
    }

    # Floor vs VT contrast: Cohen's d and eta^2
    floor_g = groups[0]
    vt_g = groups[1]
    pooled_sd = np.sqrt(
        ((len(floor_g) - 1) * floor_g.var(ddof=1) + (len(vt_g) - 1) * vt_g.var(ddof=1))
        / (len(floor_g) + len(vt_g) - 2)
    )
    cohens_d = (floor_g.mean() - vt_g.mean()) / pooled_sd
    total = np.concatenate(groups)
    grand_mean = total.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((total - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total
    print(f"  Floor-vs-VT Cohen's d = {cohens_d:.2f}  (paper: |d|=0.83)")
    print(f"  eta^2 = {eta2:.3f}  (paper: 0.12)")
    out["oneway_rf_over_epi"]["floor_vs_vt_cohens_d"] = float(cohens_d)
    out["oneway_rf_over_epi"]["eta_squared"] = float(eta2)

    # ------------------------------------------------------------------
    # Section: two-way ANOVA on raw 900-row PSR with morph blocking
    # (paper F(4,794)=12.54, p=6.8e-10, partial eta^2=0.059)
    # ------------------------------------------------------------------
    section_header("C. Two-way ANOVA (protocol * family + morph) on 900 obs")
    rows = []
    for proto in ("episodic", "strict", "reset_free"):
        for b in common:
            for s, v in psr[proto][b].items():
                rows.append(
                    {
                        "psr": v,
                        "protocol": proto,
                        "family": family[b],
                        "morph": b,
                        "seed": s,
                    }
                )
    df = pd.DataFrame(rows)
    assert len(df) == 900, f"Expected 900 rows, got {len(df)}"
    model = ols("psr ~ C(protocol)*C(family) + C(morph)", data=df).fit()
    tbl = anova_lm(model, typ=2)
    ss_int = tbl.loc["C(protocol):C(family)", "sum_sq"]
    ss_res = tbl.loc["Residual", "sum_sq"]
    p_eta2 = ss_int / (ss_int + ss_res)
    print(tbl.to_string())
    print(f"  partial eta^2 (interaction) = {p_eta2:.4f}  (paper: 0.059)")
    out["twoway_interaction_900"] = {
        "interaction_f": float(tbl.loc["C(protocol):C(family)", "F"]),
        "interaction_p": float(tbl.loc["C(protocol):C(family)", "PR(>F)"]),
        "protocol_f": float(tbl.loc["C(protocol)", "F"]),
        "protocol_p": float(tbl.loc["C(protocol)", "PR(>F)"]),
        "family_f": float(tbl.loc["C(family)", "F"]),
        "family_p": float(tbl.loc["C(family)", "PR(>F)"]),
        "partial_eta_squared": float(p_eta2),
        "df_residual": int(tbl.loc["Residual", "df"]),
    }

    # Dependence-aware inference for the same interaction term: refit the
    # blocked OLS model with cluster-robust SEs (cluster unit = morphology)
    # and do a joint Wald test on the four interaction coefficients. This
    # addresses the pseudoreplication concern that 900 rows are clustered
    # by 100 morphs. Also fit a GEE with morphology as the cluster and an
    # exchangeable working correlation as a second robust check.
    try:
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cr = ols(
                "psr ~ C(protocol)*C(family) + C(morph)",
                data=df,
            ).fit(cov_type="cluster", cov_kwds={"groups": df.morph})
            interaction_cols = [c for c in model_cr.params.index if ":" in c]
            idx_map = {n: i for i, n in enumerate(model_cr.params.index)}
            R = np.zeros((len(interaction_cols), len(model_cr.params)))
            for i, col in enumerate(interaction_cols):
                R[i, idx_map[col]] = 1
            wald_cr = model_cr.wald_test(R, use_f=True, scalar=True)
            cr_F = float(wald_cr.statistic)
            cr_p = float(wald_cr.pvalue)
            cr_dfn = int(wald_cr.df_num)
            cr_dfd = int(wald_cr.df_denom)

            from statsmodels.genmod.cov_struct import Exchangeable  # noqa: PLC0415
            from statsmodels.genmod.generalized_estimating_equations import (  # noqa: PLC0415
                GEE,
            )

            gee = GEE.from_formula(
                "psr ~ C(protocol) * C(family)",
                groups="morph",
                data=df,
                cov_struct=Exchangeable(),
            ).fit()
            gee_interaction_cols = [c for c in gee.params.index if ":" in c]
            R_gee = np.zeros((len(gee_interaction_cols), len(gee.params)))
            idx_map_gee = {n: i for i, n in enumerate(gee.params.index)}
            for i, col in enumerate(gee_interaction_cols):
                R_gee[i, idx_map_gee[col]] = 1
            wald_gee = gee.wald_test(R_gee, use_f=False, scalar=True)
            gee_chi2 = float(wald_gee.statistic)
            gee_p = float(wald_gee.pvalue)
            gee_df = int(wald_gee.df_denom)

        print(
            f"  Cluster-robust F-Wald (interaction; cluster=morph): "
            f"F({cr_dfn},{cr_dfd})={cr_F:.3f}, p={cr_p:.4g}  "
            "(paper cluster-robust: F(4,99)=4.78, p=0.001)"
        )
        print(
            f"  GEE exchangeable Wald (cluster=morph): "
            f"chi^2({gee_df})={gee_chi2:.3f}, p={gee_p:.4g}"
        )
        out["twoway_interaction_900"].update(
            {
                "cluster_robust_wald_F": cr_F,
                "cluster_robust_wald_p": cr_p,
                "cluster_robust_wald_df": (cr_dfn, cr_dfd),
                "gee_wald_chi2": gee_chi2,
                "gee_wald_p": gee_p,
                "gee_wald_df": gee_df,
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"  Cluster-robust / GEE Wald failed: {type(e).__name__}: {e}")

    # Within-protocol rank versions. Paper reports two (one for each model
    # specification): per-seed rank model gives p < 1e-8, seed-averaged rank
    # model gives p = 0.008. Both preserve the interaction significance.
    df_rank = df.copy()
    df_rank["rank"] = df_rank.groupby(["protocol", "seed"])["psr"].rank()
    m_rank = ols("rank ~ C(protocol)*C(family) + C(morph)", data=df_rank).fit()
    tbl_rank = anova_lm(m_rank, typ=2)
    p_rank_ps = float(tbl_rank.loc["C(protocol):C(family)", "PR(>F)"])
    print(
        f"  Per-seed rank interaction p = {p_rank_ps:.3e}  "
        f"(paper: p < 1e-8)"
    )
    out["twoway_interaction_900"]["rank_interaction_p_per_seed"] = p_rank_ps

    # ------------------------------------------------------------------
    # Section: seed-averaged two-way ANOVA (appendix F(4,194)=4.79)
    # ------------------------------------------------------------------
    section_header("D. Seed-averaged two-way ANOVA")
    rows_sa = []
    for proto in ("episodic", "strict", "reset_free"):
        for b in common:
            rows_sa.append(
                {
                    "psr": psr_mean[proto][b],
                    "protocol": proto,
                    "family": family[b],
                    "morph": b,
                }
            )
    df_sa = pd.DataFrame(rows_sa)
    m_sa = ols("psr ~ C(protocol)*C(family) + C(morph)", data=df_sa).fit()
    tbl_sa = anova_lm(m_sa, typ=2)
    print(tbl_sa.to_string())
    out["twoway_interaction_seed_avg"] = {
        "interaction_f": float(tbl_sa.loc["C(protocol):C(family)", "F"]),
        "interaction_p": float(tbl_sa.loc["C(protocol):C(family)", "PR(>F)"]),
        "df_residual": int(tbl_sa.loc["Residual", "df"]),
    }

    # Seed-averaged rank ANOVA (paper: p=0.008)
    df_sa_rank = df_sa.copy()
    df_sa_rank["rank"] = df_sa_rank.groupby("protocol")["psr"].rank()
    m_sa_rank = ols("rank ~ C(protocol)*C(family) + C(morph)", data=df_sa_rank).fit()
    tbl_sa_rank = anova_lm(m_sa_rank, typ=2)
    p_sa_rank = float(tbl_sa_rank.loc["C(protocol):C(family)", "PR(>F)"])
    print(
        f"  Seed-averaged rank interaction p = {p_sa_rank:.4f}  "
        f"(paper: 0.008)"
    )
    out["twoway_interaction_seed_avg"]["rank_interaction_p"] = p_sa_rank

    # ------------------------------------------------------------------
    # Section: per-seed one-way ANOVA (paper F=5.02/4.90/3.24)
    # ------------------------------------------------------------------
    section_header("E. Per-seed one-way ANOVA on RF/Epi ratios by family")
    per_seed_rec = {}
    for s in SEEDS:
        grs = {f: [] for f in ("floor", "vt", "mvt")}
        for b in common:
            if s in psr["episodic"][b] and s in psr["reset_free"][b]:
                e_v = psr["episodic"][b][s]
                if e_v > 0:
                    grs[family[b]].append(psr["reset_free"][b][s] / e_v)
        F_s, p_s = stats.f_oneway(*[np.array(g) for g in grs.values()])
        per_seed_rec[s] = {"f": float(F_s), "p": float(p_s)}
        print(f"  seed {s}: F = {F_s:.2f}, p = {p_s:.4f}")
    out["per_seed_oneway"] = per_seed_rec

    # ------------------------------------------------------------------
    # Section: Spearman rank correlations with bootstrap CIs
    # (paper: 0.723 [0.582,0.826]; 0.458 [0.262,0.624]; 0.206 [-0.018,0.404])
    # ------------------------------------------------------------------
    section_header("F. Spearman correlations between protocols (bootstrap CIs)")
    arr = {p: np.array([psr_mean[p][b] for b in common]) for p in psr_mean}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_all = rng.integers(0, len(common), size=(BOOTSTRAP_B, len(common)))
    corr_rec = {}
    for n1, n2 in (
        ("episodic", "reset_free"),
        ("episodic", "strict"),
        ("reset_free", "strict"),
    ):
        a, b_arr = arr[n1], arr[n2]
        rho, _ = stats.spearmanr(a, b_arr)
        rhos = np.empty(BOOTSTRAP_B)
        for i in range(BOOTSTRAP_B):
            ii = idx_all[i]
            rhos[i], _ = stats.spearmanr(a[ii], b_arr[ii])
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        corr_rec[f"{n1}_vs_{n2}"] = {
            "rho": float(rho),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
        print(f"  {n1:>10s} vs {n2:<10s}: rho = {rho:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
    out["spearman_rankings"] = corr_rec

    # Top-10 overlaps with bootstrap.
    # Bootstrap: fix one protocol's top-10, resample morphologies for the other,
    # take the 10 DISTINCT highest-ranked original morphologies, count overlap.
    # This matches the paper's "bounded set-count on a resample-variable top-10
    # pool" framing and reproduces the asymmetric CIs.
    section_header("F'. Top-10 overlap between protocols (asymmetric bootstrap)")

    def bootstrap_overlap(
        a_fixed: np.ndarray, b_resampled: np.ndarray, idx_matrix: np.ndarray
    ) -> np.ndarray:
        fixed_top10 = set(np.argsort(-a_fixed)[:10])
        rec = np.empty(idx_matrix.shape[0], dtype=int)
        for i in range(idx_matrix.shape[0]):
            ii = idx_matrix[i]
            sb_sorted = ii[np.argsort(-b_resampled[ii])]
            seen: set[int] = set()
            distinct: list[int] = []
            for x in sb_sorted:
                if int(x) not in seen:
                    seen.add(int(x))
                    distinct.append(int(x))
                if len(distinct) == 10:
                    break
            rec[i] = len(fixed_top10 & set(distinct))
        return rec

    top10_rec = {}
    for n1, n2 in (
        ("episodic", "reset_free"),
        ("episodic", "strict"),
        ("reset_free", "strict"),
    ):
        a, b_arr = arr[n1], arr[n2]
        point = len(set(np.argsort(-a)[:10]) & set(np.argsort(-b_arr)[:10]))
        overlaps = bootstrap_overlap(a, b_arr, idx_all)
        lo, hi = np.percentile(overlaps / 10.0, [2.5, 97.5])
        top10_rec[f"{n1}_vs_{n2}"] = {
            "point": point,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
        print(
            f"  {n1:>10s} vs {n2:<10s}: {point}/10 "
            f"(bootstrap 95% CI on overlap/10: [{lo:.2f}, {hi:.2f}])"
        )
    out["top10_overlaps"] = top10_rec

    # ------------------------------------------------------------------
    # Section: distance-based replication (paper 12/1, 4/10, 5/0; neutral 0/1, 0/10, 0/0)
    # ------------------------------------------------------------------
    section_header("G. Distance-based replication counts")
    dist = {
        ek_label: {
            p: seed_avg(load_per_seed(t, eval_key=ek, metric="distance"))
            for p, t in TAUS.items()
        }
        for ek_label, ek in (
            ("tau_eval_0p20", "eval_tau020"),
            ("tau_eval_neutral", "eval_neutral"),
        )
    }
    dist_rec = {}
    for ek_label in ("tau_eval_0p20", "tau_eval_neutral"):
        print(f"  {ek_label}:")
        fam_rec = {}
        ek_d = dist[ek_label]
        for fam in ("floor", "vt", "mvt"):
            bodies_f = [b for b in common if family[b] == fam]
            n_rf = sum(
                1
                for b in bodies_f
                if ek_d["episodic"][b] >= 5.0 and ek_d["reset_free"][b] < 1.0
            )
            n_str = sum(
                1
                for b in bodies_f
                if ek_d["episodic"][b] >= 5.0 and ek_d["strict"][b] < 1.0
            )
            fam_rec[fam] = {"walk_epi_stop_rf": n_rf, "walk_epi_stop_strict": n_str}
            print(
                f"    {fam:5s}: walk-epi-stop-RF = {n_rf:2d}, walk-epi-stop-Str = {n_str:2d}"
            )
        dist_rec[ek_label] = fam_rec
    out["distance_transitions"] = dist_rec

    # Distance RF/Epi ANOVA (paper: F=8.37, p<0.001)
    section_header("G'. Distance RF/Epi ANOVA by family")
    ek_d = dist["tau_eval_0p20"]
    gr = []
    for fam in ("floor", "vt", "mvt"):
        bodies_f = [b for b in common if family[b] == fam]
        gr.append(
            np.array(
                [
                    ek_d["reset_free"][b] / ek_d["episodic"][b]
                    for b in bodies_f
                    if ek_d["episodic"][b] > 0
                ]
            )
        )
    F_d, p_d = stats.f_oneway(*gr)
    print(f"  F = {F_d:.2f}, p = {p_d:.5f}  (paper: F=8.37, p<0.001)")
    out["distance_rf_over_epi_anova"] = {"f": float(F_d), "p": float(p_d)}

    # Fixed-horizon fitness/Tmax robustness (paper Appendix H.7)
    section_header("G''. Fixed-horizon fitness/Tmax RF/Epi ANOVA")
    fitness = {
        p: seed_avg(load_per_seed(t, eval_key=EVAL_KEY, metric="fitness"))
        for p, t in TAUS.items()
    }
    fit_rec: dict[str, Any] = {}
    fit_groups = []
    for fam in ("floor", "vt", "mvt"):
        bodies_f = [b for b in common if family[b] == fam]
        rf_ratios = np.array(
            [
                fitness["reset_free"][b] / fitness["episodic"][b]
                for b in bodies_f
                if fitness["episodic"][b] > 0
            ]
        )
        str_ratios = np.array(
            [
                fitness["strict"][b] / fitness["episodic"][b]
                for b in bodies_f
                if fitness["episodic"][b] > 0
            ]
        )
        fit_groups.append(rf_ratios)
        fit_rec[fam] = {
            "n": len(rf_ratios),
            "fit_str_over_epi": float(np.mean(str_ratios)),
            "fit_rf_over_epi": float(np.mean(rf_ratios)),
        }
        print(
            f"  {fam.upper():5s} n={len(rf_ratios):3d}  "
            f"fitness Str/Epi={np.mean(str_ratios):.3f}  "
            f"fitness RF/Epi={np.mean(rf_ratios):.3f}"
        )
    F_fit, p_fit = stats.f_oneway(*fit_groups)
    fit_rank_corr = {}
    for protocol in ("episodic", "strict", "reset_free"):
        rho = stats.spearmanr(
            [psr_mean[protocol][b] for b in common],
            [fitness[protocol][b] / 1000.0 for b in common],
        ).statistic
        fit_rank_corr[protocol] = float(rho)
    print(f"  F = {F_fit:.2f}, p = {p_fit:.6f}  (paper: F=8.37, p=4.4e-4)")
    print(
        "  PSR-vs-fitness/Tmax Spearman rho = "
        + ", ".join(f"{k}:{v:.3f}" for k, v in fit_rank_corr.items())
    )
    out["fixed_horizon_fitness_tmax"] = {
        "family_ratios": fit_rec,
        "oneway_anova_fit_rf_epi": {"f": float(F_fit), "p": float(p_fit)},
        "rank_corr_psr_vs_fitness_tmax": fit_rank_corr,
    }

    # ------------------------------------------------------------------
    # Section: residual-Delta regression + family ANOVA
    # (paper: R^2=0.39, residual=78% SD, F=8.89 p=0.0003, H=15.46 p=0.0004)
    # ------------------------------------------------------------------
    section_header("H. Residual Delta regression on z_0 + n_limbs")
    with FEATURES.open() as f:
        feats = json.load(f)
    delta, z0, nl, fams = [], [], [], []
    for b in common:
        if b not in feats:
            continue
        fv = feats[b]
        delta.append(psr_mean["episodic"][b] - psr_mean["reset_free"][b])
        z0.append(fv["init_height"])
        # paper: "all non-torso body elements"
        nl.append(fv["n_bodies"] - 1)
        fams.append(family[b])
    delta = np.array(delta)
    z0 = np.array(z0)
    nl = np.array(nl)
    X = np.column_stack([z0, nl, np.ones_like(z0)])
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
    pred = X @ beta
    resid = delta - pred
    r2 = 1 - float(np.sum(resid**2)) / float(np.sum((delta - delta.mean()) ** 2))
    frac = float(resid.std(ddof=1)) / float(delta.std(ddof=1))
    print(f"  R^2 = {r2:.4f}  (paper: 0.39)")
    print(
        f"  beta_z0 = {beta[0]:.4f}  beta_n_limbs = {beta[1]:.4f}  intercept = {beta[2]:.4f}"
    )
    print(
        f"  raw Δ SD = {delta.std(ddof=1):.4f}, resid SD = {resid.std(ddof=1):.4f}, "
        f"frac = {frac:.4f}  (paper: 78%)"
    )
    groups_r = {f: [] for f in ("floor", "vt", "mvt")}
    fam_means = {}
    for i, f in enumerate(fams):
        groups_r[f].append(resid[i])
    for f in ("floor", "vt", "mvt"):
        g = np.array(groups_r[f])
        d_g = np.array([delta[i] for i, ff in enumerate(fams) if ff == f])
        print(
            f"    {f:5s}: n={len(g):3d}  raw Δ = {d_g.mean():+.3f} ± {d_g.std(ddof=1):.3f}"
            f"  residual Δ = {g.mean():+.3f} ± {g.std(ddof=1):.3f}"
        )
        fam_means[f] = {
            "n": len(g),
            "raw_mean": float(d_g.mean()),
            "raw_sd": float(d_g.std(ddof=1)),
            "residual_mean": float(g.mean()),
            "residual_sd": float(g.std(ddof=1)),
        }
    F_r, p_rF = stats.f_oneway(*[np.array(groups_r[f]) for f in ("floor", "vt", "mvt")])
    H_r, p_rH = stats.kruskal(*[np.array(groups_r[f]) for f in ("floor", "vt", "mvt")])
    print(f"  ANOVA F = {F_r:.3f}, p = {p_rF:.6f}  (paper: F=8.89, p=0.0003)")
    print(f"  Kruskal-Wallis H = {H_r:.3f}, p = {p_rH:.6f}  (paper: H=15.46, p=0.0004)")
    out["residual_delta_regression"] = {
        "r_squared": float(r2),
        "beta_z0": float(beta[0]),
        "beta_n_limbs": float(beta[1]),
        "intercept": float(beta[2]),
        "raw_delta_sd": float(delta.std(ddof=1)),
        "residual_delta_sd": float(resid.std(ddof=1)),
        "residual_frac_of_raw": frac,
        "family_stats": fam_means,
        "family_anova_f": float(F_r),
        "family_anova_p": float(p_rF),
        "family_kruskal_h": float(H_r),
        "family_kruskal_p": float(p_rH),
    }

    # ------------------------------------------------------------------
    # Section: neutral-evaluation robustness
    # (paper: Floor RF/Epi=0.643, Str/Epi=0.763; VT 0.953/0.673; MVT 0.847/0.929)
    # ------------------------------------------------------------------
    section_header("I. Neutral-evaluation (tau_eval=0) robustness")
    psr_neutral = {
        p: seed_avg(load_per_seed(t, eval_key="eval_neutral")) for p, t in TAUS.items()
    }
    neutral_rec = {}
    for fam in ("floor", "vt", "mvt"):
        bodies_f = [b for b in common if family[b] == fam]
        rfepi = float(
            np.mean(
                [
                    psr_neutral["reset_free"][b] / psr_neutral["episodic"][b]
                    for b in bodies_f
                    if psr_neutral["episodic"][b] > 0
                ]
            )
        )
        stepi = float(
            np.mean(
                [
                    psr_neutral["strict"][b] / psr_neutral["episodic"][b]
                    for b in bodies_f
                    if psr_neutral["episodic"][b] > 0
                ]
            )
        )
        neutral_rec[fam] = {"rf_over_epi": rfepi, "str_over_epi": stepi}
        print(f"  {fam.upper():5s}: RF/Epi = {rfepi:.3f}, Str/Epi = {stepi:.3f}")
    out["neutral_evaluation_ratios"] = neutral_rec

    # ------------------------------------------------------------------
    # Pointer: permutation interaction p-values live in their own artifact
    # (too slow to redo B=5000 refits on every reproduction run). Paper's
    # permutation claims come from scripts/permutation_interaction.py.
    # To prevent stale numbers if the corpus changes, hash the 900-row
    # (body, protocol, seed, psr) input and compare against the cached hash.
    import hashlib  # noqa: PLC0415

    perm_path = OUT_JSON.parent / "permutation_interaction_report.json"
    corpus_input_hash: str | None = None
    try:
        corpus_long = df.sort_values(["morph", "protocol", "seed"])[
            ["morph", "protocol", "seed", "psr"]
        ]
        payload = corpus_long.to_csv(index=False).encode("utf-8")
        corpus_input_hash = hashlib.sha256(payload).hexdigest()[:16]
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not compute corpus input hash for permutation cache: {e}")

    if perm_path.exists():
        print("\n" + "=" * 72)
        print("J. Permutation interaction tests (cached)")
        print("=" * 72)
        with perm_path.open() as f:
            perm = json.load(f)
        cached_hash = perm.get("corpus_input_sha256_16")
        hash_ok = (
            corpus_input_hash is not None
            and cached_hash is not None
            and cached_hash == corpus_input_hash
        )
        allow_stale = os.environ.get("ALLOW_STALE_PERMUTATION_CACHE") == "1"
        if corpus_input_hash is None:
            print(
                "  WARN: could not compute current corpus hash; permutation "
                "numbers will NOT be copied into the reproduction artifact."
            )
        elif cached_hash is None:
            print(
                "  WARN: permutation cache has no corpus_input_sha256_16 field. "
                "Re-run scripts/permutation_interaction.py to regenerate with "
                "hash. Skipping permutation block in reproduction artifact."
            )
        elif not hash_ok:
            msg = (
                f"STALE CACHE: permutation_interaction_report.json hash "
                f"{cached_hash} does not match current corpus hash "
                f"{corpus_input_hash}. Re-run "
                "scripts/permutation_interaction.py."
            )
            if allow_stale:
                print(f"  {msg}\n  ALLOW_STALE_PERMUTATION_CACHE=1 set; "
                      "continuing anyway (NOT acceptable for paper claims).")
            else:
                raise RuntimeError(
                    msg
                    + " (Set ALLOW_STALE_PERMUTATION_CACHE=1 to override "
                    "for debugging only.)"
                )
        for scheme_key in ("scheme_A", "scheme_B"):
            if scheme_key in perm:
                s = perm[scheme_key]
                print(
                    f"  {scheme_key}: F_obs={s['F_observed']:.2f}, "
                    f"null p95={s['null_p95']:.2f}, "
                    f"p_perm={s['p_permutation']:.4g} (B={s['B']})"
                )
        if hash_ok:
            out["permutation_interaction"] = perm
        out["permutation_interaction_cache_meta"] = {
            "current_corpus_hash": corpus_input_hash,
            "cached_corpus_hash": cached_hash,
            "hash_match": hash_ok,
            "permutation_copied_into_artifact": hash_ok,
        }

    OUT_JSON.parent.mkdir(exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote reproduction artifact to {OUT_JSON}")


if __name__ == "__main__":
    main()
