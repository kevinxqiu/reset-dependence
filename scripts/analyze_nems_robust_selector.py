#!/usr/bin/env python3
"""Pre-registered NEMS held-out robust selector analysis.

Spec: records/nems_robust_selector_spec.md (FROZEN 2026-04-23).

Tests whether a worst-case-PSR selector fit on the 100-body corpus
generalises zero-shot to the 60-body NEMS panel. Writes the analysis
JSON and prints the gate verdict.

Exit code 0 regardless of verdict.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
NEMS_DIR = RESULTS / "nems_panel"
OUT = REPO / "records" / "nems_robust_selector_analysis.json"

CORPUS_FEATS_JSON = RESULTS / "morph_features.json"
NEMS_FEATS_JSON = RESULTS / "morph_features_nems.json"

SEEDS = (42, 43, 44)
TAUS_CORPUS = {"epi": "0p20", "strict": "0p50", "rf": "0p00"}
TAUS_NEMS = {"epi": "0p20", "strict": "0p50", "rf": "0p00"}
EVAL_KEY = "eval_tau020"
PRIMARY_K = 10
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42

# Frozen five-feature set from M2 analysis (see records/nems_robust_selector_spec.md).
FEATURES = (
    "init_height",
    "n_first_level_limbs",
    "tree_depth",
    "n_joints",
    "n_bodies",
)

ALPHAS = (0.1, 1.0, 10.0)


def load_corpus_psr(tau_str: str) -> dict[str, float]:
    vals: defaultdict[str, list[float]] = defaultdict(list)
    for s in SEEDS:
        path = RESULTS / f"mjx_ppo_{tau_str}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get(EVAL_KEY, {}).get("per_step_reward")
            if psr is not None:
                vals[body].append(float(psr))
    return {body: float(np.mean(v)) for body, v in vals.items()}


def load_nems_psr(tau_str: str) -> dict[str, float]:
    vals: defaultdict[str, list[float]] = defaultdict(list)
    for s in SEEDS:
        path = NEMS_DIR / f"tau{tau_str}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get(EVAL_KEY, {}).get("per_step_reward")
            if psr is not None:
                vals[body].append(float(psr))
    return {body: float(np.mean(v)) for body, v in vals.items()}


def build_feature_matrix(body_ids: list[str], feats: dict) -> np.ndarray:
    rows = []
    for bid in body_ids:
        row = [float(feats[bid][f]) for f in FEATURES]
        rows.append(row)
    return np.array(rows, dtype=float)


def cv_select_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return best alpha by 5-fold corpus CV, with its CV R^2."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for alpha in ALPHAS:
        cv_r2s = []
        for train_idx, val_idx in kf.split(x):
            scaler = StandardScaler().fit(x[train_idx])
            x_tr = scaler.transform(x[train_idx])
            x_va = scaler.transform(x[val_idx])
            model = Ridge(alpha=alpha).fit(x_tr, y[train_idx])
            pred = model.predict(x_va)
            ss_res = float(np.sum((y[val_idx] - pred) ** 2))
            ss_tot = float(np.sum((y[val_idx] - y[val_idx].mean()) ** 2))
            cv_r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        scores[alpha] = float(np.mean(cv_r2s))
    best_alpha = max(scores, key=lambda a: scores[a])
    return best_alpha, scores[best_alpha]


def fit_corpus_and_predict_nems(
    corpus_x: np.ndarray,
    corpus_y: np.ndarray,
    nems_x: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    alpha, cv_r2 = cv_select_alpha(corpus_x, corpus_y)
    scaler = StandardScaler().fit(corpus_x)
    x_tr = scaler.transform(corpus_x)
    x_nems = scaler.transform(nems_x)
    model = Ridge(alpha=alpha).fit(x_tr, corpus_y)
    in_sample_r2 = float(model.score(x_tr, corpus_y))
    preds_nems = model.predict(x_nems)
    return preds_nems, float(alpha), cv_r2, in_sample_r2


def top_k_retention(
    predicted_scores: np.ndarray,
    target_strict: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Return (retention, delta_vs_oracle)."""
    idx_sel = np.argsort(-predicted_scores)[:k]
    idx_oracle = np.argsort(-target_strict)[:k]
    mean_sel = float(np.mean(target_strict[idx_sel]))
    mean_oracle = float(np.mean(target_strict[idx_oracle]))
    retention = mean_sel / mean_oracle if mean_oracle > 0 else float("nan")
    return retention, mean_sel - mean_oracle


def bootstrap_metric(
    pred_wc: np.ndarray,
    pred_epi: np.ndarray,
    strict_vec: np.ndarray,
    k: int,
    rng: np.random.Generator,
    b: int = BOOTSTRAP_B,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap retention and delta-vs-epi over body resamples."""
    retentions_wc = np.empty(b)
    retentions_epi = np.empty(b)
    deltas = np.empty(b)
    n = len(strict_vec)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        s = strict_vec[idx]
        pwc = pred_wc[idx]
        pepi = pred_epi[idx]
        r_wc, _ = top_k_retention(pwc, s, k)
        r_epi, _ = top_k_retention(pepi, s, k)
        retentions_wc[i] = r_wc
        retentions_epi[i] = r_epi
        # delta-vs-epi: mean strict-PSR of WC-selected minus mean of epi-selected
        sel_wc = np.argsort(-pwc)[:k]
        sel_epi = np.argsort(-pepi)[:k]
        deltas[i] = float(np.mean(s[sel_wc])) - float(np.mean(s[sel_epi]))
    return {
        "retention_wc": (
            float(np.mean(retentions_wc)),
            float(np.percentile(retentions_wc, 2.5)),
            float(np.percentile(retentions_wc, 97.5)),
        ),
        "retention_epi": (
            float(np.mean(retentions_epi)),
            float(np.percentile(retentions_epi, 2.5)),
            float(np.percentile(retentions_epi, 97.5)),
        ),
        "delta_wc_vs_epi": (
            float(np.mean(deltas)),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ),
    }


def apply_gate(
    retention_ci: tuple[float, float, float], delta_ci: tuple[float, float, float]
) -> str:
    ret_mean, ret_lo, ret_hi = retention_ci
    d_mean, d_lo, d_hi = delta_ci
    main_text = ret_lo > 0.85 and d_lo > 0.05
    appendix_a = ret_lo > 0.70 and ret_mean > 0.85
    appendix_b = (d_lo <= 0.0 <= d_hi) and d_mean > 0.05 and ret_mean > 0.85
    if main_text:
        return "MAIN-TEXT"
    if appendix_a or appendix_b:
        return "APPENDIX"
    return "DROP"


def main() -> None:
    # Load features
    with CORPUS_FEATS_JSON.open() as f:
        corpus_feats = json.load(f)
    with NEMS_FEATS_JSON.open() as f:
        nems_feats = json.load(f)

    # Load PSR
    corpus_psr = {name: load_corpus_psr(t) for name, t in TAUS_CORPUS.items()}
    nems_psr = {name: load_nems_psr(t) for name, t in TAUS_NEMS.items()}

    # QC 4: check no identity overlap
    overlap = set(corpus_psr["epi"]) & set(nems_psr["epi"])
    if overlap:
        raise RuntimeError(f"QC FAIL: {len(overlap)} body IDs overlap corpus and NEMS")

    # Align bodies (corpus needs all 3 protocols AND features)
    corpus_bodies = sorted(
        set(corpus_psr["epi"])
        & set(corpus_psr["strict"])
        & set(corpus_psr["rf"])
        & set(corpus_feats)
    )
    nems_bodies = sorted(
        set(nems_psr["epi"])
        & set(nems_psr["strict"])
        & set(nems_psr["rf"])
        & set(nems_feats)
    )

    epi_corpus = np.array([corpus_psr["epi"][b] for b in corpus_bodies])
    str_corpus = np.array([corpus_psr["strict"][b] for b in corpus_bodies])
    rf_corpus = np.array([corpus_psr["rf"][b] for b in corpus_bodies])
    wc_corpus = np.minimum(np.minimum(epi_corpus, str_corpus), rf_corpus)

    str_nems = np.array([nems_psr["strict"][b] for b in nems_bodies])

    corpus_x = build_feature_matrix(corpus_bodies, corpus_feats)
    nems_x = build_feature_matrix(nems_bodies, nems_feats)

    # Fit WC predictor on corpus, apply to NEMS
    pred_wc_nems, alpha_wc, cv_r2_wc, in_r2_wc = fit_corpus_and_predict_nems(
        corpus_x, wc_corpus, nems_x
    )
    # Fit Epi predictor on corpus, apply to NEMS
    pred_epi_nems, alpha_epi, cv_r2_epi, in_r2_epi = fit_corpus_and_predict_nems(
        corpus_x, epi_corpus, nems_x
    )

    # QC 5: predictor CV R^2 on corpus >= 0.20
    qc_r2_pass = cv_r2_wc >= 0.20

    # Point estimates at k = PRIMARY_K
    retention_wc_point, delta_wc_point = top_k_retention(
        pred_wc_nems, str_nems, PRIMARY_K
    )
    retention_epi_point, delta_epi_point = top_k_retention(
        pred_epi_nems, str_nems, PRIMARY_K
    )

    # Bootstrap
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = bootstrap_metric(pred_wc_nems, pred_epi_nems, str_nems, PRIMARY_K, rng)
    retention_ci = boot["retention_wc"]
    delta_ci = boot["delta_wc_vs_epi"]

    # Secondary k for transparency (NOT promoted)
    aux = {}
    for k in (5, 20):
        rng_aux = np.random.default_rng(BOOTSTRAP_SEED + k)
        boot_aux = bootstrap_metric(pred_wc_nems, pred_epi_nems, str_nems, k, rng_aux)
        aux[f"k{k}"] = boot_aux

    # Gate
    if not qc_r2_pass:
        verdict = "DROP"
        verdict_reason = f"QC FAIL: CV R^2 = {cv_r2_wc:.3f} < 0.20 threshold"
    else:
        verdict = apply_gate(retention_ci, delta_ci)
        verdict_reason = (
            f"retention CI [{retention_ci[1]:.3f}, {retention_ci[2]:.3f}] mean {retention_ci[0]:.3f}; "
            f"delta-vs-epi CI [{delta_ci[1]:.3f}, {delta_ci[2]:.3f}] mean {delta_ci[0]:.3f}"
        )

    out = {
        "spec": "records/nems_robust_selector_spec.md",
        "spec_frozen": "2026-04-23",
        "primary_k": PRIMARY_K,
        "n_corpus": len(corpus_bodies),
        "n_nems": len(nems_bodies),
        "predictor_worst_case": {
            "alpha": alpha_wc,
            "cv_r2": cv_r2_wc,
            "in_sample_r2": in_r2_wc,
        },
        "predictor_epi": {
            "alpha": alpha_epi,
            "cv_r2": cv_r2_epi,
            "in_sample_r2": in_r2_epi,
        },
        "point_estimates_k10": {
            "retention_wc": retention_wc_point,
            "retention_epi": retention_epi_point,
            "delta_wc_vs_oracle": delta_wc_point,
            "delta_epi_vs_oracle": delta_epi_point,
        },
        "bootstrap_k10": {
            "retention_wc_ci": retention_ci,
            "retention_epi_ci": boot["retention_epi"],
            "delta_wc_vs_epi_ci": delta_ci,
        },
        "secondary_k_aux": {
            k: {
                "retention_wc_ci": v["retention_wc"],
                "retention_epi_ci": v["retention_epi"],
                "delta_wc_vs_epi_ci": v["delta_wc_vs_epi"],
            }
            for k, v in aux.items()
        },
        "qc": {
            "no_id_overlap": len(overlap) == 0,
            "cv_r2_pass": qc_r2_pass,
            "cv_r2_threshold": 0.20,
            "n_corpus_expected": 100,
            "n_nems_expected": 60,
        },
        "gate_verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    # Pretty print summary
    print("=" * 70)
    print("NEMS Held-Out Robust Selector Analysis")
    print("=" * 70)
    print(f"Corpus bodies: {len(corpus_bodies)}   NEMS bodies: {len(nems_bodies)}")
    print(f"ID overlap: {len(overlap)}")
    print(f"Worst-case predictor  CV R^2 = {cv_r2_wc:.3f} (alpha = {alpha_wc})")
    print(f"Epi predictor         CV R^2 = {cv_r2_epi:.3f} (alpha = {alpha_epi})")
    print(f"QC R^2 >= 0.20: {qc_r2_pass}")
    print()
    print(f"Point estimate (k={PRIMARY_K}):")
    print(f"  Retention WC:  {retention_wc_point:.3f}")
    print(f"  Retention Epi: {retention_epi_point:.3f}")
    print()
    print(f"Bootstrap 95% CI (k={PRIMARY_K}, B={BOOTSTRAP_B}):")
    print(
        f"  Retention WC CI: [{retention_ci[1]:.3f}, {retention_ci[2]:.3f}] (mean {retention_ci[0]:.3f})"
    )
    print(
        f"  Delta-vs-epi CI: [{delta_ci[1]:.3f}, {delta_ci[2]:.3f}] (mean {delta_ci[0]:.3f})"
    )
    print()
    print(f"Gate verdict: {verdict}")
    print(f"  Reason: {verdict_reason}")
    print()
    print(f"Full analysis written to {OUT}")


if __name__ == "__main__":
    main()
