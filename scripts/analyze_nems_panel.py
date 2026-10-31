#!/usr/bin/env python3
"""Analyze the NEMS 540-run panel against the pre-registered spec.

Spec: records/nems_task_spec.md (frozen 2026-04-16).

Primary endpoint: does a structural-feature predictor of per-body protocol
sensitivity, trained on the 100-body UNIMAL corpus, transfer zero-shot
to the 60 NEMS bodies?

Outputs:
  records/nems_panel_report.md
  records/nems_analysis.json
  figures/fig_nems.pdf
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
NEMS_DIR = REPO / "results/nems_panel"
CORPUS_DIR = REPO / "results"


def _first_existing_path(candidates: tuple[Path | str | None, ...], fallback: Path) -> Path:
    for path in candidates:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.is_dir():
            return candidate
    return fallback


NEMS_XML_DIR = _first_existing_path(
    (
        os.environ.get("RFC_NEMS_XML_DIR"),
        REPO / "assets/generated_morphologies/xml",
    ),
    REPO / "assets/generated_morphologies/xml",
)
CORPUS_XML_DIR = _first_existing_path(
    (
        os.environ.get("RFC_UNIMAL_XML_DIR"),
        REPO / "assets/unimals_100/xml",
    ),
    REPO / "assets/unimals_100/xml",
)
PANEL_JSON = REPO / "records/nems_panel_bodies.json"
OUT_MD = REPO / "records/nems_panel_report.md"
OUT_JSON = REPO / "records/nems_analysis.json"
OUT_FIG = REPO / "figures/fig_nems.pdf"

TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}
SEEDS = (42, 43, 44)
N_BOOT = 10_000

# 5 features, matching the paper's Table tab:features
FEATURE_NAMES = (
    "initial_height",
    "num_first_level_limbs",
    "tree_depth",
    "num_joints",
    "num_bodies",
)


def extract_features(xml_path: Path) -> dict[str, float]:
    """Pull the 5 pre-registered structural features from a UNIMAL XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bodies = root.findall(".//body")
    joints = [
        j
        for j in root.findall(".//joint")
        if j.get("type", "hinge") == "hinge" and j.get("class") != "free"
    ]
    torso = root.find('.//body[@name="torso/0"]')
    if torso is not None:
        pos = torso.get("pos", "0 0 0").split()
        initial_height = float(pos[2]) if len(pos) >= 3 else 0.0
        num_first_level_limbs = len([b for b in torso if b.tag == "body"])
        tree_depth = _tree_depth(torso)
    else:
        initial_height = 0.0
        num_first_level_limbs = 0
        tree_depth = 0
    return {
        "initial_height": initial_height,
        "num_first_level_limbs": num_first_level_limbs,
        "tree_depth": tree_depth,
        "num_joints": len(joints),
        "num_bodies": len(bodies),
    }


def _tree_depth(elem) -> int:
    children = [c for c in elem if c.tag == "body"]
    if not children:
        return 0
    return 1 + max(_tree_depth(c) for c in children)


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def load_nems_panel() -> dict[str, dict]:
    """Return {body_name: {protocol: {seed: {psr, distance, ep_len, fall_rate}}}}.

    Enforces two pre-registration invariants:
      1. The set of body names analysed equals the 60 names in
         records/nems_panel_bodies.json exactly.
      2. Every included body has full 3-seed x 3-protocol coverage. Bodies
         missing any cell are dropped with a warning, NOT silently included
         with partial seeds.
    """
    with PANEL_JSON.open() as f:
        frozen = json.load(f)
    frozen_names = {p["name"] for p in frozen["panel"]}

    raw: dict = defaultdict(lambda: defaultdict(dict))
    missing_files = []
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            path = NEMS_DIR / f"tau{tau}_seed{seed}" / "results.json"
            if not path.is_file():
                missing_files.append(str(path))
                continue
            with path.open() as f:
                data = json.load(f)
            for name, rec in data.items():
                if name not in frozen_names:
                    continue  # Ignore any drift/contamination
                ev = rec.get("eval_tau020", {})
                raw[name][proto][seed] = {
                    "psr": float(ev.get("per_step_reward", float("nan"))),
                    "distance": float(ev.get("distance", float("nan"))),
                    "ep_len": float(ev.get("mean_ep_len", float("nan"))),
                    "fall_rate": float(ev.get("fall_rate", float("nan"))),
                }

    if missing_files:
        print(f"WARNING: {len(missing_files)} results.json missing (first 5):")
        for m in missing_files[:5]:
            print(f"  - {m}")

    # Enforce full coverage: each included body needs 3 seeds x 3 protocols.
    # "Coverage" here is cell-presence (records exist in results.json); the
    # PSR values inside may still be NaN (see write_report panel-health line
    # for NaN-cell reporting).
    expected_cells = len(TAUS) * len(SEEDS)
    full_coverage: dict = {}
    partial = []
    for name in frozen_names:
        cells = sum(len(raw[name].get(p, {})) for p in TAUS)
        if cells == expected_cells:
            full_coverage[name] = dict(raw[name])
        else:
            partial.append((name, cells))
    missing_names = frozen_names - set(full_coverage.keys()) - {p[0] for p in partial}
    if partial:
        print(
            f"ERROR: {len(partial)} frozen-panel bodies have partial coverage "
            f"(expected {expected_cells} cells). First 10:"
        )
        for n, c in partial[:10]:
            print(f"  - {n}: {c} / {expected_cells} cells")
    if missing_names:
        print(
            f"ERROR: {len(missing_names)} frozen-panel bodies have ZERO "
            f"cells in results/. First 5: {sorted(missing_names)[:5]}"
        )
    import os  # noqa: PLC0415

    allow_partial = os.environ.get("ALLOW_PARTIAL_NEMS") == "1"
    if partial or missing_names:
        msg = (
            f"NEMS panel incomplete: {len(frozen_names) - len(full_coverage)} "
            f"of {len(frozen_names)} frozen bodies lack full "
            f"{expected_cells}-cell coverage. Pre-registration requires exact "
            "panel match; refusing to analyze partial panel."
        )
        if allow_partial:
            print(
                f"NOTE: {msg} ALLOW_PARTIAL_NEMS=1 set; continuing with "
                "partial panel (NOT acceptable for paper claims)."
            )
        else:
            raise RuntimeError(
                msg
                + " Set ALLOW_PARTIAL_NEMS=1 in the environment to override "
                "for debugging (not acceptable for paper claims)."
            )
    print(
        f"NEMS panel: {len(full_coverage)} bodies with full "
        f"{expected_cells}-cell coverage (out of {len(frozen_names)} frozen)."
    )
    return full_coverage


def load_corpus_panel(panel_names: set[str] | None = None) -> dict[str, dict]:
    """Seed-averaged per-body PSR dict for corpus (100 UNIMAL bodies).

    Explicitly raises on missing fields rather than defaulting to 0.0, to
    avoid silent corruption of the fitted corpus target.

    `panel_names`: when provided, restrict iteration to these body names so
    that an unrelated divergent body in the corpus does not block the NEMS
    analysis (would previously raise ValueError for any non-NEMS body with a
    missing eval block).
    """
    out: dict = defaultdict(lambda: defaultdict(list))
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            path = CORPUS_DIR / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            if not path.is_file():
                continue
            with path.open() as f:
                data = json.load(f)
            for name, rec in data.items():
                if panel_names is not None and name not in panel_names:
                    continue
                ev = rec.get("eval_tau020")
                if ev is None or "per_step_reward" not in ev:
                    raise ValueError(
                        f"Corpus record {name} {proto} seed={seed} missing "
                        f"eval_tau020.per_step_reward; refusing to silently "
                        f"default to 0.0"
                    )
                out[name][proto].append(float(ev["per_step_reward"]))
    return {
        name: {p: float(np.mean(v)) for p, v in protos.items()}
        for name, protos in out.items()
    }


def seed_average(nems_panel: dict) -> dict:
    """Collapse seeds: {body: {protocol: {psr_mean, dist_mean, ...}}}."""
    out = {}
    for name, protos in nems_panel.items():
        out[name] = {}
        for proto, per_seed in protos.items():
            psrs = [v["psr"] for v in per_seed.values()]
            dists = [v["distance"] for v in per_seed.values()]
            lens = [v["ep_len"] for v in per_seed.values()]
            out[name][proto] = {
                "psr": float(np.nanmean(psrs)) if psrs else float("nan"),
                "distance": float(np.nanmean(dists)) if dists else float("nan"),
                "ep_len": float(np.nanmean(lens)) if lens else float("nan"),
                "n_seeds": sum(not np.isnan(x) for x in psrs),
            }
    return out


def fit_corpus_predictors(corpus: dict, xml_dir: Path) -> dict:
    """Fit three corpus predictors (OLS, Ridge, RandomForest) and report LOO-CV.

    Primary model remains OLS (per pre-registration). Ridge and RandomForest
    are reported as sensitivity analyses (design review, round 3): features
    have condition number ~121 (collinearity from num_joints and num_bodies
    both tracking num_limbs), so regularization and non-parametric fits
    guard against unstable OLS extrapolation on the 30% of NEMS bodies that
    fall outside the corpus initial_height range.
    """
    X, y = [], []
    for name, protos in corpus.items():
        if (
            "episodic" not in protos
            or "strict" not in protos
            or "reset_free" not in protos
        ):
            continue
        xml_p = xml_dir / f"{name}.xml"
        if not xml_p.is_file():
            continue
        feats = extract_features(xml_p)
        X.append([feats[k] for k in FEATURE_NAMES])
        y.append(protos["strict"] - protos["reset_free"])
    X = np.array(X)
    y = np.array(y)

    models = {
        "ols": LinearRegression().fit(X, y),
        "ridge": Ridge(alpha=1.0).fit(X, y),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y),
    }

    # LOO-CV inside the corpus: benchmark for transfer expectations
    loo = LeaveOneOut()
    loo_rho: dict[str, float] = {}
    for tag, cls_factory in (
        ("ols", lambda: LinearRegression()),
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("rf", lambda: RandomForestRegressor(n_estimators=100, random_state=42)),
    ):
        preds = np.empty(len(y))
        for tr, te in loo.split(X):
            preds[te] = cls_factory().fit(X[tr], y[tr]).predict(X[te])[0]
        rho, _ = spearmanr(preds, y)
        loo_rho[tag] = float(rho)

    print(
        f"Corpus fit on {len(y)} bodies. Within-corpus LOO-CV Spearman rho "
        f"by model: OLS={loo_rho['ols']:.3f}, Ridge={loo_rho['ridge']:.3f}, "
        f"RF={loo_rho['rf']:.3f}"
    )
    return {"models": models, "loo_rho": loo_rho, "n_corpus": len(y)}


def primary_analysis(
    nems_seed_avg: dict, corpus_fit: dict, xml_dir: Path, corpus_xml_dir: Path
) -> dict:
    """Zero-shot transfer test under OLS (primary) + Ridge/RF (sensitivity).

    Also computes an OOD-excluded variant: drop NEMS bodies whose
    initial_height is outside the corpus range, then re-run the primary
    OLS test on the in-distribution subset only (design review
    concern: 30% of NEMS bodies extrapolate beyond corpus).
    """
    # Build corpus feature range for OOD filter
    corpus_feats = []
    for xml_p in sorted(corpus_xml_dir.glob("*.xml")):
        corpus_feats.append(extract_features(xml_p))
    ih_min = float(min(f["initial_height"] for f in corpus_feats))
    ih_max = float(max(f["initial_height"] for f in corpus_feats))

    rows = []
    for name, protos in nems_seed_avg.items():
        xml_p = xml_dir / f"{name}.xml"
        if not xml_p.is_file():
            continue
        if (
            "episodic" not in protos
            or "strict" not in protos
            or "reset_free" not in protos
        ):
            continue
        feats = extract_features(xml_p)
        s_psr = protos["strict"]["psr"]
        r_psr = protos["reset_free"]["psr"]
        if np.isnan(s_psr) or np.isnan(r_psr):
            continue
        observed = s_psr - r_psr
        xvec = np.array([[feats[k] for k in FEATURE_NAMES]])
        preds = {
            tag: float(corpus_fit["models"][tag].predict(xvec)[0])
            for tag in ("ols", "ridge", "rf")
        }
        rows.append(
            {
                "name": name,
                "features": feats,
                "observed_str_minus_rf": float(observed),
                "predicted_ols": preds["ols"],
                "predicted_ridge": preds["ridge"],
                "predicted_rf": preds["rf"],
                "in_distribution": ih_min <= feats["initial_height"] <= ih_max,
            }
        )

    if not rows:
        return {"error": "no bodies loaded"}

    def _bootstrap_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict:
        sp, sp_p = spearmanr(predicted, observed)
        pe, pe_p = pearsonr(predicted, observed)
        rng = np.random.default_rng(42)
        n = len(observed)
        boot_sp = np.empty(N_BOOT)
        boot_pe = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            s, _ = spearmanr(predicted[idx], observed[idx])
            p, _ = pearsonr(predicted[idx], observed[idx])
            boot_sp[i] = s if not np.isnan(s) else 0.0
            boot_pe[i] = p if not np.isnan(p) else 0.0
        return {
            "spearman_rho": float(sp),
            "spearman_p": float(sp_p),
            "spearman_ci95": (
                float(np.percentile(boot_sp, 2.5)),
                float(np.percentile(boot_sp, 97.5)),
            ),
            "pearson_r": float(pe),
            "pearson_p": float(pe_p),
            "pearson_ci95": (
                float(np.percentile(boot_pe, 2.5)),
                float(np.percentile(boot_pe, 97.5)),
            ),
        }

    observed = np.array([r["observed_str_minus_rf"] for r in rows])
    in_dist = np.array([r["in_distribution"] for r in rows])

    all_bodies_metrics = {
        tag: _bootstrap_metrics(
            np.array([r[f"predicted_{tag}"] for r in rows]), observed
        )
        for tag in ("ols", "ridge", "rf")
    }
    in_dist_metrics = {}
    if in_dist.sum() >= 10:
        obs_in = observed[in_dist]
        for tag in ("ols", "ridge", "rf"):
            pred_in = np.array(
                [
                    r[f"predicted_{tag}"]
                    for r, b in zip(rows, in_dist, strict=False)
                    if b
                ]
            )
            in_dist_metrics[tag] = _bootstrap_metrics(pred_in, obs_in)

    # Primary pre-registered endpoint: OLS on all bodies
    primary = all_bodies_metrics["ols"]
    positive_transfer = bool(
        primary["spearman_rho"] > 0.3 and primary["spearman_ci95"][0] > 0.0
    )

    return {
        "n_bodies": len(rows),
        "n_in_distribution": int(in_dist.sum()),
        "corpus_initial_height_range": (ih_min, ih_max),
        "corpus_loo_rho": corpus_fit["loo_rho"],
        "per_body": rows,
        "metrics_all": all_bodies_metrics,
        "metrics_in_distribution": in_dist_metrics,
        "primary_endpoint_ols": primary,
        "positive_transfer": positive_transfer,
    }


CLUSTER_FEATURE_KEYS = (
    "num_limbs",
    "torso_mode_vertical",  # 1 for vertical, 0 for horizontal_y
    "initial_height",
    "num_joints",
    "tree_depth",
)


def _cluster_features_for_panel(
    nems_seed_avg: dict, xml_dir: Path, panel_manifest: list[dict]
) -> tuple[list[str], np.ndarray, dict[str, dict]]:
    """Build the pre-registered cluster feature matrix for the panel.

    Spec (records/nems_task_spec.md): cluster on
      (num_limbs, torso_mode, initial_height, num_joints, tree_depth).
    torso_mode is encoded as a binary indicator (vertical=1, horizontal_y=0).
    """
    panel_by_name = {p["name"]: p for p in panel_manifest}
    raw_feats = {}
    for name in nems_seed_avg:
        xml_p = xml_dir / f"{name}.xml"
        if not xml_p.is_file():
            continue
        raw_feats[name] = extract_features(xml_p)
    names = sorted(raw_feats.keys())
    X = np.array(
        [
            [
                float(panel_by_name[n]["num_limbs"]),
                1.0 if panel_by_name[n]["torso_mode"] == "vertical" else 0.0,
                raw_feats[n]["initial_height"],
                raw_feats[n]["num_joints"],
                raw_feats[n]["tree_depth"],
            ]
            for n in names
        ]
    )
    return names, X, raw_feats


def cluster_and_anova(
    nems_seed_avg: dict,
    xml_dir: Path,
    panel_manifest: list[dict],
    nems_panel: dict | None = None,
) -> dict:
    """3-cluster K-means on pre-registered features + blocked 2-way ANOVA.

    Pre-registered primary-secondary (per nems_task_spec.md): cluster on
    (num_limbs, torso_mode, initial_height, num_joints, tree_depth). Run
    protocol x cluster two-way ANOVA with body blocking on PSR; report
    F and partial eta^2 on the interaction term.

    Primary ANOVA is per-seed (matches main-text specification in method.tex).
    Seed-averaged version is reported as an alternative for comparison.
    """
    names, X, raw_feats = _cluster_features_for_panel(
        nems_seed_avg, xml_dir, panel_manifest
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    labels = km.labels_
    cluster_map = {n: int(labels[i]) for i, n in enumerate(names)}

    # Per-seed long-form data for primary blocked ANOVA
    rows_per_seed: list[dict] = []
    if nems_panel is not None:
        for name, c_id in cluster_map.items():
            protos = nems_panel.get(name, {})
            for proto in ("episodic", "strict", "reset_free"):
                per_seed = protos.get(proto, {})
                for seed_id, v in per_seed.items():
                    psr = v.get("psr") if isinstance(v, dict) else None
                    if psr is None or (isinstance(psr, float) and np.isnan(psr)):
                        continue
                    rows_per_seed.append(
                        {
                            "morph_name": name,
                            "cluster": f"C{c_id}",
                            "protocol": proto,
                            "seed": int(seed_id),
                            "psr": float(psr),
                        }
                    )

    # Seed-averaged long-form data (kept as secondary/for comparison)
    rows_avg: list[dict] = []
    for name, c_id in cluster_map.items():
        protos = nems_seed_avg.get(name, {})
        for proto in ("episodic", "strict", "reset_free"):
            psr = protos.get(proto, {}).get("psr", None)
            if psr is None or np.isnan(psr):
                continue
            rows_avg.append(
                {
                    "morph_name": name,
                    "cluster": f"C{c_id}",
                    "protocol": proto,
                    "psr": float(psr),
                }
            )

    def _fit_blocked(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        try:
            import pandas as pd  # noqa: PLC0415
            from statsmodels.formula.api import ols as sm_ols  # noqa: PLC0415
            from statsmodels.stats.anova import anova_lm  # noqa: PLC0415

            df = pd.DataFrame(rows)
            model = sm_ols(
                "psr ~ C(protocol) * C(cluster) + C(morph_name)",
                data=df,
            ).fit()
            tbl = anova_lm(model, typ=2)
            interaction_row = tbl.loc["C(protocol):C(cluster)"]
            ss_inter = float(interaction_row["sum_sq"])
            ss_resid = float(tbl.loc["Residual"]["sum_sq"])
            partial_eta2 = ss_inter / (ss_inter + ss_resid)
            return {
                "n_obs": int(len(df)),
                "interaction_F": float(interaction_row["F"]),
                "interaction_p": float(interaction_row["PR(>F)"]),
                "interaction_df": (
                    int(interaction_row["df"]),
                    int(tbl.loc["Residual"]["df"]),
                ),
                "interaction_partial_eta2": partial_eta2,
                "protocol_F": float(tbl.loc["C(protocol)"]["F"]),
                "protocol_p": float(tbl.loc["C(protocol)"]["PR(>F)"]),
                "cluster_F": float(tbl.loc["C(cluster)"]["F"]),
                "cluster_p": float(tbl.loc["C(cluster)"]["PR(>F)"]),
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    blocked_anova_per_seed = _fit_blocked(rows_per_seed)
    blocked_anova_seed_avg = _fit_blocked(rows_avg)
    # Primary reported ANOVA is per-seed, matching method.tex specification
    blocked_anova = blocked_anova_per_seed or blocked_anova_seed_avg

    # Per-protocol one-way ANOVA: primary is per-seed to match the main
    # blocked-ANOVA specification; seed-averaged kept alongside for comparison.
    def _per_protocol_oneway(rows_in: list[dict]) -> dict:
        by_proto_cluster: dict = defaultdict(lambda: defaultdict(list))
        for r in rows_in:
            by_proto_cluster[r["protocol"]][r["cluster"]].append(r["psr"])
        out: dict = {}
        for proto, cmap in by_proto_cluster.items():
            groups = [np.array(v) for v in cmap.values() if len(v) > 1]
            if len(groups) >= 2:
                F, p = f_oneway(*groups)
                H, p_kw = kruskal(*groups)
                out[proto] = {
                    "F": float(F),
                    "p": float(p),
                    "H": float(H),
                    "p_kw": float(p_kw),
                    "n_groups": len(groups),
                    "n_obs": int(sum(len(g) for g in groups)),
                }
        return out

    anova_per_protocol = _per_protocol_oneway(rows_per_seed) if rows_per_seed else {}
    anova_per_protocol_seed_avg = _per_protocol_oneway(rows_avg)

    # Str-minus-RF paired delta per cluster
    str_minus_rf_by_cluster = defaultdict(list)
    for name, c_id in cluster_map.items():
        protos = nems_seed_avg.get(name, {})
        s = protos.get("strict", {}).get("psr", None)
        r = protos.get("reset_free", {}).get("psr", None)
        if s is None or r is None or np.isnan(s) or np.isnan(r):
            continue
        str_minus_rf_by_cluster[c_id].append(s - r)

    centroids = scaler.inverse_transform(km.cluster_centers_)
    cluster_centroid = {
        int(i): {k: float(v) for k, v in zip(CLUSTER_FEATURE_KEYS, cen, strict=False)}
        for i, cen in enumerate(centroids)
    }

    return {
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in range(3)},
        "cluster_feature_keys": list(CLUSTER_FEATURE_KEYS),
        "cluster_centroid_features": cluster_centroid,
        "anova_per_protocol": anova_per_protocol,
        "anova_per_protocol_seed_avg": anova_per_protocol_seed_avg,
        "blocked_two_way_anova": blocked_anova,
        "blocked_two_way_anova_per_seed": blocked_anova_per_seed,
        "blocked_two_way_anova_seed_avg": blocked_anova_seed_avg,
        "str_minus_rf_by_cluster": {
            int(c): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v),
            }
            for c, v in str_minus_rf_by_cluster.items()
        },
        "cluster_labels": cluster_map,
    }


def baseline_skill_control(
    nems_seed_avg: dict, xml_dir: Path, panel_manifest: list[dict]
) -> dict:
    """Pre-registered baseline-skill control: residual PSR_RF given PSR_epi + features.

    Spec (nems_task_spec.md): regression PSR_RF ~ PSR_epi + initial_height +
    num_limbs. Reports residual per-body, which we then can regress on
    cluster labels to see whether clusters explain protocol sensitivity
    AFTER removing baseline-skill and size effects.
    """
    panel_by_name = {p["name"]: p for p in panel_manifest}
    rows = []
    for name, protos in nems_seed_avg.items():
        epi = protos.get("episodic", {}).get("psr")
        rf = protos.get("reset_free", {}).get("psr")
        if epi is None or rf is None or np.isnan(epi) or np.isnan(rf):
            continue
        feats = extract_features(xml_dir / f"{name}.xml")
        rows.append(
            {
                "name": name,
                "psr_epi": float(epi),
                "psr_rf": float(rf),
                "initial_height": float(feats["initial_height"]),
                "num_limbs": int(panel_by_name[name]["num_limbs"]),
            }
        )
    if len(rows) < 10:
        return {"error": f"too few bodies ({len(rows)}) for regression"}
    X = np.array([[r["psr_epi"], r["initial_height"], r["num_limbs"]] for r in rows])
    y = np.array([r["psr_rf"] for r in rows])
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    residuals = y - preds
    r2 = float(1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2))

    # Standardized beta coefficients: z-score X and y before OLS so the
    # coefficients are directly comparable across predictors of different units.
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)
    model_std = LinearRegression().fit(Xs, ys)

    return {
        "n_bodies": len(rows),
        "raw_coefficients": {
            "psr_epi": float(model.coef_[0]),
            "initial_height": float(model.coef_[1]),
            "num_limbs": float(model.coef_[2]),
            "intercept": float(model.intercept_),
        },
        "standardized_coefficients": {
            "psr_epi": float(model_std.coef_[0]),
            "initial_height": float(model_std.coef_[1]),
            "num_limbs": float(model_std.coef_[2]),
        },
        "r_squared": r2,
        "residuals_by_body": {
            r["name"]: float(resid) for r, resid in zip(rows, residuals, strict=False)
        },
    }


def distance_transitions(nems_seed_avg: dict, cluster_map: dict) -> dict:
    """Per-cluster walks/stops transition counts at 1m/5m anchors."""
    counts: dict = defaultdict(lambda: defaultdict(int))
    for name, protos in nems_seed_avg.items():
        c = cluster_map.get(name)
        if c is None:
            continue
        d = {p: protos.get(p, {}).get("distance", float("nan")) for p in TAUS}
        if any(np.isnan(v) for v in d.values()):
            continue
        counts[c]["total"] += 1
        counts[c]["epi_walks"] += d["episodic"] >= 5.0
        counts[c]["rf_walks"] += d["reset_free"] >= 5.0
        counts[c]["str_walks"] += d["strict"] >= 5.0
        if d["episodic"] >= 5.0 and d["reset_free"] < 1.0:
            counts[c]["walks_epi_then_stop_rf"] += 1
        if d["episodic"] >= 5.0 and d["strict"] < 1.0:
            counts[c]["walks_epi_then_stop_str"] += 1
        if d["episodic"] < 1.0 and d["reset_free"] >= 5.0:
            counts[c]["stop_epi_then_walks_rf"] += 1
        if d["episodic"] < 1.0 and d["strict"] >= 5.0:
            counts[c]["stop_epi_then_walks_str"] += 1
    return {int(c): dict(v) for c, v in counts.items()}


def plot_primary(primary: dict, cluster: dict, out_path: Path) -> None:
    if "error" in primary:
        print(f"Skip plot (primary error: {primary['error']})")
        return
    rows = primary["per_body"]
    cluster_labels = cluster["cluster_labels"]
    observed = np.array([r["observed_str_minus_rf"] for r in rows])
    predicted = np.array([r["predicted_ols"] for r in rows])
    colors = np.array([cluster_labels.get(r["name"], 0) for r in rows])
    cmap = {0: "#E69F00", 1: "#56B4E9", 2: "#009E73"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for c in (0, 1, 2):
        mask = colors == c
        ax.scatter(
            predicted[mask],
            observed[mask],
            c=cmap[c],
            s=35,
            alpha=0.8,
            label=f"cluster {c} (n={mask.sum()})",
            edgecolors="none",
        )
    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, lw=0.8)
    ax.set_xlabel("predicted Str-minus-RF PSR (corpus OLS)")
    ax.set_ylabel("observed Str-minus-RF PSR (NEMS)")
    pe = primary["primary_endpoint_ols"]
    sp_lo, sp_hi = pe["spearman_ci95"]
    ax.set_title(
        f"Zero-shot transfer: Spearman rho={pe['spearman_rho']:.2f} "
        f"[{sp_lo:.2f}, {sp_hi:.2f}]",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Cluster bar: Str-minus-RF by cluster
    ax2 = axes[1]
    clusters_sorted = sorted(cluster["str_minus_rf_by_cluster"].keys())
    means = [cluster["str_minus_rf_by_cluster"][c]["mean"] for c in clusters_sorted]
    stds = [
        cluster["str_minus_rf_by_cluster"][c]["std"]
        / max(1, np.sqrt(cluster["str_minus_rf_by_cluster"][c]["n"]))
        for c in clusters_sorted
    ]
    ax2.bar(
        [f"C{c}" for c in clusters_sorted],
        means,
        yerr=stds,
        capsize=4,
        color=[cmap[c] for c in clusters_sorted],
    )
    ax2.axhline(0, color="grey", linestyle="--", lw=0.6)
    ax2.set_ylabel("mean Str-minus-RF PSR")
    ax2.set_title("Paired delta by structural cluster")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.2f}, {hi:.2f}]"


def write_report(
    nems_seed_avg: dict,
    primary: dict,
    cluster: dict,
    transitions: dict,
    out_path: Path,
    baseline: dict | None = None,
    nems_panel: dict | None = None,
) -> None:
    lines = []
    lines.append("# NEMS (Non-Evolved Morphology Substrate): Results")
    lines.append("")
    lines.append(
        "Pre-registered analysis per `records/nems_task_spec.md`. "
        "Panel: 60 UNIMAL-init bodies (pre-DERL-selection)."
    )
    lines.append("")

    # Seed coverage: count at the per-seed level, not seed-avg (nanmean would
    # hide partial-NaN bodies that still have valid averages).
    nan_cells: list[tuple[str, str, int]] = []
    bodies_with_nan_cells: set[str] = set()
    if nems_panel is not None:
        for name, protos in nems_panel.items():
            for proto, per_seed in protos.items():
                for seed_id, v in per_seed.items():
                    psr = v.get("psr") if isinstance(v, dict) else None
                    if psr is None or (isinstance(psr, float) and np.isnan(psr)):
                        nan_cells.append((name, proto, int(seed_id)))
                        bodies_with_nan_cells.add(name)
    lines.append(
        f"- Bodies with full 9-cell coverage (all 3 seeds valid): "
        f"{len(nems_seed_avg) - len(bodies_with_nan_cells)}"
    )
    lines.append(
        f"- Bodies with >= 1 NaN protocol-seed cell: {len(bodies_with_nan_cells)} "
        f"(total NaN cells: {len(nan_cells)})"
    )
    if bodies_with_nan_cells:
        lines.append(f"  - bodies: {sorted(bodies_with_nan_cells)}")
    lines.append("")

    # Primary endpoint
    lines.append("## Primary endpoint: zero-shot structural transfer")
    lines.append("")
    if "error" in primary:
        lines.append(f"ERROR: {primary['error']}")
    else:
        pri = primary["primary_endpoint_ols"]
        sp = pri["spearman_rho"]
        sp_lo, sp_hi = pri["spearman_ci95"]
        pe_r = pri["pearson_r"]
        pe_lo, pe_hi = pri["pearson_ci95"]
        lines.append(
            f"- Corpus OLS predictor -> NEMS bodies (all n={primary['n_bodies']}): "
            f"Spearman rho = {sp:.3f} {_fmt_ci(sp_lo, sp_hi)}, "
            f"Pearson r = {pe_r:.3f} {_fmt_ci(pe_lo, pe_hi)}."
        )
        lines.append(
            f"- Positive-transfer threshold (rho > 0.3, 95% CI excludes 0): "
            f"**{'MET' if primary['positive_transfer'] else 'NOT MET'}**."
        )
        lines.append("")
        lines.append("## Sensitivity: alternative corpus models (all NEMS bodies)")
        lines.append("")
        lines.append("| model | Spearman rho (95% CI) | Pearson r (95% CI) |")
        lines.append("| --- | --- | --- |")
        for tag in ("ols", "ridge", "rf"):
            m = primary["metrics_all"][tag]
            sp_lo2, sp_hi2 = m["spearman_ci95"]
            pe_lo2, pe_hi2 = m["pearson_ci95"]
            lines.append(
                f"| {tag} | {m['spearman_rho']:.3f} "
                f"{_fmt_ci(sp_lo2, sp_hi2)} | "
                f"{m['pearson_r']:.3f} {_fmt_ci(pe_lo2, pe_hi2)} |"
            )
        lines.append("")
        ih_lo, ih_hi = primary["corpus_initial_height_range"]
        lines.append(
            f"## Sensitivity: in-distribution subset only (corpus init_z "
            f"range [{ih_lo:.2f}, {ih_hi:.2f}])"
        )
        lines.append("")
        lines.append(
            f"In-distribution NEMS bodies: {primary['n_in_distribution']} / {primary['n_bodies']}"
        )
        lines.append("")
        if primary["metrics_in_distribution"]:
            lines.append("| model | Spearman rho (95% CI) | Pearson r (95% CI) |")
            lines.append("| --- | --- | --- |")
            for tag in ("ols", "ridge", "rf"):
                m = primary["metrics_in_distribution"][tag]
                sp_lo2, sp_hi2 = m["spearman_ci95"]
                pe_lo2, pe_hi2 = m["pearson_ci95"]
                lines.append(
                    f"| {tag} | {m['spearman_rho']:.3f} "
                    f"{_fmt_ci(sp_lo2, sp_hi2)} | "
                    f"{m['pearson_r']:.3f} {_fmt_ci(pe_lo2, pe_hi2)} |"
                )
        else:
            lines.append("(fewer than 10 in-distribution bodies; skipped.)")
        lines.append("")
        loo = primary["corpus_loo_rho"]
        lines.append(
            "## Calibration: within-corpus LOO-CV Spearman rho "
            "(upper bound for transfer)"
        )
        lines.append("")
        lines.append(
            f"OLS={loo['ols']:.3f}, Ridge={loo['ridge']:.3f}, "
            f"RF={loo['rf']:.3f}. A NEMS transfer rho close to this range "
            f"indicates strong structural-feature transfer; materially lower "
            f"indicates a distribution-shift penalty."
        )
    lines.append("")

    # Secondary: clusters
    lines.append("## Secondary: 3-cluster K-means on structural features")
    lines.append("")
    sizes = cluster["cluster_sizes"]
    lines.append(f"- Cluster sizes: {sizes}")
    lines.append("")
    lines.append("Cluster centroids (back-transformed features, pre-registered keys):")
    lines.append("")
    lines.append(
        "| cluster | num_limbs | torso_vertical | init_z | num_joints | tree_depth |"
    )
    lines.append("|---|---|---|---|---|---|")
    for c, feats in sorted(cluster["cluster_centroid_features"].items()):
        lines.append(
            f"| C{c} | {feats['num_limbs']:.1f} | "
            f"{feats['torso_mode_vertical']:.2f} | "
            f"{feats['initial_height']:.2f} | "
            f"{feats['num_joints']:.1f} | "
            f"{feats['tree_depth']:.1f} |"
        )
    lines.append("")

    # Secondary: blocked two-way ANOVA (pre-registered primary secondary)
    lines.append(
        "## Secondary: blocked protocol-by-cluster two-way ANOVA (with body blocking)"
    )
    lines.append("")
    ba = cluster.get("blocked_two_way_anova")
    if ba is None:
        lines.append("Not computed.")
    elif "error" in ba:
        lines.append(f"ERROR: {ba['error']}")
    else:
        df1, df2 = ba["interaction_df"]
        lines.append(
            f"- Interaction F({df1},{df2}) = {ba['interaction_F']:.2f}, "
            f"p = {ba['interaction_p']:.4f}, partial eta^2 = "
            f"{ba['interaction_partial_eta2']:.3f}"
        )
        lines.append(
            f"- Protocol main effect: F = {ba['protocol_F']:.2f}, "
            f"p = {ba['protocol_p']:.4f}"
        )
        lines.append(
            f"- Cluster main effect: F = {ba['cluster_F']:.2f}, "
            f"p = {ba['cluster_p']:.4f}"
        )
    lines.append("")

    # Secondary: per-protocol ANOVA (descriptive)
    lines.append(
        "## Secondary: per-protocol one-way ANOVA across clusters (descriptive)"
    )
    lines.append("")
    lines.append("| protocol | F | p | H (Kruskal) | p_KW |")
    lines.append("|---|---|---|---|---|")
    for p in ("episodic", "strict", "reset_free"):
        a = cluster["anova_per_protocol"].get(p)
        if a is None:
            lines.append(f"| {p} | - | - | - | - |")
        else:
            lines.append(
                f"| {p} | {a['F']:.2f} | {a['p']:.4f} | "
                f"{a['H']:.2f} | {a['p_kw']:.4f} |"
            )
    lines.append("")

    # Secondary: Str-minus-RF by cluster
    lines.append("## Str-minus-RF paired delta by cluster")
    lines.append("")
    lines.append("| cluster | n | mean Str-RF | sd |")
    lines.append("|---|---|---|---|")
    for c in sorted(cluster["str_minus_rf_by_cluster"].keys()):
        d = cluster["str_minus_rf_by_cluster"][c]
        lines.append(f"| C{c} | {d['n']} | {d['mean']:+.3f} | {d['std']:.3f} |")
    lines.append("")

    # Distance transitions
    lines.append("## Distance transitions by cluster (anchors 1m/5m)")
    lines.append("")
    lines.append(
        "| cluster | total | epi_walks | rf_walks | str_walks | "
        "walks_epi->stop_rf | walks_epi->stop_str | "
        "stop_epi->walks_rf | stop_epi->walks_str |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in sorted(transitions.keys()):
        t = transitions[c]
        lines.append(
            f"| C{c} | {t.get('total', 0)} | "
            f"{t.get('epi_walks', 0)} | {t.get('rf_walks', 0)} | "
            f"{t.get('str_walks', 0)} | "
            f"{t.get('walks_epi_then_stop_rf', 0)} | "
            f"{t.get('walks_epi_then_stop_str', 0)} | "
            f"{t.get('stop_epi_then_walks_rf', 0)} | "
            f"{t.get('stop_epi_then_walks_str', 0)} |"
        )
    # Baseline-skill control
    lines.append("## Baseline-skill control")
    lines.append("")
    if baseline is None:
        lines.append("Not computed.")
    elif "error" in baseline:
        lines.append(f"ERROR: {baseline['error']}")
    else:
        raw = baseline.get("raw_coefficients") or baseline.get("coefficients", {})
        std = baseline.get("standardized_coefficients", {})
        lines.append(
            f"Regression `PSR_RF ~ PSR_epi + initial_height + num_limbs` on "
            f"{baseline['n_bodies']} NEMS bodies: R^2 = "
            f"{baseline['r_squared']:.3f}."
        )
        lines.append("")
        lines.append(
            f"- Raw coefficients: Coef(PSR_epi) = {raw.get('psr_epi', float('nan')):.3f}; "
            f"Coef(initial_height) = {raw.get('initial_height', float('nan')):.3f}; "
            f"Coef(num_limbs) = {raw.get('num_limbs', float('nan')):.3f}; "
            f"intercept = {raw.get('intercept', float('nan')):.3f}."
        )
        if std:
            lines.append(
                f"- Standardized betas: PSR_epi = {std['psr_epi']:+.3f}; "
                f"initial_height = {std['initial_height']:+.3f}; "
                f"num_limbs = {std['num_limbs']:+.3f}."
            )
    lines.append("")

    try:
        lines.append(f"Figure: `{OUT_FIG.relative_to(REPO)}`.")
    except ValueError:
        lines.append(f"Figure: `{OUT_FIG}`.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> int:
    nems_panel = load_nems_panel()
    if not nems_panel:
        print("ERROR: no NEMS results loaded. Are results/nems_panel/ populated?")
        return 1

    with PANEL_JSON.open() as f:
        panel_manifest = json.load(f)["panel"]

    nems_seed_avg = seed_average(nems_panel)
    corpus = load_corpus_panel()
    print(
        f"Loaded NEMS panel ({len(nems_seed_avg)} bodies) and corpus "
        f"({len(corpus)} bodies)."
    )

    corpus_fit = fit_corpus_predictors(corpus, CORPUS_XML_DIR)

    primary = primary_analysis(nems_seed_avg, corpus_fit, NEMS_XML_DIR, CORPUS_XML_DIR)
    cluster = cluster_and_anova(
        nems_seed_avg, NEMS_XML_DIR, panel_manifest, nems_panel=nems_panel
    )
    transitions = distance_transitions(nems_seed_avg, cluster["cluster_labels"])
    baseline = baseline_skill_control(nems_seed_avg, NEMS_XML_DIR, panel_manifest)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(
            {
                "primary": primary,
                "cluster": cluster,
                "transitions": transitions,
                "baseline_skill_control": baseline,
            },
            f,
            indent=2,
            sort_keys=True,
            default=lambda x: float(x) if hasattr(x, "item") else str(x),
        )
    print(f"Wrote {OUT_JSON}")

    plot_primary(primary, cluster, OUT_FIG)
    write_report(
        nems_seed_avg,
        primary,
        cluster,
        transitions,
        OUT_MD,
        baseline,
        nems_panel=nems_panel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
