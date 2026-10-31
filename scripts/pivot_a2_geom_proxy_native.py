"""
Phase A.2 (native MuJoCo version).

The JAX/MJX version is too slow on CPU (compilation overhead per morph).
This version uses native MuJoCo C bindings which are fast on CPU without JIT.

Same design: compute three geometric proxies on all 100 morphs, correlate with
R_obs and compute lift over the features-only baseline.

Proxies (per-morph scalars, mean over K=30 disturbed states per morph):
- P1_passive_final: mean (final_z / init_z) after H zero-torque steps from disturbed state
- P2_passive_max: max z / init_z during passive rollout
- P3_random_final: mean (final_z / init_z) under uniform random actions
- P4_random_max: max z under random-action rollout
- P5_settle_success: fraction of disturbed starts that recover to z >= 0.8 * init_z under zero torque

Outputs:
- results/pivot/a2_geom_proxy.json
- records/a2_geom_proxy_report.md
- figures/pivot/fig_a2_proxy_correlations.pdf
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_mjx import load_unimal_xml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
UNIMAL = Path(
    os.environ.get(
        "UNIMAL_XML_DIR",
        str(REPO / "assets/unimals_100/xml"),
    )
)
OUT_JSON = REPO / "results/pivot/a2_geom_proxy.json"
OUT_MD = REPO / "records/a2_geom_proxy_report.md"
OUT_FIG = REPO / "figures/pivot/fig_a2_proxy_correlations.pdf"

K_TRIALS = 30  # disturbances per morph
H_SETTLE = 50  # zero-torque settling to produce disturbed state
H_ROLLOUT = 500  # recovery rollout length
DISTURB_QVEL_SCALE = 3.0
FRAME_SKIP = 4  # match train_mjx.py

FEATURES = [
    "n_bodies",
    "n_joints",
    "n_motors",
    "n_geoms",
    "init_height",
    "tree_depth",
    "total_volume",
    "n_first_level_limbs",
]

TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]
RNG_SEED = 42


def family_of(m: str) -> str:
    for f in ("floor", "vt", "mvt"):
        if m.startswith(f + "-"):
            return f
    raise ValueError(m)


def load_r_obs():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    morphs = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    r_obs = {}
    for m in morphs:
        fr = float(
            np.mean([data[("0p00", s)][m]["eval_tau020"]["fall_rate"] for s in SEEDS])
        )
        r_obs[m] = 1.0 - fr
    return morphs, r_obs


def step_frame_skip(model, data, ctrl):
    """Step with frame_skip to match train_mjx.py semantics."""
    data.ctrl[:] = ctrl
    for _ in range(FRAME_SKIP):
        mujoco.mj_step(model, data)


def compute_proxies_for_morph(xml_path: Path, seed: int) -> dict:
    model = load_unimal_xml(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    init_qpos = data.qpos.copy()
    data.qvel.copy()
    init_z = float(init_qpos[2])
    n_act = int(model.nu)
    nv = int(model.nv)

    rng = np.random.default_rng(seed)

    results = {
        "settle_final_z": np.zeros(K_TRIALS),
        "settle_max_z": np.zeros(K_TRIALS),
        "passive_final_z": np.zeros(K_TRIALS),
        "passive_max_z": np.zeros(K_TRIALS),
        "random_final_z": np.zeros(K_TRIALS),
        "random_max_z": np.zeros(K_TRIALS),
    }

    for k in range(K_TRIALS):
        # Disturbance: inject random velocities then settle briefly under zero torque
        data.qpos[:] = init_qpos
        data.qvel[:] = rng.normal(0.0, DISTURB_QVEL_SCALE, size=nv)
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        # Settle phase (H_SETTLE frame-skipped steps, zero torque)
        settle_max = init_z
        for _ in range(H_SETTLE):
            step_frame_skip(model, data, np.zeros(n_act))
            z = float(data.qpos[2])
            if z > settle_max:
                settle_max = z
        results["settle_max_z"][k] = settle_max
        results["settle_final_z"][k] = float(data.qpos[2])

        # Snapshot the disturbed state after settle
        disturbed_qpos = data.qpos.copy()
        disturbed_qvel = data.qvel.copy()

        # Passive rollout (zero torque, H_ROLLOUT frame-skipped steps)
        data.qpos[:] = disturbed_qpos
        data.qvel[:] = disturbed_qvel
        mujoco.mj_forward(model, data)
        passive_max = float(data.qpos[2])
        for _ in range(H_ROLLOUT):
            step_frame_skip(model, data, np.zeros(n_act))
            z = float(data.qpos[2])
            if z > passive_max:
                passive_max = z
        results["passive_max_z"][k] = passive_max
        results["passive_final_z"][k] = float(data.qpos[2])

        # Random rollout from same disturbed state
        data.qpos[:] = disturbed_qpos
        data.qvel[:] = disturbed_qvel
        mujoco.mj_forward(model, data)
        random_max = float(data.qpos[2])
        for _ in range(H_ROLLOUT):
            ctrl = rng.uniform(-1.0, 1.0, size=n_act)
            step_frame_skip(model, data, ctrl)
            z = float(data.qpos[2])
            if z > random_max:
                random_max = z
        results["random_max_z"][k] = random_max
        results["random_final_z"][k] = float(data.qpos[2])

    proxies = {
        "init_z": init_z,
        "n_act": n_act,
        "P1_passive_final": float(np.mean(results["passive_final_z"] / init_z)),
        "P1_passive_final_std": float(np.std(results["passive_final_z"] / init_z)),
        "P2_passive_max": float(np.mean(results["passive_max_z"] / init_z)),
        "P3_random_final": float(np.mean(results["random_final_z"] / init_z)),
        "P4_random_max": float(np.mean(results["random_max_z"] / init_z)),
        "P5_settle_success": float(np.mean(results["settle_final_z"] > 0.8 * init_z)),
        "disturbed_final_z_mean": float(np.mean(results["settle_final_z"] / init_z)),
    }
    return proxies


def loo_ridge(X, y, alpha=1.0):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = Ridge(alpha=alpha).fit(X_tr, y[train_idx])
        preds[test_idx] = model.predict(X_te)
    return preds


def main() -> int:
    print("=== A.2 Geometric Proxy (native MuJoCo) ===")
    morphs, r_obs = load_r_obs()
    print(f"Loaded R_obs for {len(morphs)} morphs")

    with open(RESULTS / "morph_features.json") as f:
        morph_feats = json.load(f)

    all_proxies = {}
    failed = []
    t0 = time.time()
    for i, m in enumerate(morphs, 1):
        xml_path = UNIMAL / f"{m}.xml"
        if not xml_path.exists():
            failed.append((m, "missing xml"))
            continue
        try:
            t_s = time.time()
            all_proxies[m] = compute_proxies_for_morph(xml_path, seed=RNG_SEED + i)
            t_e = time.time() - t_s
            if i <= 3 or i % 10 == 0:
                p = all_proxies[m]
                print(
                    f"[{i:3d}/{len(morphs)}] {m[:45]:45s} "
                    f"P1={p['P1_passive_final']:.3f} "
                    f"P3={p['P3_random_final']:.3f} "
                    f"{t_e:.2f}s"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(morphs)}] FAILED {m}: {e}")
            traceback.print_exc()
            failed.append((m, str(e)))

    print(f"\nTotal: {time.time() - t0:.1f}s, success={len(all_proxies)}/{len(morphs)}")

    if len(all_proxies) < 50:
        print("Too few successes; aborting.")
        return 1

    ok_morphs = sorted(all_proxies.keys())
    y = np.array([r_obs[m] for m in ok_morphs])
    families = np.array([family_of(m) for m in ok_morphs])
    X_feat = np.array(
        [[morph_feats[m][f] for f in FEATURES] for m in ok_morphs], dtype=float
    )

    proxy_names = [
        "P1_passive_final",
        "P2_passive_max",
        "P3_random_final",
        "P4_random_max",
        "P5_settle_success",
    ]

    print("\n=== Proxy correlations with R_obs ===")
    summary = {}
    for p in proxy_names:
        pvals = np.array([all_proxies[m][p] for m in ok_morphs])
        rho, pval = stats.spearmanr(pvals, y)
        summary[p] = {
            "mean": float(np.mean(pvals)),
            "std": float(np.std(pvals)),
            "rho_overall": float(rho),
            "p_overall": float(pval),
            "rho_by_family": {},
        }
        for fam in ("floor", "vt", "mvt"):
            mask = families == fam
            if mask.sum() >= 3:
                rfam, pfam = stats.spearmanr(pvals[mask], y[mask])
                summary[p]["rho_by_family"][fam] = {
                    "rho": float(rfam),
                    "p": float(pfam),
                    "n": int(mask.sum()),
                }
        print(
            f"  {p}: rho={rho:+.4f} p={pval:.2e} "
            f"[floor={summary[p]['rho_by_family'].get('floor', {}).get('rho', float('nan')):+.3f}, "
            f"vt={summary[p]['rho_by_family'].get('vt', {}).get('rho', float('nan')):+.3f}, "
            f"mvt={summary[p]['rho_by_family'].get('mvt', {}).get('rho', float('nan')):+.3f}]"
        )

    # Features baseline on this morph set
    print("\n=== Features + proxy regressions (LOO R^2) ===")
    best_r2_feat = -np.inf
    best_alpha_feat = 1.0
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0]:
        preds = loo_ridge(X_feat, y, alpha=alpha)
        r2 = r2_score(y, preds)
        if r2 > best_r2_feat:
            best_r2_feat = r2
            best_alpha_feat = alpha
    print(f"Features only: LOO R^2 = {best_r2_feat:.4f} (alpha={best_alpha_feat})")
    summary["baseline"] = {
        "features_only_loo_r2": float(best_r2_feat),
        "alpha": float(best_alpha_feat),
        "n_morphs": int(len(ok_morphs)),
    }

    summary["feature_lift"] = {}
    for p in proxy_names:
        pvals = np.array([all_proxies[m][p] for m in ok_morphs])
        X_aug = np.column_stack([X_feat, pvals])
        best_r2 = -np.inf
        for alpha in [0.1, 1.0, 3.0, 10.0, 30.0]:
            r2 = r2_score(y, loo_ridge(X_aug, y, alpha=alpha))
            if r2 > best_r2:
                best_r2 = r2
        lift = best_r2 - best_r2_feat
        summary["feature_lift"][p] = {
            "features_plus_proxy_loo_r2": float(best_r2),
            "lift_over_baseline": float(lift),
        }
        print(f"  features + {p}: LOO R^2 = {best_r2:.4f} (lift {lift:+.4f})")

    X_all = np.column_stack(
        [X_feat]
        + [np.array([all_proxies[m][p] for m in ok_morphs]) for p in proxy_names]
    )
    best_r2_all = -np.inf
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0]:
        r2 = r2_score(y, loo_ridge(X_all, y, alpha=alpha))
        if r2 > best_r2_all:
            best_r2_all = r2
    summary["features_plus_all_proxies"] = {
        "loo_r2": float(best_r2_all),
        "lift": float(best_r2_all - best_r2_feat),
    }
    print(
        f"features + all proxies: LOO R^2 = {best_r2_all:.4f} (lift {best_r2_all - best_r2_feat:+.4f})"
    )

    X_proxies_only = np.column_stack(
        [np.array([all_proxies[m][p] for m in ok_morphs]) for p in proxy_names]
    )
    best_r2_proxy = -np.inf
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0]:
        r2 = r2_score(y, loo_ridge(X_proxies_only, y, alpha=alpha))
        if r2 > best_r2_proxy:
            best_r2_proxy = r2
    summary["proxies_only"] = {"loo_r2": float(best_r2_proxy)}
    print(f"proxies-only: LOO R^2 = {best_r2_proxy:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "n_successful": len(ok_morphs),
                "n_failed": len(failed),
                "failed": failed,
                "per_morph_proxies": all_proxies,
                "summary": summary,
                "config": {
                    "K_TRIALS": K_TRIALS,
                    "H_SETTLE": H_SETTLE,
                    "H_ROLLOUT": H_ROLLOUT,
                    "DISTURB_QVEL_SCALE": DISTURB_QVEL_SCALE,
                    "FRAME_SKIP": FRAME_SKIP,
                    "RNG_SEED": RNG_SEED,
                    "backend": "mujoco-native",
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"\nWrote {OUT_JSON}")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = {"floor": "#e41a1c", "vt": "#377eb8", "mvt": "#4daf4a"}
    for ax, p in zip(axes.flat, proxy_names):
        pvals = np.array([all_proxies[m][p] for m in ok_morphs])
        for fam in ("floor", "vt", "mvt"):
            mask = families == fam
            ax.scatter(
                pvals[mask],
                y[mask],
                c=colors[fam],
                label=fam,
                alpha=0.7,
                s=30,
                edgecolors="black",
                linewidths=0.3,
            )
        ax.set_xlabel(p)
        ax.set_ylabel("R_obs")
        rho = summary[p]["rho_overall"]
        lift = summary["feature_lift"][p]["lift_over_baseline"]
        ax.set_title(f"rho={rho:+.3f}  lift={lift:+.3f}")
        ax.grid(True, alpha=0.3)
    last = axes.flat[-1]
    lifts = [summary["feature_lift"][p]["lift_over_baseline"] for p in proxy_names]
    last.barh(proxy_names, lifts)
    last.axvline(0, color="black", linewidth=0.5)
    last.set_xlabel(f"Lift over features (baseline = {best_r2_feat:.3f})")
    last.grid(True, axis="x", alpha=0.3)
    axes.flat[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")

    # Report
    lines = []
    lines.append("# A.2 Geometric Proxy Report (native MuJoCo)")
    lines.append("")
    lines.append(f"Successful morphs: {len(ok_morphs)} / {len(morphs)}")
    lines.append("Target: R_obs = 1 - fall_rate_rf (eval_tau020, seed-averaged)")
    lines.append(f"Features-only LOO R^2 baseline: {best_r2_feat:.4f}")
    lines.append("")
    lines.append("## Proxy correlations (Spearman rho with R_obs)")
    lines.append("")
    lines.append("| proxy | mean | rho | p | lift over features |")
    lines.append("|---|---|---|---|---|")
    for p in proxy_names:
        s = summary[p]
        lift = summary["feature_lift"][p]["lift_over_baseline"]
        lines.append(
            f"| {p} | {s['mean']:.3f} | {s['rho_overall']:+.4f} | "
            f"{s['p_overall']:.2e} | {lift:+.4f} |"
        )
    lines.append("")
    lines.append("## Combined regressions")
    lines.append("")
    lines.append(f"- Features only: LOO R^2 = {best_r2_feat:.4f}")
    lines.append(
        f"- Features + all proxies: LOO R^2 = {best_r2_all:.4f} "
        f"(lift {best_r2_all - best_r2_feat:+.4f})"
    )
    lines.append(f"- Proxies only: LOO R^2 = {best_r2_proxy:.4f}")
    lines.append("")
    lines.append("## Interpretation")
    best_single_lift = max(
        summary["feature_lift"][p]["lift_over_baseline"] for p in proxy_names
    )
    if best_r2_all - best_r2_feat > 0.05:
        lines.append(
            f"- **Proxies add information**: total lift = {best_r2_all - best_r2_feat:+.3f} "
            "(exploratory)."
        )
    elif best_single_lift > 0.02:
        lines.append(
            f"- **Marginal lift**: best single proxy = {best_single_lift:+.3f}. "
            "The lift is small."
        )
    else:
        lines.append(
            "- **No lift**: geometric proxies as defined do not add information "
            "beyond structural features. Features already capture what these proxies measure. "
            "No added predictive value."
        )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
