#!/usr/bin/env python3
"""Recompute every Standard-based paper statistic on body subsets.

Standard's fall clause (0.20 z0) cannot fire for bodies whose threshold lies
below the torso radius (scripts/audit_standard_threshold.py). This script
recomputes the paper's Standard-based numbers on three body sets:

  all   the 100-body corpus, as in the paper;
  g72   bodies where 0.20 z0 >= torso radius (the clause can fire);
  e67   g72 minus bodies whose recorded tau_eval=0.20 evaluations never
        differ from neutral evaluation for any of 15 policies.

Reads records/standard_threshold_audit.json for the sets.
Writes records/standard_threshold_subset_stats.json.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
PROTOS = ("std", "strict", "rf")
TAU = {"std": "0p20", "strict": "0p50", "rf": "0p00"}
SWEEP = ("0p00", "0p10", "0p20", "0p35", "0p50")
FAMS = ("floor", "vt", "mvt")
FEATS = ("init_height", "n_first_level_limbs", "tree_depth", "n_joints", "n_bodies")
KS = (5, 10, 20)
B_BOOT = 10_000
B_PERM = 5_000


def fam(b):
    return b.split("-")[0]


def load(
    results: Path, tt: str, key: str = "eval_tau020", metric: str = "per_step_reward"
):
    out = defaultdict(dict)
    for s in SEEDS:
        with open(results / f"mjx_ppo_{tt}_seed{s}" / "results.json") as f:
            for b, r in json.load(f).items():
                out[b][s] = float(r[key][metric])
    return out


def sa(ps, bodies):
    return np.array([np.mean([ps[b][s] for s in SEEDS]) for b in bodies])


# ---------------------------------------------------------------- blocked ANOVA
class BlockedAnova:
    """Type II protocol:family F with morphology fixed effects (fast refits)."""

    def __init__(self, y, morph, prot):
        n = len(y)
        nm = morph.max() + 1
        M = np.zeros((n, nm))
        M[np.arange(n), morph] = 1
        P = np.stack([(prot == 1), (prot == 2)], 1).astype(float)
        Q, _ = np.linalg.qr(np.hstack([M, P]))
        self.Q, self.prot, self.morph = Q, prot, morph
        self.y_res = y - Q @ (Q.T @ y)
        self.rss_red = float(self.y_res @ self.y_res)
        self.df_res = n - (nm + 2 + 4)

    def F(self, fam_of_morph):
        f = fam_of_morph[self.morph]
        cols = [
            ((self.prot == p) & (f == g)).astype(float) for p in (1, 2) for g in (1, 2)
        ]
        Ir = np.stack(cols, 1)
        Ir = Ir - self.Q @ (self.Q.T @ Ir)
        beta, *_ = np.linalg.lstsq(Ir, self.y_res, rcond=None)
        fit = Ir @ beta
        ss_int = float(fit @ fit)
        rss_full = self.rss_red - ss_int
        Fv = (ss_int / 4) / (rss_full / self.df_res)
        return Fv, ss_int / (ss_int + rss_full)


def blocked(df, value="psr"):
    bodies = sorted(df.morph.unique())
    mi = {b: i for i, b in enumerate(bodies)}
    morph = df.morph.map(mi).to_numpy()
    prot = df.protocol.map({"std": 0, "strict": 1, "rf": 2}).to_numpy()
    fom = np.array([FAMS.index(fam(b)) for b in bodies])
    ba = BlockedAnova(df[value].to_numpy(float), morph, prot)
    Fv, pe2 = ba.F(fom)
    p = float(stats.f.sf(Fv, 4, ba.df_res))
    return ba, fom, Fv, p, pe2


def perm_blocked(ba, fom, Fobs, rng, B=B_PERM):
    null = np.empty(B)
    for i in range(B):
        null[i] = ba.F(rng.permutation(fom))[0]
    return {
        "F_obs": Fobs,
        "null_p95": float(np.percentile(null, 95)),
        "p_perm": float((1 + np.sum(null >= Fobs)) / (B + 1)),
    }


# ---------------------------------------------------------------- per-set stats
def compute(bodies, D, feats, rng_seed=42):
    out = {
        "n": len(bodies),
        "n_by_family": {f: sum(fam(b) == f for b in bodies) for f in FAMS},
    }
    rng = np.random.default_rng(rng_seed)
    fa = np.array([fam(b) for b in bodies])
    psr = {p: sa(D["psr"][p], bodies) for p in PROTOS}
    rf_std = psr["rf"] / psr["std"]
    str_std = psr["strict"] / psr["std"]

    # Table 1
    t1 = {}
    for f in FAMS:
        m = fa == f
        t1[f] = {
            "n": int(m.sum()),
            "std": float(psr["std"][m].mean()),
            "strict": float(psr["strict"][m].mean()),
            "rf": float(psr["rf"][m].mean()),
            "rf_std": float(rf_std[m].mean()),
            "str_std": float(str_std[m].mean()),
            "rf_loss_pct": float(100 * (1 - rf_std[m].mean())),
            "str_loss_pct": float(100 * (1 - str_std[m].mean())),
        }
    out["table1"] = t1

    # One-way ANOVA on RF/Std (+ permutation)
    groups = [rf_std[fa == f] for f in FAMS]
    F1, p1 = stats.f_oneway(*groups)
    H1, pk = stats.kruskal(*groups)
    null = np.empty(10_000)
    for i in range(10_000):
        fp = rng.permutation(fa)
        null[i] = stats.f_oneway(*[rf_std[fp == f] for f in FAMS])[0]
    out["oneway_rf_std"] = {
        "F": float(F1),
        "p": float(p1),
        "H": float(H1),
        "p_kw": float(pk),
        "perm_p": float((1 + np.sum(null >= F1)) / 10_001),
        "perm_null_p95": float(np.percentile(null, 95)),
    }
    # Str/Std one-way too (strict-side contrast)
    F2, p2 = stats.f_oneway(*[str_std[fa == f] for f in FAMS])
    out["oneway_str_std"] = {"F": float(F2), "p": float(p2)}

    # Per-seed
    per_seed = {}
    for s in SEEDS:
        r = np.array([D["psr"]["rf"][b][s] / D["psr"]["std"][b][s] for b in bodies])
        g = [r[fa == f] for f in FAMS]
        Fs, ps = stats.f_oneway(*g)
        per_seed[s] = {
            "F": float(Fs),
            "p": float(ps),
            **{f: float(x.mean()) for f, x in zip(FAMS, g)},
        }
    out["per_seed_rf_std"] = per_seed

    # Two-way blocked ANOVAs
    rows = [
        {
            "psr": D["psr"][p][b][s],
            "protocol": p,
            "family": fam(b),
            "morph": b,
            "seed": s,
        }
        for p in PROTOS
        for b in bodies
        for s in SEEDS
    ]
    df = pd.DataFrame(rows)
    ba, fom, Fv, pv, pe2 = blocked(df)
    out["twoway_perseed"] = {
        "F": Fv,
        "df": [4, ba.df_res],
        "p": pv,
        "partial_eta2": pe2,
    }
    out["twoway_perseed"]["perm"] = perm_blocked(ba, fom, Fv, rng)
    dfr = df.copy()
    dfr["psr"] = dfr.groupby(["protocol", "seed"])["psr"].rank()
    _, _, Fr, pr, _ = blocked(dfr)
    out["twoway_rank_perseed"] = {"F": Fr, "p": pr}
    dfa = df.groupby(["protocol", "family", "morph"], as_index=False)["psr"].mean()
    ba_a, _, Fa, pa, pe2a = blocked(dfa)
    out["twoway_seedavg"] = {
        "F": Fa,
        "df": [4, ba_a.df_res],
        "p": pa,
        "partial_eta2": pe2a,
    }

    # Cluster-robust Wald, GEE, MixedLM
    from statsmodels.formula.api import mixedlm, ols
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.generalized_estimating_equations import GEE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcr = ols("psr ~ C(protocol)*C(family) + C(morph)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df.morph}
        )
        ic = [c for c in mcr.params.index if ":" in c]
        R = np.zeros((len(ic), len(mcr.params)))
        for i, c in enumerate(ic):
            R[i, list(mcr.params.index).index(c)] = 1
        w = mcr.wald_test(R, use_f=True, scalar=True)
        gee = GEE.from_formula(
            "psr ~ C(protocol)*C(family)",
            groups="morph",
            data=df,
            cov_struct=Exchangeable(),
        ).fit()
        icg = [c for c in gee.params.index if ":" in c]
        Rg = np.zeros((len(icg), len(gee.params)))
        for i, c in enumerate(icg):
            Rg[i, list(gee.params.index).index(c)] = 1
        wg = gee.wald_test(Rg, use_f=False, scalar=True)
        mm = mixedlm(
            "psr ~ C(protocol, Treatment(reference='std')) * C(family, Treatment(reference='floor'))",
            data=df,
            groups=df.morph,
        ).fit(reml=True)
        ci = mm.conf_int()
        key_vt = [k for k in mm.params.index if "T.rf" in k and "T.vt" in k][0]
        key_rf = [k for k in mm.params.index if "T.rf" in k and ":" not in k][0]
    out["cluster_wald"] = {
        "F": float(w.statistic),
        "df": [int(w.df_num), int(w.df_denom)],
        "p": float(w.pvalue),
    }
    out["gee"] = {"chi2": float(wg.statistic), "p": float(wg.pvalue)}
    out["mixedlm"] = {
        "rf_main": [float(mm.params[key_rf]), *map(float, ci.loc[key_rf])],
        "vt_x_rf": [float(mm.params[key_vt]), *map(float, ci.loc[key_vt])],
    }

    # Leave-one-family-out blocked ANOVAs (2-family: 2 interaction dfs)
    lofo = {}
    for drop in FAMS:
        keep = [f for f in FAMS if f != drop]
        d2 = df[df.family.isin(keep)]
        bs = sorted(d2.morph.unique())
        mi = {b: i for i, b in enumerate(bs)}
        y = d2.psr.to_numpy(float)
        morph = d2.morph.map(mi).to_numpy()
        prot = d2.protocol.map({"std": 0, "strict": 1, "rf": 2}).to_numpy()
        f1 = np.array([fam(b) == keep[1] for b in bs])[morph]
        n = len(y)
        M = np.zeros((n, len(bs)))
        M[np.arange(n), morph] = 1
        P = np.stack([(prot == 1), (prot == 2)], 1).astype(float)
        Q, _ = np.linalg.qr(np.hstack([M, P]))
        yr = y - Q @ (Q.T @ y)
        ic = np.stack([((prot == p) & f1).astype(float) for p in (1, 2)], 1)
        ic = ic - Q @ (Q.T @ ic)
        beta, *_ = np.linalg.lstsq(ic, yr, rcond=None)
        ssi = float((ic @ beta) @ (ic @ beta))
        dfr_ = n - (len(bs) + 2 + 2)
        Fl = (ssi / 2) / ((yr @ yr - ssi) / dfr_)
        lofo[f"drop_{drop}"] = {
            "F": float(Fl),
            "df": [2, dfr_],
            "p": float(stats.f.sf(Fl, 2, dfr_)),
        }
    out["lofo"] = lofo

    def boot_diff(x, y):
        bx = rng.choice(x, size=(B_BOOT, len(x)))
        by = rng.choice(y, size=(B_BOOT, len(y)))
        dd = bx.mean(1) - by.mean(1)
        return [float(x.mean() - y.mean()), *map(float, np.percentile(dd, [2.5, 97.5]))]

    out["boot_floor_minus_vt_rf_std"] = boot_diff(
        rf_std[fa == "floor"], rf_std[fa == "vt"]
    )
    out["boot_vt_minus_mvt_str_std"] = boot_diff(
        str_std[fa == "vt"], str_std[fa == "mvt"]
    )
    out["boot_vt_minus_floor_str_std"] = boot_diff(
        str_std[fa == "vt"], str_std[fa == "floor"]
    )

    # Balanced-family subsampling
    nmin = min(out["n_by_family"].values())
    wins = 0
    for _ in range(1000):
        fl = rng.choice(rf_std[fa == "floor"], nmin, replace=False).mean()
        vt = rng.choice(rf_std[fa == "vt"], nmin, replace=False).mean()
        wins += fl < vt
    out["balanced_floor_lt_vt"] = {"n_per_family": nmin, "frac": wins / 1000}

    # Spearman + top-10
    idx = np.random.default_rng(42).integers(0, len(bodies), size=(B_BOOT, len(bodies)))
    sp = {}
    for a, b in (("std", "rf"), ("std", "strict"), ("rf", "strict")):
        rho = stats.spearmanr(psr[a], psr[b])[0]
        boots = np.array(
            [stats.spearmanr(psr[a][ii], psr[b][ii])[0] for ii in idx[:2000]]
        )
        top = len(set(np.argsort(-psr[a])[:10]) & set(np.argsort(-psr[b])[:10]))
        sp[f"{a}_vs_{b}"] = {
            "rho": float(rho),
            "ci_B2000": list(map(float, np.percentile(boots, [2.5, 97.5]))),
            "top10": top,
        }
    out["spearman"] = sp

    # Selection retention
    methods = {
        "std_only": psr["std"],
        "strict_only": psr["strict"],
        "rf_only": psr["rf"],
        "mean3": (psr["std"] + psr["strict"] + psr["rf"]) / 3,
        "min3": np.minimum(np.minimum(psr["std"], psr["strict"]), psr["rf"]),
        "min2_strict_rf": np.minimum(psr["strict"], psr["rf"]),
    }
    sel = {}
    for mn, sc in methods.items():
        sel[mn] = {}
        for on, tg in (
            ("std", psr["std"]),
            ("strict", psr["strict"]),
            ("rf", psr["rf"]),
        ):
            for k in KS:
                sel[mn][f"{on}_k{k}"] = float(
                    tg[np.argsort(-sc)[:k]].mean() / np.sort(tg)[::-1][:k].mean()
                )
        sel[mn]["worst3"] = {
            k: min(sel[mn][f"{o}_k{k}"] for o in ("std", "strict", "rf")) for k in KS
        }
        sel[mn]["worst_strict_rf"] = {
            k: min(sel[mn][f"{o}_k{k}"] for o in ("strict", "rf")) for k in KS
        }
    # bootstrap lower bounds for min3 and min2 against each oracle
    lb = {}
    for mn in ("min3", "min2_strict_rf", "std_only"):
        for on in ("std", "strict", "rf"):
            for k in KS:
                bt = np.empty(B_BOOT)
                for i in range(B_BOOT):
                    ii = idx[i]
                    s_ = {
                        "min3": np.minimum(
                            np.minimum(psr["std"][ii], psr["strict"][ii]), psr["rf"][ii]
                        ),
                        "min2_strict_rf": np.minimum(psr["strict"][ii], psr["rf"][ii]),
                        "std_only": psr["std"][ii],
                    }[mn]
                    t_ = psr[on][ii]
                    bt[i] = (
                        t_[np.argsort(-s_)[:k]].mean() / np.sort(t_)[::-1][:k].mean()
                    )
                lb[f"{mn}_{on}_k{k}"] = list(map(float, np.percentile(bt, [2.5, 97.5])))
    sel["boot_ci"] = lb
    sel["top10_family_split"] = {
        mn: {f: int(sum(fa[i] == f for i in np.argsort(-sc)[:10])) for f in FAMS}
        for mn, sc in methods.items()
    }
    out["selection"] = sel

    # Distance transitions, neutral ratios, fitness ratios
    for lab, key in (("tau020", "eval_tau020"), ("neutral", "eval_neutral")):
        dist = {p: sa(D[f"dist_{lab}"][p], bodies) for p in PROTOS}
        tr = {}
        for f in FAMS:
            m = fa == f
            tr[f] = {
                "std_walk_rf_stop": int(
                    np.sum(m & (dist["std"] >= 5) & (dist["rf"] < 1))
                ),
                "std_walk_str_stop": int(
                    np.sum(m & (dist["std"] >= 5) & (dist["strict"] < 1))
                ),
                "std_stop_rf_walk": int(
                    np.sum(m & (dist["std"] < 1) & (dist["rf"] >= 5))
                ),
                "std_stop_str_walk": int(
                    np.sum(m & (dist["std"] < 1) & (dist["strict"] >= 5))
                ),
            }
        out[f"distance_transitions_{lab}"] = tr
        if lab == "tau020":
            dr = dist["rf"] / dist["std"]
            Fd, pd_ = stats.f_oneway(*[dr[(fa == f) & (dist["std"] > 0)] for f in FAMS])
            out["distance_rf_std_anova"] = {"F": float(Fd), "p": float(pd_)}
    pn = {p: sa(D["psr_neutral"][p], bodies) for p in PROTOS}
    out["neutral_ratios"] = {
        f: {
            "rf_std": float((pn["rf"] / pn["std"])[fa == f].mean()),
            "str_std": float((pn["strict"] / pn["std"])[fa == f].mean()),
        }
        for f in FAMS
    }
    fit = {p: sa(D["fitness"][p], bodies) for p in PROTOS}
    fr = fit["rf"] / fit["std"]
    Ff, pf = stats.f_oneway(*[fr[fa == f] for f in FAMS])
    out["fitness_ratios"] = {f: float(fr[fa == f].mean()) for f in FAMS}
    out["fitness_ratios"]["anova"] = {"F": float(Ff), "p": float(pf)}

    # Features: Spearman with RF/Std and Str/Std
    X = {k: np.array([feats[b][k] for b in bodies], float) for k in FEATS}
    out["feature_spearman"] = {
        k: {
            "rf_std": list(map(float, stats.spearmanr(X[k], rf_std))),
            "str_std": list(map(float, stats.spearmanr(X[k], str_std))),
        }
        for k in FEATS
    }

    # Residual-Delta two-feature regressions and M1/M2/M3 on Delta_str_rf
    def ols_r2(Xm, y):
        Xm = np.column_stack([Xm, np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        res = y - Xm @ beta
        return 1 - res @ res / np.sum((y - y.mean()) ** 2), res, Xm @ beta

    d_reset = psr["std"] - psr["rf"]
    d_str_rf = psr["strict"] - psr["rf"]
    resid = {}
    for lab, y, cols in (
        ("dreset_z0_nbodies-1", d_reset, [X["init_height"], X["n_bodies"] - 1]),
        (
            "dstrrf_z0_firstlimbs",
            d_str_rf,
            [X["init_height"], X["n_first_level_limbs"]],
        ),
    ):
        r2, res, _ = ols_r2(np.column_stack(cols), y)
        g = [res[fa == f] for f in FAMS]
        Fr_, pr_ = stats.f_oneway(*g)
        resid[lab] = {
            "R2": float(r2),
            **{f: float(x.mean()) for f, x in zip(FAMS, g)},
            "F": float(Fr_),
            "p": float(pr_),
        }
    out["residual_delta"] = resid
    m2 = np.column_stack([X[k] for k in FEATS])
    mods = {}
    for lab, Xm in (
        ("M1", psr["std"][:, None]),
        ("M2", m2),
        ("M3", np.column_stack([psr["std"], m2])),
    ):
        r2, _, pred = ols_r2(Xm, d_str_rf)
        mods[lab] = {"R2": float(r2), "rho": float(stats.spearmanr(pred, d_str_rf)[0])}
    out["structural_models_dstrrf"] = mods

    # Threshold sweep family means
    out["sweep_family_means"] = {
        tt: {f: float(sa(D["sweep"][tt], bodies)[fa == f].mean()) for f in FAMS}
        for tt in SWEEP
    }

    # VT specialists (fig_top1_vt) and gallery picks (Delta_reset percentiles)
    vt = fa == "vt"
    spec = {}
    for p in PROTOS:
        others = [q for q in PROTOS if q != p]
        sc = psr[p] - np.maximum(psr[others[0]], psr[others[1]])
        i = np.where(vt)[0][np.argmax(sc[vt])]
        spec[p] = {"body": bodies[i], "spec": float(sc[i])}
    out["vt_specialists"] = spec
    gal = {}
    for f in FAMS:
        m = np.where(fa == f)[0]
        dsub = d_reset[m]
        gal[f] = {}
        for q in (25, 50, 75):
            target = np.percentile(dsub, q)
            j = m[np.argmin(np.abs(dsub - target))]
            gal[f][f"P{q}"] = {"body": bodies[j], "delta_reset": float(d_reset[j])}
    out["gallery_picks"] = gal
    out["ranks"] = {
        p: {bodies[i]: r + 1 for r, i in enumerate(np.argsort(-psr[p]))} for p in PROTOS
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=os.environ.get("RFC_RESULTS", REPO / "results")
    )
    ap.add_argument(
        "--audit", default=REPO / "records/standard_threshold_audit.json"
    )
    ap.add_argument(
        "--out", default=REPO / "records/standard_threshold_subset_stats.json"
    )
    args = ap.parse_args()
    results = Path(args.results)
    audit = json.load(open(args.audit))["bodies"]
    feats = json.load(open(results / "morph_features.json"))

    D = {
        "psr": {p: load(results, TAU[p]) for p in PROTOS},
        "psr_neutral": {p: load(results, TAU[p], "eval_neutral") for p in PROTOS},
        "fitness": {p: load(results, TAU[p], metric="fitness") for p in PROTOS},
        "dist_tau020": {p: load(results, TAU[p], metric="distance") for p in PROTOS},
        "dist_neutral": {
            p: load(results, TAU[p], "eval_neutral", "distance") for p in PROTOS
        },
        "sweep": {tt: load(results, tt) for tt in SWEEP},
    }
    allb = sorted(D["psr"]["std"])
    g72 = [b for b in allb if not audit[b]["geometric_inert"]["0.20"]]
    e67 = [b for b in g72 if not audit[b]["eval_tau020_all_identical_to_neutral"]]
    inert = [b for b in allb if audit[b]["geometric_inert"]["0.20"]]

    res = {"sets": {"all": allb, "g72": g72, "e67": e67, "inert28": inert}}
    for name, bs in (("all", allb), ("g72", g72), ("e67", e67)):
        print(f"computing {name} (n={len(bs)})", flush=True)
        res[name] = compute(bs, D, feats)

    # Inert bodies: Standard vs Reset-free are one condition; show run-to-run noise.
    std = {b: [D["psr"]["std"][b][s] for s in SEEDS] for b in inert}
    rf = {b: [D["psr"]["rf"][b][s] for s in SEEDS] for b in inert}
    t10 = {b: [D["sweep"]["0p10"][b][s] for s in SEEDS] for b in inert}
    ratio = np.array([np.mean(rf[b]) / np.mean(std[b]) for b in inert])
    same_seed = np.array([abs(std[b][i] - rf[b][i]) for b in inert for i in range(3)])
    cross_seed = np.array(
        [
            abs(std[b][i] - std[b][j])
            for b in inert
            for i in range(3)
            for j in range(3)
            if i < j
        ]
    )
    res["inert28"] = {
        "rf_std_mean": float(ratio.mean()),
        "rf_std_sd": float(ratio.std(ddof=1)),
        "rf_std_min": float(ratio.min()),
        "rf_std_max": float(ratio.max()),
        "by_family": {
            f: float(ratio[[fam(b) == f for b in inert]].mean()) for f in FAMS
        },
        "mean_abs_std_minus_rf_same_seed": float(same_seed.mean()),
        "mean_abs_std_between_seeds": float(cross_seed.mean()),
        "corr_std_rf_perseed": float(
            np.corrcoef(
                [v for b in inert for v in std[b]], [v for b in inert for v in rf[b]]
            )[0, 1]
        ),
        "tau010_over_std": float(
            np.mean([np.mean(t10[b]) / np.mean(std[b]) for b in inert])
        ),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
