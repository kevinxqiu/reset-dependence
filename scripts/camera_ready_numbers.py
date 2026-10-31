#!/usr/bin/env python3
"""Recompute every number printed in the camera-ready version of the paper.

Paper: "Reset Dependence Is Embodied" (NeurIPS 2026, Evaluations and Datasets
track). One seeded CPU script. It reads the raw run records and a few
geometry and probe records, recomputes each statistic, and writes

  records/camera_ready_numbers.json   all values at full precision
  records/camera_ready_numbers.md     one row per number

Inputs (all read-only):
  --results    run records, default results/ (from the dataset)
               (mjx_ppo_{0p00,0p10,0p20,0p35,0p50}_seed{42,43,44}, sac_mjx_*,
               boundary_free, boundary_sweep, equivalence_pilot, nems_panel,
               asymptotic_5m, morph_features.json)
  --audit      records/standard_threshold_audit.json (z0, torso radius,
               geometric inert flags, probe minima, factorial training dones)
  --passive    zero-action rollouts of all 100 bodies (all100_passive.json)
  --settle     3 s zero-action settle heights (settle.json)
  --probe-dir  training-like torso probes behind the 60 probe-safe bodies
               (train_like_probe_all_20.json, train_like_probe_near_100.json,
               placebo_safe60_bodies.json)
  --shoot, --search  random open-loop probe records (sk_shoot.json,
               vt10_search.json), feasibility probes only
  --xml-dir    the 100 UNIMAL XMLs (duplicate hashes and, unless --no-physics,
               a MuJoCo re-run of the zero-action rollouts)
  --gait-traces  joint-angle and action traces of the nine illustrative retrained
               controllers (forward displacement and early ends of their rollouts)
  --logs       the kept training logs (logs.zip of the dataset, read in place,
               or the folder it was unzipped into, by default the --results
               folder): runs whose log reaches the end of training
  --review-panels  per-cell tables, specifications and reports of the
               review-period panels (panels.zip of the dataset, read in place,
               or the review-panels/ folder it unzips into, the default): run
               counts, designs, wall times, rank correlations and seed-half
               references

The --passive, --settle, --probe-dir, --shoot and --search records were made
on 2026-10-05 and ship in the dataset under sim-records/ (the defaults below). A
number whose input record is missing is skipped with a log line and shows as
"not computed" in the comparison table; nothing falls back silently. With
--xml-dir present, the MuJoCo re-run recomputes the zero-action counts and
steps without the records.

Rules are named Reset-free (tau = 0), Standard (tau = 0.20; the keys write it
"len", after a draft name) and Strict (tau = 0.50), always in ascending tau order.
The common evaluator is tau_eval = 0.20 (eval_tau020), the neutral evaluator
has no fall clause (eval_neutral) and the Strict evaluator is tau_eval = 0.50
(eval_tau050). PSR is per-step reward (episode return over episode length).

Conventions (fixed for every number):
  * Bodies are in sorted name order. Rankings use three-seed mean PSR.
  * Spearman rho uses average ranks for ties. Top-k sets use a stable sort.
  * Body bootstrap: B = 10,000 resamples with replacement, seeds fixed. The
    resample matrix of a body set is numpy default_rng(42).integers(0, n,
    size=(B, n)), shared by every statistic on that set (paired). 95%
    percentile intervals. Intervals are conditional on the three seeds.
  * Permutation tests: B = 10,000, a fresh numpy default_rng(20260923) per
    test, p = (1 + #{null >= observed}) / (B + 1). Two-sided tests compare
    absolute values. Family labels are permuted across whole bodies.
  * Retention is the mean over bodies of the ratio of three-seed means.
  * Post hoc flag: "yes" for analyses added after review (every body subset,
    D, the exchangeability null, the Spearman-Brown reference, top-20, the
    rule-pair split, the cut-off-run analysis, the same-rule repeat,
    held-out-seed selection and the other checks marked so), "no (reviewed
    version)" for analyses reported in the reviewed version without a prior
    decision rule, and "no (planned)" for tests whose decision rule was fixed
    in advance.

Usage:
  python scripts/camera_ready_numbers.py [--results DIR] [--boot 10000]
      [--perm 10000] [--no-physics]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]


def _repo_relative(path) -> str:
    """Record an input path relative to the repository when it lies inside it, so outputs do not carry local paths."""
    p = Path(path).expanduser().resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)
DATA = REPO  # unpack the dataset (runs.zip) at the repository root
# Simulation records made on 2026-10-05 (zero-action rollouts, settle heights, probes);
# they ship in the dataset under sim-records/.
SIM = DATA / "sim-records"

SEEDS = (42, 43, 44)
RULES = ("rf", "len", "str")  # ascending tau
TAU_DIR = {"rf": "0p00", "len": "0p20", "str": "0p50"}
RULE_NAME = {"rf": "Reset-free", "len": "Standard", "str": "Strict"}
SWEEP = ("0p00", "0p10", "0p20", "0p35", "0p50")
PAIRS = (("rf", "len"), ("len", "str"), ("rf", "str"))  # table order
FAMS = ("floor", "vt", "mvt")
FAM_NAME = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}
EV_COMMON, EV_NEUTRAL, EV_STRICT = "eval_tau020", "eval_neutral", "eval_tau050"
EV_LABEL = {
    EV_COMMON: "common (tau_eval = 0.20)",
    EV_NEUTRAL: "neutral (no fall clause)",
    EV_STRICT: "Strict evaluator (tau_eval = 0.50)",
}
BOOT_SEED = 42
PERM_SEED = 20260923
TORSO_R = 0.1
KS = (5, 10, 20)

PH_YES = "yes"
PH_REV = "no (reviewed version)"
PH_PLAN = "no (planned)"
PH_NA = "n/a (record count)"

SET_LABEL = {
    "all100": "100",
    "g72": "72 (Standard cutoff at or above the torso radius)",
    "s89": "89 (100 without the 11 cut-off bodies)",
    "s77": "77 (100 without the 23 cut-off-seed bodies)",
    "s68": "68 (72 without its 4 cut-off bodies)",
    "s65": "65 (72 without its 7 cut-off-seed bodies)",
    "no_vt10": "90 (100 without the 10 VT cut-off bodies)",
    "inert28": "28 (Standard cutoff below the torso radius)",
    "inert10_94": "94 (tau = 0.10 cutoff below the torso radius)",
    "probe_safe67": "67 (94 minus bodies the audit probes pushed below 0.10 z0)",
    "probe_safe60": "60 probe-safe (67 minus bodies the training-like probes pushed below 0.10 z0)",
    "clean59": "59 (94 and Standard can act and no cut-off seed)",
}
SUBSETS_MAIN = ("all100", "g72", "s89", "s77", "s68", "s65")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------- registry
def _py(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    return v


class Registry:
    """Every printed number, with the metadata the markdown table needs."""

    def __init__(self):
        self.rows = []
        self.index = {}
        self.section = ""

    def add(self, key, value, *, ci=None, body_set="", evaluator="", definition="", post_hoc="", dec=3, note="", kind="num"):
        if key in self.index:
            raise KeyError(f"duplicate key {key}")
        row = dict(
            key=key,
            value=_py(value),
            ci=_py(ci) if ci is not None else None,
            body_set=body_set,
            evaluator=evaluator,
            definition=definition,
            post_hoc=post_hoc,
            dec=dec,
            note=note,
            kind=kind,
            section=self.section,
        )
        self.index[key] = len(self.rows)
        self.rows.append(row)
        return value

    def get(self, key):
        return self.rows[self.index[key]]


def fmt_num(v, dec=3, kind="num"):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if kind == "p":
            if v < 1e-3:
                return f"{v:.1e}"
            return f"{v:.4f}"
        return f"{v:.{dec}f}"
    if isinstance(v, (list, tuple)):
        if len(v) > 12 and all(isinstance(x, str) for x in v):
            return f"{len(v)} ids (see JSON)"
        return "[" + ", ".join(fmt_num(x, dec, kind) for x in v) + "]"
    if isinstance(v, dict):
        return "; ".join(f"{k}: {fmt_num(x, dec, kind)}" for k, x in v.items())
    return str(v)


# ----------------------------------------------------------------- data
class Data:
    def __init__(self, results: Path, audit_path: Path):
        self.results = results
        self.raw = {}
        for t in SWEEP:
            for s in SEEDS:
                p = results / f"mjx_ppo_{t}_seed{s}" / "results.json"
                with open(p) as f:
                    d = json.load(f)
                tau = float(t.replace("p", "."))
                for b, r in d.items():
                    if r["seed"] != s or abs(r["threshold"] - tau) > 1e-9 or r["timesteps"] != 5_000_000:
                        raise ValueError(f"record label mismatch in {p}: {b}")
                self.raw[(t, s)] = d
        self.bodies = sorted(self.raw[("0p20", 42)])
        for k, d in self.raw.items():
            if sorted(d) != self.bodies:
                raise ValueError(f"body set differs in {k}")
        self.n = len(self.bodies)
        if self.n != 100:
            raise ValueError(f"expected 100 bodies, found {self.n}")
        self.fam = np.array([b.split("-")[0] for b in self.bodies])
        with open(audit_path) as f:
            self.audit_full = json.load(f)
        self.audit = self.audit_full["bodies"]
        self.z0 = np.array([float(self.audit[b]["z0"]) for b in self.bodies])
        with open(results / "morph_features.json") as f:
            self.feats = json.load(f)
        z0f = np.array([float(self.feats[b]["init_height"]) for b in self.bodies])
        if np.max(np.abs(z0f - self.z0)) > 1e-9:
            raise ValueError("audit z0 differs from morph_features init_height")
        radii = {float(self.audit[b]["torso_radius"]) for b in self.bodies}
        if radii != {TORSO_R}:
            raise ValueError(f"torso radii are not all {TORSO_R}: {radii}")

    def get(self, t, ev=EV_COMMON, metric="per_step_reward"):
        return np.array([[self.raw[(t, s)][b][ev][metric] for s in SEEDS] for b in self.bodies], float)

    def V(self, ev=EV_COMMON, metric="per_step_reward"):
        """(n, 3 rules in RULES order, 3 seeds)."""
        return np.stack([self.get(TAU_DIR[r], ev, metric) for r in RULES], axis=1)


# ----------------------------------------------------------------- helpers
def rank_z(x, axis):
    r = stats.rankdata(x, axis=axis)
    r = r - r.mean(axis=axis, keepdims=True)
    nrm = np.sqrt((r**2).sum(axis=axis, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / nrm


def spearman_last(a, b):
    za = rank_z(a, -1)
    zb = rank_z(b, -1)
    return (za * zb).sum(-1)


def rho(a, b):
    return float(stats.spearmanr(a, b).correlation)


def topk_set(x, k):
    return set(np.argsort(-np.asarray(x), kind="stable")[:k].tolist())


def topk_overlap(a, b, k):
    return len(topk_set(a, k) & topk_set(b, k))


def topk_mask_rows(M, k):
    idx = np.argsort(-M, axis=-1, kind="stable")[..., :k]
    out = np.zeros(M.shape, bool)
    np.put_along_axis(out, idx, True, axis=-1)
    return out


def boot_index(n, B):
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(B, n))


def pct_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def perm_rng():
    return np.random.default_rng(PERM_SEED)


def pval(null, obs):
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))


def pval_two(null, obs):
    null = np.asarray(null)
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (len(null) + 1))


# ----------------------------------------------------------------- rank statistics
def ranking_components(V, idx=None, chunk=500):
    """Rank statistics on (n, 3, 3) arrays, optionally over bootstrap draws.

    Returns a dict of 1-d arrays (one entry per draw): rho3 for every rule pair
    (three-seed means), W per rule (mean single-seed rho over the 3 seed pairs),
    C per pair (mean over the 9 cross-rule single-seed pairs) and C_aligned
    (same-seed pairs).
    """
    if idx is None:
        idx = np.arange(V.shape[0])[None, :]
    acc = collections.defaultdict(list)
    for c0 in range(0, len(idx), chunk):
        v = V[idx[c0 : c0 + chunk]]  # d x n x 3 x 3
        d, n = v.shape[:2]
        z = rank_z(v.reshape(d, n, 9), axis=1)
        G = np.einsum("dni,dnj->dij", z, z)
        zm = rank_z(v.mean(-1), axis=1)
        Gm = np.einsum("dni,dnj->dij", zm, zm)
        for i, r in enumerate(RULES):
            acc[f"W_{r}"].append(np.mean([G[:, 3 * i + s, 3 * i + t] for s, t in itertools.combinations(range(3), 2)], axis=0))
        for a, b in PAIRS:
            ia, ib = RULES.index(a), RULES.index(b)
            acc[f"rho3_{a}_{b}"].append(Gm[:, ia, ib])
            acc[f"C_{a}_{b}"].append(np.mean([G[:, 3 * ia + s, 3 * ib + t] for s in range(3) for t in range(3)], axis=0))
            acc[f"Cal_{a}_{b}"].append(np.mean([G[:, 3 * ia + s, 3 * ib + s] for s in range(3)], axis=0))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    for a, b in PAIRS:
        out[f"D_{a}_{b}"] = (out[f"W_{a}"] + out[f"W_{b}"]) / 2 - out[f"C_{a}_{b}"]
        out[f"Dal_{a}_{b}"] = (out[f"W_{a}"] + out[f"W_{b}"]) / 2 - out[f"Cal_{a}_{b}"]
    for r in RULES:
        w = out[f"W_{r}"]
        out[f"R3_{r}"] = 3 * w / (1 + 2 * w)
    for a, b in PAIRS:
        ref = np.sqrt(out[f"R3_{a}"] * out[f"R3_{b}"])
        out[f"ref_{a}_{b}"] = ref
        out[f"gap_{a}_{b}"] = ref - out[f"rho3_{a}_{b}"]
        out[f"dis_{a}_{b}"] = out[f"rho3_{a}_{b}"] / ref
    return out


def exch_null(V, a, b, B, ks=(5, 10, 20), chunk=1000):
    """Rule-exchangeability null: within each body the 6 runs of rules a and b
    are split at random into two pseudo-rules of three runs (raw PSR)."""
    ia, ib = RULES.index(a), RULES.index(b)
    cols = np.concatenate([V[:, ia, :], V[:, ib, :]], axis=1)
    n = len(cols)
    obs_a, obs_b = V[:, ia, :].mean(1), V[:, ib, :].mean(1)
    obs = dict(rho=rho(obs_a, obs_b), **{f"top{k}": topk_overlap(obs_a, obs_b, k) for k in ks})
    rng = perm_rng()
    rhos, ovs = [], {k: [] for k in ks}
    for c0 in range(0, B, chunk):
        m = min(chunk, B - c0)
        perm = np.argsort(rng.random((m, n, 6)), axis=2)
        sh = np.take_along_axis(np.broadcast_to(cols, (m, n, 6)), perm, axis=2)
        pa, pb = sh[:, :, :3].mean(2), sh[:, :, 3:].mean(2)
        rhos.append(spearman_last(pa, pb))
        for k in ks:
            ovs[k].append((topk_mask_rows(pa, k) & topk_mask_rows(pb, k)).sum(1))
    rhos = np.concatenate(rhos)
    ovs = {k: np.concatenate(v) for k, v in ovs.items()}
    res = dict(
        obs=obs,
        null_rho_mean=float(rhos.mean()),
        null_rho_ci=pct_ci(rhos),
        p_rho=float((1 + np.sum(rhos <= obs["rho"])) / (B + 1)),
    )
    for k in ks:
        res[f"null_top{k}_mean"] = float(ovs[k].mean())
        res[f"p_top{k}"] = float((1 + np.sum(ovs[k] <= obs[f"top{k}"])) / (B + 1))
        res[f"frac_top{k}_le_obs"] = float(np.mean(ovs[k] <= obs[f"top{k}"]))
    return res


# ----------------------------------------------------------------- interaction tests
class BlockedF:
    """Closed-form rule x family F for a balanced body-blocked design.

    v has shape (n bodies, P rules, S seeds). The model has body and rule
    effects, the test adds rule x family cells. Equal to the least-squares
    nested F on the explicit design (checked in check_blocked_f)."""

    def __init__(self, v):
        self.v = v
        n, P, S = v.shape
        self.n, self.P, self.S = n, P, S
        m = v.mean(axis=2)
        self.w = m - m.mean(axis=1, keepdims=True)
        self.gbar = self.w.mean(axis=0)
        self.sse_red = float(np.sum((v - v.mean(axis=(1, 2), keepdims=True) - self.gbar[None, :, None]) ** 2))

    def F(self, lab):
        groups = np.unique(lab)
        ss = 0.0
        for g in groups:
            sel = lab == g
            ss += sel.sum() * np.sum((self.w[sel].mean(axis=0) - self.gbar) ** 2)
        ss *= self.S
        q = (self.P - 1) * (len(groups) - 1)
        df = self.n * self.P * self.S - self.n - (self.P - 1) - q
        return (ss / q) / ((self.sse_red - ss) / df), q, df

    def F_batch(self, labs, groups):
        """labs: (m, n) integer codes in range(len(groups))."""
        S = self.S
        ss = np.zeros(len(labs))
        for gi in range(len(groups)):
            M = (labs == gi).astype(float)  # m x n
            cnt = M.sum(1)
            mu = (M @ self.w) / cnt[:, None]
            ss += cnt * ((mu - self.gbar) ** 2).sum(1)
        ss *= S
        q = (self.P - 1) * (len(groups) - 1)
        df = self.n * self.P * S - self.n - (self.P - 1) - q
        return (ss / q) / ((self.sse_red - ss) / df)


def design_matrix(fam, P, S):
    n = len(fam)
    present = [f for f in FAMS if (fam == f).any()]
    rows, groups = [], []
    for i in range(n):
        for p in range(P):
            body = np.zeros(n)
            body[i] = 1
            rule = np.array([float(p == q) for q in range(1, P)])
            inter = np.array([float(p == q and fam[i] == f) for f in present[1:] for q in range(1, P)])
            r = np.concatenate([body, rule, inter])
            for _ in range(S):
                rows.append(r)
                groups.append(i)
    q = (P - 1) * (len(present) - 1)
    return np.array(rows), np.array(groups), q


def lstsq_F(v, fam):
    n, P, S = v.shape
    X, _, q = design_matrix(fam, P, S)
    y = v.reshape(-1)
    full = np.linalg.lstsq(X, y, rcond=None)[0]
    sse_f = float(np.sum((y - X @ full) ** 2))
    Xr = X[:, :-q]
    red = np.linalg.lstsq(Xr, y, rcond=None)[0]
    sse_r = float(np.sum((y - Xr @ red) ** 2))
    df = len(y) - X.shape[1]
    F = ((sse_r - sse_f) / q) / (sse_f / df)
    return F, q, df


def cr1_wald(v, fam):
    import statsmodels.api as sm

    n, P, S = v.shape
    X, g, q = design_matrix(fam, P, S)
    y = v.reshape(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(y, X).fit()
        rob = fit.get_robustcov_results(cov_type="cluster", groups=g, use_correction=True, df_correction=True)
        t = rob.f_test(np.eye(X.shape[1])[-q:])
    return float(np.squeeze(t.fvalue)), int(q), float(t.df_denom), float(t.pvalue)


def interaction_tests(v, fam, B, want_cr1=True):
    """Per-seed blocked F, seed-mean F, CR1 body-cluster Wald, label permutation."""
    bf = BlockedF(v)
    F, q, df = bf.F(fam)
    out = dict(perseed_F=F, perseed_df=[q, df], perseed_p=float(stats.f.sf(F, q, df)))
    bm = BlockedF(v.mean(axis=2, keepdims=True))
    Fm, qm, dfm = bm.F(fam)
    out.update(seedmean_F=Fm, seedmean_df=[qm, dfm], seedmean_p=float(stats.f.sf(Fm, qm, dfm)))
    if want_cr1:
        Fc, qc, dfc, pc = cr1_wald(v, fam)
        out.update(cr1_F=Fc, cr1_df=[qc, dfc], cr1_p=pc)
    groups = [f for f in FAMS if (fam == f).any()]
    codes = np.array([groups.index(f) for f in fam])
    rng = perm_rng()
    labs = np.stack([rng.permutation(codes) for _ in range(B)])
    null = np.concatenate([bf.F_batch(labs[i : i + 2000], groups) for i in range(0, B, 2000)])
    out["perm_p"] = pval(null, F)
    out["perm_null_p95"] = float(np.percentile(null, 95))
    return out


def check_blocked_f(v, fam):
    F1 = BlockedF(v).F(fam)[0]
    F2 = lstsq_F(v, fam)[0]
    if abs(F1 - F2) > 1e-8 * max(1.0, F1):
        raise AssertionError(f"closed-form F {F1} differs from lstsq F {F2}")


def contrast_perm(x, fam, f1, f2, B):
    keep = np.isin(fam, [f1, f2])
    xs, lab = x[keep], fam[keep]
    obs = xs[lab == f1].mean() - xs[lab == f2].mean()
    rng = perm_rng()
    null = np.empty(B)
    for i in range(B):
        pl = rng.permutation(lab)
        null[i] = xs[pl == f1].mean() - xs[pl == f2].mean()
    return float(obs), pval_two(null, obs)


def oneway_perm(x, fam, B):
    groups = [f for f in FAMS if (fam == f).any()]
    F, p = stats.f_oneway(*[x[fam == f] for f in groups])
    rng = perm_rng()
    null = np.empty(B)
    for i in range(B):
        pl = rng.permutation(fam)
        null[i] = stats.f_oneway(*[x[pl == f] for f in groups]).statistic
    return float(F), float(p), pval(null, F)


def signflip(d, B):
    rng = perm_rng()
    s = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (s * d[None, :]).mean(1)
    return float(d.mean()), pval_two(null, d.mean())


# ----------------------------------------------------------------- selection
def retention(score, target, k):
    sel = np.argsort(-score, kind="stable")[:k]
    return float(target[sel].mean() / np.sort(target)[::-1][:k].mean())


def selectors(M):
    """M: n x 3 seed means in RULES order (rf, len, str)."""
    return {
        "len_only": M[:, 1],
        "str_only": M[:, 2],
        "rf_only": M[:, 0],
        "mean3": M.mean(axis=1),
        "min3": M.min(axis=1),
        "min_str_rf": np.minimum(M[:, 2], M[:, 0]),
    }


def retention_batch(scores, target, k):
    """scores, target: (d, n). Mean target of the top-k by score over the
    oracle top-k mean."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    sel = np.take_along_axis(target, order, axis=1).mean(1)
    orc = -np.sort(-target, axis=1)[:, :k].mean(1)
    return sel / orc


SEL5 = ("rf", "len", "str", "mean", "min")


def heldout_batch(Vb, k):
    """Select on two seeds, score on the third. Vb: (d, n, 3 rules, 3 seeds).
    Returns fold-averaged retention (d, 5 selectors, 3 targets)."""
    rets = []
    for held in range(3):
        tr = Vb[..., [s for s in range(3) if s != held]].mean(axis=-1)  # d x n x 3
        te = Vb[..., held]
        sc = np.concatenate([tr, tr.mean(axis=-1, keepdims=True), tr.min(axis=-1, keepdims=True)], axis=-1)
        order = np.argsort(-sc, axis=1, kind="stable")[:, :k, :]
        sel = np.empty((Vb.shape[0], 5, 3))
        for m in range(5):
            g = np.take_along_axis(te, np.repeat(order[:, :, m][:, :, None], 3, axis=2), axis=1)
            sel[:, m, :] = g.mean(axis=1)
        orc = np.sort(te, axis=1)[:, -k:, :].mean(axis=1)
        rets.append(sel / orc[:, None, :])
    return np.mean(rets, axis=0)


def rank_within(x):
    # ties (bootstrap duplicates, equal ranks) are broken by body order so the
    # result does not depend on the numpy build's default sort
    o = np.argsort(x, kind="stable")
    r = np.empty_like(o)
    r[o] = np.arange(1, len(x) + 1)
    return r


def lopo_rules(scores, train):
    rules = {}
    for p in train:
        rules[f"by_{p}"] = scores[p]
    st = np.stack([scores[p] for p in train])
    rules["RobustMin"] = st.min(axis=0)
    rules["RobustMean"] = st.mean(axis=0)
    rk = np.stack([rank_within(scores[p]) for p in train])
    rules["Borda"] = rk.sum(axis=0)
    rules["MinimaxRank"] = rk.min(axis=0)
    return rules


def topk_mean(score, target, k):
    return float(np.mean(target[np.argsort(-score, kind="stable")[:k]]))


# ----------------------------------------------------------------- physics (optional)
FLOOR_COMMENT = '<!-- <geom name="floor" type="plane" pos="0 0 0" size="50 50 1" material="grid"/> -->'
FLOOR_ACTIVE = '<geom name="floor" type="plane" pos="0 0 0" size="50 50 1" contype="1" conaffinity="1" rgba="0.9 0.9 0.9 1"/>'


def zero_action_physics(xml_dir: Path, bodies, steps=150, frame_skip=4):
    """Nominal reset (no noise), zero control, train_mjx floor substitution.
    Records torso height per control step, first step below 0.5 z0 and first
    floor contact (deterministic)."""
    import mujoco

    out = {}
    for b in bodies:
        s = (xml_dir / f"{b}.xml").read_text()
        if FLOOR_COMMENT not in s:
            raise ValueError(f"floor comment not found in {b}")
        m = mujoco.MjModel.from_xml_string(s.replace(FLOOR_COMMENT, FLOOR_ACTIVE))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso/0")
        z0 = float(d.qpos[2])
        zs = [z0]
        cross = contact = None
        for t in range(1, steps + 1):
            d.ctrl[:] = 0.0
            mujoco.mj_step(m, d, nstep=frame_skip)
            z = float(d.qpos[2])
            zs.append(z)
            if contact is None:
                for i in range(d.ncon):
                    c = d.contact[i]
                    if (c.geom1 == floor or c.geom2 == floor) and c.dist <= 0:
                        contact = t
                        break
            if cross is None and z < 0.5 * z0:
                cross = t
        out[b] = dict(
            z0=z0,
            nu=int(m.nu),
            torso_radius=float(m.geom_size[tid][0]),
            z50=zs[50],
            z75=zs[75],
            z150=zs[150],
            cross=cross,
            contact=contact,
            dt_control=float(m.opt.timestep * frame_skip),
        )
    return out


# ----------------------------------------------------------------- sections
def section_sets(D: Data, R: Registry, args):
    """Body sets, counts and geometry. Returns dict of boolean masks."""
    R.section = "Body sets, counts and threshold geometry"
    n, fam, z0 = D.n, D.fam, D.z0
    A = D.audit
    inert20 = np.array([A[b]["geometric_inert"]["0.20"] for b in D.bodies])
    inert10 = np.array([A[b]["geometric_inert"]["0.10"] for b in D.bodies])
    inert35 = np.array([A[b]["geometric_inert"]["0.35"] for b in D.bodies])
    inert50 = np.array([A[b]["geometric_inert"]["0.50"] for b in D.bodies])
    if not np.array_equal(inert20, 0.20 * z0 < TORSO_R - 1e-12):
        raise AssertionError("audit inert flags at 0.20 do not match 0.20 z0 < 0.1 m")
    if not np.array_equal(inert10, 0.10 * z0 < TORSO_R - 1e-12):
        raise AssertionError("audit inert flags at 0.10 do not match 0.10 z0 < 0.1 m")
    Lstr = D.get("0p50", EV_STRICT, "mean_ep_len")
    cut11 = Lstr.mean(1) <= 20
    cut23 = (Lstr <= 20).any(1)
    vt10 = cut11 & (fam == "vt")
    probe_min = np.array([A[b]["probe_min_torso_z"]["min"] for b in D.bodies])
    probe_safe67 = inert10 & (probe_min >= 0.10 * z0)
    sets = {
        "all100": np.ones(n, bool),
        "g72": ~inert20,
        "s89": ~cut11,
        "s77": ~cut23,
        "s68": ~inert20 & ~cut11,
        "s65": ~inert20 & ~cut23,
        "no_vt10": ~vt10,
        "inert28": inert20,
        "inert10_94": inert10,
        "probe_safe67": probe_safe67,
        "clean59": inert10 & ~inert20 & ~cut23,
    }
    extra = dict(cut11=cut11, cut23=cut23, vt10=vt10, inert20=inert20, inert10=inert10, Lstr=Lstr)

    # probe-safe 60 (training-like probes from the placebo mechanism check)
    pdir = Path(args.probe_dir)
    f20, f100 = pdir / "train_like_probe_all_20.json", pdir / "train_like_probe_near_100.json"
    if f20.exists() and f100.exists():
        PR = json.loads(f20.read_text())
        NR = json.loads(f100.read_text())

        def reach(Rec, b):
            return b in Rec and (Rec[b]["gauss"]["eps_below"] > 0 or Rec[b]["bang"]["eps_below"] > 0)

        tl = np.array([reach(PR, b) for b in D.bodies])
        near = np.array([reach(NR, b) for b in D.bodies])
        safe60 = probe_safe67 & ~tl & ~near
        sets["probe_safe60"] = safe60
        frozen = pdir / "placebo_safe60_bodies.json"
        if frozen.exists():
            fl = set(json.loads(frozen.read_text()))
            mine = {b for b, k in zip(D.bodies, safe60) if k}
            extra["safe60_matches_frozen_list"] = mine == fl
        extra["probe_reach"] = dict(
            audit=int((inert10 & ~probe_safe67).sum()),
            training_like=int((inert10 & tl).sum()),
            near_100=int((inert10 & near).sum()),
            union=int((inert10 & ~safe60).sum()),
        )
    else:
        log(f"probe records not found in {pdir}; the 60 probe-safe set is not computed")

    def famsplit(mask):
        return {f: int((mask & (fam == f)).sum()) for f in FAMS}

    defs = {
        "all100": "All 100 DERL-evolved UNIMAL bodies.",
        "g72": "Bodies whose Standard cutoff 0.20 z0 is at or above the 0.1 m torso radius (geometry-defined).",
        "s89": "Without the 11 cut-off bodies (Strict-trained seed-mean episode length under the Strict evaluator at most 20 steps; outcome-defined).",
        "s77": "Without the 23 cut-off-seed bodies (any Strict-trained seed with mean Strict-evaluator episode length at most 20 steps; outcome-defined).",
        "s68": "The 72 without the cut-off bodies among them.",
        "s65": "The 72 without the cut-off-seed bodies among them.",
        "no_vt10": "Without the 10 VT bodies among the 11 cut-off bodies.",
        "inert28": "Bodies whose Standard cutoff 0.20 z0 lies below the 0.1 m torso radius.",
        "inert10_94": "Bodies whose tau = 0.10 cutoff 0.10 z0 lies below the 0.1 m torso radius.",
        "probe_safe67": "The 94 minus bodies whose audit probe minimum torso height fell below 0.10 z0.",
        "probe_safe60": "The 67 minus bodies that 20-episode or 100-episode training-like probes (random and bang-bang actions) pushed below 0.10 z0.",
        "clean59": "The 94 that are also in the 72 and have no cut-off seed.",
    }
    for name, m in sets.items():
        ph = PH_REV if name == "all100" else PH_YES
        R.add(f"sets.{name}.n", int(m.sum()), body_set=SET_LABEL[name], definition=defs[name], post_hoc=ph)
        for f in FAMS:
            R.add(
                f"sets.{name}.{f}",
                famsplit(m)[f],
                body_set=SET_LABEL[name],
                definition=f"{FAM_NAME[f]} bodies in the set.",
                post_hoc=ph,
            )
        R.add(
            f"sets.{name}.ids",
            [b for b, k in zip(D.bodies, m) if k],
            body_set=SET_LABEL[name],
            definition="Body IDs in the set.",
            post_hoc=ph,
            kind="ids",
        )
    if "safe60_matches_frozen_list" in extra:
        R.add(
            "sets.probe_safe60.matches_frozen_list",
            extra["safe60_matches_frozen_list"],
            body_set=SET_LABEL["probe_safe60"],
            post_hoc=PH_YES,
            definition="Reconstructed 60 equals placebo_safe60_bodies.json from the placebo mechanism check.",
        )
        for k, v in extra["probe_reach"].items():
            R.add(
                f"sets.probe_reach.{k}",
                v,
                body_set=SET_LABEL["inert10_94"],
                post_hoc=PH_YES,
                definition=f"Bodies of the 94 whose torso reached 0.10 z0 in the {k.replace('_', ' ')} probe(s).",
            )

    # geometry counts
    R.add(
        "geom.inert20.z0_030",
        int((inert20 & np.isclose(z0, 0.30)).sum()),
        body_set=SET_LABEL["inert28"],
        definition="Standard-inert bodies with reset height z0 = 0.30 m.",
        post_hoc=PH_YES,
    )
    R.add(
        "geom.inert20.z0_045",
        int((inert20 & np.isclose(z0, 0.45)).sum()),
        body_set=SET_LABEL["inert28"],
        definition="Standard-inert bodies with reset height z0 = 0.45 m.",
        post_hoc=PH_YES,
    )
    R.add(
        "geom.inert20.z0_values",
        sorted(set(np.round(z0[inert20], 3).tolist())),
        body_set=SET_LABEL["inert28"],
        definition="Distinct reset heights among the Standard-inert bodies (m).",
        post_hoc=PH_YES,
        dec=2,
    )
    R.add(
        "geom.g72.min_z0",
        float(z0[~inert20].min()),
        body_set=SET_LABEL["g72"],
        dec=2,
        definition="Smallest reset height among the 72 (m).",
        post_hoc=PH_YES,
    )
    R.add(
        "geom.inert35.n",
        int(inert35.sum()),
        body_set="100",
        post_hoc=PH_YES,
        definition="Bodies whose tau = 0.35 cutoff lies below the torso radius.",
    )
    R.add(
        "geom.inert50.n",
        int(inert50.sum()),
        body_set="100",
        post_hoc=PH_YES,
        definition="Bodies whose tau = 0.50 cutoff lies below the torso radius.",
    )
    R.add(
        "geom.tau035.min_margin_m",
        float((0.35 * z0 - TORSO_R).min()),
        body_set="100",
        dec=4,
        post_hoc=PH_YES,
        definition="Smallest margin of the tau = 0.35 cutoff above the torso radius (m), at z0 = 0.30 m.",
    )
    R.add(
        "geom.lenient_penetration_m",
        float(TORSO_R - 0.20 * 0.30),
        body_set="25 bodies at z0 = 0.30 m",
        dec=3,
        post_hoc=PH_YES,
        definition="Floor penetration a 0.1 m torso needs to reach the 0.06 m Standard cutoff at z0 = 0.30 m.",
    )
    R.add(
        "geom.z0_range",
        [float(z0.min()), float(z0.max())],
        body_set="100",
        dec=2,
        post_hoc=PH_REV,
        definition="Range of reset heights z0 (m).",
    )
    R.add(
        "geom.lenient_cutoff_range_m",
        [float(0.2 * z0.min()), float(0.2 * z0.max())],
        body_set="100",
        dec=3,
        post_hoc=PH_REV,
        definition="Range of Standard cutoffs 0.20 z0 (m).",
    )
    R.add(
        "geom.strict_cutoff_range_m",
        [float(0.5 * z0.min()), float(0.5 * z0.max())],
        body_set="100",
        dec=3,
        post_hoc=PH_REV,
        definition="Range of Strict cutoffs 0.50 z0 (m).",
    )
    derl = 0.5 * (z0 - 0.2)
    frac = derl / z0
    R.add(
        "geom.derl_fraction_range",
        [float(frac.min()), float(frac.max())],
        body_set="100",
        dec=3,
        post_hoc=PH_YES,
        definition="DERL's cutoff 0.5 (z0 minus 0.2 m) as a fraction of our z0 (min, max).",
    )
    R.add(
        "geom.derl_closer_to_lenient",
        int((np.abs(frac - 0.2) < np.abs(frac - 0.5)).sum()),
        body_set="100",
        post_hoc=PH_YES,
        definition="Bodies whose DERL cutoff fraction is closer to 0.20 than to 0.50.",
    )
    R.add(
        "geom.derl_below_radius",
        int((derl < TORSO_R).sum()),
        body_set="100",
        post_hoc=PH_YES,
        definition="Bodies whose DERL cutoff lies below the torso radius (cannot act without penetration).",
    )

    # cut-off sets
    R.add(
        "cut.cut11_inert",
        int((cut11 & inert20).sum()),
        body_set="11",
        post_hoc=PH_YES,
        definition="Cut-off bodies that are also Standard-inert.",
    )
    R.add(
        "cut.cut23_inert",
        int((cut23 & inert20).sum()),
        body_set="23",
        post_hoc=PH_YES,
        definition="Cut-off-seed bodies that are also Standard-inert.",
    )
    R.add(
        "cut.cut11_ids",
        [b for b, k in zip(D.bodies, cut11) if k],
        body_set="11",
        post_hoc=PH_YES,
        kind="ids",
        definition="The 11 cut-off bodies.",
    )
    R.add(
        "cut.cut23_ids",
        [b for b, k in zip(D.bodies, cut23) if k],
        body_set="23",
        post_hoc=PH_YES,
        kind="ids",
        definition="The 23 cut-off-seed bodies.",
    )
    R.add(
        "cut.vt10_ids",
        [b for b, k in zip(D.bodies, vt10) if k],
        body_set="10",
        post_hoc=PH_YES,
        kind="ids",
        definition="The 10 VT cut-off bodies.",
    )
    R.add(
        "cut.vt10.z0_030",
        int((vt10 & np.isclose(z0, 0.30)).sum()),
        body_set="10",
        post_hoc=PH_YES,
        definition="VT cut-off bodies with z0 = 0.30 m.",
    )
    R.add(
        "cut.vt10.taller",
        int((vt10 & ~np.isclose(z0, 0.30)).sum()),
        body_set="10",
        post_hoc=PH_YES,
        definition="VT cut-off bodies with z0 above 0.30 m.",
    )
    R.add(
        "cut.vt10.taller_z0",
        sorted(np.round(z0[vt10 & ~np.isclose(z0, 0.30)], 2).tolist()),
        body_set="10",
        dec=2,
        post_hoc=PH_YES,
        definition="Reset heights of the taller VT cut-off bodies (m).",
    )
    Lall = {t: D.get(t, EV_STRICT, "mean_ep_len") for t in SWEEP}
    corp = np.stack([Lall[TAU_DIR[r]] for r in RULES], axis=1)  # n x 3 x 3
    R.add(
        "cut.vt10.max_strict_eval_len_any_rule",
        float(corp[vt10].max()),
        body_set="10",
        dec=2,
        evaluator=EV_LABEL[EV_STRICT],
        post_hoc=PH_YES,
        definition="Largest seed-level mean episode length under the Strict evaluator over all 9 corpus PPO controllers of each VT cut-off body.",
    )
    R.add(
        "cut.vt10.max_strict_eval_len_rf_len",
        float(corp[vt10][:, [0, 1], :].max()),
        body_set="10",
        dec=2,
        evaluator=EV_LABEL[EV_STRICT],
        post_hoc=PH_YES,
        definition="Same, over the Reset-free- and Standard-trained controllers only.",
    )
    sw = np.stack([Lall[t] for t in SWEEP], axis=1)
    R.add(
        "cut.vt10.max_strict_eval_len_sweep",
        float(sw[vt10].max()),
        body_set="10",
        dec=2,
        evaluator=EV_LABEL[EV_STRICT],
        post_hoc=PH_YES,
        definition="Same, over all 15 controllers of the five-threshold sweep (tau = 0, 0.10, 0.20, 0.35, 0.50).",
    )
    floor_cut = [b for b, k in zip(D.bodies, cut11 & (fam == "floor")) if k]
    R.add(
        "cut.floor_cutoff_body",
        floor_cut,
        body_set="11",
        post_hoc=PH_YES,
        kind="ids",
        definition="The Floor body among the 11 cut-off bodies.",
    )

    # cut-off runs by family
    S = D.get("0p50")
    Lm = D.get("0p20").mean(1)
    cutrun = Lstr <= 20
    for f in FAMS:
        sel = fam == f
        rs = S[sel] / Lm[sel, None]
        c = cutrun[sel]
        R.add(
            f"cut.runs.{f}.n_cut",
            int(c.sum()),
            body_set="100",
            post_hoc=PH_YES,
            definition=f"{FAM_NAME[f]} Strict-trained runs with mean Strict-evaluator episode length at most 20 steps (cut-off runs).",
        )
        R.add(
            f"cut.runs.{f}.n_runs",
            int(c.size),
            body_set="100",
            post_hoc=PH_YES,
            definition=f"{FAM_NAME[f]} Strict-trained runs (bodies x 3 seeds).",
        )
        R.add(
            f"cut.runs.{f}.ret_noncut",
            float(rs[~c].mean()),
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="Mean over non-cut-off Strict runs of run PSR over the body's three-seed Standard mean.",
        )
        R.add(
            f"cut.runs.{f}.ret_cut",
            float(rs[c].mean()) if c.any() else None,
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="Same over cut-off runs.",
        )
        R.add(
            f"cut.runs.{f}.ret_all",
            float(rs.mean()),
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="Same over all Strict runs of the family.",
        )
        z30 = sel & np.isclose(z0, 0.30)
        surv = (~cutrun[z30]).any(1)
        R.add(
            f"cut.z030_surviving_strict_seed.{f}",
            int(surv.sum()),
            body_set="25 bodies at z0 = 0.30 m",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_STRICT],
            definition=f"{FAM_NAME[f]} bodies at z0 = 0.30 m with at least one Strict-trained seed whose Strict-evaluator episodes average more than 20 steps.",
        )
        R.add(
            f"cut.z030_surviving_strict_seed_50.{f}",
            int((Lstr[z30] > 50).any(1).sum()),
            body_set="25 bodies at z0 = 0.30 m",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_STRICT],
            definition=f"Variant: {FAM_NAME[f]} bodies at z0 = 0.30 m with a Strict-trained seed whose Strict-evaluator episodes average more than 50 steps (the count is the same for any threshold from 49 to 580 steps).",
        )
        R.add(
            f"cut.z030_bodies.{f}",
            int(z30.sum()),
            body_set="25 bodies at z0 = 0.30 m",
            post_hoc=PH_YES,
            definition=f"{FAM_NAME[f]} bodies at z0 = 0.30 m.",
        )
    sm = S.mean(1)
    lowest12 = np.argsort(sm, kind="stable")[:12]
    R.add(
        "cut.vt10_in_lowest12_strict",
        int(vt10[lowest12].sum()),
        body_set="100",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="VT cut-off bodies among the 12 lowest-ranked bodies after Strict training (three-seed mean PSR).",
    )

    # inert-28 records
    ident, n_pol = 0, 0
    for b in np.array(D.bodies)[inert20]:
        for t in SWEEP:
            for s in SEEDS:
                r = D.raw[(t, s)][b]
                n_pol += 1
                ident += int(r[EV_COMMON] == r[EV_NEUTRAL])
    R.add(
        "inert28.identical_records",
        ident,
        body_set=SET_LABEL["inert28"],
        post_hoc=PH_YES,
        evaluator="common vs neutral",
        definition="Policies (28 bodies x 5 training thresholds x 3 seeds) whose common-evaluator record equals the neutral record in every field.",
    )
    R.add("inert28.policies", n_pol, body_set=SET_LABEL["inert28"], post_hoc=PH_YES, definition="Policies checked.")
    z45 = inert20 & np.isclose(z0, 0.45)
    ident45 = sum(int(D.raw[(t, s)][b][EV_COMMON] == D.raw[(t, s)][b][EV_NEUTRAL]) for b in np.array(D.bodies)[z45] for t in SWEEP for s in SEEDS)
    R.add(
        "inert28.z045.identical_policies",
        ident45,
        body_set="3 bodies at z0 = 0.45 m",
        post_hoc=PH_YES,
        definition="Policies of the 3 bodies at z0 = 0.45 m with identical common and neutral records (each record covers 20 episodes).",
    )
    R.add(
        "inert28.z045.episodes",
        int(z45.sum() * len(SWEEP) * 3 * 20),
        body_set="3 bodies at z0 = 0.45 m",
        post_hoc=PH_YES,
        definition="Recorded evaluation episodes of these bodies under the common evaluator.",
    )
    marg = D.audit_full["summary"]["probe_margin_0.20_inert"]
    m30 = [marg[b] for b, k in zip(D.bodies, inert20 & np.isclose(z0, 0.30)) if k]
    m45 = [marg[b] for b, k in zip(D.bodies, z45) if k]
    R.add(
        "inert28.z030.min_probe_margin_m",
        float(min(m30)),
        body_set="25 bodies at z0 = 0.30 m",
        dec=4,
        post_hoc=PH_YES,
        definition="Smallest audit-probe margin of the torso above the Standard cutoff (m).",
    )
    R.add(
        "inert28.z045.probe_margins_m",
        sorted(m45),
        body_set="3 bodies at z0 = 0.45 m",
        dec=4,
        post_hoc=PH_YES,
        definition="Audit-probe margins of the torso above the Standard cutoff (m), negative when the probe went below.",
    )
    never = D.audit_full["summary"]["firing_0.20_never_eval_fall"]
    R.add(
        "inert28.practically_inert_mvt",
        len(never),
        body_set="72",
        post_hoc=PH_YES,
        definition="Bodies of the 72 whose Standard cutoff never ended a recorded common-evaluator episode.",
    )
    R.add(
        "inert28.practically_inert_mvt_z0",
        [float(min(D.audit[b]["z0"] for b in never)), float(max(D.audit[b]["z0"] for b in never))],
        body_set="72",
        dec=2,
        post_hoc=PH_YES,
        definition="Reset-height range of those bodies (m).",
    )
    fact = D.audit_full["summary"]["factorial_tau0p20"]["inert_bodies"]
    dones = [v for b in fact for v in fact[b].values()]
    R.add(
        "inert28.factorial_runs",
        len(dones),
        body_set="10 inert bodies in the factorial panel",
        post_hoc=PH_YES,
        definition="Standard-trained factorial runs (5M steps) of inert bodies with recorded training done counts.",
    )
    R.add(
        "inert28.factorial_runs_zero_dones",
        int(sum(v == 0 for v in dones)),
        body_set="10 inert bodies in the factorial panel",
        post_hoc=PH_YES,
        definition="Of those, runs with zero training dones.",
    )
    dones50 = [
        v
        for b, k in zip(D.bodies, inert20)
        if k and "factorial_tau0p50_5M_train_dones" in D.audit[b]
        for v in D.audit[b]["factorial_tau0p50_5M_train_dones"].values()
    ]
    if dones50:
        spd = 5e6 / np.array(dones50, float)
        R.add(
            "inert28.factorial_strict_runs",
            len(spd),
            body_set="10 inert bodies in the factorial panel",
            post_hoc=PH_YES,
            definition="Strict-trained factorial runs (5M steps) of inert bodies with recorded training done counts.",
        )
        R.add(
            "inert28.factorial_strict_runs_le10_steps",
            int((spd <= 10).sum()),
            body_set="10 inert bodies in the factorial panel",
            post_hoc=PH_YES,
            definition="Of those, runs whose training episodes averaged at most 10 steps (5M steps over recorded dones).",
        )
        R.add(
            "inert28.factorial_strict_min_steps_per_done",
            float(spd.min()),
            body_set="10 inert bodies in the factorial panel",
            dec=1,
            post_hoc=PH_YES,
            definition="Shortest average training episode among those runs (steps).",
        )
        R.add(
            "inert28.factorial_strict_median_steps_per_done",
            float(np.median(spd)),
            body_set="10 inert bodies in the factorial panel",
            dec=1,
            post_hoc=PH_YES,
            definition="Median average training episode length over those runs (steps).",
        )
    rfl = D.get("0p00").mean(1) / D.get("0p20").mean(1)
    R.add(
        "inert28.rf_len.mean",
        float(rfl[inert20].mean()),
        body_set=SET_LABEL["inert28"],
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Mean per-body Reset-free/Standard ratio of three-seed mean PSR.",
    )
    R.add(
        "inert28.rf_len.range",
        [float(rfl[inert20].min()), float(rfl[inert20].max())],
        body_set=SET_LABEL["inert28"],
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Range of that ratio.",
    )

    # zero-action records
    passive_p, settle_p = Path(args.passive), Path(args.settle)
    passive = settle = None
    if passive_p.exists():
        passive = {r["body"]: r for r in json.loads(passive_p.read_text())}
        R.add(
            "zero.below_strict_1s",
            int(sum(passive[b]["zero_action_height_step50"] < 0.5 * passive[b]["z0"] for b in D.bodies)),
            body_set="100",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Bodies whose torso is below 0.5 z0 after 50 control steps (1 s) with zero action (all100_passive.json).",
        )
        z30b = [b for b, k in zip(D.bodies, np.isclose(z0, 0.30)) if k]
        R.add(
            "zero.z030.cross_steps",
            sorted({passive[b]["zero_action_first_cross_step"] for b in z30b}),
            body_set="25 bodies at z0 = 0.30 m",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            dec=0,
            definition="Distinct control steps at which the torso first drops below 0.5 z0.",
        )
        R.add(
            "zero.z030.contact_steps",
            sorted({passive[b]["zero_action_first_floor_contact_step"] for b in z30b}),
            body_set="25 bodies at z0 = 0.30 m",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            dec=0,
            definition="Distinct control steps of first floor contact.",
        )
        tall = [b for b, k in zip(D.bodies, vt10 & ~np.isclose(z0, 0.30)) if k]
        R.add(
            "zero.vt10_taller.cross_steps",
            sorted({passive[b]["zero_action_first_cross_step"] for b in tall}),
            body_set="3 taller VT cut-off bodies",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            dec=0,
            definition="Distinct first steps below 0.5 z0.",
        )
        R.add(
            "zero.vt10_taller.contact_steps",
            sorted({passive[b]["zero_action_first_floor_contact_step"] for b in tall}),
            body_set="3 taller VT cut-off bodies",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            dec=0,
            definition="Distinct first floor-contact steps.",
        )
        R.add(
            "zero.floor_cutoff_body.cross_step",
            passive[floor_cut[0]]["zero_action_first_cross_step"] if floor_cut else None,
            body_set="1",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            dec=0,
            definition="First step below 0.5 z0 for the Floor cut-off body (None: never within 60 steps).",
        )
        lp = np.array([passive[b]["lowest_point_at_reset"] for b in D.bodies])
        R.add(
            "zero.lowest_point_at_reset_m",
            [float(lp.min()), float(lp.max())],
            body_set="100",
            dec=3,
            evaluator="reset pose",
            post_hoc=PH_YES,
            definition="Range of the lowest body point above the floor in the reset pose (m).",
        )
        nu = np.array([passive[b]["nu"] for b in D.bodies])
        R.add(
            "bodies.actuators_range",
            [int(nu.min()), int(nu.max())],
            body_set="100",
            post_hoc=PH_REV,
            dec=0,
            definition="Range of actuated hinge joints (MuJoCo nu), which is also the action dimension.",
        )
        R.add(
            "bodies.obs_dim_range",
            [int(11 + 2 * nu.min()), int(11 + 2 * nu.max())],
            body_set="100",
            dec=0,
            post_hoc=PH_REV,
            definition="Observation dimension 11 + 2 nu: torso height (1), orientation quaternion (4), hinge angles (nu), free-joint velocities (6), hinge velocities (nu).",
        )
        extra["nu"] = nu
    else:
        log(f"zero-action rollout records not found at {passive_p}")
    if settle_p.exists():
        settle = {r[1]: r for r in json.loads(settle_p.read_text())}
        below3 = np.array([settle[b][3] < 0.5 * settle[b][2] for b in D.bodies])
        R.add(
            "zero.below_strict_3s",
            int(below3.sum()),
            body_set="100",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Bodies whose torso has settled below 0.5 z0 after 3 s with zero action (settle.json).",
        )
        vt10_derl = [b for b, k in zip(D.bodies, vt10) if k and settle[b][3] < 0.5 * (settle[b][2] - 0.2)]
        R.add(
            "zero.vt10_below_derl_cutoff_3s",
            vt10_derl,
            body_set="10",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            kind="ids",
            definition="VT cut-off bodies whose zero-action torso after 3 s lies below DERL's cutoff 0.5 (z0 minus 0.2 m).",
        )
        R.add(
            "zero.vt10_below_derl_cutoff_3s.n",
            len(vt10_derl),
            body_set="10",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Number of such bodies.",
        )
        R.add(
            "zero.vt10_below_derl_cutoff_3s.z0",
            sorted({round(settle[b][2], 2) for b in vt10_derl}),
            body_set="10",
            dec=2,
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Their reset heights (m).",
        )
    else:
        log(f"settle records not found at {settle_p}")

    # random open-loop probes (feasibility only, not proof)
    vt10_ids = [b for b, k in zip(D.bodies, vt10) if k]
    shoot_p, search_p = Path(args.shoot), Path(args.search)
    if shoot_p.exists():
        sh = json.loads(shoot_p.read_text())
        for b in sorted(sh):
            R.add(
                f"probe.shoot.{b}.reach50",
                int(sh[b]["n_reach50"]),
                body_set="1",
                evaluator="random open-loop actions from the nominal reset",
                post_hoc=PH_YES,
                definition=f"{b}: of 300 random open-loop action sequences (100 constant bang-bang, 100 per-step uniform, 100 smoothed), those keeping the torso at or above 0.5 z0 for 50 control steps (sk_shoot.json).",
            )
        R.add(
            "probe.shoot.vt10_probed",
            [b for b in sorted(sh) if b in vt10_ids],
            body_set="10",
            post_hoc=PH_YES,
            kind="ids",
            evaluator="random open-loop actions from the nominal reset",
            definition="VT cut-off bodies included in that 300-sequence probe (the others were not probed there).",
        )
    else:
        log(f"random-shooting probe record not found at {shoot_p}")
    if search_p.exists():
        se = {r["body"]: r for r in json.loads(search_p.read_text())}
        got = {b: int(se[b]["search_n_survive_ge50"]) for b in vt10_ids if b in se}
        trials = sorted({int(se[b]["search_trials"]) for b in got})
        R.add(
            "probe.search.vt10.n_probed",
            len(got),
            body_set="10",
            post_hoc=PH_YES,
            evaluator="random open-loop actions from the nominal reset",
            definition=f"VT cut-off bodies in the larger random-shooting search ({trials} sequences per body, vt10_search.json).",
        )
        R.add(
            "probe.search.vt10.n_never_up",
            int(sum(v == 0 for v in got.values())),
            body_set="10",
            post_hoc=PH_YES,
            evaluator="random open-loop actions from the nominal reset",
            definition="Of those, bodies for which no sampled sequence kept the torso at or above 0.5 z0 for 50 control steps.",
        )
        R.add(
            "probe.search.vt10.survivors",
            {b: v for b, v in got.items() if v > 0},
            body_set="10",
            post_hoc=PH_YES,
            dec=0,
            evaluator="random open-loop actions from the nominal reset",
            definition="Bodies with at least one surviving sequence, and how many sequences survived.",
        )
    else:
        log(f"random-shooting search record not found at {search_p}")

    # physics re-run (optional)
    if not args.no_physics and Path(args.xml_dir).exists():
        try:
            log("re-running zero-action rollouts in MuJoCo")
            ph = zero_action_physics(Path(args.xml_dir), D.bodies)
            z0p = np.array([ph[b]["z0"] for b in D.bodies])
            R.add(
                "physics.max_abs_z0_diff",
                float(np.max(np.abs(z0p - z0))),
                body_set="100",
                dec=6,
                evaluator="zero-action rollout (re-run)",
                post_hoc=PH_YES,
                definition="Largest difference between the simulated reset height and the audit z0 (m).",
            )
            R.add(
                "physics.torso_radii",
                sorted({round(ph[b]["torso_radius"], 6) for b in D.bodies}),
                body_set="100",
                dec=3,
                evaluator="XML",
                post_hoc=PH_YES,
                definition="Distinct torso sphere radii in the XMLs (m).",
            )
            for lab, key in (("1s", "z50"), ("1p5s", "z75"), ("3s", "z150")):
                R.add(
                    f"physics.below_strict_{lab}",
                    int(sum(ph[b][key] < 0.5 * ph[b]["z0"] for b in D.bodies)),
                    body_set="100",
                    evaluator="zero-action rollout (re-run)",
                    post_hoc=PH_YES,
                    definition=f"Bodies whose torso is below 0.5 z0 at {lab.replace('p', '.')} with zero action (MuJoCo re-run).",
                )
            z30b = [b for b, k in zip(D.bodies, np.isclose(z0, 0.30)) if k]
            tall = [b for b, k in zip(D.bodies, vt10 & ~np.isclose(z0, 0.30)) if k]
            for lab, bl in (("z030", z30b), ("vt10_taller", tall)):
                R.add(
                    f"physics.{lab}.cross_steps",
                    sorted({ph[b]["cross"] for b in bl}),
                    body_set=lab.replace("_", " "),
                    evaluator="zero-action rollout (re-run)",
                    post_hoc=PH_YES,
                    dec=0,
                    definition="Distinct first control steps below 0.5 z0 (MuJoCo re-run).",
                )
                R.add(
                    f"physics.{lab}.contact_steps",
                    sorted({ph[b]["contact"] for b in bl}),
                    body_set=lab.replace("_", " "),
                    evaluator="zero-action rollout (re-run)",
                    post_hoc=PH_YES,
                    dec=0,
                    definition="Distinct first floor-contact steps (MuJoCo re-run).",
                )
            nu_p = np.array([ph[b]["nu"] for b in D.bodies])
            R.add(
                "physics.actuators_range",
                [int(nu_p.min()), int(nu_p.max())],
                body_set="100",
                dec=0,
                post_hoc=PH_REV,
                evaluator="XML",
                definition="Range of actuated hinge joints (MuJoCo nu) from the XMLs.",
            )
            if passive is not None:
                d50 = max(abs(ph[b]["z50"] - passive[b]["zero_action_height_step50"]) for b in D.bodies)
                # the records searched the first 60 control steps only
                same_c = all(
                    (ph[b]["cross"] if ph[b]["cross"] is not None and ph[b]["cross"] <= 60 else None) == passive[b]["zero_action_first_cross_step"]
                    for b in D.bodies
                )
                same_t = all(ph[b]["contact"] == passive[b]["zero_action_first_floor_contact_step"] for b in D.bodies)
                R.add(
                    "physics.vs_records.max_abs_z_1s",
                    float(d50),
                    body_set="100",
                    dec=6,
                    post_hoc=PH_YES,
                    evaluator="re-run vs all100_passive.json",
                    definition="Largest difference in torso height at 1 s between the re-run and the records (records rounded to 0.1 mm).",
                )
                R.add(
                    "physics.vs_records.same_cross_steps",
                    bool(same_c),
                    body_set="100",
                    post_hoc=PH_YES,
                    evaluator="re-run vs all100_passive.json",
                    definition="First steps below 0.5 z0 agree for every body.",
                )
                R.add(
                    "physics.vs_records.same_contact_steps",
                    bool(same_t),
                    body_set="100",
                    post_hoc=PH_YES,
                    evaluator="re-run vs all100_passive.json",
                    definition="First floor-contact steps agree for every body.",
                )
            if settle is not None:
                d3 = max(abs(ph[b]["z150"] - settle[b][3]) for b in D.bodies)
                R.add(
                    "physics.vs_records.max_abs_z_3s",
                    float(d3),
                    body_set="100",
                    dec=6,
                    post_hoc=PH_YES,
                    evaluator="re-run vs settle.json",
                    definition="Largest difference in torso height at 3 s.",
                )
            extra["physics"] = ph
        except Exception as exc:  # disclosed, never silent
            log(f"physics re-run failed: {exc!r}")
            R.add(
                "physics.error",
                repr(exc),
                post_hoc=PH_YES,
                definition="The MuJoCo re-run failed; records were used instead.",
                kind="text",
            )

    # body description
    nb = np.array([D.feats[b]["n_bodies"] for b in D.bodies])
    td = np.array([D.feats[b]["tree_depth"] for b in D.bodies])
    R.add(
        "bodies.n_bodies_range",
        [int(nb.min()), int(nb.max())],
        body_set="100",
        dec=0,
        post_hoc=PH_REV,
        definition="Range of rigid bodies per XML (morph_features n_bodies).",
    )
    R.add(
        "bodies.tree_depth_range",
        [int(td.min()), int(td.max())],
        body_set="100",
        dec=0,
        post_hoc=PH_REV,
        definition="Range of morph_features tree_depth (levels of bodies below the torso, torso children = 1).",
    )
    for f in FAMS:
        R.add(
            f"bodies.family.{f}",
            int((fam == f).sum()),
            body_set="100",
            post_hoc=PH_REV,
            definition=f"{FAM_NAME[f]} bodies in the corpus.",
        )
    return sets, extra


def section_duplicates(D: Data, R: Registry, args, sets):
    R.section = "Duplicate bodies"
    xd = Path(args.xml_dir)
    if not xd.exists():
        log(f"XML dir {xd} not found; duplicate check skipped")
        return
    h = {b: hashlib.sha256((xd / f"{b}.xml").read_bytes()).hexdigest() for b in D.bodies}
    by = collections.defaultdict(list)
    for b, x in h.items():
        by[x].append(b)
    pairs = [sorted(v) for v in by.values() if len(v) > 1]
    R.add("dup.distinct_files", len(by), body_set="100", post_hoc=PH_YES, definition="Distinct XML files by SHA-256.")
    R.add("dup.pairs", pairs, body_set="100", post_hoc=PH_YES, kind="ids", definition="Groups of identical XML files.")
    R.add(
        "dup.pair_families",
        sorted(p[0].split("-")[0] for p in pairs),
        body_set="100",
        post_hoc=PH_YES,
        definition="Family of each identical group.",
    )
    V = D.V()
    base = ranking_components(V)
    max_rho = max_D = 0.0
    for choice in itertools.product(*[range(len(p)) for p in pairs]):
        drop = {p[c] for p, c in zip(pairs, choice)}
        keep = np.array([b not in drop for b in D.bodies])
        c = ranking_components(V[keep])
        for a, b in PAIRS:
            max_rho = max(max_rho, abs(c[f"rho3_{a}_{b}"][0] - base[f"rho3_{a}_{b}"][0]))
            max_D = max(max_D, abs(c[f"D_{a}_{b}"][0] - base[f"D_{a}_{b}"][0]))
    n_choice = int(np.prod([len(p) for p in pairs]))
    R.add(
        "dup.drop_choices",
        n_choice,
        body_set="100",
        post_hoc=PH_YES,
        definition="Ways to drop one body from each identical group.",
    )
    R.add(
        "dup.max_abs_change_rho",
        float(max_rho),
        body_set="100 minus one of each pair",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Largest change in any three-seed rank correlation over all drop choices.",
    )
    R.add(
        "dup.max_abs_change_D",
        float(max_D),
        body_set="100 minus one of each pair",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Largest change in any excess disagreement D over all drop choices.",
    )


def section_ranking(D: Data, R: Registry, args, sets, extra):
    R.section = "Rank agreement across rules and seeds"
    B = args.boot
    V = D.V()
    fam = D.fam
    pair_lab = {p: f"{RULE_NAME[p[0]]} vs {RULE_NAME[p[1]]}" for p in PAIRS}
    for sname in ("all100", "g72", "s89", "s77", "s68", "s65", "no_vt10"):
        m = sets[sname]
        Vs = V[m]
        pt = ranking_components(Vs)
        bt = ranking_components(Vs, boot_index(len(Vs), B))
        ph = PH_REV if sname == "all100" else PH_YES
        for a, b in PAIRS:
            k = f"{a}_{b}"
            R.add(
                f"rank.rho3.{k}.{sname}",
                float(pt[f"rho3_{k}"][0]),
                ci=pct_ci(bt[f"rho3_{k}"]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                dec=3,
                definition=f"Spearman rho between {pair_lab[(a, b)]} rankings of three-seed mean PSR; 95% body-bootstrap interval.",
            )
            R.add(
                f"rank.D.{k}.{sname}",
                float(pt[f"D_{k}"][0]),
                ci=pct_ci(bt[f"D_{k}"]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                dec=3,
                definition=f"Excess disagreement D = (W_a + W_b)/2 minus the mean single-seed rho over the 9 cross-rule pairs ({pair_lab[(a, b)]}).",
            )
            R.add(
                f"rank.C.{k}.{sname}",
                float(pt[f"C_{k}"][0]),
                body_set=SET_LABEL[sname],
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Mean single-seed rho over the 9 {pair_lab[(a, b)]} seed pairs.",
            )
            if sname in ("all100", "g72", "s89", "s77"):
                R.add(
                    f"rank.noise_ref.{k}.{sname}",
                    float(pt[f"ref_{k}"][0]),
                    ci=pct_ci(bt[f"ref_{k}"]),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    definition="Noise-only reference sqrt(R3_a R3_b) for three-seed rankings, R3 = 3W/(1+2W) (Spearman-Brown from different-seed pairs).",
                )
                R.add(
                    f"rank.noise_gap.{k}.{sname}",
                    float(pt[f"gap_{k}"][0]),
                    ci=pct_ci(bt[f"gap_{k}"]),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    definition="Noise-only reference minus observed three-seed rho.",
                )
                R.add(
                    f"rank.disattenuated.{k}.{sname}",
                    float(pt[f"dis_{k}"][0]),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    definition="Observed three-seed rho over the noise-only reference.",
                )
            R.add(
                f"rank.top10.{k}.{sname}",
                topk_overlap(Vs[:, RULES.index(a)].mean(1), Vs[:, RULES.index(b)].mean(1), 10),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition="Bodies shared by the two top-10 sets of three-seed mean PSR (descriptive).",
            )
        for r in RULES:
            R.add(
                f"rank.W.{r}.{sname}",
                float(pt[f"W_{r}"][0]),
                ci=pct_ci(bt[f"W_{r}"]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                definition=f"W: mean Spearman rho between single-seed {RULE_NAME[r]} rankings over the 3 seed pairs.",
            )
            if sname == "all100":
                R.add(
                    f"rank.R3.{r}.{sname}",
                    float(pt[f"R3_{r}"][0]),
                    body_set=SET_LABEL[sname],
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Spearman-Brown reliability of the three-seed {RULE_NAME[r]} ranking, 3W/(1+2W).",
                )
        extra.setdefault("D_rf_str_by_set", {})[sname] = float(pt["D_rf_str"][0])
    sub = [extra["D_rf_str_by_set"][s] for s in ("g72", "s89", "s77", "s68", "s65")]
    R.add(
        "rank.D.rf_str.subset_range",
        [min(sub), max(sub)],
        body_set="72, 89, 77, 68, 65",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Range of Strict vs Reset-free D over the five subsets without defect-affected bodies.",
    )

    # single-seed detail on all 100
    m = sets["all100"]
    for i, r in enumerate(RULES):
        vals, ovs = [], []
        for s, t in itertools.combinations(range(3), 2):
            x = rho(V[:, i, s], V[:, i, t])
            o = topk_overlap(V[:, i, s], V[:, i, t], 10)
            vals.append(x)
            ovs.append(o)
            R.add(
                f"rank.single.within.{r}.{SEEDS[s]}_{SEEDS[t]}",
                x,
                body_set="100",
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                definition=f"Spearman rho between the single-seed {RULE_NAME[r]} rankings of seeds {SEEDS[s]} and {SEEDS[t]}.",
            )
        R.add(
            f"rank.single.within_top10.{r}",
            ovs,
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            dec=0,
            definition=f"Top-10 overlaps between single-seed {RULE_NAME[r]} rankings (seed pairs 42-43, 42-44, 43-44).",
        )
        R.add(
            f"rank.single.within_top10_mean.{r}",
            float(np.mean(ovs)),
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            dec=2,
            definition="Mean of those overlaps.",
        )
    allw = [R.get(f"rank.single.within.{r}.{SEEDS[s]}_{SEEDS[t]}")["value"] for r in RULES for s, t in itertools.combinations(range(3), 2)]
    R.add(
        "rank.single.within.range",
        [min(allw), max(allw)],
        body_set="100",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Range of the 9 within-rule single-seed rho values.",
    )
    allo = [x for r in RULES for x in R.get(f"rank.single.within_top10.{r}")["value"]]
    R.add(
        "rank.single.within_top10.range",
        [min(allo), max(allo)],
        body_set="100",
        dec=0,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Range of within-rule single-seed top-10 overlaps.",
    )
    for a, b in PAIRS:
        ia, ib = RULES.index(a), RULES.index(b)
        vals = [rho(V[:, ia, s], V[:, ib, t]) for s in range(3) for t in range(3)]
        ovs = [topk_overlap(V[:, ia, s], V[:, ib, t], 10) for s in range(3) for t in range(3)]
        R.add(
            f"rank.single.cross.{a}_{b}.range",
            [min(vals), max(vals)],
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition=f"Range of single-seed rho over the 9 {pair_lab[(a, b)]} seed pairs.",
        )
        R.add(
            f"rank.single.cross.{a}_{b}.values",
            vals,
            body_set="100",
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="The 9 values, first-rule seed major (42/42, 42/43, ..., 44/44).",
        )
        R.add(
            f"rank.single.cross_top10.{a}_{b}.range",
            [min(ovs), max(ovs)],
            body_set="100",
            dec=0,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Range of top-10 overlaps over the 9 single-seed cross pairs.",
        )
        R.add(
            f"rank.single.cross_top10.{a}_{b}.mean",
            float(np.mean(ovs)),
            body_set="100",
            dec=2,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Mean of those overlaps.",
        )

    # neutral evaluator
    Vn = D.V(EV_NEUTRAL)
    for sname in ("all100", "g72"):
        mm = sets[sname]
        pt = ranking_components(Vn[mm])
        bt = ranking_components(Vn[mm], boot_index(int(mm.sum()), B))
        for a, b in PAIRS:
            k = f"{a}_{b}"
            R.add(
                f"rank.neutral.rho3.{k}.{sname}",
                float(pt[f"rho3_{k}"][0]),
                ci=pct_ci(bt[f"rho3_{k}"]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_NEUTRAL],
                post_hoc=PH_YES,
                definition=f"Spearman rho between {pair_lab[(a, b)]} rankings of three-seed mean PSR under the neutral evaluator.",
            )
            R.add(
                f"rank.neutral.D.{k}.{sname}",
                float(pt[f"D_{k}"][0]),
                ci=pct_ci(bt[f"D_{k}"]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_NEUTRAL],
                post_hoc=PH_YES,
                definition="Excess disagreement D under the neutral evaluator.",
            )

    # exchangeability null
    for sname in ("all100", "g72", "s77"):
        mm = sets[sname]
        for a, b in (("rf", "str"),) if sname != "all100" else PAIRS:
            ex = exch_null(V[mm], a, b, args.perm)
            k = f"{a}_{b}"
            R.add(
                f"rank.exch.{k}.{sname}.null_rho",
                ex["null_rho_mean"],
                ci=ex["null_rho_ci"],
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                definition="Rule-exchangeability null (each body's 6 runs split at random into two pseudo-rules of 3): mean null rho and its central 95% range.",
            )
            R.add(
                f"rank.exch.{k}.{sname}.p_rho",
                ex["p_rho"],
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                kind="p",
                definition="Permutation p for observed rho at or below the null.",
            )
            for kk in (5, 10, 20):
                R.add(
                    f"rank.exch.{k}.{sname}.obs_top{kk}",
                    ex["obs"][f"top{kk}"],
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    definition=f"Observed top-{kk} overlap of three-seed means.",
                )
                R.add(
                    f"rank.exch.{k}.{sname}.null_top{kk}",
                    ex[f"null_top{kk}_mean"],
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    dec=2,
                    definition=f"Mean top-{kk} overlap under the null.",
                )
                R.add(
                    f"rank.exch.{k}.{sname}.p_top{kk}",
                    ex[f"p_top{kk}"],
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    kind="p",
                    definition=f"Permutation p for an overlap at or below the observed top-{kk} overlap.",
                )
            R.add(
                f"rank.exch.{k}.{sname}.frac_top10_le_obs",
                ex["frac_top10_le_obs"],
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                dec=3,
                definition="Share of null draws whose top-10 overlap is at or below the observed one.",
            )

    # within-family and family-centred
    M = V.mean(-1)
    for sname in ("all100", "g72"):
        mm = sets[sname]
        for f in FAMS:
            sel = mm & (fam == f)
            a, b = M[sel, 2], M[sel, 0]
            idx = boot_index(int(sel.sum()), B)
            bs = spearman_last(a[idx], b[idx])
            R.add(
                f"rank.within_family.rf_str.{f}.{sname}",
                rho(a, b),
                ci=pct_ci(bs),
                body_set=f"{SET_LABEL[sname]}, {FAM_NAME[f]} only",
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                definition=f"Strict vs Reset-free rho within {FAM_NAME[f]} (three-seed means).",
            )
        Mc = M[mm].copy()
        fk = fam[mm]
        for f in FAMS:
            Mc[fk == f] -= Mc[fk == f].mean(0, keepdims=True)
        R.add(
            f"rank.family_centred.rf_str.{sname}",
            rho(Mc[:, 2], Mc[:, 0]),
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="Strict vs Reset-free rho after subtracting each family's mean PSR per rule.",
        )
    # Fig. 10 diagonal counts
    for f in FAMS:
        sel = fam == f
        R.add(
            f"rank.fig10.{f}.str_gt_rf",
            int((M[sel, 2] > M[sel, 0]).sum()),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"{FAM_NAME[f]} bodies whose three-seed Strict PSR exceeds their Reset-free PSR.",
        )
        R.add(
            f"rank.fig10.{f}.str_lt_rf",
            int((M[sel, 2] < M[sel, 0]).sum()),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"{FAM_NAME[f]} bodies whose three-seed Strict PSR is below their Reset-free PSR.",
        )
    # top-k extras
    for kk in (5, 20):
        R.add(
            f"rank.top{kk}.rf_str.all100",
            topk_overlap(M[:, 2], M[:, 0], kk),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"Strict and Reset-free top-{kk} overlap (three-seed means).",
        )
    t_str, t_rf = topk_set(M[:, 2], 10), topk_set(M[:, 0], 10)
    cut23 = extra["cut23"]
    leave_rf = sorted(t_rf - t_str)
    leave_str = sorted(t_str - t_rf)
    R.add(
        "rank.top10_leavers.rf_not_str.n_cut23",
        int(sum(cut23[i] for i in leave_rf)),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Of the 5 Reset-free top-10 bodies outside the Strict top 10, cut-off-seed bodies.",
    )
    R.add(
        "rank.top10_leavers.str_not_rf.n_cut23",
        int(sum(cut23[i] for i in leave_str)),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Of the 5 Strict top-10 bodies outside the Reset-free top 10, cut-off-seed bodies.",
    )
    R.add(
        "rank.top10_leavers.rf_not_str.ids",
        [D.bodies[i] for i in leave_rf],
        body_set="100",
        post_hoc=PH_YES,
        kind="ids",
        evaluator=EV_LABEL[EV_COMMON],
        definition="Reset-free top-10 bodies outside the Strict top 10.",
    )

    # simulated independent three-seed reruns (normal and residual-bootstrap noise models)
    obs10 = topk_overlap(M[:, 2], M[:, 0], 10)
    sims = {}
    for r in ("str", "rf"):
        Y = V[:, RULES.index(r), :]
        ybar = Y.mean(1)
        e = Y - ybar[:, None]
        s2 = (e**2).sum(1) / 2
        vnm = s2 / 3
        lam = math.sqrt(max(0.0, 1 - vnm.mean() / ybar.var(ddof=1)))
        mu_shr = ybar.mean() + lam * (ybar - ybar.mean())
        for model in ("normal", "resid_boot", "normal_shrunk"):
            rng = np.random.default_rng(BOOT_SEED)
            Bs = B
            n = len(ybar)
            if model == "resid_boot":
                sc = e * math.sqrt(1.5)
                A_ = ybar[None] + np.take_along_axis(np.broadcast_to(sc, (Bs, n, 3)), rng.integers(0, 3, (Bs, n, 3)), axis=2).mean(2)
                B_ = ybar[None] + np.take_along_axis(np.broadcast_to(sc, (Bs, n, 3)), rng.integers(0, 3, (Bs, n, 3)), axis=2).mean(2)
            else:
                mu = mu_shr if model == "normal_shrunk" else ybar
                A_ = mu[None] + rng.standard_normal((Bs, n)) * np.sqrt(vnm)[None]
                B_ = mu[None] + rng.standard_normal((Bs, n)) * np.sqrt(vnm)[None]
            ov = (topk_mask_rows(A_, 10) & topk_mask_rows(B_, 10)).sum(1)
            sims[(r, model)] = (float(ov.mean()), float(np.mean(ov <= obs10)))
            R.add(
                f"rank.sim_rerun.{r}.{model}.top10_mean",
                float(ov.mean()),
                body_set="100",
                dec=2,
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Mean top-10 overlap of two simulated independent three-seed {RULE_NAME[r]} reruns ({model.replace('_', ' ')} noise model).",
            )
            R.add(
                f"rank.sim_rerun.{r}.{model}.P_le_obs",
                float(np.mean(ov <= obs10)),
                body_set="100",
                dec=3,
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Share of simulated rerun pairs with top-10 overlap at or below the observed Strict vs Reset-free 5.",
            )
    means = [v[0] for v in sims.values()]
    probs = [v[1] for v in sims.values()]
    R.add(
        "rank.sim_rerun.top10_mean_range",
        [min(means), max(means)],
        body_set="100",
        dec=2,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Range over the 6 rule x noise-model simulations.",
    )
    R.add(
        "rank.sim_rerun.P_le_obs_range",
        [min(probs), max(probs)],
        body_set="100",
        dec=3,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Range of the corresponding shares.",
    )


def family_block(D, V, mask):
    M = V[mask].mean(-1)
    fam = D.fam[mask]
    rf_len = M[:, 0] / M[:, 1]
    str_len = M[:, 2] / M[:, 1]
    return M, fam, rf_len, str_len


def section_family(D: Data, R: Registry, args, sets, extra):
    R.section = "Family means, retention and family contrasts"
    B, P = args.boot, args.perm
    V = D.V()
    for sname in SUBSETS_MAIN + ("no_vt10",):
        M, fam, rf_len, str_len = family_block(D, V, sets[sname])
        ph = PH_REV if sname == "all100" else PH_YES
        wc = {}
        for f in FAMS:
            sel = fam == f
            for i, r in enumerate(RULES):
                R.add(
                    f"fam.mean.{r}.{f}.{sname}",
                    float(M[sel, i].mean()),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=ph,
                    definition=f"{FAM_NAME[f]} mean of three-seed mean PSR after {RULE_NAME[r]} training.",
                )
            R.add(
                f"fam.ret.rf_len.{f}.{sname}",
                float(rf_len[sel].mean()),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition=f"{FAM_NAME[f]} retention Reset-free/Standard: mean per-body ratio of three-seed means.",
            )
            R.add(
                f"fam.ret.str_len.{f}.{sname}",
                float(str_len[sel].mean()),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition=f"{FAM_NAME[f]} retention Strict/Standard: mean per-body ratio of three-seed means.",
            )
            R.add(
                f"fam.loss_pct.rf.{f}.{sname}",
                float(100 * (1 - rf_len[sel].mean())),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                dec=1,
                definition="Reset-free loss in percent, 100 (1 minus retention).",
            )
            R.add(
                f"fam.loss_pct.str.{f}.{sname}",
                float(100 * (1 - str_len[sel].mean())),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                dec=1,
                definition="Strict loss in percent, 100 (1 minus retention).",
            )
            wc[f] = min(rf_len[sel].mean(), str_len[sel].mean())
            R.add(
                f"fam.worst_case.{f}.{sname}",
                float(wc[f]),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition="Worst-case family retention, the smaller of the two family retentions.",
            )
        best = max(wc, key=wc.get)
        R.add(
            f"fam.worst_case.best_family.{sname}",
            FAM_NAME[best],
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=ph,
            kind="text",
            definition="Family with the highest worst-case retention.",
        )
        if sname == "no_vt10":
            continue
        # contrasts
        ratio_name = {"rf_len": "Reset-free/Standard", "str_len": "Strict/Standard"}
        for lab, x, f1, f2 in (
            ("rf_len", rf_len, "floor", "vt"),
            ("rf_len", rf_len, "floor", "mvt"),
            ("rf_len", rf_len, "vt", "mvt"),
            ("str_len", str_len, "vt", "floor"),
            ("str_len", str_len, "vt", "mvt"),
        ):
            d, p = contrast_perm(x, fam, f1, f2, P)
            R.add(
                f"fam.contrast.{lab}.{f1}_minus_{f2}.{sname}",
                d,
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition=f"{FAM_NAME[f1]} minus {FAM_NAME[f2]} mean {ratio_name[lab]} retention.",
            )
            R.add(
                f"fam.contrast.{lab}.{f1}_minus_{f2}.{sname}.p",
                p,
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                kind="p",
                definition="Two-sided permutation p (labels of the two families permuted across bodies).",
            )
        dd = rf_len - str_len
        d, p = contrast_perm(dd, fam, "floor", "vt", P)
        R.add(
            f"fam.contrast.diff.floor_minus_vt.{sname}",
            d,
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition="Floor minus VT on (Reset-free/Standard minus Strict/Standard).",
        )
        R.add(
            f"fam.contrast.diff.floor_minus_vt.{sname}.p",
            p,
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            kind="p",
            definition="Two-sided permutation p.",
        )
        for f in ("floor", "vt"):
            mu, p = signflip(dd[fam == f], P)
            R.add(
                f"fam.within.{f}.rf_len_minus_str_len.{sname}",
                mu,
                body_set=f"{SET_LABEL[sname]}, {FAM_NAME[f]} only",
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                definition=f"{FAM_NAME[f]} mean of Reset-free/Standard minus Strict/Standard (positive: Strict costs more).",
            )
            R.add(
                f"fam.within.{f}.rf_len_minus_str_len.{sname}.p",
                p,
                body_set=f"{SET_LABEL[sname]}, {FAM_NAME[f]} only",
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=PH_YES,
                kind="p",
                definition="Two-sided sign-flip p.",
            )
        for lab, x in (("rf_len", rf_len), ("str_len", str_len)):
            F, p, pp = oneway_perm(x, fam, P)
            R.add(
                f"fam.oneway.{lab}.{sname}.F",
                F,
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                definition=f"One-way ANOVA F of per-body {ratio_name[lab]} retention across the three families.",
            )
            R.add(
                f"fam.oneway.{lab}.{sname}.p",
                p,
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                kind="p",
                definition="Parametric p.",
            )
            R.add(
                f"fam.oneway.{lab}.{sname}.perm_p",
                pp,
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                kind="p",
                definition="Family-label permutation p.",
            )
    # pairwise p range for Floor minus VT on RF/Len
    ps = [R.get(f"fam.contrast.rf_len.floor_minus_vt.{s}.p")["value"] for s in SUBSETS_MAIN]
    R.add(
        "fam.contrast.rf_len.floor_minus_vt.max_p",
        max(ps),
        body_set="100, 72, 89, 77, 68, 65",
        kind="p",
        evaluator=EV_LABEL[EV_COMMON],
        post_hoc=PH_YES,
        definition="Largest Floor minus VT permutation p over the six body sets.",
    )

    # family-mean bootstrap intervals (Figure 3, right panel)
    for sname in ("all100", "g72"):
        M, fam, rf_len, str_len = family_block(D, V, sets[sname])
        for f in FAMS:
            sel = fam == f
            idx = boot_index(int(sel.sum()), B)
            for lab, x in (("rf_len", rf_len[sel]), ("str_len", str_len[sel])):
                R.add(
                    f"fam.ret_ci.{lab}.{f}.{sname}",
                    float(x.mean()),
                    ci=pct_ci(x[idx].mean(1)),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=PH_YES,
                    definition=f"{FAM_NAME[f]} retention with a 95% bootstrap interval over the family's bodies (Figure 3, right panel).",
                )

    # neutral evaluator and fixed-horizon return
    for ev, metric, tag, evl in (
        (EV_NEUTRAL, "per_step_reward", "neutral", EV_LABEL[EV_NEUTRAL]),
        (EV_COMMON, "fitness", "fixed_horizon", "common, fixed-horizon return (return / 1,000)"),
    ):
        Vx = D.V(ev, metric)
        for sname in ("all100", "g72"):
            M, fam, rf_len, str_len = family_block(D, Vx, sets[sname])
            ph = PH_REV if sname == "all100" else PH_YES
            for f in FAMS:
                sel = fam == f
                R.add(
                    f"fam.{tag}.rf_len.{f}.{sname}",
                    float(rf_len[sel].mean()),
                    body_set=SET_LABEL[sname],
                    evaluator=evl,
                    post_hoc=ph,
                    definition=f"{FAM_NAME[f]} Reset-free/Standard retention.",
                )
                R.add(
                    f"fam.{tag}.str_len.{f}.{sname}",
                    float(str_len[sel].mean()),
                    body_set=SET_LABEL[sname],
                    evaluator=evl,
                    post_hoc=ph,
                    definition=f"{FAM_NAME[f]} Strict/Standard retention.",
                )
            d, p = contrast_perm(rf_len, fam, "floor", "vt", P)
            R.add(
                f"fam.{tag}.rf_len.floor_minus_vt.{sname}",
                d,
                body_set=SET_LABEL[sname],
                evaluator=evl,
                post_hoc=PH_YES,
                definition="Floor minus VT Reset-free/Standard retention.",
            )
            R.add(
                f"fam.{tag}.rf_len.floor_minus_vt.{sname}.p",
                p,
                body_set=SET_LABEL[sname],
                evaluator=evl,
                post_hoc=PH_YES,
                kind="p",
                definition="Two-sided permutation p.",
            )
            F, p0 = stats.f_oneway(*[rf_len[fam == f] for f in FAMS])
            R.add(
                f"fam.{tag}.rf_len.oneway.{sname}.F",
                float(F),
                body_set=SET_LABEL[sname],
                evaluator=evl,
                post_hoc=ph,
                definition=f"One-way ANOVA F of Reset-free/Standard retention, df (2, {len(fam) - 3}).",
            )
            R.add(
                f"fam.{tag}.rf_len.oneway.{sname}.p",
                float(p0),
                body_set=SET_LABEL[sname],
                evaluator=evl,
                post_hoc=ph,
                kind="p",
                definition="Parametric p.",
            )

    # per-seed replication
    for sname in ("all100", "g72"):
        mm = sets[sname]
        fam = D.fam[mm]
        Vs = V[mm]
        ph = PH_REV if sname == "all100" else PH_YES
        order_ok = []
        for j, s in enumerate(SEEDS):
            r = Vs[:, 0, j] / Vs[:, 1, j]
            g = [r[fam == f] for f in FAMS]
            F, p = stats.f_oneway(*g)
            for f, x in zip(FAMS, g):
                R.add(
                    f"fam.per_seed.{s}.{f}.{sname}",
                    float(x.mean()),
                    body_set=SET_LABEL[sname],
                    evaluator=EV_LABEL[EV_COMMON],
                    post_hoc=ph,
                    definition=f"Seed {s}: {FAM_NAME[f]} mean per-body Reset-free/Standard ratio.",
                )
            R.add(
                f"fam.per_seed.{s}.{sname}.p",
                float(p),
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                kind="p",
                definition=f"Seed {s}: one-way ANOVA p of the per-seed ratio across families.",
            )
            order_ok.append(bool(g[0].mean() < g[1].mean()))
        R.add(
            f"fam.per_seed.floor_below_vt_all_seeds.{sname}",
            all(order_ok),
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=ph,
            definition="Floor mean ratio is below VT's in every seed.",
        )
        # balanced subsampling
        M = Vs.mean(-1)
        ratio = M[:, 0] / M[:, 1]
        rng = np.random.default_rng(BOOT_SEED)
        nmin = min(int((fam == f).sum()) for f in FAMS)
        cnt = 0
        for _ in range(1000):
            fl = rng.choice(np.flatnonzero(fam == "floor"), nmin, replace=False)
            vt = rng.choice(np.flatnonzero(fam == "vt"), nmin, replace=False)
            cnt += int(ratio[fl].mean() < ratio[vt].mean())
        R.add(
            f"fam.balanced.floor_lt_vt.{sname}",
            cnt,
            body_set=SET_LABEL[sname],
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=ph,
            definition=f"Of 1,000 balanced subsamples ({nmin} bodies per family), those with Floor mean Reset-free/Standard below VT's.",
        )


def section_interaction(D: Data, R: Registry, args, sets):
    R.section = "Rule x family interaction tests"
    P = args.perm
    V = D.V()
    check_blocked_f(V, D.fam)
    names = {"three": (0, 1, 2), "len_rf": (1, 0), "len_str": (1, 2), "str_rf": (2, 0)}
    lab = {
        "three": "three rules",
        "len_rf": "Standard vs Reset-free",
        "len_str": "Standard vs Strict",
        "str_rf": "Strict vs Reset-free (no Standard)",
    }
    for sname in SUBSETS_MAIN:
        mm = sets[sname]
        fam = D.fam[mm]
        for tname, cols in names.items():
            v = V[mm][:, list(cols), :]
            res = interaction_tests(v, fam, P)
            ph = PH_REV if (sname == "all100" and tname == "three") else PH_YES
            base = f"inter.{tname}.{sname}"
            R.add(
                f"{base}.perseed_F",
                res["perseed_F"],
                body_set=SET_LABEL[sname],
                evaluator=EV_LABEL[EV_COMMON],
                post_hoc=ph,
                dec=2,
                note=f"df {res['perseed_df']}",
                definition=f"Per-seed blocked ANOVA F for rule x family ({lab[tname]}); seeds treated as independent (supplementary).",
            )
            R.add(
                f"{base}.perseed_df",
                res["perseed_df"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                dec=0,
                definition="Its degrees of freedom.",
            )
            R.add(
                f"{base}.perseed_p",
                res["perseed_p"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its parametric p.",
            )
            R.add(
                f"{base}.seedmean_F",
                res["seedmean_F"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                dec=2,
                evaluator=EV_LABEL[EV_COMMON],
                note=f"df {res['seedmean_df']}",
                definition="Blocked ANOVA F on three-seed means (one row per body and rule).",
            )
            R.add(
                f"{base}.seedmean_p",
                res["seedmean_p"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its parametric p.",
            )
            R.add(
                f"{base}.cr1_F",
                res["cr1_F"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                dec=2,
                evaluator=EV_LABEL[EV_COMMON],
                note=f"df {res['cr1_df']}",
                definition="Body-clustered (CR1) Wald F for the rule x family terms on per-seed rows.",
            )
            R.add(
                f"{base}.cr1_df",
                res["cr1_df"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                dec=0,
                definition="Its degrees of freedom.",
            )
            R.add(
                f"{base}.cr1_p",
                res["cr1_p"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its p.",
            )
            R.add(
                f"{base}.perm_p",
                res["perm_p"],
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Body-level permutation p: family labels permuted across bodies, statistic the per-seed blocked F.",
            )
    pl = [R.get(f"inter.len_rf.{s}.perm_p")["value"] for s in SUBSETS_MAIN]
    R.add(
        "inter.len_rf.max_perm_p",
        max(pl),
        body_set="100, 72, 89, 77, 68, 65",
        kind="p",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Largest Standard vs Reset-free x family permutation p over the six sets.",
    )
    ps = [R.get(f"inter.len_str.{s}.perm_p")["value"] for s in SUBSETS_MAIN]
    R.add(
        "inter.len_str.min_perm_p",
        min(ps),
        body_set="100, 72, 89, 77, 68, 65",
        kind="p",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Smallest Standard vs Strict x family permutation p over the six sets.",
    )
    R.add(
        "inter.three.all100.inflation",
        R.get("inter.three.all100.perseed_F")["value"] / R.get("inter.three.all100.cr1_F")["value"],
        body_set="100",
        dec=2,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Per-seed F over the body-clustered Wald F.",
    )

    # LOFO two-family tests, mixed model, GEE, residual MS ratio, rank F
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.generalized_estimating_equations import GEE

    for sname in ("all100", "g72"):
        mm = sets[sname]
        fam = D.fam[mm]
        Vs = V[mm]
        ph = PH_REV if sname == "all100" else PH_YES
        for drop, tag in (("mvt", "floor_vt"), ("floor", "vt_mvt"), ("vt", "floor_mvt")):
            keep = fam != drop
            res = interaction_tests(Vs[keep], fam[keep], P, want_cr1=False)
            R.add(
                f"inter.lofo.{tag}.{sname}.F",
                res["perseed_F"],
                body_set=f"{SET_LABEL[sname]}, without {FAM_NAME[drop]}",
                dec=2,
                post_hoc=ph,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Two-family per-seed blocked F for rule x family.",
            )
            R.add(
                f"inter.lofo.{tag}.{sname}.df",
                res["perseed_df"],
                body_set=f"{SET_LABEL[sname]}, without {FAM_NAME[drop]}",
                dec=0,
                post_hoc=ph,
                definition="Its degrees of freedom.",
            )
            R.add(
                f"inter.lofo.{tag}.{sname}.p",
                res["perseed_p"],
                body_set=f"{SET_LABEL[sname]}, without {FAM_NAME[drop]}",
                kind="p",
                post_hoc=ph,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its parametric p (seeds as independent).",
            )
            R.add(
                f"inter.lofo.{tag}.{sname}.perm_p",
                res["perm_p"],
                body_set=f"{SET_LABEL[sname]}, without {FAM_NAME[drop]}",
                kind="p",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Body-level permutation p for the two-family interaction.",
            )
        rows = []
        bnames = np.array(D.bodies)[mm]
        for i in range(len(bnames)):
            for j, r in enumerate(RULES):
                for k, s in enumerate(SEEDS):
                    rows.append(dict(psr=Vs[i, j, k], protocol=r, family=fam[i], morph=bnames[i], seed=s))
        df = pd.DataFrame(rows)
        dfa = df.groupby(["protocol", "family", "morph"], as_index=False)["psr"].mean()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_mix = "psr ~ C(protocol, Treatment('len')) * C(family, Treatment('floor'))"
            for tag, data in (("perseed", df), ("seedmean", dfa)):
                fit = smf.mixedlm(f_mix, data, groups=data.morph).fit(reml=True)
                ci = fit.conf_int()
                kvt = [t for t in fit.fe_params.index if "T.rf" in t and "T.vt" in t and ":" in t][0]
                krf = [t for t in fit.fe_params.index if "T.rf" in t and ":" not in t][0]
                phm = ph if tag == "perseed" else PH_YES
                R.add(
                    f"inter.mixed.{tag}.vt_x_rf.{sname}",
                    float(fit.params[kvt]),
                    ci=[float(x) for x in ci.loc[kvt]],
                    body_set=SET_LABEL[sname],
                    post_hoc=phm,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Random-intercept model ({tag.replace('perseed', 'per-seed rows').replace('seedmean', 'seed means')}): VT x Reset-free coefficient with 95% Wald interval.",
                )
                R.add(
                    f"inter.mixed.{tag}.rf_main_floor.{sname}",
                    float(fit.params[krf]),
                    ci=[float(x) for x in ci.loc[krf]],
                    body_set=SET_LABEL[sname],
                    post_hoc=phm,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition="Reset-free main effect, which is the Floor (reference family) Reset-free minus Standard difference.",
                )
            M = Vs.mean(-1)
            R.add(
                f"inter.rf_minus_len.average.{sname}",
                float((M[:, 0] - M[:, 1]).mean()),
                body_set=SET_LABEL[sname],
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Average over bodies of Reset-free minus Standard three-seed mean PSR.",
            )
            gee = GEE.from_formula("psr ~ C(protocol) * C(family)", groups="morph", data=df, cov_struct=Exchangeable()).fit()
            icg = [c for c in gee.params.index if ":" in c]
            Rg = np.zeros((len(icg), len(gee.params)))
            for i, c in enumerate(icg):
                Rg[i, list(gee.params.index).index(c)] = 1
            wg = gee.wald_test(Rg, use_f=False, scalar=True)
            R.add(
                f"inter.gee.{sname}.chi2",
                float(wg.statistic),
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                dec=2,
                evaluator=EV_LABEL[EV_COMMON],
                definition="GEE (exchangeable, clustered by body) Wald chi-square, 4 df, for rule x family.",
            )
            R.add(
                f"inter.gee.{sname}.p",
                float(wg.pvalue),
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its p.",
            )
            full = "psr ~ C(morph) + C(protocol) + C(protocol):C(family)"
            m1 = smf.ols(full, data=df).fit()
            m2 = smf.ols(full, data=dfa).fit()
        R.add(
            f"inter.ms_ratio_seedmean_perseed.{sname}",
            float(m2.mse_resid / m1.mse_resid),
            body_set=SET_LABEL[sname],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Residual mean square of the seed-mean model over that of the per-seed model.",
        )
        rk = stats.rankdata(Vs, axis=0)
        Fr, qr, dfr = BlockedF(rk).F(fam)
        R.add(
            f"inter.rank_perseed.{sname}.F",
            Fr,
            body_set=SET_LABEL[sname],
            post_hoc=ph,
            dec=2,
            evaluator=EV_LABEL[EV_COMMON],
            note=f"df [{qr}, {dfr}]",
            definition="Per-seed blocked F on ranks within each rule and seed column.",
        )


def section_selection(D: Data, R: Registry, args, sets, extra):
    R.section = "Selection audit (in sample, held-out seed, leave-one-rule-out)"
    B = args.boot
    V = D.V()
    sel_name = {
        "len_only": "Standard only",
        "str_only": "Strict only",
        "rf_only": "Reset-free only",
        "mean3": "mean of the three rules",
        "min3": "minimum over the three rules (maximin)",
        "min_str_rf": "minimum of Strict and Reset-free",
    }
    for sname in ("all100", "g72", "s89", "s77", "s65"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        fam = D.fam[mm]
        n = len(M)
        ph = PH_REV if sname == "all100" else PH_YES
        sc = selectors(M)
        for sn, s in sc.items():
            worst = {}
            for k in KS:
                vals = []
                for i, r in enumerate(RULES):
                    x = retention(s, M[:, i], k)
                    vals.append(x)
                    if sn in ("len_only", "min3", "mean3") or (sn == "str_only" and k == 5):
                        R.add(
                            f"sel.in.{sn}.{r}.k{k}.{sname}",
                            100 * x,
                            body_set=SET_LABEL[sname],
                            dec=1,
                            post_hoc=ph,
                            evaluator=EV_LABEL[EV_COMMON],
                            definition=f"In sample: selecting the top {k} by {sel_name[sn]} keeps this percentage of the {RULE_NAME[r]} oracle top-{k} mean PSR.",
                        )
                worst[k] = min(vals)
                R.add(
                    f"sel.in.{sn}.worst.k{k}.{sname}",
                    100 * worst[k],
                    body_set=SET_LABEL[sname],
                    dec=1,
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"In sample, {sel_name[sn]}: smallest retention over the three oracles at k = {k} (percent).",
                )
        idx = boot_index(n, B)
        Mb = M[idx]  # B x n x 3
        lows = {}
        for sn in ("min3", "mean3", "len_only"):
            scb = {"min3": Mb.min(-1), "mean3": Mb.mean(-1), "len_only": Mb[:, :, 1]}[sn]
            lo_all = []
            for i, r in enumerate(RULES):
                for k in KS:
                    bt = retention_batch(scb, Mb[:, :, i], k)
                    ci = pct_ci(bt)
                    lo_all.append(ci[0])
                    if sn == "min3" or (sn == "len_only" and r == "str"):
                        R.add(
                            f"sel.in_ci.{sn}.{r}.k{k}.{sname}",
                            100 * retention(sc[sn], M[:, i], k),
                            ci=[100 * ci[0], 100 * ci[1]],
                            body_set=SET_LABEL[sname],
                            dec=1,
                            post_hoc=ph,
                            evaluator=EV_LABEL[EV_COMMON],
                            definition="In-sample retention with a 95% body-bootstrap interval (which bodies are in the corpus, not seed noise).",
                        )
            lows[sn] = min(lo_all)
            if sn != "len_only":
                R.add(
                    f"sel.in_ci.{sn}.lowest_lower_bound.{sname}",
                    100 * lows[sn],
                    body_set=SET_LABEL[sname],
                    dec=1,
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Smallest 2.5% bootstrap bound over the 9 oracle x k cells for {sel_name[sn]}.",
                )
        for sn in ("min3", "len_only"):
            t10 = np.argsort(-sc[sn], kind="stable")[:10]
            for f in FAMS:
                R.add(
                    f"sel.top10_family.{sn}.{f}.{sname}",
                    int((fam[t10] == f).sum()),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} bodies among the top 10 by {sel_name[sn]}.",
                )
            if sname == "all100":
                inert = extra["inert20"][t10]
                R.add(
                    f"sel.top10_inert.{sn}.{sname}",
                    int(inert.sum()),
                    body_set="100",
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Standard-inert bodies among the top 10 by {sel_name[sn]}.",
                )
                R.add(
                    f"sel.top10_inert_mvt.{sn}.{sname}",
                    int((inert & (fam[t10] == "mvt")).sum()),
                    body_set="100",
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Standard-inert MVT bodies among the top 10 by {sel_name[sn]}.",
                )
        # held-out seed
        Vs = V[mm]
        for k in KS:
            ret = heldout_batch(Vs[None], k)[0]
            worst = ret.min(axis=-1)
            for j, sn in enumerate(SEL5):
                if sn in ("min", "str", "len", "mean"):
                    R.add(
                        f"sel.heldout.{sn}.k{k}.{sname}",
                        100 * float(worst[j]),
                        body_set=SET_LABEL[sname],
                        dec=1,
                        post_hoc=PH_YES,
                        evaluator=EV_LABEL[EV_COMMON],
                        definition=f"Held-out seed: choose on two seeds, score on the third; fold-averaged retention, smallest over the three rules (selector {sn}, k = {k}, percent).",
                    )
            bw = []
            for st in range(0, B, 250):
                bw.append(heldout_batch(Vs[idx[st : st + 250]], k).min(axis=-1))
            bw = np.concatenate(bw)
            for a_, b_ in (("min", "str"), ("min", "len"), ("min", "mean")):
                ia, ib = SEL5.index(a_), SEL5.index(b_)
                d = 100 * (bw[:, ia] - bw[:, ib])
                R.add(
                    f"sel.heldout.diff.{a_}_minus_{b_}.k{k}.{sname}",
                    100 * float(worst[ia] - worst[ib]),
                    ci=pct_ci(d),
                    body_set=SET_LABEL[sname],
                    dec=1,
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Held-out seed, {a_} minus {b_} worst-case retention (points) with paired body-bootstrap interval.",
                )

    # Standard-only vs Strict oracle under the neutral evaluator
    Vn = D.V(EV_NEUTRAL)
    for sname in ("all100", "g72"):
        M = Vn[sets[sname]].mean(-1)
        R.add(
            f"sel.neutral.len_only.str.k5.{sname}",
            100 * retention(M[:, 1], M[:, 2], 5),
            body_set=SET_LABEL[sname],
            dec=1,
            post_hoc=PH_YES if sname != "all100" else PH_REV,
            evaluator=EV_LABEL[EV_NEUTRAL],
            definition="Standard-only top-5 selection keeps this percentage of the Strict oracle top-5 mean PSR (neutral evaluator).",
        )

    # anatomy of the 49.0%
    M = V.mean(-1)
    t5 = np.argsort(-M[:, 1], kind="stable")[:5]
    rank_str = {i: r + 1 for r, i in enumerate(np.argsort(-M[:, 2], kind="stable"))}
    R.add(
        "sel.len_top5.n_inert",
        int(extra["inert20"][t5].sum()),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Standard-inert bodies among Standard's top 5.",
    )
    R.add(
        "sel.len_top5.strict_ranks",
        sorted(rank_str[i] for i in t5),
        body_set="100",
        post_hoc=PH_YES,
        dec=0,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Strict ranks of Standard's top-5 bodies (1 = best).",
    )
    worst_i = t5[np.argmin(M[t5, 2])]
    R.add(
        "sel.len_top5.lowest_strict_body",
        D.bodies[worst_i],
        body_set="100",
        post_hoc=PH_YES,
        kind="text",
        evaluator=EV_LABEL[EV_COMMON],
        definition="Standard top-5 body with the lowest Strict PSR (largest shortfall).",
    )
    R.add(
        "sel.len_top5.lowest_strict_body_is_cut11",
        bool(extra["cut11"][worst_i]),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="That body is a cut-off body.",
    )
    R.add(
        "sel.len_top5.ids",
        [D.bodies[i] for i in t5],
        body_set="100",
        post_hoc=PH_YES,
        kind="ids",
        evaluator=EV_LABEL[EV_COMMON],
        definition="Standard's top-5 bodies.",
    )

    # leave-one-rule-out (planned test, frozen before submission)
    R.section = "Leave-one-rule-out selection test (planned, failed)"
    M = V.mean(-1)
    scores = {r: M[:, i] for i, r in enumerate(RULES)}
    n = len(M)
    robust = ("RobustMin", "RobustMean", "Borda", "MinimaxRank")
    pair = {}
    for hold in RULES:
        train = [r for r in RULES if r != hold]
        rules = lopo_rules(scores, train)
        tgt = scores[hold]
        for k in (5, 10):
            for rn, s in rules.items():
                R.add(
                    f"lopo.{hold}_held.{rn}.k{k}",
                    100 * topk_mean(s, tgt, k) / np.sort(tgt)[::-1][:k].mean(),
                    body_set="100",
                    dec=1,
                    post_hoc=PH_PLAN,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{RULE_NAME[hold]} held out: rule {rn} built from the other two rules keeps this percentage of the held-out top-{k} mean PSR.",
                )
        rs = {r: topk_mean(rules[r], tgt, 5) for r in robust}
        ns = {f"by_{p}": topk_mean(rules[f"by_{p}"], tgt, 5) for p in train}
        br, bn = max(rs, key=rs.get), max(ns, key=ns.get)
        pair[hold] = (br, bn, rs[br] - ns[bn])
    diffs = [pair[h][2] for h in RULES]
    rng = np.random.default_rng(BOOT_SEED)
    bidx = rng.integers(0, n, size=(args.boot, n))
    boot = np.empty(args.boot)
    for i in range(args.boot):
        ii = bidx[i]
        per = []
        for hold in RULES:
            train = [r for r in RULES if r != hold]
            rb = lopo_rules({p: scores[p][ii] for p in RULES}, train)
            tb = scores[hold][ii]
            br, bn, _ = pair[hold]
            per.append(topk_mean(rb[br], tb, 5) - topk_mean(rb[bn], tb, 5))
        boot[i] = np.mean(per)
    R.add(
        "lopo.gate.mean_gain_k5",
        float(np.mean(diffs)),
        ci=pct_ci(boot),
        body_set="100",
        post_hoc=PH_PLAN,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Mean over the three held-out rules of (best robust rule minus best single rule) top-5 PSR, with paired bootstrap interval; the gate needed at least 0.05 with the interval above 0.",
    )
    R.add(
        "lopo.gate.wins_k5",
        int(sum(d > 0 for d in diffs)),
        body_set="100",
        post_hoc=PH_PLAN,
        definition="Held-out rules where the best robust rule beat the best single rule at k = 5.",
    )
    sec = max(min(R.get(f"lopo.{h}_held.{r}.k10")["value"] for h in RULES) for r in robust)
    R.add(
        "lopo.gate.best_min_retention_k10",
        sec,
        body_set="100",
        dec=1,
        post_hoc=PH_PLAN,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Best robust rule's smallest k = 10 retention over the three splits; the secondary gate needed 85.",
    )
    R.add(
        "lopo.str_held.best_robust_k10",
        max(R.get(f"lopo.str_held.{r}.k10")["value"] for r in robust),
        body_set="100",
        dec=1,
        post_hoc=PH_PLAN,
        evaluator=EV_LABEL[EV_COMMON],
        definition="With Strict held out, the best robust rule's k = 10 retention.",
    )
    for h in RULES:
        R.add(
            f"lopo.{h}_held.best_pair_k5",
            f"{pair[h][0]} vs {pair[h][1]}: {pair[h][2]:+.3f} PSR",
            body_set="100",
            post_hoc=PH_PLAN,
            kind="text",
            definition="Best robust and best single rule at k = 5 and their PSR difference.",
        )


def repeat_block(D, R, args, mask, sname):
    B = args.boot
    T = {t: D.get(t) for t in SWEEP}
    idx_all = np.where(mask)[0]
    n = len(idx_all)
    bidx = boot_index(n, B)
    Mt = {t: T[t][mask].mean(1) for t in SWEEP}

    def boot_rho(a, b):
        return spearman_last(a[bidx], b[bidx])

    pl = boot_rho(Mt["0p00"], Mt["0p10"])
    rs = boot_rho(Mt["0p00"], Mt["0p50"])
    lab = SET_LABEL[sname]
    R.add(
        f"repeat.{sname}.rho_tau0_tau010",
        rho(Mt["0p00"], Mt["0p10"]),
        ci=pct_ci(pl),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="The repeat: Spearman rho between three-seed rankings after tau = 0 and tau = 0.10 training (same seeds, separate launches).",
    )
    R.add(
        f"repeat.{sname}.rho_rf_str",
        rho(Mt["0p00"], Mt["0p50"]),
        ci=pct_ci(rs),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Strict vs Reset-free rho on the same bodies.",
    )
    R.add(
        f"repeat.{sname}.diff",
        rho(Mt["0p00"], Mt["0p10"]) - rho(Mt["0p00"], Mt["0p50"]),
        ci=pct_ci(pl - rs),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Repeat rho minus Strict vs Reset-free rho, paired bootstrap interval.",
    )
    for a, b, tag in (
        ("0p20", "0p00", "len_rf"),
        ("0p20", "0p50", "len_str"),
        ("0p00", "0p35", "rf_t035"),
        ("0p10", "0p50", "t010_str"),
    ):
        R.add(
            f"repeat.{sname}.rho_{tag}",
            rho(Mt[a], Mt[b]),
            ci=pct_ci(boot_rho(Mt[a], Mt[b])),
            body_set=lab,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"Three-seed rho, tau = {a.replace('p', '.')} vs tau = {b.replace('p', '.')}.",
        )
    R.add(
        f"repeat.{sname}.top10_repeat",
        topk_overlap(Mt["0p00"], Mt["0p10"], 10),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Top-10 overlap of the two repeat rankings.",
    )
    R.add(
        f"repeat.{sname}.top10_rf_str",
        topk_overlap(Mt["0p00"], Mt["0p50"], 10),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Top-10 overlap of Strict and Reset-free rankings.",
    )
    A, Bm = T["0p00"][mask], T["0p10"][mask]
    same = np.mean([rho(A[:, s], Bm[:, s]) for s in range(3)])
    rAB = np.mean([rho(A[:, s], Bm[:, t]) for s in range(3) for t in range(3) if s != t])
    rAA = np.mean([rho(A[:, s], A[:, t]) for s, t in itertools.combinations(range(3), 2)])
    rBB = np.mean([rho(Bm[:, s], Bm[:, t]) for s, t in itertools.combinations(range(3), 2)])
    R.add(
        f"repeat.{sname}.single_same_seed",
        float(same),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Mean single-run rho between the two arms for the same seed.",
    )
    R.add(
        f"repeat.{sname}.single_diff_seed",
        float(rAB),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Mean single-run rho between the two arms for different seeds.",
    )
    R.add(
        f"repeat.{sname}.independent_seed_estimate",
        float(3 * rAB / math.sqrt((1 + 2 * rAA) * (1 + 2 * rBB))),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Expected three-seed agreement with independent seeds, 3 r_diff / sqrt((1 + 2 W_tau0)(1 + 2 W_tau0.10)).",
    )
    S = T["0p50"][mask]
    rSS = np.mean([rho(S[:, s], S[:, t]) for s, t in itertools.combinations(range(3), 2)])
    ref = math.sqrt((3 * rAA / (1 + 2 * rAA)) * (3 * rSS / (1 + 2 * rSS)))
    R.add(
        f"repeat.{sname}.strict_rf_noise_ref",
        ref,
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Strict vs Reset-free noise-only reference on these bodies, sqrt(R3_RF R3_Strict).",
    )
    R.add(
        f"repeat.{sname}.W_rf",
        float(rAA),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Single-seed Reset-free agreement W on these bodies.",
    )
    R.add(
        f"repeat.{sname}.W_str",
        float(rSS),
        body_set=lab,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Single-seed Strict agreement W on these bodies.",
    )


def section_repeat(D: Data, R: Registry, args, sets):
    R.section = "Same-rule repeat (tau = 0 vs tau = 0.10, post hoc)"
    for sname in ("probe_safe60", "inert10_94", "probe_safe67", "clean59"):
        if sname in sets:
            repeat_block(D, R, args, sets[sname], sname)
    T0, T1 = D.get("0p00"), D.get("0p10")
    for b in ("vt-5506-15-9-02-19-03-39", "vt-1409-15-5-02-21-44-24"):
        i = D.bodies.index(b)
        R.add(
            f"repeat.acting.{b}.tau0",
            T0[i].tolist(),
            body_set="1",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Per-seed PSR (seeds 42, 43, 44) after tau = 0 training, a VT body where tau = 0.10 changed training.",
        )
        R.add(
            f"repeat.acting.{b}.tau010",
            T1[i].tolist(),
            body_set="1",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Per-seed PSR after tau = 0.10 training.",
        )
    m = sets["inert10_94"]
    d = T1[m].mean(1) - T0[m].mean(1)
    R.add(
        "sweep.tau010_minus_tau0.mean",
        float(d.mean()),
        body_set=SET_LABEL["inert10_94"],
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Mean paired difference of three-seed PSR, tau = 0.10 minus tau = 0.",
    )
    R.add(
        "sweep.tau010_minus_tau0.p",
        float(stats.ttest_rel(T1[m].mean(1), T0[m].mean(1)).pvalue),
        body_set=SET_LABEL["inert10_94"],
        post_hoc=PH_YES,
        kind="p",
        evaluator=EV_LABEL[EV_COMMON],
        definition="Paired t-test p.",
    )


def section_sweep(D: Data, R: Registry, args, sets, extra):
    R.section = "Five-threshold sweep"
    T = {t: D.get(t).mean(1) for t in SWEEP}
    for sname in ("all100", "g72"):
        mm = sets[sname]
        for t in SWEEP:
            for f in FAMS:
                sel = mm & (D.fam == f)
                R.add(
                    f"sweep.fam_mean.{t}.{f}.{sname}",
                    float(T[t][sel].mean()),
                    body_set=SET_LABEL[sname],
                    post_hoc=PH_REV if sname == "all100" else PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} mean three-seed PSR after tau = {t.replace('p', '.')} training.",
                )
    vt10 = extra["vt10"]
    z30 = np.isclose(D.z0, 0.30)
    sel = vt10 & z30
    R.add(
        "sweep.tau035.vt10_short.psr",
        [float(x) for x in T["0p35"][sel]],
        body_set="7 short VT cut-off bodies",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Three-seed PSR after tau = 0.35 training.",
    )
    R.add(
        "sweep.tau035.vt10_short.n_near_zero",
        int((np.abs(T["0p35"][sel]) < 0.2).sum()),
        body_set="7 short VT cut-off bodies",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Of these, bodies whose three-seed PSR after tau = 0.35 training is near 0 (|PSR| below 0.2, under a tenth of the corpus Standard mean of about 2.4).",
    )
    vt = D.fam == "vt"
    r35 = T["0p35"] / T["0p20"]
    R.add(
        "sweep.tau035.vt_ratio.all",
        float(r35[vt].mean()),
        body_set="37 VT",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="VT mean per-body ratio of tau = 0.35 over Standard three-seed PSR.",
    )
    R.add(
        "sweep.tau035.vt_ratio.no_vt10",
        float(r35[vt & ~vt10].mean()),
        body_set="27 VT",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Same without the 10 VT cut-off bodies.",
    )


def section_exemplars(D: Data, R: Registry, args, sets, extra):
    R.section = "Illustrative bodies (recorded corpus runs)"
    M = D.V().mean(-1)
    rank = {r: {i: k + 1 for k, i in enumerate(np.argsort(-M[:, j], kind="stable"))} for j, r in enumerate(RULES)}
    dist = {r: D.get(TAU_DIR[r], EV_NEUTRAL, "distance") for r in RULES}
    for alias, b in (("floor_c", "floor-1409-11-14-01-13-30-11"), ("floor_b", "floor-1409-0-7-01-15-15-26")):
        i = D.bodies.index(b)
        for r in RULES:
            R.add(
                f"ex.{alias}.rank.{r}",
                rank[r][i],
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"{b}: rank among 100 by three-seed mean PSR after {RULE_NAME[r]} training (1 = best).",
            )
            R.add(
                f"ex.{alias}.dist.{r}",
                dist[r][i].tolist(),
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_NEUTRAL],
                dec=1,
                definition=f"{b}: mean distance (m) per seed (42, 43, 44) over 20 neutral-evaluator episodes after {RULE_NAME[r]} training.",
            )
        R.add(f"ex.{alias}.z0", float(D.z0[i]), body_set="1", dec=2, post_hoc=PH_YES, definition="Reset height (m).")
        R.add(f"ex.{alias}.in_g72", bool(sets["g72"][i]), body_set="1", post_hoc=PH_YES, definition="In the 72.")
        R.add(
            f"ex.{alias}.in_cut23",
            bool(extra["cut23"][i]),
            body_set="1",
            post_hoc=PH_YES,
            definition="In a cut-off set (23).",
        )
    later = D.results / "asymptotic_5m" / "floor-1409-11-14-01-13-30-11"
    if later.exists():
        vals = []
        for s in SEEDS:
            d = json.loads((later / f"tau0p00_seed{s}" / "results.json").read_text())
            rec = d["floor-1409-11-14-01-13-30-11"]
            if rec["threshold"] != 0.0 or rec["seed"] != s or rec.get("termination_mode", "height") != "height":
                raise ValueError("unexpected later-code record for Floor C")
            vals.append(rec[EV_NEUTRAL]["distance"])
        R.add(
            "ex.floor_c.later_code_rf_dist",
            vals,
            body_set="1",
            dec=2,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_NEUTRAL],
            definition="Floor C mean distance per seed (m) after Reset-free retraining with later code (asymptotic_5m panel, 5M steps).",
        )
    fb = D.bodies.index("floor-1409-0-7-01-15-15-26")
    allb = np.concatenate([dist[r][fb] for r in RULES])
    R.add(
        "ex.floor_b.dist_range",
        [float(allb.min()), float(allb.max())],
        body_set="1",
        dec=1,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Range of Floor B's per-seed mean distance over all 9 runs (m).",
    )
    fails = []
    for i in range(D.n):
        if (dist["rf"][i] < 5).all() and (dist["len"][i] >= 20).all() and (dist["str"][i] >= 20).all():
            fails.append(i)
    R.add(
        "ex.rf_failures.ids",
        [D.bodies[i] for i in fails],
        body_set="100",
        post_hoc=PH_YES,
        kind="ids",
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Bodies that move less than 5 m after Reset-free training in every seed and at least 20 m after Standard and after Strict training in every seed.",
    )
    R.add(
        "ex.rf_failures.n",
        len(fails),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Number of such bodies.",
    )
    R.add(
        "ex.rf_failures.len_ranks",
        [rank["len"][i] for i in fails],
        body_set="100",
        post_hoc=PH_YES,
        dec=0,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Their Standard ranks.",
    )
    R.add(
        "ex.rf_failures.best_by_len_rank",
        D.bodies[min(fails, key=lambda i: rank["len"][i])] if fails else None,
        body_set="100",
        post_hoc=PH_YES,
        kind="text",
        evaluator=EV_LABEL[EV_COMMON],
        definition="Highest-ranked of them by Standard rank.",
    )


def section_distance(D: Data, R: Registry, args, sets, extra):
    R.section = "Distance, episode length and control effort"
    for ev, tag in ((EV_COMMON, "common"), (EV_NEUTRAL, "neutral")):
        Dd = D.V(ev, "distance").mean(-1)
        for sname in ("all100", "g72"):
            mm = sets[sname]
            ph = PH_REV if sname == "all100" else PH_YES
            for f in FAMS:
                sel = mm & (D.fam == f)
                R.add(
                    f"dist.{tag}.rf_stop.{f}.{sname}",
                    int((sel & (Dd[:, 1] >= 5) & (Dd[:, 0] < 1)).sum()),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[ev],
                    definition=f"{FAM_NAME[f]} bodies walking at least 5 m after Standard training but less than 1 m after Reset-free training (three-seed mean distance).",
                )
                R.add(
                    f"dist.{tag}.str_stop.{f}.{sname}",
                    int((sel & (Dd[:, 1] >= 5) & (Dd[:, 2] < 1)).sum()),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[ev],
                    definition=f"{FAM_NAME[f]} bodies walking at least 5 m after Standard training but less than 1 m after Strict training.",
                )
        if tag == "common":
            vtstop = (D.fam == "vt") & (Dd[:, 1] >= 5) & (Dd[:, 2] < 1)
            R.add(
                "dist.common.vt_str_stop.n_inert",
                int((vtstop & extra["inert20"]).sum()),
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="VT Strict stops that are Standard-inert bodies.",
            )
            R.add(
                "dist.common.vt_str_stop.n_cut11",
                int((vtstop & extra["cut11"]).sum()),
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="VT Strict stops that are cut-off bodies (the 11).",
            )
            R.add(
                "dist.common.vt_str_stop.n_cut23",
                int((vtstop & extra["cut23"]).sum()),
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="VT Strict stops that are cut-off-seed bodies (the 23).",
            )
            R.add(
                "dist.common.vt10_not_str_stop",
                [D.bodies[i] for i in np.where(extra["vt10"] & ~vtstop)[0]],
                body_set="10",
                post_hoc=PH_YES,
                kind="ids",
                evaluator=EV_LABEL[EV_COMMON],
                definition="VT cut-off bodies that are not Strict stops (they move at least 1 m after Strict training).",
            )
            floor_stop = (D.fam == "floor") & (Dd[:, 1] >= 5) & (Dd[:, 0] < 1)
    Dn = D.V(EV_NEUTRAL, "distance").mean(-1)
    R.add(
        "dist.floor_rf_stops.neutral_ge5",
        int((floor_stop & (Dn[:, 0] >= 5)).sum()),
        body_set="12 Floor Reset-free stops",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Of the Floor Reset-free stops (common evaluator), bodies walking at least 5 m under the neutral evaluator after Reset-free training.",
    )
    R.add(
        "dist.floor_rf_stops.neutral_min",
        float(Dn[floor_stop, 0].min()),
        body_set="12 Floor Reset-free stops",
        dec=2,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Smallest neutral-evaluator Reset-free distance among them (m).",
    )
    R.add(
        "dist.floor_c.rf_neutral_mean",
        float(Dn[D.bodies.index("floor-1409-11-14-01-13-30-11"), 0]),
        body_set="1",
        dec=2,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Floor C three-seed mean Reset-free distance (m).",
    )
    fr = D.V(EV_COMMON, "fall_rate").mean(-1)
    el = D.V(EV_COMMON, "mean_ep_len").mean(-1)
    for sname, lab in (("all100", "41 Floor"), ("g72", "33 Floor in the 72")):
        sel = sets[sname] & (D.fam == "floor")
        ph = PH_REV if sname == "all100" else PH_YES
        R.add(
            f"dist.floor.rf_fall_rate.{sname}",
            float(fr[sel, 0].mean()),
            body_set=lab,
            post_hoc=ph,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Mean fall rate of Reset-free-trained Floor controllers.",
        )
        for j, r in enumerate(RULES):
            R.add(
                f"dist.floor.ep_len.{r}.{sname}",
                float(el[sel, j].mean()),
                body_set=lab,
                post_hoc=ph,
                dec=1,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Mean episode length of {RULE_NAME[r]}-trained Floor controllers.",
            )
            R.add(
                f"dist.floor.neutral_dist.{r}.{sname}",
                float(Dn[sel, j].mean()),
                body_set=lab,
                post_hoc=ph,
                dec=1,
                evaluator=EV_LABEL[EV_NEUTRAL],
                definition=f"Mean distance per 1,000-step episode of {RULE_NAME[r]}-trained Floor controllers (m).",
            )
    # control effort: sum ||a||^2 per episode = (distance / dt - return) / 0.001, dt = 0.02 s
    for ev in (EV_NEUTRAL, EV_COMMON):
        dist_ = D.V(ev, "distance")
        fit_ = D.V(ev, "fitness")
        len_ = D.V(ev, "mean_ep_len")
        a2 = (dist_ / 0.02 - fit_) / 1e-3 / len_
        cells = [float(a2[D.fam == f][:, j, :].mean()) for f in FAMS for j in range(3)]
        R.add(
            f"dist.action_sq_per_step.range.{'neutral' if ev == EV_NEUTRAL else 'common'}",
            [min(cells), max(cells)],
            body_set="100",
            dec=2,
            post_hoc=PH_REV,
            evaluator=EV_LABEL[ev],
            definition="Range over the 9 family x rule cells of the mean per-step squared action norm, (distance/0.02 minus return)/0.001 over episode length (ratio of episode means).",
        )


def section_panels(D: Data, R: Registry, args, sets, extra):
    res = D.results
    # ---------------- SAC panel
    R.section = "SAC panel (24 bodies, planned)"
    sac_list = Path(args.sac_bodies)
    if sac_list.exists():
        bodies = sorted(l.strip().removesuffix(".xml") for l in sac_list.read_text().splitlines() if l.strip())
        fam = np.array([b.split("-")[0] for b in bodies])

        def load(prefix):
            A = np.empty((len(bodies), 3, 3))
            for ri, r in enumerate(RULES):
                for si, s in enumerate(SEEDS):
                    d = json.loads((res / f"{prefix}_{TAU_DIR[r]}_seed{s}" / "results.json").read_text())
                    for bi, b in enumerate(bodies):
                        A[bi, ri, si] = d[b][EV_COMMON]["per_step_reward"]
            return A

        SAC, PPO = load("sac_mjx"), load("mjx_ppo")
        inert = np.array([D.audit[b]["geometric_inert"]["0.20"] for b in bodies])
        R.add("sac.n_bodies", len(bodies), body_set="24 (SAC panel)", post_hoc=PH_PLAN, definition="SAC panel bodies.")
        for f in FAMS:
            R.add(
                f"sac.n.{f}",
                int((fam == f).sum()),
                body_set="24 (SAC panel)",
                post_hoc=PH_PLAN,
                definition=f"{FAM_NAME[f]} bodies.",
            )
        R.add(
            "sac.n_inert",
            int(inert.sum()),
            body_set="24 (SAC panel)",
            post_hoc=PH_YES,
            definition="Standard-inert bodies in the panel.",
        )
        for name, A in (("sac", SAC), ("ppo24", PPO)):
            # order rules as (len, str, rf) to match the planned analysis' reference level; F does not depend on it
            v = A[:, [1, 2, 0], :]
            F, q, df = BlockedF(v).F(fam)
            ph = PH_PLAN if name == "sac" else PH_YES
            R.add(
                f"sac.{name}.perrun_F",
                F,
                body_set="24 (SAC panel)",
                dec=2,
                post_hoc=ph,
                evaluator=EV_LABEL[EV_COMMON],
                note=f"df [{q}, {df}]",
                definition="Per-run blocked rule x family F (the planned SAC gate statistic for SAC).",
            )
            R.add(
                f"sac.{name}.perrun_p",
                float(stats.f.sf(F, q, df)),
                body_set="24 (SAC panel)",
                post_hoc=ph,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its p; the planned gate required p < 0.01.",
            )
            Fc, qc, dfc, pc = cr1_wald(v, fam)
            R.add(
                f"sac.{name}.cr1_p",
                pc,
                body_set="24 (SAC panel)",
                post_hoc=PH_YES,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                note=f"F({qc}, {dfc:.0f}) = {Fc:.2f}",
                definition="Body-clustered Wald p for rule x family.",
            )
            m = A.mean(2)
            for f in FAMS:
                ok = (fam == f) & (m[:, 0] > 0.05)
                R.add(
                    f"sac.{name}.str_rf.{f}",
                    float(np.mean(m[ok, 2] / m[ok, 0])),
                    body_set="24 (SAC panel)",
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} mean per-body Strict/Reset-free ratio of three-seed means (bodies with Reset-free PSR above 0.05).",
                )
                R.add(
                    f"sac.{name}.str_rf_n.{f}",
                    int(ok.sum()),
                    body_set="24 (SAC panel)",
                    post_hoc=ph,
                    definition="Bodies entering that mean.",
                )
        ms, mp = SAC.mean(2), PPO.mean(2)
        valid = (ms[:, 0] > 0.05) & (mp[:, 0] > 0.05)
        rs, rp = ms[valid, 2] / ms[valid, 0], mp[valid, 2] / mp[valid, 0]
        bidx = boot_index(int(valid.sum()), args.boot)
        bs = spearman_last(rs[bidx], rp[bidx])
        R.add(
            "sac.cross_rho",
            rho(rs, rp),
            ci=pct_ci(bs),
            body_set=f"{int(valid.sum())} joint-valid SAC bodies",
            post_hoc=PH_PLAN,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Spearman rho between SAC and PPO per-body Strict/Reset-free ratios (Reset-free PSR above 0.05 under both); bootstrap interval post hoc.",
        )
        vtv = fam[valid] == "vt"
        R.add(
            "sac.cross_rho_vt",
            rho(rs[vtv], rp[vtv]),
            body_set="13 VT SAC bodies",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Same within VT.",
        )
        okf = valid & (fam == "floor")
        R.add(
            "sac.sac.str_rf.floor_joint_valid",
            float(np.mean(ms[okf, 2] / ms[okf, 0])),
            body_set="Floor joint-valid SAC bodies",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="SAC Floor mean Strict/Reset-free ratio on the joint-valid bodies.",
        )
        agree = np.sign(ms[:, 2] - ms[:, 0]) == np.sign(mp[:, 2] - mp[:, 0])
        R.add(
            "sac.vt_same_sign",
            int(agree[fam == "vt"].sum()),
            body_set="13 VT SAC bodies",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="VT bodies whose Strict minus Reset-free difference has the same sign under SAC and PPO.",
        )
        keep = ~inert
        v = SAC[keep][:, [1, 2, 0], :]
        F, q, df = BlockedF(v).F(fam[keep])
        R.add(
            "sac.sac.perrun_p_noninert",
            float(stats.f.sf(F, q, df)),
            body_set="16 SAC bodies where Standard can act",
            kind="p",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            note=f"F({q}, {df}) = {F:.2f}",
            definition="Per-run rule x family p without the Standard-inert bodies.",
        )
        tt = [json.loads((res / f"sac_mjx_{TAU_DIR[r]}_seed{s}" / "results.json").read_text())[b]["train_time_s"] for r in RULES for s in SEEDS for b in bodies]
        extra["sac_hours"] = float(np.sum(tt) / 3600)
        i = bodies.index("vt-1409-12-10-02-21-04-52")
        L = [json.loads((res / f"sac_mjx_0p50_seed{s}" / "results.json").read_text())[bodies[i]][EV_STRICT]["mean_ep_len"] for s in SEEDS]
        R.add(
            "sac.vt_1409_12_10.strict_eval_len",
            L,
            body_set="1",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_STRICT],
            dec=1,
            definition="SAC Strict-trained mean episode length per seed under the Strict evaluator for vt-1409-12-10 (a VT cut-off body under PPO).",
        )
    else:
        log(f"SAC body list {sac_list} not found; SAC panel skipped")

    # ---------------- no-time-limit panel
    R.section = "No-time-limit panel (30 bodies, planned, later code)"
    bfj = Path(args.bf_bodies)
    if bfj.exists() and (res / "boundary_free").exists():
        bodies = [b["body"] for b in json.loads(bfj.read_text())["panel"]]
        trf = {}
        for s in SEEDS:
            d = json.loads((res / "boundary_free" / f"seed{s}" / "results.json").read_text())
            for b in bodies:
                trf.setdefault(b, []).append(d[b][EV_COMMON]["per_step_reward"])
        t0 = D.get("0p00")
        x = np.array([np.mean(trf[b]) for b in sorted(bodies)])
        y = np.array([t0[D.bodies.index(b)].mean() for b in sorted(bodies)])
        rng = np.random.default_rng(BOOT_SEED)
        boots = []
        for _ in range(args.boot):
            ii = rng.integers(0, len(x), len(x))
            r_ = stats.spearmanr(x[ii], y[ii]).statistic
            if np.isfinite(r_):
                boots.append(r_)
        R.add(
            "bf.rho",
            rho(x, y),
            ci=pct_ci(boots),
            body_set="30 (no-time-limit panel)",
            post_hoc=PH_PLAN,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Spearman rho between three-seed PSR after Reset-free training without a time limit (later code) and corpus Reset-free training; planned check: upper bound below 0.95.",
        )
        for f in FAMS:
            bs_ = [b for b in sorted(bodies) if b.startswith(f + "-")]
            delta = np.array([np.mean(trf[b]) - t0[D.bodies.index(b)].mean() for b in bs_])
            rng = np.random.default_rng(BOOT_SEED)
            boots = np.array([np.median(rng.choice(delta, size=len(delta), replace=True)) for _ in range(args.boot)])
            R.add(
                f"bf.median_delta.{f}",
                float(np.median(delta)),
                ci=pct_ci(boots),
                body_set=f"{len(bs_)} {FAM_NAME[f]} (no-time-limit panel)",
                post_hoc=PH_PLAN,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Median over bodies of panel minus corpus Reset-free three-seed PSR, bootstrap interval of the median.",
            )
    else:
        log("no-time-limit panel inputs not found; skipped")

    # ---------------- bridge (12-body retraining with later code)
    R.section = "12-body retraining with later code (planned, failed)"
    bj = Path(args.bridge_bodies)
    eq = res / "equivalence_pilot" / "jax"
    if bj.exists() and eq.exists():
        bl = json.loads(bj.read_text())["bodies"]
        names = [b["name"] for b in bl]
        new = {}
        for b in names:
            for r in RULES:
                for s in SEEDS:
                    d = json.loads((eq / b / f"tau{TAU_DIR[r]}_seed{s}" / "results.json").read_text())
                    if len(d) != 1:
                        raise ValueError(f"bridge record {b} {r} {s} holds {len(d)} bodies")
                    new[(b, r, s)] = next(iter(d.values()))

        def mean_new(b, r, ev):
            return float(np.mean([new[(b, r, s)][ev]["per_step_reward"] for s in SEEDS]))

        def mean_old(b, r, ev):
            return float(D.get(TAU_DIR[r], ev)[D.bodies.index(b)].mean())

        matched = {"rf": EV_NEUTRAL, "len": EV_COMMON, "str": EV_STRICT}
        fails = 0
        for r in RULES:
            o = np.array([mean_old(b, r, matched[r]) for b in names])
            nn = np.array([mean_new(b, r, matched[r]) for b in names])
            rr = rho(o, nn)
            drift = float(np.median(np.abs(nn - o) / np.maximum(0.05, np.abs(o))))
            lr = stats.linregress(o, nn)
            crit = [rr > 0.85, drift < 0.10, 0.9 <= lr.slope <= 1.1, abs(lr.intercept) < 0.05]
            fails += sum(not c for c in crit)
            R.add(
                f"bridge.rho_matched.{r}",
                rr,
                body_set="12 (bridge)",
                post_hoc=PH_PLAN,
                evaluator=EV_LABEL[matched[r]],
                definition=f"Spearman rho between corpus and later-code three-seed PSR after {RULE_NAME[r]} training, scored by the rule's own evaluator.",
            )
            oc = np.array([mean_old(b, r, EV_COMMON) for b in names])
            nc = np.array([mean_new(b, r, EV_COMMON) for b in names])
            R.add(
                f"bridge.rho_common.{r}",
                rho(oc, nc),
                body_set="12 (bridge)",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Same under the common evaluator.",
            )
        o = np.array([mean_old(b, "len", EV_COMMON) - mean_old(b, "rf", EV_COMMON) for b in names])
        nn = np.array([mean_new(b, "len", EV_COMMON) - mean_new(b, "rf", EV_COMMON) for b in names])
        rd = rho(o, nn)
        sg = int(np.sum(np.sign(o) == np.sign(nn)))
        fails += int(not rd > 0.9) + int(not sg / len(names) >= 0.9)
        R.add(
            "bridge.delta_reset_rho",
            rd,
            body_set="12 (bridge)",
            post_hoc=PH_PLAN,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Spearman rho of Standard minus Reset-free PSR, corpus vs later code.",
        )
        R.add(
            "bridge.delta_reset_sign_agree",
            sg,
            body_set="12 (bridge)",
            post_hoc=PH_PLAN,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Bodies whose Standard minus Reset-free sign agrees.",
        )
        R.add(
            "bridge.criteria_failed",
            fails,
            body_set="12 (bridge)",
            post_hoc=PH_PLAN,
            definition="Failed sub-criteria of 14 (per rule: rho > 0.85, median relative drift < 0.10, slope in [0.9, 1.1], |intercept| < 0.05; plus Delta_reset rho > 0.9 and sign agreement at least 0.9).",
        )
        for f in FAMS:
            bs_ = [b for b in names if b.startswith(f + "-")]
            drift = [mean_new(b, "rf", EV_COMMON) - mean_old(b, "rf", EV_COMMON) for b in bs_]
            R.add(
                f"bridge.rf_drift_median.{f}",
                float(np.median(drift)),
                body_set=f"{len(bs_)} {FAM_NAME[f]} bridge bodies",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Median over bodies of later-code minus corpus three-seed PSR after Reset-free training.",
            )
        tt = [new[k]["train_time_s"] for k in new]
        extra["bridge_hours"] = float(np.sum(tt) / 3600)
    else:
        log("bridge inputs not found; skipped")

    # ---------------- compute
    R.section = "Compute (recorded wall time, train plus evaluation)"

    def hours(paths):
        tot, n = 0.0, 0
        for p in paths:
            d = json.loads(Path(p).read_text())
            for r in d.values():
                if isinstance(r, dict) and "train_time_s" in r:
                    tot += float(r["train_time_s"])
                    n += 1
        return tot / 3600, n

    corpus = [res / f"mjx_ppo_{TAU_DIR[r]}_seed{s}" / "results.json" for r in RULES for s in SEEDS]
    h, n = hours(corpus)
    R.add(
        "compute.corpus.hours",
        h,
        body_set="900 corpus runs",
        dec=1,
        post_hoc=PH_NA,
        definition="Summed recorded wall time (h); one run per GPU.",
    )
    R.add("compute.corpus.runs", n, body_set="900 corpus runs", post_hoc=PH_NA, definition="Runs with a recorded time.")
    R.add(
        "compute.corpus.hours_per_run",
        h / n,
        body_set="900 corpus runs",
        dec=3,
        post_hoc=PH_NA,
        definition="Mean hours per run.",
    )
    mid = [res / f"mjx_ppo_{t}_seed{s}" / "results.json" for t in ("0p10", "0p35") for s in SEEDS]
    h, n = hours(mid)
    R.add(
        "compute.tau010_035.hours",
        h,
        body_set=f"{n} runs",
        dec=1,
        post_hoc=PH_NA,
        definition="Summed wall time of the tau = 0.10 and 0.35 runs (h), four runs per GPU.",
    )
    if "sac_hours" in extra:
        R.add(
            "compute.sac.hours",
            extra["sac_hours"],
            body_set="216 SAC runs",
            dec=1,
            post_hoc=PH_NA,
            definition="Summed wall time (h), one run per GPU.",
        )
    if "bridge_hours" in extra:
        R.add(
            "compute.bridge.run_hours",
            extra["bridge_hours"],
            body_set="108 bridge runs",
            dec=1,
            post_hoc=PH_NA,
            definition="Summed wall time (run-hours), four runs per GPU.",
        )
        R.add(
            "compute.bridge.gpu_hours",
            extra["bridge_hours"] / 4,
            body_set="108 bridge runs",
            dec=1,
            post_hoc=PH_NA,
            definition="Run-hours over four (GPU-hours at four runs per GPU).",
        )
    for tag, pat in (
        ("no_time_limit", "boundary_free/*/results.json"),
        ("boundary_duration", "boundary_sweep/*/results.json"),
        ("non_evolved", "nems_panel/*/results.json"),
        ("asymptotic", "asymptotic_5m/*/*/results.json"),
    ):
        ps = sorted(res.glob(pat))
        if ps:
            h, n = hours(ps)
            R.add(
                f"compute.{tag}.hours",
                h,
                body_set=f"{n} runs",
                dec=1,
                post_hoc=PH_NA,
                definition=f"Summed wall time of the {tag.replace('_', ' ')} panel (h).",
            )

    # ---------------- job-level offsets in the corpus
    R.section = "Run records: job and array-task offsets (post hoc)"
    idx_of = {}
    Y = np.empty((3, D.n, 3))
    for ti, r in enumerate(RULES):
        for si, s in enumerate(SEEDS):
            d = D.raw[(TAU_DIR[r], s)]
            for bi, b in enumerate(D.bodies):
                Y[ti, bi, si] = d[b][EV_COMMON]["per_step_reward"]
                idx_of[b] = d[b]["morph_idx"]
    chunk = np.array([idx_of[b] // 25 for b in D.bodies])
    Rr = Y - Y.mean(axis=2, keepdims=True)

    def s_job(X):
        return D.n * (X.mean(axis=1) ** 2).sum()

    def s_chunk(X):
        return sum((chunk == c).sum() * (X[:, chunk == c, :].mean(axis=1) ** 2).sum() for c in range(4))

    obs_j, obs_c = s_job(Rr), s_chunk(Rr)
    rng = perm_rng()
    nj, nc = np.empty(args.perm), np.empty(args.perm)
    for i in range(args.perm):
        perm = np.argsort(rng.random((3, D.n, 3)), axis=2)
        Rp = np.take_along_axis(Rr, perm, axis=2)
        nj[i], nc[i] = s_job(Rp), s_chunk(Rp)
    R.add(
        "jobs.offset_p.job",
        pval(nj, obs_j),
        body_set="100",
        kind="p",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Permutation p for a shared score offset per (rule, seed) job: seed labels permuted within body and rule; statistic the size-weighted sum of squared unit-mean residuals.",
    )
    R.add(
        "jobs.offset_p.array_task",
        pval(nc, obs_c),
        body_set="100",
        kind="p",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Same for job x array task (array task = morph_idx // 25).",
    )
    fam_by_chunk = {int(c): {f: int(((chunk == c) & (D.fam == f)).sum()) for f in FAMS} for c in range(4)}
    R.add(
        "jobs.array_task_families",
        fam_by_chunk,
        body_set="100",
        post_hoc=PH_YES,
        dec=0,
        definition="Family composition of each array task.",
    )


def section_structure(D: Data, R: Registry, args, sets):
    R.section = "Morphological descriptors (descriptive)"
    feats = ("init_height", "n_first_level_limbs", "tree_depth", "n_joints", "n_bodies")
    V = D.V()
    for sname in ("all100", "g72"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        rf_len, str_len = M[:, 0] / M[:, 1], M[:, 2] / M[:, 1]
        X = {k: np.array([D.feats[b][k] for b in np.array(D.bodies)[mm]], float) for k in feats}
        ph = PH_REV if sname == "all100" else PH_YES
        for k in feats:
            for lab, y in (("rf_len", rf_len), ("str_len", str_len)):
                r_, p_ = stats.spearmanr(X[k], y)
                R.add(
                    f"struct.spearman.{k}.{lab}.{sname}",
                    float(r_),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    note=f"p = {p_:.2g}",
                    definition=f"Spearman rho between {k} and per-body {lab.replace('_', '/')} retention.",
                )
    for sname in ("all100", "no_vt10", "s89"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        fam = D.fam[mm]
        X = np.column_stack([[D.feats[b][k] for b in np.array(D.bodies)[mm]] for k in feats]).astype(float)
        y = M[:, 2] - M[:, 0]

        def ols(Xm, yy):
            Xm = np.column_stack([Xm, np.ones(len(yy))])
            beta = np.linalg.lstsq(Xm, yy, rcond=None)[0]
            res_ = yy - Xm @ beta
            return 1 - res_ @ res_ / np.sum((yy - yy.mean()) ** 2), res_, Xm @ beta

        ph = PH_REV if sname == "all100" else PH_YES
        for lab, Xm in (("M1", M[:, 1:2]), ("M2", X), ("M3", np.column_stack([M[:, 1], X]))):
            r2, _, pred = ols(Xm, y)
            R.add(
                f"struct.{lab}.R2.{sname}",
                float(r2),
                body_set=SET_LABEL[sname],
                post_hoc=ph,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"In-sample R^2 of {lab} for Strict minus Reset-free PSR (M1 Standard skill, M2 five descriptors, M3 both).",
            )
            if sname == "all100":
                R.add(
                    f"struct.{lab}.rho.{sname}",
                    rho(pred, y),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"Spearman rho between {lab} fitted values and the outcome.",
                )
        if sname != "all100":
            continue
        z0 = X[:, 0]
        for tag, yy, x2, desc in (
            (
                "described",
                y,
                X[:, 1],
                "Strict minus Reset-free PSR on z0 and first-level limb count (the model the text describes)",
            ),
            (
                "mislabelled",
                M[:, 1] - M[:, 0],
                X[:, 4] - 1,
                "Standard minus Reset-free PSR on z0 and body count minus one (source of the reviewed numbers)",
            ),
        ):
            r2, res_, _ = ols(np.column_stack([z0, x2]), yy)
            g = [res_[fam == f] for f in FAMS]
            F, p = stats.f_oneway(*g)
            R.add(
                f"struct.two_feature.{tag}.R2",
                float(r2),
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Two-feature control, {desc}: R^2.",
            )
            R.add(
                f"struct.two_feature.{tag}.resid_means",
                [float(x.mean()) for x in g],
                body_set="100",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="Family residual means (Floor, VT, MVT).",
            )
            R.add(
                f"struct.two_feature.{tag}.F",
                float(F),
                body_set="100",
                post_hoc=PH_YES,
                dec=2,
                evaluator=EV_LABEL[EV_COMMON],
                definition="One-way ANOVA F of the residuals across families, df (2, 97).",
            )
            R.add(
                f"struct.two_feature.{tag}.p",
                float(p),
                body_set="100",
                post_hoc=PH_YES,
                kind="p",
                evaluator=EV_LABEL[EV_COMMON],
                definition="Its p.",
            )
        g0 = [y[fam == f] for f in FAMS]
        R.add(
            "struct.dstrrf.oneway_F",
            float(stats.f_oneway(*g0).statistic),
            body_set="100",
            post_hoc=PH_YES,
            dec=2,
            evaluator=EV_LABEL[EV_COMMON],
            definition="One-way ANOVA F of Strict minus Reset-free PSR across families before any control.",
        )
        R.add(
            "struct.n_bodies_vs_n_joints_r",
            float(np.corrcoef(X[:, 4], X[:, 3])[0, 1]),
            body_set="100",
            post_hoc=PH_YES,
            definition="Pearson r between body count and joint count.",
        )


# ----------------------------------------------------------------- appendix values added 2026-10-09
# Each extra_* function below runs right after the section it extends, uses that section's name,
# and only adds rows. None of them reuses a random stream of an existing row: every bootstrap
# draws its own resample matrix with boot_index (default_rng(42)), as the existing rows do.


def _load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def extra_sets(D: Data, R: Registry, args, sets, extra):
    R.section = "Body sets, counts and threshold geometry"
    z0, fam = D.z0, D.fam
    inert20, cut11, cut23, vt10, Lstr = extra["inert20"], extra["cut11"], extra["cut23"], extra["vt10"], extra["Lstr"]
    z30 = np.isclose(z0, 0.30)
    derl = 0.5 * (z0 - 0.2)
    R.add(
        "geom.strict_minus_derl_cutoff_m",
        [float((0.5 * z0 - derl).min()), float((0.5 * z0 - derl).max())],
        body_set="100",
        dec=3,
        post_hoc=PH_YES,
        definition="Range over bodies of our Strict cutoff 0.5 z0 minus DERL's cutoff 0.5 (z0 minus 0.2 m) (m).",
    )
    R.add(
        "geom.derl_fraction_median",
        float(np.median(derl / z0)),
        body_set="100",
        dec=2,
        post_hoc=PH_YES,
        definition="Median over bodies of DERL's cutoff as a fraction of our z0.",
    )
    R.add(
        "geom.derl_cutoff_z030_m",
        float(0.5 * (0.30 - 0.2)),
        body_set="25 bodies at z0 = 0.30 m",
        dec=2,
        post_hoc=PH_YES,
        definition="DERL's cutoff 0.5 (z0 minus 0.2 m) at z0 = 0.30 m (m).",
    )
    R.add(
        "geom.derl_penetration_z030_m",
        float(TORSO_R - 0.5 * (0.30 - 0.2)),
        body_set="25 bodies at z0 = 0.30 m",
        dec=2,
        post_hoc=PH_YES,
        definition="Floor penetration a 0.1 m torso needs to reach DERL's cutoff at z0 = 0.30 m (m).",
    )
    vals, cnts = np.unique(np.round(z0, 3), return_counts=True)
    R.add("geom.z0_n_distinct", int(len(vals)), body_set="100", post_hoc=PH_YES, definition="Distinct reset heights z0.")
    for v, c in zip(vals[:3], cnts[:3]):
        R.add(
            f"geom.z0_count.{f'{v:.2f}'.replace('.', 'p')}",
            int(c),
            body_set="100",
            post_hoc=PH_YES,
            definition=f"Bodies with z0 = {v:.2f} m (the three lowest reset heights).",
        )
    R.add(
        "geom.inert20.cutoffs_m",
        sorted({round(float(x), 6) for x in 0.2 * z0[inert20]}),
        body_set=SET_LABEL["inert28"],
        dec=2,
        post_hoc=PH_YES,
        definition="Distinct Standard cutoffs 0.20 z0 of the 28 bodies (m).",
    )
    m = 0.5 * z0[inert20] - TORSO_R
    R.add(
        "geom.inert20.strict_margin_m",
        [float(m.min()), float(m.max())],
        body_set=SET_LABEL["inert28"],
        dec=3,
        post_hoc=PH_YES,
        definition="Range of the Strict cutoff 0.5 z0 above the 0.1 m torso radius for the 28 bodies (m).",
    )
    tall = ~extra["inert10"]
    R.add(
        "geom.tau010_above_radius.n",
        int(tall.sum()),
        body_set="100",
        post_hoc=PH_YES,
        definition="Bodies whose tau = 0.10 cutoff lies at or above the torso radius (the tallest bodies).",
    )
    R.add(
        "geom.tau010_above_radius.z0",
        sorted({round(float(x), 2) for x in z0[tall]}),
        body_set="6 tallest",
        dec=2,
        post_hoc=PH_YES,
        definition="Their reset heights (m).",
    )
    R.add(
        "geom.tau010_above_radius.margin_m",
        [float((0.1 * z0[tall] - TORSO_R).min()), float((0.1 * z0[tall] - TORSO_R).max())],
        body_set="6 tallest",
        dec=4,
        post_hoc=PH_YES,
        definition="Range of the tau = 0.10 cutoff above the torso radius for them (m).",
    )
    marg = D.audit_full["summary"]["probe_margin_0.20_inert"]
    m30 = [marg[b] for b, k in zip(D.bodies, inert20 & z30) if k]
    R.add(
        "inert28.z030.max_probe_margin_m",
        float(max(m30)),
        body_set="25 bodies at z0 = 0.30 m",
        dec=4,
        post_hoc=PH_YES,
        definition="Largest audit-probe margin of the torso above the Standard cutoff (m); with inert28.z030.min_probe_margin_m the range of the probe margins.",
    )
    R.add(
        "inert28.episodes",
        int(inert20.sum() * len(SWEEP) * len(SEEDS) * 20),
        body_set=SET_LABEL["inert28"],
        post_hoc=PH_YES,
        definition="Recorded evaluation episodes of the 28 bodies under the common evaluator (420 policies x 20 episodes).",
    )
    never = D.audit_full["summary"]["firing_0.20_never_eval_fall"]
    mm = [0.2 * D.audit[b]["z0"] - TORSO_R for b in never]
    R.add(
        "inert28.practically_inert_mvt_margin_m",
        [float(min(mm)), float(max(mm))],
        body_set="72",
        dec=3,
        post_hoc=PH_YES,
        definition="Range of the Standard cutoff above the torso radius for the 5 bodies of the 72 whose Standard cutoff never ended a recorded episode (m).",
    )
    fact = D.audit_full["summary"]["factorial_tau0p20"]["inert_bodies"]
    R.add(
        "inert28.factorial_bodies",
        len(fact),
        body_set="10 inert bodies in the factorial panel",
        post_hoc=PH_YES,
        definition="Standard-inert bodies in the panel without a time limit.",
    )
    d50 = {b: D.audit[b]["factorial_tau0p50_5M_train_dones"] for b, k in zip(D.bodies, inert20) if k and "factorial_tau0p50_5M_train_dones" in D.audit[b]}
    if d50:
        spd = np.array([5e6 / v for d in d50.values() for v in d.values()], float)
        R.add(
            "inert28.factorial_strict_le10.max_steps_per_done",
            float(spd[spd <= 10].max()),
            body_set="10 inert bodies in the factorial panel",
            dec=1,
            post_hoc=PH_YES,
            definition="Longest average training episode among the Strict-trained factorial runs that average at most 10 steps (steps).",
        )
        b = "vt-5506-3-17-02-18-33-45"
        if b in d50:
            R.add(
                f"inert28.factorial_strict.dones.{b}",
                [int(d50[b][str(s)]) for s in SEEDS],
                body_set="1",
                dec=0,
                post_hoc=PH_YES,
                definition="Recorded training episode ends (dones) of each Strict-trained seed (42, 43, 44) in about 5M steps.",
            )
    passive = {r["body"]: r for r in _load_json(args.passive)} if Path(args.passive).exists() else None
    settle = {r[1]: r for r in _load_json(args.settle)} if Path(args.settle).exists() else None
    if passive is not None:
        lp = np.array([passive[b]["lowest_point_at_reset"] for b in D.bodies])
        R.add(
            "zero.lowest_point_at_reset_median_m",
            float(np.median(lp)),
            body_set="100",
            dec=2,
            evaluator="reset pose",
            post_hoc=PH_YES,
            definition="Median over bodies of the lowest body point above the floor in the reset pose (m).",
        )
        tallvt = [b for b, k in zip(D.bodies, vt10 & ~z30) if k]
        z1 = [passive[b]["zero_action_height_step50"] for b in tallvt]
        R.add(
            "zero.vt10_taller.z_1s_m",
            [float(min(z1)), float(max(z1))],
            body_set="3 taller VT cut-off bodies",
            dec=3,
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Range of their zero-action torso heights after 50 control steps (1 s) (m).",
        )
        R.add(
            "cut.vt10.taller_strict_cutoffs_m",
            sorted({round(0.5 * passive[b]["z0"], 6) for b in tallvt}),
            body_set="3 taller VT cut-off bodies",
            dec=3,
            post_hoc=PH_YES,
            definition="Their Strict cutoffs 0.5 z0 (m).",
        )
    if settle is not None:
        z30b = [b for b, k in zip(D.bodies, z30) if k]
        rest = np.array([settle[b][3] for b in z30b])
        R.add(
            "zero.z030.rest_height_3s_m",
            [float(rest.min()), float(rest.max())],
            body_set="25 bodies at z0 = 0.30 m",
            dec=3,
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Range of the zero-action torso height after 3 s (settle.json) (m).",
        )
        R.add(
            "zero.z030.rest_at_radius.n",
            int((np.abs(rest - TORSO_R) < 0.005).sum()),
            body_set="25 bodies at z0 = 0.30 m",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Of those, bodies whose torso rests within 5 mm of 0.10 m after 3 s (the others rest on their limbs).",
        )
        R.add(
            "zero.z030.rest_above_radius.ids",
            [b for b, x in zip(z30b, rest) if abs(x - TORSO_R) >= 0.005],
            body_set="25 bodies at z0 = 0.30 m",
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            kind="ids",
            definition="Bodies at z0 = 0.30 m whose torso rests more than 5 mm above 0.10 m after 3 s.",
        )
        vt10_derl = [b for b, k in zip(D.bodies, vt10) if k and settle[b][3] < 0.5 * (settle[b][2] - 0.2)]
        R.add(
            "zero.vt10_below_derl_cutoff_3s.derl_cutoff_m",
            sorted({round(0.5 * (settle[b][2] - 0.2), 6) for b in vt10_derl}),
            body_set="10",
            dec=3,
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="DERL's cutoff for those bodies (m).",
        )
        R.add(
            "zero.vt10_below_derl_cutoff_3s.z_3s_m",
            sorted(float(settle[b][3]) for b in vt10_derl),
            body_set="10",
            dec=3,
            evaluator="zero-action rollout",
            post_hoc=PH_YES,
            definition="Their zero-action torso heights after 3 s (m).",
        )
    pdir = Path(args.probe_dir)
    p20, p100 = _load_json(pdir / "train_like_probe_all_20.json"), _load_json(pdir / "train_like_probe_near_100.json")
    if p20 is not None:
        z45 = [b for b, k in zip(D.bodies, inert20 & np.isclose(z0, 0.45)) if k]
        mins = sorted(float(p20[b]["gauss"]["min_z_ctrl"]) for b in z45)
        R.add(
            "probe.train_like.z045.min_torso_m",
            mins,
            body_set="3 bodies at z0 = 0.45 m",
            dec=4,
            evaluator="training-like probe (Gaussian actions, training reset noise, 20 episodes)",
            post_hoc=PH_YES,
            definition="Minimum torso height per body over 20 episodes with random Gaussian actions and the training reset noise (m).",
        )
        R.add(
            "probe.train_like.z045.max_below_cutoff_m",
            float(max(0.2 * 0.45 - x for x in mins)),
            body_set="3 bodies at z0 = 0.45 m",
            dec=4,
            evaluator="training-like probe (Gaussian actions, training reset noise, 20 episodes)",
            post_hoc=PH_YES,
            definition="Largest depth of those minima below the 0.09 m Standard cutoff (m).",
        )
        if p100 is not None:
            reach = []
            for b, k in zip(D.bodies, extra["inert10"]):
                if not k:
                    continue
                zs = [D.audit[b]["probe_min_torso_z"]["min"]]
                hit = D.audit[b]["probe_min_torso_z"]["min"] < 0.10 * D.audit[b]["z0"]
                for rec in (p20, p100):
                    if b in rec:
                        zs += [rec[b]["gauss"]["min_z_ctrl"], rec[b]["bang"]["min_z_ctrl"]]
                        hit = hit or rec[b]["gauss"]["eps_below"] > 0 or rec[b]["bang"]["eps_below"] > 0
                if hit:
                    reach.append(TORSO_R - min(zs))
            R.add(
                "probe.reach.torso_sink_m",
                [float(min(reach)), float(max(reach))],
                body_set=f"{len(reach)} of the 94 whose probes reached the tau = 0.10 cutoff",
                dec=3,
                evaluator="audit and training-like probes",
                post_hoc=PH_YES,
                definition="Range over those bodies of the deepest probe torso-center depth below the 0.1 m radius (m).",
            )
    se = _load_json(args.search)
    if se is not None:
        se = {r["body"]: r for r in se}
        short = [b for b, k in zip(D.bodies, vt10 & z30) if k and b in se]
        tallvt = [b for b, k in zip(D.bodies, vt10 & ~z30) if k and b in se]
        R.add(
            "probe.search.trials",
            sorted({int(se[b]["search_trials"]) for b in se}),
            body_set=f"{len(se)} probed bodies",
            dec=0,
            evaluator="random open-loop actions from the nominal reset",
            post_hoc=PH_YES,
            definition="Random open-loop action sequences per body in the larger search (vt10_search.json).",
        )
        R.add(
            "probe.search.vt10_short.max_best_survival_steps",
            int(max(se[b]["search_best_survival_steps"] for b in short)),
            body_set="7 short VT cut-off bodies",
            evaluator="random open-loop actions from the nominal reset",
            post_hoc=PH_YES,
            definition="Longest survival (control steps above 0.5 z0) of any sampled sequence on the seven short VT cut-off bodies.",
        )
        R.add(
            "probe.search.vt10_taller.n_survive_ge50",
            {b: int(se[b]["search_n_survive_ge50"]) for b in tallvt},
            body_set="3 taller VT cut-off bodies",
            dec=0,
            evaluator="random open-loop actions from the nominal reset",
            post_hoc=PH_YES,
            definition="Sequences that kept each taller VT cut-off body at or above 0.5 z0 for 50 control steps.",
        )
        fcb = [b for b, k in zip(D.bodies, cut11 & (fam == "floor")) if k]
        if fcb and fcb[0] in se:
            R.add(
                "probe.search.floor_cutoff_body.n_survive_ge50",
                int(se[fcb[0]]["search_n_survive_ge50"]),
                body_set="1",
                evaluator="random open-loop actions from the nominal reset",
                post_hoc=PH_YES,
                definition="Sequences that kept the Floor cut-off body at or above 0.5 z0 for 50 control steps.",
            )
    fcb = [i for i in np.where(cut11 & (fam == "floor"))[0]]
    if fcb:
        i = fcb[0]
        R.add("cut.floor_cutoff_body.z0", float(z0[i]), body_set="1", dec=2, post_hoc=PH_YES, definition="Reset height of the Floor cut-off body (m).")
        for r in RULES:
            L = D.get(TAU_DIR[r], EV_STRICT, "mean_ep_len")[i]
            R.add(
                f"cut.floor_cutoff_body.strict_eval_len.{r}",
                [float(x) for x in L],
                body_set="1",
                dec=1,
                evaluator=EV_LABEL[EV_STRICT],
                post_hoc=PH_YES,
                definition=f"Mean Strict-evaluator episode length per seed (42, 43, 44) after {RULE_NAME[r]} training.",
            )
            R.add(
                f"cut.floor_cutoff_body.strict_eval_len_mean.{r}",
                float(L.mean()),
                body_set="1",
                dec=1,
                evaluator=EV_LABEL[EV_STRICT],
                post_hoc=PH_YES,
                definition="Three-seed mean of those lengths.",
            )
    g72 = sets["g72"]
    R.add("cut.cut11_in_g72", int((cut11 & g72).sum()), body_set="72", post_hoc=PH_YES, definition="Cut-off bodies in the 72.")
    R.add("cut.cut23_in_g72", int((cut23 & g72).sum()), body_set="72", post_hoc=PH_YES, definition="Cut-off-seed bodies in the 72.")
    R.add("cut.cut23_not_cut11", int((cut23 & ~cut11).sum()), body_set="23", post_hoc=PH_YES, definition="Cut-off-seed bodies that are not cut-off bodies.")
    aff = inert20 | cut23
    R.add("cut.affected.n", int(aff.sum()), body_set="35", post_hoc=PH_YES, definition="Bodies affected by either threshold defect (the 28 or the 23).")
    for i in np.where(aff)[0]:
        flags = [lab for lab, x in (("inert", inert20[i]), ("cut-off body", cut11[i])) if x]
        R.add(
            f"cut.affected.{D.bodies[i]}",
            [round(float(z0[i]), 2), int((Lstr[i] <= 20).sum())],
            body_set="35",
            dec=2,
            post_hoc=PH_YES,
            note=", ".join(flags),
            definition="Affected body: z0 (m) and the number of its Strict-trained seeds whose Strict-evaluated episodes average at most 20 steps; the note flags Standard-inert and cut-off bodies.",
        )
    S = D.get("0p50")
    Lm = D.get("0p20").mean(1)
    cutrun = Lstr <= 20
    for f in FAMS:
        sel = fam == f
        rs = S[sel] / Lm[sel, None]
        c = cutrun[sel]
        R.add(
            f"cut.runs.{f}.loss_from_cut_pts",
            float(100 * (1 - rs[c]).sum() / c.size),
            body_set="100",
            dec=1,
            evaluator=EV_LABEL[EV_COMMON],
            post_hoc=PH_YES,
            definition=f"Points of {FAM_NAME[f]}'s Strict loss (100 minus 100 x mean run Strict/Standard ratio) contributed by its cut-off runs: 100 x sum over cut-off runs of (1 minus ratio) over all {FAM_NAME[f]} Strict runs.",
        )
    R.add("cut.runs.total.n_cut", int(cutrun.sum()), body_set="100", post_hoc=PH_YES, definition="Cut-off runs over all families.")
    R.add("cut.runs.total.n_runs", int(cutrun.size), body_set="100", post_hoc=PH_YES, definition="Strict-trained runs (100 bodies x 3 seeds).")
    vtz = (fam == "vt") & z30
    best = Lstr[vtz].max(1)
    R.add(
        "cut.z030_vt_short_seed.n",
        int((best <= 500).sum()),
        body_set="13 VT bodies at z0 = 0.30 m",
        evaluator=EV_LABEL[EV_STRICT],
        post_hoc=PH_YES,
        definition="VT bodies at z0 = 0.30 m without a Strict-trained seed whose Strict-evaluated episodes average more than 500 steps.",
    )
    R.add(
        "cut.z030_vt_short_seed.max_best_seed_len",
        float(best[best <= 500].max()),
        body_set="13 VT bodies at z0 = 0.30 m",
        dec=1,
        evaluator=EV_LABEL[EV_STRICT],
        post_hoc=PH_YES,
        definition="Longest best-seed mean Strict-evaluated episode among them (steps).",
    )


def extra_ranking(D: Data, R: Registry, args, sets, extra):
    R.section = "Rank agreement across rules and seeds"
    B = args.boot
    V = D.V()
    fam = D.fam
    pt72 = ranking_components(V[sets["g72"]])
    for r in RULES:
        R.add(
            f"rank.R3.{r}.g72",
            float(pt72[f"R3_{r}"][0]),
            body_set=SET_LABEL["g72"],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"Spearman-Brown reliability of the three-seed {RULE_NAME[r]} ranking, 3W/(1+2W).",
        )
    pt = ranking_components(V)
    for r in ("rf", "str"):
        i = RULES.index(r)
        Y = V[:, i, :]
        obs = np.mean([rho(Y[:, s], Y[:, [t for t in range(3) if t != s]].mean(1)) for s in range(3)])
        W = float(pt[f"W_{r}"][0])
        R.add(
            f"rank.sb_check.{r}.pred",
            math.sqrt(W * 2 * W / (1 + W)),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"Spearman-Brown prediction of the agreement between one {RULE_NAME[r]} seed and the mean of the other two, sqrt(W x 2W/(1+W)).",
        )
        R.add(
            f"rank.sb_check.{r}.obs",
            float(obs),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Observed agreement between one seed and the mean of the other two (mean over the three held-out seeds).",
        )
    # top-20 overlaps of the simulated reruns: the same draws as rank.sim_rerun.* (checked on the top-10 means)
    M = V.mean(-1)
    t20 = {}
    for r in ("str", "rf"):
        Y = V[:, RULES.index(r), :]
        ybar = Y.mean(1)
        e = Y - ybar[:, None]
        vnm = (e**2).sum(1) / 2 / 3
        lam = math.sqrt(max(0.0, 1 - vnm.mean() / ybar.var(ddof=1)))
        mu_shr = ybar.mean() + lam * (ybar - ybar.mean())
        n = len(ybar)
        for model in ("normal", "resid_boot", "normal_shrunk"):
            rng = np.random.default_rng(BOOT_SEED)
            if model == "resid_boot":
                sc = e * math.sqrt(1.5)
                A_ = ybar[None] + np.take_along_axis(np.broadcast_to(sc, (B, n, 3)), rng.integers(0, 3, (B, n, 3)), axis=2).mean(2)
                B_ = ybar[None] + np.take_along_axis(np.broadcast_to(sc, (B, n, 3)), rng.integers(0, 3, (B, n, 3)), axis=2).mean(2)
            else:
                mu = mu_shr if model == "normal_shrunk" else ybar
                A_ = mu[None] + rng.standard_normal((B, n)) * np.sqrt(vnm)[None]
                B_ = mu[None] + rng.standard_normal((B, n)) * np.sqrt(vnm)[None]
            ov10 = (topk_mask_rows(A_, 10) & topk_mask_rows(B_, 10)).sum(1)
            if abs(float(ov10.mean()) - R.get(f"rank.sim_rerun.{r}.{model}.top10_mean")["value"]) > 1e-12:
                raise AssertionError("simulated reruns differ from rank.sim_rerun")
            ov20 = (topk_mask_rows(A_, 20) & topk_mask_rows(B_, 20)).sum(1)
            t20[(r, model)] = float(ov20.mean())
            R.add(
                f"rank.sim_rerun.{r}.{model}.top20_mean",
                float(ov20.mean()),
                body_set="100",
                dec=2,
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Mean top-20 overlap of two simulated independent three-seed {RULE_NAME[r]} reruns ({model.replace('_', ' ')} noise model, the draws of rank.sim_rerun.{r}.{model}).",
            )
    R.add(
        "rank.sim_rerun.top20_mean_range",
        [min(t20.values()), max(t20.values())],
        body_set="100",
        dec=2,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Range of the simulated top-20 overlaps over the 6 rule x noise-model simulations.",
    )
    for sname in ("all100", "g72"):
        mm = sets[sname]
        for f in FAMS:
            sel = mm & (fam == f)
            Vs = V[sel]
            with np.errstate(invalid="ignore"):  # the noise reference is unused here; small families give negative W draws
                p_ = ranking_components(Vs)
                b_ = ranking_components(Vs, boot_index(len(Vs), B))
            R.add(
                f"rank.within_family.D.rf_str.{f}.{sname}",
                float(p_["D_rf_str"][0]),
                ci=pct_ci(b_["D_rf_str"]),
                body_set=f"{SET_LABEL[sname]}, {FAM_NAME[f]} only",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Strict vs Reset-free excess disagreement D within {FAM_NAME[f]}, with a body bootstrap over the family's bodies.",
            )
        d = M[mm, 2] - M[mm, 0]
        fk = fam[mm]
        ssb = sum((fk == g).sum() * (d[fk == g].mean() - d.mean()) ** 2 for g in FAMS)
        R.add(
            f"rank.family_r2.dstr_rf.{sname}",
            float(ssb / ((d - d.mean()) ** 2).sum()),
            body_set=SET_LABEL[sname],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Share of the variance of per-body Strict minus Reset-free PSR (three-seed means) explained by family means.",
        )
    Vf = D.V(EV_COMMON, "fitness")
    for sname in ("all100", "g72"):
        mm = sets[sname]
        p_ = ranking_components(Vf[mm])
        for a, b in PAIRS:
            k = f"{a}_{b}"
            R.add(
                f"rank.fixed_horizon.rho3.{k}.{sname}",
                float(p_[f"rho3_{k}"][0]),
                body_set=SET_LABEL[sname],
                post_hoc=PH_YES,
                evaluator="common, fixed-horizon return (return / 1,000)",
                definition=f"Spearman rho between {RULE_NAME[a]} and {RULE_NAME[b]} rankings of three-seed mean fixed-horizon return.",
            )
            R.add(
                f"rank.fixed_horizon.D.{k}.{sname}",
                float(p_[f"D_{k}"][0]),
                body_set=SET_LABEL[sname],
                post_hoc=PH_YES,
                evaluator="common, fixed-horizon return (return / 1,000)",
                definition="Excess disagreement D with fixed-horizon return.",
            )
        Mf = Vf[mm].mean(-1)
        R.add(
            f"rank.psr_vs_fixed_horizon.min_rho.{sname}",
            min(rho(M[mm, j], Mf[:, j]) for j in range(3)),
            body_set=SET_LABEL[sname],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Smallest over the three training rules of the Spearman rho between per-body rankings by PSR and by fixed-horizon return.",
        )


def extra_family(D: Data, R: Registry, args, sets, extra):
    R.section = "Family means, retention and family contrasts"
    B = args.boot
    V = D.V()
    M, fam, rf_len, str_len = family_block(D, V, sets["inert28"])
    for f in FAMS:
        sel = fam == f
        R.add(
            f"fam.ret.rf_len.{f}.inert28",
            float(rf_len[sel].mean()),
            body_set=SET_LABEL["inert28"],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"{FAM_NAME[f]} Reset-free/Standard retention on the 28.",
        )
        R.add(
            f"fam.ret.str_len.{f}.inert28",
            float(str_len[sel].mean()),
            body_set=SET_LABEL["inert28"],
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"{FAM_NAME[f]} Strict/Standard retention on the 28.",
        )
    for sname in ("all100", "g72"):
        mm = sets[sname]
        fk = D.fam[mm]
        ph = PH_REV if sname == "all100" else PH_YES
        R.add(
            f"fam.oneway.df.{sname}",
            [len(FAMS) - 1, int(mm.sum()) - len(FAMS)],
            body_set=SET_LABEL[sname],
            dec=0,
            post_hoc=ph,
            definition="Degrees of freedom of a one-way ANOVA across the three families on this body set.",
        )
        for j, s in enumerate(SEEDS):
            r = V[mm][:, 0, j] / V[mm][:, 1, j]
            F, _ = stats.f_oneway(*[r[fk == f] for f in FAMS])
            R.add(
                f"fam.per_seed.{s}.{sname}.F",
                float(F),
                body_set=SET_LABEL[sname],
                dec=2,
                post_hoc=ph,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"Seed {s}: one-way ANOVA F of the per-seed Reset-free/Standard ratio across families.",
            )
    for sname in ("all100", "g72", "no_vt10"):
        M, fk, rf_len, str_len = family_block(D, V, sets[sname])
        for lab, x, f1, f2 in (("rf_len", rf_len, "floor", "vt"), ("str_len", str_len, "vt", "mvt"), ("str_len", str_len, "vt", "floor")):
            a, b = x[fk == f1], x[fk == f2]
            d = a[boot_index(len(a), B)].mean(1) - b[boot_index(len(b), B)].mean(1)
            R.add(
                f"fam.contrast_ci.{lab}.{f1}_minus_{f2}.{sname}",
                float(a.mean() - b.mean()),
                ci=pct_ci(d),
                body_set=SET_LABEL[sname],
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition=f"{FAM_NAME[f1]} minus {FAM_NAME[f2]} mean {lab.replace('_', '/')} retention, with a bootstrap that resamples each family's bodies independently (one boot_index matrix per family).",
            )
    M = V.mean(-1)
    for f in FAMS:
        sel = D.fam == f
        R.add(
            f"fam.ratio_of_means.str_rf.{f}.all100",
            float(M[sel, 2].mean() / M[sel, 0].mean()),
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"{FAM_NAME[f]} mean Strict PSR over mean Reset-free PSR (ratio of family means).",
        )
    Vs = D.V(EV_STRICT)
    R.add(
        "eval.strict.len_psr_below_0p1.n",
        int((Vs.mean(-1)[:, 1] < 0.1).sum()),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_STRICT],
        definition="Bodies whose Standard-trained three-seed mean PSR under the Strict evaluator is below 0.1.",
    )


def extra_interaction(D: Data, R: Registry, args, sets):
    R.section = "Rule x family interaction tests"
    V = D.V()
    for sname in ("all100", "g72"):
        mm = sets[sname]
        fam = D.fam[mm]
        ph = PH_REV if sname == "all100" else PH_YES
        rk = stats.rankdata(V[mm], axis=0)
        Fr, qr, dfr = BlockedF(rk).F(fam)
        R.add(f"inter.rank_perseed.{sname}.df", [int(qr), int(dfr)], body_set=SET_LABEL[sname], dec=0, post_hoc=ph, definition="Degrees of freedom of the rank ANOVA on runs.")
        R.add(
            f"inter.rank_perseed.{sname}.p",
            float(stats.f.sf(Fr, qr, dfr)),
            body_set=SET_LABEL[sname],
            kind="p",
            post_hoc=ph,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Parametric p of the rank ANOVA on runs (supplementary).",
        )
        bm = BlockedF(V[mm].mean(axis=2, keepdims=True))
        _, qm, dfm = bm.F(fam)
        R.add(f"inter.three.{sname}.seedmean_df", [int(qm), int(dfm)], body_set=SET_LABEL[sname], dec=0, post_hoc=ph, definition="Degrees of freedom of the seed-mean ANOVA.")
    fam = D.fam
    bf, bm = BlockedF(V), BlockedF(V.mean(axis=2, keepdims=True))

    def ss_int(model, S):
        return S * sum((fam == g).sum() * np.sum((model.w[fam == g].mean(axis=0) - model.gbar) ** 2) for g in FAMS)

    si, sm = ss_int(bf, 3), ss_int(bm, 1)
    ei, em = bf.sse_red - si, bm.sse_red - sm
    _, q1, df1 = bf.F(fam)
    _, q2, df2 = bm.F(fam)
    for key, val, dec, desc in (
        ("ss_interaction_perseed", si, 2, "Interaction sum of squares, ANOVA on runs."),
        ("ss_interaction_seedmean", sm, 2, "Interaction sum of squares, ANOVA on seed means."),
        ("ss_resid_perseed", ei, 1, f"Residual sum of squares, ANOVA on runs ({df1} df)."),
        ("ss_resid_seedmean", em, 1, f"Residual sum of squares, ANOVA on seed means ({df2} df)."),
        ("ms_resid_perseed", ei / df1, 3, "Residual mean square, ANOVA on runs."),
        ("ms_resid_seedmean", em / df2, 3, "Residual mean square, ANOVA on seed means."),
        ("ss_seed", float(((V - V.mean(axis=2, keepdims=True)) ** 2).sum()), 1, "Seed-to-seed sum of squares within each body and rule."),
        ("seedmean_partial_eta2", sm / (sm + em), 3, "Partial eta^2 of the interaction in the seed-mean ANOVA."),
    ):
        R.add(f"inter.three.all100.{key}", float(val), body_set="100", dec=dec, post_hoc=PH_REV, evaluator=EV_LABEL[EV_COMMON], definition=desc)
    R.add(
        "inter.three.all100.df_seed",
        int(V.shape[0] * V.shape[1] * (V.shape[2] - 1)),
        body_set="100",
        post_hoc=PH_REV,
        definition="Degrees of freedom of the seed-to-seed sum of squares (100 bodies x 3 rules x 2).",
    )
    ps = [R.get(f"inter.len_str.{s}.perm_p")["value"] for s in SUBSETS_MAIN if s != "all100"]
    R.add(
        "inter.len_str.min_perm_p_subsets",
        min(ps),
        body_set="72, 89, 77, 68, 65",
        kind="p",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Smallest Standard vs Strict x family permutation p over the five sets other than all 100.",
    )


def extra_selection(D: Data, R: Registry, args, sets, extra):
    R.section = "Selection audit (in sample, held-out seed, leave-one-rule-out)"
    B = args.boot
    V = D.V()
    crits = ("rf_only", "len_only", "str_only", "mean3")
    own = {"rf_only": "rf", "len_only": "len", "str_only": "str"}
    for sname in ("all100", "g72"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        sc = selectors(M)
        ph = PH_REV if sname == "all100" else PH_YES
        idx = boot_index(len(M), B)
        Mb = M[idx]
        bsc = {"rf_only": Mb[:, :, 0], "len_only": Mb[:, :, 1], "str_only": Mb[:, :, 2], "mean3": Mb.mean(-1)}
        for sn in crits:
            for i, r in enumerate(RULES):
                if own.get(sn) == r:
                    continue
                for k in KS:
                    if sname == "all100":
                        key = f"sel.in_ci.{sn}.{r}.k{k}.{sname}"
                        if key in R.index:
                            continue
                        ci = pct_ci(retention_batch(bsc[sn], Mb[:, :, i], k))
                        R.add(
                            key,
                            100 * retention(sc[sn], M[:, i], k),
                            ci=[100 * ci[0], 100 * ci[1]],
                            body_set=SET_LABEL[sname],
                            dec=1,
                            post_hoc=ph,
                            evaluator=EV_LABEL[EV_COMMON],
                            definition="In-sample retention with a 95% body-bootstrap interval (the resample matrix of sel.in_ci.min3).",
                        )
                    else:
                        key = f"sel.in.{sn}.{r}.k{k}.{sname}"
                        if key in R.index:
                            continue
                        R.add(
                            key,
                            100 * retention(sc[sn], M[:, i], k),
                            body_set=SET_LABEL[sname],
                            dec=1,
                            post_hoc=ph,
                            evaluator=EV_LABEL[EV_COMMON],
                            definition=f"In sample: selecting the top {k} by {sn} keeps this percentage of the {RULE_NAME[r]} oracle top-{k} mean PSR.",
                        )
        for sn in ("rf_only", "str_only", "mean3"):
            t10 = np.argsort(-sc[sn], kind="stable")[:10]
            for f in FAMS:
                R.add(
                    f"sel.top10_family.{sn}.{f}.{sname}",
                    int((D.fam[mm][t10] == f).sum()),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} bodies among the top 10 by {sn}.",
                )
    M = V.mean(-1)
    sc = selectors(M)
    R.add(
        "sel.in.single_rule.max_of_min_worst.all100",
        max(100 * min(retention(sc[sn], M[:, i], k) for i in range(3) for k in KS) for sn in ("rf_only", "len_only", "str_only")),
        body_set="100",
        dec=1,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Largest over the three single-rule criteria of the smallest in-sample retention over targets and k (each single-rule criterion falls below this for some target and k).",
    )
    for ev, tag in ((EV_NEUTRAL, "neutral"), (EV_STRICT, "strict")):
        Ve = D.V(ev)
        for sname in ("all100", "g72"):
            Me = Ve[sets[sname]].mean(-1)
            for i, r in enumerate(RULES):
                if r == "len":
                    continue
                for k in KS:
                    key = f"sel.{tag}.len_only.{r}.k{k}.{sname}"
                    if key in R.index:
                        continue
                    R.add(
                        key,
                        100 * retention(Me[:, 1], Me[:, i], k),
                        body_set=SET_LABEL[sname],
                        dec=1,
                        post_hoc=PH_REV if sname == "all100" else PH_YES,
                        evaluator=EV_LABEL[ev],
                        definition=f"Standard-only top-{k} selection keeps this percentage of the {RULE_NAME[r]} oracle top-{k} mean PSR under the {tag} evaluator.",
                    )
    for k in KS:
        ret = heldout_batch(V[None], k)[0]
        R.add(
            f"sel.heldout.rf.k{k}.all100",
            100 * float(ret.min(axis=-1)[SEL5.index("rf")]),
            body_set="100",
            dec=1,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"Held-out seed, Reset-free-only selection: fold-averaged retention, smallest over the three rules (k = {k}, percent).",
        )
    t5 = np.argsort(-M[:, 1], kind="stable")[:5]
    R.add(
        "sel.len_top5.lowest_strict_psr",
        float(M[t5[np.argmin(M[t5, 2])], 2]),
        body_set="100",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Three-seed mean Strict PSR of the Standard top-5 body with the lowest Strict PSR.",
    )
    R.section = "Leave-one-rule-out selection test (planned, failed)"
    robust = ("RobustMin", "RobustMean", "Borda", "MinimaxRank")
    for sname in ("all100", "g72"):
        Ms = V[sets[sname]].mean(-1)
        scores = {r: Ms[:, i] for i, r in enumerate(RULES)}
        for hold in RULES:
            train = [r for r in RULES if r != hold]
            rules = lopo_rules(scores, train)
            tgt = scores[hold]
            rs = {r: topk_mean(rules[r], tgt, 5) for r in robust}
            ns = {f"by_{p}": topk_mean(rules[f"by_{p}"], tgt, 5) for p in train}
            br, bn = max(rs, key=rs.get), max(ns, key=ns.get)
            if sname == "all100":
                R.add(
                    f"lopo.{hold}_held.best_pair_k5.psr",
                    [float(rs[br]), float(ns[bn])],
                    body_set="100",
                    dec=3,
                    post_hoc=PH_PLAN,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition="Top-5 mean held-out PSR of the best combination rule and of the best single rule (lopo.*.best_pair_k5).",
                )
            else:
                R.add(
                    f"lopo.g72.{hold}_held.best_gain_k5",
                    float(rs[br] - ns[bn]),
                    body_set=SET_LABEL["g72"],
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition="On the 72: best combination rule minus best single rule, top-5 mean held-out PSR.",
                )


def extra_repeat(D: Data, R: Registry, args, sets, extra):
    R.section = "Same-rule repeat (tau = 0 vs tau = 0.10, post hoc)"
    R.add(
        "repeat.acting.z0",
        [float(D.z0[D.bodies.index(b)]) for b in ("vt-5506-15-9-02-19-03-39", "vt-1409-15-5-02-21-44-24")],
        body_set="2",
        dec=2,
        post_hoc=PH_YES,
        definition="Reset heights of the two VT bodies on which tau = 0.10 changed training (m).",
    )


def extra_sweep(D: Data, R: Registry, args, sets, extra):
    R.section = "Five-threshold sweep"
    T0, T1 = D.get("0p00").mean(1), D.get("0p10").mean(1)
    mvt = np.where(D.fam == "mvt")[0]
    order = mvt[np.argsort(-D.z0[mvt], kind="stable")]
    for tag, sel in (("tall4", order[:4]), ("rest18", order[4:])):
        R.add(f"sweep.mvt_{tag}.n", int(len(sel)), body_set="22 MVT", post_hoc=PH_YES, definition=f"MVT bodies in the group ({'the four tallest' if tag == 'tall4' else 'the others'}).")
        R.add(f"sweep.mvt_{tag}.z0_range", [float(D.z0[sel].min()), float(D.z0[sel].max())], body_set="22 MVT", dec=2, post_hoc=PH_YES, definition="Their reset heights (m).")
        R.add(
            f"sweep.mvt_{tag}.psr_tau0_tau010",
            [float(T0[sel].mean()), float(T1[sel].mean())],
            body_set="22 MVT",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Mean three-seed PSR after tau = 0 and after tau = 0.10 training.",
        )


def extra_exemplars(D: Data, R: Registry, args, sets, extra):
    R.section = "Illustrative bodies (recorded corpus runs)"
    M = D.V().mean(-1)
    rank = {r: {i: k + 1 for k, i in enumerate(np.argsort(-M[:, j], kind="stable"))} for j, r in enumerate(RULES)}
    dist = {r: D.get(TAU_DIR[r], EV_NEUTRAL, "distance") for r in RULES}
    fa = D.bodies.index("floor-5506-10-6-01-15-48-35")
    R.add(
        "ex.floor_a.dist.rf",
        dist["rf"][fa].tolist(),
        body_set="1",
        dec=1,
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="floor-5506-10-6-01-15-48-35 (Floor A of the reviewed version): mean distance (m) per seed after Reset-free training.",
    )
    ma = D.bodies.index("mvt-5506-13-10-17-12-25-45")
    R.add("ex.mvt_a.z0", float(D.z0[ma]), body_set="1", dec=2, post_hoc=PH_YES, definition="MVT A reset height (m).")
    for r in RULES:
        R.add(
            f"ex.mvt_a.rank.{r}",
            rank[r][ma],
            body_set="100",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"MVT A: rank among 100 by three-seed mean PSR after {RULE_NAME[r]} training (1 = best).",
        )
    for alias, b in (("floor_b", "floor-1409-0-7-01-15-15-26"), ("mvt_a", "mvt-5506-13-10-17-12-25-45")):
        i = D.bodies.index(b)
        R.add(
            f"ex.{alias}.walking_runs",
            int(sum((dist[r][i] >= 5).sum() for r in RULES)),
            body_set="1",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_NEUTRAL],
            definition="Of its 9 recorded corpus runs, runs whose controllers travel at least 5 m per episode on average.",
        )
    R.add(
        "ex.illustrative_pair.walking_runs",
        int(sum((dist[r][D.bodies.index(b)] >= 5).sum() for r in RULES for b in ("floor-1409-0-7-01-15-15-26", "mvt-5506-13-10-17-12-25-45"))),
        body_set="2",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Of the 18 recorded corpus runs of Floor B and MVT A, runs whose controllers travel at least 5 m per episode on average.",
    )
    gt = Path(args.gait_traces)
    if gt.exists():
        lens = set()
        for alias, b in (("floor_a", "floor-5506-10-6-01-15-48-35"), ("floor_b", "floor-1409-0-7-01-15-15-26"), ("mvt_a", "mvt-5506-13-10-17-12-25-45")):
            for r in RULES:
                p = gt / f"{b}_tau{TAU_DIR[r]}.traces.npz"
                if not p.exists():
                    log(f"gait trace {p} not found")
                    continue
                with np.load(p, allow_pickle=False) as tr:
                    q = tr["qpos_trace"]
                    done = int(tr["done_step"])
                lens.add(int(q.shape[0] - 1))
                # Same endpoint as Figure 5 (scripts/figures/render_temporal_gait.py): a rollout that ended
                # early is measured up to its done step, not over the frames recorded after it.
                end = min(done, q.shape[0] - 1) if done > 0 else q.shape[0] - 1
                R.add(
                    f"ex.retrain.{alias}.{r}.dx",
                    float(q[end, 0] - q[0, 0]),
                    body_set="1",
                    dec=2,
                    post_hoc=PH_YES,
                    evaluator="illustrative retrain (later code, seed 42, deterministic mean action)",
                    definition=f"{b}: forward torso displacement (m) of the controller retrained once after {RULE_NAME[r]} training, over the recorded rollout up to its early end if it ended early, as plotted in Figure 5 (gait-traces).",
                )
                if done >= 0:
                    R.add(
                        f"ex.retrain.{alias}.{r}.done_step",
                        done,
                        body_set="1",
                        post_hoc=PH_YES,
                        evaluator="illustrative retrain (later code, seed 42, deterministic mean action)",
                        definition="Control step at which the recorded rollout ended early (fall).",
                    )
        if lens:
            R.add("ex.retrain.steps", sorted(lens), body_set="3 bodies x 3 rules", dec=0, post_hoc=PH_YES, definition="Control steps of the recorded illustrative rollouts.")
    else:
        log(f"gait traces not found at {gt}; illustrative retrain rows skipped")
    vt = np.where(D.fam == "vt")[0]
    V = D.V()

    def specialists(Mx):
        out = {}
        for j, r in enumerate(RULES):
            others = [k for k in range(3) if k != j]
            out[r] = int(vt[int(np.argmax(Mx[vt, j] - Mx[vt][:, others].max(1)))])
        return out

    full = specialists(M)
    folds = [specialists(V[..., [t for t in range(3) if t != s]].mean(-1)) for s in range(3)]
    for r in RULES:
        R.add(
            f"ex.vt_specialist.{r}.id",
            D.bodies[full[r]],
            body_set="37 VT",
            kind="text",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition=f"VT body whose three-seed PSR after {RULE_NAME[r]} training most exceeds its best PSR under the other two rules.",
        )
        R.add(
            f"ex.vt_specialist.{r}.loso_recovered",
            int(sum(f[r] == full[r] for f in folds)),
            body_set="37 VT",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Of the 3 leave-one-seed-out folds, folds whose two-seed means pick the same specialist.",
        )
    R.add(
        "ex.vt_specialist.len.strict_psr",
        float(M[full["len"], 2]),
        body_set="1",
        post_hoc=PH_YES,
        evaluator=EV_LABEL[EV_COMMON],
        definition="Three-seed mean PSR after Strict training of the Standard specialist.",
    )


def extra_distance(D: Data, R: Registry, args, sets, extra):
    R.section = "Distance, episode length and control effort"
    for ev, tag in ((EV_COMMON, "common"), (EV_NEUTRAL, "neutral")):
        dist_ = D.V(ev, "distance")
        fit_ = D.V(ev, "fitness")
        with np.errstate(divide="ignore", invalid="ignore"):
            a2m = (dist_ / 0.02 - fit_) / 1e-3 / dist_
        walk = dist_ >= 5
        cells = {}
        for f in FAMS:
            for j, r in enumerate(RULES):
                cells[(f, r)] = float(a2m[D.fam == f][:, j, :][walk[D.fam == f][:, j, :]].mean())
        R.add(
            f"dist.effort_per_m.range.{tag}",
            [min(cells.values()), max(cells.values())],
            body_set="runs travelling at least 5 m per episode",
            dec=1,
            post_hoc=PH_YES,
            evaluator=EV_LABEL[ev],
            definition="Range over the 9 family x rule cells of the mean per-run squared action norm per meter, (distance/0.02 minus return)/0.001 over distance.",
        )
        if tag == "common":
            for r in RULES:
                R.add(
                    f"dist.effort_per_m.floor.{r}.common",
                    cells[("floor", r)],
                    body_set="Floor runs travelling at least 5 m per episode",
                    dec=1,
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[ev],
                    definition=f"Floor mean squared action norm per meter after {RULE_NAME[r]} training.",
                )


def extra_panels(D: Data, R: Registry, args, sets, extra):
    res = D.results
    B = args.boot
    R.section = "SAC panel (24 bodies, planned)"
    sac_list = Path(args.sac_bodies)
    if sac_list.exists():
        bodies = sorted(l.strip().removesuffix(".xml") for l in sac_list.read_text().splitlines() if l.strip())
        fam = np.array([b.split("-")[0] for b in bodies])
        recs = {}

        def load(prefix, ev=EV_COMMON):
            A = np.empty((len(bodies), 3, 3))
            for ri, r in enumerate(RULES):
                for si, s in enumerate(SEEDS):
                    if (prefix, r, s) not in recs:
                        recs[(prefix, r, s)] = json.loads((res / f"{prefix}_{TAU_DIR[r]}_seed{s}" / "results.json").read_text())
                    d = recs[(prefix, r, s)]
                    for bi, b in enumerate(bodies):
                        A[bi, ri, si] = d[b][ev]["per_step_reward"]
            return A

        SAC, PPO = load("sac_mjx"), load("mjx_ppo")
        inert = np.array([D.audit[b]["geometric_inert"]["0.20"] for b in bodies])
        ms, mp = SAC.mean(2), PPO.mean(2)
        valid = (ms[:, 0] > 0.05) & (mp[:, 0] > 0.05)
        R.add("sac.n_joint_valid", int(valid.sum()), body_set="24 (SAC panel)", post_hoc=PH_PLAN, definition="Bodies whose three-seed Reset-free PSR exceeds 0.05 under both SAC and PPO.")
        for name, m in (("sac", ms), ("ppo24", mp)):
            for f in FAMS:
                sel = fam == f
                for j, r in enumerate(RULES):
                    R.add(
                        f"sac.{name}.fam_mean.{r}.{f}",
                        float(m[sel, j].mean()),
                        body_set="24 (SAC panel)",
                        post_hoc=PH_PLAN,
                        evaluator=EV_LABEL[EV_COMMON],
                        definition=f"{FAM_NAME[f]} mean of three-seed mean PSR after {RULE_NAME[r]} training ({'SAC' if name == 'sac' else 'PPO'}).",
                    )
                R.add(
                    f"sac.{name}.fam_str_minus_rf.{f}",
                    float((m[sel, 2] - m[sel, 0]).mean()),
                    body_set="24 (SAC panel)",
                    post_hoc=PH_PLAN,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} mean Strict minus Reset-free PSR ({'SAC' if name == 'sac' else 'PPO'}).",
                )
                ok = valid & sel
                R.add(
                    f"sac.{name}.str_rf_joint.{f}",
                    float(np.mean(m[ok, 2] / m[ok, 0])),
                    body_set=f"{int(valid.sum())} joint-valid SAC bodies",
                    post_hoc=PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"{FAM_NAME[f]} mean per-body Strict/Reset-free ratio on the joint-valid bodies ({'SAC' if name == 'sac' else 'PPO'}).",
                )
            sel = fam == "vt"
            dvt = m[sel, 2] - m[sel, 0]
            R.add(
                f"sac.{name}.vt_str_minus_rf",
                float(dvt.mean()),
                ci=pct_ci(dvt[boot_index(int(sel.sum()), B)].mean(1)),
                body_set="13 VT SAC bodies",
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_COMMON],
                definition="VT mean Strict minus Reset-free PSR with a 95% bootstrap interval over the 13 VT bodies.",
            )
        rs, rp = ms[valid, 2] / ms[valid, 0], mp[valid, 2] / mp[valid, 0]
        R.add("sac.cross_rho_p", float(stats.spearmanr(rs, rp).pvalue), body_set=f"{int(valid.sum())} joint-valid SAC bodies", kind="p", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Parametric p of the SAC vs PPO Spearman rho.")
        loo = [rho(np.delete(rs, i), np.delete(rp, i)) for i in range(len(rs))]
        R.add("sac.cross_rho_loo_range", [min(loo), max(loo)], body_set=f"{int(valid.sum())} joint-valid SAC bodies", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Range of the SAC vs PPO rho when one body at a time is left out.")
        vtv = fam[valid] == "vt"
        bi = boot_index(int(vtv.sum()), B)
        R.add(
            "sac.cross_rho_vt_boot",
            rho(rs[vtv], rp[vtv]),
            ci=pct_ci(spearman_last(rs[vtv][bi], rp[vtv][bi])),
            body_set="13 VT SAC bodies",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="SAC vs PPO rho within VT with a body-bootstrap interval (value of sac.cross_rho_vt).",
        )
        g = valid & ~inert
        R.add("sac.n_joint_valid_g72", int(g.sum()), body_set="24 (SAC panel)", post_hoc=PH_YES, definition="Joint-valid bodies whose Standard cutoff lies above the torso radius.")
        R.add("sac.cross_rho_g72", rho(ms[g, 2] / ms[g, 0], mp[g, 2] / mp[g, 0]), body_set=f"{int(g.sum())} joint-valid SAC bodies in the 72", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="SAC vs PPO rho of the per-body Strict/Reset-free ratios on those bodies.")
        for f in FAMS:
            R.add(f"sac.n_inert.{f}", int((inert & (fam == f)).sum()), body_set="24 (SAC panel)", post_hoc=PH_YES, definition=f"Standard-inert {FAM_NAME[f]} bodies in the panel.")
        ident = 0
        for r in RULES:
            for s in SEEDS:
                d = recs[("sac_mjx", r, s)]
                ident += sum(int(d[b][EV_COMMON] == d[b][EV_NEUTRAL]) for b in np.array(bodies)[inert])
        R.add("sac.inert_identical_policies", ident, body_set=f"{int(inert.sum())} Standard-inert SAC bodies", post_hoc=PH_YES, evaluator="common vs neutral", definition="SAC policies of the Standard-inert bodies whose common-evaluator record equals the neutral record.")
        R.add("sac.n_noninert", int((~inert).sum()), body_set="24 (SAC panel)", post_hoc=PH_YES, definition="Bodies whose Standard cutoff lies above the torso radius.")
        for f in FAMS:
            ok = ~inert & (fam == f) & (ms[:, 0] > 0.05)
            R.add(f"sac.sac.str_rf_noninert.{f}", float(np.mean(ms[ok, 2] / ms[ok, 0])), body_set="16 SAC bodies where Standard can act", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition=f"SAC {FAM_NAME[f]} mean per-body Strict/Reset-free ratio without the Standard-inert bodies.")
        sd = [float(np.median(SAC[:, j, :].std(axis=1))) for j in range(3)]
        for j, r in enumerate(RULES):
            R.add(f"sac.sac.median_seed_sd.{r}", sd[j], body_set="24 (SAC panel)", post_hoc=PH_PLAN, evaluator=EV_LABEL[EV_COMMON], definition=f"Median over bodies of the seed standard deviation (population) of PSR after {RULE_NAME[r]} training.")
        R.add("sac.sac.reliability_threshold", 2 * max(sd[0], sd[2]), body_set="24 (SAC panel)", post_hoc=PH_PLAN, evaluator=EV_LABEL[EV_COMMON], definition="Planned reliability threshold: twice the larger of the Reset-free and Strict median seed SDs (records/sac_full_panel_spec.md).")
        R.add("sac.n_valid_runs", int(np.isfinite(SAC).sum()), body_set="24 (SAC panel)", post_hoc=PH_PLAN, definition="SAC runs with a finite common-evaluator PSR.")
        short = [b for b in bodies if b in D.bodies and extra["vt10"][D.bodies.index(b)] and np.isclose(D.z0[D.bodies.index(b)], 0.30)]
        if short:
            Ls = [recs[("sac_mjx", "str", s)][b][EV_STRICT]["mean_ep_len"] for b in short for s in SEEDS]
            R.add(
                "sac.short_cutoff.strict_eval_len_range",
                [float(min(Ls)), float(max(Ls))],
                body_set=f"{len(short)} short VT cut-off bodies in the SAC panel",
                dec=1,
                post_hoc=PH_YES,
                evaluator=EV_LABEL[EV_STRICT],
                definition="Range over their Strict-trained SAC seeds of the mean Strict-evaluator episode length (steps).",
            )
        m44 = SAC[:, :, 2]
        for f in FAMS:
            ok = (fam == f) & (m44[:, 0] > 0.05)
            R.add(f"sac.sac.str_rf_seed44.{f}", float(np.mean(m44[ok, 2] / m44[ok, 0])), body_set="24 (SAC panel)", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition=f"SAC {FAM_NAME[f]} mean per-body Strict/Reset-free ratio from seed 44 alone (bodies with seed-44 Reset-free PSR above 0.05).")
        v = SAC[:, [1, 2, 0], :]
        F, q, df = BlockedF(v).F(fam)
        R.add("sac.sac.perrun_df", [int(q), int(df)], body_set="24 (SAC panel)", dec=0, post_hoc=PH_PLAN, definition="Degrees of freedom of the planned per-run test.")
        Fc, qc, dfc, pc = cr1_wald(v, fam)
        R.add("sac.sac.cr1_F", Fc, body_set="24 (SAC panel)", dec=2, post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Body-clustered Wald F for rule x family (its p is sac.sac.cr1_p).")
        R.add("sac.sac.cr1_df", [int(qc), int(round(dfc))], body_set="24 (SAC panel)", dec=0, post_hoc=PH_YES, definition="Its degrees of freedom.")
        keep = ~inert
        F, q, df = BlockedF(SAC[keep][:, [1, 2, 0], :]).F(fam[keep])
        R.add("sac.sac.perrun_noninert_F", F, body_set="16 SAC bodies where Standard can act", dec=2, post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Per-run F without the Standard-inert bodies (its p is sac.sac.perrun_p_noninert).")
        R.add("sac.sac.perrun_noninert_df", [int(q), int(df)], body_set="16 SAC bodies where Standard can act", dec=0, post_hoc=PH_YES, definition="Its degrees of freedom.")
        F, q, df = BlockedF(SAC[:, [2, 0], :]).F(fam)
        R.add("sac.sac.perrun_rf_str_F", F, body_set="24 (SAC panel)", dec=2, post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Per-run rule x family F with Reset-free and Strict training only.")
        R.add("sac.sac.perrun_rf_str_df", [int(q), int(df)], body_set="24 (SAC panel)", dec=0, post_hoc=PH_YES, definition="Its degrees of freedom.")
        R.add("sac.sac.perrun_rf_str_p", float(stats.f.sf(F, q, df)), body_set="24 (SAC panel)", kind="p", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Its p.")
        # body-level tests of the full panel (seed-averaged model and family-label permutation)
        for name, A in (("sac", SAC), ("ppo24", PPO)):
            it = interaction_tests(A[:, [1, 2, 0], :], fam, args.perm, want_cr1=False)
            R.add(f"sac.{name}.seedmean_F", it["seedmean_F"], body_set="24 (SAC panel)", dec=2, post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Rule x family F on three-seed means (seed-averaged model).")
            R.add(f"sac.{name}.seedmean_df", [int(x) for x in it["seedmean_df"]], body_set="24 (SAC panel)", dec=0, post_hoc=PH_YES, definition="Its degrees of freedom.")
            R.add(f"sac.{name}.seedmean_p", it["seedmean_p"], body_set="24 (SAC panel)", kind="p", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Its p.")
            R.add(f"sac.{name}.perm_p", it["perm_p"], body_set="24 (SAC panel)", kind="p", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Body-level permutation p (family labels permuted across bodies, statistic the per-run F).")
        # the 96-run pilot: Reset-free and Strict training with seeds 42 and 43 (part of the 216 runs)
        pil = SAC[:, [0, 2], :2]
        R.add("sac.pilot.runs", int(pil.size), body_set="24 (SAC panel), seeds 42 and 43", post_hoc=PH_YES, definition="Runs of the pilot: Reset-free and Strict training with seeds 42 and 43 (trained 9 April 2026, part of the 216).")
        mpil = pil.mean(2)
        for f in FAMS:
            ok = (fam == f) & (mpil[:, 0] > 0.05)
            R.add(f"sac.pilot.str_rf.{f}", float(np.mean(mpil[ok, 1] / mpil[ok, 0])), body_set="24 (SAC panel), seeds 42 and 43", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition=f"Pilot (analyzed 9 April 2026, before the decision rule): {FAM_NAME[f]} mean per-body Strict/Reset-free ratio of two-seed means (Reset-free PSR above 0.05).")
        F, q, df = BlockedF(pil).F(fam)
        R.add("sac.pilot.perrun_F", F, body_set="24 (SAC panel), seeds 42 and 43", dec=2, post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], note=f"df [{q}, {df}]", definition="Pilot: per-run rule x family F (96 runs).")
        R.add("sac.pilot.perrun_df", [int(q), int(df)], body_set="24 (SAC panel), seeds 42 and 43", dec=0, post_hoc=PH_YES, definition="Its degrees of freedom.")
        mpp = PPO[:, [0, 2], :].mean(2)
        okp = (mpil[:, 0] > 0.05) & (mpp[:, 0] > 0.05)
        R.add("sac.pilot.cross_rho", rho(mpil[okp, 1] / mpil[okp, 0], mpp[okp, 1] / mpp[okp, 0]), body_set=f"{int(okp.sum())} pilot bodies valid under both", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Pilot: Spearman rho between SAC per-body Strict/Reset-free ratios of two-seed means and the PPO corpus ratios of three-seed means.")
    else:
        log(f"SAC body list {sac_list} not found; SAC additions skipped")
    M = D.V().mean(-1)
    ok = M[:, 0] > 0.05
    R.add("sac.corpus_ppo.n_valid", int(ok.sum()), body_set="100", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="PPO corpus bodies whose three-seed Reset-free PSR exceeds 0.05.")
    for f in FAMS:
        sel = ok & (D.fam == f)
        R.add(f"sac.corpus_ppo.n_valid.{f}", int(sel.sum()), body_set="100", post_hoc=PH_YES, definition=f"Of those, {FAM_NAME[f]} bodies.")
        R.add(f"sac.corpus_ppo.str_rf.{f}", float(np.mean(M[sel, 2] / M[sel, 0])), body_set="100", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition=f"PPO corpus {FAM_NAME[f]} mean per-body Strict/Reset-free ratio on those bodies.")

    R.section = "No-time-limit panel (30 bodies, planned, later code)"
    bfj = Path(args.bf_bodies)
    if bfj.exists():
        bb = [b["body"] for b in json.loads(bfj.read_text())["panel"]]
        R.add("bf.n_bodies", len(bb), body_set="30 (no-time-limit panel)", post_hoc=PH_PLAN, definition="Bodies in the panel without a time limit.")
        for f in FAMS:
            R.add(f"bf.n_bodies.{f}", int(sum(b.startswith(f + "-") for b in bb)), body_set="30 (no-time-limit panel)", post_hoc=PH_PLAN, definition=f"{FAM_NAME[f]} bodies in it.")
    ad = res / "asymptotic_5m"
    if ad.exists():
        drift = []
        for p in sorted(ad.iterdir()):
            b = p.name
            if not b.startswith("floor-") or b not in D.bodies:
                continue
            vals = []
            for s in SEEDS:
                q = p / f"tau0p00_seed{s}" / "results.json"
                if q.exists():
                    rec = json.loads(q.read_text())[b]
                    if rec["threshold"] != 0.0 or rec["seed"] != s:
                        raise ValueError(f"unexpected record {q}")
                    vals.append(rec[EV_COMMON]["per_step_reward"])
            if len(vals) == 3:
                drift.append(np.mean(vals) - M[D.bodies.index(b), 0])
        R.add("asym.rf_drift.floor.n", len(drift), body_set="Floor bodies of the longer-budget panel (5M-step arm)", post_hoc=PH_YES, definition="Floor bodies with all three Reset-free seeds in the 5M-step arm.")
        R.add(
            "asym.rf_drift.floor.median",
            float(np.median(drift)),
            body_set="Floor bodies of the longer-budget panel (5M-step arm)",
            post_hoc=PH_YES,
            evaluator=EV_LABEL[EV_COMMON],
            definition="Median over those bodies of later-code minus corpus three-seed Reset-free PSR (same 5M steps and 1,000-step limit).",
        )

    R.section = "12-body retraining with later code (planned, failed)"
    bj = Path(args.bridge_bodies)
    if bj.exists():
        names = [b["name"] for b in json.loads(bj.read_text())["bodies"]]
        R.add("bridge.n_bodies", len(names), body_set="12 (bridge)", post_hoc=PH_PLAN, definition="Bodies in the retraining panel.")
        for f in FAMS:
            R.add(f"bridge.n_bodies.{f}", int(sum(b.startswith(f + "-") for b in names)), body_set="12 (bridge)", post_hoc=PH_PLAN, definition=f"{FAM_NAME[f]} bodies in it.")
        R.add("bridge.criteria_total", 4 * len(RULES) + 2, body_set="12 (bridge)", post_hoc=PH_PLAN, definition="Sub-criteria of the planned decision rule (4 per rule, plus 2 for the Standard minus Reset-free difference).")

    R.section = "Compute (recorded wall time, train plus evaluation)"

    def count(paths):
        tot, n, bods = 0.0, 0, set()
        for p in paths:
            d = json.loads(Path(p).read_text())
            for b, r in d.items():
                if isinstance(r, dict) and "train_time_s" in r:
                    tot += float(r["train_time_s"])
                    n += 1
                    bods.add(b)
        return tot / 3600, n, len(bods)

    panels = (
        ("corpus", [res / f"mjx_ppo_{TAU_DIR[r]}_seed{s}" / "results.json" for r in RULES for s in SEEDS]),
        ("tau010_035", [res / f"mjx_ppo_{t}_seed{s}" / "results.json" for t in ("0p10", "0p35") for s in SEEDS]),
        ("sac", [res / f"sac_mjx_{TAU_DIR[r]}_seed{s}" / "results.json" for r in RULES for s in SEEDS]),
        ("bridge", sorted(res.glob("equivalence_pilot/jax/*/*/results.json"))),
        ("no_time_limit", sorted(res.glob("boundary_free/*/results.json"))),
        ("boundary_duration", sorted(res.glob("boundary_sweep/*/results.json"))),
        ("factorial", sorted(res.glob("factorial/*/results.json"))),
        ("contact", sorted(res.glob("alt_termination/*/results.json"))),
        ("non_evolved", sorted(res.glob("nems_panel/*/results.json"))),
        ("asymptotic", sorted(res.glob("asymptotic_5m/*/*/results.json"))),
        ("menagerie", sorted(res.glob("menagerie_panel/*/results.json"))),
    )
    total = 0
    for tag, paths in panels:
        paths = [p for p in paths if Path(p).exists()]
        if not paths:
            log(f"compute: no records for {tag}")
            continue
        h, n, nb = count(paths)
        total += n
        if tag != "corpus":
            R.add(f"compute.{tag}.runs", n, body_set=f"{n} runs", post_hoc=PH_NA, definition=f"Runs with a recorded time in the {tag.replace('_', ' ')} records.")
        R.add(f"compute.{tag}.bodies", nb, body_set=f"{n} runs", post_hoc=PH_NA, definition="Distinct bodies (robots for the existing-robot panel) in those records.")
        if tag in ("factorial", "contact", "menagerie"):
            R.add(f"compute.{tag}.hours", h, body_set=f"{n} runs", dec=1, post_hoc=PH_NA, definition=f"Summed wall time of the {tag.replace('_', ' ')} records (h).")
    names = sorted(p.parent.name for p in res.glob("menagerie_panel/*/results.json"))
    if names:
        R.add("compute.menagerie.seeds", len({n.rsplit("_seed", 1)[1] for n in names}), body_set="60 runs", post_hoc=PH_NA, definition="Seeds per robot and rule in the stopped existing-robot panel.")
    R.add("compute.release_runs", total, body_set="all run records", post_hoc=PH_NA, definition="Runs with a recorded time over all eleven panels in the run records (the dataset's run summaries).")
    vals, fin = 0, 0
    for t in ("0p00", "0p20", "0p50"):
        for s in SEEDS:
            for b, r in D.raw[(t, s)].items():
                for ev in (EV_COMMON, EV_NEUTRAL, EV_STRICT):
                    for x in r[ev].values():
                        vals += 1
                        fin += int(np.isfinite(x))
    R.add("records.corpus.eval_values", vals, body_set="900 corpus runs", post_hoc=PH_NA, definition="Stored evaluation values (900 runs x 3 evaluators x 9 fields).")
    R.add("records.corpus.eval_values_finite", fin, body_set="900 corpus runs", post_hoc=PH_NA, definition="Of those, finite values.")
    tt = [D.raw[(TAU_DIR[r], s)][b]["train_time_s"] for r in RULES for s in SEEDS for b in D.bodies]
    R.add("records.corpus.wall_time_s_range", [float(min(tt)), float(max(tt))], body_set="900 corpus runs", dec=0, post_hoc=PH_NA, definition="Range of recorded run times (s).")
    R.add(
        "records.corpus.rf_neutral_min_ep_len",
        float(D.get("0p00", EV_NEUTRAL, "mean_ep_len").min()),
        body_set="300 Reset-free corpus runs",
        dec=0,
        post_hoc=PH_NA,
        evaluator=EV_LABEL[EV_NEUTRAL],
        definition="Shortest mean neutral-evaluator episode of a Reset-free-trained corpus controller (steps).",
    )


def extra_structure(D: Data, R: Registry, args, sets):
    R.section = "Morphological descriptors (descriptive)"
    B = args.boot
    feats = ("init_height", "n_first_level_limbs", "tree_depth", "n_joints", "n_bodies")
    V = D.V()
    for sname in ("all100", "g72"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        rf_len, str_len = M[:, 0] / M[:, 1], M[:, 2] / M[:, 1]
        X = {k: np.array([D.feats[b][k] for b in np.array(D.bodies)[mm]], float) for k in feats}
        for k in feats:
            for lab, y in (("rf_len", rf_len), ("str_len", str_len)):
                R.add(
                    f"struct.spearman.{k}.{lab}.{sname}.p",
                    float(stats.spearmanr(X[k], y).pvalue),
                    body_set=SET_LABEL[sname],
                    kind="p",
                    post_hoc=PH_REV if sname == "all100" else PH_YES,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition=f"p of the Spearman rho between {k} and {lab.replace('_', '/')} retention.",
                )
        ps = [R.get(f"struct.spearman.{k}.{lab}.{sname}.p")["value"] for k in feats for lab in ("rf_len", "str_len")]
        R.add(
            f"struct.spearman.n_bonferroni.{sname}",
            int(sum(p * len(ps) < 0.05 for p in ps)),
            body_set=SET_LABEL[sname],
            post_hoc=PH_YES,
            definition="Feature correlations (of the ten) that survive a Bonferroni correction at 0.05.",
        )
    for sname in ("all100", "g72"):
        mm = sets[sname]
        M = V[mm].mean(-1)
        X = np.column_stack([[D.feats[b][k] for b in np.array(D.bodies)[mm]] for k in feats]).astype(float)
        y = M[:, 2] - M[:, 0]
        idx = boot_index(len(y), B)

        def fit(Xm):
            Xm = np.column_stack([Xm, np.ones(len(y))])
            beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
            r_ = y - Xm @ beta
            n_, p_ = Xm.shape
            return 1 - r_ @ r_ / np.sum((y - y.mean()) ** 2), n_ * math.log(r_ @ r_ / n_) + 2 * p_, Xm @ beta, beta

        for lab, Xm in (("M1", M[:, 1:2]), ("M2", X), ("M3", np.column_stack([M[:, 1], X])), ("Mh", X[:, :1])):
            r2, aic, pred, _ = fit(Xm)
            ph = PH_REV if (sname == "all100" and lab != "Mh") else PH_YES
            if sname == "g72" or lab == "Mh":
                R.add(f"struct.{lab}.R2.{sname}", float(r2), body_set=SET_LABEL[sname], post_hoc=ph, evaluator=EV_LABEL[EV_COMMON], definition=f"In-sample R^2 of {lab} for Strict minus Reset-free PSR (Mh: reset height alone).")
            if sname == "all100":
                R.add(f"struct.{lab}.aic.{sname}", float(aic), body_set=SET_LABEL[sname], dec=1, post_hoc=ph, evaluator=EV_LABEL[EV_COMMON], definition="AIC of the Gaussian linear model, n log(RSS/n) + 2p with p counting the intercept.")
                R.add(
                    f"struct.{lab}.rho_boot.{sname}",
                    rho(pred, y),
                    ci=pct_ci(spearman_last(pred[idx], y[idx])),
                    body_set=SET_LABEL[sname],
                    post_hoc=ph,
                    evaluator=EV_LABEL[EV_COMMON],
                    definition="Spearman rho between fitted and observed values with a body-bootstrap interval (fitted values held fixed).",
                )
        if sname != "all100":
            continue
        Xs = (X - X.mean(0)) / X.std(0, ddof=1)
        _, _, _, beta = fit(Xs)
        for k, b_ in zip(feats, beta[:5]):
            R.add(f"struct.M2.std_coef.{k}.all100", float(b_), body_set="100", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="M2 coefficient per sample standard deviation of the feature (outcome in PSR).")
        R.add("struct.spearman_dstrrf.n_bodies.all100", rho(X[:, 4], y), body_set="100", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="Spearman rho between body count and Strict minus Reset-free PSR.")
        r2, _, _, _ = fit(X[:, :4])
        R.add("struct.M2_without_n_bodies.R2.all100", float(r2), body_set="100", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="In-sample R^2 of M2 without the body count.")
        fam = D.fam
        R.add("struct.dstrrf.oneway_p", float(stats.f_oneway(*[y[fam == f] for f in FAMS]).pvalue), body_set="100", kind="p", post_hoc=PH_YES, evaluator=EV_LABEL[EV_COMMON], definition="p of the one-way ANOVA of Strict minus Reset-free PSR across families (its F is struct.dstrrf.oneway_F).")


# ----------------------------------------------------------------- records added 2026-10-10
# Rows behind values first printed in camera-ready round 3: the upper height bound in the run summaries, the kept
# training logs (logs.zip of the dataset) and the per-cell tables of the review-period panels (panels.zip). They are
# counts, sums and Spearman correlations without random draws, so no random stream moves, and they are added after
# every other row. Each input is optional: without it the rows are skipped with a log line.
N_EVAL = 20  # evaluation episodes per evaluator (N_EVAL of the trainers)
EP_LIMIT = 1000  # step limit of every corpus episode
LOG_START = re.compile(r"^\[\d+/\d+\]\s+(\S+)\s+\(custom PPO, threshold=([0-9.]+), seed=(\d+)\)")
LOG_START_ROBOT = re.compile(r"^Menagerie PPO: robot=(\S+) tau=([0-9.]+) seed=(\d+)")
LOG_SKIP = re.compile(r"^\[\d+/\d+\]\s+Skip\b")
LOG_DONE = re.compile(r"^Training done in \d")
LOG_PANELS = (  # (tag of the compute rows, run folders, label), in the order of the paper's provenance table
    ("corpus", ("mjx_ppo_0p00_seed", "mjx_ppo_0p20_seed", "mjx_ppo_0p50_seed"), "Corpus runs (three rules)"),
    ("tau010_035", ("mjx_ppo_0p10_seed", "mjx_ppo_0p35_seed"), "Runs at tau = 0.10 and 0.35"),
    ("sac", ("sac_mjx_",), "SAC runs"),
    ("bridge", ("equivalence_pilot/",), "Runs of the retraining with later code"),
    ("no_time_limit", ("boundary_free/",), "Runs without a time limit"),
    ("boundary_duration", ("boundary_sweep/",), "Runs of the time-limit-duration panel"),
    ("factorial", ("factorial/",), "Runs of the rule-by-time-limit panel"),
    ("contact", ("alt_termination/",), "Runs of the contact-termination panel"),
    ("non_evolved", ("nems_panel/",), "Runs of the non-evolved panel"),
    ("asymptotic", ("asymptotic_5m",), "Runs of the 5M-step arm of the longer-budget panel"),
    ("menagerie", ("menagerie_panel",), "Runs of the stopped existing-robot panel"),
)
REVIEW_PANELS = (  # (folder in panels.zip, key tag, label)
    ("humanoid-fall-rule", "humanoid_fall_rule", "Humanoid fall-rule panel"),
    ("humanoid-time-limit", "humanoid_time_limit", "Humanoid time-limit panel"),
    ("ant-reset-init", "ant_reset_init", "Ant reset-initialization panel"),
    ("ant-breadth", "ant_breadth", "Ant breadth panel"),
    ("evogym-time-limit", "evogym_time_limit", "Evolution Gym time-limit panel"),
)


def _tree_files(path, suffix, top):
    """(relative path, reader) for the files ending in suffix in a folder or a zip archive, sorted by path.

    A leading top folder (results/ or review-panels/, as the dataset archives store them) is dropped."""
    path = Path(path).expanduser()
    out = []
    if path.is_file() and zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        for info in zf.infolist():
            if not info.is_dir() and info.filename.endswith(suffix):
                out.append((info.filename.removeprefix(top + "/"), lambda n=info.filename: zf.read(n)))
    elif path.is_dir():
        for p in path.rglob("*" + suffix):
            if p.is_file():
                out.append((p.relative_to(path).as_posix().removeprefix(top + "/"), p.read_bytes))
    return sorted(out, key=lambda x: x[0])


def _log_tag(folder):
    for tag, prefixes, _ in LOG_PANELS:
        if folder.startswith(prefixes):
            return tag
    return None


def _count_logs(files):
    """Runs that start in a log and runs whose log reaches the end of training.

    A run starts at its start line ("[i/N] <body> (custom PPO, threshold=T, seed=S)", or "Menagerie PPO: robot=R
    tau=T seed=S" for the existing robots) and counts as logged when a later line of the same file, before the next
    start line, reads "Training done in ...". Runs are keyed by run folder (the path above logs/), body or robot,
    tau and seed, so a run that appears in two log files counts once."""
    started, done = set(), set()
    for rel, read in files:
        folder = rel.rsplit("/logs/", 1)[0]
        cur = None
        for line in read().decode("utf-8", "replace").splitlines():
            s = line.strip()
            m = LOG_START.match(s) or LOG_START_ROBOT.match(s)
            if m:
                cur = (folder, m.group(1), float(m.group(2)), int(m.group(3)))
                started.add(cur)
            elif LOG_SKIP.match(s):
                cur = None
            elif cur is not None and LOG_DONE.match(s):
                done.add(cur)
                cur = None
    return started, done


def _seed_half(M):
    """Median over complementary seed splits (n // 2 against the rest) of the Spearman rho between split means,
    as in the frozen analysis code of the review-period panels (scripts/review_panels.py)."""
    n = M.shape[1]
    every = frozenset(range(n))
    seen, out = set(), []
    for first in itertools.combinations(range(n), n // 2):
        a = sorted(first)
        b = sorted(every - frozenset(first))
        pair = tuple(sorted((tuple(a), tuple(b))))
        if pair in seen:
            continue
        seen.add(pair)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = rho(M[:, a].mean(1), M[:, b].mean(1))
        if np.isfinite(r):
            out.append(r)
    return float(np.median(out))


def _cell_matrix(rows, morphs, seeds):
    by = {(int(r["morphology_index"]), int(r["seed"])): float(r["score"]) for r in rows}
    if set(by) != {(m, s) for m in morphs for s in seeds} or len(by) != len(rows):
        raise ValueError("a review-panel setting is not a complete body by seed table")
    return np.array([[by[(m, s)] for s in seeds] for m in morphs], float)


def extra_records(D: Data, R: Registry, args):
    res = D.results
    # ---------------- upper height bound
    R.section = "Upper height bound (run summaries, post hoc)"
    hits = []
    for r in RULES:
        for s in SEEDS:
            for b in D.bodies:
                rec = D.raw[(TAU_DIR[r], s)][b]
                if rec[EV_NEUTRAL]["mean_ep_len"] < EP_LIMIT or rec[EV_NEUTRAL]["fall_rate"] > 0:
                    hits.append((b, r, s, rec))
    n_ctrl = len(RULES) * len(SEEDS) * D.n
    R.add(
        "ub.corpus.controllers_ended",
        len(hits),
        body_set=f"{n_ctrl} corpus runs",
        evaluator=EV_LABEL[EV_NEUTRAL],
        post_hoc=PH_YES,
        definition="Corpus controllers with at least one neutral-evaluator episode shorter than the 1,000-step limit. The neutral evaluator has no fall clause, so only the upper height bound max(3 z0, 2 m) can end its episodes early.",
    )
    ids, ends, same = [], [], []
    for b, r, s, rec in hits:
        x = rec[EV_NEUTRAL]["fall_rate"] * N_EVAL
        if abs(x - round(x)) > 1e-9:
            raise ValueError(f"neutral fall rate of {b} ({r}, seed {s}) is not a multiple of 1/{N_EVAL}")
        ids.append(f"{b} ({RULE_NAME[r]}, seed {s})")
        ends.append(int(round(x)))
        same.append(bool(rec[EV_COMMON] == rec[EV_NEUTRAL]))
    R.add("ub.corpus.ended_ids", ids, body_set=f"{n_ctrl} corpus runs", evaluator=EV_LABEL[EV_NEUTRAL], post_hoc=PH_YES, definition="Those controllers (body, training rule, seed).")
    R.add(
        "ub.corpus.episodes_ended",
        ends,
        body_set=f"{n_ctrl} corpus runs",
        evaluator=EV_LABEL[EV_NEUTRAL],
        post_hoc=PH_YES,
        definition=f"Episodes of each that ended before the limit, of its {N_EVAL} neutral-evaluator episodes (fall_rate x {N_EVAL}).",
    )
    R.add(
        "ub.corpus.common_equals_neutral",
        same,
        body_set=f"{n_ctrl} corpus runs",
        evaluator="common vs neutral",
        post_hoc=PH_YES,
        definition="Whether the common-evaluator record of each equals its neutral record field by field. The evaluators share their random streams, so equal records mean that the same episodes ended under both.",
    )
    bf = sorted(res.glob("boundary_free/seed*/results.json"))
    if bf:
        n, ended = 0, 0
        for p in bf:
            for b, rec in json.loads(p.read_text()).items():
                if rec["threshold"] != 0.0 or rec["max_steps"] <= rec["timesteps"]:
                    raise ValueError(f"unexpected record in the panel without a time limit: {p} {b}")
                n += 1
                ended += int(rec["implementation_reset_count"])
        R.add(
            "ub.no_time_limit.training_ends",
            ended,
            body_set=f"{n} runs without a time limit",
            post_hoc=PH_YES,
            definition="Training-episode ends summed over the Reset-free runs without a time limit (implementation_reset_count, every episode end over all environments and updates). These runs have no fall clause and a step limit beyond the budget, so only the upper height bound could end an episode.",
        )
    else:
        log("records of the panel without a time limit not found; its upper-bound row skipped")

    # ---------------- training logs
    R.section = "Training logs kept (run counts)"
    src = Path(args.logs).expanduser() if args.logs else res
    files = [f for f in _tree_files(src, ".log", "results") if "/logs/" in f[0]]
    if files:
        started, done = _count_logs(files)
        other = sorted({k[0] for k in started if _log_tag(k[0]) is None})
        if other:
            log(f"training logs of {len(other)} run folders outside the listed panels are not counted")
        for tag, _, label in LOG_PANELS:
            key = f"compute.{tag}.runs"
            n_runs = R.get(key)["value"] if key in R.index else None
            R.add(
                f"logs.{tag}.runs_logged",
                sum(_log_tag(k[0]) == tag for k in done),
                body_set=f"{n_runs} runs" if n_runs is not None else label,
                post_hoc=PH_NA,
                definition=f"{label} whose kept training log reaches the end of training ('Training done').",
            )
        bfd = [k for k in done if _log_tag(k[0]) == "no_time_limit"]
        R.add(
            "logs.no_time_limit.runs_logged_by_seed",
            [sum(k[3] == s for k in bfd) for s in SEEDS],
            body_set="90 runs without a time limit",
            post_hoc=PH_NA,
            definition="Those runs by seed (42, 43, 44).",
        )
        R.add(
            "logs.no_time_limit.runs_started",
            sum(_log_tag(k[0]) == "no_time_limit" for k in started),
            body_set="90 runs without a time limit",
            post_hoc=PH_NA,
            definition="Runs of that panel that start in a kept log (its log files stop partway through the panel).",
        )
    else:
        log(f"no training logs found in {src}; log counts skipped (unzip logs.zip or pass --logs logs.zip)")

    # ---------------- review-period panels
    R.section = "Review-period panels (per-cell tables)"
    files = dict(_tree_files(args.review_panels, ".csv", "review-panels") + _tree_files(args.review_panels, ".json", "review-panels"))
    need = [f"{p}/cells.csv" for p, _, _ in REVIEW_PANELS] + [f"{p}/panel_spec.json" for p, _, _ in REVIEW_PANELS if p != "humanoid-time-limit"]
    need.append("humanoid-time-limit/composite_spec.json")
    if not all(n in files for n in need):
        log(f"review-period panel tables not found in {args.review_panels}; their rows skipped (unzip panels.zip or pass --review-panels panels.zip)")
        return
    tables = {p: list(csv.DictReader(io.StringIO(files[f"{p}/cells.csv"]().decode("utf-8")))) for p, _, _ in REVIEW_PANELS}
    total = 0
    for p, tag, label in REVIEW_PANELS:
        rows = tables[p]
        total += len(rows)
        pools = sorted({int(r["pool_id"]) for r in rows})
        per_pool = sorted({len({r["morphology_index"] for r in rows if int(r["pool_id"]) == q}) for q in pools})
        secs = []
        for r in rows:
            try:
                x = float(r["train_seconds"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                secs.append(x)
        R.add(f"panels.{tag}.cells", len(rows), body_set=label, post_hoc=PH_NA, definition="Cells of the panel's table, one training run each.")
        R.add(f"panels.{tag}.pools", len(pools), body_set=label, post_hoc=PH_NA, definition="Pools of bodies.")
        R.add(f"panels.{tag}.bodies_per_pool", per_pool[0] if len(per_pool) == 1 else per_pool, body_set=label, post_hoc=PH_NA, definition="Bodies per pool.")
        R.add(f"panels.{tag}.seeds", len({r["seed"] for r in rows}), body_set=label, post_hoc=PH_NA, definition="Training seeds per body and setting.")
        R.add(f"panels.{tag}.hardware", ", ".join(sorted({r["hardware"] for r in rows})), body_set=label, post_hoc=PH_NA, definition="Hardware class of the cells (gh200: NVIDIA GH200, h100: NVIDIA H100, cpu: CPU).")
        R.add(f"panels.{tag}.cells_with_time", len(secs), body_set=label, post_hoc=PH_NA, definition="Cells that store a training wall time (train_seconds).")
        if secs:
            R.add(f"panels.{tag}.train_hours", sum(secs) / 3600, body_set=label, dec=1, post_hoc=PH_NA, definition="Summed training wall time of those cells (h).")
    R.add("panels.cells_total", total, body_set="five review-period panels", post_hoc=PH_NA, definition="Cells (training runs) of the five review-period panels.")
    ev = "the score each frozen report ranks"
    for p, tag, label in REVIEW_PANELS:
        if p == "humanoid-time-limit":
            continue
        spec = json.loads(files[f"{p}/panel_spec.json"]())
        comps = [tuple(c) for c in spec["comparisons"]]
        groups = collections.defaultdict(list)
        for r in tables[p]:
            groups[(int(r["pool_id"]), r["task"])].append(r)
        rhos, halves = [], []
        for g in sorted(groups):
            rs = groups[g]
            morphs = sorted({int(r["morphology_index"]) for r in rs})
            seeds = sorted({int(r["seed"]) for r in rs})
            by = collections.defaultdict(list)
            for r in rs:
                by[r["protocol"]].append(r)
            M = {k: _cell_matrix(v, morphs, seeds) for k, v in by.items()}
            for a, b in comps:
                rhos.append(rho(M[a].mean(1), M[b].mean(1)))
                halves += [_seed_half(M[a]), _seed_half(M[b])]
        R.add(
            f"panels.{tag}.rho_range",
            [min(rhos), max(rhos)],
            body_set=label,
            evaluator=ev,
            dec=4,
            post_hoc=PH_PLAN,
            definition=f"Spearman rho between the seed-mean rankings of the two settings, as a range over the panel's {len(rhos)} declared comparison{'s' if len(rhos) > 1 else ''} (one per pool{', task' if len({g[1] for g in groups}) > 1 else ''} and pair of settings).",
        )
        R.add(
            f"panels.{tag}.seed_half_range",
            [min(halves), max(halves)],
            body_set=label,
            evaluator=ev,
            dec=4,
            post_hoc=PH_PLAN,
            definition="Range over the settings in those comparisons of the seed-half reference: the median over complementary seed splits (two against the rest) of the rho between the split means.",
        )
    cs = json.loads(files["humanoid-time-limit/composite_spec.json"]())
    shared = cs["shared_contract"]
    morphs, seeds = shared["morphology_indices"], shared["seeds"]
    child = {"bootstrap": tables["humanoid-fall-rule"], "terminal": tables["humanoid-time-limit"]}
    prim, halves = {"native": [], "none": []}, []
    for pool in shared["pool_ids"]:
        for rule in prim:
            Ms = [
                _cell_matrix([r for r in child[arm] if int(r["pool_id"]) == pool and r["protocol"] == cs["protocol_mapping"][arm][rule]], morphs, seeds)
                for arm in ("bootstrap", "terminal")
            ]
            prim[rule].append(rho(Ms[0].mean(1), Ms[1].mean(1)))
            halves += [_seed_half(M) for M in Ms]
    both = prim["native"] + prim["none"]
    lab = "Humanoid time-limit panel (bootstrapped arms from the fall-rule panel)"
    R.add(
        "panels.humanoid_time_limit.rho_range",
        [min(both), max(both)],
        body_set=lab,
        evaluator=ev,
        dec=4,
        post_hoc=PH_PLAN,
        definition=f"Range over its {len(both)} primary comparisons (three pools, native fall rule and none) of the Spearman rho between the seed-mean rankings with terminal and with bootstrapped time limits.",
    )
    for rule in prim:
        R.add(
            f"panels.humanoid_time_limit.rho_range_{rule}",
            [min(prim[rule]), max(prim[rule])],
            body_set=lab,
            evaluator=ev,
            dec=4,
            post_hoc=PH_PLAN,
            definition=f"The same over the three pools with {'the native fall rule' if rule == 'native' else 'no fall rule'}.",
        )
    R.add(
        "panels.humanoid_time_limit.seed_half_range",
        [min(halves), max(halves)],
        body_set=lab,
        evaluator=ev,
        dec=4,
        post_hoc=PH_PLAN,
        definition="Range over the four arms of the three pools of the seed-half reference (median over complementary seed splits of the rho between split means).",
    )


# ----------------------------------------------------------------- expected values (earlier summaries)
# (key, expected, decimals, source). expected is a number, [pt, lo, hi] for a value with an
# interval, a list for list-valued keys, a string for text keys, or ("<=", x) / (">=", x) / ("<", x).
FS = "results summary"
BP = "draft text"
EXPECTED = [
    # ranking, Table 2
    ("rank.rho3.rf_str.all100", [0.21, -0.02, 0.40], 2, FS),
    ("rank.rho3.rf_str.g72", [0.26, 0.00, 0.49], 2, FS),
    ("rank.rho3.rf_str.no_vt10", [0.27, 0.04, 0.48], 2, FS),
    ("rank.rho3.rf_str.s77", [0.45, 0.22, 0.63], 2, FS),
    ("rank.rho3.rf_str.s89", 0.25, 2, BP),
    ("rank.neutral.rho3.rf_str.all100", [0.31, 0.12, 0.49], 2, FS),
    ("rank.rho3.rf_len.all100", [0.72, 0.58, 0.83], 2, BP),
    ("rank.rho3.len_str.all100", [0.46, 0.26, 0.62], 2, BP),
    ("rank.rho3.rf_len.g72", 0.68, 2, BP),
    ("rank.rho3.len_str.g72", 0.47, 2, BP),
    ("rank.W.rf.all100", 0.82, 2, BP),
    ("rank.W.len.all100", 0.74, 2, BP),
    ("rank.W.str.all100", 0.65, 2, BP),
    ("rank.D.rf_str.all100", [0.53, 0.37, 0.70], 2, FS),
    ("rank.D.rf_len.all100", [0.17, 0.08, 0.27], 2, FS),
    ("rank.D.rf_len.g72", [0.20, 0.09, 0.33], 2, FS),
    ("rank.D.len_str.all100", [0.29, 0.17, 0.43], 2, BP),
    ("rank.D.len_str.g72", [0.30, 0.15, 0.49], 2, BP),
    ("rank.D.rf_str.g72", [0.52, 0.32, 0.73], 2, BP),
    ("rank.D.rf_str.s89", 0.42, 2, BP),
    ("rank.D.rf_str.s77", 0.38, 2, BP),
    ("rank.D.rf_len.s89", 0.16, 2, BP),
    ("rank.D.rf_len.s77", 0.20, 2, BP),
    ("rank.D.rf_str.subset_range", [0.38, 0.52], 2, FS),
    ("rank.top10.rf_len.all100", 8, 0, BP),
    ("rank.top10.len_str.all100", 4, 0, BP),
    ("rank.top10.rf_str.all100", 5, 0, FS),
    ("rank.single.within.len.42_43", 0.70, 2, FS),
    ("rank.single.within.len.42_44", 0.79, 2, FS),
    ("rank.single.within.len.43_44", 0.74, 2, FS),
    ("rank.single.within.str.42_43", 0.56, 2, FS),
    ("rank.single.within.str.42_44", 0.69, 2, FS),
    ("rank.single.within.str.43_44", 0.70, 2, FS),
    ("rank.single.within.rf.42_43", 0.85, 2, FS),
    ("rank.single.within.rf.42_44", 0.76, 2, FS),
    ("rank.single.within.rf.43_44", 0.85, 2, FS),
    ("rank.single.within.range", [0.56, 0.85], 2, FS),
    ("rank.single.cross.rf_str.range", [0.09, 0.29], 2, FS),
    ("rank.single.cross.rf_len.range", [0.43, 0.71], 2, BP),
    ("rank.single.cross.len_str.range", [0.34, 0.49], 2, BP),
    ("rank.single.within_top10.range", [3, 8], 0, FS),
    ("rank.single.within_top10_mean.len", 4.3, 1, FS),
    ("rank.single.within_top10_mean.str", 4.3, 1, FS),
    ("rank.single.within_top10_mean.rf", 5.0, 1, FS),
    ("rank.exch.rf_str.all100.null_rho", [0.64, 0.55, 0.73], 2, FS),
    ("rank.exch.rf_str.all100.p_rho", ("<=", 1e-4 + 1e-12), 4, FS),
    ("rank.exch.rf_str.all100.frac_top10_le_obs", 0.25, 2, FS),
    ("rank.exch.rf_str.all100.null_top20", 13.7, 1, BP),
    ("rank.exch.rf_str.all100.p_top20", 0.001, 3, BP),
    ("rank.exch.rf_str.all100.obs_top20", 10, 0, BP),
    ("rank.R3.rf.all100", 0.93, 2, BP),
    ("rank.R3.len.all100", 0.90, 2, BP),
    ("rank.R3.str.all100", 0.85, 2, BP),
    ("rank.noise_ref.rf_str.all100", [0.89, 0.84, 0.92], 2, FS),
    ("rank.noise_gap.rf_str.all100", [0.68, 0.48, 0.89], 2, "placebo statistics check"),
    ("rank.within_family.rf_str.floor.all100", 0.02, 2, FS),
    ("rank.within_family.rf_str.vt.all100", 0.13, 2, FS),
    ("rank.within_family.rf_str.mvt.all100", 0.55, 2, FS),
    ("rank.family_centred.rf_str.all100", 0.215, 3, FS),
    ("rank.top10_leavers.rf_not_str.n_cut23", 2, 0, BP),
    ("rank.sim_rerun.top10_mean_range", [6.5, 7.3], 1, BP),
    ("rank.sim_rerun.P_le_obs_range", [0.03, 0.16], 2, BP),
    ("rank.fig10.floor.str_gt_rf", 24, 0, BP),
    ("rank.fig10.vt.str_lt_rf", 23, 0, BP),
    ("cut.vt10_in_lowest12_strict", 10, 0, BP),
    # sets and geometry
    ("sets.inert28.n", 28, 0, FS),
    ("sets.inert28.floor", 8, 0, FS),
    ("sets.inert28.vt", 13, 0, FS),
    ("sets.inert28.mvt", 7, 0, FS),
    ("geom.inert20.z0_030", 25, 0, FS),
    ("geom.inert20.z0_045", 3, 0, FS),
    ("sets.g72.floor", 33, 0, FS),
    ("sets.g72.vt", 24, 0, FS),
    ("sets.g72.mvt", 15, 0, FS),
    ("sets.inert10_94.n", 94, 0, FS),
    ("geom.inert35.n", 0, 0, FS),
    ("geom.inert50.n", 0, 0, FS),
    ("geom.tau035.min_margin_m", 0.005, 3, FS),
    ("sets.s89.n", 89, 0, FS),
    ("sets.s77.n", 77, 0, FS),
    ("sets.s68.n", 68, 0, BP),
    ("sets.s65.n", 65, 0, BP),
    ("sets.no_vt10.n", 90, 0, FS),
    ("cut.cut11_inert", 7, 0, FS),
    ("cut.cut23_inert", 16, 0, FS),
    ("cut.vt10.z0_030", 7, 0, FS),
    ("cut.vt10.taller", 3, 0, FS),
    ("cut.vt10.max_strict_eval_len_any_rule", ("<=", 26), 1, FS),
    ("cut.vt10.max_strict_eval_len_rf_len", ("<=", 20), 1, FS),
    ("cut.runs.floor.n_cut", 17, 0, BP),
    ("cut.runs.vt.n_cut", 31, 0, BP),
    ("cut.runs.mvt.n_cut", 1, 0, BP),
    ("cut.runs.floor.ret_noncut", 0.92, 2, BP),
    ("cut.runs.vt.ret_noncut", 0.92, 2, BP),
    ("cut.runs.mvt.ret_noncut", 0.93, 2, BP),
    ("cut.z030_surviving_strict_seed.floor", 8, 0, BP),
    ("cut.z030_surviving_strict_seed.mvt", 4, 0, BP),
    ("cut.z030_surviving_strict_seed.vt", 5, 0, BP),
    ("cut.z030_surviving_strict_seed_50.vt", 5, 0, "draft text, 50-step variant"),
    ("cut.z030_surviving_strict_seed_50.floor", 8, 0, "draft text, 50-step variant"),
    ("cut.z030_surviving_strict_seed_50.mvt", 4, 0, "draft text, 50-step variant"),
    ("inert28.identical_records", 420, 0, FS),
    ("inert28.rf_len.mean", 0.98, 2, FS),
    ("inert28.rf_len.range", [0.87, 1.20], 2, FS),
    ("inert28.z030.min_probe_margin_m", 0.008, 3, BP),
    ("inert28.z045.episodes", 900, 0, BP),
    ("inert28.practically_inert_mvt", 5, 0, BP),
    ("inert28.factorial_runs_zero_dones", 29, 0, BP),
    ("inert28.factorial_runs", 30, 0, BP),
    ("geom.lenient_penetration_m", 0.04, 2, BP),
    ("zero.below_strict_1s", 45, 0, BP),
    ("zero.below_strict_3s", 53, 0, FS),
    ("physics.below_strict_1p5s", 46, 0, BP),
    ("zero.z030.cross_steps", [9], 0, FS),
    ("zero.z030.contact_steps", [11], 0, FS),
    ("zero.vt10_taller.cross_steps", [15], 0, BP),
    ("zero.vt10_taller.contact_steps", [10, 11], 0, BP),
    ("zero.lowest_point_at_reset_m", [0.15, 0.25], 2, BP),
    ("probe.shoot.floor-5506-10-6-01-15-48-35.reach50", 75, 0, BP),
    ("probe.shoot.vt-5506-3-17-02-18-33-45.reach50", 0, 0, BP),
    ("probe.shoot.vt-1409-6-3-02-22-09-55.reach50", 0, 0, BP),
    ("zero.vt10_below_derl_cutoff_3s.n", 2, 0, BP),
    ("zero.vt10_below_derl_cutoff_3s.z0", [0.65], 2, BP),
    ("geom.derl_fraction_range", [0.17, 0.40], 2, FS),
    ("geom.derl_closer_to_lenient", 66, 0, FS),
    ("geom.derl_below_radius", 25, 0, FS),
    ("geom.z0_range", [0.30, 1.04], 2, BP),
    ("geom.lenient_cutoff_range_m", [0.06, 0.21], 2, BP),
    ("geom.strict_cutoff_range_m", [0.15, 0.52], 2, BP),
    ("bodies.actuators_range", [5, 15], 0, BP),
    ("bodies.obs_dim_range", [21, 41], 0, BP),
    ("bodies.n_bodies_range", [5, 11], 0, BP),
    ("bodies.tree_depth_range", [1, 4], 0, BP),
    ("dup.distinct_files", 97, 0, FS),
    ("dup.max_abs_change_rho", ("<=", 0.04), 2, BP),
    ("dup.max_abs_change_D", ("<=", 0.03), 2, BP),
    ("dup.drop_choices", 8, 0, BP),
    # family
    ("fam.ret.rf_len.floor.all100", 0.55, 2, FS),
    ("fam.ret.str_len.floor.all100", 0.78, 2, FS),
    ("fam.ret.rf_len.vt.all100", 0.84, 2, FS),
    ("fam.ret.str_len.vt.all100", 0.66, 2, FS),
    ("fam.ret.rf_len.mvt.all100", 0.72, 2, FS),
    ("fam.ret.str_len.mvt.all100", 0.91, 2, FS),
    ("fam.mean.len.floor.all100", 2.428, 3, FS),
    ("fam.mean.str.floor.all100", 1.837, 3, FS),
    ("fam.mean.rf.floor.all100", 1.453, 3, FS),
    ("fam.mean.len.vt.all100", 2.304, 3, FS),
    ("fam.mean.str.vt.all100", 1.467, 3, FS),
    ("fam.mean.rf.vt.all100", 1.999, 3, FS),
    ("fam.mean.len.mvt.all100", 2.651, 3, FS),
    ("fam.mean.str.mvt.all100", 2.317, 3, FS),
    ("fam.mean.rf.mvt.all100", 2.088, 3, FS),
    ("fam.ret.rf_len.floor.g72", 0.44, 2, FS),
    ("fam.ret.str_len.floor.g72", 0.84, 2, FS),
    ("fam.ret.rf_len.vt.g72", 0.76, 2, FS),
    ("fam.ret.str_len.vt.g72", 0.83, 2, FS),
    ("fam.ret.rf_len.mvt.g72", 0.60, 2, FS),
    ("fam.ret.str_len.mvt.g72", 0.99, 2, FS),
    ("fam.mean.len.floor.g72", 2.343, 3, FS),
    ("fam.mean.str.floor.g72", 1.941, 3, FS),
    ("fam.mean.rf.floor.g72", 1.146, 3, FS),
    ("fam.mean.len.vt.g72", 2.434, 3, FS),
    ("fam.mean.str.vt.g72", 1.866, 3, FS),
    ("fam.mean.rf.vt.g72", 1.980, 3, FS),
    ("fam.mean.len.mvt.g72", 2.271, 3, FS),
    ("fam.mean.str.mvt.g72", 2.197, 3, FS),
    ("fam.mean.rf.mvt.g72", 1.492, 3, FS),
    ("fam.ret.rf_len.floor.s89", 0.56, 2, BP),
    ("fam.ret.rf_len.vt.s89", 0.81, 2, BP),
    ("fam.ret.rf_len.mvt.s89", 0.72, 2, BP),
    ("fam.ret.rf_len.floor.s77", 0.47, 2, BP),
    ("fam.ret.rf_len.vt.s77", 0.80, 2, BP),
    ("fam.ret.rf_len.mvt.s77", 0.71, 2, BP),
    ("fam.loss_pct.rf.floor.all100", 45, 0, BP),
    ("fam.loss_pct.rf.vt.all100", 16, 0, BP),
    ("fam.loss_pct.rf.floor.g72", 56, 0, BP),
    ("fam.loss_pct.rf.vt.g72", 24, 0, BP),
    ("fam.loss_pct.rf.mvt.g72", 40, 0, BP),
    ("fam.loss_pct.str.floor.all100", 22, 0, FS),
    ("fam.loss_pct.str.vt.all100", 34, 0, FS),
    ("fam.loss_pct.str.mvt.all100", 9, 0, BP),
    ("fam.loss_pct.str.floor.g72", 16, 0, FS),
    ("fam.loss_pct.str.vt.g72", 17, 0, FS),
    ("fam.loss_pct.str.mvt.g72", 1, 0, BP),
    ("fam.loss_pct.str.floor.s89", 19, 0, BP),
    ("fam.loss_pct.str.vt.s89", 10, 0, BP),
    ("fam.loss_pct.str.mvt.s89", 9, 0, BP),
    ("fam.loss_pct.str.floor.s77", 8, 0, BP),
    ("fam.loss_pct.str.vt.s77", 4, 0, BP),
    ("fam.loss_pct.str.mvt.s77", 6, 0, BP),
    ("fam.loss_pct.str.vt.no_vt10", 10, 0, FS),
    ("fam.worst_case.best_family.all100", "MVT", None, BP),
    ("fam.worst_case.mvt.all100", 0.72, 2, BP),
    ("fam.worst_case.best_family.g72", "VT", None, FS),
    ("fam.worst_case.vt.g72", 0.76, 2, FS),
    ("fam.worst_case.mvt.g72", 0.60, 2, FS),
    ("fam.worst_case.best_family.s89", "VT", None, BP),
    ("fam.worst_case.vt.s89", 0.81, 2, BP),
    ("fam.worst_case.best_family.s77", "VT", None, BP),
    ("fam.worst_case.vt.s77", 0.80, 2, BP),
    ("fam.contrast.rf_len.floor_minus_vt.all100", -0.30, 2, FS),
    ("fam.contrast.rf_len.floor_minus_vt.all100.p", 0.0006, 4, FS),
    ("fam.contrast.rf_len.floor_minus_vt.g72", -0.32, 2, FS),
    ("fam.contrast.rf_len.floor_minus_vt.g72.p", 0.002, 3, FS),
    ("fam.contrast.rf_len.floor_minus_vt.s89", -0.25, 2, BP),
    ("fam.contrast.rf_len.floor_minus_vt.s89.p", 0.01, 2, BP),
    ("fam.contrast.rf_len.floor_minus_vt.s77", -0.33, 2, BP),
    ("fam.contrast.rf_len.floor_minus_vt.s77.p", 0.003, 3, BP),
    ("fam.contrast.rf_len.floor_minus_vt.max_p", ("<=", 0.011), 3, BP),
    ("fam.contrast.str_len.vt_minus_floor.all100.p", 0.25, 2, FS),
    ("fam.contrast.diff.floor_minus_vt.all100", -0.42, 2, BP),
    ("fam.contrast.diff.floor_minus_vt.all100.p", 0.006, 3, BP),
    ("fam.contrast.diff.floor_minus_vt.g72", -0.34, 2, BP),
    ("fam.contrast.diff.floor_minus_vt.g72.p", 0.043, 3, BP),
    ("fam.within.vt.rf_len_minus_str_len.all100", 0.18, 2, "ranking check"),
    ("fam.within.vt.rf_len_minus_str_len.all100.p", 0.09, 2, BP),
    ("fam.within.floor.rf_len_minus_str_len.all100", -0.23, 2, "ranking check"),
    ("fam.within.floor.rf_len_minus_str_len.all100.p", 0.02, 2, "ranking check"),
    ("fam.neutral.rf_len.floor.all100", 0.643, 3, FS),
    ("fam.neutral.rf_len.vt.all100", 0.953, 3, FS),
    ("fam.neutral.rf_len.mvt.all100", 0.847, 3, FS),
    ("fam.neutral.rf_len.floor.g72", 0.561, 3, FS),
    ("fam.neutral.rf_len.vt.g72", 0.934, 3, FS),
    ("fam.neutral.rf_len.mvt.g72", 0.786, 3, FS),
    ("fam.neutral.str_len.floor.g72", 0.83, 2, BP),
    ("fam.neutral.str_len.vt.g72", 0.85, 2, BP),
    ("fam.fixed_horizon.rf_len.floor.all100", 0.47, 2, BP),
    ("fam.fixed_horizon.rf_len.vt.all100", 0.85, 2, BP),
    ("fam.fixed_horizon.rf_len.mvt.all100", 0.68, 2, BP),
    ("fam.fixed_horizon.rf_len.floor.g72", 0.35, 2, BP),
    ("fam.fixed_horizon.rf_len.vt.g72", 0.77, 2, BP),
    ("fam.fixed_horizon.rf_len.mvt.g72", 0.54, 2, BP),
    ("fam.fixed_horizon.rf_len.oneway.g72.F", 7.13, 2, BP),
    ("fam.fixed_horizon.rf_len.oneway.g72.p", 0.0015, 4, BP),
    ("fam.fixed_horizon.str_len.floor.g72", 0.87, 2, BP),
    ("fam.fixed_horizon.str_len.vt.g72", 0.88, 2, BP),
    ("fam.per_seed.44.g72.p", 0.076, 3, BP),
    ("fam.per_seed.floor_below_vt_all_seeds.all100", True, None, FS),
    ("fam.per_seed.floor_below_vt_all_seeds.g72", True, None, FS),
    # interaction
    ("inter.three.all100.perm_p", 0.005, 3, FS),
    ("inter.three.all100.cr1_F", 4.79, 2, FS),
    ("inter.three.all100.cr1_p", 0.001, 3, FS),
    ("inter.three.all100.perseed_F", 12.54, 2, FS),
    ("inter.three.all100.perseed_df", [4, 794], 0, FS),
    ("inter.three.all100.cr1_df", [4, 99], 0, FS),
    ("inter.three.g72.perm_p", 0.01, 2, FS),
    ("inter.three.g72.cr1_p", 0.002, 3, FS),
    ("inter.three.s89.perm_p", 0.14, 2, FS),
    ("inter.three.s89.cr1_p", 0.007, 3, FS),
    ("inter.three.s77.perm_p", 0.008, 3, BP),
    ("inter.three.all100.inflation", 2.6, 1, BP),
    ("inter.len_rf.max_perm_p", ("<=", 0.008), 3, FS),
    ("inter.len_str.all100.perm_p", 0.13, 2, FS),
    ("inter.len_str.min_perm_p", (">=", 0.13), 2, BP),
    ("inter.len_rf.all100.perm_p", ("<", 0.001), 4, BP),
    ("inter.len_rf.g72.perm_p", 0.004, 3, BP),
    ("inter.len_rf.s89.perm_p", 0.007, 3, BP),
    ("inter.len_rf.s77.perm_p", ("<", 0.001), 4, BP),
    ("inter.str_rf.all100.perseed_F", 17.0, 1, FS),
    ("inter.str_rf.g72.perseed_F", 14.9, 1, FS),
    ("inter.str_rf.all100.perm_p", 0.007, 3, BP),
    ("inter.str_rf.g72.perm_p", 0.019, 3, BP),
    ("inter.lofo.floor_vt.all100.df", [2, 620], 0, FS),
    ("inter.lofo.vt_mvt.all100.df", [2, 468], 0, FS),
    ("inter.lofo.floor_vt.g72.df", [2, 452], 0, BP),
    ("inter.lofo.vt_mvt.g72.df", [2, 308], 0, BP),
    ("inter.lofo.floor_vt.all100.perm_p", 0.004, 3, BP),
    ("inter.lofo.vt_mvt.all100.perm_p", 0.047, 3, BP),
    ("inter.lofo.vt_mvt.g72.perm_p", 0.07, 2, BP),
    ("inter.mixed.seedmean.vt_x_rf.all100", [0.67, 0.22, 1.13], 2, BP),
    ("inter.mixed.perseed.vt_x_rf.all100", [0.67, 0.39, 0.95], 2, "reviewed version"),
    ("inter.mixed.seedmean.rf_main_floor.all100", -0.975, 3, BP),
    ("inter.rf_minus_len.average.all100", -0.64, 2, BP),
    ("inter.ms_ratio_seedmean_perseed.all100", 0.873, 3, FS),
    # selection
    ("sel.in.len_only.str.k5.all100", 49.0, 1, FS),
    ("sel.in.len_only.str.k5.g72", 75.7, 1, FS),
    ("sel.in.len_only.str.k5.s89", 72.7, 1, BP),
    ("sel.in.len_only.str.k5.s77", 95.6, 1, FS),
    ("sel.in.len_only.str.k10.g72", 82.8, 1, BP),
    ("sel.in.len_only.str.k20.g72", 78.2, 1, BP),
    ("sel.neutral.len_only.str.k5.all100", 49.3, 1, BP),
    ("sel.neutral.len_only.str.k5.g72", 65.6, 1, BP),
    ("sel.in.min3.worst.k5.all100", 92.3, 1, BP),
    ("sel.in.min3.worst.k10.all100", 92.8, 1, BP),
    ("sel.in.min3.worst.k20.all100", 91.4, 1, BP),
    ("sel.in.min3.worst.k5.g72", 93.4, 1, BP),
    ("sel.in.min3.worst.k10.g72", 90.6, 1, FS),
    ("sel.in.min3.worst.k20.g72", 91.4, 1, BP),
    ("sel.in.mean3.worst.k5.g72", 94.1, 1, BP),
    ("sel.in.mean3.worst.k10.g72", 93.1, 1, BP),
    ("sel.in.mean3.worst.k20.g72", 93.9, 1, BP),
    ("sel.in_ci.min3.lowest_lower_bound.g72", 84.4, 1, BP),
    ("sel.in_ci.mean3.lowest_lower_bound.g72", 80.1, 1, BP),
    ("sel.heldout.min.k5.all100", 82.8, 1, FS),
    ("sel.heldout.min.k10.all100", 84.3, 1, BP),
    ("sel.heldout.min.k20.all100", 85.2, 1, BP),
    ("sel.heldout.diff.min_minus_str.k20.all100", [12.5, 3.4, 25.5], 1, BP),
    ("sel.top10_family.min3.mvt.all100", 8, 0, BP),
    ("sel.top10_inert_mvt.min3.all100", 5, 0, BP),
    ("sel.top10_family.min3.floor.g72", 3, 0, BP),
    ("sel.top10_family.min3.vt.g72", 3, 0, BP),
    ("sel.top10_family.min3.mvt.g72", 4, 0, BP),
    ("sel.len_top5.n_inert", 4, 0, BP),
    ("sel.len_top5.lowest_strict_body", "vt-1409-12-10-02-21-04-52", None, BP),
    ("lopo.gate.mean_gain_k5", [0.04, -0.42, 0.23], 2, FS),
    ("lopo.str_held.RobustMin.k5", 74.3, 1, FS),
    ("lopo.str_held.best_robust_k10", 81.9, 1, FS),
    # repeat
    ("repeat.probe_safe60.rho_tau0_tau010", [0.92, 0.85, 0.96], 2, BP),
    ("repeat.probe_safe60.rho_rf_str", [0.44, 0.19, 0.65], 2, BP),
    ("repeat.probe_safe60.top10_repeat", 6, 0, BP),
    ("repeat.probe_safe60.independent_seed_estimate", 0.89, 2, BP),
    ("repeat.probe_safe60.strict_rf_noise_ref", 0.88, 2, BP),
    ("repeat.inert10_94.rho_tau0_tau010", [0.95, 0.92, 0.97], 2, FS),
    ("repeat.inert10_94.rho_rf_str", [0.27, 0.05, 0.46], 2, FS),
    ("repeat.inert10_94.top10_repeat", 6, 0, FS),
    ("repeat.probe_safe67.rho_tau0_tau010", 0.94, 2, "placebo statistics check"),
    ("sets.probe_safe60.n", 60, 0, BP),
    ("sets.probe_reach.union", 34, 0, "placebo mechanism check"),
    ("sweep.tau010_minus_tau0.mean", 0.02, 2, BP),
    ("sweep.tau010_minus_tau0.p", 0.59, 2, BP),
    ("sweep.tau035.vt10_short.n_near_zero", 5, 0, BP),
    ("sweep.tau035.vt_ratio.all", 0.87, 2, BP),
    ("sweep.tau035.vt_ratio.no_vt10", 1.03, 2, BP),
    # exemplars
    ("ex.floor_c.rank.rf", 88, 0, FS),
    ("ex.floor_c.rank.len", 17, 0, FS),
    ("ex.floor_c.rank.str", 13, 0, FS),
    ("ex.floor_b.rank.rf", 4, 0, FS),
    ("ex.floor_b.rank.len", 6, 0, FS),
    ("ex.floor_b.rank.str", 1, 0, FS),
    ("ex.floor_c.dist.rf", [-0.1, 1.8, 1.9], 1, BP),
    ("ex.floor_c.dist.len", [54.7, 70.8, 59.6], 1, BP),
    ("ex.floor_c.dist.str", [31.3, 50.1, 65.9], 1, BP),
    ("ex.floor_b.dist_range", [63, 87], 0, BP),
    ("ex.rf_failures.n", 3, 0, BP),
    ("ex.rf_failures.best_by_len_rank", "floor-1409-11-14-01-13-30-11", None, BP),
    ("ex.floor_c.z0", 0.75, 2, BP),
    ("ex.floor_b.z0", 0.65, 2, BP),
    ("ex.floor_c.later_code_rf_dist", [1.0, 0.8, 1.1], 1, BP),
    ("bridge.rf_drift_median.floor", -0.42, 2, BP),
    ("inert28.factorial_strict_median_steps_per_done", 9, 0, "draft text (about every 9 steps)"),
    # distance
    ("dist.common.rf_stop.floor.all100", 12, 0, BP),
    ("dist.common.rf_stop.vt.all100", 4, 0, BP),
    ("dist.common.rf_stop.mvt.all100", 5, 0, BP),
    ("dist.common.rf_stop.floor.g72", 12, 0, BP),
    ("dist.common.rf_stop.vt.g72", 4, 0, BP),
    ("dist.common.rf_stop.mvt.g72", 5, 0, BP),
    ("dist.common.str_stop.floor.all100", 1, 0, BP),
    ("dist.common.str_stop.vt.all100", 10, 0, BP),
    ("dist.common.str_stop.mvt.all100", 0, 0, BP),
    ("dist.common.vt_str_stop.n_inert", 7, 0, BP),
    ("dist.common.str_stop.vt.g72", 3, 0, BP),
    ("dist.common.vt_str_stop.n_cut11", 10, 0, BP),
    ("dist.common.vt_str_stop.n_cut23", 10, 0, "draft text, read as the 23"),
    ("dist.floor_rf_stops.neutral_ge5", 5, 0, BP),
    ("dist.floor_c.rf_neutral_mean", 1.2, 1, BP),
    ("dist.floor.neutral_dist.rf.all100", 31.2, 1, BP),
    ("dist.floor.neutral_dist.len.all100", 44.5, 1, BP),
    ("dist.floor.neutral_dist.rf.g72", 25.5, 1, BP),
    ("dist.floor.neutral_dist.len.g72", 41.7, 1, BP),
    ("dist.floor.rf_fall_rate.all100", 0.61, 2, BP),
    ("dist.floor.rf_fall_rate.g72", 0.76, 2, BP),
    ("dist.floor.ep_len.rf.all100", 512, 0, BP),
    ("dist.floor.ep_len.len.all100", 828, 0, BP),
    ("dist.floor.ep_len.rf.g72", 393, 0, BP),
    ("dist.floor.ep_len.len.g72", 786, 0, BP),
    ("dist.action_sq_per_step.range.neutral", [8.8, 9.8], 1, BP),
    # structure
    ("struct.spearman.init_height.rf_len.all100", -0.53, 2, BP),
    ("struct.spearman.init_height.rf_len.g72", -0.39, 2, BP),
    ("struct.spearman.init_height.str_len.all100", 0.38, 2, FS),
    ("struct.spearman.init_height.str_len.g72", 0.03, 2, FS),
    ("struct.two_feature.described.R2", 0.415, 3, FS),
    ("struct.two_feature.described.resid_means", [0.177, -0.342, 0.244], 3, FS),
    ("struct.two_feature.described.F", 3.32, 2, FS),
    ("struct.two_feature.described.p", 0.04, 2, BP),
    ("struct.dstrrf.oneway_F", 5.12, 2, BP),
    ("struct.n_bodies_vs_n_joints_r", 0.84, 2, BP),
    ("struct.M2.R2.no_vt10", 0.43, 2, BP),
    ("struct.M1.R2.no_vt10", 0.18, 2, BP),
    # panels
    ("sac.cross_rho", [0.75, 0.40, 0.94], 2, BP),
    ("sac.cross_rho_vt", 0.89, 2, BP),
    ("sac.sac.str_rf.floor", 2.41, 2, BP),
    ("sac.sac.str_rf.vt", 1.52, 2, BP),
    ("sac.sac.str_rf.mvt", 0.96, 2, BP),
    ("sac.ppo24.str_rf.floor", 2.18, 2, BP),
    ("sac.ppo24.str_rf.vt", 1.27, 2, BP),
    ("sac.ppo24.str_rf.mvt", 0.91, 2, BP),
    ("sac.sac.str_rf.floor_joint_valid", 2.65, 2, BP),
    ("sac.vt_same_sign", 12, 0, BP),
    ("sac.sac.perrun_p", 4.6e-5, 6, FS),
    ("sac.sac.cr1_p", 0.016, 3, FS),
    ("sac.ppo24.cr1_p", 0.018, 3, BP),
    ("sac.n_inert", 8, 0, FS),
    ("sac.sac.perrun_p_noninert", 7.5e-4, 5, BP),
    ("bf.rho", [0.85, 0.67, 0.93], 2, FS),
    ("bf.median_delta.floor", -0.54, 2, BP),
    ("bf.median_delta.mvt", -0.38, 2, BP),
    ("bridge.rho_matched.rf", 0.97, 2, FS),
    ("bridge.rho_matched.len", 0.88, 2, FS),
    ("bridge.rho_matched.str", 0.73, 2, FS),
    ("bridge.delta_reset_rho", 0.84, 2, FS),
    ("bridge.delta_reset_sign_agree", 10, 0, FS),
    ("bridge.criteria_failed", 8, 0, FS),
    ("compute.corpus.hours", 980, 0, FS),
    ("compute.corpus.hours_per_run", 1.09, 2, BP),
    ("compute.sac.hours", 105, 0, FS),
    ("compute.tau010_035.hours", 1006, 0, BP),
    ("compute.bridge.run_hours", 228, 0, FS),
    ("compute.bridge.gpu_hours", 57, 0, FS),
    ("compute.no_time_limit.hours", 180, 0, BP),
    ("compute.boundary_duration.hours", 357, 0, BP),
    ("compute.non_evolved.hours", 1025, 0, BP),
    ("compute.asymptotic.hours", 183, 0, BP),
    ("jobs.offset_p.job", 0.81, 2, BP),
    ("jobs.offset_p.array_task", 0.56, 2, BP),
]

NOT_COVERED = [
    (
        "Review-period panels (Humanoid fall rules and time limits, Ant reset initialization and breadth, Evolution Gym timeouts): the bootstrap intervals of rho, the top-10 overlaps, the counts of bodies with negative reward, D and the 33-comparison adjustment, and the panel settings (trainers, budgets and timeouts). Their run counts, pools, seeds, hardware, wall times, rank correlations and seed-half references are recomputed here (panels.*)",
        "Raw records are not in the results directory (cluster storage). Their per-cell scores, specifications and frozen reports are in panels.zip of the dataset, and scripts/review_panels.py recomputes every statistic in those reports, including the intervals with the reports' seeds and the top-10 overlaps. D and the 33-comparison adjustment come from a separate analysis script that is not released.",
    ),
    (
        "Structure appendix beyond in-sample fits: leave-one-out R^2 (M2 0.40, M1 0.07, height 0.37; on the 72 M1 0.08, M2 0.25, height 0.10), the leave-one-out Spearman rho of M2 (0.65) and its p-value, leave-one-family-out rho with intervals and held-out R^2 (M2 0.12, -0.25, 0.41; height 0.33, -0.06, 0.55; M2 on the 72 -0.17, -0.11, -0.60), the nine-fold crossfit (median rho 0.51 against 0.30; 8 of 9 folds on the 72) and the non-evolved transfer test (0.41 [0.18, 0.60] overall, 0.06 [-0.29, 0.38] within the evolved height range of 43 bodies)",
        "scripts/structural_feature_regression.py and the authors' independent verification scripts (not released).",
    ),
    (
        "Other planned tests in the register: predicting worst-case reward (R^2 = 0.06), the rule x time-limit factorial (F(2,505) = 3.72, p = 0.025; median shifts -0.38 [-0.67, -0.18], -0.02 [-0.17, 0.16] and +0.05 [-0.13, 0.13]; rank correlations 0.71 and 0.94), contact-based termination (per-run p = 1.4e-4, rho = 0.19 [-0.18, 0.50]), the boundary-duration sweep (rho 0.92 [0.78, 0.97] and 0.85 [0.68, 0.94]; slope 0.099 [0.053, 0.150] per tenfold limit), existing-robots panel (2 of 4 robots passed the quality check)",
        "Their recorded analyses: records/{nems_robust_selector_analysis,factorial_analysis,alt_termination_analysis,boundary_sweep_analysis}.json and records/menagerie_panel_report.md.",
    ),
    (
        "SAC scale check and body choice: the check run on 8 bodies (the plan specified 1) that failed on 4 (Standard to Reset-free ratios 2.75 to 12.84 against a planned 0.5 to 2), its replacement by agreement with PPO (6 of 7 within a factor of 2), the 120 runs of the second batch, and the earlier mechanism study's choice of the 24 bodies (8 that fell often, 8 that barely moved, 8 unaffected)",
        "records/sac_full_panel_launch_log.md and records/sac_full_panel_spec.md (scale check, replacement and batches); scripts/mechanism_morphologies.txt lists the 24 bodies without their categories. The pilot statistics and the seed-averaged and permutation p-values are recomputed (sac.pilot.*, sac.*.seedmean_*, sac.*.perm_p).",
    ),
    (
        "Fall rules in public code (DERL's ratio 0.5 and 0.2 m clearance with a limit bootstrapped at 1,000 steps; MetaMorph's 0.3; Transform2Act's 0.7 m, 20 and 60 degree limits and alive bonuses 1.0 and 0.1; BodyGen; Evolution Gym), the Gymnasium healthy ranges (Ant [0.2, 1.0] m below its 0.25 m torso radius, Hopper above 0.7 m, Walker2d [0.8, 2.0] m, Humanoid [1.0, 2.0] m, alive bonus 1.0 and 5.0) and their fractions of the initial torso height (0.27, 0.56, 0.64, 0.71), DERL budget facts, compute hardware of the UNIMAL panels, the source of the 100 bodies (MetaMorph's training set, chosen from the 300 bodies that DERL evolved, 100 in each of three environments) and the page of DERL's statement on early termination",
        "Source code and launch records, not run results: records/termination_rules_public_code.md (DERL, MetaMorph, Transform2Act, BodyGen, Evolution Gym) and the Gymnasium 1.0.0 MuJoCo environment files (healthy_z_range and initial torso heights of the v4 and v5 environments, not copied into the release). The body source is stated in MetaMorph (Gupta et al., ICLR 2022, Appendix A.2) and DERL (Gupta et al., Nature Communications 12, 5721, 2021, Methods).",
    ),
    (
        "Training, evaluation and probe settings: the PPO preset (5M steps as 2,441 updates of 2,048 steps, 128 environments, 16 steps per rollout, minibatch 256, 10 epochs, learning rate 3e-4, discount 0.99, GAE 0.95, clip 0.2, value and entropy coefficients 0.5 and 0.01, gradient norm 0.5, two layers of 64 units, physics step 0.005 s, frame skip 4, control step 0.02 s), the reward and reset constants (action cost 0.001, reset noise 0.01, upper height bound max(3 z0, 2 m)), 20 evaluation episodes, the SAC settings (Brax 0.14.2, 128 environments, batch 256, learning rate 3e-4, discount 0.99, reward scaling 1, replay buffer 100,000 with a minimum of 1,280, two layers of 256 units, target rate 0.005), the probe designs (drops from 1 m, 20-episode probes, 300 random sequences per body in the smaller probe) and the quantities that follow from them (about 39,000 steps per environment; about 13 and 3 episodes per environment at limits of 3,001 and 10,001 steps), and the Stable-Baselines3 PPO defaults that the preset changes (minibatch 64, entropy coefficient 0)",
        "Launch code and trainers, not run results: corpus_revision/ (the corpus-era trainer and launchers), scripts/train_mjx.py, scripts/train_sac_mjx.py and the probe scripts in records/sim-record-scripts; the 300-sequence probe (sim-records/sk_shoot.json) was made by an independent verification script (not released). The Stable-Baselines3 defaults are those of its PPO class in version 2.3.2.",
    ),
    (
        "Launch history, dates and software versions: the launch, freeze, registration and commit dates (for example the corpus-era trainer revisions of 5 and 8 April 2026), the diagnosis of the later code in April 2026, runs per GPU, four array tasks of 25 bodies, the software of the training environment (MuJoCo and MJX 3.6.0, JAX 0.9.2) and the pinned versions of public code",
        "Git history of the trainers and launchers (corpus_revision/ keeps both corpus-era trainer revisions and the launchers; slurm/ holds the later launchers), and the dated planned-test records in records/.",
    ),
    (
        "Decision rules and designs of the planned tests: thresholds such as rho > 0.85, median drift below 0.10, slope in [0.9, 1.1], intercept below 0.05 and 90% sign agreement (retraining), rho > 0.3 (transfer), p < 0.025 and rho >= 0.70 (longer budget), p < 0.01 and rho >= 0.60 with contact rules of 1, 50 or unlimited steps (contact termination), 0.05 PSR in 2 of 3 splits and 85% (leave-one-rule-out), upper bounds below 0.95 and partial eta^2 > 0.02 with limits of 3,001 and 10,001 steps (time limits), R^2 >= 0.20 (feature-based selector), p < 0.01 (SAC and A2C)",
        "The frozen specifications in records/ (equivalence_pilot_report.md and the *_spec.md and *_prereg.md files). The 14 retraining sub-criteria and the SAC reliability threshold of 0.514 are recomputed (bridge.criteria_total, sac.sac.reliability_threshold).",
    ),
    (
        "The paused replication (60 non-evolved bodies, 360 runs, 309 finished, Holm-adjusted one-sided p < 0.025, about 220 GPU-hours) and the compute of the unfinished 50M-step phase of the longer-budget panel",
        "The replication's registration and the cluster job records; neither is part of the release.",
    ),
    (
        "Crossing steps with the training reset noise (step 9 for the seven short VT cut-off bodies, steps 13 to 16 for the three taller ones)",
        "An independent verification script and its output (not released); the zero-action records in sim-records/ use the nominal reset.",
    ),
    (
        "Statements about Monte Carlo variation between random streams: bootstrap endpoints move by about 0.01, and the upper end of the VT minus MVT interval on the 72 depends on the stream",
        "Comparisons of this script's seeded streams with those of the authors' independent verification scripts (not released) and with the earlier summaries (the comparison section above)."
    ),
    (
        "Definitions and analysis constants printed with the results: the rule and evaluator thresholds (tau = 0, 0.10, 0.20, 0.35, 0.50), the 1,000-step limit and the 5M-step budget, seeds 42, 43 and 44, k = 5, 10 and 20, B = 10,000 and 95% intervals, the 20-step cut-off-run threshold, the 500-step survival threshold, the 5 m and 1 m walking and stopping thresholds, the 0.05 PSR floor for ratios, the 0.1 PSR diagnostic threshold, the 50-step probe criterion and the gallery percentiles",
        "Inputs, not results: the run configurations and this script's constants and definitions (RULES, SWEEP, SEEDS, KS, BOOT_SEED, PERM_SEED and the thresholds in the row definitions).",
    ),
    ("A2C panel (24 of 216 cells, 48 calibration runs, p = 0.073)", "The analyzed runs are lost; no A2C statistic is recomputed (disclosed in the paper)."),
    (
        "Identifiers printed with the references and acknowledgments: grant numbers, DOIs, volume, issue and page numbers of the cited works, and page locators of quoted passages",
        "The grant agreements and the bibliographic records of the cited works (publisher pages and Crossref), not results.",
    ),
]


CMP_NOTES = {
    "cut.z030_surviving_strict_seed.vt": "Definition. With the cut-off-run threshold (a Strict seed averaging more than 20 steps under the Strict evaluator) 7 of the 13 VT bodies at z0 = 0.30 m have a surviving seed. The source's 5 of 13 holds only for a threshold of about 50 steps or more, because the best seeds of vt-1409-8-6 and vt-1409-9-14 last 25.8 and 48.8 steps (see the _50 variant). Floor (8 of 8) and MVT (4 of 4) do not depend on the threshold.",
    "bodies.tree_depth_range": "Convention. morph_features tree_depth counts body levels below the torso and ranges 2 to 5. The source's 1 to 4 needs a different convention. Check before printing.",
    "dist.common.vt_str_stop.n_cut11": "Nine of the 10 VT Strict stops are cut-off bodies. The tenth, vt-1409-9-14, is a cut-off-seed body, so all 10 are in the 23. The VT cut-off body vt-1409-3-14 is not a Strict stop (1.7 m after Strict training).",
    "ex.floor_b.dist_range": "Rounding. Floor B's nine runs cover 63.5 to 86.7 m, which rounds to 64 to 87 m.",
    "fam.neutral.str_len.floor.g72": "Rounding error in the source. 0.8245 rounds to 0.82 (the recorded subset statistics give the same 0.8245).",
    "struct.M1.R2.no_vt10": "Rounding error in the source. 0.1749 rounds to 0.17 (M1 is 0.20 without the 11 cut-off bodies).",
    "sac.sac.perrun_p_noninert": "Rounding. 7.45e-4 rounds to 7.4e-4 at two significant digits.",
    "inter.len_str.min_perm_p": "Wording. The smallest value, 0.127 on all 100, rounds to 0.13 but is below it. Write p = 0.13, not p >= 0.13.",
    "inter.len_rf.max_perm_p": "Monte Carlo. The largest value over the six sets is 0.0081 (the 65) with this script's stream. Write p < 0.01.",
    "inter.three.all100.perm_p": "Monte Carlo. 0.0045 here (prints as 0.004) and 0.0052 in the reviewed version's stream (seed 42, B = 5,000). Print this script's value, or p < 0.01.",
    "fam.within.vt.rf_len_minus_str_len.all100.p": "Monte Carlo. 0.097 here, which prints as 0.10 rather than 0.09.",
    "fam.contrast.rf_len.floor_minus_vt.s77.p": "Monte Carlo. 0.0012 here and 0.0021 in an independent verification's stream; every stream gives p below 0.005.",
    "rank.sim_rerun.top10_mean_range": "Monte Carlo. 6.56 to 7.30 here (prints as 6.6 to 7.3).",
    "rank.sim_rerun.P_le_obs_range": "Monte Carlo. 0.026 to 0.152 here (prints as 0.03 to 0.15).",
    "rank.exch.rf_str.all100.frac_top10_le_obs": "Monte Carlo. 24% here, 25% in an independent verification's stream.",
    "inert28.factorial_strict_median_steps_per_done": "Scope. 16 of the 30 Strict-trained factorial runs of inert bodies average at most 10 steps per training episode (shortest 9.0); most of the other 14 average hundreds of steps. 'About every 9 steps' describes those 16 runs, not every inert-body run.",
}


def classify(c, R):
    if c["status"] == "match":
        return ""
    if c["key"] in CMP_NOTES:
        return CMP_NOTES[c["key"]]
    if c["status"] == "interval differs":
        return "Monte Carlo. Same point value; an interval endpoint moves with the bootstrap resample stream (the source used another seed)."
    if c["key"] in R.index and R.get(c["key"])["kind"] == "p":
        return "Monte Carlo. Permutation p from a different random stream; the Monte Carlo standard error is about sqrt(p(1-p)/B)."
    return ""


def compare_expected(R: Registry):
    out = []
    for key, exp, dec, src in EXPECTED:
        if key not in R.index:
            out.append(dict(key=key, expected=exp, computed=None, status="not computed", source=src))
            continue
        row = R.get(key)
        val, ci = row["value"], row["ci"]
        status = "match"
        if isinstance(exp, tuple):
            op, thr = exp
            ok = {"<=": val <= thr, ">=": val >= thr, "<": val < thr}[op]
            status = "match" if ok else "DIFFERENT"
            comp = val
        elif isinstance(exp, bool) or isinstance(exp, str):
            comp = val
            status = "match" if val == exp else "DIFFERENT"
        elif isinstance(exp, list) and row["ci"] is not None and len(exp) == 3 and not isinstance(val, list):
            comp = [val, ci[0], ci[1]]
            got = [round(x + 0.0, dec) for x in comp]
            want = [round(x, dec) for x in exp]
            if got[0] != want[0]:
                status = "DIFFERENT"
            elif got[1:] != want[1:]:
                status = "interval differs"
        elif isinstance(exp, list):
            comp = val
            if not isinstance(val, list) or len(val) != len(exp):
                status = "DIFFERENT"
            else:
                status = "match" if [round(float(x), dec) for x in val] == [round(float(x), dec) for x in exp] else "DIFFERENT"
        else:
            comp = val
            if val is None:
                status = "DIFFERENT"
            elif dec == 0:
                status = "match" if round(float(val)) == round(float(exp)) else "DIFFERENT"
            else:
                status = "match" if abs(round(float(val), dec) - round(float(exp), dec)) < 0.5 * 10 ** (-dec) else "DIFFERENT"
        out.append(dict(key=key, expected=exp, computed=comp, status=status, source=src, dec=dec))
    for c in out:
        c["note"] = classify(c, R)
    return out


# ----------------------------------------------------------------- output
def write_outputs(R: Registry, comp, args, meta):
    js = dict(
        meta=meta,
        numbers={r["key"]: {k: r[k] for k in ("value", "ci", "body_set", "evaluator", "definition", "post_hoc", "note", "section")} for r in R.rows},
        comparison=[{k: _py(v) for k, v in c.items()} for c in comp],
        not_covered=[dict(item=a, where=b) for a, b in NOT_COVERED],
    )
    Path(args.out_json).write_text(json.dumps(_py(js), indent=1, ensure_ascii=False) + "\n")

    def esc(s):
        return str(s).replace("|", "/").replace("\n", " ")

    L = []
    L.append("# Camera-ready numbers")
    L.append("")
    L.append(
        f"Generated by `scripts/camera_ready_numbers.py` on {meta['generated']}. "
        f"Every value below was recomputed from the raw run records; full precision is in `camera_ready_numbers.json`."
    )
    L.append("")
    L.append("Inputs: " + "; ".join(f"`{k}` = `{v}`" for k, v in meta["inputs"].items()) + ".")
    L.append("")
    L.append(
        "The zero-action, probe and random-shooting records (`passive`, `settle`, `probe_dir`, `shoot`, `search`) were made "
        "on 2026-10-05 and ship in the dataset under `sim-records/`. "
        "The MuJoCo re-run (`physics.*` rows) already recomputes the zero-action counts and steps from the XMLs. "
        "The `logs.*` rows read the kept training logs (`logs.zip` of the dataset) and the `panels.*` rows the per-cell "
        "tables of the review-period panels (`panels.zip`). "
        f"Run time {meta['runtime_s']} s on CPU; package versions are in the JSON."
    )
    L.append("")
    L.append(
        "Conventions. Rules are Reset-free (tau = 0), Standard (tau = 0.20; keys write it len, after a draft name) and "
        "Strict (tau = 0.50), in ascending tau. Rankings use three-seed mean PSR (per-step reward) under the common evaluator "
        "(tau_eval = 0.20) unless the evaluator column says otherwise. Intervals are 95% percentile intervals from "
        f"B = {args.boot:,} body resamples drawn with numpy default_rng({BOOT_SEED}) (one resample matrix per body set, "
        "shared by every statistic on that set), conditional on the three seeds. Permutation tests use "
        f"B = {args.perm:,} draws from a fresh default_rng({PERM_SEED}) per test and p = (1 + #null at or beyond observed)/(B + 1). "
        'Retention is the mean over bodies of the per-body ratio of three-seed means. The post hoc column says "yes" for '
        'analyses added after review, "no (reviewed version)" for analyses reported in the reviewed version without a prior '
        'decision rule, and "no (planned)" for tests whose rule was fixed in advance. Body sets defined by outcomes '
        "(the 89, 77, 68, 65 and 90) are post hoc sensitivity sets."
    )
    L.append("")
    # comparison first
    L.append("## Comparison with the authors' earlier summaries")
    L.append("")
    L.append(
        f"{len(comp)} values from the results summary and the draft text that the paper was written from were checked: {sum(c['status'] == 'match' for c in comp)} match at the printed precision, "
        f"{sum(c['status'] == 'interval differs' for c in comp)} match in the point value but differ in an interval endpoint, "
        f"{sum(c['status'] == 'DIFFERENT' for c in comp)} differ, and {sum(c['status'] == 'not computed' for c in comp)} were not computed."
    )
    L.append("")
    L.append(
        "Expected values are shown at their printed precision and computed values with one more decimal. "
        "For a value with an interval both are given as [value, lower, upper]. Differences marked Monte Carlo come from "
        "a different random stream, not from different data or definitions: with B = 10,000 a permutation p near 0.005 "
        "has a Monte Carlo standard error of about 0.0007, and bootstrap interval endpoints move by about 0.01. "
        "The paper prints this script's values."
    )
    L.append("")
    L.append("| key | expected | computed | status | source | note |")
    L.append("|---|---|---|---|---|---|")
    for c in sorted(comp, key=lambda c: (c["status"] == "match", not c["note"].startswith("Monte Carlo"), c["key"])):
        dec = c.get("dec")
        dec = 3 if dec is None else max(dec, 0)
        exp = c["expected"]
        exp_s = " ".join(map(str, exp)) if isinstance(exp, tuple) else fmt_num(_py(exp), dec)
        comp_s = fmt_num(
            _py(c["computed"]),
            dec + 1 if isinstance(c["computed"], float) or (isinstance(c["computed"], list) and any(isinstance(x, float) for x in c["computed"])) else dec,
        )
        L.append(f"| `{c['key']}` | {esc(exp_s)} | {esc(comp_s)} | {c['status']} | {c['source']} | {esc(c['note'])} |")
    L.append("")
    # numbers by section
    by = collections.OrderedDict()
    for r in R.rows:
        by.setdefault(r["section"], []).append(r)
    for sec, rows in by.items():
        L.append(f"## {sec}")
        L.append("")
        L.append("| key | value | CI | body set | evaluator | definition | post hoc |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            val = fmt_num(r["value"], r["dec"], r["kind"])
            if r["note"]:
                val = f"{val} ({r['note']})"
            ci = fmt_num(r["ci"], r["dec"], r["kind"]) if r["ci"] is not None else ""
            L.append(f"| `{r['key']}` | {esc(val)} | {esc(ci)} | {esc(r['body_set'])} | {esc(r['evaluator'])} | {esc(r['definition'])} | {r['post_hoc']} |")
        L.append("")
    L.append("## Printed numbers this script does not recompute")
    L.append("")
    L.append("| numbers | where they come from |")
    L.append("|---|---|")
    for a, b in NOT_COVERED:
        L.append(f"| {esc(a)} | {esc(b)} |")
    L.append("")
    Path(args.out_md).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(DATA / "results"))
    ap.add_argument("--audit", default=str(REPO / "records" / "standard_threshold_audit.json"))
    ap.add_argument("--passive", default=str(SIM / "all100_passive.json"))
    ap.add_argument("--settle", default=str(SIM / "settle.json"))
    ap.add_argument("--probe-dir", default=str(SIM / "probes"))
    ap.add_argument("--shoot", default=str(SIM / "sk_shoot.json"))
    ap.add_argument("--search", default=str(SIM / "vt10_search.json"))
    ap.add_argument("--xml-dir", default=str(DATA / "assets" / "unimals_100" / "xml"))
    ap.add_argument("--gait-traces", default=str(DATA / "gait-traces"))
    ap.add_argument("--logs", default=None, help="logs.zip, or the folder it was unzipped into (default: the --results folder)")
    ap.add_argument("--review-panels", default=str(DATA / "review-panels"), help="panels.zip, or the review-panels/ folder it unzips into")
    ap.add_argument("--sac-bodies", default=str(REPO / "scripts" / "mechanism_morphologies.txt"))
    ap.add_argument("--bf-bodies", default=str(REPO / "records" / "boundary_free_bodies.json"))
    ap.add_argument("--bridge-bodies", default=str(REPO / "records" / "equivalence_pilot_bodies.json"))
    ap.add_argument("--out-json", default=str(REPO / "records" / "camera_ready_numbers.json"))
    ap.add_argument("--out-md", default=str(REPO / "records" / "camera_ready_numbers.md"))
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--perm", type=int, default=10_000)
    ap.add_argument("--no-physics", action="store_true", help="skip the MuJoCo re-run of the zero-action rollouts")
    args = ap.parse_args()
    t0 = time.time()
    results = Path(args.results).expanduser()
    D = Data(results, Path(args.audit))
    R = Registry()
    log("sets and counts")
    sets, extra = section_sets(D, R, args)
    extra_sets(D, R, args, sets, extra)
    log("duplicates")
    section_duplicates(D, R, args, sets)
    log("ranking")
    section_ranking(D, R, args, sets, extra)
    extra_ranking(D, R, args, sets, extra)
    log("family")
    section_family(D, R, args, sets, extra)
    extra_family(D, R, args, sets, extra)
    log("interaction")
    section_interaction(D, R, args, sets)
    extra_interaction(D, R, args, sets)
    log("selection")
    section_selection(D, R, args, sets, extra)
    extra_selection(D, R, args, sets, extra)
    log("repeat")
    section_repeat(D, R, args, sets)
    extra_repeat(D, R, args, sets, extra)
    log("sweep")
    section_sweep(D, R, args, sets, extra)
    extra_sweep(D, R, args, sets, extra)
    log("exemplars")
    section_exemplars(D, R, args, sets, extra)
    extra_exemplars(D, R, args, sets, extra)
    log("distance")
    section_distance(D, R, args, sets, extra)
    extra_distance(D, R, args, sets, extra)
    log("panels")
    section_panels(D, R, args, sets, extra)
    extra_panels(D, R, args, sets, extra)
    log("structure")
    section_structure(D, R, args, sets)
    extra_structure(D, R, args, sets)
    log("records: upper height bound, training logs, review-period panels")
    extra_records(D, R, args)
    comp = compare_expected(R)
    try:
        import mujoco  # noqa: F401  (version record only, the physics re-run is optional)
    except ImportError:
        pass
    meta = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        runtime_s=round(time.time() - t0, 1),
        inputs={k: _repo_relative(v) for k, v in dict(
            results=results,
            audit=args.audit,
            passive=args.passive,
            settle=args.settle,
            probe_dir=args.probe_dir,
            shoot=args.shoot,
            search=args.search,
            xml_dir=args.xml_dir,
            gait_traces=args.gait_traces,
            logs=args.logs if args.logs else results,
            review_panels=args.review_panels,
        ).items()},
        bootstrap=dict(B=args.boot, seed=BOOT_SEED),
        permutation=dict(B=args.perm, seed=PERM_SEED),
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            scipy=__import__("scipy").__version__,
            pandas=pd.__version__,
            statsmodels=__import__("statsmodels").__version__,
            mujoco=getattr(sys.modules.get("mujoco"), "__version__", "n/a"),
        ),
        n_numbers=len(R.rows),
    )
    write_outputs(R, comp, args, meta)
    bad = [c for c in comp if c["status"] != "match"]
    log(f"wrote {args.out_json} and {args.out_md}: {len(R.rows)} numbers, {len(comp)} checked, {len(bad)} not matching, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
