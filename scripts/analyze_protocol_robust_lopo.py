#!/usr/bin/env python3
"""Pre-registered LOPO (leave-one-protocol-out) analysis of protocol-robust
selection.

Implements the spec frozen at records/protocol_robust_lopo_spec.md.

Selection rules (7):
    Naive:  SelectByEpi, SelectByStr, SelectByRF
    Robust: RobustMin, RobustMean, Borda, MinimaxRank

For each held-out protocol P_hold in {epi, str, rf}:
    - Build rankings from the other two protocols only
    - Score each rule's top-k on P_hold (at tau_eval=0.20, seed-mean PSR)
    - Bootstrap CI over bodies (B=10000)
    - Paired bootstrap CI on (robust - naive) differences

Gate decision:
    Primary:   best-robust vs best-naive at k=5, mean across 3 splits >= 0.05,
               95% paired-bootstrap CI excludes zero, >= 2/3 splits robust > naive.
    Secondary: at k=10, any robust rule has retention >= 0.85 in all 3 splits.

Writes: records/protocol_robust_lopo_analysis.json
        records/protocol_robust_lopo_report.md

Usage:
    python scripts/analyze_protocol_robust_lopo.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "records" / "protocol_robust_lopo_analysis.json"
OUT_MD = REPO / "records" / "protocol_robust_lopo_report.md"

SEEDS = (42, 43, 44)
TAU_DIR = {"epi": "0p20", "str": "0p50", "rf": "0p00"}
EVAL_KEY = "eval_tau020"
PROTOCOLS = ("epi", "str", "rf")
KS = (5, 10)
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
PRIMARY_MARGIN = 0.05
SECONDARY_RETENTION = 0.85


def load_seed_avg(train_tau_dir: str) -> dict[str, float]:
    vals: defaultdict[str, list[float]] = defaultdict(list)
    for s in SEEDS:
        path = RESULTS / f"mjx_ppo_{train_tau_dir}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get(EVAL_KEY, {}).get("per_step_reward")
            if psr is not None:
                vals[body].append(float(psr))
    return {body: float(np.mean(v)) for body, v in vals.items()}


def rank_within(score: np.ndarray) -> np.ndarray:
    """Return ranks (1=worst, n=best) of score array."""
    order = np.argsort(score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(score) + 1)
    return ranks


def top_k_mean(score: np.ndarray, target: np.ndarray, k: int) -> float:
    idx = np.argsort(-score)[:k]
    return float(np.mean(target[idx]))


def oracle_top_k_mean(target: np.ndarray, k: int) -> float:
    return float(np.mean(np.sort(target)[::-1][:k]))


def build_rankings(
    scores: dict[str, np.ndarray], train_protocols: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Return per-rule scores (higher = better-ranked). Uses only train_protocols."""
    train_arrs = [scores[p] for p in train_protocols]
    stacked = np.stack(train_arrs, axis=0)  # (n_proto, n_bodies)
    rules: dict[str, np.ndarray] = {}
    # Naive baselines: only evaluable if their protocol is in train_protocols.
    for p in train_protocols:
        naive_name = {"epi": "SelectByEpi", "str": "SelectByStr", "rf": "SelectByRF"}[p]
        rules[naive_name] = scores[p]
    # Robust rules: aggregate across train_protocols.
    rules["RobustMin"] = np.min(stacked, axis=0)
    rules["RobustMean"] = np.mean(stacked, axis=0)
    # Borda: sum of ranks. Higher rank = better. Top-k = largest sum.
    ranks = np.stack([rank_within(scores[p]) for p in train_protocols], axis=0)
    rules["Borda"] = np.sum(ranks, axis=0)
    # MinimaxRank: best worst-case rank. Want LARGEST min-rank.
    rules["MinimaxRank"] = np.min(ranks, axis=0)
    return rules


def main() -> None:
    psr = {name: load_seed_avg(TAU_DIR[name]) for name in PROTOCOLS}
    common = sorted(
        set(psr["epi"]) & set(psr["str"]) & set(psr["rf"])
    )
    n = len(common)
    print(f"Loaded {n} bodies at {EVAL_KEY}")
    assert n == 100, f"Expected 100 bodies, got {n}"

    scores = {p: np.array([psr[p][b] for b in common]) for p in PROTOCOLS}
    # Deterministic bootstrap indices
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))

    robust_names = ("RobustMin", "RobustMean", "Borda", "MinimaxRank")
    naive_name_for = {"epi": "SelectByEpi", "str": "SelectByStr", "rf": "SelectByRF"}
    # Per-split results
    splits: dict[str, dict] = {}

    for hold in PROTOCOLS:
        train = tuple(p for p in PROTOCOLS if p != hold)
        rules = build_rankings(scores, train)
        held_target = scores[hold]
        oracle_k = {k: oracle_top_k_mean(held_target, k) for k in KS}
        per_rule: dict[str, dict] = {}
        for rname, rscore in rules.items():
            cell = {}
            for k in KS:
                pt = top_k_mean(rscore, held_target, k)
                ret = pt / oracle_k[k]
                # Bootstrap over bodies
                boot = np.empty(BOOTSTRAP_B)
                boot_ret = np.empty(BOOTSTRAP_B)
                for i in range(BOOTSTRAP_B):
                    ii = boot_idx[i]
                    # Re-rank under resampled bodies (rebuild from scores[train_p][ii])
                    # For rank-based rules, re-rank within the resample.
                    if rname == "RobustMin":
                        bs = np.min(np.stack([scores[p][ii] for p in train]), axis=0)
                    elif rname == "RobustMean":
                        bs = np.mean(np.stack([scores[p][ii] for p in train]), axis=0)
                    elif rname == "Borda":
                        bs = np.sum(
                            np.stack([rank_within(scores[p][ii]) for p in train]), axis=0
                        )
                    elif rname == "MinimaxRank":
                        bs = np.min(
                            np.stack([rank_within(scores[p][ii]) for p in train]), axis=0
                        )
                    else:
                        # naive SelectByX
                        proto = {
                            "SelectByEpi": "epi",
                            "SelectByStr": "str",
                            "SelectByRF": "rf",
                        }[rname]
                        bs = scores[proto][ii]
                    bo = held_target[ii]
                    sel_mean = np.mean(bo[np.argsort(-bs)[:k]])
                    oracle_mean = np.mean(np.sort(bo)[::-1][:k])
                    boot[i] = sel_mean
                    boot_ret[i] = sel_mean / oracle_mean if oracle_mean > 0 else np.nan
                ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
                ret_lo, ret_hi = np.quantile(boot_ret, [0.025, 0.975])
                cell[f"k{k}"] = {
                    "held_psr": float(pt),
                    "held_psr_ci": [float(ci_lo), float(ci_hi)],
                    "retention": float(ret),
                    "retention_ci": [float(ret_lo), float(ret_hi)],
                }
            per_rule[rname] = cell
        splits[hold] = {
            "held_out": hold,
            "train_protocols": list(train),
            "oracle_top_k": {f"k{k}": oracle_k[k] for k in KS},
            "rules": per_rule,
        }

    # Paired bootstrap differences: best-robust vs best-naive, at k=5 and k=10
    # "best" defined on the held-out PSR point estimate in each split.
    pair_diffs: dict[str, dict] = {}
    for hold in PROTOCOLS:
        train = tuple(p for p in PROTOCOLS if p != hold)
        rules = build_rankings(scores, train)
        held_target = scores[hold]
        pair_diffs[hold] = {}
        for k in KS:
            robust_scores_at_k = {
                r: top_k_mean(rules[r], held_target, k) for r in robust_names
            }
            naive_scores_at_k = {
                naive_name_for[p]: top_k_mean(rules[naive_name_for[p]], held_target, k)
                for p in train
            }
            best_robust = max(robust_scores_at_k, key=robust_scores_at_k.get)
            best_naive = max(naive_scores_at_k, key=naive_scores_at_k.get)
            diff_pt = robust_scores_at_k[best_robust] - naive_scores_at_k[best_naive]
            # Paired bootstrap on difference
            boot_diff = np.empty(BOOTSTRAP_B)
            for i in range(BOOTSTRAP_B):
                ii = boot_idx[i]
                if best_robust == "RobustMin":
                    rs = np.min(np.stack([scores[p][ii] for p in train]), axis=0)
                elif best_robust == "RobustMean":
                    rs = np.mean(np.stack([scores[p][ii] for p in train]), axis=0)
                elif best_robust == "Borda":
                    rs = np.sum(
                        np.stack([rank_within(scores[p][ii]) for p in train]), axis=0
                    )
                else:  # MinimaxRank
                    rs = np.min(
                        np.stack([rank_within(scores[p][ii]) for p in train]), axis=0
                    )
                nproto = {
                    "SelectByEpi": "epi",
                    "SelectByStr": "str",
                    "SelectByRF": "rf",
                }[best_naive]
                ns = scores[nproto][ii]
                bo = held_target[ii]
                rm = np.mean(bo[np.argsort(-rs)[:k]])
                nm = np.mean(bo[np.argsort(-ns)[:k]])
                boot_diff[i] = rm - nm
            lo, hi = np.quantile(boot_diff, [0.025, 0.975])
            pair_diffs[hold][f"k{k}"] = {
                "best_robust": best_robust,
                "best_naive": best_naive,
                "robust_psr": robust_scores_at_k[best_robust],
                "naive_psr": naive_scores_at_k[best_naive],
                "diff_pt": float(diff_pt),
                "diff_ci": [float(lo), float(hi)],
            }

    # Gate evaluation
    diffs_k5 = [pair_diffs[h]["k5"]["diff_pt"] for h in PROTOCOLS]
    wins_k5 = sum(d > 0 for d in diffs_k5)
    mean_diff_k5 = float(np.mean(diffs_k5))
    # Bootstrap the mean-across-splits difference
    # For each bootstrap sample, compute all 3 split diffs and their mean.
    boot_mean_diff_k5 = np.empty(BOOTSTRAP_B)
    for i in range(BOOTSTRAP_B):
        ii = boot_idx[i]
        per_split = []
        for hold in PROTOCOLS:
            train = tuple(p for p in PROTOCOLS if p != hold)
            rules_boot = build_rankings(
                {p: scores[p][ii] for p in PROTOCOLS}, train
            )
            held_target_boot = scores[hold][ii]
            # Best robust and best naive: use the NAMES pre-selected on point estimate
            best_robust = pair_diffs[hold]["k5"]["best_robust"]
            best_naive = pair_diffs[hold]["k5"]["best_naive"]
            rs = rules_boot[best_robust]
            ns = rules_boot[best_naive]
            rm = np.mean(held_target_boot[np.argsort(-rs)[:5]])
            nm = np.mean(held_target_boot[np.argsort(-ns)[:5]])
            per_split.append(rm - nm)
        boot_mean_diff_k5[i] = np.mean(per_split)
    mean_diff_k5_ci = [
        float(q) for q in np.quantile(boot_mean_diff_k5, [0.025, 0.975])
    ]
    primary_pass = (
        mean_diff_k5 >= PRIMARY_MARGIN
        and mean_diff_k5_ci[0] > 0
        and wins_k5 >= 2
    )

    # Secondary: at k=10, any robust rule has retention >= 0.85 in ALL splits
    secondary_pass = False
    for r in robust_names:
        rets = [splits[h]["rules"][r]["k10"]["retention"] for h in PROTOCOLS]
        if all(x >= SECONDARY_RETENTION for x in rets):
            secondary_pass = True
            break

    if primary_pass and secondary_pass:
        decision = "body_text"
    elif primary_pass:
        decision = "main_inclusion"
    elif secondary_pass:
        decision = "appendix_only"
    else:
        decision = "drop"

    report = {
        "spec": "records/protocol_robust_lopo_spec.md",
        "n_bodies": n,
        "splits": splits,
        "pair_diffs": pair_diffs,
        "gate": {
            "primary_pass": bool(primary_pass),
            "secondary_pass": bool(secondary_pass),
            "decision": decision,
            "diffs_k5": diffs_k5,
            "wins_k5": wins_k5,
            "mean_diff_k5": mean_diff_k5,
            "mean_diff_k5_ci": mean_diff_k5_ci,
            "primary_margin_required": PRIMARY_MARGIN,
            "secondary_retention_required": SECONDARY_RETENTION,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")

    # Markdown summary
    with OUT_MD.open("w") as f:
        f.write("# Protocol-robust selection LOPO analysis\n\n")
        f.write(f"N bodies = {n}\n\n")
        f.write("## Per-split held-out PSR at k=5\n\n")
        f.write("| Held out | Best naive | PSR | Best robust | PSR | Diff | 95% CI |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for hold in PROTOCOLS:
            pd = pair_diffs[hold]["k5"]
            f.write(
                f"| {hold} | {pd['best_naive']} | {pd['naive_psr']:.3f} "
                f"| {pd['best_robust']} | {pd['robust_psr']:.3f} "
                f"| {pd['diff_pt']:+.3f} | [{pd['diff_ci'][0]:+.3f}, {pd['diff_ci'][1]:+.3f}] |\n"
            )
        f.write("\n")
        f.write("## Gate decision\n\n")
        f.write(f"- Mean diff (robust - naive) at k=5 across splits: {mean_diff_k5:+.3f}\n")
        f.write(
            f"- 95% paired-bootstrap CI on mean diff: [{mean_diff_k5_ci[0]:+.3f},"
            f" {mean_diff_k5_ci[1]:+.3f}]\n"
        )
        f.write(f"- Splits where robust > naive at k=5: {wins_k5}/3\n")
        f.write(
            f"- Primary gate (mean diff >= {PRIMARY_MARGIN}, CI > 0, >= 2/3 wins):"
            f" {'PASS' if primary_pass else 'FAIL'}\n"
        )
        f.write(
            f"- Secondary gate (some robust rule has retention >= {SECONDARY_RETENTION}"
            f" at k=10 in all splits): {'PASS' if secondary_pass else 'FAIL'}\n"
        )
        f.write(f"\n**Decision: {decision}**\n")

    print(f"Wrote {OUT_MD}")
    print(f"\nDecision: {decision}")
    print(f"Primary gate: {'PASS' if primary_pass else 'FAIL'}")
    print(f"Secondary gate: {'PASS' if secondary_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
