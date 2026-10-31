#!/usr/bin/env python3
"""Recompute the statistics of the review-period panels from their per-cell tables.

During the review discussion (July 2026) we trained five panels outside the
UNIMAL corpus: Humanoid fall rules, Humanoid time-limit handling, Ant reset
initialization, Ant protocol breadth and Evolution Gym time-limit handling.
panels.zip of the dataset holds one CSV row per cell (morphology, protocol,
seed and hardware class) with the score that each frozen report ranks, the
Humanoid evaluation curves, and the frozen reports, specifications,
contracts and inventories. Unzip it at the repository root (it creates
review-panels/) or pass --panels.

This script uses only the CSV tables, the panel specifications (for the
declared comparisons and analysis settings) and the composite specification
of the Humanoid time-limit panel. It rebuilds every report with the
estimators and random streams of the frozen analysis code: Spearman rho of
seed-mean scores, its 95% morphology-bootstrap interval (B = 10,000,
numpy default_rng with the seed of each comparison), top-10 overlap with ties
broken by the lower morphology index, winner regret, the medians over
complementary seed splits, cell counts, the per-morphology means, standard
deviations and seed scores, the pool heterogeneity summary and, for the
Humanoid time-limit composite, the correlation difference and the rank
stability over evaluation points. It then compares every value with the
shipped report.json (for the Ant breadth panel, whose report.json was not
kept, with the frozen summary table) and prints the result. Exit status 1
if any value differs.

The paired protocol-label permutation p-values and their Holm adjustments
are recomputed only to check the shipped reports. The paper does not use
them: that test also responds to differences in reward level between two
settings, so a large p-value does not show that rankings agree.

    python scripts/review_panels.py                      # about 4 minutes on a laptop CPU
    python scripts/review_panels.py --skip-permutation   # about 2 minutes, leaves the p-values unchecked

Writes records/review_panels_check.json (summary of the comparison).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import warnings
from collections import defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
PANELS = ("humanoid-fall-rule", "humanoid-time-limit", "ant-reset-init", "ant-breadth", "evogym-time-limit")
TOL = 1e-12  # relative tolerance for floats that are not bit-identical

# --------------------------------------------------------------------------
# Estimators (the frozen analysis code, with the same random streams)
# --------------------------------------------------------------------------


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """scipy.stats.spearmanr(x, y).statistic for 1-D float arrays, without its p-value."""
    if (x[0] == x).all() or (y[0] == y).all():
        return float("nan")  # constant input: undefined, as in scipy
    ranked = np.column_stack((rankdata(x), rankdata(y)))
    return float(np.corrcoef(ranked, rowvar=False)[1, 0])


def _spearman_scipy(x: np.ndarray, y: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x, y).statistic)


def _pick_spearman():
    """Use the fast path only if it is bit-identical to scipy on this installation."""
    rng = np.random.default_rng(12345)
    for n in (4, 20, 30):
        for _ in range(300):
            x = np.round(rng.normal(size=n), 1)  # rounding creates ties
            y = np.round(rng.normal(size=n), 1)
            idx = rng.integers(0, n, size=n)
            for a, b in ((x, y), (x[idx], y[idx]), (np.ones(n), y)):
                fa, sa = _spearman_fast(a, b), _spearman_scipy(a, b)
                if not (fa == sa or (np.isnan(fa) and np.isnan(sa))):
                    return _spearman_scipy, "scipy.stats.spearmanr"
    return _spearman_fast, "rankdata and corrcoef (bit-identical to scipy.stats.spearmanr here)"


SPEARMAN, SPEARMAN_NOTE = _pick_spearman()


def bootstrap_spearman(first, second, n_resamples, seed):
    rho = SPEARMAN(first, second)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_resamples):
        idx = rng.integers(0, first.size, size=first.size)
        r = SPEARMAN(first[idx], second[idx])
        if np.isfinite(r):
            boot.append(r)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"rho": rho, "ci_low": float(lo), "ci_high": float(hi), "n_resamples": len(boot), "seed": int(seed)}


def bootstrap_spearman_difference(first_a, second_a, first_b, second_b, n_resamples, seed):
    rho_a, rho_b = SPEARMAN(first_a, second_a), SPEARMAN(first_b, second_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, first_a.size, size=first_a.size)
        ra, rb = SPEARMAN(first_a[idx], second_a[idx]), SPEARMAN(first_b[idx], second_b[idx])
        if np.isfinite(ra) and np.isfinite(rb):
            diffs.append(ra - rb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"rho_a": rho_a, "rho_b": rho_b, "difference": rho_a - rho_b, "ci_low": float(lo),
            "ci_high": float(hi), "n_resamples": len(diffs), "seed": int(seed)}


def top_k_overlap(first, second, k):
    idx = np.arange(first.size)
    a = set(np.lexsort((idx, -first))[:k].tolist())
    b = set(np.lexsort((idx, -second))[:k].tolist())
    return len(a & b)


def winner_regret(selection, common):
    return float(np.max(common) - common[int(np.argmax(selection))])


def seed_split_reliability(scores):
    n = scores.shape[1]
    every = frozenset(range(n))
    seen, rhos = set(), []
    for first_tuple in combinations(range(n), n // 2):
        first = frozenset(first_tuple)
        second = every - first
        pair = tuple(sorted((tuple(sorted(first)), tuple(sorted(second)))))
        if pair in seen:
            continue
        seen.add(pair)
        r = SPEARMAN(scores[:, list(first)].mean(axis=1), scores[:, list(second)].mean(axis=1))
        if np.isfinite(r):
            rhos.append(r)
    return {"median_rho": float(np.median(rhos)), "correlations": rhos, "n_splits": len(rhos)}


def permutation_test(first, second, n_permutations, seed):
    rho = SPEARMAN(first.mean(axis=1), second.mean(axis=1))
    observed = 1.0 - rho
    rng = np.random.default_rng(seed)
    exceed = undefined = 0
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=first.shape, dtype=np.int8).astype(bool)
        r = SPEARMAN(np.where(swap, second, first).mean(axis=1), np.where(swap, first, second).mean(axis=1))
        if np.isfinite(r):
            null = 1.0 - r
        else:
            undefined += 1
            null = 2.0
        if null >= observed:
            exceed += 1
    return {"rho": rho, "statistic": float(observed), "p_value": float((1 + exceed) / (1 + n_permutations)),
            "n_permutations": int(n_permutations), "undefined_permutations": int(undefined), "seed": int(seed),
            "n_morphologies": int(first.shape[0]), "n_seeds": int(first.shape[1])}


def holm_adjust(p_values: dict) -> dict:
    adjusted, running = {}, 0.0
    for rank, (name, value) in enumerate(sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

INT_COLUMNS = {"pool_id", "morphology_index", "seed", "eval_index", "step"}
FLOAT_COLUMNS = {"score", "eval_episode_reward"}


def read_table(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            for c in INT_COLUMNS & r.keys():
                r[c] = int(r[c])
            for c in FLOAT_COLUMNS & r.keys():
                r[c] = float(r[c])
            rows.append(r)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(rows, morphologies, seeds):
    by = {(r["morphology_index"], r["seed"]): r["score"] for r in rows}
    if set(by) != {(m, s) for m in morphologies for s in seeds} or len(by) != len(rows):
        raise SystemExit("ERROR: a protocol grid is not a complete morphology by seed table")
    return np.array([[by[(m, s)] for s in seeds] for m in morphologies], dtype=np.float64)


def morphology_scores(mat, morphologies, seeds):
    return [{"morphology_index": int(m), "mean": float(mat[i].mean()),
             "std": float(mat[i].std(ddof=1 if len(seeds) > 1 else 0)),
             "seed_scores": [{"seed": int(s), "score": float(mat[i, j])} for j, s in enumerate(seeds)]}
            for i, m in enumerate(morphologies)]


def pair_summary(first, second, a, b, analysis, seed, skip_perm, reliability=True):
    fm, sm = first.mean(axis=1), second.mean(axis=1)
    out = {
        "protocols": [a, b],
        "rank_correlation": bootstrap_spearman(fm, sm, analysis["n_resamples"], seed),
        "top_k": {"k": analysis["top_k"], "overlap": top_k_overlap(fm, sm, analysis["top_k"])},
        "winner_regret": {f"{a}_selected_under_{b}": winner_regret(fm, sm),
                          f"{b}_selected_under_{a}": winner_regret(sm, fm)},
        "seed_split_reliability": ({a: seed_split_reliability(first), b: seed_split_reliability(second)}
                                   if reliability else {}),
    }
    if not skip_perm:
        out["permutation_test"] = permutation_test(first, second, analysis["n_permutations"], seed)
    return out


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def sanitized_contract(c: dict) -> dict:
    pa = c.get("panel_approval")
    return {"schema_version": c["schema_version"], "panel_spec_sha256": c["panel_spec_sha256"],
            "allowed_git_revisions": list(c["allowed_git_revisions"]),
            "analysis_git_revision": c["analysis_git_revision"],
            "revision_compatibility_basis_recorded": bool(c.get("revision_compatibility_basis")),
            "legacy_admission": dict(c["legacy_admission"]),
            "panel_approval": None if pa is None else {k: pa[k] for k in (
                "registry_git_revision", "selection_anchor_git_revision", "selection_anchor_sha256")}}


def panel_report(folder: Path, skip_perm: bool) -> dict:
    """Rebuild a single-panel report from cells.csv and panel_spec.json."""
    spec = json.loads((folder / "panel_spec.json").read_text())
    analysis = spec["analysis"]
    rows = read_table(folder / "cells.csv")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["space"], r["pool_id"], False, (r["task"],) if r["task"] else (), r["hardware"])].append(r)
    comparisons_declared = [tuple(c) for c in spec["comparisons"]]
    panels, raw_p = [], {}
    for panel_index, (key, recs) in enumerate(sorted(groups.items())):
        space, pool_id, wide, tasks, hardware = key
        by_protocol = defaultdict(list)
        for r in recs:
            by_protocol[r["protocol"]].append(r)
        morphologies = sorted({r["morphology_index"] for r in recs})
        seeds = sorted({r["seed"] for r in recs})
        mats = {p: matrix(rs, morphologies, seeds) for p, rs in by_protocol.items()}
        panel_seed = analysis["seed"] + panel_index * 10_000
        comps = []
        for ci, (a, b) in enumerate(comparisons_declared):
            c = {"comparison_id": f"{a}__vs__{b}",
                 **pair_summary(mats[a], mats[b], a, b, analysis, panel_seed + ci, skip_perm,
                                reliability=len(seeds) >= 4)}
            comps.append(c)
            if not skip_perm:
                raw_p[f"panel-{panel_index}:{c['comparison_id']}"] = c["permutation_test"]["p_value"]
        panels.append({
            "space": space, "pool_id": pool_id, "wide_pool": wide, "tasks": list(tasks), "hardware": hardware,
            "reportable": True, "git_revisions": sorted({r["git_revision"] for r in recs if r["git_revision"]}),
            "total_cells": len(recs), "completed_cells": len(recs),
            "protocol_cell_counts": {p: len(by_protocol[p]) for p in sorted(by_protocol)},
            "comparisons": comps,
            "morphology_scores": {p: morphology_scores(m, morphologies, seeds) for p, m in mats.items()},
            "issues": [],
        })
    if not skip_perm:
        adj = holm_adjust(raw_p)
        for i, panel in enumerate(panels):
            for c in panel["comparisons"]:
                c["permutation_test"]["holm_p_value"] = adj[f"panel-{i}:{c['comparison_id']}"]
    het = defaultdict(list)
    for panel in panels:
        for c in panel["comparisons"]:
            het[(panel["space"], panel["wide_pool"], tuple(panel["tasks"]), panel["hardware"],
                 c["comparison_id"])].append({"pool_id": panel["pool_id"], "rho": c["rank_correlation"]["rho"]})
    heterogeneity = []
    for (space, wide, tasks, hardware, cid), values in sorted(het.items()):
        if len(values) < 2:
            continue
        rhos = [v["rho"] for v in values]
        heterogeneity.append({"space": space, "wide_pool": wide, "tasks": list(tasks), "hardware": hardware,
                              "comparison_id": cid, "pool_count": len(values),
                              "pool_ids": sorted(v["pool_id"] for v in values),
                              "rho_by_pool": sorted(values, key=lambda v: v["pool_id"]),
                              "rho_median": float(np.median(rhos)), "rho_min": float(np.min(rhos)),
                              "rho_max": float(np.max(rhos))})
    contract = json.loads((folder / "panel_contract.json").read_text())
    return {
        "schema_version": 1, "panel_spec": spec, "panel_contract": sanitized_contract(contract),
        "reportable": True, "status": "REPORTABLE", "panels": panels, "pool_heterogeneity": heterogeneity,
        "settings": {**analysis, "hardware_pooling": False, "declared_comparisons_only": True,
                     "holm_family": "all declared comparisons across all report panels",
                     "result_root_ids": list(spec["result_root_ids"])},
    }


def composite_report(boot_dir: Path, term_dir: Path, skip_perm: bool) -> dict:
    """Rebuild the Humanoid time-limit composite from both Humanoid panels' tables."""
    spec = json.loads((term_dir / "composite_spec.json").read_text())
    shared, analysis = spec["shared_contract"], spec["analysis"]
    morphologies, seeds = shared["morphology_indices"], shared["seeds"]
    cells = {"bootstrap": read_table(boot_dir / "cells.csv"), "terminal": read_table(term_dir / "cells.csv")}
    prog = {"bootstrap": read_table(boot_dir / "progress.csv"), "terminal": read_table(term_dir / "progress.csv")}
    arms = {}
    for child, mapping in spec["protocol_mapping"].items():
        for logical, actual in mapping.items():
            arms[(child, logical)] = actual
    pools, raw_p = [], {}
    for pool_index, pool_id in enumerate(shared["pool_ids"]):
        mats = {arm: matrix([r for r in cells[arm[0]] if r["pool_id"] == pool_id and r["protocol"] == actual],
                            morphologies, seeds) for arm, actual in arms.items()}

        def summary(a_arm, b_arm, cid, a, b, seed):
            s = {"comparison_id": cid, **pair_summary(mats[a_arm], mats[b_arm], a, b, analysis, seed, skip_perm),
                 "morphology_scores": {a: morphology_scores(mats[a_arm], morphologies, seeds),
                                       b: morphology_scores(mats[b_arm], morphologies, seeds)}}
            return s

        primary = []
        for ci, rule in enumerate(("native", "none")):
            cid = f"pool-{pool_id}:{rule}-bootstrap-vs-terminal"
            s = summary(("bootstrap", rule), ("terminal", rule), cid, f"{rule}-bootstrap", f"{rule}-terminal",
                        analysis["seed"] + pool_index * 100 + ci)
            s.update(inference_role="primary", multiplicity_controlled=True)
            primary.append(s)
            if not skip_perm:
                raw_p[cid] = s["permutation_test"]["p_value"]
        boot_fall = summary(("bootstrap", "none"), ("bootstrap", "native"), f"pool-{pool_id}:none-vs-native-bootstrap",
                            "none-bootstrap", "native-bootstrap", analysis["seed"] + pool_index * 100 + 10)
        term_fall = summary(("terminal", "none"), ("terminal", "native"), f"pool-{pool_id}:none-vs-native-terminal",
                            "none-terminal", "native-terminal", analysis["seed"] + pool_index * 100 + 11)
        for s in (boot_fall, term_fall):
            s.update(inference_role="secondary", multiplicity_controlled=False)
        diff = bootstrap_spearman_difference(
            mats[("terminal", "none")].mean(axis=1), mats[("terminal", "native")].mean(axis=1),
            mats[("bootstrap", "none")].mean(axis=1), mats[("bootstrap", "native")].mean(axis=1),
            analysis["n_resamples"], analysis["seed"] + pool_index * 100 + 12)
        stability = {}
        for (child, logical), actual in arms.items():
            pts = defaultdict(dict)
            for r in prog[child]:
                if r["pool_id"] == pool_id and r["protocol"] == actual:
                    pts[(r["morphology_index"], r["seed"])][r["eval_index"]] = (r["step"], r["eval_episode_reward"])
            n_eval = shared["num_evals"]
            steps = [i * shared["actual_total_steps"] // (n_eval - 1) for i in range(n_eval)]
            sc = np.empty((n_eval, len(morphologies), len(seeds)))
            for i, m in enumerate(morphologies):
                for j, s in enumerate(seeds):
                    cell = pts[(m, s)]
                    if [cell[e][0] for e in range(n_eval)] != steps:
                        raise SystemExit(f"ERROR: evaluation steps of pool {pool_id} {actual} {m}/{s} differ")
                    sc[:, i, j] = [cell[e][1] for e in range(n_eval)]
            if not np.array_equal(sc[-1], mats[(child, logical)]):
                raise SystemExit(f"ERROR: last evaluation differs from the cell score ({actual}, pool {pool_id})")
            final = sc[-1].mean(axis=1)
            points = []
            for e, step in enumerate(steps):
                r = SPEARMAN(sc[e].mean(axis=1), final)
                points.append({"step": step, "rho_to_final": r if np.isfinite(r) else None, "defined": bool(np.isfinite(r))})
            stability[f"{logical}-{child}"] = {"points": points}
        pools.append({"pool_id": pool_id, "primary_comparisons": primary,
                      "secondary_comparisons": {"bootstrap_fall_rule": boot_fall, "terminal_fall_rule": term_fall,
                                                "correlation_difference": {"direction": "terminal-minus-bootstrap", **diff}},
                      "rank_stability": stability})
    if not skip_perm:
        if list(raw_p) != analysis["holm_family"]:
            raise SystemExit("ERROR: primary comparisons differ from the frozen Holm family")
        adj = holm_adjust(raw_p)
        for pool in pools:
            for c in pool["primary_comparisons"]:
                c["permutation_test"]["holm_p_value"] = adj[c["comparison_id"]]
    children = {}
    for child, folder in (("bootstrap", boot_dir), ("terminal", term_dir)):
        revs = {r["git_revision"] for r in cells[child]}
        children[child] = {"panel_id": cells[child][0]["panel_id"], "execution_git_revision": revs.pop() if len(revs) == 1 else sorted(revs),
                           "panel_contract_sha256": sha256_file(folder / "panel_contract.json"),
                           "inventory_sha256": sha256_file(folder / "inventory.jsonl")}
    return {"schema_version": 1, "composite_id": spec["composite_id"], "reportable": True, "status": "REPORTABLE",
            "children": children,
            "settings": {**analysis, "hardware": shared["hardware"], "hardware_pooling": False,
                         "morphology_is_inference_unit": True, "declared_comparisons_only": True},
            "descriptive_summary": {"median_pool_correlation_difference": float(np.median(
                [p["secondary_comparisons"]["correlation_difference"]["difference"] for p in pools])),
                "role": "descriptive", "hypothesis_test": False},
            "pools": pools}


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list) and obj and all(isinstance(v, (dict, list)) for v in obj):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _leaf_match(want, have) -> str | None:
    """'exact', 'close' (floats within TOL, relative) or None."""
    if isinstance(want, list) and isinstance(have, list):
        if len(want) != len(have):
            return None
        kinds = [_leaf_match(w, h) for w, h in zip(want, have)]
        return None if None in kinds else ("close" if "close" in kinds else "exact")
    if isinstance(want, bool) or isinstance(have, bool) or not isinstance(want, (int, float)):
        return "exact" if (want == have and type(want) is type(have)) else None
    if not isinstance(have, (int, float)):
        return None
    if want == have:
        return "exact"
    if isinstance(want, float) and abs(have - want) <= TOL * max(1.0, abs(want)):
        return "close"
    return None


# Paths of the recomputed statistics; the rest of a report is settings, identifiers and the copied
# specification, which are compared too.
STAT_PATHS = re.compile(r"^(panels|pools|pool_heterogeneity|descriptive_summary)")


def compare(shipped: dict, recomputed: dict, ignore: tuple[str, ...]):
    got = dict(flatten(recomputed))
    stats = {"compared": 0, "exact": 0, "close": 0, "statistics": 0}
    mismatches, missing = [], []
    for path, want in flatten(shipped):
        if any(re.search(p, path) for p in ignore):
            missing.append(path)
            continue
        if path not in got:
            mismatches.append((path, want, "<not recomputed>"))
            continue
        kind = _leaf_match(want, got[path])
        n = len(want) if isinstance(want, list) and want else 1
        stats["compared"] += n
        if STAT_PATHS.match(path) and (isinstance(want, (int, float)) or isinstance(want, list)) \
                and not isinstance(want, bool):
            stats["statistics"] += n
        if kind is None:
            mismatches.append((path, want, got[path]))
        else:
            stats[kind] += n
    return stats, mismatches, missing


FROZEN_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([-.\d]+)\s*\|\s*\[([-.\d]+),\s*([-.\d]+)\]\s*\|\s*(\d+)/10\s*\|$")
BREADTH_LABELS = {"Deterministic reset initialization": "reset-deterministic",
                  "Terminal timeout handling": "timeout-terminal", "Observation noise": "observation-noise"}


def compare_frozen_summary(text: str, rebuilt: dict):
    """Ant breadth: the frozen summary table rounds to three decimals."""
    by = {}
    for panel in rebuilt["panels"]:
        for c in panel["comparisons"]:
            by[(panel["pool_id"], c["protocols"][1])] = c
    stats = {"compared": 0, "exact": 0, "close": 0, "statistics": 0}
    mismatches = []

    def check(path, frozen: str, value, places=3):
        stats["compared"] += 1
        stats["statistics"] += 1
        want = Decimal(frozen if not frozen.startswith(("." , "-.")) else frozen.replace(".", "0.", 1))
        have = Decimal(f"{value:.{places}f}") if places else Decimal(value)
        if have == want:
            stats["exact"] += 1
        else:
            mismatches.append((path, frozen, f"{value}"))

    rows = [FROZEN_ROW.match(line.strip()) for line in text.splitlines()]
    rows = [m for m in rows if m]
    for m in rows:
        pool, label, rho, lo, hi, top = int(m[1]), m[2], m[3], m[4], m[5], m[6]
        c = by[(pool, BREADTH_LABELS[label])]
        base = f"pool {pool} {BREADTH_LABELS[label]}"
        check(f"{base} rho", rho, c["rank_correlation"]["rho"])
        check(f"{base} ci_low", lo, c["rank_correlation"]["ci_low"])
        check(f"{base} ci_high", hi, c["rank_correlation"]["ci_high"])
        check(f"{base} top-10", top, c["top_k"]["overlap"], places=0)
    cells = re.search(r"contains ([\d,]+)/([\d,]+) cells", text)
    total = sum(p["total_cells"] for p in rebuilt["panels"])
    for i in (1, 2):
        check(f"cells ({'completed' if i == 1 else 'expected'})", cells[i].replace(",", ""), total, places=0)
    if len(rows) != 9:
        mismatches.append(("frozen table rows", "9", str(len(rows))))
    return stats, mismatches


def headline(name: str, report: dict) -> list[dict]:
    out = []
    if "panels" in report:
        for p in report["panels"]:
            for c in p["comparisons"]:
                rc = c["rank_correlation"]
                out.append({"panel": name, "pool": p["pool_id"], "task": "+".join(p["tasks"]),
                            "comparison": " vs ".join(c["protocols"]), "rho": rc["rho"], "ci": [rc["ci_low"], rc["ci_high"]],
                            "top10": c["top_k"]["overlap"],
                            "seed_half_medians": {k: v["median_rho"] for k, v in c["seed_split_reliability"].items()}})
    else:
        for p in report["pools"]:
            for c in p["primary_comparisons"] + [p["secondary_comparisons"]["bootstrap_fall_rule"],
                                                 p["secondary_comparisons"]["terminal_fall_rule"]]:
                rc = c["rank_correlation"]
                out.append({"panel": name, "pool": p["pool_id"], "task": "", "comparison": " vs ".join(c["protocols"]),
                            "rho": rc["rho"], "ci": [rc["ci_low"], rc["ci_high"]], "top10": c["top_k"]["overlap"],
                            "seed_half_medians": {k: v["median_rho"] for k, v in c["seed_split_reliability"].items()},
                            "role": c["inference_role"]})
            d = p["secondary_comparisons"]["correlation_difference"]
            out.append({"panel": name, "pool": p["pool_id"], "task": "",
                        "comparison": "fall-rule rho, terminal minus bootstrapped", "rho": d["difference"],
                        "ci": [d["ci_low"], d["ci_high"]], "top10": None, "seed_half_medians": {}, "role": "secondary"})
    return out


def integrity(folder: Path) -> list[tuple[str, bool]]:
    """Digests of the shipped inputs against the values recorded by the frozen analysis."""
    checks = []
    spec = json.loads((folder / "panel_spec.json").read_text())
    contract = json.loads((folder / "panel_contract.json").read_text())
    checks.append(("inventory.jsonl sha256 = panel_spec inventory_sha256",
                   sha256_file(folder / "inventory.jsonl") == spec["inventory_sha256"]))
    inv = [json.loads(line) for line in (folder / "inventory.jsonl").read_text().splitlines() if line.strip()]
    cells = read_table(folder / "cells.csv")

    def key(r):
        task = r["tasks"][0] if "tasks" in r else r.get("task", "")
        return (r["pool_id"], task, r["morphology_index"], r["protocol"], r["seed"])

    checks.append(("cells.csv rows = inventory cells", sorted(map(key, cells)) == sorted(map(key, inv))))
    checks.append(("cell count = expected_cell_count", len(cells) == spec["expected_cell_count"]))
    checks.append(("training revisions allowed by the contract",
                   {r["git_revision"] for r in cells} <= set(contract["allowed_git_revisions"])))
    man = folder / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text())
        if "panel_contract_sha256" in m:
            checks.append(("panel_contract.json sha256 = manifest", sha256_file(folder / "panel_contract.json") == m["panel_contract_sha256"]))
        for child, want in m.get("child_contract_sha256", {}).items():
            other = folder if child == "terminal" else folder.parent / "humanoid-fall-rule"
            checks.append((f"{child} child panel_contract.json sha256 = manifest",
                           sha256_file(other / "panel_contract.json") == want))
        for f, want in m.get("files", {}).items():
            same = sha256_file(folder / f) == want
            if f == "report.json" and not same and "internal design note" in (folder / f).read_text():
                checks.append((f"{f} sha256 = manifest (masked copy, one design-note path; not comparable)", None))
            else:
                checks.append((f"{f} sha256 = manifest", same))
        if "composite_spec_sha256" in m:
            checks.append(("composite_spec.json sha256 = manifest (masked copy, one design-note path; not comparable)", None))
        verdict = folder / "wrapper-verdict.json"
        if verdict.is_file():
            v = json.loads(verdict.read_text())
            checks.append(("wrapper-verdict status verified, manifest.json sha256 = verdict",
                           v.get("status") == "verified" and sha256_file(man) == v.get("manifest_sha256")))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panels", type=Path, default=REPO / "review-panels", help="unzipped review-panels/ folder")
    ap.add_argument("--out", type=Path, default=REPO / "records" / "review_panels_check.json")
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--skip-permutation", action="store_true", help="skip the (withdrawn) permutation p-values")
    args = ap.parse_args()
    root = args.panels
    if not (root / PANELS[0] / "cells.csv").is_file():
        print(f"ERROR: {root} does not hold the panel tables; unzip panels.zip at the repository root or pass --panels")
        return 1
    print(f"Spearman: {SPEARMAN_NOTE}")
    if args.skip_permutation:
        print("Permutation p-values and Holm adjustments are skipped and left unchecked.")

    perm_paths = (r"permutation_test", r"holm") if args.skip_permutation else ()
    # Values the tables cannot reproduce: provenance of the analysis run and digests of raw files.
    not_from_tables = (r"^analysis_provenance\.", r"rank_stability\..*\.progress_sha256$",
                       r"^children\.\w+\.panel_spec_sha256$")
    results, records, failed = {}, {}, False
    for name in PANELS:
        folder = root / name
        print(f"\n== {name}")
        if name == "humanoid-time-limit":
            rebuilt = composite_report(root / "humanoid-fall-rule", folder, args.skip_permutation)
        else:
            rebuilt = panel_report(folder, args.skip_permutation)
        if (folder / "report.json").is_file():
            shipped = json.loads((folder / "report.json").read_text())
            stats, mismatches, skipped = compare(shipped, rebuilt, not_from_tables + perm_paths)
            source = "report.json"
        else:
            stats, mismatches = compare_frozen_summary((folder / "frozen_report.md").read_text(), rebuilt)
            skipped = []
            source = "frozen_report.md (summary table, three decimals; the report.json was not kept)"
        checks = integrity(folder)
        bad_checks = [c for c, ok in checks if ok is False]
        for c, ok in checks:
            print(f"  {'ok  ' if ok else ('n/a ' if ok is None else 'FAIL')} {c}")
        n_cells = sum(1 for _ in open(folder / "cells.csv")) - 1
        print(f"  {n_cells} cells; compared {stats['compared']} values with {source} "
              f"({stats['statistics']} of them statistics and counts): "
              f"{stats['exact']} identical, {stats['close']} within {TOL:g} (relative), {len(mismatches)} different")
        skipped_kinds = sorted({re.sub(r"\[\d+\]", "[]", p) for p in skipped})
        if skipped:
            print(f"  not recomputable from the tables ({len(skipped)} values): " + "; ".join(skipped_kinds[:6])
                  + (" ..." if len(skipped_kinds) > 6 else ""))
        for path, want, have in mismatches[:20]:
            print(f"  DIFFERENT {path}: shipped {want!r}, recomputed {have!r}")
        failed |= bool(mismatches) or bool(bad_checks)
        heads = headline(name, rebuilt)
        for h in heads:
            sh = ", ".join(f"{k} {v:.2f}" for k, v in h["seed_half_medians"].items())
            top = "" if h["top10"] is None else f", top-10 {h['top10']}"
            task = f" {h['task']}" if h["task"] else ""
            print(f"  pool {h['pool']}{task} {h['comparison']}: rho {h['rho']:.3f} "
                  f"[{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]{top}" + (f"; seed halves {sh}" if sh else ""))
        results[name] = {"cells": n_cells, "compared_with": source, **stats, "different": len(mismatches),
                         "mismatches": [list(map(str, m)) for m in mismatches],
                         "not_recomputable": skipped_kinds, "integrity": [[c, ok] for c, ok in checks]}
        records[name] = heads

    total = {k: sum(r[k] for r in results.values()) for k in ("compared", "statistics", "exact", "close", "different")}
    print(f"\nAll panels: compared {total['compared']} values ({total['statistics']} statistics and counts), "
          f"{total['exact']} identical, {total['close']} within {TOL:g}, {total['different']} different.")
    print("RESULT: " + ("every recomputed value matches" if not failed else "DIFFERENCES FOUND"))
    if not args.no_write:
        out = {"script": "scripts/review_panels.py", "spearman": SPEARMAN_NOTE,
               "permutation_p_values": "skipped" if args.skip_permutation else "recomputed (not reported; see the docstring)",
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
               "totals": total, "all_match": not failed, "panels": results, "headline": records}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
