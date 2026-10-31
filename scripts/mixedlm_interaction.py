#!/usr/bin/env python3
"""Reproduce the mixed-effects random-intercept interaction coefficient.

The appendix (section `sec:per_seed_appendix`, "Alternative blocked
models") reports the VT-by-reset-free interaction coefficient = +0.67,
95% CI [0.39, 0.95] from a random-intercept-per-morphology LMM on the
per-seed 900-row corpus. `scripts/reproduce_paper_stats.py` does not
emit this row (statsmodels MixedLM is comparatively slow; the main
reproduction script keeps runtime short). This script regenerates it
on demand.

Usage:
    python scripts/mixedlm_interaction.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
OUT = PROJ / "records" / "mixedlm_interaction.json"

SEEDS = (42, 43, 44)
TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}
EVAL_KEY = "eval_tau020"


def family_of(name: str) -> str | None:
    for f in ("floor", "mvt", "vt"):
        if name.startswith(f):
            return f
    return None


def load_long() -> pd.DataFrame:
    per_seed: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for proto, tau in TAUS.items():
        for s in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{s}" / "results.json"
            data = json.load(path.open())
            for body, rec in data.items():
                v = rec.get(EVAL_KEY, {}).get("per_step_reward")
                if v is not None:
                    per_seed[(proto, body)][s] = float(v)
    rows = []
    for (proto, body), seed_map in per_seed.items():
        fam = family_of(body)
        if fam is None:
            continue
        for s, v in seed_map.items():
            rows.append(
                {"psr": v, "protocol": proto, "family": fam, "morph": body, "seed": s}
            )
    df = pd.DataFrame(rows)
    common = df.groupby("morph")["protocol"].nunique() == 3
    df = df[df["morph"].isin(common[common].index)].copy()
    return df


def main() -> None:
    df = load_long()
    print(f"Fit on {len(df)} rows, {df['morph'].nunique()} bodies")
    model = smf.mixedlm(
        "psr ~ C(protocol, Treatment(reference='episodic')) * "
        "C(family, Treatment(reference='floor'))",
        data=df,
        groups=df["morph"],
    )
    fit = model.fit(reml=True, method="powell")
    params = fit.params
    conf = fit.conf_int(alpha=0.05)
    rows = []
    for term in params.index:
        coef = float(params[term])
        lo, hi = float(conf.loc[term, 0]), float(conf.loc[term, 1])
        rows.append({"term": term, "coef": coef, "ci_lo": lo, "ci_hi": hi})
    vt_rf = [
        r
        for r in rows
        if "T.vt" in r["term"] and "T.reset_free" in r["term"] and ":" in r["term"]
    ]
    vt_str = [
        r
        for r in rows
        if "T.vt" in r["term"] and "T.strict" in r["term"] and ":" in r["term"]
    ]
    rf_main = [r for r in rows if "T.reset_free" in r["term"] and ":" not in r["term"]]
    payload = {
        "n_rows": int(len(df)),
        "n_bodies": int(df["morph"].nunique()),
        "reference_protocol": "episodic",
        "reference_family": "floor",
        "all_terms": rows,
        "vt_by_reset_free_interaction": vt_rf[0] if vt_rf else None,
        "vt_by_strict_interaction": vt_str[0] if vt_str else None,
        "reset_free_main": rf_main[0] if rf_main else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"VT x reset_free interaction: {vt_rf[0] if vt_rf else 'MISSING'}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
