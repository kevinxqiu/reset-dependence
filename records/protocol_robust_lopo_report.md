# Protocol-robust selection LOPO analysis

N bodies = 100

## Per-split held-out PSR at k=5

| Held out | Best naive | PSR | Best robust | PSR | Diff | 95% CI |
|---|---|---|---|---|---|---|
| epi | SelectByRF | 3.650 | RobustMin | 3.524 | -0.126 | [-0.592, +0.059] |
| str | SelectByRF | 2.340 | RobustMin | 2.567 | +0.227 | [-0.139, +0.626] |
| rf | SelectByEpi | 3.614 | MinimaxRank | 3.636 | +0.023 | [-0.878, +0.217] |

## Gate decision

- Mean diff (robust - naive) at k=5 across splits: +0.041
- 95% paired-bootstrap CI on mean diff: [-0.421, +0.228]
- Splits where robust > naive at k=5: 2/3
- Primary gate (mean diff >= 0.05, CI > 0, >= 2/3 wins): FAIL
- Secondary gate (some robust rule has retention >= 0.85 at k=10 in all splits): FAIL

**Decision: drop**
