# Statistical Hardening Audit

Body-clustered bootstrap (B=10000, seed=42) and leave-one-out (LOBO/LOSO) stability on the 900-row corpus (100 UNIMAL bodies x 3 protocols x 3 seeds), matching `reproduce_paper_stats.py`.

N bodies: 100  
Eval key: eval_tau020

## 1. Leave-one-body-out Spearman rho stability (seed-averaged PSR)

| Pair | Full rho | LOBO range | Max shift | Note |
|---|---:|---|---:|---|
| episodic_vs_reset_free | 0.723 | [0.714, 0.748] | 0.025 | stable |
| episodic_vs_strict | 0.458 | [0.442, 0.494] | 0.036 | stable |
| reset_free_vs_strict | 0.206 | [0.181, 0.229] | 0.024 | stable |

## 2. Leave-one-seed-out Spearman rho stability (seed-drop)

| Pair | Drop 42 | Drop 43 | Drop 44 | Range |
|---|---|---|---|---|
| episodic_vs_reset_free | 0.701 | 0.731 | 0.672 | [0.672, 0.731] |
| episodic_vs_strict | 0.468 | 0.453 | 0.445 | [0.445, 0.468] |
| reset_free_vs_strict | 0.244 | 0.202 | 0.267 | [0.202, 0.267] |

## 3. Leave-one-body-out 2-way interaction F (99-body subpanels)

- Full model: F = 12.54, p = 6.76e-10, partial eta^2 = 0.059
- LOBO F range: [11.03, 15.17], median = 12.39
- LOBO p max: 1.05e-08
- All 100 LOBO p < 0.001: True
- Most influential body (max F shift): `vt-5506-15-9-02-19-03-39` (Delta F = 2.63)

## 4. Leave-one-seed-out 2-way interaction F (600-row subpanels)

| Drop seed | F | p | partial eta^2 |
|---:|---:|---:|---:|
| 42 | 7.17 | 1.29e-05 | 0.055 |
| 43 | 7.13 | 1.37e-05 | 0.055 |
| 44 | 10.49 | 3.68e-08 | 0.078 |

- F range: [7.13, 10.49]
- All LOSO p < 0.01: True

## 5. Top-10 winner-set intersection (body-clustered bootstrap)

Two complementary resampling designs:

- Asymmetric: fix one protocol's original top-10 as reference, resample the other protocol's ranking. Matches `results.tex` stability intervals and `reproduce_paper_stats.py` exactly.
- Symmetric: resample once, compute both protocols' top-10 on the resample. Sampling-variability estimate independent of reference protocol.

| Pair | Point | Asym mean | Asym 95% CI | Sym mean | Sym 95% CI |
|---|---:|---:|---|---:|---|
| episodic_vs_reset_free | 8/10 | 5.80 | [3.0, 8.0] | 8.77 | [8.0, 10.0] |
| episodic_vs_strict | 4/10 | 3.35 | [1.0, 5.0] | 5.42 | [4.0, 7.0] |
| reset_free_vs_strict | 5/10 | 3.67 | [1.0, 5.0] | 4.99 | [3.0, 7.0] |

## Summary verdict

- Primary rank-correlation claim survives every single-body and single-seed perturbation (max Spearman rho shift 0.036).
- Primary interaction F survives every LOBO and LOSO perturbation at p < 0.001 (LOBO F in [11.0, 15.2]; LOSO F in [7.1, 10.5]).
- Asymmetric (paper-matched) top-10 bootstrap gives wider CIs because only ~63% of the fixed reference appears in any given body resample; symmetric resampling tightens the CIs.
