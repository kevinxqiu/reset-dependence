# Equivalence Pilot Report (Eq-A: JAX on the MJX venv vs corpus)

**Verdict:** FAIL

- Complete bodies: 12 / 12

## D6 per-tau metrics (matched eval key)

| tau | n | eval | rho_PSR | rel_drift | slope | b0 | rho_pass | drift_pass | slope_pass | b0_pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 0p00 | 12 | eval_neutral | 0.965 | 0.086 | 0.932 | 0.090 | True | True | True | False |
| 0p20 | 12 | eval_tau020 | 0.881 | 0.083 | 0.817 | 0.645 | True | True | False | False |
| 0p50 | 12 | eval_tau050 | 0.734 | 0.142 | 0.946 | 0.086 | False | False | True | False |

## Delta_reset (eval_tau020 for both train taus)

- n = 12
- eval_key = eval_tau020
- Spearman rho = 0.839 (needs > 0.9): pass=False
- Sign agreement = 0.83 (needs >= 0.9): pass=False

## Failures
- tau=0p00: |intercept|=0.090 >= 0.05
- tau=0p20: slope=0.817 outside [0.9, 1.1]
- tau=0p20: |intercept|=0.645 >= 0.05
- tau=0p50: Spearman rho_PSR=0.734 <= 0.85
- tau=0p50: median_rel_drift=0.142 >= 0.10
- tau=0p50: |intercept|=0.086 >= 0.05
- Delta_reset Spearman rho=0.839 <= 0.9
- Delta_reset sign agreement=0.83 < 0.9