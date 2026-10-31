# A.2 Geometric Proxy Report (native MuJoCo)

Successful morphs: 100 / 100
Target: R_obs = 1 - fall_rate_rf (eval_tau020, seed-averaged)
Features-only LOO R^2 baseline: 0.5266

## Proxy correlations (Spearman rho with R_obs)

| proxy | mean | rho | p | lift over features |
|---|---|---|---|---|
| P1_passive_final | 0.385 | -0.1006 | 3.19e-01 | -0.0023 |
| P2_passive_max | 0.668 | -0.0072 | 9.43e-01 | -0.0011 |
| P3_random_final | 0.385 | +0.4916 | 2.06e-07 | +0.0334 |
| P4_random_max | 0.865 | +0.1344 | 1.82e-01 | +0.0045 |
| P5_settle_success | 0.145 | +0.1342 | 1.83e-01 | +0.0485 |

## Combined regressions

- Features only: LOO R^2 = 0.5266
- Features + all proxies: LOO R^2 = 0.6836 (lift +0.1570)
- Proxies only: LOO R^2 = 0.4909

## Interpretation
- **Proxies add information**: total lift = +0.157 (exploratory).
