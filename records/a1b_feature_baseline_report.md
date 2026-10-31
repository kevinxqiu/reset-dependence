# A.1b Feature Baseline Report

## Targets
- `R_obs = 1 - fall_rate_rf(m)`: direct observable of body-level recoverability
- `rf_epi_ratio(m)`: the paper's existing ratio

## Leave-one-out ridge results

| target | design | best alpha | LOO R^2 | Spearman rho | MAE |
|---|---|---|---|---|---|
| R_obs | features_only | 3.0 | 0.5266 | 0.7891 | 0.2290 |
| R_obs | features_plus_family | 3.0 | 0.5669 | 0.8111 | 0.2249 |
| R_obs | family_only | n/a | 0.1059 | 0.0282 | 0.0000 |
| rf_epi_ratio | features_only | 3.0 | 0.3939 | 0.5871 | 0.2245 |
| rf_epi_ratio | features_plus_family | 10.0 | 0.4190 | 0.6066 | 0.2278 |
| rf_epi_ratio | family_only | n/a | 0.0611 | -0.0023 | 0.0000 |

## the design review's claim
> design review: leave-one-out ridge on 8 features gives R^2 ~ 0.39 without family, ~ 0.42 with family.

- R_obs features_only LOO R^2 = **0.5266**
- R_obs features_plus_family LOO R^2 = **0.5669**

## Interpretation
- Baseline established. Any geometric proxy in A.2 must beat LOO R^2 = 0.527 on features alone (or the ~ +0.03 lift from adding family information) to be worth pursuing.

## Feature importance (R_obs target, standardized ridge coefficients)

| feature | standardized coef |
|---|---|
| total_volume | +0.2475 |
| tree_depth | -0.1585 |
| init_height | -0.1455 |
| n_geoms | -0.0523 |
| n_bodies | -0.0523 |
| n_joints | -0.0383 |
| n_motors | -0.0383 |
| n_first_level_limbs | +0.0307 |

(See `paper/figures/pivot/fig_a1b_feature_importance.pdf` for the full chart including family dummies.)
