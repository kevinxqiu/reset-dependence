# A.1 Verification Report

## Claim under test
> the design review's claim: Spearman rho between per-morph reset-free fall rate and RF/Epi per-step reward ratio is approximately -0.82.

## Metric definitions (corrected from plan v1)

- `fall_rate_rf(m)`: tau=0.00 policy, eval under eval_tau020, fall_rate
- `mean_ep_len_rf(m)`: tau=0.00 policy, eval under eval_tau020, mean_ep_len
- `psr_epi(m)`: tau=0.20 policy, eval under eval_tau020, per_step_reward
- `psr_rf(m)`: tau=0.00 policy, eval under eval_tau020, per_step_reward
- `rf_epi_ratio(m) = psr_rf(m) / psr_epi(m)` (matches `extract_morph_features.py`)

## Results

### fall_rate_rf vs rf_epi_ratio (the design review's claim)

| scope | n | rho | 95% CI | p |
|---|---|---|---|---|
| overall | 100 | -0.8219 | [-0.895, -0.710] | 1.08e-25 |
| floor | 41 | -0.9069 | [-0.937, -0.832] | 3.18e-16 |
| vt | 37 | -0.6297 | [-0.838, -0.308] | 2.97e-05 |
| mvt | 22 | -0.7510 | [-0.930, -0.355] | 5.63e-05 |

### mean_ep_len_rf vs rf_epi_ratio (alternative observable)

| scope | n | rho | 95% CI | p |
|---|---|---|---|---|
| overall | 100 | 0.8126 | [0.701, 0.880] | 1.05e-24 |
| floor | 41 | 0.8659 | [0.753, 0.908] | 2.69e-13 |
| vt | 37 | 0.6249 | [0.294, 0.846] | 3.56e-05 |
| mvt | 22 | 0.7190 | [0.348, 0.881] | 1.63e-04 |

### Self-consistency checks

- fall_rate_rf vs mean_ep_len_rf: rho=-0.9904 CI=[-0.996, -0.978]
- fall_rate_rf vs absolute psr_rf: rho=-0.7868 CI=[-0.868, -0.672]

## Interpretation

- **design review claim confirmed**: |rho| >= 0.7 on the primary comparison.

- Within-family correlations are non-trivial: signal is not purely family-label-driven.
