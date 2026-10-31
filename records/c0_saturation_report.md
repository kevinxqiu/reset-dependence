# C.0.1 Saturation-Aware Reanalysis Report

- n morphs: 100
- R_obs = 1 saturated: **38 / 100**
- R_mel = 1 saturated: 38 / 100

## Alternative targets and feature predictability

| target | full n=100 LOO R^2 | n=62 non-sat LOO R^2 |
|---|---|---|
| R_obs (1 - fall_rate) | 0.5266 | 0.1426 |
| R_mel (mean_ep_len / 1000) | 0.4807 | 0.1094 |

## Composite (saturation-aware) targets

| target | full n=100 LOO R^2 |
|---|---|
| R_composite (R_obs with psr_epi tiebreaker) | 0.5283 |
| R_composite_rank (normalized rank) | 0.5858 |

## Per-family R_obs

| family | n | saturated | LOO R^2 |
|---|---|---|---|
| floor | 41 | 8 | 0.4622 |
| vt | 37 | 17 | 0.4442 |
| mvt | 22 | 13 | 0.4822 |

## Target agreement

| pair | Spearman | Pearson |
|---|---|---|
| R_obs vs R_mel | 0.9905 | 0.9863 |
| R_obs vs R_composite_rank | 0.9720 | 0.9313 |
| R_mel vs R_composite_rank | 0.9627 | 0.9142 |

## Interpretation
- R_obs is predictable at R^2 = 0.527 on the full 100 bodies and 0.143 on the non-saturated 62 subset.
- Non-trivial gap between full and non-saturated predictability. Ceiling-tied morphs are contributing to feature-to-R_obs signal.
