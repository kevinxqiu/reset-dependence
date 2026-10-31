# C.0.2 Metric-Robustness Report

## Alternative observables

| name | definition | mean | std | saturated |
|---|---|---|---|---|
| R_obs_frac | 1 - fall_rate (binary per-episode, averaged 20 eps) | 0.595 | 0.433 | 38 |
| R_obs_len | mean_ep_len / 1000 (continuous upright time) | 0.674 | 0.376 | 38 |
| R_obs_psr_absolute | per_step_reward absolute value | 1.795 | 1.158 | 1 |
| R_obs_psr_ratio | per_step_reward / psr_epi (paper's RF/Epi) | 0.694 | 0.385 | 1 |
| R_obs_fitness_absolute | total fitness (sum reward) absolute | 1640.344 | 1244.471 | 1 |
| R_obs_distance_absolute | total forward distance absolute | 32.927 | 24.948 | 1 |

## Pairwise Spearman rho

| a | b | rho |
|---|---|---|
| R_obs_len | R_obs_frac | +0.9905 |
| R_obs_psr_absolute | R_obs_frac | +0.7866 |
| R_obs_psr_absolute | R_obs_len | +0.7811 |
| R_obs_psr_ratio | R_obs_frac | +0.8225 |
| R_obs_psr_ratio | R_obs_len | +0.8126 |
| R_obs_psr_ratio | R_obs_psr_absolute | +0.8075 |
| R_obs_fitness_absolute | R_obs_frac | +0.8301 |
| R_obs_fitness_absolute | R_obs_len | +0.8292 |
| R_obs_fitness_absolute | R_obs_psr_absolute | +0.9867 |
| R_obs_fitness_absolute | R_obs_psr_ratio | +0.8241 |
| R_obs_distance_absolute | R_obs_frac | +0.8303 |
| R_obs_distance_absolute | R_obs_len | +0.8293 |
| R_obs_distance_absolute | R_obs_psr_absolute | +0.9864 |
| R_obs_distance_absolute | R_obs_psr_ratio | +0.8244 |
| R_obs_distance_absolute | R_obs_fitness_absolute | +0.9999 |

## Top-k overlap

### top-10

| a | b | overlap |
|---|---|---|
| R_obs_frac | R_obs_len | 4/10 |
| R_obs_frac | R_obs_psr_absolute | 2/10 |
| R_obs_frac | R_obs_psr_ratio | 2/10 |
| R_obs_frac | R_obs_fitness_absolute | 2/10 |
| R_obs_frac | R_obs_distance_absolute | 2/10 |
| R_obs_len | R_obs_psr_absolute | 1/10 |
| R_obs_len | R_obs_psr_ratio | 2/10 |
| R_obs_len | R_obs_fitness_absolute | 1/10 |
| R_obs_len | R_obs_distance_absolute | 1/10 |
| R_obs_psr_absolute | R_obs_psr_ratio | 2/10 |
| R_obs_psr_absolute | R_obs_fitness_absolute | 10/10 |
| R_obs_psr_absolute | R_obs_distance_absolute | 10/10 |
| R_obs_psr_ratio | R_obs_fitness_absolute | 2/10 |
| R_obs_psr_ratio | R_obs_distance_absolute | 2/10 |
| R_obs_fitness_absolute | R_obs_distance_absolute | 10/10 |

### top-20

| a | b | overlap |
|---|---|---|
| R_obs_frac | R_obs_len | 15/20 |
| R_obs_frac | R_obs_psr_absolute | 8/20 |
| R_obs_frac | R_obs_psr_ratio | 4/20 |
| R_obs_frac | R_obs_fitness_absolute | 8/20 |
| R_obs_frac | R_obs_distance_absolute | 8/20 |
| R_obs_len | R_obs_psr_absolute | 7/20 |
| R_obs_len | R_obs_psr_ratio | 6/20 |
| R_obs_len | R_obs_fitness_absolute | 7/20 |
| R_obs_len | R_obs_distance_absolute | 7/20 |
| R_obs_psr_absolute | R_obs_psr_ratio | 7/20 |
| R_obs_psr_absolute | R_obs_fitness_absolute | 20/20 |
| R_obs_psr_absolute | R_obs_distance_absolute | 20/20 |
| R_obs_psr_ratio | R_obs_fitness_absolute | 7/20 |
| R_obs_psr_ratio | R_obs_distance_absolute | 7/20 |
| R_obs_fitness_absolute | R_obs_distance_absolute | 20/20 |

### top-30

| a | b | overlap |
|---|---|---|
| R_obs_frac | R_obs_len | 26/30 |
| R_obs_frac | R_obs_psr_absolute | 20/30 |
| R_obs_frac | R_obs_psr_ratio | 15/30 |
| R_obs_frac | R_obs_fitness_absolute | 20/30 |
| R_obs_frac | R_obs_distance_absolute | 20/30 |
| R_obs_len | R_obs_psr_absolute | 18/30 |
| R_obs_len | R_obs_psr_ratio | 17/30 |
| R_obs_len | R_obs_fitness_absolute | 19/30 |
| R_obs_len | R_obs_distance_absolute | 19/30 |
| R_obs_psr_absolute | R_obs_psr_ratio | 17/30 |
| R_obs_psr_absolute | R_obs_fitness_absolute | 29/30 |
| R_obs_psr_absolute | R_obs_distance_absolute | 29/30 |
| R_obs_psr_ratio | R_obs_fitness_absolute | 17/30 |
| R_obs_psr_ratio | R_obs_distance_absolute | 17/30 |
| R_obs_fitness_absolute | R_obs_distance_absolute | 30/30 |

## Verdict

- min pairwise rho: 0.7811
- mean pairwise |rho|: 0.8627
- **All pairs |rho| >= 0.7**: every pair of observables correlates at |rho| >= 0.7, the check this script applies. The stricter gate in its docstring (rho >= 0.85 for every pair and top-k stability >= 70 percent) is not met.
