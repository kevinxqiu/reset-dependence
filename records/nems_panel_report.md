# NEMS (Non-Evolved Morphology Substrate): Results

Pre-registered analysis per `records/nems_task_spec.md`. Panel: 60 UNIMAL-init bodies (pre-DERL-selection).

- Bodies with full 9-cell coverage (all 3 seeds valid): 57
- Bodies with >= 1 NaN protocol-seed cell: 3 (total NaN cells: 4)
  - bodies: ['gen-0044', 'gen-0187', 'gen-0388']

## Primary endpoint: zero-shot structural transfer

- Corpus OLS predictor -> NEMS bodies (all n=60): Spearman rho = 0.412 [0.18, 0.60], Pearson r = 0.415 [0.24, 0.56].
- Positive-transfer threshold (rho > 0.3, 95% CI excludes 0): **MET**.

## Sensitivity: alternative corpus models (all NEMS bodies)

| model | Spearman rho (95% CI) | Pearson r (95% CI) |
| --- | --- | --- |
| ols | 0.412 [0.18, 0.60] | 0.415 [0.24, 0.56] |
| ridge | 0.357 [0.11, 0.56] | 0.375 [0.19, 0.53] |
| rf | 0.409 [0.16, 0.61] | 0.297 [0.08, 0.50] |

## Sensitivity: in-distribution subset only (corpus init_z range [0.30, 1.04])

In-distribution NEMS bodies: 43 / 60

| model | Spearman rho (95% CI) | Pearson r (95% CI) |
| --- | --- | --- |
| ols | 0.058 [-0.29, 0.38] | 0.092 [-0.23, 0.38] |
| ridge | -0.010 [-0.34, 0.32] | 0.042 [-0.27, 0.34] |
| rf | 0.139 [-0.20, 0.44] | 0.062 [-0.23, 0.34] |

## Calibration: within-corpus LOO-CV Spearman rho (upper bound for transfer)

OLS=0.653, Ridge=0.668, RF=0.777. A NEMS transfer rho close to this range indicates strong structural-feature transfer; materially lower indicates a distribution-shift penalty.

## Secondary: 3-cluster K-means on structural features

- Cluster sizes: {0: 24, 1: 14, 2: 22}

Cluster centroids (back-transformed features, pre-registered keys):

| cluster | num_limbs | torso_vertical | init_z | num_joints | tree_depth |
|---|---|---|---|---|---|
| C0 | 7.0 | 0.33 | 1.10 | 13.5 | 3.4 |
| C1 | 3.9 | 0.00 | 0.68 | 9.5 | 2.1 |
| C2 | 4.9 | 1.00 | 0.75 | 10.4 | 2.2 |

## Secondary: blocked protocol-by-cluster two-way ANOVA (with body blocking)

- Interaction F(4,470) = 12.47, p = 0.0000, partial eta^2 = 0.096
- Protocol main effect: F = 347.42, p = 0.0000
- Cluster main effect: F = 8.41, p = 0.0003

## Secondary: per-protocol one-way ANOVA across clusters (descriptive)

| protocol | F | p | H (Kruskal) | p_KW |
|---|---|---|---|---|
| episodic | 9.14 | 0.0002 | 14.48 | 0.0007 |
| strict | 11.06 | 0.0000 | 18.90 | 0.0001 |
| reset_free | 3.48 | 0.0330 | 3.75 | 0.1536 |

## Str-minus-RF paired delta by cluster

| cluster | n | mean Str-RF | sd |
|---|---|---|---|
| C0 | 24 | +1.583 | 0.952 |
| C1 | 14 | +1.208 | 0.550 |
| C2 | 22 | +0.751 | 0.993 |

## Distance transitions by cluster (anchors 1m/5m)

| cluster | total | epi_walks | rf_walks | str_walks | walks_epi->stop_rf | walks_epi->stop_str | stop_epi->walks_rf | stop_epi->walks_str |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 24 | 14 | 2 | 13 | 10 | 0 | 0 | 0 |
| C1 | 14 | 4 | 1 | 6 | 2 | 0 | 0 | 0 |
| C2 | 22 | 10 | 3 | 7 | 6 | 3 | 0 | 0 |
## Baseline-skill control

Regression `PSR_RF ~ PSR_epi + initial_height + num_limbs` on 60 NEMS bodies: R^2 = 0.091.

- Raw coefficients: Coef(PSR_epi) = 0.200; Coef(initial_height) = -0.616; Coef(num_limbs) = 0.046; intercept = 0.205.
- Standardized betas: PSR_epi = +0.258; initial_height = -0.327; num_limbs = +0.154.

Figure: `paper/figures/fig_nems.pdf`.