# Pivot C.1: Delta_reset Baseline Controls

Analysis run on 100 corpus bodies with complete (tau=0, tau=0.2) x 3-seed coverage on eval_tau020.

## Question

Is `Delta_reset(m) = PSR_train_tau020(eval_tau020) - PSR_train_tau000(eval_tau020)` a morphology-intrinsic property, or is it a function of baseline no-reset skill (PSR_train_tau000) and/or body size (init_height)?

## Results

### Control 1: Delta_reset vs baseline PSR_tau000
- Spearman rho = -0.742 (p = 9.93e-19)
- Pearson r = -0.785
- n = 100
- **Interpretation:** Delta_reset is strongly negatively correlated with baseline skill. Bodies that cannot learn without resets gain most from reset training.

### Control 2: Delta_reset vs init_height (body size)
- Spearman rho = 0.550 (p = 3.14e-09)
- Pearson r = 0.610
- n = 100

### Combined regression: Delta_reset ~ baseline + height
- R^2 = 0.676
- Residual std / raw Delta_reset std = 0.57

## Residual Delta_reset by family

After subtracting the fitted baseline + height linear model, does the residual Delta_reset still show morphology-family structure?

| Family | n | Raw Delta_reset mean +/- std | Residual mean +/- std |
|---|---|---|---|
| floor | 41 | +0.975 +/- 0.805 | +0.152 +/- 0.453 |
| vt | 37 | +0.305 +/- 0.620 | -0.218 +/- 0.396 |
| mvt | 22 | +0.563 +/- 0.802 | +0.084 +/- 0.406 |

- Kruskal-Wallis H = 13.01, **p = 0.0015**
- One-way ANOVA F = 7.79, **p = 0.0007**

- **Interpretation:** The residual Delta_reset after removing baseline + height still differs by family (Floor residuals positive, VT negative). Because PSR_RF enters both the baseline control and Delta_reset, this control is circular and does not establish independence from baseline skill.

## Status

Historical exploratory control, superseded because PSR_RF appears on both sides of the regression. The camera-ready paper does not use these statistics.
