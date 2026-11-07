# NEMS Held-Out Robust Selector: DROP Decision (2026-04-23)

## Outcome

**DROP** per pre-registered QC gate 5 (`records/nems_robust_selector_spec.md`).

## Numbers

- 5-fold corpus CV R^2 for the worst-case-PSR predictor: **0.060**
- Pre-registered QC threshold: R^2 >= 0.20
- Gate decision: R^2 below threshold, predictor not interpretably informative;
  DROP without evaluating the downstream retention gate.

Supplementary numbers (computed but not used in the verdict, since QC gate 5
fired first and the downstream retention gate was not evaluated):

- Worst-case predictor best alpha: 10.0 (ridge)
- Epi predictor CV R^2: -0.044 (also below threshold)
- Point estimate retention at k=10 (informational only, not promoted):
  WC-selected NEMS-strict retention 0.498, Epi-selected 0.405.
- Bootstrap CIs at k=10 are computed in the analysis JSON but return NaN
  because one NEMS body (gen-0388) has NaN strict PSR from divergent
  training (already disclosed in NEMS appendix); resamples that include it
  propagate NaN. This does not affect the verdict, which was set by the
  upstream CV-R^2 gate.

## Interpretation

The five-feature structural model (initial height, first-level limbs, tree
depth, joints, bodies) is fit on corpus worst-case PSR. It explains only
6% of corpus variance at 5-fold CV, so it cannot support a zero-shot
selector claim on NEMS. This is different from the M2 predictor used in
`sec:nems_result`, which targets Delta_str-rf (the protocol gap) at in-sample
R^2 = 0.46 and zero-shot NEMS transfer rho = 0.41. Predicting a robustness
floor across three protocols is harder than predicting a two-protocol
delta.

## Action

- The selector was never scored, so the paper does not report this test.
- Analysis JSON deposited at `records/nems_robust_selector_analysis.json`.
- This decision log deposited at the current path for auditability.
- No follow-up experiment planned. A feature-based selector that generalises
  across morphology spaces remains an open direction.

## Background

A design review on 2026-04-23 proposed this test as the one addition before
the submission, with a hard preregistered gate. The gate stopped the test at
its quality check, as specified.
