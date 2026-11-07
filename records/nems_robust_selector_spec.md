# NEMS Held-Out Robust Selector: Pre-Registered External-Validity Test

FROZEN 2026-04-23 (before any analysis). Do not edit after first run of
`scripts/analyze_nems_robust_selector.py`.

## Purpose

The paper's Section 4.3 claim is that a worst-case-PSR selector on the 100-body
corpus recovers >= 91% of every single-protocol oracle at k in {5, 10, 20}
(see `scripts/compute_robust_selection.py` and `records/protocol_robust_selection.json`).
An anticipated objection: this is an in-sample demonstration. Does the selector
generalise to morphologies outside the corpus?

This spec pre-registers a zero-shot test on the 60-body NEMS panel
(pre-DERL-selection UNIMAL morphologies). Freeze BEFORE touching NEMS
outcome data.

## Rationale and relation to prior work

- LOPO analysis (`records/protocol_robust_lopo_spec.md`, ran
  2026-04-19) DROPPED per its gate (tests cross-protocol generalisation, not
  cross-morphology); the paper reports it in Section 4.3 and Appendix E. This NEMS test is a different
  question: held-out MORPHOLOGIES, not held-out PROTOCOLS.
- The NEMS structural-predictor M2 transfer (`sec:nems_result`) shows feature
  prediction of Delta_str-rf transfers at rho = 0.41. That is NOT the same as
  predicting worst-case PSR. M2 predicts the sign-flip-relevant delta, not the
  robustness floor. A new predictor must be fit.

## Data

- Corpus: 100 UNIMAL bodies x 3 tau x 3 seeds = 900 runs.
  Source: `results/asymptotic_5m/*/results.json` per body and
  `results/mjx_ppo_0pXX_seedYY/results.json` per seed, the same data
  underpinning Section 4.3.
- NEMS: 60 pre-DERL UNIMAL bodies x 3 tau x 3 seeds = 540 runs.
  Source: `results/nems_panel/`.
- Per-body per-protocol score: seed-mean PSR at tau_eval = 0.20
  (same convention as Table 1 and compute_robust_selection.py).
- Morphology features (input to predictor): the existing 5-feature set used
  in M2 analysis -- initial height, first-level limbs, tree depth, joints,
  bodies -- frozen at `results/morph_features.json` (corpus) and
  `results/morph_features_nems.json` (NEMS).

## Predictor

A ridge regression from the 5 structural features to corpus worst-case PSR.

    target_i = min over p in {epi, strict, rf} of seed-mean PSR of body i under tau_train = p at tau_eval = 0.20

    X: 100 rows x 5 features (standardised using corpus train-fold means and stds)
    y: 100 rows, worst-case PSR per body

    Model: sklearn Ridge(alpha=1.0). Any alpha in {0.1, 1, 10} may be used
    PROVIDED the choice is made via 5-fold cross-validation on the corpus only,
    NEVER on NEMS. Log the selected alpha in the analysis output.

Applied zero-shot to NEMS:

    For each NEMS body j, compute predicted_score_j = predictor(features_j).
    Rank NEMS by predicted_score_j (larger = more robust).
    Selected set = top k = 10 NEMS bodies by predicted_score.

## Evaluation

Primary metric: zero-shot strict retention at k = 10.

    retention = mean over selected NEMS bodies of NEMS-measured PSR_strict
              / mean over NEMS strict-oracle top-10 of NEMS-measured PSR_strict

    Oracle = top-10 NEMS bodies by their own measured NEMS PSR_strict.

Secondary metric: delta vs episodic baseline at k = 10.

    delta = mean PSR_strict of selected NEMS top-10 by predicted_worst_case_PSR
          - mean PSR_strict of selected NEMS top-10 by predicted_PSR_epi

    Uses a second corpus-fitted ridge predictor (same features) with target
    y = PSR_epi (instead of worst-case PSR). Apply zero-shot to NEMS features.
    Rank NEMS by predicted_PSR_epi, take top-10, evaluate strict retention.

## Confidence intervals

Bootstrap with 10,000 resamples at body level. Resample NEMS bodies with
replacement, refit ranking from the frozen predictor on the resampled
indices, recompute retention and delta. Report 95% percentile CI per metric.

## Pre-committed gates (decision BEFORE seeing results)

| Outcome | Criterion |
|---|---|
| MAIN-TEXT | retention CI lower > 0.85 AND delta-vs-epi CI lower > 0.05 |
| APPENDIX | (retention CI lower > 0.70 AND retention point estimate > 0.85) OR (delta-vs-epi CI crosses zero but point estimate > 0.05 with retention point estimate > 0.85) |
| DROP | neither condition |

The APPENDIX gate requires weaker evidence than the MAIN-TEXT gate.

A DROP outcome ends the test, and the analysis output is deposited in the
records.

> Note added for the release: this test stopped at its quality gate
> (cross-validated R^2 = 0.06), so no selector was scored; the paper does not
> report it.

## QC gates (any failure forces DROP)

1. All corpus features and NEMS features available for all bodies. Any
   missing feature forces DROP.
2. Corpus PSR available for all 100 x 3 tau cells at the seed-mean level.
3. NEMS PSR available for all 60 x 3 tau cells at the seed-mean level.
4. No corpus/NEMS body identity overlap (verify UNIMAL IDs disjoint).
5. Predictor R^2 on 5-fold corpus CV >= 0.20. Below this the predictor is
   too weak to be informative and the zero-shot comparison is not
   interpretable.
6. No post-hoc k shopping. k = 10 is the only primary reported. k = 5 and
   k = 20 optional secondaries, NOT promoted even if they look better.

## Guardrails

- The predictor is fit BEFORE touching any NEMS outcome.
- Alpha selected via corpus-only 5-fold CV.
- No feature engineering beyond the 5 features locked in M2 analysis.
- Gate decision is pre-committed and INFORMATIONAL only: this file declares
  it, `scripts/analyze_nems_robust_selector.py` reports which bucket the
  result lands in, and the paper is only edited after the gate is applied.
- If the result lands in DROP, the analysis JSON is still deposited at
  `records/nems_robust_selector_analysis.json` for auditability.

## Risks and mitigation

1. Predictor R^2 on corpus is weak (< 0.20). Would force DROP. Mitigation:
   this IS the gate. If features cannot predict worst-case PSR on corpus,
   the zero-shot extension cannot be interpreted.
2. NEMS strict-oracle mean is close to episodic-oracle mean (small dynamic
   range). Would compress retention ratios. Mitigation: inherent to NEMS
   data; accept. Gate thresholds are set knowing this risk.
3. Bootstrap CI is wide because n = 60 NEMS bodies. Mitigation: accept.
   Gate requires CI lower bound, not point estimate, to clear.

## Script

`scripts/analyze_nems_robust_selector.py` to be written fresh. Must:

- Load corpus per-body per-protocol seed-mean PSR at tau_eval = 0.20.
- Load NEMS per-body per-protocol seed-mean PSR at tau_eval = 0.20.
- Load both morphology feature sets.
- Verify no body-ID overlap (QC 4).
- Fit two Ridge predictors on corpus: y1 = worst-case PSR, y2 = PSR_epi.
- Apply zero-shot to NEMS features. Rank and select top-10 for each.
- Compute retention and delta, with bootstrap CIs.
- Write `records/nems_robust_selector_analysis.json` with all
  numbers, including the gate verdict.
- Exit code 0 regardless of verdict.
