# SAC Full 3-tau 3-seed Panel: Pre-Registered Expansion of the Cross-Algorithm Pilot

FROZEN 2026-04-24 (before any new training). Do not edit after first new
SLURM launch. Combines T1 (seed bump 2 -> 3) and T4 (tau bump 2 -> 3) from
the 2026-04-24 design review, run as a single panel.

## Motivation

The current SAC appendix (Section K, results/sac_mjx_0pXX_seedYY/) is a
24-body x 2-tau x 2-seed pilot (n=96) framed as a per-body Str/RF
sensitivity test, NOT a main-text replication. Headline: per-body Str/RF
Spearman rho vs PPO = 0.78 (p = 1.3e-5, n = 23); blocked ANOVA interaction
F(2, 69) = 5.42, p = 0.007 replicating the corpus direction.

The pilot has two gaps:

- tau = 0.20 is missing, so we cannot report RF/Epi or Str/Epi ratios under
  SAC, only Str/RF sensitivity.
- Two seeds give limited precision even for a pilot.

This expansion fills the full 3-tau x 3-seed matrix (216 runs total, 120
new runs) so SAC can be reported either as a main-text cross-algorithm
replication or, if the interaction does not hold across 3 tau, as a
pilot with tighter scoping.

## Design

- Same 24 morphologies as the existing pilot (`scripts/mechanism_morphologies.txt`).
- Protocols: tau in {0.00, 0.20, 0.50}. tau = 0.20 is NEW.
- Seeds: {42, 43, 44}. Seed 44 is NEW.
- Budget: 5M environment steps per cell (identical to PPO corpus).
- Hyperparameters: identical to existing pilot. Brax SAC implementation,
  same code path as `slurm/train_sac_mjx.sh`.
- Evaluation: same `eval_tau020` convention (tau_eval = 0.20, 20 episodes
  seed-averaged), matching the main-text convention.

New cells to run:

| tau   | seed 42 | seed 43 | seed 44 |
|-------|---------|---------|---------|
| 0.00  | exists  | exists  | NEW     |
| 0.20  | NEW     | NEW     | NEW     |
| 0.50  | exists  | exists  | NEW     |

5 new (tau, seed) pairs x 24 bodies = 120 new runs. Total panel 216.

## Hypotheses

**Primary (T4 cross-algorithm reversal replication).** The
protocol-by-family interaction observed in PPO (F > corpus threshold, p
< 1e-9) holds in SAC on the full 3-tau x 3-seed panel. Per-seed blocked
ANOVA:

    PSR_ij ~ tau_i * family_i + body_j + epsilon_ij

    H0: interaction F p-value >= 0.05.
    H1: interaction F p-value < 0.05.

**Secondary (T1 uncertainty tightening).** Adding seed 44 narrows the
pilot's per-seed body-to-body agreement CI without reversing sign.

    Rho-PPO-SAC-95CI_new subset Rho-PPO-SAC-95CI_pilot (at pilot point
    estimate rho = 0.78, CI should shrink or stay equivalent).

## Primary outcome

Per-seed blocked ANOVA interaction statistic on n = 216 (24 bodies x 3 tau
x 3 seeds). Report F, df, p. Secondary:

- Per-family Str/RF and RF/Epi and Str/Epi ratios with SE over 3 seeds.
- Per-body cross-algorithm Spearman rho between PPO Str/RF and SAC Str/RF
  on the same 24 bodies (with seed 44 included).
- Directional family ordering on all three pairwise protocol ratios.

## Gate (pre-committed BEFORE seeing expansion results)

All gate criteria evaluated on the SEED-AVERAGED per-body Str/RF ratios and
the per-seed blocked ANOVA on the combined 216-cell panel. Floor-below
counts use the seed-averaged PSR_epi per body (not any single cell), to
avoid one noisy seed forcing a DROP.

| Outcome | Criterion |
|---|---|
| MAIN-TEXT upgrade | Interaction p < 0.01 AND seed-averaged family-mean Str/RF satisfies the corpus ordering Floor > VT > MVT (all three strict). Move SAC from appendix to a one-paragraph main-text cross-algorithm replication block. |
| APPENDIX-STAY | Interaction 0.01 <= p < 0.05 OR the full Floor > VT > MVT ordering is violated (e.g., VT > Floor or MVT > VT). Update Section K with 3-tau numbers AND an explicit paragraph describing the expansion; numbers-only update is NOT sufficient. |
| DROP | Interaction p >= 0.05 OR seed-averaged PSR_epi < 0.05 for > 3 of 24 bodies (rf-floor exclusion). Report the pilot as before, and disclose the expansion and its null result in one Section K sentence. |

## Smoke test (required before launching the full tau = 0.20 block)

Before submitting the 72 tau = 0.20 runs, smoke-test 1 body at tau = 0.20
with seed 42 to confirm Brax SAC behaves at this unseen truncation level.
Pre-committed smoke criterion:

    Final eval_tau020 per-step reward > 0 (not all-fall collapse)
    AND Str/RF sensitivity at the smoke body is within 2x of the pilot
    value for the same body at tau = 0.

If the smoke body fails either check, pause the expansion and diagnose.
Smoke body: same convention as other smoke tests, index 0 in
`scripts/mechanism_morphologies.txt`, roughly 2 hours wall-clock.

## QC gates (any failure forces DROP)

1. All 120 new cells complete (no SLURM timeouts or exceptions). Failed
   cells are defined only by timeout, exception, missing result-entry,
   or non-finite eval_tau020.per_step_reward. Failed cells may be rerun
   up to 2 times with the SAME seed; no seed substitution.
2. At least 216 - 12 = 204 valid cells (allows up to 12 NaN/divergent
   cells across the entire panel, matching the existing SAC rf-floor
   disclosure convention).
3. No silent NaN in eval_tau020.per_step_reward for any valid cell.
4. Reliability gate: median per-body seed-SD at tau = 0.20 <= 2x the
   LARGER of (median per-body seed-SD at tau = 0.00, median per-body
   seed-SD at tau = 0.50). Computed across the full combined 216-cell
   panel (3 seeds per cell so per-body SD is meaningful). If median SD
   at either existing tau is below 0.05, use an absolute threshold of
   0.10.
5. Code-identity check: before launching seed-44 runs, record the git
   HEAD commit hash in the analysis JSON and verify that no training
   code file in `scripts/` or `slurm/` has been modified between the
   pilot runs' creation date (results/sac_mjx_* directories, ~2026-04-14)
   and the expansion launch. If any script has drifted, rerun the pilot
   cells at the current hash instead of launching seed 44 at a different
   hash.

## Guardrails

1. NO post-hoc tweaking of hyperparameters, body list, or evaluation
   convention.
2. Gate applied BEFORE any paper text changes. Analysis cache at
   `records/sac_full_panel_analysis.json` is the audit artefact.
3. If gate misses, original Section K text is preserved; the expansion is
   disclosed as "we ran the full 3-tau x 3-seed panel; interaction held /
   did not hold" in one sentence with pointer to supplement.
4. The per-body Str/RF Spearman rho vs PPO may NOT be reported with seed
   44 only; the headline rho remains over the 2-seed subset plus seed 44,
   to match the rest of the paper's treatment of cross-algorithm
   agreement.
5. We stop at 3 seeds; no opportunistic addition of seed 45.

## Compute

5 new (tau, seed) combos x 3 array tasks x 8 bodies per task = 15 SLURM
jobs. Common partition (`qos=<qos>`). Each job ~2 hr at
4-way packing on common-partition GPUs. Total ~30 GPU-hr, ~1 wall-clock
day if queue is light.

## Scripts and launch

- Submission template (existing): `slurm/train_sac_mjx.sh` with
  `--export=THRESHOLD=X,SEED=Y`.
- Analysis (new): `scripts/analyze_sac_full_panel.py` applying the gate
  table above and writing
  `records/sac_full_panel_analysis.json`.
- Integration trigger: only after gate is applied and artefact is
  deposited.

## Risks and mitigation

1. SAC at tau = 0.20 diverges differently from SAC at tau in {0, 0.5}.
   Mitigation: QC 4 requires seed-SD within 3x of existing tau. Fail
   closed if this trips.
2. The existing pilot's Floor rf-floor excluded 1 body; the expanded
   panel may exclude more. Mitigation: gate's "<= 12 NaN bodies" bound.
3. Queue congestion at common partition. Mitigation: no A100 partition
   use for SAC; all jobs go to common (confirmed 2026-04-24).
4. Gate lands APPENDIX-STAY: no paper-text change, but the expansion
   still completes the 3 x 3 matrix. Mitigation: accept.
