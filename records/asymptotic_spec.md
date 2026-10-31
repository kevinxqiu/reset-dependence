# Asymptotic 50M Path B Pre-Registration (FROZEN 2026-04-18 20:20 local)

Frozen after design review (2026-04-18) of the prior draft
`asymptotic_spec_draft.md`. This is the binding spec for the 18-body
Path B asymptotic experiment; any deviation after commit forces a drop
of the component.

## Planned hypothesis (Path B, no cross-stack pooling)

Test whether, on the current code stack (post-corpus drift), the 5M
family-by-protocol interaction is recovered on the 18-body panel and
remains present at 50M on the same bodies and same code. **This study makes no numeric
pooling or comparability claim between corpus-5M and current-50M.**

## Motivation

The main-paper corpus runs at 5M steps. A critic may argue the
interaction is an undertraining artifact. The 5M-vs-asymptotic test on
the same code stack is the direct response to that concern. Eq-A
(2026-04-15) established that current code does not exactly reproduce
corpus-5M per-body numbers (Spearman rho=0.839 on Delta_reset, required
>=0.90 per D6 gate); we therefore compare 5M-current and 50M-current
on the same bodies and same code, not against the corpus.

## Panel (frozen per records/asymptotic_panel_bodies.json)

- 18 UNIMAL bodies: Floor=7, VT=6, MVT=5
- Originally selected 2026-04-15 for Phase 1b, stratified on
  (Delta_reset, baseline_psr_tau000) to address the corpus
  rho(Delta_reset, baseline_psr) = -0.74 confound.
- taus: {0.00, 0.20, 0.50}; seeds: {42, 43, 44}
- Two step budgets: 5M (current-code baseline) and 50M (asymptotic)
- **50M runs are from scratch, not resumed from 5M. Matched seed IDs
  are used only for paired stability analysis, not for training-state
  continuation.**
- Total cells: 18 * 3 * 3 * 2 = 324 cells

## Training (same hyperparameters as corpus)

- MJX PPO: `--num_envs 128 --unroll_length 16 --n_epochs 10
  --minibatch_size 256`
- Standard UNIMAL locomotion-v0 (same reward, same observation as main
  corpus)
- Height-based termination (not contact-duration)
- Code stack: current HEAD (train_mjx.py 489 lines beyond commit
  a7c2783). Drift documented in `records/eqa_failure_report.md`.

## QC gate (frozen, applied before inclusion decision)

- Cell valid if: results.json exists, no NaN in any PSR field, final
  eval PSR is finite.
- Include gate: >= 292/324 cells valid (>= 90%), matching main-corpus
  QC density.
- Fail gate means drop entire component (no partial inclusion).

## Primary analyses (frozen)

1. **5M-current reproducibility**: per-seed 2-way ANOVA
   `psr ~ C(protocol) * C(family) + C(body)` on 18-body 5M-current data
   (162 rows expected). Report F_inter, p_inter, direction of family
   ratios (Floor, VT, MVT RF/Epi and Str/Epi).

2. **50M-current asymptotic**: same model on 18-body 50M-current data.

3. **Within-budget ranking**: per-body Delta_reset = PSR(episodic) minus
   PSR(reset_free), Spearman rho between 5M-current Delta_reset and
   50M-current Delta_reset on the 18 bodies.

4. **Family direction preservation**: count of direction checks
   (4 total) that match the corpus reversal pattern:
   - Floor RF/Epi < VT RF/Epi at 5M-current
   - Floor RF/Epi < VT RF/Epi at 50M-current (ANCHOR, required)
   - VT Str/Epi < MVT Str/Epi at 5M-current
   - VT Str/Epi < MVT Str/Epi at 50M-current

5. **Backup dependence-aware tests**: cluster-robust Wald on the
   interaction coefficients with body as cluster, and seed-averaged
   2-way ANOVA (54 rows), both at 50M. Report both alongside per-seed.

## Inclusion gates (frozen; revised per design review 2026-04-18)

Applied after analysis. No post-hoc tweaking.

- **Body-text promotion** (Results paragraph + abstract mention):
  - 5M-current interaction p < 0.10 with correct direction
  - 50M-current interaction p < 0.025
  - Spearman rho(5M Delta, 50M Delta) >= 0.70
  - 4/4 family directions preserved

- **Main inclusion** (short Results paragraph + appendix):
  - 5M-current interaction p < 0.10 with correct direction
  - 50M-current interaction p < 0.05
  - Spearman rho(5M Delta, 50M Delta) >= 0.60
  - 3/4 family directions preserved, AND Floor<VT RF/Epi at 50M
    (the anchor contrast) must be preserved (no reversal)

- **Appendix-only** (appendix subsection, narrow scope):
  - 50M-current interaction p < 0.10
  - Spearman rho(5M Delta, 50M Delta) >= 0.50
  - Floor<VT RF/Epi at 50M preserved (anchor, required)

- **Drop**: any other outcome. Limitations note: "We piloted an 18-body
  50M asymptotic study on the current code stack; the family
  interaction did not meet the pre-registered strength threshold at
  asymptote." Deposit JSON in supplement.

## Abort rule on 5M-current phase

If 5M-current comes back with:
- wrong family direction (Floor RF/Epi >= VT RF/Epi), OR
- interaction p >= 0.10, OR
- anchor reversal (Floor RF/Epi >= VT RF/Epi at 5M),

then abort the 50M extension; do not launch. Disclose the 5M-current
results as a pilot in Limitations and do not pursue asymptotic claim.

## Pre-registration note

This spec is a prospective analysis plan frozen before any Path B
outcomes, after a separate alt-termination pilot (different bodies,
different termination rule). The drift acknowledged in Eq-A is
pre-existing and disclosed. No 50M training data exist at spec freeze
time beyond the 4-run canary in `results/asymptotic_50m/` (which was
truncated at 2M steps and is not used in analysis).

## Compute

Per-run at 4-pack on TITAN V / 2080 Ti, SPS ~776 from canary evidence:
- 5M: ~1.8h per run
- 50M: ~18h per run

Panel compute (1 run per SLURM job to stay within 36h limit):
- 5M-current: 162 runs * 1.8h / 36 parallel slots = ~8h wall-clock
- 50M-current: 162 runs * 18h / 36 parallel slots = ~81h wall-clock
  (~3.4 days)
- Total GPU-hours: 291 + 2916 = 3207 GPU-hours
- No checkpoint/resume needed since each run fits in 36h

## SLURM strategy

Per design review: existing `slurm/asymptotic_panel_50m.sh` packs 9 runs
per body task (~54h at 4-pack), which exceeds the 36h job limit and
relies on checkpoint-resume that is not implemented. Restructure to
1 run per SLURM job:
- 162 array tasks (one per body * tau * seed cell)
- Each task runs 1 training at 50M, ~18h wall-clock
- Max concurrent via QOS; drain naturally via SLURM scheduler

A 1-run canary at 50M must pass (no NaN, reasonable PSR, completes in
time) before the full 162-run launch.

## Timeline

- 2026-04-18 ~22:00: submit 1-run 50M canary (queues behind alt-term)
- 2026-04-19 ~10:00: alt-term panel finishes (expected)
- 2026-04-19 ~10:00-16:00: canary completes; verify
- 2026-04-19 ~16:00: launch full 5M-current panel (162 runs, ~8h)
- 2026-04-20 ~00:00: 5M-current done; apply abort gate
- 2026-04-20 ~01:00: if abort gate passes, launch 50M-current panel
  (162 runs, ~3.4 days)
- 2026-04-23 ~11:00: 50M-current done
- 2026-04-23 to 2026-04-25: analyze + integrate per gate outcome
- 2026-04-25 to 2026-04-28: adversarial design review + fixes
- 2026-04-28 to 2026-05-04: polish, supplement
- 2026-05-05: submission ready
- 2026-05-06 23:59 AoE: submit

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 5M-current wrong direction | Abort gate; drop 50M, disclose pilot |
| 50M training diverges (NaN) | Cell marked invalid; QC gate handles |
| SLURM queue congestion | Launch early (~22:00 tonight), use fair-share |
| Canary fails | Do not launch full panel, revisit |
| Soft result (rho in [0.50, 0.60)) | Pre-reg drops to appendix-only |
| Drift worse at 50M | Already priced in: we do not pool with corpus |

## Files

- `records/asymptotic_spec.md` (this file)
- `records/asymptotic_panel_bodies.json` (panel, frozen 2026-04-15)
- `slurm/asymptotic_panel_50m_canary.sh` (to be revised for 1-run)
- `slurm/asymptotic_panel_50m.sh` (to be revised for 1-run-per-job)
- `slurm/asymptotic_panel_5m.sh` (new, for 5M-current)
- `scripts/analyze_asymptotic.py` (new, gate applier)

## Commit policy

Spec commits before any asymptotic training is submitted. Any change
after commit is a deviation; the entire component drops.
