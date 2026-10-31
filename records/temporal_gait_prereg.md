# Pre-registration: Temporal Gait Figure Retrains

Frozen 2026-04-22 before any retraining. Locks body panel, seed, and protocol
grid so the figure cannot be retroactively curated.

## Purpose

design review recommended collapsing the static top-bodies figure
into a 3 bodies x 3 protocols temporal-gait figure, rendered in the
swim2real color-coded-time style. Corpus runs did not save policy
checkpoints, so we must retrain the selected bodies under identical
hyperparameters with `--save-checkpoints` before we can render anything
visually honest.

## Bodies (frozen)

Selected deterministically from the top-5-per-protocol results to span
the three family-protocol overlaps. No aesthetic criterion applied.

| Body                              | Family | Reason                                        |
|-----------------------------------|--------|-----------------------------------------------|
| floor-5506-10-6-01-15-48-35       | floor  | Rank 3 Episodic, Rank 2 Reset-free (top-3)    |
| mvt-5506-13-10-17-12-25-45        | mvt    | Rank 3 Reset-free, Rank 2 Strict (top-3)      |
| floor-1409-0-7-01-15-15-26        | floor  | Rank 1 Strict, Rank 4 Reset-free (top-5)      |

Two of the three are in the per-protocol top-3 under every protocol they
appear in. The third (`floor-1409-0-7`) is Strict's top-1 but sits at
rank 4 under Reset-free; we include it anyway because dropping it would
make the panel omit the Strict-dominant body. The three bodies span two
families (Floor, MVT); a third family (VT) is absent from the protocol
overlaps and we do not fabricate a VT cell.

## Grid (frozen)

Seeds: 42 only (one run per cell; seed variance is quantified in the main
corpus, not this figure). Protocols: tau = 0.0 (reset-free), 0.20
(episodic), 0.50 (strict). Hyperparameters pinned to corpus PPO
(`--num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256`).
Training length: 5M steps. Checkpoints saved every 1M steps via
`--save-checkpoints`.

Total runs: 3 bodies x 3 protocols x 1 seed = 9 runs.

## Render rule (frozen)

From each final 5M-step checkpoint we render a deterministic rollout
(policy mean action, no sampling) for 500 env-steps with
`scripts/render_rollout.py` extended to also dump `qpos_trace.npz`
(full qpos sequence) and `action_trace.npz` (action sequence). The
per-cell visual is:

- Top-view x-y trajectory colored by time (viridis purple->yellow,
  swim2real style), shared axis scale across cells of a row.
- 3-5 side-view body silhouettes sampled uniformly along the rollout,
  overlaid semi-transparently at the trajectory positions.

## Energy metric (frozen)

Primary: control-effort cost of transport (C-COT) =
`sum_t ||a_t||^2 / distance_travelled_m` computed from the saved action
trace. Reported in "action^2 per meter" units. Do NOT call it physical
energy; label as "control-effort COT" or "actuation cost per meter".

Secondary: per-step action-L2 mean across the rollout (side panel).

## Include gates

- Render gate: all 9 cells must produce valid qpos_trace (>= 100 frames,
  non-constant). If a cell fails, re-seed (seed=43) and re-render; if
  still fails, leave the cell blank and annotate in the caption.
- Energy paragraph gate: action-L2 per meter must differ across protocols
  within the same body by >= 20% for at least 2 of 3 bodies, otherwise
  energy is mentioned only in Limitations (no new subsection).
- Pareto gate: the `PSR vs fall_rate` and `PSR vs C-COT` appendix plots
  ship regardless of significance; they are descriptive.

## Amendments

None.
