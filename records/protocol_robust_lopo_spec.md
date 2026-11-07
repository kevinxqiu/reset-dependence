# Protocol-Robust Selection: Pre-registered LOPO Evaluation

STATUS 2026-04-19. This spec drove the leave-one-protocol-out (LOPO) analysis
(`scripts/analyze_protocol_robust_lopo.py`, run 2026-04-19). The pre-committed
gate (best-robust vs best-naive at k=5, paired-bootstrap CI excluding zero, 2/3
wins) returned DROP: mean diff = +0.04 PSR across splits, CI = [-0.42, +0.23],
2/3 splits favor robust. The paper reports this outcome in Section 4.3 and
Appendix E; its selection claim is the in-sample worst-case retention across
the three protocols (>=92% at k=5), not generalization to a held-out protocol.
The pre-commitments below are unchanged.

---

FROZEN 2026-04-19. Do not edit after first `scripts/analyze_protocol_robust_lopo.py` run.
Rationale: a design review (2026-04-19) recommended turning the
existing Section 4.3 robust-selection paragraph into a rigorous
leave-one-protocol-out (LOPO) evaluation with pre-committed selection rules and a
paired significance test, to test whether robust selection is a validated
practical method ("protocol choice matters, here is the fix").

## Objective

Test whether protocol-robust selection rules beat naive single-protocol
selection under leave-one-protocol-out cross-validation on the existing
900-run corpus (100 UNIMAL bodies, 3 protocols, 3 seeds, all scored at
$\tau_{\text{eval}} = 0.20$).

## Data

- Source: `results/v3_main_sweep/**/results.json` (same data that underpins
  Section 4.3 retention table and `scripts/compute_robust_selection.py`).
- Body universe: the 100-body UNIMAL corpus. No filtering. No NEMS. No alt-term.
- Per-body per-protocol score: seed-mean per-step reward under
  $\tau_{\text{eval}} = 0.20$ (identical to the retention-table score).
- Per-body per-protocol score for bootstrap CIs: per-seed PSR (3 seeds each).

## Protocols

$P \in \{\text{epi}, \text{str}, \text{rf}\}$ where epi $= \tau_{\text{train}} = 0.20$,
str $= \tau_{\text{train}} = 0.50$, rf $= \tau_{\text{train}} = 0.00$.

## Selection rules (pre-committed BEFORE implementation)

### Naive baselines (3)

1. **SelectByEpi**: top-$k$ bodies by episodic-training seed-mean PSR.
2. **SelectByStr**: top-$k$ bodies by strict-training seed-mean PSR.
3. **SelectByRF**: top-$k$ bodies by reset-free-training seed-mean PSR.

### Protocol-robust rules (4)

4. **RobustMin**: top-$k$ bodies by $\min_{p} \text{PSR}_p$. (Already in paper; re-run under the LOPO split.)
5. **RobustMean**: top-$k$ bodies by $\text{mean}_{p} \text{PSR}_p$. (Already in paper; re-run under the LOPO split.)
6. **Borda**: rank each body within each protocol (1 = worst, 100 = best). Sum ranks across protocols. Top-$k$ have the largest sum. **NEW.**
7. **MinimaxRank**: top-$k$ bodies by $\max_{p} \text{rank}_p$ (smallest max rank, i.e., best worst-case rank). **NEW.**

Rules 6 and 7 are new additions motivated by the design review's "pre-commit 2-3 simple
rules and show all of them" guidance. Rules 4 and 5 are reprised from Section 4.3
so the LOPO analysis is self-contained.

## Evaluation: Leave-One-Protocol-Out (LOPO)

For each held-out protocol $P_{\text{hold}} \in \{\text{epi}, \text{str}, \text{rf}\}$:

1. Construct selection rankings using only the OTHER two protocols
   $\mathcal{P}_{\text{train}} = \{\text{epi}, \text{str}, \text{rf}\} \setminus \{P_{\text{hold}}\}$:
   - Naive rules degenerate: `SelectByX` uses protocol $X$ if $X \in \mathcal{P}_{\text{train}}$,
     otherwise that naive baseline is not evaluable for this split (reported as n/a).
   - Robust rules aggregate over the 2 training protocols only.
2. Score each selection rule's top-$k$ bodies by their mean PSR on the HELD-OUT protocol
   $P_{\text{hold}}$ (same $\tau_{\text{eval}} = 0.20$ evaluation).
3. Compute the retention ratio: (rule's top-$k$ held-out mean PSR) /
   (oracle top-$k$ held-out mean PSR), where oracle = top-$k$ bodies by $P_{\text{hold}}$-training PSR.

Total: 3 LOPO splits $\times$ 2 k-values ($k \in \{5, 10\}$) $\times$ 7 rules = 42 cells
(minus n/a cells for naive baselines where the naive's protocol is held out).

## Statistics

- **Point estimate**: per rule, per split, per $k$: mean held-out PSR and retention ratio.
- **Bootstrap CI**: 95% percentile CI from 10,000 resamples of the 100 bodies
  (each resample independently re-ranks and re-selects top-$k$). Report for each cell.
- **Paired permutation test**: for each (robust rule, naive rule, held-out $P$, $k$) pair,
  test the null that the robust rule's top-$k$ held-out PSR does not exceed the
  naive rule's top-$k$ held-out PSR. 10,000 permutations at the body level.
  Report uncorrected $p$-values.

## Pre-committed gates (decision BEFORE seeing results)

Primary gate: across the 3 LOPO splits at $k = 5$, the best robust rule's
seed-mean held-out PSR exceeds the best naive rule's seed-mean held-out PSR by
$\geq 0.05$ PSR in $\geq 2$ of 3 splits, with the paired-bootstrap 95% CI on the
mean across-split difference excluding zero.

Secondary gate: at $k = 10$ across all 3 LOPO splits, at least one robust rule's
retention ratio vs. oracle is $\geq 0.85$ in every split.

Outcome mapping:

- **Body-text (primary+secondary pass)**: extend Section 4.3 with one paragraph
  reporting best robust rule's gain over best naive at $k=5$, and add a short
  LOPO-retention table. Update abstract mitigation clause.
- **Main-inclusion (primary pass, secondary fail)**: add a 1-paragraph LOPO
  discussion to Section 4.3; keep abstract as-is.
- **Appendix-only (primary fail, secondary pass)**: put the LOPO table and
  paired tests in a dedicated Appendix subsection; one-sentence pointer in
  Section 4.3.
- **Drop (both fail)**: one-sentence disclosure in Section 4.3 ("under a
  leave-one-protocol-out cross-validation, robust aggregation did not
  significantly outperform the best naive baseline"), full analysis in
  supplement.

Any outcome: deposit full analysis at `records/protocol_robust_lopo_analysis.json`.

## Guardrails

1. NO post-hoc rule tweaking. The 4 robust rules are fixed above.
2. NO cherry-picking across $k$: we report $k=5$ and $k=10$ together.
3. NO re-scoping the held-out protocol after seeing LOPO results.
4. Paper text must report per-split outcomes, not just aggregate.
5. If the primary gate is marginal (one split fails by < 0.02 PSR), treat as
   primary fail, not pass.

## Analysis script

`scripts/analyze_protocol_robust_lopo.py` (to be implemented). Must:

- Load per-body per-protocol per-seed PSR from `results/v3_main_sweep/`.
- Implement the 7 rules exactly as specified above.
- Run LOPO, bootstrap CIs, paired permutation tests.
- Emit `records/protocol_robust_lopo_analysis.json` and a markdown
  summary.
- Print gate decision to stdout.
- Exit code 0 regardless of gate outcome (decision is informational).
