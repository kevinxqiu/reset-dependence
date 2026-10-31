# A.0 Metric-Semantics Audit Report

Audit date: 2026-04-11
Corpus: 100 morphs x 3 protocols x 3 seeds = 900 records
Family distribution: {'floor': 41, 'mvt': 22, 'vt': 37}

## Provenance (from first record)

- `algorithm`: `PPO`
- `backend`: `MJX-CustomPPO`
- `timesteps`: `5000000`
- `train_time_s_approx`: `3508.6837396621704`
- `threshold_field`: `0.0`

## Headline findings

### 1. design review claim: fall_rate under eval_neutral is ~always 0 for tau=0.00 runs
- Verdict: **confirmed**
- Mean: 0.000000, Max: 0.000000
- Zeros: 300 of 300

### 2. Is eval_tau020 fall_rate usable as a recoverability signal?
- Mean: 0.4048, median: 0.1000
- Range: [0.0000, 1.0000]
- Zeros: 133, Ones: 93, Total: 300
- Useful as continuous signal: **True**

### 3. Alternative: mean_ep_len under eval_tau020 for tau=0.00
- Mean: 673.92, median: 962.20
- Range: [33.45, 1000.00]
- (max possible ep_len = 1000)

### 4. Episodic baseline (tau=0.20 training, tau=0.20 eval)
- fall_rate mean: 0.2145
- mean_ep_len mean: 876.26

### 5. Policy checkpoints (for R_task feasibility)
- Checkpoint files found: **0**
- R_task from existing corpus feasible: **False**

## Implications for plan v2

- A.1 metric must use `eval_tau020`, not `eval_neutral`. `eval_neutral` fall_rate is useless.
- `eval_tau020` fall_rate is a usable continuous signal.
- **R_task cannot be computed from the existing corpus** (no checkpoints). Any disturbance-recovery benchmark requires retraining with checkpointing.
