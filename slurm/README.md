# Launch recipes

These SLURM scripts document the grids, seeds, thresholds, output directories
and GPU packing behind each panel. They are not needed to check the paper's
numbers, which come from the released run summaries on CPU.

The scripts keep standard `#SBATCH` resource headers. Pass the partition,
account and QOS of your cluster at submission time, and set:

```bash
export RFC_REPO_DIR=/path/to/this/repository
export RFC_VENV=/path/to/a/venv/with/requirements-training.txt
export RFC_UNIMAL_XML_DIR=/path/to/assets/unimals_100/xml   # default: assets/unimals_100/xml
```

Pass job parameters as `sbatch --export=ALL,THRESHOLD=0.20,SEED=42 ...`: the
`ALL` keeps the variables above in the job's environment. Create the
`--output` log directory before `sbatch`; SLURM opens it before the script
runs.

| Script | Panel (results directory) |
| --- | --- |
| `../corpus_revision/train_mjx_launcher_2026-04-07.sh` | 900-run corpus at tau = 0, 0.20, 0.50 (`mjx_ppo_{0p00,0p20,0p50}_*`), first launches |
| `../corpus_revision/backfill_mjx_launcher_2026-04-08.sh` | Corpus backfill of bodies the first launches left unfinished |
| `pivot_threshold_fast.sh` | Runs at tau = 0.10 and 0.35 (`mjx_ppo_{0p10,0p35}_*`), four processes per GPU |
| `pivot_threshold_clean.sh` | One-process-per-GPU variant of the same sweep, written on 12 April 2026 |
| `backfill_mjx.sh` | The corpus backfill launcher, with private paths replaced by the environment variables above |
| `train_sac_mjx.sh`, `sac_full_panel_expansion.sh` | SAC panel (`sac_mjx_*`) |
| `equivalence_pilot.sh` | Retraining with later code (`equivalence_pilot/`) |
| `boundary_free_smoke.sh`, `boundary_free_panel.sh` | No time limit (`boundary_free/`) |
| `boundary_sweep_panel.sh` | Time-limit durations (`boundary_sweep/`) |
| `factorial_panel.sh` | Rule by time limit (`factorial/`) |
| `alt_termination_smoke.sh`, `alt_termination.sh` | Contact termination (`alt_termination/`) |
| `nems_panel.sh` | Non-evolved bodies (`nems_panel/`); needs the generated bodies under `assets/generated_morphologies/`, which are not distributed |
| `asymptotic_5m.sh` | Longer budget, 5M-step arm (`asymptotic_5m/`) |
| `asymptotic_50m_v2.sh`, `asymptotic_panel_50m.sh`, `asymptotic_panel_50m_canary.sh` | Longer budget, 50M-step phase (not completed, no released results) |
| `menagerie_panel.sh` | Existing robots (`menagerie_panel/`); needs a MuJoCo Menagerie checkout in `~/mujoco_menagerie` |
| `temporal_gait_retrain.sh`, `traces_only_rollouts.sh`, `render_temporal_rollouts.sh` | Nine illustrative controllers and their gait traces (`gait-traces/`) |
| `render_video.sh` | Rollout videos of one Floor body from a checkpoint of the longer-budget panel (not used in the paper) |

The trainer in `scripts/train_mjx.py` is a later revision than the one that
trained the corpus. Runs with later code differ from the corpus in absolute
PSR and, after Strict training, in ranking (see the retraining test in the
paper's appendix). To repeat the corpus as it was run, use
`corpus_revision/train_mjx_c8089cf.py`.
