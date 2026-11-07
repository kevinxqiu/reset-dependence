# Corpus-era training code

The 900 corpus runs (tau = 0, 0.20, 0.50, seeds 42 to 44) were trained over
several launches between 5 and 8 April 2026, before the code in `scripts/`
gained the options used by the later panels. These files are the code of that
period, taken from the git history of the development repository, which is
not public (the commit hashes below refer to it). Only site-specific lines were
edited for release (the cluster QOS line, the working directory, the
environment path and the default body folder, each marked in the file);
every training line is as it ran. The two launchers are records of how the
corpus was launched, when `scripts/train_mjx.py` was a corpus-era revision; to
rerun a corpus cell, use the command below with the archived trainer.

| File | Source | Role |
| --- | --- | --- |
| `train_mjx_ee751af.py` | `scripts/train_mjx.py` at commit `ee751af` (5 April 2026) | Trainer at the time of the first launches. Its defaults are the corpus settings: 128 environments, unroll 16, 10 epochs, minibatch 256, 5M steps |
| `train_mjx_c8089cf.py` | `scripts/train_mjx.py` at commit `c8089cf` (8 April 2026) | Trainer used with the backfill launcher. Its defaults changed on 7 April (2,048 environments, 50M steps), so the backfill passes the corpus settings explicitly |
| `train_mjx_launcher_2026-04-07.sh` | `slurm/train_mjx.sh` at commit `79aa5c7` (7 April 2026) | Launcher of the corpus jobs (four array tasks of 25 bodies in name order); it passes no hyperparameter flags |
| `backfill_mjx_launcher_2026-04-08.sh` | `slurm/backfill_mjx.sh` at commit `d0eb62a` (8 April 2026) | Backfill launcher for unfinished bodies; passes `--timesteps 5000000 --num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256` |

The run records store only the rule, seed and step budget (paper, Appendix
A.1), not which launch or code revision produced each run; the other settings
are those that the launch code in this folder sets.

To rerun a corpus cell with the backfill-era trainer (it is self-contained and
imports nothing else from this repository), pass the settings explicitly:

```bash
python corpus_revision/train_mjx_c8089cf.py --threshold 0.50 --seed 42 --start_idx 0 --end_idx 25 \
    --timesteps 5000000 --num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256 \
    --xml_dir assets/unimals_100/xml --out_dir retrain/corpus_strict_seed42
```

Without `--out_dir` the trainer writes to `results/mjx_ppo_<tau>_seed<seed>/`,
where the released summaries unzip, and skips every body already there.

Training needs a CUDA GPU and the pins in `requirements-training.txt`.
