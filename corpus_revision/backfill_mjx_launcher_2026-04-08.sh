#!/bin/bash
#SBATCH --job-name=rfc-bkfl
#SBATCH --partition=common
# (cluster QOS line removed for release; pass your own to sbatch)
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=results/mjx_ppo/logs/bkfl_%x_%A_%a.log
#SBATCH --mem=16G

# Historical record of a corpus launch, edited for release only where marked.
# At the time, scripts/train_mjx.py was a corpus-era trainer revision, now kept in
# corpus_revision/ (see the table in corpus_revision/README.md); in this repository
# scripts/train_mjx.py is the later, maintained trainer. To repeat a corpus cell, use
# the command in corpus_revision/README.md rather than this launcher.
#
# Backfill missing morphologies with ORIGINAL 5M-step hyperparameters.
# The original runs used NUM_ENVS=128, UNROLL=16, EPOCHS=10, MB=256.
# Must match to keep results comparable.
#
# Submit for each missing condition:
#   sbatch --export=THRESHOLD=0.00,SEED=42 slurm/backfill_mjx.sh
#   sbatch --export=THRESHOLD=0.00,SEED=43 slurm/backfill_mjx.sh
#   sbatch --export=THRESHOLD=0.00,SEED=44 slurm/backfill_mjx.sh
#   sbatch --export=THRESHOLD=0.50,SEED=44 slurm/backfill_mjx.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"  # edited for release: was the authors' checkout
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi  # edited for release: was the authors' environment

mkdir -p results/mjx_ppo/logs

START=$((SLURM_ARRAY_TASK_ID * 25))
END=$(((SLURM_ARRAY_TASK_ID + 1) * 25))
if [ $END -gt 100 ]; then END=100; fi

echo "=== BACKFILL MJX-PPO 5M: tau=${THRESHOLD}, seed=${SEED}, morphs ${START}-${END} ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Using ORIGINAL hyperparams: num_envs=128, unroll=16, epochs=10, mb=256, timesteps=5M"
echo ""

python scripts/train_mjx.py \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --start_idx "$START" \
    --end_idx "$END" \
    --timesteps 5000000 \
    --num_envs 128 \
    --unroll_length 16 \
    --n_epochs 10 \
    --minibatch_size 256

echo ""
echo "Done: $(date)"
