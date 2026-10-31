#!/bin/bash
#SBATCH --job-name=rfc-bkfl
#SBATCH --partition=common
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=results/mjx_ppo/logs/bkfl_%x_%A_%a.log
#SBATCH --mem=16G

# Backfill missing morphologies with ORIGINAL 5M-step hyperparameters.
# The original runs used NUM_ENVS=128, UNROLL=16, EPOCHS=10, MB=256.
# Must match to keep results comparable.
#
# Submit for each missing condition:
#   sbatch --export=ALL,THRESHOLD=0.00,SEED=42 slurm/backfill_mjx.sh
#   sbatch --export=ALL,THRESHOLD=0.00,SEED=43 slurm/backfill_mjx.sh
#   sbatch --export=ALL,THRESHOLD=0.00,SEED=44 slurm/backfill_mjx.sh
#   sbatch --export=ALL,THRESHOLD=0.50,SEED=44 slurm/backfill_mjx.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

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
