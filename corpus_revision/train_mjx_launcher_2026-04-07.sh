#!/bin/bash
#SBATCH --job-name=rfc-mjx
#SBATCH --partition=common
# (cluster QOS line removed for release; pass your own to sbatch)
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=results/mjx_ppo/logs/mjx_%x_%A_%a.log
#SBATCH --mem=16G

# Historical record of a corpus launch, edited for release only where marked.
# At the time, scripts/train_mjx.py was a corpus-era trainer revision, now kept in
# corpus_revision/ (see the table in corpus_revision/README.md); in this repository
# scripts/train_mjx.py is the later, maintained trainer. To repeat a corpus cell, use
# the command in corpus_revision/README.md rather than this launcher.
#
# MJX-PPO: 100 morphologies x 5M steps, GPU-accelerated
# Submit 9 jobs (3 thresholds x 3 seeds):
#   sbatch --export=THRESHOLD=0.20,SEED=42 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.20,SEED=43 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.20,SEED=44 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.50,SEED=42 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.50,SEED=43 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.50,SEED=44 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.00,SEED=42 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.00,SEED=43 slurm/train_mjx.sh
#   sbatch --export=THRESHOLD=0.00,SEED=44 slurm/train_mjx.sh

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

echo "=== MJX-PPO 5M: tau=${THRESHOLD}, seed=${SEED}, morphs ${START}-${END} ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "JAX devices: $(python3 -c 'import jax; print(jax.devices())' 2>/dev/null)"
echo ""

python scripts/train_mjx.py \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --start_idx "$START" \
    --end_idx "$END"

echo ""
echo "Done: $(date)"
