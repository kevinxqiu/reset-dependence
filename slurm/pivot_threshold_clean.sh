#!/bin/bash
#SBATCH --job-name=rfc-thr-clean
#SBATCH --partition=common
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=results/mjx_ppo_%x/logs/thr_%A_%a.log
#SBATCH --mem=16G

# Clean threshold sweep for the Protocol-Conditioned Morphology Response paper.
# Trains 100 morphs at a specified threshold using the SAME pipeline and
# hyperparameters as the 900-run corpus (MJX custom PPO, 5M steps).
#
# Submit once per (threshold, seed) pair:
#   sbatch --export=ALL,THRESHOLD=0.10,SEED=42,TAU_LABEL=0p10 slurm/pivot_threshold_clean.sh
#   sbatch --export=ALL,THRESHOLD=0.10,SEED=43,TAU_LABEL=0p10 slurm/pivot_threshold_clean.sh
#   sbatch --export=ALL,THRESHOLD=0.10,SEED=44,TAU_LABEL=0p10 slurm/pivot_threshold_clean.sh
#   sbatch --export=ALL,THRESHOLD=0.35,SEED=42,TAU_LABEL=0p35 slurm/pivot_threshold_clean.sh
#   sbatch --export=ALL,THRESHOLD=0.35,SEED=43,TAU_LABEL=0p35 slurm/pivot_threshold_clean.sh
#   sbatch --export=ALL,THRESHOLD=0.35,SEED=44,TAU_LABEL=0p35 slurm/pivot_threshold_clean.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

OUT_DIR="results/mjx_ppo_${TAU_LABEL}_seed${SEED}"
mkdir -p "${OUT_DIR}/logs"

START=$((SLURM_ARRAY_TASK_ID * 25))
END=$(((SLURM_ARRAY_TASK_ID + 1) * 25))
if [ $END -gt 100 ]; then END=100; fi

echo "=== Clean threshold sweep: tau=${THRESHOLD}, seed=${SEED}, morphs ${START}-${END} ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

# ORIGINAL corpus hyperparams. Must match to keep results comparable.
python scripts/train_mjx.py \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --timesteps 5000000 \
    --start_idx "$START" \
    --end_idx "$END" \
    --num_envs 128 \
    --unroll_length 16 \
    --n_epochs 10 \
    --minibatch_size 256 \
    --out_dir "$OUT_DIR"

echo ""
echo "Done: $(date)"
