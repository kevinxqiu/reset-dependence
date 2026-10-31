#!/bin/bash
#SBATCH --job-name=rfc-sacm
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=results/sac_mjx/logs/sac_%x_%A_%a.log
#SBATCH --mem=16G

# MJX-SAC: 24 stratified morphologies x 5M steps, GPU-accelerated
# Submit 9 jobs (3 thresholds x 3 seeds):
#   sbatch --export=ALL,THRESHOLD=0.20,SEED=42 slurm/train_sac_mjx.sh
#   sbatch --export=ALL,THRESHOLD=0.50,SEED=42 slurm/train_sac_mjx.sh
#   sbatch --export=ALL,THRESHOLD=0.00,SEED=42 slurm/train_sac_mjx.sh
#   (repeat for seeds 43, 44)

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

mkdir -p results/sac_mjx/logs

START=$((SLURM_ARRAY_TASK_ID * 8))
END=$(((SLURM_ARRAY_TASK_ID + 1) * 8))
if [ $END -gt 24 ]; then END=24; fi

echo "=== MJX-SAC 5M: tau=${THRESHOLD}, seed=${SEED}, morphs ${START}-${END} ==="
echo "Host: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "JAX devices: $(python3 -c 'import jax; print(jax.devices())' 2>/dev/null)"
echo ""

python scripts/train_sac_mjx.py \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --morphology-list scripts/mechanism_morphologies.txt \
    --start_idx "$START" \
    --end_idx "$END"

echo ""
echo "Done: $(date)"
