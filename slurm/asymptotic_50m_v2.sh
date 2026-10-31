#!/bin/bash
#SBATCH --job-name=asymp-50m
#SBATCH --partition=common
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-161
#SBATCH --output=results/asymptotic_50m_v2/logs/asymp50m_%A_%a.log
#SBATCH --mem=8G

# Asymptotic 50M main experiment on current code stack.
# 162 array tasks = 18 bodies * 3 taus * 3 seeds, 1 run per task.
# Frozen by records/asymptotic_spec.md.
#
# Task-id -> (body, tau, seed) unpacking:
#   body_idx = task_id // 9
#   tau_idx  = (task_id // 3) % 3
#   seed_idx = task_id % 3
#
# Expected wall: ~18h per run with 4-pack contention on a TITAN V
# (5M takes ~1.8h per run -> 50M ~18h). 30h SLURM limit gives 67% margin.
# DO NOT submit until 5M abort gate passes per asymptotic_spec.md.

set -uo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

TASK_ID=$SLURM_ARRAY_TASK_ID
BODY_IDX=$((TASK_ID / 9))
TAU_IDX=$(((TASK_ID / 3) % 3))
SEED_IDX=$((TASK_ID % 3))

TAUS=(0.00 0.20 0.50)
SEEDS=(42 43 44)
TAU="${TAUS[$TAU_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TAU_LABEL=$(python -c "print('${TAU}'.replace('.', 'p'))")

BODY_NAME=$(python -c "import json; print(json.load(open('records/asymptotic_panel_bodies.json'))['bodies'][$BODY_IDX]['name'])")

OUT_BASE="results/asymptotic_50m_v2/${BODY_NAME}"
OUT_DIR="${OUT_BASE}/tau${TAU_LABEL}_seed${SEED}"
mkdir -p "$OUT_DIR"

BODY_LIST="${OUT_BASE}/morph.txt"
echo "$BODY_NAME" > "$BODY_LIST"

echo "=== Asymptotic 50M: body=${BODY_NAME} tau=${TAU} seed=${SEED} (task_id=${TASK_ID}) ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

python scripts/train_mjx.py \
  --threshold "$TAU" \
  --seed "$SEED" \
  --impl jax \
  --timesteps 50000000 \
  --num_envs 128 \
  --unroll_length 16 \
  --n_epochs 10 \
  --minibatch_size 256 \
  --morphology-list "$BODY_LIST" \
  --out_dir "$OUT_DIR" \
  --xml_dir "${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"

echo "Done: $(date)"
