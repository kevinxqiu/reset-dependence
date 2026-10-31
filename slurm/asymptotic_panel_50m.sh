#!/bin/bash
#SBATCH --job-name=asymp-50m
#SBATCH --partition=common
#SBATCH --time=1-20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-17
#SBATCH --output=results/asymptotic_50m/logs/asymp_%A_%a.log
#SBATCH --mem=32G

# Phase 1b 50M asymptotic validation.
# 18 array tasks, one per asymptotic-panel body. Each task runs 3 tau x 3
# seeds = 9 training runs at 50M steps, 4-packed on one GPU.
# Expected per-task wall: ~30 hr at 4-pack (~14 hr per proc * ceil(9/4)).
#
# Output: results/asymptotic_50m/jax/{morph}/tau{TAU}_seed{SEED}/results.json
# DELIBERATELY DISJOINT from the 5M corpus path (results/mjx_ppo_*) so
# analysis scripts can cleanly keep the two regimes separate.
#
# Prerequisite: slurm/asymptotic_panel_50m_canary.sh must pass before
# submitting this. The canary validates output path, provenance writes,
# checkpoint saving, and 4-pack launching; this then runs the full panel.
#
# Submit:
#   sbatch slurm/asymptotic_panel_50m.sh

# NOTE: -e deliberately omitted, see equivalence_pilot.sh rationale.
set -uo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PACK_COUNT=4

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

IMPL=jax
BODY_IDX=$SLURM_ARRAY_TASK_ID

BODY_NAME=$(python -c "import json; print(json.load(open('records/asymptotic_panel_bodies.json'))['bodies'][$BODY_IDX]['name'])")

OUT_BASE="results/asymptotic_50m/${IMPL}/${BODY_NAME}"
mkdir -p "${OUT_BASE}/logs"

BODY_LIST="${OUT_BASE}/morph.txt"
echo "$BODY_NAME" > "$BODY_LIST"

echo "=== Asymptotic 50M: body=${BODY_NAME} (idx=${BODY_IDX}), impl=${IMPL} ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

MAX_PARALLEL=4
ANY_FAIL=0
launched=0
for TAU in 0.00 0.20 0.50; do
  TAU_LABEL=$(python -c "print('${TAU}'.replace('.', 'p'))")
  for SEED in 42 43 44; do
    OUT_DIR="${OUT_BASE}/tau${TAU_LABEL}_seed${SEED}"
    mkdir -p "$OUT_DIR"
    LABEL="tau${TAU_LABEL}_seed${SEED}"

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n
      rc=$?
      if [ "$rc" -ne 0 ] && [ "$rc" -ne 127 ]; then
        ANY_FAIL=1
        echo "  child exited rc=$rc"
      fi
    done

    echo "  Launching ${LABEL} (active before launch: $(jobs -rp | wc -l))"
    python scripts/train_mjx.py \
      --threshold "$TAU" \
      --seed "$SEED" \
      --impl "$IMPL" \
      --timesteps 50000000 \
      --num_envs 128 \
      --unroll_length 16 \
      --n_epochs 10 \
      --minibatch_size 256 \
      --save-checkpoints \
      --morphology-list "$BODY_LIST" \
      --out_dir "$OUT_DIR" \
      --xml_dir "${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}" \
      > "${OUT_BASE}/logs/${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
    launched=$((launched + 1))
    sleep 10
  done
done

echo "All ${launched} launched. Draining stragglers..."
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
  wait -n
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 127 ]; then
    ANY_FAIL=1
    echo "  child exited rc=$rc"
  fi
done

if [ "$ANY_FAIL" -ne 0 ]; then
  echo "WARNING: at least one run failed"
  exit 1
fi
echo "Done: $(date)"
