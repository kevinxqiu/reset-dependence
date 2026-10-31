#!/bin/bash
#SBATCH --job-name=warp-equiv
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=results/equivalence_pilot/logs/equiv_%A_%a.log
#SBATCH --mem=32G

# Phase 1a-bis equivalence pilot, Eq-A arm (JAX only).
# 12 array tasks, one per equivalence-pilot body. Each task runs 3 tau x 3
# seeds = 9 training runs in 4-pack on one GPU (pack_count=4 stagger).
#
# Warp backend declined 2026-04-15 after preflight (jobs 152217 and 152236)
# confirmed mjx 3.7 DataWarp schema is incompat with Brax's MJX pipeline.
# See records/warp_preflight_report.md. This pilot runs ONLY the
# JAX Eq-A arm: MJX 3.6.0 + JAX 0.9.2 vs the 900-run corpus on
# the same 12 bodies, at corpus config. D6 gate: Spearman rho > 0.85 PSR,
# rho > 0.9 Delta_reset, median relative PSR drift < 0.10, sign agreement
# >= 90%, regression slope in [0.9, 1.1].
#
# Submit:
#   sbatch slurm/equivalence_pilot.sh

# NOTE: -e deliberately omitted. `wait -n` returns the child's exit code,
# and under -e a single nonzero child aborts the script BEFORE the
# bookkeeping can update ANY_FAIL, leaving 8 sibling procs for SLURM to
# kill instead of draining them. -u / -o pipefail are kept.
set -uo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PACK_COUNT=4

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

# Warp declined 2026-04-15 (records/warp_preflight_report.md).
# Hardcode jax so a stale ambient IMPL from the submit shell cannot silently
# violate the stack freeze.
IMPL=jax
BODY_IDX=$SLURM_ARRAY_TASK_ID

BODY_NAME=$(python -c "import json; print(json.load(open('records/equivalence_pilot_bodies.json'))['bodies'][$BODY_IDX]['name'])")

# Path keyed by morph name, so reordering the body JSON does not mix bodies
# under the same body{idx} path.
OUT_BASE="results/equivalence_pilot/${IMPL}/${BODY_NAME}"
mkdir -p "${OUT_BASE}/logs"

BODY_LIST="${OUT_BASE}/morph.txt"
echo "$BODY_NAME" > "$BODY_LIST"

echo "=== Equivalence pilot: body=${BODY_NAME} (idx=${BODY_IDX}), impl=${IMPL} ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

# Run 3 tau x 3 seeds = 9 processes, but pack at most MAX_PARALLEL on the
# GPU at any time. Use `wait -n` to block until any one running job exits,
# then launch the next. 10s stagger between launches keeps JAX/Warp
# compile-time RAM peaks from colliding.
MAX_PARALLEL=4
ANY_FAIL=0
launched=0
for TAU in 0.00 0.20 0.50; do
  TAU_LABEL=$(python -c "print('${TAU}'.replace('.', 'p'))")
  for SEED in 42 43 44; do
    OUT_DIR="${OUT_BASE}/tau${TAU_LABEL}_seed${SEED}"
    mkdir -p "$OUT_DIR"
    LABEL="tau${TAU_LABEL}_seed${SEED}"

    # Block until <MAX_PARALLEL jobs are running.
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
      # `wait -n` returns when ANY one of the currently-tracked background
      # jobs exits; non-zero exit propagates as the return value here.
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
      --timesteps 5000000 \
      --num_envs 128 \
      --unroll_length 16 \
      --n_epochs 10 \
      --minibatch_size 256 \
      --warp_njmax "${WARP_NJMAX:-256}" \
      --warp_nconmax "${WARP_NCONMAX:-128}" \
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
