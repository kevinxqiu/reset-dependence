#!/bin/bash
#SBATCH --job-name=asymp-canary
#SBATCH --partition=common
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=results/asymptotic_50m/logs/canary_%j.log
#SBATCH --mem=16G

# Phase 1b 50M asymptotic - CANARY test.
# Runs 2 bodies x 2 taus x 1 seed = 4 runs, 4-packed on 1 GPU, at 2M steps.
# This is a cheap validation that the code changes added since corpus
# training (provenance capture, --save-checkpoints, pipeline_init override,
# --impl flag) do not break the JAX path or collide with corpus output
# directories. Expected wall ~30-40 min.
#
# What this validates:
#   - results/asymptotic_50m/ output tree is correctly created (new path).
#   - Provenance block is written into each results.json.
#   - --save-checkpoints produces checkpoints at 1M intervals.
#   - 4-pack launch still works on Entropy GPUs.
#
# What this does NOT validate:
#   - 50M walltime (extrapolate from per-proc throughput: ~9 hr/proc solo,
#     ~12-14 hr/proc at 4-pack; 9 procs / 4-pack fits in 36 hr SLURM limit).
#
# Submit:
#   sbatch slurm/asymptotic_panel_50m_canary.sh
#
# Verify after completion:
#   ls results/asymptotic_50m/jax/*/tau*_seed*/results.json
#   ls results/asymptotic_50m/jax/*/tau*_seed*/checkpoints/*.npz
#   jq '."'<morph>'" | {provenance, total_train_steps}' results/asymptotic_50m/jax/<morph>/<tau_seed>/results.json

# NOTE: -e deliberately omitted, see equivalence_pilot.sh rationale.
set -uo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PACK_COUNT=4

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

IMPL=jax

# Pick the first 2 bodies from the 18-body asymptotic panel.
CANARY_BODIES=$(python -c "
import json
bodies = json.load(open('records/asymptotic_panel_bodies.json'))['bodies']
print(' '.join(b['name'] for b in bodies[:2]))
")

OUT_BASE="results/asymptotic_50m/${IMPL}"
mkdir -p "${OUT_BASE}/logs"

echo "=== Asymptotic 50M canary: 2 bodies x 2 taus x 1 seed @ 2M steps ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo "Canary bodies: $CANARY_BODIES"
echo ""

MAX_PARALLEL=4
ANY_FAIL=0
launched=0

for BODY_NAME in $CANARY_BODIES; do
  BODY_OUT="${OUT_BASE}/${BODY_NAME}"
  mkdir -p "$BODY_OUT"
  BODY_LIST="${BODY_OUT}/morph.txt"
  echo "$BODY_NAME" > "$BODY_LIST"

  for TAU in 0.00 0.20; do
    TAU_LABEL=$(python -c "print('${TAU}'.replace('.', 'p'))")
    SEED=42
    OUT_DIR="${BODY_OUT}/tau${TAU_LABEL}_seed${SEED}"
    mkdir -p "$OUT_DIR"
    LABEL="${BODY_NAME}_tau${TAU_LABEL}_seed${SEED}"

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
      --timesteps 2000000 \
      --num_envs 128 \
      --unroll_length 16 \
      --n_epochs 10 \
      --minibatch_size 256 \
      --save-checkpoints \
      --morphology-list "$BODY_LIST" \
      --out_dir "$OUT_DIR" \
      --xml_dir "${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}" \
      > "${OUT_BASE}/logs/canary_${LABEL}_${SLURM_JOB_ID}.log" 2>&1 &
    launched=$((launched + 1))
    sleep 10
  done
done

echo "All ${launched} canary launched. Draining..."
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
  wait -n
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 127 ]; then
    ANY_FAIL=1
    echo "  child exited rc=$rc"
  fi
done

if [ "$ANY_FAIL" -ne 0 ]; then
  echo "CANARY FAILED: at least one run failed. DO NOT submit full Phase 1b."
  exit 1
fi

# Verify each canary run wrote both results.json AND at least one checkpoint.
echo ""
echo "=== Canary artifact check ==="
ALL_OK=1
for BODY_NAME in $CANARY_BODIES; do
  for TAU_LABEL in 0p00 0p20; do
    RJSON="${OUT_BASE}/${BODY_NAME}/tau${TAU_LABEL}_seed42/results.json"
    CKPT_DIR="${OUT_BASE}/${BODY_NAME}/tau${TAU_LABEL}_seed42/checkpoints"
    if [ ! -s "$RJSON" ]; then
      echo "MISSING: $RJSON"
      ALL_OK=0
    fi
    NCKPT=$(ls "$CKPT_DIR"/*.npz 2>/dev/null | wc -l)
    if [ "$NCKPT" -lt 1 ]; then
      echo "MISSING CHECKPOINTS: $CKPT_DIR (found $NCKPT)"
      ALL_OK=0
    fi
    if [ -s "$RJSON" ]; then
      HAS_PROV=$(python -c "import json; d=json.load(open('$RJSON')); e=next(iter(d.values())); print('yes' if e.get('provenance') else 'no')" 2>/dev/null)
      if [ "$HAS_PROV" != "yes" ]; then
        echo "MISSING PROVENANCE: $RJSON"
        ALL_OK=0
      fi
    fi
  done
done

if [ "$ALL_OK" -ne 1 ]; then
  echo "CANARY ARTIFACT CHECK FAILED."
  exit 1
fi
echo "Canary OK: all 4 runs produced results.json with provenance + checkpoints."
echo "Done: $(date)"
