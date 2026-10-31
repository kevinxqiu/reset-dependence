#!/bin/bash
#SBATCH --job-name=rfc-tgait
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/temporal_gait_%A.log
#SBATCH --mem=32G

# Temporal-gait retrain: 3 bodies x 1 tau x seed 42 x 5M steps, packed 3-up
# per job; submit once per tau for the full 3x3 grid.
#
# Usage:
#   for TAU in 0.00 0.20 0.50; do
#     TAU_LABEL=$(python -c "print(f'{float($TAU):.2f}'.replace('.','p'))")
#     sbatch --export=ALL,TAU=$TAU,LABEL=tau${TAU_LABEL}_5M slurm/temporal_gait_retrain.sh
#   done
#
# Body list is frozen in records/temporal_gait_bodies.txt.

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

MORPH_LIST="records/temporal_gait_bodies.txt"
XML_DIR="${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"
SEED=42

: "${TAU:?need --export=TAU=<float>}"
: "${LABEL:?need --export=LABEL=<str>}"

TOTAL_MORPHS=$(wc -l < "$MORPH_LIST")
if [ "$TOTAL_MORPHS" -ne 3 ]; then
    echo "ERROR: expected 3 bodies, got $TOTAL_MORPHS" >&2
    exit 2
fi

N_PARALLEL=3
export PACK_COUNT="$N_PARALLEL"

OUT_DIR="results/temporal_gait/${LABEL}_seed${SEED}"
mkdir -p "${OUT_DIR}/logs"

echo "=== Temporal-gait retrain: tau=${TAU}, seed=${SEED}, N_bodies=${TOTAL_MORPHS} ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

PIDS=()
p=0
while read -r MORPH; do
    [ -z "$MORPH" ] && continue

    SUBLIST="${OUT_DIR}/logs/sublist_p${p}.txt"
    echo "$MORPH" > "$SUBLIST"

    echo "  Launching process ${p}: body=${MORPH}"

    python scripts/train_mjx.py \
        --threshold "$TAU" \
        --seed "$SEED" \
        --timesteps 5000000 \
        --xml_dir "$XML_DIR" \
        --morphology-list "$SUBLIST" \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --impl jax \
        --termination-mode height \
        --max-steps 1000 \
        --save-checkpoints \
        --out_dir "$OUT_DIR" \
        > "${OUT_DIR}/logs/proc_${SLURM_JOB_ID}_p${p}.log" 2>&1 &
    PIDS+=($!)

    p=$((p + 1))
    if [ "$p" -lt "$N_PARALLEL" ]; then
        sleep 10
    fi
done < "$MORPH_LIST"

echo "Waiting for ${#PIDS[@]} parallel processes..."
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: process ${i} (PID ${PIDS[$i]}) failed"
        FAIL=1
    else
        echo "Process ${i} (PID ${PIDS[$i]}) finished OK"
    fi
done

echo ""
if [ $FAIL -eq 1 ]; then
    echo "WARNING: some processes failed, check per-process logs"
    echo "Done: $(date)"
    exit 1
fi

echo "All processes completed successfully"
echo "Done: $(date)"
