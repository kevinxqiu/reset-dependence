#!/bin/bash
#SBATCH --job-name=rfc-fact
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/factorial_panel_%A.log
#SBATCH --mem=32G

# Factorial panel: 30 UNIMAL bodies x 1 tau x 3 seeds = 90 runs per submission.
# Pre-registered in records/factorial_prereg.md. Reuses the boundary-free
# 30-body panel and same seeds.
#
# Submit once per (TAU, SEED) combination. Example for full 3x2 completion:
#   for TAU in 0.20 0.50; do
#     TAU_LABEL=$(python -c "print(str($TAU).replace('.','p'))")
#     for SEED in 42 43 44; do
#       LABEL="tau${TAU_LABEL}_5M_seed${SEED}"
#       sbatch --export=ALL,TAU=$TAU,SEED=$SEED,LABEL=$LABEL slurm/factorial_panel.sh
#     done
#   done

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

N_PARALLEL=4
export PACK_COUNT="$N_PARALLEL"

MORPH_LIST="records/boundary_free_bodies.txt"
PANEL_JSON="records/boundary_free_bodies.json"
XML_DIR="${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"

if [ ! -f "$MORPH_LIST" ]; then
    echo "ERROR: $MORPH_LIST missing; commit it before submitting." >&2
    exit 2
fi

TOTAL_MORPHS=$(wc -l < "$MORPH_LIST")
if [ "$TOTAL_MORPHS" -ne 30 ]; then
    echo "ERROR: expected 30 bodies in panel, got $TOTAL_MORPHS" >&2
    exit 2
fi

: "${TAU:?need --export=TAU=<float>}"
: "${SEED:?need --export=SEED=<int>}"
: "${LABEL:?need --export=LABEL=<str>}"

OUT_DIR="results/factorial/${LABEL}"
mkdir -p "${OUT_DIR}/logs"

echo "=== Factorial panel: tau=${TAU}, seed=${SEED}, N_bodies=${TOTAL_MORPHS} ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

CHUNK=$(( (TOTAL_MORPHS + N_PARALLEL - 1) / N_PARALLEL ))

PIDS=()
for p in $(seq 0 $((N_PARALLEL - 1))); do
    P_START=$((p * CHUNK))
    P_END=$(((p + 1) * CHUNK))
    if [ $P_END -gt $TOTAL_MORPHS ]; then P_END=$TOTAL_MORPHS; fi
    if [ $P_START -ge $P_END ]; then continue; fi

    SUBLIST="${OUT_DIR}/logs/sublist_p${p}.txt"
    sed -n "$((P_START + 1)),${P_END}p" "$MORPH_LIST" > "$SUBLIST"

    echo "  Launching process ${p}: $((P_END - P_START)) morphs (lines ${P_START}..${P_END})"

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
        --max-steps 5000001 \
        --out_dir "$OUT_DIR" \
        > "${OUT_DIR}/logs/proc_${SLURM_JOB_ID}_p${p}.log" 2>&1 &
    PIDS+=($!)

    if [ "$p" -lt $((N_PARALLEL - 1)) ]; then
        sleep 10
    fi
done

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
