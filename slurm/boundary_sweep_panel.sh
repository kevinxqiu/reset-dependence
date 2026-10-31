#!/bin/bash
#SBATCH --job-name=rfc-bsweep
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/boundary_sweep_panel_%A.log
#SBATCH --mem=32G

# Boundary duration sweep: 30 UNIMAL bodies x tau=0 x 1 max_steps value x 3 seeds.
# Reuses the same fixed 30-body panel as boundary-free.
#
# Submit per (MAX_STEPS, SEED). The meaningful intermediate values are:
# B=100001 replaced with B=3001 (per-env steps 39k < 100001 so truncation
# never fires; B=100001 was redundant with B=5M). Current sweep runs
# MAX_STEPS in {3001, 10001} at tau=0 alongside corpus (B=1000) and
# boundary-free (B=5000001):
#   for MAX in 3001 10001; do
#     MAX_LABEL="B${MAX}"
#     for SEED in 42 43 44; do
#       LABEL="${MAX_LABEL}_seed${SEED}"
#       sbatch --export=ALL,MAX_STEPS=$MAX,SEED=$SEED,LABEL=$LABEL \
#              slurm/boundary_sweep_panel.sh
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
    echo "ERROR: $MORPH_LIST missing; include it before submitting." >&2
    exit 2
fi

TOTAL_MORPHS=$(wc -l < "$MORPH_LIST")
if [ "$TOTAL_MORPHS" -ne 30 ]; then
    echo "ERROR: expected 30 bodies in panel, got $TOTAL_MORPHS" >&2
    exit 2
fi

: "${MAX_STEPS:?need --export=MAX_STEPS=<int>}"
: "${SEED:?need --export=SEED=<int>}"
: "${LABEL:?need --export=LABEL=<str>}"

OUT_DIR="results/boundary_sweep/${LABEL}"
mkdir -p "${OUT_DIR}/logs"

echo "=== Boundary-sweep panel: max_steps=${MAX_STEPS}, seed=${SEED}, N_bodies=${TOTAL_MORPHS} ==="
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
        --threshold 0.0 \
        --seed "$SEED" \
        --timesteps 5000000 \
        --xml_dir "$XML_DIR" \
        --morphology-list "$SUBLIST" \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --max-steps "$MAX_STEPS" \
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
