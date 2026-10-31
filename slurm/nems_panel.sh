#!/bin/bash
#SBATCH --job-name=rfc-nems
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=results/nems_panel/logs/nems_%x_%A.log
#SBATCH --mem=32G

# NEMS (Non-Evolved Morphology Substrate) panel: run the protocol sweep on
# 60 UNIMAL-substrate bodies from the pre-DERL-selection init population
# in assets/generated_morphologies/. Directly addresses the paper's
# stated Limitation #1 (DERL evolutionary confound).
#
# Panel frozen in records/nems_panel_bodies.txt by
# scripts/nems_select_bodies.py. Corpus-matched hyperparameters to make
# NEMS results directly comparable to the main UNIMAL corpus.
#
# Submit once per (threshold, seed) pair (9 jobs total, 3 tau x 3 seeds):
#   for TAU in 0.00 0.20 0.50; do
#     for SEED in 42 43 44; do
#       LABEL=$(python3 -c "print('${TAU}'.replace('.', 'p'))")
#       sbatch --export=ALL,THRESHOLD=$TAU,SEED=$SEED,TAU_LABEL=$LABEL \
#         slurm/nems_panel.sh
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
export PACK_COUNT="$N_PARALLEL"  # So train_mjx.py records correct pack count
MORPH_LIST="records/nems_panel_bodies.txt"
XML_DIR="${RFC_NEMS_XML_DIR:-assets/generated_morphologies/xml}"

if [ ! -f "$MORPH_LIST" ]; then
    echo "ERROR: panel file $MORPH_LIST not found; run scripts/nems_select_bodies.py" >&2
    exit 2
fi

TOTAL_MORPHS=$(wc -l < "$MORPH_LIST")
if [ "$TOTAL_MORPHS" -ne 60 ]; then
    echo "ERROR: expected 60 bodies in panel, got $TOTAL_MORPHS" >&2
    exit 2
fi

OUT_DIR="results/nems_panel/tau${TAU_LABEL}_seed${SEED}"
mkdir -p "${OUT_DIR}/logs"

echo "=== NEMS panel: tau=${THRESHOLD}, seed=${SEED}, N=${TOTAL_MORPHS} bodies ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

# Split the 60-body panel into N_PARALLEL chunks by writing per-process morph-list
# files. Process p gets morphs [p * ceil(N/P), (p+1) * ceil(N/P)).
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
        --threshold "$THRESHOLD" \
        --seed "$SEED" \
        --timesteps 5000000 \
        --xml_dir "$XML_DIR" \
        --morphology-list "$SUBLIST" \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --out_dir "$OUT_DIR" \
        > "${OUT_DIR}/logs/proc_${SLURM_JOB_ID}_p${p}.log" 2>&1 &
    PIDS+=($!)

    # Stagger JAX compilation starts
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
