#!/bin/bash
#SBATCH --job-name=rfc-thr-fast
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=results/mjx_ppo_%x/logs/thr_%A_%a.log
#SBATCH --mem=32G

# Four-process launcher written on 13 April 2026 for the runs at tau = 0.10
# and 0.35 (mjx_ppo_0p10_*, mjx_ppo_0p35_*). It did not launch the 900-run
# corpus at tau = 0, 0.20 and 0.50; the corpus launchers and trainer are in
# corpus_revision/.
# Packs N_PARALLEL training processes on one GPU because each num_envs=128
# process uses only a small fraction of the device. Each process handles a
# non-overlapping morphology slice. save_locked() in train_mjx.py uses
# fcntl.flock, so concurrent writes to results.json are safe.
#
# Submit once per (threshold, seed) pair:
#   sbatch --export=ALL,THRESHOLD=0.10,SEED=42,TAU_LABEL=0p10 slurm/pivot_threshold_fast.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

N_PARALLEL=4

OUT_DIR="results/mjx_ppo_${TAU_LABEL}_seed${SEED}"
mkdir -p "${OUT_DIR}/logs"

# Each array task covers 25 morphs (0-24, 25-49, 50-74, 75-99)
TASK_START=$((SLURM_ARRAY_TASK_ID * 25))
TASK_END=$(((SLURM_ARRAY_TASK_ID + 1) * 25))
if [ $TASK_END -gt 100 ]; then TASK_END=100; fi

TASK_MORPHS=$((TASK_END - TASK_START))
CHUNK=$((TASK_MORPHS / N_PARALLEL))

echo "=== Fast threshold sweep: tau=${THRESHOLD}, seed=${SEED}, morphs ${TASK_START}-${TASK_END} ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

PIDS=()
for p in $(seq 0 $((N_PARALLEL - 1))); do
    P_START=$((TASK_START + p * CHUNK))
    if [ "$p" -eq $((N_PARALLEL - 1)) ]; then
        P_END=$TASK_END
    else
        P_END=$((TASK_START + (p + 1) * CHUNK))
    fi

    echo "  Launching process ${p}: morphs [${P_START}, ${P_END})"

    python scripts/train_mjx.py \
        --threshold "$THRESHOLD" \
        --seed "$SEED" \
        --timesteps 5000000 \
        --start_idx "$P_START" \
        --end_idx "$P_END" \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --out_dir "$OUT_DIR" \
        > "${OUT_DIR}/logs/proc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_p${p}.log" 2>&1 &
    PIDS+=($!)

    # Stagger launches by 10s to avoid simultaneous JAX compilation OOM
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
