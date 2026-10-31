#!/bin/bash
#SBATCH --job-name=rfc-menagerie
#SBATCH --partition=common
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=results/menagerie_panel/logs/menagerie_%A_%a.log
#SBATCH --mem=32G

# Menagerie cross-embodiment panel, 60 runs total.
# 4 robots x 3 taus x 5 seeds = 60 (robot, tau, seed) triples, split across
# 15 array tasks with 4-way packing per GPU. Total effective GPU-hours is
# ~20 at ~1.2 hr per 5M-step run under 4-pack.
#
# Submit once (no arguments):
#   sbatch slurm/menagerie_panel.sh
#
# Outputs:
#   results/menagerie_panel/{robot}_tau{0pXX}_seed{N}/results.json
#   results/menagerie_panel/logs/menagerie_<job>_<task>.log (SLURM)
#   results/menagerie_panel/logs/proc_<job>_<task>_p<p>.log (per-process)

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

N_PARALLEL=4

OUT_DIR="results/menagerie_panel"
mkdir -p "${OUT_DIR}/logs"

ROBOTS=(unitree_go2 anybotics_anymal_c toddlerbot_2xc unitree_g1)
TAUS=(0.00 0.20 0.50)
SEEDS=(42 43 44 45 46)

# Enumerate all 60 (robot, tau, seed) combos in a frozen order.
COMBOS=()
for robot in "${ROBOTS[@]}"; do
    for tau in "${TAUS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COMBOS+=("${robot}:${tau}:${seed}")
        done
    done
done

TOTAL=${#COMBOS[@]}
if [ $TOTAL -ne 60 ]; then
    echo "ERROR: expected 60 combos, got $TOTAL" >&2
    exit 1
fi

TASK_START=$((SLURM_ARRAY_TASK_ID * N_PARALLEL))
TASK_END=$((TASK_START + N_PARALLEL))
if [ $TASK_END -gt $TOTAL ]; then TASK_END=$TOTAL; fi

echo "=== Menagerie panel: task ${SLURM_ARRAY_TASK_ID}, combos ${TASK_START}-$((TASK_END-1)) ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

PIDS=()
for i in $(seq $TASK_START $((TASK_END - 1))); do
    COMBO="${COMBOS[$i]}"
    ROBOT="${COMBO%%:*}"
    REST="${COMBO#*:}"
    TAU="${REST%%:*}"
    SEED="${REST#*:}"

    TAU_LABEL="${TAU/./p}"
    RUN_OUT="${OUT_DIR}/${ROBOT}_tau${TAU_LABEL}_seed${SEED}"
    mkdir -p "$RUN_OUT"

    echo "  [${i}] launching ${ROBOT} tau=${TAU} seed=${SEED}"

    python scripts/menagerie_train_mjx.py \
        --robot "$ROBOT" \
        --threshold "$TAU" \
        --seed "$SEED" \
        --timesteps 5000000 \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --out_dir "$RUN_OUT" \
        > "${OUT_DIR}/logs/proc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_p${i}.log" 2>&1 &
    PIDS+=($!)

    # Stagger launches by 10s to avoid simultaneous JAX compilation OOM
    if [ "$i" -lt $((TASK_END - 1)) ]; then
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
