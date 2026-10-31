#!/bin/bash
#SBATCH --job-name=rfc-traces
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/traces_%A.log
#SBATCH --mem=16G

# Produce 9 traces.npz files (qpos + action) for the 3x3 temporal-gait
# figure. No rendering; scripts/figures/render_temporal_gait.py builds the figure
# from traces alone.

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

BODIES=(
    "floor-5506-10-6-01-15-48-35"
    "mvt-5506-13-10-17-12-25-45"
    "floor-1409-0-7-01-15-15-26"
)
PROTOCOLS=(
    "0.00:tau0p00"
    "0.20:tau0p20"
    "0.50:tau0p50"
)
SEED=42
STEP=5000000
OUT_DIR="figures/gait"
mkdir -p "$OUT_DIR"

for BODY in "${BODIES[@]}"; do
    for PROTO in "${PROTOCOLS[@]}"; do
        TAU="${PROTO%%:*}"
        LABEL="${PROTO##*:}"
        CKPT="results/temporal_gait/${LABEL}_5M_seed${SEED}/checkpoints/${BODY}_step${STEP}.npz"
        OUT="$OUT_DIR/${BODY}_${LABEL}.traces.npz"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP: checkpoint missing $CKPT"
            continue
        fi
        if [ -f "$OUT" ]; then
            echo "SKIP: already produced $OUT"
            continue
        fi
        echo "=== Traces for $BODY under $LABEL (tau=$TAU) ==="
        python scripts/traces_only_rollout.py \
            --morph "$BODY" \
            --checkpoint "$CKPT" \
            --threshold "$TAU" \
            --seed "$SEED" \
            --output "$OUT"
    done
done

echo "All 9 traces produced."
