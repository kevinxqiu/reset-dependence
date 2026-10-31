#!/bin/bash
#SBATCH --job-name=rfc-trender
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/temporal_render_%A.log
#SBATCH --mem=16G

# Render 9 rollouts from the temporal-gait retrain checkpoints:
# 3 bodies x 3 protocols at 5M steps. Each rollout yields MP4, mid-frame PNG,
# and traces.npz (qpos + action sequences).

set -euo pipefail
export PYTHONUNBUFFERED=1
# osmesa (software rendering) is slower than egl but does not require a
# working GPU EGL context, which several common-partition nodes lack at
# runtime despite having --gres=gpu:1.
export MUJOCO_GL=osmesa
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
        OUT="$OUT_DIR/${BODY}_${LABEL}.mp4"
        if [ ! -f "$CKPT" ]; then
            echo "SKIP: checkpoint missing $CKPT"
            continue
        fi
        if [ -f "${OUT%.mp4}.traces.npz" ]; then
            echo "SKIP: already rendered $OUT"
            continue
        fi
        echo "=== Rendering $BODY under $LABEL (tau=$TAU) ==="
        # Use the policy's TRAINING threshold so the rollout reflects the
        # regime the policy was optimized for. A uniform test threshold (e.g.,
        # 0.20) would impose fall terminations on reset-free (tau=0) policies
        # that never saw such a rule during training.
        python scripts/render_rollout.py \
            --space unimal \
            --morph "$BODY" \
            --checkpoint "$CKPT" \
            --threshold "$TAU" \
            --seed "$SEED" \
            --output "$OUT"
    done
done

echo "All rollouts rendered. Files in $OUT_DIR/"
