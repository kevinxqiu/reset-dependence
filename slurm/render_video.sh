#!/bin/bash
#SBATCH --job-name=rfc-render
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=figures/videos/render_%A.log
#SBATCH --mem=16G

# Render deterministic policy rollouts for the floor-5506-10-6 asymptotic
# checkpoint at tau=0.00 and tau=0.20, seed 42. Produces two MP4s + PNG stills
# under figures/videos/.

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
    source "${RFC_VENV}/bin/activate"
fi

MORPH="floor-5506-10-6-01-15-48-35"
OUT_DIR="figures/videos"
mkdir -p "$OUT_DIR"

for TAU in 0.00 0.20; do
    TAUP=$(echo "$TAU" | tr . p)
    CKPT="results/asymptotic_50m/jax/${MORPH}/tau${TAUP}_seed42/checkpoints/${MORPH}_step1001472.npz"
    OUT_MP4="${OUT_DIR}/floor_5506_10_6_tau${TAUP}.mp4"

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: checkpoint missing at $CKPT"
        exit 1
    fi

    echo "=== Rendering tau=$TAU (tracking camera) from $CKPT ==="
    python scripts/render_rollout.py \
        --space unimal \
        --morph "$MORPH" \
        --checkpoint "$CKPT" \
        --threshold "$TAU" \
        --seed 42 \
        --output "$OUT_MP4"
    echo ""

    OUT_FIXED="${OUT_DIR}/floor_5506_10_6_tau${TAUP}_fixed.mp4"
    echo "=== Rendering tau=$TAU (fixed camera) ==="
    python scripts/render_rollout.py \
        --space unimal \
        --morph "$MORPH" \
        --checkpoint "$CKPT" \
        --threshold "$TAU" \
        --seed 42 \
        --fixed-camera \
        --output "$OUT_FIXED"
    echo ""
done

echo "Done. Videos:"
ls -la "$OUT_DIR"/*.mp4
