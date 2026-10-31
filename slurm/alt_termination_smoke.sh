#!/bin/bash
#SBATCH --job-name=rfc-altt-smoke
#SBATCH --partition=common
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=results/alt_termination_smoke/logs/smoke_%A.log
#SBATCH --mem=32G

# Alt-termination smoke test: 1 body x 3 modes x 1 seed x 500k steps.
# Must pass before submitting the full 270-run panel.
#
# Success criteria:
# - No NaN in any reported PSR or training loss
# - Reasonable distances traveled (> 0 for locomotion reward)
# - N=50 contact-duration PSR within +-30% of corpus tau=0.20 PSR for the
#   same body (paper says if this fails, re-examine N-step scaling)
# - Training log shows steady loss progression, no divergence
#
# Submit:
#   sbatch slurm/alt_termination_smoke.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

# First body from the alt-termination panel (floor family, quintile 0).
MORPH=$(head -n 1 records/alt_termination_bodies.txt)
SEED=42
XML_DIR="${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"

OUT_DIR="results/alt_termination_smoke"
mkdir -p "${OUT_DIR}/logs"

# Single-line morph list file for this smoke
SMOKE_LIST="${OUT_DIR}/logs/smoke_morph_list.txt"
echo "$MORPH" > "$SMOKE_LIST"

echo "=== Alt-termination smoke: body=$MORPH, seed=$SEED, 3 modes x 500k steps ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

# Run sequentially to keep smoke clean (not testing parallelism here)
for N in 1 50 100000; do
    echo "--- Mode N=$N ---"
    SUB_OUT="${OUT_DIR}/N${N}"
    mkdir -p "${SUB_OUT}/logs"

    python scripts/train_mjx.py \
        --threshold 0.0 \
        --seed "$SEED" \
        --timesteps 500000 \
        --xml_dir "$XML_DIR" \
        --morphology-list "$SMOKE_LIST" \
        --num_envs 128 \
        --unroll_length 16 \
        --n_epochs 10 \
        --minibatch_size 256 \
        --out_dir "$SUB_OUT" \
        --termination-mode contact-duration \
        --contact-termination-steps "$N" \
        2>&1 | tee "${SUB_OUT}/logs/smoke.log"
    echo ""
done

echo "Smoke test complete. Inspect:"
for N in 1 50 100000; do
    RJ="${OUT_DIR}/N${N}/results.json"
    if [ -f "$RJ" ]; then
        echo "  $RJ:"
        python -c "
import json
with open('$RJ') as f:
    d = json.load(f)
for morph, rec in d.items():
    std = rec.get('eval_tau020', {})
    print(f'    {morph}: fitness={std.get(\"fitness\", \"nan\"):.1f} '
          f'psr={std.get(\"per_step_reward\", \"nan\"):.4f} '
          f'dist={std.get(\"distance\", \"nan\"):.2f} '
          f'fall={std.get(\"fall_rate\", \"nan\"):.2f} '
          f'N={rec.get(\"contact_termination_steps\")}')
"
    fi
done
