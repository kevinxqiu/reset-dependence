#!/bin/bash
#SBATCH --job-name=rfc-bfree-smoke
#SBATCH --partition=common
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/boundary_free_smoke_%A.log
#SBATCH --mem=32G

# IMPORTANT: Before sbatch, ensure the SBATCH --output directory exists:
#   mkdir -p ~/reset-free-codesign/logs
# SLURM opens the output file before the script body runs, so the mkdir
# inside this script is too late for --output.

# Boundary-free smoke test: 1 body x 1 seed x 500k steps with
# --max-steps 500001 (truncation predicate never fires).
#
# Success criteria:
# - No NaN in any reported PSR or training loss.
# - implementation_reset_rate < 0.05 (rare too_high safety firings only).
# - policy_entropy_trace[-1] > 0 (not a degenerate zero-entropy policy).
# - value_loss_trace[-1] finite (no critic divergence).
# - per_env_step_count_final: all 128 entries within +/-1 of each other
#   (equal-progress envs, modulo final-update rounding).
# - Total wall-clock < 10 minutes.
#
# If any criterion fails: do NOT launch the 90-run panel; debug first.
#
# Submit:
#   sbatch slurm/boundary_free_smoke.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then source "${RFC_VENV}/bin/activate"; fi

# First body from the boundary-free panel (floor family, quintile 0).
SMOKE_MORPH=$(python -c "
import json
with open('records/boundary_free_bodies.json') as f:
    d = json.load(f)
print(d['panel'][0]['body'])
")
SEED=42
XML_DIR="${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"

OUT_DIR="results/boundary_free_smoke"
mkdir -p "${OUT_DIR}/logs"

SMOKE_LIST="${OUT_DIR}/logs/smoke_morph_list.txt"
mkdir -p "$(dirname "$SMOKE_LIST")"
echo "$SMOKE_MORPH" > "$SMOKE_LIST"

echo "=== Boundary-free smoke: body=$SMOKE_MORPH, seed=$SEED, 500k steps ==="
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

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
    --max-steps 500001 \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "${OUT_DIR}/logs/smoke.log"

echo ""
echo "=== Smoke test complete. Inspecting diagnostics ==="
RJ="${OUT_DIR}/results.json"
if [ -f "$RJ" ]; then
    python -c "
import json
import math
import sys
with open('$RJ') as f:
    d = json.load(f)
ok = True
for morph, rec in d.items():
    std = rec.get('eval_tau020', {})
    psr = std.get('per_step_reward')
    reset_rate = rec.get('implementation_reset_rate')
    ent_trace = rec.get('policy_entropy_trace') or []
    vloss_trace = rec.get('value_loss_trace') or []
    ploss_trace = rec.get('policy_loss_trace') or []
    step_counts = rec.get('per_env_step_count_final') or []
    print(f'Morph: {morph}')
    print(f'  PSR={psr}')
    print(f'  distance={std.get(\"distance\")}')
    print(f'  fall_rate={std.get(\"fall_rate\")}')
    print(f'  impl_reset_rate={reset_rate}')
    print(f'  entropy_trace={ent_trace}')
    print(f'  value_loss_trace={vloss_trace}')
    print(f'  policy_loss_trace={ploss_trace}')
    if step_counts:
        print(f'  per_env_step_count: min={min(step_counts)}, max={max(step_counts)}')
    # Finite checks
    if psr is None or not math.isfinite(psr):
        print(f'  FAIL: PSR not finite ({psr})')
        ok = False
    if reset_rate is None or not math.isfinite(reset_rate) or reset_rate > 0.05:
        print(f'  FAIL: impl_reset_rate={reset_rate} (must be finite and <= 0.05)')
        ok = False
    if not ent_trace or not math.isfinite(ent_trace[-1]) or ent_trace[-1] <= 0:
        print(f'  FAIL: policy entropy non-positive or non-finite')
        ok = False
    if not vloss_trace or not math.isfinite(vloss_trace[-1]):
        print(f'  FAIL: value loss non-finite')
        ok = False
    if not ploss_trace or not math.isfinite(ploss_trace[-1]):
        print(f'  FAIL: policy loss non-finite')
        ok = False
    if step_counts and (max(step_counts) - min(step_counts)) > 2:
        print(f'  FAIL: per-env step counts diverge by more than 1')
        ok = False
sys.exit(0 if ok else 1)
"
else
    echo "FAIL: results.json not produced"
    exit 1
fi
