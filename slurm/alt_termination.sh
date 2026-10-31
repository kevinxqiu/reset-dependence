#!/bin/bash
#SBATCH --job-name=rfc-altt
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=results/alt_termination/logs/altt_%A.log
#SBATCH --mem=32G

# Alt-termination panel: run the protocol sweep on 30 UNIMAL bodies under
# contact-duration termination (analog of height-based tau). Pre-registered
# in records/alt_termination_spec.md; panel frozen in
# records/alt_termination_bodies.json.
#
# Modes (N = contact-duration threshold in simulator steps):
#   N=1       (strict analogue; terminate on first torso-floor contact)
#   N=50      (episodic analogue; ~1s at dt=0.02)
#   N=100000  (reset-free analogue; effectively never terminate)
#
# Submit once per (mode, seed) pair (9 jobs total, 3 N x 3 seeds):
#   for N in 1 50 100000; do
#     for SEED in 42 43 44; do
#       LABEL="N${N}_seed${SEED}"
#       sbatch --export=ALL,CONTACT_N=$N,SEED=$SEED,LABEL=$LABEL \
#         slurm/alt_termination.sh
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

MORPH_LIST="records/alt_termination_bodies.txt"
PANEL_JSON="records/alt_termination_bodies.json"
XML_DIR="${RFC_UNIMAL_XML_DIR:-assets/unimals_100/xml}"

# Regenerate the text list from JSON if missing (deterministic;
# alt_termination_bodies.json is the source of truth).
if [ ! -f "$MORPH_LIST" ]; then
    echo "Generating $MORPH_LIST from $PANEL_JSON"
    python -c "
import json
with open('$PANEL_JSON') as f:
    d = json.load(f)
with open('$MORPH_LIST','w') as f:
    for entry in d['panel']:
        f.write(entry['body'] + '\n')
"
fi

TOTAL_MORPHS=$(wc -l < "$MORPH_LIST")
if [ "$TOTAL_MORPHS" -ne 30 ]; then
    echo "ERROR: expected 30 bodies in panel, got $TOTAL_MORPHS" >&2
    exit 2
fi

OUT_DIR="results/alt_termination/${LABEL}"
mkdir -p "${OUT_DIR}/logs"

echo "=== Alt-termination panel: N=${CONTACT_N}, seed=${SEED}, N_bodies=${TOTAL_MORPHS} ==="
echo "Host: $(hostname), N_PARALLEL=${N_PARALLEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""

# Split panel into N_PARALLEL chunks; process p gets morphs
# [p * ceil(N/P), (p+1) * ceil(N/P)).
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

    # --threshold is required by argparse but ignored in contact-duration
    # mode. Pass a dummy value.
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
        --out_dir "$OUT_DIR" \
        --termination-mode contact-duration \
        --contact-termination-steps "$CONTACT_N" \
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
