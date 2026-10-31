#!/bin/bash
# SAC full 3-tau 3-seed panel expansion launch script (archival).
# Spec: records/sac_full_panel_spec.md (FROZEN).
# Launch log: records/sac_full_panel_launch_log.md.
#
# Kept as the record of how the 120-run expansion was launched on 2026-04-24,
# AFTER the smoke test (SLURM 158579, tau=0.20 seed=42 array 0) had passed its
# smoke criteria as amended on 2026-04-24 (records/sac_full_panel_launch_log.md). scripts/verify_sac_smoke.py implements that
# gate and this script invokes it as a hard precondition. On the released,
# complete panel the gate refuses by design (the smoke results now hold all
# 24 bodies), so to train SAC cells, call slurm/train_sac_mjx.sh directly.
#
# Submits 14 array tasks (across 5 sbatch calls) covering 112 bodies
# (the remainder after the 8-body smoke). The training script runs
# single-process per GPU; pass cluster-specific partition/QOS flags to
# sbatch at submit time if your scheduler requires them.
#
# The pre-reg analysis gate is applied by scripts/analyze_sac_full_panel.py
# after all 216 cells (96 existing + 120 new) are valid.
#
# Usage:
#   ./slurm/sac_full_panel_expansion.sh           # dry run (no submission)
#   ./slurm/sac_full_panel_expansion.sh --submit  # actually submit
#
# On --submit, if any sbatch call fails mid-sequence, the script
# automatically scancels any jobs it already submitted, to avoid
# orphaned partial launches.

set -Eeuo pipefail  # -E propagates ERR trap into shell functions (submit_one)

# Pinned expected baseline SHA256s (QC 5 code-identity drift check).
# These hashes are pinned to the released copies of the three files. Their
# edits since the launch are path, logging and error-handling cleanup;
# training semantics are unchanged. Any drift aborts the launch.
EXPECTED_SHA256_train_py="1c8451141e659c88dd5396907c9377f7a06d81ab6511afa36ad60d12658c3d55"
EXPECTED_SHA256_train_sh="fdd1aab04d4725f90324e60f6a3941b40c79446529b38e38c258db16d8022b80"
EXPECTED_SHA256_morph_txt="aa66b82229e519b65ba9618643de77bf1219b99daec1f23835a30ef9172de158"

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=1
fi

cd "${RFC_REPO_DIR:-$PWD}"
if [ -n "${RFC_VENV:-}" ]; then
  source "${RFC_VENV}/bin/activate"
fi

echo "=== SAC full panel expansion: $(date -u +%FT%TZ) ==="
HEAD_NOW=$(git rev-parse HEAD)
echo "Current HEAD: ${HEAD_NOW}"
echo "Branch: $(git branch --show-current)"
echo ""

# Pre-flight: smoke gate must pass before we submit anything.
echo "--- Pre-flight: verifying smoke pre-reg criteria ---"
if ! python3 scripts/verify_sac_smoke.py; then
  echo ""
  echo "ABORT: scripts/verify_sac_smoke.py failed (exit != 0)."
  echo "       Smoke did not pass pre-reg criteria. Do not launch."
  exit 1
fi
echo ""

# Pre-flight: QC 5 code-identity drift check. Compare current SHA256
# of the three training-pipeline files against pinned baselines. Any
# drift fails closed.
echo "--- Pre-flight: code-identity drift check (QC 5) ---"
check_hash() {
  local file="$1"
  local expected="$2"
  if [[ ! -f "$file" ]]; then
    echo "ABORT: required file missing: $file"
    exit 1
  fi
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "ABORT: SHA256 drift detected on $file"
    echo "  expected: $expected"
    echo "  actual:   $actual"
    echo "  Training code changed between pilot (2026-04-09) and launch."
    echo "  Per QC 5, expansion launch is invalid until drift is resolved."
    exit 1
  fi
  echo "  OK  $file  $actual"
}
check_hash scripts/train_sac_mjx.py "$EXPECTED_SHA256_train_py"
check_hash slurm/train_sac_mjx.sh "$EXPECTED_SHA256_train_sh"
check_hash scripts/mechanism_morphologies.txt "$EXPECTED_SHA256_morph_txt"
echo ""

if [[ $SUBMIT -eq 0 ]]; then
  echo "--- DRY RUN (no sbatch calls) ---"
  echo "Would submit 5 sbatch calls covering 14 array tasks:"
  echo "  sbatch --array=1-2 --export=ALL,THRESHOLD=0.20,SEED=42 slurm/train_sac_mjx.sh  # 2 tasks, 16 bodies"
  echo "  sbatch --array=0-2 --export=ALL,THRESHOLD=0.20,SEED=43 slurm/train_sac_mjx.sh  # 3 tasks, 24 bodies"
  echo "  sbatch --array=0-2 --export=ALL,THRESHOLD=0.20,SEED=44 slurm/train_sac_mjx.sh  # 3 tasks, 24 bodies"
  echo "  sbatch --array=0-2 --export=ALL,THRESHOLD=0.00,SEED=44 slurm/train_sac_mjx.sh  # 3 tasks, 24 bodies"
  echo "  sbatch --array=0-2 --export=ALL,THRESHOLD=0.50,SEED=44 slurm/train_sac_mjx.sh  # 3 tasks, 24 bodies"
  echo ""
  echo "Total: 14 array tasks, 112 bodies, 112 new cells. (8 smoke bodies already done.)"
  echo ""
  echo "Re-run with --submit to actually submit."
  exit 0
fi

echo "--- Submitting expansion jobs ---"

# Rollback: if we fail mid-sequence, scancel anything already submitted
# to avoid orphans. Array is populated as submits succeed.
EXPANSION_IDS=()
rollback_submitted() {
  local code=$?
  if [[ ${#EXPANSION_IDS[@]} -gt 0 && $code -ne 0 ]]; then
    echo ""
    echo "ROLLBACK: a submission failed after ${#EXPANSION_IDS[@]} job(s) were"
    echo "         already queued. scancel-ing to avoid orphan launches."
    for jid in "${EXPANSION_IDS[@]}"; do
      scancel "$jid" 2>&1 || echo "  warn: scancel $jid failed (may already be terminal)"
    done
  fi
  exit $code
}
trap rollback_submitted ERR

submit_one() {
  local jid
  jid=$(sbatch --parsable "$@")
  EXPANSION_IDS+=("$jid")
  echo "  submitted: $jid  ($*)"
}

submit_one --array=1-2 --export=ALL,THRESHOLD=0.20,SEED=42 slurm/train_sac_mjx.sh
submit_one --array=0-2 --export=ALL,THRESHOLD=0.20,SEED=43 slurm/train_sac_mjx.sh
submit_one --array=0-2 --export=ALL,THRESHOLD=0.20,SEED=44 slurm/train_sac_mjx.sh
submit_one --array=0-2 --export=ALL,THRESHOLD=0.00,SEED=44 slurm/train_sac_mjx.sh
submit_one --array=0-2 --export=ALL,THRESHOLD=0.50,SEED=44 slurm/train_sac_mjx.sh

# All 5 submits succeeded. Disarm the rollback trap.
trap - ERR

echo ""
echo "Submitted 5 array jobs, SLURM IDs: ${EXPANSION_IDS[*]}"
echo ""
echo "Current queue (best-effort print):"
# Non-pipefail-sensitive: use a subshell so a SIGPIPE from `head` does
# not propagate to the outer script and look like a submission failure.
(set +o pipefail; squeue -u "$USER" --format="%.10i %P %.20j %T %.10M" 2>&1 | head -25) || true

echo ""
echo "--- Launch log append template (append to records/sac_full_panel_launch_log.md) ---"
cat <<EOF
## $(date -u +%FT%TZ): Expansion launch
- HEAD: ${HEAD_NOW}
- SLURM IDs: ${EXPANSION_IDS[*]}
- Smoke gate passed (scripts/verify_sac_smoke.py exit 0).
- Code-identity (QC 5) drift check passed:
  - scripts/train_sac_mjx.py: ${EXPECTED_SHA256_train_py}
  - slurm/train_sac_mjx.sh: ${EXPECTED_SHA256_train_sh}
  - scripts/mechanism_morphologies.txt: ${EXPECTED_SHA256_morph_txt}
EOF
