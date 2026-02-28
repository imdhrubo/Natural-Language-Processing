#!/bin/bash
#SBATCH --job-name=hw02_retrain
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

# 1) Activate environment
source ~/venvs/NLP/bin/activate

# 2) Go to your real project directory (no mktemp)
PROJ=/users/kmahbub/itcs6101/Natural-Language-Processing/hw02/numpy
cd "$PROJ"

mkdir -p logs

TRACE="traces.${SLURM_JOB_ID}.txt"
rm -f "$TRACE"

log() { echo "$@" | tee -a "$TRACE"; }

log "=== ENV ==="
log "PWD: $(pwd)"
log "HOST: $(hostname)"
log "DATE: $(date)"
log "PYTHON: $(which python3)"
log "========================================"

run_train () {
  local CONTEXT_SIZE="$1"
  local OUTBASE="vec.ctx${CONTEXT_SIZE}.txt"
  local VOCAB="vocab.ctx${CONTEXT_SIZE}.txt"

  log ""
  log "========================================"
  log "TRAIN START: context=${CONTEXT_SIZE}"
  log "$(date)"

  # Train + save vocab + save embeddings
  python3 lm_embed_main.py \
    -train ../data/wiki-1B.txt \
    -savevocab "$VOCAB" \
    -output "$OUTBASE" \
    -size 200 \
    -context "${CONTEXT_SIZE}" \
    -subsample 1e-4 \
    -negative 5 \
    -iter 3 \
    |& tee -a "$TRACE"

  log "$(date)"
  log "TRAIN END: context=${CONTEXT_SIZE}"

  # Evaluate using the matching vocab file and target embeddings (.1)
  log "TEST START: context=${CONTEXT_SIZE}"
  python3 test_embeddings.py "$VOCAB" "${OUTBASE}.1" << 'EOF' |& tee -a "$TRACE"
book
trip
paris
stop
write
language
beautiful
bad
quickly
amazing

paris france tokyo
balloon air bucket
novel writer music
white snow red
liquid water solid
water liquid ice
EOF
  log "TEST END: context=${CONTEXT_SIZE}"
}

# Optional: keep old outputs as backup
if [ -f traces.txt ]; then
  mv -f traces.txt "traces.backup.$(date +%Y%m%d_%H%M%S).txt"
fi

run_train 5
run_train 15

# Create a "default" vocab.txt that matches ctx15 (or choose ctx5)
cp -f vocab.ctx15.txt vocab.txt

# Create stable final trace name
cp -f "$TRACE" traces.txt

log ""
log "ALL DONE"
log "$(date)"

echo "Done. Outputs in: $PROJ"
echo "  - vec.ctx5.txt.1 / vec.ctx5.txt.2"
echo "  - vec.ctx15.txt.1 / vec.ctx15.txt.2"
echo "  - vocab.ctx5.txt / vocab.ctx15.txt"
echo "  - traces.txt (and logs/...)"