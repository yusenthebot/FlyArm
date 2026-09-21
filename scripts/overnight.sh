#!/usr/bin/env bash
# Overnight pipeline: wait for training runs, record labelled rollouts, regenerate the report.
# Each step is logged and independent, so one failure never blocks the remaining deliverables.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
OUT=runs/overnight
VIDEOS=$OUT/videos
mkdir -p "$VIDEOS"
FLYARM="uv run --no-sync flyarm"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/pipeline.log"; }
finished() { grep -q '"status": "\(complete\|failed\)"' "$1/results.json" 2>/dev/null; }
wait_for() { until finished "$1"; do sleep 60; done; log "finished: $1"; }
step() { log "start: $*"; if "$@" >> "$OUT/pipeline.log" 2>&1; then log "ok: $1 $2"; else log "FAILED: $*"; fi; }
report() { step $FLYARM report --output "$OUT/REPORT.md"; }

log "pipeline started"
report

# 1. B1a pick-and-place: labelled rollouts of every checkpoint (3 held-out episodes each).
wait_for runs/whole-brain-pick-place-002
for kind in connectome shuffled gru; do
  step $FLYARM whole-brain record --run runs/whole-brain-pick-place-001 --kind $kind --seed 0 \
    --episodes 3 --output "$VIDEOS/pick-place-$kind-seed0.mp4"
  for seed in 1 2; do
    step $FLYARM whole-brain record --run runs/whole-brain-pick-place-002 --kind $kind \
      --seed $seed --episodes 3 --output "$VIDEOS/pick-place-$kind-seed$seed.mp4"
  done
done
step $FLYARM whole-brain record --run runs/whole-brain-reach-001 --kind connectome --seed 0 \
  --episodes 4 --output "$VIDEOS/reach-connectome-seed0.mp4"
report

# 2. B2 kitchen-complete: per-controller clips and same-episode 2x2 comparisons.
wait_for runs/flyleg-kitchen-complete-001
step $FLYARM flyleg record --run runs/flyleg-kitchen-complete-001 --seed 0 --episodes 0 1 2 \
  --output "$VIDEOS"
for seed in 1 2; do
  step $FLYARM flyleg record --run runs/flyleg-kitchen-complete-001 --seed $seed --episodes 0 \
    --output "$VIDEOS"
done
report

# 3. B2 kitchen-mixed (compositional generalization split), then its rollouts.
if [ ! -e runs/flyleg-kitchen-mixed-001 ]; then
  step $FLYARM flyleg run --config configs/flyleg-kitchen-mixed.json \
    --output runs/flyleg-kitchen-mixed-001
fi
step $FLYARM flyleg record --run runs/flyleg-kitchen-mixed-001 --seed 0 --episodes 0 1 \
  --output "$VIDEOS"
report
log "pipeline finished"
