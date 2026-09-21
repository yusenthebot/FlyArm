#!/usr/bin/env bash
# Milestone pipeline: wait for the chunked kitchen run and the pick-place replication runs,
# record labelled rollouts of every new checkpoint, then regenerate the report.
# Each step is logged and independent, so one failure never blocks the remaining deliverables.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
OUT=runs/milestone
VIDEOS=$OUT/videos
mkdir -p "$VIDEOS/kitchen-chunk"
FLYARM="uv run --no-sync flyarm"
KITCHEN=runs/flyleg-kitchen-complete-chunk-001

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/pipeline.log"; }
finished() { grep -q '"status": "\(complete\|failed\)"' "$1/results.json" 2>/dev/null; }
wait_for() { until finished "$1"; do sleep 60; done; log "finished: $1"; }
step() { log "start: $*"; if "$@" >> "$OUT/pipeline.log" 2>&1; then log "ok: $1 $2"; else log "FAILED: $*"; fi; }
report() { step $FLYARM report --output "$OUT/REPORT.md"; }

log "pipeline started"
wait_for "$KITCHEN"
for seed in 0 1 2; do
  step $FLYARM flyleg record --run "$KITCHEN" --seed $seed --episodes 0 --output "$VIDEOS/kitchen-chunk"
done
report

wait_for runs/whole-brain-pick-place-003
for seed in 3 4 5; do
  step $FLYARM whole-brain record --run runs/whole-brain-pick-place-003 --kind connectome \
    --seed $seed --episodes 3 --output "$VIDEOS/pick-place-connectome-seed$seed.mp4"
  for replicate in 0 1; do
    step $FLYARM whole-brain record --run runs/whole-brain-pick-place-003 --kind shuffled \
      --seed $seed --replicate $replicate --episodes 3 \
      --output "$VIDEOS/pick-place-shuffled-seed$seed-r$replicate.mp4"
  done
done
report
wait_for runs/whole-brain-pick-place-004
report
log "pipeline finished"
