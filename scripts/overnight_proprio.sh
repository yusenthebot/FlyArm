#!/usr/bin/env bash
# Proprioception-only fly variant: train, then record its rollouts and refresh the report.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
OUT=runs/overnight
mkdir -p "$OUT/videos"
FLYARM="uv run --no-sync flyarm"
log() { echo "[$(date '+%F %T')] proprio: $*" | tee -a "$OUT/pipeline.log"; }
step() { log "start: $*"; if "$@" >> "$OUT/pipeline.log" 2>&1; then log "ok"; else log "FAILED: $*"; fi; }
RUN=runs/flyleg-kitchen-complete-proprio-001
step $FLYARM flyleg run --config configs/flyleg-kitchen-complete-proprio.json --output $RUN
step $FLYARM flyleg record --run $RUN --seed 0 --episodes 0 1 2 --output "$OUT/videos/proprio"
step $FLYARM report --output "$OUT/REPORT.md"
log "finished"
