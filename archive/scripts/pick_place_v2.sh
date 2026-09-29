#!/usr/bin/env bash
# B1a pick-and-place protocol v2 (stage-resynchronized teacher): seeds 0-2 then 3-5, each with
# the connectome, two independent shuffles and the GRU. Starts after run 003 finishes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until grep -q '"status": "\(complete\|failed\)"' runs/whole-brain-pick-place-003/results.json 2>/dev/null; do
  sleep 120
done
for part in a b; do
  output=runs/whole-brain-pick-place-v2$part
  [ -e "$output" ] && continue
  echo "[$(date '+%F %T')] start $output"
  uv run --no-sync flyarm whole-brain run --config configs/whole-brain-pick-place-v2$part.json \
    --output "$output" > "$output.log" 2>&1
  echo "[$(date '+%F %T')] exit $? $output"
done
